import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from app.database import get_session
from app.dependencies import backtest_service, prediction_service
from app.config import settings
from app.models import Match, ModelVersion, Season
from app.schemas import ModelStatusResponse, SyncLiveResponse, TrainRequest, TrainResponse
from app.security import verify_admin_key
from app.services import model_persistence
from app.services.live_sync_service import LiveSyncService
from app.services.team_rating_service import TeamRatingService

# `dependencies=[Depends(verify_admin_key)]`: bu router altındaki HER
# endpoint (ratings/recompute, models/train, sync/live, ...) artık
# X-Admin-Api-Key header'ı gerektiriyor -- eskiden tamamen açıktı, herhangi
# biri /models/train'i tetikleyip CPU tüketebilir/veriyi bozabilirdi.
router = APIRouter(dependencies=[Depends(verify_admin_key)])
logger = logging.getLogger(__name__)
team_rating_service = TeamRatingService()
live_sync_service = LiveSyncService()


def _finished_matches_eager(db: Session) -> list:
    """Bkz. app.main._finished_matches_eager -- similarity/ML eğitiminde
    kullanılan Match nesneleri, eğitim isteğinin session'ı kapandıktan
    sonra da (sonraki HTTP isteklerinde) okunacağı için ilişkiler burada
    eager-load edilmelidir, aksi halde DetachedInstanceError oluşur."""
    return (
        db.query(Match)
        .options(
            joinedload(Match.home_team),
            joinedload(Match.away_team),
            joinedload(Match.season).joinedload(Season.competition),
        )
        .filter(Match.status == "finished", Match.home_score.isnot(None))
        .all()
    )


@router.post("/ratings/recompute")
async def recompute_ratings(db: Session = Depends(get_session)):
    """Tüm bitmiş maçları kronolojik olarak gezip Elo/form/hücum/savunma
    rating'lerini ve maç-öncesi xG tahminlerini yeniden hesaplar
    (plan madde 2-4). Yeni maç verisi eklendikten sonra çalıştırılmalı."""
    n = team_rating_service.recompute_all_ratings(db)
    db.commit()
    return {"status": "completed", "matches_processed": n}


@router.post("/models/train", response_model=TrainResponse)
async def train_models(request: TrainRequest, db: Session = Depends(get_session)) -> TrainResponse:
    """Similarity + ML Engine'i sızıntısız walk-forward backtest ile eğitir,
    Brier/Log Loss hesaplar, ensemble ağırlıklarını öğrenir ve (istenirse)
    aktif ModelVersion olarak kaydeder (plan madde 3, 15, 26)."""
    matches = _finished_matches_eager(db)

    result = backtest_service.walk_forward(db, matches, n_folds=request.n_folds, min_train=request.min_train)

    if result.get("status") != "completed":
        return TrainResponse(status=result.get("status", "failed"), detail=str(result))

    # Nihai similarity + ml modelini TÜM geçmiş veriyle eğit (canlı tahmin için)
    prediction_service.similarity.train(matches)
    X, y = prediction_service.build_training_matrix(matches)
    prediction_service.ml.train(X, y, min_matches=settings.min_matches_to_train_ml)

    # Eğitilmiş modelleri diske yaz (settings.model_path) -- production'da
    # process her yeniden başladığında main.py bunları YENİDEN EĞİTMEK
    # yerine buradan YÜKLEYECEK.
    model_persistence.save_all(prediction_service, settings.model_path)

    learned_weights = result.get("learned_ensemble_weights")
    if request.persist_weights and learned_weights:
        prediction_service.set_learned_weights(learned_weights)

        db.query(ModelVersion).filter(ModelVersion.name == "ensemble", ModelVersion.is_active == True).update(
            {"is_active": False}
        )
        version = ModelVersion(
            name="ensemble",
            version=f"walkforward-{len(matches)}",
            ensemble_weights=learned_weights,
            brier_score=result.get("overall_brier_score"),
            log_loss=result.get("overall_log_loss"),
            calibration_error=(result.get("calibration") or {}).get("calibration_error"),
            trained_on_matches=len(matches),
            is_active=True,
        )
        db.add(version)
        db.commit()

    return TrainResponse(
        status="completed",
        n_matches_evaluated=result.get("n_matches_evaluated"),
        overall_brier_score=result.get("overall_brier_score"),
        overall_log_loss=result.get("overall_log_loss"),
        learned_ensemble_weights=learned_weights,
        folds=result.get("folds"),
    )


@router.get("/models/status", response_model=ModelStatusResponse)
async def model_status() -> ModelStatusResponse:
    return ModelStatusResponse(
        similarity_trained=prediction_service.similarity.nn is not None,
        similarity_match_count=len(prediction_service.similarity.match_data or []),
        ml_trained=prediction_service.ml.is_trained,
        ml_backends=prediction_service.ml.available_backends,
        ml_training_size=prediction_service.ml.training_size,
        calibration_fitted=prediction_service.calibration.is_fitted,
        weights_are_learned=prediction_service.weights_are_learned,
        current_weights=prediction_service.model_weights,
    )


@router.post("/models/load")
async def load_models(db: Session = Depends(get_session)):
    """Diskteki (settings.model_path) kaydedilmiş ML/similarity/calibration
    modellerini + DB'deki aktif ensemble ağırlıklarını belleğe yükler.
    `main.py` bunu her process başlangıcında otomatik çağırır; bu endpoint
    manuel olarak (örn. modeller elle diske kopyalandıysa) tekrar
    tetiklemek için var."""
    result = model_persistence.load_all(prediction_service, settings.model_path, db)
    return {"status": "completed", **result}


@router.post("/sync/live", response_model=SyncLiveResponse)
async def sync_live(
    leagues: Optional[List[str]] = Query(default=None, description="Boş bırakılırsa SYNC_LEAGUES env değeri kullanılır"),
    db: Session = Depends(get_session),
) -> SyncLiveResponse:
    """Canlı/günün maçlarını ve oranlarını dış API'lerden çekip DB'ye yazar,
    rating'leri günceller ve scheduled/live maçların tahminlerini tazeler.

    GitHub Actions tarafından 15 dakikada bir çağrılması öngörülüyor --
    kullanıcı sayfayı açtığında değil, arka planda tetiklenir (bkz. mimari
    tartışması: canlı veri toplama isteğe bağlı DEĞİL, zamanlanmış)."""
    target_leagues = leagues or settings.sync_leagues_list
    if not target_leagues:
        raise HTTPException(400, "Senkronize edilecek lig yok (SYNC_LEAGUES boş)")
    result = await live_sync_service.sync_live(db, prediction_service, target_leagues)
    return SyncLiveResponse(**result)


@router.post("/models/load-active-weights")
async def load_active_weights(db: Session = Depends(get_session)):
    """En son aktif ModelVersion satırındaki öğrenilmiş ağırlıkları
    uygulamaya yükler (örn. yeniden başlatma sonrası)."""
    version = (
        db.query(ModelVersion)
        .filter(ModelVersion.name == "ensemble", ModelVersion.is_active == True)
        .order_by(ModelVersion.created_at.desc())
        .first()
    )
    if not version or not version.ensemble_weights:
        raise HTTPException(404, "Kayıtlı aktif model ağırlığı bulunamadı")
    prediction_service.set_learned_weights(version.ensemble_weights)
    return {"status": "loaded", "weights": prediction_service.model_weights}
