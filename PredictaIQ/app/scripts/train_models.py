#!/usr/bin/env python
"""Similarity + ML Engine'i eğitir, walk-forward backtest çalıştırır ve
öğrenilen ensemble ağırlıklarını aktif ModelVersion olarak kaydeder.

Kullanım:
    python -m app.scripts.train_models --folds 5 --min-train 60
"""
import argparse
import logging

from app.config import settings
from app.database import SessionLocal
from app.dependencies import backtest_service, prediction_service
from app.models import Match, ModelVersion, Season
from app.services import model_persistence
from app.services.team_rating_service import TeamRatingService
from sqlalchemy.orm import joinedload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _finished_matches_eager(db):
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


def main():
    parser = argparse.ArgumentParser(description="PredictaIQ model eğitimi / backtest")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=60)
    parser.add_argument("--no-persist", action="store_true", help="Öğrenilen ağırlıkları DB'ye kaydetme")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        logger.info("Rating'ler yeniden hesaplanıyor...")
        TeamRatingService().recompute_all_ratings(db)
        db.commit()

        matches = _finished_matches_eager(db)
        logger.info("%d bitmiş maç ile walk-forward backtest başlıyor...", len(matches))

        result = backtest_service.walk_forward(db, matches, n_folds=args.folds, min_train=args.min_train)
        logger.info("Backtest sonucu: %s", {k: v for k, v in result.items() if k != "folds"})

        if result.get("status") != "completed":
            logger.warning("Backtest tamamlanamadı, model eğitimi durduruluyor.")
            return

        prediction_service.similarity.train(matches)
        X, y = prediction_service.build_training_matrix(matches)
        prediction_service.ml.train(X, y)

        # Eğitilmiş modelleri diske yaz -- GitHub Actions bu script'i
        # doğrudan CLI olarak çalıştırırsa (HTTP endpoint yerine) da
        # sonuç aynı model_path'e yazılır ve API process'i bir sonraki
        # restart'ta veya `POST /admin/models/load` ile bunu okuyabilir.
        model_persistence.save_all(prediction_service, settings.model_path)

        weights = result.get("learned_ensemble_weights")
        if weights:
            prediction_service.set_learned_weights(weights)
            logger.info("Öğrenilen ensemble ağırlıkları: %s", weights)

            if not args.no_persist:
                db.query(ModelVersion).filter(
                    ModelVersion.name == "ensemble", ModelVersion.is_active == True
                ).update({"is_active": False})
                db.add(ModelVersion(
                    name="ensemble",
                    version=f"cli-{len(matches)}",
                    ensemble_weights=weights,
                    brier_score=result.get("overall_brier_score"),
                    log_loss=result.get("overall_log_loss"),
                    trained_on_matches=len(matches),
                    is_active=True,
                ))
                db.commit()
                logger.info("ModelVersion kaydedildi.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
