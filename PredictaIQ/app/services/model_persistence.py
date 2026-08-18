"""Eğitilmiş tüm model artefaktlarının (ML ensemble, similarity, calibration,
ensemble ağırlıkları) tek bir yerden kaydedilip yüklenmesi.

Bu modül, `main.py`'deki eski "her açılışta yeniden eğit" mantığının yerini
alıyor: production'da modeller sadece GitHub Actions/manuel bir
`POST /api/v1/admin/models/train` çağrısıyla eğitilir ve diske
(`settings.model_path`) yazılır; process her başladığında (deploy,
restart, yeni instance) sadece bu diskteki dosyalar YÜKLENİR, yeniden
eğitim yapılmaz.

NOT: `settings.model_path` kalıcı bir disk olmalı. Railway/Render/Fly gibi
platformlarda bu, bir "volume" olarak mount edilmeli -- aksi halde her
deploy'da model dosyaları sıfırlanır ve tekrar `models/train` çağırmak
gerekir (zararsız ama gereksiz bir gecikmeye yol açar).
"""
from __future__ import annotations

import logging
import os

from sqlalchemy.orm import Session

from app.models import ModelVersion
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)


def save_all(prediction_service: PredictionService, model_path: str) -> None:
    prediction_service.ml.save(os.path.join(model_path, "ml"))
    prediction_service.similarity.save(os.path.join(model_path, "similarity"))
    prediction_service.calibration.save(os.path.join(model_path, "calibration"))


def load_all(prediction_service: PredictionService, model_path: str, db: Session) -> dict:
    """Diskten (varsa) modelleri + DB'den (varsa) aktif ensemble ağırlıklarını
    yükler. Hiçbir şey bulunamazsa hata FIRLATMAZ -- uygulama boş modellerle
    açılır ve `/api/v1/admin/models/train` ile daha sonra eğitilebilir."""
    result = {
        "ml_loaded": False,
        "similarity_loaded": False,
        "calibration_loaded": False,
        "weights_loaded": False,
    }

    try:
        result["ml_loaded"] = prediction_service.ml.load(os.path.join(model_path, "ml"))
    except Exception as exc:  # pragma: no cover
        logger.warning("ML Engine yüklenirken hata (yoksayılıyor): %s", exc)

    try:
        result["similarity_loaded"] = prediction_service.similarity.load(
            os.path.join(model_path, "similarity"), db
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Similarity yüklenirken hata (yoksayılıyor): %s", exc)

    try:
        result["calibration_loaded"] = prediction_service.calibration.load(
            os.path.join(model_path, "calibration")
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Calibration yüklenirken hata (yoksayılıyor): %s", exc)

    try:
        version = (
            db.query(ModelVersion)
            .filter(ModelVersion.name == "ensemble", ModelVersion.is_active == True)
            .order_by(ModelVersion.created_at.desc())
            .first()
        )
        if version and version.ensemble_weights:
            prediction_service.set_learned_weights(version.ensemble_weights)
            result["weights_loaded"] = True
    except Exception as exc:  # pragma: no cover
        logger.warning("Ensemble ağırlıkları yüklenirken hata (yoksayılıyor): %s", exc)

    logger.info("Model yükleme sonucu: %s", result)
    return result
