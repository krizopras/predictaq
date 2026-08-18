"""ML Engine: XGBoost + LightGBM (+ opsiyonel CatBoost) çoklu sınıf modeli.

Plan madde 13 (Model 3) ve madde 26'daki "ML Engine" katmanının karşılığı.
Eski kodda requirements.txt'de bu kütüphaneler kuruluydu ama onları
kullanan tek bir satır kod yoktu. Bu servis:

- feature_engineering.build_match_feature_vector ile üretilen, sızıntısız
  feature vektörlerini kullanarak 3 sınıflı (home/draw/away) olasılık
  tahmini yapan bir gradient boosting sınıflandırıcı eğitir.
- Birden fazla kütüphaneyi (varsa) soft-voting ile birleştirir; hiçbiri
  kurulu değilse veya veri yetersizse sessizce devre dışı kalır ve
  PredictionService bu modelin ağırlığını diğerlerine dağıtır.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:  # pragma: no cover
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:  # pragma: no cover
    _HAS_CATBOOST = False


class MLEnsembleService:
    def __init__(self):
        self.models: Dict[str, object] = {}
        self.is_trained = False
        self.n_features: Optional[int] = None
        self.training_size = 0

    @property
    def available_backends(self) -> List[str]:
        backends = []
        if _HAS_XGB:
            backends.append("xgboost")
        if _HAS_LGB:
            backends.append("lightgbm")
        if _HAS_CATBOOST:
            backends.append("catboost")
        return backends

    def train(self, X: np.ndarray, y: np.ndarray, min_matches: int = 150) -> Dict:
        """X: (n_samples, n_features) feature matrisi. y: 0=home,1=draw,2=away.

        Yalnızca `min_matches`'ten fazla örnek varsa eğitilir; az veriyle
        gradient boosting modelleri kolayca overfit eder ve ensemble'a
        gürültü katar.
        """
        n = len(y)
        if n < min_matches:
            logger.info("ML Engine: %d maç yeterli değil (min %d), eğitim atlandı", n, min_matches)
            self.is_trained = False
            return {"trained": False, "reason": "insufficient_data", "n_matches": n}

        self.n_features = X.shape[1]
        trained_backends = []

        if _HAS_XGB:
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", n_jobs=-1,
            )
            model.fit(X, y)
            self.models["xgboost"] = model
            trained_backends.append("xgboost")

        if _HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="multiclass", num_class=3, verbosity=-1,
            )
            model.fit(X, y)
            self.models["lightgbm"] = model
            trained_backends.append("lightgbm")

        if _HAS_CATBOOST:
            model = CatBoostClassifier(
                iterations=300, depth=5, learning_rate=0.05,
                loss_function="MultiClass", verbose=False,
            )
            model.fit(X, y)
            self.models["catboost"] = model
            trained_backends.append("catboost")

        self.is_trained = len(self.models) > 0
        self.training_size = n
        return {"trained": self.is_trained, "backends": trained_backends, "n_matches": n}

    def predict_proba(self, feature_vector: np.ndarray) -> Optional[Dict[str, float]]:
        """Tek bir maç için home/draw/away olasılıklarını döner. Model
        eğitilmemişse None döner (PredictionService bu durumda ML modelinin
        ağırlığını diğer modellere dağıtmalı)."""
        if not self.is_trained or not self.models:
            return None
        if self.n_features is not None and len(feature_vector) != self.n_features:
            logger.warning("ML Engine: feature boyutu uyuşmuyor, tahmin atlanıyor")
            return None

        X = feature_vector.reshape(1, -1)
        probs = []
        for name, model in self.models.items():
            try:
                p = model.predict_proba(X)[0]
                probs.append(p)
            except Exception as exc:  # pragma: no cover
                logger.warning("ML Engine backend %s tahmin hatası: %s", name, exc)

        if not probs:
            return None

        avg = np.mean(probs, axis=0)
        avg = avg / avg.sum()
        return {"home": float(avg[0]), "draw": float(avg[1]), "away": float(avg[2])}

    # ------------------------------------------------------------------
    # Kalıcılık (joblib) -- production'da modeli her process başlangıcında
    # yeniden EĞİTMEK yerine, GitHub Actions/manuel bir eğitim işinden
    # sonra diske yazılan modeli sadece YÜKLEMEK için kullanılır.
    # ------------------------------------------------------------------
    def save(self, dir_path: str) -> None:
        """Eğitilmiş her backend'i {dir_path}/{backend}.joblib olarak,
        meta bilgileri (n_features, training_size) meta.json olarak kaydeder.
        Model eğitilmemişse (self.is_trained False) hiçbir şey yazmaz."""
        if not self.is_trained:
            logger.info("ML Engine: eğitilmemiş model, kaydedilecek bir şey yok")
            return
        os.makedirs(dir_path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(dir_path, f"{name}.joblib"))
        with open(os.path.join(dir_path, "meta.json"), "w") as f:
            json.dump(
                {
                    "n_features": self.n_features,
                    "training_size": self.training_size,
                    "backends": list(self.models.keys()),
                },
                f,
            )
        logger.info("ML Engine: %s içine kaydedildi (%s)", dir_path, list(self.models.keys()))

    def load(self, dir_path: str) -> bool:
        """Diskten (varsa) eğitilmiş modelleri yükler. Dizin/dosyalar yoksa
        sessizce False döner -- ilk kurulumda henüz hiç eğitim yapılmamış
        olabilir, bu bir hata değildir."""
        meta_path = os.path.join(dir_path, "meta.json")
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            loaded = {}
            for name in meta.get("backends", []):
                model_path = os.path.join(dir_path, f"{name}.joblib")
                if os.path.exists(model_path):
                    loaded[name] = joblib.load(model_path)
            if not loaded:
                return False
            self.models = loaded
            self.n_features = meta.get("n_features")
            self.training_size = meta.get("training_size", 0)
            self.is_trained = True
            logger.info("ML Engine: %s içinden yüklendi (%s)", dir_path, list(loaded.keys()))
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("ML Engine yüklenemedi (yoksayılıyor): %s", exc)
            return False

    def feature_importance(self) -> Dict[str, List[float]]:
        """Her backend için feature importance (varsa) döner -- teşhis/rapor
        amaçlı, ensemble hesaplamasını etkilemez."""
        result = {}
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                result[name] = [float(v) for v in model.feature_importances_]
        return result
