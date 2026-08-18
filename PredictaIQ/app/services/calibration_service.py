import logging
import os

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple
import joblib

logger = logging.getLogger(__name__)

class CalibrationService:
    def __init__(self):
        self.isotonic_reg = None
        self.platt_scaling = None
        self.calibration_data = []
        # Çok sınıflı (home/draw/away) kalibrasyon için sınıf başına bir
        # isotonic regressor. PredictionService'in nihai ensemble çıktısını
        # kalibre etmek için kullanılır (plan madde 14->15 akışı).
        self.class_calibrators: Dict[str, IsotonicRegression] = {}

    def fit_multiclass(self, predictions_by_class: Dict[str, List[float]],
                        outcomes_by_class: Dict[str, List[int]]) -> None:
        """Her sınıf (home/draw/away) için ayrı bir isotonic regressor eğitir.

        predictions_by_class["home"] = [model'in home olasılığı, ...]
        outcomes_by_class["home"]    = [1 eğer gerçekten home kazandıysa else 0, ...]
        """
        for cls in ("home", "draw", "away"):
            preds = predictions_by_class.get(cls, [])
            outs = outcomes_by_class.get(cls, [])
            if len(preds) < 20:
                continue
            reg = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
            reg.fit(preds, outs)
            self.class_calibrators[cls] = reg

    def calibrate_probabilities(self, home: float, draw: float, away: float) -> Dict[str, float]:
        """Ensemble'ın (home, draw, away) çıktısını kalibre edip yeniden
        normalize eder. Bir sınıf için kalibratör yoksa o sınıf ham
        değeriyle geçer."""
        raw = {"home": home, "draw": draw, "away": away}
        calibrated = {}
        for cls, val in raw.items():
            reg = self.class_calibrators.get(cls)
            if reg is not None:
                calibrated[cls] = float(reg.predict([val])[0])
            else:
                calibrated[cls] = val

        total = sum(calibrated.values())
        if total <= 0:
            return raw
        return {k: v / total for k, v in calibrated.items()}

    @property
    def is_fitted(self) -> bool:
        return len(self.class_calibrators) > 0

    def fit(self, predictions: List[float], outcomes: List[int]):
        """Kalibrasyon modelini eğitir"""
        if len(predictions) < 10:
            return
        
        # Isotonic Regression
        self.isotonic_reg = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_reg.fit(predictions, outcomes)
        
        # Platt Scaling (lojistik regresyon ile)
        from sklearn.linear_model import LogisticRegression
        self.platt_scaling = LogisticRegression()
        self.platt_scaling.fit(np.array(predictions).reshape(-1, 1), outcomes)
    
    def calibrate(self, probabilities: List[float]) -> List[float]:
        """Olasılıkları kalibre eder"""
        if self.isotonic_reg is not None:
            return self.isotonic_reg.transform(probabilities)
        return probabilities
    
    def calculate_brier_score(self, predictions: List[float], outcomes: List[int]) -> float:
        """Brier Score hesaplar"""
        return np.mean((np.array(predictions) - np.array(outcomes)) ** 2)
    
    def calculate_log_loss(self, predictions: List[float], outcomes: List[int]) -> float:
        """Log Loss hesaplar"""
        eps = 1e-15
        pred = np.clip(np.asarray(predictions, dtype=float), eps, 1 - eps)
        outcomes_arr = np.asarray(outcomes, dtype=float)
        return -np.mean(outcomes_arr * np.log(pred) + (1 - outcomes_arr) * np.log(1 - pred))
    
    def calculate_calibration_error(self, predictions: List[float], outcomes: List[int],
                                   n_bins: int = 10) -> Dict:
        """Kalibrasyon hatasını hesaplar.

        NOT: sklearn.calibration_curve, veri az/dar bir aralıkta toplanmışsa
        istenen n_bins sayısından DAHA AZ bin döndürebilir. Önceki sürüm bunu
        hesaba katmadan `np.histogram(..., bins=n_bins)` ile sabit n_bins
        uzunluklu bir dizi üretiyordu ve iki dizi farklı uzunlukta olduğunda
        broadcast hatası veriyordu. Çözüm: aynı bin kenarlarını
        (aynı `bins=` parametresi) her iki hesaplamada da kullanmak.
        """
        prob_true, prob_pred = calibration_curve(outcomes, predictions, n_bins=n_bins, strategy="uniform")

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_counts, _ = np.histogram(predictions, bins=bin_edges)
        # calibration_curve, boş bin'leri otomatik atlar; bu yüzden dönen
        # prob_true/prob_pred uzunluğu, dolu bin sayısına eşittir. Ağırlıklı
        # ortalamayı hizalamak için sadece dolu bin'lerin sayacını kullan.
        non_empty_counts = bin_counts[bin_counts > 0]
        n = min(len(prob_true), len(non_empty_counts))
        bin_errors = np.abs(prob_true[:n] - prob_pred[:n])
        weights = non_empty_counts[:n]

        total_error = float(np.sum(bin_errors * weights) / np.sum(weights)) if np.sum(weights) > 0 else None

        return {
            "calibration_error": total_error,
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "bin_counts": bin_counts.tolist()
        }
    
    def reliability_diagram(self, predictions: List[float], outcomes: List[int],
                           n_bins: int = 10) -> Dict:
        """Reliability diagram verilerini oluşturur"""
        prob_true, prob_pred = calibration_curve(outcomes, predictions, n_bins=n_bins)
        
        return {
            "x": prob_pred.tolist(),
            "y": prob_true.tolist(),
            "perfect": [0, 1],
            "perfect_x": [0, 1],
            "perfect_y": [0, 1]
        }
    
    def save_model(self, path: str):
        """Kalibrasyon modelini kaydeder"""
        if self.isotonic_reg is not None:
            joblib.dump(self.isotonic_reg, f"{path}_isotonic.joblib")
        if self.platt_scaling is not None:
            joblib.dump(self.platt_scaling, f"{path}_platt.joblib")
    
    def load_model(self, path: str):
        """Kalibrasyon modelini yükler"""
        try:
            self.isotonic_reg = joblib.load(f"{path}_isotonic.joblib")
            self.platt_scaling = joblib.load(f"{path}_platt.joblib")
        except:
            print("Kalibrasyon modeli yüklenemedi")

    # ------------------------------------------------------------------
    # PredictionService.calibrate_probabilities() TARAFINDAN GERÇEKTEN
    # KULLANILAN state `class_calibrators` (home/draw/away başına bir
    # isotonic regressor) -- yukarıdaki save_model/load_model tek-sınıflı
    # isotonic_reg/platt_scaling içindir ve ensemble tarafından hiç
    # kullanılmaz. Production kalıcılığı için doğru state bu ikisidir.
    # ------------------------------------------------------------------
    def save(self, dir_path: str) -> None:
        if not self.class_calibrators:
            logger.info("Calibration: eğitilmemiş, kaydedilecek bir şey yok")
            return
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self.class_calibrators, os.path.join(dir_path, "class_calibrators.joblib"))
        logger.info("Calibration: %s içine kaydedildi", dir_path)

    def load(self, dir_path: str) -> bool:
        file_path = os.path.join(dir_path, "class_calibrators.joblib")
        if not os.path.exists(file_path):
            return False
        try:
            self.class_calibrators = joblib.load(file_path)
            logger.info("Calibration: %s içinden yüklendi (%s)", dir_path, list(self.class_calibrators.keys()))
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("Calibration yüklenemedi (yoksayılıyor): %s", exc)
            return False