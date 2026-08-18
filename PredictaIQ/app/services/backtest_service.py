"""Backtest / model doğrulama motoru.

Plan madde 26'nın en kritik uyarısı: "gelecekteki oranları veya maç
sonucunu geçmişteki feature'lara sızdırmayacağız. Aksi halde model
testte mükemmel görünür ama gerçek hayatta çöker." Bu servis bunu
somutlaştırıyor:

1. Maçları KRONOLOJİK olarak K katmana (fold) böler.
2. Her fold için, SADECE o foldun başlangıcından ÖNCEKİ maçlarla
   similarity + ML modellerini eğitir, foldun içindeki maçları tahmin
   eder. Yani hiçbir zaman "gelecekteki" bir maç, geçmiş bir maçın
   eğitim setine sızmaz.
3. Tüm foldlardaki tahminleri biriktirip Brier Score, Log Loss ve
   kalibrasyon eğrisini hesaplar (CalibrationService üzerinden).
4. Plan madde 3'ün "ağırlıkları sabit bırakmak yerine öğrenmeye izin
   verelim" talebini karşılamak için: biriken out-of-sample alt-model
   tahminlerinden (poisson/elo/xg/similarity/market/market_movement/ml),
   log loss'u minimize eden non-negative, toplamı 1 olan ensemble
   ağırlıklarını `scipy.optimize` ile öğrenir.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from app.models import Match

logger = logging.getLogger(__name__)

OUTCOME_INDEX = {"home": 0, "draw": 1, "away": 2}
SUB_MODEL_KEYS = ["poisson", "elo", "xg", "similarity", "market", "market_movement", "ml"]


def _actual_outcome(match: Match) -> str:
    if match.home_score > match.away_score:
        return "home"
    if match.home_score == match.away_score:
        return "draw"
    return "away"


class BacktestService:
    def __init__(self, prediction_service):
        # Döngüsel import'tan kaçınmak için tip belirtmiyoruz; PredictionService
        # instance'ı beklenir. Calibration servisi KASITLI olarak
        # prediction_service.calibration ile AYNI nesne -- ayrı bir örnek
        # oluşturulursa burada eğitilen kalibratör, canlı tahminlerde
        # kullanılan nesneye hiç yansımaz (önceki sürümdeki hata).
        self.prediction_service = prediction_service
        self.calibration = prediction_service.calibration

    def walk_forward(self, db, matches: List[Match], n_folds: int = 5, min_train: int = 60) -> Dict:
        finished = sorted(
            [m for m in matches if m.home_score is not None and m.away_score is not None],
            key=lambda m: m.date,
        )
        if len(finished) < min_train + 10:
            return {
                "status": "insufficient_data",
                "n_matches": len(finished),
                "required_minimum": min_train + 10,
            }

        # İlk min_train maç her zaman eğitim setinde; kalanı fold'lara böl.
        holdout = finished[min_train:]
        fold_size = max(1, len(holdout) // n_folds)

        all_predicted_home: List[float] = []
        all_outcomes_home: List[int] = []
        all_predicted_by_class: Dict[str, List[float]] = {"home": [], "draw": [], "away": []}
        all_outcomes_by_class: Dict[str, List[int]] = {"home": [], "draw": [], "away": []}
        sub_model_probs: Dict[str, List[List[float]]] = {k: [] for k in SUB_MODEL_KEYS}
        sub_model_outcomes: List[int] = []
        fold_reports = []

        for fold_idx in range(n_folds):
            start = min_train + fold_idx * fold_size
            end = start + fold_size if fold_idx < n_folds - 1 else len(finished)
            if start >= len(finished):
                break

            train_matches = finished[:start]
            test_matches = finished[start:end]
            if not test_matches:
                continue

            # --- SIZINTISIZ YENİDEN EĞİTİM: sadece geçmiş veriyle ---
            self.prediction_service.similarity.train(train_matches)
            X_train, y_train = self.prediction_service.build_training_matrix(train_matches)
            self.prediction_service.ml.train(X_train, y_train)

            fold_home_preds, fold_home_outcomes = [], []
            for m in test_matches:
                result = self.prediction_service.predict_match(m, apply_calibration=False)
                probs = result["model_probability"]
                outcome = _actual_outcome(m)
                outcome_idx = OUTCOME_INDEX[outcome]

                fold_home_preds.append(probs["home"])
                fold_home_outcomes.append(1 if outcome == "home" else 0)
                all_predicted_home.append(probs["home"])
                all_outcomes_home.append(1 if outcome == "home" else 0)

                for cls in ("home", "draw", "away"):
                    all_predicted_by_class[cls].append(probs[cls])
                    all_outcomes_by_class[cls].append(1 if outcome == cls else 0)

                for key in SUB_MODEL_KEYS:
                    detail = result["model_details"].get(key)
                    if not detail:
                        continue
                    vec = [detail.get("home", 1 / 3), detail.get("draw", 1 / 3), detail.get("away", 1 / 3)]
                    sub_model_probs[key].append(vec)
                sub_model_outcomes.append(outcome_idx)

            if fold_home_preds:
                brier = self.calibration.calculate_brier_score(fold_home_preds, fold_home_outcomes)
                fold_reports.append({
                    "fold": fold_idx + 1,
                    "train_size": len(train_matches),
                    "test_size": len(test_matches),
                    "brier_score_home": float(brier),
                })

        if not all_predicted_home:
            return {"status": "no_predictions_generated"}

        overall_brier = float(self.calibration.calculate_brier_score(all_predicted_home, all_outcomes_home))
        overall_logloss = float(self.calibration.calculate_log_loss(all_predicted_home, all_outcomes_home))
        try:
            calib = self.calibration.calculate_calibration_error(all_predicted_home, all_outcomes_home)
        except Exception as exc:  # az veri ile calibration_curve patlayabilir
            logger.warning("Kalibrasyon eğrisi hesaplanamadı: %s", exc)
            calib = None

        learned_weights = self._learn_ensemble_weights(sub_model_probs, sub_model_outcomes)

        # Nihai calibration modelini TÜM out-of-sample tahminlerle eğit ki
        # canlı tahminlerde kullanılabilsin. Çok sınıflı (home/draw/away)
        # kalibrasyon kullanılır -- PredictionService.calibrate_probabilities
        # bunu bekler.
        self.calibration.fit_multiclass(all_predicted_by_class, all_outcomes_by_class)

        return {
            "status": "completed",
            "n_matches_evaluated": len(all_predicted_home),
            "folds": fold_reports,
            "overall_brier_score": overall_brier,
            "overall_log_loss": overall_logloss,
            "calibration": calib,
            "learned_ensemble_weights": learned_weights,
        }

    def _learn_ensemble_weights(self, sub_model_probs: Dict[str, List[List[float]]],
                                 outcomes: List[int]) -> Dict[str, float]:
        """Log loss'u minimize eden non-negative, toplamı 1 olan ağırlıkları
        scipy.optimize.minimize (SLSQP) ile öğrenir.

        Yeterli veri yoksa veya bir alt modelin hiç tahmini yoksa, o modele
        varsayılan (eşit) ağırlık verilir.
        """
        active_keys = [k for k in SUB_MODEL_KEYS if len(sub_model_probs.get(k, [])) == len(outcomes) and len(outcomes) > 0]
        if len(active_keys) < 2 or len(outcomes) < 20:
            n = len(SUB_MODEL_KEYS)
            return {k: round(1 / n, 4) for k in SUB_MODEL_KEYS}

        stacked = np.array([sub_model_probs[k] for k in active_keys])  # (n_models, n_samples, 3)
        y = np.array(outcomes)
        eps = 1e-9

        def neg_log_likelihood(weights):
            w = np.abs(weights)
            w = w / (w.sum() + eps)
            blended = np.tensordot(w, stacked, axes=(0, 0))  # (n_samples, 3)
            blended = np.clip(blended, eps, 1 - eps)
            picked = blended[np.arange(len(y)), y]
            return -np.mean(np.log(picked))

        x0 = np.ones(len(active_keys)) / len(active_keys)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1}]
        bounds = [(0.0, 1.0)] * len(active_keys)

        try:
            res = minimize(neg_log_likelihood, x0, method="SLSQP", bounds=bounds,
                            constraints=constraints, options={"maxiter": 200})
            weights = np.abs(res.x)
            weights = weights / weights.sum()
        except Exception as exc:  # pragma: no cover
            logger.warning("Ensemble ağırlık öğrenimi başarısız, eşit ağırlık kullanılıyor: %s", exc)
            weights = x0

        result = {k: round(float(w), 4) for k, w in zip(active_keys, weights)}
        for k in SUB_MODEL_KEYS:
            result.setdefault(k, 0.0)
        return result
