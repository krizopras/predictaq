"""PredictaIQ Ensemble Prediction Service.

Plan madde 12-21'in tam karşılığı. Eski sürümden farkları:

1. Poisson/xG modelleri artık maçın GERÇEKLEŞEN xG'sini değil, maç öncesi
   tahmini xG'yi (Match.home_xg_pre/away_xg_pre) kullanır -- sızıntı yok.
2. Market Movement, ayrı bir model (Model 6) olarak ensemble'a katılıyor.
3. ML Engine (XGBoost/LightGBM/CatBoost) ensemble'a katılıyor; eğitilmemişse
   ağırlığı otomatik olarak diğer modellere dağıtılır.
4. Kadro/sakatlık etkisi (PlayerImpactService), Elo/Poisson girdilerine
   bir "elo_penalty" olarak enjekte ediliyor.
5. Ensemble ağırlıkları artık sabit değil: BacktestService'in öğrendiği
   ağırlıklar varsa onlar kullanılır (ModelVersion.ensemble_weights),
   yoksa duyarlı varsayılanlara düşülür.
6. Son olasılıklar CalibrationService ile kalibre edilir (varsa).
7. Çıktı; historical similar matches, market movement detayı, data
   quality skoru gibi plan madde 20'nin istediği alanları içerir.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from app.models import Match
from app.services.calibration_service import CalibrationService
from app.services.elo_service import EloService
from app.services.feature_engineering import build_match_feature_vector
from app.services.market_movement_service import MarketMovementModel
from app.services.ml_service import MLEnsembleService
from app.services.odds_service import OddsService
from app.services.player_service import PlayerImpactService
from app.services.poisson_service import PoissonService
from app.services.similarity_service import SimilarityService
from app.services.xg_service import XGService

DEFAULT_WEIGHTS = {
    "poisson": 0.20,
    "elo": 0.15,
    "xg": 0.15,
    "similarity": 0.20,
    "market": 0.15,
    "market_movement": 0.05,
    "ml": 0.10,
}


class PredictionService:
    def __init__(self):
        self.poisson = PoissonService()
        self.elo = EloService()
        self.xg = XGService()
        self.similarity = SimilarityService()
        self.odds = OddsService()
        self.calibration = CalibrationService()
        self.market_movement = MarketMovementModel(self.odds)
        self.ml = MLEnsembleService()
        self.player_impact = PlayerImpactService()

        # Varsayılan ağırlıklar; BacktestService.walk_forward çalıştırılıp
        # `set_learned_weights` çağrıldığında bu değerler güncellenir
        # (plan madde 3: "ağırlıkları ML'e öğretelim").
        self.model_weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)
        self.weights_are_learned = False

    # ------------------------------------------------------------------
    # Ağırlık yönetimi
    # ------------------------------------------------------------------
    def set_learned_weights(self, weights: Dict[str, float]) -> None:
        cleaned = {k: max(0.0, float(v)) for k, v in weights.items() if k in DEFAULT_WEIGHTS}
        total = sum(cleaned.values())
        if total <= 0:
            return
        self.model_weights = {k: cleaned.get(k, 0.0) / total for k in DEFAULT_WEIGHTS}
        self.weights_are_learned = True

    def _active_weights(self, available_models: List[str]) -> Dict[str, float]:
        """Bir modelin bu tahmin için verisi yoksa (örn. ML eğitilmemiş,
        market movement verisi yoksa), ağırlığını mevcut olan diğer
        modellere orantılı şekilde yeniden dağıtır."""
        weights = {k: self.model_weights.get(k, 0.0) for k in available_models}
        total = sum(weights.values())
        if total <= 0:
            n = len(available_models)
            return {k: 1 / n for k in available_models}
        return {k: v / total for k, v in weights.items()}

    # ------------------------------------------------------------------
    # Eğitim verisi / feature matrix
    # ------------------------------------------------------------------
    def build_training_matrix(self, matches: List[Match]) -> Tuple[np.ndarray, np.ndarray]:
        """ML Engine ve backtest için (X, y) eğitim matrisini kurar.
        Sadece bitmiş maçlar ve sızıntısız (pre-match) alanlar kullanılır."""
        X, y = [], []
        outcome_map = {"home": 0, "draw": 1, "away": 2}
        for m in matches:
            if m.home_score is None or m.away_score is None:
                continue
            vec = build_match_feature_vector(
                home_elo=m.home_elo, away_elo=m.away_elo,
                home_form5=m.home_team.form_last5 if m.home_team else None,
                away_form5=m.away_team.form_last5 if m.away_team else None,
                home_xg_pre=m.home_xg_pre, away_xg_pre=m.away_xg_pre,
                opening_home_odds=m.opening_home_odds, opening_draw_odds=m.opening_draw_odds,
                opening_away_odds=m.opening_away_odds,
                closing_home_odds=m.closing_home_odds, closing_draw_odds=m.closing_draw_odds,
                closing_away_odds=m.closing_away_odds,
            )
            X.append(vec)
            if m.home_score > m.away_score:
                y.append(0)
            elif m.home_score == m.away_score:
                y.append(1)
            else:
                y.append(2)
        if not X:
            return np.empty((0, 0)), np.empty((0,))
        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------
    def predict_match(self, match: Match, apply_calibration: bool = True,
                       home_injury_impact: Optional[float] = None,
                       away_injury_impact: Optional[float] = None) -> Dict:
        """Tüm modelleri kullanarak maç tahmini yapar."""

        home_elo_base = match.home_team.elo if match.home_team else (match.home_elo or 1500)
        away_elo_base = match.away_team.elo if match.away_team else (match.away_elo or 1500)

        # Kadro/sakatlık etkisini Elo'ya enjekte et (madde 5)
        h_inj = home_injury_impact if home_injury_impact is not None else 0.0
        a_inj = away_injury_impact if away_injury_impact is not None else 0.0
        home_elo_adj = home_elo_base + h_inj  # h_inj zaten negatif bir Elo cezası olarak gelir
        away_elo_adj = away_elo_base + a_inj

        # Maç öncesi beklenen gol (sızıntısız); yoksa Elo farkından kabaca türet
        home_xg_pre = match.home_xg_pre
        away_xg_pre = match.away_xg_pre
        if home_xg_pre is None or away_xg_pre is None:
            elo_diff = home_elo_adj - away_elo_adj
            home_xg_pre = float(np.clip(1.45 + elo_diff / 400.0, 0.3, 4.0))
            away_xg_pre = float(np.clip(1.15 - elo_diff / 400.0, 0.3, 4.0))

        # 1. Poisson modeli
        poisson_result = self.poisson.predict_match(home_xg_pre, away_xg_pre)
        poisson_probs = {"home": poisson_result["home_win"], "draw": poisson_result["draw"], "away": poisson_result["away_win"]}

        # 2. Elo modeli (kadro/sakatlık ile ayarlanmış)
        elo_result = self.elo.calculate_win_probability(home_elo_adj, away_elo_adj)

        # 3. xG modeli (maç öncesi tahmini xG üzerinden)
        xg_result = self.xg.expected_goals_to_probability(home_xg_pre, away_xg_pre)
        xg_probs = {"home": xg_result["home"], "draw": xg_result["draw"], "away": xg_result["away"]}

        # 4. Benzerlik modeli (madde 8, 22)
        league_id = None
        try:
            league_id = match.season.competition_id if match.season else None
        except Exception:
            league_id = None
        similarity_result = self.similarity.find_similar_matches(
            match, n_neighbors=50, home_injury_impact=h_inj, away_injury_impact=a_inj, league_id=league_id
        )
        if similarity_result.get("count", 0) == 0:
            similarity_probs = None  # ağırlığı diğer modellere dağıtılacak
        else:
            similarity_probs = {"home": similarity_result["home"], "draw": similarity_result["draw"], "away": similarity_result["away"]}

        # 5. Piyasa modeli (marj arındırılmış)
        market_probs = self.odds.normalize_odds(
            match.closing_home_odds or 2.0,
            match.closing_draw_odds or 3.5,
            match.closing_away_odds or 4.0
        )

        # 6. Market Movement modeli (Model 6, madde 13/18)
        movement_result = self.market_movement.predict(
            match.opening_home_odds, match.opening_draw_odds, match.opening_away_odds,
            match.closing_home_odds, match.closing_draw_odds, match.closing_away_odds,
        )
        market_movement_probs = None
        if movement_result.get("has_movement_data"):
            market_movement_probs = {k: movement_result[k] for k in ("home", "draw", "away")}

        # 7. ML Engine (madde 13, Model 3)
        ml_vector = build_match_feature_vector(
            home_elo=home_elo_adj, away_elo=away_elo_adj,
            home_form5=match.home_team.form_last5 if match.home_team else None,
            away_form5=match.away_team.form_last5 if match.away_team else None,
            home_xg_pre=home_xg_pre, away_xg_pre=away_xg_pre,
            opening_home_odds=match.opening_home_odds, opening_draw_odds=match.opening_draw_odds,
            opening_away_odds=match.opening_away_odds,
            closing_home_odds=match.closing_home_odds, closing_draw_odds=match.closing_draw_odds,
            closing_away_odds=match.closing_away_odds,
            home_injury_impact=h_inj, away_injury_impact=a_inj,
        )
        ml_probs = self.ml.predict_proba(ml_vector)

        # --- Ensemble ---
        model_probs = {
            "poisson": poisson_probs,
            "elo": elo_result,
            "xg": xg_probs,
            "similarity": similarity_probs,
            "market": market_probs,
            "market_movement": market_movement_probs,
            "ml": ml_probs,
        }
        available = [k for k, v in model_probs.items() if v is not None]
        weights = self._active_weights(available)

        ensemble_home = sum(weights[k] * model_probs[k]["home"] for k in available)
        ensemble_draw = sum(weights[k] * model_probs[k]["draw"] for k in available)
        ensemble_away = sum(weights[k] * model_probs[k]["away"] for k in available)

        total = ensemble_home + ensemble_draw + ensemble_away
        if total > 0:
            ensemble_home /= total
            ensemble_draw /= total
            ensemble_away /= total
        else:
            ensemble_home, ensemble_draw, ensemble_away = 1 / 3, 1 / 3, 1 / 3

        # --- Kalibrasyon (madde 15) ---
        calibration_applied = False
        if apply_calibration and self.calibration.is_fitted:
            calibrated = self.calibration.calibrate_probabilities(ensemble_home, ensemble_draw, ensemble_away)
            ensemble_home, ensemble_draw, ensemble_away = calibrated["home"], calibrated["draw"], calibrated["away"]
            calibration_applied = True

        confidence = self._calculate_confidence(ensemble_home, ensemble_draw, ensemble_away)
        data_quality = self._calculate_data_quality(match, available)

        # Fair odds
        fair_home = 1 / ensemble_home if ensemble_home > 0 else 999
        fair_draw = 1 / ensemble_draw if ensemble_draw > 0 else 999
        fair_away = 1 / ensemble_away if ensemble_away > 0 else 999

        # Value analysis (madde 21)
        value_home = self.odds.calculate_ev(ensemble_home, match.closing_home_odds or 0)
        value_draw = self.odds.calculate_ev(ensemble_draw, match.closing_draw_odds or 0)
        value_away = self.odds.calculate_ev(ensemble_away, match.closing_away_odds or 0)

        return {
            "model_probability": {"home": ensemble_home, "draw": ensemble_draw, "away": ensemble_away},
            "fair_odds": {"home": fair_home, "draw": fair_draw, "away": fair_away},
            "confidence": confidence,
            "data_quality": data_quality,
            "calibration_applied": calibration_applied,
            "weights_used": weights,
            "weights_are_learned": self.weights_are_learned,
            "model_details": {
                "poisson": poisson_probs,
                "elo": elo_result,
                "xg": xg_result,
                "similarity": similarity_result,
                "market": market_probs,
                "market_movement": movement_result,
                "ml": ml_probs,
            },
            "value": {"home": value_home, "draw": value_draw, "away": value_away},
            "recommendation": self._get_recommendation(
                ensemble_home, ensemble_draw, ensemble_away,
                value_home, value_draw, value_away
            ),
        }

    # ------------------------------------------------------------------
    def _calculate_confidence(self, home: float, draw: float, away: float) -> float:
        """Modelin güven seviyesini hesaplar."""
        probs = [home, draw, away]
        max_prob = max(probs)
        second_max = sorted(probs)[-2]
        margin = max_prob - second_max

        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(3)
        entropy_score = 1 - (entropy / max_entropy)

        confidence = 0.6 * (1 - 2 / 3 * (1 - margin * 3)) + 0.4 * entropy_score
        return max(0, min(1, confidence)) * 100

    def _calculate_data_quality(self, match: Match, available_models: List[str]) -> float:
        """Plan madde 20'deki 'DATA QUALITY 91/100' göstergesinin karşılığı.
        Kaç veri katmanının dolu olduğuna göre 0-100 arası bir skor üretir."""
        checks = [
            match.home_elo is not None,
            match.away_elo is not None,
            match.home_xg_pre is not None,
            match.away_xg_pre is not None,
            bool(match.opening_home_odds),
            bool(match.closing_home_odds),
            "similarity" in available_models,
            "ml" in available_models,
            "market_movement" in available_models,
        ]
        return round(100 * sum(checks) / len(checks), 1)

    def _get_recommendation(self, home: float, draw: float, away: float,
                             value_home: Dict, value_draw: Dict, value_away: Dict) -> Dict:
        """Bahis önerisi oluşturur. NOT: Bu tek başına bahis tavsiyesi değildir;
        calibration/backtest sonuçlarıyla birlikte değerlendirilmelidir."""
        recommendations = []

        if value_home["is_positive"] and home > 0.35:
            recommendations.append({
                "bet": "Home", "odds_type": "1X2", "edge": value_home["edge"], "ev": value_home["ev"],
                "confidence": "High" if home > 0.50 else "Medium",
            })
        if value_draw["is_positive"] and draw > 0.25:
            recommendations.append({
                "bet": "Draw", "odds_type": "1X2", "edge": value_draw["edge"], "ev": value_draw["ev"],
                "confidence": "High" if draw > 0.35 else "Medium",
            })
        if value_away["is_positive"] and away > 0.30:
            recommendations.append({
                "bet": "Away", "odds_type": "1X2", "edge": value_away["edge"], "ev": value_away["ev"],
                "confidence": "High" if away > 0.45 else "Medium",
            })

        if recommendations:
            best = max(recommendations, key=lambda x: x["ev"])
            best["is_best"] = True

        return {
            "recommendations": recommendations,
            "total_positive_ev": sum(1 for r in recommendations if r["ev"] > 0),
        }
