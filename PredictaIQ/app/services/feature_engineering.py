"""Merkezi feature engineering katmanı.

Plan madde 22'de tarif edilen `match_vector` burada tek bir yerde
üretiliyor ve hem SimilarityService (KNN) hem MLEnsembleService
(XGBoost/LightGBM/CatBoost) hem de PredictionService (market movement,
elo, vs.) aynı fonksiyonu kullanıyor. Bunun tek yerde toplanmasının
sebebi tutarlılık: iki servisin birbirinden bağımsız, birbiriyle
örtüşmeyen feature setleri üretmesi (eski koddaki durum) hem bakımı
zorlaştırıyor hem de "similarity 12 feature kullanıyor, ML 40 feature
kullanıyor ama ikisi de farklı normalize ediyor" gibi tutarsızlıklara
yol açıyordu.

KRİTİK -- veri sızıntısı (leakage) önleme:
Buradaki hiçbir feature, maçın SONUCUNA (skor, xG gerçekleşen değeri vb.)
bağlı olmamalı. Sadece maç başlamadan ÖNCE bilinebilecek bilgiler
kullanılır: rating'ler, opening odds, kickoff'a kadarki oran hareketi,
kadro/sakatlık durumu. `home_xg`/`away_xg` alanları burada "maçın
sonucunda gerçekleşen xG" değil, DB'de varsa "maç öncesi tahmini
takım gücü" olarak treat edilir -- bu yüzden ml_service ve
similarity_service eğitim setini kurarken sadece `status == 'finished'`
olan GEÇMİŞ maçların rating/odds bilgisini, o maçın KENDİ SONUCUNU
tahmin etmek için kullanır; gelecekteki hiçbir maçın bilgisi geçmişe
sızdırılmaz (bkz. backtest_service.py).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

FEATURE_NAMES: List[str] = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form5",
    "away_form5",
    "form_diff",
    "home_xg_pre",
    "away_xg_pre",
    "xg_diff_pre",
    "opening_home_prob",
    "opening_draw_prob",
    "opening_away_prob",
    "closing_home_prob",
    "closing_draw_prob",
    "closing_away_prob",
    "odds_movement_home_pct",
    "odds_movement_draw_pct",
    "odds_movement_away_pct",
    "bookmaker_overround",
    "home_injury_impact",
    "away_injury_impact",
    "home_advantage_flag",
]


def _implied_probs(home_odds, draw_odds, away_odds) -> tuple:
    odds = [home_odds, draw_odds, away_odds]
    if any(o is None or o <= 0 for o in odds):
        return 0.33, 0.33, 0.34
    implied = [1.0 / o for o in odds]
    total = sum(implied)
    if total <= 0:
        return 0.33, 0.33, 0.34
    return tuple(p / total for p in implied)


def _pct_change(opening: Optional[float], closing: Optional[float]) -> float:
    if not opening or not closing or opening <= 0:
        return 0.0
    return (closing - opening) / opening * 100.0


def build_match_feature_vector(
    *,
    home_elo: Optional[float],
    away_elo: Optional[float],
    home_form5: Optional[float],
    away_form5: Optional[float],
    home_xg_pre: Optional[float],
    away_xg_pre: Optional[float],
    opening_home_odds: Optional[float],
    opening_draw_odds: Optional[float],
    opening_away_odds: Optional[float],
    closing_home_odds: Optional[float],
    closing_draw_odds: Optional[float],
    closing_away_odds: Optional[float],
    home_injury_impact: float = 0.0,
    away_injury_impact: float = 0.0,
) -> np.ndarray:
    """Tek bir maç için sabit sıralı, sabit uzunlukta feature vektörü üretir.

    Tüm girdiler maçın KİCKOFF ANINDA bilinen değerler olmalıdır.
    """
    home_elo = home_elo if home_elo is not None else 1500.0
    away_elo = away_elo if away_elo is not None else 1500.0
    home_form5 = home_form5 if home_form5 is not None else 50.0
    away_form5 = away_form5 if away_form5 is not None else 50.0
    home_xg_pre = home_xg_pre if home_xg_pre is not None else 1.4
    away_xg_pre = away_xg_pre if away_xg_pre is not None else 1.2

    op_h, op_d, op_a = _implied_probs(opening_home_odds, opening_draw_odds, opening_away_odds)
    cl_h, cl_d, cl_a = _implied_probs(closing_home_odds, closing_draw_odds, closing_away_odds)

    overround = 0.0
    if closing_home_odds and closing_draw_odds and closing_away_odds:
        overround = (1 / closing_home_odds + 1 / closing_draw_odds + 1 / closing_away_odds) - 1

    return np.array([
        home_elo,
        away_elo,
        home_elo - away_elo,
        home_form5,
        away_form5,
        home_form5 - away_form5,
        home_xg_pre,
        away_xg_pre,
        home_xg_pre - away_xg_pre,
        op_h, op_d, op_a,
        cl_h, cl_d, cl_a,
        _pct_change(opening_home_odds, closing_home_odds),
        _pct_change(opening_draw_odds, closing_draw_odds),
        _pct_change(opening_away_odds, closing_away_odds),
        overround,
        home_injury_impact,
        away_injury_impact,
        1.0,  # home_advantage_flag -- her zaman ev sahibi lehine sabit gösterge
    ], dtype=float)


def match_to_feature_dict(vector: np.ndarray) -> Dict[str, float]:
    return {name: float(val) for name, val in zip(FEATURE_NAMES, vector)}
