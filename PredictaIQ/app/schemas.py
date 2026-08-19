from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Tahmin (Prediction)
# ---------------------------------------------------------------------------

class PredictionDetail(BaseModel):
    """Plan madde 20'deki zengin sonuç ekranının karşılığı: alt-model
    detayları, benzer maçlar, oran hareketi ve veri kalitesi tek yanıtta.

    NOT: Eski `routers/predictions.py` bu sınıfı schemas.py'den import
    ediyordu ama sınıf hiç tanımlanmamıştı -- bu da uygulamanın açılışta
    ImportError ile çökmesine sebep oluyordu.
    """
    poisson: Dict[str, float]
    elo: Dict[str, float]
    xg: Dict[str, float]
    similarity: Dict[str, Any]
    market: Dict[str, float]
    market_movement: Optional[Dict[str, Any]] = None
    ml: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    fixture_id: str
    home_team: str
    away_team: str
    match_date: datetime
    probabilities: Dict[str, float]
    fair_odds: Dict[str, float]
    confidence: float
    data_quality: Optional[float] = None
    calibration_applied: Optional[bool] = None
    weights_used: Optional[Dict[str, float]] = None
    weights_are_learned: Optional[bool] = None
    model_details: Optional[PredictionDetail] = None
    value: Dict[str, Any]
    recommendation: Dict[str, Any]


class PredictionRequest(BaseModel):
    fixture_id: str
    home_score: int
    away_score: int


# ---------------------------------------------------------------------------
# Maçlar
# ---------------------------------------------------------------------------

class MatchResponse(BaseModel):
    # Pydantic'in "model_" önekine verdiği uyarıyı kapatıyoruz
    model_config = ConfigDict(protected_namespaces=())

    id: str
    sportsdata_id: Optional[str]
    home_team: str
    away_team: str
    date: datetime
    status: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_xg_pre: Optional[float] = None
    away_xg_pre: Optional[float] = None
    opening_home_odds: Optional[float] = None
    opening_draw_odds: Optional[float] = None
    opening_away_odds: Optional[float] = None
    closing_home_odds: Optional[float] = None
    closing_draw_odds: Optional[float] = None
    closing_away_odds: Optional[float] = None
    # Son ensemble tahmini -- /admin/sync/live her koşuşta scheduled/live
    # maçlar için bunu tazeler, frontend ayrı bir /predictions çağrısı
    # yapmadan listede doğrudan gösterebilir.
    model_home_prob: Optional[float] = None
    model_draw_prob: Optional[float] = None
    model_away_prob: Optional[float] = None
    model_confidence: Optional[float] = None


class TeamResponse(BaseModel):
    id: str
    name: str
    country: Optional[str] = None
    elo: float
    attack: float
    defense: float
    form_last3: Optional[float] = None
    form_last5: Optional[float] = None
    form_last10: Optional[float] = None
    home_power: Optional[float] = None
    away_power: Optional[float] = None


# ---------------------------------------------------------------------------
# Benzerlik / Tarihsel
# ---------------------------------------------------------------------------

class SimilarityResponse(BaseModel):
    count: int
    home: float
    draw: float
    away: float
    matches: List[Dict[str, Any]]


class HistoricalOutcomeQuery(BaseModel):
    """'Aynı oranlarda geçmişte ne oldu?' sorgusu (plan madde 8)."""
    home_odds: float
    draw_odds: float
    away_odds: float
    league_id: Optional[str] = None
    tolerance_pct: float = Field(default=8.0, description="Oran aralığı toleransı (yüzde)")


# ---------------------------------------------------------------------------
# Oranlar
# ---------------------------------------------------------------------------

class OddsAnalysisResponse(BaseModel):
    normalized_odds: Dict[str, float]
    movement: Dict[str, Any]
    consensus: Dict[str, Any]
    sharp_movement: Dict[str, Any]
    value_analysis: Dict[str, Any]


# ---------------------------------------------------------------------------
# Admin / Eğitim
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    n_folds: int = Field(default=5, ge=2, le=10)
    min_train: int = Field(default=60, ge=10)
    persist_weights: bool = True


class TrainResponse(BaseModel):
    status: str
    n_matches_evaluated: Optional[int] = None
    overall_brier_score: Optional[float] = None
    overall_log_loss: Optional[float] = None
    learned_ensemble_weights: Optional[Dict[str, float]] = None
    folds: Optional[List[Dict[str, Any]]] = None
    detail: Optional[str] = None


class SyncLiveResponse(BaseModel):
    """`/admin/sync/live` yanıtı: her lig için kaç fixture çekildiği ve
    kaç maçın tahmininin tazelendiği (madde: 15 dakikada bir GitHub
    Actions tarafından tetiklenir)."""
    leagues: Dict[str, Any]
    matches_upserted: int
    predictions_updated: int


class ModelStatusResponse(BaseModel):
    similarity_trained: bool
    similarity_match_count: int
    ml_trained: bool
    ml_backends: List[str]
    ml_training_size: int
    calibration_fitted: bool
    weights_are_learned: bool
    current_weights: Dict[str, float]
