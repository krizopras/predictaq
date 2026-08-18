from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_session
from app.dependencies import prediction_service
from app.models import Competition, Match, Prediction, Season
from app.schemas import PredictionResponse, PredictionRequest
from app.services.player_service import PlayerImpactService

router = APIRouter()
player_impact_service = PlayerImpactService()


def _build_response(match: Match, prediction: dict) -> PredictionResponse:
    return PredictionResponse(
        fixture_id=match.sportsdata_id or str(match.id),
        home_team=match.home_team.name if match.home_team else "Unknown",
        away_team=match.away_team.name if match.away_team else "Unknown",
        match_date=match.date,
        probabilities=prediction["model_probability"],
        fair_odds=prediction["fair_odds"],
        confidence=prediction["confidence"],
        data_quality=prediction.get("data_quality"),
        calibration_applied=prediction.get("calibration_applied"),
        weights_used=prediction.get("weights_used"),
        weights_are_learned=prediction.get("weights_are_learned"),
        model_details=prediction["model_details"],
        value=prediction["value"],
        recommendation=prediction["recommendation"],
    )


@router.get("/match/{match_id}", response_model=PredictionResponse)
async def predict_match(match_id: str, db: Session = Depends(get_session)) -> PredictionResponse:
    """Belirli bir maç için tahmin yapar (sportsdata_id veya iç UUID ile)."""
    match = db.query(Match).filter(Match.sportsdata_id == match_id).first()
    if not match:
        match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Maç bulunamadı")

    # Kadro/sakatlık etkisini hesapla (madde 5) ve tahmine enjekte et
    home_injury = player_impact_service.team_injury_impact(db, match.home_team_id) if match.home_team_id else {"elo_penalty": 0.0}
    away_injury = player_impact_service.team_injury_impact(db, match.away_team_id) if match.away_team_id else {"elo_penalty": 0.0}

    prediction = prediction_service.predict_match(
        match,
        home_injury_impact=home_injury.get("elo_penalty", 0.0),
        away_injury_impact=away_injury.get("elo_penalty", 0.0),
    )

    db_pred = Prediction(
        match_id=match.id,
        model_name="ensemble_v2",
        home_prob=prediction["model_probability"]["home"],
        draw_prob=prediction["model_probability"]["draw"],
        away_prob=prediction["model_probability"]["away"],
        confidence=prediction["confidence"] / 100,
        value_score=prediction["value"]["home"]["ev"] / 100,
        model_details=prediction["model_details"],
    )
    db.add(db_pred)
    db.commit()

    return _build_response(match, prediction)


@router.get("/batch", response_model=List[PredictionResponse])
async def predict_batch(
    league: Optional[str] = None,
    date: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_session),
) -> List[PredictionResponse]:
    """Toplu maç tahmini yapar."""
    query = db.query(Match)

    if league:
        query = query.join(Season, Match.season_id == Season.id).join(
            Competition, Season.competition_id == Competition.id
        ).filter(Competition.name == league)

    if date:
        try:
            target_date = datetime.fromisoformat(date)
            query = query.filter(Match.date >= target_date)
        except ValueError:
            pass

    matches = query.filter(Match.status == "scheduled").limit(limit).all()
    responses = []
    for match in matches:
        pred = prediction_service.predict_match(match)
        responses.append(_build_response(match, pred))
    return responses


@router.post("/update")
async def update_prediction(request: PredictionRequest, db: Session = Depends(get_session)):
    """Maç sonucu ile tahmin performansını günceller."""
    match = db.query(Match).filter(Match.sportsdata_id == request.fixture_id).first()
    if not match:
        raise HTTPException(404, "Maç bulunamadı")

    match.home_score = request.home_score
    match.away_score = request.away_score
    match.status = "finished"

    db.commit()

    return {"message": "Maç sonucu güncellendi"}
