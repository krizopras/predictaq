from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import Match, OddsSnapshot
from app.schemas import OddsAnalysisResponse
from app.services.odds_service import OddsService

router = APIRouter()
odds_service = OddsService()


@router.get("/analysis/{match_id}", response_model=OddsAnalysisResponse)
async def odds_analysis(match_id: str, db: Session = Depends(get_session)) -> OddsAnalysisResponse:
    """Plan madde 6-19'daki oran analizini tek yanıtta toplar: marjı
    arındırılmış piyasa olasılığı, açılış->kapanış hareketi, bookmaker
    konsensüsü/dağılımı ve sharp hareket tespiti."""
    match = db.query(Match).filter(Match.sportsdata_id == match_id).first()
    if not match:
        match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Maç bulunamadı")

    normalized = odds_service.normalize_odds(
        match.closing_home_odds or 0, match.closing_draw_odds or 0, match.closing_away_odds or 0
    )

    movement = {
        "home": odds_service.calculate_movement(match.opening_home_odds or 0, match.closing_home_odds or 0),
        "draw": odds_service.calculate_movement(match.opening_draw_odds or 0, match.closing_draw_odds or 0),
        "away": odds_service.calculate_movement(match.opening_away_odds or 0, match.closing_away_odds or 0),
    }

    snapshots = (
        db.query(OddsSnapshot)
        .filter(OddsSnapshot.match_id == match.id)
        .order_by(OddsSnapshot.timestamp.asc())
        .all()
    )
    snapshot_dicts = [
        {"bookmaker": s.bookmaker, "home_odds": s.home_odds, "draw_odds": s.draw_odds, "away_odds": s.away_odds}
        for s in snapshots
    ]
    consensus = odds_service.calculate_bookmaker_consensus(snapshot_dicts)
    sharp_movement = odds_service.identify_sharp_movement(snapshot_dicts)

    value_analysis = {
        "home": odds_service.calculate_ev(normalized["home"], match.closing_home_odds or 0),
        "draw": odds_service.calculate_ev(normalized["draw"], match.closing_draw_odds or 0),
        "away": odds_service.calculate_ev(normalized["away"], match.closing_away_odds or 0),
    }

    return OddsAnalysisResponse(
        normalized_odds=normalized,
        movement=movement,
        consensus=consensus,
        sharp_movement=sharp_movement,
        value_analysis=value_analysis,
    )


@router.get("/snapshots/{match_id}")
async def odds_snapshots(match_id: str, db: Session = Depends(get_session)) -> List[dict]:
    """Bir maça ait tüm bookmaker/zaman snapshot'larını döner (madde 6-7)."""
    match = db.query(Match).filter(Match.sportsdata_id == match_id).first()
    if not match:
        match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Maç bulunamadı")

    snapshots = (
        db.query(OddsSnapshot)
        .filter(OddsSnapshot.match_id == match.id)
        .order_by(OddsSnapshot.timestamp.asc())
        .all()
    )
    return [
        {
            "bookmaker": s.bookmaker,
            "source": s.source,
            "timestamp": s.timestamp.isoformat() if s.timestamp else None,
            "home_odds": s.home_odds,
            "draw_odds": s.draw_odds,
            "away_odds": s.away_odds,
        }
        for s in snapshots
    ]
