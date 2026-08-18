from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_session
from app.dependencies import prediction_service
from app.models import Competition, Match, Season
from app.schemas import HistoricalOutcomeQuery, SimilarityResponse

router = APIRouter()


@router.post("/same-odds", response_model=SimilarityResponse)
async def outcomes_at_same_odds(
    query: HistoricalOutcomeQuery, db: Session = Depends(get_session)
) -> SimilarityResponse:
    """Plan madde 8: '1.70-1.80 arası oran verilen maçlarda ne oldu?' sorgusu.

    KNN modeline ihtiyaç duymadan, doğrudan kapanış oranı aralığına göre
    veritabanı sorgusu yapar -- planın ilk (basit) katmanı budur.
    """
    tol = query.tolerance_pct / 100.0

    def _range(value: float):
        return value * (1 - tol), value * (1 + tol)

    h_lo, h_hi = _range(query.home_odds)
    d_lo, d_hi = _range(query.draw_odds)
    a_lo, a_hi = _range(query.away_odds)

    db_query = (
        db.query(Match)
        .filter(Match.status == "finished")
        .filter(Match.home_score.isnot(None))
        .filter(Match.closing_home_odds.between(h_lo, h_hi))
        .filter(Match.closing_draw_odds.between(d_lo, d_hi))
        .filter(Match.closing_away_odds.between(a_lo, a_hi))
    )

    if query.league_id:
        db_query = db_query.join(Season, Match.season_id == Season.id).filter(
            Season.competition_id == query.league_id
        )

    matches = db_query.limit(2000).all()
    total = len(matches)
    if total == 0:
        return SimilarityResponse(count=0, home=0.33, draw=0.33, away=0.34, matches=[])

    home_wins = sum(1 for m in matches if m.home_score > m.away_score)
    draws = sum(1 for m in matches if m.home_score == m.away_score)
    away_wins = total - home_wins - draws

    sample = []
    for m in matches[:20]:
        sample.append({
            "home_team": m.home_team.name if m.home_team else "Unknown",
            "away_team": m.away_team.name if m.away_team else "Unknown",
            "date": m.date.isoformat(),
            "score": f"{m.home_score}-{m.away_score}",
            "closing_odds": [m.closing_home_odds, m.closing_draw_odds, m.closing_away_odds],
        })

    return SimilarityResponse(
        count=total,
        home=home_wins / total,
        draw=draws / total,
        away=away_wins / total,
        matches=sample,
    )


@router.get("/similar/{match_id}", response_model=SimilarityResponse)
async def similar_matches(match_id: str, n: int = 50, db: Session = Depends(get_session)) -> SimilarityResponse:
    """Plan madde 22: eğitilmiş KNN benzerlik motoruyla, verilen bir maça en
    çok benzeyen geçmiş maçları döner (Elo, form, xG, oran ve oran hareketi
    birlikte değerlendirilir)."""
    match = db.query(Match).filter(Match.sportsdata_id == match_id).first()
    if not match:
        match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Maç bulunamadı")

    league_id = match.season.competition_id if match.season else None
    result = prediction_service.similarity.find_similar_matches(match, n_neighbors=n, league_id=league_id)
    return SimilarityResponse(
        count=result.get("count", 0),
        home=result.get("home", 0.33),
        draw=result.get("draw", 0.33),
        away=result.get("away", 0.34),
        matches=result.get("matches", []),
    )
