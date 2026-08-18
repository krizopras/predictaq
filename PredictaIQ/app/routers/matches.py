from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_session
from app.models import Competition, Match, Season, Team
from app.schemas import MatchResponse, TeamResponse

router = APIRouter()


def _match_to_response(match: Match) -> MatchResponse:
    return MatchResponse(
        id=str(match.id),
        sportsdata_id=match.sportsdata_id,
        home_team=match.home_team.name if match.home_team else "Unknown",
        away_team=match.away_team.name if match.away_team else "Unknown",
        date=match.date,
        status=match.status or "scheduled",
        home_score=match.home_score,
        away_score=match.away_score,
        home_xg=match.home_xg,
        away_xg=match.away_xg,
        home_xg_pre=match.home_xg_pre,
        away_xg_pre=match.away_xg_pre,
        opening_home_odds=match.opening_home_odds,
        opening_draw_odds=match.opening_draw_odds,
        opening_away_odds=match.opening_away_odds,
        closing_home_odds=match.closing_home_odds,
        closing_draw_odds=match.closing_draw_odds,
        closing_away_odds=match.closing_away_odds,
        model_home_prob=match.model_home_prob,
        model_draw_prob=match.model_draw_prob,
        model_away_prob=match.model_away_prob,
        model_confidence=match.model_confidence,
    )


@router.get("/", response_model=List[MatchResponse])
async def list_matches(
    league: Optional[str] = None,
    status: Optional[str] = Query(default=None, description="scheduled | live | finished | postponed | cancelled"),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    team: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_session),
) -> List[MatchResponse]:
    query = db.query(Match)

    if league:
        query = query.join(Season, Match.season_id == Season.id).join(
            Competition, Season.competition_id == Competition.id
        ).filter(Competition.name == league)

    if status:
        query = query.filter(Match.status == status)

    if date_from:
        try:
            query = query.filter(Match.date >= datetime.fromisoformat(date_from))
        except ValueError:
            pass

    if date_to:
        try:
            query = query.filter(Match.date <= datetime.fromisoformat(date_to))
        except ValueError:
            pass

    if team:
        query = query.join(Team, (Match.home_team_id == Team.id) | (Match.away_team_id == Team.id)).filter(
            Team.name.ilike(f"%{team}%")
        )

    matches = query.order_by(Match.date.desc()).limit(limit).all()
    return [_match_to_response(m) for m in matches]


@router.get("/{match_id}", response_model=MatchResponse)
async def get_match(match_id: str, db: Session = Depends(get_session)) -> MatchResponse:
    match = db.query(Match).filter(Match.sportsdata_id == match_id).first()
    if not match:
        match = db.query(Match).filter(Match.id == match_id).first()
    if not match:
        raise HTTPException(404, "Maç bulunamadı")
    return _match_to_response(match)


@router.get("/team/{team_id}", response_model=TeamResponse)
async def get_team(team_id: str, db: Session = Depends(get_session)) -> TeamResponse:
    team = db.query(Team).filter(Team.sportsdata_id == team_id).first()
    if not team:
        team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(404, "Takım bulunamadı")
    return TeamResponse(
        id=str(team.id), name=team.name, country=team.country,
        elo=team.elo or 1500, attack=team.attack or 50, defense=team.defense or 50,
        form_last3=team.form_last3, form_last5=team.form_last5, form_last10=team.form_last10,
        home_power=team.home_power, away_power=team.away_power,
    )
