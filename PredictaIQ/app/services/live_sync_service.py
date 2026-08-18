"""Canlı maç + oran verisini dış API'lerden çekip veritabanına yazan servis.

Bu, mimarideki eksik halkaydı: `sportsdata_client.py` / `odds_api_client.py`
sadece dış API'den veri ÇEKİYORDU ama hiçbir router bunu `Match` /
`OddsSnapshot` tablolarına YAZMIYORDU. GitHub Actions'ın 15 dakikada bir
tetiklediği `/api/v1/admin/sync/live` endpoint'i bu servisi çağırır.

ÖNEMLİ NOT (dürüstçe belirtilmeli): SportsData.io soccer API'sinin ham JSON
alan adları (GameId/HomeTeamId vb.) bu ortamda ağ erişimi olmadığı için
resmi dokümantasyondan doğrulanamadı. Aşağıdaki `_pick()` yardımcı
fonksiyonu, yaygın SportsData.io soccer şeması konvansiyonlarına göre
birkaç olası alan adını dener. İlk canlı çalıştırmada `raw_sample` log
satırını kontrol edip gerekirse `_FIELD_CANDIDATES` sözlüğünü gerçek API
yanıtına göre güncellemen gerekebilir.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models import Competition, Match, OddsSnapshot, Season, Team
from app.services.odds_api_client import OddsAPIClient
from app.services.player_service import PlayerImpactService
from app.services.prediction_service import PredictionService
from app.services.sportsdata_client import SportsDataClient
from app.services.team_rating_service import TeamRatingService

logger = logging.getLogger(__name__)

_STATUS_MAP = {
    "scheduled": "scheduled", "notstarted": "scheduled", "pre": "scheduled",
    "inprogress": "live", "live": "live", "1h": "live", "2h": "live", "ht": "live",
    "final": "finished", "finished": "finished", "ft": "finished", "aet": "finished",
    "postponed": "postponed", "canceled": "cancelled", "cancelled": "cancelled",
}

_FIELD_CANDIDATES = {
    "game_id": ["GameId", "FixtureId", "Id", "GameID"],
    "home_team_id": ["HomeTeamId", "HomeTeamID"],
    "away_team_id": ["AwayTeamId", "AwayTeamID"],
    "home_team_name": ["HomeTeamName", "HomeTeam"],
    "away_team_name": ["AwayTeamName", "AwayTeam"],
    "date": ["DateTime", "Day", "Date"],
    "status": ["Status"],
    "home_score": ["HomeTeamScore", "HomeScore"],
    "away_score": ["AwayTeamScore", "AwayScore"],
    "venue": ["Stadium", "Venue"],
    "round": ["Week", "Round"],
}


def _pick(raw: Dict[str, Any], key: str) -> Any:
    for candidate in _FIELD_CANDIDATES.get(key, []):
        if candidate in raw and raw[candidate] is not None:
            return raw[candidate]
    return None


def _normalize_status(raw_status: Optional[str]) -> str:
    if not raw_status:
        return "scheduled"
    return _STATUS_MAP.get(str(raw_status).strip().lower(), "scheduled")


def _parse_date(raw_date: Any) -> Optional[datetime]:
    if raw_date is None:
        return None
    if isinstance(raw_date, datetime):
        return raw_date
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(raw_date)[: len(fmt) + 2], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Tarih parse edilemedi: %r", raw_date)
        return None


class LiveSyncService:
    def __init__(self):
        self.sportsdata = SportsDataClient()
        self.odds_api = OddsAPIClient()

    def _get_or_create_team(self, db: Session, sportsdata_id: Optional[str], name: Optional[str]) -> Optional[Team]:
        if not name:
            return None
        team = None
        if sportsdata_id:
            team = db.query(Team).filter(Team.sportsdata_id == str(sportsdata_id)).first()
        if not team:
            team = db.query(Team).filter(Team.name == name).first()
        if not team:
            team = Team(sportsdata_id=str(sportsdata_id) if sportsdata_id else None, name=name)
            db.add(team)
            db.flush()
        elif sportsdata_id and not team.sportsdata_id:
            team.sportsdata_id = str(sportsdata_id)
        return team

    def _get_or_create_season(self, db: Session, league: str) -> Season:
        competition = db.query(Competition).filter(Competition.name == league).first()
        if not competition:
            competition = Competition(name=league, level=1)
            db.add(competition)
            db.flush()
        season = (
            db.query(Season)
            .filter(Season.competition_id == competition.id, Season.current == True)
            .first()
        )
        if not season:
            season = Season(competition_id=competition.id, name="current", current=True)
            db.add(season)
            db.flush()
        return season

    def _upsert_match(self, db: Session, raw: Dict[str, Any], league: str) -> Optional[Match]:
        game_id = _pick(raw, "game_id")
        if game_id is None:
            logger.warning("sportsdata_id'siz fixture atlandı: %r", raw)
            return None
        game_id = str(game_id)

        home_team = self._get_or_create_team(db, _pick(raw, "home_team_id"), _pick(raw, "home_team_name"))
        away_team = self._get_or_create_team(db, _pick(raw, "away_team_id"), _pick(raw, "away_team_name"))
        if not home_team or not away_team:
            logger.warning("Takım bilgisi eksik, fixture %s atlandı", game_id)
            return None

        season = self._get_or_create_season(db, league)

        match = db.query(Match).filter(Match.sportsdata_id == game_id).first()
        if not match:
            match = Match(sportsdata_id=game_id, season_id=season.id)
            db.add(match)

        parsed_date = _parse_date(_pick(raw, "date"))
        if parsed_date:
            match.date = parsed_date
        elif match.date is None:
            match.date = datetime.utcnow()

        match.home_team_id = home_team.id
        match.away_team_id = away_team.id
        match.status = _normalize_status(_pick(raw, "status"))
        match.venue = _pick(raw, "venue")
        match.round = str(_pick(raw, "round")) if _pick(raw, "round") is not None else match.round

        home_score = _pick(raw, "home_score")
        away_score = _pick(raw, "away_score")
        if home_score is not None:
            match.home_score = int(home_score)
        if away_score is not None:
            match.away_score = int(away_score)

        db.flush()
        return match

    async def _sync_odds_for_match(self, db: Session, match: Match) -> None:
        if not match.sportsdata_id:
            return
        try:
            odds_rows = await self.sportsdata.get_odds(match.sportsdata_id)
        except Exception as exc:
            logger.warning("Oran çekilemedi (fixture %s): %s", match.sportsdata_id, exc)
            return

        for row in odds_rows or []:
            home_odds = row.get("HomeMoneyLine") or row.get("HomeOdds") or row.get("Home")
            draw_odds = row.get("DrawMoneyLine") or row.get("DrawOdds") or row.get("Draw")
            away_odds = row.get("AwayMoneyLine") or row.get("AwayOdds") or row.get("Away")
            bookmaker = row.get("Sportsbook") or row.get("Bookmaker") or "sportsdata"
            if home_odds is None or draw_odds is None or away_odds is None:
                continue
            snapshot = OddsSnapshot(
                match_id=match.id,
                bookmaker=str(bookmaker),
                source="sportsdata",
                home_odds=float(home_odds),
                draw_odds=float(draw_odds),
                away_odds=float(away_odds),
            )
            db.add(snapshot)

            # Opening odds boşsa ilk snapshot'ı opening, en güncelini closing say.
            if match.opening_home_odds is None:
                match.opening_home_odds = float(home_odds)
                match.opening_draw_odds = float(draw_odds)
                match.opening_away_odds = float(away_odds)
            match.closing_home_odds = float(home_odds)
            match.closing_draw_odds = float(draw_odds)
            match.closing_away_odds = float(away_odds)

    async def sync_live(self, db: Session, prediction_service: PredictionService,
                         leagues: List[str]) -> Dict[str, Any]:
        """Her lig için canlı/günün maçlarını çeker, DB'ye yazar, rating'leri
        günceller ve scheduled/live maçlar için tahminleri tazeler."""
        summary: Dict[str, Any] = {"leagues": {}, "matches_upserted": 0, "predictions_updated": 0}
        touched_matches: List[Match] = []

        for league in leagues:
            try:
                raw_fixtures = await self.sportsdata.get_live_scores(league)
            except Exception as exc:
                logger.error("Lig %s için canlı veri çekilemedi: %s", league, exc)
                summary["leagues"][league] = {"status": "error", "detail": str(exc)}
                continue

            n_upserted = 0
            for raw in raw_fixtures or []:
                match = self._upsert_match(db, raw, league)
                if match:
                    n_upserted += 1
                    touched_matches.append(match)
            db.commit()
            summary["leagues"][league] = {"status": "ok", "n_fixtures": n_upserted}
            summary["matches_upserted"] += n_upserted

        # Oranları senkronize et (opsiyonel -- sportsdata odds endpoint'i
        # kotanı zorlayabilir, sadece scheduled/live maçlar için çekiyoruz)
        for match in touched_matches:
            if match.status in ("scheduled", "live"):
                await self._sync_odds_for_match(db, match)
        db.commit()

        # Rating'leri güncelle (yeni bitmiş maçlar varsa Elo/form değişmiş olabilir)
        try:
            TeamRatingService().recompute_all_ratings(db)
            db.commit()
        except Exception as exc:
            logger.warning("Rating recompute başarısız (yoksayılıyor): %s", exc)
            db.rollback()

        # scheduled/live maçlar için tahminleri tazele
        player_impact_service = PlayerImpactService()
        pending = [m for m in touched_matches if m.status in ("scheduled", "live")]
        for match in pending:
            try:
                db.refresh(match)
                home_injury = player_impact_service.team_injury_impact(db, match.home_team_id) if match.home_team_id else {"elo_penalty": 0.0}
                away_injury = player_impact_service.team_injury_impact(db, match.away_team_id) if match.away_team_id else {"elo_penalty": 0.0}
                prediction = prediction_service.predict_match(
                    match,
                    home_injury_impact=home_injury.get("elo_penalty", 0.0),
                    away_injury_impact=away_injury.get("elo_penalty", 0.0),
                )
                match.model_home_prob = prediction["model_probability"]["home"]
                match.model_draw_prob = prediction["model_probability"]["draw"]
                match.model_away_prob = prediction["model_probability"]["away"]
                match.model_confidence = prediction["confidence"]
                summary["predictions_updated"] += 1
            except Exception as exc:
                logger.warning("Tahmin güncellenemedi (match %s): %s", match.sportsdata_id, exc)
        db.commit()

        return summary
