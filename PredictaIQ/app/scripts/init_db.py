#!/usr/bin/env python
"""Veritabanını başlatır ve Football-Data.co.uk'tan tarihsel veri yükler.

Eski sürüm sadece SportsDataClient.get_teams() çağırıp sonucu print
ediyordu, hiçbir şeyi veritabanına yazmıyordu -- bu script artık
gerçekten Competition/Season/Team/Match satırları oluşturuyor ki
similarity/ML modelleri sıfırdan eğitilebilsin (plan madde 9, 26:
"ilk sürüm için Football-Data ile başlayalım").

Kullanım:
    python -m app.scripts.init_db --league E0 --seasons 2223 2324
"""
import argparse
import asyncio
import logging

from app.database import Base, SessionLocal, engine
from app.models import Competition, Match, Season, Team
from app.services.football_data_client import LEAGUE_CODES, FootballDataClient
from app.services.team_rating_service import TeamRatingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_league_seasons(league_code: str, season_codes: list[str]) -> int:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    client = FootballDataClient()

    total_inserted = 0
    try:
        competition = db.query(Competition).filter(Competition.name == LEAGUE_CODES.get(league_code, league_code)).first()
        if not competition:
            competition = Competition(name=LEAGUE_CODES.get(league_code, league_code), country="", level=1)
            db.add(competition)
            db.flush()

        team_cache: dict[str, Team] = {t.name: t for t in db.query(Team).all()}

        def get_or_create_team(name: str) -> Team:
            if name in team_cache:
                return team_cache[name]
            team = Team(name=name)
            db.add(team)
            db.flush()
            team_cache[name] = team
            return team

        for season_code in season_codes:
            season = db.query(Season).filter(
                Season.competition_id == competition.id, Season.name == season_code
            ).first()
            if not season:
                season = Season(competition_id=competition.id, name=season_code)
                db.add(season)
                db.flush()

            df = await client.fetch_season_csv(season_code, league_code)
            records = client.parse_matches(df)
            logger.info("%s %s: %d maç bulundu", league_code, season_code, len(records))

            for rec in records:
                home = get_or_create_team(rec["home_team"])
                away = get_or_create_team(rec["away_team"])
                exists = db.query(Match).filter(
                    Match.season_id == season.id,
                    Match.home_team_id == home.id,
                    Match.away_team_id == away.id,
                    Match.date == rec["date"],
                ).first()
                if exists:
                    continue
                match = Match(
                    season_id=season.id, home_team_id=home.id, away_team_id=away.id,
                    date=rec["date"], status=rec["status"],
                    home_score=rec["home_score"], away_score=rec["away_score"],
                    home_shots=rec["home_shots"], away_shots=rec["away_shots"],
                    home_shots_on_target=rec["home_shots_on_target"],
                    away_shots_on_target=rec["away_shots_on_target"],
                    home_corners=rec["home_corners"], away_corners=rec["away_corners"],
                    closing_home_odds=rec["closing_home_odds"],
                    closing_draw_odds=rec["closing_draw_odds"],
                    closing_away_odds=rec["closing_away_odds"],
                )
                db.add(match)
                total_inserted += 1

            db.commit()

        logger.info("Toplam %d yeni maç eklendi. Rating'ler hesaplanıyor...", total_inserted)
        rating_service = TeamRatingService()
        rating_service.recompute_all_ratings(db)
        db.commit()
    finally:
        db.close()

    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="PredictaIQ veritabanını Football-Data.co.uk ile başlat")
    parser.add_argument("--league", default="E0", help="Football-Data lig kodu (örn. E0=Premier League)")
    parser.add_argument("--seasons", nargs="+", default=["2223", "2324"], help="Sezon kodları (örn. 2324)")
    args = parser.parse_args()

    asyncio.run(load_league_seasons(args.league, args.seasons))


if __name__ == "__main__":
    main()
