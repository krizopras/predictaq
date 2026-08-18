"""Football-Data.co.uk CSV yükleyici.

Plan madde 9, 12, 24: API anahtarı gerektirmeyen, uzun tarihsel
(bazı liglerde 1990'lara kadar giden) sonuç + istatistik + bookmaker
odds veri setinin bootstrap kaynağı. Model eğitimi için "soğuk
başlangıç" verisi sağlar -- yeni kurulan bir PredictaIQ, gerçek zamanlı
veri birikmesini beklemeden bu veriyle similarity/ML modellerini
eğitebilir.

CSV şeması (Football-Data.co.uk standart formatı):
Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR,
B365H,B365D,B365A, ... (opening/closing/max/avg oran sütunları sağlayıcıya göre değişir)
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

# Football-Data.co.uk lig kodları (yaygın örnekler)
LEAGUE_CODES = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "T1": "Super Lig",
}

ODDS_COLUMN_CANDIDATES = {
    "home": ["B365H", "AvgH", "PSH", "BbAvH"],
    "draw": ["B365D", "AvgD", "PSD", "BbAvD"],
    "away": ["B365A", "AvgA", "PSA", "BbAvA"],
}


class FootballDataClient:
    def __init__(self):
        self.base_url = settings.football_data_base_url
        self.timeout = 30.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def fetch_season_csv(self, season_code: str, league_code: str = "E0") -> pd.DataFrame:
        """season_code örn. '2324' (2023-24 sezonu). Dönen değer pandas
        DataFrame; hiçbir sonuç bulunamazsa boş DataFrame döner."""
        url = f"{self.base_url}/{season_code}/{league_code}.csv"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            if response.status_code != 200 or not response.content:
                return pd.DataFrame()
            try:
                return pd.read_csv(io.BytesIO(response.content), encoding="latin1")
            except Exception:
                return pd.DataFrame()

    @staticmethod
    def _first_available(row: pd.Series, candidates: List[str]) -> Optional[float]:
        for col in candidates:
            if col in row and pd.notna(row[col]):
                try:
                    return float(row[col])
                except (TypeError, ValueError):
                    continue
        return None

    def parse_matches(self, df: pd.DataFrame) -> List[Dict]:
        """DataFrame'i PredictaIQ'nun Match modeline uygun sözlük listesine
        çevirir (henüz DB'ye yazmaz -- init_db/train_models script'i bu
        sözlükleri kullanarak Team/Match kayıtlarını oluşturur)."""
        if df.empty:
            return []

        records = []
        for _, row in df.iterrows():
            if pd.isna(row.get("HomeTeam")) or pd.isna(row.get("AwayTeam")):
                continue
            try:
                date = pd.to_datetime(row.get("Date"), dayfirst=True).to_pydatetime()
            except Exception:
                continue

            records.append({
                "home_team": str(row["HomeTeam"]).strip(),
                "away_team": str(row["AwayTeam"]).strip(),
                "date": date,
                "home_score": int(row["FTHG"]) if pd.notna(row.get("FTHG")) else None,
                "away_score": int(row["FTAG"]) if pd.notna(row.get("FTAG")) else None,
                "home_shots": int(row["HS"]) if pd.notna(row.get("HS")) else None,
                "away_shots": int(row["AS"]) if pd.notna(row.get("AS")) else None,
                "home_shots_on_target": int(row["HST"]) if pd.notna(row.get("HST")) else None,
                "away_shots_on_target": int(row["AST"]) if pd.notna(row.get("AST")) else None,
                "home_corners": int(row["HC"]) if pd.notna(row.get("HC")) else None,
                "away_corners": int(row["AC"]) if pd.notna(row.get("AC")) else None,
                "closing_home_odds": self._first_available(row, ODDS_COLUMN_CANDIDATES["home"]),
                "closing_draw_odds": self._first_available(row, ODDS_COLUMN_CANDIDATES["draw"]),
                "closing_away_odds": self._first_available(row, ODDS_COLUMN_CANDIDATES["away"]),
                "status": "finished",
            })
        return records
