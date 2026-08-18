"""The Odds API istemcisi.

Plan madde 7, 12, 24: canlı ve tarihsel bookmaker oranları için ikinci,
bağımsız bir kaynak. Resmi dokümantasyona göre historical odds
featured marketlerde 2020'den itibaren, yakın dönemde 5 dakikalık
snapshot çözünürlüğünde sağlanıyor.

`settings.odds_api_key` boşsa bu istemci kullanılmaz; PredictaIQ tek
kaynağa (Sportsdata.io) bağımlı kalmaz ama API anahtarı yoksa zarif bir
şekilde devre dışı kalır.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings


class OddsAPIClient:
    def __init__(self):
        self.api_key = settings.odds_api_key
        self.base_url = settings.odds_api_base_url
        self.timeout = 30.0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _get(self, path: str, params: Optional[Dict] = None) -> object:
        if not self.is_configured:
            raise RuntimeError("ODDS_API_KEY tanımlı değil -- The Odds API devre dışı")
        params = dict(params or {})
        params["apiKey"] = self.api_key
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}{path}", params=params)
            response.raise_for_status()
            return response.json()

    async def list_sports(self) -> List[Dict]:
        return await self._get("/sports")

    async def get_live_odds(self, sport_key: str, regions: str = "eu", markets: str = "h2h") -> List[Dict]:
        """Canlı/yaklaşan maç oranları (madde 6: opening/current snapshot)."""
        return await self._get(f"/sports/{sport_key}/odds", {"regions": regions, "markets": markets})

    async def get_historical_odds(self, sport_key: str, date_iso: str, regions: str = "eu",
                                   markets: str = "h2h") -> Dict:
        """Belirli bir zaman noktasındaki (snapshot) tarihsel oranlar
        (madde 7: 5 dakikalık çözünürlüklü tarihsel snapshot'lar)."""
        return await self._get(
            f"/historical/sports/{sport_key}/odds",
            {"regions": regions, "markets": markets, "date": date_iso},
        )

    async def get_historical_events(self, sport_key: str, date_iso: str) -> Dict:
        return await self._get(f"/historical/sports/{sport_key}/events", {"date": date_iso})
