import httpx
from typing import Dict, List, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings
import asyncio

class SportsDataClient:
    def __init__(self):
        self.api_key = settings.sportsdata_api_key
        self.base_url = settings.sportsdata_base_url
        self.timeout = 30.0
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Sportsdata.io API'sine istek atar"""
        params = params or {}
        params["key"] = self.api_key
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
    
    async def get_teams(self, league: str) -> List[Dict]:
        """Tüm takımları getir"""
        endpoint = f"/soccer/teams/{league}"
        return await self._make_request(endpoint)
    
    async def get_team_details(self, team_id: str) -> Dict:
        """Takım detaylarını getir"""
        endpoint = f"/soccer/team/{team_id}"
        return await self._make_request(endpoint)
    
    async def get_fixtures(self, league: str, season: str, offset: int = 0, limit: Optional[int] = None) -> List[Dict]:
        """Maç fikstürünü getir (opsiyonel sayfalama parametreleriyle)"""
        endpoint = f"/soccer/fixtures/{league}/{season}"
        params = {}
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        return await self._make_request(endpoint, params)
    
    async def get_fixture_details(self, fixture_id: str) -> Dict:
        """Maç detaylarını getir"""
        endpoint = f"/soccer/fixture/{fixture_id}"
        return await self._make_request(endpoint)
    
    async def get_live_scores(self, league: str) -> List[Dict]:
        """Canlı skorları getir"""
        endpoint = f"/soccer/scores/{league}"
        return await self._make_request(endpoint)
    
    async def get_team_stats(self, team_id: str, season: str) -> Dict:
        """Takım istatistiklerini getir"""
        endpoint = f"/soccer/stats/{team_id}/{season}"
        return await self._make_request(endpoint)
    
    async def get_match_stats(self, fixture_id: str) -> Dict:
        """Maç istatistiklerini getir"""
        endpoint = f"/soccer/matchstats/{fixture_id}"
        return await self._make_request(endpoint)
    
    async def get_odds(self, fixture_id: str, bookmaker: Optional[str] = None) -> List[Dict]:
        """Maç oranlarını getir"""
        endpoint = f"/soccer/odds/{fixture_id}"
        return await self._make_request(endpoint)
    
    async def get_historical_odds(self, fixture_id: str) -> List[Dict]:
        """Tarihsel oran snapshotlarını getir"""
        endpoint = f"/soccer/odds/historical/{fixture_id}"
        return await self._make_request(endpoint)
    
    async def get_players(self, team_id: str) -> List[Dict]:
        """Takım oyuncularını getir"""
        endpoint = f"/soccer/players/{team_id}"
        return await self._make_request(endpoint)
    
    async def get_injuries(self, league: str) -> List[Dict]:
        """Sakatlık listesini getir"""
        endpoint = f"/soccer/injuries/{league}"
        return await self._make_request(endpoint)
    
    async def get_standings(self, league: str, season: str) -> List[Dict]:
        """Puan durumunu getir"""
        endpoint = f"/soccer/standings/{league}/{season}"
        return await self._make_request(endpoint)
    
    async def get_head_to_head(self, team1: str, team2: str) -> List[Dict]:
        """Takımlar arası geçmiş maçları getir"""
        endpoint = f"/soccer/h2h/{team1}/{team2}"
        return await self._make_request(endpoint)
    
    async def batch_get_fixtures(self, league: str, season: str, batch_size: int = 50,
                                  max_batches: int = 200) -> List[Dict]:
        """Büyük verileri batch halinde getir.

        NOT: Önceki sürüm `offset`'i hiçbir zaman API'ye GÖNDERMİYORDU --
        yani her iterasyonda TAM OLARAK AYNI sayfa tekrar isteniyordu.
        Sonuç, sunucu hep aynı (dolu) veriyi döndürdüğü sürece sonsuz
        döngüye giren ve API'yi gereksiz yere spam'leyen bir kod. Artık
        `offset` her turda ilerletiliyor, ayrıca sonsuz döngüyü kesin
        olarak önlemek için bir `max_batches` üst sınırı var.
        """
        all_fixtures: List[Dict] = []
        offset = 0

        for _ in range(max_batches):
            fixtures = await self.get_fixtures(league, season, offset=offset, limit=batch_size)
            if not fixtures:
                break
            all_fixtures.extend(fixtures)
            offset += len(fixtures)

            if len(fixtures) < batch_size:
                break
        else:
            import logging
            logging.getLogger(__name__).warning(
                "batch_get_fixtures: max_batches (%d) sınırına ulaşıldı, veri eksik olabilir", max_batches
            )

        return all_fixtures