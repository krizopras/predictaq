from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Sportsdata.io (ana veri kaynağı: takım/oyuncu/maç/istatistik) ---
    sportsdata_api_key: str
    sportsdata_base_url: str = "https://api.sportsdata.io/v3"

    # --- The Odds API (ikincil kaynak: canlı + tarihsel bookmaker odds) ---
    # Plan madde 7, 12, 24: tek sağlayıcıya bağımlı kalınmaması gerektiği
    # için ikinci bir odds kaynağı olarak eklendi. Anahtar verilmezse bu
    # kaynak sessizce devre dışı kalır (opsiyonel).
    odds_api_key: Optional[str] = None
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    # --- Football-Data.co.uk (üçüncü kaynak: uzun tarihsel CSV veri seti) ---
    # Plan madde 9: model eğitimi / "aynı oranlarda ne oldu" analizleri için
    # başlangıç veri tabanı. API key gerektirmez, doğrudan CSV indirilir.
    football_data_base_url: str = "https://www.football-data.co.uk/mmz4281"

    # --- Sportmonks (opsiyonel dördüncü kaynak) ---
    sportmonks_api_key: Optional[str] = None
    sportmonks_base_url: str = "https://api.sportmonks.com/v3/football"

    # Database
    database_url: str
    test_database_url: Optional[str] = None

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Model / eğitim
    model_path: str = "./models"
    historical_data_path: str = "./data/historical_matches.parquet"
    min_matches_to_train_similarity: int = 30
    min_matches_to_train_ml: int = 150
    min_predictions_to_calibrate: int = 50
    similarity_neighbors: int = 50
    elo_time_decay_half_life_days: int = 180  # "6 ay önceki maç daha az önemli"

    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    # /api/v1/admin/* endpoint'lerini korumak için paylaşımlı anahtar
    # (GitHub Actions cron'u bunu X-Admin-Api-Key header'ında gönderir).
    admin_api_key: str = ""

    # Canlı veri senkronizasyonunda (`/admin/sync/live`) taranacak ligler.
    # SportsData.io lig kodları, virgülle ayrılmış (örn. "EPL,ESP1,ITSA").
    sync_leagues: str = "EPL"

    @property
    def sync_leagues_list(self) -> List[str]:
        return [l.strip() for l in self.sync_leagues.split(",") if l.strip()]

    # API
    api_version: str = "v1"
    debug: bool = False
    # NOT: pydantic-settings, List[str] alanları için env değerini önce JSON
    # olarak parse etmeye çalışır ve düz virgüllü string'lerde patlar. Bunu
    # önlemek için ham değeri str olarak alıp bir property ile listeye
    # çeviriyoruz.
    cors_allow_origins_raw: str = Field(default="http://localhost:3000", alias="CORS_ALLOW_ORIGINS")

    @property
    def cors_allow_origins(self) -> List[str]:
        return [o.strip() for o in self.cors_allow_origins_raw.split(",") if o.strip()]

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "populate_by_name": True,
        "protected_namespaces": (),
        "extra": "ignore",
    }


settings = Settings()
