import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine, SessionLocal
from app.dependencies import prediction_service
from app.routers import admin, historical, matches, odds, predictions
from app.services import model_persistence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_models_from_disk() -> None:
    """Acilista modelleri EGITMEZ -- sadece settings.model_path'ten (bir
    GitHub Actions/manuel models/train cagrisinin onceden diske yazdigi)
    kaydedilmis ML/similarity/calibration modellerini ve DB'deki aktif
    ensemble agirliklarini YUKLER.

    Bu degisiklik bilincli: eski _bootstrap_models() her process
    baslangicinda (her deploy, her restart, her yeni instance) rating
    recompute + similarity/ML egitimini best-effort calistiriyordu. Bu,
    production'da ongorulemeyen, tekrarlayan CPU harcamasi ve yavas
    acilis demekti. Artik egitim SADECE POST /api/v1/admin/models/train
    ile (GitHub Actions cron veya elle) tetikleniyor; process acilisi
    her zaman hizli ve deterministik.

    Diskte/DB'de kayitli bir sey yoksa (ilk kurulum) hata FIRLATMAZ --
    uygulama bos modellerle acilir, tahminler diger modellere (Poisson/
    Elo/xG/market) duser, /admin/models/train cagrilinca tamamlanir.
    """
    db = SessionLocal()
    try:
        result = model_persistence.load_all(prediction_service, settings.model_path, db)
        logger.info("Model yukleme: %s", result)
    except Exception as exc:  # pragma: no cover
        logger.warning("Model yukleme sirasinda hata (yoksayiliyor): %s", exc)
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    logger.info("Starting up PredictaIQ...")

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")

    if not settings.admin_api_key and not settings.debug:
        logger.warning(
            "ADMIN_API_KEY tanimli degil -- /api/v1/admin/* endpoint'leri "
            "(models/train, sync/live dahil) auth'suz, herkese acik durumda. "
            "Production'da mutlaka bir ADMIN_API_KEY set edin."
        )

    _load_models_from_disk()
    logger.info("Models loaded")

    yield

    logger.info("Shutting down PredictaIQ...")


app = FastAPI(
    title="PredictaIQ",
    version="2.0.0",
    description="Advanced Football Prediction Engine",
    lifespan=lifespan,
)

# CORS -- eski kodda allow_origins=["*"] + allow_credentials=True birlikte
# kullaniliyordu, ki bu tarayicilarda gecersiz/guvensiz bir kombinasyondur
# (CORS spesifikasyonu credentials ile wildcard origin'e izin vermez).
# Origin listesi artik .env uzerinden (CORS_ALLOW_ORIGINS) yapilandiriliyor.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(matches.router, prefix="/api/v1/matches", tags=["matches"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(historical.router, prefix="/api/v1/historical", tags=["historical"])
app.include_router(odds.router, prefix="/api/v1/odds", tags=["odds"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])


@app.get("/")
async def root():
    return {
        "service": "PredictaIQ",
        "version": "2.0.0",
        "status": "active",
        "endpoints": [
            "/api/v1/matches",
            "/api/v1/predictions",
            "/api/v1/historical",
            "/api/v1/odds",
            "/api/v1/admin",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
