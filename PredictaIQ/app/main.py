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
    db = SessionLocal()
    try:
        result = model_persistence.load_all(prediction_service, settings.model_path, db)
        logger.info("Model yukleme: %s", result)
    except Exception as exc:
        logger.warning("Model yukleme sirasinda hata (yoksayiliyor): %s", exc)
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    logger.info("Starting up PredictaIQ...")

    # Veritabanı tablolarını güvenli ve hızlı şekilde bağla
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully.")
    except Exception as exc:
        logger.error("Database initialization failed during startup: %s", exc)

    if not settings.admin_api_key and not settings.debug:
        logger.warning(
            "ADMIN_API_KEY tanimli degil -- /api/v1/admin/* endpoint'leri "
            "(models/train, sync/live dahil) auth'suz, herkese acik durumda. "
            "Production'da mutlaka bir ADMIN_API_KEY set edin."
        )

    try:
        _load_models_from_disk()
        logger.info("Models load sequence completed.")
    except Exception as exc:
        logger.warning("Skipping model load on startup: %s", exc)

    yield

    logger.info("Shutting down PredictaIQ...")


# redirect_slashes=True ile /matches ve /matches/ aramalarının ikisi de 200 OK döner.
app = FastAPI(
    title="PredictaIQ",
    version="2.0.0",
    description="Advanced Football Prediction Engine",
    lifespan=lifespan,
    redirect_slashes=True,
)

# CORS -- Her ihtimale karşı listeyi ve varsayılan wildcard (*) garantisini sağlıyoruz.
origins = settings.cors_allow_origins
if isinstance(origins, str):
    origins = [origin.strip() for origin in origins.split(",")]

if not origins or origins == [""]:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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


# Healthcheck Railway için anında 200 OK dönecek şekilde ultra hafifletildi
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
