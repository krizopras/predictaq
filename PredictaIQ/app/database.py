from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool

from app.config import settings

_is_sqlite = settings.database_url.startswith("sqlite")

if _is_sqlite:
    # SQLite yalnızca test/yerel geliştirme için (Postgres'e özgü QueuePool
    # ayarları burada geçerli değil; connect_args ile thread paylaşımı
    # açılır ki pytest / bu depo içindeki smoke testleri tek dosyalık
    # in-memory DB ile çalışabilsin).
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_engine(
        settings.database_url,
        poolclass=QueuePool,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_session() -> Session:
    """FastAPI Depends() ile kullanılan generator-tabanlı session sağlayıcı."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Servis/script katmanında `with get_db() as db:` şeklinde kullanılan
    context-manager sürümü. Eski kodda hem router'lar hem servisler
    `get_db` adını FastAPI-tarzı `Depends` jeneratörü gibi kullanıyordu;
    bu isim çakışmasını önlemek için iki ayrı fonksiyon tanımlandı ve
    router'lar `get_session`, servis/scriptler `get_db` kullanıyor.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
