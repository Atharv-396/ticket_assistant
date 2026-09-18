"""
Database engine and session factory.

SQLAlchemy + SQLite.
- check_same_thread=False: required for SQLite with FastAPI's async request handling
  (multiple threads share the connection pool).
- Tables are created automatically via init_db() called at startup.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def init_db() -> None:
    """
    Create all tables that do not yet exist.  Safe to call repeatedly —
    SQLAlchemy's create_all is idempotent (it checks before creating).
    """
    import src.models  # noqa: F401 — side-effect: registers models with Base
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency that yields a database session per request and
    guarantees the session is closed when the request finishes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
