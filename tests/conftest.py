"""
Shared test fixtures — v2.

Uses StaticPool so all SQLAlchemy connections share one DBAPI connection,
which is required for in-memory SQLite (each new connection otherwise gets
a fresh empty database).

The `client` fixture:
  - Creates a fresh in-memory DB per test.
  - Overrides get_db to use the test engine.
  - Patches src.main.init_db to a no-op (prevents lifespan from touching real DB).
  - Cleans up after each test.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch

from src.database import Base, get_db
from src.main import app


@pytest.fixture()
def client():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    import src.models  # noqa: F401
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with patch("src.main.init_db"):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()
