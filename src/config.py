"""
Application configuration loaded from environment variables via python-dotenv.

All secrets (JWT_SECRET, GEMINI_API_KEY) are read from .env — never hard-coded.
Why dotenv? Simple, standard, keeps secrets out of source control.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level above src/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


class Settings:
    # ── Database ──────────────────────────────────────────────────────────────
    # SQLite file path.  The default places app.db in the project root.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

    # ── Gemini ────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    # gemini-1.5-flash: fast, cheap, good enough for structured support decisions
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    # text-embedding-004: Google's latest general-purpose embedding model
    GEMINI_EMBEDDING_MODEL: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"
    )

    # ── Knowledge base ────────────────────────────────────────────────────────
    # Path to the HR-provided .md policy files.
    # Resolved relative to this file so it works from any working directory.
    # New knowledge_base/ lives inside the project root alongside src/
    # __file__ is G:\mansor_lab\ticket_assistant\src\config.py
    # .parent       → ticket_assistant/src
    # .parent.parent → ticket_assistant  (project root, where knowledge_base/ lives)
    KB_DIR: Path = Path(
        os.getenv(
            "KB_DIR",
            str(
                Path(__file__).resolve().parent.parent / "knowledge_base"
            ),
        )
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # Where ingested chunk embeddings are stored (a single JSON file, no vector DB).
    # Why JSON? Simple, portable, easy to inspect, no extra dependencies.
    EMBEDDINGS_PATH: Path = Path(
        os.getenv(
            "EMBEDDINGS_PATH",
            str(
                Path(__file__).resolve().parent.parent / "data" / "embeddings.json"
            ),
        )
    )
    # Top-k chunks retrieved per query.  4 is a good balance: enough context
    # without overloading the prompt.
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    # Max words per chunk.  ~300 words ≈ 400 tokens — keeps chunks meaningful
    # but small enough that similarity search stays accurate.
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "300"))


settings = Settings()
