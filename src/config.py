import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    # =========================
    # Database
    # =========================
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./support_assistant.db",
    )

    # =========================
    # JWT Authentication
    # =========================
    JWT_SECRET: str = os.getenv(
        "JWT_SECRET",
        "change-me-in-production",
    )

    JWT_ALGORITHM: str = os.getenv(
        "JWT_ALGORITHM",
        "HS256",
    )

    JWT_EXPIRE_MINUTES: int = int(
        os.getenv("JWT_EXPIRE_MINUTES", "60")
    )

    # =========================
    # Gemini
    # =========================
    GEMINI_API_KEY: str = os.getenv(
        "GEMINI_API_KEY",
        "",
    )

    GEMINI_MODEL: str = os.getenv(
        "GEMINI_MODEL",
        "gemini-1.5-flash",
    )

    GEMINI_EMBEDDING_MODEL: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "models/gemini-embedding-001",
    )

    # =========================
    # Knowledge Base / RAG
    # =========================
    KB_DIR: Path = Path(
        os.getenv(
            "KB_DIR",
            str(BASE_DIR / "knowledge_base"),
        )
    )

    EMBEDDINGS_PATH: Path = Path(
        os.getenv(
            "EMBEDDINGS_PATH",
            str(BASE_DIR / "data" / "embeddings.json"),
        )
    )

    # Number of policy chunks retrieved for each ticket
    RETRIEVAL_TOP_K: int = int(
        os.getenv("RETRIEVAL_TOP_K", "4")
    )

    # Maximum words/tokens used when creating chunks
    CHUNK_SIZE: int = int(
        os.getenv("CHUNK_SIZE", "300")
    )


settings = Settings()