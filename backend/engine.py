"""Database engine and session factory.

Reads DATABASE_URL from .env. Falls back to sqlite:///./blackbox.db if unset.
Creates all tables on startup via create_db_and_tables().
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine


def _load_env(path: Path = Path(".env")) -> None:
    """Minimal .env loader — no external dependency needed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_env()

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./blackbox.db")

# SQLite needs check_same_thread=False for FastAPI's async context.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)


def create_db_and_tables() -> None:
    """Create all SQLModel tables. Safe to call repeatedly — skips existing tables."""
    # Import db module to register all table models with SQLModel.metadata
    import backend.db  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session():
    """Dependency for FastAPI: yields a DB session per request."""
    with Session(engine) as session:
        yield session
