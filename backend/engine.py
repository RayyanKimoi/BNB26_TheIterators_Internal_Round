"""Database engine and session factory.

Reads DATABASE_URL from .env. Falls back to sqlite:///./blackbox.db if unset.
Creates all tables on startup via create_db_and_tables().
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import inspect, text
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
    """Create all SQLModel tables, then add any columns missing from them.

    Safe to call repeatedly.
    """
    # Import db module to register all table models with SQLModel.metadata
    import backend.db  # noqa: F401

    SQLModel.metadata.create_all(engine)
    ensure_schema()


def ensure_schema() -> list[str]:
    """Add columns present in the models but missing from existing tables.

    `create_all` creates missing tables but never alters existing ones, so a
    database created before a model gained a column stays stale and every
    query against that column fails at runtime. This walks the metadata and
    issues `ALTER TABLE ... ADD COLUMN` for anything absent.

    Deliberately not Alembic: CLAUDE.md bars frameworks the PRD does not call
    for, and the migrations this project needs are additive nullable columns.
    If a migration ever needs to drop a column, rename one, or backfill data,
    that is the point to reach for a real migration tool.

    Returns the list of applied statements, which is empty on a current
    database. Idempotent, and works on both SQLite and Postgres.
    """
    import backend.db  # noqa: F401

    applied: list[str] = []
    inspector = inspect(engine)
    for table in SQLModel.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing:
                continue
            if not column.nullable and column.default is None:
                # Cannot add a NOT NULL column to a populated table without a
                # default. Surface it rather than failing cryptically later.
                raise RuntimeError(
                    f"cannot auto-add non-nullable column {table.name}.{column.name}; "
                    f"give it a default or write a real migration"
                )
            ddl_type = column.type.compile(engine.dialect)
            statement = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}'
            with engine.begin() as conn:
                conn.execute(text(statement))
            applied.append(statement)
    return applied


def get_session():
    """Dependency for FastAPI: yields a DB session per request."""
    with Session(engine) as session:
        yield session
