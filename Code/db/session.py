"""SQLAlchemy engine, session factory, and DB initializer."""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session

from Code.db.models import Base, User
from Code.db import fts as _fts


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(plain: str, hashed: str) -> bool:
    return _hash_password(plain) == hashed

# Load .env from project root before reading any env vars
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)


def _build_url() -> str:
    if url := os.getenv("DATABASE_URL"):
        return url
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db = os.getenv("MYSQL_DB", "resume_parser")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4"


engine = create_engine(
    _build_url(),
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _ensure_columns() -> None:
    """Add columns introduced after the table was first created (idempotent).

    Base.metadata.create_all() creates new tables but never alters existing
    ones, so new columns on a pre-existing table must be added explicitly.
    """
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "resume" not in tables:
        return

    resume_cols = {c["name"] for c in inspector.get_columns("resume")}
    if "lex_tokens_json" not in resume_cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE resume ADD COLUMN lex_tokens_json TEXT"))

    if "job_description" in tables:
        jd_cols = {c["name"] for c in inspector.get_columns("job_description")}
        if "positions" not in jd_cols:
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE job_description ADD COLUMN positions INT NOT NULL DEFAULT 1"
                ))

    if "match" in tables:
        match_cols = {c["name"] for c in inspector.get_columns("match")}
        if "status" not in match_cols:
            with engine.begin() as conn:
                conn.execute(text(
                    "ALTER TABLE `match` ADD COLUMN status VARCHAR(32) NULL"
                ))


def init_db() -> None:
    """Create all tables (idempotent), set up FTS index, and seed default user."""
    Base.metadata.create_all(engine)
    _ensure_columns()
    _fts.init_fts(engine)
    with SessionLocal() as session:
        if not session.query(User).first():
            default_user = User(
                username=os.getenv("APP_USERNAME", "admin"),
                password_hash=_hash_password(os.getenv("APP_PASSWORD", "admin123")),
            )
            session.add(default_user)
            session.commit()


@contextmanager
def get_db():
    """Yield a session; roll back on exception, always close."""
    session: Session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
