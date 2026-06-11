"""SQLAlchemy engine, session factory, and DB initializer."""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
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


def init_db() -> None:
    """Create all tables (idempotent), set up FTS index, and seed default user."""
    Base.metadata.create_all(engine)
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
