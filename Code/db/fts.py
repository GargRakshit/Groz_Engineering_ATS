"""Text search helpers for resume search.

Uses LIKE-based search across denormalized fts_name/fts_skills/fts_summary columns.
This is reliable across MySQL versions with no minimum word length or stopword
limitations that affect MySQL FULLTEXT.
"""
from __future__ import annotations

from sqlalchemy import or_, text
from sqlalchemy.orm import Session


def init_fts(engine) -> None:
    """Drop the legacy FULLTEXT index if it exists (idempotent)."""
    with engine.connect() as conn:
        try:
            conn.execute(text("DROP INDEX resume_fts_idx ON resume"))
            conn.commit()
        except Exception:
            pass  # index doesn't exist — nothing to do


def fts_search(session: Session, query: str) -> list[int]:
    """Return resume IDs whose FTS columns contain all query terms (AND logic)."""
    from Code.db.models import Resume

    terms = [t.strip() for t in query.split() if t.strip()]
    if not terms:
        return []

    q = session.query(Resume.id)
    for term in terms:
        pattern = f"%{term}%"
        q = q.filter(
            or_(
                Resume.fts_name.ilike(pattern),
                Resume.fts_skills.ilike(pattern),
                Resume.fts_summary.ilike(pattern),
            )
        )
    return [r[0] for r in q.all()]
