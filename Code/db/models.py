"""SQLAlchemy 2.0 declarative models for Resume Parser."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Boolean,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class JobDescription(Base):
    __tablename__ = "job_description"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    requirements_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    matches: Mapped[list[Match]] = relationship("Match", back_populates="job_description", cascade="all, delete-orphan")


class Resume(Base):
    __tablename__ = "resume"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_file: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    archive_file: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    document_type: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    parsed_json: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    # Denormalized columns for text search
    fts_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fts_skills: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fts_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Precomputed lexical features for stage-1 retrieval (BM25 token counts +
    # token forms). Populated at ingest so a new JD can shortlist the whole
    # corpus without re-tokenizing every resume. See Code/matching/retrieve.py.
    lex_tokens_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parsed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    candidate: Mapped[Optional[Candidate]] = relationship(
        "Candidate", back_populates="resume", cascade="all, delete-orphan", uselist=False
    )
    matches: Mapped[list[Match]] = relationship("Match", back_populates="resume", cascade="all, delete-orphan")


class Candidate(Base):
    __tablename__ = "candidate"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    resume_id: Mapped[int] = mapped_column(Integer, ForeignKey("resume.id", ondelete="CASCADE"), unique=True, nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    linkedin: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    github: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    portfolio: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    resume: Mapped[Resume] = relationship("Resume", back_populates="candidate")


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)


class Match(Base):
    __tablename__ = "match"
    __table_args__ = (UniqueConstraint("resume_id", "jd_id", name="uq_match_resume_jd"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    resume_id: Mapped[int] = mapped_column(Integer, ForeignKey("resume.id", ondelete="CASCADE"), nullable=False)
    jd_id: Mapped[int] = mapped_column(Integer, ForeignKey("job_description.id", ondelete="CASCADE"), nullable=False)
    ats_score: Mapped[float] = mapped_column(Float, nullable=False)
    score_breakdown_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    matched_skills_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    missing_skills_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    resume: Mapped[Resume] = relationship("Resume", back_populates="matches")
    job_description: Mapped[JobDescription] = relationship("JobDescription", back_populates="matches")
