"""Tests for Code/db/ — models, session, FTS, repository."""
import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from Code.db.models import Base, Candidate, Education, Experience, Match, Resume, Skill
from Code.db.fts import init_fts, fts_search
from Code.db.repository import find_jd_by_path, save_jd, save_resume, search_resumes
from Code.parser.schemas import (
    Candidate as CandidateSchema,
    Education as EducationSchema,
    Experience as ExperienceSchema,
    ParserMetadata,
    ResumeData,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    init_fts(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def make_resume_data(
    name: str = "Test User",
    email: str = "test@example.com",
    skills: list[str] | None = None,
    experience_years: int = 2,
) -> ResumeData:
    skills = skills or ["Python", "SQL"]
    return ResumeData(
        candidate=CandidateSchema(full_name=name, email=email, phone="9999999999"),
        summary=f"{name} summary",
        skills=skills,
        education=[
            EducationSchema(degree="Bachelor of Technology", field_of_study="Computer Science",
                            institution="Test University", end_date="2020-05")
        ],
        experience=[
            ExperienceSchema(company="Acme Corp", role="Engineer",
                             start_date="2021-01", end_date="2023-01", is_current=False)
        ],
        certifications=[],
        achievements=[],
        languages=[],
        qualifications=[],
        projects=[],
        parser_metadata=ParserMetadata(confidence_score=0.85, missing_important_fields=[],
                                       possible_issues=[], is_messy_resume=False),
    )


# ---------------------------------------------------------------------------
# 1. init_db creates all tables + FTS
# ---------------------------------------------------------------------------

def test_init_db_creates_tables(engine):
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    expected = {"resume", "candidate", "skill", "resume_skill",
                "experience", "education", "match", "job_description"}
    assert expected.issubset(tables)


# ---------------------------------------------------------------------------
# 2. save_resume inserts rows
# ---------------------------------------------------------------------------

def test_save_resume_inserts(session):
    rd = make_resume_data()
    r = save_resume(session, rd, "resume_test.pdf")

    assert session.query(Resume).count() == 1
    assert session.query(Candidate).count() == 1
    assert session.query(Experience).count() == 1
    assert session.query(Education).count() == 1
    assert session.query(Skill).count() == 2   # Python, SQL

    assert r.source_file == "resume_test.pdf"
    assert r.fts_name == "Test User"
    assert "python" in r.fts_skills.lower()


# ---------------------------------------------------------------------------
# 3. save_resume upsert — second call must UPDATE not INSERT
# ---------------------------------------------------------------------------

def test_save_resume_upsert(session):
    rd = make_resume_data()
    r1 = save_resume(session, rd, "resume_upsert.pdf")

    # Change skills and re-save
    rd2 = make_resume_data(name="Test User", skills=["Python", "Django", "FastAPI"])
    r2 = save_resume(session, rd2, "resume_upsert.pdf")

    assert session.query(Resume).count() == 1
    assert r1.id == r2.id
    # Skills should now reflect the new list
    assert session.query(Skill).count() >= 3


# ---------------------------------------------------------------------------
# 4. save_jd upsert
# ---------------------------------------------------------------------------

def test_save_jd_upsert(session):
    jd1 = save_jd(session, "JD1.pdf", "/path/JD1.pdf")
    jd2 = save_jd(session, "JD1.pdf", "/path/JD1.pdf")

    from Code.db.models import JobDescription
    assert session.query(JobDescription).count() == 1
    assert jd1.id == jd2.id


# ---------------------------------------------------------------------------
# 5. Match row created when score_breakdown provided
# ---------------------------------------------------------------------------

def test_match_saved_with_score(session):
    from Code.db.models import JobDescription
    jd = save_jd(session, "JD.pdf", "/path/JD.pdf")
    rd = make_resume_data()
    breakdown = {"overall": 0.75, "skills": 0.8, "experience": 0.7,
                 "qualifications": 0.6, "education": 0.9, "certifications": 1.0,
                 "years_experience": 2.0, "meets_experience": True,
                 "education_met": True, "certifications_met": True}

    save_resume(session, rd, "scored.pdf",
                score_breakdown=breakdown, jd_id=jd.id,
                matched_skills=["Python"], missing_skills=["Java"])

    match = session.query(Match).first()
    assert match is not None
    assert match.ats_score == pytest.approx(0.75)
    assert json.loads(match.matched_skills_json) == ["Python"]
    assert json.loads(match.missing_skills_json) == ["Java"]


# ---------------------------------------------------------------------------
# 6. Match upsert — re-save same resume+jd keeps 1 row
# ---------------------------------------------------------------------------

def test_match_upsert(session):
    jd = save_jd(session, "JD.pdf", "/path/JD2.pdf")
    rd = make_resume_data()
    breakdown = {"overall": 0.60}

    save_resume(session, rd, "dup_match.pdf", score_breakdown=breakdown, jd_id=jd.id)
    save_resume(session, rd, "dup_match.pdf", score_breakdown={"overall": 0.65}, jd_id=jd.id)

    assert session.query(Match).count() == 1
    assert session.query(Match).first().ats_score == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# 7. FTS search finds resume by skill (mocked for SQLite test env)
# ---------------------------------------------------------------------------

def test_search_resumes_by_query(session):
    rd = make_resume_data(name="Alice Smith", skills=["Python", "Machine Learning"])
    save_resume(session, rd, "alice.pdf")

    results = search_resumes(session, query="python")
    assert len(results) == 1
    assert results[0]["candidate_name"] == "Alice Smith"


# ---------------------------------------------------------------------------
# 8. search_resumes filters by min_score
# ---------------------------------------------------------------------------

def test_search_resumes_min_score(session):
    jd = save_jd(session, "JD.pdf", "/path/JD3.pdf")

    rd_high = make_resume_data(name="High Scorer", email="high@example.com")
    save_resume(session, rd_high, "high.pdf", score_breakdown={"overall": 0.80}, jd_id=jd.id)

    rd_low = make_resume_data(name="Low Scorer", email="low@example.com")
    save_resume(session, rd_low, "low.pdf", score_breakdown={"overall": 0.30}, jd_id=jd.id)

    results = search_resumes(session, min_score=0.5)
    names = [r["candidate_name"] for r in results]
    assert "High Scorer" in names
    assert "Low Scorer" not in names
