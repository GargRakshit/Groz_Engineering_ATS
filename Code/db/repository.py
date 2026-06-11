"""Persistence helpers: upsert resumes/JDs, search candidates."""
from __future__ import annotations

import json
from typing import Optional

from sqlalchemy.orm import Session

from Code.db.fts import fts_search
from Code.db.models import (
    Candidate,
    Education,
    Experience,
    JobDescription,
    Match,
    Resume,
    Skill,
    resume_skill,
)
from Code.parser.schemas import JDRequirements, ResumeData


# ---------------------------------------------------------------------------
# JD helpers
# ---------------------------------------------------------------------------

def find_jd_by_path(session: Session, file_path: str) -> Optional[JobDescription]:
    return session.query(JobDescription).filter_by(file_path=file_path).first()


def save_jd(
    session: Session,
    name: str,
    file_path: str,
    requirements: Optional[JDRequirements] = None,
) -> JobDescription:
    jd = session.query(JobDescription).filter_by(file_path=file_path).first()
    req_json = requirements.model_dump_json() if requirements else None
    if jd is None:
        jd = JobDescription(name=name, file_path=file_path, requirements_json=req_json)
        session.add(jd)
    else:
        jd.name = name
        jd.requirements_json = req_json
    session.commit()
    session.refresh(jd)
    return jd


# ---------------------------------------------------------------------------
# Resume upsert
# ---------------------------------------------------------------------------

def get_candidates_for_dedup(session: Session) -> list[dict]:
    """Return minimal candidate records for duplicate detection."""
    rows = session.query(Candidate, Resume.source_file).join(Resume).all()
    return [
        {
            "source_file": source_file,
            "name": c.full_name,
            "email": c.email,
            "phone": c.phone,
        }
        for c, source_file in rows
    ]


def save_resume(
    session: Session,
    resume_data: ResumeData,
    source_file: str,
    archive_file: Optional[str] = None,
    score_breakdown: Optional[dict] = None,
    jd_id: Optional[int] = None,
    matched_skills: Optional[list[str]] = None,
    missing_skills: Optional[list[str]] = None,
) -> Resume:
    """Upsert a resume and optionally save a Match record.

    On re-run with the same source_file: updates all fields in-place
    (preserves id and created_at). Child rows (Experience, Education, Skills)
    are deleted and re-inserted on every update.
    """
    # --- build denorm FTS strings ---
    fts_name = resume_data.candidate.full_name if resume_data.candidate else None
    fts_skills = " ".join(resume_data.skills) if resume_data.skills else None
    fts_summary = resume_data.summary

    # --- upsert Resume row ---
    resume = session.query(Resume).filter_by(source_file=source_file).first()
    is_new = resume is None
    if is_new:
        resume = Resume(source_file=source_file)
        session.add(resume)

    resume.parsed_json = resume_data.model_dump_json()
    resume.confidence_score = resume_data.parser_metadata.confidence_score if resume_data.parser_metadata else 0.0
    resume.document_type = source_file.rsplit(".", 1)[-1].lower() if "." in source_file else None
    resume.fts_name = fts_name
    resume.fts_skills = fts_skills
    resume.fts_summary = fts_summary
    resume.parsed_at = resume_data.parsed_at
    if archive_file is not None:
        resume.archive_file = archive_file

    # Flush to get resume.id before inserting children
    session.flush()

    # --- candidate (delete+recreate on update) ---
    if not is_new:
        session.query(Candidate).filter_by(resume_id=resume.id).delete()
    c = resume_data.candidate
    session.add(Candidate(
        resume_id=resume.id,
        full_name=c.full_name if c else None,
        email=c.email if c else None,
        phone=c.phone if c else None,
        location=c.location if c else None,
        linkedin=c.linkedin if c else None,
        github=c.github if c else None,
        portfolio=c.portfolio if c else None,
    ))

    # --- experience (delete + re-insert) ---
    session.query(Experience).filter_by(resume_id=resume.id).delete()
    for exp in (resume_data.experience or []):
        session.add(Experience(
            resume_id=resume.id,
            company=exp.company,
            role=exp.role,
            start_date=exp.start_date,
            end_date=exp.end_date,
            is_current=exp.is_current,
            description=exp.description,
        ))

    # --- education (delete + re-insert) ---
    session.query(Education).filter_by(resume_id=resume.id).delete()
    for edu in (resume_data.education or []):
        session.add(Education(
            resume_id=resume.id,
            degree=edu.degree,
            field_of_study=edu.field_of_study,
            institution=edu.institution,
            start_date=edu.start_date,
            end_date=edu.end_date,
            grade=edu.grade,
        ))

    # --- skills (rebuild association table) ---
    session.execute(resume_skill.delete().where(resume_skill.c.resume_id == resume.id))
    for skill_name in (resume_data.skills or []):
        norm = skill_name.lower()
        skill = session.query(Skill).filter_by(name=norm).first()
        if skill is None:
            skill = Skill(name=norm)
            session.add(skill)
            session.flush()
        session.execute(resume_skill.insert().values(resume_id=resume.id, skill_id=skill.id))

    # --- match (upsert if score provided) ---
    if score_breakdown is not None and jd_id is not None:
        match = session.query(Match).filter_by(resume_id=resume.id, jd_id=jd_id).first()
        ats_score = score_breakdown.get("overall", 0.0)
        if match is None:
            match = Match(resume_id=resume.id, jd_id=jd_id)
            session.add(match)
        match.ats_score = ats_score
        match.score_breakdown_json = json.dumps(score_breakdown)
        match.matched_skills_json = json.dumps(matched_skills or [])
        match.missing_skills_json = json.dumps(missing_skills or [])

    session.commit()
    session.refresh(resume)
    return resume


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def get_all_jds(session: Session) -> list[JobDescription]:
    """Return all job descriptions ordered by most recently created."""
    return session.query(JobDescription).order_by(JobDescription.created_at.desc()).all()


def get_resume_by_id(
    session: Session,
    resume_id: int,
    jd_id: Optional[int] = None,
) -> Optional[dict]:
    """Return a resume + candidate + best match as a dict for the detail view."""
    resume = session.query(Resume).filter_by(id=resume_id).first()
    if not resume:
        return None

    candidate = session.query(Candidate).filter_by(resume_id=resume_id).first()

    if jd_id is not None:
        match = session.query(Match).filter_by(resume_id=resume_id, jd_id=jd_id).first()
    else:
        match = (
            session.query(Match)
            .filter_by(resume_id=resume_id)
            .order_by(Match.ats_score.desc())
            .first()
        )

    parsed = json.loads(resume.parsed_json) if resume.parsed_json else {}

    return {
        "id": resume.id,
        "source_file": resume.source_file,
        "candidate": {
            "full_name": candidate.full_name if candidate else None,
            "email": candidate.email if candidate else None,
            "phone": candidate.phone if candidate else None,
            "location": candidate.location if candidate else None,
            "linkedin": candidate.linkedin if candidate else None,
            "github": candidate.github if candidate else None,
            "portfolio": candidate.portfolio if candidate else None,
        },
        "parsed": parsed,
        "score_breakdown": json.loads(match.score_breakdown_json) if match and match.score_breakdown_json else None,
        "matched_skills": json.loads(match.matched_skills_json) if match and match.matched_skills_json else [],
        "missing_skills": json.loads(match.missing_skills_json) if match and match.missing_skills_json else [],
        "match_jd_id": match.jd_id if match else None,
    }


def search_resumes(
    session: Session,
    query: Optional[str] = None,
    min_score: Optional[float] = None,
    jd_id: Optional[int] = None,
    min_years: Optional[float] = None,
    ids: Optional[list[int]] = None,
) -> list[dict]:
    """Return resumes as dicts, optionally filtered by FTS query, min ATS score, JD, years, or IDs."""
    resume_ids: Optional[list[int]] = None

    if query:
        resume_ids = fts_search(session, query)
        if not resume_ids:
            return []

    # Build base query
    q = session.query(Resume)
    if resume_ids is not None:
        q = q.filter(Resume.id.in_(resume_ids))
    if ids is not None:
        q = q.filter(Resume.id.in_(ids))

    resumes = q.all()

    results = []
    for r in resumes:
        best_match: Optional[Match] = None
        if jd_id is not None:
            best_match = session.query(Match).filter_by(resume_id=r.id, jd_id=jd_id).first()
        else:
            best_match = (
                session.query(Match)
                .filter_by(resume_id=r.id)
                .order_by(Match.ats_score.desc())
                .first()
            )

        ats_score = best_match.ats_score if best_match else None
        breakdown = json.loads(best_match.score_breakdown_json) if best_match and best_match.score_breakdown_json else None

        if min_score is not None and (ats_score is None or ats_score < min_score):
            continue

        if min_years is not None:
            years = breakdown.get("years_experience") if breakdown else None
            if years is None or years < min_years:
                continue

        parsed = json.loads(r.parsed_json) if r.parsed_json else {}
        skills_list = parsed.get("skills", [])[:6]
        cand = parsed.get("candidate") or {}

        results.append({
            "id": r.id,
            "source_file": r.source_file,
            "candidate_name": r.fts_name,
            "email": cand.get("email"),
            "location": cand.get("location"),
            "ats_score": ats_score,
            "score_breakdown": breakdown,
            "skills_list": skills_list,
            "match_jd_id": best_match.jd_id if best_match else None,
            "status": r.status,
        })

    results.sort(key=lambda x: (x["ats_score"] or 0.0), reverse=True)
    return results
