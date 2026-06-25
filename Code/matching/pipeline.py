"""Scoring orchestration.

Single entry point used by run.py and app.py. Scoring is phrase-driven:
for each JD requirement phrase (skills + responsibilities + qualifications)
a cross-encoder pair (phrase, resume_full_text) is scored in one batched call.
match_score = weighted mean of calibrated CE scores. Pills and score are derived
from the exact same computation.

LLM judge removed. llm_judge.py and retrieve.py remain in place but are no
longer called by any production code path.
"""
from __future__ import annotations

import json
import logging

from Code.matching.bm25e import (
    matched_terms,
    relevant_years,
    resume_full_text,
    score_resume as bm25e_score,
)
from Code.matching.education import check_certifications, meets_requirement
from Code.matching.experience import meets_min_experience
from Code.scoring import build_score_breakdown, compute_ats_score
from Code.parser.schemas import ResumeData


# ---------------------------------------------------------------------------
# Shared scoring factors → breakdown
# ---------------------------------------------------------------------------

def _factors(resume_data, jd_reqs, jd_text: str, match_score: float):
    """Compute relevant years, overall ATS, and the standard breakdown dict
    given a match_score."""
    rel_yrs, tot_yrs = relevant_years(resume_data, jd_reqs, jd_text)
    overall = compute_ats_score(match_score, rel_yrs)
    exp_ok, _ = meets_min_experience(
        rel_yrs, jd_reqs.min_years_experience if jd_reqs else None
    )
    edu_ok, _ = meets_requirement(
        resume_data.education,
        (jd_reqs.required_education_level or "") if jd_reqs else "",
    )
    _, _, missing_certs = check_certifications(
        resume_data.certifications,
        (jd_reqs.required_certifications or []) if jd_reqs else [],
    )
    bd = build_score_breakdown(
        match_score=match_score, overall=overall,
        years_experience=tot_yrs, relevant_years=rel_yrs,
        meets_experience=exp_ok, education_met=edu_ok,
        certifications_met=not missing_certs,
    )
    return overall, bd


def _phrase_result(resume_data, jd_text: str, jd_reqs) -> dict:
    """Per-requirement CE scoring (the sole scoring path)."""
    match_score = bm25e_score(resume_data, jd_text, jd_reqs)
    overall, bd = _factors(resume_data, jd_reqs, jd_text, match_score)
    bd["scorer"] = "phrase_ce"
    matched, missing = matched_terms(resume_full_text(resume_data), jd_text, jd_reqs)
    return {"ats_score": overall, "breakdown": bd, "matched": matched, "missing": missing}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_one(resume_data, jd_text: str, jd_reqs, *, force: bool = False) -> dict:
    """Score a single resume against one JD. Used on upload and explicit re-score."""
    return _phrase_result(resume_data, jd_text, jd_reqs)


def score_jd_against_corpus(session, jd_id: int, jd_text: str, jd_reqs, resume_rows) -> dict:
    """Score all resumes against one JD (used when adding a new JD).
    Persists a Match row per resume and commits."""
    count = 0
    for r in resume_rows:
        if not r.parsed_json:
            continue
        try:
            rd = ResumeData.model_validate_json(r.parsed_json)
            persist_match(session, r.id, jd_id, _phrase_result(rd, jd_text, jd_reqs))
            count += 1
        except Exception:
            logging.exception("Scoring failed for resume %s", r.id)
    session.commit()
    return {"judged": 0, "total": count, "scorer": "phrase_ce"}


def persist_match(session, resume_id: int, jd_id: int, result: dict) -> None:
    from Code.db.models import Match
    m = session.query(Match).filter_by(resume_id=resume_id, jd_id=jd_id).first()
    if m is None:
        m = Match(resume_id=resume_id, jd_id=jd_id)
        session.add(m)
    m.ats_score = result["ats_score"]
    m.score_breakdown_json = json.dumps(result["breakdown"])
    m.matched_skills_json = json.dumps(result["matched"])
    m.missing_skills_json = json.dumps(result["missing"])
