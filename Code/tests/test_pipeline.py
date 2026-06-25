"""Tests for pipeline.persist_match and score_one routing."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from Code.matching import pipeline


def _reqs(req_sk=None):
    return SimpleNamespace(
        required_skills=req_sk or [], preferred_skills=[],
        key_responsibilities=[],
        required_qualifications=[], preferred_qualifications=[],
        min_years_experience=None, required_education_level=None,
        required_certifications=[],
    )


# ---------------------------------------------------------------------------
# persist_match — upsert behaviour
# ---------------------------------------------------------------------------

def test_persist_match_creates_new_row():
    session = MagicMock()
    session.query.return_value.filter_by.return_value.first.return_value = None
    from Code.db.models import Match
    result = {
        "ats_score": 0.75,
        "breakdown": {"overall": 0.75},
        "matched": ["python"],
        "missing": ["java"],
    }
    pipeline.persist_match(session, resume_id=1, jd_id=2, result=result)
    session.add.assert_called_once()
    added = session.add.call_args[0][0]
    assert added.ats_score == 0.75
    assert json.loads(added.matched_skills_json) == ["python"]


def test_persist_match_updates_existing_row():
    existing = MagicMock()
    session = MagicMock()
    session.query.return_value.filter_by.return_value.first.return_value = existing
    result = {
        "ats_score": 0.5,
        "breakdown": {},
        "matched": [],
        "missing": ["fettling"],
    }
    pipeline.persist_match(session, resume_id=3, jd_id=4, result=result)
    session.add.assert_not_called()
    assert existing.ats_score == 0.5
    assert json.loads(existing.missing_skills_json) == ["fettling"]


# ---------------------------------------------------------------------------
# score_one — always uses phrase_ce path
# ---------------------------------------------------------------------------

@patch("Code.matching.pipeline._phrase_result",
       return_value={"ats_score": 0.6, "breakdown": {"scorer": "phrase_ce"}, "matched": [], "missing": []})
def test_score_one_uses_phrase_result(mock_phrase):
    out = pipeline.score_one(SimpleNamespace(), "jd", None)
    assert out["ats_score"] == 0.6
    mock_phrase.assert_called_once()


@patch("Code.matching.pipeline._phrase_result",
       return_value={"ats_score": 0.4, "breakdown": {"scorer": "phrase_ce"}, "matched": [], "missing": []})
def test_score_one_force_flag_ignored(mock_phrase):
    # force=True has no effect in the new implementation (no judge to force)
    out = pipeline.score_one(SimpleNamespace(), "jd", None, force=True)
    assert out["ats_score"] == 0.4
    mock_phrase.assert_called_once()
