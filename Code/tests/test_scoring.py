"""Tests for Code/scoring.py — weighted ATS score computation."""
import pytest

from Code.scoring import build_score_breakdown, compute_ats_score, W_MATCH, W_EXP, EXP_CAP


# ---------------------------------------------------------------------------
# Weight constants sanity
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    assert abs(W_MATCH + W_EXP - 1.0) < 1e-9


def test_exp_cap_positive():
    assert EXP_CAP > 0


# ---------------------------------------------------------------------------
# compute_ats_score
# ---------------------------------------------------------------------------

def test_perfect_score():
    # match=1.0, enough years to max out exp score
    assert compute_ats_score(1.0, EXP_CAP) == pytest.approx(1.0)


def test_zero_score():
    assert compute_ats_score(0.0, 0.0) == pytest.approx(0.0)


def test_known_good():
    # match=0.6, years=5 → exp_score=0.5
    # overall = 0.70*0.6 + 0.30*0.5 = 0.42 + 0.15 = 0.57
    assert compute_ats_score(0.6, 5.0) == pytest.approx(0.57, abs=1e-4)


def test_exp_score_capped_at_one():
    # 20 years → exp_score capped at 1.0
    assert compute_ats_score(0.5, 20.0) == pytest.approx(W_MATCH * 0.5 + W_EXP * 1.0, abs=1e-4)


def test_exp_score_fractional():
    # 5 years out of 10 cap → exp_score = 0.5
    result = compute_ats_score(0.0, 5.0)
    assert result == pytest.approx(W_EXP * 0.5, abs=1e-4)


def test_match_only():
    # 0 years → exp_score=0; result driven purely by match
    result = compute_ats_score(0.8, 0.0)
    assert result == pytest.approx(W_MATCH * 0.8, abs=1e-4)


def test_determinism():
    r1 = compute_ats_score(0.55, 4.2)
    r2 = compute_ats_score(0.55, 4.2)
    assert r1 == r2


# ---------------------------------------------------------------------------
# build_score_breakdown
# ---------------------------------------------------------------------------

def test_breakdown_keys():
    bd = build_score_breakdown(
        match_score=0.6,
        overall=0.57,
        years_experience=5.0,
        meets_experience=True,
        education_met=True,
        certifications_met=False,
    )
    assert set(bd.keys()) == {
        "overall", "match_score", "years_experience",
        "meets_experience", "education_met", "certifications_met",
    }


def test_breakdown_overall_matches_compute():
    match = 0.6
    yrs = 5.0
    overall = compute_ats_score(match, yrs)
    bd = build_score_breakdown(
        match_score=match,
        overall=overall,
        years_experience=yrs,
        meets_experience=True,
        education_met=False,
        certifications_met=True,
    )
    assert bd["overall"] == pytest.approx(overall, abs=1e-4)
    assert bd["education_met"] is False
    assert bd["certifications_met"] is True
    assert bd["years_experience"] == pytest.approx(yrs, abs=0.05)


def test_breakdown_rounding():
    bd = build_score_breakdown(
        match_score=0.12345,
        overall=0.0864,
        years_experience=1.123,
        meets_experience=False,
        education_met=False,
        certifications_met=False,
    )
    assert bd["match_score"] == round(0.12345, 4)
    assert bd["years_experience"] == round(1.123, 1)


def test_breakdown_boolean_fields():
    bd = build_score_breakdown(0.5, 0.5, 3.0, True, False, True)
    assert bd["meets_experience"] is True
    assert bd["education_met"] is False
    assert bd["certifications_met"] is True
