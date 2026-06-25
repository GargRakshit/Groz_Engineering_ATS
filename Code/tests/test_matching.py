"""Tests for per-requirement CE scoring and retained matchers."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from Code.matching.bm25e import (
    _all_jd_phrases,
    _calibrate,
    _compute_phrase_ce,
    _term_match_score,
    matched_terms,
    relevant_years,
    score_resume,
    token_forms,
    tokenize,
    CE_CAL_HI,
    CE_CAL_LO,
    PHRASE_MATCH_THRESHOLD,
)
from Code.matching.kg_normalize import (
    _build_punct_vocab,
    expand_text,
    kg_phrases,
    normalize_punct,
)
from Code.matching.duplicate import DuplicateMatch, find_duplicates
from Code.matching.education import meets_requirement
from Code.matching.experience import meets_min_experience, total_years


# ---------------------------------------------------------------------------
# tokenize — stopwords, lemmatization, punct-strip
# ---------------------------------------------------------------------------

def test_tokenize_lowercases():
    assert tokenize("Python React") == ["python", "react"]


def test_tokenize_tech_terms_not_dropped_or_split():
    cpp_result = tokenize("C++ developer")
    assert "c" not in cpp_result
    assert any(len(t) > 1 and "c" in t for t in cpp_result)

    node_result = tokenize("Node.js developer")
    assert "nodejs" in node_result
    assert "node" not in node_result
    assert "js" not in node_result

    net_result = tokenize(".NET developer")
    assert "net" in net_result


def test_tokenize_empty():
    assert tokenize("") == []


def test_tokenize_removes_stopwords():
    result = tokenize("the quick brown fox is running")
    assert "the" not in result
    assert "is" not in result
    assert "quick" in result
    assert "brown" in result


def test_tokenize_lemmatizes_verbs():
    result = tokenize("developing managed applications systems")
    assert "develop" in result
    assert "manage" in result
    assert "application" in result
    assert "system" in result


def test_tokenize_csharp_and_aspnet():
    result = tokenize("C# and ASP.NET MVC developer")
    assert "csharp" in result
    assert "aspnet" in result


def test_tokenize_no_whitespace_tokens():
    result = tokenize("  python   django  ")
    assert all(t.strip() for t in result)


# ---------------------------------------------------------------------------
# _calibrate
# ---------------------------------------------------------------------------

def test_calibrate_below_lo_is_zero():
    assert _calibrate(CE_CAL_LO) == 0.0
    assert _calibrate(0.0) == 0.0
    assert _calibrate(CE_CAL_LO - 0.1) == 0.0


def test_calibrate_above_hi_is_one():
    assert _calibrate(CE_CAL_HI) == 1.0
    assert _calibrate(1.0) == 1.0


def test_calibrate_midpoint_linear():
    mid = (CE_CAL_LO + CE_CAL_HI) / 2
    assert _calibrate(mid) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# KG normalization
# ---------------------------------------------------------------------------

_SAMPLE_KG = {
    "k8s": ["kubernetes"],
    "kubernetes": ["k8s"],
    "machine learning": ["ml", "artificial intelligence"],
    "ml": ["machine learning"],
    "die casting": ["pressure die casting", "aluminum die casting"],
}

_SAMPLE_PUNCT_KG = {
    "node.js":      ["web platform development software"],
    "c#":           ["c sharp", "object oriented software"],
    "scikit-learn": ["sklearn tools"],
    "asp.net":      ["aspnet framework"],
    "python":       ["programming language"],
}


def test_build_punct_vocab_long_canonical():
    vocab = _build_punct_vocab(_SAMPLE_PUNCT_KG)
    assert vocab["node.js"] == "nodejs"
    assert vocab["scikit-learn"] == "scikitlearn"
    assert vocab["asp.net"] == "aspnet"


def test_build_punct_vocab_short_canonical_fallback():
    vocab = _build_punct_vocab(_SAMPLE_PUNCT_KG)
    assert vocab["c#"] == "csharp"


def test_build_punct_vocab_excludes_plain_words():
    vocab = _build_punct_vocab(_SAMPLE_PUNCT_KG)
    assert "python" not in vocab


def test_build_punct_vocab_empty_expansions():
    assert _build_punct_vocab({}) == {}


@patch("Code.matching.kg_normalize._PUNCT_VOCAB", None)
@patch("Code.matching.kg_normalize._vocab_pattern", None)
@patch("Code.matching.kg_normalize._expansions", _SAMPLE_PUNCT_KG)
def test_normalize_punct_replaces_known_terms():
    result = normalize_punct("node.js developer using scikit-learn")
    assert "nodejs" in result
    assert "scikitlearn" in result
    assert "node.js" not in result
    assert "scikit-learn" not in result


@patch("Code.matching.kg_normalize._PUNCT_VOCAB", None)
@patch("Code.matching.kg_normalize._vocab_pattern", None)
@patch("Code.matching.kg_normalize._expansions", _SAMPLE_PUNCT_KG)
def test_normalize_punct_leaves_unknown_terms():
    result = normalize_punct("react.js developer")
    assert "react.js" in result


@patch("Code.matching.kg_normalize._PUNCT_VOCAB", None)
@patch("Code.matching.kg_normalize._vocab_pattern", None)
@patch("Code.matching.kg_normalize._expansions", {})
def test_normalize_punct_empty_kg_passthrough():
    text = "C++ developer"
    assert normalize_punct(text) == text


@patch("Code.matching.kg_normalize._expansions", _SAMPLE_KG)
def test_kg_expands_unigram():
    result = expand_text("experience with k8s")
    assert "kubernetes" in result


@patch("Code.matching.kg_normalize._expansions", _SAMPLE_KG)
def test_kg_expands_bigram():
    result = expand_text("die casting experience")
    assert "pressure die casting" in result
    assert "aluminum die casting" in result


@patch("Code.matching.kg_normalize._expansions", _SAMPLE_KG)
def test_kg_no_duplicate_if_already_present():
    result = expand_text("kubernetes and k8s experience")
    assert result.count("kubernetes") == 1


@patch("Code.matching.kg_normalize._expansions", _SAMPLE_KG)
def test_kg_longest_match_wins():
    result = expand_text("machine learning engineer")
    assert "ml" in result
    assert "artificial intelligence" in result


@patch("Code.matching.kg_normalize._expansions", {})
def test_kg_empty_expansions_returns_original():
    text = "python developer"
    assert expand_text(text) == text


@patch("Code.matching.kg_normalize._expansions", _SAMPLE_KG)
def test_kg_no_match_returns_original():
    text = "java spring boot"
    assert expand_text(text) == text


# ---------------------------------------------------------------------------
# kg_phrases — longest-match phrase scan
# ---------------------------------------------------------------------------

@patch("Code.matching.kg_normalize._expansions", _SAMPLE_KG)
def test_kg_phrases_finds_unigram():
    assert kg_phrases("experience with k8s") == ["k8s"]


@patch("Code.matching.kg_normalize._expansions", _SAMPLE_KG)
def test_kg_phrases_longest_match_no_overlap():
    found = kg_phrases("machine learning engineer")
    assert "machine learning" in found
    assert "ml" not in found


@patch("Code.matching.kg_normalize._expansions", _SAMPLE_KG)
def test_kg_phrases_multiple():
    found = kg_phrases("die casting and kubernetes work")
    assert "die casting" in found
    assert "kubernetes" in found


@patch("Code.matching.kg_normalize._expansions", {})
def test_kg_phrases_empty_kg():
    assert kg_phrases("python developer") == []


# ---------------------------------------------------------------------------
# _all_jd_phrases — unified phrase pool
# ---------------------------------------------------------------------------

def _reqs(req_sk=None, pref_sk=None, req_q=None, pref_q=None, resp=None):
    return SimpleNamespace(
        required_skills=req_sk or [],
        preferred_skills=pref_sk or [],
        required_qualifications=req_q or [],
        preferred_qualifications=pref_q or [],
        key_responsibilities=resp or [],
    )


def test_all_jd_phrases_required_preferred_weights():
    phrases = _all_jd_phrases(_reqs(req_sk=["python"], pref_sk=["django"]), "ignored")
    assert ("python", 1.0) in phrases
    assert ("django", 0.5) in phrases


def test_all_jd_phrases_dedup_required_wins():
    # same phrase in required_skills AND preferred_skills → only the required entry (×1.0)
    phrases = _all_jd_phrases(
        _reqs(req_sk=["Aluminum Die Casting"], pref_sk=["Aluminum Die Casting"]),
        "ignored",
    )
    matched = [p for p in phrases if p[0] == "Aluminum Die Casting"]
    assert len(matched) == 1
    assert matched[0][1] == 1.0


def test_all_jd_phrases_includes_responsibilities():
    phrases = _all_jd_phrases(
        _reqs(req_sk=["die casting"], resp=["oversee fettling and shot blasting operations"]),
        "ignored",
    )
    texts = [p for p, _ in phrases]
    assert "oversee fettling and shot blasting operations" in texts
    weights = {p: w for p, w in phrases}
    assert weights["oversee fettling and shot blasting operations"] == 0.8


def test_all_jd_phrases_responsibilities_dedup():
    # responsibility already in required_skills → not added again
    phrases = _all_jd_phrases(
        _reqs(req_sk=["die casting"], resp=["die casting"]),
        "ignored",
    )
    matched = [p for p, _ in phrases if p == "die casting"]
    assert len(matched) == 1


def test_all_jd_phrases_qualifications_weights():
    phrases = _all_jd_phrases(
        _reqs(req_q=["operational leadership"], pref_q=["budget ownership"]),
        "ignored",
    )
    weights = {p: w for p, w in phrases}
    assert weights["operational leadership"] == 1.0
    assert weights["budget ownership"] == 0.5


@patch("Code.matching.kg_normalize._expansions", _SAMPLE_KG)
def test_all_jd_phrases_fallback_when_no_reqs():
    phrases = _all_jd_phrases(None, "die casting and kubernetes work")
    texts = [p for p, _ in phrases]
    assert "die casting" in texts
    assert "kubernetes" in texts


@patch("Code.matching.kg_normalize._expansions", _SAMPLE_KG)
def test_all_jd_phrases_fallback_when_reqs_empty():
    phrases = _all_jd_phrases(_reqs(), "die casting work")
    texts = [p for p, _ in phrases]
    assert "die casting" in texts


# ---------------------------------------------------------------------------
# _compute_phrase_ce + score_resume + matched_terms (mocked CE)
# ---------------------------------------------------------------------------

def _resume(skills):
    return SimpleNamespace(skills=skills, experience=[], education=[], qualifications=[])


@patch("Code.matching.kg_normalize._expansions", {})
@patch("Code.matching.bm25e._get_ce")
def test_compute_phrase_ce_calibrates(mock_get_ce):
    # raw 0.93 → calibrated 1.0; raw 0.55 → calibrated 0.0 (below CE_CAL_LO=0.65)
    # _expansions={} disables term-match fallback so only CE calibration is tested.
    mock_get_ce.return_value.predict.return_value = np.array([0.93, 0.55])
    phrases = [("python", 1.0), ("java", 1.0)]
    scores = _compute_phrase_ce("some text", phrases)
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert scores[1] == pytest.approx(0.0, abs=1e-6)


@patch("Code.matching.kg_normalize._expansions", {})
@patch("Code.matching.bm25e._get_ce")
def test_score_resume_weighted_mean(mock_get_ce):
    # two phrases ×1.0 and ×0.5; calibrated scores 1.0 and 0.0
    # _expansions={} prevents term-match fallback from boosting "django" via KG synonyms.
    mock_get_ce.return_value.predict.return_value = np.array([0.93, 0.55])
    reqs = _reqs(req_sk=["python"], pref_sk=["django"])
    result = score_resume(_resume(["python"]), "jd text", reqs)
    # weighted mean: (1.0 * 1.0 + 0.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5
    assert result == pytest.approx(1.0 / 1.5, abs=0.01)


@patch("Code.matching.bm25e._get_ce")
def test_score_resume_all_match(mock_get_ce):
    mock_get_ce.return_value.predict.return_value = np.array([0.95, 0.95])
    reqs = _reqs(req_sk=["python", "django"])
    result = score_resume(_resume(["python", "django"]), "jd text", reqs)
    # both calibrated to 1.0; weighted mean = 1.0
    assert result == pytest.approx(1.0, abs=0.01)


@patch("Code.matching.bm25e._get_ce")
def test_score_resume_no_match(mock_get_ce):
    # raw 0.55 is below CE_CAL_LO=0.65 → calibrated 0.0
    # resume text="cooking" has no die casting/fettling synonyms → term-match also 0.0
    mock_get_ce.return_value.predict.return_value = np.array([0.55, 0.55])
    reqs = _reqs(req_sk=["die casting", "fettling"])
    result = score_resume(_resume(["cooking"]), "jd text", reqs)
    assert result == pytest.approx(0.0, abs=0.01)


@patch("Code.matching.bm25e._get_ce")
def test_score_resume_empty_resume(mock_get_ce):
    result = score_resume(_resume([]), "python developer", None)
    assert result == 0.0
    mock_get_ce.assert_not_called()


@patch("Code.matching.kg_normalize._expansions", {})
@patch("Code.matching.bm25e._get_ce")
def test_matched_terms_threshold(mock_get_ce):
    # score above threshold → matched; below → missing
    # raw 0.93 → cal 1.0 (matched); raw 0.55 → cal 0.0 (missing, below LO=0.65)
    # _expansions={} prevents term-match fallback from boosting "java".
    mock_get_ce.return_value.predict.return_value = np.array([0.93, 0.55])
    reqs = _reqs(req_sk=["python", "java"])
    matched, missing = matched_terms("some resume text", "jd text", reqs)
    assert "python" in matched
    assert "java" in missing


@patch("Code.matching.kg_normalize._expansions", {})
@patch("Code.matching.bm25e._get_ce")
def test_matched_terms_no_overlap_in_matched_and_missing(mock_get_ce):
    mock_get_ce.return_value.predict.return_value = np.array([0.93, 0.55])
    reqs = _reqs(req_sk=["python", "java"])
    matched, missing = matched_terms("some resume text", "jd text", reqs)
    assert set(matched) & set(missing) == set()


@patch("Code.matching.bm25e._get_ce")
def test_matched_terms_responsibilities_in_pool(mock_get_ce):
    # responsibility phrase should appear in matched/missing pills
    mock_get_ce.return_value.predict.return_value = np.array([0.90, 0.71])
    reqs = _reqs(req_sk=["die casting"], resp=["oversee fettling operations"])
    matched, missing = matched_terms("fettling trimming press operations", "jd text", reqs)
    # "die casting" (0.90 → cal 0.893): matched
    # "oversee fettling operations" (0.71 → cal 0.214 < 0.40): missing
    # (no KG entry for the full phrase "oversee fettling operations" → term-match=0)
    all_pills = matched + missing
    assert "oversee fettling operations" in all_pills


# ---------------------------------------------------------------------------
# _term_match_score — synonym fallback
# ---------------------------------------------------------------------------

_FETTLING_KG = {
    "fettling": ["trimming press", "trim press", "deburring", "flash removal"],
    "trimming press": ["fettling", "trim press"],
    "shot blasting": ["grit blasting", "abrasive blasting", "blast cleaning"],
}


@patch("Code.matching.kg_normalize._expansions", _FETTLING_KG)
def test_term_match_direct_phrase():
    # phrase itself appears in the resume text
    assert _term_match_score("trimming press", "manages trimming press units") == 0.80


@patch("Code.matching.kg_normalize._expansions", _FETTLING_KG)
def test_term_match_via_synonym():
    # "fettling" not in resume, but KG synonym "trimming press" is
    assert _term_match_score("fettling", "operates 11 trimming press units") == 0.80


@patch("Code.matching.kg_normalize._expansions", _FETTLING_KG)
def test_term_match_no_match_returns_zero():
    # neither phrase nor synonyms appear in the text
    assert _term_match_score("fettling", "python machine learning developer") == 0.0


@patch("Code.matching.kg_normalize._expansions", {})
def test_term_match_empty_kg_returns_zero():
    # no KG expansions → only direct containment checked
    assert _term_match_score("fettling", "works with trimming press") == 0.0


@patch("Code.matching.kg_normalize._expansions", _FETTLING_KG)
def test_term_match_only_checks_first_5_synonyms():
    # "shot blasting" has 3 synonyms; last one is "blast cleaning"
    assert _term_match_score("shot blasting", "performs blast cleaning operations") == 0.80


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------

def _exp(start, end=None, current=False, role=None, company=None, description=""):
    return SimpleNamespace(
        start_date=start, end_date=end, is_current=current,
        role=role, company=company, description=description,
    )


def test_total_years_no_overlap():
    exps = [_exp("2018-01", "2020-01"), _exp("2020-01", "2022-01")]
    assert total_years(exps) == pytest.approx(4.0, abs=0.1)


def test_total_years_with_overlap():
    exps = [_exp("2018-01", "2021-01"), _exp("2019-01", "2022-01")]
    assert total_years(exps) == pytest.approx(4.0, abs=0.1)


def test_meets_min_experience_pass():
    ok, _ = meets_min_experience(5.0, 4.0)
    assert ok


def test_meets_min_experience_fail():
    ok, _ = meets_min_experience(1.0, 3.0)
    assert not ok


def test_meets_min_experience_no_minimum():
    ok, _ = meets_min_experience(0.0, None)
    assert ok


# ---------------------------------------------------------------------------
# relevant_years — per-experience domain relevance
# ---------------------------------------------------------------------------

IDF_REL = {
    "die": 7.5, "casting": 7.9, "cast": 7.5, "aluminum": 7.4,
    "engineer": 3.5, "production": 3.1, "python": 4.3,
}


def _full_resume(skills, experience):
    return SimpleNamespace(
        skills=skills, experience=experience, education=[], qualifications=[],
    )


@patch("Code.matching.bm25e.load_idf", return_value=IDF_REL)
def test_relevant_years_filters_off_domain_entries(mock_idf):
    resume = _full_resume(
        ["die casting"],
        [
            _exp("2018-01", "2022-01", role="Die Casting Engineer"),
            _exp("2022-01", "2024-01", role="Software Developer",
                 description="python web development"),
        ],
    )
    reqs = _reqs(req_sk=["die casting"])
    rel, total = relevant_years(resume, reqs, "die casting jd text")
    assert total == pytest.approx(6.0, abs=0.1)
    assert rel == pytest.approx(4.0, abs=0.1)


@patch("Code.matching.bm25e.load_idf", return_value=IDF_REL)
def test_relevant_years_all_entries_relevant(mock_idf):
    resume = _full_resume(
        [],
        [
            _exp("2019-01", "2021-01", role="Die Casting Engineer"),
            _exp("2021-01", "2024-01", role="Engineer",
                 description="aluminum casting production"),
        ],
    )
    reqs = _reqs(req_sk=["aluminum die casting"])
    rel, total = relevant_years(resume, reqs, "jd")
    assert rel == pytest.approx(total, abs=0.1)


@patch("Code.matching.bm25e.load_idf", return_value=IDF_REL)
def test_relevant_years_scaled_fallback(mock_idf):
    # Domain evidence only in skills, none in any experience entry →
    # relevant years = total × simple keyword coverage (die casting phrase present)
    resume = _full_resume(
        ["die casting", "aluminum"],
        [_exp("2020-01", "2024-01", role="Engineer",
              description="production work")],
    )
    reqs = _reqs(req_sk=["die casting"])
    rel, total = relevant_years(resume, reqs, "jd")
    assert total == pytest.approx(4.0, abs=0.1)
    # "die casting" present in skills → coverage = 1.0 → rel ≈ total
    assert rel == pytest.approx(total, abs=0.2)


@patch("Code.matching.bm25e.load_idf", return_value=IDF_REL)
def test_relevant_years_off_domain_resume_gets_zero(mock_idf):
    # AI-student vs die-casting JD: no entry relevant, no domain vocab in skills
    resume = _full_resume(
        ["python", "tensorflow"],
        [_exp("2023-01", "2024-01", role="ML Intern",
              description="built python models")],
    )
    reqs = _reqs(req_sk=["die casting"])
    rel, total = relevant_years(resume, reqs, "die casting jd text")
    assert total == pytest.approx(1.0, abs=0.1)
    assert rel == 0.0


@patch("Code.matching.bm25e.load_idf", return_value=IDF_REL)
def test_relevant_years_no_experience(mock_idf):
    resume = _full_resume(["die casting"], [])
    rel, total = relevant_years(resume, _reqs(req_sk=["die casting"]), "jd")
    assert rel == 0.0 and total == 0.0


@patch("Code.matching.kg_normalize._expansions", {})
@patch("Code.matching.bm25e.load_idf", return_value=IDF_REL)
def test_relevant_years_no_jd_basis_counts_all(mock_idf):
    resume = _full_resume(
        [], [_exp("2020-01", "2024-01", role="Engineer")]
    )
    rel, total = relevant_years(resume, None, "generic text with no kg phrases")
    assert rel == pytest.approx(total, abs=0.1)
    assert total == pytest.approx(4.0, abs=0.1)


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------

def _edu(degree):
    return SimpleNamespace(degree=degree)


def test_education_bachelor_meets_bachelor():
    ok, _ = meets_requirement([_edu("Bachelor of Engineering")], "bachelor")
    assert ok


def test_education_master_meets_bachelor():
    ok, _ = meets_requirement([_edu("Master of Science")], "bachelor")
    assert ok


def test_education_high_school_fails_bachelor():
    ok, _ = meets_requirement([_edu("High School Diploma")], "bachelor")
    assert not ok


def test_education_empty_entries():
    ok, _ = meets_requirement([], "bachelor")
    assert not ok


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _cand(name, phone=None, email=None, source="test.pdf"):
    return {"name": name, "phone": phone, "email": email, "source_file": source}


def test_duplicate_exact_email():
    existing = [_cand("Alice", email="a@b.com", source="old.pdf")]
    results = find_duplicates("Alice", None, "a@b.com", existing)
    assert len(results) == 1


def test_duplicate_normalized_phone():
    existing = [_cand("Bob", phone="+91-9876543210", source="old.pdf")]
    results = find_duplicates("Bob", "9876543210", None, existing)
    assert len(results) == 1


def test_duplicate_fuzzy_name():
    existing = [_cand("Rajesh Kumar", source="old.pdf")]
    results = find_duplicates("Rajesh  Kumar", None, None, existing)
    assert len(results) == 1


def test_no_duplicate():
    existing = [_cand("Charlie", email="c@d.com")]
    results = find_duplicates("Alice", None, "a@b.com", existing)
    assert results == []
