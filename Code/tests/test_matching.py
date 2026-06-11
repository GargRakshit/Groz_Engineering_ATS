"""Tests for the BM25+CE matching package and retained matchers."""
import math
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from Code.matching.bm25e import (
    _bm25_raw,
    _normalize_bm25,
    score_section,
    tokenize,
)
from Code.matching.kg_normalize import _build_punct_vocab, expand_text, normalize_punct
from Code.matching.duplicate import DuplicateMatch, find_duplicates
from Code.matching.education import meets_requirement
from Code.matching.experience import meets_min_experience, total_years


# ---------------------------------------------------------------------------
# tokenize — stopwords, lemmatization, punct-strip
# ---------------------------------------------------------------------------

def test_tokenize_lowercases():
    assert tokenize("Python React") == ["python", "react"]


def test_tokenize_tech_terms_not_dropped_or_split():
    # C++ must produce a stable multi-char token (not collapsed to 'c' and dropped)
    cpp_result = tokenize("C++ developer")
    assert "c" not in cpp_result
    assert any(len(t) > 1 and "c" in t for t in cpp_result)

    # Node.js must be unified — not split into 'node' + 'js'
    node_result = tokenize("Node.js developer")
    assert "nodejs" in node_result
    assert "node" not in node_result
    assert "js" not in node_result

    # .NET must survive as a whole token
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
# BM25 raw + normalize
# ---------------------------------------------------------------------------

IDF_SAMPLE = {"python": 2.0, "django": 3.0, "rest": 1.5, "api": 1.2}


def test_bm25_raw_exact_match():
    score = _bm25_raw(["python"], ["python", "django"], IDF_SAMPLE)
    assert score > 0


def test_bm25_raw_no_overlap():
    assert _bm25_raw(["java"], ["python", "django"], IDF_SAMPLE) == 0.0


def test_bm25_raw_empty_query():
    assert _bm25_raw([], ["python"], IDF_SAMPLE) == 0.0


def test_bm25_raw_empty_doc():
    assert _bm25_raw(["python"], [], IDF_SAMPLE) == 0.0


def test_normalize_bm25_perfect():
    raw = _bm25_raw(["python"], ["python"], IDF_SAMPLE)
    norm = _normalize_bm25(raw, ["python"], IDF_SAMPLE)
    assert norm == pytest.approx(1.0, abs=0.01)


def test_normalize_bm25_zero_query():
    assert _normalize_bm25(0.0, [], IDF_SAMPLE) == 0.0


def test_bm25_oov_uses_fallback_idf():
    score = _bm25_raw(["unknownxyz"], ["unknownxyz"], {})
    assert score > 0


# ---------------------------------------------------------------------------
# score_section (mocked cross-encoder)
# ---------------------------------------------------------------------------

@patch("Code.matching.bm25e._ce_score", return_value=0.0)
def test_score_section_empty_section(mock_ce):
    assert score_section("", "some jd text", {}) == 0.0
    mock_ce.assert_not_called()


@patch("Code.matching.bm25e._ce_score", return_value=0.0)
def test_score_section_empty_jd(mock_ce):
    assert score_section("some resume text", "", {}) == 0.0
    mock_ce.assert_not_called()


@patch("Code.matching.bm25e._ce_score", return_value=1.0)
def test_score_section_identical_texts(mock_ce):
    # BM25 ≈ 1.0 on identical text, CE mocked to 1.0 → result ≈ 1.0
    text = "python django rest api development"
    result = score_section(text, text, IDF_SAMPLE)
    assert result >= 0.9


@patch("Code.matching.bm25e._ce_score", return_value=0.0)
def test_score_section_no_overlap_low_ce(mock_ce):
    # BM25 = 0 (no overlap), CE = 0 → result = 0.0
    result = score_section("python developer", "die casting machine operator", IDF_SAMPLE)
    assert result == pytest.approx(0.0)


@patch("Code.matching.bm25e._ce_score", return_value=0.5)
def test_score_section_returns_float_in_range(mock_ce):
    result = score_section("machine learning engineer", "python data science ml", IDF_SAMPLE)
    assert 0.0 <= result <= 1.0


@patch("Code.matching.bm25e._ce_score", return_value=0.8)
def test_score_section_alpha_weighting(mock_ce):
    # BM25 ≈ 0 (no overlap), CE = 0.8 → result ≈ (1-0.4)*0.8 = 0.48
    result = score_section("java spring boot", "python django flask", IDF_SAMPLE, alpha=0.4)
    assert result == pytest.approx(0.48, abs=0.05)


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

# KG entries with meaningful punctuation — used by _build_punct_vocab tests.
_SAMPLE_PUNCT_KG = {
    "node.js":      ["web platform development software"],   # canonical: 'nodejs'
    "c#":           ["c sharp", "object oriented software"], # canonical: 'c' (fallback→'csharp')
    "scikit-learn": ["sklearn tools"],                       # canonical: 'scikitlearn'
    "asp.net":      ["aspnet framework"],                    # canonical: 'aspnet'
    # plain word — must NOT enter the vocab
    "python":       ["programming language"],
}


# ---------------------------------------------------------------------------
# _build_punct_vocab (unit tests, no global state)
# ---------------------------------------------------------------------------

def test_build_punct_vocab_long_canonical():
    vocab = _build_punct_vocab(_SAMPLE_PUNCT_KG)
    assert vocab["node.js"] == "nodejs"
    assert vocab["scikit-learn"] == "scikitlearn"
    assert vocab["asp.net"] == "aspnet"


def test_build_punct_vocab_short_canonical_fallback():
    # 'c#' → strip non-word → 'c' (len 1) → fallback to shortest exp canonical
    # 'c sharp' → strip → 'csharp' (len 6 ≥ 2)
    vocab = _build_punct_vocab(_SAMPLE_PUNCT_KG)
    assert vocab["c#"] == "csharp"


def test_build_punct_vocab_excludes_plain_words():
    vocab = _build_punct_vocab(_SAMPLE_PUNCT_KG)
    assert "python" not in vocab


def test_build_punct_vocab_empty_expansions():
    assert _build_punct_vocab({}) == {}


# ---------------------------------------------------------------------------
# normalize_punct (patches global caches so tests are isolated)
# ---------------------------------------------------------------------------

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
    # 'react.js' is not in _SAMPLE_PUNCT_KG → left unchanged
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
# Experience
# ---------------------------------------------------------------------------

def _exp(start, end=None, current=False):
    return SimpleNamespace(start_date=start, end_date=end, is_current=current)


def test_total_years_no_overlap():
    exps = [_exp("2018-01", "2020-01"), _exp("2020-01", "2022-01")]
    assert total_years(exps) == pytest.approx(4.0, abs=0.1)


def test_total_years_with_overlap():
    exps = [_exp("2018-01", "2021-01"), _exp("2019-01", "2022-01")]
    assert total_years(exps) == pytest.approx(4.0, abs=0.1)


def test_meets_min_experience_pass():
    exps = [_exp("2018-01", "2023-01")]
    ok, _ = meets_min_experience(exps, 4.0)
    assert ok


def test_meets_min_experience_fail():
    exps = [_exp("2022-01", "2023-01")]
    ok, _ = meets_min_experience(exps, 3.0)
    assert not ok


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
