"""Stage-1 candidate retrieval for two-stage scoring.

The expensive reranker (LLM judge / cross-encoder) must only run on a small
shortlist, never on the whole corpus. This module produces that shortlist using
the cheap, model-free lexical signals already in bm25e (BM25 over corpus IDF +
JD-phrase coverage), so a new JD can be ranked against thousands of resumes in
milliseconds.

To stay fast at corpus scale, each resume's lexical representation is computed
ONCE at ingest (resume_lex_features) and stored in Resume.lex_tokens_json; the
shortlist then reads stored features and does pure arithmetic — no spaCy, no
model, at query time.
"""
from __future__ import annotations

import json
from collections import Counter

from Code.matching.bm25e import (
    _bm25_raw,
    _jd_phrases,
    _normalize_bm25,
    phrase_match,
    resume_full_text,
    token_forms,
    tokenize,
    _idf_threshold,
)
from Code.matching.corpus import load_idf
from Code.matching.kg_normalize import expand_text

# Stage-1 score = LEX_BM25 * bm25 + LEX_PHRASE * phrase. Phrase coverage is the
# stronger domain signal, so it dominates; BM25 breaks ties / adds recall.
LEX_BM25: float = 0.4
LEX_PHRASE: float = 0.6


def resume_lex_features(resume_data) -> dict:
    """Compute the stored lexical representation of a resume (run once at ingest).

    Returns {"tf": {token: count}, "forms": [token forms]} over the KG-expanded
    full resume text. Serialized into Resume.lex_tokens_json.
    """
    expanded = expand_text(resume_full_text(resume_data))
    tf = Counter(tokenize(expanded))
    forms = sorted(token_forms(expanded))
    return {"tf": dict(tf), "forms": forms}


def _features_from_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "tf" not in data or "forms" not in data:
        return None
    return data


def lex_score(
    jd_query_tokens: list[str],
    jd_phrases: list[tuple[str, float]],
    features: dict,
    idf: dict[str, float],
) -> float:
    """Cheap lexical relevance of one resume to a JD from stored features."""
    tf = features.get("tf") or {}
    doc_tokens: list[str] = []
    for tok, count in tf.items():
        doc_tokens.extend([tok] * int(count))
    bm25 = _normalize_bm25(
        _bm25_raw(jd_query_tokens, doc_tokens, idf), jd_query_tokens, idf
    )
    forms = set(features.get("forms") or [])
    phrase, _, _ = phrase_match(jd_phrases, forms, idf)
    return LEX_BM25 * bm25 + LEX_PHRASE * phrase


def _jd_query_tokens(jd_text: str, idf: dict[str, float]) -> list[str]:
    """JD BM25 query tokens with generic (low-IDF) terms dropped — mirrors the
    filtering in bm25e.score_resume so stage-1 and stage-2 agree on vocabulary."""
    q_tokens = tokenize(jd_text)
    threshold = _idf_threshold(idf)
    if threshold is not None:
        filtered = [t for t in q_tokens if idf.get(t, threshold) >= threshold]
        if filtered:
            return filtered
    return q_tokens


def lexical_relevance(resume_data, jd_text: str, jd_reqs) -> float:
    """Stage-1 lexical score for a single resume computed on the fly (used by the
    upload path to gate whether the judge runs for a given resume/JD pair)."""
    idf = load_idf()
    features = resume_lex_features(resume_data)
    return lex_score(
        _jd_query_tokens(jd_text, idf), _jd_phrases(jd_reqs, jd_text), features, idf
    )


def shortlist(
    jd_text: str,
    jd_reqs,
    corpus: list[tuple[int, str | None]],
    k: int,
) -> list[tuple[int, float]]:
    """Rank the whole corpus by cheap lexical score and return the top-K.

    corpus: list of (resume_id, lex_tokens_json) for every resume.
    Returns [(resume_id, lex_score), ...] sorted desc, length <= k.
    Resumes with no stored features score 0 (still returned if k allows, so a
    freshly-ingested corpus never silently drops everyone).
    """
    idf = load_idf()
    q_tokens = _jd_query_tokens(jd_text, idf)
    jd_phr = _jd_phrases(jd_reqs, jd_text)

    scored: list[tuple[int, float]] = []
    for resume_id, raw in corpus:
        features = _features_from_json(raw)
        score = lex_score(q_tokens, jd_phr, features, idf) if features else 0.0
        scored.append((resume_id, round(score, 4)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(k, 0)]
