"""BM25 + Cross-Encoder hybrid scoring for resume–JD matching.

Tokenization pipeline: tech-term normalization → punct-strip → spacy stopword
removal + lemmatization.
Scoring pipeline: α·BM25(lemmatized) + (1-α)·CrossEncoder(raw text).
KG expansion (expand_text) is applied to the BM25 document side before tokenization.

Cross-encoder: Alibaba-NLP/gte-reranker-modernbert-base
  - 8192-token context (ModernBERT architecture, trained for long docs)
  - LoCo long-document retrieval score: 90.68
  - Drop-in CrossEncoder replacement, Apache 2.0
"""
import re
from collections import Counter

import numpy as np
from sentence_transformers import CrossEncoder

from Code.matching.corpus import load_idf
from Code.matching.kg_normalize import expand_text, normalize_punct

_PUNCT = re.compile(r"[^\w\s]")
_CE_MODEL_NAME = "Alibaba-NLP/gte-reranker-modernbert-base"

_nlp = None
_ce_model: CrossEncoder | None = None


# ---------------------------------------------------------------------------
# Lazy model loaders
# ---------------------------------------------------------------------------

def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


def _get_ce() -> CrossEncoder:
    global _ce_model
    if _ce_model is None:
        _ce_model = CrossEncoder(_CE_MODEL_NAME)
    return _ce_model


# ---------------------------------------------------------------------------
# Tokenization: punct-strip → spacy (stopwords + lemmatization)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, normalize KG-known punct terms, strip punctuation, remove stopwords, lemmatize."""
    normalized = normalize_punct(text.lower())
    cleaned = _PUNCT.sub(" ", normalized)
    if not cleaned.strip():
        return []
    doc = _get_nlp()(cleaned)
    return [
        t.lemma_
        for t in doc
        if not t.is_stop and not t.is_space and len(t.text.strip()) > 1
    ]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def _bm25_raw(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
    avgdl: int = 150,
) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_len = len(doc_tokens)
    tf = Counter(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        if term not in tf:
            continue
        term_idf = idf.get(term, 1.0)
        term_tf = tf[term]
        score += term_idf * (term_tf * (k1 + 1)) / (
            term_tf + k1 * (1 - b + b * doc_len / avgdl)
        )
    return score


def _normalize_bm25(
    raw: float, query_tokens: list[str], idf: dict[str, float], k1: float = 1.5
) -> float:
    perfect = sum(idf.get(t, 1.0) * (k1 + 1) / (1 + k1) for t in set(query_tokens))
    if perfect == 0:
        return 0.0
    return min(raw / perfect, 1.0)


# ---------------------------------------------------------------------------
# Cross-encoder scorer
# ---------------------------------------------------------------------------

def _ce_score(jd_text: str, resume_text: str) -> float:
    """Score (jd_text, resume_text) pair via cross-encoder. Returns sigmoid [0, 1]."""
    if not resume_text.strip():
        return 0.0
    raw = float(_get_ce().predict([(jd_text, resume_text)])[0])
    return float(1 / (1 + np.exp(-raw)))


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_section(
    section_text: str,
    jd_text: str,
    idf: dict[str, float],
    alpha: float = 0.4,
    use_kg: bool = False,
) -> float:
    """Score text against JD text.

    Final score = α·BM25(lemmatized) + (1-α)·CrossEncoder(raw).
    """
    if not section_text.strip() or not jd_text.strip():
        return 0.0

    q_tokens = tokenize(jd_text)
    doc_text = expand_text(section_text) if use_kg else section_text
    d_tokens = tokenize(doc_text)

    raw = _bm25_raw(q_tokens, d_tokens, idf)
    bm25 = _normalize_bm25(raw, q_tokens, idf)
    ce = _ce_score(jd_text, section_text)

    return round(alpha * bm25 + (1 - alpha) * ce, 4)


def matched_terms(resume_text: str, jd_text: str) -> tuple[list[str], list[str]]:
    """Return (matched_jd_terms, missing_jd_terms) using lemmatized tokens."""
    q_tokens = set(tokenize(jd_text))
    d_token_set = set(tokenize(expand_text(resume_text)))
    matched = sorted(t for t in q_tokens if t in d_token_set)
    missing = sorted(t for t in q_tokens if t not in d_token_set)
    return matched, missing


# ---------------------------------------------------------------------------
# Full resume scoring
# ---------------------------------------------------------------------------

def score_resume(resume, jd_raw_text: str) -> float:
    """Score entire resume against entire JD. Returns a single float in [0, 1]."""
    idf = load_idf()

    full_text = " ".join(filter(None, [
        " ".join(resume.skills),
        " ".join(
            " ".join(filter(None, [
                e.role,
                e.company,
                " ".join(e.description) if e.description else None,
            ]))
            for e in resume.experience
        ),
        " ".join(
            " ".join(filter(None, [ed.degree, ed.field_of_study, ed.institution]))
            for ed in resume.education
        ),
        " ".join(resume.qualifications),
    ]))

    return score_section(full_text, jd_raw_text, idf, use_kg=True)
