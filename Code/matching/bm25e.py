"""Per-requirement CE scoring for resume–JD matching.

For each JD requirement phrase (required_skills, preferred_skills,
key_responsibilities, required_qualifications, preferred_qualifications)
a cross-encoder pair (phrase, resume_full_text) is scored in a single
batched call. The match_score is the weighted mean of calibrated CE scores;
the matched/missing pills are those same scores thresholded.

Weights per source:
  required_skills          ×1.0
  preferred_skills         ×0.5
  key_responsibilities     ×0.8
  required_qualifications  ×1.0
  preferred_qualifications ×0.5

CE calibration: for per-phrase scoring (short phrase vs long resume), the
gte reranker's sigmoid outputs cluster in a narrower band than full-doc
scoring. Off-domain phrases score ~0.53–0.63; domain-adjacent-but-absent
phrases score ~0.63–0.73; genuinely matched phrases score ~0.80–0.93.
_calibrate() maps [CE_CAL_LO=0.65, CE_CAL_HI=0.93] → [0, 1].
Constants are empirical for this model/task — re-measure if model is swapped.

Term-match fallback: when the CE score is below PHRASE_MATCH_THRESHOLD but a
KG synonym of the phrase appears verbatim in the resume text, the score is
raised to 0.80. This handles cases where the CE sees an unfamiliar domain
synonym and undershoots (e.g. "Fettling" → resume says "trimming press").

Cross-encoder: Alibaba-NLP/gte-reranker-modernbert-base
  - 8192-token context (ModernBERT architecture)
  - LoCo long-document retrieval score: 90.68
  - Apache 2.0 license
  - sentence-transformers' CrossEncoder applies sigmoid internally for
    single-label models; predict() already returns probabilities — do NOT
    re-apply sigmoid (that crushes [0,1] into [0.5, 0.73]).

KG expansion (expand_text) is used for:
  - JD phrase expansion (asymmetric query expansion): phrases are expanded before
    the CE call so domain synonyms (e.g. "Fettling" → "Fettling trimming press
    deburring") reach the CE; the resume full_text is NOT expanded.
  - relevant_years: expanding experience-entry text before domain-token matching.
"""
import re

from sentence_transformers import CrossEncoder

from Code.matching.corpus import load_idf
from Code.matching.kg_normalize import expand_text, kg_phrases, normalize_punct

_PUNCT = re.compile(r"[^\w\s]")
_CE_MODEL_NAME = "Alibaba-NLP/gte-reranker-modernbert-base"

# CE calibration bounds (empirical for gte-reranker-modernbert-base).
# Raw sigmoid at/below LO → 0, at/above HI → 1.
CE_CAL_LO: float = 0.65
CE_CAL_HI: float = 0.93
# Min corpus IDF for a token to count as domain-specific when deciding
# whether an experience entry is relevant to the JD.
# die≈7.5, casting≈7.9, aluminum≈7.4 pass; machine≈4.2, engineer≈3.5 don't.
EXP_REL_IDF: float = 5.0
# Per-phrase calibrated CE score at/above which a phrase counts as matched.
PHRASE_MATCH_THRESHOLD: float = 0.40

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


def token_forms(text: str) -> set[str]:
    """Set of both lemma and raw (lowercased) forms of every content token.

    spaCy lemmas are context-dependent ('casting' → 'cast' in one sentence,
    'casting' in another), so phrase containment checks both forms on both
    sides instead of relying on lemma stability."""
    normalized = normalize_punct(text.lower())
    cleaned = _PUNCT.sub(" ", normalized)
    if not cleaned.strip():
        return set()
    doc = _get_nlp()(cleaned)
    forms: set[str] = set()
    for t in doc:
        if t.is_stop or t.is_space or len(t.text.strip()) <= 1:
            continue
        forms.add(t.lemma_)
        forms.add(t.text)
    return forms


# ---------------------------------------------------------------------------
# CE calibration
# ---------------------------------------------------------------------------

def _calibrate(raw: float) -> float:
    """Map a raw CE sigmoid score onto [0, 1] using the empirical band of the
    current model (see CE_CAL_LO/CE_CAL_HI). Off-domain text sits at or below
    LO and maps to 0; a strong same-domain match sits near HI and maps to 1."""
    return min(max((raw - CE_CAL_LO) / (CE_CAL_HI - CE_CAL_LO), 0.0), 1.0)


# ---------------------------------------------------------------------------
# Unified JD phrase pool
# ---------------------------------------------------------------------------

def _all_jd_phrases(jd_reqs, jd_text: str) -> list[tuple[str, float]]:
    """Build the unified weighted phrase pool from all JDRequirements fields.

    Sources and weights (first occurrence by lowercase text wins):
      required_skills          ×1.0
      preferred_skills         ×0.5
      key_responsibilities     ×0.8
      required_qualifications  ×1.0
      preferred_qualifications ×0.5

    Falls back to kg_phrases(jd_text) at ×1.0 when jd_reqs is None or
    all lists are empty (e.g. old cached JD without extracted fields)."""
    if jd_reqs is not None:
        seen: dict[str, tuple[str, float]] = {}

        def _add(items, weight):
            for s in (items or []):
                if s and s.lower() not in seen:
                    seen[s.lower()] = (s, weight)

        _add(jd_reqs.required_skills, 1.0)
        _add(jd_reqs.preferred_skills, 0.5)
        _add(getattr(jd_reqs, "key_responsibilities", None), 0.8)
        _add(jd_reqs.required_qualifications, 1.0)
        _add(jd_reqs.preferred_qualifications, 0.5)

        phrases = list(seen.values())
        if phrases:
            return phrases

    # Fallback: scan the raw JD text for KG-known phrases
    return [(p, 1.0) for p in kg_phrases(jd_text)]


# ---------------------------------------------------------------------------
# Batched CE scoring
# ---------------------------------------------------------------------------

def _term_match_score(phrase: str, full_text: str) -> float:
    """Fallback score when CE misses a domain synonym.

    Returns 0.80 if the phrase itself OR any of its KG synonyms appears
    verbatim in the resume text. Applied ONLY when the CE score is below
    PHRASE_MATCH_THRESHOLD, so it never overrides a CE match.

    Example: "Fettling" → KG synonyms include "trimming press" → if Ankesh's
    resume contains "trimming press", this returns 0.80 and the phrase is
    counted as matched. Devashish (no trimming press) stays at 0.0.
    """
    from Code.matching.kg_normalize import load_expansions
    text_lower = full_text.lower()
    phrase_lower = phrase.lower()
    if phrase_lower in text_lower:
        return 0.80
    for syn in load_expansions().get(phrase_lower, [])[:5]:
        if syn in text_lower:
            return 0.80
    return 0.0


def _compute_phrase_ce(full_text: str, phrases: list[tuple[str, float]]) -> list[float]:
    """Score each phrase against full_text via the cross-encoder (one batched call).

    CE scores are calibrated then augmented with a term-match fallback:
    if the calibrated CE score is below PHRASE_MATCH_THRESHOLD but the phrase
    (or a KG synonym) literally appears in the resume text, the score is raised
    to 0.80. This handles domain synonyms the CE misses — e.g. "Fettling"
    when the resume says "trimming press" — without inflating already-matched
    phrases. Resume full_text is NOT KG-expanded (CE handles general semantics
    on the document side; expansion there adds noise).

    Returns calibrated+fallback scores in [0, 1], same order as input phrases."""
    pairs = [(p, full_text) for p, _ in phrases]
    raw = _get_ce().predict(pairs)
    result = []
    for (p, _), r in zip(phrases, raw):
        cal = _calibrate(float(r))
        if cal < PHRASE_MATCH_THRESHOLD:
            cal = max(cal, _term_match_score(p, full_text))
        result.append(cal)
    return result


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------

def score_resume(resume, jd_raw_text: str, jd_reqs=None) -> float:
    """Score resume against a JD. Returns a single float in [0, 1].

    match_score = IDF-weighted mean of per-requirement CE scores.
    Uses _all_jd_phrases to build the phrase pool (skills + responsibilities +
    qualifications); _compute_phrase_ce for the batched CE call.
    """
    full_text = resume_full_text(resume)
    if not full_text.strip() or not jd_raw_text.strip():
        return 0.0
    phrases = _all_jd_phrases(jd_reqs, jd_raw_text)
    if not phrases:
        return 0.0
    scores = _compute_phrase_ce(full_text, phrases)
    total_w = sum(w for _, w in phrases)
    return round(sum(s * w for s, (_, w) in zip(scores, phrases)) / total_w, 4)


def matched_terms(
    resume_text: str, jd_text: str, jd_reqs=None
) -> tuple[list[str], list[str]]:
    """Return (matched, missing) JD requirement phrases for the UI pills.

    A phrase is matched when its calibrated CE score ≥ PHRASE_MATCH_THRESHOLD.
    Score and pills are derived from the exact same computation (same phrase pool,
    same CE batch call) so they always tell the same story."""
    if not resume_text.strip():
        return [], []
    phrases = _all_jd_phrases(jd_reqs, jd_text)
    if not phrases:
        return [], []
    scores = _compute_phrase_ce(resume_text, phrases)
    matched = [p for (p, _), s in zip(phrases, scores) if s >= PHRASE_MATCH_THRESHOLD]
    missing = [p for (p, _), s in zip(phrases, scores) if s < PHRASE_MATCH_THRESHOLD]
    return list(dict.fromkeys(matched)), list(dict.fromkeys(missing))


def score_and_match_batch(
    resumes: list, jd_raw_text: str, jd_reqs=None
) -> list[tuple[float, list[str], list[str]]]:
    """Score multiple resumes and compute matched/missing pills in one CE call.

    All (phrase, resume_text) pairs across every resume are sent to the CE in a
    single batched prediction, which is substantially faster than calling
    score_resume() + matched_terms() N times (N CE calls → 1 CE call).

    Returns list of (match_score, matched_phrases, missing_phrases), same order
    as *resumes*.
    """
    if not resumes:
        return []
    phrases = _all_jd_phrases(jd_reqs, jd_raw_text)
    if not phrases or not jd_raw_text.strip():
        return [(0.0, [], []) for _ in resumes]

    full_texts = [resume_full_text(r) for r in resumes]
    n_phrases = len(phrases)
    total_w = sum(w for _, w in phrases)

    # One mega-batch: for each resume, pair every phrase with that resume's text
    all_pairs = [(p, ft) for ft in full_texts for p, _ in phrases]
    raw_all = _get_ce().predict(all_pairs, batch_size=256, show_progress_bar=False)

    results = []
    for i, ft in enumerate(full_texts):
        if not ft.strip():
            results.append((0.0, [], []))
            continue
        cal_scores = []
        for j, (p, _) in enumerate(phrases):
            cal = _calibrate(float(raw_all[i * n_phrases + j]))
            if cal < PHRASE_MATCH_THRESHOLD:
                cal = max(cal, _term_match_score(p, ft))
            cal_scores.append(cal)

        match_score = round(
            sum(s * w for s, (_, w) in zip(cal_scores, phrases)) / total_w, 4
        )
        matched = list(dict.fromkeys(
            p for (p, _), s in zip(phrases, cal_scores) if s >= PHRASE_MATCH_THRESHOLD
        ))
        missing = list(dict.fromkeys(
            p for (p, _), s in zip(phrases, cal_scores) if s < PHRASE_MATCH_THRESHOLD
        ))
        results.append((match_score, matched, missing))

    return results


# ---------------------------------------------------------------------------
# Resume text assembly
# ---------------------------------------------------------------------------

def _experience_text(e) -> str:
    """Role + company + description of one experience entry as plain text.

    description has been both str (current schema) and List[str] (older
    parses persisted in the DB) — handle both; joining a str would
    character-split it."""
    if isinstance(e.description, list):
        desc = " ".join(e.description)
    else:
        desc = e.description or ""
    return " ".join(filter(None, [e.role, e.company, desc]))


def resume_full_text(resume) -> str:
    """Assemble the full evidence text of a parsed resume (skills, experience,
    education, qualifications). Used for both scoring and matched_terms so the
    UI pills agree with the score."""
    return " ".join(filter(None, [
        " ".join(resume.skills),
        " ".join(_experience_text(e) for e in resume.experience),
        " ".join(
            " ".join(filter(None, [ed.degree, ed.field_of_study, ed.institution]))
            for ed in resume.education
        ),
        " ".join(resume.qualifications),
    ]))


# ---------------------------------------------------------------------------
# Relevant years of experience
# ---------------------------------------------------------------------------

def _relevance_token_set(jd_reqs, jd_text: str, idf: dict[str, float]) -> set[str]:
    """Token forms (lemma + raw) that mark an experience entry as on-domain.

    Drawn from all JD phrases (skills, responsibilities, qualifications), keeping
    only tokens rare enough to be domain-specific (IDF ≥ EXP_REL_IDF; OOV counts
    as rare): 'die'/'casting'/'aluminum' qualify, 'engineer'/'machine' do not."""
    rel: set[str] = set()
    for phrase, _ in _all_jd_phrases(jd_reqs, jd_text):
        cleaned = _PUNCT.sub(" ", normalize_punct(phrase.lower()))
        if not cleaned.strip():
            continue
        for t in _get_nlp()(cleaned):
            if t.is_stop or t.is_space or len(t.text.strip()) <= 1:
                continue
            forms = {t.lemma_, t.text}
            if max(idf.get(x, EXP_REL_IDF) for x in forms) >= EXP_REL_IDF:
                rel |= forms
    return rel


def relevant_years(resume, jd_reqs, jd_raw_text: str) -> tuple[float, float]:
    """Years of experience in jobs relevant to the JD. Returns
    (relevant_years, total_years).

    An experience entry is relevant when its KG-expanded token forms (role +
    company + description) contain at least one domain-marker token from the
    JD (see _relevance_token_set). Years are computed with the same
    overlap-merging logic as total experience.

    Scaled fallback: when no individual entry carries domain evidence but the
    resume as a whole does (e.g. die-casting terms listed under skills only),
    relevant years = total years × simple keyword coverage of JD phrases.
    A fully off-domain resume gets coverage ≈ 0 and therefore ≈ 0 relevant years."""
    from Code.matching.experience import total_years

    idf = load_idf()
    total = total_years(resume.experience)
    if total == 0.0:
        return 0.0, 0.0

    rel_tokens = _relevance_token_set(jd_reqs, jd_raw_text, idf)
    if not rel_tokens:
        # No domain-marker basis in the JD — count everything.
        return total, total

    relevant = []
    for e in resume.experience:
        text = _experience_text(e)
        if not text.strip():
            continue
        if token_forms(expand_text(text)) & rel_tokens:
            relevant.append(e)

    if relevant:
        return total_years(relevant), total

    # Scaled fallback: no individual experience entry carries domain tokens, but
    # check if the resume as a whole has domain vocabulary (skills section, summary).
    # Scale total years by what fraction of JD phrases the resume covers (simple
    # keyword containment, not CE) — off-domain resumes get ≈ 0.
    phrases = _all_jd_phrases(jd_reqs, jd_raw_text)
    if not phrases:
        return 0.0, total

    resume_forms = token_forms(expand_text(resume_full_text(resume)))
    hit = sum(
        1 for p, _ in phrases
        if token_forms(p) & resume_forms
    )
    coverage = hit / len(phrases)
    return round(total * coverage, 1), total
