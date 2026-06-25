"""Knowledge graph pre-pass for BM25+E scoring.

Loads pre-built synonym/expansion data from Data/kg_expansions.json (produced by
Scripts/build_kg.py from ESCO, O*NET, and ConceptNet sources).

expand_text() appends KG-derived synonyms and related terms to input text so
BM25 sees more matching tokens. Embeddings receive the original text unchanged
(they already handle semantics; expansion there adds noise).

normalize_punct() uses the same KG to decide whether punctuation in a term is
semantically meaningful (e.g. C++, .NET, Node.js) and replaces those terms with
canonical punct-free tokens before generic punctuation stripping. Any term that
appears as a key in the KG and contains +, #, . or - is treated as a meaningful
entity; all others are left to the generic punct-stripper.
"""
import logging
import re
from pathlib import Path

_expansions: dict[str, list[str]] | None = None
_KG_PATH = Path(__file__).parent.parent.parent / "Data" / "kg_expansions.json"
_PUNCT = re.compile(r"[^\w\s]")

# Caches for the punct-normalisation vocabulary (built lazily from the KG).
_PUNCT_VOCAB: dict[str, str] | None = None
_vocab_pattern: re.Pattern | None = None

# Characters whose presence in a KG key means the punctuation is meaningful.
_MEANINGFUL_PUNCT = re.compile(r"[+#.\-]")


def load_expansions() -> dict[str, list[str]]:
    global _expansions
    if _expansions is not None:
        return _expansions
    if not _KG_PATH.exists():
        logging.warning(
            "KG expansions not found at %s. Run Scripts/build_kg.py after downloading "
            "ESCO, O*NET, and ConceptNet sources. KG pre-pass disabled.",
            _KG_PATH,
        )
        _expansions = {}
        return _expansions
    import json
    with open(_KG_PATH, encoding="utf-8") as f:
        _expansions = json.load(f)
    for key, terms in _CURATED_EXPANSIONS.items():
        existing = _expansions.setdefault(key, [])
        existing[:0] = [t for t in terms if t not in existing]
    logging.info("Loaded KG expansions: %d terms", len(_expansions))
    return _expansions


# Curated domain abbreviations missing from ESCO/O*NET/ConceptNet. Merged into
# the loaded KG (appended to any existing expansions, never overwriting).
_CURATED_EXPANSIONS: dict[str, list[str]] = {
    # Die-casting abbreviations — include "aluminum die casting" directly so a
    # single expand_text pass is enough (no recursive re-expansion needed).
    "pdc": ["pressure die casting", "die casting", "aluminum die casting"],
    "hpdc": ["high pressure die casting", "die casting", "aluminum die casting"],
    "lpdc": ["low pressure die casting", "die casting", "aluminum die casting"],
    "gdc": ["gravity die casting", "die casting", "aluminum die casting"],
    # 3-gram key that fires for "high/low pressure die casting" resumes.
    "pressure die casting": ["aluminum die casting", "die casting"],
    # British ↔ American spelling
    "aluminium": ["aluminum"],
    "aluminium die casting": ["aluminum die casting"],
    # Fettling synonyms — "trimming press" is the machine used for fettling
    # (removing runners, gates, flash from castings) in die casting plants.
    # Both directions are needed:
    #   resume has "trimming press" → JD phrase "fettling" should match (forward: fettling→trimming press)
    #   JD has "fettling" → resume with "trimming press" should match (reverse, handled by expand_text on phrases)
    "fettling": ["trimming press", "trim press", "deburring", "flash removal"],
    "trimming press": ["fettling", "trim press", "deburring"],
    "trim press": ["fettling", "trimming press"],
    "fettling press": ["fettling", "trimming press"],
    "deflashing": ["fettling", "trimming press"],
    "flash removal": ["fettling", "deburring"],
    # Shot blasting synonyms
    "shot blasting": ["grit blasting", "abrasive blasting", "blast cleaning"],
    "shot blast": ["grit blasting", "abrasive blasting"],
    "grit blasting": ["shot blasting", "abrasive blasting"],
}


_MAX_EXPANSIONS_PER_TERM = 3


def kg_phrases(text: str) -> list[str]:
    """Return KG keys present in text via longest-match on 1–3 grams.

    'machine learning' is matched as a unit rather than 'machine' and
    'learning' separately; matched positions are consumed so shorter grams
    cannot overlap a longer match. Order follows scan order (3-grams first).
    """
    expansions = load_expansions()
    if not expansions:
        return []

    tokens = _PUNCT.sub(" ", text.lower()).split()
    if not tokens:
        return []

    found: list[str] = []
    matched_positions: set[int] = set()

    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            if any(j in matched_positions for j in range(i, i + n)):
                continue
            gram = " ".join(tokens[i : i + n])
            if gram in expansions:
                found.append(gram)
                matched_positions.update(range(i, i + n))

    return found


def expand_text(text: str, max_per_term: int = _MAX_EXPANSIONS_PER_TERM) -> str:
    """Append KG synonyms/related terms to text before BM25 tokenization.

    Uses kg_phrases() longest-match scanning. Caps expansions per matched
    term to prevent high-connectivity nodes (e.g. 'mechanical engineering') from
    flooding the token space with spurious manufacturing terms.
    """
    expansions = load_expansions()
    if not expansions:
        return text

    added: list[str] = []
    text_lower = text.lower()

    for gram in kg_phrases(text):
        count = 0
        for term in expansions[gram]:
            if count >= max_per_term:
                break
            if term.lower() not in text_lower:
                added.append(term)
                count += 1

    if added:
        deduped = list(dict.fromkeys(added))
        return text + " " + " ".join(deduped)
    return text


# ---------------------------------------------------------------------------
# Punct normalisation — KG-driven
# ---------------------------------------------------------------------------

def _build_punct_vocab(expansions: dict[str, list[str]]) -> dict[str, str]:
    """Return {kg_key: canonical_token} for every KG key that contains meaningful
    punctuation (+, #, ., -).

    Canonical form: strip all non-word characters and join.
    If that collapses to a single character (e.g. 'c' from 'c++'), fall back to
    the shortest expansion whose own canonical form is ≥ 2 characters.
    """
    _strip = re.compile(r"\W")   # removes punct, spaces, everything non-word
    vocab: dict[str, str] = {}

    for key, exps in expansions.items():
        if not _MEANINGFUL_PUNCT.search(key):
            continue

        canonical = _strip.sub("", key)
        if len(canonical) >= 2:
            vocab[key] = canonical
            continue

        # Canonical collapsed to ≤ 1 char — find shortest usable expansion.
        best: str | None = None
        for exp in sorted(exps, key=lambda e: len(_strip.sub("", e))):
            exp_can = _strip.sub("", exp)
            if len(exp_can) >= 2:
                best = exp_can
                break
        if best:
            vocab[key] = best

    return vocab


def _load_vocab() -> tuple[dict[str, str], re.Pattern | None]:
    """Lazy-load and cache the punct vocab and its compiled regex."""
    global _PUNCT_VOCAB, _vocab_pattern
    if _PUNCT_VOCAB is None:
        _PUNCT_VOCAB = _build_punct_vocab(load_expansions())
        if _PUNCT_VOCAB:
            keys = sorted(_PUNCT_VOCAB.keys(), key=len, reverse=True)
            _vocab_pattern = re.compile("|".join(re.escape(k) for k in keys))
        else:
            _vocab_pattern = None
    return _PUNCT_VOCAB, _vocab_pattern


def normalize_punct(text: str) -> str:
    """Replace KG-known punct-bearing terms with canonical (punct-free) tokens.

    Applied *before* generic punctuation stripping so that 'C++' is not silently
    reduced to the single letter 'c', 'Node.js' is not split into 'node' and 'js',
    etc.  Only terms that appear as keys in the KG are touched; everything else is
    left for the caller's punct-stripper.
    """
    vocab, pattern = _load_vocab()
    if not pattern:
        return text
    return pattern.sub(lambda m: vocab[m.group(0)], text)
