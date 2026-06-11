import re
from typing import Optional

DEGREE_LEVELS: dict[str, int] = {
    "high school": 1,
    "associate":   2,
    "bachelor":    3,
    "master":      4,
    "phd":         5,
}

_PHD_KW        = ("phd", "ph.d", "doctor", "doctorate")
_MASTER_KW     = ("master", "msc", "m.s", "mba", "m.eng", "m.tech", "mtech", "m.e")
_BACHELOR_KW   = ("bachelor", "b.s", "b.a", "b.eng", "b.tech", "b.sc",
                   "undergraduate", "btech", "b.e", "honours")
_ASSOCIATE_KW  = ("associate", "a.a", "a.s")
_HIGHSCHOOL_KW = ("high school", "diploma", "ged", "secondary",
                   "matriculation", "10th", "12th", "hsc", "ssc")

_TIERS = [
    (5, _PHD_KW),
    (4, _MASTER_KW),
    (3, _BACHELOR_KW),
    (2, _ASSOCIATE_KW),
    (1, _HIGHSCHOOL_KW),
]

# Collapse "B. Tech." → "b.tech." so spaced abbreviations match keywords
_SPACE_DOT = re.compile(r'\.\s+')


def _degree_level(degree_str: str) -> Optional[int]:
    low = _SPACE_DOT.sub(".", degree_str.strip().lower())
    for level, keywords in _TIERS:
        if any(kw in low for kw in keywords):
            return level
    return None


def meets_requirement(
    education_entries,
    required_level: Optional[str],
) -> tuple[bool, str]:
    if required_level is None:
        return True, "No education requirement specified"

    req_int = DEGREE_LEVELS.get(required_level.strip().lower())
    if req_int is None:
        return True, f"Unrecognized required level '{required_level}' — skipping check"

    best_level: Optional[int] = None
    best_label: Optional[str] = None
    for edu in education_entries:
        if not edu.degree:
            continue
        lvl = _degree_level(edu.degree)
        if lvl is not None and (best_level is None or lvl > best_level):
            best_level = lvl
            best_label = edu.degree

    if best_level is None:
        return False, "No parseable degree found"
    if best_level >= req_int:
        return True, f"'{best_label}' meets '{required_level}' requirement"
    return False, f"Highest degree '{best_label}' is below required '{required_level}'"


def check_certifications(
    resume_certs: list[str],
    required_certs: list[str],
) -> tuple[bool, list[str], list[str]]:
    """Check required certifications against resume certifications.

    Returns (all_met, matched_list, missing_list).
    Uses case-insensitive substring matching — 'ISO 9001' matches
    'ISO 9001:2015 Quality Management System'.
    """
    if not required_certs:
        return True, [], []

    resume_lower = [c.lower() for c in resume_certs]
    matched, missing = [], []

    for req in required_certs:
        req_lower = req.lower()
        # Match if either is a substring of the other
        found = any(
            req_lower in rc or rc in req_lower
            for rc in resume_lower
        )
        (matched if found else missing).append(req)

    return len(missing) == 0, matched, missing
