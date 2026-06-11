import re
from typing import NamedTuple

from rapidfuzz import fuzz

NAME_THRESHOLD = 92  # high to avoid flagging common names that belong to different people


class DuplicateMatch(NamedTuple):
    source_file: str
    matched_on: list[str]   # subset of ["email", "phone", "name"]
    existing_name: str


def _normalize_phone(phone: str | None) -> str | None:
    if not phone:
        return None
    digits = re.sub(r"[^\d]", "", phone)
    if len(digits) == 11 and digits.startswith("1"):   # US/Canada +1
        digits = digits[1:]
    elif len(digits) == 12 and digits.startswith("91"):  # India +91
        digits = digits[2:]
    return digits if len(digits) >= 7 else None


def find_duplicates(
    name: str | None,
    phone: str | None,
    email: str | None,
    existing: list[dict],
) -> list[DuplicateMatch]:
    """Detect whether an incoming candidate matches any previously seen record.

    existing: list of dicts with keys source_file, name, phone, email.
    Returns potential duplicates sorted by match strength (most fields matched first).
    Email and phone are strong signals; name alone is weak and only flagged when
    neither email nor phone matched (to avoid false positives on common names).
    """
    norm_phone = _normalize_phone(phone)
    norm_email = email.strip().lower() if email else None

    results: list[DuplicateMatch] = []
    for record in existing:
        matched_on: list[str] = []

        if norm_email and record.get("email"):
            if record["email"].strip().lower() == norm_email:
                matched_on.append("email")

        if norm_phone and record.get("phone"):
            if _normalize_phone(record.get("phone")) == norm_phone:
                matched_on.append("phone")

        # Name-only check: only run when stronger signals didn't already fire,
        # because fuzzy name matching produces false positives on common names.
        if not matched_on and name and record.get("name"):
            score = fuzz.token_sort_ratio(
                name.strip().lower(), record["name"].strip().lower()
            )
            if score >= NAME_THRESHOLD:
                matched_on.append("name")

        if matched_on:
            results.append(DuplicateMatch(
                source_file=record["source_file"],
                matched_on=matched_on,
                existing_name=record.get("name") or "Unknown",
            ))

    results.sort(key=lambda m: len(m.matched_on), reverse=True)
    return results
