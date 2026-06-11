from datetime import date
from typing import Optional


def _parse_date(s: str) -> date:
    parts = s.split("-")
    return date(int(parts[0]), int(parts[1]) if len(parts) > 1 else 1, 1)


def total_years(experiences) -> float:
    today = date.today()
    intervals: list[tuple[date, date]] = []

    for exp in experiences:
        if not exp.start_date:
            continue
        try:
            start = _parse_date(exp.start_date)
        except (ValueError, AttributeError):
            continue
        if exp.is_current or not exp.end_date:
            end = today
        else:
            try:
                end = _parse_date(exp.end_date)
            except (ValueError, AttributeError):
                end = today
        if end < start:
            end = start
        intervals.append((start, end))

    if not intervals:
        return 0.0

    intervals.sort(key=lambda x: x[0])
    merged: list[list[date]] = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    total_months = sum(
        (e.year - s.year) * 12 + (e.month - s.month)
        for s, e in merged
    )
    return round(total_months / 12, 1)


def meets_min_experience(experiences, min_years: Optional[float]) -> tuple[bool, str]:
    if min_years is None:
        return True, "No minimum specified"
    years = total_years(experiences)
    msg = f"{years} years of experience (required: {min_years})"
    return years >= min_years, msg
