"""ATS scoring: weighted combination of BM25+CE match score and years of
relevant experience."""

W_MATCH: float = 0.60
W_EXP: float = 0.40
EXP_CAP: float = 10.0  # years that map to experience score 1.0


def compute_ats_score(match_score: float, years_experience: float) -> float:
    """Return overall ATS score in [0, 1].

    overall = 0.60 * match_score + 0.40 * min(years / 10.0, 1.0)

    Callers pass *relevant* years (bm25e.relevant_years), so off-domain
    experience does not inflate the score.
    """
    exp_score = min(years_experience / EXP_CAP, 1.0)
    return round(W_MATCH * match_score + W_EXP * exp_score, 4)


def build_score_breakdown(
    match_score: float,
    overall: float,
    years_experience: float,
    relevant_years: float,
    meets_experience: bool,
    education_met: bool,
    certifications_met: bool,
) -> dict:
    return {
        "overall":            round(overall, 4),
        "match_score":        round(match_score, 4),
        "years_experience":   round(years_experience, 1),
        "relevant_years":     round(relevant_years, 1),
        "meets_experience":   meets_experience,
        "education_met":      education_met,
        "certifications_met": certifications_met,
    }
