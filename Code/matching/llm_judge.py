"""Stage-2 reranker: a local LLM-as-judge over the top-K shortlist.

Runs a small instruct model in-process via llama-cpp-python (no daemon, no HTTP)
with JSON-schema-constrained output, so each judgement is a compact, reliable
rubric score. Prompts are built from the structured ResumeData / JDRequirements
(not raw documents) to keep token counts — and therefore latency — low.

Everything here is gated by JUDGE_ENABLED; when off or unavailable the pipeline
falls back to the deterministic cross-encoder path, so this module is opt-in and
carries no regression risk.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

_ROOT = Path(__file__).parent.parent.parent

# ── Config (env-overridable) ────────────────────────────────────────────────
def _env_bool(name: str, default: bool) -> bool:
    return (os.getenv(name) or str(default)).strip().lower() in {"1", "true", "yes", "on"}


JUDGE_ENABLED: bool = _env_bool("JUDGE_ENABLED", False)
JUDGE_MODEL_PATH: str = os.getenv("JUDGE_MODEL_PATH") or str(
    _ROOT / "models" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
)
JUDGE_TOP_K: int = int(os.getenv("JUDGE_TOP_K") or 50)
JUDGE_N_CTX: int = int(os.getenv("JUDGE_N_CTX") or 4096)
JUDGE_THREADS: int = int(os.getenv("JUDGE_THREADS") or (os.cpu_count() or 4))
JUDGE_MAX_TOKENS: int = int(os.getenv("JUDGE_MAX_TOKENS") or 512)
JUDGE_SEED: int = int(os.getenv("JUDGE_SEED") or 0)

_llm = None


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

class JudgeResult(BaseModel):
    skills: int = Field(ge=0, le=100)
    relevant_experience: int = Field(ge=0, le=100)
    qualifications: int = Field(ge=0, le=100)
    overall: int = Field(ge=0, le=100)
    matched: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @field_validator("skills", "relevant_experience", "qualifications", "overall", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            return max(0, min(100, int(round(float(v)))))
        except (TypeError, ValueError):
            return 0


# JSON schema handed to llama.cpp to constrain generation (kept flat — no $refs).
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {"type": "integer"},
        "relevant_experience": {"type": "integer"},
        "qualifications": {"type": "integer"},
        "overall": {"type": "integer"},
        "matched": {"type": "array", "items": {"type": "string"}},
        "missing": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": [
        "skills", "relevant_experience", "qualifications",
        "overall", "matched", "missing", "reasoning",
    ],
}


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """True when the judge is enabled and its model file exists on disk."""
    if not JUDGE_ENABLED:
        return False
    return Path(JUDGE_MODEL_PATH).exists()


def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama(
            model_path=JUDGE_MODEL_PATH,
            n_ctx=JUDGE_N_CTX,
            n_threads=JUDGE_THREADS,
            seed=JUDGE_SEED,
            verbose=False,
        )
    return _llm


# ---------------------------------------------------------------------------
# Prompt building (compact, structured)
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a strict technical recruiter scoring how well a candidate's resume "
    "fits one specific job. Judge ONLY on evidence present in the resume. A "
    "candidate from an unrelated field, or with no demonstrated experience in the "
    "job's domain, must score near 0 on overall and relevant_experience — do not "
    "give credit for generic professionalism. Reply with JSON only."
)


def _fmt_list(items, empty="(none)") -> str:
    items = [str(i).strip() for i in (items or []) if str(i).strip()]
    return ", ".join(items) if items else empty


def _fmt_experience(resume_data) -> str:
    lines = []
    for e in resume_data.experience or []:
        when = f"{e.start_date or '?'}–{'present' if e.is_current else (e.end_date or '?')}"
        head = " / ".join(filter(None, [e.role, e.company])) or "role"
        desc = e.description
        if isinstance(desc, list):
            desc = " ".join(desc)
        desc = (desc or "").strip()
        if len(desc) > 320:
            desc = desc[:320] + "…"
        lines.append(f"- {head} ({when}): {desc}" if desc else f"- {head} ({when})")
    return "\n".join(lines) if lines else "(none listed)"


def _fmt_education(resume_data) -> str:
    degs = []
    for ed in resume_data.education or []:
        degs.append(" ".join(filter(None, [ed.degree, ed.field_of_study])).strip())
    degs = [d for d in degs if d]
    return _fmt_list(degs)


def build_judge_prompt(resume_data, jd_reqs, jd_text: str) -> str:
    """Compact, structured prompt from JDRequirements + ResumeData."""
    jd_lines = ["JOB REQUIREMENTS"]
    if jd_reqs is not None:
        jd_lines += [
            f"Required skills: {_fmt_list(jd_reqs.required_skills)}",
            f"Preferred skills: {_fmt_list(jd_reqs.preferred_skills)}",
            f"Required qualifications: {_fmt_list(jd_reqs.required_qualifications)}",
            f"Preferred qualifications: {_fmt_list(jd_reqs.preferred_qualifications)}",
            f"Minimum years experience: {jd_reqs.min_years_experience if jd_reqs.min_years_experience is not None else 'unspecified'}",
            f"Required education: {jd_reqs.required_education_level or 'unspecified'}",
        ]
    else:
        snippet = (jd_text or "").strip()[:800]
        jd_lines.append(snippet)

    cand_lines = [
        "CANDIDATE",
        f"Skills: {_fmt_list(resume_data.skills)}",
        "Experience:",
        _fmt_experience(resume_data),
        f"Qualifications: {_fmt_list(resume_data.qualifications)}",
        f"Education: {_fmt_education(resume_data)}",
    ]

    rubric = (
        "Score each field 0-100 based strictly on the evidence:\n"
        "- skills: how well the candidate's skills cover the job's required/preferred skills\n"
        "- relevant_experience: how much of the candidate's experience is in THIS job's "
        "domain and at an appropriate level (unrelated-domain experience scores low)\n"
        "- qualifications: demonstrated qualifications vs the job's\n"
        "- overall: holistic fit for THIS job (unrelated field → near 0)\n"
        "matched: job requirements/skills clearly evidenced in the resume\n"
        "missing: job requirements/skills not evidenced\n"
        "reasoning: one or two sentences justifying the overall score."
    )

    return "\n".join(jd_lines) + "\n\n" + "\n".join(cand_lines) + "\n\n" + rubric


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge(resume_data, jd_reqs, jd_text: str) -> JudgeResult:
    """Score one resume against one JD with the local LLM. Raises on model error
    (callers gate on is_available() and fall back to the cross-encoder)."""
    llm = _get_llm()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": build_judge_prompt(resume_data, jd_reqs, jd_text)},
        ],
        response_format={"type": "json_object", "schema": _JUDGE_SCHEMA},
        temperature=0.0,
        seed=JUDGE_SEED,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    content = out["choices"][0]["message"]["content"] or "{}"
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        logging.warning("LLM judge returned non-JSON output; scoring 0.")
        return JudgeResult(skills=0, relevant_experience=0, qualifications=0, overall=0)
    return JudgeResult.model_validate(data)
