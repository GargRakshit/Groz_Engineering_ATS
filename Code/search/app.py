"""FastAPI search UI for Resume Parser."""
from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from sqlalchemy import text
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv()

from Code.db.models import JobDescription, Match, Resume
from Code.db.repository import (
    get_all_jds,
    get_candidates_for_dedup,
    get_resume_by_id,
    save_jd,
    save_resume,
    search_resumes,
)
from Code.db.models import User
from Code.db.session import check_password, get_db, init_db
from Code.matching.bm25e import matched_terms, score_resume as bm25e_score
from Code.matching.duplicate import find_duplicates
from Code.matching.education import check_certifications, meets_requirement
from Code.matching.experience import meets_min_experience, total_years
from Code.parser.extract import clean_extracted_text, extract_document_text_and_links
from Code.parser.prompts import build_jd_extraction_prompt, build_resume_extraction_prompt
from Code.parser.providers import get_provider
from Code.parser.schemas import JDRequirements, ResumeData
from Code.scoring import build_score_breakdown, compute_ats_score

ROOT = Path(__file__).parent.parent.parent
ARCHIVE_DIR  = ROOT / "Archive"
JD_CACHE_DIR = ROOT / "JDCache"
IMAGES_DIR = Path(__file__).parent / "images"

_PUBLIC_PATHS = {"/login", "/images"}

# Inactivity timeout: session expires after this many seconds with no requests.
# The same value is set as the cookie max_age so browser-restored cookies also
# expire — no reliance on the browser clearing session cookies on close.
SESSION_TTL = 15 * 60  # 15 minutes

SESSION_COOKIE = "rp_session"

# token → {"username": str, "last_seen": float}
_active_sessions: dict[str, dict] = {}
# username → token (enforces one active session per user)
_user_tokens: dict[str, str] = {}


def _is_public_path(path: str) -> bool:
    return any(path == public or path.startswith(f"{public}/") for public in _PUBLIC_PATHS)


def _cleanup_expired() -> None:
    """Remove sessions whose inactivity window has elapsed."""
    now = time.time()
    expired = [t for t, s in _active_sessions.items() if now - s["last_seen"] > SESSION_TTL]
    for token in expired:
        username = _active_sessions.pop(token, {}).get("username")
        if username and _user_tokens.get(username) == token:
            _user_tokens.pop(username, None)


def _is_authenticated(request: Request) -> bool:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return False
    s = _active_sessions.get(token)
    if not s:
        return False
    if time.time() - s["last_seen"] > SESSION_TTL:
        # Expired — clean up lazily
        username = _active_sessions.pop(token, {}).get("username")
        if username and _user_tokens.get(username) == token:
            _user_tokens.pop(username, None)
        return False
    return True


class _AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = request.cookies.get(SESSION_COOKIE)
        s = _active_sessions.get(token) if token else None

        # Treat expired session as unauthenticated
        if s and time.time() - s["last_seen"] > SESSION_TTL:
            username = _active_sessions.pop(token, {}).get("username")
            if username and _user_tokens.get(username) == token:
                _user_tokens.pop(username, None)
            s = None

        if not _is_public_path(request.url.path) and not s:
            accept = request.headers.get("accept", "")
            if "text/html" in accept:
                return RedirectResponse("/login", status_code=303)
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        response = await call_next(request)

        # Slide the expiry window on every authenticated request
        if s and token:
            s["last_seen"] = time.time()
            response.set_cookie(
                SESSION_COOKIE, token,
                httponly=True, samesite="lax",
                max_age=SESSION_TTL,
            )

        return response


app = FastAPI(title="Resume Search")
app.add_middleware(_AuthMiddleware)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ── Jinja2 helpers for avatar rendering ──
_AVATAR_PALETTE = [
    ("#dbeafe", "#1d4ed8"),  # blue
    ("#d1fae5", "#065f46"),  # green
    ("#fef3c7", "#92400e"),  # amber
    ("#fce7f3", "#9d174d"),  # pink
    ("#ede9fe", "#5b21b6"),  # purple
    ("#fee2e2", "#991b1b"),  # red
    ("#e0f2fe", "#0369a1"),  # sky
    ("#fef9c3", "#854d0e"),  # yellow
]

def _initials(name: str) -> str:
    parts = (name or "?").strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return (parts[0][:2] if len(parts[0]) >= 2 else parts[0][0]).upper()

def _avatar_color(name: str) -> tuple[str, str]:
    h = sum(ord(c) for c in (name or ""))
    return _AVATAR_PALETTE[h % len(_AVATAR_PALETTE)]

templates.env.filters["initials"] = _initials
templates.env.filters["avatar_color"] = _avatar_color


@app.on_event("startup")
def startup() -> None:
    init_db()
    from Code.matching.bm25e import _get_ce, _get_nlp
    _get_nlp()
    _get_ce()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_file(resume: Resume) -> Optional[Path]:
    """Locate the physical file for a resume (archive_file or source fallback)."""
    if resume.archive_file:
        p = Path(resume.archive_file)
        if p.exists():
            return p
    if resume.source_file:
        # Files processed by run.py are moved to Archive/<original_name>
        src = Path(resume.source_file)
        candidate = ARCHIVE_DIR / src.name
        if candidate.exists():
            return candidate
    return None


def _get_unique_path(directory: Path, filename: str) -> Path:
    dest = directory / filename
    if not dest.exists():
        return dest
    stem, suffix = Path(filename).stem, Path(filename).suffix
    i = 1
    while True:
        dest = directory / f"{stem}_{i}{suffix}"
        if not dest.exists():
            return dest
        i += 1


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    if _is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None, "username": ""})


@app.post("/login", response_model=None)
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    force: str = Form(""),
) -> HTMLResponse | RedirectResponse:
    with get_db() as session:
        user = session.query(User).filter_by(username=username).first()

    if not user or not check_password(password, user.password_hash):
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "invalid_credentials", "username": username},
            status_code=401,
        )

    # Clean up any expired sessions before checking concurrency.
    _cleanup_expired()

    # Block concurrent logins unless the user explicitly forces a new session.
    if username in _user_tokens and force != "1":
        return templates.TemplateResponse(
            request, "login.html",
            {"error": "already_logged_in", "username": username},
            status_code=200,
        )

    # Evict old session for this user (force login or first login).
    if username in _user_tokens:
        _active_sessions.pop(_user_tokens[username], None)

    token = secrets.token_urlsafe(32)
    _active_sessions[token] = {"username": username, "last_seen": time.time()}
    _user_tokens[username] = token

    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax", max_age=SESSION_TTL)
    return response


@app.get("/logout")
async def logout(request: Request) -> RedirectResponse:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        username = _active_sessions.pop(token, {}).get("username")
        if username:
            _user_tokens.pop(username, None)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, jd_id: Optional[int] = None) -> HTMLResponse:
    with get_db() as session:
        jds = get_all_jds(session)
        results = _load_results(session, jd_id)

    return templates.TemplateResponse(request, "index.html", {
        "jds": jds,
        "results": results,
        "current_jd_id": jd_id,
    })


def _load_results(session, jd_id: Optional[int], ids: Optional[list[int]] = None) -> list[dict]:
    """Load all resumes with per-JD scores attached."""
    results = search_resumes(session, jd_id=jd_id, ids=ids)

    if results:
        resume_ids = [r["id"] for r in results]
        all_matches = (
            session.query(Match)
            .filter(Match.resume_id.in_(resume_ids))
            .all()
        )
        scores_by_resume: dict[int, dict] = {}
        for m in all_matches:
            scores_by_resume.setdefault(m.resume_id, {})[str(m.jd_id)] = round(m.ats_score, 4)

        for r in results:
            jd_scores = scores_by_resume.get(r["id"], {})
            best = max(jd_scores.values()) if jd_scores else None
            r["all_scores"] = {**jd_scores, "best": best}

    return results


@app.get("/resume/{resume_id}", response_class=HTMLResponse)
async def resume_detail(
    request: Request,
    resume_id: int,
    jd_id: Optional[int] = None,
) -> HTMLResponse:
    with get_db() as session:
        data = get_resume_by_id(session, resume_id, jd_id=jd_id)

    if data is None:
        return HTMLResponse(
            content="<h1>404 — Resume not found</h1><p><a href='/'>Back to search</a></p>",
            status_code=404,
        )

    return templates.TemplateResponse(request, "resume.html", {
        "data": data,
        "back_url": "/",
    })


@app.get("/file/{resume_id}")
async def serve_file(resume_id: int) -> FileResponse:
    with get_db() as session:
        resume = session.query(Resume).filter_by(id=resume_id).first()
        if resume is None:
            return HTMLResponse(content="Not found", status_code=404)
        file_path = _find_file(resume)

    if file_path is None:
        return HTMLResponse(content="File not available", status_code=404)

    suffix = file_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    # Serve inline so browsers render PDFs in-place (iframe viewer).
    # DOCX falls back gracefully — browser offers a download prompt.
    response = FileResponse(path=str(file_path), media_type=media_type)
    response.headers["Content-Disposition"] = f"inline; filename*=utf-8''{file_path.name}"
    return response


_VALID_STATUSES = {None, "shortlisted", "interviewed", "selected", "rejected"}


class StatusUpdate(BaseModel):
    status: Optional[str] = None


@app.patch("/resumes/{resume_id}/status")
async def update_resume_status(resume_id: int, body: StatusUpdate) -> JSONResponse:
    if body.status not in _VALID_STATUSES:
        return JSONResponse({"error": "invalid status"}, status_code=400)
    with get_db() as session:
        resume = session.query(Resume).filter_by(id=resume_id).first()
        if not resume:
            return JSONResponse({"error": "not found"}, status_code=404)
        resume.status = body.status
        session.commit()
    return JSONResponse({"status": body.status})


@app.post("/upload")
async def upload_resumes(
    files: list[UploadFile] = File(...),
    jd_id: Optional[int] = Form(None),
) -> JSONResponse:
    results = []

    for upload in files:
        result: dict = {"filename": upload.filename or "unknown"}
        try:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in {".pdf", ".docx"}:
                result["status"] = "error"
                result["error"] = f"Unsupported format '{suffix}'. Only PDF and DOCX are accepted."
                results.append(result)
                continue

            # Save to Archive/
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            dest_path = _get_unique_path(ARCHIVE_DIR, upload.filename)
            dest_path.write_bytes(await upload.read())

            # Extract and clean text
            raw_text, links = extract_document_text_and_links(dest_path)
            text = clean_extracted_text(raw_text)

            # LLM parse
            provider = get_provider()
            prompt = build_resume_extraction_prompt(text, links)
            resume_data = provider.extract_resume(prompt)

            cand = resume_data.candidate

            # Duplicate check (against existing DB records)
            with get_db() as session:
                existing = get_candidates_for_dedup(session)

            dupes = find_duplicates(
                name=cand.full_name if cand else None,
                phone=cand.phone if cand else None,
                email=cand.email if cand else None,
                existing=existing,
            )

            # Save to DB first (scoring happens below)
            with get_db() as session:
                saved = save_resume(
                    session,
                    resume_data=resume_data,
                    source_file=str(dest_path),
                    archive_file=str(dest_path),
                )

            # Score against every JD in the system
            with get_db() as session:
                jd_rows = [
                    (j.id, j.file_path, j.requirements_json)
                    for j in get_all_jds(session)
                ]

            for _jd_id, _jd_path, _jd_req_json in jd_rows:
                if not Path(_jd_path).exists():
                    continue
                try:
                    jd_raw, _ = extract_document_text_and_links(Path(_jd_path))
                    jd_text = clean_extracted_text(jd_raw)
                    jd_reqs = JDRequirements.model_validate_json(_jd_req_json) if _jd_req_json else None

                    match_score = bm25e_score(resume_data, jd_text)
                    yrs = total_years(resume_data.experience)
                    overall = compute_ats_score(match_score, yrs)
                    exp_ok, _ = meets_min_experience(
                        resume_data.experience,
                        jd_reqs.min_years_experience if jd_reqs else None,
                    )
                    edu_ok, _ = meets_requirement(
                        resume_data.education,
                        (jd_reqs.required_education_level or "") if jd_reqs else "",
                    )
                    _, _, missing_certs = check_certifications(
                        resume_data.certifications,
                        (jd_reqs.required_certifications or []) if jd_reqs else [],
                    )
                    score_bd = build_score_breakdown(
                        match_score=match_score, overall=overall, years_experience=yrs,
                        meets_experience=exp_ok, education_met=edu_ok,
                        certifications_met=not missing_certs,
                    )
                    terms = " ".join(filter(None, resume_data.skills + resume_data.qualifications))
                    matched_sk, missing_sk = matched_terms(terms, jd_text)

                    with get_db() as session:
                        m = session.query(Match).filter_by(resume_id=saved.id, jd_id=_jd_id).first()
                        if m is None:
                            m = Match(resume_id=saved.id, jd_id=_jd_id)
                            session.add(m)
                        m.ats_score = overall
                        m.score_breakdown_json = json.dumps(score_bd)
                        m.matched_skills_json = json.dumps(matched_sk)
                        m.missing_skills_json = json.dumps(missing_sk)
                        session.commit()
                except Exception:
                    pass

            result.update({
                "status": "added",
                "candidate_name": cand.full_name if cand else upload.filename,
                "resume_id": saved.id,
                "duplicates": [
                    {
                        "name": d.existing_name,
                        "matched_on": d.matched_on,
                    }
                    for d in dupes
                ],
            })

        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            # Clean up partial file if saved
            try:
                if "dest_path" in dir() and dest_path.exists():
                    dest_path.unlink()
            except Exception:
                pass

        results.append(result)

    return JSONResponse(content={"results": results})


class _DeleteRequest(BaseModel):
    ids: list[int]


def _unlink(p: Optional[Path]) -> None:
    try:
        if p and p.exists():
            p.unlink()
    except Exception:
        pass


def _find_under(directory: Path, name: str) -> Optional[Path]:
    """Return the first file matching `name` anywhere under `directory`."""
    try:
        return next(directory.rglob(name), None)
    except Exception:
        return None


@app.delete("/resumes")
async def delete_resumes(req: _DeleteRequest) -> JSONResponse:
    deleted: list[int] = []
    with get_db() as session:
        for rid in req.ids:
            resume = session.query(Resume).filter_by(id=rid).first()
            if not resume:
                continue

            source  = Path(resume.source_file)  if resume.source_file  else None
            archive = Path(resume.archive_file)  if resume.archive_file else (
                _find_under(ARCHIVE_DIR, source.name) if source else None
            )
            # Direct SQL delete — PRAGMA foreign_keys=ON cascades to all child tables
            # and the resume_bd FTS trigger removes the FTS entry automatically
            session.execute(text("DELETE FROM resume WHERE id = :rid"), {"rid": rid})
            deleted.append(rid)

            _unlink(archive)
            if source and source != archive:
                _unlink(source)
        session.commit()
    return JSONResponse(content={"deleted": deleted})


class _DeleteJDRequest(BaseModel):
    ids: list[int]


@app.delete("/jds")
async def delete_jds(req: _DeleteJDRequest) -> JSONResponse:
    deleted: list[int] = []
    with get_db() as session:
        for jid in req.ids:
            jd = session.query(JobDescription).filter_by(id=jid).first()
            if not jd:
                continue

            jd_file  = Path(jd.file_path) if jd.file_path else None
            jd_cache = JD_CACHE_DIR / f"{jd_file.name}.jdcache" if jd_file else None

            # Direct SQL delete — FK cascade removes all Match records for this JD
            session.execute(text("DELETE FROM job_description WHERE id = :jid"), {"jid": jid})
            deleted.append(jid)

            _unlink(jd_file)
            _unlink(jd_cache)
        session.commit()
    return JSONResponse(content={"deleted": deleted})


# ---------------------------------------------------------------------------
# Card data for dynamic DOM injection
# ---------------------------------------------------------------------------

@app.get("/resumes/cards")
async def get_resume_cards(ids: str = "") -> JSONResponse:
    """Return card data for comma-separated resume IDs (used after upload to inject new cards)."""
    id_list = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    if not id_list:
        return JSONResponse(content={"results": []})
    with get_db() as session:
        results = _load_results(session, jd_id=None, ids=id_list)
    return JSONResponse(content={"results": results})


# ---------------------------------------------------------------------------
# Re-score selected resumes against a JD
# ---------------------------------------------------------------------------

class _RescoreRequest(BaseModel):
    ids: list[int]
    jd_id: int


@app.post("/resumes/rescore")
async def rescore_resumes(req: _RescoreRequest) -> JSONResponse:
    scored = []
    with get_db() as session:
        jd_row = session.query(JobDescription).filter_by(id=req.jd_id).first()
        if not jd_row:
            return JSONResponse(content={"error": "JD not found"}, status_code=404)
        if not Path(jd_row.file_path).exists():
            return JSONResponse(content={"error": "JD file not found on disk"}, status_code=404)

        jd_raw, _ = extract_document_text_and_links(Path(jd_row.file_path))
        jd_text = clean_extracted_text(jd_raw)
        jd_reqs = JDRequirements.model_validate_json(jd_row.requirements_json) if jd_row.requirements_json else None

        for rid in req.ids:
            resume = session.query(Resume).filter_by(id=rid).first()
            if not resume or not resume.parsed_json:
                continue
            try:
                resume_data = ResumeData.model_validate_json(resume.parsed_json)
                match_score = bm25e_score(resume_data, jd_text)
                yrs = total_years(resume_data.experience)
                overall = compute_ats_score(match_score, yrs)
                exp_ok, _ = meets_min_experience(
                    resume_data.experience,
                    jd_reqs.min_years_experience if jd_reqs else None,
                )
                edu_ok, _ = meets_requirement(
                    resume_data.education,
                    (jd_reqs.required_education_level or "") if jd_reqs else "",
                )
                _, _, missing_certs = check_certifications(
                    resume_data.certifications,
                    (jd_reqs.required_certifications or []) if jd_reqs else [],
                )
                score_bd = build_score_breakdown(
                    match_score=match_score, overall=overall, years_experience=yrs,
                    meets_experience=exp_ok, education_met=edu_ok,
                    certifications_met=not missing_certs,
                )
                terms = " ".join(filter(None, resume_data.skills + resume_data.qualifications))
                matched, missing = matched_terms(terms, jd_text)

                m = session.query(Match).filter_by(resume_id=rid, jd_id=req.jd_id).first()
                if m is None:
                    m = Match(resume_id=rid, jd_id=req.jd_id)
                    session.add(m)
                m.ats_score = overall
                m.score_breakdown_json = json.dumps(score_bd)
                m.matched_skills_json = json.dumps(matched)
                m.missing_skills_json = json.dumps(missing)
                scored.append({"resume_id": rid, "score": round(overall, 4)})
            except Exception:
                pass
        session.commit()
    return JSONResponse(content={"results": scored})


# ---------------------------------------------------------------------------
# Add a new Job Description + auto-score recent resumes
# ---------------------------------------------------------------------------

@app.post("/jd")
async def add_jd(
    file: UploadFile = File(...),
    name: str = Form(""),
) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        return JSONResponse(
            content={"error": f"Unsupported format '{suffix}'. Use PDF, DOCX, or TXT."},
            status_code=400,
        )

    jd_dir = ARCHIVE_DIR / "Job Descriptions"
    jd_dir.mkdir(parents=True, exist_ok=True)
    dest = _get_unique_path(jd_dir, file.filename or f"jd{suffix}")
    dest.write_bytes(await file.read())

    jd_name = name.strip() or dest.stem

    jd_raw, _ = extract_document_text_and_links(dest)
    jd_text = clean_extracted_text(jd_raw)

    try:
        provider = get_provider()
        prompt = build_jd_extraction_prompt(jd_text)
        jd_reqs: Optional[JDRequirements] = provider.extract_jd(prompt)
    except Exception:
        jd_reqs = None

    with get_db() as session:
        jd_row = save_jd(session, name=jd_name, file_path=str(dest), requirements=jd_reqs)
        jd_id = jd_row.id

    cutoff = datetime.utcnow() - timedelta(days=183)
    scored_count = 0
    with get_db() as session:
        recent = session.query(Resume).filter(Resume.created_at >= cutoff).all()
        for resume in recent:
            if not resume.parsed_json:
                continue
            try:
                resume_data = ResumeData.model_validate_json(resume.parsed_json)
                match_score = bm25e_score(resume_data, jd_text)
                yrs = total_years(resume_data.experience)
                overall = compute_ats_score(match_score, yrs)
                exp_ok, _ = meets_min_experience(
                    resume_data.experience,
                    jd_reqs.min_years_experience if jd_reqs else None,
                )
                edu_ok, _ = meets_requirement(
                    resume_data.education,
                    (jd_reqs.required_education_level or "") if jd_reqs else "",
                )
                _, _, missing_certs = check_certifications(
                    resume_data.certifications,
                    (jd_reqs.required_certifications or []) if jd_reqs else [],
                )
                score_bd = build_score_breakdown(
                    match_score=match_score, overall=overall, years_experience=yrs,
                    meets_experience=exp_ok, education_met=edu_ok,
                    certifications_met=not missing_certs,
                )
                terms = " ".join(filter(None, resume_data.skills + resume_data.qualifications))
                matched, missing = matched_terms(terms, jd_text)

                m = session.query(Match).filter_by(resume_id=resume.id, jd_id=jd_id).first()
                if m is None:
                    m = Match(resume_id=resume.id, jd_id=jd_id)
                    session.add(m)
                m.ats_score = overall
                m.score_breakdown_json = json.dumps(score_bd)
                m.matched_skills_json = json.dumps(matched)
                m.missing_skills_json = json.dumps(missing)
                scored_count += 1
            except Exception:
                pass
        session.commit()

    return JSONResponse(content={"jd_id": jd_id, "name": jd_name, "scored_count": scored_count})
