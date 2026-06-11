"""Resume Parser CLI.

Usage:
    python -m Code.run --jd <jd_file> <resume> [<resume> ...]

Arguments:
    --jd   path to the job description (PDF, DOCX, or TXT)
    resume  one or more resume files or directories (PDF/DOCX)

Env vars:
    LLM_PROVIDER   — groq | gemini | openrouter | ollama  (default: openrouter)
    TESSERACT_CMD  — path to tesseract.exe (Windows)
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from Code.db.repository import save_jd, save_resume
from Code.db.session import get_db, init_db
from Code.matching.bm25e import matched_terms, score_resume as bm25e_score
from Code.matching.education import check_certifications, meets_requirement
from Code.matching.experience import meets_min_experience, total_years
from Code.parser.extract import clean_extracted_text, extract_document_text_and_links
from Code.parser.prompts import build_resume_extraction_prompt
from Code.parser.providers import get_provider, load_or_extract_jd
from Code.scoring import build_score_breakdown, compute_ats_score


ARCHIVE_DIR    = ROOT / "Archive"
JD_CACHE_DIR   = ROOT / "JDCache"
JD_ARCHIVE_DIR = ARCHIVE_DIR / "Job Descriptions"


def get_unique_path(directory: Path, file_name: str) -> Path:
    candidate = directory / file_name
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    counter = 1
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def collect_resume_files(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in {".pdf", ".docx"}
            ))
        elif p.is_file() and p.suffix.lower() in {".pdf", ".docx"}:
            files.append(p)
    return files


def process_resume(
    source_path: Path,
    jd_text: str,
    jd_reqs,
    jd_db_id: int,
    provider,
    archive_dir: Path,
) -> None:
    start = time.perf_counter()

    # 1. Extract and clean resume text
    raw_text, links = extract_document_text_and_links(source_path)
    text = clean_extracted_text(raw_text)

    # 2. LLM extraction
    prompt = build_resume_extraction_prompt(text, links)
    resume_data = provider.extract_resume(prompt)

    # 3. Scoring
    match_score = bm25e_score(resume_data, jd_text)
    yrs = total_years(resume_data.experience)
    overall = compute_ats_score(match_score, yrs)

    exp_ok, _ = meets_min_experience(resume_data.experience, jd_reqs.min_years_experience)
    edu_ok, _ = meets_requirement(resume_data.education, jd_reqs.required_education_level or "")
    _, _, missing_certs = check_certifications(
        resume_data.certifications, jd_reqs.required_certifications or []
    )
    cert_ok = not missing_certs

    score_bd = build_score_breakdown(
        match_score=match_score,
        overall=overall,
        years_experience=yrs,
        meets_experience=exp_ok,
        education_met=edu_ok,
        certifications_met=cert_ok,
    )

    # 4. Matched / missing terms
    resume_text_for_terms = " ".join(filter(None, resume_data.skills + resume_data.qualifications))
    m_terms, miss_terms = matched_terms(resume_text_for_terms, jd_text)

    # 5. Archive path
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = get_unique_path(archive_dir, source_path.name)

    elapsed = round(time.perf_counter() - start, 3)

    # 6. Persist to DB
    with get_db() as session:
        save_resume(
            session,
            resume_data=resume_data,
            source_file=str(source_path),
            archive_file=str(archive_path),
            score_breakdown=score_bd,
            jd_id=jd_db_id,
            matched_skills=m_terms,
            missing_skills=miss_terms,
        )

    # 7. Move to Archive
    shutil.move(str(source_path), str(archive_path))

    name = (resume_data.candidate.full_name if resume_data.candidate else None) or source_path.stem
    print(f"  [done] {name}  ({elapsed}s, score={overall:.3f})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume Parser CLI")
    parser.add_argument("--jd", required=True, metavar="JD_FILE",
                        help="Path to the job description (PDF, DOCX, or TXT)")
    parser.add_argument("resumes", nargs="+", metavar="RESUME",
                        help="Resume files or directories (PDF/DOCX)")
    args = parser.parse_args()

    jd_path = Path(args.jd)
    if not jd_path.exists():
        sys.exit(f"JD file not found: {jd_path}")

    resume_files = collect_resume_files([Path(p) for p in args.resumes])
    if not resume_files:
        sys.exit("No PDF/DOCX resume files found in the given paths.")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    JD_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    init_db()
    provider = get_provider()

    jd_raw, _ = extract_document_text_and_links(jd_path)
    jd_text = clean_extracted_text(jd_raw)

    jd_reqs = load_or_extract_jd(
        str(jd_path),
        provider,
        archive_dir=JD_ARCHIVE_DIR,
        cache_dir=JD_CACHE_DIR,
    )

    with get_db() as session:
        jd_row = save_jd(session, name=jd_path.stem, file_path=str(jd_path), requirements=jd_reqs)
        jd_db_id = jd_row.id

    print(f"\nJD: {jd_path.name}  ({len(resume_files)} resume(s))", flush=True)

    processed_count = 0
    for source_path in resume_files:
        print(f"  Processing {source_path.name}...", end=" ", flush=True)
        try:
            process_resume(
                source_path=source_path,
                jd_text=jd_text,
                jd_reqs=jd_reqs,
                jd_db_id=jd_db_id,
                provider=provider,
                archive_dir=ARCHIVE_DIR,
            )
            processed_count += 1
        except Exception as exc:
            print(f"\n  [ERROR] {source_path.name}: {exc}", flush=True)

    print(f"\nDone. {processed_count} resume(s) processed.")


if __name__ == "__main__":
    main()
