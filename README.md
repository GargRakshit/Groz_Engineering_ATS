# Resume Parser — Application Tracking System

A self-hosted applicant tracking system that parses resumes and job descriptions, scores candidates against open roles, and surfaces ranked results through a web UI — all running locally with no cloud dependency at runtime.

> **Context:** This project was developed during an internship at [Groz-Engineering-Tools], a tools manufacturer. The Groz branding visible in the UI (logo, orange accent colour) reflects that origin. The codebase is shared publicly as a portfolio piece and as a reference for others building similar tools.

---

## Work in Progress

This project was built in stages, moving progressively away from LLM-heavy processing toward fully deterministic, API-free scoring.

**Where it started:** The original system had the LLM do everything — extract resume data, match skills, score candidates, and justify the match. This was fast to build but non-deterministic, expensive, and hard to audit.

**Where it is now:** The LLM is responsible for extraction only — one call per resume, one call per job description. All matching, scoring, and ranking is done in Python with a per-phrase Cross-Encoder pipeline that produces deterministic results.

**Where it is going:** Eliminate the LLM dependency for extraction too, replacing it with a local pipeline. The goal is a system that works fully offline with zero API calls.

| Stage | Status | What the LLM does |
|---|---|---|
| 1 — LLM does everything | ✅ Done (legacy, deleted) | Extract + match + score |
| 2 — LLM does extraction only | ✅ Current | Extract structured data from text |
| 3 — Local NLP extraction | 🔲 Planned | Nothing — fully offline |

---

## Features

- **Resume parsing** — Extracts candidate info, skills, experience, education, certifications, and qualifications from PDF and DOCX files
- **JD parsing** — Extracts required/preferred skills, key responsibilities, experience requirements, education level, and certifications from job description documents
- **Per-phrase CE scoring** — Every JD requirement phrase is scored independently against the full resume text via a batched cross-encoder call (`gte-reranker-modernbert-base`, 8192-token context). Score and matched/missing pills are derived from the same computation, so they always agree.
- **Calibrated scoring** — Raw CE sigmoid is linearly mapped to [0, 1] over the empirical output band [0.65, 0.93]. A term-match fallback raises scores for domain synonyms the CE misses (e.g. "Fettling" → resume says "trimming press").
- **Relevant-years experience** — Only experience in roles on the JD's domain counts toward the score, detected via IDF-filtered domain tokens and KG expansion.
- **Knowledge-graph expansion** — 117k synonym entries (ESCO + O\*NET + ConceptNet) used for experience relevance detection and the term-match fallback.
- **ATS score formula** — `0.60 × match_score + 0.40 × min(relevant_years / 10, 1.0)`
- **Web UI** — Candidate cards, filters (name, score, years, status, JD), upload overlay, JD manager, per-resume detail view, dark/light mode
- **Multi-JD upload** — Upload several JD files at once; each gets its own positions count set during upload
- **Vacancy tracking** — Each JD tracks open positions; marking a candidate "Selected" auto-decrements the vacancy count; badge shows "Fulfilled" when all seats are filled. Positions are editable at any time from the JD manager.
- **Per-JD candidate status** — Shortlisted / Interviewed / Selected / Rejected is tracked per JD (not globally), so the same candidate can be at different stages across open roles. Status dropdown is disabled until a JD is selected.
- **View original files** — "View JD" button on each JD in the manager; "View Resume" button on every candidate's score report page.
- **Multi-provider LLM** — Groq, Gemini, OpenRouter, Ollama — switch with a single env var
- **Session auth** — Login required; one active session per user; 15-minute inactivity timeout

---

## Architecture

```
Resume PDF/DOCX ──► extract text ──► LLM (extraction only) ──► ResumeData (Pydantic)
                                                                        │
JD PDF/DOCX ──────► extract text ──► LLM (cached by SHA-256) ──► JDRequirements
                                                                        │
                     ┌──────── matching/scoring (pure Python) ─────────┘
                     │  Per-phrase Cross-Encoder (gte-reranker-modernbert-base)
                     │  Unified phrase pool: skills + responsibilities + qualifications
                     │  CE calibration · term-match fallback · relevant-years calc
                     └──────────────────────────────────────────────────
                                          │
                               MySQL (SQLAlchemy 2.0)
                                          │
                            FastAPI + Jinja2 web UI
```

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Document extraction | PyMuPDF, python-docx, Tesseract OCR |
| Schema validation | Pydantic v2 |
| LLM providers | google-genai, openai-compat (Groq), ollama, urllib (OpenRouter) |
| NLP / scoring | spaCy `en_core_web_sm`, `Alibaba-NLP/gte-reranker-modernbert-base` (sentence-transformers) |
| Database | SQLAlchemy 2.0, MySQL (PyMySQL) |
| Web UI | FastAPI, Jinja2, vanilla JS |

---

## Installation

### Prerequisites

- Python 3.10+
- MySQL 8.0+ (running locally or remotely)
- Tesseract OCR — only needed for scanned PDFs ([Windows installer](https://github.com/UB-Mannheim/tesseract/wiki))
- An API key for at least one LLM provider (see [LLM Providers](#llm-providers))

### Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/GargRakshit/Groz_Engineering_ATS.git
cd Groz_Engineering_ATS

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy language model
python -m spacy download en_core_web_sm
```

### Environment file

Create a `.env` file in the project root:

```dotenv
# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=yourpassword
MYSQL_DB=resume_parser

# Pick one LLM provider
LLM_PROVIDER=groq          # groq | gemini | openrouter | ollama

GROQ_API_KEY=gsk_...
# GENAI_API_KEY=AIza...
# OPENROUTER_API_KEY=sk-or-...
# OLLAMA_MODEL=llama3.2

# Web UI credentials (overrides default admin/admin123)
APP_USERNAME=admin
APP_PASSWORD=your_strong_password

# Windows only — path to Tesseract (only needed for scanned PDFs)
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Start the UI

```bash
uvicorn Code.search.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Default login: `admin` / `admin123` — **change this before use** via `APP_USERNAME` / `APP_PASSWORD` in `.env`.

---

## Corpus Data (not included in repo)

The `Data/` folder ships with two pre-built derived files (`kg_expansions.json` and `idf_weights.json`) so the system works out of the box. The raw source datasets used to build them are **not included** because they total ~11 GB.

If you want to rebuild them from scratch (e.g. to update the IDF weights with a newer job postings dataset), download the following:

### 1. LinkedIn Job Postings 2023–24 (IDF weights)
Used to compute corpus-level IDF values for domain-token detection.

**Download:** [kaggle.com/datasets/arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

Place the extracted files in `Data/Corpus/LinkedIn Job Postings 2023-24/`.

### 2. ESCO Dataset v1.2.1 (KG expansion)
European Skills, Competences, Qualifications and Occupations taxonomy — provides skill synonyms and occupational relationships.

**Download:** [esco.ec.europa.eu/en/use-esco/download](https://esco.ec.europa.eu/en/use-esco/download)  
Select: *ESCO dataset v1.2.1 — classification — en — csv*

Place in `Data/Corpus/ESCO dataset - v1.2.1 - classification - en - csv/`.

### 3. O\*NET 30.3 Database (KG expansion)
U.S. occupational knowledge database — provides job titles, skills, knowledge areas, and transferable skills.

**Download:** [onetcenter.org/database.html](https://www.onetcenter.org/database.html#all-files)  
Select: *Text* format, version 30.3.

Place in `Data/Corpus/db_30_3_text/`.

### 4. ConceptNet 5.7 Assertions (KG expansion)
Commonsense knowledge graph — provides synonym and related-term relationships filtered to the ESCO/O\*NET vocabulary.

**Download:** [github.com/commonsense/conceptnet5 — Releases](https://github.com/commonsense/conceptnet5/wiki/Downloads)  
File: `conceptnet-assertions-5.7.0.csv.gz`

Place the extracted CSV in `Data/Corpus/conceptnet-assertions-5.7.0.csv/`.

---

## LLM Providers

| Provider | Env var | Notes |
|---|---|---|
| **Groq** | `GROQ_API_KEY` | Recommended — fast, generous free tier, JSON mode |
| **Gemini** | `GENAI_API_KEY` | Google AI Studio key |
| **OpenRouter** | `OPENROUTER_API_KEY` | Access to many models via one key |
| **Ollama** | _(none)_ | Fully local; set `OLLAMA_MODEL` |

Set `LLM_PROVIDER=groq` (or `gemini` / `openrouter` / `ollama`) in `.env`.

---

## Project Structure

```
Resume Parser/
├── Code/
│   ├── parser/         # Text extraction, LLM prompts, provider implementations
│   ├── matching/       # Per-phrase CE scoring, KG expansion, experience/education checks
│   ├── db/             # SQLAlchemy models (5 tables), MySQL session, LIKE-based search
│   ├── search/         # FastAPI app + Jinja2 templates
│   ├── scoring.py      # ATS score formula (0.60 × match + 0.40 × relevant_years)
│   └── run.py          # CLI batch processor
├── Data/
│   ├── kg_expansions.json   # Pre-built KG synonym map (117k entries)
│   └── idf_weights.json     # Pre-built IDF weights (LinkedIn corpus)
├── Archive/            # Archived resume and JD files (gitignored — personal data)
├── JDCache/            # JD extraction cache keyed by SHA-256 (gitignored)
├── DocWork/            # Technical documentation
└── requirements.txt
```

---

## CLI Usage

```bash
# Process a folder of resumes against a single JD
python -m Code.run --jd "path/to/job_description.pdf" "path/to/resumes/"

# Or pass files explicitly
python -m Code.run --jd jd.pdf resume1.pdf resume2.docx
```

---

## Running Tests

```bash
python -m pytest Code/tests/ -v
# 91 tests — matching, scoring, database, pipeline
```

---

## License

Internal tool — all rights reserved.
