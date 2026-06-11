# Resume Parser — Application Tracking System

A self-hosted applicant tracking system that parses resumes and job descriptions, scores candidates against open roles, and surfaces ranked results through a web UI — all running locally with no cloud dependency at runtime.

> **Context:** This project was developed during an internship at [Groz-Engineering-Tools], a tools manufacturer. The Groz branding visible in the UI (logo, orange accent colour) reflects that origin. The codebase is shared publicly as a portfolio piece and as a reference for others building similar tools.

---

## Work in Progress

This project is being built in stages, moving progressively away from LLM-heavy processing toward fully deterministic, API-free scoring.

**Where it started:** The original system had the LLM do everything — extract resume data, match skills, score candidates, and justify the match. This was fast to build but non-deterministic (the same resume could score differently on two runs), expensive (long prompts), and hard to audit.

**Where it is now:** The LLM is responsible for extraction only — one call per resume, one call per job description. All matching, scoring, and ranking is done in Python with a hybrid BM25 + cross-encoder pipeline that produces deterministic results.

**Where it is going:** Eliminate the LLM dependency for extraction too, replacing it with a local pipeline. The goal is a system that works fully offline with zero API calls — useful in environments where data cannot leave the premises or with no API calls budget.

| Stage | Status | What the LLM does |
|---|---|---|
| 1 — LLM does everything | ✅ Done (legacy, deleted) | Extract + match + score |
| 2 — LLM does extraction only | ✅ Current | Extract structured data from text |
| 3 — Local NLP extraction | 🔲 Planned | Nothing — fully offline |

---

## Features

- **Resume parsing** — Extracts candidate info, skills, experience, education, certifications, and qualifications from PDF and DOCX files
- **JD parsing** — Extracts required/preferred skills, experience requirements, education level, and certifications from job description documents
- **Hybrid scoring** — BM25 (lexical, corpus-IDF-weighted) + cross-encoder (`gte-reranker-modernbert-base`, 8192-token context) produce a single deterministic ATS score
- **Knowledge-graph expansion** — BM25 inputs are expanded with synonyms from ESCO, O\*NET, and ConceptNet (117k entries), catching abbreviations and domain-specific aliases
- **Web UI** — Candidate cards, filters (name, score, years, status, JD), upload overlay, JD manager, per-resume detail view, dark/light mode
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
                     │  BM25 (corpus IDF) + Cross-Encoder reranker
                     │  KG synonym expansion (ESCO + O*NET + ConceptNet)
                     │  Experience date math · Education degree level
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
| NLP / scoring | spaCy `en_core_web_sm`, `cross-encoder/gte-reranker-modernbert-base` (sentence-transformers) |
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
git clone https://github.com/<your-username>/<repo-name>.git
cd resume-parser

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

# Web UI session key — change this
SECRET_KEY=change-me-to-a-random-string

# Windows only — path to Tesseract (only needed for scanned PDFs)
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Start the UI

```bash
uvicorn Code.search.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Default login: `admin` / `admin123` (change immediately via MySQL).

---

## Corpus Data (not included in repo)

The `Data/` folder ships with two pre-built derived files (`kg_expansions.json` and `idf_weights.json`) so the system works out of the box. The raw source datasets used to build them are **not included** because they total ~11 GB.

If you want to rebuild them from scratch (e.g. to update the IDF weights with a newer job postings dataset), download the following:

### 1. LinkedIn Job Postings 2023–24 (IDF weights)
Used to compute corpus-level IDF values for BM25 scoring.

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

### Rebuilding the derived files

Once the source data is in place, run the build scripts (coming as part of Stage 3 tooling — for now, contact the maintainer for the scripts used to generate the current `Data/*.json` files).

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
│   ├── matching/       # BM25+CE scoring, KG expansion, experience/education checks
│   ├── db/             # SQLAlchemy models, MySQL session, LIKE-based search
│   ├── search/         # FastAPI app + Jinja2 templates
│   ├── scoring.py      # ATS score formula
│   └── run.py          # CLI batch processor
├── Data/
│   ├── kg_expansions.json   # Pre-built KG synonym map (117k entries)
│   └── idf_weights.json     # Pre-built BM25 IDF weights
├── Archive/            # Archived resume and JD files (gitignored — personal data)
├── JDCache/            # JD extraction cache keyed by SHA-256 (gitignored)
├── DocWork/            # Documentation, architecture diagrams
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
# 66 tests — matching, scoring, database
```

---

## License

Internal tool — all rights reserved.
