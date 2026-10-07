# AI Resume Screening & Ranking System — Engineering Plan

## 1. Objective & Problem Statement
Build a reliable, production-minded backend screening and ranking pipeline that:
1. **Ingests** a directory of candidate resumes (`.pdf`, `.docx`, `.txt`) in a single batch run without failing on malformed/unreadable files or duplicates.
2. **Extracts** structured candidate profiles (name, email, phone, GitHub URL/username from both text and PDF `/URI` hyperlink annotations, skills, projects, and work experience).
3. **Filters** candidates using deterministic, explainable **Hard Eligibility Rules** (requiring genuine **Python evidence** AND genuine **AI/LLM/RAG/Agentic evidence**).
4. **Enriches** eligible candidates with public **GitHub Activity** via the GitHub REST API (with disk caching, optional `GITHUB_TOKEN`, bounded concurrency, and graceful fallback on rate limits or missing profiles).
5. **Scores & Ranks** eligible candidates on a **100-point rubric** with explicit **project-quality penalties** (deducting 5–15 points for thin LLM API wrappers or shallow/tutorial projects and discounting skills-only keyword stuffing).
6. **Outputs** machine-readable `output/results.json`, `output/shortlist.json`, `output/results.csv`, and `output/report.html`, along with both a **CLI** (`main.py`) and a **FastAPI REST API** (`src/api.py`).

---

## 2. System Architecture & Module Breakdown

```text
Kasparro - Assessment/
├── main.py                  # CLI entrypoint (--input, --output, --csv, --html, --serve)
├── PLAN.md                  # Architectural & implementation plan
├── README.md                # Setup, usage, Design Decisions, and If I Had More Time
├── requirements.txt         # Pinned project dependencies
├── .env.example             # Environment variable template (no secrets)
├── src/
│   ├── __init__.py
│   ├── config.py            # Weights, thresholds, model config, env vars, skill taxonomies
│   ├── models.py            # Pydantic v2 schemas for parsing, eligibility, LLM, GitHub, output
│   ├── parser.py            # Layout-aware PDF/DOCX/TXT parser + PDF hyperlink & font-size extractor
│   ├── eligibility.py       # Rule-based hard eligibility filter (Python + AI/Agentic evidence)
│   ├── github_enricher.py   # Async bounded-concurrency GitHub API client + disk cache + 10-pt scorer
│   ├── llm_adapter.py       # Provider-agnostic LLM structured output adapter (OpenAI/Gemini/Groq + fallback)
│   ├── scorer.py            # 100-point hybrid scoring engine + shallow-wrapper penalty logic
│   ├── pipeline.py          # Batch orchestrator (ingest -> deduplicate -> filter -> enrich -> score -> rank)
│   ├── reporter.py          # JSON, CSV, HTML, and CLI terminal summary generators
│   └── api.py               # FastAPI server (POST /screen, GET /results, GET /shortlist, GET /health)
├── tests/
│   └── test_pipeline.py     # Unit & integration tests (eligibility, scoring, penalties, parser, API)
└── output/
    ├── results.json         # Complete screening output (batch summary, ranked eligible, rejected)
    ├── shortlist.json       # Top-level array of ranked candidates matching Section 7 snippet
    ├── results.csv          # Tabular export of all screened candidates
    └── report.html          # Visual HTML breakdown report
```

---

## 3. Core Engineering Decisions

### 3.1 Multi-Engine Layout-Aware Resume Parsing (`src/parser.py`)
- **Dual PDF Strategy**:
  - Uses `pdfminer.six` (`LAParams`) to preserve vertical column blocks in multi-column resumes (such as `candidate_15.pdf` and `candidate_23.pdf`).
  - Uses `pdfplumber` to:
    1. Extract embedded PDF `/URI` hyperlink annotations (`page.hyperlinks`) — critical because 29 of the 50 resumes hide their GitHub URL inside a `"GitHub"` hyperlink label rather than plain text.
    2. Extract Page 1 character font sizes (`extract_words(extra_attrs=['size'])`) to accurately identify the candidate's name even when contact details appear above or beside the name (`candidate_09`, `candidate_15`, `candidate_17`, `candidate_32`, `candidate_35`, `candidate_36`, `candidate_37`, `candidate_44`).
    3. Provide line-by-line fallback text when PDF font descriptors omit `FontBBox` (`candidate_20.pdf`).
- **Bonus Formats & Fault Tolerance**:
  - Supports `.pdf`, `.docx`, and `.txt`.
  - Computes SHA-256 content hashes to detect duplicate files.
  - Catches any read/parse exception per file and records it in `failed_resumes` without interrupting the batch.

### 3.2 Hard Eligibility Rules (`src/eligibility.py`)
Kept strictly deterministic and outside the LLM:
1. **Python Requirement**:
   - Candidate must demonstrate Python in skills, projects, or work experience.
   - Furthermore, if `Python` appears only as a bare word in a generic languages list while 100% of projects, internships, and backend frameworks are Java/Spring or JS/MERN with no Python ecosystem usage, this is flagged.
2. **AI / Agentic / RAG Requirement**:
   - Candidate must show at least one meaningful AI/LLM/RAG/agentic project, framework, or implementation (`LangChain`, `LangGraph`, `LlamaIndex`, `Google ADK`, `CrewAI`, `AutoGen`, `LiveKit Agents`, `MCP`, `RAG` pipelines, vector search with `pgvector`/`Qdrant`/`Pinecone`/`ChromaDB`/`FAISS`/`Weaviate`/`Milvus`, tool-calling agents, multi-agent workflows, LLM evaluation/orchestration).
   - **False-Positive Guardrails**:
     - Mentions of AI coding assistants (`Claude Code`, `Claude AI`, `Cursor`, `GitHub Copilot`) strictly as developer productivity tools do **not** satisfy the AI/agentic project requirement.
     - Generic database/file "data retrieval" (e.g., `"pagination for scalable data retrieval"`, `"GCS file upload and retrieval"`, `"database storage and retrieval"`) without LLM/RAG/vector context does **not** satisfy the requirement.

### 3.3 100-Point Scoring Model & Project-Quality Penalties (`src/scorer.py`)
| Category | Max Points | Signals Rewarded |
| :--- | :---: | :--- |
| **AI / Agentic / RAG Project Depth** | **40** | Stateful/multi-agent orchestration (`LangGraph`, `CrewAI`, `AutoGen`, `LiveKit Agents`, `MCP`), tool calling, RAG pipelines, vector DBs (`pgvector`, `Qdrant`, `Pinecone`, `Milvus`, `Weaviate`, `ChromaDB`, `FAISS`), reranking/hybrid search, evaluation/benchmarking, guardrails, production metrics. |
| **Python & Backend Engineering** | **30** | Python depth in projects/internships, `FastAPI` (plus `Django`/`Flask`), `async`/`asyncio`/streaming/WebSockets, `PostgreSQL`, `Redis`. |
| **Cloud / Deployment / Full Stack** | **15** | `GCP` (`Vertex AI`, `GCS`, `Cloud Run`), `Docker`, `Kubernetes`, CI/CD (`GitHub Actions`, `Jenkins`), cloud deployment (`AWS`, `Azure`, `Render`, `Vercel`), and `React`/`Next.js`/`TypeScript` end-to-end integration. |
| **GitHub Activity** | **10** | `0–5 pts` for recent push/commit activity + `0–5 pts` for maintained (non-fork) and Python/AI-relevant public repositories. |
| **Engineering Depth Signals** | **5** | Automated testing (`pytest`, Playwright), system architecture (microservices, multi-tenant, RBAC), caching, queues (`Kafka`, `BullMQ`, `Celery`), observability (`Grafana`, `Prometheus`, `Loki`, `Kibana`), concurrency, fault tolerance (circuit breakers, retries, idempotency). |

- **Project-Quality Penalties (5–15 pts deducted from AI Project Depth)**:
  - **Thin API Wrapper Penalty (5–15 pts)**: Applied when a candidate's AI projects consist only of basic `OpenAI`/`Gemini`/`Groq` API calls wrapped in a simple UI/endpoint without retrieval, vector databases, multi-step agent state, tool calling, or evaluation.
  - **Keyword-Only Discount**: Framework names appearing only in the skills section without corresponding evidence in project/internship descriptions receive a steep discount.

### 3.4 Lightweight GitHub Enrichment (`src/github_enricher.py`)
- Uses `/users/{username}/repos?sort=pushed&per_page=100` to retrieve repository metadata, languages, fork status, descriptions, and recent push timestamps in **1 API call per unique username** (preserving the 60 req/hr unauthenticated rate limit).
- Persists responses in `.cache/github_cache.json` and bounds concurrency via `asyncio.Semaphore`.
- Never fails the batch on 404, private profile, rate limit, or timeout.
