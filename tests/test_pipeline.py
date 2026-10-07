"""Unit and integration tests for the AI Resume Screening & Ranking System."""

from __future__ import annotations

import json
import urllib.error
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import ScreeningConfig
from src.eligibility import evaluate_eligibility
from src.github_enricher import GitHubEnricher
from src.llm_adapter import LLMProviderAdapter
from src.models import GitHubEnrichment, ParsedResume
from src.parser import parse_resume_file, split_into_sections
from src.pipeline import ResumeScreeningPipeline
from src.scorer import build_candidate_result, evaluate_resume_deterministic


def _make_resume(
    name: str,
    raw_text: str,
    github_username: str | None = None,
    parse_error: str | None = None,
) -> ParsedResume:
    from src.parser import extract_matched_skills

    return ParsedResume(
        file_name=f"{name.lower().replace(' ', '_')}.pdf",
        file_path=f"/tmp/{name.lower().replace(' ', '_')}.pdf",
        content_hash=f"hash_{name}",
        candidate_name=name,
        email=f"{name.lower().replace(' ', '.')}@example.com",
        github_url=f"https://github.com/{github_username}" if github_username else None,
        github_username=github_username,
        raw_text=raw_text,
        plumber_text=raw_text,
        sections=split_into_sections(raw_text),
        matched_skills=extract_matched_skills(raw_text),
        parse_error=parse_error,
    )


# ------------------------------------------------------------------------------
# 1. Hard Eligibility Filter Tests (Section 3)
# ------------------------------------------------------------------------------


def test_eligibility_rejects_react_java_only_profile() -> None:
    resume = _make_resume(
        "Frontend Dev",
        """
        Frontend Dev
        Skills
        Java, Spring Boot, React.js, Next.js, TypeScript, PostgreSQL, Docker
        Projects
        Built an e-commerce platform with Spring Boot REST APIs and React frontend.
        Implemented server-side pagination for scalable data retrieval.
        """,
    )
    res = evaluate_eligibility(resume)
    assert res.eligible is False
    assert "No evidence of Python stack" in res.rejection_reasons
    assert "No AI/agentic project evidence" in res.rejection_reasons
    assert "Java" in res.matched_skills
    assert "React" in res.matched_skills


def test_eligibility_rejects_python_without_ai_evidence() -> None:
    resume = _make_resume(
        "Backend Python Dev",
        """
        Backend Python Dev
        Skills
        Python, FastAPI, Django, PostgreSQL, Redis, Docker
        Experience
        Built scalable REST APIs using Python, FastAPI, and PostgreSQL for inventory management.
        Tools: Git, GitHub Copilot, Claude Code, Cursor.
        """,
    )
    res = evaluate_eligibility(resume)
    assert res.eligible is False
    assert res.has_python_evidence is True
    assert res.has_ai_agentic_evidence is False
    assert res.rejection_reasons == ["No AI/agentic project evidence"]


def test_eligibility_rejects_nodejs_rag_without_python() -> None:
    resume = _make_resume(
        "Node AI Dev",
        """
        Node AI Dev
        Skills
        TypeScript, JavaScript, Node.js, Express, React, PostgreSQL, pgvector, OpenAI API
        Projects
        Built an AI-powered RAG platform using Node.js, React, PostgreSQL, and pgvector with OpenAI embeddings.
        """,
    )
    res = evaluate_eligibility(resume)
    assert res.eligible is False
    assert res.has_python_evidence is False
    assert res.has_ai_agentic_evidence is True
    assert res.rejection_reasons == ["No evidence of Python stack"]


def test_eligibility_accepts_python_plus_ai_with_react_java_present() -> None:
    resume = _make_resume(
        "Asha Rao",
        """
        Asha Rao
        Skills
        Python, FastAPI, Java, React, Next.js, LangGraph, LangChain, PostgreSQL, pgvector, Docker, GCP
        Projects
        Built a stateful multi-agent workflow with LangGraph, tool calling, and pgvector RAG retrieval in Python and FastAPI.
        """,
    )
    res = evaluate_eligibility(resume)
    assert res.eligible is True
    assert res.rejection_reasons == []
    assert "Python" in res.matched_skills
    assert "LangGraph" in res.matched_skills


# ------------------------------------------------------------------------------
# 2. Scoring & Project-Quality Penalty Tests (Section 4)
# ------------------------------------------------------------------------------


def test_scoring_rewards_deep_agentic_rag_and_penalizes_thin_wrapper() -> None:
    deep_agentic_resume = _make_resume(
        "Deep AI Engineer",
        """
        Deep AI Engineer
        Summary
        Python backend developer building production LangGraph multi-agent systems and RAG pipelines.
        Skills
        Python, FastAPI, Asyncio, PostgreSQL, pgvector, Redis, LangChain, LangGraph, Qdrant, Ollama, Docker, GCP, React, Next.js, Pytest, Grafana, Kafka
        Experience
        AI Systems Intern
        - Architected a 7-stage async FastAPI multi-agent NL-to-SQL pipeline using LangGraph and tool calling with PostgreSQL and pgvector, achieving 93% accuracy across 1,000 queries.
        - Built hybrid semantic search and reranking over Qdrant with structured Pydantic output validation to mitigate hallucinations.
        - Deployed containerized microservices with Docker on GCP with Redis caching, Kafka event queues, Grafana observability, and pytest suites.
        """,
        github_username="deepai",
    )

    shallow_wrapper_resume = _make_resume(
        "Wrapper Candidate",
        """
        Wrapper Candidate
        Skills
        Python, Flask, Streamlit, Gemini, Groq, SQL
        Projects
        Chatbot Using Google Gemini API
        - Designed and implemented a chatbot using Python, Streamlit, and Google Gemini API to answer user questions.
        - Integrated Groq API to generate quiz questions from prompts.
        """,
        github_username="wrapperdev",
    )

    deep_eval = evaluate_resume_deterministic(deep_agentic_resume)
    wrapper_eval = evaluate_resume_deterministic(shallow_wrapper_resume)

    assert deep_eval.is_thin_api_wrapper is False
    assert deep_eval.wrapper_penalty_points == 0
    assert deep_eval.ai_project_depth >= 34
    assert deep_eval.python_backend >= 26
    assert deep_eval.cloud_fullstack >= 12
    assert deep_eval.engineering_depth >= 4

    # Thin wrapper must receive a 5-15 point penalty and score much lower on AI depth
    assert wrapper_eval.is_thin_api_wrapper is True
    assert 5 <= wrapper_eval.wrapper_penalty_points <= 15
    assert wrapper_eval.ai_project_depth <= 12

    gh_mock = GitHubEnrichment(
        username="test",
        profile_url="https://github.com/test",
        status="enriched",
        score=8,
        summary="Active GitHub profile.",
    )
    deep_result = build_candidate_result(
        deep_agentic_resume,
        evaluate_eligibility(deep_agentic_resume),
        gh_mock,
    )
    wrapper_result = build_candidate_result(
        shallow_wrapper_resume,
        evaluate_eligibility(shallow_wrapper_resume),
        gh_mock,
    )
    assert deep_result.total_score > wrapper_result.total_score + 35


def test_scoring_rewards_applied_project_evidence_over_skills_only() -> None:
    skills_only_resume = _make_resume(
        "Keyword Stuffer",
        """
        Keyword Stuffer
        Skills
        Python, FastAPI, Django, Asyncio, PostgreSQL, Redis, LangChain, LangGraph, RAG, pgvector, Docker, GCP
        Projects
        Simple Expense Tracker
        - Built a command-line script in Python to calculate monthly expenses and print a summary table.
        """,
    )
    applied_resume = _make_resume(
        "Applied Builder",
        """
        Applied Builder
        Skills
        Python, FastAPI, Asyncio, PostgreSQL, Redis, LangGraph, RAG, pgvector
        Projects
        Enterprise Agentic RAG Platform
        - Built an async FastAPI backend with PostgreSQL (pgvector) and Redis caching powering a LangGraph multi-agent RAG pipeline with tool calling and reranking.
        """,
    )
    skills_eval = evaluate_resume_deterministic(skills_only_resume)
    applied_eval = evaluate_resume_deterministic(applied_resume)

    assert applied_eval.ai_project_depth > skills_eval.ai_project_depth + 10
    assert applied_eval.python_backend > skills_eval.python_backend


# ------------------------------------------------------------------------------
# 3. GitHub Enrichment & Error Handling Tests (Section 5)
# ------------------------------------------------------------------------------


def test_github_scoring_and_graceful_failure(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    recent_iso = (now - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    repos = [
        {
            "name": f"agent-rag-system-{i}",
            "fork": False,
            "language": "Python",
            "description": "Stateful LangGraph RAG agent with FastAPI",
            "pushed_at": recent_iso,
            "stargazers_count": 5,
            "topics": ["ai", "langgraph", "python"],
        }
        for i in range(6)
    ]
    enrichment = GitHubEnricher.score_repos(
        username="asha-rao",
        profile_url="https://github.com/asha-rao",
        repos=repos,
        reference_now=now,
    )
    assert enrichment.status == "enriched"
    assert enrichment.recent_activity_score == 5
    assert enrichment.relevant_repos_score >= 4
    assert 9 <= enrichment.score <= 10

    # Verify graceful rate-limit handling (HTTP 403) does not raise
    cfg = ScreeningConfig(cache_dir=tmp_path / ".cache", use_cache=False)
    enricher = GitHubEnricher(cfg)
    dummy_resume = _make_resume("Rate Limited User", "Python LangGraph RAG", github_username="ratelimited")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/users/ratelimited/repos",
            code=403,
            msg="rate limit exceeded",
            hdrs=None,  # type: ignore
            fp=None,
        )
        res = enricher.enrich_candidate_sync(dummy_resume)
        assert res.status == "rate_limited"
        assert res.score == 0


# ------------------------------------------------------------------------------
# 4. LLM Adapter Fallback & Batch Pipeline Fault-Tolerance Tests (Section 6 & 9)
# ------------------------------------------------------------------------------


def test_llm_adapter_graceful_fallback_on_api_error(tmp_path: Path) -> None:
    cfg = ScreeningConfig(
        cache_dir=tmp_path / ".cache",
        enable_llm=True,
        llm_provider="groq",
        groq_api_key="test-fake-key",
        use_cache=False,
    )
    adapter = LLMProviderAdapter(cfg)
    resume = _make_resume("Test User", "Python FastAPI LangGraph RAG")

    with patch("urllib.request.urlopen", side_effect=RuntimeError("Simulated LLM outage")):
        assert adapter.evaluate_resume(resume) is None


def test_batch_pipeline_handles_docx_txt_duplicates_and_corrupt_pdf(tmp_path: Path) -> None:
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    output_file = tmp_path / "output" / "results.json"

    # 1. Valid TXT resume (Eligible)
    txt_content = """Asha Rao
asha@example.com | https://github.com/asharao
Experience
Built an async FastAPI and PostgreSQL (pgvector) backend with Redis caching for a LangGraph multi-agent RAG system deployed on GCP with Docker and pytest.
"""
    (resumes_dir / "cand_txt.txt").write_text(txt_content, encoding="utf-8")

    # 2. Duplicate TXT resume (identical content -> should be deduplicated)
    (resumes_dir / "cand_txt_dup.txt").write_text(txt_content, encoding="utf-8")

    # 3. Valid DOCX resume (Rejected: Java/React only)
    docx_path = resumes_dir / "cand_docx.docx"
    with zipfile.ZipFile(docx_path, "w") as zf:
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>Rohan Verma</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Skills: Java, Spring Boot, React, MySQL</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Built CRUD microservices in Java and Spring Boot.</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        zf.writestr("word/document.xml", xml)

    # 4. Corrupt/Malformed PDF file (must not crash batch!)
    (resumes_dir / "corrupt.pdf").write_bytes(b"%PDF-1.4 CORRUPT GARBAGE BYTES")

    cfg = ScreeningConfig(
        input_dir=resumes_dir,
        output_path=output_file,
        cache_dir=tmp_path / ".cache",
        enable_github=False,
        enable_llm=False,
    )
    pipeline = ResumeScreeningPipeline(cfg)
    report = pipeline.run(write_outputs=True)

    assert report.batch_summary.total_resumes == 4
    assert report.batch_summary.duplicate_files == 1
    assert report.batch_summary.eligible == 1
    assert report.batch_summary.rejected == 1
    assert report.batch_summary.failed_unreadable == 1
    assert report.ranked_candidates[0].candidate_name == "Asha Rao"
    assert report.ranked_candidates[0].rank == 1
    assert output_file.exists()
    assert (output_file.parent / "shortlist.json").exists()

    shortlist = json.loads((output_file.parent / "shortlist.json").read_text(encoding="utf-8"))
    assert len(shortlist) == 1
    assert shortlist[0]["rank"] == 1
    assert shortlist[0]["eligible"] is True


def test_fastapi_endpoints(tmp_path: Path) -> None:
    import asyncio
    from fastapi import HTTPException
    from src.api import ScreenRequest, get_results, get_shortlist, health_check, screen_resumes

    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "c1.txt").write_text(
        "Asha Rao\nSkills: Python, FastAPI, LangGraph, RAG, pgvector\nBuilt a LangGraph RAG agent in Python and FastAPI.",
        encoding="utf-8",
    )
    out_file = tmp_path / "out" / "results.json"

    health = asyncio.run(health_check())
    assert health["status"] == "ok"

    report = asyncio.run(
        screen_resumes(
            ScreenRequest(
                input_dir=str(resumes_dir),
                output_path=str(out_file),
                enable_github=False,
                enable_llm=False,
            )
        )
    )
    assert report.batch_summary.total_resumes == 1
    assert report.batch_summary.eligible == 1

    results_dict = asyncio.run(get_results())
    assert results_dict["batch_summary"]["eligible"] == 1

    shortlist_list = asyncio.run(get_shortlist())
    assert len(shortlist_list) == 1
    assert shortlist_list[0]["candidate_name"] == "Asha Rao"

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(screen_resumes(ScreenRequest(input_dir=str(tmp_path / "does_not_exist"))))
    assert exc_info.value.status_code == 400


def test_generated_50_resume_outputs_integrity() -> None:
    results_path = Path("output/results.json")
    shortlist_path = Path("output/shortlist.json")
    csv_path = Path("output/results.csv")
    html_path = Path("output/report.html")

    if not results_path.exists():
        pytest.skip("output/results.json not generated yet")

    assert shortlist_path.exists()
    assert csv_path.exists()
    assert html_path.exists()

    data = json.loads(results_path.read_text(encoding="utf-8"))
    bs = data["batch_summary"]
    assert bs["total_resumes"] == 50
    assert bs["eligible"] + bs["rejected"] + bs["failed_unreadable"] == 50

    ranked = data["ranked_candidates"]
    assert len(ranked) == bs["eligible"]

    prev_score = 101
    for idx, c in enumerate(ranked, start=1):
        assert c["rank"] == idx
        assert c["eligible"] is True
        sb = c["score_breakdown"]
        calc_total = (
            sb["ai_project_depth"]
            + sb["python_backend"]
            + sb["cloud_fullstack"]
            + sb["github"]
            + sb["engineering_depth"]
        )
        assert c["total_score"] == calc_total
        assert c["total_score"] <= prev_score
        prev_score = c["total_score"]

    for rej in data["rejected_candidates"]:
        assert rej["eligible"] is False
        assert len(rej["rejection_reasons"]) >= 1
        if "No evidence of Python stack" in rej["rejection_reasons"]:
            assert "Python" not in rej["matched_skills"]
            assert "Asyncio" not in rej["matched_skills"]
            assert "Pytest" not in rej["matched_skills"]

    # Verify project summaries are complete sentences and not truncated mid-phrase
    dangling_endings = (
        "built the.",
        "on freeswitch +.",
        "grounded in a real.",
        "integrated generic.",
        "services for.",
        "across 3.",
        "and real-time.",
        "using the.",
    )
    for c in ranked:
        summary_low = c["project_summary"].lower()
        for bad_ending in dangling_endings:
            assert not summary_low.endswith(bad_ending), (
                f"{c['file_name']} summary ended with truncated phrase '{bad_ending}': {c['project_summary']}"
            )
        assert "plat- form" not in summary_low
        assert "pub- lish" not in summary_low
        assert "cut off" not in summary_low
        assert "truncated" not in summary_low
        assert "\u2011" not in c["project_summary"]
        assert "\u202f" not in c["project_summary"]
        for con in c["concerns"]:
            assert "truncated" not in con.lower()
            assert "\u2011" not in con
            assert "\u202f" not in con

    # Verify both thin-wrapper (-10), CV-only (-8), and shallow POC (-6) penalties are represented
    penalties_by_file = {c["file_name"]: c["penalty_applied"] for c in ranked}
    assert penalties_by_file["candidate_15.pdf"] == 8
    assert penalties_by_file["candidate_37.pdf"] == 0
    assert penalties_by_file["candidate_23.pdf"] == 6
    assert penalties_by_file["candidate_36.pdf"] == 6
    assert penalties_by_file["candidate_49.pdf"] == 6
    assert penalties_by_file["candidate_25.pdf"] == 10


def test_multiline_bullet_reconstruction_and_no_work_text_tripling() -> None:
    from src.scorer import _get_work_and_project_text, extract_concise_project_summary

    wrapped_resume = _make_resume(
        "Wrapped Candidate",
        """
        Wrapped Candidate
        Projects
        • Built the core frontend architecture for an enterprise Agentic AI Plat-
        form — Journey Builder, Component Registry, MCP-based data source
        connectors, and a LangGraph multi-agent orchestration layer.
        • Orchestrated a multi-agent chat system on a Langraph StateGraph — a supervisor routing each turn across 3
        2020 – 2024
        specialized agents (search, summary, parallel) via reducer-based shared state.
        """,
    )
    work_text = _get_work_and_project_text(wrapped_resume)
    # Ensure lines are not tripled across sections/raw_text/plumber_text
    assert len(work_text.splitlines()) == 2
    assert "Platform" in work_text
    assert "Plat- form" not in work_text

    summary = extract_concise_project_summary(wrapped_resume)
    assert "Platform" in summary
    assert "specialized agents (search, summary, parallel)" in summary
    assert not summary.endswith("across 3.")


def test_engineering_depth_regex_stems_and_agent_qa_evaluation() -> None:
    resume = _make_resume(
        "Aditi QA Engineer",
        """
        Aditi QA Engineer
        Experience
        AI Engineer Intern
        - Built an automated agent QA framework with scripted and LLM-to-LLM test modes and an LLM scoring engine, cutting manual validation time by 60%.
        - Improved and scaled a production voice AI calling system to handle 200+ automated lead calls per day.
        - Implemented token optimization, idempotency keys, rate limiting, retries with exponential backoff, and unit testing suites in Python and FastAPI.
        """,
    )
    elig = evaluate_eligibility(resume)
    assert elig.eligible is True

    det = evaluate_resume_deterministic(resume)
    assert det.is_thin_api_wrapper is False
    assert det.wrapper_penalty_points == 0
    assert det.ai_project_depth >= 17
    assert det.engineering_depth >= 3


def test_cli_respects_max_concurrency_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    from main import parse_args

    monkeypatch.setenv("MAX_CONCURRENCY", "9")
    args_default = parse_args(["--input", "./resumes", "--output", "./output/results.json"])
    assert args_default.concurrency is None
    cfg = ScreeningConfig()
    assert cfg.max_concurrency == 9


def test_env_security_and_dotenv_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config import _load_env_file

    env_example = Path(".env.example").read_text(encoding="utf-8")
    assert "gsk_" not in env_example
    assert "sk-" not in env_example
    assert "AIza" not in env_example

    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore

    custom_env = tmp_path / ".env"
    custom_env.write_text(
        'export TEST_CUSTOM_ENV_VAR="quoted_val"\n'
        'TEST_INLINE_COMMENT=clean_val # comment\n'
        'TEST_QUOTED_WITH_COMMENT="openai/gpt-oss-120b" # inline comment\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("TEST_CUSTOM_ENV_VAR", raising=False)
    monkeypatch.delenv("TEST_INLINE_COMMENT", raising=False)
    monkeypatch.delenv("TEST_QUOTED_WITH_COMMENT", raising=False)
    _load_env_file(custom_env)
    import os

    assert os.environ.get("TEST_CUSTOM_ENV_VAR") == "quoted_val"
    assert os.environ.get("TEST_INLINE_COMMENT") == "clean_val"
    assert os.environ.get("TEST_QUOTED_WITH_COMMENT") == "openai/gpt-oss-120b"


def test_groq_model_fallback_and_json_normalization(tmp_path: Path) -> None:
    cfg = ScreeningConfig(
        cache_dir=tmp_path / ".cache",
        enable_llm=True,
        llm_provider="groq",
        llm_model="llama-3.3-70b-versatile",
        groq_api_key="test-fake-key",
        use_cache=True,
    )
    adapter = LLMProviderAdapter(cfg)
    resume = _make_resume("Asha Rao", "Python FastAPI LangGraph RAG pgvector")

    called_models: list[str] = []

    def fake_call_openai_compatible(url: str, api_key: str, model: str, user_prompt: str) -> str:
        called_models.append(model)
        if model == "llama-3.3-70b-versatile":
            raise urllib.error.HTTPError(
                url=url,
                code=404,
                msg="model_not_found",
                hdrs=None,  # type: ignore
                fp=None,
            )
        if model == "openai/gpt-oss-120b":
            # Simulate truncated/malformed JSON on first active model -> should fall back to next model
            return '{"ai_project_depth": 35, "truncated'
        return """```json
        {
          "ai_project_depth": 45.2,
          "python_backend": 28,
          "cloud_fullstack": 14,
          "engineering_depth": 5,
          "is_thin_api_wrapper": false,
          "wrapper_penalty_points": -6,
          "project_summary": "Built a multi\u2011agent LangGraph RAG system in Python and FastAPI with sub\u2011600\u202fms latency.",
          "strengths": ["Strong agentic architecture", "FastAPI backend"],
          "concerns": ["Minor gap"],
          "score_evidence": {"ai_project_depth": ["LangGraph", "pgvector"]}
        }
        ```"""

    with patch.object(adapter, "_call_openai_compatible", side_effect=fake_call_openai_compatible):
        res = adapter.evaluate_resume(resume)

    assert res is not None
    assert called_models == ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert res.ai_project_depth == 40  # Clamped from 45.2 to max 40
    assert res.wrapper_penalty_points == 6  # Normalized from -6 to abs(6)
    assert res.score_evidence["ai_project_depth"] == "LangGraph; pgvector"
    assert "\u2011" not in res.project_summary
    assert "\u202f" not in res.project_summary
    assert "multi-agent" in res.project_summary
    assert "sub-600 ms" in res.project_summary


def test_openai_default_model_and_long_resume_prompt_compaction(tmp_path: Path) -> None:
    cfg = ScreeningConfig(
        cache_dir=tmp_path / ".cache",
        enable_llm=True,
        llm_provider="openai",
        llm_model="openai/gpt-oss-120b",  # Default Groq model ID with slash prefix
        openai_api_key="sk-fake-openai-key",
        use_cache=False,
    )
    adapter = LLMProviderAdapter(cfg)
    padding = ("Skills and Experience line with extra blank lines.\n\n\n") * 120
    long_text = f"Long Resume Candidate\n{padding}\nPROJECTS\nBuilt a LangGraph RAG agent with pgvector in Python and FastAPI."
    assert len(long_text) > 5500
    resume = _make_resume("Long Resume Candidate", long_text)

    captured: dict[str, str] = {}

    def fake_call_openai_compatible(url: str, api_key: str, model: str, user_prompt: str) -> str:
        captured["model"] = model
        captured["user_prompt"] = user_prompt
        return json.dumps(
            {
                "ai_project_depth": 32,
                "python_backend": 25,
                "cloud_fullstack": 10,
                "engineering_depth": 3,
                "is_thin_api_wrapper": False,
                "wrapper_penalty_points": 0,
                "project_summary": "Built a LangGraph RAG agent with pgvector in Python and FastAPI.",
                "strengths": ["LangGraph RAG"],
                "concerns": ["None"],
                "score_evidence": {"ai_project_depth": "LangGraph + pgvector"},
            }
        )

    with patch.object(adapter, "_call_openai_compatible", side_effect=fake_call_openai_compatible):
        res = adapter.evaluate_resume(resume)

    assert res is not None
    assert captured["model"] == "gpt-4o-mini"
    assert "Built a LangGraph RAG agent with pgvector in Python and FastAPI." in captured["user_prompt"]

