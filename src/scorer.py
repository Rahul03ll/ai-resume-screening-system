"""100-point candidate scoring engine with project-quality penalties and hybrid LLM + deterministic evaluation (Section 4)."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from src.config import (
    WEIGHT_AI_PROJECT_DEPTH,
    WEIGHT_CLOUD_FULLSTACK,
    WEIGHT_ENGINEERING_DEPTH,
    WEIGHT_GITHUB_ACTIVITY,
    WEIGHT_PYTHON_BACKEND,
)
from src.models import (
    CandidateResult,
    EligibilityResult,
    GitHubEnrichment,
    LLMResumeEvaluation,
    ParsedResume,
    ScoreBreakdown,
)


SKILL_LIST_PREFIX_RE = re.compile(
    r"^(?:technical\s+skills|skills|programming(?:\s+languages)?|languages|frameworks(?:\s*&\s*libraries)?|backend(?:\s*&\s*apis)?|frontend(?:\s*&\s*full-stack)?|databases?(?:\s*(?:&|and)\s*(?:tools|devops|infra))?|cloud(?:\s*&\s*devops)?|devops|tools(?:\s*(?:&|and)\s*technologies)?(?:\s+used)?|core(?:\s+cs|\s+concepts)?|ai\s*/?\s*ml(?:\s*&\s*genai)?|ai\s+engineering|genai(?:\s*&\s*agents)?|tech\s+stack|technologies(?:\s+used)?|engineering\s+tools)\s*[:\-]",
    re.IGNORECASE,
)

ACTION_OR_PROSE_RE = re.compile(
    r"\b(?:built|building|designed|architected|developed|engineered|implemented|deployed|integrated|orchestrated|created|optimized|reduced|automated|shipped|replaced|configured|delivered|contributed|achieved|trained|enhanced|improved|led|owned|scaled|migrated|constructed|wired|put\s+together|leveraging|leverages|utilizing|uses|used|powering|enabling|supporting)\b",
    re.IGNORECASE,
)

MARGIN_DATE_OR_GPA_RE = re.compile(
    r"^(?:"
    r"(?:\([A-Za-z\s-]+\)\s*)?(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+)?\d{4}\s*[-–—to]+\s*(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+)?(?:\d{4}|present|current|ongoing)(?:\s*\|.*)?|"
    r"(?:\d{1,2}/\d{4})\s*[-–—to]+\s*(?:\d{1,2}/\d{4}|present|current)|"
    r"(?:c?gpa|gpa|percentage)\s*[:\-]?\s*\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?%?|"
    r"[a-z0-9.-]+\.(?:io|ai|com|in|org|dev|xyz)\s*\|\s*[A-Za-z\s,().-]+|"
    r"[A-Za-z\s]+,\s*[A-Z]{2}\s*\|\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}|"
    r"(?:Pune|Bangalore|Bengaluru|Hyderabad|Mumbai|Chennai|Delhi|Noida|Gurgaon|Gurugram|Remote|Ongoing|GitHub|Live\s+Demo|Source\s+Code)|"
    r"https?://\S+"
    r")$",
    re.IGNORECASE,
)

BIO_OR_OBJECTIVE_PREFIX_RE = re.compile(
    r"^(?:aspiring\s+to|seeking\s+to|looking\s+for|developer,\s+building|software\s+engineer\s+with\s+\d|ai/ml\s+engineer\s+with|full[\s-]+stack\s+(?:engineer|developer)\s+with|strong\s+experience\s+building)\b",
    re.IGNORECASE,
)

CONTINUATION_TAIL_RE = re.compile(
    r"(?:[,;:+/&(\-–—]|\b(?:and|or|with|for|to|in|by|of|on|from|using|via|across|over|into|through|the|a|an|as|at|that|which|including|featuring|leveraging|real|generic|automated|integrated|built|designed|powered|based|multi))\s*$",
    re.IGNORECASE,
)

DANGLING_TAIL_STRIP_RE = re.compile(
    r"(?:\s+(?:and|or|with|for|to|in|by|of|on|from|using|via|across|over|into|through|the|a|an|as|at|\+|&|-|–|—))+$",
    re.IGNORECASE,
)


def _reconstruct_bullets_from_lines(raw_lines: List[str]) -> List[str]:
    """
    Reconstruct complete multi-line bullets and prose sentences from PDF-wrapped lines.
    - Skips standalone right-margin date/CGPA/location/URL fragments so they do not split ongoing sentences.
    - Rejoins hyphenated line wraps (e.g. 'Plat-\\nform' -> 'Platform', 'pub-\\nlish' -> 'publish').
    - Merges wrapped continuation lines until sentence termination.
    """
    merged: List[str] = []
    for raw in raw_lines:
        line = re.sub(r"\(cid:\d+\)", "", raw).strip()
        if not line:
            continue
        if MARGIN_DATE_OR_GPA_RE.match(line):
            continue

        is_new_bullet = bool(re.match(r"^[●•▪◦\*]\s*|^(?:[-–—])\s+", line))
        cleaned = re.sub(r"^[●•▪◦\*]\s*|^(?:[-–—])\s+", "", line).strip()
        if not cleaned or MARGIN_DATE_OR_GPA_RE.match(cleaned):
            continue

        # Strip inline trailing right-margin date range if attached at the end of a header/line
        cleaned = re.sub(
            r"\s+(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4})\s*[-–—]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4}|Present|Current)\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if not cleaned:
            continue

        is_skill_prefix = bool(SKILL_LIST_PREFIX_RE.match(cleaned))
        is_project_or_role_header = bool(
            re.match(r"^[A-Z][A-Za-z0-9\s&/().:—–-]{2,55}\s*\|\s*[A-Za-z]", cleaned)
        )

        if (
            merged
            and not is_new_bullet
            and not is_skill_prefix
            and not is_project_or_role_header
            and not merged[-1].endswith((".", "!", "?"))
            and len(merged[-1]) < 560
        ):
            prev = merged[-1]
            # Check for hyphenated word wrap (e.g. "Plat-" + "form")
            if re.search(r"[A-Za-z]-$", prev) and cleaned[0].islower():
                merged[-1] = f"{prev[:-1]}{cleaned}"
                continue

            starts_like_continuation = (
                cleaned[0].islower()
                or cleaned[0].isdigit()
                or cleaned[0] in ("(", "+", "/", "&")
            )
            prev_expects_continuation = bool(CONTINUATION_TAIL_RE.search(prev))
            prev_is_active_prose = (
                bool(ACTION_OR_PROSE_RE.search(prev))
                and not SKILL_LIST_PREFIX_RE.match(prev)
                and not bool(re.match(r"^[A-Z][A-Za-z0-9\s&/().:—–-]{2,55}\s*\|\s*[A-Za-z]", prev))
            )

            if starts_like_continuation or prev_expects_continuation or prev_is_active_prose:
                merged[-1] = f"{prev} {cleaned}"
                continue

        merged.append(cleaned)

    return [re.sub(r"\s+", " ", m).strip() for m in merged if m.strip()]


def _get_work_and_project_text(resume: ParsedResume) -> str:
    """
    Extract applied project and work experience evidence while excluding bare comma-separated
    skill lists, without duplicating lines across sections/raw_text/plumber_text.
    """
    collected: List[str] = []
    seen_norm: set[str] = set()

    def _add_line(text_line: str) -> None:
        norm = re.sub(r"\s+", " ", text_line).strip().lower()
        if norm and norm not in seen_norm:
            seen_norm.add(norm)
            collected.append(text_line.strip())

    # 1. Reconstruct bullets from dedicated work/project/summary/achievement sections
    for sec in ("experience", "projects", "summary", "achievements"):
        if sec in resume.sections:
            sec_bullets = _reconstruct_bullets_from_lines(resume.sections[sec].splitlines())
            for b in sec_bullets:
                if not SKILL_LIST_PREFIX_RE.match(b):
                    _add_line(b)

    # 2. In multi-column PDFs (e.g. candidate_41, candidate_12, candidate_50), project bullets
    # can fall under 'skills' or 'header' after column interleaving. Reconstruct bullets first
    # so wrapped continuation lines stay attached to their action-verb sentence.
    for sec in ("skills", "header", "certifications", "education"):
        if sec in resume.sections:
            sec_bullets = _reconstruct_bullets_from_lines(resume.sections[sec].splitlines())
            for b in sec_bullets:
                if SKILL_LIST_PREFIX_RE.match(b):
                    continue
                if ACTION_OR_PROSE_RE.search(b):
                    _add_line(b)

    # 3. Fallback if sections were empty
    if not collected:
        source_text = resume.raw_text if len(resume.raw_text.splitlines()) >= 5 else resume.plumber_text
        for b in _reconstruct_bullets_from_lines(source_text.splitlines()):
            if not SKILL_LIST_PREFIX_RE.match(b) and (ACTION_OR_PROSE_RE.search(b) or len(b) > 65):
                _add_line(b)

    return "\n".join(collected)


def _clean_summary_sentence(sentence: str, max_len: int = 340) -> str:
    """Normalize a reconstructed bullet into a clean, grammatically complete sentence."""
    s = re.sub(r"\s+", " ", sentence).strip().rstrip(" ,;:-–—")
    # Collapse accidental doubled PDF words like "channels channels" or "pipelinespipelines"
    s = re.sub(r"\b([A-Za-z]{4,})\1\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\b([A-Za-z]{3,})\s+\1\b", r"\1", s, flags=re.IGNORECASE)
    # Remove leading project title prefixes like "Project Name | Python, FastAPI " if attached
    s = re.sub(
        r"^[A-Z][A-Za-z0-9\s&/().-]{2,45}\s*\|\s*[A-Za-z0-9,\s./+-]{3,65}\s+(?=(?:Built|Designed|Architected|Developed|Engineered|Implemented|Created|Integrated|Orchestrated)\b)",
        "",
        s,
    )
    if len(s) > max_len:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if len(p.strip()) >= 45]
        if len(parts) > 1:
            best_parts: List[str] = []
            for p in parts:
                if len(p) <= max_len and ACTION_OR_PROSE_RE.search(p):
                    best_parts.append(p)
            if best_parts:
                s = best_parts[0]
            else:
                s = parts[0]
        if len(s) > max_len:
            cut = s[:max_len]
            last_semi = cut.rfind(";")
            last_comma = cut.rfind(",")
            boundary = max(last_semi, last_comma)
            if boundary >= 80:
                s = cut[:boundary]
            else:
                last_space = cut.rfind(" ")
                s = cut[:last_space] if last_space > 80 else cut
    s = DANGLING_TAIL_STRIP_RE.sub("", s).rstrip(" .,;:-–—")
    return f"{s}." if s else ""


def extract_concise_project_summary(resume: ParsedResume) -> str:
    """Extract an evidence-backed 1-2 sentence summary of the candidate's strongest AI/backend work."""
    raw_lines = [l for l in resume.raw_text.splitlines() if l.strip()]
    if len(raw_lines) < 5:
        raw_lines = [l for l in resume.plumber_text.splitlines() if l.strip()]

    merged_bullets = _reconstruct_bullets_from_lines(raw_lines)
    lines = [l for l in merged_bullets if len(l) > 40]

    # Prioritize action-verb sentences mentioning agentic/RAG/LLM/FastAPI architecture
    priority_keywords = [
        r"\b(?:langg?raph|multi[\s-]+agent|tool[\s-]+calling|mcp|livekit\s+agents?|crew\s*ai|autogen|claude\s+agent|agent\s+qa|voice\s+ai)\b",
        r"\b(?:rag\b|graph\s*rag|pgvector|qdrant|pinecone|chromadb|milvus|weaviate|faiss|semantic\s+search|nl[\s-]+to[\s-]+sql|text[\s-]+to[\s-]+sql)\b",
        r"\b(?:llm|openai|gemini|groq|ollama|vllm|llama|fastapi)\b",
    ]

    selected: List[str] = []
    for kw_pat in priority_keywords:
        for line in lines:
            if len(line) < 45 or len(line) > 560:
                continue
            if re.match(r"^(?:skills|technical\s+skills|languages|frameworks|tools|databases|education|certifications)\b", line, re.I):
                continue
            if SKILL_LIST_PREFIX_RE.match(line) or BIO_OR_OBJECTIVE_PREFIX_RE.match(line):
                continue
            if re.match(r"^[A-Z][A-Za-z0-9\s&/().:—–-]{2,55}\s*\|\s*[A-Za-z]", line):
                continue
            if re.search(kw_pat, line, re.IGNORECASE) and ACTION_OR_PROSE_RE.search(line):
                clean = _clean_summary_sentence(line)
                if clean and len(clean) >= 45 and clean not in selected:
                    selected.append(clean)
                if len(selected) >= 2:
                    break
        if len(selected) >= 2:
            break

    if not selected:
        for line in lines:
            if (
                45 <= len(line) <= 500
                and ACTION_OR_PROSE_RE.search(line)
                and not SKILL_LIST_PREFIX_RE.match(line)
                and not BIO_OR_OBJECTIVE_PREFIX_RE.match(line)
            ):
                clean = _clean_summary_sentence(line)
                if clean and len(clean) >= 45 and clean not in selected:
                    selected.append(clean)
                if len(selected) >= 2:
                    break

    if selected:
        return " ".join(selected[:2])
    return "Resume describes software development coursework and projects."


def _score_ai_project_depth(
    full_text: str, work_text: str
) -> Tuple[int, bool, int, str, List[str], List[str]]:
    """
    Score AI / Agentic / RAG Project Depth (0-40 points) and apply explicit 5-15 pt penalties
    for thin API-wrapper or shallow/tutorial projects (Section 4).
    """
    raw_pts = 0
    evidence_items: List[str] = []
    strengths: List[str] = []
    concerns: List[str] = []

    # 1. Agentic Orchestration & Tool Calling (up to 14 pts)
    has_agent_fw_work = bool(
        re.search(
            r"\b(?:langg?raph|crew\s*ai|autogen|google\s+adk|livekit\s+agents?|claude\s+agent\s+sdk|mcp|model\s+context\s+protocol)\b",
            work_text,
            re.I,
        )
    )
    has_agent_fw_any = bool(
        re.search(
            r"\b(?:langg?raph|langchain|llama\s*index|crew\s*ai|autogen|google\s+adk|livekit\s+agents?|claude\s+agent\s+sdk|haystack|mcp)\b",
            full_text,
            re.I,
        )
    )
    has_multi_agent_or_tools = bool(
        re.search(
            r"\b(?:multi[\s-]+agent|multi[\s-]+step\s+ai\s+agent|multi[\s-]+node\s+workflow|shared\s+graph\s+state|conditional\s+routing|agent[\s-]+based\s+architecture|tool[\s-]+calling|function[\s-]+calling|tool[\s-]+augmented|supervisor|stateful\s+graph|agent\s+orchestration|orchestrating.*agents|agent\s+qa\s+framework|llm[\s-]+to[\s-]+llm|multi[\s-]+step\s+gemini\s+pipeline|nl[\s-]+to[\s-]+sql|text[\s-]+to[\s-]+sql)\b",
            work_text,
            re.I,
        )
    )

    if has_agent_fw_work:
        raw_pts += 8
        evidence_items.append("Agentic orchestration framework applied in projects/work")
    elif has_agent_fw_any:
        raw_pts += 4
        evidence_items.append("AI orchestration framework listed")

    if has_multi_agent_or_tools:
        raw_pts += 6
        evidence_items.append("Multi-agent state or tool-calling architecture")
        strengths.append("Strong agentic / tool-calling workflow architecture")

    # 2. RAG & Vector Retrieval Depth (up to 13 pts)
    has_rag_work = bool(
        re.search(
            r"\b(?:rag\b|retrieval[\s-]+augmented|graph\s*rag|semantic\s+retrieval|semantic\s+search|hybrid\s+search)\b",
            work_text,
            re.I,
        )
    )
    has_vector_db_work = bool(
        re.search(
            r"\b(?:pgvector|qdrant|pinecone|chromadb|milvus|weaviate|faiss)\b",
            work_text,
            re.I,
        )
    )
    has_vector_db_any = bool(
        re.search(
            r"\b(?:pgvector|qdrant|pinecone|chromadb|milvus|weaviate|faiss|vector\s+database|vector\s+store)\b",
            full_text,
            re.I,
        )
    )
    has_advanced_retrieval = bool(
        re.search(
            r"\b(?:rerank(?:ing|er)?|chunking|docling|llama\s*parse|graph\s*rag|neo4j|cypher|schema\s+grounding|schema\s+compression|hybrid\s+retrieval|metadata\s+extraction)\b",
            work_text,
            re.I,
        )
    )
    has_llm_mention = bool(
        re.search(
            r"\b(?:llm|gpt|openai|gemini|groq|claude|llama|ollama|rag\b|langchain|langg?raph)\b",
            full_text,
            re.I,
        )
    )

    if has_rag_work:
        raw_pts += 6
        evidence_items.append("RAG / semantic retrieval pipeline implemented")
    elif re.search(r"\brag\b", full_text, re.I):
        raw_pts += 2

    if has_vector_db_work:
        raw_pts += 4
        evidence_items.append("Production vector database integrated in project/work")
        if has_rag_work or has_llm_mention:
            strengths.append("End-to-end RAG with vector database retrieval")
    elif has_vector_db_any:
        raw_pts += 2

    if has_advanced_retrieval:
        raw_pts += 3
        evidence_items.append("Advanced document parsing, chunking, reranking, or GraphRAG")

    # 3. Production AI Engineering, Evaluation & Inference Depth (up to 13 pts)
    has_eval_or_guardrails = bool(
        re.search(
            r"\b(?:hallucinations?|guardrails?|structured\s+(?:pydantic\s+)?outputs?|pydantic\s+(?:output\s+)?validation|llm\s+scor(?:ing|er)|llm[\s-]+as[\s-]+a[\s-]+judge|agent\s+qa|ragas|deepeval|eval(?:uation)?\s+(?:pipeline|suite|framework|harness|engine)|confidence\s+scor(?:ing|e)|token\s+(?:reduction|optimization)|prompt\s+(?:compression|caching)|self[\s-]+correcting)\b",
            work_text,
            re.I,
        )
    )
    has_local_or_voice_inference = bool(
        re.search(
            r"\b(?:ollama|vllm|h100|litellm|stt|tts|speech[\s-]+to[\s-]+text|voice\s+ai|ai\s+voice|freeswitch|deepgram|elevenlabs|whisper|multimodal\s+(?:rag|llm|agent|pipeline)|qwen|mistral)\b",
            work_text,
            re.I,
        )
    )
    ai_project_mentions = len(
        re.findall(
            r"\b(?:rag\b|langg?raph|langchain|llamaindex|pgvector|qdrant|pinecone|chromadb|milvus|weaviate|faiss|ollama|vllm|openai|gemini|groq|agentic|multi[\s-]+agent|mcp|voice\s+ai|agent\s+qa|llm[\s-]+to[\s-]+llm)\b",
            work_text,
            re.I,
        )
    )

    if has_eval_or_guardrails:
        raw_pts += 5
        evidence_items.append("Evaluation metrics, structured output validation, or hallucination mitigation")
        strengths.append("Production AI reliability (evaluation / structured validation / optimization)")

    if has_local_or_voice_inference:
        raw_pts += 4
        evidence_items.append("Self-hosted LLM, multi-provider routing, or real-time voice/multimodal AI")

    if ai_project_mentions >= 4:
        raw_pts += 4
        evidence_items.append("Multiple deep AI/agentic systems across projects/experience")
    elif ai_project_mentions >= 2:
        raw_pts += 2

    # Baseline floor if eligible via LLM API integration so penalty deduction is transparent
    if raw_pts < 16:
        raw_pts = max(raw_pts, 16)

    # 4. Project-Quality & Thin API-Wrapper Penalties (5-15 points, Section 4)
    is_thin_wrapper = False
    penalty = 0

    has_any_deep_ai = (
        has_agent_fw_work
        or has_multi_agent_or_tools
        or has_vector_db_work
        or has_advanced_retrieval
        or has_eval_or_guardrails
    )

    if not has_llm_mention and has_vector_db_work:
        # e.g., Computer vision face embedding search in pgvector without any LLM/RAG/agentic workflow
        is_thin_wrapper = True
        penalty = 8
        concerns.append("Vector search is used for CV face embeddings rather than LLM/RAG/agentic workflows (-8 pts)")
    elif not has_any_deep_ai and not has_rag_work:
        # Pure API wrapper around OpenAI/Gemini/Groq/LLaMA without RAG, vector store, evaluation, or agent orchestration
        is_thin_wrapper = True
        penalty = 10
        concerns.append("AI projects appear to be thin wrappers around LLM API calls without vector retrieval or agentic state (-10 pts)")
    elif (
        not has_vector_db_any
        and not has_multi_agent_or_tools
        and not has_eval_or_guardrails
        and not has_advanced_retrieval
    ):
        # Mentions RAG/LangChain/LangGraph briefly in a shallow POC without vector DB, tool calling, or evaluation details
        is_thin_wrapper = True
        penalty = 6
        concerns.append("Shallow AI project descriptions lacking vector DB, tool-calling, or evaluation details (-6 pts)")

    # Additional check: if primary AI project was built in Node.js/NestJS/Go while Python is secondary
    if not re.search(r"\b(?:fastapi|django|flask|pytorch|tensorflow|ollama|docling|haystack)\b", work_text, re.I):
        penalty = max(penalty, 5)
        concerns.append("Limited evidence of Python backend frameworks directly powering the AI pipeline (-5 pts)")

    final_ai_score = max(4, min(WEIGHT_AI_PROJECT_DEPTH, raw_pts - penalty))
    evidence_str = "; ".join(evidence_items) if evidence_items else "Basic LLM API integration"
    if penalty > 0:
        evidence_str += f" (Penalty -{penalty} pts applied for shallow/wrapper AI depth)"

    return final_ai_score, is_thin_wrapper, penalty, evidence_str, strengths, concerns


def _score_python_backend(
    full_text: str, work_text: str
) -> Tuple[int, str, List[str], List[str]]:
    """
    Score Python & Backend Engineering (0-30 points):
    Rewards Python, FastAPI, async programming, PostgreSQL, and Redis in projects/internships
    over keyword-only skill lists.
    """
    pts = 0
    ev: List[str] = []
    strengths: List[str] = []
    concerns: List[str] = []

    # 1. Python depth in projects/internships (up to 10 pts)
    py_work_hits = len(re.findall(r"\bpython\b", work_text, re.I))
    has_py_backend_work = bool(
        re.search(r"\b(?:fastapi|django|flask|asyncio|pydantic|sqlalchemy|celery)\b", work_text, re.I)
    )
    if (py_work_hits >= 2 and has_py_backend_work) or (
        has_py_backend_work and re.search(r"\bpython\b", full_text, re.I)
    ):
        pts += 10
        ev.append("Strong Python implementation across projects/internships")
    elif py_work_hits >= 1 or has_py_backend_work:
        pts += 7
        ev.append("Python used in project/internship implementation")
    else:
        pts += 3
        ev.append("Python listed primarily in skills section")
        concerns.append("Python appears mainly in skills section with limited project-level depth")

    # 2. FastAPI / Python Web Frameworks (up to 8 pts)
    if re.search(r"\bfast\s*api\b", work_text, re.I):
        pts += 7
        ev.append("FastAPI in projects/production work")
        if re.search(r"\b(?:django|flask|pydantic|sqlalchemy)\b", full_text, re.I):
            pts += 1
        strengths.append("Production FastAPI backend engineering")
    elif re.search(r"\bfast\s*api\b", full_text, re.I):
        pts += 4
        ev.append("FastAPI in technical skills")
    elif re.search(r"\b(?:django|flask)\b", work_text, re.I):
        pts += 4
        ev.append("Django/Flask backend in projects")
        concerns.append("Uses Flask/Django rather than FastAPI")
    else:
        concerns.append("No FastAPI backend evidence")

    # 3. Async programming / streaming / concurrency (up to 4 pts)
    if re.search(
        r"\b(?:asyncio|async\s+python|async\s+fastapi|asynchronous|sse\b|streaming|websockets?|webrtc|celery)\b",
        work_text,
        re.I,
    ):
        pts += 4
        ev.append("Async programming / real-time streaming in backend")
        strengths.append("Async Python / real-time streaming architecture")
    elif re.search(r"\b(?:async|websocket)\b", full_text, re.I):
        pts += 2
        ev.append("Async / WebSocket familiarity")
    else:
        concerns.append("Limited async Python evidence")

    # 4. PostgreSQL (up to 5 pts)
    if re.search(r"\b(?:postgresql|postgres|pgvector)\b", work_text, re.I):
        pts += 5
        ev.append("PostgreSQL used in projects/experience")
    elif re.search(r"\b(?:postgresql|postgres)\b", full_text, re.I):
        pts += 3
        ev.append("PostgreSQL listed in skills")
    elif re.search(r"\b(?:mysql|sqlite|sql|mongodb)\b", full_text, re.I):
        pts += 1
        concerns.append("No PostgreSQL evidence (uses MySQL/SQLite/MongoDB)")
    else:
        concerns.append("No PostgreSQL evidence")

    # 5. Redis (up to 3 pts)
    if re.search(r"\bredis\b", work_text, re.I):
        pts += 3
        ev.append("Redis used for caching/queues in projects/work")
    elif re.search(r"\bredis\b", full_text, re.I):
        pts += 2
        ev.append("Redis listed in skills")
    else:
        concerns.append("Limited Redis evidence")

    return min(WEIGHT_PYTHON_BACKEND, pts), "; ".join(ev), strengths, concerns


def _score_cloud_fullstack(
    full_text: str, work_text: str
) -> Tuple[int, str, List[str], List[str]]:
    """
    Score Cloud / Deployment / Full Stack (0-15 points):
    Rewards GCP, Docker, cloud deployment, and supporting React/Next.js full-stack evidence.
    """
    pts = 0
    ev: List[str] = []
    strengths: List[str] = []
    concerns: List[str] = []

    has_gcp = bool(re.search(r"\b(?:gcp|google\s+cloud|vertex\s+ai|cloud\s+run|bigquery)\b", full_text, re.I))
    has_aws_azure = bool(
        re.search(r"\b(?:aws|amazon\s+web\s+services|ec2|s3|aws\s+lambda|ecs|fargate|bedrock|azure)\b", full_text, re.I)
    )
    has_docker = bool(re.search(r"\bdocker\b", full_text, re.I))
    has_k8s = bool(re.search(r"\b(?:kubernetes|k8s)\b", full_text, re.I))
    has_cicd_deploy = bool(
        re.search(
            r"\b(?:ci/cd|github\s+actions|jenkins|argocd|vercel|render|digitalocean|railway|heroku|deployed|cloud[\s-]+native)\b",
            full_text,
            re.I,
        )
    )
    has_react_next = bool(re.search(r"\b(?:react(?:\.js|js)?|next(?:\.js|js))\b", full_text, re.I))
    has_ts = bool(re.search(r"\btypescript\b", full_text, re.I))

    if has_gcp:
        pts += 5
        ev.append("GCP / Google Cloud")
        if has_aws_azure:
            pts += 1
            ev.append("Multi-cloud (AWS/Azure)")
    elif has_aws_azure:
        pts += 4
        ev.append("AWS/Azure cloud infrastructure")

    if has_docker:
        pts += 4
        ev.append("Docker containerization")
    if has_k8s:
        pts += 1
        ev.append("Kubernetes")
    if has_cicd_deploy:
        pts += 2
        ev.append("CI/CD & cloud deployment")

    if has_react_next:
        pts += 2
        ev.append("React/Next.js full-stack frontend")
        if has_ts or re.search(r"\b(?:react|next\.js|nextjs)\b", work_text, re.I):
            pts += 1

    if pts >= 11:
        strengths.append("End-to-end cloud & Docker deployment with full-stack integration")
    elif not has_docker and not has_gcp and not has_aws_azure:
        concerns.append("Limited cloud infrastructure and Docker containerization evidence")

    return min(WEIGHT_CLOUD_FULLSTACK, pts), "; ".join(ev) or "Minimal cloud/deployment signals", strengths, concerns


def _score_engineering_depth(
    full_text: str, work_text: str
) -> Tuple[int, str, List[str]]:
    """
    Score Engineering Depth Signals (0-5 points):
    Testing, architecture, caching, queues, observability, concurrency, failure handling.
    """
    pts = 0
    ev: List[str] = []
    strengths: List[str] = []

    if re.search(
        r"\b(?:pytest|unit\s+test\w*|integration\s+test\w*|playwright|selenium|benchmark\w*|test\s+suite|eval(?:uation)?\s+(?:suite|framework|pipeline|harness))\b",
        full_text,
        re.I,
    ):
        pts += 1
        ev.append("Automated testing / benchmarking")

    if re.search(
        r"\b(?:microservices?|multi[\s-]+tenant|rbac|role[\s-]+based\s+access|event[\s-]+driven|system\s+design)\b",
        full_text,
        re.I,
    ):
        pts += 1
        ev.append("System architecture (microservices/multi-tenant/RBAC)")

    if re.search(
        r"\b(?:caching|redis|token\s+optimiz\w*|schema\s+compression|latency\s+by\s+\d+%|sub-\d+ms|\d+ms\s+query)\b",
        full_text,
        re.I,
    ):
        pts += 1
        ev.append("Caching & latency/performance optimization")

    if re.search(
        r"\b(?:kafka|celery|rabbitmq|bullmq|qstash|message\s+queues?|background\s+workers?)\b",
        full_text,
        re.I,
    ):
        pts += 1
        ev.append("Message queues / async background workers")

    if re.search(
        r"\b(?:grafana|prometheus|loki|promtail|kibana|opentelemetry|langsmith|cloudwatch|circuit\s+breakers?|idempoten\w*|rate[\s-]+limit\w*|retries|retry|exponential\s+backoff|air[\s-]+gapped)\b",
        full_text,
        re.I,
    ):
        pts += 1
        ev.append("Observability, telemetry & fault-handling patterns")

    if pts >= 4:
        strengths.append("Strong systems engineering depth (observability, fault tolerance, testing)")

    return min(WEIGHT_ENGINEERING_DEPTH, pts), "; ".join(ev) or "Standard application development", strengths


def evaluate_resume_deterministic(resume: ParsedResume) -> LLMResumeEvaluation:
    """
    Deterministic, evidence-extracting resume evaluator that produces a complete
    `LLMResumeEvaluation` structured object. Used both as a guardrail baseline and
    as the automatic fallback when no LLM API key is set or an API call fails.
    """
    full_text = f"{resume.raw_text}\n{resume.plumber_text}"
    work_text = _get_work_and_project_text(resume)

    ai_score, is_wrapper, penalty, ai_ev, ai_str, ai_con = _score_ai_project_depth(
        full_text, work_text
    )
    py_score, py_ev, py_str, py_con = _score_python_backend(full_text, work_text)
    cl_score, cl_ev, cl_str, cl_con = _score_cloud_fullstack(full_text, work_text)
    eng_score, eng_ev, eng_str = _score_engineering_depth(full_text, work_text)

    strengths = (ai_str + py_str + cl_str + eng_str)[:4]
    if not strengths:
        strengths = ["Meets baseline Python and AI eligibility criteria"]

    concerns = (ai_con + py_con + cl_con)[:3]
    if not concerns:
        concerns = ["No major technical gaps identified against internship rubric"]

    summary = extract_concise_project_summary(resume)

    return LLMResumeEvaluation(
        ai_project_depth=ai_score,
        python_backend=py_score,
        cloud_fullstack=cl_score,
        engineering_depth=eng_score,
        is_thin_api_wrapper=is_wrapper,
        wrapper_penalty_points=penalty,
        project_summary=summary,
        strengths=strengths,
        concerns=concerns,
        score_evidence={
            "ai_project_depth": ai_ev,
            "python_backend": py_ev,
            "cloud_fullstack": cl_ev,
            "engineering_depth": eng_ev,
        },
    )


def build_candidate_result(
    resume: ParsedResume,
    eligibility: EligibilityResult,
    github: GitHubEnrichment,
    llm_eval: Optional[LLMResumeEvaluation] = None,
) -> CandidateResult:
    """
    Build the final `CandidateResult` for either an eligible or rejected candidate.
    If `llm_eval` is provided from a live LLM call, blends it with the deterministic
    evaluation while enforcing category caps and thin-wrapper penalty rules.
    """
    if not eligibility.eligible:
        return CandidateResult(
            rank=None,
            candidate_name=resume.candidate_name,
            eligible=False,
            total_score=0,
            score_breakdown=None,
            matched_skills=resume.matched_skills,
            project_summary=extract_concise_project_summary(resume),
            github_summary=github.summary,
            strengths=[],
            concerns=eligibility.rejection_reasons,
            rejection_reasons=eligibility.rejection_reasons,
            file_name=resume.file_name,
            email=resume.email,
            github_url=resume.github_url,
            github_status=github.status,
            penalty_applied=0,
            score_evidence={},
        )

    det_eval = evaluate_resume_deterministic(resume)

    if llm_eval is not None:
        # Hybrid combination: average LLM judgment with deterministic evidence & enforce calibrated wrapper penalty
        penalty = det_eval.wrapper_penalty_points
        llm_ai_capped = (
            min(llm_eval.ai_project_depth, WEIGHT_AI_PROJECT_DEPTH - penalty)
            if penalty > 0
            else llm_eval.ai_project_depth
        )
        ai_pts = round((llm_ai_capped + det_eval.ai_project_depth) / 2)
        py_pts = round((llm_eval.python_backend + det_eval.python_backend) / 2)
        cl_pts = round((llm_eval.cloud_fullstack + det_eval.cloud_fullstack) / 2)
        eng_pts = round((llm_eval.engineering_depth + det_eval.engineering_depth) / 2)
        raw_llm_summary = re.sub(
            r"\b([A-Za-z]{2,})-\s+([a-z]{2,})\b",
            r"\1\2",
            llm_eval.project_summary or "",
        )
        cleaned_llm_summary = (
            _clean_summary_sentence(raw_llm_summary, max_len=420)
            if raw_llm_summary
            else ""
        )
        summary = (
            cleaned_llm_summary
            if len(cleaned_llm_summary) >= 40
            else det_eval.project_summary
        )
        strengths = (
            list(llm_eval.strengths)
            if llm_eval.strengths
            else list(det_eval.strengths)
        )
        concerns = (
            list(llm_eval.concerns)
            if llm_eval.concerns
            else list(det_eval.concerns)
        )
        if penalty > 0 and det_eval.concerns:
            for det_c in det_eval.concerns:
                if "pts)" in det_c and det_c not in concerns:
                    filtered_concerns = [
                        c
                        for c in concerns
                        if not re.search(
                            r"\b(?:thin\s+wrappers?|simple\s+llm\s+wrappers?|truncated)\b",
                            c,
                            re.I,
                        )
                    ]
                    concerns = [det_c] + filtered_concerns[:2]
                    break
        evidence = dict(det_eval.score_evidence)
        for k, v in llm_eval.score_evidence.items():
            if v and k in evidence:
                evidence[k] = f"{v} [Deterministic check: {det_eval.score_evidence[k]}]"
            elif v:
                evidence[k] = v
    else:
        ai_pts = det_eval.ai_project_depth
        py_pts = det_eval.python_backend
        cl_pts = det_eval.cloud_fullstack
        eng_pts = det_eval.engineering_depth
        penalty = det_eval.wrapper_penalty_points
        summary = det_eval.project_summary
        strengths = list(det_eval.strengths)
        concerns = list(det_eval.concerns)
        evidence = dict(det_eval.score_evidence)

    gh_pts = min(WEIGHT_GITHUB_ACTIVITY, max(0, github.score))
    evidence["github"] = github.summary

    if gh_pts >= 7 and "Active public GitHub portfolio" not in strengths and len(strengths) < 4:
        strengths.append(f"Active GitHub (@{github.username}, {github.python_ai_repos_count} Python/AI repos)")
    elif github.status in ("missing", "not_found") and len(concerns) < 3:
        concerns.append("Missing or unverifiable public GitHub profile")

    breakdown = ScoreBreakdown(
        ai_project_depth=min(WEIGHT_AI_PROJECT_DEPTH, max(0, ai_pts)),
        python_backend=min(WEIGHT_PYTHON_BACKEND, max(0, py_pts)),
        cloud_fullstack=min(WEIGHT_CLOUD_FULLSTACK, max(0, cl_pts)),
        github=gh_pts,
        engineering_depth=min(WEIGHT_ENGINEERING_DEPTH, max(0, eng_pts)),
    )

    total_score = (
        breakdown.ai_project_depth
        + breakdown.python_backend
        + breakdown.cloud_fullstack
        + breakdown.github
        + breakdown.engineering_depth
    )

    return CandidateResult(
        rank=None,  # Assigned after sorting all eligible candidates
        candidate_name=resume.candidate_name,
        eligible=True,
        total_score=total_score,
        score_breakdown=breakdown,
        matched_skills=resume.matched_skills,
        project_summary=summary,
        github_summary=github.summary,
        strengths=strengths,
        concerns=concerns,
        rejection_reasons=[],
        file_name=resume.file_name,
        email=resume.email,
        github_url=resume.github_url,
        github_status=github.status,
        penalty_applied=penalty,
        score_evidence=evidence,
    )
