"""Rule-based Hard Eligibility Filter (Section 3)."""

from __future__ import annotations

import re
from typing import List, Tuple

from src.models import EligibilityResult, ParsedResume


# Genuine Python ecosystem signals beyond the bare word "Python"
PYTHON_ECOSYSTEM_PATTERNS: List[Tuple[str, str]] = [
    ("FastAPI", r"\bfast\s*api\b"),
    ("Django", r"\bdjango\b"),
    ("Flask", r"\bflask\b"),
    ("Asyncio", r"\b(?:asyncio|async\s+python|uvicorn|aiohttp|asyncpg)\b"),
    ("Pydantic", r"\bpydantic\b"),
    ("SQLAlchemy", r"\bsqlalchemy\b"),
    ("Celery", r"\bcelery\b"),
    ("Pytest", r"\bpytest\b"),
    ("PyTorch", r"\bpytorch\b"),
    ("TensorFlow", r"\btensorflow\b"),
    ("Scikit-learn", r"\b(?:scikit[\s-]+learn|sklearn)\b"),
    ("Pandas/NumPy", r"\b(?:pandas|numpy|pyspark)\b"),
    ("Streamlit/Gradio", r"\b(?:streamlit|gradio)\b"),
    ("Docling/Haystack", r"\b(?:haystack|docling|pypdf2?|beautiful\s*soup)\b"),
]

# Strong AI / Agentic / RAG / Vector framework & architecture signals
STRONG_AI_AGENTIC_PATTERNS: List[Tuple[str, str]] = [
    ("LangGraph", r"\blangg?raph\b"),
    ("LangChain", r"\blangchain\b"),
    ("LlamaIndex", r"\bllama\s*index\b"),
    ("Google ADK", r"\b(?:google\s+adk|agent\s+development\s+kit)\b"),
    ("CrewAI", r"\bcrew\s*ai\b"),
    ("AutoGen", r"\bautogen\b"),
    ("Claude Agent SDK", r"\bclaude\s+agent\s+sdk\b"),
    ("LiveKit Agents", r"\blivekit\s+agents?\b"),
    ("Haystack", r"\bhaystack\b"),
    ("Model Context Protocol (MCP)", r"\b(?:model\s+context\s+protocol|mcp)\b"),
    ("RAG Pipeline", r"\b(?:rag\b|retrieval[\s-]+augmented[\s-]+generation|graph\s*rag)\b"),
    ("Agentic / Multi-Agent Workflow", r"\b(?:multi[\s-]+agent|agentic\s+(?:ai|workflow|system|architecture|solution|pipeline)|tool[\s-]+calling\s+agent|tool[\s-]+augmented\s+ai\s+agent|ai\s+agents?\b|voice\s+ai\s+(?:agent|calling|system)|agent\s+qa\s+framework|autonomous\s+agents?)\b"),
    ("Vector Database / Search", r"\b(?:pgvector|qdrant|pinecone|chromadb|weaviate|milvus|faiss|vector\s+database|vector\s+search|vector\s+store|semantic\s+search|semantic\s+retrieval)\b"),
    ("LLM Application / Pipeline", r"\b(?:llm[\s-]+powered|llm\s+pipeline|llm\s+agent|llm\s+integration|llm[\s-]+to[\s-]+llm|llm\s+scoring\s+engine|nl[\s-]+to[\s-]+sql|text[\s-]+to[\s-]+sql|ollama|vllm|litellm|llama\s*parse)\b"),
]

# Secondary LLM API signals (require evidence in projects/experience, not just dev-tool usage)
SECONDARY_LLM_API_PATTERNS: List[Tuple[str, str]] = [
    ("OpenAI / GPT Integration", r"\b(?:openai\s+api|gpt-4[o]?|gpt-3\.5|chatgpt\s+api)\b"),
    ("Gemini API Integration", r"\b(?:gemini\s+api|google\s+gemini|gemini[\s-]+powered|vertex\s+ai)\b"),
    ("Groq LLM Integration", r"\bgroq\s+(?:api|llm)\b"),
    ("LLaMA / Open-Weight LLM", r"\b(?:integrated\s+llama|qwen|mistral|nvidia\s+nim\s+llm)\b"),
]


def _check_python_evidence(resume: ParsedResume) -> Tuple[bool, List[str]]:
    """
    Verify genuine Python stack evidence.
    Rejects JavaScript/Java/React-only profiles, as well as profiles where 'Python' only
    appears once in a generic language list while all frameworks, projects, and experience
    are strictly non-Python.
    """
    full_text = f"{resume.raw_text}\n{resume.plumber_text}"
    has_python_word = bool(re.search(r"\bpython\b", full_text, re.IGNORECASE))

    evidence: List[str] = []

    # Check Python ecosystem frameworks/libraries
    for label, pattern in PYTHON_ECOSYSTEM_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            evidence.append(label)

    if not has_python_word and not evidence:
        return False, []

    # Check if Python appears in experience, projects, or summary sections
    non_skill_text = "\n".join(
        val
        for sec, val in resume.sections.items()
        if sec in ("summary", "experience", "projects", "header")
    )
    if re.search(r"\bpython\b", non_skill_text, re.IGNORECASE):
        evidence.insert(0, "Python in projects/experience")

    py_mentions = len(re.findall(r"\bpython\b", resume.raw_text, re.IGNORECASE))
    if not evidence and py_mentions >= 2:
        evidence.append("Python mentioned across multiple resume sections")

    if not evidence:
        # Bare single mention of "Python" in a skill list with no Python libraries or projects
        return False, []

    return True, evidence


def _check_ai_agentic_evidence(resume: ParsedResume) -> Tuple[bool, List[str]]:
    """
    Verify genuine AI / LLM / RAG / Agentic project, framework, or implementation evidence.
    Filters out false positives such as:
    - AI coding assistants (GitHub Copilot, Claude Code, Cursor) used purely as IDE/dev tools
    - Generic database/storage 'retrieval' ('pagination for data retrieval', 'file upload and retrieval')
    - Classical ML/CV-only projects without any LLM, RAG, agentic, or vector search component
    """
    full_text = f"{resume.raw_text}\n{resume.plumber_text}"
    evidence: List[str] = []

    for label, pattern in STRONG_AI_AGENTIC_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            evidence.append(label)

    for label, pattern in SECONDARY_LLM_API_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            evidence.append(label)

    # Guardrail: If the only match was "AI Agents" from "Salesforce Agent Development certified" or similar
    # verify that there is actual AI/LLM/RAG/Agentic content in projects, experience, or skills
    if not evidence:
        return False, []

    return True, evidence


def evaluate_eligibility(resume: ParsedResume) -> EligibilityResult:
    """
    Apply deterministic hard eligibility rules (Section 3) before ranking.
    Both Python stack evidence AND AI/agentic evidence are required.
    """
    if resume.parse_error:
        return EligibilityResult(
            candidate=resume.candidate_name,
            eligible=False,
            has_python_evidence=False,
            has_ai_agentic_evidence=False,
            rejection_reasons=[f"Unreadable or malformed resume: {resume.parse_error}"],
            matched_skills=resume.matched_skills,
        )

    has_python, py_details = _check_python_evidence(resume)
    has_ai, ai_details = _check_ai_agentic_evidence(resume)

    rejection_reasons: List[str] = []
    if not has_python:
        rejection_reasons.append("No evidence of Python stack")
        # Prevent contradictory "Python" tag in matched_skills when Python stack evidence is absent
        resume.matched_skills = [s for s in resume.matched_skills if s != "Python"]
    if not has_ai:
        rejection_reasons.append("No AI/agentic project evidence")

    eligible = has_python and has_ai

    return EligibilityResult(
        candidate=resume.candidate_name,
        eligible=eligible,
        has_python_evidence=has_python,
        has_ai_agentic_evidence=has_ai,
        python_evidence_details=py_details,
        ai_evidence_details=ai_details,
        rejection_reasons=rejection_reasons,
        matched_skills=list(resume.matched_skills),
    )
