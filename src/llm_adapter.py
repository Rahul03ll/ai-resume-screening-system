"""Provider-agnostic LLM adapter with Pydantic structured output validation and disk caching (Section 6)."""

from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import ScreeningConfig
from src.models import LLMResumeEvaluation, ParsedResume

GROQ_FALLBACK_MODELS: List[str] = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]


SYSTEM_PROMPT = """You are a senior staff backend & AI engineer evaluating SDE Intern resumes.
Score the eligible candidate strictly against the following rubric and return ONLY valid JSON matching the schema:
1. ai_project_depth (0-40): Reward real AI systems (agents, LangGraph, CrewAI, tool calling, RAG, vector DBs like pgvector/Qdrant/Pinecone/Chroma/Milvus, reranking, evaluation, state management).
   PENALTY RULE: Deduct 5-15 points when an 'AI project' is only a thin wrapper around an LLM/API call with no meaningful retrieval, state management, orchestration, or evaluation, or when frameworks only appear in the skills list.
2. python_backend (0-30): Reward Python, FastAPI, async programming (asyncio/streaming), PostgreSQL, and Redis in projects/internships over keyword-only skill lists.
3. cloud_fullstack (0-15): Reward GCP, Docker, Kubernetes, CI/CD, cloud deployment (AWS/Azure/Render/Vercel), and supporting React/Next.js full-stack integration.
4. engineering_depth (0-5): Reward testing (pytest), architecture (microservices/RBAC/multi-tenant), caching, queues (Kafka/Celery), observability (Grafana/Prometheus/logging), concurrency, and failure handling.

Return JSON with exact keys:
- ai_project_depth (int 0-40)
- python_backend (int 0-30)
- cloud_fullstack (int 0-15)
- engineering_depth (int 0-5)
- is_thin_api_wrapper (bool)
- wrapper_penalty_points (int 0-15)
- project_summary (string, concise 1-2 sentence evidence-backed summary of AI & backend work)
- strengths (list of 2-4 short strings)
- concerns (list of 1-3 short strings)
- score_evidence (dict mapping category names to 1-sentence evidence strings)
"""


def _normalize_unicode_text(text: str) -> str:
    """Replace non-breaking hyphens/spaces and broken PDF word wraps with clean ASCII-safe punctuation."""
    s = (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", " - ")
        .replace("\u202f", " ")
        .replace("\u00a0", " ")
    )
    s = re.sub(r"\b([A-Za-z]{2,})-\s+([a-z]{2,})\b", r"\1\2", s)
    return re.sub(r"\s+", " ", s).strip()


def _clamp_int(val: Any, low: int, high: int, use_abs: bool = False) -> int:
    try:
        num = int(round(float(val)))
        if use_abs:
            num = abs(num)
        return max(low, min(high, num))
    except Exception:
        return low


def _parse_and_validate_eval(raw_json: str) -> LLMResumeEvaluation:
    """Parse raw LLM JSON output, strip optional markdown fences, normalize bounds, and validate with Pydantic."""
    cleaned = raw_json.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object")

    raw_evidence = data.get("score_evidence") or {}
    norm_evidence: Dict[str, str] = {}
    if isinstance(raw_evidence, dict):
        for k, v in raw_evidence.items():
            if isinstance(v, list):
                norm_evidence[str(k)] = _normalize_unicode_text(
                    "; ".join(str(x) for x in v if x)
                )
            elif v is not None:
                norm_evidence[str(k)] = _normalize_unicode_text(str(v))

    raw_strengths = data.get("strengths") or []
    strengths = (
        [
            _normalize_unicode_text(str(s))
            for s in raw_strengths
            if _normalize_unicode_text(str(s))
        ][:4]
        if isinstance(raw_strengths, list)
        else []
    )
    raw_concerns = data.get("concerns") or []
    concerns = (
        [
            _normalize_unicode_text(str(c))
            for c in raw_concerns
            if _normalize_unicode_text(str(c))
        ][:3]
        if isinstance(raw_concerns, list)
        else []
    )

    normalized = {
        "ai_project_depth": _clamp_int(data.get("ai_project_depth", 0), 0, 40),
        "python_backend": _clamp_int(data.get("python_backend", 0), 0, 30),
        "cloud_fullstack": _clamp_int(data.get("cloud_fullstack", 0), 0, 15),
        "engineering_depth": _clamp_int(data.get("engineering_depth", 0), 0, 5),
        "is_thin_api_wrapper": bool(data.get("is_thin_api_wrapper", False)),
        "wrapper_penalty_points": _clamp_int(
            data.get("wrapper_penalty_points", 0), 0, 15, use_abs=True
        ),
        "project_summary": _normalize_unicode_text(str(data.get("project_summary") or "")),
        "strengths": strengths,
        "concerns": concerns,
        "score_evidence": norm_evidence,
    }
    return LLMResumeEvaluation.model_validate(normalized)


class LLMProviderAdapter:
    """
    Small provider-agnostic adapter for structured LLM resume evaluation.
    Supports Groq, OpenAI, and Google Gemini via environment variable API keys.
    Falls back gracefully on any failure or when no API key is configured.
    """

    def __init__(self, config: ScreeningConfig) -> None:
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.config.use_cache:
            return
        path: Path = self.config.llm_cache_path
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._cache = data
            except Exception:
                self._cache = {}

    def save_cache(self) -> None:
        if not self.config.use_cache or not self._cache:
            return
        try:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = json.dumps(self._cache, indent=2, ensure_ascii=False)
            self.config.llm_cache_path.write_text(payload, encoding="utf-8")
        except Exception:
            pass

    @property
    def active_provider(self) -> Optional[str]:
        """Resolve which LLM provider is available based on config and environment keys."""
        if not self.config.enable_llm:
            return None
        pref = self.config.llm_provider
        if pref == "groq" and self.config.groq_api_key:
            return "groq"
        if pref == "openai" and self.config.openai_api_key:
            return "openai"
        if pref == "gemini" and self.config.gemini_api_key:
            return "gemini"
        if pref == "auto":
            if self.config.groq_api_key:
                return "groq"
            if self.config.openai_api_key:
                return "openai"
            if self.config.gemini_api_key:
                return "gemini"
        return None

    def evaluate_resume(self, resume: ParsedResume) -> Optional[LLMResumeEvaluation]:
        """
        Evaluate a resume using the configured LLM provider and validate output against
        the Pydantic `LLMResumeEvaluation` schema. Returns None if no key is configured
        or if the API call fails, allowing seamless fallback to deterministic scoring.
        """
        provider = self.active_provider
        if not provider:
            return None

        cache_key = f"{provider}:{self.config.llm_model}:{resume.content_hash}"
        if self.config.use_cache and cache_key in self._cache:
            try:
                return _parse_and_validate_eval(json.dumps(self._cache[cache_key]))
            except Exception:
                pass

        source_text = (
            resume.raw_text
            if len(resume.raw_text.strip()) >= 50
            else resume.plumber_text
        )
        compact_text = re.sub(r"\n{2,}", "\n", source_text).strip()[:11000]

        user_prompt = (
            f"Candidate Name: {resume.candidate_name}\n"
            f"Matched Skills: {', '.join(resume.matched_skills)}\n\n"
            f"Resume Text:\n{compact_text}"
        )

        try:
            if provider == "groq":
                parsed_eval = self._call_groq_with_fallback(
                    api_key=self.config.groq_api_key or "",
                    preferred_model=self.config.llm_model or "openai/gpt-oss-120b",
                    user_prompt=user_prompt,
                )
            elif provider == "openai":
                model = (
                    self.config.llm_model
                    if "/" not in self.config.llm_model and "gpt" in self.config.llm_model
                    else "gpt-4o-mini"
                )
                raw_json = self._call_openai_compatible(
                    url="https://api.openai.com/v1/chat/completions",
                    api_key=self.config.openai_api_key or "",
                    model=model,
                    user_prompt=user_prompt,
                )
                parsed_eval = _parse_and_validate_eval(raw_json)
            elif provider == "gemini":
                model = (
                    self.config.llm_model
                    if "gemini" in self.config.llm_model
                    else "gemini-2.5-flash"
                )
                raw_json = self._call_gemini(
                    api_key=self.config.gemini_api_key or "",
                    model=model,
                    user_prompt=user_prompt,
                )
                parsed_eval = _parse_and_validate_eval(raw_json)
            else:
                return None

            if self.config.use_cache:
                with self._lock:
                    self._cache[cache_key] = parsed_eval.model_dump()
            return parsed_eval
        except Exception:
            # Fail gracefully per resume (Section 6 requirement)
            return None

    def _call_groq_with_fallback(
        self, api_key: str, preferred_model: str, user_prompt: str
    ) -> LLMResumeEvaluation:
        """Call Groq chat completions with automatic model fallback, JSON validation, and rate-limit retry."""
        models: List[str] = [preferred_model] + [
            m for m in GROQ_FALLBACK_MODELS if m != preferred_model
        ]
        last_err: Optional[Exception] = None
        for attempt in range(6):
            rate_limited = False
            for model in models:
                try:
                    raw_json = self._call_openai_compatible(
                        url="https://api.groq.com/openai/v1/chat/completions",
                        api_key=api_key,
                        model=model,
                        user_prompt=user_prompt,
                    )
                    return _parse_and_validate_eval(raw_json)
                except urllib.error.HTTPError as e:
                    last_err = e
                    if e.code == 429:
                        rate_limited = True
                        continue
                    if e.code in (400, 404, 500, 502, 503):
                        continue
                    raise
                except (ValueError, KeyError, json.JSONDecodeError) as e:
                    # If one model returns malformed/truncated JSON, try next fallback model
                    last_err = e
                    continue
            if rate_limited and attempt < 5:
                time.sleep(3.0 * (attempt + 1) + random.uniform(0.1, 0.8))
            else:
                break
        if last_err is not None:
            raise last_err
        raise RuntimeError("Groq evaluation failed across all candidate models")

    def _call_openai_compatible(
        self, url: str, api_key: str, model: str, user_prompt: str
    ) -> str:
        payload = {
            "model": model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Kasparro-AI-Resume-Screener/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.llm_timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]

    def _call_gemini(self, api_key: str, model: str, user_prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.llm_timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["candidates"][0]["content"]["parts"][0]["text"]
