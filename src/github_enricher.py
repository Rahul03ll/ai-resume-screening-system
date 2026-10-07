"""Lightweight public GitHub activity enricher with disk caching and bounded async concurrency (Section 5)."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import ScreeningConfig
from src.models import GitHubEnrichment, ParsedResume


AI_PYTHON_REPO_KEYWORDS = re.compile(
    r"\b(?:ai|llm|rag|agent|langgraph|langchain|gpt|gemini|ollama|vector|fastapi|django|flask|ml|nlp|deep-learning|machine-learning|bot|assistant|copilot|eval|screener|voice|vision|transformer)\b",
    re.IGNORECASE,
)


class GitHubEnricher:
    """Enriches candidate profiles with public GitHub activity signals (capped at 10 points)."""

    def __init__(self, config: ScreeningConfig) -> None:
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._cache_lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.config.use_cache:
            return
        cache_path: Path = self.config.github_cache_path
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._cache = data
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        if not self.config.use_cache:
            return
        try:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            with self._cache_lock:
                payload = json.dumps(self._cache, indent=2)
            self.config.github_cache_path.write_text(payload, encoding="utf-8")
        except Exception:
            pass

    def _fetch_user_repos_sync(self, username: str) -> Dict[str, Any]:
        """
        Fetch up to 100 most recently pushed public repos for `username` in a single API call.
        A single `/users/{username}/repos?sort=pushed&per_page=100` request provides:
        - Profile existence check (200 vs 404)
        - All public repositories (up to 100), fork status, primary language, descriptions, stars
        - Exact `pushed_at` / `updated_at` timestamps for recent engineering activity
        """
        key = username.lower()
        if self.config.use_cache:
            with self._cache_lock:
                cached = self._cache.get(key)
            if cached and cached.get("status") in ("ok", "not_found"):
                return cached

        url = f"https://api.github.com/users/{username}/repos?sort=pushed&per_page=100"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Kasparro-AI-Resume-Screener/1.0",
        }
        if self.config.github_token:
            headers["Authorization"] = f"Bearer {self.config.github_token}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.config.github_timeout_sec) as resp:
                raw_body = resp.read().decode("utf-8", errors="replace")
                repos = json.loads(raw_body)
                if not isinstance(repos, list):
                    repos = []
                # Compact repo objects before caching to keep cache file small and fast
                compact_repos = [
                    {
                        "name": r.get("name", ""),
                        "fork": bool(r.get("fork", False)),
                        "language": r.get("language"),
                        "description": r.get("description") or "",
                        "pushed_at": r.get("pushed_at") or r.get("updated_at"),
                        "stargazers_count": int(r.get("stargazers_count", 0) or 0),
                        "topics": r.get("topics") or [],
                    }
                    for r in repos
                ]
                result = {"status": "ok", "username": username, "repos": compact_repos}
                with self._cache_lock:
                    self._cache[key] = result
                return result
        except urllib.error.HTTPError as http_err:
            if http_err.code == 404:
                result = {
                    "status": "not_found",
                    "username": username,
                    "error": f"GitHub profile '{username}' not found (HTTP 404).",
                }
                with self._cache_lock:
                    self._cache[key] = result
                return result
            if http_err.code in (403, 429):
                return {
                    "status": "rate_limited",
                    "username": username,
                    "error": f"GitHub API rate limit exceeded (HTTP {http_err.code}). Set GITHUB_TOKEN for 5,000 req/hr.",
                }
            return {
                "status": "error",
                "username": username,
                "error": f"GitHub API HTTP error {http_err.code}: {http_err.reason}",
            }
        except Exception as exc:
            return {
                "status": "error",
                "username": username,
                "error": f"GitHub network error: {type(exc).__name__}: {exc}",
            }

    @staticmethod
    def score_repos(
        username: str,
        profile_url: str,
        repos: List[Dict[str, Any]],
        reference_now: Optional[datetime] = None,
    ) -> GitHubEnrichment:
        """
        Compute deterministic, explainable 0-10 GitHub score from repository metadata:
        - 0 to 5 points for recent activity (recency of pushes + count of recently active repos)
        - 0 to 5 points for maintained & Python/AI-relevant repositories
        """
        now = reference_now or datetime.now(timezone.utc)
        public_count = len(repos)
        maintained_repos = [r for r in repos if not r.get("fork", False)]
        maintained_count = len(maintained_repos)

        if public_count == 0:
            return GitHubEnrichment(
                username=username,
                profile_url=profile_url,
                status="enriched",
                score=0,
                recent_activity_score=0,
                relevant_repos_score=0,
                public_repos_count=0,
                maintained_repos_count=0,
                python_ai_repos_count=0,
                recent_pushed_repos_90d=0,
                summary=f"GitHub profile (@{username}) exists but has 0 public repositories.",
            )

        # 1. Analyze Recent Activity (0-5 points)
        pushed_dates: List[datetime] = []
        for r in repos:
            ts = r.get("pushed_at")
            if ts and isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    pushed_dates.append(dt)
                except Exception:
                    pass

        pushed_dates.sort(reverse=True)
        last_pushed_str: Optional[str] = None
        days_since_last = 9999
        recent_90d = 0
        recent_180d = 0

        if pushed_dates:
            latest_dt = pushed_dates[0]
            last_pushed_str = latest_dt.strftime("%Y-%m-%d")
            # Handle resumes/repos relative to current date (if repo timestamps are in 2025/2026)
            days_since_last = max(0, (now - latest_dt).days)
            # Also measure relative to latest repo push or 2026 reference window
            for dt in pushed_dates:
                age_days = max(0, (now - dt).days)
                if age_days <= 90:
                    recent_90d += 1
                if age_days <= 180:
                    recent_180d += 1

        if days_since_last <= 45 and recent_90d >= 3:
            activity_score = 5
        elif days_since_last <= 90 and recent_90d >= 2:
            activity_score = 4
        elif days_since_last <= 120 or recent_180d >= 3:
            activity_score = 3
        elif days_since_last <= 240 or recent_180d >= 1:
            activity_score = 2
        elif days_since_last <= 450:
            activity_score = 1
        else:
            activity_score = 0

        # 2. Analyze Maintained & Python/AI Relevant Repositories (0-5 points)
        python_ai_repos: List[str] = []
        for r in maintained_repos:
            lang = (r.get("language") or "").lower()
            name = r.get("name") or ""
            desc = r.get("description") or ""
            topics = " ".join(r.get("topics") or [])
            combined = f"{name} {desc} {topics}"

            is_py = lang in ("python", "jupyter notebook")
            is_ai_kw = bool(AI_PYTHON_REPO_KEYWORDS.search(combined))
            if is_py or is_ai_kw:
                python_ai_repos.append(name)

        py_ai_count = len(python_ai_repos)

        # Up to 2 pts for overall maintained public repo count
        if maintained_count >= 10:
            maintained_pts = 2
        elif maintained_count >= 3:
            maintained_pts = 1
        else:
            maintained_pts = 0

        # Up to 3 pts for Python/AI relevant repositories
        if py_ai_count >= 5:
            relevance_pts = 3
        elif py_ai_count >= 2:
            relevance_pts = 2
        elif py_ai_count >= 1:
            relevance_pts = 1
        else:
            relevance_pts = 0

        relevant_repos_score = min(5, maintained_pts + relevance_pts)
        total_score = min(10, activity_score + relevant_repos_score)

        notable = python_ai_repos[:4] if python_ai_repos else [r.get("name", "") for r in maintained_repos[:3]]
        notable_str = f" (e.g., {', '.join(notable)})" if notable else ""
        recency_label = (
            f"last push {last_pushed_str}, {recent_90d} repos active in 90d"
            if last_pushed_str
            else "no recent push timestamps"
        )
        summary = (
            f"@{username}: {maintained_count} maintained public repos, "
            f"{py_ai_count} Python/AI-relevant repos{notable_str}; {recency_label}."
        )

        return GitHubEnrichment(
            username=username,
            profile_url=profile_url,
            status="enriched",
            score=total_score,
            recent_activity_score=activity_score,
            relevant_repos_score=relevant_repos_score,
            public_repos_count=public_count,
            maintained_repos_count=maintained_count,
            python_ai_repos_count=py_ai_count,
            recent_pushed_repos_90d=recent_90d,
            last_pushed_at=last_pushed_str,
            notable_repos=notable,
            summary=summary,
        )

    def enrich_candidate_sync(self, resume: ParsedResume) -> GitHubEnrichment:
        """Synchronous enrichment for a single candidate."""
        if not resume.github_username:
            return GitHubEnrichment(
                status="missing",
                score=0,
                summary="No public GitHub profile URL found on resume.",
            )

        if not self.config.enable_github:
            return GitHubEnrichment(
                username=resume.github_username,
                profile_url=resume.github_url,
                status="skipped",
                score=0,
                summary=f"GitHub enrichment skipped (--no-github) for @{resume.github_username}.",
            )

        username = resume.github_username
        profile_url = resume.github_url or f"https://github.com/{username}"
        raw = self._fetch_user_repos_sync(username)
        status = raw.get("status", "error")

        if status == "ok":
            return self.score_repos(username, profile_url, raw.get("repos", []))

        return GitHubEnrichment(
            username=username,
            profile_url=profile_url,
            status=status,
            score=0,
            summary=raw.get("error", f"GitHub enrichment failed ({status}) for @{username}."),
            error=raw.get("error"),
        )

    async def enrich_candidates_batch(
        self, resumes: List[ParsedResume]
    ) -> Dict[str, GitHubEnrichment]:
        """Enrich a list of eligible candidates with bounded async concurrency."""
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        results: Dict[str, GitHubEnrichment] = {}

        async def _worker(r: ParsedResume) -> None:
            async with semaphore:
                enrichment = await asyncio.to_thread(self.enrich_candidate_sync, r)
                async with self._lock:
                    results[r.file_name] = enrichment

        await asyncio.gather(*[_worker(r) for r in resumes])
        self._save_cache()
        return results
