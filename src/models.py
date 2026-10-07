"""Pydantic v2 data models for resume parsing, eligibility, LLM structured output, GitHub enrichment, and final output."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ParsedResume(BaseModel):
    """Structured representation of an ingested resume file."""

    file_name: str
    file_path: str
    content_hash: str = ""
    candidate_name: str = "Unknown Candidate"
    email: Optional[str] = None
    phone: Optional[str] = None
    github_url: Optional[str] = None
    github_username: Optional[str] = None
    github_repo_urls: List[str] = Field(default_factory=list)
    raw_text: str = ""
    plumber_text: str = ""
    hyperlinks: List[str] = Field(default_factory=list)
    sections: Dict[str, str] = Field(default_factory=dict)
    matched_skills: List[str] = Field(default_factory=list)
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    parse_error: Optional[str] = None


class EligibilityResult(BaseModel):
    """Hard eligibility evaluation result (Section 3)."""

    candidate: str
    eligible: bool
    has_python_evidence: bool
    has_ai_agentic_evidence: bool
    python_evidence_details: List[str] = Field(default_factory=list)
    ai_evidence_details: List[str] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)


class GitHubEnrichment(BaseModel):
    """Public GitHub activity enrichment result (Section 5)."""

    username: Optional[str] = None
    profile_url: Optional[str] = None
    status: str = "missing"  # enriched, missing, not_found, rate_limited, error, skipped
    score: int = Field(default=0, ge=0, le=10)
    recent_activity_score: int = Field(default=0, ge=0, le=5)
    relevant_repos_score: int = Field(default=0, ge=0, le=5)
    public_repos_count: int = 0
    maintained_repos_count: int = 0
    python_ai_repos_count: int = 0
    recent_pushed_repos_90d: int = 0
    last_pushed_at: Optional[str] = None
    notable_repos: List[str] = Field(default_factory=list)
    summary: str = "No public GitHub profile found on resume."
    error: Optional[str] = None


class LLMResumeEvaluation(BaseModel):
    """Structured output schema for LLM semantic extraction & project-quality judgment (Section 6)."""

    ai_project_depth: int = Field(
        ...,
        ge=0,
        le=40,
        description="Score (0-40) for AI / Agentic / RAG project depth after deducting shallow API-wrapper penalties.",
    )
    python_backend: int = Field(
        ...,
        ge=0,
        le=30,
        description="Score (0-30) for Python, FastAPI, async programming, PostgreSQL, and Redis evidence.",
    )
    cloud_fullstack: int = Field(
        ...,
        ge=0,
        le=15,
        description="Score (0-15) for GCP, Docker, cloud deployment, and supporting React/Next.js full-stack evidence.",
    )
    engineering_depth: int = Field(
        ...,
        ge=0,
        le=5,
        description="Score (0-5) for testing, architecture, caching, queues, observability, concurrency, and fault handling.",
    )
    is_thin_api_wrapper: bool = Field(
        default=False,
        description="True if the candidate's AI project is mainly a thin wrapper around an LLM API call without retrieval, orchestration, or state.",
    )
    wrapper_penalty_points: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Points deducted (0 or 5-15) for shallow LLM API wrapper or tutorial-only AI projects.",
    )
    project_summary: str = Field(
        ...,
        description="Short evidence-backed summary of the candidate's AI/agentic and backend projects/experience.",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="2-4 concise, evidence-backed engineering strengths.",
    )
    concerns: List[str] = Field(
        default_factory=list,
        description="1-3 concise concerns or gaps (e.g., missing Redis/GCP, thin wrapper, limited backend depth).",
    )
    score_evidence: Dict[str, str] = Field(
        default_factory=dict,
        description="Category-by-category explanation with concrete evidence from the resume.",
    )


class ScoreBreakdown(BaseModel):
    """100-point score breakdown across the 5 required categories (Section 4 & 7)."""

    ai_project_depth: int = Field(default=0, ge=0, le=40)
    python_backend: int = Field(default=0, ge=0, le=30)
    cloud_fullstack: int = Field(default=0, ge=0, le=15)
    github: int = Field(default=0, ge=0, le=10)
    engineering_depth: int = Field(default=0, ge=0, le=5)


class CandidateResult(BaseModel):
    """Final candidate screening record matching Section 7 Required Output."""

    rank: Optional[int] = None
    candidate_name: str
    eligible: bool
    total_score: int = 0
    score_breakdown: Optional[ScoreBreakdown] = None
    matched_skills: List[str] = Field(default_factory=list)
    project_summary: str = ""
    github_summary: str = ""
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    # Additional traceability metadata
    file_name: str = ""
    email: Optional[str] = None
    github_url: Optional[str] = None
    github_status: str = "missing"
    penalty_applied: int = 0
    score_evidence: Dict[str, str] = Field(default_factory=dict)


class FailedResumeRecord(BaseModel):
    """Record for a malformed, unreadable, or duplicate resume."""

    file_name: str
    error: str


class BatchSummary(BaseModel):
    """Batch-level execution statistics (Section 7 Minimum final outputs)."""

    total_resumes: int
    successfully_parsed: int
    duplicate_files: int = 0
    eligible: int
    rejected: int
    failed_unreadable: int
    github_enriched: int = 0
    evaluation_mode: str = "hybrid"


class ScreeningReport(BaseModel):
    """Complete structured output written to output/results.json."""

    batch_summary: BatchSummary
    ranked_candidates: List[CandidateResult] = Field(default_factory=list)
    rejected_candidates: List[CandidateResult] = Field(default_factory=list)
    failed_resumes: List[FailedResumeRecord] = Field(default_factory=list)
    results: List[CandidateResult] = Field(default_factory=list)
