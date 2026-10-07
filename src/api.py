"""FastAPI interface for the AI Resume Screening & Ranking System (Section 8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import ScreeningConfig
from src.models import ScreeningReport
from src.pipeline import ResumeScreeningPipeline


app = FastAPI(
    title="AI Resume Screening & Ranking API",
    description="Backend screening pipeline for SDE Intern (Python + AI/Agentic Systems) resumes.",
    version="1.0.0",
)

_LATEST_REPORT: Optional[ScreeningReport] = None
_DEFAULT_OUTPUT_PATH = Path("./output/results.json")


class ScreenRequest(BaseModel):
    """Request payload for POST /screen."""

    input_dir: str = Field(default="./resumes", description="Directory containing resume files")
    output_path: str = Field(default="./output/results.json", description="Output JSON path")
    enable_github: bool = Field(default=True, description="Whether to enrich with public GitHub API")
    enable_llm: bool = Field(default=True, description="Whether to use LLM evaluation if API key is set")
    max_concurrency: int = Field(default=5, ge=1, le=20)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "ai-resume-screener"}


@app.post("/screen", response_model=ScreeningReport)
async def screen_resumes(req: ScreenRequest = ScreenRequest()) -> ScreeningReport:
    """Trigger a screening run over `input_dir` and persist results to `output_path`."""
    global _LATEST_REPORT
    input_path = Path(req.input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Input directory not found: {req.input_dir}",
        )

    config = ScreeningConfig(
        input_dir=input_path,
        output_path=Path(req.output_path),
        enable_github=req.enable_github,
        enable_llm=req.enable_llm,
        max_concurrency=req.max_concurrency,
    )
    pipeline = ResumeScreeningPipeline(config=config)
    report = await pipeline.run_async(write_outputs=True)
    _LATEST_REPORT = report
    return report


@app.get("/results")
async def get_results() -> Dict[str, Any]:
    """Return the latest screening report from memory or `output/results.json`."""
    if _LATEST_REPORT is not None:
        return _LATEST_REPORT.model_dump()

    if _DEFAULT_OUTPUT_PATH.exists():
        try:
            return json.loads(_DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to read results file: {exc}"
            ) from exc

    raise HTTPException(
        status_code=404,
        detail="No screening results found yet. Run POST /screen or `python main.py` first.",
    )


@app.get("/shortlist")
async def get_shortlist() -> List[Dict[str, Any]]:
    """Return the ranked eligible shortlist array matching Section 7 output format."""
    data = await get_results()
    return data.get("ranked_candidates", [])
