"""End-to-end resume screening and ranking pipeline orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from src.config import ScreeningConfig
from src.eligibility import evaluate_eligibility
from src.github_enricher import GitHubEnricher
from src.llm_adapter import LLMProviderAdapter
from src.models import (
    BatchSummary,
    CandidateResult,
    EligibilityResult,
    FailedResumeRecord,
    GitHubEnrichment,
    LLMResumeEvaluation,
    ParsedResume,
    ScreeningReport,
)
from src.parser import SUPPORTED_EXTENSIONS, parse_resume_file
from src.reporter import write_csv_output, write_html_report, write_json_outputs
from src.scorer import build_candidate_result


class ResumeScreeningPipeline:
    """Orchestrates ingestion, deduplication, hard filtering, GitHub enrichment, scoring, and ranking."""

    def __init__(self, config: Optional[ScreeningConfig] = None) -> None:
        self.config = config or ScreeningConfig()
        self.github_enricher = GitHubEnricher(self.config)
        self.llm_adapter = LLMProviderAdapter(self.config)

    def discover_resume_files(self) -> List[Path]:
        """Find all supported resume files (.pdf, .docx, .txt) in input_dir."""
        input_dir = self.config.input_dir
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        files = [
            p
            for p in sorted(input_dir.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return files

    async def run_async(
        self,
        write_outputs: bool = True,
        csv_path: Optional[Path] = None,
        html_path: Optional[Path] = None,
    ) -> ScreeningReport:
        """Run the complete screening pipeline asynchronously with bounded concurrency."""
        files = self.discover_resume_files()
        total_files = len(files)

        # 1. Parse all resume files (CPU-bound, run in thread pool with bounded concurrency)
        parse_sem = asyncio.Semaphore(self.config.max_concurrency)

        async def _parse_one(fp: Path) -> ParsedResume:
            async with parse_sem:
                return await asyncio.to_thread(parse_resume_file, fp)

        parsed_list: List[ParsedResume] = await asyncio.gather(
            *[_parse_one(fp) for fp in files]
        )

        # 2. Deduplicate by content hash & separate failed/unreadable files
        seen_hashes: Dict[str, str] = {}
        valid_resumes: List[ParsedResume] = []
        failed_records: List[FailedResumeRecord] = []
        duplicate_count = 0

        for parsed in parsed_list:
            if parsed.parse_error:
                failed_records.append(
                    FailedResumeRecord(
                        file_name=parsed.file_name,
                        error=parsed.parse_error,
                    )
                )
                continue

            if parsed.content_hash and parsed.content_hash in seen_hashes:
                parsed.is_duplicate = True
                parsed.duplicate_of = seen_hashes[parsed.content_hash]
                duplicate_count += 1
                continue

            if parsed.content_hash:
                seen_hashes[parsed.content_hash] = parsed.file_name
            valid_resumes.append(parsed)

        # 3. Apply Hard Eligibility Filter (Section 3)
        eligibility_map: Dict[str, EligibilityResult] = {}
        eligible_resumes: List[ParsedResume] = []
        rejected_resumes: List[ParsedResume] = []

        for r in valid_resumes:
            elig = evaluate_eligibility(r)
            eligibility_map[r.file_name] = elig
            if elig.eligible:
                eligible_resumes.append(r)
            else:
                rejected_resumes.append(r)

        # 4. Enrich Eligible Candidates with Public GitHub Activity (Section 5)
        github_map = await self.github_enricher.enrich_candidates_batch(eligible_resumes)

        # 5. Optional LLM Evaluation for Eligible Candidates (Section 6)
        llm_map: Dict[str, Optional[LLMResumeEvaluation]] = {}
        active_provider = self.llm_adapter.active_provider
        if active_provider:
            llm_sem = asyncio.Semaphore(self.config.max_concurrency)

            async def _eval_llm(r: ParsedResume) -> None:
                async with llm_sem:
                    res = await asyncio.to_thread(self.llm_adapter.evaluate_resume, r)
                    llm_map[r.file_name] = res

            await asyncio.gather(*[_eval_llm(r) for r in eligible_resumes])
            self.llm_adapter.save_cache()

        # 6. Score Eligible Candidates & Build Results
        eligible_results: List[CandidateResult] = []
        for r in eligible_resumes:
            gh = github_map.get(
                r.file_name,
                GitHubEnrichment(status="missing", summary="No GitHub profile found."),
            )
            llm_eval = llm_map.get(r.file_name)
            cand_res = build_candidate_result(
                resume=r,
                eligibility=eligibility_map[r.file_name],
                github=gh,
                llm_eval=llm_eval,
            )
            eligible_results.append(cand_res)

        # Sort highest score first; tie-break by AI depth, Python backend depth, then name
        eligible_results.sort(
            key=lambda c: (
                c.total_score,
                c.score_breakdown.ai_project_depth if c.score_breakdown else 0,
                c.score_breakdown.python_backend if c.score_breakdown else 0,
                c.score_breakdown.github if c.score_breakdown else 0,
                c.candidate_name,
            ),
            reverse=True,
        )
        for idx, cand in enumerate(eligible_results, start=1):
            cand.rank = idx

        # 7. Build Rejected Candidate Records
        rejected_results: List[CandidateResult] = []
        for r in rejected_resumes:
            gh_placeholder = GitHubEnrichment(
                username=r.github_username,
                profile_url=r.github_url,
                status="skipped",
                score=0,
                summary=(
                    f"GitHub profile (@{r.github_username}) detected; skipped API enrichment because candidate failed hard eligibility filter."
                    if r.github_username
                    else "No public GitHub profile found on resume."
                ),
            )
            rej_res = build_candidate_result(
                resume=r,
                eligibility=eligibility_map[r.file_name],
                github=gh_placeholder,
                llm_eval=None,
            )
            rejected_results.append(rej_res)

        gh_enriched_count = sum(
            1 for g in github_map.values() if g.status == "enriched"
        )
        eval_mode = (
            f"hybrid ({active_provider} + deterministic)"
            if active_provider and any(v is not None for v in llm_map.values())
            else "deterministic (evidence-backed semantic rules)"
        )

        batch_summary = BatchSummary(
            total_resumes=total_files,
            successfully_parsed=len(valid_resumes) + duplicate_count,
            duplicate_files=duplicate_count,
            eligible=len(eligible_results),
            rejected=len(rejected_results),
            failed_unreadable=len(failed_records),
            github_enriched=gh_enriched_count,
            evaluation_mode=eval_mode,
        )

        report = ScreeningReport(
            batch_summary=batch_summary,
            ranked_candidates=eligible_results,
            rejected_candidates=rejected_results,
            failed_resumes=failed_records,
            results=eligible_results + rejected_results,
        )

        if write_outputs:
            out_path = self.config.output_path
            write_json_outputs(report, out_path)
            write_csv_output(report, csv_path or out_path.with_suffix(".csv"))
            write_html_report(report, html_path or (out_path.parent / "report.html"))

        return report

    def run(
        self,
        write_outputs: bool = True,
        csv_path: Optional[Path] = None,
        html_path: Optional[Path] = None,
    ) -> ScreeningReport:
        """Synchronous wrapper around `run_async`."""
        return asyncio.run(
            self.run_async(
                write_outputs=write_outputs,
                csv_path=csv_path,
                html_path=html_path,
            )
        )
