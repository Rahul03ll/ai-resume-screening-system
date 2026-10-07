"""Multi-format output generator: JSON, Shortlist JSON, CSV, HTML report, and CLI terminal table."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Optional

from src.models import ScreeningReport


def write_json_outputs(
    report: ScreeningReport,
    output_path: Path,
    shortlist_path: Optional[Path] = None,
) -> None:
    """
    Write the comprehensive `ScreeningReport` to `output_path` (e.g. output/results.json)
    and also write the top-level Section 7 array format to `output/shortlist.json`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if shortlist_path is None:
        shortlist_path = output_path.parent / "shortlist.json"

    shortlist_records = [
        {
            "rank": c.rank,
            "candidate_name": c.candidate_name,
            "eligible": c.eligible,
            "total_score": c.total_score,
            "score_breakdown": c.score_breakdown.model_dump() if c.score_breakdown else None,
            "matched_skills": c.matched_skills,
            "project_summary": c.project_summary,
            "github_summary": c.github_summary,
            "strengths": c.strengths,
            "concerns": c.concerns,
        }
        for c in report.ranked_candidates
    ]
    shortlist_path.write_text(
        json.dumps(shortlist_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv_output(report: ScreeningReport, csv_path: Path) -> None:
    """Write machine-readable CSV of all candidates (ranked eligible first, then rejected)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "file_name",
        "candidate_name",
        "email",
        "eligible",
        "total_score",
        "ai_project_depth",
        "python_backend",
        "cloud_fullstack",
        "github",
        "engineering_depth",
        "penalty_applied",
        "matched_skills",
        "rejection_reasons",
        "github_url",
        "github_summary",
        "project_summary",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in report.results:
            sb = c.score_breakdown
            writer.writerow(
                {
                    "rank": c.rank if c.rank is not None else "",
                    "file_name": c.file_name,
                    "candidate_name": c.candidate_name,
                    "email": c.email or "",
                    "eligible": c.eligible,
                    "total_score": c.total_score,
                    "ai_project_depth": sb.ai_project_depth if sb else 0,
                    "python_backend": sb.python_backend if sb else 0,
                    "cloud_fullstack": sb.cloud_fullstack if sb else 0,
                    "github": sb.github if sb else 0,
                    "engineering_depth": sb.engineering_depth if sb else 0,
                    "penalty_applied": c.penalty_applied,
                    "matched_skills": ", ".join(c.matched_skills),
                    "rejection_reasons": "; ".join(c.rejection_reasons),
                    "github_url": c.github_url or "",
                    "github_summary": c.github_summary,
                    "project_summary": c.project_summary,
                }
            )


def write_html_report(report: ScreeningReport, html_path: Path) -> None:
    """Generate a clean self-contained HTML report showing top candidates, score breakdowns, and rejected profiles."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    bs = report.batch_summary

    eligible_rows = []
    for c in report.ranked_candidates:
        sb = c.score_breakdown
        ai = sb.ai_project_depth if sb else 0
        py = sb.python_backend if sb else 0
        cl = sb.cloud_fullstack if sb else 0
        gh = sb.github if sb else 0
        eng = sb.engineering_depth if sb else 0
        gh_link = (
            f'<a href="{html.escape(c.github_url)}" target="_blank">{html.escape(c.github_url.split("/")[-1])}</a>'
            if c.github_url
            else "—"
        )
        pen_badge = (
            f'<span class="badge badge-warn">-{c.penalty_applied} wrapper/depth penalty</span>'
            if c.penalty_applied > 0
            else ""
        )
        eligible_rows.append(
            f"""
            <tr>
                <td class="rank">#{c.rank}</td>
                <td>
                    <strong>{html.escape(c.candidate_name)}</strong><br>
                    <small>{html.escape(c.file_name)} | {gh_link}</small><br>
                    {pen_badge}
                </td>
                <td class="score">{c.total_score}<small>/100</small></td>
                <td>
                    <div class="breakdown">
                        <span>AI/RAG: <b>{ai}</b>/40</span>
                        <span>Py/BE: <b>{py}</b>/30</span>
                        <span>Cloud/FS: <b>{cl}</b>/15</span>
                        <span>GitHub: <b>{gh}</b>/10</span>
                        <span>Eng: <b>{eng}</b>/5</span>
                    </div>
                </td>
                <td><small>{html.escape(", ".join(c.matched_skills[:10]))}</small></td>
                <td>
                    <div class="summary">{html.escape(c.project_summary)}</div>
                    <div class="meta"><b>Strengths:</b> {html.escape("; ".join(c.strengths))}</div>
                    <div class="meta"><b>Concerns:</b> {html.escape("; ".join(c.concerns))}</div>
                    <div class="meta"><b>GitHub:</b> {html.escape(c.github_summary)}</div>
                </td>
            </tr>
            """
        )

    rejected_rows = []
    for c in report.rejected_candidates:
        reasons_html = ", ".join(html.escape(r) for r in c.rejection_reasons)
        rejected_rows.append(
            f"""
            <tr>
                <td>{html.escape(c.file_name)}</td>
                <td><strong>{html.escape(c.candidate_name)}</strong></td>
                <td><span class="badge badge-reject">{reasons_html}</span></td>
                <td><small>{html.escape(", ".join(c.matched_skills))}</small></td>
            </tr>
            """
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Resume Screening & Ranking Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
h1, h2 {{ color: #1e293b; }}
.stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
.stat-card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 20px; min-width: 140px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
.stat-card .num {{ font-size: 24px; font-weight: 700; color: #2563eb; }}
.stat-card .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 32px; }}
th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; vertical-align: top; font-size: 14px; }}
th {{ background: #f1f5f9; font-weight: 600; color: #334155; }}
.rank {{ font-weight: 700; color: #1d4ed8; font-size: 16px; width: 48px; }}
.score {{ font-weight: 800; font-size: 18px; color: #059669; width: 80px; }}
.breakdown {{ display: flex; flex-direction: column; gap: 2px; font-size: 12px; min-width: 125px; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
.badge-warn {{ background: #fef3c7; color: #92400e; }}
.badge-reject {{ background: #fee2e2; color: #991b1b; }}
.summary {{ margin-bottom: 6px; color: #1e293b; }}
.meta {{ font-size: 12px; color: #475569; margin-top: 2px; }}
</style>
</head>
<body>
<h1>AI Resume Screening & Ranking Report</h1>
<div class="stats">
  <div class="stat-card"><div class="num">{bs.total_resumes}</div><div class="label">Total Resumes</div></div>
  <div class="stat-card"><div class="num">{bs.successfully_parsed}</div><div class="label">Parsed</div></div>
  <div class="stat-card"><div class="num">{bs.eligible}</div><div class="label">Eligible</div></div>
  <div class="stat-card"><div class="num">{bs.rejected}</div><div class="label">Rejected</div></div>
  <div class="stat-card"><div class="num">{bs.failed_unreadable}</div><div class="label">Failed / Unreadable</div></div>
  <div class="stat-card"><div class="num">{bs.github_enriched}</div><div class="label">GitHub Enriched</div></div>
</div>
<h2>Ranked Eligible Candidates ({bs.eligible})</h2>
<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Candidate</th>
      <th>Total</th>
      <th>Breakdown</th>
      <th>Matched Skills</th>
      <th>Evidence Summary, Strengths & Concerns</th>
    </tr>
  </thead>
  <tbody>
    {"".join(eligible_rows)}
  </tbody>
</table>
<h2>Rejected Candidates ({bs.rejected})</h2>
<table>
  <thead>
    <tr>
      <th>File</th>
      <th>Candidate</th>
      <th>Rejection Reasons</th>
      <th>Matched Skills</th>
    </tr>
  </thead>
  <tbody>
    {"".join(rejected_rows)}
  </tbody>
</table>
</body>
</html>
"""
    html_path.write_text(doc, encoding="utf-8")


def format_terminal_summary(report: ScreeningReport, top_n: int = 15) -> str:
    """Format a readable terminal summary of batch stats, top ranked candidates, and rejected candidates."""
    bs = report.batch_summary
    lines = [
        "=" * 98,
        " AI RESUME SCREENING & RANKING SYSTEM — BATCH SUMMARY",
        "=" * 98,
        f" Total Resumes      : {bs.total_resumes}",
        f" Successfully Parsed: {bs.successfully_parsed} (Duplicates skipped: {bs.duplicate_files})",
        f" Eligible Candidates: {bs.eligible}",
        f" Rejected Candidates: {bs.rejected}",
        f" Failed / Unreadable: {bs.failed_unreadable}",
        f" GitHub Enriched    : {bs.github_enriched}",
        f" Evaluation Mode    : {bs.evaluation_mode}",
        "-" * 98,
        f" TOP {min(top_n, len(report.ranked_candidates))} RANKED ELIGIBLE CANDIDATES",
        "-" * 98,
        f" {'Rank':<5} {'Candidate Name':<28} {'File':<17} {'Total':<6} {'AI(40)':<7} {'Py(30)':<7} {'Cl(15)':<7} {'GH(10)':<7} {'Eng(5)':<6}",
        "-" * 98,
    ]
    for c in report.ranked_candidates[:top_n]:
        sb = c.score_breakdown
        ai = sb.ai_project_depth if sb else 0
        py = sb.python_backend if sb else 0
        cl = sb.cloud_fullstack if sb else 0
        gh = sb.github if sb else 0
        eng = sb.engineering_depth if sb else 0
        lines.append(
            f" #{c.rank:<4} {c.candidate_name[:27]:<28} {c.file_name:<17} {c.total_score:<6} {ai:<7} {py:<7} {cl:<7} {gh:<7} {eng:<6}"
        )

    lines.extend(
        [
            "-" * 98,
            f" REJECTED CANDIDATES ({len(report.rejected_candidates)})",
            "-" * 98,
        ]
    )
    for c in report.rejected_candidates:
        reasons = "; ".join(c.rejection_reasons)
        lines.append(f" [REJECT] {c.file_name:<17} | {c.candidate_name[:26]:<26} | {reasons}")

    if report.failed_resumes:
        lines.extend(["-" * 98, f" FAILED / UNREADABLE RESUMES ({len(report.failed_resumes)})", "-" * 98])
        for f in report.failed_resumes:
            lines.append(f" [ERROR]  {f.file_name:<17} | {f.error}")

    lines.append("=" * 98)
    return "\n".join(lines)
