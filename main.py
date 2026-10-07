"""CLI entrypoint for the AI Resume Screening & Ranking System (Section 8).

Usage:
    python main.py --input ./resumes --output ./output/results.json
    python main.py --serve --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import ScreeningConfig
from src.pipeline import ResumeScreeningPipeline
from src.reporter import format_terminal_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Resume Screening & Ranking System for SDE Intern (Python + AI/Agentic)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="./resumes",
        help="Path to input directory containing resumes (.pdf, .docx, .txt). Default: ./resumes",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output/results.json",
        help="Path to output JSON file. Default: ./output/results.json",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional custom path for CSV output (defaults to <output_stem>.csv).",
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Optional custom path for HTML report (defaults to output/report.html).",
    )
    parser.add_argument(
        "--no-github",
        action="store_true",
        help="Disable public GitHub API enrichment.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable external LLM API calls and use deterministic evidence scoring only.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading/writing disk cache for GitHub and LLM calls.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Maximum bounded async concurrency for parsing, GitHub, and LLM calls (defaults to MAX_CONCURRENCY env var or 5).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=35,
        help="Number of top ranked candidates to print in terminal summary (default: 35).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI server instead of running CLI batch mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for FastAPI server when --serve is passed (default: 8000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args(argv)

    if args.serve:
        import uvicorn

        uvicorn.run("src.api:app", host="0.0.0.0", port=args.port, reload=False)
        return 0

    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        return 1

    config = ScreeningConfig(
        input_dir=input_dir,
        output_path=output_path,
    )
    if args.concurrency is not None:
        config.max_concurrency = max(1, args.concurrency)
    if args.no_github:
        config.enable_github = False
    if args.no_llm:
        config.enable_llm = False
    if args.no_cache:
        config.use_cache = False

    pipeline = ResumeScreeningPipeline(config=config)
    report = pipeline.run(
        write_outputs=True,
        csv_path=Path(args.csv) if args.csv else None,
        html_path=Path(args.html) if args.html else None,
    )

    print(format_terminal_summary(report, top_n=args.top))
    print(f"\nOutputs written to:")
    print(f"  - JSON (Full Report): {output_path}")
    print(f"  - JSON (Shortlist)  : {output_path.parent / 'shortlist.json'}")
    print(f"  - CSV               : {Path(args.csv) if args.csv else output_path.with_suffix('.csv')}")
    print(f"  - HTML Report       : {Path(args.html) if args.html else output_path.parent / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
