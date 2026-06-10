"""
Default output paths and batched report generation for a run directory.
"""

from __future__ import annotations

from pathlib import Path

from fightsafe_ai.reports.html import generate_html_report
from fightsafe_ai.reports.markdown import generate_markdown_report
from fightsafe_ai.reports.summary import generate_summary_json


def write_all_default_reports(run_dir: Path) -> tuple[Path, Path, Path]:
    """
    Write ``report.md``, ``report.html``, and ``summary.json`` under *run_dir*.

    The caller (CLI) is responsible for prerequisite checks.
    """
    r = run_dir.expanduser().resolve()
    p_md = generate_markdown_report(r, r / "report.md")
    p_html = generate_html_report(r, r / "report.html")
    p_json = generate_summary_json(r, r / "summary.json")
    return p_md, p_html, p_json
