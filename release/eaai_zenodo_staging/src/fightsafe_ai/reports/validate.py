"""
Prerequisites for generating reports from a pipeline run directory.

The CLI defers to these helpers to keep :mod:`fightsafe_ai.cli` thin.
"""

from __future__ import annotations

from pathlib import Path


# Files the report generators need for a meaningful, consistent output.
REPORT_REQUIRED_RELATIVE: tuple[str, ...] = (
    "risk_scores.csv",
    "events.json",
)


def missing_report_artifacts(run_dir: Path) -> list[Path]:
    """
    Return paths to required files that are missing (empty list if the run is usable).

    If ``run_dir`` is not a directory, returns a single-element list with that path.
    """
    root = run_dir.expanduser().resolve()
    if not root.is_dir():
        return [root]
    missing: list[Path] = []
    for name in REPORT_REQUIRED_RELATIVE:
        p = root / name
        if not p.is_file():
            missing.append(p)
    return missing


def report_prereq_error_message(run_dir: Path, missing: list[Path]) -> str:
    """
    User-facing text when :func:`missing_report_artifacts` is non-empty
    (stderr + non-zero exit).
    """
    root = run_dir.expanduser().resolve()
    lines: list[str] = []
    if not root.is_dir():
        lines.append(f"Not a directory: {root}")
    else:
        lines.append("Missing required file(s) for report generation:")
        for p in missing:
            lines.append(f"  - {p}")
    run_for_help = str(root) if root.is_dir() else str(run_dir.expanduser())
    if not run_for_help.endswith("/"):
        run_for_help = run_for_help + "/"
    lines.append(
        f"Produce and validate a complete pipeline run, e.g.:\n  fightsafe qa --run {run_for_help}"
    )
    return "\n".join(lines) + "\n"
