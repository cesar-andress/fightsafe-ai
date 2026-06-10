#!/usr/bin/env python3
"""
Regenerate print-ready figures and LaTeX table snippets from a local FightSafe run directory.

By default, writes to paper/figures/ and paper/tables/ using the run directory basename
(``runs/demo/`` -> ``demo_*``). Does **not** require committing run data; only the generated
assets under paper/ (when small) are versioned at your discretion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _add_src_to_path() -> None:
    p = str(_REPO_ROOT / "src")
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    _add_src_to_path()
    from fightsafe_ai.reports.paper_assets import (
        compute_paper_run_metrics,
        write_artifacts_tex,
        write_events_tex,
        write_summary_tex,
    )
    from fightsafe_ai.visualization.plots import (
        plot_pose_coverage,
        plot_risk_timeline,
    )

    ap = argparse.ArgumentParser(
        description="Build paper/figures and paper/tables from a run directory (risk, pose, events, …).",
    )
    ap.add_argument(
        "--run",
        type=Path,
        default=Path("runs/demo"),
        help="Pipeline run root (e.g. runs/demo).",
    )
    ap.add_argument(
        "--paper",
        type=Path,
        default=Path("paper"),
        help="Paper directory containing figures/ and tables/ (default: paper).",
    )
    ap.add_argument(
        "--name",
        type=str,
        default=None,
        help="File name prefix (default: basename of --run, e.g. demo).",
    )
    args = ap.parse_args()
    run = (_REPO_ROOT / args.run).resolve() if not args.run.is_absolute() else args.run.resolve()
    paper = (
        (_REPO_ROOT / args.paper).resolve()
        if not args.paper.is_absolute()
        else args.paper.resolve()
    )
    name = (args.name or run.name or "run").strip() or "run"

    figs = paper / "figures"
    tabs = paper / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    plot_risk_timeline(run, output_path=figs / f"{name}_risk_timeline.png")
    plot_pose_coverage(run, output_path=figs / f"{name}_pose_coverage.png")

    m = compute_paper_run_metrics(
        run, run_path_display=str(args.run).replace("\\", "/").rstrip("/") or str(run)
    )
    write_summary_tex(tabs / f"{name}_summary.tex", tag=name, m=m)
    write_artifacts_tex(tabs / f"{name}_artifacts.tex", tag=name, m=m)
    write_events_tex(tabs / f"{name}_events.tex", tag=name, run_dir=run)

    print("Wrote:")
    for p in sorted(figs.glob(f"{name}_*.png")) + sorted(tabs.glob(f"{name}_*.tex")):
        rel = p.relative_to(_REPO_ROOT)
        print(f"  {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
