#!/usr/bin/env python3
r"""
Regenerate paper figures and LaTeX table fragments from local **batch** and **ablation** exports.

This is the **results pipeline** for ``paper/main.tex``:

- Optional: one **batch** run directory (e.g. ``runs/demo``) → risk / pose / event plots and
  per-run ``\texttt{.tex}`` fragments (when ``risk_scores.csv`` / ``events.json`` exist).
- **Ablation** (if ``ablation\_all\_runs.csv`` is present and complete) → ablation table, bar
  charts, and (when per-case ablation subfolders exist) the **ablation risk timeline** PNG
  copied from a case-study export.
- **Quantitative** table: requires aggregate CSV \emph{and} per-case ``risk\_series\_*.csv``
  files (see ``tools/generate\_quantitative\_observations\_tex.py``).
- If inputs are missing, writes **TODO** placeholders via ``fightsafe\_ai.reports.paper\_build``
  (no fabricated numbers).

Does **not** run pose pipelines or network downloads.

Examples::

    python scripts/generate_paper_assets.py
    python scripts/generate_paper_assets.py --run runs/demo --paper paper
    python scripts/generate_paper_assets.py --skip-run --ablation-csv runs/case_studies/ablation_summary/ablation_all_runs.csv

See also: ``tools/generate\_paper\_assets.py`` (legacy single-run helper),
``tools/generate\_ablation\_paper\_assets.py``, ``tools/generate\_quantitative\_observations\_tex.py``.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _add_paths() -> None:
    src = str(_REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_tool_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _copy_ablation_risk_timeline(ablation_summary: Path, paper_figures: Path) -> bool:
    """Copy first available per-case ``ablation_risk_timeline.png`` into paper figures."""
    mod = _load_tool_module(
        "gap_tools",
        _REPO_ROOT / "tools" / "generate_ablation_paper_assets.py",
    )
    order = list(mod.RUN_ORDER)
    paper_figures.mkdir(parents=True, exist_ok=True)
    dest = paper_figures / "ablation_risk_timeline.png"
    for run_id in order:
        src = ablation_summary / run_id / "ablation_risk_timeline.png"
        if src.is_file():
            shutil.copy2(src, dest)
            print(
                f"Copied ablation risk timeline: {src.relative_to(_REPO_ROOT)} → {dest.relative_to(_REPO_ROOT)}"
            )
            return True
    print(
        "No per-case ablation_risk_timeline.png found under",
        ablation_summary,
        "(run fightsafe risk-ablation-all to populate).",
    )
    return False


def _try_ablation_csv(csv_path: Path, paper_dir: Path) -> bool:
    mod = _load_tool_module(
        "gapa",
        _REPO_ROOT / "tools" / "generate_ablation_paper_assets.py",
    )
    if not csv_path.is_file():
        print(f"Ablation CSV not found: {csv_path}")
        return False
    try:
        mod.generate_ablation_paper_assets_from_csv(csv_path, paper_dir)
        print(f"Wrote ablation table + bar charts from {csv_path.relative_to(_REPO_ROOT)}")
        return True
    except (OSError, ValueError, KeyError) as e:
        print(f"Ablation asset generation failed ({e}); writing placeholder table.")
        return False


def _try_quantitative_observations_tex(
    ablation_summary: Path, paper_dir: Path, csv_path: Path
) -> bool:
    tool = _REPO_ROOT / "tools" / "generate_quantitative_observations_tex.py"
    if not tool.is_file():
        return False
    proc = subprocess.run(  # noqa: S603 — fixed repo tool path and argv
        [
            sys.executable,
            str(tool),
            "--csv",
            str(csv_path),
            "--base-dir",
            str(ablation_summary),
            "--paper",
            str(paper_dir),
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        print(proc.stdout.strip() or "Wrote quantitative_observations.tex")
        return True
    print("Quantitative observations script skipped:", (proc.stderr or proc.stdout or "").strip())
    return False


def _run_batch_run_assets(run: Path, paper: Path, *, name: str) -> None:
    _add_paths()
    from fightsafe_ai.reports.paper_assets import (
        compute_paper_run_metrics,
        write_artifacts_tex,
        write_events_tex,
        write_summary_tex,
    )
    from fightsafe_ai.visualization.plots import (
        plot_events_timeline,
        plot_pose_coverage,
        plot_risk_timeline,
    )

    run = run.expanduser().resolve()
    paper = paper.expanduser().resolve()
    if not run.is_dir():
        print(f"Skip batch run assets: not a directory {run}")
        return

    figs = paper / "figures"
    tabs = paper / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    plot_risk_timeline(run, output_path=figs / f"{name}_risk_timeline.png")
    plot_pose_coverage(run, output_path=figs / f"{name}_pose_coverage.png")
    plot_events_timeline(run, output_path=figs / f"{name}_events_timeline.png")

    # Stable names referenced from main.tex (Temporal Event Analysis figure).
    shutil.copy2(figs / f"{name}_events_timeline.png", figs / "events_timeline.png")
    print(f"Copied → {figs / 'events_timeline.png'}")

    try:
        display = str(run.relative_to(_REPO_ROOT))
    except ValueError:
        display = str(run)
    m = compute_paper_run_metrics(run, run_path_display=display.replace("\\", "/"))
    write_summary_tex(tabs / f"{name}_summary.tex", tag=name, m=m)
    write_artifacts_tex(tabs / f"{name}_artifacts.tex", tag=name, m=m)
    write_events_tex(tabs / f"{name}_events.tex", tag=name, run_dir=run)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", type=Path, default=Path("runs/demo"), help="Batch pipeline run root.")
    p.add_argument("--paper", type=Path, default=Path("paper"), help="Paper directory.")
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Prefix for batch outputs (default: basename of --run).",
    )
    p.add_argument(
        "--ablation-csv",
        type=Path,
        default=Path("runs/case_studies/ablation_summary/ablation_all_runs.csv"),
        help="Aggregate ablation CSV from fightsafe risk-ablation-all.",
    )
    p.add_argument(
        "--ablation-summary",
        type=Path,
        default=Path("runs/case_studies/ablation_summary"),
        help="Directory with per-case subfolders (risk_series_*.csv, ablation_risk_timeline.png).",
    )
    p.add_argument("--skip-run", action="store_true", help="Do not read batch --run directory.")
    p.add_argument("--skip-ablation", action="store_true", help="Skip ablation CSV / figures.")
    p.add_argument(
        "--skip-quantitative", action="store_true", help="Skip quantitative_observations.tex."
    )
    args = p.parse_args(argv)

    paper_dir = (
        (_REPO_ROOT / args.paper).resolve()
        if not args.paper.is_absolute()
        else args.paper.resolve()
    )
    name = (args.name or args.run.name or "run").strip() or "run"

    if not args.skip_run:
        run_dir = (
            (_REPO_ROOT / args.run).resolve() if not args.run.is_absolute() else args.run.resolve()
        )
        try:
            _run_batch_run_assets(run_dir, paper_dir, name=name)
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"Batch run asset generation skipped or partial: {e}")

    ablation_csv = (
        (_REPO_ROOT / args.ablation_csv).resolve()
        if not args.ablation_csv.is_absolute()
        else args.ablation_csv.resolve()
    )
    ablation_summary = (
        (_REPO_ROOT / args.ablation_summary).resolve()
        if not args.ablation_summary.is_absolute()
        else args.ablation_summary.resolve()
    )

    _add_paths()
    from fightsafe_ai.reports.paper_build import (
        write_ablation_table_placeholder,
        write_quantitative_table_placeholder,
    )

    ablation_tex = paper_dir / "tables" / "ablation_selected_results.tex"
    quantitative_tex = paper_dir / "tables" / "quantitative_observations.tex"

    if not args.skip_ablation:
        ok_ab = _try_ablation_csv(ablation_csv, paper_dir)
        if not ok_ab:
            if not ablation_tex.is_file():
                write_ablation_table_placeholder(paper_dir)
                print(f"Wrote placeholder {ablation_tex}")
            else:
                print(f"Kept existing {ablation_tex} (ablation regeneration failed).")
        else:
            _copy_ablation_risk_timeline(ablation_summary, paper_dir / "figures")

    if not args.skip_quantitative and not args.skip_ablation:
        ok_q = _try_quantitative_observations_tex(ablation_summary, paper_dir, ablation_csv)
        if not ok_q:
            if not quantitative_tex.is_file():
                write_quantitative_table_placeholder(paper_dir)
                print(f"Wrote placeholder {quantitative_tex}")
            else:
                print(f"Kept existing {quantitative_tex} (quantitative regeneration failed).")
    elif args.skip_quantitative:
        print("Skipped quantitative_observations.tex (--skip-quantitative).")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
