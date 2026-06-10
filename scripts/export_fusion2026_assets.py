#!/usr/bin/env python3
"""Regenerate fusion2026 ablation figures/tables from bundled ablation exports."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    fusion_dir = root / "../fusion2026"
    ablation_csv = root / "runs/case_studies/ablation_summary/ablation_all_runs.csv"
    repro_dir = root / "outputs/repro/fusion2026"
    figures_data = fusion_dir / "figures/data"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fusion-dir", type=Path, default=fusion_dir)
    p.add_argument("--ablation-csv", type=Path, default=ablation_csv)
    p.add_argument("--output-dir", type=Path, default=repro_dir)
    p.add_argument(
        "--install-tables",
        action="store_true",
        help="Copy generated LaTeX tables into fusion-dir/tables/ (off by default).",
    )
    p.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip matplotlib figure regeneration.",
    )
    args = p.parse_args(argv)

    if not args.ablation_csv.is_file():
        print(f"ERROR: missing ablation CSV: {args.ablation_csv}", file=sys.stderr)
        return 1
    if not args.fusion_dir.joinpath("main.tex").is_file():
        print(f"ERROR: fusion manuscript not found: {args.fusion_dir}/main.tex", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Generate into repro staging area so manuscript tables are not overwritten unless requested.
    paper_staging = args.output_dir / "paper_staging"
    paper_staging.mkdir(parents=True, exist_ok=True)
    (paper_staging / "tables").mkdir(exist_ok=True)
    (paper_staging / "figures").mkdir(exist_ok=True)

    # Sync snapshot CSV into fusion figures/data for regenerate_figures.py
    figures_data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.ablation_csv, figures_data / "ablation_all_runs.csv")
    case_a = root / "runs/case_studies/ablation_summary/case_a_knockdown"
    for src_name, dest_name in (
        ("risk_series_full_fusion.csv", "risk_series_full_fusion.csv"),
        ("risk_series_full_fusion_without_interactions.csv", "risk_series_full_fusion_without_interactions.csv"),
        ("risk_series_full_fusion_with_limb_anomaly_disabled.csv", "risk_series_full_fusion_with_limb_anomaly_disabled.csv"),
    ):
        src = case_a / src_name
        if src.is_file():
            shutil.copy2(src, figures_data / dest_name)

    # Generate LaTeX tables via ablation tooling (avoids full package import chain)
    ablation_tool = root / "tools/generate_ablation_paper_assets.py"
    quant_tool = root / "tools/generate_quantitative_observations_tex.py"
    ablation_summary = root / "runs/case_studies/ablation_summary"

    _run(
        [
            sys.executable,
            str(ablation_tool),
            "--csv",
            str(args.ablation_csv),
            "--paper",
            str(paper_staging),
        ],
        cwd=root,
    )
    if quant_tool.is_file():
        _run(
            [
                sys.executable,
                str(quant_tool),
                "--csv",
                str(args.ablation_csv),
                "--base-dir",
                str(ablation_summary),
                "--paper",
                str(paper_staging),
            ],
            cwd=root,
        )

    # Copy generated tables to repro output (manuscript tables untouched unless --install-tables)
    for name in ("ablation_selected_results.tex", "quantitative_observations.tex"):
        src = paper_staging / "tables" / name
        if src.is_file():
            shutil.copy2(src, tables_dir / name)
            print(f"Copied table snapshot -> {tables_dir / name}")

    if args.install_tables:
        for name in ("ablation_selected_results.tex", "quantitative_observations.tex"):
            src = tables_dir / name
            if src.is_file():
                shutil.copy2(src, args.fusion_dir / "tables" / name)
                print(f"Installed -> {args.fusion_dir / 'tables' / name}")

    if not args.skip_figures:
        regen = args.fusion_dir / "scripts/regenerate_figures.py"
        if regen.is_file():
            _run([sys.executable, str(regen)], cwd=args.fusion_dir)
            for stem in (
                "ablation_risk_timeline",
                "ablation_high_critical_frames",
                "ablation_candidate_events",
                "events_timeline",
            ):
                for ext in ("pdf", "png"):
                    src = args.fusion_dir / "figures" / f"{stem}.{ext}"
                    if src.is_file():
                        shutil.copy2(src, figures_dir / src.name)
                        print(f"Copied figure -> {figures_dir / src.name}")
        else:
            print(f"WARNING: {regen} not found; skipping figure regeneration.")

    print(f"Fusion reproducibility assets under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
