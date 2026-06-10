#!/usr/bin/env python3
"""Verify that reproduced metrics match reference snapshots shipped in data/repro/."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _approx_equal(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


def verify_sinica(results_csv: Path, reference_csv: Path) -> list[str]:
    errors: list[str] = []
    if not results_csv.is_file():
        return [f"Missing reproduced CSV: {results_csv}"]
    if not reference_csv.is_file():
        return [f"Missing reference CSV: {reference_csv}"]

    repro = {r["scope"]: r for r in _read_csv_rows(results_csv) if r.get("scope") == "micro"}
    ref = {r["scope"]: r for r in _read_csv_rows(reference_csv) if r.get("scope") == "micro"}
    if "micro" not in repro or "micro" not in ref:
        return ["micro row missing in sinica tapko_results.csv"]

    r, g = repro["micro"], ref["micro"]
    for key in ("tp", "fp", "fn"):
        if r.get(key) != g.get(key):
            errors.append(f"sinica {key}: got {r.get(key)} expected {g.get(key)}")
    for key in ("precision", "recall", "f1", "false_positives_per_minute"):
        if not _approx_equal(float(r[key]), float(g[key]), tol=1e-4):
            errors.append(f"sinica {key}: got {r[key]} expected {g[key]}")
    return errors


def verify_fusion_ablation(ablation_csv: Path) -> list[str]:
    errors: list[str] = []
    if not ablation_csv.is_file():
        return [f"Missing ablation CSV: {ablation_csv}"]
    rows = _read_csv_rows(ablation_csv)
    if len(rows) < 18:
        errors.append(f"fusion ablation CSV has only {len(rows)} rows (expected >= 18)")
    modes = {r.get("ablation_mode") for r in rows}
    for required in (
        "full_fusion",
        "full_fusion_without_interactions",
        "full_fusion_with_limb_anomaly_disabled",
    ):
        if required not in modes:
            errors.append(f"fusion ablation missing mode: {required}")
    return errors


def verify_sports_stats(stats_json: Path) -> list[str]:
    errors: list[str] = []
    if not stats_json.is_file():
        return [f"Missing dataset statistics: {stats_json}"]
    data = json.loads(stats_json.read_text(encoding="utf-8"))
    if int(data.get("videos", 0)) < 1:
        errors.append("sports: expected at least one video in dataset_statistics.json")
    return errors


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    repro_root = root / "data/repro"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper", choices=("fusion", "sinica", "sports", "all"), default="all")
    args = p.parse_args(argv)

    all_errors: list[str] = []

    if args.paper in ("fusion", "all"):
        all_errors.extend(
            verify_fusion_ablation(root / "runs/case_studies/ablation_summary/ablation_all_runs.csv")
        )

    if args.paper in ("sinica", "all"):
        repro_csv = root / "outputs/tapko/jedi_submissions_eval/tapko_results.csv"
        ref_csv = repro_root / "sinica2026/reference/tapko_results.csv"
        if not repro_csv.is_file():
            # Reference-only check when video not re-run
            if ref_csv.is_file():
                print("sinica: using reference snapshot only (no fresh eval outputs).")
            else:
                all_errors.append("sinica: no reproduced or reference tapko_results.csv")
        else:
            all_errors.extend(verify_sinica(repro_csv, ref_csv))

    if args.paper in ("sports", "all"):
        all_errors.extend(verify_sports_stats(root / "data/FightSafeBench/dataset_statistics.json"))

    if all_errors:
        print("VERIFICATION FAILED:")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print("All verification checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
