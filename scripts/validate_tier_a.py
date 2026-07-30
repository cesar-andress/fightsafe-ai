#!/usr/bin/env python3.12
"""Validate Tier A reproducibility from the repository root."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "validation"
VAL.mkdir(exist_ok=True)
CANON = ROOT / "canonical_results" / "run_20260730_005150"


def resolve_paper1() -> Path | None:
    import os

    env = os.environ.get("FIGHTSAFE_PAPER1_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_dir() else None
    sibling = (ROOT.parent / "paper1").resolve()
    return sibling if sibling.is_dir() else None


PAPER = resolve_paper1()


def log(name: str, text: str) -> None:
    (VAL / name).write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    results: dict[str, str] = {}
    failed = False

    results["python"] = sys.version.split()[0]
    results["python_check"] = "PASS" if sys.version.startswith("3.12") else "WARN: expected 3.12"

    cp = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_checksums.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    log("checksums.txt", cp.stdout + cp.stderr)
    results["checksums"] = "PASS" if cp.returncode == 0 else "FAIL"
    if cp.returncode != 0:
        failed = True

    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    imp = subprocess.run(
        [
            sys.executable,
            "-c",
            "from fightsafe_ai.evaluation.aggregation_schemes import SCHEME_ORDER;"
            "assert set(['equal','weighted','max']).issubset(SCHEME_ORDER); print('OK', SCHEME_ORDER)",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    log("import_aggregation.txt", imp.stdout + imp.stderr)
    results["import_aggregation"] = "PASS" if imp.returncode == 0 else "FAIL"
    if imp.returncode != 0:
        failed = True

    agg = ROOT / "src/fightsafe_ai/evaluation/aggregation_schemes.py"
    expect = "4d19f0f69238c9afdf365b6baeae26986a2d7086fb2b24170f5b2ce5b6209281"
    got = sha256(agg)
    results["aggregation_hash"] = "PASS" if got == expect else f"FAIL {got}"
    if got != expect:
        failed = True

    import pandas as pd

    summary = pd.read_csv(CANON / "experiment_summary.csv")
    paired = pd.read_csv(CANON / "paired_comparisons.csv")
    dropout = pd.read_csv(CANON / "dropout_results.csv")
    numbers_path = CANON.parent / "analysis" / "numbers.json"
    if not numbers_path.is_file():
        results["numerical_integrity"] = f"FAIL missing {numbers_path.relative_to(ROOT)}"
        failed = True
        numbers = {}
    else:
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))

    def get_micro(method: str) -> float:
        row = summary[(summary.experiment == "aggregation") & (summary.method == method)].iloc[0]
        return float(row.micro_f1)

    if numbers:
        checks = {
            "equal_micro_f1": abs(get_micro("equal") - numbers["micro_f1_equal"]) < 1e-12,
            "weighted_micro_f1": abs(get_micro("weighted") - numbers["micro_f1_weighted"]) < 1e-12,
            "max_micro_f1": abs(get_micro("max") - numbers["micro_f1_max"]) < 1e-12,
            "approx_equal_0.502": abs(get_micro("equal") - 0.502) < 5e-4,
            "approx_weighted_0.502": abs(get_micro("weighted") - 0.502) < 5e-4,
            "approx_max_0.178": abs(get_micro("max") - 0.178) < 5e-4,
        }
        mm = dropout[dropout.video_id == "__MICRO_MACRO__"]
        ex = float(mm[(mm.p == 0.5) & (mm["mode"] == "explicit_alpha0")].micro_f1.mean())
        nv = float(mm[(mm.p == 0.5) & (mm["mode"] == "naive_zero")].micro_f1.mean())
        checks["dropout_explicit_p05"] = abs(ex - numbers["dropout_p05_explicit"]) < 1e-12
        checks["dropout_naive_p05"] = abs(nv - numbers["dropout_p05_naive"]) < 1e-12
        checks["approx_explicit_0.509"] = abs(ex - 0.509) < 5e-4
        checks["approx_naive_0.471"] = abs(nv - 0.471) < 5e-4
        p_int = float(
            paired.loc[
                paired.comparison == "weighted_intON_minus_intOFF", "permutation_pvalue_twosided"
            ].iloc[0]
        )
        checks["paired_int_p"] = abs(p_int - numbers["paired_int_p"]) < 1e-12
        log("numerical_checks.json", json.dumps({k: bool(v) for k, v in checks.items()}, indent=2))
        if not all(checks.values()):
            results["numerical_integrity"] = "FAIL"
            failed = True
        else:
            results["numerical_integrity"] = "PASS"

    if PAPER is None:
        results["generate_assets"] = "SKIP (paper1 not found; set FIGHTSAFE_PAPER1_DIR)"
        results["regenerated_numbers"] = "SKIP"
        results["regenerated_tables"] = "SKIP"
        results["regenerated_figures"] = "SKIP"
        results["manuscript_build"] = "SKIP (paper1 not found)"
        results["supplementary_build"] = "SKIP"
    else:
        gen = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_eaai_assets.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        log("generate_assets.txt", gen.stdout + gen.stderr)
        results["generate_assets"] = "PASS" if gen.returncode == 0 else "FAIL"
        if gen.returncode != 0:
            failed = True
        else:
            after_nums = json.loads((PAPER / "tables/numbers.json").read_text(encoding="utf-8"))
            # Keep frozen software-side copy in sync for GitHub-only checks
            soft_nums = CANON.parent / "analysis" / "numbers.json"
            soft_nums.write_text(json.dumps(after_nums, indent=2) + "\n", encoding="utf-8")
            num_ok = all(
                abs(float(after_nums[k]) - float(numbers[k])) < 1e-12
                for k in numbers
                if k != "canon_relative"
            )
            results["regenerated_numbers"] = "PASS" if num_ok else "FAIL"
            if not num_ok:
                failed = True
            agg_tex = (PAPER / "tables/tab_aggregation.tex").read_text(encoding="utf-8")
            for token in ("0.502", "0.178"):
                if token not in agg_tex:
                    results["regenerated_tables"] = f"FAIL missing {token}"
                    failed = True
                    break
            else:
                results["regenerated_tables"] = "PASS"
            fig_ok = all(
                (PAPER / "figures" / n).is_file()
                for n in [
                    "fig_aggregation.pdf",
                    "fig_architecture.pdf",
                    "fig_dropout.pdf",
                    "fig_failures.pdf",
                    "fig_interactions.pdf",
                    "fig_per_video.pdf",
                    "fig_video_contribution.pdf",
                ]
            )
            results["regenerated_figures"] = "PASS" if fig_ok else "FAIL"
            if not fig_ok:
                failed = True

        if shutil.which("latexmk"):
            ms = subprocess.run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"],
                cwd=PAPER,
                capture_output=True,
                text=True,
            )
            log("manuscript_build.txt", ms.stdout[-4000:] + "\n" + ms.stderr[-2000:])
            results["manuscript_build"] = "PASS" if ms.returncode == 0 else "FAIL"
            if ms.returncode != 0:
                failed = True
            results["supplementary_build"] = results["manuscript_build"]
        else:
            results["manuscript_build"] = "SKIP (latexmk not found)"
            results["supplementary_build"] = "SKIP"

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_aggregation_schemes.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    log("pytest_aggregation.txt", test.stdout + test.stderr)
    results["unit_tests"] = "PASS" if test.returncode == 0 else "FAIL"
    if test.returncode != 0:
        failed = True

    abs_hits = []
    skip_suffixes = {
        ".pdf",
        ".png",
        ".xlsx",
        ".ods",
        ".parquet",
        ".pkl",
        ".pyc",
        ".csv",
        ".fls",
        ".fdb_latexmk",
    }
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(
            rel.startswith(pref)
            for pref in (
                "validation/",
                "optional_tier_b/",
                "data/repro/",
                "outputs/",
                "runs/",
                ".git/",
            )
        ):
            continue
        if "__pycache__" in rel or ".venv" in rel:
            continue
        if Path(rel).suffix.lower() in skip_suffixes:
            continue
        if rel in {
            "scripts/validate_tier_a.py",
            "environment/pip_freeze.txt",
            "environment/pip_freeze.ORIGINAL_NOTE.md",
        }:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if (
            re.search(r"(?<!REDACTED_LOCAL_EDITABLE: -e )(?<!<)/home/cesar/[A-Za-z0-9_./-]+", text)
            and "/home/cesar" in text
            and "REDACTED" not in text
            and "<LOCAL_HOME" not in text
        ):
            abs_hits.append(rel)
    log("absolute_path_hits.txt", "\n".join(abs_hits) if abs_hits else "NONE\n")
    results["absolute_paths"] = "PASS" if not abs_hits else f"FAIL {abs_hits[:10]}"
    if abs_hits:
        failed = True

    meta_hits = []
    for rel in ("CITATION.cff", ".zenodo.json"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        if re.search(r"\bISWA\b|Information Fusion", text, re.I):
            meta_hits.append(rel)
    results["metadata"] = "PASS" if not meta_hits else f"FAIL {meta_hits}"
    if meta_hits:
        failed = True

    results["license"] = "PASS" if (ROOT / "LICENSE").is_file() else "FAIL"
    if results["license"] != "PASS":
        failed = True

    mandatory = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        ".zenodo.json",
        "NOTICE_DATA.md",
        "scripts/generate_eaai_assets.py",
        "scripts/verify_checksums.py",
        "src/fightsafe_ai/evaluation/aggregation_schemes.py",
        "configs/risk_fusion.yaml",
        "configs/risk_rules.yaml",
        "canonical_results/run_20260730_005150/experiment_summary.csv",
        "canonical_results/analysis/numbers.json",
        "checksums/SHA256SUMS",
    ]
    missing = [m for m in mandatory if not (ROOT / m).is_file()]
    log("missing_files.txt", "\n".join(missing) if missing else "NONE\n")
    results["missing_files"] = "PASS" if not missing else f"FAIL {missing}"
    if missing:
        failed = True

    fc = list((ROOT / "optional_tier_b/inputs/features_cache").glob("*.pkl"))
    results["features_cache_absent"] = "PASS" if not fc else "FAIL"
    if fc:
        failed = True

    results["overall"] = "FAIL" if failed else "PASS"
    log("summary.json", json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
