"""
Summarize multiple BoxingVI batch evaluation directories (e.g. strike-percentile sweeps).

For each ``--input-dir``, reads ``boxingvi_results_all.csv`` (from
:func:`fightsafe_ai.evaluation.boxingvi_batch_eval._write_aggregate_table_csv``), infers
``strike_percentile`` from the folder name (``.../boxingvi_batch_p90`` → 90), and writes
``sweep_summary.{csv,tex,md}`` under ``--output-dir``.

CLI example::

    python -m fightsafe_ai.evaluation.summarize_sweeps \\
      --input-dirs outputs/evaluation/boxingvi_batch_p85 ... \\
      --output-dir outputs/evaluation/sweeps
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final


_STRIKE_PCT_PATTERN: Final[re.Pattern[str]] = re.compile(r"_p(\d+)(?:\b|/|$)", re.IGNORECASE)
_STRIKE_PCT_SUFFIX: Final[re.Pattern[str]] = re.compile(r"_p(\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SweepMetrics:
    """One row of pooled metrics for a single sweep / input directory."""

    strike_percentile: int
    input_dir: Path
    tp: int
    fp: int
    fn: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    mean_latency: float


def infer_strike_percentile(input_dir: Path) -> int | None:
    """
    Parse ``_p<nn>`` from the last path segment, e.g. ``boxingvi_batch_p85`` → 85.
    """
    name = Path(input_dir).resolve().name
    m = _STRIKE_PCT_SUFFIX.search(name)
    if m:
        return int(m.group(1))
    m2 = _STRIKE_PCT_PATTERN.search(name)
    if m2:
        return int(m2.group(1))
    return None


def _float_cell(s: str | None) -> float | None:
    t = (s or "").strip()
    if not t:
        return None
    return float(t)


def _int_cell(s: str | None) -> int | None:
    t = (s or "").strip()
    if not t:
        return None
    return int(t)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _compute_from_per_video_rows(rows: list[dict[str, str]]) -> SweepMetrics:
    """Recompute micro/macro from per-video rows (OK / CACHED with counts), same spirit as batch eval."""
    ok_rows: list[dict[str, str]] = []
    for r in rows:
        vid = (r.get("video_id") or "").strip()
        if vid in ("", "__micro__", "__macro__"):
            continue
        st = (r.get("status") or "").strip()
        if st not in {"OK", "CACHED"}:
            continue
        tp = _int_cell(r.get("TP"))
        if tp is None:
            continue
        fp = _int_cell(r.get("FP"))
        fn = _int_cell(r.get("FN"))
        if fp is None or fn is None:
            continue
        ok_rows.append(r)

    if not ok_rows:
        raise ValueError("No OK/CACHED rows with TP/FP/FN in CSV.")

    micro_tp = sum(_int_cell(r.get("TP")) or 0 for r in ok_rows)
    micro_fp = sum(_int_cell(r.get("FP")) or 0 for r in ok_rows)
    micro_fn = sum(_int_cell(r.get("FN")) or 0 for r in ok_rows)
    p_d = micro_tp + micro_fp
    r_d = micro_tp + micro_fn
    micro_p = float(micro_tp / p_d) if p_d > 0 else 0.0
    micro_r = float(micro_tp / r_d) if r_d > 0 else 0.0
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if micro_p + micro_r > 0 else 0.0

    precs = [_float_cell(r.get("precision")) for r in ok_rows]
    recs = [_float_cell(r.get("recall")) for r in ok_rows]
    f1s = [_float_cell(r.get("F1")) for r in ok_rows]
    lats = [_float_cell(r.get("mean_latency")) for r in ok_rows]

    def _mean(vals: list[float | None]) -> float:
        xs = [float(v) for v in vals if v is not None]
        return float(sum(xs) / len(xs)) if xs else 0.0

    macro_p = _mean(precs)
    macro_r = _mean(recs)
    macro_f1 = _mean(f1s)
    mean_lat = _mean(lats)

    return SweepMetrics(
        strike_percentile=0,
        input_dir=Path("."),
        tp=micro_tp,
        fp=micro_fp,
        fn=micro_fn,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        mean_latency=mean_lat,
    )


def parse_boxingvi_results_all_csv(path: Path, *, input_dir: Path) -> SweepMetrics:
    """Load metrics from ``boxingvi_results_all.csv`` (aggregate rows or derived from per-video)."""
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    pct = infer_strike_percentile(input_dir)
    if pct is None:
        raise ValueError(
            f"Cannot infer strike percentile from directory name: {input_dir.name!r} "
            f"(expected substring like _p85)."
        )

    micro_row = next((r for r in rows if (r.get("video_id") or "").strip() == "__micro__"), None)
    macro_row = next((r for r in rows if (r.get("video_id") or "").strip() == "__macro__"), None)

    if micro_row is not None and macro_row is not None:
        tp = _int_cell(micro_row.get("TP")) or 0
        fp = _int_cell(micro_row.get("FP")) or 0
        fn = _int_cell(micro_row.get("FN")) or 0
        return SweepMetrics(
            strike_percentile=pct,
            input_dir=input_dir.resolve(),
            tp=tp,
            fp=fp,
            fn=fn,
            micro_precision=_float_cell(micro_row.get("precision")) or 0.0,
            micro_recall=_float_cell(micro_row.get("recall")) or 0.0,
            micro_f1=_float_cell(micro_row.get("F1")) or 0.0,
            macro_precision=_float_cell(macro_row.get("precision")) or 0.0,
            macro_recall=_float_cell(macro_row.get("recall")) or 0.0,
            macro_f1=_float_cell(macro_row.get("F1")) or 0.0,
            mean_latency=_float_cell(macro_row.get("mean_latency")) or 0.0,
        )

    derived = _compute_from_per_video_rows(rows)
    return SweepMetrics(
        strike_percentile=pct,
        input_dir=input_dir.resolve(),
        tp=derived.tp,
        fp=derived.fp,
        fn=derived.fn,
        micro_precision=derived.micro_precision,
        micro_recall=derived.micro_recall,
        micro_f1=derived.micro_f1,
        macro_precision=derived.macro_precision,
        macro_recall=derived.macro_recall,
        macro_f1=derived.macro_f1,
        mean_latency=derived.mean_latency,
    )


def select_recommended(sweeps: list[SweepMetrics]) -> SweepMetrics:
    """Highest macro F1; ties broken by higher macro recall."""
    if not sweeps:
        raise ValueError("No sweeps to compare.")
    return max(sweeps, key=lambda s: (s.macro_f1, s.macro_recall, s.micro_recall))


def write_sweep_summary_csv(path: Path, sweeps: list[SweepMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strike_percentile",
        "input_dir",
        "TP",
        "FP",
        "FN",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "mean_latency",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in sorted(sweeps, key=lambda x: x.strike_percentile):
            w.writerow(
                {
                    "strike_percentile": s.strike_percentile,
                    "input_dir": str(s.input_dir),
                    "TP": s.tp,
                    "FP": s.fp,
                    "FN": s.fn,
                    "micro_precision": f"{s.micro_precision:.6f}",
                    "micro_recall": f"{s.micro_recall:.6f}",
                    "micro_f1": f"{s.micro_f1:.6f}",
                    "macro_precision": f"{s.macro_precision:.6f}",
                    "macro_recall": f"{s.macro_recall:.6f}",
                    "macro_f1": f"{s.macro_f1:.6f}",
                    "mean_latency": f"{s.mean_latency:.6f}",
                }
            )


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def write_sweep_summary_tex(path: Path, sweeps: list[SweepMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by fightsafe_ai.evaluation.summarize_sweeps",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Strike-percentile sweep summary (BoxingVI batch "
        "\\texttt{boxingvi\\_results\\_all.csv} pools).}",
        "\\label{tab:boxingvi-sweep-summary}",
        "\\begin{tabular}{@{}rrrrrrrrrrr@{}}",
        "\\toprule",
        "\\textbf{p\\%} & \\textbf{TP} & \\textbf{FP} & \\textbf{FN} & "
        "$\\mu$P & $\\mu$R & $\\mu$F1 & MP & MR & MF1 & $\\overline{\\Delta t}$ \\\\",
        "\\midrule",
    ]
    for s in sorted(sweeps, key=lambda x: x.strike_percentile):
        lines.append(
            f"{s.strike_percentile} & {s.tp} & {s.fp} & {s.fn} & "
            f"{s.micro_precision:.4f} & {s.micro_recall:.4f} & {s.micro_f1:.4f} & "
            f"{s.macro_precision:.4f} & {s.macro_recall:.4f} & {s.macro_f1:.4f} & "
            f"{s.mean_latency:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sweep_summary_md(
    path: Path, sweeps: list[SweepMetrics], *, recommended: SweepMetrics
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BoxingVI sweep summary",
        "",
        "Pooled metrics from each input directory's ``boxingvi_results_all.csv``.",
        "",
        "| p% | TP | FP | FN | μP | μR | μF1 | MP | MR | MF1 | mean Δt (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(sweeps, key=lambda x: x.strike_percentile):
        lines.append(
            f"| {s.strike_percentile} | {s.tp} | {s.fp} | {s.fn} | "
            f"{s.micro_precision:.4f} | {s.micro_recall:.4f} | {s.micro_f1:.4f} | "
            f"{s.macro_precision:.4f} | {s.macro_recall:.4f} | {s.macro_f1:.4f} | "
            f"{s.mean_latency:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Recommended configuration",
            "",
            f"- **strike_percentile** = **{recommended.strike_percentile}** "
            f"(highest macro F1; ties → higher macro recall).",
            f"- Input directory: `{recommended.input_dir}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_console_report(
    sweeps: list[SweepMetrics],
    *,
    recommended: SweepMetrics,
) -> str:
    best_micro = max(sweeps, key=lambda s: s.micro_f1)
    best_macro = max(sweeps, key=lambda s: (s.macro_f1, s.macro_recall))
    best_recall = max(sweeps, key=lambda s: s.macro_recall)
    lines = [
        "Sweep summary",
        f"  configurations: {len(sweeps)}",
        "",
        f"  Best micro F1:  {best_micro.micro_f1:.4f}  (p={best_micro.strike_percentile}, {best_micro.input_dir})",
        f"  Best macro F1:  {best_macro.macro_f1:.4f}  (p={best_macro.strike_percentile}, {best_macro.input_dir})",
        f"  Best macro recall: {best_recall.macro_recall:.4f}  (p={best_recall.strike_percentile}, {best_recall.input_dir})",
        "",
        "  Recommended configuration:",
        f"    strike_percentile = {recommended.strike_percentile}",
        "    (macro F1 tie-break: higher macro recall)",
        f"    directory: {recommended.input_dir}",
        "",
    ]
    return "\n".join(lines)


def run_summarize(
    input_dirs: list[Path],
    output_dir: Path,
    *,
    print_report: bool = True,
) -> list[SweepMetrics]:
    """Parse all input dirs, write summary artifacts, optionally print report."""
    sweeps: list[SweepMetrics] = []
    for raw in input_dirs:
        d = Path(raw).expanduser().resolve()
        csv_path = d / "boxingvi_results_all.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing {csv_path}")
        sweeps.append(parse_boxingvi_results_all_csv(csv_path, input_dir=d))

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rec = select_recommended(sweeps)

    write_sweep_summary_csv(out / "sweep_summary.csv", sweeps)
    write_sweep_summary_tex(out / "sweep_summary.tex", sweeps)
    write_sweep_summary_md(out / "sweep_summary.md", sweeps, recommended=rec)

    if print_report:
        print(format_console_report(sweeps, recommended=rec), end="")
    return sweeps


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Directories each containing boxingvi_results_all.csv (e.g. .../boxingvi_batch_p90).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write sweep_summary.{csv,tex,md}",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Do not print console summary.")
    args = p.parse_args(argv)

    try:
        run_summarize(
            [Path(x) for x in args.input_dirs],
            Path(args.output_dir),
            print_report=not args.quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "SweepMetrics",
    "format_console_report",
    "infer_strike_percentile",
    "main",
    "parse_boxingvi_results_all_csv",
    "run_summarize",
    "select_recommended",
    "write_sweep_summary_csv",
    "write_sweep_summary_md",
    "write_sweep_summary_tex",
]
