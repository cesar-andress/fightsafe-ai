#!/usr/bin/env python3
"""Export sinica2026 TapKO pilot tables from tapko_results.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


CLASS_SHORT = {
    "extreme_vulnerability.no_intelligent_defense": ("NID", r"\path{extreme_vulnerability.no_intelligent_defense}"),
    "submission_signal.foot_tap": ("FT", r"\path{submission_signal.foot_tap}"),
    "submission_signal.hand_tap": ("HT", r"\path{submission_signal.hand_tap}"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f4(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return "---"
    return f"{float(value):.4f}"


def _texttt(value: str) -> str:
    return "\\texttt{" + value.replace("_", r"\_") + "}"


def _i(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return "0"
    return str(int(float(value)))


def write_pilot_results_tex(rows: list[dict[str, str]], out: Path, *, video_id: str, n_candidates: int) -> None:
    micro = next(r for r in rows if r.get("scope") == "micro")
    duration = float(micro.get("total_video_duration_min") or 0.0)
    n_refs = 10  # draft reference windows in jedi_submissions.json (documented in paper)

    body = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Supervisory workflow demonstration interval bookkeeping "
        "(draft references, one validation clip).}\n"
        "\\label{tab:tapko_pilot_results}\n"
        "\\small\n"
        "\\begin{tabular}{@{}lr@{}}\n"
        "\\toprule\n"
        "\\textbf{Metric} & \\textbf{Value} \\\\\n"
        "\\midrule\n"
        f"Video ID & {_texttt(video_id)} \\\\\n"
        f"Duration (min) & ${duration:.4f}$ \\\\\n"
        f"Annotated candidate windows & ${n_refs}$ \\\\\n"
        f"Predicted candidates & ${n_candidates}$ \\\\\n"
        f"TP & {_i(micro.get('tp'))} \\\\\n"
        f"FP & {_i(micro.get('fp'))} \\\\\n"
        f"FN & {_i(micro.get('fn'))} \\\\\n"
        f"Precision & {_f4(micro.get('precision'))} \\\\\n"
        f"Recall & {_f4(micro.get('recall'))} \\\\\n"
        f"F1 & {_f4(micro.get('f1'))} \\\\\n"
        f"F2 & {_f4(micro.get('f2'))} \\\\\n"
        f"Mean onset latency (s) & {_f4(micro.get('mean_onset_latency_sec'))} \\\\\n"
        f"False positives per minute & {_f4(micro.get('false_positives_per_minute'))} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def write_per_class_tex(rows: list[dict[str, str]], out: Path) -> None:
    class_rows = [r for r in rows if r.get("scope") == "per_class"]
    lines = []
    legend = []
    for r in class_rows:
        label = (r.get("label") or "").strip()
        short, path = CLASS_SHORT.get(label, (label[:3].upper(), label))
        lines.append(
            f"{short} & ${_i(r.get('tp'))}$ & ${_i(r.get('fp'))}$ & ${_i(r.get('fn'))}$ & "
            f"${_f4(r.get('precision'))}$ & ${_f4(r.get('recall'))}$ & ${_f4(r.get('f1'))}$ \\\\"
        )
        legend.append(f"{short} = {path}")

    legend_block = ";\n".join(legend) + "."
    body = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Per-channel supervisory load (draft references; FightSafe-TapKO prototype export).}\n"
        "\\label{tab:tapko_pilot_per_class}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lrrrrrr@{}}\n"
        "\\toprule\n"
        "\\textbf{Class} & \\textbf{TP} & \\textbf{FP} & \\textbf{FN} & "
        "\\textbf{P} & \\textbf{R} & \\textbf{F1} \\\\\n"
        "\\midrule\n"
        + "\n".join(lines)
        + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n\n"
        "\\vspace{0.12em}\n"
        f"{{\\small {legend_block}}}\n"
        "\\end{table}\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def count_predictions(path: Path) -> int:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON list in {path}")
    return len(data)


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    default_csv = root / "outputs/tapko/jedi_submissions_eval/tapko_results.csv"
    default_pred = root / "outputs/tapko/jedi_submissions/tapko_predictions.json"
    default_out = root / "outputs/repro/sinica2026/tables"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-csv", type=Path, default=default_csv)
    p.add_argument("--predictions-json", type=Path, default=default_pred)
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument(
        "--install",
        action="store_true",
        help="Also copy tables into SINICA_DIR/tables/ (default: ../sinica2026).",
    )
    p.add_argument("--sinica-dir", type=Path, default=Path("../sinica2026"))
    p.add_argument("--video-id", default="jedi_submissions")
    args = p.parse_args(argv)

    if not args.results_csv.is_file():
        print(f"ERROR: results CSV not found: {args.results_csv}", file=sys.stderr)
        return 1
    if not args.predictions_json.is_file():
        print(f"ERROR: predictions JSON not found: {args.predictions_json}", file=sys.stderr)
        return 1

    rows = _read_rows(args.results_csv)
    n_pred = count_predictions(args.predictions_json)

    results_tex = args.output_dir / "tapko_pilot_results.tex"
    per_class_tex = args.output_dir / "tapko_pilot_per_class.tex"
    write_pilot_results_tex(rows, results_tex, video_id=args.video_id, n_candidates=n_pred)
    write_per_class_tex(rows, per_class_tex)
    print(f"Wrote {results_tex}")
    print(f"Wrote {per_class_tex}")

    if args.install:
        sinica_tables = (args.sinica_dir if args.sinica_dir.is_absolute() else root / args.sinica_dir) / "tables"
        sinica_tables.mkdir(parents=True, exist_ok=True)
        for src in (results_tex, per_class_tex):
            dest = sinica_tables / src.name
            dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Installed {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
