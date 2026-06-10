"""
Summarize a TapKO **detect** + **evaluate** folder pair into a short Markdown report.

Reads standard artefacts:
``tapko_results.csv``, ``tapko_error_analysis.md`` under ``--eval-dir``;
``tapko_manifest.json``, ``tapko_predictions.json`` under ``--detect-dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_CSV = "tapko_results.csv"
DEFAULT_ERROR_MD = "tapko_error_analysis.md"
DEFAULT_MANIFEST_JSON = "tapko_manifest.json"
DEFAULT_PREDICTIONS_JSON = "tapko_predictions.json"

DEFAULT_PILOT_INTERPRETATION = (
    "This is a diagnostic end-to-end pilot. It validates the pipeline but demonstrates "
    "that heuristic-only detection is insufficient without visual annotation refinement, "
    "hard negatives, and temporal context."
)


def _read_micro_row(csv_path: Path) -> dict[str, Any]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("scope") or "").strip().lower() == "micro":
                return dict(row)
    raise ValueError(f"No micro aggregate row found in {csv_path}")


def _safe_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = row.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _safe_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    raw = row.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Manifest must be a JSON object: {path}")
    return data


def count_prediction_candidates(predictions_path: Path) -> int:
    raw = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"Predictions JSON must be an array: {predictions_path}")
    return len(raw)


def parse_error_examples_table(md_path: Path, *, limit: int = 5) -> list[list[str]]:
    """
    Parse the markdown examples table from ``tapko_error_analysis.md``.

    Returns up to ``limit`` rows as lists of cell strings
    (video_id, category, detail, ref, pred, ref_iv, pred_iv, iou).
    """
    text = md_path.read_text(encoding="utf-8")
    in_examples = False
    header_seen = False
    rows_out: list[list[str]] = []

    for line in text.splitlines():
        if line.strip().startswith("## Examples"):
            in_examples = True
            header_seen = False
            continue
        if in_examples and line.startswith("## "):
            break
        if not in_examples:
            continue
        stripped = line.strip()
        # Markdown separator: |---|---|... (not necessarily starting with "|---")
        if stripped.startswith("|") and "---" in stripped:
            header_seen = True
            continue
        if not header_seen or not stripped.startswith("|"):
            continue
        parsed = _parse_md_table_row(stripped)
        if parsed is None:
            continue
        rows_out.append(parsed)
        if len(rows_out) >= limit:
            break

    return rows_out


def _parse_md_table_row(line: str) -> list[str] | None:
    """Split a markdown pipe row into cells; tolerate ``|`` inside *detail*."""
    if not line.startswith("|"):
        return None
    cells = [c.strip() for c in line.split("|")]
    while cells and cells[0] == "":
        cells.pop(0)
    while cells and cells[-1] == "":
        cells.pop()
    if len(cells) < 8:
        return None
    if len(cells) == 8:
        return cells
    front = cells[:2]
    back = cells[-5:]
    detail_parts = cells[2:-5]
    merged_detail = " | ".join(detail_parts)
    return [front[0], front[1], merged_detail, *back]


def duration_minutes_micro(row: dict[str, Any], manifest: dict[str, Any]) -> float:
    d = _safe_float(row, "total_video_duration_min", 0.0)
    if d > 0.0:
        return d
    n_frames = manifest.get("n_frames")
    fps = manifest.get("fps")
    if isinstance(n_frames, (int, float)) and isinstance(fps, (int, float)) and float(fps) > 0:
        return float(n_frames) / float(fps) / 60.0
    return 0.0


def build_tapko_run_summary_md(
    *,
    video_id: str,
    duration_min: float,
    n_candidates: int,
    micro: dict[str, Any],
    error_example_rows: list[list[str]],
    interpretation: str,
) -> str:
    tp = _safe_int(micro, "tp")
    fp = _safe_int(micro, "fp")
    fn = _safe_int(micro, "fn")
    prec = _safe_float(micro, "precision")
    rec = _safe_float(micro, "recall")
    f1 = _safe_float(micro, "f1")
    f2 = _safe_float(micro, "f2")
    fp_min = _safe_float(micro, "false_positives_per_minute")

    lines = [
        "# TapKO run summary",
        "",
        "## Overview",
        "",
        f"- **video_id**: `{video_id}`",
        f"- **duration (min)**: {duration_min:.4f}",
        f"- **predicted candidates**: {n_candidates}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| TP | {tp} |",
        f"| FP | {fp} |",
        f"| FN | {fn} |",
        f"| Precision | {prec:.4f} |",
        f"| Recall | {rec:.4f} |",
        f"| F1 | {f1:.4f} |",
        f"| F2 | {f2:.4f} |",
        f"| False positives / minute | {fp_min:.4f} |",
        "",
        "## Top error examples",
        "",
    ]

    if error_example_rows:
        lines.extend(
            [
                "| video_id | category | detail | ref label | pred label | ref interval | pred interval | IoU |",
                "|----------|----------|--------|-----------|------------|--------------|---------------|-----|",
            ]
        )
        for cells in error_example_rows:
            esc = [_md_escape_cell(c) for c in cells]
            lines.append("| " + " | ".join(esc) + " |")
    else:
        lines.append("_No example rows found in error analysis (empty or missing table)._")
    lines.extend(["", "## Interpretation", "", interpretation.strip(), ""])

    return "\n".join(lines)


def _md_escape_cell(s: str) -> str:
    t = str(s).replace("\n", " ").strip()
    t = re.sub(r"\|", r"\\|", t)
    return t


def summarize_tapko_run_to_markdown(
    eval_dir: Path,
    detect_dir: Path,
    *,
    interpretation: str | None = None,
    results_csv_name: str = DEFAULT_RESULTS_CSV,
    error_md_name: str = DEFAULT_ERROR_MD,
    manifest_name: str = DEFAULT_MANIFEST_JSON,
    predictions_name: str = DEFAULT_PREDICTIONS_JSON,
    top_errors: int = 5,
) -> str:
    """
    Load TapKO artefact files and return a Markdown summary string.

    Parameters
    ----------
    eval_dir
        Directory containing ``tapko_results.csv`` and ``tapko_error_analysis.md``.
    detect_dir
        Directory containing ``tapko_manifest.json`` and ``tapko_predictions.json``.
    interpretation
        Closing paragraph; defaults to :data:`DEFAULT_PILOT_INTERPRETATION`.
    """
    eval_dir = eval_dir.expanduser().resolve()
    detect_dir = detect_dir.expanduser().resolve()

    csv_path = eval_dir / results_csv_name
    err_path = eval_dir / error_md_name
    man_path = detect_dir / manifest_name
    pred_path = detect_dir / predictions_name

    for p in (csv_path, err_path, man_path, pred_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    micro = _read_micro_row(csv_path)
    manifest = load_manifest(man_path)
    vid = str(manifest.get("video_id") or "").strip() or "(unknown)"
    dur = duration_minutes_micro(micro, manifest)
    n_pred = count_prediction_candidates(pred_path)
    examples = parse_error_examples_table(err_path, limit=top_errors)
    interp = interpretation if interpretation is not None else DEFAULT_PILOT_INTERPRETATION

    return build_tapko_run_summary_md(
        video_id=vid,
        duration_min=dur,
        n_candidates=n_pred,
        micro=micro,
        error_example_rows=examples,
        interpretation=interp,
    )


def write_tapko_run_summary(
    eval_dir: Path,
    detect_dir: Path,
    output_md: Path,
    *,
    interpretation: str | None = None,
    results_csv_name: str = DEFAULT_RESULTS_CSV,
    error_md_name: str = DEFAULT_ERROR_MD,
    manifest_name: str = DEFAULT_MANIFEST_JSON,
    predictions_name: str = DEFAULT_PREDICTIONS_JSON,
    top_errors: int = 5,
) -> Path:
    """Write summary Markdown to ``output_md``; returns the resolved path."""
    body = summarize_tapko_run_to_markdown(
        eval_dir,
        detect_dir,
        interpretation=interpretation,
        results_csv_name=results_csv_name,
        error_md_name=error_md_name,
        manifest_name=manifest_name,
        predictions_name=predictions_name,
        top_errors=top_errors,
    )
    out = output_md.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Summarize TapKO detect + evaluate outputs into a Markdown report.",
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help=f"Evaluation output directory (expects {DEFAULT_RESULTS_CSV}, {DEFAULT_ERROR_MD}).",
    )
    p.add_argument(
        "--detect-dir",
        type=Path,
        required=True,
        help=f"Detection run directory (expects {DEFAULT_MANIFEST_JSON}, {DEFAULT_PREDICTIONS_JSON}).",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        required=True,
        help="Path to write the summary Markdown file.",
    )
    p.add_argument(
        "--interpretation",
        type=str,
        default=None,
        help="Final interpretation paragraph (default: pilot text).",
    )
    p.add_argument(
        "--top-errors",
        type=int,
        default=5,
        help="Number of error-example rows to include from tapko_error_analysis.md (default: 5).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    write_tapko_run_summary(
        args.eval_dir,
        args.detect_dir,
        args.output_md,
        interpretation=args.interpretation,
        top_errors=max(1, int(args.top_errors)),
    )


if __name__ == "__main__":
    main()

__all__ = [
    "DEFAULT_PILOT_INTERPRETATION",
    "build_tapko_run_summary_md",
    "count_prediction_candidates",
    "load_manifest",
    "parse_error_examples_table",
    "summarize_tapko_run_to_markdown",
    "write_tapko_run_summary",
]
