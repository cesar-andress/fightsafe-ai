"""
Batch event-level evaluation for illustrative case-study runs vs manual annotation JSON files.

Maps narrative annotation filenames (``case_a_knockdown.json``) to pipeline output directories
(``cs_knockdown_001``, …) per ``configs/case_studies.yaml``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fightsafe_ai.annotation.loader import load_annotation_file
from fightsafe_ai.annotation.validator import validate_annotation_file
from fightsafe_ai.evaluation.event_matching import (
    annotation_file_to_ground_truth_windows,
    events_json_to_windows,
)
from fightsafe_ai.evaluation.metrics import evaluate_event_prediction


# Narrative id (annotation filename stem) -> output_dir under runs/case_studies
NARRATIVE_STEM_TO_OUTPUT_DIR: dict[str, str] = {
    "case_a_knockdown": "cs_knockdown_001",
    "case_b_tap": "cs_surrender_001",
    "case_c_limb_anomaly": "cs_limb_001",
    "case_d_post_choke_confusion": "case_d_post_choke_confusion",
    "case_e_manifest_superiority": "case_e_manifest_superiority",
    "case_f_referee_errors": "case_f_referee_errors",
}

ORDERED_NARRATIVE_KEYS: tuple[str, ...] = tuple(NARRATIVE_STEM_TO_OUTPUT_DIR.keys())


Status = Literal[
    "ok",
    "annotation_pending",
    "missing_annotation_file",
    "missing_run_dir",
    "missing_events_json",
    "annotation_invalid",
]


@dataclass(frozen=True, slots=True)
class CaseStudyEvalRow:
    case_id: str
    status: Status
    predicted_events: int | None
    annotated_events: int | None
    true_positives: int | None
    false_positives: int | None
    false_negatives: int | None
    precision: float | None
    recall: float | None
    f1: float | None
    mean_onset_delay_seconds: float | None
    mean_absolute_onset_delay_seconds: float | None


def _safe_count_events_json(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, list):
        return None
    return len(raw)


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def evaluate_one_case(
    narrative_key: str,
    *,
    runs_dir: Path,
    annotations_dir: Path,
    iou_threshold: float = 0.1,
    tolerance_seconds: float = 0.0,
    require_same_label: bool = False,
) -> CaseStudyEvalRow:
    """Evaluate a single narrative key; returns a row (possibly annotation_pending)."""
    ann_path = annotations_dir / f"{narrative_key}.json"
    out_dir_name = NARRATIVE_STEM_TO_OUTPUT_DIR.get(narrative_key)
    if out_dir_name is None:
        raise KeyError(f"Unknown narrative key {narrative_key!r}")

    if not ann_path.is_file():
        return CaseStudyEvalRow(
            case_id=narrative_key,
            status="missing_annotation_file",
            predicted_events=None,
            annotated_events=None,
            true_positives=None,
            false_positives=None,
            false_negatives=None,
            precision=None,
            recall=None,
            f1=None,
            mean_onset_delay_seconds=None,
            mean_absolute_onset_delay_seconds=None,
        )

    errs = validate_annotation_file(ann_path)
    if errs:
        return CaseStudyEvalRow(
            case_id=narrative_key,
            status="annotation_invalid",
            predicted_events=None,
            annotated_events=None,
            true_positives=None,
            false_positives=None,
            false_negatives=None,
            precision=None,
            recall=None,
            f1=None,
            mean_onset_delay_seconds=None,
            mean_absolute_onset_delay_seconds=None,
        )

    doc = load_annotation_file(ann_path)
    case_id = doc.case_id or narrative_key
    n_ann = len(doc.events)

    run_dir = runs_dir / out_dir_name
    events_path = run_dir / "events.json"

    if not run_dir.is_dir():
        return CaseStudyEvalRow(
            case_id=case_id,
            status="missing_run_dir",
            predicted_events=None,
            annotated_events=n_ann,
            true_positives=None,
            false_positives=None,
            false_negatives=None,
            precision=None,
            recall=None,
            f1=None,
            mean_onset_delay_seconds=None,
            mean_absolute_onset_delay_seconds=None,
        )

    n_pred = _safe_count_events_json(events_path)
    if n_pred is None:
        return CaseStudyEvalRow(
            case_id=case_id,
            status="missing_events_json",
            predicted_events=None,
            annotated_events=n_ann,
            true_positives=None,
            false_positives=None,
            false_negatives=None,
            precision=None,
            recall=None,
            f1=None,
            mean_onset_delay_seconds=None,
            mean_absolute_onset_delay_seconds=None,
        )

    if n_ann == 0:
        return CaseStudyEvalRow(
            case_id=case_id,
            status="annotation_pending",
            predicted_events=n_pred,
            annotated_events=0,
            true_positives=None,
            false_positives=None,
            false_negatives=None,
            precision=None,
            recall=None,
            f1=None,
            mean_onset_delay_seconds=None,
            mean_absolute_onset_delay_seconds=None,
        )

    pred = events_json_to_windows(events_path)
    ref = annotation_file_to_ground_truth_windows(ann_path)
    r = evaluate_event_prediction(
        pred,
        ref,
        iou_threshold=iou_threshold,
        tolerance_seconds=tolerance_seconds,
        require_same_label=require_same_label,
    )
    return CaseStudyEvalRow(
        case_id=case_id,
        status="ok",
        predicted_events=r.n_predicted,
        annotated_events=r.n_ground_truth,
        true_positives=r.true_positives,
        false_positives=r.false_positives,
        false_negatives=r.false_negatives,
        precision=r.precision,
        recall=r.recall,
        f1=r.f1,
        mean_onset_delay_seconds=r.mean_onset_delay_seconds,
        mean_absolute_onset_delay_seconds=r.mean_abs_onset_delay_seconds,
    )


def run_case_study_batch_evaluation(
    *,
    runs_dir: Path,
    annotations_dir: Path,
    iou_threshold: float = 0.1,
    tolerance_seconds: float = 0.0,
    require_same_label: bool = False,
    keys: tuple[str, ...] | None = None,
) -> list[CaseStudyEvalRow]:
    """Evaluate every narrative key (default: all six illustrative cases)."""
    seq = keys if keys is not None else ORDERED_NARRATIVE_KEYS
    return [
        evaluate_one_case(
            k,
            runs_dir=runs_dir,
            annotations_dir=annotations_dir,
            iou_threshold=iou_threshold,
            tolerance_seconds=tolerance_seconds,
            require_same_label=require_same_label,
        )
        for k in seq
    ]


def rows_to_csv_dicts(rows: list[CaseStudyEvalRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        row: dict[str, Any] = {
            "case_id": r.case_id,
            "evaluation_status": r.status,
            "predicted_events": r.predicted_events if r.predicted_events is not None else "",
            "annotated_events": r.annotated_events if r.annotated_events is not None else "",
            "true_positives": r.true_positives if r.true_positives is not None else "",
            "false_positives": r.false_positives if r.false_positives is not None else "",
            "false_negatives": r.false_negatives if r.false_negatives is not None else "",
            "precision": r.precision if r.precision is not None else "",
            "recall": r.recall if r.recall is not None else "",
            "f1": r.f1 if r.f1 is not None else "",
            "mean_onset_delay_seconds": r.mean_onset_delay_seconds
            if r.mean_onset_delay_seconds is not None
            else "",
            "mean_absolute_onset_delay_seconds": r.mean_absolute_onset_delay_seconds
            if r.mean_absolute_onset_delay_seconds is not None
            else "",
        }
        out.append(row)
    return out


def write_case_study_evaluation_csv(path: Path, rows: list[CaseStudyEvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "evaluation_status",
        "predicted_events",
        "annotated_events",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "mean_onset_delay_seconds",
        "mean_absolute_onset_delay_seconds",
    ]
    dicts = rows_to_csv_dicts(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(dicts)


def write_case_study_evaluation_tex(path: Path, rows: list[CaseStudyEvalRow]) -> None:
    """Write a booktabs table fragment (full table environment with caption + label)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by fightsafe evaluate-case-studies — do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Illustrative case studies: preliminary \\textbf{event-level} metrics vs manual annotations "
        "(when \\texttt{status} is \\texttt{ok}). "
        "\\textbf{annotation\\_pending}: empty ground-truth events — metrics omitted (not fabricated). "
        "Not a benchmark; not clinical or officiating validation.}",
        "\\label{tab:case-study-annotation-eval}",
        "\\begin{tabular}{@{}l l r r r r r r r r r r@{}}",
        "\\toprule",
        "\\textbf{case\\_id} & \\textbf{status} & \\textbf{pred} & \\textbf{ann} & "
        "\\textbf{TP} & \\textbf{FP} & \\textbf{FN} & \\textbf{P} & \\textbf{R} & \\textbf{F1} & "
        "\\textbf{mean $\\Delta$} & \\textbf{mean $|\\Delta|$} \\\\",
        "\\midrule",
    ]

    def _cell_num(x: float | int | None) -> str:
        if x is None:
            return "---"
        if isinstance(x, float):
            return f"{x:.4f}".rstrip("0").rstrip(".")
        return str(x)

    for r in rows:
        status_tex = _tex_escape(r.status)
        line = (
            f"{_tex_escape(r.case_id)} & {status_tex} & "
            f"{_cell_num(r.predicted_events)} & {_cell_num(r.annotated_events)} & "
            f"{_cell_num(r.true_positives)} & {_cell_num(r.false_positives)} & {_cell_num(r.false_negatives)} & "
            f"{_cell_num(r.precision)} & {_cell_num(r.recall)} & {_cell_num(r.f1)} & "
            f"{_cell_num(r.mean_onset_delay_seconds)} & {_cell_num(r.mean_absolute_onset_delay_seconds)} \\\\"
        )
        lines.append(line)

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "NARRATIVE_STEM_TO_OUTPUT_DIR",
    "ORDERED_NARRATIVE_KEYS",
    "CaseStudyEvalRow",
    "evaluate_one_case",
    "run_case_study_batch_evaluation",
    "write_case_study_evaluation_csv",
    "write_case_study_evaluation_tex",
]
