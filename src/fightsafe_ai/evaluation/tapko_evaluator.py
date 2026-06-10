"""
TapKO **event evaluator**: predicted intervals vs TapKO manual annotations.

Matching uses temporal IoU with optional symmetric time tolerance (see
:func:`~fightsafe_ai.evaluation.event_matching.match_events`). Supports **exact**
event types or **family**-level agreement (``submission_signal``, ``extreme_vulnerability``,
``negative``).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from fightsafe_ai.annotation.tapko_schema import (
    TapkoAnnotationDocument,
    TapkoAnnotationStatus,
    parse_tapko_json,
)
from fightsafe_ai.evaluation.event_matching import match_events
from fightsafe_ai.evaluation.event_metrics import EventMatch, EventWindow, temporal_iou


TapkoMatchMode = Literal["exact", "family"]

DEFAULT_OUTPUT_CSV: Final[str] = "tapko_results.csv"
DEFAULT_OUTPUT_TEX: Final[str] = "tapko_results.tex"
DEFAULT_OUTPUT_MD: Final[str] = "tapko_error_analysis.md"

WARN_DIAGNOSTIC_METRICS: Final[str] = (
    "WARNING: annotations are not visually confirmed; metrics are diagnostic only."
)


def warn_if_annotations_not_visually_confirmed(doc: TapkoAnnotationDocument) -> None:
    """Emit stderr when ground-truth QA gate is not ``visually_confirmed`` (evaluation still runs)."""

    if doc.annotation_status != TapkoAnnotationStatus.VISUALLY_CONFIRMED:
        print(WARN_DIAGNOSTIC_METRICS, file=sys.stderr)


def tapko_namespace(event_type: str) -> str:
    """First segment of a TapKO ``event_type`` (e.g. ``submission_signal``)."""
    s = str(event_type).strip()
    if "." not in s:
        return s
    return s.split(".", 1)[0]


def labels_compatible(ref_label: str, pred_label: str, *, mode: TapkoMatchMode) -> bool:
    if mode == "exact":
        return ref_label.strip() == pred_label.strip()
    return tapko_namespace(ref_label) == tapko_namespace(pred_label)


def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    if a0 > a1:
        a0, a1 = a1, a0
    if b0 > b1:
        b0, b1 = b1, b0
    return max(a0, b0) < min(a1, b1)


def _f_beta(precision: float, recall: float, *, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0.0 and math.isfinite(den) else 0.0


@dataclass
class TapkoEvalConfig:
    """Evaluation hyper-parameters."""

    iou_threshold: float = 0.3
    tolerance_seconds: float = 0.0
    match_mode: TapkoMatchMode = "exact"
    beta_f2: float = 2.0
    onset_early_seconds: float = 0.05
    """If ``pred.start < ref.start - onset_early_seconds``, tag **early** detection."""


@dataclass(frozen=True)
class TapkoClassMetrics:
    """Precision / recall / F-scores for one ``event_type`` label."""

    label: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    f2: float


@dataclass
class TapkoErrorRecord:
    """One row for qualitative error analysis."""

    video_id: str
    category: str
    detail: str
    ref_label: str | None
    pred_label: str | None
    ref_start: float | None
    ref_end: float | None
    pred_start: float | None
    pred_end: float | None
    iou: float | None


@dataclass
class TapkoEvalResult:
    """Full aggregate outcomes plus per-video bookkeeping."""

    config: TapkoEvalConfig
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    f2: float
    mean_onset_latency_sec: float
    mean_abs_onset_latency_sec: float
    false_positives_per_minute: float
    total_video_duration_min: float
    micro: dict[str, float]
    macro: dict[str, float]
    per_class: dict[str, TapkoClassMetrics]
    matches: list[EventMatch]
    errors: list[TapkoErrorRecord] = field(default_factory=list)


def annotations_to_windows_by_video(doc: TapkoAnnotationDocument) -> dict[str, list[EventWindow]]:
    """Group validated TapKO annotations into :class:`EventWindow` lists per ``video_id``."""
    out: dict[str, list[EventWindow]] = defaultdict(list)
    for a in doc.annotations:
        out[a.video_id].append(
            EventWindow(
                start=float(a.start_time),
                end=float(a.end_time),
                label=str(a.event_type.value),
            )
        )
    return dict(out)


def predictions_from_json_list(raw: Sequence[Mapping[str, Any]]) -> dict[str, list[EventWindow]]:
    """Build per-video prediction windows from a list of dicts (e.g. exported ``events.json``)."""
    out: dict[str, list[EventWindow]] = defaultdict(list)
    for row in raw:
        vid = row.get("video_id")
        if vid is None:
            continue
        t0 = row.get("start_time", row.get("startTime"))
        t1 = row.get("end_time", row.get("endTime"))
        lab = row.get("event_type", row.get("eventType", "UNKNOWN"))
        if t0 is None or t1 is None:
            continue
        out[str(vid)].append(
            EventWindow(start=float(t0), end=float(t1), label=str(lab).strip() or "UNKNOWN")
        )
    return dict(out)


def load_tapko_predictions_json(path: Path | str) -> dict[str, list[EventWindow]]:
    """Load ``[{video_id, start_time, end_time, event_type}, ...]`` JSON."""
    p = Path(path).expanduser().resolve()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("Predictions JSON must be an array of objects.")
    return predictions_from_json_list(raw)


def load_tapko_ground_truth_json(path: Path | str) -> TapkoAnnotationDocument:
    """Load and validate a TapKO annotation document."""
    p = Path(path).expanduser().resolve()
    return parse_tapko_json(p.read_text(encoding="utf-8"))


def _negative_windows_for_video(
    doc: TapkoAnnotationDocument, video_id: str
) -> dict[str, list[EventWindow]]:
    """Collect hard-negative intervals for FP tagging (scramble / hand posting)."""
    buckets: dict[str, list[EventWindow]] = defaultdict(list)
    for a in doc.annotations:
        if a.video_id != video_id:
            continue
        lab = str(a.event_type.value)
        if lab == "negative.normal_scramble":
            buckets["scramble"].append(
                EventWindow(start=float(a.start_time), end=float(a.end_time), label=lab)
            )
        elif lab == "negative.hand_posting":
            buckets["hand_posting"].append(
                EventWindow(start=float(a.start_time), end=float(a.end_time), label=lab)
            )
    return dict(buckets)


def _window_index(lst: list[EventWindow], w: EventWindow) -> int:
    for i, x in enumerate(lst):
        if x == w:
            return i
    return -1


def _fp_overlaps_negative(
    pred: EventWindow,
    negs: list[EventWindow],
) -> bool:
    for n in negs:
        if temporal_iou(pred, n) > 0.0 or _intervals_overlap(pred.start, pred.end, n.start, n.end):
            return True
    return False


def evaluate_tapko(
    ground_truth: TapkoAnnotationDocument,
    predictions: dict[str, list[EventWindow]],
    *,
    video_durations_sec: Mapping[str, float] | None = None,
    config: TapkoEvalConfig | None = None,
) -> TapkoEvalResult:
    """
    Evaluate predicted event intervals against TapKO references.

    Ground-truth positives are **all** non-negative TapKO types in the document (submission and
    extreme-vulnerability labels). ``negative.*`` intervals are excluded from TP/FN tallies but
    used to tag false positives during scramble/posting in :attr:`TapkoEvalResult.errors`.

    Parameters
    ----------
    ground_truth
        Validated TapKO document.
    predictions
        Mapping ``video_id -> list[EventWindow]`` (predicted ``label`` = event type).
    video_durations_sec
        Optional clip durations for **false positives per minute**. If omitted, durations are
        inferred per video as the maximum of annotation ends and prediction ends (weak lower bound).
    """
    cfg = config or TapkoEvalConfig()
    warn_if_annotations_not_visually_confirmed(ground_truth)
    gt_by_vid = annotations_to_windows_by_video(ground_truth)

    # Reference positives: submission_signal.* and extreme_vulnerability.* only.
    def _is_eval_positive_label(lab: str) -> bool:
        ns = tapko_namespace(lab)
        return ns in {"submission_signal", "extreme_vulnerability"}

    ref_pos_by_vid: dict[str, list[EventWindow]] = {}
    for vid, wins in gt_by_vid.items():
        ref_pos_by_vid[vid] = [w for w in wins if _is_eval_positive_label(w.label)]

    all_videos = sorted(set(ref_pos_by_vid.keys()) | set(predictions.keys()))

    inferred_duration: dict[str, float] = {}
    for vid in all_videos:
        ends: list[float] = []
        for w in gt_by_vid.get(vid, []):
            ends.extend([w.start, w.end])
        for w in predictions.get(vid, []):
            ends.extend([w.start, w.end])
        inferred_duration[vid] = max(ends) if ends else 0.0

    durations_sec: dict[str, float] = {}
    for vid in all_videos:
        if video_durations_sec is not None and vid in video_durations_sec:
            durations_sec[vid] = float(video_durations_sec[vid])
        else:
            durations_sec[vid] = inferred_duration.get(vid, 0.0)

    total_duration_min = sum(max(0.0, durations_sec.get(v, 0.0)) for v in all_videos) / 60.0

    tp = fp = fn = 0
    matches_out: list[EventMatch] = []
    errors: list[TapkoErrorRecord] = []

    onset_errors: list[float] = []
    abs_onset_errors: list[float] = []

    tp_per_class: dict[str, int] = defaultdict(int)
    fp_per_class: dict[str, int] = defaultdict(int)
    fn_per_class: dict[str, int] = defaultdict(int)

    ref_counts: dict[str, int] = defaultdict(int)
    pred_counts: dict[str, int] = defaultdict(int)

    for vid in all_videos:
        refs = ref_pos_by_vid.get(vid, [])
        preds = predictions.get(vid, [])
        for w in refs:
            ref_counts[w.label] += 1
        for w in preds:
            pred_counts[w.label] += 1

        neg_ctx = _negative_windows_for_video(ground_truth, vid)
        scramble_w = neg_ctx.get("scramble", [])
        posting_w = neg_ctx.get("hand_posting", [])

        structural = match_events(
            preds,
            refs,
            iou_threshold=cfg.iou_threshold,
            tolerance_seconds=cfg.tolerance_seconds,
            require_same_label=False,
        )
        matched_ref: set[int] = set()
        matched_pred: set[int] = set()
        ref_list = list(refs)
        pred_list = list(preds)

        for m in structural:
            ri = _window_index(ref_list, m.ref)
            pj = _window_index(pred_list, m.pred)
            if ri < 0 or pj < 0:
                continue
            matched_ref.add(ri)
            matched_pred.add(pj)

            ok = labels_compatible(m.ref.label, m.pred.label, mode=cfg.match_mode)
            if ok:
                tp += 1
                tp_per_class[m.ref.label] += 1
                matches_out.append(m)
                delta = m.pred.start - m.ref.start
                onset_errors.append(delta)
                abs_onset_errors.append(abs(delta))
                if cfg.match_mode == "family" and m.ref.label.strip() != m.pred.label.strip():
                    errors.append(
                        TapkoErrorRecord(
                            video_id=vid,
                            category="wrong_subtype",
                            detail="IoU match with same namespace but different subtype (family TP)",
                            ref_label=m.ref.label,
                            pred_label=m.pred.label,
                            ref_start=m.ref.start,
                            ref_end=m.ref.end,
                            pred_start=m.pred.start,
                            pred_end=m.pred.end,
                            iou=m.iou,
                        )
                    )
                early_thr = float(cfg.onset_early_seconds)
                if m.pred.start < m.ref.start - early_thr:
                    errors.append(
                        TapkoErrorRecord(
                            video_id=vid,
                            category="early_detection",
                            detail=f"pred onset earlier than ref by {-delta:.3f}s",
                            ref_label=m.ref.label,
                            pred_label=m.pred.label,
                            ref_start=m.ref.start,
                            ref_end=m.ref.end,
                            pred_start=m.pred.start,
                            pred_end=m.pred.end,
                            iou=m.iou,
                        )
                    )
                elif m.pred.start > m.ref.start + early_thr:
                    errors.append(
                        TapkoErrorRecord(
                            video_id=vid,
                            category="late_detection",
                            detail=f"pred onset later than ref by {delta:.3f}s",
                            ref_label=m.ref.label,
                            pred_label=m.pred.label,
                            ref_start=m.ref.start,
                            ref_end=m.ref.end,
                            pred_start=m.pred.start,
                            pred_end=m.pred.end,
                            iou=m.iou,
                        )
                    )
            else:
                fp += 1
                fn += 1
                fp_per_class[m.pred.label] += 1
                fn_per_class[m.ref.label] += 1
                fam_same = tapko_namespace(m.ref.label) == tapko_namespace(m.pred.label)
                if fam_same:
                    errors.append(
                        TapkoErrorRecord(
                            video_id=vid,
                            category="wrong_subtype",
                            detail="structural match but incompatible labels for current mode",
                            ref_label=m.ref.label,
                            pred_label=m.pred.label,
                            ref_start=m.ref.start,
                            ref_end=m.ref.end,
                            pred_start=m.pred.start,
                            pred_end=m.pred.end,
                            iou=m.iou,
                        )
                    )
                else:
                    errors.append(
                        TapkoErrorRecord(
                            video_id=vid,
                            category="wrong_subtype",
                            detail="structural match across namespaces (family mismatch)",
                            ref_label=m.ref.label,
                            pred_label=m.pred.label,
                            ref_start=m.ref.start,
                            ref_end=m.ref.end,
                            pred_start=m.pred.start,
                            pred_end=m.pred.end,
                            iou=m.iou,
                        )
                    )

        for ri, rw in enumerate(ref_list):
            if ri not in matched_ref:
                fn += 1
                fn_per_class[rw.label] += 1
                errors.append(
                    TapkoErrorRecord(
                        video_id=vid,
                        category="missed_event",
                        detail="no prediction reached IoU threshold",
                        ref_label=rw.label,
                        pred_label=None,
                        ref_start=rw.start,
                        ref_end=rw.end,
                        pred_start=None,
                        pred_end=None,
                        iou=None,
                    )
                )

        for pj, pw in enumerate(pred_list):
            if pj not in matched_pred:
                fp += 1
                fp_per_class[pw.label] += 1
                cat = "false_positive"
                detail = "unmatched prediction"
                if _fp_overlaps_negative(pw, scramble_w):
                    cat = "false_positive_scramble"
                    detail = "unmatched; overlaps negative.normal_scramble interval"
                elif _fp_overlaps_negative(pw, posting_w):
                    cat = "false_positive_hand_posting"
                    detail = "unmatched; overlaps negative.hand_posting interval"
                errors.append(
                    TapkoErrorRecord(
                        video_id=vid,
                        category=cat,
                        detail=detail,
                        ref_label=None,
                        pred_label=pw.label,
                        ref_start=None,
                        ref_end=None,
                        pred_start=pw.start,
                        pred_end=pw.end,
                        iou=None,
                    )
                )

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    f2 = _f_beta(precision, recall, beta=cfg.beta_f2)

    mean_onset = sum(onset_errors) / len(onset_errors) if onset_errors else 0.0
    mean_abs_onset = sum(abs_onset_errors) / len(abs_onset_errors) if abs_onset_errors else 0.0
    fp_per_min = _safe_div(fp, total_duration_min) if total_duration_min > 0 else 0.0

    micro = {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "mean_onset_latency_sec": mean_onset,
        "mean_abs_onset_latency_sec": mean_abs_onset,
        "false_positives_per_minute": fp_per_min,
    }

    all_classes = sorted(set(ref_counts.keys()) | set(pred_counts.keys()))
    per_class: dict[str, TapkoClassMetrics] = {}
    f1s: list[float] = []
    f2s: list[float] = []
    for lab in all_classes:
        tcp = tp_per_class.get(lab, 0)
        fcp = fp_per_class.get(lab, 0)
        fcn = fn_per_class.get(lab, 0)
        p_c = _safe_div(tcp, tcp + fcp)
        r_c = _safe_div(tcp, tcp + fcn)
        f1_c = _safe_div(2 * p_c * r_c, p_c + r_c)
        f2_c = _f_beta(p_c, r_c, beta=cfg.beta_f2)
        per_class[lab] = TapkoClassMetrics(
            label=lab,
            tp=tcp,
            fp=fcp,
            fn=fcn,
            precision=p_c,
            recall=r_c,
            f1=f1_c,
            f2=f2_c,
        )
        if ref_counts.get(lab, 0) > 0 or pred_counts.get(lab, 0) > 0:
            f1s.append(f1_c)
            f2s.append(f2_c)

    macro_p = sum(per_class[k].precision for k in per_class) / len(per_class) if per_class else 0.0
    macro_r = sum(per_class[k].recall for k in per_class) / len(per_class) if per_class else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    macro_f2 = sum(f2s) / len(f2s) if f2s else 0.0

    macro = {
        "precision": macro_p,
        "recall": macro_r,
        "f1": macro_f1,
        "f2": macro_f2,
    }

    return TapkoEvalResult(
        config=cfg,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        f2=f2,
        mean_onset_latency_sec=mean_onset,
        mean_abs_onset_latency_sec=mean_abs_onset,
        false_positives_per_minute=fp_per_min,
        total_video_duration_min=total_duration_min,
        micro=micro,
        macro=macro,
        per_class=per_class,
        matches=matches_out,
        errors=errors,
    )


def write_tapko_results_csv(result: TapkoEvalResult, path: Path | str) -> None:
    """Write aggregate + per-class metrics to CSV."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "scope": "micro",
            "label": "",
            "tp": result.tp,
            "fp": result.fp,
            "fn": result.fn,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "f2": result.f2,
            "mean_onset_latency_sec": result.mean_onset_latency_sec,
            "mean_abs_onset_latency_sec": result.mean_abs_onset_latency_sec,
            "false_positives_per_minute": result.false_positives_per_minute,
            "total_video_duration_min": result.total_video_duration_min,
        }
    )
    rows.append(
        {
            "scope": "macro",
            "label": "",
            "tp": "",
            "fp": "",
            "fn": "",
            "precision": result.macro["precision"],
            "recall": result.macro["recall"],
            "f1": result.macro["f1"],
            "f2": result.macro["f2"],
            "mean_onset_latency_sec": "",
            "mean_abs_onset_latency_sec": "",
            "false_positives_per_minute": "",
            "total_video_duration_min": "",
        }
    )
    for lab, cm in sorted(result.per_class.items()):
        rows.append(
            {
                "scope": "per_class",
                "label": lab,
                "tp": cm.tp,
                "fp": cm.fp,
                "fn": cm.fn,
                "precision": cm.precision,
                "recall": cm.recall,
                "f1": cm.f1,
                "f2": cm.f2,
                "mean_onset_latency_sec": "",
                "mean_abs_onset_latency_sec": "",
                "false_positives_per_minute": "",
                "total_video_duration_min": "",
            }
        )

    fieldnames = [
        "scope",
        "label",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "f2",
        "mean_onset_latency_sec",
        "mean_abs_onset_latency_sec",
        "false_positives_per_minute",
        "total_video_duration_min",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_tapko_results_tex(result: TapkoEvalResult, path: Path | str) -> None:
    """Write a compact LaTeX tabular suitable for the paper supplement."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Scope & TP & FP & FN & P & R & F1 \\",
        r"\hline",
        (
            f"micro & {result.tp} & {result.fp} & {result.fn} & "
            f"{result.precision:.4f} & {result.recall:.4f} & {result.f1:.4f} \\\\"
        ),
        (
            f"macro & --- & --- & --- & "
            f"{result.macro['precision']:.4f} & {result.macro['recall']:.4f} & "
            f"{result.macro['f1']:.4f} \\\\"
        ),
        r"\hline",
        (
            f"\\multicolumn{{7}}{{l}}{{"
            f"\\footnotesize IoU$\\geq${cfg.iou_threshold}, "
            f"tol={cfg.tolerance_seconds}s, mode={cfg.match_mode}, "
            f"F2={result.f2:.4f}, FP/min={result.false_positives_per_minute:.4f}, "
            f"$|\\Delta onset|$={result.mean_abs_onset_latency_sec:.4f}s}} \\\\"
        ),
        r"\hline",
        r"\end{tabular}",
        "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")


def write_tapko_error_analysis_md(result: TapkoEvalResult, path: Path | str) -> None:
    """Summarize error categories for reviewer-facing QA."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    for e in result.errors:
        counts[e.category] += 1

    lines = [
        "# TapKO error analysis",
        "",
        "Decision-support evaluation — not officiating metrics.",
        "",
        "## Counts by category",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]
    for cat, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {cat} | {n} |")
    lines.extend(
        [
            "",
            "## Examples (up to 40 rows)",
            "",
            "| video_id | category | detail | ref | pred | ref interval | pred interval | IoU |",
            "|----------|----------|--------|-----|------|--------------|---------------|-----|",
        ]
    )
    for e in result.errors[:40]:
        ref_iv = (
            f"{e.ref_start:.3f}–{e.ref_end:.3f}"
            if e.ref_start is not None and e.ref_end is not None
            else ""
        )
        pred_iv = (
            f"{e.pred_start:.3f}–{e.pred_end:.3f}"
            if e.pred_start is not None and e.pred_end is not None
            else ""
        )
        iou_s = f"{e.iou:.4f}" if e.iou is not None else ""
        lines.append(
            f"| {e.video_id} | {e.category} | {e.detail} | "
            f"{e.ref_label or ''} | {e.pred_label or ''} | {ref_iv} | {pred_iv} | {iou_s} |"
        )
    lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")


def run_tapko_evaluation_and_write(
    ground_truth_path: Path | str,
    predictions_path: Path | str,
    output_dir: Path | str,
    *,
    video_durations_sec: Mapping[str, float] | None = None,
    config: TapkoEvalConfig | None = None,
    csv_name: str = DEFAULT_OUTPUT_CSV,
    tex_name: str = DEFAULT_OUTPUT_TEX,
    md_name: str = DEFAULT_OUTPUT_MD,
) -> TapkoEvalResult:
    """
    Load GT/preds from JSON, evaluate, and write ``tapko_results.csv``, ``.tex``, and error MD.

    Parameters
    ----------
    ground_truth_path
        TapKO annotation JSON (:class:`~fightsafe_ai.annotation.tapko_schema.TapkoAnnotationDocument`).
    predictions_path
        JSON array of prediction dicts (``video_id``, ``start_time``, ``end_time``, ``event_type``).
    output_dir
        Directory receiving the three output files.
    """
    gt = load_tapko_ground_truth_json(ground_truth_path)
    pred_map = load_tapko_predictions_json(predictions_path)
    res = evaluate_tapko(
        gt,
        pred_map,
        video_durations_sec=video_durations_sec,
        config=config,
    )
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_tapko_results_csv(res, out / csv_name)
    write_tapko_results_tex(res, out / tex_name)
    write_tapko_error_analysis_md(res, out / md_name)
    return res


__all__ = [
    "DEFAULT_OUTPUT_CSV",
    "DEFAULT_OUTPUT_MD",
    "DEFAULT_OUTPUT_TEX",
    "WARN_DIAGNOSTIC_METRICS",
    "TapkoClassMetrics",
    "TapkoErrorRecord",
    "TapkoEvalConfig",
    "TapkoEvalResult",
    "annotations_to_windows_by_video",
    "evaluate_tapko",
    "labels_compatible",
    "load_tapko_ground_truth_json",
    "load_tapko_predictions_json",
    "predictions_from_json_list",
    "run_tapko_evaluation_and_write",
    "tapko_namespace",
    "warn_if_annotations_not_visually_confirmed",
    "write_tapko_error_analysis_md",
    "write_tapko_results_csv",
    "write_tapko_results_tex",
]
