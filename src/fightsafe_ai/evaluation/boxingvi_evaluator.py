"""
Event-level evaluation: FightSafe BoxingVI predictions vs punch annotations (Excel).

Ground-truth intervals come from ``Annotation_files/V*.xlsx`` or ``annotations/V*.xlsx``
(columns: start frame, end frame, class). Punch rows are mapped to a generic **impact** label.

Predictions are read from ``boxingvi_predictions_<video_id>.json`` (see
:mod:`fightsafe_ai.evaluation.boxingvi_skeleton_runner`).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

import fightsafe_ai.datasets.boxingvi as boxingvi_ds
from fightsafe_ai.datasets.boxingvi import BoxingVIEvent
from fightsafe_ai.evaluation.event_metrics import EventWindow
from fightsafe_ai.evaluation.metrics import EventEvaluationResult, evaluate_event_prediction


logger = logging.getLogger(__name__)

# BoxingVI punch labels → generic impact ground-truth (slug keys match boxingvi._slug_key).
_PUNCH_CLASS_SLUGS: Final[frozenset[str]] = frozenset(
    {
        "cross",
        "jab",
        "lead_hook",
        "lead_uppercut",
        "rear_hook",
        "rear_uppercut",
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


def resolve_annotation_xlsx(dataset_root: Path, video_id: str) -> Path:
    """
    Prefer ``Annotation_files/<video_id>.xlsx``, then ``annotations/<video_id>.xlsx``.
    """
    root = Path(dataset_root).expanduser().resolve()
    for sub in ("Annotation_files", "annotations"):
        p = root / sub / f"{video_id}.xlsx"
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No annotation workbook found for {video_id!r} under {root}/Annotation_files or "
        f"{root}/annotations (expected {video_id}.xlsx)."
    )


def _is_gt_punch_impact(class_name: str) -> bool:
    slug = boxingvi_ds._slug_key(str(class_name))
    return slug in _PUNCH_CLASS_SLUGS


def resolve_skeleton_frame_count(dataset_root: Path, video_id: str) -> int | None:
    """
    Return ``T`` from ``<dataset_root>/skeleton/<video_id>.npy`` (first dimension of the array).

    Returns ``None`` if the file is missing.
    """
    root = Path(dataset_root).expanduser().resolve()
    stem = str(video_id).strip()
    if not stem:
        return None
    p = root / "skeleton" / f"{stem}.npy"
    if not p.is_file():
        return None
    arr = np.load(p, mmap_mode="r", allow_pickle=False)
    return int(arr.shape[0])


def _resolve_annotation_frame_offset(
    *,
    min_ann_start: int,
    max_ann_end: int,
    num_skeleton_frames: int | None,
    frame_offset: int | None,
    no_frame_offset: bool,
) -> int:
    """
    Ground-truth annotations may use **global** video indices while skeleton arrays are **local**.

    - ``no_frame_offset``: force offset 0 (no correction).
    - ``frame_offset`` set to an ``int``: use that offset explicitly.
    - Otherwise (**auto**): if ``num_skeleton_frames`` is known and
      ``max_ann_end >= num_skeleton_frames``, use ``min_ann_start`` as offset.
    """
    if no_frame_offset:
        return 0
    if frame_offset is not None:
        return int(frame_offset)
    if num_skeleton_frames is not None and max_ann_end >= num_skeleton_frames:
        return int(min_ann_start)
    return 0


def load_ground_truth_impact_windows(
    *,
    dataset_root: Path,
    video_id: str,
    fps: float,
    annotation_path: Path | None = None,
    num_skeleton_frames: int | None = None,
    frame_offset: int | None = None,
    no_frame_offset: bool = False,
) -> list[EventWindow]:
    """
    Load BoxingVI Excel rows whose class is a punch type; convert frame intervals to seconds.

    Annotations are read with the same parser as :func:`fightsafe_ai.datasets.boxingvi.load_events_from_xlsx`
    (all sheets, variable header row). If the workbook cannot be parsed, logs a warning and returns
    an empty list.

    When ``num_skeleton_frames`` is set, annotations are shifted into **local** skeleton indices
    (subtract offset), filtered to the skeleton range, and clipped to ``[0, T-1]``.

    Time interval: half-open ``[start_frame, end_frame]`` treated as inclusive frame indices on
    **both ends**, so ``end_sec = (local_end_frame + 1) / fps`` after alignment.

    **Offset (auto):** if the largest annotation end frame is ``>= num_skeleton_frames``, the offset
    is set to the minimum start frame in the sheet so local indices align with the skeleton array.
    """
    path = (
        annotation_path
        if annotation_path is not None
        else resolve_annotation_xlsx(dataset_root, video_id)
    )
    path = Path(path).expanduser().resolve()
    boxingvi_ds._require_openpyxl()
    # Use the same multi-sheet + header-row scan as :func:`load_events_from_xlsx` (not a single
    # ``read_excel(..., header=0)`` on sheet 0), so workbooks with a non-default layout still load.
    try:
        events: list[BoxingVIEvent] = boxingvi_ds.load_events_from_xlsx(
            path, skip_invalid_rows=True
        )
    except ValueError as exc:
        logger.warning(
            "No parseable BoxingVI events in %s (%s). Using empty ground-truth windows.",
            path,
            exc,
        )
        return []
    fd = float(fps)
    if fd <= 0:
        raise ValueError("fps must be positive.")

    if not events:
        return []

    all_starts = [e.start_frame for e in events]
    all_ends = [e.end_frame for e in events]
    min_ann_start = int(min(all_starts))
    max_ann_end = int(max(all_ends))

    offset = _resolve_annotation_frame_offset(
        min_ann_start=min_ann_start,
        max_ann_end=max_ann_end,
        num_skeleton_frames=num_skeleton_frames,
        frame_offset=frame_offset,
        no_frame_offset=no_frame_offset,
    )

    out: list[EventWindow] = []
    first_local_debug: list[tuple[int, int, float, float]] = []

    for ev in events:
        if not _is_gt_punch_impact(ev.class_name):
            continue
        ls = int(ev.start_frame) - offset
        le = int(ev.end_frame) - offset
        if num_skeleton_frames is not None and not no_frame_offset:
            t_max = int(num_skeleton_frames) - 1
            if le < 0 or ls >= int(num_skeleton_frames):
                continue
            ls = max(0, ls)
            le = min(t_max, le)
            if ls > le:
                continue
        t0 = float(ls) / fd
        t1 = float(le + 1) / fd
        out.append(EventWindow(start=t0, end=t1, label="impact"))
        if len(first_local_debug) < 5:
            first_local_debug.append((ls, le, t0, t1))

    logger.info(
        "BoxingVI GT alignment: skeleton_frames=%s annotation_min_start=%s annotation_max_end=%s "
        "offset=%s valid_punch_gt=%s",
        num_skeleton_frames,
        min_ann_start,
        max_ann_end,
        offset,
        len(out),
    )
    for i, (lf0, lf1, ts0, ts1) in enumerate(first_local_debug):
        logger.info(
            "  local_gt[%s]: frames [%s,%s] -> time [%.6f, %.6f]s",
            i,
            lf0,
            lf1,
            ts0,
            ts1,
        )

    return out


def _qualifies_impact_like_prediction(obj: Any) -> bool:
    """
    True for impact / strike / high-risk style predictions.

    - Risk rows (``event_level``): typically HIGH or CRITICAL.
    - Anomaly / bus rows: category ``impact`` or event_type containing ``strike`` / ``impact`` / ``risk.fused``.
    """
    if not isinstance(obj, dict):
        return False
    el = obj.get("event_level")
    if el is not None and str(el).strip().upper() in {"HIGH", "CRITICAL"}:
        return True
    cat = str(obj.get("category", "")).strip().lower()
    if cat in {"impact", "strike_impact", "strike"}:
        return True
    et = str(obj.get("event_type", "")).lower()
    if any(
        k in et
        for k in (
            "strike",
            "impact",
            "risk.fused",
            "high_risk",
        )
    ):
        return True
    title = str(obj.get("title", "")).lower()
    return bool("high_risk" in title or "guard_strike" in title)


PredictionSubset = Literal["full_fusion", "risk_only", "strike_only"]


def _windows_from_event_dicts(rows: list[Any]) -> list[EventWindow]:
    out: list[EventWindow] = []
    for ev in rows:
        if not isinstance(ev, dict):
            continue
        if not _qualifies_impact_like_prediction(ev):
            continue
        t0 = ev.get("start_time")
        t1 = ev.get("end_time")
        if t0 is None or t1 is None:
            continue
        lab = str(ev.get("event_level") or ev.get("event_type") or "pred")
        out.append(
            EventWindow(
                start=float(t0),
                end=float(t1),
                label=lab[:80],
            )
        )
    return out


def load_prediction_impact_windows(
    predictions_path: Path,
    *,
    subset: PredictionSubset | str = "full_fusion",
) -> list[EventWindow]:
    """
    Load prediction intervals from a boxingvi skeleton-runner JSON.

    ``subset``:

    - ``full_fusion``: merged ``events`` plus serialized ``anomaly_events`` (default).
    - ``risk_only``: interpretable risk episodes only (``events_risk_only``).
    - ``strike_only``: heuristic wrist-speed strikes only (``strike_events``).
    """
    p = Path(predictions_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Predictions file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("predictions JSON must be an object.")

    sub = str(subset).strip().lower().replace("-", "_")
    if sub == "strike_only":
        strikes = raw.get("strike_events")
        if isinstance(strikes, list):
            return _windows_from_event_dicts(strikes)
        return []

    if sub == "risk_only":
        risk_only = raw.get("events_risk_only")
        out: list[EventWindow] = []
        if isinstance(risk_only, list):
            out.extend(_windows_from_event_dicts(risk_only))
        return out

    # full_fusion: merged timeline + anomaly detections
    out = []
    block = raw.get("events")
    if isinstance(block, list):
        out.extend(_windows_from_event_dicts(block))
    anom = raw.get("anomaly_events")
    if isinstance(anom, list):
        for ev in anom:
            if not isinstance(ev, dict):
                continue
            if not _qualifies_impact_like_prediction(ev):
                continue
            t0 = ev.get("start_time")
            t1 = ev.get("end_time")
            if t0 is None or t1 is None:
                continue
            lab = str(ev.get("event_type") or "anomaly")
            out.append(
                EventWindow(
                    start=float(t0),
                    end=float(t1),
                    label=lab[:80],
                )
            )
    return out


@dataclass(frozen=True, slots=True)
class BoxingVIEvalResult:
    video_id: str
    n_ground_truth_punches: int
    n_predicted_qualified: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_detection_latency_seconds: float
    mean_abs_detection_latency_seconds: float
    iou_threshold: float
    tolerance_seconds: float
    annotation_path: str
    predictions_path: str
    raw: EventEvaluationResult

    def to_csv_row(self) -> dict[str, Any]:
        r = self.raw
        return {
            "video_id": self.video_id,
            "n_ground_truth_punches": self.n_ground_truth_punches,
            "n_predicted_qualified": self.n_predicted_qualified,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": f"{self.precision:.6f}",
            "recall": f"{self.recall:.6f}",
            "f1": f"{self.f1:.6f}",
            "mean_detection_latency_seconds": f"{self.mean_detection_latency_seconds:.6f}",
            "mean_abs_detection_latency_seconds": f"{self.mean_abs_detection_latency_seconds:.6f}",
            "iou_threshold": f"{self.iou_threshold:.6f}",
            "tolerance_seconds": f"{self.tolerance_seconds:.6f}",
            "annotation_path": self.annotation_path,
            "predictions_path": self.predictions_path,
            "n_matches": len(r.matches),
        }


def evaluate_boxingvi_video(
    *,
    dataset_root: Path,
    video_id: str,
    predictions_path: Path,
    fps: float,
    tolerance_seconds: float = 0.5,
    iou_threshold: float = 0.01,
    annotation_path: Path | None = None,
    num_skeleton_frames: int | None = None,
    infer_skeleton_frames: bool = True,
    frame_offset: int | None = None,
    no_frame_offset: bool = False,
    prediction_subset: PredictionSubset | str = "full_fusion",
) -> BoxingVIEvalResult:
    """
    Compare impact-like predictions to punch ground-truth intervals.

    **Latency** = ``pred.start - ref.start`` for matched pairs (positive ⇒ prediction starts late).

    When ``infer_skeleton_frames`` is true and ``num_skeleton_frames`` is omitted, the skeleton
    length is read from ``<dataset_root>/skeleton/<video_id>.npy`` (if present) to align
    annotation frame indices to local skeleton frames.
    """
    vid = str(video_id).strip()
    root = Path(dataset_root).expanduser().resolve()
    t_frames = num_skeleton_frames
    if t_frames is None and infer_skeleton_frames:
        t_frames = resolve_skeleton_frame_count(root, vid)
    gt = load_ground_truth_impact_windows(
        dataset_root=dataset_root,
        video_id=vid,
        fps=float(fps),
        annotation_path=annotation_path,
        num_skeleton_frames=t_frames,
        frame_offset=frame_offset,
        no_frame_offset=no_frame_offset,
    )
    pred = load_prediction_impact_windows(
        predictions_path,
        subset=prediction_subset,
    )

    ann_p = (
        str(annotation_path.expanduser().resolve())
        if annotation_path
        else str(resolve_annotation_xlsx(Path(dataset_root), vid))
    )

    raw = evaluate_event_prediction(
        pred,
        gt,
        iou_threshold=float(iou_threshold),
        tolerance_seconds=float(tolerance_seconds),
        require_same_label=False,
    )

    return BoxingVIEvalResult(
        video_id=vid,
        n_ground_truth_punches=len(gt),
        n_predicted_qualified=len(pred),
        true_positives=raw.true_positives,
        false_positives=raw.false_positives,
        false_negatives=raw.false_negatives,
        precision=raw.precision,
        recall=raw.recall,
        f1=raw.f1,
        mean_detection_latency_seconds=raw.mean_onset_delay_seconds,
        mean_abs_detection_latency_seconds=raw.mean_abs_onset_delay_seconds,
        iou_threshold=float(iou_threshold),
        tolerance_seconds=float(tolerance_seconds),
        annotation_path=ann_p,
        predictions_path=str(Path(predictions_path).expanduser().resolve()),
        raw=raw,
    )


def write_boxingvi_results_csv(
    path: Path, row: BoxingVIEvalResult, *, append: bool = False
) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    d = row.to_csv_row()
    fieldnames = list(d.keys())
    write_header = not path.is_file() or not append
    mode = "a" if append and path.is_file() else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w" or write_header:
            w.writeheader()
        w.writerow(d)


def write_boxingvi_results_tex(path: Path, row: BoxingVIEvalResult) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    def cell_num(x: float, nd: int = 4) -> str:
        return f"{x:.{nd}f}".rstrip("0").rstrip(".")

    lines = [
        "% Auto-generated by fightsafe_ai.evaluation.boxingvi_evaluator — edit only if you know what changed.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{BoxingVI event-level evaluation (punch annotations vs impact-like predictions). "
        "Decision-support metrics only; not clinical validation.}",
        "\\label{tab:boxingvi-event-eval}",
        "\\begin{tabular}{@{}lccccccccc@{}}",
        "\\toprule",
        "\\textbf{video} & \\textbf{GT} & \\textbf{Pred} & \\textbf{TP} & \\textbf{FP} & \\textbf{FN} & "
        "\\textbf{P} & \\textbf{R} & \\textbf{F1} & \\textbf{mean $\\Delta$t (s)} \\\\",
        "\\midrule",
        f"{_tex_escape(row.video_id)} & {row.n_ground_truth_punches} & {row.n_predicted_qualified} & "
        f"{row.true_positives} & {row.false_positives} & {row.false_negatives} & "
        f"{cell_num(row.precision)} & {cell_num(row.recall)} & {cell_num(row.f1)} & "
        f"{cell_num(row.mean_detection_latency_seconds)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BoxingVI punch annotation vs FightSafe predictions."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--video-id", type=str, required=True)
    parser.add_argument(
        "--predictions", type=Path, required=True, help="boxingvi_predictions_<id>.json"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--tolerance-seconds", type=float, default=0.5)
    parser.add_argument(
        "--iou-threshold", type=float, default=0.01, help="Min temporal IoU on dilated windows"
    )
    parser.add_argument(
        "--annotation-xlsx",
        type=Path,
        default=None,
        help="Override annotation path (default: Auto-discover Annotation_files/ or annotations/)",
    )
    offset_mx = parser.add_mutually_exclusive_group()
    offset_mx.add_argument(
        "--no-frame-offset",
        action="store_true",
        help="Use annotation frame indices as-is (offset 0; disables auto alignment).",
    )
    offset_mx.add_argument(
        "--frame-offset",
        type=str,
        default="auto",
        metavar="VAL",
        help="auto (default): infer offset when global annotation indices exceed skeleton length; "
        "or an explicit non-negative integer.",
    )
    parser.add_argument(
        "--num-skeleton-frames",
        type=int,
        default=None,
        metavar="T",
        help="Override skeleton frame count T (default: infer from skeleton/<video_id>.npy).",
    )
    parser.add_argument(
        "--no-infer-skeleton",
        action="store_true",
        help="Do not read skeleton/<video_id>.npy for frame count (disables auto alignment unless "
        "--num-skeleton-frames is set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write boxingvi_results.{csv,tex} (default: outputs/evaluation)",
    )
    parser.add_argument("--append-csv", action="store_true", help="Append row to CSV if it exists")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.no_frame_offset:
        fo_spec: int | None = None
        no_off = True
    else:
        raw_off = str(args.frame_offset).strip().lower()
        if raw_off == "auto":
            fo_spec = None
        else:
            fo_spec = int(raw_off)
        no_off = False

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("outputs/evaluation")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "boxingvi_results.csv"
    tex_path = out_dir / "boxingvi_results.tex"

    res = evaluate_boxingvi_video(
        dataset_root=args.dataset_root.expanduser().resolve(),
        video_id=args.video_id,
        predictions_path=args.predictions.expanduser().resolve(),
        fps=float(args.fps),
        tolerance_seconds=float(args.tolerance_seconds),
        iou_threshold=float(args.iou_threshold),
        annotation_path=args.annotation_xlsx.expanduser().resolve()
        if args.annotation_xlsx
        else None,
        num_skeleton_frames=args.num_skeleton_frames,
        infer_skeleton_frames=not bool(args.no_infer_skeleton),
        frame_offset=fo_spec,
        no_frame_offset=no_off,
    )

    write_boxingvi_results_csv(csv_path, res, append=bool(args.append_csv))
    write_boxingvi_results_tex(tex_path, res)

    logger.info(
        "TP=%s FP=%s FN=%s P=%.4f R=%.4f F1=%.4f mean_latency=%.4fs",
        res.true_positives,
        res.false_positives,
        res.false_negatives,
        res.precision,
        res.recall,
        res.f1,
        res.mean_detection_latency_seconds,
    )
    logger.info("Wrote %s and %s", csv_path, tex_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
