"""Frame-level, coverage, and event-retrieval metrics (numpy-free; research / evaluation)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from fightsafe_ai.evaluation.event_matching import match_events
from fightsafe_ai.evaluation.event_metrics import EventWindow


def _binary_vec(y: Sequence[Any], positive_label: Any) -> list[int]:
    if isinstance(positive_label, bool):
        return [1 if (bool(v) is positive_label) else 0 for v in y]
    return [1 if v == positive_label else 0 for v in y]


def frame_level_precision(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    positive_label: Any = 1,
) -> float:
    """
    Proportion of predicted positive frames that are true positives.

    ``positive_label`` matches :func:`precision_recall_f1` (e.g. ``1`` for int labels, ``True`` for bool).
    """
    p, _, _ = precision_recall_f1(
        _binary_vec(y_true, positive_label),
        _binary_vec(y_pred, positive_label),
        positive_label=1,
    )
    return p


def frame_level_recall(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    positive_label: Any = 1,
) -> float:
    """Proportion of ground-truth positive frames that are correctly predicted positive."""
    _, r, _ = precision_recall_f1(
        _binary_vec(y_true, positive_label),
        _binary_vec(y_pred, positive_label),
        positive_label=1,
    )
    return r


def frame_level_f1(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    positive_label: Any = 1,
) -> float:
    """Harmonic mean of frame-level precision and recall."""
    _, _, f1 = precision_recall_f1(
        _binary_vec(y_true, positive_label),
        _binary_vec(y_pred, positive_label),
        positive_label=1,
    )
    return f1


def precision_recall_f1(
    y_true: list[int],
    y_pred: list[int],
    positive_label: int = 1,
) -> tuple[float, float, float]:
    """
    Binary (or one-vs-rest) **precision, recall, F1** for integer labels.

    Returns ``(0,0,0)`` on empty input or zero predicted positives.
    """
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return 0.0, 0.0, 0.0
    tp = fp = fn = 0
    for i in range(n):
        t = 1 if y_true[i] == positive_label else 0
        p = 1 if y_pred[i] == positive_label else 0
        if t and p:
            tp += 1
        elif p and not t:
            fp += 1
        elif t and not p:
            fn += 1
    if tp + fp == 0:
        prec = 0.0
    else:
        prec = tp / (tp + fp)
    if tp + fn == 0:
        rec = 0.0
    else:
        rec = tp / (tp + fn)
    if prec + rec == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def false_alarm_rate_frames(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    positive_label: Any = 1,
) -> float:
    """
    **Frame-level false alarm rate** — :math:`\\mathrm{FP} / N_{\\mathrm{neg}}` where
    :math:`N_{\\mathrm{neg}}` is the number of ground-truth **negative** frames.

    If there are no ground-truth negatives, returns ``0.0`` (undefined convention).
    """
    t_bin = _binary_vec(y_true, positive_label)
    p_bin = _binary_vec(y_pred, positive_label)
    n = min(len(t_bin), len(p_bin))
    if n == 0:
        return 0.0
    fp = 0
    n_neg = 0
    for i in range(n):
        if t_bin[i] == 0:
            n_neg += 1
            if p_bin[i] == 1:
                fp += 1
    if n_neg == 0:
        return 0.0
    return fp / n_neg


def pose_coverage(pose_frame_valid: Sequence[bool]) -> float:
    """
    Share of frames with a **valid** pose flag (e.g. joint visibility / detection success).

    Returns ``0.0`` for empty input.
    """
    if not pose_frame_valid:
        return 0.0
    return sum(1 for v in pose_frame_valid if v) / len(pose_frame_valid)


# --- Event vs ground-truth comparison (temporal matching) ------------------------


@dataclass(frozen=True, slots=True)
class EventMatchDelay:
    """Onset and offset time differences (pred - ref) for a matched pair, in seconds."""

    ref_start: float
    ref_end: float
    pred_start: float
    pred_end: float
    onset_delay_seconds: float
    offset_delay_seconds: float
    iou: float


@dataclass
class EventEvaluationResult:
    """Result of :func:`evaluate_event_prediction`. JSON-serializable via :func:`asdict`."""

    n_predicted: int
    n_ground_truth: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    iou_threshold: float
    tolerance_seconds: float
    require_same_label: bool
    mean_onset_delay_seconds: float
    mean_abs_onset_delay_seconds: float
    matches: list[EventMatchDelay] = field(repr=False, default_factory=list)
    iou_by_match: list[float] = field(repr=False, default_factory=list)


def evaluate_event_prediction(
    predicted: list[EventWindow],
    ground_truth: list[EventWindow],
    *,
    iou_threshold: float = 0.1,
    tolerance_seconds: float = 0.0,
    require_same_label: bool = False,
) -> EventEvaluationResult:
    """
    Match predicted to reference using :func:`fightsafe_ai.evaluation.event_matching.match_events`
    and compute standard retrieval metrics plus per-match timing deltas.

    **Onset / offset delay** = ``pred_t - ref_t`` (negative means the model flags early).
    """
    m = match_events(
        predicted,
        ground_truth,
        iou_threshold=iou_threshold,
        tolerance_seconds=tolerance_seconds,
        require_same_label=require_same_label,
    )
    tp = len(m)
    n_p = len(predicted)
    n_r = len(ground_truth)
    fp = n_p - tp
    fn = n_r - tp
    prec = tp / n_p if n_p else 0.0
    rec = tp / n_r if n_r else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    delays: list[EventMatchDelay] = []
    onset_errs: list[float] = []
    abs_onset: list[float] = []
    ious: list[float] = []
    for p in m:
        r0, r1 = p.ref.start, p.ref.end
        p0, p1 = p.pred.start, p.pred.end
        d_on = p0 - r0
        d_off = p1 - r1
        delays.append(
            EventMatchDelay(
                ref_start=r0,
                ref_end=r1,
                pred_start=p0,
                pred_end=p1,
                onset_delay_seconds=d_on,
                offset_delay_seconds=d_off,
                iou=p.iou,
            )
        )
        onset_errs.append(d_on)
        abs_onset.append(abs(d_on))
        ious.append(p.iou)
    med = sum(onset_errs) / len(onset_errs) if onset_errs else 0.0
    mae = sum(abs_onset) / len(abs_onset) if abs_onset else 0.0
    return EventEvaluationResult(
        n_predicted=n_p,
        n_ground_truth=n_r,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=prec,
        recall=rec,
        f1=f1,
        iou_threshold=iou_threshold,
        tolerance_seconds=tolerance_seconds,
        require_same_label=require_same_label,
        mean_onset_delay_seconds=med,
        mean_abs_onset_delay_seconds=mae,
        matches=delays,
        iou_by_match=ious,
    )


def event_evaluation_to_json_dict(
    r: EventEvaluationResult,
) -> dict[str, Any]:
    """
    Build a fully JSON-encodable dict (lists of floats, no custom objects).
    """
    m_list = [asdict(x) for x in r.matches] if r.matches else []
    return {
        "n_predicted": r.n_predicted,
        "n_ground_truth": r.n_ground_truth,
        "true_positives": r.true_positives,
        "false_positives": r.false_positives,
        "false_negatives": r.false_negatives,
        "precision": r.precision,
        "recall": r.recall,
        "f1": r.f1,
        "iou_threshold": r.iou_threshold,
        "tolerance_seconds": r.tolerance_seconds,
        "require_same_label": r.require_same_label,
        "mean_onset_delay_seconds": r.mean_onset_delay_seconds,
        "mean_abs_onset_delay_seconds": r.mean_abs_onset_delay_seconds,
        "iou_by_match": list(r.iou_by_match),
        "matches": m_list,
    }
