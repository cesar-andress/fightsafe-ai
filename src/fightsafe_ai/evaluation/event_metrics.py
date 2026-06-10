"""Event-level overlap, timing, and alert metrics (research / evaluation, not officiating)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


__all__ = [
    "AlertConsistencyResult",
    "EventLevelMetrics",
    "EventMatch",
    "EventOverlapResult",
    "EventWindow",
    "alert_consistency",
    "best_match",
    "event_level_f1",
    "event_level_metrics",
    "event_level_precision",
    "event_level_recall",
    "false_alarm_rate_events",
    "match_events_greedy_iou",
    "mean_absolute_onset_error",
    "mean_time_to_alert_seconds",
    "temporal_iou",
]


@dataclass(frozen=True, slots=True)
class EventWindow:
    """A ground-truth or predicted time span in frame indices or seconds."""

    start: float
    end: float
    label: str = "event"


@dataclass
class EventOverlapResult:
    """Container for :func:`temporal_iou`."""

    iou: float
    pred: EventWindow
    ref: EventWindow


@dataclass(frozen=True, slots=True)
class EventMatch:
    """One matched reference / prediction pair after IoU-based association."""

    ref: EventWindow
    pred: EventWindow
    iou: float


@dataclass(frozen=True, slots=True)
class EventLevelMetrics:
    """Bundle of event-level scores from :func:`event_level_metrics`."""

    precision: float
    recall: float
    f1: float
    matches: list[EventMatch]
    n_ref: int
    n_pred: int
    true_positives: int


@dataclass(frozen=True, slots=True)
class AlertConsistencyResult:
    """Result of :func:`alert_consistency`."""

    exact_match_rate: float
    n: int
    n_agree: int


def temporal_iou(a: EventWindow, b: EventWindow) -> float:
    """
    1D temporal IoU on the real line (assumes start <= end; swaps if not).

    Returns ``0.0`` for degenerate or non-overlapping intervals.
    """
    a0, a1 = (a.start, a.end) if a.start <= a.end else (a.end, a.start)
    b0, b1 = (b.start, b.end) if b.start <= b.end else (b.end, b.start)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0.0:
        return 0.0
    la = a1 - a0
    lb = b1 - b0
    if la <= 0.0 or lb <= 0.0:
        return 0.0
    union = la + lb - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def best_match(pred: list[EventWindow], ref: list[EventWindow]) -> list[EventOverlapResult]:
    """
    Greedy match each **pred** to the best IoU **ref** (simple baseline; not Hungarian).
    """
    out: list[EventOverlapResult] = []
    used: set[int] = set()
    for p in pred:
        best_iou = 0.0
        best_j: int | None = None
        for j, r in enumerate(ref):
            if j in used:
                continue
            t = temporal_iou(p, r)
            if t > best_iou:
                best_iou = t
                best_j = j
        if best_j is not None:
            used.add(best_j)
            r = ref[best_j]
        else:
            r = p
        out.append(EventOverlapResult(iou=best_iou, pred=p, ref=r))
    return out


def match_events_greedy_iou(
    ref: list[EventWindow],
    pred: list[EventWindow],
    *,
    iou_threshold: float = 0.3,
) -> list[EventMatch]:
    """
    Greedily match **predicted** events to **reference** events (largest IoU first).

    Each ref and each pred is used at most once. Pairs with IoU below ``iou_threshold``
    are discarded. This is a research baseline, not a unique optimal assignment.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, r in enumerate(ref):
        for j, p in enumerate(pred):
            t = temporal_iou(r, p)
            if t >= iou_threshold:
                pairs.append((t, i, j))
    pairs.sort(key=lambda x: -x[0])
    used_r: set[int] = set()
    used_p: set[int] = set()
    matches: list[EventMatch] = []
    for t, i, j in pairs:
        if i in used_r or j in used_p:
            continue
        used_r.add(i)
        used_p.add(j)
        matches.append(EventMatch(ref=ref[i], pred=pred[j], iou=t))
    return matches


def event_level_metrics(
    ref: list[EventWindow],
    pred: list[EventWindow],
    *,
    iou_threshold: float = 0.3,
) -> EventLevelMetrics:
    """
    Event-level **precision** , **recall** , and **F1** from one-to-one IoU matches.

    A predicted event is a **true positive** if it matches a reference event with
    :func:`temporal_iou` ≥ ``iou_threshold`` under the greedy matcher.
    """
    m = match_events_greedy_iou(ref, pred, iou_threshold=iou_threshold)
    tp = len(m)
    n_p = len(pred)
    n_r = len(ref)
    prec = tp / n_p if n_p > 0 else 0.0
    rec = tp / n_r if n_r > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return EventLevelMetrics(
        precision=prec,
        recall=rec,
        f1=f1,
        matches=m,
        n_ref=n_r,
        n_pred=n_p,
        true_positives=tp,
    )


def event_level_precision(
    ref: list[EventWindow], pred: list[EventWindow], *, iou_threshold: float = 0.3
) -> float:
    return event_level_metrics(ref, pred, iou_threshold=iou_threshold).precision


def event_level_recall(
    ref: list[EventWindow], pred: list[EventWindow], *, iou_threshold: float = 0.3
) -> float:
    return event_level_metrics(ref, pred, iou_threshold=iou_threshold).recall


def event_level_f1(
    ref: list[EventWindow], pred: list[EventWindow], *, iou_threshold: float = 0.3
) -> float:
    return event_level_metrics(ref, pred, iou_threshold=iou_threshold).f1


def mean_absolute_onset_error(matches: list[EventMatch], *, absolute: bool = True) -> float:
    """
    Mean of onset errors for matched event pairs: ``pred.start - ref.start``.

    If ``absolute`` is true, use absolute error. Returns ``0.0`` with no matches.
    """
    if not matches:
        return 0.0
    errs: list[float] = []
    for m in matches:
        d = m.pred.start - m.ref.start
        errs.append(abs(d) if absolute else d)
    s = sum(errs)
    return s / len(errs)


def false_alarm_rate_events(
    ref: list[EventWindow], pred: list[EventWindow], *, iou_threshold: float = 0.3
) -> float:
    """
    **Proportion of predicted events** that are **unmatched** (no reference with IoU
    ≥ threshold) — a "false discovery" rate among *predicted* event segments.
    """
    m = match_events_greedy_iou(ref, pred, iou_threshold=iou_threshold)
    tp = len(m)
    fp = len(pred) - tp
    if not pred:
        return 0.0
    return fp / len(pred)


def mean_time_to_alert_seconds(
    ref: list[EventWindow],
    alert_times_sec: list[float],
) -> float:
    """
    Mean delay from each **reference onset** to the first **alert** at or after that onset.

    For each reference window ``R`` (onset = ``min(R.start, R.end)``), finds
    ``min{ t in alert_times : t >= onset } - onset``. Windows with no such alert are
    skipped. If no usable pairs exist, returns ``0.0``.

    ``alert_times_sec`` may be unsorted; it is sorted internally.
    """
    if not ref or not alert_times_sec:
        return 0.0
    times = sorted(alert_times_sec)
    delays: list[float] = []
    for rw in ref:
        r0 = rw.start if rw.start <= rw.end else rw.end
        t_alert: float | None = None
        for t in times:
            if t >= r0:
                t_alert = t
                break
        if t_alert is not None and isfinite(t_alert) and isfinite(r0):
            delays.append(max(0.0, t_alert - r0))
    if not delays:
        return 0.0
    return sum(delays) / len(delays)


def alert_consistency(
    levels_a: list[str] | tuple[str, ...],
    levels_b: list[str] | tuple[str, ...],
) -> AlertConsistencyResult:
    """
    **Per-frame / per-timestep** exact agreement between two **aligned** label sequences
    (e.g. two runs of the HCI refinery or two threshold settings).

    Returns the share of positions where ``levels_a[i] == levels_b[i]`` (string equality,
    case-sensitive). Empty input yields rate ``0.0`` and ``n=0``.
    """
    n = min(len(levels_a), len(levels_b))
    if n == 0:
        return AlertConsistencyResult(exact_match_rate=0.0, n=0, n_agree=0)
    agree = sum(1 for i in range(n) if levels_a[i] == levels_b[i])
    return AlertConsistencyResult(
        exact_match_rate=agree / n,
        n=n,
        n_agree=agree,
    )
