"""
Temporal **matching** of predicted event intervals to ground-truth reference intervals.

Builds on :class:`~fightsafe_ai.evaluation.event_metrics.EventWindow` and greedy IoU
assignment. For dilation / tolerance, intervals are **symmetrically expanded** before
overlap is tested (so small clock shifts still allow a match).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fightsafe_ai.annotation.loader import load_annotation_file
from fightsafe_ai.annotation.validator import (
    validate_annotation_file as validate_annotation_file_path,
)
from fightsafe_ai.evaluation.event_metrics import (
    EventMatch,
    EventWindow,
    match_events_greedy_iou,
    temporal_iou,
)


def _norm_label(s: str) -> str:
    t = str(s).strip()
    t = re.sub(r"\s+", " ", t)
    return t.upper() if t else "UNKNOWN"


def _dilate(w: EventWindow, half_pad: float) -> EventWindow:
    if half_pad <= 0.0:
        return w
    a0, a1 = (w.start, w.end) if w.start <= w.end else (w.end, w.start)
    s = max(0.0, a0 - half_pad)
    e = a1 + half_pad
    return EventWindow(start=s, end=e, label=w.label)


def _greedy_match_with_dilated(
    ref: list[EventWindow],
    pred: list[EventWindow],
    *,
    iou_threshold: float,
    tolerance_seconds: float,
) -> list[EventMatch]:
    """
    Like :func:`match_events_greedy_iou` but uses dilated windows for the IoU
    *eligibility* score while returning original windows in each :class:`EventMatch`.
    """
    h = 0.5 * float(tolerance_seconds)
    r_eff = [_dilate(r, h) for r in ref]
    p_eff = [_dilate(p, h) for p in pred]
    pairs: list[tuple[float, int, int]] = []
    for i, reff in enumerate(r_eff):
        for j, peff in enumerate(p_eff):
            iou_d = temporal_iou(reff, peff)
            if iou_d >= iou_threshold:
                pairs.append((iou_d, i, j))
    pairs.sort(key=lambda x: -x[0])
    used_r: set[int] = set()
    used_p: set[int] = set()
    matches: list[EventMatch] = []
    for _iou_d, i, j in pairs:
        if i in used_r or j in used_p:
            continue
        t_raw = temporal_iou(ref[i], pred[j])
        matches.append(EventMatch(ref=ref[i], pred=pred[j], iou=t_raw))
        used_r.add(i)
        used_p.add(j)
    return matches


def match_events(
    predicted: list[EventWindow],
    ground_truth: list[EventWindow],
    *,
    iou_threshold: float = 0.1,
    tolerance_seconds: float = 0.0,
    require_same_label: bool = False,
) -> list[EventMatch]:
    """
    Greedily match each prediction to a reference, maximizing **temporal IoU** (with
    optional pre-dilation) subject to a minimum IoU and optional label agreement.

    Parameters
    ----------
    predicted
        Detected / pipeline segments.
    ground_truth
        Human reference segments (e.g. from an annotation file).
    iou_threshold
        Minimum IoU (0–1) on **dilated** or raw intervals, depending on ``tolerance_seconds``.
    tolerance_seconds
        If > 0, each interval is expanded by ``tolerance_seconds / 2`` on each end before
        IoU is used for *pair* eligibility. Original intervals are used in the returned
        :class:`EventMatch` and for the stored ``iou`` (raw, undilated).
    require_same_label
        If true, only pairs with the same ``EventWindow.label`` (case-insensitive) are
        considered. **Note:** default pipeline ``events.json`` often uses
        ``event_level`` (e.g. HIGH) while hand labels use :class:`event_type` (e.g. KO);
        leave ``False`` unless labels are aligned.
    """
    ref, pred = list(ground_truth), list(predicted)
    if require_same_label:
        all_labels: set[str] = set()
        for w in ref + pred:
            all_labels.add(_norm_label(w.label))
        out: list[EventMatch] = []
        for lab in sorted(all_labels):
            sub_r = [w for w in ref if _norm_label(w.label) == lab]
            sub_p = [w for w in pred if _norm_label(w.label) == lab]
            if tolerance_seconds and tolerance_seconds > 0.0:
                out.extend(
                    _greedy_match_with_dilated(
                        sub_r,
                        sub_p,
                        iou_threshold=iou_threshold,
                        tolerance_seconds=tolerance_seconds,
                    )
                )
            else:
                out.extend(match_events_greedy_iou(sub_r, sub_p, iou_threshold=iou_threshold))
        return out
    if tolerance_seconds and tolerance_seconds > 0.0:
        return _greedy_match_with_dilated(
            ref, pred, iou_threshold=iou_threshold, tolerance_seconds=tolerance_seconds
        )
    return match_events_greedy_iou(ref, pred, iou_threshold=iou_threshold)


# --- File adapters (JSON -> EventWindow) ----------------------------------------


def _as_float(x: Any) -> float:
    return float(x)


def events_json_to_windows(path: Path) -> list[EventWindow]:
    """
    Load a FightSafe run ``events.json`` (list of event dicts) and build :class:`EventWindow`
    rows. Uses ``start_time`` / ``end_time``; label = ``event_type`` if set, else
    ``event_level``, else ``\"UNKNOWN\"``.
    """
    p = path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("events.json must be a JSON array of events.")
    out: list[EventWindow] = []
    for o in raw:
        if not isinstance(o, dict):
            continue
        t0 = o.get("start_time", o.get("startTime"))
        t1 = o.get("end_time", o.get("endTime"))
        if t0 is None or t1 is None:
            continue
        lab = (
            o.get("event_type") or o.get("eventType") or o.get("event_level") or o.get("eventLevel")
        )
        lab_s = str(lab).strip() if lab is not None else "UNKNOWN"
        out.append(
            EventWindow(
                start=_as_float(t0),
                end=_as_float(t1),
                label=lab_s,
            )
        )
    return out


def annotation_file_to_ground_truth_windows(path: Path) -> list[EventWindow]:
    """Build reference :class:`EventWindow` from a validated :mod:`fightsafe_ai.annotation` file."""
    p = path.expanduser().resolve()
    errs = validate_annotation_file_path(p)
    if errs:
        msg = "; ".join(errs[:5])
        raise ValueError(f"Invalid annotation file {path}: {msg}")
    doc = load_annotation_file(p)
    return [
        EventWindow(
            start=float(e.start_time),
            end=float(e.end_time),
            label=str(e.event_type.value),
        )
        for e in doc.events
    ]


__all__ = [
    "annotation_file_to_ground_truth_windows",
    "events_json_to_windows",
    "match_events",
    "validate_annotation_file_path",
]
