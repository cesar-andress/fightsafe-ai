"""Tests for event matching and event-level evaluation with synthetic windows."""

from __future__ import annotations

import json
from pathlib import Path

from fightsafe_ai.evaluation.event_matching import events_json_to_windows, match_events
from fightsafe_ai.evaluation.event_metrics import EventWindow, temporal_iou
from fightsafe_ai.evaluation.metrics import (
    evaluate_event_prediction,
    event_evaluation_to_json_dict,
)


def _w(s: float, e: float, label: str = "X") -> EventWindow:
    return EventWindow(start=s, end=e, label=label)


def test_match_events_perfect_overlap() -> None:
    ref = [_w(0, 1, "A")]
    pred = [_w(0, 1, "B")]
    m = match_events(pred, ref, iou_threshold=0.1, require_same_label=False)
    assert len(m) == 1
    assert m[0].iou == 1.0
    r = evaluate_event_prediction(pred, ref, iou_threshold=0.1, require_same_label=False)
    assert r.true_positives == 1
    assert r.false_positives == 0
    assert r.false_negatives == 0
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.f1 == 1.0
    assert r.mean_onset_delay_seconds == 0.0
    d = event_evaluation_to_json_dict(r)
    assert d["true_positives"] == 1
    assert "matches" in d and len(d["matches"]) == 1


def test_match_events_false_positive() -> None:
    ref: list[EventWindow] = []
    pred = [_w(0, 1, "A")]
    r = evaluate_event_prediction(pred, ref, iou_threshold=0.1)
    assert r.true_positives == 0
    assert r.false_positives == 1
    assert r.false_negatives == 0


def test_match_events_false_negative() -> None:
    ref = [_w(10, 12, "A")]
    pred: list[EventWindow] = []
    r = evaluate_event_prediction(pred, ref, iou_threshold=0.1)
    assert r.true_positives == 0
    assert r.false_positives == 0
    assert r.false_negatives == 1


def test_tolerance_allows_slight_misalign() -> None:
    ref = [_w(0.0, 2.0, "A")]
    pred = [_w(0.2, 2.2, "A")]
    iou0 = temporal_iou(ref[0], pred[0])
    m0 = match_events(pred, ref, iou_threshold=0.2, tolerance_seconds=0.0, require_same_label=False)
    m1 = match_events(pred, ref, iou_threshold=0.2, tolerance_seconds=0.3, require_same_label=False)
    if iou0 < 0.2:
        assert len(m0) == 0
    assert len(m1) == 1


def test_require_same_label() -> None:
    ref = [_w(0, 1, "FALL"), _w(2, 3, "KO")]
    pred = [_w(0, 1, "FALL"), _w(2, 3, "FALL")]
    m = match_events(pred, ref, iou_threshold=0.1, require_same_label=True)
    assert len(m) == 1
    m2 = match_events(pred, ref, iou_threshold=0.1, require_same_label=False)
    assert len(m2) == 2


def test_events_json_to_windows(tmp_path: Path) -> None:
    p = tmp_path / "e.json"
    p.write_text(
        json.dumps(
            [
                {
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "event_level": "HIGH",
                }
            ]
        ),
        encoding="utf-8",
    )
    w = events_json_to_windows(p)
    assert len(w) == 1
    assert w[0].start == 1.0
    assert w[0].label == "HIGH"
