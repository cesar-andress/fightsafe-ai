"""Synthetic tests for ``fightsafe_ai.evaluation`` metrics and ablation presets."""

from __future__ import annotations

import pytest

from fightsafe_ai.evaluation.ablation import (
    AblationScenario,
    ablation_param_template,
    all_ablation_scenarios,
    make_ablation_row,
    sort_rows_by_metric,
)
from fightsafe_ai.evaluation.event_metrics import (
    EventWindow,
    alert_consistency,
    event_level_metrics,
    false_alarm_rate_events,
    match_events_greedy_iou,
    mean_absolute_onset_error,
    mean_time_to_alert_seconds,
    temporal_iou,
)
from fightsafe_ai.evaluation.metrics import (
    false_alarm_rate_frames,
    frame_level_f1,
    frame_level_precision,
    frame_level_recall,
    pose_coverage,
    precision_recall_f1,
)


pytestmark = pytest.mark.unit


def test_frame_level_perfect() -> None:
    y = [0, 1, 1, 0, 1]
    assert frame_level_precision(y, y) == 1.0
    assert frame_level_recall(y, y) == 1.0
    assert frame_level_f1(y, y) == 1.0


def test_frame_level_partial() -> None:
    yt = [0, 1, 1, 0]
    yp = [0, 1, 0, 0]  # one FN
    p, r, f1 = precision_recall_f1(yt, yp, positive_label=1)
    assert abs(frame_level_precision(yt, yp) - p) < 1e-9
    assert abs(frame_level_recall(yt, yp) - r) < 1e-9
    assert abs(frame_level_f1(yt, yp) - f1) < 1e-9
    assert p == 1.0
    assert r == 0.5
    assert abs(f1 - 2.0 / 3.0) < 1e-9


def test_false_alarm_rate_frames() -> None:
    # 3 GT negatives; 1 FP
    yt = [0, 0, 0, 1]
    yp = [0, 1, 0, 1]
    assert false_alarm_rate_frames(yt, yp) == pytest.approx(1.0 / 3.0)


def test_pose_coverage() -> None:
    assert pose_coverage([True, True, False]) == pytest.approx(2.0 / 3.0)
    assert pose_coverage([]) == 0.0


def test_temporal_iou_overlap() -> None:
    a = EventWindow(0.0, 2.0)
    b = EventWindow(1.0, 3.0)
    assert temporal_iou(a, b) == pytest.approx(1.0 / 3.0)


def test_match_events_greedy() -> None:
    ref = [EventWindow(0, 10)]
    pred = [EventWindow(0, 10), EventWindow(20, 30)]
    m = match_events_greedy_iou(ref, pred, iou_threshold=0.5)
    assert len(m) == 1
    em = event_level_metrics(ref, pred, iou_threshold=0.5)
    assert em.true_positives == 1
    assert em.n_ref == 1
    assert em.n_pred == 2
    assert em.precision == 0.5
    assert em.recall == 1.0


def test_mean_absolute_onset_error() -> None:
    ref = [EventWindow(0, 5)]
    pred = [EventWindow(2, 7)]
    m = match_events_greedy_iou(ref, pred, iou_threshold=0.1)
    assert m
    err = mean_absolute_onset_error(m, absolute=True)
    assert err == pytest.approx(2.0)


def test_false_alarm_rate_events() -> None:
    ref = [EventWindow(0, 1)]
    pred = [EventWindow(0, 1), EventWindow(10, 11)]
    assert false_alarm_rate_events(ref, pred, iou_threshold=0.3) == pytest.approx(0.5)


def test_mean_time_to_alert_seconds() -> None:
    ref = [EventWindow(10.0, 20.0), EventWindow(30.0, 40.0)]
    alerts = [10.0, 31.0]
    mtt = mean_time_to_alert_seconds(ref, alerts)
    assert mtt == pytest.approx(0.5)  # 0 and 1 s delays


def test_alert_consistency() -> None:
    r = alert_consistency(["INFO", "WATCH", "WATCH"], ["INFO", "INFO", "WATCH"])
    assert r.n == 3
    assert r.n_agree == 2
    assert r.exact_match_rate == pytest.approx(2.0 / 3.0)


def test_ablation_scenarios_order_and_templates() -> None:
    assert len(all_ablation_scenarios()) == 5
    b0 = ablation_param_template(AblationScenario.BASELINE_BIOMECHANICS_ONLY)
    assert b0.get("use_action_layer") is False
    assert b0.get("use_anomaly_layer") is False
    full = ablation_param_template(AblationScenario.FULL_FUSION)
    assert full["use_action_layer"] and full["use_anomaly_layer"] and full["use_risk_fusion"]
    llm = ablation_param_template(AblationScenario.FULL_FUSION_LLM)
    assert llm["use_llm_explanation"] is True


def test_make_ablation_row_and_sort() -> None:
    r1 = make_ablation_row(AblationScenario.FULL_FUSION, metrics={"f1": 0.4})
    r2 = make_ablation_row(AblationScenario.BASELINE_BIOMECHANICS_ONLY, metrics={"f1": 0.9})
    s = sort_rows_by_metric([r1, r2], "f1")
    assert s[0].metrics["f1"] == 0.9
