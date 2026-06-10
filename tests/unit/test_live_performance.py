"""Unit tests for live performance monitoring."""

from __future__ import annotations

import pytest

from fightsafe_ai.live.performance import (
    ExponentialMovingAverage,
    LivePerformanceMonitor,
    budget_infer_stride,
    merge_infer_strides,
)


pytestmark = pytest.mark.unit


def test_ema_first_value_seeds() -> None:
    ema = ExponentialMovingAverage(alpha=0.5)
    assert ema.update(10.0) == pytest.approx(10.0)
    assert ema.update(20.0) == pytest.approx(15.0)


def test_monitor_fps_and_infer_stride_ramps_when_slow() -> None:
    mon = LivePerformanceMonitor(
        fps_threshold=100.0,
        fps_recover=200.0,
        max_infer_stride=4,
        stride_adjust_every_frames=1,
    )
    for _ in range(5):
        mon.tick_display_loop(dt_seconds=1.0 / 30.0)
    assert mon.infer_stride >= 2


def test_monitor_records_inference_and_latency() -> None:
    mon = LivePerformanceMonitor(fps_threshold=5.0, fps_recover=60.0)
    mon.record_inference(infer_seconds=0.02, queue_to_done_seconds=0.05)
    mon.tick_display_loop(dt_seconds=1.0 / 60.0)
    s = mon.snapshot()
    assert s.infer_ms > 15.0
    assert s.latency_ms > 40.0


def test_monitor_render() -> None:
    mon = LivePerformanceMonitor(fps_threshold=5.0, fps_recover=60.0)
    mon.record_render(0.008)
    assert mon.snapshot().render_ms > 5.0


def test_monitor_frame_processing_and_target_fps() -> None:
    mon = LivePerformanceMonitor(target_fps=24.0, fps_threshold=99.0, fps_recover=60.0)
    mon.record_frame_processing(0.016)
    s = mon.snapshot()
    assert s.frame_processing_ms > 10.0
    assert s.target_fps == pytest.approx(24.0)
    assert s.stride_adaptive == s.infer_stride


def test_budget_infer_stride_and_merge() -> None:
    assert budget_infer_stride(display_hz=30.0, inference_fps=12.0, max_stride=8) == 3
    assert budget_infer_stride(display_hz=30.0, inference_fps=None, max_stride=8) == 1
    assert merge_infer_strides(budget_stride=2, adaptive_stride=1, max_stride=8) == 2
    assert merge_infer_strides(budget_stride=1, adaptive_stride=4, max_stride=8) == 4
    assert merge_infer_strides(budget_stride=10, adaptive_stride=2, max_stride=8) == 8
