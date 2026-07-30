"""
Lightweight live performance metrics (stdlib only: :func:`time.perf_counter`).

Tracks:

- **FPS**: exponential moving average of the display loop rate.
- **Frame processing time**: EMA of wall time per UI iteration (decode hook → ``imshow``).
- **Inference time**: EMA of worker time inside ``pipeline.process_frame``.
- **Rendering time**: EMA of overlay + panel + ``imshow``.
- **End-to-end latency**: EMA of submit → inference completion (queue wait + inference).

Adaptive stride: when FPS EMA stays below ``fps_threshold`` (target budget), ``infer_stride``
increases so inference runs every N-th frame while the UI loop keeps full frame rate.

Budget stride (optional): ``budget_infer_stride(display_hz, inference_fps)`` yields a fixed
skip factor ``ceil(display_hz / inference_fps)`` so e.g. 30 Hz display vs 12 Hz inference
runs pose/risk every ~3rd frame while reusing the last pose on skipped frames.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass


class ExponentialMovingAverage:
    """Scalar EMA; first sample seeds the average."""

    __slots__ = ("_alpha", "_value")

    def __init__(self, *, alpha: float = 0.12) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = float(alpha)
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = float(x)
        else:
            a = self._alpha
            self._value = a * float(x) + (1.0 - a) * self._value
        return self._value


@dataclass(slots=True)
class PerformanceSnapshot:
    """
    Read-only snapshot for HUD / logging.

    Times are in milliseconds except ``fps_ema`` and ``target_fps`` (Hz).
    ``latency_ms`` is end-to-end: job enqueue → inference completion (wait + model).
    """

    fps_ema: float
    frame_processing_ms: float
    infer_ms: float
    render_ms: float
    latency_ms: float
    infer_stride: int
    below_fps_threshold: bool
    target_fps: float
    #: ``ceil(display_hz / inference_fps)`` cap when using CLI inference budget (default 1).
    stride_budget: int = 1
    #: Stride suggested by FPS lag detector before merging with ``stride_budget``.
    stride_adaptive: int = 1


class LivePerformanceMonitor:
    """
    Thread-safe monitor for the live loop + inference worker.

    Uses only :func:`time.perf_counter` at call sites (see :mod:`fightsafe_ai.live.live_runner`).
    """

    def __init__(
        self,
        *,
        fps_alpha: float = 0.1,
        time_alpha: float = 0.15,
        fps_threshold: float = 18.0,
        fps_recover: float = 22.0,
        target_fps: float | None = None,
        max_infer_stride: int = 8,
        stride_adjust_every_frames: int = 20,
    ) -> None:
        self._fps = ExponentialMovingAverage(alpha=fps_alpha)
        self._frame_ms = ExponentialMovingAverage(alpha=time_alpha)
        self._infer_ms = ExponentialMovingAverage(alpha=time_alpha)
        self._render_ms = ExponentialMovingAverage(alpha=time_alpha)
        self._latency_ms = ExponentialMovingAverage(alpha=time_alpha)

        if target_fps is not None:
            self._fps_threshold = float(target_fps)
        else:
            self._fps_threshold = float(fps_threshold)
        self._fps_recover = float(fps_recover)
        self._max_infer_stride = max(1, int(max_infer_stride))
        self._stride_adjust_every = max(1, int(stride_adjust_every_frames))

        self._infer_stride = 1
        self._loop_frames = 0
        self._lock = threading.Lock()

    @property
    def infer_stride(self) -> int:
        return self._infer_stride

    @property
    def target_fps(self) -> float:
        """FPS budget used for adaptive inference stride (same as constructor ``fps_threshold`` / ``target_fps``)."""
        return self._fps_threshold

    def tick_display_loop(self, *, dt_seconds: float) -> None:
        """Call once per displayed frame with Δt since previous frame (wall clock)."""
        dt = max(float(dt_seconds), 1e-9)
        inst_fps = 1.0 / dt
        with self._lock:
            self._fps.update(inst_fps)
            self._loop_frames += 1
            fps = self._fps.value
            if fps is None:
                return
            if self._loop_frames % self._stride_adjust_every != 0:
                return
            if fps < self._fps_threshold:
                self._infer_stride = min(self._max_infer_stride, self._infer_stride + 1)
            elif fps > self._fps_recover:
                self._infer_stride = max(1, self._infer_stride - 1)

    def record_frame_processing(self, seconds: float) -> None:
        """Main thread: one full displayed frame (read → compose → ``imshow``), wall-clock."""
        with self._lock:
            self._frame_ms.update(max(float(seconds), 0.0) * 1000.0)

    def record_inference(self, *, infer_seconds: float, queue_to_done_seconds: float) -> None:
        """Worker: pure inference time and submit→done latency."""
        with self._lock:
            self._infer_ms.update(max(infer_seconds, 0.0) * 1000.0)
            self._latency_ms.update(max(queue_to_done_seconds, 0.0) * 1000.0)

    def record_render(self, render_seconds: float) -> None:
        """UI thread: overlay + panel + imshow."""
        with self._lock:
            self._render_ms.update(max(render_seconds, 0.0) * 1000.0)

    def snapshot(self) -> PerformanceSnapshot:
        with self._lock:
            fps = self._fps.value or 0.0
            frame_ms = self._frame_ms.value or 0.0
            infer = self._infer_ms.value or 0.0
            rend = self._render_ms.value or 0.0
            lat = self._latency_ms.value or 0.0
            stride = self._infer_stride
            below = fps < self._fps_threshold and fps > 1e-6
            tgt = self._fps_threshold
        return PerformanceSnapshot(
            fps_ema=fps,
            frame_processing_ms=frame_ms,
            infer_ms=infer,
            render_ms=rend,
            latency_ms=lat,
            infer_stride=stride,
            below_fps_threshold=below,
            target_fps=tgt,
            stride_budget=1,
            stride_adaptive=stride,
        )


def budget_infer_stride(
    *,
    display_hz: float,
    inference_fps: float | None,
    max_stride: int,
) -> int:
    """
    Nominal frame skip from display vs inference rate targets (``ceil(display / infer)``).

    When ``inference_fps`` is unset, returns 1 (no budget-driven skipping).
    """
    if inference_fps is None or inference_fps <= 0.0:
        return 1
    dh = max(float(display_hz), 1e-9)
    lim = max(1, int(max_stride))
    return max(1, min(lim, math.ceil(dh / float(inference_fps))))


def merge_infer_strides(
    *,
    budget_stride: int,
    adaptive_stride: int,
    max_stride: int,
) -> int:
    """Effective stride: honor both Hz budget and adaptive lag, capped at ``max_stride``."""
    lim = max(1, int(max_stride))
    b = max(1, int(budget_stride))
    a = max(1, int(adaptive_stride))
    return min(lim, max(b, a))


__all__ = [
    "ExponentialMovingAverage",
    "LivePerformanceMonitor",
    "PerformanceSnapshot",
    "budget_infer_stride",
    "merge_infer_strides",
]
