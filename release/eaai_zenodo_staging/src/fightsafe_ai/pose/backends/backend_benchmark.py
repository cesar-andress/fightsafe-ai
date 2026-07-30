"""
Timing and quality summary for a pose :class:`~fightsafe_ai.pose.backends.base.BasePoseEstimator`
on a directory of frame images (benchmark / regression aid).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fightsafe_ai.pose.backends.base import BasePoseEstimator
from fightsafe_ai.utils.sorting import natural_sort_paths


@dataclass
class PoseBenchmarkResult:
    """Aggregated run over a list of still images (folder of frames)."""

    backend_name: str
    device: str
    n_frames: int
    n_frames_read: int
    wall_time_sec: float
    frames_per_sec: float
    pose_coverage: float
    """Share of loaded frames for which the backend returned at least one keypoint."""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "backend": self.backend_name,
            "device": self.device,
            "n_frames": self.n_frames,
            "n_frames_read": self.n_frames_read,
            "wall_time_sec": round(self.wall_time_sec, 4),
            "fps": round(self.frames_per_sec, 2),
            "pose_coverage": round(self.pose_coverage, 4),
        }
        d.update(self.extra)
        return d


def _backend_label(estimator: BasePoseEstimator) -> str:
    n = getattr(estimator, "backend_name", None)
    if isinstance(n, str) and n.strip():
        return n.strip()
    return estimator.__class__.__name__


def _device_label(estimator: BasePoseEstimator) -> str:
    d = getattr(estimator, "device_label", None)
    if isinstance(d, str) and d.strip():
        return d.strip()
    if estimator.__class__.__name__.startswith("MediaPipe"):
        return "CPU (MediaPipe Tasks; GPU optional via platform delegates)"
    return "unknown"


def _frame_globs(patterns: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not patterns:
        return ("*.jpg", "*.jpeg", "*.png")
    return tuple(str(p) for p in patterns if p and str(p).strip())


def run_pose_backend_benchmark(
    estimator: BasePoseEstimator,
    frames_dir: Path,
    *,
    glob_patterns: tuple[str, ...] | list[str] | None = None,
) -> PoseBenchmarkResult:
    """
    Time per-frame :meth:`~fightsafe_ai.pose.backends.base.BasePoseEstimator.estimate_frame`
    on every image under ``frames_dir`` (read with OpenCV; benchmark uses **in-memory** arrays).

    ``pose_coverage`` = fraction of successfully read frames where
    ``len(estimate_frame(...).keypoints) > 0``.
    """
    frames_dir = frames_dir.expanduser().resolve()
    pats = _frame_globs(glob_patterns)
    paths: list[Path] = []
    for pat in pats:
        paths.extend(frames_dir.glob(pat))
    paths = natural_sort_paths([p for p in paths if p.is_file()])

    t0 = time.perf_counter()
    n_read = 0
    n_with_pose = 0
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        n_read += 1
        pr = estimator.estimate_frame(np.asarray(im))
        if pr.keypoints:
            n_with_pose += 1
    elapsed = time.perf_counter() - t0
    back = _backend_label(estimator)
    dev = _device_label(estimator)
    n = n_read
    cov = (float(n_with_pose) / float(n)) if n else 0.0
    fps = float(n) / float(elapsed) if elapsed > 0 else 0.0
    return PoseBenchmarkResult(
        backend_name=back,
        device=dev,
        n_frames=len(paths),
        n_frames_read=n_read,
        wall_time_sec=elapsed,
        frames_per_sec=fps,
        pose_coverage=cov,
        extra={},
    )


__all__ = [
    "PoseBenchmarkResult",
    "run_pose_backend_benchmark",
]
