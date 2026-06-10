"""Benchmark helper tests (MockPoseBackend only; no GPU stack)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fightsafe_ai.pose.backends.backend_benchmark import run_pose_backend_benchmark
from fightsafe_ai.pose.backends.mock_backend import MockPoseBackend


pytestmark = pytest.mark.unit


def test_run_pose_backend_benchmark_on_mock(tmp_path: Path) -> None:
    for i in range(3):
        p = tmp_path / f"frame_{i:04d}.jpg"
        bgr = np.zeros((32, 48, 3), dtype=np.uint8)
        bgr[8:24, 8:32] = (0, 255, 0)
        cv2.imwrite(str(p), bgr)
    m = MockPoseBackend(return_empty=False)
    r = run_pose_backend_benchmark(m, tmp_path)
    assert r.n_frames == 3
    assert r.n_frames_read == 3
    assert r.pose_coverage == 1.0
    assert r.wall_time_sec > 0
    assert r.backend_name == "mock"
    assert "n/a" in r.device.lower() or "mock" in r.device.lower()
    d = r.as_dict()
    assert d["backend"] == "mock"
    assert "fps" in d


def test_empty_folder_benchmark(tmp_path: Path) -> None:
    m = MockPoseBackend(return_empty=False)
    r = run_pose_backend_benchmark(m, tmp_path)
    assert r.n_frames == 0
    assert r.pose_coverage == 0.0
