"""Heuristic strike detector on COCO skeleton sequences."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from tests.support.subprocess_env import REPO_ROOT, env_with_src_pythonpath

from fightsafe_ai.evaluation.boxingvi_strike_detector import detect_strike_events


pytestmark = pytest.mark.unit


def test_detect_strike_events_wrist_spike_produces_impact() -> None:
    """Synthetic abrupt wrist motion should yield at least one HIGH strike candidate."""
    t_max, p_max = 120, 1
    sk = np.zeros((t_max, p_max, 17, 2), dtype=np.float64)
    for t in range(t_max):
        for j in range(5, 11):
            sk[t, 0, j, 0] = 0.15 + 0.002 * t
            sk[t, 0, j, 1] = 0.45
    # Single-frame large displacement on both wrists (indices 9, 10)
    sk[55, 0, 9, 0] = 4.5
    sk[55, 0, 10, 0] = 4.5

    out = detect_strike_events(
        sk,
        fps=30.0,
        percentile=80.0,
        merge_frames=6,
        min_valid_keypoints=5,
    )
    assert len(out) >= 1
    assert out[0]["category"] == "impact"
    assert out[0]["event_level"] == "HIGH"
    assert out[0]["event_type"] == "boxingvi.strike_candidate"


def test_detect_strike_events_empty_when_static() -> None:
    sk = np.zeros((10, 1, 17, 2), dtype=np.float64)
    assert detect_strike_events(sk, fps=30.0, percentile=90.0) == []


def test_runner_cli_accepts_strike_flags_and_json_has_strike_events(tmp_path: Path) -> None:
    root = tmp_path / "boxingvi"
    (root / "skeleton").mkdir(parents=True)
    t_max, p_max = 80, 1
    sk = np.zeros((t_max, p_max, 17, 2), dtype=np.float32)
    for t in range(t_max):
        for j in range(5, 11):
            sk[t, 0, j, 0] = 0.1 + 0.001 * t
            sk[t, 0, j, 1] = 0.3
    sk[40, 0, 9, 0] = 3.0
    sk[40, 0, 10, 0] = 3.0
    np.save(root / "skeleton" / "SX.npy", sk)

    out_dir = tmp_path / "out"
    r = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "fightsafe_ai.evaluation.boxingvi_skeleton_runner",
            "--dataset-root",
            str(root),
            "--video-id",
            "SX",
            "--fps",
            "30",
            "--output-dir",
            str(out_dir),
            "--enable-strike-detector",
            "--strike-percentile",
            "75",
            "--strike-merge-frames",
            "4",
        ],
        cwd=REPO_ROOT,
        env=env_with_src_pythonpath(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    js = out_dir / "boxingvi_predictions_SX.json"
    assert js.is_file()
    data = json.loads(js.read_text(encoding="utf-8"))
    assert "strike_events" in data
    assert isinstance(data["strike_events"], list)
