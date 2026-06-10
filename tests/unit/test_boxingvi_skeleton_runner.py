"""Skeleton-only BoxingVI evaluation runner (no RGB)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tests.support.subprocess_env import REPO_ROOT, env_with_src_pythonpath

from fightsafe_ai.evaluation.boxingvi_skeleton_runner import (
    build_landmark_sequence_from_skeleton,
    run_boxingvi_skeleton_evaluation,
)


pytestmark = pytest.mark.unit


def _valid_coco_skeleton(num_frames: int = 12) -> np.ndarray:
    """Enough spatial extent + joints for biomechanical path."""
    sk = np.zeros((num_frames, 17, 2), dtype=np.float32)
    for t in range(num_frames):
        for j in range(17):
            sk[t, j, 0] = 0.15 + 0.005 * t + 0.03 * j
            sk[t, j, 1] = 0.25 + 0.02 * j
    return sk


def test_build_landmark_sequence_skips_all_zero() -> None:
    arr = np.zeros((3, 17, 2), dtype=np.float32)
    frames, per = build_landmark_sequence_from_skeleton(arr, min_valid_keypoints=4)
    assert len(frames) == 3
    assert all(f[1] is None for f in frames)
    assert all(p is None for p in per)


def test_run_boxingvi_skeleton_writes_outputs(tmp_path: Path) -> None:
    root = tmp_path / "boxingvi"
    (root / "skeleton").mkdir(parents=True)
    np.save(root / "skeleton" / "VT.npy", _valid_coco_skeleton(15))

    out_dir = tmp_path / "evaluation_out"
    summary = run_boxingvi_skeleton_evaluation(
        dataset_root=root,
        video_id="VT",
        fps=30.0,
        output_dir=out_dir,
    )

    assert summary["n_frames_risk"] >= 1
    json_path = out_dir / "boxingvi_predictions_VT.json"
    csv_path = out_dir / "boxingvi_predictions_VT.csv"
    assert json_path.is_file()
    assert csv_path.is_file()
    assert summary["outputs"]["json"] == str(json_path)


def test_cli_module_runs(tmp_path: Path) -> None:
    import subprocess
    import sys

    root = tmp_path / "boxingvi"
    (root / "skeleton").mkdir(parents=True)
    np.save(root / "skeleton" / "V1.npy", _valid_coco_skeleton(8))

    r = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "fightsafe_ai.evaluation.boxingvi_skeleton_runner",
            "--dataset-root",
            str(root),
            "--video-id",
            "V1",
            "--fps",
            "30",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=REPO_ROOT,
        env=env_with_src_pythonpath(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert (tmp_path / "out" / "boxingvi_predictions_V1.csv").is_file()
