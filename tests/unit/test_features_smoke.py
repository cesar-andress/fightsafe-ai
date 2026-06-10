"""Smoke tests for biomechanical features (requires a working MediaPipe + NumPy stack)."""

from pathlib import Path

import numpy as np
import pytest


def _mediapipe_imports() -> bool:
    try:
        __import__("mediapipe")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _mediapipe_imports(),
    reason="mediapipe (or its NumPy stack) not importable in this environment",
)


def test_compute_pose_features_empty_dir(tmp_path: Path) -> None:
    from fightsafe_ai.features.biomechanics import compute_pose_features

    df = compute_pose_features(tmp_path, fps=10.0)
    assert len(df) == 0
    assert "torso_angle_deg" in df.columns


def test_compute_pose_features_single_csv(tmp_path: Path) -> None:
    from fightsafe_ai.features.biomechanics import compute_pose_features

    kp = tmp_path / "frame_0001.csv"
    kp.write_text(
        "region,landmark,x,y,visibility\n"
        "torso,left_shoulder,0.4,0.3,1\n"
        "torso,right_shoulder,0.6,0.3,1\n"
        "hips,left_hip,0.42,0.55,1\n"
        "hips,right_hip,0.58,0.55,1\n"
        "legs,left_knee,0.43,0.7,1\n"
        "legs,right_knee,0.57,0.7,1\n"
        "legs,left_ankle,0.44,0.9,1\n"
        "legs,right_ankle,0.56,0.9,1\n"
        "legs,left_heel,0.44,0.92,1\n"
        "legs,right_heel,0.56,0.92,1\n"
        "legs,left_foot_index,0.45,0.93,1\n"
        "legs,right_foot_index,0.55,0.93,1\n",
        encoding="utf-8",
    )
    df = compute_pose_features(tmp_path, fps=10.0, rolling_window=2, ground_y_threshold=0.8)
    assert len(df) == 1
    assert not np.isnan(df.iloc[0]["torso_angle_deg"])
    assert df.iloc[0]["near_ground"] in (True, False)
    assert "knee_flexion_left_deg" in df.columns
    assert "anomaly_score" in df.columns
    assert "anomaly_type" in df.columns
    assert 0.0 <= float(df.iloc[0]["anomaly_score"]) <= 1.0
