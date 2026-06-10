"""
Deterministic, in-memory data builders for tests.

Used by ``tests/test_policy_synthetic_examples.py`` to exemplify **docs/testing-policy.md**
— no network, no large videos, no random flakiness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_keypoint_long_format_tiny(hip_y: float = 0.5, frame_id: str = "0") -> pd.DataFrame:
    """One frame, two hips + ankles, enough for :func:`compute_biomechanical_features`."""
    rows: list[dict[str, str | float]] = [
        {"frame_id": frame_id, "keypoint_name": "nose", "x": 0.5, "y": 0.2},
        {"frame_id": frame_id, "keypoint_name": "left_shoulder", "x": 0.4, "y": 0.3},
        {"frame_id": frame_id, "keypoint_name": "right_shoulder", "x": 0.6, "y": 0.3},
        {"frame_id": frame_id, "keypoint_name": "left_hip", "x": 0.42, "y": hip_y},
        {"frame_id": frame_id, "keypoint_name": "right_hip", "x": 0.58, "y": hip_y},
        {"frame_id": frame_id, "keypoint_name": "left_ankle", "x": 0.44, "y": 0.92},
        {"frame_id": frame_id, "keypoint_name": "right_ankle", "x": 0.56, "y": 0.92},
    ]
    return pd.DataFrame(rows)


def make_temporal_feature_input_small(n: int = 8) -> pd.DataFrame:
    """
    Smooth, fixed hip trace + static companion columns (no RNG — fully deterministic).
    Callers pass ``fps`` to ``compute_temporal_features`` separately (e.g. 30).
    """
    if n < 3:
        raise ValueError("n must be at least 3 for a meaningful rolling window")
    hip = np.linspace(0.45, 0.55, n, dtype=float)
    return pd.DataFrame(
        {
            "hip_vertical_position": hip,
            "head_vertical_position": hip + 0.05,
            "torso_angle_degrees": np.zeros(n),
            "is_low_posture": np.zeros(n, dtype=bool),
        }
    )


def make_interpretable_risk_feature_frame() -> pd.DataFrame:
    """Rows sufficient for :func:`compute_interpretable_risk` (interpretable rules path)."""
    return pd.DataFrame(
        {
            "hip_vertical_velocity": [0.0, 2.0, 0.0],
            "head_vertical_velocity": [0.0, 0.0, 0.0],
            "torso_angle_deg": [5.0, 80.0, 5.0],
            "low_posture_duration_frames": [0.0, 20.0, 0.0],
            "instability_score": [0.0, 0.2, 0.0],
            "near_ground": [False, True, False],
            "guard_level": [0.0, 0.0, 0.0],
            "facing_away_score": [0.0, 0.0, 0.0],
            "reaction_delay_score": [0.0, 0.0, 0.0],
            "anomaly_score": [0.0, 0.0, 0.0],
        }
    )


def make_frame_risk_for_event_merge() -> pd.DataFrame:
    """
    Two separated HIGH/CRITICAL streaks; merge gap threshold decides one vs two events.
    """
    n = 5
    dt = 0.1
    t = np.arange(n, dtype=float) * dt
    levels = np.array(["HIGH", "HIGH", "LOW", "HIGH", "HIGH"], dtype=object)
    scores = np.array([0.5, 0.5, 0.0, 0.6, 0.6])
    return pd.DataFrame(
        {
            "frame_id": [str(i) for i in range(n)],
            "timestamp": t,
            "risk_score": scores,
            "risk_level": levels,
        }
    )
