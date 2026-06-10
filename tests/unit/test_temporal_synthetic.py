"""
Temporal features on synthetic biomechanical tables (isolated :mod:`fightsafe_ai.features.temporal`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tests.support.isolated import load_temporal


_tm: Any = None


def _t() -> Any:
    global _tm
    if _tm is None:
        _tm = load_temporal()
    return _tm


def test_instability_nonzero_on_oscillating_hip() -> None:
    t = _t()
    n = 20
    hip = 0.5 + 0.05 * np.sin(np.linspace(0, 4 * np.pi, n))
    df = pd.DataFrame(
        {
            "hip_vertical_position": hip,
            "head_vertical_position": hip + 0.05,
            "torso_angle_degrees": np.zeros(n),
            "is_low_posture": [False] * n,
        }
    )
    out = t.compute_temporal_features(df, fps=30, rolling_window_frames=5, min_periods=1)
    inst = out["instability_score"].to_numpy()
    assert np.nanmax(inst) > 1e-6


def test_pre_smooth_config_runs() -> None:
    t = _t()
    n = 12
    x = np.linspace(0.4, 0.55, n)
    df = pd.DataFrame(
        {
            "hip_vertical_position": x,
            "head_vertical_position": x,
            "torso_angle_degrees": np.zeros(n),
            "is_low_posture": [False] * n,
        }
    )
    sm = t.compute_temporal_features(
        df,
        fps=20,
        config=t.TemporalFeatureConfig(
            pre_smooth=True,
            pre_smooth_window_frames=3,
            rolling_window_frames=3,
        ),
    )
    assert np.isfinite(sm["hip_vertical_velocity"].to_numpy()).all()
