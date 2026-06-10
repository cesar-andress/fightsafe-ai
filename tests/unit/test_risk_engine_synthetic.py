"""
Synthetic :func:`detect_risk_events` tests (no video I/O, isolated engine import).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.support.isolated import load_risk_engine


_eng: Any = None


def _e() -> Any:
    global _eng
    if _eng is None:
        _eng = load_risk_engine()
    return _eng


def _minimal_feature_table(
    n: int,
    *,
    near_ground: list[bool] | None = None,
) -> pd.DataFrame:
    """Columns required by the legacy risk engine."""
    if near_ground is None:
        near_ground = [False] * n
    return pd.DataFrame(
        {
            "frame_id": [str(i) for i in range(n)],
            "torso_angle_deg": np.zeros(n),
            "hip_vertical_velocity": np.zeros(n),
            "keypoint_position_variance": np.full(n, 1e-6),
            "near_ground": near_ground,
        }
    )


def test_detect_risk_events_empty() -> None:
    e = _e()
    p = e.RiskRuleParams()
    empty = _minimal_feature_table(0)
    out = e.detect_risk_events(empty, p)
    assert len(out) == 0
    assert "risk_score" in out.columns


def test_detect_risk_score_bounded_and_tilt() -> None:
    e = _e()
    p = e.RiskRuleParams(
        torso_angle_threshold_deg=10.0,
        hip_velocity_threshold=0.1,
    )
    df = _minimal_feature_table(3)
    df["torso_angle_deg"] = [0.0, 45.0, 0.0]
    df["hip_vertical_velocity"] = [0.0, 0.5, 0.0]
    out = e.detect_risk_events(df, p)
    assert len(out) == 3
    assert (out["risk_score"] >= 0).all() and (out["risk_score"] <= 1.0 + 1e-9).all()
    # Middle row: strong tilt+velocity
    assert float(out["risk_from_tilt_velocity"].iloc[1]) > 0.2


def test_ground_streak_saturates_after_min_frames() -> None:
    e = _e()
    p = e.RiskRuleParams(near_ground_min_frames=2)
    # streak 1,2,3 — row index 2 has streak 3 > 2 => full r_ground branch
    df = _minimal_feature_table(3, near_ground=[True, True, True])
    out = e.detect_risk_events(df, p)
    assert float(out["risk_from_ground"].iloc[2]) == pytest.approx(1.0, abs=1e-5)


def test_missing_columns_raises() -> None:
    e = _e()
    bad = pd.DataFrame({"torso_angle_deg": [1.0]})
    with pytest.raises(ValueError, match="Missing"):
        e.detect_risk_events(bad, None)
