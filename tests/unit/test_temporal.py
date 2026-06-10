"""Unit tests for temporal feature computation (isolated import)."""

import numpy as np
import pandas as pd
import pytest
from tests.support.isolated import load_temporal


_temporal = load_temporal()
TemporalFeatureConfig = _temporal.TemporalFeatureConfig
compute_temporal_features = _temporal.compute_temporal_features


def _bio_df(n: int) -> pd.DataFrame:
    """Synthetic biomechanical table (one row per frame)."""
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "hip_vertical_position": 0.1 * t,
            "head_vertical_position": 0.1 * t + 0.05,
            "torso_angle_degrees": np.zeros(n),
            "is_low_posture": t >= 2,
        }
    )


def test_velocities_linear_hip_at_fps_10() -> None:
    df = _bio_df(5)
    out = compute_temporal_features(df, fps=10, rolling_window_frames=2)
    # Linear ramp: y = 0.1 * t_frame (0..4); dy/dt = 0.1 / (1/fps) wait
    # frame index 0,1,2,3,4 with hip = 0,0.1,0.2,0.3,0.4 — delta per frame 0.1, dt=0.1s -> 1.0 /s
    assert np.allclose(out["hip_vertical_velocity"].to_numpy(), np.full(5, 1.0), rtol=1e-5)
    assert np.allclose(out["head_vertical_velocity"].to_numpy(), np.full(5, 1.0), rtol=1e-5)
    assert np.allclose(out["torso_angle_velocity"].to_numpy(), 0.0, atol=1e-9)


def test_instability_constant_series_near_zero() -> None:
    df = pd.DataFrame(
        {
            "hip_vertical_position": np.ones(5) * 0.3,
            "head_vertical_position": np.ones(5) * 0.25,
            "torso_angle_degrees": np.full(5, 10.0),
            "is_low_posture": [False] * 5,
        }
    )
    out = compute_temporal_features(df, fps=10, rolling_window_frames=3, min_periods=1)
    # Pandas can yield NaN on the first row of rolling std; interior windows are ~0.
    s = out["instability_score"].to_numpy()
    s_valid = s[np.isfinite(s)]
    assert (np.abs(s_valid) < 1e-9).all()


def test_low_posture_rolling_sum() -> None:
    df = pd.DataFrame(
        {
            "hip_vertical_position": np.ones(5) * 0.3,
            "head_vertical_position": np.ones(5) * 0.25,
            "torso_angle_degrees": np.zeros(5),
            "is_low_posture": [True, True, True, False, False],
        }
    )
    out = compute_temporal_features(df, fps=10, rolling_window_frames=3, min_periods=1)
    # rolling sum: last full window [True,True,True] = 3 at index 2
    assert int(out["low_posture_duration_frames"].iloc[2]) == 3
    # index 3: T,T,F
    assert int(out["low_posture_duration_frames"].iloc[3]) == 2


def test_config_overrides() -> None:
    with pytest.raises(ValueError):
        TemporalFeatureConfig(rolling_window_frames=0)

    with pytest.raises(ValueError):
        compute_temporal_features(_bio_df(1), fps=0)

    cfg = TemporalFeatureConfig(rolling_window_frames=7)
    df = _bio_df(8)
    narrow = compute_temporal_features(df, fps=10, config=cfg, rolling_window_frames=2)
    wide = compute_temporal_features(df, fps=10, rolling_window_frames=7)
    # Wider vs narrower rolling std on a linear ramp differ (explicit arg overrides config).
    assert not np.allclose(
        narrow["instability_score"].to_numpy(dtype=float),
        wide["instability_score"].to_numpy(dtype=float),
        rtol=1e-3,
        atol=1e-3,
        equal_nan=True,
    )


def test_empty_input_columns_present() -> None:
    out = compute_temporal_features(pd.DataFrame(), fps=10, rolling_window_frames=1)
    assert "hip_vertical_velocity" in out.columns
    assert "reaction_delay_score" in out.columns
    assert len(out) == 0


def test_missing_column_raises() -> None:
    df = _bio_df(3)
    bad = df.drop(columns=["is_low_posture"])
    with pytest.raises(ValueError, match="is_low_posture"):
        compute_temporal_features(bad, fps=10)


def test_critical_vertical_velocity_constant_ramp() -> None:
    """Linear ramp: hip y = a * frame_index; vertical speed ~ a * fps (per second)."""
    n, fps, a = 6, 10, 0.2
    t = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "hip_vertical_position": a * t,
            "head_vertical_position": a * t,
            "torso_angle_degrees": np.zeros(n),
            "is_low_posture": [False] * n,
        }
    )
    out = compute_temporal_features(df, fps=fps, rolling_window_frames=2, min_periods=1)
    v = out["hip_vertical_velocity"].to_numpy()
    # numpy.gradient on uniform series ~ constant interior
    assert np.isfinite(v).all()
    assert np.allclose(v, a * float(fps), rtol=0.01, atol=0.01)


def test_critical_instability_score_oscillating_hip() -> None:
    """Alternating hip y: rolling std on core axis (head+torso) is > 0 (nontrivial instability)."""
    n = 20
    # Binary alternation 0.3/0.7 on hip y; head fixed; torso 0
    y_hip = np.array([0.3 if i % 2 == 0 else 0.7 for i in range(n)], dtype=float)
    df = pd.DataFrame(
        {
            "hip_vertical_position": y_hip,
            "head_vertical_position": np.full(n, 0.25),
            "torso_angle_degrees": np.zeros(n),
            "is_low_posture": [False] * n,
        }
    )
    out = compute_temporal_features(df, fps=10, rolling_window_frames=4, min_periods=1)
    inst = out["instability_score"].to_numpy()
    i_ok = int(np.argwhere(np.isfinite(inst) & (inst > 0)).min())
    assert inst[i_ok] > 0.01
