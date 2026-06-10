"""Heuristic limb anomaly columns (non-clinical MVP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fightsafe_ai.features.anomaly import (
    COL_ANOMALY_SCORE,
    COL_ANOMALY_TYPE,
    add_limb_anomaly_columns,
)
from fightsafe_ai.risk.limb_tier import LIMB_TIER_CRITICAL, LIMB_TIER_HIGH
from fightsafe_ai.risk.rules import (
    RULE_LIMB_ANOMALY,
    InterpretableRiskConfig,
    load_interpretable_risk_config,
)
from fightsafe_ai.risk.scorer import compute_interpretable_risk


def test_add_limb_anomaly_columns_missing_inputs_zero() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    out = add_limb_anomaly_columns(df, fps=30.0)
    assert (out[COL_ANOMALY_SCORE] == 0.0).all()
    assert (out[COL_ANOMALY_TYPE] == "none").all()


def test_add_limb_anomaly_extreme_knee_increases_score() -> None:
    n = 3
    # Very different L/R flexion -> asymmetry; one leg "deep" bend
    deg_l = np.array([5.0, 120.0, 5.0], dtype=float)
    deg_r = np.array([5.0, 5.0, 5.0], dtype=float)
    ay = np.full(n, 0.9, dtype=float)
    df = pd.DataFrame(
        {
            "knee_flexion_left_deg": deg_l,
            "knee_flexion_right_deg": deg_r,
            "ankle_y_min": ay,
        }
    )
    out = add_limb_anomaly_columns(df, fps=30.0)
    s = out[COL_ANOMALY_SCORE].to_numpy()
    assert (s >= 0.0).all() and (s <= 1.0).all()
    assert float(s[1]) > float(s[0])


def test_tier_override_pushes_to_high() -> None:
    cfg = load_interpretable_risk_config(None)
    df = pd.DataFrame(
        {
            "hip_vertical_velocity": [0.0],
            "head_vertical_velocity": [0.0],
            "torso_angle_deg": [5.0],
            "low_posture_duration_frames": [0.0],
            "instability_score": [0.0],
            "near_ground": [False],
            "guard_level": [0.0],
            "facing_away_score": [0.0],
            "reaction_delay_score": [0.0],
            "anomaly_score": [LIMB_TIER_HIGH + 0.01],
        }
    )
    out = compute_interpretable_risk(df, config=cfg)
    assert out["risk_level"].iloc[0] in ("HIGH", "CRITICAL")
    tr = out["triggered_rules"].iloc[0]
    assert isinstance(tr, list) and RULE_LIMB_ANOMALY in tr


def test_tier_override_critical() -> None:
    cfg = load_interpretable_risk_config(None)
    df = pd.DataFrame(
        {
            "hip_vertical_velocity": [0.0],
            "head_vertical_velocity": [0.0],
            "torso_angle_deg": [5.0],
            "low_posture_duration_frames": [0.0],
            "instability_score": [0.0],
            "near_ground": [False],
            "guard_level": [0.0],
            "facing_away_score": [0.0],
            "reaction_delay_score": [0.0],
            "anomaly_score": [LIMB_TIER_CRITICAL + 0.01],
        }
    )
    out = compute_interpretable_risk(df, config=cfg)
    assert out["risk_level"].iloc[0] == "CRITICAL"
    assert out["triggered_rules"].iloc[0].count(RULE_LIMB_ANOMALY) == 1


def test_interpretable_config_default_includes_limb() -> None:
    cfg = InterpretableRiskConfig.default()
    assert hasattr(cfg, "limb_anomaly")
    a = cfg.aggregation
    assert a.weight_limb_anomaly == pytest.approx(0.08, abs=1e-4)
    w_sum = sum(
        (
            a.weight_fast_downward,
            a.weight_large_torso,
            a.weight_low_posture,
            a.weight_instability,
            a.weight_post_fall,
            a.weight_low_guard,
            a.weight_facing_away,
            a.weight_reaction_delay,
            a.weight_loss_of_control,
            a.weight_clear_danger_fall,
            a.weight_intervention_urgent,
            a.weight_high_risk_guard_strike,
            a.weight_limb_anomaly,
        )
    )
    assert w_sum == pytest.approx(1.0, abs=1e-5)
