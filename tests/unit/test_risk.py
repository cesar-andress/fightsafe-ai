"""
Unit tests: risk in [0,1] and threshold to risk_level mapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fightsafe_ai.risk.rules import (
    ALL_RULE_NAMES,
    InterpretableAggregationConfig,
    InterpretableRiskConfig,
    load_interpretable_risk_config,
    map_score_to_risk_level,
)
from fightsafe_ai.risk.scorer import compute_interpretable_risk


def _minimal_config() -> InterpretableRiskConfig:
    return load_interpretable_risk_config(None)


def test_risk_score_bounded_zero_one_synthetic() -> None:
    """Every row with synthetic features has risk_score in [0,1]."""
    n = 12
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "frame_id": [str(i) for i in range(n)],
            "hip_vertical_velocity": rng.normal(0.0, 0.5, n),
            "head_vertical_velocity": rng.normal(0.0, 0.2, n),
            "torso_angle_degrees": rng.uniform(0, 50, n),
            "low_posture_duration_frames": rng.integers(0, 40, size=n),
            "instability_score": rng.uniform(0, 0.2, n),
            "near_ground": rng.random(n) > 0.5,
            "guard_level": rng.uniform(0, 0.1, n),
            "facing_away_score": rng.uniform(0, 0.1, n),
            "reaction_delay_score": rng.uniform(0, 0.1, n),
        }
    )
    out = compute_interpretable_risk(df, config=_minimal_config())
    scores = out["risk_score"].to_numpy(dtype=float)
    assert np.isfinite(scores).all()
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_risk_score_empty_input() -> None:
    """Empty DataFrame: no rows; risk columns present and empty, no error at boundaries."""
    out = compute_interpretable_risk(
        pd.DataFrame(),
        config=_minimal_config(),
    )
    assert len(out) == 0
    assert out["risk_score"].dtype == float or len(out["risk_score"]) == 0


def test_risk_level_threshold_boundaries() -> None:
    """map_score_to_risk_level respects level_medium_min < level_high_min < level_critical_min."""
    agg = InterpretableAggregationConfig(
        level_medium_min=0.2,
        level_high_min=0.5,
        level_critical_min=0.8,
    )
    levels = map_score_to_risk_level(
        np.array(
            [0.19, 0.2, 0.45, 0.5, 0.7, 0.8, 0.99, float("nan")],
            dtype=float,
        ),
        agg,
    )
    assert levels[0] == "LOW"
    assert levels[1] == "MEDIUM"
    assert levels[2] == "MEDIUM"
    assert levels[3] == "HIGH"
    assert levels[4] == "HIGH"
    assert levels[5] == "CRITICAL"
    assert levels[6] == "CRITICAL"
    assert levels[7] == "LOW"  # no finito


def test_invalid_aggregation_cutoffs_rejected() -> None:
    bad = InterpretableAggregationConfig(
        level_medium_min=0.5,
        level_high_min=0.5,
        level_critical_min=0.8,
    )
    with pytest.raises(ValueError, match="level_medium_min < level_high_min"):
        map_score_to_risk_level(np.array([0.5], dtype=float), bad)


def test_interpretable_risk_populates_all_rule_keys_in_components() -> None:
    """Build+score must preserve rule name columns (2-row matrix, all component columns present)."""
    base = {
        "frame_id": "0",
        "hip_vertical_velocity": 0.0,
        "head_vertical_velocity": 0.0,
        "torso_angle_degrees": 0.0,
        "low_posture_duration_frames": 0,
        "instability_score": 0.0,
        "near_ground": True,
        "guard_level": 0.0,
        "facing_away_score": 0.0,
        "reaction_delay_score": 0.0,
    }
    df = pd.DataFrame([base, {**base, "frame_id": "1", "hip_vertical_velocity": 3.0}])
    out = compute_interpretable_risk(
        df,
        config=_minimal_config(),
        include_rule_component_columns=True,
    )
    for name in ALL_RULE_NAMES:
        col = f"risk_component_{name}"
        assert col in out.columns
        v = out[col].to_numpy()
        assert (v >= 0.0).all() and (v <= 1.0).all()
