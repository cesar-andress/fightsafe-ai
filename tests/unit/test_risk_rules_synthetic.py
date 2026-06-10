"""
Interpretable risk rule components and level mapping (isolated from package ``__init__``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.support.isolated import load_risk_rules, load_risk_scorer


_rules: Any = None
_scorer: Any = None


def _R() -> tuple[Any, Any]:
    global _rules, _scorer
    if _rules is None:
        _rules = load_risk_rules()
    if _scorer is None:
        _scorer = load_risk_scorer()
    return _rules, _scorer


def test_component_monotonic_downward() -> None:
    r, _ = _R()
    cfg = r.FastDownwardConfig(velocity_threshold=0.2, velocity_scale=0.5)
    v = np.array([0.0, 0.3, 0.6], dtype=float)
    c0 = r.component_fast_downward(v, cfg)
    assert c0[0] < c0[1] < c0[2]
    assert (c0 <= 1.0).all() and (c0 >= 0).all()


def test_renormalize_weights() -> None:
    r, _ = _R()
    names = r.ALL_RULE_NAMES
    w = dict.fromkeys(names, 0.2)
    active = {n: (n == r.RULE_FAST_DOWNWARD) for n in names}
    wn = r._renormalize_weights(w, active)
    assert wn[r.RULE_FAST_DOWNWARD] == pytest.approx(1.0, abs=1e-5)
    assert sum(wn[n] for n in names) == pytest.approx(1.0, abs=1e-5)


def test_map_score_to_risk_level() -> None:
    r, _ = _R()
    agg = r.InterpretableAggregationConfig(
        level_medium_min=0.2, level_high_min=0.5, level_critical_min=0.8
    )
    s = np.array([0.0, 0.3, 0.6, 0.9], dtype=float)
    levels = r.map_score_to_risk_level(s, agg)
    assert (
        levels[0] == "LOW"
        and levels[1] == "MEDIUM"
        and levels[2] == "HIGH"
        and levels[3] == "CRITICAL"
    )


def test_scorer_mvp_deterministic() -> None:
    r, s = _R()
    cfg = r.InterpretableRiskConfig.default()
    df = pd.DataFrame(
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
            "strike_incoming_proxy": [0.0, 0.0, 0.0],
        }
    )
    out = s.compute_interpretable_risk(df, config=cfg)
    assert len(out) == 3
    assert (out["risk_score"] >= 0).all() and (out["risk_score"] <= 1.0 + 1e-9).all()
    m = out["triggered_rules"].iloc[1]
    assert isinstance(m, list) and len(m) >= 1
