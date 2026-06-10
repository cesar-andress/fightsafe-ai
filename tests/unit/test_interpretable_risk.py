"""
Interpretable MVP risk scorer (rules + YAML).

Loads submodules with :mod:`importlib` so tests do not import the top-level
``fightsafe_ai`` package (which pulls optional heavy dependencies).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# tests/unit -> repo root
_SRC = Path(__file__).resolve().parents[2] / "src"
_ROOT = Path(__file__).resolve().parents[2]


def _ensure_pkg(name: str, path: str) -> None:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [path]
        sys.modules[name] = m


def _load_risk_for_tests() -> tuple[Any, Any]:
    _ensure_pkg("fightsafe_ai", str(_SRC / "fightsafe_ai"))
    _ensure_pkg("fightsafe_ai.risk", str(_SRC / "fightsafe_ai" / "risk"))
    _ensure_pkg("fightsafe_ai.utils", str(_SRC / "fightsafe_ai" / "utils"))

    us_spec = importlib.util.spec_from_file_location(
        "fightsafe_ai.utils.sorting",
        _SRC / "fightsafe_ai" / "utils" / "sorting.py",
    )
    assert us_spec and us_spec.loader
    us_mod = importlib.util.module_from_spec(us_spec)
    sys.modules["fightsafe_ai.utils.sorting"] = us_mod
    us_spec.loader.exec_module(us_mod)

    if "fightsafe_ai.exceptions" not in sys.modules:
        ex_spec = importlib.util.spec_from_file_location(
            "fightsafe_ai.exceptions",
            _SRC / "fightsafe_ai" / "exceptions.py",
        )
        assert ex_spec and ex_spec.loader
        ex_mod = importlib.util.module_from_spec(ex_spec)
        sys.modules["fightsafe_ai.exceptions"] = ex_mod
        ex_spec.loader.exec_module(ex_mod)

    r_spec = importlib.util.spec_from_file_location(
        "fightsafe_ai.risk.rules",
        _SRC / "fightsafe_ai" / "risk" / "rules.py",
    )
    assert r_spec and r_spec.loader
    rules_mod = importlib.util.module_from_spec(r_spec)
    sys.modules["fightsafe_ai.risk.rules"] = rules_mod
    r_spec.loader.exec_module(rules_mod)

    t_spec = importlib.util.spec_from_file_location(
        "fightsafe_ai.risk.time_order",
        _SRC / "fightsafe_ai" / "risk" / "time_order.py",
    )
    assert t_spec and t_spec.loader
    t_mod = importlib.util.module_from_spec(t_spec)
    sys.modules["fightsafe_ai.risk.time_order"] = t_mod
    t_spec.loader.exec_module(t_mod)

    lt_spec = importlib.util.spec_from_file_location(
        "fightsafe_ai.risk.limb_tier",
        _SRC / "fightsafe_ai" / "risk" / "limb_tier.py",
    )
    assert lt_spec and lt_spec.loader
    lt_mod = importlib.util.module_from_spec(lt_spec)
    sys.modules["fightsafe_ai.risk.limb_tier"] = lt_mod
    lt_spec.loader.exec_module(lt_mod)

    s_spec = importlib.util.spec_from_file_location(
        "fightsafe_ai.risk.scorer",
        _SRC / "fightsafe_ai" / "risk" / "scorer.py",
    )
    assert s_spec and s_spec.loader
    scorer_mod = importlib.util.module_from_spec(s_spec)
    sys.modules["fightsafe_ai.risk.scorer"] = scorer_mod
    s_spec.loader.exec_module(scorer_mod)
    return rules_mod, scorer_mod


_rules, _scorer = _load_risk_for_tests()
RULE_FAST_DOWNWARD = _rules.RULE_FAST_DOWNWARD
load_interpretable_risk_config = _rules.load_interpretable_risk_config
compute_interpretable_risk = _scorer.compute_interpretable_risk
build_combat_mvp_frame_risk = _scorer.build_combat_mvp_frame_risk
COL_RISK_LEVEL = _scorer.COL_RISK_LEVEL
COL_RISK_SCORE = _scorer.COL_RISK_SCORE
COL_TIMESTAMP = _scorer.COL_TIMESTAMP


def _full_frame_row() -> dict[str, Any]:
    return {
        "frame_id": "frame_000001",
        "hip_vertical_velocity": 0.8,
        "head_vertical_velocity": 0.1,
        "torso_angle_deg": 50.0,
        "low_posture_duration_frames": 20.0,
        "instability_score": 0.2,
        "near_ground": True,
        "guard_level": 0.15,
        "facing_away_score": 0.1,
        "reaction_delay_score": 0.0,
        "anomaly_score": 0.0,
        "strike_incoming_proxy": 0.0,
    }


def test_compute_interpretable_risk_columns_and_levels() -> None:
    df = pd.DataFrame([_full_frame_row() for _ in range(3)])
    yml = _ROOT / "configs" / "risk_rules.yaml"
    out = compute_interpretable_risk(df, include_rule_component_columns=True, rules_yaml=yml)
    assert COL_RISK_SCORE in out.columns
    assert COL_RISK_LEVEL in out.columns
    assert "triggered_rules" in out.columns
    assert f"risk_component_{RULE_FAST_DOWNWARD}" in out.columns
    assert (out[COL_RISK_SCORE] >= 0).all() and (out[COL_RISK_SCORE] <= 1).all()
    assert set(out[COL_RISK_LEVEL].unique()) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    tr = out["triggered_rules"].iloc[0]
    assert isinstance(tr, list)


def test_load_yaml_from_project_configs() -> None:
    yml = _ROOT / "configs" / "risk_rules.yaml"
    assert yml.is_file()
    cfg = load_interpretable_risk_config(yml)
    assert cfg.aggregation.level_critical_min > cfg.aggregation.level_high_min


def test_empty_dataframe() -> None:
    yml = _ROOT / "configs" / "risk_rules.yaml"
    r = compute_interpretable_risk(pd.DataFrame(), rules_yaml=yml)
    assert len(r) == 0
    assert COL_RISK_SCORE in r.columns


def test_renormalize_when_columns_missing() -> None:
    df = pd.DataFrame(
        {
            "hip_vertical_velocity": [0.0, 1.0],
            "torso_angle_deg": [0.0, 60.0],
        }
    )
    yml = _ROOT / "configs" / "risk_rules.yaml"
    out = compute_interpretable_risk(df, rules_yaml=yml)
    assert len(out) == 2
    assert np.isfinite(out[COL_RISK_SCORE].iloc[0])


def test_build_combat_mvp_frame_risk_timestamp_and_columns() -> None:
    yml = _ROOT / "configs" / "risk_rules.yaml"
    rows = [
        {**_full_frame_row(), "frame_id": "frame_000002"},
        {**_full_frame_row(), "frame_id": "frame_000001"},
    ]
    out = build_combat_mvp_frame_risk(pd.DataFrame(rows), fps=10.0, rules_yaml=yml)
    assert COL_TIMESTAMP in out.columns
    assert "frame_index" in out.columns
    assert out.iloc[0]["frame_id"] == "frame_000001"
    assert out.iloc[0][COL_TIMESTAMP] == 0.0
    assert out.iloc[1][COL_TIMESTAMP] == 0.1
    assert 0.0 <= float(out.iloc[0][COL_RISK_SCORE]) <= 1.0
    assert out.iloc[0][COL_RISK_LEVEL] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert isinstance(out.iloc[0]["triggered_rules"], list)
