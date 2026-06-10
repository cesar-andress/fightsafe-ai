"""Risk YAML parsing (isolated :mod:`fightsafe_ai.risk.models` import)."""

from pathlib import Path

import yaml
from tests.support.isolated import load_risk_models


_m = load_risk_models()
RiskRuleParams = _m.RiskRuleParams
risk_rules_from_yaml = _m.risk_rules_from_yaml


def test_risk_rules_from_yaml_roundtrip(tmp_path: Path) -> None:
    raw = {
        "tilt_velocity": {"torso_angle_threshold_deg": 30.0},
        "ground_contact": {"near_ground_min_frames": 15},
        "erratic_motion": {"variance_window": 11},
        "aggregation": {"risk_flag_threshold": 0.6},
    }
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    params = risk_rules_from_yaml(p)
    assert isinstance(params, RiskRuleParams)
    assert params.torso_angle_threshold_deg == 30.0
    assert params.near_ground_min_frames == 15
    assert params.erratic_variance_window == 11
    assert params.risk_flag_threshold == 0.6
