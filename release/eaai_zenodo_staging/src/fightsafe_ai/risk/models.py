"""
FightSafe AI

AI-assisted safety detection for combat sports officiating.

Authors:
- David Martin Moncunill (david.martinm@ucjc.edu)
- César Andrés Sánchez (cesar.andress@ucjc.edu)

Affiliation:
Camilo José Cela University (UCJC)
Madrid, Spain

This module is part of a research-oriented system for human-in-the-loop safety analysis.

Strongly typed configuration for heuristic risk rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fightsafe_ai.exceptions import ConfigurationError


@dataclass(frozen=True)
class RiskRuleParams:
    """Flat rule parameters consumed by :func:`~fightsafe_ai.risk.engine.detect_risk_events`."""

    torso_angle_threshold_deg: float = 28.0
    hip_velocity_threshold: float = 0.45
    tilt_velocity_angle_scale: float = 20.0
    tilt_velocity_speed_scale: float = 1.0
    near_ground_min_frames: int = 20
    erratic_variance_window: int = 9
    erratic_variance_factor: float = 2.2
    jerk_threshold: float = 3.5
    weight_tilt_velocity: float = 0.34
    weight_ground: float = 0.33
    weight_erratic: float = 0.33
    risk_flag_threshold: float = 0.5


def risk_rules_from_yaml(path: Path) -> RiskRuleParams:
    """Parse ``configs/risk_rules.yaml`` nested dict into :class:`RiskRuleParams`."""
    if not path.is_file():
        raise ConfigurationError(f"Risk rules file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return RiskRuleParams()
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"Expected mapping in YAML: {path}")

    tilt = raw.get("tilt_velocity", {}) or {}
    ground = raw.get("ground_contact", {}) or {}
    err = raw.get("erratic_motion", {}) or {}
    agg = raw.get("aggregation", {}) or {}

    flat: dict[str, Any] = {
        "torso_angle_threshold_deg": tilt.get("torso_angle_threshold_deg", 28.0),
        "hip_velocity_threshold": tilt.get("hip_velocity_threshold", 0.45),
        "tilt_velocity_angle_scale": tilt.get("tilt_velocity_angle_scale", 20.0),
        "tilt_velocity_speed_scale": tilt.get("tilt_velocity_speed_scale", 1.0),
        "near_ground_min_frames": int(ground.get("near_ground_min_frames", 20)),
        "erratic_variance_window": int(err.get("variance_window", 9)),
        "erratic_variance_factor": err.get("variance_factor", 2.2),
        "jerk_threshold": err.get("jerk_threshold", 3.5),
        "weight_tilt_velocity": agg.get("weight_tilt_velocity", 0.34),
        "weight_ground": agg.get("weight_ground", 0.33),
        "weight_erratic": agg.get("weight_erratic", 0.33),
        "risk_flag_threshold": agg.get("risk_flag_threshold", 0.5),
    }
    return RiskRuleParams(**flat)
