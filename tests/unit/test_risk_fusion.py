"""Unit tests for risk level ordering and the fusion :class:`RiskDecision` engine."""

from __future__ import annotations

from typing import Any

import pytest

from fightsafe_ai.action.base import ActionType
from fightsafe_ai.anomaly.base import AnomalyType
from fightsafe_ai.risk.fusion import (
    RiskFusionInput,
    compute_risk_decision,
    fuse_weighted_mean,
)
from fightsafe_ai.risk.levels import (
    RISK_LEVEL_ORDER,
    RiskLevelName,
    max_risk_level,
    parse_risk_level,
    risk_level_rank,
)


pytestmark = pytest.mark.unit


def _fi(**kwargs: Any) -> RiskFusionInput:
    base: dict[str, Any] = {
        "timestamp": 1.0,
        "fighter_id": "0",
    }
    base.update(kwargs)
    return RiskFusionInput(
        float(base["timestamp"]),
        str(base["fighter_id"]),
        action_conf=base.get("action_conf") or {},
        anomaly_conf=base.get("anomaly_conf") or {},
        surrender_detected=bool(base.get("surrender_detected", False)),
        surrender_confidence=float(base.get("surrender_confidence", 0.0)),
        instability_score=float(base.get("instability_score", 0.0)),
        hip_vertical_velocity=base.get("hip_vertical_velocity"),
        inactivity_score=float(base.get("inactivity_score", 0.0)),
    )


def test_fuse_weighted_mean_unchanged() -> None:
    s = fuse_weighted_mean({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 1.0})
    assert abs(s - 0.5) < 1e-6


def test_risk_level_ordering() -> None:
    assert RISK_LEVEL_ORDER[0] is RiskLevelName.LOW
    assert max_risk_level(RiskLevelName.LOW, RiskLevelName.MEDIUM) is RiskLevelName.MEDIUM
    assert max_risk_level(*RISK_LEVEL_ORDER) is RiskLevelName.CRITICAL
    assert parse_risk_level("MEDIUM") is RiskLevelName.MEDIUM
    assert risk_level_rank(RiskLevelName.HIGH) == 2


def test_fusion_surrender_critical() -> None:
    d = compute_risk_decision(
        _fi(surrender_detected=True, surrender_confidence=0.6),
    )
    assert d.risk_level is RiskLevelName.CRITICAL
    assert "surrender" in d.triggered_signals[0] or d.triggered_signals[0].endswith("critical")
    assert any("Surrender" in x or "surrender" in x for x in d.explanation_facts)


def test_fusion_fall_inactivity_critical() -> None:
    d = compute_risk_decision(
        _fi(
            anomaly_conf={
                AnomalyType.FALL_TORSO_ANGLE_COLLAPSE: 0.45,
                AnomalyType.INACTIVITY_LOW_KEYPOINT_MOTION: 0.4,
            },
            inactivity_score=0.45,
        )
    )
    assert d.risk_level is RiskLevelName.CRITICAL
    assert (
        "fall_and_inactivity" in d.triggered_signals[0] or "fall" in d.triggered_signals[0].lower()
    )


def test_fusion_fall_high_downward_high() -> None:
    d = compute_risk_decision(
        _fi(
            anomaly_conf={AnomalyType.FALL_TORSO_ANGLE_COLLAPSE: 0.22},
            hip_vertical_velocity=0.7,
        )
    )
    assert d.risk_level is RiskLevelName.HIGH
    assert any("downward" in t or "fall_high" in t for t in d.triggered_signals)


def test_fusion_limb_high_and_critical() -> None:
    hi = compute_risk_decision(_fi(anomaly_conf={AnomalyType.LIMB_BILATERAL_ASYMMETRY: 0.58}))
    assert hi.risk_level is RiskLevelName.HIGH
    cr = compute_risk_decision(_fi(anomaly_conf={AnomalyType.LIMB_SUDDEN_SUPPORT_LOSS: 0.8}))
    assert cr.risk_level is RiskLevelName.CRITICAL


def test_fusion_low_guard_to_medium() -> None:
    d = compute_risk_decision(
        _fi(
            action_conf={ActionType.LOW_GUARD: 0.5},
        )
    )
    assert d.risk_level is RiskLevelName.MEDIUM
    assert "low_guard" in d.triggered_signals[0]


def test_fusion_turned_back_instability() -> None:
    med = compute_risk_decision(
        _fi(
            action_conf={ActionType.TURNED_BACK: 0.45},
            instability_score=0.5,
        )
    )
    assert med.risk_level in (RiskLevelName.MEDIUM, RiskLevelName.HIGH)
    hi = compute_risk_decision(
        _fi(
            action_conf={ActionType.TURNED_BACK: 0.45},
            instability_score=0.7,
        )
    )
    assert hi.risk_level is RiskLevelName.HIGH
    assert any("turned_back" in t for t in hi.triggered_signals)


def test_fusion_surrender_like_anomaly_critical() -> None:
    d = compute_risk_decision(
        _fi(
            anomaly_conf={AnomalyType.SURRENDER_TAP_LIKE: 0.5},
        )
    )
    assert d.risk_level is RiskLevelName.CRITICAL
