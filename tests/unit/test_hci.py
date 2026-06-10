"""Human-in-the-loop referee alert layer (recommendations only, not decisions)."""

from __future__ import annotations

import pytest

from fightsafe_ai.hci import (
    RISK_LEVEL_TO_ALERT,
    RefereeAlertLevel,
    RefereeAlertManager,
    generate_referee_alert,
)
from fightsafe_ai.hci.alerts import RefereeAlert
from fightsafe_ai.risk.fusion import RiskFusionInput, compute_risk_decision
from fightsafe_ai.risk.rules import RULE_INSTABILITY, RULE_LIMB_ANOMALY


def test_risk_to_alert_map_complete() -> None:
    assert RISK_LEVEL_TO_ALERT["LOW"] is RefereeAlertLevel.INFO
    assert RISK_LEVEL_TO_ALERT["MEDIUM"] is RefereeAlertLevel.WATCH
    assert RISK_LEVEL_TO_ALERT["HIGH"] is RefereeAlertLevel.WARNING
    assert RISK_LEVEL_TO_ALERT["CRITICAL"] is RefereeAlertLevel.STOP_RECOMMENDED


def test_generate_referee_alert_fields() -> None:
    a = generate_referee_alert(
        {
            "risk_level": "high",
            "timestamp": 12.5,
            "fighter_id": "A1",
            "triggered_rules": [RULE_INSTABILITY, RULE_LIMB_ANOMALY],
            "risk_score": 0.67,
        }
    )
    assert a.alert_level is RefereeAlertLevel.WARNING
    assert a.fighter_id == "A1"
    assert a.timestamp == 12.5
    assert "0.67" in a.short_message
    assert len(a.triggered_signals) == 2
    assert RULE_INSTABILITY in a.triggered_signals[0]
    assert (
        "review" in a.recommended_human_action.lower()
        or "procedure" in a.recommended_human_action.lower()
    )


def test_invalid_risk_falls_back_to_info() -> None:
    a = generate_referee_alert(
        {
            "risk_level": "invalid_band",
            "timestamp": 0.0,
            "triggered_rules": [],
        }
    )
    assert a.alert_level is RefereeAlertLevel.INFO


def test_manager_matches_generate() -> None:
    m = RefereeAlertManager()
    d = {
        "risk_level": "CRITICAL",
        "timestamp": 1.0,
        "triggered_rules": [],
        "risk_score": 0.9,
    }
    g = m.generate(d)
    assert g.alert_level is RefereeAlertLevel.STOP_RECOMMENDED
    assert generate_referee_alert(d).short_message == g.short_message


def test_from_series() -> None:
    pd = pytest.importorskip("pandas")
    m = RefereeAlertManager()
    row = pd.Series(
        {
            "risk_level": "MEDIUM",
            "timestamp": 0.1,
            "triggered_rules": [RULE_INSTABILITY],
            "risk_score": 0.3,
        }
    )
    a = m.from_series(row)
    assert a.alert_level is RefereeAlertLevel.WATCH
    assert a.timestamp == 0.1


def test_from_risk_decision() -> None:
    rd = compute_risk_decision(
        RiskFusionInput(
            timestamp=2.0,
            fighter_id="b2",
            action_conf={},
            surrender_detected=True,
            surrender_confidence=0.6,
        )
    )
    a = RefereeAlertManager.from_risk_decision(rd)
    assert isinstance(a, RefereeAlert)
    assert a.fighter_id == "b2"
    assert a.alert_level is RefereeAlertLevel.STOP_RECOMMENDED
    assert a.triggered_signals
