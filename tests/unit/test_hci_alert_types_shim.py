"""Thin re-export module :mod:`fightsafe_ai.hci.alert_types` (coverage + API stability)."""

from __future__ import annotations

import fightsafe_ai.hci.alert_types as at
import fightsafe_ai.hci.alerts as core


def test_alert_types_reexports_equal_primary_module() -> None:
    assert at.RISK_LEVEL_TO_ALERT is core.RISK_LEVEL_TO_ALERT
    assert at.Alert is core.Alert
    assert at.RefereeAlert is core.RefereeAlert
    assert at.RefereeAlertLevel is core.RefereeAlertLevel
    assert at.generate_referee_alert is core.generate_referee_alert


def test_all_exports_listed() -> None:
    assert set(at.__all__) == {
        "RISK_LEVEL_TO_ALERT",
        "Alert",
        "RefereeAlert",
        "RefereeAlertLevel",
        "generate_referee_alert",
    }
