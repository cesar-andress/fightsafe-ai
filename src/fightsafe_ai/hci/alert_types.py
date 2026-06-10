"""
Backward-compatible re-exports for referee alert types (see :mod:`fightsafe_ai.hci.alerts`).

**Decision-support only** — not automated officiating.
"""

from __future__ import annotations

from fightsafe_ai.hci.alerts import (
    RISK_LEVEL_TO_ALERT,
    Alert,
    RefereeAlert,
    RefereeAlertLevel,
    generate_referee_alert,
)


__all__ = [
    "RISK_LEVEL_TO_ALERT",
    "Alert",
    "RefereeAlert",
    "RefereeAlertLevel",
    "generate_referee_alert",
]
