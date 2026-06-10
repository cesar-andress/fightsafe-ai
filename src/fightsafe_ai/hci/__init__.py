"""
Human-in-the-loop (HCI) layer: how FightSafe **recommends** attention to a referee.

**This is a decision-support system, not an automated referee.** Nothing here stops a
fight or records a result; it only structures alerts for human judgment.
"""

from fightsafe_ai.hci.alert_manager import RefereeAlertManager
from fightsafe_ai.hci.alerts import (
    RISK_LEVEL_TO_ALERT,
    Alert,
    RefereeAlert,
    RefereeAlertLevel,
    generate_referee_alert,
)
from fightsafe_ai.hci.referee_messages import (
    message_for_level,
    recommended_action_for_risk,
    short_message_for_risk,
    template_payload,
    validate_no_forbidden_phrase,
)


__all__ = [
    "RISK_LEVEL_TO_ALERT",
    "Alert",
    "RefereeAlert",
    "RefereeAlertLevel",
    "RefereeAlertManager",
    "generate_referee_alert",
    "message_for_level",
    "recommended_action_for_risk",
    "short_message_for_risk",
    "template_payload",
    "validate_no_forbidden_phrase",
]
