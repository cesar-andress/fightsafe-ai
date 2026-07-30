"""
Temporal **event** helpers built on pose sequences (decision-support candidates, not rulings).
"""

from fightsafe_ai.events.tap_detector import (
    EVENT_FOOT_TAP,
    EVENT_HAND_TAP,
    TapCandidateEvent,
    TapDetectorConfig,
    detect_tap_candidates,
)
from fightsafe_ai.events.vulnerability_detector import (
    EVENT_CHOKE_UNCONSCIOUSNESS_CANDIDATE,
    EVENT_KO_COLLAPSE,
    EVENT_NO_INTELLIGENT_DEFENSE,
    EVENT_POST_IMPACT_INACTIVITY,
    VulnerabilityCandidateEvent,
    VulnerabilityDetectorConfig,
    detect_vulnerability_candidates,
)


__all__ = [
    "EVENT_CHOKE_UNCONSCIOUSNESS_CANDIDATE",
    "EVENT_FOOT_TAP",
    "EVENT_HAND_TAP",
    "EVENT_KO_COLLAPSE",
    "EVENT_NO_INTELLIGENT_DEFENSE",
    "EVENT_POST_IMPACT_INACTIVITY",
    "TapCandidateEvent",
    "TapDetectorConfig",
    "VulnerabilityCandidateEvent",
    "VulnerabilityDetectorConfig",
    "detect_tap_candidates",
    "detect_vulnerability_candidates",
]
