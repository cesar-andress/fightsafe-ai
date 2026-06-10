"""Action recognition interfaces, heuristics, and structured signals (not autonomous scoring)."""

from fightsafe_ai.action.base import (
    ActionLabel,
    ActionSignal,
    ActionType,
    BaseActionRecognizer,
    BaseActionSignalEmitter,
    landmark_map_xy,
)
from fightsafe_ai.action.defense import (
    defensive_incapacity_confidence,
    guard_open_proxy,
    low_guard_confidence,
    turned_back_confidence,
)
from fightsafe_ai.action.punch_kick import (
    LimbMotionFeatures,
    body_scale,
    kick_activity_confidence,
    limb_motion_features,
    punch_activity_confidence,
    strike_energy_proxy,
)
from fightsafe_ai.action.temporal_classifier import (
    HeuristicMVPActionDetector,
    HeuristicMVPConfig,
    confidence_weighted_mean,
    majority_vote,
    run_sequence_mvp,
)


__all__ = [
    "ActionLabel",
    "ActionSignal",
    "ActionType",
    "BaseActionRecognizer",
    "BaseActionSignalEmitter",
    "HeuristicMVPActionDetector",
    "HeuristicMVPConfig",
    "LimbMotionFeatures",
    "body_scale",
    "confidence_weighted_mean",
    "defensive_incapacity_confidence",
    "guard_open_proxy",
    "kick_activity_confidence",
    "landmark_map_xy",
    "limb_motion_features",
    "low_guard_confidence",
    "majority_vote",
    "punch_activity_confidence",
    "run_sequence_mvp",
    "strike_energy_proxy",
    "turned_back_confidence",
]
