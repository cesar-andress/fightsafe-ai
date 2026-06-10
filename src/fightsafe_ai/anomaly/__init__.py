"""Anomaly and safety-signal modules (no clinical claims; decision-support only)."""

from fightsafe_ai.anomaly.base import (
    AnomalySignal,
    AnomalyType,
    BaseAnomalyModule,
    BaseTimeSeriesAnomalyDetector,
    pose_sequence_from_landmark_dicts,
)
from fightsafe_ai.anomaly.fall_detector import (
    FallDetector,
    FallDetectorConfig,
    fall_likelihood_from_y_coords,
)
from fightsafe_ai.anomaly.inactivity_detector import (
    InactivityDetector,
    InactivityDetectorConfig,
    inactivity_score,
)
from fightsafe_ai.anomaly.limb_anomaly import (
    COL_ANOMALY_SCORE,
    COL_ANOMALY_TYPE,
    LimbAnomalyDetector,
    LimbAnomalyDetectorConfig,
    add_limb_anomaly_columns,
)
from fightsafe_ai.anomaly.surrender_detector import (
    COL_SURRENDER_CONFIDENCE,
    SurrenderAnomalyDetector,
    SurrenderDetectionResult,
    SurrenderHeuristicConfig,
    apply_surrender_overrides_to_risk_dataframe,
    detect_surrender,
)


__all__ = [
    "COL_ANOMALY_SCORE",
    "COL_ANOMALY_TYPE",
    "COL_SURRENDER_CONFIDENCE",
    "AnomalySignal",
    "AnomalyType",
    "BaseAnomalyModule",
    "BaseTimeSeriesAnomalyDetector",
    "FallDetector",
    "FallDetectorConfig",
    "InactivityDetector",
    "InactivityDetectorConfig",
    "LimbAnomalyDetector",
    "LimbAnomalyDetectorConfig",
    "SurrenderAnomalyDetector",
    "SurrenderDetectionResult",
    "SurrenderHeuristicConfig",
    "add_limb_anomaly_columns",
    "apply_surrender_overrides_to_risk_dataframe",
    "detect_surrender",
    "fall_likelihood_from_y_coords",
    "inactivity_score",
    "pose_sequence_from_landmark_dicts",
]
