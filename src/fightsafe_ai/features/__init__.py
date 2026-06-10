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

Biomechanical features from keypoint sequences.
"""

from fightsafe_ai.features.anomaly import (
    COL_ANOMALY_SCORE,
    COL_ANOMALY_TYPE,
    add_limb_anomaly_columns,
)
from fightsafe_ai.features.biomechanics import (
    compute_biomechanical_features,
    compute_body_centers,
    compute_pose_features,
    compute_torso_angle,
)
from fightsafe_ai.features.temporal import (
    ReactionDelayConfig,
    TemporalFeatureConfig,
    compute_temporal_features,
)


__all__ = [
    "COL_ANOMALY_SCORE",
    "COL_ANOMALY_TYPE",
    "ReactionDelayConfig",
    "TemporalFeatureConfig",
    "add_limb_anomaly_columns",
    "compute_biomechanical_features",
    "compute_body_centers",
    "compute_pose_features",
    "compute_temporal_features",
    "compute_torso_angle",
]
