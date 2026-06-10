"""
Structured anomaly / safety-signal types (not clinical, not a diagnosis).

**Limitations (all detectors in this package):**

- 2D pose, camera-dependent; depth, self-occlusion, and contact are not observed.
- Heuristic thresholds are engineering defaults, not normative human data or org rulebooks.
- Outputs are **cues** for review alongside human judgment; they do not certify injury, stoppage,
  or loss of ability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from fightsafe_ai.pose.keypoints import Keypoint, PoseResult


class AnomalyType(StrEnum):
    """MVP high-level categories; sub-behavior is in ``AnomalySignal.evidence`` when needed."""

    # --- FallDetector ---
    FALL_RAPID_DOWNWARD = "FALL_RAPID_DOWNWARD"
    FALL_TORSO_ANGLE_COLLAPSE = "FALL_TORSO_ANGLE_COLLAPSE"
    FALL_PROLONGED_LOW_POSTURE = "FALL_PROLONGED_LOW_POSTURE"
    # --- InactivityDetector ---
    INACTIVITY_LOW_KEYPOINT_MOTION = "INACTIVITY_LOW_KEYPOINT_MOTION"
    INACTIVITY_LOW_COM_MOTION = "INACTIVITY_LOW_COM_MOTION"
    # --- LimbAnomalyDetector ---
    LIMB_JOINT_ANGULAR_ANOMALY = "LIMB_JOINT_ANGULAR_ANOMALY"
    LIMB_BILATERAL_ASYMMETRY = "LIMB_BILATERAL_ASYMMETRY"
    LIMB_SUDDEN_SUPPORT_LOSS = "LIMB_SUDDEN_SUPPORT_LOSS"
    # --- Surrender-like (vision-only) ---
    SURRENDER_OSCILLATION = "SURRENDER_OSCILLATION"
    SURRENDER_TAP_LIKE = "SURRENDER_TAP_LIKE"
    SURRENDER_RHYTHMIC_HANDS = "SURRENDER_RHYTHMIC_HANDS"


@dataclass(frozen=True, slots=True)
class AnomalySignal:
    """
    A single window-level or frame-indexed heuristic (decision-support, not a medical label).

    ``anomaly_type`` is a machine-readable key; ``evidence`` should hold small numeric
    or short-string features for explainability and tuning.
    """

    timestamp: float
    fighter_id: str
    anomaly_type: AnomalyType
    confidence: float
    evidence: dict[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")


def pose_sequence_from_landmark_dicts(
    frames: list[dict[str, tuple[float, float]]],
    *,
    id_prefix: str = "f",
) -> list[PoseResult]:
    """Build :class:`PoseResult` for each frame for use with :mod:`fightsafe_ai.risk.surrender`."""
    out: list[PoseResult] = []
    for i, f in enumerate(frames):
        kps = [Keypoint(n, float(x), float(y), None, None) for n, (x, y) in f.items()]
        out.append(PoseResult(f"{id_prefix}{i}", kps))
    return out


class BaseTimeSeriesAnomalyDetector(ABC):
    """
    Ingests time-aligned 2D landmark dicts; emits zero or more :class:`AnomalySignal` for the
    *last* frame in the current window (timestamp = last time).
    """

    @abstractmethod
    def analyze(
        self,
        times: list[float],
        frames: list[dict[str, tuple[float, float]]],
        fighter_id: str,
    ) -> list[AnomalySignal]: ...


# Back-compat: earlier stub name; prefer :class:`BaseTimeSeriesAnomalyDetector`.
class BaseAnomalyModule(BaseTimeSeriesAnomalyDetector):
    """Alias for the time-series contract used by MVP detectors."""
