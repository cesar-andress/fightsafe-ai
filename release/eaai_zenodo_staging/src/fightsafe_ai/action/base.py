"""Action recognition types: structured signals and abstract recognizer interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fightsafe_ai.pose.keypoints import Keypoint


@dataclass(frozen=True, slots=True)
class ActionLabel:
    """A single action hypothesis for a window or frame (legacy string label)."""

    name: str
    confidence: float
    frame_index: int


class ActionType(StrEnum):
    """
    High-level action categories from heuristic / learned recognizers (not risk levels).

    ``DEFENSIVE_INCAPACITY`` is a composite: exposed guard with little protective motion
    (decision-support; not medical incapacity).
    """

    PUNCH_ACTIVITY = "PUNCH_ACTIVITY"
    KICK_ACTIVITY = "KICK_ACTIVITY"
    LOW_GUARD = "LOW_GUARD"
    TURNED_BACK = "TURNED_BACK"
    DEFENSIVE_INCAPACITY = "DEFENSIVE_INCAPACITY"


@dataclass(frozen=True, slots=True)
class ActionSignal:
    """
    One emitted interpretation at an instant, separate from risk scores.

    ``evidence`` holds small numeric / string features for explainability and tuning.
    """

    timestamp: float
    fighter_id: str
    action_type: ActionType
    confidence: float
    evidence: dict[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0,1]")


def landmark_map_xy(keypoints: Iterable[Keypoint]) -> dict[str, tuple[float, float]]:
    """Map blaze-style keypoint names to ``(x, y)`` in estimator coordinates."""
    return {k.name: (k.x, k.y) for k in keypoints}


class BaseActionRecognizer(ABC):
    """
    Map pose or appearance features to action labels.

    For structured outputs prefer :class:`HeuristicMVPActionDetector` or a future
    learnable backend that returns :class:`ActionSignal` lists.
    """

    @abstractmethod
    def classify_window(self, window: list[dict[str, Any]]) -> list[ActionLabel]:
        """
        Parameters
        ----------
        window
            A short temporal window of per-frame feature dicts (implementation-defined).
        """
        ...


class BaseActionSignalEmitter(ABC):
    """
    Emits :class:`ActionSignal` for each frame; keeps action understanding pluggable
    (SportsMOT-style, ByteTrack+AR net, etc.).
    """

    @abstractmethod
    def process_frame(
        self,
        timestamp: float,
        fighter_id: str,
        current_landmarks: dict[str, tuple[float, float]],
        previous_landmarks: dict[str, tuple[float, float]] | None,
        dt: float,
    ) -> list[ActionSignal]: ...
