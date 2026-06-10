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

Pose estimation result types (dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Keypoint:
    """
    Single 2D/3D landmark with optional depth and visibility.

    Coordinates follow the estimator convention (e.g. MediaPipe: normalized image x/y,
    relative z scale; world landmarks may use metric z — stored here when provided).
    """

    name: str
    x: float
    y: float
    z: float | None = None
    visibility: float | None = None


@dataclass
class PoseResult:
    """Pose inference for one image or video frame."""

    frame_id: str
    keypoints: list[Keypoint] = field(default_factory=list)


type KeypointsResult = PoseResult
