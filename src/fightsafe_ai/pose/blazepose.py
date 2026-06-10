"""
BlazePose 33-landmark names (same order as legacy ``mediapipe.solutions.pose.PoseLandmark``).
"""

from __future__ import annotations


# fmt: off
BLAZEPOSE_33: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
# fmt: on

_NAME_TO_INDEX: dict[str, int] = {n: i for i, n in enumerate(BLAZEPOSE_33)}


def blazepose_index(name: str) -> int | None:
    """Index 0..32 for a landmark ``name``, or ``None`` if not part of the 33-pose set."""
    return _NAME_TO_INDEX.get(name)
