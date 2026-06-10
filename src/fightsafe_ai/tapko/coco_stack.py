"""Load consolidated or per-frame pose CSV into a ``(T, 17, 2)`` COCO-17 stack."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fightsafe_ai.exceptions import VideoIOError
from fightsafe_ai.keypoints.io import load_landmark_maps_ordered


# Same order as live pipeline / YOLO backends (must match :mod:`fightsafe_ai.events.tap_detector`).
COCO17_POSE_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def landmark_map_to_coco17_xy(lm: dict[str, tuple[float, float]] | None) -> np.ndarray:
    """One frame: ``(17, 2)`` with NaNs for missing joints."""
    row = np.full((17, 2), np.nan, dtype=np.float64)
    if not lm:
        return row
    for i, name in enumerate(COCO17_POSE_NAMES):
        p = lm.get(name)
        if p is None:
            continue
        try:
            row[i, 0] = float(p[0])
            row[i, 1] = float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
    return row


def load_coco17_stack_from_pose_csv(source: Path) -> np.ndarray:
    """
    Load MediaPipe / consolidated ``pose_keypoints.csv`` or a directory of per-frame CSVs.

    Returns
    -------
    np.ndarray
        Shape ``(T, 17, 2)``, float64.
    """
    p = source.expanduser().resolve()
    if not p.exists():
        raise VideoIOError(f"Pose source not found: {p}")
    ordered = load_landmark_maps_ordered(p)
    if not ordered:
        raise VideoIOError(f"No pose frames loaded from {p}")
    t_n = len(ordered)
    out = np.full((t_n, 17, 2), np.nan, dtype=np.float64)
    for ti, (_label, lm) in enumerate(ordered):
        out[ti] = landmark_map_to_coco17_xy(lm)
    return out


__all__ = [
    "COCO17_POSE_NAMES",
    "landmark_map_to_coco17_xy",
    "load_coco17_stack_from_pose_csv",
]
