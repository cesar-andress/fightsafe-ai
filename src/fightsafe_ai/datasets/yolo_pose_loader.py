"""
Helpers for **locally produced** YOLO-pose / Ultralytics-style outputs (no weights, no download).

Training videos and label files are **not** part of this package. See ``docs/datasets.md``
for registry entries and format notes.

Optional runtime dependency: ``ultralytics`` (not required for the pure helpers below).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class YOLOKeypointFrame:
    """A minimal frame of keypoint (x, y) pairs in pixel or normalized space (caller-defined)."""

    frame_id: int
    xy: list[tuple[float, float]]


def n_keypoints_to_blazepose_hint() -> str:
    """
    Return a static note: COCO-17 to BlazePose-33 mapping is **out of core**; do it in lab code.
    """
    return (
        "COCO-17 to BlazePose-33 mapping is not part of the FightSafe core path; use lab scripts."
    )


def as_plain_dict(frames: list[YOLOKeypointFrame]) -> list[dict[str, Any]]:
    """Serialize frames to JSON-friendly dict rows (pure, for tooling)."""
    return [
        {
            "frame_id": f.frame_id,
            "kpts": [{"x": x, "y": y} for x, y in f.xy],
        }
        for f in frames
    ]


def describe_yolo_pose_label_convention() -> str:
    """
    Short English description of common Ultralytics YOLO-pose label file expectations.

    Per-image ``.txt`` lines may list class id, box, and kpt groups; exact layout is version
    specific—always read upstream docs for your installed ``ultralytics`` version.
    """
    return (
        "Typical YOLO-pose labels are per-image text files with normalized coordinates; "
        "use Ultralytics export docs to match your dataset version. FightSafe does not "
        "parse these files by default—add a project-specific reader if needed."
    )
