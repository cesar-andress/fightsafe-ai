"""
Per-frame associations (:class:`Tracklet`) and persistent state (:class:`FighterTrack`).

Adapters for **ByteTrack**, **DeepSORT**, **SportsMOT**, **SportSORT**, etc. can expose their
internal tracks as :class:`FighterTrack` or emit :class:`Tracklet` rows each frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Tracklet:
    """
    One **associated** detection at a given frame: a stable ``track_id`` and optional box.

    ``track_id`` is a tracker handle for one physical subject in the clip, **not** a
    competition ID.
    """

    track_id: int
    frame_index: int
    box_xyxy: tuple[float, float, float, float] | None = None
    confidence: float = 1.0
    class_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fighter_id(self) -> int:
        """Alias for :attr:`track_id` (``fighter_id`` in risk / CSV columns)."""
        return self.track_id


@dataclass
class FighterTrack:
    """
    Persistent state for one track across frames (IoU MVP; Kalman / ReID hooks reserved).

    ``last_box`` is a length-4 float64 array ``[x1, y1, x2, y2]`` in detector space.
    """

    track_id: int
    last_frame: int
    last_box: np.ndarray
    hits: int = 1
    time_since_update: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tracklet(self, frame_index: int) -> Tracklet:
        """Build a :class:`Tracklet` for this track at ``frame_index`` (uses ``last_box``)."""
        b = self.last_box
        t = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        return Tracklet(
            track_id=self.track_id,
            frame_index=frame_index,
            box_xyxy=t,
            confidence=1.0,
            class_id=0,
            metadata=dict(self.metadata),
        )
