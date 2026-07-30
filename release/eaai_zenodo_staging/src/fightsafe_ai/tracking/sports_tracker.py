"""
IoU + greedy association **MVP** tracker and lightweight **SportsTracker** facade.

Sufficient for **tests** and single-/dual-fighter clips with box detections. Replace with
**ByteTrack** / **DeepSORT** / **SportsMOT** pipelines for production research.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fightsafe_ai.tracking.base import BaseTracker
from fightsafe_ai.tracking.identity import assign_identities_greedy_iou
from fightsafe_ai.tracking.tracklet import FighterTrack, Tracklet


@dataclass
class SportsTracker(BaseTracker):
    """
    Greedy IoU association: each new box matches an *unused* previous best-IoU track or
    spawns a new id.

    Maintains :attr:`fighter_tracks` for inspection and for future
    Kalman / *lost-track* policy (``time_since_update`` reserved).
    """

    iou_threshold: float = 0.3
    _next_id: int = 0
    _last_boxes: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    fighter_tracks: dict[int, FighterTrack] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear state (new video / new clip)."""
        self._next_id = 0
        self._last_boxes = {}
        self.fighter_tracks = {}

    def update(self, frame_index: int, detections: list[dict[str, Any]]) -> list[Tracklet]:
        det_boxes: list[np.ndarray] = []
        confidences: list[float] = []
        for d in detections:
            raw = d.get("box_xyxy")
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.float64).ravel()[:4]
            if arr.size != 4:
                continue
            det_boxes.append(arr)
            c = d.get("confidence", 1.0)
            confidences.append(float(c) if c is not None else 1.0)

        if not det_boxes:
            self._last_boxes = {}
            return []

        track_ids, self._next_id, new_state = assign_identities_greedy_iou(
            self._last_boxes,
            det_boxes,
            iou_threshold=float(self.iou_threshold),
            next_id=self._next_id,
        )
        self._last_boxes = new_state

        out: list[Tracklet] = []
        for i, (tid, box) in enumerate(
            zip(track_ids, det_boxes, strict=True),
        ):
            if tid < 0:
                continue
            b4 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            if tid in self.fighter_tracks:
                ft = self.fighter_tracks[tid]
                ft.last_frame = int(frame_index)
                ft.last_box = box.copy()
                ft.hits += 1
                ft.time_since_update = 0
            else:
                self.fighter_tracks[tid] = FighterTrack(
                    track_id=tid,
                    last_frame=int(frame_index),
                    last_box=box.copy(),
                    hits=1,
                    time_since_update=0,
                )
            out.append(
                Tracklet(
                    track_id=tid,
                    frame_index=int(frame_index),
                    box_xyxy=b4,
                    confidence=confidences[i] if i < len(confidences) else 1.0,
                )
            )
        return out
