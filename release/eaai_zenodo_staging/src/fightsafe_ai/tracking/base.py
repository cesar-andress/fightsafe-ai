"""
Abstract **multi-target tracker** interface (fighter / athlete handles before risk).

**Future integration (swap ``update`` body, keep :class:`Tracklet` output):**

- *SportsMOT* / multi-object sports benchmarks — feed MOT-style state into your adapter.
- **ByteTrack** — two-stage high/low + Kalman; map ``track_id`` to :class:`Tracklet`.
- **DeepSORT** / StrongSORT — add appearance embedding; same emission API.
- **SportSORT** (and variants) — use their association, then wrap rows as :class:`Tracklet`.

The FightSafe default :class:`SportsTracker` is a **greedy IoU** baseline only, not
broadcast-grade tracking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fightsafe_ai.tracking.tracklet import Tracklet


class BaseTracker(ABC):
    """
    Map per-frame **detections** to stable **track** ids; does not assign match outcomes.
    """

    @abstractmethod
    def update(self, frame_index: int, detections: list[dict[str, Any]]) -> list[Tracklet]:
        """
        Parameters
        ----------
        frame_index
            Monotonic index (0-based).
        detections
            Each item may include ``"box_xyxy"``: ``[x1,y1,x2,y2]`` in pixel or normalized
            coordinates, consistent across the clip. Optional: ``"confidence"``, ``"class_id"``.
        """
        ...

    def reset(self) -> None:
        """Optional: clear state between videos (MVP: override in concrete trackers)."""
        return
