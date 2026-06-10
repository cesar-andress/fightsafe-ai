"""Fighter tracking: IoU-MVP, identity helpers, and extension points for MOT backends."""

from fightsafe_ai.tracking.base import BaseTracker
from fightsafe_ai.tracking.identity import (
    assign_identities_greedy_iou,
    box_iou_xyxy,
    format_track_label,
    merge_track_id_sets,
)
from fightsafe_ai.tracking.sports_tracker import SportsTracker
from fightsafe_ai.tracking.tracklet import FighterTrack, Tracklet


__all__ = [
    "BaseTracker",
    "FighterTrack",
    "SportsTracker",
    "Tracklet",
    "assign_identities_greedy_iou",
    "box_iou_xyxy",
    "format_track_label",
    "merge_track_id_sets",
]
