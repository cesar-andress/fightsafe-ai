"""
Box **IoU** and **greedy identity** assignment (temporal continuity by reusing track IDs).

`assign_identities_greedy_iou` is the MVP policy; **ByteTrack** / **DeepSORT** / **SportsMOT** /
**SportSORT** replace this with stronger motion and appearance models while still emitting
:class:`~fightsafe_ai.tracking.tracklet.Tracklet` rows.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "assign_identities_greedy_iou",
    "box_iou_xyxy",
    "format_track_label",
    "merge_track_id_sets",
]


def format_track_label(track_id: int) -> str:
    """Return a human-readable label for a track id (decision-support UI)."""
    return f"Fighter {int(track_id) + 1}"


def merge_track_id_sets(a: set[int], b: set[int]) -> set[int]:
    """Set union for active-id bookkeeping (pure)."""
    return set(a) | set(b)


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoU of two axis-aligned boxes ``[x1,y1,x2,y2]``.

    Clamps degenerate box areas. Returns 0.0 for no overlap.
    """
    a = np.asarray(a, dtype=np.float64).ravel()[:4]
    b = np.asarray(b, dtype=np.float64).ravel()[:4]
    if a.size < 4 or b.size < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def assign_identities_greedy_iou(
    prev_id_to_box: dict[int, np.ndarray],
    det_boxes: list[np.ndarray],
    *,
    iou_threshold: float,
    next_id: int,
) -> tuple[list[int], int, dict[int, np.ndarray]]:
    """
    For each detection **in order**, link an **unused** previous track with best IoU, or spawn
    a new id when below threshold (simple temporal association).

    Parameters
    ----------
    prev_id_to_box
        Map ``track_id`` → last frame's box, each length-4 **xyxy** float64.
    det_boxes
        Same-order boxes for the current frame.
    iou_threshold
        Match threshold in ``[0,1]`` (MVP; tune for resolution and sport).
    next_id
        Monotonic id counter for new tracks (caller-owned).

    Returns
    -------
    track_ids
        One id per **valid** box in ``det_boxes`` (same length).
    next_id
        Updated id counter.
    new_state
        New ``track_id`` → box map for the **current** frame to use as
        ``prev_id_to_box`` on the next call.
    """
    new_state: dict[int, np.ndarray] = {}
    track_ids: list[int] = []
    for det in det_boxes:
        d = np.asarray(det, dtype=np.float64).ravel()[:4]
        if d.size < 4:
            track_ids.append(-1)
            continue
        best_tid: int | None = None
        best_iou = 0.0
        for tid, prev in prev_id_to_box.items():
            if tid in new_state:
                continue
            j = box_iou_xyxy(prev, d)
            if j > best_iou:
                best_iou = j
                best_tid = tid
        if best_tid is not None and best_iou >= float(iou_threshold):
            tid = int(best_tid)
        else:
            tid = int(next_id)
            next_id += 1
        new_state[tid] = d.copy()
        track_ids.append(tid)
    return track_ids, next_id, new_state
