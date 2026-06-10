"""Synthetic bounding-box tests for IoU identity and :class:`SportsTracker`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fightsafe_ai.risk.scorer import COL_FIGHTER_ID, compute_interpretable_risk
from fightsafe_ai.tracking import (
    SportsTracker,
    assign_identities_greedy_iou,
    box_iou_xyxy,
)
from fightsafe_ai.tracking.base import BaseTracker
from fightsafe_ai.tracking.tracklet import FighterTrack, Tracklet


pytestmark = pytest.mark.unit


def test_box_iou_overlap() -> None:
    a = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    b = np.array([5.0, 5.0, 15.0, 15.0], dtype=np.float64)
    iou = box_iou_xyxy(a, b)
    # Intersection 5x5=25, union=100+100-25=175
    assert abs(iou - 25.0 / 175.0) < 1e-6


def test_box_iou_disjoint() -> None:
    a = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    b = np.array([5.0, 5.0, 6.0, 6.0], dtype=np.float64)
    assert box_iou_xyxy(a, b) == 0.0


def test_assign_identities_new_tracks() -> None:
    prev: dict[int, np.ndarray] = {}
    dets = [
        np.array([0, 0, 10, 10], dtype=np.float64),
        np.array([50, 50, 60, 60], dtype=np.float64),
    ]
    tids, nid, st = assign_identities_greedy_iou(prev, dets, iou_threshold=0.5, next_id=0)
    assert tids == [0, 1] and nid == 2
    assert set(st.keys()) == {0, 1}


def test_assign_identities_continuity() -> None:
    b0 = np.array([0, 0, 10, 10], dtype=np.float64)
    tids0, n0, _prev = assign_identities_greedy_iou({}, [b0], iou_threshold=0.3, next_id=0)
    assert tids0[0] == 0
    b1 = np.array([0.1, 0.1, 10.1, 10.1], dtype=np.float64)
    tids1, n1, _ = assign_identities_greedy_iou(
        {tids0[0]: b0},
        [b1],
        iou_threshold=0.3,
        next_id=n0,
    )
    assert tids1[0] == 0
    assert n1 == n0  # no new id


def test_sports_tracker_two_moving_fighters() -> None:
    tr = SportsTracker(iou_threshold=0.4)
    # Frame 0: two boxes
    t0 = tr.update(
        0,
        [
            {"box_xyxy": [0, 0, 10, 10], "confidence": 0.9},
            {"box_xyxy": [100, 100, 120, 120], "confidence": 0.85},
        ],
    )
    assert len(t0) == 2
    assert {x.track_id for x in t0} == {0, 1}
    # Small motion
    t1 = tr.update(
        1,
        [
            {"box_xyxy": [1, 1, 11, 11]},
            {"box_xyxy": [100, 100, 120, 120]},  # track 1 static
        ],
    )
    ids1 = {x.track_id: x for x in t1}
    assert ids1[0].box_xyxy is not None
    # Same ids
    assert {x.track_id for x in t1} == {0, 1}
    assert 0 in tr.fighter_tracks and 1 in tr.fighter_tracks
    assert isinstance(tr.fighter_tracks[0], FighterTrack)
    tr.reset()
    assert tr.fighter_tracks == {}


def test_fighter_track_to_tracklet() -> None:
    ft = FighterTrack(
        track_id=2,
        last_frame=0,
        last_box=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
    )
    tl = ft.to_tracklet(1)
    assert isinstance(tl, Tracklet) and tl.fighter_id == 2
    assert tl.frame_index == 1


def test_risk_preserves_fighter_id() -> None:
    from fightsafe_ai.risk.rules import load_interpretable_risk_config

    cfg = load_interpretable_risk_config(None)
    df = pd.DataFrame(
        {
            "frame_id": [0, 1, 0, 1],
            COL_FIGHTER_ID: [0, 0, 1, 1],
            "large_torso_angle_01": [0.0, 0.1, 0.0, 0.0],
        }
    )
    out = compute_interpretable_risk(df, config=cfg, pose_per_frame=None)
    assert COL_FIGHTER_ID in out.columns
    assert out[COL_FIGHTER_ID].tolist() == [0, 0, 1, 1]


def test_base_tracker_default_reset_is_noop() -> None:
    class _DummyTracker(BaseTracker):
        def update(self, frame_index: int, detections: list[dict[str, Any]]) -> list[Tracklet]:
            return []

    t = _DummyTracker()
    t.reset()
