"""Live pipeline integration with BoxingVI-style strike detector (heuristic)."""

from __future__ import annotations

import numpy as np
import pytest

from fightsafe_ai.live.live_pipeline import LivePipeline, LivePipelineConfig
from fightsafe_ai.pose.keypoints import Keypoint, PoseResult


pytestmark = pytest.mark.unit


def test_pose_to_coco17_xy_maps_named_joints() -> None:
    pose = PoseResult(
        frame_id="",
        keypoints=[
            Keypoint(name="left_wrist", x=0.1, y=0.2),
            Keypoint(name="right_wrist", x=0.9, y=0.8),
            Keypoint(name="nose", x=0.5, y=0.15),
        ],
    )
    lp = LivePipeline(LivePipelineConfig(video_fps=30.0))
    xy = lp._pose_to_coco17_xy(pose)
    assert xy.shape == (17, 2)
    assert np.isfinite(xy[9]).all() and float(xy[9, 0]) == pytest.approx(0.1)
    assert np.isfinite(xy[10]).all() and float(xy[10, 0]) == pytest.approx(0.9)


def test_effective_strike_percentile_follows_sensitivity() -> None:
    hi = LivePipeline(LivePipelineConfig(strike_percentile=85.0, live_sensitivity="high"))
    assert hi._effective_strike_percentile() == pytest.approx(75.0)
    lo = LivePipeline(LivePipelineConfig(strike_percentile=85.0, live_sensitivity="low"))
    assert lo._effective_strike_percentile() == pytest.approx(92.0)
    md = LivePipeline(LivePipelineConfig(strike_percentile=88.0, live_sensitivity="medium"))
    assert md._effective_strike_percentile() == pytest.approx(88.0)


def test_strike_dict_maps_to_safety_event() -> None:
    lp = LivePipeline(LivePipelineConfig(video_fps=30.0))
    row = {
        "start_time": 1.0,
        "end_time": 1.15,
        "start_frame": "30",
        "end_frame": "34",
        "score": 0.91,
        "level": "HIGH",
        "title": "Strike candidate",
        "description": "test desc",
    }
    ev = lp._strike_dict_to_event(row, (1.0, 1.15))
    assert ev.category.value == "impact"
    assert ev.level.value == "HIGH"
    assert ev.start_time == 1.0 and ev.end_time == 1.15
    assert "boxingvi_strike_" in ev.event_type
    assert ev.source == "boxingvi.strike_detector"
