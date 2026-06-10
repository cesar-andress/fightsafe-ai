"""Unit tests for :mod:`fightsafe_ai.live.live_pipeline` with mocked pose backend."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from fightsafe_ai.live.live_pipeline import LiveFrameResult, LivePipeline, LivePipelineConfig
from fightsafe_ai.pose.keypoints import Keypoint, PoseResult


pytestmark = pytest.mark.unit


def test_live_pipeline_config_defaults() -> None:
    cfg = LivePipelineConfig()
    assert cfg.pose_backend == "torch"
    assert cfg.video_fps == 30.0


def test_process_frame_returns_empty_when_no_keypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_estimator = MagicMock()
    fake_estimator.predict = MagicMock(return_value=PoseResult(frame_id="", keypoints=[]))

    monkeypatch.setattr(
        "fightsafe_ai.live.live_pipeline.create_runtime_pose_estimator",
        lambda *args, **kwargs: fake_estimator,
    )

    pipe = LivePipeline(
        LivePipelineConfig(
            video_fps=30.0, buffer_seconds=0.5, smooth_seconds=0.2, max_infer_hz=60.0
        )
    )
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    out: LiveFrameResult = pipe.process_frame(frame, 0.0, frame_index=0)
    assert out["events"] == []
    assert out["pose"] is None
    fake_estimator.predict.assert_called()


def test_process_frame_with_mock_pose_advances_last_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke path: single landmark keeps buffers alive for downstream stages."""

    kp = Keypoint("nose", 0.5, 0.5, 0.0, 1.0)

    def _predict(_frame: np.ndarray) -> PoseResult:
        return PoseResult(frame_id="", keypoints=[kp])

    fake_estimator = MagicMock()
    fake_estimator.predict = _predict

    monkeypatch.setattr(
        "fightsafe_ai.live.live_pipeline.create_runtime_pose_estimator",
        lambda *args, **kwargs: fake_estimator,
    )

    pipe = LivePipeline(
        LivePipelineConfig(
            video_fps=30.0,
            buffer_seconds=2.0,
            smooth_seconds=0.5,
            max_infer_hz=60.0,
            rolling_window=3,
        )
    )
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    last: LiveFrameResult | None = None
    for i in range(25):
        last = pipe.process_frame(frame, i / 30.0, frame_index=i)

    assert last is not None
    assert last["pose"] is not None
    assert pipe.last_pose is not None
    assert len(pipe.last_pose.keypoints) >= 1
