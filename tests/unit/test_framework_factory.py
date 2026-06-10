"""Tests for pose backend factory (mock path avoids MediaPipe weights)."""

from __future__ import annotations

import numpy as np
import pytest

from fightsafe_ai.exceptions import ConfigurationError
from fightsafe_ai.pose.factory import create_pose_estimator


pytestmark = pytest.mark.unit


def test_create_mediapipe() -> None:
    est = create_pose_estimator("mediapipe")
    assert est.__class__.__name__ == "MediaPipePoseBackend"


def test_create_mock() -> None:
    est = create_pose_estimator("mock", return_empty=True)
    r = est.estimate_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    assert r.keypoints == []


def test_create_unknown() -> None:
    with pytest.raises(ConfigurationError):
        create_pose_estimator("not_a_backend")


def test_create_rtmpose() -> None:
    from fightsafe_ai.pose.backends.rtmpose_backend import RTMPoseBackend

    est = create_pose_estimator("rtmpose", device="cpu")
    assert isinstance(est, RTMPoseBackend)
