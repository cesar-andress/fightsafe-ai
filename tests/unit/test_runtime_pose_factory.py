"""Runtime pose estimator factory (torch | onnx | tensorrt)."""

from __future__ import annotations

import numpy as np
import pytest

from fightsafe_ai.exceptions import ConfigurationError
from fightsafe_ai.pose.backends.onnx_estimator import OnnxPoseEstimator
from fightsafe_ai.pose.backends.pose_estimator import PoseEstimator
from fightsafe_ai.pose.backends.runtime_factory import create_runtime_pose_estimator
from fightsafe_ai.pose.backends.tensorrt_estimator import TensorRTPoseEstimator
from fightsafe_ai.pose.backends.torch_estimator import TorchPoseEstimator


pytestmark = pytest.mark.unit


def test_factory_torch_is_torch_estimator() -> None:
    est = create_runtime_pose_estimator("torch", device="cpu")
    assert isinstance(est, TorchPoseEstimator)
    assert isinstance(est, PoseEstimator)


def test_factory_onnx_and_tensorrt_types() -> None:
    onnx_e = create_runtime_pose_estimator("onnx")
    assert isinstance(onnx_e, OnnxPoseEstimator)
    trt = create_runtime_pose_estimator("tensorrt")
    assert isinstance(trt, TensorRTPoseEstimator)
    trt2 = create_runtime_pose_estimator("trt")
    assert isinstance(trt2, TensorRTPoseEstimator)


def test_stub_predict_returns_pose_result() -> None:
    est = create_runtime_pose_estimator("tensorrt")
    r = est.predict(np.zeros((8, 8, 3), dtype=np.uint8))
    assert r.keypoints == []


def test_unknown_backend_raises() -> None:
    with pytest.raises(ConfigurationError):
        create_runtime_pose_estimator("mediapipe")
