"""
Pluggable **runtime** pose backends (single-frame :class:`PoseEstimator`).

For **folder/batch** CSV export, see :class:`~fightsafe_ai.pose.backends.base.BasePoseEstimator`
and :func:`~fightsafe_ai.pose.factory.create_pose_estimator`.

Typical live usage::

    from fightsafe_ai.pose.backends import PoseEstimator, create_runtime_pose_estimator

    est: PoseEstimator = create_runtime_pose_estimator(
        "torch", device="cuda", pose2d=None, use_fp16=False
    )
    pose = est.predict(frame_bgr)

CLI (live runner): ``--pose-backend torch|onnx|tensorrt`` (default ``torch``).
"""

from __future__ import annotations

from fightsafe_ai.pose.backends.backend_benchmark import (
    PoseBenchmarkResult,
    run_pose_backend_benchmark,
)
from fightsafe_ai.pose.backends.base import BasePoseBackend, BasePoseEstimator
from fightsafe_ai.pose.backends.constants import (
    RUNTIME_BACKEND_CLI_CHOICES,
    assert_valid_runtime_backend,
    normalize_runtime_backend,
)
from fightsafe_ai.pose.backends.device_runtime import configure_cuda_inference, resolve_torch_device
from fightsafe_ai.pose.backends.mediapipe_backend import (
    MediaPipePoseBackend,
    MediaPipePoseEstimator,
)
from fightsafe_ai.pose.backends.mock_backend import MockPoseBackend
from fightsafe_ai.pose.backends.onnx_estimator import OnnxPoseEstimator
from fightsafe_ai.pose.backends.pose_estimator import PoseEstimator
from fightsafe_ai.pose.backends.rtmpose_backend import RTMPoseBackend
from fightsafe_ai.pose.backends.runtime_factory import create_runtime_pose_estimator
from fightsafe_ai.pose.backends.tensorrt_estimator import TensorRTPoseEstimator
from fightsafe_ai.pose.backends.torch_estimator import TorchPoseEstimator
from fightsafe_ai.pose.backends.yolo_pose_backend import YOLOPoseBackend


__all__ = [
    "RUNTIME_BACKEND_CLI_CHOICES",
    "BasePoseBackend",
    "BasePoseEstimator",
    "MediaPipePoseBackend",
    "MediaPipePoseEstimator",
    "MockPoseBackend",
    "OnnxPoseEstimator",
    "PoseBenchmarkResult",
    "PoseEstimator",
    "RTMPoseBackend",
    "TensorRTPoseEstimator",
    "TorchPoseEstimator",
    "YOLOPoseBackend",
    "assert_valid_runtime_backend",
    "configure_cuda_inference",
    "create_runtime_pose_estimator",
    "normalize_runtime_backend",
    "resolve_torch_device",
    "run_pose_backend_benchmark",
]
