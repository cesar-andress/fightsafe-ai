"""Factory for :class:`~fightsafe_ai.pose.backends.pose_estimator.PoseEstimator` (live / low-latency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fightsafe_ai.pose.backends.constants import assert_valid_runtime_backend
from fightsafe_ai.pose.backends.onnx_estimator import OnnxPoseEstimator
from fightsafe_ai.pose.backends.pose_estimator import PoseEstimator
from fightsafe_ai.pose.backends.tensorrt_estimator import TensorRTPoseEstimator
from fightsafe_ai.pose.backends.torch_estimator import TorchPoseEstimator


def create_runtime_pose_estimator(kind: str, **kwargs: Any) -> PoseEstimator:
    """
    Build a runtime pose backend (backend-agnostic entry point).

    Parameters
    ----------
    kind
        ``torch`` — :class:`~fightsafe_ai.pose.backends.torch_estimator.TorchPoseEstimator` (RTMPose).
        ``onnx`` — :class:`~fightsafe_ai.pose.backends.onnx_estimator.OnnxPoseEstimator`.
        ``tensorrt`` / ``trt`` — :class:`~fightsafe_ai.pose.backends.tensorrt_estimator.TensorRTPoseEstimator`.

    kwargs
        Forwarded where applicable:

        - **torch**: ``device``, ``pose2d``, ``use_fp16``.
        - **onnx**: ``model_path`` / ``onnx_model``, ``use_fp16``, ``prefer_cuda``, ``cuda_device_id``.
        - **tensorrt**: reserved for future ``engine_path`` / TRT session options.
    """
    k = assert_valid_runtime_backend(kind)
    if k == "torch":
        return TorchPoseEstimator(**kwargs)
    if k == "onnx":
        kw = dict(kwargs)
        mp = kw.pop("model_path", None) or kw.pop("onnx_model", None)
        path = Path(mp) if mp else None
        return OnnxPoseEstimator(model_path=path, **kw)
    return TensorRTPoseEstimator(**kwargs)


__all__ = ["create_runtime_pose_estimator"]
