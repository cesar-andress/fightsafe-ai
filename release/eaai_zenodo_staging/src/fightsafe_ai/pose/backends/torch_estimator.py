"""PyTorch pose backend (RTMPose / MMPose stack via :class:`~fightsafe_ai.pose.backends.rtmpose_backend.RTMPoseBackend`)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fightsafe_ai.pose.backends.device_runtime import configure_cuda_inference, resolve_torch_device
from fightsafe_ai.pose.backends.pose_estimator import PoseEstimator
from fightsafe_ai.pose.backends.rtmpose_backend import RTMPoseBackend
from fightsafe_ai.pose.keypoints import KeypointsResult


class TorchPoseEstimator(PoseEstimator):
    """
    Torch runtime: delegates to :class:`~fightsafe_ai.pose.backends.rtmpose_backend.RTMPoseBackend`.

    Extra kwargs accepted for API symmetry but only ``device`` / ``pose2d`` are forwarded.
    """

    __slots__ = ("_backend",)

    def __init__(
        self,
        *,
        device: str = "auto",
        pose2d: str | None = None,
        use_fp16: bool = False,
        **kwargs: Any,
    ) -> None:
        _ = kwargs  # static_image_mode et al. — not used by RTMPose
        if resolve_torch_device(device).startswith("cuda"):
            configure_cuda_inference()
        rtm_kw: dict[str, Any] = {"device": device, "use_fp16": bool(use_fp16)}
        if pose2d is not None:
            rtm_kw["pose2d"] = pose2d
        self._backend = RTMPoseBackend(**rtm_kw)

    def predict(self, frame: np.ndarray) -> KeypointsResult:
        return self._backend.estimate_frame(frame)


__all__ = ["TorchPoseEstimator"]
