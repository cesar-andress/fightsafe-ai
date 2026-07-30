"""
Runtime pose API: **single-frame** inference via :meth:`predict`.

Batch/offline pipelines use :class:`~fightsafe_ai.pose.backends.base.BasePoseEstimator`
(``estimate_folder`` → CSV).

Implementations live in this package:

- :class:`~fightsafe_ai.pose.backends.torch_estimator.TorchPoseEstimator` — RTMPose / MMPose (default).
- :class:`~fightsafe_ai.pose.backends.onnx_estimator.OnnxPoseEstimator` — ONNX Runtime (optional decode).
- :class:`~fightsafe_ai.pose.backends.tensorrt_estimator.TensorRTPoseEstimator` — stub until engines wired.

Use :func:`~fightsafe_ai.pose.backends.runtime_factory.create_runtime_pose_estimator` to construct
implementations without importing backend classes at call sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from fightsafe_ai.pose.keypoints import KeypointsResult


class PoseEstimator(ABC):
    """
    Pluggable single-frame pose backend (BGR ``uint8`` OpenCV frames).

    Call :meth:`predict` only; do not depend on concrete subclasses outside ``pose.backends``.
    """

    __slots__ = ()

    @abstractmethod
    def predict(self, frame: np.ndarray) -> KeypointsResult:
        """
        Run inference on one frame.

        Parameters
        ----------
        frame
            ``HxWxC`` array, typically **BGR** ``uint8`` (OpenCV convention).

        Returns
        -------
        KeypointsResult
            Alias of :class:`~fightsafe_ai.pose.keypoints.PoseResult` (frame id + keypoints list).
        """
        ...


__all__ = ["PoseEstimator"]
