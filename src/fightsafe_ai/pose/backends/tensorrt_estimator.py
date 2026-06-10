"""TensorRT pose backend (stub — integrate ``tensorrt`` + engine path when assets exist)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from fightsafe_ai.pose.backends.pose_estimator import PoseEstimator
from fightsafe_ai.pose.keypoints import KeypointsResult, PoseResult


logger = logging.getLogger(__name__)


class TensorRTPoseEstimator(PoseEstimator):
    """
    Placeholder for TensorRT-accelerated inference.

    Does not load ``.engine`` / ``.plan`` assets yet; returns an empty
    :class:`~fightsafe_ai.pose.keypoints.PoseResult`. Reserved kwargs: ``engine_path``.
    """

    __slots__ = ("_engine_path",)

    def __init__(self, *, engine_path: str | None = None, **kwargs: Any) -> None:
        _ = kwargs
        self._engine_path = engine_path

    def predict(self, frame: np.ndarray) -> KeypointsResult:
        logger.debug("TensorRTPoseEstimator stub: returning empty pose.")
        _ = frame
        return PoseResult(frame_id="", keypoints=[])


__all__ = ["TensorRTPoseEstimator"]
