"""
Pluggable **pose** backends and CSV export (BlazePose via MediaPipe by default).
"""

from __future__ import annotations

from fightsafe_ai.pose.backends.constants import RUNTIME_BACKEND_CLI_CHOICES
from fightsafe_ai.pose.backends.mediapipe_backend import (
    MediaPipePoseBackend,
    MediaPipePoseEstimator,
)
from fightsafe_ai.pose.backends.pose_estimator import PoseEstimator
from fightsafe_ai.pose.backends.runtime_factory import create_runtime_pose_estimator
from fightsafe_ai.pose.base import BasePoseEstimator
from fightsafe_ai.pose.factory import create_pose_estimator
from fightsafe_ai.pose.keypoints import Keypoint, KeypointsResult, PoseResult


__all__ = [
    "RUNTIME_BACKEND_CLI_CHOICES",
    "BasePoseEstimator",
    "Keypoint",
    "KeypointsResult",
    "MediaPipePoseBackend",
    "MediaPipePoseEstimator",
    "PoseEstimator",
    "PoseResult",
    "create_pose_estimator",
    "create_runtime_pose_estimator",
]
