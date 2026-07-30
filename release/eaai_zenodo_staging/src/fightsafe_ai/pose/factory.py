"""Config-driven construction of :class:`~fightsafe_ai.pose.backends.base.BasePoseEstimator` instances."""

from __future__ import annotations

from typing import Any

from fightsafe_ai.exceptions import ConfigurationError
from fightsafe_ai.pose.backends.base import BasePoseEstimator


def create_pose_estimator(kind: str, **kwargs: Any) -> BasePoseEstimator:
    """
    Factory for pose backends. ``kind`` is case-insensitive.

    * ``mediapipe`` (default MVP) — :class:`~fightsafe_ai.pose.backends.mediapipe_backend.MediaPipePoseBackend`
    * ``mock`` — :class:`~fightsafe_ai.pose.backends.mock_backend.MockPoseBackend`
    * ``yolo`` / ``yolo_pose`` — :class:`~fightsafe_ai.pose.backends.yolo_pose_backend.YOLOPoseBackend` (optional ``ultralytics``)
    * ``rtmpose`` / ``mmpose`` — :class:`~fightsafe_ai.pose.backends.rtmpose_backend.RTMPoseBackend` (optional ``mmpose``)
    """
    k = (kind or "mediapipe").strip().lower()
    if k in ("mediapipe", "mp", "blaze"):
        from fightsafe_ai.pose.backends.mediapipe_backend import MediaPipePoseBackend

        return MediaPipePoseBackend(
            static_image_mode=bool(kwargs.get("static_image_mode", True)),
            model_complexity=int(kwargs.get("model_complexity", 1)),
            min_detection_confidence=float(kwargs.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(kwargs.get("min_tracking_confidence", 0.5)),
            glob_patterns=kwargs.get("glob_patterns"),
        )
    if k in ("mock", "mock_backend", "dummy"):
        from fightsafe_ai.pose.backends.mock_backend import MockPoseBackend

        return MockPoseBackend(
            glob_patterns=kwargs.get("glob_patterns"),
            return_empty=bool(kwargs.get("return_empty", False)),
        )
    if k in ("yolo", "yolo_pose", "yolo-pose", "ultralytics"):
        from fightsafe_ai.pose.backends.yolo_pose_backend import YOLOPoseBackend

        return YOLOPoseBackend(
            model_name=str(kwargs.get("model_name", "yolo11n-pose.pt")),
            device=str(kwargs.get("device", "auto")),
            glob_patterns=kwargs.get("glob_patterns"),
        )
    if k in ("rtmpose", "mmpose", "rtm"):
        from fightsafe_ai.pose.backends.rtmpose_backend import RTMPoseBackend

        return RTMPoseBackend(
            pose2d=str(kwargs.get("pose2d", "rtmpose-m_8xb256-210e_coco-256x192")),
            device=str(kwargs.get("device", "auto")),
            glob_patterns=kwargs.get("glob_patterns"),
        )
    raise ConfigurationError(f"Unknown pose backend: {kind!r}")
