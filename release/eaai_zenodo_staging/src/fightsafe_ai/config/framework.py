"""
Framework component selection (YAML or defaults).

Merges optional ``configs/framework.yaml`` with built-in defaults. Does not require network
access to import.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from fightsafe_ai.config.loader import load_yaml_file, merge_dicts


DEFAULT_FRAMEWORK: dict[str, Any] = {
    "pose": {
        "backend": "mediapipe",
        "device": "auto",
        "yolo": {
            "model": "yolo11n-pose.pt",
        },
        "rtmpose": {
            "pose2d": "rtmpose-m_8xb256-210e_coco-256x192",
        },
    },
    "tracking": {
        "enabled": False,
    },
    "action": {
        "enabled": False,
    },
    "anomaly": {
        "fall_detection": True,
        "inactivity": True,
    },
    "llm": {
        "explainability": "optional",
    },
}

DEFAULT_FRAMEWORK_REL = Path("configs") / "framework.yaml"


def load_framework_config(
    path: Path | None = None,
) -> dict[str, Any]:
    """
    Return merged framework config. If ``path`` is omitted, tries ``configs/framework.yaml``
    next to the current working directory; otherwise returns defaults only.
    """
    out: dict[str, Any] = copy.deepcopy(DEFAULT_FRAMEWORK)
    candidate = path or (Path.cwd() / DEFAULT_FRAMEWORK_REL)
    if candidate.is_file():
        data = load_yaml_file(candidate)
        if isinstance(data, dict):
            return merge_dicts(out, data)
    return out


def pose_backend_name(framework: dict[str, Any] | None) -> str:
    """``pose.backend`` string, defaulting to ``mediapipe``."""
    if not framework:
        return "mediapipe"
    pose = framework.get("pose")
    if isinstance(pose, dict) and "backend" in pose:
        return str(pose["backend"])
    return "mediapipe"


def pose_init_kwargs_for_backend(framework: dict[str, Any] | None, backend: str) -> dict[str, Any]:
    """
    Build keyword arguments for :func:`fightsafe_ai.pose.factory.create_pose_estimator`
    from ``configs/framework.yaml`` (``pose.device``, nested ``yolo`` / ``rtmpose`` blocks).
    """
    b = (backend or "mediapipe").strip().lower()
    if not framework:
        return {}
    pose = framework.get("pose")
    if not isinstance(pose, dict):
        return {}
    out: dict[str, Any] = {}
    dev = pose.get("device", "auto")
    if b in (
        "yolo",
        "yolo_pose",
        "yolo-pose",
        "ultralytics",
        "rtmpose",
        "mmpose",
        "rtm",
    ):
        out["device"] = str(dev)
    if b in ("yolo", "yolo_pose", "yolo-pose", "ultralytics"):
        yo = pose.get("yolo")
        if isinstance(yo, dict) and "model" in yo:
            out["model_name"] = str(yo["model"])
    if b in ("rtmpose", "mmpose", "rtm"):
        rt = pose.get("rtmpose")
        if isinstance(rt, dict) and "pose2d" in rt:
            out["pose2d"] = str(rt["pose2d"])
    return out
