"""
YOLO / Ultralytics **pose** backend (optional ``ultralytics`` + ``torch``).

Uses GPU when available and ``device`` is ``auto`` (see :func:`~fightsafe_ai.pose.backends.device_runtime.resolve_torch_device`).
If ``ultralytics`` is not installed, operations degrade gracefully: empty keypoints and a
clear log message (no hard dependency in this package).
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fightsafe_ai.pose.backends.base import BasePoseEstimator
from fightsafe_ai.pose.backends.device_runtime import resolve_torch_device
from fightsafe_ai.pose.backends.mock_backend import MockPoseBackend
from fightsafe_ai.pose.keypoints import Keypoint, PoseResult
from fightsafe_ai.utils.sorting import natural_sort_paths


logger = logging.getLogger(__name__)

try:  # pragma: no cover
    import importlib.util

    _HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None
except (ImportError, OSError, ValueError):
    _HAS_ULTRALYTICS = False

# COCO 17 (person) — names for tabular output (not BlazePose-33; downstream may treat as soft).
_COCO17: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_DEFAULT_GLOBS: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png")


def _keypoints_from_ultralytics_result(
    result: Any,
    image: np.ndarray,
) -> list[Keypoint]:
    """Extract the first detected person's COCO-17 keypoints, normalized to [0, 1]."""
    h, w = int(image.shape[0]), int(image.shape[1])
    if h <= 0 or w <= 0:
        return []
    if not hasattr(result, "keypoints") or result.keypoints is None:
        return []
    kp = result.keypoints
    t_xyn = getattr(kp, "xyn", None)
    if t_xyn is not None and len(t_xyn) > 0:
        t = t_xyn[0]
        out: list[Keypoint] = []
        for i in range(min(17, len(t))):
            out.append(
                Keypoint(
                    name=_COCO17[i],
                    x=float(t[i][0]),
                    y=float(t[i][1]),
                    z=0.0,
                    visibility=1.0,
                )
            )
        return out
    t_xy = getattr(kp, "xy", None)
    if t_xy is not None and len(t_xy) > 0:
        t = t_xy[0]
        out2: list[Keypoint] = []
        for i in range(min(17, len(t))):
            out2.append(
                Keypoint(
                    name=_COCO17[i],
                    x=float(t[i][0]) / float(w),
                    y=float(t[i][1]) / float(h),
                    z=0.0,
                    visibility=1.0,
                )
            )
        return out2
    return []


class YOLOPoseBackend(BasePoseEstimator):
    """
    YOLOv8/11-pose (Ultralytics) with optional GPU acceleration.

    * If ``ultralytics`` and ``torch`` are available, loads ``model_name`` and runs
      per-frame / per-folder pose, mapping COCO-17 to tabular keypoints.
    * If not, returns empty keypoints and logs a one-line reason (tests stay offline).
    """

    def __init__(
        self,
        *,
        model_name: str = "yolo11n-pose.pt",
        device: str = "auto",
        glob_patterns: Iterable[str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device_str = resolve_torch_device(device)
        self._device_label: str
        if self._device_str.startswith("cuda"):
            self._device_label = f"CUDA ({self._device_str})"
        elif self._device_str == "mps":
            self._device_label = "MPS (Apple)"
        else:
            self._device_label = "CPU"
        self._engine: Any = None
        self._glob_patterns = tuple(glob_patterns) if glob_patterns else _DEFAULT_GLOBS
        if _HAS_ULTRALYTICS:
            try:
                from ultralytics import YOLO  # type: ignore[import-not-found]

                self._engine = YOLO(model_name)
                if self._device_str != "cpu":
                    try:
                        to = getattr(self._engine, "to", None)
                        if callable(to):
                            to(self._device_str)
                    except (OSError, RuntimeError, ValueError) as e:
                        logger.warning(
                            "Could not move YOLO model to %s: %s; using CPU.", self._device_str, e
                        )
            except (OSError, ValueError) as e:
                logger.error("Ultralytics YOLO model load failed: %s", e)
                self._engine = None
        else:
            logger.info(
                "ultralytics is not installed: YOLO pose backend is unavailable. "
                "Install the optional 'ultralytics' (and torch) package to enable this backend, "
                "or keep the default MediaPipe backend."
            )
        self._mock = MockPoseBackend(glob_patterns=glob_patterns, return_empty=True)

    @property
    def device_label(self) -> str:
        return self._device_label

    @property
    def backend_name(self) -> str:
        return "yolo-pose (ultralytics)"

    def estimate_frame(self, image: np.ndarray) -> PoseResult:
        if self._engine is None:
            return self._mock.estimate_frame(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("image must be HxWxC with at least 3 channels (BGR).")
        results = self._engine.predict(
            source=image,
            device=self._device_str,
            verbose=False,
        )
        if not results:
            return PoseResult(frame_id="", keypoints=[])
        k = _keypoints_from_ultralytics_result(results[0], image)
        return PoseResult(frame_id="", keypoints=k)

    def estimate_folder(self, input_dir: Path, output_csv: Path) -> Path:
        if self._engine is None:
            return self._mock.estimate_folder(input_dir, output_csv)
        input_dir = input_dir.expanduser().resolve()
        output_csv = output_csv.expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for pat in self._glob_patterns:
            paths.extend(sorted(input_dir.glob(pat)))
        paths = natural_sort_paths([p for p in paths if p.is_file()])
        fieldnames = ["frame_id", "keypoint_name", "x", "y", "z", "visibility"]
        with output_csv.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            for p in paths:
                im = cv2.imread(str(p))
                if im is None:
                    continue
                r = self.estimate_frame(im)
                fid = p.stem
                for kp in r.keypoints:
                    w.writerow(
                        {
                            "frame_id": fid,
                            "keypoint_name": kp.name,
                            "x": f"{kp.x:.8f}",
                            "y": f"{kp.y:.8f}",
                            "z": f"{kp.z:.8f}",
                            "visibility": f"{kp.visibility:.8f}",
                        }
                    )
        logger.info("YOLOPoseBackend wrote keypoints -> %s", output_csv)
        return output_csv.resolve()
