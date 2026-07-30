"""
BlazePose backend for FightSafe (MediaPipe Tasks, TensorFlow Lite).

In **CPU mode** (this backend’s default: no GPU delegate configured), it is
**normal** to see **third-party** log lines on stderr from **TensorFlow Lite**
and **MediaPipe** while the ``PoseLandmarker`` runs, for example:

* **XNNPACK delegate** messages — TFLite reporting that the XNNPACK execution
  path is in use. These are **informational** for typical desktop runs, not
  an error from our pipeline.
* **Feedback Manager** / similar MediaPipe **runtime warnings** — often **harmless**
  noise; they do **not** by themselves mean pose outputs are wrong.
* **NORM_RECT**-related lines — can appear in relation to **ROI** / input
  projection; they **do not** by themselves prove invalid landmarks.

**FightSafe does not treat** these third-party console messages as pipeline
failures. A failed run is indicated by missing or corrupt artifacts, raised
Python exceptions, or a failing :func:`~fightsafe_ai.qa.validators.run_quality_checks`
(see ``docs/troubleshooting.md`` in the repository).
"""

from __future__ import annotations

import contextlib
import csv
import logging
import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from fightsafe_ai.pose.backends.base import BasePoseEstimator
from fightsafe_ai.pose.blazepose import BLAZEPOSE_33
from fightsafe_ai.pose.keypoints import Keypoint, PoseResult
from fightsafe_ai.utils.sorting import natural_sort_paths


logger = logging.getLogger(__name__)

_MODEL_BUNDLE_NAMES: tuple[str, str, str] = (
    "pose_landmarker_lite",
    "pose_landmarker_full",
    "pose_landmarker_heavy",
)

# Google-hosted MediaPipe model bundles (float16 / v1)
_MODEL_GCS = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/{name}/float16/1/{name}.task"
)

_DEFAULT_GLOBS: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png")


def _default_cache_dir() -> Path:
    return (
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        / "fightsafe_ai"
        / "mediapipe"
    )


def _model_bundle_for_complexity(model_complexity: int) -> str:
    """Map ``model_complexity`` 0,1,2 to Tasks bundle name."""
    if model_complexity <= 0:
        return _MODEL_BUNDLE_NAMES[0]
    if model_complexity == 1:
        return _MODEL_BUNDLE_NAMES[1]
    return _MODEL_BUNDLE_NAMES[2]


def _ensure_model_file(bundle_name: str) -> Path:
    """Return path to a ``.task`` bundle, downloading to cache on first use."""
    dest = _default_cache_dir() / f"{bundle_name}.task"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _MODEL_GCS.format(name=bundle_name)
    logger.info("Downloading MediaPipe model %s from %s", bundle_name, url)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        part = dest.with_suffix(dest.suffix + ".part")
        with part.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    part.replace(dest)
    return dest


def _mp_tasks() -> tuple[Any, Any, Any, Any]:
    from mediapipe.tasks.python import core, vision
    from mediapipe.tasks.python.vision.core import (
        image as mp_image,
        vision_task_running_mode as running_mode_lib,
    )

    return vision, mp_image, running_mode_lib, core


def _image_from_rgb(rgb: np.ndarray) -> Any:
    _, mp_image, _, _ = _mp_tasks()
    if rgb.dtype != np.uint8 or not rgb.flags.c_contiguous:
        rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return mp_image.Image(
        image_format=mp_image.ImageFormat.SRGB,
        data=rgb,
    )


def _visibility(landmark: Any) -> float:
    v = getattr(landmark, "visibility", None)
    if v is not None:
        return float(v)
    p = getattr(landmark, "presence", None)
    if p is not None:
        return float(p)
    return 0.0


@contextlib.contextmanager
def _landmarker(
    model_path: Path,
    *,
    use_video: bool,
    min_detection: float,
    min_tracking: float,
) -> Iterator[Any]:
    vision, _, running_mode_lib, core = _mp_tasks()
    rm = (
        running_mode_lib.VisionTaskRunningMode.VIDEO
        if use_video
        else running_mode_lib.VisionTaskRunningMode.IMAGE
    )
    options = vision.PoseLandmarkerOptions(
        base_options=core.base_options.BaseOptions(model_asset_path=os.fspath(model_path)),
        running_mode=rm,
        min_pose_detection_confidence=min_detection,
        min_pose_presence_confidence=min_detection,
        min_tracking_confidence=min_tracking,
    )
    with vision.PoseLandmarker.create_from_options(options) as lm:
        yield lm


def _list_images(folder: Path, patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        for p in folder.glob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                paths.append(p)
    return natural_sort_paths(paths)


def _pose_result_from_task_landmarks(
    frame_id: str,
    landmarks: list[Any] | None,
) -> PoseResult:
    """Build :class:`PoseResult` from a list of ``NormalizedLandmark``."""
    if not landmarks or len(landmarks) < 33:
        return PoseResult(frame_id=frame_id, keypoints=[])

    keypoints: list[Keypoint] = []
    for i in range(33):
        pt = landmarks[i]
        keypoints.append(
            Keypoint(
                name=BLAZEPOSE_33[i],
                x=float(pt.x) if pt.x is not None else 0.0,
                y=float(pt.y) if pt.y is not None else 0.0,
                z=float(pt.z) if pt.z is not None else 0.0,
                visibility=_visibility(pt),
            )
        )
    return PoseResult(frame_id=frame_id, keypoints=keypoints)


def _to_mp_result(frame_id: str, task_result: Any, writer: Any | None) -> int:
    """Write one pose to CSV; return the number of keypoint rows written (0 if no pose)."""
    rows = 0
    if not task_result.pose_landmarks:
        if writer is not None and frame_id:
            logger.debug("No pose for frame_id=%s", frame_id)
        return 0
    lms = task_result.pose_landmarks[0]
    if len(lms) < 33:
        return 0
    if writer is not None and frame_id:
        for i in range(33):
            pt = lms[i]
            writer.writerow(
                {
                    "frame_id": frame_id,
                    "keypoint_name": BLAZEPOSE_33[i],
                    "x": f"{float(pt.x) if pt.x is not None else 0.0:.8f}",
                    "y": f"{float(pt.y) if pt.y is not None else 0.0:.8f}",
                    "z": f"{float(pt.z) if pt.z is not None else 0.0:.8f}",
                    "visibility": f"{_visibility(pt):.8f}",
                }
            )
            rows += 1
    return rows


class MediaPipePoseBackend(BasePoseEstimator):
    """
    BlazePose via MediaPipe Tasks (``PoseLandmarker``). Default research / MVP backend.

    Uses a bundled ``.task`` model cached under the user cache directory (see
    :func:`_default_cache_dir`). CSV columns:
    ``frame_id``, ``keypoint_name``, ``x``, ``y``, ``z``, ``visibility``.

    On **CPU**, expect TensorFlow Lite / MediaPipe **console noise** (XNNPACK
    delegate, Feedback Manager, occasional NORM_RECT text); the pipeline does
    **not** use that as a success criterion—see the module docstring and
    ``docs/troubleshooting.md``.
    """

    def __init__(
        self,
        *,
        static_image_mode: bool = True,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        glob_patterns: Iterable[str] | None = None,
    ) -> None:
        self._static_image_mode = static_image_mode
        self._model_complexity = int(model_complexity)
        self._min_detection = float(min_detection_confidence)
        self._min_tracking = float(min_tracking_confidence)
        self._glob_patterns = tuple(glob_patterns) if glob_patterns else _DEFAULT_GLOBS
        self._bundle = _model_bundle_for_complexity(self._model_complexity)
        self._cpu_mode_hints_logged = False

    @property
    def device_label(self) -> str:
        return "CPU (MediaPipe Tasks default; no CUDA switch in this backend)."

    @property
    def backend_name(self) -> str:
        return "mediapipe (blazepose tasks)"

    def _get_model_path(self) -> Path:
        return _ensure_model_file(self._bundle)

    def estimate_frame(self, image: np.ndarray) -> PoseResult:
        """Return pose for one BGR image (shape H x W x C)."""
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("image must be a HxWxC array with at least 3 channels (BGR).")

        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        mpimg = _image_from_rgb(rgb)

        p = self._get_model_path()
        if not self._static_image_mode:
            with _landmarker(
                p,
                use_video=True,
                min_detection=self._min_detection,
                min_tracking=self._min_tracking,
            ) as pl:
                r = pl.detect_for_video(mpimg, timestamp_ms=0)
        else:
            with _landmarker(
                p,
                use_video=False,
                min_detection=self._min_detection,
                min_tracking=self._min_tracking,
            ) as pl:
                r = pl.detect(mpimg)

        if not r.pose_landmarks:
            logger.debug("MediaPipe Tasks returned no pose.")
            return PoseResult(frame_id="", keypoints=[])
        return _pose_result_from_task_landmarks("", r.pose_landmarks[0])

    def estimate_folder(self, input_dir: Path, output_csv: Path) -> Path:
        """Write all frames under ``input_dir`` to a single ``output_csv``."""
        input_dir = input_dir.expanduser().resolve()
        output_csv = output_csv.expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        images = _list_images(input_dir, self._glob_patterns)
        if not images:
            logger.warning("No images matched patterns %s in %s", self._glob_patterns, input_dir)
            output_csv.write_text(
                "frame_id,keypoint_name,x,y,z,visibility\n",
                encoding="utf-8",
            )
            return output_csv.resolve()

        if not self._cpu_mode_hints_logged:
            self._cpu_mode_hints_logged = True
            logger.info(
                "If TensorFlow Lite or MediaPipe print XNNPACK / Feedback Manager / "
                "NORM_RECT lines to the console during pose, that is often expected in "
                "CPU mode and is not a FightSafe pipeline failure by itself."
            )

        use_video = not self._static_image_mode
        fieldnames = ["frame_id", "keypoint_name", "x", "y", "z", "visibility"]
        rows_written = 0
        model = self._get_model_path()
        with (
            _landmarker(
                model,
                use_video=use_video,
                min_detection=self._min_detection,
                min_tracking=self._min_tracking,
            ) as pose,
            output_csv.open("w", newline="", encoding="utf-8") as fp,
        ):
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for idx, img_path in enumerate(images):
                frame_id = img_path.stem
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    logger.warning("Unreadable image skipped: %s", img_path)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mpimg = _image_from_rgb(rgb)
                if use_video:
                    ts = idx * 33
                    r = pose.detect_for_video(mpimg, timestamp_ms=ts)
                else:
                    r = pose.detect(mpimg)
                rows_written += _to_mp_result(frame_id, r, writer)

        logger.info(
            "Wrote %s keypoint rows for %s frames -> %s",
            rows_written,
            len(images),
            output_csv,
        )
        return output_csv.resolve()


# MVP stable public name
MediaPipePoseEstimator = MediaPipePoseBackend
