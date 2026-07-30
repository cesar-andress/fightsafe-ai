"""
RTMPose / MMPose backend (optional ``mmpose`` + ``torch``).

This is an **optional** integration: the core package does not depend on OpenMMLab.
When ``mmpose`` is not installed, the backend logs a clear message and writes **empty**
keypoints (same contract as :class:`~fightsafe_ai.pose.backends.mock_backend.MockPoseBackend`
with ``return_empty=True``) so CI and unit tests stay lightweight.

When ``mmpose`` is available, uses :class:`MMPoseInferencer` (MMPose 1.x) on each frame
with the configured ``pose2d`` model alias, honouring ``device`` (``auto`` → CUDA if available).
"""

from __future__ import annotations

import contextlib
import csv
import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fightsafe_ai.pose.backends.base import BasePoseEstimator
from fightsafe_ai.pose.backends.device_runtime import configure_cuda_inference, resolve_torch_device
from fightsafe_ai.pose.backends.mock_backend import MockPoseBackend
from fightsafe_ai.pose.keypoints import Keypoint, PoseResult
from fightsafe_ai.utils.sorting import natural_sort_paths


logger = logging.getLogger(__name__)

try:  # pragma: no cover
    import importlib.util

    _HAS_MMPOSE = importlib.util.find_spec("mmpose") is not None
except (ImportError, OSError, ValueError):
    _HAS_MMPOSE = False

# COCO-17 (same as YOLO backend) for a stable column namespace when MMPose returns COCO keypoints.
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


def _pose_result_from_mmpose(predictions: Any) -> list[Keypoint]:
    """
    Best-effort parse of MMPose prediction objects (structure varies by version).
    Expects a list of x,y in image space or [0,1] for the first person, length >= 17.
    """
    if predictions is None:
        return []
    # Common: list of instance dicts with 'keypoints' ndarray (K,2) or (K,3)
    try:
        if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
            inst = predictions[0]
            if isinstance(inst, dict) and "keypoints" in inst:
                k = np.asarray(inst["keypoints"])
            else:
                k = np.asarray(inst)
        else:
            k = np.asarray(getattr(predictions, "keypoints", predictions))
    except (TypeError, ValueError, AttributeError):
        return []
    if k.size < 34:  # need at least 17*2
        return []
    k = k.reshape(-1, 2) if k.ndim >= 2 else k.reshape(17, 2)
    out: list[Keypoint] = []
    for i in range(min(17, len(k))):
        out.append(
            Keypoint(
                name=_COCO17[i],
                x=float(k[i, 0]),
                y=float(k[i, 1]),
                z=0.0,
                visibility=1.0,
            )
        )
    return out


class RTMPoseBackend(BasePoseEstimator):
    """
    RTMPose via MMPose inferencer when the optional stack is present.

    Parameters
    ----------
    pose2d
        Model alias accepted by :class:`MMPoseInferencer` (e.g. a RTMPose COCO config name).
    device
        ``auto`` | ``cpu`` | ``cuda`` | ``cuda:0`` | ``mps`` (Torch semantics).
    """

    def __init__(
        self,
        *,
        pose2d: str = "rtmpose-m_8xb256-210e_coco-256x192",
        device: str = "auto",
        use_fp16: bool = False,
        glob_patterns: Iterable[str] | None = None,
    ) -> None:
        self._pose2d = str(pose2d)
        self._device_str = resolve_torch_device(device)
        self._use_fp16 = bool(use_fp16)
        if self._device_str.startswith("cuda"):
            self._device_label: str = f"CUDA ({self._device_str})"
        elif self._device_str == "mps":
            self._device_label = "MPS (Apple)"
        else:
            self._device_label = "CPU"
        self._inferencer: Any = None
        self._glob_patterns = tuple(glob_patterns) if glob_patterns else _DEFAULT_GLOBS
        if self._device_str.startswith("cuda"):
            configure_cuda_inference()
        if _HAS_MMPOSE:
            try:
                from mmpose.apis import MMPoseInferencer

                self._inferencer = MMPoseInferencer(
                    pose2d=self._pose2d,
                    device=self._device_str,
                )
            except (ImportError, OSError, ValueError, RuntimeError) as e:
                logger.warning(
                    "MMPose RTMPose inferencer could not be initialised (%s). "
                    "Falling back to empty keypoints. Check mmpose/mmengine/mmcv versions.",
                    e,
                )
                self._inferencer = None
        else:
            logger.info(
                "mmpose is not installed: RTMPose backend is unavailable. "
                "Install the optional OpenMMLab stack (e.g. mmpose, mmengine) or use 'mediapipe' / 'yolo'."
            )
        self._mock = MockPoseBackend(glob_patterns=glob_patterns, return_empty=True)

    @property
    def device_label(self) -> str:
        return self._device_label

    @property
    def backend_name(self) -> str:
        return "rtmpose (mmpose)"

    def _infer_keypoints_bgr(self, bgr: np.ndarray) -> list[Keypoint]:
        if self._inferencer is None or bgr.ndim != 3:
            return []
        tmp_path: Path | None = None
        try:
            try:
                import torch
            except ImportError:
                torch = None

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            use_amp = bool(
                self._use_fp16 and self._device_str.startswith("cuda") and torch is not None
            )

            def _run_infer(fn: Any) -> Any:
                if torch is None:
                    return fn()
                amp: Any = (
                    torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
                    if use_amp
                    else contextlib.nullcontext()
                )
                with torch.inference_mode(), amp:
                    return fn()

            out = None
            for call in (
                lambda: self._inferencer(rgb, return_datasamples=False),
                lambda: self._inferencer({"img": rgb}, return_datasamples=False),
            ):
                try:
                    out = _run_infer(call)
                    break
                except (TypeError, ValueError, RuntimeError, AttributeError):
                    continue
            if out is None:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                cv2.imwrite(str(tmp_path), bgr)
                out = _run_infer(lambda: self._inferencer(str(tmp_path), return_datasamples=False))

            if isinstance(out, dict) and "predictions" in out:
                preds = out["predictions"]
            else:
                preds = out
            kps = _pose_result_from_mmpose(preds)
            h, w = bgr.shape[0], bgr.shape[1]
            norm: list[Keypoint] = []
            for kp in kps:
                x, y = float(kp.x), float(kp.y)
                if x > 1.0 or y > 1.0:
                    x, y = x / float(w), y / float(h)
                norm.append(
                    Keypoint(
                        name=kp.name,
                        x=x,
                        y=y,
                        z=kp.z,
                        visibility=kp.visibility,
                    )
                )
            return norm
        except (OSError, ValueError, TypeError, RuntimeError) as e:
            logger.debug("MMPose inferencer failed for one frame: %s", e)
            return []
        finally:
            if tmp_path is not None and tmp_path.is_file():
                with contextlib.suppress(OSError):
                    tmp_path.unlink()

    def estimate_frame(self, image: np.ndarray) -> PoseResult:
        if self._inferencer is None:
            return self._mock.estimate_frame(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("image must be HxWxC with at least 3 channels (BGR).")
        k = self._infer_keypoints_bgr(image)
        return PoseResult(frame_id="", keypoints=k)

    def estimate_folder(self, input_dir: Path, output_csv: Path) -> Path:
        if self._inferencer is None:
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
                for kp in r.keypoints:
                    w.writerow(
                        {
                            "frame_id": p.stem,
                            "keypoint_name": kp.name,
                            "x": f"{kp.x:.8f}",
                            "y": f"{kp.y:.8f}",
                            "z": f"{kp.z:.8f}",
                            "visibility": f"{kp.visibility:.8f}",
                        }
                    )
        logger.info("RTMPoseBackend wrote keypoints -> %s", output_csv)
        return output_csv.resolve()
