"""
Deterministic mock pose backend for **unit tests** (no MediaPipe, no network, no weights).

Emits a stable single-person stick figure in normalized coordinates so tabular tests can
exercise the pipeline without loading real models.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from fightsafe_ai.pose.backends.base import BasePoseEstimator
from fightsafe_ai.pose.blazepose import BLAZEPOSE_33
from fightsafe_ai.pose.keypoints import Keypoint, PoseResult
from fightsafe_ai.utils.sorting import natural_sort_paths


logger = logging.getLogger(__name__)

_DEFAULT_GLOBS: tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png")


def _default_keypoints() -> list[Keypoint]:
    """A fixed synthetic pose in [0,1] (deterministic, not biomechanically valid)."""
    out: list[Keypoint] = []
    for i, name in enumerate(BLAZEPOSE_33):
        t = i / 32.0
        x = 0.5 + 0.1 * float(np.sin(t * np.pi))
        y = 0.1 + 0.75 * t
        out.append(
            Keypoint(
                name=name,
                x=float(x),
                y=float(y),
                z=0.0,
                visibility=0.99,
            )
        )
    return out


class MockPoseBackend(BasePoseEstimator):
    """No-op / synthetic backend for fast tests."""

    def __init__(
        self,
        *,
        glob_patterns: Iterable[str] | None = None,
        return_empty: bool = False,
    ) -> None:
        self._glob_patterns = tuple(glob_patterns) if glob_patterns else _DEFAULT_GLOBS
        self._return_empty = return_empty

    @property
    def device_label(self) -> str:
        return "n/a (mock backend)"

    @property
    def backend_name(self) -> str:
        return "mock"

    def estimate_frame(self, image: np.ndarray) -> PoseResult:
        if self._return_empty:
            return PoseResult(frame_id="", keypoints=[])
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("image must be HxWxC with at least 3 channels (BGR).")
        _ = image.shape
        return PoseResult(frame_id="", keypoints=_default_keypoints())

    def estimate_folder(self, input_dir: Path, output_csv: Path) -> Path:
        input_dir = input_dir.expanduser().resolve()
        output_csv = output_csv.expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for pat in self._glob_patterns:
            paths.extend(sorted(input_dir.glob(pat)))
        fieldnames = ["frame_id", "keypoint_name", "x", "y", "z", "visibility"]
        kpts = [] if self._return_empty else _default_keypoints()
        with output_csv.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames)
            w.writeheader()
            for p in natural_sort_paths([x for x in paths if x.is_file()]):
                fid = p.stem
                for kp in kpts:
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
        logger.info("MockPoseBackend wrote synthetic keypoints -> %s", output_csv)
        return output_csv.resolve()
