"""
Abstract base for interchangeable pose estimation backends (BlazePose, YOLO-pose, mocks).

Decision-support use only. Implementations should emit stable, lower_snake_case landmark
names for downstream feature and risk tables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from fightsafe_ai.pose.keypoints import PoseResult


class BasePoseEstimator(ABC):
    """
    Interface for frame-wise pose estimation and batch export to CSV.

    Implementations should populate :class:`~fightsafe_ai.pose.keypoints.PoseResult`
    with stable landmark naming so tabular pipelines can join features across frames.
    """

    @abstractmethod
    def estimate_frame(self, image: np.ndarray) -> PoseResult:
        """
        Run pose estimation on a single image array.

        Parameters
        ----------
        image
            HWC array (typically **BGR** uint8 from OpenCV). Implementations may convert
            to RGB internally.

        Returns
        -------
        PoseResult
            ``frame_id`` may be a placeholder (e.g. empty string) when not tied to a file;
            ``keypoints`` lists all detected landmarks (possibly empty if no pose).
        """
        ...

    @abstractmethod
    def estimate_folder(self, input_dir: Path, output_csv: Path) -> Path:
        """
        Process all supported images under ``input_dir`` and write **one** consolidated CSV.

        The CSV must include at minimum:
        ``frame_id``, ``keypoint_name``, ``x``, ``y``, ``z``, ``visibility`` for
        the standard FightSafe long format.

        Parameters
        ----------
        input_dir
            Directory containing raster frames (e.g. ``*.jpg``).
        output_csv
            Destination file path; parent directories are created if missing.

        Returns
        -------
        Path
            Resolved path to the written CSV.
        """
        ...


# Alias for new code that prefers the name *backend*.
BasePoseBackend = BasePoseEstimator

__all__ = ["BasePoseBackend", "BasePoseEstimator"]
