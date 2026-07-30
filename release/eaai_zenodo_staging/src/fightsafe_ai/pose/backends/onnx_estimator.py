"""ONNX Runtime pose backend with optional CUDA EP and FP16 I/O (decode wiring remains model-specific)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fightsafe_ai.pose.backends.pose_estimator import PoseEstimator
from fightsafe_ai.pose.keypoints import KeypointsResult, PoseResult


logger = logging.getLogger(__name__)


class OnnxPoseEstimator(PoseEstimator):
    """
    ONNX Runtime inference with GPU preference when ``onnxruntime-gpu`` is installed.

    Install GPU build::

        pip install onnxruntime-gpu

    Full pose decode depends on the exported head layout; until wired, :meth:`predict` may
    return an empty pose while still executing the network (useful for latency benchmarks).
    """

    __slots__ = ("_cuda_device_id", "_model_path", "_session", "_use_fp16")

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        use_fp16: bool = False,
        cuda_device_id: int = 0,
        prefer_cuda: bool = True,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        self._model_path = Path(model_path).expanduser().resolve() if model_path else None
        self._use_fp16 = bool(use_fp16)
        self._cuda_device_id = int(cuda_device_id)
        self._session = None

        if self._model_path is None or not self._model_path.is_file():
            return

        try:
            import onnxruntime as ort

            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            avail = set(ort.get_available_providers())
            providers: list[str | tuple[str, dict[str, Any]]] = []
            if prefer_cuda and "CUDAExecutionProvider" in avail:
                providers.append(("CUDAExecutionProvider", {"device_id": self._cuda_device_id}))
            elif prefer_cuda:
                logger.warning(
                    "CUDAExecutionProvider not available (install onnxruntime-gpu for RTX-class GPUs); "
                    "using CPU. pip install onnxruntime-gpu"
                )
            providers.append("CPUExecutionProvider")

            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=sess_opts,
                providers=providers,
            )
            logger.info(
                "OnnxPoseEstimator loaded %s active_providers=%s",
                self._model_path,
                self._session.get_providers(),
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("OnnxPoseEstimator could not load ONNX (%s); empty poses.", exc)

    def feed_dict_from_frame(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """Resize/normalize frame to model input (NCHW); dtype float32 or float16."""
        if self._session is None:
            return {}
        inp = self._session.get_inputs()[0]
        name = inp.name
        shape = inp.shape

        def _dim(i: int, default: int) -> int:
            if i >= len(shape):
                return default
            v = shape[i]
            if isinstance(v, int) and v > 0:
                return int(v)
            return default

        h_in = _dim(2, 256)
        w_in = _dim(3, 192)
        resized = cv2.resize(frame_bgr, (w_in, h_in))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]
        if self._use_fp16:
            x = x.astype(np.float16)
        return {name: x}

    def predict(self, frame: np.ndarray) -> KeypointsResult:
        if self._session is None:
            logger.debug("OnnxPoseEstimator: no session; empty pose.")
            return PoseResult(frame_id="", keypoints=[])

        feeds = self.feed_dict_from_frame(frame)
        if not feeds:
            return PoseResult(frame_id="", keypoints=[])

        try:
            self._session.run(None, feeds)
        except Exception as exc:
            logger.debug("OnnxPoseEstimator forward failed: %s", exc)
            return PoseResult(frame_id="", keypoints=[])

        return PoseResult(frame_id="", keypoints=[])


__all__ = ["OnnxPoseEstimator"]
