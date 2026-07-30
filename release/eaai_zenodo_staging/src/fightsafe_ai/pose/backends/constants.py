"""
Runtime pose backend names (single-frame :class:`~fightsafe_ai.pose.backends.pose_estimator.PoseEstimator`).

Separate from **batch** backends (:class:`~fightsafe_ai.pose.backends.base.BasePoseEstimator`)
used by ``estimate-pose`` / folder CSV export (e.g. mediapipe, mock).
"""

from __future__ import annotations


# Strings accepted by CLI ``--pose-backend`` for live / low-latency paths.
RUNTIME_BACKEND_CLI_CHOICES: tuple[str, ...] = ("torch", "onnx", "tensorrt")

_RUNTIME_ALIASES: dict[str, str] = {
    "trt": "tensorrt",
}

_VALID_RUNTIME = frozenset({"torch", "onnx", "tensorrt"})


def normalize_runtime_backend(kind: str | None) -> str:
    """
    Normalize user input (default ``torch``; ``trt`` → ``tensorrt``).
    """
    k = (kind or "torch").strip().lower()
    return _RUNTIME_ALIASES.get(k, k)


def assert_valid_runtime_backend(kind: str) -> str:
    """Return canonical name or raise ``ConfigurationError``."""
    from fightsafe_ai.exceptions import ConfigurationError

    k = normalize_runtime_backend(kind)
    if k not in _VALID_RUNTIME:
        raise ConfigurationError(
            f"Unknown runtime pose backend: {kind!r} (expected one of {sorted(_VALID_RUNTIME)})"
        )
    return k


__all__ = [
    "RUNTIME_BACKEND_CLI_CHOICES",
    "assert_valid_runtime_backend",
    "normalize_runtime_backend",
]
