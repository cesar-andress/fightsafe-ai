"""CUDA inference helpers (optional torch)."""

from __future__ import annotations

import pytest

from fightsafe_ai.pose.backends.device_runtime import configure_cuda_inference, resolve_torch_device


pytestmark = pytest.mark.unit


def test_resolve_explicit_cuda_alias() -> None:
    assert resolve_torch_device("cuda") == "cuda:0"
    assert resolve_torch_device("gpu") == "cuda:0"


def test_configure_cuda_inference_does_not_raise() -> None:
    """Idempotent; no-ops when torch or CUDA is absent."""
    configure_cuda_inference()
