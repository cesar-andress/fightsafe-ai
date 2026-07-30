"""Shared compute device resolution for optional GPU-accelerated pose backends (Torch-based)."""

from __future__ import annotations

import importlib


def configure_cuda_inference(
    *,
    cudnn_benchmark: bool = True,
    allow_tf32: bool = True,
) -> None:
    """
    Tune PyTorch CUDA for throughput on NVIDIA GPUs (e.g. Ada Lovelace RTX 4090).

    Enables cuDNN autotuning, TF32 tensor cores on Ampere+, and higher matmul precision.
    Safe to call once at process startup before heavy inference. No-ops if CUDA unavailable.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, ValueError):
        pass
    if allow_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except AttributeError:
            pass


def resolve_torch_device(preference: str) -> str:
    """
    Map a user preference to a string suitable for PyTorch and Ultralytics / MMPose.

    * ``auto`` — prefer ``cuda:0`` if ``torch.cuda.is_available()``, else ``mps`` on Apple
      Silicon when available, else ``cpu``.
    * ``cpu``, ``cuda``, ``cuda:0``, ``mps`` — passed through (with ``cuda`` normalized to ``cuda:0``).

    If ``torch`` is not installed, returns ``cpu`` (callers that require Torch should
    have already failed earlier).
    """
    pref = (preference or "auto").strip().lower()
    if pref in ("cpu", "mps"):
        return pref
    if pref in ("cuda", "gpu"):
        return "cuda:0"
    if pref.startswith("cuda:"):
        return pref

    if pref != "auto":
        return "cpu"

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


__all__ = ["configure_cuda_inference", "resolve_torch_device"]
