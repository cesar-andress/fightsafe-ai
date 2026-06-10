"""
NVIDIA GPU metrics via optional NVML (pynvml / nvidia-ml-py).

The dashboard does **not** require this module: if ``pynvml`` is missing or NVML
fails, callers get a structured "unavailable" payload without raising.
"""

from __future__ import annotations

import logging
import threading
from typing import Any


logger = logging.getLogger(__name__)

_lock = threading.Lock()
_nvml_initialized = False
_pynvml: Any = None
_pynvml_import_error: str | None = None


def _try_import_pynvml() -> bool:
    global _pynvml, _pynvml_import_error
    if _pynvml is not None or _pynvml_import_error is not None:
        return _pynvml is not None
    try:
        import pynvml  # type: ignore[import-untyped]

        _pynvml = pynvml
        return True
    except ImportError as exc:
        _pynvml_import_error = str(exc)
        return False


def _decode_name(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip() or "GPU"
    return str(raw).strip() or "GPU"


def _mib_from_bytes(n: int | float) -> float:
    return float(n) / (1024.0 * 1024.0)


def _ensure_nvml() -> tuple[bool, str | None]:
    """
    Initialize NVML once. Returns (ok, error_message).
    """

    global _nvml_initialized
    if not _try_import_pynvml():
        return False, "pynvml not installed (optional: pip install nvidia-ml-py)"
    assert _pynvml is not None
    with _lock:
        if _nvml_initialized:
            return True, None
        try:
            _pynvml.nvmlInit()
            _nvml_initialized = True
            return True, None
        except Exception as exc:
            msg = str(exc) or type(exc).__name__
            logger.debug("NVML init failed: %s", msg)
            return False, msg


def shutdown_gpu_monitor() -> None:
    """Release NVML. Safe to call multiple times or if NVML was never initialized."""

    global _nvml_initialized
    if not _try_import_pynvml() or _pynvml is None:
        return
    with _lock:
        if not _nvml_initialized:
            return
        try:
            _pynvml.nvmlShutdown()
        except Exception:
            logger.debug("nvmlShutdown failed", exc_info=True)
        finally:
            _nvml_initialized = False


def _unavailable_payload(*, reason: str, detail: str | None = None) -> dict[str, Any]:
    return {
        "nvidia_nvml_available": False,
        "status": reason,
        "message": detail,
        "gpu_name": None,
        "gpu_utilization_percent": None,
        "memory_used_mib": None,
        "memory_total_mib": None,
        "memory_percent": None,
        "temperature_c": None,
        "power_draw_watts": None,
        "power_available": False,
    }


def get_nvidia_gpu_metrics(device_index: int = 0) -> dict[str, Any]:
    """
    Sample the primary NVIDIA GPU via NVML.

    Never raises. Returns a dict suitable for JSON, including nulls when data is missing.
    """

    ok, err = _ensure_nvml()
    if not ok:
        return _unavailable_payload(reason="nvml_unavailable", detail=err)

    assert _pynvml is not None
    try:
        n = int(_pynvml.nvmlDeviceGetCount())
        if n <= 0:
            return _unavailable_payload(reason="no_gpu", detail="No NVIDIA GPU reported by NVML")
        idx = max(0, min(device_index, n - 1))
        handle = _pynvml.nvmlDeviceGetHandleByIndex(idx)

        name = _decode_name(_pynvml.nvmlDeviceGetName(handle))

        util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = float(util.gpu)

        mem_info = _pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_b = int(mem_info.used)
        total_b = int(mem_info.total)
        used_mib = _mib_from_bytes(used_b)
        total_mib = max(_mib_from_bytes(total_b), 1e-6)
        mem_pct = 100.0 * float(used_b) / float(total_b) if total_b > 0 else None

        temp_c: float | None = None
        try:
            temp_c = float(_pynvml.nvmlDeviceGetTemperature(handle, _pynvml.NVML_TEMPERATURE_GPU))
        except Exception as exc:
            logger.debug("NVML temperature query unsupported or failed: %s", exc)

        power_w: float | None = None
        power_ok = False
        try:
            mw = _pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = float(mw) / 1000.0
            power_ok = True
        except Exception as exc:
            logger.debug("NVML power draw query unsupported or failed: %s", exc)

        return {
            "nvidia_nvml_available": True,
            "status": "ok",
            "message": None,
            "gpu_name": name,
            "gpu_utilization_percent": gpu_util,
            "memory_used_mib": round(used_mib, 2),
            "memory_total_mib": round(total_mib, 2),
            "memory_percent": round(mem_pct, 2) if mem_pct is not None else None,
            "temperature_c": round(temp_c, 1) if temp_c is not None else None,
            "power_draw_watts": round(power_w, 2) if power_w is not None else None,
            "power_available": power_ok,
        }
    except Exception as exc:
        msg = str(exc) or type(exc).__name__
        logger.debug("NVML sample failed: %s", msg)
        return _unavailable_payload(reason="nvml_error", detail=msg)


__all__ = ["get_nvidia_gpu_metrics", "shutdown_gpu_monitor"]
