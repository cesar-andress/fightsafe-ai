"""GPU monitor never raises and returns a stable JSON-shaped dict."""

from __future__ import annotations

from fightsafe_ai.live.gpu_monitor import get_nvidia_gpu_metrics, shutdown_gpu_monitor


def test_get_nvidia_gpu_metrics_structure() -> None:
    d = get_nvidia_gpu_metrics()
    assert isinstance(d["nvidia_nvml_available"], bool)
    assert "gpu_name" in d
    assert "gpu_utilization_percent" in d
    assert "memory_used_mib" in d
    assert "memory_total_mib" in d
    assert "memory_percent" in d
    assert "temperature_c" in d
    assert "power_draw_watts" in d
    assert "status" in d
    shutdown_gpu_monitor()
