"""Optional Ollama narrative for :func:`~fightsafe_ai.pipeline.mvp_report.write_mvp_report` (offline-safe)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def try_optional_ollama_narrative(
    n_events: int,
    llm_config: Path | None,
    *,
    model: str | None = None,
    temperature: float | None = None,
    force_enabled: bool = False,
) -> str:
    """If Ollama is reachable, one short optional paragraph; else empty (offline-safe)."""
    if n_events == 0:
        return ""
    try:
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen

        from fightsafe_ai.llm.ollama_client import load_ollama_config
    except ImportError:
        return ""

    try:
        cfg = load_ollama_config(llm_config)
    except (OSError, ValueError, Exception):
        return ""
    if not bool(getattr(cfg, "enabled", False)) and not force_enabled:
        return ""

    base = getattr(cfg, "base_url", "http://localhost:11434").rstrip("/")
    m = model if model and str(model).strip() else getattr(cfg, "model", "llama3.1")
    model = str(m).strip()
    temp = (
        float(temperature) if temperature is not None else float(getattr(cfg, "temperature", 0.2))
    )
    if not base.startswith("http://") and not base.startswith("https://"):
        return ""
    try:
        req = Request(
            f"{base}/api/tags",
            method="GET",
        )
        with urlopen(req, timeout=1.5) as resp:
            if resp.status != 200:
                return ""
    except (HTTPError, URLError, OSError, TimeoutError):
        return ""
    # Reachable: tiny generate (keeps MVP self-contained; user may use full explanations in explain-events)
    prompt = (
        f"Summarize in 2-3 short English sentences for a combat sports official: this clip had "
        f"{n_events} automated HIGH/CRITICAL risk event(s) from a heuristic, decision-support system "
        "(not a medical diagnosis; not autonomous officiating)."
    )
    try:
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temp},
            }
        ).encode("utf-8")
        r2 = Request(
            f"{base}/api/generate",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(r2, timeout=30.0) as resp2:
            if resp2.status != 200:
                return ""
            data: dict[str, Any] = json.loads(resp2.read().decode("utf-8", errors="replace"))
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError, KeyError, TypeError):
        return ""
    text = (data or {}).get("response") or ""
    return str(text).strip() if text else ""


__all__ = ["try_optional_ollama_narrative"]
