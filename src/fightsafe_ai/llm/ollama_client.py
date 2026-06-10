"""
FightSafe AI

AI-assisted safety detection for combat sports officiating.

Authors:
- David Martin Moncunill (david.martinm@ucjc.edu)
- César Andrés Sánchez (cesar.andress@ucjc.edu)

Affiliation:
Camilo José Cela University (UCJC)
Madrid, Spain

This module is part of a research-oriented system for human-in-the-loop safety analysis.

HTTP client for a **local** Ollama instance (``/api/generate``).

This module does not install or start Ollama; it only speaks the documented REST API.
"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.request import Request, urlopen

from fightsafe_ai.exceptions import LLMError
from fightsafe_ai.llm.base import BaseLLMClient


logger = logging.getLogger(__name__)


def is_ollama_model_load_or_resource_error(exc: BaseException) -> bool:
    """
    Return True when failure is likely Ollama/model load, memory, or server resource limits.

    Used to disable LLM for the rest of a pipeline run instead of logging once per event.
    """
    msg = str(exc).lower()
    # Ollama often returns HTTP 500 while pulling or loading a model into VRAM/RAM.
    for code in ("http 500", "http 502", "http 503"):
        if code in msg:
            return True
    # Body snippets from Ollama / OS when overloaded
    needles = (
        "out of memory",
        "oom",
        "cuda",
        "vram",
        "unable to load",
        "failed to load",
        "model loading",
        "resource",
        "server busy",
        "context length",
        "kv cache",
    )
    if any(n in msg for n in needles):
        return True
    # LLMError from our client embeds HTTP code and optional JSON error text
    return "ollama http" in msg and any(x in msg for x in ("500", "502", "503"))


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 120.0


@dataclass(frozen=True)
class OllamaClientConfig:
    """Settings loaded from :file:`configs/llm.yaml` or constructed in code."""

    enabled: bool = False
    base_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = DEFAULT_TIMEOUT
    # Optional multimodal (Ollama vision models); does not change pose/risk scoring.
    vision_model: str = "qwen2.5vl:7b"
    enable_vlm_review: bool = False
    max_frames_per_event: int = 4


def load_ollama_config(path: Path | None = None) -> OllamaClientConfig:
    """
    Load the ``ollama:`` block from :file:`configs/llm.yaml`.

    Parameters
    ----------
    path
        If ``None``, uses ``<project root>/configs/llm.yaml`` (see
        :func:`fightsafe_ai.llm.config.load_llm_file_config`).
    """
    from fightsafe_ai.llm.config import load_llm_file_config

    return load_llm_file_config(path).ollama


def load_ollama_client_from_yaml(path: Path | None = None) -> OllamaClient:
    """Convenience: :func:`load_ollama_config` + :class:`OllamaClient` constructor."""
    return OllamaClient(load_ollama_config(path=path))


class OllamaClient(BaseLLMClient):
    """
    Local Ollama HTTP API: ``POST {base_url}/api/generate`` with ``"stream": false``.

    Respects :class:`OllamaClientConfig` (``model``, ``temperature``, ``timeout``). This client
    does not influence pose, features, or risk scores — use only for optional narrative.
    """

    def __init__(
        self,
        config: OllamaClientConfig | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> None:
        c = config or OllamaClientConfig()
        self._base = (base_url or c.base_url).rstrip("/")
        self._model = model if model is not None else c.model
        self._temperature = c.temperature if temperature is None else float(temperature)
        self._timeout = c.timeout if timeout is None else float(timeout)
        self._url = f"{self._base}/api/generate"
        self._url_chat = f"{self._base}/api/chat"

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str) -> str:
        body: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
            },
        }
        data = json.dumps(body).encode("utf-8")
        req = Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        open_kw: dict[str, Any] = {"timeout": self._timeout}
        if self._url.lower().startswith("https://"):
            open_kw["context"] = ssl.create_default_context()
        try:
            with urlopen(req, **open_kw) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")[:2000] if e.fp else ""
            logger.warning("Ollama HTTP %s: %s", e.code, err_text)
            raise LLMError(f"Ollama HTTP {e.code}: {err_text or e.reason}") from e
        except urllib.error.URLError as e:
            logger.warning("Ollama connection failed: %s", e)
            raise LLMError(f"Ollama unavailable: {e.reason}") from e
        try:
            parsed: dict[str, Any] = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as e:
            raise LLMError("Ollama returned non-JSON body") from e
        if not isinstance(parsed, dict):
            raise LLMError("Ollama response JSON must be an object")
        text = parsed.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Ollama response missing a non-empty 'response' string")
        return text.strip()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
    ) -> str:
        """
        Call ``POST /api/chat`` (multimodal-capable: user messages may include an ``"images"`` list
        of base64 strings per Ollama's API). Returns the assistant message ``content`` string.
        """
        mdl = (model or self._model).strip() or self._model
        body: dict[str, Any] = {
            "model": mdl,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._temperature,
            },
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = Request(
            self._url_chat,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        open_kw: dict[str, Any] = {"timeout": self._timeout}
        if self._url_chat.lower().startswith("https://"):
            open_kw["context"] = ssl.create_default_context()
        try:
            with urlopen(req, **open_kw) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")[:2000] if e.fp else ""
            logger.warning("Ollama HTTP %s (chat): %s", e.code, err_text)
            raise LLMError(f"Ollama chat HTTP {e.code}: {err_text or e.reason}") from e
        except urllib.error.URLError as e:
            logger.warning("Ollama chat connection failed: %s", e)
            raise LLMError(f"Ollama chat unavailable: {e.reason}") from e
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as e:
            raise LLMError("Ollama chat returned non-JSON body") from e
        if not isinstance(parsed, dict):
            raise LLMError("Ollama chat response JSON must be an object")
        message = (
            cast("dict[str, Any]", parsed.get("message", {}))
            if isinstance(parsed.get("message"), dict)
            else {}
        )
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Ollama chat response missing a non-empty message.content string")
        return text.strip()
