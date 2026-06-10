"""
FightSafe AI

AI-assisted safety detection for combat sports officiating.

Optional **Ollama vision-language** review of event frames (interpretability only).

This module does **not** change pose, features, or deterministic risk scores. Output is
annotation and human-in-the-loop support; it is not a medical or definitive assessment.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Final, Protocol

from fightsafe_ai.exceptions import LLMError
from fightsafe_ai.llm.ollama_client import OllamaClient, OllamaClientConfig, load_ollama_config


logger = logging.getLogger(__name__)


class OllamaChatClient(Protocol):
    """Protocol for Ollama ``/api/chat``; implemented by :class:`~fightsafe_ai.llm.ollama_client.OllamaClient`."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
    ) -> str: ...


# Keys required in the JSON object from the VLM (and present in the deterministic fallback).
_RESULT_KEYS: Final[tuple[str, ...]] = (
    "visible_fighters",
    "apparent_posture_issue",
    "possible_surrender_gesture",
    "possible_loss_of_control",
    "uncertainty_notes",
    "human_review_recommendation",
)

_SYSTEM_VLM: Final[str] = (
    "You are a vision helper for sports safety *review* software. You must: "
    "(1) never give medical or clinical diagnosis; (2) never overrule or contradict "
    "deterministic risk numbers from the system—they are authorities; your role is only "
    "to describe what may be visible in the provided frames. "
    "(3) output **only** a single JSON object with the exact keys the user will specify, "
    "no markdown, no prose before or after."
)


def _fallback_outcome(reason: str) -> dict[str, Any]:
    return {
        "visible_fighters": None,
        "apparent_posture_issue": "not_assessed",
        "possible_surrender_gesture": "not_assessed",
        "possible_loss_of_control": "not_assessed",
        "uncertainty_notes": reason,
        "human_review_recommendation": (
            "Rely on deterministic risk scoring and the referee workflow; "
            "VLM text is not available (see uncertainty_notes)."
        ),
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("```"):
        t2 = re.sub(r"^```[a-zA-Z]*\s*\n", "", t)
        t2 = t2.rsplit("```", 1)[0].strip()
        t = t2
    try:
        o = json.loads(t)
        if isinstance(o, dict):
            return o
    except json.JSONDecodeError:
        pass
    i = t.find("{")
    j = t.rfind("}")
    if 0 <= i < j:
        try:
            o2 = json.loads(t[i : j + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(o2, dict):
            return o2
    return None


def _normalize_vlm_result(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in _RESULT_KEYS:
        v = raw.get(k)
        out[k] = v
    return out


def _encode_image_b64(p: Path) -> str:
    b = p.read_bytes()
    return base64.b64encode(b).decode("ascii")


def _build_user_prompt(event_data: dict[str, Any], n_frames: int) -> str:
    ed = json.dumps(event_data, ensure_ascii=False, indent=2)[:8000]
    return (
        f"Context (structured event metadata, may be partial):\n{ed}\n\n"
        f"You are given {n_frames} image(s) (frames) from a combat-sports or similar clip. "
        "Analyse *only* what is plausibly visible. Respond with a single JSON object and "
        "exactly these keys (use JSON strings, numbers, booleans, or null as appropriate—no "
        "nested objects unless the value is a list of short strings for visible_fighters if clear):\n"
        f"- {_RESULT_KEYS[0]}: estimate of how many people are plausibly visible, or a short list "
        f" of roles if inferable, else null.\n"
        f"- {_RESULT_KEYS[1]}: short phrase if the posture/stance is unclear, unusual, or cannot "
        f" be judged, else a neutral string.\n"
        f"- {_RESULT_KEYS[2]}: not_assessed|possible|unclear|unlikely as appropriate; never state "
        f"  certainty of intent.\n"
        f"- {_RESULT_KEYS[3]}: not_assessed|possible|unclear|unlikely; never a medical label.\n"
        f"- {_RESULT_KEYS[4]}: one short paragraph for limits of view, lighting, and ambiguity.\n"
        f"- {_RESULT_KEYS[5]}: one sentence: whether a human re-watch of the clip is suggested "
        f" (does not override system risk numbers).\n"
    )


def review_event_frames(
    image_paths: list[Path],
    event_data: dict[str, Any],
    model: str = "qwen2.5vl:7b",
    *,
    ollama_config: OllamaClientConfig | None = None,
    ollama_client: OllamaChatClient | None = None,
) -> dict[str, Any]:
    """
    Run an optional Ollama **vision** model on a small set of paths (frames or thumbnails).

    Returns a flat dict with keys:
    ``visible_fighters``, ``apparent_posture_issue``,
    ``possible_surrender_gesture``, ``possible_loss_of_control``,
    ``uncertainty_notes``, ``human_review_recommendation``.

    The VLM must not provide medical diagnosis or overrule stored risk; if Ollama is
    disabled, vision review is off, or the HTTP call fails, returns a **deterministic**
    fallback (same keys).
    """
    cfg = ollama_config if ollama_config is not None else load_ollama_config()
    if not cfg.enabled or not cfg.enable_vlm_review:
        return _fallback_outcome(
            "VLM review is disabled in configs/llm.yaml (enable: ollama.enabled, enable_vlm_review)."
        )

    max_f = int(cfg.max_frames_per_event) if int(cfg.max_frames_per_event) > 0 else 0
    if max_f == 0:
        return _fallback_outcome("max_frames_per_event is 0; no VLM call attempted.")

    paths: list[Path] = [Path(p) for p in image_paths if Path(p).is_file()]
    paths = paths[:max_f]
    if not paths:
        return _fallback_outcome("No existing image files were provided for this event.")

    vlm_model = (model or cfg.vision_model or "qwen2.5vl:7b").strip() or "qwen2.5vl:7b"
    try:
        images = [_encode_image_b64(p) for p in paths]
    except OSError as e:
        logger.warning("vision_reviewer: failed to read an image: %s", e)
        return _fallback_outcome(f"Failed to read image bytes: {e!s}")

    client = ollama_client if ollama_client is not None else OllamaClient(cfg)
    user = {
        "role": "user",
        "content": _build_user_prompt(event_data, len(paths)),
        "images": images,
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_VLM},
        user,
    ]
    try:
        raw_text = client.chat(messages, model=vlm_model)
    except (LLMError, OSError, RuntimeError) as e:
        logger.warning("vision_reviewer: Ollama vision call failed: %s", e)
        return _fallback_outcome(f"Ollama VLM call failed: {e!s}")

    parsed = _extract_json_object(raw_text)
    if not parsed:
        return _fallback_outcome(
            "VLM did not return parseable JSON; use deterministic results only."
        )
    return _normalize_vlm_result(parsed)


__all__ = ["review_event_frames"]
