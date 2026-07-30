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

YAML file :file:`configs/llm.yaml` — Ollama transport + explanation style flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from fightsafe_ai.exceptions import ConfigurationError
from fightsafe_ai.llm.ollama_client import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    OllamaClientConfig,
)


# Ordered least → most severe (aligns with :mod:`fightsafe_ai.risk.scorer` band names).
RISK_EVENT_LEVELS: Final[tuple[str, ...]] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


@dataclass(frozen=True)
class ExplanationsConfig:
    """
    How prompts and :func:`~fightsafe_ai.llm.risk_explainer.fallback_risk_explanation`
    phrase rule names, disclaimers, and the “review this clip” nudge.
    """

    include_triggered_rules: bool = True
    include_safety_disclaimer: bool = True
    recommend_human_review_threshold: str = "HIGH"


@dataclass(frozen=True)
class LLMFileConfig:
    """Full contents of ``configs/llm.yaml``."""

    ollama: OllamaClientConfig
    explanations: ExplanationsConfig


def _default_llm_yaml_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "llm.yaml"


def _as_bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("true", "yes", "1", "on"):
        return True
    if s in ("false", "no", "0", "off"):
        return False
    return default


def _parse_explanations(x: Any) -> ExplanationsConfig:
    if not x or not isinstance(x, dict):
        return ExplanationsConfig()
    thr = (
        str(
            x.get(
                "recommend_human_review_threshold",
                ExplanationsConfig().recommend_human_review_threshold,
            )
        ).strip()
        or "HIGH"
    )
    return ExplanationsConfig(
        include_triggered_rules=_as_bool(x.get("include_triggered_rules", True), True),
        include_safety_disclaimer=_as_bool(x.get("include_safety_disclaimer", True), True),
        recommend_human_review_threshold=thr,
    )


def _as_int(v: Any, default: int) -> int:
    if v is None:
        return default
    if isinstance(v, int):
        return max(0, v)
    if isinstance(v, float):
        return max(0, int(v))
    try:
        n = int(str(v).strip())
    except (TypeError, ValueError):
        return default
    return max(0, n)


def _parse_ollama(o: Any) -> OllamaClientConfig:
    if not o or not isinstance(o, dict):
        return OllamaClientConfig()
    def_vm = o.get("vision_model", OllamaClientConfig().vision_model)
    mfe = o.get("max_frames_per_event", OllamaClientConfig().max_frames_per_event)
    return OllamaClientConfig(
        enabled=_as_bool(o.get("enabled", False), False),
        base_url=str(o.get("base_url", DEFAULT_OLLAMA_URL)).rstrip("/"),
        model=str(o.get("model", DEFAULT_MODEL)),
        temperature=float(o.get("temperature", DEFAULT_TEMPERATURE)),
        timeout=float(o.get("timeout_seconds", o.get("timeout", DEFAULT_TIMEOUT))),
        vision_model=str(def_vm) if str(def_vm).strip() else OllamaClientConfig().vision_model,
        enable_vlm_review=_as_bool(o.get("enable_vlm_review", False), False),
        max_frames_per_event=_as_int(mfe, OllamaClientConfig().max_frames_per_event),
    )


def event_level_reaches_threshold(event_level: str | None, threshold: str) -> bool:
    """
    Return whether ``event_level`` is at least as severe as ``threshold`` on the
    ``LOW < MEDIUM < HIGH < CRITICAL`` scale.

    ``threshold`` of ``"NONE"`` (or empty) => never nudge. Unknown levels are treated
    as ``"LOW"`` for comparison.
    """
    t = (threshold or "NONE").strip().upper()
    if t in ("NONE", "OFF", ""):
        return False
    if t not in RISK_EVENT_LEVELS:
        t = "HIGH"
    el = (event_level or "LOW").strip().upper()
    if el not in RISK_EVENT_LEVELS:
        el = "LOW"
    return RISK_EVENT_LEVELS.index(el) >= RISK_EVENT_LEVELS.index(t)


def load_llm_file_config(path: Path | None = None) -> LLMFileConfig:
    """
    Read ``ollama:`` and ``explanations:`` blocks from the YAML file.

    Parameters
    ----------
    path
        Defaults to **<project root> / configs / llm.yaml** next to the repo.
    """
    p = path or _default_llm_yaml_path()
    if not p.is_file():
        raise ConfigurationError(f"LLM config not found: {p}")
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Root of LLM config must be a mapping: {p}")
    o = data.get("ollama", {})
    if not isinstance(o, dict):
        o = {}
    ex = data.get("explanations", {})
    if not isinstance(ex, dict):
        ex = {}
    return LLMFileConfig(ollama=_parse_ollama(o), explanations=_parse_explanations(ex))


__all__ = [
    "RISK_EVENT_LEVELS",
    "ExplanationsConfig",
    "LLMFileConfig",
    "_default_llm_yaml_path",
    "event_level_reaches_threshold",
    "load_llm_file_config",
]
