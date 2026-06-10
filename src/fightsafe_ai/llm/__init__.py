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

Optional **local** LLM helpers (Ollama) to narrate **structured** risk output.

**Does not** perform pose estimation, feature computation, or risk detection. Use for
human-in-the-loop review only.
"""

from __future__ import annotations

from fightsafe_ai.llm.base import BaseLLMClient
from fightsafe_ai.llm.config import (
    ExplanationsConfig,
    LLMFileConfig,
    event_level_reaches_threshold,
    load_llm_file_config,
)
from fightsafe_ai.llm.explainer import (
    explain_event,
    explain_multi_signal,
    multi_signal_context_to_dict,
    referee_alert_to_dict,
    risk_decision_to_dict,
)
from fightsafe_ai.llm.ollama_client import (
    OllamaClient,
    OllamaClientConfig,
    load_ollama_client_from_yaml,
    load_ollama_config,
)
from fightsafe_ai.llm.prompts import (
    build_clip_summary_prompt,
    build_multi_signal_explanation_prompt,
    build_risk_explanation_prompt,
    explain_risk_event_prompt,
    suggest_annotation_prompt,
    summarize_clip_prompt,
)
from fightsafe_ai.llm.report_enricher import (
    enrich_clip_narrative,
    rule_based_clip_narrative,
)
from fightsafe_ai.llm.report_generator import generate_clip_report
from fightsafe_ai.llm.risk_explainer import (
    explain_risk_event,
    fallback_multi_signal_explanation,
    fallback_risk_explanation,
)
from fightsafe_ai.llm.vision_reviewer import review_event_frames


__all__ = [
    "BaseLLMClient",
    "ExplanationsConfig",
    "LLMFileConfig",
    "OllamaClient",
    "OllamaClientConfig",
    "build_clip_summary_prompt",
    "build_multi_signal_explanation_prompt",
    "build_risk_explanation_prompt",
    "enrich_clip_narrative",
    "event_level_reaches_threshold",
    "explain_event",
    "explain_multi_signal",
    "explain_risk_event",
    "explain_risk_event_prompt",
    "fallback_multi_signal_explanation",
    "fallback_risk_explanation",
    "generate_clip_report",
    "load_llm_file_config",
    "load_ollama_client_from_yaml",
    "load_ollama_config",
    "multi_signal_context_to_dict",
    "referee_alert_to_dict",
    "review_event_frames",
    "risk_decision_to_dict",
    "rule_based_clip_narrative",
    "suggest_annotation_prompt",
    "summarize_clip_prompt",
]
