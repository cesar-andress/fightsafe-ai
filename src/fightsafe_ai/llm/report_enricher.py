"""
Enrich run / clip **reports** with optional LLM text — never with pose, features, or risk scores.

If Ollama is disabled or unreachable, :func:`rule_based_clip_narrative` provides deterministic copy.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fightsafe_ai.exceptions import ConfigurationError, LLMError
from fightsafe_ai.llm.base import BaseLLMClient
from fightsafe_ai.llm.prompts import build_clip_summary_prompt


logger = logging.getLogger(__name__)


def rule_based_clip_narrative(run_data: dict[str, Any]) -> str:
    """
    Non-LLM summary bullets from structured run or clip metadata (for reports when Ollama is off).
    """
    parts: list[str] = [
        "Summary from **algorithmic** risk cues (heuristic, decision-support only).",
        "This is not a medical diagnosis.",
    ]
    if run_data.get("clip_id") is not None:
        parts.append(f"- **Clip / run id:** {run_data['clip_id']!s}")
    if run_data.get("max_risk_score") is not None:
        parts.append(f"- **Max risk score (0–1):** {run_data['max_risk_score']!s}")
    evc = run_data.get("event_count")
    if evc is None and isinstance(run_data.get("detected_events"), list):
        evc = len(run_data["detected_events"])
    if evc is not None:
        parts.append(f"- **Event segments (elevated):** {evc!s}")
    raw = json.dumps(
        {
            k: v
            for k, v in run_data.items()
            if k not in ("explanations", "ollama_explanations", "ai_explanations")
        },
        default=str,
        indent=2,
    )
    parts.append("")
    parts.append("Structured context (for reviewers):")
    parts.append(f"```json\n{raw}\n```")
    return "\n".join(parts)


def enrich_clip_narrative(
    run_data: dict[str, Any],
    client: BaseLLMClient | None = None,
) -> str:
    """
    Return narrative text for the "AI explanation" section of a clip / run report.

    If ``client`` is ``None`` or Ollama is disabled in :file:`configs/llm.yaml`, returns
    :func:`rule_based_clip_narrative`. On LLM failure, falls back the same way.
    """
    from fightsafe_ai.llm.config import load_llm_file_config

    if client is None:
        return rule_based_clip_narrative(run_data)

    try:
        if not load_llm_file_config().ollama.enabled:
            return rule_based_clip_narrative(run_data)
    except (ConfigurationError, OSError, ValueError) as e:
        logger.debug("LLM off or config missing; rule-based report text: %s", e)
        return rule_based_clip_narrative(run_data)

    prompt = build_clip_summary_prompt(run_data)
    try:
        return client.generate(prompt).strip()
    except LLMError as e:
        logger.warning("enrich_clip_narrative: LLM failed, using rule-based text: %s", e)
        return rule_based_clip_narrative(run_data)
    except OSError as e:  # pragma: no cover
        logger.warning("enrich_clip_narrative: I/O error, using rule-based text: %s", e)
        return rule_based_clip_narrative(run_data)


__all__ = ["enrich_clip_narrative", "rule_based_clip_narrative"]
