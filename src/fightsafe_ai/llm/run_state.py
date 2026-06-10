"""
Per-pipeline-run state for optional Ollama explanations (avoid repeated failures).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)

_LLM_SHUTDOWN_WARNING = (
    "Ollama model load/resource error: disabling LLM for the remainder of this run; "
    "using deterministic template explanations for all remaining events. "
    "Deterministic risk scores and events.json are unchanged."
)


@dataclass
class LLMExplanationRunState:
    """
    Tracks whether the optional LLM path should be skipped after a fatal load/resource error.

    Serialized to ``llm_explanation_state.json`` at the run root for ``summary.json`` / QA.
    """

    llm_requested: bool = False
    llm_disabled_for_run: bool = False
    llm_fallback_used: bool = False
    llm_error: str | None = None
    _shutdown_logged: bool = field(default=False, repr=False)

    def skip_llm_calls(self) -> bool:
        return bool(self.llm_disabled_for_run)

    def record_resource_or_load_failure(self) -> None:
        """First resource/load failure: disable LLM for this run and emit one warning."""
        if self.llm_disabled_for_run:
            return
        self.llm_disabled_for_run = True
        self.llm_fallback_used = True
        self.llm_error = "model failed to load"
        if not self._shutdown_logged:
            logger.warning(_LLM_SHUTDOWN_WARNING)
            self._shutdown_logged = True

    def record_other_llm_failure(self) -> None:
        """Non-fatal path: template used for this event only; LLM may work on the next."""
        self.llm_fallback_used = True

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "llm_requested": bool(self.llm_requested),
            "llm_available": not bool(self.llm_disabled_for_run),
            "llm_fallback": bool(self.llm_fallback_used),
            "llm_error": self.llm_error,
        }


__all__ = ["LLMExplanationRunState"]
