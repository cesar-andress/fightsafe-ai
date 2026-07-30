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

Abstract interface for text-generation backends (local Ollama, future providers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """
    Pluggable text generator for **decision-support** copy only.

    Implementations must not override computer-vision or rule-based risk detection;
    they only turn structured analytics into human-readable text.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Return model text for a fully formatted prompt (system + user instructions
        are already embedded in ``prompt`` by callers such as
        :mod:`fightsafe_ai.llm.risk_explainer`).
        """
        raise NotImplementedError
