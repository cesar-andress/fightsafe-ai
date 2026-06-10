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

Domain-specific exceptions for FightSafe AI.
"""


class FightSafeError(Exception):
    """Base error for predictable failures (I/O, configuration, validation)."""


class ConfigurationError(FightSafeError):
    """Raised when YAML/env configuration is missing or invalid."""


class VideoIOError(FightSafeError):
    """Raised when OpenCV cannot open read/write resources."""


class VideoDownloadError(FightSafeError):
    """Raised when ``yt-dlp`` fails or the output file cannot be resolved."""


class VideoCutError(FightSafeError):
    """Raised when FFmpeg clip extraction fails or validation errors occur."""


class PoseEstimationError(FightSafeError):
    """Raised when pose estimation fails irrecoverably."""


class LLMError(FightSafeError):
    """Raised when a local/remote LLM call fails (e.g. Ollama unreachable or bad response)."""
