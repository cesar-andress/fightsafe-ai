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

Natural sorting for filenames containing numeric indices (e.g. ``frame_0007.jpg``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def natural_sort_strings(values: Iterable[str]) -> list[str]:
    """Sort strings so embedded digit runs compare numerically (e.g. ``frame_2`` before ``frame_10``)."""

    def key(s: str) -> list[Any]:
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

    return sorted(values, key=key)


def natural_sort_paths(paths: Iterable[Path]) -> list[Path]:
    """Sort paths so numeric substrings compare numerically."""

    def key(p: Path) -> list[Any]:
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.stem)]

    return sorted(paths, key=key)
