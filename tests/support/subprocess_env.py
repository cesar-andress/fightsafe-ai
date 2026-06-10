"""Environment helpers for subprocess CLI tests without editable installation."""

from __future__ import annotations

import os
from pathlib import Path


# fightsafe-ai repository root (parent of ``tests/`` and ``src/``).
REPO_ROOT = Path(__file__).resolve().parents[2]


def env_with_src_pythonpath() -> dict[str, str]:
    """
    Return ``os.environ`` with ``PYTHONPATH`` pointing at ``<repo>/src``.

    Use for ``subprocess.run(..., "-m", "fightsafe_ai....")`` so the package resolves
    without ``pip install -e``.
    """
    return {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
