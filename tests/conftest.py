"""Pytest: put ``src`` and the repository root on ``sys.path`` for reliable imports."""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
# Allow ``from tests.fixtures`` / ``from tests.support`` from test modules.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
