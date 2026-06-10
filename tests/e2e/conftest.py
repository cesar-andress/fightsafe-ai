"""Default marker for tests under ``tests/e2e/`` (out of default CI collection)."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.e2e
