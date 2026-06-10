"""Extra unit coverage for :mod:`fightsafe_ai.qa.validators` (pure checks, no full runs)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fightsafe_ai.qa.validators import (
    VALID_RISK_LEVELS,
    check_risk_score_range,
    check_run_directory_exists,
)


pytestmark = pytest.mark.unit


def test_check_run_directory_exists_passes(tmp_path: Path) -> None:
    r = check_run_directory_exists(tmp_path)
    assert r.status == "pass"


def test_check_run_directory_exists_fails_on_file(tmp_path: Path) -> None:
    f = tmp_path / "not_a_dir"
    f.write_text("x", encoding="utf-8")
    r = check_run_directory_exists(f)
    assert r.status == "fail"


def test_check_risk_score_range_flags_out_of_01() -> None:
    bad = pd.DataFrame({"risk_score": [0.0, 1.2, 0.5]})
    out = check_risk_score_range(bad)
    assert any(x.status == "fail" for x in out)


def test_valid_risk_levels_frozenset() -> None:
    assert "CRITICAL" in VALID_RISK_LEVELS
    assert "LOW" in VALID_RISK_LEVELS
