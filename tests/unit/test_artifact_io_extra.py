"""Exercises for :mod:`fightsafe_ai.pipeline.artifact_io` branches (JSON/CSV helpers)."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from fightsafe_ai.pipeline.artifact_io import (
    risk_scores_dataframe_for_csv,
    sanitize_for_json,
)


pytestmark = pytest.mark.unit


def test_sanitize_for_json_scalars() -> None:
    assert sanitize_for_json(None) is None
    assert sanitize_for_json("a") == "a"
    assert sanitize_for_json(True) is True
    assert sanitize_for_json(b"hi") == "hi"
    assert sanitize_for_json(np.int64(3)) == 3
    assert sanitize_for_json(np.float32(0.5)) == pytest.approx(0.5)
    assert sanitize_for_json(np.bool_(True)) is True
    assert sanitize_for_json(np.array([1.0, 2.0])) == [1.0, 2.0]


def test_sanitize_for_json_containers() -> None:
    assert sanitize_for_json((1, "x", None)) == [1, "x", None]
    d = {1: "a", "b": 2.0}
    out = sanitize_for_json(d)
    assert out == {"1": "a", "b": 2.0}


def test_sanitize_for_json_reals_and_float() -> None:
    assert sanitize_for_json(float("nan")) is None
    assert sanitize_for_json(float("inf")) is None
    assert math.isclose(float(sanitize_for_json(1.5)), 1.5)
    v = 2.0
    assert math.isclose(float(sanitize_for_json(v)), 2.0)


def test_sanitize_for_json_timestamp() -> None:
    ts = pd.Timestamp("2020-01-01T12:00:00")
    s = sanitize_for_json(ts)
    assert "2020-01-01" in s
    d = datetime(2019, 3, 3, 1, 2, 3)
    assert "2019-03-03" in str(sanitize_for_json(d))


def test_sanitize_for_json_fallback_str() -> None:
    class _O:
        def __str__(self) -> str:
            return "objx"

    assert sanitize_for_json(_O()) == "objx"
    assert "1" in sanitize_for_json({1, 2, 3})


def test_risk_scores_dataframe_for_csv_serialization() -> None:
    df = pd.DataFrame(
        {
            "risk_score": [0.1, 0.2],
            "triggered_rules": [
                ["a", "b"],
                None,
            ],
        }
    )
    out = risk_scores_dataframe_for_csv(df)
    assert "a" in str(out["triggered_rules"].iloc[0])
    a = out["triggered_rules"].iloc[1]
    assert a is None or a == ""


def test_risk_scores_dataframe_for_csv_empty_rule_cell() -> None:
    df = pd.DataFrame(
        {
            "risk_score": [0.1],
            "triggered_rules": [float("nan")],
        }
    )
    out = risk_scores_dataframe_for_csv(df)
    cell = out["triggered_rules"].iloc[0]
    assert (
        cell is None
        or str(cell) == "nan"
        or (isinstance(cell, float) and math.isnan(cell))
        or cell == ""
    )
