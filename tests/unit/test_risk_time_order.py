"""Tests for :mod:`fightsafe_ai.risk.time_order`."""

from __future__ import annotations

import pandas as pd
import pytest

from fightsafe_ai.risk.time_order import sort_frames_add_timestamp


pytestmark = pytest.mark.unit


def test_sort_frames_add_timestamp_orders_and_sets_clock() -> None:
    df = pd.DataFrame(
        {
            "frame_id": ["10", "2", "1"],
            "score": [0.1, 0.2, 0.3],
        }
    )
    out = sort_frames_add_timestamp(df, fps=10.0)
    assert list(out["frame_id"]) == ["1", "2", "10"]
    assert out["timestamp"].iloc[0] == 0.0
    assert out["timestamp"].iloc[1] == pytest.approx(0.1)
    assert out["timestamp"].iloc[2] == pytest.approx(0.2)
