"""
Unit tests: event fusion and duration (synthetic tabular data, no video file).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fightsafe_ai.risk.events import (
    RiskEventExtractionConfig,
    frame_risk_to_events,
)


DT = 0.05  # 20 fps equivalent, deterministic


def _synthetic_frame_risk(
    n: int,
    *,
    event_ranges: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """One row per frame, monotone timestamps; elevated risk in inclusive index ranges (iloc)."""
    t = np.arange(n, dtype=float) * DT
    levels = np.array(["LOW"] * n, dtype=object)
    scores = np.full(n, 0.05, dtype=float)
    for a, b in event_ranges or []:
        b1 = min(b + 1, n)
        for i in range(a, b1):
            levels[i] = "HIGH" if (i - a) % 2 == 0 else "CRITICAL"
            scores[i] = 0.4 + 0.05 * (i % 3)
    return pd.DataFrame(
        {
            "frame_id": [str(i) for i in range(n)],
            "timestamp": t,
            "risk_score": scores,
            "risk_level": levels,
        }
    )


def test_event_merging_fuses_adjacent_runs() -> None:
    """With large enough merge_gap_frames, two HIGH/CRITICAL runs with few LOWs in between merge."""
    n = 7
    t = np.arange(n, dtype=float) * DT
    levels = np.array(
        ["HIGH", "HIGH", "LOW", "HIGH", "HIGH", "LOW", "LOW"],
        dtype=object,
    )
    scores = np.array([0.5, 0.5, 0.05, 0.6, 0.6, 0.05, 0.05], dtype=float)
    df = pd.DataFrame(
        {
            "frame_id": [str(i) for i in range(n)],
            "timestamp": t,
            "risk_score": scores,
            "risk_level": levels,
        }
    )
    merged = frame_risk_to_events(
        df,
        RiskEventExtractionConfig(merge_gap_frames=2, min_duration_seconds=0.0),
    )
    # Gap between end of first block (index 1) and start of second (index 3): 1 separator row
    assert len(merged) == 1
    assert str(merged.iloc[0]["start_frame"]) == "0"
    assert str(merged.iloc[0]["end_frame"]) == "4"


def test_event_merging_respects_too_large_gap() -> None:
    """Same pattern; merge_gap_frames=1 does not merge the two runs."""
    n = 5
    t = np.arange(n, dtype=float) * DT
    levels = np.array(
        ["HIGH", "HIGH", "LOW", "HIGH", "HIGH"],
        dtype=object,
    )
    scores = np.array([0.5, 0.5, 0.0, 0.6, 0.6], dtype=float)
    df = pd.DataFrame(
        {
            "frame_id": [str(i) for i in range(n)],
            "timestamp": t,
            "risk_score": scores,
            "risk_level": levels,
        }
    )
    not_merged = frame_risk_to_events(
        df,
        RiskEventExtractionConfig(merge_gap_frames=1, min_duration_seconds=0.0),
    )
    assert len(not_merged) == 2


def test_event_duration_matches_timestamp_span() -> None:
    """duration_seconds equals end_time - start_time (synthetic table)."""
    df = _synthetic_frame_risk(6, event_ranges=[(0, 5)])
    out = frame_risk_to_events(
        df, RiskEventExtractionConfig(merge_gap_frames=0, min_duration_seconds=0.0)
    )
    assert len(out) == 1
    row = out.iloc[0]
    expected = 5.0 * DT
    assert float(row["duration_seconds"]) == pytest.approx(expected, rel=1e-6)
    assert float(row["end_time"]) - float(row["start_time"]) == pytest.approx(expected)


def test_min_duration_filter_drops_short_event() -> None:
    """Events with duration strictly less than min_duration_seconds are dropped."""
    df = _synthetic_frame_risk(2, event_ranges=[(0, 1)])
    out = frame_risk_to_events(
        df,
        RiskEventExtractionConfig(
            merge_gap_frames=0,
            min_duration_seconds=float(DT) * 1.0 + 1e-3,
        ),
    )
    assert len(out) == 0
