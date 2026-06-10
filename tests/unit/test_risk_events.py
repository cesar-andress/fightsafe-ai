"""
Tests for frame risk → event aggregation (:mod:`fightsafe_ai.risk.events`).

Loaded via :func:`tests.support.isolated.load_risk_events` (no full package import).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.support.isolated import load_risk_events


_ev = load_risk_events()
frame_risk_to_events = _ev.frame_risk_to_events
frame_risk_to_events_list = _ev.frame_risk_to_events_list
RiskEventExtractionConfig = _ev.RiskEventExtractionConfig
COL = _ev  # column constants on module

DT = 0.1  # 100 ms per frame in synthetic data


def _frame(
    n: int,
    *,
    high_ranges: list[tuple[int, int | None]] | None = None,
) -> pd.DataFrame:
    """n rows, timestamps 0, dt, 2*dt, ...; set risk_level HIGH on inclusive iloc ranges."""
    t = np.arange(n, dtype=float) * DT
    levels = np.array(["LOW"] * n, dtype=object)
    scores = np.zeros(n, dtype=float)
    for lo, hi in high_ranges or []:
        h = n if hi is None else min(hi + 1, n)
        for i in range(lo, h):
            levels[i] = "CRITICAL" if (i - lo) % 2 == 0 else "HIGH"  # mix; event_level → CRITICAL
            scores[i] = 0.4 + 0.1 * (i % 3)
    for i in range(n):
        if str(levels[i]) == "LOW":
            scores[i] = 0.05
    return pd.DataFrame(
        {
            "frame_id": [str(i) for i in range(n)],
            "timestamp": t,
            "risk_score": scores,
            "risk_level": levels,
        }
    )


def test_single_run_one_event() -> None:
    df = _frame(5, high_ranges=[(0, 4)])
    out = frame_risk_to_events(
        df, RiskEventExtractionConfig(merge_gap_frames=0, min_duration_seconds=0.0)
    )
    assert len(out) == 1
    assert str(out.iloc[0]["start_frame"]) == "0"
    assert out.iloc[0]["event_level"] == "CRITICAL"
    assert out.iloc[0]["duration_seconds"] == pytest.approx(4 * DT)
    assert out.iloc[0]["max_risk_score"] >= 0.4


def test_merge_two_runs_with_small_gap() -> None:
    # HIGH at 0-1, one LOW at 2, HIGH at 3-4  →  gap of 0 "middle" between runs?
    # rows: 0 H, 1 H, 2 L, 3 H, 4 H  → first run (0,1) second (3,4) gap = 3-1-1 = 1
    n = 5
    t = np.arange(n, dtype=float) * DT
    levels = np.array(["HIGH", "HIGH", "LOW", "HIGH", "HIGH"], dtype=object)
    scores = np.array([0.5, 0.5, 0.0, 0.6, 0.6])
    df = pd.DataFrame(
        {
            "frame_id": [str(i) for i in range(n)],
            "timestamp": t,
            "risk_score": scores,
            "risk_level": levels,
        }
    )
    merged = frame_risk_to_events(
        df, RiskEventExtractionConfig(merge_gap_frames=2, min_duration_seconds=0.0)
    )
    assert len(merged) == 1
    assert merged.iloc[0]["start_time"] == pytest.approx(0.0)
    assert merged.iloc[0]["end_time"] == pytest.approx(4 * DT)
    assert float(merged.iloc[0]["duration_seconds"]) == pytest.approx(4 * DT, rel=1e-6)

    not_merged = frame_risk_to_events(
        df, RiskEventExtractionConfig(merge_gap_frames=1, min_duration_seconds=0.0)
    )
    # gap=1, merge only if gap < 1 → gap 0 allowed? gap is 1: 1 < 1 is false, no merge
    assert len(not_merged) == 2


def test_no_merge_when_gap_too_large() -> None:
    n = 8
    t = np.arange(n, dtype=float) * DT
    levels = np.array(
        ["HIGH", "HIGH", "LOW", "LOW", "LOW", "HIGH", "HIGH", "LOW"],
        dtype=object,
    )
    scores = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0])
    df = pd.DataFrame(
        {
            "frame_id": [str(i) for i in range(n)],
            "timestamp": t,
            "risk_score": scores,
            "risk_level": levels,
        }
    )
    # first run 0-1, second 5-6, gap = 5 - 1 - 1 = 3 non-event rows (2,3,4)
    out = frame_risk_to_events(
        df, RiskEventExtractionConfig(merge_gap_frames=3, min_duration_seconds=0.0)
    )
    # gap < 3 is false for gap=3, so 2 events
    assert len(out) == 2
    out_merge = frame_risk_to_events(
        df, RiskEventExtractionConfig(merge_gap_frames=4, min_duration_seconds=0.0)
    )
    # gap=3 < 4 → 1 event
    assert len(out_merge) == 1


def test_min_duration_drops_short_event() -> None:
    df = _frame(3, high_ranges=[(0, 2)])
    out = frame_risk_to_events(
        df, RiskEventExtractionConfig(merge_gap_frames=0, min_duration_seconds=1.0)
    )
    # duration 2*DT = 0.2 < 1.0
    assert len(out) == 0

    out_ok = frame_risk_to_events(
        df, RiskEventExtractionConfig(merge_gap_frames=0, min_duration_seconds=0.15)
    )
    assert len(out_ok) == 1


def test_empty_input() -> None:
    out = frame_risk_to_events(pd.DataFrame(), RiskEventExtractionConfig())
    assert len(out) == 0
    assert list(out.columns)


def test_frame_risk_to_events_list() -> None:
    df = _frame(4, high_ranges=[(0, 3)])
    rows = frame_risk_to_events_list(df, RiskEventExtractionConfig(min_duration_seconds=0.0))
    assert len(rows) == 1
    assert rows[0]["event_id"] == 0
    assert "duration_seconds" in rows[0]


def test_config_rejects_invalid_merge_gap() -> None:
    with pytest.raises(ValueError, match="merge_gap"):
        RiskEventExtractionConfig(merge_gap_frames=-1)


def test_unsorted_table_sorted_by_time_before_runs() -> None:
    """Rows out of time order: sorted by ``timestamp`` then ``frame_id`` before run detection."""
    df = pd.DataFrame(
        {
            "frame_id": ["b", "a"],
            "frame_index": [1, 0],
            "timestamp": [0.2, 0.0],
            "risk_score": [0.9, 0.1],
            "risk_level": ["CRITICAL", "LOW"],
        }
    )
    out = frame_risk_to_events(df, RiskEventExtractionConfig(min_duration_seconds=0.0))
    assert len(out) == 1
    assert str(out.iloc[0]["start_frame"]) == "b"
    # Single high-risk row: end extends by one inferred frame step (0.2 s) after t=0.2
    assert out.iloc[0]["start_time"] == pytest.approx(0.2, abs=1e-6)
    assert out.iloc[0]["end_time"] == pytest.approx(0.4, abs=1e-6)
    assert float(out.iloc[0]["duration_seconds"]) > 0.0


def test_single_high_frame_gets_positive_duration() -> None:
    """One isolated HIGH row: start_time < end_time (one frame step)."""
    df = pd.DataFrame(
        {
            "frame_id": ["0", "1", "2"],
            "timestamp": [0.0, 0.1, 0.2],
            "risk_score": [0.0, 0.9, 0.0],
            "risk_level": ["LOW", "HIGH", "LOW"],
        }
    )
    out = frame_risk_to_events(
        df,
        RiskEventExtractionConfig(merge_gap_frames=0, min_duration_seconds=0.0, fps=10.0),
    )
    assert len(out) == 1
    st = float(out.iloc[0]["start_time"])
    en = float(out.iloc[0]["end_time"])
    assert st < en
    assert en == pytest.approx(st + 0.1)
    assert float(out.iloc[0]["duration_seconds"]) == pytest.approx(0.1)


def test_multi_frame_event_strictly_positive_duration() -> None:
    df = _frame(4, high_ranges=[(1, 2)])
    out = frame_risk_to_events(df, RiskEventExtractionConfig(min_duration_seconds=0.0))
    assert len(out) == 1
    assert float(out.iloc[0]["start_time"]) < float(out.iloc[0]["end_time"])


def test_merged_runs_strictly_positive_duration() -> None:
    df = pd.DataFrame(
        {
            "frame_id": [str(i) for i in range(5)],
            "timestamp": [0.0, 0.1, 0.2, 0.3, 0.4],
            "risk_score": [0.6, 0.6, 0.0, 0.7, 0.7],
            "risk_level": ["HIGH", "HIGH", "LOW", "HIGH", "HIGH"],
        }
    )
    merged = frame_risk_to_events(
        df, RiskEventExtractionConfig(merge_gap_frames=2, min_duration_seconds=0.0)
    )
    assert len(merged) == 1
    assert float(merged.iloc[0]["start_time"]) < float(merged.iloc[0]["end_time"])


def test_events_never_zero_duration() -> None:
    """Every emitted row satisfies end > start (QA-compatible)."""
    df = pd.DataFrame(
        {
            "frame_id": ["a", "b", "c", "d", "e"],
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5],
            "risk_score": [0.9, 0.05, 0.9, 0.05, 0.9],
            "risk_level": ["CRITICAL", "LOW", "HIGH", "LOW", "CRITICAL"],
        }
    )
    out = frame_risk_to_events(
        df, RiskEventExtractionConfig(merge_gap_frames=2, min_duration_seconds=0.0)
    )
    for _, row in out.iterrows():
        assert float(row["start_time"]) < float(row["end_time"])
        assert float(row["duration_seconds"]) > 0.0


def test_regression_sparse_high_risk_json_compatible_intervals() -> None:
    """Isolated high-risk frames produce strict intervals for ``events.json`` / QA."""
    import json

    df = pd.DataFrame(
        {
            "frame_id": [str(i) for i in range(10)],
            "timestamp": [i * 0.1 for i in range(10)],
            "risk_score": [0.05] * 10,
            "risk_level": ["LOW"] * 10,
        }
    )
    for idx in (1, 3, 7):
        df.loc[idx, "risk_level"] = "HIGH"
        df.loc[idx, "risk_score"] = 0.85
    evs = frame_risk_to_events_list(
        df, RiskEventExtractionConfig(merge_gap_frames=2, min_duration_seconds=0.0)
    )
    raw = json.dumps(evs)
    back = json.loads(raw)
    assert isinstance(back, list)
    for ev in back:
        assert float(ev["start_time"]) < float(ev["end_time"])
