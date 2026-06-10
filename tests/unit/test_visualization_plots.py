"""Tests for matplotlib run plots (no display)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests.fixtures.mvp_runs import write_minimal_pipeline_run

from fightsafe_ai.visualization.plots import (
    plot_event_timeline,
    plot_events_timeline,
    plot_risk_timeline,
)


pytestmark = pytest.mark.slow


def test_plot_risk_and_events_timeline(tmp_path: Path) -> None:
    r = tmp_path / "run1"
    write_minimal_pipeline_run(
        r,
        include_frames_dir=False,
        event={"start_time": 0.05, "end_time": 0.15, "max_risk_score": 0.8, "event_level": "HIGH"},
    )
    p1 = plot_risk_timeline(r, r / "risk_timeline.png")
    p2 = plot_events_timeline(r, r / "events_timeline.png")
    assert p1.is_file() and p1.stat().st_size > 100
    assert p2.is_file() and p2.stat().st_size > 100
    p1b = plot_risk_timeline(r)
    p2b = plot_event_timeline(r)
    assert p1b == r / "risk_timeline.png" and p2b == r / "events_timeline.png"


def test_plot_events_timeline_empty_events(tmp_path: Path) -> None:
    r = tmp_path / "empty_ev"
    r.mkdir()
    pd.DataFrame(
        {
            "timestamp": [0.0, 1.0],
            "risk_score": [0.1, 0.2],
            "risk_level": ["LOW", "LOW"],
        }
    ).to_csv(r / "risk_scores.csv", index=False)
    (r / "events.json").write_text("[]", encoding="utf-8")
    p = plot_events_timeline(r, r / "e.png")
    assert p.is_file() and p.stat().st_size > 50
