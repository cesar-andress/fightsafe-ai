""":func:`fightsafe_ai.reports.summary.build_summary_dict` edge cases."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fightsafe_ai.reports.summary import build_summary_dict, infer_input_video_path, load_qa_dict


pytestmark = pytest.mark.unit


def test_build_summary_dict_with_qa_pass(tmp_path: Path) -> None:
    run = tmp_path / "s"
    run.mkdir()
    pd.DataFrame(
        {
            "frame_id": [1, 1, 2],
            "timestamp": [0.0, 0.5, 1.0],
            "risk_score": [0.1, 0.2, 0.3],
        }
    ).to_csv(run / "risk_scores.csv", index=False)
    (run / "events.json").write_text(
        json.dumps(
            [
                {
                    "event_id": 0,
                    "event_level": "HIGH",
                    "start_time": 0.0,
                    "end_time": 0.5,
                }
            ]
        ),
        encoding="utf-8",
    )
    (run / "qa_report.json").write_text(
        json.dumps({"passed": True, "total_checks": 3, "failed_checks": 0}),
        encoding="utf-8",
    )
    d = build_summary_dict(run)
    assert d["qa_status"] == "pass"
    assert d["highest_event_level"] == "HIGH"
    assert d["number_of_events"] == 1
    assert d["total_frames"] == 2


def test_build_summary_dict_qa_fail_status(tmp_path: Path) -> None:
    run = tmp_path / "s2"
    run.mkdir()
    pd.DataFrame(
        {
            "frame_id": [0],
            "timestamp": [0.0],
            "risk_score": [0.0],
        }
    ).to_csv(run / "risk_scores.csv", index=False)
    (run / "events.json").write_text("[]", encoding="utf-8")
    (run / "qa_report.json").write_text(
        json.dumps({"passed": False, "total_checks": 1, "failed_checks": 1}),
        encoding="utf-8",
    )
    d = build_summary_dict(run)
    assert d["qa_status"] == "fail"


def test_infer_input_video_from_report_md(tmp_path: Path) -> None:
    p = tmp_path / "report.md"
    p.write_text("We analyzed `data/input_clip.mp4` in this run.\n", encoding="utf-8")
    assert infer_input_video_path(p) is not None


def test_load_qa_dict_malformed_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "qa.json"
    p.write_text("not json", encoding="utf-8")
    assert load_qa_dict(p) is None
