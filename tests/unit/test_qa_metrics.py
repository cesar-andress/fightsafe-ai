"""Pure helpers in :mod:`fightsafe_ai.qa.metrics` and JSON shape for :func:`write_qa_report_json`."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fightsafe_ai.qa import metrics as qm
from fightsafe_ai.qa.validators import run_quality_checks, write_qa_report_json


def test_build_run_metrics_columns(tmp_path: Path) -> None:
    root = tmp_path
    r = pd.DataFrame(
        {
            "frame_id": ["0", "1", "2"],
            "timestamp": [0.0, 0.1, 0.2],
            "risk_score": [0.1, 0.4, 0.1],
        }
    )
    p = pd.DataFrame(
        {
            "frame_id": ["0", "0", "1", "1"],
            "keypoint_name": ["n", "l", "n", "l"],
            "x": [0, 0, 0, 0],
            "y": [0, 0, 0, 0],
        }
    )
    m = qm.build_run_metrics(
        frames_dir=root,
        risk_df=r,
        pose_df=p,
        number_of_events=0,
        pose_coverage_percent=90.0,
    )
    for k in qm.METRIC_KEYS:
        assert k in m
    assert m["number_of_events"] == 0
    assert m["max_risk_score"] == 0.4
    assert abs(m["average_risk_score"] - (0.1 + 0.4 + 0.1) / 3) < 1e-6
    assert m["frames_with_pose"] == 2
    assert m["pose_coverage_percent"] == 90.0
    assert m["duration_seconds"] == pytest.approx(0.2)


def test_is_constant_risk_score() -> None:
    a = pd.DataFrame({"risk_score": [0.3, 0.3, 0.3]})
    assert qm.is_constant_risk_score(a) is True
    b = pd.DataFrame({"risk_score": [0.3, 0.4]})
    assert qm.is_constant_risk_score(b) is False
    assert qm.is_constant_risk_score(pd.DataFrame({"risk_score": [0.1]})) is False


def test_duration_seconds_from_risk_branches() -> None:
    assert qm.duration_seconds_from_risk(None) == 0.0
    assert qm.duration_seconds_from_risk(pd.DataFrame({"risk_score": [1.0]})) == 0.0
    one = pd.DataFrame({"timestamp": [0.5], "risk_score": [0.0]})
    assert qm.duration_seconds_from_risk(one) == 0.0
    two = pd.DataFrame(
        {"timestamp": [0.0, 0.5], "risk_score": [0.1, 0.2]},
    )
    assert qm.duration_seconds_from_risk(two) == pytest.approx(0.5)
    partial = pd.DataFrame(
        {"timestamp": [0.0, float("nan")], "risk_score": [0.0, 0.0]},
    )
    assert qm.duration_seconds_from_risk(partial) == 0.0


def test_count_total_frames_without_frame_id_uses_len(tmp_path: Path) -> None:
    risk = pd.DataFrame({"nope": [1, 2], "timestamp": [0.0, 0.1]})
    assert qm.count_total_frames(tmp_path, risk) == 2


def test_average_and_max_risk_all_invalid() -> None:
    r = pd.DataFrame({"risk_score": [float("nan"), "bad"]})
    assert qm.average_risk_score(r) is None
    assert qm.max_risk_score(r) is None


def test_is_constant_risk_drops_to_short_after_coerce() -> None:
    r = pd.DataFrame({"risk_score": [float("nan"), float("nan")]})
    assert qm.is_constant_risk_score(r) is False


def test_count_frames_with_pose_empty() -> None:
    assert qm.count_frames_with_pose(None) == 0
    assert qm.count_frames_with_pose(pd.DataFrame({"x": [1]})) == 0


def test_merge_metrics_fills_keys() -> None:
    m = qm.merge_metrics({"max_risk_score": 0.5})
    for k in qm.QA_REPORT_METRIC_KEYS:
        assert k in m
    assert m["max_risk_score"] == 0.5
    assert m["total_frames"] is None
    assert m["llm_enabled"] is None


def test_qa_report_json_has_canonical_metrics(tmp_path: Path) -> None:
    run = tmp_path / "r"
    run.mkdir()
    pd.DataFrame(
        {
            "frame_id": ["0", "1"],
            "timestamp": [0.0, 0.0],
            "risk_score": [0.2, 0.2],
            "risk_level": ["LOW", "LOW"],
        }
    ).to_csv(run / "risk_scores.csv", index=False)
    (run / "events.json").write_text("[]", encoding="utf-8")
    (run / "pose_keypoints.csv").write_text(
        "frame_id,keypoint_name,x,y\n0,n,0,0\n1,n,0,0\n", encoding="utf-8"
    )
    (run / "features.csv").write_text("frame_id\n0\n", encoding="utf-8")
    (run / "report.md").write_text("x", encoding="utf-8")
    (run / "output_overlay.mp4").write_bytes(b"ok")
    (run / "frames").mkdir()
    (run / "frames" / "a.jpg").write_bytes(b"\xff\xd8\xff")
    rep = run_quality_checks(run, require_frames=True)
    p = write_qa_report_json(run / "qa_report.json", rep)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "metrics" in data
    for k in qm.QA_REPORT_METRIC_KEYS:
        assert k in data["metrics"]
    assert data["metrics"]["number_of_events"] == 0
    names = {r["name"] for r in data["results"]}
    assert "metric_constant_risk" in names
    assert "metric_no_events" in names
