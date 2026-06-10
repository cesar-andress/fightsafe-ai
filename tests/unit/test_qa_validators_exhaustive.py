"""Additional branch coverage for :mod:`fightsafe_ai.qa.validators`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fightsafe_ai.qa import validators as v


pytestmark = pytest.mark.unit


def test_validate_risk_artifact_flags_non_monotonic_timestamps(tmp_path: Path) -> None:
    p = tmp_path / "risk.csv"
    pd.DataFrame(
        {
            "frame_id": ["0", "1"],
            "timestamp": [0.2, 0.1],
            "risk_score": [0.3, 0.4],
            "risk_level": ["LOW", "LOW"],
        }
    ).to_csv(p, index=False)
    chain, df = v.validate_risk_artifact(p)
    assert df is not None
    assert any(r.name == "monotonic_timestamps" and r.status == "fail" for r in chain)


def test_check_risk_level_rejects_invalid_level() -> None:
    df = pd.DataFrame(
        {
            "risk_level": ["LOW", "not_a_level", "MEDIUM"],
        }
    )
    out = v.check_risk_level_values(df)
    assert any(x.status == "fail" for x in out)


def test_check_monotonic_timestamps_all_nan() -> None:
    df = pd.DataFrame({"timestamp": [float("nan"), float("nan")]})
    out = v.check_monotonic_timestamps(df)
    assert any("fail" in r.status for r in out)


def test_check_event_time_order_bad_start_end() -> None:
    events: list[dict[str, float]] = [
        {"start_time": 1.0, "end_time": 0.0},
    ]
    out = v.check_event_time_order(events)
    assert out[0].status == "fail"


def test_load_events_list_null_json(tmp_path: Path) -> None:
    p = tmp_path / "e.json"
    p.write_text("null", encoding="utf-8")
    r, evs = v.load_events_list(p)
    assert r.status == "warn"
    assert evs is None


def test_load_events_list_invalid_root_type(tmp_path: Path) -> None:
    p = tmp_path / "e.json"
    p.write_text("3.14", encoding="utf-8")
    r, evs = v.load_events_list(p)
    assert r.status == "fail" and evs is None


def test_validate_feature_artifact_missing_frame_id(tmp_path: Path) -> None:
    p = tmp_path / "f.csv"
    pd.DataFrame({"x": [1]}).to_csv(p, index=False)
    out = v.validate_feature_artifact(p)
    assert any("features_columns" in r.name and r.status == "fail" for r in out)


def test_check_report_md_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "r.md"
    p.write_text("", encoding="utf-8")
    r = v.check_report_md(p)
    assert r.status == "fail"


def test_try_read_csv_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_file.csv"
    r, dff = v.try_read_csv(missing, "missing")
    assert r.status == "fail" and dff is None


def test_check_required_artifacts_no_frames_with_require_false(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "pose_keypoints.csv").write_text("x", encoding="utf-8")
    (run / "features.csv").write_text("x", encoding="utf-8")
    (run / "risk_scores.csv").write_text("x", encoding="utf-8")
    (run / "events.json").write_text("[]", encoding="utf-8")
    (run / "output_overlay.mp4").write_text("x", encoding="utf-8")
    (run / "report.md").write_text("#\n", encoding="utf-8")
    res = v.check_required_artifacts(run, require_frames=False)
    names = [r.name for r in res]
    assert "dir_frames" in names
    assert any(r.name == "dir_frames" and r.status == "warn" for r in res)
