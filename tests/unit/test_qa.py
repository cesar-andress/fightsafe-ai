"""
Unit tests: QA for missing run artifacts and invalid CSV.
"""

from __future__ import annotations

from pathlib import Path

from fightsafe_ai.qa.validators import check_required_artifacts, try_read_csv


def test_missing_files_detected_in_run_dir(tmp_path: Path) -> None:
    """With no files created, presence checks fail (or warn as configured)."""
    run = tmp_path / "empty_run"
    run.mkdir()
    results = check_required_artifacts(run, require_frames=True)
    names = {r.name: r.status for r in results}
    assert names.get("file_pose_keypoints_csv") == "fail"
    assert names.get("file_risk_scores_csv") == "fail"
    assert names.get("file_events_json") == "fail"
    assert names.get("dir_frames") == "fail"


def test_required_artifacts_with_optional_frames_warns(tmp_path: Path) -> None:
    """With require_frames=False, missing frames/ is not a failure."""
    run = tmp_path / "no_frames"
    run.mkdir()
    r = check_required_artifacts(run, require_frames=False)
    by_name = {x.name: x for x in r}
    assert by_name["dir_frames"].status == "warn"


def test_invalid_csv_read_returns_fail(tmp_path: Path) -> None:
    """Unparseable CSV for pandas -> status fail, no DataFrame."""
    bad = tmp_path / "broken.csv"
    # Unbalanced quotes -> ParserError
    bad.write_text('a,"b\nc', encoding="utf-8")
    res, dff = try_read_csv(bad, "test_label")
    assert res.status == "fail"
    assert dff is None
    assert "Parser" in res.message or "Parser" in (res.details or "")
