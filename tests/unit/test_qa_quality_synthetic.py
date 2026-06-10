"""Synthetic run dir for :func:`fightsafe_ai.qa.quality_report.run_quality_checks` (unit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fightsafe_ai.qa.quality_report import run_quality_checks


def test_run_quality_checks_synthetic_mvp_dir(tmp_path: Path) -> None:
    run = tmp_path / "runpack"
    run.mkdir()
    (run / "frames").mkdir()
    for i in range(1, 4):
        (run / "frames" / f"frame_{i:06d}.jpg").write_bytes(b"\xff\xd8" + b"\0" * 4)

    pd.DataFrame(
        {
            "frame_id": [f"frame_{i:06d}" for i in (1, 1, 2, 2, 3, 3)],
            "keypoint_name": ["nose", "left_hip"] * 3,
            "x": [0.5] * 6,
            "y": [0.4, 0.6] * 3,
        }
    ).to_csv(run / "pose_keypoints.csv", index=False)

    pd.DataFrame(
        {
            "frame_id": [f"frame_{i:06d}" for i in (1, 2, 3)],
            "torso_angle_deg": [10.0, 10.0, 10.0],
        }
    ).to_csv(run / "features.csv", index=False)

    pd.DataFrame(
        {
            "frame_id": [f"frame_{i:06d}" for i in (1, 2, 3)],
            "timestamp": [0.0, 0.1, 0.2],
            "risk_score": [0.1, 0.2, 0.15],
            "risk_level": ["LOW", "LOW", "MEDIUM"],
            "triggered_rules": ["[]", "[]", "[]"],
        }
    ).to_csv(run / "risk_scores.csv", index=False)

    (run / "events.json").write_text("[]", encoding="utf-8")
    (run / "output_overlay.mp4").write_bytes(b"not-really-mp4" * 5)
    (run / "report.md").write_text("# Test report\n", encoding="utf-8")

    rep = run_quality_checks(run, require_frames=True)
    assert rep.run_dir == run.resolve()
    assert rep.metrics["n_risk_rows"] == 3
    assert "pose_coverage_percent" in rep.metrics
