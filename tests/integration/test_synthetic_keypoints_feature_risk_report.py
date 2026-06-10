"""
Integration: consolidated keypoints CSV -> features (biomechanics) -> risk engine -> Markdown report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fightsafe_ai.features.biomechanics import compute_pose_features
from fightsafe_ai.reports.markdown import generate_markdown_report
from fightsafe_ai.risk.engine import detect_risk_events


pytestmark = [pytest.mark.integration]


def _toy_consolidated_pose(p: Path) -> None:
    rows = []
    for fid, hip_y in [("0", 0.45), ("1", 0.50), ("2", 0.75)]:
        for name, x, y in [
            ("nose", 0.5, 0.2),
            ("left_shoulder", 0.4, 0.3),
            ("right_shoulder", 0.6, 0.3),
            ("left_hip", 0.42, hip_y),
            ("right_hip", 0.58, hip_y),
        ]:
            rows.append(
                {
                    "frame_id": fid,
                    "keypoint_name": name,
                    "x": x,
                    "y": y,
                }
            )
    pd.DataFrame(rows).to_csv(p, index=False)


def test_compute_features_detect_risk_generate_markdown(tmp_path: Path) -> None:
    pose = tmp_path / "pose_keypoints.csv"
    _toy_consolidated_pose(pose)
    features = tmp_path / "features.csv"
    feat_df = compute_pose_features(pose, fps=10.0, rolling_window=2, ground_y_threshold=0.6)
    feat_df.to_csv(features, index=False)

    risk = detect_risk_events(feat_df, None)
    risk.to_csv(tmp_path / "risk_scores.csv", index=False)

    evs = [
        {
            "event_id": 0,
            "start_time": 0.0,
            "end_time": 0.2,
            "max_risk_score": 0.5,
            "event_level": "MEDIUM",
        }
    ]
    (tmp_path / "events.json").write_text(json.dumps(evs, indent=2) + "\n", encoding="utf-8")

    out_md = generate_markdown_report(tmp_path, tmp_path / "report.md")
    text = out_md.read_text(encoding="utf-8")
    assert "risk" in text.lower() or "event" in text.lower() or "run" in text.lower()
    assert len(text) > 20
