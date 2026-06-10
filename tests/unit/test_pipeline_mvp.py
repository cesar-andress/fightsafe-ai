"""Unit tests for pure helpers in the MVP pipeline (no I/O, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fightsafe_ai.pipeline.mvp import (
    risk_scores_dataframe_for_csv,
    sanitize_for_json,
)
from fightsafe_ai.pipeline.mvp_report import (
    MVPReportConfig,
    generate_mvp_report_markdown,
    write_mvp_report,
)


def test_sanitize_for_json_primitives() -> None:
    assert sanitize_for_json(None) is None
    assert sanitize_for_json(3) == 3
    assert sanitize_for_json(np.int64(4)) == 4
    assert sanitize_for_json(1.5) == 1.5
    assert sanitize_for_json(float("nan")) is None
    assert sanitize_for_json([np.float64(2.0), "a"]) == [2.0, "a"]


def test_sanitize_for_json_nested_dict() -> None:
    d = {"a": np.int32(1), "b": {"c": np.array([1.0, 2.0])}}
    out = sanitize_for_json(d)
    assert out == {"a": 1, "b": {"c": [1.0, 2.0]}}


def test_risk_scores_dataframe_for_csv_serializes_rules() -> None:
    df = pd.DataFrame(
        {
            "frame_id": ["a"],
            "risk_score": [0.5],
            "triggered_rules": [["rule_a", "rule_b"]],
        }
    )
    w = risk_scores_dataframe_for_csv(df)
    cell = w["triggered_rules"].iloc[0]
    assert json.loads(cell) == ["rule_a", "rule_b"]


def test_generate_mvp_report_markdown_sections(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    ev = root / "events.json"
    ev.write_text("[]", encoding="utf-8")
    risk = root / "risk.csv"
    pd.DataFrame(
        {
            "frame_id": ["f1", "f2"],
            "timestamp": [0.0, 0.1],
            "risk_score": [0.1, 0.9],
            "risk_level": ["LOW", "CRITICAL"],
            "triggered_rules": ["[]", '["fast_downward_motion"]'],
        }
    ).to_csv(risk, index=False)

    cfg = MVPReportConfig(
        video_path=Path("demo.mp4"),
        output_root=root,
        events_path=ev,
        risk_scores_path=risk,
        rules_config_path=None,
        sampling_fps=10.0,
        n_sampled_frames=2,
    )
    md = generate_mvp_report_markdown(cfg)
    assert "## 1. Demo clip" in md
    assert "## 2. Pipeline outputs" in md
    assert "## 3. Detected risk events" in md
    assert "## 4. Highest risk moment" in md
    assert "## 5. Triggered rules" in md
    assert "## 6. Human review recommendation" in md
    assert "## 7. Limitations" in md
    assert "## 8. Safety disclaimer" in md
    assert "0.9000" in md
    assert "fast_downward_motion" in md

    out = root / "report.md"
    assert write_mvp_report(out, cfg) == out.resolve()
    assert out.is_file() and len(out.read_text(encoding="utf-8")) > 200
