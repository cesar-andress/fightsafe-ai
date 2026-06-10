"""
Synthetic on-disk “run directory” trees (small CSV/JSON) for report, plot, and QA tests.

No videos; optional fake MP4 bytes for validators that require the file to exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


_MINIMAL_RISK = {
    "frame_id": [f"f{i}" for i in range(3)],
    "timestamp": [0.0, 0.1, 0.2],
    "risk_score": [0.1, 0.9, 0.2],
    "risk_level": ["LOW", "HIGH", "MEDIUM"],
}
_DEFAULT_EVENT: dict[str, Any] = {
    "event_id": 0,
    "start_time": 0.0,
    "end_time": 0.1,
    "max_risk_score": 0.8,
    "event_level": "HIGH",
}


def write_minimal_pipeline_run(
    run_dir: Path,
    *,
    include_frames_dir: bool = True,
    include_mvp_artifacts: bool = False,
    event: dict[str, Any] | None = None,
) -> Path:
    """
    Write ``risk_scores.csv`` and ``events.json`` (required for reports / plots).

    If ``include_mvp_artifacts`` is True, also add minimal ``qa_report.json``,
    ``output_overlay.mp4``, ``report.md``, and ``frames/`` so
    :func:`fightsafe_ai.qa.run_quality_checks` passes a full MVP file checklist.
    """
    run = run_dir.expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    ev = {**_DEFAULT_EVENT, **(event or {})}
    pd.DataFrame(_MINIMAL_RISK).to_csv(run / "risk_scores.csv", index=False)
    (run / "events.json").write_text(
        json.dumps([ev], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if include_mvp_artifacts:
        (run / "frames").mkdir(exist_ok=True)
        (run / "qa_report.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "total_checks": 5,
                    "failed_checks": 0,
                    "warnings": [],
                }
            ),
            encoding="utf-8",
        )
        (run / "output_overlay.mp4").write_bytes(b"fake" * 2)
        (run / "report.md").write_text("Demo `clip.mp4` in `data/`\n", encoding="utf-8")
    elif include_frames_dir:
        (run / "frames").mkdir(exist_ok=True)
    return run


def write_mvp_qa_passing_run(run_dir: Path) -> Path:
    """
    On-disk run directory that should pass :func:`fightsafe_ai.qa.quality_report.run_quality_checks`.

    Writes the standard MVP filenames with minimal valid tabular content: one ``.jpg`` in
    ``frames/``, matching ``frame_id`` coverage, monotonic timestamps, and valid risk levels.
    """
    run = run_dir.expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    frames = run / "frames"
    frames.mkdir(exist_ok=True)
    (frames / "frame_000001.jpg").write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF"
    )  # tiny valid JPEG header

    pd.DataFrame(
        {
            "frame_id": ["0", "0", "0"],
            "keypoint_name": ["nose", "left_hip", "right_hip"],
            "x": [0.5, 0.42, 0.58],
            "y": [0.2, 0.5, 0.5],
        }
    ).to_csv(run / "pose_keypoints.csv", index=False)

    pd.DataFrame({"frame_id": ["0"], "timestamp": [0.0]}).to_csv(run / "features.csv", index=False)

    pd.DataFrame(
        {
            "frame_id": ["0", "0", "0"],
            "timestamp": [0.0, 0.1, 0.2],
            "risk_score": [0.1, 0.5, 0.2],
            "risk_level": ["LOW", "MEDIUM", "LOW"],
        }
    ).to_csv(run / "risk_scores.csv", index=False)

    (run / "events.json").write_text(
        json.dumps(
            [
                {
                    "event_id": 0,
                    "start_time": 0.0,
                    "end_time": 0.15,
                    "max_risk_score": 0.5,
                    "event_level": "MEDIUM",
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run / "output_overlay.mp4").write_bytes(b"ftyp" + b"\x00" * 20)
    (run / "report.md").write_text(
        "## Run\n\nInput video: `data/clip.mp4` (synthetic test).\n",
        encoding="utf-8",
    )
    return run
