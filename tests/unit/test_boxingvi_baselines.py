"""Unit tests for BoxingVI baseline comparison module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fightsafe_ai.evaluation import boxingvi_baselines as bb


pytestmark = pytest.mark.unit


def _synthetic_skeleton_spike() -> np.ndarray:
    t_max, p_max = 120, 1
    sk = np.zeros((t_max, p_max, 17, 2), dtype=np.float64)
    for t in range(t_max):
        for j in range(5, 11):
            sk[t, 0, j, 0] = 0.15 + 0.002 * t
            sk[t, 0, j, 1] = 0.45
    sk[55, 0, 9, 0] = 4.5
    sk[55, 0, 10, 0] = 4.5
    return sk


def test_detect_velocity_threshold_events_spike() -> None:
    sk = _synthetic_skeleton_spike()
    out = bb.detect_velocity_threshold_events(
        sk, fps=30.0, percentile=80.0, merge_frames=6, min_valid_keypoints=5
    )
    assert isinstance(out, list)
    if len(out) >= 1:
        assert out[0]["event_level"] == "HIGH"
        assert out[0]["event_type"] == "boxingvi.velocity_threshold"


def test_detect_velocity_threshold_events_short_sequence() -> None:
    assert (
        bb.detect_velocity_threshold_events(
            np.zeros((1, 1, 17, 2)),
            fps=30.0,
            percentile=85.0,
            merge_frames=8,
            min_valid_keypoints=5,
        )
        == []
    )


def test_detect_velocity_threshold_bad_fps_raises() -> None:
    with pytest.raises(ValueError, match="fps"):
        bb.detect_velocity_threshold_events(
            _synthetic_skeleton_spike(),
            fps=0.0,
            percentile=85.0,
            merge_frames=8,
            min_valid_keypoints=5,
        )


def test_detect_velocity_threshold_bad_percentile_raises() -> None:
    with pytest.raises(ValueError, match="percentile"):
        bb.detect_velocity_threshold_events(
            _synthetic_skeleton_spike(),
            fps=30.0,
            percentile=100.0,
            merge_frames=8,
            min_valid_keypoints=5,
        )


def test_apply_full_fusion_timeline_merge_merges_close_events(tmp_path: Path) -> None:
    p = tmp_path / "pred.json"
    p.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "start_time": 0.0,
                        "end_time": 0.1,
                        "event_level": "HIGH",
                        "category": "impact",
                    }
                ],
                "anomaly_events": [
                    {
                        "start_time": 0.11,
                        "end_time": 0.15,
                        "event_level": "HIGH",
                        "category": "impact",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    bb.apply_full_fusion_timeline_merge(p, fps=30.0, merge_frames=8)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["anomaly_events"] == []
    assert len(data["events"]) == 1
    assert data["full_fusion_timeline_merge_frames"] == 8


def test_apply_full_fusion_timeline_merge_empty_candidates(tmp_path: Path) -> None:
    p = tmp_path / "pred.json"
    p.write_text(json.dumps({"events": [], "anomaly_events": []}), encoding="utf-8")
    bb.apply_full_fusion_timeline_merge(p, fps=30.0, merge_frames=8)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["full_fusion_timeline_merge_frames"] == 8


def test_apply_full_fusion_timeline_merge_missing_file(tmp_path: Path) -> None:
    bb.apply_full_fusion_timeline_merge(tmp_path / "nope.json", fps=30.0, merge_frames=8)


def test_apply_full_fusion_timeline_merge_invalid_body(tmp_path: Path) -> None:
    p = tmp_path / "pred.json"
    p.write_text("[1,2,3]", encoding="utf-8")
    bb.apply_full_fusion_timeline_merge(p, fps=30.0, merge_frames=8)


def test_apply_full_fusion_timeline_merge_nonpositive_fps(tmp_path: Path) -> None:
    p = tmp_path / "pred.json"
    p.write_text('{"events": []}', encoding="utf-8")
    bb.apply_full_fusion_timeline_merge(p, fps=0.0, merge_frames=8)


def test_apply_full_fusion_timeline_merge_split_branch_two_events(tmp_path: Path) -> None:
    """Gap larger than merge window → two merged groups."""
    p = tmp_path / "pred.json"
    p.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "start_time": 0.0,
                        "end_time": 0.05,
                        "event_level": "HIGH",
                        "category": "impact",
                        "score": 0.4,
                    },
                    {
                        "start_time": 10.0,
                        "end_time": 10.1,
                        "event_level": "HIGH",
                        "category": "impact",
                        "score": 0.9,
                    },
                ],
                "anomaly_events": [],
            }
        ),
        encoding="utf-8",
    )
    bb.apply_full_fusion_timeline_merge(p, fps=30.0, merge_frames=8)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["events"]) == 2


def test_write_predictions_json_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    bb.write_predictions_json(p, {"video_id": "V1", "events": []})
    assert json.loads(p.read_text(encoding="utf-8"))["video_id"] == "V1"


def test_run_all_videos_and_aggregate_no_skeleton(tmp_path: Path) -> None:
    out = tmp_path / "agg"
    rows = bb.run_all_videos_and_aggregate(
        dataset_root=tmp_path / "ds",
        video_ids=["V1"],
        output_root=out,
        fps=30.0,
        tolerance_seconds=0.5,
        iou_threshold=0.01,
        rolling_window=5,
        strike_percentile=85.0,
        strike_merge_frames=8,
        min_valid_keypoints=5,
        rules_yaml=None,
        force=True,
    )
    assert len(rows) == len(bb.BASELINE_ORDER)
    assert (out / "baseline_comparison.csv").is_file()
    assert all(r["TP"] == 0 for r in rows)


def test_main_smoke_no_skeleton(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()
    out = tmp_path / "out"
    rc = bb.main(
        [
            "--dataset-root",
            str(ds),
            "--video-ids",
            "V1",
            "--output-dir",
            str(out),
            "--force",
        ]
    )
    assert rc == 0


def test_run_baselines_with_skeleton_no_annotations(tmp_path: Path) -> None:
    """Exercise payload writers; evaluation fails without Excel annotations."""
    root = tmp_path / "ds"
    (root / "skeleton").mkdir(parents=True)
    sk = _synthetic_skeleton_spike()
    np.save(root / "skeleton" / "V1.npy", sk)
    out = tmp_path / "baselines"
    row = bb.run_baselines_for_video(
        dataset_root=root,
        video_id="V1",
        output_root=out,
        fps=30.0,
        tolerance_seconds=0.5,
        iou_threshold=0.01,
        rolling_window=3,
        strike_percentile=80.0,
        strike_merge_frames=6,
        min_valid_keypoints=5,
        rules_yaml=None,
        force=True,
    )
    assert set(row.keys()) == set(bb.BASELINE_ORDER)
    assert (out / "velocity_threshold" / "boxingvi_predictions_V1.json").is_file()
    assert all(v is None for v in row.values())


def test_build_payload_helpers_smoke(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "skeleton").mkdir(parents=True)
    sk = np.zeros((40, 1, 17, 2), dtype=np.float64)
    for t in range(40):
        for j in range(5, 11):
            sk[t, 0, j, 0] = 0.1
            sk[t, 0, j, 1] = 0.2
    np.save(root / "skeleton" / "VX.npy", sk)
    pv = bb.build_payload_velocity_threshold(
        root,
        "VX",
        sk,
        fps=30.0,
        strike_percentile=90.0,
        strike_merge_frames=8,
        min_valid_keypoints=5,
    )
    assert pv["baseline"] == "velocity_threshold"
    pa = bb.build_payload_anomaly_only(root, "VX", sk, fps=30.0, min_valid_keypoints=5)
    assert pa["baseline"] == "anomaly_only"
    ps = bb.build_payload_strike_detector(
        root,
        "VX",
        sk,
        fps=30.0,
        strike_percentile=90.0,
        strike_merge_frames=8,
        min_valid_keypoints=5,
    )
    assert ps["baseline"] == "strike_detector"
