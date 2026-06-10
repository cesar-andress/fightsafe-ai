"""BoxingVI event-level evaluation (punch GT vs impact-like predictions)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fightsafe_ai.evaluation.boxingvi_evaluator import (
    evaluate_boxingvi_video,
    load_ground_truth_impact_windows,
    load_prediction_impact_windows,
    resolve_skeleton_frame_count,
    write_boxingvi_results_csv,
    write_boxingvi_results_tex,
)


pytest.importorskip("openpyxl")

pytestmark = pytest.mark.unit


def _write_xlsx_punches(path: Path) -> None:
    """One jab from 0.0s to ~1.0s at 30 fps (frames 0–29 inclusive)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "Start Frame": [0],
            "End Frame": [29],
            "Class": ["Jab"],
        }
    )
    df.to_excel(path, index=False, engine="openpyxl")


def test_load_gt_from_annotation_files_dir(tmp_path: Path) -> None:
    ann = tmp_path / "Annotation_files" / "V1.xlsx"
    _write_xlsx_punches(ann)
    w = load_ground_truth_impact_windows(dataset_root=tmp_path, video_id="V1", fps=30.0)
    assert len(w) == 1
    assert abs(w[0].start - 0.0) < 1e-9
    assert abs(w[0].end - 1.0) < 1e-9  # (29+1)/30


def test_prediction_filter_risk_event(tmp_path: Path) -> None:
    p = tmp_path / "pred.json"
    p.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "start_time": 0.2,
                        "end_time": 0.5,
                        "event_level": "HIGH",
                    }
                ],
                "anomaly_events": [],
            }
        ),
        encoding="utf-8",
    )
    w = load_prediction_impact_windows(p)
    assert len(w) == 1
    assert w[0].start == 0.2


def test_gt_auto_frame_offset_infers_min_start(tmp_path: Path) -> None:
    """Skeleton is local 0..T-1; annotations use global indices starting at 1000."""
    (tmp_path / "Annotation_files").mkdir(parents=True)
    ann = tmp_path / "Annotation_files" / "V1.xlsx"
    df = pd.DataFrame(
        {
            "Start_Frame": [1000, 2000],
            "Ending_Frame": [1010, 2010],
            "Class": ["jab", "cross"],
        }
    )
    df.to_excel(ann, index=False, engine="openpyxl")

    sk = tmp_path / "skeleton"
    sk.mkdir(parents=True, exist_ok=True)
    np.save(sk / "V1.npy", np.zeros((100, 17, 2), dtype=np.float32))

    w = load_ground_truth_impact_windows(
        dataset_root=tmp_path,
        video_id="V1",
        fps=30.0,
        num_skeleton_frames=100,
        frame_offset=None,
        no_frame_offset=False,
    )
    assert len(w) == 1
    assert abs(w[0].start - 0.0) < 1e-9
    assert abs(w[0].end - (11.0 / 30.0)) < 1e-6


def test_resolve_skeleton_frame_count_multi_dim(tmp_path: Path) -> None:
    sk = tmp_path / "skeleton"
    sk.mkdir(parents=True)
    np.save(sk / "V1.npy", np.zeros((1866, 25, 17, 2), dtype=np.float32))
    assert resolve_skeleton_frame_count(tmp_path, "V1") == 1866


def test_no_frame_offset_keeps_global_indices(tmp_path: Path) -> None:
    """Without alignment, global frame indices map directly to timestamps (no skeleton clipping)."""
    (tmp_path / "Annotation_files").mkdir(parents=True)
    ann = tmp_path / "Annotation_files" / "V1.xlsx"
    pd.DataFrame({"Start_Frame": [1000], "Ending_Frame": [1010], "Class": ["jab"]}).to_excel(
        ann, index=False, engine="openpyxl"
    )

    w = load_ground_truth_impact_windows(
        dataset_root=tmp_path,
        video_id="V1",
        fps=30.0,
        num_skeleton_frames=100,
        frame_offset=None,
        no_frame_offset=True,
    )
    assert len(w) == 1
    assert abs(w[0].start - (1000.0 / 30.0)) < 1e-6


def test_explicit_frame_offset(tmp_path: Path) -> None:
    (tmp_path / "Annotation_files").mkdir(parents=True)
    ann = tmp_path / "Annotation_files" / "V1.xlsx"
    pd.DataFrame({"Start_Frame": [3314], "Ending_Frame": [3320], "Class": ["jab"]}).to_excel(
        ann, index=False, engine="openpyxl"
    )

    w = load_ground_truth_impact_windows(
        dataset_root=tmp_path,
        video_id="V1",
        fps=30.0,
        num_skeleton_frames=1866,
        frame_offset=3314,
        no_frame_offset=False,
    )
    assert len(w) == 1
    assert abs(w[0].start - 0.0) < 1e-9
    assert abs(w[0].end - (7.0 / 30.0)) < 1e-6


def test_evaluate_uses_skeleton_length_for_auto_offset(tmp_path: Path) -> None:
    """Predictions in local seconds overlap GT after auto alignment from skeleton/T."""
    (tmp_path / "Annotation_files").mkdir(parents=True)
    pd.DataFrame(
        {
            "Start_Frame": [1000],
            "Ending_Frame": [1010],
            "Class": ["jab"],
        }
    ).to_excel(tmp_path / "Annotation_files" / "V1.xlsx", index=False, engine="openpyxl")

    sk = tmp_path / "skeleton"
    sk.mkdir(parents=True)
    np.save(sk / "V1.npy", np.zeros((100, 17, 2), dtype=np.float32))

    pred_path = tmp_path / "boxingvi_predictions_V1.json"
    pred_path.write_text(
        json.dumps(
            {
                "events": [
                    {"start_time": 0.05, "end_time": 0.30, "event_level": "HIGH"},
                ],
            }
        ),
        encoding="utf-8",
    )

    r = evaluate_boxingvi_video(
        dataset_root=tmp_path,
        video_id="V1",
        predictions_path=pred_path,
        fps=30.0,
        tolerance_seconds=1.0,
        iou_threshold=0.01,
    )
    assert r.n_ground_truth_punches == 1
    assert r.true_positives == 1


def test_end_to_end_tp(tmp_path: Path) -> None:
    (tmp_path / "Annotation_files").mkdir()
    _write_xlsx_punches(tmp_path / "Annotation_files" / "V1.xlsx")

    pred_path = tmp_path / "boxingvi_predictions_V1.json"
    pred_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "start_time": 0.1,
                        "end_time": 0.4,
                        "event_level": "HIGH",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    r = evaluate_boxingvi_video(
        dataset_root=tmp_path,
        video_id="V1",
        predictions_path=pred_path,
        fps=30.0,
        tolerance_seconds=0.5,
        iou_threshold=0.01,
    )
    assert r.n_ground_truth_punches == 1
    assert r.true_positives == 1
    assert r.false_positives == 0
    assert r.false_negatives == 0
    assert r.precision == 1.0
    assert r.recall == 1.0

    out_csv = tmp_path / "out.csv"
    out_tex = tmp_path / "out.tex"
    write_boxingvi_results_csv(out_csv, r, append=False)
    write_boxingvi_results_tex(out_tex, r)
    assert out_csv.is_file()
    assert "V1" in out_csv.read_text()
    assert out_tex.is_file()
    assert "V1" in out_tex.read_text()
