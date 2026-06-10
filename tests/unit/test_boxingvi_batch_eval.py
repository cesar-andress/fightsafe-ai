"""Batch BoxingVI evaluation (skeleton + metrics)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fightsafe_ai.evaluation.boxingvi_batch_eval import (
    _compute_micro_macro,
    _write_aggregate_table_csv,
    _write_aggregate_table_tex,
    run_batch,
)


pytest.importorskip("openpyxl")

pytestmark = pytest.mark.unit


def _tiny_layout(root: Path) -> None:
    ann = root / "annotations"
    sk = root / "skeleton"
    ann.mkdir(parents=True)
    sk.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "Start Frame": [0, 100],
            "End Frame": [30, 160],
            "Class": ["jab", "cross"],
        }
    )
    df.to_excel(ann / "V1.xlsx", index=False, engine="openpyxl")
    np.save(sk / "V1.npy", np.zeros((200, 17, 2), dtype=np.float32))


def test_compute_micro_macro_two_videos() -> None:
    from fightsafe_ai.evaluation.boxingvi_evaluator import BoxingVIEvalResult
    from fightsafe_ai.evaluation.metrics import EventEvaluationResult

    raw1 = EventEvaluationResult(
        n_predicted=3,
        n_ground_truth=2,
        true_positives=2,
        false_positives=1,
        false_negatives=0,
        precision=2 / 3,
        recall=1.0,
        f1=0.8,
        iou_threshold=0.1,
        tolerance_seconds=0.5,
        require_same_label=False,
        matches=[],
        mean_onset_delay_seconds=0.0,
        mean_abs_onset_delay_seconds=0.0,
    )
    raw2 = EventEvaluationResult(
        n_predicted=0,
        n_ground_truth=1,
        true_positives=0,
        false_positives=0,
        false_negatives=1,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        iou_threshold=0.1,
        tolerance_seconds=0.5,
        require_same_label=False,
        matches=[],
        mean_onset_delay_seconds=0.0,
        mean_abs_onset_delay_seconds=0.0,
    )
    r1 = BoxingVIEvalResult(
        video_id="A",
        n_ground_truth_punches=2,
        n_predicted_qualified=3,
        true_positives=2,
        false_positives=1,
        false_negatives=0,
        precision=raw1.precision,
        recall=raw1.recall,
        f1=raw1.f1,
        mean_detection_latency_seconds=0.1,
        mean_abs_detection_latency_seconds=0.1,
        iou_threshold=0.1,
        tolerance_seconds=0.5,
        annotation_path="a",
        predictions_path="b",
        raw=raw1,
    )
    r2 = BoxingVIEvalResult(
        video_id="B",
        n_ground_truth_punches=1,
        n_predicted_qualified=0,
        true_positives=0,
        false_positives=0,
        false_negatives=1,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        mean_detection_latency_seconds=0.0,
        mean_abs_detection_latency_seconds=0.0,
        iou_threshold=0.1,
        tolerance_seconds=0.5,
        annotation_path="a",
        predictions_path="b",
        raw=raw2,
    )
    s = _compute_micro_macro([r1, r2])
    assert s["micro_tp"] == 2
    assert s["micro_fp"] == 1
    assert s["micro_fn"] == 1
    assert s["macro_f1"] == pytest.approx(0.4)


def test_run_batch_one_video(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _tiny_layout(root)
    out = tmp_path / "out"
    rows, summary = run_batch(
        dataset_root=root,
        video_ids=["V1"],
        fps=30.0,
        output_dir=out,
        tolerance_seconds=0.5,
        iou_threshold=0.01,
        strike_percentile=90.0,
        strike_merge_frames=8,
        enable_strike_detector=True,
    )
    assert len(rows) == 1
    assert rows[0].status == "OK"
    assert (out / "boxingvi_predictions_V1.json").is_file()
    assert (out / "boxingvi_results_V1.csv").is_file()
    assert (out / "boxingvi_results_all.csv").is_file() is False  # main() writes that
    assert "micro_f1" in summary


def test_run_batch_cached_when_results_csv_exists(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _tiny_layout(root)
    out = tmp_path / "out"
    run_batch(
        dataset_root=root,
        video_ids=["V1"],
        fps=30.0,
        output_dir=out,
        tolerance_seconds=0.5,
        iou_threshold=0.01,
        strike_percentile=90.0,
        strike_merge_frames=8,
        enable_strike_detector=True,
    )
    rows, summary = run_batch(
        dataset_root=root,
        video_ids=["V1"],
        fps=30.0,
        output_dir=out,
        tolerance_seconds=0.5,
        iou_threshold=0.01,
        strike_percentile=90.0,
        strike_merge_frames=8,
        enable_strike_detector=True,
        overwrite=False,
    )
    assert len(rows) == 1
    assert rows[0].status == "CACHED"
    assert rows[0].tp is not None
    assert "micro_f1" in summary


def test_aggregate_csv_columns(tmp_path: Path) -> None:
    from fightsafe_ai.evaluation.boxingvi_batch_eval import VideoBatchRow

    rows = [
        VideoBatchRow(
            video_id="V1",
            status="OK",
            tp=1,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            mean_latency_seconds=0.05,
            error=None,
        )
    ]
    summary = {
        "micro_tp": 1,
        "micro_fp": 0,
        "micro_fn": 0,
        "micro_precision": 1.0,
        "micro_recall": 1.0,
        "micro_f1": 1.0,
        "macro_f1": 1.0,
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "mean_latency_mean": 0.05,
    }
    path = tmp_path / "boxingvi_results_all.csv"
    _write_aggregate_table_csv(path, rows, summary)
    text = path.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert header.startswith("video_id,TP,FP,FN,precision,recall,F1,mean_latency,status")
    assert "__micro__" in text and "__macro__" in text


def test_aggregate_tex_escapes_micro_macro_underscores(tmp_path: Path) -> None:
    from fightsafe_ai.evaluation.boxingvi_batch_eval import VideoBatchRow

    rows = [
        VideoBatchRow(
            video_id="V1",
            status="OK",
            tp=1,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            mean_latency_seconds=0.05,
            error=None,
        )
    ]
    summary = {
        "micro_tp": 1,
        "micro_fp": 0,
        "micro_fn": 0,
        "micro_precision": 1.0,
        "micro_recall": 1.0,
        "micro_f1": 1.0,
        "macro_f1": 1.0,
        "macro_precision": 1.0,
        "macro_recall": 1.0,
        "mean_latency_mean": 0.05,
    }
    path = tmp_path / "boxingvi_results_all.tex"
    _write_aggregate_table_tex(path, rows, summary)
    tex = path.read_text(encoding="utf-8")
    assert r"\textbf{\_\_micro\_\_}" in tex and r"\textbf{\_\_macro\_\_}" in tex
    assert r"\textbf{__micro__}" not in tex
