"""Tests for TapKO interval evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fightsafe_ai.annotation.tapko_schema import parse_tapko_dict
from fightsafe_ai.evaluation.event_metrics import EventWindow
from fightsafe_ai.evaluation.tapko_evaluator import (
    WARN_DIAGNOSTIC_METRICS,
    TapkoEvalConfig,
    evaluate_tapko,
    labels_compatible,
    load_tapko_predictions_json,
    run_tapko_evaluation_and_write,
    tapko_namespace,
    warn_if_annotations_not_visually_confirmed,
)


def test_namespace_and_labels_compatible() -> None:
    assert tapko_namespace("submission_signal.hand_tap") == "submission_signal"
    assert (
        labels_compatible(
            "submission_signal.hand_tap",
            "submission_signal.foot_tap",
            mode="exact",
        )
        is False
    )
    assert (
        labels_compatible(
            "submission_signal.hand_tap",
            "submission_signal.foot_tap",
            mode="family",
        )
        is True
    )


def test_perfect_match_one_video(tmp_path: Path) -> None:
    gt = parse_tapko_dict(
        {
            "format_version": "1.0",
            "schema_id": "fightsafe_ai.tapko_annotation",
            "annotation_status": "visually_confirmed",
            "annotations": [
                {
                    "video_id": "v1",
                    "source_uri": "x://a.mp4",
                    "start_time": 10.0,
                    "end_time": 11.0,
                    "event_type": "submission_signal.hand_tap",
                    "visibility": "clear",
                    "occlusion_level": "none",
                    "actor_id": "a",
                    "confidence": 1.0,
                    "rater_id": "r1",
                    "requires_audio": False,
                },
            ],
        }
    )
    preds = {
        "v1": [
            EventWindow(
                start=10.0,
                end=11.0,
                label="submission_signal.hand_tap",
            ),
        ],
    }
    res = evaluate_tapko(gt, preds, config=TapkoEvalConfig(iou_threshold=0.5))
    assert res.tp == 1 and res.fp == 0 and res.fn == 0
    assert res.precision == 1.0 and res.recall == 1.0
    assert res.mean_abs_onset_latency_sec == 0.0


def test_writes_outputs(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "schema_id": "fightsafe_ai.tapko_annotation",
                "annotation_status": "visually_confirmed",
                "annotations": [
                    {
                        "video_id": "v1",
                        "source_uri": "x://a.mp4",
                        "start_time": 1.0,
                        "end_time": 2.0,
                        "event_type": "submission_signal.hand_tap",
                        "visibility": "clear",
                        "occlusion_level": "none",
                        "actor_id": "a",
                        "confidence": 1.0,
                        "rater_id": "r1",
                        "requires_audio": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(
        json.dumps(
            [
                {
                    "video_id": "v1",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "event_type": "submission_signal.hand_tap",
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    run_tapko_evaluation_and_write(gt_path, pred_path, out)
    assert (out / "tapko_results.csv").is_file()
    assert (out / "tapko_results.tex").is_file()
    assert (out / "tapko_error_analysis.md").is_file()


def test_fp_tagged_scramble_overlap(tmp_path: Path) -> None:
    gt = parse_tapko_dict(
        {
            "format_version": "1.0",
            "schema_id": "fightsafe_ai.tapko_annotation",
            "annotation_status": "visually_confirmed",
            "annotations": [
                {
                    "video_id": "v1",
                    "source_uri": "x://a.mp4",
                    "start_time": 20.0,
                    "end_time": 25.0,
                    "event_type": "negative.normal_scramble",
                    "visibility": "clear",
                    "occlusion_level": "none",
                    "actor_id": "a",
                    "confidence": 1.0,
                    "rater_id": "r1",
                    "requires_audio": False,
                },
            ],
        }
    )
    preds = {
        "v1": [
            EventWindow(
                start=21.0,
                end=22.0,
                label="submission_signal.hand_tap",
            ),
        ],
    }
    res = evaluate_tapko(gt, preds, config=TapkoEvalConfig(iou_threshold=0.3))
    cats = [e.category for e in res.errors]
    assert "false_positive_scramble" in cats
    assert res.fp == 1


def test_predictions_loader_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tapko_predictions_json(tmp_path / "does_not_exist.json")


def test_evaluate_warns_stderr_when_not_visually_confirmed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    gt = parse_tapko_dict(
        {
            "format_version": "1.0",
            "schema_id": "fightsafe_ai.tapko_annotation",
            "annotation_status": "draft_transcript_derived",
            "annotations": [
                {
                    "video_id": "v1",
                    "source_uri": "x://a.mp4",
                    "start_time": 10.0,
                    "end_time": 11.0,
                    "event_type": "submission_signal.hand_tap",
                    "visibility": "clear",
                    "occlusion_level": "none",
                    "actor_id": "a",
                    "confidence": 1.0,
                    "rater_id": "r1",
                    "requires_audio": False,
                },
            ],
        }
    )
    evaluate_tapko(gt, {}, config=TapkoEvalConfig(iou_threshold=0.5))
    err = capsys.readouterr().err
    assert WARN_DIAGNOSTIC_METRICS in err


def test_evaluate_no_warning_when_visually_confirmed(capsys: pytest.CaptureFixture[str]) -> None:
    gt = parse_tapko_dict(
        {
            "format_version": "1.0",
            "schema_id": "fightsafe_ai.tapko_annotation",
            "annotation_status": "visually_confirmed",
            "annotations": [
                {
                    "video_id": "v1",
                    "source_uri": "x://a.mp4",
                    "start_time": 10.0,
                    "end_time": 11.0,
                    "event_type": "submission_signal.hand_tap",
                    "visibility": "clear",
                    "occlusion_level": "none",
                    "actor_id": "a",
                    "confidence": 1.0,
                    "rater_id": "r1",
                    "requires_audio": False,
                },
            ],
        }
    )
    evaluate_tapko(gt, {}, config=TapkoEvalConfig(iou_threshold=0.5))
    assert WARN_DIAGNOSTIC_METRICS not in capsys.readouterr().err


def test_warn_helper_prints_for_rejected_status(capsys: pytest.CaptureFixture[str]) -> None:
    gt = parse_tapko_dict(
        {
            "format_version": "1.0",
            "schema_id": "fightsafe_ai.tapko_annotation",
            "annotation_status": "rejected",
            "annotations": [],
        }
    )
    warn_if_annotations_not_visually_confirmed(gt)
    assert WARN_DIAGNOSTIC_METRICS in capsys.readouterr().err
