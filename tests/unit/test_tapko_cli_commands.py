"""CLI TapKO commands (export, validate, evaluate wiring)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fightsafe_ai.cli import app


pytestmark = pytest.mark.unit

runner = CliRunner()


def test_tapko_export_examples_writes_valid_json(tmp_path: Path) -> None:
    from fightsafe_ai.annotation.tapko_schema import parse_tapko_json

    r = runner.invoke(
        app,
        ["tapko-export-examples", "--output-dir", str(tmp_path)],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    minimal = tmp_path / "tapko_example_minimal.json"
    full = tmp_path / "tapko_example_full.json"
    assert minimal.is_file() and full.is_file()
    parse_tapko_json(minimal.read_text(encoding="utf-8"))
    parse_tapko_json(full.read_text(encoding="utf-8"))


def test_tapko_validate_annotations_ok(tmp_path: Path) -> None:
    from fightsafe_ai.annotation.tapko_schema import EXAMPLE_DOCUMENT_MINIMAL

    p = tmp_path / "ann.json"
    p.write_text(json.dumps(EXAMPLE_DOCUMENT_MINIMAL), encoding="utf-8")
    r = runner.invoke(app, ["tapko-validate-annotations", "--annotations", str(p)])
    assert r.exit_code == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


def test_tapko_validate_annotations_legacy_events_bundle(tmp_path: Path) -> None:
    """Pilot JSON: metadata at root and intervals under ``events`` (not ``annotations``)."""

    legacy = {
        "dataset_version": "tapko_v0.1",
        "video_id": "clip_a",
        "source_uri": "https://example.org/v.mp4",
        "fps": 30,
        "annotation_status": "draft_transcript_derived",
        "notes": "Bundle notes.",
        "events": [
            {
                "event_id": "e001",
                "event_type": "submission_signal.hand_tap",
                "start_time": 1.0,
                "end_time": 2.0,
                "visibility": "unknown",
                "occlusion_level": "unknown",
                "actor_id": "unknown",
                "target_id": "unknown",
                "confidence": 0.5,
                "requires_audio": False,
                "notes": "Per-interval note.",
            },
        ],
    }
    p = tmp_path / "legacy_bundle.json"
    p.write_text(json.dumps(legacy), encoding="utf-8")
    r = runner.invoke(app, ["tapko-validate-annotations", "--annotations", str(p)])
    assert r.exit_code == 0, r.stdout + r.stderr
    assert "1 annotation interval" in r.stdout


def test_tapko_evaluate_runs_on_fixture_pair(tmp_path: Path) -> None:
    """Smoke: evaluator writes CSV/TeX/MD (same fixtures as test_tapko_evaluator)."""
    from fightsafe_ai.annotation.tapko_schema import parse_tapko_dict

    gt = parse_tapko_dict(
        {
            "format_version": "1.0",
            "schema_id": "fightsafe_ai.tapko_annotation",
            "annotation_status": "visually_confirmed",
            "annotations": [
                {
                    "video_id": "v1",
                    "source_uri": "file:///x.mp4",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "event_type": "submission_signal.hand_tap",
                    "visibility": "clear",
                    "occlusion_level": "none",
                    "actor_id": "a",
                    "confidence": 0.9,
                    "rater_id": "r",
                    "requires_audio": False,
                }
            ],
        }
    )
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(gt.model_dump_json(indent=2), encoding="utf-8")
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(
        json.dumps(
            [
                {
                    "video_id": "v1",
                    "start_time": 1.1,
                    "end_time": 2.1,
                    "event_type": "submission_signal.hand_tap",
                }
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "eval_out"
    r = runner.invoke(
        app,
        [
            "tapko-evaluate",
            "--annotations",
            str(gt_path),
            "--predictions",
            str(pred_path),
            "--output-dir",
            str(out_dir),
            "--iou-threshold",
            "0.3",
        ],
    )
    assert r.exit_code == 0, r.stdout + r.stderr
    assert (out_dir / "tapko_results.csv").is_file()
    assert (out_dir / "tapko_results.tex").is_file()
    assert (out_dir / "tapko_error_analysis.md").is_file()


def test_landmark_map_to_coco17_xy_roundtrip() -> None:
    from fightsafe_ai.tapko.coco_stack import COCO17_POSE_NAMES, landmark_map_to_coco17_xy

    lm = {COCO17_POSE_NAMES[0]: (0.1, 0.2)}
    xy = landmark_map_to_coco17_xy(lm)
    assert xy.shape == (17, 2)
    assert xy[0, 0] == 0.1 and xy[0, 1] == 0.2
