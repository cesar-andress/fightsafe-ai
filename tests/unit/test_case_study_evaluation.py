"""Unit tests for illustrative case-study batch evaluation vs annotation JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fightsafe_ai.evaluation.case_study_evaluation import (
    CaseStudyEvalRow,
    evaluate_one_case,
    run_case_study_batch_evaluation,
    write_case_study_evaluation_csv,
    write_case_study_evaluation_tex,
)


pytestmark = pytest.mark.unit


def _write_annotation(
    path: Path, *, events: list[dict[str, Any]], case_id: str = "case_a_knockdown"
) -> None:
    doc = {
        "format_version": "1.0",
        "case_id": case_id,
        "source_reference": "https://example.com/watch",
        "video": "https://example.com/watch",
        "time_unit": "seconds",
        "clip_start_time": None,
        "clip_end_time": None,
        "events": events,
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _write_events_json(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(events), encoding="utf-8")


def test_empty_annotation_no_fabricated_metrics(tmp_path: Path) -> None:
    """Empty ``events`` → annotation_pending; precision/recall/F1 omitted (not 0 as if computed)."""
    ann = tmp_path / "annotations"
    runs = tmp_path / "runs"
    ann.mkdir()
    runs.mkdir()
    _write_annotation(ann / "case_a_knockdown.json", events=[])
    (runs / "cs_knockdown_001").mkdir()
    _write_events_json(
        runs / "cs_knockdown_001" / "events.json",
        [{"start_time": 0.0, "end_time": 1.0, "event_level": "HIGH"}],
    )
    r = evaluate_one_case(
        "case_a_knockdown",
        runs_dir=runs,
        annotations_dir=ann,
        iou_threshold=0.1,
        tolerance_seconds=0.0,
    )
    assert r.status == "annotation_pending"
    assert r.annotated_events == 0
    assert r.predicted_events == 1
    assert r.precision is None
    assert r.recall is None
    assert r.f1 is None
    assert r.true_positives is None


def test_perfect_match_one_event(tmp_path: Path) -> None:
    ann = tmp_path / "annotations"
    runs = tmp_path / "runs"
    ann.mkdir()
    runs.mkdir()
    _write_annotation(
        ann / "case_a_knockdown.json",
        events=[
            {
                "event_id": "e1",
                "start_time": 1.0,
                "end_time": 3.0,
                "event_type": "INSTABILITY",
                "confidence": 1.0,
                "notes": "unit test",
            }
        ],
    )
    (runs / "cs_knockdown_001").mkdir()
    _write_events_json(
        runs / "cs_knockdown_001" / "events.json",
        [{"start_time": 1.0, "end_time": 3.0, "event_type": "INSTABILITY"}],
    )
    r = evaluate_one_case(
        "case_a_knockdown",
        runs_dir=runs,
        annotations_dir=ann,
        iou_threshold=0.1,
        tolerance_seconds=0.0,
    )
    assert r.status == "ok"
    assert r.true_positives == 1
    assert r.false_positives == 0
    assert r.false_negatives == 0
    assert r.precision == pytest.approx(1.0)
    assert r.recall == pytest.approx(1.0)
    assert r.f1 == pytest.approx(1.0)


def test_one_false_positive(tmp_path: Path) -> None:
    ann = tmp_path / "annotations"
    runs = tmp_path / "runs"
    ann.mkdir()
    runs.mkdir()
    _write_annotation(
        ann / "case_a_knockdown.json",
        events=[
            {
                "start_time": 10.0,
                "end_time": 12.0,
                "event_type": "FALL",
                "confidence": 0.9,
                "notes": None,
            }
        ],
    )
    (runs / "cs_knockdown_001").mkdir()
    _write_events_json(
        runs / "cs_knockdown_001" / "events.json",
        [
            {"start_time": 10.0, "end_time": 12.0, "event_type": "FALL"},
            {"start_time": 50.0, "end_time": 52.0, "event_type": "HIGH"},
        ],
    )
    r = evaluate_one_case(
        "case_a_knockdown",
        runs_dir=runs,
        annotations_dir=ann,
        iou_threshold=0.1,
        tolerance_seconds=0.0,
    )
    assert r.status == "ok"
    assert r.true_positives == 1
    assert r.false_positives == 1
    assert r.false_negatives == 0
    assert r.precision == pytest.approx(1.0 / 2.0)
    assert r.recall == pytest.approx(1.0)


def test_one_false_negative(tmp_path: Path) -> None:
    ann = tmp_path / "annotations"
    runs = tmp_path / "runs"
    ann.mkdir()
    runs.mkdir()
    _write_annotation(
        ann / "case_a_knockdown.json",
        events=[
            {
                "start_time": 0.0,
                "end_time": 2.0,
                "event_type": "KO",
                "confidence": 1.0,
                "notes": None,
            }
        ],
    )
    (runs / "cs_knockdown_001").mkdir()
    _write_events_json(runs / "cs_knockdown_001" / "events.json", [])
    r = evaluate_one_case(
        "case_a_knockdown",
        runs_dir=runs,
        annotations_dir=ann,
        iou_threshold=0.1,
        tolerance_seconds=0.0,
    )
    assert r.status == "ok"
    assert r.true_positives == 0
    assert r.false_positives == 0
    assert r.false_negatives == 1
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.f1 == 0.0


def test_onset_delay_computation(tmp_path: Path) -> None:
    ann = tmp_path / "annotations"
    runs = tmp_path / "runs"
    ann.mkdir()
    runs.mkdir()
    _write_annotation(
        ann / "case_a_knockdown.json",
        events=[{"start_time": 0.0, "end_time": 2.0, "event_type": "INSTABILITY"}],
    )
    (runs / "cs_knockdown_001").mkdir()
    _write_events_json(
        runs / "cs_knockdown_001" / "events.json",
        [{"start_time": 0.5, "end_time": 2.5, "event_level": "HIGH"}],
    )
    r = evaluate_one_case(
        "case_a_knockdown",
        runs_dir=runs,
        annotations_dir=ann,
        iou_threshold=0.1,
        tolerance_seconds=0.0,
    )
    assert r.status == "ok"
    assert r.true_positives == 1
    assert r.mean_onset_delay_seconds == pytest.approx(0.5)
    assert r.mean_absolute_onset_delay_seconds == pytest.approx(0.5)


def test_batch_evaluation_multiple_cases(tmp_path: Path) -> None:
    ann = tmp_path / "annotations"
    runs = tmp_path / "runs"
    ann.mkdir()
    runs.mkdir()
    _write_annotation(
        ann / "case_a_knockdown.json",
        events=[{"start_time": 1.0, "end_time": 2.0, "event_type": "FALL"}],
        case_id="case_a_knockdown",
    )
    _write_annotation(ann / "case_b_tap.json", events=[], case_id="case_b_tap")
    (runs / "cs_knockdown_001").mkdir()
    (runs / "cs_surrender_001").mkdir()
    _write_events_json(
        runs / "cs_knockdown_001" / "events.json",
        [{"start_time": 1.0, "end_time": 2.0, "event_type": "FALL"}],
    )
    _write_events_json(
        runs / "cs_surrender_001" / "events.json", [{"start_time": 0.0, "end_time": 1.0}]
    )

    rows = run_case_study_batch_evaluation(
        runs_dir=runs,
        annotations_dir=ann,
        keys=("case_a_knockdown", "case_b_tap"),
        iou_threshold=0.1,
        tolerance_seconds=0.0,
    )
    assert len(rows) == 2
    assert rows[0].case_id == "case_a_knockdown"
    assert rows[0].status == "ok"
    assert rows[0].f1 == pytest.approx(1.0)
    assert rows[1].case_id == "case_b_tap"
    assert rows[1].status == "annotation_pending"


def test_write_case_study_evaluation_tex_smoke(tmp_path: Path) -> None:
    """Covers TeX writer (escapes, numeric cells, parent dir creation)."""
    rows = [
        CaseStudyEvalRow(
            case_id="case_a_knockdown",
            status="ok",
            predicted_events=1,
            annotated_events=1,
            true_positives=1,
            false_positives=0,
            false_negatives=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            mean_onset_delay_seconds=0.25,
            mean_absolute_onset_delay_seconds=0.25,
        ),
        CaseStudyEvalRow(
            case_id="case_x_100%_special",
            status="annotation_pending",
            predicted_events=3,
            annotated_events=0,
            true_positives=None,
            false_positives=None,
            false_negatives=None,
            precision=None,
            recall=None,
            f1=None,
            mean_onset_delay_seconds=None,
            mean_absolute_onset_delay_seconds=None,
        ),
    ]
    out = tmp_path / "nested" / "eval.tex"
    write_case_study_evaluation_tex(out, rows)
    text = out.read_text(encoding="utf-8")
    assert "\\label{tab:case-study-annotation-eval}" in text
    assert "case\\_x\\_100\\%\\_special" in text
    assert "annotation\\_pending" in text


def test_csv_includes_evaluation_status(tmp_path: Path) -> None:
    ann = tmp_path / "annotations"
    runs = tmp_path / "runs"
    ann.mkdir()
    runs.mkdir()
    _write_annotation(ann / "case_a_knockdown.json", events=[])
    (runs / "cs_knockdown_001").mkdir()
    _write_events_json(runs / "cs_knockdown_001" / "events.json", [])
    rows: list[CaseStudyEvalRow] = [
        evaluate_one_case(
            "case_a_knockdown",
            runs_dir=runs,
            annotations_dir=ann,
            iou_threshold=0.1,
            tolerance_seconds=0.0,
        )
    ]
    out = tmp_path / "out.csv"
    write_case_study_evaluation_csv(out, rows)
    text = out.read_text(encoding="utf-8")
    assert "evaluation_status" in text
    assert "annotation_pending" in text
