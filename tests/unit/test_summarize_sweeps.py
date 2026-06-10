"""Tests for sweep summarizer across BoxingVI batch CSV exports."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fightsafe_ai.evaluation import summarize_sweeps as ss
from fightsafe_ai.evaluation.boxingvi_batch_eval import (
    VideoBatchRow,
    _write_aggregate_table_csv,
)


pytestmark = pytest.mark.unit


def test_infer_strike_percentile_from_folder_name() -> None:
    assert ss.infer_strike_percentile(Path("outputs/evaluation/boxingvi_batch_p85")) == 85
    assert ss.infer_strike_percentile(Path("runs/sweep_subdir/boxingvi_batch_p97")) == 97


def test_infer_strike_percentile_missing_returns_none(tmp_path: Path) -> None:
    assert ss.infer_strike_percentile(tmp_path / "no_suffix_here") is None


def test_parse_with_aggregate_rows(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "boxingvi_batch_p90"
    sweep_dir.mkdir()
    rows = [
        VideoBatchRow(
            video_id="V1",
            status="OK",
            tp=2,
            fp=1,
            fn=1,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            mean_latency_seconds=0.1,
            error=None,
        )
    ]
    summary = {
        "micro_tp": 2,
        "micro_fp": 1,
        "micro_fn": 1,
        "micro_precision": 0.666667,
        "micro_recall": 0.666667,
        "micro_f1": 0.666667,
        "macro_f1": 0.5,
        "macro_precision": 0.5,
        "macro_recall": 0.5,
        "mean_latency_mean": 0.1,
    }
    csv_path = sweep_dir / "boxingvi_results_all.csv"
    _write_aggregate_table_csv(csv_path, rows, summary)

    m = ss.parse_boxingvi_results_all_csv(csv_path, input_dir=sweep_dir)
    assert m.strike_percentile == 90
    assert m.tp == 2 and m.fp == 1 and m.fn == 1
    assert abs(m.micro_f1 - 0.666667) < 1e-3
    assert m.macro_f1 == 0.5
    assert abs(m.mean_latency - 0.1) < 1e-6


def test_parse_without_aggregate_derives_from_per_video(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "boxingvi_batch_p80"
    sweep_dir.mkdir()
    csv_path = sweep_dir / "boxingvi_results_all.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
                "F1",
                "mean_latency",
                "status",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "video_id": "V1",
                "TP": "1",
                "FP": "1",
                "FN": "1",
                "precision": "0.333333",
                "recall": "0.333333",
                "F1": "0.333333",
                "mean_latency": "0.05",
                "status": "OK",
            }
        )
        w.writerow(
            {
                "video_id": "V2",
                "TP": "1",
                "FP": "0",
                "FN": "0",
                "precision": "1",
                "recall": "1",
                "F1": "1",
                "mean_latency": "0.06",
                "status": "OK",
            }
        )

    m = ss.parse_boxingvi_results_all_csv(csv_path, input_dir=sweep_dir)
    assert m.strike_percentile == 80
    assert m.tp == 2 and m.fp == 1 and m.fn == 1


def test_select_recommended_prefers_macro_f1_then_recall() -> None:
    low = ss.SweepMetrics(
        strike_percentile=85,
        input_dir=Path("/a"),
        tp=1,
        fp=1,
        fn=1,
        micro_precision=0.5,
        micro_recall=0.5,
        micro_f1=0.5,
        macro_precision=0.5,
        macro_recall=0.5,
        macro_f1=0.8,
        mean_latency=0.1,
    )
    tie_same_f1 = ss.SweepMetrics(
        strike_percentile=90,
        input_dir=Path("/b"),
        tp=1,
        fp=1,
        fn=1,
        micro_precision=0.5,
        micro_recall=0.5,
        micro_f1=0.5,
        macro_precision=0.6,
        macro_recall=0.75,
        macro_f1=0.8,
        mean_latency=0.1,
    )
    winner = ss.select_recommended([low, tie_same_f1])
    assert winner.strike_percentile == 90


def test_run_summarize_end_to_end(tmp_path: Path) -> None:
    """Multiple sweep dirs → summary files on disk."""
    dirs = []
    for pct, mf1, mrec in ((85, 0.7, 0.6), (90, 0.85, 0.7), (95, 0.85, 0.8)):
        d = tmp_path / f"boxingvi_batch_p{pct}"
        d.mkdir()
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
            "macro_f1": mf1,
            "macro_precision": mf1,
            "macro_recall": mrec,
            "mean_latency_mean": 0.05,
        }
        _write_aggregate_table_csv(d / "boxingvi_results_all.csv", rows, summary)
        dirs.append(d)

    out = tmp_path / "sweeps"
    ss.run_summarize(dirs, out, print_report=False)

    assert (out / "sweep_summary.csv").is_file()
    assert (out / "sweep_summary.tex").is_file()
    assert (out / "sweep_summary.md").is_file()
    text = (out / "sweep_summary.md").read_text(encoding="utf-8")
    assert "95" in text
    assert "Recommended" in text


def test_main_returns_error_when_csv_missing(tmp_path: Path) -> None:
    d = tmp_path / "boxingvi_batch_p50"
    d.mkdir(parents=True)
    rc = ss.main(["--input-dirs", str(d), "--output-dir", str(tmp_path / "out"), "-q"])
    assert rc == 1


def test_main_returns_zero_when_ok(tmp_path: Path) -> None:
    d = tmp_path / "boxingvi_batch_p88"
    d.mkdir()
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
    _write_aggregate_table_csv(d / "boxingvi_results_all.csv", rows, summary)
    out = tmp_path / "sweep_out"
    rc = ss.main(["--input-dirs", str(d), "--output-dir", str(out), "-q"])
    assert rc == 0
    assert (out / "sweep_summary.csv").is_file()


def test_format_console_report_contains_bests(tmp_path: Path) -> None:
    s85 = ss.SweepMetrics(
        strike_percentile=85,
        input_dir=tmp_path / "p85",
        tp=1,
        fp=0,
        fn=0,
        micro_precision=1.0,
        micro_recall=1.0,
        micro_f1=0.9,
        macro_precision=1.0,
        macro_recall=0.5,
        macro_f1=0.7,
        mean_latency=0.1,
    )
    s90 = ss.SweepMetrics(
        strike_percentile=90,
        input_dir=tmp_path / "p90",
        tp=1,
        fp=0,
        fn=0,
        micro_precision=1.0,
        micro_recall=1.0,
        micro_f1=0.95,
        macro_precision=1.0,
        macro_recall=0.9,
        macro_f1=0.85,
        mean_latency=0.1,
    )
    rec = ss.select_recommended([s85, s90])
    txt = ss.format_console_report([s85, s90], recommended=rec)
    assert "Best micro F1" in txt and "0.95" in txt
    assert "Recommended" in txt and "90" in txt
