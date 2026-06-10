"""Tests for TapKO run summary Markdown utility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fightsafe_ai.evaluation.tapko_summarize_run import (
    DEFAULT_PILOT_INTERPRETATION,
    build_tapko_run_summary_md,
    parse_error_examples_table,
    summarize_tapko_run_to_markdown,
    write_tapko_run_summary,
)


def test_parse_error_examples_skips_header_and_limits(tmp_path: Path) -> None:
    md = tmp_path / "tapko_error_analysis.md"
    md.write_text(
        "\n".join(
            [
                "# TapKO error analysis",
                "",
                "## Examples (up to 40 rows)",
                "",
                "| video_id | category | detail | ref | pred | ref interval | pred interval | IoU |",
                "|----------|----------|--------|-----|------|--------------|---------------|-----|",
                "| v1 | false_positive | first | r0 | p0 | 0.0–1.0 | 1.0–2.0 | 0.0 |",
                "| v1 | missed_event | second | r1 | p1 | 5.0–6.0 | 6.0–7.0 | 0.0 |",
                "| v1 | late_detection | third | x | y | 1–2 | 3–4 | 0.5 |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rows = parse_error_examples_table(md, limit=2)
    assert len(rows) == 2
    assert rows[0][0] == "v1" and rows[0][1] == "false_positive"
    assert rows[1][1] == "missed_event"


def test_parse_row_with_pipe_in_detail(tmp_path: Path) -> None:
    md = tmp_path / "e.md"
    md.write_text(
        "## Examples\n\n"
        "| video_id | category | detail | ref | pred | ref interval | pred interval | IoU |\n"
        "|----------|----------|--------|-----|------|--------------|---------------|-----|\n"
        "| v1 | cat | part1 | part2 | R | P | 1–2 | 3–4 | 0.5 |\n",
        encoding="utf-8",
    )
    rows = parse_error_examples_table(md, limit=5)
    assert len(rows) == 1
    assert "part1" in rows[0][2] and "part2" in rows[0][2]


def test_summarize_end_to_end(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    det_dir = tmp_path / "det"
    eval_dir.mkdir()
    det_dir.mkdir()

    (eval_dir / "tapko_results.csv").write_text(
        "scope,label,tp,fp,fn,precision,recall,f1,f2,"
        "mean_onset_latency_sec,mean_abs_onset_latency_sec,false_positives_per_minute,"
        "total_video_duration_min\n"
        "micro,,1,2,3,0.1,0.2,0.3,0.4,0.5,0.6,7.8,11.25\n",
        encoding="utf-8",
    )
    (eval_dir / "tapko_error_analysis.md").write_text(
        "\n".join(
            [
                "## Examples",
                "",
                "| video_id | category | detail | ref | pred | ref interval | pred interval | IoU |",
                "|----------|----------|--------|-----|------|--------------|---------------|-----|",
                "| vid_a | false_positive | fp note | refx | lab | 0–1 | 1–2 | 0.00 |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (det_dir / "tapko_manifest.json").write_text(
        json.dumps(
            {
                "video_id": "jedi_submissions",
                "fps": 30,
                "n_frames": 33474,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (det_dir / "tapko_predictions.json").write_text(
        json.dumps([{"video_id": "jedi_submissions", "start_time": 0.0, "end_time": 1.0}] * 337),
        encoding="utf-8",
    )

    md_text = summarize_tapko_run_to_markdown(eval_dir, det_dir)
    assert "jedi_submissions" in md_text
    assert "337" in md_text
    assert "TP | 1 |" in md_text
    assert "False positives / minute | 7.8000 |" in md_text or "7.8000" in md_text
    assert DEFAULT_PILOT_INTERPRETATION.split()[0] in md_text
    assert "false_positive" in md_text


def test_write_summary_writes_file(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    det_dir = tmp_path / "det"
    eval_dir.mkdir()
    det_dir.mkdir()
    (eval_dir / "tapko_results.csv").write_text(
        "scope,label,tp,fp,fn,precision,recall,f1,f2,"
        "mean_onset_latency_sec,mean_abs_onset_latency_sec,false_positives_per_minute,"
        "total_video_duration_min\n"
        "micro,,0,0,0,0,0,0,0,0,0,0,1\n",
        encoding="utf-8",
    )
    (eval_dir / "tapko_error_analysis.md").write_text(
        "# TapKO error analysis\n\n## Examples\n\n"
        "| video_id | category | detail | ref | pred | ref interval | pred interval | IoU |\n"
        "|----------|----------|--------|-----|------|--------------|---------------|-----|\n",
        encoding="utf-8",
    )
    (det_dir / "tapko_manifest.json").write_text(
        json.dumps({"video_id": "v", "fps": 30, "n_frames": 300}),
        encoding="utf-8",
    )
    (det_dir / "tapko_predictions.json").write_text("[]", encoding="utf-8")
    out = tmp_path / "out.md"
    path = write_tapko_run_summary(eval_dir, det_dir, out, interpretation="Custom closing.")
    assert path.read_text(encoding="utf-8").endswith("Custom closing.\n")


def test_build_summary_custom_interpretation() -> None:
    md = build_tapko_run_summary_md(
        video_id="x",
        duration_min=10.0,
        n_candidates=5,
        micro={
            "tp": "1",
            "fp": "2",
            "fn": "3",
            "precision": "0.25",
            "recall": "0.33",
            "f1": "0.28",
            "f2": "0.30",
            "false_positives_per_minute": "1.5",
        },
        error_example_rows=[],
        interpretation="Only custom.",
    )
    assert "Only custom." in md
    assert "`x`" in md


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        summarize_tapko_run_to_markdown(tmp_path / "no_eval", tmp_path / "no_det")
