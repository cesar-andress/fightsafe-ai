"""Edge-case tests for :mod:`fightsafe_ai.reports.summary` helpers."""

from __future__ import annotations

from pathlib import Path

from fightsafe_ai.reports.summary import infer_input_video_path, load_events_list, load_qa_dict


def test_load_qa_dict_invalid_json_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "qa.json"
    p.write_text("{", encoding="utf-8")
    assert load_qa_dict(p) is None


def test_load_events_list_non_list_json_dict_wrap(tmp_path: Path) -> None:
    p = tmp_path / "e.json"
    p.write_text('{"event_id": 1, "event_level": "LOW"}', encoding="utf-8")
    out = load_events_list(p)
    assert len(out) == 1 and out[0]["event_id"] == 1


def test_infer_input_video_path_backtick_mp4(tmp_path: Path) -> None:
    md = tmp_path / "report.md"
    md.write_text("Source `demo_clip.mp4` for review.", encoding="utf-8")
    assert infer_input_video_path(md) == "demo_clip.mp4"


def test_infer_input_video_path_source_line(tmp_path: Path) -> None:
    md = tmp_path / "report2.md"
    md.write_text("The input video `other.mov` is used.", encoding="utf-8")
    assert infer_input_video_path(md) == "other.mov"
