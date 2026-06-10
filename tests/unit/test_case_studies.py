"""Unit tests for :mod:`fightsafe_ai.case_studies` (no network, no long pipeline)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tests.fixtures.mvp_runs import write_minimal_pipeline_run, write_mvp_qa_passing_run

from fightsafe_ai.case_studies import clip
from fightsafe_ai.case_studies.runner import load_case_study_file
from fightsafe_ai.case_studies.tables import build_events_table_tex, write_global_summaries
from fightsafe_ai.exceptions import VideoCutError


pytestmark = pytest.mark.unit


def test_load_case_study_file_minimal(tmp_path: Path) -> None:
    p = tmp_path / "cs.yaml"
    p.write_text(
        "version: '1.0'\n"
        "base_dir: runs/case_studies\ncase_studies:\n"
        "  - case_id: a\n    title: t\n    url: http://x\n"
        "    expected_focus: f\n    output_dir: o\n    notes: n\n"
        "    start_time: null\n    end_time: null\n",
        encoding="utf-8",
    )
    c = load_case_study_file(p)
    assert len(c.case_studies) == 1
    assert c.case_studies[0]["case_id"] == "a"


def test_build_events_table_with_event(tmp_path: Path) -> None:
    write_minimal_pipeline_run(tmp_path, include_mvp_artifacts=True)
    tex = build_events_table_tex(tmp_path)
    assert "tabular" in tex


def test_ffprobe_missing() -> None:
    with pytest.raises(VideoCutError):
        clip.ffprobe_duration_seconds(Path("/nonexistent/abc.mp4"))


def test_write_global_summaries_one_row(tmp_path: Path) -> None:
    w = write_mvp_qa_passing_run(tmp_path / "c1")
    (w / "qa_report.json").write_text(
        '{"metrics": {"total_frames": 1, "frames_with_pose": 1, "pose_coverage_percent": 100.0, "max_risk_score": 0.5}, "passed": true}',
        encoding="utf-8",
    )
    c1, t = write_global_summaries(
        tmp_path / "sum",
        [({"case_id": "A", "expected_focus": "x", "notes": "n"}, w)],
    )
    assert c1.is_file()
    assert t.is_file()
    assert t.read_text(encoding="utf-8") and "A" in t.read_text()


def test_repo_case_studies_yaml_loads() -> None:
    root = Path(__file__).resolve().parents[2]
    p = root / "configs" / "case_studies.yaml"
    if not p.is_file():
        pytest.skip("case_studies.yaml not in tree")
    c = load_case_study_file(p)
    assert len(c.case_studies) == 6
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert str(d.get("version", "")) in ("1.0", "1.1", "1.0.0")


def test_prepare_input_clip_full_copies_mp4(tmp_path: Path) -> None:
    src = tmp_path / "dl.mp4"
    src.write_bytes(b"\x00" * 32)
    outd = tmp_path / "r1"
    p = clip.prepare_input_clip_full(src, outd)
    assert p.name == "input_clip.mp4"
    assert p.is_file() and p.stat().st_size == 32


def test_run_case_studies_from_config_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise orchestration without ``yt-dlp`` or the full pipeline."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "version: '1.0'\nbase_dir: case_st_run\ncase_studies:\n"
        "  - case_id: t1\n    title: t\n    url: http://x\n    expected_focus: f\n"
        "    output_dir: c1\n    notes: n\n    start_time: null\n    end_time: null\n",
        encoding="utf-8",
    )

    def _fake(
        case: object,
        rdir: Path,
        *,
        rules_yaml: object = None,
        fps: int = 10,
    ) -> tuple[Path, bool, None]:
        _ = (rules_yaml, fps)  # match run_one_case() keyword signature
        r = rdir.resolve()
        write_mvp_qa_passing_run(r)
        (r / "qa_report.json").write_text(
            '{"metrics": {"total_frames": 1, "frames_with_pose": 1, "pose_coverage_percent": 1.0, "max_risk_score": 0.1}, "passed": true}',
            encoding="utf-8",
        )
        (r / "events.json").write_text("[]\n", encoding="utf-8")
        (r / "risk_scores.csv").write_text(
            "frame_id,timestamp,risk_score,risk_level,triggered_rules\n0,0,0,LOW,[]\n",
            encoding="utf-8",
        )
        return r, True, None

    from fightsafe_ai.case_studies import runner

    monkeypatch.setattr(runner, "run_one_case", _fake)
    res = runner.run_case_studies_from_config(cfg)
    assert len(res) == 1 and res[0].get("ok") is True
    base = tmp_path / "case_st_run"
    assert (base / "c1" / "events_table.tex").is_file()
    assert (base / "case_study_summary.csv").is_file()
