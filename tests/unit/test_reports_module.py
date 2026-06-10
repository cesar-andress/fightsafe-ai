"""Tests for ``fightsafe_ai.reports`` (synthetic run directories)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tests.fixtures.mvp_runs import write_minimal_pipeline_run

from fightsafe_ai.reports import (
    build_summary_dict,
    generate_html_report,
    generate_markdown_report,
    generate_summary_json,
    missing_report_artifacts,
    report_prereq_error_message,
    write_all_default_reports,
)


def test_build_summary_qa_unknown_without_file(tmp_path: Path) -> None:
    r = tmp_path / "no_qa"
    r.mkdir()
    pd.DataFrame(
        {"frame_id": ["a"], "timestamp": [0.0], "risk_score": [0.5], "risk_level": ["LOW"]}
    ).to_csv(r / "risk_scores.csv", index=False)
    (r / "events.json").write_text("[]", encoding="utf-8")
    d = build_summary_dict(r)
    assert d["qa_status"] == "unknown"


def test_build_summary_dict(tmp_path: Path) -> None:
    r = tmp_path / "mp4_run"
    write_minimal_pipeline_run(r, include_mvp_artifacts=True)
    d = build_summary_dict(r)
    assert d["clip_id"] == "mp4_run"
    assert d["total_frames"] == 3
    assert d["number_of_events"] == 1
    assert d["highest_event_level"] == "HIGH"
    assert d["qa_status"] == "pass"
    assert d["max_risk_score"] is not None
    assert abs(d["max_risk_score"] - 0.9) < 1e-5


def test_generate_summary_json(tmp_path: Path) -> None:
    r = tmp_path / "r1"
    write_minimal_pipeline_run(r, include_mvp_artifacts=True)
    p = r / "summary_out.json"
    out = generate_summary_json(r, p)
    assert out == p
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["number_of_events"] == 1


def test_generate_markdown_report_sections(tmp_path: Path) -> None:
    r = tmp_path / "r2"
    write_minimal_pipeline_run(r, include_mvp_artifacts=True)
    p = r / "extra_report.md"
    out = generate_markdown_report(r, p)
    assert out.is_file()
    t = p.read_text(encoding="utf-8")
    for section in (
        "Run overview",
        "Input video",
        "Pipeline outputs",
        "Risk summary",
        "Detected events",
        "AI Explanation (Optional)",
        "Highest risk moment",
        "QA summary",
        "Human review recommendation",
        "Limitations",
        "Safety disclaimer",
    ):
        assert "##" in t and section in t
    assert "LLM-generated explanations are for interpretability only" in t


def test_missing_report_artifacts_ok(tmp_path: Path) -> None:
    r = tmp_path / "ok"
    write_minimal_pipeline_run(r, include_mvp_artifacts=True)
    assert missing_report_artifacts(r) == []


def test_missing_report_artifacts_no_dir(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "run"
    out = missing_report_artifacts(missing)
    assert len(out) == 1
    assert out[0] == missing.expanduser().resolve()


def test_missing_report_artifacts_missing_risk(tmp_path: Path) -> None:
    r = tmp_path / "partial"
    r.mkdir()
    (r / "events.json").write_text("[]", encoding="utf-8")
    m = missing_report_artifacts(r)
    assert len(m) == 1
    assert m[0].name == "risk_scores.csv"


def test_report_prereq_error_message_mentions_qa(tmp_path: Path) -> None:
    r = tmp_path / "bad"
    r.mkdir()
    msg = report_prereq_error_message(r, [r / "risk_scores.csv"])
    assert "fightsafe qa" in msg
    assert "risk_scores.csv" in msg


def test_write_all_default_reports(tmp_path: Path) -> None:
    r = tmp_path / "bundle"
    write_minimal_pipeline_run(r, include_mvp_artifacts=True)
    p1, p2, p3 = write_all_default_reports(r)
    assert p1 == r / "report.md" and p1.is_file()
    assert p2 == r / "report.html" and p2.is_file()
    assert p3 == r / "summary.json" and p3.is_file()


def test_ai_section_prefers_explanation_file_and_labels_llm(tmp_path: Path) -> None:
    """A non-template ``## Explanation`` body in ``explanations/event_0000.md`` is labeled as model-style."""
    r = tmp_path / "run_ex"
    write_minimal_pipeline_run(r, include_mvp_artifacts=True)
    expl = r / "explanations"
    expl.mkdir(parents=True)
    body = (
        "The segment shows rapid movement consistent with a knockdown response; not a medical read."
    )
    (expl / "event_0000.md").write_text(
        f"# Event stub\n\n---\n\n## Explanation\n\n{body}\n\n---\n\n*footer*\n",
        encoding="utf-8",
    )
    t = generate_markdown_report(r, r / "bundle.md").read_text(encoding="utf-8")
    assert "AI Explanation (Optional)" in t
    assert "Model" in t
    assert "rapid movement" in t

    p_html = r / "out.html"
    generate_html_report(r, p_html)
    h = p_html.read_text(encoding="utf-8")
    assert "Model narrative" in h
    assert "ai-explanations" in h


def test_generate_html_no_js(tmp_path: Path) -> None:
    r = tmp_path / "r3"
    write_minimal_pipeline_run(r, include_mvp_artifacts=True)
    p = r / "view.html"
    generate_html_report(r, p)
    t = p.read_text(encoding="utf-8")
    assert "<html" in t
    assert "output_overlay" in t
    assert "<script" not in t  # no external/inline script per spec
    assert "table" in t
    assert "decision-support" in t and "medical diagnosis" in t
    assert "Risk statistics" in t
    assert "Detected events" in t
    assert "AI Explanation (Optional)" in t
    assert "LLM-generated explanations are for interpretability only" in t
    assert "Highest risk moment" in t
    assert "QA results" in t
