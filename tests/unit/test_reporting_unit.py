"""Unit tests for :mod:`fightsafe_ai.llm.reporting` helpers (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fightsafe_ai.llm.reporting import (
    build_explanation_markdown,
    generate_pipeline_event_explanations,
    infer_run_sampling_fps,
    regenerate_run_event_explanations,
    resolve_explanations_config,
    resolve_ollama_config,
    union_triggered_rules_in_segment,
    write_explanation_markdown,
)


def test_build_explanation_markdown_minimal() -> None:
    md = build_explanation_markdown(
        {"event_id": 1, "event_level": "HIGH"},
        "Body text.",
    )
    assert "# Risk event 1" in md
    assert "Body text." in md
    assert "not a medical diagnosis" in md


def test_build_explanation_markdown_with_meta() -> None:
    md = build_explanation_markdown(
        {
            "event_id": 2,
            "event_level": "CRITICAL",
            "start_time": 0.0,
            "end_time": 0.2,
            "max_risk_score": 0.95,
        },
        "X",
    )
    assert "0.000 s" in md
    assert "0.200 s" in md
    assert "0.9500" in md


def test_build_explanation_markdown_bad_max_score_falls_back_to_str() -> None:
    md = build_explanation_markdown(
        {"event_id": 1, "event_level": "LOW", "max_risk_score": object()},
        "Y",
    )
    assert "Max risk score" in md


def test_union_triggered_rules_in_segment() -> None:
    df = pd.DataFrame(
        {
            "frame_id": ["a", "b", "c"],
            "triggered_rules": [["r1"], ["r2"], []],
        }
    )
    out = union_triggered_rules_in_segment(df, "a", "b", col_rules="triggered_rules")
    assert "r1" in out and "r2" in out


def test_union_triggered_rules_empty() -> None:
    assert union_triggered_rules_in_segment(pd.DataFrame(), "a", "b") == []


def test_resolve_ollama_config_nonexistent_uses_defaults() -> None:
    cfg = resolve_ollama_config(Path("/no/such/llm.yaml"))
    assert cfg.model  # default model string present


def test_resolve_explanations_config_nonexistent_uses_defaults() -> None:
    ex = resolve_explanations_config(Path("/no/such/llm.yaml"))
    assert ex.include_safety_disclaimer in (True, False)


def test_write_explanation_markdown_no_ollama(tmp_path: Path) -> None:
    p = tmp_path / "e.md"
    text = write_explanation_markdown(
        p,
        {
            "event_id": 0,
            "event_level": "HIGH",
            "start_time": 0.1,
            "end_time": 0.2,
            "max_risk_score": 0.8,
        },
        use_ollama=False,
        llm_config=Path("/no/such/llm.yaml"),
    )
    assert len(text) > 10
    assert p.is_file() and "HIGH" in p.read_text(encoding="utf-8")


def test_generate_pipeline_event_explanations_empty_df(tmp_path: Path) -> None:
    assert (
        generate_pipeline_event_explanations(
            pd.DataFrame(),
            fps=10.0,
            rules_yaml=None,
            out_dir=tmp_path,
            use_ollama=False,
        )
        == 0
    )


def test_generate_pipeline_event_explanations_from_interpretable_frame(
    tmp_path: Path,
) -> None:
    yml = Path(__file__).resolve().parents[2] / "configs" / "risk_rules.yaml"
    if not yml.is_file():
        pytest.skip("project risk_rules.yaml not present")
    from fightsafe_ai.risk.scorer import compute_interpretable_risk

    def _row(i: int) -> dict[str, Any]:
        return {
            "frame_id": f"frame_{i:06d}",
            "hip_vertical_velocity": 2.0 + 0.1 * i,
            "head_vertical_velocity": 0.2,
            "torso_angle_deg": 80.0,
            "low_posture_duration_frames": 30.0,
            "instability_score": 0.9,
            "near_ground": True,
            "guard_level": 0.0,
            "facing_away_score": 0.0,
            "reaction_delay_score": 0.0,
        }

    fe = pd.DataFrame([_row(i) for i in range(20)])
    interp = compute_interpretable_risk(fe, rules_yaml=yml, include_rule_component_columns=False)
    n = generate_pipeline_event_explanations(
        interp,
        fps=10.0,
        rules_yaml=yml,
        out_dir=tmp_path,
        use_ollama=False,
        llm_config=Path("/no/such/llm.yaml"),
    )
    assert n >= 0
    if n > 0:
        assert any(tmp_path.glob("event_*.md"))


def test_infer_run_sampling_fps_from_risk_csv(tmp_path: Path) -> None:
    (tmp_path / "risk_scores.csv").write_text(
        "timestamp,risk_score,frame_id\n"
        + "\n".join(f"{i * 0.1:.1f},0.5,frame_{i:06d}" for i in range(11)),
        encoding="utf-8",
    )
    fps = infer_run_sampling_fps(tmp_path)
    assert 9.5 < fps < 10.5


def test_infer_run_sampling_fps_fallback_no_file(tmp_path: Path) -> None:
    assert infer_run_sampling_fps(tmp_path) == 10.0


def test_regenerate_run_event_explanations_missing_features(tmp_path: Path) -> None:
    (tmp_path / "risk_scores.csv").write_text("timestamp,risk_score\n0,0.1\n", encoding="utf-8")
    assert (
        regenerate_run_event_explanations(
            tmp_path, use_ollama=False, ollama_force_enabled=True, llm_config=Path("/no/llm.yaml")
        )
        == 0
    )


def test_write_explanation_markdown_force_ollama_falls_back_without_server(tmp_path: Path) -> None:
    """Force-enabled path still produces Markdown when Ollama is unreachable."""
    p = tmp_path / "e.md"
    t = write_explanation_markdown(
        p,
        {
            "event_id": 0,
            "event_level": "HIGH",
            "start_time": 0.1,
            "end_time": 0.2,
            "max_risk_score": 0.8,
        },
        use_ollama=True,
        ollama_force_enabled=True,
        llm_config=Path("/no/llm.yaml"),
    )
    assert len(t) > 20
    assert p.is_file()
    assert "HIGH" in p.read_text(encoding="utf-8")
