"""Placeholders for paper tables when run exports are absent."""

from __future__ import annotations

from pathlib import Path

from fightsafe_ai.reports.paper_build import (
    ensure_boxingvi_paper_table_fragments,
    write_ablation_table_placeholder,
    write_baseline_comparison_placeholder,
    write_boxingvi_results_all_placeholder,
    write_quantitative_table_placeholder,
)


def test_placeholders_are_tex_and_contain_todo(tmp_path: Path) -> None:
    a = write_ablation_table_placeholder(tmp_path)
    q = write_quantitative_table_placeholder(tmp_path)
    assert "% [TODO]" in a.read_text(encoding="utf-8")
    assert "\\label{tab:risk-ablation-selected}" in a.read_text(encoding="utf-8")
    assert "% [TODO]" in q.read_text(encoding="utf-8")
    assert "\\label{tab:quantitative-ablation-summary}" in q.read_text(encoding="utf-8")


def test_boxingvi_placeholders_and_ensure(tmp_path: Path) -> None:
    b = write_boxingvi_results_all_placeholder(tmp_path)
    c = write_baseline_comparison_placeholder(tmp_path)
    assert "\\label{tab:boxingvi-batch-eval}" in b.read_text(encoding="utf-8")
    assert "\\label{tab:boxingvi-baselines}" in c.read_text(encoding="utf-8")
    (tmp_path / "tables" / "boxingvi_results_all.tex").unlink()
    w = ensure_boxingvi_paper_table_fragments(tmp_path)
    assert (tmp_path / "tables" / "boxingvi_results_all.tex").is_file()
    assert len(w) == 1
