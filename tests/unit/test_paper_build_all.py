"""Unit tests for paper build orchestration (no full LaTeX / dataset)."""

from __future__ import annotations

from pathlib import Path

from fightsafe_ai.paper.build_all import (
    _sanitize_boxingvi_results_all_tex,
    collect_transitive_tex_inputs,
)


def test_transitive_inputs_finds_ablation_table(tmp_path: Path) -> None:
    paper = tmp_path / "paper"
    (paper / "tables").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\\input{ablation_results}\n",
        encoding="utf-8",
    )
    (paper / "ablation_results.tex").write_text(
        "\\input{tables/ablation_selected_results}\n",
        encoding="utf-8",
    )
    (paper / "tables" / "ablation_selected_results.tex").write_text(
        "% leaf\n",
        encoding="utf-8",
    )
    main = paper / "main.tex"
    got = collect_transitive_tex_inputs(paper, main)
    assert (paper / "ablation_results.tex").resolve() in got
    assert (paper / "tables" / "ablation_selected_results.tex").resolve() in got


def test_sanitize_boxingvi_results_all_tex_fixes_legacy_micro_macro(tmp_path: Path) -> None:
    path = tmp_path / "boxingvi_results_all.tex"
    path.write_text(
        "\\midrule\n\\textbf{__micro__} & 1 & 2 & 3 \\\\\n\\textbf{__macro__} & & & \\\\\n",
        encoding="utf-8",
    )
    _sanitize_boxingvi_results_all_tex(path)
    body = path.read_text(encoding="utf-8")
    assert r"\textbf{\_\_micro\_\_}" in body
    assert r"\textbf{\_\_macro\_\_}" in body
    assert "\\textbf{__micro__}" not in body
