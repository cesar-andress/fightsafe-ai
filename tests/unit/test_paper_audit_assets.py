"""Tests for paper asset audit."""

from __future__ import annotations

from pathlib import Path

from fightsafe_ai.paper.audit_assets import (
    audit_assets,
    docs_figures_not_copied_to_paper,
)


def test_docs_gap_detects_missing_stem(tmp_path: Path) -> None:
    doc = tmp_path / "docs" / "figures"
    pap = tmp_path / "paper" / "figures"
    doc.mkdir(parents=True)
    pap.mkdir(parents=True)
    (doc / "only_in_docs.svg").write_bytes(b"<svg/>")
    (pap / "present.png").write_bytes(b"\x89PNG")
    gap = docs_figures_not_copied_to_paper(
        paper_dir=tmp_path / "paper",
        docs_figures=doc,
        cwd=tmp_path,
    )
    assert any("only_in_docs" in g for g in gap)


def test_audit_assets_smoke_minimal_paper(tmp_path: Path) -> None:
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text("% minimal\n", encoding="utf-8")
    (paper / "figures").mkdir(parents=True)
    doc = tmp_path / "docs" / "figures"
    doc.mkdir(parents=True)
    out = audit_assets(paper_dir=paper, docs_figures=doc, cwd=tmp_path)
    assert "unused_paper_figures" in out
    assert "recommendations" in out
