"""
Audit paper figures and ``.tex`` fragments: orphans, missing copies from ``docs/figures``,
and unreachable inputs under ``paper/``.

Does **not** delete or modify files.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fightsafe_ai.paper.build_all import (
    collect_transitive_tex_inputs,
    list_tex_not_reachable_from_main,
    list_unused_figures,
)


_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".eps", ".gif", ".webp"}
)


def _stem_key(path: Path) -> str:
    return path.stem.lower()


def _list_image_files(fig_dir: Path) -> list[Path]:
    if not fig_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(fig_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            out.append(p)
    return out


def docs_figures_not_copied_to_paper(
    *,
    paper_dir: Path,
    docs_figures: Path,
    cwd: Path | None = None,
) -> list[str]:
    """
    Image files under ``docs/figures`` whose **stem** does not appear as any file stem
    under ``paper/figures`` (same stem, any supported extension).
    """
    paper_dir = paper_dir.resolve()
    docs_figures = docs_figures.resolve()
    cwd = (cwd or Path.cwd()).resolve()
    paper_fig = paper_dir / "figures"
    paper_stems = {_stem_key(p) for p in _list_image_files(paper_fig)}
    missing: list[str] = []
    for src in _list_image_files(docs_figures):
        if _stem_key(src) not in paper_stems:
            try:
                missing.append(str(src.relative_to(cwd)))
            except ValueError:
                missing.append(str(src))
    return sorted(missing)


def unreferenced_table_tex_files(paper_dir: Path) -> list[str]:
    """``.tex`` files under ``paper/tables`` not in the ``main.tex`` input closure."""
    paper_dir = paper_dir.resolve()
    main_path = paper_dir / "main.tex"
    tables_dir = paper_dir / "tables"
    if not main_path.is_file() or not tables_dir.is_dir():
        return []
    reachable = collect_transitive_tex_inputs(paper_dir, main_path)
    out: list[str] = []
    for p in sorted(tables_dir.rglob("*.tex")):
        if p.resolve() not in reachable:
            try:
                out.append(str(p.relative_to(paper_dir)))
            except ValueError:
                out.append(str(p))
    return out


def build_recommendations(
    *,
    unused_paper_figures: list[str],
    docs_missing_in_paper: list[str],
    orphan_tex: list[str],
    unreferenced_tables: list[str],
) -> list[str]:
    rec: list[str] = []
    if unused_paper_figures:
        rec.append(
            "Unused files in paper/figures: add \\includegraphics{...} from the document "
            "(or a file \\input{} from main.tex), or archive/remove if obsolete."
        )
    if docs_missing_in_paper:
        rec.append(
            "docs/figures assets without a matching stem in paper/figures: copy needed images "
            "(e.g. shutil or build pipeline) so paper builds stay self-contained under figures/."
        )
    if orphan_tex:
        rec.append(
            "Unreachable .tex under paper/: wire via \\input{...} from main.tex (or drop "
            "from version control if duplicate/dead)."
        )
    if unreferenced_tables:
        rec.append(
            "Unreachable tables/*.tex: \\input{tables/...} from an included fragment or remove "
            "if superseded."
        )
    if not rec:
        rec.append("No issues detected by this audit.")
    return rec


def audit_assets(
    *,
    paper_dir: Path,
    docs_figures: Path,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """
    Run full audit. Figure usage follows the **main.tex closure** (main plus recursive
    ``\\input`` / ``\\include``), consistent with LaTeX compilation.
    """
    paper_dir = paper_dir.expanduser().resolve()
    docs_figures = docs_figures.expanduser().resolve()
    cw = (cwd or Path.cwd()).resolve()

    unused_paper = list_unused_figures(paper_dir)
    docs_not_in_paper = docs_figures_not_copied_to_paper(
        paper_dir=paper_dir,
        docs_figures=docs_figures,
        cwd=cw,
    )
    orphan_tex = list_tex_not_reachable_from_main(paper_dir)
    unref_tables = unreferenced_table_tex_files(paper_dir)

    recommendations = build_recommendations(
        unused_paper_figures=unused_paper,
        docs_missing_in_paper=docs_not_in_paper,
        orphan_tex=orphan_tex,
        unreferenced_tables=unref_tables,
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "paper_dir": str(paper_dir),
        "docs_figures_dir": str(docs_figures),
        "note": (
            "Unused paper figures are computed vs \\includegraphics references in main.tex "
            "and all tex files reachable from main via \\input/\\include."
        ),
        "unused_paper_figures": unused_paper,
        "docs_figures_not_in_paper": docs_not_in_paper,
        "orphan_tex_under_paper": orphan_tex,
        "unreferenced_tables_tex": unref_tables,
        "recommendations": recommendations,
    }


def _is_under_tables(rel: str) -> bool:
    return rel.replace("\\", "/").startswith("tables/")


def _non_empty_list(val: object) -> bool:
    return isinstance(val, list) and len(val) > 0


def print_console_table(payload: dict[str, Any]) -> None:
    unused = list(payload["unused_paper_figures"])
    docs_gap = list(payload["docs_figures_not_in_paper"])
    orphans_all = list(payload["orphan_tex_under_paper"])
    orphans = [p for p in orphans_all if not _is_under_tables(p)]
    tables = list(payload["unreferenced_tables_tex"])
    recs = list(payload["recommendations"])

    sections: list[tuple[str, list[str]]] = [
        ("1. Unused in paper/figures (vs main.tex \\input closure + \\includegraphics)", unused),
        ("2. docs/figures not mirrored in paper/figures (by filename stem)", docs_gap),
        (
            "3. Orphan .tex under paper/ (outside tables/; see also section 4)",
            orphans,
        ),
        ("4. Unreferenced paper/tables/*.tex", tables),
        ("Recommendations", recs),
    ]

    print("Paper asset audit")
    print(f"  paper_dir: {payload['paper_dir']}")
    print(f"  docs_figures: {payload['docs_figures_dir']}")
    print()

    for title, items in sections:
        print(title)
        print("-" * min(88, max(len(title), 12)))
        if not items:
            print("  (none)\n")
            continue
        for line in items:
            print(f"  - {line}")
        print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper-dir", type=Path, default=Path("paper"))
    p.add_argument("--docs-figures", type=Path, default=Path("docs/figures"))
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write JSON report (default: <paper-dir>/paper_asset_audit.json)",
    )
    args = p.parse_args(argv)

    repo_cwd = Path.cwd()
    paper_dir = args.paper_dir
    if not paper_dir.is_absolute():
        paper_dir = (repo_cwd / paper_dir).resolve()
    else:
        paper_dir = paper_dir.resolve()

    docs_fig = args.docs_figures
    if not docs_fig.is_absolute():
        docs_fig = (repo_cwd / docs_fig).resolve()
    else:
        docs_fig = docs_fig.resolve()

    payload = audit_assets(paper_dir=paper_dir, docs_figures=docs_fig, cwd=repo_cwd)

    json_path = args.json_out
    if json_path is None:
        json_path = paper_dir / "paper_asset_audit.json"
    else:
        json_path = json_path.expanduser().resolve()

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print_console_table(payload)
    print(f"Wrote {json_path}")

    has_issues = (
        _non_empty_list(payload["unused_paper_figures"])
        or _non_empty_list(payload["docs_figures_not_in_paper"])
        or _non_empty_list(payload["orphan_tex_under_paper"])
        or _non_empty_list(payload["unreferenced_tables_tex"])
    )
    return 1 if has_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_assets", "main"]
