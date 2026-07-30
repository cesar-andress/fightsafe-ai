"""
Reproducible paper build: tests, BoxingVI inspect, batch eval, asset generation, LaTeX.

Does not start PostgreSQL, download models, or invent metrics. Failures are recorded in
``paper/paper_qa_report.md`` and exit non-zero unless ``--continue-on-error``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fightsafe_ai.paper.asset_manifest import (
    assets_by_path,
    boxingvi_batch_expected_tex_paths,
    build_asset_entry,
    collect_boxingvi_dataset_source_paths,
    collect_generate_paper_assets_source_paths,
    collect_includegraphics_basenames,
    collect_transitive_tex_inputs,
    figure_asset_used_in_main_tex,
    hash_boxingvi_batch_sources,
    hash_generate_paper_sources,
    load_previous_manifest,
    rel_repo,
    sha256_file,
    tex_asset_used_in_main,
)
from fightsafe_ai.reports.paper_build import ensure_boxingvi_paper_table_fragments


_REPO_ROOT = Path(__file__).resolve().parents[3]

DOCS_FIGURE_NAMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("framework_architecture", ("png", "svg")),
    ("risk_fusion_model", ("png", "svg")),
    ("evaluation_protocol", ("png", "svg")),
    ("human_in_the_loop_alerts", ("png", "svg")),
    ("event_detection", ("png", "svg")),
)

DEFAULT_ABLATION_CSV = Path("runs/case_studies/ablation_summary/ablation_all_runs.csv")
DEFAULT_ABLATION_SUMMARY = Path("runs/case_studies/ablation_summary")


@dataclass
class StepResult:
    name: str
    ok: bool
    returncode: int | None
    command: list[str]
    stdout_tail: str = ""
    stderr_tail: str = ""
    detail: str = ""


@dataclass
class BuildState:
    steps: list[StepResult] = field(default_factory=list)
    inspect_stdout: str = ""
    inspect_returncode: int | None = None
    copied_tables: list[str] = field(default_factory=list)
    copied_figures: list[str] = field(default_factory=list)
    skipped_figures: list[str] = field(default_factory=list)
    latex_log_excerpt: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    assets: list[dict[str, Any]] = field(default_factory=list)


def _repo_root() -> Path:
    return _REPO_ROOT


def _sanitize_boxingvi_results_all_tex(path: Path) -> None:
    """Fix legacy TeX with raw ``__micro__`` / ``__macro__`` in ``\\textbf{}`` (underscores break text mode)."""
    path = Path(path)
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    fixed = text.replace(r"\textbf{__micro__}", r"\textbf{\_\_micro\_\_}").replace(
        r"\textbf{__macro__}", r"\textbf{\_\_macro\_\_}"
    )
    if fixed != text:
        path.write_text(fixed, encoding="utf-8")


def _tail(s: str, max_chars: int = 8000) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return f"...[truncated]\n{s[-max_chars:]}"


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def list_tex_not_reachable_from_main(paper_dir: Path) -> list[str]:
    """``.tex`` files under ``paper_dir`` not pulled in by ``main.tex`` (best-effort)."""
    paper_dir = paper_dir.resolve()
    main_path = paper_dir / "main.tex"
    if not main_path.is_file():
        return []
    reachable = collect_transitive_tex_inputs(paper_dir, main_path)
    all_tex = {p.resolve() for p in paper_dir.rglob("*.tex")}
    orphan = sorted(all_tex - reachable - {main_path.resolve()})
    try:
        return [str(p.relative_to(paper_dir)) for p in orphan]
    except ValueError:
        return [str(p) for p in orphan]


def list_unused_figures(paper_dir: Path) -> list[str]:
    """Files in ``paper/figures`` never referenced by ``\\includegraphics`` in the main closure."""
    paper_dir = paper_dir.resolve()
    main_path = paper_dir / "main.tex"
    fig_dir = paper_dir / "figures"
    if not main_path.is_file() or not fig_dir.is_dir():
        return []
    reachable_tex = collect_transitive_tex_inputs(paper_dir, main_path)
    reachable_tex.add(main_path.resolve())
    used = collect_includegraphics_basenames(paper_dir, reachable_tex)
    unused: list[str] = []
    for f in sorted(fig_dir.iterdir()):
        if f.is_file() and f.name not in used:
            unused.append(f.name)
    return unused


def _copy_if_missing(src: Path, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return False
    if not src.is_file():
        return False
    shutil.copy2(src, dest)
    return True


def copy_docs_figures_to_paper(repo_root: Path, paper_dir: Path, state: BuildState) -> None:
    docs_fig = repo_root / "docs" / "figures"
    dest_dir = paper_dir / "figures"
    for stem, exts in DOCS_FIGURE_NAMES:
        copied = False
        for ext in exts:
            src = docs_fig / f"{stem}.{ext}"
            if not src.is_file():
                continue
            dest = dest_dir / f"{stem}.{ext}"
            if dest.is_file():
                state.skipped_figures.append(f"{stem}.{ext} (already in paper/figures)")
                copied = True
                break
            if _copy_if_missing(src, dest):
                state.copied_figures.append(f"{stem}.{ext}")
                copied = True
                break
        if not copied:
            tried = ", ".join(f"{stem}.{e}" for e in exts)
            state.skipped_figures.append(f"{stem} (no source: checked {tried} under docs/figures)")


def run_pytest(repo_root: Path) -> StepResult:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit",
        "tests/integration",
    ]
    code, out, err = _run_cmd(cmd, cwd=repo_root)
    ok = code == 0
    return StepResult(
        name="pytest",
        ok=ok,
        returncode=code,
        command=cmd,
        stdout_tail=_tail(out),
        stderr_tail=_tail(err),
    )


def run_boxingvi_inspect(repo_root: Path, dataset_root: Path) -> StepResult:
    cmd = [
        sys.executable,
        "-m",
        "fightsafe_ai.datasets.boxingvi",
        "--dataset-root",
        str(dataset_root),
        "--inspect",
    ]
    code, out, err = _run_cmd(cmd, cwd=repo_root)
    return StepResult(
        name="boxingvi_inspect",
        ok=code == 0,
        returncode=code,
        command=cmd,
        stdout_tail=_tail(out),
        stderr_tail=_tail(err),
    )


def build_boxingvi_batch_cmd(
    *,
    dataset_root: Path,
    video_ids: list[str],
    output_dir: Path,
    fps: float,
    strike_percentile: float,
    strike_merge_frames: int,
    tolerance_seconds: float,
    compare_baselines: bool,
    force: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fightsafe_ai.evaluation.boxingvi_batch_eval",
        "--dataset-root",
        str(dataset_root),
        "--video-ids",
        *video_ids,
        "--fps",
        str(fps),
        "--output-dir",
        str(output_dir),
        "--strike-percentile",
        str(strike_percentile),
        "--strike-merge-frames",
        str(strike_merge_frames),
        "--tolerance-seconds",
        str(tolerance_seconds),
    ]
    if compare_baselines:
        cmd.append("--compare-baselines")
    if force:
        cmd.append("--force")
    return cmd


def run_boxingvi_batch_eval(
    repo_root: Path,
    *,
    dataset_root: Path,
    video_ids: list[str],
    output_dir: Path,
    fps: float,
    strike_percentile: float,
    strike_merge_frames: int,
    tolerance_seconds: float,
    compare_baselines: bool,
    force: bool,
) -> StepResult:
    cmd = build_boxingvi_batch_cmd(
        dataset_root=dataset_root,
        video_ids=video_ids,
        output_dir=output_dir,
        fps=fps,
        strike_percentile=strike_percentile,
        strike_merge_frames=strike_merge_frames,
        tolerance_seconds=tolerance_seconds,
        compare_baselines=compare_baselines,
        force=force,
    )
    code, out, err = _run_cmd(cmd, cwd=repo_root)
    return StepResult(
        name="boxingvi_batch_eval",
        ok=code == 0,
        returncode=code,
        command=cmd,
        stdout_tail=_tail(out),
        stderr_tail=_tail(err),
    )


def _should_skip_boxingvi_batch_eval(
    output_dir: Path, *, compare_baselines: bool, force: bool
) -> bool:
    if force:
        return False
    return all(
        p.is_file()
        for p in boxingvi_batch_expected_tex_paths(output_dir, compare_baselines=compare_baselines)
    )


def record_boxingvi_batch_export_assets(
    state: BuildState,
    repo_root: Path,
    *,
    dataset_root: Path,
    video_ids: list[str],
    output_dir: Path,
    args_ns: argparse.Namespace,
    skipped: bool,
) -> None:
    cmd_list = build_boxingvi_batch_cmd(
        dataset_root=dataset_root,
        video_ids=video_ids,
        output_dir=output_dir,
        fps=float(args_ns.fps),
        strike_percentile=float(args_ns.strike_percentile),
        strike_merge_frames=int(args_ns.strike_merge_frames),
        tolerance_seconds=float(args_ns.tolerance_seconds),
        compare_baselines=bool(args_ns.compare_baselines),
        force=bool(args_ns.force),
    )
    fp = hash_boxingvi_batch_sources(
        repo_root,
        dataset_root,
        video_ids,
        fps=float(args_ns.fps),
        strike_percentile=float(args_ns.strike_percentile),
        strike_merge_frames=int(args_ns.strike_merge_frames),
        tolerance_seconds=float(args_ns.tolerance_seconds),
        compare_baselines=bool(args_ns.compare_baselines),
    )
    src_paths = collect_boxingvi_dataset_source_paths(dataset_root, video_ids)
    cmd_str = shlex.join(cmd_list)
    status = "skipped_existing_outputs" if skipped else "generated"
    for tex in boxingvi_batch_expected_tex_paths(
        output_dir, compare_baselines=bool(args_ns.compare_baselines)
    ):
        if not tex.is_file():
            continue
        entry = build_asset_entry(
            asset_path=tex,
            repo_root=repo_root,
            source_paths=src_paths,
            command=cmd_str,
            status=status,
            used_in_main_tex=False,
        )
        merged = dict(entry["source_hashes"])
        merged.update(fp)
        entry["source_hashes"] = merged
        state.assets.append(entry)


def build_generate_paper_assets_cmd(repo_root: Path, paper_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "scripts" / "generate_paper_assets.py"),
        "--paper",
        str(paper_dir),
        "--skip-run",
    ]


def should_skip_generate_paper_assets(
    repo_root: Path,
    paper_dir: Path,
    *,
    ablation_csv: Path,
    ablation_summary: Path,
    force: bool,
    prev_by_asset: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    if force:
        return False, ""
    ablation_tex = paper_dir / "tables" / "ablation_selected_results.tex"
    quant_tex = paper_dir / "tables" / "quantitative_observations.tex"
    if not ablation_tex.is_file() or not quant_tex.is_file():
        return False, "missing paper tables"
    fp = hash_generate_paper_sources(repo_root, ablation_csv, ablation_summary)
    key = rel_repo(repo_root, ablation_tex)
    prev = prev_by_asset.get(key) or prev_by_asset.get(rel_repo(repo_root, quant_tex))
    if prev is None:
        return True, "outputs present (no matching manifest entry)"
    if (prev.get("source_hashes") or {}) == fp:
        return True, "sources unchanged"
    return False, "sources changed vs manifest"


def record_generate_paper_outputs_assets(
    state: BuildState,
    repo_root: Path,
    paper_dir: Path,
    *,
    ablation_csv: Path,
    ablation_summary: Path,
    cmd_list: list[str],
    skipped: bool,
) -> None:
    src_paths = collect_generate_paper_assets_source_paths(
        repo_root, ablation_csv, ablation_summary
    )
    fp = hash_generate_paper_sources(repo_root, ablation_csv, ablation_summary)
    cmd_str = shlex.join(cmd_list) if cmd_list else "generate_paper_assets"
    status = "skipped_existing_outputs" if skipped else "generated"
    outputs = [
        paper_dir / "tables" / "ablation_selected_results.tex",
        paper_dir / "tables" / "quantitative_observations.tex",
        paper_dir / "figures" / "ablation_risk_timeline.png",
    ]
    for out in outputs:
        if not out.is_file():
            continue
        ru = out.relative_to(paper_dir).as_posix()
        if out.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg"}:
            used = figure_asset_used_in_main_tex(paper_dir, out.name)
        else:
            used = tex_asset_used_in_main(paper_dir, ru)
        entry = build_asset_entry(
            asset_path=out,
            repo_root=repo_root,
            source_paths=src_paths,
            command=cmd_str,
            status=status,
            used_in_main_tex=used,
        )
        merged = dict(entry["source_hashes"])
        merged.update(fp)
        entry["source_hashes"] = merged
        state.assets.append(entry)


def append_docs_figure_manifest_entries(
    repo_root: Path, paper_dir: Path, state: BuildState
) -> None:
    docs_fig = repo_root / "docs" / "figures"
    dest_dir = paper_dir / "figures"
    for stem, exts in DOCS_FIGURE_NAMES:
        for ext in exts:
            dest = dest_dir / f"{stem}.{ext}"
            src = docs_fig / f"{stem}.{ext}"
            if not dest.is_file():
                continue
            src_paths = [src] if src.is_file() else []
            st = "present" if src.is_file() else "present_without_doc_source"
            entry = build_asset_entry(
                asset_path=dest,
                repo_root=repo_root,
                source_paths=src_paths,
                command="copy_docs_figures_to_paper",
                status=st,
                used_in_main_tex=figure_asset_used_in_main_tex(paper_dir, dest.name),
            )
            state.assets.append(entry)
            break


def record_paper_table_copy_assets(
    state: BuildState,
    repo_root: Path,
    paper_dir: Path,
    *,
    dest: Path,
    src: Path | None,
    copy_status: str,
) -> None:
    rel = dest.relative_to(paper_dir).as_posix()
    used = tex_asset_used_in_main(paper_dir, rel)
    src_paths = [src] if src is not None and src.is_file() else []
    entry = build_asset_entry(
        asset_path=dest,
        repo_root=repo_root,
        source_paths=src_paths,
        command=f"shutil.copy2 {src} → {dest}" if src is not None else "shutil.copy2 (no source)",
        status=copy_status,
        used_in_main_tex=used,
    )
    state.assets.append(entry)


def copy_generated_tables(
    repo_root: Path,
    output_dir: Path,
    paper_dir: Path,
    *,
    compare_baselines: bool,
    force: bool,
    state: BuildState,
) -> list[StepResult]:
    results: list[StepResult] = []
    tables_dir = paper_dir / "tables"
    pairs: list[tuple[Path, Path]] = [
        (
            output_dir / "boxingvi_results_all.tex",
            tables_dir / "boxingvi_results_all.tex",
        ),
    ]
    if compare_baselines:
        pairs.append(
            (
                output_dir / "baseline_comparison.tex",
                tables_dir / "baseline_comparison.tex",
            )
        )

    for src_p, dest_p in pairs:
        src = src_p.resolve()
        dest = dest_p.resolve()
        cmd = ["copy", str(src), str(dest)]
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not src.is_file():
            try:
                rel = str(src.relative_to(repo_root))
            except ValueError:
                rel = str(src)
            if dest.is_file():
                record_paper_table_copy_assets(
                    state,
                    repo_root,
                    paper_dir,
                    dest=dest,
                    src=None,
                    copy_status="kept_existing_missing_export",
                )
                results.append(
                    StepResult(
                        name=f"copy_table:{dest.name}",
                        ok=True,
                        returncode=None,
                        command=cmd,
                        detail=f"kept existing {dest.name} (export missing at {rel})",
                    )
                )
            else:
                record_paper_table_copy_assets(
                    state,
                    repo_root,
                    paper_dir,
                    dest=dest,
                    src=None,
                    copy_status="missing_source_and_dest",
                )
                results.append(
                    StepResult(
                        name=f"copy_table:{dest.name}",
                        ok=False,
                        returncode=None,
                        command=cmd,
                        detail=f"Source missing: {rel}",
                    )
                )
            continue

        hs = sha256_file(src)
        hd = sha256_file(dest)

        if not force and dest.is_file() and hs is not None and hd is not None and hs == hd:
            record_paper_table_copy_assets(
                state,
                repo_root,
                paper_dir,
                dest=dest,
                src=src,
                copy_status="unchanged",
            )
            results.append(
                StepResult(
                    name=f"copy_table:{dest.name}",
                    ok=True,
                    returncode=None,
                    command=["shutil.copy2", str(src), str(dest)],
                    detail=f"unchanged (same hash as export): {dest.name}",
                )
            )
            continue

        prev_dest = dest.is_file()
        shutil.copy2(src, dest)
        rel = str(dest.relative_to(paper_dir)) if dest.is_relative_to(paper_dir) else str(dest)
        state.copied_tables.append(rel)
        record_paper_table_copy_assets(
            state,
            repo_root,
            paper_dir,
            dest=dest,
            src=src,
            copy_status="updated" if prev_dest else "copied",
        )
        results.append(
            StepResult(
                name=f"copy_table:{dest.name}",
                ok=True,
                returncode=0,
                command=["shutil.copy2", str(src), str(dest)],
                detail=str(dest),
            )
        )
    out_tex = tables_dir / "boxingvi_results_all.tex"
    _sanitize_boxingvi_results_all_tex(out_tex)
    return results


def run_generate_paper_assets(repo_root: Path, paper_dir: Path) -> StepResult:
    script = repo_root / "scripts" / "generate_paper_assets.py"
    cmd = build_generate_paper_assets_cmd(repo_root, paper_dir)
    if not script.is_file():
        return StepResult(
            name="generate_paper_assets",
            ok=False,
            returncode=None,
            command=cmd,
            detail=f"Missing {script}",
        )
    code, out, err = _run_cmd(cmd, cwd=repo_root)
    return StepResult(
        name="generate_paper_assets",
        ok=code == 0,
        returncode=code,
        command=cmd,
        stdout_tail=_tail(out),
        stderr_tail=_tail(err),
    )


def run_latex_compile(paper_dir: Path, state: BuildState) -> list[StepResult]:
    main_tex = paper_dir / "main.tex"
    if not main_tex.is_file():
        return [
            StepResult(
                name="pdflatex",
                ok=False,
                returncode=None,
                command=[],
                detail=f"Missing {main_tex}",
            )
        ]

    results: list[StepResult] = []

    def pdflatex() -> StepResult:
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "main.tex",
        ]
        code, out, err = _run_cmd(cmd, cwd=paper_dir)
        log_path = paper_dir / "main.log"
        excerpt = ""
        if log_path.is_file():
            try:
                excerpt = _tail(log_path.read_text(encoding="utf-8", errors="replace"), 12000)
            except OSError:
                excerpt = ""
        state.latex_log_excerpt = excerpt
        return StepResult(
            name="pdflatex",
            ok=code == 0,
            returncode=code,
            command=cmd,
            stdout_tail=_tail(out),
            stderr_tail=_tail(err),
        )

    def bibtex() -> StepResult:
        cmd = ["bibtex", "main"]
        code, out, err = _run_cmd(cmd, cwd=paper_dir)
        return StepResult(
            name="bibtex",
            ok=code == 0,
            returncode=code,
            command=cmd,
            stdout_tail=_tail(out),
            stderr_tail=_tail(err),
        )

    results.append(pdflatex())
    results.append(bibtex())
    results.append(pdflatex())
    results.append(pdflatex())
    return results


def write_assets_manifest(
    path: Path,
    *,
    state: BuildState,
    repo_root: Path,
    args_ns: argparse.Namespace,
    orphan_tex: list[str],
    unused_figures: list[str],
) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root),
        "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args_ns).items()},
        "steps": [
            {
                "name": s.name,
                "ok": s.ok,
                "returncode": s.returncode,
                "command": [str(x) for x in s.command],
                "detail": s.detail,
            }
            for s in state.steps
        ],
        "copied_tables": state.copied_tables,
        "copied_figures": state.copied_figures,
        "skipped_or_missing_figures": state.skipped_figures,
        "tex_files_not_in_main_closure": orphan_tex,
        "unused_figures_in_paper_figures": unused_figures,
        "assets": state.assets,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    state.manifest = payload


def write_qa_report(
    path: Path,
    *,
    state: BuildState,
    orphan_tex: list[str],
    unused_figures: list[str],
    overall_ok: bool,
) -> None:
    lines: list[str] = [
        "# Paper build QA report",
        "",
        f"Generated (UTC): `{datetime.now(UTC).isoformat()}`",
        "",
        "## Overall status",
        "",
        "PASS" if overall_ok else "FAIL",
        "",
        "## Tests (`pytest tests/unit tests/integration`)",
        "",
    ]
    st = next((s for s in state.steps if s.name == "pytest"), None)
    if st and st.detail == "skipped":
        lines.append("- (skipped: `--run-tests` not set)")
        lines.append("")
    elif st:
        lines.extend(
            [
                f"- **Status:** {'OK' if st.ok else 'FAILED'} (exit {st.returncode})",
                "",
                "<details><summary>stdout (tail)</summary>",
                "",
                "```",
                st.stdout_tail or "(empty)",
                "```",
                "",
                "</details>",
                "",
            ]
        )
    else:
        lines.append("- (unknown)")
        lines.append("")

    lines.extend(
        [
            "## BoxingVI inspect",
            "",
        ]
    )
    insp = next((s for s in state.steps if s.name == "boxingvi_inspect"), None)
    if insp:
        lines.extend(
            [
                f"- **Status:** {'OK' if insp.ok else 'FAILED'} (exit {insp.returncode})",
                "",
                "<details><summary>stdout (tail)</summary>",
                "",
                "```",
                insp.stdout_tail or "(empty)",
                "```",
                "",
                "</details>",
                "",
            ]
        )
    else:
        lines.append("- (not run)\n")

    lines.extend(["## BoxingVI batch evaluation", ""])
    be = next((s for s in state.steps if s.name == "boxingvi_batch_eval"), None)
    if be:
        lines.extend(
            [
                f"- **Status:** {'OK' if be.ok else 'FAILED'} (exit {be.returncode})",
                "",
                "<details><summary>stdout (tail)</summary>",
                "",
                "```",
                be.stdout_tail or "(empty)",
                "```",
                "",
                "</details>",
                "",
            ]
        )
    else:
        lines.append("- (not run)\n")

    lines.extend(
        [
            "## Generated / copied tables",
            "",
        ]
    )
    if state.copied_tables:
        for t in state.copied_tables:
            lines.append(f"- `{t}`")
    else:
        lines.append("- (none recorded)")
    lines.append("")
    for s in state.steps:
        if s.name.startswith("copy_table:"):
            lines.append(f"- **{s.name}:** {'OK' if s.ok else 'FAILED'} {s.detail}")
    lines.append("")

    lines.extend(
        [
            "## Copied figures (`docs/figures` -> `paper/figures`, if missing)",
            "",
        ]
    )
    if state.copied_figures:
        for f in state.copied_figures:
            lines.append(f"- Copied: `{f}`")
    else:
        lines.append("- (none new)")
    lines.append("")
    if state.skipped_figures:
        lines.append("### Skipped / missing sources")
        lines.append("")
        for f in state.skipped_figures:
            lines.append(f"- {f}")
        lines.append("")

    lines.extend(["## `scripts/generate_paper_assets.py`", ""])
    ga = next((s for s in state.steps if s.name == "generate_paper_assets"), None)
    if ga:
        lines.extend(
            [
                f"- **Status:** {'OK' if ga.ok else 'FAILED'} (exit {ga.returncode})",
                "",
            ]
        )
    else:
        lines.append("- (not run)\n")

    lines.extend(["## LaTeX compile", ""])
    latex_steps = [s for s in state.steps if s.name in ("pdflatex", "bibtex")]
    if latex_steps:
        for s in latex_steps:
            lines.append(f"- **{s.name}:** {'OK' if s.ok else 'FAILED'} (exit {s.returncode})")
        lines.append("")
        if state.latex_log_excerpt:
            lines.extend(
                [
                    "<details><summary>main.log (tail)</summary>",
                    "",
                    "```",
                    state.latex_log_excerpt[:12000],
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )
    else:
        lines.append("- (not run: `--compile` not set)\n")

    lines.extend(
        [
            "## Unused files in `paper/figures`",
            "",
            "(No `\\includegraphics{...}` reference in `main.tex` closure.)",
            "",
        ]
    )
    if unused_figures:
        for u in unused_figures:
            lines.append(f"- `{u}`")
    else:
        lines.append("- (none detected)")
    lines.append("")

    lines.extend(
        [
            "## `.tex` files not reachable from `main.tex` via `\\input`",
            "",
        ]
    )
    if orphan_tex:
        for o in orphan_tex:
            lines.append(f"- `{o}`")
    else:
        lines.append("- (none detected)")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-root", type=Path, default=Path("data/boxingvi"))
    p.add_argument(
        "--video-ids",
        nargs="+",
        default=[f"V{i}" for i in range(1, 11)],
    )
    p.add_argument("--paper-dir", type=Path, default=Path("../fusion2026"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation/boxingvi_batch"),
    )
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--strike-percentile", type=float, default=85.0)
    p.add_argument("--strike-merge-frames", type=int, default=8)
    p.add_argument("--tolerance-seconds", type=float, default=0.5)
    p.add_argument("--run-tests", action="store_true")
    p.add_argument("--compare-baselines", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run all steps even when one fails; exit 1 if any step failed.",
    )
    args = p.parse_args(argv)

    repo_root = _repo_root()
    paper_dir = (
        (repo_root / args.paper_dir).resolve()
        if not args.paper_dir.is_absolute()
        else args.paper_dir.resolve()
    )
    dataset_root = (
        (repo_root / args.dataset_root).resolve()
        if not args.dataset_root.is_absolute()
        else args.dataset_root.resolve()
    )
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )

    manifest_path = paper_dir / "assets_manifest.json"
    prev_manifest = load_previous_manifest(manifest_path)
    prev_by_asset = assets_by_path(prev_manifest)

    ablation_csv = (repo_root / DEFAULT_ABLATION_CSV).resolve()
    ablation_summary = (repo_root / DEFAULT_ABLATION_SUMMARY).resolve()

    state = BuildState()
    any_fail = False

    def record(sr: StepResult) -> None:
        nonlocal any_fail
        state.steps.append(sr)
        if not sr.ok:
            any_fail = True

    def should_stop() -> bool:
        return any_fail and not args.continue_on_error

    if args.run_tests:
        record(run_pytest(repo_root))
    else:
        state.steps.append(
            StepResult(
                name="pytest",
                ok=True,
                returncode=None,
                command=[],
                detail="skipped",
            )
        )

    if not should_stop():
        sr = run_boxingvi_inspect(repo_root, dataset_root)
        record(sr)
        state.inspect_stdout = sr.stdout_tail
        state.inspect_returncode = sr.returncode

    skip_batch = _should_skip_boxingvi_batch_eval(
        output_dir,
        compare_baselines=bool(args.compare_baselines),
        force=bool(args.force),
    )
    if not should_stop():
        if skip_batch:
            record(
                StepResult(
                    name="boxingvi_batch_eval",
                    ok=True,
                    returncode=None,
                    command=[],
                    detail="skipped (expected `.tex` exports already exist; use --force to regenerate)",
                )
            )
        else:
            record(
                run_boxingvi_batch_eval(
                    repo_root,
                    dataset_root=dataset_root,
                    video_ids=list(args.video_ids),
                    output_dir=output_dir,
                    fps=float(args.fps),
                    strike_percentile=float(args.strike_percentile),
                    strike_merge_frames=int(args.strike_merge_frames),
                    tolerance_seconds=float(args.tolerance_seconds),
                    compare_baselines=bool(args.compare_baselines),
                    force=bool(args.force),
                )
            )
        record_boxingvi_batch_export_assets(
            state,
            repo_root,
            dataset_root=dataset_root,
            video_ids=list(args.video_ids),
            output_dir=output_dir,
            args_ns=args,
            skipped=skip_batch,
        )

    if not should_stop():
        for c in copy_generated_tables(
            repo_root,
            output_dir,
            paper_dir,
            compare_baselines=bool(args.compare_baselines),
            force=bool(args.force),
            state=state,
        ):
            record(c)

    gen_cmd = build_generate_paper_assets_cmd(repo_root, paper_dir)
    skip_gen, skip_gen_reason = should_skip_generate_paper_assets(
        repo_root,
        paper_dir,
        ablation_csv=ablation_csv,
        ablation_summary=ablation_summary,
        force=bool(args.force),
        prev_by_asset=prev_by_asset,
    )
    if not should_stop():
        if skip_gen:
            record(
                StepResult(
                    name="generate_paper_assets",
                    ok=True,
                    returncode=None,
                    command=gen_cmd,
                    detail=f"skipped ({skip_gen_reason})",
                )
            )
        else:
            record(run_generate_paper_assets(repo_root, paper_dir))
        record_generate_paper_outputs_assets(
            state,
            repo_root,
            paper_dir,
            ablation_csv=ablation_csv,
            ablation_summary=ablation_summary,
            cmd_list=gen_cmd,
            skipped=skip_gen,
        )

    if not should_stop():
        copy_docs_figures_to_paper(repo_root, paper_dir, state)
        state.steps.append(
            StepResult(
                name="copy_docs_figures",
                ok=True,
                returncode=0,
                command=["copy_docs_figures_to_paper"],
                detail=f"{len(state.copied_figures)} copied; {len(state.skipped_figures)} notes",
            )
        )

    if args.compile:
        if should_stop():
            state.steps.append(
                StepResult(
                    name="pdflatex",
                    ok=False,
                    returncode=None,
                    command=[],
                    detail="skipped (stopped after earlier failure; use --continue-on-error to run)",
                )
            )
        else:
            for ls in run_latex_compile(paper_dir, state):
                record(ls)
    else:
        state.steps.append(
            StepResult(
                name="pdflatex",
                ok=True,
                returncode=None,
                command=[],
                detail="skipped (--compile not set)",
            )
        )

    append_docs_figure_manifest_entries(repo_root, paper_dir, state)

    wbox = ensure_boxingvi_paper_table_fragments(paper_dir)
    if wbox:
        state.steps.append(
            StepResult(
                name="ensure_boxingvi_paper_tables",
                ok=True,
                returncode=0,
                command=[],
                detail="wrote " + ", ".join(p.name for p in wbox),
            )
        )

    orphan_tex = list_tex_not_reachable_from_main(paper_dir)
    unused_figs = list_unused_figures(paper_dir)

    report_path = paper_dir / "paper_qa_report.md"
    write_assets_manifest(
        manifest_path,
        state=state,
        repo_root=repo_root,
        args_ns=args,
        orphan_tex=orphan_tex,
        unused_figures=unused_figs,
    )
    overall_ok = not any_fail
    write_qa_report(
        report_path,
        state=state,
        orphan_tex=orphan_tex,
        unused_figures=unused_figs,
        overall_ok=overall_ok,
    )

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "collect_transitive_tex_inputs",
    "list_tex_not_reachable_from_main",
    "list_unused_figures",
    "main",
]
