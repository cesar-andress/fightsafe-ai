"""Hashing, TeX closure helpers, and manifest entries for ``assets_manifest.json``."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str | None:
    """Return ``sha256:<hex>`` for file contents, or ``None`` if missing/unreadable."""
    if not path.is_file():
        return None
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"
    except OSError:
        return None


def rel_repo(repo_root: Path, path: Path) -> str:
    """Stable posix path relative to repo root."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_previous_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def assets_by_path(prev: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index prior ``assets`` list by ``asset_path`` string."""
    out: dict[str, dict[str, Any]] = {}
    for a in prev.get("assets") or []:
        if isinstance(a, dict) and "asset_path" in a:
            out[str(a["asset_path"])] = a
    return out


def build_asset_entry(
    *,
    asset_path: Path,
    repo_root: Path,
    source_paths: list[Path],
    command: str,
    status: str,
    used_in_main_tex: bool,
) -> dict[str, Any]:
    src_hashes: dict[str, str | None] = {}
    src_rel: list[str] = []
    for sp in source_paths:
        rp = rel_repo(repo_root, sp)
        src_rel.append(rp)
        src_hashes[rp] = sha256_file(sp)
    return {
        "asset_path": rel_repo(repo_root, asset_path),
        "source_paths": src_rel,
        "source_hashes": src_hashes,
        "command": command,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "used_in_main_tex": used_in_main_tex,
    }


# --- TeX closure (also used by build QA helpers) ---------------------------------


def resolve_input_tex_path(paper_dir: Path, arg: str) -> Path:
    arg = arg.strip()
    p = paper_dir / arg
    if p.suffix.lower() != ".tex":
        p = p.with_suffix(".tex")
    return p.resolve()


def collect_transitive_tex_inputs(paper_dir: Path, main_tex: Path) -> set[Path]:
    """Files reachable from ``main.tex`` via ``\\input`` / ``\\include`` (recursive)."""
    paper_dir = paper_dir.resolve()
    main_tex = main_tex.resolve()
    seen: set[Path] = set()
    queue: deque[Path] = deque([main_tex])
    pat = re.compile(r"\\(?:input|include)\{([^}]+)\}")

    while queue:
        path = queue.popleft()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in pat.finditer(text):
            child = resolve_input_tex_path(paper_dir, m.group(1))
            if child.is_file() and child not in seen:
                queue.append(child)
    return seen


def collect_includegraphics_basenames(paper_dir: Path, tex_files: set[Path]) -> set[str]:
    """Basenames referenced via ``\\includegraphics`` in given tex files."""
    pat = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    out: set[str] = set()
    for path in tex_files:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in pat.finditer(text):
            raw = m.group(1).strip()
            for prefix in ("figures/", "../docs/figures/", "docs/figures/"):
                if raw.startswith(prefix):
                    raw = raw[len(prefix) :]
                    break
            out.add(Path(raw).name)
    return out


def tex_asset_used_in_main(paper_dir: Path, rel_under_paper: str) -> bool:
    """True if ``rel_under_paper`` appears in the ``\\input`` / ``\\include`` closure of ``main.tex``."""
    paper_dir = paper_dir.resolve()
    main_path = paper_dir / "main.tex"
    if not main_path.is_file():
        return False
    rel = rel_under_paper.replace("\\", "/").lstrip("/")
    reachable = collect_transitive_tex_inputs(paper_dir, main_path)
    reachable.add(main_path.resolve())
    pat = re.compile(r"\\(?:input|include)\{([^}]+)\}")
    for p in reachable:
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in pat.finditer(text):
            child = resolve_input_tex_path(paper_dir, m.group(1))
            if not child.is_file():
                continue
            try:
                r = child.relative_to(paper_dir).as_posix()
            except ValueError:
                continue
            if r == rel:
                return True
    return False


def figure_asset_used_in_main_tex(paper_dir: Path, figure_filename: str) -> bool:
    """True if basename appears in ``\\includegraphics`` across ``main.tex`` closure."""
    paper_dir = paper_dir.resolve()
    main_path = paper_dir / "main.tex"
    if not main_path.is_file():
        return False
    reachable = collect_transitive_tex_inputs(paper_dir, main_path)
    reachable.add(main_path.resolve())
    used = collect_includegraphics_basenames(paper_dir, reachable)
    return figure_filename in used


# --- BoxingVI / ablation source bundles ----------------------------------------


def collect_boxingvi_dataset_source_paths(dataset_root: Path, video_ids: list[str]) -> list[Path]:
    """Annotation workbooks and skeleton arrays used by batch eval (best-effort)."""
    root = Path(dataset_root).expanduser().resolve()
    found: list[Path] = []
    for vid in video_ids:
        stem = str(vid).strip()
        if not stem:
            continue
        for sub in ("Annotation_files", "annotations"):
            p = root / sub / f"{stem}.xlsx"
            if p.is_file():
                found.append(p)
                break
        sk = root / "skeleton" / f"{stem}.npy"
        if sk.is_file():
            found.append(sk)
    return sorted({p.resolve() for p in found}, key=lambda x: x.as_posix())


def boxingvi_batch_expected_tex_paths(output_dir: Path, *, compare_baselines: bool) -> list[Path]:
    out = Path(output_dir).expanduser().resolve()
    paths = [out / "boxingvi_results_all.tex"]
    if compare_baselines:
        paths.append(out / "baseline_comparison.tex")
    return paths


def hash_boxingvi_batch_sources(
    repo_root: Path,
    dataset_root: Path,
    video_ids: list[str],
    *,
    fps: float,
    strike_percentile: float,
    strike_merge_frames: int,
    tolerance_seconds: float,
    compare_baselines: bool,
) -> dict[str, str | None]:
    """Paths + hashes for dataset inputs and a stable params blob."""
    paths = collect_boxingvi_dataset_source_paths(dataset_root, video_ids)
    out: dict[str, str | None] = {}
    for p in paths:
        out[rel_repo(repo_root, p)] = sha256_file(p)
    sig = {
        "video_ids": [str(v).strip() for v in video_ids],
        "fps": fps,
        "strike_percentile": strike_percentile,
        "strike_merge_frames": strike_merge_frames,
        "tolerance_seconds": tolerance_seconds,
        "compare_baselines": compare_baselines,
    }
    blob = json.dumps(sig, sort_keys=True, separators=(",", ":")).encode()
    out["__boxingvi_params__.json"] = f"sha256:{hashlib.sha256(blob).hexdigest()}"
    return out


def collect_generate_paper_assets_source_paths(
    repo_root: Path,
    ablation_csv: Path,
    ablation_summary: Path,
) -> list[Path]:
    """Scripts and ablation exports that feed ``scripts/generate_paper_assets.py``."""
    root = Path(repo_root).expanduser().resolve()
    paths: list[Path] = []
    script = root / "scripts" / "generate_paper_assets.py"
    if script.is_file():
        paths.append(script)
    for rel in (
        "tools/generate_ablation_paper_assets.py",
        "tools/generate_quantitative_observations_tex.py",
    ):
        p = root / rel
        if p.is_file():
            paths.append(p)
    ac = Path(ablation_csv).expanduser().resolve()
    if ac.is_file():
        paths.append(ac)
    summ = Path(ablation_summary).expanduser().resolve()
    if summ.is_dir():
        n_csv = 0
        for p in sorted(summ.rglob("*.csv")):
            if n_csv >= 400:
                break
            if p.is_file():
                paths.append(p)
                n_csv += 1
        for p in summ.rglob("ablation_risk_timeline.png"):
            paths.append(p)
            break
    return sorted({p.resolve() for p in paths}, key=lambda x: x.as_posix())


def hash_generate_paper_sources(
    repo_root: Path,
    ablation_csv: Path,
    ablation_summary: Path,
) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in collect_generate_paper_assets_source_paths(repo_root, ablation_csv, ablation_summary):
        out[rel_repo(repo_root, p)] = sha256_file(p)
    return out


__all__ = [
    "assets_by_path",
    "boxingvi_batch_expected_tex_paths",
    "build_asset_entry",
    "collect_boxingvi_dataset_source_paths",
    "collect_generate_paper_assets_source_paths",
    "collect_includegraphics_basenames",
    "collect_transitive_tex_inputs",
    "figure_asset_used_in_main_tex",
    "hash_boxingvi_batch_sources",
    "hash_generate_paper_sources",
    "load_previous_manifest",
    "rel_repo",
    "resolve_input_tex_path",
    "sha256_file",
    "tex_asset_used_in_main",
]
