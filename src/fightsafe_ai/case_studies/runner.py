"""
Run all case studies from :file:`configs/case_studies.yaml` (local {YouTube} only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fightsafe_ai.case_studies.clip import prepare_input_clip
from fightsafe_ai.case_studies.tables import build_events_table_tex, write_global_summaries
from fightsafe_ai.exceptions import VideoCutError, VideoDownloadError
from fightsafe_ai.pipeline import youtube_demo
from fightsafe_ai.pipeline.demo import run_e2e_demo
from fightsafe_ai.qa.quality_report import QualityReport
from fightsafe_ai.video.downloader import download_video
from fightsafe_ai.visualization.plots import plot_pose_coverage


logger = logging.getLogger(__name__)

CASE_SOURCE_DIRNAME = youtube_demo.DEMO_YOUTUBE_SOURCE_DIRNAME


@dataclass(frozen=True)
class CaseStudyFileConfig:
    version: str
    base_dir: str
    case_studies: list[dict[str, Any]]


def _repo_root_from_config(config_path: Path) -> Path:
    """
    Config lives in ``<repo>/configs/…``; return repository root.
    If structure differs, use ``config_path`` parent twice.
    """
    p = config_path.expanduser().resolve()
    if p.parent.name == "configs" and p.parent.parent.is_dir():
        return p.parent.parent
    return p.parent


def load_case_study_file(path: Path) -> CaseStudyFileConfig:
    raw = path.expanduser().resolve()
    with raw.open(encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ValueError("Case study YAML must be a mapping at root.")
    cs = d.get("case_studies", [])
    if not isinstance(cs, list) or not cs:
        raise ValueError("case_studies: non-empty list required")
    for i, c in enumerate(cs):
        if not isinstance(c, dict):
            raise ValueError(f"case_studies[{i}] must be a mapping")
        for key in ("case_id", "title", "url", "expected_focus", "output_dir"):
            if not str(c.get(key, "")).strip():
                raise ValueError(f"case_studies[{i}].{key} is required")
    return CaseStudyFileConfig(
        version=str(d.get("version", "1.0")),
        base_dir=str(d.get("base_dir", "runs/case_studies")),
        case_studies=list(cs),
    )


def _pose_plot_if_available(run_dir: Path) -> None:
    if not (run_dir / "pose_keypoints.csv").is_file():
        return
    try:
        out = run_dir / "pose_coverage.png"
        plot_pose_coverage(run_dir, output_path=out)
        logger.info("Wrote %s", out)
    except (FileNotFoundError, OSError, ValueError) as e:
        logger.info("pose_coverage.png skipped: %s", e)


def run_one_case(
    case: dict[str, Any],
    case_run_root: Path,
    *,
    rules_yaml: Path | None = None,
    fps: int = 10,
) -> tuple[Path, bool, QualityReport | None]:
    """Download, prepare clip, run :func:`run_e2e_demo`, optional pose coverage plot."""
    url = str(case["url"]).strip()
    case_run_root = case_run_root.expanduser().resolve()
    case_run_root.mkdir(parents=True, exist_ok=True)
    (case_run_root / CASE_SOURCE_DIRNAME).mkdir(parents=True, exist_ok=True)

    dl_name = f"{str(case.get('case_id', 'v')).replace('/', '_')}.mp4"
    try:
        full = download_video(url, case_run_root / CASE_SOURCE_DIRNAME, filename=dl_name)
    except VideoDownloadError as e:
        raise VideoDownloadError(f"{case.get('case_id')}: {e}") from e

    st = case.get("start_time")
    et = case.get("end_time")
    st_s = None if st is None else str(st).strip() or None
    et_s = None if et is None else str(et).strip() or None
    if st_s in {"null", "None"}:
        st_s = None
    if et_s in {"null", "None"}:
        et_s = None
    try:
        clip = prepare_input_clip(
            full,
            case_run_root,
            st_s,
            et_s,
            clip_basename=youtube_demo.DEMO_YOUTUBE_INPUT_CLIP,
        )
    except (ValueError, VideoCutError) as e:
        if isinstance(e, ValueError):
            raise ValueError(f"{case.get('case_id')}: {e}") from e
        raise VideoCutError(f"{case.get('case_id')}: {e}") from e

    out_paths, qa_ok, qreport = run_e2e_demo(
        clip,
        case_run_root,
        rules_yaml=rules_yaml,
        fps=fps,
    )
    _pose_plot_if_available(case_run_root)
    (case_run_root / "case_study_meta.yaml").write_text(
        yaml.safe_dump(
            {
                k: v
                for k, v in case.items()
                if k in ("case_id", "title", "url", "expected_focus", "notes")
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return out_paths.root, bool(qa_ok), qreport


def run_case_studies_from_config(
    config_path: Path,
    *,
    rules_yaml: Path | None = None,
    fps: int = 10,
) -> list[dict[str, Any]]:
    """
    For each case: download, optional trim, :func:`run_e2e_demo`, write
    ``events_table.tex`` and global ``case_study_summary`` under the case-study base dir.
    """
    cfg = load_case_study_file(config_path)
    repo = _repo_root_from_config(config_path)
    out_base = (repo / Path(cfg.base_dir).as_posix()).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    report_rows: list[tuple[dict[str, Any], Path]] = []
    out: list[dict[str, Any]] = []
    for case in cfg.case_studies:
        cid = str(case["output_dir"] or case["case_id"])
        rdir = out_base / cid
        item: dict[str, Any] = {
            "case_id": case.get("case_id"),
            "output_dir": str(rdir),
            "ok": False,
        }
        try:
            rroot, ok, _q = run_one_case(case, rdir, rules_yaml=rules_yaml, fps=fps)
            item["ok"] = bool(ok)
            item["run_root"] = str(rroot)
            etp = rroot / "events_table.tex"
            etp.write_text(build_events_table_tex(rroot), encoding="utf-8")
            logger.info("Wrote %s", etp)
            report_rows.append((case, rroot))
        except (OSError, VideoDownloadError, VideoCutError, ValueError, RuntimeError) as e:
            logger.exception("Case %s failed: %s", case.get("case_id"), e)
            item["error"] = str(e)
        out.append(item)
    if report_rows:
        write_global_summaries(out_base, report_rows)
        logger.info("Wrote %s and case_study_summary.tex", out_base / "case_study_summary.csv")
    return out
