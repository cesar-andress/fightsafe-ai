"""Run TapKO pose detectors and write JSON / CSV / Markdown (offline; no database)."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fightsafe_ai.events.tap_detector import TapCandidateEvent, detect_tap_candidates
from fightsafe_ai.events.vulnerability_detector import (
    VulnerabilityCandidateEvent,
    detect_vulnerability_candidates,
)
from fightsafe_ai.exceptions import VideoIOError
from fightsafe_ai.pipeline import steps
from fightsafe_ai.pipeline.output_paths import paths_for_run_root
from fightsafe_ai.tapko.coco_stack import load_coco17_stack_from_pose_csv


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TapkoDetectOutputs:
    """Paths written by :func:`run_tapko_detect`."""

    root: Path
    predictions_json: Path
    predictions_csv: Path
    report_md: Path
    manifest_json: Path
    pose_csv: Path


def _events_to_prediction_rows(
    video_id: str,
    tap_ev: list[TapCandidateEvent],
    vuln_ev: list[VulnerabilityCandidateEvent],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tap in tap_ev:
        rows.append(
            {
                "video_id": video_id,
                "start_time": float(tap.start_time),
                "end_time": float(tap.end_time),
                "event_type": str(tap.event_type),
                "score": float(tap.score),
                "level": None,
                "repetition_count": int(tap.repetition_count),
                "explanation": str(tap.explanation or ""),
                "requires_human_confirmation": bool(tap.requires_human_confirmation),
                "evidence": dict(tap.evidence),
            }
        )
    for vuln in vuln_ev:
        rows.append(
            {
                "video_id": video_id,
                "start_time": float(vuln.start_time),
                "end_time": float(vuln.end_time),
                "event_type": str(vuln.event_type),
                "score": float(vuln.score),
                "level": str(vuln.level),
                "repetition_count": None,
                "explanation": str(vuln.explanation or ""),
                "requires_human_confirmation": bool(vuln.requires_human_confirmation),
                "evidence": dict(vuln.evidence),
            }
        )
    rows.sort(key=lambda r: (float(r["start_time"]), str(r["event_type"])))
    return rows


def _write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "event_type",
        "start_time",
        "end_time",
        "score",
        "level",
        "repetition_count",
        "explanation",
        "requires_human_confirmation",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat: dict[str, Any] = {}
            for k in fieldnames:
                v = r.get(k, "")
                if v is None:
                    flat[k] = ""
                else:
                    flat[k] = v
            w.writerow(flat)


def _write_report_md(
    path: Path,
    *,
    video_id: str,
    source_video: Path,
    fps: float,
    pose_source: str,
    pose_csv: Path,
    n_frames: int,
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# TapKO detection report",
        "",
        "**Decision-support candidates only** — not official match outcomes or medical findings.",
        "",
        "## Run",
        "",
        f"- **Video id**: `{video_id}`",
        f"- **Source**: `{source_video}`",
        f"- **FPS**: {fps:g}",
        f"- **Pose source**: {pose_source}",
        f"- **Pose CSV**: `{pose_csv}`",
        f"- **Frames (pose)**: {n_frames}",
        f"- **Candidates emitted**: {len(rows)}",
        "",
        "## Events",
        "",
    ]
    if not rows:
        lines.append("_No tap or vulnerability candidates exceeded detector thresholds._")
    else:
        lines.extend(
            [
                "| Start (s) | End (s) | Type | Score | Notes |",
                "|-----------|---------|------|-------|-------|",
            ]
        )
        for r in rows:
            expl = str(r.get("explanation", "")).replace("|", "\\|").replace("\n", " ")
            if len(expl) > 120:
                expl = expl[:117] + "..."
            lines.append(
                f"| {float(r['start_time']):.3f} | {float(r['end_time']):.3f} | "
                f"`{r['event_type']}` | {float(r['score']):.4f} | {expl} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_tapko_detect(
    *,
    source_video: Path,
    output_dir: Path,
    fps: float,
    video_id: str | None = None,
    pose_csv: Path | None = None,
    pose_backend: str = "mediapipe",
    model_complexity: int = 1,
    min_detection: float = 0.5,
) -> TapkoDetectOutputs:
    """
    Extract pose (unless ``pose_csv`` is provided), run tap + vulnerability detectors, write artifacts.

    Writes under ``output_dir``:

    * ``tapko_predictions.json`` — array of prediction dicts (compatible with TapKO evaluate).
    * ``tapko_predictions.csv`` — tabular subset for spreadsheets.
    * ``tapko_report.md`` — human-readable summary.
    * ``tapko_manifest.json`` — provenance (video path, fps, pose path).
    * When pose is extracted: ``frames/``, ``pose_keypoints.csv`` (standard MVP layout).
    """
    vid_path = source_video.expanduser().resolve()
    if not vid_path.is_file():
        raise VideoIOError(f"Video not found: {vid_path}")

    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    vid_key = video_id if video_id else vid_path.stem

    pred_json = root / "tapko_predictions.json"
    pred_csv = root / "tapko_predictions.csv"
    report_md = root / "tapko_report.md"
    manifest_path = root / "tapko_manifest.json"

    pose_path = pose_csv.expanduser().resolve() if pose_csv is not None else None
    pose_source = "precomputed"

    if pose_path is not None:
        if not pose_path.exists():
            raise VideoIOError(f"--pose-csv not found: {pose_path}")
        stack_xy = load_coco17_stack_from_pose_csv(pose_path)
        pose_csv_out = pose_path
    else:
        pose_source = "extracted"
        paths = paths_for_run_root(root)
        extract_fps = max(1, round(float(fps)))
        logger.info("TapKO: extracting frames -> %s @ %s FPS", paths.frames_dir, extract_fps)
        steps.step01_extract_frames(vid_path, paths, fps=extract_fps)
        logger.info("TapKO: pose (%s) -> %s", pose_backend, paths.pose_keypoints_csv)
        steps.step02_estimate_pose(
            paths,
            model_complexity=int(model_complexity),
            min_detection=float(min_detection),
            pose_backend=str(pose_backend),
        )
        pose_csv_out = paths.pose_keypoints_csv
        stack_xy = load_coco17_stack_from_pose_csv(pose_csv_out)

    t_n = int(stack_xy.shape[0])
    if t_n < 8:
        raise VideoIOError(
            f"Need at least 8 pose frames; got {t_n}. Check video length or pose CSV."
        )

    fps_f = float(max(fps, 1e-6))
    tap_ev = detect_tap_candidates(stack_xy, fps_f)
    vuln_ev = detect_vulnerability_candidates(stack_xy, fps_f)

    rows = _events_to_prediction_rows(vid_key, tap_ev, vuln_ev)

    pred_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_predictions_csv(pred_csv, rows)

    manifest: dict[str, Any] = {
        "schema_id": "fightsafe_ai.tapko_detect_manifest",
        "video_id": vid_key,
        "source_video": str(vid_path),
        "fps": float(fps_f),
        "pose_source": pose_source,
        "pose_csv": str(pose_csv_out),
        "n_frames": t_n,
        "n_tap_candidates": len(tap_ev),
        "n_vulnerability_candidates": len(vuln_ev),
        "predictions_json": pred_json.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _write_report_md(
        report_md,
        video_id=vid_key,
        source_video=vid_path,
        fps=fps_f,
        pose_source=pose_source,
        pose_csv=pose_csv_out,
        n_frames=t_n,
        rows=rows,
    )

    return TapkoDetectOutputs(
        root=root,
        predictions_json=pred_json,
        predictions_csv=pred_csv,
        report_md=report_md,
        manifest_json=manifest_path,
        pose_csv=pose_csv_out,
    )


__all__ = ["TapkoDetectOutputs", "run_tapko_detect"]
