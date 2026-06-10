"""
FightSafe AI

AI-assisted safety detection for combat sports officiating.

Authors:
- David Martin Moncunill (david.martinm@ucjc.edu)
- César Andrés Sánchez (cesar.andress@ucjc.edu)

Affiliation:
Camilo José Cela University (UCJC)
Madrid, Spain

This module is part of a research-oriented system for human-in-the-loop safety analysis.

Typer-based CLI for FightSafe AI. Business logic lives in library modules; this file only
validates inputs, dispatches, prints summaries, and maps errors to exit codes.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import typer


# Shown at the top of `fightsafe --help` (subcommands listed below by Typer).
_FIGHTSAFE_APP_HELP = """
FightSafe AI — decision-support for combat sports safety (pose, heuristics, reports)

Primary commands:
  demo              Full pipeline (MVP) + QA + plots + report bundle; prints key output paths
  demo-youtube      Download URL, cut [start,end), then same as demo
  qa                Quality checks; writes qa_report.json
  build-events      Per-frame risk CSV -> merged events JSON (for custom pipelines)
  report            Subcommands: `report html` (and generate-report for full bundle)
  plot-risk         risk_timeline.png and events_timeline.png from a run directory
  risk-ablation     Formal fusion ablation (behavior metrics) from a run with features.csv
  risk-ablation-all Same for every case subdir under a base directory
  tapko-detect       Pose → tap/vulnerability detectors → tapko_predictions.* + report (no DB)
  tapko-evaluate     Annotation JSON vs predictions JSON → CSV/TeX/MD metrics
  tapko-validate-annotations  Validate a TapKO annotation file (schema)
  tapko-export-examples       Write example TapKO JSON files for tooling
  annotate-template  Empty JSON template for manual evaluation labels (safety events)
  validate-annotations  Check a filled annotation file against the schema
  suggest-annotations   LLM + optional VLM drafts -> annotation_suggestions.json (not ground truth)
  run-case-studies    YouTube case-study set from YAML (illustrative; not a benchmark)
  evaluate          Compare pipeline events.json to manual ground-truth annotations
  evaluate-case-studies  Batch event-level metrics for illustrative case studies vs annotations/
  download          Fetch video with yt-dlp
  generate-report   report.md, report.html, summary.json from a run directory
  extract-frames    Sample frames to JPEGs
  bench-pose-backends  Torch RTMPose vs ONNX Runtime (FPS / latency; GPU-oriented)
  export-rtmpose-onnx   Export MMPose checkpoint to ONNX (optional mmengine/mmpose)
  (see --help for estimate-pose, compute-features, detect-risk, render-overlay, run-pipeline, …)

Authors: D. Martin Moncunill, C. A. Sánchez (UCJC), Madrid, Spain
""".strip()

app = typer.Typer(
    name="fightsafe",
    help=_FIGHTSAFE_APP_HELP,
    no_args_is_help=True,
)

report_app = typer.Typer(
    name="report",
    help="Generate report artifacts from a completed pipeline run directory.",
    no_args_is_help=True,
)


# --- Validation & UX helpers (CLI-only; no domain rules) ---


def _err(msg: str) -> None:
    typer.echo(msg, err=True)


def _fail(msg: str, code: int = 1) -> None:
    _err(msg)
    raise typer.Exit(code)


def _ok(msg: str) -> None:
    typer.echo(msg)


def _require_file(path: Path, label: str = "File") -> Path:
    """Resolve *path*; must exist as a file. *label* is for error messages (positional or keyword)."""
    p = path.expanduser().resolve()
    if not p.is_file():
        _fail(f"{label} not found or not a file: {p}")
    return p


def _require_dir(path: Path, *, label: str) -> Path:
    p = path.expanduser().resolve()
    if not p.is_dir():
        _fail(f"{label} not found or not a directory: {p}")
    return p


def _ensure_output_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_fps(name: str, v: float) -> float:
    if v <= 0 or v > 1_000_000:
        _fail(f"{name} must be a positive, reasonable number; got {v!r}.")
    return v


def _validate_fps_int(name: str, v: int) -> int:
    if v < 1 or v > 1_000_000:
        _fail(f"{name} must be an integer >= 1; got {v!r}.")
    return v


def _validate_complexity(v: int) -> int:
    if v not in (0, 1, 2):
        _fail(f"--complexity must be 0, 1, or 2; got {v!r}.")
    return v


def _validate_rolling(v: int) -> int:
    if v < 1:
        _fail(f"rolling window must be >= 1; got {v!r}.")
    return v


def _require_nonempty_url(url: str) -> str:
    s = (url or "").strip()
    if not s:
        _fail("URL is empty. Pass a non-empty video or stream URL.", code=2)
    return s


def _validate_llm_temperature(v: float | None) -> None:
    if v is None:
        return
    if not 0.0 <= float(v) <= 2.0:
        _fail("--llm-temperature must be in [0.0, 2.0].", code=2)


def _print_qa_terminal_summary(report: Any) -> None:
    """Print QA status, check counts, warnings, max risk, and event count (``fightsafe qa``)."""
    m = dict(getattr(report, "metrics", None) or {})
    n_warn = len(getattr(report, "warnings", None) or [])
    status = "PASS" if getattr(report, "passed", False) else "FAIL"
    max_r = m.get("max_risk_score")
    if isinstance(max_r, (int, float)) and max_r == max_r:
        max_s = f"{float(max_r):.4f}"
    else:
        max_s = "n/a"
    ne = m.get("n_events")
    if isinstance(ne, (int, float)) and ne == ne:
        try:
            ev_s = str(int(ne))
        except (TypeError, ValueError, OverflowError):
            ev_s = "n/a"
    else:
        ev_s = "n/a"
    _ok("")
    _ok("=== Quality assurance ===")
    _ok(f"  status:            {status}")
    _ok(f"  checks (total):   {int(getattr(report, 'total_checks', 0))}")
    _ok(f"  failed:            {int(getattr(report, 'failed_checks', 0))}")
    _ok(f"  warnings:         {n_warn}")
    _ok(f"  max risk score:   {max_s}")
    _ok(f"  events (listed):  {ev_s}")
    pc = m.get("pose_coverage_percent")
    if isinstance(pc, (int, float)) and pc == pc:
        _ok(f"  pose coverage:    {float(pc):.1f}%")
    if "llm_enabled" in m:
        le = m.get("llm_enabled")
        lu = m.get("llm_used")
        ls = m.get("llm_success_rate")
        le_s = "yes" if le is True else "no" if le is False else "n/a"
        lu_s = "yes" if lu is True else "no" if lu is False else "n/a"
        if isinstance(ls, (int, float)) and ls == ls:
            lr_s = f"{float(ls):.0%}" if 0.0 <= float(ls) <= 1.0 else f"{float(ls):.4f}"
        else:
            lr_s = "n/a"
        _ok(f"  LLM (config on):  {le_s}")
        _ok(f"  LLM (artifacts):  {lu_s}")
        _ok(f"  LLM success rate: {lr_s}")
    _ok("")


def _display_path(path: Path) -> str:
    """Prefer a path relative to the current working directory for readable CLI output."""
    p = path.expanduser().resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(p.relative_to(cwd))
    except ValueError:
        return str(p)


def _print_demo_completed_outputs(root: Path, *, heading: str = "Demo completed.") -> None:
    """
    Print the canonical list of deliverables after ``demo`` / ``demo-youtube``.

    Paths match :func:`~fightsafe_ai.pipeline.output_paths.paths_for_run_root` layout.
    """
    r = root.expanduser().resolve()
    overlay = r / "output_overlay.mp4"
    report_html = r / "report.html"
    qa_json = r / "qa_report.json"
    risk_png = r / "risk_timeline.png"
    _ok("")
    _ok(heading)
    _ok("")
    _ok("Outputs:")
    _ok(f"- Overlay video: {_display_path(overlay)}")
    _ok(f"- Report: {_display_path(report_html)}")
    _ok(f"- QA: {_display_path(qa_json)}")
    _ok(f"- Risk plot: {_display_path(risk_png)}")
    _ok("")
    _ok(f"Run directory: {_display_path(r)}")


# --- 1. download -----------------------------------------------------------------


@app.command(
    "download",
    help="Download a video or stream (yt-dlp) into a directory. Output codec depends on the source; "
    "if later stages read zero frames, re-encode to H.264 (see docs/internet-video-codecs.md).",
)
def cmd_download(
    url: str = typer.Argument(help="Non-empty video or stream URL"),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output directory (created if needed)",
    ),
    filename: str | None = typer.Option(
        None,
        "-n",
        "--filename",
        help="Optional file name in that directory",
    ),
) -> None:
    from fightsafe_ai.exceptions import VideoDownloadError
    from fightsafe_ai.video.downloader import download_video

    u = _require_nonempty_url(url)
    out = output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        p = download_video(u, out, filename=filename)
    except VideoDownloadError as exc:
        _err(str(exc))
        raise typer.Exit(1) from exc
    _ok(f"Downloaded: {p}")


# --- cut (utility) ----------------------------------------------------------------


@app.command("cut", help="Extract [start, end) with FFmpeg (stream copy).")
def cmd_cut(
    input_video: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Input media",
    ),
    out: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output file path (parent directory created if needed)",
    ),
    start: str = typer.Option(
        ...,
        "--start",
        "-s",
        help="Start (seconds or MM:SS)",
    ),
    end: str = typer.Option(
        ...,
        "--end",
        "-e",
        help="End (seconds or MM:SS)",
    ),
) -> None:
    from fightsafe_ai.exceptions import VideoCutError
    from fightsafe_ai.video.cutter import cut_clip

    s0, t0 = (x.strip() for x in (start, end))
    if not s0 or not t0:
        _fail("Both --start and --end must be non-empty.")
    _ensure_output_parent(out)
    try:
        path = cut_clip(input_video, s0, t0, out)
    except VideoCutError as exc:
        _err(str(exc))
        raise typer.Exit(1) from exc
    _ok(f"Wrote clip: {path}")


# --- 2. extract-frames -----------------------------------------------------------


@app.command("extract-frames", help="Write sampled video frames to a directory (JPEG).")
def cmd_extract_frames(
    video: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        help=(
            "Input video. Prefer H.264 in MP4; many internet/YouTube files need re-encoding "
            "for OpenCV (see docs/internet-video-codecs.md)."
        ),
    ),
    out: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Frame output directory (created if needed)",
    ),
    fps: int = typer.Option(10, "--fps", help="Target frame sampling rate (>= 1)"),
) -> None:
    from fightsafe_ai.exceptions import VideoIOError
    from fightsafe_ai.video.frame_extractor import NO_FRAMES_USER_HINT, extract_frames

    _validate_fps_int("FPS", fps)
    v = _require_file(video, label="Video")
    out_dir = out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths = extract_frames(v, out_dir, fps=fps)
    except (VideoIOError, ValueError) as exc:
        _err(str(exc))
        raise typer.Exit(1) from exc
    n = len(paths)
    if n == 0:
        _err(
            f"No frames written under {out_dir} (empty stream or I/O error).\n{NO_FRAMES_USER_HINT}"
        )
        raise typer.Exit(1)
    _ok(f"Success: {n} frame file(s) -> {out_dir}")


# --- 3. estimate-pose ------------------------------------------------------------


@app.command("estimate-pose", help="BlazePose on a folder of images -> one pose CSV.")
def cmd_estimate_pose(
    images_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Consolidated keypoints CSV (parent dirs created as needed)",
    ),
    complexity: int = typer.Option(1, "--complexity", help="0 / 1 / 2"),
    min_detection: float = typer.Option(0.5, "--min-detection", help="[0,1] min detection score"),
    pattern: str | None = typer.Option(
        None,
        "--pattern",
        help="Optional single glob, e.g. *.png (default: *.jpg, *.jpeg, *.png in estimator).",
    ),
    pose_backend: str = typer.Option(
        "mediapipe",
        "--pose-backend",
        help="mediapipe (default), mock, yolo, or rtmpose (optional ultralytics / mmpose).",
    ),
) -> None:
    if not 0.0 <= float(min_detection) <= 1.0:
        _fail("--min-detection must be between 0.0 and 1.0.")
    _validate_complexity(complexity)
    d = _require_dir(images_dir, label="Images directory")
    from fightsafe_ai.pose.factory import create_pose_estimator

    globs = (pattern.strip(),) if pattern and pattern.strip() else None
    estimator = create_pose_estimator(
        pose_backend,
        model_complexity=complexity,
        min_detection_confidence=min_detection,
        glob_patterns=globs,
    )
    _ensure_output_parent(output)
    p = estimator.estimate_folder(d, output)
    _ok(f"Pose keypoints: {p}")


@app.command(
    "bench-pose-backends",
    help="Benchmark Torch RTMPose vs ONNX Runtime (FPS / latency); install CUDA PyTorch + onnxruntime-gpu for GPU.",
)
def cmd_bench_pose_backends(
    frames_dir: Path = typer.Option(
        ...,
        "--frames-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory of frame images for timing.",
    ),
    onnx_model: Path = typer.Option(
        ...,
        "--onnx-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="ONNX file for the ONNX Runtime path.",
    ),
    limit: int = typer.Option(100, "--limit", help="Max frames to benchmark."),
    warmup: int = typer.Option(5, "--warmup", help="Warmup iterations per backend."),
    pose2d: str = typer.Option(
        "rtmpose-m_8xb256-210e_coco-256x192",
        "--pose2d",
        help="MMPose pose2d model id for the Torch path.",
    ),
    device: str = typer.Option("cuda:0", "--device", help="Torch device (e.g. cuda:0, cpu)."),
    fp16: bool = typer.Option(False, "--fp16", help="Torch AMP FP16 and ONNX float16 inputs."),
    onnx_device_id: int = typer.Option(
        0, "--onnx-device-id", help="CUDA device index for ONNX Runtime."
    ),
) -> None:
    from fightsafe_ai.pose.bench_torch_onnx import run_benchmark

    globs = ("*.jpg", "*.jpeg", "*.png")
    result = run_benchmark(
        frames_dir=frames_dir,
        globs=globs,
        limit=limit,
        warmup=warmup,
        pose2d=pose2d,
        device=device,
        fp16=fp16,
        onnx_model=onnx_model,
        onnx_device_id=onnx_device_id,
    )
    if result.get("error") == "no_images":
        _fail("No images loaded from --frames-dir (use JPEG/PNG under that directory).")


@app.command(
    "export-rtmpose-onnx",
    help="Export an MMPose RTMPose checkpoint to ONNX (requires mmengine + mmpose + torch).",
)
def cmd_export_rtmpose_onnx(
    config: Path = typer.Option(
        ...,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="MMEngine config .py path.",
    ),
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Model .pth checkpoint.",
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output .onnx path (parent dirs created as needed).",
    ),
    height: int = typer.Option(256, "--height"),
    width: int = typer.Option(192, "--width"),
    opset: int = typer.Option(17, "--opset"),
    device: str = typer.Option(
        "cpu", "--device", help="Torch device for loading weights before export."
    ),
) -> None:
    from fightsafe_ai.pose.export_rtmpose_onnx import export_rtmpose_to_onnx

    code = export_rtmpose_to_onnx(
        config,
        checkpoint,
        output,
        height=height,
        width=width,
        opset=opset,
        device=device,
    )
    if code != 0:
        raise typer.Exit(code)
    _ok(f"ONNX export written to {output.expanduser().resolve()}")


# --- 4. compute-features ---------------------------------------------------------


@app.command(
    "compute-features", help="Frame-wise biomechanics from keypoint CSV(s) or a directory of CSVs."
)
def cmd_compute_features(
    keypoints_source: Path = typer.Argument(
        ...,
        exists=True,
        help="Per-frame CSV directory or one consolidated pose.csv",
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output features CSV",
    ),
    fps: float = typer.Option(10.0, "--fps", help="Assumed sequence sampling rate in Hz"),
    rolling_window: int = typer.Option(5, "--rolling-window", help="Rolling window size (>= 1)"),
    ground_y: float = typer.Option(0.82, "--ground-y", help="Normalized y threshold (near ground)"),
) -> None:
    from fightsafe_ai.features.biomechanics import compute_pose_features

    _validate_fps("FPS", float(fps))
    _validate_rolling(rolling_window)
    if not 0.0 < float(ground_y) < 1.0:
        _fail("--ground-y should lie strictly between 0 and 1 (image-normalized y).", code=2)
    k = keypoints_source.expanduser().resolve()
    if not k.is_file() and not k.is_dir():
        _fail(f"Keypoints path must be a file or directory: {k}", code=2)
    _ensure_output_parent(output)
    df = compute_pose_features(
        k,
        fps=float(fps),
        rolling_window=rolling_window,
        ground_y_threshold=ground_y,
    )
    if len(df) == 0:
        _err("No feature rows were produced; check the keypoints input and FPS.")
        raise typer.Exit(1)
    df.to_csv(output, index=False)
    _ok(f"Wrote {output} ({len(df)} rows).")


# --- 5. detect-risk --------------------------------------------------------------


@app.command(
    "detect-risk",
    help="Heuristic risk columns from a features table (see configs/risk_rules.yaml).",
)
def cmd_detect_risk(
    features_csv: Path = typer.Argument(
        ..., exists=True, file_okay=True, help="Features CSV from compute-features"
    ),
    output: Path = typer.Option(..., "-o", "--output", help="Output risk-augmented CSV path"),
    rules: Path | None = typer.Option(
        None, "--rules", help="Path to risk_rules.yaml (overrides in-memory defaults)"
    ),
) -> None:
    from fightsafe_ai.risk.engine import detect_risk_events
    from fightsafe_ai.risk.models import risk_rules_from_yaml

    feat_path = _require_file(features_csv, label="Features file")
    try:
        feat = pd.read_csv(feat_path)
    except (OSError, ValueError) as e:
        _err(f"Could not read {feat_path}: {e}")
        raise typer.Exit(1) from e
    if "near_ground" in feat.columns:
        feat = feat.copy()
        feat["near_ground"] = (
            feat["near_ground"].astype(str).str.lower().isin(("1", "true", "t", "yes"))
        )
    params = risk_rules_from_yaml(rules) if rules and rules.is_file() else None
    _ensure_output_parent(output)
    try:
        out_df = detect_risk_events(feat, params)
    except ValueError as exc:
        _err(str(exc))
        raise typer.Exit(2) from exc
    out_df.to_csv(output, index=False)
    n_flags = int(out_df["risk_flag"].sum()) if len(out_df) and "risk_flag" in out_df.columns else 0
    _ok(f"Wrote {output} ({len(out_df)} rows, {n_flags} risk-flagged frame(s)).")


@app.command(
    "build-events",
    help="Merge per-frame risk rows into time-bounded events; writes a JSON list (MVP / interpretable risk CSV).",
)
def cmd_build_events(
    risk_csv: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        help="CSV with frame_id, timestamp, risk_score, risk_level (one row per frame, time-ordered)",
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output path for the events list (JSON array of event objects)",
    ),
    min_duration: float = typer.Option(
        0.0,
        "--min-duration",
        help="Omit events whose duration in seconds is strictly less than this",
    ),
) -> None:
    from fightsafe_ai.risk.events import RiskEventExtractionConfig, frame_risk_to_events_list

    path = _require_file(risk_csv, label="Risk CSV")
    if min_duration < 0:
        _fail("--min-duration must be non-negative.", code=2)
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError) as e:
        _err(f"Could not read {path}: {e}")
        raise typer.Exit(1) from e
    cfg = RiskEventExtractionConfig(min_duration_seconds=float(min_duration))
    try:
        events = frame_risk_to_events_list(df, config=cfg)
    except ValueError as exc:
        _err(str(exc))
        raise typer.Exit(2) from exc
    out = output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(events, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _ok(f"Wrote {len(events)} event(s) -> {out}")


# --- 6. render-overlay -----------------------------------------------------------


def _run_render_overlay(
    video: Path,
    keypoints: Path,
    output: Path,
    risk: Path | None,
    rules: Path | None,
    rolling_window: int,
    config: Path | None,
) -> int:
    from fightsafe_ai.config.loader import load_yaml_file
    from fightsafe_ai.logutil import configure_cli_pipeline_logging
    from fightsafe_ai.visualization.overlay import render_risk_overlay_video

    configure_cli_pipeline_logging()
    viz: dict[str, Any] | None = None
    if config and config.is_file():
        y = load_yaml_file(config)
        viz = y.get("visualization")
    return int(
        render_risk_overlay_video(
            video,
            keypoints,
            output,
            risk_csv=risk,
            risk_rules_yaml=rules,
            rolling_window=rolling_window,
            viz_config=viz,
        )
    )


@app.command(
    "render-overlay",
    help="Render pose skeleton and risk HUD on a time-aligned source video; write MP4 (OpenCV).",
)
def cmd_render_overlay(
    video: Path = typer.Argument(
        ..., exists=True, file_okay=True, help="Source video (time-aligned with keypoint frames)"
    ),
    keypoints: Path = typer.Argument(
        ...,
        exists=True,
        help="Keypoints: directory of per-frame CSVs or one consolidated pose.csv",
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output .mp4",
    ),
    risk: Path | None = typer.Option(
        None,
        "--risk",
        help="Optional per-frame risk CSV; if omitted, risk is estimated from keypoints in-process",
    ),
    rules: Path | None = typer.Option(
        None,
        "--rules",
        help="risk_rules.yaml path (used when risk is not passed from --risk)",
    ),
    rolling_window: int = typer.Option(
        5, "--rolling-window", help="Rolling window for in-process risk"
    ),
    config: Path | None = typer.Option(
        None, "--config", help="Optional YAML with top-level 'visualization' block (styling)"
    ),
) -> None:
    _validate_rolling(rolling_window)
    v = _require_file(video, label="Video")
    if not keypoints.is_file() and not keypoints.is_dir():
        _fail(f"Keypoints not found: {keypoints}", code=2)
    _ensure_output_parent(output)
    if risk is not None and not risk.is_file():
        _fail(f"--risk file not found: {risk}", code=2)
    if rules is not None and not rules.is_file():
        _fail(f"--rules file not found: {rules}", code=2)
    if config is not None and not config.is_file():
        _fail(f"--config file not found: {config}", code=2)
    code = _run_render_overlay(
        v,
        keypoints.expanduser().resolve(),
        output.expanduser().resolve(),
        risk,
        rules,
        rolling_window,
        config,
    )
    if code != 0:
        raise typer.Exit(code)
    _ok(f"Overlay video: {output.expanduser().resolve()}")


# --- 7. explain-event (LLM) ------------------------------------------------------


@app.command(
    "explain-event",
    help="Build a Markdown explanation for one risk event (from JSON) using Ollama or a template.",
)
def cmd_explain_event(
    event_json: Path = typer.Option(
        ...,
        "--event-json",
        help="Path to a JSON file: one object (or a list, first item used) with event fields",
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        help="Output .md file path (parent directories are created as needed)",
    ),
    use_ollama: bool = typer.Option(
        True,
        "--use-ollama/--no-use-ollama",
        help="If true, call a local Ollama server; if false, use the template only",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Override the Ollama model (defaults to configs/llm.yaml or llama3.1)",
    ),
    llm_config: Path | None = typer.Option(
        None,
        "--llm-config",
        help="Path to configs/llm.yaml",
    ),
) -> None:
    from fightsafe_ai.llm.reporting import write_explanation_markdown

    path = _require_file(event_json, label="--event-json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _err(f"Invalid JSON: {e}")
        raise typer.Exit(1) from e

    if isinstance(raw, list):
        if not raw:
            _fail("JSON list is empty.")
        if len(raw) > 1:
            _err("Multiple events in the file; using the first entry only.")
        el = raw[0]
        if not isinstance(el, dict):
            _fail("Each event must be a JSON object.")
        event_data = cast("dict[str, Any]", el)
    elif isinstance(raw, dict):
        event_data = cast("dict[str, Any]", raw)
    else:
        _fail("Top-level JSON must be an object or a list of objects.")

    out = output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = write_explanation_markdown(
        out,
        event_data,
        use_ollama=use_ollama,
        ollama_model=model,
        llm_config=llm_config,
    )
    if out.suffix.lower() != ".md":
        _err("Note: a .md extension is recommended for the output path.")
    out_sz = out.stat().st_size if out.is_file() else 0
    _ok(f"Wrote {out} ({out_sz} bytes, {len(text)} chars of explanation).")


# --- 8. generate-report ----------------------------------------------------------


def _require_report_artifacts(run: Path) -> Path:
    from fightsafe_ai.reports.validate import (
        missing_report_artifacts,
        report_prereq_error_message,
    )

    root = run.expanduser().resolve()
    if not root.is_dir():
        _fail(f"Run path is not a directory: {root}", code=2)
    missing = missing_report_artifacts(root)
    if missing:
        _err(report_prereq_error_message(root, missing))
        raise typer.Exit(1)
    return root


def _maybe_regenerate_llm_explanations(
    root: Path,
    *,
    use_llm: bool,
    llm_model: str | None,
    llm_temperature: float | None,
    llm_config: Path | None,
) -> None:
    if not use_llm:
        return
    from fightsafe_ai.llm.reporting import regenerate_run_event_explanations

    log = logging.getLogger("fightsafe_ai.cli")
    n = int(
        regenerate_run_event_explanations(
            root,
            use_ollama=True,
            ollama_model=llm_model,
            ollama_temperature=llm_temperature,
            ollama_force_enabled=True,
            llm_config=llm_config,
        )
    )
    if n > 0:
        _ok(
            f"LLM: wrote or refreshed {n} event explanation file(s) under explanations/ "
            "(Ollama is best-effort; template text is used if the model is unavailable)."
        )
    else:
        log.warning(
            "LLM: no per-event explanation files were produced. "
            "Need features.csv in the run directory, or there were no merged risk events to explain."
        )


@app.command(
    "generate-report",
    help="Build report.md, report.html, and/or summary.json from a completed run directory.",
)
def cmd_generate_report(
    run: Path = typer.Option(
        ...,
        "--run",
        "-r",
        help="Pipeline run directory (must include risk_scores.csv, events.json, and related inputs)",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="If set, write a single file: 'markdown' | 'html' | 'summary' (or md/json aliases).",
    ),
    use_llm: bool = typer.Option(
        False,
        "--use-llm",
        help="Regenerate per-event explanation Markdown with Ollama (best-effort) before writing reports",
    ),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        help="Ollama model (default from configs/llm.yaml)",
    ),
    llm_temperature: float | None = typer.Option(
        None,
        "--llm-temperature",
        help="Ollama temperature (0.0–2.0; default from configs/llm.yaml)",
    ),
    llm_config: Path | None = typer.Option(
        None,
        "--llm-config",
        help="Path to configs/llm.yaml",
    ),
) -> None:
    from fightsafe_ai.reports.html import generate_html_report
    from fightsafe_ai.reports.markdown import generate_markdown_report
    from fightsafe_ai.reports.outputs import write_all_default_reports
    from fightsafe_ai.reports.summary import generate_summary_json

    _validate_llm_temperature(llm_temperature)
    root = _require_report_artifacts(run)
    _maybe_regenerate_llm_explanations(
        root,
        use_llm=use_llm,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_config=llm_config,
    )
    if not only:
        paths = list(write_all_default_reports(root))
        for p in paths:
            _ok(f"Wrote {p}")
        return

    o = re.sub(r"[^a-z]", "", only.lower())
    if o in ("md", "markdown"):
        p = generate_markdown_report(root, root / "report.md")
    elif o in ("html",):
        p = generate_html_report(root, root / "report.html")
    elif o in ("summary", "json", "js"):
        p = generate_summary_json(root, root / "summary.json")
    else:
        _fail(
            f"Invalid --only: {only!r}. Use markdown, html, or summary (or 'md' / 'json' aliases).",
            code=2,
        )
    _ok(f"Wrote {p}")


@report_app.command(
    "html",
    help="Write report.html (optional --use-llm to refresh event explanations first, like generate-report).",
)
def cmd_report_html(
    run: Path = typer.Option(
        ...,
        "--run",
        "-r",
        help="Pipeline run directory (must include risk_scores.csv, events.json, and related inputs)",
    ),
    use_llm: bool = typer.Option(
        False,
        "--use-llm",
        help="Regenerate per-event explanation Markdown with Ollama (best-effort) before report.html",
    ),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        help="Ollama model (default from configs/llm.yaml)",
    ),
    llm_temperature: float | None = typer.Option(
        None,
        "--llm-temperature",
        help="Ollama temperature (0.0–2.0; default from configs/llm.yaml)",
    ),
    llm_config: Path | None = typer.Option(
        None,
        "--llm-config",
        help="Path to configs/llm.yaml",
    ),
) -> None:
    from fightsafe_ai.reports.html import generate_html_report

    _validate_llm_temperature(llm_temperature)
    root = _require_report_artifacts(run)
    _maybe_regenerate_llm_explanations(
        root,
        use_llm=use_llm,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_config=llm_config,
    )
    p = generate_html_report(root, root / "report.html")
    _ok(f"Wrote {p}")


# --- 9. qa -----------------------------------------------------------------------


@app.command(
    "qa",
    help="Run quality checks on a pipeline run directory; writes qa_report.json by default.",
)
def cmd_qa(
    run: Path = typer.Option(
        ...,
        "--run",
        "-r",
        help="Run directory (expected MVP-style outputs)",
    ),
    require_frames: bool = typer.Option(
        True,
        "--require-frames/--no-require-frames",
        help="Require a frames/ subtree (on for a full run).",
    ),
    write_json: bool = typer.Option(
        True,
        "--write-json/--no-write-json",
        help="Write or skip <run>/qa_report.json",
    ),
) -> None:
    from fightsafe_ai.qa.validators import run_quality_checks, write_qa_report_json

    root = _require_dir(run, label="Run directory")
    try:
        report = run_quality_checks(root, require_frames=require_frames)
    except (OSError, ValueError) as exc:
        _err(str(exc))
        raise typer.Exit(1) from exc
    if write_json:
        p = write_qa_report_json(root / "qa_report.json", report)
        _ok(f"Wrote {p}")
    _print_qa_terminal_summary(report)
    if not report.passed:
        for r in report.results:
            if r.status == "fail":
                _err(f"  [fail] {r.name}: {r.message}")
        raise typer.Exit(1)


# --- 10. run-pipeline (MVP only, no QA or summary bundle) ------------------------


@app.command(
    "run-pipeline",
    help="MVP: frames → MediaPipe → features → interpretable risk → events → output_overlay.mp4 + report.md.",
)
def cmd_run_pipeline(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        exists=True,
        file_okay=True,
        help=(
            "Input video (H.264 in MP4 is most portable). Web/YouTube files are often "
            "AV1/VP9/HEVC; if frame extraction returns zero frames, re-encode to H.264 with "
            "ffmpeg (see docs/internet-video-codecs.md)."
        ),
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        file_okay=False,
        help="Run root directory (created): standard MVP artifact names under it",
    ),
    fps: int = typer.Option(10, "--fps", help="Extraction and feature sampling FPS (>= 1)"),
    rules: Path | None = typer.Option(
        None, "--rules", help="Override risk rules YAML (else project configs/ if present)"
    ),
    rolling_window: int = typer.Option(5, "--rolling-window"),
    ground_y: float = typer.Option(0.82, "--ground-y"),
    complexity: int = typer.Option(1, "--complexity", help="BlazePose: 0 / 1 / 2"),
    min_detection: float = typer.Option(0.5, "--min-detection"),
    pose_backend: str = typer.Option(
        "mediapipe",
        "--pose-backend",
        help="Pose model: mediapipe (default), mock (tests), yolo (optional ultralytics)",
    ),
    explain_events: bool = typer.Option(
        False,
        "--explain-events",
        help="Optionally write Markdown for HIGH/CRITICAL segments (Ollama or template; see --llm-config).",
    ),
    explanations_use_ollama: bool = typer.Option(
        True,
        "--explanations-ollama/--explanations-no-ollama",
        help="With --explain-events: use Ollama when on; use --explanations-no-ollama for template-only",
    ),
    ollama_explain_model: str | None = typer.Option(
        None,
        "--ollama-explain-model",
        help="Override Ollama model for per-event explanations with --explain-events",
    ),
    report_ollama: bool = typer.Option(
        False,
        "--report-ollama/--no-report-ollama",
        help="If Ollama is up, add an optional short narrative in report.md (MVP is offline by default).",
    ),
    llm_config: Path | None = typer.Option(
        None,
        "--llm-config",
        help="Path to configs/llm.yaml for optional LLM steps",
    ),
) -> None:
    from fightsafe_ai.exceptions import VideoIOError
    from fightsafe_ai.logutil import configure_cli_pipeline_logging
    from fightsafe_ai.pipeline.runner import RunPipelineConfig, run_pipeline

    if not 0.0 <= float(min_detection) <= 1.0:
        _fail("--min-detection must be in [0,1].", code=2)
    _validate_complexity(complexity)
    _validate_fps_int("FPS", fps)
    _validate_rolling(rolling_window)
    configure_cli_pipeline_logging()
    root = output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    rules_path: Path | None = rules
    if rules_path is None:
        for candidate in (Path("configs/risk_rules.yaml"), root / "risk_rules.yaml"):
            if candidate.is_file():
                rules_path = candidate
                break
    if rules is not None and not rules.is_file():
        _fail(f"--rules file not found: {rules}", code=2)

    try:
        result = run_pipeline(
            video,
            root,
            RunPipelineConfig(
                rules_yaml=rules_path if rules_path and rules_path.is_file() else None,
                pose_backend=pose_backend,
                fps=fps,
                rolling_window=rolling_window,
                ground_y=ground_y,
                model_complexity=complexity,
                min_detection=min_detection,
                explain_events=explain_events,
                explanations_use_ollama=explanations_use_ollama,
                ollama_explain_model=ollama_explain_model,
                llm_config=llm_config,
                report_ollama=report_ollama,
                include_mvp_report=True,
                include_qa=False,
                include_plots=False,
                include_report_bundle=False,
            ),
        )
        paths = result.paths
    except (VideoIOError, ValueError, OSError) as exc:
        _err(str(exc))
        raise typer.Exit(1) from exc

    _ok("")
    _ok("MVP run completed successfully.")
    _ok(f"Run root:  {paths.root}")
    _ok("Artifacts:")
    _ok(f"  • frames/             {paths.frames_dir}/")
    _ok(f"  • pose                {paths.pose_keypoints_csv}")
    _ok(f"  • features            {paths.features_csv}")
    _ok(f"  • risk_scores         {paths.risk_scores_csv}")
    _ok(f"  • events              {paths.events_json}")
    _ok(f"  • output_overlay      {paths.output_overlay_mp4}")
    _ok(f"  • report (Markdown)   {paths.report_md}")


# --- 11. demo (full e2e: MVP + QA + report bundle) -------------------------------


@app.command(
    "demo",
    help=(
        "Full end-to-end workflow: same MVP as run-pipeline, then QA, Matplotlib plots, and "
        "report.md / report.html / summary.json. Prints key paths (overlay MP4, report.html, "
        "qa_report.json, risk_timeline.png) and a QA summary. Use --use-llm for optional Ollama text."
    ),
)
def cmd_demo(
    video: Path = typer.Option(
        ...,
        "--video",
        "-v",
        exists=True,
        file_okay=True,
        help="Input video (e.g. data/clips/demo.mp4)",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        file_okay=False,
        help="Run directory (created; e.g. runs/demo/)",
    ),
    fps: int = typer.Option(10, "--fps", help="Extraction and feature sampling FPS (>= 1)"),
    rules: Path | None = typer.Option(
        None, "--rules", help="Override risk rules YAML (else project configs/ if present)"
    ),
    rolling_window: int = typer.Option(5, "--rolling-window"),
    ground_y: float = typer.Option(0.82, "--ground-y"),
    complexity: int = typer.Option(1, "--complexity", help="BlazePose: 0 / 1 / 2"),
    min_detection: float = typer.Option(0.5, "--min-detection"),
    use_llm: bool = typer.Option(
        False,
        "--use-llm",
        help="Call local Ollama for per-event explanations and an optional report narrative (best-effort)",
    ),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        help="Ollama model (default from configs/llm.yaml)",
    ),
    llm_temperature: float | None = typer.Option(
        None,
        "--llm-temperature",
        help="Ollama sampling temperature (0.0–2.0; default from configs/llm.yaml)",
    ),
    llm_config: Path | None = typer.Option(
        None,
        "--llm-config",
        help="Path to configs/llm.yaml for optional LLM steps",
    ),
) -> None:
    from fightsafe_ai.exceptions import VideoIOError
    from fightsafe_ai.logutil import configure_cli_pipeline_logging
    from fightsafe_ai.pipeline.demo import run_e2e_demo

    if not 0.0 <= float(min_detection) <= 1.0:
        _fail("--min-detection must be in [0,1].", code=2)
    _validate_complexity(complexity)
    _validate_fps_int("FPS", fps)
    _validate_rolling(rolling_window)
    _validate_llm_temperature(llm_temperature)
    v = _require_file(video, label="Video")
    if rules is not None and not rules.is_file():
        _fail(f"--rules file not found: {rules}", code=2)

    configure_cli_pipeline_logging()
    root = output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    rules_path: Path | None = rules
    if rules_path is None:
        for candidate in (Path("configs/risk_rules.yaml"), root / "risk_rules.yaml"):
            if candidate.is_file():
                rules_path = candidate
                break

    use_ollama = use_llm
    try:
        paths, qa_ok, qreport = run_e2e_demo(
            v,
            root,
            rules_yaml=rules_path if rules_path and rules_path.is_file() else None,
            fps=fps,
            rolling_window=rolling_window,
            ground_y=ground_y,
            model_complexity=complexity,
            min_detection=min_detection,
            use_ollama=use_ollama,
            ollama_explain_model=llm_model,
            ollama_explain_temperature=llm_temperature,
            ollama_force_enabled=use_llm,
            llm_config=llm_config,
        )
    except (VideoIOError, ValueError, OSError) as exc:
        _err(str(exc))
        raise typer.Exit(1) from exc

    _print_demo_completed_outputs(paths.root, heading="Demo completed.")
    _print_qa_terminal_summary(qreport)

    if use_llm:
        _ok("")
        _ok(
            "Ollama: per-event text and an optional report narrative are attempted; "
            "if the server is unavailable, the run continues with template text (no hard failure)."
        )
    else:
        _ok("")
        _ok(
            "LLM: off for this run. Pass --use-llm to call Ollama for human-readable event explanations."
        )

    if not qa_ok:
        _ok("")
        _err("Not all QA checks passed — see qa_report.json for details.")
        raise typer.Exit(1)


@app.command(
    "demo-youtube",
    help=(
        "Download a video URL (yt-dlp), cut [start, end) with FFmpeg, then run the same "
        "end-to-end workflow as `fightsafe demo` (MVP + QA + report bundle). "
        "Requires yt-dlp and ffmpeg on PATH."
    ),
)
def cmd_demo_youtube(
    url: str = typer.Option(
        ...,
        "--url",
        "-u",
        help="Video or stream page URL (e.g. YouTube watch URL)",
    ),
    start: str = typer.Option(
        ...,
        "--start",
        "-s",
        help="Segment start (seconds or MM:SS), same as `fightsafe cut`",
    ),
    end: str = typer.Option(
        ...,
        "--end",
        "-e",
        help="Segment end (seconds or MM:SS, exclusive), same as `fightsafe cut`",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        file_okay=False,
        help="Run directory (created; same role as `fightsafe demo -o`)",
    ),
    download_name: str | None = typer.Option(
        None,
        "--download-name",
        help="Optional file name for the full download under source/; .mp4 added if missing",
    ),
    fps: int = typer.Option(10, "--fps", help="Extraction and feature sampling FPS (>= 1)"),
    rules: Path | None = typer.Option(
        None, "--rules", help="Override risk rules YAML (else project configs/ if present)"
    ),
    rolling_window: int = typer.Option(5, "--rolling-window"),
    ground_y: float = typer.Option(0.82, "--ground-y"),
    complexity: int = typer.Option(1, "--complexity", help="BlazePose: 0 / 1 / 2"),
    min_detection: float = typer.Option(0.5, "--min-detection"),
    use_llm: bool = typer.Option(
        False,
        "--use-llm",
        help="Call local Ollama for per-event explanations and an optional report narrative (best-effort)",
    ),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        help="Ollama model (default from configs/llm.yaml)",
    ),
    llm_temperature: float | None = typer.Option(
        None,
        "--llm-temperature",
        help="Ollama sampling temperature (0.0–2.0; default from configs/llm.yaml)",
    ),
    llm_config: Path | None = typer.Option(
        None,
        "--llm-config",
        help="Path to configs/llm.yaml for optional LLM steps",
    ),
) -> None:
    from fightsafe_ai.exceptions import VideoCutError, VideoDownloadError, VideoIOError
    from fightsafe_ai.logutil import configure_cli_pipeline_logging
    from fightsafe_ai.pipeline.youtube_demo import (
        DEMO_YOUTUBE_INPUT_CLIP,
        run_demo_youtube,
    )

    if not 0.0 <= float(min_detection) <= 1.0:
        _fail("--min-detection must be in [0,1].", code=2)
    _validate_complexity(complexity)
    _validate_fps_int("FPS", fps)
    _validate_rolling(rolling_window)
    _validate_llm_temperature(llm_temperature)
    s0, t0 = (x.strip() for x in (start, end))
    if not s0 or not t0:
        _fail("Both --start and --end must be non-empty.", code=2)
    if rules is not None and not rules.is_file():
        _fail(f"--rules file not found: {rules}", code=2)

    u = _require_nonempty_url(url)
    root = output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    dl_name: str | None = download_name
    if dl_name is not None and not str(dl_name).strip():
        dl_name = None

    configure_cli_pipeline_logging()
    rules_path: Path | None = rules
    if rules_path is None:
        for candidate in (Path("configs/risk_rules.yaml"), root / "risk_rules.yaml"):
            if candidate.is_file():
                rules_path = candidate
                break

    use_ollama = use_llm
    try:
        paths, qa_ok, qreport = run_demo_youtube(
            u,
            s0,
            t0,
            root,
            download_filename=dl_name,
            rules_yaml=rules_path if rules_path and rules_path.is_file() else None,
            fps=fps,
            rolling_window=rolling_window,
            ground_y=ground_y,
            model_complexity=complexity,
            min_detection=min_detection,
            use_ollama=use_ollama,
            ollama_explain_model=llm_model,
            ollama_explain_temperature=llm_temperature,
            ollama_force_enabled=use_llm,
            llm_config=llm_config,
        )
    except VideoDownloadError as exc:
        _err("Download step failed. Fix the URL, network, or install yt-dlp; see log above.")
        _err(str(exc))
        raise typer.Exit(1) from exc
    except VideoCutError as exc:
        _err("Cut step failed. Check --start / --end against the full download duration.")
        _err(str(exc))
        raise typer.Exit(1) from exc
    except (VideoIOError, ValueError, OSError) as exc:
        _err(str(exc))
        raise typer.Exit(1) from exc

    _print_demo_completed_outputs(paths.root, heading="YouTube demo completed.")
    _ok(f"Clipped input video: {_display_path(paths.root / DEMO_YOUTUBE_INPUT_CLIP)}")
    _print_qa_terminal_summary(qreport)

    if use_llm:
        _ok("")
        _ok(
            "Ollama: per-event text and an optional report narrative are attempted; "
            "if the server is unavailable, the run continues with template text (no hard failure)."
        )
    else:
        _ok("")
        _ok(
            "LLM: off for this run. Pass --use-llm to call Ollama for human-readable event explanations."
        )

    if not qa_ok:
        _ok("")
        _err("Not all QA checks passed — see qa_report.json for details.")
        raise typer.Exit(1)


# --- 12. Manual evaluation annotations (ground truth) -----------------------------


@app.command(
    "annotate-template",
    help="Write an empty manual ground-truth JSON (edit in an editor, then validate).",
)
def cmd_annotate_template(
    video: Path = typer.Option(
        ...,
        "--video",
        help="Path or id of the video this file will describe (stored as a string; need not exist yet).",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path, e.g. annotations/demo_annotations.json",
    ),
) -> None:
    from fightsafe_ai.annotation.loader import new_empty_template, save_annotation_file

    p = video.expanduser()
    ref = str(p).replace("\\", "/")
    doc = new_empty_template(video=ref)
    out = output.expanduser().resolve()
    _ensure_output_parent(out)
    save_annotation_file(out, doc)
    _ok(f"Wrote empty annotation template: {out}")
    _ok(
        "Next: add objects to `events` (start_time, end_time, event_type, optional confidence, notes)."
    )
    _ok("See docs/annotation.md. Validate with: fightsafe validate-annotations <file>")


@app.command(
    "validate-annotations",
    help="Validate a manual annotation JSON (schema, times, event types).",
)
def cmd_validate_annotations(
    path: Path = typer.Argument(
        ...,
        help="Path to an annotation file (e.g. annotations/demo_annotations.json)",
    ),
) -> None:
    from fightsafe_ai.annotation.loader import load_annotation_file
    from fightsafe_ai.annotation.validator import format_overlap_warnings, validate_annotation_file

    p = path.expanduser().resolve()
    errs = validate_annotation_file(p)
    if errs:
        for line in errs:
            _err(line)
        raise typer.Exit(1)
    doc = load_annotation_file(p)
    for w in format_overlap_warnings(doc):
        _ok(f"Note: {w}")
    _ok("OK: annotation file is valid.")


@app.command(
    "suggest-annotations",
    help="Draft event labels (Ollama) for a pipeline run; requires human confirm — not ground truth.",
)
def cmd_suggest_annotations(
    run: Path = typer.Option(
        ...,
        "--run",
        "-r",
        help="Pipeline run directory (expects events.json; risk_scores.csv and frames/ optional)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON (default: <run>/annotation_suggestions.json)",
    ),
    use_ollama: bool = typer.Option(
        False,
        "--use-ollama",
        help="Call local Ollama to propose draft labels (see configs/llm.yaml: ollama.enabled).",
    ),
    use_vlm: bool = typer.Option(
        False,
        "--use-vlm",
        help="Include optional VLM text per event (configs/llm.yaml: enable_vlm_review) for the text prompt.",
    ),
    max_frames: int = typer.Option(
        3,
        "--max-frames",
        help="Up to this many on-disk frame paths per event for VLM and bundle metadata.",
    ),
    llm_config: Path | None = typer.Option(
        None,
        "--llm-config",
        help="Override path to llm.yaml (default: <repo>/configs/llm.yaml)",
    ),
) -> None:
    from fightsafe_ai.annotation.llm_assist import run_pipeline_suggest_annotations

    root = _require_dir(run, label="--run")
    if not (root / "events.json").is_file():
        _fail(f"Missing events.json in run directory: {root}", code=2)
    out = output.expanduser().resolve() if output else (root / "annotation_suggestions.json")
    if not use_ollama and not use_vlm:
        _ok(
            "No --use-ollama: writing a shell file with run context only. "
            "Re-run with --use-ollama to generate draft labels (requires ollama.enabled in configs/llm.yaml)."
        )
    elif use_vlm and not use_ollama:
        _ok(
            "Including optional VLM text in the output file; text-model suggestions need --use-ollama."
        )
    doc = run_pipeline_suggest_annotations(
        root,
        out,
        use_ollama=use_ollama,
        use_vlm=use_vlm,
        max_frames_per_event=max_frames,
        llm_config=llm_config.expanduser().resolve() if llm_config else None,
    )
    n = len(doc.get("suggestions", []))
    _ok(f"Wrote {out} ({n} draft suggestion(s); not ground truth — see disclaimer in file).")


@app.command(
    "run-case-studies",
    help="Download each URL in a case-study YAML, run the full pipeline (illustrative, not a benchmark).",
)
def cmd_run_case_studies(
    config: Path = typer.Option(
        Path("configs/case_studies.yaml"),
        "--config",
        "-c",
        help="Path to case_studies.yaml",
    ),
    rules: Path | None = typer.Option(
        None,
        "--rules",
        help="Optional risk_rules.yaml (else default from configs/)",
    ),
    fps: int = typer.Option(10, "--fps", help="Extraction and feature sampling FPS (>= 1)"),
) -> None:
    from fightsafe_ai.case_studies.runner import run_case_studies_from_config

    yml = _require_file(config, label="--config")
    rules_p = _require_file(rules, label="--rules") if rules is not None else None
    _validate_fps_int("FPS", fps)
    try:
        out = run_case_studies_from_config(yml, rules_yaml=rules_p, fps=fps)
    except (OSError, ValueError) as exc:
        _fail(str(exc), code=2)
    n_ok = sum(1 for x in out if x.get("ok"))
    _ok(f"Case studies finished: {n_ok}/{len(out)} run(s) passed QA. See per-case output_dir.")
    for row in out:
        s = f"  • {row.get('case_id')}: {row.get('output_dir', '')} ok={row.get('ok')}"
        if row.get("error"):
            s += f"  ({row['error']})"
        _ok(s)


@app.command(
    "evaluate",
    help="Match predicted events (events.json) to ground truth (annotation JSON) and print metrics.",
)
def cmd_evaluate(
    predicted: Path = typer.Option(
        ...,
        "--predicted",
        help="Path to a pipeline `events.json` (array of time-bounded events).",
    ),
    ground_truth: Path = typer.Option(
        ...,
        "--ground-truth",
        help="Path to a manual annotation file, e.g. `annotations/demo_annotations.json`",
    ),
    output: Path = typer.Option(
        Path("metrics.json"),
        "--output",
        "-o",
        help="Write a JSON report (default: ./metrics.json).",
    ),
    iou_threshold: float = typer.Option(
        0.1,
        "--iou-threshold",
        help="Minimum temporal IoU to accept a (pred, ref) pair (0-1).",
    ),
    tolerance_seconds: float = typer.Option(
        0.0,
        "--tolerance-seconds",
        help="Dilate each interval by tolerance_s/2 on each side for overlap matching.",
    ),
    require_same_label: bool = typer.Option(
        False,
        "--require-same-label",
        help="Only match events with the same label (pipeline `event_level` vs GT `event_type` often differ).",
    ),
) -> None:
    from fightsafe_ai.evaluation.event_matching import (
        annotation_file_to_ground_truth_windows,
        events_json_to_windows,
    )
    from fightsafe_ai.evaluation.metrics import (
        evaluate_event_prediction,
        event_evaluation_to_json_dict,
    )

    p_pred = _require_file(predicted, label="--predicted")
    p_ref = _require_file(ground_truth, label="--ground-truth")
    pred = events_json_to_windows(p_pred)
    try:
        ref = annotation_file_to_ground_truth_windows(p_ref)
    except (OSError, ValueError, TypeError) as e:
        _fail(str(e))
    r = evaluate_event_prediction(
        pred,
        ref,
        iou_threshold=iou_threshold,
        tolerance_seconds=tolerance_seconds,
        require_same_label=require_same_label,
    )
    payload = event_evaluation_to_json_dict(r)
    payload["source_paths"] = {
        "predicted": str(p_pred),
        "ground_truth": str(p_ref),
    }
    out = output.expanduser().resolve()
    _ensure_output_parent(out)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _ok("--- Event-level evaluation (FightSafe) ---")
    _ok(
        f"  Predicted: {r.n_predicted:>4d}  |  "
        f"Ground truth: {r.n_ground_truth:>4d}  |  "
        f"TP={r.true_positives:>4d}  FP={r.false_positives:>4d}  FN={r.false_negatives:>4d}"
    )
    _ok(
        f"  Precision {r.precision:6.3f}   Recall {r.recall:6.3f}   F1 {r.f1:6.3f}   "
        f"(IoU min={iou_threshold}, tolerance_s={tolerance_seconds}, same_label={require_same_label})"
    )
    _ok(
        f"  Onset delay (mean pred−ref) {r.mean_onset_delay_seconds:+.3f} s  |  "
        f"mean |delay| {r.mean_abs_onset_delay_seconds:.3f} s"
    )
    _ok(f"  Wrote: {out}")


@app.command(
    "evaluate-case-studies",
    help="Compare each illustrative case-study run's events.json to annotations/<stem>.json; "
    "write CSV + TeX summary.",
)
def cmd_evaluate_case_studies(
    runs_dir: Path = typer.Option(
        Path("runs/case_studies"),
        "--runs-dir",
        help="Parent directory containing per-case output folders (e.g. cs_knockdown_001/)",
    ),
    annotations_dir: Path = typer.Option(
        Path("annotations"),
        "--annotations-dir",
        help="Directory with case_a_knockdown.json, … (same stem keys as config narrative ids)",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Output directory for evaluation_all_cases.csv and evaluation_all_cases.tex",
    ),
    write_paper_tex: bool = typer.Option(
        True,
        "--write-paper-tex/--no-write-paper-tex",
        help="Also write the TeX fragment for the fusion manuscript (see --paper-tex-path).",
    ),
    paper_tex_path: Path = typer.Option(
        Path("../fusion2026/tables/evaluation_all_cases.tex"),
        "--paper-tex-path",
        help="LaTeX table path when --write-paper-tex is set.",
    ),
    iou_threshold: float = typer.Option(
        0.1,
        "--iou-threshold",
        help="Minimum temporal IoU for a match (same as fightsafe evaluate).",
    ),
    tolerance_seconds: float = typer.Option(
        0.0,
        "--tolerance-seconds",
        help="Symmetric dilation before overlap test.",
    ),
    require_same_label: bool = typer.Option(
        False,
        "--require-same-label",
        help="Require matching event_type / label (often leave off for case studies).",
    ),
) -> None:
    from fightsafe_ai.evaluation.case_study_evaluation import (
        run_case_study_batch_evaluation,
        write_case_study_evaluation_csv,
        write_case_study_evaluation_tex,
    )

    runs = _require_dir(runs_dir, label="--runs-dir")
    ann = _require_dir(annotations_dir, label="--annotations-dir")
    out_root = output.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = run_case_study_batch_evaluation(
        runs_dir=runs,
        annotations_dir=ann,
        iou_threshold=iou_threshold,
        tolerance_seconds=tolerance_seconds,
        require_same_label=require_same_label,
    )
    csv_path = out_root / "evaluation_all_cases.csv"
    tex_path = out_root / "evaluation_all_cases.tex"
    write_case_study_evaluation_csv(csv_path, rows)
    write_case_study_evaluation_tex(tex_path, rows)

    if write_paper_tex:
        p_tex = paper_tex_path.expanduser().resolve()
        _ensure_output_parent(p_tex)
        write_case_study_evaluation_tex(p_tex, rows)
        _ok(f"  Also wrote paper TeX: {p_tex}")

    _ok("--- Case-study event evaluation (batch) ---")
    for r in rows:
        _ok(
            f"  {r.case_id}: {r.status}  |  pred={r.predicted_events!s}  ann={r.annotated_events!s}"
        )
    _ok(f"  Wrote: {csv_path}")
    _ok(f"  Wrote: {tex_path}")


# --- 14. plot-risk (optional) ----------------------------------------------------


@app.command(
    "plot-risk",
    help="Write risk_timeline.png and events_timeline.png under the run (matplotlib).",
)
def cmd_plot_risk(
    run: Path = typer.Option(
        ...,
        "--run",
        "-r",
        help="Pipeline run directory (must include risk_scores.csv, events.json)",
    ),
) -> None:
    from fightsafe_ai.visualization.plots import (
        plot_events_timeline,
        plot_risk_timeline,
    )

    root = _require_report_artifacts(run)
    p1 = plot_risk_timeline(root, root / "risk_timeline.png")
    p2 = plot_events_timeline(root, root / "events_timeline.png")
    for p in (p1, p2):
        _ok(f"Wrote {p}")


# --- risk-ablation (formal fusion behavior metrics; not semantic validation) ----------


@app.command(
    "risk-ablation",
    help=(
        "Run interpretable multi-signal fusion ablations on features.csv; writes CSV/TeX/timeline "
        "(behavior metrics only, not accuracy)."
    ),
)
def cmd_risk_ablation(
    run: Path = typer.Option(
        ...,
        "--run",
        "-r",
        help="Run directory (must contain features.csv) or path to features.csv",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Directory for ablation_results.csv, TeX, PNG, and per-mode risk_series CSVs",
    ),
    fusion_config: Path | None = typer.Option(
        None,
        "--fusion-config",
        help="YAML path (default: configs/risk_fusion.yaml next to project root)",
    ),
    rules_config: Path | None = typer.Option(
        None,
        "--rules-config",
        help="MVP interpretable rules YAML (default: configs/risk_rules.yaml)",
    ),
    fps: float | None = typer.Option(
        None,
        "--fps",
        help="Override frame rate for timestamps and event density (else qa_report.json or 10)",
    ),
    mode: list[str] | None = typer.Option(
        None,
        "--mode",
        help="Ablation mode (repeatable). Default: all built-in modes.",
    ),
) -> None:
    from fightsafe_ai.evaluation.risk_ablation import ABLATION_MODES, run_risk_ablation

    root = run.expanduser().resolve()
    if not root.exists():
        _fail(f"Path not found: {root}")
    out = output.expanduser().resolve()
    modes = list(mode) if mode else None
    if modes:
        for m in modes:
            if m not in ABLATION_MODES:
                _fail(f"Unknown ablation mode {m!r}. Choose from: {', '.join(ABLATION_MODES)}")
    p = run_risk_ablation(
        root,
        out,
        configs=modes,
        fusion_yaml=fusion_config,
        rules_yaml=rules_config,
        fps=fps,
    )
    _ok("--- Formal risk fusion ablation (behavior metrics) ---")
    _ok(f"  Wrote directory: {p}")


@app.command(
    "risk-ablation-all",
    help="Run risk-ablation for each subdirectory of --base-dir that contains features.csv.",
)
def cmd_risk_ablation_all(
    base_dir: Path = typer.Option(
        ...,
        "--base-dir",
        help="Parent directory (e.g. runs/case_studies)",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Summary output root (e.g. runs/case_studies/ablation_summary/)",
    ),
    fusion_config: Path | None = typer.Option(
        None,
        "--fusion-config",
        help="YAML for formal fusion weights/thresholds (default: packaged risk_fusion.yaml)",
    ),
    rules_config: Path | None = typer.Option(
        None,
        "--rules-config",
        help="MVP risk_rules.yaml",
    ),
    fps: float | None = typer.Option(
        None,
        "--fps",
        help="Override FPS for all runs",
    ),
    mode: list[str] | None = typer.Option(
        None,
        "--mode",
        help="Ablation mode (repeatable). Default: all built-in modes.",
    ),
) -> None:
    from fightsafe_ai.evaluation.risk_ablation import ABLATION_MODES, run_risk_ablation_all

    base = base_dir.expanduser().resolve()
    if not base.is_dir():
        _fail(f"Not a directory: {base}")
    out = output.expanduser().resolve()
    modes = list(mode) if mode else None
    if modes:
        for m in modes:
            if m not in ABLATION_MODES:
                _fail(f"Unknown ablation mode {m!r}. Choose from: {', '.join(ABLATION_MODES)}")
    p = run_risk_ablation_all(
        base,
        out,
        fusion_yaml=fusion_config,
        rules_yaml=rules_config,
        fps=fps,
        configs=modes,
    )
    _ok("--- Risk ablation (all runs with features.csv) ---")
    _ok(f"  Summary root: {p}")


# --- TapKO (submission_signal.* / extreme_vulnerability.*) --------------------------------


@app.command(
    "tapko-detect",
    help=(
        "Sample frames from a video, estimate pose (unless --pose-csv), run tap_detector and "
        "vulnerability_detector; write tapko_predictions.json, tapko_predictions.csv, "
        "tapko_report.md, and tapko_manifest.json under --output-dir. No PostgreSQL."
    ),
)
def cmd_tapko_detect(
    source: Path = typer.Option(
        ...,
        "--source",
        "-s",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Input video file.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        file_okay=False,
        help="Run directory (created): receives predictions, report, and extracted pose artifacts.",
    ),
    fps: float = typer.Option(
        ...,
        "--fps",
        help="Frame sampling rate for extraction / detectors (Hz).",
    ),
    pose_csv: Path | None = typer.Option(
        None,
        "--pose-csv",
        help=(
            "Optional consolidated pose_keypoints.csv or per-frame CSV directory (skip video pose)."
        ),
    ),
    video_id: str | None = typer.Option(
        None,
        "--video-id",
        help="Stable id for prediction rows (default: video file stem).",
    ),
    pose_backend: str = typer.Option(
        "mediapipe",
        "--pose-backend",
        help="Pose backend when extracting from video (mediapipe, mock, yolo, rtmpose, …).",
    ),
    complexity: int = typer.Option(1, "--complexity", help="MediaPipe model complexity 0/1/2."),
    min_detection: float = typer.Option(
        0.5, "--min-detection", help="Min detection confidence [0,1] for pose."
    ),
) -> None:
    from fightsafe_ai.tapko.detect_run import run_tapko_detect

    _validate_fps("FPS", float(fps))
    if not 0.0 <= float(min_detection) <= 1.0:
        _fail("--min-detection must be in [0,1].", code=2)
    _validate_complexity(complexity)
    src = _require_file(source, label="--source")
    out = output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    pose_opt = pose_csv.expanduser().resolve() if pose_csv is not None else None
    if pose_opt is not None and not pose_opt.exists():
        _fail(f"--pose-csv not found: {pose_opt}")
    try:
        paths = run_tapko_detect(
            source_video=src,
            output_dir=out,
            fps=float(fps),
            video_id=(video_id.strip() if video_id and video_id.strip() else None),
            pose_csv=pose_opt,
            pose_backend=str(pose_backend),
            model_complexity=int(complexity),
            min_detection=float(min_detection),
        )
    except Exception as exc:
        _err(str(exc))
        raise typer.Exit(1) from exc
    _ok("TapKO detection finished.")
    _ok(f"  Predictions (JSON): {paths.predictions_json}")
    _ok(f"  Predictions (CSV):  {paths.predictions_csv}")
    _ok(f"  Report (Markdown): {paths.report_md}")
    _ok(f"  Manifest:           {paths.manifest_json}")
    _ok(f"  Pose CSV used:      {paths.pose_csv}")


@app.command(
    "tapko-evaluate",
    help=(
        "Compare TapKO prediction intervals (tapko_predictions.json) to manual annotations; "
        "writes tapko_results.csv, tapko_results.tex, tapko_error_analysis.md in --output-dir."
    ),
)
def cmd_tapko_evaluate(
    annotations: Path = typer.Option(
        ...,
        "--annotations",
        exists=True,
        file_okay=True,
        help="Ground-truth TapKO annotation JSON (schema: fightsafe_ai.tapko_annotation).",
    ),
    predictions: Path = typer.Option(
        ...,
        "--predictions",
        exists=True,
        file_okay=True,
        help="Predictions JSON: array of {video_id, start_time, end_time, event_type, …}.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        file_okay=False,
        help="Directory for tapko_results.csv, .tex, and tapko_error_analysis.md.",
    ),
    iou_threshold: float = typer.Option(
        0.3,
        "--iou-threshold",
        help="Temporal IoU threshold for matching.",
    ),
    tolerance_seconds: float = typer.Option(
        0.0,
        "--tolerance-seconds",
        help="Symmetric interval dilation before IoU (seconds).",
    ),
    match_mode: str = typer.Option(
        "exact",
        "--match-mode",
        help="exact | family (namespace-level agreement).",
    ),
) -> None:
    from fightsafe_ai.evaluation.tapko_evaluator import (
        TapkoEvalConfig,
        run_tapko_evaluation_and_write,
    )

    ann = _require_file(annotations, label="--annotations")
    pred = _require_file(predictions, label="--predictions")
    out = output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    mm = str(match_mode).strip().lower()
    if mm not in ("exact", "family"):
        _fail("--match-mode must be 'exact' or 'family'.", code=2)
    if not 0.0 < float(iou_threshold) <= 1.0:
        _fail("--iou-threshold must be in (0, 1].", code=2)
    cfg = TapkoEvalConfig(
        iou_threshold=float(iou_threshold),
        tolerance_seconds=float(tolerance_seconds),
        match_mode=("exact" if mm == "exact" else "family"),
    )
    try:
        res = run_tapko_evaluation_and_write(
            ann,
            pred,
            out,
            config=cfg,
        )
    except Exception as exc:
        _err(str(exc))
        raise typer.Exit(1) from exc
    _ok("TapKO evaluation finished.")
    _ok(
        f"  TP={res.tp} FP={res.fp} FN={res.fn}  P={res.precision:.4f} R={res.recall:.4f} F1={res.f1:.4f}"
    )
    _ok(f"  Output directory: {out}")


@app.command(
    "tapko-validate-annotations",
    help="Validate a TapKO annotation JSON file against the Pydantic schema.",
)
def cmd_tapko_validate_annotations(
    annotations: Path = typer.Option(
        ...,
        "--annotations",
        exists=True,
        file_okay=True,
        help="TapKO annotation JSON path.",
    ),
) -> None:
    from fightsafe_ai.annotation.tapko_schema import parse_tapko_json

    p = _require_file(annotations, label="--annotations")
    try:
        doc = parse_tapko_json(p.read_text(encoding="utf-8"))
    except Exception as exc:
        _err(str(exc))
        raise typer.Exit(1) from exc
    _ok(f"OK: {len(doc.annotations)} annotation interval(s) in {p}")


@app.command(
    "tapko-export-examples",
    help="Write example TapKO annotation JSON files (minimal + full) into --output-dir.",
)
def cmd_tapko_export_examples(
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        file_okay=False,
        help="Directory to create/receive example JSON files.",
    ),
) -> None:
    from fightsafe_ai.annotation.tapko_schema import EXAMPLE_DOCUMENT_FULL, EXAMPLE_DOCUMENT_MINIMAL

    out = output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "tapko_example_minimal.json").write_text(
        json.dumps(EXAMPLE_DOCUMENT_MINIMAL, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "tapko_example_full.json").write_text(
        json.dumps(EXAMPLE_DOCUMENT_FULL, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _ok(f"Wrote {out / 'tapko_example_minimal.json'}")
    _ok(f"Wrote {out / 'tapko_example_full.json'}")


app.add_typer(report_app, name="report")


def main() -> None:
    """Console entry point: ``fightsafe`` (see :attr:`app`)."""
    app()


if __name__ == "__main__":
    app()
