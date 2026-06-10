"""
Render demo videos with optional pose overlay, risk HUD, and elevated-risk warning UI.

Uses OpenCV for drawing and video I/O. Pose wiring follows MediaPipe landmark names
(see :mod:`fightsafe_ai.keypoints.io`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from fightsafe_ai.exceptions import VideoIOError
from fightsafe_ai.features.biomechanics import compute_pose_features
from fightsafe_ai.keypoints.io import load_indexed_sequence
from fightsafe_ai.risk.engine import detect_risk_events
from fightsafe_ai.risk.models import RiskRuleParams, risk_rules_from_yaml


logger = logging.getLogger(__name__)

_pose_connection_pairs: list[tuple[int, int]] | None = None


def _pose_connection_pairs_list() -> list[tuple[int, int]]:
    """BlazePose skeleton edges (index pairs), from MediaPipe Tasks ``PoseLandmarksConnections``."""
    global _pose_connection_pairs
    if _pose_connection_pairs is None:
        from mediapipe.tasks.python.vision.pose_landmarker import (
            PoseLandmarksConnections,
        )

        _pose_connection_pairs = [(c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS]
    return _pose_connection_pairs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OverlayVizConfig:
    """Visual defaults for :func:`render_risk_overlay` and :func:`render_risk_overlay_video`."""

    skeleton_line_bgr: tuple[int, int, int] = (80, 220, 80)
    skeleton_joint_bgr: tuple[int, int, int] = (240, 240, 240)
    skeleton_line_thickness: int = 3
    skeleton_joint_radius: int = 4
    skeleton_vis_min: float = 0.25

    hud_margin_x: int = 20
    hud_margin_y: int = 40
    hud_score_scale: float = 0.75
    hud_level_scale: float = 0.7
    hud_text_thickness: int = 2

    # When ``risk_flag`` but no HIGH/CRITICAL banner (e.g. MEDIUM/LOW)
    risk_flag_tint_strength: float = 0.16
    risk_flag_border_bgr: tuple[int, int, int] = (0, 0, 255)
    risk_flag_border_thickness: int = 8

    warning_banner_height_ratio: float = 0.085
    warning_banner_bgr: tuple[int, int, int] = (0, 0, 210)
    warning_banner_text_bgr: tuple[int, int, int] = (255, 255, 255)
    warning_banner_text_scale: float = 0.85
    warning_tint_strength: float = 0.08
    warning_messages: Mapping[str, str] = field(
        default_factory=lambda: {
            "CRITICAL": "CRITICAL RISK",
            "HIGH": "HIGH RISK",
        }
    )


# ---------------------------------------------------------------------------
# I/O: pose + risk
# ---------------------------------------------------------------------------


def load_pose_indexed_sequence(
    pose_path: Path, *, glob_pattern: str = "*.csv"
) -> list[dict[int, tuple[float, float, float]]]:
    """
    Load per-frame MediaPipe-indexed landmarks. Returns ``[]`` if the path is missing
    or unreadable (skeleton skipped; a warning is logged).
    """
    pose_path = pose_path.expanduser().resolve()
    if not pose_path.is_file() and not pose_path.is_dir():
        logger.warning("Pose path not found, skipping skeleton: %s", pose_path)
        return []
    try:
        return load_indexed_sequence(pose_path, glob_pattern=glob_pattern)
    except Exception as exc:
        logger.warning("Could not load pose from %s: %s", pose_path, exc)
        return []


def _parse_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(("1", "true", "t", "yes"))


def read_risk_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise VideoIOError(f"Risk CSV not found: {path}")
    df = pd.read_csv(path)
    if "near_ground" in df.columns:
        df = df.copy()
        df["near_ground"] = _parse_bool_series(df["near_ground"])
    if "risk_flag" in df.columns:
        df = df.copy()
        df["risk_flag"] = _parse_bool_series(df["risk_flag"])
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index", kind="mergesort").reset_index(drop=True)
    return df


def risk_values_for_frame(
    risk_df: pd.DataFrame,
    frame_idx: int,
) -> tuple[float, str, str | None, bool]:
    """
    Return ``(risk_score, level_display, banner_tier, risk_flag)``.

    * ``banner_tier`` is ``"HIGH"``, ``"CRITICAL"``, or ``None``. The red top bar
      is shown for explicit HIGH/CRITICAL, or for **risk_flag**-only data (no ``risk_level``).
    """
    row: pd.Series | None = None
    if "frame_index" in risk_df.columns:
        m = risk_df["frame_index"] == frame_idx
        if m.any():
            row = risk_df.loc[m].iloc[0]
    if row is None and frame_idx < len(risk_df):
        row = risk_df.iloc[frame_idx]
    if row is None or len(risk_df) == 0:
        return (float("nan"), "—", None, False)

    score = float("nan")
    if "risk_score" in row.index and pd.notna(row.get("risk_score")):
        try:
            score = float(row["risk_score"])
        except (TypeError, ValueError):
            pass

    has_level = "risk_level" in risk_df.columns
    flag = (
        bool(row["risk_flag"])
        if "risk_flag" in risk_df.columns and pd.notna(row.get("risk_flag"))
        else False
    )
    banner: str | None = None
    level_display = "—"

    if has_level:
        if pd.notna(row.get("risk_level")):
            level_display = str(row["risk_level"]).strip()
            lu = level_display.upper()
            if lu == "CRITICAL":
                banner = "CRITICAL"
            elif lu == "HIGH":
                banner = "HIGH"
    elif flag:
        banner = "HIGH"
        level_display = "ALERT"

    return (score, level_display, banner, flag)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_text_outlined(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float = 0.75,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (16, 16, 20),
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, font_scale, bg, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, fg, thickness, cv2.LINE_AA)


def draw_skeleton_bgr(
    frame: np.ndarray,
    landmarks: dict[int, tuple[float, float, float]],
    width: int,
    height: int,
    cfg: OverlayVizConfig,
) -> None:
    def pt_xy(idx: int) -> tuple[int, int] | None:
        if idx not in landmarks:
            return None
        xn, yn, vis = landmarks[idx]
        if vis < cfg.skeleton_vis_min:
            return None
        return int(xn * width), int(yn * height)

    for ia, ib in _pose_connection_pairs_list():
        pa, pb = pt_xy(ia), pt_xy(ib)
        if pa is None or pb is None:
            continue
        cv2.line(
            frame,
            pa,
            pb,
            cfg.skeleton_line_bgr,
            cfg.skeleton_line_thickness,
            lineType=cv2.LINE_AA,
        )
    for idx in landmarks:
        p = pt_xy(idx)
        if p is None:
            continue
        cv2.circle(
            frame, p, cfg.skeleton_joint_radius, cfg.skeleton_joint_bgr, -1, lineType=cv2.LINE_AA
        )
        cv2.circle(frame, p, cfg.skeleton_joint_radius, (32, 32, 32), 1, lineType=cv2.LINE_AA)


def _apply_tint(frame: np.ndarray, bgr: tuple[int, int, int], strength: float) -> None:
    overlay = frame.copy()
    overlay[:] = bgr
    cv2.addWeighted(overlay, strength, frame, 1.0 - strength, 0, dst=frame)


def _draw_risk_flag_accent(frame: np.ndarray, cfg: OverlayVizConfig) -> None:
    _apply_tint(frame, (55, 55, 220), cfg.risk_flag_tint_strength)
    h, w = frame.shape[:2]
    cv2.rectangle(
        frame,
        (0, 0),
        (w - 1, h - 1),
        cfg.risk_flag_border_bgr,
        cfg.risk_flag_border_thickness,
    )


def draw_elevated_risk_banner(
    frame: np.ndarray,
    tier: str,
    cfg: OverlayVizConfig,
) -> int:
    """
    Red top bar; return y offset below the banner for stacking the HUD.
    """
    h, w = frame.shape[:2]
    bh = max(32, int(h * cfg.warning_banner_height_ratio))
    sub = frame[0:bh, :]
    color = np.array(cfg.warning_banner_bgr, dtype=np.float32)
    blended = sub.astype(np.float32) * 0.4 + color * 0.6
    sub[:, :] = np.clip(blended, 0, 255).astype(np.uint8)
    label = cfg.warning_messages.get(tier, "ELEVATED RISK")
    font = cv2.FONT_HERSHEY_SIMPLEX
    th = 2
    (tw, th0), _ = cv2.getTextSize(label, font, cfg.warning_banner_text_scale, th)
    x = (w - tw) // 2
    y = (bh + th0) // 2
    cv2.putText(
        frame, label, (x, y), font, cfg.warning_banner_text_scale, (8, 8, 8), 4, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        label,
        (x, y),
        font,
        cfg.warning_banner_text_scale,
        cfg.warning_banner_text_bgr,
        2,
        cv2.LINE_AA,
    )
    return bh + 8


def draw_risk_hud(
    frame: np.ndarray,
    risk_score: float,
    level_display: str,
    top_offset: int,
    cfg: OverlayVizConfig,
) -> None:
    y0 = top_offset
    line1 = f"Risk: {risk_score:.2f}" if np.isfinite(risk_score) else "Risk: —"
    line2 = f"Level: {level_display}"
    x = cfg.hud_margin_x
    _draw_text_outlined(
        frame,
        line1,
        (x, y0 + cfg.hud_margin_y),
        font_scale=cfg.hud_score_scale,
        thickness=cfg.hud_text_thickness,
    )
    _draw_text_outlined(
        frame,
        line2,
        (x, y0 + cfg.hud_margin_y + 34),
        font_scale=cfg.hud_level_scale,
        thickness=cfg.hud_text_thickness,
    )


def _one_frame(
    frame: np.ndarray,
    w: int,
    h: int,
    landmarks: dict[int, tuple[float, float, float]],
    score: float,
    level_display: str,
    banner_tier: str | None,
    risk_flag: bool,
    cfg: OverlayVizConfig,
) -> None:
    top = 0
    if banner_tier in ("HIGH", "CRITICAL"):
        top = draw_elevated_risk_banner(frame, banner_tier, cfg)
        _apply_tint(frame, (0, 0, 200), cfg.warning_tint_strength)
    elif risk_flag and banner_tier is None:
        _draw_risk_flag_accent(frame, cfg)
    if landmarks:
        draw_skeleton_bgr(frame, landmarks, w, h, cfg)
    draw_risk_hud(frame, score, level_display, top, cfg)


# ---------------------------------------------------------------------------
# Core encode loop
# ---------------------------------------------------------------------------


def _render_core(
    video_path: Path,
    pose_seq: list[dict[int, tuple[float, float, float]]],
    risk_df: pd.DataFrame,
    output_path: Path,
    cfg: OverlayVizConfig,
) -> tuple[int, float]:
    if "risk_score" not in risk_df.columns:
        raise VideoIOError("Risk data must include column 'risk_score'.")
    if not video_path.is_file():
        raise VideoIOError(f"Video not found: {video_path}")

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoIOError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if w <= 0 or h <= 0:
        cap.release()
        raise VideoIOError("Invalid frame size.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise VideoIOError(f"Cannot create output: {output_path}")

    n_pose = len(pose_seq)
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            land = pose_seq[frame_idx] if frame_idx < n_pose else {}
            sc, lv, ban, flg = risk_values_for_frame(risk_df, frame_idx)
            _one_frame(frame, w, h, land, sc, lv, ban, flg, cfg)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    if frame_idx == 0:
        raise VideoIOError("No frames read from video.")
    if n_pose and frame_idx > n_pose:
        logger.warning("Video has %s frames but only %s pose frames.", frame_idx, n_pose)
    if len(risk_df) and "frame_index" not in risk_df.columns and frame_idx > len(risk_df):
        logger.warning("More video frames (%s) than risk rows (%s).", frame_idx, len(risk_df))
    return frame_idx, fps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_risk_overlay(
    video_path: Path,
    pose_csv: Path,
    risk_csv: Path,
    output_path: Path,
    *,
    viz_config: OverlayVizConfig | None = None,
) -> Path:
    """
    Build a demo video: original frames, optional skeleton, on-screen **risk** and **level**,
    and a **red warning banner** when the level is HIGH/CRITICAL (or legacy ``risk_flag``-only
    with no ``risk_level`` column, shown as **HIGH** styling).

    Parameters
    ----------
    video_path
        Input video (OpenCV readable).
    pose_csv
        Consolidated keypoints CSV or directory of per-frame CSVs (see
        :func:`load_indexed_sequence`).
    risk_csv
        At least ``risk_score``; optional ``risk_level``, ``frame_index``, ``risk_flag``.
    output_path
        Output path (e.g. ``.mp4``, codec ``mp4v``).
    """
    risk_df = read_risk_csv(risk_csv)
    if "risk_score" not in risk_df.columns:
        raise VideoIOError(f"Risk CSV must include column 'risk_score': {risk_csv}")
    cfg = viz_config or OverlayVizConfig()
    pose_seq = load_pose_indexed_sequence(pose_csv)
    n, fpsz = _render_core(video_path, pose_seq, risk_df, output_path, cfg)
    op = output_path.resolve()
    logger.info("Wrote %s (%s frames @ %.2f FPS).", op, n, fpsz)
    return op


def _prepare_risk_frame(
    risk_csv_path: Path | None,
    keypoints_source: Path,
    video_fps: float,
    risk_rules: Path | None,
    rolling_window: int,
) -> pd.DataFrame:
    if risk_csv_path is not None and risk_csv_path.is_file():
        return read_risk_csv(risk_csv_path)
    params: RiskRuleParams = risk_rules_from_yaml(risk_rules) if risk_rules else RiskRuleParams()
    feat = compute_pose_features(
        keypoints_source,
        fps=float(video_fps),
        rolling_window=rolling_window,
    )
    return detect_risk_events(feat, params)


def _viz_dict_to_config(
    viz_config: Mapping[str, Any] | None,
) -> OverlayVizConfig:
    if viz_config is None:
        return OverlayVizConfig()
    if isinstance(viz_config, OverlayVizConfig):
        return viz_config
    o = OverlayVizConfig()
    sk = (viz_config or {}).get("skeleton", {}) or {}
    if "line_bgr" in sk:
        lb = sk["line_bgr"]
        o.skeleton_line_bgr = (int(lb[0]), int(lb[1]), int(lb[2]))
    if "joint_bgr" in sk:
        jb = sk["joint_bgr"]
        o.skeleton_joint_bgr = (int(jb[0]), int(jb[1]), int(jb[2]))
    o.skeleton_line_thickness = int(sk.get("line_thickness", o.skeleton_line_thickness))
    o.skeleton_joint_radius = int(sk.get("joint_radius", o.skeleton_joint_radius))
    rk = (viz_config or {}).get("risk", {}) or {}
    o.risk_flag_border_thickness = int(rk.get("border_thickness", o.risk_flag_border_thickness))
    o.risk_flag_tint_strength = float(rk.get("overlay_strength", o.risk_flag_tint_strength))
    return o


def render_risk_overlay_video(
    video_path: Path,
    keypoints_source: Path,
    output_path: Path,
    *,
    risk_csv: Path | None = None,
    risk_rules_yaml: Path | None = None,
    rolling_window: int = 5,
    viz_config: Mapping[str, Any] | None = None,
) -> int:
    """
    CLI-oriented wrapper: return exit code 0 on success, 1 on failure.
    If ``risk_csv`` is omitted, risk is computed from keypoints and rules.
    """
    ocfg = _viz_dict_to_config(viz_config)

    if not video_path.is_file():
        logger.error("Video not found: %s", video_path)
        return 1
    if not (keypoints_source.is_file() or keypoints_source.is_dir()):
        logger.error("Keypoints source not found: %s", keypoints_source)
        return 1

    try:
        pose_seq = load_indexed_sequence(keypoints_source)
    except Exception as exc:
        logger.error("Could not load keypoints from %s: %s", keypoints_source, exc)
        return 1
    if not pose_seq:
        logger.error("No pose frames found under %s", keypoints_source)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 1
    vfps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    try:
        risk_df = _prepare_risk_frame(
            risk_csv,
            keypoints_source,
            vfps,
            risk_rules_yaml,
            rolling_window,
        )
    except Exception as exc:
        logger.error("Risk preparation failed: %s", exc)
        return 1
    if risk_df is None or "risk_score" not in risk_df.columns:
        logger.error("Risk data must contain column 'risk_score'.")
        return 1

    if "risk_level" not in risk_df.columns and "risk_flag" in risk_df.columns:
        risk_df = risk_df.copy()
        risk_df["risk_level"] = np.where(
            risk_df["risk_flag"].astype(bool),
            "HIGH",
            "LOW",
        )

    try:
        n, fpsz = _render_core(video_path, pose_seq, risk_df, output_path, ocfg)
    except VideoIOError as e:
        logger.error("%s", e)
        return 1
    op = output_path.resolve()
    logger.info("Wrote %s (%s frames @ %.2f FPS).", op, n, fpsz)
    return 0
