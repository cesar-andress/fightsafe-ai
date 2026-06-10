"""
Per-frame benchmark features from consolidated ``pose_keypoints.csv``.

FightSafe-Bench (paper3): joint angles, angular kinematics, symmetry, COM proxies,
motion energy, and posture stability — independent of fusion / risk layers.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fightsafe_ai.features.biomechanics import (
    compute_body_centers,
    compute_torso_angle,
    knee_flexion_deg,
)
from fightsafe_ai.keypoints.io import load_landmark_maps_ordered


logger = logging.getLogger(__name__)

XY = tuple[float, float]

# Landmarks used for COM weighting (MediaPipe names present in TapKO exports).
_COM_LANDMARKS: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_FRAME_INDEX_RE = re.compile(r"(\d+)$")


def _pt(landmarks: dict[str, XY], name: str) -> XY | None:
    return landmarks.get(name)


def _interior_angle_deg(a: XY | None, b: XY | None, c: XY | None) -> float:
    """Interior angle at vertex ``b`` formed by segments ``b→a`` and ``b→c`` (degrees)."""
    if a is None or b is None or c is None:
        return float(np.nan)
    v1 = (float(a[0] - b[0]), float(a[1] - b[1]))
    v2 = (float(c[0] - b[0]), float(c[1] - b[1]))
    n1 = float(np.hypot(v1[0], v1[1])) + 1e-9
    n2 = float(np.hypot(v2[0], v2[1])) + 1e-9
    cos_t = float(np.clip((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_t)))


def _flexion_angle_deg(a: XY | None, b: XY | None, c: XY | None) -> float:
    """Joint flexion: ``180° - interior angle`` at ``b`` (0 = extended)."""
    interior = _interior_angle_deg(a, b, c)
    if not np.isfinite(interior):
        return float(np.nan)
    return float(np.clip(180.0 - interior, 0.0, 180.0))


def _shoulder_abduction_deg(hip: XY | None, shoulder: XY | None, elbow: XY | None) -> float:
    """Angle between torso (hip→shoulder) and upper arm (shoulder→elbow) at the shoulder."""
    return _interior_angle_deg(hip, shoulder, elbow)


def _visibility_weighted_com(
    landmarks: dict[str, XY],
    visibilities: dict[str, float],
) -> tuple[float, float, float]:
    """
    Approximate 2D center of mass as visibility-weighted mean of key landmark positions.

    Returns ``(com_x, com_y, effective_mass)`` where effective_mass is sum of weights.
    """
    wx = wy = w_sum = 0.0
    for name in _COM_LANDMARKS:
        p = landmarks.get(name)
        if p is None:
            continue
        w = float(visibilities.get(name, 1.0))
        if not np.isfinite(w) or w <= 0.0:
            continue
        wx += w * float(p[0])
        wy += w * float(p[1])
        w_sum += w
    if w_sum <= 0.0:
        return float(np.nan), float(np.nan), 0.0
    return wx / w_sum, wy / w_sum, w_sum


def _load_visibility_maps(pose_csv: Path) -> dict[str, dict[str, float]]:
    """frame_id -> keypoint_name -> visibility from long-format pose CSV."""
    df = pd.read_csv(pose_csv, usecols=["frame_id", "keypoint_name", "visibility"])
    out: dict[str, dict[str, float]] = {}
    for frame_id, grp in df.groupby("frame_id", sort=False):
        vis_map: dict[str, float] = {}
        for r in grp.itertuples(index=False):
            if pd.notna(r.visibility):
                vis_map[str(r.keypoint_name)] = float(np.asarray(r.visibility).item())
        out[str(frame_id)] = vis_map
    return out


def _frame_sort_key(frame_id: str) -> tuple[int, str]:
    m = _FRAME_INDEX_RE.search(str(frame_id))
    if m:
        return (int(m.group(1)), str(frame_id))
    return (0, str(frame_id))


def _angular_derivative(series: np.ndarray, dt: float) -> np.ndarray:
    """First time derivative; ``np.gradient`` with spacing ``dt`` (handles NaNs as NaN out)."""
    if len(series) == 0:
        return series
    if len(series) == 1:
        return np.array([np.nan], dtype=float)
    grad = np.gradient(series.astype(float), dt)
    return np.asarray(grad, dtype=float)


def _per_frame_geometry(
    landmarks: dict[str, XY] | None,
    visibilities: dict[str, float],
) -> dict[str, float]:
    if not landmarks:
        return {k: float(np.nan) for k in _geometry_column_names()}

    centers = compute_body_centers(landmarks)
    sm = (
        (centers["shoulder_center_x"], centers["shoulder_center_y"])
        if np.isfinite(centers["shoulder_center_x"])
        else None
    )
    hm = (
        (centers["hip_center_x"], centers["hip_center_y"])
        if np.isfinite(centers["hip_center_x"])
        else None
    )

    torso = compute_torso_angle(sm, hm)

    l_elbow = _flexion_angle_deg(
        _pt(landmarks, "left_shoulder"),
        _pt(landmarks, "left_elbow"),
        _pt(landmarks, "left_wrist"),
    )
    r_elbow = _flexion_angle_deg(
        _pt(landmarks, "right_shoulder"),
        _pt(landmarks, "right_elbow"),
        _pt(landmarks, "right_wrist"),
    )
    l_knee = knee_flexion_deg(
        _pt(landmarks, "left_hip"),
        _pt(landmarks, "left_knee"),
        _pt(landmarks, "left_ankle"),
    )
    r_knee = knee_flexion_deg(
        _pt(landmarks, "right_hip"),
        _pt(landmarks, "right_knee"),
        _pt(landmarks, "right_ankle"),
    )
    l_hip = _flexion_angle_deg(
        _pt(landmarks, "left_shoulder"),
        _pt(landmarks, "left_hip"),
        _pt(landmarks, "left_knee"),
    )
    r_hip = _flexion_angle_deg(
        _pt(landmarks, "right_shoulder"),
        _pt(landmarks, "right_hip"),
        _pt(landmarks, "right_knee"),
    )
    l_sh_abd = _shoulder_abduction_deg(
        _pt(landmarks, "left_hip"),
        _pt(landmarks, "left_shoulder"),
        _pt(landmarks, "left_elbow"),
    )
    r_sh_abd = _shoulder_abduction_deg(
        _pt(landmarks, "right_hip"),
        _pt(landmarks, "right_shoulder"),
        _pt(landmarks, "right_elbow"),
    )

    com_x, com_y, com_weight = _visibility_weighted_com(landmarks, visibilities)

    la, ra = _pt(landmarks, "left_ankle"), _pt(landmarks, "right_ankle")
    lw, rw = _pt(landmarks, "left_wrist"), _pt(landmarks, "right_wrist")
    lk, rk = _pt(landmarks, "left_knee"), _pt(landmarks, "right_knee")

    base_width = float(np.nan)
    if la is not None and ra is not None:
        base_width = abs(float(la[0]) - float(ra[0]))

    def sym_diff(a: float, b: float) -> float:
        if not np.isfinite(a) or not np.isfinite(b):
            return float(np.nan)
        return abs(a - b)

    def height_sym(left: XY | None, right: XY | None) -> float:
        if left is None or right is None:
            return float(np.nan)
        return abs(float(left[1]) - float(right[1]))

    angle_sym_elbow = sym_diff(l_elbow, r_elbow)
    angle_sym_knee = sym_diff(l_knee, r_knee)
    angle_sym_hip = sym_diff(l_hip, r_hip)
    height_sym_wrist = height_sym(lw, rw)
    height_sym_ankle = height_sym(la, ra)
    height_sym_knee = height_sym(lk, rk)

    # Normalized symmetry index in [0,1]: mean absolute left-right angle gap / 180.
    sym_vals = [angle_sym_elbow, angle_sym_knee, angle_sym_hip]
    finite_sym = [v for v in sym_vals if np.isfinite(v)]
    limb_symmetry_index = float(np.mean(finite_sym) / 180.0) if finite_sym else float(np.nan)

    return {
        **centers,
        "torso_angle_deg": torso,
        "left_elbow_flexion_deg": l_elbow,
        "right_elbow_flexion_deg": r_elbow,
        "left_knee_flexion_deg": l_knee,
        "right_knee_flexion_deg": r_knee,
        "left_hip_flexion_deg": l_hip,
        "right_hip_flexion_deg": r_hip,
        "left_shoulder_abduction_deg": l_sh_abd,
        "right_shoulder_abduction_deg": r_sh_abd,
        "com_x": com_x,
        "com_y": com_y,
        "com_weight_sum": com_weight,
        "base_of_support_width": base_width,
        "limb_angle_symmetry_index": limb_symmetry_index,
        "elbow_angle_symmetry_deg": angle_sym_elbow,
        "knee_angle_symmetry_deg": angle_sym_knee,
        "hip_angle_symmetry_deg": angle_sym_hip,
        "wrist_height_symmetry": height_sym_wrist,
        "ankle_height_symmetry": height_sym_ankle,
        "knee_height_symmetry": height_sym_knee,
    }


def _geometry_column_names() -> list[str]:
    return [
        "shoulder_center_x",
        "shoulder_center_y",
        "hip_center_x",
        "hip_center_y",
        "head_x",
        "head_y",
        "torso_angle_deg",
        "left_elbow_flexion_deg",
        "right_elbow_flexion_deg",
        "left_knee_flexion_deg",
        "right_knee_flexion_deg",
        "left_hip_flexion_deg",
        "right_hip_flexion_deg",
        "left_shoulder_abduction_deg",
        "right_shoulder_abduction_deg",
        "com_x",
        "com_y",
        "com_weight_sum",
        "base_of_support_width",
        "limb_angle_symmetry_index",
        "elbow_angle_symmetry_deg",
        "knee_angle_symmetry_deg",
        "hip_angle_symmetry_deg",
        "wrist_height_symmetry",
        "ankle_height_symmetry",
        "knee_height_symmetry",
    ]


_ANGLE_COLS: tuple[str, ...] = (
    "torso_angle_deg",
    "left_elbow_flexion_deg",
    "right_elbow_flexion_deg",
    "left_knee_flexion_deg",
    "right_knee_flexion_deg",
    "left_hip_flexion_deg",
    "right_hip_flexion_deg",
    "left_shoulder_abduction_deg",
    "right_shoulder_abduction_deg",
)


def extract_benchmark_features(
    ordered_frames: list[tuple[str, dict[str, XY] | None]],
    visibility_by_frame: dict[str, dict[str, float]],
    *,
    fps: float,
) -> pd.DataFrame:
    """
    Build per-frame benchmark feature table from ordered landmark maps.

    Parameters
    ----------
    ordered_frames
        ``(frame_id, landmarks)`` in temporal order.
    visibility_by_frame
        Per-frame visibility weights for COM (from pose CSV).
    fps
        Nominal frame rate for velocity / acceleration / energy (Hz).
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    dt = 1.0 / float(fps)

    rows: list[dict[str, float | str | int]] = []
    for frame_index, (frame_id, lm) in enumerate(ordered_frames):
        vis = visibility_by_frame.get(str(frame_id), {})
        geom = _per_frame_geometry(lm, vis)
        rows.append({"frame_id": str(frame_id), "frame_index": frame_index, **geom})

    df = pd.DataFrame(rows)
    n = len(df)
    if n == 0:
        return df

    # Angular velocities and accelerations (deg/s, deg/s²).
    for col in _ANGLE_COLS:
        ang = df[col].to_numpy(dtype=float)
        vel = _angular_derivative(ang, dt)
        acc = _angular_derivative(vel, dt)
        df[f"{col}_velocity_deg_s"] = vel
        df[f"{col}_acceleration_deg_s2"] = acc

    # COM kinematics (normalized coord / s).
    com_x = df["com_x"].to_numpy(dtype=float)
    com_y = df["com_y"].to_numpy(dtype=float)
    com_vx = _angular_derivative(com_x, dt)
    com_vy = _angular_derivative(com_y, dt)
    com_ax = _angular_derivative(com_vx, dt)
    com_ay = _angular_derivative(com_vy, dt)
    df["com_velocity_x"] = com_vx
    df["com_velocity_y"] = com_vy
    df["com_speed"] = np.hypot(com_vx, com_vy)
    df["com_acceleration_x"] = com_ax
    df["com_acceleration_y"] = com_ay
    df["com_acceleration_mag"] = np.hypot(com_ax, com_ay)

    # Motion energy: sum of squared landmark speeds (visibility-weighted).
    motion_energy = np.zeros(n, dtype=float)
    landmark_speed_energy = np.zeros(n, dtype=float)
    for i, (frame_id, lm) in enumerate(ordered_frames):
        if i == 0 or lm is None:
            motion_energy[i] = np.nan
            landmark_speed_energy[i] = np.nan
            continue
        _prev_id, prev_lm = ordered_frames[i - 1]
        if prev_lm is None:
            motion_energy[i] = np.nan
            landmark_speed_energy[i] = np.nan
            continue
        vis = visibility_by_frame.get(str(frame_id), {})
        se = 0.0
        w_sum = 0.0
        for name in _COM_LANDMARKS:
            p = lm.get(name)
            p0 = prev_lm.get(name)
            if p is None or p0 is None:
                continue
            w = float(vis.get(name, 1.0))
            if w <= 0:
                continue
            vx = (float(p[0]) - float(p0[0])) / dt
            vy = (float(p[1]) - float(p0[1])) / dt
            se += w * (vx * vx + vy * vy)
            w_sum += w
        landmark_speed_energy[i] = se / w_sum if w_sum > 0 else np.nan
        motion_energy[i] = se

    df["motion_energy"] = motion_energy
    df["landmark_speed_energy_mean"] = landmark_speed_energy

    # Posture stability metrics.
    df["posture_stability_com_speed_inv"] = 1.0 / (df["com_speed"] + 1e-3)
    df["posture_stability_base_com_ratio"] = df["base_of_support_width"] / (df["com_speed"] + 1e-3)
    torso_vel = df["torso_angle_deg_velocity_deg_s"].abs()
    df["posture_stability_torso_angular_calm"] = 1.0 / (torso_vel + 1e-2)

    # Rolling stability (5-frame window ≈ rolling_window in MVP).
    win = 5
    df["posture_stability_com_speed_std_5f"] = (
        df["com_speed"].rolling(window=win, min_periods=2).std()
    )
    df["posture_stability_torso_std_5f"] = (
        df["torso_angle_deg"].rolling(window=win, min_periods=2).std()
    )

    df["fps"] = float(fps)
    df["timestamp_sec"] = df["frame_index"].astype(float) * dt
    return df


def resolve_fps_from_manifest(pose_csv: Path, manifest: Path | None = None) -> float:
    """Read ``fps`` from sibling ``tapko_manifest.json`` if present, else 30."""
    candidates = []
    if manifest is not None:
        candidates.append(manifest)
    candidates.append(pose_csv.parent / "tapko_manifest.json")
    for path in candidates:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                fps = float(data.get("fps", 30.0))
                if fps > 0:
                    return fps
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                logger.warning("Could not read fps from %s", path)
    return 30.0


def extract_benchmark_features_from_pose_csv(
    pose_csv: Path,
    *,
    fps: float | None = None,
    manifest: Path | None = None,
) -> pd.DataFrame:
    """Load ``pose_keypoints.csv`` and return benchmark feature dataframe."""
    pose_csv = pose_csv.expanduser().resolve()
    if not pose_csv.is_file():
        raise FileNotFoundError(f"Pose CSV not found: {pose_csv}")

    ordered = load_landmark_maps_ordered(pose_csv)
    ordered.sort(key=lambda t: _frame_sort_key(t[0]))
    vis_maps = _load_visibility_maps(pose_csv)
    rate = float(fps) if fps is not None else resolve_fps_from_manifest(pose_csv, manifest)
    return extract_benchmark_features(ordered, vis_maps, fps=rate)


def write_benchmark_features_csv(
    pose_csv: Path,
    output_csv: Path,
    *,
    fps: float | None = None,
    manifest: Path | None = None,
) -> pd.DataFrame:
    """Extract features and write ``output_csv``; returns dataframe."""
    df = extract_benchmark_features_from_pose_csv(pose_csv, fps=fps, manifest=manifest)
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote %s rows to %s", len(df), output_csv)
    return df


__all__ = [
    "extract_benchmark_features",
    "extract_benchmark_features_from_pose_csv",
    "resolve_fps_from_manifest",
    "write_benchmark_features_csv",
]
