"""
Biomechanical features from pose keypoints (tabular long format or loaded CSV sources).

Provides pure helpers for centers and torso angle, plus :func:`compute_biomechanical_features`
for frame-wise metrics. :func:`compute_pose_features` loads CSV sources and augments with
temporal statistics for the risk layer.

Authorship: D. Martin-Moncunill, C. A. Sánchez (Camilo José Cela University, UCJC, Spain).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fightsafe_ai.keypoints.io import load_landmark_maps_ordered
from fightsafe_ai.utils.sorting import natural_sort_strings


# ---------------------------------------------------------------------------
# Pure geometry helpers (unit-testable)
# ---------------------------------------------------------------------------

LandmarkMap = dict[str, tuple[float, float]]
XY = tuple[float, float]


def _midpoint(a: XY | None, b: XY | None) -> XY | None:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _mid_wrist_xy(pts: LandmarkMap | None) -> XY | None:
    """Midpoint of wrists when available (for inbound strike speed proxy)."""
    if not pts:
        return None
    lw, rw = pts.get("left_wrist"), pts.get("right_wrist")
    if lw is None and rw is None:
        return None
    if lw is None:
        assert rw is not None
        return (float(rw[0]), float(rw[1]))
    if rw is None:
        return (float(lw[0]), float(lw[1]))
    return ((float(lw[0]) + float(rw[0])) * 0.5, (float(lw[1]) + float(rw[1])) * 0.5)


def compute_body_centers(landmarks: Mapping[str, XY]) -> dict[str, float]:
    """
    Compute shoulder midpoint, hip midpoint, and head (nose) coordinates.

    Missing symmetric pairs or ``nose`` yield NaN for the corresponding fields.

    Parameters
    ----------
    landmarks:
        Maps MediaPipe-style names (e.g. ``left_shoulder``, ``nose``) to ``(x, y)`` in
        normalized image coordinates.

    Returns
    -------
    dict
        Keys: ``shoulder_center_x``, ``shoulder_center_y``, ``hip_center_x``, ``hip_center_y``,
        ``head_x``, ``head_y``. Values are floats or ``nan``.
    """
    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    lh = landmarks.get("left_hip")
    rh = landmarks.get("right_hip")
    nose = landmarks.get("nose")

    sm = _midpoint(ls, rs)
    hm = _midpoint(lh, rh)

    def xy_or_nan(p: XY | None, idx: int) -> float:
        if p is None:
            return float(np.nan)
        return float(p[idx])

    return {
        "shoulder_center_x": xy_or_nan(sm, 0),
        "shoulder_center_y": xy_or_nan(sm, 1),
        "hip_center_x": xy_or_nan(hm, 0),
        "hip_center_y": xy_or_nan(hm, 1),
        "head_x": xy_or_nan(nose, 0),
        "head_y": xy_or_nan(nose, 1),
    }


def compute_torso_angle(
    shoulder_center: XY | None,
    hip_center: XY | None,
) -> float:
    """
    Angle of the torso (hip → shoulder) vs. upward vertical in **degrees**.

    Image coordinates use **y down**; vertical “up” is ``(0, -1)``, so:

    ``atan2(dx, -dy)`` with ``dx = shoulder_x - hip_x``, ``dy = shoulder_y - hip_y``.

    Returns NaN if either center is missing.
    """
    if shoulder_center is None or hip_center is None:
        return float(np.nan)
    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]
    return float(np.degrees(np.arctan2(dx, -dy)))


def _shoulder_hip_centers_from_dict(
    landmarks: Mapping[str, XY],
) -> tuple[XY | None, XY | None]:
    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    lh = landmarks.get("left_hip")
    rh = landmarks.get("right_hip")
    return _midpoint(ls, rs), _midpoint(lh, rh)


def compute_body_height_proxy(landmarks: Mapping[str, XY]) -> float:
    """
    Vertical extent across head, shoulders, hips, and ankles when available.

    ``max(y) - min(y)`` over ``{nose, left_shoulder, right_shoulder, left_hip, right_hip,
    left_ankle, right_ankle}`` intersected with ``landmarks``. Returns NaN if fewer than
    two points are available.
    """
    names = (
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_ankle",
        "right_ankle",
    )
    ys = [landmarks[n][1] for n in names if n in landmarks]
    if len(ys) < 2:
        return float(np.nan)
    return float(max(ys) - min(ys))


def knee_flexion_deg(hip: XY | None, knee: XY | None, ankle: XY | None) -> float:
    """
    **Flexion (degrees) at the knee** — interior joint angle: 0° = fully extended, ~180° line.

    2D image; **not** a clinical goniometry measurement. Returns NaN if a point is missing.
    """
    if hip is None or knee is None or ankle is None:
        return float("nan")
    a = (float(hip[0] - knee[0]), float(hip[1] - knee[1]))  # knee -> hip
    b = (float(ankle[0] - knee[0]), float(ankle[1] - knee[1]))  # knee -> ankle
    la = float(np.hypot(a[0], a[1])) + 1e-9
    lb = float(np.hypot(b[0], b[1])) + 1e-9
    c = (a[0] * b[0] + a[1] * b[1]) / (la * lb)
    c = float(np.clip(c, -1.0, 1.0))
    interior = float(np.degrees(np.arccos(c)))
    flexion = 180.0 - interior
    if not np.isfinite(flexion):
        return float("nan")
    return max(0.0, min(180.0, flexion))


def _ankle_y_min(pts: LandmarkMap | None) -> float:
    if pts is None:
        return float("nan")
    ys = [pts[k][1] for k in ("left_ankle", "right_ankle") if k in pts]
    if not ys:
        return float("nan")
    return float(max(ys))


def compute_guard_and_facing_scores(
    landmarks: Mapping[str, XY] | None,
    body_height_proxy: float,
) -> tuple[float, float]:
    """
    Per-frame heuristics (normalized coordinates, y-down).

    - **guard_level** (0–1): higher means hands (wrist mid) are clearly *below* the head — a
      **low guard** relative to the face (decision-support proxy, not a strike count).
    - **facing_away_score** (0–1): higher means nose is strongly offset from the mid-shoulder
      line (proxy for *turning away* / non-frontal in monocular view).

    Returns ``(0.0, 0.0)`` if landmarks or body height are unusable.
    """
    if landmarks is None or not np.isfinite(body_height_proxy) or body_height_proxy <= 1e-6:
        return 0.0, 0.0
    nose = landmarks.get("nose")
    lw = landmarks.get("left_wrist")
    rw = landmarks.get("right_wrist")
    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    if ls is None or rs is None or nose is None or (lw is None and rw is None):
        return 0.0, 0.0
    wys = [w[1] for w in (lw, rw) if w is not None]
    w_mid_y = float(sum(wys) / len(wys)) if wys else float("nan")
    n_y, n_x = float(nose[1]), float(nose[0])
    if not np.isfinite(w_mid_y) or not np.isfinite(n_y):
        return 0.0, 0.0
    # Hands below head in image => larger y => low guard
    g_raw = (w_mid_y - n_y) / max(body_height_proxy, 1e-6)
    guard_level = float(np.clip((g_raw - 0.05) / 0.55, 0.0, 1.0))
    mxs = (float(ls[0]) + float(rs[0])) * 0.5
    shoulder_w = abs(float(rs[0] - ls[0])) + 1e-6
    f_raw = abs(n_x - mxs) / shoulder_w
    facing_away = float(np.clip((f_raw - 0.12) / 0.5, 0.0, 1.0))
    return guard_level, facing_away


def compute_is_low_posture(
    hip_vertical_position: float,
    *,
    threshold: float = 0.58,
) -> bool:
    """
    Heuristic: subject is in a **low / crouched** posture when hip height (y) exceeds threshold.

    In normalized coordinates, larger ``y`` is lower in the image.
    """
    if hip_vertical_position != hip_vertical_position:  # NaN
        return False
    return bool(hip_vertical_position >= threshold)


def compute_biomechanical_features(
    df: pd.DataFrame,
    *,
    low_posture_hip_threshold: float = 0.58,
) -> pd.DataFrame:
    """
    Build one row per ``frame_id`` from long-format pose rows.

    Expected columns at minimum: ``frame_id``, ``keypoint_name``, ``x``, ``y``.
    Optional: ``z``, ``visibility`` (ignored for geometry here but preserved if merged later).

    Output columns
    --------------
    ``frame_id``, ``shoulder_center_x``, ``shoulder_center_y``, ``hip_center_x``, ``hip_center_y``,
    ``head_x``, ``head_y``, ``torso_angle_degrees``, ``hip_vertical_position``, ``head_vertical_position``,
    ``body_height_proxy``, ``is_low_posture``
    """
    required = {"frame_id", "keypoint_name", "x", "y"}
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "frame_id",
                "shoulder_center_x",
                "shoulder_center_y",
                "hip_center_x",
                "hip_center_y",
                "head_x",
                "head_y",
                "torso_angle_degrees",
                "hip_vertical_position",
                "head_vertical_position",
                "body_height_proxy",
                "is_low_posture",
            ]
        )
    cols = set(df.columns.astype(str))
    if not required.issubset(cols):
        raise ValueError(f"DataFrame must contain columns {required}, got {sorted(cols)}")

    work = df[list(required | ({"z", "visibility"} & cols))].copy()
    work["frame_id"] = work["frame_id"].astype(str)
    work["keypoint_name"] = work["keypoint_name"].astype(str)

    frame_ids = natural_sort_strings(work["frame_id"].unique().tolist())
    out_rows: list[dict[str, object]] = []

    for fid in frame_ids:
        g = work[work["frame_id"] == fid]
        landmarks = _landmarks_tuple_map(g)
        centers = compute_body_centers(landmarks)
        sm, hm = _shoulder_hip_centers_from_dict(landmarks)
        torso_deg = compute_torso_angle(sm, hm)

        hip_y = centers["hip_center_y"]
        head_y = centers["head_y"]
        height_px = compute_body_height_proxy(landmarks)

        row = {
            "frame_id": fid,
            "shoulder_center_x": centers["shoulder_center_x"],
            "shoulder_center_y": centers["shoulder_center_y"],
            "hip_center_x": centers["hip_center_x"],
            "hip_center_y": centers["hip_center_y"],
            "head_x": centers["head_x"],
            "head_y": centers["head_y"],
            "torso_angle_degrees": torso_deg,
            "hip_vertical_position": hip_y,
            "head_vertical_position": head_y,
            "body_height_proxy": height_px,
            "is_low_posture": compute_is_low_posture(hip_y, threshold=low_posture_hip_threshold),
        }
        out_rows.append(row)

    return pd.DataFrame.from_records(out_rows)


def _landmarks_tuple_map(group: pd.DataFrame) -> LandmarkMap:
    """Last row wins if duplicate keypoint_name."""
    out: LandmarkMap = {}
    for _, r in group.iterrows():
        try:
            name = str(r["keypoint_name"])
            x = float(r["x"])
            y = float(r["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isnan(x) or np.isnan(y):
            continue
        out[name] = (x, y)
    return out


# ---------------------------------------------------------------------------
# CSV loading + temporal stack for risk pipeline
# ---------------------------------------------------------------------------


def _long_dataframe_from_keypoints_source(
    keypoints_source: Path,
    glob_pattern: str,
) -> pd.DataFrame:
    """Load consolidated CSV or expand directory of legacy CSVs into long format."""
    keypoints_source = keypoints_source.expanduser().resolve()

    if keypoints_source.is_file() and keypoints_source.suffix.lower() == ".csv":
        raw = pd.read_csv(keypoints_source)
        return _normalize_long_columns(raw)

    landmark_frames = load_landmark_maps_ordered(keypoints_source, glob_pattern=glob_pattern)
    rows: list[dict[str, object]] = []
    for fid, pts in landmark_frames:
        if not pts:
            continue
        frame_key = Path(fid).stem
        for name, (x, y) in pts.items():
            rows.append(
                {
                    "frame_id": frame_key,
                    "keypoint_name": name,
                    "x": x,
                    "y": y,
                    "z": np.nan,
                    "visibility": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _normalize_long_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Ensure column names and dtypes for consolidated exports."""
    df = raw.copy()
    rename = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rename)
    need = {"frame_id", "keypoint_name", "x", "y"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV missing columns {need}: {list(df.columns)}")
    if "z" not in df.columns:
        df["z"] = np.nan
    if "visibility" not in df.columns:
        df["visibility"] = np.nan
    return df


def _feet_lowest_y(points: LandmarkMap | None) -> float:
    if not points:
        return float(np.nan)
    names = (
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    )
    ys = [points[n][1] for n in names if n in points]
    if not ys:
        return float(np.nan)
    return float(max(ys))


def _rolling_mean_variance(series_list: list[pd.Series], window: int) -> pd.Series:
    if not series_list:
        return pd.Series(dtype=float)
    acc: pd.Series | None = None
    for s in series_list:
        v = s.rolling(window=window, min_periods=1).var()
        acc = v if acc is None else acc + v
    assert acc is not None
    return acc / len(series_list)


def build_biomechanical_mvp_dataframe(
    keypoints_source: Path,
    fps: float = 10.0,
    rolling_window: int = 5,
    ground_y_threshold: float = 0.82,
    stability_landmarks: list[str] | None = None,
    glob_pattern: str = "*.csv",
    low_posture_hip_threshold: float = 0.58,
) -> pd.DataFrame:
    """
    Load keypoints, compute biomechanics, and assemble the per-frame table **before** temporal
    statistics (:func:`fightsafe_ai.features.temporal.compute_temporal_features`).

    The end-to-end pipeline calls this and temporal augmentation as separate stages; for a single
    combined step use :func:`compute_pose_features`.
    """
    long_df = _long_dataframe_from_keypoints_source(keypoints_source, glob_pattern)
    bio = compute_biomechanical_features(
        long_df, low_posture_hip_threshold=low_posture_hip_threshold
    )

    if bio.empty:
        bio_cols = compute_biomechanical_features(pd.DataFrame()).columns.tolist()
        extra = [
            "frame_index",
            "source_csv",
            "torso_angle_deg",
            "hip_center_y",
            "hip_vertical_velocity",
            "keypoint_position_variance",
            "near_ground",
            "time_near_ground_cumulative_sec",
            "guard_level",
            "facing_away_score",
            "knee_flexion_left_deg",
            "knee_flexion_right_deg",
            "ankle_y_min",
            "strike_incoming_proxy",
        ]
        merged = bio_cols + [c for c in extra if c not in bio_cols]
        return pd.DataFrame(columns=merged)

    landmark_frames = load_landmark_maps_ordered(keypoints_source, glob_pattern=glob_pattern)
    fid_order = bio["frame_id"].astype(str).tolist()
    pts_by_fid: dict[str, LandmarkMap | None] = {}
    for label, pts in landmark_frames:
        pts_by_fid[Path(label).stem] = pts

    records: list[dict[str, object]] = []
    per_frame_points: list[LandmarkMap | None] = []

    for i, fid in enumerate(fid_order):
        pts = pts_by_fid.get(fid)
        per_frame_points.append(pts)
        row_raw = bio.iloc[i].to_dict()
        rec: dict[str, object] = {str(k): v for k, v in row_raw.items()}
        rec["frame_index"] = i
        rec["source_csv"] = fid
        fy = _feet_lowest_y(pts)
        rec["near_ground"] = bool(not np.isnan(fy) and fy >= ground_y_threshold)
        bhx = rec.get("body_height_proxy")
        if bhx is None:
            bh = float("nan")
        elif isinstance(bhx, (int, float, np.integer, np.floating)):
            bh = float(bhx)
        else:
            bh = float("nan")
        g_lv, f_away = compute_guard_and_facing_scores(pts, bh)
        rec["guard_level"] = g_lv
        rec["facing_away_score"] = f_away
        if pts:
            lhip = pts.get("left_hip")
            lk = pts.get("left_knee")
            la = pts.get("left_ankle")
            rhip = pts.get("right_hip")
            rk = pts.get("right_knee")
            ra = pts.get("right_ankle")
            rec["knee_flexion_left_deg"] = knee_flexion_deg(lhip, lk, la)
            rec["knee_flexion_right_deg"] = knee_flexion_deg(rhip, rk, ra)
            rec["ankle_y_min"] = _ankle_y_min(pts)
        else:
            rec["knee_flexion_left_deg"] = float("nan")
            rec["knee_flexion_right_deg"] = float("nan")
            rec["ankle_y_min"] = float("nan")
        strike_incoming_proxy = 0.0
        if i > 0 and pts:
            p0 = per_frame_points[i - 1]
            w0, w1 = _mid_wrist_xy(p0), _mid_wrist_xy(pts)
            if w0 is not None and w1 is not None:
                dt_step = 1.0 / max(float(fps), 1e-6)
                spd = float(np.hypot(w1[0] - w0[0], w1[1] - w0[1]) / dt_step)
                strike_incoming_proxy = float(np.clip(spd / 3.0, 0.0, 1.0))
        rec["strike_incoming_proxy"] = strike_incoming_proxy
        records.append(rec)

    df = pd.DataFrame.from_records(records)

    # Risk layer aliases (historical column names)
    df["torso_angle_deg"] = df["torso_angle_degrees"]
    df["hip_center_y"] = df["hip_vertical_position"]

    dt = 1.0 / fps if fps > 0 else np.nan
    hy = df["hip_vertical_position"].astype(float)
    if np.isfinite(dt) and len(hy) >= 2:
        hy_arr = np.asarray(hy.to_numpy(dtype=float), dtype=float)
        df["hip_vertical_velocity"] = np.gradient(hy_arr, dt)
    elif len(hy) < 2:
        # One frame: gradient is undefined; use 0.0 so downstream risk columns are finite.
        df["hip_vertical_velocity"] = 0.0
    else:
        df["hip_vertical_velocity"] = np.nan

    stability_landmarks = stability_landmarks or [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    coord_series: list[pd.Series] = []
    for lm in stability_landmarks:
        xs: list[float] = []
        ys: list[float] = []
        for pts in per_frame_points:
            if pts and lm in pts:
                xs.append(pts[lm][0])
                ys.append(pts[lm][1])
            else:
                xs.append(np.nan)
                ys.append(np.nan)
        coord_series.append(pd.Series(xs, dtype=float))
        coord_series.append(pd.Series(ys, dtype=float))

    if coord_series:
        df["keypoint_position_variance"] = _rolling_mean_variance(coord_series, rolling_window)

    if np.isfinite(dt):
        near = df["near_ground"].astype(bool)
        df["time_near_ground_cumulative_sec"] = (near.astype(float) * dt).cumsum()
    else:
        df["time_near_ground_cumulative_sec"] = np.nan

    return df


def build_biomechanical_mvp_dataframe_from_landmark_sequence(
    frames: Sequence[tuple[str, LandmarkMap | None]],
    *,
    fps: float = 10.0,
    rolling_window: int = 5,
    ground_y_threshold: float = 0.82,
    stability_landmarks: list[str] | None = None,
    low_posture_hip_threshold: float = 0.58,
) -> pd.DataFrame:
    """
    Same columns as :func:`build_biomechanical_mvp_dataframe`, but from an in-memory
    time-ordered sequence ``(frame_id, landmarks)`` for live / streaming use.

    Empty or all-empty landmark maps yield an empty framed DataFrame (same schema as the path-based builder).
    """
    rows: list[dict[str, object]] = []
    for fid, pts in frames:
        if not pts:
            continue
        sid = str(fid)
        for name, xy in pts.items():
            rows.append(
                {
                    "frame_id": sid,
                    "keypoint_name": str(name),
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "z": np.nan,
                    "visibility": 1.0,
                }
            )
    long_df = pd.DataFrame(rows)
    bio = compute_biomechanical_features(
        long_df, low_posture_hip_threshold=low_posture_hip_threshold
    )

    if bio.empty:
        bio_cols = compute_biomechanical_features(pd.DataFrame()).columns.tolist()
        extra = [
            "frame_index",
            "source_csv",
            "torso_angle_deg",
            "hip_center_y",
            "hip_vertical_velocity",
            "keypoint_position_variance",
            "near_ground",
            "time_near_ground_cumulative_sec",
            "guard_level",
            "facing_away_score",
            "knee_flexion_left_deg",
            "knee_flexion_right_deg",
            "ankle_y_min",
            "strike_incoming_proxy",
        ]
        merged = bio_cols + [c for c in extra if c not in bio_cols]
        return pd.DataFrame(columns=merged)

    fid_order = bio["frame_id"].astype(str).tolist()
    pts_by_fid: dict[str, LandmarkMap | None] = {str(a): b for a, b in frames}

    records: list[dict[str, object]] = []
    per_frame_points: list[LandmarkMap | None] = []

    for i, fid in enumerate(fid_order):
        pts = pts_by_fid.get(fid)
        per_frame_points.append(pts)
        row_raw = bio.iloc[i].to_dict()
        rec: dict[str, object] = {str(k): v for k, v in row_raw.items()}
        rec["frame_index"] = i
        rec["source_csv"] = fid
        fy = _feet_lowest_y(pts)
        rec["near_ground"] = bool(not np.isnan(fy) and fy >= ground_y_threshold)
        bhx = rec.get("body_height_proxy")
        if bhx is None:
            bh = float("nan")
        elif isinstance(bhx, (int, float, np.integer, np.floating)):
            bh = float(bhx)
        else:
            bh = float("nan")
        g_lv, f_away = compute_guard_and_facing_scores(pts, bh)
        rec["guard_level"] = g_lv
        rec["facing_away_score"] = f_away
        if pts:
            lhip = pts.get("left_hip")
            lk = pts.get("left_knee")
            la = pts.get("left_ankle")
            rhip = pts.get("right_hip")
            rk = pts.get("right_knee")
            ra = pts.get("right_ankle")
            rec["knee_flexion_left_deg"] = knee_flexion_deg(lhip, lk, la)
            rec["knee_flexion_right_deg"] = knee_flexion_deg(rhip, rk, ra)
            rec["ankle_y_min"] = _ankle_y_min(pts)
        else:
            rec["knee_flexion_left_deg"] = float("nan")
            rec["knee_flexion_right_deg"] = float("nan")
            rec["ankle_y_min"] = float("nan")
        strike_incoming_proxy = 0.0
        if i > 0 and pts:
            p0 = per_frame_points[i - 1]
            w0, w1 = _mid_wrist_xy(p0), _mid_wrist_xy(pts)
            if w0 is not None and w1 is not None:
                dt_step = 1.0 / max(float(fps), 1e-6)
                spd = float(np.hypot(w1[0] - w0[0], w1[1] - w0[1]) / dt_step)
                strike_incoming_proxy = float(np.clip(spd / 3.0, 0.0, 1.0))
        rec["strike_incoming_proxy"] = strike_incoming_proxy
        records.append(rec)

    df = pd.DataFrame.from_records(records)

    df["torso_angle_deg"] = df["torso_angle_degrees"]
    df["hip_center_y"] = df["hip_vertical_position"]

    dt = 1.0 / fps if fps > 0 else np.nan
    hy = df["hip_vertical_position"].astype(float)
    if np.isfinite(dt) and len(hy) >= 2:
        hy_arr = np.asarray(hy.to_numpy(dtype=float), dtype=float)
        df["hip_vertical_velocity"] = np.gradient(hy_arr, dt)
    elif len(hy) < 2:
        df["hip_vertical_velocity"] = 0.0
    else:
        df["hip_vertical_velocity"] = np.nan

    stability_landmarks = stability_landmarks or [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    coord_series: list[pd.Series] = []
    for lm in stability_landmarks:
        xs: list[float] = []
        ys: list[float] = []
        for pts in per_frame_points:
            if pts and lm in pts:
                xs.append(pts[lm][0])
                ys.append(pts[lm][1])
            else:
                xs.append(np.nan)
                ys.append(np.nan)
        coord_series.append(pd.Series(xs, dtype=float))
        coord_series.append(pd.Series(ys, dtype=float))

    if coord_series:
        df["keypoint_position_variance"] = _rolling_mean_variance(coord_series, rolling_window)

    if np.isfinite(dt):
        near = df["near_ground"].astype(bool)
        df["time_near_ground_cumulative_sec"] = (near.astype(float) * dt).cumsum()
    else:
        df["time_near_ground_cumulative_sec"] = np.nan

    return df


def compute_pose_features(
    keypoints_source: Path,
    fps: float = 10.0,
    rolling_window: int = 5,
    ground_y_threshold: float = 0.82,
    stability_landmarks: list[str] | None = None,
    glob_pattern: str = "*.csv",
    low_posture_hip_threshold: float = 0.58,
) -> pd.DataFrame:
    """
    Load keypoints from disk, compute biomechanics, then temporal / stability metrics.

    Also calls :func:`fightsafe_ai.features.temporal.compute_temporal_features` so each row has
    ``instability_score``, ``low_posture_duration_frames``, and head/torso velocities required
    by the interpretable combat MVP rules in :func:`fightsafe_ai.risk.scorer.build_combat_mvp_frame_risk`.

    Output includes columns required by :func:`fightsafe_ai.risk.engine.detect_risk_events`
    (``torso_angle_deg``, ``hip_vertical_velocity``, ``keypoint_position_variance``,
    ``near_ground``, …) alongside the biomechanical columns from
    :func:`compute_biomechanical_features`.
    """
    df = build_biomechanical_mvp_dataframe(
        keypoints_source,
        fps=fps,
        rolling_window=rolling_window,
        ground_y_threshold=ground_y_threshold,
        stability_landmarks=stability_landmarks,
        glob_pattern=glob_pattern,
        low_posture_hip_threshold=low_posture_hip_threshold,
    )
    if df.empty:
        return df

    from fightsafe_ai.features.anomaly import add_limb_anomaly_columns
    from fightsafe_ai.features.temporal import compute_temporal_features

    temp = compute_temporal_features(
        df,
        int(max(1, round(float(fps)))),
        rolling_window_frames=rolling_window,
    )
    return add_limb_anomaly_columns(temp, float(fps))
