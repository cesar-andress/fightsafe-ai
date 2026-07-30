"""
Temporal features on biomechanical frame sequences: velocities, rolling instability, posture span.

All operations assume **one row per frame in time order** (sorted upstream).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Default column names expected on the input biomechanics table
_DEFAULT_HIP_Y = "hip_vertical_position"
_DEFAULT_HEAD_Y = "head_vertical_position"
_DEFAULT_TORSO = "torso_angle_degrees"
_DEFAULT_LOW = "is_low_posture"

# Output column names
_COL_HIP_VEL = "hip_vertical_velocity"
_COL_HEAD_VEL = "head_vertical_velocity"
_COL_TORSO_VEL = "torso_angle_velocity"
_COL_INSTAB = "instability_score"
_COL_LOW_DUR = "low_posture_duration_frames"
_COL_REACTION = "reaction_delay_score"


@dataclass(frozen=True)
class ReactionDelayConfig:
    """
    Heuristic for **reduced movement / slow response** after a sudden hip deceleration (proxy
    for impact-like acceleration in the signal, not a medical “concussion” label).
    """

    impact_jerk_threshold: float = 1.15
    """Minimum |Δ hip_velocity| / dt (per second) to mark a candidate “impact” frame."""
    post_impact_lookback_frames: int = 18
    """Within this many past frames, stillness accumulates toward the score."""
    still_speed_ref: float = 0.14
    """Scale for ``|head_vel| + |hip_vel|`` — below this looks “still” post-event."""


@dataclass(frozen=True)
class TemporalFeatureConfig:
    """
    Configuration for :func:`compute_temporal_features`.

    Parameters
    ----------
    rolling_window_frames
        Size of the rolling window for **standard deviation** (instability) and for the
        **count** of low-posture frames. Must be at least 1.
    min_periods
        Minimum number of observations in window required to produce a value (pandas ``min_periods``).
    pre_smooth
        If True, apply a trailing rolling **mean** to position/angle series before taking
        the time derivative (reduces high-frequency noise). Uses ``pre_smooth_window_frames``.
    pre_smooth_window_frames
        Window for the optional pre-derivative smooth; must be at least 1.
    """

    rolling_window_frames: int = 5
    min_periods: int = 1
    pre_smooth: bool = False
    pre_smooth_window_frames: int = 3
    reaction_delay: ReactionDelayConfig | None = None

    def __post_init__(self) -> None:
        if self.rolling_window_frames < 1:
            raise ValueError("rolling_window_frames must be >= 1.")
        if self.min_periods < 1:
            raise ValueError("min_periods must be >= 1.")
        if self.pre_smooth_window_frames < 1:
            raise ValueError("pre_smooth_window_frames must be >= 1.")


def _time_step_seconds(fps: int) -> float:
    if fps <= 0:
        raise ValueError("fps must be a positive integer.")
    return 1.0 / float(fps)


def _gradient_series_per_second(series: pd.Series, dt: float) -> pd.Series:
    """
    Time derivative in **per second** units, using ``numpy.gradient`` on the 1D array.

    For uniform spacing ``dt`` between samples, this matches central differences at interior points.
    """
    values = series.to_numpy(dtype=float, copy=True)
    if values.size == 0:
        return pd.Series(dtype=float, index=series.index)
    if values.size < 2:
        return pd.Series(0.0, index=series.index, dtype=float)
    g = np.gradient(values, dt)
    return pd.Series(g, index=series.index, dtype=float)


def _rolling_std(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.rolling(window=window, min_periods=min_periods).std()


def _reaction_delay_score(
    hip_vel: np.ndarray,
    head_vel: np.ndarray,
    dt: float,
    cfg: ReactionDelayConfig,
) -> np.ndarray:
    """
    0–1: higher when movement stays low while a **recent** large hip jerk was seen (rolling max).
    """
    n = len(hip_vel)
    if n == 0:
        return np.zeros(0, dtype=float)
    dt = max(float(dt), 1e-9)
    jerk = np.abs(np.diff(hip_vel, prepend=hip_vel[0])) / dt
    w = max(2, int(cfg.post_impact_lookback_frames))
    rmax = np.asarray(
        pd.Series(jerk).rolling(window=w, min_periods=1).max().to_numpy(),
        dtype=float,
    )
    had_impact = rmax > float(cfg.impact_jerk_threshold)
    sref = max(float(cfg.still_speed_ref), 1e-9)
    still = 1.0 - np.clip((np.abs(head_vel) + np.abs(hip_vel)) * 0.5 / sref, 0.0, 1.0)
    raw = np.nan_to_num(still * had_impact.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(np.asarray(raw, dtype=np.float64), 0.0, 1.0)


def _optional_smooth(
    series: pd.Series,
    window: int,
    min_periods: int,
    enabled: bool,
) -> pd.Series:
    if not enabled:
        return series
    return series.rolling(window=window, min_periods=min_periods).mean()


def compute_temporal_features(
    features_df: pd.DataFrame,
    fps: int,
    *,
    config: TemporalFeatureConfig | None = None,
    rolling_window_frames: int | None = None,
    min_periods: int | None = None,
    pre_smooth: bool | None = None,
    pre_smooth_window_frames: int | None = None,
    column_hip_y: str = _DEFAULT_HIP_Y,
    column_head_y: str = _DEFAULT_HEAD_Y,
    column_torso_angle: str = _DEFAULT_TORSO,
    column_low_posture: str = _DEFAULT_LOW,
) -> pd.DataFrame:
    """
    Augment a biomechanical feature table with velocity and rolling temporal metrics.

    **Velocities** (per second): time derivatives of ``hip_vertical_position``,
    ``head_vertical_position``, and ``torso_angle_degrees`` via :func:`numpy.gradient`
    with spacing ``1 / fps``.

    **instability_score**: for each row, the mean of rolling **standard deviations** (within
    ``rolling_window_frames``) of hip y, head y, and torso angle. Higher values indicate
    more frame-to-frame variation in the window (heuristic for shaky / erratic motion).

    **low_posture_duration_frames**: rolling **sum** of the boolean ``is_low_posture`` over
    ``rolling_window_frames`` (number of low-posture frames in that window; not wall-clock
    seconds—divide by ``fps`` if needed).

    Parameters
    ----------
    features_df
        One row per frame, time-ordered. Must include the columns named by the ``column_*``
        parameters (defaults: ``hip_vertical_position``, ``head_vertical_position``,
        ``torso_angle_degrees``, ``is_low_posture``).
    fps
        Video sampling rate in Hz (used only for time scaling of derivatives).
    config
        Optional bundle of window settings. Individual keyword arguments override
        ``config`` when both are provided (explicit kwargs win over ``config``).
    rolling_window_frames
        Window size for rolling std (instability) and rolling sum (low posture). Default 5.
    min_periods
        Pandas ``min_periods`` for rolling operations. Default 1.
    pre_smooth
        If True, smooth input series before differentiation (see :class:`TemporalFeatureConfig`).
    pre_smooth_window_frames
        Window for pre-smoothing when ``pre_smooth`` is True.
    column_hip_y, column_head_y, column_torso_angle, column_low_posture
        Column name overrides if your table uses different labels.

    Returns
    -------
    pd.DataFrame
        A **copy** of ``features_df`` with added columns: ``hip_vertical_velocity``,
        ``head_vertical_velocity``, ``torso_angle_velocity``, ``instability_score``,
        ``low_posture_duration_frames``, ``reaction_delay_score``.

    Raises
    ------
    ValueError
        If required columns are missing, ``fps`` is invalid, or window parameters are invalid.
    """
    cfg = config or TemporalFeatureConfig()
    w = cfg.rolling_window_frames if rolling_window_frames is None else int(rolling_window_frames)
    mp = cfg.min_periods if min_periods is None else int(min_periods)
    do_smooth = cfg.pre_smooth if pre_smooth is None else bool(pre_smooth)
    psw = (
        cfg.pre_smooth_window_frames
        if pre_smooth_window_frames is None
        else int(pre_smooth_window_frames)
    )

    if w < 1 or mp < 1 or psw < 1:
        raise ValueError("Window parameters must be >= 1.")
    if fps <= 0:
        raise ValueError("fps must be a positive integer.")

    required = {column_hip_y, column_head_y, column_torso_angle, column_low_posture}
    if features_df is None or features_df.empty:
        out = features_df.copy() if features_df is not None else pd.DataFrame()
        idx = out.index
        for col in (
            _COL_HIP_VEL,
            _COL_HEAD_VEL,
            _COL_TORSO_VEL,
            _COL_INSTAB,
            _COL_LOW_DUR,
            _COL_REACTION,
        ):
            out[col] = pd.Series(dtype=float, index=idx)
        return out

    missing = [c for c in required if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = features_df.copy()
    dt = _time_step_seconds(fps)

    hip = out[column_hip_y].astype(float)
    head = out[column_head_y].astype(float)
    torso = out[column_torso_angle].astype(float)
    low = out[column_low_posture].fillna(False).astype(bool)

    hip_s = _optional_smooth(hip, psw, min_periods=1, enabled=do_smooth)
    head_s = _optional_smooth(head, psw, min_periods=1, enabled=do_smooth)
    torso_s = _optional_smooth(torso, psw, min_periods=1, enabled=do_smooth)

    out[_COL_HIP_VEL] = _gradient_series_per_second(hip_s, dt)
    out[_COL_HEAD_VEL] = _gradient_series_per_second(head_s, dt)
    out[_COL_TORSO_VEL] = _gradient_series_per_second(torso_s, dt)

    rs_hip = _rolling_std(hip, window=w, min_periods=mp)
    rs_head = _rolling_std(head, window=w, min_periods=mp)
    rs_torso = _rolling_std(torso, window=w, min_periods=mp)
    out[_COL_INSTAB] = (rs_hip + rs_head + rs_torso) / 3.0

    out[_COL_LOW_DUR] = low.astype(np.int8).rolling(window=w, min_periods=mp).sum()

    rdc = cfg.reaction_delay if cfg.reaction_delay is not None else ReactionDelayConfig()
    hva = out[_COL_HIP_VEL].to_numpy(dtype=float, copy=False)
    hdv = out[_COL_HEAD_VEL].to_numpy(dtype=float, copy=False)
    out[_COL_REACTION] = _reaction_delay_score(hva, hdv, dt, rdc)
    return out
