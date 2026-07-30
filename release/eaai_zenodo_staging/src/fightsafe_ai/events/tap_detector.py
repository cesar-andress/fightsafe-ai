"""
Interpretable **tap-out candidate** detector from 2D pose streams.

This module does **not** perform official submission detection, refereeing, or promotion rules
enforcement. It emits **visual candidate intervals** only—rhythmic contact proxies from
monocular pose—that **always** require human confirmation before any operational use.

Outputs ``submission_signal.hand_tap`` / ``submission_signal.foot_tap`` **candidates** only:
not official submissions, not referee decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np


# TapKO schema alignment (paper2 / annotation/tapko_schema.py)
EVENT_HAND_TAP: Final[str] = "submission_signal.hand_tap"
EVENT_FOOT_TAP: Final[str] = "submission_signal.foot_tap"

# COCO-17 indices (same layout as BoxingVI / live pipeline).
_I_NOSE = 0
_I_L_SH, _I_R_SH = 5, 6
_I_L_EL, _I_R_EL = 7, 8
_I_L_WR, _I_R_WR = 9, 10
_I_L_HIP, _I_R_HIP = 11, 12
_I_L_ANK, _I_R_ANK = 15, 16

_DISCLAIMER = (
    "Heuristic candidate only—not an official submission signal; requires human confirmation."
)


def _finite_xy(xy: np.ndarray, idx: int) -> bool:
    if idx >= xy.shape[0]:
        return False
    x, y = float(xy[idx, 0]), float(xy[idx, 1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    return abs(x) + abs(y) > 1e-12


def _midpoint(xy: np.ndarray, a: int, b: int) -> tuple[float, float] | None:
    if not (_finite_xy(xy, a) and _finite_xy(xy, b)):
        return None
    return (float(xy[a, 0] + xy[b, 0]) * 0.5, float(xy[a, 1] + xy[b, 1]) * 0.5)


def _ensure_tx17x2(keypoints: np.ndarray, person_index: int) -> np.ndarray:
    """Return ``(T, 17, 2)`` float64 for one person."""
    a = np.asarray(keypoints, dtype=np.float64)
    if a.ndim == 3:
        _, j, _ = a.shape
        if j != 17:
            raise ValueError(f"Expected 17 joints, got shape {a.shape}")
        return a[:, :, :2].copy()
    if a.ndim == 4:
        _, p, j, _ = a.shape
        if j != 17:
            raise ValueError(f"Expected 17 joints, got shape {a.shape}")
        if not (0 <= person_index < p):
            raise IndexError(f"person_index {person_index} out of range for P={p}")
        return a[:, person_index, :, :2].copy()
    raise ValueError(f"Unsupported keypoints shape {a.shape}; use (T,17,C) or (T,P,17,C).")


def _pair_torso_xy(xy_opponent: np.ndarray) -> tuple[float, float] | None:
    """Mid-hips on opponent for wrist--opponent proximity."""
    return _midpoint(xy_opponent, _I_L_HIP, _I_R_HIP)


@dataclass
class TapDetectorConfig:
    """Heuristic thresholds (normalized image coordinates, y grows downward).

    All tunables used by :func:`detect_tap_candidates` live here—there are no hidden constants
    in the scoring path except numerical epsilons (``1e-9``-scale guards).
    """

    min_repetitions: int = 2
    repetition_window_sec: float = 1.5
    pad_frames: int = 4
    smooth_window: int = 3
    # Strength is built from velocity percentiles; gate hand taps to hands-low / mat band.
    velocity_percentile_hand: float = 75.0
    velocity_percentile_foot: float = 75.0
    min_hand_mat_band_ratio: float = 0.55
    """``wrist_mid_y`` must exceed ``shoulder_mid_y + ratio * (foot_y - shoulder_mid_y)``."""
    opponent_touch_gamma: float = 12.0
    """Steepness for exp(-gamma * dist) opponent proximity bonus."""
    opponent_bonus_weight: float = 0.25
    """Scales opponent proximity term inside hand impulse strength."""
    arms_trapped_wrist_shoulder_max: float = 0.14
    """Normalized distance wrists--shoulders below this boosts foot-tap context."""
    arms_trapped_exp_scale: float = 18.0
    """Exponent scale ``exp(-scale * d_ws)`` for wrist--shoulder proximity in trapped score."""
    arms_tucked_elbow_margin_y: float = 0.02
    """If wrist y not above elbow y + margin, trapped tuck weight is reduced."""
    arms_tucked_weight_high: float = 1.0
    arms_tucked_weight_low: float = 0.45
    impulse_smooth_sigma_frames: float = 2.0
    min_score_emit: float = 0.22
    peak_percentile_hand: float = 88.0
    peak_percentile_foot: float = 85.0
    peak_height_fraction_of_gate: float = 0.42
    """Local maxima must exceed ``max(min_score_emit, percentile_gate * this)``."""
    hand_oscillation_weight: float = 0.35
    """Weights wrist vertical oscillation relative to median oscillation in hand impulse."""
    foot_oscillation_weight: float = 0.45
    """Weights ankle oscillation relative to median in foot impulse."""
    foot_trapped_boost_low: float = 0.55
    foot_trapped_boost_high: float = 0.45
    """Foot impulse scale ``low + high * arms_trapped_score``."""
    mat_contact_boost_idle: float = 1.0
    mat_contact_boost_base: float = 0.55
    mat_contact_boost_scale: float = 0.45
    """When ``mat_contact_proxy`` is set: ``base + scale * proxy``."""
    strength_clip: float = 12.0
    score_normalize_divisor: float = 8.0
    """Maps mean interval strength to ``[0, 1]`` via ``mean_strength / divisor``."""
    contact_band_pass_min_fraction: float = 0.5
    """``contact_band_passed`` evidence True if mean mat gate over interval >= this."""
    posting_risk_osc_scale: float = 5.0
    """Larger → ``hand_posting_risk`` drops faster when oscillation is present."""
    scramble_risk_vel_clip: float = 6.0
    """Clip for normalizing wrist velocity in scramble-risk proxy."""


@dataclass(frozen=True)
class TapCandidateEvent:
    """One contiguous **visual** tap candidate interval with audit-friendly evidence.

    Not an official submission outcome: ``requires_human_confirmation`` is always True for
    emitted TapKO candidates.
    """

    event_type: str
    start_time: float
    end_time: float
    score: float
    repetition_count: int
    evidence: dict[str, Any]
    explanation: str
    requires_human_confirmation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "score": self.score,
            "repetition_count": self.repetition_count,
            "evidence": dict(self.evidence),
            "explanation": self.explanation,
            "requires_human_confirmation": self.requires_human_confirmation,
        }


def _smooth_moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64, copy=False)
    w = max(1, int(window))
    pad = w // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    ker = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(xp, ker, mode="valid")


def _velocity_mag(xy: np.ndarray, fps: float, idx_a: int, idx_b: int) -> np.ndarray:
    """Per-frame max speed over two joints (image-plane units * fps)."""
    t = xy.shape[0]
    out = np.zeros(t, dtype=np.float64)
    out[0] = 0.0
    for ti in range(1, t):
        best = 0.0
        for ji in (idx_a, idx_b):
            if not (_finite_xy(xy[ti], ji) and _finite_xy(xy[ti - 1], ji)):
                continue
            dx = float(xy[ti, ji, 0] - xy[ti - 1, ji, 0])
            dy = float(xy[ti, ji, 1] - xy[ti - 1, ji, 1])
            best = max(best, math.hypot(dx, dy) * float(fps))
        out[ti] = best
    return out


def _vertical_acceleration(y_series: np.ndarray, fps: float) -> np.ndarray:
    """Second central difference * fps^2 (oscillation proxy)."""
    t = len(y_series)
    acc = np.zeros(t, dtype=np.float64)
    if t < 3:
        return acc
    for i in range(1, t - 1):
        acc[i] = (float(y_series[i + 1]) - 2 * float(y_series[i]) + float(y_series[i - 1])) * (
            float(fps) ** 2
        )
    return acc


def _percentile_gate(signal: np.ndarray, pct: float) -> float:
    s = signal[np.isfinite(signal)]
    if s.size == 0:
        return 0.0
    return float(np.percentile(s, pct))


def _local_maxima_indices(x: np.ndarray, min_height: float) -> list[int]:
    out: list[int] = []
    if len(x) < 3:
        return out
    for i in range(1, len(x) - 1):
        if x[i] >= min_height and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            out.append(i)
    return out


def _cluster_peaks(
    peak_frames: list[int],
    *,
    min_rep: int,
    window_frames: int,
) -> list[tuple[int, int, int, list[int]]]:
    """
    Greedy clusters: within ``window_frames`` of the **first** peak in a cluster,
    keep peaks; emit ``(first_frame, last_frame, count, peak_list)`` if ``count >= min_rep``.
    """
    if not peak_frames:
        return []
    peaks = sorted(peak_frames)
    clusters: list[tuple[int, int, int, list[int]]] = []
    i = 0
    while i < len(peaks):
        start_f = peaks[i]
        end_idx = i
        while end_idx + 1 < len(peaks) and peaks[end_idx + 1] - start_f <= window_frames:
            end_idx += 1
        chunk = peaks[i : end_idx + 1]
        cnt = len(chunk)
        if cnt >= min_rep:
            clusters.append((chunk[0], chunk[-1], cnt, chunk))
            i = end_idx + 1
        else:
            i += 1
    return clusters


def _hand_mat_band_ok(
    wrist_mid_y: float,
    shoulder_mid_y: float,
    foot_mid_y: float,
    ratio: float,
) -> bool:
    if not (
        math.isfinite(wrist_mid_y) and math.isfinite(shoulder_mid_y) and math.isfinite(foot_mid_y)
    ):
        return False
    span = foot_mid_y - shoulder_mid_y
    if span <= 1e-6:
        return False
    thr = shoulder_mid_y + ratio * span
    return wrist_mid_y >= thr


def _arms_trapped_score(xy: np.ndarray, cfg: TapDetectorConfig) -> float:
    """Heuristic 0--1: wrists close to shoulders + elbows bent inward."""
    lw = _midpoint(xy, _I_L_WR, _I_R_WR)
    ls = _midpoint(xy, _I_L_SH, _I_R_SH)
    le = _midpoint(xy, _I_L_EL, _I_R_EL)
    if lw is None or ls is None or le is None:
        return 0.0
    d_ws = math.hypot(lw[0] - ls[0], lw[1] - ls[1])
    tucked = (
        cfg.arms_tucked_weight_high
        if lw[1] < le[1] + cfg.arms_tucked_elbow_margin_y
        else cfg.arms_tucked_weight_low
    )
    return float(np.clip(math.exp(-cfg.arms_trapped_exp_scale * d_ws) * tucked, 0.0, 1.0))


def _negative_aware_evidence(
    sli: slice,
    *,
    branch: str,
    cfg: TapDetectorConfig,
    mat_gate: np.ndarray,
    trapped: np.ndarray,
    osc_wrist: np.ndarray,
    osc_ankle: np.ndarray,
    wrist_vel_s: np.ndarray,
    ankle_vel_s: np.ndarray,
    hand_thr: float,
    foot_thr: float,
) -> dict[str, Any]:
    """Risk proxies for confusion with hard negatives (posting, scramble, escape)."""
    if branch == "hand_tap":
        osc_m = float(np.mean(osc_wrist[sli]))
        hand_posting_risk = float(np.clip(math.exp(-cfg.posting_risk_osc_scale * osc_m), 0.0, 1.0))
        mg = mat_gate[sli].astype(np.float64)
        vel_norm = np.clip(wrist_vel_s[sli] / (hand_thr + 1e-9), 0.0, cfg.scramble_risk_vel_clip)
        scramble_risk = float(
            np.clip(np.mean((1.0 - mg) * vel_norm) / cfg.scramble_risk_vel_clip, 0.0, 1.0)
        )
    else:
        osc_m = float(np.mean(osc_ankle[sli]))
        hand_posting_risk = float(np.clip(math.exp(-cfg.posting_risk_osc_scale * osc_m), 0.0, 1.0))
        tr = trapped[sli].astype(np.float64)
        vel_norm = np.clip(ankle_vel_s[sli] / (foot_thr + 1e-9), 0.0, cfg.scramble_risk_vel_clip)
        scramble_risk = float(
            np.clip(np.mean((1.0 - tr) * vel_norm) / cfg.scramble_risk_vel_clip, 0.0, 1.0)
        )

    mat_frac = float(np.mean(mat_gate[sli].astype(np.float64)))
    contact_band_passed = bool(mat_frac >= cfg.contact_band_pass_min_fraction)

    return {
        "hand_posting_risk": hand_posting_risk,
        "scramble_risk": scramble_risk,
        "single_contact_rejected": False,
        "contact_band_passed": contact_band_passed,
    }


def detect_tap_candidates(
    keypoints: np.ndarray,
    fps: float,
    *,
    person_index: int = 0,
    opponent_keypoints: np.ndarray | None = None,
    opponent_person_index: int = 1,
    mat_contact_proxy: np.ndarray | None = None,
    tracking_ids: np.ndarray | None = None,
    config: TapDetectorConfig | None = None,
) -> list[TapCandidateEvent]:
    """
    Build **visual** tap-out **candidates** from a single-athlete pose stream.

    This is **not** official submission detection, referee signalling, or rules enforcement.
    Outputs are **candidate intervals** derived from motion proxies only; each row includes
    negative-aware evidence fields and ``requires_human_confirmation=True``.

    Parameters
    ----------
    keypoints:
        ``(T, 17, C)`` or ``(T, P, 17, C)`` with C>=2 (x,y[,conf]). Uses COCO-17 ordering.
    fps:
        Video frame rate (Hz) for time conversion.
    opponent_keypoints:
        Optional second skeleton ``(T,17,C)`` or multi-person same shape as ``keypoints``
        (uses ``opponent_person_index`` when 4D). Enables wrist--opponent proximity bonus.
    mat_contact_proxy:
        Optional ``(T,)`` in ``[0,1]`` boosting confidence when feet/mat audio-end devices
        register contact (defaults to neutral if absent).
    tracking_ids:
        Reserved for future per-frame identity filtering; currently unused.

    Returns
    -------
    list[TapCandidateEvent]
        Sorted by ``start_time``. Each event includes ``event_type``, ``start_time``,
        ``end_time``, ``score``, ``repetition_count``, ``evidence``, ``explanation``, and
        ``requires_human_confirmation=True``. Evidence contains ``hand_posting_risk``,
        ``scramble_risk``, ``single_contact_rejected``, and ``contact_band_passed``.
    """
    _ = tracking_ids  # reserved
    cfg = config or TapDetectorConfig()
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("fps must be positive and finite.")

    xy = _ensure_tx17x2(keypoints, person_index)
    t_n = xy.shape[0]
    if t_n < 5:
        return []

    times = np.arange(t_n, dtype=np.float64) / float(fps)

    wrist_vel = _velocity_mag(xy, fps, _I_L_WR, _I_R_WR)

    ankle_vel = _velocity_mag(xy, fps, _I_L_ANK, _I_R_ANK)

    wrist_mid_y = np.zeros(t_n, dtype=np.float64)
    shoulder_mid_y = np.zeros(t_n, dtype=np.float64)
    foot_mid_y = np.zeros(t_n, dtype=np.float64)
    for ti in range(t_n):
        row = xy[ti]
        w = _midpoint(row, _I_L_WR, _I_R_WR)
        s = _midpoint(row, _I_L_SH, _I_R_SH)
        f = _midpoint(row, _I_L_ANK, _I_R_ANK)
        wrist_mid_y[ti] = w[1] if w else float("nan")
        shoulder_mid_y[ti] = s[1] if s else float("nan")
        foot_mid_y[ti] = f[1] if f else float("nan")

    wrist_vel_s = _smooth_moving_avg(wrist_vel, cfg.smooth_window)
    ankle_vel_s = _smooth_moving_avg(ankle_vel, cfg.smooth_window)

    osc_wrist = np.abs(_vertical_acceleration(wrist_mid_y, fps))
    osc_ankle = np.abs(_vertical_acceleration(foot_mid_y, fps))

    mat_gate = np.array(
        [
            _hand_mat_band_ok(
                wrist_mid_y[i], shoulder_mid_y[i], foot_mid_y[i], cfg.min_hand_mat_band_ratio
            )
            for i in range(t_n)
        ],
        dtype=bool,
    )

    hand_thr = _percentile_gate(wrist_vel_s, cfg.velocity_percentile_hand)
    foot_thr = _percentile_gate(ankle_vel_s, cfg.velocity_percentile_foot)

    mat_boost = np.full(t_n, cfg.mat_contact_boost_idle, dtype=np.float64)
    if mat_contact_proxy is not None:
        mcp = np.asarray(mat_contact_proxy, dtype=np.float64).reshape(-1)
        if mcp.shape[0] != t_n:
            raise ValueError("mat_contact_proxy must have shape (T,) matching keypoints.")
        mat_boost = cfg.mat_contact_boost_base + cfg.mat_contact_boost_scale * np.clip(
            mcp, 0.0, 1.0
        )

    opp_bonus = np.zeros(t_n, dtype=np.float64)
    if opponent_keypoints is not None:
        ox = _ensure_tx17x2(opponent_keypoints, opponent_person_index)
        if ox.shape[0] != t_n:
            raise ValueError("opponent_keypoints must have same T as keypoints.")
        for ti in range(t_n):
            w = _midpoint(xy[ti], _I_L_WR, _I_R_WR)
            ot = _pair_torso_xy(ox[ti])
            if w is None or ot is None:
                continue
            d = math.hypot(w[0] - ot[0], w[1] - ot[1])
            opp_bonus[ti] = math.exp(-cfg.opponent_touch_gamma * d)

    hand_raw = (
        (wrist_vel_s / (hand_thr + 1e-9))
        * (1.0 + cfg.hand_oscillation_weight * osc_wrist / (np.nanmedian(osc_wrist) + 1e-9))
        * mat_boost
        * (1.0 + cfg.opponent_bonus_weight * opp_bonus)
    )
    hand_raw = np.nan_to_num(hand_raw, nan=0.0, posinf=0.0, neginf=0.0)
    hand_raw = np.clip(hand_raw, 0.0, cfg.strength_clip)
    hand_strength = hand_raw * mat_gate.astype(np.float64)

    trapped = np.array([_arms_trapped_score(xy[ti], cfg) for ti in range(t_n)], dtype=np.float64)
    foot_raw = (
        (ankle_vel_s / (foot_thr + 1e-9))
        * (1.0 + cfg.foot_oscillation_weight * osc_ankle / (np.nanmedian(osc_ankle) + 1e-9))
        * (cfg.foot_trapped_boost_low + cfg.foot_trapped_boost_high * trapped)
        * mat_boost
    )
    foot_raw = np.nan_to_num(foot_raw, nan=0.0, posinf=0.0, neginf=0.0)
    foot_strength = np.clip(foot_raw, 0.0, cfg.strength_clip)

    win_frames = max(3, int(cfg.repetition_window_sec * fps))

    events: list[TapCandidateEvent] = []

    def emit_branch(
        strength: np.ndarray,
        thr_pct: float,
        event_type: str,
        branch: str,
        extra_ev: dict[str, Any],
    ) -> None:
        level_at_pct = _percentile_gate(strength, thr_pct)
        min_h = max(cfg.min_score_emit, level_at_pct * cfg.peak_height_fraction_of_gate)
        peaks = _local_maxima_indices(strength, min_h)
        clusters = _cluster_peaks(peaks, min_rep=cfg.min_repetitions, window_frames=win_frames)
        pad = max(0, int(cfg.pad_frames))
        for pf, pl, rc, peak_list in clusters:
            sli = slice(max(0, pf - pad), min(t_n, pl + pad + 1))
            interval_strength = float(np.mean(strength[sli]))
            score = float(
                np.clip(interval_strength / (cfg.score_normalize_divisor + 1e-12), 0.0, 1.0)
            )
            if score < cfg.min_score_emit:
                continue
            st = float(times[max(0, pf - pad)])
            et = float(times[min(t_n - 1, pl + pad)])
            na = _negative_aware_evidence(
                sli,
                branch=branch,
                cfg=cfg,
                mat_gate=mat_gate,
                trapped=trapped,
                osc_wrist=osc_wrist,
                osc_ankle=osc_ankle,
                wrist_vel_s=wrist_vel_s,
                ankle_vel_s=ankle_vel_s,
                hand_thr=float(hand_thr),
                foot_thr=float(foot_thr),
            )
            ev: dict[str, Any] = {
                "branch": branch,
                "repetition_count": int(rc),
                "peak_frames": [int(x) for x in peak_list],
                "velocity_gate_hand": float(hand_thr),
                "velocity_gate_foot": float(foot_thr),
                "mat_gate_fraction": float(np.mean(mat_gate[sli])),
                "mat_contact_boost_mean": float(np.mean(mat_boost[sli])),
                "opponent_proximity_mean": float(np.mean(opp_bonus[sli])),
                "wrist_oscillation_mean": float(np.mean(osc_wrist[sli])),
                "arms_trapped_mean": float(np.mean(trapped[sli])),
                "ankle_oscillation_mean": float(np.mean(osc_ankle[sli])),
            }
            ev.update(na)
            ev.update(extra_ev)
            expl = (
                f"{branch} candidate: {rc} impulses within {cfg.repetition_window_sec:.2f}s; "
                f"score={score:.3f}. {_DISCLAIMER}"
            )
            events.append(
                TapCandidateEvent(
                    event_type=event_type,
                    start_time=st,
                    end_time=et,
                    score=score,
                    repetition_count=int(rc),
                    evidence=ev,
                    explanation=expl,
                    requires_human_confirmation=True,
                )
            )

    emit_branch(
        hand_strength,
        cfg.peak_percentile_hand,
        EVENT_HAND_TAP,
        "hand_tap",
        {"wrist_velocity_burst_proxy": float(np.max(wrist_vel_s))},
    )
    emit_branch(
        foot_strength,
        cfg.peak_percentile_foot,
        EVENT_FOOT_TAP,
        "foot_tap",
        {"ankle_velocity_burst_proxy": float(np.max(ankle_vel_s))},
    )

    events.sort(key=lambda e: e.start_time)
    return events
