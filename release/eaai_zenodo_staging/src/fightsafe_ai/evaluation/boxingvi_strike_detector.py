"""
Heuristic **strike candidates** from multi-person COCO-17 skeleton sequences (no annotations).

Uses wrist-speed peaks — research / decision-support only; not a validated punch detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np


# COCO-17 indices (same layout as yolo/mediapipe exports).
_KP_L_SH: Final[int] = 5
_KP_R_SH: Final[int] = 6
_KP_L_EL: Final[int] = 7
_KP_R_EL: Final[int] = 8
_KP_L_WR: Final[int] = 9
_KP_R_WR: Final[int] = 10
_FOCUS_IDX: Final[tuple[int, ...]] = (
    _KP_L_SH,
    _KP_R_SH,
    _KP_L_EL,
    _KP_R_EL,
    _KP_L_WR,
    _KP_R_WR,
)

_SMOOTH_WIN: Final[int] = 5
_EPS: Final[float] = 1e-12


@dataclass
class StrikeEvent:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    score: float
    category: str = "impact"
    level: str = "HIGH"


def _finite_xy(xy: np.ndarray, i: int) -> bool:
    if i >= xy.shape[0]:
        return False
    x, y = float(xy[i, 0]), float(xy[i, 1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    return abs(x) > _EPS or abs(y) > _EPS


def _count_valid_focus(xy: np.ndarray) -> int:
    return sum(1 for i in _FOCUS_IDX if _finite_xy(xy, i))


def _wrist_speed_mag(
    xy_prev: np.ndarray,
    xy_curr: np.ndarray,
    fps: float,
) -> float:
    """Max of left/right wrist speed magnitude (image plane units × fps)."""
    best = 0.0
    for wi in (_KP_L_WR, _KP_R_WR):
        if not (_finite_xy(xy_prev, wi) and _finite_xy(xy_curr, wi)):
            continue
        dx = float(xy_curr[wi, 0]) - float(xy_prev[wi, 0])
        dy = float(xy_curr[wi, 1]) - float(xy_prev[wi, 1])
        best = max(best, math.hypot(dx, dy) * float(fps))
    return best


def _person_strike_score(
    xy_prev: np.ndarray,
    xy_curr: np.ndarray,
    fps: float,
    min_valid_keypoints: int,
) -> float | None:
    if _count_valid_focus(xy_curr) < min_valid_keypoints:
        return None
    if _count_valid_focus(xy_prev) < min_valid_keypoints:
        return None
    return _wrist_speed_mag(xy_prev, xy_curr, fps)


def _ensure_thpj2(skeleton: np.ndarray) -> np.ndarray:
    """Return ``(T, P, 17, 2)`` float64."""
    a = np.asarray(skeleton)
    if a.ndim == 2 and a.shape[1] in (34, 51):
        a = a.reshape(a.shape[0], 17, -1)
    if a.ndim == 3:
        # (T, 17, C)
        t, j, _c = a.shape
        if j != 17:
            raise ValueError(f"Expected 17 joints per row, got shape {a.shape}")
        xy = a[:, :, :2].astype(np.float64, copy=False)
        return xy.reshape(t, 1, 17, 2)
    if a.ndim == 4:
        t, _p, j, _c = a.shape
        if j != 17:
            raise ValueError(f"Expected 17 joints, got shape {a.shape}")
        return a[..., :2].astype(np.float64, copy=False)
    raise ValueError(f"Unsupported skeleton shape {a.shape}; expect (T,17,C) or (T,P,17,C).")


def _smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    w = int(window) | 1
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.convolve(xp, np.ones(w, dtype=np.float64) / float(w), mode="valid")
    return out.astype(np.float64, copy=False)


def _segments_above_threshold(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive frame indices where ``mask`` is True."""
    if mask.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _merge_segments(gaps: list[tuple[int, int]], merge_frames: int) -> list[tuple[int, int]]:
    if not gaps:
        return []
    gaps = sorted(gaps)
    mg = max(0, int(merge_frames))
    out: list[list[int]] = [list(gaps[0])]
    for s, e in gaps[1:]:
        _ps, pe = out[-1]
        gap = s - pe - 1
        if gap <= mg:
            out[-1][1] = max(pe, e)
        else:
            out.append([s, e])
    return [(int(a[0]), int(a[1])) for a in out]


def detect_strike_events(
    skeleton: np.ndarray,
    fps: float = 30.0,
    percentile: float = 85.0,
    merge_frames: int = 8,
    min_valid_keypoints: int = 5,
) -> list[dict[str, Any]]:
    """
    Peak wrist-speed segments on skeleton-only data; returns JSON-serializable dicts.

    Parameters
    ----------
    skeleton
        Shape ``(T, P, 17, 2)`` or ``(T, 17, 2|3)``. Uses wrists (indices 9, 10) and ranks up to
        two persons per frame by valid keypoints among shoulders/elbows/wrists (indices 5–10).
    """
    arr = _ensure_thpj2(skeleton)
    t_max, p_max, _, _ = arr.shape
    if t_max < 2:
        return []

    fd = float(fps)
    if fd <= 0:
        raise ValueError("fps must be positive.")
    pct = float(percentile)
    if not 0.0 < pct < 100.0:
        raise ValueError("percentile must lie strictly between 0 and 100.")

    raw_score = np.zeros(t_max, dtype=np.float64)
    for ti in range(1, t_max):
        prev_f = arr[ti - 1]
        curr_f = arr[ti]
        ranked_pids: list[tuple[int, int]] = []
        for pid in range(p_max):
            nvalid = _count_valid_focus(curr_f[pid])
            if nvalid >= min_valid_keypoints:
                ranked_pids.append((nvalid, pid))
        ranked_pids.sort(key=lambda x: (-x[0], x[1]))
        best_scores: list[float] = []
        for _nv, pid in ranked_pids[:2]:
            sc = _person_strike_score(prev_f[pid], curr_f[pid], fd, min_valid_keypoints)
            if sc is not None:
                best_scores.append(float(sc))
        if best_scores:
            raw_score[ti] = max(best_scores)

    sm = _smooth_1d(raw_score, _SMOOTH_WIN)
    positive = sm[sm > _EPS]
    if positive.size == 0:
        return []

    thr = float(np.percentile(positive, pct))
    mask = sm > max(thr, _EPS)
    segs = _segments_above_threshold(mask)
    segs = _merge_segments(segs, merge_frames)
    if not segs:
        return []

    pad_w = max(4, len(str(t_max - 1)))
    out: list[dict[str, Any]] = []
    for k, (sf, ef) in enumerate(segs):
        seg_scores = sm[sf : ef + 1]
        peak = float(np.max(seg_scores)) if seg_scores.size else 0.0
        t0 = float(sf) / fd
        t1 = float(ef + 1) / fd
        eid = f"strike_{k + 1:05d}"
        sf_s = str(sf).zfill(pad_w)
        ef_s = str(ef).zfill(pad_w)
        desc = (
            f"Heuristic wrist-speed strike candidate (percentile={pct:.1f} "
            f"on non-zero smoothed scores; merge_gap≤{merge_frames} frames)."
        )
        out.append(
            {
                "event_id": eid,
                "start_frame": sf_s,
                "end_frame": ef_s,
                "start_time": t0,
                "end_time": t1,
                "max_risk_score": peak,
                "score": peak,
                "event_level": "HIGH",
                "level": "HIGH",
                "category": "impact",
                "event_type": "boxingvi.strike_candidate",
                "title": "Strike candidate",
                "description": desc,
            }
        )
    return out


__all__ = ["StrikeEvent", "detect_strike_events"]
