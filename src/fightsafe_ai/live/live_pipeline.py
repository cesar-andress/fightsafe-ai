"""
Real-time frame processor: pose → features → risk → anomaly signals on a rolling buffer.

**Non-blocking UI:** this class only runs CPU work for one frame per call. For true real-time
targets, run :meth:`LivePipeline.process_frame` in a **background thread** (or process) and pass
frames through a small bounded queue so the display loop never waits on pose inference; see
``live_runner`` worker pattern. Use ``max_infer_hz`` to cap per-second inference cost.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypedDict

import numpy as np

from fightsafe_ai.action.base import ActionSignal, ActionType
from fightsafe_ai.action.temporal_classifier import HeuristicMVPActionDetector, HeuristicMVPConfig
from fightsafe_ai.anomaly.base import AnomalySignal, AnomalyType
from fightsafe_ai.anomaly.fall_detector import FallDetector
from fightsafe_ai.anomaly.inactivity_detector import InactivityDetector
from fightsafe_ai.anomaly.limb_anomaly import LimbAnomalyDetector
from fightsafe_ai.evaluation.boxingvi_strike_detector import detect_strike_events
from fightsafe_ai.features.anomaly import add_limb_anomaly_columns
from fightsafe_ai.features.biomechanics import (
    LandmarkMap,
    build_biomechanical_mvp_dataframe_from_landmark_sequence,
)
from fightsafe_ai.features.temporal import compute_temporal_features
from fightsafe_ai.live.event_bus import (
    EventCategory,
    SafetyEvent,
    SafetyLevel,
    normalize_level_from_score,
)
from fightsafe_ai.live.live_sensitivity import apply_interpretable_sensitivity
from fightsafe_ai.pose.backends import PoseEstimator, create_runtime_pose_estimator
from fightsafe_ai.pose.keypoints import PoseResult
from fightsafe_ai.risk.rules import InterpretableRiskConfig, load_interpretable_risk_config
from fightsafe_ai.risk.scorer import (
    COL_RISK_LEVEL,
    COL_RISK_SCORE,
    COL_TRIGGERED,
    build_combat_mvp_frame_risk,
)


# COCO-17 names (same order as YOLO/RTMPose runtime backends).
_COCO17_POSE_NAMES: Final[tuple[str, ...]] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
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


def _repo_root_configs_risk_rules() -> Path | None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *list(here.parents)]:
        p = parent / "configs" / "risk_rules.yaml"
        if p.is_file():
            return p
    return None


def _map_risk_literal_to_safety(level: str) -> SafetyLevel:
    u = (level or "").strip().upper()
    if u == "LOW":
        return SafetyLevel.INFO
    if u == "MEDIUM":
        return SafetyLevel.WARNING
    if u == "HIGH":
        return SafetyLevel.HIGH
    if u == "CRITICAL":
        return SafetyLevel.CRITICAL
    return SafetyLevel.INFO


def _safety_rank(level: SafetyLevel) -> int:
    return {
        SafetyLevel.INFO: 0,
        SafetyLevel.WARNING: 1,
        SafetyLevel.HIGH: 2,
        SafetyLevel.CRITICAL: 3,
    }[level]


def _triggered_to_list(val: Any) -> list[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    return [str(val)]


def _anomaly_to_safety(conf: float) -> SafetyLevel:
    return SafetyLevel.WARNING if conf < 0.55 else SafetyLevel.HIGH


def _category_from_anomaly_type(t: AnomalyType) -> EventCategory:
    name = t.value
    if name.startswith("FALL"):
        return EventCategory.FALL
    if name.startswith("INACTIVITY"):
        return EventCategory.INACTIVITY
    if name.startswith("SURRENDER"):
        return EventCategory.UNKNOWN
    if "ASYMMETRY" in name or "ANGULAR" in name or "SUPPORT" in name or name.startswith("LIMB"):
        return EventCategory.IMBALANCE
    return EventCategory.UNKNOWN


def _category_from_risk_rules(names: list[str]) -> EventCategory:
    blob = " ".join(names).lower()
    if "high_risk_guard_strike" in blob:
        return EventCategory.STRIKE_IMPACT
    if any(k in blob for k in ("impact", "reaction_delay", "deceleration")):
        return EventCategory.STRIKE_IMPACT
    if any(k in blob for k in ("fall", "downward", "post_fall", "ground", "clear_danger")):
        return EventCategory.FALL
    if "inactivity" in blob or "low_motion" in blob:
        return EventCategory.INACTIVITY
    if any(k in blob for k in ("imbalance", "instability", "limb", "guard", "loss_of_control")):
        return EventCategory.IMBALANCE
    return EventCategory.UNKNOWN


def _friendly_risk_title(category: EventCategory, triggered: list[str]) -> str:
    blob = " ".join(triggered).lower()
    if category == EventCategory.FALL or any(k in blob for k in ("fall", "downward", "ground")):
        return "Possible fall"
    if category == EventCategory.INACTIVITY:
        return "Inactivity"
    if category == EventCategory.IMBALANCE:
        return "Unstable posture"
    if category == EventCategory.STRIKE_IMPACT:
        return "High impact"
    return "Elevated risk"


def _friendly_anomaly_title(t: AnomalyType) -> str:
    name = t.value
    if name.startswith("FALL"):
        return "Possible fall"
    if name.startswith("INACTIVITY"):
        return "Inactivity"
    if name.startswith("LIMB") or "ASYMMETRY" in name or "SUPPORT" in name or "ANGULAR" in name:
        return "Unstable posture"
    if name.startswith("SURRENDER"):
        return "Review posture"
    return "Anomaly signal"


def _friendly_action_title(action_type: ActionType) -> str:
    return {
        ActionType.PUNCH_ACTIVITY: "Punch activity",
        ActionType.KICK_ACTIVITY: "Kick activity",
        ActionType.LOW_GUARD: "Low guard",
        ActionType.TURNED_BACK: "Turned back",
        ActionType.DEFENSIVE_INCAPACITY: "Defensive exposure",
    }.get(action_type, "Action signal")


def _category_for_action(action_type: ActionType) -> EventCategory:
    if action_type in (ActionType.PUNCH_ACTIVITY, ActionType.KICK_ACTIVITY):
        return EventCategory.STRIKE_IMPACT
    if action_type in (
        ActionType.LOW_GUARD,
        ActionType.TURNED_BACK,
        ActionType.DEFENSIVE_INCAPACITY,
    ):
        return EventCategory.IMBALANCE
    return EventCategory.UNKNOWN


def _risk_explanation(raw_level: str, score: float, triggered: list[str]) -> str:
    base = f"Fused interpretable risk band={raw_level}; score={score:.2f}."
    if triggered:
        return f"{base} Cues: {', '.join(triggered[:6])}."
    return f"{base} No interpretable rule above threshold."


def _event_needs_persistence_gate(ev: SafetyEvent) -> bool:
    return ev.event_type.startswith("anomaly.") or ev.event_type.startswith("action.")


class LiveFrameResult(TypedDict):
    """
    One frame's outputs from :meth:`LivePipeline.process_frame`.

    ``risk_level`` is the time-smoothed discrete band; ``risk_score`` is the latest fused scalar;
    ``smoothed_risk_score`` is the mean of the last *K* raw scores (``smoothing_score_frames``).
    """

    risk_level: SafetyLevel
    events: list[SafetyEvent]
    pose: PoseResult | None
    risk_score: float
    smoothed_risk_score: float
    raw_risk_level: str


class _EpisodePersistenceGate:
    """Emit episodic events only after they persist for min_seconds (anti-flicker)."""

    def __init__(self, min_seconds: float) -> None:
        self._min = float(min_seconds)
        self._since: dict[str, float] = {}
        self._emitted: dict[str, bool] = {}

    def filter(self, timestamp_seconds: float, events: list[SafetyEvent]) -> list[SafetyEvent]:
        gated_in = [e for e in events if _event_needs_persistence_gate(e)]
        passthrough = [e for e in events if not _event_needs_persistence_gate(e)]

        keys_now = {self._key(e) for e in gated_in}
        for k in list(self._since.keys()):
            if k not in keys_now:
                del self._since[k]
                self._emitted.pop(k, None)

        out_gated: list[SafetyEvent] = []
        for ev in gated_in:
            k = self._key(ev)
            if k not in self._since:
                self._since[k] = float(timestamp_seconds)
                self._emitted[k] = False
            first = self._since[k]
            if not self._emitted[k] and float(timestamp_seconds) - first >= self._min:
                out_gated.append(ev)
                self._emitted[k] = True

        return passthrough + out_gated

    @staticmethod
    def _key(ev: SafetyEvent) -> str:
        return ev.event_type


@dataclass
class LivePipelineConfig:
    """Configuration for :class:`LivePipeline` (live runner / API thread construct this object)."""

    pose_backend: str = "torch"
    pose_device: str = "auto"
    pose_fp16: bool = False
    pose2d: str | None = None
    onnx_model: str | Path | None = None
    video_fps: float = 30.0
    buffer_seconds: float = 1.5
    smooth_seconds: float = 1.5
    rolling_window: int = 5
    max_infer_hz: float = 12.0
    rules_yaml: Path | None = None
    enable_action: bool = True
    action_config: HeuristicMVPConfig | None = None
    """Minimum time an anomaly/action signal must persist before it is emitted (seconds)."""
    event_min_duration_seconds: float = 0.35
    """Recent smoothed risk samples for diagnostics (frame-equivalent window length)."""
    event_history_frames: int = 18
    gate_anomaly_and_action_events: bool = True
    # Rolling mean of raw risk_score over this many inference steps (see LiveFrameResult).
    smoothing_score_frames: int = 12
    #: Live / dashboard only: ``medium`` = YAML as-is; ``high``/``low`` = bounded multipliers
    #: (see :mod:`fightsafe_ai.live.live_sensitivity`).
    live_sensitivity: Literal["low", "medium", "high"] = "medium"
    #: Heuristic wrist-speed strike segments (same core as ``boxingvi_skeleton_runner``).
    enable_strike_detector: bool = False
    strike_percentile: float = 85.0
    strike_merge_frames: int = 8
    #: Run strike detection every N **pose** updates (buffer-length changes each call).
    strike_infer_stride: int = 2
    #: Minimum poses in the rolling buffer before running detection.
    strike_min_buffer_frames: int = 16
    #: Only emit strikes whose segment **ends** within this many frames of the buffer tail.
    strike_emit_tail_frames: int = 6
    #: TapKO-aligned pose detectors (:mod:`fightsafe_ai.events`) — candidates only, not rulings.
    enable_tapko_detectors: bool = True
    tapko_infer_stride: int = 6
    tapko_min_buffer_frames: int = 48


class LivePipeline:
    """
    Frame-by-frame pipeline: **pose** → **features** → **risk** → **anomalies** (fall, limb,
    inactivity) → optional **action** heuristics; then temporal **smoothing** and **event** list.

    Internal state: deques for landmark/pose history, a score ring for averaged risk, and
    time-based level smoothing (persistent high bands). Does not perform I/O or blocking waits.
    """

    def __init__(self, config: LivePipelineConfig | None = None, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        config
            Preferred entry point for callers.
        kwargs
            If ``config`` is omitted, keyword arguments matching :class:`LivePipelineConfig`
            fields are accepted for backward compatibility.
        """
        if config is None:
            cfg_dict = {
                k: v for k, v in kwargs.items() if k in LivePipelineConfig.__dataclass_fields__
            }
            unknown = set(kwargs) - set(cfg_dict)
            if unknown:
                bad = ", ".join(sorted(unknown))
                raise TypeError(f"LivePipeline: unexpected keyword arguments: {bad}")
            config = LivePipelineConfig(**cfg_dict)
        self._config = config

        self._fps = float(config.video_fps)
        self._rolling = int(config.rolling_window)
        self._smooth_seconds = float(config.smooth_seconds)
        self._rules_yaml = (
            config.rules_yaml if config.rules_yaml is not None else _repo_root_configs_risk_rules()
        )
        self._interpretable_config: InterpretableRiskConfig | None = None
        if (
            str(config.live_sensitivity).lower() != "medium"
            and self._rules_yaml is not None
            and Path(self._rules_yaml).is_file()
        ):
            base = load_interpretable_risk_config(Path(self._rules_yaml))
            self._interpretable_config = apply_interpretable_sensitivity(
                base, config.live_sensitivity
            )

        self._stride = max(1, round(self._fps / max(config.max_infer_hz, 1e-6)))
        buf_len = max(12, int(self._fps * config.buffer_seconds) + 4)
        self._buffer: deque[tuple[str, LandmarkMap]] = deque(maxlen=buf_len)
        self._times: deque[float] = deque(maxlen=buf_len)
        self._poses: deque[PoseResult] = deque(maxlen=buf_len)
        self._seq = 0
        self._auto_frame_index = 0

        self._last_pose: PoseResult | None = None
        self._last_lm: LandmarkMap | None = None
        self._prev_lm_action: LandmarkMap | None = None
        self._prev_ts_action: float | None = None

        self._estimator: PoseEstimator = create_runtime_pose_estimator(
            config.pose_backend,
            device=config.pose_device,
            pose2d=config.pose2d,
            use_fp16=bool(config.pose_fp16),
            model_path=Path(config.onnx_model).expanduser() if config.onnx_model else None,
        )

        self._fall = FallDetector()
        self._limb = LimbAnomalyDetector()
        self._inactivity = InactivityDetector()
        acfg = config.action_config if config.action_config is not None else HeuristicMVPConfig()
        self._action = HeuristicMVPActionDetector(acfg)

        self._level_history: list[tuple[float, SafetyLevel]] = []
        self._risk_level_ring: deque[SafetyLevel] = deque(
            maxlen=max(8, config.event_history_frames)
        )
        self._score_ring: deque[float] = deque(
            maxlen=max(1, int(config.smoothing_score_frames)),
        )

        min_gate = (
            config.event_min_duration_seconds if config.gate_anomaly_and_action_events else 0.0
        )
        self._episode_gate = _EpisodePersistenceGate(min_gate)

        self._strike_step = 0
        self._strike_signatures: set[tuple[float, float]] = set()

        self._tapko_step = 0
        self._tapko_cycle = 0
        self._tapko_dedup: set[tuple[str, float, float]] = set()

        self.last_pose: PoseResult | None = None
        self.smoothed_risk_level: SafetyLevel = SafetyLevel.INFO
        self.raw_risk_level: str = "LOW"
        self.last_risk_score: float = 0.0
        self.last_triggered_rules: list[str] = []

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_seconds: float,
        *,
        frame_index: int | None = None,
    ) -> LiveFrameResult:
        """
        Process one BGR frame at ``timestamp_seconds`` (seconds from clip start or monotonic clock).

        Steps: (1) pose → (2) biomechanical + temporal features → (3) fused risk score →
        (4) anomaly detectors (fall, unstable posture / limb, inactivity) → (5) push buffers →
        (6) smooth discrete risk band + mean score ring → (7) optional action signals + persistence
        gate → (8) return structured result.

        Callers should invoke this off the render hot path when latency matters.
        """
        if frame_index is None:
            frame_index = self._auto_frame_index
            self._auto_frame_index += 1

        run_pose = (frame_index % self._stride == 0) or self._last_lm is None
        pose: PoseResult
        lm: LandmarkMap
        if run_pose:
            pose_new = self._estimator.predict(frame_bgr)
            if not pose_new.keypoints:
                self.last_pose = None
                return self._make_result(
                    risk_level=self.smoothed_risk_level,
                    events=[],
                    pose=None,
                    risk_score=self.last_risk_score,
                    smoothed_risk_score=self._mean_risk_score(),
                    raw_risk_level=self.raw_risk_level,
                )
            lm = {kp.name: (kp.x, kp.y) for kp in pose_new.keypoints}
            self._last_pose = pose_new
            self._last_lm = lm
            pose = pose_new
        else:
            if self._last_pose is None or self._last_lm is None:
                return self._make_result(
                    risk_level=self.smoothed_risk_level,
                    events=[],
                    pose=None,
                    risk_score=self.last_risk_score,
                    smoothed_risk_score=self._mean_risk_score(),
                    raw_risk_level=self.raw_risk_level,
                )
            pose = self._last_pose
            lm = self._last_lm

        fid = f"f{self._seq}"
        self._seq += 1

        self._buffer.append((fid, lm))
        self._times.append(timestamp_seconds)
        self._poses.append(pose)

        seq = list(self._buffer)
        times = list(self._times)

        mvp = build_biomechanical_mvp_dataframe_from_landmark_sequence(
            seq,
            fps=self._fps,
            rolling_window=self._rolling,
        )
        if mvp.empty or len(mvp) == 0:
            self.last_pose = pose
            return self._make_result(
                risk_level=self.smoothed_risk_level,
                events=[],
                pose=pose,
                risk_score=self.last_risk_score,
                smoothed_risk_score=self._mean_risk_score(),
                raw_risk_level=self.raw_risk_level,
            )

        fps_i = int(max(1, round(self._fps)))
        temp = compute_temporal_features(mvp, fps_i, rolling_window_frames=self._rolling)
        with_limb = add_limb_anomaly_columns(temp, float(self._fps))
        poses_list = list(self._poses)
        risk_df = build_combat_mvp_frame_risk(
            with_limb,
            self._fps,
            config=self._interpretable_config,
            rules_yaml=self._rules_yaml if self._interpretable_config is None else None,
            pose_per_frame=poses_list,
        )
        last = risk_df.iloc[-1]
        score = float(last[COL_RISK_SCORE])
        raw_level = str(last[COL_RISK_LEVEL])
        triggered = _triggered_to_list(last.get(COL_TRIGGERED))

        self._score_ring.append(score)
        smoothed_score = self._mean_risk_score()

        raw_safe = _map_risk_literal_to_safety(raw_level)
        smooth = self._smooth_level(timestamp_seconds, raw_safe)
        self._risk_level_ring.append(smooth)

        self.last_pose = pose
        self.smoothed_risk_level = smooth
        self.raw_risk_level = raw_level
        self.last_risk_score = score
        self.last_triggered_rules = triggered

        r_cat = _category_from_risk_rules(triggered)
        risk_title = _friendly_risk_title(r_cat, triggered)
        events: list[SafetyEvent] = [
            SafetyEvent(
                event_type="risk.fused_interpretable",
                category=r_cat,
                start_time=timestamp_seconds,
                end_time=timestamp_seconds,
                level=smooth,
                score=score,
                title=risk_title,
                description=", ".join(triggered[:8]) if triggered else "no rule above epsilon",
                explanation=_risk_explanation(raw_level, score, triggered),
                source="risk.scorer",
            )
        ]
        if self._config.enable_strike_detector:
            events.extend(self._collect_strike_events(float(timestamp_seconds)))

        if self._config.enable_tapko_detectors:
            events.extend(self._collect_tapko_events(float(timestamp_seconds)))

        landmark_dicts = [dict(mp) for _, mp in seq]
        for det in (self._fall, self._limb, self._inactivity):
            for anomaly_sig in det.analyze(times, landmark_dicts, "fighter_0"):
                events.extend(self._signal_to_events(anomaly_sig))

        if self._config.enable_action:
            prev_lm = self._prev_lm_action
            if self._prev_ts_action is None:
                dt = max(1.0 / max(self._fps, 1e-6), 1e-3)
            else:
                dt = max(float(timestamp_seconds) - float(self._prev_ts_action), 1e-3)
            for action_sig in self._action.process_frame(
                float(timestamp_seconds),
                "fighter_0",
                dict(lm),
                dict(prev_lm) if prev_lm is not None else None,
                dt,
            ):
                events.extend(self._action_signal_to_events(action_sig))
            self._prev_lm_action = dict(lm)
            self._prev_ts_action = float(timestamp_seconds)

        if self._config.gate_anomaly_and_action_events:
            events = self._episode_gate.filter(float(timestamp_seconds), events)

        return self._make_result(
            risk_level=smooth,
            events=events,
            pose=pose,
            risk_score=score,
            smoothed_risk_score=smoothed_score,
            raw_risk_level=raw_level,
        )

    def _effective_strike_percentile(self) -> float:
        """``live_sensitivity`` slightly adjusts strike percentile (dashboard / demos only)."""
        base = float(self._config.strike_percentile)
        sens = str(self._config.live_sensitivity).lower()
        if sens == "high":
            return float(max(50.0, min(97.0, base - 10.0)))
        if sens == "low":
            return float(max(50.0, min(97.0, base + 7.0)))
        return base

    def _pose_to_coco17_xy(self, pose: PoseResult | None) -> np.ndarray:
        """Project one pose to fixed COCO-17 ``(17, 2)`` (NaNs when a joint name is missing)."""
        xy = np.full((17, 2), np.nan, dtype=np.float64)
        if pose is None or not pose.keypoints:
            return xy
        by_name = {kp.name: (float(kp.x), float(kp.y)) for kp in pose.keypoints}
        for i, name in enumerate(_COCO17_POSE_NAMES):
            if name in by_name:
                xy[i, 0] = by_name[name][0]
                xy[i, 1] = by_name[name][1]
        return xy

    def _collect_tapko_events(self, timestamp_seconds: float) -> list[SafetyEvent]:
        cfg = self._config
        poses = list(self._poses)
        times = list(self._times)
        n = len(poses)
        if n < int(cfg.tapko_min_buffer_frames):
            return []
        self._tapko_step += 1
        if self._tapko_step % max(1, int(cfg.tapko_infer_stride)) != 0:
            return []
        self._tapko_cycle += 1
        if self._tapko_cycle % 160 == 0:
            self._tapko_dedup.clear()
        stack = np.stack([self._pose_to_coco17_xy(p) for p in poses], axis=0)

        from fightsafe_ai.live.tapko_live_events import tapko_detectors_to_safety_events

        return tapko_detectors_to_safety_events(
            stack_xy=stack,
            media_times=times,
            fps=float(self._fps),
            timestamp_seconds=float(timestamp_seconds),
            dedup_sigs=self._tapko_dedup,
        )

    def _collect_strike_events(self, timestamp_seconds: float) -> list[SafetyEvent]:
        cfg = self._config
        poses = list(self._poses)
        n = len(poses)
        if n < int(cfg.strike_min_buffer_frames):
            return []
        self._strike_step += 1
        if self._strike_step % max(1, int(cfg.strike_infer_stride)) != 0:
            return []
        stack = np.stack([self._pose_to_coco17_xy(p) for p in poses], axis=0)
        pct = self._effective_strike_percentile()
        rows = detect_strike_events(
            stack,
            fps=float(self._fps),
            percentile=pct,
            merge_frames=int(cfg.strike_merge_frames),
        )
        tail = max(0, int(cfg.strike_emit_tail_frames))
        out: list[SafetyEvent] = []
        t_newest = float(timestamp_seconds)
        if self._times:
            t_newest = float(self._times[-1])
        for row in rows:
            try:
                ef_i = int(str(row.get("end_frame", "0")).strip())
            except ValueError:
                continue
            if ef_i < n - 1 - tail:
                continue
            t0 = float(row["start_time"])
            t1 = float(row["end_time"])
            sig = (round(t0, 3), round(t1, 3))
            if sig in self._strike_signatures:
                continue
            if t1 < t_newest - 4.0 / max(self._fps, 1e-6):
                continue
            self._strike_signatures.add(sig)
            out.append(self._strike_dict_to_event(row, sig))
        return out

    def _strike_dict_to_event(self, row: dict[str, Any], sig: tuple[float, float]) -> SafetyEvent:
        """One strike segment → :class:`SafetyEvent` (distinct ``event_type`` for EventBus merge rules)."""
        raw_lvl = str(row.get("level") or row.get("event_level") or "HIGH").strip().upper()
        if raw_lvl == "CRITICAL":
            lvl = SafetyLevel.CRITICAL
        elif raw_lvl == "HIGH":
            lvl = SafetyLevel.HIGH
        elif raw_lvl == "WARNING":
            lvl = SafetyLevel.WARNING
        else:
            lvl = SafetyLevel.HIGH
        score = float(row.get("score") or row.get("max_risk_score") or 0.0)
        title = str(row.get("title") or "Strike candidate")
        desc = str(row.get("description") or "")
        etype = f"boxingvi_strike_s{int(sig[0] * 1e6)}_e{int(sig[1] * 1e6)}"
        return SafetyEvent(
            event_type=etype,
            category=EventCategory.IMPACT,
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            level=lvl,
            score=score,
            title=title[:120],
            description=desc[:500],
            explanation=(
                f"Heuristic wrist-speed strike candidate (BoxingVI-style); "
                f"segment [{row.get('start_frame')},{row.get('end_frame')}] "
                f"score={score:.3f}."
            )[:500],
            source="boxingvi.strike_detector",
        )

    def _mean_risk_score(self) -> float:
        if not self._score_ring:
            return 0.0
        return float(sum(self._score_ring) / len(self._score_ring))

    @staticmethod
    def _make_result(
        *,
        risk_level: SafetyLevel,
        events: list[SafetyEvent],
        pose: PoseResult | None,
        risk_score: float,
        smoothed_risk_score: float,
        raw_risk_level: str,
    ) -> LiveFrameResult:
        return {
            "risk_level": risk_level,
            "events": events,
            "pose": pose,
            "risk_score": risk_score,
            "smoothed_risk_score": smoothed_risk_score,
            "raw_risk_level": raw_risk_level,
        }

    def _smooth_level(self, ts: float, level: SafetyLevel) -> SafetyLevel:
        self._level_history.append((ts, level))
        cutoff = ts - self._smooth_seconds
        self._level_history = [(t, lvl) for t, lvl in self._level_history if t >= cutoff]
        if not self._level_history:
            return level
        return max((lvl for _, lvl in self._level_history), key=_safety_rank)

    def _signal_to_events(self, sig: AnomalySignal) -> list[SafetyEvent]:
        lvl = _anomaly_to_safety(sig.confidence)
        title = _friendly_anomaly_title(sig.anomaly_type)
        desc = ",".join(f"{k}={v}" for k, v in list(sig.evidence.items())[:4])
        ts = float(sig.timestamp)
        ac = _category_from_anomaly_type(sig.anomaly_type)
        expl = f"Heuristic {sig.anomaly_type.value.replace('_', ' ').lower()} (confidence {sig.confidence:.2f})."
        return [
            SafetyEvent(
                event_type=f"anomaly.{sig.anomaly_type.value.lower()}",
                category=ac,
                start_time=ts,
                end_time=ts,
                level=lvl,
                score=float(sig.confidence),
                title=title[:120],
                description=desc[:500],
                explanation=expl[:500],
                source=f"anomaly.{sig.anomaly_type.value.split('_')[0].lower()}",
            )
        ]

    def _action_signal_to_events(self, sig: ActionSignal) -> list[SafetyEvent]:
        lvl = normalize_level_from_score(float(sig.confidence))
        cat = _category_for_action(sig.action_type)
        title = _friendly_action_title(sig.action_type)
        desc = ",".join(f"{k}={v}" for k, v in list(sig.evidence.items())[:4])
        ts = float(sig.timestamp)
        expl = f"Action {sig.action_type.value.replace('_', ' ').lower()} (confidence {sig.confidence:.2f})."
        return [
            SafetyEvent(
                event_type=f"action.{sig.action_type.value.lower()}",
                category=cat,
                start_time=ts,
                end_time=ts,
                level=lvl,
                score=float(sig.confidence),
                title=title[:120],
                description=desc[:500],
                explanation=expl[:500],
                source="action.heuristic_mvp",
            )
        ]


__all__ = ["LiveFrameResult", "LivePipeline", "LivePipelineConfig"]
