"""
Frame-wise **combat MVP** interpretable risk: score (0–1), levels, and ``triggered_rules``.

**Decision-support only (not a medical device):** outputs support human review; they do not
diagnose injury or replace official judgment. See :mod:`fightsafe_ai.risk.rules` and
``configs/risk_rules.yaml`` for tunable, transparent rule components.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fightsafe_ai.exceptions import ConfigurationError
from fightsafe_ai.pose.keypoints import PoseResult
from fightsafe_ai.risk.limb_tier import apply_limb_anomaly_tier_overrides
from fightsafe_ai.risk.rules import (
    ALL_RULE_NAMES,
    RULE_CLEAR_DANGER_FALL,
    RULE_FACING_AWAY,
    RULE_FAST_DOWNWARD,
    RULE_HIGH_RISK_GUARD_STRIKE,
    RULE_INSTABILITY,
    RULE_INTERVENTION_URGENT,
    RULE_LARGE_TORSO,
    RULE_LIMB_ANOMALY,
    RULE_LOSS_OF_CONTROL,
    RULE_LOW_GUARD,
    RULE_LOW_POSTURE,
    RULE_POST_FALL,
    RULE_REACTION_DELAY,
    InterpretableAggregationConfig,
    InterpretableRiskConfig,
    _renormalize_weights,
    build_rule_components,
    load_interpretable_risk_config,
    map_score_to_risk_level,
)
from fightsafe_ai.risk.surrender import (
    COL_SURRENDER_CONFIDENCE,
    SurrenderHeuristicConfig,
    apply_surrender_overrides_to_risk_dataframe,
)
from fightsafe_ai.risk.time_order import sort_frames_add_timestamp


COL_RISK_SCORE: str = "risk_score"
COL_RISK_LEVEL: str = "risk_level"
COL_TRIGGERED: str = "triggered_rules"
COL_TIMESTAMP: str = "timestamp"
COL_FRAME_INDEX: str = "frame_index"
# Optional: stable tracker id from ``fightsafe_ai.tracking`` (multi-fighter / session handles).
COL_FIGHTER_ID: str = "fighter_id"


def _default_project_risk_yaml() -> Path:
    """``configs/risk_rules.yaml`` next to the project root (package parent)."""
    return Path(__file__).resolve().parents[3] / "configs" / "risk_rules.yaml"


def _weight_map(agg: InterpretableAggregationConfig) -> dict[str, float]:
    return {
        RULE_FAST_DOWNWARD: float(agg.weight_fast_downward),
        RULE_LARGE_TORSO: float(agg.weight_large_torso),
        RULE_LOW_POSTURE: float(agg.weight_low_posture),
        RULE_INSTABILITY: float(agg.weight_instability),
        RULE_POST_FALL: float(agg.weight_post_fall),
        RULE_LOW_GUARD: float(agg.weight_low_guard),
        RULE_FACING_AWAY: float(agg.weight_facing_away),
        RULE_REACTION_DELAY: float(agg.weight_reaction_delay),
        RULE_LOSS_OF_CONTROL: float(agg.weight_loss_of_control),
        RULE_CLEAR_DANGER_FALL: float(agg.weight_clear_danger_fall),
        RULE_INTERVENTION_URGENT: float(agg.weight_intervention_urgent),
        RULE_HIGH_RISK_GUARD_STRIKE: float(agg.weight_high_risk_guard_strike),
        RULE_LIMB_ANOMALY: float(agg.weight_limb_anomaly),
    }


def _triggered_list_for_row(
    components: dict[str, np.ndarray],
    row_index: int,
    epsilon: float,
) -> list[str]:
    out: list[str] = []
    for name in ALL_RULE_NAMES:
        v = float(components[name][row_index])
        if v > epsilon:
            out.append(name)
    return out


def compute_interpretable_risk(
    features_df: pd.DataFrame,
    *,
    config: InterpretableRiskConfig | None = None,
    rules_yaml: Path | None = None,
    include_rule_component_columns: bool = False,
    pose_per_frame: list[PoseResult] | None = None,
    surrender_window_frames: int = 22,
    surrender_config: SurrenderHeuristicConfig | None = None,
) -> pd.DataFrame:
    """
    Add ``risk_score`` (0–1), ``risk_level`` (``LOW`` | ``MEDIUM`` | ``HIGH`` | ``CRITICAL``),
    and ``triggered_rules`` (``list[str]`` per row).

    **Inputs (best-effort):** any subset of the columns used in
    :func:`~fightsafe_ai.risk.rules.build_rule_components`. Missing columns disable
    the corresponding rule; weights are renormalized over active rules. If
    :data:`COL_FIGHTER_ID` (``fighter_id``) is present, it is **carried through** to the
    output (for per-fighter reporting); rule components are unchanged and remain **row-wise**.

    **Configuration:** pass ``config``, or pass ``rules_yaml`` to load
    :class:`~fightsafe_ai.risk.rules.InterpretableRiskConfig`, or use defaults
    (in-memory). If ``rules_yaml`` is omitted, ``configs/risk_rules.yaml`` under
    the project root is used when the file exists; otherwise built-in defaults apply.

    Parameters
    ----------
    features_df
        One row per frame, time-ordered, with optional biomechanical + temporal fields.
    config
        Pre-loaded interpretable config; overrides ``rules_yaml`` when set.
    rules_yaml
        Path to YAML (see ``interpretable_aggregation`` and rule blocks in the file).
    include_rule_component_columns
        If True, add one float column per rule (0–1) named ``risk_component_<rule_name>``
        for inspection and debugging.
    pose_per_frame
        Optional list of :class:`~fightsafe_ai.pose.keypoints.PoseResult`, one per row of
        ``features_df`` (same length, time order). When provided, a prototype
        :mod:`fightsafe_ai.risk.surrender` pass can set **CRITICAL** when a tap-out gesture
        is heuristically detected in a short rolling window (see ``surrender_window_frames``).
    surrender_window_frames, surrender_config
        Forwarded to :func:`~fightsafe_ai.risk.surrender.apply_surrender_overrides_to_risk_dataframe`
        when ``pose_per_frame`` is set. Ignored if ``pose_per_frame`` is ``None``.

    Returns
    -------
    pd.DataFrame
        Copy of ``features_df`` with ``risk_score``, ``risk_level``, and ``triggered_rules``.
    """
    if features_df is None:
        raise TypeError("features_df must be a DataFrame, not None.")

    cfg: InterpretableRiskConfig
    if config is not None:
        cfg = config
    else:
        path = rules_yaml
        if path is None:
            p = _default_project_risk_yaml()
            if p.is_file():
                try:
                    cfg = load_interpretable_risk_config(p)
                except (ConfigurationError, OSError, ValueError):
                    cfg = load_interpretable_risk_config(None)
            else:
                cfg = load_interpretable_risk_config(None)
        else:
            cfg = load_interpretable_risk_config(path)

    out = features_df.copy()
    n = len(out)
    if n == 0:
        out[COL_RISK_SCORE] = pd.Series(dtype=float)
        out[COL_RISK_LEVEL] = pd.Series(dtype=object)
        out[COL_TRIGGERED] = pd.Series(dtype=object)
        if pose_per_frame is not None:
            out[COL_SURRENDER_CONFIDENCE] = pd.Series(dtype=float)
        return out

    comp, active = build_rule_components(out, cfg)
    w_raw = _weight_map(cfg.aggregation)
    w = _renormalize_weights(w_raw, active)
    keys = [k for k in ALL_RULE_NAMES if k in w]
    weighted = np.zeros(n, dtype=float)
    for k in keys:
        weighted += w[k] * comp[k]
    score = np.clip(np.nan_to_num(weighted, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    out[COL_RISK_SCORE] = score
    out[COL_RISK_LEVEL] = map_score_to_risk_level(score, cfg.aggregation)

    eps = float(cfg.aggregation.trigger_epsilon)
    triggered: list[list[str]] = [_triggered_list_for_row(comp, i, eps) for i in range(n)]
    out[COL_TRIGGERED] = triggered
    out = apply_limb_anomaly_tier_overrides(out, cfg.aggregation)

    if include_rule_component_columns:
        for name in ALL_RULE_NAMES:
            out[f"risk_component_{name}"] = comp[name]

    if pose_per_frame is not None and len(pose_per_frame) == n:
        out = apply_surrender_overrides_to_risk_dataframe(
            out,
            pose_per_frame,
            window_frames=surrender_window_frames,
            config=surrender_config,
        )
    return out


def build_combat_mvp_frame_risk(
    features_df: pd.DataFrame,
    fps: float,
    *,
    config: InterpretableRiskConfig | None = None,
    rules_yaml: Path | None = None,
    pose_per_frame: list[PoseResult] | None = None,
    surrender_window_frames: int = 22,
    surrender_config: SurrenderHeuristicConfig | None = None,
) -> pd.DataFrame:
    """
    **Combat risk MVP (first version):** one row per sampled frame, time-ordered, with:

    - ``timestamp`` — seconds, ``row_index / fps`` after natural sort by ``frame_id``
    - ``risk_score`` — in ``[0, 1]`` (weighted mean of active rule components)
    - ``risk_level`` — ``LOW`` | ``MEDIUM`` | ``HIGH`` | ``CRITICAL`` (YAML cutoffs)
    - ``triggered_rules`` — list of rule keys with component above ``trigger_epsilon``
    - ``frame_index`` — 0…N-1 in the same order as the stitched preview video / overlay

    Rule keys (``configs/risk_rules.yaml``) include atomic pose/temporal signals and
    **composite** multi-level alert rules (AND-style, explainable in YAML):

    - ``fast_downward_motion``, ``large_torso_angle``, ``prolonged_low_posture``,
      ``post_fall_low_movement``, ``high_instability``
    - ``low_guard``, ``facing_away``, ``reaction_delay_after_impact`` (from
      ``guard_level``, ``facing_away_score``, ``reaction_delay_score``)
    - ``loss_of_control`` — (low guard **or** facing away) with instability (MEDIUM-tier)
    - ``clear_danger_fall`` — large torso angle **and** fast downward motion (HIGH-tier fall proxy)
    - ``high_risk_guard_strike`` — low guard **and** inbound strike proxy (fast wrists / limbs)
    - ``intervention_urgent`` — post-fall stillness with prolonged low posture (CRITICAL-tier)

    Optional :class:`PoseResult` per frame (``pose_per_frame``) enables the prototype
    surrender / tap-out override in :mod:`fightsafe_ai.risk.surrender` (see that module
    for limitations).
    """
    scored = compute_interpretable_risk(
        features_df,
        config=config,
        rules_yaml=rules_yaml,
        pose_per_frame=pose_per_frame,
        surrender_window_frames=surrender_window_frames,
        surrender_config=surrender_config,
    )
    n = len(scored)
    if n == 0:
        out = scored.copy()
        if COL_TIMESTAMP not in out.columns:
            out[COL_TIMESTAMP] = pd.Series(dtype=float)
        if COL_FRAME_INDEX not in out.columns:
            out[COL_FRAME_INDEX] = pd.Series(dtype=int)
        return out

    if "frame_id" not in scored.columns:
        # Best-effort: preserve row order, synthetic ids
        work = scored.copy()
        work["frame_id"] = work.index.map(lambda i: f"row_{i}")
    else:
        work = scored

    ordered = sort_frames_add_timestamp(work, float(fps))
    ordered = ordered.reset_index(drop=True)
    ordered[COL_FRAME_INDEX] = np.arange(n, dtype=int)
    return ordered
