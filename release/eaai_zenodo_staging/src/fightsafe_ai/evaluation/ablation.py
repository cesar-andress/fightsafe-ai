"""Ablation study presets and sweep tables (pure data structures, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AblationScenario(StrEnum):
    """
    Named **configurations** for comparing pipeline depth (research, not CLI flags).

    Values describe *what to enable* in an experiment matrix; your runner maps them
    to framework / risk / LLM settings.
    """

    BASELINE_BIOMECHANICS_ONLY = "baseline_biomechanics_only"
    """Features from pose / biomechanics only; no action, anomaly, or fusion add-ons."""

    BIOMECHANICS_ACTION = "biomechanics_action"
    """Biomechanics plus **action** layer signals (e.g. guard, strike proxies)."""

    BIOMECHANICS_ANOMALY = "biomechanics_anomaly"
    """Biomechanics plus **anomaly** layer (fall, inactivity, limb, surrender heuristics)."""

    FULL_FUSION = "full_fusion"
    """Full **multi-signal fusion** (action + anomaly + risk fusion as in your config)."""

    FULL_FUSION_LLM = "full_fusion_llm"
    """As **full fusion**, plus **optional** post-hoc LLM explanation (does not change scores)."""


def ablation_param_template(scenario: AblationScenario) -> dict[str, Any]:
    """
    Suggested **boolean / string** parameters for logging and ``AblationRow.params``.

    Callers should merge with run-specific paths, seeds, and model IDs.
    """
    base: dict[str, Any] = {
        "scenario": str(scenario),
        "use_action_layer": False,
        "use_anomaly_layer": False,
        "use_risk_fusion": False,
        "use_llm_explanation": False,
    }
    if scenario == AblationScenario.BASELINE_BIOMECHANICS_ONLY:
        return {**base, "signal_scope": "biomechanics_only"}
    if scenario == AblationScenario.BIOMECHANICS_ACTION:
        return {
            **base,
            "signal_scope": "biomechanics_plus_action",
            "use_action_layer": True,
            "use_risk_fusion": True,
        }
    if scenario == AblationScenario.BIOMECHANICS_ANOMALY:
        return {
            **base,
            "signal_scope": "biomechanics_plus_anomaly",
            "use_anomaly_layer": True,
            "use_risk_fusion": True,
        }
    if scenario == AblationScenario.FULL_FUSION:
        return {
            **base,
            "signal_scope": "full_multisignal_fusion",
            "use_action_layer": True,
            "use_anomaly_layer": True,
            "use_risk_fusion": True,
        }
    return {
        **base,
        "signal_scope": "full_multisignal_fusion",
        "use_action_layer": True,
        "use_anomaly_layer": True,
        "use_risk_fusion": True,
        "use_llm_explanation": True,
    }


def all_ablation_scenarios() -> tuple[AblationScenario, ...]:
    """Ordered tuple of all enum members (stable for tables)."""
    return (
        AblationScenario.BASELINE_BIOMECHANICS_ONLY,
        AblationScenario.BIOMECHANICS_ACTION,
        AblationScenario.BIOMECHANICS_ANOMALY,
        AblationScenario.FULL_FUSION,
        AblationScenario.FULL_FUSION_LLM,
    )


def make_ablation_row(
    scenario: AblationScenario,
    *,
    metrics: dict[str, float] | None = None,
) -> AblationRow:
    """Convenience: :class:`AblationRow` with :func:`ablation_param_template` filled in."""
    return AblationRow(
        name=str(scenario.value),
        params=ablation_param_template(scenario),
        metrics=dict(metrics) if metrics else {},
    )


@dataclass
class AblationRow:
    """One configuration row in an ablation study (metrics filled by the caller)."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def sort_rows_by_metric(rows: list[AblationRow], key: str) -> list[AblationRow]:
    """Return rows sorted by ``metrics[key]`` **descending**; missing key → -inf sort key."""
    return sorted(
        rows,
        key=lambda r: r.metrics.get(key, float("-inf")),
        reverse=True,
    )
