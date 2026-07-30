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

Public package exports favor stable, documented entry points for downstream apps.
"""

from fightsafe_ai.__version__ import __version__
from fightsafe_ai.features.biomechanics import compute_pose_features
from fightsafe_ai.pipeline.runner import RunPipelineConfig, RunPipelineResult, run_pipeline
from fightsafe_ai.risk.engine import RiskEngine, detect_risk_events
from fightsafe_ai.risk.models import RiskRuleParams, risk_rules_from_yaml
from fightsafe_ai.visualization.overlay import render_risk_overlay_video


__all__ = [
    "RiskEngine",
    "RiskRuleParams",
    "RunPipelineConfig",
    "RunPipelineResult",
    "__version__",
    "compute_pose_features",
    "detect_risk_events",
    "render_risk_overlay_video",
    "risk_rules_from_yaml",
    "run_pipeline",
]
