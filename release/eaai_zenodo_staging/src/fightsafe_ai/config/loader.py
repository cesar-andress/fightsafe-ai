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

YAML helpers for application and risk-rule configuration.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, cast

import yaml

from fightsafe_ai.exceptions import ConfigurationError


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file into a nested dictionary."""
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"Expected mapping at root of YAML: {path}")
    return data


def merge_dicts(
    base: MutableMapping[str, Any], override: MutableMapping[str, Any]
) -> dict[str, Any]:
    """Deep-merge ``override`` into ``base`` (mutates ``base``)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            merge_dicts(
                cast("MutableMapping[str, Any]", base[k]),
                cast("MutableMapping[str, Any]", v),
            )
        else:
            base[k] = v
    return dict(base)


def load_default_paths(
    default_yaml: Path,
    risk_yaml: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load ``configs/default.yaml`` and optionally ``configs/risk_rules.yaml``."""
    app = load_yaml_file(default_yaml)
    risk: dict[str, Any] = {}
    if risk_yaml is not None:
        risk = load_yaml_file(risk_yaml)
    return app, risk
