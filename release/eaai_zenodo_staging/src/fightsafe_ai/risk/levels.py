"""
Canonical **risk band** names for interpretable combat-safety scoring.

These align with :data:`fightsafe_ai.risk.rules.RiskLevel` and reporting / HCI layers.
The fusion engine in :mod:`fightsafe_ai.risk.fusion` uses
:data:`RISK_LEVEL_ORDER` and :func:`max_risk_level` to combine rules.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


# Backward-compatible typing name (see :mod:`fightsafe_ai.risk.rules`)
RiskLevelStr = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class RiskLevelName(StrEnum):
    """Ordered risk tiers (decision-support; not medical or regulatory)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Strictly increasing; index usable as numeric rank
RISK_LEVEL_ORDER: tuple[RiskLevelName, ...] = (
    RiskLevelName.LOW,
    RiskLevelName.MEDIUM,
    RiskLevelName.HIGH,
    RiskLevelName.CRITICAL,
)


def risk_level_rank(name: RiskLevelName) -> int:
    return RISK_LEVEL_ORDER.index(name)


def max_risk_level(*levels: RiskLevelName) -> RiskLevelName:
    """
    Return the most severe of one or more :class:`RiskLevelName` values (fusion / rule max).
    With no args, returns ``LOW``.
    """
    if not levels:
        return RiskLevelName.LOW
    return max(levels, key=risk_level_rank)


def parse_risk_level(value: str | None) -> RiskLevelName | None:
    """Return the enum member for a string, or ``None`` if unknown."""
    if value is None:
        return None
    s = str(value).strip().upper()
    try:
        return RiskLevelName(s)
    except ValueError:
        return None
