"""
Facade for producing referee :class:`RefereeAlert` instances from frame-level pipeline output
or a fused :class:`~fightsafe_ai.risk.fusion.RiskDecision`.

**This is a decision-support system, not an automated referee.**
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fightsafe_ai.hci.alerts import RefereeAlert, generate_referee_alert


class RefereeAlertManager:
    """
    Presents per-frame :class:`RefereeAlert` values for HITL review. Does not cache state or
    send commands to timing officials.
    """

    def generate(self, frame_data: Mapping[str, Any]) -> RefereeAlert:
        """Build a :class:`RefereeAlert` from one frame's risk record (dict-like)."""
        return generate_referee_alert(frame_data)

    @staticmethod
    def from_risk_decision(d: Any) -> RefereeAlert:
        """
        Map a :class:`~fightsafe_ai.risk.fusion.RiskDecision` into a referee-facing alert
        (same copy rules as :func:`generate_referee_alert`).
        """
        ex = tuple(getattr(d, "explanation_facts", ()) or ())
        rj = " | ".join(ex) if ex else ""
        rl = getattr(d, "risk_level", None)
        band = str(rl.value) if rl is not None and hasattr(rl, "value") else "LOW"
        return generate_referee_alert(
            {
                "timestamp": float(getattr(d, "timestamp", 0.0)),
                "fighter_id": str(getattr(d, "fighter_id", "0")),
                "risk_level": band,
                "risk_score": float(getattr(d, "risk_score", 0.0)),
                "triggered_signals": list(getattr(d, "triggered_signals", ())),
                "explanation_facts": ex,
                "reason": rj,
            }
        )

    @staticmethod
    def from_series(row: Any) -> RefereeAlert:
        """
        Convenience: accept a :class:`pandas.Series` (uses ``row.to_dict()`` when available).
        """
        if hasattr(row, "to_dict"):
            return generate_referee_alert(row.to_dict())
        return generate_referee_alert(dict(row))
