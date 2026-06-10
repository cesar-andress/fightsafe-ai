"""
Convert TapKO-style detector outputs (:mod:`fightsafe_ai.events`) into :class:`SafetyEvent`
rows for the live dashboard (decision-support only — not officiating).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from fightsafe_ai.events.tap_detector import TapCandidateEvent, detect_tap_candidates
from fightsafe_ai.events.vulnerability_detector import (
    VulnerabilityCandidateEvent,
    detect_vulnerability_candidates,
)
from fightsafe_ai.live.event_bus import (
    EventCategory,
    SafetyEvent,
    normalize_level_from_score,
)


# TapKO schema IDs exposed to the live API / dashboard (subset may be produced by pose-only detectors).
TAPKO_EVENT_TYPES: Final[tuple[str, ...]] = (
    "submission_signal.hand_tap",
    "submission_signal.foot_tap",
    "submission_signal.verbal_tap",
    "submission_signal.technical_submission_candidate",
    "extreme_vulnerability.ko_collapse",
    "extreme_vulnerability.no_intelligent_defense",
    "extreme_vulnerability.post_impact_inactivity",
    "extreme_vulnerability.choke_unconsciousness_candidate",
)


def tapko_family_subtype(event_type: str) -> tuple[str, str]:
    """Split ``namespace.subtype`` into family and subtype strings."""
    s = str(event_type).strip()
    if "." not in s:
        return ("unknown", s)
    fam, rest = s.split(".", 1)
    return (fam, rest)


def _category_for_tapko(event_type: str) -> EventCategory:
    fam, _ = tapko_family_subtype(event_type)
    if fam == "submission_signal":
        return EventCategory.SUBMISSION_SIGNAL
    if fam == "extreme_vulnerability":
        return EventCategory.EXTREME_VULNERABILITY
    return EventCategory.UNKNOWN


def _fmt_float(x: Any, nd: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v:.{nd}f}"


def tapko_evidence_summary(
    evidence: dict[str, Any],
    *,
    event_type: str,
    repetition_count: int | None = None,
    candidate_level: str | None = None,
) -> str:
    """
    Short human-readable line for dashboard/API (not a clinical or officiating interpretation).

    Full structured evidence remains in ``metadata["evidence"]``.
    """
    fam, sub = tapko_family_subtype(event_type)
    parts: list[str] = []

    if fam == "submission_signal":
        rc = evidence.get("repetition_count")
        if rc is None:
            rc = repetition_count
        if rc is not None:
            parts.append(f"impulses={int(rc)}")
        mg = evidence.get("mat_gate_fraction")
        if mg is not None:
            parts.append(f"mat_gate={_fmt_float(mg, 2)}")
        am = evidence.get("arms_trapped_mean")
        if am is not None:
            parts.append(f"arms_trapped={_fmt_float(am, 2)}")
        om = evidence.get("opponent_proximity_mean")
        if om is not None:
            parts.append(f"opp_prox={_fmt_float(om, 2)}")
        if sub == "verbal_tap":
            parts.insert(0, "verbal_proxy_requires_audio")
        if sub == "technical_submission_candidate":
            parts.insert(0, "technical_proxy_pose_only")
    elif fam == "extreme_vulnerability":
        if candidate_level:
            parts.append(f"candidate_level={candidate_level}")
        for key, label in (
            ("head_drop_score", "head_drop"),
            ("collapse_score", "collapse"),
            ("inactivity_score", "inactivity"),
            ("guard_loss_score", "guard_loss"),
            ("grounded_score", "grounded"),
            ("choke_context_proxy", "choke_px"),
        ):
            if key in evidence:
                parts.append(f"{label}={_fmt_float(evidence[key], 3)}")
        pip = evidence.get("post_impact_context")
        if pip is not None:
            parts.append(f"post_impact_ctx={bool(pip)}")
    else:
        parts.append(f"type={event_type}")

    if not parts:
        parts.append("evidence_keys=" + ",".join(sorted(evidence.keys())[:8]))

    out = "; ".join(parts)
    return out[:900]


def _uniform_idx(t_sec: float, fps: float, n: int) -> int:
    if n <= 0:
        return 0
    return int(np.clip(round(float(t_sec) * float(fps)), 0, n - 1))


def _media_span(media_times: Sequence[float], i0: int, i1: int) -> tuple[float, float]:
    if not media_times:
        return (0.0, 0.0)
    a = int(np.clip(i0, 0, len(media_times) - 1))
    b = int(np.clip(i1, 0, len(media_times) - 1))
    if a > b:
        a, b = b, a
    return (float(media_times[a]), float(media_times[b]))


def tapko_detectors_to_safety_events(
    *,
    stack_xy: np.ndarray,
    media_times: Sequence[float],
    fps: float,
    timestamp_seconds: float,
    dedup_sigs: set[tuple[str, float, float]],
) -> list[SafetyEvent]:
    """
    Run TapKO-aligned detectors on a pose stack aligned with ``media_times``.

    Parameters
    ----------
    stack_xy
        ``(T, 17, 2)`` normalized image coordinates.
    media_times
        Per-frame media timestamps (seconds), length ``T``.
    dedup_sigs
        Mutable set used to suppress duplicate intervals across strides; keyed by
        rounded ``(event_type, start, end)``.
    """
    if stack_xy.ndim != 3 or stack_xy.shape[1] != 17:
        return []
    t_n = int(stack_xy.shape[0])
    if t_n < 24:
        return []

    mt = list(media_times)[:t_n]
    if len(mt) != t_n:
        return []
    fps_f = float(max(fps, 1e-6))

    tap_ev = detect_tap_candidates(stack_xy, fps_f)
    vuln_ev = detect_vulnerability_candidates(stack_xy, fps_f)

    out: list[SafetyEvent] = []

    def emit_candidate(
        ev: TapCandidateEvent | VulnerabilityCandidateEvent,
        *,
        title_short: str,
    ) -> None:
        i0 = _uniform_idx(float(ev.start_time), fps_f, t_n)
        i1 = _uniform_idx(float(ev.end_time), fps_f, t_n)
        m0, m1 = _media_span(mt, i0, i1)
        sig = (
            str(ev.event_type),
            round(m0, 3),
            round(m1, 3),
        )
        if sig in dedup_sigs:
            return
        dedup_sigs.add(sig)

        et = str(ev.event_type)
        fam, sub = tapko_family_subtype(et)
        score = float(ev.score)
        lvl = normalize_level_from_score(score)

        evidence = dict(ev.evidence) if getattr(ev, "evidence", None) else {}
        rep_ct: int | None = None
        cand_lvl: str | None = None
        if isinstance(ev, TapCandidateEvent):
            rep_ct = int(ev.repetition_count)
        elif isinstance(ev, VulnerabilityCandidateEvent):
            cand_lvl = str(ev.level)

        summary = tapko_evidence_summary(
            evidence,
            event_type=et,
            repetition_count=rep_ct,
            candidate_level=cand_lvl,
        )
        meta: dict[str, Any] = {
            "tapko_family": fam,
            "tapko_subtype": sub,
            "full_event_type": et,
            "evidence": evidence,
            "evidence_summary": summary,
        }
        if isinstance(ev, TapCandidateEvent):
            meta["repetition_count"] = int(ev.repetition_count)
        elif isinstance(ev, VulnerabilityCandidateEvent):
            meta["level"] = str(ev.level)
        expl = str(ev.explanation or "").strip()

        out.append(
            SafetyEvent(
                event_type=et,
                category=_category_for_tapko(et),
                start_time=float(m0),
                end_time=float(m1),
                level=lvl,
                score=score,
                title=title_short,
                description=summary,
                explanation=expl,
                source="tapko.live_detectors",
                last_seen_time=max(float(timestamp_seconds), float(m1)),
                metadata=meta,
                requires_human_confirmation=True,
            )
        )

    title_map: dict[str, str] = {
        "submission_signal.hand_tap": "TapKO · Hand tap (candidate)",
        "submission_signal.foot_tap": "TapKO · Foot tap (candidate)",
        "submission_signal.verbal_tap": "TapKO · Verbal tap (candidate)",
        "submission_signal.technical_submission_candidate": "TapKO · Technical submission (candidate)",
        "extreme_vulnerability.ko_collapse": "TapKO · KO / collapse (candidate)",
        "extreme_vulnerability.no_intelligent_defense": "TapKO · No intelligent defense (candidate)",
        "extreme_vulnerability.post_impact_inactivity": "TapKO · Post-impact inactivity (candidate)",
        "extreme_vulnerability.choke_unconsciousness_candidate": "TapKO · Choke compromise (candidate)",
    }

    for tap_c in tap_ev:
        emit_candidate(
            tap_c,
            title_short=title_map.get(tap_c.event_type, f"TapKO · {tap_c.event_type}"),
        )
    for vuln_c in vuln_ev:
        emit_candidate(
            vuln_c,
            title_short=title_map.get(vuln_c.event_type, f"TapKO · {vuln_c.event_type}"),
        )

    return out


__all__ = [
    "TAPKO_EVENT_TYPES",
    "tapko_detectors_to_safety_events",
    "tapko_evidence_summary",
    "tapko_family_subtype",
]
