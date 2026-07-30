"""JSON helpers for API exposure (core domain types stay in :mod:`fightsafe_ai.live.event_bus`)."""

from __future__ import annotations

from typing import Any

from fightsafe_ai.live.event_bus import SafetyEvent


def safety_event_to_json(event: SafetyEvent) -> dict[str, Any]:
    """Serialize a :class:`~fightsafe_ai.live.event_bus.SafetyEvent` for HTTP/WebSocket payloads."""
    out: dict[str, Any] = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "category": event.category.value,
        "start_time": event.start_time,
        "end_time": event.end_time,
        "last_seen_time": event.last_seen_time,
        "duration": round(event.duration, 6),
        "is_finished": event.is_finished,
        "level": event.level.value,
        "score": event.score,
        "title": event.title,
        "description": event.description,
        "explanation": event.explanation,
        "source": event.source,
        "timestamp_seconds": event.timestamp_seconds,
        "requires_human_confirmation": bool(event.requires_human_confirmation),
    }
    if event.metadata:
        out["metadata"] = dict(event.metadata)
    return out


__all__ = ["safety_event_to_json"]
