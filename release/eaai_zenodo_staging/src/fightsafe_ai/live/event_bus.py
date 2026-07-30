"""
Stateful live safety event manager: identifiers, open episodes, merge, cooldown, idle close.

Each logical episode has ``event_id``, ``start_time``, ``last_seen_time``, and ``duration``.
While a condition persists within ``merge_gap_seconds``, the same episode is extended; when
observations stop for longer than ``silence_close_seconds``, :meth:`EventBus.tick` marks it
finished. Cooldown suppresses opening a **new** episode of the same ``event_type`` too soon
after the previous one ended.
"""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class SafetyLevel(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EventCategory(StrEnum):
    """Stable CSV/JSON category tokens (``impact`` = strike / collision cues)."""

    FALL = "fall"
    INACTIVITY = "inactivity"
    IMBALANCE = "imbalance"
    IMPACT = "impact"
    STRIKE_IMPACT = "impact"  # alias for IMPACT (same token)
    SUBMISSION_SIGNAL = "submission_signal"
    EXTREME_VULNERABILITY = "extreme_vulnerability"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class SafetyEvent:
    """
    One safety episode. The bus may merge successive observations into the same instance.

    ``event_id`` is assigned when the episode is first stored (empty until then).
    ``is_finished`` becomes true when the episode closes (silence, superseded, or explicit).
    """

    event_type: str
    category: EventCategory
    start_time: float
    end_time: float
    level: SafetyLevel
    score: float
    title: str
    description: str
    explanation: str
    source: str
    last_seen_time: float | None = None
    event_id: str = ""
    is_finished: bool = False
    metadata: dict[str, Any] | None = None
    requires_human_confirmation: bool = False

    def __post_init__(self) -> None:
        if self.last_seen_time is None:
            ls = float(self.end_time)
        else:
            ls = float(self.last_seen_time)
        self.last_seen_time = ls
        self.end_time = max(self.end_time, ls)

    @property
    def duration(self) -> float:
        """Span from ``start_time`` to last observation (open) or close instant (finished)."""
        ls = self.last_seen_time
        assert ls is not None
        ref_end = float(self.end_time) if self.is_finished else float(ls)
        return max(0.0, ref_end - float(self.start_time))

    @property
    def timestamp_seconds(self) -> float:
        """Latest observation instant (compatible with older call sites)."""
        ls = self.last_seen_time
        assert ls is not None
        return float(ls)


def _level_rank(level: SafetyLevel) -> int:
    return {
        SafetyLevel.INFO: 0,
        SafetyLevel.WARNING: 1,
        SafetyLevel.HIGH: 2,
        SafetyLevel.CRITICAL: 3,
    }[level]


@dataclass(frozen=True, slots=True)
class SeverityNormalization:
    """Inclusive lower bounds on ``score`` for each band (see :func:`normalize_level_from_score`)."""

    warning: float = 0.28
    high: float = 0.48
    critical: float = 0.72


def normalize_level_from_score(
    score: float,
    *,
    thresholds: SeverityNormalization | None = None,
) -> SafetyLevel:
    """
    Map a raw score in ``[0, 1]`` to a discrete :class:`SafetyLevel`.

    Bands (default): INFO < WARNING < HIGH < CRITICAL.
    """
    th = thresholds or SeverityNormalization()
    s = float(score)
    if s >= th.critical:
        return SafetyLevel.CRITICAL
    if s >= th.high:
        return SafetyLevel.HIGH
    if s >= th.warning:
        return SafetyLevel.WARNING
    return SafetyLevel.INFO


def _finalize_level_from_score(event: SafetyEvent, thresholds: SeverityNormalization) -> None:
    """Raise severity to the band implied by ``score``, never downgrade explicit operator levels."""
    from_norm = normalize_level_from_score(event.score, thresholds=thresholds)
    event.level = max(from_norm, event.level, key=_level_rank)


class EventBus:
    """
    Stateful manager for live episodes.

    - **Open episodes** live in ``active_events`` (``event_id`` → episode).
    - **At most one open episode per** ``event_type`` (tracked via internal type→id map).
    - **Merge** extends ``last_seen_time`` / duration when the same condition reappears inside
      ``merge_gap_seconds``.
    - **Cooldown** blocks spawning a *new* episode for a type shortly after the prior ended.
    - **Idle close**: call :meth:`tick` each frame with timeline ``now_seconds`` so episodes with
      no refresh within ``silence_close_seconds`` are marked finished.
    """

    def __init__(
        self,
        *,
        max_events: int = 10_000,
        cooldown_seconds: float = 2.5,
        merge_gap_seconds: float = 0.45,
        visual_expire_seconds: float = 14.0,
        silence_close_seconds: float | None = None,
        severity_thresholds: SeverityNormalization | None = None,
        cooldown_by_event_type: dict[str, float] | None = None,
    ) -> None:
        self._events: deque[SafetyEvent] = deque(maxlen=max_events)
        self._cooldown_seconds = float(cooldown_seconds)
        self._merge_gap_seconds = float(merge_gap_seconds)
        self._visual_expire_seconds = float(visual_expire_seconds)
        self._silence_close_seconds = (
            float(silence_close_seconds)
            if silence_close_seconds is not None
            else max(2.0 * self._merge_gap_seconds, 0.6)
        )
        self._severity_thresholds = severity_thresholds or SeverityNormalization()
        self._cooldown_by_event_type = (
            dict(cooldown_by_event_type) if cooldown_by_event_type else None
        )

        self._latest_by_type: dict[str, SafetyEvent] = {}
        self._active_events: dict[str, SafetyEvent] = {}
        self._active_type_to_id: dict[str, str] = {}
        self._next_id = 0

    @property
    def cooldown_seconds(self) -> float:
        return self._cooldown_seconds

    @property
    def merge_gap_seconds(self) -> float:
        return self._merge_gap_seconds

    @property
    def visual_expire_seconds(self) -> float:
        return self._visual_expire_seconds

    @property
    def silence_close_seconds(self) -> float:
        return self._silence_close_seconds

    @property
    def active_events(self) -> dict[str, SafetyEvent]:
        """Snapshot of **open** episodes keyed by ``event_id`` (shallow copy)."""
        return dict(self._active_events)

    def _alloc_event_id(self) -> str:
        self._next_id += 1
        return f"evt-{self._next_id:08d}"

    def _cooldown_for(self, event_type: str) -> float:
        if self._cooldown_by_event_type and event_type in self._cooldown_by_event_type:
            return float(self._cooldown_by_event_type[event_type])
        return self._cooldown_seconds

    def _register_open(self, ev: SafetyEvent) -> None:
        if not ev.event_id:
            ev.event_id = self._alloc_event_id()
        self._active_events[ev.event_id] = ev
        self._active_type_to_id[ev.event_type] = ev.event_id

    def _finish_episode(self, ev: SafetyEvent, *, close_time: float | None = None) -> None:
        if ev.is_finished:
            return
        t = float(close_time if close_time is not None else (ev.last_seen_time or ev.end_time))
        ev.is_finished = True
        ev.end_time = max(ev.end_time, t)
        eid = ev.event_id
        if eid and eid in self._active_events:
            del self._active_events[eid]
        if self._active_type_to_id.get(ev.event_type) == eid:
            del self._active_type_to_id[ev.event_type]

    def _finish_if_open(self, ev: SafetyEvent) -> None:
        if ev.event_id and ev.event_id in self._active_events:
            self._finish_episode(ev, close_time=float(ev.last_seen_time or ev.end_time))

    def _merge_into(self, prev: SafetyEvent, incoming: SafetyEvent) -> None:
        prev.start_time = min(prev.start_time, incoming.start_time)
        pls = prev.last_seen_time if prev.last_seen_time is not None else prev.end_time
        ils = incoming.last_seen_time if incoming.last_seen_time is not None else incoming.end_time
        prev.last_seen_time = max(pls, ils)
        prev.end_time = prev.last_seen_time
        prev.level = max(prev.level, incoming.level, key=_level_rank)
        prev.score = max(prev.score, incoming.score)
        _finalize_level_from_score(prev, self._severity_thresholds)
        if len(incoming.description) > len(prev.description):
            prev.description = incoming.description
        if incoming.explanation:
            prev.explanation = incoming.explanation
        if incoming.title:
            prev.title = incoming.title
        if incoming.metadata:
            if prev.metadata:
                merged = dict(prev.metadata)
                merged.update(incoming.metadata)
                prev.metadata = merged
            else:
                prev.metadata = dict(incoming.metadata)
        if incoming.requires_human_confirmation:
            prev.requires_human_confirmation = True

    def _should_merge(self, prev: SafetyEvent, incoming: SafetyEvent) -> bool:
        """Same-type successive observations close in time → extend episode."""
        ils = incoming.last_seen_time if incoming.last_seen_time is not None else incoming.end_time
        pls = prev.last_seen_time if prev.last_seen_time is not None else prev.end_time
        delta = float(ils) - float(pls)
        return delta <= self._merge_gap_seconds + 1e-9

    def add_event(self, event: SafetyEvent, *, force: bool = False) -> bool:
        """
        Ingest an observation. May merge into the open episode, finish/replace on gap, or skip
        under cooldown. Returns True if state changed (merge, new episode, or first append).
        """
        _finalize_level_from_score(event, self._severity_thresholds)

        prev = self._latest_by_type.get(event.event_type)
        if prev is not None and self._should_merge(prev, event):
            self._merge_into(prev, event)
            return True

        if prev is not None and not self._should_merge(prev, event) and not force:
            prev_seen = prev.last_seen_time if prev.last_seen_time is not None else prev.end_time
            gap_since_close = float(event.start_time) - float(prev_seen)
            cd = self._cooldown_for(event.event_type)
            if gap_since_close > self._merge_gap_seconds + 1e-9 and 0.0 <= gap_since_close < cd:
                return False

        if prev is not None and not self._should_merge(prev, event):
            self._finish_if_open(prev)

        self._events.append(event)
        self._latest_by_type[event.event_type] = event
        self._register_open(event)
        return True

    def tick(self, now_seconds: float) -> None:
        """
        Mark open episodes as **finished** when no refresh occurred within
        ``silence_close_seconds`` of ``now_seconds`` (call once per frame on the media timeline).
        """
        now = float(now_seconds)
        to_close: list[SafetyEvent] = []
        for ev in list(self._active_events.values()):
            ls = ev.last_seen_time if ev.last_seen_time is not None else ev.end_time
            if now - float(ls) > self._silence_close_seconds + 1e-9:
                to_close.append(ev)
        for ev in to_close:
            self._finish_episode(ev, close_time=now)

    def get_active_events(self, *, limit: int = 100) -> list[SafetyEvent]:
        """
        Return **open** episodes (not finished), newest refresh first.

        For TTL-filtered panels use :meth:`get_visible_events`.
        """
        if limit <= 0:
            return []
        rows = sorted(
            self._active_events.values(),
            key=lambda e: float(e.last_seen_time or e.end_time),
            reverse=True,
        )
        return rows[:limit]

    def get_recent_events(self, limit: int = 10) -> list[SafetyEvent]:
        """Most recent stored episodes in append order (oldest first within the slice)."""
        if limit <= 0:
            return []
        return list(self._events)[-limit:]

    def get_visible_events(
        self,
        *,
        now_seconds: float,
        limit: int = 10,
        expire_after_seconds: float | None = None,
    ) -> list[SafetyEvent]:
        """
        Episodes relevant for the UI: ``last_seen_time + expire_after >= now_seconds``.

        Returned newest-first (most recently refreshed first).
        """
        ttl = float(
            expire_after_seconds
            if expire_after_seconds is not None
            else self._visual_expire_seconds
        )
        cutoff = float(now_seconds) - ttl

        def _seen(ev: SafetyEvent) -> float:
            return float(ev.last_seen_time if ev.last_seen_time is not None else ev.end_time)

        visible = [e for e in self._events if _seen(e) >= cutoff]
        visible.sort(key=_seen, reverse=True)
        return visible[:limit]

    def all_events(self) -> list[SafetyEvent]:
        return list(self._events)

    def export_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = [_event_to_jsonable(e) for e in self._events]
        meta: dict[str, Any] = {
            "schema_version": 4,
            "severity_normalization": asdict(self._severity_thresholds),
            "cooldown_seconds": self._cooldown_seconds,
            "cooldown_by_event_type": self._cooldown_by_event_type,
            "merge_gap_seconds": self._merge_gap_seconds,
            "visual_expire_seconds": self._visual_expire_seconds,
            "silence_close_seconds": self._silence_close_seconds,
            "events": payload,
        }
        p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def export_csv(self, path: str | Path) -> None:
        """
        One row per stored episode.

        Columns: ``event_id``, ``timestamp`` (= ``last_seen_time``), ``level``, ``category``,
        ``duration``, ``finished``, ``description``.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "event_id",
            "event_type",
            "start_time",
            "end_time",
            "timestamp",
            "level",
            "category",
            "duration",
            "finished",
            "description",
        ]
        rows = [_event_to_csv_row(e) for e in self._events]
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)


def _event_to_jsonable(event: SafetyEvent) -> dict[str, Any]:
    row: dict[str, Any] = {
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
        row["metadata"] = dict(event.metadata)
    return row


def _event_to_csv_row(event: SafetyEvent) -> dict[str, Any]:
    ls = event.last_seen_time if event.last_seen_time is not None else event.end_time
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "start_time": event.start_time,
        "end_time": event.end_time,
        "timestamp": ls,
        "level": event.level.value,
        "category": event.category.value,
        "duration": round(event.duration, 4),
        "finished": event.is_finished,
        "description": event.description,
    }


__all__ = [
    "EventBus",
    "EventCategory",
    "SafetyEvent",
    "SafetyLevel",
    "SeverityNormalization",
    "normalize_level_from_score",
]
