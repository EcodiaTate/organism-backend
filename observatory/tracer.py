"""
EcodiaOS - Event Tracer

Subscribes to all Synapse events and maintains rolling statistics:
- Per-system emission counts (1m / 5m / 15m windows)
- Per-event-type counts and last-seen timestamps
- Event flow graph (source_system -> event_type)
- Silent system detection
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from systems.synapse.event_bus import EventBus
from systems.synapse.types import SynapseEvent


# Rolling window durations in seconds
_WINDOWS = {"1m": 60, "5m": 300, "15m": 900}


@dataclass
class _RollingCounter:
    """Timestamps of recent events for rolling window counts."""

    timestamps: list[float] = field(default_factory=list)

    def record(self) -> None:
        self.timestamps.append(time.monotonic())

    def count(self, window_s: float) -> int:
        cutoff = time.monotonic() - window_s
        # Lazy eviction of old entries
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0)
        return len(self.timestamps)


class EventTracer:
    """
    Subscribes to every Synapse event and maintains diagnostic metrics.

    Attach to the bus via `tracer.attach(event_bus)`. Does not modify
    event delivery - read-only observation.
    """

    def __init__(self) -> None:
        self._per_system: dict[str, _RollingCounter] = defaultdict(_RollingCounter)
        self._per_type: dict[str, _RollingCounter] = defaultdict(_RollingCounter)
        self._last_seen: dict[str, float] = {}  # event_type -> monotonic time
        self._first_seen: dict[str, float] = {}  # event_type -> monotonic time
        self._flow_edges: dict[str, set[str]] = defaultdict(set)  # system -> {event_types}
        self._total: int = 0
        self._start_time: float = time.monotonic()
        self._attached = False

    def attach(self, bus: EventBus) -> None:
        """Subscribe to all events on the bus."""
        if self._attached:
            return
        bus.subscribe_all(self._on_event)
        self._attached = True

    async def _on_event(self, event: SynapseEvent) -> None:
        """Handler called for every non-high-frequency event."""
        self._total += 1
        now = time.monotonic()
        source = event.source_system or "unknown"
        etype = event.event_type.value

        self._per_system[source].record()
        self._per_type[etype].record()
        self._last_seen[etype] = now
        if etype not in self._first_seen:
            self._first_seen[etype] = now
        self._flow_edges[source].add(etype)

    def snapshot(self) -> dict[str, Any]:
        """Return full diagnostic snapshot."""
        now = time.monotonic()
        uptime_s = now - self._start_time

        # Per-system rolling counts
        systems: dict[str, dict[str, int]] = {}
        for sys_id, counter in sorted(self._per_system.items()):
            systems[sys_id] = {
                label: counter.count(window_s)
                for label, window_s in _WINDOWS.items()
            }

        # Per-type counts + last seen
        types: dict[str, dict[str, Any]] = {}
        for etype, counter in sorted(self._per_type.items()):
            types[etype] = {
                "counts": {
                    label: counter.count(window_s)
                    for label, window_s in _WINDOWS.items()
                },
                "last_seen_ago_s": round(now - self._last_seen.get(etype, now), 1),
            }

        # Top emitters (by 5m window)
        top_emitters = sorted(
            systems.items(),
            key=lambda kv: kv[1].get("5m", 0),
            reverse=True,
        )[:10]

        return {
            "uptime_s": round(uptime_s, 1),
            "total_events": self._total,
            "events_per_second": round(self._total / max(uptime_s, 1), 2),
            "unique_types_seen": len(self._per_type),
            "unique_systems_seen": len(self._per_system),
            "top_emitters": [
                {"system": sys_id, **counts}
                for sys_id, counts in top_emitters
            ],
            "per_system": systems,
            "per_type": types,
        }

    def flow_graph(self) -> dict[str, list[str]]:
        """Return the event flow graph: system -> [event_types emitted]."""
        return {
            sys_id: sorted(etypes)
            for sys_id, etypes in sorted(self._flow_edges.items())
        }

    def silent_systems(self, known_systems: list[str]) -> list[str]:
        """Return systems that have never emitted an event."""
        return [s for s in known_systems if s not in self._per_system]
