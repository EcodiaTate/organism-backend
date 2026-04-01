"""
EcodiaOS - Closure Loop Tracker

Monitors the 6 canonical closure loops defined in primitives/closure.py.
For each loop, tracks:
- Whether the trigger event has ever fired
- Whether the response event has ever fired
- Round-trip latency (trigger -> response)
- Last fire time
- Status: ACTIVE / STALE / NEVER_FIRED
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from primitives.closure import ALL_CLOSURE_LOOPS, ClosureLoopDefinition
from systems.synapse.event_bus import EventBus
from systems.synapse.types import SynapseEvent, SynapseEventType


# A loop is STALE if no trigger has fired in this many seconds
_STALE_THRESHOLD_S = 300  # 5 minutes


@dataclass
class _LoopState:
    """Tracking state for a single closure loop."""

    definition: ClosureLoopDefinition
    trigger_count: int = 0
    response_count: int = 0
    last_trigger_time: float | None = None
    last_response_time: float | None = None
    # Rolling latencies (last 20)
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def status(self) -> str:
        now = time.monotonic()
        if self.trigger_count == 0:
            return "NEVER_FIRED"
        if self.last_trigger_time and (now - self.last_trigger_time) > _STALE_THRESHOLD_S:
            return "STALE"
        return "ACTIVE"

    @property
    def avg_latency_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        return round(sum(self.latencies_ms) / len(self.latencies_ms), 1)

    def record_trigger(self) -> None:
        self.trigger_count += 1
        self.last_trigger_time = time.monotonic()

    def record_response(self) -> None:
        self.response_count += 1
        now = time.monotonic()
        self.last_response_time = now
        if self.last_trigger_time is not None:
            latency = (now - self.last_trigger_time) * 1000
            self.latencies_ms.append(latency)
            if len(self.latencies_ms) > 20:
                self.latencies_ms.pop(0)


class ClosureLoopTracker:
    """
    Tracks closure loop health by subscribing to trigger and response events.
    """

    def __init__(self) -> None:
        self._loops: dict[str, _LoopState] = {}
        for loop_def in ALL_CLOSURE_LOOPS:
            self._loops[loop_def.loop_id] = _LoopState(definition=loop_def)
        self._attached = False

    def attach(self, bus: EventBus) -> None:
        """Subscribe to all trigger and response events for closure loops."""
        if self._attached:
            return

        for loop_id, state in self._loops.items():
            trigger = state.definition.trigger_event
            response = state.definition.response_event

            # Try to resolve event type strings to SynapseEventType enum values
            try:
                trigger_type = SynapseEventType(trigger)
                bus.subscribe(trigger_type, self._make_trigger_handler(loop_id))
            except ValueError:
                pass  # Event type not yet in enum

            try:
                response_type = SynapseEventType(response)
                bus.subscribe(response_type, self._make_response_handler(loop_id))
            except ValueError:
                pass  # Event type not yet in enum

        self._attached = True

    def _make_trigger_handler(self, loop_id: str):  # noqa: ANN202
        async def handler(event: SynapseEvent) -> None:
            self._loops[loop_id].record_trigger()
        return handler

    def _make_response_handler(self, loop_id: str):  # noqa: ANN202
        async def handler(event: SynapseEvent) -> None:
            self._loops[loop_id].record_response()
        return handler

    def snapshot(self) -> list[dict[str, Any]]:
        """Return status of all closure loops."""
        now = time.monotonic()
        results = []
        for loop_id, state in self._loops.items():
            last_trigger_ago = None
            if state.last_trigger_time is not None:
                last_trigger_ago = round(now - state.last_trigger_time, 1)

            results.append({
                "loop_id": loop_id,
                "name": state.definition.name,
                "status": state.status,
                "is_critical": state.definition.is_critical,
                "source": state.definition.source_system.value,
                "sink": state.definition.sink_system.value,
                "trigger_event": state.definition.trigger_event,
                "response_event": state.definition.response_event,
                "trigger_count": state.trigger_count,
                "response_count": state.response_count,
                "last_trigger_ago_s": last_trigger_ago,
                "avg_latency_ms": state.avg_latency_ms,
                "timeout_ms": state.definition.timeout_ms,
            })
        return results
