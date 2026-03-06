"""
EcodiaOS — Soma Signal Buffer

Bounded ring buffer for interoceptive signal ingestion. Accepts Synapse
events (via EventBus subscription) and structured log entries, normalizing
everything into the unified SomaSignal format.

Provides windowed retrieval for the StateVectorConstructor and downstream
analysis engines.

Iron Rules:
  - No allocations on the hot path beyond the ring buffer append.
  - Thread-safe for concurrent injection from the EventBus callback
    (asyncio single-threaded, but guarded against re-entrancy).
  - Dropping oldest signals silently when the buffer is full.
"""

from __future__ import annotations

import contextlib
import time
from collections import deque

import structlog

from systems.soma.types import SignalSource, SomaSignal
from systems.synapse.types import SynapseEvent, SynapseEventType

logger = structlog.get_logger("systems.soma.signal_buffer")

# Map Synapse event types to human-readable status categories
_STATUS_MAP: dict[SynapseEventType, str] = {
    SynapseEventType.SYSTEM_FAILED: "error",
    SynapseEventType.SYSTEM_OVERLOADED: "warning",
    SynapseEventType.SYSTEM_DEGRADED: "warning",
    SynapseEventType.CLOCK_OVERRUN: "warning",
    SynapseEventType.SAFE_MODE_ENTERED: "error",
    SynapseEventType.BENCHMARK_REGRESSION: "warning",
    SynapseEventType.COMPUTE_CAPACITY_EXHAUSTED: "error",
    SynapseEventType.COMPUTE_MIGRATION_FAILED: "error",
    SynapseEventType.MODEL_HOT_SWAP_FAILED: "error",
    SynapseEventType.CATASTROPHIC_FORGETTING_DETECTED: "error",
    SynapseEventType.THREAT_DETECTED: "error",
    SynapseEventType.EMERGENCY_WITHDRAWAL: "error",
    SynapseEventType.CONNECTOR_ERROR: "error",
    SynapseEventType.ENTITY_FORMATION_FAILED: "error",
    SynapseEventType.SKIA_HEARTBEAT_LOST: "error",
}


class SignalBuffer:
    """Bounded ring buffer for interoceptive signals.

    Signals are appended from two sources:
      1. Synapse EventBus subscription (``ingest_synapse_event``)
      2. Direct structured log injection (``ingest_log``)

    Retrieval is by time window: ``get_window(since_monotonic)`` returns
    all signals newer than the given monotonic timestamp and leaves them
    in the buffer (non-destructive read).
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._buffer: deque[SomaSignal] = deque(maxlen=max_size)
        self._total_ingested: int = 0

    # ─── Ingestion ────────────────────────────────────────────────

    def ingest_synapse_event(self, event: SynapseEvent) -> None:
        """Normalize a Synapse event into a SomaSignal and buffer it."""
        status = _STATUS_MAP.get(event.event_type, "info")
        if event.event_type in (
            SynapseEventType.SYSTEM_STARTED,
            SynapseEventType.SYSTEM_RECOVERED,
            SynapseEventType.SAFE_MODE_EXITED,
            SynapseEventType.ACTION_COMPLETED,
        ):
            status = "success"

        latency: float | None = None
        if "latency_ms" in event.data:
            with contextlib.suppress(TypeError, ValueError):
                latency = float(event.data["latency_ms"])
        elif "elapsed_ms" in event.data:
            with contextlib.suppress(TypeError, ValueError):
                latency = float(event.data["elapsed_ms"])

        resource_delta: dict[str, float] | None = None
        if "economic_delta" in event.data:
            with contextlib.suppress(TypeError, ValueError):
                resource_delta = {"economic_usd": float(event.data["economic_delta"])}

        signal = SomaSignal(
            timestamp=time.monotonic(),
            source=SignalSource.SYNAPSE_EVENT,
            system_id=event.source_system,
            function_id=event.data.get("function_id"),
            status=status,
            latency_ms=latency,
            resource_delta=resource_delta,
            payload={
                "event_type": event.event_type.value,
                "event_id": event.id,
                **{
                    k: v
                    for k, v in event.data.items()
                    if k not in ("function_id", "latency_ms", "elapsed_ms", "economic_delta")
                },
            },
        )
        self._buffer.append(signal)
        self._total_ingested += 1

    def ingest_log(
        self,
        system_id: str,
        status: str = "info",
        function_id: str | None = None,
        latency_ms: float | None = None,
        resource_delta: dict[str, float] | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        """Inject a structured log entry as a SomaSignal."""
        signal = SomaSignal(
            timestamp=time.monotonic(),
            source=SignalSource.STRUCTURED_LOG,
            system_id=system_id,
            function_id=function_id,
            status=status,
            latency_ms=latency_ms,
            resource_delta=resource_delta,
            payload=dict(payload) if payload else {},
        )
        self._buffer.append(signal)
        self._total_ingested += 1

    def ingest_health_snapshot(
        self,
        system_id: str,
        status: str,
        latency_ms: float,
        details: dict[str, object],
    ) -> None:
        """Inject a health() snapshot as a SomaSignal."""
        signal = SomaSignal(
            timestamp=time.monotonic(),
            source=SignalSource.SYSTEM_HEALTH,
            system_id=system_id,
            status=status,
            latency_ms=latency_ms,
            payload=dict(details),
        )
        self._buffer.append(signal)
        self._total_ingested += 1

    def ingest_cycle_telemetry(
        self,
        cycle_number: int,
        elapsed_ms: float,
        overrun: bool,
        broadcast_count: int,
    ) -> None:
        """Inject per-theta-cycle telemetry as a SomaSignal."""
        signal = SomaSignal(
            timestamp=time.monotonic(),
            source=SignalSource.CYCLE_TELEMETRY,
            system_id="synapse",
            status="warning" if overrun else "success",
            latency_ms=elapsed_ms,
            payload={
                "cycle_number": cycle_number,
                "overrun": overrun,
                "broadcast_count": broadcast_count,
            },
        )
        self._buffer.append(signal)
        self._total_ingested += 1

    def ingest_resource_metrics(
        self,
        system_id: str,
        cpu_percent: float,
        memory_mb: float,
    ) -> None:
        """Inject CPU/memory resource metrics as a SomaSignal."""
        signal = SomaSignal(
            timestamp=time.monotonic(),
            source=SignalSource.RESOURCE_METRICS,
            system_id=system_id,
            status="info",
            resource_delta={"cpu_percent": cpu_percent, "memory_mb": memory_mb},
            payload={"cpu_percent": cpu_percent, "memory_mb": memory_mb},
        )
        self._buffer.append(signal)
        self._total_ingested += 1

    # ─── Retrieval ────────────────────────────────────────────────

    def get_window(self, since_monotonic: float) -> list[SomaSignal]:
        """Return all signals with timestamp >= since_monotonic.

        Non-destructive: signals remain in the buffer.
        Returns in chronological order.
        """
        result: list[SomaSignal] = []
        for signal in self._buffer:
            if signal.timestamp >= since_monotonic:
                result.append(signal)
        return result

    def get_all(self) -> list[SomaSignal]:
        """Return all buffered signals in chronological order."""
        return list(self._buffer)

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def total_ingested(self) -> int:
        return self._total_ingested
