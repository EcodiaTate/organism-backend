"""
EcodiaOS — Soma State Vector Constructor

Aggregates raw SomaSignals from the SignalBuffer into a fixed-dimensional
OrganismStateVector at each time window. Each system gets a 7D feature
slice: call_rate, error_rate, mean_latency, latency_variance, success_ratio,
resource_rate, and event_entropy (Shannon entropy over event types).

Missing systems produce zero-valued slices — the vector dimension stays
consistent for downstream manifold computations.

Pure numerical computation. No I/O.
"""

from __future__ import annotations

import math
from collections import defaultdict

import structlog

from systems.soma.types import (
    OrganismStateVector,
    SomaSignal,
    SystemStateSlice,
)

logger = structlog.get_logger("systems.soma.state_vector")


class StateVectorConstructor:
    """Aggregates a window of SomaSignals into an OrganismStateVector.

    Maintains the canonical list of known system_ids so that the vector
    dimension is stable across cycles even when some systems are quiet.
    """

    def __init__(self) -> None:
        # Canonical ordering — grows as new systems are discovered,
        # never shrinks. Ensures flat vectors are dimensionally consistent.
        self._known_systems: list[str] = []
        self._known_set: set[str] = set()

    @property
    def system_order(self) -> list[str]:
        """Current canonical ordering for flat vector construction."""
        return list(self._known_systems)

    def construct(
        self,
        signals: list[SomaSignal],
        window_duration_s: float,
        cycle_number: int = 0,
    ) -> OrganismStateVector:
        """Build an OrganismStateVector from a window of signals.

        Args:
            signals: All SomaSignals within the current time window.
            window_duration_s: Duration of the window in seconds (for rate computation).
            cycle_number: Current theta cycle number.

        Returns:
            OrganismStateVector with per-system feature slices.
        """
        if window_duration_s <= 0.0:
            window_duration_s = 0.15  # default theta cycle

        # Group signals by system_id
        by_system: dict[str, list[SomaSignal]] = defaultdict(list)
        for sig in signals:
            by_system[sig.system_id].append(sig)

        # Register newly discovered systems
        for sid in by_system:
            if sid not in self._known_set:
                self._known_systems.append(sid)
                self._known_set.add(sid)

        # Build per-system slices
        systems: dict[str, SystemStateSlice] = {}
        for sid, sigs in by_system.items():
            systems[sid] = self._aggregate_system(sigs, window_duration_s)

        timestamp = signals[-1].timestamp if signals else 0.0

        return OrganismStateVector(
            timestamp=timestamp,
            cycle_number=cycle_number,
            systems=systems,
        )

    def _aggregate_system(
        self,
        signals: list[SomaSignal],
        window_s: float,
    ) -> SystemStateSlice:
        """Compute the 7D feature slice for a single system's signals."""
        n = len(signals)
        if n == 0:
            return SystemStateSlice()

        # call_rate: events per second
        call_rate = n / window_s

        # error_rate: fraction of signals with status == "error"
        error_count = sum(1 for s in signals if s.status == "error")
        error_rate = error_count / n

        # success_ratio: fraction with status in ("success", "info")
        success_count = sum(1 for s in signals if s.status in ("success", "info"))
        success_ratio = success_count / n

        # mean_latency and latency_variance
        latencies = [s.latency_ms for s in signals if s.latency_ms is not None]
        if latencies:
            mean_lat = sum(latencies) / len(latencies)
            if len(latencies) > 1:
                var_lat = sum((x - mean_lat) ** 2 for x in latencies) / (len(latencies) - 1)
            else:
                var_lat = 0.0
        else:
            mean_lat = 0.0
            var_lat = 0.0

        # resource_rate: sum of absolute resource deltas per second
        total_resource = 0.0
        for s in signals:
            if s.resource_delta:
                total_resource += sum(abs(v) for v in s.resource_delta.values())
        resource_rate = total_resource / window_s

        # event_entropy: Shannon entropy over event type distribution
        event_entropy = self._shannon_entropy(signals)

        return SystemStateSlice(
            call_rate=call_rate,
            error_rate=error_rate,
            mean_latency_ms=mean_lat,
            latency_variance=var_lat,
            success_ratio=success_ratio,
            resource_rate=resource_rate,
            event_entropy=event_entropy,
        )

    @staticmethod
    def _shannon_entropy(signals: list[SomaSignal]) -> float:
        """Compute Shannon entropy over event type keys within signals.

        Uses a combination of source type and payload event_type (if present)
        as the event category for entropy computation.
        """
        n = len(signals)
        if n <= 1:
            return 0.0

        counts: dict[str, int] = defaultdict(int)
        for s in signals:
            # Build a category key from source + payload event_type
            event_key = s.payload.get("event_type", s.source.value)
            key = f"{s.source.value}:{event_key}" if isinstance(event_key, str) else s.source.value
            counts[key] += 1

        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / n
                entropy -= p * math.log2(p)

        return entropy
