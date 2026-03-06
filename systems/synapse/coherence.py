"""
EcodiaOS — Cross-System Coherence Monitor

IIT-inspired measurement of consciousness quality. Measures how much
information is integrated across the organism's cognitive systems
rather than processed independently.

Four metrics compose the coherence snapshot:

  phi_approximation    — Integration of information across systems
  system_resonance     — How in-sync system responses are (latency uniformity)
  broadcast_diversity  — Entropy of broadcast content sources (topic diversity)
  response_synchrony   — Temporal uniformity of system response latencies

This is the closest computational analogue to Tononi's Integrated
Information Theory (2004) that can be computed online from cycle telemetry.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from systems.synapse.types import (
    CoherenceSnapshot,
    SynapseEvent,
    SynapseEventType,
)

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.synapse.coherence")

# Window size for coherence computation
_COHERENCE_WINDOW: int = 50

# Minimum data points before computing coherence
_MIN_DATA_POINTS: int = 10

# Threshold for emitting a coherence shift event
_SHIFT_THRESHOLD: float = 0.15

# Weight for each component in the composite score
_W_PHI: float = 0.35
_W_RESONANCE: float = 0.25
_W_DIVERSITY: float = 0.20
_W_SYNCHRONY: float = 0.20


class CoherenceMonitor:
    """
    Measures cross-system integration quality using IIT-inspired metrics.

    Fed per-cycle data by SynapseService:
    - record_broadcast(): per-cycle broadcast source/salience data
    - compute(): triggered every N cycles to produce a CoherenceSnapshot

    The composite score represents the organism's "consciousness quality" —
    how well its systems are working together as an integrated whole
    rather than isolated modules.
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._event_bus = event_bus
        self._logger = logger.bind(component="coherence_monitor")

        # Per-cycle data (rolling window)
        self._broadcast_sources: deque[str] = deque(maxlen=_COHERENCE_WINDOW)
        self._broadcast_saliences: deque[float] = deque(maxlen=_COHERENCE_WINDOW)
        self._had_broadcasts: deque[bool] = deque(maxlen=_COHERENCE_WINDOW)

        # Per-cycle system response data (populated externally)
        self._response_latencies: deque[list[float]] = deque(maxlen=_COHERENCE_WINDOW)
        self._responding_system_counts: deque[int] = deque(maxlen=_COHERENCE_WINDOW)
        self._total_registered_systems: int = 0

        # Latest snapshot
        self._latest: CoherenceSnapshot = CoherenceSnapshot()
        self._previous_composite: float = 0.0

        # Metrics
        self._total_computations: int = 0

    # ─── Data Recording ──────────────────────────────────────────────

    def record_broadcast(
        self,
        source: str,
        salience: float,
        had_content: bool,
        response_latencies: list[float] | None = None,
        responding_systems: int = 0,
    ) -> None:
        """
        Record per-cycle broadcast data for coherence computation.

        Called by SynapseService after every theta tick.

        Args:
            source: The broadcast content source identifier (e.g., "api.text_chat")
            salience: The broadcast's composite salience score
            had_content: Whether the cycle produced a broadcast
            response_latencies: Optional list of per-system response times (ms)
            responding_systems: Number of systems that received the broadcast
        """
        self._broadcast_sources.append(source if had_content else "")
        self._broadcast_saliences.append(salience)
        self._had_broadcasts.append(had_content)

        if response_latencies is not None:
            self._response_latencies.append(response_latencies)
        self._responding_system_counts.append(responding_systems)

    def set_total_systems(self, count: int) -> None:
        """Set the total number of registered systems (for phi computation)."""
        self._total_registered_systems = count

    # ─── Computation ─────────────────────────────────────────────────

    async def compute(self) -> CoherenceSnapshot:
        """
        Compute the coherence snapshot from accumulated cycle data.

        Called every N cycles (default 50) by SynapseService.
        Returns the new CoherenceSnapshot.
        """
        n = len(self._broadcast_saliences)
        if n < _MIN_DATA_POINTS:
            return self._latest

        phi = self._compute_phi()
        resonance = self._compute_resonance()
        diversity = self._compute_diversity()
        synchrony = self._compute_synchrony()

        composite = (
            _W_PHI * phi
            + _W_RESONANCE * resonance
            + _W_DIVERSITY * diversity
            + _W_SYNCHRONY * synchrony
        )

        snapshot = CoherenceSnapshot(
            phi_approximation=round(phi, 4),
            system_resonance=round(resonance, 4),
            broadcast_diversity=round(diversity, 4),
            response_synchrony=round(synchrony, 4),
            composite=round(composite, 4),
            window_cycles=n,
        )

        # Check for significant shift
        shift = abs(composite - self._previous_composite)
        if shift > _SHIFT_THRESHOLD and self._total_computations > 0:
            await self._emit_shift(snapshot, shift)

        self._previous_composite = composite
        self._latest = snapshot
        self._total_computations += 1

        return snapshot

    # ─── Component Metrics ───────────────────────────────────────────

    def _compute_phi(self) -> float:
        """
        Approximate integrated information (Φ).

        Computed from:
        - Response rate: fraction of registered systems that respond
        - Response diversity: how many different systems participate
        - Temporal integration: broadcast density (information flowing)

        Higher Φ = more information is being integrated across the whole.
        """
        if self._total_registered_systems == 0:
            return 0.0

        n = len(self._had_broadcasts)
        if n == 0:
            return 0.0

        # Response rate: average responding systems / total systems
        if self._responding_system_counts:
            avg_responding = (
                sum(self._responding_system_counts) / len(self._responding_system_counts)
            )
            response_rate = min(1.0, avg_responding / max(1, self._total_registered_systems))
        else:
            response_rate = 0.0

        # Broadcast density: fraction of cycles with content
        broadcast_count = sum(1 for b in self._had_broadcasts if b)
        density = broadcast_count / n

        # Integration factor: salience-weighted density
        saliences = list(self._broadcast_saliences)
        salience_mean = sum(saliences) / n if saliences else 0.0
        integration = density * (0.5 + 0.5 * salience_mean)

        # Φ ≈ response_rate × integration × diversity_factor
        diversity_factor = self._compute_diversity()
        phi = response_rate * integration * (0.5 + 0.5 * diversity_factor)

        return min(1.0, phi)

    def _compute_resonance(self) -> float:
        """
        System resonance: how in-sync system responses are.

        Low latency variance across systems = high resonance.
        If all systems respond in roughly the same time, they're "resonating."
        """
        if not self._response_latencies:
            return 0.5  # Default moderate resonance when no data

        # Compute mean variance across all recorded latency sets
        variances = []
        for latencies in self._response_latencies:
            if len(latencies) >= 2:
                mean = sum(latencies) / len(latencies)
                var = sum((lat - mean) ** 2 for lat in latencies) / len(latencies)
                variances.append(var)

        if not variances:
            return 0.5

        mean_variance = sum(variances) / len(variances)
        # Normalise: variance=0 → resonance=1.0, high variance → resonance→0
        resonance = 1.0 / (1.0 + math.sqrt(mean_variance) / 50.0)
        return min(1.0, resonance)

    def _compute_diversity(self) -> float:
        """
        Broadcast diversity: Shannon entropy of broadcast sources.

        Are we processing diverse topics or stuck on one input?
        Higher entropy = healthier, more diverse cognitive activity.
        """
        sources = [s for s in self._broadcast_sources if s]
        if not sources:
            return 0.0

        # Count unique sources
        counts: dict[str, int] = {}
        for s in sources:
            counts[s] = counts.get(s, 0) + 1

        total = len(sources)
        unique_count = len(counts)

        if unique_count <= 1:
            return 0.0

        # Shannon entropy (normalised to [0, 1])
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalise by log2(unique_count) to get [0, 1]
        max_entropy = math.log2(unique_count)
        if max_entropy == 0:
            return 0.0

        return min(1.0, entropy / max_entropy)

    def _compute_synchrony(self) -> float:
        """
        Response synchrony: uniformity of response latencies.

        1.0 / (1.0 + std_dev(response_latencies))

        When all systems respond in uniform time, synchrony is high.
        """
        if not self._response_latencies:
            return 0.5

        # Flatten all latencies
        all_latencies = []
        for latencies in self._response_latencies:
            all_latencies.extend(latencies)

        if len(all_latencies) < 2:
            return 0.5

        mean = sum(all_latencies) / len(all_latencies)
        variance = sum((lat - mean) ** 2 for lat in all_latencies) / len(all_latencies)
        std_dev = math.sqrt(variance)

        # Normalise: std_dev=0 → synchrony=1.0
        synchrony = 1.0 / (1.0 + std_dev / 20.0)
        return min(1.0, synchrony)

    # ─── Events ──────────────────────────────────────────────────────

    async def _emit_shift(self, snapshot: CoherenceSnapshot, shift: float) -> None:
        """Emit a coherence shift event when composite changes significantly."""
        if self._event_bus is None:
            return

        direction = "increasing" if snapshot.composite > self._previous_composite else "decreasing"

        self._logger.info(
            "coherence_shift_detected",
            direction=direction,
            shift=round(shift, 4),
            new_composite=snapshot.composite,
            previous=round(self._previous_composite, 4),
        )

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.COHERENCE_SHIFT,
            data={
                "direction": direction,
                "shift": round(shift, 4),
                "composite": snapshot.composite,
                "phi": snapshot.phi_approximation,
                "resonance": snapshot.system_resonance,
                "diversity": snapshot.broadcast_diversity,
                "synchrony": snapshot.response_synchrony,
            },
        ))

    # ─── Accessors ───────────────────────────────────────────────────

    @property
    def latest(self) -> CoherenceSnapshot:
        return self._latest

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_computations": self._total_computations,
            "latest_composite": self._latest.composite,
            "latest_phi": self._latest.phi_approximation,
            "data_points": len(self._broadcast_saliences),
            "registered_systems": self._total_registered_systems,
        }
