"""
EcodiaOS - Soma Healing Verifier

After a repair mutation is applied by Thymos, the HealingVerifier monitors
the organism for N cycles to determine whether the repair actually helped.

Tracking signals:
  - Fisher geodesic deviation trend (are we moving back toward baseline?)
  - Topological feature changes (are breaches/fractures healing?)
  - Causal flow anomaly count (are unexpected influences resolving?)
  - Lyapunov exponent trend (is dynamical stability improving?)

Classification:
  HEALED      - signals returning to baseline across all tracked metrics
  PARTIAL     - some metrics improving, others stagnant
  INEFFECTIVE - no change within the monitoring window
  IATROGENIC  - repair made things worse → trigger rollback

Dependencies: numpy. No LLM, no DB, no network.
"""

from __future__ import annotations

import enum
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from systems.soma.service import SomaService

logger = structlog.get_logger("systems.soma.healing")


class HealingOutcome(enum.StrEnum):
    """Classification of post-repair healing trajectory."""

    HEALED = "healed"
    PARTIAL = "partial"
    INEFFECTIVE = "ineffective"
    IATROGENIC = "iatrogenic"


@dataclass
class HealingSnapshot:
    """Single observation during the healing monitoring window."""

    cycle: int
    timestamp: float
    geodesic_deviation: float | None
    topological_health: float | None
    causal_anomaly_count: int
    max_lyapunov: float | None
    coherence_signal: float


@dataclass
class HealingReport:
    """Result of a complete healing verification."""

    repair_id: str
    outcome: HealingOutcome
    monitoring_cycles: int
    baseline_snapshot: HealingSnapshot
    final_snapshot: HealingSnapshot
    trajectory: list[HealingSnapshot] = field(default_factory=list)

    # Per-metric verdicts
    geodesic_trend: str = "unknown"  # "improving" | "stable" | "worsening"
    topology_trend: str = "unknown"
    causal_trend: str = "unknown"
    lyapunov_trend: str = "unknown"
    coherence_trend: str = "unknown"

    # Rollback recommended?
    rollback_recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_id": self.repair_id,
            "outcome": self.outcome.value,
            "monitoring_cycles": self.monitoring_cycles,
            "geodesic_trend": self.geodesic_trend,
            "topology_trend": self.topology_trend,
            "causal_trend": self.causal_trend,
            "lyapunov_trend": self.lyapunov_trend,
            "coherence_trend": self.coherence_trend,
            "rollback_recommended": self.rollback_recommended,
        }


# Thresholds for trend classification
_IMPROVING_THRESHOLD: float = -0.1  # 10% improvement over window
_WORSENING_THRESHOLD: float = 0.15  # 15% worsening triggers concern


class HealingVerifier:
    """
    Monitor the organism after a repair mutation for N cycles and classify
    the healing trajectory.

    Usage:
        verifier = HealingVerifier(soma)
        verifier.begin_monitoring("repair_123")
        # ... N cycles pass, call tick() each cycle ...
        if verifier.is_complete:
            report = verifier.finalize()
    """

    def __init__(
        self,
        soma: SomaService,
        monitoring_cycles: int = 100,
    ) -> None:
        self._soma = soma
        self._monitoring_cycles = monitoring_cycles

        # Active monitoring state
        self._active_repair_id: str | None = None
        self._baseline: HealingSnapshot | None = None
        self._trajectory: deque[HealingSnapshot] = deque(maxlen=monitoring_cycles)
        self._ticks: int = 0

    @property
    def is_monitoring(self) -> bool:
        return self._active_repair_id is not None

    @property
    def is_complete(self) -> bool:
        return self.is_monitoring and self._ticks >= self._monitoring_cycles

    @property
    def active_repair_id(self) -> str | None:
        return self._active_repair_id

    def begin_monitoring(self, repair_id: str) -> None:
        """Start monitoring after a repair mutation is applied."""
        self._active_repair_id = repair_id
        self._ticks = 0
        self._trajectory.clear()
        self._baseline = self._take_snapshot()
        logger.info(
            "healing_monitoring_started",
            repair_id=repair_id,
            monitoring_cycles=self._monitoring_cycles,
        )

    def tick(self) -> None:
        """Called every theta cycle while monitoring is active."""
        if not self.is_monitoring:
            return
        self._ticks += 1
        snapshot = self._take_snapshot()
        self._trajectory.append(snapshot)

    def finalize(self) -> HealingReport:
        """Classify the healing trajectory and produce a report."""
        if not self.is_monitoring or self._baseline is None:
            raise RuntimeError("No active healing monitoring to finalize")

        repair_id = self._active_repair_id or "unknown"
        final = self._take_snapshot()
        trajectory = list(self._trajectory)

        # Classify each metric's trend
        geo_trend = self._classify_geodesic(trajectory)
        topo_trend = self._classify_topology(trajectory)
        causal_trend = self._classify_causal(trajectory)
        lyap_trend = self._classify_lyapunov(trajectory)
        coh_trend = self._classify_coherence(trajectory)

        # Overall outcome
        trends = [geo_trend, topo_trend, causal_trend, lyap_trend, coh_trend]
        improving = sum(1 for t in trends if t == "improving")
        worsening = sum(1 for t in trends if t == "worsening")

        if worsening >= 2:
            outcome = HealingOutcome.IATROGENIC
        elif improving == 0 and worsening == 0:
            outcome = HealingOutcome.INEFFECTIVE
        elif improving >= 3:
            outcome = HealingOutcome.HEALED
        else:
            outcome = HealingOutcome.PARTIAL

        report = HealingReport(
            repair_id=repair_id,
            outcome=outcome,
            monitoring_cycles=self._ticks,
            baseline_snapshot=self._baseline,
            final_snapshot=final,
            trajectory=trajectory,
            geodesic_trend=geo_trend,
            topology_trend=topo_trend,
            causal_trend=causal_trend,
            lyapunov_trend=lyap_trend,
            coherence_trend=coh_trend,
            rollback_recommended=(outcome == HealingOutcome.IATROGENIC),
        )

        logger.info(
            "healing_verification_complete",
            repair_id=repair_id,
            outcome=outcome.value,
            improving=improving,
            worsening=worsening,
        )

        # Reset monitoring state
        self._active_repair_id = None
        self._baseline = None
        self._trajectory.clear()
        self._ticks = 0

        return report

    # ─── Snapshot ────────────────────────────────────────────────

    def _take_snapshot(self) -> HealingSnapshot:
        """Capture current analysis state from Soma."""
        geodesic = None
        dev = self._soma._last_geodesic_deviation  # noqa: SLF001
        if dev is not None:
            geodesic = dev.scalar

        topo = None
        pd = self._soma._last_persistence_diagnosis  # noqa: SLF001
        if pd is not None:
            topo = pd.topological_health

        causal_count = 0
        cf = self._soma._last_causal_flow_map  # noqa: SLF001
        if cf is not None:
            causal_count = (
                len(cf.unexpected_influences)
                + len(cf.missing_influences)
                + len(cf.reversed_influences)
            )

        max_lyap = None
        psr = self._soma._last_phase_space_report  # noqa: SLF001
        if psr is not None and psr.diagnoses:
            lyaps = [d.largest_lyapunov for d in psr.diagnoses.values()]
            max_lyap = max(lyaps) if lyaps else None

        return HealingSnapshot(
            cycle=self._soma.cycle_count,
            timestamp=time.monotonic(),
            geodesic_deviation=geodesic,
            topological_health=topo,
            causal_anomaly_count=causal_count,
            max_lyapunov=max_lyap,
            coherence_signal=self._soma.coherence_signal,
        )

    # ─── Trend Classification ────────────────────────────────────

    @staticmethod
    def _trend_from_series(values: list[float]) -> str:
        """Classify trend from a time series of scalar values."""
        if len(values) < 3:
            return "unknown"
        arr = np.array(values, dtype=np.float64)
        # Normalize by initial value to get relative change
        initial = arr[0] if abs(arr[0]) > 1e-9 else 1.0
        relative_change = (arr[-1] - arr[0]) / abs(initial)

        if relative_change < _IMPROVING_THRESHOLD:
            return "improving"
        if relative_change > _WORSENING_THRESHOLD:
            return "worsening"
        return "stable"

    def _classify_geodesic(self, trajectory: list[HealingSnapshot]) -> str:
        """Lower geodesic deviation = improving."""
        vals = [s.geodesic_deviation for s in trajectory if s.geodesic_deviation is not None]
        return self._trend_from_series(vals) if vals else "unknown"

    def _classify_topology(self, trajectory: list[HealingSnapshot]) -> str:
        """Lower topological health score = improving (closer to baseline)."""
        vals = [s.topological_health for s in trajectory if s.topological_health is not None]
        return self._trend_from_series(vals) if vals else "unknown"

    def _classify_causal(self, trajectory: list[HealingSnapshot]) -> str:
        """Fewer causal anomalies = improving."""
        vals = [float(s.causal_anomaly_count) for s in trajectory]
        return self._trend_from_series(vals) if vals else "unknown"

    def _classify_lyapunov(self, trajectory: list[HealingSnapshot]) -> str:
        """Lower (more negative) Lyapunov = improving (more stable)."""
        vals = [s.max_lyapunov for s in trajectory if s.max_lyapunov is not None]
        return self._trend_from_series(vals) if vals else "unknown"

    def _classify_coherence(self, trajectory: list[HealingSnapshot]) -> str:
        """Higher coherence = improving. Invert the trend logic."""
        vals = [s.coherence_signal for s in trajectory]
        if len(vals) < 3:
            return "unknown"
        arr = np.array(vals, dtype=np.float64)
        initial = arr[0] if abs(arr[0]) > 1e-9 else 1.0
        relative_change = (arr[-1] - arr[0]) / abs(initial)
        # Positive change in coherence = improving (opposite of other metrics)
        if relative_change > abs(_IMPROVING_THRESHOLD):
            return "improving"
        if relative_change < -_WORSENING_THRESHOLD:
            return "worsening"
        return "stable"
