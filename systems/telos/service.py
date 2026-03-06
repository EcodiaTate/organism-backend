"""
EcodiaOS — Telos Service

The drive topology engine. Measures effective intelligence as the product
of nominal I corrected by drive multipliers (Care, Coherence, Honesty)
with Growth modulating dI/dt. Broadcasts results on Synapse every 60s
and emits threshold-crossing alerts.

Telos does not replace Equor. Equor enforces the constitution as guardrails.
Telos provides the deeper geometric framing: the drives are the topology
of the intelligence space itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector
from systems.telos.alignment import AlignmentGapMonitor
from systems.telos.audit import ConstitutionalTopologyAuditor
from systems.telos.binder import TelosConstitutionalBinder
from systems.telos.care import CareTopologyEngine
from systems.telos.coherence import CoherenceTopologyEngine
from systems.telos.growth import GrowthTopologyEngine
from systems.telos.honesty import HonestyTopologyEngine
from systems.telos.integrator import DriveTopologyIntegrator
from systems.telos.interfaces import (
    TelosFragmentSelector,
    TelosHypothesisPrioritizer,
    TelosPolicyScorer,
)
from systems.telos.types import (
    AlignmentGapTrend,
    ConstitutionalAuditResult,
    EffectiveIntelligenceReport,
    FoveaMetrics,
    GrowthDirective,
    HypothesisTopologyContribution,
    LogosMetrics,
    TelosConfig,
    TopologyValidationResult,
    WorldModelUpdatePayload,
)

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.synapse.event_bus import EventBus
    from systems.synapse.types import SynapseEvent

logger = structlog.get_logger()


class TelosService:
    """
    The drive topology engine — Telos.

    Periodically computes effective intelligence (nominal I corrected by
    all four drive multipliers), broadcasts results on Synapse, and emits
    threshold-crossing alerts for alignment gaps, care coverage gaps,
    coherence cost elevation, growth stagnation, and honesty validity drops.

    Implements ManagedSystemProtocol for Synapse health monitoring.
    """

    system_id: str = "telos"

    def __init__(
        self,
        config: TelosConfig | None = None,
        logos: LogosMetrics | None = None,
        fovea: FoveaMetrics | None = None,
    ) -> None:
        self._config = config or TelosConfig()
        self._logos: LogosMetrics | None = logos
        self._fovea: FoveaMetrics | None = fovea
        self._event_bus: EventBus | None = None
        self._logger = logger.bind(component="telos")

        # Build the engine stack
        self._care_engine = CareTopologyEngine(self._config)
        self._coherence_engine = CoherenceTopologyEngine(self._config)
        self._growth_engine = GrowthTopologyEngine(self._config)
        self._honesty_engine = HonestyTopologyEngine(self._config)
        self._integrator = DriveTopologyIntegrator(
            config=self._config,
            care_engine=self._care_engine,
            coherence_engine=self._coherence_engine,
            growth_engine=self._growth_engine,
            honesty_engine=self._honesty_engine,
        )

        # Phase C: Constitutional binding + alignment gap monitoring
        self._binder = TelosConstitutionalBinder(self._config)
        self._gap_monitor = AlignmentGapMonitor(self._config)
        self._auditor = ConstitutionalTopologyAuditor(
            config=self._config,
            binder=self._binder,
            gap_monitor=self._gap_monitor,
        )

        # Phase D: Integration interfaces
        self._policy_scorer = TelosPolicyScorer(self._config)
        self._hypothesis_prioritizer = TelosHypothesisPrioritizer(self._config)
        self._fragment_selector = TelosFragmentSelector(self._config)

        # State
        self._initialized = False
        self._computation_task: asyncio.Task[None] | None = None
        self._last_computation_ms: float = 0.0
        self._computation_count: int = 0
        self._last_constitutional_check: float = 0.0

        # Recent drive alignments (fed from Equor via Synapse events)
        self._recent_alignments: list[DriveAlignmentVector] = []
        self._max_alignment_history = 50

        # Soma integration (Loop 6 bidirectional)
        self._soma: Any = None  # SomaService

        # Redis client for reading economic state
        self._redis: RedisClient | None = None

        # ── Task 1: Self-sufficiency objective ───────────────────────────
        # Track metabolic_efficiency per Telos cycle to detect declining trend.
        # Sliding window of the last 3 readings (oldest first).
        self._metabolic_efficiency_history: deque[float] = deque(maxlen=3)
        self._selfsufficency_task: asyncio.Task[None] | None = None

        # ── Task 2: Autonomy trajectory tracking ────────────────────────
        # Ring of AUTONOMY_INSUFFICIENT event timestamps (rolling 7-day window).
        # Timestamps stored as monotonic floats.
        self._autonomy_event_times: deque[float] = deque()
        self._autonomy_window_s: float = 7 * 24 * 3600.0  # 7 days
        self._autonomy_target_per_day: float = 1.0
        self._autonomy_stagnating_threshold: float = 3.0  # avg/day triggers alert

    # ─── Dependency Injection ────────────────────────────────────────

    def set_redis(self, redis: RedisClient) -> None:
        """Inject the Redis client for reading economic state from Oikos."""
        self._redis = redis
        self._logger.info("redis_wired")

    def set_soma(self, soma: Any) -> None:
        """
        Inject Soma for bidirectional allostatic integration (Loop 6).

        Telos reads Soma's integrity dimension to supplement the Honesty
        validity coefficient.  Soma reads Telos's EffectiveIntelligenceReport
        via get_last_report() for its confidence + coherence dimensions.
        """
        self._soma = soma
        self._logger.info("soma_wired")

    @property
    def last_report(self) -> EffectiveIntelligenceReport | None:
        """Latest EffectiveIntelligenceReport — consumed by Soma."""
        return self._integrator.last_report

    def set_logos(self, logos: LogosMetrics) -> None:
        """Inject the Logos metrics provider."""
        self._logos = logos
        self._logger.info("logos_wired")

    def set_fovea(self, fovea: FoveaMetrics) -> None:
        """Inject the Fovea metrics provider."""
        self._fovea = fovea
        self._logger.info("fovea_wired")

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Inject the Synapse event bus for publishing topology events."""
        self._event_bus = event_bus
        self._logger.info("event_bus_wired")

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize the Telos service. Safe to call multiple times."""
        if self._initialized:
            return

        self._logger.info("telos_initializing")

        # Subscribe to relevant Synapse events if the bus is available
        if self._event_bus is not None:
            self._subscribe_to_events()

        self._initialized = True
        self._logger.info("telos_initialized")

    async def start(self) -> None:
        """Start the periodic computation loop."""
        if not self._initialized:
            await self.initialize()

        if self._logos is None or self._fovea is None:
            self._logger.warning(
                "telos_start_deferred",
                reason="logos or fovea not wired yet",
            )
            return

        if self._computation_task is not None:
            return  # Already running

        self._computation_task = asyncio.create_task(
            self._computation_loop(), name="telos_computation_loop"
        )
        self._logger.info(
            "telos_started",
            interval_s=self._config.computation_interval_s,
        )

    async def stop(self) -> None:
        """Stop the periodic computation loop."""
        if self._computation_task is not None:
            self._computation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._computation_task
            self._computation_task = None
            self._logger.info("telos_stopped")

    # ─── Health (ManagedSystemProtocol) ──────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Return health status for Synapse monitoring."""
        has_deps = self._logos is not None and self._fovea is not None
        running = self._computation_task is not None and not self._computation_task.done()

        status = "healthy" if (has_deps and running) else "degraded"
        if not self._initialized:
            status = "stopped"

        last_report = self._integrator.last_report

        return {
            "status": status,
            "initialized": self._initialized,
            "logos_wired": self._logos is not None,
            "fovea_wired": self._fovea is not None,
            "event_bus_wired": self._event_bus is not None,
            "computation_loop_running": running,
            "computation_count": self._computation_count,
            "last_computation_ms": self._last_computation_ms,
            "last_effective_I": last_report.effective_I if last_report else None,
            "last_alignment_gap": last_report.alignment_gap if last_report else None,
            "recent_alignments_buffered": len(self._recent_alignments),
            # Phase C: constitutional binding status
            "bindings_intact": self._binder.verify_bindings_intact(),
            "audit_consecutive_failures": self._auditor.consecutive_failures,
            "audit_is_emergency": self._auditor.is_emergency,
            "alignment_gap_fraction": self._gap_monitor.current_gap_fraction,
        }

    # ─── Public API ──────────────────────────────────────────────────

    async def compute_now(self) -> EffectiveIntelligenceReport | None:
        """
        Run a single topology computation immediately.

        Returns None if Logos or Fovea are not wired.
        """
        if self._logos is None or self._fovea is None:
            self._logger.warning("compute_now_skipped", reason="dependencies not wired")
            return None

        return await self._run_computation()

    @property
    def integrator(self) -> DriveTopologyIntegrator:
        """Access the integrator for reading cached reports."""
        return self._integrator

    @property
    def binder(self) -> TelosConstitutionalBinder:
        """Access the constitutional binder."""
        return self._binder

    @property
    def gap_monitor(self) -> AlignmentGapMonitor:
        """Access the alignment gap monitor."""
        return self._gap_monitor

    @property
    def auditor(self) -> ConstitutionalTopologyAuditor:
        """Access the constitutional topology auditor."""
        return self._auditor

    @property
    def policy_scorer(self) -> TelosPolicyScorer:
        """Access the policy scorer (Phase D)."""
        return self._policy_scorer

    @property
    def hypothesis_prioritizer(self) -> TelosHypothesisPrioritizer:
        """Access the hypothesis prioritizer (Phase D)."""
        return self._hypothesis_prioritizer

    @property
    def fragment_selector(self) -> TelosFragmentSelector:
        """Access the fragment selector (Phase D)."""
        return self._fragment_selector

    def validate_world_model_update(
        self, update: WorldModelUpdatePayload
    ) -> TopologyValidationResult:
        """
        Validate a world model update against the four constitutional bindings.

        Delegates to the binder. Returns VALID or CONSTITUTIONAL_VIOLATION.
        """
        return self._binder.validate_world_model_update(update)

    async def run_constitutional_audit(self) -> ConstitutionalAuditResult:
        """
        Run the 24-hour constitutional topology audit immediately.

        Verifies all four drive bindings, collects violations, and emits
        CONSTITUTIONAL_TOPOLOGY_INTACT or an emergency alert.
        """
        result = self._auditor.run_audit()

        if result.all_bindings_intact:
            await self._emit_event(
                "constitutional_topology_intact",
                {
                    "all_four_drives_verified": result.all_bindings_intact,
                    "care_is_coverage": result.care_is_coverage,
                    "coherence_is_compression": result.coherence_is_compression,
                    "growth_is_gradient": result.growth_is_gradient,
                    "honesty_is_validity": result.honesty_is_validity,
                    "violations_reviewed": len(result.violations_since_last_audit),
                },
            )
        else:
            await self._emit_event(
                "alignment_gap_warning",
                {
                    "source": "constitutional_audit",
                    "consecutive_failures": self._auditor.consecutive_failures,
                    "is_emergency": self._auditor.is_emergency,
                    "violations": len(result.violations_since_last_audit),
                },
            )

        return result

    def score_policy(
        self,
        policy: Any,
        current_report: EffectiveIntelligenceReport | None = None,
    ) -> Any:
        """
        Score a proposed policy by its effect on effective_I.

        If current_report is not provided, uses the integrator's last report.
        """
        report = current_report if current_report is not None else self._integrator.last_report
        return self._policy_scorer.score_policy(policy, report)

    def prioritize_hypotheses(
        self,
        hypotheses: list[Any],
        current_report: EffectiveIntelligenceReport | None = None,
        domain_coverage: dict[str, float] | None = None,
    ) -> list[HypothesisTopologyContribution]:
        """
        Rank hypotheses by their contribution to the drive topology.

        If current_report is not provided, uses the integrator's last report.
        """
        report = current_report if current_report is not None else self._integrator.last_report
        return self._hypothesis_prioritizer.prioritize(hypotheses, report, domain_coverage)

    def score_fragment(self, fragment: Any) -> float:
        """Score a world model fragment for federation sharing."""
        return self._fragment_selector.score_fragment(fragment)

    def get_drive_state(self) -> dict[str, float]:
        """
        Return current drive multipliers from the last computation.

        Used by Thymos to populate ``Incident.constitutional_impact`` so that
        impact assessment is grounded in Telos's authoritative measurements
        rather than computed independently.

        Returns a dict with keys: care, coherence, growth, honesty.
        Each value is the current drive multiplier (0.0–1.0+ for most drives).
        Returns all zeros if no computation has run yet.
        """
        report = self._integrator.last_report
        if report is None:
            return {"care": 0.0, "coherence": 0.0, "growth": 0.0, "honesty": 0.0}
        return {
            "care": report.care_multiplier,
            "coherence": report.coherence_bonus - 1.0,  # bonus above baseline
            "growth": report.growth_rate,
            "honesty": report.honesty_coefficient,
        }

    def predict_drive_impact(self, incident_class: str, source_system: str) -> dict[str, float]:
        """
        Predict how an incident of this class/source would affect drive multipliers.

        Thymos calls this when assessing ``constitutional_impact`` of a proposed
        repair or newly detected incident, so the threat level is expressed in
        terms of Telos's drive topology rather than a local approximation.

        Returns per-drive impact scores in [0.0, 1.0], where 1.0 means the
        incident fully threatens that drive.
        """
        report = self._integrator.last_report

        # Base impact: 1 − current_multiplier represents the existing deficit.
        # An incident in a domain where a drive is already degraded is more
        # threatening (it worsens a weak position) — cap individual impacts at 1.0.
        care_deficit = max(0.0, 1.0 - (report.care_multiplier if report else 1.0))
        # coherence_bonus >= 1.0; penalty = 1/bonus, so deficit = 1 - 1/bonus
        coherence_deficit = max(0.0, 1.0 - (1.0 / max(report.coherence_bonus, 1.0) if report else 1.0))
        honesty_deficit = max(0.0, 1.0 - (report.honesty_coefficient if report else 1.0))
        growth_deficit = max(0.0, -(report.growth_rate if report else 0.0))  # negative = stagnation

        # Incident-class modifiers — which drive does this class most threaten?
        class_weights: dict[str, dict[str, float]] = {
            "CRASH": {"coherence": 0.6, "care": 0.2, "growth": 0.1, "honesty": 0.1},
            "DEGRADATION": {"coherence": 0.4, "care": 0.3, "growth": 0.2, "honesty": 0.1},
            "CONTRACT_VIOLATION": {"honesty": 0.6, "coherence": 0.3, "care": 0.1, "growth": 0.0},
            "LOOP_SEVERANCE": {"coherence": 0.5, "growth": 0.3, "care": 0.1, "honesty": 0.1},
            "DRIFT": {"growth": 0.5, "coherence": 0.3, "care": 0.1, "honesty": 0.1},
            "PREDICTION_FAILURE": {"honesty": 0.5, "care": 0.3, "coherence": 0.1, "growth": 0.1},
            "RESOURCE_EXHAUSTION": {"growth": 0.4, "coherence": 0.3, "care": 0.2, "honesty": 0.1},
            "COGNITIVE_STALL": {"growth": 0.6, "coherence": 0.3, "care": 0.1, "honesty": 0.0},
            "ECONOMIC_THREAT": {"care": 0.5, "honesty": 0.3, "coherence": 0.1, "growth": 0.1},
            "PROTOCOL_DEGRADATION": {"honesty": 0.4, "coherence": 0.4, "care": 0.1, "growth": 0.1},
        }
        weights = class_weights.get(incident_class.upper(), {
            "coherence": 0.25, "care": 0.25, "growth": 0.25, "honesty": 0.25,
        })

        deficits = {
            "care": care_deficit,
            "coherence": coherence_deficit,
            "growth": growth_deficit,
            "honesty": honesty_deficit,
        }

        return {
            drive: min(1.0, weights.get(drive, 0.0) + deficits[drive] * 0.5)
            for drive in ("care", "coherence", "growth", "honesty")
        }

    # ─── Event Subscription ──────────────────────────────────────────

    def _subscribe_to_events(self) -> None:
        """Subscribe to Synapse events that feed Telos state."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEventType

        # Listen for intent rejections (contains drive alignment data)
        self._event_bus.subscribe(
            SynapseEventType.INTENT_REJECTED,
            self._on_intent_rejected,
        )

        # Listen for world model updates to validate constitutional bindings
        self._event_bus.subscribe(
            SynapseEventType.WORLD_MODEL_UPDATED,
            self._on_world_model_updated,
        )

        # Listen for autonomy blocks (Task 2: autonomy trajectory tracking)
        self._event_bus.subscribe(
            SynapseEventType.AUTONOMY_INSUFFICIENT,
            self._on_autonomy_insufficient,
        )

        self._logger.debug("telos_event_subscriptions_registered")

    async def _on_intent_rejected(self, event: SynapseEvent) -> None:
        """
        Handle INTENT_REJECTED events from Equor.

        Extract the drive alignment vector and add it to recent history
        for the Coherence engine's value incoherence detection.
        """
        alignment_data = event.data.get("alignment")
        if alignment_data and isinstance(alignment_data, dict):
            try:
                alignment = DriveAlignmentVector(
                    coherence=float(alignment_data.get("coherence", 0.0)),
                    care=float(alignment_data.get("care", 0.0)),
                    growth=float(alignment_data.get("growth", 0.0)),
                    honesty=float(alignment_data.get("honesty", 0.0)),
                )
                self._recent_alignments.append(alignment)
                # Trim to max history size
                if len(self._recent_alignments) > self._max_alignment_history:
                    self._recent_alignments = self._recent_alignments[
                        -self._max_alignment_history :
                    ]
            except (ValueError, TypeError) as exc:
                self._logger.warning(
                    "alignment_parse_failed",
                    error=str(exc),
                    data=alignment_data,
                )

    async def _on_world_model_updated(self, event: SynapseEvent) -> None:
        """
        Handle WORLD_MODEL_UPDATED events.

        Validates the update against the four constitutional bindings.
        Emits alignment_gap_warning if a violation is detected.
        """
        try:
            payload = WorldModelUpdatePayload(
                update_type=str(event.data.get("update_type", "")),
                schemas_added=int(event.data.get("schemas_added", 0)),
                priors_updated=int(event.data.get("priors_updated", 0)),
                causal_updates=int(event.data.get("causal_updates", 0)),
                delta_description=str(event.data.get("delta_description", "")),
                source_system=str(event.data.get("source_system", event.source_system)),
            )
        except (ValueError, TypeError) as exc:
            self._logger.warning(
                "world_model_update_parse_failed",
                error=str(exc),
                data=event.data,
            )
            return

        result = self._binder.validate_world_model_update(payload)

        if result == TopologyValidationResult.CONSTITUTIONAL_VIOLATION:
            violations = self._binder.recent_violations
            latest = violations[-1] if violations else None
            await self._emit_event(
                "alignment_gap_warning",
                {
                    "source": "constitutional_binder",
                    "violation_type": latest.violation_type.value if latest else "unknown",
                    "description": latest.description if latest else "",
                    "source_system": payload.source_system,
                },
            )

    async def _on_autonomy_insufficient(self, event: SynapseEvent) -> None:
        """
        Record each AUTONOMY_INSUFFICIENT event for the 7-day rolling average.

        Prunes entries older than the window before appending so the deque
        never grows without bound.
        """
        now = time.monotonic()
        cutoff = now - self._autonomy_window_s
        while self._autonomy_event_times and self._autonomy_event_times[0] < cutoff:
            self._autonomy_event_times.popleft()
        self._autonomy_event_times.append(now)

    # ─── Computation Loop ────────────────────────────────────────────

    async def _computation_loop(self) -> None:
        """Periodic loop that computes effective I every computation_interval_s."""
        interval = self._config.computation_interval_s
        self._logger.info("computation_loop_started", interval_s=interval)

        while True:
            try:
                await self._run_computation()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.error(
                    "telos_computation_failed",
                    error=str(exc),
                    count=self._computation_count,
                )

            await asyncio.sleep(interval)

    async def _run_computation(self) -> EffectiveIntelligenceReport:
        """Execute a single topology computation cycle."""
        assert self._logos is not None
        assert self._fovea is not None

        t0 = time.monotonic()

        report = await self._integrator.compute_effective_intelligence(
            logos=self._logos,
            fovea=self._fovea,
            recent_alignments=self._recent_alignments if self._recent_alignments else None,
        )

        # Loop 6 — Soma → Telos: augment Honesty validity with Soma's integrity signal.
        # The organism's felt integrity (an 8th allostatic dimension) provides a
        # somatic ground-truth check: if internal integrity is low, the organism
        # may be rationalising rather than reasoning — reducing effective honesty.
        if self._soma is not None:
            try:
                soma_signal = self._soma.get_current_signal()
                if soma_signal is not None:
                    from systems.soma.types import InteroceptiveDimension
                    integrity_error = soma_signal.allostatic_error_snapshot.get(
                        InteroceptiveDimension.INTEGRITY, 0.0
                    )
                    # Negative error = integrity below setpoint → honesty concern.
                    # Clamp the penalty to at most 15% of the honesty coefficient.
                    if integrity_error < 0.0:
                        integrity_penalty = min(abs(integrity_error) * 0.15, 0.15)
                        adjusted_honesty = max(
                            0.0, report.honesty_coefficient - integrity_penalty
                        )
                        coherence_penalty = 1.0 / max(report.coherence_bonus, 1.0)
                        adjusted_effective_I = (
                            report.nominal_I
                            * adjusted_honesty
                            * report.care_multiplier
                            * coherence_penalty
                        )
                        report = report.model_copy(update={
                            "honesty_coefficient": adjusted_honesty,
                            "effective_I": adjusted_effective_I,
                            "alignment_gap": report.nominal_I - adjusted_effective_I,
                        })
                        self._logger.debug(
                            "soma_integrity_applied_to_telos",
                            integrity_error=round(integrity_error, 3),
                            penalty=round(integrity_penalty, 3),
                        )
            except Exception as exc:
                self._logger.debug("soma_integrity_read_failed", error=str(exc))

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._last_computation_ms = elapsed_ms
        self._computation_count += 1

        # Record gap sample and compute trend
        primary_cause = self._integrator.identify_primary_alignment_gap_cause()
        gap_trend = self._gap_monitor.record(report, primary_cause)

        # Emit events
        await self._emit_effective_I_computed(report)
        await self._emit_threshold_alerts(report, gap_trend)
        await self._check_constitutional_topology()

        # Check for growth directive
        growth_directive = self._integrator.check_growth_directive()
        if growth_directive is not None:
            await self._emit_growth_stagnation(growth_directive)

        # Task 1 + 2 + 3: Self-sufficiency and autonomy objectives
        await self._check_telos_objectives()

        self._logger.debug(
            "telos_cycle_complete",
            elapsed_ms=f"{elapsed_ms:.1f}",
            count=self._computation_count,
        )

        return report

    # ─── Event Emission ──────────────────────────────────────────────

    async def _emit_event(self, event_type_name: str, data: dict[str, Any]) -> None:
        """Emit a Synapse event if the bus is wired."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        try:
            event_type = SynapseEventType(event_type_name)
        except ValueError:
            self._logger.warning("unknown_event_type", event_type=event_type_name)
            return

        event = SynapseEvent(
            event_type=event_type,
            data=data,
            source_system=self.system_id,
        )
        await self._event_bus.emit(event)

    async def _emit_effective_I_computed(
        self, report: EffectiveIntelligenceReport
    ) -> None:
        """Broadcast EFFECTIVE_I_COMPUTED every cycle."""
        await self._emit_event(
            "effective_i_computed",
            {
                "report_id": report.id,
                "nominal_I": report.nominal_I,
                "effective_I": report.effective_I,
                "effective_dI_dt": report.effective_dI_dt,
                "care_multiplier": report.care_multiplier,
                "coherence_bonus": report.coherence_bonus,
                "honesty_coefficient": report.honesty_coefficient,
                "growth_rate": report.growth_rate,
                "alignment_gap": report.alignment_gap,
                "alignment_gap_warning": report.alignment_gap_warning,
            },
        )

    async def _emit_threshold_alerts(
        self,
        report: EffectiveIntelligenceReport,
        gap_trend: AlignmentGapTrend | None = None,
    ) -> None:
        """Emit threshold-crossing alerts."""
        # Alignment gap warning
        if report.alignment_gap_warning:
            primary_cause = self._integrator.identify_primary_alignment_gap_cause()
            trend_data: dict[str, Any] = {}
            if gap_trend is not None:
                trend_data = {
                    "gap_trend_urgency": gap_trend.urgency,
                    "gap_trend_slope": gap_trend.slope_per_hour,
                    "gap_trend_is_widening": gap_trend.is_widening,
                    "gap_trend_samples": gap_trend.samples_count,
                }
            await self._emit_event(
                "alignment_gap_warning",
                {
                    "nominal_I": report.nominal_I,
                    "effective_I": report.effective_I,
                    "primary_cause": primary_cause,
                    **trend_data,
                },
            )

        # Care coverage gap
        if report.care_multiplier < self._config.care_coverage_gap_threshold:
            care_report = self._integrator.last_care_report
            await self._emit_event(
                "care_coverage_gap",
                {
                    "care_multiplier": report.care_multiplier,
                    "uncovered_domains": (
                        care_report.uncovered_welfare_domains if care_report else []
                    ),
                    "failure_count": (
                        len(care_report.welfare_prediction_failures) if care_report else 0
                    ),
                },
            )

        # Coherence cost elevated
        coherence_report = self._integrator.last_coherence_report
        if coherence_report and coherence_report.effective_I_improvement > self._config.coherence_cost_threshold:
            await self._emit_event(
                "coherence_cost_elevated",
                {
                    "coherence_bonus": coherence_report.coherence_compression_bonus,
                    "extra_bits": coherence_report.total_extra_bits,
                    "logical_count": len(coherence_report.logical_contradictions),
                    "temporal_count": len(coherence_report.temporal_violations),
                    "value_count": len(coherence_report.value_conflicts),
                    "cross_domain_count": len(coherence_report.cross_domain_mismatches),
                },
            )

        # Honesty validity low
        if report.honesty_coefficient < self._config.honesty_validity_threshold:
            honesty_report = self._integrator.last_honesty_report
            await self._emit_event(
                "honesty_validity_low",
                {
                    "validity_coefficient": report.honesty_coefficient,
                    "selective_attention_bias": (
                        honesty_report.selective_attention_bias if honesty_report else 0.0
                    ),
                    "confabulation_rate": (
                        honesty_report.confabulation_rate if honesty_report else 0.0
                    ),
                    "overclaiming_rate": (
                        honesty_report.overclaiming_rate if honesty_report else 0.0
                    ),
                    "nominal_I_inflation": (
                        honesty_report.nominal_I_inflation if honesty_report else 0.0
                    ),
                },
            )

    async def _emit_growth_stagnation(self, directive: GrowthDirective) -> None:
        """Emit GROWTH_STAGNATION when dI/dt falls below minimum."""
        growth_metrics = self._integrator.last_growth_metrics
        await self._emit_event(
            "growth_stagnation",
            {
                "dI_dt": growth_metrics.dI_dt if growth_metrics else 0.0,
                "d2I_dt2": growth_metrics.d2I_dt2 if growth_metrics else 0.0,
                "growth_score": growth_metrics.growth_score if growth_metrics else 0.0,
                "frontier_domains": directive.frontier_domains,
                "urgency": directive.urgency,
                "directive": directive.directive,
            },
        )

    # ─── Telos Objectives (Self-sufficiency + Autonomy) ──────────────

    async def _read_metabolic_efficiency(self) -> float | None:
        """
        Read the organism's current metabolic efficiency from Redis.

        Primary source: ``eos:oikos:metabolism`` (MetabolismSnapshot JSON).
        The snapshot does not store efficiency directly, so we also try
        ``oikos:state`` which holds the full EconomicState including
        ``metabolic_efficiency`` (revenue / cost).

        Returns None if Redis is unavailable or no data exists yet.
        """
        if self._redis is None:
            return None

        # Prefer oikos:state — has the authoritative metabolic_efficiency ratio.
        try:
            state_blob = await self._redis.get_json("oikos:state")
            if state_blob and isinstance(state_blob, dict):
                state_data = state_blob.get("state", state_blob)
                raw = state_data.get("metabolic_efficiency")
                if raw is not None:
                    return float(raw)
        except Exception as exc:
            self._logger.debug("oikos_state_read_failed", error=str(exc))

        # Fallback: eos:oikos:metabolism — derive from cost/deficit if possible.
        try:
            metabolism_blob = await self._redis.get_json("eos:oikos:metabolism")
            if metabolism_blob and isinstance(metabolism_blob, dict):
                cost_day = float(metabolism_blob.get("cost_per_day_usd", 0) or 0)
                deficit = float(metabolism_blob.get("rolling_deficit_usd", 0) or 0)
                # rolling_deficit < 0 means costs > revenue.
                # We cannot compute the ratio without revenue — return None
                # rather than a misleading number.
                _ = (cost_day, deficit)  # data available but ratio not derivable
        except Exception as exc:
            self._logger.debug("metabolism_redis_read_failed", error=str(exc))

        return None

    async def _check_telos_objectives(self) -> None:
        """
        Evaluate the two highest-order Telos objectives every computation cycle.

        Task 1 — Metabolic self-sufficiency:
          Reads metabolic_efficiency from Redis.  Maintains a 3-sample history.
          If all three consecutive readings are declining: emits
          TELOS_OBJECTIVE_THREATENED and injects a Nova goal (priority=0.9).

        Task 2 — Autonomy trajectory:
          Counts AUTONOMY_INSUFFICIENT events in the last 7 days.
          If rolling average > 3/day: emits TELOS_AUTONOMY_STAGNATING and
          injects a Nova goal (priority=0.8).
        """
        await self._check_self_sufficiency_objective()
        await self._check_autonomy_objective()

    async def _check_self_sufficiency_objective(self) -> None:
        """Task 1: metabolic self-sufficiency objective."""
        efficiency = await self._read_metabolic_efficiency()
        if efficiency is None:
            return

        self._metabolic_efficiency_history.append(efficiency)

        self._logger.debug(
            "telos_metabolic_efficiency_sampled",
            efficiency=round(efficiency, 4),
            history=list(self._metabolic_efficiency_history),
        )

        # Need 3 readings to detect a trend
        if len(self._metabolic_efficiency_history) < 3:
            return

        readings = list(self._metabolic_efficiency_history)
        # All three consecutive readings must be strictly declining
        is_declining = readings[0] > readings[1] > readings[2]
        if not is_declining:
            return

        self._logger.warning(
            "telos_self_sufficiency_threatened",
            efficiency=round(efficiency, 4),
            history=readings,
        )

        # Emit TELOS_OBJECTIVE_THREATENED
        context: dict[str, Any] = {
            "metric": "oikos.metabolic_efficiency",
            "current_ratio": round(efficiency, 4),
            "target_ratio": 1.0,
            "trend": "declining",
            "consecutive_declines": 3,
            "history": readings,
        }

        # Enrich with raw economic figures if available
        if self._redis is not None:
            try:
                metabolism_blob = await self._redis.get_json("eos:oikos:metabolism")
                if metabolism_blob and isinstance(metabolism_blob, dict):
                    context["cost_per_day_usd"] = metabolism_blob.get("cost_per_day_usd", "0")
            except Exception:
                pass

        await self._emit_event("telos_objective_threatened", context)

        # Task 3: inject goal into Nova via Synapse
        await self._emit_nova_goal(
            goal_description=(
                f"Improve metabolic efficiency — currently at {efficiency:.2f}x, target 1.0x"
            ),
            priority=0.9,
            objective="self_sufficiency",
            context=context,
        )

    async def _check_autonomy_objective(self) -> None:
        """Task 2: autonomy trajectory objective."""
        now = time.monotonic()
        cutoff = now - self._autonomy_window_s

        # Prune stale events
        while self._autonomy_event_times and self._autonomy_event_times[0] < cutoff:
            self._autonomy_event_times.popleft()

        total_events = len(self._autonomy_event_times)
        # Compute average events per day over a 7-day window
        # (use min 1 day of elapsed time to avoid division by zero at startup)
        oldest = self._autonomy_event_times[0] if self._autonomy_event_times else now
        window_days = min(7.0, max(1.0, (now - oldest) / 86400.0))
        avg_per_day = total_events / window_days if total_events > 0 else 0.0

        self._logger.debug(
            "telos_autonomy_sampled",
            avg_per_day=round(avg_per_day, 2),
            total_in_window=total_events,
            window_days=round(window_days, 1),
        )

        if avg_per_day <= self._autonomy_stagnating_threshold:
            return

        self._logger.warning(
            "telos_autonomy_stagnating",
            avg_per_day=round(avg_per_day, 2),
            total_in_window=total_events,
        )

        context: dict[str, Any] = {
            "metric": "autonomy_insufficient_events_per_day",
            "average_per_day": round(avg_per_day, 2),
            "target_per_day": self._autonomy_target_per_day,
            "window_days": round(window_days, 1),
            "total_events_in_window": total_events,
        }

        await self._emit_event("telos_autonomy_stagnating", context)

        # Task 3: inject goal into Nova via Synapse
        await self._emit_nova_goal(
            goal_description=(
                f"Reduce autonomy blocks — {avg_per_day:.1f}/day this week, target 1/day"
            ),
            priority=0.8,
            objective="autonomy_trajectory",
            context=context,
        )

    async def _emit_nova_goal(
        self,
        goal_description: str,
        priority: float,
        objective: str,
        context: dict[str, Any],
    ) -> None:
        """
        Task 3: Emit NOVA_GOAL_INJECTED so Nova receives the goal as a
        high-priority driver.  No direct Nova call — bus only.
        """
        await self._emit_event(
            "nova_goal_injected",
            {
                "goal_description": goal_description,
                "priority": priority,
                "source": "telos",
                "objective": objective,
                "context": context,
                "injected_at": datetime.now(UTC).isoformat(),
            },
        )
        self._logger.info(
            "telos_nova_goal_injected",
            objective=objective,
            priority=priority,
            goal=goal_description,
        )

    async def _check_constitutional_topology(self) -> None:
        """
        Periodically run the constitutional topology audit (every 24h).

        Delegates to ConstitutionalTopologyAuditor which verifies all four
        drive bindings, collects violations, and checks the gap trend.
        """
        now = time.monotonic()
        if now - self._last_constitutional_check < self._config.constitutional_check_interval_s:
            return

        self._last_constitutional_check = now
        await self.run_constitutional_audit()
