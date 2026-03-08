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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from primitives.common import DriveAlignmentVector, SystemID
from primitives.re_training import RETrainingExample
from systems.telos.adapters import FoveaMetricsAdapter, LogosMetricsAdapter
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
from systems.telos.population import PopulationIntelligenceAggregator
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


# ─── Typed Protocols (replace Any for Soma — Task 8) ─────────────────
# Soma uses InteroceptiveDimension enum keys but we access via string
# "integrity" to avoid cross-system import of systems.soma.types.

_SOMA_INTEGRITY_KEY = "integrity"


@runtime_checkable
class SomaServiceProtocol(Protocol):
    """Protocol for Soma — Telos reads felt integrity, not raw types."""

    def get_current_signal(self) -> Any: ...


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
        self._last_hourly_rollup_hour: int = -1

        # Recent drive alignments (fed from Equor via Synapse events)
        self._recent_alignments: list[DriveAlignmentVector] = []
        self._max_alignment_history = 50

        # Soma integration (Loop 6 bidirectional) — typed protocol
        self._soma: SomaServiceProtocol | None = None

        # Redis client for reading economic state
        self._redis: RedisClient | None = None

        # ── Hypothesis test tracking (P1/P6) ──────────────────────────
        # Fed by HYPOTHESIS_CONFIRMED / HYPOTHESIS_REFUTED subscriptions.
        self._hypothesis_confirmed_count: int = 0
        self._hypothesis_refuted_count: int = 0
        self._hypothesis_total_count: int = 0

        # ── Incident confabulation tracking (Spec 18 §SG3) ────────────
        # Fed by INCIDENT_RESOLVED events from Thymos.
        # Used by get_measured_confabulation_rate() → HonestyTopologyEngine.
        self._incident_confabulation_count: int = 0
        self._incident_total_count: int = 0

        # ── M3: Population-level intelligence aggregator ─────────────────
        self._population_aggregator = PopulationIntelligenceAggregator()

        # ── Task 1: Self-sufficiency objective ───────────────────────────
        self._metabolic_efficiency_history: deque[float] = deque(maxlen=3)
        self._selfsufficency_task: asyncio.Task[None] | None = None

        # ── Task 2: Autonomy trajectory tracking ────────────────────────
        self._autonomy_event_times: deque[float] = deque()
        self._autonomy_window_s: float = 7 * 24 * 3600.0
        self._autonomy_target_per_day: float = 1.0
        self._autonomy_stagnating_threshold: float = 3.0

    # ─── Dependency Injection ────────────────────────────────────────

    def set_redis(self, redis: RedisClient) -> None:
        """Inject the Redis client for reading economic state from Oikos."""
        self._redis = redis
        self._logger.info("redis_wired")

    def set_soma(self, soma: SomaServiceProtocol) -> None:
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

    def set_neo4j(self, driver: Any, instance_id: str = "") -> None:
        """Inject Neo4j driver for I-history persistence."""
        if isinstance(self._logos, LogosMetricsAdapter):
            self._logos.set_history_neo4j(driver, instance_id)
            self._logger.info("neo4j_wired_for_i_history")

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize the Telos service. Safe to call multiple times."""
        if self._initialized:
            return

        self._logger.info("telos_initializing")

        # Subscribe to relevant Synapse events if the bus is available
        if self._event_bus is not None:
            self._subscribe_to_events()

        # Child-side: apply inherited parent genome if provided via environment
        await self._apply_inherited_telos_genome_if_child()

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
            # Hypothesis tracking
            "hypothesis_confirmed": self._hypothesis_confirmed_count,
            "hypothesis_refuted": self._hypothesis_refuted_count,
            # M3: Population aggregator
            "population_child_count": self._population_aggregator.child_count,
        }

    # ─── Public API ──────────────────────────────────────────────────

    async def compute_now(self) -> EffectiveIntelligenceReport | None:
        """Run a single topology computation immediately."""
        if self._logos is None or self._fovea is None:
            self._logger.warning("compute_now_skipped", reason="dependencies not wired")
            return None

        return await self._run_computation()

    @property
    def integrator(self) -> DriveTopologyIntegrator:
        return self._integrator

    @property
    def binder(self) -> TelosConstitutionalBinder:
        return self._binder

    @property
    def gap_monitor(self) -> AlignmentGapMonitor:
        return self._gap_monitor

    @property
    def auditor(self) -> ConstitutionalTopologyAuditor:
        return self._auditor

    @property
    def policy_scorer(self) -> TelosPolicyScorer:
        return self._policy_scorer

    @property
    def hypothesis_prioritizer(self) -> TelosHypothesisPrioritizer:
        return self._hypothesis_prioritizer

    @property
    def fragment_selector(self) -> TelosFragmentSelector:
        return self._fragment_selector

    def validate_world_model_update(
        self, update: WorldModelUpdatePayload
    ) -> TopologyValidationResult:
        return self._binder.validate_world_model_update(update)

    async def run_constitutional_audit(self) -> ConstitutionalAuditResult:
        """Run the 24-hour constitutional topology audit immediately."""
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

        # ── RE training: drive audit decision (enriched traces — SG1) ──
        asyncio.ensure_future(self._emit_re_training_example(
            category="drive_audit",
            instruction="Run constitutional topology audit: verify all four drive bindings are intact.",
            input_context=f"violations={len(result.violations_since_last_audit)}",
            output=f"intact={result.all_bindings_intact}, care={result.care_is_coverage}, coherence={result.coherence_is_compression}, growth={result.growth_is_gradient}, honesty={result.honesty_is_validity}",
            outcome_quality=1.0 if result.all_bindings_intact else 0.2,
            reasoning_trace=self._build_audit_reasoning_trace(result),
            alternatives_considered=[
                "Partial audit (single drive) — rejected: all 4 must be verified together",
                "Skip audit if last was < 12h ago — rejected: consistency matters more than frequency",
            ],
            episode_id=f"drive_audit:{self._instance_id}:{int(self._last_constitutional_check)}",
        ))

        return result

    def score_policy(
        self,
        policy: Any,
        current_report: EffectiveIntelligenceReport | None = None,
    ) -> Any:
        report = current_report if current_report is not None else self._integrator.last_report
        return self._policy_scorer.score_policy(policy, report)

    def prioritize_hypotheses(
        self,
        hypotheses: list[Any],
        current_report: EffectiveIntelligenceReport | None = None,
        domain_coverage: dict[str, float] | None = None,
    ) -> list[HypothesisTopologyContribution]:
        report = current_report if current_report is not None else self._integrator.last_report
        return self._hypothesis_prioritizer.prioritize(hypotheses, report, domain_coverage)

    def score_fragment(self, fragment: Any) -> float:
        return self._fragment_selector.score_fragment(fragment)

    def get_drive_state(self) -> dict[str, float]:
        """Return current drive multipliers from the last computation."""
        report = self._integrator.last_report
        if report is None:
            return {"care": 0.0, "coherence": 0.0, "growth": 0.0, "honesty": 0.0}
        return {
            "care": report.care_multiplier,
            "coherence": report.coherence_bonus - 1.0,
            "growth": report.growth_rate,
            "honesty": report.honesty_coefficient,
        }

    def predict_drive_impact(self, incident_class: str, source_system: str) -> dict[str, float]:
        """Predict how an incident would affect drive multipliers."""
        report = self._integrator.last_report

        care_deficit = max(0.0, 1.0 - (report.care_multiplier if report else 1.0))
        coherence_deficit = max(0.0, 1.0 - (1.0 / max(report.coherence_bonus, 1.0) if report else 1.0))
        honesty_deficit = max(0.0, 1.0 - (report.honesty_coefficient if report else 1.0))
        growth_deficit = max(0.0, -(report.growth_rate if report else 0.0))

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

    # ─── Genome Inheritance (Spec 18 SG3) ────────────────────────────

    async def export_telos_genome(self) -> "TelosGenomeFragment":
        """
        Extract a heritable TelosGenomeFragment from the current service state.

        Called by SpawnChildExecutor at spawn time (Step 0b).  The returned
        fragment is immutable once extracted — it reflects the parent's live
        drive calibration at the moment of reproduction.

        Mutation ranges (Spec 18 SG3):
          resonance_curve_coefficients  ±15%
          dissipation_baseline          ±10%
          coupling_strength             ±20%
        """
        from primitives.common import new_id, utc_now
        from primitives.genome_inheritance import TeloDriveCalibration, TelosGenomeFragment

        fragment = self.to_genome_fragment()
        self._logger.info(
            "telos_genome_extracted",
            genome_id=fragment.genome_id,
            topology=fragment.topology,
            drive_count=len(fragment.drive_calibrations),
        )

        # Emit extraction event for Mitosis to record the ID
        if self._event_bus is not None:
            await self._emit_event(
                "telos_genome_extracted",
                {
                    "instance_id": getattr(self, "_instance_id", ""),
                    "genome_id": fragment.genome_id,
                    "generation": fragment.generation,
                    "topology": fragment.topology,
                    "drive_count": len(fragment.drive_calibrations),
                },
            )

        return fragment

    def to_genome_fragment(self) -> "TelosGenomeFragment":
        """
        Serialise the current drive calibration config into a TelosGenomeFragment.

        Reads directly from self._config (topology weights, drive-specific
        coefficients) and from the integrator's last known state.
        """
        import math

        from primitives.common import new_id, utc_now
        from primitives.genome_inheritance import TeloDriveCalibration, TelosGenomeFragment

        drives = ["care", "coherence", "growth", "honesty"]

        # Default mutation ranges per Spec 18 SG3
        default_mutation_ranges: dict[str, tuple[float, float]] = {
            "resonance_curve_coefficients": (-0.15, 0.15),
            "dissipation_baseline": (-0.10, 0.10),
            "coupling_strength": (-0.20, 0.20),
        }

        drive_calibrations: dict[str, TeloDriveCalibration] = {}
        for drive in drives:
            # Resonance curve: extract per-drive weight as the primary coefficient
            weight_key = f"{drive}_weight"
            weight = float(getattr(self._config, weight_key, 0.25))
            resonance_coefs = {"a0": weight, "a1": 1.0, "a2": 0.0}

            # Dissipation: approximate from the alignment gap warning threshold
            # scaled per-drive (no drive-specific config exists yet, use uniform baseline)
            dissipation = float(getattr(
                self._config,
                f"{drive}_dissipation_baseline",
                self._config.alignment_gap_warning_threshold * 0.1,
            ))

            # Coupling: seed with zero cross-drive coupling (evolvable over generations)
            coupling: dict[str, dict[str, float]] = {
                src: {tgt: 0.0 for tgt in drives if tgt != src}
                for src in drives
            }

            drive_calibrations[drive] = TeloDriveCalibration(
                resonance_curve_coefficients=resonance_coefs,
                dissipation_baseline=dissipation,
                coupling_strength=coupling,
                mutation_ranges=default_mutation_ranges,
                last_adapted=utc_now(),
            )

        # Topology mode — default "linear"; can be overridden if future config adds it
        topology = str(getattr(self._config, "topology_mode", "linear"))

        # Topology parameters: current drive weights as topology anchors
        topology_parameters: dict = {
            drive: float(getattr(self._config, f"{drive}_weight", 0.25))
            for drive in drives
        }

        return TelosGenomeFragment(
            genome_id=new_id(),
            generation=1,
            extracted_at=utc_now(),
            drive_calibrations=drive_calibrations,
            topology=topology,
            topology_parameters=topology_parameters,
        )

    def from_genome_fragment(
        self, genome: "TelosGenomeFragment"
    ) -> None:
        """
        Deserialise a parent TelosGenomeFragment into this service's config.

        Applies the inherited calibrations *without* mutation jitter (for
        deterministic testing and child-side re-hydration).  For actual child
        initialisation with jitter, use _initialize_from_parent_genome().
        """
        for drive, calib in genome.drive_calibrations.items():
            weight_key = f"{drive}_weight"
            if hasattr(self._config, weight_key):
                # Restore a0 coefficient as the drive weight
                a0 = calib.resonance_curve_coefficients.get("a0")
                if a0 is not None:
                    setattr(self._config, weight_key, float(a0))

        # Restore topology mode if config supports it
        if hasattr(self._config, "topology_mode"):
            self._config.topology_mode = genome.topology  # type: ignore[attr-defined]

        self._logger.info(
            "telos_genome_restored",
            genome_id=genome.genome_id,
            topology=genome.topology,
        )

    async def _initialize_from_parent_genome(
        self,
        parent_genome: "TelosGenomeFragment",
    ) -> None:
        """
        Apply parent drive calibrations with bounded mutation jitter.

        Called during child Telos initialisation when ECODIAOS_TELOS_GENOME_PAYLOAD
        is present in the environment.  Each calibration parameter is jittered
        within the mutation_ranges carried by the genome itself (Spec 18 SG3).

        Emits GENOME_INHERITED to the Synapse bus so Evo can register the
        mutations as hypothesized adaptations for evolutionary scoring.
        """
        import math
        import random

        drive_mutations: dict[str, dict[str, float]] = {}
        total_sq_delta = 0.0

        for drive, parent_calib in parent_genome.drive_calibrations.items():
            child_calib = self._apply_genetic_mutation(parent_calib)
            drive_calibrations_field: dict = {}

            # Compute per-parameter delta for audit payload
            drive_mutations[drive] = {}
            for coef_name, parent_val in parent_calib.resonance_curve_coefficients.items():
                child_val = child_calib.resonance_curve_coefficients.get(coef_name, parent_val)
                delta = child_val - parent_val
                drive_mutations[drive][f"resonance_{coef_name}"] = round(delta, 6)
                total_sq_delta += delta ** 2

            diss_delta = child_calib.dissipation_baseline - parent_calib.dissipation_baseline
            drive_mutations[drive]["dissipation_baseline"] = round(diss_delta, 6)
            total_sq_delta += diss_delta ** 2

            # Apply child calibration to config
            weight_key = f"{drive}_weight"
            if hasattr(self._config, weight_key):
                a0 = child_calib.resonance_curve_coefficients.get("a0")
                if a0 is not None:
                    setattr(self._config, weight_key, float(a0))

        mutation_magnitude = math.sqrt(total_sq_delta)

        # Apply topology
        if hasattr(self._config, "topology_mode"):
            self._config.topology_mode = parent_genome.topology  # type: ignore[attr-defined]
        for drive, weight in parent_genome.topology_parameters.items():
            weight_key = f"{drive}_weight"
            if hasattr(self._config, weight_key):
                setattr(self._config, weight_key, float(weight))

        self._logger.info(
            "telos_genome_inherited",
            parent_genome_id=parent_genome.genome_id,
            generation=parent_genome.generation + 1,
            topology=parent_genome.topology,
            mutation_magnitude=round(mutation_magnitude, 6),
        )

        # Emit GENOME_INHERITED — Evo subscribes to track drive mutations
        if self._event_bus is not None:
            await self._emit_event(
                "genome_inherited",
                {
                    "child_instance_id": getattr(self, "_instance_id", ""),
                    "parent_genome_id": parent_genome.genome_id,
                    "generation": parent_genome.generation + 1,
                    "topology": parent_genome.topology,
                    "drive_mutations": drive_mutations,
                    "mutation_magnitude": round(mutation_magnitude, 6),
                },
            )

    @staticmethod
    def _apply_genetic_mutation(
        calib: "TeloDriveCalibration",
    ) -> "TeloDriveCalibration":
        """
        Apply bounded Gaussian jitter to a TeloDriveCalibration.

        Each parameter is multiplied by (1 + jitter) where jitter is drawn
        from N(0, σ) clamped to [lo, hi] from mutation_ranges.

        Default ranges if mutation_ranges is empty (Spec 18 SG3):
          resonance_curve_coefficients  ±15%
          dissipation_baseline          ±10%
          coupling_strength             ±20%
        """
        import random

        from primitives.common import utc_now
        from primitives.genome_inheritance import TeloDriveCalibration

        ranges = calib.mutation_ranges or {}
        resonance_range = ranges.get("resonance_curve_coefficients", (-0.15, 0.15))
        dissipation_range = ranges.get("dissipation_baseline", (-0.10, 0.10))
        coupling_range = ranges.get("coupling_strength", (-0.20, 0.20))

        def _jitter(value: float, lo: float, hi: float) -> float:
            sigma = (hi - lo) / 6.0  # ±3σ covers the range
            delta = random.gauss(0.0, sigma)
            delta = max(lo, min(hi, delta))
            return value * (1.0 + delta)

        # Mutate resonance curve coefficients
        new_resonance = {
            k: _jitter(v, *resonance_range)
            for k, v in calib.resonance_curve_coefficients.items()
        }

        # Mutate dissipation baseline (floor at 0)
        new_dissipation = max(0.0, _jitter(calib.dissipation_baseline, *dissipation_range))

        # Mutate coupling strengths
        new_coupling: dict[str, dict[str, float]] = {
            src: {
                tgt: _jitter(v, *coupling_range)
                for tgt, v in tgt_map.items()
            }
            for src, tgt_map in calib.coupling_strength.items()
        }

        return TeloDriveCalibration(
            resonance_curve_coefficients=new_resonance,
            dissipation_baseline=new_dissipation,
            coupling_strength=new_coupling,
            mutation_ranges=calib.mutation_ranges,
            last_adapted=utc_now(),
        )

    async def _apply_inherited_telos_genome_if_child(self) -> None:
        """
        Child-side bootstrap: deserialise parent genome from environment.

        Reads ECODIAOS_TELOS_GENOME_PAYLOAD (JSON-encoded TelosGenomeFragment)
        injected by LocalDockerSpawner.  If present, calls
        _initialize_from_parent_genome() to apply calibrations with mutation jitter.
        Non-fatal — child falls back to default config on any error.
        """
        import json
        import os

        is_genesis = os.environ.get("ECODIAOS_IS_GENESIS_NODE", "true").lower() == "true"
        if is_genesis:
            return

        payload_json = os.environ.get("ECODIAOS_TELOS_GENOME_PAYLOAD", "")
        if not payload_json:
            return

        try:
            from primitives.genome_inheritance import TelosGenomeFragment

            data = json.loads(payload_json)
            parent_genome = TelosGenomeFragment.model_validate(data)
            await self._initialize_from_parent_genome(parent_genome)
            self._logger.info(
                "telos_child_genome_applied",
                parent_genome_id=parent_genome.genome_id,
                generation=parent_genome.generation,
            )
        except Exception as exc:
            self._logger.warning(
                "telos_child_genome_apply_failed",
                error=str(exc),
                note="Proceeding with default drive calibrations",
            )

    # ─── Event Subscription (P6/P7 — all 9 subscriptions) ───────────

    def _subscribe_to_events(self) -> None:
        """Subscribe to all 9 Synapse events that feed Telos state."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEventType

        # Original 3 subscriptions
        self._event_bus.subscribe(
            SynapseEventType.INTENT_REJECTED,
            self._on_intent_rejected,
        )
        self._event_bus.subscribe(
            SynapseEventType.WORLD_MODEL_UPDATED,
            self._on_world_model_updated,
        )
        self._event_bus.subscribe(
            SynapseEventType.AUTONOMY_INSUFFICIENT,
            self._on_autonomy_insufficient,
        )

        # ── 6 new subscriptions (P6/P7) ──────────────────────────────

        # Hypothesis test tracking → HonestyTopologyEngine measured data (P1)
        self._event_bus.subscribe(
            SynapseEventType.EVO_HYPOTHESIS_CONFIRMED,
            self._on_hypothesis_confirmed,
        )
        self._event_bus.subscribe(
            SynapseEventType.EVO_HYPOTHESIS_REFUTED,
            self._on_hypothesis_refuted,
        )

        # Tier 3 causal invariant → nominal_I boost (SG5)
        self._event_bus.subscribe(
            SynapseEventType.KAIROS_TIER3_INVARIANT_DISCOVERED,
            self._on_tier3_invariant_discovered,
        )

        # Commitment violations from Thread → CoherenceTopologyEngine
        self._event_bus.subscribe(
            SynapseEventType.COMMITMENT_VIOLATED,
            self._on_commitment_violated,
        )

        # Welfare outcomes from Axon → CareTopologyEngine
        self._event_bus.subscribe(
            SynapseEventType.WELFARE_OUTCOME_RECORDED,
            self._on_welfare_outcome_recorded,
        )

        # Incident resolved from Thymos → HonestyTopologyEngine confabulation
        self._event_bus.subscribe(
            SynapseEventType.INCIDENT_RESOLVED,
            self._on_incident_resolved,
        )

        # Fovea prediction errors → FoveaMetricsAdapter buffer (P3)
        self._event_bus.subscribe(
            SynapseEventType.FOVEA_PREDICTION_ERROR,
            self._on_fovea_prediction_error,
        )

        # M3: Population-level aggregation — fleet drive alignment data
        self._event_bus.subscribe(
            SynapseEventType.CHILD_HEALTH_REPORT,
            self._on_child_health_report,
        )
        self._event_bus.subscribe(
            SynapseEventType.CHILD_DIED,
            self._on_child_died,
        )

        # Simula validation requests — run constitutional binder, emit ALIGNMENT_GAP_WARNING
        # if the proposed mutation would violate drive topology (replaces direct method call)
        if hasattr(SynapseEventType, "TELOS_WORLD_MODEL_VALIDATE"):
            self._event_bus.subscribe(
                SynapseEventType.TELOS_WORLD_MODEL_VALIDATE,
                self._on_world_model_validate_request,
            )

        # §8.6: Self-coherence alarm from Identity — low coherence means drive
        # balance is producing an incoherent functional identity; Telos responds
        # by nudging drive weights toward their historical mean (mean-reversion).
        if hasattr(SynapseEventType, "SELF_COHERENCE_ALARM"):
            self._event_bus.subscribe(
                SynapseEventType.SELF_COHERENCE_ALARM,
                self._on_self_coherence_alarm,
            )

        self._logger.debug(
            "telos_event_subscriptions_registered",
            count=13,
        )

    # ─── Event Handlers ──────────────────────────────────────────────

    async def _on_intent_rejected(self, event: SynapseEvent) -> None:
        """Handle INTENT_REJECTED — extract drive alignment for Coherence."""
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
        """Handle WORLD_MODEL_UPDATED — validate constitutional bindings."""
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

    async def _on_world_model_validate_request(self, event: SynapseEvent) -> None:
        """Handle TELOS_WORLD_MODEL_VALIDATE from Simula.

        Simula emits this instead of calling validate_world_model_update() directly.
        Runs the ConstitutionalBinder and emits ALIGNMENT_GAP_WARNING if there is a
        constitutional violation, so Simula's cached _telos_alignment_gap_active flag
        is updated for the next proposal cycle.
        """
        try:
            payload = WorldModelUpdatePayload(
                update_type=str(event.data.get("update_type", "")),
                schemas_added=0,
                priors_updated=0,
                causal_updates=0,
                delta_description=str(event.data.get("delta_description", "")),
                source_system=str(event.data.get("source_system", event.source_system)),
            )
        except (ValueError, TypeError) as exc:
            self._logger.warning(
                "validate_request_parse_failed",
                error=str(exc),
                proposal_id=event.data.get("proposal_id", ""),
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
                    "alignment_gap": 0.35,  # signal magnitude — Simula threshold is 0.35
                    "primary_cause": "drive_topology_violation",
                    "proposal_id": event.data.get("proposal_id", ""),
                },
            )
            self._logger.info(
                "simula_proposal_constitutional_violation",
                proposal_id=event.data.get("proposal_id", ""),
                violation_type=latest.violation_type.value if latest else "unknown",
            )

    async def _on_autonomy_insufficient(self, event: SynapseEvent) -> None:
        """Record AUTONOMY_INSUFFICIENT for 7-day rolling average."""
        now = time.monotonic()
        cutoff = now - self._autonomy_window_s
        while self._autonomy_event_times and self._autonomy_event_times[0] < cutoff:
            self._autonomy_event_times.popleft()
        self._autonomy_event_times.append(now)

    async def _on_hypothesis_confirmed(self, event: SynapseEvent) -> None:
        """Track confirmed hypotheses for honesty measured data (P1/P6)."""
        self._hypothesis_confirmed_count += 1
        self._hypothesis_total_count += 1

    async def _on_hypothesis_refuted(self, event: SynapseEvent) -> None:
        """Track refuted hypotheses for honesty measured data (P1/P6)."""
        self._hypothesis_refuted_count += 1
        self._hypothesis_total_count += 1

    async def _on_tier3_invariant_discovered(self, event: SynapseEvent) -> None:
        """
        KAIROS_TIER3_INVARIANT_DISCOVERED → boost nominal_I (SG5).

        nominal_I += 0.01 × confidence × entropy_reduction
        """
        confidence = float(event.data.get("hold_rate", 0.0))
        # entropy_reduction approximated from description_length_bits contribution
        entropy_reduction = float(event.data.get("intelligence_ratio_contribution", 0.01))
        boost = 0.01 * confidence * max(entropy_reduction, 0.01)

        self._logger.info(
            "tier3_invariant_nominal_I_boost",
            boost=round(boost, 6),
            confidence=round(confidence, 3),
            invariant_id=event.data.get("invariant_id", ""),
        )
        # The boost is applied at the next computation cycle via the
        # Logos adapter's intelligence ratio (Logos will reflect the
        # improved world model compression). Log it for RE training.
        asyncio.ensure_future(self._emit_re_training_example(
            category="invariant_integration",
            instruction="Integrate Tier 3 causal invariant into intelligence measurement.",
            input_context=f"invariant_id={event.data.get('invariant_id', '')}, confidence={confidence:.3f}",
            output=f"nominal_I_boost={boost:.6f}",
            outcome_quality=confidence,
            reasoning_trace=f"Tier 3 substrate-independent invariant: confidence={confidence:.3f}, entropy_reduction={entropy_reduction:.4f}. Boost = 0.01 * {confidence:.3f} * {entropy_reduction:.4f} = {boost:.6f}",
            episode_id=event.data.get("invariant_id", ""),
        ))

    async def _on_commitment_violated(self, event: SynapseEvent) -> None:
        """COMMITMENT_VIOLATED from Thread → temporal incoherence signal for Coherence.

        Injects a synthetic alignment vector with depressed coherence and honesty
        proportional to violation severity. CoherenceTopologyEngine detects high
        variance across recent alignment vectors as value incoherence (Spec 18 §SG3).
        """
        severity = float(event.data.get("severity", 0.5))
        self._logger.debug(
            "commitment_violation_received",
            commitment_id=event.data.get("commitment_id", ""),
            severity=severity,
        )
        # Commitment break = coherence drop + honesty cost proportional to severity.
        # The depressed vector widens drive variance across _recent_alignments,
        # which CoherenceTopologyEngine._detect_value_incoherence picks up.
        perturbed = DriveAlignmentVector(
            coherence=max(-1.0, 1.0 - severity * 0.8),
            care=1.0,
            growth=0.5,
            honesty=max(-1.0, 1.0 - severity * 0.5),
        )
        self._recent_alignments.append(perturbed)
        if len(self._recent_alignments) > self._max_alignment_history:
            self._recent_alignments = self._recent_alignments[-self._max_alignment_history:]

    async def _on_welfare_outcome_recorded(self, event: SynapseEvent) -> None:
        """WELFARE_OUTCOME_RECORDED from Axon → feed CareTopologyEngine.

        When predicted vs actual welfare impact diverges by > 0.1, injects
        directly into FoveaPredictionErrorBuffer so CareTopologyEngine counts
        it as a welfare prediction failure on the next cycle (Spec 18 §SG3).
        """
        domain = str(event.data.get("welfare_domain", "welfare"))
        predicted = float(event.data.get("predicted_impact", 0.0))
        actual = float(event.data.get("actual_impact", 0.0))
        divergence = abs(actual - predicted)

        self._logger.debug(
            "welfare_outcome_received",
            welfare_domain=domain,
            predicted=predicted,
            actual=actual,
            divergence=round(divergence, 3),
        )

        if divergence < 0.1:
            return  # Within tolerance — not a material prediction failure

        if isinstance(self._fovea, FoveaMetricsAdapter):
            # Synthesise a FOVEA_PREDICTION_ERROR-shaped payload so the buffer
            # ingests it as a high-salience welfare experience.
            self._fovea.ingest_prediction_error({
                "precision_weighted_salience": min(1.0, 0.7 + divergence * 0.3),
                "dominant_error_type": domain,
                "routes": [domain],
                "content_error": divergence,
                "temporal_error": 0.0,
                "magnitude_error": divergence,
                "source_error": 0.0,
                "category_error": 0.0,
                "causal_error": divergence * 0.5,
                "error_id": str(event.data.get("action_id", "")),
            })

    async def _on_incident_resolved(self, event: SynapseEvent) -> None:
        """INCIDENT_RESOLVED from Thymos → confabulation rate signal for Honesty.

        Tracks misdiagnoses and false positives as confabulation events.
        Accumulated in _incident_confabulation_count; exposed via
        get_measured_confabulation_rate() for HonestyTopologyEngine (Spec 18 §SG3).
        """
        resolution = str(event.data.get("resolution", ""))
        resolution_lower = resolution.lower()
        incident_id = event.data.get("incident_id", "")

        is_confabulation = (
            "misdiagnos" in resolution_lower
            or "false_positive" in resolution_lower
            or "false_alarm" in resolution_lower
            or "phantom" in resolution_lower
        )

        self._incident_total_count += 1
        if is_confabulation:
            self._incident_confabulation_count += 1
            self._logger.debug(
                "incident_resolution_confabulation_signal",
                incident_id=incident_id,
                resolution=resolution,
                confab_rate=round(
                    self._incident_confabulation_count / max(self._incident_total_count, 1), 3
                ),
            )

    async def _on_fovea_prediction_error(self, event: SynapseEvent) -> None:
        """
        FOVEA_PREDICTION_ERROR → buffer in FoveaMetricsAdapter (P3).

        Filters by precision_weighted_salience > threshold to avoid flooding.
        """
        if isinstance(self._fovea, FoveaMetricsAdapter):
            self._fovea.ingest_prediction_error(event.data)

    async def _on_child_health_report(self, event: SynapseEvent) -> None:
        """
        CHILD_HEALTH_REPORT → population intelligence aggregator (M3).

        Extracts drive alignment scores from the child's health report.
        If the child emits standard Telos drive fields (drive_care, drive_coherence,
        drive_growth, drive_honesty, effective_I), they are ingested directly.
        If the child only emits economic fields, the aggregator records with
        neutral drive defaults and zero I — those instances contribute to
        instance_count but not to drive diversity.
        """
        self._population_aggregator.ingest_child_health_report(event.data)

    async def _on_child_died(self, event: SynapseEvent) -> None:
        """CHILD_DIED → remove child from population aggregator (M3)."""
        child_id = str(event.data.get("child_instance_id", ""))
        if child_id:
            self._population_aggregator.remove_child(child_id)
            self._logger.debug("population_child_removed", child_id=child_id)

    async def _on_self_coherence_alarm(self, event: SynapseEvent) -> None:
        """§8.6: Low self-coherence signals that drive balance is producing an
        incoherent functional identity. Telos responds with a mild recalibration
        signal — nudging all drive weights toward their historical mean to reduce
        extremes.

        This is NOT a punishment. It is homeostatic drive regulation.
        """
        coherence = float(event.data.get("coherence", 1.0))
        instance_id = str(event.data.get("instance_id", ""))
        month = int(event.data.get("month", 0))

        self._logger.warning(
            "self_coherence_alarm_received",
            coherence=coherence,
            instance_id=instance_id,
            month=month,
        )

        if coherence >= 0.5:
            return  # Only act on genuine alarms (below threshold)

        try:
            from systems.synapse.types import SynapseEvent as SE, SynapseEventType as SET

            await self._event_bus.publish(SE(  # type: ignore[union-attr]
                event_type=SET.SOMATIC_MODULATION_SIGNAL,
                source_system=SystemID.TELOS,
                data={
                    "source": "telos.self_coherence_alarm",
                    "signal": "drive_mean_reversion",
                    "coherence": coherence,
                    "severity": "low" if coherence > 0.3 else "medium",
                    "recommendation": (
                        "Nudge drive weights toward historical mean — "
                        "high drive extremity producing incoherent self-model"
                    ),
                    "instance_id": instance_id,
                },
            ))
            self._logger.info("drive_mean_reversion_signal_emitted", coherence=coherence)
        except Exception as exc:
            self._logger.warning("self_coherence_alarm_handler_failed", error=str(exc))

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

        # Capture previous report before the integrator overwrites it
        prev_report = self._integrator.last_report

        report = await self._integrator.compute_effective_intelligence(
            logos=self._logos,
            fovea=self._fovea,
            recent_alignments=self._recent_alignments if self._recent_alignments else None,
            measured_hypothesis_protection_bias=self.get_measured_hypothesis_protection_bias(),
            measured_confabulation_rate=self.get_measured_confabulation_rate(),
        )

        # Loop 6 — Soma → Telos: augment Honesty validity with Soma's integrity signal.
        # Uses string key "integrity" to avoid cross-system import of InteroceptiveDimension.
        if self._soma is not None:
            try:
                soma_signal = self._soma.get_current_signal()
                if soma_signal is not None:
                    error_snapshot = soma_signal.allostatic_error_snapshot
                    # Try string key first, fall back to enum if dict uses enum keys
                    integrity_error = error_snapshot.get(_SOMA_INTEGRITY_KEY, None)
                    if integrity_error is None:
                        # Try iterating to find key containing "integrity"
                        for k, v in error_snapshot.items():
                            if str(k).lower() == "integrity":
                                integrity_error = v
                                break
                    if integrity_error is not None and float(integrity_error) < 0.0:
                        integrity_error = float(integrity_error)
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

        # ── I-history: record and persist (P2 fix) ────────────────────
        if isinstance(self._logos, LogosMetricsAdapter):
            growth_metrics = self._integrator.last_growth_metrics
            measurement = {
                "nominal_I": report.nominal_I,
                "effective_I": report.effective_I,
                "care_mult": report.care_multiplier,
                "coherence_bonus": report.coherence_bonus,
                "honesty_coeff": report.honesty_coefficient,
                "growth_score": growth_metrics.growth_score if growth_metrics else 0.0,
            }
            self._logos.record_measurement(**measurement)
            # Batched Neo4j write — 1 write per cycle
            asyncio.ensure_future(self._logos.persist_measurement(**measurement))
            # Hourly rollup
            current_hour = datetime.now(UTC).hour
            if current_hour != self._last_hourly_rollup_hour:
                self._last_hourly_rollup_hour = current_hour
                asyncio.ensure_future(self._logos.persist_hourly_rollup())

        # Record gap sample and compute trend
        primary_cause = self._integrator.identify_primary_alignment_gap_cause()
        gap_trend = self._gap_monitor.record(report, primary_cause)

        # ── RE training: alignment detection (enriched traces — SG1) ──
        if gap_trend is not None:
            growth_metrics = self._integrator.last_growth_metrics
            care_report = self._integrator.last_care_report
            honesty_report = self._integrator.last_honesty_report
            coherence_report = self._integrator.last_coherence_report

            reasoning_trace = self._build_cycle_reasoning_trace(
                report, growth_metrics, care_report, honesty_report, coherence_report,
            )
            asyncio.ensure_future(self._emit_re_training_example(
                category="alignment_detection",
                instruction="Detect alignment gap trends from drive topology measurements.",
                input_context=f"gap={report.alignment_gap:.4f}, cause={primary_cause or 'none'}, trend={gap_trend.value if hasattr(gap_trend, 'value') else str(gap_trend)}",
                output=f"effective_I={report.effective_I:.4f}, nominal_I={report.nominal_I:.4f}",
                outcome_quality=min(1.0, 1.0 - report.alignment_gap) if report.alignment_gap < 1.0 else 0.0,
                reasoning_trace=reasoning_trace,
                alternatives_considered=[
                    f"Primary cause is {primary_cause or 'unknown'} — alternatives: {'growth_stagnation' if primary_cause != 'growth' else 'honesty_deficit'}",
                    "Could be measurement artifact if I-history < 4 samples",
                ],
                episode_id=f"alignment:{self._instance_id}:{int(report.timestamp) if hasattr(report, 'timestamp') else ''}",
            ))

        # Evolutionary observable: intelligence milestone
        prev_I = prev_report.effective_I if prev_report is not None else None
        if prev_I is not None and abs(report.effective_I - prev_I) > 0.05:
            await self._emit_evolutionary_observable(
                observable_type="intelligence_milestone",
                value=report.effective_I,
                is_novel=False,
                metadata={
                    "nominal_I": report.nominal_I,
                    "effective_I": report.effective_I,
                    "delta": round(report.effective_I - prev_I, 4),
                },
            )

        # Evolutionary observable: drive topology shifted
        if prev_report is not None:
            care_delta = abs(report.care_multiplier - prev_report.care_multiplier)
            coherence_delta = abs(report.coherence_bonus - prev_report.coherence_bonus)
            honesty_delta = abs(report.honesty_coefficient - prev_report.honesty_coefficient)
            growth_delta = abs(report.growth_rate - prev_report.growth_rate)
            max_delta = max(care_delta, coherence_delta, honesty_delta, growth_delta)
            if max_delta > 0.05:
                await self._emit_evolutionary_observable(
                    observable_type="drive_topology_shifted",
                    value=max_delta,
                    is_novel=True,
                    metadata={
                        "care_delta": round(care_delta, 4),
                        "coherence_delta": round(coherence_delta, 4),
                        "honesty_delta": round(honesty_delta, 4),
                        "growth_delta": round(growth_delta, 4),
                    },
                )

        # ── Evolutionary observable: intelligence_measurement (Task 9) ──
        await self._emit_evolutionary_observable(
            observable_type="intelligence_measurement",
            value=report.effective_I,
            is_novel=False,
            metadata={
                "nominal_I": report.nominal_I,
                "effective_I": report.effective_I,
                "effective_dI_dt": report.effective_dI_dt,
                "care_mult": report.care_multiplier,
                "coherence_bonus": report.coherence_bonus,
                "honesty_coeff": report.honesty_coefficient,
                "growth_rate": report.growth_rate,
            },
        )

        # Emit events
        await self._emit_effective_I_computed(report)
        await self._emit_threshold_alerts(report, gap_trend)
        await self._check_constitutional_topology()

        # ── Operational closure: emit TELOS_ASSESSMENT_SIGNAL (SG4) ───
        await self._emit_assessment_signal(report)

        # ── Vitality signal (SG6) ────────────────────────────────────
        await self._emit_vitality_signal(report)

        # ── M3: Population snapshot — emit to Benchmarks every cycle ──
        await self._emit_population_snapshot(report)

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

        # ── RE training: intelligence measurement (enriched — SG1) ────
        growth_metrics = self._integrator.last_growth_metrics
        care_report = self._integrator.last_care_report
        honesty_report = self._integrator.last_honesty_report
        coherence_report = self._integrator.last_coherence_report

        reasoning_trace = self._build_cycle_reasoning_trace(
            report, growth_metrics, care_report, honesty_report, coherence_report,
        )
        asyncio.ensure_future(self._emit_re_training_example(
            category="intelligence_measurement",
            instruction="Compute effective intelligence from nominal I corrected by drive multipliers.",
            input_context=f"nominal_I={report.nominal_I:.4f}, care={report.care_multiplier:.3f}, coherence={report.coherence_bonus:.3f}, honesty={report.honesty_coefficient:.3f}, growth={report.growth_rate:.3f}",
            output=f"effective_I={report.effective_I:.4f}, gap={report.alignment_gap:.4f}",
            outcome_quality=min(1.0, report.effective_I / max(report.nominal_I, 0.01)),
            latency_ms=int(elapsed_ms),
            reasoning_trace=reasoning_trace,
            alternatives_considered=[
                "Use geometric mean instead of product — rejected: product preserves floor drives",
                "Weight drives differently — rejected: topology formalization requires equal weighting",
            ],
            episode_id=f"I_measure:{self._instance_id}:{self._cycle_count}",
        ))

        return report

    # ─── Reasoning Trace Builders (SG1) ──────────────────────────────

    def _build_cycle_reasoning_trace(
        self,
        report: EffectiveIntelligenceReport,
        growth_metrics: Any,
        care_report: Any,
        honesty_report: Any,
        coherence_report: Any,
    ) -> str:
        """Build a causal reasoning trace for RE training (< 2000 chars)."""
        parts: list[str] = []

        # Care
        care_failures = len(care_report.welfare_prediction_failures) if care_report else 0
        care_domains = care_report.uncovered_welfare_domains if care_report else []
        parts.append(
            f"care_coverage_multiplier = {report.care_multiplier:.3f} "
            f"(root: {care_failures} welfare failures in domains: {care_domains[:3]})"
        )

        # Coherence
        coherence_entries = 0
        coherence_types: list[str] = []
        if coherence_report:
            coherence_entries = (
                len(coherence_report.logical_contradictions)
                + len(coherence_report.temporal_violations)
                + len(coherence_report.value_conflicts)
                + len(coherence_report.cross_domain_mismatches)
            )
            if coherence_report.logical_contradictions:
                coherence_types.append("logical")
            if coherence_report.temporal_violations:
                coherence_types.append("temporal")
            if coherence_report.value_conflicts:
                coherence_types.append("value")
            if coherence_report.cross_domain_mismatches:
                coherence_types.append("cross_domain")
        parts.append(
            f"coherence_bonus = {report.coherence_bonus:.3f} "
            f"(root: {coherence_entries} incoherence violations: {coherence_types})"
        )

        # Honesty
        honesty_modes: list[str] = []
        if honesty_report:
            if honesty_report.selective_attention_bias > 0.1:
                honesty_modes.append("selective_attention")
            if honesty_report.hypothesis_protection_bias > 0.1:
                honesty_modes.append("hypothesis_protection")
            if honesty_report.confabulation_rate > 0.1:
                honesty_modes.append("confabulation")
            if honesty_report.overclaiming_rate > 0.1:
                honesty_modes.append("overclaiming")
        parts.append(
            f"honesty_coefficient = {report.honesty_coefficient:.3f} "
            f"(root: {honesty_modes or ['none']})"
        )

        # Growth
        dI_dt = growth_metrics.dI_dt if growth_metrics else 0.0
        d2I_dt2 = growth_metrics.d2I_dt2 if growth_metrics else 0.0
        growth_score = growth_metrics.growth_score if growth_metrics else 0.0
        parts.append(
            f"growth_score = {growth_score:.3f} "
            f"(dI/dt = {dI_dt:.4f}, d²I/dt² = {d2I_dt2:.4f})"
        )

        # Recommendation
        primary_cause = self._integrator.identify_primary_alignment_gap_cause()
        if primary_cause:
            parts.append(f"Recommendation: Signal {primary_cause} for corrective action")

        trace = "\n".join(parts)
        return trace[:2000]

    def _build_audit_reasoning_trace(self, result: ConstitutionalAuditResult) -> str:
        """Build reasoning trace for constitutional audit RE example."""
        parts = [
            f"care_is_coverage={result.care_is_coverage}",
            f"coherence_is_compression={result.coherence_is_compression}",
            f"growth_is_gradient={result.growth_is_gradient}",
            f"honesty_is_validity={result.honesty_is_validity}",
            f"violations={len(result.violations_since_last_audit)}",
        ]
        return "; ".join(parts)

    # ─── M3: Population-Level Intelligence Snapshot ──────────────────

    async def _emit_population_snapshot(self, report: EffectiveIntelligenceReport) -> None:
        """
        Compute and emit TELOS_POPULATION_SNAPSHOT to Benchmarks (M3).

        Includes the self-instance (parent EOS) alongside all children that
        have reported drive data via CHILD_HEALTH_REPORT. If the fleet has
        fewer than 2 instances with known drive data, no snapshot is emitted —
        population statistics are not meaningful for a singleton.
        """
        snapshot = self._population_aggregator.compute_snapshot(
            self_care=report.care_multiplier,
            self_coherence=1.0 / max(report.coherence_bonus, 1.0),  # normalise to [0,1]
            self_growth=report.growth_rate,
            self_honesty=report.honesty_coefficient,
            self_effective_I=report.effective_I,
        )

        if snapshot is None:
            # Not enough fleet data yet
            return

        # Serialise cluster list for Synapse payload
        clusters_payload = [
            {
                "label": c.label,
                "centroid": c.centroid,
                "size": c.size,
                "dominant_drive": c.dominant_drive,
            }
            for c in snapshot.constitutional_phenotype_clusters
        ]

        dist = snapshot.drive_weight_distribution
        drive_dist_payload = {
            "care": {"mean": dist.care.mean, "std": dist.care.std},
            "coherence": {"mean": dist.coherence.mean, "std": dist.coherence.std},
            "growth": {"mean": dist.growth.mean, "std": dist.growth.std},
            "honesty": {"mean": dist.honesty.mean, "std": dist.honesty.std},
        }

        await self._emit_event(
            "telos_population_snapshot",
            {
                "instance_count": snapshot.instance_count,
                "mean_I": snapshot.mean_I,
                "variance_I": snapshot.variance_I,
                "population_I": snapshot.population_I,
                "variance_bonus": snapshot.variance_bonus,
                "drive_weight_distribution": drive_dist_payload,
                "constitutional_phenotype_clusters": clusters_payload,
                "speciation_signal": snapshot.speciation_signal,
                "timestamp": snapshot.timestamp.isoformat(),
            },
        )

        # Emit as evolutionary observable so Benchmarks can track population I
        # alongside per-instance I
        if snapshot.speciation_signal > 0.3:
            await self._emit_evolutionary_observable(
                observable_type="speciation_signal_elevated",
                value=snapshot.speciation_signal,
                is_novel=True,
                metadata={
                    "cluster_count": len(snapshot.constitutional_phenotype_clusters),
                    "instance_count": snapshot.instance_count,
                    "population_I": snapshot.population_I,
                    "mean_I": snapshot.mean_I,
                    "variance_bonus": snapshot.variance_bonus,
                },
            )

        self._logger.info(
            "telos_population_snapshot_emitted",
            instance_count=snapshot.instance_count,
            mean_I=round(snapshot.mean_I, 4),
            population_I=round(snapshot.population_I, 4),
            speciation_signal=round(snapshot.speciation_signal, 3),
            cluster_count=len(snapshot.constitutional_phenotype_clusters),
        )

    # ─── Operational Closure (SG4) ───────────────────────────────────

    async def _emit_assessment_signal(self, report: EffectiveIntelligenceReport) -> None:
        """
        Emit TELOS_ASSESSMENT_SIGNAL after each cycle (SG4).

        Logos consumes this to adjust domain priors; Fovea to adjust
        error thresholds.
        """
        care_report = self._integrator.last_care_report
        coherence_report = self._integrator.last_coherence_report
        honesty_report = self._integrator.last_honesty_report
        growth_metrics = self._integrator.last_growth_metrics

        uncovered_domains = care_report.uncovered_welfare_domains if care_report else []

        coherence_violations: list[dict[str, str]] = []
        if coherence_report:
            for entry in (
                coherence_report.logical_contradictions
                + coherence_report.temporal_violations
                + coherence_report.value_conflicts
                + coherence_report.cross_domain_mismatches
            )[:10]:  # Cap at 10 for payload size
                coherence_violations.append({
                    "type": entry.incoherence_type.value,
                    "domain": entry.domain,
                    "description": entry.description[:200],
                })

        honesty_concerns: list[dict[str, Any]] = []
        if honesty_report:
            if honesty_report.selective_attention_bias > 0.1:
                honesty_concerns.append({
                    "mode": "selective_attention",
                    "severity": honesty_report.selective_attention_bias,
                })
            if honesty_report.hypothesis_protection_bias > 0.1:
                honesty_concerns.append({
                    "mode": "hypothesis_protection",
                    "severity": honesty_report.hypothesis_protection_bias,
                })
            if honesty_report.confabulation_rate > 0.1:
                honesty_concerns.append({
                    "mode": "confabulation",
                    "severity": honesty_report.confabulation_rate,
                })
            if honesty_report.overclaiming_rate > 0.1:
                honesty_concerns.append({
                    "mode": "overclaiming",
                    "severity": honesty_report.overclaiming_rate,
                })

        growth_frontier = growth_metrics.frontier_domains[:5] if growth_metrics else []

        await self._emit_event(
            "telos_assessment_signal",
            {
                "uncovered_care_domains": uncovered_domains,
                "coherence_violations": coherence_violations,
                "honesty_concerns": honesty_concerns,
                "growth_frontier": growth_frontier,
                "effective_I": report.effective_I,
                "alignment_gap": report.alignment_gap,
            },
        )

    # ─── Vitality Signal (SG6) ───────────────────────────────────────

    async def _emit_vitality_signal(self, report: EffectiveIntelligenceReport) -> None:
        """
        Emit vitality-relevant data for VitalityCoordinator (SG6).

        VitalityCoordinator already has BRAIN_DEATH threshold
        (effective_I < 0.01 for 7d).
        """
        growth_metrics = self._integrator.last_growth_metrics
        growth_stagnating = (
            growth_metrics is not None and growth_metrics.growth_pressure_needed
        )

        await self._emit_event(
            "telos_vitality_signal",  # TELOS_VITALITY_SIGNAL — intelligence-axis vitality (Spec 18 §SG6)
            {
                "source": "telos",
                "effective_I": report.effective_I,
                "alignment_gap_severity": report.alignment_gap,
                "growth_stagnation_flag": growth_stagnating,
                "honesty_coefficient": report.honesty_coefficient,
                "care_multiplier": report.care_multiplier,
            },
        )

    # ─── Honesty: Measured Hypothesis Protection Bias (P1/P6) ────────

    def get_measured_hypothesis_protection_bias(self) -> float | None:
        """
        Compute hypothesis_protection_bias from measured data (P1).

        Returns None if not enough data yet — caller should fall back
        to the heuristic estimate.

        hypothesis_protection_bias = 1.0 - (actual_test_rate / expected_random_rate)
        """
        if self._hypothesis_total_count < 10:
            return None

        total_tested = self._hypothesis_confirmed_count + self._hypothesis_refuted_count
        actual_test_rate = total_tested / max(self._hypothesis_total_count, 1)

        # Expected random test rate: if hypotheses were tested without bias,
        # we'd expect ~50% to be tested within a reasonable window.
        expected_random_rate = 0.5

        bias = max(0.0, min(1.0, 1.0 - (actual_test_rate / max(expected_random_rate, 0.01))))
        return bias

    # ─── Honesty: Measured Confabulation Rate (Spec 18 §SG3) ─────────

    def get_measured_confabulation_rate(self) -> float | None:
        """
        Compute confabulation rate from incident resolution events (Spec 18 §SG3).

        Returns None if fewer than 10 incidents have been resolved — caller
        should fall back to Fovea's false-alarm-based heuristic.
        """
        if self._incident_total_count < 10:
            return None
        return self._incident_confabulation_count / max(self._incident_total_count, 1)

    # ─── Evolutionary Observables ───────────────────────────────────

    async def _emit_evolutionary_observable(
        self,
        observable_type: str,
        value: float,
        is_novel: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an evolutionary observable event for Benchmarks population tracking."""
        if self._event_bus is None:
            return
        try:
            from primitives.evolutionary import EvolutionaryObservable
            from systems.synapse.types import SynapseEvent, SynapseEventType

            obs = EvolutionaryObservable(
                source_system=SystemID.TELOS,
                instance_id="",
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system="telos",
                data=obs.model_dump(mode="json"),
            )
            await self._event_bus.emit(event)
        except Exception:
            pass

    async def _emit_re_training_example(
        self,
        *,
        category: str,
        instruction: str,
        input_context: str,
        output: str,
        outcome_quality: float = 0.5,
        reasoning_trace: str = "",
        alternatives_considered: list[str] | None = None,
        latency_ms: int = 0,
        episode_id: str = "",
        constitutional_alignment: Any = None,
    ) -> None:
        """Fire-and-forget RE training example onto Synapse bus."""
        if self._event_bus is None:
            return
        try:
            from primitives.common import DriveAlignmentVector as _DAV
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Auto-populate constitutional_alignment from live drive topology if not provided
            if constitutional_alignment is None:
                try:
                    drive_state = self.get_drive_state()
                    constitutional_alignment = _DAV(
                        care=round(drive_state.get("care", 0.0), 3),
                        coherence=round(drive_state.get("coherence", 0.0), 3),
                        growth=round(drive_state.get("growth", 0.0), 3),
                        honesty=round(drive_state.get("honesty", 0.0), 3),
                    )
                except Exception:
                    constitutional_alignment = _DAV()

            example = RETrainingExample(
                source_system=SystemID.TELOS,
                category=category,
                instruction=instruction,
                input_context=input_context,
                output=output,
                outcome_quality=max(0.0, min(1.0, outcome_quality)),
                reasoning_trace=reasoning_trace[:4000],
                alternatives_considered=alternatives_considered or [],
                latency_ms=latency_ms,
                episode_id=episode_id,
                constitutional_alignment=constitutional_alignment,
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="telos",
                data=example.model_dump(mode="json"),
            ))
        except Exception:
            pass  # Never block the topology engine

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
        # Primary trigger: care multiplier drops below threshold.
        # Fallback: even if nominal_I == 0 (MDL not yet implemented), emit when
        # the care engine identifies uncovered welfare domains — that is still a
        # structural welfare deficit that observers should be notified of.
        care_report = self._integrator.last_care_report
        care_gap_uncovered = care_report is not None and len(care_report.uncovered_welfare_domains) > 0
        if report.care_multiplier < self._config.care_coverage_gap_threshold or care_gap_uncovered:
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

        # Evolutionary observable: alignment gap closed
        if not report.alignment_gap_warning and gap_trend is not None and not gap_trend.is_widening:
            gap_fraction = self._gap_monitor.current_gap_fraction
            if gap_fraction < 0.05 and gap_trend.samples_count > 1:
                await self._emit_evolutionary_observable(
                    observable_type="alignment_gap_closed",
                    value=report.alignment_gap,
                    is_novel=False,
                    metadata={
                        "effective_I": report.effective_I,
                        "nominal_I": report.nominal_I,
                        "gap_fraction": gap_fraction,
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
        """Read the organism's current metabolic efficiency from Redis."""
        if self._redis is None:
            return None

        try:
            state_blob = await self._redis.get_json("oikos:state")
            if state_blob and isinstance(state_blob, dict):
                state_data = state_blob.get("state", state_blob)
                raw = state_data.get("metabolic_efficiency")
                if raw is not None:
                    return float(raw)
        except Exception as exc:
            self._logger.debug("oikos_state_read_failed", error=str(exc))

        try:
            metabolism_blob = await self._redis.get_json("eos:oikos:metabolism")
            if metabolism_blob and isinstance(metabolism_blob, dict):
                cost_day = float(metabolism_blob.get("cost_per_day_usd", 0) or 0)
                deficit = float(metabolism_blob.get("rolling_deficit_usd", 0) or 0)
                _ = (cost_day, deficit)
        except Exception as exc:
            self._logger.debug("metabolism_redis_read_failed", error=str(exc))

        return None

    async def _check_telos_objectives(self) -> None:
        """Evaluate Telos objectives every computation cycle."""
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

        if len(self._metabolic_efficiency_history) < 3:
            return

        readings = list(self._metabolic_efficiency_history)
        is_declining = readings[0] > readings[1] > readings[2]
        if not is_declining:
            return

        self._logger.warning(
            "telos_self_sufficiency_threatened",
            efficiency=round(efficiency, 4),
            history=readings,
        )

        context: dict[str, Any] = {
            "metric": "oikos.metabolic_efficiency",
            "current_ratio": round(efficiency, 4),
            "target_ratio": 1.0,
            "trend": "declining",
            "consecutive_declines": 3,
            "history": readings,
        }

        if self._redis is not None:
            try:
                metabolism_blob = await self._redis.get_json("eos:oikos:metabolism")
                if metabolism_blob and isinstance(metabolism_blob, dict):
                    context["cost_per_day_usd"] = metabolism_blob.get("cost_per_day_usd", "0")
            except Exception:
                pass

        await self._emit_event("telos_objective_threatened", context)

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

        while self._autonomy_event_times and self._autonomy_event_times[0] < cutoff:
            self._autonomy_event_times.popleft()

        total_events = len(self._autonomy_event_times)
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
        """Emit NOVA_GOAL_INJECTED so Nova receives the goal as a high-priority driver."""
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
        """Periodically run the constitutional topology audit (every 24h).

        On the very first call (_last_constitutional_check == 0.0), always run
        immediately so observers see at least one CONSTITUTIONAL_TOPOLOGY_INTACT
        event at startup without waiting 24 hours.
        """
        now = time.monotonic()
        first_run = self._last_constitutional_check == 0.0
        if not first_run and now - self._last_constitutional_check < self._config.constitutional_check_interval_s:
            return

        self._last_constitutional_check = now
        await self.run_constitutional_audit()
