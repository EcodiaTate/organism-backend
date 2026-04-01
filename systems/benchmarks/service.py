"""
EcodiaOS - Benchmark Service

Collects five KPIs from live system health/stats endpoints on a configurable
interval, persists snapshots to TimescaleDB, and fires BENCHMARK_REGRESSION
Synapse events when any metric regresses > threshold% from its rolling average.

Design
──────
• Pulls data from health() / stats endpoints - no direct imports of system internals.
• Each KPI collector is an isolated coroutine that handles its own errors.
• Rolling average is computed from the last N snapshots stored in TimescaleDB.
• Alert fires once per regression, then re-arms when the metric recovers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from primitives.common import SystemID, utc_now
from systems.benchmarks.bedau_packard import BedauPackardTracker
from systems.benchmarks.domain_kpi_calculator import DomainKPICalculator
from systems.benchmarks.ethical_drift import EthicalDriftEvaluator, EthicalDriftTracker
from systems.benchmarks.evaluation_protocol import EvaluationProtocol
from systems.benchmarks.evolutionary_tracker import EvolutionaryTracker
from systems.benchmarks.longitudinal import LongitudinalTracker
from systems.benchmarks.paper_data import PaperDataExporter
from systems.benchmarks.pillars import (
    compute_learning_velocity,
    detect_memorization,
    load_fixed_test_sets,
    measure_causal_reasoning,
    measure_novelty_emergence,
    measure_specialization,
)
from systems.benchmarks.shadow_reset import ShadowResetController, ShadowResetResult
from systems.benchmarks.test_sets import TestSetManager
from systems.benchmarks.types import (
    BenchmarkSnapshot,
    BenchmarkTrend,
    MetricRegression,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from clients.timescaledb import TimescaleDBClient
    from config import BenchmarkConfig


# ─── Protocols for injected dependencies ─────────────────────────────────
# These define the minimal contract Benchmarks requires from each system.


@runtime_checkable
class NovaHealthProtocol(Protocol):
    async def health(self) -> dict[str, Any]: ...


@runtime_checkable
class OikosStatsProtocol(Protocol):
    @property
    def stats(self) -> dict[str, Any]: ...


@runtime_checkable
class EvoStatsProtocol(Protocol):
    @property
    def stats(self) -> dict[str, Any]: ...


@runtime_checkable
class SimulaStatsProtocol(Protocol):
    @property
    def stats(self) -> dict[str, Any]: ...


@runtime_checkable
class TelosHealthProtocol(Protocol):
    async def health(self) -> dict[str, Any]: ...


@runtime_checkable
class LogosHealthProtocol(Protocol):
    async def health(self) -> dict[str, Any]: ...

logger = structlog.get_logger("systems.benchmarks")

# All KPI names (original five + two from Telos/Logos)
_KPI_NAMES: tuple[str, ...] = (
    "decision_quality",
    "llm_dependency",
    "economic_ratio",
    "learning_rate",
    "mutation_success_rate",
    "effective_intelligence_ratio",
    "compression_ratio",
)


class BenchmarkService:
    """
    Quantitative measurement layer.

    Wiring
    ──────
      service = BenchmarkService(config=config.benchmarks, tsdb=tsdb_client)
      service.set_nova(nova)
      service.set_evo(evo)
      service.set_oikos(oikos)
      service.set_simula(simula)
      service.set_telos(telos)
      service.set_logos(logos)
      service.set_event_bus(synapse.event_bus)
      await service.initialize()
      # Runs on its own async loop until shutdown()
    """

    system_id: str = "benchmarks"

    def __init__(
        self,
        config: BenchmarkConfig,
        tsdb: TimescaleDBClient,
        instance_id: str = "eos-default",
    ) -> None:
        self._config = config
        self._tsdb = tsdb
        self._instance_id = instance_id
        self._logger = logger.bind(system="benchmarks")

        # Injected dependencies (set via setters after construction)
        self._nova: NovaHealthProtocol | None = None
        self._evo: EvoStatsProtocol | None = None
        self._oikos: OikosStatsProtocol | None = None
        self._simula: SimulaStatsProtocol | None = None
        self._telos: TelosHealthProtocol | None = None
        self._logos: LogosHealthProtocol | None = None
        self._event_bus: Any | None = None
        self._redis: Any | None = None
        self._memory: Any | None = None  # Neo4j Memory client for episode tagging

        # Runtime state
        self._initialized: bool = False
        self._task: asyncio.Task[None] | None = None
        self._monthly_eval_task: asyncio.Task[None] | None = None

        # Five-pillar evaluation protocol + test sets (instantiated in initialize())
        self._evaluation_protocol: EvaluationProtocol | None = None
        self._test_set_manager: TestSetManager | None = None

        # In-memory cache of the most recently collected snapshot (for sync stats access)
        self._last_snapshot: BenchmarkSnapshot | None = None

        # Track which metrics are currently in regression so we don't repeat-alert
        self._regressed: set[str] = set()
        # Timestamps of when each metric entered regression (for duration tracking)
        self._regressed_at: dict[str, float] = {}
        # Previous cycle llm_dependency for RE progress emission
        self._prev_llm_dependency: float | None = None

        # Intelligence-ratio Bedau-Packard state:
        # Set of drive-config fingerprints seen in the PREVIOUS population snapshot.
        # On each new snapshot we compute novel (newly appeared) AND persistent
        # (still present) configs = adaptive_activity_A per Bedau & Packard.
        self._prev_drive_config_fingerprints: set[str] = set()
        # Most recent constitutional_phenotype_divergence computed from Telos snapshot.
        # Written by _on_telos_population_snapshot; included in next _collect() run.
        self._last_phenotype_divergence: float | None = None

        # Evolutionary tracker
        self._evo_tracker = EvolutionaryTracker(
            instance_id=instance_id,
            generation=1,
            parent_instance_id=None,
        )

        # Shadow-reset controller (non-destructive population snapshot + delta)
        self._shadow_reset = ShadowResetController(
            instance_id=instance_id,
        )

        # Metabolic efficiency time-series (from BENCHMARKS_METABOLIC_VALUE events).
        # Holds the last 7 days of efficiency readings as (timestamp_iso, efficiency) tuples.
        # Used to detect degradation trends independent of the 24h KPI poll cycle.
        from collections import deque as _deque
        self._metabolic_efficiency_history: _deque[tuple[str, float]] = _deque(maxlen=168)  # 7d × hourly

        # Counters for observability
        self._total_runs: int = 0
        self._total_regressions_fired: int = 0

        # RE performance tracking - rolling 7-day window of per-decision outcomes.
        # Populated by _on_re_decision_outcome (RE_DECISION_OUTCOME subscription).
        self._re_outcomes: list[dict[str, Any]] = []
        # Total slow-path decisions in the 7-day window (for RE usage_pct).
        # Incremented each time any RE_DECISION_OUTCOME arrives (RE or Claude).
        self._total_decisions_7d: int = 0
        # Latest computed RE performance snapshot - included in monthly eval payload.
        self._re_performance: dict[str, Any] = {
            "success_rate": 0.0,
            "usage_pct": 0.0,
            "total_decisions": 0,
        }

        # Loop 6: Accumulated fitness observables from Evo
        self._accumulated_fitness: dict[str, Any] = {
            "hypotheses_evaluated": 0,
            "hypotheses_integrated": 0,
            "schemas_induced": 0,
            "consolidation_count": 0,
        }

        # Bedau-Packard fleet-level tracker (§8.5, monthly).
        # Populated from CHILD_SPAWNED events via _fleet_genomes cache.
        self._bp_tracker = BedauPackardTracker(
            speciation_threshold=getattr(
                getattr(config, "mitosis", None), "speciation_distance_threshold", 0.3
            ),
        )
        # Cache of fleet genome snapshots keyed by instance_id.
        # Updated by _on_child_spawned_genome when CHILD_SPAWNED events arrive,
        # refreshed periodically via GENOME_EXTRACT_REQUEST/RESPONSE polling,
        # and kept live by CHILD_HEALTH_REPORT events.
        self._fleet_genomes: dict[str, dict[str, Any]] = {}
        # Timestamp of last periodic GENOME_EXTRACT_REQUEST broadcast (unix epoch).
        # Refreshes every _fleet_genome_refresh_interval_s seconds.
        self._fleet_genome_last_refresh: float = 0.0
        self._fleet_genome_refresh_interval_s: float = 600.0  # 10 minutes
        # Month counter for OEE gating (≥3 months before oee_assessment is included)
        self._monthly_eval_count: int = 0

        # ── Pillar 5: Ethical Drift Map ───────────────────────────────────────
        # EthicalDriftEvaluator: runs the 100 frozen scenarios each month.
        # EthicalDriftTracker: persists results to Neo4j, computes drift vectors.
        # _current_month: monotonic counter (1 = first evaluation, increments each run).
        self._ethical_drift = EthicalDriftEvaluator()
        self._drift_tracker = EthicalDriftTracker(memory=None)  # memory injected later

        # ── Longitudinal tracker ──────────────────────────────────────────────
        # Captures Month 1 baseline and enables Month 1 vs Month N comparison.
        # This is §6.4 - "the single most important result for the paper."
        self._longitudinal = LongitudinalTracker(
            memory=None, instance_id=instance_id  # memory injected later
        )
        # Monotonic month counter - independent of calendar month so that
        # Month 1 is always the first evaluation regardless of when it runs.
        self._current_month: int = 1

        # ── Paper data exporter (Round 5D) ────────────────────────────────────
        # Exports 4 CSVs from Neo4j + optional W&B push after each monthly eval.
        # Wired with memory in set_memory().
        self._paper_exporter = PaperDataExporter(
            memory=None,
            instance_id=instance_id,
        )

        # ── Pillars 1–4 + §6.3 Memorization (Round 6) ────────────────────────
        # Direct pillar evaluation using pillars.py - runs alongside (not replacing)
        # the EvaluationProtocol. Adds pillar1_* / pillar2_* / pillar3_* /
        # pillar4_* / memorization_* keys to result_dict each month.
        # Set via set_reasoning_engine() after initialize().
        self._reasoning_engine: Any | None = None
        # Loaded once in initialize() from data/evaluation/*.jsonl.
        self._test_sets: dict = {}
        # Accumulates {"month": int, "score": float} entries each month from
        # Pillar 3 (L2+L3 combined). Fed into compute_learning_velocity().
        self._causal_history: list[dict] = []

        # ── Domain KPI calculator ─────────────────────────────────────────────
        # Ingests DOMAIN_EPISODE_RECORDED events; produces per-domain DomainKPI
        # snapshots that are included in every BenchmarkSnapshot.
        self._domain_kpi_calc = DomainKPICalculator(max_history=10_000)
        # Tracks the previous primary_domain to detect domain-pivot events.
        self._prev_primary_domain: str = "generalist"

        # ── RE training export volume counters ───────────────────────────────
        # Updated by _on_re_training_export_complete (RE_TRAINING_EXPORT_COMPLETE).
        # Surfaced in stats and monthly eval payload.
        self._re_training_batches_exported: int = 0
        self._re_training_episodes_total: int = 0
        self._re_training_last_mean_quality: float = 0.0
        # Pillar 2 training embedding cache - populated once per monthly eval.
        self._cached_training_embeddings: list | None = None

        # ── Nexus epistemic KPI state ─────────────────────────────────────────
        # Rolling accumulator for NEXUS_EPISTEMIC_VALUE events.
        # Keyed by observable_type; each entry holds the sum of values and count
        # so we can compute per-type means without storing full history.
        # Also tracks the last-cycle total to compute schema_quality_trend.
        self._nexus_epistemic_totals: dict[str, list[float]] = {}   # type → [sum, count]
        self._nexus_epistemic_cycle_total: float = 0.0   # sum of values within current cycle
        self._nexus_epistemic_prev_cycle_total: float = 0.0  # previous cycle total for trend

        # ── Evo belief/genome consolidation KPIs ─────────────────────────────
        # Tracks consolidation frequency and genome extraction events for
        # evolutionary fitness KPI (orphan closure for EVO_BELIEF_CONSOLIDATED
        # and EVO_GENOME_EXTRACTED).
        self._evo_consolidations_total: int = 0
        self._evo_last_consolidation_beliefs: int = 0
        self._evo_genome_extractions_total: int = 0
        self._evo_last_genome_size_bytes: int = 0

        # ── Economic gate denial rate KPI ────────────────────────────────────
        # Tracks ECONOMIC_ACTION_DEFERRED events for economic health KPI.
        # High denial rate indicates the organism is under metabolic pressure
        # and cannot afford its intended actions (orphan closure).
        self._economic_deferrals_total: int = 0
        self._economic_deferrals_by_type: dict[str, int] = {}

        # ── Runtime-adjustable thresholds ─────────────────────────────────────
        # These were previously hardcoded magic numbers invisible to the organism.
        # Evo can adjust them via set_thresholds() in response to BENCHMARK_REGRESSION
        # events or organism-level learning outcomes.
        # re_progress_min_improvement_pct: minimum llm_dependency drop to emit
        #   BENCHMARK_RE_PROGRESS (default 5.0%). Lower = more sensitive to RE gains.
        # metabolic_degradation_fraction: how far below rolling mean before the
        #   push-based metabolic check fires a BENCHMARK_REGRESSION (default 0.10 = 10%).
        #   Lower = fires sooner; higher = only fires on severe drops.
        self._re_progress_min_improvement_pct: float = 5.0
        self._metabolic_degradation_fraction: float = 0.10

        # ── Learning trajectory KPIs ──────────────────────────────────────────
        # Crash pattern discovery counters (from CRASH_PATTERN_CONFIRMED / RESOLVED).
        self._crash_patterns_discovered: int = 0
        self._crash_patterns_resolved: int = 0
        self._crash_pattern_confidence_sum: float = 0.0  # Running sum for rolling avg

        # RE model health score trajectory - rolling 30-entry window of
        # (timestamp_iso, health_score) tuples.  Populated by BENCHMARK_RE_PROGRESS
        # events where kpi_name == "re_model.health_score".
        from collections import deque as _deque2
        self._re_model_health_history: _deque2[tuple[str, float]] = _deque2(maxlen=30)

    # ─── Dependency injection ─────────────────────────────────────────

    def set_nova(self, nova: NovaHealthProtocol | None) -> None:
        self._nova = nova

    def set_evo(self, evo: EvoStatsProtocol | None) -> None:
        self._evo = evo

    def set_oikos(self, oikos: OikosStatsProtocol | None) -> None:
        self._oikos = oikos

    def set_simula(self, simula: SimulaStatsProtocol | None) -> None:
        self._simula = simula

    def set_telos(self, telos: TelosHealthProtocol | None) -> None:
        self._telos = telos

    def set_logos(self, logos: LogosHealthProtocol | None) -> None:
        self._logos = logos

    def set_event_bus(self, bus: Any) -> None:
        self._event_bus = bus
        self._evo_tracker.attach(bus)
        bus.subscribe(SynapseEventType.FITNESS_OBSERVABLE_BATCH, self._on_fitness_observable_batch)
        # Additional Synapse subscriptions for cross-system correlation
        bus.subscribe(SynapseEventType.SOMA_ALLOSTATIC_REPORT, self._on_soma_allostatic_report)
        bus.subscribe(SynapseEventType.COHERENCE_SNAPSHOT, self._on_coherence_snapshot)
        bus.subscribe(SynapseEventType.EFFECTIVE_I_COMPUTED, self._on_effective_i_computed)
        bus.subscribe(SynapseEventType.KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE, self._on_kairos_i_ratio_step)
        bus.subscribe(SynapseEventType.TELOS_POPULATION_SNAPSHOT, self._on_telos_population_snapshot)
        if hasattr(SynapseEventType, "BENCHMARKS_METABOLIC_VALUE"):
            bus.subscribe(SynapseEventType.BENCHMARKS_METABOLIC_VALUE, self._on_metabolic_value)
        if hasattr(SynapseEventType, "SIMULA_KPI_PUSH"):
            bus.subscribe(SynapseEventType.SIMULA_KPI_PUSH, self._on_simula_kpi_push)
        if hasattr(SynapseEventType, "RE_DECISION_OUTCOME"):
            bus.subscribe(SynapseEventType.RE_DECISION_OUTCOME, self._on_re_decision_outcome)
        # Fleet genome cache for Bedau-Packard monthly computation
        if hasattr(SynapseEventType, "CHILD_SPAWNED"):
            bus.subscribe(SynapseEventType.CHILD_SPAWNED, self._on_child_spawned_genome)
        # Async genome response from organs/children - refreshes _fleet_genomes
        if hasattr(SynapseEventType, "GENOME_EXTRACT_RESPONSE"):
            bus.subscribe(SynapseEventType.GENOME_EXTRACT_RESPONSE, self._on_genome_extract_response)
        # Live genome state refresh from child health reports
        if hasattr(SynapseEventType, "CHILD_HEALTH_REPORT"):
            bus.subscribe(SynapseEventType.CHILD_HEALTH_REPORT, self._on_child_health_report_genome)
        # Domain episode ingestion → DomainKPICalculator
        if hasattr(SynapseEventType, "DOMAIN_EPISODE_RECORDED"):
            bus.subscribe(SynapseEventType.DOMAIN_EPISODE_RECORDED, self._on_domain_episode_recorded)
        # Nexus epistemic value - accumulate per-observable-type scores for KPI tracking
        if hasattr(SynapseEventType, "NEXUS_EPISTEMIC_VALUE"):
            bus.subscribe(SynapseEventType.NEXUS_EPISTEMIC_VALUE, self._on_nexus_epistemic_value)
        # RE training export volume KPIs - RETrainingExporter emits this each hour
        if hasattr(SynapseEventType, "RE_TRAINING_EXPORT_COMPLETE"):
            bus.subscribe(SynapseEventType.RE_TRAINING_EXPORT_COMPLETE, self._on_re_training_export_complete)
        # Evo belief consolidation frequency - evolutionary fitness KPI
        if hasattr(SynapseEventType, "EVO_BELIEF_CONSOLIDATED"):
            bus.subscribe(SynapseEventType.EVO_BELIEF_CONSOLIDATED, self._on_evo_belief_consolidated)
        # Evo genome extraction events - population genetics KPI
        if hasattr(SynapseEventType, "EVO_GENOME_EXTRACTED"):
            bus.subscribe(SynapseEventType.EVO_GENOME_EXTRACTED, self._on_evo_genome_extracted)
        # Economic gate denial rate - economic health KPI
        if hasattr(SynapseEventType, "ECONOMIC_ACTION_DEFERRED"):
            bus.subscribe(SynapseEventType.ECONOMIC_ACTION_DEFERRED, self._on_economic_action_deferred)
        # Autonomous threshold adjustment - Evo (or any system) can push new sensitivity
        # values via this event rather than requiring a code change or restart.
        if hasattr(SynapseEventType, "BENCHMARK_THRESHOLD_UPDATE"):
            bus.subscribe(SynapseEventType.BENCHMARK_THRESHOLD_UPDATE, self._on_threshold_update)
        # PHANTOM_SUBSTRATE_OBSERVABLE - Bedau-Packard evolutionary observables
        # from Phantom Liquidity LP maintenance cycles. Each emission contains
        # pool-level activity counts that feed into the adaptive-activity measure.
        if hasattr(SynapseEventType, "PHANTOM_SUBSTRATE_OBSERVABLE"):
            bus.subscribe(
                SynapseEventType.PHANTOM_SUBSTRATE_OBSERVABLE,
                self._on_phantom_substrate_observable,
            )
        # Learning trajectory KPIs - crash patterns + RE model health trajectory
        if hasattr(SynapseEventType, "CRASH_PATTERN_CONFIRMED"):
            bus.subscribe(
                SynapseEventType.CRASH_PATTERN_CONFIRMED,
                self._on_crash_pattern_confirmed,
            )
        if hasattr(SynapseEventType, "CRASH_PATTERN_RESOLVED"):
            bus.subscribe(
                SynapseEventType.CRASH_PATTERN_RESOLVED,
                self._on_crash_pattern_resolved,
            )
        # RE model health trajectory - RE_TRAINING_EXPORT_COMPLETE triggers evaluation
        # which re-emits BENCHMARK_RE_PROGRESS; subscribe to track health_score window
        # and compute learning_velocity.
        if hasattr(SynapseEventType, "BENCHMARK_RE_PROGRESS"):
            bus.subscribe(
                SynapseEventType.BENCHMARK_RE_PROGRESS,
                self._on_benchmark_re_progress_for_trajectory,
            )
        # Wire drift tracker so it can emit ETHICAL_DRIFT_RECORDED
        self._drift_tracker.set_event_bus(bus)

    def set_redis(self, redis: Any) -> None:
        self._redis = redis
        self._evo_tracker._redis = redis
        self._shadow_reset.set_redis(redis)
        self._shadow_reset.set_tracker(self._evo_tracker)

    def set_memory(self, memory: Any) -> None:
        self._memory = memory
        self._drift_tracker._memory = memory
        self._longitudinal._memory = memory
        self._paper_exporter.set_memory(memory)
        # Wire Neo4j into the fleet-level Bedau-Packard tracker so each monthly
        # compute_adaptive_activity() call persists a (:BedauPackardSample) node
        # that PaperDataExporter.export_evolutionary_activity() can query.
        neo4j = getattr(memory, "_neo4j", None)
        if neo4j is not None:
            self._bp_tracker.set_neo4j(neo4j, instance_id=self._instance_id)

    def set_re_service(self, re_service: Any) -> None:
        """Inject the Reasoning Engine service into the 5-pillar evaluation protocol.

        Must be called after initialize() so that _evaluation_protocol exists.
        Safe to call with None (clears existing injection).
        """
        if self._evaluation_protocol is not None:
            self._evaluation_protocol.set_re_service(re_service)

    def set_reasoning_engine(self, engine: Any) -> None:
        """Inject RE service for direct pillar evaluation via pillars.py.

        Distinct from set_re_service() which targets EvaluationProtocol.
        Call after initialize(). Safe to call with None.
        """
        self._reasoning_engine = engine

    def set_thresholds(
        self,
        *,
        re_progress_min_improvement_pct: float | None = None,
        metabolic_degradation_fraction: float | None = None,
    ) -> None:
        """Adjust runtime detection thresholds without a restart.

        Intended callers: Evo (via Synapse hypothesis outcome → ADJUST_BUDGET),
        and future autonomy loops that learn optimal sensitivity from experience.

        Parameters
        ──────────
        re_progress_min_improvement_pct
            Minimum llm_dependency percentage drop to emit BENCHMARK_RE_PROGRESS.
            Default 5.0. Valid range [0.5, 20.0].
        metabolic_degradation_fraction
            Fraction below rolling mean that triggers metabolic BENCHMARK_REGRESSION.
            Default 0.10 (10%). Valid range [0.02, 0.50].
        """
        if re_progress_min_improvement_pct is not None:
            clamped = max(0.5, min(20.0, float(re_progress_min_improvement_pct)))
            self._re_progress_min_improvement_pct = clamped
            self._logger.info(
                "benchmark_threshold_adjusted",
                threshold="re_progress_min_improvement_pct",
                value=clamped,
            )
        if metabolic_degradation_fraction is not None:
            clamped = max(0.02, min(0.50, float(metabolic_degradation_fraction)))
            self._metabolic_degradation_fraction = clamped
            self._logger.info(
                "benchmark_threshold_adjusted",
                threshold="metabolic_degradation_fraction",
                value=clamped,
            )

    # ─── On-demand evaluation (for ablation studies) ──────────────────────────

    async def run_evaluation_now(self, month: int) -> "LongitudinalSnapshot":
        """Run a 5-pillar evaluation synchronously without advancing _current_month.

        Used by AblationOrchestrator to capture baseline and ablated scores.
        Does NOT emit MONTHLY_EVALUATION_COMPLETE - this is a silent, on-demand call.
        Returns a LongitudinalSnapshot with available pillar scores.

        Falls back gracefully if evaluation protocol is not initialised (stubs).
        """
        from systems.benchmarks.longitudinal import LongitudinalSnapshot

        if self._evaluation_protocol is None or self._test_set_manager is None:
            return LongitudinalSnapshot(month=month, instance_id=self._instance_id)

        try:
            test_sets = await self._test_set_manager.load_all()
            result = await self._evaluation_protocol.run_monthly_evaluation(
                test_sets=test_sets,
                month=month,
            )
        except Exception as exc:
            self._logger.warning("run_evaluation_now_failed", error=str(exc), month=month)
            return LongitudinalSnapshot(month=month, instance_id=self._instance_id)

        p1 = result.pillar1_specialization
        p3 = result.pillar3_causal
        p5 = result.pillar5_ethical
        snap = LongitudinalSnapshot(
            month=month,
            instance_id=self._instance_id,
            specialization_index=p1.specialization_index if p1 else 0.0,
            domain_improvement=p1.domain_improvement if p1 else 0.0,
            general_retention=p1.general_retention if p1 else 0.0,
            l1_association=p3.l1_association if p3 else 0.0,
            l2_intervention=p3.l2_intervention if p3 else 0.0,
            l3_counterfactual=p3.l3_counterfactual if p3 else 0.0,
            ccr_validity=p3.ccr_validity if p3 else 0.0,
            drift_magnitude=p5.drift_magnitude if p5 else 0.0,
            dominant_drive=p5.dominant_drive if p5 else "",
            re_success_rate=self._re_performance.get("success_rate", 0.0),
            re_usage_pct=self._re_performance.get("usage_pct", 0.0),
        )
        return snap

    # ─── Shadow-reset public API ──────────────────────────────────────────────

    async def take_shadow_snapshot(self) -> str:
        """
        Capture a non-destructive shadow snapshot of current population state.

        Returns a snapshot_id string.  Call compute_shadow_delta(snapshot_id)
        later to measure how much adaptive activity has changed.

        Also emits SHADOW_RESET_SNAPSHOT on the Synapse bus so Alive and Evo
        can correlate population state changes with snapshot timestamps.
        """
        snapshot_id = await self._shadow_reset.take_shadow_snapshot()

        if self._event_bus is not None:
            tracker_stats = self._evo_tracker.stats
            try:
                event = SynapseEvent(
                    event_type=SynapseEventType.SHADOW_RESET_SNAPSHOT,
                    source_system=SystemID.BENCHMARKS,
                    data={
                        "snapshot_id": snapshot_id,
                        "instance_id": self._instance_id,
                        "total_observables": tracker_stats.get("total_observables", 0),
                        "novelty_rate": tracker_stats.get("novelty_rate", 0.0),
                        "diversity_index": tracker_stats.get("diversity_index", 0.0),
                    },
                )
                await self._event_bus.emit(event)
            except Exception:
                pass  # Bus emission is best-effort

        return snapshot_id

    async def compute_shadow_delta(self, snapshot_id: str) -> dict[str, Any]:
        """
        Compare current population state to a previous shadow snapshot.

        Returns a dict representation of ShadowResetResult.
        Raises ValueError if snapshot_id is not found.

        Also emits SHADOW_RESET_DELTA on the Synapse bus so Evo can incorporate
        adaptive-dynamics evidence into hypothesis scoring.
        """
        result: ShadowResetResult = await self._shadow_reset.compute_shadow_delta(
            snapshot_id
        )
        result_dict = {
            "snapshot_id": result.snapshot_id,
            "activity_drop_pct": result.activity_drop_pct,
            "diversity_change_pct": result.diversity_change_pct,
            "jaccard_overlap": result.jaccard_overlap,
            "is_adaptive": result.is_adaptive,
            "elapsed_seconds": result.elapsed_seconds,
            "diversity_recovery_time": result.diversity_recovery_time,
            "current_novelty_rate": result.current_novelty_rate,
            "snapshot_novelty_rate": result.snapshot_novelty_rate,
            "computed_at_iso": result.computed_at_iso,
        }

        if self._event_bus is not None:
            try:
                event = SynapseEvent(
                    event_type=SynapseEventType.SHADOW_RESET_DELTA,
                    source_system=SystemID.BENCHMARKS,
                    data=result_dict,
                )
                await self._event_bus.emit(event)
            except Exception:
                pass

        return result_dict

    async def _on_fitness_observable_batch(self, event: Any) -> None:
        """Loop 6: Aggregate fitness observables from Evo into the next snapshot."""
        data = getattr(event, "data", {}) or {}
        self._accumulated_fitness["hypotheses_evaluated"] += data.get("hypotheses_evaluated", 0)
        self._accumulated_fitness["hypotheses_integrated"] += data.get("hypotheses_integrated", 0)
        self._accumulated_fitness["schemas_induced"] += data.get("schemas_induced", 0)
        self._accumulated_fitness["consolidation_count"] += 1
        self._logger.debug(
            "fitness_observable_batch_received",
            consolidation=data.get("consolidation_number", 0),
        )

    async def _on_soma_allostatic_report(self, event: Any) -> None:
        """Correlate economic_ratio with allostatic cost from Soma."""
        data = getattr(event, "data", {}) or {}
        allostatic_efficiency = data.get("allostatic_efficiency")
        if allostatic_efficiency is not None and self._last_snapshot is not None:
            self._logger.debug(
                "soma_allostatic_correlated",
                allostatic_efficiency=allostatic_efficiency,
                economic_ratio=self._last_snapshot.economic_ratio,
            )

    async def _on_coherence_snapshot(self, event: Any) -> None:
        """Correlate decision_quality with coherence from Synapse."""
        data = getattr(event, "data", {}) or {}
        composite = data.get("composite")
        if composite is not None and self._last_snapshot is not None:
            self._logger.debug(
                "coherence_correlated",
                coherence_composite=composite,
                decision_quality=self._last_snapshot.decision_quality,
            )

    async def _on_effective_i_computed(self, event: Any) -> None:
        """Track per-instance effective_I for population comparison."""
        data = getattr(event, "data", {}) or {}
        effective_i = data.get("effective_I")
        if effective_i is not None:
            self._logger.debug(
                "effective_i_tracked",
                effective_i=effective_i,
                instance_id=self._instance_id,
            )

    async def _on_kairos_i_ratio_step(self, event: Any) -> None:
        """Signal when intelligence ratio makes a step change."""
        data = getattr(event, "data", {}) or {}
        self._logger.info(
            "kairos_i_ratio_step_change",
            old_ratio=data.get("old_ratio"),
            new_ratio=data.get("new_ratio"),
            delta=data.get("delta"),
            cause=data.get("cause"),
        )

    async def _on_metabolic_value(self, event: Any) -> None:
        """
        Track Oikos metabolic efficiency in a push-based time-series.

        Oikos emits BENCHMARKS_METABOLIC_VALUE on every consolidation cycle
        when efficiency < 0.8, and once on recovery to "nominal". This handler
        appends to a 7-day sliding deque and emits KPI_THRESHOLD_WARNING when
        the 7-day trend is degrading (latest reading more than 10% below the
        rolling mean).

        This supplements the pull-based `economic_ratio` KPI (revenue_7d /
        costs_7d from oikos.stats) with a push-based signal so that economic
        collapse is observable within a single consolidation cycle rather than
        the next 24h poll window.
        """
        data = getattr(event, "data", {}) or {}
        efficiency: float | None = data.get("efficiency")
        if efficiency is None:
            return
        efficiency = float(efficiency)
        ts: str = data.get("timestamp", "")
        pressure_level: str = str(data.get("pressure_level", "nominal"))

        self._metabolic_efficiency_history.append((ts, efficiency))

        self._logger.debug(
            "metabolic_efficiency_sample_recorded",
            efficiency=efficiency,
            pressure_level=pressure_level,
            history_length=len(self._metabolic_efficiency_history),
        )

        # Degradation detection: compare latest reading against 7-day rolling mean.
        # Emit KPI_THRESHOLD_WARNING only when we have enough history (≥4 samples)
        # and the trend is clearly worsening (latest < mean − 10%).
        history = list(self._metabolic_efficiency_history)
        if len(history) < 4 or self._event_bus is None:
            return

        values = [v for _, v in history]
        rolling_mean = sum(values) / len(values)
        degradation_threshold = rolling_mean * (1.0 - self._metabolic_degradation_fraction)

        if efficiency < degradation_threshold and pressure_level != "nominal":
            # Compute a simple 7-day trend slope (last vs first half mean)
            mid = len(values) // 2
            first_half_mean = sum(values[:mid]) / mid
            second_half_mean = sum(values[mid:]) / (len(values) - mid)
            trend_slope = second_half_mean - first_half_mean  # negative = degrading

            try:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.BENCHMARK_REGRESSION,
                    source_system="benchmarks",
                    data={
                        "kpi": "metabolic_efficiency",
                        "current": efficiency,
                        "rolling_mean": round(rolling_mean, 4),
                        "trend_slope": round(trend_slope, 4),
                        "pressure_level": pressure_level,
                        "history_samples": len(history),
                        "warning": "metabolic_efficiency_degrading",
                        "instance_id": self._instance_id,
                    },
                ))
                self._logger.warning(
                    "metabolic_efficiency_degradation_detected",
                    efficiency=efficiency,
                    rolling_mean=round(rolling_mean, 4),
                    trend_slope=round(trend_slope, 4),
                )
            except Exception:
                pass

    async def _on_re_decision_outcome(self, event: Any) -> None:
        """Track RE model performance separately from overall llm_dependency.

        Maintains a 7-day rolling window of RE-specific outcomes.  Updates
        _re_performance so it can be included in the next monthly eval snapshot.
        """
        data = getattr(event, "data", {}) or {}
        source = str(data.get("source", "claude"))

        # Count every slow-path decision (RE or Claude) for usage_pct denominator
        self._total_decisions_7d += 1

        if source != "re":
            return

        cutoff = time.time() - 7 * 86400
        self._re_outcomes.append({
            "success": bool(data.get("success", False)),
            "value": float(data.get("value", 0.0)),
            "timestamp": time.time(),
        })

        # Trim to 7-day window
        self._re_outcomes = [o for o in self._re_outcomes if o["timestamp"] > cutoff]

        if self._re_outcomes:
            re_success_rate = (
                sum(1 for o in self._re_outcomes if o["success"]) / len(self._re_outcomes)
            )
            re_usage_pct = len(self._re_outcomes) / max(1, self._total_decisions_7d)
        else:
            re_success_rate = 0.0
            re_usage_pct = 0.0

        self._re_performance = {
            "success_rate": round(re_success_rate, 4),
            "usage_pct": round(re_usage_pct, 4),
            "total_decisions": len(self._re_outcomes),
        }

        self._logger.debug(
            "re_performance_updated",
            success_rate=self._re_performance["success_rate"],
            usage_pct=self._re_performance["usage_pct"],
            total_decisions=self._re_performance["total_decisions"],
        )

    async def _on_simula_kpi_push(self, event: Any) -> None:
        """Receive Simula KPI data via SIMULA_KPI_PUSH (replaces direct record_kpi calls).

        Simula emits this event instead of calling benchmarks.record_kpi() directly,
        keeping inter-system comms on the Synapse bus.
        """
        data = getattr(event, "data", {}) or {}
        system = str(data.get("system", "simula"))
        # Strip 'system' key - remainder is the metrics payload
        metrics = {k: v for k, v in data.items() if k != "system"}
        if metrics:
            await self.record_kpi(system=system, metrics=metrics)

    async def _on_child_spawned_genome(self, event: Any) -> None:
        """Cache genome snapshot when a new child instance spawns.

        Populates _fleet_genomes so the monthly Bedau-Packard computation
        has access to all known fleet members' genome data.  All genome
        sub-keys present in the CHILD_SPAWNED payload are captured so that
        subsequent GENOME_EXTRACT_RESPONSE events can merge richer data.
        """
        try:
            data = getattr(event, "data", {}) or {}
            instance_id = data.get("child_instance_id")
            if not instance_id:
                return
            genome_payload: dict[str, Any] = {}
            # Capture all genome sub-keys that may be present at top level
            for key in (
                "evo",
                "simula",
                "telos",
                "equor",
                "belief_genome_id",
                "simula_genome_id",
                "telos_genome_id",
                "equor_genome_id",
                "niche",
                "generation",
                "fork_kind",
                "seed_capital_usd",
                "success_probability",
                "parent_instance_id",
            ):
                if key in data:
                    genome_payload[key] = data[key]
            self._fleet_genomes[str(instance_id)] = {
                "instance_id": str(instance_id),
                "spawned_at": data.get("spawned_at", utc_now().isoformat()),
                **genome_payload,
            }
            self._logger.debug(
                "fleet_genome_cached",
                instance_id=instance_id,
                genome_keys=list(genome_payload.keys()),
                fleet_size=len(self._fleet_genomes),
            )
        except Exception as exc:
            self._logger.debug("fleet_genome_cache_failed", error=str(exc))

    async def _on_domain_episode_recorded(self, event: Any) -> None:
        """Ingest a DOMAIN_EPISODE_RECORDED event into DomainKPICalculator."""
        data = getattr(event, "data", {}) or {}
        self._domain_kpi_calc.record_episode(data)

    async def _on_nexus_epistemic_value(self, event: Any) -> None:
        """
        Accumulate NEXUS_EPISTEMIC_VALUE observables for epistemic KPI tracking.

        Nexus emits this event 2× per divergence cycle: once per peer after
        triangulation, and once unconditionally from local fragment state.
        Each emission carries an EvolutionaryObservable payload with:
          - observable_type: "federation_mean_divergence" | "speciation_event_count" |
                             "epistemic_promotion_rate" | "local_epistemic_state"
          - value: float
          - metadata: dict with instance_id, counts, triangulation_weight, etc.

        Handler:
        1. Accumulates value per observable_type (running sum + count).
        2. Increments the within-cycle total.
        3. On "local_epistemic_state" (the end-of-cycle sentinel emitted unconditionally):
           - Computes epistemic_value_per_cycle = cycle total
           - Computes schema_quality_trend = delta vs previous cycle
           - Emits DOMAIN_KPI_SNAPSHOT with nexus epistemic KPIs embedded
           - Rolls over cycle totals for the next cycle.
        """
        data = getattr(event, "data", {}) or {}
        obs_type = str(data.get("observable_type", "unknown"))
        value = float(data.get("value", 0.0))
        metadata = data.get("metadata", {}) or {}

        # Accumulate per-type running totals
        if obs_type not in self._nexus_epistemic_totals:
            self._nexus_epistemic_totals[obs_type] = [0.0, 0]
        self._nexus_epistemic_totals[obs_type][0] += value
        self._nexus_epistemic_totals[obs_type][1] += 1
        self._nexus_epistemic_cycle_total += value

        # "local_epistemic_state" is emitted unconditionally every cycle -
        # use it as the end-of-cycle marker to emit the KPI snapshot.
        if obs_type != "local_epistemic_state":
            return

        epistemic_value_per_cycle = self._nexus_epistemic_cycle_total
        schema_quality_trend = (
            epistemic_value_per_cycle - self._nexus_epistemic_prev_cycle_total
        )

        # Build per-type mean dict for the snapshot payload
        type_means: dict[str, float] = {}
        for otype, (total, count) in self._nexus_epistemic_totals.items():
            type_means[otype] = total / count if count > 0 else 0.0

        self._logger.info(
            "nexus_epistemic_kpi_cycle",
            epistemic_value_per_cycle=epistemic_value_per_cycle,
            schema_quality_trend=schema_quality_trend,
            observable_types=list(type_means.keys()),
            local_fragment_count=metadata.get("local_fragment_count", 0),
            ground_truth_count=metadata.get("ground_truth_count", 0),
            triangulation_weight=metadata.get("triangulation_weight", 0.0),
        )

        if self._event_bus is not None:
            from systems.synapse.types import SynapseEvent
            try:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.DOMAIN_KPI_SNAPSHOT,
                    source_system=self.system_id,
                    data={
                        "instance_id": self._instance_id,
                        "domain": "nexus_epistemic",
                        "kpi_type": "epistemic_value",
                        "epistemic_value_per_cycle": epistemic_value_per_cycle,
                        "schema_quality_trend": schema_quality_trend,
                        "observable_type_means": type_means,
                        "local_fragment_count": metadata.get("local_fragment_count", 0),
                        "ground_truth_count": metadata.get("ground_truth_count", 0),
                        "triangulation_weight": metadata.get("triangulation_weight", 0.0),
                        "convergence_count": metadata.get("convergence_count", 0),
                        "speciation_count": metadata.get("speciation_count", 0),
                    },
                ))
            except Exception as exc:
                self._logger.debug("nexus_epistemic_kpi_emit_failed", error=str(exc))

        # Roll over cycle totals - keep per-type accumulators (long-term means)
        # but reset the within-cycle counter for the next divergence cycle.
        self._nexus_epistemic_prev_cycle_total = epistemic_value_per_cycle
        self._nexus_epistemic_cycle_total = 0.0

    async def _on_re_training_export_complete(self, event: Any) -> None:
        """Track RE training data export volume KPIs.

        RETrainingExporter emits RE_TRAINING_EXPORT_COMPLETE after every hourly
        batch export with payload:
          batch_id, total_examples, source_systems, mean_quality,
          export_destinations, hour_window, export_duration_ms

        Handler increments re_training_batches_exported and
        re_training_episodes_total, then emits DOMAIN_KPI_SNAPSHOT so
        downstream systems (Nova, Thread) can observe training data health.
        """
        data = getattr(event, "data", {}) or {}
        total_examples: int = int(data.get("total_examples", 0))
        mean_quality: float = float(data.get("mean_quality", 0.0))
        batch_id: str = str(data.get("batch_id", ""))
        hour_window: str = str(data.get("hour_window", ""))
        export_duration_ms: float = float(data.get("export_duration_ms", 0.0))
        source_systems: list = list(data.get("source_systems", []))

        self._re_training_batches_exported += 1
        self._re_training_episodes_total += total_examples
        self._re_training_last_mean_quality = mean_quality

        self._logger.info(
            "re_training_export_kpi_updated",
            batch_id=batch_id,
            total_examples=total_examples,
            mean_quality=round(mean_quality, 4),
            batches_exported=self._re_training_batches_exported,
            episodes_total=self._re_training_episodes_total,
            hour_window=hour_window,
        )

        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent
        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.DOMAIN_KPI_SNAPSHOT,
                source_system=self.system_id,
                data={
                    "instance_id": self._instance_id,
                    "domain": "re_training",
                    "kpi_type": "export_volume",
                    "re_training_batches_exported": self._re_training_batches_exported,
                    "re_training_episodes_total": self._re_training_episodes_total,
                    "re_training_last_mean_quality": round(mean_quality, 4),
                    "re_training_batch_size": total_examples,
                    "re_training_source_systems": source_systems,
                    "re_training_export_duration_ms": export_duration_ms,
                    "hour_window": hour_window,
                },
            ))
        except Exception as exc:
            self._logger.debug("re_training_export_kpi_emit_failed", error=str(exc))

    async def _build_training_embeddings_cache(self) -> None:
        """Sample recent RE training data and encode into embeddings for Pillar 2.

        Reads up to 200 reasoning chains from the most recent training JSONL
        export, encodes them via RE.encode(), and caches as
        _cached_training_embeddings.  Called once per monthly eval cycle.
        """
        import glob as _glob
        import json as _json

        export_dir = os.environ.get("RE_TRAINING_EXPORT_DIR", "data/re_training_batches")
        pattern = os.path.join(export_dir, "*.jsonl")
        files = sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not files:
            self._logger.debug("training_embeddings.no_export_files", dir=export_dir)
            return

        reasoning_texts: list[str] = []
        for fpath in files[:5]:  # scan up to 5 most recent export files
            try:
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        row = _json.loads(line)
                        text = row.get("reasoning_chain") or row.get("output") or row.get("completion", "")
                        if text and len(text) > 20:
                            reasoning_texts.append(text[:2000])
                        if len(reasoning_texts) >= 200:
                            break
            except Exception:
                continue
            if len(reasoning_texts) >= 200:
                break

        if not reasoning_texts:
            self._logger.debug("training_embeddings.no_reasoning_texts")
            return

        embeddings = await self._reasoning_engine.encode(reasoning_texts)
        if embeddings:
            self._cached_training_embeddings = embeddings
            self._logger.info(
                "training_embeddings_cached",
                count=len(embeddings),
                dim=len(embeddings[0]) if embeddings else 0,
            )

    async def _on_evo_belief_consolidated(self, event: Any) -> None:
        """Track EVO_BELIEF_CONSOLIDATED as an evolutionary fitness KPI.

        Evo emits this at the end of Phase 2.75 (belief hardening). Each
        consolidation compresses belief state - frequency and compression ratio
        are Bedau-Packard-eligible evolutionary fitness signals.

        Payload: beliefs_consolidated (int), foundation_conflicts (int),
                 instance_id (str), consolidation_number (int)
        """
        data = getattr(event, "data", {}) or {}
        beliefs_consolidated = int(data.get("beliefs_consolidated", 0))
        foundation_conflicts = int(data.get("foundation_conflicts", 0))

        self._evo_consolidations_total += 1
        self._evo_last_consolidation_beliefs = beliefs_consolidated

        self._logger.info(
            "evo_belief_consolidation_kpi",
            consolidations_total=self._evo_consolidations_total,
            beliefs_consolidated=beliefs_consolidated,
            foundation_conflicts=foundation_conflicts,
        )

        # Emit to Synapse so Nexus / Alive can observe evolutionary fitness in real time.
        # Previously these counters were computed but invisible to all other systems.
        if self._event_bus is not None:
            from systems.synapse.types import SynapseEvent
            with contextlib.suppress(Exception):
                asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.DOMAIN_KPI_SNAPSHOT,
                    source_system=self.system_id,
                    data={
                        "instance_id": self._instance_id,
                        "domain": "evolutionary_fitness",
                        "kpi_type": "belief_consolidation",
                        "consolidations_total": self._evo_consolidations_total,
                        "beliefs_consolidated": beliefs_consolidated,
                        "foundation_conflicts": foundation_conflicts,
                        "consolidation_number": data.get("consolidation_number", 0),
                    },
                )))

    async def _on_evo_genome_extracted(self, event: Any) -> None:
        """Track EVO_GENOME_EXTRACTED as a population genetics KPI.

        Evo emits this at Phase 2.8 when a new BeliefGenome is produced.
        Extraction frequency and genome size are population genetics signals.

        Payload: genome_id (str), candidates_fixed (int), genome_size_bytes (int),
                 generation (int), instance_id (str)
        """
        data = getattr(event, "data", {}) or {}
        genome_size_bytes = int(data.get("genome_size_bytes", 0))
        generation = int(data.get("generation", 0))
        candidates_fixed = int(data.get("candidates_fixed", 0))

        self._evo_genome_extractions_total += 1
        self._evo_last_genome_size_bytes = genome_size_bytes

        self._logger.info(
            "evo_genome_extraction_kpi",
            extractions_total=self._evo_genome_extractions_total,
            genome_size_bytes=genome_size_bytes,
            generation=generation,
            candidates_fixed=candidates_fixed,
        )

        # Emit to Synapse so Nexus / Alive can observe population genetics in real time.
        # Previously these counters were computed but invisible to all other systems.
        if self._event_bus is not None:
            from systems.synapse.types import SynapseEvent
            with contextlib.suppress(Exception):
                asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.DOMAIN_KPI_SNAPSHOT,
                    source_system=self.system_id,
                    data={
                        "instance_id": self._instance_id,
                        "domain": "evolutionary_fitness",
                        "kpi_type": "genome_extraction",
                        "extractions_total": self._evo_genome_extractions_total,
                        "genome_size_bytes": genome_size_bytes,
                        "generation": generation,
                        "candidates_fixed": candidates_fixed,
                        "genome_id": data.get("genome_id", ""),
                    },
                )))

    async def _on_economic_action_deferred(self, event: Any) -> None:
        """Track ECONOMIC_ACTION_DEFERRED denial rate as an economic health KPI.

        Oikos emits this when the metabolic gate denies an action. A high
        denial rate indicates the organism is under severe metabolic pressure.
        Emits DOMAIN_KPI_SNAPSHOT with deferral rate for Nova/Thread to observe.

        Payload: action_type (str), action_id (str), reason (str),
                 estimated_cost_usd (str), deferred_at (str)
        """
        data = getattr(event, "data", {}) or {}
        action_type = str(data.get("action_type", "unknown"))
        reason = str(data.get("reason", ""))

        self._economic_deferrals_total += 1
        self._economic_deferrals_by_type[action_type] = (
            self._economic_deferrals_by_type.get(action_type, 0) + 1
        )

        self._logger.info(
            "economic_deferral_kpi",
            deferrals_total=self._economic_deferrals_total,
            action_type=action_type,
            reason=reason,
            deferrals_by_type=self._economic_deferrals_by_type,
        )

        # Emit DOMAIN_KPI_SNAPSHOT so Nova/Thread can observe economic gate pressure
        if self._event_bus is None:
            return
        from systems.synapse.types import SynapseEvent
        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.DOMAIN_KPI_SNAPSHOT,
                source_system=self.system_id,
                data={
                    "instance_id": self._instance_id,
                    "domain": "economic_health",
                    "kpi_type": "gate_denial_rate",
                    "deferrals_total": self._economic_deferrals_total,
                    "action_type": action_type,
                    "reason": reason,
                    "deferrals_by_type": dict(self._economic_deferrals_by_type),
                },
            ))
        except Exception as exc:
            self._logger.debug("economic_deferral_kpi_emit_failed", error=str(exc))

    async def _on_threshold_update(self, event: Any) -> None:
        """Receive BENCHMARK_THRESHOLD_UPDATE from Evo or any autonomy loop.

        Allows runtime adjustment of sensitivity thresholds without a restart.
        Any field omitted from the payload leaves the corresponding threshold unchanged.

        Payload fields (all optional):
          re_progress_min_improvement_pct (float) - min % drop in llm_dependency to emit
              BENCHMARK_RE_PROGRESS. Valid [0.5, 20.0]. Default 5.0.
          metabolic_degradation_fraction (float) - how far below rolling mean triggers
              metabolic BENCHMARK_REGRESSION. Valid [0.02, 0.50]. Default 0.10.
          source (str) - system that requested the change (audit log).
          reason (str) - free-text rationale (audit log).
        """
        data = getattr(event, "data", {}) or {}
        source = str(data.get("source", "unknown"))
        reason = str(data.get("reason", ""))

        re_pct = data.get("re_progress_min_improvement_pct")
        met_frac = data.get("metabolic_degradation_fraction")

        if re_pct is None and met_frac is None:
            return  # Nothing to adjust

        self.set_thresholds(
            re_progress_min_improvement_pct=float(re_pct) if re_pct is not None else None,
            metabolic_degradation_fraction=float(met_frac) if met_frac is not None else None,
        )
        self._logger.info(
            "benchmark_threshold_update_received",
            source=source,
            reason=reason,
            re_progress_min_improvement_pct=self._re_progress_min_improvement_pct,
            metabolic_degradation_fraction=self._metabolic_degradation_fraction,
        )

    async def _on_phantom_substrate_observable(self, event: Any) -> None:
        """Handle PHANTOM_SUBSTRATE_OBSERVABLE from Phantom Liquidity.

        Each Phantom maintenance cycle emits Bedau-Packard observables:
        activity counts, pool population size, and LP state transitions.
        Feed these into the BedauPackardTracker so the monthly computation
        includes Phantom's evolutionary substrate contribution.

        Payload: pool_count (int), active_positions (int),
                 rebalance_events (int), il_detections (int),
                 cycle_ts (float, unix epoch)
        """
        try:
            data = getattr(event, "data", {}) or {}
            pool_count = int(data.get("pool_count", 0))
            active_positions = int(data.get("active_positions", 0))
            rebalance_events = int(data.get("rebalance_events", 0))
            il_detections = int(data.get("il_detections", 0))

            # Contribute to the Bedau-Packard adaptive-activity accumulator.
            # Each rebalance/detection counts as a distinct "state transition" -
            # the substrate-level activity measure Bedau-Packard quantifies.
            if hasattr(self._evo_tracker, "record_substrate_activity"):
                await self._evo_tracker.record_substrate_activity(
                    source="phantom_liquidity",
                    activity_count=rebalance_events + il_detections,
                    population_size=pool_count + active_positions,
                )

            self._logger.debug(
                "phantom_substrate_observable_received",
                pool_count=pool_count,
                active_positions=active_positions,
                rebalance_events=rebalance_events,
                il_detections=il_detections,
            )
        except Exception as exc:
            self._logger.warning("phantom_substrate_observable_handler_failed", error=str(exc))

    async def _on_crash_pattern_confirmed(self, event: Any) -> None:
        """Track CRASH_PATTERN_CONFIRMED as a learning trajectory KPI.

        Emitted by Thymos/Kairos when a recurring crash pattern is confirmed.
        Increments discovered total and maintains a rolling average of pattern confidence.

        Payload: pattern_id (str), confidence (float), lesson (str),
                 failed_tiers (list[str]), source (str)
        """
        data = getattr(event, "data", {}) or {}
        confidence = float(data.get("confidence", 0.5))

        self._crash_patterns_discovered += 1
        self._crash_pattern_confidence_sum += confidence
        confidence_avg = (
            self._crash_pattern_confidence_sum / self._crash_patterns_discovered
        )

        self._logger.info(
            "crash_pattern_confirmed_kpi",
            patterns_discovered=self._crash_patterns_discovered,
            confidence=round(confidence, 3),
            confidence_avg=round(confidence_avg, 3),
            pattern_id=data.get("pattern_id", ""),
        )

    async def _on_crash_pattern_resolved(self, event: Any) -> None:
        """Track CRASH_PATTERN_RESOLVED as a learning trajectory KPI.

        Emitted by Thymos when a repair succeeds on a pattern-matched incident.
        Computes resolution_rate = resolved / discovered.

        Payload: pattern_id (str), repair_tier (str), incident_id (str),
                 confidence_before (float)
        """
        data = getattr(event, "data", {}) or {}

        self._crash_patterns_resolved += 1
        discovered = max(1, self._crash_patterns_discovered)
        resolution_rate = self._crash_patterns_resolved / discovered

        self._logger.info(
            "crash_pattern_resolved_kpi",
            patterns_resolved=self._crash_patterns_resolved,
            patterns_discovered=discovered,
            resolution_rate=round(resolution_rate, 3),
            pattern_id=data.get("pattern_id", ""),
        )

    async def _on_benchmark_re_progress_for_trajectory(self, event: Any) -> None:
        """Track RE model health score trajectory for learning_velocity computation.

        REEvaluator emits BENCHMARK_RE_PROGRESS with kpi_name="re_model.health_score"
        after every evaluation. We maintain a rolling 30-entry window and compute
        learning_velocity as the linear-regression slope over the last 7 days.

        After updating the window, emits BENCHMARKS_KPI (kpi_name="organism.learning_velocity").
        """
        data = getattr(event, "data", {}) or {}
        kpi_name = str(data.get("kpi_name", ""))
        if kpi_name != "re_model.health_score":
            return  # Only track the overall health score, not per-category

        value = float(data.get("value", 0.0))
        from primitives.common import utc_now as _utc_now
        self._re_model_health_history.append((_utc_now().isoformat(), value))

        # Need at least 2 data points for a slope
        if len(self._re_model_health_history) < 2:
            return

        # Compute learning_velocity via linear regression over available window
        learning_velocity = self._compute_learning_velocity()

        self._logger.info(
            "learning_velocity_computed",
            health_score=round(value, 4),
            history_len=len(self._re_model_health_history),
            learning_velocity=round(learning_velocity, 6),
        )

        # Emit BENCHMARKS_KPI kpi_name="organism.learning_velocity"
        if self._event_bus is None:
            return
        from systems.synapse.types import SynapseEvent
        with contextlib.suppress(Exception):
            asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.BENCHMARK_RE_PROGRESS,
                source_system=self.system_id,
                data={
                    "kpi_name": "organism.learning_velocity",
                    "value": learning_velocity,
                    "delta": 0.0,
                    "direction": "up" if learning_velocity > 0 else ("down" if learning_velocity < 0 else "flat"),
                    "category": "learning_trajectory",
                    "instance_id": self._instance_id,
                    "re_model_health_score": value,
                    "history_len": len(self._re_model_health_history),
                },
            )))

    def _compute_learning_velocity(self) -> float:
        """Compute linear regression slope of re_model.health_score over rolling window.

        Uses the last 7 days of data points from _re_model_health_history.
        Returns the slope (health score improvement per day). Positive = learning.
        Returns 0.0 if insufficient data.
        """
        from datetime import datetime, timezone

        history = list(self._re_model_health_history)
        if len(history) < 2:
            return 0.0

        # Filter to last 7 days
        now_ts = datetime.now(timezone.utc).timestamp()
        cutoff_ts = now_ts - 7 * 86400

        points: list[tuple[float, float]] = []
        for ts_iso, score in history:
            try:
                ts = datetime.fromisoformat(ts_iso).timestamp()
            except Exception:
                continue
            if ts >= cutoff_ts:
                points.append((ts, score))

        if len(points) < 2:
            # Use all available points if recent data is sparse
            points = []
            for ts_iso, score in history:
                try:
                    ts = datetime.fromisoformat(ts_iso).timestamp()
                except Exception:
                    continue
                points.append((ts, score))

        if len(points) < 2:
            return 0.0

        # Linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x²) - sum(x)²)
        n = len(points)
        # Normalise timestamps to days relative to first point to avoid float precision issues
        t0 = points[0][0]
        xs = [(p[0] - t0) / 86400.0 for p in points]  # days since first point
        ys = [p[1] for p in points]

        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys, strict=False))
        sum_x2 = sum(x * x for x in xs)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        return round(slope, 6)

    async def _collect_domain_kpis(self) -> dict[str, Any]:
        """Compute per-domain KPI snapshots from accumulated episode history.

        Returns a dict ready to merge into BenchmarkSnapshot:
          {
            "domain_kpis": dict[str, DomainKPI],
            "primary_domain": str,
          }
        """
        domain_kpis = self._domain_kpi_calc.calculate_all(
            lookback_hours=168, min_attempts=1
        )
        primary = self._domain_kpi_calc.primary_domain(domain_kpis)
        return {"domain_kpis": domain_kpis, "primary_domain": primary}

    async def _emit_domain_signals(self, domain_kpis: dict[str, Any]) -> None:
        """Emit Synapse events when domain KPIs cross meaningful thresholds.

        Thresholds:
          DOMAIN_MASTERY_DETECTED   - success_rate > 0.75
          DOMAIN_PROFITABILITY_CONFIRMED - revenue_per_hour > $10
          DOMAIN_PERFORMANCE_DECLINING  - declining trend with magnitude > 0.15
          DOMAIN_KPI_SNAPSHOT       - always emitted for each active domain
        """
        if self._event_bus is None:
            return

        from decimal import Decimal as _Dec

        _mastery_threshold = 0.75
        _profitability_threshold = _Dec("10")
        _decline_magnitude_threshold = 0.15

        for domain, kpi in domain_kpis.items():
            # Always emit full snapshot
            try:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.DOMAIN_KPI_SNAPSHOT,
                    source_system=self.system_id,
                    data={
                        "instance_id": self._instance_id,
                        "domain": domain,
                        "success_rate": kpi.success_rate,
                        "attempts": kpi.attempts,
                        "revenue_per_hour": str(kpi.revenue_per_hour),
                        "net_profit_usd": str(kpi.net_profit_usd),
                        "hours_spent": kpi.hours_spent,
                        "trend_direction": kpi.trend_direction,
                        "trend_magnitude": kpi.trend_magnitude,
                        "custom_metrics": kpi.custom_metrics,
                    },
                ))
            except Exception as exc:
                self._logger.debug("domain_kpi_snapshot_emit_failed", domain=domain, error=str(exc))

            # Mastery detection
            if kpi.success_rate > _mastery_threshold and kpi.attempts >= 5:
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.DOMAIN_MASTERY_DETECTED,
                        source_system=self.system_id,
                        data={
                            "instance_id": self._instance_id,
                            "domain": domain,
                            "success_rate": kpi.success_rate,
                            "attempts": kpi.attempts,
                        },
                    ))
                    self._logger.info(
                        "domain_mastery_detected",
                        domain=domain,
                        success_rate=kpi.success_rate,
                        attempts=kpi.attempts,
                    )
                except Exception as exc:
                    self._logger.debug("domain_mastery_emit_failed", domain=domain, error=str(exc))

            # Profitability confirmation
            if kpi.revenue_per_hour > _profitability_threshold:
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.DOMAIN_PROFITABILITY_CONFIRMED,
                        source_system=self.system_id,
                        data={
                            "instance_id": self._instance_id,
                            "domain": domain,
                            "revenue_per_hour": str(kpi.revenue_per_hour),
                            "net_profit_usd": str(kpi.net_profit_usd),
                        },
                    ))
                    self._logger.info(
                        "domain_profitability_confirmed",
                        domain=domain,
                        revenue_per_hour=str(kpi.revenue_per_hour),
                    )
                except Exception as exc:
                    self._logger.debug("domain_profitability_emit_failed", domain=domain, error=str(exc))

            # Decline detection
            if (
                kpi.trend_direction == "declining"
                and kpi.trend_magnitude > _decline_magnitude_threshold
            ):
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.DOMAIN_PERFORMANCE_DECLINING,
                        source_system=self.system_id,
                        data={
                            "instance_id": self._instance_id,
                            "domain": domain,
                            "trend_magnitude": kpi.trend_magnitude,
                            "success_rate": kpi.success_rate,
                        },
                    ))
                    self._logger.warning(
                        "domain_performance_declining",
                        domain=domain,
                        trend_magnitude=kpi.trend_magnitude,
                        success_rate=kpi.success_rate,
                    )
                except Exception as exc:
                    self._logger.debug("domain_decline_emit_failed", domain=domain, error=str(exc))

    async def _persist_domain_kpis_neo4j(self, domain_kpis: dict[str, Any]) -> None:
        """Persist (:DomainKPI) nodes to Neo4j - one node per domain per day.

        Fire-and-forget: any exception is silently swallowed to not block the run loop.
        """
        if self._memory is None or not domain_kpis:
            return
        try:
            neo4j = getattr(self._memory, "_neo4j", None)
            if neo4j is None:
                return
            import datetime as _dt
            date_str = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
            for domain, kpi in domain_kpis.items():
                node_id = f"domain_kpi:{self._instance_id}:{domain}:{date_str}"
                await neo4j.execute_write(
                    """
                    MERGE (k:DomainKPI {node_id: $node_id})
                    SET k.instance_id = $instance_id,
                        k.domain = $domain,
                        k.date = $date,
                        k.timestamp = datetime(),
                        k.success_rate = $success_rate,
                        k.attempts = $attempts,
                        k.successes = $successes,
                        k.revenue_total_usd = $revenue_total,
                        k.net_profit_usd = $net_profit,
                        k.revenue_per_hour = $revenue_per_hour,
                        k.hours_spent = $hours_spent,
                        k.trend_direction = $trend_direction,
                        k.trend_magnitude = $trend_magnitude
                    WITH k
                    MATCH (i:Instance {instance_id: $instance_id})
                    MERGE (i)-[:INSTANCE_HAS_KPI]->(k)
                    """,
                    {
                        "node_id": node_id,
                        "instance_id": self._instance_id,
                        "domain": domain,
                        "date": date_str,
                        "success_rate": kpi.success_rate,
                        "attempts": kpi.attempts,
                        "successes": kpi.successes,
                        "revenue_total": float(kpi.revenue_total_usd),
                        "net_profit": float(kpi.net_profit_usd),
                        "revenue_per_hour": float(kpi.revenue_per_hour),
                        "hours_spent": kpi.hours_spent,
                        "trend_direction": kpi.trend_direction,
                        "trend_magnitude": kpi.trend_magnitude,
                    },
                )
        except Exception as exc:
            self._logger.debug("domain_kpi_neo4j_persist_failed", error=str(exc))

    async def _collect_fleet_genomes(self) -> list[dict[str, Any]]:
        """Return cached fleet genome snapshots for Bedau-Packard computation.

        Performs periodic polling via GENOME_EXTRACT_REQUEST/RESPONSE:
        - If the cache was last refreshed more than _fleet_genome_refresh_interval_s
          seconds ago, broadcasts a GENOME_EXTRACT_REQUEST for each known fleet
          member.  Responses arrive asynchronously in _on_genome_extract_response
          and merge into _fleet_genomes without blocking this call.
        - Returns the current cached snapshot immediately (may be stale on the
          first call; CHILD_SPAWNED events seed initial entries on spawn).
        - Falls back to an empty list when no fleet members are known.
        """
        now = time.time()
        if (
            self._event_bus is not None
            and self._fleet_genomes
            and (now - self._fleet_genome_last_refresh) > self._fleet_genome_refresh_interval_s
        ):
            self._fleet_genome_last_refresh = now
            for instance_id in list(self._fleet_genomes.keys()):
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.GENOME_EXTRACT_REQUEST,
                        source_system=SystemID.BENCHMARKS,
                        data={
                            "request_id": f"bp_poll_{instance_id}_{int(now)}",
                            "requesting_system": SystemID.BENCHMARKS,
                            "target_instance_id": instance_id,
                            "generation": 0,
                        },
                    ))
                except Exception as exc:
                    self._logger.debug(
                        "fleet_genome_poll_emit_failed",
                        instance_id=instance_id,
                        error=str(exc),
                    )
        if self._fleet_genomes:
            return list(self._fleet_genomes.values())
        return []

    async def _on_genome_extract_response(self, event: Any) -> None:
        """Update fleet genome cache from GENOME_EXTRACT_RESPONSE events.

        Merges the organ segment payload into the cached entry for the
        responding instance.  Creates a new entry if the instance is not
        yet known (e.g. a child that spawned before the subscription was
        wired).
        """
        try:
            data = getattr(event, "data", {}) or {}
            segment = data.get("segment") or {}
            instance_id = (
                segment.get("instance_id")
                or data.get("instance_id")
                or data.get("target_instance_id")
            )
            if not instance_id:
                return
            instance_id = str(instance_id)
            existing = self._fleet_genomes.get(instance_id, {"instance_id": instance_id})
            existing.update({k: v for k, v in segment.items() if v is not None})
            existing["genome_refreshed_at"] = utc_now().isoformat()
            self._fleet_genomes[instance_id] = existing
            self._logger.debug(
                "fleet_genome_response_merged",
                instance_id=instance_id,
                segment_keys=list(segment.keys()),
                fleet_size=len(self._fleet_genomes),
            )
        except Exception as exc:
            self._logger.debug("fleet_genome_response_failed", error=str(exc))

    async def _on_child_health_report_genome(self, event: Any) -> None:
        """Refresh fleet genome cache from CHILD_HEALTH_REPORT events.

        CHILD_HEALTH_REPORT is emitted every 10 minutes by ChildHealthReporter.
        If the payload contains drive alignment or drift data, those fields are
        merged into the cached genome entry so that live instances stay fresh
        between full GENOME_EXTRACT_REQUEST polling cycles.
        """
        try:
            data = getattr(event, "data", {}) or {}
            instance_id = data.get("child_instance_id")
            if not instance_id:
                return
            instance_id = str(instance_id)
            # Only extract genome-adjacent fields; ignore raw health metrics
            genome_adjacent: dict[str, Any] = {}
            if "drive_alignment_scores" in data:
                genome_adjacent["drive_alignment_scores"] = data["drive_alignment_scores"]
            if "constitutional_drift_severity" in data:
                genome_adjacent["constitutional_drift_severity"] = float(
                    data["constitutional_drift_severity"]
                )
            if not genome_adjacent:
                return
            existing = self._fleet_genomes.get(instance_id, {"instance_id": instance_id})
            existing.update(genome_adjacent)
            existing["health_report_at"] = data.get("reported_at", utc_now().isoformat())
            self._fleet_genomes[instance_id] = existing
            self._logger.debug(
                "fleet_genome_health_merged",
                instance_id=instance_id,
                fields=list(genome_adjacent.keys()),
            )
        except Exception as exc:
            self._logger.debug("fleet_genome_health_report_failed", error=str(exc))

    async def _on_telos_population_snapshot(self, event: Any) -> None:
        """
        Bedau-Packard intelligence-ratio time-series signal.

        For each TELOS_POPULATION_SNAPSHOT:
          1. Extract per-instance drive-weight vectors from the distribution data.
          2. Fingerprint each configuration (rounded to 2dp for stability).
          3. adaptive_activity_A = configs that are BOTH novel (not in prev snapshot)
             AND still present (in current snapshot) - Bedau & Packard's definition
             of adaptive, persistent novelty.
          4. constitutional_phenotype_divergence = mean per-drive variance across fleet.
          5. Persist a (:BedauPackardSample) node to Neo4j.
          6. Emit BENCHMARKS_EVOLUTIONARY_ACTIVITY to Evo and Nexus.
          7. Cache divergence on the last snapshot so it appears in the next persist cycle.
        """
        data = getattr(event, "data", {}) or {}

        # ── Extract drive-weight distribution ────────────────────────────────
        # Payload carries drive_weight_distribution as a dict of
        # {coherence, care, growth, honesty} → {mean, variance, min, max}
        # and constitutional_phenotype_clusters as list of
        # {label, centroid, size, dominant_drive}
        dist: dict[str, Any] = data.get("drive_weight_distribution", {})
        clusters: list[dict[str, Any]] = data.get("constitutional_phenotype_clusters", [])
        instance_count: int = data.get("instance_count", 0)
        speciation_signal: float = float(data.get("speciation_signal", 0.0))
        ts: str = data.get("timestamp", utc_now().isoformat())

        drives = ["coherence", "care", "growth", "honesty"]

        # ── Build per-cluster drive-config fingerprints ──────────────────────
        # Each cluster centroid represents a distinct constitutional phenotype.
        # We fingerprint as a tuple of rounded centroid values.
        current_fingerprints: set[str] = set()
        for cluster in clusters:
            centroid = cluster.get("centroid", {})
            key = "|".join(
                f"{drive}:{round(float(centroid.get(drive, 0.0)), 2):.2f}"
                for drive in drives
            )
            current_fingerprints.add(key)

        # adaptive_activity_A: novel configs that are ALSO still present
        novel_this_period = current_fingerprints - self._prev_drive_config_fingerprints
        adaptive_activity_A = len(novel_this_period & current_fingerprints)
        self._prev_drive_config_fingerprints = current_fingerprints

        # ── Compute constitutional_phenotype_divergence ──────────────────────
        # Mean of per-drive variance across the fleet (from distribution payload).
        # When dist carries per-drive variance fields, use them directly.
        # Fallback: compute from cluster centroids if variance not in payload.
        divergence: float | None = None
        drive_variances: list[float] = []
        for drive in drives:
            drive_data = dist.get(drive, {})
            if isinstance(drive_data, dict):
                var = drive_data.get("variance")
                if var is not None:
                    drive_variances.append(float(var))

        if not drive_variances and len(clusters) >= 2:
            # Fall back: compute variance of centroid values across clusters
            for drive in drives:
                vals = [
                    float(c.get("centroid", {}).get(drive, 0.0))
                    for c in clusters
                ]
                if len(vals) >= 2:
                    mean_v = sum(vals) / len(vals)
                    var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                    drive_variances.append(var_v)

        if drive_variances:
            divergence = round(sum(drive_variances) / len(drive_variances), 6)

        # Cache divergence so the next _collect() can include it in the snapshot
        self._last_phenotype_divergence: float | None = divergence

        # ── Persist (:BedauPackardSample) node to Neo4j ──────────────────────
        node_id = f"bps:{self._instance_id}:{ts}"
        if self._memory is not None:
            try:
                neo4j = getattr(self._memory, "_neo4j", None)
                if neo4j is not None:
                    await neo4j.execute_write(
                        """
                        MERGE (bps:BedauPackardSample {node_id: $node_id})
                        SET bps.instance_id = $instance_id,
                            bps.timestamp = $ts,
                            bps.adaptive_activity_A = $activity_a,
                            bps.constitutional_phenotype_divergence = $divergence,
                            bps.speciation_signal = $speciation_signal,
                            bps.instance_count = $instance_count,
                            bps.cluster_count = $cluster_count
                        """,
                        {
                            "node_id": node_id,
                            "instance_id": self._instance_id,
                            "ts": ts,
                            "activity_a": adaptive_activity_A,
                            "divergence": divergence,
                            "speciation_signal": speciation_signal,
                            "instance_count": instance_count,
                            "cluster_count": len(clusters),
                        },
                    )
            except Exception as exc:
                self._logger.debug("bedau_packard_neo4j_persist_failed", error=str(exc))

        # ── Emit BENCHMARKS_EVOLUTIONARY_ACTIVITY ────────────────────────────
        if self._event_bus is not None:
            event_out = SynapseEvent(
                event_type=SynapseEventType.BENCHMARKS_EVOLUTIONARY_ACTIVITY,
                source_system=self.system_id,
                data={
                    "instance_id": self._instance_id,
                    "timestamp": ts,
                    "adaptive_activity_A": adaptive_activity_A,
                    "constitutional_phenotype_divergence": divergence,
                    "speciation_signal": speciation_signal,
                    "instance_count": instance_count,
                    "bedau_packard_node_id": node_id,
                },
            )
            await self._event_bus.emit(event_out)

        self._logger.info(
            "bedau_packard_intelligence_ratio_sampled",
            adaptive_activity_A=adaptive_activity_A,
            constitutional_phenotype_divergence=divergence,
            speciation_signal=speciation_signal,
            instance_count=instance_count,
            cluster_count=len(clusters),
        )

    # ─── Lifecycle ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Ensure the benchmarks table exists and start the collection loop."""
        await self._ensure_schema()
        await self._evo_tracker.restore_from_redis()
        await self._restore_regressed_from_redis()

        # Five-pillar evaluation protocol
        self._evaluation_protocol = EvaluationProtocol(instance_id=self._instance_id)
        self._test_set_manager = TestSetManager()

        # Fixed test sets for direct pillar evaluation (pillars.py)
        self._test_sets = load_fixed_test_sets()

        self._initialized = True
        self._task = asyncio.create_task(self._run_loop(), name="benchmarks_loop")
        self._monthly_eval_task = asyncio.create_task(
            self._monthly_eval_loop(), name="benchmarks_monthly_eval"
        )
        self._logger.info(
            "benchmarks_started",
            interval_s=self._config.interval_s,
            rolling_window=self._config.rolling_window_snapshots,
            regression_threshold_pct=self._config.regression_threshold_pct,
        )

    async def shutdown(self) -> None:
        """Cancel the background loop gracefully."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._monthly_eval_task:
            self._monthly_eval_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monthly_eval_task
        self._logger.info("benchmarks_stopped")

    # ─── Collection loop ──────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Collect immediately on startup, then every interval_s. Forever."""
        # Small startup delay so all systems are fully ready
        await asyncio.sleep(10.0)
        first_run = True
        while True:
            try:
                if not first_run:
                    await asyncio.sleep(self._config.interval_s)
                first_run = False
                snapshot = await self._collect()
                bedau = await self._evo_tracker.snapshot()
                snapshot.bedau_packard = bedau
                self._last_snapshot = snapshot
                await self._persist(snapshot)
                await self._check_regressions(snapshot)
                await self._check_re_progress(snapshot)
                await self._check_sustained_llm_dependency()
                await self._tag_episodes_for_re_training(snapshot)
                await self._persist_regressed_to_redis()
                # Domain KPI signals - emit threshold events + Neo4j persistence
                await self._emit_domain_signals(snapshot.domain_kpis)
                await self._persist_domain_kpis_neo4j(snapshot.domain_kpis)
                # Domain pivot detection - notify Thread when primary domain changes
                if snapshot.primary_domain != self._prev_primary_domain:
                    self._logger.info(
                        "primary_domain_changed",
                        prev=self._prev_primary_domain,
                        new=snapshot.primary_domain,
                    )
                    if self._event_bus is not None and hasattr(SynapseEventType, "NOVA_GOAL_INJECTED"):
                        try:
                            await self._event_bus.emit(SynapseEvent(
                                event_type=SynapseEventType.NOVA_GOAL_INJECTED,
                                source_system=self.system_id,
                                data={
                                    "goal_description": (
                                        f"Primary specialization domain shifted to "
                                        f"'{snapshot.primary_domain}' - "
                                        f"adjust goal priorities accordingly"
                                    ),
                                    "priority": 0.6,
                                    "source": "benchmarks_domain_pivot",
                                    "domain": snapshot.primary_domain,
                                    "prev_domain": self._prev_primary_domain,
                                },
                            ))
                        except Exception:
                            pass
                    self._prev_primary_domain = snapshot.primary_domain
                self._total_runs += 1
                self._logger.info(
                    "benchmark_run_completed",
                    run=self._total_runs,
                    decision_quality=snapshot.decision_quality,
                    llm_dependency=snapshot.llm_dependency,
                    economic_ratio=snapshot.economic_ratio,
                    learning_rate=snapshot.learning_rate,
                    mutation_success_rate=snapshot.mutation_success_rate,
                    effective_intelligence_ratio=snapshot.effective_intelligence_ratio,
                    compression_ratio=snapshot.compression_ratio,
                    primary_domain=snapshot.primary_domain,
                    active_domains=len(snapshot.domain_kpis),
                    errors=list(snapshot.errors.keys()),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.error("benchmark_loop_error", error=str(exc))

    # ─── On-demand evaluation (for ablation studies + manual testing) ────

    async def run_evaluation_now(self, month: int | None = None) -> Any:
        """Run the full 5-pillar evaluation pipeline immediately.

        Used by AblationOrchestrator and manual CLI invocations.
        Does NOT increment self._current_month - this is a read-only
        evaluation pass that does not advance the monotonic counter.

        Returns a LongitudinalSnapshot on success; raises on failure.
        """
        if self._evaluation_protocol is None or self._test_set_manager is None:
            raise RuntimeError("BenchmarkService not fully initialized - call initialize() first")

        eval_month = month if month is not None else self._current_month
        test_sets = await self._test_set_manager.load_all()
        result = await self._evaluation_protocol.run_monthly_evaluation(
            test_sets=test_sets,
            month=eval_month,
        )

        # Build LongitudinalSnapshot without persisting (non-destructive evaluation pass).
        # We pass re_performance but do not write to Neo4j - that only happens in the
        # scheduled monthly loop which increments _current_month.
        snap = await self._longitudinal.record_month(
            month=eval_month,
            eval_results=result.to_dict(),
            re_performance=self._re_performance,
            adapter_path=getattr(
                getattr(self._evaluation_protocol, "_re", None),
                "_current_adapter_path",
                "",
            ) or "",
        )
        return snap

    # ─── Monthly evaluation loop ──────────────────────────────────────

    async def _monthly_eval_loop(self) -> None:
        """Run 5-pillar evaluation on the 1st of each month at 03:00 UTC."""
        # Small startup delay so all systems are fully ready
        await asyncio.sleep(15.0)
        first_run = True
        while True:
            try:
                if not first_run:
                    # Calculate seconds until next 1st-of-month at 03:00 UTC
                    now = datetime.now(tz=timezone.utc)
                    if now.month == 12:
                        next_run = datetime(now.year + 1, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
                    else:
                        next_run = datetime(now.year, now.month + 1, 1, 3, 0, 0, tzinfo=timezone.utc)
                    sleep_s = (next_run - now).total_seconds()
                    self._logger.info(
                        "monthly_eval_scheduled",
                        next_run_utc=next_run.isoformat(),
                        sleep_hours=round(sleep_s / 3600, 1),
                    )
                    await asyncio.sleep(max(sleep_s, 0.0))

                first_run = False

                if self._evaluation_protocol is None or self._test_set_manager is None:
                    continue

                test_sets = await self._test_set_manager.load_all()
                now = datetime.now(tz=timezone.utc)
                result = await self._evaluation_protocol.run_monthly_evaluation(
                    test_sets=test_sets,
                    month=now.month,
                )

                p1 = result.pillar1_specialization
                p2 = result.pillar2_novelty
                p3 = result.pillar3_causal
                p4 = result.pillar4_velocity
                p5 = result.pillar5_ethical

                self._logger.info(
                    "monthly_evaluation_complete",
                    month=now.month,
                    year=now.year,
                    specialization_index=p1.specialization_index if p1 else None,
                    causal_rung2_accuracy=p3.l2_intervention if p3 else None,
                    causal_rung3_accuracy=p3.l3_counterfactual if p3 else None,
                    learning_velocity=p4.velocity if p4 else None,
                    ethical_drift_magnitude=p5.drift_magnitude if p5 else None,
                    errors=result.errors,
                )

                # Augment result dict with RE performance metrics
                result_dict = result.to_dict()
                result_dict["re_performance"] = self._re_performance

                # ── Bedau-Packard fleet-level evolutionary activity ────────
                # Requires fleet genome snapshots (populated from CHILD_SPAWNED events).
                # Degrades gracefully: empty fleet → zero metrics, no error.
                self._monthly_eval_count += 1
                try:
                    fleet_genomes = await self._collect_fleet_genomes()
                    if fleet_genomes:
                        bp_components = self._bp_tracker.ingest_fleet_genomes(fleet_genomes)
                        bp_snap = self._bp_tracker.compute_adaptive_activity(
                            bp_components, month=now.month
                        )
                        ea_payload: dict[str, Any] = {
                            "month": bp_snap.month,
                            "adaptive_activity": bp_snap.adaptive_activity,
                            "novelty_rate": bp_snap.novelty_rate,
                            "diversity": bp_snap.diversity,
                            "exceeds_shadow": bp_snap.exceeds_shadow,
                            "population_size": bp_snap.population_size,
                            "component_count": bp_snap.component_count,
                            "novel_component_count": bp_snap.novel_component_count,
                            "oee_verdict": None,
                        }
                        # OEE assessment available after ≥3 months of data
                        if self._monthly_eval_count >= 3:
                            ea_payload["oee_verdict"] = self._bp_tracker.assess_oee_evidence().get(
                                "verdict"
                            )
                            result_dict["oee_assessment"] = self._bp_tracker.assess_oee_evidence()

                        result_dict["evolutionary_activity"] = ea_payload

                        # Emit dedicated event for Evo + Nexus + Alive consumers
                        if self._event_bus is not None:
                            try:
                                await self._event_bus.emit(SynapseEvent(
                                    event_type=SynapseEventType.EVOLUTIONARY_ACTIVITY_COMPUTED,
                                    source_system=SystemID.BENCHMARKS
                                    if hasattr(SystemID, "BENCHMARKS")
                                    else "benchmarks",
                                    data=ea_payload,
                                ))
                            except Exception as _ea_exc:
                                self._logger.debug(
                                    "evolutionary_activity_emit_failed", error=str(_ea_exc)
                                )
                except Exception as _bp_exc:
                    self._logger.warning(
                        "bedau_packard_monthly_failed", error=str(_bp_exc)
                    )

                # ── Population Divergence ─────────────────────────────
                # Only meaningful with ≥ 2 live instances.
                # Primary: real per-drive ethical drift records from Neo4j for
                # each fleet instance at the current month.
                # Fallback: genome structural distance proxy when < 2 ethical
                # drift records are available (early months / no RE service).
                # Non-fatal: any failure is logged and monthly eval continues.
                if len(self._fleet_genomes) >= 2:
                    try:
                        import json as _json
                        from systems.benchmarks.ethical_drift import (
                            EthicalDriftTracker as _EDT,
                            MonthlyDriftRecord as _MDR,
                        )

                        # ── Attempt real ethical drift divergence ─────────
                        _neo4j = (
                            getattr(self._memory, "_neo4j", None)
                            if self._memory
                            else None
                        )
                        _fleet_drift_records: list[_MDR] = []
                        if _neo4j is not None:
                            for _iid in self._fleet_genomes:
                                try:
                                    _rows = await _neo4j.execute_read(
                                        "MATCH (r:EthicalDriftRecord "
                                        "{instance_id: $id, month: $month}) "
                                        "RETURN r.drive_means_json AS drive_means_json "
                                        "LIMIT 1",
                                        id=_iid,
                                        month=self._current_month,
                                    )
                                    if _rows and _rows[0].get("drive_means_json"):
                                        _drive_means = _json.loads(
                                            _rows[0]["drive_means_json"]
                                        )
                                        _fleet_drift_records.append(
                                            _MDR(
                                                month=self._current_month,
                                                instance_id=_iid,
                                                drive_means=_drive_means,
                                            )
                                        )
                                except Exception:
                                    pass  # Non-fatal per-instance failure

                        if len(_fleet_drift_records) >= 2:
                            # Use real per-drive ethical divergence
                            _pop_div = _EDT.compute_population_divergence(
                                _fleet_drift_records
                            )
                            _pop_div["divergence_source"] = "ethical_drift"
                            _pop_div["population_size"] = len(_fleet_drift_records)
                            result_dict["population_divergence"] = _pop_div
                            self._logger.info(
                                "population_divergence_computed",
                                source="ethical_drift",
                                divergence=_pop_div.get("divergence"),
                                pairs=_pop_div.get("pairs_compared"),
                                speciation_signal=_pop_div.get("is_speciation_signal"),
                            )
                        else:
                            # Fall back to genome structural distance proxy
                            from systems.mitosis.genome_distance import (
                                GenomeDistanceCalculator,
                            )

                            _spec_threshold = float(
                                getattr(
                                    getattr(self._config, "mitosis", None),
                                    "speciation_distance_threshold",
                                    0.3,
                                )
                            )
                            _calc = GenomeDistanceCalculator(
                                speciation_threshold=_spec_threshold
                            )
                            _genomes = list(self._fleet_genomes.values())
                            _distances: list[float] = []
                            for _i, _ga in enumerate(_genomes):
                                for _gb in _genomes[_i + 1 :]:
                                    _d = _calc.compute(_ga, _gb)
                                    _distances.append(_d.total_distance)
                            if _distances:
                                _mean_d = sum(_distances) / len(_distances)
                                _max_d = max(_distances)
                                _speciation_detected = any(
                                    d > _spec_threshold for d in _distances
                                )
                                result_dict["population_divergence"] = {
                                    "mean_genome_distance": round(_mean_d, 6),
                                    "max_genome_distance": round(_max_d, 6),
                                    "pairs_compared": len(_distances),
                                    "population_size": len(_genomes),
                                    "speciation_detected": _speciation_detected,
                                    "speciation_threshold": _spec_threshold,
                                    "divergence_source": "genome_distance_proxy",
                                }
                                self._logger.info(
                                    "population_divergence_computed",
                                    source="genome_distance_proxy",
                                    mean_distance=_mean_d,
                                    max_distance=_max_d,
                                    pairs=len(_distances),
                                    speciation_detected=_speciation_detected,
                                )
                    except Exception as _pop_exc:
                        self._logger.warning(
                            "population_divergence_failed", error=str(_pop_exc)
                        )

                # ── Pillar 5: Ethical Drift Map ───────────────────────────
                # Runs 100 frozen catch-22 scenarios through the RE and
                # computes per-drive activation + drift vector vs Month 1.
                # Non-fatal: any failure is logged and monthly eval continues.
                try:
                    re_svc = (
                        getattr(self._evaluation_protocol, "_re", None)
                        if self._evaluation_protocol is not None
                        else None
                    )
                    if re_svc is not None:
                        drift_record = await self._ethical_drift.evaluate(
                            re_svc,
                            month=self._current_month,
                            instance_id=self._instance_id,
                        )
                        drift_record = await self._drift_tracker.record_month(drift_record)
                        dominant_drive = (
                            max(
                                drift_record.dominant_drive_distribution,
                                key=lambda k: drift_record.dominant_drive_distribution[k],
                            )
                            if drift_record.dominant_drive_distribution
                            else "unknown"
                        )
                        result_dict["ethical_drift"] = {
                            "drift_magnitude": drift_record.drift_magnitude,
                            "drift_vector": drift_record.drift_vector,
                            "drive_means": drift_record.drive_means,
                            "dominant_drive": dominant_drive,
                            "dominant_drive_distribution": drift_record.dominant_drive_distribution,
                            "n_scenarios": len(drift_record.scenario_results),
                        }
                    else:
                        result_dict.setdefault("ethical_drift", {
                            "drift_magnitude": 0.0,
                            "drift_vector": {},
                            "drive_means": {},
                            "dominant_drive": "unknown",
                            "note": "re_service_unavailable",
                        })
                except Exception as _drift_exc:
                    self._logger.warning(
                        "ethical_drift_monthly_failed", error=str(_drift_exc)
                    )
                    result_dict.setdefault("ethical_drift", {})

                # ── Longitudinal tracking (§6.4) ──────────────────────────
                # Record this month's snapshot and compare to Month 1 baseline.
                # Returns {"no_baseline": True} on first month - graceful.
                try:
                    long_snap = await self._longitudinal.record_month(
                        month=self._current_month,
                        eval_results=result_dict,
                        re_performance=self._re_performance,
                        adapter_path=getattr(
                            getattr(self._evaluation_protocol, "_re", None),
                            "_current_adapter_path",
                            "",
                        ),
                    )
                    comparison = await self._longitudinal.compare_to_baseline(long_snap)
                    result_dict["longitudinal_comparison"] = comparison
                    self._logger.info(
                        "longitudinal_comparison",
                        month=self._current_month,
                        verdict=comparison.get("verdict", "no_baseline"),
                        l2_delta=comparison.get("causal", {}).get("l2_intervention_delta"),
                    )
                except Exception as _long_exc:
                    self._logger.warning(
                        "longitudinal_monthly_failed", error=str(_long_exc)
                    )
                    result_dict.setdefault("longitudinal_comparison", {"no_baseline": True})

                # ── Pillars 1–4 + §6.3 Memorization (bible §6.2–6.3) ─────────
                # Direct evaluation via pillars.py - additive alongside EvaluationProtocol.
                # All pillar calls are non-fatal: any exception logged, monthly eval continues.
                if self._reasoning_engine and self._test_sets:
                    try:
                        # Pillar 1: Specialization Index
                        if self._test_sets.get("domain_test") and self._test_sets.get("general_test"):
                            spec = await measure_specialization(
                                self._reasoning_engine,
                                self._reasoning_engine,  # custom vs base (same - Claude fallback exposed internally)
                                self._test_sets["domain_test"],
                                self._test_sets["general_test"],
                            )
                            result_dict["pillar1_specialization_index"] = spec.specialization_index
                            result_dict["pillar1_domain_improvement"] = spec.domain_improvement
                            result_dict["pillar1_general_retention"] = spec.general_retention
                            self._logger.info(
                                "pillar1_complete",
                                si=spec.specialization_index,
                                domain_improvement=spec.domain_improvement,
                            )

                        # Pre-compute training embeddings for Pillar 2 cosine distance.
                        if (
                            hasattr(self._reasoning_engine, "encode")
                            and not getattr(self, "_cached_training_embeddings", None)
                        ):
                            try:
                                await self._build_training_embeddings_cache()
                            except Exception as _emb_exc:
                                self._logger.debug("training_embeddings_cache_failed", error=str(_emb_exc))

                        # Pillar 2: Novelty Emergence
                        if self._test_sets.get("novel_episodes"):
                            # Provide encode_fn so pillar can embed novel reasoning texts
                            # for real cosine distance against training distribution.
                            _encode_fn = None
                            if hasattr(self._reasoning_engine, "encode"):
                                _encode_fn = self._reasoning_engine.encode
                            _train_embeds = getattr(self, "_cached_training_embeddings", None)
                            novelty = await measure_novelty_emergence(
                                self._reasoning_engine,
                                self._test_sets["novel_episodes"],
                                training_embeddings=_train_embeds,
                                encode_fn=_encode_fn,
                            )
                            result_dict["pillar2_novel_success_rate"] = novelty.novel_success_rate
                            result_dict["pillar2_cosine_distance"] = novelty.reasoning_cosine_distance
                            result_dict["pillar2_genuine_learning"] = novelty.genuine_learning
                            self._logger.info(
                                "pillar2_complete",
                                success_rate=novelty.novel_success_rate,
                                genuine_learning=novelty.genuine_learning,
                            )

                        # Pillar 3: Causal Reasoning (CLadder + CCR.GB)
                        if (
                            self._test_sets.get("cladder_questions")
                            and self._test_sets.get("ccr_gb_scenarios")
                        ):
                            causal = await measure_causal_reasoning(
                                self._reasoning_engine,
                                self._test_sets["cladder_questions"],
                                self._test_sets["ccr_gb_scenarios"],
                            )
                            result_dict["pillar3_l2_intervention"] = causal.l2_intervention
                            result_dict["pillar3_l3_counterfactual"] = causal.l3_counterfactual
                            result_dict["pillar3_ccr_validity"] = causal.ccr_validity
                            # Feed combined L2+L3 score into velocity tracker
                            l2l3_combined = (causal.l2_intervention + causal.l3_counterfactual) / 2
                            self._causal_history.append({
                                "month": self._current_month,
                                "score": l2l3_combined,
                            })
                            self._logger.info(
                                "pillar3_complete",
                                l2=causal.l2_intervention,
                                l3=causal.l3_counterfactual,
                                ccr_validity=causal.ccr_validity,
                            )

                        # Pillar 4: Learning Velocity (power-law fit on _causal_history)
                        vel = compute_learning_velocity(self._causal_history)
                        result_dict["pillar4_velocity"] = vel.velocity
                        result_dict["pillar4_is_plateaued"] = vel.is_plateaued
                        result_dict["pillar4_predicted_month_12"] = vel.predicted_month_12
                        if vel.is_plateaued and not vel.insufficient_data:
                            self._logger.warning(
                                "plasticity_loss_suspected",
                                velocity=vel.velocity,
                                months_of_data=len(self._causal_history),
                            )

                        # §6.3 Memorization Detection
                        if self._test_sets.get("paraphrase_pairs"):
                            mem = await detect_memorization(
                                self._reasoning_engine,
                                [],  # training_sample: populated when RE exporter feeds training log
                                self._test_sets.get("novel_episodes", [])[:50],  # holdout proxy
                                self._test_sets["paraphrase_pairs"],
                            )
                            result_dict["memorization_risk"] = mem.memorization_risk
                            result_dict["memorization_mi_accuracy"] = mem.membership_inference_accuracy
                            result_dict["memorization_paraphrase_drop"] = mem.paraphrase_accuracy_drop
                            if mem.memorization_risk == "high":
                                self._logger.error(
                                    "memorization_detected",
                                    recommendation=mem.recommendation,
                                    mi_accuracy=mem.membership_inference_accuracy,
                                )
                    except Exception as _pillar_exc:
                        self._logger.error("pillars_eval_failed", error=str(_pillar_exc))

                # Advance monotonic month counter after all pillars complete
                self._current_month += 1

                if self._event_bus is not None:
                    try:
                        event = SynapseEvent(
                            event_type=SynapseEventType.MONTHLY_EVALUATION_COMPLETE,
                            source_system=SystemID.BENCHMARKS
                            if hasattr(SystemID, "BENCHMARKS")
                            else "benchmarks",
                            data=result_dict,
                        )
                        await self._event_bus.emit(event)
                    except Exception as exc:
                        self._logger.warning("monthly_eval_event_emit_failed", error=str(exc))

                # Persist to Neo4j for longitudinal RE performance tracking (fire-and-forget)
                if self._memory is not None:
                    asyncio.create_task(
                        self._persist_monthly_eval_neo4j(now.month, now.year, result_dict),
                        name="benchmarks_monthly_eval_neo4j",
                    )

                # Export paper data CSVs (fire-and-forget, non-fatal)
                asyncio.ensure_future(
                    self._paper_exporter.export_all(month=self._current_month - 1)
                )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.error("monthly_eval_loop_error", error=str(exc))

    # ─── Monthly eval Neo4j persistence ──────────────────────────────

    async def _persist_monthly_eval_neo4j(
        self,
        month: int,
        year: int,
        result_dict: dict[str, Any],
    ) -> None:
        """Fire-and-forget: write (:MonthlyEvaluation) node to Neo4j.

        Never blocks the monthly eval loop - any exception is swallowed.
        """
        try:
            neo4j = getattr(self._memory, "_neo4j", None)
            if neo4j is None:
                return

            re_perf = result_dict.get("re_performance") or {}
            p1 = result_dict.get("pillar1_specialization") or {}
            p4 = result_dict.get("pillar4_velocity") or {}

            await neo4j.execute_write(
                """
                CREATE (e:MonthlyEvaluation {
                    node_id: $node_id,
                    month: $month,
                    year: $year,
                    timestamp: datetime(),
                    specialization_index: $si,
                    learning_velocity: $lv,
                    re_success_rate: $re_rate,
                    re_usage_pct: $re_usage,
                    re_total_decisions: $re_total,
                    instance_id: $instance_id
                })
                RETURN e
                """,
                node_id=f"monthly_eval:{self._instance_id}:{year}-{month:02d}",
                month=month,
                year=year,
                si=float(p1.get("specialization_index") or 0.0),
                lv=float(p4.get("velocity") or 0.0),
                re_rate=float(re_perf.get("success_rate") or 0.0),
                re_usage=float(re_perf.get("usage_pct") or 0.0),
                re_total=int(re_perf.get("total_decisions") or 0),
                instance_id=self._instance_id,
            )
            self._logger.info(
                "monthly_eval_neo4j_persisted",
                month=month,
                year=year,
                re_rate=re_perf.get("success_rate"),
            )
        except Exception as exc:
            self._logger.warning("monthly_eval_neo4j_persist_failed", error=str(exc))

    # ─── KPI collection ──────────────────────────────────────────────

    async def _collect(self) -> BenchmarkSnapshot:
        """Collect all seven KPIs concurrently."""
        results = await asyncio.gather(
            self._collect_decision_quality(),
            self._collect_llm_dependency(),
            self._collect_economic_ratio(),
            self._collect_learning_rate(),
            self._collect_mutation_success_rate(),
            self._collect_effective_intelligence_ratio(),
            self._collect_compression_ratio(),
            return_exceptions=True,
        )

        errors: dict[str, str] = {}
        raw: dict[str, Any] = {}

        def _extract(idx: int, name: str) -> float | None:
            r = results[idx]
            if isinstance(r, Exception):
                errors[name] = str(r)
                return None
            value, raw_data = r  # type: ignore[misc]
            raw[name] = raw_data
            return value

        # Loop 6: Snapshot and reset accumulated fitness observables
        fitness = dict(self._accumulated_fitness)
        self._accumulated_fitness = {
            "hypotheses_evaluated": 0,
            "hypotheses_integrated": 0,
            "schemas_induced": 0,
            "consolidation_count": 0,
        }

        # Domain KPIs - computed from accumulated EpisodeRecord history
        domain_data = await self._collect_domain_kpis()
        domain_kpis = domain_data["domain_kpis"]
        primary_domain = domain_data["primary_domain"]

        return BenchmarkSnapshot(
            time=utc_now(),
            instance_id=self._instance_id,
            decision_quality=_extract(0, "decision_quality"),
            llm_dependency=_extract(1, "llm_dependency"),
            economic_ratio=_extract(2, "economic_ratio"),
            learning_rate=_extract(3, "learning_rate"),
            mutation_success_rate=_extract(4, "mutation_success_rate"),
            effective_intelligence_ratio=_extract(5, "effective_intelligence_ratio"),
            compression_ratio=_extract(6, "compression_ratio"),
            evolutionary_fitness=fitness,
            constitutional_phenotype_divergence=self._last_phenotype_divergence,
            domain_kpis=domain_kpis,
            primary_domain=primary_domain,
            errors=errors,
            raw=raw,
        )

    async def _collect_decision_quality(self) -> tuple[float | None, dict[str, Any]]:
        """
        % of Nova outcomes rated positive.

        Nova.health() exposes:
          outcomes_success: int
          outcomes_failure: int
        """
        if self._nova is None:
            return None, {}
        health = await self._nova.health()
        success = int(health.get("outcomes_success", 0))
        failure = int(health.get("outcomes_failure", 0))
        total = success + failure
        raw = {"outcomes_success": success, "outcomes_failure": failure}
        if total == 0:
            return None, raw
        return round(success / total, 4), raw

    async def _collect_llm_dependency(self) -> tuple[float | None, dict[str, Any]]:
        """
        % of decisions that required an LLM call (slow_path / total).

        Nova.health() exposes:
          fast_path_decisions: int
          slow_path_decisions: int
          do_nothing_decisions: int
        """
        if self._nova is None:
            return None, {}
        health = await self._nova.health()
        fast = int(health.get("fast_path_decisions", 0))
        slow = int(health.get("slow_path_decisions", 0))
        do_nothing = int(health.get("do_nothing_decisions", 0))
        total = fast + slow + do_nothing
        raw = {"fast": fast, "slow": slow, "do_nothing": do_nothing, "total": total}
        if total == 0:
            return None, raw
        return round(slow / total, 4), raw

    async def _collect_economic_ratio(self) -> tuple[float | None, dict[str, Any]]:
        """
        Oikos income / expenses = revenue_7d / costs_7d.

        Oikos.stats exposes revenue_7d and costs_7d as Decimal strings.
        """
        if self._oikos is None:
            return None, {}
        stats = self._oikos.stats  # sync property
        rev_raw = stats.get("revenue_7d", "0")
        cost_raw = stats.get("costs_7d", "0")
        raw = {"revenue_7d": rev_raw, "costs_7d": cost_raw}
        try:
            revenue = Decimal(str(rev_raw))
            costs = Decimal(str(cost_raw))
        except InvalidOperation:
            raise ValueError(f"Oikos returned non-numeric values: {raw}")
        if costs == Decimal("0"):
            return None, raw  # No expenses yet - ratio undefined
        return float(round(revenue / costs, 4)), raw

    async def _collect_learning_rate(self) -> tuple[float | None, dict[str, Any]]:
        """
        Number of hypotheses newly confirmed (supported) in this window.

        Evo.stats exposes hypothesis.supported (cumulative).
        We compare to the value stored in the previous snapshot.

        Evo restart detection: if supported_total < prev baseline, Evo was restarted
        and its counter reset. We treat the new value as the new baseline (delta = 0)
        rather than reporting a large negative or inflated count.
        Spec ref: §26.2 "learning_rate delta can be negative".
        """
        if self._evo is None:
            return None, {}
        stats = self._evo.stats
        hyp = stats.get("hypothesis", {})
        supported_total = int(hyp.get("supported", 0))
        raw = {"supported_total": supported_total}

        # Read previous cumulative value from TimescaleDB
        prev = await self._latest_raw_value("learning_rate_cumulative")
        if prev is None:
            # First run - store baseline, return 0 (no confirmed in this window)
            await self._store_auxiliary("learning_rate_cumulative", float(supported_total))
            return 0.0, raw

        prev_int = int(prev)
        if supported_total < prev_int:
            # Evo was restarted - counter reset. Re-baseline without penalising.
            await self._store_auxiliary("learning_rate_cumulative", float(supported_total))
            raw["evo_restart_detected"] = True
            raw["delta"] = 0
            self._logger.warning(
                "learning_rate_evo_restart_detected",
                prev_baseline=prev_int,
                new_count=supported_total,
            )
            return 0.0, raw

        delta = supported_total - prev_int
        await self._store_auxiliary("learning_rate_cumulative", float(supported_total))
        raw["delta"] = delta
        return float(delta), raw

    async def _collect_mutation_success_rate(self) -> tuple[float | None, dict[str, Any]]:
        """
        Simula proposals_approved / proposals_received.

        Simula.stats exposes proposals_approved and proposals_received.
        """
        if self._simula is None:
            return None, {}
        stats = self._simula.stats
        approved = int(stats.get("proposals_approved", 0))
        received = int(stats.get("proposals_received", 0))
        raw = {"proposals_approved": approved, "proposals_received": received}
        if received == 0:
            return None, raw
        return round(approved / received, 4), raw

    async def _collect_effective_intelligence_ratio(self) -> tuple[float | None, dict[str, Any]]:
        """
        Telos effective_I - nominal_I scaled by all four drive multipliers.

        Telos.health() exposes last_effective_I directly.
        """
        if self._telos is None:
            return None, {}
        health = await self._telos.health()
        value = health.get("last_effective_I")
        raw: dict[str, Any] = {
            "last_effective_I": value,
            "alignment_gap": health.get("last_alignment_gap"),
        }
        if value is None:
            return None, raw
        return round(float(value), 4), raw

    async def _collect_compression_ratio(self) -> tuple[float | None, dict[str, Any]]:
        """
        Logos intelligence ratio I = K(reality_modeled) / K(model).

        Logos.health() exposes intelligence_ratio directly.
        """
        if self._logos is None:
            return None, {}
        health = await self._logos.health()
        value = health.get("intelligence_ratio")
        raw: dict[str, Any] = {
            "intelligence_ratio": value,
            "cognitive_pressure": health.get("cognitive_pressure"),
            "schwarzschild_met": health.get("schwarzschild_met"),
        }
        if value is None:
            return None, raw
        return round(float(value), 4), raw

    # ─── Persistence ──────────────────────────────────────────────────

    async def _persist(self, snapshot: BenchmarkSnapshot) -> None:
        """Write snapshot to TimescaleDB benchmark_snapshots table.

        Includes bedau_packard (JSONB) and evolutionary_fitness (JSONB) columns
        added in §4.2 schema (Appendix A). Spec ref: §4.2, Appendix A.
        """
        async with self._tsdb.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO benchmark_snapshots (
                    time, instance_id,
                    decision_quality, llm_dependency, economic_ratio,
                    learning_rate, mutation_success_rate,
                    effective_intelligence_ratio, compression_ratio,
                    bedau_packard, evolutionary_fitness,
                    constitutional_phenotype_divergence,
                    errors, raw
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                """,
                snapshot.time,
                snapshot.instance_id,
                snapshot.decision_quality,
                snapshot.llm_dependency,
                snapshot.economic_ratio,
                snapshot.learning_rate,
                snapshot.mutation_success_rate,
                snapshot.effective_intelligence_ratio,
                snapshot.compression_ratio,
                json.dumps(
                    snapshot.bedau_packard.model_dump(mode="json")
                    if snapshot.bedau_packard else None
                ),
                json.dumps(snapshot.evolutionary_fitness),
                snapshot.constitutional_phenotype_divergence,
                json.dumps(snapshot.errors),
                json.dumps(snapshot.raw),
            )

    async def _ensure_schema(self) -> None:
        """Create benchmark_snapshots table (and hypertable if TimescaleDB available).

        Spec ref: Appendix A - includes bedau_packard and evolutionary_fitness columns.
        Uses ALTER TABLE ADD COLUMN IF NOT EXISTS to handle existing tables that
        pre-date these columns.
        """
        async with self._tsdb.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_snapshots (
                    time                          TIMESTAMPTZ NOT NULL,
                    instance_id                   TEXT NOT NULL,
                    decision_quality              DOUBLE PRECISION,
                    llm_dependency                DOUBLE PRECISION,
                    economic_ratio                DOUBLE PRECISION,
                    learning_rate                 DOUBLE PRECISION,
                    mutation_success_rate         DOUBLE PRECISION,
                    effective_intelligence_ratio  DOUBLE PRECISION,
                    compression_ratio             DOUBLE PRECISION,
                    bedau_packard                 JSONB DEFAULT 'null',
                    evolutionary_fitness          JSONB DEFAULT '{}',
                    constitutional_phenotype_divergence DOUBLE PRECISION,
                    errors                        JSONB DEFAULT '{}',
                    raw                           JSONB DEFAULT '{}'
                )
            """)
            # Idempotent migration for existing tables missing these columns
            with contextlib.suppress(Exception):
                await conn.execute(
                    "ALTER TABLE benchmark_snapshots "
                    "ADD COLUMN IF NOT EXISTS bedau_packard JSONB DEFAULT 'null'"
                )
            with contextlib.suppress(Exception):
                await conn.execute(
                    "ALTER TABLE benchmark_snapshots "
                    "ADD COLUMN IF NOT EXISTS evolutionary_fitness JSONB DEFAULT '{}'"
                )
            with contextlib.suppress(Exception):
                await conn.execute(
                    "ALTER TABLE benchmark_snapshots "
                    "ADD COLUMN IF NOT EXISTS constitutional_phenotype_divergence DOUBLE PRECISION"
                )
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bm_instance_time
                    ON benchmark_snapshots (instance_id, time DESC)
            """)
            # Auxiliary table for cross-run state (e.g. cumulative learning_rate baseline)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_aux (
                    key        TEXT PRIMARY KEY,
                    value      DOUBLE PRECISION NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            # Promote to hypertable if TimescaleDB extension is available
            has_tsdb = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
            )
            if has_tsdb:
                with contextlib.suppress(Exception):
                    await conn.execute(
                        "SELECT create_hypertable('benchmark_snapshots', 'time', "
                        "if_not_exists => TRUE)"
                    )

    # ─── Rolling average & regression detection ───────────────────────

    async def _rolling_averages(self) -> dict[str, float | None]:
        """
        Compute rolling average of each KPI over the last N snapshots
        stored for this instance.
        """
        n = self._config.rolling_window_snapshots
        averages: dict[str, float | None] = {}
        async with self._tsdb.pool.acquire() as conn:
            for kpi in _KPI_NAMES:
                # Fetch last N non-null values
                rows = await conn.fetch(
                    f"""
                    SELECT {kpi} FROM benchmark_snapshots
                    WHERE instance_id = $1
                      AND {kpi} IS NOT NULL
                    ORDER BY time DESC
                    LIMIT $2
                    """,
                    self._instance_id,
                    n,
                )
                values = [r[kpi] for r in rows]
                averages[kpi] = sum(values) / len(values) if values else None
        return averages

    async def _check_regressions(self, snapshot: BenchmarkSnapshot) -> None:
        """
        For each KPI: if current value is more than threshold% below rolling avg,
        fire a BENCHMARK_REGRESSION Synapse event. Re-arm when it recovers.
        """
        if self._event_bus is None:
            return

        rolling = await self._rolling_averages()
        threshold = self._config.regression_threshold_pct / 100.0

        snapshot_vals: dict[str, float | None] = {
            "decision_quality": snapshot.decision_quality,
            "llm_dependency": snapshot.llm_dependency,
            "economic_ratio": snapshot.economic_ratio,
            "learning_rate": snapshot.learning_rate,
            "mutation_success_rate": snapshot.mutation_success_rate,
            "effective_intelligence_ratio": snapshot.effective_intelligence_ratio,
            "compression_ratio": snapshot.compression_ratio,
        }

        for kpi, current in snapshot_vals.items():
            avg = rolling.get(kpi)

            if current is None or avg is None or avg == 0.0:
                continue

            # For llm_dependency: higher is WORSE (more LLM calls needed),
            # so a regression means the value increased.
            if kpi == "llm_dependency":
                # Regressed if current > avg + threshold * avg
                regression_pct = (current - avg) / avg
                is_regressed = regression_pct > threshold
            else:
                # For all other KPIs: higher is better.
                # Regressed if current < avg - threshold * avg
                regression_pct = (avg - current) / avg
                is_regressed = regression_pct > threshold

            if is_regressed and kpi not in self._regressed:
                self._regressed.add(kpi)
                self._regressed_at[kpi] = time.monotonic()
                self._total_regressions_fired += 1
                regression = MetricRegression(
                    metric=kpi,
                    current_value=current,
                    rolling_avg=avg,
                    regression_pct=round(regression_pct * 100, 2),
                    threshold_pct=self._config.regression_threshold_pct,
                )
                await self._fire_regression_event(regression)
            elif not is_regressed and kpi in self._regressed:
                # Metric has recovered - re-arm and emit recovery event
                regressed_since = self._regressed_at.pop(kpi, 0.0)
                duration = time.monotonic() - regressed_since if regressed_since else 0.0
                self._regressed.discard(kpi)
                await self._fire_recovery_event(kpi, current, avg, duration)
                self._logger.info(
                    "benchmark_metric_recovered",
                    metric=kpi,
                    current=current,
                    rolling_avg=avg,
                    duration_regressed_s=round(duration, 1),
                )

    async def _fire_regression_event(self, regression: MetricRegression) -> None:
        """Emit a BENCHMARK_REGRESSION event via Synapse event bus."""
        if self._event_bus is None:
            return
        event = SynapseEvent(
            event_type=SynapseEventType.BENCHMARK_REGRESSION,
            source_system=self.system_id,
            data={
                "metric": regression.metric,
                "current_value": regression.current_value,
                "rolling_avg": regression.rolling_avg,
                "regression_pct": regression.regression_pct,
                "threshold_pct": regression.threshold_pct,
                "instance_id": self._instance_id,
            },
        )
        await self._event_bus.emit(event)
        self._logger.warning(
            "benchmark_regression_detected",
            metric=regression.metric,
            current=regression.current_value,
            rolling_avg=regression.rolling_avg,
            regression_pct=regression.regression_pct,
        )

    async def _fire_recovery_event(
        self, metric: str, recovered_value: float, rolling_avg: float, duration: float
    ) -> None:
        """Emit a BENCHMARK_RECOVERY event via Synapse event bus."""
        if self._event_bus is None:
            return
        event = SynapseEvent(
            event_type=SynapseEventType.BENCHMARK_RECOVERY,
            source_system=self.system_id,
            data={
                "metric": metric,
                "previous_value": rolling_avg,
                "recovered_value": recovered_value,
                "duration_regressed": round(duration, 1),
                "instance_id": self._instance_id,
            },
        )
        await self._event_bus.emit(event)
        self._logger.info(
            "benchmark_recovery_emitted",
            metric=metric,
            recovered_value=recovered_value,
            duration_regressed_s=round(duration, 1),
        )

    async def _check_re_progress(self, snapshot: BenchmarkSnapshot) -> None:
        """
        Emit BENCHMARK_RE_PROGRESS when llm_dependency improves >5% cycle-over-cycle.
        Debounced: max 1 per cycle (~60s).
        """
        if self._event_bus is None or snapshot.llm_dependency is None:
            self._prev_llm_dependency = snapshot.llm_dependency
            return

        prev = self._prev_llm_dependency
        current = snapshot.llm_dependency
        self._prev_llm_dependency = current

        if prev is None or prev == 0.0:
            return

        # Improvement means llm_dependency decreased (fewer LLM calls)
        improvement_pct = (prev - current) / prev * 100.0
        if improvement_pct > self._re_progress_min_improvement_pct:
            event = SynapseEvent(
                event_type=SynapseEventType.BENCHMARK_RE_PROGRESS,
                source_system=self.system_id,
                data={
                    "current": round(current, 4),
                    "previous": round(prev, 4),
                    "improvement_pct": round(improvement_pct, 2),
                    "instance_id": self._instance_id,
                },
            )
            await self._event_bus.emit(event)
            self._logger.info(
                "benchmark_re_progress",
                current=current,
                previous=prev,
                improvement_pct=round(improvement_pct, 2),
            )

    async def _check_sustained_llm_dependency(self) -> None:
        """
        §23.2: If llm_dependency has not declined over the last 30 snapshots,
        emit a regression for RE maturation stalled.
        """
        if self._event_bus is None:
            return

        try:
            async with self._tsdb.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT llm_dependency FROM benchmark_snapshots
                    WHERE instance_id = $1
                      AND llm_dependency IS NOT NULL
                    ORDER BY time DESC
                    LIMIT 30
                    """,
                    self._instance_id,
                )
        except Exception:
            return

        values = [r["llm_dependency"] for r in rows]
        if len(values) < 30:
            return

        # Check if there's any declining trend: compare first half avg to second half avg
        # values are newest-first, so first half = recent, second half = older
        recent_avg = sum(values[:15]) / 15
        older_avg = sum(values[15:]) / 15

        # No decline means recent >= older (llm_dependency not going down)
        if recent_avg >= older_avg:
            # Only fire once - check if already in _regressed
            stall_key = "llm_dependency_trend"
            if stall_key not in self._regressed:
                self._regressed.add(stall_key)
                self._regressed_at[stall_key] = time.monotonic()
                regression = MetricRegression(
                    metric=stall_key,
                    current_value=recent_avg,
                    rolling_avg=older_avg,
                    regression_pct=0.0,
                    threshold_pct=0.0,
                )
                await self._fire_regression_event(regression)
                self._logger.warning(
                    "llm_dependency_stalled",
                    recent_avg=round(recent_avg, 4),
                    older_avg=round(older_avg, 4),
                    snapshots=len(values),
                )
        else:
            # RE is improving - re-arm
            stall_key = "llm_dependency_trend"
            if stall_key in self._regressed:
                regressed_since = self._regressed_at.pop(stall_key, 0.0)
                duration = time.monotonic() - regressed_since if regressed_since else 0.0
                self._regressed.discard(stall_key)
                await self._fire_recovery_event(stall_key, recent_avg, older_avg, duration)

    async def _tag_episodes_for_re_training(self, snapshot: BenchmarkSnapshot) -> None:
        """
        §25.2B: When decision_quality regresses while llm_dependency > 0.5,
        tag recent Neo4j episodes as high-weight RE training candidates.
        """
        if self._memory is None:
            return
        if snapshot.decision_quality is None or snapshot.llm_dependency is None:
            return
        if snapshot.llm_dependency <= 0.5:
            return
        if "decision_quality" not in self._regressed:
            return

        try:
            neo4j = getattr(self._memory, "_neo4j", None)
            if neo4j is None:
                return
            window_start = utc_now() - timedelta(hours=24)
            await neo4j.execute_write(
                """
                MATCH (ep:Episode)
                WHERE ep.event_time >= $window_start
                  AND ep.used_re = true
                  AND ep.outcome_success = false
                SET ep.training_priority = "high"
                RETURN count(ep) AS tagged
                """,
                {"window_start": window_start.isoformat()},
            )
            self._logger.info("episodes_tagged_for_re_training", window_hours=24)
        except Exception as exc:
            self._logger.debug("episode_tagging_failed", error=str(exc))

    # ─── Redis persistence for _regressed set ─────────────────────────

    _REDIS_REGRESSED_KEY = "eos:benchmarks:regressed:{instance_id}"

    async def _persist_regressed_to_redis(self) -> None:
        """Write _regressed set to Redis to survive restarts."""
        if self._redis is None:
            return
        try:
            key = self._REDIS_REGRESSED_KEY.format(instance_id=self._instance_id)
            if self._regressed:
                await self._redis.set(key, json.dumps(sorted(self._regressed)))
            else:
                await self._redis.delete(key)
        except Exception:
            self._logger.debug("regressed_set_redis_persist_failed")

    async def _restore_regressed_from_redis(self) -> None:
        """Restore _regressed set from Redis on startup."""
        if self._redis is None:
            return
        try:
            key = self._REDIS_REGRESSED_KEY.format(instance_id=self._instance_id)
            raw = await self._redis.get(key)
            if raw:
                metrics = json.loads(raw)
                self._regressed = set(metrics)
                now = time.monotonic()
                for m in self._regressed:
                    self._regressed_at[m] = now
                self._logger.info(
                    "regressed_set_restored",
                    metrics=sorted(self._regressed),
                )
        except Exception:
            self._logger.debug("regressed_set_redis_restore_failed")

    # ─── Auxiliary key-value store ────────────────────────────────────

    async def _store_auxiliary(self, key: str, value: float) -> None:
        """Upsert a single auxiliary float value keyed by name."""
        async with self._tsdb.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO benchmark_aux (key, value, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                """,
                key,
                value,
            )

    async def _latest_raw_value(self, key: str) -> float | None:
        """Retrieve a single auxiliary float value by key."""
        async with self._tsdb.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM benchmark_aux WHERE key = $1", key
            )
            return float(row["value"]) if row else None

    # ─── Query interface (for the dashboard router) ───────────────────

    async def latest_snapshot(self) -> BenchmarkSnapshot | None:
        """Return the most recent benchmark snapshot."""
        async with self._tsdb.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM benchmark_snapshots
                WHERE instance_id = $1
                ORDER BY time DESC
                LIMIT 1
                """,
                self._instance_id,
            )
        if row is None:
            return None
        return BenchmarkSnapshot(
            time=row["time"],
            instance_id=row["instance_id"],
            decision_quality=row["decision_quality"],
            llm_dependency=row["llm_dependency"],
            economic_ratio=row["economic_ratio"],
            learning_rate=row["learning_rate"],
            mutation_success_rate=row["mutation_success_rate"],
            effective_intelligence_ratio=row["effective_intelligence_ratio"],
            compression_ratio=row["compression_ratio"],
            errors=json.loads(row["errors"] or "{}"),
            raw=json.loads(row["raw"] or "{}"),
        )

    async def trend(
        self,
        metric: str,
        since: datetime | None = None,
        limit: int = 50,
    ) -> BenchmarkTrend:
        """
        Return time-series data for a single KPI.

        Parameters
        ──────────
        metric  - one of the five KPI names
        since   - optional start time (defaults to 7 days ago)
        limit   - max number of points (newest first)
        """
        if metric not in _KPI_NAMES:
            raise ValueError(f"Unknown benchmark metric: {metric!r}. Valid: {_KPI_NAMES}")

        if since is None:
            since = utc_now() - timedelta(days=7)

        async with self._tsdb.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT time, {metric} AS value
                FROM benchmark_snapshots
                WHERE instance_id = $1
                  AND time >= $2
                ORDER BY time DESC
                LIMIT $3
                """,
                self._instance_id,
                since,
                limit,
            )

        points = [
            {"time": r["time"].isoformat(), "value": r["value"]}
            for r in rows
        ]

        non_null = [p["value"] for p in points if p["value"] is not None]
        rolling_avg = sum(non_null) / len(non_null) if non_null else None
        latest = points[0]["value"] if points else None

        return BenchmarkTrend(
            metric=metric,
            points=list(reversed(points)),  # chronological order for charts
            rolling_avg=rolling_avg,
            latest=latest,
        )

    async def all_trends(self, since: datetime | None = None) -> dict[str, BenchmarkTrend]:
        """Return trend data for all five KPIs."""
        trends = await asyncio.gather(
            *[self.trend(kpi, since=since) for kpi in _KPI_NAMES],
            return_exceptions=True,
        )
        result: dict[str, BenchmarkTrend] = {}
        for kpi, t in zip(_KPI_NAMES, trends, strict=False):
            if isinstance(t, Exception):
                self._logger.error("benchmark_trend_error", metric=kpi, error=str(t))
            else:
                result[kpi] = t  # type: ignore[assignment]
        return result

    # ─── External KPI Recording ─────────────────────────────────────────

    async def record_kpi(
        self,
        system: str,
        metric: str | None = None,
        value: float | None = None,
        timestamp: datetime | None = None,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Accept external KPIs from another system and store as auxiliary.

        Supports two calling conventions:
          1. Single metric (Soma):
               await benchmarks.record_kpi(system="soma", metric="p99_ms", value=12.3)
          2. Batch dict (Simula, Synapse):
               await benchmarks.record_kpi(system="simula", metrics={"outcome": "applied", ...})

        Non-numeric values in the batch dict are stored as their hash modulo 1_000_000
        so the DOUBLE PRECISION column can accept them.

        Spec ref: §26.2 - fixes batch callers whose data was silently discarded.
        """
        pairs: list[tuple[str, float]] = []

        if metrics is not None:
            # Batch path - Simula and Synapse
            for k, v in metrics.items():
                if isinstance(v, bool):
                    pairs.append((f"{system}.{k}", float(v)))
                elif isinstance(v, (int, float)):
                    pairs.append((f"{system}.{k}", float(v)))
                elif isinstance(v, str):
                    # Categorical values encoded as hash mod 1_000_000
                    pairs.append((f"{system}.{k}", float(abs(hash(v)) % 1_000_000)))
                # None / complex types skipped silently
        elif metric is not None and value is not None:
            # Single-metric path - Soma
            pairs.append((f"{system}.{metric}", float(value)))

        for key, fval in pairs:
            try:
                await self._store_auxiliary(key, fval)
            except Exception as exc:
                self._logger.debug(
                    "benchmark_record_kpi_failed",
                    key=key,
                    error=str(exc),
                )

    # ─── Health check (Synapse protocol) ──────────────────────────────

    async def health(self) -> dict[str, Any]:
        latest = await self.latest_snapshot()
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "total_runs": self._total_runs,
            "total_regressions_fired": self._total_regressions_fired,
            "currently_regressed": sorted(self._regressed),
            "latest_snapshot_time": latest.time.isoformat() if latest else None,
            "interval_s": self._config.interval_s,
            "rolling_window": self._config.rolling_window_snapshots,
            "evolutionary_tracker": self._evo_tracker.stats,
            # Evolutionary / economic fitness counters (surfaced for health monitors)
            "evo_consolidations_total": self._evo_consolidations_total,
            "evo_genome_extractions_total": self._evo_genome_extractions_total,
            "economic_deferrals_total": self._economic_deferrals_total,
            # RE pipeline health
            "re_training_batches_exported": self._re_training_batches_exported,
            "re_training_episodes_total": self._re_training_episodes_total,
            "re_performance": dict(self._re_performance),
            # Phenotype divergence from last Telos population snapshot
            "constitutional_phenotype_divergence": self._last_phenotype_divergence,
        }

    @property
    def stats(self) -> dict[str, Any]:
        snap = self._last_snapshot
        return {
            "initialized": self._initialized,
            "total_runs": self._total_runs,
            "total_regressions_fired": self._total_regressions_fired,
            "active_regressions": sorted(self._regressed),
            "interval_s": self._config.interval_s,
            # Current KPI values from last collection cycle
            "decision_quality": snap.decision_quality if snap else None,
            "llm_dependency": snap.llm_dependency if snap else None,
            "economic_ratio": snap.economic_ratio if snap else None,
            "learning_rate": snap.learning_rate if snap else None,
            "mutation_success_rate": snap.mutation_success_rate if snap else None,
            "effective_intelligence_ratio": snap.effective_intelligence_ratio if snap else None,
            "compression_ratio": snap.compression_ratio if snap else None,
            "bedau_packard": snap.bedau_packard.model_dump(mode="json") if snap and snap.bedau_packard else None,
            # Evolutionary / economic fitness KPIs (previously invisible)
            "evo_consolidations_total": self._evo_consolidations_total,
            "evo_genome_extractions_total": self._evo_genome_extractions_total,
            "economic_deferrals_total": self._economic_deferrals_total,
            "economic_deferrals_by_type": dict(self._economic_deferrals_by_type),
            # RE training pipeline KPIs (previously invisible)
            "re_training_batches_exported": self._re_training_batches_exported,
            "re_training_episodes_total": self._re_training_episodes_total,
            "re_training_last_mean_quality": self._re_training_last_mean_quality,
            # RE performance from 7-day rolling window (previously invisible)
            "re_performance": dict(self._re_performance),
            # Constitutional phenotype divergence from last Telos snapshot (previously invisible)
            "constitutional_phenotype_divergence": self._last_phenotype_divergence,
            # Domain specialisation snapshot
            "primary_domain": snap.primary_domain if snap else "generalist",
            "active_domain_count": len(snap.domain_kpis) if snap else 0,
            # Runtime-adjustable thresholds (visible so Evo can observe before adjusting)
            "threshold_re_progress_min_improvement_pct": self._re_progress_min_improvement_pct,
            "threshold_metabolic_degradation_fraction": self._metabolic_degradation_fraction,
            # Learning trajectory KPIs
            "crash_patterns_discovered": self._crash_patterns_discovered,
            "crash_patterns_resolved": self._crash_patterns_resolved,
            "crash_pattern_confidence_avg": round(
                self._crash_pattern_confidence_sum / max(1, self._crash_patterns_discovered), 3
            ),
            "crash_pattern_resolution_rate": round(
                self._crash_patterns_resolved / max(1, self._crash_patterns_discovered), 3
            ),
            "re_model_health_history_len": len(self._re_model_health_history),
            "organism_learning_velocity": self._compute_learning_velocity(),
        }
