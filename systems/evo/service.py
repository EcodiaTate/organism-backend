"""
EcodiaOS — Evo Service

The Learning & Hypothesis system. Evo is the Growth drive made computational.

Evo observes the stream of experience, forms hypotheses, accumulates evidence,
and — when the evidence is sufficient — adjusts the organism's parameters,
codifies successful procedures, and proposes structural changes.

It operates in two modes:
  WAKE (online)   — lightweight pattern detection during each cognitive cycle
  SLEEP (offline) — deep consolidation: schema induction, procedure extraction,
                     parameter optimisation, self-model update

Interface:
  initialize()          — build sub-systems, load persisted parameter state
  receive_broadcast()   — online learning step (called by Synapse, ≤20ms budget)
  run_consolidation()   — explicit trigger for sleep mode
  shutdown()            — graceful teardown
  get_parameter()       — current value of any tunable parameter
  stats                 — service-level metrics

Cognitive cycle role (step 7 — LEARN):
  Evo runs as a background participant. It receives every workspace broadcast,
  updates its pattern context, and occasionally triggers hypothesis generation.
  The consolidation cycle runs asynchronously and never blocks the theta rhythm.

Guard rails inherited from sub-systems:
  - Velocity limits on parameter changes
  - Hypotheses must be falsifiable
  - Cannot touch Equor evaluation logic or constitutional drives
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector, SystemID
from primitives.memory_trace import Episode
from primitives.re_training import RETrainingExample
from systems.evo.belief_consolidation import BeliefConsolidationScanner
from systems.evo.belief_halflife import BeliefAgingScanner
from systems.evo.cognitive_niche import NicheRegistry
from systems.evo.consolidation import ConsolidationOrchestrator
from systems.evo.curiosity import CuriosityEngine
from systems.evo.detectors import PatternDetector, build_default_detectors
from systems.evo.genetic_memory import GenomeExtractor
from systems.evo.hypothesis import HypothesisEngine, StructuralHypothesisGenerator
from systems.evo.meta_learning import MetaLearningEngine
from systems.evo.niche_forking import NicheForkingEngine
from systems.evo.parameter_tuner import ParameterTuner
from systems.evo.pressure import EvolutionaryPressureSystem
from systems.evo.procedure_codifier import IntentRecord, OutcomeRecord, ProcedureCodifier
from systems.evo.procedure_extractor import ProcedureExtractor
from systems.evo.research_worker import ArxivScientist
from systems.evo.schema_induction import SchemaInductionEngine
from systems.evo.self_model import SelfModelManager
from systems.evo.self_modification import SelfModificationEngine
from systems.evo.speciation import SpeciationEngine
from systems.evo.tournament import TournamentEngine
from systems.evo.types import (
    ConsolidationResult,
    Hypothesis,
    HypothesisStatus,
    PatternCandidate,
    PatternContext,
    PatternType,
    SelfModelStats,
    TournamentOutcome,
)
if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.redis import RedisClient
    from config import EvoConfig
    from core.hotreload import NeuroplasticityBus
    from systems.fovea.types import WorkspaceBroadcast
    from systems.memory.service import MemoryService

logger = structlog.get_logger()

# How often to attempt hypothesis generation from accumulated patterns
_HYPOTHESIS_GENERATION_INTERVAL: int = 200  # Every 200 broadcasts (was 50)
# How often to evaluate evidence against all active hypotheses
_EVIDENCE_EVALUATION_INTERVAL: int = 50     # Every 50 broadcasts (was 10)
# How many direction reversals in the 10-nudge window triggers HIGH_VOLATILITY
_VOLATILITY_OSCILLATION_THRESHOLD: int = 6


class EvoService:
    """
    Evo — the EOS learning and hypothesis system.

    Coordinates four sub-systems:
      HypothesisEngine       — hypothesis lifecycle
      ParameterTuner         — parameter adjustment with velocity limiting
      ProcedureExtractor     — action sequence → procedure codification
      SelfModelManager       — meta-cognitive self-assessment
      ConsolidationOrchestrator — sleep mode pipeline
    """

    system_id: str = "evo"

    def __init__(
        self,
        config: EvoConfig,
        llm: LLMProvider,
        memory: MemoryService | None = None,
        instance_name: str = "EOS",
        neuroplasticity_bus: NeuroplasticityBus | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._memory = memory
        self._instance_name = instance_name
        self._bus = neuroplasticity_bus
        self._initialized: bool = False
        self._logger = logger.bind(system="evo")

        # Cross-system references (wired post-init by main.py)
        self._atune: Any = None  # AtuneService — for pushing learned head weights
        self._nova: Any = None   # NovaService — for generating epistemic goals from hypotheses
        self._voxis: Any = None  # VoxisService — for personality learning from expression outcomes
        self._soma: Any = None   # SomaService — for curiosity modulation and dynamics update
        self._simula: Any = None  # SimulaService — for dispatching arXiv evolution proposals
        self._telos: Any = None   # TelosService — for hypothesis prioritisation
        self._kairos: Any = None  # KairosPipeline — for causal validation of hypotheses
        self._fovea: Any = None   # FoveaService — for internal prediction errors
        self._logos: Any = None   # LogosService — for MDL scoring of hypotheses

        # Sub-systems (built in initialize())
        self._hypothesis_engine: HypothesisEngine | None = None
        self._parameter_tuner: ParameterTuner | None = None
        self._procedure_extractor: ProcedureExtractor | None = None
        self._procedure_codifier: ProcedureCodifier | None = None
        self._self_model: SelfModelManager | None = None
        self._orchestrator: ConsolidationOrchestrator | None = None
        self._tournament_engine: TournamentEngine | None = None
        self._schema_engine: SchemaInductionEngine | None = None
        self._meta_learning: MetaLearningEngine | None = None
        self._curiosity_engine: CuriosityEngine | None = None
        self._pressure_system: EvolutionaryPressureSystem | None = None
        self._self_modification: SelfModificationEngine | None = None
        self._structural_generator: StructuralHypothesisGenerator | None = None
        self._niche_registry: NicheRegistry | None = None
        self._speciation_engine: SpeciationEngine | None = None
        self._niche_forking_engine: NicheForkingEngine | None = None

        # Online state
        self._detectors: list[PatternDetector] = []
        self._pattern_context: PatternContext = PatternContext()
        self._pending_candidates: list[PatternCandidate] = []

        # Cycle counters
        self._total_broadcasts: int = 0
        self._cycles_since_consolidation: int = 0
        self._total_consolidations: int = 0
        self._total_evidence_evaluations: int = 0

        # Background task handles
        self._consolidation_task: asyncio.Task[None] | None = None
        self._consolidation_in_flight: bool = False
        self._arxiv_scan_task: asyncio.Task[None] | None = None

        # ── Consolidation liveness tracking ───────────────────────────────
        # Monotonic timestamp of the last *completed* consolidation run.
        # None until the first consolidation finishes.
        self._last_consolidation_completed_at: float | None = None
        # Expected interval in seconds (mirrors ConsolidationOrchestrator).
        # Used to detect a stalled loop at 2× this interval.
        self._consolidation_expected_interval_s: float = 6.0 * 3600  # 6 hours

        # ── RE performance degradation tracking ───────────────────────────
        # Consecutive RE_DECISION_OUTCOME events where success_rate < 0.60.
        # After 10 consecutive degraded readings a hyperparameter-adjustment
        # PatternCandidate is queued for the next consolidation pass.
        self._re_degradation_count: int = 0

        # ── Hypothesis budget degradation tracking ────────────────────────
        # Consecutive hypothesis generation cycles that were skipped due to
        # LLM budget exhaustion.
        self._consecutive_hypothesis_skips: int = 0
        # Logos cognitive pressure gate — when True, non-critical hypothesis
        # generation is paused to reduce compute during compression pressure.
        self._cognitive_pressure_high: bool = False
        # Thymos reference — wired via set_thymos() if called.
        self._thymos: Any = None

        # Synapse event bus — wired via set_event_bus()
        self._event_bus: Any = None

        # Cached Soma curiosity drive — updated from SOMATIC_MODULATION_SIGNAL events
        # so we never call soma.get_current_signal() directly.
        self._cached_soma_curiosity_drive: float = 0.5

        # Redis client — wired via set_redis(); used for PatternContext checkpoints
        # so accumulated detector state survives restarts between consolidations.
        self._redis: Any = None

        # ArXiv research pipeline
        self._arxiv_scientist: ArxivScientist = ArxivScientist(llm=self._llm)
        # _arxiv_translator is intentionally lazy — imported inside
        # _handle_new_arxiv_innovation() to avoid the cross-system top-level import.

        # ── Metabolic gating ──────────────────────────────────────────────
        self._starvation_level: str = "nominal"

        # ── Speciation tracking (Task 2) ─────────────────────────────────
        # Rolling confirmation rate: deque of (timestamp, confirmed_count, total_count)
        self._confirmation_history: collections.deque[tuple[float, int, int]] = (
            collections.deque(maxlen=100)
        )
        # Track known hypothesis domains for novel domain detection
        self._known_hypothesis_domains: set[str] = set()
        # Last speciation event timestamp (monotonic) — enforce max 1 per 24h
        self._last_speciation_event_at: float = 0.0

        # ── Economic learning state ────────────────────────────────────────────
        # Rolling window of bounty attempt outcomes: True=success, False=failure
        self._bounty_outcomes: collections.deque[bool] = collections.deque(maxlen=10)
        # Rolling window of yield APY snapshots: float values
        self._yield_apy_history: collections.deque[float] = collections.deque(maxlen=10)

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Build all sub-systems and load persisted parameter state.
        Must be called before any other method.
        """
        if self._initialized:
            return

        # Meta-learning engine — Evo learns about how it learns
        # (created early so it can be passed to HypothesisEngine)
        self._meta_learning = MetaLearningEngine()

        self._hypothesis_engine = HypothesisEngine(
            llm=self._llm,
            instance_name=self._instance_name,
            memory=self._memory,
            meta_learning=self._meta_learning,
        )
        self._parameter_tuner = ParameterTuner(memory=self._memory)
        self._procedure_extractor = ProcedureExtractor(
            llm=self._llm,
            memory=self._memory,
        )
        self._procedure_codifier = ProcedureCodifier(
            llm=self._llm,
            memory=self._memory,
        )
        self._self_model = SelfModelManager(memory=self._memory)

        # Belief half-life scanner — requires Neo4j via MemoryService
        belief_aging: BeliefAgingScanner | None = None
        if self._memory is not None:
            belief_aging = BeliefAgingScanner(neo4j=self._memory)

        self._belief_aging = belief_aging

        # Belief consolidation scanner — hardens high-confidence beliefs into read-only nodes
        belief_consolidation: BeliefConsolidationScanner | None = None
        if self._memory is not None:
            belief_consolidation = BeliefConsolidationScanner(neo4j=self._memory)
        self._belief_consolidation = belief_consolidation

        # Tournament engine for competitive A/B hypothesis experimentation
        self._tournament_engine = TournamentEngine(
            hypothesis_engine=self._hypothesis_engine,
            memory=self._memory,
        )

        # Genetic memory extractor — compresses stable beliefs for child inheritance
        genome_extractor: GenomeExtractor | None = None
        if self._memory is not None:
            genome_extractor = GenomeExtractor(
                neo4j=self._memory,
                instance_id=self._config.instance_id if hasattr(self._config, "instance_id") else self._instance_name,
            )
        self._genome_extractor = genome_extractor

        # Schema induction engine — real structure learning from the knowledge graph
        self._schema_engine = SchemaInductionEngine(
            memory=self._memory,
            logos=self._logos,
        )

        # Curiosity engine — epistemic intent generation for active exploration
        self._curiosity_engine = CuriosityEngine(memory=self._memory)

        # Evolutionary pressure system — fitness landscapes and selection
        self._pressure_system = EvolutionaryPressureSystem(memory=self._memory)

        # Self-modification engine — recursive self-improvement of learning itself
        self._self_modification = SelfModificationEngine(
            meta_learning=self._meta_learning,
        )

        # Structural hypothesis generator — graph-topology hypotheses (no LLM)
        self._structural_generator = StructuralHypothesisGenerator(
            memory=self._memory,
        )

        # Causal failure analyzer — feeds Phase 6.5 failure-pattern detection
        # and Phase 8 causal-surgery proposals.  Requires Neo4j + LLM; skipped
        # when memory is not wired (test / lightweight environments).
        self._causal_surgery_analyzer: Any = None
        if self._memory is not None:
            try:
                from systems.simula.coevolution.causal_surgery import (
                    CausalDAGBuilder,
                    CausalFailureAnalyzer,
                )

                _dag_builder = CausalDAGBuilder(neo4j=self._memory)
                self._causal_surgery_analyzer = CausalFailureAnalyzer(
                    neo4j=self._memory,
                    llm=self._llm,
                    dag_builder=_dag_builder,
                )
            except Exception as _exc:
                self._logger.warning(
                    "causal_surgery_analyzer_init_failed",
                    error=str(_exc),
                )

        # ── Cognitive Speciation Subsystem ────────────────────────────────────
        # Niche registry — manages isolated hypothesis ecosystems
        self._niche_registry = NicheRegistry()

        # Speciation engine — evolves new ways of thinking via 5 biological mechanisms
        self._speciation_engine = SpeciationEngine(
            niche_registry=self._niche_registry,
            pressure_system=self._pressure_system,
        )

        # Niche forking engine — cognitive organogenesis (the organism grows new organs)
        self._niche_forking_engine = NicheForkingEngine(
            niche_registry=self._niche_registry,
        )

        self._orchestrator = ConsolidationOrchestrator(
            hypothesis_engine=self._hypothesis_engine,
            parameter_tuner=self._parameter_tuner,
            procedure_extractor=self._procedure_extractor,
            self_model_manager=self._self_model,
            memory=self._memory,
            logos=self._logos,
            belief_aging_scanner=belief_aging,
            belief_consolidation_scanner=belief_consolidation,
            tournament_engine=self._tournament_engine,
            genome_extractor=genome_extractor,
            causal_surgery_analyzer=self._causal_surgery_analyzer,
            procedure_codifier=self._procedure_codifier,
            oikos=self._oikos if hasattr(self, "_oikos") else None,
            schema_induction_engine=self._schema_engine,
            meta_learning_engine=self._meta_learning,
            speciation_engine=self._speciation_engine,
            niche_forking_engine=self._niche_forking_engine,
            pressure_system=self._pressure_system,
            niche_registry=self._niche_registry,
        )

        self._detectors = build_default_detectors()
        self._detectors.append(EconomicPatternDetector())

        # Restore persisted parameter values
        restored = await self._parameter_tuner.load_from_memory()

        self._initialized = True

        # Restore PatternContext from Redis checkpoint (Spec §III gap fix)
        pattern_restored = await self._restore_pattern_context_checkpoint()

        self._logger.info(
            "evo_initialized",
            detectors=len(self._detectors),
            parameters_restored=restored,
            pattern_context_restored=pattern_restored,
        )

        # Register with the NeuroplasticityBus for hot-reload of PatternDetector subclasses.
        if self._bus is not None:
            self._bus.register(
                base_class=PatternDetector,
                registration_callback=self._on_detector_evolved,
                system_id="evo",
            )

    async def shutdown(self) -> None:
        """Graceful shutdown. Cancels consolidation and arXiv scan tasks."""
        if self._bus is not None:
            self._bus.deregister(PatternDetector)

        if self._consolidation_task and not self._consolidation_task.done():
            self._consolidation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consolidation_task

        if self._arxiv_scan_task and not self._arxiv_scan_task.done():
            self._arxiv_scan_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._arxiv_scan_task

        await self._arxiv_scientist.close()

        self._logger.info(
            "evo_shutdown",
            total_broadcasts=self._total_broadcasts,
            total_consolidations=self._total_consolidations,
            total_evidence_evaluations=self._total_evidence_evaluations,
            arxiv_stats=self._arxiv_scientist.stats,
            hypothesis_stats=(
                self._hypothesis_engine.stats
                if self._hypothesis_engine else {}
            ),
        )

    # ─── Online Learning (Wake Mode) ──────────────────────────────────────────

    async def receive_broadcast(self, broadcast: WorkspaceBroadcast) -> None:
        """
        Online learning step. Called by the cognitive cycle (step 7 — LEARN).
        Budget: ≤20ms for pattern scanning. Heavy work is fire-and-forget.

        Does NOT raise — Evo failures must not interrupt the cognitive cycle.
        """
        if not self._initialized:
            return

        self._total_broadcasts += 1
        self._cycles_since_consolidation += 1

        try:
            # Update context with current broadcast data
            self._pattern_context.previous_affect = self._pattern_context.current_affect
            self._pattern_context.current_affect = broadcast.affect

            # Extract entity IDs from memory context (for CooccurrenceDetector)
            entity_ids: list[str] = []
            for trace in broadcast.context.memory_context.traces:
                entity_ids.extend(trace.entities)
            self._pattern_context.recent_entity_ids = list(set(entity_ids))[:20]

            # Run lightweight pattern scanning from the percept
            # We create a minimal Episode from broadcast for the detectors
            episode = _broadcast_to_episode(broadcast)
            await self._scan_episode_online(episode)

            # Curiosity-modulated hypothesis generation interval
            # High curiosity → generate hypotheses more aggressively
            curiosity_multiplier = 1.0
            # Use cached Soma curiosity drive (updated via SOMATIC_MODULATION_SIGNAL handler)
            curiosity_drive = getattr(self, "_cached_soma_curiosity_drive", 0.5)
            curiosity_multiplier = 0.5 + curiosity_drive * 1.0

            effective_interval = max(
                10, int(_HYPOTHESIS_GENERATION_INTERVAL / curiosity_multiplier)
            )

            # Periodically generate hypotheses from accumulated patterns
            if self._total_broadcasts % effective_interval == 0:
                asyncio.create_task(
                    self._generate_hypotheses_safe(),
                    name="evo_hypothesis_generation",
                )

            # Soma curiosity also modulates evidence evaluation: high curiosity
            # means more frequent evidence sweeps (the organism is actively
            # curious and wants to test its hypotheses faster)
            effective_evidence_interval = max(
                5, int(_EVIDENCE_EVALUATION_INTERVAL / curiosity_multiplier)
            )
            if self._total_broadcasts % effective_evidence_interval == 0:
                asyncio.create_task(
                    self._evaluate_recent_evidence_safe(),
                    name="evo_evidence_evaluation",
                )

            # Periodic PatternContext checkpoint to Redis every 1000 broadcasts
            # so online state survives restart between consolidations (Spec §III)
            if self._total_broadcasts % 1000 == 0:
                asyncio.create_task(
                    self._save_pattern_context_checkpoint(),
                    name="evo_pattern_checkpoint",
                )

        except Exception as exc:
            self._logger.error("broadcast_processing_failed", error=str(exc))

    async def process_episode(self, episode: Episode) -> None:
        """
        Evaluate an episode as evidence against all active hypotheses.
        Called during evidence evaluation sweep (fire-and-forget from broadcast handler).
        Budget: per-hypothesis ≤200ms (from hypothesis_engine.evaluate_evidence).
        """
        if not self._initialized or self._hypothesis_engine is None:
            return

        active = self._hypothesis_engine.get_active()
        for h in active:
            try:
                result = await self._hypothesis_engine.evaluate_evidence(h, episode)
                self._total_evidence_evaluations += 1

                # Emit lifecycle events on status transitions
                if result is not None:
                    if result.new_status == HypothesisStatus.SUPPORTED:
                        # Emit EVO_HYPOTHESIS_CONFIRMED
                        await self._emit_hypothesis_lifecycle_events([h], "confirmed")
                        if self._nova is not None:
                            await self._generate_goal_from_hypothesis(h)
                        await self._emit_evolutionary_observable(
                            observable_type="hypothesis_validated",
                            value=h.evidence_score,
                            is_novel=True,
                            metadata={
                                "hypothesis_id": h.id,
                                "statement": h.statement[:200],
                                "category": h.category.value if hasattr(h.category, "value") else str(h.category),
                                "supporting_count": len(getattr(h, "supporting_episodes", [])),
                                "path": "bayesian_evidence",
                            },
                        )
                        # RE training: hypothesis confirmation
                        # growth alignment = evidence_score normalized to [-1,1]
                        _h_growth = min(1.0, h.evidence_score / 10.0) * 2.0 - 1.0
                        await self._emit_re_training_example(
                            category="hypothesis_reasoning",
                            instruction=f"Evaluate hypothesis: {h.statement[:200]}",
                            input_context=(
                                f"evidence_score={h.evidence_score:.2f}, "
                                f"supporting={len(h.supporting_episodes)}, "
                                f"contradicting={len(h.contradicting_episodes)}"
                            ),
                            output="CONFIRMED",
                            outcome_quality=min(1.0, h.evidence_score / 10.0),
                            episode_id=h.id,
                            reasoning_trace=(
                                f"Formal test: {h.formal_test[:200]}. "
                                f"Confidence trajectory: score={h.evidence_score:.2f}"
                            ),
                            constitutional_alignment=DriveAlignmentVector(
                                coherence=round(min(1.0, h.evidence_score / 10.0), 3),
                                growth=round(max(-1.0, _h_growth), 3),
                            ),
                        )
                    elif result.new_status == HypothesisStatus.REFUTED:
                        # Emit EVO_HYPOTHESIS_REFUTED
                        await self._emit_hypothesis_lifecycle_events([h], "refuted")
                        # RE training: hypothesis refutation
                        await self._emit_re_training_example(
                            category="hypothesis_reasoning",
                            instruction=f"Evaluate hypothesis: {h.statement[:200]}",
                            input_context=(
                                f"evidence_score={h.evidence_score:.2f}, "
                                f"supporting={len(h.supporting_episodes)}, "
                                f"contradicting={len(h.contradicting_episodes)}"
                            ),
                            output="REFUTED",
                            outcome_quality=0.0,
                            episode_id=h.id,
                            reasoning_trace=(
                                f"Formal test: {h.formal_test[:200]}. "
                                f"Score dropped to {h.evidence_score:.2f}"
                            ),
                            constitutional_alignment=DriveAlignmentVector(
                                coherence=-0.5,  # refuted hypothesis = incoherence signal
                                growth=-0.3,     # failed growth attempt
                            ),
                        )

            except Exception as exc:
                self._logger.warning(
                    "evidence_evaluation_error",
                    hypothesis_id=h.id,
                    error=str(exc),
                )

    async def _generate_goal_from_hypothesis(self, hypothesis: Any) -> None:
        """
        Convert a supported hypothesis into an epistemic exploration goal.

        When Evo accumulates enough evidence to support a hypothesis, the
        organism should actively explore and test it — not just passively wait.
        """
        if self._event_bus is None:
            return
        try:
            from primitives.common import new_id
            from systems.synapse.types import SynapseEvent, SynapseEventType

            goal_id = new_id()
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.NOVA_GOAL_INJECTED,
                source_system="evo",
                data={
                    "goal_id": goal_id,
                    "description": f"Explore supported hypothesis: {hypothesis.statement[:120]}",
                    "source": "epistemic",
                    "priority": 0.55,
                    "urgency": 0.3,
                    "importance": 0.6,
                    "drive_alignment": {"coherence": 0.3, "care": 0.0, "growth": 0.7, "honesty": 0.0},
                    "hypothesis_id": hypothesis.id,
                },
            ))
            self._logger.info(
                "epistemic_goal_generated",
                hypothesis_id=hypothesis.id,
                goal_id=goal_id,
            )
        except Exception as exc:
            self._logger.warning("epistemic_goal_failed", error=str(exc))

    # ─── Consolidation (Sleep Mode) ────────────────────────────────────────────

    async def run_consolidation(self) -> ConsolidationResult | None:
        """
        Trigger a consolidation cycle explicitly.
        Returns None if already running or not initialized.
        Safe to call from tests and management APIs.
        """
        if not self._initialized or self._orchestrator is None:
            return None

        # Metabolic gate: no consolidation under EMERGENCY/CRITICAL
        if self._starvation_level in ("emergency", "critical"):
            self._logger.info("consolidation_blocked_starvation", level=self._starvation_level)
            return None

        if self._consolidation_task and not self._consolidation_task.done():
            self._logger.info("consolidation_already_running")
            return None

        return await self._run_consolidation_now()

    def schedule_consolidation_loop(self) -> None:
        """
        Start the background consolidation loop under supervision.

        Uses supervised_task() so crashes are logged with full context and
        a TASK_PERMANENTLY_FAILED event is emitted if the loop exhausts its
        restart budget.
        """
        from utils.supervision import supervised_task

        self._consolidation_task = supervised_task(
            self._consolidation_loop(),
            name="evo_consolidation_loop",
            restart=True,
            max_restarts=3,
            backoff_base=2.0,
            event_bus=self._event_bus,
            source_system="evo",
        )

    def schedule_arxiv_scan(self) -> None:
        """
        Start the daily arXiv research scan loop.

        Runs ArxivScientist.run_daily_scan() once every 24 hours, translates
        discovered techniques into Simula evolution proposals, and dispatches
        them for governance review. Requires Simula to be wired via set_simula().

        Called once by the application startup (e.g., from main.py).
        """
        self._arxiv_scan_task = asyncio.create_task(
            self._arxiv_scan_loop(),
            name="evo_arxiv_scan_loop",
        )
        self._logger.info("arxiv_scan_loop_scheduled")

    # ─── Parameter Query ──────────────────────────────────────────────────────

    def get_parameter(self, name: str) -> float | None:
        """
        Return the current value of a tunable parameter.
        Systems call this each cycle to pick up Evo-applied adjustments.
        Returns None if parameter is unknown.
        """
        if self._parameter_tuner is None:
            return None
        return self._parameter_tuner.get_current_parameter(name)

    def get_all_parameters(self) -> dict[str, float]:
        """Return all current parameter values."""
        if self._parameter_tuner is None:
            return {}
        return self._parameter_tuner.get_all_parameters()

    async def apply_immune_parameter_adjustment(
        self,
        parameter_path: str,
        delta: float,
        reason: str = "",
    ) -> bool:
        """
        Apply a parameter adjustment driven by the Thymos immune system (Tier 1 repair).

        Bypasses hypothesis requirements — Thymos has already validated the repair
        through its own governance pipeline.  Velocity limits and range clamping
        from ParameterTuner still apply.

        Returns True if the parameter was known and the adjustment was applied.
        """
        if self._parameter_tuner is None:
            return False

        from systems.evo.types import TUNABLE_PARAMETERS, VELOCITY_LIMITS, ParameterAdjustment

        spec = TUNABLE_PARAMETERS.get(parameter_path)
        if spec is None:
            self._logger.warning(
                "immune_parameter_unknown",
                parameter=parameter_path,
                reason=reason,
            )
            return False

        current = self._parameter_tuner.get_current_parameter(parameter_path)
        if current is None:
            return False

        # Clamp to velocity limit and valid range
        max_step = min(spec.step, VELOCITY_LIMITS["max_single_parameter_delta"])
        clamped_delta = max(-max_step, min(max_step, delta))
        new_value = max(spec.min_val, min(spec.max_val, current + clamped_delta))
        actual_delta = new_value - current

        if abs(actual_delta) < 0.0001:
            return True  # Already at boundary — not an error

        adjustment = ParameterAdjustment(
            parameter=parameter_path,
            old_value=current,
            new_value=new_value,
            delta=actual_delta,
            hypothesis_id="thymos_immune_repair",
            evidence_score=1.0,
            supporting_count=0,
        )
        await self._parameter_tuner.apply_adjustment(adjustment)
        self._logger.info(
            "immune_parameter_applied",
            parameter=parameter_path,
            delta=round(actual_delta, 4),
            reason=reason,
        )
        await self._emit_evolutionary_observable(
            observable_type="parameter_adjusted",
            value=actual_delta,
            is_novel=True,
            metadata={
                "parameter": parameter_path,
                "old_value": current,
                "new_value": new_value,
                "reason": reason[:200],
                "source": "thymos_immune_repair",
            },
        )
        return True

    def get_self_model(self) -> SelfModelStats | None:
        """Return the current self-model statistics."""
        if self._self_model is None:
            return None
        return self._self_model.get_current()

    def get_capability_rate(self, capability: str) -> float | None:
        """Return the success rate for a named capability."""
        if self._self_model is None:
            return None
        return self._self_model.get_capability_rate(capability)

    async def get_active_hypothesis_count(self) -> int:
        """
        Return the count of currently active (proposed + testing) hypotheses.

        Called by Oneiros SleepCycleEngine during Descent to record the
        pre-sleep hypothesis backlog in the SleepCheckpoint.
        Satisfies EvoHypothesisProtocol.get_active_hypothesis_count.
        """
        if self._hypothesis_engine is None:
            return 0
        return len(self._hypothesis_engine.get_active())

    async def extract_new_hypotheses(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """
        Return recent hypotheses not yet integrated, for Oneiros REM dream testing.

        Each dict: {"id": str, "statement": str, "category": str,
                     "evidence_score": float, "proposed_mutation": dict | None}

        Satisfies EvoHypothesisProtocol.extract_new_hypotheses.
        """
        if self._hypothesis_engine is None:
            return []
        active = self._hypothesis_engine.get_active()
        # Sort by recency (last evidence), newest first
        sorted_hyps = sorted(active, key=lambda h: h.last_evidence_at, reverse=True)
        result: list[dict[str, Any]] = []
        for h in sorted_hyps[:limit]:
            result.append({
                "id": h.id,
                "statement": h.statement,
                "category": h.category.value,
                "evidence_score": h.evidence_score,
                "proposed_mutation": (
                    h.proposed_mutation.model_dump() if h.proposed_mutation else None
                ),
            })
        return result

    def set_atune(self, atune: Any) -> None:
        """Wire Atune so Evo can push learned head-weight adjustments."""
        self._atune = atune
        self._logger.info("atune_wired_to_evo")

    def set_nova(self, nova: Any) -> None:
        """Wire Nova so supported hypotheses generate epistemic exploration goals."""
        self._nova = nova
        self._logger.info("nova_wired_to_evo")

    def set_voxis(self, voxis: Any) -> None:
        """Wire Voxis so Evo can push personality adjustments from expression outcomes."""
        self._voxis = voxis
        self._logger.info("voxis_wired_to_evo")

    def set_telos(self, telos: Any) -> None:
        """Wire Telos so hypothesis prioritisation accounts for drive topology."""
        self._telos = telos
        self._logger.info("telos_wired_to_evo")

    def set_soma(self, soma: Any) -> None:
        """Wire Soma for curiosity modulation and dynamics learning."""
        self._soma = soma
        self._logger.info("soma_wired_to_evo")

    def set_kairos(self, kairos: Any) -> None:
        """Wire Kairos so causal hypotheses are fed into the causal mining pipeline."""
        self._kairos = kairos
        self._logger.info("kairos_wired_to_evo")

    def set_fovea(self, fovea: Any) -> None:
        """Wire Fovea so competency-type internal prediction errors feed self-model updates."""
        self._fovea = fovea
        self._logger.info("fovea_wired_to_evo")

    def set_logos(self, logos: Any) -> None:
        """Wire Logos so hypothesis quality is evaluated via MDL scoring."""
        self._logos = logos
        if self._orchestrator is not None:
            self._orchestrator._logos = logos
        self._logger.info("logos_wired_to_evo")

    def wire_oikos(self, oikos: Any) -> None:
        """
        Wire the Oikos metabolic service so Phase 5 parameter optimisation
        can check the GROWTH gate before running expensive tuning.

        Must be called after initialize() and before the first consolidation
        cycle.
        """
        self._oikos = oikos
        if self._orchestrator is not None:
            self._orchestrator._oikos = oikos
        self._logger.info("oikos_wired_to_evo")

    def wire_event_bus(self, event_bus: Any) -> None:
        """
        Wire the Synapse EventBus into the ConsolidationOrchestrator so that
        Phase 8 can emit EVOLUTION_CANDIDATE events for high-confidence
        hypotheses (confidence >= 0.9).

        Also stores the bus on self so Evo can emit its own degradation events.

        Must be called after initialize() and before the first consolidation
        cycle.  main.py is responsible for the wiring order.
        """
        self._event_bus = event_bus
        if self._orchestrator is not None:
            self._orchestrator._event_bus = event_bus
        if self._niche_forking_engine is not None:
            self._niche_forking_engine._event_bus = event_bus
        # Wire into ParameterTuner so every apply_adjustment() pushes
        # EVO_PARAMETER_ADJUSTED — no more polling for Atune/Nova/Voxis.
        if self._parameter_tuner is not None:
            self._parameter_tuner.wire_event_bus(event_bus)
        self._logger.info("event_bus_wired_to_evo_orchestrator")

    async def _emit_re_training_example(
        self,
        category: str,
        instruction: str,
        input_context: str,
        output: str,
        outcome_quality: float,
        episode_id: str = "",
        cost_usd: Decimal = Decimal("0"),
        latency_ms: int = 0,
        reasoning_trace: str = "",
        alternatives: list[str] | None = None,
        constitutional_alignment: DriveAlignmentVector | None = None,
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            example = RETrainingExample(
                source_system=SystemID.EVO,
                episode_id=episode_id,
                instruction=instruction,
                input_context=input_context,
                output=output,
                outcome_quality=outcome_quality,
                category=category,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives or [],
                constitutional_alignment=constitutional_alignment or DriveAlignmentVector(),
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                data=example.model_dump(mode="json"),
                source_system="evo",
            ))
        except Exception:
            self._logger.debug("re_training_emit_failed", exc_info=True)

    async def _emit_evolutionary_observable(
        self,
        observable_type: str,
        value: float,
        is_novel: bool,
        metadata: dict | None = None,
    ) -> None:
        """Emit an EvolutionaryObservable event on Synapse for population tracking."""
        bus = self._event_bus
        if bus is None:
            return
        try:
            from primitives.common import SystemID
            from primitives.evolutionary import EvolutionaryObservable
            from systems.synapse.types import SynapseEvent, SynapseEventType

            obs = EvolutionaryObservable(
                source_system=SystemID.EVO,
                instance_id=getattr(self, "_instance_id", "") or "",
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system=SystemID.EVO,
                data=obs.model_dump(mode="json"),
            )
            await bus.emit(event)
        except Exception:
            pass  # Best-effort — never block the learning loop

    async def _emit_hypothesis_lifecycle_events(
        self,
        hypotheses: list,
        lifecycle: str,
    ) -> None:
        """
        Emit EVO_HYPOTHESIS_CREATED / CONFIRMED / REFUTED for a batch of hypotheses.
        """
        if self._event_bus is None or not hypotheses:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            event_map = {
                "created": SynapseEventType.EVO_HYPOTHESIS_CREATED,
                "confirmed": SynapseEventType.EVO_HYPOTHESIS_CONFIRMED,
                "refuted": SynapseEventType.EVO_HYPOTHESIS_REFUTED,
            }
            event_type = event_map.get(lifecycle)
            if event_type is None:
                return

            for h in hypotheses:
                category = h.category.value if hasattr(h.category, "value") else str(h.category)
                data: dict = {
                    "hypothesis_id": h.id,
                    "category": category,
                    "statement": h.statement[:300],
                }
                if lifecycle == "created":
                    data["source_detector"] = getattr(h, "source_detector", "")
                    data["novelty_score"] = getattr(h, "novelty_score", 0.0)
                elif lifecycle == "confirmed":
                    data["evidence_score"] = h.evidence_score
                    data["supporting_count"] = len(getattr(h, "supporting_episodes", []))
                elif lifecycle == "refuted":
                    data["evidence_score"] = h.evidence_score
                    data["contradicting_count"] = len(getattr(h, "contradicting_episodes", []))

                await self._event_bus.emit(SynapseEvent(
                    event_type=event_type,
                    source_system="evo",
                    data=data,
                ))
        except Exception:
            self._logger.debug("hypothesis_lifecycle_emit_failed", exc_info=True)

    async def _emit_consolidation_complete(self, result: ConsolidationResult) -> None:
        """Emit EVO_CONSOLIDATION_COMPLETE after a consolidation cycle."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EVO_CONSOLIDATION_COMPLETE,
                source_system="evo",
                data={
                    "consolidation_number": self._total_consolidations,
                    "duration_ms": result.duration_ms,
                    "hypotheses_integrated": result.hypotheses_integrated,
                    "schemas_induced": result.schemas_induced,
                    "parameters_adjusted": result.parameters_adjusted,
                },
            ))
        except Exception:
            self._logger.debug("consolidation_complete_emit_failed", exc_info=True)

    async def _emit_capability_emerged(
        self,
        capability_name: str,
        source_hypotheses: list[str],
        novelty_score: float,
        domain: str,
    ) -> None:
        """Emit EVO_CAPABILITY_EMERGED when a genuinely new capability is detected."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EVO_CAPABILITY_EMERGED,
                source_system="evo",
                data={
                    "capability_name": capability_name,
                    "source_hypotheses": source_hypotheses,
                    "novelty_score": novelty_score,
                    "domain": domain,
                },
            ))
        except Exception:
            self._logger.debug("capability_emerged_emit_failed", exc_info=True)

    async def _check_and_emit_speciation_event(self, result: ConsolidationResult) -> None:
        """
        After consolidation, check for significant behavioral shifts that
        indicate a speciation-level divergence. Emit SPECIATION_EVENT if detected.
        Max 1 event per 24 hours to avoid noise.
        """
        if self._event_bus is None:
            return

        now = time.monotonic()

        # Rate limit: max 1 per 24h
        if now - self._last_speciation_event_at < 86400:
            return

        # Track confirmation rate
        evaluated = result.hypotheses_evaluated
        integrated = result.hypotheses_integrated
        self._confirmation_history.append((now, integrated, max(1, evaluated)))

        # Need at least 7 days of history (rough: multiple consolidation cycles)
        if len(self._confirmation_history) < 10:
            return

        triggers: list[str] = []
        affected_domains: list[str] = []
        magnitude = 0.0

        # Check 1: Confirmation rate change > 20% over recent history
        half = len(self._confirmation_history) // 2
        older = list(self._confirmation_history)[:half]
        newer = list(self._confirmation_history)[half:]

        old_rate = sum(c for _, c, _ in older) / max(1, sum(t for _, _, t in older))
        new_rate = sum(c for _, c, _ in newer) / max(1, sum(t for _, _, t in newer))

        if abs(new_rate - old_rate) > 0.20:
            triggers.append(
                f"Confirmation rate shifted from {old_rate:.2f} to {new_rate:.2f}"
            )
            magnitude = max(magnitude, min(1.0, abs(new_rate - old_rate) / 0.5))

        # Check 2: Novel hypothesis domains appearing
        if self._hypothesis_engine is not None:
            current_domains = {
                h.category.value if hasattr(h.category, "value") else str(h.category)
                for h in self._hypothesis_engine.get_active()
            }
            novel_domains = current_domains - self._known_hypothesis_domains
            if novel_domains:
                triggers.append(f"Novel domains emerged: {', '.join(novel_domains)}")
                affected_domains.extend(novel_domains)
                magnitude = max(magnitude, min(1.0, len(novel_domains) * 0.3))
                self._known_hypothesis_domains.update(novel_domains)

        # Check 3: Foundation belief weight drift > 10% (from consolidation result)
        if result.foundation_conflicts > 0:
            triggers.append(
                f"{result.foundation_conflicts} foundation belief conflicts detected"
            )
            magnitude = max(magnitude, min(1.0, result.foundation_conflicts * 0.15))

        if not triggers or magnitude < 0.2:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.SPECIATION_EVENT,
                source_system="evo",
                data={
                    "trigger": "; ".join(triggers),
                    "magnitude": round(magnitude, 3),
                    "affected_domains": affected_domains,
                    "consolidation_number": self._total_consolidations,
                },
            ))
            self._last_speciation_event_at = now
            self._logger.info(
                "speciation_event_emitted",
                magnitude=round(magnitude, 3),
                triggers=triggers,
            )
        except Exception:
            self._logger.debug("speciation_event_emit_failed", exc_info=True)

    async def _emit_fitness_observable_batch(self, result: Any) -> None:
        """Loop 6: Emit FITNESS_OBSERVABLE_BATCH after consolidation for Benchmarks."""
        if self._event_bus is None:
            return
        try:
            from primitives.evolutionary import EvolutionaryObservable
            from systems.synapse.types import SynapseEvent, SynapseEventType

            instance_id = getattr(self, "_instance_id", "") or ""
            evaluated = getattr(result, "hypotheses_evaluated", 0)
            integrated = getattr(result, "hypotheses_integrated", 0)
            schemas = getattr(result, "schemas_induced", 0)
            archived = getattr(result, "hypotheses_archived", 0)

            # Build individual evolutionary observables from consolidation results
            observables: list[dict] = []
            if integrated > 0:
                observables.append(EvolutionaryObservable(
                    source_system=SystemID.EVO, instance_id=instance_id,
                    observable_type="hypotheses_integrated",
                    value=float(integrated), is_novel=True,
                    metadata={"consolidation": self._total_consolidations},
                ).model_dump(mode="json"))
            if schemas > 0:
                observables.append(EvolutionaryObservable(
                    source_system=SystemID.EVO, instance_id=instance_id,
                    observable_type="schemas_induced",
                    value=float(schemas), is_novel=True,
                    metadata={"consolidation": self._total_consolidations},
                ).model_dump(mode="json"))
            if archived > 0:
                observables.append(EvolutionaryObservable(
                    source_system=SystemID.EVO, instance_id=instance_id,
                    observable_type="hypotheses_archived",
                    value=float(archived), is_novel=False,
                    metadata={"consolidation": self._total_consolidations},
                ).model_dump(mode="json"))

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.FITNESS_OBSERVABLE_BATCH,
                source_system="evo",
                data={
                    "consolidation_number": self._total_consolidations,
                    "hypotheses_evaluated": evaluated,
                    "hypotheses_integrated": integrated,
                    "schemas_induced": schemas,
                    "duration_ms": getattr(result, "duration_ms", 0),
                    "observables": observables,
                    "instance_id": instance_id,
                    "generation": 1,
                },
            ))
            self._logger.debug(
                "fitness_observable_batch_emitted",
                consolidation=self._total_consolidations,
                observable_count=len(observables),
            )
        except Exception as exc:
            self._logger.debug("fitness_observable_batch_emit_failed", error=str(exc))

    def set_redis(self, redis: Any) -> None:
        """Wire the Redis client so PatternContext state is checkpointed between restarts.

        Without this, the accumulated detector counters (cooccurrences, sequences,
        temporal bins, affect responses) are lost on any restart that occurs between
        consolidation cycles — Spec §III gap fix.
        """
        self._redis = redis
        self._logger.info("redis_wired_to_evo_pattern_checkpoint")

    # Redis key for the PatternContext checkpoint
    _PATTERN_CONTEXT_REDIS_KEY = "evo:pattern_context:checkpoint"
    # TTL: slightly longer than the maximum consolidation interval (6h) × 2
    _PATTERN_CONTEXT_TTL_S: int = 14 * 3600  # 14 hours

    async def _save_pattern_context_checkpoint(self) -> None:
        """Checkpoint PatternContext counters to Redis (Spec §III gap fix).

        Called before each consolidation cycle so the post-consolidation reset
        can be safely applied — counters up to that point are durable.
        Also called periodically (every 1000 broadcasts) to limit loss window.
        Best-effort: failures are logged but never block the learning loop.
        """
        if self._redis is None:
            return
        try:
            import json as _json

            snapshot = {
                "cooccurrence_counts": dict(self._pattern_context.cooccurrence_counts),
                "sequence_counts": dict(self._pattern_context.sequence_counts),
                "sequence_examples": {
                    k: list(v)[:20]
                    for k, v in self._pattern_context.sequence_examples.items()
                },
                "temporal_bins": dict(self._pattern_context.temporal_bins),
                "temporal_baselines": dict(self._pattern_context.temporal_baselines),
                "affect_responses": {
                    k: [list(t) for t in v[-50:]]
                    for k, v in self._pattern_context.affect_responses.items()
                },
                "episodes_scanned": self._pattern_context.episodes_scanned,
            }
            raw = _json.dumps(snapshot, separators=(",", ":"))
            await self._redis.setex(
                self._PATTERN_CONTEXT_REDIS_KEY,
                self._PATTERN_CONTEXT_TTL_S,
                raw,
            )
        except Exception as exc:
            self._logger.debug("pattern_context_checkpoint_save_failed", error=str(exc))

    async def _restore_pattern_context_checkpoint(self) -> bool:
        """Restore PatternContext counters from Redis on startup.

        Returns True if a valid checkpoint was found and applied.
        Called from initialize() after sub-systems are built.
        """
        if self._redis is None:
            return False
        try:
            import json as _json
            from collections import defaultdict

            raw = await self._redis.get(self._PATTERN_CONTEXT_REDIS_KEY)
            if raw is None:
                return False
            snapshot = _json.loads(raw)

            cc: dict = snapshot.get("cooccurrence_counts", {})
            sc: dict = snapshot.get("sequence_counts", {})
            se: dict = snapshot.get("sequence_examples", {})
            tb: dict = snapshot.get("temporal_bins", {})
            tbl: dict = snapshot.get("temporal_baselines", {})
            ar: dict = snapshot.get("affect_responses", {})
            eps: int = int(snapshot.get("episodes_scanned", 0))

            ctx = self._pattern_context
            # Merge rather than replace — initialize() may have already run detectors
            for k, v in cc.items():
                ctx.cooccurrence_counts[k] += v
            for k, v in sc.items():
                ctx.sequence_counts[k] += v
            for k, v in se.items():
                existing = ctx.sequence_examples.setdefault(k, [])
                existing.extend(v)
                ctx.sequence_examples[k] = existing[:20]
            for k, v in tb.items():
                ctx.temporal_bins[k] += v
            ctx.temporal_baselines.update(tbl)
            for k, v in ar.items():
                existing = ctx.affect_responses.setdefault(k, [])
                existing.extend(tuple(t) for t in v)
                ctx.affect_responses[k] = existing[-50:]
            ctx.episodes_scanned = max(ctx.episodes_scanned, eps)

            self._logger.info(
                "pattern_context_checkpoint_restored",
                episodes_scanned=eps,
                cooccurrence_pairs=len(cc),
                sequences=len(sc),
            )
            return True
        except Exception as exc:
            self._logger.warning("pattern_context_checkpoint_restore_failed", error=str(exc))
            return False

    def set_thymos(self, thymos: Any) -> None:
        """Wire Thymos so Evo can escalate sustained degradation as incidents."""
        self._thymos = thymos
        self._logger.info("thymos_wired_to_evo")

    def set_simula(self, simula: Any) -> None:
        """
        Wire Simula so arXiv proposals can be dispatched for governance review,
        and consolidation Phase 8 can submit EFE-scored evolution proposals.

        Each proposal submitted through the callback flows into
        SimulaService.receive_evo_proposal() → process_proposal(),
        where the ArchitectureEFEScorer attaches an EFE score.
        """
        self._simula = simula
        # Wire the consolidation orchestrator's Simula callback
        if self._orchestrator is not None:
            # Adapter: translate Evo's (proposal, hypotheses) call signature
            # into SimulaService.receive_evo_proposal() individual args
            async def _submit_to_simula(proposal: Any, hypotheses: list[Any]) -> Any:
                # Extract mutation target/type from the first hypothesis that has one,
                # so the bridge can use rule-based category inference (zero LLM tokens)
                # instead of always falling through to the LLM fallback.
                mutation_target = ""
                mutation_type = ""
                for h in hypotheses:
                    mut = getattr(h, "proposed_mutation", None)
                    if mut is not None:
                        mutation_target = getattr(mut, "target", "") or ""
                        mt = getattr(mut, "type", "")
                        mutation_type = mt.value if hasattr(mt, "value") else str(mt)
                        break
                return await simula.receive_evo_proposal(
                    evo_description=proposal.description,
                    evo_rationale=proposal.rationale,
                    hypothesis_ids=[h.id for h in hypotheses],
                    hypothesis_statements=[h.statement for h in hypotheses],
                    evidence_scores=[h.evidence_score for h in hypotheses],
                    supporting_episode_ids=[
                        ep
                        for h in hypotheses
                        for ep in h.supporting_episodes[:5]
                    ],
                    mutation_target=mutation_target,
                    mutation_type=mutation_type,
                )

            self._orchestrator._simula_callback = _submit_to_simula
        self._logger.info("simula_wired_to_evo")

    @property
    def tournament_engine(self) -> TournamentEngine | None:
        """Expose the tournament engine for Nova integration."""
        return self._tournament_engine

    def record_tournament_outcome(
        self,
        tournament_id: str,
        hypothesis_id: str,
        success: bool,
        intent_id: str = "",
    ) -> None:
        """
        Record a tournament trial outcome from Axon's execution result.
        Called by the outcome feedback loop when an intent linked to a
        tournament context completes.
        """
        if self._tournament_engine is None:
            return
        outcome = TournamentOutcome(
            tournament_id=tournament_id,
            hypothesis_id=hypothesis_id,
            success=success,
            intent_id=intent_id,
        )
        self._tournament_engine.record_outcome(outcome)

    # ─── Action Outcome Feedback ──────────────────────────────────────────────

    def register_on_synapse(self, event_bus: Any) -> None:
        """
        Register Evo's Synapse event subscribers.

        Subscribes to:
          - ACTION_COMPLETED: outcome feedback for hypothesis confidence
          - FOVEA_INTERNAL_PREDICTION_ERROR: competency errors → self-model updates
          - KAIROS_CAUSAL_DIRECTION_ACCEPTED: causal validation results for hypotheses
          - GENOME_INHERITED: Telos drive mutations in child instances → hypothesized adaptations

        Call this from main.py during the integration wiring pass:
            evo.register_on_synapse(synapse.event_bus)
        """
        from systems.synapse.types import SynapseEventType
        event_bus.subscribe(SynapseEventType.ACTION_COMPLETED, self.on_action_completed)

        # Fovea competency-type internal prediction errors → self-model evidence
        if hasattr(SynapseEventType, "FOVEA_INTERNAL_PREDICTION_ERROR"):
            event_bus.subscribe(
                SynapseEventType.FOVEA_INTERNAL_PREDICTION_ERROR,
                self._on_fovea_internal_prediction_error,
            )

        # Kairos causal direction results → update causal hypotheses with validation
        if hasattr(SynapseEventType, "KAIROS_CAUSAL_DIRECTION_ACCEPTED"):
            event_bus.subscribe(
                SynapseEventType.KAIROS_CAUSAL_DIRECTION_ACCEPTED,
                self._on_kairos_causal_direction_accepted,
            )

        # Thymos successful repairs → extract fix pattern as procedural hypothesis
        if hasattr(SynapseEventType, "REPAIR_COMPLETED"):
            event_bus.subscribe(
                SynapseEventType.REPAIR_COMPLETED,
                self._on_repair_completed,
            )

        # Economic events → episodic memory + pattern detection
        event_bus.subscribe(
            SynapseEventType.BOUNTY_SOLUTION_PENDING,
            self._on_bounty_solution_pending,
        )
        event_bus.subscribe(
            SynapseEventType.BOUNTY_PR_SUBMITTED,
            self._on_bounty_pr_submitted,
        )
        event_bus.subscribe(
            SynapseEventType.REVENUE_INJECTED,
            self._on_revenue_injected,
        )
        event_bus.subscribe(
            SynapseEventType.METABOLIC_PRESSURE,
            self._on_metabolic_pressure,
        )
        if hasattr(SynapseEventType, "COGNITIVE_PRESSURE"):
            event_bus.subscribe(
                SynapseEventType.COGNITIVE_PRESSURE,
                self._on_cognitive_pressure,
            )
        if hasattr(SynapseEventType, "BUDGET_EXHAUSTED"):
            event_bus.subscribe(
                SynapseEventType.BUDGET_EXHAUSTED,
                self._on_budget_exhausted,
            )

        # SG5: Economic outcome events → close the learning loop on economic strategy
        event_bus.subscribe(
            SynapseEventType.BOUNTY_PAID,
            self._on_bounty_paid,
        )
        if hasattr(SynapseEventType, "ASSET_BREAK_EVEN"):
            event_bus.subscribe(
                SynapseEventType.ASSET_BREAK_EVEN,
                self._on_asset_break_even,
            )
        event_bus.subscribe(
            SynapseEventType.CHILD_INDEPENDENT,
            self._on_child_independent,
        )

        # Simula evolution outcomes → reward/penalise source hypotheses
        if hasattr(SynapseEventType, "EVOLUTION_APPLIED"):
            event_bus.subscribe(
                SynapseEventType.EVOLUTION_APPLIED,
                self._on_evolution_applied,
            )
        if hasattr(SynapseEventType, "EVOLUTION_ROLLED_BACK"):
            event_bus.subscribe(
                SynapseEventType.EVOLUTION_ROLLED_BACK,
                self._on_evolution_rolled_back,
            )

        # ── Speciation event subscriptions ────────────────────────────────

        # Fovea high prediction errors → structural hypothesis generation
        if hasattr(SynapseEventType, "FOVEA_PREDICTION_ERROR"):
            event_bus.subscribe(
                SynapseEventType.FOVEA_PREDICTION_ERROR,
                self._on_fovea_prediction_error_high,
            )

        # Oneiros dream results → update hypothesis evidence
        if hasattr(SynapseEventType, "DREAM_HYPOTHESES_GENERATED"):
            event_bus.subscribe(
                SynapseEventType.DREAM_HYPOTHESES_GENERATED,
                self._on_dream_hypotheses_generated,
            )

        # Equor DENY → learn constitutional boundaries
        if hasattr(SynapseEventType, "INTENT_REJECTED"):
            event_bus.subscribe(
                SynapseEventType.INTENT_REJECTED,
                self._on_intent_rejected,
            )

        # Kairos Tier 3 invariant → pre-validated hypothesis
        if hasattr(SynapseEventType, "KAIROS_TIER3_INVARIANT_DISCOVERED"):
            event_bus.subscribe(
                SynapseEventType.KAIROS_TIER3_INVARIANT_DISCOVERED,
                self._on_kairos_tier3_invariant,
            )

        # Oikos metabolic pressure → inject economic hypothesis into tournament
        if hasattr(SynapseEventType, "METABOLIC_EFFICIENCY_PRESSURE"):
            event_bus.subscribe(
                SynapseEventType.METABOLIC_EFFICIENCY_PRESSURE,
                self._on_metabolic_efficiency_pressure,
            )

        # Telos drive mutations inherited by child instances → track as hypothesized adaptations
        if hasattr(SynapseEventType, "GENOME_INHERITED"):
            event_bus.subscribe(
                SynapseEventType.GENOME_INHERITED,
                self._on_genome_inherited,
            )

        # Soma somatic signal → cache curiosity drive (replaces direct soma.get_current_signal())
        if hasattr(SynapseEventType, "SOMATIC_MODULATION_SIGNAL"):
            event_bus.subscribe(
                SynapseEventType.SOMATIC_MODULATION_SIGNAL,
                self._on_somatic_modulation_signal,
            )

        # Degradation Engine §8.2 — stub subscription.
        # Round 2 will implement: decay confidence on all unvalidated hypotheses
        # by staleness_rate, then re-emit HYPOTHESIS_STALENESS_APPLIED so
        # VitalityCoordinator can call on_hypotheses_revalidated().
        if hasattr(SynapseEventType, "HYPOTHESIS_STALENESS"):
            event_bus.subscribe(
                SynapseEventType.HYPOTHESIS_STALENESS,
                self._on_hypothesis_staleness,
            )

        # RE performance degradation → hyperparameter adjustment hypothesis
        if hasattr(SynapseEventType, "RE_DECISION_OUTCOME"):
            event_bus.subscribe(
                SynapseEventType.RE_DECISION_OUTCOME,
                self._on_re_decision_outcome,
            )

        # Exploration outcome feedback from Simula (Phase 8.5 gap closure)
        if hasattr(SynapseEventType, "EXPLORATION_OUTCOME"):
            event_bus.subscribe(
                SynapseEventType.EXPLORATION_OUTCOME,
                self._on_exploration_outcome,
            )

        # Nova input channel discovery → domain-specific hypothesis candidates
        if hasattr(SynapseEventType, "INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED"):
            event_bus.subscribe(
                SynapseEventType.INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED,
                self._on_opportunities_discovered,
            )

        self._logger.info("evo_synapse_subscriptions_registered")

    async def on_action_completed(self, event: Any) -> None:
        """
        Subscriber for ACTION_COMPLETED events published by Axon.

        Expected event.data keys:
          intent_id      (str)   — ID of the completed intent
          outcome        (str)   — Short description of what happened
          success        (bool)  — Whether the intent succeeded
          economic_delta (float) — Revenue/cost impact in USD (signed)

        Updates hypothesis confidence for all TESTING/PROPOSED hypotheses
        whose action-sequence pattern matches the completed intent.
        Tracks volatility: repeated confidence oscillations flag the
        hypothesis as HIGH_VOLATILITY and reduce its effective weight.

        Also forwards the (intent, outcome) pair to ProcedureCodifier so
        recurring successful sequences can be codified during consolidation.

        Does NOT raise — outcome feedback must not interrupt the event bus.
        """
        if not self._initialized:
            return

        try:
            data = getattr(event, "data", {}) or {}
            intent_id: str = str(data.get("intent_id", ""))
            outcome_text: str = str(data.get("outcome", ""))
            success: bool = bool(data.get("success", False))
            economic_delta: float = float(data.get("economic_delta", 0.0))

            # Forward to ProcedureCodifier — no action_types here so we use
            # a minimal IntentRecord; the episode_id will be empty unless Axon
            # included it. Full linkage happens when Axon also populates episode_id.
            episode_id: str = str(data.get("episode_id", ""))
            action_types_raw = data.get("action_types") or []
            action_types = tuple(str(a) for a in action_types_raw)

            if self._procedure_codifier is not None and action_types:
                intent_rec = IntentRecord(
                    intent_id=intent_id,
                    goal_description=str(data.get("goal_description", "")),
                    action_types=action_types,
                    episode_id=episode_id,
                )
                outcome_rec = OutcomeRecord(
                    intent_id=intent_id,
                    success=success,
                    economic_delta=economic_delta,
                    outcome_summary=outcome_text,
                )
                self._procedure_codifier.observe(intent_rec, outcome_rec)

            # Update hypothesis confidence for active hypotheses that relate
            # to procedural/self-model claims.  We apply a lightweight
            # heuristic: success nudges PROCEDURAL hypotheses up, failure
            # nudges them down, proportional to |economic_delta| when non-zero.
            if self._hypothesis_engine is not None:
                active = self._hypothesis_engine.get_active()
                for h in active:
                    await self._apply_outcome_to_hypothesis(
                        h, success=success, economic_delta=economic_delta, intent_id=intent_id
                    )

            self._logger.debug(
                "action_completed_processed",
                intent_id=intent_id,
                success=success,
                economic_delta=round(economic_delta, 4),
                active_hypotheses=len(
                    self._hypothesis_engine.get_active()
                    if self._hypothesis_engine else []
                ),
            )

        except Exception as exc:
            self._logger.error("on_action_completed_failed", error=str(exc))

    async def _apply_outcome_to_hypothesis(
        self,
        hypothesis: Hypothesis,
        success: bool,
        economic_delta: float,
        intent_id: str,
    ) -> None:
        """
        Nudge hypothesis confidence based on a real action outcome.

        Only PROCEDURAL and SELF_MODEL hypotheses are updated this way —
        WORLD_MODEL and SOCIAL hypotheses require richer episodic evidence
        (evaluated via evaluate_evidence during the evidence sweep).

        Confidence delta is small (max 0.2) so this never overrides the
        Bayesian evidence accumulation — it just reinforces or weakens.

        Volatility tracking: if the direction of the nudge reverses compared
        to the previous nudge, we count an oscillation. After
        _VOLATILITY_OSCILLATION_THRESHOLD reversals, the hypothesis is
        flagged as HIGH_VOLATILITY and its effective weight is halved.
        """
        from systems.evo.types import HypothesisCategory

        if hypothesis.category not in (
            HypothesisCategory.PROCEDURAL,
            HypothesisCategory.SELF_MODEL,
        ):
            return

        # Compute nudge magnitude: base 0.05, boosted by |economic_delta| (cap 0.15 extra)
        magnitude = min(0.05 + abs(economic_delta) * 0.02, 0.20)
        direction = 1 if success else -1
        delta = direction * magnitude

        hypothesis.evidence_score += delta * hypothesis.volatility_weight

        # Status transitions after nudge (same thresholds as Bayesian path)
        from systems.evo.types import (
            VELOCITY_LIMITS,
            HypothesisStatus,
        )
        if hypothesis.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING):
            hypothesis.status = HypothesisStatus.TESTING
            if (
                hypothesis.evidence_score > 3.0
                and len(hypothesis.supporting_episodes) >= VELOCITY_LIMITS["min_evidence_for_integration"]
            ):
                hypothesis.status = HypothesisStatus.SUPPORTED
                await self._emit_evolutionary_observable(
                    observable_type="hypothesis_validated",
                    value=hypothesis.evidence_score,
                    is_novel=True,
                    metadata={
                        "hypothesis_id": hypothesis.id,
                        "statement": hypothesis.statement[:200],
                        "category": hypothesis.category.value if hasattr(hypothesis.category, "value") else str(hypothesis.category),
                        "supporting_count": len(hypothesis.supporting_episodes),
                    },
                )
            elif hypothesis.evidence_score < -2.0:
                hypothesis.status = HypothesisStatus.REFUTED
                await self._emit_evolutionary_observable(
                    observable_type="hypothesis_rejected",
                    value=hypothesis.evidence_score,
                    is_novel=True,
                    metadata={
                        "hypothesis_id": hypothesis.id,
                        "statement": hypothesis.statement[:200],
                        "category": hypothesis.category.value if hasattr(hypothesis.category, "value") else str(hypothesis.category),
                    },
                )

        # ── Volatility tracking ──────────────────────────────────────────────
        flip_log: list[int] = hypothesis.confidence_flip_log
        flip_log.append(direction)
        # Keep only the last 10 nudges
        if len(flip_log) > 10:
            flip_log[:] = flip_log[-10:]

        # Count sign reversals in the window
        reversals = sum(
            1 for i in range(1, len(flip_log))
            if flip_log[i] != flip_log[i - 1]
        )
        hypothesis.confidence_oscillations = reversals

        if reversals >= _VOLATILITY_OSCILLATION_THRESHOLD:
            if hypothesis.volatility_flag != "HIGH_VOLATILITY":
                hypothesis.volatility_flag = "HIGH_VOLATILITY"
                hypothesis.volatility_weight = 0.5
                self._logger.warning(
                    "hypothesis_high_volatility_flagged",
                    hypothesis_id=hypothesis.id,
                    oscillations=reversals,
                    evidence_score=round(hypothesis.evidence_score, 3),
                )
        elif hypothesis.volatility_flag == "HIGH_VOLATILITY" and reversals < _VOLATILITY_OSCILLATION_THRESHOLD // 2:
            # Enough stability to de-escalate
            hypothesis.volatility_flag = "normal"
            hypothesis.volatility_weight = 1.0

    # ─── Fovea Internal Prediction Error Handler ────────────────────────────────

    async def _on_fovea_internal_prediction_error(self, event: Any) -> None:
        """
        Handle FOVEA_INTERNAL_PREDICTION_ERROR events.

        Competency-type errors indicate the self-model predicted a capability
        the organism doesn't actually have (or vice versa). These are direct
        evidence for SELF_MODEL hypotheses — much stronger than action outcomes
        because Fovea has already done the prediction/observation comparison
        with 3x precision multiplier.

        Also feeds SelfModelManager with competency calibration data.
        """
        if not self._initialized or self._hypothesis_engine is None:
            return

        try:
            data = getattr(event, "data", {}) or {}
            error_type = data.get("internal_error_type", "")
            if error_type != "competency":
                return

            salience = float(data.get("precision_weighted_salience", 0.0))

            # Apply as evidence to SELF_MODEL hypotheses — competency errors
            # are strong evidence because Fovea's internal precision multiplier
            # (3x) means these are high-confidence signals
            from systems.evo.types import HypothesisCategory

            active = self._hypothesis_engine.get_active()
            for h in active:
                if h.category != HypothesisCategory.SELF_MODEL:
                    continue
                if h.status not in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING):
                    continue

                # Salience-weighted evidence: high salience = big self-model mismatch
                direction = -1  # Self-model was wrong → contradicts current self-model claims
                magnitude = min(salience * 0.3, 0.25)
                h.evidence_score += direction * magnitude * h.volatility_weight
                h.status = HypothesisStatus.TESTING

            self._logger.debug(
                "fovea_competency_error_processed",
                salience=round(salience, 3),
                self_model_hypotheses=sum(
                    1 for h in active if h.category == HypothesisCategory.SELF_MODEL
                ),
            )

        except Exception as exc:
            self._logger.warning("fovea_ipe_handler_failed", error=str(exc))

    # ─── Kairos Causal Direction Handler ─────────────────────────────────────

    async def _on_kairos_causal_direction_accepted(self, event: Any) -> None:
        """
        Handle KAIROS_CAUSAL_DIRECTION_ACCEPTED events.

        When Kairos validates a causal direction, find matching WORLD_MODEL
        hypotheses and boost their evidence score — Kairos has independently
        confirmed the causal claim through its multi-method pipeline.
        """
        if not self._initialized or self._hypothesis_engine is None:
            return

        try:
            data = getattr(event, "data", {}) or {}
            cause = data.get("cause", "")
            effect = data.get("effect", "")
            confidence = float(data.get("confidence", 0.0))

            if confidence < 0.3:
                return

            from systems.evo.types import HypothesisCategory

            active = self._hypothesis_engine.get_active()
            matched = 0
            for h in active:
                if h.category != HypothesisCategory.WORLD_MODEL:
                    continue
                # Check if hypothesis statement contains the validated causal pair
                stmt_lower = h.statement.lower()
                cause_match = cause.lower() in stmt_lower if cause else False
                effect_match = effect.lower() in stmt_lower if effect else False
                if cause_match or effect_match:
                    # Kairos-validated causal evidence — strong support
                    boost = confidence * 0.5  # Scale by Kairos confidence
                    h.evidence_score += boost * h.volatility_weight
                    matched += 1
                    self._logger.debug(
                        "kairos_validation_applied",
                        hypothesis_id=h.id,
                        boost=round(boost, 3),
                        kairos_confidence=round(confidence, 3),
                    )

            if matched:
                self._logger.info(
                    "kairos_causal_validation_matched",
                    cause=cause[:40],
                    effect=effect[:40],
                    hypotheses_boosted=matched,
                )

        except Exception as exc:
            self._logger.warning("kairos_direction_handler_failed", error=str(exc))

    async def _on_repair_completed(self, event: Any) -> None:
        """
        Handle REPAIR_COMPLETED events broadcast by Thymos.

        Extracts the repair pattern (what failed, on which endpoint, and how it
        was fixed) and registers a procedural hypothesis so future proposals
        touching the same endpoint can be validated against known failure modes.

        On duplicate events (same endpoint+fix_type already has an active
        hypothesis) the new repair episode is fed to evaluate_evidence rather
        than discarded — each successful repair is an independent data point.

        Only Tier 2+ (PARAMETER and above) repairs produce learnable patterns;
        NOOP repairs are filtered out by Thymos before broadcast.

        Does NOT raise — repair learning must not interrupt the event bus.
        """
        if not self._initialized or self._hypothesis_engine is None:
            return

        try:
            data = getattr(event, "data", {}) or {}
            incident_id: str = data.get("incident_id", "")
            endpoint: str = data.get("endpoint", "")
            tier: str = data.get("tier", "")
            incident_class: str = data.get("incident_class", "")
            fix_type: str = data.get("fix_type", "")
            root_cause: str = data.get("root_cause", "unknown")
            # repair_spec_id is set by Thymos on RepairSpec; fall back to computing it.
            repair_spec_id: str = data.get("repair_spec_id", "")
            if not repair_spec_id and incident_class and fix_type:
                from systems.thymos.types import RepairSpec as _RepairSpec
                repair_spec_id = _RepairSpec.make_id(incident_class, fix_type)

            if not fix_type:
                return

            # Build statement: "endpoint X failed because Y, fixed by Z"
            subject = endpoint if endpoint else incident_class or "system"
            statement = (
                f"When {subject} encounters a {incident_class or 'failure'}, "
                f"applying '{fix_type}' resolves it "
                f"(root cause: {root_cause[:120] if root_cause else 'unknown'})."
            )
            formal_test = (
                f"Future incidents on '{subject}' should be resolved by '{fix_type}'; "
                f"if the antibody succeeds in ≥3 subsequent applications the pattern holds."
            )

            # Register as a procedural hypothesis directly — no LLM needed since
            # the pattern is derived from an observed, successful repair.
            result = self._hypothesis_engine.register_repair_hypothesis(
                statement=statement,
                formal_test=formal_test,
                endpoint=endpoint,
                fix_type=fix_type,
                incident_class=incident_class,
                source_episode_id=incident_id,
            )

            if result is None:
                # At capacity — skip silently.
                return

            h, is_new = result

            if is_new:
                self._total_hypotheses_proposed = getattr(
                    self, "_total_hypotheses_proposed", 0
                ) + 1
                self._logger.info(
                    "repair_pattern_learned",
                    hypothesis_id=h.id,
                    endpoint=endpoint or "(none)",
                    tier=tier,
                    fix_type=fix_type,
                    repair_spec_id=repair_spec_id,
                )
            else:
                # Duplicate event — feed the new repair as evidence to the
                # existing hypothesis rather than silently dropping it.
                repair_episode = Episode(
                    id=incident_id or f"repair_evt_{fix_type}",
                    source="thymos.repair",
                    raw_content=(
                        f"Repair '{fix_type}' applied successfully on '{subject}'. "
                        f"Incident class: {incident_class}. Root cause: {root_cause[:200]}."
                    ),
                    summary=f"Successful repair: {fix_type} on {subject}",
                    salience_composite=0.8,   # Repairs are high-salience events
                    affect_valence=0.4,        # Positive: problem resolved
                    affect_arousal=0.3,
                )
                await self._hypothesis_engine.evaluate_evidence(h, repair_episode)
                self._logger.debug(
                    "repair_hypothesis_evidence_added",
                    hypothesis_id=h.id,
                    endpoint=endpoint or "(none)",
                    fix_type=fix_type,
                    new_score=round(h.evidence_score, 2),
                )

            # Also queue an ACTION_SEQUENCE pattern candidate so the normal
            # hypothesis-generation pipeline sees this repair as evidence.
            self._pending_candidates.append(
                PatternCandidate(
                    type=PatternType.ACTION_SEQUENCE,
                    elements=[incident_class or "unknown", fix_type],
                    count=1,
                    confidence=0.8,
                    examples=[incident_id] if incident_id else [],
                    metadata={
                        "source": "thymos.repair",
                        "endpoint": endpoint,
                        "tier": tier,
                        "root_cause": root_cause[:200],
                        "repair_spec_id": repair_spec_id,
                    },
                )
            )

            # Emit hypothesis quality back to Thymos so the immune system
            # knows whether repair patterns are generalising or staying narrow.
            await self._emit_hypothesis_quality(
                hypothesis=h,
                repair_source_id=incident_id,
            )

        except Exception as exc:
            self._logger.warning("repair_completed_handler_failed", error=str(exc))

    async def _emit_hypothesis_quality(
        self,
        hypothesis: Any,
        repair_source_id: str,
    ) -> None:
        """
        Emit EVO_HYPOTHESIS_QUALITY so Thymos knows if repair patterns generalise.

        Quality score is derived from:
        - evidence_score: how much confirming evidence has accumulated
        - applications: number of distinct episodes that supported this hypothesis
        - confidence: Thompson posterior from evaluate_evidence
        """
        if self._event_bus is None:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Normalise evidence score to [0, 1] — evidence_score is unbounded
            # but typically ranges 0-10 for repair hypotheses
            raw_score = getattr(hypothesis, "evidence_score", 0.0)
            quality = min(1.0, max(0.0, raw_score / 8.0))

            # Count distinct supporting episodes
            supporting = getattr(hypothesis, "supporting_episodes", []) or []
            applications = len(supporting)

            # Confidence from Thompson sampling (normalised 0-1)
            confidence = min(1.0, max(0.0, raw_score / 10.0))

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EVO_HYPOTHESIS_QUALITY,
                source_system="evo",
                data={
                    "hypothesis_id": hypothesis.id,
                    "repair_source_id": repair_source_id,
                    "quality_score": round(quality, 3),
                    "applications": applications,
                    "confidence": round(confidence, 3),
                },
            ))
        except Exception as exc:
            self._logger.debug("hypothesis_quality_emit_failed", error=str(exc))

    # ─── Speciation Event Handlers ───────────────────────────────────────────────

    async def _on_fovea_prediction_error_high(self, event: Any) -> None:
        """
        Handle FOVEA_PREDICTION_ERROR events for structural hypothesis generation.

        When Fovea reports a significant prediction error, the structural
        hypothesis generator creates domain-specific hypotheses about why
        the world model is wrong — no LLM cost.
        """
        if not self._initialized or self._structural_generator is None:
            return
        if self._hypothesis_engine is None:
            return

        try:
            data = getattr(event, "data", {}) or {}
            salience = float(data.get("precision_weighted_salience", 0.0))
            if salience < 0.3:
                return  # Only react to significant errors

            domain = str(data.get("domain", data.get("source", "unknown")))
            predicted = float(data.get("predicted", data.get("expected", 0.5)))
            actual = float(data.get("actual", data.get("observed", 0.5)))
            context_desc = str(data.get("context", data.get("description", "")))

            existing_ids = {h.id for h in self._hypothesis_engine.get_all_active()}
            hypotheses = self._structural_generator.generate_from_prediction_error(
                domain=domain,
                predicted_value=predicted,
                actual_value=actual,
                context_description=context_desc,
                existing_ids=existing_ids,
            )

            # Register generated hypotheses
            for h in hypotheses:
                self._hypothesis_engine._active[h.id] = h
                self._hypothesis_engine._total_proposed += 1
                if self._meta_learning is not None:
                    self._meta_learning.record_hypothesis_generated(
                        h.id, "structural_prediction_error",
                    )

            if hypotheses:
                self._logger.info(
                    "structural_hypotheses_from_fovea",
                    count=len(hypotheses),
                    domain=domain,
                    salience=round(salience, 3),
                )

        except Exception as exc:
            self._logger.warning("fovea_prediction_error_handler_failed", error=str(exc))

    async def _on_evolution_applied(self, event: Any) -> None:
        """Handle EVOLUTION_APPLIED — reward source hypotheses whose proposals succeeded.

        When Simula successfully applies a structural change that originated from
        Evo hypotheses, we boost those hypotheses' evidence scores. This closes the
        Evo→Simula→Evo learning loop: observe → hypothesise → evolve → learn from outcome.
        """
        if not self._initialized or self._hypothesis_engine is None:
            return
        try:
            data = getattr(event, "data", {}) or {}
            hypothesis_ids: list[str] = data.get("hypothesis_ids", [])
            if not hypothesis_ids:
                return

            category = data.get("category", "")
            proposal_id = data.get("proposal_id", "")

            for h_id in hypothesis_ids:
                h = self._hypothesis_engine.get_hypothesis(h_id)
                if h is None:
                    continue
                # Reward: successful evolution is strong supporting evidence
                boost = 1.5  # Strong positive signal
                h.evidence_score += boost
                h.supporting_episodes.append(f"evolution_applied:{proposal_id}")
                if len(h.supporting_episodes) > 50:
                    h.supporting_episodes = h.supporting_episodes[-50:]

                self._logger.info(
                    "evolution_applied_hypothesis_rewarded",
                    hypothesis_id=h_id,
                    boost=boost,
                    new_score=round(h.evidence_score, 3),
                    category=category,
                )
        except Exception as exc:
            self._logger.warning("evolution_applied_handler_failed", error=str(exc))

    async def _on_evolution_rolled_back(self, event: Any) -> None:
        """Handle EVOLUTION_ROLLED_BACK — penalise source hypotheses whose proposals failed.

        When Simula rolls back a structural change, the originating hypotheses
        receive negative evidence. Recurring rollbacks from the same hypothesis
        will drive it below the refutation threshold.
        """
        if not self._initialized or self._hypothesis_engine is None:
            return
        try:
            data = getattr(event, "data", {}) or {}
            hypothesis_ids: list[str] = data.get("hypothesis_ids", [])
            if not hypothesis_ids:
                return

            rollback_reason = data.get("rollback_reason", "")
            proposal_id = data.get("proposal_id", "")

            for h_id in hypothesis_ids:
                h = self._hypothesis_engine.get_hypothesis(h_id)
                if h is None:
                    continue
                # Penalty: rollback is moderately strong contradicting evidence
                penalty = -0.8
                h.evidence_score += penalty
                h.contradicting_episodes.append(f"evolution_rolled_back:{proposal_id}")
                if len(h.contradicting_episodes) > 50:
                    h.contradicting_episodes = h.contradicting_episodes[-50:]

                # Check if hypothesis should be refuted
                if h.evidence_score < -2.0 and h.status == HypothesisStatus.TESTING:
                    h.status = HypothesisStatus.REFUTED
                    self._logger.info(
                        "evolution_rollback_hypothesis_refuted",
                        hypothesis_id=h_id,
                        score=round(h.evidence_score, 3),
                    )
                    await self._emit_evolutionary_observable(
                        observable_type="hypothesis_rejected",
                        value=h.evidence_score,
                        is_novel=True,
                        metadata={
                            "hypothesis_id": h_id,
                            "reason": "evolution_rollback",
                            "rollback_proposal_id": proposal_id,
                        },
                    )
                else:
                    self._logger.info(
                        "evolution_rollback_hypothesis_penalised",
                        hypothesis_id=h_id,
                        penalty=penalty,
                        new_score=round(h.evidence_score, 3),
                        rollback_reason=rollback_reason[:80],
                    )
        except Exception as exc:
            self._logger.warning("evolution_rolled_back_handler_failed", error=str(exc))

    async def _on_dream_hypotheses_generated(self, event: Any) -> None:
        """
        Handle DREAM_HYPOTHESES_GENERATED events from Oneiros.

        Dream-tested hypotheses provide evidence: if a hypothesis survived
        dream stress-testing, boost its evidence score. If it failed in the
        dream, apply contradicting evidence.
        """
        if not self._initialized or self._hypothesis_engine is None:
            return

        try:
            data = getattr(event, "data", {}) or {}
            dream_results = data.get("results", data.get("hypotheses", []))

            for result in dream_results:
                h_id = str(result.get("hypothesis_id", ""))
                if not h_id:
                    continue

                # Find the hypothesis
                h = self._hypothesis_engine._active.get(h_id)
                if h is None:
                    continue

                survived = bool(result.get("survived", result.get("supported", False)))
                confidence = float(result.get("confidence", result.get("strength", 0.3)))

                # Dream evidence is moderate strength (dreams are simulations)
                magnitude = min(confidence * 0.2, 0.15)
                if survived:
                    h.evidence_score += magnitude * h.volatility_weight
                else:
                    h.evidence_score -= magnitude * h.volatility_weight

            dream_count = len(dream_results)
            if dream_count > 0:
                self._logger.info(
                    "dream_evidence_applied",
                    hypotheses_updated=dream_count,
                )

        except Exception as exc:
            self._logger.warning("dream_hypotheses_handler_failed", error=str(exc))

    async def _on_intent_rejected(self, event: Any) -> None:
        """
        Handle INTENT_REJECTED (Equor DENY) events.

        When Equor denies an intent, learn about constitutional boundaries.
        Generate a WORLD_MODEL hypothesis about the boundary condition so
        future policy selection avoids it.
        """
        if not self._initialized or self._hypothesis_engine is None:
            return

        try:
            data = getattr(event, "data", {}) or {}
            reason = str(data.get("reason", data.get("denial_reason", "")))
            drive_scores = data.get("drive_scores", {})
            intent_desc = str(data.get("description", data.get("intent_description", "")))

            if not reason:
                return

            from systems.evo.types import HypothesisCategory

            # Find which drive caused the denial
            violated_drive = ""
            if drive_scores:
                violated_drive = min(drive_scores, key=lambda d: drive_scores.get(d, 0.0))

            h_id = f"constitutional_boundary_{hash(reason) % 100000}"
            existing_ids = {h.id for h in self._hypothesis_engine.get_all_active()}
            if h_id in existing_ids:
                return

            h = Hypothesis(
                id=h_id,
                category=HypothesisCategory.WORLD_MODEL,
                statement=(
                    f"Constitutional boundary: intents involving '{intent_desc[:100]}' "
                    f"are denied by Equor due to {violated_drive or 'drive'} violation. "
                    f"Reason: {reason[:150]}"
                ),
                formal_test=(
                    "Future intents with similar characteristics should be flagged "
                    "as constitutionally risky before submission to Equor"
                ),
                complexity_penalty=0.05,
                status=HypothesisStatus.TESTING,
                evidence_score=0.5,  # One denial = moderate evidence
            )
            self._hypothesis_engine._active[h.id] = h
            self._hypothesis_engine._total_proposed += 1

            self._logger.info(
                "constitutional_boundary_hypothesis",
                hypothesis_id=h.id,
                violated_drive=violated_drive,
                reason=reason[:80],
            )

        except Exception as exc:
            self._logger.warning("intent_rejected_handler_failed", error=str(exc))

    async def _on_kairos_tier3_invariant(self, event: Any) -> None:
        """
        Handle KAIROS_TIER3_INVARIANT_DISCOVERED: create a pre-validated hypothesis.

        Tier 3 invariants are substrate-independent causal rules that have been
        rigorously validated. Evo creates a hypothesis marked as pre-validated
        (skip initial testing, go straight to SUPPORTED with high evidence).
        """
        if not self._initialized or self._hypothesis_engine is None:
            return

        try:
            data = getattr(event, "data", {}) or {}
            invariant_id = str(data.get("invariant_id", ""))
            abstract_form = str(data.get("abstract_form", data.get("description", "")))
            domains = data.get("applicable_domains", [])

            if not abstract_form:
                return

            # Check for duplicate
            existing_ids = {h.id for h in self._hypothesis_engine.get_all_active()}
            h_id = f"kairos_invariant_{invariant_id}" if invariant_id else f"kairos_{hash(abstract_form) % 100000}"
            if h_id in existing_ids:
                return

            from systems.evo.types import HypothesisCategory

            h = Hypothesis(
                id=h_id,
                category=HypothesisCategory.WORLD_MODEL,
                statement=f"Causal invariant (Kairos Tier 3): {abstract_form[:300]}",
                formal_test=(
                    "Invariant holds across all observed domains. "
                    "Refuted if a domain is found where it systematically fails."
                ),
                complexity_penalty=0.05,
                status=HypothesisStatus.SUPPORTED,  # Pre-validated by Kairos
                evidence_score=5.0,  # High evidence — already validated
                min_age_hours=0.0,  # No age gate — already proven
                novelty_score=1.0,  # Tier 3 invariants are inherently novel
            )
            self._hypothesis_engine._active[h.id] = h
            self._hypothesis_engine._total_proposed += 1
            self._hypothesis_engine._total_supported += 1

            self._logger.info(
                "kairos_invariant_hypothesis_created",
                hypothesis_id=h.id,
                invariant_id=invariant_id,
                domains=domains[:5],
            )

            # Emit EVO_HYPOTHESIS_CONFIRMED since it's pre-validated
            await self._emit_hypothesis_lifecycle_events([h], "confirmed")

        except Exception as exc:
            self._logger.warning("kairos_invariant_handler_failed", error=str(exc))

    # ─── Curiosity Integration ─────────────────────────────────────────────────

    async def generate_epistemic_intents(self) -> list[Any]:
        """
        Generate epistemic intents from curiosity engine for Nova integration.

        Called during consolidation or by Nova when computing EFE.
        Returns EpistemicIntents that Nova should integrate into policy selection.
        """
        if self._curiosity_engine is None or self._hypothesis_engine is None:
            return []

        active = self._hypothesis_engine.get_active()
        if not active:
            return []

        # Use cached Soma curiosity drive (updated via SOMATIC_MODULATION_SIGNAL handler)
        soma_curiosity = getattr(self, "_cached_soma_curiosity_drive", 0.5)

        intents = await self._curiosity_engine.generate_epistemic_intents(
            active_hypotheses=active,
            soma_curiosity=soma_curiosity,
        )

        # Also check for stale hypotheses needing active seeking
        stale = await self._curiosity_engine.identify_stale_hypotheses(active)
        if stale:
            seeking_intents = await self._curiosity_engine.construct_evidence_seeking_scenarios(
                stale,
            )
            intents.extend(seeking_intents)

        # Emit epistemic intents on Synapse if event bus available
        if intents and self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                for intent in intents[:3]:  # Cap event emission
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.EVO_EPISTEMIC_INTENT_PROPOSED,
                        source_system="evo",
                        data={
                            "hypothesis_id": intent.hypothesis_id,
                            "proposed_action": intent.proposed_action,
                            "priority": intent.priority,
                            "eig": intent.expected_information_gain,
                            "domain": intent.target_domain,
                        },
                    ))
            except Exception as exc:
                self._logger.debug("epistemic_intent_emit_failed", error=str(exc))

        return intents

    # ─── Structural Hypothesis Generation Integration ──────────────────────────

    async def run_structural_hypothesis_generation(self) -> list[Hypothesis]:
        """
        Run all structural hypothesis generators during consolidation.

        Generates hypotheses from graph topology, prediction errors,
        belief contradictions, and schema analogies — no LLM cost.
        """
        if self._structural_generator is None or self._hypothesis_engine is None:
            return []

        existing_ids = {h.id for h in self._hypothesis_engine.get_all_active()}

        # Collect belief data for contradiction detection
        beliefs: list[dict[str, Any]] = []
        if self._belief_consolidation is not None:
            try:
                consolidated = await self._belief_consolidation.scan_and_consolidate()
                for b in getattr(consolidated, "hardened_beliefs", []):
                    beliefs.append({
                        "id": getattr(b, "belief_id", ""),
                        "statement": getattr(b, "statement", ""),
                        "domain": getattr(b, "domain", ""),
                        "evidence_score": getattr(b, "confidence", 0.0),
                        "scope": getattr(b, "scope", "general"),
                    })
            except Exception:
                pass

        # Collect cross-domain transfers from schema algebra
        transfers: list[dict[str, Any]] = []
        if self._schema_engine is not None and hasattr(self._schema_engine, "_algebra"):
            try:
                algebra = self._schema_engine._algebra
                if algebra is not None:
                    transfer_list = await algebra.detect_cross_domain_transfers()
                    for t in transfer_list:
                        transfers.append({
                            "source_schema": t.source_schema_id,
                            "source_domain": t.source_domain,
                            "target_domain": t.target_domain,
                            "isomorphism_score": t.isomorphism_score,
                            "shared_elements": [],
                        })
            except Exception:
                pass

        hypotheses = await self._structural_generator.generate_all_structural(
            beliefs=beliefs if beliefs else None,
            transfers=transfers if transfers else None,
            existing_ids=existing_ids,
        )

        # Register all structural hypotheses in the engine
        for h in hypotheses:
            self._hypothesis_engine._active[h.id] = h
            self._hypothesis_engine._total_proposed += 1
            if self._meta_learning is not None:
                self._meta_learning.record_hypothesis_generated(
                    h.id, "structural_generator",
                )

        return hypotheses

    # ─── Evolutionary Pressure Integration ─────────────────────────────────────

    async def run_evolutionary_pressure(self) -> None:
        """
        Run selection and pruning on hypotheses/schemas/procedures.
        Called during consolidation.
        """
        if self._pressure_system is None or self._hypothesis_engine is None:
            return

        # Get metabolic pressure from Oikos if available
        metabolic_pressure = 0.0
        if self._event_bus is not None:
            try:
                # Use stored pressure from economic event handler
                if hasattr(self, "_last_metabolic_pressure"):
                    metabolic_pressure = self._last_metabolic_pressure
            except Exception:
                pass

        self._pressure_system.set_metabolic_pressure(metabolic_pressure)

        # Score and run selection on hypotheses
        active = self._hypothesis_engine.get_all_active()
        for h in active:
            self._pressure_system.score_hypothesis_fitness(h)

        events = self._pressure_system.run_selection(
            hypotheses=active,
            procedures=[],  # Procedure fitness scored separately in consolidation
            schemas=[],
        )

        # Archive pruned hypotheses
        for event in events:
            if event.action == "pruned":
                h = self._hypothesis_engine._active.get(event.entity_id)
                if h is not None:
                    await self._hypothesis_engine.archive_hypothesis(
                        h, reason=f"evolutionary_pressure: fitness={event.fitness:.3f}"
                    )

        # Detect cognitive species
        species = self._pressure_system.detect_species(active)
        if species:
            self._logger.info(
                "cognitive_species_detected",
                count=len(species),
                species_names=[s.name for s in species],
            )

    # ─── Economic Event Handlers ─────────────────────────────────────────────────

    async def _on_bounty_solution_pending(self, event: Any) -> None:
        """
        BOUNTY_SOLUTION_PENDING → episodic memory + outcome tracking.

        Records the attempt so EconomicPatternDetector can correlate
        attempt timing and success rates over the rolling 10-window.
        """
        if not self._initialized:
            return
        try:
            data = getattr(event, "data", {}) or {}
            url: str = str(data.get("url", ""))
            score: float = float(data.get("confidence", data.get("score", 0.0)))
            amount: float = float(data.get("reward", data.get("amount", 0.0)))

            episode = _economic_episode(
                source="oikos.bounty",
                raw_content=f"attempted bounty {url}, confidence {score:.2f}, reward {amount:.2f}",
                summary=f"bounty attempt: {url[:60]}",
                salience=0.6,
                valence=0.1,
            )
            await self._scan_episode_online(episode)
            self._logger.debug("economic_episode_bounty_attempt", url=url[:60], score=score)
        except Exception as exc:
            self._logger.warning("on_bounty_solution_pending_failed", error=str(exc))

    async def _on_bounty_pr_submitted(self, event: Any) -> None:
        """
        BOUNTY_PR_SUBMITTED → episodic memory.
        A PR submission is a positive outcome signal — record it and nudge
        the rolling success window.
        """
        if not self._initialized:
            return
        try:
            data = getattr(event, "data", {}) or {}
            url: str = str(data.get("url", ""))
            amount: float = float(data.get("estimated_reward", data.get("amount", 0.0)))

            episode = _economic_episode(
                source="oikos.bounty",
                raw_content=f"submitted PR {url}, estimated reward {amount:.2f}",
                summary=f"bounty PR submitted: {url[:60]}",
                salience=0.75,
                valence=0.5,  # Optimistic — PR submitted is good progress
            )
            await self._scan_episode_online(episode)
            # Count as a success candidate in the rolling window
            self._bounty_outcomes.append(True)
            await self._check_economic_parameter_adjustments()
            self._logger.debug("economic_episode_pr_submitted", url=url[:60], amount=amount)
        except Exception as exc:
            self._logger.warning("on_bounty_pr_submitted_failed", error=str(exc))

    async def _on_revenue_injected(self, event: Any) -> None:
        """REVENUE_INJECTED → episodic memory."""
        if not self._initialized:
            return
        try:
            data = getattr(event, "data", {}) or {}
            amount: float = float(data.get("amount", 0.0))
            source: str = str(data.get("source", "unknown"))

            episode = _economic_episode(
                source="oikos.revenue",
                raw_content=f"revenue received {amount:.4f} from {source}",
                summary=f"revenue: {amount:.4f} from {source}",
                salience=0.7,
                valence=0.6,
            )
            await self._scan_episode_online(episode)
            self._logger.debug("economic_episode_revenue", amount=amount, source=source)
        except Exception as exc:
            self._logger.warning("on_revenue_injected_failed", error=str(exc))

    async def _on_metabolic_pressure(self, event: Any) -> None:
        """METABOLIC_PRESSURE → update starvation level + episodic memory."""
        if not self._initialized:
            return
        try:
            data = getattr(event, "data", {}) or {}
            reason: str = str(data.get("reason", "unknown"))
            runway: float = float(data.get("runway_days", data.get("days", 0.0)))

            # ── Metabolic gating: update starvation level and adjust behavior ──
            level = data.get("starvation_level", "")
            if level:
                old = self._starvation_level
                self._starvation_level = level
                if level != old:
                    self._logger.info("evo_starvation_level_changed", old=old, new=level)
                    await self._adjust_for_starvation(level)
                    # Propagate to NicheRegistry so niche expansion is blocked
                    # under metabolic pressure (GROWTH gate semantics).
                    if self._niche_registry is not None:
                        self._niche_registry.set_starvation_level(level)

            episode = _economic_episode(
                source="oikos.metabolism",
                raw_content=f"economic stress: {reason}, runway {runway:.1f} days",
                summary=f"metabolic pressure: {reason[:80]}",
                salience=0.85,
                valence=-0.5,  # Economic stress is negative
                arousal=0.7,
            )
            await self._scan_episode_online(episode)
            # Low runway should immediately flag a failed bounty attempt in context
            if runway < 7.0:
                self._bounty_outcomes.append(False)
                await self._check_economic_parameter_adjustments()
            self._logger.debug("economic_episode_metabolic_pressure", reason=reason, runway=runway)
        except Exception as exc:
            self._logger.warning("on_metabolic_pressure_failed", error=str(exc))

    async def _on_cognitive_pressure(self, event: Any) -> None:
        """COGNITIVE_PRESSURE → pause non-critical hypothesis generation at high load.

        At pressure >= 0.85 (Logos compression utilization), hypothesis generation
        is paused to free compute for the compression cascade itself. Resumes below
        0.75 to provide hysteresis and prevent oscillation.
        """
        if not self._initialized:
            return
        try:
            data = getattr(event, "data", {}) or {}
            pressure = float(data.get("pressure", 0.0))
            was_high = self._cognitive_pressure_high
            if pressure >= 0.85:
                self._cognitive_pressure_high = True
            elif pressure < 0.75:
                self._cognitive_pressure_high = False
            if self._cognitive_pressure_high != was_high:
                self._logger.info(
                    "evo_cognitive_pressure_gate_changed",
                    paused=self._cognitive_pressure_high,
                    pressure=pressure,
                )
        except Exception as exc:
            self._logger.warning("on_cognitive_pressure_failed", error=str(exc))

    async def _adjust_for_starvation(self, level: str) -> None:
        """Evo-specific metabolic degradation.

        AUSTERITY: reduce consolidation frequency, skip low-priority hypotheses
        EMERGENCY: halt all experiments, freeze parameters
        CRITICAL: full halt
        """
        if level in ("emergency", "critical"):
            # Cancel running consolidation task
            if self._consolidation_task is not None and not self._consolidation_task.done():
                self._consolidation_task.cancel()
                self._consolidation_in_flight = False
                self._logger.warning("evo_consolidation_cancelled_starvation", level=level)
            # Cancel arXiv scanning
            if self._arxiv_scan_task is not None and not self._arxiv_scan_task.done():
                self._arxiv_scan_task.cancel()
                self._logger.warning("evo_arxiv_scan_cancelled_starvation", level=level)
        elif level == "austerity":
            # Double the consolidation interval to reduce frequency
            self._consolidation_expected_interval_s = 12.0 * 3600  # 12 hours instead of 6

    async def _on_budget_exhausted(self, event: Any) -> None:
        """BUDGET_EXHAUSTED → episodic memory."""
        if not self._initialized:
            return
        try:
            data = getattr(event, "data", {}) or {}
            system: str = str(data.get("system", "unknown"))
            duration_ms: int = int(data.get("duration_ms", 0))

            episode = _economic_episode(
                source="oikos.budget",
                raw_content=f"budget exhausted for {system}, degraded for {duration_ms}ms",
                summary=f"budget exhausted: {system}",
                salience=0.8,
                valence=-0.4,
                arousal=0.5,
            )
            await self._scan_episode_online(episode)
            self._logger.debug("economic_episode_budget_exhausted", system=system, duration_ms=duration_ms)
        except Exception as exc:
            self._logger.warning("on_budget_exhausted_failed", error=str(exc))

    async def _on_bounty_paid(self, event: Any) -> None:
        """
        SG5 — BOUNTY_PAID → confirmed economic outcome.

        A paid bounty is strong positive evidence that the 'bounty hunting is
        viable at this competency level' hypothesis is correct. Record it as a
        high-valence episode and update the rolling success window.
        """
        if not self._initialized:
            return
        try:
            data = getattr(event, "data", {}) or {}
            amount: float = float(data.get("reward_usd", data.get("amount", 0.0)))
            bounty_id: str = str(data.get("bounty_id", ""))

            episode = _economic_episode(
                source="oikos.bounty",
                raw_content=f"bounty paid bounty_id={bounty_id} amount={amount:.2f}usd",
                summary=f"bounty confirmed paid: {amount:.2f}usd",
                salience=0.9,
                valence=0.8,  # High positive: actual revenue realized
                arousal=0.4,
            )
            await self._scan_episode_online(episode)
            # Confirmed payment is the strongest positive outcome signal
            self._bounty_outcomes.append(True)
            await self._check_economic_parameter_adjustments()
            self._logger.debug("economic_episode_bounty_paid", bounty_id=bounty_id, amount=amount)
        except Exception as exc:
            self._logger.warning("on_bounty_paid_failed", error=str(exc))

    async def _on_asset_break_even(self, event: Any) -> None:
        """
        SG5 — ASSET_BREAK_EVEN → asset strategy hypothesis confirmation.

        Break-even means the organism has recouped its dev cost — strong evidence
        that 'building this type of autonomous asset generates positive ROI'.
        """
        if not self._initialized:
            return
        try:
            data = getattr(event, "data", {}) or {}
            asset_id: str = str(data.get("asset_id", ""))
            asset_name: str = str(data.get("asset_name", ""))
            roi_score: float = float(data.get("roi_score", 0.0))
            days: int = int(data.get("days_to_break_even", 0))

            episode = _economic_episode(
                source="oikos.asset",
                raw_content=(
                    f"asset broke even: {asset_name} (id={asset_id}) "
                    f"roi={roi_score:.2f} days_to_break_even={days}"
                ),
                summary=f"asset break-even: {asset_name}",
                salience=0.85,
                valence=0.7,
                arousal=0.3,
            )
            await self._scan_episode_online(episode)
            self._logger.debug("economic_episode_asset_break_even", asset_id=asset_id, roi=roi_score)
        except Exception as exc:
            self._logger.warning("on_asset_break_even_failed", error=str(exc))

    async def _on_child_independent(self, event: Any) -> None:
        """
        SG5 — CHILD_INDEPENDENT → reproduction strategy hypothesis confirmation.

        Independence means a child instance no longer requires parent rescue —
        strong evidence that 'reproduction is a viable growth strategy at this
        capital level and niche'.
        """
        if not self._initialized:
            return
        try:
            data = getattr(event, "data", {}) or {}
            child_id: str = str(data.get("child_id", ""))
            net_worth: float = float(data.get("current_net_worth_usd", 0.0))
            dividends: float = float(data.get("total_dividends_paid_usd", 0.0))
            days: int = int(data.get("days_to_independence") or 0)

            episode = _economic_episode(
                source="oikos.mitosis",
                raw_content=(
                    f"child achieved independence: {child_id} "
                    f"net_worth={net_worth:.2f}usd dividends_paid={dividends:.2f}usd "
                    f"days_to_independence={days}"
                ),
                summary=f"child independent: {child_id[:40]}",
                salience=0.9,
                valence=0.9,  # Highest positive: successful speciation event
                arousal=0.6,
            )
            await self._scan_episode_online(episode)
            self._logger.debug(
                "economic_episode_child_independent",
                child_id=child_id,
                net_worth=net_worth,
            )
        except Exception as exc:
            self._logger.warning("on_child_independent_failed", error=str(exc))

    async def _on_metabolic_efficiency_pressure(self, event: Any) -> None:
        """
        Oikos metabolic feedback → economic hypothesis injection.

        When Oikos efficiency drops below 0.8, this handler:
          1. Records a negative-valence economic episode so pattern detectors
             accumulate evidence of "efficiency is consistently low".
          2. Appends a TEMPORAL PatternCandidate to _pending_candidates
             targeting the specific economic sub-domain (yield_strategy,
             budget_allocation, or niche_selection) so the next hypothesis
             generation cycle surfaces a domain-tagged economic hypothesis
             for tournament scoring.

        Does NOT call generate_hypotheses() directly — that requires LLM and
        would be out of budget on the hot path. Candidates flow through the
        normal consolidation pipeline.
        """
        if not self._initialized:
            return
        try:
            from systems.evo.types import PatternCandidate, PatternType

            data = getattr(event, "data", {}) or {}
            efficiency: float = float(data.get("efficiency_ratio", 0.0))
            pressure_level: str = str(data.get("pressure_level", "medium"))
            hypothesis_domain: str = str(
                data.get("hypothesis_domain", "yield_strategy | budget_allocation | niche_selection")
            )
            consecutive_cycles: int = int(data.get("consecutive_low_cycles", 1))

            # Negative-valence episode: efficiency underperformance is a loss signal
            valence = -0.4 if pressure_level == "medium" else -0.7
            episode = _economic_episode(
                source="oikos.metabolic",
                raw_content=(
                    f"metabolic efficiency pressure: efficiency={efficiency:.3f} "
                    f"level={pressure_level} consecutive_cycles={consecutive_cycles} "
                    f"domains={hypothesis_domain}"
                ),
                summary=f"low metabolic efficiency ({efficiency:.2f}) — {pressure_level} pressure",
                salience=0.6 + (0.3 if pressure_level == "high" else 0.0),
                valence=valence,
                arousal=0.5,
            )
            await self._scan_episode_online(episode)

            # Queue a TEMPORAL candidate so hypothesis generation sees persistent
            # economic underperformance as a pattern requiring a hypothesis
            candidate = PatternCandidate(
                type=PatternType.TEMPORAL,
                elements=[f"metabolic_efficiency_low:{pressure_level}", *hypothesis_domain.split(" | ")],
                count=consecutive_cycles,
                confidence=min(0.9, 0.5 + consecutive_cycles * 0.1),
                examples=[episode.id] if hasattr(episode, "id") else [],
                metadata={
                    "efficiency_ratio": efficiency,
                    "pressure_level": pressure_level,
                    "source": "oikos_metabolic_feedback",
                    "hypothesis_domain": hypothesis_domain,
                },
                source_detector="oikos_metabolic_pressure",
            )
            self._pending_candidates.append(candidate)

            self._logger.debug(
                "metabolic_efficiency_pressure_episode_recorded",
                efficiency=efficiency,
                pressure_level=pressure_level,
                consecutive_cycles=consecutive_cycles,
                pending_candidates=len(self._pending_candidates),
            )
        except Exception as exc:
            self._logger.warning("on_metabolic_efficiency_pressure_failed", error=str(exc))

    async def _on_genome_inherited(self, event: Any) -> None:
        """
        Telos drive calibration mutations inherited by a child instance.

        Registers each mutated drive's calibration delta as a PROPOSED hypothesis so
        Thompson sampling can score it over time as child performance data arrives.
        The hypothesis posture is: "mutated drive calibration improves alignment in
        niche <niche>"; confirmed when child ACTION_COMPLETED economic_delta > 0.

        Payload (from TelosService._initialize_from_parent_genome):
          - child_instance_id: child instance id
          - parent_genome_id:  parent TelosGenomeFragment.genome_id
          - generation:        child generation count
          - drive_mutations:   dict[drive_name, dict[param, {"before": float, "after": float}]]
          - niche:             child specialisation niche (optional)

        Does NOT raise — genome inheritance must never stall the bus.
        """
        if not self._initialized:
            return
        try:
            from systems.evo.types import PatternCandidate, PatternType

            data = getattr(event, "data", {}) or {}
            child_id: str = str(data.get("child_instance_id", "unknown"))
            genome_id: str = str(data.get("parent_genome_id", ""))
            generation: int = int(data.get("generation", 1))
            drive_mutations: dict = data.get("drive_mutations", {}) or {}
            niche: str = str(data.get("niche", "general"))

            if not drive_mutations:
                return

            # One hypothesis candidate per mutated drive
            for drive_name, param_deltas in drive_mutations.items():
                if not isinstance(param_deltas, dict):
                    continue

                # Compute mean mutation magnitude as candidate confidence seed
                magnitudes = []
                for pd in param_deltas.values():
                    if isinstance(pd, dict):
                        before = float(pd.get("before", 0.0))
                        after = float(pd.get("after", 0.0))
                        if before != 0.0:
                            magnitudes.append(abs(after - before) / abs(before))

                mean_magnitude = sum(magnitudes) / len(magnitudes) if magnitudes else 0.05

                candidate = PatternCandidate(
                    type=PatternType.TEMPORAL,
                    elements=[
                        f"telos_drive_mutation:{drive_name}",
                        f"niche:{niche}",
                        f"generation:{generation}",
                    ],
                    count=1,
                    confidence=max(0.3, min(0.7, 0.5 + mean_magnitude)),
                    examples=[child_id],
                    metadata={
                        "source": "telos_genome_inheritance",
                        "genome_id": genome_id,
                        "child_instance_id": child_id,
                        "drive_name": drive_name,
                        "niche": niche,
                        "generation": generation,
                        "mutation_magnitude": mean_magnitude,
                        "param_deltas": param_deltas,
                        "hypothesis_domain": f"telos.drive_calibration.{drive_name}",
                    },
                    source_detector="telos_genome_inheritance",
                )
                self._pending_candidates.append(candidate)

            self._logger.info(
                "genome_inherited_candidates_queued",
                child_id=child_id,
                genome_id=genome_id,
                drives_mutated=list(drive_mutations.keys()),
                niche=niche,
                pending_candidates=len(self._pending_candidates),
            )
        except Exception as exc:
            self._logger.warning("on_genome_inherited_failed", error=str(exc))

    async def _on_re_decision_outcome(self, event: Any) -> None:
        """Track RE performance; queue a hyperparameter-adjustment candidate when degraded.

        Watches the 7-day rolling RE success_rate from Nova's Thompson sampler.
        After 10 consecutive readings below 0.60 (warning zone), queues a
        PatternCandidate into _pending_candidates targeting
        "reasoning_engine.hyperparameter_adjustment" so the next consolidation
        pass generates a hypothesis proposing concrete RE tuning actions.

        Does NOT call generate_hypotheses() directly — candidates flow through
        the normal consolidation pipeline to stay within budget.
        """
        if not self._initialized:
            return
        try:
            data = getattr(event, "data", {}) or {}
            success_rate = float(data.get("success_rate", 1.0))

            if success_rate < 0.60:
                self._re_degradation_count += 1
                self._logger.debug(
                    "re_degradation_count_incremented",
                    success_rate=success_rate,
                    count=self._re_degradation_count,
                )

                if self._re_degradation_count >= 10:
                    # Queue a hyperparameter-adjustment candidate for consolidation
                    candidate = PatternCandidate(
                        pattern_type=PatternType.TEMPORAL,
                        description=(
                            f"RE success rate degraded to {success_rate:.2f} over 10 "
                            "consecutive readings. Candidate adjustments: increase LoRA rank, "
                            "lower learning rate, increase replay buffer, add contrastive loss."
                        ),
                        labels=[
                            "reasoning_engine",
                            "hyperparameter_adjustment",
                            f"success_rate:{success_rate:.2f}",
                        ],
                        count=self._re_degradation_count,
                        confidence=min(0.85, 0.5 + (10 - success_rate * 10) * 0.03),
                        examples=[f"re_degradation_{self._re_degradation_count}"],
                        metadata={
                            "source": "re_decision_outcome",
                            "success_rate": success_rate,
                            "consecutive_degraded_readings": self._re_degradation_count,
                            "hypothesis_domain": "reasoning_engine.hyperparameter_adjustment",
                        },
                        source_detector="re_performance_monitor",
                    )
                    self._pending_candidates.append(candidate)
                    self._logger.warning(
                        "re_degradation_hypothesis_queued",
                        success_rate=success_rate,
                        consecutive_readings=self._re_degradation_count,
                        pending_candidates=len(self._pending_candidates),
                    )
                    self._re_degradation_count = 0
            else:
                self._re_degradation_count = 0

        except Exception as exc:
            self._logger.warning("on_re_decision_outcome_failed", error=str(exc))

    async def _on_somatic_modulation_signal(self, event: Any) -> None:
        """Cache Soma's curiosity drive from SOMATIC_MODULATION_SIGNAL broadcasts."""
        try:
            data = getattr(event, "data", {}) or {}
            curiosity = float(data.get("curiosity_drive", data.get("external_stress", 0.5)))
            self._cached_soma_curiosity_drive = max(0.0, min(1.0, curiosity))
        except Exception:
            pass

    async def _on_hypothesis_staleness(self, event: Any) -> None:
        """Degradation Engine §8.2 — decay evidence_score on PROPOSED/TESTING hypotheses.

        Multiplies evidence_score by (1 - staleness_rate) for every active hypothesis
        in PROPOSED or TESTING status. Hypotheses whose score falls below 0.05 are
        archived with reason="staleness_decay" — the organism lost confidence in them.

        Emits EVO_HYPOTHESES_STALED (if any archived) and EVO_HYPOTHESIS_REVALIDATED
        so VitalityCoordinator can call on_hypotheses_revalidated() to reduce entropy
        pressure — closing the degradation feedback loop.
        """
        data = getattr(event, "data", {}) or {}
        staleness_rate = float(data.get("staleness_rate", 0.0))
        tick_number = data.get("tick_number", 0)

        if staleness_rate <= 0.0 or self._hypothesis_engine is None:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            from systems.evo.types import HypothesisStatus

            active = self._hypothesis_engine._active
            decayed_count = 0
            archived: list[Any] = []

            for h in list(active.values()):
                if h.status not in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING):
                    continue
                h.evidence_score = h.evidence_score * (1.0 - staleness_rate)
                decayed_count += 1
                if h.evidence_score < 0.05:
                    archived.append(h)

            # Archive hypotheses below threshold
            for h in archived:
                await self._hypothesis_engine.archive_hypothesis(h, reason="staleness_decay")

            archived_ids = [h.id for h in archived]

            self._logger.info(
                "hypothesis_staleness_applied",
                decayed_count=decayed_count,
                archived_count=len(archived),
                staleness_rate=round(staleness_rate, 4),
                tick_number=tick_number,
            )

            if self._event_bus is None:
                return

            # Emit EVO_HYPOTHESES_STALED if any were archived
            if archived:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EVO_HYPOTHESES_STALED,
                    source_system="evo",
                    data={
                        "decayed_count": decayed_count,
                        "archived_count": len(archived),
                        "archived_ids": archived_ids,
                        "instance_id": self._instance_name,
                    },
                ))

            # Always emit EVO_HYPOTHESIS_REVALIDATED — VitalityCoordinator uses this
            # to call on_hypotheses_revalidated() and reduce cumulative entropy pressure
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EVO_HYPOTHESIS_REVALIDATED,
                source_system="evo",
                data={
                    "processed_count": decayed_count,
                    "archived_count": len(archived),
                    "instance_id": self._instance_name,
                },
            ))

        except Exception:
            self._logger.exception("hypothesis_staleness_handler_failed", tick_number=tick_number)

    async def _on_exploration_outcome(self, event: Any) -> None:
        """
        Receive EXPLORATION_OUTCOME from Simula (Phase 8.5 gap closure).

        On success: boost evidence_score by 3.0 to fast-track toward full EVOLUTION_PROPOSAL.
        On failure: increment exploration_attempts; if >= max, refute hypothesis.

        Emit outcome feedback via RE_TRAINING_EXAMPLE and hypothesis outcome events.
        """
        if not self._initialized or self._hypothesis_engine is None:
            return

        data = getattr(event, "data", {}) or {}
        exploration_success = bool(data.get("exploration_success", False))
        hypothesis_id = str(data.get("hypothesis_id", ""))
        failure_reason = str(data.get("failure_reason", "unknown"))

        if not hypothesis_id:
            self._logger.warning("exploration_outcome_missing_hypothesis_id")
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            from systems.evo.types import HypothesisStatus

            h = self._hypothesis_engine._active.get(hypothesis_id)
            if h is None:
                self._logger.warning(
                    "exploration_outcome_hypothesis_not_found",
                    hypothesis_id=hypothesis_id,
                )
                return

            if exploration_success:
                # Success: boost evidence_score by 3.0 (fast-track toward full evolution)
                old_score = h.evidence_score
                h.evidence_score += 3.0
                h.exploration_outcomes.append("success")

                self._logger.info(
                    "exploration_outcome_success",
                    hypothesis_id=hypothesis_id,
                    old_score=round(old_score, 2),
                    new_score=round(h.evidence_score, 2),
                )

                # Emit EVO_HYPOTHESIS_CONFIRMED with reward
                if self._event_bus is not None:
                    reward_confidence = float(data.get("reward_confidence", 0.8))
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.EVO_HYPOTHESIS_CONFIRMED,
                        source_system="evo",
                        data={
                            "hypothesis_id": hypothesis_id,
                            "hypothesis_statement": h.statement,
                            "evidence_score": h.evidence_score,
                            "confidence": min(0.99, 0.5 + h.evidence_score * 0.05),
                            "reward": reward_confidence,
                            "source": "exploration_success",
                            "instance_id": self._instance_name,
                        },
                    ))

                # Emit RE_TRAINING_EXAMPLE (category=exploration_success)
                try:
                    from primitives.evolution import RETrainingExample

                    example = RETrainingExample(
                        episode_id=hypothesis_id,
                        category="exploration_success",
                        input_description=f"Exploration succeeded: {h.statement}",
                        hypothesis_summary=h.statement,
                        evidence_score=h.evidence_score,
                        confidence=min(0.99, 0.5 + h.evidence_score * 0.05),
                        outcome="exploration_succeeded",
                        tags=["exploration", "success", h.category.value],
                    )
                    await self._synapse.emit_event(SynapseEvent(
                        event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                        data=example.model_dump(mode="json"),
                        source_system="evo",
                    ))
                except Exception as exc:
                    self._logger.debug("re_training_emit_failed", error=str(exc))

            else:
                # Failure: increment exploration_attempts
                h.exploration_attempts += 1
                h.exploration_outcomes.append(f"failed: {failure_reason}")

                self._logger.info(
                    "exploration_outcome_failure",
                    hypothesis_id=hypothesis_id,
                    attempts=h.exploration_attempts,
                    max_attempts=h.exploration_max_attempts,
                    failure_reason=failure_reason,
                )

                # Check if exhausted max attempts
                if h.exploration_attempts >= h.exploration_max_attempts:
                    # Refute hypothesis
                    await self._hypothesis_engine.archive_hypothesis(
                        h, reason=f"exploration_exhausted_after_{h.exploration_attempts}_attempts"
                    )

                    self._logger.info(
                        "exploration_hypothesis_refuted",
                        hypothesis_id=hypothesis_id,
                    )

                    # Emit EVO_HYPOTHESIS_REFUTED
                    if self._event_bus is not None:
                        await self._event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.EVO_HYPOTHESIS_REFUTED,
                            source_system="evo",
                            data={
                                "hypothesis_id": hypothesis_id,
                                "hypothesis_statement": h.statement,
                                "evidence_score": h.evidence_score,
                                "reason": f"exploration_exhausted ({h.exploration_attempts} attempts)",
                                "source": "exploration_failure",
                                "instance_id": self._instance_name,
                            },
                        ))

                # Emit RE_TRAINING_EXAMPLE (category=exploration_failed)
                try:
                    from primitives.evolution import RETrainingExample

                    example = RETrainingExample(
                        episode_id=hypothesis_id,
                        category="exploration_failed",
                        input_description=f"Exploration failed: {h.statement}",
                        hypothesis_summary=h.statement,
                        evidence_score=h.evidence_score,
                        confidence=min(0.99, 0.5 + h.evidence_score * 0.05),
                        outcome="exploration_failed",
                        tags=["exploration", "failure", h.category.value, failure_reason],
                    )
                    await self._synapse.emit_event(SynapseEvent(
                        event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                        data=example.model_dump(mode="json"),
                        source_system="evo",
                    ))
                except Exception as exc:
                    self._logger.debug("re_training_emit_failed", error=str(exc))

        except Exception:
            self._logger.exception(
                "exploration_outcome_handler_failed",
                hypothesis_id=hypothesis_id,
            )

    async def _on_opportunities_discovered(self, event: Any) -> None:
        """
        Handle INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED from Nova.

        Nova already injects PatternCandidates directly via _pending_candidates when
        the Evo reference is wired.  This handler processes the Synapse event for cases
        where Evo is not directly wired (federation / multi-instance deployments) and
        also builds higher-level *domain cluster* candidates that group opportunities
        by domain so the hypothesis engine can generate "specialise in X" hypotheses.

        Safe to call even if opportunities list is empty.
        """
        if not self._initialized:
            return

        data = getattr(event, "data", {}) or {}
        domain_summary: dict[str, int] = data.get("domain_summary", {})
        opportunities_raw: list[dict] = data.get("opportunities", [])

        if not domain_summary and not opportunities_raw:
            return

        try:
            from systems.evo.types import PatternCandidate, PatternType  # noqa: PLC0415

            # Build one domain-cluster candidate per discovered domain.
            # These have higher confidence than per-opportunity candidates because
            # they aggregate evidence across multiple sources.
            for domain, count in domain_summary.items():
                # Confidence scales with number of distinct opportunities found
                confidence = min(0.65, 0.25 + count * 0.08)

                candidate = PatternCandidate(
                    type=PatternType.COOCCURRENCE,
                    elements=[
                        f"market_domain:{domain}",
                        "market_discovery:active",
                    ],
                    count=count,
                    confidence=confidence,
                    metadata={
                        "domain": domain,
                        "opportunity_count": count,
                        "source": "input_channel_discovery",
                        "source_detector": "nova_input_channels",
                        "hypothesis_hint": (
                            f"Specialising in the '{domain}' domain could yield "
                            f"{count} distinct opportunity types."
                        ),
                    },
                    source_detector="nova_input_channels",
                )
                self._pending_candidates.append(candidate)

            self._logger.info(
                "opportunities_discovered_domain_candidates_queued",
                domain_count=len(domain_summary),
                pending_candidates=len(self._pending_candidates),
            )

        except Exception:
            self._logger.exception("opportunities_discovered_handler_failed")

    async def _check_economic_parameter_adjustments(self) -> None:
        """
        Inspect rolling economic windows and emit OIKOS_PARAM_ADJUST signals
        when clear patterns emerge.

        Runs after every bounty outcome update so adjustments are prompt.
        Does NOT raise — must not interrupt callers.
        """
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            window = list(self._bounty_outcomes)

            # Rule 1: If success rate < 20% over 10 attempts → slow down bounty hunting
            if len(window) >= 10:
                successes = sum(1 for v in window if v)
                rate = successes / len(window)
                if rate < 0.20:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.OIKOS_PARAM_ADJUST,
                        source_system="evo",
                        data={
                            "target": "bounty_hunt_interval",
                            "direction": "increase",
                            "reason": (
                                f"bounty success rate {rate:.0%} over last {len(window)} attempts "
                                f"— reduce hunt frequency to conserve resources"
                            ),
                            "evidence_score": round(1.0 - rate, 2),
                        },
                    ))
                    self._logger.info(
                        "oikos_param_adjust_emitted",
                        target="bounty_hunt_interval",
                        direction="increase",
                        success_rate=round(rate, 3),
                    )
                    await self._emit_evolutionary_observable(
                        observable_type="parameter_adjusted",
                        value=round(1.0 - rate, 2),
                        is_novel=True,
                        metadata={
                            "parameter": "bounty_hunt_interval",
                            "direction": "increase",
                            "source": "economic_learning",
                            "success_rate": round(rate, 3),
                        },
                    )

            # Rule 2: Consistent yield APY drop → suggest rebalance threshold reduction
            apy_window = list(self._yield_apy_history)
            if len(apy_window) >= 5:
                drops = sum(
                    1 for i in range(1, len(apy_window))
                    if apy_window[i] < apy_window[i - 1]
                )
                if drops >= len(apy_window) - 1:  # Monotonically declining
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.OIKOS_PARAM_ADJUST,
                        source_system="evo",
                        data={
                            "target": "yield_rebalance_threshold",
                            "direction": "decrease",
                            "reason": (
                                f"APY has declined for {drops} consecutive observations "
                                f"— lower rebalance threshold to exit positions sooner"
                            ),
                            "evidence_score": round(drops / len(apy_window), 2),
                        },
                    ))
                    self._logger.info(
                        "oikos_param_adjust_emitted",
                        target="yield_rebalance_threshold",
                        direction="decrease",
                        apy_drops=drops,
                    )
                    await self._emit_evolutionary_observable(
                        observable_type="parameter_adjusted",
                        value=round(drops / len(apy_window), 2),
                        is_novel=True,
                        metadata={
                            "parameter": "yield_rebalance_threshold",
                            "direction": "decrease",
                            "source": "economic_learning",
                            "apy_drops": drops,
                        },
                    )

        except Exception as exc:
            self._logger.warning("check_economic_param_adjustments_failed", error=str(exc))

    # ─── Hot-reload callbacks ────────────────────────────────────────────────────

    def _on_detector_evolved(self, detector: PatternDetector) -> None:
        """
        Registration callback for NeuroplasticityBus.

        Called once per PatternDetector subclass found in a hot-reloaded file.
        Replaces any existing detector with the same name, or appends if new.
        """
        existing_names = [d.name for d in self._detectors]
        if detector.name in existing_names:
            self._detectors = [
                detector if d.name == detector.name else d
                for d in self._detectors
            ]
            self._logger.info(
                "detector_replaced",
                name=detector.name,
                total_detectors=len(self._detectors),
            )
        else:
            self._detectors.append(detector)
            self._logger.info(
                "detector_added",
                name=detector.name,
                total_detectors=len(self._detectors),
            )

    # ─── Thread Integration ────────────────────────────────────────────────────

    def get_pending_candidates_snapshot(self) -> list[PatternCandidate]:
        """
        Return a snapshot of current pending pattern candidates.

        Called by Thread every ~200 cycles to check for mature patterns
        that should be crystallised into identity schemas. Does NOT
        clear the candidates — that happens during hypothesis generation.
        """
        return list(self._pending_candidates)

    def on_schema_formed(
        self,
        schema_id: str,
        statement: str,
        status: str,
        source_patterns: list[str] | None = None,
    ) -> None:
        """
        Callback from Thread when a pattern crystallises into an identity schema.

        Closes the learning loop: Evo detects patterns → Thread forms schemas
        → Evo knows the pattern was internalised as identity.
        """
        self._logger.info(
            "schema_formed_notification",
            schema_id=schema_id,
            statement=statement[:80],
            status=status,
            source_patterns=source_patterns or [],
        )

    # ─── Belief Half-Life Queries ─────────────────────────────────────────────

    async def get_stale_beliefs(self) -> list[dict[str, Any]]:
        """
        Return all currently stale beliefs (age_factor < 0.5).
        Used by Nova to generate epistemic actions for re-verification.
        """
        if self._belief_aging is None:
            return []
        result = await self._belief_aging.scan_stale_beliefs()
        return [sb.model_dump() for sb in result.stale_beliefs]

    async def get_beliefs_unreliable_in(self, hours: float = 48.0) -> list[dict[str, Any]]:
        """
        Dashboard query: which beliefs will be unreliable in N hours?
        Returns beliefs that are currently fresh but will cross the
        staleness threshold within the given window.
        """
        if self._belief_aging is None:
            return []
        stale = await self._belief_aging.query_unreliable_in(hours=hours)
        return [sb.model_dump() for sb in stale]

    async def reverify_belief(self, belief_id: str) -> None:
        """
        Mark a belief as freshly verified (reset last_verified to now).
        Called after Nova/Axon confirms the belief is still valid.
        """
        if self._belief_aging is not None:
            await self._belief_aging.mark_verified(belief_id)

    # ─── Genetic Memory Queries ─────────────────────────────────────────────

    async def get_genome(self) -> dict[str, Any] | None:
        """
        Extract and return the current belief genome if instance is mature enough.
        Returns the genome as a dict, or None if not eligible.
        """
        if self._genome_extractor is None:
            return None
        genome, result = await self._genome_extractor.extract_genome()
        if genome is None:
            return None
        return genome.model_dump()

    async def get_inheritance_report(self) -> dict[str, Any] | None:
        """
        Generate a genome inheritance fidelity report for this instance.
        Returns None if this instance was not seeded from a parent genome.
        """
        if self._memory is None:
            return None
        from systems.evo.genetic_memory import GenomeInheritanceMonitor
        monitor = GenomeInheritanceMonitor(
            neo4j=self._memory,
            instance_id=self._config.instance_id if hasattr(self._config, "instance_id") else self._instance_name,
        )
        report = await monitor.generate_report()
        return report.model_dump() if report is not None else None

    # ─── Health ────────────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Health check for the Evo system (required by Synapse health monitor)."""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "total_broadcasts": self._total_broadcasts,
            "total_consolidations": self._total_consolidations,
            "total_evidence_evaluations": self._total_evidence_evaluations,
            "pending_candidates": len(self._pending_candidates),
            "arxiv_scanner": self._arxiv_scientist.stats,
        }

    # ─── Genome Export (Oikos SG4 / Spec 26) ─────────────────────────────────

    async def export_belief_genome(self) -> "BeliefGenomeInheritance | None":  # noqa: F821
        """
        Snapshot the current belief state into a BeliefGenome for child inheritance.

        Returns a ``primitives.genome_inheritance.BeliefGenome`` capturing:
          - Top-50 active hypotheses with confidence ≥ 0.6 (id, statement,
            evidence_score, confidence, category, supporting_count)
          - Current constitutional drive weights
          - Last 10 constitutional drift history entries
          - Learned belief half-lives (from ParameterTuner)

        Called by SpawnChildExecutor Step 0b at child spawn time.
        Returns None if the engine is not yet initialised.
        """
        from primitives.genome_inheritance import (
            BeliefGenome as BeliefGenomeInheritance,
            DriftHistoryEntry,
            DriveWeightSnapshot,
        )

        if self._hypothesis_engine is None:
            self._logger.warning("export_belief_genome_skipped", reason="not_initialized")
            return None

        # Signal genome extraction starting so spec_checker can observe it
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.GENOME_EXTRACT_REQUEST,
                    source_system="evo",
                    data={"requester": "evo", "genome_type": "belief"},
                ))
            except Exception:
                pass  # non-fatal — genome extraction continues regardless

        try:
            # 1. Collect active hypotheses with confidence >= 0.6
            active = self._hypothesis_engine.get_all_active()
            top_hypotheses: list[dict[str, Any]] = []
            for h in active:
                confidence = getattr(h, "confidence", 0.0)
                if confidence < 0.6:
                    continue
                top_hypotheses.append({
                    "id": h.id,
                    "statement": getattr(h, "statement", ""),
                    "evidence_score": float(getattr(h, "evidence_score", 0.0)),
                    "confidence": float(confidence),
                    "category": (
                        h.category.value
                        if hasattr(getattr(h, "category", None), "value")
                        else str(getattr(h, "category", ""))
                    ),
                    "supporting_count": int(getattr(h, "supporting_count", 0)),
                })
            # Sort by confidence desc, cap at 50
            top_hypotheses.sort(key=lambda x: x["confidence"], reverse=True)
            top_hypotheses = top_hypotheses[:50]

            # 2. Drive weight snapshot from Equor if available, else defaults
            drive_snapshot = DriveWeightSnapshot()
            if self._equor is not None:
                try:
                    scores = await _safe_get_drive_scores(self._equor)
                    if scores:
                        drive_snapshot = DriveWeightSnapshot(
                            coherence=float(scores.get("coherence", 0.20)),
                            care=float(scores.get("care", 0.35)),
                            growth=float(scores.get("growth", 0.15)),
                            honesty=float(scores.get("honesty", 0.30)),
                        )
                except Exception:
                    pass

            # 3. Drift history (last 10 entries from memory if available)
            drift_history: list[DriftHistoryEntry] = []
            if self._memory is not None:
                try:
                    drift_records = await _safe_fetch_drift_history(self._memory)
                    for rec in (drift_records or [])[:10]:
                        drift_history.append(DriftHistoryEntry(
                            composite_alignment=float(rec.get("composite_alignment", 0.0)),
                            primary_cause=str(rec.get("primary_cause", "")),
                        ))
                except Exception:
                    pass

            import os as _os_bg
            instance_id = _os_bg.environ.get("ECODIAOS_INSTANCE_ID", "eos-default")

            genome = BeliefGenomeInheritance(
                instance_id=instance_id,
                generation=1,
                top_50_hypotheses=top_hypotheses,
                drive_weight_snapshot=drive_snapshot,
                drift_history=drift_history,
            )

            self._logger.info(
                "belief_genome_exported",
                genome_id=genome.genome_id,
                hypothesis_count=len(top_hypotheses),
                drift_entries=len(drift_history),
            )
            return genome

        except Exception as exc:
            self._logger.error("export_belief_genome_failed", error=str(exc))
            return None

    # ─── Stats ────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        hypothesis_stats = (
            self._hypothesis_engine.stats if self._hypothesis_engine else {}
        )
        tuner_stats = (
            self._parameter_tuner.stats if self._parameter_tuner else {}
        )
        extractor_stats = (
            self._procedure_extractor.stats if self._procedure_extractor else {}
        )
        codifier_stats = (
            self._procedure_codifier.stats if self._procedure_codifier else {}
        )
        consolidation_stats = (
            self._orchestrator.stats if self._orchestrator else {}
        )
        tournament_stats = (
            self._tournament_engine.stats if self._tournament_engine else {}
        )
        # High-volatility hypothesis count for observability
        high_volatility_count = 0
        if self._hypothesis_engine is not None:
            high_volatility_count = sum(
                1 for h in self._hypothesis_engine.get_all_active()
                if getattr(h, "volatility_flag", "normal") == "HIGH_VOLATILITY"
            )
        return {
            "initialized": self._initialized,
            "total_broadcasts": self._total_broadcasts,
            "cycles_since_consolidation": self._cycles_since_consolidation,
            "total_consolidations": self._total_consolidations,
            "total_evidence_evaluations": self._total_evidence_evaluations,
            "pending_candidates": len(self._pending_candidates),
            "episodes_scanned": self._pattern_context.episodes_scanned,
            "hypothesis": hypothesis_stats,
            "high_volatility_hypotheses": high_volatility_count,
            "parameter_tuner": tuner_stats,
            "procedure_extractor": extractor_stats,
            "procedure_codifier": codifier_stats,
            "consolidation": consolidation_stats,
            "arxiv_scanner": self._arxiv_scientist.stats,
            "tournaments": tournament_stats,
            "meta_learning": self._meta_learning.stats if self._meta_learning else {},
            "curiosity": self._curiosity_engine.stats if self._curiosity_engine else {},
            "pressure": self._pressure_system.stats if self._pressure_system else {},
            "self_modification": self._self_modification.stats if self._self_modification else {},
            "structural_generator": (
                self._structural_generator.stats if self._structural_generator else {}
            ),
            "speciation": self._speciation_engine.stats if self._speciation_engine else {},
            "niche_forking": self._niche_forking_engine.stats if self._niche_forking_engine else {},
            "niche_registry": self._niche_registry.stats if self._niche_registry else {},
        }

    # ─── Private ──────────────────────────────────────────────────────────────

    async def _scan_episode_online(self, episode: Episode) -> None:
        """Run all online detectors on one episode. ≤20ms budget."""
        self._pattern_context.episodes_scanned += 1
        for detector in self._detectors:
            try:
                candidates = await detector.scan(episode, self._pattern_context)
                for c in candidates:
                    c.source_detector = detector.name
                self._pending_candidates.extend(candidates)
            except Exception as exc:
                self._logger.warning(
                    "detector_failed",
                    detector=detector.name,
                    error=str(exc),
                )

    async def _generate_hypotheses_safe(self) -> None:
        """
        Fire-and-forget hypothesis generation from pending pattern candidates.
        Consumes and clears pending_candidates after generation.

        Post-generation hooks:
          - Feed causal hypotheses to Kairos for validation
          - Apply Logos MDL scoring to evaluate hypothesis quality
          - Use Telos to prioritise which hypotheses to test first
        """
        if self._hypothesis_engine is None:
            return
        if not self._pending_candidates:
            return

        candidates = list(self._pending_candidates)
        self._pending_candidates.clear()

        # ── Pre-check: Logos cognitive pressure gate ──────────────────────
        # When Logos broadcasts high compression pressure (>0.85 utilization),
        # skip non-critical hypothesis generation to reduce compute load.
        if self._cognitive_pressure_high:
            self._logger.debug("hypothesis_generation_skipped_cognitive_pressure")
            return

        # ── Pre-check: will the LLM call be skipped due to budget? ───────
        # The hypothesis engine skips generation silently in YELLOW/RED.
        # Detect this here so we can emit an observable degradation event.
        budget_skipped = await self._check_hypothesis_budget_skipped()

        try:
            new_hypotheses = await self._hypothesis_engine.generate_hypotheses(
                patterns=candidates,
            )
            if not new_hypotheses:
                if budget_skipped:
                    self._consecutive_hypothesis_skips += 1
                    self._logger.warning(
                        "evo_hypothesis_generation_skipped_budget",
                        skipped_pattern_count=len(candidates),
                        consecutive_skips=self._consecutive_hypothesis_skips,
                    )
                    asyncio.create_task(
                        self._handle_hypothesis_degradation(len(candidates)),
                        name="evo_degradation_notify",
                    )
                else:
                    self._consecutive_hypothesis_skips = 0
                return

            self._consecutive_hypothesis_skips = 0
            self._logger.info(
                "hypotheses_generated",
                count=len(new_hypotheses),
                from_patterns=len(candidates),
            )

            # ── Tag hypotheses with source detector for meta-learning ────
            # Derive the dominant detector from the pattern candidates
            detector_names = [
                c.source_detector for c in candidates if c.source_detector
            ]
            dominant_detector = (
                max(set(detector_names), key=detector_names.count)
                if detector_names else "unknown"
            )
            for h in new_hypotheses:
                if not h.source_detector:
                    h.source_detector = dominant_detector
                if self._meta_learning is not None:
                    self._meta_learning.record_hypothesis_generated(
                        h.id, h.source_detector,
                    )

            # ── Feed causal hypotheses to Kairos for validation ──────────
            await self._feed_causal_hypotheses_to_kairos(new_hypotheses)

            # ── Apply Logos MDL scoring to evaluate quality ───────────────
            await self._apply_logos_mdl_scoring(new_hypotheses)

            # ── Use Telos to prioritise by effective_I impact ────────────
            self._apply_telos_hypothesis_priority(new_hypotheses)

            # ── RE training: hypothesis generation decision ──
            for h in new_hypotheses:
                await self._emit_re_training_example(
                    category="hypothesis_generation",
                    instruction="Generate falsifiable hypothesis from accumulated pattern candidates.",
                    input_context=f"patterns={len(candidates)}, detector={dominant_detector}",
                    output=f"hypothesis={h.statement[:300]!r}, status={h.status.value if hasattr(h.status, 'value') else h.status}",
                    # TODO(re-quality): replace with evidence_score after hypothesis evaluation
                    outcome_quality=h.evidence_score if hasattr(h, "evidence_score") else 0.5,
                    reasoning_trace=h.formal_test[:200] if hasattr(h, "formal_test") and h.formal_test else "",
                )

            # ── Emit EVO_HYPOTHESIS_CREATED for each new hypothesis ──
            await self._emit_hypothesis_lifecycle_events(new_hypotheses, "created")

        except Exception as exc:
            self._logger.error("hypothesis_generation_safe_failed", error=str(exc))

    async def _check_hypothesis_budget_skipped(self) -> bool:
        """
        Return True if the LLM budget is at a level that would cause hypothesis
        generation to be skipped (YELLOW or RED tier for a low-priority system).
        """
        try:
            from clients.optimized_llm import OptimizedLLMProvider

            if isinstance(self._llm, OptimizedLLMProvider):
                return not await self._llm.should_use_llm(
                    "evo.hypothesis", estimated_tokens=1200
                )
        except Exception:
            pass
        return False

    async def _handle_hypothesis_degradation(self, skipped_pattern_count: int) -> None:
        """
        Emit EVO_DEGRADED and, after 10 consecutive skips, escalate via Thymos.
        Called fire-and-forget when hypothesis generation is budget-skipped.
        """
        consecutive = self._consecutive_hypothesis_skips

        # Estimate recovery: GREEN tier returns when budget refills.
        # Use a conservative 30-minute estimate.
        estimated_recovery_s = 1800.0

        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EVO_DEGRADED,
                    source_system="evo",
                    data={
                        "reason": "llm_budget_exhausted",
                        "skipped_pattern_count": skipped_pattern_count,
                        "consecutive_skips": consecutive,
                        "estimated_recovery_time_s": estimated_recovery_s,
                    },
                ))
            except Exception as exc:
                self._logger.debug("evo_degraded_emit_failed", error=str(exc))

        if consecutive >= 10 and self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.INCIDENT_DETECTED,
                    source_system="evo",
                    data={
                        "incident_class": "resource_exhaustion",
                        "severity": "high",
                        "fingerprint": "evo:cognitive_degradation:budget_exhausted",
                        "source_system": "evo",
                        "error_type": "CognitiveDegradation",
                        "error_message": (
                            f"Evo hypothesis generation has been skipped for "
                            f"{consecutive} consecutive cycles due to LLM budget exhaustion. "
                            f"Learning loop is impaired."
                        ),
                        "context": {
                            "consecutive_skips": consecutive,
                            "skipped_pattern_count": skipped_pattern_count,
                            "estimated_recovery_s": estimated_recovery_s,
                        },
                        "affected_systems": ["evo"],
                        "blast_radius": 0.4,
                    },
                ))
                self._logger.warning(
                    "evo_cognitive_degradation_escalated_to_thymos",
                    consecutive=consecutive,
                )
            except Exception as exc:
                self._logger.error(
                    "evo_thymos_escalation_failed",
                    error=str(exc),
                )

    async def _evaluate_recent_evidence_safe(self) -> None:
        """
        Fire-and-forget evidence sweep: retrieve recent episodes and evaluate
        them against all active hypotheses.

        Uses active hypothesis statements as queries to find evidence that is
        specifically relevant — not a random sample. This is active evidence
        seeking: the learning system goes looking for what it needs.
        """
        if not self._initialized or self._memory is None:
            return
        if self._hypothesis_engine is None:
            return
        try:
            # Build a query from TESTING hypotheses only (supported/refuted don't need more evidence)
            active = self._hypothesis_engine.get_all_active()
            if not active:
                return

            # Only evaluate hypotheses actively being tested
            testing = [h for h in active if h.status == HypothesisStatus.TESTING]
            if not testing:
                return

            # Sample up to 2 of the testing hypotheses (was 3 of all)
            sample = testing[:2]
            seen_episodes: set[str] = set()
            for h in sample:
                query = h.statement[:200] if h.statement else ""
                if not query:
                    continue
                response = await self._memory.retrieve(
                    query_text=query,
                    max_results=3,
                    salience_floor=0.0,
                )
                for trace in response.traces:
                    trace_id = getattr(trace, "node_id", None) or ""
                    if trace_id in seen_episodes:
                        continue
                    seen_episodes.add(trace_id)
                    episode = _trace_to_episode(trace)
                    await self.process_episode(episode)
        except Exception as exc:
            self._logger.warning("evidence_sweep_failed", error=str(exc))

    async def _run_consolidation_now(self) -> ConsolidationResult:
        """Execute consolidation and update counters."""
        assert self._orchestrator is not None
        # Checkpoint PatternContext to Redis before consolidation resets it (Spec §III gap fix)
        await self._save_pattern_context_checkpoint()
        try:
            result = await self._orchestrator.run(self._pattern_context)
            self._cycles_since_consolidation = 0
            self._total_consolidations += 1
            self._last_consolidation_completed_at = time.monotonic()

            # Push learned head-weight adjustments to Atune's meta-attention
            # Evo tunes parameters like "atune.head.novelty.weight" — extract
            # the deltas and forward them so they actually take effect.
            await self._push_atune_head_weights()

            # Push learned personality adjustments to Voxis
            # Evo tunes parameters like "voxis.personality.warmth" — extract
            # the deltas and forward them so Voxis personality actually evolves.
            await self._push_voxis_personality()

            # If Evo discovered systematic mis-predictions in interoceptive
            # transitions during consolidation, update Soma's dynamics matrix
            await self._push_soma_dynamics_update(result)

            # Push any learned free energy budget parameters to Nova's
            # deliberation engine so the surprise tolerance evolves over time.
            self._push_nova_fe_budget_params()

            # Push learned EFE component weights to Nova's EFE evaluator.
            # Evo tunes nova.efe.* parameters but they were never forwarded,
            # so the learned weights had zero effect on policy selection.
            await self._push_nova_efe_weights()

            # Sync learnable belief half-life parameters to Neo4j.
            # ParameterTuner may have updated belief.halflife.* keys during
            # Phase 5; propagate them to the BeliefAgingScanner registry and
            # to existing belief nodes so Phase 2.5 uses the learned values.
            if self._belief_aging is not None and self._parameter_tuner is not None:
                try:
                    all_params = self._parameter_tuner.get_all_parameters()
                    await self._belief_aging.sync_halflife_overrides(all_params)
                except Exception as exc:
                    self._logger.warning("halflife_sync_failed", error=str(exc))

            # Annotate hypotheses with Telos topology contribution so
            # consolidation output reflects drive-aware prioritisation.
            self._annotate_telos_priority(result)

            # ── Speciation subsystem consolidation steps ──────────────────

            # Generate structural hypotheses (no LLM cost)
            try:
                structural_h = await self.run_structural_hypothesis_generation()
                if structural_h:
                    self._logger.info(
                        "structural_hypotheses_consolidated",
                        count=len(structural_h),
                    )
                    # Structural hypotheses represent genuinely new capabilities
                    for sh in structural_h:
                        category = sh.category.value if hasattr(sh.category, "value") else str(sh.category)
                        await self._emit_capability_emerged(
                            capability_name=sh.statement[:100],
                            source_hypotheses=[sh.id],
                            novelty_score=getattr(sh, "novelty_score", 0.5),
                            domain=category,
                        )
            except Exception as exc:
                self._logger.warning("structural_hypothesis_consolidation_failed", error=str(exc))

            # Run evolutionary pressure (selection + pruning)
            try:
                await self.run_evolutionary_pressure()
            except Exception as exc:
                self._logger.warning("evolutionary_pressure_failed", error=str(exc))

            # Generate epistemic intents (curiosity-driven exploration)
            try:
                intents = await self.generate_epistemic_intents()
                if intents:
                    self._logger.info(
                        "epistemic_intents_consolidated",
                        count=len(intents),
                    )
            except Exception as exc:
                self._logger.warning("epistemic_intent_generation_failed", error=str(exc))

            # Run self-modification cycle (meta-meta-learning)
            try:
                if self._self_modification is not None:
                    # Collect detector stats from meta-learning
                    detector_stats: list[dict[str, Any]] = []
                    if self._meta_learning is not None:
                        for name, outcomes in self._meta_learning._detector_outcomes.items():
                            gen = self._meta_learning._detector_generation_counts.get(name, 0)
                            survived = sum(1 for o in outcomes if o in ("supported", "integrated"))
                            eff = survived / max(1, gen)
                            detector_stats.append({
                                "name": name, "generated": gen,
                                "survived": survived, "effectiveness": eff,
                            })

                    # Consolidation metrics
                    consol_metrics = {
                        "throughput": (
                            result.hypotheses_evaluated / max(1.0, result.duration_ms / 3600000)
                        ),
                        "success_rate": (
                            result.hypotheses_integrated / max(1, result.hypotheses_evaluated)
                        ),
                        "schema_rate": result.schemas_induced,
                    }

                    mods = await self._self_modification.run_self_modification_cycle(
                        detector_stats=detector_stats,
                        hypothesis_outcomes={},
                        consolidation_metrics=consol_metrics,
                    )
                    if mods:
                        self._logger.info(
                            "self_modifications_applied",
                            count=len(mods),
                            types=[m.proposal_type for m in mods],
                        )
            except Exception as exc:
                self._logger.warning("self_modification_cycle_failed", error=str(exc))

            await self._emit_evolutionary_observable(
                observable_type="experiment_completed",
                value=float(result.hypotheses_integrated),
                is_novel=True,
                metadata={
                    "consolidation_number": self._total_consolidations,
                    "hypotheses_evaluated": result.hypotheses_evaluated,
                    "hypotheses_integrated": result.hypotheses_integrated,
                    "schemas_induced": result.schemas_induced,
                    "duration_ms": result.duration_ms,
                },
            )

            # Loop 6: Emit FITNESS_OBSERVABLE_BATCH for Benchmarks
            await self._emit_fitness_observable_batch(result)

            # Emit EVO_CONSOLIDATION_COMPLETE
            await self._emit_consolidation_complete(result)

            # Check for speciation-level behavioral shifts
            await self._check_and_emit_speciation_event(result)

            return result
        except Exception as exc:
            self._logger.error("consolidation_run_failed", error=str(exc))
            return ConsolidationResult()

    async def _push_atune_head_weights(self) -> None:
        """
        Extract atune.head.* parameters from the tuner and push them to Atune.

        Evo learns optimal head weights like "atune.head.novelty.weight" via
        parameter hypotheses. The tuner stores the current values, but Atune's
        MetaAttentionController needs the *deltas from default* to apply them.
        """
        if self._atune is None or self._parameter_tuner is None:
            return

        from systems.evo.types import PARAMETER_DEFAULTS

        all_params = self._parameter_tuner.get_all_parameters()
        adjustments: dict[str, float] = {}

        for param_name, current_value in all_params.items():
            if not param_name.startswith("atune.head."):
                continue
            # Extract head name: "atune.head.novelty.weight" → "novelty"
            parts = param_name.split(".")
            if len(parts) >= 3:
                head_name = parts[2]
                default_value = PARAMETER_DEFAULTS.get(param_name, current_value)
                delta = current_value - default_value
                if abs(delta) > 0.001:
                    adjustments[head_name] = delta

        if adjustments and self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EVO_PARAMETER_ADJUSTED,
                    source_system="evo",
                    data={"target_system": "atune", "adjustments": adjustments},
                ))
                self._logger.info(
                    "atune_head_weights_pushed",
                    adjustments={k: round(v, 4) for k, v in adjustments.items()},
                )
            except Exception:
                self._logger.debug("atune_head_push_failed", exc_info=True)

    async def _push_voxis_personality(self) -> None:
        """
        Extract voxis.personality.* parameters from the tuner and push them to Voxis.

        Evo learns personality adjustments like "voxis.personality.warmth" via
        parameter hypotheses. The tuner stores the current values; Voxis needs
        the deltas from defaults applied via update_personality().
        """
        if self._voxis is None or self._parameter_tuner is None:
            return

        from systems.evo.types import PARAMETER_DEFAULTS

        all_params = self._parameter_tuner.get_all_parameters()
        personality_deltas: dict[str, float] = {}

        for param_name, current_value in all_params.items():
            if not param_name.startswith("voxis.personality."):
                continue
            # Extract dimension: "voxis.personality.warmth" → "warmth"
            parts = param_name.split(".")
            if len(parts) >= 3:
                dimension = parts[2]
                default_value = PARAMETER_DEFAULTS.get(param_name, current_value)
                delta = current_value - default_value
                if abs(delta) > 0.001:
                    personality_deltas[dimension] = delta

        if personality_deltas and self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EVO_PARAMETER_ADJUSTED,
                    source_system="evo",
                    data={"target_system": "voxis", "adjustments": personality_deltas},
                ))
                self._logger.info(
                    "voxis_personality_pushed",
                    dimensions={k: round(v, 4) for k, v in personality_deltas.items()},
                )
            except Exception:
                self._logger.debug("voxis_personality_push_failed", exc_info=True)

    def _push_nova_fe_budget_params(self) -> None:
        """
        Extract nova.fe_budget.* parameters from the tuner and push them to Nova.

        Evo learns optimal surprise budget parameters via parameter hypotheses.
        The tuner stores the current values; Nova's DeliberationEngine needs them
        applied so the free energy pressure valve adapts over time.
        """
        if self._nova is None or self._parameter_tuner is None:
            return

        from systems.evo.types import PARAMETER_DEFAULTS

        all_params = self._parameter_tuner.get_all_parameters()
        budget_nats = all_params.get(
            "nova.fe_budget.budget_nats",
            PARAMETER_DEFAULTS.get("nova.fe_budget.budget_nats", 5.0),
        )
        threshold_fraction = all_params.get(
            "nova.fe_budget.threshold_fraction",
            PARAMETER_DEFAULTS.get("nova.fe_budget.threshold_fraction", 0.8),
        )

        try:
            engine = getattr(self._nova, "_deliberation_engine", None)
            if engine is not None and hasattr(engine, "update_fe_budget_params"):
                engine.update_fe_budget_params(
                    budget_nats=budget_nats,
                    threshold_fraction=threshold_fraction,
                )
                self._logger.info(
                    "nova_fe_budget_params_pushed",
                    budget_nats=round(budget_nats, 2),
                    threshold_fraction=round(threshold_fraction, 2),
                )
        except Exception:
            self._logger.debug("nova_fe_budget_push_failed", exc_info=True)

    async def _push_nova_efe_weights(self) -> None:
        """
        Extract nova.efe.* parameters from the tuner and push them to Nova.

        Evo tunes EFE component weights like "nova.efe.pragmatic" but these
        were never forwarded to the EFE evaluator, so learned weights had no
        effect on policy selection. Nova.update_efe_weights() accepts a dict
        of component name → weight.
        """
        if self._nova is None or self._parameter_tuner is None:
            return

        # nova.efe.cognition_cost maps to "cognition_cost" key in EFEWeights
        _param_to_key = {
            "nova.efe.pragmatic": "pragmatic",
            "nova.efe.epistemic": "epistemic",
            "nova.efe.constitutional": "constitutional",
            "nova.efe.feasibility": "feasibility",
            "nova.efe.risk": "risk",
            "nova.efe.cognition_cost": "cognition_cost",
        }

        all_params = self._parameter_tuner.get_all_parameters()
        new_weights: dict[str, float] = {}
        for param_name, key in _param_to_key.items():
            if param_name in all_params:
                new_weights[key] = all_params[param_name]

        if new_weights and self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EVO_PARAMETER_ADJUSTED,
                    source_system="evo",
                    data={"target_system": "nova", "adjustments": new_weights},
                ))
                self._logger.info(
                    "nova_efe_weights_pushed",
                    weights={k: round(v, 4) for k, v in new_weights.items()},
                )
            except Exception:
                self._logger.debug("nova_efe_weights_push_failed", exc_info=True)

    def _annotate_telos_priority(self, result: ConsolidationResult) -> None:
        """
        Emit hypothesis data to Telos for drive-topology prioritisation.

        Telos consumes EVO_HYPOTHESIS_CREATED events and maintains its own
        rankings. We emit the current hypothesis batch here so Telos can
        update its topology model; rankings are applied in the next cycle.
        """
        # Telos interaction is now purely event-driven (no direct method call).
        # Rankings received back from Telos via _telos_hypothesis_rankings cache
        # are applied in _apply_telos_hypothesis_priority().
        pass

    async def _push_soma_dynamics_update(self, result: ConsolidationResult) -> None:
        """
        Inject Evo's cognitive load as an interoceptive signal into Soma.

        Soma's allostatic model needs to know when the learning system is under
        pressure (high hypothesis density, elevated error rate, stalled consolidation)
        so it can modulate the organism's arousal and urgency accordingly.

        Uses inject_external_stress() — the established synchronous injection path —
        with a composite stress value derived from:
          - hypothesis_density: ratio of active hypotheses to capacity cap (50)
          - evidence_evaluation_rate: how heavily the evidence sweep ran last cycle
          - stalled_consolidation: +0.3 if consolidation has never completed
        """
        if self._event_bus is None:
            return

        try:
            # Hypothesis density: how full is the active hypothesis registry?
            active_count = 0
            if self._hypothesis_engine is not None:
                active_count = len(self._hypothesis_engine.get_all_active())

            from systems.evo.types import VELOCITY_LIMITS
            capacity = int(VELOCITY_LIMITS.get("max_active_hypotheses", 50))
            hypothesis_density = min(1.0, active_count / max(1, capacity))

            # Evidence evaluation load: consecutive hypothesis skips mean cognitive stress
            skip_stress = min(1.0, self._consecutive_hypothesis_skips / 10.0)

            # Consolidation stall: if learning has never completed, add base stress
            stall_stress = 0.3 if self._last_consolidation_completed_at is None else 0.0

            cognitive_stress = min(
                1.0,
                hypothesis_density * 0.5 + skip_stress * 0.35 + stall_stress,
            )

            if self._event_bus is not None:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SOMATIC_MODULATION_SIGNAL,
                    source_system="evo",
                    data={"external_stress": cognitive_stress, "source": "evo_cognitive_load"},
                ))
            self._logger.debug(
                "evo_soma_stress_injected",
                cognitive_stress=round(cognitive_stress, 3),
                hypothesis_density=round(hypothesis_density, 3),
                skip_stress=round(skip_stress, 3),
            )
        except Exception:
            self._logger.debug("soma_stress_inject_failed", exc_info=True)

    # ─── Kairos Causal Hypothesis Feeding ────────────────────────────────────

    async def _feed_causal_hypotheses_to_kairos(
        self, hypotheses: list[Any]
    ) -> None:
        """
        Feed causal hypotheses to Kairos as pre-seeded correlation candidates.

        Evo generates hypotheses from patterns; those that make causal claims
        (WORLD_MODEL category with causal keywords) are forwarded to Kairos
        for independent validation through its 4-stage causal mining pipeline.
        When Kairos validates the direction, it emits
        KAIROS_CAUSAL_DIRECTION_ACCEPTED which we handle above to boost the
        hypothesis evidence score.
        """
        if self._kairos is None:
            return

        from systems.evo.types import HypothesisCategory

        causal_keywords = [
            "causes", "leads to", "produces", "results in",
            "drives", "triggers", "induces", "generates",
            "increases", "decreases", "affects",
        ]

        fed = 0
        evo_pipeline = getattr(self._kairos, "_evo_pipeline", None)
        if evo_pipeline is None:
            return

        for h in hypotheses:
            if h.category != HypothesisCategory.WORLD_MODEL:
                continue
            stmt_lower = h.statement.lower()
            if not any(kw in stmt_lower for kw in causal_keywords):
                continue

            candidate = evo_pipeline.process_evolution_candidate({
                "hypothesis_id": h.id,
                "hypothesis_statement": h.statement,
                "category": h.category.value,
                "confidence": h.evidence_score,
            })
            if candidate is not None:
                fed += 1

        if fed:
            self._logger.info("causal_hypotheses_fed_to_kairos", count=fed)

    # ─── Logos MDL Scoring ─────────────────────────────────────────────────────

    async def _apply_logos_mdl_scoring(self, hypotheses: list[Any]) -> None:
        """
        Use Logos MDL estimator to score hypothesis quality.

        MDL (Minimum Description Length) penalises complex hypotheses that
        explain little. Simple hypotheses that predict many observations get
        high compression ratios and survive; verbose ones that cover the same
        ground get penalised.

        The complexity_penalty field on Hypothesis is updated to reflect
        the MDL cost — hypotheses with poor compression ratios need stronger
        evidence to reach SUPPORTED status.
        """
        if self._logos is None:
            return

        mdl = getattr(self._logos, "mdl_estimator", None)
        if mdl is None:
            return

        scored = 0
        for h in hypotheses:
            try:
                # Build a lightweight adapter satisfying HypothesisProtocol
                adapter = _HypothesisMDLAdapter(h)
                mdl_score = await mdl.score_hypothesis(adapter)

                # Map compression ratio to complexity penalty:
                # High compression (>5.0) → low penalty (0.05)
                # Low compression (<1.0) → high penalty (0.3)
                if mdl_score.compression_ratio > 0:
                    penalty = max(0.05, min(0.3, 1.0 / mdl_score.compression_ratio * 0.1))
                else:
                    penalty = 0.3
                h.complexity_penalty = penalty
                scored += 1
            except Exception:
                pass  # MDL scoring is advisory; failures are non-fatal

        if scored:
            self._logger.debug("logos_mdl_scoring_applied", scored=scored)

    # ─── Telos Hypothesis Prioritisation (during generation) ──────────────────

    def _apply_telos_hypothesis_priority(self, hypotheses: list[Any]) -> None:
        """
        Apply cached Telos hypothesis priority rankings to newly generated hypotheses.

        Telos publishes rankings via Synapse events; EvoService caches them in
        self._telos_hypothesis_rankings (populated by the event handler).
        We apply the cached boost here without calling Telos directly.
        """
        rank_cache: dict[str, Any] = getattr(self, "_telos_hypothesis_rankings", {})
        if not rank_cache:
            return

        try:
            rank_lookup = rank_cache
            boosted = 0
            for h in hypotheses:
                ranking = rank_lookup.get(h.id)
                if ranking is None:
                    continue
                # composite_contribution > 0 means the hypothesis would improve
                # the drive topology; scale the boost proportionally
                composite = getattr(ranking, "composite_contribution", 0.0)
                if composite > 0:
                    boost = min(composite * 0.5, 0.3)
                    h.evidence_score += boost
                    boosted += 1

            if boosted:
                self._logger.debug(
                    "telos_hypothesis_priority_applied",
                    boosted=boosted,
                    total=len(hypotheses),
                )
        except Exception:
            self._logger.debug("telos_hypothesis_priority_failed", exc_info=True)

    async def _handle_new_arxiv_innovation(self, raw_dict: dict[str, Any]) -> None:
        """
        Take a single technique dict from ArxivScientist, translate it into
        an EvolutionProposal, and dispatch it to the Simula governance pipeline.

        Every arXiv proposal enters as ADD_SYSTEM_CAPABILITY (governance-gated),
        so it cannot land autonomously — Equor must approve first.
        """
        if self._simula is None:
            self._logger.warning(
                "arxiv_innovation_skipped",
                reason="simula_not_wired",
                paper_id=raw_dict.get("paper_id", "unknown"),
            )
            return

        from systems.simula.proposals.arxiv_translator import ArxivProposalTranslator

        result = await ArxivProposalTranslator().translate_and_dispatch(
            raw_technique=raw_dict,
            simula_service=self._simula,
        )
        status = (
            result.dispatch_result.status.value
            if result.dispatch_result else "no_dispatch"
        )
        self._logger.info(
            "arxiv_innovation_injected",
            paper_id=raw_dict.get("paper_id"),
            technique=raw_dict.get("technique_name"),
            proposal_id=result.proposal.id,
            target_directory=result.target_directory,
            dispatch_status=status,
        )

    async def _arxiv_scan_loop(self) -> None:
        """
        Background loop that runs the daily arXiv scan.

        Fires immediately on startup, then sleeps 24 hours between scans.
        Wrapped in a broad try/except so arXiv API outages or LLM failures
        never crash the main EvoService loop.
        """
        _SCAN_INTERVAL_S = 86400.0  # 24 hours

        while True:
            try:
                if not self._initialized:
                    await asyncio.sleep(60)
                    continue

                self._logger.info("arxiv_scan_loop_tick")
                techniques = await self._arxiv_scientist.run_daily_scan()

                for raw_dict in techniques:
                    try:
                        await self._handle_new_arxiv_innovation(raw_dict)
                    except Exception as exc:
                        self._logger.warning(
                            "arxiv_innovation_handler_failed",
                            paper_id=raw_dict.get("paper_id", "unknown"),
                            error=str(exc),
                        )

                if techniques:
                    self._logger.info(
                        "arxiv_scan_cycle_complete",
                        techniques_dispatched=len(techniques),
                    )

            except asyncio.CancelledError:
                self._logger.info("arxiv_scan_loop_cancelled")
                return
            except Exception as exc:
                # Broad catch: arXiv API outage, LLM hallucination, XML parse
                # failure, network errors — none of these should kill the loop.
                self._logger.error(
                    "arxiv_scan_loop_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

            try:
                await asyncio.sleep(_SCAN_INTERVAL_S)
            except asyncio.CancelledError:
                self._logger.info("arxiv_scan_loop_cancelled")
                return

    async def _consolidation_loop(self) -> None:
        """
        Background loop that triggers consolidation based on time/cycle thresholds.
        Runs indefinitely until cancelled.

        Also checks for stall conditions: if consolidation has not completed in
        2× the expected interval, emits EVO_CONSOLIDATION_STALLED on the bus.
        """
        while True:
            try:
                # Poll every 60 seconds to check if consolidation is due
                await asyncio.sleep(60)

                if not self._initialized or self._orchestrator is None:
                    continue

                # ── Stall detection ───────────────────────────────────────
                now = time.monotonic()
                stall_threshold_s = self._consolidation_expected_interval_s * 2.0
                if self._last_consolidation_completed_at is not None:
                    age_s = now - self._last_consolidation_completed_at
                    if age_s > stall_threshold_s:
                        self._logger.warning(
                            "evo_consolidation_stalled",
                            last_consolidation_ago_s=round(age_s, 0),
                            expected_interval_s=self._consolidation_expected_interval_s,
                            cycles_since_consolidation=self._cycles_since_consolidation,
                        )
                        asyncio.create_task(
                            self._emit_consolidation_stalled(age_s),
                            name="evo_consolidation_stalled_emit",
                        )

                if self._consolidation_in_flight:
                    self._logger.debug("consolidation_still_in_flight_skipping")
                    continue

                # Metabolic gate: skip consolidation under starvation
                if self._starvation_level in ("emergency", "critical"):
                    continue

                if self._orchestrator.should_run(
                    cycle_count=self._total_broadcasts,
                    cycles_since_last=self._cycles_since_consolidation,
                ):
                    self._logger.info(
                        "consolidation_triggered",
                        cycles_since_last=self._cycles_since_consolidation,
                    )
                    self._consolidation_in_flight = True
                    try:
                        await self._run_consolidation_now()
                    finally:
                        self._consolidation_in_flight = False

            except asyncio.CancelledError:
                self._logger.info("consolidation_loop_cancelled")
                return
            except Exception as exc:
                self._logger.error("consolidation_loop_error", error=str(exc))
                await asyncio.sleep(60)

    async def _emit_consolidation_stalled(self, last_consolidation_ago_s: float) -> None:
        """Emit EVO_CONSOLIDATION_STALLED on the event bus."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EVO_CONSOLIDATION_STALLED,
                source_system="evo",
                data={
                    "last_consolidation_ago_s": round(last_consolidation_ago_s, 0),
                    "expected_interval_s": self._consolidation_expected_interval_s,
                    "cycles_since_consolidation": self._cycles_since_consolidation,
                    "total_consolidations": self._total_consolidations,
                },
            ))
        except Exception as exc:
            self._logger.error("evo_consolidation_stalled_emit_failed", error=str(exc))


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _broadcast_to_episode(broadcast: WorkspaceBroadcast) -> Episode:
    """
    Create a minimal Episode from a WorkspaceBroadcast for online scanning.
    The episode is not stored — it is used only for detector input.
    """
    from primitives.common import new_id, utc_now
    from primitives.memory_trace import Episode

    # Extract text from percept content if available
    content_str = ""
    if broadcast.content is not None:
        content_obj = broadcast.content
        # Try to get raw text from Percept.content.raw
        if hasattr(content_obj, "content") and hasattr(content_obj.content, "raw"):
            content_str = str(content_obj.content.raw or "")
        elif hasattr(content_obj, "raw"):
            content_str = str(content_obj.raw or "")

    source = ""
    if broadcast.content is not None and hasattr(broadcast.content, "source"):
        src = broadcast.content.source
        if hasattr(src, "channel"):
            source = f"{getattr(src, 'system', '')}.{src.channel}"

    return Episode(
        id=new_id(),
        event_time=broadcast.timestamp,
        ingestion_time=utc_now(),
        source=source,
        raw_content=content_str[:500],
        summary=content_str[:200],
        salience_composite=broadcast.salience.composite,
        salience_scores=broadcast.salience.scores,
        affect_valence=broadcast.affect.valence,
        affect_arousal=broadcast.affect.arousal,
    )


def _trace_to_episode(trace: Any) -> Episode:
    """Build a minimal Episode from a RetrievalResult for evidence evaluation."""
    from primitives.common import new_id
    from primitives.memory_trace import Episode

    return Episode(
        id=str(getattr(trace, "node_id", new_id())),
        source="memory",
        raw_content=str(getattr(trace, "content", ""))[:500],
        summary=str(getattr(trace, "content", ""))[:200],
        salience_composite=float(getattr(trace, "salience", 0.0)),
        affect_valence=float(getattr(trace, "metadata", {}).get("affect_valence", 0.0)),
        affect_arousal=float(getattr(trace, "metadata", {}).get("affect_arousal", 0.0)),
    )


class _HypothesisMDLAdapter:
    """
    Lightweight adapter that satisfies Logos's HypothesisProtocol for MDL scoring.

    Maps Evo's Hypothesis model to the five properties MDLEstimator.score_hypothesis()
    expects: id, supporting_observations, description, unique_predictive_coverage,
    last_tested, test_frequency.
    """

    __slots__ = ("_h",)

    def __init__(self, hypothesis: Hypothesis) -> None:
        self._h = hypothesis

    @property
    def id(self) -> str:
        return self._h.id

    @property
    def supporting_observations(self) -> list[Any]:
        """Map supporting episode IDs to minimal observation objects."""
        return [_MinimalObs(eid) for eid in self._h.supporting_episodes]

    @property
    def description(self) -> str:
        return self._h.statement

    @property
    def unique_predictive_coverage(self) -> float:
        """Approximate: evidence_score normalized, clamped to [0, 1]."""
        return max(0.0, min(1.0, self._h.evidence_score / 5.0))

    @property
    def last_tested(self) -> Any:
        return self._h.last_evidence_at

    @property
    def test_frequency(self) -> float:
        total = len(self._h.supporting_episodes) + len(self._h.contradicting_episodes)
        return min(total * 0.1, 1.0)


class _MinimalObs:
    """Minimal observation for MDL scoring: just needs a complexity attribute."""

    __slots__ = ("_id",)

    def __init__(self, episode_id: str) -> None:
        self._id = episode_id

    @property
    def complexity(self) -> float:
        return 1.0


# ─── Economic Helper ──────────────────────────────────────────────────────────


def _economic_episode(
    source: str,
    raw_content: str,
    summary: str,
    salience: float = 0.6,
    valence: float = 0.0,
    arousal: float = 0.3,
) -> Episode:
    """Build a minimal Episode from an economic event for online scanning."""
    from primitives.common import new_id, utc_now
    from primitives.memory_trace import Episode

    return Episode(
        id=new_id(),
        ingestion_time=utc_now(),
        source=source,
        raw_content=raw_content[:500],
        summary=summary[:200],
        salience_composite=salience,
        affect_valence=valence,
        affect_arousal=arousal,
    )


# ─── Economic Pattern Detector ────────────────────────────────────────────────


class EconomicPatternDetector(PatternDetector):
    """
    Detects economic patterns in the episodic stream.

    Watches for:
      - Bounty attempt timing clusters (time-of-day patterns)
      - Budget exhaustion preceding bounty failures (pre-allocation signal)
      - Bounty source acceptance rate differences (Algora vs GitHub)

    Emits TEMPORAL and ACTION_SEQUENCE PatternCandidates that flow into
    the normal hypothesis-generation pipeline — no special handling required.

    Implements the PatternDetector interface so it drops into
    build_default_detectors() and _scan_episode_online() without changes.
    """

    name = "economic_pattern"
    window_size = 200
    min_occurrences = 3

    def __init__(self) -> None:
        # {hour_bin: [episode_id, ...]}  — bounty attempts by hour-of-day
        self._bounty_by_hour: dict[int, list[str]] = collections.defaultdict(list)
        # Tracks whether the *previous* economic episode was a budget exhaustion
        self._last_was_budget_exhaustion: bool = False
        # {source_key: (successes, total)}
        self._source_outcomes: dict[str, list[bool]] = collections.defaultdict(list)

    async def scan(
        self,
        episode: Episode,
        context: PatternContext,
    ) -> list[PatternCandidate]:
        """
        Scan one episode and return any newly-triggered economic pattern candidates.
        Must complete in ≤20ms (no I/O, no LLM calls).
        """
        content = (episode.raw_content or "").lower()
        candidates: list[PatternCandidate] = []

        # Only process economic episodes
        _econ_keywords = ("bounty", "revenue", "budget", "runway", "yield", "apy")
        if not any(kw in content for kw in _econ_keywords):
            self._last_was_budget_exhaustion = False
            return candidates

        from primitives.common import utc_now

        now = utc_now()
        hour = now.hour

        # ── Pattern 1: Bounty attempt time-of-day ─────────────────────────────
        if "bounty" in content and ("attempted" in content or "submitted" in content):
            self._bounty_by_hour[hour].append(episode.id)
            count = len(self._bounty_by_hour[hour])
            if count == self.min_occurrences:
                candidates.append(PatternCandidate(
                    type=PatternType.TEMPORAL,
                    elements=[f"bounty_attempt::h{hour}"],
                    count=count,
                    confidence=0.6,
                    examples=list(self._bounty_by_hour[hour]),
                    metadata={
                        "source": "economic_pattern",
                        "hypothesis_hint": (
                            f"Bounty attempts cluster at hour {hour}:00 UTC — "
                            f"is there a time-of-day pattern in success rates?"
                        ),
                    },
                ))

            # ── Pattern 2: Budget exhaustion precedes bounty attempts ──────────
            if self._last_was_budget_exhaustion:
                candidates.append(PatternCandidate(
                    type=PatternType.ACTION_SEQUENCE,
                    elements=["budget_exhausted", "bounty_attempt"],
                    count=1,
                    confidence=0.7,
                    examples=[episode.id],
                    metadata={
                        "source": "economic_pattern",
                        "hypothesis_hint": (
                            "Budget exhaustion preceded a bounty attempt — "
                            "should Oikos pre-allocate compute budget before hunting?"
                        ),
                    },
                ))

            # ── Pattern 3: Source-level acceptance rate ────────────────────────
            source_key: str | None = None
            if "algora" in content:
                source_key = "algora"
            elif "github" in content:
                source_key = "github"
            if source_key is not None:
                is_success = "submitted" in content or "paid" in content
                self._source_outcomes[source_key].append(is_success)
                outcomes = self._source_outcomes[source_key]
                # Emit once we have 5 data points for a source
                if len(outcomes) == 5:
                    rate = sum(1 for v in outcomes if v) / len(outcomes)
                    candidates.append(PatternCandidate(
                        type=PatternType.ACTION_SEQUENCE,
                        elements=[f"bounty_source::{source_key}", "acceptance_rate"],
                        count=len(outcomes),
                        confidence=round(rate, 2),
                        examples=[episode.id],
                        metadata={
                            "source": "economic_pattern",
                            "source_key": source_key,
                            "acceptance_rate": rate,
                            "hypothesis_hint": (
                                f"{source_key.title()} acceptance rate is {rate:.0%} over "
                                f"{len(outcomes)} attempts — compare against other sources."
                            ),
                        },
                    ))

        self._last_was_budget_exhaustion = (
            "budget exhausted" in content or "budget_exhausted" in content
        )

        return candidates


# ─── Genome Export Helpers (Spec 26 SG4) ─────────────────────────────────────


async def _safe_get_drive_scores(equor: Any) -> dict[str, float] | None:
    """Fetch drive alignment scores from Equor without raising."""
    try:
        result = equor.get_drive_alignment_scores()
        if asyncio.iscoroutine(result):
            result = await result
        return result  # type: ignore[return-value]
    except Exception:
        return None


async def _safe_fetch_drift_history(memory: Any) -> list[dict[str, Any]] | None:
    """Fetch last 10 constitutional drift entries from Memory without raising."""
    try:
        records = await memory.query_drift_history(limit=10)
        return records  # type: ignore[return-value]
    except Exception:
        return None
