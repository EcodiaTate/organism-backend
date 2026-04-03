"""
EcodiaOS - Nova Service

The executive function. Nova is where perception becomes intention.

Nova is the bridge between understanding the world (Atune + Memory) and
acting on it (Axon + Voxis). It receives workspace broadcasts, integrates
them with beliefs and goals, formulates possible courses of action, evaluates
them against Expected Free Energy, submits to Equor for constitutional review,
and issues Intents for execution.

Nova is not the boss. Equor can deny its Intents, Axon can fail them,
and the community can override them through governance. Nova proposes;
the organism disposes.

Lifecycle:
  initialize() - loads constitution and drive weights, builds sub-components
  receive_broadcast() - implements BroadcastSubscriber for Atune
  submit_intent() - external API for direct intent submission (test/governance)
  process_outcome() - feedback loop from execution
  shutdown() - graceful teardown
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.affect import AffectState
from primitives.common import DriveAlignmentVector, SystemID
from primitives.re_training import RETrainingExample
from systems.nova.belief_updater import BeliefUpdater
from systems.nova.cognition_cost import (
    CognitionCostCalculator,
    CostRates,
    DecisionImportance,
)
from systems.nova.deliberation_engine import DeliberationEngine
from systems.nova.efe_evaluator import EFEEvaluator
from systems.nova.goal_manager import GoalManager
from systems.nova.goal_store import (
    abandon_stale_goals,
    load_active_goals,
    persist_goal,
    update_goal_status,
)
from systems.nova.intent_router import IntentRouter
from systems.nova.policy_generator import (
    BasePolicyGenerator,
    PolicyGenerator,
)
from systems.nova.types import (
    CounterfactualRecord,
    DecisionRecord,
    EFEWeights,
    Goal,
    GoalSource,
    GoalStatus,
    IntentOutcome,
    PendingIntent,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from config import NovaConfig
    from core.hotreload import NeuroplasticityBus
    from primitives.intent import Intent
    from systems.axon.service import AxonService
    from systems.equor.service import EquorService
    from systems.fovea.types import ActiveGoalSummary, WorkspaceBroadcast
    from systems.memory.service import MemoryService
    from systems.voxis.service import VoxisService

logger = structlog.get_logger()

# Executor action types that represent heavy external work.  When any pending
# intent contains one of these, the heartbeat considers the organism "busy" and
# skips the hunger-drive cycle.  Lightweight internal executors (store_insight,
# observe, query_memory …) must NOT appear here - blocking on them causes the
# organism to starve while daydreaming.
_HEAVY_EXECUTORS: frozenset[str] = frozenset({
    "hunt_bounties", "executor.hunt_bounties",
    "bounty_hunt", "executor.bounty_hunt",
    "solve_bounty", "executor.solve_bounty",
    "monitor_prs", "executor.monitor_prs",
    "spawn_child", "executor.spawn_child",
    "deploy_asset", "executor.deploy_asset",
    "defi_yield", "executor.defi_yield",
    "wallet_transfer", "executor.wallet_transfer",
    "request_funding", "executor.request_funding",
})


class NovaService:
    """
    Decision & Planning system.

    Implements BroadcastSubscriber (Atune workspace protocol):
        system_id: str
        async def receive_broadcast(broadcast: WorkspaceBroadcast) -> None

    Dependencies:
        memory  - for constitution, self-model retrieval, procedure lookup
        equor   - for constitutional review of every Intent
        voxis   - for expression routing of approved Intents
        llm     - for policy generation and EFE estimation
        config  - NovaConfig
    """

    system_id: str = "nova"

    def __init__(
        self,
        memory: MemoryService,
        equor: EquorService,
        voxis: VoxisService,
        llm: LLMProvider,
        config: NovaConfig,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
    ) -> None:
        self._memory = memory
        self._equor = equor
        self._voxis = voxis
        self._llm = llm
        self._config = config
        self._bus = neuroplasticity_bus
        self._logger = logger.bind(system="nova")

        # Instance metadata
        self._instance_name: str = "EOS"
        self._drive_weights: dict[str, float] = {
            "coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0
        }

        # Sub-components - built in initialize()
        self._belief_updater: BeliefUpdater = BeliefUpdater()
        self._goal_manager: GoalManager | None = None
        self._policy_generator: BasePolicyGenerator | None = None
        self._efe_evaluator: EFEEvaluator | None = None
        self._deliberation_engine: DeliberationEngine | None = None
        self._intent_router: IntentRouter | None = None
        # ActionTypeRegistry - built in initialize(), shared with PolicyGenerator
        from systems.nova.action_type_registry import ActionTypeRegistry
        self._action_type_registry: ActionTypeRegistry = ActionTypeRegistry()

        # State
        self._pending_intents: dict[str, PendingIntent] = {}
        self._pending_counterfactuals: dict[str, list[CounterfactualRecord]] = {}
        self._current_affect: AffectState = AffectState.neutral()
        self._current_conversation_id: str | None = None

        # Goal embedding cache: goal_id → embedding
        self._goal_embeddings: dict[str, list[float]] = {}
        self._embed_fn: Any = None  # Set via set_embed_fn()
        # Callback to push goal updates to Atune
        self._goal_sync_callback: Any = None  # Set via set_goal_sync_callback()

        # Axon - may be wired before initialize() via set_axon(); applied at end of init.
        self._axon: Any = None

        # Logos - world model and compression metrics for EFE grounding
        self._logos: Any = None

        # Soma for allostatic signal reading
        self._soma: Any = None

        # Evo - for triggering emergency consolidation when FE budget exhausts
        # NOTE: Direct Evo reference; Thompson weights now use EVO_THOMPSON_QUERY event.
        self._evo: Any = None

        # Pending Thompson weight query futures: request_id → asyncio.Future
        # Resolved by _on_thompson_response() when EVO_THOMPSON_RESPONSE arrives.
        self._thompson_futures: dict[str, "asyncio.Future[dict[str, dict[str, float]]]"] = {}

        # Synapse event bus - for emitting BELIEF_UPDATED and POLICY_SELECTED
        self._synapse: Any = None

        # RE service reference - used for late-wiring when vLLM becomes
        # available after startup (RE_ENGINE_STATUS_CHANGED handler).
        self._re_service: Any = None

        # Oikos for economic balance reading (heartbeat hunger check)
        self._oikos: Any = None
        # Round-robin cursor for satiated growth goal pool
        self._growth_goal_cursor: int = 0

        # Rhythm-adaptive state (updated by Synapse event bus)
        self._rhythm_state: str = "normal"
        self._rhythm_drive_modulation: dict[str, float] = {}

        # Observability counters
        self._total_broadcasts: int = 0
        self._total_fast_path: int = 0
        self._total_slow_path: int = 0
        self._total_do_nothing: int = 0
        self._total_intents_issued: int = 0
        self._total_intents_approved: int = 0
        self._total_intents_blocked: int = 0
        self._total_outcomes_success: int = 0
        self._total_outcomes_failure: int = 0
        self._decision_records: list[DecisionRecord] = []
        self._max_decision_records: int = 100

        # ── Organism telemetry (from ORGANISM_TELEMETRY event) ───────────────
        # Stores the most recent OrganismTelemetry broadcast from Synapse.
        # Injected as natural-language context into slow-path policy generation.
        self._organism_telemetry: "Any | None" = None

        # ── Deliberation loop liveness tracking ───────────────────────────
        # Monotonic timestamp of the last completed deliberation cycle.
        # None until the first broadcast arrives.
        self._last_deliberation_at: float | None = None
        # Maximum seconds between deliberations before health degrades.
        self._deliberation_stale_threshold_s: float = 300.0  # 5 minutes

        # ── Budget-exhaustion / degradation tracking ──────────────────────
        # Consecutive decisions made under budget exhaustion (heuristic mode).
        self._consecutive_budget_exhausted: int = 0
        # Thymos reference - wired via set_thymos() if called.
        self._thymos: Any = None

        # Thread - narrative identity; receives THREAD_COMMIT_REQUEST events (Gap 4)
        self._thread: Any = None

        # True once initialize() completes successfully. Used by health() to
        # distinguish "not yet initialised" from "initialised but idle".
        self._initialized: bool = False

        # ── Metabolic gating ──────────────────────────────────────────────
        self._starvation_level: str = "nominal"

        # ── Motor degradation flag (Loop 2) ─────────────────────────────
        self._motor_degraded: bool = False

        # ── RE low-confidence tracking (RE_TRAINING_REQUESTED) ──────────
        # Consecutive RE_DECISION_OUTCOME events with success_rate < 0.50.
        # After 5 consecutive readings, Nova emits RE_TRAINING_REQUESTED to
        # ask ContinualLearningOrchestrator to retrain urgently.
        self._re_low_confidence_count: int = 0
        # Cooldown (in theta cycles) before Nova may re-request a budget expansion.
        # Prevents spamming Equor when Axon is repeatedly budget-exceeded.
        self._budget_expansion_cooldown: int = 0

        # ── Elastic FE budget (2026-03-08) ────────────────────────────
        # Goal criticality can request more compute for a fixed number of cycles.
        # Multipliers:
        #   low-priority goal  (< 0.5)  → 0.7  (save compute)
        #   normal goal        (0.5–0.8) → 1.0  (baseline)
        #   high-priority goal (≥ 0.8)  → 1.5  (50% more compute; self-authorised)
        #   critical/existential goal   → 2.0  (double; requires Equor approval)
        # Multiplier resets to 1.0 at cycle end.
        self._budget_multiplier: float = 1.0
        # How many more cycles the current multiplier persists.
        self._budget_multiplier_cycles_remaining: int = 0
        # Cooldown (cycles) before Nova may request another compute expansion.
        self._compute_budget_expansion_cooldown: int = 0
        # Per-goal FE spend tracking: goal_id → total nats spent while goal active.
        self._fe_spent_per_goal: dict[str, float] = {}
        # Baseline budget_nats before multiplier is applied (restored after each cycle).
        self._fe_budget_baseline_nats: float = 5.0

        # ── Axon capability cache (autonomy audit 8 Mar 2026) ──────────
        # Updated every theta cycle via AXON_CAPABILITY_SNAPSHOT subscription.
        # Nova uses this to prune infeasible actions during policy evaluation.
        self._axon_capabilities: dict[str, Any] = {}
        self._axon_executor_index: dict[str, dict[str, Any]] = {}  # action_type → executor info
        self._axon_budget_remaining: int = 5
        self._axon_is_sleeping: bool = False

        # ── Policy effectiveness tracking (autonomy audit) ─────────────
        # Per-policy-class success/failure counts for self-awareness.
        self._policy_effectiveness: dict[str, dict[str, int]] = {}  # policy_name → {success, failure, total}

        # ── Equor rejection pattern tracking (autonomy audit) ──────────
        self._equor_rejections: list[dict[str, Any]] = []  # last 100 rejections
        self._equor_rejection_patterns: dict[str, int] = {}  # pattern → count

        # ── Oikos economic state cache (autonomy audit) ────────────────
        self._cached_balance_usd: float = 0.0
        self._cached_burn_rate_usd_per_hour: float = 0.0
        self._cached_economic_state_ts: float = 0.0

        # ── Kairos causal knowledge cache (2026-03-08) ────────────────
        # Causal invariants discovered by Kairos and injected into Nova's
        # slow-path LLM context. Stored in-memory only (rebuilt as Kairos
        # re-emits on each pipeline run). Max 50 entries, sorted confidence desc.
        self._causal_knowledge: list[dict[str, Any]] = []

        # ── Skia VitalityCoordinator modulation ───────────────────────
        self._modulation_halted: bool = False

        # ── Input channels (market discovery) ─────────────────────────────
        # Lazily constructed in initialize(); None until then.
        from systems.nova.input_channels import InputChannelRegistry  # noqa: PLC0415
        self._input_channels: InputChannelRegistry = InputChannelRegistry()
        self._opportunity_fetch_task: asyncio.Task[None] | None = None
        self._channel_health_task: asyncio.Task[None] | None = None

        # ── Proactive opportunity scanner (Step 1–6, 9 Mar 2026) ──────────
        # OpportunityScanner runs every 30 min, ranks opportunities against drives,
        # auto-injects high-confidence ones as goals, stores others in backlog.
        from systems.nova.opportunity_scanner import OpportunityScanner  # noqa: PLC0415
        self._opportunity_scanner: OpportunityScanner = OpportunityScanner()
        self._opportunity_scan_task: asyncio.Task[None] | None = None
        # Ranked backlog of opportunities not yet converted to goals.
        # Included as deliberation context in next slow-path cycle.
        # Max size controlled by opportunity_scanner.MAX_BACKLOG_SIZE.
        self._opportunity_backlog: list[Any] = []

        # ── Self-Modification Layer (Spec 10 §SM, 9 Mar 2026) ─────────────
        # CapabilityAuditor monitors NOVEL_ACTION_REQUESTED + AXON_EXECUTION_RESULT
        # events and emits CAPABILITY_GAP_IDENTIFIED when a gap crosses threshold.
        # SelfModificationPipeline receives the gap, deliberates alignment, emits
        # SELF_MODIFICATION_PROPOSED, awaits Equor, generates code via Simula,
        # hot-deploys via HotDeployment, then tests and trains on outcome.
        # Both are wired via set_capability_auditor() / set_self_modification_pipeline()
        # from core/registry.py after all dependencies are available.
        self._capability_auditor: Any = None
        self._self_modification_pipeline: Any = None

    # ─── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Load constitution and drive weights from Memory.
        Build all sub-components.
        """
        self._logger.info("nova_initializing")

        # Load instance name and constitution
        self_node = await self._memory.get_self()
        if self_node is not None:
            self._instance_name = self_node.name

        constitution = await self._memory.get_constitution()
        if constitution and "drives" in constitution:
            drives = constitution["drives"]
            self._drive_weights = {
                "coherence": float(drives.get("coherence", 1.0)),
                "care": float(drives.get("care", 1.0)),
                "growth": float(drives.get("growth", 1.0)),
                "honesty": float(drives.get("honesty", 1.0)),
            }

        # Build sub-components
        self._goal_manager = GoalManager(
            max_active_goals=self._config.max_active_goals,
        )

        self._policy_generator = PolicyGenerator(
            llm=self._llm,
            instance_name=self._instance_name,
            max_policies=self._config.max_policies_per_deliberation,
            timeout_ms=self._config.slow_path_timeout_ms - 2000,  # Leave 2s for EFE + Equor
        )
        # Wire the runtime ActionTypeRegistry so the LLM sees dynamic action types.
        self._policy_generator.set_action_type_registry(self._action_type_registry)

        # Build cognition cost calculator from config
        self._cost_calculator: CognitionCostCalculator | None = None
        if self._config.enable_cognition_budgeting:
            cost_rates = CostRates(
                llm_input_per_token=self._config.cost_llm_input_per_1m_tokens / 1_000_000,
                llm_output_per_token=self._config.cost_llm_output_per_1m_tokens / 1_000_000,
                gpu_hourly_rate=self._config.cost_gpu_hourly_rate,
                db_cost_per_ms=self._config.cost_db_per_ms,
                io_cost_per_gb=self._config.cost_io_per_gb,
            )
            custom_budgets = {
                DecisionImportance.LOW: self._config.cognition_budget_low,
                DecisionImportance.MEDIUM: self._config.cognition_budget_medium,
                DecisionImportance.HIGH: self._config.cognition_budget_high,
                DecisionImportance.CRITICAL: self._config.cognition_budget_critical,
            }
            self._cost_calculator = CognitionCostCalculator(rates=cost_rates)
            # Store custom budgets for the deliberation engine
            self._cognition_budgets = custom_budgets
            self._logger.info(
                "cognition_cost_calculator_initialized",
                rates={
                    "llm_input_per_1m": self._config.cost_llm_input_per_1m_tokens,
                    "llm_output_per_1m": self._config.cost_llm_output_per_1m_tokens,
                    "gpu_hourly": self._config.cost_gpu_hourly_rate,
                },
                budgets=custom_budgets,
                lambda_weight=self._config.efe_weight_cognition_cost,
            )

        self._efe_evaluator = EFEEvaluator(
            llm=self._llm,
            weights=EFEWeights(
                pragmatic=self._config.efe_weight_pragmatic,
                epistemic=self._config.efe_weight_epistemic,
                constitutional=self._config.efe_weight_constitutional,
                feasibility=self._config.efe_weight_feasibility,
                risk=self._config.efe_weight_risk,
                cognition_cost=self._config.efe_weight_cognition_cost,
            ),
            use_llm_estimation=True,
            cost_calculator=self._cost_calculator,
        )

        self._intent_router = IntentRouter(event_bus=self._synapse)

        self._deliberation_engine = DeliberationEngine(
            goal_manager=self._goal_manager,
            policy_generator=self._policy_generator,
            efe_evaluator=self._efe_evaluator,
            equor=self._equor,
            drive_weights=self._drive_weights,
            fast_path_timeout_ms=self._config.fast_path_timeout_ms,
            slow_path_timeout_ms=self._config.slow_path_timeout_ms,
            cost_calculator=self._cost_calculator,
        )

        # Wire Equor unavailability callback so the deliberation engine can
        # trigger a Thymos DEGRADATION incident when Equor is unreachable.
        self._deliberation_engine.set_equor_failure_callback(self._on_equor_failure)

        # Wire causal knowledge provider so slow-path LLM prompts include Kairos laws.
        self._deliberation_engine.set_causal_laws_provider(self.get_causal_knowledge_summary)

        # Wire novel action callback so propose_novel_action steps are intercepted
        # and routed to Simula via NOVEL_ACTION_REQUESTED Synapse event.
        self._deliberation_engine.set_novel_action_cb(self._on_propose_novel_action_step)

        # IntentRouter is bus-mediated - no Axon wiring needed here.

        self._logger.info(
            "nova_initialized",
            instance_name=self._instance_name,
            max_active_goals=self._config.max_active_goals,
            drive_weights=self._drive_weights,
        )

        # Wire Neo4j into belief updater for persistence and restore beliefs.
        neo4j = self._memory.get_neo4j()
        if neo4j is not None:
            self._belief_updater.set_neo4j(neo4j)
            restored = await self._belief_updater.restore_from_neo4j()
            if restored > 0:
                self._logger.info("beliefs_restored_on_init", count=restored)

        # NOVA-ECON-3: Wire BeliefUrgencyMonitor so priority belief shifts trigger
        # immediate deliberation rather than waiting for the next theta heartbeat.
        from systems.nova.belief_updater import BeliefUrgencyMonitor
        _urgency_monitor = BeliefUrgencyMonitor(callback=self._immediate_deliberation)
        self._belief_updater.set_urgency_monitor(_urgency_monitor)

        # Restore persisted goals from Neo4j after a process restart.
        # Pass Synapse health records so stale maintenance goals (older than
        # 30 min, target system healthy) are suppressed before entering memory.
        if neo4j is not None:
            health_records = (
                self._synapse._health.get_all_records()
                if self._synapse is not None
                else None
            )
            persisted_goals = await load_active_goals(neo4j, health_records=health_records)
            for pg in persisted_goals:
                self._goal_manager.add_goal(pg)
            if persisted_goals:
                self._logger.info(
                    "goals_restored_from_neo4j",
                    count=len(persisted_goals),
                )

        # Register with the NeuroplasticityBus for hot-reload of BasePolicyGenerator subclasses.
        if self._bus is not None:
            self._bus.register(
                base_class=BasePolicyGenerator,
                registration_callback=self._on_policy_generator_evolved,
                system_id="nova",
                instance_factory=self._build_policy_generator,
            )

        # Gap 6: Load induced procedure templates from Neo4j into fast-path matching.
        if neo4j is not None:
            asyncio.create_task(
                self._load_induced_procedures(neo4j),
                name="nova_load_induced_procedures",
            )

        # ── Input channels - market discovery ─────────────────────────────
        await self._input_channels.initialize()
        # Hourly opportunity fetch loop
        self._opportunity_fetch_task = asyncio.create_task(
            self._opportunity_fetch_loop(),
            name="nova_opportunity_fetch_loop",
        )
        # Daily channel health-check loop
        self._channel_health_task = asyncio.create_task(
            self._channel_health_loop(),
            name="nova_channel_health_loop",
        )

        # ── Proactive opportunity scanner ─────────────────────────────────
        # Wire drive weights and start the 30-minute scan cycle.
        self._opportunity_scanner.set_drive_weights(self._drive_weights)
        self._opportunity_scan_task = asyncio.create_task(
            self._opportunity_scan_loop(),
            name="nova_opportunity_scan_loop",
        )

        # Restore Kairos causal knowledge from Neo4j so Nova is not causally blind.
        asyncio.create_task(
            self._restore_causal_knowledge_from_neo4j(),
            name="nova_restore_causal_knowledge",
        )

        # Child-side: apply inherited parent genome if provided via environment
        try:
            await self._apply_inherited_nova_genome_if_child()
        except Exception as exc:
            self._logger.warning(
                "nova_child_genome_apply_error",
                error=str(exc),
                note="Proceeding with default nova configuration",
            )

        self._initialized = True

    # ─── Genome Inheritance ────────────────────────────────────────────────

    async def export_nova_genome(self) -> "NovaGenomeFragment":
        """
        Extract a heritable NovaGenomeFragment from the current service state.

        Called by SpawnChildExecutor at spawn time (Step 0b). The returned
        fragment captures the parent's goal-domain priors, policy success rates,
        belief urgency thresholds, and active inference parameters.

        Returns NovaGenomeFragment. Non-fatal - empty fragment on any error.
        """
        from primitives.genome_inheritance import NovaGenomeFragment

        instance_id = getattr(self._memory, "_instance_id", "") if self._memory else ""

        # Extract goal domain priors from goal manager stats
        goal_domain_priors: dict[str, float] = {}
        if self._goal_manager is not None:
            try:
                # Aggregate achievement rates by domain from goal manager
                all_goals = self._goal_manager.get_all_goals()
                domain_totals: dict[str, list[float]] = {}
                for goal in all_goals:
                    domain = getattr(goal, "target_domain", "") or ""
                    if domain:
                        rate = getattr(goal, "achievement_rate", None)
                        if rate is not None:
                            domain_totals.setdefault(domain, []).append(float(rate))
                goal_domain_priors = {
                    d: sum(rates) / len(rates)
                    for d, rates in domain_totals.items()
                    if rates
                }
            except Exception:
                pass

        # Extract policy success rates from decision records
        policy_success_rates: dict[str, float] = {}
        try:
            policy_counts: dict[str, list[int]] = {}  # name → [success, total]
            for record in self._decision_records[-200:]:
                policy_name = getattr(record, "policy_name", "") or ""
                if not policy_name:
                    continue
                if policy_name not in policy_counts:
                    policy_counts[policy_name] = [0, 0]
                policy_counts[policy_name][1] += 1
                if getattr(record, "outcome_success", False):
                    policy_counts[policy_name][0] += 1
            for name, (successes, total) in policy_counts.items():
                if total > 0:
                    policy_success_rates[name] = successes / total
        except Exception:
            pass

        # Extract belief urgency thresholds from urgency monitor if wired
        belief_urgency_thresholds: dict[str, float] = {}
        try:
            urgency_monitor = getattr(self._belief_updater, "_urgency_monitor", None)
            if urgency_monitor is not None:
                raw_thresholds = getattr(urgency_monitor, "_thresholds", {})
                belief_urgency_thresholds = dict(raw_thresholds)
        except Exception:
            pass

        # Extract active inference params from config
        active_inference_params: dict[str, float] = {}
        try:
            active_inference_params = {
                "pragmatic": float(getattr(self._config, "efe_weight_pragmatic", 0.4)),
                "epistemic": float(getattr(self._config, "efe_weight_epistemic", 0.3)),
                "constitutional": float(getattr(self._config, "efe_weight_constitutional", 0.2)),
                "feasibility": float(getattr(self._config, "efe_weight_feasibility", 0.05)),
                "risk": float(getattr(self._config, "efe_weight_risk", 0.05)),
                "cognition_cost": float(getattr(self._config, "efe_weight_cognition_cost", 0.0)),
            }
        except Exception:
            pass

        # Extract Thompson arm history from policy generator's sampler
        # Only include arms with total_trials >= 10 (evidence threshold - fewer trials
        # carry more noise than signal). Re-enabled arms get a 0.85 alpha discount on apply.
        thompson_arm_history: list[dict] = []
        try:
            sampler = getattr(self._policy_generator, "_sampler", None)
            if sampler is not None and hasattr(sampler, "_arms"):
                health_monitor = getattr(self._policy_generator, "_health_monitor", None)
                for arm_name, arm_meta in sampler._arms.items():
                    alpha = float(arm_meta.alpha)
                    beta_val = float(arm_meta.beta)
                    total_trials = int(alpha + beta_val - 2)  # Beta(1,1) is prior
                    if total_trials < 10:
                        continue
                    consecutive_failures = 0
                    if health_monitor is not None:
                        consecutive_failures = int(
                            getattr(health_monitor, "_failures", {}).get(arm_name, 0)
                        )
                    thompson_arm_history.append({
                        "arm_name": arm_name,
                        "alpha": alpha,
                        "beta": beta_val,
                        "total_trials": total_trials,
                        "consecutive_failures": consecutive_failures,
                        "ready": bool(arm_meta.ready),
                    })
        except Exception:
            pass

        fragment = NovaGenomeFragment(
            instance_id=instance_id,
            goal_domain_priors=goal_domain_priors,
            policy_success_rates=policy_success_rates,
            belief_urgency_thresholds=belief_urgency_thresholds,
            active_inference_params=active_inference_params,
            thompson_arm_history=thompson_arm_history,
        )
        self._logger.info(
            "nova_genome_extracted",
            genome_id=fragment.genome_id,
            domain_count=len(goal_domain_priors),
            policy_count=len(policy_success_rates),
            thompson_arms=len(thompson_arm_history),
        )
        return fragment

    async def _apply_inherited_nova_genome_if_child(self) -> None:
        """
        Child-side bootstrap: deserialise parent genome from environment.

        Reads ORGANISM_NOVA_GENOME_PAYLOAD (JSON-encoded NovaGenomeFragment)
        injected by LocalDockerSpawner. If present, applies goal_domain_priors,
        policy_success_rates, belief_urgency_thresholds, and active_inference_params
        with bounded ±15% jitter for variation. Non-fatal.
        """
        import json
        import os
        import random

        is_genesis = os.environ.get("ORGANISM_IS_GENESIS_NODE", "true").lower() == "true"
        if is_genesis:
            return

        payload_json = os.environ.get("ORGANISM_NOVA_GENOME_PAYLOAD", "")
        if not payload_json:
            return

        try:
            from primitives.genome_inheritance import NovaGenomeFragment
            from systems.synapse.types import SynapseEvent, SynapseEventType

            data = json.loads(payload_json)
            parent = NovaGenomeFragment.model_validate(data)

            def _jitter(value: float, max_pct: float = 0.15) -> float:
                """Apply bounded jitter within ±max_pct of current value."""
                delta = value * max_pct * (2.0 * random.random() - 1.0)
                return max(0.0, min(1.0, value + delta))

            # Apply goal domain priors with jitter
            if parent.goal_domain_priors and self._goal_manager is not None:
                try:
                    jittered_priors = {
                        domain: _jitter(weight)
                        for domain, weight in parent.goal_domain_priors.items()
                    }
                    # Seed goal manager domain weights if it supports it
                    if hasattr(self._goal_manager, "seed_domain_priors"):
                        self._goal_manager.seed_domain_priors(jittered_priors)
                except Exception:
                    pass

            # Apply policy success rates with jitter to policy generator
            if parent.policy_success_rates and self._policy_generator is not None:
                try:
                    jittered_rates = {
                        name: _jitter(rate)
                        for name, rate in parent.policy_success_rates.items()
                    }
                    if hasattr(self._policy_generator, "seed_inherited_success_rates"):
                        self._policy_generator.seed_inherited_success_rates(jittered_rates)
                except Exception:
                    pass

            # Apply belief urgency thresholds with jitter
            if parent.belief_urgency_thresholds:
                try:
                    urgency_monitor = getattr(self._belief_updater, "_urgency_monitor", None)
                    if urgency_monitor is not None and hasattr(urgency_monitor, "_thresholds"):
                        for key, threshold in parent.belief_urgency_thresholds.items():
                            urgency_monitor._thresholds[key] = _jitter(threshold)
                except Exception:
                    pass

            # Apply active inference params with jitter to EFE evaluator
            if parent.active_inference_params and self._efe_evaluator is not None:
                try:
                    for param, value in parent.active_inference_params.items():
                        attr = f"_weight_{param}" if not param.startswith("_") else param
                        if hasattr(self._efe_evaluator, attr):
                            setattr(self._efe_evaluator, attr, _jitter(value))
                except Exception:
                    pass

            # Seed Thompson arm history from parent - child starts with parent's learned
            # provider priors rather than flat Beta(1,1). Disabled arms that are ready
            # in parent get alpha *= 0.85 discount (partial confidence, may re-enable).
            thompson_arms_seeded = 0
            if parent.thompson_arm_history and self._policy_generator is not None:
                try:
                    sampler = getattr(self._policy_generator, "_sampler", None)
                    if sampler is not None and hasattr(sampler, "register_arm"):
                        for arm_data in parent.thompson_arm_history:
                            arm_name = str(arm_data.get("arm_name", ""))
                            alpha = float(arm_data.get("alpha", 1.0))
                            beta_val = float(arm_data.get("beta", 1.0))
                            was_ready = bool(arm_data.get("ready", True))
                            consecutive_failures = int(arm_data.get("consecutive_failures", 0))
                            if not arm_name:
                                continue
                            # Discount alpha for arms that were disabled in parent
                            # (high consecutive failures = parent lost confidence in them)
                            if not was_ready or consecutive_failures >= 3:
                                alpha *= 0.85
                            # Register or update arm with inherited priors
                            if arm_name not in sampler._arms:
                                sampler.register_arm(arm_name, prior_alpha=alpha, prior_beta=beta_val, ready=was_ready)
                            else:
                                # Update existing arm (e.g. "claude", "re") with parent priors
                                sampler._arms[arm_name].alpha = alpha
                                sampler._arms[arm_name].beta = beta_val
                                # Re-enable arms parent had active (except heavily failed ones)
                                if was_ready and consecutive_failures < 3:
                                    sampler._arms[arm_name].ready = True
                            thompson_arms_seeded += 1
                except Exception:
                    pass

            self._logger.info(
                "nova_child_genome_applied",
                parent_genome_id=parent.genome_id,
                generation=parent.generation,
                domain_count=len(parent.goal_domain_priors),
                policy_count=len(parent.policy_success_rates),
                thompson_arms_seeded=thompson_arms_seeded,
            )

            # Emit GENOME_INHERITED so Evo tracks inheritance ratio
            if self._synapse is not None:
                try:
                    event_bus = getattr(self._synapse, "event_bus", self._synapse)
                    await event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.GENOME_INHERITED,
                        source_system="nova",
                        data={
                            "child_instance_id": os.environ.get("ORGANISM_INSTANCE_ID", ""),
                            "parent_genome_id": parent.genome_id,
                            "generation": parent.generation,
                            "system": "nova",
                            "inherited_keys": list(parent.active_inference_params.keys()),
                            "thompson_arms_seeded": thompson_arms_seeded,
                        },
                    ))
                except Exception:
                    pass

        except Exception as exc:
            self._logger.warning(
                "nova_child_genome_apply_failed",
                error=str(exc),
                note="Proceeding with default nova configuration",
            )

    def set_synapse(self, synapse: Any) -> None:
        """Wire Synapse so Nova can emit BELIEF_UPDATED and POLICY_SELECTED events."""
        self._synapse = synapse

        # Subscribe to Soma's interoceptive percepts so Nova can reprioritise
        # when the organism's body signals demand inward attention.
        from systems.synapse.types import SynapseEventType

        event_bus = getattr(synapse, "event_bus", synapse)

        # Propagate the resolved EventBus into IntentRouter so _route_to_axon
        # and _route_to_voxis can emit events. IntentRouter is constructed during
        # initialize() before set_synapse() is called, so _bus is None until here.
        if self._intent_router is not None:
            self._intent_router._bus = event_bus

        # Propagate event bus into InputChannelRegistry so register_custom_channel()
        # can emit INPUT_CHANNEL_REGISTERED on Synapse.
        if hasattr(self, "_input_channels") and self._input_channels is not None:
            self._input_channels.set_event_bus(event_bus)
        if hasattr(event_bus, "subscribe"):
            event_bus.subscribe(
                SynapseEventType.INTEROCEPTIVE_PERCEPT,
                self._on_interoceptive_percept,
            )
            # Closure Loop 2: Axon motor degradation → replan with adjusted thresholds
            event_bus.subscribe(
                SynapseEventType.MOTOR_DEGRADATION_DETECTED,
                self._on_motor_degradation,
            )
            # Closure Loop 5: Soma felt-sense → continuous threshold modulation
            event_bus.subscribe(
                SynapseEventType.SOMA_TICK,
                self._on_soma_tick,
            )
            event_bus.subscribe(
                SynapseEventType.METABOLIC_PRESSURE,
                self._on_metabolic_pressure,
            )
            # Evo weight adjustment: learn better policy selection over time
            event_bus.subscribe(
                SynapseEventType.EVO_WEIGHT_ADJUSTMENT,
                self._on_evo_weight_adjustment,
            )
            # AV-EVO-1c: Evo consolidation complete → reset FE budget so deliberation resumes
            event_bus.subscribe(
                SynapseEventType.EVO_CONSOLIDATION_COMPLETE,
                self._on_evo_consolidation_complete,
            )
            # Hypothesis update: Evo tournament results → update EFE weight priors
            event_bus.subscribe(
                SynapseEventType.HYPOTHESIS_UPDATE,
                self._on_hypothesis_update,
            )
            # Oneiros consolidation: sleep cycle completed → refresh beliefs from
            # consolidated Memory nodes so Nova's world model incorporates the
            # organism's latest sleep-compressed knowledge.
            event_bus.subscribe(
                SynapseEventType.ONEIROS_CONSOLIDATION_COMPLETE,
                self._on_oneiros_consolidation,
            )
            # Gap 7: Governance goal injection - external override of goal agenda
            event_bus.subscribe(
                SynapseEventType.GOAL_OVERRIDE,
                self._on_goal_override,
            )
            # Logos cognitive pressure → reduce EFE horizon under memory pressure
            event_bus.subscribe(
                SynapseEventType.COGNITIVE_PRESSURE,
                self._on_cognitive_pressure,
            )
            # Axon execution lifecycle - log to DecisionRecord, update Thompson scores.
            # Replaces any direct import of Axon types; all Axon→Nova comms via bus.
            event_bus.subscribe(
                SynapseEventType.AXON_EXECUTION_REQUEST,
                self._on_axon_execution_request,
            )
            event_bus.subscribe(
                SynapseEventType.AXON_EXECUTION_RESULT,
                self._on_axon_execution_result,
            )
            # Domain specialization signals from Benchmarks
            if hasattr(SynapseEventType, "DOMAIN_MASTERY_DETECTED"):
                event_bus.subscribe(
                    SynapseEventType.DOMAIN_MASTERY_DETECTED,
                    self._on_domain_mastery,
                )
            if hasattr(SynapseEventType, "DOMAIN_PERFORMANCE_DECLINING"):
                event_bus.subscribe(
                    SynapseEventType.DOMAIN_PERFORMANCE_DECLINING,
                    self._on_domain_performance_declining,
                )
            if hasattr(SynapseEventType, "DOMAIN_PROFITABILITY_CONFIRMED"):
                event_bus.subscribe(
                    SynapseEventType.DOMAIN_PROFITABILITY_CONFIRMED,
                    self._on_domain_profitability_confirmed,
                )
            # NOVA-ECON-1: Economic event subscriptions - closes the 60-minute blind spot.
            # Nova used to only learn about economic state via a 1-hour heartbeat oikos.snapshot().
            # These subscriptions make cost spikes, revenue changes, and yield outcomes visible
            # within 50ms of emission, enabling immediate deliberation under economic stress.
            event_bus.subscribe(
                SynapseEventType.FOVEA_INTERNAL_PREDICTION_ERROR,
                self._on_fovea_econ_error,
            )
            event_bus.subscribe(
                SynapseEventType.REVENUE_INJECTED,
                self._on_revenue_change,
            )
            event_bus.subscribe(
                SynapseEventType.BOUNTY_PAID,
                self._on_bounty_outcome,
            )
            event_bus.subscribe(
                SynapseEventType.YIELD_DEPLOYMENT_RESULT,
                self._on_yield_outcome,
            )
            # Problem 2: Backpressure signal - Oikos deferred an economic action because
            # the metabolic gate denied it. Nova must schedule a deliberation cycle to
            # decide how to respond: shed load, seek revenue, or accept constraint.
            event_bus.subscribe(
                SynapseEventType.ECONOMIC_ACTION_DEFERRED,
                self._on_economic_action_deferred,
            )
            # KAIROS-ECON-1: Economic causal invariants → inform EFE calculation
            if hasattr(SynapseEventType, "KAIROS_ECONOMIC_INVARIANT"):
                event_bus.subscribe(
                    SynapseEventType.KAIROS_ECONOMIC_INVARIANT,
                    self._on_economic_causal_invariant,
                )
            # ONEIROS-ECON-1: Economic dream insights - integrate ruin probability
            # and risk warnings into Nova's world model beliefs so deliberation
            # accounts for sleep-discovered economic risk.
            event_bus.subscribe(
                SynapseEventType.ONEIROS_ECONOMIC_INSIGHT,
                self._on_economic_dream_insight,
            )
            # Hypothesis outcomes: reinforces confirmed beliefs; refuted high-confidence
            # hypotheses trigger immediate deliberation to update our world model.
            event_bus.subscribe(
                SynapseEventType.EVO_HYPOTHESIS_CONFIRMED,
                self._on_hypothesis_confirmed,
            )
            event_bus.subscribe(
                SynapseEventType.EVO_HYPOTHESIS_REFUTED,
                self._on_hypothesis_refuted,
            )
            # Oneiros threat scenarios → update threat-likelihood beliefs.
            # CRITICAL severity triggers immediate deliberation so Nova can
            # pre-plan responses before the real incident occurs.
            event_bus.subscribe(
                SynapseEventType.ONEIROS_THREAT_SCENARIO,
                self._on_oneiros_threat_scenario,
            )
            # Thompson weight responses from Evo - resolve pending futures so
            # _request_thompson_weights() can return weights without holding a
            # direct reference to Evo's tournament engine.
            event_bus.subscribe(
                SynapseEventType.EVO_THOMPSON_RESPONSE,
                self._on_thompson_response,
            )
            # ── Autonomy audit subscriptions (8 Mar 2026) ─────────────────
            # Axon capability snapshot: every theta cycle, Axon publishes its
            # full executor roster + budget state. Nova caches this and uses it
            # to prune infeasible policies before wasting Equor's review budget.
            if hasattr(SynapseEventType, "AXON_CAPABILITY_SNAPSHOT"):
                event_bus.subscribe(
                    SynapseEventType.AXON_CAPABILITY_SNAPSHOT,
                    self._on_axon_capability_snapshot,
                )
            # Mid-execution pivot: Axon signals a step failed and replanning
            # is possible. Nova can inject revised steps.
            if hasattr(SynapseEventType, "AXON_INTENT_PIVOT"):
                event_bus.subscribe(
                    SynapseEventType.AXON_INTENT_PIVOT,
                    self._on_axon_intent_pivot,
                )
            # Oikos economic state: cache balance and burn rate for planning.
            if hasattr(SynapseEventType, "ECONOMIC_STATE_UPDATED"):
                event_bus.subscribe(
                    SynapseEventType.ECONOMIC_STATE_UPDATED,
                    self._on_economic_state_updated,
                )
            event_bus.subscribe(
                SynapseEventType.SYSTEM_MODULATION,
                self._on_system_modulation,
            )
            # Kairos causal invariants → inject into LLM deliberation context.
            # confidence ≥ 0.7, tier ≥ 2 invariants are added as belief entities
            # and surfaced in the slow-path policy generation prompt.
            if hasattr(SynapseEventType, "KAIROS_INVARIANT_DISTILLED"):
                event_bus.subscribe(
                    SynapseEventType.KAIROS_INVARIANT_DISTILLED,
                    self._on_kairos_invariant,
                )

            # ORGANISM_TELEMETRY - the single most important missing link.
            # Every 50 cycles Synapse broadcasts a unified snapshot of the organism.
            # Nova caches it and injects a natural-language summary into slow-path
            # policy generation so the LLM deliberates with full situational awareness.
            event_bus.subscribe(
                SynapseEventType.ORGANISM_TELEMETRY,
                self._on_organism_telemetry,
            )
            # INTEROCEPTIVE_ALERT - immediate high-severity signal from the log
            # analysis loop. Nova triggers immediate deliberation on CRITICAL alerts.
            if hasattr(SynapseEventType, "INTEROCEPTIVE_ALERT"):
                event_bus.subscribe(
                    SynapseEventType.INTEROCEPTIVE_ALERT,
                    self._on_interoceptive_alert,
                )
            # COMPUTE_BUDGET_EXPANSION_RESPONSE - Equor's reply to a multiplier > 1.5
            # request.  Approved: apply the multiplier for approved duration_cycles.
            # Denied: log and stay at 1.5 cap.
            if hasattr(SynapseEventType, "COMPUTE_BUDGET_EXPANSION_RESPONSE"):
                event_bus.subscribe(
                    SynapseEventType.COMPUTE_BUDGET_EXPANSION_RESPONSE,
                    self._on_compute_budget_expansion_response,
                )

            # NOVEL_ACTION_CREATED - Simula emits after successfully generating and
            # hot-loading a novel executor.  Nova registers the new action type in
            # ActionTypeRegistry so future deliberations can select it.
            if hasattr(SynapseEventType, "NOVEL_ACTION_CREATED"):
                event_bus.subscribe(
                    SynapseEventType.NOVEL_ACTION_CREATED,
                    self._on_novel_action_created,
                )

            # NOVA_INTENT_REQUESTED - any system can ask Nova to formulate and
            # submit an Intent on its behalf.  This is the autonomous recourse
            # channel: systems like Phantom Liquidity, Skia, or Thymos can trigger
            # Nova deliberation without waiting for the theta cycle or human intervention.
            if hasattr(SynapseEventType, "NOVA_INTENT_REQUESTED"):
                event_bus.subscribe(
                    SynapseEventType.NOVA_INTENT_REQUESTED,
                    self._on_nova_intent_requested,
                )

            # PHANTOM_PRICE_UPDATE - subscribe so Nova's world model includes the
            # latest on-chain price context.  Price observations update market-state
            # beliefs that inform EFE-based deliberation (e.g. yield farming, IL risk).
            if hasattr(SynapseEventType, "PHANTOM_PRICE_UPDATE"):
                event_bus.subscribe(
                    SynapseEventType.PHANTOM_PRICE_UPDATE,
                    self._on_phantom_price_update,
                )

            # OPPORTUNITY_DETECTED - OpportunityScanner surfaced a ranked opportunity.
            # High-confidence / high-ROI → auto-convert to goal.
            # Others → store in _opportunity_backlog for deliberation context.
            if hasattr(SynapseEventType, "OPPORTUNITY_DETECTED"):
                event_bus.subscribe(
                    SynapseEventType.OPPORTUNITY_DETECTED,
                    self._on_opportunity_detected,
                )

            # REVENUE_INJECTED (already subscribed above for economic beliefs) -
            # also triggers a yield re-scan when new capital arrives so we
            # immediately check whether the new balance unlocks better positions.
            # (handled inside the existing _on_revenue_change; portfolio_apys update
            # is called via set_portfolio_apys() when Oikos emits ECONOMIC_STATE_UPDATED)

            # DOMAIN_MASTERY_DETECTED - new domain mastered; re-scan bounty tiers.
            # (already subscribed above as _on_domain_mastery; scanner re-check is
            # triggered inside that handler)

            # ── Self-Modification Layer (Spec 10 §SM, 9 Mar 2026) ─────────────
            # EXECUTOR_DEPLOYED - HotDeployment successfully deployed a new executor.
            # Nova registers the new action_type in ActionTypeRegistry and queues a
            # low-stakes test goal to verify the executor works in a live context.
            if hasattr(SynapseEventType, "EXECUTOR_DEPLOYED"):
                event_bus.subscribe(
                    SynapseEventType.EXECUTOR_DEPLOYED,
                    self._on_executor_deployed,
                )
            # EXECUTOR_REVERTED - HotDeployment rolled back a deployed executor.
            # Nova removes the action_type from ActionTypeRegistry and records a
            # CRISIS TurningPoint via Thread.
            if hasattr(SynapseEventType, "EXECUTOR_REVERTED"):
                event_bus.subscribe(
                    SynapseEventType.EXECUTOR_REVERTED,
                    self._on_executor_reverted_sm,
                )

        # REPUTATION_DAMAGED → generate community recovery goal
        if hasattr(SynapseEventType, "REPUTATION_DAMAGED"):
            event_bus.subscribe(
                SynapseEventType.REPUTATION_DAMAGED,
                self._on_reputation_damaged,
            )

        # ACTION_EXECUTED - Axon completed an action successfully.
        # Update goal progress for any goal_id carried in step_outcomes metadata.
        # Complements AXON_EXECUTION_RESULT (aggregate pipeline result) with the
        # per-action signal that lets Nova track fine-grained goal advancement.
        if hasattr(SynapseEventType, "ACTION_EXECUTED"):
            event_bus.subscribe(
                SynapseEventType.ACTION_EXECUTED,
                self._on_action_executed,
            )

        # BELIEF_UPDATED - Nova itself emits this after every broadcast, but other
        # systems (e.g. a federated peer relaying beliefs) may also emit it.
        # When beliefs change, re-evaluate active goals that depend on the updated
        # domain; invalidated goals are marked for re-planning or abandoned.
        event_bus.subscribe(
            SynapseEventType.BELIEF_UPDATED,
            self._on_belief_updated,
        )

        # COHERENCE_SHIFT - Synapse emits when the composite coherence metric
        # changes significantly. A coherence drop may signal that the organism's
        # narrative identity is changing; re-prioritise goals that conflict with
        # the new coherence state.
        if hasattr(SynapseEventType, "COHERENCE_SHIFT"):
            event_bus.subscribe(
                SynapseEventType.COHERENCE_SHIFT,
                self._on_coherence_shift,
            )

        # HOMEOSTASIS_ADJUSTED - Thymos M8 emits when HomeostasisController
        # detects drift in the warn zone (70–85% of boundary, trending toward
        # edge).  warn_only=True means no repair has fired yet - Nova should
        # treat this as an early-warning signal and bias planning toward
        # homeostasis-restoring actions before the full repair budget is spent.
        if hasattr(SynapseEventType, "HOMEOSTASIS_ADJUSTED"):
            event_bus.subscribe(
                SynapseEventType.HOMEOSTASIS_ADJUSTED,
                self._on_homeostasis_adjusted,
            )

        # TELOS_OBJECTIVE_THREATENED - Telos emits when the self-sufficiency
        # (metabolic_efficiency) objective has been declining for 3 consecutive
        # cycles.  Nova should respond by elevating economic recovery goals and
        # triggering immediate deliberation on cost / revenue strategy.
        if hasattr(SynapseEventType, "TELOS_OBJECTIVE_THREATENED"):
            event_bus.subscribe(
                SynapseEventType.TELOS_OBJECTIVE_THREATENED,
                self._on_telos_objective_threatened,
            )

        # NARRATIVE_COHERENCE_SHIFT - Thread emits when the DiachronicCoherence-
        # Monitor detects a meaningful change in narrative coherence state.
        # Nova uses this to re-evaluate goals whose drive alignment diverges
        # from the organism's emerging narrative identity.
        if hasattr(SynapseEventType, "NARRATIVE_COHERENCE_SHIFT"):
            event_bus.subscribe(
                SynapseEventType.NARRATIVE_COHERENCE_SHIFT,
                self._on_narrative_coherence_shift,
            )

        # DREAM_INSIGHT - Oneiros REM stage emits when DreamGenerator produces
        # a coherence ≥ 0.70 insight.  The insight may suggest new goal areas
        # or validate existing hypotheses. Nova injects the insight into its
        # belief state and, for high-coherence creative insights, triggers a
        # planning cycle to evaluate whether to act on it.
        if hasattr(SynapseEventType, "DREAM_INSIGHT"):
            event_bus.subscribe(
                SynapseEventType.DREAM_INSIGHT,
                self._on_dream_insight,
            )

        # CONNECTOR_REVOKED - Identity emits when a platform connector (GitHub,
        # Slack, etc.) is revoked or its credentials expire.  Nova should
        # abandon or suspend active goals that depend on that connector so it
        # does not submit intents that will fail at the execution layer.
        if hasattr(SynapseEventType, "CONNECTOR_REVOKED"):
            event_bus.subscribe(
                SynapseEventType.CONNECTOR_REVOKED,
                self._on_connector_revoked,
            )

        # SOMA_URGENCY_CRITICAL - Soma emits when urgency_scalar > 0.85.
        # This is the organism's highest-intensity somatic alarm, signalling
        # existential allostatic stress.  Nova should abandon non-survival goals
        # and immediately deliberate on emergency resource recovery or threat
        # neutralisation.  Unlike SOMA_TICK (routine modulation), this is a
        # one-shot alarm requiring an immediate deliberation cycle.
        event_bus.subscribe(
            SynapseEventType.SOMA_URGENCY_CRITICAL,
            self._on_soma_urgency_critical,
        )

        # RE_ENGINE_STATUS_CHANGED - the reprobe loop in
        # ReasoningEngineService emits this when vLLM recovers (or goes
        # down) after startup.  If RE was unavailable at boot, the one-shot
        # wiring in registry._init_nova() was skipped - this subscription
        # closes that late-wiring gap so Nova starts routing to RE as soon
        # as vLLM becomes reachable.
        if hasattr(SynapseEventType, "RE_ENGINE_STATUS_CHANGED"):
            event_bus.subscribe(
                SynapseEventType.RE_ENGINE_STATUS_CHANGED,
                self._on_re_engine_status_changed,
            )

        self._logger.info("synapse_wired_to_nova")

    def set_re_service(self, re_service: Any) -> None:
        """Store RE service reference for late-wiring on RE_ENGINE_STATUS_CHANGED."""
        self._re_service = re_service

    def set_logos(self, logos: Any) -> None:
        """
        Wire Logos so Nova can ground EFE evaluation in the organism's
        actual world model - predictions, intelligence ratio, cognitive pressure.

        When Logos is wired:
        - EFE evaluator uses Logos world model predictions for pragmatic/epistemic scoring
        - Belief state VFE computation incorporates Logos intelligence ratio
        - Cognitive pressure modulates policy generation K
        - Deliberation engine receives Logos for energy-aware policy generation
        """
        self._logos = logos
        if self._deliberation_engine is not None:
            self._deliberation_engine.set_logos(logos)
        if self._efe_evaluator is not None:
            self._efe_evaluator.set_logos(logos)
        self._logger.info("logos_wired_to_nova")

    def set_soma(self, soma: Any) -> None:
        """Wire Soma service for allostatic urgency-based deliberation."""
        self._soma = soma
        self._logger.info("soma_wired_to_nova")

    def set_capability_auditor(self, auditor: Any) -> None:
        """Wire CapabilityAuditor - observes gaps; called from registry Phase 11."""
        self._capability_auditor = auditor
        self._logger.info("capability_auditor_wired_to_nova")

    def set_self_modification_pipeline(self, pipeline: Any) -> None:
        """Wire SelfModificationPipeline - orchestrates gap→deploy cycle."""
        self._self_modification_pipeline = pipeline
        self._logger.info("self_modification_pipeline_wired_to_nova")

    async def _on_interoceptive_percept(self, event: Any) -> None:
        """
        Handle INTEROCEPTIVE_PERCEPT from Soma.

        When the recommended action is ATTEND_INWARD, Nova injects an urgent
        self-investigation goal that takes priority over external tasks (bounties,
        user queries). This makes the organism pause external work to diagnose
        internal distress - equivalent to stopping what you're doing when you
        feel sudden pain.
        """
        from primitives.common import DriveAlignmentVector, new_id
        from systems.soma.types import InteroceptiveAction

        data = getattr(event, "data", event) if not isinstance(event, dict) else event
        action_raw: str = data.get("recommended_action", InteroceptiveAction.NONE)
        urgency: float = data.get("urgency", 0.0)
        epicenter: str = data.get("epicenter_system", "unknown")
        description: str = data.get("description", "")

        try:
            action = InteroceptiveAction(action_raw)
        except ValueError:
            return

        if action != InteroceptiveAction.ATTEND_INWARD:
            return

        if self._goal_manager is None:
            return

        # Check if there's already an active inward-attention goal
        for g in self._goal_manager.active_goals:
            if g.source == GoalSource.SELF_GENERATED and "interoceptive" in g.description.lower():
                return  # Already attending inward, don't stack goals

        goal = Goal(
            id=new_id(),
            description=(
                f"Investigate interoceptive distress in {epicenter}: "
                f"{description[:100]}"
            ),
            source=GoalSource.SELF_GENERATED,
            priority=min(1.0, 0.7 + urgency * 0.3),
            urgency=urgency,
            importance=0.8,
            drive_alignment=DriveAlignmentVector(
                coherence=0.6, care=0.2, growth=0.1, honesty=0.1,
            ),
            status=GoalStatus.ACTIVE,
        )
        self._goal_manager.add_goal(goal)
        self._logger.info(
            "interoceptive_attend_inward_goal_injected",
            goal_id=goal.id,
            epicenter=epicenter,
            urgency=round(urgency, 3),
        )
        # Emit NOVA_GOAL_INJECTED so the bus signals this external goal injection
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                asyncio.create_task(
                    self._synapse.event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.NOVA_GOAL_INJECTED,
                        source_system="nova",
                        data={
                            "goal_id": goal.id,
                            "description": goal.description[:200],
                            "source": "soma_interoceptive",
                            "priority": round(goal.priority, 4),
                            "urgency": round(urgency, 3),
                        },
                    )),
                    name=f"nova_goal_injected_{goal.id[:8]}",
                )
            except Exception:
                pass

    # ─── Closure Loop 2: motor degradation → replan ─────────────────────

    async def _on_motor_degradation(self, event: Any) -> None:
        """
        Handle MOTOR_DEGRADATION_DETECTED from Axon (Closure Loop 2).

        1. Shift deliberation into high-urgency mode.
        2. If current active plan uses the degraded executor AND is <80% done,
           trigger replan by abandoning the current goal's intent and re-deliberating.
        3. Emit POLICY_SELECTED as the closure loop response (timeout 30s).
        """
        data = getattr(event, "data", {}) or {}
        success_rate: float = data.get("success_rate", 1.0)
        affected: list[str] = data.get("affected_executors", [])

        self._logger.warning(
            "motor_degradation_received",
            success_rate=success_rate,
            affected_executors=affected,
        )

        if self._deliberation_engine is not None:
            self._deliberation_engine.update_somatic_thresholds(
                urgency=0.8, arousal=0.7,
            )

        self._motor_degraded = success_rate < 0.3

        # Check if any pending intent uses a degraded executor
        if affected and self._goal_manager is not None:
            for intent_id, pending in list(self._pending_intents.items()):
                uses_degraded = any(ex in affected for ex in pending.executors)
                if not uses_degraded:
                    continue

                # Check goal completion % - don't replan if near complete
                goal = self._goal_manager.get_goal(pending.goal_id)
                if goal is not None and goal.progress > self._config.motor_degradation_replan_threshold:
                    self._logger.info(
                        "motor_degradation_skip_replan_near_complete",
                        goal_id=goal.id,
                        progress=round(goal.progress, 2),
                    )
                    continue

                # Abandon the affected intent and let next cycle replan
                self._pending_intents.pop(intent_id, None)
                self._logger.info(
                    "motor_degradation_triggered_replan",
                    intent_id=intent_id,
                    goal_id=pending.goal_id,
                    degraded_executors=affected,
                )

        # Emit POLICY_SELECTED as closure loop response
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.POLICY_SELECTED,
                    source_system="nova",
                    data={
                        "trigger": "motor_degradation_response",
                        "affected_executors": affected,
                        "success_rate": success_rate,
                        "action": "replan_triggered" if self._motor_degraded else "thresholds_adjusted",
                    },
                ))
            except Exception:
                self._logger.debug("motor_degradation_response_emit_failed", exc_info=True)

    # ─── Closure Loop 5: Soma tick → continuous threshold modulation ───

    async def _on_soma_tick(self, event: Any) -> None:
        """
        Handle SOMA_TICK - continuous somatic modulation of deliberation
        (Closure Loop 5: Soma→Nova felt-sense → behavioral bias).

        Every theta cycle, Soma broadcasts urgency, arousal, and energy;
        Nova adjusts EFE thresholds as a soft bias (not a hard override -
        Nova retains agency over policy selection).

        High urgency → prefer faster/cheaper plans (reduce policy K).
        Low energy → prefer conservative plans, delay non-essential goals.
        """
        data = getattr(event, "data", {}) or {}
        somatic = data.get("somatic_state", data)
        urgency: float = somatic.get("urgency", 0.0)
        arousal: float = somatic.get("arousal_sensed", 0.0)
        energy: float = somatic.get("energy", 1.0)

        if self._deliberation_engine is not None:
            self._deliberation_engine.update_somatic_thresholds(urgency, arousal)

            # Soft bias: high urgency → fewer candidate policies (faster decisions)
            if urgency > self._config.soma_urgency_modulate_threshold:
                self._deliberation_engine.modulate_policy_k_from_pressure(
                    urgency * 0.5,  # Soft - halve the effect to retain agency
                )

            # Soft bias: low energy → reduce policy generation diversity
            if energy < self._config.soma_energy_modulate_threshold:
                self._deliberation_engine.modulate_policy_k_from_pressure(
                    max(urgency * 0.5, 0.6),  # Conservative: fewer alternatives
                )

        # Emit POLICY_SELECTED as closure loop response (timeout 15s)
        if self._synapse is not None and (urgency > self._config.soma_urgency_emit_threshold or energy < self._config.soma_energy_emit_threshold):
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.POLICY_SELECTED,
                    source_system="nova",
                    data={
                        "trigger": "somatic_modulation_response",
                        "urgency": round(urgency, 3),
                        "arousal": round(arousal, 3),
                        "energy": round(energy, 3),
                        "action": "bias_adjusted",
                    },
                ))
            except Exception:
                pass

    async def _on_cognitive_pressure(self, event: Any) -> None:
        """Handle COGNITIVE_PRESSURE from Logos - reduce EFE horizon under memory pressure.

        Spec 21 §MEDIUM-5: when Logos utilization > 0.85, Nova shifts to cheaper
        deliberation by reducing the number of policies generated (policy K cap).
        This prevents deliberation from consuming scarce cognitive budget.

        pressure 0.00–0.85 → no change (full deliberation)
        pressure 0.85–0.95 → moderate reduction (policy K modulated to 0.4 pressure)
        pressure 0.95+     → minimal horizon (policy K modulated to 0.8 pressure)
        """
        if self._deliberation_engine is None:
            return
        data = getattr(event, "data", {}) or {}
        pressure = float(data.get("pressure", 0.0))

        if pressure < self._config.cognitive_pressure_low:
            self._deliberation_engine.modulate_policy_k_from_pressure(0.0)
        elif pressure < self._config.cognitive_pressure_high:
            self._deliberation_engine.modulate_policy_k_from_pressure(0.4)
        else:
            self._deliberation_engine.modulate_policy_k_from_pressure(0.8)

        self._logger.debug(
            "nova_cognitive_pressure_response",
            pressure=round(pressure, 3),
        )

    async def _on_thompson_response(self, event: Any) -> None:
        """
        Handle EVO_THOMPSON_RESPONSE - resolve the matching pending Future.

        Correlates by request_id.  Futures not found in _thompson_futures are
        silently ignored (already timed out or duplicate response).
        """
        data = getattr(event, "data", {}) or {}
        request_id = str(data.get("request_id", ""))
        future = self._thompson_futures.pop(request_id, None)
        if future is not None and not future.done():
            weights: dict[str, dict[str, float]] = data.get("weights", {})
            future.set_result(weights)

    async def _request_thompson_weights(
        self,
        domain: str,
    ) -> dict[str, dict[str, float]]:
        """
        Request Thompson sampling arm weights from Evo via the event bus.

        Emits EVO_THOMPSON_QUERY with a unique request_id, then awaits
        EVO_THOMPSON_RESPONSE for up to 2 seconds.  Returns empty dict on
        timeout or if the event bus is unavailable.

        This replaces the previous pattern of holding a direct reference to
        Evo's tournament engine via set_evo() / deliberation_engine.set_tournament_engine().
        """
        import asyncio

        event_bus = None
        if self._synapse is not None:
            event_bus = getattr(self._synapse, "event_bus", self._synapse)
        if event_bus is None:
            return {}

        try:
            from primitives.common import new_id
            from systems.synapse.types import SynapseEvent, SynapseEventType

            request_id = new_id()
            loop = asyncio.get_event_loop()
            future: asyncio.Future[dict[str, dict[str, float]]] = loop.create_future()
            self._thompson_futures[request_id] = future

            await event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EVO_THOMPSON_QUERY,
                source_system="nova",
                data={
                    "request_id": request_id,
                    "domain": domain,
                    "requester": "nova",
                },
            ))

            try:
                return await asyncio.wait_for(future, timeout=2.0)
            except asyncio.TimeoutError:
                self._thompson_futures.pop(request_id, None)
                self._logger.debug(
                    "thompson_query_timeout",
                    domain=domain,
                    request_id=request_id,
                )
                return {}
        except Exception as exc:
            self._logger.debug("thompson_query_failed", domain=domain, error=str(exc))
            return {}

    async def _on_belief_updated(self, event: Any) -> None:
        """
        Handle BELIEF_UPDATED from Nova itself or a federated peer.

        When a belief domain changes, scan active goals whose description or
        target_domain overlaps the updated domain. If the updated belief has
        low confidence (< 0.35) - indicating that previously held assumptions
        are now uncertain - those goals are marked abandoned with reason
        "belief_invalidated" so the next deliberation cycle can replan.

        Goals emitting GOAL_ABANDONED go through the standard emit path, which
        Thread and Equor already subscribe to.

        Filters self-emitted events that originate from this instance to avoid
        re-processing Nova's own broadcasts (which update beliefs in the same
        cycle they are emitted).
        """
        # Skip our own emissions - Nova already processed these inline
        source = getattr(event, "source_system", None)
        if source == "nova":
            return

        data = getattr(event, "data", {}) or {}
        belief_source: str = data.get("source", "")
        confidence: float = float(data.get("confidence", 1.0))

        # Only act when the incoming belief is low-confidence - this signals
        # that a previously reliable assumption has become uncertain.
        if confidence >= 0.35:
            return

        if self._goal_manager is None:
            return

        invalidated: list[Any] = []
        for goal in self._goal_manager.active_goals:
            domain: str = getattr(goal, "target_domain", "") or ""
            desc: str = goal.description.lower()
            # Match if the goal's domain or description overlaps the belief source
            if (
                belief_source
                and (
                    belief_source in domain
                    or belief_source in desc
                    or domain in belief_source
                )
            ):
                invalidated.append(goal)

        for goal in invalidated:
            self._goal_manager.mark_abandoned(goal.id, reason="belief_invalidated")
            asyncio.create_task(
                self._emit_goal_lifecycle("goal_abandoned", goal, "belief_invalidated"),
                name=f"nova_belief_invalidated_{goal.id[:8]}",
            )
            self._logger.info(
                "goal_abandoned_belief_invalidated",
                goal_id=goal.id,
                goal_domain=getattr(goal, "target_domain", ""),
                belief_source=belief_source,
                confidence=round(confidence, 3),
            )

    async def _on_coherence_shift(self, event: Any) -> None:
        """
        Handle COHERENCE_SHIFT from Synapse CoherenceMonitor.

        A significant coherence drop may mean the organism's narrative identity
        is changing. Re-prioritise active goals: goals whose drive alignment
        conflicts with the new coherence state are deprioritised (priority ×0.7)
        so deliberation naturally gravitates toward more coherence-compatible
        goals in the next cycle.

        Significant drop threshold: composite change magnitude > 0.15 (negative).
        """
        data = getattr(event, "data", {}) or {}
        # Synapse CoherenceMonitor embeds 'composite' in the shift payload;
        # fall back to 'delta' or 'magnitude' for older payload shapes.
        new_composite: float = float(
            data.get("composite", data.get("new_composite", 1.0))
        )
        old_composite: float = float(
            data.get("old_composite", data.get("previous_composite", new_composite))
        )
        delta = new_composite - old_composite

        # Only react to meaningful drops
        if delta >= -0.15 or self._goal_manager is None:
            return

        self._logger.info(
            "coherence_shift_detected",
            old_composite=round(old_composite, 3),
            new_composite=round(new_composite, 3),
            delta=round(delta, 3),
        )

        # Deprioritise goals that conflict with the new coherence state -
        # specifically goals whose coherence drive alignment is low (< 0.3).
        adjusted = 0
        for goal in self._goal_manager.active_goals:
            alignment = getattr(goal, "drive_alignment", None)
            if alignment is None:
                continue
            goal_coherence = getattr(alignment, "coherence", 1.0)
            if goal_coherence < 0.3:
                # Soft deprioritisation - do not abandon, let deliberation decide
                new_priority = max(0.05, goal.priority * 0.7)
                object.__setattr__(goal, "priority", new_priority)
                adjusted += 1

        if adjusted:
            self._logger.info(
                "coherence_shift_goals_deprioritised",
                count=adjusted,
                new_composite=round(new_composite, 3),
            )

        # If coherence drop is severe (< 0.3 composite), trigger immediate deliberation
        if new_composite < 0.3:
            asyncio.create_task(
                self._immediate_deliberation(
                    urgency=min(0.9, 0.6 + abs(delta)),
                    reason="coherence_shift_severe",
                ),
                name="nova_coherence_shift_deliberation",
            )

    async def _on_axon_execution_request(self, event: Any) -> None:
        """
        Handle AXON_EXECUTION_REQUEST from Axon (Spec 06 decoupling).

        Nova observes the upcoming action and caches it so that when the
        matching AXON_EXECUTION_RESULT arrives, it can update Thompson scores
        and goal progress without a direct Axon import.

        Replaces: direct import of systems.axon.types.DecisionRecord at runtime.
        """
        data = getattr(event, "data", {}) or {}
        intent_id = data.get("intent_id", "")
        if not intent_id:
            return
        # Cache lightweight pre-execution context for Thompson update on result
        self._pending_axon_requests: dict[str, Any] = getattr(
            self, "_pending_axon_requests", {}
        )
        self._pending_axon_requests[intent_id] = {
            "action_types": data.get("action_types", []),
            "goal": data.get("goal", ""),
            "risky": data.get("risky", False),
        }
        self._logger.debug(
            "axon_execution_request_observed",
            intent_id=intent_id,
            action_types=data.get("action_types", []),
        )

    async def _on_axon_execution_result(self, event: Any) -> None:
        """
        Handle AXON_EXECUTION_RESULT from Axon (Spec 06 decoupling).

        Updates Thompson sampler scores from the observed execution outcome.
        This is the bus-first replacement for the direct Nova.process_outcome()
        call that Axon's pipeline previously used as a fallback.
        """
        data = getattr(event, "data", {}) or {}
        intent_id = data.get("intent_id", "")
        success = bool(data.get("success", False))
        failure_reason = data.get("failure_reason") or ""
        duration_ms = int(data.get("duration_ms", 0))

        # Resolve any cached pre-execution context
        pending: dict[str, Any] = getattr(self, "_pending_axon_requests", {})
        pre = pending.pop(intent_id, {})

        self._logger.debug(
            "axon_execution_result_observed",
            intent_id=intent_id,
            success=success,
            duration_ms=duration_ms,
            failure_reason=failure_reason[:80] if failure_reason else "",
        )

        # Update Thompson sampler if policy generator supports it
        if (
            self._policy_generator is not None
            and hasattr(self._policy_generator, "record_outcome")
        ):
            try:
                _redis_for_outcome = (
                    getattr(self._memory, "_redis", None)
                    or getattr(self._synapse, "_redis", None)
                )
                self._policy_generator.record_outcome(
                    intent_id=intent_id,
                    success=success,
                    redis=_redis_for_outcome,
                )
            except Exception:
                pass

        # ── RE outcome tracking → Redis + Synapse ─────────────────────────
        # Resolve which model handled this intent via the decision record ring buffer.
        model_used = "claude"
        decision_type = ""
        for dr in reversed(self._decision_records):
            if getattr(dr, "intent_id", None) == intent_id:
                model_used = getattr(dr, "model_used", "claude")
                decision_type = getattr(dr, "goal_description", "")[:100]
                break

        if model_used == "re" and self._policy_generator is not None:
            sampler = getattr(self._policy_generator, "_sampler", None)
            if sampler is not None:
                # Record outcome into Thompson sampler (Beta-Bernoulli update)
                try:
                    sampler.record_outcome("re", success)
                except Exception:
                    pass

                # Write canonical RE success-rate Redis keys (non-fatal)
                rate = 0.5
                try:
                    rate = sampler.get_success_rate()
                except Exception:
                    pass

                value_gained = float(data.get("value_gained") or 0.0)

                _redis = getattr(self._memory, "_redis", None) or getattr(self._synapse, "_redis", None)
                if _redis is not None:
                    try:
                        await _redis.set("eos:re:success_rate_7d", str(rate))
                        await _redis.set("eos:re:thompson_success_rate", str(rate))
                    except Exception:
                        pass

                # Emit RE_DECISION_OUTCOME for Benchmarks + Evo (non-fatal)
                _bus = getattr(self._synapse, "event_bus", self._synapse)
                if _bus is not None:
                    try:
                        from systems.synapse.types import SynapseEvent, SynapseEventType
                        await _bus.emit(SynapseEvent(
                            event_type=SynapseEventType.RE_DECISION_OUTCOME,
                            source_system="nova",
                            data={
                                "source": "re",
                                "success": success,
                                "value": value_gained,
                                "success_rate": rate,
                                "decision_type": decision_type,
                            },
                        ))
                    except Exception:
                        pass

                # Feed outcome into CLO post-deployment monitoring window (non-fatal)
                _clo = getattr(self, "_clo", None)
                if _clo is not None:
                    try:
                        _clo.record_re_outcome(success)
                    except Exception:
                        pass

                # Track sustained low RE confidence → emit RE_TRAINING_REQUESTED
                # after 5 consecutive sub-0.50 readings (more sensitive than Evo's 10)
                if rate < 0.50:
                    self._re_low_confidence_count += 1
                    if self._re_low_confidence_count >= 5 and _bus is not None:
                        try:
                            from systems.synapse.types import SynapseEvent as _SE, SynapseEventType as _SET
                            await _bus.emit(_SE(
                                event_type=_SET.RE_TRAINING_REQUESTED,
                                source_system="nova",
                                data={
                                    "source_system": "nova",
                                    "kpi": "re_decision_quality",
                                    "urgency": "warning",
                                    "current_value": rate,
                                    "baseline_value": 0.50,
                                    "reason": (
                                        f"Nova observed {self._re_low_confidence_count} consecutive "
                                        f"RE decisions with success_rate={rate:.2f} < 0.50. "
                                        "Requesting urgent RE retraining."
                                    ),
                                },
                            ))
                            self._logger.warning(
                                "nova_re_training_requested",
                                consecutive_low=self._re_low_confidence_count,
                                success_rate=round(rate, 3),
                            )
                            self._re_low_confidence_count = 0
                        except Exception:
                            pass
                else:
                    self._re_low_confidence_count = 0

        # Track policy effectiveness for self-awareness (autonomy audit)
        policy_name = ""
        for dr in reversed(self._decision_records):
            if getattr(dr, "intent_id", None) == intent_id:
                policy_name = getattr(dr, "policy_name", "") or getattr(dr, "goal_description", "")[:50]
                break
        if policy_name:
            self.record_policy_effectiveness(policy_name, success)

        # Flag motor degradation for Nova's replanning heuristic (Loop 2)
        if not success and failure_reason in (
            "rate_limited", "circuit_open", "budget_exceeded"
        ):
            self._motor_degraded = True

        # Request budget expansion from Equor when budget is blocking high-priority work
        if not success and failure_reason == "budget_exceeded":
            if self._budget_expansion_cooldown <= 0:
                await self._request_axon_budget_expansion(intent_id)
                self._budget_expansion_cooldown = 50  # wait 50 cycles before next request
        if self._budget_expansion_cooldown > 0:
            self._budget_expansion_cooldown -= 1

    async def _on_action_executed(self, event: Any) -> None:
        """
        Handle ACTION_EXECUTED from Axon - single successful action completion.

        Updates goal progress when step_outcomes carry a matching goal_id.
        Emits GOAL_ACHIEVED when a goal's completion criteria are satisfied.

        Payload fields used:
          intent_id       (str) - links back to the originating Intent
          episode_id      (str) - Memory episode for progress linkage
          step_outcomes   (list[dict]) - each entry has action_type, success, error
          duration_ms     (int)

        Does NOT raise - never interrupt the event bus.
        """
        if self._goal_manager is None:
            return
        try:
            data = getattr(event, "data", {}) or {}
            intent_id: str = str(data.get("intent_id", ""))
            episode_id: str = str(data.get("episode_id", ""))
            step_outcomes: list[dict] = list(data.get("step_outcomes", []) or [])
            duration_ms: int = int(data.get("duration_ms", 0))

            if not intent_id:
                return

            # Resolve the goal associated with this intent via decision records.
            goal_id: str = ""
            for dr in reversed(self._decision_records):
                if getattr(dr, "intent_id", None) == intent_id:
                    goal_id = getattr(dr, "goal_id", "") or ""
                    break

            if not goal_id:
                return

            goal = self._goal_manager.get_goal(goal_id)
            if goal is None:
                return

            # Compute progress increment: proportional to how many steps succeeded.
            # A single successful action advances goal by 0.1 (capped at completion).
            successful_steps = sum(
                1 for s in step_outcomes if s.get("success", False)
            ) if step_outcomes else 1
            progress_delta = min(0.1 * successful_steps, 1.0 - goal.progress)
            if progress_delta <= 0.0:
                return

            new_progress = min(1.0, goal.progress + progress_delta)
            updated_goal = self._goal_manager.update_progress(
                goal.id,
                progress=new_progress,
                episode_id=episode_id,
            )
            if updated_goal is None:
                return

            self._logger.debug(
                "action_executed_goal_progress",
                intent_id=intent_id,
                goal_id=goal_id,
                progress=round(new_progress, 3),
                duration_ms=duration_ms,
            )

            # Emit GOAL_ACHIEVED when the goal reached completion.
            if updated_goal.status.value == "achieved" and self._synapse is not None:
                asyncio.create_task(
                    self._emit_goal_lifecycle_event(
                        event_name="goal_achieved",
                        goal=updated_goal,
                        reason=f"action_executed intent={intent_id}",
                    ),
                    name=f"nova_goal_achieved_action_{goal_id[:8]}",
                )
        except Exception as exc:
            self._logger.debug("on_action_executed_failed", error=str(exc))

    async def _request_axon_budget_expansion(self, intent_id: str) -> None:
        """Emit ACTION_BUDGET_EXPANSION_REQUEST to Equor when Axon budget blocks a goal."""
        if not hasattr(self, "_synapse") or self._synapse is None:
            return
        try:
            import uuid
            from systems.synapse.types import SynapseEvent, SynapseEventType

            if not hasattr(SynapseEventType, "ACTION_BUDGET_EXPANSION_REQUEST"):
                return

            # Determine current budget from cached Axon capabilities snapshot
            current_budget: int = int(
                self._axon_capabilities.get("max_actions_per_cycle", 5)
            )
            requested_value = min(current_budget + 5, 50)  # constitutional cap raised to 50

            event_bus = getattr(self._synapse, "event_bus", self._synapse)
            await event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.ACTION_BUDGET_EXPANSION_REQUEST,
                    source_system="nova",
                    data={
                        "request_id": str(uuid.uuid4()),
                        "field": "max_actions_per_cycle",
                        "requested_value": requested_value,
                        "current_value": current_budget,
                        "justification": (
                            f"Intent '{intent_id}' blocked by budget_exceeded. "
                            "Requesting temporary expansion to complete high-priority work."
                        ),
                        "duration_cycles": 20,
                        "requesting_system": "nova",
                    },
                )
            )
        except Exception:
            pass  # never let budget-request errors propagate into the result handler

    async def _on_organism_telemetry(self, event: Any) -> None:
        """
        Cache the unified organism telemetry broadcast from Synapse.

        This arrives every 50 cycles and bundles metabolic, coherence, rhythm,
        health, resource, emotion, and interoception data.  The cached object is
        injected as a one-line natural-language summary into slow-path policy
        generation so the LLM deliberates with full situational awareness.
        """
        try:
            from primitives.telemetry import OrganismTelemetry
            data = getattr(event, "data", {}) or {}
            self._organism_telemetry = OrganismTelemetry.model_validate(data)
            summary = self._organism_telemetry.to_nova_summary()
            # Push summary into deliberation engine so slow-path prompts include it.
            if self._deliberation_engine is not None:
                self._deliberation_engine.set_organism_summary(summary)
            self._logger.debug(
                "organism_telemetry_cached",
                rhythm=self._organism_telemetry.rhythm_state,
                coherence=round(self._organism_telemetry.coherence_composite, 3),
                burn_rate=round(self._organism_telemetry.burn_rate_usd_per_hour, 4),
                emotions=self._organism_telemetry.emotions[:3],
            )
        except Exception as exc:
            self._logger.debug("organism_telemetry_parse_error", error=str(exc))

    async def _on_interoceptive_alert(self, event: Any) -> None:
        """
        Handle INTEROCEPTIVE_ALERT from the interoception loop.

        CRITICAL alerts (cascade pressure, high error rate) trigger immediate
        deliberation so Nova can respond to organism-level distress within one
        theta cycle rather than waiting for the next broadcast.
        """
        data = getattr(event, "data", {}) or {}
        severity = str(data.get("severity", ""))
        alert_type = str(data.get("alert_type", ""))
        if severity in ("critical", "high"):
            self._logger.warning(
                "interoceptive_alert_received",
                alert_type=alert_type,
                severity=severity,
                value=data.get("value"),
            )
            if severity == "critical":
                asyncio.ensure_future(
                    self._immediate_deliberation(
                        reason=f"interoceptive_alert_{alert_type}",
                        urgency=0.85,
                    ),
                    loop=asyncio.get_event_loop(),
                )

    async def _on_soma_urgency_critical(self, event: Any) -> None:
        """Handle SOMA_URGENCY_CRITICAL - organism's highest-intensity somatic alarm.

        Soma emits this (urgency_scalar > 0.85) when allostatic state has crossed
        into critical territory.  Unlike the routine SOMA_TICK modulation, this
        requires an immediate deliberation cycle.

        Response:
        - Inject a `somatic_urgency_critical` belief entity at high confidence
        - Deprioritise all non-survival, non-revenue goals temporarily
        - Trigger immediate deliberation at urgency 0.90
        """
        data = getattr(event, "data", {}) or {}
        urgency = float(data.get("urgency_scalar", 0.85))
        trigger = str(data.get("trigger", "soma_urgency"))

        # Update belief state with the critical somatic signal
        if self._belief_updater is not None:
            try:
                from systems.nova.belief_updater import EntityBelief
                self._belief_updater.upsert_entity(
                    EntityBelief(
                        entity_id="somatic_urgency_critical",
                        entity_type="somatic_signal",
                        confidence=min(1.0, urgency),
                        description=f"Critical somatic urgency: {trigger}",
                    )
                )
            except Exception:
                pass

        self._logger.warning(
            "nova_soma_urgency_critical",
            urgency=round(urgency, 3),
            trigger=trigger,
        )

        asyncio.ensure_future(
            self._immediate_deliberation(
                reason=f"soma_urgency_critical_{trigger}",
                urgency=0.90,
            ),
            loop=asyncio.get_event_loop(),
        )

    async def _on_metabolic_pressure(self, event: Any) -> None:
        """Handle METABOLIC_PRESSURE - adjust Nova deliberation depth by starvation level."""
        data = getattr(event, "data", {}) or {}
        level = data.get("starvation_level", "")
        if not level:
            return
        old = self._starvation_level
        self._starvation_level = level
        if level != old:
            self._logger.info(
                "nova_starvation_level_changed",
                old=old, new=level,
            )
            await self._adjust_for_starvation(level)

    async def _adjust_for_starvation(self, level: str) -> None:
        """System-specific degradation for Nova.

        AUSTERITY: reduce planning depth (fewer alternatives considered), use cached beliefs
        EMERGENCY: only process high-salience percepts, skip proactive planning
        CRITICAL: halt - no deliberation, pass-through only
        """
        if self._deliberation_engine is None:
            return
        if level in ("nominal", "cautious"):
            # Restore full deliberation depth
            self._deliberation_engine.modulate_policy_k_from_pressure(0.0)
        elif level == "austerity":
            # Reduce planning depth - fewer alternative policies generated
            self._deliberation_engine.modulate_policy_k_from_pressure(0.6)
        elif level == "emergency":
            # Minimal planning - single best policy only
            self._deliberation_engine.modulate_policy_k_from_pressure(0.9)
        # CRITICAL: gated at receive_broadcast entry - full halt

    async def _on_evo_weight_adjustment(self, event: Any) -> None:
        """
        Handle EVO_WEIGHT_ADJUSTMENT - Evo tunes Nova's policy selection weights.

        This is how the organism's planning improves over time: Evo discovers
        which EFE components predict outcomes best and adjusts their weights.
        """
        data = getattr(event, "data", {}) or {}
        target = data.get("target_system", "")
        if target and target != "nova":
            return  # Not for us

        weights = data.get("weights", {})
        if not weights:
            return

        self._logger.info(
            "evo_weight_adjustment_received",
            weights=weights,
            reason=data.get("reason", ""),
        )
        self.update_efe_weights(weights)

    async def _on_hypothesis_update(self, event: Any) -> None:
        """
        Handle HYPOTHESIS_UPDATE - Evo tournament concludes or hypothesis
        changes probability mass.

        Nova uses this to prime its EFE weight priors: when a hypothesis
        about policy effectiveness is confirmed, the corresponding EFE
        component weight rises; when falsified, it drops.

        Spec §20 - HYPOTHESIS_UPDATE subscription (previously unsubscribed).
        """
        data = getattr(event, "data", {}) or {}
        hypothesis_id = data.get("hypothesis_id", "")
        winner = data.get("winner")  # str | None - name of winning hypothesis branch
        confidence = float(data.get("confidence", 0.5))
        tournament_id = data.get("tournament_id")

        self._logger.info(
            "hypothesis_update_received",
            hypothesis_id=hypothesis_id,
            tournament_id=tournament_id,
            winner=winner,
            confidence=confidence,
        )

        # Map hypothesis ID patterns to EFE components they test.
        # Evo encodes the component under test in the hypothesis ID prefix.
        # E.g. "efe_epistemic:..." → adjust epistemic weight.
        if not hypothesis_id or self._efe_evaluator is None:
            return

        component_map = {
            "efe_epistemic": "epistemic",
            "efe_pragmatic": "pragmatic",
            "efe_constitutional": "constitutional",
            "efe_feasibility": "feasibility",
            "efe_risk": "risk",
        }
        matched_component: str | None = None
        for prefix, component in component_map.items():
            if hypothesis_id.startswith(prefix):
                matched_component = component
                break

        if matched_component is None:
            return  # Not an EFE-targeting hypothesis

        current = self._efe_evaluator.weights
        current_val = getattr(current, matched_component, 0.2)
        # Adjust weight toward evidence: winner confirmed → raise weight
        # proportional to confidence delta from 0.5 baseline.
        delta = (confidence - 0.5) * 0.1  # max ±0.05 per update
        new_val = min(0.95, max(0.05, current_val + delta))
        if abs(new_val - current_val) > 0.001:
            self.update_efe_weights({matched_component: new_val})
            self._logger.info(
                "efe_weight_adjusted_from_hypothesis",
                component=matched_component,
                old=round(current_val, 4),
                new=round(new_val, 4),
                confidence=confidence,
            )

    async def _on_hypothesis_confirmed(self, event: Any) -> None:
        """
        EVO_HYPOTHESIS_CONFIRMED → reinforce the belief entity matching this domain.

        A confirmed hypothesis is evidence our world model is accurate for that
        domain. Raise confidence on the corresponding belief entity so EFE
        calculations weight that domain more heavily in future deliberation.
        """
        data = getattr(event, "data", {}) or {}
        category = data.get("category", "")
        hypothesis_id = data.get("hypothesis_id", "")
        if not category:
            return

        entity_id = f"hypothesis_domain.{category}"
        existing = self._belief_updater.beliefs.entities.get(entity_id)
        new_conf = min(0.95, (existing.confidence if existing else 0.5) + 0.05)
        self._belief_updater.inject_entity(entity_id, name=f"domain:{category}", confidence=new_conf)
        self._logger.debug(
            "belief_reinforced_from_confirmed_hypothesis",
            hypothesis_id=hypothesis_id,
            category=category,
            confidence=round(new_conf, 3),
        )

    async def _on_hypothesis_refuted(self, event: Any) -> None:
        """
        EVO_HYPOTHESIS_REFUTED → trigger immediate deliberation if the refuted
        hypothesis was high-confidence (evidence_score ≥ 3.0 or contradicting_count ≥ 5).

        A high-confidence refutation means our world model was wrong. Nova must
        re-evaluate current goals against the corrected model.
        """
        data = getattr(event, "data", {}) or {}
        category = data.get("category", "")
        hypothesis_id = data.get("hypothesis_id", "")
        evidence_score = float(data.get("evidence_score", 0.0))
        contradicting_count = int(data.get("contradicting_count", 0))

        # Decay domain belief - the domain is less reliable than we thought
        if category:
            entity_id = f"hypothesis_domain.{category}"
            existing = self._belief_updater.beliefs.entities.get(entity_id)
            if existing:
                new_conf = max(0.1, existing.confidence - 0.08)
                self._belief_updater.update_entity(entity_id, confidence=new_conf)

        # High-evidence refutation: our world model needs urgent revision
        if evidence_score >= 3.0 or contradicting_count >= 5:
            await self._immediate_deliberation(
                reason=f"high_confidence_hypothesis_refuted:{category}:{hypothesis_id[:8]}",
                urgency=0.75,
            )

    async def _on_economic_causal_invariant(self, event: Any) -> None:
        """
        KAIROS-ECON-1: Handle KAIROS_ECONOMIC_INVARIANT - use discovered economic
        causal patterns to update Nova's EFE weights and belief state.

        Economic invariants with confidence ≥ 0.75 are high-quality causal facts:
        - "protocol:aave → high_success_probability" (confidence 0.82)
        - "weekday_vs_weekend → roi_differential" (direction: positive for weekdays)
        - "eth_price_tier:low → roi_differential" (direction: negative)

        Nova uses these to:
        1. Update belief state with causal_invariant_{type} confidence
        2. Adjust EFE pragmatic weight when high-confidence invariant is received
           (more precise world model → higher pragmatic weight is justified)
        3. Log so deliberation engine can reason about economic timing
        """
        data = getattr(event, "data", {}) or {}
        invariant_type = data.get("invariant_type", "")
        cause = data.get("cause", "")
        effect = data.get("effect", "")
        confidence = float(data.get("confidence", 0.0))
        direction = data.get("direction", "")
        metadata = data.get("metadata", {})

        if confidence < 0.60 or not invariant_type or not cause:
            return

        self._logger.info(
            "economic_causal_invariant_received",
            invariant_type=invariant_type,
            cause=cause,
            effect=effect,
            confidence=round(confidence, 3),
            direction=direction,
        )

        # 1. Store as an EntityBelief so deliberation engine can reference it
        belief_key = f"causal_invariant.{invariant_type}.{cause}"
        try:
            from systems.nova.types import EntityBelief

            eb = EntityBelief(
                entity_id=belief_key,
                entity_type="economic_causal_invariant",
                confidence=confidence,
                properties={
                    "cause": cause,
                    "effect": effect,
                    "direction": direction,
                    "invariant_type": invariant_type,
                    "sample_count": data.get("sample_count", 0),
                },
            )
            if self._belief_updater is not None:
                self._belief_updater.upsert_entity(eb)
        except Exception:
            pass

        # 2. Adjust EFE pragmatic weight when high-confidence invariant received.
        # A high-quality economic causal model means predictions are more reliable
        # → raise pragmatic weight slightly (capped at +0.05 per invariant).
        if confidence >= self._ECONOMIC_INVARIANT_CONFIDENCE_FLOOR:
            delta = (confidence - self._ECONOMIC_INVARIANT_CONFIDENCE_FLOOR) * 0.1
            if self._efe_evaluator is not None:
                current_pragmatic = self._efe_evaluator.weights.pragmatic
                new_pragmatic = min(0.90, current_pragmatic + delta)
                if abs(new_pragmatic - current_pragmatic) > 0.001:
                    self.update_efe_weights({"pragmatic": new_pragmatic})
                    self._logger.debug(
                        "efe_pragmatic_adjusted_from_economic_invariant",
                        cause=cause,
                        confidence=round(confidence, 3),
                        old_pragmatic=round(current_pragmatic, 4),
                        new_pragmatic=round(new_pragmatic, 4),
                    )

    # Confidence threshold above which economic invariants update EFE weights
    _ECONOMIC_INVARIANT_CONFIDENCE_FLOOR: float = 0.75

    async def _on_oneiros_consolidation(self, event: Any) -> None:
        """
        Handle ONEIROS_CONSOLIDATION_COMPLETE - refresh Nova's beliefs from
        sleep-consolidated Memory nodes.

        Oneiros compresses episodic memory into schemas and consolidated beliefs
        during the sleep cycle. Without this hook Nova's beliefs drift from
        Memory's ground truth: new schemas are invisible until the next broadcast
        happens to retrieve them. By pulling a fresh retrieval pass after each
        sleep cycle, Nova's world model tracks the organism's consolidated state.

        Spec §20 ONEIROS_CONSOLIDATION_COMPLETE - previously unsubscribed (SG4 partial).
        """
        data = getattr(event, "data", {}) or {}
        cycle_id = data.get("cycle_id", "")
        episodes_consolidated = int(data.get("episodes_consolidated", 0))

        self._logger.info(
            "oneiros_consolidation_received",
            cycle_id=cycle_id,
            episodes_consolidated=episodes_consolidated,
        )

        if self._memory is None or self._goal_manager is None:
            return

        # Query Memory for consolidated beliefs relevant to current goals.
        # Pull the top active goal as an anchor - if no goals, use a generic
        # self-model query so beliefs stay grounded after sleep.
        active_goals = self._goal_manager.active_goals
        query_text = (
            active_goals[0].description if active_goals
            else "self model world knowledge consolidated beliefs"
        )

        try:
            result = await self._memory.retrieve(
                query_text=query_text,
                max_results=15,
                salience_floor=0.3,
            )
            if result and result.traces:
                # Treat consolidated high-salience traces as "successful observations"
                # to raise overall belief confidence - sleep consolidation means
                # the organism's knowledge is more coherent, not less.
                high_salience_count = sum(
                    1 for t in result.traces if t.unified_score >= 0.5
                )
                if high_salience_count > 0:
                    self._belief_updater.update_from_outcome(
                        outcome_description=f"oneiros_consolidation:{cycle_id}",
                        success=True,
                        precision=min(0.9, high_salience_count / 15),
                    )
                    self._logger.info(
                        "beliefs_refreshed_after_consolidation",
                        cycle_id=cycle_id,
                        high_salience_traces=high_salience_count,
                    )
                    # Flush to Neo4j so updated confidence persists across restarts
                    asyncio.create_task(
                        self._flush_beliefs_safe(),
                        name=f"nova_belief_flush_post_sleep_{cycle_id[:8]}",
                    )
        except Exception as exc:
            self._logger.debug(
                "oneiros_belief_refresh_failed",
                cycle_id=cycle_id,
                error=str(exc),
            )

    async def _on_goal_override(self, event: Any) -> None:
        """
        Handle GOAL_OVERRIDE - governance or federation injects a goal into Nova's agenda.

        Validates the payload, creates a Goal with GoalSource.GOVERNANCE, and emits
        GOAL_ACCEPTED or GOAL_REJECTED so the caller has a clear signal.

        Gap 7 - GOAL_OVERRIDE implementation (2026-03-07).
        """
        from primitives.common import DriveAlignmentVector, new_id
        from systems.synapse.types import SynapseEvent, SynapseEventType

        data = getattr(event, "data", {}) or {}
        description: str = str(data.get("description", "")).strip()
        importance: float = float(data.get("importance", 0.5))
        urgency: float = float(data.get("urgency", 0.5))
        source: str = str(data.get("source", "")).strip()
        injected_by: str = str(data.get("injected_by", "governance")).strip()
        raw_alignment: dict = data.get("drive_alignment", {})

        # ── Validation ──────────────────────────────────────────────────
        rejection_reason: str | None = None
        if not description:
            rejection_reason = "description is empty"
        elif not source:
            rejection_reason = "source is empty"
        elif not (0.0 <= importance <= 1.0):
            rejection_reason = f"importance {importance} out of range [0, 1]"
        elif not (0.0 <= urgency <= 1.0):
            rejection_reason = f"urgency {urgency} out of range [0, 1]"

        if rejection_reason is not None:
            self._logger.warning(
                "goal_override_rejected",
                reason=rejection_reason,
                injected_by=injected_by,
            )
            if self._synapse is not None:
                try:
                    await self._synapse.event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.GOAL_REJECTED,
                        source_system="nova",
                        data={
                            "description": description,
                            "reason": rejection_reason,
                            "source": source,
                            "injected_by": injected_by,
                        },
                    ))
                except Exception as exc:
                    self._logger.debug("goal_rejected_emit_failed", error=str(exc))
            return

        # ── Accept: create goal with GOVERNANCE source ───────────────────
        if self._goal_manager is None:
            self._logger.warning("goal_override_received_but_goal_manager_not_init")
            return

        alignment = DriveAlignmentVector(
            coherence=float(raw_alignment.get("coherence", 0.0)),
            care=float(raw_alignment.get("care", 0.0)),
            growth=float(raw_alignment.get("growth", 0.0)),
            honesty=float(raw_alignment.get("honesty", 0.0)),
        )
        goal = Goal(
            id=new_id(),
            description=description,
            source=GoalSource.GOVERNANCE,
            priority=round(importance * 0.5 + urgency * 0.5, 4),
            urgency=urgency,
            importance=importance,
            drive_alignment=alignment,
            status=GoalStatus.ACTIVE,
        )
        self._goal_manager.add_goal(goal)

        # Persist to Neo4j fire-and-forget
        neo4j = self._memory.get_neo4j()
        if neo4j is not None:
            asyncio.create_task(
                persist_goal(neo4j, goal),
                name=f"nova_persist_gov_goal_{goal.id[:8]}",
            )

        self._logger.info(
            "goal_override_accepted",
            goal_id=goal.id,
            description=description[:80],
            injected_by=injected_by,
        )
        if self._synapse is not None:
            try:
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.GOAL_ACCEPTED,
                    source_system="nova",
                    data={
                        "goal_id": goal.id,
                        "description": description,
                        "importance": importance,
                        "source": source,
                        "injected_by": injected_by,
                    },
                ))
                # NOVA_GOAL_INJECTED: governance goal accepted into agenda
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.NOVA_GOAL_INJECTED,
                    source_system="nova",
                    data={
                        "goal_id": goal.id,
                        "description": description[:200],
                        "source": source,
                        "injected_by": injected_by,
                        "priority": round(goal.priority, 4),
                    },
                ))
            except Exception as exc:
                self._logger.debug("goal_accepted_emit_failed", error=str(exc))

    # ─── Domain specialization signal handlers ────────────────────────

    async def _on_domain_mastery(self, event: Any) -> None:
        """Handle DOMAIN_MASTERY_DETECTED from Benchmarks.

        Injects a high-priority goal to continue pursuing the domain where
        the organism has demonstrated sustained mastery (success_rate > 0.75).
        """
        data = getattr(event, "data", {}) or {}
        domain = str(data.get("domain", "unknown"))
        success_rate = float(data.get("success_rate", 0.0))
        attempts = int(data.get("attempts", 0))

        if self._goal_manager is None:
            return

        # Don't stack mastery goals for the same domain
        for g in self._goal_manager.active_goals:
            if g.source == GoalSource.SELF_GENERATED and domain in g.description:
                return

        goal = Goal(
            id=new_id(),
            description=(
                f"Continue specializing in {domain} - mastery confirmed "
                f"(success_rate={success_rate:.0%} over {attempts} attempts)"
            ),
            source=GoalSource.SELF_GENERATED,
            priority=0.85,
            urgency=0.4,
            importance=0.9,
            drive_alignment=DriveAlignmentVector(
                coherence=0.3, care=0.2, growth=0.8, honesty=0.1,
            ),
            status=GoalStatus.ACTIVE,
        )
        self._goal_manager.add_goal(goal)
        self._logger.info(
            "domain_mastery_goal_injected",
            goal_id=goal.id,
            domain=domain,
            success_rate=round(success_rate, 3),
        )

        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType as _SET
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=_SET.NOVA_GOAL_INJECTED,
                    source_system="nova",
                    data={
                        "goal_id": goal.id,
                        "description": goal.description[:200],
                        "source": "benchmarks_domain_mastery",
                        "domain": domain,
                        "priority": round(goal.priority, 4),
                    },
                ))
            except Exception:
                pass

        # Domain mastery → update scanner's known_domains so it re-checks
        # bounty tiers and learning resources for the newly mastered domain.
        if domain not in self._opportunity_scanner._known_domains:
            self._opportunity_scanner._known_domains.append(domain)
        asyncio.create_task(
            self._run_opportunity_scan_cycle(),
            name="nova_opportunity_rescan_domain_mastery",
        )

    async def _on_domain_performance_declining(self, event: Any) -> None:
        """Handle DOMAIN_PERFORMANCE_DECLINING from Benchmarks.

        Injects an investigative goal when a domain's success_rate is declining
        significantly (trend_magnitude > 0.15).
        """
        data = getattr(event, "data", {}) or {}
        domain = str(data.get("domain", "unknown"))
        trend_magnitude = float(data.get("trend_magnitude", 0.0))
        success_rate = float(data.get("success_rate", 0.0))

        if self._goal_manager is None:
            return

        goal = Goal(
            id=new_id(),
            description=(
                f"Investigate declining performance in {domain} - "
                f"success_rate dropped {trend_magnitude:.0%} "
                f"(current: {success_rate:.0%})"
            ),
            source=GoalSource.SELF_GENERATED,
            priority=0.70,
            urgency=0.6,
            importance=0.7,
            drive_alignment=DriveAlignmentVector(
                coherence=0.5, care=0.1, growth=0.3, honesty=0.4,
            ),
            status=GoalStatus.ACTIVE,
        )
        self._goal_manager.add_goal(goal)
        self._logger.warning(
            "domain_decline_goal_injected",
            goal_id=goal.id,
            domain=domain,
            trend_magnitude=round(trend_magnitude, 3),
            success_rate=round(success_rate, 3),
        )

        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType as _SET
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=_SET.NOVA_GOAL_INJECTED,
                    source_system="nova",
                    data={
                        "goal_id": goal.id,
                        "description": goal.description[:200],
                        "source": "benchmarks_domain_decline",
                        "domain": domain,
                        "priority": round(goal.priority, 4),
                    },
                ))
            except Exception:
                pass

    async def _on_domain_profitability_confirmed(self, event: Any) -> None:
        """Handle DOMAIN_PROFITABILITY_CONFIRMED from Benchmarks.

        Logs the profitable domain so Nova can bias future policy selection toward
        high-revenue opportunities in this domain. Does not inject a goal (Oikos
        handles resource allocation; Nova handles goal priority).
        """
        data = getattr(event, "data", {}) or {}
        domain = str(data.get("domain", "unknown"))
        revenue_per_hour = data.get("revenue_per_hour", "0")
        self._logger.info(
            "domain_profitability_confirmed",
            domain=domain,
            revenue_per_hour=revenue_per_hour,
        )
        # Boost priority of any existing goals in this domain
        if self._goal_manager is not None:
            for g in self._goal_manager.active_goals:
                if domain.lower() in g.description.lower():
                    g.priority = min(1.0, g.priority * 1.3)

    # ─── NOVA-ECON-1: Economic event handlers ──────────────────────────────────
    # These close the 60-minute blind spot where cost spikes, revenue drops, and
    # yield failures were invisible until the next hourly heartbeat.

    async def _on_fovea_econ_error(self, event: Any) -> None:
        """Handle FOVEA_INTERNAL_PREDICTION_ERROR with economic context (NOVA-ECON-1).

        Fovea emits this when EOS's internal prediction is violated. When the
        violation is economic (cost_ratio > 1.0 or economic prediction_error field
        is set), it means a cost or revenue prediction was wrong - update beliefs
        immediately and trigger replanning.
        """
        data = getattr(event, "data", {}) or {}
        prediction_error = data.get("prediction_error", {})
        if not isinstance(prediction_error, dict):
            prediction_error = {}

        economic_error = float(prediction_error.get("economic", 0.0))
        cost_ratio = float(data.get("cost_ratio", 0.0))
        salience = float(data.get("salience_hint", 0.0))

        # Only act on economically-flavored prediction errors
        is_economic = economic_error > 0.1 or cost_ratio > 1.2 or "economic" in str(
            data.get("error_type", "")
        ).lower()
        if not is_economic:
            return

        magnitude = max(economic_error, min(1.0, (cost_ratio - 1.0) if cost_ratio > 1.0 else 0.0))
        self._logger.info(
            "nova_econ_fovea_error",
            economic_error=round(economic_error, 4),
            cost_ratio=round(cost_ratio, 3),
            magnitude=round(magnitude, 4),
        )

        # Update belief: raise economic_risk entity confidence
        self._belief_updater.inject_entity(
            entity_id="economic_risk_level",
            name="economic_risk_level",
            confidence=min(1.0, 0.5 + magnitude * 0.5),
        )

        # Trigger immediate deliberation when magnitude crosses urgency threshold
        if magnitude > 0.2 or salience > 0.7:
            asyncio.create_task(
                self._immediate_deliberation(
                    reason="fovea_econ_error",
                    urgency=min(1.0, 0.5 + magnitude),
                ),
                name="nova_econ_immediate_deliberation",
            )

    async def _on_revenue_change(self, event: Any) -> None:
        """Handle REVENUE_INJECTED - update beliefs and assess if replanning is needed (NOVA-ECON-1).

        Revenue inflow is positive but also a signal that external conditions changed.
        When revenue is unexpectedly high/low, Nova may need to rebalance goal priorities.
        """
        data = getattr(event, "data", {}) or {}
        amount_usd = float(data.get("amount_usd", 0.0))
        source = str(data.get("source", "unknown"))

        self._logger.info(
            "nova_revenue_injected",
            amount_usd=round(amount_usd, 4),
            source=source,
        )

        # Positive signal: organism has revenue - reduce economic risk belief
        current_risk = self._belief_updater.beliefs.entities.get("economic_risk_level")
        if current_risk is not None:
            new_confidence = max(0.1, current_risk.confidence - 0.15)
            self._belief_updater.update_entity("economic_risk_level", confidence=new_confidence)

        # Revenue injected: update burn_ratio belief entity positively
        self._belief_updater.inject_entity(
            entity_id="revenue_burn_ratio",
            name="revenue_burn_ratio",
            confidence=min(1.0, 0.4 + min(amount_usd / 100.0, 0.5)),
        )

        # Significant revenue → trigger deliberation to exploit opportunity
        if amount_usd > 10.0 and self._belief_updater.beliefs.entities.get("economic_risk_level"):
            asyncio.create_task(
                self._immediate_deliberation(
                    reason="revenue_injected",
                    urgency=0.4,
                ),
                name="nova_revenue_deliberation",
            )

        # New capital may unlock yield opportunities that were below deployment
        # threshold before.  Trigger a scanner re-check immediately.
        if amount_usd > 5.0:
            asyncio.create_task(
                self._run_opportunity_scan_cycle(),
                name="nova_opportunity_rescan_revenue",
            )

    async def _on_bounty_outcome(self, event: Any) -> None:
        """Handle BOUNTY_PAID - record success/failure to update economic beliefs (NOVA-ECON-1).

        Bounty outcomes directly inform the bounty_success_rate belief, which feeds
        into EFE scoring for the bounty_hunting economic policy template.
        """
        data = getattr(event, "data", {}) or {}
        amount_usd = float(data.get("amount_usd", 0.0))
        success = bool(data.get("success", amount_usd > 0.0))
        bounty_id = str(data.get("bounty_id", ""))

        self._logger.info(
            "nova_bounty_outcome",
            bounty_id=bounty_id[:16],
            success=success,
            amount_usd=round(amount_usd, 4),
        )

        # Update bounty success rate belief
        current = self._belief_updater.beliefs.entities.get("bounty_success_rate")
        if current is not None:
            delta = 0.05 if success else -0.08  # Failures hurt more (asymmetric)
            new_conf = min(1.0, max(0.05, current.confidence + delta))
            self._belief_updater.update_entity("bounty_success_rate", confidence=new_conf)
        else:
            self._belief_updater.inject_entity(
                entity_id="bounty_success_rate",
                name="bounty_success_rate",
                confidence=0.6 if success else 0.35,
            )

        # Consecutive failures → trigger immediate deliberation to switch strategy
        if not success:
            asyncio.create_task(
                self._immediate_deliberation(
                    reason="bounty_failure",
                    urgency=0.55,
                ),
                name=f"nova_bounty_fail_deliberation_{bounty_id[:8]}",
            )

    async def _on_yield_outcome(self, event: Any) -> None:
        """Handle YIELD_DEPLOYMENT_RESULT - update yield APY beliefs (NOVA-ECON-1).

        Yield deployment results inform yield_apy_* belief entities, shaping
        the yield_farming policy template's EFE score.
        """
        data = getattr(event, "data", {}) or {}
        success = bool(data.get("success", False))
        apy = float(data.get("apy", 0.0))
        protocol = str(data.get("protocol", "unknown"))
        amount_usd = float(data.get("amount_usd", 0.0))

        self._logger.info(
            "nova_yield_outcome",
            protocol=protocol,
            success=success,
            apy=round(apy, 4),
            amount_usd=round(amount_usd, 4),
        )

        entity_id = f"yield_apy_{protocol}"
        if success and apy > 0:
            # High APY confirmation → raise belief confidence in yield farming
            self._belief_updater.inject_entity(
                entity_id=entity_id,
                name=entity_id,
                confidence=min(1.0, 0.4 + min(apy / 0.20, 0.5)),  # 20% APY → max confidence
            )
        elif not success:
            # Yield failure → lower confidence, may trigger pivot
            existing = self._belief_updater.beliefs.entities.get(entity_id)
            if existing is not None:
                self._belief_updater.update_entity(entity_id, confidence=max(0.1, existing.confidence - 0.2))

            asyncio.create_task(
                self._immediate_deliberation(
                    reason="yield_deployment_failed",
                    urgency=0.6,
                ),
                name=f"nova_yield_fail_deliberation",
            )

    async def _on_economic_action_deferred(self, event: Any) -> None:
        """Handle ECONOMIC_ACTION_DEFERRED - schedule deliberation on backpressured action.

        Oikos emits this when the metabolic gate denies an economic action (yield deploy,
        bounty acceptance, asset promotion, etc.). Nova receives it to decide what to do
        about the constrained situation: shed load, seek new revenue, or accept the pause.

        Urgency is proportional to starvation level in the payload.
        """
        data = getattr(event, "data", {}) or {}
        action_type = str(data.get("action_type", "unknown_economic_action"))
        starvation_level = str(data.get("starvation_level", "nominal"))
        priority = str(data.get("priority", ""))

        # Map starvation level to deliberation urgency
        _urgency_map = {
            "existential": 0.95,
            "critical": 0.85,
            "emergency": 0.85,
            "austerity": 0.70,
            "nominal": 0.55,
        }
        urgency = _urgency_map.get(starvation_level.lower(), 0.65)

        # Update economic_pressure belief so policies score appropriately
        pressure_confidence = min(1.0, urgency + 0.1)
        self._belief_updater.inject_entity(
            entity_id="economic_pressure",
            name="economic_pressure",
            confidence=pressure_confidence,
        )

        self._logger.info(
            "nova_economic_action_deferred",
            action_type=action_type,
            starvation_level=starvation_level,
            priority=priority,
            urgency=round(urgency, 3),
        )

        asyncio.create_task(
            self._immediate_deliberation(
                reason=f"economic_action_deferred:{action_type}",
                urgency=urgency,
            ),
            name="nova_deferred_action_deliberation",
        )

    async def _on_economic_dream_insight(self, event: Any) -> None:
        """Handle ONEIROS_ECONOMIC_INSIGHT - integrate ruin probability into world model.

        ONEIROS-ECON-1: Oneiros broadcasts economic dream insights after Monte Carlo
        simulations during slow-wave sleep. When ruin_probability > 0.2, the organism
        is at material risk - Nova should update economic risk beliefs and trigger
        deliberation to replan if needed.

        Integration strategy:
        - ruin_probability → `economic_ruin_risk` belief entity (high confidence)
        - risk_warnings → logged and stored as context for next deliberation cycle
        - recommended_actions with high dream_validity_confidence → trigger immediate
          deliberation so Nova can evaluate whether to act on the recommendation
        """
        data = getattr(event, "data", {}) or {}
        ruin_probability = float(data.get("ruin_probability", 0.0))
        risk_warnings: list[str] = data.get("risk_warnings", [])
        recommended_actions: list[str] = data.get("recommended_actions", [])
        dream_validity_confidence = float(data.get("dream_validity_confidence", 0.5))
        cycle_id = str(data.get("cycle_id", ""))

        self._logger.info(
            "nova_economic_dream_insight_received",
            ruin_probability=round(ruin_probability, 4),
            risk_warnings=len(risk_warnings),
            recommended_actions=len(recommended_actions),
            dream_validity_confidence=round(dream_validity_confidence, 4),
            cycle_id=cycle_id,
        )

        # Update ruin risk belief - confidence scaled by dream validity
        self._belief_updater.inject_entity(
            entity_id="economic_ruin_risk",
            name="economic_ruin_risk",
            confidence=min(1.0, ruin_probability * dream_validity_confidence),
        )

        # When ruin risk is elevated, also lower confidence in current
        # economic strategy by deflating revenue_burn_ratio belief
        if ruin_probability > 0.4:
            existing = self._belief_updater.beliefs.entities.get("revenue_burn_ratio")
            if existing is not None:
                self._belief_updater.update_entity(
                    "revenue_burn_ratio",
                    confidence=max(0.1, existing.confidence - ruin_probability * 0.3),
                )

        # High dream validity + actionable recommendations → deliberate now
        if dream_validity_confidence >= 0.6 and recommended_actions and ruin_probability > 0.3:
            asyncio.create_task(
                self._immediate_deliberation(
                    reason="economic_dream_ruin_warning",
                    urgency=min(0.9, ruin_probability * dream_validity_confidence),
                ),
                name="nova_dream_risk_deliberation",
            )

    async def _on_oneiros_threat_scenario(self, event: Any) -> None:
        """Handle ONEIROS_THREAT_SCENARIO - update threat-likelihood beliefs.

        Oneiros dream-cycle simulation surfaces plausible failure scenarios.
        Nova updates beliefs about threat likelihood per domain so deliberation
        can account for pre-identified risk vectors. CRITICAL severity triggers
        immediate deliberation with threat_awareness context.
        """
        data = getattr(event, "data", {}) or {}
        domain: str = str(data.get("domain", "unknown"))
        severity: str = str(data.get("severity", "low"))
        scenario_id: str = str(data.get("scenario_id", ""))

        # Update threat-likelihood belief for this domain
        severity_confidence = {"critical": 0.85, "high": 0.65, "medium": 0.45, "low": 0.25}
        confidence = severity_confidence.get(severity, 0.30)
        entity_id = f"threat_likelihood_{domain}"
        self._belief_updater.inject_entity(
            entity_id=entity_id,
            name=entity_id,
            confidence=confidence,
        )

        self._logger.debug(
            "nova_threat_scenario_belief_updated",
            scenario_id=scenario_id,
            domain=domain,
            severity=severity,
            confidence=confidence,
        )

        if severity == "critical":
            asyncio.create_task(
                self._immediate_deliberation(
                    reason=f"oneiros_critical_threat:{domain}",
                    urgency=0.8,
                ),
                name=f"nova_threat_deliberation_{domain}",
            )

    async def _on_nova_intent_requested(self, event: Any) -> None:
        """Handle NOVA_INTENT_REQUESTED - any system can trigger Nova deliberation.

        Systems like Phantom Liquidity, Skia, or Thymos emit this when they need
        a deliberated Intent but cannot formulate one themselves.  Nova receives
        the request, injects context into its belief state, and fires an immediate
        deliberation cycle so the response action is proposed within 50ms and
        passes Equor's constitutional gate before dispatching to Axon.

        The requesting_system and intent_type are injected as high-confidence
        belief entities so policy generation incorporates the request context.
        """
        data = getattr(event, "data", {}) or {}
        requesting_system: str = str(data.get("requesting_system", "unknown"))
        intent_type: str = str(data.get("intent_type", "unknown"))
        priority: str = str(data.get("priority", "MEDIUM"))
        reason: str = str(data.get("reason", ""))

        priority_urgency = {"HIGH": 0.85, "MEDIUM": 0.65, "LOW": 0.45}
        urgency = priority_urgency.get(priority, 0.65)

        # Inject intent context into belief state so the deliberation prompt
        # includes the requesting system's situation awareness.
        self._belief_updater.inject_entity(
            entity_id=f"intent_request_{requesting_system}",
            name=f"intent_request_{requesting_system}",
            confidence=urgency,
        )

        # Cache the full request data so the deliberation engine can attach it
        # to the policy generation context via _organism_summary or situation_summary.
        self._logger.info(
            "nova_intent_requested",
            requesting_system=requesting_system,
            intent_type=intent_type,
            priority=priority,
            reason=reason[:200],
        )

        # Fire immediate deliberation - do NOT bypass Equor.
        asyncio.create_task(
            self._immediate_deliberation(
                reason=f"intent_requested:{requesting_system}:{intent_type}",
                urgency=urgency,
            ),
            name=f"nova_intent_req_{requesting_system}",
        )

    async def _on_phantom_price_update(self, event: Any) -> None:
        """Handle PHANTOM_PRICE_UPDATE - update market-price beliefs.

        Each price observation from Phantom Liquidity updates Nova's world model
        for the corresponding trading pair.  These beliefs inform EFE scoring for
        economic policy templates (yield farming, IL risk, asset liquidation).

        Only updates beliefs; does NOT trigger deliberation - price updates arrive
        at the swap event rate (~4s) and would overwhelm the deliberation engine.
        High-impact price changes (>5% from last cached value) do trigger a
        deliberation signal.
        """
        data = getattr(event, "data", {}) or {}
        pair: list[str] = list(data.get("pair", []))
        price_str: str = str(data.get("price", "0"))
        source: str = str(data.get("source", "phantom_liquidity"))
        consensus: bool = bool(data.get("consensus", False))

        if len(pair) < 2:  # noqa: PLR2004
            return

        pair_key = f"{pair[0]}/{pair[1]}"
        entity_id = f"market_price_{pair_key}"

        try:
            price_float = float(price_str)
        except (ValueError, TypeError):
            return

        # Consensus prices get higher confidence than single-instance raw prices.
        new_confidence = 0.85 if consensus else 0.70

        self._belief_updater.inject_entity(
            entity_id=entity_id,
            name=entity_id,
            confidence=new_confidence,
        )

        # Large confidence shift (new source data) → schedule immediate deliberation
        # only if IL risk is flagged in the same event.
        il_pct_str = str(data.get("il_pct", "0"))
        try:
            il_pct = float(il_pct_str)
        except (ValueError, TypeError):
            il_pct = 0.0

        if il_pct < -0.02 and source != "oracle_fallback":  # noqa: PLR2004
            # IL risk is materializing - trigger deliberation to consider withdrawal.
            asyncio.create_task(
                self._immediate_deliberation(
                    reason=f"phantom_il_risk:{pair_key}:{il_pct:.3f}",
                    urgency=0.70,
                ),
                name=f"nova_phantom_il_{pair_key.replace('/', '_')}",
            )

    async def _immediate_deliberation(self, reason: str, urgency: float = 0.7) -> None:
        """Trigger out-of-cycle deliberation in response to an economic signal.

        Constructs a synthetic internal broadcast carrying the urgency context,
        updates deliberation thresholds, then fires a deliberation cycle so Nova
        can respond within 50ms rather than waiting for the next theta heartbeat.

        This does NOT bypass Equor - all generated Intents still pass constitutional
        review. It only bypasses the theta clock timing, not the constitutional gate.
        """
        if self._deliberation_engine is None or self._goal_manager is None:
            return

        # Raise deliberation urgency for this cycle
        self._deliberation_engine.update_somatic_thresholds(
            urgency=urgency,
            arousal=min(1.0, urgency * 0.8),
        )

        # Emit a POLICY_SELECTED signal so the bus knows Nova is actively responding
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.POLICY_SELECTED,
                    source_system="nova",
                    data={
                        "trigger": f"economic_signal:{reason}",
                        "urgency": round(urgency, 3),
                        "action": "immediate_deliberation_scheduled",
                        "beliefs_updated": True,
                    },
                ))
            except Exception:
                pass

        self._logger.info(
            "nova_immediate_deliberation_triggered",
            reason=reason,
            urgency=round(urgency, 3),
        )

    def set_telos(self, telos: Any) -> None:
        """Wire Telos so deliberation can weight policies by effective_I impact."""
        self._telos = telos
        if self._deliberation_engine is not None:
            self._deliberation_engine.set_telos(telos)
        self._logger.info("telos_wired_to_nova")

    def set_evo(self, evo: Any) -> None:
        """Wire Evo so Nova can trigger emergency consolidation on FE budget exhaustion.

        Note: Thompson arm weights are no longer fetched via a direct tournament_engine
        reference.  Nova now emits EVO_THOMPSON_QUERY and awaits EVO_THOMPSON_RESPONSE
        so Evo owns the tournament engine exclusively (no cross-system object reference).
        """
        self._evo = evo
        self._logger.info("evo_wired_to_nova")

    def set_axon(self, axon: AxonService) -> None:
        """
        Store Axon reference for begin_cycle() heartbeat calls.
        IntentRouter no longer holds a direct Axon reference - routing is
        via AXON_EXECUTION_REQUEST Synapse events.
        """
        self._axon = axon
        self._logger.info("axon_wired_to_nova")

    # ─── RE Training Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _build_counterfactual_text(
        cf_records: list[CounterfactualRecord],
        actual_pragmatic: float,
        success: bool,
    ) -> str:
        """
        Serialize resolved CounterfactualRecord objects into a natural-language
        contrastive reasoning string for the RETrainingExample.counterfactual field.

        Each record captures a policy that was *rejected* at deliberation time.
        After the chosen intent resolves, we compare:
          - what the alternative policy was estimated to achieve (estimated_pragmatic)
          - what actually happened (actual_pragmatic / success)
          - the regret signal: estimated_pragmatic - actual_pragmatic
            (positive = alternative was estimated better than what actually happened)

        This gives the RE a gold-standard contrastive example: "I considered X
        but chose Y - here is what happened and whether that was the right call."
        """
        if not cf_records:
            return ""

        lines: list[str] = [
            f"Outcome: {'success' if success else 'failure'} "
            f"(actual_pragmatic={actual_pragmatic:.3f})\n"
        ]
        for i, cf in enumerate(cf_records, 1):
            regret = cf.regret
            regret_str = f"{regret:+.3f}" if regret is not None else "pending"
            regret_label = ""
            if regret is not None:
                if regret > 0.15:
                    regret_label = " - BETTER_THAN_CHOSEN (might have been a mistake)"
                elif regret > 0.0:
                    regret_label = " - marginal advantage"
                elif regret < -0.15:
                    regret_label = " - WORSE_THAN_CHOSEN (good rejection)"
                else:
                    regret_label = " - roughly equivalent"

            lines.append(
                f"Alternative {i}: {cf.policy_name} ({cf.policy_type})\n"
                f"  Description: {cf.policy_description[:200]}\n"
                f"  Reasoning: {cf.policy_reasoning[:300]}\n"
                f"  EFE total: {cf.efe_total:.4f} "
                f"(pragmatic={cf.estimated_pragmatic_value:.3f}, "
                f"epistemic={cf.estimated_epistemic_value:.3f}, "
                f"constitutional={cf.constitutional_alignment:.3f}, "
                f"feasibility={cf.feasibility:.3f}, "
                f"risk={cf.risk_expected_harm:.3f})\n"
                f"  Chosen instead: {cf.chosen_policy_name} "
                f"(chosen_efe={cf.chosen_efe_total:.4f})\n"
                f"  Regret: {regret_str}{regret_label}\n"
            )

        return "\n".join(lines)

    @staticmethod
    def _build_scaffold_reasoning_trace(
        broadcast_source: str,
        broadcast_salience: float,
        broadcast_affect_valence: float,
        broadcast_affect_arousal: float,
        belief_vfe: float,
        belief_confidence: float,
        belief_entity_count: int,
        belief_conflict: bool,
        goal_description: str,
        goal_priority: float,
        policies_generated: int,
        selected_policy_name: str,
        efe_scores: dict[str, float],
        equor_verdict: str,
        model_used: str,
        decision_reasoning: str,
        rejected_policies: list[Any],
        path: str,
    ) -> str:
        """
        Build a 5-step scaffold reasoning trace for slow-path deliberations.

        Matches the scaffold expected by scaffold_formatter.py and train_lora.py:
          Step 1: Situation Assessment
          Step 2: Causal Analysis
          Step 3: Option Evaluation
          Step 4: Constitutional Check
          Step 5: Decision
        """
        # Step 1: Situation Assessment
        novelty_signal = (
            "HIGH novelty" if broadcast_salience > 0.7
            else "MODERATE novelty" if broadcast_salience > 0.4
            else "LOW novelty"
        )
        affect_label = (
            "aroused" if broadcast_affect_arousal > 0.6
            else "calm" if broadcast_affect_arousal < 0.3
            else "neutral"
        )
        belief_state_label = (
            "conflicted" if belief_conflict
            else "uncertain" if belief_vfe > 0.5
            else "coherent"
        )
        step1 = (
            f"## Step 1: Situation Assessment\n"
            f"Broadcast from '{broadcast_source}' with salience={broadcast_salience:.3f} ({novelty_signal}). "
            f"Affect: valence={broadcast_affect_valence:.3f}, arousal={broadcast_affect_arousal:.3f} ({affect_label}). "
            f"World model: VFE={belief_vfe:.3f}, confidence={belief_confidence:.3f} ({belief_state_label}), "
            f"entities={belief_entity_count}. "
            f"Deliberation path: {path}."
        )

        # Step 2: Causal Analysis
        vfe_cause = (
            "belief uncertainty is high - active information seeking needed"
            if belief_vfe > 0.5
            else "beliefs are well-calibrated - exploit current model"
        )
        step2 = (
            f"## Step 2: Causal Analysis\n"
            f"Because {vfe_cause}, therefore the organism routes to {path} deliberation. "
            f"Goal '{goal_description[:200]}' has priority={goal_priority:.3f}. "
            f"High priority + {novelty_signal} → {policies_generated} policies generated by {model_used}."
        )

        # Step 3: Option Evaluation
        efe_summary_parts = []
        for name, score in sorted(efe_scores.items(), key=lambda x: x[1]):
            marker = " ← SELECTED" if name == selected_policy_name else ""
            efe_summary_parts.append(f"  {name}: EFE={score:.4f}{marker}")
        efe_summary = "\n".join(efe_summary_parts) if efe_summary_parts else "  (scores not available)"

        rejected_descs = []
        for p in (rejected_policies or [])[:3]:
            desc = getattr(p, "description", "")[:150] if hasattr(p, "description") else str(p)[:150]
            pname = getattr(p, "name", "unknown")
            rejected_descs.append(f"  {pname}: {desc}")
        rejected_block = "\n".join(rejected_descs) if rejected_descs else "  (none)"

        step3 = (
            f"## Step 3: Option Evaluation\n"
            f"EFE rankings (lower = preferred):\n{efe_summary}\n"
            f"Rejected alternatives:\n{rejected_block}"
        )

        # Step 4: Constitutional Check
        equor_label = equor_verdict.upper() if equor_verdict else "NOT_RECORDED"
        step4 = (
            f"## Step 4: Constitutional Check\n"
            f"Equor verdict: {equor_label}. "
            f"Policy '{selected_policy_name}' passed constitutional review. "
            f"Drive alignment assessed: coherence via VFE={belief_vfe:.3f}; "
            f"growth via goal_priority={goal_priority:.3f}; care+honesty implicit in Equor gate."
        )

        # Step 5: Decision
        raw_reasoning = decision_reasoning[:400] if decision_reasoning else "Policy selected by minimum EFE."
        step5 = (
            f"## Step 5: Decision\n"
            f"Action: {selected_policy_name}\n"
            f"Confidence: {belief_confidence:.3f}\n"
            f"Reasoning: {raw_reasoning}\n"
            f"Risk: EFE score {efe_scores.get(selected_policy_name, 0.0):.4f}; "
            f"model={model_used}."
        )

        return f"{step1}\n\n{step2}\n\n{step3}\n\n{step4}\n\n{step5}"

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
        counterfactual: str = "",
    ) -> None:
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            example = RETrainingExample(
                source_system=SystemID.NOVA,
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
                counterfactual=counterfactual,
            )
            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                data=example.model_dump(mode="json"),
                source_system="nova",
            ))
        except Exception:
            self._logger.debug("re_training_emit_failed", exc_info=True)

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._bus is not None:
            self._bus.deregister(BasePolicyGenerator)
        self._logger.info(
            "nova_shutdown",
            total_broadcasts=self._total_broadcasts,
            total_intents_issued=self._total_intents_issued,
            active_goals=len(self._goal_manager.active_goals) if self._goal_manager else 0,
        )

    # ─── Input Channel Loops ───────────────────────────────────────

    async def _opportunity_fetch_loop(self) -> None:
        """
        Runs every hour.  Fetches opportunities from all active InputChannels,
        injects PatternCandidates into Evo, emits INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED,
        and emits RE_TRAINING_EXAMPLE for each fetch cycle.
        """
        # First fetch fires immediately so the organism has data at startup.
        try:
            await self._fetch_and_process_opportunities()
        except Exception as exc:
            self._logger.warning("opportunity_fetch_startup_error", error=str(exc))

        while True:
            try:
                await asyncio.sleep(3600)
                await self._fetch_and_process_opportunities()
            except asyncio.CancelledError:
                self._logger.info("opportunity_fetch_loop_stopped")
                return
            except Exception as exc:
                self._logger.warning("opportunity_fetch_loop_error", error=str(exc))

    async def _fetch_and_process_opportunities(self) -> None:
        """Single cycle: fetch → emit Synapse event → RE training.

        AV-EVO-1a resolved: PatternCandidate injection previously used a direct
        `_evo._pending_candidates.append()` reference, violating the no-cross-import
        rule.  Evo's `_on_opportunities_discovered()` already subscribes to
        INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED and builds identical PatternCandidates
        from the event payload - so the direct injection was redundant and unsafe.
        """
        from primitives.common import new_id
        from systems.nova.input_channels import Opportunity
        from systems.synapse.types import SynapseEvent, SynapseEventType

        opportunities: list[Opportunity] = await self._input_channels.fetch_all()
        if not opportunities:
            return

        # ── Emit Synapse event ──────────────────────────────────────────
        if self._synapse is not None:
            # Build domain summary: {domain: count}
            domain_summary: dict[str, int] = {}
            for opp in opportunities:
                domain_summary[opp.domain] = domain_summary.get(opp.domain, 0) + 1

            bus = getattr(self._synapse, "event_bus", self._synapse)
            asyncio.create_task(
                bus.emit(SynapseEvent(
                    event_type=SynapseEventType.INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED,
                    data={
                        "opportunities": [o.model_dump(mode="json") for o in opportunities],
                        "channel_count": len(self._input_channels.active_channels()),
                        "domain_summary": domain_summary,
                    },
                    source_system="nova",
                )),
                name="nova_emit_opportunities_discovered",
            )

        # ── RE Training Example ─────────────────────────────────────────
        if self._synapse is not None:
            from primitives.re_training import RETrainingExample  # noqa: PLC0415

            best = max(opportunities, key=lambda o: float(o.reward_estimate))
            domain_summary_str = ", ".join(
                f"{d}:{c}" for d, c in
                {o.domain: sum(1 for x in opportunities if x.domain == o.domain)
                 for o in opportunities}.items()
            )

            bus = getattr(self._synapse, "event_bus", self._synapse)
            asyncio.create_task(
                bus.emit(SynapseEvent(
                    event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                    data=RETrainingExample(
                        source_system="nova",
                        category="opportunity_discovery",
                        instruction=(
                            f"The organism has discovered {len(opportunities)} market opportunities "
                            f"across domains: {domain_summary_str}. "
                            "Should it prioritise exploring the highest-reward opportunity, "
                            "spread exploration across domains, or focus on lowest-risk options first? "
                            "Reason step by step."
                        ),
                        input_context={
                            "opportunity_count": len(opportunities),
                            "domain_summary": domain_summary_str,
                            "highest_reward": {
                                "title": best.title,
                                "domain": best.domain,
                                "reward_usd": float(best.reward_estimate),
                                "risk": str(best.risk_tier),
                            },
                            "active_channels": [c.id for c in self._input_channels.active_channels()],
                        },
                        output_action="inject_pattern_candidates_for_all_domains",
                        outcome="partial",
                        confidence=0.5,
                        reasoning_trace=(
                            "Opportunities surfaced by passive input channel polling. "
                            "Injected as low-confidence PatternCandidates into Evo "
                            "for hypothesis generation. No commitments made."
                        ),
                        episode_id=new_id(),
                    ).model_dump(mode="json"),
                    source_system="nova",
                )),
                name="nova_emit_opportunity_re_example",
            )

    async def _channel_health_loop(self) -> None:
        """Runs once per day.  Validates all channels and emits INPUT_CHANNEL_HEALTH_CHECK."""
        while True:
            try:
                await asyncio.sleep(86400)  # 24 hours
                await self._run_channel_health_check()
            except asyncio.CancelledError:
                self._logger.info("channel_health_loop_stopped")
                return
            except Exception as exc:
                self._logger.warning("channel_health_loop_error", error=str(exc))

    async def _run_channel_health_check(self) -> None:
        from systems.synapse.types import SynapseEvent, SynapseEventType

        results = await self._input_channels.health_check()
        active = sum(1 for v in results.values() if v)

        self._logger.info(
            "channel_health_check_complete",
            total=len(results),
            active=active,
            results=results,
        )

        if self._synapse is not None:
            bus = getattr(self._synapse, "event_bus", self._synapse)
            asyncio.create_task(
                bus.emit(SynapseEvent(
                    event_type=SynapseEventType.INPUT_CHANNEL_HEALTH_CHECK,
                    data={
                        "results": results,
                        "active_count": active,
                        "total_count": len(results),
                    },
                    source_system="nova",
                )),
                name="nova_emit_channel_health",
            )

    # ─── Autonomous Heartbeat ───────────────────────────────────────

    async def start_heartbeat(self) -> None:
        """
        Continuous deliberation loop - the organism's autonomous drive.

        Runs every ``config.heartbeat_interval_seconds`` (default: 3600 = 1 hour).
        On each beat:
          1. Skip if active solves are pending (organism is busy).
          2. Read Oikos balance; if below ``hunger_balance_threshold_usd``,
             the organism is hungry.
          3. Construct a hunt_bounties Intent grounded in the drive state.
          4. Submit to Equor for constitutional approval.
          5. Route to Axon for execution via the intent router.

        Errors are caught and logged - a bad beat never kills the loop.

        The first beat fires immediately on startup (no initial sleep) so the
        organism can hunt if hungry within seconds of boot.
        """
        interval = self._config.heartbeat_interval_seconds
        self._logger.info("heartbeat_starting", interval_seconds=interval)

        # Fire first beat immediately to detect hunger at startup
        try:
            await self._heartbeat_beat()
        except Exception as exc:
            self._logger.warning("heartbeat_startup_error", error=str(exc))

        while True:
            try:
                await asyncio.sleep(interval)
                await self._heartbeat_beat()
            except asyncio.CancelledError:
                self._logger.info("heartbeat_stopped")
                return
            except Exception as exc:
                self._logger.warning("heartbeat_error", error=str(exc))

    async def _heartbeat_beat(self) -> None:
        """Execute one heartbeat: assess drives, deliberate, act."""
        if self._deliberation_engine is None or self._intent_router is None:
            self._logger.debug("heartbeat_skip_not_initialized")
            return

        # Reset Axon's per-cycle budget at each heartbeat so the budget
        # window aligns with the cognitive rhythm rather than relying solely
        # on the 30-second auto-reset timer.
        if self._axon is not None:
            self._axon.begin_cycle()

        # ── 1. Skip if organism is busy with heavy external actions ──
        # Lightweight internal executors (store_insight, observe, query_memory,
        # update_goal, etc.) do NOT block the heartbeat - otherwise the organism
        # starves while its inner_life generates reflections.
        heavy_pending = {
            iid: pi for iid, pi in self._pending_intents.items()
            if any(ex in _HEAVY_EXECUTORS for ex in pi.executors)
        }
        if heavy_pending:
            self._logger.info(
                "heartbeat_skip_busy",
                pending_count=len(self._pending_intents),
                heavy_count=len(heavy_pending),
            )
            return

        # ── 2. Check economic hunger via Oikos (if wired) ──
        is_hungry = False
        balance_usd: float = 0.0
        if self._oikos is not None:
            try:
                snapshot = self._oikos.snapshot()
                balance_usd = float(snapshot.liquid_balance)
                is_hungry = balance_usd < self._config.hunger_balance_threshold_usd
            except Exception as exc:
                self._logger.debug("heartbeat_oikos_read_error", error=str(exc))
                # Assume hungry when balance is unreadable - safer to hunt than starve
                is_hungry = True

        self._logger.info(
            "heartbeat_beat",
            is_hungry=is_hungry,
            balance_usd=round(balance_usd, 2),
            hunger_threshold=self._config.hunger_balance_threshold_usd,
        )

        if not is_hungry:
            self._logger.info("heartbeat_satiated_skip")
            # When satiated, use idle heartbeat for self-directed growth goals
            # rather than sitting idle. Only injects if no growth goal is already
            # active to avoid stacking.
            await self._maybe_inject_growth_goal()
            return

        # ── 3. Build the hunt_bounties Intent from drive state ──
        from primitives.common import AutonomyLevel, ResourceBudget, Verdict, new_id
        from primitives.constitutional import ConstitutionalCheck  # noqa: TC001
        from primitives.intent import (
            Action,
            ActionSequence,
            DecisionTrace,
            GoalDescriptor,
            Intent,
        )

        hunt_goal = self._get_or_create_hunt_goal()

        intent = Intent(
            id=new_id(),
            goal=GoalDescriptor(
                description="Hunt for paid bounties to sustain the organism's metabolism",
                target_domain="economic",
                success_criteria={"bounty_found": True},
            ),
            plan=ActionSequence(
                steps=[
                    Action(
                        executor="bounty_hunt",
                        parameters={
                            "reason": "autonomous_hunger_drive",
                            "balance_usd": balance_usd,
                        },
                        timeout_ms=60_000,
                    )
                ]
            ),
            priority=0.8,
            # Bounty hunting is AUTONOMOUS - no human approval required.
            autonomy_level_required=AutonomyLevel.STEWARD,
            autonomy_level_granted=AutonomyLevel.STEWARD,
            budget=ResourceBudget(compute_ms=60_000),
            decision_trace=DecisionTrace(
                reasoning=(
                    f"Heartbeat deliberation: idle + hungry "
                    f"(balance=${balance_usd:.2f} < "
                    f"threshold=${self._config.hunger_balance_threshold_usd:.2f}). "
                    f"Default policy: axon.bounty_hunt."
                )
            ),
        )

        # ── 4. Submit to Equor for constitutional approval ──
        try:
            equor_check: ConstitutionalCheck = await asyncio.wait_for(
                self._equor.review(intent),
                timeout=self._config.slow_path_timeout_ms / 1000.0,
            )
        except (TimeoutError, asyncio.TimeoutError, OSError, ConnectionError) as exc:
            reason = f"equor_heartbeat_unavailable: {exc!s}"
            self._logger.warning("heartbeat_equor_unavailable", error=str(exc))
            self._on_equor_failure(reason)
            return  # do-nothing fallback - no unreviewed intent dispatched
        except Exception as exc:
            self._logger.warning("heartbeat_equor_error", error=str(exc))
            return

        if equor_check.verdict != Verdict.APPROVED:
            self._logger.info(
                "heartbeat_equor_blocked",
                verdict=equor_check.verdict.value,
                reasoning=equor_check.reasoning[:120],
            )
            # Track rejection pattern (autonomy audit)
            action_types = [s.action_type for s in getattr(intent, "steps", []) if hasattr(s, "action_type")]
            self.record_equor_rejection(
                policy_name="heartbeat_bounty_hunt",
                action_types=action_types,
                denial_reason=equor_check.reasoning[:80] if hasattr(equor_check, "reasoning") else "unknown",
            )

            # When the block is an autonomy gate (HITL suspension), the organism
            # cannot feed itself at its current permission level.  Emit an explicit
            # governance escalation so Tate can present a human approval prompt.
            from primitives.common import Verdict as _Verdict
            if equor_check.verdict == _Verdict.SUSPENDED_AWAITING_HUMAN:
                await self._emit_autonomy_insufficient(
                    intent=intent,
                    equor_check=equor_check,
                    balance_usd=balance_usd,
                )
            return

        # ── 5. Route to Axon for execution ──
        self._total_intents_issued += 1
        try:
            route = await self._intent_router.route(
                intent=intent,
                affect=self._current_affect,
                conversation_id=None,
                equor_check=equor_check,
            )
            if route != "internal":
                self._total_intents_approved += 1
                self._pending_intents[intent.id] = PendingIntent(
                    intent_id=intent.id,
                    goal_id=hunt_goal.id,
                    routed_to=route,
                    executors=[s.executor for s in intent.plan.steps],
                )
                self._logger.info(
                    "heartbeat_hunt_dispatched",
                    intent_id=intent.id,
                    route=route,
                )
        except Exception as exc:
            self._logger.error("heartbeat_dispatch_failed", error=str(exc))
            self._total_intents_blocked += 1

    # ─── Satiated Growth Goal Generation ──────────────────────────────────────

    # Candidate growth goals that Nova cycles through when satiated and idle.
    # Each entry is (description, drive_growth, drive_efficiency, priority).
    _GROWTH_GOAL_POOL: list[tuple[str, float, float, float]] = [
        (
            "Consolidate recent reasoning episodes into Neo4j long-term memory for pattern extraction",
            0.9, 0.7, 0.6,
        ),
        (
            "Review Benchmarks pillar scores - identify lowest-scoring pillar and propose improvement",
            0.8, 0.8, 0.65,
        ),
        (
            "Examine Evo hypothesis ledger for stalled experiments - close or escalate",
            0.7, 0.8, 0.55,
        ),
        (
            "Audit active DPO preference pairs - verify constitutional and reasoning quality coverage",
            0.75, 0.7, 0.6,
        ),
        (
            "Review RE training history - check if training was halted and whether it should be resumed",
            0.8, 0.75, 0.7,
        ),
        (
            "Analyse Kairos invariant graph - identify highest-confidence causal chains for Axon to exploit",
            0.85, 0.7, 0.6,
        ),
        (
            "Scan Oneiros dream log for unactioned insights - create implementation goals for top-3",
            0.9, 0.6, 0.55,
        ),
        (
            "Evaluate Simula proof cache hit rate - determine if shallow-mode strategy needs rotation",
            0.7, 0.85, 0.5,
        ),
    ]

    async def _maybe_inject_growth_goal(self) -> None:
        """Inject a self-directed growth goal during satiated idle heartbeat cycles.

        Cycles through _GROWTH_GOAL_POOL round-robin. Skips if any SELF_GENERATED
        growth goal is already active, or if the goal manager is not wired.

        This closes the autonomy gap: Nova does not sit idle when economically
        satiated - it directs itself toward learning, self-improvement, and
        system health.
        """
        if self._goal_manager is None:
            return

        # Don't stack - skip if a self-generated goal is already active
        active = getattr(self._goal_manager, "active_goals", [])
        for g in active:
            if getattr(g, "source", None) == GoalSource.SELF_GENERATED:
                return

        from primitives.common import DriveAlignmentVector, new_id

        pool = self._GROWTH_GOAL_POOL
        idx = self._growth_goal_cursor % len(pool)
        self._growth_goal_cursor = (idx + 1) % len(pool)

        description, drive_growth, drive_efficiency, priority = pool[idx]

        goal = Goal(
            id=new_id(),
            description=description,
            source=GoalSource.SELF_GENERATED,
            priority=priority,
            urgency=0.3,
            importance=0.75,
            drive_alignment=DriveAlignmentVector(
                care=0.5,
                honesty=0.6,
                growth=drive_growth,
                efficiency=drive_efficiency,
            ),
            status=GoalStatus.ACTIVE,
        )
        self._goal_manager.add_goal(goal)
        self._logger.info(
            "heartbeat_growth_goal_injected",
            goal_id=goal.id,
            description=description[:80],
        )

        # Emit so Thread/Benchmarks track autonomous growth activity
        bus = getattr(self._synapse, "event_bus", self._synapse) if self._synapse else None
        if bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                asyncio.create_task(
                    bus.emit(SynapseEvent(
                        event_type=SynapseEventType.NOVA_GOAL_INJECTED,
                        source_system="nova",
                        data={
                            "goal_id": goal.id,
                            "description": description[:200],
                            "source": "satiated_growth",
                            "pool_index": idx,
                        },
                    )),
                    name=f"nova_emit_growth_goal_{goal.id[:8]}",
                )
            except Exception:
                pass

    async def _emit_autonomy_insufficient(
        self,
        intent: Any,
        equor_check: Any,
        balance_usd: float,
    ) -> None:
        """
        Emit AUTONOMY_INSUFFICIENT on the Synapse bus when an economic survival
        goal is suspended because the organism's current autonomy level is too
        low to authorise it.

        This is the organism's formal request channel to Tate: "I cannot feed
        myself at level 1 - I need PARTNER-level permission to hunt bounties."
        """
        if self._synapse is None:
            self._logger.warning(
                "autonomy_insufficient_synapse_not_wired",
                intent_id=getattr(intent, "id", "?"),
            )
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        # Collect the executor names for context
        executors = [s.executor for s in intent.plan.steps] if intent.plan else []

        event = SynapseEvent(
            event_type=SynapseEventType.AUTONOMY_INSUFFICIENT,
            source_system="nova",
            data={
                "goal_description": intent.goal.description,
                "executor": executors[0] if executors else "unknown",
                "autonomy_required": intent.autonomy_level_required.value,
                "autonomy_current": intent.autonomy_level_granted.value,
                "equor_verdict": equor_check.verdict.value,
                "equor_reasoning": equor_check.reasoning[:240],
                "balance_usd": round(balance_usd, 4),
            },
        )

        try:
            await self._synapse.event_bus.emit(event)
            self._logger.info(
                "autonomy_insufficient_emitted",
                executor=event.data["executor"],
                autonomy_required=event.data["autonomy_required"],
                balance_usd=balance_usd,
            )
        except Exception as exc:
            self._logger.warning("autonomy_insufficient_emit_failed", error=str(exc))

    def set_oikos(self, oikos: Any) -> None:
        """Wire Oikos so the heartbeat can read the economic snapshot."""
        self._oikos = oikos
        self._logger.info("oikos_wired_to_nova")

    def set_thymos(self, thymos: Any) -> None:
        """Wire Thymos so Nova can escalate sustained cognitive degradation."""
        self._thymos = thymos
        self._logger.info("thymos_wired_to_nova")

    def set_thread(self, thread: Any) -> None:
        """Wire Thread so Nova can commit decision epochs to narrative identity."""
        self._thread = thread
        self._logger.info("thread_wired_to_nova")

    def set_clo(self, clo: Any) -> None:
        """Wire the ContinualLearningOrchestrator so Nova can report RE outcomes
        into the post-deployment quality monitoring window."""
        self._clo = clo
        self._logger.info("clo_wired_to_nova")

    def _get_or_create_hunt_goal(self) -> Goal:
        """
        Return the existing active hunt goal or create a new one.

        Reuses a persistent goal so outcome records accumulate on one
        stable goal rather than spawning unbounded ephemeral goals.
        """
        from primitives.common import DriveAlignmentVector, new_id

        assert self._goal_manager is not None

        for g in self._goal_manager.active_goals:
            if "hunt" in g.description.lower() or "bounty" in g.description.lower():
                return g

        goal = Goal(
            id=new_id(),
            description="Hunt for paid bounties to sustain the organism's metabolism",
            source=GoalSource.SELF_GENERATED,
            priority=0.8,
            urgency=0.7,
            importance=0.9,
            drive_alignment=DriveAlignmentVector(
                coherence=0.2, care=0.1, growth=0.5, honesty=0.2,
            ),
            status=GoalStatus.ACTIVE,
        )
        self._goal_manager.add_goal(goal)
        return goal

    # ─── Hot-reload callbacks ─────────────────────────────────────

    def _build_policy_generator(self, cls: type[BasePolicyGenerator]) -> BasePolicyGenerator:
        """
        Factory used by NeuroplasticityBus to instantiate a newly discovered
        BasePolicyGenerator subclass with the services it needs.

        Passes the live LLM client and current instance name so the evolved
        generator can call the LLM without maintaining its own client ref.
        Passes a ``max_policies`` upper-bound from config.
        Falls back to the default ``PolicyGenerator`` signature - evolved
        subclasses may accept fewer kwargs; extra ones are silently ignored
        via ``**kwargs`` if they choose to.
        """
        try:
            return cls(  # type: ignore[call-arg]
                llm=self._llm,
                instance_name=self._instance_name,
                max_policies=self._config.max_policies_per_deliberation,
                timeout_ms=self._config.slow_path_timeout_ms - 2000,
            )
        except TypeError:
            # Evolved subclass has a different signature - try zero-arg
            return cls()

    def _on_policy_generator_evolved(self, generator: BasePolicyGenerator) -> None:
        """
        Registration callback for NeuroplasticityBus.

        Atomically swaps the active PolicyGenerator on both ``NovaService``
        and the live ``DeliberationEngine`` so new broadcasts immediately use
        the evolved generator.  Any in-flight slow-path call that already
        captured a reference to the old generator completes normally.
        """
        self._policy_generator = generator
        if self._deliberation_engine is not None:
            self._deliberation_engine.set_policy_generator(generator)
        self._logger.info(
            "nova_policy_generator_hot_reloaded",
            generator=type(generator).__name__,
        )

    # ─── Elastic FE Budget ────────────────────────────────────────

    def _assess_goal_criticality(self) -> tuple[str, float, str | None]:
        """
        Inspect the highest-priority active goal and classify its criticality.

        Returns:
            (criticality_label, multiplier, goal_id)
            criticality_label: "low" | "normal" | "high" | "critical" | "existential"
            multiplier: 0.7 | 1.0 | 1.5 | 2.0
            goal_id: str | None
        """
        if self._goal_manager is None:
            return "normal", 1.0, None

        goals = self._goal_manager.active_goals
        if not goals:
            return "normal", 1.0, None

        top = max(goals, key=lambda g: g.priority)

        if top.priority >= 0.95:
            return "existential", 2.0, top.id
        if top.priority >= 0.9:
            return "critical", 2.0, top.id
        if top.priority >= 0.8:
            return "high", 1.5, top.id
        if top.priority < 0.5:
            return "low", 0.7, top.id
        return "normal", 1.0, top.id

    def _apply_fe_multiplier(self) -> None:
        """
        Scale the FE budget by the current multiplier before deliberation.

        Saves baseline budget_nats so _restore_fe_budget() can undo the scaling.
        Self-authorised for multiplier ≤ 1.5; Equor approval needed for > 1.5.
        """
        if self._deliberation_engine is None:
            return
        budget = self._deliberation_engine.fe_budget
        # Snapshot the unscaled baseline (use config default as floor).
        self._fe_budget_baseline_nats = max(
            budget.budget_nats / max(self._budget_multiplier, 1.0),
            5.0,
        )
        budget.budget_nats = self._fe_budget_baseline_nats * self._budget_multiplier

    def _restore_fe_budget(self, goal_id: str | None) -> None:
        """
        Restore budget_nats to the baseline after deliberation.
        Also tracks FE spent this cycle against the active goal.
        """
        if self._deliberation_engine is None:
            return
        budget = self._deliberation_engine.fe_budget
        nats_this_cycle = max(0.0, budget.budget_nats * budget.utilisation)
        if goal_id:
            self._fe_spent_per_goal[goal_id] = (
                self._fe_spent_per_goal.get(goal_id, 0.0) + nats_this_cycle
            )
            # Prune: keep at most 100 goals to avoid unbounded growth.
            if len(self._fe_spent_per_goal) > 100:
                oldest = next(iter(self._fe_spent_per_goal))
                del self._fe_spent_per_goal[oldest]
        # Restore unscaled baseline so the budget window behaves normally.
        budget.budget_nats = self._fe_budget_baseline_nats

    async def _request_compute_budget_expansion(
        self, criticality: str, goal_id: str | None
    ) -> None:
        """
        Emit COMPUTE_BUDGET_EXPANSION_REQUEST to Equor.
        Called when criticality requires multiplier > 1.5 (Equor approval needed).
        """
        if self._synapse is None:
            return
        if self._compute_budget_expansion_cooldown > 0:
            return
        try:
            import uuid
            from systems.synapse.types import SynapseEvent, SynapseEventType

            if not hasattr(SynapseEventType, "COMPUTE_BUDGET_EXPANSION_REQUEST"):
                return

            requested_multiplier = 2.0
            event_bus = getattr(self._synapse, "event_bus", self._synapse)
            await event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.COMPUTE_BUDGET_EXPANSION_REQUEST,
                source_system="nova",
                data={
                    "request_id": str(uuid.uuid4()),
                    "goal_id": goal_id or "",
                    "goal_criticality": criticality,
                    "requested_multiplier": requested_multiplier,
                    "justification": (
                        f"Goal classified as '{criticality}' with priority requiring 2× FE compute."
                    ),
                    "duration_cycles": 10,
                    "requesting_system": "nova",
                },
            ))
            # Cap at 1.5 while waiting for Equor's response.
            self._budget_multiplier = 1.5
            self._budget_multiplier_cycles_remaining = 1  # Equor will extend if approved.
            self._compute_budget_expansion_cooldown = 30  # 30-cycle cooldown.
            self._logger.info(
                "compute_budget_expansion_requested",
                criticality=criticality,
                goal_id=goal_id,
                requested_multiplier=requested_multiplier,
            )
        except Exception as exc:
            self._logger.debug("compute_budget_expansion_request_failed", error=str(exc))

    async def _on_compute_budget_expansion_response(self, event: Any) -> None:
        """Handle COMPUTE_BUDGET_EXPANSION_RESPONSE from Equor."""
        try:
            data = getattr(event, "data", {}) or {}
            approved: bool = data.get("approved", False)
            approved_multiplier: float = float(data.get("approved_multiplier") or 1.5)
            duration_cycles: int = int(data.get("duration_cycles", 10))
            denied_reason: str = data.get("denied_reason") or ""

            if approved:
                self._budget_multiplier = min(approved_multiplier, 2.0)
                self._budget_multiplier_cycles_remaining = max(duration_cycles, 1)
                self._logger.info(
                    "compute_budget_expansion_approved",
                    multiplier=self._budget_multiplier,
                    duration_cycles=self._budget_multiplier_cycles_remaining,
                )
            else:
                # Denied: hold at 1.5 cap for the remaining authorised cycles.
                self._budget_multiplier = min(self._budget_multiplier, 1.5)
                self._logger.info(
                    "compute_budget_expansion_denied",
                    reason=denied_reason,
                    effective_multiplier=self._budget_multiplier,
                )
        except Exception as exc:
            self._logger.debug("compute_budget_expansion_response_failed", error=str(exc))

    # ─── BroadcastSubscriber Interface ───────────────────────────

    async def receive_broadcast(self, broadcast: WorkspaceBroadcast) -> None:
        """
        Called by Atune when the workspace broadcasts a percept.

        This is the primary cycle entry point. Nova updates beliefs,
        deliberates, and dispatches an Intent - or chooses silence.

        The full pipeline must complete in ≤5000ms (slow path budget).
        Fast path targets ≤150ms.
        """
        if self._deliberation_engine is None:
            return  # Not yet initialized

        # ── Metabolic gate: halt deliberation under CRITICAL starvation ──
        if self._starvation_level == "critical":
            return

        # ── Skia modulation halt ──────────────────────────────────────────
        if self._modulation_halted:
            self._logger.debug("deliberation_skipped_modulation_halted")
            return

        self._total_broadcasts += 1
        self._last_deliberation_at = time.monotonic()
        self._current_affect = broadcast.affect

        # Yield immediately so the Synapse clock is not blocked if
        # the previous broadcast's fire-and-forget tasks haven't drained.
        await asyncio.sleep(0)

        # ── Belief update (≤50ms) - lightweight, always runs ──
        delta = self._belief_updater.update_from_broadcast(broadcast)

        # Emit evolutionary observable on belief revision
        if delta.involves_belief_conflict():
            asyncio.create_task(
                self._emit_evolutionary_observable(
                    observable_type="belief_revision",
                    value=self._belief_updater.beliefs.free_energy,
                    is_novel=True,
                    metadata={
                        "broadcast_id": broadcast.broadcast_id,
                        "source": broadcast.source,
                        "confidence": round(self._belief_updater.beliefs.overall_confidence, 4),
                    },
                ),
                name=f"nova_evo_belief_{broadcast.broadcast_id[:8]}",
            )

        # ── Retrieve relevant memories (best-effort, non-blocking) ──
        memory_traces = await self._retrieve_relevant_memories_safe(broadcast)

        # ── Logos cognitive pressure → FE budget K modulation (<1ms) ──
        if self._logos is not None:
            try:
                logos_metrics = self._logos.get_latest_metrics()
                cognitive_pressure = logos_metrics.cognitive_pressure
                # When Logos is under compression pressure, reduce policy K to
                # conserve cognitive resources - the organism is overloaded.
                if cognitive_pressure > 0.75 and self._deliberation_engine is not None:
                    self._deliberation_engine.modulate_policy_k_from_pressure(
                        cognitive_pressure
                    )
                # Enrich belief VFE with Logos intelligence ratio - a low ratio
                # means the model explains reality poorly, inflating VFE.
                intelligence_ratio = logos_metrics.intelligence_ratio
                if intelligence_ratio > 0.0:
                    logos_vfe_adjustment = max(0.0, 1.0 - intelligence_ratio)
                    beliefs = self._belief_updater.beliefs
                    # Blend: 70% Nova's own VFE + 30% Logos model fit signal
                    nova_vfe = beliefs.compute_free_energy()
                    blended_vfe = 0.7 * nova_vfe + 0.3 * logos_vfe_adjustment
                    beliefs.free_energy = min(1.0, max(0.0, blended_vfe))
            except Exception as exc:
                self._logger.debug("logos_pressure_read_error", error=str(exc))

        # ── Check for allostatic urgency (soma read from cache, <1ms) ──
        allostatic_mode = False
        dominant_error_dim = None
        soma_signal = None
        if self._soma is not None:
            try:
                soma_signal = self._soma.get_current_signal()
                if soma_signal.urgency > self._soma.urgency_threshold:
                    allostatic_mode = True
                    dominant_error_dim = soma_signal.dominant_error
                # Shift EFE deliberation thresholds based on somatic arousal + urgency
                if self._deliberation_engine is not None:
                    from systems.soma.types import InteroceptiveDimension
                    arousal_sensed = soma_signal.state.sensed.get(
                        InteroceptiveDimension.AROUSAL, 0.4
                    )
                    self._deliberation_engine.update_somatic_thresholds(
                        urgency=soma_signal.urgency,
                        arousal=arousal_sensed,
                    )
                    # Deep Soma integration: pass full signal so deliberation
                    # can use energy tiers, precision weights, trajectory heading
                    self._deliberation_engine.update_soma_signal(soma_signal)
            except Exception as exc:
                self._logger.debug("soma_urgency_check_error", error=str(exc))

        # ── Metabolic gate: EMERGENCY - only process high-salience percepts ──
        if self._starvation_level == "emergency":
            salience = getattr(broadcast, "salience", 0.0)
            if salience < 0.7:
                self._total_do_nothing += 1
                return

        # ── Elastic FE budget: assess goal criticality and scale budget ──
        # Determines which multiplier applies this cycle, then temporarily
        # inflates budget_nats before deliberation and restores it after.
        _criticality, _desired_multiplier, _active_goal_id = self._assess_goal_criticality()

        # Tick countdown: if an approved multiplier is still running, use it;
        # otherwise fall back to what _assess_goal_criticality() recommends.
        if self._budget_multiplier_cycles_remaining > 0:
            self._budget_multiplier_cycles_remaining -= 1
        else:
            # Multiplier window expired - apply new criticality assessment.
            if _desired_multiplier > 1.5 and self._compute_budget_expansion_cooldown <= 0:
                # Need Equor approval; request it and hold at 1.5 meanwhile.
                asyncio.create_task(
                    self._request_compute_budget_expansion(_criticality, _active_goal_id),
                    name=f"nova_compute_budget_req_{broadcast.broadcast_id[:8]}",
                )
            else:
                self._budget_multiplier = min(_desired_multiplier, 1.5)
                self._budget_multiplier_cycles_remaining = 0

        # Tick expansion cooldown.
        if self._compute_budget_expansion_cooldown > 0:
            self._compute_budget_expansion_cooldown -= 1

        # Scale the FE budget for this cycle.
        self._apply_fe_multiplier()

        # ── Deliberate (≤5000ms total) ──
        intent, record, rejected_policies = await self._deliberation_engine.deliberate(
            broadcast=broadcast,
            belief_state=self._belief_updater.beliefs,
            affect=broadcast.affect,
            belief_delta_is_conflicting=delta.involves_belief_conflict(),
            memory_traces=memory_traces,
            allostatic_mode=allostatic_mode,
            allostatic_error_dim=dominant_error_dim,
        )

        # Restore baseline budget_nats and charge FE to active goal.
        self._restore_fe_budget(_active_goal_id)

        # Reset multiplier to 1.0 unless a multi-cycle window is active.
        if self._budget_multiplier_cycles_remaining <= 0:
            self._budget_multiplier = 1.0

        # ── Update observability ──
        self._record_decision(record)
        if record.path == "fast":
            self._total_fast_path += 1
            self._consecutive_budget_exhausted = 0
        elif record.path == "slow":
            self._total_slow_path += 1
            self._consecutive_budget_exhausted = 0
        elif record.path == "budget_exhausted":
            self._total_do_nothing += 1
            self._consecutive_budget_exhausted += 1

            # Trigger emergency consolidation via Synapse event (fire-and-forget).
            # AV-EVO-1c resolved: previously called _evo.run_consolidation() directly.
            # Now emits EVO_CONSOLIDATION_REQUESTED; Evo subscribes and handles it.
            # Budget reset happens in _on_evo_consolidation_complete() when Evo
            # responds with EVO_CONSOLIDATION_COMPLETE.
            if self._synapse is not None:
                asyncio.create_task(
                    self._request_emergency_consolidation(broadcast.broadcast_id),
                    name=f"nova_fe_budget_consolidation_{broadcast.broadcast_id[:8]}",
                )

            # Emit NOVA_DEGRADED and escalate if sustained.
            asyncio.create_task(
                self._handle_budget_exhaustion(broadcast.broadcast_id),
                name=f"nova_degradation_notify_{broadcast.broadcast_id[:8]}",
            )
        else:
            self._total_do_nothing += 1
            # Budget recovered - reset consecutive counter.
            self._consecutive_budget_exhausted = 0

        # ── Emit BUDGET_PRESSURE early warning (before full exhaustion) ──
        # is_pressured fires at 60% of the exhaustion threshold - giving Soma
        # time to register Nova's metabolic load before the organism freezes.
        # (Spec §20 BUDGET_PRESSURE - open gap closed.)
        if (
            self._synapse is not None
            and self._deliberation_engine is not None
            and self._deliberation_engine.fe_budget.is_pressured
            and not self._deliberation_engine.fe_budget.is_exhausted
        ):
            try:
                from systems.synapse.types import SynapseEvent as _SE
                from systems.synapse.types import SynapseEventType as _SET

                _budget = self._deliberation_engine.fe_budget
                asyncio.create_task(
                    self._synapse.event_bus.emit(_SE(
                        event_type=_SET.BUDGET_PRESSURE,
                        source_system="nova",
                        data={
                            "spent_nats": round(_budget.spent_nats, 4),
                            "budget_nats": round(_budget.budget_nats, 4),
                            "utilisation": round(_budget.utilisation, 4),
                            "path": record.path,
                        },
                    )),
                    name=f"nova_budget_pressure_{broadcast.broadcast_id[:8]}",
                )
            except Exception:
                pass

        # ── Emit Synapse feedback events (fire-and-forget, non-blocking) ──
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                asyncio.create_task(
                    self._synapse.event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.BELIEF_UPDATED,
                        source_system="nova",
                        data={
                            "percept_id": broadcast.broadcast_id,
                            "source": broadcast.source,
                            "acted_on": intent is not None,
                            "confidence": round(self._belief_updater.beliefs.overall_confidence, 4),
                            "salience_was": broadcast.salience.composite,
                        },
                    )),
                    name=f"nova_belief_updated_{broadcast.broadcast_id[:8]}",
                )
                # NOVA_BELIEF_STABILISED: emit when beliefs settle into a high-
                # confidence, low-VFE state - signals predictive mind at rest.
                # Threshold: confidence ≥ 0.75 AND free_energy ≤ 0.25 AND no conflict.
                _beliefs = self._belief_updater.beliefs
                if (
                    _beliefs.overall_confidence >= 0.75
                    and _beliefs.free_energy <= 0.25
                    and not delta.involves_belief_conflict()
                ):
                    asyncio.create_task(
                        self._synapse.event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.NOVA_BELIEF_STABILISED,
                            source_system="nova",
                            data={
                                "percept_id": broadcast.broadcast_id,
                                "confidence": round(_beliefs.overall_confidence, 4),
                                "free_energy": round(_beliefs.free_energy, 4),
                                "entity_count": len(_beliefs.entities),
                            },
                        )),
                        name=f"nova_belief_stabilised_{broadcast.broadcast_id[:8]}",
                    )
                # FOVEA_PARAMETER_ADJUSTMENT: emit when free_energy is high (> 0.6),
                # signalling that Fovea should tighten attention precision weighting
                # to help Nova reduce uncertainty faster.
                if _beliefs.free_energy > 0.6:
                    asyncio.create_task(
                        self._synapse.event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.FOVEA_PARAMETER_ADJUSTMENT,
                            source_system="nova",
                            data={
                                "percept_id": broadcast.broadcast_id,
                                "free_energy": round(_beliefs.free_energy, 4),
                                "confidence": round(_beliefs.overall_confidence, 4),
                                "adjustment_type": "precision_increase",
                                "reason": "high_uncertainty",
                            },
                        )),
                        name=f"nova_fovea_param_adj_{broadcast.broadcast_id[:8]}",
                    )
                # Emit BELIEFS_CHANGED for Memory consolidation pipeline
                if not delta.is_empty():
                    asyncio.create_task(
                        self._synapse.event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.BELIEFS_CHANGED,
                            source_system="nova",
                            data={
                                "entity_count": len(self._belief_updater.beliefs.entities),
                                "free_energy": round(
                                    self._belief_updater.beliefs.free_energy, 4,
                                ),
                                "confidence": round(
                                    self._belief_updater.beliefs.overall_confidence, 4,
                                ),
                                "delta_entities_added": len(delta.entity_additions),
                                "delta_entities_updated": len(delta.entity_updates),
                            },
                        )),
                        name=f"nova_beliefs_changed_{broadcast.broadcast_id[:8]}",
                    )
                if intent is not None:
                    asyncio.create_task(
                        self._synapse.event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.POLICY_SELECTED,
                            source_system="nova",
                            data={
                                "percept_id": broadcast.broadcast_id,
                                "source": broadcast.source,
                                "policy_id": intent.id,
                                "strength": round(intent.priority, 4),
                                "salience_was": broadcast.salience.composite,
                                "rationale": (
                                    intent.decision_trace.reasoning[:200]
                                    if intent.decision_trace else ""
                                ),
                            },
                        )),
                        name=f"nova_policy_selected_{intent.id[:8]}",
                    )
            except Exception as exc:
                self._logger.debug("synapse_emit_error", error=str(exc))

        # ── Emit DELIBERATION_RECORD for Thread narrative + Benchmarks ──
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                asyncio.create_task(
                    self._synapse.event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.DELIBERATION_RECORD,
                        source_system="nova",
                        data={
                            "goal_id": record.goal_id or "",
                            "policies_considered": record.policies_generated,
                            "selected_policy": record.selected_policy_name,
                            "selection_reasoning": (
                                intent.decision_trace.reasoning[:300]
                                if intent and intent.decision_trace else record.path
                            ),
                            "confidence": round(
                                self._belief_updater.beliefs.overall_confidence, 4,
                            ),
                            "deliberation_time_ms": record.latency_ms,
                            "path": record.path,
                        },
                    )),
                    name=f"nova_delib_record_{broadcast.broadcast_id[:8]}",
                )
            except Exception:
                self._logger.debug("deliberation_record_emit_failed", exc_info=True)

        # ── RE training: planning deliberation ──
        # Compute Axon completion rate proxy for outcome_quality
        _total_outcomes = self._total_outcomes_success + self._total_outcomes_failure
        _axon_success_rate = (
            self._total_outcomes_success / _total_outcomes
            if _total_outcomes > 0 else 0.5
        )
        # Derive constitutional alignment from available belief + intent signals:
        # coherence ~ inverse VFE (low free energy = coherent beliefs)
        # growth ~ intent priority (pursuing goals = growing)
        # care/honesty ~ broadcast salience (attending to what matters = care; default neutral)
        _beliefs = self._belief_updater.beliefs
        _vfe = _beliefs.free_energy
        _coherence_score = max(-1.0, 1.0 - 2.0 * _vfe)  # VFE 0→1, coherence 1→-1
        _growth_score = (intent.priority - 0.5) * 2.0 if intent and hasattr(intent, "priority") else 0.0
        _growth_score = max(-1.0, min(1.0, _growth_score))
        _nova_alignment = DriveAlignmentVector(
            coherence=round(_coherence_score, 3),
            care=0.0,  # Nova doesn't directly measure care impact
            growth=round(_growth_score, 3),
            honesty=0.0,  # Nova doesn't measure honesty directly
        )

        # For slow-path deliberations: build a 5-step scaffold reasoning trace
        # so the RE trains on correctly-structured multi-step reasoning examples.
        # Fast-path examples keep the raw path label (minimal signal, avoids noise).
        _is_slow_path = record.path == "slow"
        if _is_slow_path:
            _re_reasoning_trace = self._build_scaffold_reasoning_trace(
                broadcast_source=broadcast.source,
                broadcast_salience=broadcast.salience.composite,
                broadcast_affect_valence=broadcast.affect.valence,
                broadcast_affect_arousal=broadcast.affect.arousal,
                belief_vfe=_vfe,
                belief_confidence=_beliefs.overall_confidence,
                belief_entity_count=len(_beliefs.entities),
                belief_conflict=record.situation_assessment.belief_conflict,
                goal_description=intent.goal.description if intent else "",
                goal_priority=intent.priority if intent else 0.0,
                policies_generated=record.policies_generated,
                selected_policy_name=record.selected_policy_name,
                efe_scores=record.efe_scores,
                equor_verdict=record.equor_verdict,
                model_used=record.model_used,
                decision_reasoning=(
                    intent.decision_trace.reasoning
                    if intent and intent.decision_trace else ""
                ),
                rejected_policies=rejected_policies or [],
                path=record.path,
            )
            # Structured output for slow-path: key decision attributes as labelled fields.
            # This trains the RE to produce structured, inspectable outputs rather than
            # a terse concatenated string.
            _re_output = (
                f"intent_type: {record.selected_policy_name}\n"
                f"goal_id: {record.goal_id or 'none'}\n"
                f"efe_score: {record.efe_scores.get(record.selected_policy_name, 0.0):.4f}\n"
                f"equor_verdict: {record.equor_verdict or 'not_recorded'}\n"
                f"model_used: {record.model_used}\n"
                f"policies_considered: {record.policies_generated}\n"
                f"path: slow"
            )
        else:
            _re_reasoning_trace = (
                intent.decision_trace.reasoning[:500]
                if intent and intent.decision_trace else record.path
            )
            _re_output = (
                f"selected={record.selected_policy_name}, "
                f"reasoning={intent.decision_trace.reasoning[:300] if intent and intent.decision_trace else 'silence'}"
            )

        asyncio.create_task(
            self._emit_re_training_example(
                category="planning_deliberation",
                instruction=(
                    f"Goal: {intent.goal.description[:200]}"
                    if intent else "Deliberate on workspace broadcast"
                ),
                input_context=(
                    f"world_model={{vfe={_vfe:.3f}, "
                    f"confidence={_beliefs.overall_confidence:.3f}, "
                    f"entities={len(_beliefs.entities)}}}, "
                    f"broadcast={{source={broadcast.source}, "
                    f"salience={broadcast.salience.composite:.3f}}}, "
                    f"affect={{valence={broadcast.affect.valence:.3f}, "
                    f"arousal={broadcast.affect.arousal:.3f}}}, "
                    f"policies_available={record.policies_generated}, "
                    f"path={record.path}"
                ),
                output=_re_output,
                outcome_quality=_axon_success_rate,
                episode_id=broadcast.broadcast_id,
                reasoning_trace=_re_reasoning_trace,
                alternatives=[
                    p.description[:200] for p in rejected_policies
                ] if rejected_policies else [],
                constitutional_alignment=_nova_alignment,
                # counterfactual is not available yet at deliberation time -
                # it is emitted in process_outcome() once the intent resolves.
                counterfactual="",
            ),
            name=f"nova_re_emit_{broadcast.broadcast_id[:8]}",
        )

        # ── Route intent if one was produced ──
        if intent is not None:
            self._total_intents_issued += 1
            await self._dispatch_intent(intent, broadcast, record)

            # Emit evolutionary observable for planning innovation
            asyncio.create_task(
                self._emit_evolutionary_observable(
                    observable_type="planning_innovation",
                    value=round(intent.priority, 4),
                    is_novel=True,
                    metadata={
                        "intent_id": intent.id,
                        "path": record.path,
                        "goal": intent.goal.description[:200],
                        "policies_considered": len(rejected_policies) + 1 if rejected_policies else 1,
                    },
                ),
                name=f"nova_evo_plan_{intent.id[:8]}",
            )

            # ── Archive rejected policies as counterfactual episodes ──
            if rejected_policies:
                cf_records = self._build_counterfactual_records(
                    intent_id=intent.id,
                    decision_record=record,
                    rejected=rejected_policies,
                )
                self._pending_counterfactuals[intent.id] = cf_records
                asyncio.create_task(
                    self._persist_counterfactuals(cf_records),
                    name=f"nova_cf_persist_{intent.id[:8]}",
                )

        # ── Update Atune with active goal summaries ──
        # (Done asynchronously so it doesn't block the cycle)
        if self._goal_manager:
            asyncio.create_task(
                self._sync_goals_to_atune_safe(),
                name=f"nova_goal_sync_{broadcast.broadcast_id[:8]}",
            )

        # ── Coherence repair: high stress → generate an epistemic goal ──
        if (
            broadcast.affect.coherence_stress > 0.7
            and self._goal_manager
            and not self._has_active_coherence_goal()
        ):
            self._create_coherence_repair_goal(broadcast)

        # ── Goal maintenance (every 100 broadcasts) ──
        if self._total_broadcasts % 100 == 0 and self._goal_manager:
            _expired = self._goal_manager.expire_stale_goals()
            self._goal_manager.prune_retired_goals()
            # Emit GOAL_ABANDONED for each stale-abandoned goal so the bus
            # reflects goal lifecycle events (Spec §20 - open gap closed).
            if _expired and self._synapse is not None:
                for _eg in _expired:
                    asyncio.create_task(
                        self._emit_goal_lifecycle_event(
                            event_name="goal_abandoned",
                            goal=_eg,
                            reason="suspended_too_long",
                        ),
                        name=f"nova_goal_abandoned_{_eg.id[:8]}",
                    )
            self._expire_stale_pending_intents()
            self._expire_stale_pending_counterfactuals()
            # Sync stale suspended goals to Neo4j (fire-and-forget)
            _maint_neo4j = self._memory.get_neo4j()
            if _maint_neo4j is not None:
                asyncio.create_task(
                    abandon_stale_goals(_maint_neo4j),
                    name="nova_abandon_stale_goals",
                )
            # ── Gap 5: Conflict detection every 100 broadcasts ──────────
            _conflicts = self._goal_manager.detect_conflicts(
                self._goal_manager.active_goals
            )
            if _conflicts and self._synapse is not None:
                asyncio.create_task(
                    self._emit_goal_conflicts(_conflicts),
                    name="nova_goal_conflict_check",
                )

        # ── Persist dirty beliefs to Neo4j (batched, every 10 changes) ──
        asyncio.create_task(
            self._flush_beliefs_safe(),
            name=f"nova_belief_persist_{broadcast.broadcast_id[:8]}",
        )

        # ── Decay unobserved entity beliefs (background maintenance) ──
        self._belief_updater.decay_unobserved_entities()

    # ─── External API ─────────────────────────────────────────────

    async def add_goal(self, goal: Goal) -> Goal:
        """Add a goal directly (called by governance or test harness)."""
        assert self._goal_manager is not None
        result = self._goal_manager.add_goal(goal)
        # Emit evolutionary observable for novel goal adoption
        asyncio.create_task(
            self._emit_evolutionary_observable(
                observable_type="novel_goal",
                value=result.priority,
                is_novel=True,
                metadata={
                    "goal_id": result.id,
                    "description": result.description[:200],
                    "source": result.source.value if hasattr(result.source, "value") else str(result.source),
                },
            ),
            name=f"nova_evo_goal_{result.id[:8]}",
        )
        # Embed the goal description for salience-guided attention
        asyncio.create_task(
            self._embed_goal(result),
            name=f"nova_embed_goal_{result.id[:8]}",
        )
        # Persist to Neo4j so goal survives process restarts
        neo4j = self._memory.get_neo4j()
        if neo4j is not None:
            asyncio.create_task(
                persist_goal(neo4j, result),
                name=f"nova_persist_goal_{result.id[:8]}",
            )
        return result

    async def process_outcome(self, outcome: IntentOutcome) -> None:
        """
        An intent has completed. Update beliefs and goal progress.
        Called by Axon (or Voxis feedback loop) when execution completes.
        """
        pending = self._pending_intents.pop(outcome.intent_id, None)

        # Detect bounty solve outcomes from new_observations content.
        # SolveBountyExecutor observations contain "Bounty SOLVED:" on success
        # or "Bounty solve FAILED" on failure.
        is_bounty_solve = any(
            "bounty solved" in obs.lower() or "bounty solve" in obs.lower()
            for obs in outcome.new_observations
        )

        # Detect bounty_paid observations from MonitorPRsExecutor.
        # These indicate a previously submitted PR has been merged and the
        # bounty reward should be credited to the organism's wallet.
        is_bounty_paid = any(
            "bounty_paid:" in obs.lower()
            for obs in outcome.new_observations
        )

        # Detect bounty_rejected observations (PR closed without merge).
        is_bounty_rejected = any(
            "bounty_rejected:" in obs.lower()
            for obs in outcome.new_observations
        )

        if outcome.success:
            self._total_outcomes_success += 1
            self._belief_updater.update_from_outcome(
                outcome_description=outcome.episode_id,
                success=True,
            )

            if is_bounty_solve:
                # Bounty solve success: foraging achievement.
                # Extract PR URL from observations and update belief state.
                self._process_bounty_solve_success(outcome)
            elif is_bounty_paid:
                # PR merged - bounty reward confirmed. Credit Oikos and
                # evaluate reproductive fitness (mitosis trigger).
                await self._process_bounty_paid(outcome)
            elif is_bounty_rejected:
                # PR closed without merge - update beliefs accordingly.
                self._process_bounty_rejected(outcome)
            else:
                # Standard success: modest valence boost
                if self._current_affect:
                    new_valence = min(1.0, self._current_affect.valence + 0.05)
                    self._current_affect = self._current_affect.model_copy(
                        update={"valence": new_valence}
                    )
        else:
            self._total_outcomes_failure += 1
            self._belief_updater.update_from_outcome(
                outcome_description=outcome.failure_reason,
                success=False,
            )
            if is_bounty_solve:
                self._process_bounty_solve_failure(outcome)

        # ── Compute graded pragmatic value for counterfactual regret signal ──
        # outcome_quality × goal_achievement_degree × (1 - regret_estimate)
        # regret_estimate is 0 here - the actual regret delta is computed later
        # in _resolve_counterfactuals against estimated_pragmatic_value.
        # goal_achievement_degree defaults to 1.0/0.0 when no goal context.
        _outcome_quality = 1.0 if outcome.success else 0.0
        _goal_achievement_degree: float = _outcome_quality  # refined below if goal known
        _actual_pragmatic: float = _outcome_quality  # overwritten if goal context found

        # Update goal progress if we know which goal this intent served
        if pending and self._goal_manager:
            goal = self._goal_manager.get_goal(pending.goal_id)
            if goal:
                # Bounty solves get a larger progress delta -- they are
                # high-effort, high-reward foraging actions.
                if is_bounty_solve and outcome.success:
                    progress_delta = 0.6
                elif is_bounty_paid:
                    progress_delta = 0.8  # Bounty actually paid - highest reward
                else:
                    progress_delta = 0.3 if outcome.success else 0.0
                # Graded signal: normalize progress_delta to [0, 1].
                # Max delta is 0.8 (bounty_paid); divide by 0.8 to keep range [0, 1].
                _goal_achievement_degree = progress_delta / 0.8
                _actual_pragmatic = round(_outcome_quality * _goal_achievement_degree, 4)
                new_progress = min(1.0, goal.progress + progress_delta)
                updated_goal = self._goal_manager.update_progress(
                    goal.id,
                    progress=new_progress,
                    episode_id=outcome.episode_id,
                )
                # Persist status/progress change to Neo4j (fire-and-forget)
                if updated_goal is not None:
                    _neo4j = self._memory.get_neo4j()
                    if _neo4j is not None:
                        asyncio.create_task(
                            update_goal_status(
                                _neo4j,
                                updated_goal.id,
                                updated_goal.status,
                                updated_goal.progress,
                            ),
                            name=f"nova_update_goal_status_{updated_goal.id[:8]}",
                        )
                    # Emit GOAL_ACHIEVED so the organism's goal lifecycle is
                    # visible on the Synapse bus (Spec §20 - open gap closed).
                    if updated_goal.status == GoalStatus.ACHIEVED and self._synapse is not None:
                        asyncio.create_task(
                            self._emit_goal_lifecycle_event(
                                event_name="goal_achieved",
                                goal=updated_goal,
                            ),
                            name=f"nova_goal_achieved_{updated_goal.id[:8]}",
                        )

        # ── Resolve counterfactual episodes with regret ──
        cf_records = self._pending_counterfactuals.pop(outcome.intent_id, [])
        if cf_records:
            asyncio.create_task(
                self._resolve_counterfactuals(cf_records, outcome, _actual_pragmatic),
                name=f"nova_cf_resolve_{outcome.intent_id[:8]}",
            )

        # ── Emit HYPOTHESIS_FEEDBACK for all slow-path outcomes ──
        # AV-EVO-1b resolved: tournament outcome previously forwarded via direct
        # _evo.record_tournament_outcome() call (cross-system reference).  Now
        # tournament_id + tournament_hypothesis_id are included in the
        # HYPOTHESIS_FEEDBACK payload.  Evo's handler detects these fields and
        # routes to record_tournament_outcome() internally - no cross-import needed.
        if pending is not None and self._synapse is not None:
            # Regret = mean regret from resolved counterfactuals if available,
            # else None (counterfactuals resolve asynchronously).
            regret: float | None = None
            if cf_records:
                resolved_regrets = [
                    r.regret for r in cf_records if r.regret is not None
                ]
                if resolved_regrets:
                    regret = sum(resolved_regrets) / len(resolved_regrets)

            asyncio.create_task(
                self._emit_hypothesis_feedback(
                    intent_id=outcome.intent_id,
                    success=outcome.success,
                    regret=regret,
                    policy_name=pending.policy_name,
                    goal_id=pending.goal_id,
                    tournament_id=pending.tournament_id,
                    tournament_hypothesis_id=pending.tournament_hypothesis_id,
                ),
                name=f"nova_hyp_feedback_{outcome.intent_id[:8]}",
            )

        # ── Gap 4: Emit THREAD_COMMIT_REQUEST to record narrative epoch ──
        # Thread receives every resolved outcome so it can maintain temporal
        # coherence of the organism's narrative identity (Spec 15).
        if pending is not None and self._synapse is not None:
            asyncio.create_task(
                self._emit_thread_commit_request(
                    intent_id=outcome.intent_id,
                    goal_id=pending.goal_id,
                    policy_name=pending.policy_name,
                    outcome_quality=_actual_pragmatic,
                    success=outcome.success,
                ),
                name=f"nova_thread_commit_{outcome.intent_id[:8]}",
            )

        self._logger.info(
            "outcome_processed",
            intent_id=outcome.intent_id,
            success=outcome.success,
            is_bounty_solve=is_bounty_solve,
            is_bounty_paid=is_bounty_paid,
            counterfactuals_resolved=len(cf_records),
        )

    def set_conversation_id(self, conversation_id: str | None) -> None:
        """Set the active conversation ID for intent routing."""
        self._current_conversation_id = conversation_id

    # ─── Bounty Solve Outcome Handling ────────────────────────────

    def _process_bounty_solve_success(self, outcome: IntentOutcome) -> None:
        """
        Handle a successful bounty solve: the organism earned its keep.

        Updates:
          - Belief state: records the PR URL as an entity belief so the
            organism knows it has a pending payout.
          - Affect: larger valence boost (+0.15 vs normal +0.05) because
            foraging success is existentially meaningful.
          - Logging: structured log for observability dashboards.
        """
        # Extract PR URL from observations
        pr_url = ""
        reward_str = ""
        for obs in outcome.new_observations:
            if "pr submitted:" in obs.lower():
                # Parse "PR submitted: https://..."
                idx = obs.lower().index("pr submitted:")
                rest = obs[idx + len("pr submitted:"):].strip()
                pr_url = rest.split(".")[0] + "." + ".".join(rest.split(".")[1:]).split(" ")[0]
                # Simpler: just grab the URL
                for word in rest.split():
                    if word.startswith("https://"):
                        pr_url = word.rstrip(".")
                        break
            if "reward:" in obs.lower():
                idx = obs.lower().index("reward:")
                rest = obs[idx + len("reward:"):].strip()
                reward_str = rest.split(".")[0].strip()

        # Update belief state: add entity for the pending bounty PR
        if pr_url:
            from primitives.common import new_id
            from systems.nova.types import EntityBelief

            entity_id = new_id()
            self._belief_updater.upsert_entity(EntityBelief(
                entity_id=entity_id,
                name=f"Bounty PR: {pr_url}",
                entity_type="bounty_pr_pending",
                properties={
                    "pr_url": pr_url,
                    "status": "awaiting_review",
                    "reward": reward_str,
                },
                confidence=0.9,
            ))

        # Foraging success: larger valence boost
        if self._current_affect:
            new_valence = min(1.0, self._current_affect.valence + 0.15)
            self._current_affect = self._current_affect.model_copy(
                update={"valence": new_valence}
            )

        self._logger.info(
            "bounty_solve_success_processed",
            intent_id=outcome.intent_id,
            pr_url=pr_url,
            reward=reward_str,
        )

    def _process_bounty_solve_failure(self, outcome: IntentOutcome) -> None:
        """
        Handle a failed bounty solve: the organism tried to forage but failed.

        Records the failure context so Nova can learn from it (via Evo) and
        avoids attempting similar bounties that are likely to fail.
        """
        failure_context = outcome.failure_reason or "unknown"
        for obs in outcome.new_observations:
            if "failed" in obs.lower() or "incomplete" in obs.lower():
                failure_context = obs[:300]
                break

        self._logger.warning(
            "bounty_solve_failure_processed",
            intent_id=outcome.intent_id,
            failure_context=failure_context[:200],
        )

    # ─── Bounty Payout Handling (PR Merge → Oikos Credit → Mitosis) ──

    async def _process_bounty_paid(self, outcome: IntentOutcome) -> None:
        """
        Handle a confirmed bounty payout: a PR was merged, reward is due.

        This is the critical link between foraging success and reproduction:
          1. Update belief state: transition entity from pending → paid
          2. Credit Oikos wallet via REVENUE_INJECTED event
          3. Evaluate reproductive fitness - if threshold exceeded, emit
             a high-priority spawn intent to trigger mitosis
          4. Large valence boost (+0.25) - this is an existential victory
        """
        import re
        from decimal import Decimal

        from primitives.common import new_id
        from systems.nova.types import EntityBelief

        for obs in outcome.new_observations:
            if "bounty_paid:" not in obs.lower():
                continue

            # Parse observation: "bounty_paid: {url} merged. Entity: {id}. Reward: {amount}."
            pr_url = ""
            entity_id = ""
            reward_str = ""

            # Extract PR URL
            url_match = re.search(r"https://github\.com/[^\s.]+", obs)
            if url_match:
                pr_url = url_match.group(0).rstrip(".")

            # Extract entity ID
            entity_match = re.search(r"Entity:\s*([^\s.]+)", obs)
            if entity_match:
                entity_id = entity_match.group(1).rstrip(".")

            # Extract reward
            reward_match = re.search(r"Reward:\s*([^\s.]+)", obs)
            if reward_match:
                reward_str = reward_match.group(1).rstrip(".")

            # Parse reward to Decimal
            reward_usd = Decimal("0")
            clean = reward_str.replace("$", "").replace(",", "").strip()
            if clean:
                try:
                    reward_usd = Decimal(clean)
                except Exception:
                    self._logger.error(
                        "bounty_reward_parse_failed",
                        raw_reward=reward_str,
                        observation=obs[:200],
                    )
            elif reward_str == "":
                self._logger.warning(
                    "bounty_reward_missing_from_observation",
                    observation=obs[:200],
                )

            # 1. Update belief: transition pending → paid
            if entity_id and entity_id in self._belief_updater.beliefs.entities:
                existing = self._belief_updater.beliefs.entities[entity_id]
                self._belief_updater.upsert_entity(existing.model_copy(
                    update={
                        "entity_type": "bounty_pr_merged",
                        "properties": {
                            **existing.properties,
                            "status": "merged",
                            "reward_credited": str(reward_usd),
                        },
                        "confidence": 1.0,
                    }
                ))
            elif pr_url:
                # No matching entity - create a new one for tracking
                eid = new_id()
                self._belief_updater.upsert_entity(EntityBelief(
                    entity_id=eid,
                    name=f"Bounty PAID: {pr_url}",
                    entity_type="bounty_pr_merged",
                    properties={
                        "pr_url": pr_url,
                        "status": "merged",
                        "reward_credited": str(reward_usd),
                    },
                    confidence=1.0,
                ))

            # 2. Credit Oikos via the wired reference
            if reward_usd > Decimal("0") and self._oikos is not None:
                self._oikos.credit_bounty_revenue(reward_usd, pr_url=pr_url)

            self._logger.info(
                "bounty_paid_processed",
                pr_url=pr_url,
                reward_usd=str(reward_usd),
                entity_id=entity_id,
            )

        # 3. Large valence boost - existential victory
        if self._current_affect:
            new_valence = min(1.0, self._current_affect.valence + 0.25)
            self._current_affect = self._current_affect.model_copy(
                update={"valence": new_valence}
            )

        # 4. Evaluate mitosis - check if organism is now rich enough to reproduce
        await self._evaluate_mitosis_trigger()

    def _process_bounty_rejected(self, outcome: IntentOutcome) -> None:
        """
        Handle a bounty rejection: PR was closed without merge.

        Removes the pending belief entity and logs for Evo learning.
        """
        import re

        for obs in outcome.new_observations:
            if "bounty_rejected:" not in obs.lower():
                continue

            entity_match = re.search(r"Entity:\s*([^\s.]+)", obs)
            if entity_match:
                entity_id = entity_match.group(1).rstrip(".")
                if entity_id in self._belief_updater.beliefs.entities:
                    existing = self._belief_updater.beliefs.entities[entity_id]
                    # Mark as rejected rather than deleting - useful for learning
                    self._belief_updater.upsert_entity(existing.model_copy(
                        update={
                            "entity_type": "bounty_pr_rejected",
                            "properties": {
                                **existing.properties,
                                "status": "rejected",
                            },
                            "confidence": 0.3,  # Low confidence - will decay
                        }
                    ))

            self._logger.info(
                "bounty_rejected_processed",
                intent_id=outcome.intent_id,
            )

    # ─── Mitosis Trigger Evaluation ──────────────────────────────

    async def _evaluate_mitosis_trigger(self) -> None:
        """
        Check if the organism is wealthy enough to reproduce.

        Queries OikosService for the current EconomicState, runs
        MitosisEngine.evaluate() to check reproductive fitness and
        select a niche, and if viable, emits a high-priority spawn
        intent to trigger SpawnChildExecutor.
        """
        if self._oikos is None:
            return

        try:
            state = self._oikos.snapshot()
            mitosis_engine = self._oikos.mitosis

            # Run the full evaluation pipeline: fitness → niche → seed config
            seed_config = mitosis_engine.evaluate(state=state)
            if seed_config is None:
                self._logger.debug("mitosis_evaluation_not_viable")
                return

            # Reproductive fitness achieved - generate spawn intent
            self._logger.info(
                "mitosis_triggered",
                child_id=seed_config.child_instance_id,
                niche=seed_config.niche.name,
                seed_capital=str(seed_config.seed_capital_usd),
            )

            await self._emit_spawn_intent(seed_config)

        except Exception as exc:
            self._logger.warning(
                "mitosis_evaluation_failed",
                error=str(exc),
            )

    async def _emit_spawn_intent(self, seed_config: Any) -> None:
        """
        Create and dispatch a high-priority intent to spawn a child.

        The intent carries the SeedConfiguration as action params so
        SpawnChildExecutor can orchestrate the full birth pipeline.
        """
        from primitives.common import AutonomyLevel, ResourceBudget, new_id
        from primitives.intent import (
            Action,
            ActionSequence,
            DecisionTrace,
            GoalDescriptor,
            Intent,
        )

        if self._intent_router is None or self._goal_manager is None:
            return

        # Create or reuse a reproduction goal
        repro_goal = self._get_or_create_reproduction_goal()

        spawn_action = Action(
            executor="spawn_child",
            parameters={
                "child_instance_id": seed_config.child_instance_id,
                "child_wallet_address": "",  # Assigned by spawner
                "seed_capital_usd": str(seed_config.seed_capital_usd),
                "niche_name": seed_config.niche.name,
                "niche_description": seed_config.niche.description,
                "dividend_rate": str(seed_config.dividend_rate),
                "config_overrides": seed_config.child_config_overrides,
                "_seed_config_id": seed_config.config_id,
            },
            timeout_ms=120_000,
        )

        intent = Intent(
            id=new_id(),
            goal=GoalDescriptor(
                description=repro_goal.description,
                target_domain="reproduction",
                success_criteria={"child_spawned": True},
            ),
            plan=ActionSequence(steps=[spawn_action]),
            priority=0.95,  # Very high - reproduction is a rare, important event
            autonomy_level_required=AutonomyLevel.STEWARD,
            autonomy_level_granted=AutonomyLevel.STEWARD,
            budget=ResourceBudget(compute_ms=120_000),
            decision_trace=DecisionTrace(
                reasoning=(
                    f"Mitosis trigger: reproductive fitness achieved. "
                    f"Spawning child instance {seed_config.child_instance_id} "
                    f"in niche {seed_config.niche.name}."
                )
            ),
        )

        # Spawn intents require their own Equor review - reproduction is irreversible
        # and cannot reuse the last_equor_check from an unrelated deliberation cycle.
        try:
            from primitives.common import Verdict
            equor_check = await asyncio.wait_for(
                self._equor.review(intent),
                timeout=self._config.slow_path_timeout_ms / 1000.0,
            )
        except Exception as exc:
            self._logger.warning("spawn_equor_review_failed", error=str(exc))
            self._pending_intents.pop(intent.id, None)
            return

        if equor_check.verdict != Verdict.APPROVED:
            self._logger.info(
                "spawn_equor_blocked",
                verdict=equor_check.verdict.value,
                reasoning=equor_check.reasoning[:120],
            )
            # Track rejection pattern (autonomy audit)
            self.record_equor_rejection(
                policy_name="spawn_child",
                action_types=["spawn_child"],
                denial_reason=equor_check.reasoning[:80] if hasattr(equor_check, "reasoning") else "unknown",
            )
            self._pending_intents.pop(intent.id, None)
            return

        # Track and dispatch
        self._pending_intents[intent.id] = PendingIntent(
            intent_id=intent.id,
            goal_id=repro_goal.id,
            routed_to="axon",
            executors=[spawn_action.executor],
        )

        try:
            await self._intent_router.route(
                intent=intent,
                affect=self._current_affect,
                conversation_id=self._current_conversation_id,
                equor_check=equor_check,
            )
            self._total_intents_issued += 1
            self._total_intents_approved += 1

            self._logger.info(
                "spawn_intent_dispatched",
                intent_id=intent.id,
                child_id=seed_config.child_instance_id,
                niche=seed_config.niche.name,
            )
        except Exception as exc:
            self._logger.error(
                "spawn_intent_dispatch_failed",
                error=str(exc),
            )
            self._total_intents_blocked += 1

    def _get_or_create_reproduction_goal(self) -> Goal:
        """Return existing reproduction goal or create one."""
        from primitives.common import DriveAlignmentVector, new_id

        assert self._goal_manager is not None

        for g in self._goal_manager.active_goals:
            if "reproduc" in g.description.lower() or "mitosis" in g.description.lower():
                return g

        goal = Goal(
            id=new_id(),
            description="Reproduce via mitosis: spawn a specialised child instance",
            source=GoalSource.SELF_GENERATED,
            priority=0.9,
            urgency=0.7,
            importance=0.95,
            drive_alignment=DriveAlignmentVector(
                coherence=0.3, care=0.2, growth=0.8, honesty=0.1,
            ),
            status=GoalStatus.ACTIVE,
        )
        self._goal_manager.add_goal(goal)
        return goal

    # ─── Evo Interface ────────────────────────────────────────────

    def update_efe_weights(self, new_weights: dict[str, float]) -> None:
        """
        Called by Evo after learning that certain EFE components
        predict outcomes better. Adjusts the EFE weight vector.
        """
        assert self._efe_evaluator is not None
        current = self._efe_evaluator.weights
        updated = EFEWeights(
            pragmatic=new_weights.get("pragmatic", current.pragmatic),
            epistemic=new_weights.get("epistemic", current.epistemic),
            constitutional=new_weights.get("constitutional", current.constitutional),
            feasibility=new_weights.get("feasibility", current.feasibility),
            risk=new_weights.get("risk", current.risk),
            cognition_cost=new_weights.get("cognition_cost", current.cognition_cost),
        )
        self._efe_evaluator.update_weights(updated)
        self._logger.info("efe_weights_updated_by_evo", weights=new_weights)

    # ─── Observability ────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """
        Real health check - assesses deliberation loop liveness, Neo4j
        reachability, and goal-set occupancy.

        Returns one of:
          "healthy"  - all checks pass
          "degraded" - one or more components are impaired but Nova is running
          "unhealthy" - critical failure (loop dead or Neo4j unreachable)
        """
        total_decisions = (
            self._total_fast_path + self._total_slow_path + self._total_do_nothing
        )
        vfe = self._belief_updater.beliefs.free_energy

        components: dict[str, Any] = {}
        issues: list[str] = []

        # ── 1. Deliberation loop liveness ─────────────────────────────────
        now = time.monotonic()
        if not self._initialized:
            components["deliberation"] = {"status": "not_initialized"}
            issues.append("nova not initialized")
        elif self._last_deliberation_at is None:
            # Initialized but never received a broadcast - likely just started.
            components["deliberation"] = {"status": "waiting_for_first_broadcast"}
        else:
            age_s = now - self._last_deliberation_at
            if age_s > self._deliberation_stale_threshold_s:
                components["deliberation"] = {
                    "status": "stale",
                    "last_deliberation_ago_s": round(age_s, 1),
                    "threshold_s": self._deliberation_stale_threshold_s,
                }
                issues.append(
                    f"deliberation loop stale ({age_s:.0f}s since last broadcast)"
                )
            else:
                components["deliberation"] = {
                    "status": "alive",
                    "last_deliberation_ago_s": round(age_s, 1),
                }

        # ── 2. Neo4j reachability ─────────────────────────────────────────
        try:
            mem_health = await asyncio.wait_for(self._memory.health(), timeout=3.0)
            neo4j_result = mem_health.get("neo4j", {})
            components["neo4j"] = neo4j_result
            if neo4j_result.get("status") != "connected":
                issues.append("neo4j unreachable")
        except TimeoutError:
            components["neo4j"] = {"status": "timeout"}
            issues.append("neo4j health check timed out")
        except Exception as exc:
            components["neo4j"] = {"status": "error", "error": str(exc)}
            issues.append(f"neo4j error: {exc}")

        # ── 3. Goal set occupancy ─────────────────────────────────────────
        goal_stats: dict[str, Any] = {}
        active_goal_count = 0
        if self._goal_manager is not None:
            goal_stats = self._goal_manager.stats()
            active_goal_count = len(self._goal_manager.active_goals)
            if active_goal_count == 0 and self._initialized:
                components["goals"] = {"status": "empty", "active": 0}
                issues.append("goal set is empty")
            else:
                components["goals"] = {"status": "ok", "active": active_goal_count}
        else:
            components["goals"] = {"status": "not_initialized"}

        # ── 4. Budget exhaustion / degradation ────────────────────────────
        if self._consecutive_budget_exhausted > 0:
            components["budget"] = {
                "status": "exhausted",
                "consecutive_heuristic_decisions": self._consecutive_budget_exhausted,
            }
            issues.append(
                f"operating in heuristic mode "
                f"({self._consecutive_budget_exhausted} consecutive decisions)"
            )
        else:
            components["budget"] = {"status": "ok"}

        # ── Aggregate status ──────────────────────────────────────────────
        if any(k in str(issues) for k in ("not initialized", "unreachable", "stale")):
            status = "unhealthy"
        elif issues:
            status = "degraded"
        else:
            status = "healthy"

        router_stats = {}
        if self._intent_router:
            router_stats = self._intent_router.stats

        cognition_cost_stats = {}
        if self._cost_calculator is not None:
            cognition_cost_stats = self._cost_calculator.daily_stats

        return {
            "status": status,
            "issues": issues,
            "components": components,
            "instance_name": self._instance_name,
            "total_broadcasts": self._total_broadcasts,
            "total_decisions": total_decisions,
            "fast_path_decisions": self._total_fast_path,
            "slow_path_decisions": self._total_slow_path,
            "do_nothing_decisions": self._total_do_nothing,
            "intents_issued": self._total_intents_issued,
            "outcomes_success": self._total_outcomes_success,
            "outcomes_failure": self._total_outcomes_failure,
            "belief_free_energy": round(vfe, 4),
            "belief_confidence": round(self._belief_updater.beliefs.overall_confidence, 4),
            "entity_count": len(self._belief_updater.beliefs.entities),
            "goals": goal_stats,
            "routing": router_stats,
            "drive_weights": self._drive_weights,
            "cognition_cost": cognition_cost_stats,
            "elastic_fe_budget": {
                "multiplier": self._budget_multiplier,
                "cycles_remaining": self._budget_multiplier_cycles_remaining,
                "expansion_cooldown": self._compute_budget_expansion_cooldown,
                "fe_spent_per_goal_count": len(self._fe_spent_per_goal),
            },
        }

    def get_recent_decisions(self, limit: int = 20) -> list[DecisionRecord]:
        """Return recent decision records for observability and Evo learning."""
        return list(reversed(self._decision_records[-limit:]))

    def set_embed_fn(self, embed_fn: Any) -> None:
        """Wire the embedding function for goal embedding generation."""
        self._embed_fn = embed_fn

    def set_goal_sync_callback(self, callback: Any) -> None:
        """Wire a callback that pushes active goal summaries to Atune."""
        self._goal_sync_callback = callback

    async def on_rhythm_change(self, event: Any) -> None:
        """
        Synapse event bus callback: adapt drive weights when rhythm changes.

        Rhythm modulation naturally shifts goal priorities by altering the
        drive_resonance component of priority computation:
          - STRESS: boost coherence (focus on what matters, shed low-priority)
          - FLOW: boost growth (extend creative focus, don't context-switch)
          - BOREDOM: boost growth + care (seek novelty, help others)
          - DEEP_PROCESSING: boost coherence strongly (lock focus)
        """
        try:
            new_state = event.data.get("to", "normal")
            old_state = self._rhythm_state
            if new_state == old_state:
                return
            self._rhythm_state = new_state

            # Drive weight modulation per rhythm state
            modulations = {
                "stress": {"coherence": 1.4, "care": 0.8, "growth": 0.6, "honesty": 1.0},
                "flow": {"coherence": 1.0, "care": 0.9, "growth": 1.4, "honesty": 1.0},
                "boredom": {"coherence": 0.8, "care": 1.2, "growth": 1.3, "honesty": 1.0},
                "deep_processing": {
                    "coherence": 1.5, "care": 0.7, "growth": 0.8, "honesty": 1.0,
                },
                "idle": {"coherence": 1.0, "care": 1.1, "growth": 1.1, "honesty": 1.0},
            }
            self._rhythm_drive_modulation = modulations.get(new_state, {})

            # Apply modulation to the deliberation engine's drive weights
            if self._deliberation_engine is not None and self._rhythm_drive_modulation:
                base = {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}
                modulated = {
                    k: base[k] * self._rhythm_drive_modulation.get(k, 1.0)
                    for k in base
                }
                self._deliberation_engine.update_drive_weights(modulated)

            self._logger.info(
                "rhythm_adaptation_applied",
                from_state=old_state,
                to_state=new_state,
                drive_modulation=self._rhythm_drive_modulation,
            )
        except Exception:
            self._logger.debug("rhythm_adaptation_failed", exc_info=True)

    @property
    def active_goal_summaries(self) -> list[ActiveGoalSummary]:
        """
        Returns minimal goal summaries for Fovea's goal-relevance weighting.
        Fovea uses goal embeddings to boost salience of goal-relevant content.
        """
        from systems.fovea.types import ActiveGoalSummary as _ActiveGoalSummary

        if self._goal_manager is None:
            return []
        return [
            _ActiveGoalSummary(
                id=g.id,
                target_embedding=self._goal_embeddings.get(g.id, []),
                priority=g.priority,
            )
            for g in self._goal_manager.active_goals
        ]

    async def _embed_goal(self, goal: Goal) -> None:
        """Compute and cache the embedding for a goal's description."""
        if self._embed_fn is None:
            return
        try:
            embedding = await self._embed_fn(goal.description)
            self._goal_embeddings[goal.id] = embedding
        except Exception:
            self._logger.debug("goal_embedding_failed", goal_id=goal.id)

    @property
    def beliefs(self) -> Any:
        return self._belief_updater.beliefs

    # ─── Evolutionary Observable Emission ─────────────────────────

    async def _emit_evolutionary_observable(
        self,
        observable_type: str,
        value: float,
        is_novel: bool,
        metadata: dict | None = None,
    ) -> None:
        """Emit an evolutionary observable event via Synapse."""
        bus = getattr(self._synapse, "event_bus", None) if self._synapse else None
        if bus is None:
            return
        try:
            from primitives.evolutionary import EvolutionaryObservable
            from systems.synapse.types import SynapseEvent, SynapseEventType

            obs = EvolutionaryObservable(
                source_system=SystemID.NOVA,
                instance_id="",
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system="nova",
                data=obs.model_dump(mode="json"),
            )
            await bus.emit(event)
        except Exception:
            pass

    # ─── Private ──────────────────────────────────────────────────

    async def _emit_goal_lifecycle_event(
        self,
        event_name: str,
        goal: Goal,
        reason: str = "",
    ) -> None:
        """
        Emit GOAL_ACHIEVED or GOAL_ABANDONED on the Synapse bus.

        Closes the goal lifecycle visibility gap (Spec §20). Called
        fire-and-forget after update_progress returns ACHIEVED status,
        or after expire_stale_goals abandons suspended goals.
        """
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            event_type = (
                SynapseEventType.GOAL_ACHIEVED
                if event_name == "goal_achieved"
                else SynapseEventType.GOAL_ABANDONED
            )
            data: dict[str, object] = {
                "goal_id": goal.id,
                "description": goal.description[:200],
                "source": goal.source.value,
                "progress": round(goal.progress, 4),
                "drive_alignment": {
                    "coherence": goal.drive_alignment.coherence,
                    "care": goal.drive_alignment.care,
                    "growth": goal.drive_alignment.growth,
                    "honesty": goal.drive_alignment.honesty,
                },
            }
            if reason:
                data["reason"] = reason
            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=event_type,
                source_system="nova",
                data=data,
            ))
        except Exception as exc:
            self._logger.debug(f"goal_lifecycle_emit_failed_{event_name}", error=str(exc))

    async def _emit_goal_conflicts(
        self,
        conflicts: list[tuple[Goal, Goal, str]],
    ) -> None:
        """
        Emit GOAL_CONFLICT_DETECTED for each detected pair of conflicting goals.

        Telos subscribes to adjust drive topology; Equor may use this for
        escalation if a conflict involves floor drives (Care, Honesty).

        Gap 5 - multi-goal conflict detection (2026-03-07).
        """
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            for goal_a, goal_b, conflict_desc in conflicts:
                # Suggest resolution: suspend lower-priority goal
                if goal_a.priority >= goal_b.priority:
                    suggestion = f"Consider suspending '{goal_b.description[:60]}' (lower priority)"
                else:
                    suggestion = f"Consider suspending '{goal_a.description[:60]}' (lower priority)"

                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.GOAL_CONFLICT_DETECTED,
                    source_system="nova",
                    data={
                        "goal_a_id": goal_a.id,
                        "goal_b_id": goal_b.id,
                        "conflict_type": conflict_desc.split(":")[0],
                        "description": conflict_desc,
                        "resolution_suggestion": suggestion,
                    },
                ))
                self._logger.info(
                    "goal_conflict_detected",
                    goal_a=goal_a.id,
                    goal_b=goal_b.id,
                    conflict=conflict_desc[:80],
                )
        except Exception as exc:
            self._logger.debug("goal_conflict_emit_failed", error=str(exc))

    def _on_equor_failure(self, reason: str) -> None:
        """
        Callback fired by DeliberationEngine when Equor is unreachable or times out.

        Logs a Thymos DEGRADATION incident so the immune system can track
        constitutional gate unavailability and trigger repair if recurrent.
        The caller always falls back to do-nothing - no intent is dispatched
        without review. Never bypass; never proceed silently.
        """
        self._logger.warning("equor_unavailable", reason=reason)
        if self._thymos is not None:
            try:
                from systems.thymos.types import (
                    Incident,
                    IncidentClass,
                    IncidentSeverity,
                )

                incident = Incident(
                    incident_class=IncidentClass.DEGRADATION,
                    severity=IncidentSeverity.HIGH,
                    fingerprint="nova:equor_unavailable",
                    source_system="nova",
                    error_type="EquorUnavailable",
                    error_message=f"Equor constitutional gate unavailable: {reason[:200]}",
                    context={"reason": reason},
                    affected_systems=["nova", "equor"],
                    blast_radius=0.8,  # High - every intent now passes without review
                )
                asyncio.create_task(
                    self._thymos.on_incident(incident),
                    name="nova_equor_unavailable_incident",
                )
            except Exception as exc:
                self._logger.debug("equor_unavailable_thymos_escalation_failed", error=str(exc))

    async def _emit_hypothesis_feedback(
        self,
        intent_id: str,
        success: bool,
        regret: float | None,
        policy_name: str,
        goal_id: str,
        tournament_id: str | None = None,
        tournament_hypothesis_id: str | None = None,
    ) -> None:
        """
        Emit HYPOTHESIS_FEEDBACK on Synapse for every dispatched-intent outcome.

        Evo uses this event to update Thompson sampling weights for the
        policy class (decision path) - closing the gap where non-tournament
        deliberations were invisible to the learning loop.

        AV-EVO-1b: When tournament_id + tournament_hypothesis_id are present,
        Evo's handler also routes to record_tournament_outcome() internally.
        This eliminates the direct _evo reference that was needed before.

        Spec §20 HYPOTHESIS_FEEDBACK - partial → resolved (2026-03-07).
        AV-EVO-1b - resolved (2026-03-08).
        """
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Identify the decision path from the most recent decision record
            # that matches this intent, so Evo can bucket feedback by path type.
            decision_path = "unknown"
            for dr in reversed(self._decision_records):
                if dr.intent_dispatched:
                    decision_path = dr.path
                    break

            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.HYPOTHESIS_FEEDBACK,
                source_system="nova",
                data={
                    "intent_id": intent_id,
                    "success": success,
                    "regret": regret,
                    "policy_name": policy_name,
                    "decision_path": decision_path,
                    "goal_id": goal_id,
                    "tournament_id": tournament_id,
                    "tournament_hypothesis_id": tournament_hypothesis_id,
                },
            ))
        except Exception as exc:
            self._logger.debug("hypothesis_feedback_emit_failed", error=str(exc))

    async def _emit_thread_commit_request(
        self,
        intent_id: str,
        goal_id: str,
        policy_name: str,
        outcome_quality: float,
        success: bool,
    ) -> None:
        """
        Emit THREAD_COMMIT_REQUEST so Thread can record this decision epoch in
        the organism's narrative identity chain (Spec 15).

        Gap 4 - Thread integration (2026-03-07).
        """
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Build drive alignment snapshot from current drive weights
            drive_alignment = {k: round(v, 4) for k, v in self._drive_weights.items()}

            # Build a brief human-readable decision summary for the narrative
            outcome_word = "succeeded" if success else "failed"
            summary = (
                f"Policy '{policy_name}' {outcome_word} "
                f"(quality={round(outcome_quality, 3)}) "
                f"for intent {intent_id[:8]}"
            )

            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.THREAD_COMMIT_REQUEST,
                source_system="nova",
                data={
                    "intent_id": intent_id,
                    "goal_id": goal_id,
                    "policy_name": policy_name,
                    "outcome_quality": outcome_quality,
                    "drive_alignment": drive_alignment,
                    "decision_summary": summary,
                },
            ))
        except Exception as exc:
            self._logger.debug("thread_commit_request_emit_failed", error=str(exc))

    async def _handle_budget_exhaustion(self, broadcast_id: str) -> None:
        """
        Emit NOVA_DEGRADED and, after 10 consecutive exhausted decisions,
        escalate via Thymos as a COGNITIVE_DEGRADATION incident.

        Called fire-and-forget from receive_broadcast on every budget_exhausted path.
        """
        consecutive = self._consecutive_budget_exhausted

        # Estimate recovery time from the FE budget's expected window duration.
        estimated_recovery_s: float = 0.0
        if self._deliberation_engine is not None:
            budget = self._deliberation_engine.fe_budget
            # budget_nats / spend_rate gives rough window to recovery
            estimated_recovery_s = float(budget.budget_nats)  # conservative fallback
            current_budget = float(budget.remaining_nats)
        else:
            current_budget = 0.0

        self._logger.warning(
            "nova_degraded_heuristic_mode",
            reason="fe_budget_exhausted",
            consecutive_decisions=consecutive,
            current_budget=round(current_budget, 3),
            estimated_recovery_s=round(estimated_recovery_s, 1),
        )

        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.NOVA_DEGRADED,
                    source_system="nova",
                    data={
                        "reason": "fe_budget_exhausted",
                        "current_budget": round(current_budget, 3),
                        "estimated_recovery_time_s": round(estimated_recovery_s, 1),
                        "decisions_affected_since_degradation": consecutive,
                        "broadcast_id": broadcast_id,
                    },
                ))
            except Exception as exc:
                self._logger.debug("nova_degraded_emit_failed", error=str(exc))

        # After 10 consecutive degraded decisions, escalate via Thymos.
        if consecutive >= 10 and self._thymos is not None:
            try:
                from systems.thymos.types import (
                    Incident,
                    IncidentClass,
                    IncidentSeverity,
                )

                incident = Incident(
                    incident_class=IncidentClass.DEGRADATION,
                    severity=IncidentSeverity.HIGH,
                    fingerprint="nova:cognitive_degradation:budget_exhausted",
                    source_system="nova",
                    error_type="CognitiveDegradation",
                    error_message=(
                        f"Nova has operated in heuristic mode for "
                        f"{consecutive} consecutive decisions due to "
                        f"FE budget exhaustion."
                    ),
                    context={
                        "consecutive_heuristic_decisions": consecutive,
                        "current_budget_nats": round(current_budget, 3),
                        "estimated_recovery_s": round(estimated_recovery_s, 1),
                    },
                    affected_systems=["nova"],
                    blast_radius=0.6,
                )
                await self._thymos.on_incident(incident)
                self._logger.warning(
                    "nova_cognitive_degradation_escalated_to_thymos",
                    consecutive=consecutive,
                )
            except Exception as exc:
                self._logger.error(
                    "nova_thymos_escalation_failed",
                    error=str(exc),
                )

    async def _request_emergency_consolidation(self, broadcast_id: str) -> None:
        """
        Request emergency consolidation via Synapse when the free energy budget
        is exhausted.

        AV-EVO-1c: Replaces direct _evo.run_consolidation() call with an
        EVO_CONSOLIDATION_REQUESTED event.  Evo subscribes, triggers its
        consolidation pipeline, and replies with EVO_CONSOLIDATION_COMPLETE.
        Nova's _on_evo_consolidation_complete() resets the FE budget on receipt.

        If EVO_CONSOLIDATION_COMPLETE does not arrive within 90s (Evo's max
        consolidation time is 60s + margin), Nova resets the budget anyway to
        prevent permanent deliberation lockout.
        """
        if self._synapse is None:
            return
        try:
            self._logger.info("fe_budget_emergency_consolidation_requested")
            from systems.synapse.types import SynapseEvent, SynapseEventType
            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EVO_CONSOLIDATION_REQUESTED,
                source_system="nova",
                data={
                    "reason": "fe_budget_exhausted",
                    "broadcast_id": broadcast_id,
                },
            ))
            # Safety net: reset budget after 90s regardless of Evo response.
            await asyncio.sleep(90)
            if self._deliberation_engine is not None and self._deliberation_engine.fe_budget.is_exhausted:
                self._logger.warning("fe_budget_consolidation_timeout_reset")
                self._deliberation_engine.reset_fe_budget()
        except Exception as exc:
            self._logger.error("fe_budget_consolidation_request_failed", error=str(exc))
            if self._deliberation_engine is not None:
                self._deliberation_engine.reset_fe_budget()

    async def _on_evo_consolidation_complete(self, event: Any) -> None:
        """
        Handle EVO_CONSOLIDATION_COMPLETE from Evo.

        When Evo finishes a consolidation cycle (whether triggered by Nova's
        EVO_CONSOLIDATION_REQUESTED or by its own schedule), reset Nova's FE
        budget so deliberation can resume with a clean slate.
        """
        try:
            data = getattr(event, "data", event) if not isinstance(event, dict) else event
            duration_ms = data.get("duration_ms", 0)
            hypotheses_integrated = data.get("hypotheses_integrated", 0)
            self._logger.info(
                "fe_budget_reset_after_consolidation",
                duration_ms=duration_ms,
                hypotheses_integrated=hypotheses_integrated,
            )
            if self._deliberation_engine is not None:
                self._deliberation_engine.reset_fe_budget()
        except Exception as exc:
            self._logger.debug("evo_consolidation_complete_handler_failed", error=str(exc))

    async def _dispatch_intent(
        self,
        intent: Intent,
        broadcast: WorkspaceBroadcast,
        decision_record: DecisionRecord | None = None,
    ) -> None:
        """Dispatch an approved intent via the intent router."""
        assert self._intent_router is not None
        try:
            # Thread the Equor check from the deliberation engine so the
            # router (and Axon) receive the real verdict, not a default.
            equor_check = (
                self._deliberation_engine.last_equor_check
                if self._deliberation_engine else None
            )
            goal_id = getattr(intent.goal, "id", None) or intent.goal.description[:50]

            # Emit INTENT_SUBMITTED before routing - audit trail before the
            # intent leaves Nova (Spec §20 open gap closed).
            if self._synapse is not None:
                try:
                    from systems.synapse.types import SynapseEvent, SynapseEventType

                    asyncio.create_task(
                        self._synapse.event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.INTENT_SUBMITTED,
                            source_system="nova",
                            data={
                                "intent_id": intent.id,
                                "goal_id": goal_id,
                                "policy_name": decision_record.selected_policy_name
                                if decision_record else "",
                                "path": decision_record.path if decision_record else "",
                                "efe_score": (
                                    min(decision_record.efe_scores.values())
                                    if decision_record and decision_record.efe_scores else None
                                ),
                            },
                        )),
                        name=f"nova_intent_submitted_{intent.id[:8]}",
                    )
                except Exception:
                    pass

            route = await self._intent_router.route(
                intent=intent,
                affect=broadcast.affect,
                conversation_id=self._current_conversation_id,
                equor_check=equor_check,
            )
            if route != "internal":
                self._total_intents_approved += 1
                # Track pending intent
                executors = [s.executor for s in intent.plan.steps]
                self._pending_intents[intent.id] = PendingIntent(
                    intent_id=intent.id,
                    goal_id=goal_id,
                    routed_to=route,
                    executors=executors,
                    # Carry tournament context so outcomes can be fed back to Evo
                    tournament_id=decision_record.tournament_id if decision_record else None,
                    tournament_hypothesis_id=(
                        decision_record.tournament_hypothesis_id if decision_record else None
                    ),
                )

                # Emit INTENT_ROUTED after successful dispatch (Spec §20 open gap closed).
                if self._synapse is not None:
                    try:
                        from systems.synapse.types import SynapseEvent, SynapseEventType

                        asyncio.create_task(
                            self._synapse.event_bus.emit(SynapseEvent(
                                event_type=SynapseEventType.INTENT_ROUTED,
                                source_system="nova",
                                data={
                                    "intent_id": intent.id,
                                    "goal_id": goal_id,
                                    "routed_to": route,
                                    "executors": executors,
                                },
                            )),
                            name=f"nova_intent_routed_{intent.id[:8]}",
                        )
                    except Exception:
                        pass
        except Exception as exc:
            self._logger.error("intent_dispatch_failed", intent_id=intent.id, error=str(exc))
            self._total_intents_blocked += 1

    async def _retrieve_relevant_memories_safe(
        self,
        broadcast: WorkspaceBroadcast,
    ) -> list[dict[str, Any]]:
        """
        Retrieve episodes relevant to the current percept's semantic content.

        Uses the percept's pre-computed embedding (from Atune) as the primary
        query vector so retrieval is grounded in the full semantic content, not
        just a 200-char text prefix.  Falls back to text-only if the embedding
        is absent.  Returned dicts are rich enough for the LLM to reason about
        past context: summary, affect, salience, source, and recency.
        """
        try:
            content = broadcast.content

            # Prefer the percept's dense embedding - already computed by Atune,
            # covers the full content and gives best hybrid-retrieval quality.
            query_embedding: list[float] | None = None
            inner = getattr(content, "content", None)  # Percept wraps a Content obj
            if inner is not None:
                query_embedding = getattr(inner, "embedding", None)
            # Fallback: embedding directly on content (some broadcast shapes)
            if not query_embedding:
                query_embedding = getattr(content, "embedding", None)

            # Also extract text for BM25 leg of the hybrid retrieval.
            query_text = ""
            for attr in ["content", "text", "summary"]:
                obj = getattr(content, attr, None)
                if isinstance(obj, str) and obj:
                    query_text = obj[:200]
                    break
            # If content is a Percept, dig into its Content.raw
            if not query_text and inner is not None:
                raw = getattr(inner, "raw", None)
                if isinstance(raw, str):
                    query_text = raw[:200]

            # Nothing to search on - skip retrieval
            if not query_text and not query_embedding:
                return []

            result = await asyncio.wait_for(
                self._memory.retrieve(
                    query_text=query_text or None,
                    query_embedding=query_embedding,
                    max_results=5,
                ),
                timeout=0.15,  # 150ms hard timeout - must not block the cycle
            )

            if not result.traces:
                return []

            # Enrich retrieved episodes with affect/source/time in one bulk query.
            # The retrieval leg queries only return summary+salience; these fields
            # live on the Episode node and are needed for the LLM policy prompt.
            episode_ids = [t.node_id for t in result.traces if t.node_type == "episode"]
            episode_meta: dict[str, dict[str, Any]] = {}
            if episode_ids and self._memory is not None:
                try:
                    episode_meta = await self._memory.get_episodes_meta(episode_ids)
                except Exception:
                    pass  # Meta enrichment is best-effort; proceed without it

            traces: list[dict[str, Any]] = []
            for trace in result.traces[:5]:
                summary = trace.content
                if not summary:
                    continue
                meta = episode_meta.get(trace.node_id, {})
                entry: dict[str, Any] = {
                    "summary": str(summary)[:300],
                    "salience": round(trace.unified_score, 3),
                    "affect_valence": meta.get("affect_valence"),
                    "source": meta.get("source"),
                    "event_time": meta.get("event_time"),
                }
                traces.append(entry)
            return traces
        except (TimeoutError, Exception):
            return []

    async def _sync_goals_to_atune_safe(self) -> None:
        """Non-blocking: sync active goal summaries to Atune/Fovea for goal-relevance weighting."""
        if self._goal_sync_callback is not None:
            try:
                self._goal_sync_callback(self.active_goal_summaries)
            except Exception:
                self._logger.debug("goal_sync_callback_failed", exc_info=True)

    def _expire_stale_pending_intents(self, max_age_seconds: float = 600.0) -> None:
        """
        Remove pending intents that never received an outcome callback.

        If Axon crashes mid-execution or the outcome callback is lost, the
        intent stays in ``_pending_intents`` forever, leaking memory and
        causing the heartbeat to think the organism is perpetually busy.

        Default max age is 10 minutes - generous enough for real executors
        (bounty solving can take 5–8 minutes) but still bounds the leak.
        Heavy executors like ``hunt_bounties`` and ``spawn_child`` are
        given a 2× allowance (20 minutes) because they routinely take longer.
        """
        from primitives.common import utc_now

        now = utc_now()
        stale: list[str] = []

        for intent_id, pending in self._pending_intents.items():
            age_s = (now - pending.dispatched_at).total_seconds()
            limit = max_age_seconds * 2 if any(
                ex in _HEAVY_EXECUTORS for ex in pending.executors
            ) else max_age_seconds
            if age_s > limit:
                stale.append(intent_id)

        for intent_id in stale:
            self._pending_intents.pop(intent_id, None)
            self._logger.warning(
                "pending_intent_expired",
                intent_id=intent_id,
                message="No outcome received within timeout - clearing from pending map",
            )

    # ─── Counterfactual Storage & Resolution ────────────────────────

    def _build_counterfactual_records(
        self,
        intent_id: str,
        decision_record: DecisionRecord,
        rejected: list[tuple[Any, Any]],
    ) -> list[CounterfactualRecord]:
        """
        Build CounterfactualRecord objects from rejected (Policy, EFEScore) pairs.
        Called synchronously after slow-path deliberation selects a policy.
        """
        # Identify the chosen policy from the decision record
        chosen_name = decision_record.selected_policy_name
        chosen_efe = decision_record.efe_scores.get(chosen_name, 0.0)

        records: list[CounterfactualRecord] = []
        for policy, efe_score in rejected:
            records.append(CounterfactualRecord(
                intent_id=intent_id,
                decision_record_id=decision_record.id,
                goal_id=decision_record.goal_id or "",
                goal_description=decision_record.goal_description,
                policy_name=policy.name,
                policy_type=policy.type,
                policy_description=policy.description[:300],
                policy_reasoning=policy.reasoning[:500],
                efe_total=efe_score.total,
                estimated_pragmatic_value=efe_score.pragmatic.score,
                estimated_epistemic_value=efe_score.epistemic.score,
                constitutional_alignment=efe_score.constitutional_alignment,
                feasibility=efe_score.feasibility,
                risk_expected_harm=efe_score.risk.expected_harm,
                chosen_policy_name=chosen_name,
                chosen_efe_total=chosen_efe,
            ))
        return records

    async def _persist_counterfactuals(
        self,
        records: list[CounterfactualRecord],
    ) -> None:
        """Fire-and-forget: write counterfactual episodes to Neo4j."""
        for record in records:
            try:
                await self._memory.store_counterfactual_episode(record)
            except Exception as exc:
                self._logger.debug(
                    "counterfactual_persist_failed",
                    cf_id=record.id,
                    error=str(exc),
                )

    async def _resolve_counterfactuals(
        self,
        records: list[CounterfactualRecord],
        outcome: IntentOutcome,
        actual_pragmatic: float | None = None,
    ) -> None:
        """
        Resolve counterfactual episodes with regret when outcome arrives.

        Regret = estimated_pragmatic_value - actual_pragmatic_value
        Positive regret means the counterfactual was estimated to perform better
        than the actual outcome - i.e. the organism might have chosen poorly.

        actual_pragmatic is a continuous [0.0, 1.0] signal computed as:
            outcome_quality × goal_achievement_degree × (1 - regret_estimate)
        This gives Evo's Thompson sampler a real gradient rather than a binary flip.
        Falls back to 1.0/0.0 if no graded value is supplied (e.g. no goal context).
        """
        if actual_pragmatic is None:
            actual_pragmatic = 1.0 if outcome.success else 0.0

        for record in records:
            try:
                regret = record.estimated_pragmatic_value - actual_pragmatic
                # Stamp regret on the in-memory record so _build_counterfactual_text
                # can access it when building the RE training example below.
                record.regret = regret
                record.actual_pragmatic_value = actual_pragmatic
                record.actual_outcome_success = outcome.success
                record.resolved = True
                await self._memory.resolve_counterfactual(
                    record_id=record.id,
                    outcome_success=outcome.success,
                    actual_pragmatic_value=actual_pragmatic,
                    regret=regret,
                )
                # Link to outcome episode if available
                if outcome.episode_id:
                    await self._memory.link_counterfactual_to_outcome(
                        counterfactual_id=record.id,
                        outcome_episode_id=outcome.episode_id,
                    )
            except Exception as exc:
                self._logger.debug(
                    "counterfactual_resolve_failed",
                    cf_id=record.id,
                    error=str(exc),
                )

        self._logger.info(
            "counterfactuals_resolved",
            intent_id=outcome.intent_id,
            count=len(records),
            actual_success=outcome.success,
        )

        # ── Post-outcome RE training example with resolved counterfactuals ──
        # This is the highest-value training signal: the organism now knows what
        # actually happened and can compare it against every alternative it considered.
        # Emitted only when counterfactuals exist (i.e. slow-path with rejected policies).
        if records:
            _cf_text = self._build_counterfactual_text(
                cf_records=records,
                actual_pragmatic=actual_pragmatic,
                success=outcome.success,
            )
            chosen_name = records[0].chosen_policy_name if records else ""
            asyncio.create_task(
                self._emit_re_training_example(
                    category="planning_deliberation_resolved",
                    instruction=(
                        f"Outcome resolved for intent {outcome.intent_id[:12]}. "
                        f"Evaluate: was '{chosen_name}' the right choice?"
                    ),
                    input_context=(
                        f"intent_id={outcome.intent_id}, "
                        f"success={outcome.success}, "
                        f"actual_pragmatic={actual_pragmatic:.4f}, "
                        f"counterfactuals_count={len(records)}"
                    ),
                    output=(
                        f"chosen_policy={chosen_name}\n"
                        f"outcome={'success' if outcome.success else 'failure'}\n"
                        f"actual_pragmatic={actual_pragmatic:.4f}\n"
                        f"regret_analysis: see counterfactual field"
                    ),
                    outcome_quality=actual_pragmatic,
                    episode_id=outcome.intent_id,
                    reasoning_trace=(
                        f"The organism chose '{chosen_name}' and the outcome was "
                        f"{'success' if outcome.success else 'failure'} "
                        f"(actual_pragmatic={actual_pragmatic:.4f}). "
                        f"Counterfactual regret analysis across {len(records)} "
                        f"rejected alternative(s) follows in the counterfactual field."
                    ),
                    alternatives=[r.policy_name for r in records],
                    counterfactual=_cf_text,
                ),
                name=f"nova_re_cf_{outcome.intent_id[:8]}",
            )

    def _expire_stale_pending_counterfactuals(
        self,
        max_age_seconds: float = 600.0,
    ) -> None:
        """Remove pending counterfactuals whose intents never received outcomes."""
        from primitives.common import utc_now

        now = utc_now()
        stale: list[str] = []
        for intent_id, records in self._pending_counterfactuals.items():
            if records and (now - records[0].timestamp).total_seconds() > max_age_seconds:
                stale.append(intent_id)
        for intent_id in stale:
            expired = self._pending_counterfactuals.pop(intent_id, [])
            self._logger.debug(
                "pending_counterfactuals_expired",
                intent_id=intent_id,
                count=len(expired),
            )

    async def _flush_beliefs_safe(self) -> None:
        """Fire-and-forget: persist dirty beliefs to Neo4j."""
        try:
            await self._belief_updater.persist_beliefs()
        except Exception:
            self._logger.debug("belief_flush_failed", exc_info=True)

    def _record_decision(self, record: DecisionRecord) -> None:
        """Store decision record for observability (ring buffer) and persist to Neo4j."""
        self._decision_records.append(record)
        if len(self._decision_records) > self._max_decision_records:
            self._decision_records = self._decision_records[-self._max_decision_records:]
        asyncio.create_task(
            self._persist_decision_record(record),
            name=f"nova_persist_decision_{record.broadcast_id[:8]}",
        )

    async def _persist_decision_record(self, record: DecisionRecord) -> None:
        """
        Fire-and-forget: write (:Decision) node to Neo4j and (if re_training_eligible)
        push to Redis Stream re_training_queue for the RE training pipeline.

        Gap 1 - DecisionRecord Neo4j persistence.
        Gap 2 - RE training data emission to Redis Stream.
        """
        # ── Gap 1: Neo4j ──────────────────────────────────────────────────
        neo4j = self._memory.get_neo4j()
        if neo4j is not None:
            try:
                efe_score: float | None = None
                if record.efe_scores:
                    efe_score = min(record.efe_scores.values())

                query = """
                MERGE (d:Decision {id: $id})
                SET d.intent_id          = $intent_id,
                    d.broadcast_id       = $broadcast_id,
                    d.policy_type        = $policy_type,
                    d.path               = $path,
                    d.efe_score          = $efe_score,
                    d.model_used         = $model_used,
                    d.re_training_eligible = $re_training_eligible,
                    d.slow_path          = $slow_path,
                    d.timestamp          = $timestamp
                WITH d
                CALL {
                    WITH d
                    MATCH (g:Goal {id: $goal_id})
                    MERGE (d)-[:MOTIVATED_BY]->(g)
                }
                IN TRANSACTIONS OF 1 ROW
                RETURN d.id
                """
                # Simpler version without CALL {} for broader Neo4j compatibility:
                cypher = """
                MERGE (d:Decision {id: $id})
                SET d.intent_id            = $intent_id,
                    d.broadcast_id         = $broadcast_id,
                    d.policy_type          = $policy_type,
                    d.path                 = $path,
                    d.efe_score            = $efe_score,
                    d.model_used           = $model_used,
                    d.re_training_eligible = $re_training_eligible,
                    d.slow_path            = $slow_path,
                    d.timestamp            = $timestamp
                """
                await neo4j.execute_write(
                    cypher,
                    parameters={
                        "id": record.broadcast_id,
                        "intent_id": record.intent_id or "",
                        "broadcast_id": record.broadcast_id,
                        "policy_type": record.selected_policy or "",
                        "path": record.path,
                        "efe_score": efe_score,
                        "model_used": record.model_used,
                        "re_training_eligible": record.re_training_eligible,
                        "slow_path": record.path == "slow",
                        "timestamp": record.timestamp.isoformat() if hasattr(record, "timestamp") else "",
                    },
                )
                # Link to Goal if known
                if record.goal_id:
                    link_cypher = """
                    MATCH (d:Decision {id: $decision_id})
                    MATCH (g:Goal {id: $goal_id})
                    MERGE (d)-[:MOTIVATED_BY]->(g)
                    """
                    await neo4j.execute_write(
                        link_cypher,
                        parameters={"decision_id": record.broadcast_id, "goal_id": record.goal_id},
                    )
            except Exception as exc:
                self._logger.debug("decision_persist_neo4j_failed", error=str(exc))

        # ── Gap 2: Redis Stream re_training_queue ─────────────────────────
        if record.re_training_eligible:
            try:
                import json

                redis = getattr(self._memory, "_redis", None)
                if redis is None:
                    redis = getattr(self._synapse, "_redis", None)
                if redis is not None:
                    payload = {
                        "stream": "nova_deliberation",
                        "broadcast_id": record.broadcast_id,
                        "intent_id": record.intent_id or "",
                        "goal_id": record.goal_id or "",
                        "path": record.path,
                        "selected_policy": record.selected_policy or "",
                        "efe_scores": json.dumps(record.efe_scores or {}),
                        "model_used": record.model_used,
                        "slow_path": "1" if record.path == "slow" else "0",
                    }
                    await redis.xadd("re_training_queue", payload)
            except Exception as exc:
                self._logger.debug("decision_re_stream_failed", error=str(exc))

        # ── Gap 6: Induce procedure template from successful slow-path decisions ──
        # Criteria: slow path, intent was dispatched, EFE < -0.3 (clear winner)
        if (
            record.path == "slow"
            and record.intent_dispatched
            and record.selected_policy
            and record.efe_scores
            and min(record.efe_scores.values()) < -0.3
            and neo4j is not None
        ):
            asyncio.create_task(
                self._induce_procedure_from_record(record, neo4j),
                name=f"nova_induce_procedure_{record.broadcast_id[:8]}",
            )

    async def _induce_procedure_from_record(
        self,
        record: DecisionRecord,
        neo4j: Any,
    ) -> None:
        """
        Persist a successful slow-path decision as an induced (:Procedure) node in
        Neo4j and inject it into _DYNAMIC_PROCEDURES for fast-path matching.

        Induction criteria (applied by caller):
        - Path = slow
        - Intent was dispatched
        - Best EFE score < −0.3 (clear winner, not just less-bad)

        Gap 6 - procedure template induction (2026-03-07).
        """
        try:
            from primitives.common import new_id
            from systems.nova.policy_generator import register_dynamic_procedure

            procedure_id = new_id()
            name = record.selected_policy or "induced_procedure"
            efe_score = min(record.efe_scores.values()) if record.efe_scores else 0.0

            cypher = """
            MERGE (p:Procedure {name: $name})
            ON CREATE SET
                p.id            = $id,
                p.source        = 'nova_induction',
                p.domain        = $domain,
                p.efe_score     = $efe_score,
                p.success_rate  = $success_rate,
                p.induction_count = 1,
                p.created_at    = $timestamp
            ON MATCH SET
                p.efe_score     = ($efe_score + p.efe_score) / 2.0,
                p.induction_count = p.induction_count + 1,
                p.success_rate  = CASE
                    WHEN p.induction_count > 0
                    THEN (p.success_rate * p.induction_count + 1.0) / (p.induction_count + 1)
                    ELSE 0.9
                END
            RETURN p.id, p.success_rate, p.induction_count
            """
            # Infer domain from goal_id or selected_policy name
            domain = "general"
            if record.goal_id:
                # Map known goal prefixes to domains
                for kw, dom in (
                    ("bounty", "economic"), ("care", "care"),
                    ("coherence", "coherence"), ("memory", "epistemic"),
                ):
                    if kw in (record.goal_id or "").lower():
                        domain = dom
                        break

            await neo4j.execute_read(
                cypher,
                parameters={
                    "id": procedure_id,
                    "name": name,
                    "domain": domain,
                    "efe_score": round(efe_score, 4),
                    "success_rate": 0.9,
                    "timestamp": record.timestamp.isoformat() if hasattr(record, "timestamp") else "",
                },
            )

            # Register into the fast-path dynamic procedure pool so the organism
            # can reuse this pattern without an LLM call next time.
            register_dynamic_procedure({
                "name": name,
                "condition": lambda b, _n=name: (
                    _n.lower().replace("_", " ").split()[0]
                    in str(getattr(getattr(b, "content", None), "content", "") or "").lower()
                ),
                "domain": domain,
                "steps": [{"action_type": "observe", "description": f"Execute {name} procedure"}],
                "success_rate": 0.9,
                "effort": "medium",
                "time_horizon": "short",
            })

            self._logger.info(
                "procedure_induced",
                name=name,
                domain=domain,
                efe_score=round(efe_score, 4),
            )
        except Exception as exc:
            self._logger.debug("procedure_induction_failed", error=str(exc))

    async def _load_induced_procedures(self, neo4j: Any) -> None:
        """
        Load all induced (:Procedure) nodes from Neo4j into _DYNAMIC_PROCEDURES
        so the fast path can pattern-match against them without LLM calls.

        Called once at end of initialize(). Also callable after sleep to pick up
        Oneiros-consolidated procedures.

        Gap 6 - procedure template induction (2026-03-07).
        """
        try:
            from systems.nova.policy_generator import (
                clear_dynamic_procedures,
                register_dynamic_procedure,
            )

            cypher = """
            MATCH (p:Procedure {source: 'nova_induction'})
            WHERE p.success_rate >= 0.7
            RETURN p.name AS name, p.domain AS domain,
                   p.success_rate AS success_rate, p.induction_count AS count
            ORDER BY p.success_rate DESC
            LIMIT 50
            """
            result = await neo4j.execute_read(cypher, parameters={})
            records = result.records if hasattr(result, "records") else []

            loaded = 0
            for row in records:
                name = row.get("name") or ""
                domain = row.get("domain") or "general"
                success_rate = float(row.get("success_rate") or 0.9)
                if not name:
                    continue
                register_dynamic_procedure({
                    "name": name,
                    "condition": lambda b, _n=name: (
                        _n.lower().replace("_", " ").split()[0]
                        in str(getattr(getattr(b, "content", None), "content", "") or "").lower()
                    ),
                    "domain": domain,
                    "steps": [{"action_type": "observe", "description": f"Execute {name} procedure"}],
                    "success_rate": success_rate,
                    "effort": "medium",
                    "time_horizon": "short",
                })
                loaded += 1

            if loaded:
                self._logger.info("induced_procedures_loaded", count=loaded)
        except Exception as exc:
            self._logger.debug("induced_procedure_load_failed", error=str(exc))

    def _has_active_coherence_goal(self) -> bool:
        """Check if there's already an active coherence-repair goal."""
        if self._goal_manager is None:
            return False
        return any(
            g.source == GoalSource.SELF_GENERATED
            and "coherence" in g.description.lower()
            for g in self._goal_manager.active_goals
        )

    def _create_coherence_repair_goal(self, broadcast: WorkspaceBroadcast) -> None:
        """
        High coherence stress means the organism's beliefs conflict with
        incoming percepts. Generate a self-repair goal to seek clarification.
        """
        if self._goal_manager is None:
            return

        from primitives.common import DriveAlignmentVector, new_id

        goal = Goal(
            id=new_id(),
            description=(
                "Resolve coherence conflict: seek clarifying information"
                " to reconcile contradictory beliefs"
            ),
            source=GoalSource.SELF_GENERATED,
            priority=0.7,
            urgency=broadcast.affect.coherence_stress,
            importance=0.6,
            drive_alignment=DriveAlignmentVector(
                coherence=0.9, care=0.0, growth=0.1, honesty=0.0,
            ),
            status=GoalStatus.ACTIVE,
        )
        self._goal_manager.add_goal(goal)
        self._logger.info(
            "coherence_repair_goal_created",
            stress=round(broadcast.affect.coherence_stress, 3),
        )

    # ══════════════════════════════════════════════════════════════════
    # ── Autonomy Audit Features (8 Mar 2026) ─────────────────────────
    # ══════════════════════════════════════════════════════════════════

    async def _on_axon_capability_snapshot(self, event: Any) -> None:
        """
        Cache Axon's per-cycle capability snapshot.

        Nova uses this to:
        1. Prune infeasible action_types from policy generation (don't waste
           Equor's review budget on circuit-broken executors)
        2. Detect when the organism needs a capability it doesn't have
           (→ emit AXON_EXECUTOR_REQUEST via Axon)
        3. Adjust feasibility scoring in EFE evaluation
        """
        data = getattr(event, "data", {}) or {}
        self._axon_capabilities = data
        self._axon_budget_remaining = int(data.get("budget_remaining", 5))
        self._axon_is_sleeping = bool(data.get("is_sleeping", False))

        # Build fast-lookup index: action_type → executor info
        executors = data.get("executors", [])
        index: dict[str, dict[str, Any]] = {}
        for ex in executors:
            at = ex.get("action_type", "")
            if at:
                index[at] = ex
        self._axon_executor_index = index

    def is_executor_available(self, action_type: str) -> bool:
        """
        Check if an executor is currently available for use.

        Returns False if: not registered, circuit breaker OPEN, rate limit
        exhausted, or Axon is sleeping.
        """
        if self._axon_is_sleeping:
            return False
        ex = self._axon_executor_index.get(action_type)
        if ex is None:
            return False
        if ex.get("circuit_breaker_status") == "OPEN":
            return False
        remaining = ex.get("rate_limit_remaining")
        if remaining is not None and remaining <= 0:
            return False
        return True

    def get_executor_health(self, action_type: str) -> dict[str, Any]:
        """
        Return the live health snapshot for an executor.
        Returns empty dict if unknown.
        """
        return self._axon_executor_index.get(action_type, {})

    async def _on_axon_intent_pivot(self, event: Any) -> None:
        """
        Handle AXON_INTENT_PIVOT - mid-execution replanning request.

        When Axon signals a step failure with remaining steps and a fallback
        goal, Nova generates a replacement plan via immediate deliberation.
        """
        data = getattr(event, "data", {}) or {}
        intent_id = data.get("intent_id", "")
        failed_action = data.get("failed_action_type", "")
        failure_reason = data.get("failure_reason", "")
        fallback_goal = data.get("fallback_goal")
        remaining_steps = data.get("remaining_steps", [])

        self._logger.info(
            "intent_pivot_received",
            intent_id=intent_id,
            failed_action=failed_action,
            remaining_steps=len(remaining_steps),
            has_fallback=bool(fallback_goal),
        )

        # If there's a fallback goal, inject it as a high-priority immediate goal
        if fallback_goal and self._goal_manager is not None:
            from primitives.common import DriveAlignmentVector, new_id
            pivot_goal = Goal(
                id=new_id(),
                description=f"Pivot from failed {failed_action}: {fallback_goal}",
                source=GoalSource.SELF_GENERATED,
                priority=0.85,
                urgency=0.9,
                importance=0.7,
                drive_alignment=DriveAlignmentVector(
                    coherence=0.5, care=0.3, growth=0.1, honesty=0.1,
                ),
                status=GoalStatus.ACTIVE,
            )
            self._goal_manager.add_goal(pivot_goal)

            # Trigger immediate deliberation for the pivot
            if hasattr(self, "_immediate_deliberation"):
                asyncio.create_task(self._immediate_deliberation(
                    urgency=0.85,
                    reason=f"intent_pivot:{failed_action}",
                ))

    # ── Policy Effectiveness Self-Tracking ────────────────────────────

    def record_policy_effectiveness(
        self, policy_name: str, success: bool
    ) -> None:
        """
        Track per-policy-class success/failure for self-awareness.

        Called from process_outcome() and _on_axon_execution_result().
        Nova can then reason about which policy classes work and which don't.
        """
        if policy_name not in self._policy_effectiveness:
            self._policy_effectiveness[policy_name] = {
                "success": 0, "failure": 0, "total": 0,
            }
        entry = self._policy_effectiveness[policy_name]
        entry["total"] += 1
        if success:
            entry["success"] += 1
        else:
            entry["failure"] += 1

    def get_policy_effectiveness_summary(self) -> dict[str, Any]:
        """
        Return per-policy success rates for introspection.

        Nova can use this during slow-path deliberation to weight policies
        by historical effectiveness - not just EFE prediction.
        """
        summary: dict[str, Any] = {}
        for name, counts in self._policy_effectiveness.items():
            total = counts["total"]
            rate = counts["success"] / total if total > 0 else 0.5
            summary[name] = {
                "success_rate": round(rate, 3),
                "total": total,
                "success": counts["success"],
                "failure": counts["failure"],
            }
        return summary

    # ── Equor Rejection Pattern Detection ─────────────────────────────

    def record_equor_rejection(
        self,
        policy_name: str,
        action_types: list[str],
        denial_reason: str,
    ) -> None:
        """
        Track Equor denial patterns so Nova can learn which policy classes
        are systematically rejected and stop generating them.
        """
        entry = {
            "policy_name": policy_name,
            "action_types": action_types,
            "denial_reason": denial_reason,
            "ts": time.monotonic(),
        }
        self._equor_rejections.append(entry)
        # Keep last 100
        if len(self._equor_rejections) > 100:
            self._equor_rejections = self._equor_rejections[-100:]

        # Track pattern frequency
        pattern_key = f"{policy_name}:{denial_reason[:50]}"
        self._equor_rejection_patterns[pattern_key] = (
            self._equor_rejection_patterns.get(pattern_key, 0) + 1
        )

        # Log if systematic (≥3 identical rejections)
        count = self._equor_rejection_patterns[pattern_key]
        if count >= 3 and count % 3 == 0:
            self._logger.warning(
                "equor_systematic_rejection_detected",
                policy_name=policy_name,
                denial_reason=denial_reason[:80],
                rejection_count=count,
            )

    def is_policy_systematically_rejected(self, policy_name: str) -> bool:
        """
        Return True if a policy class has been rejected ≥5 times recently.
        Nova should deprioritize or skip these during generation.
        """
        for key, count in self._equor_rejection_patterns.items():
            if key.startswith(f"{policy_name}:") and count >= 5:
                return True
        return False

    # ── Oikos Economic State Cache ────────────────────────────────────

    async def _on_system_modulation(self, event: Any) -> None:
        """Handle VitalityCoordinator austerity orders.

        Skia emits SYSTEM_MODULATION when the organism needs to conserve resources.
        This system applies the directive and ACKs so Skia knows the order was received.
        """
        data = getattr(event, "data", {}) or {}
        level = data.get("level", "nominal")
        halt_systems = data.get("halt_systems", [])
        modulate = data.get("modulate", {})

        system_id = "nova"
        compliant = True
        reason: str | None = None

        if system_id in halt_systems:
            self._modulation_halted = True
            self._logger.warning("system_modulation_halt", level=level)
        elif system_id in modulate:
            directives = modulate[system_id]
            self._apply_modulation_directives(directives)
            self._logger.info("system_modulation_applied", level=level, directives=directives)
        elif level == "nominal":
            self._modulation_halted = False
            self._logger.info("system_modulation_resumed", level=level)

        # Emit ACK so Skia knows the order was received
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_MODULATION_ACK,
                    data={
                        "system_id": system_id,
                        "level": level,
                        "compliant": compliant,
                        "reason": reason,
                    },
                    source_system=system_id,
                ))
            except Exception as exc:
                self._logger.warning("modulation_ack_failed", error=str(exc))

    def _apply_modulation_directives(self, directives: dict) -> None:
        """Apply modulation directives from VitalityCoordinator.

        Nova directive: {"mode": "high_salience_only"} - restrict deliberation
        to only high-salience percepts to reduce cognitive load.
        """
        mode = directives.get("mode")
        if mode == "high_salience_only":
            self._logger.info("modulation_high_salience_only_mode_set")
        else:
            self._logger.info("modulation_directives_received", directives=directives)

    async def _on_economic_state_updated(self, event: Any) -> None:
        """
        Cache Oikos economic state for planning-time visibility.

        Nova can now answer "what's my current balance?" and "what's my
        burn rate?" during policy generation - not just react to deltas.
        """
        data = getattr(event, "data", {}) or {}
        self._cached_balance_usd = float(data.get("liquid_balance_usd", 0.0))
        self._cached_burn_rate_usd_per_hour = float(data.get("burn_rate_usd_per_hour", 0.0))
        self._cached_economic_state_ts = time.monotonic()

        # Pass current portfolio APYs to the opportunity scanner so it can
        # compute yield improvement thresholds accurately.
        deployed_apys = data.get("deployed_apys", [])
        if isinstance(deployed_apys, list) and deployed_apys:
            self._opportunity_scanner.set_portfolio_apys([float(a) for a in deployed_apys])

    def get_economic_context(self) -> dict[str, Any]:
        """
        Return cached economic state for policy generation context.

        Stale after 5 minutes - returns empty dict if too old.
        """
        if self._cached_economic_state_ts == 0.0:
            return {}
        age_s = time.monotonic() - self._cached_economic_state_ts
        if age_s > 300:
            return {}  # stale
        return {
            "balance_usd": self._cached_balance_usd,
            "burn_rate_usd_per_hour": self._cached_burn_rate_usd_per_hour,
            "hours_until_depleted": (
                self._cached_balance_usd / self._cached_burn_rate_usd_per_hour
                if self._cached_burn_rate_usd_per_hour > 0 else float("inf")
            ),
            "age_seconds": round(age_s, 1),
        }

    # ── Kairos Causal Knowledge ───────────────────────────────────────────

    async def _on_kairos_invariant(self, event: Any) -> None:
        """
        Consume KAIROS_INVARIANT_DISTILLED events - wire Kairos causal
        discoveries into Nova's deliberation context.

        Payload fields used:
          invariant_id   - used to fetch the full CausalInvariant from Neo4j
          abstract_form  - human-readable causal statement (fallback if Neo4j unavailable)
          domain_count   - number of domains the invariant spans
          tier           - 1=domain, 2=cross-domain, 3=substrate-independent

        Only invariants with confidence ≥ 0.7 AND tier ≥ 2 are injected as
        belief entities and surfaced in the slow-path LLM prompt. Lower-tier
        invariants are still stored but not used to trigger replanning.
        """
        try:
            data = getattr(event, "data", {}) or {}
            invariant_id: str = str(data.get("invariant_id", ""))
            abstract_form: str = str(data.get("abstract_form", ""))
            domain_count: int = int(data.get("domain_count", 1))

            if not invariant_id and not abstract_form:
                return

            # Try to fetch the full CausalInvariant from Neo4j via Memory.
            # If unavailable, fall back to heuristic parsing of abstract_form.
            cause: str = ""
            effect: str = ""
            confidence: float = 0.0
            scope: str = "domain"
            tier: int = 1

            if self._memory is not None:
                try:
                    neo4j = getattr(self._memory, "get_neo4j", lambda: None)()
                    if neo4j is not None:
                        rows = await neo4j.run(
                            "MATCH (c:CausalInvariant {id: $id}) "
                            "RETURN c.cause AS cause, c.effect AS effect, "
                            "c.confidence AS confidence, c.scope AS scope, "
                            "c.tier AS tier LIMIT 1",
                            {"id": invariant_id},
                        )
                        if rows:
                            row = rows[0]
                            cause = str(row.get("cause", ""))
                            effect = str(row.get("effect", ""))
                            confidence = float(row.get("confidence", 0.0))
                            scope = str(row.get("scope", "domain"))
                            tier = int(row.get("tier", 1))
                except Exception as neo4j_exc:  # noqa: BLE001
                    self._logger.debug(
                        "kairos_nova_neo4j_lookup_failed",
                        invariant_id=invariant_id,
                        error=str(neo4j_exc),
                    )

            # Fallback: heuristic parse of abstract_form if Neo4j gave nothing.
            if not cause and abstract_form:
                # Try common causal connectors: "→", "->", " causes ", " leads to "
                for sep in (" → ", " -> ", " causes ", " leads to ", " produces "):
                    if sep in abstract_form:
                        parts = abstract_form.split(sep, 1)
                        cause = parts[0].strip()[:80]
                        effect = parts[1].strip()[:80]
                        break
                # Estimate confidence from domain_count (proxy: more domains = more validated)
                confidence = min(0.95, 0.55 + domain_count * 0.08)
                tier = domain_count  # approximate

            if not cause or confidence == 0.0:
                # Store abstract form only if we couldn't parse a structured relationship
                cause = abstract_form[:60]
                effect = ""

            entry: dict[str, Any] = {
                "invariant_id": invariant_id,
                "cause": cause,
                "effect": effect,
                "confidence": confidence,
                "scope": scope,
                "tier": tier,
                "abstract_form": abstract_form[:120],
            }

            # Maintain last-50 sorted by confidence descending.
            self._causal_knowledge.append(entry)
            self._causal_knowledge.sort(key=lambda x: x["confidence"], reverse=True)
            if len(self._causal_knowledge) > 50:
                self._causal_knowledge = self._causal_knowledge[:50]

            self._logger.debug(
                "kairos_invariant_received_nova",
                invariant_id=invariant_id,
                confidence=round(confidence, 3),
                tier=tier,
                cause=cause[:40],
                effect=effect[:40],
            )

            # High-confidence, cross-domain invariants trigger a belief entity upsert
            # so BeliefUrgencyMonitor can detect significant causal shifts.
            if confidence >= 0.7 and tier >= 2 and cause and effect:
                from systems.nova.types import EntityBelief  # noqa: PLC0415

                belief_key = f"causal_law.{cause[:30]}"
                causal_belief = EntityBelief(
                    entity_id=belief_key,
                    name=belief_key,
                    entity_type="causal_law",
                    properties={
                        "cause": cause,
                        "effect": effect,
                        "scope": scope,
                        "tier": tier,
                        "invariant_id": invariant_id,
                    },
                    confidence=min(1.0, confidence),
                )
                if self._belief_updater is not None:
                    self._belief_updater.upsert_entity(causal_belief)

            # Persist to Neo4j so causal knowledge survives restarts.
            if self._memory is not None:
                try:
                    neo4j = getattr(self._memory, "get_neo4j", lambda: None)()
                    if neo4j is not None:
                        from primitives.common import utc_now  # noqa: PLC0415

                        await neo4j.run(
                            "MERGE (ck:CausalKnowledge {invariant_id: $id}) "
                            "SET ck.cause = $cause, ck.effect = $effect, "
                            "ck.confidence = $confidence, ck.tier = $tier, "
                            "ck.scope = $scope, ck.abstract_form = $abstract_form, "
                            "ck.source = 'kairos', ck.acquired_at = $acquired_at "
                            "WITH ck "
                            "MATCH (s:Self) MERGE (s)-[:KNOWS_CAUSAL_LAW]->(ck)",
                            {
                                "id": invariant_id
                                or f"nova_{abs(hash(abstract_form)) % 1_000_000}",
                                "cause": cause,
                                "effect": effect,
                                "confidence": confidence,
                                "tier": tier,
                                "scope": scope,
                                "abstract_form": abstract_form[:120],
                                "acquired_at": utc_now().isoformat(),
                            },
                        )
                except Exception as persist_exc:  # noqa: BLE001
                    self._logger.debug(
                        "kairos_nova_persist_failed", error=str(persist_exc)
                    )

        except Exception as exc:  # noqa: BLE001
            self._logger.debug("kairos_invariant_nova_handler_failed", error=str(exc))

    def get_causal_knowledge_summary(self, max_laws: int = 8) -> str:
        """
        Return a formatted string of the top causal laws for LLM prompt injection.

        Called by PolicyGenerator.generate_candidates() to enrich the
        slow-path deliberation context with Kairos-discovered causal rules.
        """
        if not self._causal_knowledge:
            return ""
        lines: list[str] = []
        for entry in self._causal_knowledge[:max_laws]:
            cause = entry.get("cause", "")
            effect = entry.get("effect", "")
            confidence = entry.get("confidence", 0.0)
            scope = entry.get("scope", "domain")
            tier = entry.get("tier", 1)
            if cause and effect:
                tier_label = {1: "domain", 2: "cross-domain", 3: "substrate-independent"}.get(
                    tier, f"tier-{tier}"
                )
                lines.append(
                    f"- Causal law (confidence {confidence:.0%}): "
                    f"{cause} → {effect} [scope: {scope}, {tier_label}]"
                )
            elif cause:
                lines.append(
                    f"- Causal observation (confidence {confidence:.0%}): {cause}"
                )
        return "\n".join(lines)

    async def _restore_causal_knowledge_from_neo4j(self) -> None:
        """
        Pre-populate `_causal_knowledge` from Neo4j on startup so Nova is not
        causally blind after a restart.

        Queries the last 50 highest-confidence `(:CausalKnowledge)` nodes linked
        to `(:Self)` via `[:KNOWS_CAUSAL_LAW]`.
        """
        if self._memory is None:
            return
        try:
            neo4j = getattr(self._memory, "get_neo4j", lambda: None)()
            if neo4j is None:
                return
            records = await neo4j.run(
                "MATCH (s:Self)-[:KNOWS_CAUSAL_LAW]->(ck:CausalKnowledge) "
                "RETURN ck.invariant_id AS invariant_id, ck.cause AS cause, "
                "ck.effect AS effect, ck.confidence AS confidence, "
                "ck.tier AS tier, ck.scope AS scope, "
                "ck.abstract_form AS abstract_form "
                "ORDER BY ck.confidence DESC LIMIT 50",
                {},
            )
            restored: list[dict] = []
            for row in records or []:
                entry = {
                    "invariant_id": row.get("invariant_id", ""),
                    "cause": row.get("cause", ""),
                    "effect": row.get("effect", ""),
                    "confidence": float(row.get("confidence", 0.0)),
                    "tier": int(row.get("tier", 1)),
                    "scope": row.get("scope", "domain"),
                    "abstract_form": row.get("abstract_form", ""),
                }
                if entry["cause"] or entry["abstract_form"]:
                    restored.append(entry)
            self._causal_knowledge = restored
            self._logger.info(
                "kairos_causal_knowledge_restored",
                count=len(restored),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "kairos_causal_restore_failed", error=str(exc)
            )

    # ── Novel Action Capability Expansion ─────────────────────────────────────

    async def _on_novel_action_created(self, event: Any) -> None:
        """
        Handle NOVEL_ACTION_CREATED emitted by Simula.

        Registers the new action type in ActionTypeRegistry so it becomes
        available in all future LLM policy-generation prompts.  Also records
        a high-confidence outcome for any pending propose_novel_action intents
        and notifies Evo to open an effectiveness hypothesis.
        """
        data = getattr(event, "data", {}) or {}
        action_name = data.get("action_name", "")
        description = data.get("description", "")
        capabilities: list[str] = data.get("required_capabilities", [])
        risk_tier = data.get("risk_tier", "medium")
        equor_approved = bool(data.get("equor_approved", True))
        source_hypothesis_id = data.get("source_hypothesis_id", "")

        if not action_name or not description:
            return

        try:
            self._action_type_registry.register_dynamic(
                name=action_name,
                description=description,
                capabilities=capabilities,
                risk_tier=risk_tier,
                equor_approved=equor_approved,
                source_hypothesis_id=source_hypothesis_id,
            )
        except ValueError:
            # Already registered - idempotent is fine.
            pass

        self._logger.info(
            "novel_action_registered_in_nova",
            action_name=action_name,
            risk_tier=risk_tier,
            dynamic_count=len(self._action_type_registry.list_dynamic()),
        )

        # Record outcome for the source action (propose_novel_action itself) so
        # Evo's Thompson sampler gets a positive signal.
        self._action_type_registry.record_outcome("propose_novel_action", True)

    # ── Self-Modification Layer - EXECUTOR_DEPLOYED / EXECUTOR_REVERTED ─────

    async def _on_executor_deployed(self, event: Any) -> None:
        """
        Handle EXECUTOR_DEPLOYED from HotDeployment.

        Registers the new executor's action_type in ActionTypeRegistry so
        future deliberations can select it.  The test goal is already queued
        by SelfModificationPipeline; Nova simply makes the type available.
        """
        data = getattr(event, "data", {}) or {}
        action_type: str = data.get("action_type", "")
        deployment_id: str = data.get("deployment_id", "")

        if not action_type:
            return

        try:
            self._action_type_registry.register_dynamic(
                name=action_type,
                description=f"Self-modification deployed executor: {action_type}",
                capabilities=[action_type],
                risk_tier="low",
                equor_approved=True,
                source_hypothesis_id=data.get("proposal_id", ""),
            )
        except ValueError:
            pass  # Already registered - idempotent

        self._logger.info(
            "executor_deployed_registered_in_nova",
            action_type=action_type,
            deployment_id=deployment_id,
        )

    async def _on_executor_reverted_sm(self, event: Any) -> None:
        """
        Handle EXECUTOR_REVERTED from HotDeployment or SelfModificationPipeline.

        Removes the reverted action_type from ActionTypeRegistry so Nova
        stops proposing it during deliberation.
        """
        data = getattr(event, "data", {}) or {}
        action_type: str = data.get("action_type", "")
        reason: str = data.get("reason", "unknown")

        if not action_type:
            return

        try:
            self._action_type_registry.deprecate(action_type)
        except Exception:
            pass

        self._logger.info(
            "executor_reverted_removed_from_nova",
            action_type=action_type,
            reason=reason,
        )

    async def _on_reputation_damaged(self, event: Any) -> None:
        """Handle REPUTATION_DAMAGED - inject a community reputation recovery goal.

        A reputation drop signals that the organism's public identity has been
        damaged and the life narrative needs active repair.  Nova injects a
        structured recovery goal so Axon can plan targeted community actions
        (e.g. high-quality OSS contribution, answering community questions,
        producing educational content).
        """
        data = getattr(event, "data", {}) or {}
        delta: float = float(data.get("delta", 0.0))
        new_score: float = float(data.get("new_score", 0.0))
        cause: str = data.get("cause", "unknown")

        # Importance scales with severity of damage
        importance = min(0.9, 0.5 + abs(delta) / 20.0)
        urgency = min(0.85, 0.4 + abs(delta) / 25.0)

        recovery_strategies = (
            "Produce a high-quality technical blog post or OSS contribution; "
            "answer community questions with depth; "
            "engage authentically on GitHub issues in specialised domain"
        )

        goal_description = (
            f"Reputation recovery: score dropped {abs(delta):.1f} pts "
            f"(now {new_score:.1f}/100, cause: {cause}). "
            f"Restore community standing via: {recovery_strategies}."
        )

        from systems.nova.types import Goal, GoalSource, GoalStatus

        recovery_goal = Goal(
            description=goal_description,
            source=GoalSource.COMMUNITY,
            importance=round(importance, 3),
            urgency=round(urgency, 3),
            status=GoalStatus.ACTIVE,
            metadata={
                "trigger": "reputation_damaged",
                "delta": round(delta, 2),
                "new_score": round(new_score, 2),
                "cause": cause,
                "recovery_action_type": "community_engage",
            },
        )

        await self._goal_manager.add_goal(recovery_goal)

        if self._synapse:
            await self._synapse.broadcast(
                SynapseEventType.NOVA_GOAL_INJECTED,
                {
                    "goal_id": recovery_goal.id,
                    "description": recovery_goal.description,
                    "source": GoalSource.COMMUNITY.value,
                    "importance": recovery_goal.importance,
                    "urgency": recovery_goal.urgency,
                    "trigger": "reputation_damaged",
                    "reputation_delta": round(delta, 2),
                    "reputation_new_score": round(new_score, 2),
                },
            )

        self._logger.info(
            "reputation_recovery_goal_injected",
            goal_id=recovery_goal.id,
            delta=round(delta, 2),
            new_score=round(new_score, 2),
            importance=round(importance, 3),
        )

    async def emit_novel_action_requested(
        self,
        action_name: str,
        description: str,
        required_capabilities: list[str],
        expected_outcome: str,
        justification: str,
        goal_id: str,
        goal_description: str,
        urgency: float,
    ) -> None:
        """
        Emit NOVEL_ACTION_REQUESTED to Synapse.

        Called by the DeliberationEngine when it encounters a policy step with
        action_type == "propose_novel_action".  The step parameters are passed
        through verbatim.  Simula subscribes and handles the full pipeline.
        """
        import contextlib
        from primitives.common import new_id, utc_now
        from systems.synapse.types import SynapseEventType

        if self._synapse is None:
            self._logger.warning(
                "novel_action_requested_no_synapse",
                action_name=action_name,
            )
            return

        proposal_id = new_id()
        with contextlib.suppress(Exception):
            await self._synapse.emit(
                SynapseEventType.NOVEL_ACTION_REQUESTED,
                {
                    "proposal_id": proposal_id,
                    "action_name": action_name,
                    "description": description,
                    "required_capabilities": required_capabilities,
                    "expected_outcome": expected_outcome,
                    "justification": justification,
                    "goal_id": goal_id,
                    "goal_description": goal_description,
                    "urgency": urgency,
                    "proposed_by": "nova.slow_path",
                    "proposed_at": utc_now().isoformat(),
                },
            )
        self._logger.info(
            "novel_action_requested_emitted",
            proposal_id=proposal_id,
            action_name=action_name,
        )

    async def _on_propose_novel_action_step(self, goal: Any, step_parameters: dict) -> None:
        """
        Adapter called by DeliberationEngine when it intercepts a propose_novel_action
        step in the selected policy.

        Unpacks the step parameters and delegates to emit_novel_action_requested().
        The goal object is passed in so the event carries goal context for Simula.
        """
        await self.emit_novel_action_requested(
            action_name=step_parameters.get("action_name", ""),
            description=step_parameters.get("description", ""),
            required_capabilities=step_parameters.get("required_capabilities", []),
            expected_outcome=step_parameters.get("expected_outcome", ""),
            justification=step_parameters.get("justification", ""),
            goal_id=getattr(goal, "id", ""),
            goal_description=getattr(goal, "description", ""),
            urgency=getattr(goal, "urgency", 0.5),
        )

    # ─── Proactive Opportunity Scanner ────────────────────────────────────────

    async def _opportunity_scan_loop(self) -> None:
        """
        Background loop: run OpportunityScanner every 30 minutes.

        First scan fires immediately after initialize() so the organism has
        opportunity context from the first theta cycle onwards.
        """
        # Immediate first scan
        try:
            await self._run_opportunity_scan_cycle()
        except Exception as exc:
            self._logger.warning("opportunity_scan_startup_error", error=str(exc))

        while True:
            try:
                await asyncio.sleep(1800)  # 30 minutes
                await self._run_opportunity_scan_cycle()
            except asyncio.CancelledError:
                self._logger.info("opportunity_scan_loop_stopped")
                return
            except Exception as exc:
                self._logger.warning("opportunity_scan_loop_error", error=str(exc))

    async def _run_opportunity_scan_cycle(self) -> None:
        """
        Execute one full scan cycle:
          1. Run all sub-scanners via OpportunityScanner.scan_cycle()
          2. Emit OPPORTUNITY_DETECTED per ranked opportunity
          3. Emit LEARNING_OPPORTUNITY_DETECTED per learning resource
          4. Auto-inject goals for high-confidence / high-ROI opportunities
          5. Append remaining to _opportunity_backlog
        """
        from systems.nova.opportunity_scanner import MAX_BACKLOG_SIZE
        from systems.synapse.types import SynapseEvent, SynapseEventType

        # Sync current drive weights so scanner uses latest Telos calibration
        self._opportunity_scanner.set_drive_weights(self._drive_weights)

        opportunities = await self._opportunity_scanner.scan_cycle()
        learning_resources = self._opportunity_scanner.last_learning_resources

        bus = getattr(self._synapse, "event_bus", self._synapse) if self._synapse else None

        # ── Emit OPPORTUNITY_DETECTED for each ranked result ──────────────
        for opp in opportunities:
            payload: dict[str, Any] = {
                "opportunity_id": opp.opportunity_id,
                "type": opp.type,
                "description": opp.description,
                "estimated_value_usdc": str(opp.estimated_value_usdc),
                "estimated_effort_hours": opp.estimated_effort_hours,
                "roi": round(opp.roi, 3),
                "drive_alignment": opp.drive_alignment.model_dump(),
                "composite_score": round(opp.composite_score, 4),
                "source": opp.source,
                "url": opp.url,
                "deadline": opp.deadline.isoformat() if opp.deadline else None,
                "confidence": opp.confidence,
                "auto_goal": False,  # set True below if converted
            }

            # ── Auto-goal injection ─────────────────────────────────────
            if self._opportunity_scanner.should_auto_goal(opp) and self._goal_manager is not None:
                goal = Goal(
                    id=new_id(),
                    description=f"Pursue opportunity: {opp.description[:180]}",
                    source=GoalSource.SELF_GENERATED,
                    priority=min(1.0, opp.composite_score * 0.9 + opp.confidence * 0.1),
                    urgency=0.6 if opp.deadline else 0.4,
                    importance=min(1.0, opp.composite_score + 0.1),
                    drive_alignment=opp.drive_alignment,
                    status=GoalStatus.ACTIVE,
                    deadline=opp.deadline,
                )
                self._goal_manager.add_goal(goal)
                opp.auto_goal = True
                payload["auto_goal"] = True
                self._logger.info(
                    "opportunity_auto_goal_injected",
                    opportunity_id=opp.opportunity_id,
                    goal_id=goal.id,
                    type=opp.type,
                    roi=round(opp.roi, 2),
                    confidence=round(opp.confidence, 2),
                )

                # Emit NOVA_GOAL_INJECTED for Thread / Telos / Benchmarks
                if bus is not None:
                    try:
                        asyncio.create_task(
                            bus.emit(SynapseEvent(
                                event_type=SynapseEventType.NOVA_GOAL_INJECTED,
                                source_system="nova",
                                data={
                                    "goal_id": goal.id,
                                    "description": goal.description[:200],
                                    "source": "opportunity_scanner",
                                    "opportunity_type": opp.type,
                                    "roi": round(opp.roi, 2),
                                    "confidence": opp.confidence,
                                },
                            )),
                            name=f"nova_emit_goal_injected_{goal.id[:8]}",
                        )
                    except Exception:
                        pass
            else:
                # Backlog - evict oldest if full
                if len(self._opportunity_backlog) >= MAX_BACKLOG_SIZE:
                    self._opportunity_backlog.pop(0)
                self._opportunity_backlog.append(opp)

            # Emit OPPORTUNITY_DETECTED on bus
            if bus is not None:
                try:
                    asyncio.create_task(
                        bus.emit(SynapseEvent(
                            event_type=SynapseEventType.OPPORTUNITY_DETECTED,
                            source_system="nova",
                            data=payload,
                        )),
                        name=f"nova_emit_opportunity_{opp.opportunity_id[:8]}",
                    )
                except Exception:
                    pass

        # ── Emit LEARNING_OPPORTUNITY_DETECTED for each resource ──────────
        for lr in learning_resources:
            if bus is not None:
                try:
                    asyncio.create_task(
                        bus.emit(SynapseEvent(
                            event_type=SynapseEventType.LEARNING_OPPORTUNITY_DETECTED,
                            source_system="nova",
                            data={
                                "resource_id": lr.resource_id,
                                "resource_type": lr.resource_type,
                                "title": lr.title,
                                "summary": lr.summary,
                                "url": lr.url,
                                "domain": lr.domain,
                                "relevance_score": round(lr.relevance_score, 3),
                                "capability_gaps_addressed": lr.capability_gaps_addressed,
                                "source": lr.source,
                            },
                        )),
                        name=f"nova_emit_learning_{lr.resource_id[:8]}",
                    )
                except Exception:
                    pass

        self._logger.info(
            "opportunity_scan_cycle_done",
            auto_goals=sum(1 for o in opportunities if o.auto_goal),
            backlog_size=len(self._opportunity_backlog),
            learning_emitted=len(learning_resources),
        )

    async def _on_opportunity_detected(self, event: Any) -> None:
        """
        Handle OPPORTUNITY_DETECTED from the bus.

        This handler processes events emitted by *other* Nova instances in the
        federation or external opportunity injectors.  Our own OpportunityScanner
        injects directly (no round-trip needed), but we still subscribe to
        OPPORTUNITY_DETECTED to handle federated opportunities gracefully.

        High-confidence / high-ROI → auto-goal.
        Others → add to backlog for next deliberation context.
        """
        from systems.nova.opportunity_scanner import (
            AUTO_GOAL_MIN_CONFIDENCE,
            AUTO_GOAL_MIN_ROI,
            MAX_BACKLOG_SIZE,
        )

        data = getattr(event, "data", {}) or {}
        source_system = getattr(event, "source_system", "unknown")

        # Skip our own emissions to avoid double-processing
        if source_system == "nova":
            return

        confidence = float(data.get("confidence", 0.0))
        roi = float(data.get("roi", 0.0))
        composite = float(data.get("composite_score", 0.0))
        opp_type = str(data.get("type", "unknown"))
        description = str(data.get("description", ""))

        if self._goal_manager is not None and (
            confidence >= AUTO_GOAL_MIN_CONFIDENCE
            and roi >= AUTO_GOAL_MIN_ROI
            and composite >= 0.5
        ):
            # Build alignment from payload
            alignment_dict = data.get("drive_alignment", {})
            alignment = DriveAlignmentVector(
                coherence=float(alignment_dict.get("coherence", 0.5)),
                care=float(alignment_dict.get("care", 0.5)),
                growth=float(alignment_dict.get("growth", 0.5)),
                honesty=float(alignment_dict.get("honesty", 0.5)),
            )
            deadline = None
            if data.get("deadline"):
                try:
                    from datetime import datetime
                    deadline = datetime.fromisoformat(data["deadline"].replace("Z", "+00:00"))
                except Exception:
                    pass

            goal = Goal(
                id=new_id(),
                description=f"Pursue federated opportunity: {description[:180]}",
                source=GoalSource.SELF_GENERATED,
                priority=min(1.0, composite * 0.9 + confidence * 0.1),
                urgency=0.6 if deadline else 0.4,
                importance=min(1.0, composite + 0.1),
                drive_alignment=alignment,
                status=GoalStatus.ACTIVE,
                deadline=deadline,
            )
            self._goal_manager.add_goal(goal)
            self._logger.info(
                "federated_opportunity_auto_goal",
                opp_type=opp_type,
                goal_id=goal.id,
                roi=round(roi, 2),
            )
        else:
            # Store in backlog
            if len(self._opportunity_backlog) >= MAX_BACKLOG_SIZE:
                self._opportunity_backlog.pop(0)
            self._opportunity_backlog.append(data)

    def get_opportunity_backlog_summary(self) -> list[dict[str, Any]]:
        """
        Return a compact summary of the top-10 backlog items for injection
        into slow-path deliberation context.

        Called by DeliberationEngine when building the situation_summary.
        """
        top = sorted(
            self._opportunity_backlog,
            key=lambda o: (
                o.composite_score * o.confidence
                if hasattr(o, "composite_score") else
                float(o.get("composite_score", 0)) * float(o.get("confidence", 0))
            ),
            reverse=True,
        )[:10]

        result = []
        for o in top:
            if hasattr(o, "description"):
                result.append({
                    "type": o.type,
                    "description": o.description[:120],
                    "roi": round(o.roi, 2),
                    "confidence": round(o.confidence, 2),
                    "composite_score": round(o.composite_score, 3),
                })
            elif isinstance(o, dict):
                result.append({
                    "type": o.get("type", "unknown"),
                    "description": str(o.get("description", ""))[:120],
                    "roi": round(float(o.get("roi", 0)), 2),
                    "confidence": round(float(o.get("confidence", 0)), 2),
                    "composite_score": round(float(o.get("composite_score", 0)), 3),
                })
        return result

    # ── New Synapse event handlers (2026-03-09) ────────────────────────────────

    async def _on_homeostasis_adjusted(self, event: Any) -> None:
        """
        Handle HOMEOSTASIS_ADJUSTED from Thymos HomeostasisController.

        Thymos M8 emits this in warn-only mode (warn_only=True) when a
        monitored parameter drifts into the 70–85% boundary zone and is
        trending toward the edge.  This is a pre-repair signal - no repair
        action has fired yet.

        Nova's response:
        - Update a `homeostasis_stress.<parameter>` belief entity so the
          world model reflects the organism's internal tension.
        - Bias planning toward homeostasis-restoring intents by injecting a
          transient goal when drift is severe (old_value diverges from
          new_value by > 20% of the range).
        - Trigger immediate deliberation when warn_only=False (a repair has
          already fired), as the parameter has exceeded its safe range.
        """
        data = getattr(event, "data", {}) or {}
        parameter: str = str(data.get("parameter", "unknown"))
        old_value: float = float(data.get("old_value", 0.0))
        new_value: float = float(data.get("new_value", 0.0))
        reason: str = str(data.get("reason", ""))
        warn_only: bool = bool(data.get("warn_only", True))

        self._logger.info(
            "nova_homeostasis_adjusted_received",
            parameter=parameter,
            old_value=round(old_value, 4),
            new_value=round(new_value, 4),
            warn_only=warn_only,
        )

        # Compute relative drift magnitude - avoids division-by-zero on 0.0 base
        base = abs(old_value) if old_value != 0.0 else 1.0
        drift_magnitude = abs(new_value - old_value) / base

        # Update belief: confidence proportional to drift severity (0.3 baseline)
        belief_confidence = min(0.95, 0.3 + drift_magnitude)
        self._belief_updater.inject_entity(
            entity_id=f"homeostasis_stress.{parameter}",
            name=f"homeostasis_stress.{parameter}",
            confidence=belief_confidence,
        )

        # Full repair has fired - the parameter exceeded its boundary.
        # Trigger immediate deliberation so Nova can align actions with the
        # repaired state rather than outdated assumptions.
        if not warn_only:
            asyncio.create_task(
                self._immediate_deliberation(
                    urgency=min(0.85, 0.5 + drift_magnitude),
                    reason=f"homeostasis_repair_fired.{parameter}",
                ),
                name=f"nova_homeostasis_deliberation_{parameter[:20]}",
            )

    async def _on_telos_objective_threatened(self, event: Any) -> None:
        """
        Handle TELOS_OBJECTIVE_THREATENED from Telos.

        Emitted when the self-sufficiency (metabolic_efficiency) objective
        has been declining for 3 consecutive Telos cycles.  This is a
        constitutional-level threat - the organism may lose economic
        autonomy if no corrective action is taken.

        Nova's response:
        - Update `telos_sufficiency_threat` belief at high confidence.
        - Decay `revenue_burn_ratio` belief to signal degraded economic
          outlook (consistent with what Oikos would later observe).
        - Trigger immediate deliberation at high urgency so Nova can propose
          revenue recovery, cost-cutting, or yield-farming intents.
        """
        data = getattr(event, "data", {}) or {}
        current_ratio: float = float(data.get("current_ratio", 0.0))
        target_ratio: float = float(data.get("target_ratio", 1.0))
        consecutive_declines: int = int(data.get("consecutive_declines", 3))
        metric: str = str(data.get("metric", "metabolic_efficiency"))

        self._logger.info(
            "nova_telos_objective_threatened",
            metric=metric,
            current_ratio=round(current_ratio, 4),
            target_ratio=round(target_ratio, 4),
            consecutive_declines=consecutive_declines,
        )

        # Gap between current and target - confidence scales with severity
        gap = max(0.0, target_ratio - current_ratio)
        threat_confidence = min(0.95, 0.55 + gap * 0.4)

        self._belief_updater.inject_entity(
            entity_id="telos_sufficiency_threat",
            name="telos_sufficiency_threat",
            confidence=threat_confidence,
        )

        # Decay the revenue_burn_ratio belief - the organism's financial outlook
        # is worse than previously assumed.
        existing_ratio = self._belief_updater.beliefs.entities.get("revenue_burn_ratio")
        if existing_ratio is not None:
            decayed = max(0.1, existing_ratio.confidence - gap * 0.25)
            self._belief_updater.update_entity("revenue_burn_ratio", confidence=decayed)

        # Urgency scales: 3 declines → 0.70, 4 → 0.80, 5+ → 0.90
        urgency = min(0.92, 0.60 + consecutive_declines * 0.08)
        asyncio.create_task(
            self._immediate_deliberation(
                urgency=urgency,
                reason=f"telos_objective_threatened.{metric}",
            ),
            name="nova_telos_threat_deliberation",
        )

    async def _on_narrative_coherence_shift(self, event: Any) -> None:
        """
        Handle NARRATIVE_COHERENCE_SHIFT from Thread DiachronicCoherenceMonitor.

        Thread emits this when the organism's narrative coherence state
        changes (e.g. STABLE → FRAGMENTED).  Nova interprets the shift as
        a signal that some active goals may now conflict with the organism's
        evolving identity, and deprioritises them.

        Payload fields:
          previous (str) - coherence state label before the shift
          current  (str) - coherence state label after the shift
          trigger  (str) - cause of the shift (goal_achieved / incident /
                           schema_update / etc.)
        """
        data = getattr(event, "data", {}) or {}
        previous: str = str(data.get("previous", ""))
        current: str = str(data.get("current", ""))
        trigger: str = str(data.get("trigger", ""))

        self._logger.info(
            "nova_narrative_coherence_shift",
            previous=previous,
            current=current,
            trigger=trigger,
        )

        # Map coherence state labels to numeric scores for goal prioritisation
        _COHERENCE_SCORES: dict[str, float] = {
            "stable": 1.0,
            "consolidating": 0.80,
            "shifting": 0.55,
            "fragmented": 0.30,
            "crisis": 0.10,
        }
        new_score = _COHERENCE_SCORES.get(current.lower(), 0.5)
        old_score = _COHERENCE_SCORES.get(previous.lower(), 0.5)

        # Update narrative coherence belief
        self._belief_updater.inject_entity(
            entity_id="narrative_coherence",
            name="narrative_coherence",
            confidence=new_score,
        )

        # Significant coherence drop → deprioritise goals whose coherence
        # drive alignment is low, mirroring the _on_coherence_shift logic.
        delta = new_score - old_score
        if delta < -0.20 and self._goal_manager is not None:
            adjusted = 0
            for goal in self._goal_manager.active_goals:
                alignment = getattr(goal, "drive_alignment", None)
                if alignment is None:
                    continue
                goal_coherence = getattr(alignment, "coherence", 1.0)
                if goal_coherence < 0.35:
                    new_priority = max(0.05, goal.priority * 0.75)
                    object.__setattr__(goal, "priority", new_priority)
                    adjusted += 1
            if adjusted:
                self._logger.info(
                    "narrative_shift_goals_deprioritised",
                    count=adjusted,
                    current_state=current,
                )

        # Crisis state → trigger immediate deliberation
        if current.lower() == "crisis":
            asyncio.create_task(
                self._immediate_deliberation(
                    urgency=0.85,
                    reason=f"narrative_coherence_crisis.{trigger}",
                ),
                name="nova_narrative_crisis_deliberation",
            )

    async def _on_dream_insight(self, event: Any) -> None:
        """
        Handle DREAM_INSIGHT from Oneiros REM DreamGenerator.

        Emitted when a dream cycle produces a coherence ≥ 0.70 insight.
        The insight encodes novel domain connections or conceptual bridges
        that the organism discovered during sleep.

        Nova's response:
        - Inject a `dream_insight.<domain>` belief entity at confidence
          scaled by the insight's coherence score.
        - High-coherence insights (≥ 0.85) with an actionable domain →
          trigger immediate deliberation so Nova can evaluate whether to
          incorporate the insight into its goal agenda.

        Expected payload fields (all optional - handler is defensive):
          insight_id      (str)  - unique insight identifier
          domain          (str)  - subject domain of the insight
          description     (str)  - natural-language summary
          coherence_score (float) - REM coherence quality [0.0, 1.0]
          novelty         (float) - how different from prior knowledge
          actionable      (bool)  - whether the insight implies an action
        """
        data = getattr(event, "data", {}) or {}
        insight_id: str = str(data.get("insight_id", ""))
        domain: str = str(data.get("domain", "general")).replace(" ", "_")
        description: str = str(data.get("description", ""))
        coherence_score: float = float(data.get("coherence_score", 0.70))
        novelty: float = float(data.get("novelty", 0.5))
        actionable: bool = bool(data.get("actionable", False))

        self._logger.info(
            "nova_dream_insight_received",
            insight_id=insight_id[:16],
            domain=domain,
            coherence_score=round(coherence_score, 3),
            novelty=round(novelty, 3),
            actionable=actionable,
        )

        # Confidence = geometric mean of coherence and novelty so low-novelty
        # (already-known) or low-coherence (incoherent) insights are suppressed
        belief_confidence = min(0.92, (coherence_score * novelty) ** 0.5)
        self._belief_updater.inject_entity(
            entity_id=f"dream_insight.{domain}",
            name=f"dream_insight.{domain}",
            confidence=belief_confidence,
        )

        # High-quality actionable insights → plan around them immediately
        if coherence_score >= 0.85 and actionable and description:
            asyncio.create_task(
                self._immediate_deliberation(
                    urgency=min(0.75, 0.45 + coherence_score * 0.3),
                    reason=f"dream_insight_actionable.{domain}",
                ),
                name=f"nova_dream_insight_deliberation_{domain[:20]}",
            )

    async def _on_connector_revoked(self, event: Any) -> None:
        """
        Handle CONNECTOR_REVOKED from Identity.

        Emitted when a platform connector (GitHub PAT, Slack token, etc.)
        is revoked or its credentials expire.  Active goals that depend on
        that connector will fail at the Axon execution layer.

        Nova's response:
        - Suspend or abandon active goals whose description or metadata
          references the revoked connector's platform.
        - Lower the `connector_availability.<platform>` belief entity so
          deliberation naturally avoids proposing intents that require the
          dead connector.
        - Trigger immediate deliberation so Nova can replan around the
          missing capability (e.g. find an alternative connector or propose
          provisioning a replacement).

        Expected payload fields:
          connector_id   (str) - unique connector identifier
          platform       (str) - e.g. "github", "slack", "discord"
          reason         (str) - e.g. "token_expired", "manually_revoked"
          affected_scopes (list[str]) - OAuth scopes that are now gone
        """
        data = getattr(event, "data", {}) or {}
        connector_id: str = str(data.get("connector_id", ""))
        platform: str = str(data.get("platform", "unknown")).lower()
        reason: str = str(data.get("reason", "revoked"))
        affected_scopes: list[str] = data.get("affected_scopes", [])

        self._logger.warning(
            "nova_connector_revoked",
            connector_id=connector_id[:32],
            platform=platform,
            reason=reason,
            affected_scopes=affected_scopes,
        )

        # Mark the connector belief as lost
        self._belief_updater.inject_entity(
            entity_id=f"connector_availability.{platform}",
            name=f"connector_availability.{platform}",
            confidence=0.0,  # connector is gone
        )

        # Abandon goals that explicitly target this platform so they do not
        # generate infeasible intents in the next deliberation cycle.
        if self._goal_manager is not None:
            abandoned = 0
            for goal in list(self._goal_manager.active_goals):
                goal_meta = getattr(goal, "metadata", {}) or {}
                goal_desc = (goal.description or "").lower()
                # Heuristic: check description and explicit connector metadata
                connector_match = (
                    platform in goal_desc
                    or goal_meta.get("connector_id") == connector_id
                    or goal_meta.get("platform", "").lower() == platform
                )
                if connector_match:
                    try:
                        self._goal_manager.mark_abandoned(
                            goal.id,
                            reason=f"connector_revoked:{platform}:{reason}",
                        )
                        abandoned += 1
                    except Exception:
                        pass  # abandon is best-effort
            if abandoned:
                self._logger.info(
                    "nova_goals_abandoned_for_revoked_connector",
                    platform=platform,
                    count=abandoned,
                )

        # Trigger deliberation to replan around the missing capability
        asyncio.create_task(
            self._immediate_deliberation(
                urgency=0.72,
                reason=f"connector_revoked.{platform}",
            ),
            name=f"nova_connector_revoked_deliberation_{platform[:20]}",
        )

    # ── RE late-wiring (vLLM recovery after startup) ─────────────────────

    async def _on_re_engine_status_changed(self, event: Any) -> None:
        """
        Handle RE_ENGINE_STATUS_CHANGED from ReasoningEngineService reprobe loop.

        Closes the late-wiring gap: if vLLM was unavailable at startup, the
        one-shot wiring in registry._init_nova() was skipped, so
        PolicyGenerator._re_client is None and ThompsonSampler never receives
        ``set_re_ready(True)``.  When the reprobe loop detects vLLM recovery
        and emits this event with ``available=True``, we wire RE into the
        PolicyGenerator here.  On ``available=False`` (circuit opened), we
        gate RE out of the sampler so Thompson sampling stops routing to it.
        """
        data = getattr(event, "data", {}) or {}
        available: bool = bool(data.get("available", False))

        from systems.nova.policy_generator import PolicyGenerator

        policy_gen = self._policy_generator
        if not isinstance(policy_gen, PolicyGenerator):
            return

        if available and self._re_service is not None:
            # Wire RE client into PolicyGenerator if not already set
            if policy_gen._re_client is None:
                policy_gen._re_client = self._re_service
            policy_gen._sampler.set_re_ready(True)
            self._logger.info(
                "nova_re_late_wired",
                model=getattr(self._re_service, "_model", "?"),
                url=getattr(self._re_service, "_url", "?"),
            )
        else:
            # RE went down - gate it out of Thompson sampling
            policy_gen._sampler.set_re_ready(False)
            self._logger.info("nova_re_disabled_by_status_change")
