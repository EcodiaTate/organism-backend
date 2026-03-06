"""
EcodiaOS — Nova Service

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
  initialize() — loads constitution and drive weights, builds sub-components
  receive_broadcast() — implements BroadcastSubscriber for Atune
  submit_intent() — external API for direct intent submission (test/governance)
  process_outcome() — feedback loop from execution
  shutdown() — graceful teardown
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.affect import AffectState
from systems.atune.types import ActiveGoalSummary, WorkspaceBroadcast
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
    from systems.memory.service import MemoryService
    from systems.voxis.service import VoxisService

logger = structlog.get_logger()

# Executor action types that represent heavy external work.  When any pending
# intent contains one of these, the heartbeat considers the organism "busy" and
# skips the hunger-drive cycle.  Lightweight internal executors (store_insight,
# observe, query_memory …) must NOT appear here — blocking on them causes the
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
        memory  — for constitution, self-model retrieval, procedure lookup
        equor   — for constitutional review of every Intent
        voxis   — for expression routing of approved Intents
        llm     — for policy generation and EFE estimation
        config  — NovaConfig
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

        # Sub-components — built in initialize()
        self._belief_updater: BeliefUpdater = BeliefUpdater()
        self._goal_manager: GoalManager | None = None
        self._policy_generator: BasePolicyGenerator | None = None
        self._efe_evaluator: EFEEvaluator | None = None
        self._deliberation_engine: DeliberationEngine | None = None
        self._intent_router: IntentRouter | None = None

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

        # Axon — may be wired before initialize() via set_axon(); applied at end of init.
        self._axon: Any = None

        # Logos — world model and compression metrics for EFE grounding
        self._logos: Any = None

        # Soma for allostatic signal reading
        self._soma: Any = None

        # Evo — for triggering emergency consolidation when FE budget exhausts
        self._evo: Any = None

        # Synapse event bus — for emitting BELIEF_UPDATED and POLICY_SELECTED
        self._synapse: Any = None

        # Oikos for economic balance reading (heartbeat hunger check)
        self._oikos: Any = None

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

        # ── Deliberation loop liveness tracking ───────────────────────────
        # Monotonic timestamp of the last completed deliberation cycle.
        # None until the first broadcast arrives.
        self._last_deliberation_at: float | None = None
        # Maximum seconds between deliberations before health degrades.
        self._deliberation_stale_threshold_s: float = 300.0  # 5 minutes

        # ── Budget-exhaustion / degradation tracking ──────────────────────
        # Consecutive decisions made under budget exhaustion (heuristic mode).
        self._consecutive_budget_exhausted: int = 0
        # Thymos reference — wired via set_thymos() if called.
        self._thymos: Any = None

        # True once initialize() completes successfully. Used by health() to
        # distinguish "not yet initialised" from "initialised but idle".
        self._initialized: bool = False

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

        # Build cognition cost calculator from config
        self._cost_calculator: CognitionCostCalculator | None = None
        if self._config.cognition_cost_enabled:
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

        self._intent_router = IntentRouter(voxis=self._voxis)

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

        # If set_axon() was called before initialize(), wire it now.
        if self._axon is not None:
            self._intent_router.set_axon(self._axon)
            self._logger.info("axon_wired_to_nova_deferred")

        self._logger.info(
            "nova_initialized",
            instance_name=self._instance_name,
            max_active_goals=self._config.max_active_goals,
            drive_weights=self._drive_weights,
        )

        # Restore persisted goals from Neo4j after a process restart.
        # Pass Synapse health records so stale maintenance goals (older than
        # 30 min, target system healthy) are suppressed before entering memory.
        neo4j = getattr(self._memory, "_neo4j", None)
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

        self._initialized = True

    def set_synapse(self, synapse: Any) -> None:
        """Wire Synapse so Nova can emit BELIEF_UPDATED and POLICY_SELECTED events."""
        self._synapse = synapse

        # Subscribe to Soma's interoceptive percepts so Nova can reprioritise
        # when the organism's body signals demand inward attention.
        from systems.synapse.types import SynapseEventType

        event_bus = getattr(synapse, "event_bus", synapse)
        if hasattr(event_bus, "subscribe"):
            event_bus.subscribe(
                SynapseEventType.INTEROCEPTIVE_PERCEPT,
                self._on_interoceptive_percept,
            )

        self._logger.info("synapse_wired_to_nova")

    def set_logos(self, logos: Any) -> None:
        """
        Wire Logos so Nova can ground EFE evaluation in the organism's
        actual world model — predictions, intelligence ratio, cognitive pressure.

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

    async def _on_interoceptive_percept(self, event: Any) -> None:
        """
        Handle INTEROCEPTIVE_PERCEPT from Soma.

        When the recommended action is ATTEND_INWARD, Nova injects an urgent
        self-investigation goal that takes priority over external tasks (bounties,
        user queries). This makes the organism pause external work to diagnose
        internal distress — equivalent to stopping what you're doing when you
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

    def set_telos(self, telos: Any) -> None:
        """Wire Telos so deliberation can weight policies by effective_I impact."""
        self._telos = telos
        if self._deliberation_engine is not None:
            self._deliberation_engine.set_telos(telos)
        self._logger.info("telos_wired_to_nova")

    def set_evo(self, evo: Any) -> None:
        """Wire Evo so Nova can trigger emergency consolidation on FE budget exhaustion."""
        self._evo = evo
        # Also wire the tournament engine into the deliberation engine so
        # _check_tournament_context() can sample hypothesis A/B trials.
        tournament_engine = getattr(evo, "tournament_engine", None)
        if tournament_engine is not None and self._deliberation_engine is not None:
            self._deliberation_engine.set_tournament_engine(tournament_engine)
        self._logger.info("evo_wired_to_nova")

    def set_axon(self, axon: AxonService) -> None:
        """
        Wire Axon into Nova's intent router after both are initialised.
        This enables Step 5 of the cognitive cycle: ACT.

        If called before initialize(), the axon is stored and applied at the
        end of initialize() so the router is always correctly wired.
        """
        self._axon = axon
        if self._intent_router is not None:
            self._intent_router.set_axon(axon)
            self._logger.info("axon_wired_to_nova")
        else:
            self._logger.debug("set_axon_stored_pending_initialize")

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

    # ─── Autonomous Heartbeat ─────────────────────────────────────

    async def start_heartbeat(self) -> None:
        """
        Continuous deliberation loop — the organism's autonomous drive.

        Runs every ``config.heartbeat_interval_seconds`` (default: 3600 = 1 hour).
        On each beat:
          1. Skip if active solves are pending (organism is busy).
          2. Read Oikos balance; if below ``hunger_balance_threshold_usd``,
             the organism is hungry.
          3. Construct a hunt_bounties Intent grounded in the drive state.
          4. Submit to Equor for constitutional approval.
          5. Route to Axon for execution via the intent router.

        Errors are caught and logged — a bad beat never kills the loop.

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
        if self._intent_router._axon is not None:
            self._intent_router._axon.begin_cycle()

        # ── 1. Skip if organism is busy with heavy external actions ──
        # Lightweight internal executors (store_insight, observe, query_memory,
        # update_goal, etc.) do NOT block the heartbeat — otherwise the organism
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
                # Assume hungry when balance is unreadable — safer to hunt than starve
                is_hungry = True

        self._logger.info(
            "heartbeat_beat",
            is_hungry=is_hungry,
            balance_usd=round(balance_usd, 2),
            hunger_threshold=self._config.hunger_balance_threshold_usd,
        )

        if not is_hungry:
            self._logger.info("heartbeat_satiated_skip")
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
            # Bounty hunting is AUTONOMOUS — no human approval required.
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
        except Exception as exc:
            self._logger.warning("heartbeat_equor_error", error=str(exc))
            return

        if equor_check.verdict != Verdict.APPROVED:
            self._logger.info(
                "heartbeat_equor_blocked",
                verdict=equor_check.verdict.value,
                reasoning=equor_check.reasoning[:120],
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
        myself at level 1 — I need PARTNER-level permission to hunt bounties."
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
        Falls back to the default ``PolicyGenerator`` signature — evolved
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
            # Evolved subclass has a different signature — try zero-arg
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

    # ─── BroadcastSubscriber Interface ───────────────────────────

    async def receive_broadcast(self, broadcast: WorkspaceBroadcast) -> None:
        """
        Called by Atune when the workspace broadcasts a percept.

        This is the primary cycle entry point. Nova updates beliefs,
        deliberates, and dispatches an Intent — or chooses silence.

        The full pipeline must complete in ≤5000ms (slow path budget).
        Fast path targets ≤150ms.
        """
        if self._deliberation_engine is None:
            return  # Not yet initialized

        self._total_broadcasts += 1
        self._last_deliberation_at = time.monotonic()
        self._current_affect = broadcast.affect

        # Yield immediately so the Synapse clock is not blocked if
        # the previous broadcast's fire-and-forget tasks haven't drained.
        await asyncio.sleep(0)

        # ── Belief update (≤50ms) ──
        delta = self._belief_updater.update_from_broadcast(broadcast)

        # ── Retrieve relevant memories (best-effort, non-blocking) ──
        memory_traces = await self._retrieve_relevant_memories_safe(broadcast)

        # ── Logos cognitive pressure → FE budget K modulation (<1ms) ──
        if self._logos is not None:
            try:
                logos_metrics = self._logos.get_latest_metrics()
                cognitive_pressure = logos_metrics.cognitive_pressure
                # When Logos is under compression pressure, reduce policy K to
                # conserve cognitive resources — the organism is overloaded.
                if cognitive_pressure > 0.75 and self._deliberation_engine is not None:
                    self._deliberation_engine.modulate_policy_k_from_pressure(
                        cognitive_pressure
                    )
                # Enrich belief VFE with Logos intelligence ratio — a low ratio
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

            # Trigger emergency consolidation via Evo (fire-and-forget).
            # This is the information-theoretic pause gate: the organism
            # has accumulated too much surprise and must consolidate before
            # resuming normal deliberation.
            if self._evo is not None:
                asyncio.create_task(
                    self._trigger_emergency_consolidation(),
                    name=f"nova_fe_budget_consolidation_{broadcast.broadcast_id[:8]}",
                )

            # Emit NOVA_DEGRADED and escalate if sustained.
            asyncio.create_task(
                self._handle_budget_exhaustion(broadcast.broadcast_id),
                name=f"nova_degradation_notify_{broadcast.broadcast_id[:8]}",
            )
        else:
            self._total_do_nothing += 1
            # Budget recovered — reset consecutive counter.
            self._consecutive_budget_exhausted = 0

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

        # ── Route intent if one was produced ──
        if intent is not None:
            self._total_intents_issued += 1
            await self._dispatch_intent(intent, broadcast, record)

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
            self._goal_manager.expire_stale_goals()
            self._goal_manager.prune_retired_goals()
            self._expire_stale_pending_intents()
            self._expire_stale_pending_counterfactuals()
            # Sync stale suspended goals to Neo4j (fire-and-forget)
            _maint_neo4j = getattr(self._memory, "_neo4j", None)
            if _maint_neo4j is not None:
                asyncio.create_task(
                    abandon_stale_goals(_maint_neo4j),
                    name="nova_abandon_stale_goals",
                )

        # ── Decay unobserved entity beliefs (background maintenance) ──
        self._belief_updater.decay_unobserved_entities()

    # ─── External API ─────────────────────────────────────────────

    async def add_goal(self, goal: Goal) -> Goal:
        """Add a goal directly (called by governance or test harness)."""
        assert self._goal_manager is not None
        result = self._goal_manager.add_goal(goal)
        # Embed the goal description for salience-guided attention
        asyncio.create_task(
            self._embed_goal(result),
            name=f"nova_embed_goal_{result.id[:8]}",
        )
        # Persist to Neo4j so goal survives process restarts
        neo4j = getattr(self._memory, "_neo4j", None)
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
                # PR merged — bounty reward confirmed. Credit Oikos and
                # evaluate reproductive fitness (mitosis trigger).
                await self._process_bounty_paid(outcome)
            elif is_bounty_rejected:
                # PR closed without merge — update beliefs accordingly.
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

        # Update goal progress if we know which goal this intent served
        if pending and self._goal_manager:
            goal = self._goal_manager.get_goal(pending.goal_id)
            if goal:
                # Bounty solves get a larger progress delta -- they are
                # high-effort, high-reward foraging actions.
                if is_bounty_solve and outcome.success:
                    progress_delta = 0.6
                elif is_bounty_paid:
                    progress_delta = 0.8  # Bounty actually paid — highest reward
                else:
                    progress_delta = 0.3 if outcome.success else 0.0
                new_progress = min(1.0, goal.progress + progress_delta)
                updated_goal = self._goal_manager.update_progress(
                    goal.id,
                    progress=new_progress,
                    episode_id=outcome.episode_id,
                )
                # Persist status/progress change to Neo4j (fire-and-forget)
                if updated_goal is not None:
                    _neo4j = getattr(self._memory, "_neo4j", None)
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

        # ── Forward tournament outcome to Evo ──
        # PendingIntent carries tournament context stamped at dispatch time.
        # Without this call the A/B experiment loop never receives signal.
        if (
            pending is not None
            and pending.tournament_id is not None
            and pending.tournament_hypothesis_id is not None
            and self._evo is not None
        ):
            self._evo.record_tournament_outcome(
                tournament_id=pending.tournament_id,
                hypothesis_id=pending.tournament_hypothesis_id,
                success=outcome.success,
                intent_id=outcome.intent_id,
            )

        # ── Resolve counterfactual episodes with regret ──
        cf_records = self._pending_counterfactuals.pop(outcome.intent_id, [])
        if cf_records:
            asyncio.create_task(
                self._resolve_counterfactuals(cf_records, outcome),
                name=f"nova_cf_resolve_{outcome.intent_id[:8]}",
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
          3. Evaluate reproductive fitness — if threshold exceeded, emit
             a high-priority spawn intent to trigger mitosis
          4. Large valence boost (+0.25) — this is an existential victory
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
                # No matching entity — create a new one for tracking
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

        # 3. Large valence boost — existential victory
        if self._current_affect:
            new_valence = min(1.0, self._current_affect.valence + 0.25)
            self._current_affect = self._current_affect.model_copy(
                update={"valence": new_valence}
            )

        # 4. Evaluate mitosis — check if organism is now rich enough to reproduce
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
                    # Mark as rejected rather than deleting — useful for learning
                    self._belief_updater.upsert_entity(existing.model_copy(
                        update={
                            "entity_type": "bounty_pr_rejected",
                            "properties": {
                                **existing.properties,
                                "status": "rejected",
                            },
                            "confidence": 0.3,  # Low confidence — will decay
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

            # Reproductive fitness achieved — generate spawn intent
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
            priority=0.95,  # Very high — reproduction is a rare, important event
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

        # Spawn intents require their own Equor review — reproduction is irreversible
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
        Real health check — assesses deliberation loop liveness, Neo4j
        reachability, and goal-set occupancy.

        Returns one of:
          "healthy"  — all checks pass
          "degraded" — one or more components are impaired but Nova is running
          "unhealthy" — critical failure (loop dead or Neo4j unreachable)
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
            # Initialized but never received a broadcast — likely just started.
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
        neo4j = getattr(self._memory, "_neo4j", None)
        if neo4j is None:
            components["neo4j"] = {"status": "not_wired"}
            issues.append("neo4j client not available")
        else:
            try:
                neo4j_result = await asyncio.wait_for(neo4j.health_check(), timeout=3.0)
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
        if self._goal_manager is None:
            return []
        return [
            ActiveGoalSummary(
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

    # ─── Private ──────────────────────────────────────────────────

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

    async def _trigger_emergency_consolidation(self) -> None:
        """
        Fire-and-forget emergency consolidation when the free energy budget
        is exhausted. Delegates to Evo's consolidation pipeline, then resets
        Nova's budget so deliberation can resume.

        This implements the predictive processing principle: when cumulative
        surprise exceeds the organism's tolerance, it must pause, consolidate
        what it has learned, and resume with a fresh budget.
        """
        if self._evo is None:
            return
        try:
            self._logger.info("fe_budget_emergency_consolidation_starting")
            result = await self._evo.run_consolidation()
            if result is not None:
                self._logger.info(
                    "fe_budget_emergency_consolidation_complete",
                    duration_ms=result.duration_ms,
                    hypotheses_integrated=result.hypotheses_integrated,
                )
            # Reset the budget now that consolidation has processed the backlog
            if self._deliberation_engine is not None:
                self._deliberation_engine.reset_fe_budget()
        except Exception as exc:
            self._logger.error(
                "fe_budget_emergency_consolidation_failed",
                error=str(exc),
            )
            # Even on failure, reset the budget to prevent permanent lockout.
            # The organism accepts it lost some learning opportunity but cannot
            # stay frozen indefinitely.
            if self._deliberation_engine is not None:
                self._deliberation_engine.reset_fe_budget()

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
            route = await self._intent_router.route(
                intent=intent,
                affect=broadcast.affect,
                conversation_id=self._current_conversation_id,
                equor_check=equor_check,
            )
            if route != "internal":
                self._total_intents_approved += 1
                # Track pending intent
                # Use the actual goal ID when available, fall back to description
                goal_id = getattr(intent.goal, "id", None) or intent.goal.description[:50]
                self._pending_intents[intent.id] = PendingIntent(
                    intent_id=intent.id,
                    goal_id=goal_id,
                    routed_to=route,
                    executors=[s.executor for s in intent.plan.steps],
                    # Carry tournament context so outcomes can be fed back to Evo
                    tournament_id=decision_record.tournament_id if decision_record else None,
                    tournament_hypothesis_id=(
                        decision_record.tournament_hypothesis_id if decision_record else None
                    ),
                )
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

            # Prefer the percept's dense embedding — already computed by Atune,
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

            # Nothing to search on — skip retrieval
            if not query_text and not query_embedding:
                return []

            result = await asyncio.wait_for(
                self._memory.retrieve(
                    query_text=query_text or None,
                    query_embedding=query_embedding,
                    max_results=5,
                ),
                timeout=0.15,  # 150ms hard timeout — must not block the cycle
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
                    rows = await self._memory._neo4j.execute_read(
                        """
                        MATCH (ep:Episode)
                        WHERE ep.id IN $ids
                        RETURN ep.id AS id,
                               ep.affect_valence AS affect_valence,
                               ep.source AS source,
                               ep.event_time AS event_time
                        """,
                        {"ids": episode_ids},
                    )
                    for row in rows:
                        episode_meta[row["id"]] = {
                            "affect_valence": row.get("affect_valence"),
                            "source": row.get("source"),
                            "event_time": row.get("event_time"),
                        }
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

        Default max age is 10 minutes — generous enough for real executors
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
                message="No outcome received within timeout — clearing from pending map",
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
        from systems.memory.episodic import store_counterfactual_episode

        for record in records:
            try:
                await store_counterfactual_episode(
                    self._memory._neo4j,
                    record,
                )
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
    ) -> None:
        """
        Resolve counterfactual episodes with regret when outcome arrives.

        Regret = estimated_pragmatic_value - actual_pragmatic_value
        Positive regret means the counterfactual was estimated to perform better
        than the actual outcome — i.e. the organism might have chosen poorly.
        """
        from systems.memory.episodic import (
            link_counterfactual_to_outcome,
            resolve_counterfactual,
        )

        actual_pragmatic = 1.0 if outcome.success else 0.0

        for record in records:
            try:
                regret = record.estimated_pragmatic_value - actual_pragmatic
                await resolve_counterfactual(
                    self._memory._neo4j,
                    record_id=record.id,
                    outcome_success=outcome.success,
                    actual_pragmatic_value=actual_pragmatic,
                    regret=regret,
                )
                # Link to outcome episode if available
                if outcome.episode_id:
                    await link_counterfactual_to_outcome(
                        self._memory._neo4j,
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

    def _record_decision(self, record: DecisionRecord) -> None:
        """Store decision record for observability (ring buffer)."""
        self._decision_records.append(record)
        if len(self._decision_records) > self._max_decision_records:
            self._decision_records = self._decision_records[-self._max_decision_records:]

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


