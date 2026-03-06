"""
EcodiaOS — Axon Service

The motor cortex. Axon receives approved Intents from Nova and turns them into
real-world effects — memory writes, API calls, scheduled tasks, notifications,
and federated messages.

Axon does not decide. It does not judge. It executes.
Decision authority lives in Nova. Ethical authority lives in Equor.
Axon is the disciplined hand that carries out the will.

Lifecycle:
  initialize() — builds the executor registry, wires safety systems
  execute()    — main entry point: accepts ExecutionRequest, returns AxonOutcome
  set_nova()   — wires the Nova feedback loop for outcome delivery
  shutdown()   — graceful teardown

Interface contracts (from spec):
  Validation (all steps):       ≤50ms
  Rate limit check:             ≤5ms
  Context assembly:             ≤30ms
  Simple intent (1-2 internal): ≤300ms end-to-end
  Complex intent (external):    ≤15,000ms end-to-end
"""

from __future__ import annotations

import contextlib
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.audit import AuditLogger
from systems.axon.credentials import CredentialStore
from systems.axon.executor import Executor
from systems.axon.executors import build_default_registry
from systems.axon.introspection import AxonIntrospector
from systems.axon.pipeline import ExecutionPipeline
from systems.axon.reactive import AxonReactiveAdapter
from systems.axon.safety import BudgetTracker, CircuitBreaker, RateLimiter
from systems.axon.shield import TransactionShield
from systems.axon.types import AxonOutcome, ExecutionRequest

if TYPE_CHECKING:
    from config import AxonConfig, MEVConfig
    from core.hotreload import NeuroplasticityBus
    from primitives.fast_path import FastPathIntent, FastPathOutcome
    from systems.atune.block_competition import BlockCompetitionMonitor
    from systems.axon.fast_path import FastPathExecutor
    from systems.axon.mev_analyzer import MEVAnalyzer
    from systems.axon.registry import ExecutorRegistry
    from systems.equor.template_library import TemplateLibrary
    from systems.memory.service import MemoryService
    from systems.nova.service import NovaService
    from systems.sacm.service import SACMClient
    from systems.synapse.event_bus import EventBus
    from systems.voxis.service import VoxisService

logger = structlog.get_logger()


class AxonService:
    """
    Axon — the EOS action execution system.

    AxonService is the single entry point for action execution.
    It owns and coordinates all sub-systems:
      - ExecutorRegistry: maps action types to handler implementations
      - ExecutionPipeline: runs the 8-stage execution pipeline (Stage 0–Stage 8)
      - BudgetTracker: enforces per-cycle action limits
      - RateLimiter: sliding-window per-executor rate limits
      - CircuitBreaker: per-executor open/closed/half-open state
      - CredentialStore: issues scoped, time-limited credentials
      - AuditLogger: records every execution permanently

    All sub-systems are constructed in initialize() and are immutable at runtime.
    """

    system_id: str = "axon"

    def __init__(
        self,
        config: AxonConfig,
        memory: MemoryService | None = None,
        voxis: VoxisService | None = None,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
        redis_client: Any = None,
        wallet: Any = None,
        synapse: Any = None,
        instance_id: str = "eos-default",
        github_config: Any = None,
        llm: Any = None,
        oikos: Any = None,
        spawner: Any = None,
        sacm_client: SACMClient | None = None,
        mev_config: MEVConfig | None = None,
    ) -> None:
        self._config = config
        self._memory = memory
        self._voxis = voxis
        self._bus = neuroplasticity_bus
        self._redis = redis_client
        self._wallet = wallet
        self._synapse = synapse
        self._instance_id = instance_id
        self._github_config = github_config
        self._llm = llm
        self._oikos = oikos
        self._spawner = spawner
        self._sacm_client = sacm_client
        self._mev_config = mev_config
        self._logger = logger.bind(system="axon")
        self._initialized = False

        # Sub-systems -- built in initialize()
        self._registry: ExecutorRegistry | None = None
        self._pipeline: ExecutionPipeline | None = None
        self._budget: BudgetTracker | None = None
        self._rate_limiter: RateLimiter | None = None
        self._circuit_breaker: CircuitBreaker | None = None
        self._credential_store: CredentialStore | None = None
        self._audit: AuditLogger | None = None
        self._shield: TransactionShield | None = None
        self._mev_analyzer: MEVAnalyzer | None = None
        self._block_competition_monitor: BlockCompetitionMonitor | None = None
        self._fast_path: FastPathExecutor | None = None

        # Metrics
        self._total_executions: int = 0
        self._successful_executions: int = 0
        self._failed_executions: int = 0

        # Recent outcomes ring buffer — last 50 executions for /api/v1/axon/outcomes
        self._recent_outcomes: deque[AxonOutcome] = deque(maxlen=50)

        # Optional event bus — wired via set_event_bus()
        self._event_bus: Any = None

        # Fovea — self-prediction loop: predict before action, resolve after
        self._fovea: Any = None

        # Oneiros — sleep safety: defer execution when organism is sleeping
        self._oneiros: Any = None

        # Introspection — learns from execution patterns to improve performance
        self._introspector = AxonIntrospector()

        # Reactive adapter — subscribes to Synapse events and adapts safety systems
        self._reactive: AxonReactiveAdapter | None = None

        # NeuroplasticityBus registration — done in initialize() when bus is available

    async def initialize(self) -> None:
        """
        Initialise all Axon sub-systems and build the executor registry.

        Must be called before any execute() calls.
        Idempotent — safe to call multiple times.
        """
        if self._initialized:
            return

        self._logger.info("axon_initializing", instance_id=self._instance_id)

        # Safety systems
        self._budget = BudgetTracker(self._config)
        self._rate_limiter = RateLimiter(redis_client=self._redis)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout_s=300,
            half_open_max_calls=1,
        )

        # Credential store
        self._credential_store = CredentialStore()

        # Audit logger
        self._audit = AuditLogger(memory=self._memory)

        # MEV Analyzer (Prompt #12: Predator Detection)
        if self._mev_config is not None and self._mev_config.enabled:
            from systems.atune.block_competition import (
                BlockCompetitionMonitor as _BlockCompMonitor,
            )
            from systems.axon.mev_analyzer import MEVAnalyzer as _MEVAnalyzer

            rpc_url = self._mev_config.rpc_url
            self._mev_analyzer = _MEVAnalyzer(
                rpc_url=rpc_url,
                high_risk_threshold=self._mev_config.high_risk_threshold,
                analysis_timeout_ms=self._mev_config.analysis_timeout_ms,
            )

            # Connect MEV analyzer (initialises Web3 provider)
            await self._mev_analyzer.connect()

            # Block competition monitor (feeds live data to MEV analyzer)
            self._block_competition_monitor = _BlockCompMonitor(
                rpc_url=rpc_url,
                poll_interval_s=self._mev_config.block_competition_poll_interval_s,
            )
            self._block_competition_monitor.add_listener(
                self._mev_analyzer.update_competition
            )
            await self._block_competition_monitor.start()

            self._logger.info(
                "mev_analyzer_initialized",
                rpc_available=self._mev_analyzer.has_rpc,
                block_monitor=self._block_competition_monitor.stats["running"],
            )

        # Transaction shield (Layer 1: Economic Immune System)
        self._shield = TransactionShield(
            wallet=self._wallet,
            max_slippage_bps=50,
            mev_analyzer=self._mev_analyzer,
        )

        # Build executor registry with all built-in executors
        self._registry = build_default_registry(
            memory=self._memory,
            voxis=self._voxis,
            redis_client=self._redis,
            wallet=self._wallet,
            synapse=self._synapse,
            github_config=self._github_config,
            llm=self._llm,
            oikos=self._oikos,
            spawner=self._spawner,
            sacm_client=self._sacm_client,
            event_bus=self._event_bus,
            budget_tracker=self._budget,
            circuit_breaker=self._circuit_breaker,
            rate_limiter=self._rate_limiter,
        )

        # Execution pipeline
        self._pipeline = ExecutionPipeline(
            registry=self._registry,
            budget=self._budget,
            rate_limiter=self._rate_limiter,
            circuit_breaker=self._circuit_breaker,
            credential_store=self._credential_store,
            audit_logger=self._audit,
            instance_id=self._instance_id,
            shield=self._shield,
        )

        # Reactive adapter — adapts safety systems based on organism state
        self._reactive = AxonReactiveAdapter(
            budget_tracker=self._budget,
            circuit_breaker=self._circuit_breaker,
            rate_limiter=self._rate_limiter,
        )

        self._initialized = True
        self._logger.info(
            "axon_initialized",
            executors=len(self._registry),
            executor_types=self._registry.list_types(),
        )

        # Register with the NeuroplasticityBus for hot-reload of Executor subclasses.
        if self._bus is not None:
            self._bus.register(
                base_class=Executor,
                registration_callback=self._on_executor_evolved,
                system_id="axon",
                # Skip abstract stubs that have no action_type set
                instance_qualifier=lambda cls: bool(cls.action_type),
            )

    def set_nova(self, nova: NovaService) -> None:
        """
        Wire the Nova feedback loop.

        Call this after both Nova and Axon are initialised.
        Must be called before execute() for outcome delivery to work.
        """
        if self._pipeline is None:
            raise RuntimeError("AxonService.initialize() must be called before set_nova()")
        self._pipeline.set_nova(nova)
        self._logger.info("nova_wired", system="axon")

    def set_atune(self, atune: Any) -> None:
        """
        Wire Atune so execution outcomes become workspace percepts.

        The organism should perceive its own actions — closing the
        intention→execution→perception loop.
        """
        if self._pipeline is None:
            raise RuntimeError("AxonService.initialize() must be called before set_atune()")
        self._pipeline.set_atune(atune)
        self._logger.info("atune_wired", system="axon")

    def set_wallet(self, wallet: Any) -> None:
        """
        Wire the Wallet client for metabolic actions.

        Enables Axon to execute financial actions: spending, transfers,
        on-chain transactions, and cost tracking against the energy budget.
        """
        self._wallet = wallet
        self._logger.info("wallet_wired", system="axon")

    def set_synapse(self, synapse: Any) -> None:
        """
        Wire the SynapseService so funding-request executors can read live
        metabolic state (rolling_deficit, burn_rate) and emit events.

        Call this after both AxonService and SynapseService are initialised,
        before the first cognitive cycle begins.
        """
        self._synapse = synapse
        # Propagate into the already-registered RequestFundingExecutor if the
        # registry has been built (i.e. initialize() was called before this).
        if self._registry is not None:
            executor = self._registry.get("request_funding")
            if executor is not None:
                # duck-type update — avoids importing RequestFundingExecutor here
                executor._synapse = synapse  # type: ignore[attr-defined]
        self._logger.info("synapse_wired", system="axon")

    def set_simula_service(self, simula: Any) -> None:
        """
        Wire the SimulaService so the SolveBountyExecutor can generate code
        and submit PRs for bounty issues.

        Call this after both AxonService and SimulaService are initialised,
        before the first cognitive cycle begins.
        """
        # Propagate into the already-registered SolveBountyExecutor if the
        # registry has been built (i.e. initialize() was called before this).
        if self._registry is not None:
            executor = self._registry.get("axon.solve_bounty")
            if executor is not None:
                executor._simula = simula  # type: ignore[attr-defined]
        self._logger.info("simula_service_wired", system="axon")

    def set_github_connector(self, github_connector: Any) -> None:
        """
        Wire a GitHubConnector into the BountySubmitExecutor.

        Call after both AxonService.initialize() and GitHubConnector are ready
        (i.e. after the identity connectors block in main.py). Mutates the
        already-registered BountySubmitExecutor in place — no re-initialization needed.
        """
        if self._registry is not None:
            executor = self._registry.get("submit_bounty_solution")
            if executor is not None:
                executor._github = github_connector  # type: ignore[attr-defined]
                self._logger.info("github_connector_wired_to_bounty_submit")
            else:
                self._logger.warning(
                    "bounty_submit_executor_not_found",
                    note="BountySubmitExecutor not in registry; GitHubConnector not wired",
                )

    def set_sacm(self, sacm_client: SACMClient) -> None:
        """
        Wire the SACMClient so the RemoteComputeExecutor can dispatch workloads.

        Call this after both AxonService and SACMClient are initialised,
        before the first cognitive cycle begins.  If the registry was
        already built (i.e. initialize() ran before this call), the
        existing RemoteComputeExecutor is updated in place; otherwise
        a new one is registered.
        """
        self._sacm_client = sacm_client
        if self._registry is not None:
            executor = self._registry.get("remote_compute")
            if executor is not None:
                # Hot-wire: replace the client on the live executor
                executor._client = sacm_client  # type: ignore[attr-defined]
            else:
                # Registry was built without a sacm_client — register now
                from systems.sacm.remote_compute_executor import RemoteComputeExecutor
                self._registry.register(RemoteComputeExecutor(sacm_client=sacm_client))
        self._logger.info("sacm_wired", system="axon")

    def set_template_library(self, template_library: TemplateLibrary) -> None:
        """
        Wire the Equor TemplateLibrary for Arbitrage Reflex Arc fast-path execution.

        Creates the FastPathExecutor that handles FastPathIntents, bypassing
        Nova deliberation and Equor review for pre-approved templates.

        Call this after AxonService.initialize() and after the TemplateLibrary
        has been populated with pre-approved templates.
        """
        if self._registry is None or self._rate_limiter is None or self._audit is None:
            msg = "AxonService.initialize() must be called before set_template_library()"
            raise RuntimeError(msg)

        from systems.axon.fast_path import FastPathExecutor

        self._fast_path = FastPathExecutor(
            registry=self._registry,
            template_library=template_library,
            rate_limiter=self._rate_limiter,
            audit_logger=self._audit,
            instance_id=self._instance_id,
        )
        self._logger.info("fast_path_executor_wired", system="axon")

    async def execute_fast_path(self, intent: FastPathIntent) -> FastPathOutcome:
        """
        Execute a FastPathIntent through the Arbitrage Reflex Arc.

        This is the fast-path entry point — Atune's MarketPatternDetector
        calls this directly when a market percept matches a pre-approved
        ConstitutionalTemplate. Bypasses Nova and Equor.

        Args:
            intent: FastPathIntent carrying the template pre-approval.

        Returns:
            FastPathOutcome with execution result and latency breakdown.

        Raises:
            RuntimeError: If the fast-path executor is not wired.
        """
        if self._fast_path is None:
            raise RuntimeError(
                "Fast-path executor not wired. Call set_template_library() first."
            )

        outcome = await self._fast_path.execute(intent)

        # Track in the general metrics
        self._total_executions += 1
        if outcome.success:
            self._successful_executions += 1
        else:
            self._failed_executions += 1

        return outcome

    def set_event_bus(self, event_bus: EventBus) -> None:
        """
        Wire the Synapse event bus so Axon can emit financial events.

        Call this after both AxonService and the event bus are initialised.
        When a wallet_transfer succeeds, Axon emits WALLET_TRANSFER_CONFIRMED
        which the Memory system picks up and encodes as a salience=1.0 episode.
        """
        self._event_bus = event_bus
        if self._pipeline is not None:
            self._pipeline.set_event_bus(event_bus)
        # Register reactive adapter to listen for organism state changes
        if self._reactive is not None:
            self._reactive.register_on_synapse(event_bus)
        # Register wake handler to drain deferred intents
        if hasattr(event_bus, "subscribe"):
            from systems.synapse.types import SynapseEventType
            event_bus.subscribe(
                SynapseEventType.WAKE_ONSET,
                self._on_wake_drain_queue,
            )
        self._logger.info("event_bus_wired", system="axon")

    def set_fovea(self, fovea: Any) -> None:
        """
        Wire Fovea for the self-prediction loop.

        When wired, Axon calls predict_self() before each execution and
        resolve_self() after completion. If the prediction is violated,
        the resulting InternalPredictionError flows through Fovea's
        standard pipeline with the 3x precision multiplier — the organism
        notices when its actions don't produce expected outcomes.
        """
        self._fovea = fovea
        self._logger.info("fovea_wired", system="axon")

    def set_oneiros(self, oneiros: Any) -> None:
        """
        Wire Oneiros sleep safety.

        When the organism is sleeping (Oneiros is in an active sleep stage),
        Axon defers execution of non-emergency intents. This prevents the
        motor cortex from acting while the cognitive system is consolidating —
        analogous to sleep paralysis in biological organisms.
        """
        self._oneiros = oneiros
        self._logger.info("oneiros_wired", system="axon")

    async def _on_wake_drain_queue(self, event: Any) -> None:
        """Re-execute intents that were deferred during sleep."""
        if self._reactive is None:
            return
        queued = self._reactive.drain_sleep_queue()
        if not queued:
            return
        self._logger.info(
            "wake_draining_sleep_queue",
            count=len(queued),
        )
        for request in queued:
            try:
                await self.execute(request)
            except Exception as exc:
                self._logger.warning(
                    "wake_drain_execute_error",
                    intent_id=request.intent.id,
                    error=str(exc),
                )

    def configure_credentials(self, credentials: dict[str, str]) -> None:
        """
        Load credentials into the CredentialStore.

        Called at startup to configure external service secrets.
        Format: {"service_name": "raw_secret_or_api_key"}
        """
        if self._credential_store is None:
            raise RuntimeError("AxonService.initialize() must be called first")
        self._credential_store.configure(credentials)

    def begin_cycle(self) -> None:
        """
        Notify Axon that a new cognitive cycle is starting.

        Resets the per-cycle execution budget. Call from Synapse at the
        start of each theta rhythm cycle.
        """
        if self._budget is not None:
            self._budget.begin_cycle()

    async def execute(self, request: ExecutionRequest) -> AxonOutcome:
        """
        Execute an approved Intent.

        This is the main external interface — Nova calls this via IntentRouter
        after Equor has approved the Intent.

        Args:
            request: ExecutionRequest containing the approved Intent and Equor verdict.

        Returns:
            AxonOutcome with full step-level detail and outcome summary.
            Never raises — failures are captured in the outcome.
        """
        if not self._initialized or self._pipeline is None:
            raise RuntimeError(
                "AxonService.initialize() must be called before execute()"
            )

        # ── Oneiros sleep safety gate ─────────────────────────────
        # When the organism is sleeping, defer non-emergency execution.
        # This prevents the motor cortex from acting during consolidation.
        # Defence-in-depth: reactive adapter also blocks via Synapse events,
        # but this gate catches anything that slips through.
        if self._oneiros is not None:
            try:
                is_sleeping = getattr(self._oneiros, "is_sleeping", False)
                if callable(is_sleeping):
                    is_sleeping = is_sleeping()
                if is_sleeping:
                    urgency = getattr(request.intent, "urgency", 0.0)
                    if urgency < 0.9:
                        # Queue for post-wake execution if reactive adapter is available
                        if self._reactive is not None:
                            self._reactive.queue_for_wake(request)
                        self._logger.info(
                            "execution_deferred_sleep_safety",
                            intent_id=request.intent.id,
                            goal=request.intent.goal.description[:60],
                            queued=self._reactive is not None,
                        )
                        from primitives.common import new_id
                        from systems.axon.types import ExecutionStatus
                        return AxonOutcome(
                            intent_id=request.intent.id,
                            execution_id=new_id(),
                            success=False,
                            status=ExecutionStatus.DEFERRED,
                            failure_reason="sleep_safety_deferred",
                            error=(
                                "Execution deferred: organism is sleeping. "
                                "Queued for post-wake execution."
                            ),
                        )
            except Exception as exc:
                self._logger.debug("oneiros_sleep_check_error", error=str(exc))

        self._total_executions += 1

        self._logger.info(
            "execute_start",
            intent_id=request.intent.id,
            goal=request.intent.goal.description[:60],
            steps=len(request.intent.plan.steps),
        )

        # ── Fovea: predict_self before execution ──────────────────
        # The organism predicts the outcome of its own action before
        # executing it. If the actual outcome violates the prediction,
        # the 3x precision multiplier fires an internal prediction error.
        fovea_prediction_id: str | None = None
        if self._fovea is not None:
            try:
                from systems.fovea.types import InternalErrorType
                action_types = [
                    step.executor for step in request.intent.plan.steps
                ]
                fovea_prediction_id = self._fovea.predict_self(
                    action_type=",".join(action_types) or "unknown",
                    internal_error_type=InternalErrorType.COMPETENCY,
                    predicted_state={
                        "intent_id": request.intent.id,
                        "expected_success": True,
                        "step_count": len(request.intent.plan.steps),
                        "action_types": action_types,
                        "goal": request.intent.goal.description[:100],
                    },
                )
            except Exception as exc:
                self._logger.debug("fovea_predict_self_error", error=str(exc))

        # ── Kairos: capture pre-execution state for intervention logging ──
        import time as _time
        kairos_pre_state = {
            "intent_id": request.intent.id,
            "timestamp_ms": int(_time.monotonic() * 1000),
            "goal": request.intent.goal.description[:200],
            "action_types": [step.executor for step in request.intent.plan.steps],
            "step_count": len(request.intent.plan.steps),
        }

        try:
            outcome = await self._pipeline.execute(request)
        except Exception as exc:
            # Pipeline should never raise, but guard anyway
            self._logger.error(
                "pipeline_raised_unexpectedly",
                intent_id=request.intent.id,
                error=str(exc),
            )
            from primitives.common import new_id
            from systems.axon.types import ExecutionStatus, FailureReason
            outcome = AxonOutcome(
                intent_id=request.intent.id,
                execution_id=new_id(),
                success=False,
                status=ExecutionStatus.FAILURE,
                failure_reason=FailureReason.EXECUTION_EXCEPTION.value,
                error=str(exc),
            )

        # ── Fovea: resolve_self after execution ──────────────────
        # Compare actual outcome to predicted state. Violations produce
        # InternalPredictionErrors with 3x precision multiplier.
        if self._fovea is not None and fovea_prediction_id is not None:
            try:
                internal_error = await self._fovea.resolve_self(
                    prediction_id=fovea_prediction_id,
                    actual_state={
                        "intent_id": outcome.intent_id,
                        "actual_success": outcome.success,
                        "step_count": len(outcome.step_outcomes),
                        "failure_reason": outcome.failure_reason or "",
                        "duration_ms": outcome.duration_ms,
                    },
                )
                if internal_error is not None:
                    self._logger.info(
                        "fovea_self_prediction_violated",
                        intent_id=outcome.intent_id,
                        error_magnitude=round(internal_error.magnitude, 3),
                        error_type=internal_error.internal_error_type.value
                        if hasattr(internal_error.internal_error_type, "value")
                        else str(internal_error.internal_error_type),
                    )
            except Exception as exc:
                self._logger.debug("fovea_resolve_self_error", error=str(exc))

        if outcome.success:
            self._successful_executions += 1
            # Emit WALLET_TRANSFER_CONFIRMED so Memory encodes it at salience=1.0
            if self._event_bus is not None:
                await self._emit_financial_events(outcome)
        else:
            self._failed_executions += 1

        # Emit ACTION_COMPLETED so Evo can update hypothesis confidence
        if self._event_bus is not None:
            await self._emit_action_completed(outcome)

        # ── Kairos: emit ACTION_INTERVENTION with before/after state ──
        # Structured intervention logging for causal direction testing.
        if self._event_bus is not None:
            await self._emit_kairos_intervention(outcome, kairos_pre_state)

        # ── Introspection: record outcome for learning ──────────────
        self._introspector.record_outcome(outcome)
        if self._budget is not None:
            self._introspector.record_cycle_utilization(self._budget.utilisation)

        # ── Self-healing: auto-evict persistently failing executors ──
        await self._check_self_healing(outcome)

        self._recent_outcomes.append(outcome)
        return outcome

    async def _emit_financial_events(self, outcome: AxonOutcome) -> None:
        """Emit WALLET_TRANSFER_CONFIRMED for any successful wallet_transfer steps."""
        from systems.synapse.types import SynapseEvent, SynapseEventType

        for step in outcome.step_outcomes:
            if step.action_type == "wallet_transfer" and step.result.success:
                data = dict(step.result.data)
                data["execution_id"] = outcome.execution_id
                event = SynapseEvent(
                    event_type=SynapseEventType.WALLET_TRANSFER_CONFIRMED,
                    data=data,
                    source_system="axon",
                )
                try:
                    await self._event_bus.emit(event)
                    self._logger.info(
                        "wallet_transfer_event_emitted",
                        tx_hash=data.get("tx_hash", ""),
                        token=data.get("token", ""),
                        amount=data.get("amount", ""),
                    )
                except Exception as exc:
                    self._logger.error(
                        "wallet_transfer_event_emit_failed", error=str(exc)
                    )

    async def _emit_action_completed(self, outcome: AxonOutcome) -> None:
        """Emit ACTION_COMPLETED after every execution so Evo can update hypothesis confidence."""
        from systems.synapse.types import SynapseEvent, SynapseEventType

        # Build outcome description from world-state changes, or failure reason
        if outcome.world_state_changes:
            outcome_description = "; ".join(outcome.world_state_changes)
        elif outcome.failure_reason:
            outcome_description = outcome.failure_reason
        else:
            outcome_description = "completed" if outcome.success else "failed"

        # Sum economic impact from wallet_transfer steps (positive = inflow, negative = outflow)
        # Also sum economic_delta_usd from any executor (e.g. bounty_hunt estimated reward)
        economic_delta = 0.0
        for step in outcome.step_outcomes:
            if step.action_type == "wallet_transfer" and step.result.success:
                raw = step.result.data.get("amount", "0")
                with contextlib.suppress(ValueError, TypeError):
                    economic_delta += float(str(raw).replace("$", "").replace(",", ""))
            if step.result.success and "economic_delta_usd" in step.result.data:
                with contextlib.suppress(ValueError, TypeError):
                    economic_delta += float(step.result.data["economic_delta_usd"])

        # Derive action_types from step outcomes so Evo's ProcedureCodifier can
        # group recurring sequences. episode_id lets Evo link back to the memory trace.
        action_types = [step.action_type for step in outcome.step_outcomes]

        event = SynapseEvent(
            event_type=SynapseEventType.ACTION_COMPLETED,
            source_system="axon",
            data={
                "intent_id": outcome.intent_id,
                "outcome": outcome_description,
                "success": outcome.success,
                "economic_delta": economic_delta,
                "action_types": action_types,
                "episode_id": outcome.episode_id,
            },
        )
        try:
            await self._event_bus.emit(event)
            self._logger.info(
                "action_completed_event_emitted",
                intent_id=outcome.intent_id,
                success=outcome.success,
                economic_delta=economic_delta,
            )
        except Exception as exc:
            self._logger.error(
                "action_completed_event_emit_failed",
                intent_id=outcome.intent_id,
                error=str(exc),
            )

    async def _emit_kairos_intervention(
        self, outcome: AxonOutcome, pre_state: dict[str, Any]
    ) -> None:
        """
        Emit structured ACTION_INTERVENTION for Kairos causal direction testing.

        Each intervention record carries:
        - before_state: captured before execution (goal, actions planned)
        - after_state: captured after execution (success, world changes, duration)
        - causal_metadata: timestamps, execution_id, step-level results

        Kairos uses these to build causal graphs: did this action CAUSE that
        world state change, or was the change already underway?
        """
        import time as _time

        from systems.synapse.types import SynapseEvent, SynapseEventType

        # Build the after-state snapshot
        after_state = {
            "intent_id": outcome.intent_id,
            "execution_id": outcome.execution_id,
            "success": outcome.success,
            "partial": outcome.partial,
            "failure_reason": outcome.failure_reason,
            "world_state_changes": outcome.world_state_changes[:10],
            "new_observations": outcome.new_observations[:10],
            "duration_ms": outcome.duration_ms,
            "step_results": [
                {
                    "action_type": step.action_type,
                    "success": step.result.success,
                    "side_effects": step.result.side_effects[:3],
                    "duration_ms": step.duration_ms,
                }
                for step in outcome.step_outcomes
            ],
            "timestamp_ms": int(_time.monotonic() * 1000),
        }

        # Economic impact (for causal analysis of financial interventions)
        economic_delta = 0.0
        for step in outcome.step_outcomes:
            if step.action_type == "wallet_transfer" and step.result.success:
                raw = step.result.data.get("amount", "0")
                with contextlib.suppress(ValueError, TypeError):
                    economic_delta += float(str(raw).replace("$", "").replace(",", ""))

        event = SynapseEvent(
            event_type=SynapseEventType.ACTION_COMPLETED,
            source_system="axon",
            data={
                "intervention_type": "action_execution",
                "before_state": pre_state,
                "after_state": after_state,
                "causal_metadata": {
                    "execution_id": outcome.execution_id,
                    "intent_id": outcome.intent_id,
                    "pre_timestamp_ms": pre_state.get("timestamp_ms", 0),
                    "post_timestamp_ms": after_state["timestamp_ms"],
                    "latency_ms": outcome.duration_ms,
                    "economic_delta": economic_delta,
                    "action_types": [s.action_type for s in outcome.step_outcomes],
                    # Flag for Kairos: distinguish from passive observation
                    "is_intervention": True,
                },
            },
        )
        try:
            await self._event_bus.emit(event)
        except Exception as exc:
            self._logger.debug(
                "kairos_intervention_emit_failed",
                intent_id=outcome.intent_id,
                error=str(exc),
            )

    async def _check_self_healing(self, outcome: AxonOutcome) -> None:
        """
        Auto-evict executors that are persistently failing.

        When introspection detects a degrading executor (success rate < 50%
        with 3+ consecutive failures and 5+ total executions), we:
          1. Deregister the executor from the registry
          2. Force-open its circuit breaker
          3. Emit an incident event so Thymos can attempt repair

        This prevents a broken executor from consuming budget and causing
        cascading failures across the pipeline.
        """
        degrading = self._introspector.get_degrading_executors()
        if not degrading or self._registry is None:
            return

        for profile in degrading:
            action_type = profile["action_type"]
            consecutive = profile.get("consecutive_failures", 0)

            # Only auto-evict after sustained degradation (5+ consecutive)
            if consecutive < 5:
                continue

            # Deregister the executor
            if hasattr(self._registry, "deregister"):
                evicted = self._registry.deregister(action_type)
            else:
                evicted = False

            if not evicted:
                continue

            # Force-open the circuit breaker
            if self._circuit_breaker is not None:
                self._circuit_breaker.force_open(action_type)

            self._logger.warning(
                "executor_self_healed_eviction",
                action_type=action_type,
                success_rate=profile.get("success_rate", 0),
                consecutive_failures=consecutive,
                recent_failures=profile.get("recent_failures", []),
            )

            # Emit incident so Thymos can attempt repair
            if self._event_bus is not None:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                event = SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_DEGRADED,
                    source_system="axon",
                    data={
                        "system_id": f"axon.executor.{action_type}",
                        "reason": "self_healing_eviction",
                        "success_rate": profile.get("success_rate", 0),
                        "consecutive_failures": consecutive,
                        "recent_failures": profile.get("recent_failures", []),
                    },
                )
                try:
                    await self._event_bus.emit(event)
                except Exception as exc:
                    self._logger.debug(
                        "self_healing_event_emit_failed", error=str(exc)
                    )

    def register_executor(self, executor: Any) -> None:
        """
        Register a custom executor at runtime.

        For use by integrations, plugins, and governance-approved extensions.
        The executor must be an instance of Executor ABC.
        """
        if self._registry is None:
            raise RuntimeError("AxonService.initialize() must be called first")
        self._registry.register(executor)
        self._logger.info(
            "executor_registered_runtime",
            action_type=executor.action_type,
        )

    def get_executor(self, action_type: str) -> Any | None:
        """
        Look up a registered executor by action_type.

        Returns the executor instance or None if not found.
        Used by API routes (e.g. legal.py) to retrieve executors
        that expose their own HITL state management.
        """
        if self._registry is None:
            return None
        return self._registry.get(action_type)

    async def health(self) -> dict[str, Any]:
        """Self-health report (implements ManagedSystem protocol)."""
        report: dict[str, Any] = {
            "status": "healthy" if self._initialized else "starting",
            "total_executions": self._total_executions,
            "successful": self._successful_executions,
            "failed": self._failed_executions,
            "executor_count": len(self._registry) if self._registry else 0,
        }
        if self._fast_path is not None:
            report["fast_path"] = self._fast_path.stats
        return report

    async def shutdown(self) -> None:
        """Graceful shutdown — log final stats."""
        if self._bus is not None:
            self._bus.deregister(Executor)

        # Tear down MEV subsystems
        if self._block_competition_monitor is not None:
            await self._block_competition_monitor.stop()
        if self._mev_analyzer is not None:
            await self._mev_analyzer.close()

        self._logger.info(
            "axon_shutdown",
            total_executions=self._total_executions,
            successful=self._successful_executions,
            failed=self._failed_executions,
            circuit_trips=self._circuit_breaker.trip_count()
            if self._circuit_breaker else 0,
            audit_stats=self._audit.stats if self._audit else {},
            mev_stats=self._shield.stats if self._shield else {},
        )

    def _on_executor_evolved(self, executor: Executor) -> None:
        """
        Registration callback for NeuroplasticityBus.

        Called once per Executor subclass found in a hot-reloaded file.
        Registers the new instance into the live ExecutorRegistry with
        replace=True so the existing entry (if any) is atomically replaced.
        """
        if self._registry is None:
            return
        self._registry.register(executor, replace=True)
        self._logger.info(
            "axon_executor_hot_reloaded",
            action_type=executor.action_type,
            executor=type(executor).__name__,
        )

    @property
    def recent_outcomes(self) -> list[AxonOutcome]:
        """Return recent execution outcomes (newest first), up to 50."""
        return list(reversed(self._recent_outcomes))

    @property
    def stats(self) -> dict[str, Any]:
        """Return current operational statistics."""
        return {
            "initialized": self._initialized,
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
            "executor_count": len(self._registry) if self._registry else 0,
            "executor_types": self._registry.list_types() if self._registry else [],
            "circuit_trips": self._circuit_breaker.trip_count()
            if self._circuit_breaker
            else 0,
            "budget_utilisation": self._budget.utilisation if self._budget else 0.0,
            "audit": self._audit.stats if self._audit else {},
            "fast_path": self._fast_path.stats if self._fast_path else {},
            "introspection": self._introspector.stats,
            "reactive": self._reactive.stats if self._reactive else {},
        }
