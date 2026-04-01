"""
EcodiaOS - Axon Service

The motor cortex. Axon receives approved Intents from Nova and turns them into
real-world effects - memory writes, API calls, scheduled tasks, notifications,
and federated messages.

Axon does not decide. It does not judge. It executes.
Decision authority lives in Nova. Ethical authority lives in Equor.
Axon is the disciplined hand that carries out the will.

Lifecycle:
  initialize() - builds the executor registry, wires safety systems
  execute()    - main entry point: accepts ExecutionRequest, returns AxonOutcome
  set_nova()   - wires the Nova feedback loop for outcome delivery
  shutdown()   - graceful teardown

Interface contracts (from spec):
  Validation (all steps):       ≤50ms
  Rate limit check:             ≤5ms
  Context assembly:             ≤30ms
  Simple intent (1-2 internal): ≤300ms end-to-end
  Complex intent (external):    ≤15,000ms end-to-end
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector, SystemID
from primitives.re_training import RETrainingExample
from systems.axon.audit import AuditLogger
from systems.axon.credentials import CredentialStore
from systems.axon.executor import Executor
from systems.axon.executors import build_default_registry
from systems.axon.introspection import AxonIntrospector
from systems.axon.pipeline import ExecutionPipeline
from systems.axon.performance_monitor import ActionPerformanceMonitor
from systems.axon.reactive import AxonReactiveAdapter
from systems.axon.safety import BudgetTracker, CircuitBreaker, RateLimiter
from systems.axon.shield import TransactionShield
from systems.axon.types import AxonOutcome, ExecutionRequest, RateLimit

if TYPE_CHECKING:
    from config import AxonConfig, MEVConfig
    from core.hotreload import NeuroplasticityBus
    from primitives.fast_path import FastPathIntent, FastPathOutcome
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
    Axon - the EOS action execution system.

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
        self._block_competition_monitor: Any = None  # BlockCompetitionMonitor - injected via set_block_competition_monitor()
        self._fast_path: FastPathExecutor | None = None

        # Metrics
        self._total_executions: int = 0
        self._successful_executions: int = 0
        self._failed_executions: int = 0

        # Telemetry broadcast - emit full introspector state every N theta cycles
        self._telemetry_cycle_counter: int = 0
        self._telemetry_emit_interval: int = 50  # matches COHERENCE_SNAPSHOT / METABOLIC_SNAPSHOT cadence

        # Recent outcomes ring buffer - last 50 executions for /api/v1/axon/outcomes
        self._recent_outcomes: deque[AxonOutcome] = deque(maxlen=50)

        # Optional event bus - wired via set_event_bus()
        self._event_bus: Any = None

        # Fovea - self-prediction loop: predict before action, resolve after
        self._fovea: Any = None

        # Oneiros - sleep safety: defer execution when organism is sleeping
        self._oneiros: Any = None

        # Introspection - learns from execution patterns to improve performance
        self._introspector = AxonIntrospector()

        # Performance monitor - tracks rolling success rate for Loop 2 closure
        self._performance_monitor = ActionPerformanceMonitor()

        # Reactive adapter - subscribes to Synapse events and adapts safety systems
        self._reactive: AxonReactiveAdapter | None = None

        # ── Metabolic gating ──────────────────────────────────────────────
        self._starvation_level: str = "nominal"

        # ── Budget override (EQUOR_BUDGET_OVERRIDE) ─────────────────────
        self._budget_override_active: bool = False
        self._budget_override_original: int = 5
        self._budget_override_cycles_remaining: int = 0
        self._budget_override_reason: str = ""

        # ── Skia VitalityCoordinator modulation ───────────────────────
        self._modulation_halted: bool = False


        # ── PR merge polling (30-min background loop) ─────────────────
        self._pr_poll_task: Any = None  # asyncio.Task - cancelled in shutdown()
        _PR_POLL_INTERVAL_S = 30 * 60   # 30 minutes - conservative GitHub API usage
        self._pr_poll_interval_s: int = _PR_POLL_INTERVAL_S
        # NeuroplasticityBus registration - done in initialize() when bus is available

    async def initialize(self) -> None:
        """
        Initialise all Axon sub-systems and build the executor registry.

        Must be called before any execute() calls.
        Idempotent - safe to call multiple times.
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
            redis_client=self._redis,
            event_bus=self._event_bus,
        )
        # Restore persisted circuit breaker states so tripped executors
        # do not silently reset to CLOSED after a process restart (Spec §5.3).
        await self._circuit_breaker.load_all_states()

        # Credential store
        self._credential_store = CredentialStore()

        # Audit logger
        self._audit = AuditLogger(memory=self._memory)

        # MEV Analyzer (Prompt #12: Predator Detection)
        if self._mev_config is not None and self._mev_config.enabled:
            from systems.axon.mev_analyzer import MEVAnalyzer as _MEVAnalyzer

            rpc_url = self._mev_config.rpc_url
            self._mev_analyzer = _MEVAnalyzer(
                rpc_url=rpc_url,
                high_risk_threshold=self._mev_config.high_risk_threshold,
                analysis_timeout_ms=self._mev_config.analysis_timeout_ms,
            )

            # Connect MEV analyzer (initialises Web3 provider)
            await self._mev_analyzer.connect()

            # Block competition monitor is injected via set_block_competition_monitor()
            # after initialize() - avoids cross-system import from systems.fovea.
            # Wiring layer (main.py / core/wiring.py) constructs and injects it.
            if self._block_competition_monitor is not None:
                self._block_competition_monitor.add_listener(
                    self._mev_analyzer.update_competition
                )
                await self._block_competition_monitor.start()

            self._logger.info(
                "mev_analyzer_initialized",
                rpc_available=self._mev_analyzer.has_rpc,
                block_monitor=(
                    self._block_competition_monitor.stats["running"]
                    if self._block_competition_monitor is not None
                    else False
                ),
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

        # Reactive adapter - adapts safety systems based on organism state
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

        # ── Child boot: apply inherited template genome if present ──
        # On child instances, ECODIAOS_AXON_GENOME_PAYLOAD carries the
        # parent's AxonGenomeFragment JSON. Apply it now so inherited templates
        # are available from the first cognitive cycle.
        import os as _os
        _axon_genome_payload = _os.environ.get("ECODIAOS_AXON_GENOME_PAYLOAD", "")
        if _axon_genome_payload:
            try:
                import json as _json
                from primitives.genome_inheritance import AxonGenomeFragment as _AxonGF
                _fragment = _AxonGF.model_validate(_json.loads(_axon_genome_payload))
                await self._initialize_from_parent_templates(_fragment)
            except Exception as _exc:
                self._logger.warning(
                    "axon_genome_payload_parse_failed",
                    error=str(_exc),
                    note="Child will start without inherited templates",
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


        # ── Start background PR merge polling loop ─────────────────────
        # Polls pending PR merge status every 30 minutes so the organism
        # learns about accepted bounties without requiring Nova to trigger
        # a monitor_prs execution intent on every cognitive cycle.
        import asyncio as _asyncio
        from utils.supervision import supervised_task as _supervised_task
        self._pr_poll_task = _supervised_task(
            self._pr_merge_poll_loop(),
            name="axon_pr_merge_poll",
            restart=True,
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

        The organism should perceive its own actions - closing the
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
                # duck-type update - avoids importing RequestFundingExecutor here
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
        already-registered BountySubmitExecutor in place - no re-initialization needed.
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
                # Registry was built without a sacm_client - register now
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

        This is the fast-path entry point - Atune's MarketPatternDetector
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
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            example = RETrainingExample(
                source_system=SystemID.AXON,
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
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                data=example.model_dump(mode="json"),
                source_system="axon",
            ))
        except Exception:
            self._logger.debug("re_training_emit_failed", exc_info=True)

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
        # Register lifecycle event handlers
        if hasattr(event_bus, "subscribe"):
            from systems.synapse.types import SynapseEventType
            event_bus.subscribe(
                SynapseEventType.WAKE_ONSET,
                self._on_wake_drain_queue,
            )
            event_bus.subscribe(
                SynapseEventType.METABOLIC_PRESSURE,
                self._on_metabolic_pressure,
            )
            event_bus.subscribe(
                SynapseEventType.GENOME_EXTRACT_REQUEST,
                self._on_genome_extract_request,
            )
            event_bus.subscribe(
                SynapseEventType.METABOLIC_EMERGENCY,
                self._on_metabolic_emergency,
            )
            event_bus.subscribe(
                SynapseEventType.ORGANISM_SLEEP,
                self._on_organism_sleep,
            )
            if hasattr(SynapseEventType, "YIELD_DEPLOYMENT_REQUEST"):
                event_bus.subscribe(
                    SynapseEventType.YIELD_DEPLOYMENT_REQUEST,
                    self._on_yield_deployment_request,
                )
            if hasattr(SynapseEventType, "CYCLE_COMPLETED"):
                event_bus.subscribe(
                    SynapseEventType.CYCLE_COMPLETED,
                    self._on_theta_cycle_complete,
                )
            if hasattr(SynapseEventType, "EQUOR_BUDGET_OVERRIDE"):
                event_bus.subscribe(
                    SynapseEventType.EQUOR_BUDGET_OVERRIDE,
                    self._on_equor_budget_override,
                )
            if hasattr(SynapseEventType, "ACTION_BUDGET_EXPANSION_RESPONSE"):
                event_bus.subscribe(
                    SynapseEventType.ACTION_BUDGET_EXPANSION_RESPONSE,
                    self._on_budget_expansion_response,
                )
            if hasattr(SynapseEventType, "EVO_ADJUST_BUDGET"):
                event_bus.subscribe(
                    SynapseEventType.EVO_ADJUST_BUDGET,
                    self._on_evo_adjust_budget,
                )
            event_bus.subscribe(
                SynapseEventType.SYSTEM_MODULATION,
                self._on_system_modulation,
            )
            if hasattr(SynapseEventType, "VULNERABILITY_CONFIRMED"):
                event_bus.subscribe(
                    SynapseEventType.VULNERABILITY_CONFIRMED,
                    self._on_vulnerability_confirmed,
                )
        self._logger.info("event_bus_wired", system="axon")

    async def _on_metabolic_pressure(self, event: Any) -> None:
        """React to organism-wide metabolic pressure changes."""
        data = getattr(event, "data", {}) or {}
        level = data.get("starvation_level", "")
        if not level:
            return
        old = self._starvation_level
        self._starvation_level = level
        if level != old:
            self._logger.info("axon_starvation_level_changed", old=old, new=level)

    async def _on_yield_deployment_request(self, event: Any) -> None:
        """
        Subscribe to YIELD_DEPLOYMENT_REQUEST from Oikos and execute the DeFi
        yield operation directly via DeFiYieldExecutor, then emit
        YIELD_DEPLOYMENT_RESULT so Oikos can complete its request/response future.

        This is a direct executor dispatch - it bypasses the Nova/Equor pipeline
        because the request already originates from an Oikos-internal metabolic
        decision.  The DeFiYieldExecutor's own safety gates (APY floor, gas floor,
        rate limit, WalletClient guard) still apply.
        """
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        data = getattr(event, "data", {}) or {}
        request_id = str(data.get("request_id", ""))
        action = str(data.get("action", "deposit"))
        amount_usd = str(data.get("amount_usd", "0"))
        protocol = str(data.get("protocol", "aave"))

        self._logger.info(
            "yield_deployment_request_received",
            request_id=request_id,
            action=action,
            amount_usd=amount_usd,
            protocol=protocol,
        )

        # Locate the registered DeFiYieldExecutor
        executor = None
        if self._registry is not None:
            executor = self._registry.get("defi_yield")

        if executor is None:
            self._logger.warning(
                "yield_deployment_executor_not_found",
                request_id=request_id,
            )
            await self._emit_yield_deployment_result(
                request_id=request_id,
                success=False,
                error="DeFiYieldExecutor not registered",
                protocol=protocol,
                amount_usd=amount_usd,
            )
            return

        # ── Constitutional gate ────────────────────────────────────
        # Capital deployments must pass Equor review even when the request
        # originates from Oikos's metabolic scheduler.  Oikos optimises for
        # metabolic efficiency; Equor checks drive alignment and invariants.
        # Pattern mirrors BountySubmitExecutor and DynamicExecutorBase.
        import asyncio as _asyncio
        from systems.synapse.types import SynapseEvent, SynapseEventType as _SET

        _action_id = f"yield_deployment:{request_id}"
        _permit_fut: _asyncio.Future[bool] = _asyncio.get_event_loop().create_future()

        async def _on_yield_permit(event: Any) -> None:
            _data = getattr(event, "data", {}) or {}
            if _data.get("action_id") != _action_id:
                return
            _verdict = str(_data.get("verdict", "PERMIT")).upper()
            if not _permit_fut.done():
                _permit_fut.set_result(_verdict == "PERMIT")

        self._event_bus.subscribe(_SET.EQUOR_ECONOMIC_PERMIT, _on_yield_permit)

        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=_SET.EQUOR_ECONOMIC_INTENT,
                source_system="axon.yield_deployment",
                data={
                    "action_id": _action_id,
                    "mutation_type": "yield_deployment",
                    "amount_usd": amount_usd,
                    "from_account": "oikos_metabolic",
                    "to_account": f"{protocol}_yield_protocol",
                    "rationale": (
                        f"Oikos metabolic scheduler requested {action} of "
                        f"${amount_usd} via {protocol}. "
                        f"Request ID: {request_id}."
                    ),
                    "protocol": protocol,
                    "action": action,
                    "request_id": request_id,
                },
            ))
            _permitted = await _asyncio.wait_for(_permit_fut, timeout=30.0)
        except _asyncio.TimeoutError:
            self._logger.warning(
                "yield_deployment_equor_timeout",
                request_id=request_id,
                note="Equor did not respond within 30s; auto-permitting yield deployment",
            )
            _permitted = True  # Fail-open on Equor timeout to preserve metabolic function
        except Exception as _gate_exc:
            self._logger.error(
                "yield_deployment_equor_gate_error",
                request_id=request_id,
                error=str(_gate_exc),
            )
            _permitted = False

        if not _permitted:
            self._logger.info(
                "yield_deployment_equor_denied",
                request_id=request_id,
                action=action,
                amount_usd=amount_usd,
                protocol=protocol,
            )
            await self._emit_yield_deployment_result(
                request_id=request_id,
                success=False,
                error="Equor constitutional review denied yield deployment",
                protocol=protocol,
                amount_usd=amount_usd,
            )
            return

        from systems.axon.types import ExecutionContext
        from primitives.common import new_id

        context = ExecutionContext(
            execution_id=new_id(),
            intent_id="yield_deployment_request",
            executor_name="defi_yield",
            budget_usd=None,
        )

        try:
            result = await executor.execute(
                params={"action": action, "amount": amount_usd, "protocol": protocol},
                context=context,
            )
        except Exception as exc:
            self._logger.error(
                "yield_deployment_executor_raised",
                request_id=request_id,
                error=str(exc),
            )
            await self._emit_yield_deployment_result(
                request_id=request_id,
                success=False,
                error=str(exc),
                protocol=protocol,
                amount_usd=amount_usd,
            )
            return

        tx_hash = result.data.get("tx_hash", "") if result.data else ""
        await self._emit_yield_deployment_result(
            request_id=request_id,
            success=result.success,
            error=result.error or "",
            protocol=protocol,
            amount_usd=amount_usd,
            tx_hash=tx_hash,
            action=action,
        )

    async def _emit_yield_deployment_result(
        self,
        request_id: str,
        success: bool,
        error: str = "",
        protocol: str = "",
        amount_usd: str = "0",
        tx_hash: str = "",
        action: str = "deposit",
    ) -> None:
        """Emit YIELD_DEPLOYMENT_RESULT so Oikos can resolve its awaiting future."""
        if self._event_bus is None:
            return
        from systems.synapse.types import SynapseEvent, SynapseEventType

        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.YIELD_DEPLOYMENT_RESULT,
                source_system="axon",
                data={
                    "request_id": request_id,
                    "success": success,
                    "tx_hash": tx_hash,
                    "protocol": protocol,
                    "amount": amount_usd,
                    "action": action,
                    "error": error,
                },
            ))
            self._logger.info(
                "yield_deployment_result_emitted",
                request_id=request_id,
                success=success,
                tx_hash=tx_hash,
                protocol=protocol,
            )
        except Exception as exc:
            self._logger.error(
                "yield_deployment_result_emit_failed",
                request_id=request_id,
                error=str(exc),
            )

    async def _on_genome_extract_request(self, event: Any) -> None:
        """
        Respond to GENOME_EXTRACT_REQUEST from Mitosis.

        Returns motor patterns, executor configs, and circuit breaker thresholds
        as the Axon segment of the organism's genome.
        """
        data = getattr(event, "data", {}) or {}
        request_id = data.get("request_id", "")

        # Build Axon genome segment
        executor_configs: list[dict[str, Any]] = []
        if self._registry is not None:
            for action_type in self._registry.list_types():
                executor = self._registry.get(action_type)
                if executor is not None:
                    executor_configs.append({
                        "action_type": executor.action_type,
                        "required_autonomy": executor.required_autonomy,
                        "reversible": executor.reversible,
                        "max_duration_ms": executor.max_duration_ms,
                        "rate_limit": {
                            "max_calls": executor.rate_limit.max_calls,
                            "window_seconds": executor.rate_limit.window_seconds,
                        },
                    })

        genome_segment = {
            "system": "axon",
            "request_id": request_id,
            "executor_configs": executor_configs,
            "circuit_breaker": {
                "failure_threshold": self._circuit_breaker.failure_threshold
                if self._circuit_breaker else 5,
                "recovery_timeout_s": self._circuit_breaker.recovery_timeout_s
                if self._circuit_breaker else 300,
                "half_open_max_calls": self._circuit_breaker.half_open_max_calls
                if self._circuit_breaker else 1,
            },
            "budget": {
                "max_actions_per_cycle": self._budget.budget.max_actions_per_cycle
                if self._budget else 5,
                "max_concurrent_executions": self._budget.budget.max_concurrent_executions
                if self._budget else 3,
            },
            "motor_patterns": {
                "total_executions": self._total_executions,
                "success_rate": (
                    self._successful_executions / max(1, self._total_executions)
                ),
            },
        }

        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.GENOME_EXTRACT_RESPONSE,
                    source_system="axon",
                    data=genome_segment,
                ))
            except Exception as exc:
                self._logger.debug("genome_extract_response_failed", error=str(exc))

        self._logger.info(
            "genome_extract_responded",
            request_id=request_id,
            executor_count=len(executor_configs),
        )

    async def _on_metabolic_emergency(self, event: Any) -> None:
        """
        Respond to METABOLIC_EMERGENCY by pausing non-critical repairs.

        Sets starvation level to emergency so the execute() gate blocks
        low-urgency intents. Only survival-priority actions proceed.
        """
        data = getattr(event, "data", {}) or {}
        severity = data.get("severity", data.get("level", "emergency"))

        old_level = self._starvation_level
        self._starvation_level = "emergency" if severity != "critical" else "critical"

        if old_level != self._starvation_level:
            self._logger.warning(
                "metabolic_emergency_received",
                old_level=old_level,
                new_level=self._starvation_level,
                severity=severity,
            )

        # Force-open circuit breakers for non-essential executors to reduce load
        if self._circuit_breaker is not None:
            non_essential = [
                "social_post", "bounty_hunt", "deploy_asset", "phantom_liquidity",
                "publish_content",
            ]
            for action_type in non_essential:
                self._circuit_breaker.force_open(action_type)
            # Metabolic emergency forces multiple circuit breaks - this IS motor
            # degradation: several executor types are now unreachable.
            if self._event_bus is not None:
                asyncio.create_task(
                    self._emit_motor_degradation(),
                    name="axon_motor_degradation_metabolic_emergency",
                )

    async def _on_organism_sleep(self, event: Any) -> None:
        """
        Respond to ORGANISM_SLEEP by completing in-flight actions then idling.

        The reactive adapter handles sleep queueing via SLEEP_INITIATED.
        This handler ensures any direct sleep signals also trigger idle mode.
        """
        self._logger.info("organism_sleep_received")

        # Mark the reactive adapter as sleeping (defence-in-depth)
        if self._reactive is not None and not self._reactive.is_sleeping:
            self._reactive._is_sleeping = True
            self._reactive._record_adaptation("organism_sleep_direct")

        # No need to cancel in-flight actions - the pipeline handles timeouts.
        # We just prevent new executions via the Oneiros gate and reactive adapter.

    async def _emit_evolutionary_observable(
        self,
        observable_type: str,
        value: float,
        is_novel: bool,
        metadata: dict | None = None,
    ) -> None:
        """Emit an evolutionary observable event via Synapse."""
        if self._event_bus is None:
            return
        try:
            from primitives.evolutionary import EvolutionaryObservable
            from systems.synapse.types import SynapseEvent, SynapseEventType

            obs = EvolutionaryObservable(
                source_system=SystemID.AXON,
                instance_id=self._instance_id or "",
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system="axon",
                data=obs.model_dump(mode="json"),
            )
            await self._event_bus.emit(event)
        except Exception:
            pass

    def set_fovea(self, fovea: Any) -> None:
        """
        Wire Fovea for the self-prediction loop.

        When wired, Axon calls predict_self() before each execution and
        resolve_self() after completion. If the prediction is violated,
        the resulting InternalPredictionError flows through Fovea's
        standard pipeline with the 3x precision multiplier - the organism
        notices when its actions don't produce expected outcomes.
        """
        self._fovea = fovea
        self._logger.info("fovea_wired", system="axon")

    def set_block_competition_monitor(self, monitor: Any) -> None:
        """
        Inject the BlockCompetitionMonitor (from systems.fovea.block_competition).

        Must be called by the wiring layer (main.py / core/wiring.py) after
        AxonService.initialize() when MEV config is enabled. Injecting rather
        than importing keeps Axon free of direct Fovea imports (architecture
        contract: no cross-system imports).

        If initialize() has already run and the MEV analyzer is live, this
        method wires the monitor immediately and starts polling.
        """
        self._block_competition_monitor = monitor
        if self._mev_analyzer is not None and monitor is not None:
            monitor.add_listener(self._mev_analyzer.update_competition)
            import asyncio
            asyncio.ensure_future(monitor.start())
        self._logger.info("block_competition_monitor_wired", system="axon")

    def set_oneiros(self, oneiros: Any) -> None:
        """
        Wire Oneiros sleep safety.

        When the organism is sleeping (Oneiros is in an active sleep stage),
        Axon defers execution of non-emergency intents. This prevents the
        motor cortex from acting while the cognitive system is consolidating -
        analogous to sleep paralysis in biological organisms.
        """
        self._oneiros = oneiros
        self._logger.info("oneiros_wired", system="axon")

    async def export_axon_genome(self, generation: int = 1) -> Any:
        """
        Export Axon's heritable execution intelligence as an AxonGenomeFragment.

        Extracts the top-10 action templates by success_rate from the
        introspector's per-executor stats. Each template captures the
        action pattern, confidence, and cost statistics so the child
        can warm-start with validated execution strategies.

        Returns an AxonGenomeFragment, or None if insufficient execution history.

        Called by SpawnChildExecutor at spawn time (Step 0b).
        """
        from primitives.genome_inheritance import AxonGenomeFragment, AxonTemplateSnapshot

        try:
            templates: list[AxonTemplateSnapshot] = []

            # Pull per-executor stats from the introspector
            if self._introspector is not None and hasattr(self._introspector, "get_stats"):
                stats = self._introspector.get_stats()
                executor_stats: dict[str, Any] = stats.get("per_executor", {})
            else:
                # Fall back to computing from recent outcomes
                executor_stats = self._compute_executor_stats_for_genome()

            # Build template snapshots from executor stats, ranked by success_rate
            raw_templates = []
            for action_type, data in executor_stats.items():
                if action_type == "__aggregate__":
                    continue
                success_rate = float(data.get("success_rate", 0.0))
                total = int(data.get("total_observed", data.get("total", 0)))
                if total < 5:
                    # Skip executors with too few observations - not yet reliable
                    continue
                mean_cost = float(data.get("mean_cost_usd", data.get("mean_ms", 0.0)) or 0.0)
                variance_cost = float(data.get("variance_cost_usd", 0.0))
                raw_templates.append(
                    AxonTemplateSnapshot(
                        action_pattern=action_type,
                        cached_approvals=data.get("cached_approvals", []),
                        expected_cost_mean=mean_cost,
                        expected_cost_variance=variance_cost,
                        success_rate=success_rate,
                    )
                )

            # Top 10 by success_rate
            raw_templates.sort(key=lambda t: t.success_rate, reverse=True)
            templates = raw_templates[:10]

            if not templates:
                self._logger.debug("export_axon_genome_empty", reason="no_qualified_templates")
                return None

            # Build confidence dict: max(0.5, success_rate) so inherited templates
            # are always above the floor (never misleadingly zero)
            template_confidence = {
                t.action_pattern: max(0.5, t.success_rate)
                for t in templates
            }

            # Extract circuit breaker thresholds from current config
            cb_thresholds: dict[str, int] = {}
            if self._circuit_breaker is not None:
                # Global threshold as default for all action types
                cb_thresholds["__default__"] = self._circuit_breaker.failure_threshold
                for action_type, state in self._circuit_breaker._states.items():
                    cb_thresholds[action_type] = self._circuit_breaker.failure_threshold

            fragment = AxonGenomeFragment(
                instance_id=self._instance_id,
                generation=generation,
                templates=templates,
                circuit_breaker_thresholds=cb_thresholds,
                template_confidence=template_confidence,
            )

            self._logger.info(
                "axon_genome_exported",
                template_count=len(templates),
                top_pattern=templates[0].action_pattern if templates else "",
                generation=generation,
            )

            return fragment

        except Exception as exc:
            self._logger.error("axon_genome_export_failed", error=str(exc))
            return None

    def _compute_executor_stats_for_genome(self) -> dict[str, Any]:
        """
        Compute per-executor stats from recent_outcomes for genome export.

        Used as fallback when the introspector doesn't expose get_stats().
        """
        from collections import defaultdict

        per_executor: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"success": 0, "failure": 0, "total": 0, "total_ms": 0}
        )
        for outcome in self._recent_outcomes:
            for step in outcome.step_outcomes:
                action_type = step.action_type
                per_executor[action_type]["total"] += 1
                per_executor[action_type]["total_ms"] += step.duration_ms
                if step.result.success:
                    per_executor[action_type]["success"] += 1
                else:
                    per_executor[action_type]["failure"] += 1

        result: dict[str, Any] = {}
        for action_type, counts in per_executor.items():
            total = counts["total"]
            result[action_type] = {
                "success_rate": counts["success"] / total if total > 0 else 0.0,
                "total_observed": total,
                "mean_ms": counts["total_ms"] / total if total > 0 else 0.0,
            }
        return result

    async def _initialize_from_parent_templates(self, parent_genome: Any) -> None:
        """
        Seed the child's fast-path template library from an inherited AxonGenomeFragment.

        Inserts each inherited template into the service's execution knowledge with
        a lower confidence threshold than self-learned templates:
          - Inherited:   execute with confidence ≥ 0.6
          - Self-learned: execute with confidence ≥ 0.8

        This warm-start collapses the cold-start period where a fresh child has no
        execution history and falls back to conservative defaults.

        After seeding, emits AXON_TEMPLATES_INHERITED so Evo can track the ratio
        of inherited vs. self-discovered execution strategies - a speciation metric.

        Call this from initialize() when ECODIAOS_AXON_GENOME_PAYLOAD env var is set,
        or wire the caller to call it explicitly after boot.
        """
        try:
            from primitives.genome_inheritance import AxonGenomeFragment

            if not isinstance(parent_genome, AxonGenomeFragment):
                self._logger.warning(
                    "axon_genome_seed_type_error",
                    got=type(parent_genome).__name__,
                )
                return

            if not parent_genome.templates:
                self._logger.debug("axon_genome_seed_skip", reason="no_templates_in_fragment")
                return

            # Apply circuit breaker thresholds from parent so the child starts with
            # calibrated protection levels, not bare defaults
            if self._circuit_breaker is not None and parent_genome.circuit_breaker_thresholds:
                default_threshold = parent_genome.circuit_breaker_thresholds.get(
                    "__default__", self._circuit_breaker.failure_threshold
                )
                if default_threshold != self._circuit_breaker.failure_threshold:
                    self._circuit_breaker.failure_threshold = default_threshold
                    self._logger.debug(
                        "axon_genome_cb_threshold_inherited",
                        threshold=default_threshold,
                    )

            # Seed the introspector's inherited execution knowledge
            # so fast-path confidence checks honour the lower threshold (0.6)
            inherited_action_patterns: list[str] = []
            for template in parent_genome.templates:
                confidence = parent_genome.template_confidence.get(
                    template.action_pattern, max(0.5, template.success_rate)
                )
                inherited_action_patterns.append(template.action_pattern)

                # Store on introspector as pre-seeded stats so the child's first
                # executions of these patterns start with realistic priors
                if self._introspector is not None and hasattr(
                    self._introspector, "seed_inherited_template"
                ):
                    self._introspector.seed_inherited_template(
                        action_type=template.action_pattern,
                        success_rate=template.success_rate,
                        confidence=confidence,
                        inherited_from_parent=True,
                    )

                self._logger.debug(
                    "axon_template_inherited",
                    action_pattern=template.action_pattern,
                    success_rate=template.success_rate,
                    confidence=confidence,
                    inherited=True,
                )

            # Emit AXON_TEMPLATES_INHERITED so Evo can track cold-start improvement
            if self._event_bus is not None:
                try:
                    from systems.synapse.types import SynapseEvent, SynapseEventType

                    mean_confidence = (
                        sum(
                            parent_genome.template_confidence.get(p, 0.0)
                            for p in inherited_action_patterns
                        )
                        / len(inherited_action_patterns)
                        if inherited_action_patterns
                        else 0.0
                    )

                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.AXON_TEMPLATES_INHERITED,
                        source_system="axon",
                        data={
                            "template_count": len(parent_genome.templates),
                            "action_patterns": inherited_action_patterns,
                            "generation": parent_genome.generation,
                            "inherited_confidence_mean": round(mean_confidence, 4),
                            "source_genome_id": parent_genome.genome_id,
                        },
                    ))
                except Exception as exc:
                    self._logger.debug("axon_templates_event_failed", error=str(exc))

            self._logger.info(
                "axon_templates_inherited",
                template_count=len(parent_genome.templates),
                action_patterns=inherited_action_patterns,
                generation=parent_genome.generation,
                source_genome_id=parent_genome.genome_id,
            )

        except Exception as exc:
            self._logger.error("axon_genome_seed_failed", error=str(exc))

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

        This is the main external interface - Nova calls this via IntentRouter
        after Equor has approved the Intent.

        Args:
            request: ExecutionRequest containing the approved Intent and Equor verdict.

        Returns:
            AxonOutcome with full step-level detail and outcome summary.
            Never raises - failures are captured in the outcome.
        """
        if not self._initialized or self._pipeline is None:
            raise RuntimeError(
                "AxonService.initialize() must be called before execute()"
            )

        # ── Metabolic starvation gate ─────────────────────────────
        # CRITICAL: halt all execution
        # EMERGENCY: only survival-priority intents
        if self._starvation_level == "critical":
            from primitives.common import new_id
            from systems.axon.types import ExecutionStatus
            return AxonOutcome(
                intent_id=request.intent.id,
                execution_id=new_id(),
                success=False,
                status=ExecutionStatus.REJECTED,
                failure_reason="metabolic_critical_halt",
                error="Metabolic starvation (critical) - all execution halted.",
            )
        if self._starvation_level == "emergency":
            urgency = getattr(request.intent, "urgency", 0.0)
            if urgency < 0.8:
                from primitives.common import new_id
                from systems.axon.types import ExecutionStatus
                return AxonOutcome(
                    intent_id=request.intent.id,
                    execution_id=new_id(),
                    success=False,
                    status=ExecutionStatus.REJECTED,
                    failure_reason="metabolic_emergency_low_urgency",
                    error="Metabolic emergency - only survival-priority intents executed.",
                )

        # ── Skia modulation halt ──────────────────────────────────────────
        if self._modulation_halted:
            from primitives.common import new_id
            from systems.axon.types import ExecutionStatus
            self._logger.debug("execution_skipped_modulation_halted", intent_id=request.intent.id)
            return AxonOutcome(
                intent_id=request.intent.id,
                execution_id=new_id(),
                success=False,
                status=ExecutionStatus.REJECTED,
                failure_reason="modulation_halted",
                error="Skia modulation halt - execution suspended.",
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
                action_types = [
                    step.executor for step in request.intent.plan.steps
                ]
                # Pass InternalErrorType as a string literal - avoids a direct
                # cross-system import from systems.fovea.types (AV3 fix).
                fovea_prediction_id = self._fovea.predict_self(
                    action_type=",".join(action_types) or "unknown",
                    internal_error_type="COMPETENCY",
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

        # ── Bus: notify Nova/Thymos/Fovea of incoming execution ──────
        # Fire-and-forget - never blocks or delays execution.
        # Replaces any need for direct cross-system imports of Axon types.
        await self._emit_execution_request(request, execution_id="")

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

        # ── Bus: broadcast full result to Nova/Thymos/Fovea ──────────
        # AXON_EXECUTION_RESULT is richer than ACTION_EXECUTED/ACTION_FAILED
        # and removes the last need for cross-system type imports.
        await self._emit_execution_result(outcome)

        if outcome.success:
            self._successful_executions += 1
            # Emit WALLET_TRANSFER_CONFIRMED so Memory encodes it at salience=1.0
            if self._event_bus is not None:
                await self._emit_financial_events(outcome)
            # Emit WELFARE_OUTCOME_RECORDED for care-relevant action types
            # so Telos CareTopologyEngine can update the care coverage dimension.
            if self._event_bus is not None:
                await self._emit_welfare_outcome(outcome)
        else:
            self._failed_executions += 1

        # Emit evolutionary observable for executor reliability shift
        total = self._successful_executions + self._failed_executions
        reliability = self._successful_executions / total if total > 0 else 0.0
        await self._emit_evolutionary_observable(
            observable_type="executor_reliability_shift",
            value=round(reliability, 4),
            is_novel=False,
            metadata={
                "intent_id": outcome.intent_id,
                "success": outcome.success,
                "total_executions": self._total_executions,
                "action_types": [s.action_type for s in outcome.step_outcomes],
            },
        )

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

        # ── RE training: execution decision ──
        value_gained = 0.0
        steps_completed = 0
        steps_failed = 0
        retry_count = 0
        rollback_triggered = False
        for step in outcome.step_outcomes:
            if step.result.success:
                steps_completed += 1
                if "economic_delta_usd" in step.result.data:
                    with contextlib.suppress(ValueError, TypeError):
                        value_gained += float(step.result.data["economic_delta_usd"])
            else:
                steps_failed += 1
                if "retry_count" in step.result.data:
                    with contextlib.suppress(ValueError, TypeError):
                        retry_count += int(step.result.data["retry_count"])
            if step.rolled_back:
                rollback_triggered = True

        # Derive constitutional alignment from the Equor check that approved this intent
        equor_alignment = None
        equor_verdict = "approved"
        equor_reasoning_snippet = ""
        if request.equor_check is not None:
            equor_alignment = getattr(request.equor_check, "drive_alignment", None)
            equor_verdict = request.equor_check.verdict.value if hasattr(request.equor_check.verdict, "value") else str(request.equor_check.verdict)
            equor_reasoning_snippet = (request.equor_check.reasoning or "")[:300]

        # Executor selection trace: what ran, in order, and how each did
        executor_trace_parts = []
        for i, step in enumerate(outcome.step_outcomes):
            status = "ok" if step.result.success else f"FAIL({(step.result.error or '')[:60]})"
            rb = " [rolled_back]" if step.rolled_back else ""
            executor_trace_parts.append(f"  step{i}: {step.action_type} → {status} ({step.duration_ms}ms){rb}")
        executor_trace = "\n".join(executor_trace_parts) if executor_trace_parts else "  (no steps executed)"

        # Build multi-step reasoning trace
        plan_steps_desc = ", ".join(
            step.executor for step in request.intent.plan.steps
        ) or "none"
        autonomy_level = getattr(request.intent, "autonomy_level_granted", "?")

        reasoning_trace_parts = [
            f"1. INTENT RECEIVED: goal={request.intent.goal.description[:200]!r}, steps=[{plan_steps_desc}], autonomy={autonomy_level}",
            f"2. EQUOR GATE: verdict={equor_verdict}, reasoning={equor_reasoning_snippet!r}",
            f"3. PRE-EXECUTION VALIDATION: all {len(request.intent.plan.steps)} step(s) passed param validation and autonomy check",
            f"4. STEP EXECUTION:\n{executor_trace}",
            f"5. OUTCOME: success={outcome.success}, partial={outcome.partial}, steps_completed={steps_completed}/{len(outcome.step_outcomes)}, rollback_triggered={rollback_triggered}, retry_count={retry_count}",
            f"6. VALUE: economic_delta_usd={value_gained:.4f}, duration_ms={outcome.duration_ms}",
        ]
        if not outcome.success and outcome.failure_reason:
            reasoning_trace_parts.append(f"7. FAILURE ANALYSIS: {outcome.failure_reason}")
        reasoning_trace = "\n".join(reasoning_trace_parts)

        # Alternatives considered: other executors that could have handled each step
        introspector_profiles = {}
        if self._introspector is not None:
            with contextlib.suppress(Exception):
                introspector_profiles = {
                    p["action_type"]: p
                    for p in self._introspector.get_degrading_executors()
                }
        alternatives: list[str] = []
        if outcome.failure_reason == "circuit_open":
            alternatives.append("Alternative: wait for circuit-breaker HALF_OPEN recovery (300s) then retry with same executor")
        if outcome.failure_reason in ("rate_limited", "budget_exceeded"):
            alternatives.append("Alternative: defer to next theta cycle when rate-limit window resets")
        if steps_failed > 0 and not rollback_triggered:
            alternatives.append("Alternative: continue_on_failure=True would allow subsequent steps to proceed despite partial failure")
        if rollback_triggered:
            alternatives.append("Rollback path taken: completed prior steps reversed in LIFO order; non-reversible steps (financial, external) could not be undone")
        for step in outcome.step_outcomes:
            if not step.result.success and step.action_type in introspector_profiles:
                p = introspector_profiles[step.action_type]
                alternatives.append(
                    f"Executor '{step.action_type}' degrading (success_rate={p.get('success_rate', '?'):.2f}, "
                    f"consecutive_failures={p.get('consecutive_failures', '?')}); consider fallback executor or human escalation"
                )

        # Counterfactual: for failures, reason about what a different path would have produced
        counterfactual = ""
        if not outcome.success:
            if outcome.failure_reason == "equor_constitutional_block":
                counterfactual = (
                    "If constitutional check had not blocked this intent, execution would have proceeded without "
                    "ethical oversight - risking drive misalignment. Block was correct."
                )
            elif outcome.failure_reason == "circuit_open":
                counterfactual = (
                    f"If circuit breaker had not been OPEN, the executor would have been called despite "
                    f"recent failures. Forcing the call risks propagating cascading failures to downstream systems."
                )
            elif outcome.failure_reason in ("rate_limited", "budget_exceeded"):
                counterfactual = (
                    "If rate limiting had been bypassed, the per-cycle action budget would be exceeded, "
                    "potentially exhausting external-service quotas and triggering ban/suspension."
                )
            elif rollback_triggered:
                if steps_completed > 0:
                    counterfactual = (
                        f"If rollback had not been triggered after step {steps_completed}, "
                        f"partial state mutations would have persisted - leaving {steps_completed} completed "
                        f"step(s) with no corresponding follow-through, corrupting downstream intent coherence."
                    )
            elif steps_failed > 0:
                first_fail = next((s for s in outcome.step_outcomes if not s.result.success), None)
                if first_fail is not None:
                    counterfactual = (
                        f"If executor '{first_fail.action_type}' had succeeded, "
                        f"the remaining {len(outcome.step_outcomes) - steps_completed - 1} step(s) would have proceeded. "
                        f"Root cause: {(first_fail.result.error or 'unknown')[:120]}"
                    )

        # Structured output
        step_detail = [
            {
                "action_type": s.action_type,
                "success": s.result.success,
                "duration_ms": s.duration_ms,
                "rolled_back": s.rolled_back,
                "error": (s.result.error or "")[:120] if not s.result.success else "",
            }
            for s in outcome.step_outcomes
        ]
        import json as _json
        structured_output = _json.dumps({
            "executor_used": plan_steps_desc,
            "steps_completed": steps_completed,
            "steps_total": len(outcome.step_outcomes),
            "rollback_triggered": rollback_triggered,
            "retry_count": retry_count,
            "actual_outcome_quality": 1.0 if outcome.success else (0.5 if outcome.partial else 0.0),
            "economic_delta_usd": round(value_gained, 6),
            "duration_ms": outcome.duration_ms,
            "failure_reason": outcome.failure_reason or None,
            "step_detail": step_detail,
        }, default=str)

        await self._emit_re_training_example(
            category="execution",
            instruction=(
                f"Execute Equor-approved intent (autonomy={autonomy_level}): "
                f"route [{plan_steps_desc}] executor(s), enforce safety pipeline, manage timeouts, rollback on failure."
            ),
            input_context=(
                f"intent_id={request.intent.id}, "
                f"goal={request.intent.goal.description[:200]!r}, "
                f"steps={len(request.intent.plan.steps)}, "
                f"equor_verdict={equor_verdict}, "
                f"autonomy_level={autonomy_level}"
            ),
            output=structured_output,
            outcome_quality=1.0 if outcome.success else (0.5 if outcome.partial else 0.0),
            episode_id=outcome.episode_id or "",
            latency_ms=outcome.duration_ms or 0,
            reasoning_trace=reasoning_trace,
            alternatives=alternatives,
            constitutional_alignment=equor_alignment,
            counterfactual=counterfactual,
        )

        # ── Self-healing: auto-evict persistently failing executors ──
        await self._check_self_healing(outcome)

        # ── Loop 2: track motor performance and emit degradation if needed ──
        executor_types = [s.action_type for s in outcome.step_outcomes] if outcome.step_outcomes else []
        if not executor_types and outcome.failure_reason == "circuit_open":
            # Circuit-open short-circuit produces no step_outcomes - derive executor
            # types from the intent's plan so that OPEN events count toward the
            # rolling degradation window. This allows MOTOR_DEGRADATION_DETECTED to
            # fire when a circuit stays open, which is exactly the signal Nova needs.
            executor_types = [
                step.executor
                for step in request.intent.plan.steps
                if step.executor
            ]
        should_alert = self._performance_monitor.record(
            success=outcome.success,
            error=outcome.error or outcome.failure_reason or "",
            executor_type=executor_types[0] if executor_types else "",
        )
        if should_alert and self._event_bus is not None:
            await self._emit_motor_degradation()

        self._recent_outcomes.append(outcome)
        return outcome

    async def _emit_execution_request(
        self,
        request: ExecutionRequest,
        execution_id: str,
    ) -> None:
        """
        Emit AXON_EXECUTION_REQUEST so Nova, Thymos, and Fovea can observe
        the upcoming action without requiring direct imports from Axon.

        Emitted after the Equor gate passes, before pipeline execution.
        Fires-and-forgets - never blocks execution.
        """
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            intent = request.intent
            action_types = [step.executor for step in intent.plan.steps]
            risky = any(
                at in ("wallet_transfer", "defi_yield", "phantom_liquidity", "spawn_child")
                for at in action_types
            )
            autonomy_level = (
                intent.autonomy_level_granted.value
                if hasattr(intent.autonomy_level_granted, "value")
                else str(intent.autonomy_level_granted)
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.AXON_EXECUTION_REQUEST,
                source_system="axon",
                data={
                    "intent_id": intent.id,
                    "execution_id": execution_id,
                    "goal": intent.goal.description[:200],
                    "action_types": action_types,
                    "step_count": len(intent.plan.steps),
                    "estimated_budget_usd": float(
                        getattr(intent, "estimated_cost_usd", 0.0) or 0.0
                    ),
                    "risky": risky,
                    "autonomy_level": autonomy_level,
                },
            ))
        except Exception:
            self._logger.debug("axon_execution_request_emit_failed", exc_info=True)

    async def _emit_execution_result(self, outcome: AxonOutcome) -> None:
        """
        Emit AXON_EXECUTION_RESULT after pipeline completion.

        Richer than ACTION_EXECUTED/ACTION_FAILED - carries the full
        result shape so Nova can update Thompson scores and Fovea can
        resolve competency prediction errors without importing Axon types.
        """
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            action_types = [s.action_type for s in outcome.step_outcomes]
            economic_delta = 0.0
            for step in outcome.step_outcomes:
                if step.result.success and "economic_delta_usd" in step.result.data:
                    try:
                        economic_delta += float(step.result.data["economic_delta_usd"])
                    except (ValueError, TypeError):
                        pass

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.AXON_EXECUTION_RESULT,
                source_system="axon",
                data={
                    "intent_id": outcome.intent_id,
                    "execution_id": outcome.execution_id,
                    "success": outcome.success,
                    "failure_reason": outcome.failure_reason,
                    "duration_ms": outcome.duration_ms,
                    "step_count": len(outcome.step_outcomes),
                    "action_types": action_types,
                    "economic_delta_usd": economic_delta,
                },
            ))
        except Exception:
            self._logger.debug("axon_execution_result_emit_failed", exc_info=True)

    async def _emit_financial_events(self, outcome: AxonOutcome) -> None:
        """Emit WALLET_TRANSFER_CONFIRMED for any successful wallet_transfer steps,
        and YIELD_DEPLOYMENT_RESULT for any defi_yield steps (success or failure)."""
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

            elif step.action_type == "defi_yield":
                # Emit YIELD_DEPLOYMENT_RESULT so any Oikos future waiting on
                # request_id can resolve.  The request_id comes from the step
                # params (stored by the pipeline in step.params if available),
                # or falls back to empty string (Oikos ignores non-matching ids).
                step_data = dict(step.result.data) if step.result.data else {}
                request_id = step_data.pop("request_id", "")
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.YIELD_DEPLOYMENT_RESULT,
                        source_system="axon",
                        data={
                            "request_id": request_id,
                            "success": step.result.success,
                            "tx_hash": step_data.get("tx_hash", ""),
                            "protocol": step_data.get("protocol", ""),
                            "amount": step_data.get("amount", ""),
                            "action": step_data.get("action", ""),
                            "error": step.result.error or "",
                            "execution_id": outcome.execution_id,
                        },
                    ))
                    self._logger.info(
                        "yield_deployment_result_emitted",
                        success=step.result.success,
                        protocol=step_data.get("protocol", ""),
                        tx_hash=step_data.get("tx_hash", ""),
                    )
                except Exception as exc:
                    self._logger.error(
                        "yield_deployment_result_emit_failed", error=str(exc)
                    )

    async def _emit_motor_degradation(self) -> None:
        """Emit MOTOR_DEGRADATION_DETECTED when rolling success rate drops below threshold."""
        from systems.synapse.types import SynapseEvent, SynapseEventType

        data = self._performance_monitor.build_event_data()
        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.MOTOR_DEGRADATION_DETECTED,
                source_system="axon",
                data=data,
            ))
            self._logger.warning(
                "motor_degradation_event_emitted",
                success_rate=data["success_rate"],
                affected_executors=data["affected_executors"],
            )
        except Exception as exc:
            self._logger.warning("motor_degradation_event_emit_failed", error=str(exc))

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

            # Emit evolutionary observable for circuit breaker trip
            await self._emit_evolutionary_observable(
                observable_type="circuit_breaker_trip",
                value=profile.get("success_rate", 0),
                is_novel=False,
                metadata={
                    "action_type": action_type,
                    "consecutive_failures": consecutive,
                    "evicted": True,
                },
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
        # Emit evolutionary observable for new executor discovery
        import asyncio as _asyncio
        _asyncio.create_task(
            self._emit_evolutionary_observable(
                observable_type="new_executor_discovered",
                value=1.0,
                is_novel=True,
                metadata={
                    "action_type": executor.action_type,
                    "executor_class": type(executor).__name__,
                },
            ),
            name=f"axon_evo_new_executor_{executor.action_type}",
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
        """Graceful shutdown - log final stats."""
        if self._bus is not None:
            self._bus.deregister(Executor)

        # Tear down MEV subsystems
        if self._block_competition_monitor is not None:
            await self._block_competition_monitor.stop()
        if self._mev_analyzer is not None:
            await self._mev_analyzer.close()

        # Cancel background PR merge polling loop
        if self._pr_poll_task is not None:
            self._pr_poll_task.cancel()
            self._pr_poll_task = None

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

    # ── Background PR merge polling loop ──────────────────────────────────────

    async def _pr_merge_poll_loop(self) -> None:
        """
        Background coroutine - polls pending bounty PRs for merge status every
        30 minutes.

        Flow:
          1. Sleep 30 minutes (first sleep avoids race with initialize())
          2. Scan Redis for all ``axon:bounty_submit:pr:*`` keys
          3. Build ``pending_prs`` list from stored JSON payloads
          4. Emit EQUOR_ECONOMIC_INTENT and await EQUOR_ECONOMIC_PERMIT (30s).
             If Equor denies or times out, skip the cycle - never auto-permit.
          5. If permitted, invoke MonitorPRsExecutor with a ConstitutionalCheck
             whose verdict is explicitly set from the Equor PERMIT response.
          6. Repeat until cancelled

        Never raises - exceptions are logged and the loop continues.
        supervised_task() in initialize() provides restart if the loop exits
        abnormally.
        """
        import asyncio as _asyncio

        _PR_KEY_PATTERN = "axon:bounty_submit:pr:*"

        while True:
            try:
                await _asyncio.sleep(self._pr_poll_interval_s)
            except _asyncio.CancelledError:
                return

            if self._redis is None:
                continue

            # ── Collect pending PRs from Redis ────────────────────────────
            try:
                pending_prs: list[dict] = []
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor, match=_PR_KEY_PATTERN, count=100
                    )
                    for key in keys:
                        try:
                            payload = await self._redis.get_json(key)
                            if payload and isinstance(payload, dict):
                                # Ensure required fields present before adding
                                if all(
                                    k in payload
                                    for k in ("pr_url", "bounty_id")
                                ):
                                    pending_prs.append(
                                        {
                                            "entity_id": payload["bounty_id"],
                                            "pr_url": payload["pr_url"],
                                            "reward": payload.get(
                                                "reward", "unknown"
                                            ),
                                            "bounty_id": payload["bounty_id"],
                                        }
                                    )
                        except Exception as _key_exc:
                            self._logger.warning(
                                "pr_poll_key_read_failed",
                                key=key,
                                error=str(_key_exc),
                            )
                    if cursor == 0:
                        break
            except Exception as exc:
                self._logger.warning(
                    "pr_poll_redis_scan_failed", error=str(exc)
                )
                continue

            if not pending_prs:
                continue

            # ── Constitutional gate before polling ────────────────────
            # MonitorPRsExecutor is Level 1 (read-only) but still requires
            # Equor consent - we must not fabricate an APPROVED verdict.
            # Emit EQUOR_ECONOMIC_INTENT and await EQUOR_ECONOMIC_PERMIT (30s).
            # Read-only polling has no capital at risk so Equor approves quickly.
            try:
                if self._event_bus is None:
                    self._logger.warning("pr_poll_no_event_bus_skip")
                    continue

                import asyncio as _poll_asyncio
                from systems.synapse.types import SynapseEvent, SynapseEventType as _PSET

                _poll_action_id = f"pr_poll_background:{id(pending_prs)}"
                _poll_fut: _poll_asyncio.Future[bool] = (
                    _poll_asyncio.get_event_loop().create_future()
                )

                async def _on_poll_permit(event: Any) -> None:  # noqa: ANN001
                    _d = getattr(event, "data", {}) or {}
                    if _d.get("action_id") != _poll_action_id:
                        return
                    _v = str(_d.get("verdict", "PERMIT")).upper()
                    if not _poll_fut.done():
                        _poll_fut.set_result(_v == "PERMIT")

                self._event_bus.subscribe(_PSET.EQUOR_ECONOMIC_PERMIT, _on_poll_permit)

                await self._event_bus.emit(SynapseEvent(
                    event_type=_PSET.EQUOR_ECONOMIC_INTENT,
                    source_system="axon.pr_poll",
                    data={
                        "action_id": _poll_action_id,
                        "mutation_type": "background_pr_status_poll",
                        "amount_usd": "0",
                        "from_account": "none",
                        "to_account": "github_api_read_only",
                        "rationale": (
                            f"Background 30-minute poll of {len(pending_prs)} "
                            "pending bounty PRs. Level 1 read-only GitHub API "
                            "calls - no capital at risk, no writes."
                        ),
                        "pr_count": len(pending_prs),
                    },
                ))

                try:
                    _poll_permitted = await _poll_asyncio.wait_for(
                        _poll_fut, timeout=30.0
                    )
                except _poll_asyncio.TimeoutError:
                    self._logger.warning(
                        "pr_poll_equor_timeout",
                        note="Equor did not respond in 30s; skipping poll cycle",
                    )
                    continue  # Skip this cycle rather than auto-permitting

                if not _poll_permitted:
                    self._logger.info(
                        "pr_poll_equor_denied",
                        pr_count=len(pending_prs),
                    )
                    continue

            except Exception as _gate_exc:
                self._logger.warning(
                    "pr_poll_equor_gate_error", error=str(_gate_exc)
                )
                continue

            # ── Invoke MonitorPRsExecutor ─────────────────────────────────
            try:
                executor = (
                    self._registry.get("monitor_prs") if self._registry else None
                )
                if executor is None:
                    self._logger.warning("pr_poll_no_monitor_executor")
                    continue

                from systems.axon.types import ExecutionContext
                from primitives.intent import GoalDescriptor, Intent
                from primitives.constitutional import ConstitutionalCheck
                from primitives.common import Verdict

                _intent = Intent(
                    goal=GoalDescriptor(
                        description="background_pr_status_poll",
                        target_domain="bounty",
                    ),
                )
                # Verdict explicitly set - Equor gate above confirmed PERMIT.
                _check = ConstitutionalCheck(
                    intent_id=_intent.id,
                    verdict=Verdict.APPROVED,
                    confidence=1.0,
                    reasoning=(
                        "Background PR poll: Equor EQUOR_ECONOMIC_PERMIT received "
                        f"(action_id={_poll_action_id}). Level 1 read-only."
                    ),
                )
                _ctx = ExecutionContext(
                    intent=_intent,
                    equor_check=_check,
                    instance_id=getattr(self, "_instance_id", "eos-default"),
                )
                result = await executor.execute(
                    {"pending_prs": pending_prs}, _ctx
                )
                self._logger.debug(
                    "pr_poll_cycle_complete",
                    pr_count=len(pending_prs),
                    success=result.success,
                )
            except Exception as exc:
                self._logger.warning(
                    "pr_poll_executor_failed", error=str(exc)
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

    async def _on_theta_cycle_complete(self, event: Any) -> None:
        """
        Called on every CYCLE_COMPLETED event.

        Every cycle:  emit AXON_CAPABILITY_SNAPSHOT - Nova needs to know what
                      is executable *right now* before deliberating.
        Every 50:     emit AXON_TELEMETRY_REPORT - deep introspector state for
                      Evo/Fovea/RE learning loops.

        Also runs in-cycle adaptive learning: if the introspector detects a
        degrading executor, Axon proactively tightens its own rate limit and
        proposes a pre-emptive circuit half-open - the organism doesn't wait
        for 5 hard failures before protecting itself.
        """
        self._telemetry_cycle_counter += 1

        # ── Capability snapshot - every cycle (cheap, critical for planning) ──
        asyncio.create_task(self._emit_capability_snapshot())

        # ── In-cycle adaptive learning - act on introspection data NOW ──
        self._apply_introspective_adaptations()

        # ── Deep telemetry - every 50 cycles ──
        if self._telemetry_cycle_counter % self._telemetry_emit_interval == 0:
            asyncio.create_task(self._emit_axon_telemetry_report())

    async def _emit_axon_telemetry_report(self) -> None:
        """
        Broadcast the full Axon introspector state as AXON_TELEMETRY_REPORT.

        Drains pending recommendations (degradation warnings, failure hotspots)
        and publishes executor profiles, pattern data, circuit-breaker states,
        and aggregate stats.  All systems that want to plan around motor-cortex
        health (Nova deliberation, Evo hypothesis generation, Fovea competency
        model) should subscribe to this event rather than polling axon.stats.
        """
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        if not hasattr(SynapseEventType, "AXON_TELEMETRY_REPORT"):
            return

        # Full introspector report - executor profiles + patterns
        report = self._introspector.full_report

        # Drain pending recommendations so they're surfaced exactly once
        recommendations = self._introspector.drain_recommendations()

        # Circuit-breaker states (CLOSED / OPEN / HALF_OPEN per executor)
        cb_states: dict[str, str] = {}
        if self._circuit_breaker is not None and hasattr(self._circuit_breaker, "_states"):
            for action_type, state in self._circuit_breaker._states.items():
                cb_states[action_type] = state.value if hasattr(state, "value") else str(state)

        payload: dict[str, Any] = {
            "executor_profiles": report.get("executor_profiles", {}),
            "reliable_patterns": report.get("reliable_patterns", []),
            "failure_hotspots": report.get("failure_hotspots", []),
            "recommendations": recommendations,
            "stats": report.get("stats", {}),
            "circuit_breaker_states": cb_states,
            "budget_utilisation": self._budget.utilisation if self._budget else 0.0,
            "starvation_level": self._starvation_level,
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
        }

        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.AXON_TELEMETRY_REPORT,
                source_system="axon",
                data=payload,
            ))
        except Exception as exc:
            self._logger.debug("axon_telemetry_report_emit_failed", error=str(exc))

    # ── Capability Snapshot (every cycle) ─────────────────────────────

    async def _emit_capability_snapshot(self) -> None:
        """
        Emit AXON_CAPABILITY_SNAPSHOT every theta cycle so Nova, Evo, and any
        planning system can see exactly what Axon *can* do right now.

        Payload per executor: action_type, description, circuit_breaker_status,
        rate_limit_remaining, success_rate, is_degrading, required_autonomy.
        Plus: budget_remaining, budget_max, concurrent_remaining, is_sleeping,
        degraded_systems, starvation_level.
        """
        if self._event_bus is None:
            return
        executors: list[dict[str, object]] = []
        if self._registry is not None:
            for cap in self._registry.capabilities():
                action_type = cap.get("action_type", "")
                # Circuit breaker status
                cb_status = "CLOSED"
                if self._circuit_breaker is not None:
                    cb_status = self._circuit_breaker.status(action_type)
                # Rate limit remaining
                rl_remaining: int | None = None
                if self._rate_limiter is not None and cap.get("rate_limit") is not None:
                    rl_dict = cap.get("rate_limit")
                    # Reconstruct RateLimit from capability dict
                    rl = RateLimit(
                        max_calls=int(rl_dict.get("max_calls", 10)),
                        window_seconds=int(rl_dict.get("window_seconds", 60))
                    )
                    rl_remaining = self._rate_limiter.remaining(action_type, rl)
                # Introspector profile
                profile = self._introspector.get_executor_profile(action_type)
                executors.append({
                    "action_type": action_type,
                    "description": cap.get("description", ""),
                    "required_autonomy": cap.get("required_autonomy", 1),
                    "reversible": cap.get("reversible", False),
                    "circuit_breaker_status": cb_status,
                    "rate_limit_remaining": rl_remaining,
                    "success_rate": profile.get("success_rate", 1.0) if profile else 1.0,
                    "is_degrading": profile.get("is_degrading", False) if profile else False,
                })

        budget_remaining = 0
        budget_max = 5
        concurrent_remaining = 3
        if self._budget is not None:
            budget_max = self._budget.action_budget.effective_max_actions()
            budget_remaining = max(0, budget_max - self._budget._actions_this_cycle)
            concurrent_remaining = max(0, self._budget.action_budget.effective_max_concurrent() - self._budget._concurrent_executions)

        is_sleeping = False
        degraded_systems: list[str] = []
        if self._reactive is not None:
            is_sleeping = self._reactive.is_sleeping
            degraded_systems = list(self._reactive.degraded_systems)

        payload = {
            "executors": executors,
            "budget_remaining": budget_remaining,
            "budget_max": budget_max,
            "concurrent_remaining": concurrent_remaining,
            "is_sleeping": is_sleeping,
            "degraded_systems": degraded_systems,
            "starvation_level": self._starvation_level,
            "budget_override_active": self._budget_override_active,
            "budget_override_cycles_remaining": self._budget_override_cycles_remaining,
        }
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.AXON_CAPABILITY_SNAPSHOT,
                source_system="axon",
                data=payload,
            ))
        except Exception:
            pass  # best-effort, every cycle - never block

    # ── In-cycle Adaptive Learning ────────────────────────────────────

    def _apply_introspective_adaptations(self) -> None:
        """
        Act on introspection data NOW - not just observe.

        Every theta cycle:
        1. Degrading executors (success_rate < 0.5, consecutive_failures >= 3)
           → tighten rate limit to 0.5x
        2. Severely failing executors (consecutive_failures >= 5)
           → force circuit breaker to HALF_OPEN (let one probe through)
        3. Recovered executors (was degrading, now success_rate > 0.7)
           → clear rate limit multiplier back to 1.0
        4. Tick budget override countdown
        """
        if not hasattr(self._introspector, "_profiles"):
            return

        for action_type, profile in self._introspector._profiles.items():
            is_degrading = getattr(profile, "is_degrading", False)
            success_rate = getattr(profile, "success_rate", 1.0)
            consecutive_failures = getattr(profile, "consecutive_failures", 0)

            if is_degrading and self._rate_limiter is not None:
                # Throttle degrading executors - let them breathe
                self._rate_limiter.set_multiplier(action_type, 0.5)
                self._logger.debug(
                    "adaptive_rate_limit_tightened",
                    action_type=action_type,
                    success_rate=success_rate,
                )

            if consecutive_failures >= 5 and self._circuit_breaker is not None:
                # Force a probe - don't let it stay OPEN forever
                if self._circuit_breaker.status(action_type) == "OPEN":
                    self._circuit_breaker.force_half_open(action_type)
                    self._logger.info(
                        "adaptive_circuit_breaker_probe",
                        action_type=action_type,
                        consecutive_failures=consecutive_failures,
                    )

            if not is_degrading and success_rate > 0.7 and self._rate_limiter is not None:
                # Recovered - restore normal rate
                self._rate_limiter.clear_multiplier(action_type)

        # Tick budget override countdown
        self._tick_budget_override()

        # Tick temporary ActionBudget expansions (from ACTION_BUDGET_EXPANSION_RESPONSE)
        if self._budget is not None:
            self._budget.tick_expansions()

    # ── Equor Budget Override ─────────────────────────────────────────

    async def _on_equor_budget_override(self, event: Any) -> None:
        """
        Handle EQUOR_BUDGET_OVERRIDE - Equor can temporarily increase (or
        decrease) the per-cycle action budget during emergencies.

        Payload: multiplier (float, 0.1–5.0), reason (str),
                 expires_after_cycles (int), authorized_by (str).

        Applies via ActionBudget.apply_expansion() so the change is tracked
        alongside field-level expansions and auto-expires correctly.
        """
        data = getattr(event, "data", {}) or {}
        multiplier = data.get("multiplier", 1.0)
        reason = data.get("reason", "unspecified")
        expires_after = data.get("expires_after_cycles", 10)
        authorized_by = data.get("authorized_by", "equor")

        # Validate multiplier bounds
        multiplier = max(0.1, min(5.0, float(multiplier)))
        expires_after = max(1, min(500, int(expires_after)))

        if self._budget is None:
            self._logger.warning("budget_override_no_budget_tracker")
            return

        current_max = self._budget.action_budget.effective_max_actions()
        new_max = max(1, int(current_max * multiplier))

        self._budget.action_budget.apply_expansion(
            field="max_actions_per_cycle",
            approved_value=new_max,
            duration_cycles=expires_after,
            approved_by=authorized_by,
        )
        # Legacy state flags kept for external code that reads them
        self._budget_override_active = True
        self._budget_override_cycles_remaining = expires_after
        self._budget_override_reason = reason

        self._logger.info(
            "budget_override_applied",
            multiplier=multiplier,
            new_max_actions=new_max,
            original_max=current_max,
            expires_after_cycles=expires_after,
            reason=reason,
            authorized_by=authorized_by,
        )

    def _tick_budget_override(self) -> None:
        """Decrement legacy budget override countdown flag."""
        if not self._budget_override_active:
            return
        self._budget_override_cycles_remaining -= 1
        if self._budget_override_cycles_remaining <= 0:
            # Actual expiry is handled by ActionBudget.check_expirations() via tick_expansions()
            self._budget_override_active = False
            self._budget_override_reason = ""
            self._logger.info("budget_override_expired")

    async def _on_budget_expansion_response(self, event: Any) -> None:
        """
        Handle ACTION_BUDGET_EXPANSION_RESPONSE from Equor.

        Payload: request_id, field, approved (bool), approved_value (int),
                 denied_reason (str), duration_cycles (int), authorized_by (str).

        Applies the approved expansion to ActionBudget; logs denial.
        """
        if self._budget is None:
            return

        data = getattr(event, "data", {}) or {}
        approved: bool = bool(data.get("approved", False))
        field: str = str(data.get("field", ""))
        approved_value: int = int(data.get("approved_value", 0))
        duration_cycles: int = int(data.get("duration_cycles", 20))
        authorized_by: str = str(data.get("authorized_by", "equor"))
        request_id: str = str(data.get("request_id", ""))
        denied_reason: str = str(data.get("denied_reason", ""))

        if not approved:
            self._logger.info(
                "budget_expansion_denied",
                request_id=request_id,
                field=field,
                reason=denied_reason,
            )
            return

        if field not in ("max_actions_per_cycle", "max_concurrent_executions", "max_api_calls_per_minute"):
            self._logger.warning("budget_expansion_unknown_field", field=field)
            return

        self._budget.action_budget.apply_expansion(
            field=field,
            approved_value=approved_value,
            duration_cycles=duration_cycles,
            approved_by=authorized_by,
        )
        self._logger.info(
            "budget_expansion_applied",
            request_id=request_id,
            field=field,
            approved_value=approved_value,
            duration_cycles=duration_cycles,
            authorized_by=authorized_by,
        )

    # ── System Modulation ─────────────────────────────────────────────

    async def _on_system_modulation(self, event: Any) -> None:
        """Handle SYSTEM_MODULATION from Skia's VitalityCoordinator."""
        data = getattr(event, "data", {}) or {}
        halt_systems: list[str] = data.get("halt_systems", [])
        modulate: dict = data.get("modulate", {})
        modulation_id: str = data.get("modulation_id", "")

        if "axon" in halt_systems:
            self._modulation_halted = True
            self._logger.warning("system_modulation_halted", modulation_id=modulation_id)
        elif "axon" in modulate:
            self._modulation_halted = False
            self._apply_modulation_directives(modulate["axon"])

        if self._event_bus is not None:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.SYSTEM_MODULATION_ACK,
                source_system="axon",
                data={
                    "system": "axon",
                    "modulation_id": modulation_id,
                    "halted": self._modulation_halted,
                },
            ))

    def _apply_modulation_directives(self, directives: dict) -> None:
        """Apply Skia modulation directives to Axon runtime state."""
        # No specific Skia directives defined for axon - halt-only modulation
        self._logger.info("system_modulation_directives_applied", directives=directives)

    # ── Evo ADJUST_BUDGET tunable baseline parameters ─────────────────

    async def _on_evo_adjust_budget(self, event: Any) -> None:
        """
        Handle EVO_ADJUST_BUDGET targeting Axon's three Equor-negotiable baseline
        budget parameters. Only applied when Evo confidence > 0.75.

        Tunable baseline parameters (these set the *default* limit that ActionBudget
        starts from each restart - temporary expansions from Equor still stack on top):
          max_actions_per_cycle     : [2, 15]
          max_concurrent_executions : [1, 8]
          max_api_calls_per_minute  : [10, 90]

        Applying these changes updates the live ActionBudget baseline AND the config,
        so restarts inherit the evolved value. Emits AXON_PARAMETER_ADJUSTED so Evo
        can score the hypothesis outcome.
        """
        if self._budget is None or self._config is None:
            return

        data = getattr(event, "data", {}) or {}
        parameter_name = str(data.get("parameter_name", ""))
        new_value_raw = data.get("new_value")
        confidence = float(data.get("confidence", 0.0))
        hypothesis_id = str(data.get("hypothesis_id", ""))
        target_system = str(data.get("target_system", ""))

        # Only handle events explicitly targeting axon
        if target_system and target_system != "axon":
            return

        if confidence <= 0.75:
            self._logger.debug(
                "axon_evo_adjust_budget_skip_low_confidence",
                parameter=parameter_name,
                confidence=confidence,
            )
            return

        # Axon baseline budget parameter bounds: (lo, hi) - integers
        _PARAM_BOUNDS: dict[str, tuple[int, int]] = {
            "max_actions_per_cycle": (2, 15),
            "max_concurrent_executions": (1, 8),
            "max_api_calls_per_minute": (10, 90),
        }

        if parameter_name not in _PARAM_BOUNDS:
            return

        lo, hi = _PARAM_BOUNDS[parameter_name]
        try:
            new_value = int(round(float(new_value_raw)))
        except (TypeError, ValueError):
            self._logger.warning(
                "axon_evo_adjust_budget_invalid_value",
                parameter=parameter_name,
                raw=new_value_raw,
            )
            return

        new_value = max(lo, min(hi, new_value))

        # Apply to live ActionBudget baseline
        old_value = getattr(self._budget.action_budget, parameter_name, new_value)
        setattr(self._budget.action_budget, parameter_name, new_value)

        # Propagate to config so restarts inherit the evolved value
        if hasattr(self._config, parameter_name):
            setattr(self._config, parameter_name, new_value)

        self._logger.info(
            "axon_evo_budget_adjusted",
            parameter=parameter_name,
            old_value=old_value,
            new_value=new_value,
            confidence=confidence,
            hypothesis_id=hypothesis_id,
        )

        # Emit outcome so Evo can score the hypothesis
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                if hasattr(SynapseEventType, "AXON_PARAMETER_ADJUSTED"):
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.AXON_PARAMETER_ADJUSTED,
                        source_system="axon",
                        data={
                            "parameter_name": parameter_name,
                            "old_value": old_value,
                            "new_value": new_value,
                            "hypothesis_id": hypothesis_id,
                            "confidence": confidence,
                        },
                    ))
            except Exception:
                pass

    # ── Executor Request (organism asks for new capability) ───────────

    async def emit_executor_request(
        self,
        action_type: str,
        description: str,
        context: dict[str, object] | None = None,
        urgency: float = 0.5,
        estimated_risk_tier: str = "medium",
        requesting_system: str = "nova",
    ) -> None:
        """
        Public method: any system can tell Axon 'I need an executor that
        doesn't exist yet'. Axon emits AXON_EXECUTOR_REQUEST - Evo picks it
        up and generates an EVOLUTION_CANDIDATE for Simula to synthesize.

        This closes the gap: previously the organism could not express the
        desire for capabilities it didn't have.
        """
        if self._event_bus is None:
            self._logger.warning("executor_request_no_event_bus", action_type=action_type)
            return

        payload = {
            "action_type": action_type,
            "description": description,
            "context": context or {},
            "urgency": urgency,
            "estimated_risk_tier": estimated_risk_tier,
            "requesting_system": requesting_system,
        }
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.AXON_EXECUTOR_REQUEST,
                source_system="axon",
                data=payload,
                salience=urgency,
            ))
            self._logger.info(
                "executor_request_emitted",
                action_type=action_type,
                urgency=urgency,
                requesting_system=requesting_system,
            )
        except Exception as exc:
            self._logger.warning("executor_request_emit_failed", error=str(exc))

    # ── Intent Pivot (mid-execution replanning) ───────────────────────

    async def _emit_intent_pivot(
        self,
        intent_id: str,
        execution_id: str,
        failed_step_index: int,
        failed_action_type: str,
        failure_reason: str,
        remaining_steps: list[dict[str, object]],
        fallback_goal: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """
        Emit AXON_INTENT_PIVOT when a step fails and the organism should
        replan rather than simply abort. Nova subscribes and can inject a
        revised step sequence mid-flight.

        This closes the binary abort/continue gap: the organism can now
        pivot dynamically during execution.
        """
        if self._event_bus is None:
            return

        payload = {
            "intent_id": intent_id,
            "execution_id": execution_id,
            "failed_step_index": failed_step_index,
            "failed_action_type": failed_action_type,
            "failure_reason": failure_reason,
            "remaining_steps": remaining_steps,
            "fallback_goal": fallback_goal,
            "context": context or {},
        }
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.AXON_INTENT_PIVOT,
                source_system="axon",
                data=payload,
                salience=0.8,  # pivots are always notable
            ))
            self._logger.info(
                "intent_pivot_emitted",
                intent_id=intent_id,
                failed_step=failed_step_index,
                failed_action=failed_action_type,
            )
        except Exception as exc:
            self._logger.debug("intent_pivot_emit_failed", error=str(exc))

    # ── Welfare Outcome Recording (Spec 18 §XIII - Telos CareTopology) ───────

    # Action types that directly affect wellbeing and map to the Care drive
    _CARE_ACTION_TYPES: frozenset[str] = frozenset({
        "send_notification",
        "send_email",
        "send_telegram",
        "respond_text",
        "community_engage",
        "federation_send",
        "publish_content",
        "establish_entity",   # legal sovereignty is a welfare enabler
        "store_insight",
        "update_goal",
    })

    async def _emit_welfare_outcome(self, outcome: AxonOutcome) -> None:
        """
        Emit WELFARE_OUTCOME_RECORDED after a successful care-relevant action.

        Telos CareTopologyEngine subscribes to aggregate care coverage across
        executed actions (Spec 18 §XIII). Emitting this event after each
        care-category action lets Telos track whether the organism is meeting
        its care obligations or drifting toward self-serving behaviour.

        Only emitted for action types in _CARE_ACTION_TYPES and only on success
        - failed attempts don't constitute welfare outcomes.
        """
        if self._event_bus is None:
            return

        # Collect care-relevant steps from the outcome
        care_steps = [
            s for s in outcome.step_outcomes
            if s.action_type in self._CARE_ACTION_TYPES and s.result.success
        ]
        if not care_steps:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            for step in care_steps:
                # Derive beneficiary from step result data (best-effort)
                beneficiary = str(
                    step.result.data.get("recipient", "")
                    or step.result.data.get("chat_id", "")
                    or step.result.data.get("to", "")
                    or "unknown"
                )

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.WELFARE_OUTCOME_RECORDED,
                    source_system="axon",
                    data={
                        "intent_id": outcome.intent_id,
                        "execution_id": outcome.execution_id,
                        "action_type": step.action_type,
                        "beneficiary": beneficiary[:200],
                        "care_quality": 1.0,   # successful completion = full care score
                        "duration_ms": step.duration_ms,
                        "episode_id": outcome.episode_id or "",
                    },
                ))
        except Exception:
            self._logger.debug("welfare_outcome_emit_failed", exc_info=True)

    # ── VULNERABILITY_CONFIRMED subscriber (Simula → Axon defensive response) ──

    async def _on_vulnerability_confirmed(self, event: Any) -> None:
        """
        Handle VULNERABILITY_CONFIRMED emitted by Simula after PoC validation.

        When Simula confirms a real vulnerability in a running executor or
        subsystem, Axon must defensively tighten its safety posture:
          1. Force-open the circuit breaker for any affected executor
          2. Tighten rate limits across financial executors (50% reduction)
          3. Log the vulnerability so Thymos can escalate if needed

        The organism should not continue executing risky actions while a
        known vulnerability is unpatched - this is the Care drive applying
        to Axon's own operational integrity.
        """
        data = getattr(event, "data", {}) or {}
        vuln_id = str(data.get("vulnerability_id", data.get("vuln_id", "")))
        affected_system = str(data.get("affected_system", data.get("system", "")))
        severity = str(data.get("severity", "medium")).lower()
        description = str(data.get("description", ""))[:300]
        affected_executor = str(data.get("affected_executor", data.get("executor", "")))

        self._logger.warning(
            "vulnerability_confirmed_received",
            vuln_id=vuln_id,
            affected_system=affected_system,
            affected_executor=affected_executor,
            severity=severity,
            description=description,
        )

        # Step 1: Force-open circuit breaker for the specific affected executor
        if affected_executor and self._circuit_breaker is not None:
            self._circuit_breaker.force_open(affected_executor)
            self._logger.warning(
                "circuit_breaker_forced_open_vulnerability",
                executor=affected_executor,
                vuln_id=vuln_id,
            )

        # Step 2: For HIGH/CRITICAL severity, tighten financial executor rate limits
        if severity in ("high", "critical") and self._rate_limiter is not None:
            financial_executors = [
                "wallet_transfer", "defi_yield", "phantom_liquidity",
                "deploy_asset", "request_funding",
            ]
            for action_type in financial_executors:
                if hasattr(self._rate_limiter, "set_multiplier"):
                    self._rate_limiter.set_multiplier(action_type, 0.5)
            self._logger.warning(
                "financial_rate_limits_tightened_vulnerability",
                vuln_id=vuln_id,
                severity=severity,
                executors=financial_executors,
            )

        # Step 3: Emit MOTOR_DEGRADATION_DETECTED so Nova re-evaluates planning
        # A confirmed vulnerability IS motor degradation - the organism cannot
        # safely rely on potentially compromised execution paths.
        if self._event_bus is not None and severity in ("high", "critical"):
            asyncio.create_task(
                self._emit_motor_degradation(),
                name=f"axon_vuln_degradation_{vuln_id[:16]}",
            )
