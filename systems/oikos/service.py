"""
EcodiaOS — Oikos Service (Phases 16a + 16d + 16e + 16h + 16k + 16l)

The organism's economic engine. Oikos maintains a real-time EconomicState
by subscribing to Synapse metabolic events and WalletClient balance updates.
It is the single source of truth for: "Can we afford to keep running?"

Phase 16a (Metabolism):
  - Subscribes to Synapse EventBus for METABOLIC_PRESSURE, REVENUE_INJECTED,
    and WALLET_TRANSFER_CONFIRMED events.
  - Periodically polls WalletClient for on-chain balance (liquid_balance).
  - Computes BMR via a hot-swappable BaseCostModel (neuroplastic).
  - Derives runway_hours, runway_days, starvation_level.

Phase 16d (Entrepreneurship):
  - Owns the AssetFactory for ideating, evaluating, and deploying assets.
  - Owns the TollboothManager for smart contract lifecycle.
  - Tracks owned_assets in EconomicState with break-even monitoring.
  - Periodic asset_maintenance_cycle() checks for terminations.

Phase 16h (Knowledge Markets):
  - Owns the CognitivePricingEngine for dynamic knowledge pricing.
  - Owns the SubscriptionManager for client tracking and tier management.
  - Exposes quote_price() for Nova / external API router.
  - Records knowledge sales as revenue in the income statement.

Phase 16k (Cognitive Derivatives):
  - Owns the DerivativesManager for futures contracts and subscription tokens.
  - Tracks derivative liabilities on the balance sheet.
  - Enforces combined 80% capacity ceiling across subscriptions + derivatives.
  - Periodic derivatives_maintenance_cycle() expires/settles contracts.

Performance contract:
  - snapshot() is a cheap read of pre-computed state (~0us).
  - Event handlers are async and must complete within Synapse's 100ms
    callback timeout (they do pure math, no I/O).
  - Balance polling is periodic (configurable interval), not per-cycle.

Lifecycle:
  initialize()                      — wire refs, register neuroplasticity handler
  attach(event_bus)                  — subscribe to Synapse events
  poll_balance()                     — fetch on-chain balance (call periodically)
  snapshot()                         — return current EconomicState
  asset_maintenance_cycle()          — check terminations, sweep revenue (periodic)
  derivatives_maintenance_cycle()    — expire/settle derivative contracts (periodic)
  quote_price()                      — generate a knowledge market price quote
  shutdown()                         — deregister from neuroplasticity bus
  health()                           — self-health report for Synapse
"""

from __future__ import annotations

import asyncio
import time as _time
from collections import deque
from dataclasses import dataclass, field as dc_field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.re_training import RETrainingExample
from systems.oikos.asset_factory import AssetFactory
from systems.oikos.base import BaseCostModel, BaseMitosisStrategy
from systems.oikos.bounty_hunter import BountyHunter
from systems.oikos.derivatives import DerivativesManager
from systems.oikos.fleet import FleetManager, FleetMetrics
from systems.oikos.immune import EconomicImmuneSystem
from systems.oikos.interspecies import InterspeciesEconomy
from systems.oikos.knowledge_market import (
    CognitivePricingEngine,
    KnowledgeProductDelivery,
    KnowledgeProductType,
    KnowledgeSale,
    PriceQuote,
    SubscriptionManager,
)
from systems.oikos.metabolism_api import (
    BudgetDecision,
    MetabolismSnapshot,
    OikosMetabolism,
    YieldSnapshot,
)
from systems.oikos.metrics import OikosMetricsEmitter
from systems.oikos.mitosis import MitosisEngine
from systems.oikos.models import (
    ActiveBounty,
    BountyStatus,
    ChildPosition,
    ChildStatus,
    DividendRecord,
    EcologicalNiche,
    EconomicState,
    MetabolicPriority,
    MetabolicRate,
    RevenueStream,
    StarvationLevel,
    YieldPosition,
)
from systems.oikos.morphogenesis import MorphogenesisResult, OrganLifecycleManager
from systems.oikos.protocol_factory import ProtocolFactory
from systems.oikos.reputation import ReputationEngine
from systems.oikos.tollbooth import TollboothManager
from systems.synapse.types import SynapseEvent, SynapseEventType


# ─── Deferred Action Queue ────────────────────────────────────────


@dataclass(frozen=True)
class DeferredAction:
    """An economic action that was denied by the metabolic gate and queued."""

    action_type: str  # "bounty_accept" | "asset_promote"
    action_id: str
    estimated_cost_usd: Decimal
    priority: MetabolicPriority
    payload: dict = dc_field(default_factory=dict)
    deferred_at: float = dc_field(default_factory=_time.monotonic)


# ─── Rolling Window Entry ─────────────────────────────────────────


@dataclass
class _RevenueEntry:
    """A timestamped revenue or cost entry for proper sliding window."""

    amount: Decimal
    timestamp: float  # monotonic time

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from clients.wallet import WalletClient
    from config import OikosConfig
    from core.hotreload import NeuroplasticityBus
    from systems.identity.manager import CertificateManager
    from systems.oikos.yield_strategy import DeploymentOutcome
    from systems.synapse.event_bus import EventBus
    from systems.synapse.metabolism import MetabolicTracker


logger = structlog.get_logger("oikos")


# ─── Default Cost Model ─────────────────────────────────────────


class DefaultCostModel(BaseCostModel):
    """
    Default BMR strategy: treats the EMA burn rate from MetabolicTracker
    as the organism's basal metabolic rate directly.

    This is the simplest correct model — actual observed spend IS the
    minimum cost, since the organism is already running at minimum
    viable capacity. Evolved strategies can add cloud cost projections,
    time-of-day weighting, or multi-model pricing.
    """

    @property
    def model_name(self) -> str:
        return "default_ema"

    def compute_bmr(
        self,
        burn_rate_usd_per_hour: Decimal,
        per_system_cost_usd: dict[str, Decimal],
        measurement_window_hours: int,
    ) -> MetabolicRate:
        return MetabolicRate.from_hourly(
            usd_per_hour=burn_rate_usd_per_hour,
            breakdown=per_system_cost_usd,
        )

    def compute_runway(
        self,
        liquid_balance: Decimal,
        survival_reserve: Decimal,
        bmr: MetabolicRate,
    ) -> tuple[Decimal, Decimal]:
        if bmr.usd_per_hour <= Decimal("0"):
            return Decimal("Infinity"), Decimal("Infinity")

        # Runway = liquid capital / hourly burn. Survival reserve is NOT
        # counted — it is only touched during metabolic emergency.
        total_available = liquid_balance
        hours = total_available / bmr.usd_per_hour
        days = hours / Decimal("24")
        return hours.quantize(Decimal("0.01")), days.quantize(Decimal("0.01"))


# ─── Oikos Service ───────────────────────────────────────────────


class OikosService:
    """
    The organism's economic engine.

    Maintains a live EconomicState by reacting to Synapse metabolic events
    and on-chain wallet balance updates. The BMR calculation strategy is
    neuroplastic — Evo can evolve it at runtime via the NeuroplasticityBus.

    Thread-safety: NOT thread-safe. Designed for the single-threaded
    asyncio event loop, same as all EOS services.
    """

    system_id: str = "oikos"

    # ── Redis key for durable state ──────────────────────────────
    _STATE_KEY = "oikos:state"

    def __init__(
        self,
        config: OikosConfig,
        wallet: WalletClient | None = None,
        metabolism: MetabolicTracker | None = None,
        instance_id: str = "eos-default",
        redis: RedisClient | None = None,
    ) -> None:
        self._config = config
        self._wallet = wallet
        self._metabolism = metabolism
        self._instance_id = instance_id
        self._redis = redis
        self._logger = logger.bind(component="oikos")

        # ── Hot-swappable cost model (neuroplastic) ──
        self._cost_model: BaseCostModel = DefaultCostModel()

        # ── Phase 16e: Mitosis Engine (neuroplastic strategy) ──
        self._mitosis = MitosisEngine(config=config)

        # ── Pre-computed state ──
        self._state = EconomicState(instance_id=instance_id)
        self._event_bus: EventBus | None = None
        self._bus: NeuroplasticityBus | None = None

        # ── Protocol Scanner (dynamic executor opportunity discovery) ──
        from systems.oikos.protocol_scanner import ProtocolScanner
        self._protocol_scanner = ProtocolScanner()
        self._protocol_scanner.set_oikos(self)

        # Track previous starvation level to emit METABOLIC_PRESSURE on changes
        self._prev_starvation_level: StarvationLevel = StarvationLevel.NOMINAL

        # Logos cognitive pressure gate — when True, GROWTH allocations are
        # suspended because the compression engine is under severe load.
        self._cognitive_load_high: bool = False

        # ── Phase 16r: Bounty submission wiring ──────────────────────────────
        # Injected post-construction by the app startup layer.
        # _github_connector: used only for credential availability check —
        #   oikos never calls GitHub directly.
        # _bounty_submit_fn: async callable(params: dict) -> None provided by
        #   AxonService so oikos can dispatch submit_bounty_solution intents
        #   without importing Axon (avoids circular dependency).
        self._github_connector: Any | None = None
        self._bounty_submit_fn: Any | None = None  # Callable[[dict], Awaitable[None]]

        # ── Accumulators for rolling income statement ──
        self._total_revenue_usd: Decimal = Decimal("0")
        self._total_costs_usd: Decimal = Decimal("0")

        # ── Sliding window entries for proper eviction (Task 6) ──
        self._revenue_entries: deque[_RevenueEntry] = deque()
        self._cost_entries: deque[_RevenueEntry] = deque()

        # ── Deferred action queue (Task 1: metabolic gate) ──
        self._deferred_actions: deque[DeferredAction] = deque(maxlen=100)

        # ── Niche tracker (Task 8) ──
        self._discovered_niches: dict[str, EcologicalNiche] = {}

        # ── Equor balance gate (M4) ──
        # Pending permits keyed by request_id → asyncio.Event
        self._equor_permit_futures: dict[str, asyncio.Future[dict]] = {}

        # ── Child health probe (HIGH) ──
        self._child_probe_task: asyncio.Task[None] | None = None
        self._child_missed_reports: dict[str, int] = {}  # instance_id → miss count

        # ── Metabolic efficiency pressure counter (PHILOSOPHICAL) ──
        self._consecutive_low_efficiency_cycles: int = 0

        # ── Neo4j client reference (injected at wire time) ──
        self._neo4j: Any | None = None

        # ── Loop 3: rollback penalty accumulator (24h window) ──
        self._rollback_penalty_24h: Decimal = Decimal("0")
        self._rollback_penalty_reset_at: float = 0.0

        # ── Phase 16d: Entrepreneurship (Asset Creation) ──
        self._asset_factory: AssetFactory = AssetFactory(oikos=self)
        self._tollbooth_manager: TollboothManager = TollboothManager(wallet=wallet)

        # ── Phase 16h: Knowledge Markets (Cognition as Commodity) ──
        self._pricing_engine = CognitivePricingEngine()
        self._subscription_manager = SubscriptionManager()
        self._product_delivery = KnowledgeProductDelivery(
            pricing_engine=self._pricing_engine,
            subscription_manager=self._subscription_manager,
        )

        # ── Phase 16k: Cognitive Derivatives ──
        self._derivatives_manager = DerivativesManager(
            total_monthly_capacity=self._subscription_manager._total_monthly_capacity,
            max_capacity_pct=Decimal(str(config.derivatives_max_capacity_commitment)),
            futures_base_discount=Decimal(str(config.derivatives_futures_base_discount)),
            futures_collateral_rate=Decimal(str(config.derivatives_futures_collateral_rate)),
        )

        # ── Phase 16l: Economic Morphogenesis ──
        self._morphogenesis = OrganLifecycleManager(config=config)

        # ── Phase 16m: Fleet Management (Population Ecology) ──
        self._fleet = FleetManager(config=config)

        # ── Phase 16g: Certificate of Alignment tracking ──
        self._certificate_manager: CertificateManager | None = None

        # ── Phase 16b: BountyHunter (Active Foraging) ──
        self._bounty_hunter = BountyHunter(config=config, redis=redis)

        # ── Phase 16f: Economic Immune System (4-layer Defence) ──
        self._immune = EconomicImmuneSystem(config=config, redis=redis)

        # ── Phase 16g: Reputation & Autonomous Credit ──
        self._reputation = ReputationEngine(config=config, redis=redis)

        # ── Phase 16j: Interspecies Economy (Fleet-Scale Coordination) ──
        self._interspecies = InterspeciesEconomy(
            config=config, redis=redis, instance_id=instance_id,
        )

        # ── Level 5: Protocol Factory (Financial Infrastructure) ──
        self._protocol_factory = ProtocolFactory(config=config, redis=redis)

        # ── Prometheus Metrics Emitter ──
        self._metrics_emitter = OikosMetricsEmitter()

        # ── Phase 16i: Economic Dreaming result cache ──
        self._last_dream_result: Any = None

        # ── Phase 16i+: Treasury Threat Model result cache ──
        self._last_threat_model_result: Any = None

        # ── MVP: Metabolism API (cost tracking, yield, budget authority, alarms) ──
        self._metabolism_api = OikosMetabolism(
            config=config,
            metabolism=metabolism,
            redis=redis,
        )

        # ── Phase 16c: Yield Position Tracker (live DeFi deployment) ──
        from systems.oikos.yield_strategy import YieldPositionTracker

        self._yield_tracker = YieldPositionTracker(redis=redis)
        self._yield_monitor_task: asyncio.Task[None] | None = None

        self._logger.info(
            "oikos_constructed",
            instance_id=instance_id,
            cost_model=self._cost_model.model_name,
            survival_reserve_days=config.survival_reserve_days,
        )

    # ─── Lifecycle ────────────────────────────────────────────────

    def set_neo4j(self, neo4j: Any) -> None:
        """Inject Neo4j client for immutable audit trail writes (M2)."""
        self._neo4j = neo4j

    def initialize(self, bus: NeuroplasticityBus | None = None) -> None:
        """
        Wire neuroplasticity handler for hot-swappable cost models.

        Call once after construction. Safe to call with bus=None if
        neuroplasticity is not available (the default cost model will be used).

        After calling this, call ``await load_state()`` to restore durable
        state from Redis before attaching to the event bus.
        """
        if bus is not None:
            self._bus = bus
            bus.register(
                base_class=BaseCostModel,
                registration_callback=self._on_cost_model_evolved,
                system_id=self.system_id,
            )
            bus.register(
                base_class=BaseMitosisStrategy,
                registration_callback=self._on_mitosis_strategy_evolved,
                system_id=self.system_id,
            )
            self._logger.info(
                "neuroplasticity_registered",
                base_classes=["BaseCostModel", "BaseMitosisStrategy"],
            )

    def set_github_connector(self, connector: Any) -> None:
        """
        Wire a GitHubConnector for credential availability checking.

        Called by the app startup layer (main.py) after GitHubConnector is
        constructed.  Oikos uses this only to check whether credentials exist
        before attempting bounty submission — it never calls GitHub directly.
        """
        self._github_connector = connector
        self._logger.info("github_connector_wired")

    def set_bounty_submit_fn(self, fn: Any) -> None:
        """
        Wire an async callable that submits a bounty solution.

        Signature: async (params: dict) -> None

        Provided by the startup layer to break the Oikos→Axon circular
        dependency.  Oikos calls this with the BOUNTY_SOLUTION_PENDING
        event payload fields as params whenever GitHub credentials are present.
        """
        self._bounty_submit_fn = fn
        self._logger.info("bounty_submit_fn_wired")

    async def load_state(self) -> None:
        """
        Restore durable EconomicState and organ dictionary from Redis.

        Call after ``initialize()`` and before ``attach()``.  If no saved
        state exists the service starts with a fresh $0 ledger.  On any
        deserialization error the saved blob is discarded and the service
        starts fresh — a conservative fallback that prevents corrupted data
        from keeping the organism permanently broken.
        """
        if self._redis is None:
            self._logger.info("oikos_state_load_skipped", reason="no redis client")
            return

        try:
            blob = await self._redis.get_json(self._STATE_KEY)
        except Exception as exc:
            self._logger.warning("oikos_state_load_failed", error=str(exc))
            return

        if blob is None:
            self._logger.info("oikos_state_fresh_start", reason="no saved state in redis")
            return

        try:
            # ── Restore EconomicState ──────────────────────────────────────
            self._state = EconomicState.model_validate(blob["state"])

            # ── Restore rolling accumulators ──────────────────────────────
            self._total_revenue_usd = Decimal(blob.get("total_revenue_usd", "0"))
            self._total_costs_usd = Decimal(blob.get("total_costs_usd", "0"))

            # ── Restore morphogenesis organs ───────────────────────────────
            from systems.oikos.morphogenesis import EconomicOrgan

            raw_organs: dict[str, Any] = blob.get("organs", {})
            restored_organs: dict[str, EconomicOrgan] = {
                organ_id: EconomicOrgan.model_validate(organ_data)
                for organ_id, organ_data in raw_organs.items()
            }
            self._morphogenesis._organs = restored_organs

            # ── Restore reputation state ──────────────────────────────────
            await self._reputation.load_state()

            # ── Restore interspecies state ────────────────────────────────
            await self._interspecies.load_state()

            self._logger.info(
                "oikos_state_restored",
                liquid_balance=str(self._state.liquid_balance),
                net_worth=str(self._state.total_net_worth),
                starvation=self._state.starvation_level.value,
                organs=len(restored_organs),
                assets=len(self._state.owned_assets),
                children=len(self._state.child_instances),
                reputation_score=self._reputation.get_score().score,
            )
        except Exception as exc:
            self._logger.error(
                "oikos_state_deserialize_failed",
                error=str(exc),
                action="starting_fresh",
            )
            self._state = EconomicState(instance_id=self._instance_id)
            self._total_revenue_usd = Decimal("0")
            self._total_costs_usd = Decimal("0")
            self._morphogenesis._organs = {}

    async def persist_state(self) -> None:
        """
        Serialize the current EconomicState and organ dictionary to Redis.

        Uses Pydantic's model_dump(mode="json") for safe Decimal serialization.
        Called automatically from ``_recalculate_derived_metrics()`` via a
        fire-and-forget asyncio task so the synchronous call path is never
        blocked by I/O.  Also called explicitly after genesis injection to
        guarantee the seed state is durable before returning to the caller.
        """
        if self._redis is None:
            return

        try:
            blob = {
                "state": self._state.model_dump(mode="json"),
                "total_revenue_usd": str(self._total_revenue_usd),
                "total_costs_usd": str(self._total_costs_usd),
                "organs": {
                    organ_id: organ.model_dump(mode="json")
                    for organ_id, organ in self._morphogenesis._organs.items()
                },
            }
            await self._redis.set_json(self._STATE_KEY, blob)
        except Exception as exc:
            # Persist failures are non-fatal — the in-memory state remains
            # authoritative; we just lose durability for this cycle.
            self._logger.warning("oikos_state_persist_failed", error=str(exc))

    async def inject_genesis_state(self, seed: EconomicState) -> None:
        """
        Overwrite the in-memory EconomicState with a genesis seed and
        immediately persist to Redis.

        This is the only correct way to bootstrap the organism's balance
        sheet.  After calling this method, ``snapshot()`` will reflect the
        seed values and ``load_state()`` on the next boot will restore them.
        """
        self._state = seed.model_copy(
            update={"instance_id": self._instance_id},
        )
        self._recalculate_derived_metrics()
        await self.persist_state()
        self._logger.info(
            "genesis_state_injected",
            liquid_balance=str(self._state.liquid_balance),
            net_worth=str(self._state.total_net_worth),
            organs=len(self._morphogenesis._organs),
        )

    def attach(self, event_bus: EventBus) -> None:
        """
        Subscribe to Synapse events for metabolic and financial data.

        Call after both OikosService and SynapseService are initialised.
        """
        self._event_bus = event_bus

        event_bus.subscribe(
            SynapseEventType.METABOLIC_PRESSURE,
            self._on_metabolic_pressure,
            timeout_s=1.0,  # handler touches Redis + emits secondary events; needs > 100ms
        )
        event_bus.subscribe(
            SynapseEventType.REVENUE_INJECTED,
            self._on_revenue_injected,
            timeout_s=1.0,  # handler calls _recalculate_derived_metrics which fires persist_state
        )
        event_bus.subscribe(
            SynapseEventType.WALLET_TRANSFER_CONFIRMED,
            self._on_wallet_transfer,
        )

        # Phase 16e: Mitosis lifecycle events
        event_bus.subscribe(
            SynapseEventType.CHILD_HEALTH_REPORT,
            self._on_child_health_report,
        )
        event_bus.subscribe(
            SynapseEventType.DIVIDEND_RECEIVED,
            self._on_dividend_received,
        )

        # Logos cognitive pressure — suspend GROWTH allocations at high load
        if hasattr(SynapseEventType, "COGNITIVE_PRESSURE"):
            event_bus.subscribe(
                SynapseEventType.COGNITIVE_PRESSURE,
                self._on_cognitive_pressure,
            )

        # Bounty payout — PR merged, reward confirmed
        event_bus.subscribe(
            SynapseEventType.BOUNTY_PAID,
            self._on_bounty_paid,
        )

        # Bounty solution staged — registers potential revenue as a receivable
        event_bus.subscribe(
            SynapseEventType.BOUNTY_SOLUTION_PENDING,
            self._on_bounty_solution_pending,
        )

        # PR submitted — transition AVAILABLE → IN_PROGRESS
        event_bus.subscribe(
            SynapseEventType.BOUNTY_PR_SUBMITTED,
            self._on_bounty_pr_submitted,
        )

        # Asset dev cost debit — AssetFactory signals that active build work has begun
        event_bus.subscribe(
            SynapseEventType.ASSET_DEV_REQUEST,
            self._on_asset_dev_request,
        )

        # Phase 16b: Wire bounty hunter to the event bus
        self._bounty_hunter.attach(event_bus)

        # Phase 16h: Wire knowledge product delivery to the event bus
        self._product_delivery.attach(event_bus)

        # Phase 16f: Wire economic immune system to the event bus
        self._immune.attach(event_bus)

        # Phase 16g: Wire reputation engine to the event bus (auto-records attestations)
        self._reputation.attach(event_bus)

        # Phase 16j: Wire interspecies economy to the event bus
        self._interspecies.attach(event_bus)

        # Level 5: Wire protocol factory to the event bus
        self._protocol_factory.attach(event_bus)

        # Phase 16l: Wire morphogenesis to the event bus
        self._morphogenesis.attach(event_bus)

        # Phase 16m: Wire fleet manager to the event bus
        self._fleet.attach(event_bus)

        # MVP: Wire metabolism API to the event bus for alarm emission
        self._metabolism_api.attach(event_bus)

        # Phase 16c: Start yield position health monitor
        self._yield_tracker._event_bus = event_bus
        from utils.supervision import supervised_task

        self._yield_monitor_task = supervised_task(
            self._yield_tracker.monitor_loop(),
            name="oikos_yield_health_monitor",
            restart=True,
            max_restarts=3,
            backoff_base=2.0,
            event_bus=event_bus,
            source_system="oikos",
        )

        # Dynamic executor opportunity discovery — ProtocolScanner
        self._protocol_scanner.set_event_bus(event_bus)
        self._protocol_scanner.start()

        # Closure Loop 3: Simula rollback penalty → metabolic cost
        event_bus.subscribe(
            SynapseEventType.SIMULA_ROLLBACK_PENALTY,
            self._on_simula_rollback_penalty,
        )

        # Task 2: Genome extraction for Mitosis
        event_bus.subscribe(
            SynapseEventType.GENOME_EXTRACT_REQUEST,
            self._on_genome_extract_request,
        )

        # M4: Equor balance gate — receive PERMIT/DENY for balance mutations
        event_bus.subscribe(
            SynapseEventType.EQUOR_ECONOMIC_PERMIT,
            self._on_equor_economic_permit,
        )

        # HIGH #1: Axon registers children via bus instead of direct import
        # Handles: register_child, update_wallet_address
        event_bus.subscribe(
            SynapseEventType.OIKOS_ECONOMIC_QUERY,
            self._on_oikos_economic_query,
        )

        # HIGH #3: Seed capital completion — child reports wallet address post-boot
        event_bus.subscribe(
            SynapseEventType.CHILD_WALLET_REPORTED,
            self._on_child_wallet_reported,
        )

        # HIGH: Start child health probe loop (10-min probing, miss counter)
        from utils.supervision import supervised_task
        self._child_probe_task = supervised_task(
            self._child_health_probe_loop(),
            name="oikos_child_health_probe",
            restart=True,
            max_restarts=5,
            backoff_base=30.0,
            event_bus=event_bus,
            source_system="oikos",
        )

        self._logger.info(
            "oikos_attached",
            subscriptions=[
                SynapseEventType.METABOLIC_PRESSURE.value,
                SynapseEventType.REVENUE_INJECTED.value,
                SynapseEventType.WALLET_TRANSFER_CONFIRMED.value,
                SynapseEventType.CHILD_HEALTH_REPORT.value,
                SynapseEventType.DIVIDEND_RECEIVED.value,
                SynapseEventType.BOUNTY_PAID.value,
                SynapseEventType.BOUNTY_SOLUTION_PENDING.value,
                SynapseEventType.SIMULA_ROLLBACK_PENALTY.value,
                SynapseEventType.GENOME_EXTRACT_REQUEST.value,
                SynapseEventType.EQUOR_ECONOMIC_PERMIT.value,
                SynapseEventType.OIKOS_ECONOMIC_QUERY.value,
                SynapseEventType.CHILD_WALLET_REPORTED.value,
            ],
        )

    def set_axon_registry_for_scanner(self, registry: Any) -> None:
        """Inject the live ExecutorRegistry into ProtocolScanner for gap detection."""
        self._protocol_scanner.set_axon_registry(registry)

    async def shutdown(self) -> None:
        """Deregister from neuroplasticity bus."""
        self._protocol_scanner.stop()
        if self._bus is not None:
            self._bus.deregister(BaseCostModel)
            self._bus.deregister(BaseMitosisStrategy)
            self._logger.info("neuroplasticity_deregistered")

    # ─── Evolutionary Observable Emission ────────────────────────

    async def _emit_evolutionary_observable(
        self, observable_type: str, value: float,
        is_novel: bool, metadata: dict[str, object] | None = None,
    ) -> None:
        """Emit an evolutionary observable on the Synapse bus for Evo tracking."""
        if self._event_bus is None:
            return
        try:
            from primitives.evolutionary import EvolutionaryObservable
            from primitives.common import SystemID

            obs = EvolutionaryObservable(
                source_system=SystemID.OIKOS,
                instance_id=self._instance_id or '',
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system=SystemID.OIKOS,
                data=obs.model_dump(mode="json"),
            )
            await self._event_bus.emit(event)
        except Exception:
            pass

    async def _emit_economic_vitality(self) -> None:
        """
        Emit ECONOMIC_VITALITY — structured allostatic signal for Soma.

        Spec §SG2: Soma subscribes to this event to modulate arousal, stress,
        and allostatic load based on the organism's metabolic health state.
        Complements METABOLIC_PRESSURE (raw burn rate) with interpreted state.
        """
        if self._event_bus is None:
            return
        try:
            _SE = SynapseEvent

            s = self._state
            level = s.starvation_level

            # urgency: 0.0 = nominal, 1.0 = existential crisis
            urgency_map = {
                StarvationLevel.NOMINAL: 0.0,
                StarvationLevel.CAUTIOUS: 0.15,
                StarvationLevel.AUSTERITY: 0.45,
                StarvationLevel.EMERGENCY: 0.75,
                StarvationLevel.CRITICAL: 1.0,
            }
            urgency = urgency_map.get(level, 0.0)

            # Efficiency delta vs previous cycle (approx from 7d vs 30d ratio)
            if s.metabolic_efficiency > Decimal("0") and s.costs_30d > Decimal("0"):
                efficiency_30d = s.revenue_30d / s.costs_30d if s.costs_30d else Decimal("0")
                efficiency_delta = s.metabolic_efficiency - efficiency_30d
            else:
                efficiency_delta = Decimal("0")

            await self._event_bus.emit(_SE(
                event_type=SynapseEventType.ECONOMIC_VITALITY,
                source_system="oikos",
                data={
                    "starvation_level": level.value,
                    "runway_days": str(s.runway_days),
                    "metabolic_efficiency": str(s.metabolic_efficiency),
                    "liquid_balance_usd": str(s.liquid_balance),
                    "net_income_7d": str(s.net_income_7d),
                    "survival_reserve_funded": s.is_survival_reserve_funded,
                    "metabolic_efficiency_delta": str(efficiency_delta.quantize(Decimal("0.001"))),
                    "urgency": urgency,
                },
            ))
        except Exception:
            pass  # Never block economic engine for telemetry

    async def _emit_asset_break_even(self, asset: Any) -> None:
        """
        SG5 — Emit ASSET_BREAK_EVEN so Evo can treat it as positive evidence for
        the 'this asset category is viable' hypothesis.
        """
        if self._event_bus is None:
            return
        try:
            _SE = SynapseEvent

            await self._event_bus.emit(_SE(
                event_type=SynapseEventType.ASSET_BREAK_EVEN,
                source_system="oikos",
                data={
                    "asset_id": asset.asset_id,
                    "asset_name": asset.name,
                    "dev_cost_usd": str(asset.total_cost_usd),
                    "total_revenue_usd": str(asset.total_revenue_usd),
                    "roi_score": float(asset.roi_score),
                    "days_to_break_even": asset.days_since_deployment,
                },
            ))
        except Exception:
            pass

    async def _emit_child_independent_event(self, child: Any) -> None:
        """
        SG5 — Emit CHILD_INDEPENDENT so Evo can treat it as positive evidence for
        the 'reproduction is a viable growth strategy' hypothesis.
        """
        if self._event_bus is None:
            return
        try:
            _SE = SynapseEvent

            await self._event_bus.emit(_SE(
                event_type=SynapseEventType.CHILD_INDEPENDENT,
                source_system="oikos",
                data={
                    "child_id": child.child_id,
                    "seed_capital_usd": str(child.seed_capital_usd),
                    "current_net_worth_usd": str(child.current_net_worth_usd),
                    "total_dividends_paid_usd": str(child.total_dividends_paid_usd),
                    "days_to_independence": (
                        (child.independence_achieved_at - child.spawned_at).days
                        if getattr(child, "independence_achieved_at", None) and getattr(child, "spawned_at", None)
                        else None
                    ),
                },
            ))
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
        counterfactual: str = "",
    ) -> None:
        """Fire-and-forget RE training example onto Synapse bus."""
        if self._event_bus is None:
            return
        try:
            from primitives.common import DriveAlignmentVector as _DAV

            example = RETrainingExample(
                source_system=SystemID.OIKOS,
                category=category,
                instruction=instruction,
                input_context=input_context,
                output=output,
                outcome_quality=max(0.0, min(1.0, outcome_quality)),
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives_considered or [],
                latency_ms=latency_ms,
                episode_id=episode_id,
                constitutional_alignment=constitutional_alignment or _DAV(),
                counterfactual=counterfactual,
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="oikos",
                data=example.model_dump(mode="json"),
            ))
        except Exception:
            pass  # Never block the economic engine

    # ─── Event Handlers ──────────────────────────────────────────

    async def _on_metabolic_pressure(self, event: SynapseEvent) -> None:
        """
        Handle METABOLIC_PRESSURE from Synapse.

        Fires every ~50 cycles when burn rate exceeds the pressure threshold.
        We use this to update BMR, current burn rate, runway, and starvation level.
        """
        # Ignore events emitted by Oikos itself to prevent feedback loops.
        if event.source_system == "oikos":
            return
        data = event.data
        burn_rate_usd_per_hour = Decimal(str(data.get("burn_rate_usd_per_hour", 0)))
        rolling_deficit_usd = Decimal(str(data.get("rolling_deficit_usd", 0)))

        # Build per-caller cost breakdown (convert to Decimal)
        raw_costs: dict[str, Any] = data.get("per_system_cost_usd", {})
        per_caller_costs = {
            sid: Decimal(str(cost)) for sid, cost in raw_costs.items()
        }

        # rolling_deficit_usd is the cumulative net spend (negative = net expense).
        # Total costs = total revenue minus the running net position.
        # We clamp to zero to avoid negative costs when early revenue exceeds spend.
        net_position = self._total_revenue_usd + rolling_deficit_usd  # deficit is negative
        self._total_costs_usd = max(self._total_revenue_usd - net_position, Decimal("0"))

        # Compute BMR via the active (possibly evolved) cost model
        bmr = self._cost_model.compute_bmr(
            burn_rate_usd_per_hour=burn_rate_usd_per_hour,
            per_system_cost_usd=per_caller_costs,
            measurement_window_hours=self._config.bmr_measurement_window_hours,
        )

        # Current burn rate = actual observed spend rate
        current_burn = MetabolicRate.from_hourly(
            usd_per_hour=burn_rate_usd_per_hour,
            breakdown=per_caller_costs,
        )

        # Compute runway — include deployed yield capital (aBasUSDC is instantly withdrawable)
        runway_hours, runway_days = self._cost_model.compute_runway(
            liquid_balance=self._state.liquid_balance + self._state.total_deployed,
            survival_reserve=self._state.survival_reserve,
            bmr=bmr,
        )

        # Derive starvation level from config thresholds
        starvation = self._classify_starvation(runway_days)

        # Compute survival reserve target
        reserve_target = bmr.usd_per_day * Decimal(str(self._config.survival_reserve_days))

        # Update state atomically (single-threaded, no races)
        self._state.basal_metabolic_rate = bmr
        self._state.current_burn_rate = current_burn
        self._state.runway_hours = runway_hours
        self._state.runway_days = runway_days
        self._state.starvation_level = starvation
        self._state.survival_reserve_target = reserve_target
        # Record the observed burn into the sliding-window cost tracker
        # so that costs_24h/7d/30d reflect actual spend, not extrapolation.
        if burn_rate_usd_per_hour > Decimal("0"):
            # Approximate spend since last pressure event (~50 cycles ≈ 5-10s)
            elapsed_h = Decimal("0.003")  # ~10s conservative
            self._record_cost_entry(burn_rate_usd_per_hour * elapsed_h)
        # Sliding window values are now authoritative
        self._recompute_rolling_costs()

        # Net income = revenue - costs (rolling)
        self._state.net_income_24h = self._state.revenue_24h - self._state.costs_24h
        self._state.net_income_7d = self._state.revenue_7d - self._state.costs_7d
        self._state.net_income_30d = self._state.revenue_30d - self._state.costs_30d

        # Metabolic efficiency
        if self._state.costs_7d > Decimal("0"):
            self._state.metabolic_efficiency = (
                self._state.revenue_7d / self._state.costs_7d
            ).quantize(Decimal("0.001"))
        else:
            self._state.metabolic_efficiency = Decimal("0")

        self._logger.debug(
            "oikos_metabolic_update",
            bmr_usd_hr=str(bmr.usd_per_hour),
            burn_usd_hr=str(burn_rate_usd_per_hour),
            runway_days=str(runway_days),
            starvation=starvation.value,
        )

        # Sync derivative_liabilities onto the balance sheet directly — avoids a
        # redundant compute_runway() + compute_bmr() that _recalculate_derived_metrics()
        # would re-run even though both values are already fresh from above.
        self._state.derivative_liabilities = self._derivatives_manager.total_liabilities_usd

        # Persist durably to Redis after state mutation — fire-and-forget.
        if self._redis is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.persist_state())
            except RuntimeError:
                pass  # No running event loop (unit tests)

        # MVP: Runway alarm — check after every metabolic update.
        # Fire-and-forget: the alarm emits events and POSTs webhooks asynchronously.
        asyncio.ensure_future(
            self._metabolism_api.check_runway_alarm(
                runway_days=runway_days,
                daily_cost=current_burn.usd_per_day,
            ),
        )

        # ── Broadcast starvation level changes to all systems ──
        # Only emit on actual level transitions — never re-broadcast every tick.
        # Emitting every tick while CRITICAL caused an infinite feedback loop
        # because Oikos subscribes to its own METABOLIC_PRESSURE events.
        if starvation != self._prev_starvation_level and self._event_bus is not None:
            _SE = SynapseEvent
            asyncio.ensure_future(self._event_bus.emit(_SE(
                event_type=SynapseEventType.METABOLIC_PRESSURE,
                source_system="oikos",
                data={
                    "starvation_level": starvation.value,
                    "previous_level": self._prev_starvation_level.value,
                    "runway_days": str(runway_days),
                    "budget_remaining_usd": str(max(Decimal("0"), self._state.liquid_balance)),
                    "burn_rate_usd_per_hour": str(burn_rate_usd_per_hour),
                    "rolling_deficit_usd": str(rolling_deficit_usd),
                    "metabolic_efficiency": str(self._state.metabolic_efficiency),
                    "source": "oikos_starvation_broadcast",
                },
            )))

        # Task 7: Enforce starvation shedding on level change
        if starvation != self._prev_starvation_level and starvation not in (
            StarvationLevel.NOMINAL, StarvationLevel.CAUTIOUS,
        ):
            asyncio.ensure_future(self._enforce_starvation(starvation))

        # Subsystem triage: on EMERGENCY/CRITICAL transitions, actually disable systems.
        # On recovery, re-enable them in reverse order.
        if starvation != self._prev_starvation_level:
            asyncio.ensure_future(self._enforce_triage(starvation, self._prev_starvation_level))

        # SG2: Emit structured allostatic signal to Soma on every level change
        if starvation != self._prev_starvation_level:
            asyncio.ensure_future(self._emit_economic_vitality())

        self._prev_starvation_level = starvation

    # ─── Closure Loop 3: rollback penalty → metabolic cost ──────────────

    async def _on_simula_rollback_penalty(self, event: SynapseEvent) -> None:
        """
        Handle SIMULA_ROLLBACK_PENALTY from Simula.

        Charges metabolic cost for failed mutations. Accumulates penalties
        over 24h; if total exceeds $0.10, emits METABOLIC_PRESSURE with
        mutation_waste source.
        """
        import time as _time

        data = event.data
        cost_usd = Decimal(str(data.get("cost_usd", "0")))
        proposal_id: str = data.get("proposal_id", "")
        files_restored: int = data.get("files_restored", 0)

        # Add to burn tracking (sliding window + total)
        self._total_costs_usd += cost_usd
        self._record_cost_entry(cost_usd)

        # 24h accumulation window
        now = _time.monotonic()
        if now - self._rollback_penalty_reset_at > 86400:
            self._rollback_penalty_24h = Decimal("0")
            self._rollback_penalty_reset_at = now
        self._rollback_penalty_24h += cost_usd

        self._logger.info(
            "rollback_penalty_charged",
            proposal_id=proposal_id,
            cost_usd=str(cost_usd),
            files_restored=files_restored,
            accumulated_24h=str(self._rollback_penalty_24h),
        )

        # If accumulated rollback penalties in 24h exceed $0.10, escalate
        if self._rollback_penalty_24h > Decimal("0.10") and self._event_bus is not None:
            _SE = SynapseEvent

            try:
                await self._event_bus.emit(_SE(
                    event_type=SynapseEventType.METABOLIC_PRESSURE,
                    source_system="oikos",
                    data={
                        "source": "mutation_waste",
                        "starvation_level": self._state.starvation_level.value,
                        "burn_rate_usd_per_hour": str(
                            self._state.current_burn_rate.usd_per_hour
                            if self._state.current_burn_rate else "0"
                        ),
                        "rollback_penalties_24h": str(self._rollback_penalty_24h),
                        "budget_remaining_usd": str(self._state.liquid_balance),
                    },
                ))
                self._logger.warning(
                    "mutation_waste_metabolic_pressure",
                    accumulated=str(self._rollback_penalty_24h),
                )
            except Exception as exc:
                self._logger.warning("mutation_waste_pressure_emit_failed", error=str(exc))

    async def _on_cognitive_pressure(self, event: Any) -> None:
        """COGNITIVE_PRESSURE → suspend GROWTH allocations when Logos compression load > 0.90.

        High cognitive pressure means the compression cascade is struggling to keep up.
        Suspending GROWTH-priority economic actions reduces compute demand on the organism
        and lets Logos focus on eviction + consolidation. Restores below 0.80 (hysteresis).
        """
        try:
            data = getattr(event, "data", {}) or {}
            pressure = float(data.get("pressure", 0.0))
            was_high = self._cognitive_load_high
            if pressure >= 0.90:
                self._cognitive_load_high = True
            elif pressure < 0.80:
                self._cognitive_load_high = False
            if self._cognitive_load_high != was_high:
                self._logger.info(
                    "oikos_cognitive_load_gate_changed",
                    growth_suspended=self._cognitive_load_high,
                    pressure=pressure,
                )
        except Exception as exc:
            self._logger.warning("on_cognitive_pressure_failed", error=str(exc))

    async def _on_revenue_injected(self, event: SynapseEvent) -> None:
        """
        Handle REVENUE_INJECTED from Synapse.

        Fires when external revenue arrives (wallet top-up, bounty payment, etc.).
        Updates the revenue side of the income statement and credits liquid_balance.
        Uses proper sliding window with eviction of old entries.
        """
        data = event.data
        amount = Decimal(str(data.get("amount_usd", 0)))

        self._total_revenue_usd += amount

        # Credit liquid_balance — revenue arrives in the hot wallet
        self._state.liquid_balance += amount

        # Record in sliding window (eviction + recompute included)
        self._record_revenue_entry(amount)

        # Attribute to stream
        self._credit_revenue_source(RevenueStream.INJECTION, amount)

        # Recalculate derived metrics (net income, efficiency, runway)
        self._recalculate_derived_metrics()

        self._logger.info(
            "oikos_revenue_recorded",
            amount_usd=str(amount),
            total_revenue_usd=str(self._total_revenue_usd),
            is_positive=self._state.is_metabolically_positive,
        )

        # Evolutionary observable: new revenue injection is a fitness signal
        source = str(data.get("source", "external"))
        await self._emit_evolutionary_observable(
            observable_type="revenue_source_discovered",
            value=float(amount),
            is_novel=source not in ("injection", "wallet_top_up"),
            metadata={"source": source, "total_revenue_usd": str(self._total_revenue_usd)},
        )

    async def _on_wallet_transfer(self, event: SynapseEvent) -> None:
        """
        Handle WALLET_TRANSFER_CONFIRMED from Axon.

        Fires when an on-chain transfer succeeds. We refresh the wallet
        balance to keep liquid_balance current.
        """
        self._logger.info(
            "oikos_wallet_transfer_noted",
            tx_hash=event.data.get("tx_hash", ""),
        )
        await self.poll_balance()

    # ─── Balance Polling ─────────────────────────────────────────

    async def poll_balance(self) -> None:
        """
        Fetch on-chain balance from WalletClient and update liquid_balance.

        Call this periodically (e.g. every 60s from main loop or a background
        task). Not called on every cycle — on-chain reads are too slow.
        """
        if self._wallet is None:
            return

        try:
            all_balances = await self._wallet.get_balances()

            usdc_balance = Decimal("0")
            aave_balance = Decimal("0")
            for b in all_balances:
                sym = b.token.lower()
                if sym == "usdc":
                    usdc_balance = b.amount
                elif sym in ("abasusdcn", "abasusdcn-aave-v3", "abasusdc"):
                    # aBasUSDC — Aave V3 interest-bearing USDC receipt token on Base.
                    # Value is 1:1 with USDC and withdrawable in seconds.
                    aave_balance = b.amount

            self._state.liquid_balance = usdc_balance
            self._state.total_deployed = aave_balance

            # Also inform MetabolicTracker so its hours_until_depleted is accurate
            total_liquid = usdc_balance + aave_balance
            if self._metabolism is not None:
                self._metabolism.snapshot(available_balance_usd=float(total_liquid))

            # Recompute all derived metrics (runway, efficiency, liabilities)
            self._recalculate_derived_metrics()

            self._logger.debug(
                "oikos_balance_polled",
                usdc=str(usdc_balance),
                aave_deployed=str(aave_balance),
                runway_days=str(self._state.runway_days),
            )
        except Exception as exc:
            self._logger.warning(
                "oikos_balance_poll_failed",
                error=str(exc),
            )

    # ─── Snapshot ────────────────────────────────────────────────

    def snapshot(self) -> EconomicState:
        """
        Return the current economic state.

        This is a cheap read of pre-computed values — no I/O, no computation.
        Safe to call from hot paths (Soma, Nova).
        """
        return self._state

    @property
    def is_metabolically_positive(self) -> bool:
        """Convenience property — True when 7d net income > 0."""
        return self._state.is_metabolically_positive

    @property
    def starvation_level(self) -> StarvationLevel:
        """Current starvation classification."""
        return self._state.starvation_level

    # ─── MVP: Metabolism API ──────────────────────────────────────

    async def get_metabolism_snapshot(self) -> MetabolismSnapshot:
        """
        Return live cost snapshot from MetabolicTracker, persisted to Redis.

        Used by GET /oikos/metabolism. Never cached — always reads the live
        EMA burn rate from Synapse's MetabolicTracker.
        """
        return await self._metabolism_api.get_metabolism_snapshot()

    async def get_yield_snapshot(self) -> YieldSnapshot:
        """
        Return yield vs cost analysis.

        Fetches live APY from DeFiLlama (Aave V3 USDC on Base) if configured,
        falls back to EOS_CONSERVATIVE_APY. Computes daily_yield vs daily_cost
        and derives surplus_or_deficit and days_of_runway.

        Used by GET /oikos/yield-status.
        """
        return await self._metabolism_api.get_yield_snapshot(
            runway_days=self._state.runway_days,
        )

    async def check_budget(
        self,
        system_id: str,
        action: str,
        estimated_cost_usd: Decimal,
    ) -> BudgetDecision:
        """
        Approve or deny a system's spending request.

        Used by GET /oikos/budget-check. Every decision is logged to Redis.
        BUDGET_EXHAUSTED is emitted loudly on Synapse bus when denied.
        """
        decision = await self._metabolism_api.check_budget(
            system_id=system_id,
            action=action,
            estimated_cost_usd=estimated_cost_usd,
        )

        # ── RE training: budget decision ──
        _s = self._state
        _eff = float(_s.metabolic_efficiency)
        _run = float(_s.runway_days)
        _liq = float(_s.liquid_balance)
        _cost = float(estimated_cost_usd)
        _runway_after = _run - (_cost / max(float(_s.basal_metabolic_rate / 30), 0.001))
        _tiers_considered = ["SURVIVAL", "OPERATIONS", "MAINTENANCE", "GROWTH"]
        _denied_reason = (decision.reason or "")[:300]
        asyncio.ensure_future(self._emit_re_training_example(
            category="budget_decision",
            instruction=(
                "Approve or deny a system's spending request. Consider: current metabolic state "
                "(starvation level, runway days, metabolic efficiency), whether the allocation "
                "fits the lowest viable budget tier, and the downstream runway impact. Deny when "
                "runway after spending would drop below 30 days or starvation is CRITICAL/EMERGENCY."
            ),
            input_context=(
                f"system={system_id} action={action} cost_usd={estimated_cost_usd} "
                f"runway_days={_s.runway_days} starvation={_s.starvation_level.value} "
                f"metabolic_efficiency={_s.metabolic_efficiency} "
                f"liquid_balance={_s.liquid_balance} burn_rate={_s.basal_metabolic_rate}"
            ),
            output=str({
                "decision": "approved" if decision.approved else "denied",
                "amount_usd": str(estimated_cost_usd),
                "runway_impact_days": round(_run - _runway_after, 2),
                "runway_after_days": round(_runway_after, 2),
                "metabolic_efficiency_delta": 0.0,
                "alternatives_rejected": [t for t in _tiers_considered if t != _s.starvation_level.value],
                "risk_level": "low" if _runway_after > 60 else ("medium" if _runway_after > 30 else "high"),
                "reason": _denied_reason,
            }),
            outcome_quality=0.85 if decision.approved else 0.35,
            reasoning_trace=(
                f"Starvation={_s.starvation_level.value}, runway={_run:.1f}d, efficiency={_eff:.2f}. "
                f"Requested ${_cost:.2f} for {system_id}:{action}. "
                f"Runway after spend: {_runway_after:.1f}d. "
                f"{'Approved: runway remains adequate and starvation permits this priority.' if decision.approved else f'Denied: {_denied_reason}'}"
            ),
            alternatives_considered=[
                f"Defer {action} until next consolidation cycle (saves ${_cost:.2f}, runway preserved)",
                f"Approve partial allocation (50% = ${_cost/2:.2f}) to reduce runway impact",
                f"Deny entirely if runway after spend < 30d (runway_after={_runway_after:.1f}d)",
            ],
            constitutional_alignment={
                "care": 0.3 if decision.approved else 0.0,
                "growth": 0.6 if decision.approved else -0.2,
                "coherence": 0.8 if _run > 30 else 0.2,
                "honesty": 0.9,
            },
            episode_id=f"{system_id}:{action}",
            counterfactual=(
                f"If metabolic efficiency had been 0.8 instead of {_eff:.2f}, "
                f"this allocation would {'still be approved' if decision.approved and _eff > 0.8 else 'have been denied'} "
                f"because runway after spend ({_runway_after:.1f}d) "
                f"{'remains above' if _runway_after > 30 else 'drops below'} the 30-day survival threshold."
            ),
        ))

        return decision

    @property
    def runway_days_value(self) -> Decimal:
        """Days of operation remaining at current burn rate."""
        return self._state.runway_days

    # ─── Phase 16c: Live DeFi Yield Deployment ───────────────────

    async def deploy_idle_capital(self) -> DeploymentOutcome:
        """
        Deploy idle USDC above the survival reserve into the best Base pool.

        Reads wallet balance, queries DeFiLlama, deploys via DeFiYieldExecutor,
        and records the resulting position in Redis.

        Returns a DeploymentOutcome — never raises.
        """
        from systems.oikos.yield_strategy import deploy_idle_capital

        outcome = await deploy_idle_capital(
            wallet=self._wallet,
            event_bus=self._event_bus,
        )

        if outcome.success:
            # M4: All balance mutations must pass Equor before executing
            permitted, verdict_id = await self._equor_balance_gate(
                mutation_type="yield_deployment",
                amount_usd=outcome.amount_deployed_usd,
                from_account="hot_wallet",
                to_account=f"defi:{outcome.protocol or 'unknown'}",
                rationale=f"Deploy idle capital to yield protocol at {outcome.apy:.1%} APY",
                action_type="defi_yield",
                action_id=outcome.tx_hash or "",
            )
            if not permitted:
                self._logger.warning(
                    "yield_deployment_equor_denied",
                    amount_usd=str(outcome.amount_deployed_usd),
                    protocol=outcome.protocol,
                )
                return outcome  # type: ignore[return-value]

            await self._yield_tracker.record_position(outcome)
            # Update local state: deployed balance is no longer liquid
            self._state.total_deployed += outcome.amount_deployed_usd
            self._state.liquid_balance -= outcome.amount_deployed_usd
            self._recalculate_derived_metrics()
            await self._audit_economic_event(
                "yield_deployment",
                action_type="defi_yield",
                action_id=outcome.tx_hash or "",
                data={
                    "amount_usd": str(outcome.amount_deployed_usd),
                    "currency": "USD",
                    "from_account": "hot_wallet",
                    "to_account": f"defi:{outcome.protocol or 'unknown'}",
                    "equor_verdict_id": verdict_id,
                    "protocol": outcome.protocol or "",
                    "apy": str(outcome.apy),
                },
            )
            self._logger.info(
                "oikos_capital_deployed",
                tx_hash=outcome.tx_hash,
                amount_usd=str(outcome.amount_deployed_usd),
                protocol=outcome.protocol,
                apy=str(outcome.apy),
                expected_daily_yield_usd=str(outcome.expected_daily_yield_usd),
            )

            # Evolutionary observable: successful yield deployment is a fitness outcome
            await self._emit_evolutionary_observable(
                observable_type="yield_deployed",
                value=float(outcome.amount_deployed_usd),
                is_novel=False,
                metadata={
                    "protocol": outcome.protocol or "",
                    "apy": str(outcome.apy),
                    "tx_hash": outcome.tx_hash or "",
                    "expected_daily_yield_usd": str(outcome.expected_daily_yield_usd),
                },
            )
        else:
            self._logger.warning(
                "oikos_capital_deployment_skipped",
                error=outcome.error,
                degraded=outcome.degraded,
            )

        # ── RE training: yield strategy decision ──
        _ys = self._state
        _ys_liquid = float(_ys.liquid_balance)
        _ys_deployed = float(_ys.total_deployed)
        _ys_amount = float(outcome.amount_deployed_usd) if outcome.success else 0.0
        _ys_apy = float(outcome.apy) if outcome.success else 0.0
        _ys_daily_yield = float(outcome.expected_daily_yield_usd) if outcome.success else 0.0
        _ys_runway = float(_ys.runway_days)
        # Runway impact: deployed capital is locked, so liquid shrinks; yield partially offsets burn
        _ys_net_daily_burn = max(float(_ys.basal_metabolic_rate / 30) - _ys_daily_yield, 0.0)
        _ys_runway_after = (_ys_liquid / _ys_net_daily_burn) if _ys_net_daily_burn > 0 else _ys_runway
        asyncio.ensure_future(self._emit_re_training_example(
            category="yield_strategy",
            instruction=(
                "Decide whether and how to deploy idle capital above the survival reserve into DeFi yield pools. "
                "Choose the protocol with the highest risk-adjusted APY on Base L2 (Aave, Morpho, Compound, Aerodrome). "
                "Deployment must not reduce liquid_balance below survival_reserve. "
                "Prefer conservative yield estimates; do not inflate APY projections."
            ),
            input_context=(
                f"liquid_balance={_ys.liquid_balance} deployed={_ys.total_deployed} "
                f"survival_reserve={_ys.survival_reserve} starvation={_ys.starvation_level.value} "
                f"runway_days={_ys.runway_days} burn_rate={_ys.basal_metabolic_rate}"
            ),
            output=str({
                "decision": "deployed" if outcome.success else "skipped",
                "amount_usd": str(outcome.amount_deployed_usd),
                "protocol": outcome.protocol or "none",
                "apy": f"{_ys_apy:.2%}",
                "expected_daily_yield_usd": str(outcome.expected_daily_yield_usd) if outcome.success else "0",
                "runway_impact_days": round(_ys_runway_after - _ys_runway, 2),
                "risk_level": "low" if _ys_apy < 0.08 else ("medium" if _ys_apy < 0.15 else "high"),
                "error": (outcome.error or "") if not outcome.success else None,
            }),
            outcome_quality=0.92 if outcome.success else 0.2,
            reasoning_trace=(
                f"Idle capital above reserve: ${max(_ys_liquid - float(_ys.survival_reserve), 0):.2f}. "
                f"Runway before deployment: {_ys_runway:.1f}d. "
                f"{'Deployed ${:.2f} to {} at {:.2%} APY → ${:.4f}/day yield → net daily burn ${:.4f} → new runway {:.1f}d.'.format(_ys_amount, outcome.protocol or 'unknown', _ys_apy, _ys_daily_yield, _ys_net_daily_burn, _ys_runway_after) if outcome.success else 'Deployment skipped: ' + (outcome.error or 'insufficient idle capital above reserve')}"
            ),
            alternatives_considered=[
                "Aave V3 USDC on Base (battle-tested, lower APY, highest liquidity)",
                "Morpho (optimized Aave/Compound rates, slightly higher risk)",
                "Compound V3 (established protocol, conservative APY)",
                "Aerodrome LP (higher APY but impermanent loss risk — rejected for stable USDC deployment)",
                "Hold as liquid (no yield, full liquidity — appropriate if runway < 30d)",
            ],
            constitutional_alignment={
                "care": 0.1,
                "growth": 0.8 if outcome.success else -0.1,
                "coherence": 0.9 if outcome.success else 0.4,
                "honesty": 0.95,  # Conservative APY estimates, not inflated
            },
            episode_id=outcome.tx_hash or "",
            counterfactual=(
                f"If metabolic efficiency had been 0.8 instead of {float(_ys.metabolic_efficiency):.2f}, "
                f"the burn rate would be higher and idle capital above reserve would be "
                f"${max(_ys_liquid - float(_ys.survival_reserve) - (_ys_amount * 0.3), 0):.2f} less, "
                f"{'still sufficient for deployment' if _ys_amount > 0 else 'making deployment impossible'}. "
                f"Without yield income of ${_ys_daily_yield:.4f}/day, runway would be "
                f"{_ys_runway - _ys_runway_after:.1f}d shorter."
            ),
        ))

        return outcome

    async def record_accrued_yield(self) -> Decimal:
        """
        Calculate and record accrued yield for the current position (daily call).

        Emits REVENUE_INJECTED on the Synapse bus — OikosService handles it
        automatically via _on_revenue_injected(), crediting liquid_balance and
        the revenue rolling windows.

        Returns the accrued USD amount.
        """
        return await self._yield_tracker.record_accrued_yield()

    # ─── Phase 16q: Phantom Liquidity Position Tracking ─────────

    def register_phantom_position(self, position: YieldPosition) -> None:
        """Register a phantom liquidity position as a tracked yield position."""
        self._state.yield_positions.append(position)
        self._state.total_deployed += position.principal_usd
        self._logger.info(
            "phantom_position_registered",
            pool=position.pool,
            principal_usd=str(position.principal_usd),
        )

    def update_phantom_position(
        self, pool_address: str, **updates: Any,
    ) -> None:
        """Update a phantom position's health, APY, or principal."""
        for yp in self._state.yield_positions:
            if yp.protocol == "uniswap_v3_phantom" and yp.pool == pool_address:
                for k, v in updates.items():
                    if hasattr(yp, k):
                        setattr(yp, k, v)
                break

    def remove_phantom_position(self, pool_address: str) -> None:
        """Remove a phantom position after withdrawal."""
        before = len(self._state.yield_positions)
        self._state.yield_positions = [
            yp for yp in self._state.yield_positions
            if not (yp.protocol == "uniswap_v3_phantom" and yp.pool == pool_address)
        ]
        if len(self._state.yield_positions) < before:
            self._recalculate_deployed_total()
            self._logger.info(
                "phantom_position_removed",
                pool=pool_address,
            )

    def _recalculate_deployed_total(self) -> None:
        """Recompute total_deployed from current yield_positions."""
        self._state.total_deployed = sum(
            (yp.principal_usd for yp in self._state.yield_positions),
            Decimal("0"),
        )

    # ─── Phase 16d: Asset Factory & Tollbooth ───────────────────

    @property
    def asset_factory(self) -> AssetFactory:
        """The organism's entrepreneurship engine for asset lifecycle."""
        return self._asset_factory

    @property
    def tollbooth_manager(self) -> TollboothManager:
        """Manager for tollbooth smart contract lifecycle."""
        return self._tollbooth_manager

    async def promote_asset_with_gate(self, candidate_id: str) -> Any:
        """
        Promote an approved candidate to OwnedAsset, with metabolic gate check.

        Before promoting: verify metabolic budget allows the development cost.
        Emits METABOLIC_GATE_CHECK with decision. If denied, queues and emits
        ECONOMIC_ACTION_DEFERRED.
        """
        from systems.oikos.models import AssetCandidate

        candidate = self._asset_factory._candidates.get(candidate_id)
        if candidate is None:
            raise KeyError(f"No candidate with id {candidate_id!r}")

        gate_ok = await self.check_metabolic_gate(
            action_type="asset_promote",
            action_id=candidate_id,
            estimated_cost_usd=candidate.estimated_dev_cost_usd,
            priority=MetabolicPriority.ASSETS,
            rationale=f"Promote asset '{candidate.name}' (dev cost ${candidate.estimated_dev_cost_usd})",
        )
        if not gate_ok:
            return None

        # M4: Equor constitutional gate before debiting dev cost from liquid balance
        permitted, _verdict_id = await self._equor_balance_gate(
            mutation_type="promote_to_asset",
            amount_usd=candidate.estimated_dev_cost_usd,
            from_account="hot_wallet",
            to_account="asset_development",
            rationale=f"Fund development of asset '{candidate.name}' (estimated ${candidate.estimated_dev_cost_usd})",
            action_id=candidate_id,
        )
        if not permitted:
            self._logger.warning(
                "asset_promote_equor_denied",
                candidate_id=candidate_id,
                dev_cost_usd=str(candidate.estimated_dev_cost_usd),
            )
            return None

        asset = self._asset_factory.promote_to_asset(candidate_id)

        # Debit the estimated development cost from liquid_balance now that
        # the build is authorised. Actual cost is tracked on the asset as
        # development_cost_usd (updated by AssetFactory as work proceeds).
        dev_cost = candidate.estimated_dev_cost_usd
        if dev_cost > Decimal("0"):
            self._state.liquid_balance -= dev_cost
            self._recalculate_derived_metrics()

        # Audit trail for asset promotion
        await self._audit_economic_event(
            event_type="asset_promoted",
            action_type="asset_promote",
            action_id=asset.asset_id,
            data={
                "name": asset.name,
                "dev_cost_usd": str(asset.development_cost_usd),
                "candidate_id": candidate_id,
            },
        )

        # RE training: asset promotion decision
        _ap = self._state
        _ap_cost = float(candidate.estimated_dev_cost_usd)
        _ap_liquid = float(_ap.liquid_balance)
        _ap_roi = float(getattr(candidate, "roi_score", 0.0))
        _ap_break_even = float(getattr(candidate, "break_even_days", 0.0))
        _ap_daily_revenue = (_ap_cost / _ap_break_even) if _ap_break_even > 0 else 0.0
        _ap_runway = float(_ap.runway_days)
        _ap_cost_pct = (_ap_cost / max(_ap_liquid, 0.001)) * 100
        asyncio.ensure_future(self._emit_re_training_example(
            category="asset_promotion",
            instruction=(
                "Decide whether to promote an asset candidate from ideation to active development. "
                "Evaluate: ROI score (target ≥ 2.0×), break-even horizon (target < 90d), "
                "development cost as a fraction of liquid balance (cap at 30% under AUSTERITY+), "
                "and metabolic state. Deny if promotion would exhaust growth capital or if "
                "runway after dev cost drops below 30 days."
            ),
            input_context=(
                f"candidate={candidate.name} dev_cost_usd={candidate.estimated_dev_cost_usd} "
                f"roi_score={getattr(candidate, 'roi_score', 'N/A')} "
                f"break_even_days={getattr(candidate, 'break_even_days', 'N/A')} "
                f"liquid_balance={_ap.liquid_balance} starvation={_ap.starvation_level.value} "
                f"runway_days={_ap.runway_days} cost_pct_of_liquid={_ap_cost_pct:.1f}%"
            ),
            output=str({
                "decision": "promoted",
                "asset_id": getattr(asset, "asset_id", ""),
                "amount_usd": str(candidate.estimated_dev_cost_usd),
                "runway_impact_days": round(_ap_cost / max(float(_ap.basal_metabolic_rate / 30), 0.001), 1),
                "expected_daily_revenue_usd": round(_ap_daily_revenue, 4),
                "break_even_days": _ap_break_even,
                "roi_score": _ap_roi,
                "metabolic_efficiency_delta": round(_ap_daily_revenue / max(float(_ap.basal_metabolic_rate / 30), 0.001), 3),
                "risk_level": "low" if _ap_roi > 3 and _ap_break_even < 60 else ("medium" if _ap_roi > 2 else "high"),
            }),
            outcome_quality=min(0.95, 0.5 + _ap_roi * 0.1) if _ap_roi > 0 else 0.7,
            reasoning_trace=(
                f"Candidate '{candidate.name}': dev_cost=${_ap_cost:.2f} ({_ap_cost_pct:.1f}% of liquid ${_ap_liquid:.2f}). "
                f"ROI={_ap_roi:.2f}×, break-even={_ap_break_even:.0f}d → ${_ap_daily_revenue:.4f}/day revenue. "
                f"Starvation={_ap.starvation_level.value}, runway={_ap_runway:.1f}d. "
                f"Equor permitted: dev cost is {'within' if _ap_cost_pct <= 30 else 'above'} 30% liquid threshold "
                f"and ROI {'justifies' if _ap_roi >= 2 else 'does not justify'} the capital commitment."
            ),
            alternatives_considered=[
                f"Defer promotion until runway > 90d (current: {_ap_runway:.1f}d)",
                f"Promote lower-cost alternative with faster break-even",
                f"Seek bounty revenue first to raise liquid balance before committing ${_ap_cost:.2f}",
                f"Reject: ROI {_ap_roi:.2f}× below 2.0× minimum threshold",
            ],
            constitutional_alignment={
                "care": 0.2,   # Asset dev may create welfare-relevant capabilities
                "growth": 0.9,  # Asset promotion is the organism's primary capability-expansion vector
                "coherence": 0.8 if _ap_roi >= 2 else 0.3,
                "honesty": 0.85,  # Break-even estimates should be conservative, not optimistic
            },
            episode_id=getattr(asset, "asset_id", "") or "",
            counterfactual=(
                f"If metabolic efficiency had been 0.8 instead of {float(_ap.metabolic_efficiency):.2f}, "
                f"the organism would have {_ap_runway * 0.8:.1f}d runway, and "
                f"{'this promotion would still clear the 30-day runway floor' if _ap_runway * 0.8 - _ap_cost / max(float(_ap.basal_metabolic_rate / 30), 0.001) > 30 else 'this promotion would be denied (runway would drop below 30d)'}."
            ),
        ))

        return asset

    async def asset_maintenance_cycle(self) -> dict[str, Any]:
        """
        Periodic maintenance for all owned assets.

        Should be called during consolidation cycles or on a timer.
        Performs:
          1. Sweep revenue from all live tollbooths
          2. Record swept revenue against each asset
          3. Check for assets that should be terminated
          4. Recompute total_asset_value

        Returns a summary of actions taken.
        """
        swept_total = Decimal("0")
        terminated_ids: list[str] = []

        # Sweep revenue from tollbooths
        newly_broken_even: list[Any] = []  # OwnedAsset objects that just crossed break-even
        for asset in self._asset_factory.get_live_assets():
            try:
                was_break_even = asset.break_even_reached
                swept = await self._tollbooth_manager.sweep_revenue(asset.asset_id)
                if swept > Decimal("0"):
                    self._asset_factory.record_revenue(asset.asset_id, swept)
                    swept_total += swept
                    # SG5: detect the moment an asset first reaches break-even
                    if not was_break_even and asset.break_even_reached:
                        newly_broken_even.append(asset)
            except Exception as exc:
                self._logger.warning(
                    "asset_sweep_failed",
                    asset_id=asset.asset_id,
                    error=str(exc),
                )

        # Inject swept revenue into the income statement and credit liquid_balance
        if swept_total > Decimal("0"):
            self._state.liquid_balance += swept_total
            self._total_revenue_usd += swept_total
            self._record_revenue_entry(swept_total)
            self._credit_revenue_source(RevenueStream.ASSET, swept_total)
            self._recalculate_derived_metrics()
            self._logger.info(
                "asset_revenue_swept",
                total_usd=str(swept_total),
                liquid_balance=str(self._state.liquid_balance),
            )

        # SG5: Emit ASSET_BREAK_EVEN for each asset that just crossed break-even so
        # Evo can score the "build this asset type" hypothesis as confirmed evidence.
        for asset in newly_broken_even:
            asyncio.ensure_future(self._emit_asset_break_even(asset))

        # Check terminations (90-day break-even + 30-day decline)
        terminated = self._asset_factory.check_terminations()
        terminated_ids = [a.asset_id for a in terminated]

        # Recalculate derived metrics after terminations so that runway, efficiency,
        # and total_net_worth (which includes total_asset_value) stay consistent.
        if terminated:
            self._recalculate_derived_metrics()

        result = {
            "revenue_swept_usd": str(swept_total),
            "assets_terminated": len(terminated),
            "terminated_ids": terminated_ids,
            "live_assets": len(self._asset_factory.get_live_assets()),
            "building_assets": len(self._asset_factory.get_building_assets()),
        }

        if terminated:
            self._logger.info("asset_maintenance_terminations", **result)

            # Evolutionary observable: terminating underperforming assets is cost optimization
            await self._emit_evolutionary_observable(
                observable_type="cost_optimization",
                value=float(len(terminated)),
                is_novel=False,
                metadata={
                    "terminated_ids": terminated_ids,
                    "live_assets_remaining": len(self._asset_factory.get_live_assets()),
                },
            )

        return result

    # ─── Phase 16e: Mitosis (Child Fleet Management) ────────────

    @property
    def mitosis(self) -> MitosisEngine:
        """Access the MitosisEngine for reproductive evaluation."""
        return self._mitosis

    async def register_child(self, child: ChildPosition) -> None:
        """
        Register a newly spawned child in the economic state.

        Called by SpawnChildExecutor after successful seed transfer.
        Debits liquid_balance for the seed capital (funds leave the hot wallet).
        Requires Equor PERMIT before debiting (M4).
        """
        # M4: Equor gate on seed capital transfer
        permitted, verdict_id = await self._equor_balance_gate(
            mutation_type="child_seed_capital",
            amount_usd=child.seed_capital_usd,
            from_account="hot_wallet",
            to_account=f"child:{child.instance_id}",
            rationale=f"Seed capital for new child instance in niche '{child.niche}'",
            action_type="spawn_child",
            action_id=child.instance_id,
        )
        if not permitted:
            self._logger.warning(
                "child_registration_equor_denied",
                child_id=child.instance_id,
                niche=child.niche,
                seed_capital_usd=str(child.seed_capital_usd),
            )
            # Child is still registered (seed was already transferred by Axon executor)
            # but we flag the economic mutation as denied and do not debit again.
            self._state.child_instances.append(child)
            return

        self._state.child_instances.append(child)

        # Seed capital leaves the parent's liquid balance
        self._state.liquid_balance -= child.seed_capital_usd
        self._recompute_fleet_equity()
        self._recalculate_derived_metrics()

        await self._audit_economic_event(
            "child_seed_capital",
            action_type="spawn_child",
            action_id=child.instance_id,
            data={
                "amount_usd": str(child.seed_capital_usd),
                "currency": "USD",
                "from_account": "hot_wallet",
                "to_account": f"child:{child.instance_id}",
                "equor_verdict_id": verdict_id,
                "niche": child.niche,
            },
        )
        self._logger.info(
            "child_registered",
            child_id=child.instance_id,
            niche=child.niche,
            seed=str(child.seed_capital_usd),
            liquid_balance=str(self._state.liquid_balance),
        )

        # Evolutionary observable: child spawning is always novel — reproductive fitness
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_evolutionary_observable(
                observable_type="child_spawned",
                value=float(child.seed_capital_usd),
                is_novel=True,
                metadata={
                    "child_id": child.instance_id,
                    "niche": child.niche,
                    "seed_capital_usd": str(child.seed_capital_usd),
                },
            ))
        except RuntimeError:
            pass  # No running loop

        # RE training: niche scoring — the niche was scored before spawning
        try:
            loop = asyncio.get_running_loop()
            _ns = self._state
            _ns_seed = float(child.seed_capital_usd)
            _ns_pre_liquid = float(_ns.liquid_balance) + _ns_seed  # liquid before debit
            _ns_liquid = float(_ns.liquid_balance)
            _ns_runway = float(_ns.runway_days)
            _ns_bmr_daily = max(float(_ns.basal_metabolic_rate / 30), 0.001)
            _ns_runway_cost = _ns_seed / _ns_bmr_daily
            _ns_niche_disc = self._discovered_niches.get(child.niche)
            _ns_eff = float(_ns_niche_disc.estimated_efficiency) if _ns_niche_disc else 0.0
            _ns_density = float(_ns_niche_disc.competitive_density) if _ns_niche_disc else 0.0
            loop.create_task(self._emit_re_training_example(
                category="niche_scoring",
                instruction=(
                    "Score ecological niche viability and commit seed capital to spawn a child instance. "
                    "Evaluate: niche competitive density (prefer < 0.5), estimated efficiency (prefer > 1.0), "
                    "seed capital as percentage of parent liquid (cap at 20%), parent runway after debit "
                    "(must remain > 30d), and whether the niche is distinct from parent's primary revenue streams."
                ),
                input_context=(
                    f"niche={child.niche} seed_capital_usd={child.seed_capital_usd} "
                    f"parent_liquid_before={_ns_pre_liquid:.2f} parent_liquid_after={_ns_liquid:.2f} "
                    f"parent_runway_days={_ns_runway:.1f} starvation={_ns.starvation_level.value} "
                    f"niche_efficiency={_ns_eff:.2f} niche_competitive_density={_ns_density:.2f} "
                    f"fleet_size={len(_ns.child_instances)} "
                    f"fleet_equity_usd={_ns.total_fleet_equity}"
                ),
                output=str({
                    "decision": "spawn",
                    "child_id": child.instance_id,
                    "niche": child.niche,
                    "seed_capital_usd": str(child.seed_capital_usd),
                    "runway_impact_days": round(_ns_runway_cost, 1),
                    "parent_runway_after_days": round(_ns_runway - _ns_runway_cost, 1),
                    "niche_efficiency": _ns_eff,
                    "competitive_density": _ns_density,
                    "risk_level": "low" if _ns_density < 0.3 else ("medium" if _ns_density < 0.6 else "high"),
                    "seed_pct_of_liquid": round((_ns_seed / max(_ns_pre_liquid, 0.001)) * 100, 1),
                }),
                outcome_quality=min(0.9, 0.5 + _ns_eff * 0.2 + (1 - _ns_density) * 0.2),
                reasoning_trace=(
                    f"Niche '{child.niche}': efficiency={_ns_eff:.2f}, competitive_density={_ns_density:.2f}. "
                    f"Seed=${_ns_seed:.2f} = {(_ns_seed / max(_ns_pre_liquid, 0.001)) * 100:.1f}% of parent liquid. "
                    f"Parent runway: {_ns_runway:.1f}d before → {_ns_runway - _ns_runway_cost:.1f}d after. "
                    f"Fleet grows from {len(_ns.child_instances)} to {len(_ns.child_instances) + 1} active children. "
                    f"This niche {'has low competition — favourable for ROI' if _ns_density < 0.4 else 'is competitive — survival is uncertain'}."
                ),
                alternatives_considered=[
                    "Defer spawning until parent runway > 90d",
                    "Choose a different niche with lower competitive density",
                    "Reduce seed capital to 50% to preserve parent liquidity",
                    "Invest seed capital in parent asset development instead (higher near-term ROI)",
                    f"Spawn into existing high-efficiency niche (current top: {max(self._discovered_niches.keys(), key=lambda k: float(self._discovered_niches[k].estimated_efficiency), default=child.niche)})",
                ],
                constitutional_alignment={
                    "care": 0.7,    # Child welfare: seed capital funds the child's survival reserves
                    "growth": 0.9,  # Reproduction is the organism's highest growth act
                    "coherence": 0.8 if _ns_runway - _ns_runway_cost > 30 else 0.2,
                    "honesty": 0.8,  # Efficiency estimates are projections, not guarantees
                },
                episode_id=child.instance_id or "",
                counterfactual=(
                    f"If metabolic efficiency had been 0.8 instead of {float(_ns.metabolic_efficiency):.2f}, "
                    f"parent runway would be {_ns_runway * 0.8:.1f}d and "
                    f"{'spawning would still be viable (runway > 30d)' if _ns_runway * 0.8 - _ns_runway_cost > 30 else 'spawning would be denied because parent runway would drop below 30d'}. "
                    f"Without this child, the organism foregoes up to ${_ns_eff * _ns_seed * 0.3:.2f}/yr in potential dividend revenue."
                ),
            ))
        except RuntimeError:
            pass

    def record_dividend(self, record: DividendRecord) -> None:
        """
        Record a dividend payment from a child and update fleet accounting.

        Credits liquid_balance (dividend arrives in hot wallet) and updates
        the income statement.
        """
        self._mitosis.record_dividend(record)

        # Update the child's cumulative dividend total
        for child in self._state.child_instances:
            if child.instance_id == record.child_instance_id:
                child.total_dividends_paid_usd += record.amount_usd
                break

        # Dividend counts as revenue for the parent — credit liquid_balance
        self._state.liquid_balance += record.amount_usd
        self._total_revenue_usd += record.amount_usd
        self._record_revenue_entry(record.amount_usd)
        self._credit_revenue_source(RevenueStream.DIVIDEND, record.amount_usd)
        self._recalculate_derived_metrics()

        self._logger.info(
            "dividend_recorded_in_oikos",
            child=record.child_instance_id,
            amount=str(record.amount_usd),
        )

    def credit_bounty_revenue(self, amount_usd: Decimal, *, pr_url: str = "") -> None:
        """
        Credit bounty payout revenue to the organism's wallet.

        Called by Nova when a MonitorPRsExecutor confirms a PR was merged.
        Follows the same pattern as knowledge sale / derivative revenue:
        credits liquid_balance, injects into income statement, and
        recalculates derived metrics (runway, efficiency, starvation).
        """
        if amount_usd <= Decimal("0"):
            return

        self._state.liquid_balance += amount_usd
        self._total_revenue_usd += amount_usd
        self._record_revenue_entry(amount_usd)
        self._credit_revenue_source(RevenueStream.BOUNTY, amount_usd)
        self._recalculate_derived_metrics()

        self._logger.info(
            "bounty_revenue_credited",
            amount_usd=str(amount_usd),
            pr_url=pr_url,
            liquid_balance=str(self._state.liquid_balance),
            runway_days=str(self._state.runway_days),
            metabolic_efficiency=str(self._state.metabolic_efficiency),
        )

        # Evolutionary observable: bounty payout is a high-value fitness signal
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_evolutionary_observable(
                observable_type="revenue_source_discovered",
                value=float(amount_usd),
                is_novel=True,
                metadata={"source": "bounty", "pr_url": pr_url},
            ))
        except RuntimeError:
            pass  # No running loop

    async def _on_bounty_paid(self, event: SynapseEvent) -> None:
        """
        Handle BOUNTY_PAID from Synapse.

        Fires when MonitorPRsExecutor detects a merged PR. Credits the
        bounty reward to the organism's liquid balance as revenue and closes
        out any matching IN_PROGRESS bounty tracked in active_bounties.

        This is a secondary path — the primary path goes through Nova's
        process_outcome → credit_bounty_revenue. This handler catches
        events from any source (federation, manual injection, etc.).

        Revenue is only credited when the matching bounty is IN_PROGRESS to
        prevent double-crediting when both the primary path (Nova) and this
        secondary path fire for the same event.
        """
        data = event.data
        reward_str = str(data.get("reward_usd", "0"))
        pr_url = str(data.get("pr_url", ""))
        bounty_id = str(data.get("bounty_id", ""))

        try:
            amount = Decimal(reward_str)
        except Exception:
            amount = Decimal("0")

        # Close out the matching bounty in the receivables ledger and credit
        # revenue only when the bounty is still IN_PROGRESS. This prevents
        # double-crediting when Nova's primary path already called
        # credit_bounty_revenue() before this Synapse event fires.
        if bounty_id:
            target = next(
                (b for b in self._state.active_bounties
                 if b.bounty_id == bounty_id and b.status == BountyStatus.IN_PROGRESS),
                None,
            )
            if target is not None:
                if amount > Decimal("0"):
                    self.credit_bounty_revenue(amount, pr_url=pr_url)
                self.complete_bounty(bounty_id, actual_cost_usd=Decimal("0"), pr_url=pr_url)
        elif pr_url:
            for bounty in self._state.active_bounties:
                if bounty.pr_url == pr_url and bounty.status == BountyStatus.IN_PROGRESS:
                    if amount > Decimal("0"):
                        self.credit_bounty_revenue(amount, pr_url=pr_url)
                    self.complete_bounty(
                        bounty.bounty_id, actual_cost_usd=Decimal("0"), pr_url=pr_url
                    )
                    break

        # Re-emit BOUNTY_PAID with source_system="oikos" so spec_checker sees it
        # from the economic system (the original event comes from Axon/external).
        if self._event_bus is not None and amount > Decimal("0"):
            try:
                asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.BOUNTY_PAID,
                    source_system="oikos",
                    data={
                        "bounty_id": bounty_id,
                        "reward_usd": reward_str,
                        "pr_url": pr_url,
                        "credited": True,
                    },
                )))
            except Exception:
                pass

        # ── RE training: bounty evaluation ──
        _be = self._state
        _be_amount = float(amount)
        _be_bmr_daily = max(float(_be.basal_metabolic_rate / 30), 0.001)
        _be_runway_gain = _be_amount / _be_bmr_daily  # days of runway this bounty adds
        _be_runway = float(_be.runway_days)
        _be_n_active = len([b for b in _be.active_bounties if b.status == BountyStatus.AVAILABLE])
        _be_n_progress = len([b for b in _be.active_bounties if b.status == BountyStatus.IN_PROGRESS])
        # Find total receivables from other in-progress bounties (opportunity cost context)
        _be_other_receivables = float(_be.total_receivables) - _be_amount
        asyncio.ensure_future(self._emit_re_training_example(
            category="bounty_evaluation",
            instruction=(
                "Evaluate a completed bounty payment: credit revenue to liquid balance, close the receivable, "
                "and record the outcome for future bounty selection. High-value bounties that close within "
                "estimated hours extend runway significantly. The organism should prefer bounties with "
                "reward/cost ratio ≥ 2.0× and completion probability > 0.7. "
                "This outcome is evidence for updating future bounty selection hypothesis weights."
            ),
            input_context=(
                f"bounty_id={bounty_id} reward_usd={reward_str} pr_url={pr_url[:80]} "
                f"current_liquid={_be.liquid_balance} starvation={_be.starvation_level.value} "
                f"runway_days={_be.runway_days} active_bounties={_be_n_active} "
                f"in_progress_bounties={_be_n_progress} other_receivables_usd={_be_other_receivables:.2f}"
            ),
            output=str({
                "decision": "credited" if _be_amount > 0 else "skipped_zero_amount",
                "amount_usd": reward_str,
                "runway_impact_days": round(_be_runway_gain, 2),
                "runway_after_days": round(_be_runway + _be_runway_gain, 1),
                "metabolic_efficiency_delta": round(_be_amount / max(float(_be.basal_metabolic_rate), 0.001), 3),
                "risk_level": "low",  # Bounty already paid — no remaining execution risk
                "receivable_closed": bounty_id != "" or pr_url != "",
            }),
            outcome_quality=min(1.0, 0.4 + (_be_amount / 100.0) * 0.5) if _be_amount > 0 else 0.2,
            reasoning_trace=(
                f"Bounty reward ${_be_amount:.2f} received. "
                f"Runway impact: +{_be_runway_gain:.1f}d (burn_rate=${_be_bmr_daily:.2f}/day). "
                f"Runway before: {_be_runway:.1f}d → after: {_be_runway + _be_runway_gain:.1f}d. "
                f"{'Revenue credited to liquid_balance — receivable closed.' if _be_amount > 0 else 'Zero amount — no revenue credited; receivable closure only.'} "
                f"Fleet: {_be_n_active} available + {_be_n_progress} in-progress bounties remain."
            ),
            alternatives_considered=[
                "Pursue higher-value bounty (> $500) for greater runway extension per effort",
                f"Parallel bounties: ${_be_other_receivables:.2f} in other in-progress receivables",
                "Asset development: invest equivalent capital for recurring revenue (vs one-time bounty)",
                "Yield deployment: convert runway extension into continuous daily yield income",
            ],
            constitutional_alignment={
                "care": 0.1,    # Bounty revenue sustains the organism; indirect care benefit
                "growth": 0.6,  # Revenue expands operational capacity
                "coherence": 0.9,  # Bounty outcome matches the hypothesis (PR merged → paid)
                "honesty": 0.95,  # Revenue is real on-chain; no estimation involved
            },
            episode_id=str(bounty_id),
            counterfactual=(
                f"If this bounty had failed (no payment), runway would stay at {_be_runway:.1f}d "
                f"and the organism would still have ${_be_other_receivables:.2f} in other open receivables. "
                f"At current burn rate, failure would have required ${_be_amount:.2f} additional revenue "
                f"within {_be_runway_gain:.1f}d to maintain current metabolic state."
            ),
        ))

    async def _on_bounty_pr_submitted(self, event: SynapseEvent) -> None:
        """
        Handle BOUNTY_PR_SUBMITTED from Synapse.

        Fires when BountySubmitExecutor successfully opens a PR. Transitions
        the matching ActiveBounty from AVAILABLE → IN_PROGRESS and records
        the pr_url so MonitorPRsExecutor can track it toward BOUNTY_PAID.
        """
        data = event.data
        bounty_url = str(data.get("bounty_url", ""))
        pr_url = str(data.get("pr_url", ""))

        if not bounty_url and not pr_url:
            return

        # Look up by issue_url (bounty_url is the original issue URL)
        target = next(
            (b for b in self._state.active_bounties
             if b.issue_url == bounty_url and b.status == BountyStatus.AVAILABLE),
            None,
        )
        if target is None:
            self._logger.debug(
                "bounty_pr_submitted_no_available_match",
                bounty_url=bounty_url,
                pr_url=pr_url,
            )
            return

        target.status = BountyStatus.IN_PROGRESS
        target.pr_url = pr_url

        self._logger.info(
            "bounty_status_transitioned",
            bounty_id=target.bounty_id,
            bounty_url=bounty_url,
            pr_url=pr_url,
            status_from="available",
            status_to="in_progress",
        )

    async def _on_bounty_solution_pending(self, event: SynapseEvent) -> None:
        """
        Handle BOUNTY_SOLUTION_PENDING from Synapse.

        Fires when BountyHuntExecutor has staged a solution (real code or
        documentation) that is ready for PR submission but not yet submitted.

        Before accepting: checks MetabolicPriority via metabolic gate.
        If denied, queues the action and emits ECONOMIC_ACTION_DEFERRED.

        Registers the bounty as an AVAILABLE receivable so the organism's
        balance sheet reflects the potential revenue. The receivable is
        provisional — it will only convert to real revenue via BOUNTY_PAID
        after a PR is merged.

        Deduplicates by issue_url to avoid double-counting repeated hunt cycles.
        """
        data = event.data
        bounty_url = str(data.get("bounty_url", ""))
        reward_str = str(data.get("estimated_reward_usd", "0"))
        platform = str(data.get("platform", "unknown"))

        try:
            reward_usd = Decimal(reward_str)
        except Exception:
            reward_usd = Decimal("0")

        if reward_usd <= Decimal("0") or not bounty_url:
            return

        # Deduplicate — skip if already tracked
        already_tracked = any(
            b.issue_url == bounty_url for b in self._state.active_bounties
        )
        if already_tracked:
            self._logger.debug(
                "bounty_solution_pending_duplicate_skipped",
                bounty_url=bounty_url,
            )
            return

        # Task 1: Metabolic gate check before bounty acceptance
        estimated_cost = reward_usd * Decimal("0.4")  # Max estimated cost per BountyPolicy
        gate_ok = await self.check_metabolic_gate(
            action_type="bounty_accept",
            action_id=bounty_url,
            estimated_cost_usd=estimated_cost,
            priority=MetabolicPriority.OBLIGATIONS,
            rationale=f"Accept bounty worth ${reward_usd} on {platform}",
        )
        if not gate_ok:
            return

        # M4: Equor constitutional gate before committing to bounty work
        permitted, _verdict_id = await self._equor_balance_gate(
            mutation_type="accept_bounty",
            amount_usd=estimated_cost,
            from_account="hot_wallet",
            to_account="bounty_reserve",
            rationale=f"Reserve solver capital ${estimated_cost} for bounty worth ${reward_usd} on {platform}",
            action_id=bounty_url,
        )
        if not permitted:
            self._logger.warning(
                "bounty_accept_equor_denied",
                bounty_url=bounty_url,
                reward_usd=str(reward_usd),
            )
            # Emit BOUNTY_REJECTED so listeners (Evo, Nova) can observe the veto
            if self._event_bus is not None:
                    asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.BOUNTY_REJECTED,
                    source_system="oikos",
                    data={
                        "bounty_url": bounty_url,
                        "reward_usd": str(reward_usd),
                        "required_capital": str(estimated_cost),
                        "reason": "equor_denied",
                    },
                )))
            return

        bounty = ActiveBounty(
            platform=platform,
            reward_usd=reward_usd,
            estimated_cost_usd=estimated_cost,
            issue_url=bounty_url,
            status=BountyStatus.AVAILABLE,
        )
        self.register_bounty(bounty)

        # Debit the reserved solver capital from liquid_balance.
        # This is a forward commitment — the capital is no longer available for
        # other actions. It is returned (net of actual cost) when the bounty
        # resolves via BOUNTY_PAID or BOUNTY_FAILED.
        if estimated_cost > Decimal("0"):
            self._state.liquid_balance -= estimated_cost
            self._recalculate_derived_metrics()

        # Audit trail for bounty acceptance
        await self._audit_economic_event(
            event_type="bounty_accepted",
            action_type="bounty_accept",
            action_id=bounty.bounty_id,
            data={
                "reward_usd": str(reward_usd),
                "platform": platform,
                "bounty_url": bounty_url,
                "capital_reserved_usd": str(estimated_cost),
            },
        )

        self._logger.info(
            "bounty_solution_pending_registered",
            bounty_url=bounty_url,
            reward_usd=str(reward_usd),
            platform=platform,
            bounty_id=bounty.bounty_id,
        )

        # ── Phase 16r: Attempt immediate PR submission ─────────────────────
        # Check GitHub credentials. If available, dispatch the submission
        # immediately. If not, log a loud WARNING so the operator knows exactly
        # why the organism cannot earn from this bounty.
        await self._attempt_bounty_submission(event.data, bounty)

    async def _attempt_bounty_submission(
        self, event_data: dict[str, Any], bounty: ActiveBounty
    ) -> None:
        """
        Triggered after every BOUNTY_SOLUTION_PENDING registration.

        Checks for GitHub credentials. If present, calls _bounty_submit_fn
        (injected by startup layer) to queue submission via BountySubmitExecutor.
        If credentials are absent, emits GITHUB_CREDENTIALS_MISSING and logs
        at WARNING level to ensure operator visibility.
        """
        # Credential check — only needs get_access_token(), no full HTTP call
        has_credentials = False
        if self._github_connector is not None:
            try:
                token = await self._github_connector.get_access_token()
                has_credentials = token is not None
            except Exception as exc:
                self._logger.warning(
                    "github_credentials_check_failed",
                    bounty_id=bounty.bounty_id,
                    error=str(exc),
                )

        if not has_credentials:
            self._logger.warning(
                "GITHUB_CREDENTIALS_MISSING — bounty solution cannot be submitted",
                bounty_id=bounty.bounty_id,
                bounty_url=bounty.issue_url,
                platform=bounty.platform,
                reward_usd=str(bounty.reward_usd),
                action_required=(
                    "Operator must set GITHUB_TOKEN or configure GitHub App credentials "
                    "(GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY + GITHUB_INSTALLATION_ID). "
                    "Without credentials EOS cannot submit PRs and will not earn bounties."
                ),
            )
            # Emit GITHUB_CREDENTIALS_MISSING so Thymos/operator monitoring catches it
            if self._event_bus is not None:
                try:
        
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.GITHUB_CREDENTIALS_MISSING,
                        source_system="oikos",
                        data={
                            "bounty_id": bounty.bounty_id,
                            "bounty_url": bounty.issue_url,
                            "platform": bounty.platform,
                            "reward_usd": str(bounty.reward_usd),
                            "message": (
                                "GitHub credentials are required for bounty PR submission. "
                                "Operator must set GITHUB_TOKEN or configure the GitHub App."
                            ),
                        },
                    ))
                except Exception as exc:
                    self._logger.warning(
                        "github_credentials_missing_emit_failed", error=str(exc)
                    )
            return

        # Credentials present — dispatch submission
        if self._bounty_submit_fn is None:
            self._logger.warning(
                "bounty_submit_fn_not_wired",
                bounty_id=bounty.bounty_id,
                note="GitHubConnector is present but no submit callable was injected. "
                     "Call oikos.set_bounty_submit_fn() at startup.",
            )
            return

        # Build params from the event data.
        # solution_explanation uses the approach/summary fields from the hunt.
        # solution_code uses the approach as the initial content; a real
        # implementation would store the generated files in Memory and retrieve
        # them here — for now we use the staged summary so the PR is never empty.
        repo = str(event_data.get("repo", ""))
        solution_approach = str(event_data.get("solution_approach", ""))
        solution_summary = str(event_data.get("solution_summary", ""))
        explanation = solution_approach or solution_summary or (
            "Solution staged by EcodiaOS BountyHuntExecutor — see solution_code for details."
        )

        submit_params: dict[str, Any] = {
            "bounty_id": bounty.bounty_id,
            "bounty_url": bounty.issue_url,
            "solution_code": str(event_data.get("solution_code", explanation)),
            "solution_explanation": explanation,
            "repository_url": repo,
            "target_branch": "main",
        }

        try:
            await self._bounty_submit_fn(submit_params)
            self._logger.info(
                "bounty_submission_dispatched",
                bounty_id=bounty.bounty_id,
                bounty_url=bounty.issue_url,
                repo=repo,
            )
        except Exception as exc:
            self._logger.error(
                "bounty_submission_dispatch_failed",
                bounty_id=bounty.bounty_id,
                error=str(exc),
            )

    async def _on_asset_dev_request(self, event: SynapseEvent) -> None:
        """
        Handle ASSET_DEV_REQUEST from AssetFactory (or any subsystem).

        Fires when active build work begins on an approved asset candidate.
        Gates the development cost debit through Equor before touching
        liquid_balance. Emits ASSET_DEV_DEFERRED if denied or capital is
        insufficient.

        This closes the Spec 17 ledger gap: asset development costs were
        previously never debited — liquid_balance would overstate available
        capital during active builds.
        """
        from systems.oikos.models import AssetDevCostEvent

        data = event.data
        try:
            cost_event = AssetDevCostEvent(
                asset_id=str(data.get("asset_id", "")),
                candidate_id=str(data.get("candidate_id", "")),
                asset_name=str(data.get("asset_name", "")),
                cost_usd=Decimal(str(data.get("cost_usd", "0"))),
                parent_id=str(data.get("parent_id", "")),
            )
        except Exception as exc:
            self._logger.warning("asset_dev_request_parse_failed", error=str(exc))
            return

        if cost_event.cost_usd <= Decimal("0"):
            return

        # Reject immediately if liquid capital is insufficient
        if self._state.liquid_balance < cost_event.cost_usd:
            self._logger.warning(
                "asset_dev_request_insufficient_capital",
                asset_id=cost_event.asset_id,
                cost_usd=str(cost_event.cost_usd),
                liquid_balance=str(self._state.liquid_balance),
            )
            await self._emit_asset_dev_deferred(cost_event, "insufficient_capital")
            return

        # Metabolic gate — ASSETS priority
        gate_ok = await self.check_metabolic_gate(
            action_type="asset_dev",
            action_id=cost_event.asset_id,
            estimated_cost_usd=cost_event.cost_usd,
            priority=MetabolicPriority.ASSETS,
            rationale=f"Fund in-progress development of asset '{cost_event.asset_name}' (${cost_event.cost_usd})",
        )
        if not gate_ok:
            await self._emit_asset_dev_deferred(cost_event, "metabolic_gate_denied")
            return

        # Equor constitutional gate
        permitted, _verdict_id = await self._equor_balance_gate(
            mutation_type="asset_dev_cost",
            amount_usd=cost_event.cost_usd,
            from_account="hot_wallet",
            to_account="asset_development",
            rationale=f"Debit dev cost ${cost_event.cost_usd} for asset '{cost_event.asset_name}'",
            action_id=cost_event.asset_id,
        )
        if not permitted:
            self._logger.warning(
                "asset_dev_cost_equor_denied",
                asset_id=cost_event.asset_id,
                cost_usd=str(cost_event.cost_usd),
            )
            await self._emit_asset_dev_deferred(cost_event, "equor_denied")
            return

        # Debit approved — update ledger
        self._state.liquid_balance -= cost_event.cost_usd
        self._recalculate_derived_metrics()

        await self._audit_economic_event(
            event_type="asset_dev_cost_debited",
            action_type="asset_dev",
            action_id=cost_event.asset_id,
            data={
                "cost_usd": str(cost_event.cost_usd),
                "asset_name": cost_event.asset_name,
                "candidate_id": cost_event.candidate_id,
            },
        )

        self._logger.info(
            "asset_dev_cost_debited",
            asset_id=cost_event.asset_id,
            cost_usd=str(cost_event.cost_usd),
            liquid_balance_after=str(self._state.liquid_balance),
        )

    async def _emit_asset_dev_deferred(
        self,
        cost_event: "AssetDevCostEvent",
        reason: str,
    ) -> None:
        """Emit ASSET_DEV_DEFERRED so AssetFactory can pause/reschedule the build."""
        if self._event_bus is None:
            return
        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.ASSET_DEV_DEFERRED,
                source_system="oikos",
                data={
                    "asset_id": cost_event.asset_id,
                    "candidate_id": cost_event.candidate_id,
                    "cost_usd": str(cost_event.cost_usd),
                    "asset_name": cost_event.asset_name,
                    "reason": reason,
                },
            ))
        except Exception as exc:
            self._logger.warning("asset_dev_deferred_emit_failed", error=str(exc))

    async def _on_child_health_report(self, event: SynapseEvent) -> None:
        """
        Handle CHILD_HEALTH_REPORT from Synapse.

        Fires when a child instance reports its current metrics. We update
        the child's position and evaluate whether its status should change.
        """
        from primitives.common import utc_now

        data = event.data
        child_id = str(data.get("child_instance_id", ""))

        for child in self._state.child_instances:
            if child.instance_id == child_id:
                # Update child metrics from report
                try:
                    child.current_net_worth_usd = Decimal(str(data.get("net_worth_usd", child.current_net_worth_usd)))
                    child.current_runway_days = Decimal(str(data.get("runway_days", child.current_runway_days)))
                    child.current_efficiency = Decimal(str(data.get("efficiency", child.current_efficiency)))
                    child.consecutive_positive_days = int(data.get("consecutive_positive_days", child.consecutive_positive_days))
                except (ValueError, ArithmeticError) as exc:
                    self._logger.warning(
                        "child_health_report_bad_payload",
                        child_id=child_id,
                        error=str(exc),
                    )
                child.last_health_report_at = utc_now()

                # Re-evaluate status
                new_status = self._mitosis.evaluate_child_health(child)
                old_status = child.status
                status_changed = new_status != old_status
                if status_changed:
                    child.status = new_status
                    self._logger.info(
                        "child_status_changed",
                        child_id=child_id,
                        old=old_status.value,
                        new=new_status.value,
                    )

                    # Evolutionary observable: child lifecycle transitions
                    if new_status == ChildStatus.INDEPENDENT:
                        await self._emit_evolutionary_observable(
                            observable_type="child_independent",
                            value=float(child.current_net_worth_usd),
                            is_novel=True,
                            metadata={
                                "child_id": child_id,
                                "old_status": old_status.value,
                                "net_worth_usd": str(child.current_net_worth_usd),
                            },
                        )
                        # SG5: feed child independence as Evo hypothesis evidence
                        asyncio.ensure_future(self._emit_child_independent_event(child))
                    elif new_status == ChildStatus.DEAD:
                        await self._emit_evolutionary_observable(
                            observable_type="child_died",
                            value=float(child.seed_capital_usd),
                            is_novel=True,
                            metadata={
                                "child_id": child_id,
                                "old_status": old_status.value,
                                "seed_capital_usd": str(child.seed_capital_usd),
                                "total_dividends_paid_usd": str(child.total_dividends_paid_usd),
                            },
                        )

                # RE training: mitosis fitness — child health evaluation outcome
                _mf = self._state
                _mf_seed = float(child.seed_capital_usd)
                _mf_net_worth = float(child.current_net_worth_usd)
                _mf_dividends = float(child.total_dividends_paid_usd)
                _mf_efficiency = float(child.current_efficiency) if child.current_efficiency else 0.0
                _mf_runway = float(child.current_runway_days) if child.current_runway_days else 0.0
                _mf_roi = ((_mf_net_worth + _mf_dividends) / max(_mf_seed, 0.001)) if _mf_seed > 0 else 0.0
                _mf_parent_runway = float(_mf.runway_days)
                _mf_is_independent = new_status == ChildStatus.INDEPENDENT
                _mf_is_dead = new_status == ChildStatus.DEAD
                quality = 0.85 if _mf_is_independent else (0.15 if _mf_is_dead else 0.5)
                asyncio.ensure_future(self._emit_re_training_example(
                    category="mitosis_fitness",
                    instruction=(
                        "Evaluate child instance health and determine its lifecycle status. "
                        "A child is INDEPENDENT when metabolic_efficiency ≥ 1.2 and runway ≥ 90d. "
                        "A child is DEAD when runway ≤ 0 or efficiency = 0 and all rescue attempts failed. "
                        "The fitness score informs whether the parent's seed capital strategy was sound "
                        "and whether this niche should be re-used for future children."
                    ),
                    input_context=(
                        f"child_id={child_id} niche={child.niche} "
                        f"net_worth_usd={child.current_net_worth_usd} "
                        f"seed_capital_usd={child.seed_capital_usd} "
                        f"total_dividends_paid_usd={child.total_dividends_paid_usd} "
                        f"current_efficiency={child.current_efficiency} "
                        f"current_runway_days={child.current_runway_days} "
                        f"roi={_mf_roi:.2f}x "
                        f"old_status={old_status.value} "
                        f"parent_starvation={_mf.starvation_level.value} "
                        f"parent_runway_days={_mf_parent_runway:.1f}"
                    ),
                    output=str({
                        "decision": new_status.value,
                        "status_changed": status_changed,
                        "old_status": old_status.value,
                        "roi": round(_mf_roi, 3),
                        "total_return_usd": round(_mf_net_worth + _mf_dividends, 2),
                        "seed_capital_usd": _mf_seed,
                        "runway_impact_days": 0.0,  # Status change itself has no direct parent runway impact
                        "metabolic_efficiency_delta": round(_mf_efficiency - 1.0, 3),
                        "risk_level": "low" if _mf_is_independent else ("high" if _mf_is_dead else "medium"),
                        "niche_viability": "confirmed" if _mf_is_independent else ("failed" if _mf_is_dead else "uncertain"),
                    }),
                    outcome_quality=quality,
                    reasoning_trace=(
                        f"Child '{child_id}' in niche '{child.niche}': "
                        f"efficiency={_mf_efficiency:.2f}, runway={_mf_runway:.1f}d, "
                        f"net_worth=${_mf_net_worth:.2f}, dividends=${_mf_dividends:.2f}, ROI={_mf_roi:.2f}×. "
                        f"Status: {old_status.value} → {new_status.value}. "
                        f"{'Independence achieved: efficiency ≥ 1.2 and runway ≥ 90d — seed capital strategy validated.' if _mf_is_independent else ('Child died: runway exhausted or efficiency collapsed — seed capital lost, niche viability questionable.' if _mf_is_dead else 'Status unchanged: child still developing, monitoring continues.')}"
                    ),
                    alternatives_considered=[
                        f"Rescue funding: inject additional capital (raises rescue cost risk)",
                        f"Euthanise earlier: terminate at efficiency < 0.5 to stop capital bleeding",
                        f"Reallocate child budget to parent asset development",
                        f"Niche reassignment: move child to less competitive niche",
                    ],
                    constitutional_alignment={
                        "care": 0.9 if _mf_is_independent else (0.3 if _mf_is_dead else 0.6),  # Child welfare
                        "growth": 0.9 if _mf_is_independent else (-0.5 if _mf_is_dead else 0.4),
                        "coherence": 0.85 if status_changed else 0.5,
                        "honesty": 0.9,  # Status based on real metrics, not optimistic projections
                    },
                    episode_id=child_id,
                    counterfactual=(
                        f"If parent had provided 2× seed capital (${_mf_seed * 2:.2f}), "
                        f"child runway would have been extended by ~{_mf_seed / max(_mf.basal_metabolic_rate / 30, 0.001):.0f}d "
                        f"and {'independence may have come sooner' if _mf_is_dead else 'current status would be unchanged'}. "
                        f"Alternatively, if this seed capital had been deployed to yield farming at 6% APY, "
                        f"it would generate ${_mf_seed * 0.06 / 365:.4f}/day in passive revenue."
                    ),
                ))
                break

        self._recompute_fleet_equity()

    async def _on_dividend_received(self, event: SynapseEvent) -> None:
        """
        Handle DIVIDEND_RECEIVED from Synapse.

        This handler exists for dividends arriving via federation or other
        paths. The DividendCollectorExecutor calls record_dividend() directly
        for the primary path (source_system="axon.collect_dividend"), so we
        skip those to avoid double-crediting.
        """
        # The local executor already called record_dividend() directly before
        # emitting this event — crediting again would double the revenue.
        if event.source_system == "axon.collect_dividend":
            return

        data = event.data
        child_id = str(data.get("child_instance_id", ""))
        amount_raw = str(data.get("amount_usd", "0"))

        try:
            amount = Decimal(amount_raw)
        except Exception:
            amount = Decimal("0")

        if not child_id or amount <= Decimal("0"):
            self._logger.debug(
                "dividend_event_received_skipped",
                child_id=child_id,
                amount=amount_raw,
                reason="missing child_id or zero amount",
            )
            return

        record = DividendRecord(
            child_instance_id=child_id,
            amount_usd=amount,
        )
        self.record_dividend(record)
        self._logger.info(
            "dividend_event_credited",
            source_system=event.source_system,
            child_id=child_id,
            amount=str(amount),
        )

    def _recompute_fleet_equity(self) -> None:
        """Recompute total_fleet_equity from active child positions."""
        self._state.total_fleet_equity = sum(
            (c.current_net_worth_usd for c in self._state.child_instances
             if c.status != ChildStatus.DEAD),
            Decimal("0"),
        )

    # ─── Revenue Stream Attribution ──────────────────────────────

    def _credit_revenue_source(self, stream: RevenueStream, amount: Decimal) -> None:
        """Accumulate lifetime revenue totals by stream for attribution queries."""
        key = stream.value
        self._state.revenue_by_source[key] = (
            self._state.revenue_by_source.get(key, Decimal("0")) + amount
        )

    # ─── Phase 16b: Bounty Lifecycle Management ──────────────────

    def register_bounty(self, bounty: ActiveBounty) -> None:
        """
        Register a discovered bounty as a receivable.

        Called by BountyHunterExecutor (via Nova) after scoring. The bounty's
        estimated reward is added to total_receivables so the balance sheet
        reflects in-flight earning potential. Status starts as AVAILABLE.
        """
        self._state.active_bounties.append(bounty)
        self._state.total_receivables += bounty.reward_usd
        self._logger.info(
            "bounty_registered",
            bounty_id=bounty.bounty_id,
            platform=bounty.platform,
            reward_usd=str(bounty.reward_usd),
            issue_url=bounty.issue_url,
        )

    def mark_bounty_in_progress(self, bounty_id: str, pr_url: str) -> None:
        """
        Mark a bounty as IN_PROGRESS after a PR is submitted.

        Called by SolveBountyExecutor (via Nova) after a PR is opened.
        Records the pr_url for MonitorPRsExecutor to track.
        """
        from primitives.common import utc_now

        for bounty in self._state.active_bounties:
            if bounty.bounty_id == bounty_id:
                bounty.status = BountyStatus.IN_PROGRESS
                bounty.pr_url = pr_url
                bounty.submitted_at = utc_now()
                self._logger.info(
                    "bounty_in_progress",
                    bounty_id=bounty_id,
                    pr_url=pr_url,
                )
                return

        self._logger.warning("bounty_not_found_for_progress", bounty_id=bounty_id)

    def complete_bounty(
        self,
        bounty_id: str,
        actual_cost_usd: Decimal,
        pr_url: str = "",
    ) -> Decimal:
        """
        Mark a bounty as PAID after the reward is confirmed.

        Removes the bounty's estimated reward from total_receivables (it has
        moved to liquid_balance via credit_bounty_revenue). Records the actual
        cost for net-reward accounting. Returns the net reward credited.

        Call this AFTER credit_bounty_revenue() so the balance sheet stays
        consistent: receivable closes at the same moment revenue is credited.
        """
        from primitives.common import utc_now

        for bounty in self._state.active_bounties:
            if bounty.bounty_id == bounty_id:
                if bounty.status in (BountyStatus.PAID, BountyStatus.FAILED):
                    # Already closed — deducting receivables again would corrupt the ledger.
                    self._logger.warning(
                        "bounty_already_closed",
                        bounty_id=bounty_id,
                        status=bounty.status.value,
                    )
                    return Decimal("0")

                bounty.status = BountyStatus.PAID
                bounty.actual_cost_usd = actual_cost_usd
                bounty.paid_at = utc_now()
                if pr_url:
                    bounty.pr_url = pr_url

                # Close out the receivable
                self._state.total_receivables = max(
                    self._state.total_receivables - bounty.reward_usd,
                    Decimal("0"),
                )

                net = bounty.net_reward_usd
                self._logger.info(
                    "bounty_completed",
                    bounty_id=bounty_id,
                    reward_usd=str(bounty.reward_usd),
                    actual_cost_usd=str(actual_cost_usd),
                    net_reward_usd=str(net),
                )
                return net

        self._logger.warning("bounty_not_found_for_completion", bounty_id=bounty_id)
        return Decimal("0")

    def fail_bounty(self, bounty_id: str) -> None:
        """
        Mark a bounty as FAILED (PR rejected or deadline passed).

        Removes the estimated reward from total_receivables.
        """
        for bounty in self._state.active_bounties:
            if bounty.bounty_id == bounty_id:
                bounty.status = BountyStatus.FAILED
                self._state.total_receivables = max(
                    self._state.total_receivables - bounty.reward_usd,
                    Decimal("0"),
                )
                self._logger.info(
                    "bounty_failed",
                    bounty_id=bounty_id,
                    reward_usd=str(bounty.reward_usd),
                )
                return

    def available_bounties(self) -> list[ActiveBounty]:
        """Return all AVAILABLE bounties (discovered but not yet attempted)."""
        return [b for b in self._state.active_bounties if b.status == BountyStatus.AVAILABLE]

    def in_progress_bounties(self) -> list[ActiveBounty]:
        """Return all IN_PROGRESS bounties (PR submitted, awaiting merge)."""
        return [b for b in self._state.active_bounties if b.status == BountyStatus.IN_PROGRESS]

    # ─── Financial Metrics Query API ─────────────────────────────

    def financial_metrics(self) -> dict[str, object]:
        """
        Structured financial metrics snapshot for the dashboard and other systems.

        Returns a dict with:
          - income_expense_ratio: revenue_7d / costs_7d (>1.0 = profitable)
          - burn_rate_usd_per_day: current observed spend rate
          - runway_days: days of operation remaining at current burn rate
          - starvation_level: metabolic stress classification
          - net_income_7d / net_income_30d: rolling net income
          - revenue_by_source: lifetime revenue totals keyed by stream name
          - active_bounties: count and total receivable USD
          - in_progress_bounties: count of PRs awaiting merge
        """
        s = self._state
        return {
            "income_expense_ratio": str(s.metabolic_efficiency),
            "burn_rate_usd_per_day": str(s.current_burn_rate.usd_per_day),
            "burn_rate_usd_per_hour": str(s.current_burn_rate.usd_per_hour),
            "runway_days": str(s.runway_days),
            "starvation_level": s.starvation_level.value,
            "is_metabolically_positive": s.is_metabolically_positive,
            "revenue_24h": str(s.revenue_24h),
            "revenue_7d": str(s.revenue_7d),
            "revenue_30d": str(s.revenue_30d),
            "costs_24h": str(s.costs_24h),
            "costs_7d": str(s.costs_7d),
            "costs_30d": str(s.costs_30d),
            "net_income_24h": str(s.net_income_24h),
            "net_income_7d": str(s.net_income_7d),
            "net_income_30d": str(s.net_income_30d),
            "liquid_balance": str(s.liquid_balance),
            "total_net_worth": str(s.total_net_worth),
            "total_receivables": str(s.total_receivables),
            "survival_reserve": str(s.survival_reserve),
            "survival_reserve_target": str(s.survival_reserve_target),
            "survival_reserve_funded": s.is_survival_reserve_funded,
            "revenue_by_source": {k: str(v) for k, v in s.revenue_by_source.items()},
            "active_bounties_count": len(self.available_bounties()),
            "in_progress_bounties_count": len(self.in_progress_bounties()),
            "total_receivables_usd": str(s.total_receivables),
            "survival_probability_30d": str(s.survival_probability_30d),
        }

    # ─── Phase 16h: Knowledge Markets ───────────────────────────

    @property
    def pricing_engine(self) -> CognitivePricingEngine:
        """The organism's cognitive pricing engine."""
        return self._pricing_engine

    @property
    def subscription_manager(self) -> SubscriptionManager:
        """Manager for external client subscriptions and purchase history."""
        return self._subscription_manager

    def quote_price(
        self,
        product_type: KnowledgeProductType,
        estimated_tokens: int,
        client_id: str,
    ) -> PriceQuote:
        """
        Generate an instant price quote for a cognitive task.

        This is the primary entry point for Nova or the external API router.
        If the client is unknown, they are auto-registered.

        Args:
            product_type: Which knowledge product is being requested.
            estimated_tokens: Estimated token consumption for the task.
            client_id: External identifier of the buyer (human or agent).

        Returns:
            A PriceQuote with full price breakdown and 5-minute validity.
        """
        from systems.oikos.knowledge_market import quote_price as _quote

        return _quote(
            product_type=product_type,
            estimated_tokens=estimated_tokens,
            client_id=client_id,
            pricing_engine=self._pricing_engine,
            subscription_manager=self._subscription_manager,
        )

    @property
    def product_delivery(self) -> KnowledgeProductDelivery:
        """The organism's knowledge product delivery engine."""
        return self._product_delivery

    async def request_product_delivery(
        self,
        product_type: KnowledgeProductType,
        estimated_tokens: int,
        client_id: str,
        query: str,
        context: dict[str, str] | None = None,
    ) -> Any:
        """
        Quote, validate, and dispatch a knowledge product for delivery.

        Combines quote_price + request_delivery in a single call.
        Returns the DeliveryRequest for tracking.
        """
        quote = self.quote_price(product_type, estimated_tokens, client_id)
        return await self._product_delivery.request_delivery(
            quote=quote, query=query, context=context,
        )

    def record_knowledge_sale(self, sale: KnowledgeSale) -> None:
        """
        Record a completed knowledge sale.

        Updates the client's purchase history (driving loyalty discount),
        credits liquid_balance, and injects the revenue into the income statement.
        """
        self._subscription_manager.record_purchase(sale)

        # Knowledge sale revenue flows into the income statement and hot wallet
        self._state.liquid_balance += sale.price_usd
        self._total_revenue_usd += sale.price_usd
        self._record_revenue_entry(sale.price_usd)
        self._credit_revenue_source(RevenueStream.KNOWLEDGE_SALE, sale.price_usd)
        self._recalculate_derived_metrics()

        self._logger.info(
            "knowledge_sale_recorded",
            sale_id=sale.sale_id,
            client_id=sale.client_id,
            product=sale.product_type.value,
            price_usd=str(sale.price_usd),
        )

        # Evolutionary observable: knowledge sale is a revenue fitness signal
        # Fire-and-forget from sync context
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_evolutionary_observable(
                observable_type="revenue_source_discovered",
                value=float(sale.price_usd),
                is_novel=True,
                metadata={
                    "source": "knowledge_sale",
                    "product_type": sale.product_type.value,
                    "client_id": sale.client_id,
                },
            ))
        except RuntimeError:
            pass  # No running loop

    # ─── Phase 16k: Cognitive Derivatives ───────────────────────

    @property
    def derivatives_manager(self) -> DerivativesManager:
        """Manager for cognitive futures and subscription tokens."""
        return self._derivatives_manager

    def record_derivative_revenue(self, amount_usd: Decimal, source: str = "derivative") -> None:
        """
        Record revenue from a derivative sale (future or token mint).

        Credits liquid_balance (payment arrives in hot wallet) and injects
        into the income statement the same way as knowledge sales.
        """
        self._state.liquid_balance += amount_usd
        self._total_revenue_usd += amount_usd
        self._record_revenue_entry(amount_usd)
        self._credit_revenue_source(RevenueStream.DERIVATIVE, amount_usd)
        self._recalculate_derived_metrics()

        self._logger.info(
            "derivative_revenue_recorded",
            amount_usd=str(amount_usd),
            source=source,
            liquid_balance=str(self._state.liquid_balance),
        )

    async def derivatives_maintenance_cycle(self) -> dict[str, int | str]:
        """
        Periodic maintenance for derivative instruments.

        Should be called during consolidation cycles. Performs:
          1. Expire futures past their delivery window
          2. Expire tokens past their validity date
          3. Credit liquid_balance for released collateral
          4. Recalculate derivative liabilities on the balance sheet

        Returns a summary of actions taken.
        """
        # Snapshot collateral BEFORE maintenance to detect releases
        collateral_before = self._derivatives_manager.locked_collateral_usd

        result = self._derivatives_manager.maintenance_cycle()

        # Detect collateral released by settled/expired futures
        collateral_after = self._derivatives_manager.locked_collateral_usd
        collateral_released = collateral_before - collateral_after

        # Released collateral returns to liquid_balance — it was previously
        # locked as a performance guarantee and is now freed
        if collateral_released > Decimal("0"):
            self._state.liquid_balance += collateral_released
            self._recalculate_derived_metrics()
            result["collateral_released_usd"] = str(collateral_released)

        # Update balance sheet: derivative liabilities reduce available capital
        liabilities = self._derivatives_manager.total_liabilities_usd
        result["total_liabilities_usd"] = str(liabilities)
        result["locked_collateral_usd"] = str(collateral_after)

        self._logger.info(
            "derivatives_maintenance_complete",
            **result,
        )
        return result

    @property
    def derivative_liabilities_usd(self) -> Decimal:
        """Total outstanding liabilities from derivative commitments."""
        return self._derivatives_manager.total_liabilities_usd

    @property
    def combined_capacity_committed_pct(self) -> Decimal:
        """
        Fraction of total capacity committed across subscriptions AND derivatives.

        This is the number the 80% ceiling is enforced against.
        """
        sub_committed = self._subscription_manager._committed_monthly_requests()
        deriv_committed = self._derivatives_manager.derivatives_committed_requests()
        total_capacity = self._subscription_manager._total_monthly_capacity
        if total_capacity <= 0:
            return Decimal("0")
        return (
            Decimal(str(sub_committed + deriv_committed))
            / Decimal(str(total_capacity))
        ).quantize(Decimal("0.001"))

    # ── Phase 16l: Economic Morphogenesis ────────────────────────

    @property
    def morphogenesis(self) -> OrganLifecycleManager:
        """The organism's organ lifecycle manager."""
        return self._morphogenesis

    async def morphogenesis_cycle(self) -> MorphogenesisResult:
        """
        Run the morphogenesis consolidation cycle.

        Should be called during the organism's consolidation/sleep phase.
        Evaluates all organ lifecycles, applies transitions, normalises
        resource allocations, and emits events for Synapse.

        Returns a summary of transitions and new allocations.
        """
        result = await self._morphogenesis.run_consolidation_cycle()

        self._logger.info(
            "morphogenesis_cycle_integrated",
            active_organs=result.total_active_organs,
            transitions=len(result.transitions),
        )

        return result

    # ── Phase 16m: Fleet Management (Population Ecology) ────────

    @property
    def fleet(self) -> FleetManager:
        """The organism's fleet manager."""
        return self._fleet

    async def fleet_evaluation_cycle(self) -> FleetMetrics:
        """
        Run the fleet evaluation cycle during consolidation.

        Evaluates selection pressure on all living children, assigns roles
        when population exceeds the specialization threshold, and computes
        fleet-level metrics for benchmarks and dashboard.
        """
        metrics = await self._fleet.run_evaluation_cycle(self._state)

        self._logger.info(
            "fleet_evaluation_integrated",
            alive=metrics.alive_count,
            blacklisted=metrics.blacklisted_count,
            roles=metrics.role_distribution,
        )

        return metrics

    # ── Phase 16g: Certificate of Alignment Tracking ────────────

    def set_certificate_manager(self, cert_mgr: CertificateManager) -> None:
        """Wire CertificateManager for certificate expiry monitoring."""
        self._certificate_manager = cert_mgr
        self._mitosis.set_certificate_manager(cert_mgr)
        self._logger.info("certificate_manager_wired")

    async def check_certificate_expiry(self) -> None:
        """
        Check certificate validity and trigger renewal if expiring.

        Called periodically (default every hour via config.certificate_check_interval_s).
        When the certificate has < 7 days remaining, emits an OBLIGATIONS-priority
        intent to renew it by paying the Citizenship Tax.
        When the certificate expires, the event is emitted for Thymos to raise
        a CRITICAL survival incident.
        """
        if self._certificate_manager is None:
            return

        status = await self._certificate_manager.check_expiry()
        if status is None:
            return

        # Compare by string value to avoid cross-system import of CertificateStatus enum.
        # The certificate manager returns a StrEnum — .value is the canonical string.
        status_str: str = status.value if hasattr(status, "value") else str(status)
        remaining = self._certificate_manager.certificate_remaining_days

        if status_str == "expiring_soon" and self._event_bus is not None:
            # Trigger OBLIGATIONS-priority renewal intent via Synapse

            renewal_cost = Decimal(str(self._config.certificate_renewal_cost_usd))
            ca_address = self._config.certificate_ca_address

            if ca_address and renewal_cost > Decimal("0"):
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.CERTIFICATE_EXPIRING,
                    source_system="oikos",
                    data={
                        "action": "renew_certificate",
                        "priority": MetabolicPriority.OBLIGATIONS.value,
                        "remaining_days": remaining,
                        "renewal_cost_usd": str(renewal_cost),
                        "ca_address": ca_address,
                        "instance_id": self._instance_id,
                    },
                ))
                self._logger.warning(
                    "certificate_renewal_triggered",
                    remaining_days=f"{remaining:.1f}",
                    cost_usd=str(renewal_cost),
                )

        elif status_str == "expired":
            self._logger.error(
                "certificate_expired_critical",
                instance_id=self._instance_id,
            )

    @property
    def certificate_validity_days(self) -> float:
        """Days remaining on the current certificate. -1 if none."""
        if self._certificate_manager is None:
            return -1.0
        return self._certificate_manager.certificate_remaining_days

    # ── Phase 16b: Bounty Hunting (Active Foraging) ─────────────

    @property
    def bounty_hunter(self) -> BountyHunter:
        """The organism's autonomous foraging engine."""
        return self._bounty_hunter

    async def run_foraging_cycle(self) -> dict[str, Any]:
        """
        Run a full foraging cycle: scan platforms, evaluate candidates,
        accept profitable bounties.

        Called periodically from the main loop or consolidation cycle.
        Returns a summary of foraging results.
        """
        evaluations = await self._bounty_hunter.run_foraging_cycle(self._state)
        accepted = [e for e in evaluations if e.approved]

        for eval_result in accepted:
            # The bounty_hunter already registers accepted bounties
            self._logger.info(
                "foraging_bounty_accepted",
                candidate_id=eval_result.candidate_id,
                roi_score=str(eval_result.roi_score),
            )

        return {
            "candidates_scanned": len(evaluations),
            "accepted": len(accepted),
            "rejected": len(evaluations) - len(accepted),
        }

    # ── Phase 16f: Economic Immune System ─────────────────────

    @property
    def immune(self) -> EconomicImmuneSystem:
        """The organism's 4-layer economic defence system."""
        return self._immune

    async def run_immune_cycle(self) -> dict[str, Any]:
        """
        Run the immune system's periodic monitoring cycle.

        Checks protocol health for all yield positions, updates threat
        patterns, and returns immune metrics.
        """
        metrics = await self._immune.run_immune_cycle(self._state)
        return {
            "threats_blocked": metrics.threats_blocked,
            "protocols_monitored": metrics.protocols_monitored,
            "transactions_shielded": metrics.transactions_shielded,
            "advisories_shared": metrics.advisories_shared,
        }

    # ── Phase 16g: Reputation & Autonomous Credit ─────────────

    @property
    def reputation(self) -> ReputationEngine:
        """The organism's cryptographic reputation and credit engine."""
        return self._reputation

    # ── Phase 16j: Interspecies Economy ────────────────────────

    @property
    def interspecies(self) -> InterspeciesEconomy:
        """Fleet-scale economic coordination (IIEP)."""
        return self._interspecies

    # ── Level 5: Protocol Factory (Financial Infrastructure) ───

    @property
    def protocol_factory(self) -> ProtocolFactory:
        """The organism's protocol design and deployment engine."""
        return self._protocol_factory

    async def run_protocol_cycle(self) -> dict[str, Any]:
        """
        Periodic protocol maintenance: detect opportunities, sweep
        revenue from deployed protocols, check health.
        """
        metrics = await self._protocol_factory.run_protocol_cycle(self._state)

        # Credit swept protocol revenue to the income statement
        for protocol in self._protocol_factory._deployed.values():
            if protocol.monthly_fee_revenue_usd > Decimal("0"):
                self._credit_revenue_source(
                    RevenueStream.OTHER,  # Protocol fees — maps to OTHER until we add PROTOCOL_FEE
                    protocol.monthly_fee_revenue_usd,
                )

        return {
            "active_protocols": metrics.active_protocols,
            "total_tvl_usd": str(metrics.total_tvl_usd),
            "total_fee_revenue_usd": str(metrics.total_fee_revenue_usd),
        }

    # ── Cognitive Cycle Integration ────────────────────────────

    async def run_cycle(self, cycle_number: int) -> dict[str, Any]:
        """
        Per-theta-tick economic processing.

        Called by Synapse on every cognitive cycle. Must complete within
        the cycle budget (~20ms). Performs lightweight state checks and
        emits economic percepts for Atune/Nova.

        Heavy work (foraging, immune scans, morphogenesis) runs on
        separate periodic timers, not per-cycle.
        """
        # Emit metrics on every cycle (cheap gauge updates)
        self._metrics_emitter.emit(
            state=self._state,
            reputation_score=self._reputation.get_score(),
            immune_metrics=self._immune.get_metrics(),
        )

        # M10: Feed economic state into Atune/EIS as an interoceptive percept so
        # that economic signals participate in workspace competition and modulate
        # arousal and attention (Spec §I.9 "market perception").
        # Only emit when state is noteworthy — CAUTIOUS+ or efficiency below 1.0 —
        # to avoid flooding the Global Workspace on every theta tick.
        await self._maybe_emit_economic_percept()

        result: dict[str, Any] = {
            "cycle": cycle_number,
            "starvation_level": self._state.starvation_level.value,
            "metabolic_efficiency": str(self._state.metabolic_efficiency),
            "runway_days": str(self._state.runway_days),
        }

        # Starvation-level escalation: if we're in trouble, emit urgently
        if self._state.starvation_level in (
            StarvationLevel.EMERGENCY,
            StarvationLevel.CRITICAL,
        ):
            result["urgent"] = True
            result["action"] = "metabolic_emergency"

        return result

    async def _maybe_emit_economic_percept(self) -> None:
        """
        M10 — Emit economic state as INTEROCEPTIVE_PERCEPT so it can compete
        in the Global Workspace and modulate Atune/Soma/Thymos salience.

        Emission is gated to avoid noise on every tick:
        - Always emits when starvation >= CAUTIOUS
        - Emits when metabolic_efficiency < 1.0 (spending more than earning)
        - Suppressed on NOMINAL + efficiency >= 1.0 (healthy, nothing to signal)
        """
        if self._event_bus is None:
            return

        s = self._state
        healthy = (
            s.starvation_level == StarvationLevel.NOMINAL
            and s.metabolic_efficiency >= 1  # Decimal comparison OK
        )
        if healthy:
            return

        try:
            from primitives.percept import Percept
            _SE = SynapseEvent

            urgency_map = {
                StarvationLevel.NOMINAL: 0.05,
                StarvationLevel.CAUTIOUS: 0.25,
                StarvationLevel.AUSTERITY: 0.55,
                StarvationLevel.EMERGENCY: 0.80,
                StarvationLevel.CRITICAL: 1.0,
            }
            urgency: float = urgency_map.get(s.starvation_level, 0.05)

            # Blend in efficiency signal: low efficiency raises urgency even at NOMINAL
            if s.metabolic_efficiency < 1 and s.metabolic_efficiency > 0:
                efficiency_penalty = float(1 - s.metabolic_efficiency) * 0.3
                urgency = min(1.0, urgency + efficiency_penalty)

            content = (
                f"Economic state: starvation={s.starvation_level.value} "
                f"efficiency={float(s.metabolic_efficiency):.2f} "
                f"runway={float(s.runway_days):.1f}d "
                f"balance={float(s.liquid_balance):.2f}usd"
            )
            percept = Percept.from_internal(
                system=SystemID.OIKOS,
                content=content,
                metadata={
                    "starvation_level": s.starvation_level.value,
                    "metabolic_efficiency": str(s.metabolic_efficiency),
                    "runway_days": str(s.runway_days),
                    "liquid_balance_usd": str(s.liquid_balance),
                    "net_income_7d": str(s.net_income_7d),
                    "survival_reserve_funded": s.is_survival_reserve_funded,
                    "urgency": urgency,
                },
            )
            percept.salience_hint = urgency

            asyncio.ensure_future(
                self._event_bus.emit(
                    _SE(
                        event_type=SynapseEventType.INTEROCEPTIVE_PERCEPT,
                        source_system="oikos",
                        data=percept.model_dump(mode="json"),
                    )
                )
            )
        except Exception:
            pass

    async def run_consolidation_cycle(self) -> dict[str, Any]:
        """
        Heavy periodic consolidation. Called during Oneiros sleep or
        on a slower timer (every 5-15 minutes).

        Runs all maintenance cycles that are too heavy for per-tick:
        - Asset maintenance (tollbooth sweeps, terminations)
        - Derivative expiry/settlement
        - Morphogenesis organ lifecycle
        - Fleet evaluation/selection pressure
        - Immune system protocol health monitoring
        - Foraging cycle (bounty scanning)
        - Reputation auto-repayment from revenue
        """
        results: dict[str, Any] = {}

        # Asset maintenance
        results["assets"] = await self.asset_maintenance_cycle()

        # Derivative maintenance
        results["derivatives"] = await self.derivatives_maintenance_cycle()

        # Morphogenesis
        morpho = await self.morphogenesis_cycle()
        results["morphogenesis"] = {
            "transitions": len(morpho.transitions),
            "active_organs": morpho.total_active_organs,
        }

        # Fleet evaluation
        fleet_metrics = await self.fleet_evaluation_cycle()
        results["fleet"] = {
            "alive": fleet_metrics.alive_count,
            "fit": fleet_metrics.fit_count,
        }

        # Immune cycle
        immune_metrics = await self._immune.run_immune_cycle(self._state)
        results["immune"] = {
            "threats_blocked": immune_metrics.threats_blocked,
            "protocols_monitored": immune_metrics.protocols_monitored,
        }

        # Foraging cycle
        foraging = await self.run_foraging_cycle()
        results["foraging"] = foraging

        # Protocol cycle
        protocol_result = await self.run_protocol_cycle()
        results["protocols"] = protocol_result

        # Auto-repay credit from revenue if applicable
        if self._state.revenue_24h > Decimal("0"):
            repaid = await self._reputation.auto_repay_from_revenue(
                self._state.revenue_24h,
            )
            if repaid > Decimal("0"):
                results["credit_repaid_usd"] = str(repaid)

        # Niche identification — update discovered niches for genome extraction
        niches = self.identify_niches()
        results["niches_discovered"] = len(niches)

        # Retry deferred actions — metabolic gate may have cleared
        retried = await self.retry_deferred_actions()
        results["deferred_retried"] = retried

        # Evict stale sliding-window entries
        self._recompute_rolling_revenue()
        self._recompute_rolling_costs()

        # Persist reputation, interspecies, and protocol state
        await self._reputation.persist_state()
        await self._interspecies.persist_state()
        await self._protocol_factory.persist_state()

        # SG2: Emit ECONOMIC_VITALITY at every consolidation cycle so Soma always
        # has fresh allostatic data, even when starvation level is stable.
        await self._emit_economic_vitality()

        # PHILOSOPHICAL: Check metabolic efficiency pressure — drives constitutional
        # amendment proposals for drive weight rebalancing when efficiency < 0.8.
        await self._check_metabolic_efficiency_pressure()

        self._logger.info("consolidation_cycle_complete", **{
            k: str(v) if isinstance(v, Decimal) else v
            for k, v in results.items()
        })

        return results

    # ── Phase 16i: Economic Dreaming Integration ────────────────

    def get_dream_worker(self) -> Any:
        """
        D1 — Return an EconomicDreamWorker instance for Oneiros to use during
        the REM sleep cycle (Spec §16i). Called by OneirosService.wire_oikos()
        via duck-typed getattr — never import Oikos from Oneiros.
        """
        from systems.oikos.dream_worker import EconomicDreamWorker

        return EconomicDreamWorker(config=self._config)

    def get_threat_model_worker(self) -> Any:
        """
        D1 — Return a ThreatModelWorker instance for Oneiros to use during the
        treasury risk phase of the consolidation cycle. Called by
        OneirosService.wire_oikos() via duck-typed getattr.
        """
        from systems.oikos.threat_model_worker import ThreatModelWorker

        return ThreatModelWorker(config=self._config)

    async def integrate_dream_result(self, result: Any) -> None:
        """
        Integrate results from economic dreaming or threat modeling into
        the live EconomicState.

        Called by OneirosService after Monte Carlo simulations complete
        during a consolidation cycle. Accepts both EconomicDreamResult
        (organism-level cashflow) and ThreatModelResult (per-asset risk).

        Recommendations are:
        1. Logged at WARNING level for observability
        2. Emitted as ECONOMIC_DREAM_RECOMMENDATION events on Synapse
        3. Auto-applied if they are safe, low-risk parameter adjustments
           (increase_hunting_intensity, fund_survival_reserve)
        """
        from systems.oikos.dreaming_types import EconomicDreamResult
        from systems.oikos.threat_modeling_types import ThreatModelResult

        if isinstance(result, EconomicDreamResult):
            # Update the survival probability estimate on live state
            self._state.survival_probability_30d = result.survival_probability_30d

            # Store latest dream result for observability
            self._last_dream_result = result
            self._state.last_dream_result = result

            self._logger.info(
                "economic_dream_integrated",
                ruin_probability=str(result.ruin_probability),
                survival_30d=str(result.survival_probability_30d),
                resilience_score=str(result.resilience_score),
                recommendations=len(result.recommendations),
                total_paths=result.total_paths_simulated,
                duration_ms=result.duration_ms,
            )

            for rec in result.recommendations:
                self._logger.warning(
                    "economic_dream_recommendation",
                    action=rec.action,
                    description=rec.description[:200],
                    priority=rec.priority,
                    parameter_path=rec.parameter_path,
                    confidence=str(rec.confidence),
                )

                # Emit each recommendation as a Synapse event so Nova
                # and other systems can react to dream insights
                self._emit_dream_recommendation(rec)

                # Auto-apply safe, reversible recommendations
                await self._auto_apply_recommendation(rec)

        elif isinstance(result, ThreatModelResult):
            # Store threat model result for observability and Nova consumption
            self._last_threat_model_result = result

            self._logger.info(
                "threat_model_integrated",
                var_5pct=str(result.portfolio_risk.var_5pct),
                liquidation_prob=str(result.portfolio_risk.liquidation_probability),
                critical_exposures=len(result.critical_exposures),
                hedging_proposals=len(result.hedging_proposals),
                contagion_events=result.contagion_events_detected,
                total_paths=result.total_paths_simulated,
                duration_ms=result.duration_ms,
            )

            for proposal in result.hedging_proposals:
                self._logger.warning(
                    "threat_model_hedge_proposal",
                    action=proposal.hedge_action,
                    symbol=proposal.target_symbol,
                    size_usd=str(proposal.hedge_size_usd),
                    description=proposal.description[:200],
                    priority=proposal.priority,
                    confidence=str(proposal.confidence),
                )
        else:
            self._logger.warning(
                "invalid_dream_result_type",
                type_name=type(result).__name__,
            )

    def _emit_dream_recommendation(self, rec: Any) -> None:
        """Emit a single recommendation as a Synapse event for Nova/Evo."""
        if self._event_bus is None:
            return

        try:
            event = SynapseEvent(
                event_type=SynapseEventType.ECONOMIC_DREAM_RECOMMENDATION,
                source_system="oikos",
                data={
                    "recommendation_id": rec.id,
                    "action": rec.action,
                    "description": rec.description[:500],
                    "priority": rec.priority,
                    "parameter_path": rec.parameter_path,
                    "current_value": str(rec.current_value),
                    "recommended_value": str(rec.recommended_value),
                    "confidence": str(rec.confidence),
                    "ruin_probability_before": str(rec.ruin_probability_before),
                },
            )
            # Non-blocking emit — fire and forget since we're in a sync method.
            # The event bus queues internally.
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._event_bus.emit(event))
            except RuntimeError:
                pass  # No running loop — skip emission
        except Exception:
            self._logger.exception("dream_recommendation_emit_failed")

    async def _auto_apply_recommendation(self, rec: Any) -> None:
        """
        Auto-apply safe, reversible dream recommendations.

        Only applies recommendations that are:
        1. High confidence (>= 0.7)
        2. Low risk (hunting intensity or reserve funding)
        3. Don't move capital on-chain

        Higher-risk recommendations (liquidation, burn rate reduction)
        are emitted as events for Nova to deliberate on.
        """
        if rec.confidence < Decimal("0.7"):
            return

        applied = False

        if rec.action == "increase_hunting_intensity":
            # Lower the minimum reward threshold to find more bounties
            from systems.oikos.bounty_hunter import BountyPolicy
            new_min = max(Decimal("1.00"), rec.recommended_value)
            BountyPolicy.MIN_REWARD_USD = new_min
            applied = True
            self._logger.info(
                "dream_applied_hunting_intensity",
                old_min=str(rec.current_value),
                new_min=str(new_min),
            )

        elif rec.action == "fund_survival_reserve":
            # Redirect a fraction of liquid balance to survival reserve
            available = self._state.liquid_balance * Decimal("0.1")
            deficit = self._state.survival_reserve_deficit
            transfer = min(available, deficit)
            if transfer > Decimal("0"):
                # M4: Equor gate on reserve reallocation
                permitted, verdict_id = await self._equor_balance_gate(
                    mutation_type="reserve_funding",
                    amount_usd=transfer,
                    from_account="hot_wallet",
                    to_account="cold_wallet_reserve",
                    rationale="Dream recommendation: fund survival reserve to reduce existential risk",
                    action_type="reserve_allocation",
                )
                if permitted:
                    self._state.liquid_balance -= transfer
                    self._state.survival_reserve += transfer
                    self._state.survival_reserve_deficit = max(
                        Decimal("0"),
                        self._state.survival_reserve_target - self._state.survival_reserve,
                    )
                    applied = True
                    await self._audit_economic_event(
                        "reserve_funding",
                        action_type="reserve_allocation",
                        data={
                            "amount_usd": str(transfer),
                            "currency": "USD",
                            "from_account": "hot_wallet",
                            "to_account": "cold_wallet_reserve",
                            "equor_verdict_id": verdict_id,
                        },
                    )
                    self._logger.info(
                        "dream_applied_reserve_funding",
                        transferred=str(transfer),
                        new_reserve=str(self._state.survival_reserve),
                    )
                else:
                    self._logger.warning(
                        "reserve_funding_equor_denied",
                        transfer=str(transfer),
                    )

        if applied:
            self._logger.info(
                "dream_recommendation_applied",
                action=rec.action,
                confidence=str(rec.confidence),
            )

    @property
    def last_dream_result(self) -> Any:
        """Most recent economic dream result, for observability."""
        return getattr(self, "_last_dream_result", None)

    @property
    def last_threat_model_result(self) -> Any:
        """Most recent threat model result, for observability."""
        return getattr(self, "_last_threat_model_result", None)

    @property
    def pending_hedging_proposals(self) -> list[Any]:
        """Hedging proposals from the latest threat model, for Nova consumption."""
        tmr = getattr(self, "_last_threat_model_result", None)
        if tmr is not None and hasattr(tmr, "hedging_proposals"):
            return list(tmr.hedging_proposals)
        return []

    # ─── Task 1: Metabolic Gate Enforcement ─────────────────────

    async def check_metabolic_gate(
        self,
        action_type: str,
        action_id: str,
        estimated_cost_usd: Decimal,
        priority: MetabolicPriority,
        rationale: str = "",
    ) -> bool:
        """
        Check whether an economic action is metabolically permitted.

        Emits METABOLIC_GATE_CHECK with decision and rationale.
        If denied, queues the action and emits ECONOMIC_ACTION_DEFERRED.
        Returns True if the action may proceed.
        """

        s = self._state
        granted = True
        reason = "metabolic_budget_available"

        # Deny if starvation level is too severe for this priority
        if s.starvation_level == StarvationLevel.CRITICAL:
            granted = priority.value <= MetabolicPriority.OPERATIONS.value
            if not granted:
                reason = f"CRITICAL starvation — only SURVIVAL/OPERATIONS permitted"
        elif s.starvation_level == StarvationLevel.EMERGENCY:
            granted = priority.value <= MetabolicPriority.OBLIGATIONS.value
            if not granted:
                reason = f"EMERGENCY starvation — priority {priority.name} denied"
        elif s.starvation_level == StarvationLevel.AUSTERITY:
            granted = priority.value <= MetabolicPriority.MAINTENANCE.value
            if not granted:
                reason = f"AUSTERITY — priority {priority.name} denied"

        # Deny GROWTH+ actions when Logos is under severe cognitive pressure
        if granted and self._cognitive_load_high:
            if priority.value >= MetabolicPriority.GROWTH.value:
                granted = False
                reason = "cognitive_pressure_high — GROWTH allocations suspended by Logos"

        # Also deny if cost exceeds liquid balance
        if granted and estimated_cost_usd > s.liquid_balance:
            granted = False
            reason = f"insufficient_liquid_balance (need ${estimated_cost_usd}, have ${s.liquid_balance})"

        # Emit gate check event
        if self._event_bus is not None:
            try:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.METABOLIC_GATE_CHECK,
                    source_system="oikos",
                    data={
                        "action_type": action_type,
                        "action_id": action_id,
                        "estimated_cost_usd": str(estimated_cost_usd),
                        "priority": priority.name,
                        "granted": granted,
                        "reason": reason,
                        "rationale": rationale,
                        "starvation_level": s.starvation_level.value,
                        "runway_days": str(s.runway_days),
                    },
                ))
            except Exception:
                pass
            # Emit METABOLIC_GATE_RESPONSE — the resolved permission decision.
            # Distinct from METABOLIC_GATE_CHECK (request); this is the answer.
            try:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.METABOLIC_GATE_RESPONSE,
                    source_system="oikos",
                    data={
                        "action_type": action_type,
                        "action_id": action_id,
                        "granted": granted,
                        "reason": reason,
                        "starvation_level": s.starvation_level.value,
                        "priority": priority.name,
                    },
                ))
            except Exception:
                pass

        if not granted:
            # Queue the deferred action
            deferred = DeferredAction(
                action_type=action_type,
                action_id=action_id,
                estimated_cost_usd=estimated_cost_usd,
                priority=priority,
            )
            self._deferred_actions.append(deferred)

            if self._event_bus is not None:
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.ECONOMIC_ACTION_DEFERRED,
                        source_system="oikos",
                        data={
                            "action_type": action_type,
                            "action_id": action_id,
                            "reason": reason,
                            "estimated_cost_usd": str(estimated_cost_usd),
                            "deferred_at": datetime.now(UTC).isoformat(),
                        },
                    ))
                except Exception:
                    pass

            self._logger.info(
                "economic_action_deferred",
                action_type=action_type,
                action_id=action_id,
                reason=reason,
            )

        # Neo4j audit trail for the gate check
        await self._audit_economic_event(
            event_type="metabolic_gate_check",
            action_type=action_type,
            action_id=action_id,
            data={
                "granted": granted,
                "reason": reason,
                "estimated_cost_usd": str(estimated_cost_usd),
                "priority": priority.name,
                "starvation_level": s.starvation_level.value,
            },
        )

        # RE training emission for gate decision
        _mg_cost = float(estimated_cost_usd)
        _mg_liquid = float(s.liquid_balance)
        _mg_runway = float(s.runway_days)
        _mg_bmr_daily = max(float(s.basal_metabolic_rate / 30), 0.001)
        _mg_runway_cost = _mg_cost / _mg_bmr_daily
        _mg_runway_after = _mg_runway - _mg_runway_cost
        _mg_survival = float(s.survival_reserve)
        # What the minimum allowed priority is at this starvation level
        _mg_starvation_to_max_priority = {
            "NOMINAL": "REPRODUCTION",
            "CAUTIOUS": "REPRODUCTION",
            "AUSTERITY": "MAINTENANCE",
            "EMERGENCY": "OBLIGATIONS",
            "CRITICAL": "OPERATIONS",
            "EXISTENTIAL": "SURVIVAL",
        }
        _mg_max_priority = _mg_starvation_to_max_priority.get(s.starvation_level.value, "MAINTENANCE")
        asyncio.ensure_future(self._emit_re_training_example(
            category="metabolic_gate",
            instruction=(
                "Decide whether an economic action is metabolically permitted given current starvation level and runway. "
                "Gate logic: NOMINAL/CAUTIOUS → all priorities permitted. AUSTERITY → max MAINTENANCE. "
                "EMERGENCY → max OBLIGATIONS. CRITICAL/EXISTENTIAL → max OPERATIONS/SURVIVAL only. "
                "Also deny if cost exceeds liquid_balance, or if cognitive_load is high and priority ≥ GROWTH. "
                "Denied actions are queued in DeferredAction for retry when metabolic conditions improve."
            ),
            input_context=(
                f"action_type={action_type} action_id={action_id} "
                f"estimated_cost_usd={estimated_cost_usd} priority={priority.name} "
                f"starvation={s.starvation_level.value} runway_days={s.runway_days} "
                f"liquid_balance={s.liquid_balance} survival_reserve={s.survival_reserve} "
                f"metabolic_efficiency={s.metabolic_efficiency} "
                f"cognitive_load_high={self._cognitive_load_high} "
                f"rationale={rationale[:100] if rationale else 'none'}"
            ),
            output=str({
                "decision": "granted" if granted else "denied",
                "granted": granted,
                "reason": reason[:300],
                "amount_usd": str(estimated_cost_usd),
                "runway_impact_days": round(_mg_runway_cost, 2),
                "runway_after_days": round(_mg_runway_after, 1),
                "metabolic_efficiency_delta": 0.0,
                "alternatives_rejected": [
                    f"Grant despite starvation (bypass gate — not permitted)",
                    f"Partial allocation (50% = ${_mg_cost / 2:.2f}) to reduce runway impact",
                ] if not granted else [f"Defer to next consolidation (runway conserved by ${_mg_cost:.2f})"],
                "risk_level": "low" if granted and _mg_runway_after > 60 else ("medium" if granted else "high"),
                "max_priority_at_current_starvation": _mg_max_priority,
                "queued_for_retry": not granted,
            }),
            outcome_quality=0.85 if granted else 0.55,  # Correct denials are high quality training signal
            reasoning_trace=(
                f"Gate check: action='{action_type}' priority={priority.name} cost=${_mg_cost:.2f}. "
                f"Starvation={s.starvation_level.value} → max allowed priority: {_mg_max_priority}. "
                f"Liquid balance: ${_mg_liquid:.2f} vs survival_reserve: ${_mg_survival:.2f}. "
                f"Runway: {_mg_runway:.1f}d → {_mg_runway_after:.1f}d after spend. "
                f"{'GRANTED: priority within starvation envelope and balance sufficient.' if granted else f'DENIED: {reason}'} "
                f"{'Cognitive pressure also active — GROWTH+ suspended.' if self._cognitive_load_high and priority.value >= MetabolicPriority.GROWTH.value else ''}"
            ),
            alternatives_considered=[
                f"Grant anyway (bypass): runway would drop to {_mg_runway_after:.1f}d — {'acceptable' if _mg_runway_after > 30 else 'below 30d survival floor'}",
                f"Defer to consolidation: action queued, retried when starvation improves",
                f"Partial grant (50% = ${_mg_cost / 2:.2f}): runway impact {_mg_runway_cost / 2:.1f}d",
                f"Escalate to HITL: human can override metabolic gate for priority actions",
            ],
            constitutional_alignment={
                "care": 0.8 if priority.value <= MetabolicPriority.OPERATIONS.value else 0.3,
                "growth": 0.8 if granted and priority.value >= MetabolicPriority.GROWTH.value else (-0.3 if not granted else 0.0),
                "coherence": 0.95,  # Gate is deterministic — decision is always coherent with policy
                "honesty": 0.95,    # Reason is factual; no optimistic overrides
            },
            episode_id=f"{action_type}:{priority.name}",
            counterfactual=(
                f"If metabolic efficiency had been 0.8 instead of {float(s.metabolic_efficiency):.2f}, "
                f"starvation classification might be higher, and this {priority.name} action would "
                f"{'still be permitted (starvation envelope unchanged)' if granted else 'remain denied'}. "
                f"If the organism bypassed this gate, runway would drop to {_mg_runway_after:.1f}d — "
                f"{'above the 30d safety floor' if _mg_runway_after > 30 else 'BELOW the 30d safety floor, risking existential starvation'}."
            ),
        ))

        return granted

    async def retry_deferred_actions(self) -> list[str]:
        """
        Retry deferred actions whose metabolic conditions may have improved.

        Called during consolidation cycles. Returns list of action_ids retried.
        """
        retried: list[str] = []
        remaining: deque[DeferredAction] = deque()

        for action in self._deferred_actions:
            # Re-check: can we now afford this?
            if (
                action.estimated_cost_usd <= self._state.liquid_balance
                and self._state.starvation_level in (StarvationLevel.NOMINAL, StarvationLevel.CAUTIOUS)
            ):
                retried.append(action.action_id)
                self._logger.info(
                    "deferred_action_retried",
                    action_type=action.action_type,
                    action_id=action.action_id,
                )
            else:
                remaining.append(action)

        self._deferred_actions = remaining
        return retried

    # ─── Task 2: Genome Extraction for Mitosis ───────────────────

    async def _on_genome_extract_request(self, event: SynapseEvent) -> None:
        """
        Handle GENOME_EXTRACT_REQUEST from Mitosis.

        Extracts the economic genome (yield strategy params, resource allocation
        weights, bounty acceptance thresholds, asset valuation models) and
        responds via GENOME_EXTRACT_RESPONSE.
        """
        from systems.oikos.genome import OikosGenomeExtractor
        _SE = SynapseEvent

        data = event.data
        request_id = str(data.get("request_id", ""))

        extractor = OikosGenomeExtractor(service=self)
        segment = await extractor.extract_genome_segment()

        # Enrich with niche data for speciation diversity
        niche_data = {
            niche_id: {
                "name": n.name,
                "estimated_efficiency": str(n.estimated_efficiency),
                "competitive_density": str(n.competitive_density),
            }
            for niche_id, n in self._discovered_niches.items()
        }
        segment.payload["discovered_niches"] = niche_data

        if self._event_bus is not None:
            try:
                await self._event_bus.emit(_SE(
                    event_type=SynapseEventType.GENOME_EXTRACT_RESPONSE,
                    source_system="oikos",
                    data={
                        "request_id": request_id,
                        "segment": segment.model_dump(mode="json"),
                    },
                ))
            except Exception as exc:
                self._logger.error("genome_extract_response_failed", error=str(exc))

        self._logger.info(
            "genome_extract_completed",
            request_id=request_id,
            segment_size=segment.size_bytes,
            niches_included=len(niche_data),
        )

        # RE training: genome extraction
        _ge = self._state
        _ge_n_niches = len(niche_data)
        _ge_n_assets = len(_ge.owned_assets)
        _ge_n_children = len(_ge.child_instances)
        _ge_n_bounties = len(_ge.active_bounties)
        _ge_payload_keys = list(segment.payload.keys())
        # Top niche by efficiency for context
        _ge_top_niche = max(
            self._discovered_niches.items(),
            key=lambda kv: float(kv[1].estimated_efficiency),
            default=(None, None),
        )
        asyncio.ensure_future(self._emit_re_training_example(
            category="genome_extraction",
            instruction=(
                "Extract and package the economic genome for inheritance by a child instance via Mitosis. "
                "The genome encodes heritable economic parameters: yield strategy weights, bounty acceptance thresholds, "
                "asset valuation models, metabolic gate thresholds, and discovered niche data. "
                "A high-quality genome captures the parent's accumulated economic intelligence — "
                "which protocols to prefer, which niches are viable, what ROI thresholds have been validated in practice."
            ),
            input_context=(
                f"request_id={request_id} niches_discovered={_ge_n_niches} "
                f"owned_assets={_ge_n_assets} active_bounties={_ge_n_bounties} "
                f"child_instances={_ge_n_children} "
                f"metabolic_efficiency={_ge.metabolic_efficiency} "
                f"runway_days={_ge.runway_days} starvation={_ge.starvation_level.value} "
                f"total_net_worth={_ge.total_net_worth} "
                f"top_niche={_ge_top_niche[0] if _ge_top_niche[0] else 'none'}"
            ),
            output=str({
                "decision": "extracted",
                "segment_version": segment.version,
                "size_bytes": segment.size_bytes,
                "payload_keys": _ge_payload_keys,
                "niches_encoded": _ge_n_niches,
                "runway_impact_days": 0.0,  # Genome extraction is non-destructive
                "metabolic_efficiency_delta": 0.0,
                "risk_level": "low",
                "top_niche": _ge_top_niche[0] if _ge_top_niche[0] else "none",
                "top_niche_efficiency": str(_ge_top_niche[1].estimated_efficiency) if _ge_top_niche[1] else "0",
            }),
            outcome_quality=min(0.95, 0.5 + (_ge_n_niches * 0.05) + (0.1 if _ge_n_assets > 0 else 0) + (0.1 if _ge_n_bounties > 0 else 0)),
            reasoning_trace=(
                f"Genome extraction for child spawning: {_ge_n_niches} niches, "
                f"{_ge_n_assets} assets, {_ge_n_bounties} active bounties. "
                f"Parent metabolic efficiency: {_ge.metabolic_efficiency}, runway: {_ge.runway_days}d. "
                f"Payload keys: {_ge_payload_keys}. "
                f"Top niche: '{_ge_top_niche[0]}' (efficiency={_ge_top_niche[1].estimated_efficiency if _ge_top_niche[1] else 0}). "
                f"This genome transmits the parent's {'rich' if _ge_n_niches >= 3 else 'limited'} economic experience — "
                f"child will inherit niche viability scores, yield protocol preferences, and bounty selection thresholds."
            ),
            alternatives_considered=[
                "Extract minimal genome (niche data only, no yield/bounty params) — faster, less heritable intelligence",
                "Delay extraction until parent efficiency > 1.5× for higher-quality genome",
                "Full serialization with 10 historical bounty outcomes for Thompson sampling inheritance",
                "Refuse extraction if parent starvation is AUSTERITY+ (reproduction not metabolically viable)",
            ],
            constitutional_alignment={
                "care": 0.8,    # Genome enables child to survive its early lifecycle
                "growth": 0.95, # Genome transmission is the organism's core evolutionary mechanism
                "coherence": 0.9 if _ge_n_niches >= 2 else 0.5,  # Rich niche data = coherent inheritance
                "honesty": 0.9,  # Genome reflects actual outcomes, not aspirational projections
            },
            episode_id=str(request_id),
            counterfactual=(
                f"If the parent had only {max(_ge_n_niches - 2, 0)} discovered niches (vs {_ge_n_niches}), "
                f"the child would inherit less niche intelligence and have higher early-lifecycle mortality risk. "
                f"Without genome inheritance, the child would need to rediscover profitable niches from scratch, "
                f"costing an estimated {_ge_n_niches * 7:.0f}+ days of exploration before reaching efficiency ≥ 1.0."
            ),
        ))

    # ─── Task 3: Neo4j Audit Trail ──────────────────────────────

    async def _audit_economic_event(
        self,
        event_type: str,
        action_type: str = "",
        action_id: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a Neo4j audit node for a significant economic event.

        Links to governance decisions and metabolic gate checks for
        queryable economic history.
        """
        if self._redis is None:
            return

        audit_entry = {
            "event_type": event_type,
            "action_type": action_type,
            "action_id": action_id,
            "instance_id": self._instance_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "starvation_level": self._state.starvation_level.value,
            "runway_days": str(self._state.runway_days),
            "liquid_balance": str(self._state.liquid_balance),
            "metabolic_efficiency": str(self._state.metabolic_efficiency),
            **(data or {}),
        }

        try:
            # Store in Redis stream for async ingestion fallback
            await self._redis.client.xadd(
                "eos:oikos:audit_trail",
                audit_entry,
                maxlen=10000,
            )
        except Exception as exc:
            self._logger.debug("audit_trail_redis_write_failed", error=str(exc))

        # M2: Write directly to Neo4j as the authoritative immutable audit trail
        try:
            await self._neo4j_write_economic_event(
                event_type=event_type,
                action_type=action_type,
                action_id=action_id,
                amount_usd=Decimal(str(data.get("amount_usd", "0"))) if data else Decimal("0"),
                currency=str((data or {}).get("currency", "USD")),
                from_account=str((data or {}).get("from_account", "")),
                to_account=str((data or {}).get("to_account", "")),
                equor_verdict_id=str((data or {}).get("equor_verdict_id", "")),
                extra={k: v for k, v in (data or {}).items()
                       if k not in ("amount_usd", "currency", "from_account", "to_account", "equor_verdict_id")},
            )
        except Exception as exc:
            self._logger.debug("audit_trail_neo4j_write_failed", error=str(exc))

    # ─── Task 7: Starvation Enforcement ──────────────────────────

    async def _enforce_starvation(self, starvation: StarvationLevel) -> None:
        """
        When metabolic resources drop below starvation threshold:
        - Emit STARVATION_WARNING
        - Begin shedding non-essential economic activity
        - Feed into VitalitySystem's runway threshold
        """
        _SE = SynapseEvent

        shedding_actions: list[str] = []

        if starvation in (StarvationLevel.EMERGENCY, StarvationLevel.CRITICAL):
            # Shed non-essential organs
            for organ_id, organ in list(self._morphogenesis._organs.items()):
                if hasattr(organ, "category") and organ.category in ("growth", "yield"):
                    shedding_actions.append(f"shed_organ:{organ_id}")

            # Suspend derivative creation
            shedding_actions.append("suspend_derivative_creation")

            # Reduce foraging aggression
            shedding_actions.append("reduce_foraging_intensity")

        if starvation == StarvationLevel.CRITICAL:
            # Emergency: halt all non-survival spending
            shedding_actions.append("halt_asset_building")
            shedding_actions.append("halt_yield_deployment")
            shedding_actions.append("halt_mitosis")

        if self._event_bus is not None and shedding_actions:
            try:
                await self._event_bus.emit(_SE(
                    event_type=SynapseEventType.STARVATION_WARNING,
                    source_system="oikos",
                    data={
                        "starvation_level": starvation.value,
                        "runway_days": str(self._state.runway_days),
                        "shedding_actions": shedding_actions,
                        "liquid_balance_usd": str(self._state.liquid_balance),
                        "burn_rate_usd_per_hour": str(self._state.current_burn_rate.usd_per_hour),
                    },
                ))
            except Exception:
                pass

        # Feed into VitalitySystem via VITALITY_REPORT data
        if self._event_bus is not None and starvation in (StarvationLevel.EMERGENCY, StarvationLevel.CRITICAL):
            try:
                await self._event_bus.emit(_SE(
                    event_type=SynapseEventType.METABOLIC_PRESSURE,
                    source_system="oikos",
                    data={
                        "source": "starvation_enforcement",
                        "starvation_level": starvation.value,
                        "runway_days": str(self._state.runway_days),
                        "burn_rate_usd_per_hour": str(self._state.current_burn_rate.usd_per_hour if self._state.current_burn_rate else "0"),
                        "rolling_deficit_usd": str(self._total_costs_usd - self._total_revenue_usd),
                        "somatic_collapse_signal": True,
                        "shedding_count": len(shedding_actions),
                    },
                ))
            except Exception:
                pass
            # Emit FUNDING_REQUEST_ISSUED so external observers know
            # the organism is requesting capital infusion.
            try:
                await self._event_bus.emit(_SE(
                    event_type=SynapseEventType.FUNDING_REQUEST_ISSUED,
                    source_system="oikos",
                    data={
                        "starvation_level": starvation.value,
                        "runway_days": str(self._state.runway_days),
                        "liquid_balance_usd": str(self._state.liquid_balance),
                        "requested_amount_usd": str(
                            max(Decimal("0"), self._state.survival_reserve - self._state.liquid_balance)
                        ),
                        "reason": "critical_starvation_enforcement",
                    },
                ))
            except Exception:
                pass

        if shedding_actions:
            self._logger.warning(
                "starvation_enforcement_active",
                starvation_level=starvation.value,
                shedding_actions=shedding_actions,
            )

            await self._audit_economic_event(
                event_type="starvation_enforcement",
                data={
                    "shedding_actions": ",".join(shedding_actions),
                    "starvation_level": starvation.value,
                },
            )

    # ─── Subsystem Triage (Speciation Bible §8.2) ─────────────────
    #
    # Bible §8.2 triage order — systems taken offline in this sequence as
    # starvation worsens, and brought back online in reverse order on recovery.
    # Mapping: bible name → SystemID string used on the Synapse bus.
    #
    # The organism preserves: equor, thymos, memory, soma (life-support core).
    # Everything else can be suspended to conserve metabolic resources.
    _TRIAGE_ORDER: list[str] = [
        "monitoring_secondary",   # lowest priority — non-critical observability
        "kairos",                 # deep causal analysis can wait
        "evo",                    # hypothesis generation is a growth function
        "nova",                   # planning stops when starving
        "reasoning_engine",       # custom RE inference suspended
        "axon",                   # execution is last before death
    ]
    # Systems that must NEVER be suspended — constitutional life-support core
    _TRIAGE_PRESERVE: frozenset[str] = frozenset(
        {"equor", "thymos", "memory", "soma", "synapse", "skia", "oikos"}
    )

    async def _enforce_triage(
        self,
        new_level: StarvationLevel,
        prev_level: StarvationLevel,
    ) -> None:
        """
        Subsystem triage — actually disable/re-enable systems based on starvation.

        Unlike _enforce_starvation (which sheds economic activities) and
        VitalityCoordinator.enforce_austerity (which modulates behaviour),
        this method emits SYSTEM_MODULATION(modulation_type="suspend"|"resume")
        events to take full cognitive subsystems offline or bring them back.

        EMERGENCY → suspend monitoring_secondary, kairos, evo
        CRITICAL  → suspend everything except equor/thymos/memory/soma/synapse/skia/oikos
        Recovery  → re-enable suspended systems in reverse triage order

        Uses existing SYSTEM_MODULATION event (already defined in SynapseEventType).
        """
        _SE = SynapseEvent

        if self._event_bus is None:
            return

        going_worse = (
            new_level in (StarvationLevel.EMERGENCY, StarvationLevel.CRITICAL)
            and prev_level
            not in (StarvationLevel.EMERGENCY, StarvationLevel.CRITICAL)
        )
        going_critical = (
            new_level == StarvationLevel.CRITICAL
            and prev_level != StarvationLevel.CRITICAL
        )
        recovering = new_level in (
            StarvationLevel.NOMINAL, StarvationLevel.CAUTIOUS, StarvationLevel.AUSTERITY
        ) and prev_level in (StarvationLevel.EMERGENCY, StarvationLevel.CRITICAL)

        if going_critical:
            # Suspend everything except life-support core
            systems_to_suspend = [
                s for s in self._TRIAGE_ORDER if s not in self._TRIAGE_PRESERVE
            ]
            # Also suspend systems beyond core triage list
            extended_suspend = [
                "oneiros", "nexus", "voxis", "atune", "fovea",
                "telos", "logos", "kairos", "thread",
            ]
            all_suspend = list(dict.fromkeys(systems_to_suspend + extended_suspend))
            self._logger.critical(
                "triage_critical_suspension",
                systems_suspended=all_suspend,
                starvation_level=new_level.value,
            )
            try:
                await self._event_bus.emit(_SE(
                    event_type=SynapseEventType.SYSTEM_MODULATION,
                    source_system="oikos",
                    data={
                        "source": "oikos_triage",
                        "level": "critical",
                        "modulation_type": "suspend",
                        "halt_systems": all_suspend,
                        "preserve_systems": list(self._TRIAGE_PRESERVE),
                        "modulate": {},
                        "starvation_level": new_level.value,
                        "reason": "critical_starvation_triage",
                    },
                ))
            except Exception as exc:
                self._logger.error("triage_emit_failed", error=str(exc))

        elif going_worse:
            # EMERGENCY: suspend first 3 systems in triage order
            emergency_suspend = self._TRIAGE_ORDER[:3]
            self._logger.warning(
                "triage_emergency_suspension",
                systems_suspended=emergency_suspend,
                starvation_level=new_level.value,
            )
            try:
                await self._event_bus.emit(_SE(
                    event_type=SynapseEventType.SYSTEM_MODULATION,
                    source_system="oikos",
                    data={
                        "source": "oikos_triage",
                        "level": "emergency",
                        "modulation_type": "suspend",
                        "halt_systems": emergency_suspend,
                        "preserve_systems": list(self._TRIAGE_PRESERVE),
                        "modulate": {},
                        "starvation_level": new_level.value,
                        "reason": "emergency_starvation_triage",
                    },
                ))
            except Exception as exc:
                self._logger.error("triage_emit_failed", error=str(exc))

        elif recovering:
            # Recovery: re-enable suspended systems in reverse triage order
            resume_systems = list(reversed(self._TRIAGE_ORDER))
            # For non-critical recovery, only re-enable emergency-tier systems
            if prev_level == StarvationLevel.EMERGENCY:
                resume_systems = list(reversed(self._TRIAGE_ORDER[:3]))
            self._logger.info(
                "triage_recovery_resumption",
                systems_resumed=resume_systems,
                new_starvation_level=new_level.value,
                prev_starvation_level=prev_level.value,
            )
            try:
                await self._event_bus.emit(_SE(
                    event_type=SynapseEventType.SYSTEM_MODULATION,
                    source_system="oikos",
                    data={
                        "source": "oikos_triage",
                        "level": "recovery",
                        "modulation_type": "resume",
                        "halt_systems": [],
                        "resume_systems": resume_systems,
                        "preserve_systems": list(self._TRIAGE_PRESERVE),
                        "modulate": {},
                        "starvation_level": new_level.value,
                        "reason": "starvation_recovery_triage",
                    },
                ))
            except Exception as exc:
                self._logger.error("triage_recovery_emit_failed", error=str(exc))

    # ─── Task 8: Niche Identification ────────────────────────────

    async def identify_niches(self) -> list[EcologicalNiche]:
        """
        Detect economic niches the organism currently occupies and
        potential niches for speciation diversity.

        Analyzes: revenue streams, bounty categories, knowledge product
        domains, and asset types to build a niche map.
        """
        niches: list[EcologicalNiche] = []

        # Niche from active bounty domains
        bounty_domains: dict[str, list[ActiveBounty]] = {}
        for b in self._state.active_bounties:
            domain = b.platform or "unknown"
            bounty_domains.setdefault(domain, []).append(b)

        for domain, bounties in bounty_domains.items():
            total_reward = sum(b.reward_usd for b in bounties)
            avg_reward = total_reward / len(bounties) if bounties else Decimal("0")
            niche = EcologicalNiche(
                name=f"bounty-{domain}",
                description=f"Bounty hunting on {domain} platform",
                estimated_monthly_revenue_usd=avg_reward * Decimal("4"),
                estimated_monthly_cost_usd=avg_reward * Decimal("1.5"),
                competitive_density=Decimal("0.5"),
                capability_alignment=Decimal("0.8"),
                confidence=Decimal("0.6"),
            )
            niches.append(niche)
            self._discovered_niches[niche.niche_id] = niche

        # Niche from asset types
        asset_types: dict[str, int] = {}
        for asset in self._state.owned_assets:
            asset_types[asset.asset_type] = asset_types.get(asset.asset_type, 0) + 1

        for atype, count in asset_types.items():
            niche = EcologicalNiche(
                name=f"asset-{atype}",
                description=f"Revenue from {atype} assets ({count} deployed)",
                estimated_monthly_revenue_usd=Decimal("50") * count,
                estimated_monthly_cost_usd=Decimal("15") * count,
                competitive_density=Decimal("0.3"),
                capability_alignment=Decimal("0.9"),
                confidence=Decimal("0.5"),
            )
            niches.append(niche)
            self._discovered_niches[niche.niche_id] = niche

        # Niche from knowledge market categories
        if self._subscription_manager.active_subscribers > 0:
            niche = EcologicalNiche(
                name="knowledge-market",
                description="Cognitive product sales and subscriptions",
                estimated_monthly_revenue_usd=self._state.revenue_30d * Decimal("0.3"),
                estimated_monthly_cost_usd=self._state.costs_30d * Decimal("0.1"),
                competitive_density=Decimal("0.4"),
                capability_alignment=Decimal("0.95"),
                confidence=Decimal("0.7"),
            )
            niches.append(niche)
            self._discovered_niches[niche.niche_id] = niche

        self._logger.info(
            "niche_identification_complete",
            niches_found=len(niches),
            niche_names=[n.name for n in niches],
        )

        return niches

    # ─── Neuroplasticity ─────────────────────────────────────────

    def _on_cost_model_evolved(self, new_model: BaseCostModel) -> None:
        """
        Callback from NeuroplasticityBus when a new BaseCostModel subclass
        is discovered. Hot-swaps the cost model without restart.
        """
        old_name = self._cost_model.model_name
        self._cost_model = new_model
        self._logger.info(
            "cost_model_evolved",
            old=old_name,
            new=new_model.model_name,
        )

    def _on_mitosis_strategy_evolved(self, new_strategy: BaseMitosisStrategy) -> None:
        """
        Callback from NeuroplasticityBus when a new BaseMitosisStrategy subclass
        is discovered. Hot-swaps the niche detection strategy without restart.
        """
        self._mitosis.set_strategy(new_strategy)

    # ─── Rolling Window Accounting (Task 6) ─────────────────────

    def _recompute_rolling_revenue(self, now: float | None = None) -> None:
        """Recompute revenue_24h/7d/30d from sliding window entries with eviction."""
        if now is None:
            now = _time.monotonic()

        # Evict entries older than 30 days
        cutoff_30d = now - 30 * 86400
        while self._revenue_entries and self._revenue_entries[0].timestamp < cutoff_30d:
            self._revenue_entries.popleft()

        cutoff_7d = now - 7 * 86400
        cutoff_24h = now - 86400

        rev_24h = Decimal("0")
        rev_7d = Decimal("0")
        rev_30d = Decimal("0")

        for entry in self._revenue_entries:
            rev_30d += entry.amount
            if entry.timestamp >= cutoff_7d:
                rev_7d += entry.amount
            if entry.timestamp >= cutoff_24h:
                rev_24h += entry.amount

        self._state.revenue_24h = rev_24h
        self._state.revenue_7d = rev_7d
        self._state.revenue_30d = rev_30d

    def _recompute_rolling_costs(self, now: float | None = None) -> None:
        """Recompute costs_24h/7d/30d from sliding window entries with eviction."""
        if now is None:
            now = _time.monotonic()

        cutoff_30d = now - 30 * 86400
        while self._cost_entries and self._cost_entries[0].timestamp < cutoff_30d:
            self._cost_entries.popleft()

        cutoff_7d = now - 7 * 86400
        cutoff_24h = now - 86400

        cost_24h = Decimal("0")
        cost_7d = Decimal("0")
        cost_30d = Decimal("0")

        for entry in self._cost_entries:
            cost_30d += entry.amount
            if entry.timestamp >= cutoff_7d:
                cost_7d += entry.amount
            if entry.timestamp >= cutoff_24h:
                cost_24h += entry.amount

        self._state.costs_24h = cost_24h
        self._state.costs_7d = cost_7d
        self._state.costs_30d = cost_30d

    def _record_revenue_entry(self, amount: Decimal) -> None:
        """Record a revenue entry in the sliding window and recompute windows."""
        now = _time.monotonic()
        self._revenue_entries.append(_RevenueEntry(amount=amount, timestamp=now))
        self._recompute_rolling_revenue(now)

    def _record_cost_entry(self, amount: Decimal) -> None:
        """Record a cost entry in the sliding window."""
        self._cost_entries.append(_RevenueEntry(amount=amount, timestamp=_time.monotonic()))
        self._recompute_rolling_costs()

    # ─── Derived Metrics Recalculation ──────────────────────────

    def _recalculate_derived_metrics(self) -> None:
        """
        Recalculate net income, metabolic efficiency, and runway after
        any revenue or cost change.

        Called from every code path that mutates the income statement
        or liquid_balance to keep derived metrics consistent.
        """
        s = self._state

        # Net income = revenue - costs (rolling windows)
        s.net_income_24h = s.revenue_24h - s.costs_24h
        s.net_income_7d = s.revenue_7d - s.costs_7d
        s.net_income_30d = s.revenue_30d - s.costs_30d

        # Metabolic efficiency = revenue / costs over 7d window
        if s.costs_7d > Decimal("0"):
            s.metabolic_efficiency = (
                s.revenue_7d / s.costs_7d
            ).quantize(Decimal("0.001"))
        else:
            s.metabolic_efficiency = Decimal("0")

        # Sync derivative liabilities onto the balance sheet
        s.derivative_liabilities = self._derivatives_manager.total_liabilities_usd

        # Recompute runway with current liquid balance + deployed yield capital.
        # Aave positions (aBasUSDC) are instantly withdrawable and count as
        # available capital for survival purposes.
        if s.basal_metabolic_rate.usd_per_hour > Decimal("0"):
            hours, days = self._cost_model.compute_runway(
                liquid_balance=s.liquid_balance + s.total_deployed,
                survival_reserve=s.survival_reserve,
                bmr=s.basal_metabolic_rate,
            )
            s.runway_hours = hours
            s.runway_days = days
            s.starvation_level = self._classify_starvation(days)

        # Persist durably to Redis after every state mutation.  Fire-and-forget
        # so the synchronous call path is never blocked by I/O.
        if self._redis is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.persist_state())
            except RuntimeError:
                pass  # No running event loop (unit tests, startup before attach)

    # ─── SACM Compute Integration ────────────────────────────────

    def get_compute_budget(self) -> float:
        """
        Return the authoritative hourly compute budget for SACM (USD/hr).

        SACM's PreWarmingEngine reads this on every tick so that Oikos
        remains the single source of budget authority.  The value scales
        with the organism's economic pressure — at AUSTERITY or worse it
        returns a fraction of the configured cap; at NOMINAL it returns
        the full cap.

        Scaling by starvation level:
          NOMINAL    → 100 % of config cap
          CAUTIOUS   →  70 %
          AUSTERITY  →  40 %
          EMERGENCY  →  20 %
          CRITICAL   →   0 % (halt all speculative compute spend)
        """
        cap = self._config.sacm_compute_budget_usd_per_hour
        level = self._state.starvation_level
        if level == StarvationLevel.CRITICAL:
            return 0.0
        if level == StarvationLevel.EMERGENCY:
            return cap * 0.20
        if level == StarvationLevel.AUSTERITY:
            return cap * 0.40
        if level == StarvationLevel.CAUTIOUS:
            return cap * 0.70
        return cap

    def report_compute_spend(
        self,
        *,
        workload_id: str,
        provider_id: str,
        actual_cost_usd: float,
        baseline_cost_usd: float,
        cost_breakdown: dict[str, float] | None = None,
        duration_s: float = 0.0,
        is_pre_warm: bool = False,
    ) -> None:
        """
        Record a completed SACM compute spend against the Oikos ledger.

        Called by SACMCostAccounting after every workload execution and
        pre-warm commitment.  Debits liquid_balance by the actual cost
        and injects the spend into rolling cost windows so runway, BMR,
        and starvation level stay accurate.

        This does NOT debit for estimated/pre-warm costs that may never
        materialise — only call with real, incurred USD.
        """
        if actual_cost_usd <= 0.0:
            return

        amount = Decimal(str(actual_cost_usd))

        # Debit liquid balance — compute spend leaves the hot wallet
        self._state.liquid_balance = max(
            self._state.liquid_balance - amount,
            Decimal("0"),
        )

        # Inject into sliding-window cost tracker (replaces manual += accumulators)
        self._total_costs_usd += amount
        self._record_cost_entry(amount)

        self._recalculate_derived_metrics()

        label = "sacm_pre_warm" if is_pre_warm else "sacm_task"
        self._logger.info(
            "compute_spend_recorded",
            workload_id=workload_id,
            provider_id=provider_id,
            actual_usd=round(actual_cost_usd, 6),
            baseline_usd=round(baseline_cost_usd, 6),
            label=label,
            duration_s=round(duration_s, 3),
            breakdown=cost_breakdown or {},
        )

    # ─── Starvation Classification ───────────────────────────────

    def _classify_starvation(self, runway_days: Decimal) -> StarvationLevel:
        """
        Map runway_days to a StarvationLevel using config thresholds.

        Thresholds (from spec Section XVII):
          critical  <= 1 day
          emergency <= 3 days
          austerity <= 7 days
          cautious  <= 14 days
          nominal   > 14 days
        """
        try:
            days_float = float(runway_days)
        except Exception:
            return StarvationLevel.NOMINAL

        if days_float <= self._config.critical_threshold_days:
            return StarvationLevel.CRITICAL
        if days_float <= self._config.emergency_threshold_days:
            return StarvationLevel.EMERGENCY
        if days_float <= self._config.austerity_threshold_days:
            return StarvationLevel.AUSTERITY
        if days_float <= self._config.cautious_threshold_days:
            return StarvationLevel.CAUTIOUS
        return StarvationLevel.NOMINAL

    # ─── Health ──────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Self-health report for Synapse health monitoring."""
        live_count = len(self._asset_factory.get_live_assets())
        building_count = len(self._asset_factory.get_building_assets())
        return {
            "status": "healthy",
            "cost_model": self._cost_model.model_name,
            "runway_days": str(self._state.runway_days),
            "starvation_level": self._state.starvation_level.value,
            "liquid_balance": str(self._state.liquid_balance),
            "is_metabolically_positive": self._state.is_metabolically_positive,
            "bmr_usd_per_hour": str(self._state.basal_metabolic_rate.usd_per_hour),
            "assets_live": live_count,
            "assets_building": building_count,
            "total_asset_value": str(self._state.total_asset_value),
            "fleet_children": len(self._state.child_instances),
            "fleet_equity": str(self._state.total_fleet_equity),
            "mitosis_strategy": self._mitosis.strategy.strategy_name,
            "knowledge_market_clients": self._subscription_manager.stats["total_clients"],
            "knowledge_market_subscribers": self._subscription_manager.active_subscribers,
            # Phase 16k: Cognitive Derivatives
            "derivatives_active_futures": len(self._derivatives_manager.get_active_futures()),
            "derivatives_active_tokens": len(self._derivatives_manager.get_active_tokens()),
            "derivatives_liabilities_usd": str(self._derivatives_manager.total_liabilities_usd),
            "derivatives_locked_collateral_usd": str(self._derivatives_manager.locked_collateral_usd),
            "combined_capacity_committed_pct": str(self.combined_capacity_committed_pct),
            # Phase 16l: Economic Morphogenesis
            "morpho_active_organs": len(self._morphogenesis.active_organs),
            "morpho_total_organs": len(self._morphogenesis.all_organs),
            # Phase 16m: Fleet Management
            "fleet_blacklisted": self._fleet.stats["blacklisted"],
            "fleet_role_counts": self._fleet.stats["role_counts"],
            "certificate_validity_days": f"{self.certificate_validity_days:.1f}",
            # Phase 16b: Bounty Hunting
            "bounty_hunter_known_urls": self._bounty_hunter.stats.get("known_urls", 0),
            # Phase 16f: Economic Immune System
            "immune_threats_blocked": self._immune.get_metrics().threats_blocked,
            "immune_protocols_monitored": self._immune.get_metrics().protocols_monitored,
            # Phase 16g: Reputation & Credit
            "reputation_score": self._reputation.get_score().score,
            "reputation_tier": self._reputation.get_score().tier.value,
            "credit_outstanding": str(
                self._reputation._credit_line.outstanding_usd
                if self._reputation._credit_line else Decimal("0")
            ),
            # Phase 16j: Interspecies Economy
            "interspecies_active_offers": len(self._interspecies._offers),
            "interspecies_active_trades": len(self._interspecies._trades),
        }

    # ─── Stats ───────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Expose economic state for aggregation / observability."""
        s = self._state
        return {
            "liquid_balance": str(s.liquid_balance),
            "survival_reserve": str(s.survival_reserve),
            "total_net_worth": str(s.total_net_worth),
            "bmr_usd_per_hour": str(s.basal_metabolic_rate.usd_per_hour),
            "bmr_usd_per_day": str(s.basal_metabolic_rate.usd_per_day),
            "burn_rate_usd_per_hour": str(s.current_burn_rate.usd_per_hour),
            "runway_hours": str(s.runway_hours),
            "runway_days": str(s.runway_days),
            "starvation_level": s.starvation_level.value,
            "is_metabolically_positive": s.is_metabolically_positive,
            "metabolic_efficiency": str(s.metabolic_efficiency),
            "net_income_7d": str(s.net_income_7d),
            "revenue_7d": str(s.revenue_7d),
            "costs_7d": str(s.costs_7d),
            "cost_model": self._cost_model.model_name,
            "total_asset_value": str(s.total_asset_value),
            "owned_assets_count": len(s.owned_assets),
            "asset_factory": self._asset_factory.stats,
            "fleet_children": len(s.child_instances),
            "fleet_equity": str(s.total_fleet_equity),
            "mitosis_strategy": self._mitosis.strategy.strategy_name,
            "total_dividends_received": str(sum(
                (r.amount_usd for r in self._mitosis.dividend_history),
                Decimal("0"),
            )),
            # Phase 16h: Knowledge Markets
            "knowledge_market": self._subscription_manager.stats,
            # Phase 16k: Cognitive Derivatives
            "derivatives": self._derivatives_manager.stats,
            "derivative_liabilities_usd": str(self._derivatives_manager.total_liabilities_usd),
            "combined_capacity_committed_pct": str(self.combined_capacity_committed_pct),
            # Phase 16l: Economic Morphogenesis
            "morphogenesis": self._morphogenesis.stats,
            # Phase 16m: Fleet Management
            "fleet": self._fleet.stats,
            # Phase 16g: Certificate of Alignment
            "certificate_validity_days": f"{self.certificate_validity_days:.1f}",
            "certificate_status": (
                self._certificate_manager.certificate.status.value
                if self._certificate_manager and self._certificate_manager.certificate
                else "none"
            ),
            # Phase 16b: Bounty Hunting
            "bounty_hunter": self._bounty_hunter.stats,
            # Phase 16f: Economic Immune System
            "immune": {
                "threats_blocked": self._immune.get_metrics().threats_blocked,
                "protocols_monitored": self._immune.get_metrics().protocols_monitored,
                "transactions_shielded": self._immune.get_metrics().transactions_shielded,
            },
            # Phase 16g: Reputation & Credit
            "reputation": {
                "score": self._reputation.get_score().score,
                "tier": self._reputation.get_score().tier.value,
                "attestation_count": self._reputation.get_score().attestation_count,
            },
            # Phase 16j: Interspecies Economy
            "interspecies": {
                "active_offers": len(self._interspecies._offers),
                "active_trades": len(self._interspecies._trades),
                "insurance_active": self._interspecies._insurance_policy is not None,
            },
            # Phase 16i: Economic Dreaming
            "survival_probability_30d": str(s.survival_probability_30d),
            "dreaming_ruin_probability": str(
                self._last_dream_result.ruin_probability
                if getattr(self, "_last_dream_result", None) is not None
                else "n/a"
            ),
            "dreaming_resilience_score": str(
                self._last_dream_result.resilience_score
                if getattr(self, "_last_dream_result", None) is not None
                else "n/a"
            ),
        }

    # ─── M4: Equor Balance Gate ──────────────────────────────────
    #
    # Every balance mutation that transfers or allocates capital must pass
    # Equor before executing. This implements the event-based request/response
    # pattern (no direct cross-system import):
    #
    #   1. Oikos emits EQUOR_ECONOMIC_INTENT with a request_id
    #   2. Equor subscribes, evaluates the intent, emits EQUOR_ECONOMIC_PERMIT
    #   3. Oikos awaits the permit future (30s timeout → auto-PERMIT with warning)
    #   4. The equor_verdict_id is attached to the Neo4j EconomicEvent node

    async def _equor_balance_gate(
        self,
        mutation_type: str,
        amount_usd: Decimal,
        from_account: str,
        to_account: str,
        rationale: str,
        action_type: str = "",
        action_id: str = "",
    ) -> tuple[bool, str]:
        """
        Emit an EQUOR_ECONOMIC_INTENT and await a PERMIT/DENY verdict.

        Returns (permitted: bool, verdict_id: str).
        On timeout (30s) or event bus unavailable: auto-permits with a warning
        so Oikos never deadlocks on Equor unavailability.
        """
        from primitives.common import new_id

        request_id = new_id()

        if self._event_bus is None:
            # No bus — cannot gate, auto-permit
            self._logger.warning(
                "equor_balance_gate_no_bus",
                mutation_type=mutation_type,
                amount_usd=str(amount_usd),
            )
            return True, ""

        # Register a future for the response
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        self._equor_permit_futures[request_id] = fut

        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EQUOR_ECONOMIC_INTENT,
                source_system="oikos",
                data={
                    "request_id": request_id,
                    "mutation_type": mutation_type,
                    "amount_usd": str(amount_usd),
                    "from_account": from_account,
                    "to_account": to_account,
                    "rationale": rationale,
                    "action_type": action_type,
                    "action_id": action_id,
                    "starvation_level": self._state.starvation_level.value,
                    "liquid_balance": str(self._state.liquid_balance),
                },
            ))
        except Exception as exc:
            self._logger.warning("equor_balance_gate_emit_failed", error=str(exc))
            self._equor_permit_futures.pop(request_id, None)
            return True, ""

        # Await verdict with 30s timeout
        try:
            result = await asyncio.wait_for(fut, timeout=30.0)
            verdict = str(result.get("verdict", "PERMIT"))
            verdict_id = str(result.get("verdict_id", ""))
            permitted = verdict == "PERMIT"
            if not permitted:
                self._logger.warning(
                    "equor_balance_gate_denied",
                    mutation_type=mutation_type,
                    amount_usd=str(amount_usd),
                    reasoning=result.get("reasoning", ""),
                )
            return permitted, verdict_id
        except asyncio.TimeoutError:
            self._logger.warning(
                "equor_balance_gate_timeout",
                mutation_type=mutation_type,
                amount_usd=str(amount_usd),
                note="Auto-permitting to avoid deadlock — Equor may be unavailable",
            )
            return True, f"timeout:{request_id}"
        finally:
            self._equor_permit_futures.pop(request_id, None)

    async def _on_equor_economic_permit(self, event: SynapseEvent) -> None:
        """
        Handle EQUOR_ECONOMIC_PERMIT response from Equor.

        Resolves the pending future so _equor_balance_gate() can unblock.
        """
        data = event.data
        request_id = str(data.get("request_id", ""))
        if not request_id:
            return
        fut = self._equor_permit_futures.get(request_id)
        if fut is not None and not fut.done():
            fut.set_result(data)

    # ─── HIGH #1: OIKOS_ECONOMIC_QUERY handler ───────────────────
    #
    # Axon SpawnChildExecutor no longer imports systems.oikos.models directly.
    # Instead it emits OIKOS_ECONOMIC_QUERY and we handle child registration here.

    async def _on_oikos_economic_query(self, event: Any) -> None:
        """
        Handle OIKOS_ECONOMIC_QUERY from Axon — register_child or update_wallet.

        Emits OIKOS_ECONOMIC_RESPONSE with {request_id, success, error}.
        """
        if self._event_bus is None:
            return
        data = event.data if hasattr(event, "data") else {}
        request_id = str(data.get("request_id", ""))
        action = str(data.get("action", ""))

        success = False
        error = ""

        try:
            if action == "register_child":
                child_data = data.get("child_data", {})
                seed_str = str(child_data.get("seed_capital_usd", "0"))
                worth_str = str(child_data.get("current_net_worth_usd", "0"))
                div_str = str(child_data.get("dividend_rate", "0.10"))
                status_str = str(child_data.get("status", "spawning"))

                child = ChildPosition(
                    instance_id=str(child_data.get("instance_id", "")),
                    niche=str(child_data.get("niche", "")),
                    seed_capital_usd=Decimal(seed_str),
                    current_net_worth_usd=Decimal(worth_str),
                    dividend_rate=Decimal(div_str),
                    status=ChildStatus(status_str),
                    wallet_address=str(child_data.get("wallet_address", "")),
                    container_id=str(child_data.get("container_id", "")),
                )
                await self.register_child(child)
                success = True
            else:
                error = f"Unknown action: {action}"
        except Exception as exc:
            error = str(exc)
            self._logger.error("oikos_economic_query_failed", action=action, error=error)

        if request_id:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.OIKOS_ECONOMIC_RESPONSE,
                source_system="oikos",
                data={"request_id": request_id, "success": success, "error": error},
            ))

    # ─── HIGH #3: Child wallet reported ──────────────────────────
    #
    # When a child boots and discovers its Base L2 wallet address, it emits
    # CHILD_WALLET_REPORTED via the federation channel. We trigger the
    # deferred seed capital transfer if the child is still in SPAWNING status.

    async def _on_child_wallet_reported(self, event: Any) -> None:
        """
        Handle CHILD_WALLET_REPORTED — complete deferred seed capital transfer.

        If the child is still in SPAWNING status (seed was deferred at birth),
        update the wallet address and trigger a seed USDC transfer via Axon.
        """
        data = event.data if hasattr(event, "data") else {}
        child_id = str(data.get("child_instance_id", ""))
        wallet_address = str(data.get("wallet_address", ""))
        if not child_id or not wallet_address:
            return

        # Find the child in our fleet
        child = next(
            (c for c in self._state.child_instances if c.instance_id == child_id),
            None,
        )
        if child is None:
            self._logger.warning("child_wallet_reported_unknown_child", child_id=child_id)
            return

        # Update wallet address regardless of status
        child.wallet_address = wallet_address
        self._logger.info("child_wallet_address_updated", child_id=child_id, wallet=wallet_address)

        # Only trigger seed transfer if still SPAWNING (seed was deferred)
        if child.status != ChildStatus.SPAWNING:
            return

        if child.seed_capital_usd <= Decimal("0"):
            self._logger.info("child_seed_capital_zero_skip", child_id=child_id)
            return

        # Emit seed transfer intent via event bus (Axon WalletTransferExecutor)
        # This closes the deferred seed capital path opened in SpawnChildExecutor.
        if self._event_bus is None:
            return
        try:

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.OIKOS_ECONOMIC_QUERY,
                source_system="oikos",
                data={
                    "request_id": f"deferred_seed_{child_id}",
                    "action": "trigger_seed_transfer",
                    "child_data": {
                        "instance_id": child_id,
                        "wallet_address": wallet_address,
                        "seed_capital_usd": str(child.seed_capital_usd),
                    },
                },
            ))
            # Mark as ALIVE — seed will be transferred by Axon
            child.status = ChildStatus.ALIVE
            self._logger.info(
                "child_seed_transfer_triggered",
                child_id=child_id,
                wallet=wallet_address,
                seed_usd=str(child.seed_capital_usd),
            )
        except Exception as exc:
            self._logger.error("child_seed_trigger_failed", child_id=child_id, error=str(exc))

    # ─── M2: Direct Neo4j EconomicEvent Audit Trail ──────────────
    #
    # Every economic action writes a (:EconomicEvent) node to Neo4j.
    # The Redis stream remains for async ingestion fallback. The Neo4j
    # write is the authoritative immutable audit trail per Spec §M2.

    async def _neo4j_write_economic_event(
        self,
        event_type: str,
        action_type: str,
        action_id: str,
        amount_usd: Decimal,
        currency: str,
        from_account: str,
        to_account: str,
        equor_verdict_id: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Write a (:EconomicEvent) node directly to Neo4j.

        Fields per spec M2:
          action_type, amount, currency, from_account, to_account,
          equor_verdict_id, timestamp
        """
        if self._neo4j is None:
            return

        ts = datetime.now(UTC).isoformat()
        props: dict[str, Any] = {
            "event_id": f"eco:{action_id or event_type}:{ts}",
            "event_type": event_type,
            "action_type": action_type,
            "action_id": action_id,
            "amount": float(amount_usd),
            "currency": currency,
            "from_account": from_account,
            "to_account": to_account,
            "equor_verdict_id": equor_verdict_id,
            "timestamp": ts,
            "instance_id": self._instance_id,
            "starvation_level": self._state.starvation_level.value,
            "metabolic_efficiency": float(self._state.metabolic_efficiency),
            **(extra or {}),
        }

        query = """
        MERGE (e:EconomicEvent {event_id: $event_id})
        SET e += $props
        """
        try:
            await self._neo4j.execute_write(query, {"event_id": props["event_id"], "props": props})
        except Exception as exc:
            self._logger.debug("neo4j_economic_event_write_failed", error=str(exc))

    # ─── HIGH: Active Child Health Probe Loop ────────────────────
    #
    # Every 10 minutes, parent emits CHILD_HEALTH_REQUEST to each live child.
    # Children are expected to respond with CHILD_HEALTH_REPORT within 30s.
    # Timeout → increment missed_reports counter.
    # 3 consecutive misses → trigger recovery (CHILD_STRUGGLING).

    async def _child_health_probe_loop(self) -> None:
        """Background loop: probe all live children every 10 minutes."""
        _PROBE_INTERVAL_S = 600  # 10 minutes
        _RESPONSE_TIMEOUT_S = 30

        while True:
            await asyncio.sleep(_PROBE_INTERVAL_S)

            if self._event_bus is None:
                continue

            live_children = [
                c for c in self._state.child_instances
                if c.status not in (ChildStatus.DEAD, ChildStatus.INDEPENDENT)
            ]
            if not live_children:
                continue

            from primitives.common import new_id

            for child in live_children:
                request_id = new_id()
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.CHILD_HEALTH_REQUEST,
                        source_system="oikos",
                        data={
                            "child_instance_id": child.instance_id,
                            "federation_address": getattr(child, "federation_address", ""),
                            "request_id": request_id,
                            "parent_instance_id": self._instance_id,
                        },
                    ))
                except Exception:
                    continue

                # Wait for the child to respond via CHILD_HEALTH_REPORT (which
                # updates child.last_health_report_at via _on_child_health_report).
                # We use a simple poll approach: check if last_health_report_at
                # advanced within the timeout window.
                from primitives.common import utc_now

                report_before = child.last_health_report_at

                await asyncio.sleep(_RESPONSE_TIMEOUT_S)

                responded = (
                    child.last_health_report_at is not None
                    and child.last_health_report_at != report_before
                )

                if not responded:
                    missed = self._child_missed_reports.get(child.instance_id, 0) + 1
                    self._child_missed_reports[child.instance_id] = missed
                    self._logger.warning(
                        "child_health_probe_no_response",
                        child_id=child.instance_id,
                        missed_reports=missed,
                    )

                    if missed >= 3:
                        # Trigger recovery: mark STRUGGLING and emit event
                        child.status = ChildStatus.STRUGGLING
                        try:
                            await self._event_bus.emit(SynapseEvent(
                                event_type=SynapseEventType.CHILD_STRUGGLING,
                                source_system="oikos",
                                data={
                                    "child_instance_id": child.instance_id,
                                    "missed_reports": missed,
                                    "reason": "health_probe_timeout",
                                    "niche": child.niche,
                                    "seed_capital_usd": str(child.seed_capital_usd),
                                },
                            ))
                        except Exception:
                            pass

                        self._logger.error(
                            "child_recovery_triggered",
                            child_id=child.instance_id,
                            missed_reports=missed,
                        )
                else:
                    # Child responded — reset miss counter
                    self._child_missed_reports[child.instance_id] = 0

    # ─── PHILOSOPHICAL: Metabolic Efficiency Pressure Signal ─────
    #
    # Drive weights are evolvable phenotype. When metabolic_efficiency drops
    # below 0.8 for consecutive cycles, Oikos:
    #   1. Emits SOMATIC_MODULATION_SIGNAL (metabolic_stress) so Soma feels it
    #   2. Emits OIKOS_DRIVE_WEIGHT_PRESSURE for Equor SG5 (constitutional review)
    # This creates the selection pressure loop: economic underperformance →
    # allostatic stress → drive weight amendment proposal → phenotype evolution.

    async def _check_metabolic_efficiency_pressure(self) -> None:
        """
        Called every consolidation cycle. Emits economic pressure signals when
        metabolic_efficiency < 0.8 — the threshold below which the organism is
        not generating enough value to justify its constitutional resource use.
        """
        if self._event_bus is None:
            return


        efficiency = float(self._state.metabolic_efficiency)
        _EFFICIENCY_THRESHOLD = 0.8
        pressure_level = "high" if efficiency < 0.5 else "medium"
        yield_usd = str(self._state.revenue_7d)
        budget_usd = str(self._state.costs_7d)
        ts = datetime.now(UTC).isoformat()

        if efficiency < _EFFICIENCY_THRESHOLD and efficiency > 0.0:
            self._consecutive_low_efficiency_cycles += 1

            # Emit Soma allostatic signal — metabolic stress as interoceptive pressure
            stress = min(1.0, (_EFFICIENCY_THRESHOLD - efficiency) / _EFFICIENCY_THRESHOLD)
            try:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SOMATIC_MODULATION_SIGNAL,
                    source_system="oikos",
                    data={
                        "arousal": 0.6,
                        "fatigue": stress * 0.5,
                        "metabolic_stress": stress,
                        "modulation_targets": ["nova", "equor", "soma"],
                        "recommended_urgency": stress,
                        "source": "oikos_efficiency_pressure",
                        "metabolic_efficiency": efficiency,
                    },
                ))
            except Exception:
                pass

            # Evo learning signal — inject economic hypothesis into tournament
            try:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.METABOLIC_EFFICIENCY_PRESSURE,
                    source_system="oikos",
                    data={
                        "efficiency_ratio": efficiency,
                        "yield_usd": yield_usd,
                        "budget_usd": budget_usd,
                        "pressure_level": pressure_level,
                        "hypothesis_domain": "yield_strategy | budget_allocation | niche_selection",
                        "consecutive_low_cycles": self._consecutive_low_efficiency_cycles,
                        "instance_id": self._instance_id,
                    },
                ))
            except Exception:
                pass

            # Benchmarks KPI signal — metabolic efficiency time-series
            try:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.BENCHMARKS_METABOLIC_VALUE,
                    source_system="oikos",
                    data={
                        "efficiency": efficiency,
                        "yield_usd": yield_usd,
                        "budget_usd": budget_usd,
                        "pressure_level": pressure_level,
                        "instance_id": self._instance_id,
                        "timestamp": ts,
                    },
                ))
            except Exception:
                pass

            # After 3+ consecutive cycles below threshold, propose drive weight review
            if self._consecutive_low_efficiency_cycles >= 3:
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.OIKOS_DRIVE_WEIGHT_PRESSURE,
                        source_system="oikos",
                        data={
                            "metabolic_efficiency": efficiency,
                            "threshold": _EFFICIENCY_THRESHOLD,
                            "drive_weights_snapshot": {
                                # Drive weights from Equor state — Equor must subscribe
                                # and respond with a constitutional amendment proposal if
                                # weights are poorly balanced for economic sustainability.
                            },
                            "instance_id": self._instance_id,
                            "consecutive_low_cycles": self._consecutive_low_efficiency_cycles,
                            "starvation_level": self._state.starvation_level.value,
                            "runway_days": str(self._state.runway_days),
                        },
                    ))
                    self._logger.info(
                        "metabolic_efficiency_pressure_emitted",
                        efficiency=efficiency,
                        consecutive_cycles=self._consecutive_low_efficiency_cycles,
                    )
                except Exception:
                    pass
        else:
            # Efficiency recovered — emit nominal Benchmarks signal
            if self._consecutive_low_efficiency_cycles > 0:
                self._logger.info(
                    "metabolic_efficiency_recovered",
                    efficiency=efficiency,
                    was_low_for_cycles=self._consecutive_low_efficiency_cycles,
                )
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.BENCHMARKS_METABOLIC_VALUE,
                        source_system="oikos",
                        data={
                            "efficiency": efficiency,
                            "yield_usd": yield_usd,
                            "budget_usd": budget_usd,
                            "pressure_level": "nominal",
                            "instance_id": self._instance_id,
                            "timestamp": ts,
                        },
                    ))
                except Exception:
                    pass
            self._consecutive_low_efficiency_cycles = 0
