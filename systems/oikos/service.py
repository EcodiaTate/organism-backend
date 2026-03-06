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
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

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

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from clients.wallet import WalletClient
    from config import OikosConfig
    from core.hotreload import NeuroplasticityBus
    from systems.identity.manager import CertificateManager
    from systems.oikos.yield_strategy import DeploymentOutcome
    from systems.synapse.event_bus import EventBus
    from systems.synapse.metabolism import MetabolicTracker
    from systems.synapse.types import SynapseEvent


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
        from systems.synapse.types import SynapseEventType

        self._event_bus = event_bus

        event_bus.subscribe(
            SynapseEventType.METABOLIC_PRESSURE,
            self._on_metabolic_pressure,
        )
        event_bus.subscribe(
            SynapseEventType.REVENUE_INJECTED,
            self._on_revenue_injected,
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
            ],
        )

    async def shutdown(self) -> None:
        """Deregister from neuroplasticity bus."""
        if self._bus is not None:
            self._bus.deregister(BaseCostModel)
            self._bus.deregister(BaseMitosisStrategy)
            self._logger.info("neuroplasticity_deregistered")

    # ─── Event Handlers ──────────────────────────────────────────

    async def _on_metabolic_pressure(self, event: SynapseEvent) -> None:
        """
        Handle METABOLIC_PRESSURE from Synapse.

        Fires every ~50 cycles when burn rate exceeds the pressure threshold.
        We use this to update BMR, current burn rate, runway, and starvation level.
        """
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

        # Compute runway
        runway_hours, runway_days = self._cost_model.compute_runway(
            liquid_balance=self._state.liquid_balance,
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
        self._state.costs_24h = current_burn.usd_per_day
        self._state.costs_7d = current_burn.usd_per_day * Decimal("7")
        self._state.costs_30d = current_burn.usd_per_day * Decimal("30")

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

        # Sync derivative_liabilities and persist state. The inline cost/efficiency
        # computation above mirrors _recalculate_derived_metrics but omits the
        # derivative sync and the fire-and-forget persist. Call it now so that
        # (1) derivative_liabilities is current and (2) Redis durability fires.
        self._recalculate_derived_metrics()

        # MVP: Runway alarm — check after every metabolic update.
        # Fire-and-forget: the alarm emits events and POSTs webhooks asynchronously.
        asyncio.ensure_future(
            self._metabolism_api.check_runway_alarm(
                runway_days=runway_days,
                daily_cost=current_burn.usd_per_day,
            ),
        )

    async def _on_revenue_injected(self, event: SynapseEvent) -> None:
        """
        Handle REVENUE_INJECTED from Synapse.

        Fires when external revenue arrives (wallet top-up, bounty payment, etc.).
        Updates the revenue side of the income statement and credits liquid_balance.
        """
        data = event.data
        amount = Decimal(str(data.get("amount_usd", 0)))

        self._total_revenue_usd += amount

        # Credit liquid_balance — revenue arrives in the hot wallet
        self._state.liquid_balance += amount

        # Distribute across rolling windows (simplified — Phase 16i will
        # implement proper windowed accounting with TimescaleDB)
        self._state.revenue_24h += amount
        self._state.revenue_7d += amount
        self._state.revenue_30d += amount

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
            usdc_balance = await self._wallet.get_usdc_balance()
            self._state.liquid_balance = usdc_balance

            # Also inform MetabolicTracker so its hours_until_depleted is accurate
            if self._metabolism is not None:
                self._metabolism.snapshot(available_balance_usd=float(usdc_balance))

            # Recompute all derived metrics (runway, efficiency, liabilities)
            self._recalculate_derived_metrics()

            self._logger.debug(
                "oikos_balance_polled",
                usdc=str(usdc_balance),
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
        return await self._metabolism_api.check_budget(
            system_id=system_id,
            action=action,
            estimated_cost_usd=estimated_cost_usd,
        )

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
            await self._yield_tracker.record_position(outcome)
            # Update local state: deployed balance is no longer liquid
            self._state.total_deployed += outcome.amount_deployed_usd
            self._state.liquid_balance -= outcome.amount_deployed_usd
            self._recalculate_derived_metrics()
            self._logger.info(
                "oikos_capital_deployed",
                tx_hash=outcome.tx_hash,
                amount_usd=str(outcome.amount_deployed_usd),
                protocol=outcome.protocol,
                apy=str(outcome.apy),
                expected_daily_yield_usd=str(outcome.expected_daily_yield_usd),
            )
        else:
            self._logger.warning(
                "oikos_capital_deployment_skipped",
                error=outcome.error,
                degraded=outcome.degraded,
            )

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
        for asset in self._asset_factory.get_live_assets():
            try:
                swept = await self._tollbooth_manager.sweep_revenue(asset.asset_id)
                if swept > Decimal("0"):
                    self._asset_factory.record_revenue(asset.asset_id, swept)
                    swept_total += swept
            except Exception as exc:
                self._logger.warning(
                    "asset_sweep_failed",
                    asset_id=asset.asset_id,
                    error=str(exc),
                )

        # Inject swept revenue into the income statement and credit liquid_balance
        if swept_total > Decimal("0"):
            self._state.liquid_balance += swept_total
            self._state.revenue_24h += swept_total
            self._state.revenue_7d += swept_total
            self._state.revenue_30d += swept_total
            self._total_revenue_usd += swept_total
            self._credit_revenue_source(RevenueStream.ASSET, swept_total)
            self._recalculate_derived_metrics()
            self._logger.info(
                "asset_revenue_swept",
                total_usd=str(swept_total),
                liquid_balance=str(self._state.liquid_balance),
            )

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

        return result

    # ─── Phase 16e: Mitosis (Child Fleet Management) ────────────

    @property
    def mitosis(self) -> MitosisEngine:
        """Access the MitosisEngine for reproductive evaluation."""
        return self._mitosis

    def register_child(self, child: ChildPosition) -> None:
        """
        Register a newly spawned child in the economic state.

        Called by SpawnChildExecutor after successful seed transfer.
        Debits liquid_balance for the seed capital (funds leave the hot wallet).
        """
        self._state.child_instances.append(child)

        # Seed capital leaves the parent's liquid balance
        self._state.liquid_balance -= child.seed_capital_usd
        self._recompute_fleet_equity()
        self._recalculate_derived_metrics()

        self._logger.info(
            "child_registered",
            child_id=child.instance_id,
            niche=child.niche,
            seed=str(child.seed_capital_usd),
            liquid_balance=str(self._state.liquid_balance),
        )

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
        self._state.revenue_24h += record.amount_usd
        self._state.revenue_7d += record.amount_usd
        self._state.revenue_30d += record.amount_usd
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
        self._state.revenue_24h += amount_usd
        self._state.revenue_7d += amount_usd
        self._state.revenue_30d += amount_usd
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

        bounty = ActiveBounty(
            platform=platform,
            reward_usd=reward_usd,
            estimated_cost_usd=Decimal("0"),
            issue_url=bounty_url,
            status=BountyStatus.AVAILABLE,
        )
        self.register_bounty(bounty)

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
                    from systems.synapse.types import SynapseEvent, SynapseEventType

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
                if new_status != child.status:
                    old_status = child.status
                    child.status = new_status
                    self._logger.info(
                        "child_status_changed",
                        child_id=child_id,
                        old=old_status.value,
                        new=new_status.value,
                    )
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
        self._state.revenue_24h += sale.price_usd
        self._state.revenue_7d += sale.price_usd
        self._state.revenue_30d += sale.price_usd
        self._credit_revenue_source(RevenueStream.KNOWLEDGE_SALE, sale.price_usd)
        self._recalculate_derived_metrics()

        self._logger.info(
            "knowledge_sale_recorded",
            sale_id=sale.sale_id,
            client_id=sale.client_id,
            product=sale.product_type.value,
            price_usd=str(sale.price_usd),
        )

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
        self._state.revenue_24h += amount_usd
        self._state.revenue_7d += amount_usd
        self._state.revenue_30d += amount_usd
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

        from systems.identity.certificate import CertificateStatus

        status = await self._certificate_manager.check_expiry()
        if status is None:
            return

        remaining = self._certificate_manager.certificate_remaining_days

        if status == CertificateStatus.EXPIRING_SOON and self._event_bus is not None:
            # Trigger OBLIGATIONS-priority renewal intent via Synapse
            from systems.synapse.types import SynapseEvent, SynapseEventType

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

        elif status == CertificateStatus.EXPIRED:
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

        # Emit economic free energy as an interoceptive percept for Soma/Nova
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

        # Persist reputation, interspecies, and protocol state
        await self._reputation.persist_state()
        await self._interspecies.persist_state()
        await self._protocol_factory.persist_state()

        self._logger.info("consolidation_cycle_complete", **{
            k: str(v) if isinstance(v, Decimal) else v
            for k, v in results.items()
        })

        return results

    # ── Phase 16i: Economic Dreaming Integration ────────────────

    def integrate_dream_result(self, result: Any) -> None:
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
                self._auto_apply_recommendation(rec)

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
        from systems.synapse.types import SynapseEvent, SynapseEventType

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

    def _auto_apply_recommendation(self, rec: Any) -> None:
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
                self._state.liquid_balance -= transfer
                self._state.survival_reserve += transfer
                self._state.survival_reserve_deficit = max(
                    Decimal("0"),
                    self._state.survival_reserve_target - self._state.survival_reserve,
                )
                applied = True
                self._logger.info(
                    "dream_applied_reserve_funding",
                    transferred=str(transfer),
                    new_reserve=str(self._state.survival_reserve),
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

        # Recompute runway with current liquid balance
        if s.basal_metabolic_rate.usd_per_hour > Decimal("0"):
            hours, days = self._cost_model.compute_runway(
                liquid_balance=s.liquid_balance,
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

        # Inject into rolling cost windows
        self._state.costs_24h += amount
        self._state.costs_7d += amount
        self._state.costs_30d += amount
        self._total_costs_usd += amount

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
