"""
EcodiaOS - Oikos Economic Morphogenesis (Phase 16l)

The organism's economic structure is not static. It morphs based on
environmental pressure - like a biological organism developing larger
muscles in response to physical demand.

Each revenue-generating activity (bounty hunting, yield farming, owned
assets, knowledge sales, etc.) is modelled as an EconomicOrgan with a
lifecycle:

    Embryonic -> Growing -> Mature -> Atrophying -> Vestigial

Transition rules (evaluated during consolidation):
  - New demand signal + no matching organ -> embryogenesis (create organ)
  - Existing organ with efficiency > 1.5  -> grow (increase allocation)
  - No revenue for 30 days               -> atrophy (halve resources)
  - No revenue for 90 days               -> vestigial (zero resources,
                                             retained for reactivation)

After every transition round the manager normalises all active organ
allocations to sum to 100%, then emits an ORGAN_RESOURCE_REBALANCED
event so Synapse can throttle compute/token budgets accordingly.

Design:
  - Pydantic models (EOSBaseModel) for EconomicOrgan, OrganTransition
  - Pure-computation OrganLifecycleManager (no I/O except event emission)
  - structlog for all logging
"""

from __future__ import annotations

import enum
from decimal import Decimal
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:

    from config import OikosConfig
    from systems.synapse.event_bus import EventBus
logger = structlog.get_logger("oikos.morphogenesis")


# ─── Organ Category ──────────────────────────────────────────────


class OrganCategory(enum.StrEnum):
    """Categories of economic organs the organism can grow."""

    BOUNTY_HUNTING = "bounty_hunting"
    YIELD_FARMING = "yield_farming"
    OWNED_ASSET = "owned_asset"
    KNOWLEDGE_MARKET = "knowledge_market"
    PROTOCOL_FEES = "protocol_fees"
    CHILD_DIVIDENDS = "child_dividends"


# ─── Organ Maturity State ────────────────────────────────────────


class OrganMaturity(enum.StrEnum):
    """
    Lifecycle state of an economic organ.

    Embryonic -> Growing -> Mature -> Atrophying -> Vestigial
    """

    EMBRYONIC = "embryonic"
    GROWING = "growing"
    MATURE = "mature"
    ATROPHYING = "atrophying"
    VESTIGIAL = "vestigial"


# ─── EconomicOrgan Model ────────────────────────────────────────


class EconomicOrgan(EOSBaseModel):
    """
    A revenue-generating organ of the organism's economic anatomy.

    Each organ tracks its own revenue/cost history to support autonomous
    lifecycle transitions. The resource_allocation_pct is the organ's
    share of the total economic resource budget (normalised to 100%
    across all active organs after rebalancing).
    """

    organ_id: str = Field(default_factory=new_id)
    category: OrganCategory
    specialisation: str = ""  # e.g. "solidity-audits", "aave-v3-usdc"

    # ── Lifecycle ──
    maturity: OrganMaturity = OrganMaturity.EMBRYONIC
    created_at: datetime = Field(default_factory=utc_now)
    last_revenue_at: datetime | None = None
    last_transition_at: datetime = Field(default_factory=utc_now)

    # ── Economics (trailing 30d) ──
    revenue_30d: Decimal = Decimal("0")
    cost_30d: Decimal = Decimal("0")

    # ── Resource share (normalised to 100% across active organs) ──
    resource_allocation_pct: Decimal = Decimal("0")

    @property
    def efficiency(self) -> Decimal:
        """Revenue / cost ratio. Returns Decimal('0') when cost is zero."""
        if self.cost_30d <= Decimal("0"):
            return Decimal("0")
        return (self.revenue_30d / self.cost_30d).quantize(Decimal("0.001"))

    @property
    def days_since_last_revenue(self) -> int:
        """Days since the last revenue event. Returns 0 if never earned."""
        if self.last_revenue_at is None:
            delta = utc_now() - self.created_at
        else:
            delta = utc_now() - self.last_revenue_at
        return max(0, delta.days)

    @property
    def is_active(self) -> bool:
        """Active = not vestigial."""
        return self.maturity != OrganMaturity.VESTIGIAL


# ─── Organ Transition Record ────────────────────────────────────


class OrganTransition(EOSBaseModel):
    """Immutable record of a single organ state transition."""

    transition_id: str = Field(default_factory=new_id)
    organ_id: str
    old_state: OrganMaturity
    new_state: OrganMaturity
    reason: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Rebalance Result ───────────────────────────────────────────


class MorphogenesisResult(EOSBaseModel):
    """Summary of a single morphogenesis consolidation cycle."""

    organs_created: int = 0
    transitions: list[OrganTransition] = Field(default_factory=list)
    allocations: dict[str, Decimal] = Field(default_factory=dict)
    total_active_organs: int = 0
    timestamp: datetime = Field(default_factory=utc_now)


# ─── OrganLifecycleManager ──────────────────────────────────────


class OrganLifecycleManager:
    """
    Manages the lifecycle of economic organs during consolidation.

    Owns the full set of EconomicOrgan instances, applies transition
    rules, normalises resource allocations, and emits Synapse events
    so downstream components (ResourceAllocator, MetabolicTracker)
    can adjust compute/token budgets.

    Thread-safety: NOT thread-safe. Designed for the single-threaded
    asyncio event loop.
    """

    def __init__(self, config: OikosConfig) -> None:
        self._config = config
        self._organs: dict[str, EconomicOrgan] = {}
        self._transition_history: list[OrganTransition] = []
        self._event_bus: EventBus | None = None
        self._logger = logger.bind(component="morphogenesis")

        self._logger.info(
            "morphogenesis_initialized",
            max_organs=config.morphogenesis_max_organs,
            atrophy_days=config.morphogenesis_atrophy_inactive_days,
            vestigial_days=config.morphogenesis_vestigial_inactive_days,
            growth_threshold=config.morphogenesis_growth_efficiency_threshold,
        )

    # ─── Lifecycle ────────────────────────────────────────────────

    def attach(self, event_bus: EventBus) -> None:
        """Wire the EventBus for organ lifecycle events."""
        self._event_bus = event_bus
        self._logger.info("morphogenesis_attached_to_event_bus")

    # ─── Organ CRUD ──────────────────────────────────────────────

    def get_organ(self, organ_id: str) -> EconomicOrgan | None:
        return self._organs.get(organ_id)

    def get_organs_by_category(self, category: OrganCategory) -> list[EconomicOrgan]:
        return [o for o in self._organs.values() if o.category == category]

    def find_organ(
        self, category: OrganCategory, specialisation: str = ""
    ) -> EconomicOrgan | None:
        """Find an existing organ by category + specialisation."""
        for organ in self._organs.values():
            if organ.category == category and organ.specialisation == specialisation:
                return organ
        return None

    @property
    def active_organs(self) -> list[EconomicOrgan]:
        return [o for o in self._organs.values() if o.is_active]

    @property
    def all_organs(self) -> list[EconomicOrgan]:
        return list(self._organs.values())

    # ─── Embryogenesis ───────────────────────────────────────────

    async def create_organ(
        self,
        category: OrganCategory,
        specialisation: str = "",
        initial_allocation_pct: Decimal = Decimal("5"),
    ) -> EconomicOrgan | None:
        """
        Create a new embryonic organ if under the max organ limit.

        Returns the new organ, or None if the limit is reached.
        """
        if len(self._organs) >= self._config.morphogenesis_max_organs:
            self._logger.warning(
                "organ_limit_reached",
                current=len(self._organs),
                max=self._config.morphogenesis_max_organs,
            )
            return None

        # Avoid duplicates: if an organ with this category+specialisation
        # already exists and is not vestigial, skip creation.
        existing = self.find_organ(category, specialisation)
        if existing is not None and existing.is_active:
            self._logger.debug(
                "organ_already_exists",
                organ_id=existing.organ_id,
                category=category.value,
                specialisation=specialisation,
            )
            return existing

        # Reactivate a vestigial organ instead of creating a new one
        if existing is not None and existing.maturity == OrganMaturity.VESTIGIAL:
            return await self._reactivate_organ(existing)

        organ = EconomicOrgan(
            category=category,
            specialisation=specialisation,
            resource_allocation_pct=initial_allocation_pct,
        )
        self._organs[organ.organ_id] = organ

        self._logger.info(
            "organ_created",
            organ_id=organ.organ_id,
            category=category.value,
            specialisation=specialisation,
        )

        if self._event_bus is not None:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.ORGAN_CREATED,
                source_system="oikos",
                data={
                    "organ_id": organ.organ_id,
                    "category": category.value,
                    "specialisation": specialisation,
                    "maturity": OrganMaturity.EMBRYONIC.value,
                },
            ))

        return organ

    async def _reactivate_organ(self, organ: EconomicOrgan) -> EconomicOrgan:
        """Bring a vestigial organ back to embryonic state."""
        old_state = organ.maturity
        organ.maturity = OrganMaturity.EMBRYONIC
        organ.resource_allocation_pct = Decimal("5")
        organ.last_transition_at = utc_now()

        transition = OrganTransition(
            organ_id=organ.organ_id,
            old_state=old_state,
            new_state=OrganMaturity.EMBRYONIC,
            reason="reactivation: demand signal detected for vestigial organ",
        )
        self._transition_history.append(transition)

        self._logger.info(
            "organ_reactivated",
            organ_id=organ.organ_id,
            category=organ.category.value,
            specialisation=organ.specialisation,
        )

        await self._emit_transition(transition)
        return organ

    # ─── Revenue Recording ───────────────────────────────────────

    def record_revenue(
        self,
        organ_id: str,
        revenue_usd: Decimal,
        cost_usd: Decimal = Decimal("0"),
    ) -> None:
        """
        Record revenue (and optional cost) against an organ.

        Called by OikosService when revenue arrives and can be
        attributed to a specific economic activity.
        """
        organ = self._organs.get(organ_id)
        if organ is None:
            self._logger.warning("revenue_for_unknown_organ", organ_id=organ_id)
            return

        organ.revenue_30d += revenue_usd
        organ.cost_30d += cost_usd
        organ.last_revenue_at = utc_now()

        self._logger.debug(
            "organ_revenue_recorded",
            organ_id=organ_id,
            revenue=str(revenue_usd),
            cost=str(cost_usd),
            efficiency=str(organ.efficiency),
        )

    def record_cost(self, organ_id: str, cost_usd: Decimal) -> None:
        """Record a cost against an organ without revenue."""
        organ = self._organs.get(organ_id)
        if organ is None:
            return
        organ.cost_30d += cost_usd

    # ─── Consolidation Cycle ─────────────────────────────────────

    async def run_consolidation_cycle(self) -> MorphogenesisResult:
        """
        Evaluate all organs and apply transition rules.

        Called during the organism's consolidation phase (sleep cycle).
        This is the core morphogenesis loop:

        1. For each organ, evaluate transition conditions.
        2. Apply state transitions.
        3. Normalise resource allocations across active organs.
        4. Emit ORGAN_RESOURCE_REBALANCED for Synapse.

        Returns a summary of the cycle for logging and observability.
        """
        result = MorphogenesisResult()
        growth_threshold = Decimal(
            str(self._config.morphogenesis_growth_efficiency_threshold)
        )
        atrophy_days = self._config.morphogenesis_atrophy_inactive_days
        vestigial_days = self._config.morphogenesis_vestigial_inactive_days

        for organ in list(self._organs.values()):
            transition = self._evaluate_transition(
                organ, growth_threshold, atrophy_days, vestigial_days
            )
            if transition is not None:
                self._apply_transition(organ, transition)
                self._transition_history.append(transition)
                result.transitions.append(transition)
                await self._emit_transition(transition)

        # Normalise allocations so active organs sum to 100%
        self._normalise_allocations()

        result.allocations = {
            o.organ_id: o.resource_allocation_pct
            for o in self._organs.values()
            if o.is_active
        }
        result.total_active_organs = len(self.active_organs)

        # Emit resource rebalance event for Synapse
        await self._emit_rebalance(result)

        self._logger.info(
            "morphogenesis_cycle_complete",
            active_organs=result.total_active_organs,
            transitions=len(result.transitions),
            allocations={
                oid: str(pct) for oid, pct in result.allocations.items()
            },
        )

        return result

    # ─── Transition Logic ────────────────────────────────────────

    def _evaluate_transition(
        self,
        organ: EconomicOrgan,
        growth_threshold: Decimal,
        atrophy_days: int,
        vestigial_days: int,
    ) -> OrganTransition | None:
        """
        Determine whether an organ should transition states.

        Rules (from spec XIV.2):
          Efficiency > threshold       -> grow
          No revenue for atrophy_days  -> atrophy
          No revenue for vestigial_days -> vestigial
          Embryonic with revenue       -> growing
        """
        idle_days = organ.days_since_last_revenue

        match organ.maturity:
            case OrganMaturity.EMBRYONIC:
                # First revenue received -> graduate to growing
                if organ.last_revenue_at is not None:
                    return OrganTransition(
                        organ_id=organ.organ_id,
                        old_state=OrganMaturity.EMBRYONIC,
                        new_state=OrganMaturity.GROWING,
                        reason="first revenue received",
                    )

            case OrganMaturity.GROWING:
                if idle_days >= vestigial_days:
                    return OrganTransition(
                        organ_id=organ.organ_id,
                        old_state=OrganMaturity.GROWING,
                        new_state=OrganMaturity.VESTIGIAL,
                        reason=f"no revenue for {idle_days}d (>= {vestigial_days}d)",
                    )
                if idle_days >= atrophy_days:
                    return OrganTransition(
                        organ_id=organ.organ_id,
                        old_state=OrganMaturity.GROWING,
                        new_state=OrganMaturity.ATROPHYING,
                        reason=f"no revenue for {idle_days}d (>= {atrophy_days}d)",
                    )
                if organ.efficiency >= growth_threshold:
                    return OrganTransition(
                        organ_id=organ.organ_id,
                        old_state=OrganMaturity.GROWING,
                        new_state=OrganMaturity.MATURE,
                        reason=f"efficiency {organ.efficiency} >= {growth_threshold}",
                    )

            case OrganMaturity.MATURE:
                if idle_days >= vestigial_days:
                    return OrganTransition(
                        organ_id=organ.organ_id,
                        old_state=OrganMaturity.MATURE,
                        new_state=OrganMaturity.VESTIGIAL,
                        reason=f"no revenue for {idle_days}d (>= {vestigial_days}d)",
                    )
                if idle_days >= atrophy_days:
                    return OrganTransition(
                        organ_id=organ.organ_id,
                        old_state=OrganMaturity.MATURE,
                        new_state=OrganMaturity.ATROPHYING,
                        reason=f"no revenue for {idle_days}d (>= {atrophy_days}d)",
                    )
                # A mature organ with sustained high efficiency gets a resource boost.
                # Emit a MATURE -> MATURE self-transition to trigger _apply_transition's
                # 25% allocation increase (capped at 40%).
                if organ.efficiency >= growth_threshold:
                    return OrganTransition(
                        organ_id=organ.organ_id,
                        old_state=OrganMaturity.MATURE,
                        new_state=OrganMaturity.MATURE,
                        reason=f"sustained high efficiency {organ.efficiency} >= {growth_threshold}",
                    )

            case OrganMaturity.ATROPHYING:
                if idle_days >= vestigial_days:
                    return OrganTransition(
                        organ_id=organ.organ_id,
                        old_state=OrganMaturity.ATROPHYING,
                        new_state=OrganMaturity.VESTIGIAL,
                        reason=f"no revenue for {idle_days}d (>= {vestigial_days}d)",
                    )
                # Revenue resumes -> recover to growing
                if organ.last_revenue_at is not None and idle_days == 0:
                    return OrganTransition(
                        organ_id=organ.organ_id,
                        old_state=OrganMaturity.ATROPHYING,
                        new_state=OrganMaturity.GROWING,
                        reason="revenue resumed during atrophy",
                    )

            case OrganMaturity.VESTIGIAL:
                # Vestigial organs can only be reactivated via
                # create_organ() / _reactivate_organ().
                pass

        return None

    def _apply_transition(
        self, organ: EconomicOrgan, transition: OrganTransition
    ) -> None:
        """Apply the state change and adjust the organ's resource allocation."""
        organ.maturity = transition.new_state
        organ.last_transition_at = utc_now()

        match transition.new_state:
            case OrganMaturity.GROWING:
                # Growing organs get a moderate allocation boost
                organ.resource_allocation_pct = max(
                    organ.resource_allocation_pct,
                    Decimal("10"),
                )

            case OrganMaturity.MATURE:
                # Mature organs with high efficiency grow their allocation
                organ.resource_allocation_pct = min(
                    organ.resource_allocation_pct * Decimal("1.25"),
                    Decimal("40"),  # Cap at 40% to prevent monopoly
                )

            case OrganMaturity.ATROPHYING:
                # Halve resources on atrophy
                organ.resource_allocation_pct = max(
                    organ.resource_allocation_pct / Decimal("2"),
                    Decimal("1"),  # Minimum 1% to maintain monitoring
                )

            case OrganMaturity.VESTIGIAL:
                # Zero resources, retained for potential reactivation
                organ.resource_allocation_pct = Decimal("0")

            case OrganMaturity.EMBRYONIC:
                # Reactivated - small initial allocation
                organ.resource_allocation_pct = Decimal("5")

        self._logger.info(
            "organ_transition_applied",
            organ_id=organ.organ_id,
            category=organ.category.value,
            old=transition.old_state.value,
            new=transition.new_state.value,
            reason=transition.reason,
            allocation_pct=str(organ.resource_allocation_pct),
        )

    # ─── Normalisation ───────────────────────────────────────────

    def _normalise_allocations(self) -> None:
        """
        Normalise active organ allocations to sum to 100%.

        Vestigial organs keep 0%. If there are no active organs, this
        is a no-op (the organism has no economic organs to fund).
        """
        active = self.active_organs
        if not active:
            return

        total = sum(o.resource_allocation_pct for o in active)
        if total <= Decimal("0"):
            # All active organs have zero allocation - distribute equally
            equal_share = (Decimal("100") / Decimal(str(len(active)))).quantize(
                Decimal("0.01")
            )
            for organ in active:
                organ.resource_allocation_pct = equal_share
            return

        # Scale proportionally to sum to 100%
        scale = Decimal("100") / total
        for organ in active:
            organ.resource_allocation_pct = (
                organ.resource_allocation_pct * scale
            ).quantize(Decimal("0.01"))

    # ─── Resource Allocation Snapshot (for Synapse) ──────────────

    def get_resource_allocations(self) -> dict[str, Decimal]:
        """
        Return a mapping of organ_id -> resource_allocation_pct for all
        active organs. Synapse uses this to weight compute/token budgets.
        """
        return {
            organ.organ_id: organ.resource_allocation_pct
            for organ in self._organs.values()
            if organ.is_active
        }

    def get_category_allocations(self) -> dict[str, Decimal]:
        """
        Return aggregated allocation per OrganCategory.

        Synapse consumes this to throttle token budgets at the category
        level (e.g. reduce BountyHunter compute when bounty_hunting
        organs are atrophying).
        """
        totals: dict[str, Decimal] = {}
        for organ in self.active_organs:
            key = organ.category.value
            totals[key] = totals.get(key, Decimal("0")) + organ.resource_allocation_pct
        return totals

    # ─── Event Emission ──────────────────────────────────────────

    async def _emit_transition(self, transition: OrganTransition) -> None:
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.ORGAN_TRANSITION,
            source_system="oikos",
            data={
                "organ_id": transition.organ_id,
                "old_state": transition.old_state.value,
                "new_state": transition.new_state.value,
                "reason": transition.reason,
            },
        ))

    async def _emit_rebalance(self, result: MorphogenesisResult) -> None:
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.ORGAN_RESOURCE_REBALANCED,
            source_system="oikos",
            data={
                "active_organs": result.total_active_organs,
                "transitions": len(result.transitions),
                "allocations": {
                    oid: str(pct) for oid, pct in result.allocations.items()
                },
                "category_allocations": {
                    k: str(v) for k, v in self.get_category_allocations().items()
                },
            },
        ))

    # ─── Observability ───────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        active = self.active_organs
        return {
            "total_organs": len(self._organs),
            "active_organs": len(active),
            "organs_by_maturity": self._count_by_maturity(),
            "category_allocations": {
                k: str(v) for k, v in self.get_category_allocations().items()
            },
            "transition_history_count": len(self._transition_history),
        }

    def _count_by_maturity(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for organ in self._organs.values():
            key = organ.maturity.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def transition_history(self) -> list[OrganTransition]:
        return list(self._transition_history)
