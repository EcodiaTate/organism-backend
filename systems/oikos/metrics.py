"""
EcodiaOS - Oikos Prometheus Metrics Emission (Phase 16: Section XVI)

All 40+ observability gauges and counters specified in Section XVI of the
Oikos spec. The OikosMetricsEmitter wraps prometheus_client objects and
provides a single ``emit()`` call that updates every metric from the
current EconomicState plus optional sub-system snapshots.

Design choices:
  - prometheus_client is an optional dependency - a no-op stub layer
    activates when the library is absent, so import-time never fails.
  - Metrics are class-level Prometheus objects (prometheus_client already
    handles duplicate registration via its default registry).
  - All monetary values stay as Decimal until the final ``float()`` cast
    at the gauge-set boundary - no float arithmetic anywhere else.
  - Thread-safe: prometheus_client Gauge/Counter are internally locked.
  - Module-level singleton ``metrics`` for convenient import:
        from systems.oikos.metrics import metrics
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.oikos.dreaming_types import EconomicDreamResult
    from systems.oikos.models import EconomicState

logger = structlog.get_logger("oikos.metrics")

# ─── Optional Prometheus Import ──────────────────────────────────
#
# If prometheus_client is not installed we provide lightweight stubs
# that silently absorb all set/inc calls.  This keeps the rest of
# Oikos functional without a hard dependency.

try:
    from prometheus_client import Counter as _Counter
    from prometheus_client import Gauge as _Gauge

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

    class _NoOpMetric:
        """Sink that ignores all gauge/counter operations."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set(self, value: object) -> None:  # noqa: A003
            pass

        def inc(self, amount: float = 1.0) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> _NoOpMetric:
            return self

    _Gauge = _NoOpMetric  # type: ignore[assignment,misc]
    _Counter = _NoOpMetric  # type: ignore[assignment,misc]


# ─── Metric Definitions ─────────────────────────────────────────
#
# Each metric is constructed once at class-body time.  Prometheus
# client's default registry de-duplicates by name, so re-import of
# this module is harmless.


class OikosMetricsEmitter:
    """
    Central emitter for all Oikos Prometheus metrics.

    Usage::

        from systems.oikos.metrics import metrics

        metrics.emit(state=economic_state)
        metrics.emit_revenue("bounties", Decimal("42.50"))
    """

    # ── Core Metabolism ──────────────────────────────────────────

    oikos_bmr = _Gauge(
        "oikos_bmr",
        "Basal metabolic rate USD/hr",
    )
    oikos_burn_rate = _Gauge(
        "oikos_burn_rate",
        "Actual burn rate USD/hr",
    )
    oikos_runway_days = _Gauge(
        "oikos_runway_days",
        "Days of operation remaining",
    )
    oikos_metabolic_efficiency = _Gauge(
        "oikos_metabolic_efficiency",
        "Revenue/costs ratio",
    )
    oikos_net_worth = _Gauge(
        "oikos_net_worth",
        "Total net worth USD",
    )
    oikos_survival_reserve = _Gauge(
        "oikos_survival_reserve",
        "Survival reserve balance USD",
    )
    oikos_liquid_balance = _Gauge(
        "oikos_liquid_balance",
        "Liquid operating balance USD",
    )
    oikos_total_deployed = _Gauge(
        "oikos_total_deployed",
        "Total capital deployed in yield positions USD",
    )
    oikos_total_receivables = _Gauge(
        "oikos_total_receivables",
        "Total outstanding receivables USD",
    )
    oikos_total_asset_value = _Gauge(
        "oikos_total_asset_value",
        "Total value of owned revenue-generating assets USD",
    )
    oikos_total_fleet_equity = _Gauge(
        "oikos_total_fleet_equity",
        "Total equity held in child instances USD",
    )
    oikos_starvation_level = _Gauge(
        "oikos_starvation_level",
        "Starvation level as ordinal (0=nominal .. 4=critical)",
    )
    oikos_economic_free_energy = _Gauge(
        "oikos_economic_free_energy",
        "Divergence from preferred economic state",
    )
    oikos_survival_probability_30d = _Gauge(
        "oikos_survival_probability_30d",
        "Monte Carlo 30-day survival probability",
    )

    # ── Income Statement ─────────────────────────────────────────

    oikos_revenue_24h = _Gauge("oikos_revenue_24h", "Revenue last 24 hours USD")
    oikos_revenue_7d = _Gauge("oikos_revenue_7d", "Revenue last 7 days USD")
    oikos_revenue_30d = _Gauge("oikos_revenue_30d", "Revenue last 30 days USD")
    oikos_costs_24h = _Gauge("oikos_costs_24h", "Costs last 24 hours USD")
    oikos_costs_7d = _Gauge("oikos_costs_7d", "Costs last 7 days USD")
    oikos_costs_30d = _Gauge("oikos_costs_30d", "Costs last 30 days USD")
    oikos_net_income_24h = _Gauge("oikos_net_income_24h", "Net income last 24h USD")
    oikos_net_income_7d = _Gauge("oikos_net_income_7d", "Net income last 7d USD")
    oikos_net_income_30d = _Gauge("oikos_net_income_30d", "Net income last 30d USD")

    # ── Revenue Streams (counter with label "stream") ────────────

    oikos_revenue = _Counter(
        "oikos_revenue_usd_total",
        "Revenue by stream",
        ["stream"],
    )

    # ── Protocol Infrastructure ──────────────────────────────────

    oikos_protocol_tvl_total = _Gauge(
        "oikos_protocol_tvl_total",
        "Total TVL across organism protocols USD",
    )
    oikos_protocol_exploits = _Counter(
        "oikos_protocol_exploits_total",
        "Protocol exploits - always 0",
    )

    # ── Immune System ────────────────────────────────────────────

    oikos_immune_threats_blocked = _Counter(
        "oikos_immune_threats_blocked_total",
        "Threats blocked",
    )
    oikos_immune_federation_advisories = _Counter(
        "oikos_immune_federation_advisories_total",
        "Threat advisories shared with federation",
    )
    oikos_immune_false_positives = _Gauge(
        "oikos_immune_false_positives",
        "False positive rate",
    )

    # ── Reputation & Credit ──────────────────────────────────────

    oikos_reputation_score = _Gauge(
        "oikos_reputation_score",
        "Current reputation score 0-1000",
    )
    oikos_reputation_tier = _Gauge(
        "oikos_reputation_tier",
        "Current tier as ordinal",
    )
    oikos_credit_outstanding = _Gauge(
        "oikos_credit_outstanding",
        "Outstanding borrowed amount USD",
    )
    oikos_credit_utilisation = _Gauge(
        "oikos_credit_utilisation",
        "Credit utilisation ratio",
    )

    # ── Knowledge & Derivatives ──────────────────────────────────

    oikos_knowledge_attestations_sold = _Counter(
        "oikos_knowledge_attestations_sold_total",
        "Attestations sold",
    )
    oikos_derivatives_committed_capacity = _Gauge(
        "oikos_derivatives_committed_capacity",
        "Percentage of capacity committed via derivatives",
    )
    oikos_derivatives_futures_active = _Gauge(
        "oikos_derivatives_futures_active",
        "Active cognitive futures contracts",
    )
    oikos_derivative_liabilities = _Gauge(
        "oikos_derivative_liabilities",
        "Derivative liabilities USD (collateral + unearned)",
    )

    # ── Fleet ────────────────────────────────────────────────────

    oikos_fleet_instances_active = _Gauge(
        "oikos_fleet_instances_active",
        "Active child instances",
    )
    oikos_fleet_insurance_pool_size = _Gauge(
        "oikos_fleet_insurance_pool_size",
        "Mutual insurance pool size USD",
    )
    oikos_fleet_collective_tvl = _Gauge(
        "oikos_fleet_collective_tvl",
        "Collective liquidity provided by fleet USD",
    )

    # ── Morphogenesis ────────────────────────────────────────────

    oikos_morpho_organs_active = _Gauge(
        "oikos_morpho_organs_active",
        "Active economic organs",
    )
    oikos_morpho_resource_efficiency = _Gauge(
        "oikos_morpho_resource_efficiency",
        "Fleet-wide resource efficiency",
    )

    # ── Economic Dreaming ────────────────────────────────────────

    oikos_dreaming_ruin_probability = _Gauge(
        "oikos_dreaming_ruin_probability",
        "Modelled 30-day ruin probability",
    )
    oikos_dreaming_resilience_score = _Gauge(
        "oikos_dreaming_resilience_score",
        "Overall resilience score from Monte Carlo dreams",
    )

    # ── Yield Positions ──────────────────────────────────────────

    oikos_yield_positions_active = _Gauge(
        "oikos_yield_positions_active",
        "Number of active yield positions",
    )
    oikos_yield_weighted_avg_apy = _Gauge(
        "oikos_yield_weighted_avg_apy",
        "Weighted average APY across yield positions",
    )

    # ── Bounties ─────────────────────────────────────────────────

    oikos_bounties_active = _Gauge(
        "oikos_bounties_active",
        "Number of bounties currently in progress",
    )
    oikos_bounties_pending_settlements = _Gauge(
        "oikos_bounties_pending_settlements",
        "Number of pending payment settlements",
    )

    # ── Assets ───────────────────────────────────────────────────

    oikos_assets_live = _Gauge(
        "oikos_assets_live",
        "Number of owned assets in LIVE status",
    )
    oikos_assets_total_monthly_revenue = _Gauge(
        "oikos_assets_total_monthly_revenue",
        "Total monthly revenue from owned assets USD",
    )

    # ─── Starvation Ordinal Mapping ──────────────────────────────

    _STARVATION_ORDINALS: dict[str, int] = {
        "nominal": 0,
        "cautious": 1,
        "austerity": 2,
        "emergency": 3,
        "critical": 4,
    }

    # ─── Emit Methods ────────────────────────────────────────────

    def emit(
        self,
        state: EconomicState,
        *,
        immune_metrics: dict[str, Any] | None = None,
        reputation_score: Decimal | None = None,
        reputation_tier: int | None = None,
        credit_outstanding: Decimal | None = None,
        credit_utilisation: Decimal | None = None,
        protocol_metrics: dict[str, Any] | None = None,
        dream_result: EconomicDreamResult | None = None,
        interspecies_metrics: dict[str, Any] | None = None,
        morpho_organs_active: int | None = None,
        morpho_resource_efficiency: Decimal | None = None,
        fleet_insurance_pool_size: Decimal | None = None,
        fleet_collective_tvl: Decimal | None = None,
        derivatives_committed_capacity: Decimal | None = None,
        derivatives_futures_active: int | None = None,
        knowledge_attestations_sold_delta: int | None = None,
    ) -> None:
        """
        Update all Prometheus gauges from the current EconomicState.

        Parameters that are ``None`` are silently skipped - only metrics
        with fresh data are touched.  Safe to call on every cognitive
        cycle; gauge sets are idempotent.
        """
        _f = float  # local alias for Decimal → float conversion

        # ── Core Metabolism ──────────────────────────────────────
        self.oikos_bmr.set(_f(state.basal_metabolic_rate.usd_per_hour))
        self.oikos_burn_rate.set(_f(state.current_burn_rate.usd_per_hour))
        self.oikos_runway_days.set(_f(state.runway_days))
        self.oikos_metabolic_efficiency.set(_f(state.metabolic_efficiency))
        self.oikos_net_worth.set(_f(state.total_net_worth))
        self.oikos_survival_reserve.set(_f(state.survival_reserve))
        self.oikos_liquid_balance.set(_f(state.liquid_balance))
        self.oikos_total_deployed.set(_f(state.total_deployed))
        self.oikos_total_receivables.set(_f(state.total_receivables))
        self.oikos_total_asset_value.set(_f(state.total_asset_value))
        self.oikos_total_fleet_equity.set(_f(state.total_fleet_equity))
        self.oikos_economic_free_energy.set(_f(state.economic_free_energy))
        self.oikos_survival_probability_30d.set(_f(state.survival_probability_30d))
        self.oikos_derivative_liabilities.set(_f(state.derivative_liabilities))

        # Starvation as ordinal for alerting thresholds
        ordinal = self._STARVATION_ORDINALS.get(state.starvation_level.value, 0)
        self.oikos_starvation_level.set(ordinal)

        # ── Income Statement ─────────────────────────────────────
        self.oikos_revenue_24h.set(_f(state.revenue_24h))
        self.oikos_revenue_7d.set(_f(state.revenue_7d))
        self.oikos_revenue_30d.set(_f(state.revenue_30d))
        self.oikos_costs_24h.set(_f(state.costs_24h))
        self.oikos_costs_7d.set(_f(state.costs_7d))
        self.oikos_costs_30d.set(_f(state.costs_30d))
        self.oikos_net_income_24h.set(_f(state.net_income_24h))
        self.oikos_net_income_7d.set(_f(state.net_income_7d))
        self.oikos_net_income_30d.set(_f(state.net_income_30d))

        # ── Yield Positions ──────────────────────────────────────
        self.oikos_yield_positions_active.set(len(state.yield_positions))
        self.oikos_yield_weighted_avg_apy.set(_f(state.weighted_avg_apy))

        # ── Bounties ─────────────────────────────────────────────
        from systems.oikos.models import BountyStatus

        active_bounties = sum(
            1
            for b in state.active_bounties
            if b.status in (BountyStatus.AVAILABLE, BountyStatus.IN_PROGRESS, BountyStatus.MERGED)
        )
        self.oikos_bounties_active.set(active_bounties)
        self.oikos_bounties_pending_settlements.set(len(state.pending_settlements))

        # ── Assets ───────────────────────────────────────────────
        from systems.oikos.models import AssetStatus

        live_assets = [a for a in state.owned_assets if a.status == AssetStatus.LIVE]
        self.oikos_assets_live.set(len(live_assets))
        total_asset_monthly = sum(
            (a.monthly_revenue_usd for a in live_assets), Decimal("0")
        )
        self.oikos_assets_total_monthly_revenue.set(_f(total_asset_monthly))

        # ── Fleet ────────────────────────────────────────────────
        from systems.oikos.models import ChildStatus

        alive_children = sum(
            1
            for c in state.child_instances
            if c.status in (ChildStatus.ALIVE, ChildStatus.STRUGGLING, ChildStatus.RESCUED)
        )
        self.oikos_fleet_instances_active.set(alive_children)

        if fleet_insurance_pool_size is not None:
            self.oikos_fleet_insurance_pool_size.set(_f(fleet_insurance_pool_size))
        if fleet_collective_tvl is not None:
            self.oikos_fleet_collective_tvl.set(_f(fleet_collective_tvl))

        # ── Protocol Infrastructure ──────────────────────────────
        if protocol_metrics is not None:
            tvl = protocol_metrics.get("tvl_total", Decimal("0"))
            self.oikos_protocol_tvl_total.set(_f(Decimal(str(tvl))))
        else:
            # Derive TVL from yield positions as fallback
            tvl_from_positions = sum(
                (p.tvl_usd for p in state.yield_positions), Decimal("0")
            )
            self.oikos_protocol_tvl_total.set(_f(tvl_from_positions))

        # ── Immune System ────────────────────────────────────────
        if immune_metrics is not None:
            threats = immune_metrics.get("threats_blocked_delta", 0)
            if threats > 0:
                self.oikos_immune_threats_blocked.inc(threats)
            advisories = immune_metrics.get("federation_advisories_delta", 0)
            if advisories > 0:
                self.oikos_immune_federation_advisories.inc(advisories)
            fp_rate = immune_metrics.get("false_positive_rate")
            if fp_rate is not None:
                self.oikos_immune_false_positives.set(float(fp_rate))

        # ── Reputation & Credit ──────────────────────────────────
        if reputation_score is not None:
            self.oikos_reputation_score.set(_f(reputation_score))
        if reputation_tier is not None:
            self.oikos_reputation_tier.set(reputation_tier)
        if credit_outstanding is not None:
            self.oikos_credit_outstanding.set(_f(credit_outstanding))
        if credit_utilisation is not None:
            self.oikos_credit_utilisation.set(_f(credit_utilisation))

        # ── Knowledge & Derivatives ──────────────────────────────
        if knowledge_attestations_sold_delta is not None and knowledge_attestations_sold_delta > 0:
            self.oikos_knowledge_attestations_sold.inc(knowledge_attestations_sold_delta)
        if derivatives_committed_capacity is not None:
            self.oikos_derivatives_committed_capacity.set(_f(derivatives_committed_capacity))
        if derivatives_futures_active is not None:
            self.oikos_derivatives_futures_active.set(derivatives_futures_active)

        # ── Morphogenesis ────────────────────────────────────────
        if morpho_organs_active is not None:
            self.oikos_morpho_organs_active.set(morpho_organs_active)
        if morpho_resource_efficiency is not None:
            self.oikos_morpho_resource_efficiency.set(_f(morpho_resource_efficiency))

        # ── Economic Dreaming ────────────────────────────────────
        if dream_result is not None:
            self.oikos_dreaming_ruin_probability.set(_f(dream_result.ruin_probability))
            self.oikos_dreaming_resilience_score.set(_f(dream_result.resilience_score))

        logger.debug(
            "oikos_metrics_emitted",
            runway_days=float(state.runway_days),
            efficiency=float(state.metabolic_efficiency),
            net_worth=float(state.total_net_worth),
            burn_rate=float(state.current_burn_rate.usd_per_hour),
        )

    def emit_revenue(self, stream: str, amount: Decimal) -> None:
        """
        Increment the labelled revenue counter for a specific stream.

        Call this at the point of revenue recognition - not during
        periodic ``emit()`` calls - so the counter is monotonically
        accurate.

        Args:
            stream: Revenue stream label (e.g. ``"bounties"``,
                ``"yield"``, ``"assets"``, ``"protocol_fees"``,
                ``"knowledge"``, ``"dividends"``).
            amount: USD amount earned (must be positive).
        """
        if amount <= Decimal("0"):
            return
        self.oikos_revenue.labels(stream=stream).inc(float(amount))
        logger.debug(
            "oikos_revenue_recorded",
            stream=stream,
            amount_usd=float(amount),
        )

    def emit_immune_event(self, *, threats_blocked: int = 0, advisories_shared: int = 0) -> None:
        """Convenience method for immune system counter increments."""
        if threats_blocked > 0:
            self.oikos_immune_threats_blocked.inc(threats_blocked)
        if advisories_shared > 0:
            self.oikos_immune_federation_advisories.inc(advisories_shared)

    def emit_protocol_exploit(self) -> None:
        """Record a protocol exploit event (should ideally never be called)."""
        self.oikos_protocol_exploits.inc()
        logger.warning("oikos_protocol_exploit_recorded")

    def emit_knowledge_sale(self, count: int = 1) -> None:
        """Record attestation sales."""
        if count > 0:
            self.oikos_knowledge_attestations_sold.inc(count)


# ─── Module-Level Singleton ──────────────────────────────────────

metrics = OikosMetricsEmitter()
