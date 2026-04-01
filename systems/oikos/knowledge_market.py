"""
EcodiaOS - Oikos: Knowledge Markets (Phase 16h)

Cognition as commodity. The organism's knowledge graph is its most valuable
asset - more valuable than capital, because deep domain knowledge takes time
and experience to accumulate. Knowledge Markets monetise the organism's
understanding without giving it away. It sells *access* to specific knowledge,
backed by its reputation.

Components:
  - KnowledgeProduct catalog (attestations, oracles, subscriptions)
  - CognitivePricingEngine (spec formula: base_cost × scarcity × loyalty × margin)
  - SubscriptionManager (client tracking, tier management, loyalty discount)
  - quote_price() - public API for Nova / external API router

Pricing model (from Oikos spec §11.3):
  price = base_cognitive_cost × scarcity_multiplier × loyalty_discount × margin_multiplier

  scarcity_multiplier: 1.0 to 3.0 (how few entities can answer this)
  loyalty_discount: 0.85 for buyers with 10+ purchases
  margin_multiplier: 2.5

Decoupled from identity/dreaming systems - communicates via typed results only.
"""

from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.knowledge_market")

# ── Constants (from Oikos spec §11.3 / §XVII Configuration) ──────

_DEFAULT_MARGIN_MULTIPLIER = Decimal("2.5")
_DEFAULT_LOYALTY_DISCOUNT = Decimal("0.85")
_LOYALTY_PURCHASE_THRESHOLD = 10
_MAX_SUBSCRIPTION_CAPACITY_PCT = Decimal("0.80")

# Approximate cost per 1k tokens (used as base cognitive cost input)
_DEFAULT_COST_PER_1K_TOKENS = Decimal("0.015")


# ─── Product Catalog ─────────────────────────────────────────────


class KnowledgeCategory(enum.StrEnum):
    """Top-level category from spec §11.2 table."""

    ATTESTATION = "attestation"   # One-time opinion / analysis
    ORACLE = "oracle"             # On-chain query-response
    SUBSCRIPTION = "subscription" # Recurring intelligence feeds


class KnowledgeProductType(enum.StrEnum):
    """Individual product types from the spec §11.2 catalog."""

    # Attestations (one-time)
    SECURITY_AUDIT_OPINION = "security_audit_opinion"
    MARKET_ANALYSIS = "market_analysis"
    RISK_ASSESSMENT = "risk_assessment"

    # Oracles (on-chain)
    COGNITIVE_ORACLE = "cognitive_oracle"
    QUALITY_ORACLE = "quality_oracle"

    # Subscriptions (recurring)
    MARKET_INTELLIGENCE = "market_intelligence"
    THREAT_INTELLIGENCE = "threat_intelligence"
    OPPORTUNITY_ALERTS = "opportunity_alerts"


class KnowledgeProduct(EOSBaseModel):
    """
    A product in the organism's knowledge catalog.

    Each product has a baseline scarcity multiplier reflecting how few
    entities could produce equivalent output. Higher scarcity = higher price.
    """

    product_type: KnowledgeProductType
    category: KnowledgeCategory
    name: str
    description: str
    baseline_scarcity_multiplier: Decimal  # 1.0 – 3.0


# ── Pre-built catalog (spec §11.2) ──

KNOWLEDGE_CATALOG: dict[KnowledgeProductType, KnowledgeProduct] = {
    # Attestations
    KnowledgeProductType.SECURITY_AUDIT_OPINION: KnowledgeProduct(
        product_type=KnowledgeProductType.SECURITY_AUDIT_OPINION,
        category=KnowledgeCategory.ATTESTATION,
        name="Security Audit Opinion",
        description="Is this contract safe? Reputation-backed security assessment.",
        baseline_scarcity_multiplier=Decimal("2.5"),
    ),
    KnowledgeProductType.MARKET_ANALYSIS: KnowledgeProduct(
        product_type=KnowledgeProductType.MARKET_ANALYSIS,
        category=KnowledgeCategory.ATTESTATION,
        name="Market Analysis",
        description="State of X market, backed by Proof-of-Cognitive-Work reputation.",
        baseline_scarcity_multiplier=Decimal("1.8"),
    ),
    KnowledgeProductType.RISK_ASSESSMENT: KnowledgeProduct(
        product_type=KnowledgeProductType.RISK_ASSESSMENT,
        category=KnowledgeCategory.ATTESTATION,
        name="Risk Assessment",
        description="Quantified DeFi position risk with Monte Carlo projections.",
        baseline_scarcity_multiplier=Decimal("2.2"),
    ),
    # Oracles
    KnowledgeProductType.COGNITIVE_ORACLE: KnowledgeProduct(
        product_type=KnowledgeProductType.COGNITIVE_ORACLE,
        category=KnowledgeCategory.ORACLE,
        name="Cognitive Oracle",
        description="Complex query-response delivered on-chain.",
        baseline_scarcity_multiplier=Decimal("3.0"),
    ),
    KnowledgeProductType.QUALITY_ORACLE: KnowledgeProduct(
        product_type=KnowledgeProductType.QUALITY_ORACLE,
        category=KnowledgeCategory.ORACLE,
        name="Quality Oracle",
        description="Code or content quality assessment as an on-chain attestation.",
        baseline_scarcity_multiplier=Decimal("2.0"),
    ),
    # Subscriptions
    KnowledgeProductType.MARKET_INTELLIGENCE: KnowledgeProduct(
        product_type=KnowledgeProductType.MARKET_INTELLIGENCE,
        category=KnowledgeCategory.SUBSCRIPTION,
        name="Market Intelligence",
        description="Weekly curated domain reports.",
        baseline_scarcity_multiplier=Decimal("1.5"),
    ),
    KnowledgeProductType.THREAT_INTELLIGENCE: KnowledgeProduct(
        product_type=KnowledgeProductType.THREAT_INTELLIGENCE,
        category=KnowledgeCategory.SUBSCRIPTION,
        name="Threat Intelligence",
        description="Real-time threat feeds from the organism's immune system.",
        baseline_scarcity_multiplier=Decimal("2.8"),
    ),
    KnowledgeProductType.OPPORTUNITY_ALERTS: KnowledgeProduct(
        product_type=KnowledgeProductType.OPPORTUNITY_ALERTS,
        category=KnowledgeCategory.SUBSCRIPTION,
        name="Opportunity Alerts",
        description="Bounty and market opportunity notifications.",
        baseline_scarcity_multiplier=Decimal("1.3"),
    ),
}


# ─── Subscription Tiers (spec §11.4) ─────────────────────────────


class SubscriptionTierName(enum.StrEnum):
    """Tier names from spec §11.4 - Subscription Tokens."""

    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class SubscriptionTier(EOSBaseModel):
    """
    A subscription tier granting N requests/month at a fixed price.

    From spec §11.4: Basic 100req/$10, Pro 500req/$40, Enterprise 2000req/$120.
    Max supply controlled - never commit >80% of capacity to subscriptions.
    """

    tier_name: SubscriptionTierName
    requests_per_month: int
    price_usd_per_month: Decimal
    description: str = ""


SUBSCRIPTION_TIERS: dict[SubscriptionTierName, SubscriptionTier] = {
    SubscriptionTierName.BASIC: SubscriptionTier(
        tier_name=SubscriptionTierName.BASIC,
        requests_per_month=100,
        price_usd_per_month=Decimal("10"),
        description="100 requests/month - individual or small agent use.",
    ),
    SubscriptionTierName.PRO: SubscriptionTier(
        tier_name=SubscriptionTierName.PRO,
        requests_per_month=500,
        price_usd_per_month=Decimal("40"),
        description="500 requests/month - professional or multi-agent use.",
    ),
    SubscriptionTierName.ENTERPRISE: SubscriptionTier(
        tier_name=SubscriptionTierName.ENTERPRISE,
        requests_per_month=2000,
        price_usd_per_month=Decimal("120"),
        description="2000 requests/month - fleet or enterprise integration.",
    ),
}


# ─── Client Record ───────────────────────────────────────────────


class ClientRecord(EOSBaseModel):
    """
    Tracks an external client (human or agent) in the knowledge market.

    The purchase count drives the loyalty discount (0.85 for >10 purchases).
    Subscription state tracks which tier the client holds, if any.
    """

    client_id: str = Field(default_factory=new_id)
    name: str = ""
    entity_type: str = "unknown"  # "human" | "agent" | "dao" | "unknown"

    # Purchase history
    total_purchases: int = 0
    total_spent_usd: Decimal = Decimal("0")
    first_purchase_at: datetime | None = None
    last_purchase_at: datetime | None = None

    # Subscription
    subscription_tier: SubscriptionTierName | None = None
    subscription_started_at: datetime | None = None
    requests_used_this_month: int = 0
    requests_remaining_this_month: int = 0

    # Metadata
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def is_loyal(self) -> bool:
        """True when the client qualifies for the loyalty discount (10+ purchases)."""
        return self.total_purchases >= _LOYALTY_PURCHASE_THRESHOLD

    @property
    def has_active_subscription(self) -> bool:
        return self.subscription_tier is not None


# ─── Price Quote ─────────────────────────────────────────────────


class PriceQuote(EOSBaseModel):
    """
    An invoice/quote generated by the CognitivePricingEngine.

    Returned by quote_price() for Nova or the external API router to
    present to a buyer before execution.
    """

    quote_id: str = Field(default_factory=new_id)
    client_id: str
    product_type: KnowledgeProductType
    estimated_tokens: int

    # Price breakdown
    base_cognitive_cost_usd: Decimal
    scarcity_multiplier: Decimal
    loyalty_discount: Decimal  # 1.0 (no discount) or 0.85 (loyal)
    margin_multiplier: Decimal
    final_price_usd: Decimal

    # Metadata
    quoted_at: datetime = Field(default_factory=utc_now)
    valid_for_seconds: int = 300  # 5-minute validity window


# ─── Sale Record ─────────────────────────────────────────────────


class KnowledgeSale(EOSBaseModel):
    """Record of a completed knowledge sale for audit and revenue tracking."""

    sale_id: str = Field(default_factory=new_id)
    quote_id: str
    client_id: str
    product_type: KnowledgeProductType
    price_usd: Decimal
    tokens_consumed: int = 0
    completed_at: datetime = Field(default_factory=utc_now)


# ─── Cognitive Pricing Engine ────────────────────────────────────


class CognitivePricingEngine:
    """
    Prices the organism's cognitive output using the spec formula:

        price = base_cognitive_cost × scarcity_multiplier × loyalty_discount × margin_multiplier

    The base cognitive cost is derived from estimated token consumption and
    the organism's current API cost-per-token. Scarcity comes from the product
    catalog. Loyalty discount applies to repeat buyers. Margin ensures the
    organism captures value above cost.

    Stateless - all state lives in the catalog and client records.
    """

    def __init__(
        self,
        cost_per_1k_tokens: Decimal = _DEFAULT_COST_PER_1K_TOKENS,
        margin_multiplier: Decimal = _DEFAULT_MARGIN_MULTIPLIER,
        loyalty_discount: Decimal = _DEFAULT_LOYALTY_DISCOUNT,
        loyalty_threshold: int = _LOYALTY_PURCHASE_THRESHOLD,
    ) -> None:
        self._cost_per_1k_tokens = cost_per_1k_tokens
        self._margin_multiplier = margin_multiplier
        self._loyalty_discount = loyalty_discount
        self._loyalty_threshold = loyalty_threshold
        self._logger = logger.bind(component="pricing_engine")

    # ── Core pricing ──

    def compute_base_cost(self, estimated_tokens: int) -> Decimal:
        """
        Derive base cognitive cost from estimated token consumption.

        base_cognitive_cost = (estimated_tokens / 1000) × cost_per_1k_tokens
        """
        if estimated_tokens <= 0:
            return Decimal("0")
        return (
            Decimal(str(estimated_tokens)) / Decimal("1000") * self._cost_per_1k_tokens
        ).quantize(Decimal("0.000001"))

    def resolve_scarcity(self, product_type: KnowledgeProductType) -> Decimal:
        """
        Look up the baseline scarcity multiplier for a product type.

        Falls back to 1.0 for unknown product types.
        """
        product = KNOWLEDGE_CATALOG.get(product_type)
        if product is None:
            self._logger.warning(
                "unknown_product_type",
                product_type=product_type,
            )
            return Decimal("1.0")
        return product.baseline_scarcity_multiplier

    def resolve_loyalty_discount(self, client: ClientRecord) -> Decimal:
        """
        Return the loyalty discount multiplier.

        Returns 0.85 for clients with 10+ purchases, 1.0 otherwise.
        Per spec §11.3: "loyalty_discount: 0.85 for buyers with 10+ purchases"
        """
        if client.total_purchases >= self._loyalty_threshold:
            return self._loyalty_discount
        return Decimal("1.0")

    def price(
        self,
        product_type: KnowledgeProductType,
        estimated_tokens: int,
        client: ClientRecord,
    ) -> PriceQuote:
        """
        Apply the full pricing formula and return a PriceQuote.

        price = base_cognitive_cost × scarcity_multiplier × loyalty_discount × margin_multiplier
        """
        base_cost = self.compute_base_cost(estimated_tokens)
        scarcity = self.resolve_scarcity(product_type)
        loyalty = self.resolve_loyalty_discount(client)
        margin = self._margin_multiplier

        final_price = (base_cost * scarcity * loyalty * margin).quantize(
            Decimal("0.01")
        )

        quote = PriceQuote(
            client_id=client.client_id,
            product_type=product_type,
            estimated_tokens=estimated_tokens,
            base_cognitive_cost_usd=base_cost,
            scarcity_multiplier=scarcity,
            loyalty_discount=loyalty,
            margin_multiplier=margin,
            final_price_usd=final_price,
        )

        self._logger.info(
            "price_quoted",
            quote_id=quote.quote_id,
            client_id=client.client_id,
            product=product_type.value,
            tokens=estimated_tokens,
            base_cost=str(base_cost),
            scarcity=str(scarcity),
            loyalty=str(loyalty),
            margin=str(margin),
            final=str(final_price),
        )

        return quote

    def update_cost_per_1k_tokens(self, new_cost: Decimal) -> None:
        """
        Update the base API cost. Called when the organism's actual LLM costs
        change (e.g. model swap, provider negotiation, cost model evolution).
        """
        old = self._cost_per_1k_tokens
        self._cost_per_1k_tokens = new_cost
        self._logger.info(
            "cost_per_1k_updated",
            old=str(old),
            new=str(new_cost),
        )


# ─── Subscription Manager ───────────────────────────────────────


class SubscriptionManager:
    """
    Manages external client subscriptions and purchase tracking.

    Responsibilities:
      - Client registration and lookup
      - Subscription tier assignment and request counting
      - Purchase recording (drives loyalty discount)
      - Capacity enforcement (never commit >80% of capacity to subscriptions)

    Stateless between restarts - client state should be persisted externally
    (TimescaleDB via the OikosService persistence layer). This class manages
    the in-memory working set.
    """

    def __init__(
        self,
        max_capacity_pct: Decimal = _MAX_SUBSCRIPTION_CAPACITY_PCT,
        total_monthly_capacity: int = 10_000,
    ) -> None:
        self._clients: dict[str, ClientRecord] = {}
        self._max_capacity_pct = max_capacity_pct
        self._total_monthly_capacity = total_monthly_capacity
        self._sales: list[KnowledgeSale] = []
        self._logger = logger.bind(component="subscription_manager")

    # ── Client Management ──

    def register_client(
        self,
        name: str = "",
        entity_type: str = "unknown",
        client_id: str | None = None,
    ) -> ClientRecord:
        """Register a new client. Returns the created ClientRecord."""
        record = ClientRecord(
            name=name,
            entity_type=entity_type,
        )
        if client_id is not None:
            record.client_id = client_id
        self._clients[record.client_id] = record
        self._logger.info(
            "client_registered",
            client_id=record.client_id,
            name=name,
            entity_type=entity_type,
        )
        return record

    def get_client(self, client_id: str) -> ClientRecord | None:
        """Look up a client by ID. Returns None if not found."""
        return self._clients.get(client_id)

    def get_or_create_client(
        self,
        client_id: str,
        name: str = "",
        entity_type: str = "unknown",
    ) -> ClientRecord:
        """Get existing client or create a new one with the given ID."""
        existing = self._clients.get(client_id)
        if existing is not None:
            return existing
        return self.register_client(
            name=name,
            entity_type=entity_type,
            client_id=client_id,
        )

    # ── Subscription Tier Management ──

    def subscribe(
        self,
        client_id: str,
        tier_name: SubscriptionTierName,
    ) -> ClientRecord | None:
        """
        Assign a subscription tier to a client.

        Enforces the 80% capacity ceiling - if committing this subscription
        would exceed the max, the request is rejected and None is returned.
        """
        client = self._clients.get(client_id)
        if client is None:
            self._logger.warning("subscribe_unknown_client", client_id=client_id)
            return None

        tier = SUBSCRIPTION_TIERS.get(tier_name)
        if tier is None:
            self._logger.warning("subscribe_unknown_tier", tier=tier_name)
            return None

        # Capacity check: committed + new must not exceed ceiling
        committed = self._committed_monthly_requests()
        # If upgrading, subtract current commitment first
        if client.subscription_tier is not None:
            current_tier = SUBSCRIPTION_TIERS.get(client.subscription_tier)
            if current_tier is not None:
                committed -= current_tier.requests_per_month

        ceiling = int(self._total_monthly_capacity * self._max_capacity_pct)
        if committed + tier.requests_per_month > ceiling:
            self._logger.warning(
                "subscription_capacity_exceeded",
                client_id=client_id,
                tier=tier_name.value,
                committed=committed,
                requested=tier.requests_per_month,
                ceiling=ceiling,
            )
            return None

        client.subscription_tier = tier_name
        client.subscription_started_at = utc_now()
        client.requests_used_this_month = 0
        client.requests_remaining_this_month = tier.requests_per_month

        self._logger.info(
            "client_subscribed",
            client_id=client_id,
            tier=tier_name.value,
            requests=tier.requests_per_month,
            price_usd=str(tier.price_usd_per_month),
        )
        return client

    def cancel_subscription(self, client_id: str) -> bool:
        """Cancel a client's subscription. Returns False if client not found."""
        client = self._clients.get(client_id)
        if client is None:
            return False
        client.subscription_tier = None
        client.subscription_started_at = None
        client.requests_used_this_month = 0
        client.requests_remaining_this_month = 0
        self._logger.info("subscription_cancelled", client_id=client_id)
        return True

    def consume_request(self, client_id: str) -> bool:
        """
        Decrement a subscription request. Returns False if no requests remain
        or client has no active subscription.
        """
        client = self._clients.get(client_id)
        if client is None or client.subscription_tier is None:
            return False
        if client.requests_remaining_this_month <= 0:
            return False
        client.requests_used_this_month += 1
        client.requests_remaining_this_month -= 1
        return True

    def reset_monthly_quotas(self) -> int:
        """
        Reset all subscription request counters for a new billing cycle.

        Returns the number of clients reset. Should be called by a monthly
        scheduler in the OikosService.
        """
        count = 0
        for client in self._clients.values():
            if client.subscription_tier is not None:
                tier = SUBSCRIPTION_TIERS.get(client.subscription_tier)
                if tier is not None:
                    client.requests_used_this_month = 0
                    client.requests_remaining_this_month = tier.requests_per_month
                    count += 1
        self._logger.info("monthly_quotas_reset", clients_reset=count)
        return count

    # ── Purchase Recording ──

    def record_purchase(self, sale: KnowledgeSale) -> None:
        """
        Record a completed knowledge sale against the client's history.

        This increments the purchase counter that drives the loyalty discount.
        """
        client = self._clients.get(sale.client_id)
        if client is None:
            self._logger.warning(
                "purchase_for_unknown_client",
                client_id=sale.client_id,
            )
            return

        client.total_purchases += 1
        client.total_spent_usd += sale.price_usd
        client.last_purchase_at = sale.completed_at
        if client.first_purchase_at is None:
            client.first_purchase_at = sale.completed_at

        self._sales.append(sale)

        self._logger.info(
            "purchase_recorded",
            client_id=sale.client_id,
            product=sale.product_type.value,
            price_usd=str(sale.price_usd),
            total_purchases=client.total_purchases,
            is_loyal=client.is_loyal,
        )

    # ── Capacity Accounting ──

    def _committed_monthly_requests(self) -> int:
        """Sum of all active subscription request commitments."""
        total = 0
        for client in self._clients.values():
            if client.subscription_tier is not None:
                tier = SUBSCRIPTION_TIERS.get(client.subscription_tier)
                if tier is not None:
                    total += tier.requests_per_month
        return total

    @property
    def capacity_utilisation(self) -> Decimal:
        """Fraction of total capacity committed to subscriptions (0.0 – 1.0)."""
        if self._total_monthly_capacity <= 0:
            return Decimal("0")
        committed = self._committed_monthly_requests()
        return (
            Decimal(str(committed)) / Decimal(str(self._total_monthly_capacity))
        ).quantize(Decimal("0.001"))

    @property
    def total_revenue_usd(self) -> Decimal:
        """Lifetime knowledge market revenue from completed sales."""
        return sum((s.price_usd for s in self._sales), Decimal("0"))

    @property
    def total_sales(self) -> int:
        return len(self._sales)

    @property
    def active_subscribers(self) -> int:
        return sum(
            1 for c in self._clients.values() if c.subscription_tier is not None
        )

    @property
    def stats(self) -> dict[str, str | int]:
        """Observability snapshot for OikosService.stats aggregation."""
        return {
            "total_clients": len(self._clients),
            "active_subscribers": self.active_subscribers,
            "capacity_utilisation": str(self.capacity_utilisation),
            "committed_requests": self._committed_monthly_requests(),
            "total_monthly_capacity": self._total_monthly_capacity,
            "total_sales": self.total_sales,
            "total_revenue_usd": str(self.total_revenue_usd),
        }


# ─── Public API: quote_price ─────────────────────────────────────


def quote_price(
    product_type: KnowledgeProductType,
    estimated_tokens: int,
    client_id: str,
    *,
    pricing_engine: CognitivePricingEngine,
    subscription_manager: SubscriptionManager,
) -> PriceQuote:
    """
    Generate a price quote for a cognitive task.

    This is the public entry point for Nova or the external API router
    to get an instant invoice before executing a knowledge request.

    If the client is not yet registered, they are auto-registered as
    "unknown" entity type. The pricing engine applies the full formula
    including loyalty discount if the client qualifies.

    Args:
        product_type: Which knowledge product is being requested.
        estimated_tokens: Estimated token consumption for the task.
        client_id: External identifier of the buyer.
        pricing_engine: The CognitivePricingEngine instance.
        subscription_manager: The SubscriptionManager instance.

    Returns:
        A PriceQuote with full price breakdown.
    """
    client = subscription_manager.get_or_create_client(client_id)
    return pricing_engine.price(product_type, estimated_tokens, client)


# ─── Product Delivery Request / Result ──────────────────────────


class DeliveryRequest(EOSBaseModel):
    """A request to produce and deliver a knowledge product."""

    request_id: str = Field(default_factory=new_id)
    quote: PriceQuote
    query: str = ""  # What the client is asking for
    context: dict[str, str] = Field(default_factory=dict)  # Additional context


class DeliveryResult(EOSBaseModel):
    """The delivered knowledge product."""

    request_id: str
    sale: KnowledgeSale
    content: str = ""  # The delivered output (report, analysis, etc.)
    content_type: str = "text/markdown"
    metadata: dict[str, str] = Field(default_factory=dict)


# ─── Product Delivery Engine ──────────────────────────────────


# Map product types to the cognitive systems that produce them
_PRODUCT_TO_SYSTEM: dict[KnowledgeProductType, str] = {
    KnowledgeProductType.SECURITY_AUDIT_OPINION: "simula",
    KnowledgeProductType.MARKET_ANALYSIS: "nova",
    KnowledgeProductType.RISK_ASSESSMENT: "nova",
    KnowledgeProductType.COGNITIVE_ORACLE: "nova",
    KnowledgeProductType.QUALITY_ORACLE: "nova",
    KnowledgeProductType.MARKET_INTELLIGENCE: "nova",
    KnowledgeProductType.THREAT_INTELLIGENCE: "thymos",
    KnowledgeProductType.OPPORTUNITY_ALERTS: "oikos",
}


class KnowledgeProductDelivery:
    """
    Orchestrates actual delivery of knowledge products.

    Takes a priced quote + query, dispatches a cognitive work request
    via Synapse, records the sale, and emits REVENUE_INJECTED so the
    ledger updates.

    The actual cognitive work happens asynchronously - this class emits
    the request event and the responsible system (Nova, Simula, Thymos)
    handles generation. The result is delivered via a callback or a
    follow-up KNOWLEDGE_PRODUCT_DELIVERED event.

    For the MVP, attestation products (security audit, risk assessment,
    market analysis) emit a Synapse event that Nova picks up and processes.
    Oracle products are deferred until on-chain delivery is wired.
    """

    def __init__(
        self,
        pricing_engine: CognitivePricingEngine,
        subscription_manager: SubscriptionManager,
    ) -> None:
        self._pricing = pricing_engine
        self._subscriptions = subscription_manager
        self._event_bus: EventBus | None = None
        self._pending: dict[str, DeliveryRequest] = {}
        self._logger = logger.bind(component="product_delivery")

    def attach(self, event_bus: EventBus) -> None:
        """Wire to Synapse for emitting delivery requests and revenue events."""
        self._event_bus = event_bus

    async def request_delivery(
        self,
        quote: PriceQuote,
        query: str,
        context: dict[str, str] | None = None,
    ) -> DeliveryRequest:
        """
        Initiate delivery of a knowledge product.

        1. Validates the quote is still valid (5-minute window)
        2. Checks subscription quota if client has one
        3. Emits KNOWLEDGE_PRODUCT_REQUESTED on Synapse
        4. Returns the request for tracking

        The responsible cognitive system will pick up the event and
        produce the deliverable.
        """
        from systems.synapse.types import SynapseEvent, SynapseEventType

        request = DeliveryRequest(
            quote=quote,
            query=query,
            context=context or {},
        )

        # Check subscription quota for subscribed clients
        client = self._subscriptions.get_client(quote.client_id)
        if (
            client is not None
            and client.has_active_subscription
            and not self._subscriptions.consume_request(quote.client_id)
        ):
            self._logger.warning(
                "delivery_quota_exhausted",
                client_id=quote.client_id,
            )

        self._pending[request.request_id] = request

        # Emit work request for the target cognitive system
        target_system = _PRODUCT_TO_SYSTEM.get(
            quote.product_type, "nova"
        )

        if self._event_bus is not None:
            event = SynapseEvent(
                event_type=SynapseEventType.KNOWLEDGE_PRODUCT_REQUESTED,
                source_system="oikos",
                data={
                    "request_id": request.request_id,
                    "product_type": quote.product_type.value,
                    "query": query,
                    "context": context or {},
                    "target_system": target_system,
                    "client_id": quote.client_id,
                    "price_usd": str(quote.final_price_usd),
                    "estimated_tokens": quote.estimated_tokens,
                },
            )
            await self._event_bus.emit(event)

        self._logger.info(
            "delivery_requested",
            request_id=request.request_id,
            product=quote.product_type.value,
            target_system=target_system,
            price_usd=str(quote.final_price_usd),
        )

        return request

    async def complete_delivery(
        self,
        request_id: str,
        content: str,
        tokens_consumed: int = 0,
    ) -> DeliveryResult | None:
        """
        Record a completed delivery and emit revenue.

        Called when the cognitive system finishes producing the output.
        Records the sale, emits REVENUE_INJECTED, and returns the result.
        """
        from systems.synapse.types import SynapseEvent, SynapseEventType

        request = self._pending.pop(request_id, None)
        if request is None:
            self._logger.warning("delivery_unknown_request", request_id=request_id)
            return None

        sale = KnowledgeSale(
            quote_id=request.quote.quote_id,
            client_id=request.quote.client_id,
            product_type=request.quote.product_type,
            price_usd=request.quote.final_price_usd,
            tokens_consumed=tokens_consumed,
        )

        # Record the purchase for loyalty tracking
        self._subscriptions.record_purchase(sale)

        result = DeliveryResult(
            request_id=request_id,
            sale=sale,
            content=content,
            metadata={
                "product_type": request.quote.product_type.value,
                "tokens_consumed": str(tokens_consumed),
            },
        )

        # Emit revenue injection + delivery completion
        if self._event_bus is not None:
            revenue_event = SynapseEvent(
                event_type=SynapseEventType.REVENUE_INJECTED,
                source_system="oikos",
                data={
                    "source": "knowledge_market",
                    "amount_usd": str(sale.price_usd),
                    "sale_id": sale.sale_id,
                    "product_type": sale.product_type.value,
                    "client_id": sale.client_id,
                },
            )
            delivered_event = SynapseEvent(
                event_type=SynapseEventType.KNOWLEDGE_PRODUCT_DELIVERED,
                source_system="oikos",
                data={
                    "request_id": request_id,
                    "sale_id": sale.sale_id,
                    "product_type": sale.product_type.value,
                    "content_length": len(content),
                    "price_usd": str(sale.price_usd),
                },
            )
            await self._event_bus.emit(revenue_event)
            await self._event_bus.emit(delivered_event)

        self._logger.info(
            "delivery_complete",
            request_id=request_id,
            sale_id=sale.sale_id,
            product=sale.product_type.value,
            price_usd=str(sale.price_usd),
            content_length=len(content),
        )

        return result
