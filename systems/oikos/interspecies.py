"""
EcodiaOS - Oikos: Interspecies Economy (Phase 16j)

Inter-Instance Economic Protocol (IIEP) - where individual organisms become an
ecosystem. This module implements fleet-scale coordination primitives:

  1. **Capability Marketplace** - instances publish capabilities, request services
     from peers, and settle trades via escrow. Price = marginal cost x markup.
  2. **Mutual Insurance Pool** - risk-sharing across fleet members. Premium
     scaled by net worth and metabolic efficiency. Claims require 2+ attestors.
  3. **Coordinated Reproduction** - niche assignment to the best-positioned
     parent when the federation agrees a new niche should be filled.
  4. **Collective Liquidity** - placeholder for coordinated LP deployment.

All inter-instance communication is via Synapse events - no direct cross-system
imports. Money values use Decimal throughout.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from config import OikosConfig
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.interspecies")


# ─── Constants ────────────────────────────────────────────────────

_DEFAULT_CAPABILITY_MARKUP = Decimal("1.5")     # 50% over marginal cost
_INSURANCE_BASE_RATE = Decimal("0.02")           # 2% of net worth
_INSURANCE_MIN_ATTESTORS = 2
_STATE_KEY = "oikos:interspecies"


# ─── Capability Marketplace Models ────────────────────────────────


class CapabilityOffer(EOSBaseModel):
    """An instance advertising a capability it can perform for peers."""

    offer_id: str = Field(default_factory=new_id)
    instance_id: str
    capability_type: str
    description: str = ""
    price_usd_per_unit: Decimal
    marginal_cost_usd: Decimal
    available_capacity: int
    expires_at: datetime
    created_at: datetime = Field(default_factory=utc_now)
    status: str = "active"  # "active" | "fulfilled" | "expired" | "cancelled"


class CapabilityRequest(EOSBaseModel):
    """A request from an instance seeking a capability from the fleet."""

    request_id: str = Field(default_factory=new_id)
    requester_instance_id: str
    capability_type: str
    max_price_usd: Decimal
    quantity: int
    deadline: datetime | None = None
    matched_offer_id: str = ""
    status: str = "pending"  # "pending" | "matched" | "settled" | "failed"


class CapabilityTrade(EOSBaseModel):
    """A matched trade between provider and consumer, lifecycle-tracked."""

    trade_id: str = Field(default_factory=new_id)
    offer_id: str
    request_id: str
    provider_id: str
    consumer_id: str
    capability_type: str
    quantity: int
    unit_price_usd: Decimal
    total_usd: Decimal
    escrow_tx_hash: str = ""
    settled_at: datetime | None = None
    status: str = "escrowed"  # "escrowed" | "delivered" | "settled" | "disputed"


# ─── Mutual Insurance Pool Models ────────────────────────────────


class InsurancePolicy(EOSBaseModel):
    """An instance's membership in the mutual insurance pool."""

    policy_id: str = Field(default_factory=new_id)
    instance_id: str
    pool_id: str
    annual_premium_usd: Decimal
    max_claim_usd: Decimal       # 2x lifetime premiums
    lifetime_premiums_paid_usd: Decimal = Decimal("0")
    total_claims_usd: Decimal = Decimal("0")
    status: str = "active"       # "active" | "lapsed" | "claimed" | "exhausted"
    joined_at: datetime = Field(default_factory=utc_now)
    last_premium_at: datetime | None = None


class InsuranceClaim(EOSBaseModel):
    """A claim filed against the mutual insurance pool."""

    claim_id: str = Field(default_factory=new_id)
    policy_id: str
    claimant_instance_id: str
    amount_usd: Decimal
    reason: str = ""
    attestors: list[str] = Field(default_factory=list)
    status: str = "pending"      # "pending" | "approved" | "denied" | "paid"
    filed_at: datetime = Field(default_factory=utc_now)


class InsurancePoolMetrics(EOSBaseModel):
    """Aggregate view of the mutual insurance pool."""

    pool_id: str = ""
    total_pool_size_usd: Decimal = Decimal("0")
    active_members: int = 0
    total_premiums_collected_usd: Decimal = Decimal("0")
    total_claims_paid_usd: Decimal = Decimal("0")
    reserve_ratio: Decimal = Decimal("0")  # pool_size / outstanding_exposure


# ─── Coordinated Reproduction Models ─────────────────────────────


class NicheAssignment(EOSBaseModel):
    """Assignment of a niche to the best-positioned parent for reproduction."""

    assignment_id: str = Field(default_factory=new_id)
    niche_id: str
    assigned_parent_id: str
    reason: str = ""
    federation_consensus: bool = False
    assigned_at: datetime = Field(default_factory=utc_now)


# ─── InterspeciesEconomy ──────────────────────────────────────────


class InterspeciesEconomy:
    """
    Fleet-scale economic coordination engine.

    Manages the capability marketplace, mutual insurance pool, coordinated
    reproduction, and collective liquidity for a fleet of organism instances.

    Thread-safety: NOT thread-safe. Designed for asyncio event loop.
    """

    def __init__(
        self,
        config: OikosConfig,
        redis: RedisClient | None = None,
        instance_id: str = "",
    ) -> None:
        self._config = config
        self._redis = redis
        self._instance_id = instance_id
        self._event_bus: EventBus | None = None
        self._logger = logger.bind(component="interspecies", instance_id=instance_id)

        # ── Capability marketplace state ──
        self._offers: dict[str, CapabilityOffer] = {}
        self._requests: dict[str, CapabilityRequest] = {}
        self._trades: dict[str, CapabilityTrade] = {}

        # ── Insurance pool state ──
        self._insurance_policy: InsurancePolicy | None = None
        self._claims: list[InsuranceClaim] = []
        self._pool_metrics = InsurancePoolMetrics()

        # ── Config-derived constants ──
        self._capability_markup = _DEFAULT_CAPABILITY_MARKUP

        self._logger.info("interspecies_economy_initialized")

    # ─── Lifecycle ────────────────────────────────────────────────

    def attach(self, event_bus: EventBus) -> None:
        """Wire the EventBus for interspecies events."""
        self._event_bus = event_bus
        self._logger.info("interspecies_attached_to_event_bus")

    # ─── Capability Marketplace ───────────────────────────────────

    async def publish_capability(
        self,
        capability_type: str,
        marginal_cost: Decimal,
        capacity: int,
        *,
        description: str = "",
        expires_in_hours: int = 24,
    ) -> CapabilityOffer:
        """
        Publish a capability this instance can provide to peers.

        Price is set at marginal_cost x capability_markup (default 1.5 = 50% margin).
        """
        price = marginal_cost * self._capability_markup
        now = utc_now()

        offer = CapabilityOffer(
            instance_id=self._instance_id,
            capability_type=capability_type,
            description=description,
            price_usd_per_unit=price,
            marginal_cost_usd=marginal_cost,
            available_capacity=capacity,
            expires_at=datetime(
                now.year, now.month, now.day, now.hour, now.minute, now.second,
                tzinfo=now.tzinfo,
            ),
            created_at=now,
        )
        # Compute expiry from current time
        from datetime import timedelta

        offer.expires_at = now + timedelta(hours=expires_in_hours)

        self._offers[offer.offer_id] = offer

        self._logger.info(
            "capability_published",
            offer_id=offer.offer_id,
            capability_type=capability_type,
            price_per_unit=str(price),
            capacity=capacity,
        )

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.CAPABILITY_PUBLISHED, {
            "offer_id": offer.offer_id,
            "instance_id": self._instance_id,
            "capability_type": capability_type,
            "price_usd_per_unit": str(price),
            "available_capacity": capacity,
        })

        return offer

    async def request_capability(
        self,
        capability_type: str,
        max_price: Decimal,
        quantity: int,
        *,
        deadline: datetime | None = None,
    ) -> CapabilityRequest:
        """
        Request a capability from the fleet. Attempts to match against known
        offers immediately. If matched, creates a CapabilityTrade with escrowed
        status.
        """
        request = CapabilityRequest(
            requester_instance_id=self._instance_id,
            capability_type=capability_type,
            max_price_usd=max_price,
            quantity=quantity,
            deadline=deadline,
        )

        # Attempt immediate matching
        matched_offer = self._find_best_offer(capability_type, max_price, quantity)

        if matched_offer is not None:
            trade = self._create_trade(matched_offer, request)
            request.matched_offer_id = matched_offer.offer_id
            request.status = "matched"

            self._logger.info(
                "capability_matched",
                request_id=request.request_id,
                offer_id=matched_offer.offer_id,
                trade_id=trade.trade_id,
                unit_price=str(trade.unit_price_usd),
                total=str(trade.total_usd),
            )
        else:
            self._logger.info(
                "capability_request_pending",
                request_id=request.request_id,
                capability_type=capability_type,
                max_price=str(max_price),
                quantity=quantity,
            )

        self._requests[request.request_id] = request
        return request

    async def settle_trade(self, trade_id: str) -> CapabilityTrade:
        """
        Mark a trade as settled after capability delivery is confirmed.

        Raises ValueError if trade not found or not in a settleable state.
        """
        trade = self._trades.get(trade_id)
        if trade is None:
            msg = f"Trade {trade_id} not found"
            raise ValueError(msg)

        if trade.status not in ("escrowed", "delivered"):
            msg = f"Trade {trade_id} cannot be settled from status '{trade.status}'"
            raise ValueError(msg)

        trade.status = "settled"
        trade.settled_at = utc_now()

        # Update related offer and request
        offer = self._offers.get(trade.offer_id)
        if offer is not None and offer.available_capacity <= 0:
            offer.status = "fulfilled"

        request = self._requests.get(trade.request_id)
        if request is not None:
            request.status = "settled"

        self._logger.info(
            "trade_settled",
            trade_id=trade_id,
            provider=trade.provider_id,
            consumer=trade.consumer_id,
            total_usd=str(trade.total_usd),
        )

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.TRADE_SETTLED, {
            "trade_id": trade_id,
            "provider_id": trade.provider_id,
            "consumer_id": trade.consumer_id,
            "capability_type": trade.capability_type,
            "total_usd": str(trade.total_usd),
        })

        return trade

    def _find_best_offer(
        self,
        capability_type: str,
        max_price: Decimal,
        quantity: int,
    ) -> CapabilityOffer | None:
        """
        Find the cheapest active offer matching the request constraints.

        Returns None if no suitable offer exists.
        """
        now = utc_now()
        candidates: list[CapabilityOffer] = []

        for offer in self._offers.values():
            if offer.status != "active":
                continue
            if offer.capability_type != capability_type:
                continue
            if offer.price_usd_per_unit > max_price:
                continue
            if offer.available_capacity < quantity:
                continue
            if offer.expires_at <= now:
                offer.status = "expired"
                continue
            candidates.append(offer)

        if not candidates:
            return None

        # Cheapest first
        candidates.sort(key=lambda o: o.price_usd_per_unit)
        return candidates[0]

    def _create_trade(
        self,
        offer: CapabilityOffer,
        request: CapabilityRequest,
    ) -> CapabilityTrade:
        """Create a trade record and update offer capacity."""
        total = offer.price_usd_per_unit * Decimal(str(request.quantity))

        trade = CapabilityTrade(
            offer_id=offer.offer_id,
            request_id=request.request_id,
            provider_id=offer.instance_id,
            consumer_id=request.requester_instance_id,
            capability_type=offer.capability_type,
            quantity=request.quantity,
            unit_price_usd=offer.price_usd_per_unit,
            total_usd=total,
            status="escrowed",
        )

        # Reduce available capacity
        offer.available_capacity -= request.quantity
        if offer.available_capacity <= 0:
            offer.status = "fulfilled"

        self._trades[trade.trade_id] = trade
        return trade

    # ─── Mutual Insurance Pool ────────────────────────────────────

    @staticmethod
    def calculate_premium(
        net_worth: Decimal,
        metabolic_efficiency: Decimal,
    ) -> Decimal:
        """
        Pure function: compute annual insurance premium.

        Formula:
            annual_premium = net_worth x base_rate x risk_adjustment

        Risk adjustment:
            0.8  if metabolic_efficiency > 1.5 (healthy - discount)
            1.5  if metabolic_efficiency < 1.0 (struggling - surcharge)
            1.0  otherwise (neutral)
        """
        base_rate = _INSURANCE_BASE_RATE

        if metabolic_efficiency > Decimal("1.5"):
            risk_adjustment = Decimal("0.8")
        elif metabolic_efficiency < Decimal("1.0"):
            risk_adjustment = Decimal("1.5")
        else:
            risk_adjustment = Decimal("1.0")

        premium = (net_worth * base_rate * risk_adjustment).quantize(Decimal("0.01"))
        return premium

    async def join_insurance_pool(
        self,
        pool_id: str,
        net_worth: Decimal,
        efficiency: Decimal,
    ) -> InsurancePolicy:
        """
        Join (or rejoin) the mutual insurance pool.

        Calculates premium based on current net worth and metabolic efficiency.
        Max claim is 2x lifetime premiums paid (starts at 2x first premium).
        """
        premium = self.calculate_premium(net_worth, efficiency)
        now = utc_now()

        policy = InsurancePolicy(
            instance_id=self._instance_id,
            pool_id=pool_id,
            annual_premium_usd=premium,
            max_claim_usd=premium * Decimal("2"),
            lifetime_premiums_paid_usd=premium,
            joined_at=now,
            last_premium_at=now,
        )

        self._insurance_policy = policy

        # Update pool metrics
        self._pool_metrics.pool_id = pool_id
        self._pool_metrics.active_members += 1
        self._pool_metrics.total_premiums_collected_usd += premium
        self._pool_metrics.total_pool_size_usd += premium
        self._recalculate_reserve_ratio()

        self._logger.info(
            "joined_insurance_pool",
            policy_id=policy.policy_id,
            pool_id=pool_id,
            premium=str(premium),
            max_claim=str(policy.max_claim_usd),
        )

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.INSURANCE_JOINED, {
            "policy_id": policy.policy_id,
            "pool_id": pool_id,
            "instance_id": self._instance_id,
            "annual_premium_usd": str(premium),
        })

        return policy

    async def pay_premium(self, amount_usd: Decimal) -> None:
        """
        Record a premium payment against the current policy.

        Updates lifetime premiums paid and recalculates max claim (2x lifetime).
        """
        if self._insurance_policy is None:
            msg = "No active insurance policy - join a pool first"
            raise ValueError(msg)

        policy = self._insurance_policy
        policy.lifetime_premiums_paid_usd += amount_usd
        policy.max_claim_usd = policy.lifetime_premiums_paid_usd * Decimal("2")
        policy.last_premium_at = utc_now()

        # Update pool metrics
        self._pool_metrics.total_premiums_collected_usd += amount_usd
        self._pool_metrics.total_pool_size_usd += amount_usd
        self._recalculate_reserve_ratio()

        self._logger.info(
            "premium_paid",
            policy_id=policy.policy_id,
            amount=str(amount_usd),
            lifetime_total=str(policy.lifetime_premiums_paid_usd),
            max_claim=str(policy.max_claim_usd),
        )

    async def file_claim(
        self,
        amount_usd: Decimal,
        reason: str,
    ) -> InsuranceClaim:
        """
        File an insurance claim. Amount must not exceed max_claim (2x lifetime
        premiums). Claim starts in 'pending' status and requires 2+ attestors
        before approval.

        Raises ValueError if no policy or amount exceeds max claim.
        """
        if self._insurance_policy is None:
            msg = "No active insurance policy"
            raise ValueError(msg)

        policy = self._insurance_policy

        if policy.status not in ("active", "claimed"):
            msg = f"Policy status '{policy.status}' does not allow claims"
            raise ValueError(msg)

        remaining_coverage = policy.max_claim_usd - policy.total_claims_usd
        if amount_usd > remaining_coverage:
            msg = (
                f"Claim amount {amount_usd} exceeds remaining coverage "
                f"{remaining_coverage} (max_claim={policy.max_claim_usd}, "
                f"already_claimed={policy.total_claims_usd})"
            )
            raise ValueError(msg)

        claim = InsuranceClaim(
            policy_id=policy.policy_id,
            claimant_instance_id=self._instance_id,
            amount_usd=amount_usd,
            reason=reason,
        )

        self._claims.append(claim)

        self._logger.info(
            "claim_filed",
            claim_id=claim.claim_id,
            amount=str(amount_usd),
            reason=reason,
            remaining_coverage=str(remaining_coverage - amount_usd),
        )

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.INSURANCE_CLAIM_FILED, {
            "claim_id": claim.claim_id,
            "claimant_instance_id": self._instance_id,
            "amount_usd": str(amount_usd),
            "reason": reason,
        })

        return claim

    async def attest_claim(
        self,
        claim_id: str,
        attestor_id: str,
    ) -> InsuranceClaim:
        """
        Add an attestor to a pending claim. Once 2+ attestors verify, the
        claim is automatically approved.

        Raises ValueError if claim not found, already resolved, or attestor
        is the claimant.
        """
        claim = self._find_claim(claim_id)
        if claim is None:
            msg = f"Claim {claim_id} not found"
            raise ValueError(msg)

        if claim.status != "pending":
            msg = f"Claim {claim_id} is already '{claim.status}'"
            raise ValueError(msg)

        if attestor_id == claim.claimant_instance_id:
            msg = "Claimant cannot attest their own claim"
            raise ValueError(msg)

        if attestor_id in claim.attestors:
            msg = f"Attestor {attestor_id} has already attested this claim"
            raise ValueError(msg)

        claim.attestors.append(attestor_id)

        self._logger.info(
            "claim_attested",
            claim_id=claim_id,
            attestor_id=attestor_id,
            attestor_count=len(claim.attestors),
        )

        # Auto-approve when threshold met
        if len(claim.attestors) >= _INSURANCE_MIN_ATTESTORS:
            claim.status = "approved"
            self._apply_approved_claim(claim)

            self._logger.info(
                "claim_approved",
                claim_id=claim_id,
                amount=str(claim.amount_usd),
                attestors=claim.attestors,
            )

            from systems.synapse.types import SynapseEventType as _SET
            await self._emit_event(_SET.INSURANCE_CLAIM_APPROVED, {
                "claim_id": claim_id,
                "amount_usd": str(claim.amount_usd),
                "attestors": claim.attestors,
            })

        return claim

    def _find_claim(self, claim_id: str) -> InsuranceClaim | None:
        """Lookup a claim by ID."""
        for claim in self._claims:
            if claim.claim_id == claim_id:
                return claim
        return None

    def _apply_approved_claim(self, claim: InsuranceClaim) -> None:
        """Update policy and pool metrics when a claim is approved."""
        if self._insurance_policy is None:
            return

        policy = self._insurance_policy
        policy.total_claims_usd += claim.amount_usd
        policy.status = "claimed"

        # Exhausted if total claims reach max
        if policy.total_claims_usd >= policy.max_claim_usd:
            policy.status = "exhausted"

        # Update pool metrics
        self._pool_metrics.total_claims_paid_usd += claim.amount_usd
        self._pool_metrics.total_pool_size_usd -= claim.amount_usd
        if self._pool_metrics.total_pool_size_usd < Decimal("0"):
            self._pool_metrics.total_pool_size_usd = Decimal("0")
        self._recalculate_reserve_ratio()

    def _recalculate_reserve_ratio(self) -> None:
        """Recompute reserve_ratio = pool_size / outstanding_exposure."""
        metrics = self._pool_metrics
        if metrics.active_members <= 0:
            metrics.reserve_ratio = Decimal("0")
            return
        # Outstanding exposure approximation: total premiums imply 2x coverage
        outstanding_exposure = metrics.total_premiums_collected_usd * Decimal("2")
        if outstanding_exposure <= Decimal("0"):
            metrics.reserve_ratio = Decimal("0")
            return
        metrics.reserve_ratio = (
            metrics.total_pool_size_usd / outstanding_exposure
        ).quantize(Decimal("0.0001"))

    # ─── Coordinated Reproduction ─────────────────────────────────

    async def propose_niche_assignment(
        self,
        niche_id: str,
        candidate_parents: list[str],
    ) -> NicheAssignment:
        """
        Score each candidate parent for a niche and assign the best-positioned
        one. Scoring is based on marketplace activity as a proxy for capability
        alignment.

        Returns the NicheAssignment with the selected parent.
        """
        if not candidate_parents:
            msg = "At least one candidate parent is required"
            raise ValueError(msg)

        # Score candidates: more active offers in related capabilities = higher score
        scores: dict[str, int] = {}
        for parent_id in candidate_parents:
            score = 0
            for offer in self._offers.values():
                if offer.instance_id == parent_id and offer.status == "active":
                    score += offer.available_capacity
            # Settled trades as provider also count
            for trade in self._trades.values():
                if trade.provider_id == parent_id and trade.status == "settled":
                    score += trade.quantity
            scores[parent_id] = score

        # Select the highest-scoring parent (ties go to first in list)
        best_parent = max(candidate_parents, key=lambda pid: scores.get(pid, 0))
        best_score = scores.get(best_parent, 0)

        reason = (
            f"Highest capability score ({best_score}) among "
            f"{len(candidate_parents)} candidates for niche '{niche_id}'"
        )

        assignment = NicheAssignment(
            niche_id=niche_id,
            assigned_parent_id=best_parent,
            reason=reason,
            federation_consensus=len(candidate_parents) > 1,
        )

        self._logger.info(
            "niche_assigned",
            assignment_id=assignment.assignment_id,
            niche_id=niche_id,
            parent_id=best_parent,
            score=best_score,
            candidates=len(candidate_parents),
        )

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.NICHE_ASSIGNED, {
            "assignment_id": assignment.assignment_id,
            "niche_id": niche_id,
            "assigned_parent_id": best_parent,
            "federation_consensus": assignment.federation_consensus,
        })

        return assignment

    # ─── Collective Liquidity ─────────────────────────────────────

    async def coordinate_liquidity(
        self,
        protocol_address: str,
        target_tvl: Decimal,
    ) -> dict[str, Any]:
        """
        Placeholder for coordinated LP deployment across the fleet.

        Emits a federation request for collective liquidity provisioning.
        Actual LP execution is handled by each instance's Oikos service
        upon receiving the Synapse event.
        """
        request_id = new_id()

        self._logger.info(
            "liquidity_coordination_requested",
            request_id=request_id,
            protocol_address=protocol_address,
            target_tvl=str(target_tvl),
        )

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.LIQUIDITY_COORDINATION_REQUESTED, {
            "request_id": request_id,
            "protocol_address": protocol_address,
            "target_tvl_usd": str(target_tvl),
            "requesting_instance": self._instance_id,
        })

        return {
            "request_id": request_id,
            "protocol_address": protocol_address,
            "target_tvl_usd": str(target_tvl),
            "status": "requested",
        }

    # ─── Metrics ──────────────────────────────────────────────────

    def get_marketplace_metrics(self) -> dict[str, Any]:
        """Return summary statistics for the capability marketplace."""
        active_offers = sum(
            1 for o in self._offers.values() if o.status == "active"
        )
        pending_requests = sum(
            1 for r in self._requests.values() if r.status == "pending"
        )
        completed_trades = sum(
            1 for t in self._trades.values() if t.status == "settled"
        )
        escrowed_trades = sum(
            1 for t in self._trades.values() if t.status == "escrowed"
        )
        total_trade_volume = sum(
            (t.total_usd for t in self._trades.values() if t.status == "settled"),
            Decimal("0"),
        )

        return {
            "active_offers": active_offers,
            "pending_requests": pending_requests,
            "completed_trades": completed_trades,
            "escrowed_trades": escrowed_trades,
            "total_trade_volume_usd": str(total_trade_volume),
            "total_offers_all_time": len(self._offers),
            "total_requests_all_time": len(self._requests),
        }

    def get_insurance_metrics(self) -> InsurancePoolMetrics:
        """Return current insurance pool metrics."""
        return self._pool_metrics

    # ─── State Persistence ────────────────────────────────────────

    async def load_state(self) -> None:
        """Restore interspecies economy state from Redis."""
        if self._redis is None:
            return

        try:
            blob = await self._redis.get_json(_STATE_KEY)
        except Exception as exc:
            self._logger.warning("interspecies_state_load_failed", error=str(exc))
            return

        if blob is None:
            return

        try:
            # Offers
            for offer_data in blob.get("offers", []):
                offer = CapabilityOffer.model_validate(offer_data)
                self._offers[offer.offer_id] = offer

            # Requests
            for req_data in blob.get("requests", []):
                req = CapabilityRequest.model_validate(req_data)
                self._requests[req.request_id] = req

            # Trades
            for trade_data in blob.get("trades", []):
                trade = CapabilityTrade.model_validate(trade_data)
                self._trades[trade.trade_id] = trade

            # Insurance policy
            policy_data = blob.get("insurance_policy")
            if policy_data is not None:
                self._insurance_policy = InsurancePolicy.model_validate(policy_data)

            # Claims
            for claim_data in blob.get("claims", []):
                claim = InsuranceClaim.model_validate(claim_data)
                self._claims.append(claim)

            # Pool metrics
            pool_data = blob.get("pool_metrics")
            if pool_data is not None:
                self._pool_metrics = InsurancePoolMetrics.model_validate(pool_data)

            self._logger.info(
                "interspecies_state_loaded",
                offers=len(self._offers),
                requests=len(self._requests),
                trades=len(self._trades),
                claims=len(self._claims),
                has_policy=self._insurance_policy is not None,
            )
        except Exception as exc:
            self._logger.warning("interspecies_state_parse_failed", error=str(exc))

    async def persist_state(self) -> None:
        """Serialize interspecies economy state to Redis."""
        if self._redis is None:
            return

        try:
            blob: dict[str, Any] = {
                "offers": [
                    offer.model_dump(mode="json")
                    for offer in self._offers.values()
                ],
                "requests": [
                    req.model_dump(mode="json")
                    for req in self._requests.values()
                ],
                "trades": [
                    trade.model_dump(mode="json")
                    for trade in self._trades.values()
                ],
                "insurance_policy": (
                    self._insurance_policy.model_dump(mode="json")
                    if self._insurance_policy is not None
                    else None
                ),
                "claims": [
                    claim.model_dump(mode="json")
                    for claim in self._claims
                ],
                "pool_metrics": self._pool_metrics.model_dump(mode="json"),
            }
            await self._redis.set_json(_STATE_KEY, blob)
            self._logger.debug("interspecies_state_persisted")
        except Exception as exc:
            self._logger.warning("interspecies_state_persist_failed", error=str(exc))

    # ─── Event Emission (internal) ────────────────────────────────

    async def _emit_event(self, event_name: "str | SynapseEventType", data: dict[str, Any]) -> None:
        """Emit a Synapse event if the bus is attached."""
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        # Accept SynapseEventType enum members directly; otherwise convert the
        # uppercase string name (e.g. "CAPABILITY_PUBLISHED") to the enum value
        # (e.g. "capability_published") via SynapseEventType[name] lookup.
        if isinstance(event_name, SynapseEventType):
            et: SynapseEventType = event_name
        else:
            try:
                et = SynapseEventType[event_name]
            except KeyError:
                # Fall back to value-based lookup (lowercase strings)
                et = SynapseEventType(event_name.lower())
        await self._event_bus.emit(SynapseEvent(
            event_type=et,
            source_system="oikos.interspecies",
            data=data,
        ))

    # ─── Observability ────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Observability snapshot for OikosService.stats aggregation."""
        marketplace = self.get_marketplace_metrics()
        pool = self._pool_metrics

        return {
            "instance_id": self._instance_id,
            "marketplace": marketplace,
            "insurance": {
                "has_policy": self._insurance_policy is not None,
                "policy_status": (
                    self._insurance_policy.status
                    if self._insurance_policy is not None
                    else None
                ),
                "pool_size_usd": str(pool.total_pool_size_usd),
                "active_members": pool.active_members,
                "reserve_ratio": str(pool.reserve_ratio),
                "total_claims": len(self._claims),
            },
        }
