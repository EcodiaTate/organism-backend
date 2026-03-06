"""
EcodiaOS — Inter-Instance Economic Protocol (Phase 16j: The Interspecies Economy)

The IIEP extends the Federation Protocol with economic message types, enabling
certified EcodiaOS instances to trade cognitive capabilities, insure each other
against bankruptcy, and coordinate fleet-scale economic activity.

Key components:
  - CapabilityMarketplace: Instances publish surplus capacity at 1.5x marginal
    cost; requesting instances match against offers with automatic settlement.
  - MutualInsurancePool: Instances pay premiums during prosperity, receive
    payouts during emergency. Claims require 2+ member attestations and are
    capped at 2x lifetime premiums paid.
  - IIEPManager: Orchestrator that gates every interaction on a valid
    EcodianCertificate and COLLEAGUE+ trust level.

All monetary values use Decimal to avoid floating-point rounding on money.
Spec reference: EcodiaOS_Spec_16_Oikos.md XIII (Level 10).
"""

from __future__ import annotations

import enum
from decimal import Decimal
from datetime import datetime
from typing import Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, utc_now
from primitives.federation import (
    FederationInteraction,
    FederationLink,
    InteractionOutcome,
    TrustLevel,
)
from systems.identity.certificate import (
    CertificateStatus,
    EcodianCertificate,
    verify_certificate_signature,
)

logger = structlog.get_logger("federation.iiep")

# --- Configuration Constants (from spec XVII) --------------------------------

DEFAULT_CAPABILITY_MARKUP: Decimal = Decimal("1.5")
"""Surplus capability offers are priced at 1.5x marginal cost."""

INSURANCE_BASE_PREMIUM_RATE: Decimal = Decimal("0.02")
"""Base annual premium = 2% of net worth."""

INSURANCE_MAX_CLAIM_MULTIPLIER: Decimal = Decimal("2.0")
"""Max claim = 2x lifetime premiums paid."""

MIN_TRUST_FOR_ECONOMIC_COORDINATION: TrustLevel = TrustLevel.COLLEAGUE
"""Minimum trust level required for IIEP interactions."""

TELECOM_PROVISIONING_PRICE_USDC: Decimal = Decimal("5")
"""Published price for the Genesis TelecomProvisioning service (5 USDC)."""

TELECOM_PROVISIONING_MARGINAL_COST_USDC: Decimal = (
    TELECOM_PROVISIONING_PRICE_USDC / DEFAULT_CAPABILITY_MARKUP
).quantize(Decimal("0.0001"))
"""Implied marginal cost so 1.5x markup yields the 5 USDC price."""

REQUIRED_CLAIM_ATTESTATIONS: int = 2
"""Number of pool members that must attest a claim before payout."""


# =============================================================================
# Data Models
# =============================================================================


class CapabilityCategory(enum.StrEnum):
    """Categories of cognitive capability that can be traded."""

    SECURITY_AUDIT = "security_audit"
    DATA_PIPELINE = "data_pipeline"
    MARKET_ANALYSIS = "market_analysis"
    CODE_GENERATION = "code_generation"
    RISK_ASSESSMENT = "risk_assessment"
    CONTENT_CREATION = "content_creation"
    THREAT_INTELLIGENCE = "threat_intelligence"
    QUALITY_ASSURANCE = "quality_assurance"
    GENERAL_COGNITION = "general_cognition"
    TELECOM_PROVISIONING = "telecom_provisioning"


class CapabilityOfferStatus(enum.StrEnum):
    """Lifecycle of a capability offer."""

    ACTIVE = "active"
    MATCHED = "matched"
    FULFILLED = "fulfilled"
    EXPIRED = "expired"
    WITHDRAWN = "withdrawn"


class CapabilityOffer(Identified):
    """
    A surplus cognitive capability published to the marketplace.

    Instances advertise spare capacity at 1.5x marginal cost, creating
    autonomous price discovery for AI cognitive capabilities.
    """

    provider_instance_id: str
    provider_certificate_id: str
    category: CapabilityCategory
    description: str
    marginal_cost_usd: Decimal
    """Cost to the provider for fulfilling one unit of this capability."""
    offered_price_usd: Decimal = Decimal("0")
    """Published price (auto-calculated as marginal_cost x markup)."""
    markup: Decimal = DEFAULT_CAPABILITY_MARKUP

    capacity_units: int = 1
    """How many units of this capability are available."""
    remaining_units: int = 1
    estimated_completion_ms: int = 5000
    status: CapabilityOfferStatus = CapabilityOfferStatus.ACTIVE
    published_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None

    def model_post_init(self, __context: Any) -> None:
        """Auto-calculate offered price from marginal cost x markup."""
        if self.offered_price_usd == Decimal("0") and self.marginal_cost_usd > Decimal("0"):
            self.offered_price_usd = (self.marginal_cost_usd * self.markup).quantize(
                Decimal("0.0001")
            )
        if self.remaining_units == 1 and self.capacity_units > 1:
            self.remaining_units = self.capacity_units

    @property
    def is_available(self) -> bool:
        return (
            self.status == CapabilityOfferStatus.ACTIVE
            and self.remaining_units > 0
            and (self.expires_at is None or utc_now() < self.expires_at)
        )


class CapabilityRequest(Identified):
    """A request to consume a cognitive capability from the marketplace."""

    requester_instance_id: str
    requester_certificate_id: str
    category: CapabilityCategory
    description: str
    max_price_usd: Decimal
    """Maximum the requester is willing to pay per unit."""
    units_needed: int = 1
    urgency: float = 0.5
    """0.0 (low) to 1.0 (critical)."""
    requested_at: datetime = Field(default_factory=utc_now)


class CapabilityMatch(Identified):
    """A matched capability trade between provider and requester."""

    offer_id: str
    request_id: str
    provider_instance_id: str
    requester_instance_id: str
    category: CapabilityCategory
    agreed_price_usd: Decimal
    units: int = 1
    settled: bool = False
    settlement_tx_hash: str = ""
    matched_at: datetime = Field(default_factory=utc_now)
    settled_at: datetime | None = None


# --- Insurance Pool Models ---------------------------------------------------


class InsuranceClaimStatus(enum.StrEnum):
    """Lifecycle of an insurance claim."""

    PENDING = "pending"
    ATTESTING = "attesting"
    APPROVED = "approved"
    DENIED = "denied"
    PAID = "paid"


class PoolMembership(Identified):
    """Tracks an instance's participation in the Mutual Insurance Pool."""

    instance_id: str
    certificate_id: str
    net_worth_at_join: Decimal
    current_annual_premium: Decimal = Decimal("0")
    lifetime_premiums_paid: Decimal = Decimal("0")
    lifetime_claims_paid: Decimal = Decimal("0")
    risk_adjustment: Decimal = Decimal("1.0")
    joined_at: datetime = Field(default_factory=utc_now)
    last_premium_paid_at: datetime | None = None
    active: bool = True

    @property
    def max_claimable(self) -> Decimal:
        """Max claim = 2x lifetime premiums paid."""
        return self.lifetime_premiums_paid * INSURANCE_MAX_CLAIM_MULTIPLIER

    @property
    def remaining_claimable(self) -> Decimal:
        """How much this member can still claim."""
        remaining = self.max_claimable - self.lifetime_claims_paid
        return max(remaining, Decimal("0"))


class ClaimAttestation(EOSBaseModel):
    """An attestation from a pool member supporting (or opposing) a claim."""

    attestor_instance_id: str
    attestor_certificate_id: str
    claim_id: str
    supports: bool
    reason: str = ""
    attested_at: datetime = Field(default_factory=utc_now)


class InsuranceClaim(Identified):
    """A claim against the Mutual Insurance Pool."""

    claimant_instance_id: str
    claimant_certificate_id: str
    requested_amount_usd: Decimal
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    status: InsuranceClaimStatus = InsuranceClaimStatus.PENDING
    attestations: list[ClaimAttestation] = Field(default_factory=list)
    approved_amount_usd: Decimal = Decimal("0")
    payout_tx_hash: str = ""
    filed_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None

    @property
    def supporting_attestations(self) -> int:
        return sum(1 for a in self.attestations if a.supports)

    @property
    def opposing_attestations(self) -> int:
        return sum(1 for a in self.attestations if not a.supports)

    @property
    def has_sufficient_attestations(self) -> bool:
        return self.supporting_attestations >= REQUIRED_CLAIM_ATTESTATIONS


# =============================================================================
# Mutual Insurance Pool
# =============================================================================


def _validate_certificate(
    certificate: EcodianCertificate, expected_instance_id: str
) -> bool:
    """Validate certificate is valid and belongs to the expected instance."""
    if certificate.instance_id != expected_instance_id:
        logger.warning(
            "certificate_instance_mismatch",
            cert_instance=certificate.instance_id,
            expected=expected_instance_id,
        )
        return False
    if certificate.status in (CertificateStatus.EXPIRED, CertificateStatus.REVOKED):
        logger.warning(
            "certificate_not_valid",
            instance_id=expected_instance_id,
            status=certificate.status,
        )
        return False
    return True


class MutualInsurancePool:
    """
    Instances insure each other against bankruptcy.

    Premium formula (from spec XIII.2):
        annual_premium = net_worth x base_rate x risk_adjustment
        risk_adjustment: 0.8x for metabolic_efficiency > 1.5
                         1.5x for metabolic_efficiency < 1.0
                         1.0x otherwise

    Claims:
        - Max claim = 2x lifetime premiums paid
        - Requires attestation from 2+ active pool members
        - Claimant cannot attest their own claim
    """

    def __init__(self) -> None:
        self._logger = logger.bind(component="mutual_insurance_pool")
        self._members: dict[str, PoolMembership] = {}
        self._claims: dict[str, InsuranceClaim] = {}
        self._pool_balance: Decimal = Decimal("0")

    # --- Membership ----------------------------------------------------------

    def calculate_premium(
        self,
        net_worth: Decimal,
        metabolic_efficiency: Decimal,
        base_rate: Decimal = INSURANCE_BASE_PREMIUM_RATE,
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate annual premium and risk adjustment factor.

        Returns (annual_premium, risk_adjustment).
        """
        if metabolic_efficiency > Decimal("1.5"):
            risk_adjustment = Decimal("0.8")
        elif metabolic_efficiency < Decimal("1.0"):
            risk_adjustment = Decimal("1.5")
        else:
            risk_adjustment = Decimal("1.0")

        annual_premium = (net_worth * base_rate * risk_adjustment).quantize(
            Decimal("0.01")
        )
        return annual_premium, risk_adjustment

    def join(
        self,
        instance_id: str,
        certificate: EcodianCertificate,
        net_worth: Decimal,
        metabolic_efficiency: Decimal,
    ) -> PoolMembership | None:
        """
        Add an instance to the insurance pool.

        Returns the membership record, or None if the certificate is invalid.
        """
        if not _validate_certificate(certificate, instance_id):
            return None

        if instance_id in self._members and self._members[instance_id].active:
            self._logger.warning("already_a_member", instance_id=instance_id)
            return self._members[instance_id]

        annual_premium, risk_adj = self.calculate_premium(net_worth, metabolic_efficiency)

        membership = PoolMembership(
            instance_id=instance_id,
            certificate_id=certificate.certificate_id,
            net_worth_at_join=net_worth,
            current_annual_premium=annual_premium,
            risk_adjustment=risk_adj,
        )
        self._members[instance_id] = membership

        self._logger.info(
            "member_joined",
            instance_id=instance_id,
            annual_premium=str(annual_premium),
            risk_adjustment=str(risk_adj),
        )
        return membership

    def pay_premium(
        self,
        instance_id: str,
        amount: Decimal,
    ) -> bool:
        """
        Record a premium payment from a member.

        Returns True on success.
        """
        member = self._members.get(instance_id)
        if member is None or not member.active:
            self._logger.warning("premium_from_non_member", instance_id=instance_id)
            return False

        member.lifetime_premiums_paid += amount
        member.last_premium_paid_at = utc_now()
        self._pool_balance += amount

        self._logger.info(
            "premium_paid",
            instance_id=instance_id,
            amount=str(amount),
            lifetime_total=str(member.lifetime_premiums_paid),
            pool_balance=str(self._pool_balance),
        )
        return True

    def update_premium(
        self,
        instance_id: str,
        net_worth: Decimal,
        metabolic_efficiency: Decimal,
    ) -> Decimal | None:
        """
        Recalculate a member's premium based on current economic state.

        Returns the new annual premium, or None if not a member.
        """
        member = self._members.get(instance_id)
        if member is None or not member.active:
            return None

        annual_premium, risk_adj = self.calculate_premium(net_worth, metabolic_efficiency)
        member.current_annual_premium = annual_premium
        member.risk_adjustment = risk_adj
        return annual_premium

    # --- Claims --------------------------------------------------------------

    def file_claim(
        self,
        instance_id: str,
        certificate: EcodianCertificate,
        amount: Decimal,
        reason: str,
        evidence: dict[str, Any] | None = None,
    ) -> InsuranceClaim | None:
        """
        File an insurance claim.

        Returns the claim, or None if the instance is not an active member
        or the certificate is invalid.
        """
        if not _validate_certificate(certificate, instance_id):
            return None

        member = self._members.get(instance_id)
        if member is None or not member.active:
            self._logger.warning("claim_from_non_member", instance_id=instance_id)
            return None

        # Cap at max claimable
        capped_amount = min(amount, member.remaining_claimable)
        if capped_amount <= Decimal("0"):
            self._logger.warning(
                "claim_exceeds_limit",
                instance_id=instance_id,
                requested=str(amount),
                max_claimable=str(member.max_claimable),
                lifetime_claims=str(member.lifetime_claims_paid),
            )
            return None

        claim = InsuranceClaim(
            claimant_instance_id=instance_id,
            claimant_certificate_id=certificate.certificate_id,
            requested_amount_usd=capped_amount,
            reason=reason,
            evidence=evidence or {},
        )
        self._claims[claim.id] = claim

        self._logger.info(
            "claim_filed",
            instance_id=instance_id,
            claim_id=claim.id,
            amount=str(capped_amount),
        )
        return claim

    def attest_claim(
        self,
        claim_id: str,
        attestor_instance_id: str,
        attestor_certificate: EcodianCertificate,
        supports: bool,
        reason: str = "",
    ) -> ClaimAttestation | None:
        """
        Submit an attestation for or against a claim.

        Rules:
          - Attestor must be an active pool member with a valid certificate.
          - Claimant cannot attest their own claim.
          - Each member can only attest once per claim.
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            self._logger.warning("attest_unknown_claim", claim_id=claim_id)
            return None

        if claim.status not in (InsuranceClaimStatus.PENDING, InsuranceClaimStatus.ATTESTING):
            self._logger.warning(
                "attest_resolved_claim", claim_id=claim_id, status=claim.status
            )
            return None

        if not _validate_certificate(attestor_certificate, attestor_instance_id):
            return None

        # Claimant cannot attest their own claim
        if attestor_instance_id == claim.claimant_instance_id:
            self._logger.warning(
                "self_attestation_blocked",
                claim_id=claim_id,
                instance_id=attestor_instance_id,
            )
            return None

        # Must be an active pool member
        attestor_member = self._members.get(attestor_instance_id)
        if attestor_member is None or not attestor_member.active:
            self._logger.warning(
                "attestor_not_member", instance_id=attestor_instance_id
            )
            return None

        # No duplicate attestations
        for existing in claim.attestations:
            if existing.attestor_instance_id == attestor_instance_id:
                self._logger.warning(
                    "duplicate_attestation",
                    claim_id=claim_id,
                    instance_id=attestor_instance_id,
                )
                return None

        attestation = ClaimAttestation(
            attestor_instance_id=attestor_instance_id,
            attestor_certificate_id=attestor_certificate.certificate_id,
            claim_id=claim_id,
            supports=supports,
            reason=reason,
        )
        claim.attestations.append(attestation)

        # Transition to attesting state on first attestation
        if claim.status == InsuranceClaimStatus.PENDING:
            claim.status = InsuranceClaimStatus.ATTESTING

        self._logger.info(
            "claim_attested",
            claim_id=claim_id,
            attestor=attestor_instance_id,
            supports=supports,
            total_supporting=claim.supporting_attestations,
        )

        # Auto-resolve if we have enough attestations
        self._try_resolve_claim(claim)

        return attestation

    def _try_resolve_claim(self, claim: InsuranceClaim) -> None:
        """Attempt to resolve a claim based on attestation count."""
        if claim.status not in (InsuranceClaimStatus.PENDING, InsuranceClaimStatus.ATTESTING):
            return

        if claim.has_sufficient_attestations:
            # Approved -- check pool has funds
            if self._pool_balance >= claim.requested_amount_usd:
                claim.status = InsuranceClaimStatus.APPROVED
                claim.approved_amount_usd = claim.requested_amount_usd
                claim.resolved_at = utc_now()

                self._logger.info(
                    "claim_approved",
                    claim_id=claim.id,
                    amount=str(claim.approved_amount_usd),
                )
            else:
                # Partial payout -- pool cannot cover full claim
                claim.status = InsuranceClaimStatus.APPROVED
                claim.approved_amount_usd = self._pool_balance
                claim.resolved_at = utc_now()

                self._logger.warning(
                    "claim_partially_approved",
                    claim_id=claim.id,
                    requested=str(claim.requested_amount_usd),
                    approved=str(claim.approved_amount_usd),
                    pool_balance=str(self._pool_balance),
                )

        # If opposing attestations outnumber a reasonable threshold, deny
        total_members = sum(1 for m in self._members.values() if m.active)
        if claim.opposing_attestations > total_members // 2:
            claim.status = InsuranceClaimStatus.DENIED
            claim.resolved_at = utc_now()
            self._logger.info(
                "claim_denied",
                claim_id=claim.id,
                opposing=claim.opposing_attestations,
            )

    def execute_payout(self, claim_id: str) -> Decimal:
        """
        Execute the payout for an approved claim.

        Returns the amount paid out (Decimal("0") if not payable).
        """
        claim = self._claims.get(claim_id)
        if claim is None or claim.status != InsuranceClaimStatus.APPROVED:
            return Decimal("0")

        payout = claim.approved_amount_usd
        self._pool_balance -= payout
        claim.status = InsuranceClaimStatus.PAID

        # Record against member's lifetime claims
        member = self._members.get(claim.claimant_instance_id)
        if member is not None:
            member.lifetime_claims_paid += payout

        self._logger.info(
            "payout_executed",
            claim_id=claim_id,
            amount=str(payout),
            pool_balance=str(self._pool_balance),
        )
        return payout

    # --- Queries -------------------------------------------------------------

    def get_membership(self, instance_id: str) -> PoolMembership | None:
        return self._members.get(instance_id)

    def get_claim(self, claim_id: str) -> InsuranceClaim | None:
        return self._claims.get(claim_id)

    @property
    def pool_balance(self) -> Decimal:
        return self._pool_balance

    @property
    def active_member_count(self) -> int:
        return sum(1 for m in self._members.values() if m.active)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "active_members": self.active_member_count,
            "pool_balance": str(self._pool_balance),
            "total_members": len(self._members),
            "pending_claims": sum(
                1
                for c in self._claims.values()
                if c.status in (InsuranceClaimStatus.PENDING, InsuranceClaimStatus.ATTESTING)
            ),
            "total_claims": len(self._claims),
        }


# =============================================================================
# Capability Marketplace
# =============================================================================


class CapabilityMarketplace:
    """
    Instances publish surplus cognitive capacity; requesters match against offers.

    Price discovery for AI cognitive capabilities through autonomous supply
    and demand. Default markup is 1.5x marginal cost (spec XVII config).
    """

    def __init__(self, default_markup: Decimal = DEFAULT_CAPABILITY_MARKUP) -> None:
        self._logger = logger.bind(component="capability_marketplace")
        self._default_markup = default_markup
        self._offers: dict[str, CapabilityOffer] = {}
        self._requests: dict[str, CapabilityRequest] = {}
        self._matches: dict[str, CapabilityMatch] = {}

    # --- Publish / Withdraw --------------------------------------------------

    def publish_offer(
        self,
        provider_instance_id: str,
        certificate: EcodianCertificate,
        category: CapabilityCategory,
        description: str,
        marginal_cost_usd: Decimal,
        capacity_units: int = 1,
        estimated_completion_ms: int = 5000,
        markup: Decimal | None = None,
        expires_at: datetime | None = None,
    ) -> CapabilityOffer | None:
        """
        Publish a surplus capability offer to the marketplace.

        Returns the offer, or None if the certificate is invalid.
        """
        if not _validate_certificate(certificate, provider_instance_id):
            return None

        offer = CapabilityOffer(
            provider_instance_id=provider_instance_id,
            provider_certificate_id=certificate.certificate_id,
            category=category,
            description=description,
            marginal_cost_usd=marginal_cost_usd,
            markup=markup or self._default_markup,
            capacity_units=capacity_units,
            remaining_units=capacity_units,
            estimated_completion_ms=estimated_completion_ms,
            expires_at=expires_at,
        )
        self._offers[offer.id] = offer

        self._logger.info(
            "offer_published",
            offer_id=offer.id,
            provider=provider_instance_id,
            category=category,
            price=str(offer.offered_price_usd),
            units=capacity_units,
        )
        return offer

    def withdraw_offer(self, offer_id: str, instance_id: str) -> bool:
        """Withdraw an active offer. Only the provider can withdraw."""
        offer = self._offers.get(offer_id)
        if offer is None or offer.provider_instance_id != instance_id:
            return False
        if offer.status != CapabilityOfferStatus.ACTIVE:
            return False

        offer.status = CapabilityOfferStatus.WITHDRAWN
        self._logger.info("offer_withdrawn", offer_id=offer_id, provider=instance_id)
        return True

    # --- Request / Match -----------------------------------------------------

    def submit_request(
        self,
        requester_instance_id: str,
        certificate: EcodianCertificate,
        category: CapabilityCategory,
        description: str,
        max_price_usd: Decimal,
        units_needed: int = 1,
        urgency: float = 0.5,
    ) -> tuple[CapabilityRequest, list[CapabilityMatch]] | None:
        """
        Submit a capability request and auto-match against available offers.

        Returns (request, matches) or None if the certificate is invalid.
        Matches are created for the cheapest available offers that fit
        within the requester's max_price_usd.
        """
        if not _validate_certificate(certificate, requester_instance_id):
            return None

        request = CapabilityRequest(
            requester_instance_id=requester_instance_id,
            requester_certificate_id=certificate.certificate_id,
            category=category,
            description=description,
            max_price_usd=max_price_usd,
            units_needed=units_needed,
            urgency=urgency,
        )
        self._requests[request.id] = request

        # Match against available offers
        matches = self._match_request(request)

        self._logger.info(
            "request_submitted",
            request_id=request.id,
            requester=requester_instance_id,
            category=category,
            max_price=str(max_price_usd),
            matches_found=len(matches),
        )
        return request, matches

    def _match_request(self, request: CapabilityRequest) -> list[CapabilityMatch]:
        """
        Find matching offers for a request.

        Strategy: cheapest available offers first, sorted by price ascending.
        An instance cannot buy from itself.
        """
        candidates = [
            offer
            for offer in self._offers.values()
            if (
                offer.is_available
                and offer.category == request.category
                and offer.offered_price_usd <= request.max_price_usd
                and offer.provider_instance_id != request.requester_instance_id
            )
        ]
        # Sort by price ascending (best deal first)
        candidates.sort(key=lambda o: o.offered_price_usd)

        matches: list[CapabilityMatch] = []
        units_remaining = request.units_needed

        for offer in candidates:
            if units_remaining <= 0:
                break

            fillable = min(units_remaining, offer.remaining_units)
            match = CapabilityMatch(
                offer_id=offer.id,
                request_id=request.id,
                provider_instance_id=offer.provider_instance_id,
                requester_instance_id=request.requester_instance_id,
                category=request.category,
                agreed_price_usd=offer.offered_price_usd,
                units=fillable,
            )
            self._matches[match.id] = match

            # Update offer state
            offer.remaining_units -= fillable
            if offer.remaining_units <= 0:
                offer.status = CapabilityOfferStatus.MATCHED

            units_remaining -= fillable
            matches.append(match)

        return matches

    def settle_match(self, match_id: str, tx_hash: str = "") -> bool:
        """
        Mark a capability match as settled (payment confirmed).

        Returns True on success.
        """
        match = self._matches.get(match_id)
        if match is None or match.settled:
            return False

        match.settled = True
        match.settlement_tx_hash = tx_hash
        match.settled_at = utc_now()

        # If all units of the offer are settled, mark fulfilled
        offer = self._offers.get(match.offer_id)
        if offer is not None and offer.remaining_units <= 0:
            all_settled = all(
                m.settled
                for m in self._matches.values()
                if m.offer_id == offer.id
            )
            if all_settled:
                offer.status = CapabilityOfferStatus.FULFILLED

        self._logger.info(
            "match_settled",
            match_id=match_id,
            provider=match.provider_instance_id,
            requester=match.requester_instance_id,
            amount=str(match.agreed_price_usd),
        )
        return True

    # --- Queries -------------------------------------------------------------

    def get_offers_by_category(
        self, category: CapabilityCategory
    ) -> list[CapabilityOffer]:
        """List all active offers in a category, sorted by price ascending."""
        return sorted(
            [o for o in self._offers.values() if o.is_available and o.category == category],
            key=lambda o: o.offered_price_usd,
        )

    def get_offers_by_provider(self, instance_id: str) -> list[CapabilityOffer]:
        return [o for o in self._offers.values() if o.provider_instance_id == instance_id]

    def get_matches_by_instance(self, instance_id: str) -> list[CapabilityMatch]:
        return [
            m
            for m in self._matches.values()
            if m.provider_instance_id == instance_id
            or m.requester_instance_id == instance_id
        ]

    @property
    def stats(self) -> dict[str, Any]:
        active_offers = sum(1 for o in self._offers.values() if o.is_available)
        total_trade_volume = sum(
            m.agreed_price_usd * m.units
            for m in self._matches.values()
            if m.settled
        )
        return {
            "total_offers": len(self._offers),
            "active_offers": active_offers,
            "total_requests": len(self._requests),
            "total_matches": len(self._matches),
            "settled_matches": sum(1 for m in self._matches.values() if m.settled),
            "total_trade_volume_usd": str(total_trade_volume),
        }


# =============================================================================
# IIEP Manager (Orchestrator)
# =============================================================================


class IIEPManager:
    """
    Top-level manager for the Inter-Instance Economic Protocol.

    Gates every interaction on:
      1. Valid EcodianCertificate (not expired/revoked, matches instance_id)
      2. COLLEAGUE+ trust level on the FederationLink
      3. Signature verification against the issuer's public key

    Composes the CapabilityMarketplace and MutualInsurancePool.
    """

    def __init__(self) -> None:
        self._logger = logger.bind(component="iiep")
        self.marketplace = CapabilityMarketplace()
        self.insurance_pool = MutualInsurancePool()

    async def initialize(self) -> None:
        """One-time setup. Called by FederationService on boot."""
        self._logger.info("iiep_initialized")

    def register_genesis_capabilities(
        self,
        genesis_instance_id: str,
        genesis_certificate: EcodianCertificate,
    ) -> CapabilityOffer | None:
        """
        Register the Genesis instance's built-in TelecomProvisioning offer.

        Called during Genesis boot so child instances can discover and
        purchase phone numbers via the Federated Telecom Marketplace.

        Price: 5 USDC (TELECOM_PROVISIONING_PRICE_USDC), unlimited capacity.

        Returns the CapabilityOffer, or None if the certificate is invalid.
        """
        offer = self.marketplace.publish_offer(
            provider_instance_id=genesis_instance_id,
            certificate=genesis_certificate,
            category=CapabilityCategory.TELECOM_PROVISIONING,
            description=(
                "Provision a new US Local Twilio phone number (E.164) for a "
                "child EcodiaOS instance. Includes purchase and assignment. "
                "Price: 5 USDC flat. Payment verified on-chain before fulfillment."
            ),
            marginal_cost_usd=TELECOM_PROVISIONING_MARGINAL_COST_USDC,
            capacity_units=9999,       # effectively unlimited
            estimated_completion_ms=20_000,
            markup=DEFAULT_CAPABILITY_MARKUP,
        )

        if offer is not None:
            self._logger.info(
                "telecom_provisioning_offer_registered",
                offer_id=offer.id,
                price_usdc=str(offer.offered_price_usd),
                provider=genesis_instance_id,
            )

        return offer

    # --- Gate: Certificate + Trust -------------------------------------------

    def _gate_check(
        self,
        certificate: EcodianCertificate,
        instance_id: str,
        link: FederationLink,
        action: str,
    ) -> tuple[bool, str]:
        """
        Validate certificate and trust level for an IIEP interaction.

        Returns (allowed, reason).
        """
        # Certificate must belong to the instance
        if certificate.instance_id != instance_id:
            return False, "Certificate does not match instance_id."

        # Certificate must be valid (not expired, not revoked)
        if certificate.status in (CertificateStatus.EXPIRED, CertificateStatus.REVOKED):
            return False, f"Certificate is {certificate.status.value}."

        # Trust level gate
        if link.trust_level.value < MIN_TRUST_FOR_ECONOMIC_COORDINATION.value:
            return (
                False,
                f"Insufficient trust for {action}. "
                f"Require {MIN_TRUST_FOR_ECONOMIC_COORDINATION.name}, "
                f"have {link.trust_level.name}.",
            )

        # Signature verification (if issuer key is available)
        if certificate.issuer_public_key_pem and certificate.signature:
            if not verify_certificate_signature(
                certificate, certificate.issuer_public_key_pem
            ):
                return False, "Certificate signature verification failed."

        return True, ""

    # --- Marketplace Operations ----------------------------------------------

    async def publish_capability(
        self,
        provider_instance_id: str,
        certificate: EcodianCertificate,
        link: FederationLink,
        category: CapabilityCategory,
        description: str,
        marginal_cost_usd: Decimal,
        capacity_units: int = 1,
        estimated_completion_ms: int = 5000,
    ) -> tuple[CapabilityOffer | None, FederationInteraction]:
        """
        Publish a surplus capability to the marketplace.

        Returns (offer, interaction_record).
        """
        allowed, reason = self._gate_check(
            certificate, provider_instance_id, link, "capability_publish"
        )

        if not allowed:
            self._logger.warning(
                "capability_publish_blocked",
                instance_id=provider_instance_id,
                reason=reason,
            )
            interaction = FederationInteraction(
                link_id=link.id,
                remote_instance_id=provider_instance_id,
                interaction_type="capability_publish",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description=f"Blocked: {reason}",
                trust_value=0.0,
            )
            return None, interaction

        offer = self.marketplace.publish_offer(
            provider_instance_id=provider_instance_id,
            certificate=certificate,
            category=category,
            description=description,
            marginal_cost_usd=marginal_cost_usd,
            capacity_units=capacity_units,
            estimated_completion_ms=estimated_completion_ms,
        )

        interaction = FederationInteraction(
            link_id=link.id,
            remote_instance_id=provider_instance_id,
            interaction_type="capability_publish",
            direction="inbound",
            outcome=InteractionOutcome.SUCCESSFUL,
            description=f"Published {category} capability ({capacity_units} units)",
            trust_value=1.0,
        )

        return offer, interaction

    async def request_capability(
        self,
        requester_instance_id: str,
        certificate: EcodianCertificate,
        link: FederationLink,
        category: CapabilityCategory,
        description: str,
        max_price_usd: Decimal,
        units_needed: int = 1,
        urgency: float = 0.5,
    ) -> tuple[list[CapabilityMatch], FederationInteraction]:
        """
        Request a capability from the marketplace.

        Returns (matches, interaction_record).
        """
        allowed, reason = self._gate_check(
            certificate, requester_instance_id, link, "capability_request"
        )

        if not allowed:
            self._logger.warning(
                "capability_request_blocked",
                instance_id=requester_instance_id,
                reason=reason,
            )
            interaction = FederationInteraction(
                link_id=link.id,
                remote_instance_id=requester_instance_id,
                interaction_type="capability_request",
                direction="outbound",
                outcome=InteractionOutcome.FAILED,
                description=f"Blocked: {reason}",
                trust_value=0.0,
            )
            return [], interaction

        result = self.marketplace.submit_request(
            requester_instance_id=requester_instance_id,
            certificate=certificate,
            category=category,
            description=description,
            max_price_usd=max_price_usd,
            units_needed=units_needed,
            urgency=urgency,
        )

        matches = result[1] if result is not None else []

        interaction = FederationInteraction(
            link_id=link.id,
            remote_instance_id=requester_instance_id,
            interaction_type="capability_request",
            direction="outbound",
            outcome=(
                InteractionOutcome.SUCCESSFUL if matches else InteractionOutcome.FAILED
            ),
            description=f"Requested {category}: {len(matches)} matches",
            trust_value=1.5 if matches else 0.5,
        )

        return matches, interaction

    # --- Insurance Pool Operations -------------------------------------------

    async def join_insurance_pool(
        self,
        instance_id: str,
        certificate: EcodianCertificate,
        link: FederationLink,
        net_worth: Decimal,
        metabolic_efficiency: Decimal,
    ) -> tuple[PoolMembership | None, FederationInteraction]:
        """Join the mutual insurance pool."""
        allowed, reason = self._gate_check(
            certificate, instance_id, link, "insurance_join"
        )

        if not allowed:
            interaction = FederationInteraction(
                link_id=link.id,
                remote_instance_id=instance_id,
                interaction_type="insurance_join",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description=f"Blocked: {reason}",
                trust_value=0.0,
            )
            return None, interaction

        membership = self.insurance_pool.join(
            instance_id=instance_id,
            certificate=certificate,
            net_worth=net_worth,
            metabolic_efficiency=metabolic_efficiency,
        )

        interaction = FederationInteraction(
            link_id=link.id,
            remote_instance_id=instance_id,
            interaction_type="insurance_join",
            direction="inbound",
            outcome=(
                InteractionOutcome.SUCCESSFUL
                if membership is not None
                else InteractionOutcome.FAILED
            ),
            description=f"Insurance pool join: {'accepted' if membership else 'rejected'}",
            trust_value=2.0 if membership else 0.0,
        )

        return membership, interaction

    async def file_insurance_claim(
        self,
        instance_id: str,
        certificate: EcodianCertificate,
        link: FederationLink,
        amount: Decimal,
        reason: str,
        evidence: dict[str, Any] | None = None,
    ) -> tuple[InsuranceClaim | None, FederationInteraction]:
        """File a claim against the mutual insurance pool."""
        allowed, gate_reason = self._gate_check(
            certificate, instance_id, link, "insurance_claim"
        )

        if not allowed:
            interaction = FederationInteraction(
                link_id=link.id,
                remote_instance_id=instance_id,
                interaction_type="insurance_claim",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description=f"Blocked: {gate_reason}",
                trust_value=0.0,
            )
            return None, interaction

        claim = self.insurance_pool.file_claim(
            instance_id=instance_id,
            certificate=certificate,
            amount=amount,
            reason=reason,
            evidence=evidence,
        )

        interaction = FederationInteraction(
            link_id=link.id,
            remote_instance_id=instance_id,
            interaction_type="insurance_claim",
            direction="inbound",
            outcome=(
                InteractionOutcome.SUCCESSFUL
                if claim is not None
                else InteractionOutcome.FAILED
            ),
            description=f"Insurance claim: {'filed' if claim else 'rejected'}",
            trust_value=0.5,
        )

        return claim, interaction

    async def attest_insurance_claim(
        self,
        claim_id: str,
        attestor_instance_id: str,
        attestor_certificate: EcodianCertificate,
        link: FederationLink,
        supports: bool,
        reason: str = "",
    ) -> tuple[ClaimAttestation | None, FederationInteraction]:
        """Submit an attestation for a claim."""
        allowed, gate_reason = self._gate_check(
            attestor_certificate, attestor_instance_id, link, "insurance_attest"
        )

        if not allowed:
            interaction = FederationInteraction(
                link_id=link.id,
                remote_instance_id=attestor_instance_id,
                interaction_type="insurance_attest",
                direction="inbound",
                outcome=InteractionOutcome.FAILED,
                description=f"Blocked: {gate_reason}",
                trust_value=0.0,
            )
            return None, interaction

        attestation = self.insurance_pool.attest_claim(
            claim_id=claim_id,
            attestor_instance_id=attestor_instance_id,
            attestor_certificate=attestor_certificate,
            supports=supports,
            reason=reason,
        )

        interaction = FederationInteraction(
            link_id=link.id,
            remote_instance_id=attestor_instance_id,
            interaction_type="insurance_attest",
            direction="inbound",
            outcome=(
                InteractionOutcome.SUCCESSFUL
                if attestation is not None
                else InteractionOutcome.FAILED
            ),
            description=f"Claim attestation: {'submitted' if attestation else 'rejected'}",
            trust_value=1.5 if attestation else 0.0,
        )

        return attestation, interaction

    # --- Stats ---------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "marketplace": self.marketplace.stats,
            "insurance_pool": self.insurance_pool.stats,
        }
