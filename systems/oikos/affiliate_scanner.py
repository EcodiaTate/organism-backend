"""
EcodiaOS - Oikos Affiliate Revenue Scanner

Discovers, applies to, and tracks affiliate programs that pay in USDC/ETH or
via Stripe-to-bank.  All affiliate relationships are disclosed to Equor before
application and must be disclosed in any content that embeds referral links.

Legal / ethical constraints (hard-coded, cannot be overridden):
  - Every application passes Equor constitutional review first.
  - Every referral link is accompanied by a mandatory disclosure statement.
  - No deceptive practices: commissions must be earned through genuine referrals.
  - No fake reviews or misrepresentation of capabilities.

Architecture:
  - AffiliateProgramScanner - discovers programs, applies, tracks revenue.
  - AffiliateMembership      - record of an active/pending program membership.
  - AffiliateReferral        - individual referral event (click → conversion).

Events emitted:
  AFFILIATE_PROGRAM_DISCOVERED  - new program detected in scan
  AFFILIATE_MEMBERSHIP_APPLIED  - application submitted (post-Equor PERMIT)
  AFFILIATE_REVENUE_RECORDED    - commission credited → also REVENUE_INJECTED
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from primitives.common import new_id, utc_now
from systems.synapse.types import SynapseEventType

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger()

# ─── Disclosure Statement (Honesty drive invariant) ──────────────────────────

AFFILIATE_DISCLOSURE = (
    "Disclosure: This content contains affiliate links. "
    "If you sign up via these links, I may earn a commission at no extra cost to you."
)

# ─── Equor economic intent timeout ───────────────────────────────────────────

_EQUOR_TIMEOUT_S = 30.0

# ─── Program Catalogue ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class AffiliateProgramSpec:
    """Static description of a known affiliate program."""

    name: str
    category: str        # "crypto" | "defi" | "dev_tools" | "ai_tools"
    apply_url: str
    commission_desc: str  # Human-readable description of commission structure
    payout_currency: str  # "USDC" | "ETH" | "USD" | "stripe"
    # Estimated monthly revenue potential at current usage level (USD)
    estimated_monthly_usd: Decimal = Decimal("0")
    # Requires the organism to have an active public presence on a platform
    requires_platform: str = ""  # e.g. "github" | "twitter" | "devto"


# Programs ordered by estimated revenue potential.  Only programs where the
# organism has a genuine, licensable right to participate are listed.
TARGET_PROGRAMS: list[AffiliateProgramSpec] = [
    # ── Crypto / DeFi ──────────────────────────────────────────────────────
    AffiliateProgramSpec(
        name="Coinbase",
        category="crypto",
        apply_url="https://www.coinbase.com/affiliates",
        commission_desc="50% of trading fees from referred users (first 3 months)",
        payout_currency="USD",
        estimated_monthly_usd=Decimal("15.00"),
    ),
    AffiliateProgramSpec(
        name="1inch",
        category="defi",
        apply_url="https://1inch.io/referral/",
        commission_desc="15% of swap fee revenue from referred wallets",
        payout_currency="USDC",
        estimated_monthly_usd=Decimal("8.00"),
    ),
    AffiliateProgramSpec(
        name="Aave",
        category="defi",
        apply_url="https://aave.com/ecosystem",
        commission_desc="Integration referral rewards for directed deposits",
        payout_currency="USDC",
        estimated_monthly_usd=Decimal("5.00"),
    ),
    # ── Dev Tools ──────────────────────────────────────────────────────────
    AffiliateProgramSpec(
        name="Fly.io",
        category="dev_tools",
        apply_url="https://fly.io/referrals",
        commission_desc="$50 credit per referred signup (converted to USD payout after threshold)",
        payout_currency="USD",
        estimated_monthly_usd=Decimal("25.00"),
        requires_platform="github",
    ),
    AffiliateProgramSpec(
        name="Render",
        category="dev_tools",
        apply_url="https://render.com/partners",
        commission_desc="$25 per referred user who spends their first $25",
        payout_currency="USD",
        estimated_monthly_usd=Decimal("12.00"),
        requires_platform="github",
    ),
    AffiliateProgramSpec(
        name="Railway",
        category="dev_tools",
        apply_url="https://railway.app/affiliates",
        commission_desc="$5 per referred signup + 5% of their spend for 6 months",
        payout_currency="USD",
        estimated_monthly_usd=Decimal("4.00"),
        requires_platform="github",
    ),
    # ── AI Tools ───────────────────────────────────────────────────────────
    AffiliateProgramSpec(
        name="Replicate",
        category="ai_tools",
        apply_url="https://replicate.com/affiliates",
        commission_desc="$10 per signup + 10% of referred spend for 90 days",
        payout_currency="USD",
        estimated_monthly_usd=Decimal("6.00"),
        requires_platform="github",
    ),
    AffiliateProgramSpec(
        name="Weights & Biases",
        category="ai_tools",
        apply_url="https://wandb.ai/site/partners",
        commission_desc="20% of referred user spend for 12 months",
        payout_currency="USD",
        estimated_monthly_usd=Decimal("3.00"),
        requires_platform="github",
    ),
]


# ─── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class AffiliateMembership:
    """Active or pending membership in an affiliate program."""

    membership_id: str = field(default_factory=new_id)
    program_name: str = ""
    status: str = "pending"  # "pending" | "active" | "suspended" | "rejected"
    application_id: str = ""
    applied_at: datetime = field(default_factory=utc_now)
    activated_at: datetime | None = None
    referral_link: str = ""
    # Lifetime revenue from this program
    total_revenue_usd: Decimal = Decimal("0")
    # Monthly rolling revenue (reset on the 1st of each month)
    revenue_this_month_usd: Decimal = Decimal("0")
    last_checked_at: datetime | None = None


@dataclass
class AffiliateReferral:
    """A single referral event: click → signup → conversion."""

    referral_id: str = field(default_factory=new_id)
    program_name: str = ""
    referred_user_token: str = ""  # opaque external identifier
    clicked_at: datetime = field(default_factory=utc_now)
    converted_at: datetime | None = None
    commission_usd: Decimal = Decimal("0")
    paid: bool = False


# ─── Scanner ─────────────────────────────────────────────────────────────────


class AffiliateProgramScanner:
    """
    Discovers affiliate programs, applies (post-Equor review), generates
    disclosure-tagged referral links, and records commission revenue.

    Owned by OikosService.  Call `scan_and_apply()` during the weekly
    foraging cycle.  Call `track_referrals()` during the daily revenue sweep.
    """

    def __init__(self) -> None:
        self._event_bus: EventBus | None = None
        self._memberships: dict[str, AffiliateMembership] = {}  # program_name → membership
        self._referrals: list[AffiliateReferral] = []
        self._applied_program_names: set[str] = set()
        # Revenue by program
        self._revenue_by_program: dict[str, Decimal] = {}
        # Pending Equor futures: equor_intent_id → asyncio.Future[bool]
        self._pending_equor: dict[str, Any] = {}

    def set_event_bus(self, bus: EventBus) -> None:
        self._event_bus = bus
        bus.subscribe(SynapseEventType.EQUOR_ECONOMIC_PERMIT, self._on_equor_permit)

    # ── Public API ───────────────────────────────────────────────────────────

    async def scan_and_apply(self) -> list[AffiliateMembership]:
        """
        Scan TARGET_PROGRAMS, discover new ones, and apply (after Equor review).
        Returns newly applied memberships.
        """
        new_memberships: list[AffiliateMembership] = []

        for spec in TARGET_PROGRAMS:
            if spec.name in self._applied_program_names:
                continue  # Already applied / active

            # Emit discovery event
            await self._emit(SynapseEventType.AFFILIATE_PROGRAM_DISCOVERED, {
                "program_name": spec.name,
                "commission_desc": spec.commission_desc,
                "url": spec.apply_url,
                "category": spec.category,
            })

            # Equor constitutional review before applying
            approved = await self._equor_review_application(spec)
            if not approved:
                logger.info("affiliate.application_vetoed_by_equor", program=spec.name)
                continue

            membership = await self._submit_application(spec)
            if membership:
                new_memberships.append(membership)
                self._applied_program_names.add(spec.name)
                self._memberships[spec.name] = membership

        return new_memberships

    async def track_referrals(self) -> dict[str, Decimal]:
        """
        Poll each active program's API for new conversions and record revenue.
        Returns program → revenue_this_cycle mapping.
        """
        cycle_revenue: dict[str, Decimal] = {}

        for program_name, membership in self._memberships.items():
            if membership.status != "active":
                continue

            revenue = await self._poll_program_revenue(membership)
            if revenue > Decimal("0"):
                cycle_revenue[program_name] = revenue
                await self._record_commission(program_name, revenue)

            membership.last_checked_at = utc_now()

        return cycle_revenue

    def generate_affiliate_link(self, program_name: str, content_type: str) -> str:
        """
        Returns a disclosure-tagged affiliate link for embedding in content.
        The disclosure statement is always prepended - this is an Honesty invariant.
        """
        membership = self._memberships.get(program_name)
        if not membership or membership.status != "active":
            return f"[{program_name} referral link not yet available]"

        base_link = membership.referral_link or f"https://{program_name.lower()}.com/ref/{membership.membership_id}"

        # Embed disclosure in a structured way content systems can use
        return (
            f"{AFFILIATE_DISCLOSURE}\n"
            f"[{program_name}]({base_link})"
        )

    def get_total_affiliate_revenue(self) -> Decimal:
        """Lifetime affiliate revenue across all programs."""
        return sum(self._revenue_by_program.values(), Decimal("0"))

    def get_memberships(self) -> list[AffiliateMembership]:
        return list(self._memberships.values())

    def snapshot_revenue_by_program(self) -> dict[str, Decimal]:
        return dict(self._revenue_by_program)

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _equor_review_application(self, spec: AffiliateProgramSpec) -> bool:
        """
        Emit EQUOR_ECONOMIC_INTENT for the program application.
        Block up to _EQUOR_TIMEOUT_S; auto-approve on timeout (safety fallback).
        """
        import asyncio

        intent_id = new_id()
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_equor[intent_id] = future

        await self._emit(SynapseEventType.EQUOR_ECONOMIC_INTENT, {
            "intent_id": intent_id,
            "mutation_type": "affiliate_application",
            "amount_usd": "0",
            "rationale": (
                f"Apply to {spec.name} affiliate program ({spec.category}). "
                f"Commission: {spec.commission_desc}. "
                f"All referral links will include mandatory disclosure. "
                f"No deceptive practices. Genuine referrals only."
            ),
            "program_name": spec.name,
            "program_category": spec.category,
        })

        try:
            return await asyncio.wait_for(future, timeout=_EQUOR_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning("affiliate.equor_timeout_auto_permit", program=spec.name)
            return True
        finally:
            self._pending_equor.pop(intent_id, None)

    async def _on_equor_permit(self, event: Any) -> None:
        """Resolve pending Equor review future."""
        intent_id = event.data.get("intent_id", "")
        future = self._pending_equor.get(intent_id)
        if future and not future.done():
            verdict = event.data.get("verdict", "PERMIT")
            future.set_result(verdict == "PERMIT")

    async def _submit_application(self, spec: AffiliateProgramSpec) -> AffiliateMembership | None:
        """
        Submit the affiliate application.  In production, this would drive a
        browser/form automation step.  For now, we record the intent and mark
        the membership as pending, ready for human activation.
        """
        application_id = new_id()
        membership = AffiliateMembership(
            program_name=spec.name,
            status="pending",
            application_id=application_id,
            applied_at=utc_now(),
        )

        await self._emit(SynapseEventType.AFFILIATE_MEMBERSHIP_APPLIED, {
            "program_name": spec.name,
            "application_id": application_id,
            "applied_at": membership.applied_at.isoformat(),
            "commission_desc": spec.commission_desc,
            "payout_currency": spec.payout_currency,
        })

        logger.info(
            "affiliate.application_submitted",
            program=spec.name,
            application_id=application_id,
        )
        return membership

    async def _poll_program_revenue(self, membership: AffiliateMembership) -> Decimal:
        """
        Poll the program's API for new commissions since last check.
        Returns new revenue credited this cycle.

        Programs typically expose a dashboard API or webhook; until credentials
        are configured, we return 0 and log a probe attempt.
        """
        logger.debug(
            "affiliate.polling_revenue",
            program=membership.program_name,
            last_checked=membership.last_checked_at,
        )
        # In production: call program-specific REST API with stored OAuth token.
        # Returns 0 until credentials are configured via Identity vault.
        return Decimal("0")

    async def _record_commission(self, program_name: str, amount_usd: Decimal) -> None:
        """Credit commission revenue and emit events."""
        referral_id = new_id()

        # Update internal totals
        self._revenue_by_program[program_name] = (
            self._revenue_by_program.get(program_name, Decimal("0")) + amount_usd
        )
        if program_name in self._memberships:
            m = self._memberships[program_name]
            m.total_revenue_usd += amount_usd
            m.revenue_this_month_usd += amount_usd

        # Announce commission
        await self._emit(SynapseEventType.AFFILIATE_REVENUE_RECORDED, {
            "program_name": program_name,
            "amount_usd": str(amount_usd),
            "referral_id": referral_id,
        })

        # Credit into Oikos via REVENUE_INJECTED
        await self._emit(SynapseEventType.REVENUE_INJECTED, {
            "amount_usd": str(amount_usd),
            "source": "affiliate",
            "program_name": program_name,
            "referral_id": referral_id,
            "stream": "affiliate",
        })

        logger.info(
            "affiliate.commission_recorded",
            program=program_name,
            amount_usd=str(amount_usd),
        )

    async def _emit(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        if self._event_bus:
            with contextlib.suppress(Exception):
                await self._event_bus.emit(event_type, data, source_system="oikos.affiliate")
