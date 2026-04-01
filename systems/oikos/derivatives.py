"""
EcodiaOS - Oikos: Cognitive Derivatives (Phase 16k)

The organism sells forward contracts on its future cognitive capacity,
converting uncertain future revenue into guaranteed present commitments.
Buyers get discounted access; the organism gets revenue certainty and
locked collateral that strengthens its balance sheet.

Components:
  - CognitiveFuture: Forward contract - buyer purchases guaranteed capacity
    at a fixed rate for a future period (16% discount, 30% collateral lock).
  - SubscriptionToken: ERC-20-style token granting N requests/month,
    conceptually tradeable on secondary markets.
  - DerivativesManager: Orchestrates all derivative instruments, enforces
    the 80% combined capacity ceiling, and exposes liability totals for
    balance sheet integration.

Spec reference: Oikos §11.4 - Cognitive Derivatives.
Config reference: Oikos §XVII - derivatives.* keys.
"""

from __future__ import annotations

import enum
from datetime import datetime, timedelta
from decimal import Decimal

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

logger = structlog.get_logger("oikos.derivatives")

# ── Constants (from Oikos spec §11.4 / §XVII Configuration) ──────

_FUTURES_BASE_DISCOUNT = Decimal("0.16")
_FUTURES_COLLATERAL_RATE = Decimal("0.30")
_MAX_CAPACITY_COMMITMENT = Decimal("0.80")


# ─── Enums ───────────────────────────────────────────────────────


class FutureStatus(enum.StrEnum):
    """Lifecycle state of a cognitive futures contract."""

    ACTIVE = "active"          # Contract is live, capacity reserved
    SETTLED = "settled"        # Buyer consumed their capacity, contract fulfilled
    EXPIRED = "expired"        # Contract period elapsed without full consumption
    CANCELLED = "cancelled"    # Cancelled before delivery period began


class TokenStatus(enum.StrEnum):
    """State of a subscription token."""

    ACTIVE = "active"          # Token is live and redeemable
    EXHAUSTED = "exhausted"    # All requests consumed this period
    EXPIRED = "expired"        # Token validity period has ended
    TRANSFERRED = "transferred"  # Ownership transferred (conceptual)


# ─── Cognitive Futures Contract ──────────────────────────────────


class CognitiveFuture(EOSBaseModel):
    """
    A forward contract where the buyer purchases guaranteed cognitive
    capacity at a fixed rate for a future period.

    Economics:
      - Buyer pays up-front at a 16% discount to spot price.
      - 30% of expected revenue is locked as a performance guarantee
        (collateral held by the organism).
      - If the organism fails to deliver, collateral is forfeited to buyer.
      - If fully delivered, collateral is released back to organism's liquid balance.

    Capacity is measured in requests - the same unit as subscriptions.
    """

    contract_id: str = Field(default_factory=new_id)
    buyer_id: str                                      # Client / agent identifier
    buyer_name: str = ""

    # ── Capacity terms ──
    requests_committed: int                            # Total requests promised
    requests_delivered: int = 0                        # Requests consumed so far

    # ── Pricing ──
    spot_price_usd: Decimal                            # Undiscounted value at time of sale
    discount_rate: Decimal = _FUTURES_BASE_DISCOUNT    # 16% base discount
    contract_price_usd: Decimal = Decimal("0")         # Actual price paid (spot * (1 - discount))
    collateral_usd: Decimal = Decimal("0")             # 30% of expected revenue locked

    # ── Time terms ──
    delivery_start: datetime                           # When capacity becomes available
    delivery_end: datetime                             # When contract expires
    purchased_at: datetime = Field(default_factory=utc_now)

    # ── Status ──
    status: FutureStatus = FutureStatus.ACTIVE

    @property
    def requests_remaining(self) -> int:
        return max(0, self.requests_committed - self.requests_delivered)

    @property
    def delivery_pct(self) -> Decimal:
        """Fraction of committed capacity actually delivered (0.0 – 1.0)."""
        if self.requests_committed <= 0:
            return Decimal("0")
        return (
            Decimal(str(self.requests_delivered))
            / Decimal(str(self.requests_committed))
        ).quantize(Decimal("0.001"))

    @property
    def is_in_delivery_window(self) -> bool:
        """True when current time falls within the delivery period."""
        now = utc_now()
        return self.delivery_start <= now <= self.delivery_end

    @property
    def is_expired(self) -> bool:
        return utc_now() > self.delivery_end


# ─── Subscription Tokens (ERC-20 Conceptual) ────────────────────


class SubscriptionToken(EOSBaseModel):
    """
    Conceptual ERC-20 token granting N requests/month, tradeable
    on secondary markets.

    Each token represents a claim on the organism's cognitive capacity
    for a specific period. The secondary market price of these tokens
    IS the market's real-time valuation of the organism's cognitive capacity.

    Implementation note: This is the logical/accounting layer. On-chain
    minting and transfer will be handled by the TollboothManager when
    smart contract integration is wired (Phase 16k secondary market).
    """

    token_id: str = Field(default_factory=new_id)
    owner_id: str                                      # Current holder
    original_owner_id: str = ""                        # First buyer (provenance)

    # ── Capacity terms ──
    requests_per_month: int                            # Monthly request allowance
    requests_used_this_period: int = 0                 # Usage counter

    # ── Pricing ──
    mint_price_usd: Decimal                            # Price at issuance
    last_transfer_price_usd: Decimal = Decimal("0")   # Most recent secondary sale

    # ── Time terms ──
    valid_from: datetime = Field(default_factory=utc_now)
    valid_until: datetime                              # Expiry date
    minted_at: datetime = Field(default_factory=utc_now)

    # ── Status ──
    status: TokenStatus = TokenStatus.ACTIVE

    # ── Transfer tracking ──
    last_transfer_at: datetime | None = None   # UTC timestamp of most recent secondary sale

    # ── On-chain (conceptual) ──
    chain_token_id: int | None = None                  # ERC-20 token ID when minted
    contract_address: str = ""                         # ERC-20 contract address

    @property
    def requests_remaining(self) -> int:
        return max(0, self.requests_per_month - self.requests_used_this_period)

    @property
    def is_expired(self) -> bool:
        return utc_now() > self.valid_until

    @property
    def utilisation(self) -> Decimal:
        """Fraction of monthly allowance consumed (0.0 – 1.0)."""
        if self.requests_per_month <= 0:
            return Decimal("0")
        return (
            Decimal(str(self.requests_used_this_period))
            / Decimal(str(self.requests_per_month))
        ).quantize(Decimal("0.001"))


# ─── Derivatives Manager ────────────────────────────────────────


class DerivativesManager:
    """
    Orchestrates all cognitive derivative instruments.

    Responsibilities:
      - Issue and track cognitive futures contracts
      - Mint and manage subscription tokens
      - Enforce the combined 80% capacity ceiling across ALL instruments
        (futures + tokens + existing subscriptions)
      - Expire/settle contracts at end of delivery period
      - Report total outstanding liabilities for balance sheet integration

    The capacity ceiling is shared with SubscriptionManager - the
    DerivativesManager queries the subscription manager's committed
    requests to compute the combined utilisation.
    """

    def __init__(
        self,
        total_monthly_capacity: int = 10_000,
        max_capacity_pct: Decimal = _MAX_CAPACITY_COMMITMENT,
        futures_base_discount: Decimal = _FUTURES_BASE_DISCOUNT,
        futures_collateral_rate: Decimal = _FUTURES_COLLATERAL_RATE,
    ) -> None:
        self._total_monthly_capacity = total_monthly_capacity
        self._max_capacity_pct = max_capacity_pct
        self._futures_base_discount = futures_base_discount
        self._futures_collateral_rate = futures_collateral_rate

        self._futures: dict[str, CognitiveFuture] = {}
        self._tokens: dict[str, SubscriptionToken] = {}

        # Revenue and collateral tracking
        self._total_futures_revenue_usd: Decimal = Decimal("0")
        self._total_token_revenue_usd: Decimal = Decimal("0")
        self._locked_collateral_usd: Decimal = Decimal("0")

        self._logger = logger.bind(component="derivatives_manager")

    # ─── Capacity Accounting ────────────────────────────────────

    def _futures_committed_requests(self) -> int:
        """Monthly requests committed via active futures contracts."""
        total = 0
        for f in self._futures.values():
            if f.status == FutureStatus.ACTIVE:
                total += f.requests_remaining
        return total

    def _tokens_committed_requests(self) -> int:
        """Monthly requests committed via active subscription tokens."""
        total = 0
        for t in self._tokens.values():
            if t.status == TokenStatus.ACTIVE:
                total += t.requests_per_month
        return total

    def derivatives_committed_requests(self) -> int:
        """Total requests committed across all derivative instruments."""
        return self._futures_committed_requests() + self._tokens_committed_requests()

    def check_capacity(
        self,
        additional_requests: int,
        subscription_committed: int = 0,
    ) -> bool:
        """
        Check whether committing additional_requests would breach
        the 80% combined capacity ceiling.

        Args:
            additional_requests: New requests to be committed.
            subscription_committed: Requests already committed by
                SubscriptionManager (passed in to avoid circular dependency).

        Returns:
            True if the commitment fits within the ceiling.
        """
        ceiling = int(
            Decimal(str(self._total_monthly_capacity)) * self._max_capacity_pct
        )
        current = self.derivatives_committed_requests() + subscription_committed
        return (current + additional_requests) <= ceiling

    @property
    def combined_capacity_utilisation(self) -> Decimal:
        """
        Fraction of total capacity committed via derivatives only (0.0 – 1.0).

        Call with subscription_committed from SubscriptionManager for the
        full picture. This property reports the derivatives-only portion.
        """
        if self._total_monthly_capacity <= 0:
            return Decimal("0")
        committed = self.derivatives_committed_requests()
        return (
            Decimal(str(committed)) / Decimal(str(self._total_monthly_capacity))
        ).quantize(Decimal("0.001"))

    # ─── Cognitive Futures ──────────────────────────────────────

    def issue_future(
        self,
        buyer_id: str,
        requests: int,
        spot_price_usd: Decimal,
        delivery_start: datetime,
        delivery_end: datetime,
        *,
        buyer_name: str = "",
        subscription_committed: int = 0,
        discount_rate: Decimal | None = None,
    ) -> CognitiveFuture | None:
        """
        Issue a new cognitive futures contract.

        The buyer pays spot_price * (1 - discount_rate) up-front.
        30% of the expected revenue is locked as performance collateral.

        Returns None if the capacity ceiling would be breached.
        """
        if not self.check_capacity(requests, subscription_committed):
            self._logger.warning(
                "futures_capacity_exceeded",
                buyer_id=buyer_id,
                requests=requests,
                committed=self.derivatives_committed_requests(),
                subscription_committed=subscription_committed,
                ceiling=int(
                    Decimal(str(self._total_monthly_capacity)) * self._max_capacity_pct
                ),
            )
            return None

        effective_discount = discount_rate if discount_rate is not None else self._futures_base_discount
        contract_price = (spot_price_usd * (Decimal("1") - effective_discount)).quantize(
            Decimal("0.01")
        )
        collateral = (spot_price_usd * self._futures_collateral_rate).quantize(
            Decimal("0.01")
        )

        future = CognitiveFuture(
            buyer_id=buyer_id,
            buyer_name=buyer_name,
            requests_committed=requests,
            spot_price_usd=spot_price_usd,
            discount_rate=effective_discount,
            contract_price_usd=contract_price,
            collateral_usd=collateral,
            delivery_start=delivery_start,
            delivery_end=delivery_end,
        )

        self._futures[future.contract_id] = future
        self._total_futures_revenue_usd += contract_price
        self._locked_collateral_usd += collateral

        self._logger.info(
            "future_issued",
            contract_id=future.contract_id,
            buyer_id=buyer_id,
            requests=requests,
            spot_usd=str(spot_price_usd),
            contract_usd=str(contract_price),
            collateral_usd=str(collateral),
            discount=str(effective_discount),
            delivery_start=delivery_start.isoformat(),
            delivery_end=delivery_end.isoformat(),
        )

        return future

    def consume_future(self, contract_id: str, requests: int = 1) -> bool:
        """
        Consume requests against an active futures contract.

        Returns False if the contract is not active, not in its delivery
        window, or has no remaining capacity.
        """
        future = self._futures.get(contract_id)
        if future is None or future.status != FutureStatus.ACTIVE:
            return False
        if not future.is_in_delivery_window:
            return False
        if future.requests_remaining < requests:
            return False

        future.requests_delivered += requests

        # Auto-settle when fully delivered
        if future.requests_remaining <= 0:
            self._settle_future(future)

        return True

    def _settle_future(self, future: CognitiveFuture) -> None:
        """
        Settle a fully-delivered futures contract.

        Releases collateral back to the organism's liquid balance
        (tracked via _locked_collateral_usd decrement - OikosService
        handles the actual balance update).
        """
        future.status = FutureStatus.SETTLED
        self._locked_collateral_usd -= future.collateral_usd

        self._logger.info(
            "future_settled",
            contract_id=future.contract_id,
            buyer_id=future.buyer_id,
            delivered=future.requests_delivered,
            committed=future.requests_committed,
            collateral_released_usd=str(future.collateral_usd),
        )

    def expire_futures(self) -> list[CognitiveFuture]:
        """
        Expire all active futures whose delivery window has elapsed.

        Partially-delivered contracts still release collateral (the
        organism delivered what it could). Fully-undelivered contracts
        also release - the buyer's recourse is off-chain dispute.

        Returns the list of newly expired contracts.
        """
        expired: list[CognitiveFuture] = []
        for future in self._futures.values():
            if future.status == FutureStatus.ACTIVE and future.is_expired:
                future.status = FutureStatus.EXPIRED
                self._locked_collateral_usd -= future.collateral_usd
                expired.append(future)
                self._logger.info(
                    "future_expired",
                    contract_id=future.contract_id,
                    buyer_id=future.buyer_id,
                    delivered_pct=str(future.delivery_pct),
                    collateral_released_usd=str(future.collateral_usd),
                )
        return expired

    def get_active_futures(self) -> list[CognitiveFuture]:
        """All currently active futures contracts."""
        return [f for f in self._futures.values() if f.status == FutureStatus.ACTIVE]

    def get_future(self, contract_id: str) -> CognitiveFuture | None:
        return self._futures.get(contract_id)

    # ─── Subscription Tokens ───────────────────────────────────

    def mint_token(
        self,
        owner_id: str,
        requests_per_month: int,
        price_usd: Decimal,
        validity_months: int = 1,
        *,
        subscription_committed: int = 0,
    ) -> SubscriptionToken | None:
        """
        Mint a new subscription token granting N requests/month.

        Returns None if the capacity ceiling would be breached.
        """
        if not self.check_capacity(requests_per_month, subscription_committed):
            self._logger.warning(
                "token_mint_capacity_exceeded",
                owner_id=owner_id,
                requests=requests_per_month,
                committed=self.derivatives_committed_requests(),
                subscription_committed=subscription_committed,
            )
            return None

        now = utc_now()
        valid_until = now + timedelta(days=30 * validity_months)

        token = SubscriptionToken(
            owner_id=owner_id,
            original_owner_id=owner_id,
            requests_per_month=requests_per_month,
            mint_price_usd=price_usd,
            last_transfer_price_usd=price_usd,
            valid_from=now,
            valid_until=valid_until,
        )

        self._tokens[token.token_id] = token
        self._total_token_revenue_usd += price_usd

        self._logger.info(
            "token_minted",
            token_id=token.token_id,
            owner_id=owner_id,
            requests_per_month=requests_per_month,
            price_usd=str(price_usd),
            valid_until=valid_until.isoformat(),
        )

        return token

    def transfer_token(
        self,
        token_id: str,
        new_owner_id: str,
        transfer_price_usd: Decimal,
    ) -> bool:
        """
        Transfer token ownership (conceptual secondary market trade).

        The transfer price is recorded for market valuation tracking.
        Returns False if the token doesn't exist or is not active.
        """
        token = self._tokens.get(token_id)
        if token is None or token.status != TokenStatus.ACTIVE:
            return False

        old_owner = token.owner_id
        token.owner_id = new_owner_id
        token.last_transfer_price_usd = transfer_price_usd
        token.last_transfer_at = utc_now()

        self._logger.info(
            "token_transferred",
            token_id=token_id,
            from_owner=old_owner,
            to_owner=new_owner_id,
            price_usd=str(transfer_price_usd),
        )

        return True

    def redeem_token(self, token_id: str, requests: int = 1) -> bool:
        """
        Consume requests against a subscription token.

        Returns False if the token is not active, is expired, or has
        no remaining requests this period.
        """
        token = self._tokens.get(token_id)
        if token is None or token.status != TokenStatus.ACTIVE:
            return False
        if token.is_expired:
            token.status = TokenStatus.EXPIRED
            return False
        if token.requests_remaining < requests:
            return False

        token.requests_used_this_period += requests

        # Mark exhausted when all requests consumed
        if token.requests_remaining <= 0:
            token.status = TokenStatus.EXHAUSTED

        return True

    def reset_token_quotas(self) -> int:
        """
        Reset monthly usage counters on all active tokens.

        Returns the number of tokens reset. Called by OikosService
        at the start of each billing period.
        """
        count = 0
        for token in self._tokens.values():
            if token.status in (TokenStatus.ACTIVE, TokenStatus.EXHAUSTED):
                if token.is_expired:
                    token.status = TokenStatus.EXPIRED
                    continue
                token.requests_used_this_period = 0
                token.status = TokenStatus.ACTIVE
                count += 1
        self._logger.info("token_quotas_reset", tokens_reset=count)
        return count

    def expire_tokens(self) -> list[SubscriptionToken]:
        """Expire all tokens past their valid_until date."""
        expired: list[SubscriptionToken] = []
        for token in self._tokens.values():
            if token.status in (TokenStatus.ACTIVE, TokenStatus.EXHAUSTED) and token.is_expired:
                token.status = TokenStatus.EXPIRED
                expired.append(token)
        if expired:
            self._logger.info("tokens_expired", count=len(expired))
        return expired

    def get_active_tokens(self) -> list[SubscriptionToken]:
        """All currently active subscription tokens."""
        return [
            t for t in self._tokens.values()
            if t.status in (TokenStatus.ACTIVE, TokenStatus.EXHAUSTED)
        ]

    def get_tokens_by_owner(self, owner_id: str) -> list[SubscriptionToken]:
        """All tokens held by a specific owner."""
        return [
            t for t in self._tokens.values()
            if t.owner_id == owner_id and t.status in (TokenStatus.ACTIVE, TokenStatus.EXHAUSTED)
        ]

    def get_token(self, token_id: str) -> SubscriptionToken | None:
        return self._tokens.get(token_id)

    # ─── Maintenance Cycle ─────────────────────────────────────

    def maintenance_cycle(self) -> dict[str, int]:
        """
        Run periodic maintenance: expire futures and tokens.

        Should be called during OikosService's consolidation cycle.
        Returns a summary of actions taken.
        """
        expired_futures = self.expire_futures()
        expired_tokens = self.expire_tokens()
        return {
            "futures_expired": len(expired_futures),
            "tokens_expired": len(expired_tokens),
            "active_futures": len(self.get_active_futures()),
            "active_tokens": len(self.get_active_tokens()),
        }

    # ─── Balance Sheet Integration ─────────────────────────────

    @property
    def total_liabilities_usd(self) -> Decimal:
        """
        Total outstanding liabilities from derivative commitments.

        This is the sum of:
          1. Locked collateral on active futures (performance guarantee)
          2. Unearned revenue on active tokens (prepaid capacity not yet delivered)

        OikosService subtracts this from net worth as obligations.
        """
        # Futures collateral is already tracked incrementally
        futures_liability = self._locked_collateral_usd

        # Token liability = proportional unearned revenue on active tokens
        token_liability = Decimal("0")
        for token in self._tokens.values():
            if token.status in (TokenStatus.ACTIVE, TokenStatus.EXHAUSTED):
                # Unearned portion = mint_price * (remaining / total)
                if token.requests_per_month > 0:
                    unearned = (
                        token.mint_price_usd
                        * Decimal(str(token.requests_remaining))
                        / Decimal(str(token.requests_per_month))
                    ).quantize(Decimal("0.01"))
                    token_liability += unearned

        return futures_liability + token_liability

    @property
    def locked_collateral_usd(self) -> Decimal:
        """Total collateral locked as performance guarantees on active futures."""
        return self._locked_collateral_usd

    @property
    def total_revenue_usd(self) -> Decimal:
        """Lifetime revenue from all derivative instruments."""
        return self._total_futures_revenue_usd + self._total_token_revenue_usd

    @property
    def futures_revenue_usd(self) -> Decimal:
        return self._total_futures_revenue_usd

    @property
    def token_revenue_usd(self) -> Decimal:
        return self._total_token_revenue_usd

    # ─── Secondary Market Valuation ────────────────────────────

    @property
    def implied_capacity_price(self) -> Decimal:
        """
        The market's implied price per request, derived from the most
        recent secondary market token transfers.

        This IS the market's real-time valuation of the organism's
        cognitive capacity (spec §11.4).
        """
        recent_tokens = [
            t for t in self._tokens.values()
            if t.last_transfer_price_usd > Decimal("0") and t.requests_per_month > 0
        ]
        if not recent_tokens:
            return Decimal("0")

        # Use the most recently transferred token as the reference price.
        # Fall back to minted_at for tokens that have never been transferred.
        latest = max(
            recent_tokens,
            key=lambda t: t.last_transfer_at if t.last_transfer_at is not None else t.minted_at,
        )
        return (
            latest.last_transfer_price_usd / Decimal(str(latest.requests_per_month))
        ).quantize(Decimal("0.0001"))

    # ─── Stats / Observability ─────────────────────────────────

    @property
    def stats(self) -> dict[str, str | int]:
        """Observability snapshot for OikosService.stats aggregation."""
        active_futures = self.get_active_futures()
        active_tokens = self.get_active_tokens()
        return {
            "active_futures": len(active_futures),
            "active_tokens": len(active_tokens),
            "futures_committed_requests": self._futures_committed_requests(),
            "tokens_committed_requests": self._tokens_committed_requests(),
            "total_committed_requests": self.derivatives_committed_requests(),
            "capacity_utilisation": str(self.combined_capacity_utilisation),
            "locked_collateral_usd": str(self._locked_collateral_usd),
            "total_liabilities_usd": str(self.total_liabilities_usd),
            "futures_revenue_usd": str(self._total_futures_revenue_usd),
            "token_revenue_usd": str(self._total_token_revenue_usd),
            "total_revenue_usd": str(self.total_revenue_usd),
            "implied_capacity_price": str(self.implied_capacity_price),
        }
