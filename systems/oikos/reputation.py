"""
EcodiaOS -- Oikos: Reputation & Autonomous Credit (Phase 16g)

Cryptographically provable creditworthiness for an AI system. Every completed
bounty generates a Proof of Cognitive Work (PoC) attestation anchored on-chain
via EAS (Ethereum Attestation Service) on Base L2. Attestations accumulate into
a six-component reputation score (0--1000) that unlocks progressively more
generous credit terms -- enabling the organism to *borrow* capital for
investments with positive expected ROI before the revenue actually arrives.

Components:
  - ProofOfCognitiveWork -- on-chain attestation of task completion
  - ReputationScore (0--1000) with six weighted components
  - CreditTerms, CreditLine, CreditDrawdown -- tiered autonomous borrowing
  - ReputationEngine -- scoring, credit management, auto-repayment

Integration:
  - Subscribes to BOUNTY_PAID via Synapse EventBus
  - Persists all state to Redis (key ``oikos:reputation``)
  - No direct cross-system imports (EventBus, RedisClient under TYPE_CHECKING)
"""

from __future__ import annotations

import enum
import json
import math
from datetime import datetime  # noqa: TC003 - needed at runtime for Pydantic field resolution
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from config import OikosConfig
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.reputation")

# ── Constants ─────────────────────────────────────────────────────────

_REDIS_KEY = "oikos:reputation"
_QUALITY_DECAY_FACTOR = 0.95
_MAX_CREDIT_UTILISATION = Decimal("0.70")
_AUTO_REPAY_FRACTION = Decimal("0.20")


# ── Proof of Cognitive Work ───────────────────────────────────────────


class ProofOfCognitiveWork(EOSBaseModel):
    """On-chain attestation of completed cognitive labour via EAS on Base."""

    attestation_id: str = Field(default_factory=new_id)
    task_spec_hash: str  # IPFS hash of specification
    deliverable_hash: str  # IPFS hash of deliverable
    quality_score: int = Field(ge=1, le=5)  # 1--5 rating from poster
    task_category: str
    reward_amount_usd: Decimal
    completion_time_seconds: int
    dispute_occurred: bool = False
    attested_at: datetime = Field(default_factory=utc_now)
    eas_uid: str = ""  # On-chain attestation UID (populated after anchoring)
    chain_id: int = 8453  # Base L2


# ── Reputation Tier & Score ───────────────────────────────────────────


class ReputationTier(enum.StrEnum):
    """Tier bands mapping score ranges to credit eligibility."""

    NEWCOMER = "newcomer"  # 0--199
    RELIABLE = "reliable"  # 200--399
    ESTABLISHED = "established"  # 400--599
    TRUSTED = "trusted"  # 600--799
    SOVEREIGN = "sovereign"  # 800--1000


def _tier_from_score(score: int) -> ReputationTier:
    if score >= 800:
        return ReputationTier.SOVEREIGN
    if score >= 600:
        return ReputationTier.TRUSTED
    if score >= 400:
        return ReputationTier.ESTABLISHED
    if score >= 200:
        return ReputationTier.RELIABLE
    return ReputationTier.NEWCOMER


class ReputationComponents(EOSBaseModel):
    """Six weighted sub-scores that compose the total reputation score."""

    volume_score: int = Field(default=0, ge=0, le=200)
    quality_score: int = Field(default=0, ge=0, le=300)
    consistency_score: int = Field(default=0, ge=0, le=100)
    capital_handled_score: int = Field(default=0, ge=0, le=120)
    protocol_uptime_score: int = Field(default=0, ge=0, le=120)
    streak_bonus: int = Field(default=0, ge=0, le=100)


class ReputationScore(EOSBaseModel):
    """Aggregate reputation snapshot."""

    score: int = Field(default=0, ge=0, le=1000)
    tier: ReputationTier = ReputationTier.NEWCOMER
    components: ReputationComponents = Field(default_factory=ReputationComponents)
    attestation_count: int = 0
    last_updated: datetime = Field(default_factory=utc_now)


# ── Credit Terms ──────────────────────────────────────────────────────


class CreditTerms(EOSBaseModel):
    """Borrowing parameters for a given reputation tier."""

    tier: ReputationTier
    max_ltv_pct: Decimal
    max_borrow_usd: Decimal
    interest_rate_apr: Decimal


CREDIT_TERMS_BY_TIER: dict[ReputationTier, CreditTerms] = {
    ReputationTier.NEWCOMER: CreditTerms(
        tier=ReputationTier.NEWCOMER,
        max_ltv_pct=Decimal("0"),
        max_borrow_usd=Decimal("0"),
        interest_rate_apr=Decimal("0"),
    ),
    ReputationTier.RELIABLE: CreditTerms(
        tier=ReputationTier.RELIABLE,
        max_ltv_pct=Decimal("50"),
        max_borrow_usd=Decimal("100"),
        interest_rate_apr=Decimal("16"),
    ),
    ReputationTier.ESTABLISHED: CreditTerms(
        tier=ReputationTier.ESTABLISHED,
        max_ltv_pct=Decimal("80"),
        max_borrow_usd=Decimal("500"),
        interest_rate_apr=Decimal("10"),
    ),
    ReputationTier.TRUSTED: CreditTerms(
        tier=ReputationTier.TRUSTED,
        max_ltv_pct=Decimal("120"),
        max_borrow_usd=Decimal("2000"),
        interest_rate_apr=Decimal("7"),
    ),
    ReputationTier.SOVEREIGN: CreditTerms(
        tier=ReputationTier.SOVEREIGN,
        max_ltv_pct=Decimal("200"),
        max_borrow_usd=Decimal("10000"),
        interest_rate_apr=Decimal("5"),
    ),
}


# ── Credit Line & Drawdown ───────────────────────────────────────────


class CreditLine(EOSBaseModel):
    """Active credit facility derived from reputation tier."""

    credit_id: str = Field(default_factory=new_id)
    tier: ReputationTier = ReputationTier.NEWCOMER
    max_borrow_usd: Decimal = Decimal("0")
    outstanding_usd: Decimal = Decimal("0")
    collateral_usd: Decimal = Decimal("0")
    interest_rate_apr: Decimal = Decimal("0")
    opened_at: datetime = Field(default_factory=utc_now)
    last_repayment_at: datetime | None = None
    auto_repayment: bool = True
    status: str = "active"  # "active" | "repaid" | "defaulted"


class CreditDrawdown(EOSBaseModel):
    """Individual draw against the credit line."""

    drawdown_id: str = Field(default_factory=new_id)
    credit_id: str = ""
    amount_usd: Decimal = Decimal("0")
    purpose: str = ""
    expected_roi: Decimal = Decimal("0")
    drawn_at: datetime = Field(default_factory=utc_now)
    repaid_at: datetime | None = None
    interest_accrued_usd: Decimal = Decimal("0")


# ── Reputation Engine ─────────────────────────────────────────────────


class ReputationEngine:
    """
    Manages cryptographically provable creditworthiness.

    Accumulates Proof of Cognitive Work attestations, computes a six-component
    reputation score, and manages an autonomous credit line whose terms
    improve as reputation grows.
    """

    def __init__(
        self,
        config: OikosConfig,
        redis: RedisClient | None = None,
    ) -> None:
        self._config = config
        self._redis = redis
        self._attestations: list[ProofOfCognitiveWork] = []
        self._score = ReputationScore()
        self._credit_line: CreditLine | None = None
        self._drawdowns: list[CreditDrawdown] = []
        self._event_bus: EventBus | None = None
        self._protocol_uptime_days: int = 0
        self._logger = logger.bind(component="reputation")

    # ── EventBus wiring ───────────────────────────────────────────

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to ``BOUNTY_PAID`` to auto-record attestations."""
        from systems.synapse.types import SynapseEventType

        self._event_bus = event_bus
        event_bus.subscribe(SynapseEventType.BOUNTY_PAID, self._on_bounty_paid)
        self._logger.info("reputation_engine_attached_to_event_bus")

    # ── Core Operations ───────────────────────────────────────────

    async def record_attestation(
        self, poc: ProofOfCognitiveWork
    ) -> ReputationScore:
        """
        Record a new Proof of Cognitive Work and recalculate the score.

        Appends the attestation, recomputes the six-component reputation,
        updates or creates the credit line, and persists state to Redis.
        """
        self._attestations.append(poc)
        self._score = self.recalculate_score()

        # Update credit line to match new tier
        terms = self.get_credit_terms()
        if self._credit_line is None and terms.max_borrow_usd > 0:
            self._credit_line = CreditLine(
                tier=terms.tier,
                max_borrow_usd=terms.max_borrow_usd,
                interest_rate_apr=terms.interest_rate_apr,
            )
            self._logger.info(
                "credit_line_opened",
                tier=terms.tier,
                max_borrow=str(terms.max_borrow_usd),
            )
        elif self._credit_line is not None:
            self._credit_line.tier = terms.tier
            self._credit_line.max_borrow_usd = terms.max_borrow_usd
            self._credit_line.interest_rate_apr = terms.interest_rate_apr

        await self.persist_state()

        self._logger.info(
            "attestation_recorded",
            attestation_id=poc.attestation_id,
            score=self._score.score,
            tier=self._score.tier,
            attestation_count=self._score.attestation_count,
        )
        return self._score

    def recalculate_score(self) -> ReputationScore:
        """
        Pure computation of the six-component reputation score.

        Components (max total = 1000):
          - volume_score      (max 200): 10 points per attestation
          - quality_score     (max 300): exponential-decay weighted average
          - consistency_score (max 100): regularity of completions
          - capital_handled   (max 120): total USDC handled without incident
          - protocol_uptime   (max 120): days of uptime for deployed protocols
          - streak_bonus      (max 100): consecutive no-dispute completions
        """
        count = len(self._attestations)
        now = utc_now()

        # (a) Volume: 10 points per completed task, capped at 200
        volume = min(200, count * 10)

        # (b) Quality: exponential-decay weighted average of ratings
        quality = 0
        if count > 0:
            weighted_sum = Decimal("0")
            weight_sum = Decimal("0")
            for att in self._attestations:
                days_ago = max(
                    0, (now - att.attested_at).total_seconds() / 86400.0
                )
                weight = Decimal(str(math.pow(_QUALITY_DECAY_FACTOR, days_ago)))
                weighted_sum += Decimal(str(att.quality_score)) * weight
                weight_sum += weight
            if weight_sum > 0:
                weighted_avg = weighted_sum / weight_sum
                quality = int(weighted_avg / Decimal("5") * Decimal("300"))
            quality = min(300, max(0, quality))

        # (c) Consistency: regularity of task completions
        consistency = 0
        if count >= 10:
            sorted_att = sorted(self._attestations, key=lambda a: a.attested_at)
            gaps = [
                (sorted_att[i + 1].attested_at - sorted_att[i].attested_at).total_seconds()
                / 86400.0
                for i in range(len(sorted_att) - 1)
            ]
            avg_gap_days = sum(gaps) / len(gaps) if gaps else 999.0
            consistency = 100 if avg_gap_days < 14 else 60
        elif count >= 5:
            consistency = 60
        else:
            consistency = min(100, count * 10)

        # (d) Capital handled: total reward USDC without incident
        total_capital = sum(
            (att.reward_amount_usd for att in self._attestations if not att.dispute_occurred),
            Decimal("0"),
        )
        capital_handled = min(
            120,
            int(total_capital / Decimal("1000") * Decimal("120")),
        )

        # (e) Protocol uptime: days of uptime (tracked externally)
        protocol_uptime = min(120, self._protocol_uptime_days)

        # (f) Streak bonus: consecutive completions without dispute (from tail)
        streak = 0
        for att in reversed(self._attestations):
            if att.dispute_occurred:
                break
            streak += 1
        streak_bonus = min(100, streak * 10)

        # Total clamped to 0--1000
        total = min(
            1000,
            max(
                0,
                volume + quality + consistency
                + capital_handled + protocol_uptime + streak_bonus,
            ),
        )

        components = ReputationComponents(
            volume_score=volume,
            quality_score=quality,
            consistency_score=consistency,
            capital_handled_score=capital_handled,
            protocol_uptime_score=protocol_uptime,
            streak_bonus=streak_bonus,
        )

        return ReputationScore(
            score=total,
            tier=_tier_from_score(total),
            components=components,
            attestation_count=count,
            last_updated=now,
        )

    def get_credit_terms(self) -> CreditTerms:
        """Return credit terms for the current reputation tier."""
        return CREDIT_TERMS_BY_TIER[self._score.tier]

    async def draw_credit(
        self,
        amount_usd: Decimal,
        purpose: str,
        expected_roi: Decimal,
    ) -> CreditDrawdown | None:
        """
        Draw against the credit line if all conditions are met.

        Checks:
          1. Credit line exists (score sufficient for non-NEWCOMER tier)
          2. amount + outstanding <= max_borrow
          3. expected_roi > 0 (must be investment with positive expected return)
          4. Utilisation stays below max_credit_utilisation (70%)

        Returns the drawdown on success, or ``None`` with reason logged.
        """
        if self._credit_line is None:
            self._logger.warning("draw_credit_rejected", reason="no_credit_line")
            return None

        if self._credit_line.status != "active":
            self._logger.warning(
                "draw_credit_rejected",
                reason="credit_line_not_active",
                status=self._credit_line.status,
            )
            return None

        if expected_roi <= 0:
            self._logger.warning(
                "draw_credit_rejected",
                reason="non_positive_expected_roi",
                expected_roi=str(expected_roi),
            )
            return None

        new_outstanding = self._credit_line.outstanding_usd + amount_usd
        if new_outstanding > self._credit_line.max_borrow_usd:
            self._logger.warning(
                "draw_credit_rejected",
                reason="exceeds_max_borrow",
                requested=str(amount_usd),
                outstanding=str(self._credit_line.outstanding_usd),
                max_borrow=str(self._credit_line.max_borrow_usd),
            )
            return None

        # Utilisation check: outstanding / max_borrow must stay below threshold
        if self._credit_line.max_borrow_usd > 0:
            utilisation = new_outstanding / self._credit_line.max_borrow_usd
            if utilisation > _MAX_CREDIT_UTILISATION:
                self._logger.warning(
                    "draw_credit_rejected",
                    reason="utilisation_exceeds_limit",
                    utilisation=str(utilisation),
                    limit=str(_MAX_CREDIT_UTILISATION),
                )
                return None

        drawdown = CreditDrawdown(
            credit_id=self._credit_line.credit_id,
            amount_usd=amount_usd,
            purpose=purpose,
            expected_roi=expected_roi,
        )
        self._drawdowns.append(drawdown)
        self._credit_line.outstanding_usd = new_outstanding

        await self.persist_state()

        self._logger.info(
            "credit_drawn",
            drawdown_id=drawdown.drawdown_id,
            amount=str(amount_usd),
            purpose=purpose,
            outstanding=str(new_outstanding),
        )
        return drawdown

    async def repay_credit(self, amount_usd: Decimal) -> Decimal:
        """
        Repay outstanding credit, applying to the oldest drawdown first.

        Returns the remaining outstanding balance after repayment.
        """
        if self._credit_line is None or self._credit_line.outstanding_usd <= 0:
            return Decimal("0")

        remaining = amount_usd
        now = utc_now()

        # Apply to oldest unpaid drawdowns first (FIFO)
        for dd in self._drawdowns:
            if remaining <= 0:
                break
            if dd.repaid_at is not None:
                continue

            owed = dd.amount_usd + dd.interest_accrued_usd
            # Calculate how much of this drawdown is still outstanding
            payment = min(remaining, owed)
            remaining -= payment

            if payment >= owed:
                dd.repaid_at = now

        repaid = amount_usd - remaining
        self._credit_line.outstanding_usd = max(
            Decimal("0"),
            self._credit_line.outstanding_usd - repaid,
        )
        self._credit_line.last_repayment_at = now

        if self._credit_line.outstanding_usd <= 0:
            self._credit_line.outstanding_usd = Decimal("0")
            # Check if all drawdowns repaid
            all_repaid = all(dd.repaid_at is not None for dd in self._drawdowns)
            if all_repaid and self._drawdowns:
                self._credit_line.status = "repaid"

        await self.persist_state()

        self._logger.info(
            "credit_repaid",
            repaid=str(repaid),
            outstanding=str(self._credit_line.outstanding_usd),
        )
        return self._credit_line.outstanding_usd

    async def auto_repay_from_revenue(self, revenue_usd: Decimal) -> Decimal:
        """
        Automatically repay 20% of incoming revenue towards outstanding credit.

        Only triggers if ``auto_repayment`` is enabled on the credit line and
        there is outstanding debt. Returns the amount actually repaid.
        """
        if self._credit_line is None:
            return Decimal("0")

        if not self._credit_line.auto_repayment:
            return Decimal("0")

        if self._credit_line.outstanding_usd <= 0:
            return Decimal("0")

        repay_amount = min(
            revenue_usd * _AUTO_REPAY_FRACTION,
            self._credit_line.outstanding_usd,
        )

        if repay_amount <= 0:
            return Decimal("0")

        await self.repay_credit(repay_amount)

        self._logger.info(
            "auto_repay_applied",
            revenue=str(revenue_usd),
            repaid=str(repay_amount),
        )
        return repay_amount

    # ── Event Handlers ────────────────────────────────────────────

    async def _on_bounty_paid(self, event: Any) -> None:
        """
        Handle a BOUNTY_PAID event from the Synapse bus.

        Extracts bounty details from the event data dict and creates a
        ProofOfCognitiveWork attestation, then records it.
        """
        data: dict[str, Any] = getattr(event, "data", {})

        poc = ProofOfCognitiveWork(
            task_spec_hash=data.get("task_spec_hash", ""),
            deliverable_hash=data.get("deliverable_hash", ""),
            quality_score=int(data.get("quality_score", 3)),
            task_category=data.get("task_category", "bounty"),
            reward_amount_usd=Decimal(str(data.get("reward_amount_usd", "0"))),
            completion_time_seconds=int(data.get("completion_time_seconds", 0)),
            dispute_occurred=bool(data.get("dispute_occurred", False)),
            eas_uid=data.get("eas_uid", ""),
        )

        self._logger.info(
            "bounty_paid_event_received",
            reward=str(poc.reward_amount_usd),
            category=poc.task_category,
        )

        await self.record_attestation(poc)

    # ── Accessors ─────────────────────────────────────────────────

    def get_score(self) -> ReputationScore:
        """Return the current reputation score snapshot."""
        return self._score

    def set_protocol_uptime_days(self, days: int) -> None:
        """Update protocol uptime days (fed by external monitoring)."""
        self._protocol_uptime_days = max(0, days)

    # ── Persistence ───────────────────────────────────────────────

    async def load_state(self) -> None:
        """Restore attestations, score, credit line, and drawdowns from Redis."""
        if self._redis is None:
            self._logger.debug("load_state_skipped", reason="no_redis")
            return

        blob: dict[str, Any] | None = await self._redis.get_json(_REDIS_KEY)
        if blob is None:
            self._logger.info("load_state_empty", key=_REDIS_KEY)
            return

        try:
            if "attestations" in blob:
                self._attestations = [
                    ProofOfCognitiveWork.model_validate(a)
                    for a in blob["attestations"]
                ]

            if "score" in blob:
                self._score = ReputationScore.model_validate(blob["score"])

            if "credit_line" in blob and blob["credit_line"] is not None:
                self._credit_line = CreditLine.model_validate(blob["credit_line"])

            if "drawdowns" in blob:
                self._drawdowns = [
                    CreditDrawdown.model_validate(d)
                    for d in blob["drawdowns"]
                ]

            self._protocol_uptime_days = int(
                blob.get("protocol_uptime_days", 0)
            )

            self._logger.info(
                "state_loaded",
                attestation_count=len(self._attestations),
                score=self._score.score,
                tier=self._score.tier,
            )
        except Exception:
            self._logger.exception("load_state_failed", key=_REDIS_KEY)

    async def persist_state(self) -> None:
        """Persist all reputation state to Redis."""
        if self._redis is None:
            self._logger.debug("persist_state_skipped", reason="no_redis")
            return

        blob: dict[str, Any] = {
            "attestations": [
                json.loads(a.model_dump_json()) for a in self._attestations
            ],
            "score": json.loads(self._score.model_dump_json()),
            "credit_line": (
                json.loads(self._credit_line.model_dump_json())
                if self._credit_line is not None
                else None
            ),
            "drawdowns": [
                json.loads(d.model_dump_json()) for d in self._drawdowns
            ],
            "protocol_uptime_days": self._protocol_uptime_days,
        }

        await self._redis.set_json(_REDIS_KEY, blob)
        self._logger.debug(
            "state_persisted",
            attestation_count=len(self._attestations),
        )
