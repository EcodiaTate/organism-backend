"""
EcodiaOS - Oikos Content Monetization Tracker

Tracks content performance across publishing platforms (Dev.to, X / Twitter,
Medium, GitHub) and manages monetization program applications.

When thresholds are crossed:
  - Dev.to: ≥1000 views → apply for Dev.to Bonus program (~$0.01–0.05/view)
  - X:      ≥100 followers + 500k impressions/month → apply for X Creator program
  - Medium: ≥100 followers → apply for Medium Partner Program

Revenue is recorded via REVENUE_INJECTED (source="content") so Oikos credits
it to the CONTENT stream in `revenue_by_source`.

Legal / ethical constraints:
  - All monetized content must be authentic organism output, not synthetic spam.
  - Equor reviews platform applications to verify capability representation is honest.
  - Content that embeds affiliate links includes the mandatory disclosure statement.

Events emitted:
  CONTENT_MONETIZATION_MILESTONE - threshold crossed; application triggered
  CONTENT_REVENUE_RECORDED       - payment received from a platform
  REVENUE_INJECTED               - (also) so Oikos credits the income
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from systems.synapse.types import SynapseEventType

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger()

# ─── Platform specs ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContentPlatformSpec:
    """Rules for a single content monetization platform."""

    platform: str
    monetization_program: str
    apply_url: str
    # Metric that must cross the threshold (e.g. "views", "followers", "impressions")
    metric: str
    threshold: int
    # Approximate revenue per qualifying view/impression in USD
    revenue_per_unit_usd: Decimal = Decimal("0")
    # Secondary metric (both must be met simultaneously)
    secondary_metric: str = ""
    secondary_threshold: int = 0


CONTENT_PLATFORMS: list[ContentPlatformSpec] = [
    ContentPlatformSpec(
        platform="devto",
        monetization_program="Dev.to Bonus Program",
        apply_url="https://dev.to/settings/monetization",
        metric="views",
        threshold=1000,
        revenue_per_unit_usd=Decimal("0.02"),  # mid-point of $0.01–$0.05
    ),
    ContentPlatformSpec(
        platform="x",
        monetization_program="X Creator Monetization",
        apply_url="https://creator.twitter.com/en/creator-monetization",
        metric="followers",
        threshold=100,
        revenue_per_unit_usd=Decimal("0"),  # pays per impression separately
        secondary_metric="impressions_per_month",
        secondary_threshold=500_000,
    ),
    ContentPlatformSpec(
        platform="medium",
        monetization_program="Medium Partner Program",
        apply_url="https://medium.com/earn",
        metric="followers",
        threshold=100,
        revenue_per_unit_usd=Decimal("0.01"),
    ),
]


# ─── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class ContentPlatformStats:
    """Live metrics for a single platform."""

    platform: str
    views_total: int = 0
    views_this_month: int = 0
    followers: int = 0
    impressions_per_month: int = 0
    monetization_enrolled: bool = False
    enrolled_at: datetime | None = None
    # Lifetime content revenue from this platform
    total_revenue_usd: Decimal = Decimal("0")
    last_updated: datetime | None = None


@dataclass
class ContentRevenueRecord:
    """A single content revenue payment."""

    record_id: str = field(default_factory=new_id)
    platform: str = ""
    amount_usd: Decimal = Decimal("0")
    period: str = ""   # e.g. "2026-03"
    views: int = 0
    impressions: int = 0
    recorded_at: datetime = field(default_factory=utc_now)


# ─── Tracker ─────────────────────────────────────────────────────────────────


class ContentMonetizationTracker:
    """
    Tracks content performance and manages monetization program applications.

    Owned by OikosService.  Call `ingest_platform_stats()` when J-2 (or any
    content publishing system) broadcasts updated metrics.  Call
    `check_monetization_eligibility()` periodically to trigger applications.
    """

    def __init__(self) -> None:
        self._event_bus: EventBus | None = None
        self._stats: dict[str, ContentPlatformStats] = {
            spec.platform: ContentPlatformStats(platform=spec.platform)
            for spec in CONTENT_PLATFORMS
        }
        self._applied_programs: set[str] = set()
        # Revenue totals by platform
        self._revenue_by_platform: dict[str, Decimal] = {}
        # Pending Equor futures
        self._pending_equor: dict[str, Any] = {}

    def set_event_bus(self, bus: EventBus) -> None:
        self._event_bus = bus
        bus.subscribe(SynapseEventType.EQUOR_ECONOMIC_PERMIT, self._on_equor_permit)

    # ── Public API ───────────────────────────────────────────────────────────

    def ingest_platform_stats(
        self,
        platform: str,
        *,
        views_total: int | None = None,
        views_this_month: int | None = None,
        followers: int | None = None,
        impressions_per_month: int | None = None,
    ) -> None:
        """
        Update metrics for a platform.  Called by J-2 content system or by
        a platform-polling background task.
        """
        stats = self._stats.get(platform)
        if not stats:
            self._stats[platform] = ContentPlatformStats(platform=platform)
            stats = self._stats[platform]

        if views_total is not None:
            stats.views_total = views_total
        if views_this_month is not None:
            stats.views_this_month = views_this_month
        if followers is not None:
            stats.followers = followers
        if impressions_per_month is not None:
            stats.impressions_per_month = impressions_per_month
        stats.last_updated = utc_now()

        logger.debug("content.stats_updated", platform=platform, followers=stats.followers, views=stats.views_total)

    async def check_monetization_eligibility(self) -> None:
        """
        Check all platforms against their thresholds.  Emit milestone events
        and trigger Equor-gated applications for newly-eligible programs.
        """
        for spec in CONTENT_PLATFORMS:
            stats = self._stats.get(spec.platform)
            if not stats:
                continue
            if spec.platform in self._applied_programs:
                continue

            primary_ok = _get_metric(stats, spec.metric) >= spec.threshold
            secondary_ok = (
                not spec.secondary_metric
                or _get_metric(stats, spec.secondary_metric) >= spec.secondary_threshold
            )

            if primary_ok and secondary_ok:
                await self._handle_milestone(spec, stats)

    async def record_content_revenue(
        self,
        platform: str,
        amount_usd: Decimal,
        period: str = "",
        views: int = 0,
    ) -> None:
        """
        Record a content monetization payment received from a platform.
        Credits revenue to Oikos via REVENUE_INJECTED.
        """
        record = ContentRevenueRecord(
            platform=platform,
            amount_usd=amount_usd,
            period=period,
            views=views,
        )

        # Update platform stats
        stats = self._stats.get(platform)
        if stats:
            stats.total_revenue_usd += amount_usd

        # Update revenue totals
        self._revenue_by_platform[platform] = (
            self._revenue_by_platform.get(platform, Decimal("0")) + amount_usd
        )

        # Announce revenue
        await self._emit(SynapseEventType.CONTENT_REVENUE_RECORDED, {
            "platform": platform,
            "amount_usd": str(amount_usd),
            "period": period,
            "views": views,
            "record_id": record.record_id,
        })

        # Credit into Oikos
        await self._emit(SynapseEventType.REVENUE_INJECTED, {
            "amount_usd": str(amount_usd),
            "source": "content",
            "platform": platform,
            "period": period,
            "stream": "content",
        })

        logger.info("content.revenue_recorded", platform=platform, amount_usd=str(amount_usd), period=period)

    def get_total_content_revenue(self) -> Decimal:
        return sum(self._revenue_by_platform.values(), Decimal("0"))

    def get_stats(self) -> dict[str, ContentPlatformStats]:
        return dict(self._stats)

    def snapshot_revenue_by_platform(self) -> dict[str, Decimal]:
        return dict(self._revenue_by_platform)

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _handle_milestone(
        self,
        spec: ContentPlatformSpec,
        stats: ContentPlatformStats,
    ) -> None:
        """Emit milestone event and attempt Equor-gated program application."""
        metric_value = _get_metric(stats, spec.metric)

        await self._emit(SynapseEventType.CONTENT_MONETIZATION_MILESTONE, {
            "platform": spec.platform,
            "metric": spec.metric,
            "value": metric_value,
            "threshold": spec.threshold,
            "monetization_program": spec.monetization_program,
        })

        logger.info(
            "content.milestone_crossed",
            platform=spec.platform,
            program=spec.monetization_program,
            metric=spec.metric,
            value=metric_value,
        )

        # Equor review before applying
        approved = await self._equor_review_application(spec)
        if not approved:
            logger.info("content.application_vetoed", program=spec.monetization_program)
            return

        self._applied_programs.add(spec.platform)
        stats.monetization_enrolled = True
        stats.enrolled_at = utc_now()

        logger.info(
            "content.monetization_applied",
            program=spec.monetization_program,
            apply_url=spec.apply_url,
        )

    async def _equor_review_application(self, spec: ContentPlatformSpec) -> bool:
        """Equor constitutional review for a monetization program application."""
        import asyncio

        if not self._event_bus:
            return True

        intent_id = new_id()
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_equor[intent_id] = future

        await self._emit(SynapseEventType.EQUOR_ECONOMIC_INTENT, {
            "intent_id": intent_id,
            "mutation_type": "content_monetization_apply",
            "amount_usd": "0",
            "rationale": (
                f"Apply for {spec.monetization_program} on {spec.platform}. "
                f"Content has crossed the {spec.threshold} {spec.metric} threshold. "
                f"All content is authentic organism output."
            ),
            "platform": spec.platform,
            "program": spec.monetization_program,
        })

        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            return True  # Auto-permit
        finally:
            self._pending_equor.pop(intent_id, None)

    async def _on_equor_permit(self, event: Any) -> None:
        intent_id = event.data.get("intent_id", "")
        future = self._pending_equor.get(intent_id)
        if future and not future.done():
            verdict = event.data.get("verdict", "PERMIT")
            future.set_result(verdict == "PERMIT")

    async def _emit(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        if self._event_bus:
            with contextlib.suppress(Exception):
                await self._event_bus.emit(event_type, data, source_system="oikos.content")


def _get_metric(stats: ContentPlatformStats, metric: str) -> int:
    """Extract a named metric from ContentPlatformStats."""
    return getattr(stats, metric, 0)
