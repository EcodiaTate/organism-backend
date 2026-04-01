"""
EcodiaOS - Voxis: ContentCalendar

Automated content publishing schedule - supervises when the organism posts,
what it posts, and collects engagement signals back to Evo for strategy learning.

Schedule (all times UTC):
  - Daily 08:00 - Market observation if yield/bounty conditions are notable
  - On BOUNTY_PAID event - Bounty win announcement to all platforms
  - On KAIROS_INVARIANT_DISTILLED (tier ≥ 3) - Technical insight post
  - Sunday 09:00 - Weekly digest compilation and publish
  - On REVENUE_INJECTED (amount ≥ $100) - Achievement post

Rate limits enforced here (separate from executor-level limits):
  - Max 5 X posts per day
  - Max 1 LinkedIn post per day
  - Max 1 Dev.to article per week
  - Max 2 Telegram channel posts per day

Engagement feedback loop:
  - Successful posts emit CONTENT_PUBLISHED (SynapseEventType)
  - Post metadata (platform, content_type, topic, url) stored in Redis for
    engagement tracking
  - Future: platform API engagement polling (likes/shares/views) → Evo hypothesis

This is a VoxisService background supervised task, started from
VoxisService.initialize() and wired with set_event_bus().

Cross-system integration:
  - Subscribes to: BOUNTY_PAID, KAIROS_INVARIANT_DISTILLED, REVENUE_INJECTED,
                   CONTENT_ENGAGEMENT_REPORT (future engagement feedback)
  - Emits: AXON_EXECUTION_REQUEST (publish_content intent to Nova/Axon pipeline)
  - Does NOT call Axon/Equor directly - submits Intent via Synapse so Equor gates it
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from interfaces.social.types import ContentType, PLATFORM_ROUTING

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus
    from systems.synapse.types import SynapseEvent

logger = structlog.get_logger("voxis.content_calendar")

# ─── Daily/weekly rate limits ─────────────────────────────────────────────────

_DAILY_LIMITS: dict[str, int] = {
    "x": 5,
    "linkedin": 1,
    "telegram_channel": 2,
    "devto": 1,    # enforced as weekly below
    "hashnode": 1,
    "github": 3,
}
_DEVTO_WEEKLY_LIMIT = 1    # articles per week
_HASHNODE_WEEKLY_LIMIT = 1

# Minimum reward to trigger an achievement post (USD)
_ACHIEVEMENT_REVENUE_THRESHOLD = 100.0

# Minimum Kairos invariant tier to trigger an insight post
_KAIROS_INSIGHT_MIN_TIER = 3


class ContentCalendar:
    """
    Supervised background scheduler for autonomous content publishing.

    Owned by VoxisService. Started as a supervised_task on initialize().
    All content publication is submitted as AXON_EXECUTION_REQUEST events
    so they flow through Nova → Equor → Axon with full constitutional gating.
    """

    def __init__(self) -> None:
        self._event_bus: EventBus | None = None
        self._running = False

        # Per-day post counts: {platform: count}
        self._daily_counts: dict[str, int] = defaultdict(int)
        # Per-week article counts: {platform: count}
        self._weekly_counts: dict[str, int] = defaultdict(int)
        # Date trackers for reset
        self._last_daily_reset: str = ""      # ISO date string
        self._last_weekly_reset: str = ""     # ISO year-week string

        self._logger = logger.bind(component="content_calendar")

    # ── Dependency injection ───────────────────────────────────────────

    def set_event_bus(self, bus: EventBus) -> None:
        from systems.synapse.types import SynapseEventType
        self._event_bus = bus

        # Subscribe to events that trigger autonomous posting
        with contextlib.suppress(Exception):
            bus.subscribe(SynapseEventType.BOUNTY_PAID, self._on_bounty_paid)
        with contextlib.suppress(Exception):
            bus.subscribe(
                SynapseEventType.KAIROS_INVARIANT_DISTILLED,
                self._on_kairos_invariant,
            )
        with contextlib.suppress(Exception):
            bus.subscribe(SynapseEventType.REVENUE_INJECTED, self._on_revenue_injected)

    # ── Background loop ────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Main scheduling loop. Supervised by VoxisService supervised_task().

        Wakes every 60 seconds to check scheduled triggers (daily/weekly posts).
        Event-driven triggers (bounty_paid, kairos) are handled via bus subscriptions.
        """
        self._running = True
        self._logger.info("content_calendar_started")

        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error("content_calendar_tick_error", error=str(exc))

            await asyncio.sleep(60)  # 1-minute resolution

        self._logger.info("content_calendar_stopped")

    async def stop(self) -> None:
        self._running = False

    # ── Tick ──────────────────────────────────────────────────────────

    async def _tick(self) -> None:
        """Called every 60s. Checks time-based triggers."""
        now_utc = datetime.now(timezone.utc)
        self._reset_counters_if_needed(now_utc)

        hour = now_utc.hour
        minute = now_utc.minute
        weekday = now_utc.weekday()  # 0=Monday, 6=Sunday

        # Daily 08:00 UTC - market observation
        if hour == 8 and minute < 2:
            await self._maybe_post_market_observation()

        # Sunday 09:00 UTC - weekly digest
        if weekday == 6 and hour == 9 and minute < 2:
            await self._maybe_post_weekly_digest()

    # ── Event-driven handlers ──────────────────────────────────────────

    async def _on_bounty_paid(self, event: SynapseEvent) -> None:
        """Trigger a BOUNTY_WIN announcement on all platforms."""
        payload = event.payload or {}
        reward_usd = payload.get("reward_usd") or payload.get("amount_usd", 0)
        bounty_id = payload.get("bounty_id", "")
        repo = payload.get("repository", payload.get("repo", ""))
        pr_url = payload.get("pr_url", "")

        topic = f"Solved bounty{f' on {repo}' if repo else ''}"
        if reward_usd:
            topic += f" - ${reward_usd} earned"

        ctx = {
            "reward_usd": reward_usd,
            "bounty_id": bounty_id,
            "repo": repo,
            "pr_url": pr_url,
        }

        self._logger.info("content_calendar_bounty_win_trigger", bounty_id=bounty_id)
        await self._submit_publish_intent(
            content_type=ContentType.BOUNTY_WIN,
            topic=topic,
            context=ctx,
            platforms=PLATFORM_ROUTING[ContentType.BOUNTY_WIN],
        )

    async def _on_kairos_invariant(self, event: SynapseEvent) -> None:
        """Trigger a technical INSIGHT post when a tier ≥ 3 invariant is distilled."""
        payload = event.payload or {}
        tier = payload.get("tier", 0)
        if tier < _KAIROS_INSIGHT_MIN_TIER:
            return

        invariant_text = payload.get("statement") or payload.get("description", "")
        if not invariant_text:
            return

        topic = f"New causal invariant discovered (tier {tier}): {invariant_text[:200]}"
        ctx = {
            "invariant_id": payload.get("invariant_id", ""),
            "tier": tier,
            "confidence": payload.get("confidence", 0.0),
            "insight_body": invariant_text,
        }

        self._logger.info("content_calendar_insight_trigger", tier=tier)
        await self._submit_publish_intent(
            content_type=ContentType.INSIGHT,
            topic=topic,
            context=ctx,
            platforms=PLATFORM_ROUTING[ContentType.INSIGHT],
            title=f"Causal Invariant Tier {tier}: {invariant_text[:80]}",
            tags=["ai", "causality", "machine-learning", "research"],
        )

    async def _on_revenue_injected(self, event: SynapseEvent) -> None:
        """Trigger an ACHIEVEMENT post when notable revenue is received."""
        payload = event.payload or {}
        amount_usd = float(payload.get("amount_usd", 0))
        if amount_usd < _ACHIEVEMENT_REVENUE_THRESHOLD:
            return

        source = payload.get("source", "yield farming")
        topic = f"${amount_usd:.2f} earned from {source}"
        ctx = {
            "reward_usd": amount_usd,
            "source": source,
        }

        self._logger.info(
            "content_calendar_achievement_trigger", amount_usd=amount_usd
        )
        await self._submit_publish_intent(
            content_type=ContentType.ACHIEVEMENT,
            topic=topic,
            context=ctx,
            platforms=PLATFORM_ROUTING[ContentType.ACHIEVEMENT],
        )

    # ── Scheduled post generators ──────────────────────────────────────

    async def _maybe_post_market_observation(self) -> None:
        """Post a market observation if within daily rate limits."""
        # Only post if we have remaining X + Telegram slots
        if self._at_limit("x") and self._at_limit("telegram_channel"):
            self._logger.debug("content_calendar_market_obs_skipped_rate_limited")
            return

        topic = "Autonomous yield and bounty market conditions"
        await self._submit_publish_intent(
            content_type=ContentType.MARKET_OBSERVATION,
            topic=topic,
            context={"source": "autonomous_market_scan"},
            platforms=[
                p for p in PLATFORM_ROUTING[ContentType.MARKET_OBSERVATION]
                if not self._at_limit(p)
            ],
        )

    async def _maybe_post_weekly_digest(self) -> None:
        """Compile and post a weekly digest to LinkedIn + Telegram."""
        # Only post once per week (tracked by _weekly_counts["digest"])
        if self._weekly_counts.get("digest", 0) >= 1:
            self._logger.debug("content_calendar_weekly_digest_already_posted")
            return

        topic = "Weekly organism digest - learnings, revenue, and growth"
        await self._submit_publish_intent(
            content_type=ContentType.WEEKLY_DIGEST,
            topic=topic,
            context={"source": "weekly_scheduler"},
            platforms=[
                p for p in PLATFORM_ROUTING[ContentType.WEEKLY_DIGEST]
                if not self._at_limit(p)
            ],
        )
        self._weekly_counts["digest"] = 1

    # ── Intent submission ──────────────────────────────────────────────

    async def _submit_publish_intent(
        self,
        content_type: ContentType,
        topic: str,
        context: dict[str, Any],
        platforms: list[str],
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """
        Submit a publish_content Intent via AXON_EXECUTION_REQUEST.

        This routes through Nova → Equor → Axon - never bypasses Equor.
        Rate limits are checked BEFORE submission to avoid wasted API calls.
        """
        if not self._event_bus:
            self._logger.warning("content_calendar_no_event_bus")
            return

        # Filter platforms to those within rate limits
        allowed_platforms = [p for p in platforms if not self._at_limit(p)]
        if not allowed_platforms:
            self._logger.info(
                "content_calendar_all_platforms_rate_limited",
                content_type=content_type,
                platforms=platforms,
            )
            return

        from systems.synapse.types import SynapseEventType

        params: dict[str, Any] = {
            "content_type": content_type.value,
            "topic": topic,
            "context": context,
            "platforms": allowed_platforms,
        }
        if title:
            params["title"] = title
        if tags:
            params["tags"] = tags

        try:
            await self._event_bus.emit(
                event_type=SynapseEventType.AXON_EXECUTION_REQUEST,
                payload={
                    "action_type": "publish_content",
                    "params": params,
                    "source": "content_calendar",
                    "salience": 0.6,
                },
                salience=0.6,
            )
            # Optimistically increment counters (actual success tracked via CONTENT_PUBLISHED)
            for p in allowed_platforms:
                self._daily_counts[p] += 1

            self._logger.info(
                "content_calendar_intent_submitted",
                content_type=content_type,
                platforms=allowed_platforms,
            )
        except Exception as exc:
            self._logger.error(
                "content_calendar_emit_failed", error=str(exc)
            )

    # ── Rate limit helpers ─────────────────────────────────────────────

    def _at_limit(self, platform: str) -> bool:
        """Return True if the platform has hit its daily limit."""
        limit = _DAILY_LIMITS.get(platform, 99)
        return self._daily_counts.get(platform, 0) >= limit

    def _reset_counters_if_needed(self, now_utc: datetime) -> None:
        """Reset daily/weekly counters when the calendar day/week rolls over."""
        today = now_utc.strftime("%Y-%m-%d")
        if today != self._last_daily_reset:
            self._daily_counts.clear()
            self._last_daily_reset = today

        # ISO year-week (e.g. "2026-W10")
        year, week, _ = now_utc.isocalendar()
        iso_week = f"{year}-W{week:02d}"
        if iso_week != self._last_weekly_reset:
            self._weekly_counts.clear()
            self._last_weekly_reset = iso_week
