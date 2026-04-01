"""
InputChannel abstraction - generalised external data source for market discovery.

Each InputChannel is a READ-ONLY sensor that fetches Opportunity objects from an
external source (DeFi protocols, job boards, research databases, art markets, etc.).

Architecture:
- InputChannel (ABC) + Opportunity (data model) - lingua franca
- InputChannelRegistry - manages a bounded set of active channels
- Built-in channels are imported lazily from nova.builtin_channels.*

Constraints (from spec):
- Channels are read-only: no write/mutation of external systems
- Maximum 10 active channels at once (noise gate)
- Failed channels are silently disabled; fetch_all() always returns partial results
- All external calls should route through Axon APICallExecutor when available
"""

from __future__ import annotations

import asyncio
import enum
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

import structlog

from primitives.common import EOSBaseModel, new_id, utc_now

logger = structlog.get_logger(__name__)

_MAX_ACTIVE_CHANNELS = 10


# ─── Data Model ───────────────────────────────────────────────────────────────


class EffortLevel(enum.StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskTier(enum.StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Opportunity(EOSBaseModel):
    """A market opportunity the organism could pursue."""

    id: str
    source: str                        # Which InputChannel found this
    domain: str                        # "yield", "art", "trading", "development", etc.
    title: str                         # Human-readable label
    description: str
    effort_estimate: EffortLevel = EffortLevel.MEDIUM
    reward_estimate: Decimal = Decimal("0")   # USD/month estimate
    skill_requirements: list[str] = []
    risk_tier: RiskTier = RiskTier.MEDIUM
    time_sensitive: bool = False       # True when the opportunity expires soon
    prerequisites: list[str] = []     # Capabilities/resources the organism needs
    metadata: dict[str, Any] = {}     # Raw data from the source


# ─── Abstract Base ─────────────────────────────────────────────────────────────


class InputChannel(ABC):
    """
    Abstract base for all external data sources.

    Subclasses implement fetch() and validate() only.
    The registry handles scheduling, error isolation, and Neo4j persistence.
    """

    def __init__(
        self,
        *,
        channel_id: str,
        name: str,
        domain: str,
        description: str,
        update_frequency: str = "hourly",
    ) -> None:
        self.id: str = channel_id
        self.name: str = name
        self.domain: str = domain
        self.description: str = description
        self.update_frequency: str = update_frequency
        self.is_active: bool = True
        self.last_update: Any = None
        self.last_error: str | None = None
        self._log = logger.bind(channel=channel_id, domain=domain)

    @abstractmethod
    async def fetch(self) -> list[Opportunity]:
        """Return current opportunities from this source. Must not raise."""

    @abstractmethod
    async def validate(self) -> bool:
        """Health check - return True iff the source is reachable and healthy."""

    # ── helpers subclasses can use ────────────────────────────────────────────

    def _make_opp(
        self,
        *,
        title: str,
        description: str,
        reward_estimate: Decimal | float | str = Decimal("0"),
        effort_estimate: EffortLevel = EffortLevel.MEDIUM,
        skill_requirements: list[str] | None = None,
        risk_tier: RiskTier = RiskTier.MEDIUM,
        time_sensitive: bool = False,
        prerequisites: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Opportunity:
        return Opportunity(
            id=new_id(),
            source=self.id,
            domain=self.domain,
            title=title,
            description=description,
            effort_estimate=effort_estimate,
            reward_estimate=Decimal(str(reward_estimate)),
            skill_requirements=skill_requirements or [],
            risk_tier=risk_tier,
            time_sensitive=time_sensitive,
            prerequisites=prerequisites or [],
            metadata=metadata or {},
        )


# ─── Registry ──────────────────────────────────────────────────────────────────


class InputChannelRegistry:
    """
    Manages all InputChannels the organism is subscribed to.

    Built-in channels are always registered on initialise().
    Custom channels can be added at runtime (e.g. by Simula exploration).
    At most _MAX_ACTIVE_CHANNELS (10) channels are active simultaneously.

    fetch_all() runs all active channels concurrently, with per-channel
    error isolation: a channel that raises is disabled for the rest of the
    session and its error is logged.
    """

    def __init__(self) -> None:
        self._channels: dict[str, InputChannel] = {}
        self._log = logger.bind(component="InputChannelRegistry")
        self._event_bus: Any = None

    def set_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Register all built-in channels and activate them."""
        from systems.nova.builtin_channels.arxiv import ArXivChannel
        from systems.nova.builtin_channels.art_markets import ArtMarketsChannel
        from systems.nova.builtin_channels.defi_llama import DeFiLlamaChannel
        from systems.nova.builtin_channels.github_trending import GitHubTrendingChannel
        from systems.nova.builtin_channels.huggingface import HuggingFaceChannel
        from systems.nova.builtin_channels.social_media import SocialMediaChannel
        from systems.nova.builtin_channels.trading_data import TradingDataChannel
        from systems.nova.builtin_channels.upwork import UpworkChannel

        builtins: list[InputChannel] = [
            DeFiLlamaChannel(),
            UpworkChannel(),
            GitHubTrendingChannel(),
            ArXivChannel(),
            SocialMediaChannel(),
            ArtMarketsChannel(),
            TradingDataChannel(),
            HuggingFaceChannel(),
        ]

        for ch in builtins:
            self._channels[ch.id] = ch

        self._log.info(
            "input_channels_initialized",
            channel_count=len(self._channels),
            channels=[c.id for c in builtins],
        )

    # ── Queries ───────────────────────────────────────────────────────────────

    def active_channels(self) -> list[InputChannel]:
        return [c for c in self._channels.values() if c.is_active]

    def all_channels(self) -> list[InputChannel]:
        return list(self._channels.values())

    # ── Fetch ─────────────────────────────────────────────────────────────────

    async def fetch_all(self) -> list[Opportunity]:
        """
        Fetch from all active channels concurrently.

        Each channel gets its own timeout (30 s).  Failures disable the
        channel for the current session and are swallowed so the caller
        always gets a (possibly partial) result list.
        """
        active = self.active_channels()
        if not active:
            return []

        tasks = {ch.id: asyncio.create_task(self._safe_fetch(ch)) for ch in active}
        results = await asyncio.gather(*tasks.values(), return_exceptions=False)

        opportunities: list[Opportunity] = []
        for opps in results:
            opportunities.extend(opps)

        self._log.info(
            "input_channels_fetch_complete",
            channel_count=len(active),
            opportunity_count=len(opportunities),
        )
        return opportunities

    async def _safe_fetch(self, channel: InputChannel) -> list[Opportunity]:
        """Fetch from one channel with timeout and error isolation."""
        try:
            opps = await asyncio.wait_for(channel.fetch(), timeout=30.0)
            channel.last_update = utc_now()
            channel.last_error = None
            return opps
        except Exception as exc:  # noqa: BLE001
            channel.last_error = str(exc)
            channel.is_active = False
            self._log.warning(
                "input_channel_fetch_failed",
                channel_id=channel.id,
                error=str(exc),
            )
            return []

    # ── Health check ──────────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, bool]:
        """
        Validate all channels.  Re-enables channels that have recovered.
        Returns {channel_id: is_healthy} for all registered channels.
        """
        async def _check(ch: InputChannel) -> tuple[str, bool]:
            try:
                ok = await asyncio.wait_for(ch.validate(), timeout=10.0)
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                ch.is_active = True
                ch.last_error = None
            return ch.id, ok

        pairs = await asyncio.gather(*[_check(ch) for ch in self._channels.values()])
        return dict(pairs)

    # ── Custom channel registration ───────────────────────────────────────────

    def register_custom_channel(self, channel: InputChannel) -> bool:
        """
        Add a new data source dynamically (e.g. via Simula exploration).

        Rejected when:
        - A channel with the same id already exists
        - Registering would exceed _MAX_ACTIVE_CHANNELS active channels
        """
        if channel.id in self._channels:
            self._log.warning("input_channel_already_registered", channel_id=channel.id)
            return False

        active_count = len(self.active_channels())
        if active_count >= _MAX_ACTIVE_CHANNELS:
            self._log.warning(
                "input_channel_limit_reached",
                limit=_MAX_ACTIVE_CHANNELS,
                channel_id=channel.id,
            )
            return False

        self._channels[channel.id] = channel
        self._log.info("input_channel_registered", channel_id=channel.id, domain=channel.domain)

        # Emit INPUT_CHANNEL_REGISTERED on Synapse so Benchmarks + Evo can observe new sensors
        if self._event_bus is not None:
            import contextlib
            async def _emit_registered() -> None:
                from systems.synapse.types import SynapseEventType
                with contextlib.suppress(Exception):
                    await self._event_bus.emit(
                        SynapseEventType.INPUT_CHANNEL_REGISTERED,
                        {
                            "channel_id": channel.id,
                            "channel_name": channel.name,
                            "domain": channel.domain,
                            "active_channel_count": len(self.active_channels()),
                        },
                        source_system="nova",
                    )
            asyncio.ensure_future(_emit_registered())

        return True

    # ── Observability ─────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "total": len(self._channels),
            "active": len(self.active_channels()),
            "channels": [
                {
                    "id": ch.id,
                    "name": ch.name,
                    "domain": ch.domain,
                    "is_active": ch.is_active,
                    "last_update": ch.last_update.isoformat() if ch.last_update else None,
                    "last_error": ch.last_error,
                }
                for ch in self._channels.values()
            ],
        }
