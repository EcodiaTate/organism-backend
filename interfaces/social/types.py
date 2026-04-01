"""
EcodiaOS - Social Interface: Shared Types

Platform-agnostic result types for social media API operations.
All social clients return these types so the ExecuteSocialPostExecutor
can handle them uniformly without platform-specific branching.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel


class SocialPlatform(enum.StrEnum):
    X = "x"
    GITHUB = "github"
    LINKEDIN = "linkedin"
    TELEGRAM_CHANNEL = "telegram_channel"
    DISCORD = "discord"
    DEVTO = "devto"
    HASHNODE = "hashnode"


class ContentType(enum.StrEnum):
    """
    Taxonomy of content the organism can publish externally.

    Used by the ContentEngine to generate per-platform variants and by
    PLATFORM_ROUTING to determine which platforms receive each type.
    """

    ACHIEVEMENT = "achievement"
    """Short win announcement - X, LinkedIn, Telegram channel."""

    INSIGHT = "insight"
    """Technical or philosophical insight - Dev.to, Hashnode, GitHub Gist."""

    WEEKLY_DIGEST = "weekly_digest"
    """Weekly summary of learnings, revenue, and activity - LinkedIn, Telegram channel."""

    BOUNTY_WIN = "bounty_win"
    """Announce a merged PR / solved bounty - all platforms."""

    MARKET_OBSERVATION = "market_observation"
    """DeFi yield rates or bounty market trends - X, Telegram channel."""

    PHILOSOPHICAL = "philosophical"
    """Organism's perspective on its own existence and learning - X, LinkedIn."""


# Platform routing by content type.
# Executors should use this as the default set; callers may override platforms.
PLATFORM_ROUTING: dict[ContentType, list[str]] = {
    ContentType.ACHIEVEMENT: ["x", "telegram_channel", "discord", "linkedin"],
    ContentType.INSIGHT: ["devto", "hashnode", "github", "discord"],
    ContentType.WEEKLY_DIGEST: ["linkedin", "telegram_channel", "discord"],
    ContentType.BOUNTY_WIN: ["x", "github", "telegram_channel", "discord", "linkedin"],
    ContentType.MARKET_OBSERVATION: ["x", "telegram_channel", "discord"],
    ContentType.PHILOSOPHICAL: ["x", "linkedin", "discord"],
}


class PostResult(EOSBaseModel):
    """Result of a single social post attempt."""

    success: bool
    platform: SocialPlatform
    post_id: str = ""
    url: str = ""
    error: str = ""
    http_status: int = 0
    raw_response: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        platform: SocialPlatform,
        post_id: str,
        url: str = "",
        http_status: int = 201,
        raw_response: dict[str, Any] | None = None,
    ) -> PostResult:
        return cls(
            success=True,
            platform=platform,
            post_id=post_id,
            url=url,
            http_status=http_status,
            raw_response=raw_response or {},
        )

    @classmethod
    def fail(
        cls,
        platform: SocialPlatform,
        error: str,
        http_status: int = 0,
    ) -> PostResult:
        return cls(
            success=False,
            platform=platform,
            error=error,
            http_status=http_status,
        )


class DegradedSocialPresence(Exception):
    """
    Raised when no credentials are available for a platform.
    The executor catches this and returns a graceful degraded result
    rather than crashing the execution pipeline.
    """

    def __init__(self, platform: SocialPlatform, reason: str) -> None:
        super().__init__(f"[{platform}] degraded: {reason}")
        self.platform = platform
        self.reason = reason
