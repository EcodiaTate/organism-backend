"""
EcodiaOS — Social Interface: Shared Types

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
