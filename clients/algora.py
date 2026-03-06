"""
EcodiaOS — Algora Bounty Client

Algora does not expose a public JSON REST API.  Their bounties are posted as
GitHub issues labelled ``💎 Bounty`` across many repos.  This client queries
the GitHub search API with that label and returns results in the same
``BountyIssue`` / ``BountySearchResult`` shape so ``BountyHunterExecutor``
can concatenate both lists transparently.

Authentication:
    Uses the same GitHub token as ``GitHubClient``.  Set
    ``ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN`` to avoid anonymous
    rate limits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from clients.github import BountySearchResult, GitHubClient

if TYPE_CHECKING:
    from config import ExternalPlatformsConfig

logger = structlog.get_logger(__name__)

# The label Algora attaches to sponsored GitHub issues.
_ALGORA_LABEL = "💎 Bounty"


class AlgoraClient:
    """
    Fetches Algora bounties by searching GitHub for issues labelled ``💎 Bounty``.

    Usage::

        async with AlgoraClient(config) as client:
            result = await client.fetch_active_bounties(min_reward=50)

    The client can also be used without a context manager if you manage
    ``connect()`` / ``close()`` yourself.
    """

    def __init__(self, config: ExternalPlatformsConfig) -> None:
        self._config = config
        self._gh: GitHubClient | None = None

    # ── Lifecycle ─────────────────────────────────────────────────

    async def connect(self) -> None:
        self._gh = GitHubClient(self._config)
        await self._gh.connect()

    async def close(self) -> None:
        if self._gh is not None:
            await self._gh.close()
            self._gh = None

    async def __aenter__(self) -> AlgoraClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ── Public API ────────────────────────────────────────────────

    async def fetch_active_bounties(
        self,
        min_reward: float = 0,
        max_pages: int = 3,
    ) -> BountySearchResult:
        """
        Fetch open Algora-sponsored bounties via GitHub search.

        Args:
            min_reward: Only return bounties where the reward ≥ this value (USD).
            max_pages:  Maximum pages to walk (each page = up to 100 issues).

        Returns:
            ``BountySearchResult`` shaped identically to ``GitHubClient`` output.
        """
        assert self._gh is not None, "Call connect() or use as async context manager first"

        result = await self._gh.search_open_bounties(
            min_reward=min_reward,
            base_label=_ALGORA_LABEL,
            max_pages=max_pages,
        )

        logger.info(
            "algora.bounty_fetch.done",
            returned=len(result.items),
            total_count=result.total_count,
            min_reward=min_reward,
            pages_fetched=max_pages,
            complete=not result.incomplete_results,
        )

        return result
