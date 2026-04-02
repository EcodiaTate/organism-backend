"""
EcodiaOS - GitHub API Client

Fetches open bounty issues from GitHub's search API.
Extracts reward amounts from labels (e.g. "bounty: $500") and issue bodies.
Handles rate-limit responses with a single retry after the reset window.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

import httpx
import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from config import ExternalPlatformsConfig

logger = structlog.get_logger(__name__)

# ─── Result types ─────────────────────────────────────────────────


class BountyIssue(BaseModel):
    """A single open bounty issue retrieved from GitHub."""

    id: int
    title: str
    url: str  # HTML URL for the issue
    repo: str  # owner/repo
    body: str | None = None
    reward_usd: float | None = None  # inferred from labels / body; None if unknown
    labels: list[str] = Field(default_factory=list)


class BountySearchResult(BaseModel):
    total_count: int
    items: list[BountyIssue]
    incomplete_results: bool = False


# ─── Reward extraction ────────────────────────────────────────────

# Patterns tried in order; first match wins.
_REWARD_PATTERNS: list[re.Pattern[str]] = [
    # Label: "bounty: $500", "bounty $1,000", "💰 $250"
    re.compile(r"\$\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    # "USD 500", "500 USD"
    re.compile(r"(?:USD\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*USD)", re.IGNORECASE),
    # "500 USDC / USDT"
    re.compile(r"([\d,]+(?:\.\d+)?)\s*USD[CT]", re.IGNORECASE),
]


def _extract_reward(labels: list[str], body: str | None) -> float | None:
    """Return the first numeric USD amount found in labels or body."""
    candidates = labels + ([body] if body else [])
    for text in candidates:
        for pat in _REWARD_PATTERNS:
            m = pat.search(text)
            if m:
                raw = next(g for g in m.groups() if g is not None)
                try:
                    return float(raw.replace(",", ""))
                except ValueError:
                    continue
    return None


# ─── Client ───────────────────────────────────────────────────────

_GITHUB_API = "https://api.github.com"
_SEARCH_PATH = "/search/issues"
# GitHub search returns at most 1 000 results (100 per page × 10 pages).
_MAX_PER_PAGE = 100


class GitHubClient:
    """
    Async client for GitHub's REST API v3.

    Usage::

        async with GitHubClient(config.external_platforms) as client:
            result = await client.search_open_bounties(min_reward=100)

    The client can also be used without a context manager if you manage
    ``connect()`` / ``close()`` yourself.
    """

    def __init__(self, config: ExternalPlatformsConfig) -> None:
        self._config = config
        self._http: httpx.AsyncClient | None = None

    # ── Lifecycle ─────────────────────────────────────────────────

    async def connect(self) -> None:
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._config.github_token:
            headers["Authorization"] = f"Bearer {self._config.github_token}"

        self._http = httpx.AsyncClient(
            base_url=_GITHUB_API,
            headers=headers,
            timeout=httpx.Timeout(30.0),
        )

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> GitHubClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ── Public API ────────────────────────────────────────────────

    async def search_open_bounties(
        self,
        min_reward: float = 0,
        extra_labels: list[str] | None = None,
        max_pages: int = 3,
        base_label: str = "bounty",
    ) -> BountySearchResult:
        """
        Search GitHub for open issues labelled with a bounty marker.

        Args:
            min_reward: Only return issues where the inferred reward ≥ this value.
                        Set to 0 (default) to return all bounty issues regardless of reward.
            extra_labels: Additional labels to AND-filter by (e.g. ["good first issue"]).
            max_pages: How many pages to fetch (each page = 100 results). Cap at 10.
            base_label: Primary label to search for (default ``"bounty"``).  Override to
                        target platform-specific labels such as ``"💎 Bounty"`` (Algora).

        Returns:
            BountySearchResult with matched issues, total count, and completeness flag.
        """
        assert self._http is not None, "Call connect() or use as async context manager first"

        # Build query - search for open issues with a "bounty" label.
        # GitHub's search API 422s when OR is mixed with is: qualifiers at the top
        # level (it reinterprets OR as a top-level boolean, breaking the is: scope).
        # Use a single focused label; extra_labels are AND'd in (intersect, not union).
        q = f'is:open is:issue label:"{base_label}"'
        if extra_labels:
            for lbl in extra_labels:
                q += f' label:"{lbl}"'

        all_items: list[BountyIssue] = []
        total_count = 0
        incomplete = False

        for page in range(1, min(max_pages, 10) + 1):
            params = {"q": q, "per_page": _MAX_PER_PAGE, "page": page}
            data = await self._get(_SEARCH_PATH, params=params)

            if page == 1:
                total_count = data.get("total_count", 0)
                incomplete = data.get("incomplete_results", False)

            raw_items: list[dict] = data.get("items", [])
            if not raw_items:
                break

            for item in raw_items:
                issue = self._parse_issue(item)
                if min_reward > 0 and (issue.reward_usd is None or issue.reward_usd < min_reward):
                    continue
                all_items.append(issue)

            # Stop early if this was the last page.
            if len(raw_items) < _MAX_PER_PAGE:
                break

        logger.info(
            "github.bounty_search.done",
            total_api_count=total_count,
            returned=len(all_items),
            min_reward=min_reward,
            pages_fetched=page,
        )

        return BountySearchResult(
            total_count=total_count,
            items=all_items,
            incomplete_results=incomplete,
        )

    # ── Internals ─────────────────────────────────────────────────

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """
        Perform a GET request, handling rate-limits with a single retry.

        Raises:
            httpx.HTTPStatusError: on non-rate-limit HTTP errors.
            RuntimeError: if rate-limited twice in a row.
        """
        assert self._http is not None

        for attempt in range(2):
            response = await self._http.get(path, params=params)

            if response.status_code == 403 or response.status_code == 429:
                reset_at = response.headers.get("X-RateLimit-Reset")
                remaining = response.headers.get("X-RateLimit-Remaining", "?")

                if attempt == 0 and reset_at is not None:
                    wait_s = max(0.0, float(reset_at) - time.time()) + 1.0
                    logger.warning(
                        "github.rate_limited",
                        remaining=remaining,
                        wait_s=round(wait_s, 1),
                        attempt=attempt + 1,
                    )
                    import asyncio

                    await asyncio.sleep(min(wait_s, 60.0))
                    continue

                raise RuntimeError(
                    f"GitHub rate limit exceeded (remaining={remaining}). "
                    "Set ORGANISM_EXTERNAL_PLATFORMS__GITHUB_TOKEN to raise the limit."
                )

            response.raise_for_status()
            return response.json()  # type: ignore[return-value]

        raise RuntimeError("GitHub API GET failed after retry")  # pragma: no cover

    @staticmethod
    def _parse_issue(item: dict) -> BountyIssue:
        labels = [lbl.get("name", "") for lbl in item.get("labels", [])]
        body: str | None = item.get("body")
        reward = _extract_reward(labels, body)

        repo_url: str = item.get("repository_url", "")
        # repository_url: https://api.github.com/repos/owner/repo
        repo = repo_url.removeprefix(f"{_GITHUB_API}/repos/")

        return BountyIssue(
            id=item["id"],
            title=item.get("title", ""),
            url=item.get("html_url", ""),
            repo=repo,
            body=body,
            reward_usd=reward,
            labels=labels,
        )
