"""
SocialMediaChannel - emerging niche signals from public social trend APIs.

Rather than requiring Twitter/X API credentials (paid), this channel uses:
  1. Reddit's public JSON API (no auth for read-only listing)
  2. Hacker News Algolia search API (no auth required)

Together they capture what builders and founders are discussing, which reveals
emerging market niches the organism might specialise into.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx

from systems.nova.input_channels import EffortLevel, InputChannel, Opportunity, RiskTier

_HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
_REDDIT_BASE = "https://www.reddit.com"
_SUBREDDITS = ["MachineLearning", "LocalLLaMA", "SideProject", "startups"]
_HN_QUERIES = ["Show HN", "Ask HN: who is hiring", "YC W25"]
_FETCH_TIMEOUT = 20.0


class SocialMediaChannel(InputChannel):
    """Market intelligence from Reddit and Hacker News public APIs."""

    def __init__(self) -> None:
        super().__init__(
            channel_id="social_media",
            name="Social Media Market Signals",
            domain="market_intelligence",
            description=(
                "Trending discussions on Reddit and Hacker News. Surfaces emerging "
                "niches, founder pain points, and hiring signals."
            ),
            update_frequency="daily",
        )

    async def fetch(self) -> list[Opportunity]:
        results: list[Opportunity] = []

        for sub in _SUBREDDITS:
            try:
                results.extend(await self._fetch_reddit(sub))
            except Exception as exc:
                self._log.warning("social_reddit_error", subreddit=sub, error=str(exc))

        for query in _HN_QUERIES:
            try:
                results.extend(await self._fetch_hn(query))
            except Exception as exc:
                self._log.warning("social_hn_error", query=query, error=str(exc))

        self._log.info("social_media_fetched", opportunity_count=len(results))
        return results

    async def _fetch_reddit(self, subreddit: str) -> list[Opportunity]:
        url = f"{_REDDIT_BASE}/r/{subreddit}/hot.json"
        headers = {"User-Agent": "EcodiaOS-MarketDiscovery/1.0"}

        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, headers=headers) as client:
            resp = await client.get(url, params={"limit": 5})
            if resp.status_code in (429, 403):
                return []
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        posts = data.get("data", {}).get("children", [])
        result: list[Opportunity] = []
        for post in posts[:3]:
            p: dict[str, Any] = post.get("data", {})
            title: str = p.get("title", "")[:120]
            score: int = p.get("score", 0)
            desc: str = (p.get("selftext") or "")[:200]

            result.append(
                self._make_opp(
                    title=f"[Reddit/{subreddit}] {title}",
                    description=desc or f"Trending in r/{subreddit} with {score} upvotes.",
                    reward_estimate=Decimal("0"),  # Signal, not direct revenue
                    effort_estimate=EffortLevel.LOW,
                    skill_requirements=["market_research"],
                    risk_tier=RiskTier.LOW,
                    metadata={
                        "subreddit": subreddit,
                        "score": score,
                        "url": p.get("url"),
                        "signal_type": "reddit_trending",
                    },
                )
            )
        return result

    async def _fetch_hn(self, query: str) -> list[Opportunity]:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
            resp = await client.get(
                _HN_SEARCH_URL,
                params={"query": query, "tags": "story", "hitsPerPage": 3},
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        hits: list[dict[str, Any]] = data.get("hits", [])
        result: list[Opportunity] = []
        for hit in hits[:2]:
            title: str = hit.get("title", "")[:120]
            points: int = hit.get("points", 0)

            result.append(
                self._make_opp(
                    title=f"[HN] {title}",
                    description=f"Hacker News: {title}. Points: {points}.",
                    reward_estimate=Decimal("0"),
                    effort_estimate=EffortLevel.LOW,
                    skill_requirements=["market_research"],
                    risk_tier=RiskTier.LOW,
                    metadata={
                        "hn_query": query,
                        "points": points,
                        "url": hit.get("url"),
                        "signal_type": "hn_trending",
                    },
                )
            )
        return result

    async def validate(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    _HN_SEARCH_URL,
                    params={"query": "startup", "hitsPerPage": 1},
                )
                return resp.status_code == 200
        except Exception:
            return False
