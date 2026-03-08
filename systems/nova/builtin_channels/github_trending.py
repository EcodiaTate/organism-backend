"""
GitHubTrendingChannel — trending open-source repositories as contribution /
commercialisation opportunities.

Uses GitHub's public search API (no auth for low-volume requests).
A GITHUB_TOKEN env var can be supplied to raise the rate limit from 60 to
5,000 requests/hour.

Each trending repo is mapped to an Opportunity representing a chance to:
  - contribute to a popular project (community reputation / bounties)
  - build on / wrap the library (commercialisation)
  - offer maintenance services
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

import httpx

from systems.nova.input_channels import EffortLevel, InputChannel, Opportunity, RiskTier

_SEARCH_URL = "https://api.github.com/search/repositories"
_TOPICS_OF_INTEREST = ["llm", "agent", "defi", "solidity", "fastapi", "rust", "machine-learning"]
_FETCH_TIMEOUT = 20.0


class GitHubTrendingChannel(InputChannel):
    """Trending open-source projects as contribution / build-on opportunities."""

    def __init__(self) -> None:
        super().__init__(
            channel_id="github_trending",
            name="GitHub Trending Repos",
            domain="development",
            description=(
                "Trending open-source repositories on GitHub. Signals what the dev "
                "community is building — potential contribution targets and niches."
            ),
            update_frequency="daily",
        )
        self._token: str | None = os.environ.get("GITHUB_TOKEN")

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Accept": "application/vnd.github+json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    async def fetch(self) -> list[Opportunity]:
        opportunities: list[Opportunity] = []

        for topic in _TOPICS_OF_INTEREST:
            try:
                opps = await self._search_topic(topic)
                opportunities.extend(opps)
            except Exception as exc:
                self._log.warning("github_trending_topic_error", topic=topic, error=str(exc))

        self._log.info("github_trending_fetched", opportunity_count=len(opportunities))
        return opportunities

    async def _search_topic(self, topic: str) -> list[Opportunity]:
        params = {
            "q": f"topic:{topic} pushed:>2024-01-01",
            "sort": "stars",
            "order": "desc",
            "per_page": 5,
        }
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, headers=self._headers()) as client:
            resp = await client.get(_SEARCH_URL, params=params)
            if resp.status_code == 403:
                # Rate limited — return empty rather than crashing
                return []
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        repos: list[dict[str, Any]] = data.get("items", [])
        result: list[Opportunity] = []
        for repo in repos[:3]:
            name: str = repo.get("full_name", "unknown/repo")
            stars: int = repo.get("stargazers_count", 0)
            desc: str = (repo.get("description") or "")[:200]
            lang: str = repo.get("language") or "unknown"
            has_issues: bool = repo.get("has_issues", False)
            open_issues: int = repo.get("open_issues_count", 0)

            result.append(
                self._make_opp(
                    title=f"[GitHub] {name} ({stars:,} ⭐) [{topic}]",
                    description=(
                        f"{desc or 'No description.'} "
                        f"Language: {lang}. "
                        f"Open issues: {open_issues if has_issues else 'N/A'}."
                    ),
                    # Potential bounty / consultancy revenue estimate
                    reward_estimate=Decimal(str(min(stars // 100 * 50, 5000))),
                    effort_estimate=EffortLevel.MEDIUM,
                    skill_requirements=[lang.lower() if lang != "unknown" else "programming", topic],
                    risk_tier=RiskTier.LOW,
                    prerequisites=["github_account", "git"],
                    metadata={
                        "repo": name,
                        "stars": stars,
                        "language": lang,
                        "topic": topic,
                        "open_issues": open_issues,
                        "url": repo.get("html_url"),
                    },
                )
            )
        return result

    async def validate(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0, headers=self._headers()) as client:
                resp = await client.get(_SEARCH_URL, params={"q": "stars:>1000", "per_page": 1})
                return resp.status_code in (200, 403)  # 403 = rate limited but endpoint exists
        except Exception:
            return False
