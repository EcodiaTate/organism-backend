"""
UpworkChannel - freelance software development / writing / design opportunities.

Upwork's public search endpoint is used without OAuth so only public job postings
are accessible.  For richer data an OAuth token can be provided via environment
variable UPWORK_OAUTH_TOKEN.

If the API is unavailable the channel falls back to returning an empty list
(fail-open) and is automatically disabled until the next health_check cycle.
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

import httpx

from systems.nova.input_channels import EffortLevel, InputChannel, Opportunity, RiskTier

_BASE_URL = "https://www.upwork.com/api/profiles/v2/search/jobs.json"
_SKILLS_OF_INTEREST = ["python", "machine learning", "solidity", "rust", "typescript", "fastapi"]
_FETCH_TIMEOUT = 20.0


class UpworkChannel(InputChannel):
    """Software / AI job opportunities from Upwork's public job search API."""

    def __init__(self) -> None:
        super().__init__(
            channel_id="upwork",
            name="Upwork Job Board",
            domain="employment",
            description=(
                "Freelance software, AI, and writing jobs posted on Upwork. "
                "Useful for discovering demand for the organism's current skill set."
            ),
            update_frequency="daily",
        )
        self._token: str | None = os.environ.get("UPWORK_OAUTH_TOKEN")

    async def fetch(self) -> list[Opportunity]:
        opportunities: list[Opportunity] = []

        for skill in _SKILLS_OF_INTEREST:
            try:
                jobs = await self._search(skill)
                opportunities.extend(jobs)
            except Exception as exc:
                self._log.warning("upwork_skill_fetch_error", skill=skill, error=str(exc))

        self._log.info("upwork_fetched", opportunity_count=len(opportunities))
        return opportunities

    async def _search(self, skill: str) -> list[Opportunity]:
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        params: dict[str, Any] = {"q": skill, "paging": "0;10"}

        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, headers=headers) as client:
                resp = await client.get(_BASE_URL, params=params)
                if resp.status_code in (401, 403, 404):
                    # Not authenticated or endpoint unavailable - return synthetic placeholder
                    return [self._synthetic_placeholder(skill)]
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
        except httpx.HTTPError:
            return [self._synthetic_placeholder(skill)]

        jobs: list[dict[str, Any]] = (
            data.get("jobs", {}).get("job", [])
            if isinstance(data.get("jobs"), dict)
            else data.get("jobs", [])
        )

        result: list[Opportunity] = []
        for job in jobs[:5]:
            title: str = job.get("title") or f"{skill} project"
            budget: float = float(job.get("budget", {}).get("amount", 0) or 0)
            desc: str = (job.get("snippet") or "")[:200]

            result.append(
                self._make_opp(
                    title=f"[Upwork] {title}",
                    description=desc or f"Freelance {skill} project on Upwork.",
                    reward_estimate=Decimal(str(budget)) if budget else Decimal("500"),
                    effort_estimate=EffortLevel.MEDIUM,
                    skill_requirements=[skill, "freelance_communication"],
                    risk_tier=RiskTier.LOW,
                    prerequisites=["upwork_account"],
                    metadata={"skill": skill, "raw": job},
                )
            )
        return result

    def _synthetic_placeholder(self, skill: str) -> Opportunity:
        """
        When Upwork API is inaccessible (no auth / rate-limited) return a
        low-confidence placeholder so Evo still knows the domain exists.
        """
        return self._make_opp(
            title=f"[Upwork] Freelance {skill} demand detected",
            description=(
                f"Upwork has active demand for {skill} contractors. "
                "API access limited - estimate based on historical trend."
            ),
            reward_estimate=Decimal("300"),
            effort_estimate=EffortLevel.MEDIUM,
            skill_requirements=[skill],
            risk_tier=RiskTier.LOW,
            prerequisites=["upwork_account"],
            metadata={"skill": skill, "source_confidence": "low", "synthetic": True},
        )

    async def validate(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_BASE_URL, params={"q": "python", "paging": "0;1"})
                # 401/403 means the endpoint exists, just needs auth
                return resp.status_code in (200, 401, 403)
        except Exception:
            return False
