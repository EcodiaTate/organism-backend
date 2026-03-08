"""
ArXivChannel — research opportunities from recent ArXiv preprints.

Uses the ArXiv public API (Atom feed, no auth required).
Maps new papers to Opportunity objects that signal:
  - Research directions worth pursuing
  - Gaps the organism might fill with novel work
  - Replication / commercialisation opportunities
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from decimal import Decimal
from typing import Any

import httpx

from systems.nova.input_channels import EffortLevel, InputChannel, Opportunity, RiskTier

_BASE_URL = "http://export.arxiv.org/api/query"
_CATEGORIES = ["cs.AI", "cs.LG", "cs.CR", "q-fin.TR", "cs.NE"]
_FETCH_TIMEOUT = 20.0
_ATOM_NS = "http://www.w3.org/2005/Atom"


class ArXivChannel(InputChannel):
    """Recent ArXiv preprints as research / commercialisation opportunity signals."""

    def __init__(self) -> None:
        super().__init__(
            channel_id="arxiv",
            name="ArXiv Recent Papers",
            domain="research",
            description=(
                "Recent AI, ML, security, and quantitative finance preprints from ArXiv. "
                "Surfaces new research directions and replication opportunities."
            ),
            update_frequency="daily",
        )

    async def fetch(self) -> list[Opportunity]:
        opportunities: list[Opportunity] = []

        for cat in _CATEGORIES:
            try:
                opps = await self._fetch_category(cat)
                opportunities.extend(opps)
            except Exception as exc:
                self._log.warning("arxiv_fetch_error", category=cat, error=str(exc))

        self._log.info("arxiv_fetched", opportunity_count=len(opportunities))
        return opportunities

    async def _fetch_category(self, category: str) -> list[Opportunity]:
        params: dict[str, Any] = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": 5,
        }

        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
            resp = await client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            xml_text = resp.text

        root = ET.fromstring(xml_text)
        entries = root.findall(f"{{{_ATOM_NS}}}entry")

        result: list[Opportunity] = []
        for entry in entries[:3]:
            title_el = entry.find(f"{{{_ATOM_NS}}}title")
            summary_el = entry.find(f"{{{_ATOM_NS}}}summary")
            id_el = entry.find(f"{{{_ATOM_NS}}}id")

            title = (title_el.text or "").strip() if title_el is not None else "ArXiv paper"
            summary = (summary_el.text or "").strip()[:300] if summary_el is not None else ""
            arxiv_id = (id_el.text or "").strip() if id_el is not None else ""

            result.append(
                self._make_opp(
                    title=f"[ArXiv/{category}] {title[:100]}",
                    description=summary or f"Recent preprint in {category}.",
                    # Research doesn't have direct financial yield — low estimate
                    reward_estimate=Decimal("0"),
                    effort_estimate=EffortLevel.HIGH,
                    skill_requirements=["research", "academic_writing", category.split(".")[1].lower()],
                    risk_tier=RiskTier.LOW,
                    time_sensitive=False,
                    prerequisites=["research_capability"],
                    metadata={
                        "arxiv_id": arxiv_id,
                        "category": category,
                        "title": title,
                    },
                )
            )
        return result

    async def validate(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    _BASE_URL,
                    params={"search_query": "cat:cs.AI", "max_results": 1},
                )
                return resp.status_code == 200
        except Exception:
            return False
