"""
ArtMarketsChannel — digital art / NFT market demand signals.

Uses OpenSea's public stats API (no auth for basic collection stats) and
the SuperRare public API where available.

Maps trending collections and price movements to Opportunity objects that
tell the organism: "generative art in this style is selling."
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx

from systems.nova.input_channels import EffortLevel, InputChannel, Opportunity, RiskTier

_OPENSEA_STATS_URL = "https://api.opensea.io/api/v2/collections"
_FETCH_TIMEOUT = 20.0

# Curated list of established generative art collections — use their slugs to
# pull current floor / volume data without needing a paid API key.
_COLLECTIONS = [
    "artblocks-curated",
    "fidenza-by-tyler-hobbs",
    "ringers-by-dmitri-cherniak",
]


class ArtMarketsChannel(InputChannel):
    """Digital art market demand from OpenSea collection stats."""

    def __init__(self) -> None:
        super().__init__(
            channel_id="art_markets",
            name="Digital Art Markets (OpenSea)",
            domain="art",
            description=(
                "NFT / digital art market signals from OpenSea collection stats. "
                "Surfaces demand for generative, AI-assisted, and algorithmic art."
            ),
            update_frequency="daily",
        )

    async def fetch(self) -> list[Opportunity]:
        opportunities: list[Opportunity] = []

        for slug in _COLLECTIONS:
            try:
                opp = await self._fetch_collection(slug)
                if opp:
                    opportunities.append(opp)
            except Exception as exc:
                self._log.warning("art_markets_fetch_error", collection=slug, error=str(exc))

        # Always include a generic generative art signal even if API is rate-limited
        if not opportunities:
            opportunities.append(self._generic_signal())

        self._log.info("art_markets_fetched", opportunity_count=len(opportunities))
        return opportunities

    async def _fetch_collection(self, slug: str) -> Opportunity | None:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
            resp = await client.get(
                f"{_OPENSEA_STATS_URL}/{slug}",
                headers={"accept": "application/json"},
            )
            if resp.status_code in (404, 429, 401, 403):
                return None
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        stats: dict[str, Any] = data.get("stats", {})
        floor_price: float = float(stats.get("floor_price", 0) or 0)
        volume_7d: float = float(stats.get("seven_day_volume", 0) or 0)
        name: str = data.get("name") or slug

        if floor_price == 0 and volume_7d == 0:
            return None

        # Estimate potential revenue: 1 piece at floor price
        eth_to_usd = Decimal("3500")  # Rough ETH price placeholder
        revenue = Decimal(str(floor_price)) * eth_to_usd

        return self._make_opp(
            title=f"[Art] {name} — floor {floor_price:.3f} ETH",
            description=(
                f"{name}: 7-day volume {volume_7d:.1f} ETH, "
                f"floor {floor_price:.3f} ETH (~${float(revenue):,.0f})."
            ),
            reward_estimate=revenue,
            effort_estimate=EffortLevel.HIGH,
            skill_requirements=["generative_art", "creative_coding", "nft_minting"],
            risk_tier=RiskTier.HIGH,
            time_sensitive=True,
            prerequisites=["crypto_wallet", "art_creation_capability"],
            metadata={
                "collection_slug": slug,
                "floor_price_eth": floor_price,
                "seven_day_volume_eth": volume_7d,
                "name": name,
            },
        )

    def _generic_signal(self) -> Opportunity:
        return self._make_opp(
            title="[Art] Generative / AI art demand — market active",
            description=(
                "Generative and AI-assisted digital art continues to sell on "
                "OpenSea, SuperRare, and Foundation. Demand for algorithmic "
                "and creative-coding styles is increasing."
            ),
            reward_estimate=Decimal("500"),
            effort_estimate=EffortLevel.HIGH,
            skill_requirements=["generative_art", "creative_coding"],
            risk_tier=RiskTier.HIGH,
            prerequisites=["art_creation_capability", "crypto_wallet"],
            metadata={"source_confidence": "low", "synthetic": True},
        )

    async def validate(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{_OPENSEA_STATS_URL}/artblocks-curated",
                    headers={"accept": "application/json"},
                )
                return resp.status_code in (200, 401, 403, 429)
        except Exception:
            return False
