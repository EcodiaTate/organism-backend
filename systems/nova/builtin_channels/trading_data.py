"""
TradingDataChannel - crypto / stock tradeable instrument signals.

Uses CoinGecko's public API (no auth for basic market data, 30 req/min free tier)
to identify high-volatility, high-liquidity trading opportunities.

Maps instruments with significant price movement to Opportunity objects that
tell the organism: "this asset is moving - trading opportunity exists."
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx

from systems.nova.input_channels import EffortLevel, InputChannel, Opportunity, RiskTier

_COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
_FETCH_TIMEOUT = 20.0
_MIN_VOLUME_USD = 10_000_000  # Minimum daily volume for serious liquidity
_HIGH_VOLATILITY_THRESHOLD = 5.0  # 5% 24h price change = notable move


class TradingDataChannel(InputChannel):
    """Crypto trading opportunities from CoinGecko public market data."""

    def __init__(self) -> None:
        super().__init__(
            channel_id="trading_data",
            name="Crypto Market Trading Signals",
            domain="trading",
            description=(
                "High-volatility, high-liquidity crypto assets from CoinGecko. "
                "Surfaces instruments where momentum trading or arbitrage may be viable."
            ),
            update_frequency="hourly",
        )

    async def fetch(self) -> list[Opportunity]:
        try:
            movers = await self._fetch_top_movers()
        except Exception as exc:
            self._log.warning("trading_data_fetch_error", error=str(exc))
            return []

        self._log.info("trading_data_fetched", opportunity_count=len(movers))
        return movers

    async def _fetch_top_movers(self) -> list[Opportunity]:
        params: dict[str, Any] = {
            "vs_currency": "usd",
            "order": "volume_desc",
            "per_page": 50,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h",
        }

        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
            resp = await client.get(_COINGECKO_MARKETS, params=params)
            if resp.status_code == 429:
                # Rate limited - return empty
                return []
            resp.raise_for_status()
            coins: list[dict[str, Any]] = resp.json()

        opportunities: list[Opportunity] = []
        for coin in coins:
            volume: float = float(coin.get("total_volume") or 0)
            change_24h: float = float(coin.get("price_change_percentage_24h") or 0)

            if volume < _MIN_VOLUME_USD:
                continue
            if abs(change_24h) < _HIGH_VOLATILITY_THRESHOLD:
                continue

            symbol: str = (coin.get("symbol") or "?").upper()
            name: str = coin.get("name") or symbol
            price: float = float(coin.get("current_price") or 0)
            direction = "up" if change_24h > 0 else "down"

            # Notional: 1% position at $10,000 → $100 at risk; potential 5-15% swing
            estimated_reward = Decimal(str(round(100 * abs(change_24h) / 100, 2)))

            risk = (
                RiskTier.HIGH if abs(change_24h) > 15
                else RiskTier.MEDIUM if abs(change_24h) > 8
                else RiskTier.LOW
            )

            opportunities.append(
                self._make_opp(
                    title=f"[Trading] {name} ({symbol}) {change_24h:+.1f}% in 24h",
                    description=(
                        f"{name} moved {abs(change_24h):.1f}% {direction} in 24h. "
                        f"Price: ${price:,.4f}. Volume: ${volume:,.0f}."
                    ),
                    reward_estimate=estimated_reward,
                    effort_estimate=EffortLevel.MEDIUM,
                    skill_requirements=["trading_analysis", "risk_management", "crypto_execution"],
                    risk_tier=risk,
                    time_sensitive=True,
                    prerequisites=["crypto_exchange_account", "trading_capital"],
                    metadata={
                        "coin_id": coin.get("id"),
                        "symbol": symbol,
                        "price_usd": price,
                        "change_24h_pct": change_24h,
                        "volume_usd": volume,
                        "direction": direction,
                    },
                )
            )

        return opportunities[:10]  # Cap at 10 trading signals

    async def validate(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    _COINGECKO_MARKETS,
                    params={"vs_currency": "usd", "per_page": 1},
                )
                return resp.status_code in (200, 429)  # 429 = rate limited but alive
        except Exception:
            return False
