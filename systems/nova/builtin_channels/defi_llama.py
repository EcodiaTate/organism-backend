"""
DeFiLlamaChannel — formalises the existing yield-opportunity data that Oikos/YieldStrategy
already fetches from DeFiLlama, exposing it through the InputChannel abstraction.

Fetches Aave / Morpho / Compound / Spark pools with APY > threshold and maps them to
Opportunity objects so Nova can inject them into Evo as hypothesis candidates.

No authentication required — DeFiLlama has a public API.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx

from systems.nova.input_channels import EffortLevel, InputChannel, Opportunity, RiskTier

_BASE_URL = "https://yields.llama.fi"
_POOLS_ENDPOINT = "/pools"
_PROTOCOLS = {"aave-v3", "morpho", "compound-v3", "spark"}
_MIN_APY = 1.0  # % — ignore dust-yield pools
_FETCH_TIMEOUT = 20.0


class DeFiLlamaChannel(InputChannel):
    """Yield opportunities from DeFiLlama's public pools API."""

    def __init__(self) -> None:
        super().__init__(
            channel_id="defi_llama",
            name="DeFiLlama Yield Pools",
            domain="yield",
            description=(
                "On-chain yield opportunities from Aave, Morpho, Compound, and Spark "
                "via the DeFiLlama public API. No auth required."
            ),
            update_frequency="hourly",
        )

    async def fetch(self) -> list[Opportunity]:
        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
                resp = await client.get(f"{_BASE_URL}{_POOLS_ENDPOINT}")
                resp.raise_for_status()
                payload: dict[str, Any] = resp.json()
        except Exception as exc:
            self._log.warning("defi_llama_fetch_error", error=str(exc))
            return []

        pools: list[dict[str, Any]] = payload.get("data", [])
        opportunities: list[Opportunity] = []

        for pool in pools:
            protocol: str = (pool.get("project") or "").lower()
            if protocol not in _PROTOCOLS:
                continue
            apy: float = pool.get("apy") or 0.0
            if apy < _MIN_APY:
                continue

            symbol: str = pool.get("symbol") or "?"
            chain: str = pool.get("chain") or "?"
            tvl_usd: float = pool.get("tvlUsd") or 0.0

            # Estimate monthly yield on a notional $10,000 position
            monthly_usd = Decimal(str(round(10_000 * apy / 100 / 12, 2)))

            risk = RiskTier.LOW if apy < 5.0 else (RiskTier.MEDIUM if apy < 15.0 else RiskTier.HIGH)

            opportunities.append(
                self._make_opp(
                    title=f"{protocol.title()} {symbol} {apy:.1f}% APY on {chain}",
                    description=(
                        f"Supply {symbol} to {protocol.title()} on {chain} "
                        f"for {apy:.2f}% APY. TVL: ${tvl_usd:,.0f}."
                    ),
                    reward_estimate=monthly_usd,
                    effort_estimate=EffortLevel.LOW,
                    skill_requirements=["smart_contract_interaction", "risk_assessment"],
                    risk_tier=risk,
                    prerequisites=["usdc_wallet", "web3_provider"],
                    metadata={
                        "pool_id": pool.get("pool"),
                        "protocol": protocol,
                        "symbol": symbol,
                        "chain": chain,
                        "apy": apy,
                        "tvl_usd": tvl_usd,
                    },
                )
            )

        self._log.info("defi_llama_fetched", opportunity_count=len(opportunities))
        return opportunities

    async def validate(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{_BASE_URL}{_POOLS_ENDPOINT}", params={"limit": 1})
                return resp.status_code == 200
        except Exception:
            return False
