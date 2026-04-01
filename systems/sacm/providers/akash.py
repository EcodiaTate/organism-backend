"""
EcodiaOS - SACM Akash Network Substrate Provider

Translates Akash Network marketplace pricing into normalised SubstrateOffers
for the SACM optimizer.

Akash pricing model:
  Providers bid on deployments in uAKT (micro-AKT, 1 AKT = 1,000,000 uAKT).
  Bids are per-unit per-month for CPU, memory, and storage.  GPU pricing
  is per-GPU per-month.  We convert to USD/second using:

    usd_per_second = (uakt_per_month / 1_000_000) * akt_usd / (730 * 3600)

  where 730 hours/month is the standard Akash billing assumption.

Queries the real Akash Console API at /v1/pricing and CoinGecko for the
AKT/USD exchange rate.  Falls back to the last known exchange rate if
CoinGecko is unreachable - never fabricates pricing data.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
import structlog

from primitives.common import utc_now
from systems.sacm.providers.base import (
    SubstrateOffer,
    SubstrateProvider,
    SubstrateProviderStatus,
)
from systems.sacm.workload import OffloadClass

logger = structlog.get_logger("systems.sacm.providers.akash")

# ─── Akash Market Constants ──────────────────────────────────────

_UAKT_PER_AKT: int = 1_000_000
_HOURS_PER_MONTH: float = 730.0
_SECONDS_PER_HOUR: int = 3600
_SECONDS_PER_MONTH: float = _HOURS_PER_MONTH * _SECONDS_PER_HOUR

# Initial AKT/USD rate - used until the first successful CoinGecko fetch.
# This is the March 2026 approximate rate; updated on every successful refresh.
_INITIAL_AKT_USD: float = 3.80

# Hardcoded fallback pricing (USD/second) used when the Akash API is unavailable
# on first startup (no live data yet). These reflect approximate March 2026 rates.
# CPU: ~$0.001/vCPU-hour → per second
_FALLBACK_CPU_USD_S: float = 0.001 / 3600.0
# Memory: ~$0.0003/GiB-hour → per second
_FALLBACK_MEM_USD_S: float = 0.0003 / 3600.0
# GPU: ~$0.001/GPU-hour → per second (conservative; actual GPU costs are higher
# for A100-class - this ensures offers appear in the pricing surface at startup)
_FALLBACK_GPU_USD_S: float = 0.001 / 3600.0
# Storage: ~$0.00001/GiB-hour → per second
_FALLBACK_STO_USD_S: float = 0.00001 / 3600.0

# Egress is included in Akash leases (not separately billed), but we model
# a nominal cost so the optimizer can compare against providers that do charge.
_NOMINAL_EGRESS_USD_PER_GIB: float = 0.01

# GPU VRAM for A100-class nodes (the most common Akash GPU offering)
_A100_VRAM_GIB: float = 40.0


def _uakt_month_to_usd_second(uakt_per_month: float, akt_usd: float) -> float:
    """Convert uAKT/month to USD/second."""
    akt_per_month = uakt_per_month / _UAKT_PER_AKT
    usd_per_month = akt_per_month * akt_usd
    return usd_per_month / _SECONDS_PER_MONTH


# ─── Provider Implementation ─────────────────────────────────────


class SACMAkashProvider(SubstrateProvider):
    """
    Akash Network substrate provider for SACM.

    Wraps the Akash decentralised compute marketplace and translates
    uAKT bid pricing into USD-normalised SubstrateOffers.

    Distinct from infrastructure/providers/akash.py which handles
    full deployment lifecycle - this class only deals with market
    pricing for the optimizer.
    """

    def __init__(
        self,
        api_url: str = "https://console-api.akash.network",
        offer_ttl_minutes: int = 5,
        request_timeout_s: float = 15.0,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._offer_ttl = timedelta(minutes=offer_ttl_minutes)
        self._request_timeout = request_timeout_s
        self._log = logger.bind(provider="sacm-akash")
        self._cached_akt_usd: float = _INITIAL_AKT_USD

    @property
    def provider_id(self) -> str:
        return "akash"

    @property
    def provider_name(self) -> str:
        return "Akash Network"

    async def fetch_offers(self) -> list[SubstrateOffer]:
        """
        Query the Akash Console API for current market pricing and return
        SubstrateOffers for CPU and GPU tiers.

        On API failure or first-startup unavailability, returns fallback offers
        using hardcoded conservative default rates so the optimizer always has
        Akash as a candidate rather than silently dropping it from the surface.
        Fallback offers are marked with metadata.pricing_source = 'fallback'.
        Never raises.
        """
        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                resp = await client.get(f"{self._api_url}/v1/pricing")
                if resp.status_code != 200:
                    self._log.warning(
                        "akash_pricing_api_error",
                        status=resp.status_code,
                        body=resp.text[:200],
                    )
                    return self._fallback_offers()

                data = resp.json()

                # Parse per-unit monthly costs (uAKT)
                cpu_uakt = float(data.get("cpu", 0))
                mem_uakt = float(data.get("memory", 0))
                sto_uakt = float(data.get("storage", 0))
                gpu_uakt = float(data.get("gpu", 0))

                if cpu_uakt == 0:
                    self._log.warning("akash_pricing_missing_cpu_field")
                    return self._fallback_offers()

                # Refresh AKT/USD exchange rate (best-effort)
                await self._refresh_akt_price(client)
                akt_usd = self._cached_akt_usd

                now = utc_now()
                valid_until = now + self._offer_ttl

                offers: list[SubstrateOffer] = []

                # ── CPU/general tier ──
                cpu_offer = SubstrateOffer(
                    provider_id="akash",
                    provider_name="Akash Network",
                    region="decentralised",
                    supported_classes=[
                        OffloadClass.GENERAL,
                        OffloadClass.CPU_BOUND,
                        OffloadClass.MEMORY_INTENSIVE,
                        OffloadClass.IO_BOUND,
                    ],
                    max_cpu_vcpu=32.0,
                    max_memory_gib=64.0,
                    max_gpu_units=0.0,
                    gpu_vram_gib=0.0,
                    max_storage_gib=512.0,
                    price_cpu_per_vcpu_s=_uakt_month_to_usd_second(cpu_uakt, akt_usd),
                    price_mem_per_gib_s=_uakt_month_to_usd_second(mem_uakt, akt_usd),
                    price_gpu_per_unit_s=0.0,
                    price_storage_per_gib_s=_uakt_month_to_usd_second(sto_uakt, akt_usd),
                    price_egress_per_gib=_NOMINAL_EGRESS_USD_PER_GIB,
                    status=SubstrateProviderStatus.AVAILABLE,
                    valid_until=valid_until,
                    trust_score=0.85,
                    avg_latency_overhead_s=8.0,
                    currency_native="uAKT",
                    price_native_per_hour=self._native_hourly(
                        cpu_uakt, mem_uakt, sto_uakt, gpu_uakt=0.0,
                    ),
                    exchange_rate_to_usd=akt_usd / _UAKT_PER_AKT,
                    metadata={
                        "tier": "cpu",
                        "pricing_source": "live",
                        "akt_usd_rate": str(round(akt_usd, 4)),
                        "cpu_uakt_month": str(cpu_uakt),
                        "mem_uakt_month": str(mem_uakt),
                        "sto_uakt_month": str(sto_uakt),
                    },
                )
                offers.append(cpu_offer)

                # ── GPU tier (only if GPU pricing is available) ──
                if gpu_uakt > 0:
                    gpu_offer = SubstrateOffer(
                        provider_id="akash",
                        provider_name="Akash Network (GPU)",
                        region="decentralised",
                        supported_classes=[
                            OffloadClass.GPU_HEAVY,
                            OffloadClass.GENERAL,
                        ],
                        max_cpu_vcpu=16.0,
                        max_memory_gib=128.0,
                        max_gpu_units=1.0,
                        gpu_vram_gib=_A100_VRAM_GIB,
                        max_storage_gib=256.0,
                        price_cpu_per_vcpu_s=_uakt_month_to_usd_second(cpu_uakt, akt_usd),
                        price_mem_per_gib_s=_uakt_month_to_usd_second(mem_uakt, akt_usd),
                        price_gpu_per_unit_s=_uakt_month_to_usd_second(gpu_uakt, akt_usd),
                        price_storage_per_gib_s=_uakt_month_to_usd_second(sto_uakt, akt_usd),
                        price_egress_per_gib=_NOMINAL_EGRESS_USD_PER_GIB,
                        status=SubstrateProviderStatus.AVAILABLE,
                        valid_until=valid_until,
                        trust_score=0.80,
                        avg_latency_overhead_s=12.0,
                        currency_native="uAKT",
                        price_native_per_hour=self._native_hourly(
                            cpu_uakt, mem_uakt, sto_uakt, gpu_uakt,
                        ),
                        exchange_rate_to_usd=akt_usd / _UAKT_PER_AKT,
                        metadata={
                            "tier": "gpu",
                            "gpu_model": "A100-40GB",
                            "pricing_source": "live",
                            "akt_usd_rate": str(round(akt_usd, 4)),
                            "gpu_uakt_month": str(gpu_uakt),
                        },
                    )
                    offers.append(gpu_offer)

                self._log.info(
                    "akash_offers_fetched",
                    offer_count=len(offers),
                    akt_usd=round(akt_usd, 2),
                    cpu_usd_per_vcpu_h=round(
                        cpu_offer.price_cpu_per_vcpu_s * _SECONDS_PER_HOUR, 6
                    ),
                )

                return offers

        except Exception as exc:
            self._log.error("akash_fetch_offers_failed", error=str(exc))
            return self._fallback_offers()

    async def health(self) -> SubstrateProviderStatus:
        """Check Akash Console API reachability."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._api_url}/v1/pricing")
                if resp.status_code == 200:
                    return SubstrateProviderStatus.AVAILABLE
                self._log.warning("akash_health_degraded", status=resp.status_code)
                return SubstrateProviderStatus.DEGRADED
        except Exception as exc:
            self._log.error("akash_health_unreachable", error=str(exc))
            return SubstrateProviderStatus.UNREACHABLE

    # ── Internal helpers ──────────────────────────────────────────

    def _fallback_offers(self) -> list[SubstrateOffer]:
        """
        Return conservative hardcoded SubstrateOffers for use when the
        Akash Console API is unavailable (first startup or sustained outage).

        These rates ensure Akash always appears in the optimizer's pricing
        surface rather than being silently absent, which would cause the
        optimizer to over-index on other providers. Marked with
        metadata.pricing_source = 'fallback' so callers can treat them
        with lower trust if desired.
        """
        now = utc_now()
        valid_until = now + self._offer_ttl
        return [
            SubstrateOffer(
                provider_id="akash",
                provider_name="Akash Network",
                region="decentralised",
                supported_classes=[
                    OffloadClass.GENERAL,
                    OffloadClass.CPU_BOUND,
                    OffloadClass.MEMORY_INTENSIVE,
                    OffloadClass.IO_BOUND,
                ],
                max_cpu_vcpu=32.0,
                max_memory_gib=64.0,
                max_gpu_units=0.0,
                gpu_vram_gib=0.0,
                max_storage_gib=512.0,
                price_cpu_per_vcpu_s=_FALLBACK_CPU_USD_S,
                price_mem_per_gib_s=_FALLBACK_MEM_USD_S,
                price_gpu_per_unit_s=0.0,
                price_storage_per_gib_s=_FALLBACK_STO_USD_S,
                price_egress_per_gib=_NOMINAL_EGRESS_USD_PER_GIB,
                status=SubstrateProviderStatus.DEGRADED,
                valid_until=valid_until,
                trust_score=0.60,
                avg_latency_overhead_s=15.0,
                currency_native="uAKT",
                price_native_per_hour=0.0,
                exchange_rate_to_usd=self._cached_akt_usd / _UAKT_PER_AKT,
                metadata={
                    "tier": "cpu",
                    "pricing_source": "fallback",
                    "akt_usd_rate": str(round(self._cached_akt_usd, 4)),
                },
            ),
        ]

    @staticmethod
    def _native_hourly(
        cpu_uakt: float,
        mem_uakt: float,
        sto_uakt: float,
        gpu_uakt: float,
    ) -> float:
        """Compute native hourly cost in uAKT for the EOS reference workload."""
        ref_cpu, ref_mem, ref_sto = 2.0, 4.0, 20.0
        total_uakt_month = (
            cpu_uakt * ref_cpu
            + mem_uakt * ref_mem
            + sto_uakt * ref_sto
            + gpu_uakt * (1.0 if gpu_uakt > 0 else 0.0)
        )
        return total_uakt_month / _HOURS_PER_MONTH

    async def _refresh_akt_price(self, client: httpx.AsyncClient) -> None:
        """Fetch current AKT/USD from CoinGecko. Best-effort - cached fallback on failure."""
        try:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "akash-network", "vs_currencies": "usd"},
                timeout=5.0,
            )
            if resp.status_code == 200:
                price = resp.json().get("akash-network", {}).get("usd")
                if price and float(price) > 0:
                    self._cached_akt_usd = float(price)
                    self._log.debug("akt_price_refreshed", rate=self._cached_akt_usd)
        except Exception as exc:
            self._log.debug("akt_price_refresh_failed", error=str(exc))
