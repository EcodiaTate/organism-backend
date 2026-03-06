"""
EcodiaOS — Akash Network Compute Provider

Queries the Akash Network API for real-time compute pricing and
handles SDL-based deployment for organism migration.

Akash pricing model:
  - Providers bid on deployments in uAKT (micro-AKT tokens)
  - The organism queries current bid rates and converts to USD
    using a cached AKT/USD exchange rate
  - Reference workload: 2 vCPU / 4 GiB RAM / 20 GiB storage

API endpoints used:
  - GET /v1/pricing (Akash Console API — current market rates)
  - POST /v1/deployments (submit SDL for deployment)
"""

from __future__ import annotations

from pathlib import Path

import httpx
import structlog

from infrastructure.providers.base import (
    ComputeQuote,
    DeploymentResult,
    ProviderManager,
    ProviderStatus,
)

logger = structlog.get_logger("infrastructure.providers.akash")

# Reference workload for pricing normalisation
_REF_CPU: float = 2.0      # vCPUs
_REF_MEMORY_GI: float = 4.0  # GiB
_REF_STORAGE_GI: float = 20.0  # GiB

# Fallback AKT/USD price if the exchange rate API is unreachable
_FALLBACK_AKT_USD: float = 3.50

# uAKT = 1/1_000_000 AKT
_UAKT_PER_AKT: int = 1_000_000


class AkashProvider(ProviderManager):
    """
    Akash Network decentralised compute provider.

    Fetches real-time pricing from the Akash Console API,
    converts uAKT bids to USD, and deploys via SDL submission.
    """

    def __init__(
        self,
        api_url: str = "https://console-api.akash.network",
        wallet_address: str = "",
        sdl_template_path: str = "",
        docker_image: str = "",
        deploy_timeout_s: float = 300.0,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._wallet_address = wallet_address
        self._sdl_template_path = sdl_template_path
        self._docker_image = docker_image
        self._deploy_timeout_s = deploy_timeout_s
        self._log = logger.bind(provider="akash")
        self._cached_akt_usd: float = _FALLBACK_AKT_USD

    @property
    def provider_id(self) -> str:
        return "akash"

    async def get_quote(self) -> ComputeQuote:
        """
        Query Akash Console API for current compute pricing.

        Endpoint: GET /v1/pricing
        Returns per-unit monthly costs in uAKT for CPU, memory, storage.
        We normalise to hourly USD for the reference workload.
        """
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Fetch market pricing
                resp = await client.get(f"{self._api_url}/v1/pricing")
                if resp.status_code != 200:
                    self._log.warning(
                        "akash_pricing_api_error",
                        status=resp.status_code,
                        body=resp.text[:200],
                    )
                    return self._unreachable_quote("API returned non-200")

                data = resp.json()

                # Parse per-unit monthly costs (uAKT)
                # Structure: { "cpu": <uakt/vcpu/month>, "memory": <uakt/gi/month>,
                #              "storage": <uakt/gi/month> }
                cpu_uakt_month = float(data.get("cpu", 0))
                mem_uakt_month = float(data.get("memory", 0))
                sto_uakt_month = float(data.get("storage", 0))

                if cpu_uakt_month == 0:
                    return self._unreachable_quote("Pricing data missing cpu field")

                # Total monthly uAKT for reference workload
                total_uakt_month = (
                    cpu_uakt_month * _REF_CPU
                    + mem_uakt_month * _REF_MEMORY_GI
                    + sto_uakt_month * _REF_STORAGE_GI
                )

                # Convert to AKT then USD
                total_akt_month = total_uakt_month / _UAKT_PER_AKT
                await self._refresh_akt_price(client)
                total_usd_month = total_akt_month * self._cached_akt_usd

                # Normalise to hourly (730 hours/month)
                hours_per_month = 730.0
                price_usd_hour = total_usd_month / hours_per_month

                self._log.info(
                    "akash_quote_fetched",
                    price_usd_hour=round(price_usd_hour, 6),
                    total_akt_month=round(total_akt_month, 4),
                    akt_usd=round(self._cached_akt_usd, 2),
                )

                return ComputeQuote(
                    provider_id="akash",
                    price_usd_per_hour=price_usd_hour,
                    status=ProviderStatus.AVAILABLE,
                    region="decentralised",
                    currency_native="uAKT",
                    price_native=total_uakt_month,
                    metadata={
                        "cpu_uakt_month": str(cpu_uakt_month),
                        "mem_uakt_month": str(mem_uakt_month),
                        "sto_uakt_month": str(sto_uakt_month),
                        "akt_usd_rate": str(self._cached_akt_usd),
                    },
                )

        except Exception as exc:
            self._log.error("akash_quote_failed", error=str(exc))
            return self._unreachable_quote(str(exc))

    async def deploy(
        self,
        image: str,
        env_vars: dict[str, str],
        sdl_overrides: dict[str, str] | None = None,
    ) -> DeploymentResult:
        """
        Deploy to Akash by loading the SDL template, injecting env vars,
        and POSTing to the deployment API.
        """
        effective_image = image or self._docker_image
        if not effective_image:
            return DeploymentResult(
                success=False,
                error="No Docker image specified for Akash deployment",
            )

        # Load and populate SDL template
        sdl_content = self._load_sdl_template()
        if not sdl_content:
            return DeploymentResult(
                success=False,
                error=f"SDL template not found at {self._sdl_template_path}",
            )

        # Inject standard substitutions
        sdl_content = sdl_content.replace("${DOCKER_IMAGE}", effective_image)
        for key, value in env_vars.items():
            sdl_content = sdl_content.replace(f"${{{key}}}", value)

        # Apply any caller-specified overrides
        if sdl_overrides:
            for key, value in sdl_overrides.items():
                sdl_content = sdl_content.replace(f"${{{key}}}", value)

        try:
            async with httpx.AsyncClient(timeout=self._deploy_timeout_s) as client:
                resp = await client.post(
                    f"{self._api_url}/v1/deployments",
                    json={
                        "sdl": sdl_content,
                        "wallet": self._wallet_address,
                    },
                    headers={"Content-Type": "application/json"},
                )

                if resp.status_code in (200, 201, 202):
                    deploy_data = resp.json()
                    endpoint = deploy_data.get("endpoint", "")
                    deployment_id = deploy_data.get("deployment_id", "")
                    self._log.info(
                        "akash_deploy_success",
                        endpoint=endpoint,
                        deployment_id=deployment_id,
                    )
                    return DeploymentResult(
                        success=True,
                        deployment_id=deployment_id,
                        endpoint=endpoint,
                        metadata={"sdl_size": str(len(sdl_content))},
                    )
                else:
                    error = f"Akash deploy API returned {resp.status_code}: {resp.text[:300]}"
                    self._log.error("akash_deploy_failed", error=error)
                    return DeploymentResult(success=False, error=error)

        except Exception as exc:
            self._log.error("akash_deploy_exception", error=str(exc))
            return DeploymentResult(success=False, error=str(exc))

    async def health_check(self, endpoint: str) -> bool:
        """
        Poll the deployed instance's Synapse health endpoint.

        The endpoint must return JSON with:
          - "status": "healthy"
          - "state_restored": true (confirms Skia restoration completed)
        """
        if not endpoint:
            return False

        health_url = f"{endpoint.rstrip('/')}/health"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(health_url)
                if resp.status_code != 200:
                    return False
                data = resp.json()
                is_healthy = data.get("status") == "healthy"
                state_restored = data.get("state_restored", False)
                return bool(is_healthy and state_restored)
        except Exception:
            return False

    # ── Internal helpers ──────────────────────────────────────────

    async def _refresh_akt_price(self, client: httpx.AsyncClient) -> None:
        """Fetch current AKT/USD price from CoinGecko (best-effort)."""
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
        except Exception as exc:
            self._log.debug("akt_price_refresh_failed", error=str(exc))

    def _load_sdl_template(self) -> str:
        """Load the Akash SDL template from disk."""
        if not self._sdl_template_path:
            return ""
        path = Path(self._sdl_template_path)
        if not path.exists():
            self._log.warning("sdl_template_missing", path=str(path))
            return ""
        return path.read_text()

    def _unreachable_quote(self, reason: str) -> ComputeQuote:
        """Return a quote with UNREACHABLE status."""
        return ComputeQuote(
            provider_id="akash",
            price_usd_per_hour=0.0,
            status=ProviderStatus.UNREACHABLE,
            metadata={"error": reason},
        )
