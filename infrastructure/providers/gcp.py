"""
EcodiaOS — GCP Compute Provider (Stub)

Stubbed GCP pricing check for Cloud Run. Returns published on-demand
pricing for the EOS reference workload (2 vCPU / 4 GiB / 20 GiB).

GCP Cloud Run pricing (as of 2025):
  - CPU: $0.00002400 / vCPU-second (always-allocated)
  - Memory: $0.00000250 / GiB-second
  - Minimum instances incur cost even when idle

Once GCP's Billing API integration is wired, this stub will be replaced
with live queries against the Cloud Billing Catalog API.
"""

from __future__ import annotations

import httpx
import structlog

from infrastructure.providers.base import (
    ComputeQuote,
    DeploymentResult,
    ProviderManager,
    ProviderStatus,
)

logger = structlog.get_logger("infrastructure.providers.gcp")

# Published GCP Cloud Run pricing (us-central1, always-allocated)
# Source: cloud.google.com/run/pricing
_GCP_CPU_PER_VCPU_SEC: float = 0.00002400
_GCP_MEM_PER_GIB_SEC: float = 0.00000250

# Reference workload
_REF_CPU: float = 2.0
_REF_MEMORY_GI: float = 4.0
_SECONDS_PER_HOUR: int = 3600


class GCPProvider(ProviderManager):
    """
    GCP Cloud Run compute provider.

    Currently returns published on-demand pricing (no live API query).
    Deploy and health_check delegate to Cloud Run Admin API patterns
    already established in Skia's RestorationOrchestrator.
    """

    def __init__(
        self,
        project_id: str = "",
        region: str = "australia-southeast1",
        service_name: str = "",
    ) -> None:
        self._project_id = project_id
        self._region = region
        self._service_name = service_name
        self._log = logger.bind(provider="gcp")

    @property
    def provider_id(self) -> str:
        return "gcp"

    async def get_quote(self) -> ComputeQuote:
        """
        Return the published GCP Cloud Run on-demand price.

        Uses static pricing — no API call required. Regional price
        multipliers are not applied (stub). australia-southeast1 is
        ~1.15x us-central1 but we use base rates for comparison.
        """
        cpu_hourly = _GCP_CPU_PER_VCPU_SEC * _REF_CPU * _SECONDS_PER_HOUR
        mem_hourly = _GCP_MEM_PER_GIB_SEC * _REF_MEMORY_GI * _SECONDS_PER_HOUR
        total_hourly = cpu_hourly + mem_hourly

        self._log.debug(
            "gcp_quote_computed",
            price_usd_hour=round(total_hourly, 6),
            region=self._region,
        )

        return ComputeQuote(
            provider_id="gcp",
            price_usd_per_hour=total_hourly,
            status=ProviderStatus.AVAILABLE if self._project_id else ProviderStatus.DEGRADED,
            region=self._region,
            currency_native="USD",
            price_native=total_hourly,
            metadata={
                "pricing_source": "published_on_demand",
                "cpu_per_vcpu_sec": str(_GCP_CPU_PER_VCPU_SEC),
                "mem_per_gib_sec": str(_GCP_MEM_PER_GIB_SEC),
            },
        )

    async def deploy(
        self,
        image: str,
        env_vars: dict[str, str],
        sdl_overrides: dict[str, str] | None = None,
    ) -> DeploymentResult:
        """
        Deploy to GCP Cloud Run.

        Stub: returns failure directing callers to use Skia's
        RestorationOrchestrator for Cloud Run deployments, which
        handles service-account JWT auth and revision patching.
        """
        return DeploymentResult(
            success=False,
            error=(
                "GCP deployment via ProviderManager is not yet implemented. "
                "Use Skia RestorationOrchestrator._restart_cloud_run() for "
                "Cloud Run deployments."
            ),
        )

    async def health_check(self, endpoint: str) -> bool:
        """
        Check GCP Cloud Run instance health.

        Same contract as Akash: expects /health to return
        {"status": "healthy", "state_restored": true}.
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
                return bool(
                    data.get("status") == "healthy"
                    and data.get("state_restored", False)
                )
        except Exception:
            return False
