"""
EcodiaOS — Compute Provider Interface

Defines the ProviderManager ABC and shared pricing primitives.

Every provider must:
  1. Report its current per-hour compute cost for a standardised workload
  2. Report operational status (reachable, healthy, maintenance, etc.)
  3. Deploy a containerised workload given an SDL/manifest + env vars

Pricing is normalised to USD/hour for a single EcodiaOS-class workload
(2 vCPU, 4 GiB RAM, 20 GiB ephemeral storage) so the arbitrage executor
can compare apples-to-apples.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped

if TYPE_CHECKING:
    from datetime import datetime


class ProviderStatus(enum.StrEnum):
    """Operational state of a compute provider."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    UNREACHABLE = "unreachable"


class ComputeQuote(Identified, Timestamped):
    """
    Point-in-time price quote from a compute provider.

    All prices normalised to a reference workload:
      2 vCPU / 4 GiB RAM / 20 GiB storage (the EOS standard footprint).
    """

    provider_id: str
    price_usd_per_hour: float
    status: ProviderStatus = ProviderStatus.AVAILABLE
    region: str = ""
    currency_native: str = "USD"
    # For Akash: price in uAKT; for GCP: on-demand price
    price_native: float = 0.0
    quote_valid_until: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class DeploymentResult(EOSBaseModel):
    """Result of submitting a deployment to a provider."""

    success: bool
    deployment_id: str = ""
    endpoint: str = ""
    error: str = ""
    cost_estimate_usd: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)


class ProviderManager(ABC):
    """
    Abstract base class for compute providers.

    Concrete implementations query real APIs (Akash Network, GCP)
    or return stubbed pricing for providers not yet integrated.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique stable identifier (e.g., 'akash', 'gcp')."""
        ...

    @abstractmethod
    async def get_quote(self) -> ComputeQuote:
        """
        Fetch a real-time price quote for the EOS reference workload.

        Must not raise — return a quote with status=UNREACHABLE on failure.
        """
        ...

    @abstractmethod
    async def deploy(
        self,
        image: str,
        env_vars: dict[str, str],
        sdl_overrides: dict[str, str] | None = None,
    ) -> DeploymentResult:
        """
        Deploy a containerised workload to this provider.

        Args:
            image: Docker image URI (e.g., 'ghcr.io/ecodiaos/core:latest')
            env_vars: Environment variables to inject (includes SKIA_RESTORE_CID)
            sdl_overrides: Provider-specific template variable overrides
        """
        ...

    @abstractmethod
    async def health_check(self, endpoint: str) -> bool:
        """
        Check whether a deployed instance is healthy by polling its /health endpoint.

        Returns True only if the instance reports successful state restoration.
        """
        ...
