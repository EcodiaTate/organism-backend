"""
EcodiaOS — SACM Substrate Provider Interface

Defines the abstract base class for compute substrate providers and the
SubstrateOffer dataclass that represents a point-in-time pricing offer
from a specific provider.

Relationship to infrastructure/providers/:
  The existing ProviderManager (infrastructure/providers/base.py) handles
  the lifecycle of full deployments (deploy, health_check) for organism
  migration.  SubstrateProvider is narrower — it deals only with the
  *market* interface: quoting prices for specific workload shapes and
  reporting availability.  The execution engine (out of scope here)
  bridges SubstrateProvider offers to ProviderManager deployments.

SubstrateOffer pricing model:
  Every offer carries per-unit-per-second rates for CPU, memory, GPU,
  storage, and a flat per-GiB egress rate.  The optimizer multiplies
  these by the WorkloadDescriptor's ResourceEnvelope and estimated
  duration to compute total cost.

  total_cost = (
      cpu_vcpu * duration_s * offer.price_cpu_per_vcpu_s
    + memory_gib * duration_s * offer.price_mem_per_gib_s
    + gpu_units  * duration_s * offer.price_gpu_per_unit_s
    + storage_gib * duration_s * offer.price_storage_per_gib_s
    + egress_gib  * offer.price_egress_per_gib
  )
"""

from __future__ import annotations

import enum
from datetime import datetime
from abc import ABC, abstractmethod

import structlog
from pydantic import Field, field_validator

from primitives.common import (
    Identified,
    Timestamped,
    utc_now,
)
from systems.sacm.workload import OffloadClass, ResourceEnvelope

logger = structlog.get_logger("systems.sacm.providers.base")


# ─── Provider Status ──────────────────────────────────────────────


class SubstrateProviderStatus(enum.StrEnum):
    """Operational state of a SACM substrate provider."""

    AVAILABLE = "available"
    """Accepting workloads, pricing current."""

    DEGRADED = "degraded"
    """Operational but with reduced capacity or stale pricing."""

    MAINTENANCE = "maintenance"
    """Temporarily offline for scheduled maintenance."""

    UNREACHABLE = "unreachable"
    """Cannot contact provider API — offers should not be trusted."""


# ─── Substrate Offer ──────────────────────────────────────────────


class SubstrateOffer(Identified, Timestamped):
    """
    A point-in-time pricing offer from a compute substrate provider.

    The optimizer collects SubstrateOffers from all registered providers,
    filters by workload compatibility, and scores them using the cost
    function defined in optimizer.py.

    All prices are in USD.  Providers that price in native tokens (e.g.
    uAKT for Akash) must convert to USD before constructing the offer.
    """

    # ── Provider identity ──
    provider_id: str
    """Stable provider identifier (e.g. 'akash', 'gcp', 'render')."""

    provider_name: str = ""
    """Human-readable provider name."""

    region: str = ""
    """Deployment region (e.g. 'us-central1', 'decentralised')."""

    # ── Capability ──
    supported_classes: list[OffloadClass] = Field(default_factory=lambda: [OffloadClass.GENERAL])
    """Which OffloadClasses this substrate can serve."""

    max_cpu_vcpu: float = 0.0
    """Maximum vCPUs available per workload on this substrate."""

    max_memory_gib: float = 0.0
    """Maximum memory in GiB available per workload."""

    max_gpu_units: float = 0.0
    """Maximum GPU units available (0 = no GPU)."""

    gpu_vram_gib: float = 0.0
    """VRAM per GPU unit in GiB."""

    max_storage_gib: float = 0.0
    """Maximum ephemeral storage in GiB."""

    # ── Pricing (all USD) ──
    price_cpu_per_vcpu_s: float = 0.0
    """USD per vCPU per second."""

    price_mem_per_gib_s: float = 0.0
    """USD per GiB of memory per second."""

    price_gpu_per_unit_s: float = 0.0
    """USD per GPU unit per second (0 if no GPU available)."""

    price_storage_per_gib_s: float = 0.0
    """USD per GiB of ephemeral storage per second."""

    price_egress_per_gib: float = 0.0
    """USD per GiB of network egress (flat, not time-based)."""

    # ── Validity ──
    status: SubstrateProviderStatus = SubstrateProviderStatus.AVAILABLE

    valid_until: datetime | None = None
    """When this offer expires. None = valid until next refresh."""

    # ── Trust & Quality ──
    trust_score: float = 1.0
    """Provider trust score in [0, 1]. Used by optimizer for risk-adjusted ranking.
    Decremented on verification failures, incremented on clean runs."""

    avg_latency_overhead_s: float = 5.0
    """Average non-execution overhead (dispatch, queue, result transfer) in seconds."""

    # ── Native pricing (informational) ──
    currency_native: str = "USD"
    """Native currency (e.g. 'uAKT', 'USD', 'RNDR')."""

    price_native_per_hour: float = 0.0
    """Total native-currency cost per hour for a reference workload (informational)."""

    exchange_rate_to_usd: float = 1.0
    """Native currency → USD rate at quote time."""

    # ── Metadata ──
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("trust_score")
    @classmethod
    def _clamp_trust(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @property
    def has_gpu(self) -> bool:
        return self.max_gpu_units > 0

    @property
    def is_valid(self) -> bool:
        """Whether this offer is still within its validity window."""
        if self.status in (SubstrateProviderStatus.UNREACHABLE, SubstrateProviderStatus.MAINTENANCE):
            return False
        return not (self.valid_until is not None and utc_now() > self.valid_until)

    def can_serve(self, resources: ResourceEnvelope) -> bool:
        """
        Check whether this substrate can accommodate the given resource envelope.

        Returns False if any resource dimension exceeds the substrate's capacity.
        """
        if resources.cpu_vcpu > self.max_cpu_vcpu and self.max_cpu_vcpu > 0:
            return False
        if resources.memory_gib > self.max_memory_gib and self.max_memory_gib > 0:
            return False
        if resources.gpu_units > self.max_gpu_units:
            return False
        if resources.gpu_vram_gib > self.gpu_vram_gib and resources.gpu_units > 0:
            return False
        return not (resources.storage_gib > self.max_storage_gib and self.max_storage_gib > 0)


# ─── Substrate Provider ABC ───────────────────────────────────────


class SubstrateProvider(ABC):
    """
    Abstract base class for SACM substrate providers.

    Each implementation wraps a specific compute marketplace (Akash,
    Render, GCP spot, Lambda Labs, etc.) and translates its native
    pricing model into normalised SubstrateOffers.

    Responsibilities:
      1. Fetch current pricing and construct SubstrateOffers
      2. Report provider status (available, degraded, unreachable)
      3. Provide metadata about capabilities (GPU, regions, etc.)

    Execution and verification are handled by separate subsystems —
    the provider only participates in the market/quoting phase.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique stable identifier for this provider (e.g. 'akash')."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'Akash Network')."""
        ...

    @abstractmethod
    async def fetch_offers(self) -> list[SubstrateOffer]:
        """
        Query the provider's API and return current SubstrateOffers.

        Must not raise — on failure, return an empty list and log the error.
        Implementations should cache aggressively; the oracle controls
        refresh cadence.

        Returns:
            List of SubstrateOffers with USD-normalised pricing.
            May return multiple offers (different regions, tiers, etc.).
        """
        ...

    @abstractmethod
    async def health(self) -> SubstrateProviderStatus:
        """
        Quick liveness check against the provider's API.

        Returns the current operational status without fetching full pricing.
        Used by the oracle to decide whether to include this provider
        in the next pricing surface refresh.
        """
        ...
