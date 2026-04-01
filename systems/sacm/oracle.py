"""
EcodiaOS - SACM Compute Market Oracle

The ComputeMarketOracle maintains a live "pricing surface" - the
complete set of valid SubstrateOffers from all registered providers.
It is the single source of truth the optimizer queries when placing
workloads.

Responsibilities:
  1. Manage a registry of SubstrateProviders
  2. Periodically refresh offers from all providers (with jitter to
     avoid thundering-herd effects)
  3. Expire stale offers whose validity windows have elapsed
  4. Provide filtered views of the pricing surface (by OffloadClass,
     region, provider, resource capacity)
  5. Track per-provider health and disable unhealthy providers
  6. Emit pricing telemetry for Oikos cost modelling

The oracle does NOT make placement decisions - that is the optimizer's
job (optimizer.py).  The oracle is a read-side cache of market state.

Concurrency model:
  - refresh_all() is safe to call concurrently; each provider is
    fetched independently via asyncio.gather
  - The offer store is replaced atomically (snapshot semantics)
  - Readers see a consistent snapshot even during refresh
"""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, utc_now
from systems.sacm.providers.base import (
    SubstrateOffer,
    SubstrateProvider,
    SubstrateProviderStatus,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from systems.sacm.workload import OffloadClass, ResourceEnvelope

logger = structlog.get_logger("systems.sacm.oracle")


# ─── Provider Health Tracking ─────────────────────────────────────


class ProviderHealthRecord(EOSBaseModel):
    """Tracks the operational health of a registered provider."""

    provider_id: str
    status: SubstrateProviderStatus = SubstrateProviderStatus.AVAILABLE
    last_successful_fetch: float | None = None   # epoch seconds
    last_failed_fetch: float | None = None
    consecutive_failures: int = 0
    total_fetches: int = 0
    total_offers_returned: int = 0

    @property
    def failure_rate(self) -> float:
        if self.total_fetches == 0:
            return 0.0
        return (self.total_fetches - self.total_offers_returned) / self.total_fetches


# ─── Pricing Surface Snapshot ─────────────────────────────────────


class PricingSurfaceSnapshot(Identified, Timestamped):
    """
    Immutable snapshot of the complete pricing surface at a point in time.

    The optimizer receives a snapshot and scores against it - concurrent
    refreshes do not mutate in-flight scoring.
    """

    offers: list[SubstrateOffer] = Field(default_factory=list)
    provider_health: dict[str, ProviderHealthRecord] = Field(default_factory=dict)

    @property
    def offer_count(self) -> int:
        return len(self.offers)

    @property
    def valid_offers(self) -> list[SubstrateOffer]:
        """Offers that are currently valid (not expired, not unreachable)."""
        return [o for o in self.offers if o.is_valid]

    @property
    def provider_ids(self) -> set[str]:
        return {o.provider_id for o in self.offers}

    def filter_by_class(self, offload_class: OffloadClass) -> list[SubstrateOffer]:
        """Return valid offers that support the given OffloadClass."""
        return [
            o for o in self.valid_offers
            if offload_class in o.supported_classes
        ]

    def filter_by_resources(self, resources: ResourceEnvelope) -> list[SubstrateOffer]:
        """Return valid offers that can accommodate the resource envelope."""
        return [o for o in self.valid_offers if o.can_serve(resources)]

    def filter_by_region(self, regions: Sequence[str]) -> list[SubstrateOffer]:
        """Return valid offers in any of the specified regions. Empty = all."""
        if not regions:
            return list(self.valid_offers)
        region_set = set(regions)
        return [o for o in self.valid_offers if o.region in region_set]

    def filter_eligible(
        self,
        offload_class: OffloadClass,
        resources: ResourceEnvelope,
        regions: Sequence[str] = (),
        blocked_providers: Sequence[str] = (),
    ) -> list[SubstrateOffer]:
        """
        Combined filter: class + resources + regions + blocked providers.

        This is the primary entry point the optimizer uses to get
        candidate offers for scoring.
        """
        blocked = set(blocked_providers)
        return [
            o for o in self.valid_offers
            if (
                offload_class in o.supported_classes
                and o.can_serve(resources)
                and (not regions or o.region in set(regions))
                and o.provider_id not in blocked
            )
        ]


# ─── Compute Market Oracle ────────────────────────────────────────


class ComputeMarketOracle:
    """
    Manages the SACM pricing surface by polling registered substrate providers.

    Usage:
        oracle = ComputeMarketOracle()
        oracle.register(SACMAkashProvider())
        oracle.register(SACMRenderProvider())

        await oracle.refresh_all()

        snapshot = oracle.snapshot
        candidates = snapshot.filter_eligible(
            offload_class=OffloadClass.GPU_HEAVY,
            resources=workload.resources,
        )
    """

    # How often the recovery loop re-checks UNREACHABLE providers (seconds)
    _RECOVERY_INTERVAL_S: float = 300.0  # 5 minutes

    def __init__(
        self,
        stale_threshold_minutes: float = 10.0,
        max_consecutive_failures: int = 5,
    ) -> None:
        """
        Args:
            stale_threshold_minutes: Offers older than this are pruned even
                if their valid_until hasn't expired.
            max_consecutive_failures: After this many consecutive fetch
                failures, a provider is marked UNREACHABLE.
        """
        self._providers: dict[str, SubstrateProvider] = {}
        self._health: dict[str, ProviderHealthRecord] = {}
        self._offers: list[SubstrateOffer] = []
        self._stale_threshold = timedelta(minutes=stale_threshold_minutes)
        self._max_failures = max_consecutive_failures
        self._last_refresh_epoch: float = 0.0
        self._recovery_task: asyncio.Task[None] | None = None
        self._log = logger.bind(component="sacm.oracle")

    # ── Provider Registry ─────────────────────────────────────────

    def register(self, provider: SubstrateProvider) -> None:
        """Add a substrate provider to the oracle."""
        pid = provider.provider_id
        self._providers[pid] = provider
        if pid not in self._health:
            self._health[pid] = ProviderHealthRecord(provider_id=pid)
        self._log.info("provider_registered", provider_id=pid)

    def unregister(self, provider_id: str) -> None:
        """Remove a provider and its offers from the oracle."""
        self._providers.pop(provider_id, None)
        self._health.pop(provider_id, None)
        self._offers = [o for o in self._offers if o.provider_id != provider_id]
        self._log.info("provider_unregistered", provider_id=provider_id)

    @property
    def registered_providers(self) -> list[str]:
        return list(self._providers.keys())

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the background UNREACHABLE-recovery loop.

        Call once after the oracle is fully configured and the event loop
        is running (typically in SACMService.initialize()).
        """
        if self._recovery_task is not None:
            return
        self._recovery_task = asyncio.create_task(
            self._recovery_loop(),
            name="sacm_oracle_recovery",
        )
        self._log.info("oracle_recovery_loop_started")

    async def stop(self) -> None:
        """Stop the background recovery loop."""
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            import contextlib
            with contextlib.suppress(asyncio.CancelledError):
                await self._recovery_task
            self._recovery_task = None
        self._log.info("oracle_recovery_loop_stopped")

    # ── UNREACHABLE Recovery Loop ──────────────────────────────────

    async def _recovery_loop(self) -> None:
        """
        Background task: every 5 minutes, re-health-check all providers
        currently marked UNREACHABLE.

        On a successful health() call the provider is restored to AVAILABLE
        and its consecutive_failures counter is reset so the next
        refresh_all() will include it again.
        """
        while True:
            try:
                await asyncio.sleep(self._RECOVERY_INTERVAL_S)
                await self._recover_unreachable_providers()
            except asyncio.CancelledError:
                break
            except Exception:
                self._log.exception("oracle_recovery_loop_error")

    async def _recover_unreachable_providers(self) -> None:
        """Re-health-check all UNREACHABLE providers and restore healthy ones."""
        unreachable = [
            pid
            for pid, health in self._health.items()
            if health.status == SubstrateProviderStatus.UNREACHABLE
            and pid in self._providers
        ]
        if not unreachable:
            return

        self._log.info(
            "oracle_recovery_check",
            unreachable_count=len(unreachable),
            provider_ids=unreachable,
        )

        for provider_id in unreachable:
            provider = self._providers.get(provider_id)
            if provider is None:
                continue
            try:
                status = await provider.health()
                if status == SubstrateProviderStatus.AVAILABLE:
                    health = self._health[provider_id]
                    health.status = SubstrateProviderStatus.AVAILABLE
                    health.consecutive_failures = 0
                    health.last_successful_fetch = utc_now().timestamp()
                    self._log.info(
                        "provider_restored",
                        provider_id=provider_id,
                    )
                    # Immediately fetch fresh offers so the restored provider
                    # is available to the optimizer without waiting for the
                    # next scheduled refresh_all().
                    asyncio.create_task(
                        self.refresh_provider(provider_id),
                        name=f"sacm_oracle_restore_{provider_id}",
                    )
            except Exception as exc:
                self._log.debug(
                    "provider_recovery_check_failed",
                    provider_id=provider_id,
                    error=str(exc),
                )

    # ── Refresh ───────────────────────────────────────────────────

    async def refresh_all(self) -> PricingSurfaceSnapshot:
        """
        Fetch fresh offers from all registered providers concurrently.

        Replaces the internal offer store atomically with the union of
        all successful provider responses.  Failed providers retain their
        last-known offers until the stale threshold ejects them.

        Returns the new pricing surface snapshot.
        """
        if not self._providers:
            self._log.warning("no_providers_registered")
            return self.snapshot

        tasks = {
            pid: asyncio.create_task(self._fetch_provider(pid, provider))
            for pid, provider in self._providers.items()
        }

        results: dict[str, list[SubstrateOffer]] = {}
        for pid, task in tasks.items():
            try:
                results[pid] = await task
            except Exception as exc:
                self._log.error("provider_fetch_exception", provider_id=pid, error=str(exc))
                results[pid] = []

        # Build new offer list: fresh offers from successful fetches,
        # retained non-stale offers from failed fetches
        now = utc_now()
        new_offers: list[SubstrateOffer] = []

        for pid in self._providers:
            fetched = results.get(pid, [])
            if fetched:
                new_offers.extend(fetched)
            else:
                # Retain existing offers for this provider if not stale
                retained = [
                    o for o in self._offers
                    if o.provider_id == pid and self._is_fresh(o, now)
                ]
                new_offers.extend(retained)

        # Atomic replacement
        self._offers = new_offers
        self._last_refresh_epoch = now.timestamp()

        self._log.info(
            "pricing_surface_refreshed",
            total_offers=len(new_offers),
            providers_queried=len(self._providers),
            providers_with_offers=len({o.provider_id for o in new_offers}),
        )

        return self.snapshot

    async def refresh_provider(self, provider_id: str) -> list[SubstrateOffer]:
        """Refresh offers from a single provider."""
        provider = self._providers.get(provider_id)
        if provider is None:
            self._log.warning("refresh_unknown_provider", provider_id=provider_id)
            return []

        fetched = await self._fetch_provider(provider_id, provider)

        # Replace only this provider's offers
        self._offers = [
            o for o in self._offers if o.provider_id != provider_id
        ] + fetched

        return fetched

    # ── Snapshot Access ───────────────────────────────────────────

    @property
    def snapshot(self) -> PricingSurfaceSnapshot:
        """
        Return an immutable snapshot of the current pricing surface.

        The snapshot is a value object - it will not be mutated by
        subsequent refresh calls.
        """
        return PricingSurfaceSnapshot(
            offers=list(self._offers),
            provider_health=dict(self._health),
        )

    @property
    def offer_count(self) -> int:
        return len(self._offers)

    @property
    def valid_offer_count(self) -> int:
        return sum(1 for o in self._offers if o.is_valid)

    # ── Health Queries ────────────────────────────────────────────

    def provider_health(self, provider_id: str) -> ProviderHealthRecord | None:
        return self._health.get(provider_id)

    @property
    def all_provider_health(self) -> dict[str, ProviderHealthRecord]:
        return dict(self._health)

    # ── Cheapest-offer convenience ────────────────────────────────

    def cheapest_cpu_offer(self) -> SubstrateOffer | None:
        """Return the valid offer with the lowest CPU price/second."""
        candidates = [
            o for o in self._offers
            if o.is_valid and o.price_cpu_per_vcpu_s > 0
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda o: o.price_cpu_per_vcpu_s)

    def cheapest_gpu_offer(self) -> SubstrateOffer | None:
        """Return the valid offer with the lowest GPU price/second."""
        candidates = [
            o for o in self._offers
            if o.is_valid and o.has_gpu and o.price_gpu_per_unit_s > 0
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda o: o.price_gpu_per_unit_s)

    # ── Internals ─────────────────────────────────────────────────

    async def _fetch_provider(
        self,
        provider_id: str,
        provider: SubstrateProvider,
    ) -> list[SubstrateOffer]:
        """Fetch offers from a single provider and update health tracking."""
        health = self._health.setdefault(
            provider_id, ProviderHealthRecord(provider_id=provider_id),
        )
        health.total_fetches += 1

        try:
            offers = await provider.fetch_offers()

            if offers:
                now_epoch = utc_now().timestamp()
                health.last_successful_fetch = now_epoch
                health.consecutive_failures = 0
                health.total_offers_returned += len(offers)
                health.status = SubstrateProviderStatus.AVAILABLE

                self._log.debug(
                    "provider_offers_received",
                    provider_id=provider_id,
                    count=len(offers),
                )
            else:
                self._record_failure(health, "empty response")

            return offers

        except Exception as exc:
            self._record_failure(health, str(exc))
            return []

    def _record_failure(self, health: ProviderHealthRecord, reason: str) -> None:
        """Update health record after a failed fetch."""
        health.consecutive_failures += 1
        health.last_failed_fetch = utc_now().timestamp()

        if health.consecutive_failures >= self._max_failures:
            health.status = SubstrateProviderStatus.UNREACHABLE
            self._log.warning(
                "provider_marked_unreachable",
                provider_id=health.provider_id,
                consecutive_failures=health.consecutive_failures,
                reason=reason,
            )
        else:
            health.status = SubstrateProviderStatus.DEGRADED
            self._log.warning(
                "provider_fetch_failed",
                provider_id=health.provider_id,
                consecutive_failures=health.consecutive_failures,
                reason=reason,
            )

    def _is_fresh(self, offer: SubstrateOffer, now: datetime) -> bool:
        """Check if a retained offer is still fresh enough to keep."""

        if offer.valid_until is not None and now > offer.valid_until:
            return False
        age = now - offer.created_at
        return age <= self._stale_threshold
