"""
EcodiaOS — Axon Energy-Aware Scheduler

Modulates the organism's task execution based on real-time grid carbon
intensity and energy cost. High-compute tasks (REM dreaming, GRPO
fine-tuning, bounty solving, code mutation) are deferred to "sleep cycles"
when the grid is carbon-heavy, and released when clean energy is available.

Components:
  energy_client  — provider-agnostic API clients (Electricity Maps, WattTime)
  energy_cache   — Redis-backed cache for grid readings + deferred task queue
  interceptor    — pipeline gate that evaluates and defers tasks
  types          — shared data types (GridReading, DeferredTask, etc.)

Usage:
    from systems.axon.scheduler import (
        EnergyAwareInterceptor,
        EnergyCache,
        ElectricityMapsClient,
        WattTimeClient,
    )

    # Build the provider
    provider = ElectricityMapsClient(api_key="...", latitude=-33.87, longitude=151.21)

    # Wrap in cache
    cache = EnergyCache(provider=provider, redis=redis_client, ttl_s=600)

    # Create the interceptor
    interceptor = EnergyAwareInterceptor(config=config, cache=cache)
    interceptor.set_resubmit_callback(axon_service.execute)
    await interceptor.start()

    # In the execution path:
    decision = await interceptor.evaluate(request)
    if decision.should_defer:
        await interceptor.defer(request, decision)
    else:
        outcome = await axon_pipeline.execute(request)
"""

from systems.axon.scheduler.energy_cache import EnergyCache
from systems.axon.scheduler.energy_client import (
    ElectricityMapsClient,
    EnergyProvider,
    WattTimeClient,
)
from systems.axon.scheduler.interceptor import (
    EnergyAwareInterceptor,
    classify_intent,
    is_never_defer,
)
from systems.axon.scheduler.types import (
    ComputeIntensity,
    DeferralReason,
    DeferralStatus,
    DeferredTask,
    GridReading,
    InterceptDecision,
)

__all__ = [
    # Client
    "EnergyProvider",
    "ElectricityMapsClient",
    "WattTimeClient",
    # Cache
    "EnergyCache",
    # Interceptor
    "EnergyAwareInterceptor",
    "classify_intent",
    "is_never_defer",
    # Types
    "ComputeIntensity",
    "DeferralReason",
    "DeferralStatus",
    "DeferredTask",
    "GridReading",
    "InterceptDecision",
]
