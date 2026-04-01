"""
EcodiaOS - SACM Pre-Warming Engine

Maintains a pool of "warm" compute instances ready for immediate dispatch.
The engine predicts upcoming workload demand using an exponential moving
average over recent submission history, and pre-positions capacity when:

  1. Predicted demand exceeds current warm pool size.
  2. The market oracle forecasts a price increase (buy cheap now).

Budget enforcement is strict - the engine will never exceed the hourly
spend cap defined in SACMPreWarmConfig, and will proactively drain
instances that are idle beyond their TTL.

Concurrency model:
  - pre_warm_loop() is a single long-lived asyncio task.
  - WarmInstance state is guarded by an asyncio.Lock.
  - The oracle and optimizer are read-only from this module's perspective.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import math
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped
from systems.sacm.config import SACMPreWarmConfig
from systems.sacm.optimizer import estimate_hourly_cost
from systems.sacm.providers.base import SubstrateOffer  # noqa: TC001 - Pydantic needs at runtime
from systems.sacm.workload import (
    OffloadClass,
    ResourceEnvelope,
)

if TYPE_CHECKING:
    from systems.oikos.service import OikosService
    from systems.sacm.oracle import ComputeMarketOracle, PricingSurfaceSnapshot

logger = structlog.get_logger("systems.sacm.pre_warming")


# ─── Warm Instance ───────────────────────────────────────────────────


class WarmInstanceStatus(enum.StrEnum):
    PROVISIONING = "provisioning"
    READY = "ready"
    CLAIMED = "claimed"
    RELEASING = "releasing"
    RELEASED = "released"


class WarmInstance(Identified, Timestamped):
    """A pre-provisioned compute instance sitting in the warm pool."""

    offer_id: str
    provider_id: str
    offload_class: OffloadClass
    resources: ResourceEnvelope
    hourly_cost_usd: float = 0.0
    status: WarmInstanceStatus = WarmInstanceStatus.PROVISIONING
    provisioned_at: float = Field(default_factory=time.monotonic)
    last_heartbeat: float = Field(default_factory=time.monotonic)
    claimed_by_workload_id: str = ""
    reason: str = ""
    """Why this instance was warmed (e.g. 'demand_prediction', 'price_opportunity')."""


# ─── Demand Prediction ───────────────────────────────────────────────


class DemandForecast(EOSBaseModel):
    """Per-class demand prediction for the upcoming horizon window."""

    offload_class: OffloadClass
    predicted_count: float = 0.0
    """Expected number of workloads in the next prediction_horizon_s."""

    confidence: float = 0.0
    """Rough confidence in [0, 1]. Low when history is short."""

    is_burst: bool = False
    """True when rolling stddev exceeds burst_stddev_factor × rolling mean."""


class WorkloadPredictor:
    """
    EMA-based workload demand predictor.

    Tracks workload submission timestamps per OffloadClass and produces a
    DemandForecast for each tracked class.  The predictor is deliberately
    simple - a single exponential moving average over submission rates
    with a burst detector layered on top.
    """

    def __init__(self, config: SACMPreWarmConfig) -> None:
        self._cfg = config.prediction
        self._timing = config.timing
        # Per-class sliding window of submission timestamps (monotonic seconds)
        self._history: dict[OffloadClass, deque[float]] = defaultdict(
            lambda: deque(maxlen=500),
        )
        # Per-class EMA of submission rate (workloads / second)
        self._ema_rate: dict[OffloadClass, float] = {}
        self._log = logger.bind(component="sacm.predictor")

    # ── Recording ────────────────────────────────────────────────────

    def record_submission(self, offload_class: OffloadClass) -> None:
        """Called when a new workload is submitted to SACM."""
        self._history[offload_class].append(time.monotonic())

    # ── Prediction ───────────────────────────────────────────────────

    def predict(self) -> list[DemandForecast]:
        """
        Produce demand forecasts for all tracked offload classes.

        Algorithm per class:
          1. Compute instantaneous rate = submissions in last horizon / horizon.
          2. Update EMA: ema = alpha × instant + (1 - alpha) × ema_prev.
          3. Detect burst: rolling stddev of inter-arrival times > burst_factor × mean.
          4. Predicted count = ema_rate × horizon × burst_multiplier_if_burst.

        Returns one DemandForecast per class in prediction.default_classes,
        even if that class has no history (predicted_count = 0).
        """
        now = time.monotonic()
        horizon = self._timing.prediction_horizon_s
        alpha = self._cfg.ema_alpha
        min_samples = self._cfg.min_history_samples
        forecasts: list[DemandForecast] = []

        for cls in self._cfg.default_classes:
            history = self._history.get(cls)
            if history is None or len(history) < min_samples:
                forecasts.append(
                    DemandForecast(offload_class=cls, predicted_count=0.0, confidence=0.0),
                )
                continue

            # Count submissions within the prediction horizon window
            cutoff = now - horizon
            recent = [t for t in history if t >= cutoff]
            instant_rate = len(recent) / horizon if horizon > 0 else 0.0

            # Update EMA
            prev = self._ema_rate.get(cls, instant_rate)
            ema = alpha * instant_rate + (1.0 - alpha) * prev
            self._ema_rate[cls] = ema

            # Burst detection via inter-arrival time variance
            is_burst = False
            if len(recent) >= 3:
                sorted_ts = sorted(recent)
                intervals = [
                    sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)
                ]
                mean_interval = sum(intervals) / len(intervals) if intervals else 0.0
                if mean_interval > 0:
                    variance = sum((iv - mean_interval) ** 2 for iv in intervals) / len(
                        intervals
                    )
                    stddev = math.sqrt(variance)
                    is_burst = stddev > self._cfg.burst_stddev_factor * mean_interval

            multiplier = self._cfg.burst_multiplier if is_burst else 1.0
            predicted = ema * horizon * multiplier

            confidence = min(1.0, len(recent) / (min_samples * 4))

            forecasts.append(
                DemandForecast(
                    offload_class=cls,
                    predicted_count=predicted,
                    confidence=confidence,
                    is_burst=is_burst,
                ),
            )

        return forecasts


# ─── Price Opportunity ───────────────────────────────────────────────


class PriceOpportunity(EOSBaseModel):
    """A detected opportunity to lock in capacity before a price rise."""

    offload_class: OffloadClass
    current_cheapest_usd_hr: float
    predicted_price_usd_hr: float
    increase_pct: float
    recommended_instances: int
    offer: SubstrateOffer | None = None


# ─── Pre-Warming Engine ─────────────────────────────────────────────


class PreWarmingEngine:
    """
    Manages the warm instance pool.

    Lifecycle:
      1. Instantiate with an oracle reference and config.
      2. Call start() to launch the background pre_warm_loop.
      3. Call claim() when the executor needs a warm instance.
      4. Call stop() during graceful shutdown.

    The engine never provisions instances itself - it selects the best
    offer via the optimizer and records the intent as a WarmInstance.
    An external provisioner (or the RemoteExecutionManager) fulfils the
    actual deployment and transitions the instance to READY via
    mark_ready().
    """

    def __init__(
        self,
        oracle: ComputeMarketOracle,
        config: SACMPreWarmConfig | None = None,
    ) -> None:
        self._oracle = oracle
        self._cfg = config or SACMPreWarmConfig()
        self._predictor = WorkloadPredictor(self._cfg)
        self._pool: dict[str, WarmInstance] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._last_price_check: float = 0.0
        # Snapshot cache for price-opportunity comparison
        self._prev_snapshot: PricingSurfaceSnapshot | None = None
        # oikos is OikosService at runtime; typed as Any to avoid circular import
        self._oikos: Any | None = None
        self._synapse: Any | None = None
        # provider_managers: provider_id → ProviderManager (infrastructure layer)
        # Wired via register_provider_manager() so pre-warming can call .deploy()
        self._provider_managers: dict[str, Any] = {}
        self._log = logger.bind(component="sacm.pre_warming")

    # ── Oikos Integration ─────────────────────────────────────────────

    def wire_oikos(self, oikos: OikosService) -> None:
        """
        Attach an OikosService so the pre-warm loop reads its authoritative
        compute budget on every tick instead of the local config default.

        Call after OikosService is initialised (step 15e in main.py).
        """
        self._oikos = oikos
        self._log.info("oikos_wired_to_pre_warm_engine")

    def set_synapse(self, synapse: Any) -> None:
        """Attach Synapse so pre-warm events can be emitted."""
        self._synapse = synapse
        self._log.info("synapse_wired_to_pre_warm_engine")

    def register_provider_manager(self, provider_id: str, manager: Any) -> None:
        """
        Register an infrastructure ProviderManager so pre-warming can call
        manager.deploy() to provision real instances.

        Call once per provider after ProviderManager is initialised.
        provider_id must match the SubstrateOffer.provider_id values used by
        the oracle so the correct manager is selected at provisioning time.
        """
        self._provider_managers[provider_id] = manager
        self._log.info("provider_manager_registered", provider_id=provider_id)

    # ── Public API ───────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background pre-warm loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._pre_warm_loop())
        self._log.info("pre_warm_engine_started")

    async def stop(self) -> None:
        """Gracefully shut down the pre-warm loop and release all instances."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        async with self._lock:
            for inst in self._pool.values():
                if inst.status in (WarmInstanceStatus.READY, WarmInstanceStatus.PROVISIONING):
                    inst.status = WarmInstanceStatus.RELEASED
            self._pool.clear()
        self._log.info("pre_warm_engine_stopped")

    def record_submission(self, offload_class: OffloadClass) -> None:
        """Feed the predictor when a new workload arrives."""
        self._predictor.record_submission(offload_class)

    async def claim(
        self,
        offload_class: OffloadClass,
        resources: ResourceEnvelope,
        workload_id: str,
    ) -> WarmInstance | None:
        """
        Try to claim a warm instance matching the requested class and resources.

        Returns the claimed WarmInstance or None if no match is available.
        """
        async with self._lock:
            for inst in self._pool.values():
                if (
                    inst.status == WarmInstanceStatus.READY
                    and inst.offload_class == offload_class
                    and inst.resources.cpu_vcpu >= resources.cpu_vcpu
                    and inst.resources.memory_gib >= resources.memory_gib
                    and inst.resources.gpu_units >= resources.gpu_units
                ):
                    inst.status = WarmInstanceStatus.CLAIMED
                    inst.claimed_by_workload_id = workload_id
                    self._log.info(
                        "warm_instance_claimed",
                        instance_id=inst.id,
                        workload_id=workload_id,
                    )
                    return inst
        return None

    async def mark_ready(self, instance_id: str) -> bool:
        """Transition a provisioning instance to ready."""
        async with self._lock:
            inst = self._pool.get(instance_id)
            if inst is None or inst.status != WarmInstanceStatus.PROVISIONING:
                return False
            inst.status = WarmInstanceStatus.READY
            inst.last_heartbeat = time.monotonic()
            self._log.info("warm_instance_ready", instance_id=instance_id)
            return True

    async def release(self, instance_id: str) -> bool:
        """Explicitly release an instance back to the pool or remove it."""
        async with self._lock:
            inst = self._pool.pop(instance_id, None)
            if inst is None:
                return False
            inst.status = WarmInstanceStatus.RELEASED
            self._log.info("warm_instance_released", instance_id=instance_id)
            return True

    @property
    def pool_snapshot(self) -> list[WarmInstance]:
        """Return a copy of the current warm pool for monitoring."""
        return list(self._pool.values())

    @property
    def active_count(self) -> int:
        """Number of instances that are provisioning or ready."""
        return sum(
            1
            for i in self._pool.values()
            if i.status in (WarmInstanceStatus.PROVISIONING, WarmInstanceStatus.READY)
        )

    @property
    def current_hourly_spend(self) -> float:
        """Estimated hourly cost of all active warm instances."""
        return sum(
            i.hourly_cost_usd
            for i in self._pool.values()
            if i.status in (WarmInstanceStatus.PROVISIONING, WarmInstanceStatus.READY)
        )

    # ── Background Loop ──────────────────────────────────────────────

    async def _pre_warm_loop(self) -> None:
        """
        Main control loop.  Runs until stop() is called.

        Each iteration:
          1. Expire stale warm instances past their TTL.
          2. Query the predictor for demand forecasts.
          3. If demand > current pool for a class, provision up to budget.
          4. Periodically check for price opportunities.
        """
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                self._log.exception("pre_warm_loop_error")
            await asyncio.sleep(self._cfg.timing.loop_interval_s)

    async def _tick(self) -> None:
        """Single iteration of the pre-warm loop."""
        now = time.monotonic()

        # Phase 0: pull the authoritative budget from Oikos so economic pressure
        # is reflected immediately without requiring a config reload.
        if self._oikos is not None:
            oikos_budget = self._oikos.get_compute_budget()
            self._cfg.budget.max_pre_warm_budget_usd_per_hour = oikos_budget

        # Phase 1: expire stale instances
        await self._expire_stale(now)

        # Phase 2: demand-based provisioning
        forecasts = self._predictor.predict()
        snapshot = self._oracle.snapshot

        for forecast in forecasts:
            if forecast.predicted_count <= 0 or forecast.confidence <= 0:
                continue
            await self._provision_for_demand(forecast, snapshot)

        # Phase 3: price-opportunity check (at a slower cadence)
        if now - self._last_price_check >= self._cfg.timing.price_check_interval_s:
            self._last_price_check = now
            await self._check_price_opportunities(snapshot)
            self._prev_snapshot = snapshot

    async def _expire_stale(self, now: float) -> None:
        """Release instances whose TTL has elapsed without being claimed."""
        ttl = self._cfg.timing.warm_instance_ttl_s
        async with self._lock:
            to_remove: list[str] = []
            for inst_id, inst in self._pool.items():
                if inst.status != WarmInstanceStatus.READY:
                    continue
                age = now - inst.provisioned_at
                if age > ttl:
                    inst.status = WarmInstanceStatus.RELEASED
                    to_remove.append(inst_id)
                    self._log.info(
                        "warm_instance_expired",
                        instance_id=inst_id,
                        age_s=round(age, 1),
                        ttl_s=ttl,
                    )
            for inst_id in to_remove:
                self._pool.pop(inst_id, None)

    async def _provision_for_demand(
        self,
        forecast: DemandForecast,
        snapshot: PricingSurfaceSnapshot,
    ) -> None:
        """
        Provision warm instances to meet predicted demand for a class.

        Respects all budget and instance-count caps from config.
        """
        cls = forecast.offload_class
        needed = max(0, math.ceil(forecast.predicted_count) - self._count_active(cls))
        if needed <= 0:
            return

        # Enforce per-class cap
        cap = self._cfg.budget.max_warm_instances_per_class
        current = self._count_active(cls)
        headroom = max(0, cap - current)
        needed = min(needed, headroom)

        # Enforce global cap
        global_headroom = max(0, self._cfg.budget.max_warm_instances - self.active_count)
        needed = min(needed, global_headroom)

        if needed <= 0:
            return

        # Find cheapest eligible offer for a representative workload shape
        offers = self._cheapest_offers_for_class(cls, snapshot)
        if not offers:
            return

        for offer in offers[:needed]:
            hourly = estimate_hourly_cost(
                offer,
                cpu_vcpu=offer.max_cpu_vcpu,
                memory_gib=offer.max_memory_gib,
                gpu_units=offer.max_gpu_units,
            )
            if not self._budget_allows(hourly):
                self._log.debug(
                    "pre_warm_budget_exhausted",
                    offload_class=cls,
                    hourly_cost=round(hourly, 4),
                )
                break
            if hourly > self._cfg.budget.max_single_instance_cost_usd_per_hour:
                continue

            await self._create_warm_instance(offer, cls, hourly, reason="demand_prediction")

    async def _check_price_opportunities(
        self,
        current_snapshot: PricingSurfaceSnapshot,
    ) -> None:
        """
        Compare current prices against the previous snapshot.

        If the cheapest offer in a class is significantly cheaper now than
        what we'd expect after a predicted increase, pre-position capacity.

        The heuristic: if the previous snapshot's cheapest price was P_prev
        and the current is P_now, and P_now < P_prev (prices dropped), we
        treat this as an opportunity because prices are likely to revert.
        Conversely, if P_now > P_prev by more than the threshold, the window
        has already closed.

        In a production system this would be replaced by a proper
        time-series forecast. For v1 we use a simple comparison.
        """
        if self._prev_snapshot is None:
            return

        opp_cfg = self._cfg.price_opportunity
        opp_budget = (
            self._cfg.budget.max_pre_warm_budget_usd_per_hour
            * opp_cfg.opportunity_budget_fraction
        )

        for cls in self._cfg.prediction.default_classes:
            prev_offers = self._cheapest_offers_for_class(cls, self._prev_snapshot)
            curr_offers = self._cheapest_offers_for_class(cls, current_snapshot)

            if not prev_offers or not curr_offers:
                continue

            prev_cheapest_hr = estimate_hourly_cost(
                prev_offers[0],
                cpu_vcpu=prev_offers[0].max_cpu_vcpu,
                memory_gib=prev_offers[0].max_memory_gib,
                gpu_units=prev_offers[0].max_gpu_units,
            )
            curr_cheapest_hr = estimate_hourly_cost(
                curr_offers[0],
                cpu_vcpu=curr_offers[0].max_cpu_vcpu,
                memory_gib=curr_offers[0].max_memory_gib,
                gpu_units=curr_offers[0].max_gpu_units,
            )

            if prev_cheapest_hr <= 0 or curr_cheapest_hr <= 0:
                continue

            # Price increase from previous → current (positive = prices went up)
            increase_pct = ((curr_cheapest_hr - prev_cheapest_hr) / prev_cheapest_hr) * 100.0

            # Opportunity: current prices are still low relative to the threshold
            # meaning prices haven't spiked *yet* but the trend suggests they will.
            # We look for the inverse: if prev was higher, current is a dip.
            price_dropped = curr_cheapest_hr < prev_cheapest_hr
            projected_increase = abs(increase_pct) if price_dropped else 0.0

            if projected_increase < opp_cfg.price_increase_threshold_pct:
                continue

            # How many can we afford within the opportunity budget?
            affordable = max(
                1,
                min(
                    opp_cfg.max_opportunity_instances,
                    int(opp_budget / curr_cheapest_hr) if curr_cheapest_hr > 0 else 0,
                ),
            )

            # Respect global caps
            global_headroom = max(0, self._cfg.budget.max_warm_instances - self.active_count)
            to_provision = min(affordable, global_headroom)

            self._log.info(
                "price_opportunity_detected",
                offload_class=cls,
                prev_price_hr=round(prev_cheapest_hr, 4),
                curr_price_hr=round(curr_cheapest_hr, 4),
                projected_increase_pct=round(projected_increase, 1),
                instances_to_provision=to_provision,
            )

            for offer in curr_offers[:to_provision]:
                hourly = estimate_hourly_cost(
                    offer,
                    cpu_vcpu=offer.max_cpu_vcpu,
                    memory_gib=offer.max_memory_gib,
                    gpu_units=offer.max_gpu_units,
                )
                if not self._budget_allows(hourly):
                    break
                await self._create_warm_instance(
                    offer, cls, hourly, reason="price_opportunity",
                )

    # ── Helpers ───────────────────────────────────────────────────────

    def _count_active(self, offload_class: OffloadClass) -> int:
        """Count active (provisioning + ready) instances for a class."""
        return sum(
            1
            for i in self._pool.values()
            if (
                i.offload_class == offload_class
                and i.status in (WarmInstanceStatus.PROVISIONING, WarmInstanceStatus.READY)
            )
        )

    def _budget_allows(self, additional_hourly_usd: float) -> bool:
        """Check whether adding an instance would exceed the hourly budget."""
        return (
            self.current_hourly_spend + additional_hourly_usd
            <= self._cfg.budget.max_pre_warm_budget_usd_per_hour
        )

    def _cheapest_offers_for_class(
        self,
        cls: OffloadClass,
        snapshot: PricingSurfaceSnapshot,
    ) -> list[SubstrateOffer]:
        """Return valid offers for a class, sorted cheapest-first by hourly rate."""
        offers = snapshot.filter_by_class(cls)
        if not offers:
            return []
        return sorted(
            offers,
            key=lambda o: estimate_hourly_cost(
                o,
                cpu_vcpu=o.max_cpu_vcpu,
                memory_gib=o.max_memory_gib,
                gpu_units=o.max_gpu_units,
            ),
        )

    async def _create_warm_instance(
        self,
        offer: SubstrateOffer,
        offload_class: OffloadClass,
        hourly_cost: float,
        reason: str,
    ) -> WarmInstance:
        """
        Create a new WarmInstance record in the pool.

        Spec 27 §16 + Oikos integration: runs the Oikos metabolic gate before
        committing the spend so the organism's economic health can veto
        speculative pre-warming during austerity or starvation.

        Gate parameters:
          - action_type: "sacm_pre_warm"
          - priority: GROWTH (speculative capacity, not operational survival)
          - estimated_cost: hourly_cost_usd (upper-bound for the first hour)
        """
        if self._oikos is not None:
            try:
                from decimal import Decimal

                from systems.oikos.models import MetabolicPriority

                gate_ok = await self._oikos.check_metabolic_gate(
                    action_type="sacm_pre_warm",
                    action_id=offer.id,
                    estimated_cost_usd=Decimal(str(round(hourly_cost, 6))),
                    priority=MetabolicPriority.GROWTH,
                    rationale=(
                        f"Pre-warm {offload_class} instance on {offer.provider_id} "
                        f"for {reason} at ${hourly_cost:.4f}/hr"
                    ),
                )
                if not gate_ok:
                    self._log.info(
                        "pre_warm_metabolic_gate_denied",
                        provider_id=offer.provider_id,
                        offload_class=offload_class,
                        hourly_cost_usd=round(hourly_cost, 4),
                        reason=reason,
                    )
                    # Return a stub in RELEASED state so callers skip without error
                    return WarmInstance(
                        offer_id=offer.id,
                        provider_id=offer.provider_id,
                        offload_class=offload_class,
                        resources=ResourceEnvelope(
                            cpu_vcpu=offer.max_cpu_vcpu,
                            memory_gib=offer.max_memory_gib,
                            storage_gib=offer.max_storage_gib,
                            gpu_units=offer.max_gpu_units,
                            gpu_vram_gib=offer.gpu_vram_gib,
                        ),
                        hourly_cost_usd=hourly_cost,
                        status=WarmInstanceStatus.RELEASED,
                        reason=f"metabolic_gate_denied:{reason}",
                    )
            except Exception as exc:
                # Non-fatal: if Oikos is unavailable, log and proceed
                self._log.warning(
                    "pre_warm_metabolic_gate_error",
                    error=str(exc),
                    provider_id=offer.provider_id,
                )

        inst = WarmInstance(
            offer_id=offer.id,
            provider_id=offer.provider_id,
            offload_class=offload_class,
            resources=ResourceEnvelope(
                cpu_vcpu=offer.max_cpu_vcpu,
                memory_gib=offer.max_memory_gib,
                storage_gib=offer.max_storage_gib,
                gpu_units=offer.max_gpu_units,
                gpu_vram_gib=offer.gpu_vram_gib,
            ),
            hourly_cost_usd=hourly_cost,
            status=WarmInstanceStatus.PROVISIONING,
            reason=reason,
        )
        async with self._lock:
            self._pool[inst.id] = inst
        self._log.info(
            "warm_instance_created",
            instance_id=inst.id,
            provider_id=offer.provider_id,
            offload_class=offload_class,
            hourly_cost_usd=round(hourly_cost, 4),
            reason=reason,
        )
        self._emit_pre_warm_event(inst, reason)

        # ── Provision via ProviderManager ──────────────────────────────
        # Call deploy() to provision the actual infrastructure instance.
        # Transition to READY on success, RELEASED on failure (pool cleans up).
        manager = self._provider_managers.get(offer.provider_id)
        if manager is not None:
            asyncio.create_task(
                self._provision_via_manager(inst, manager),
                name=f"sacm_provision_{inst.id[:8]}",
            )
        else:
            self._log.warning(
                "no_provider_manager_for_pre_warm",
                provider_id=offer.provider_id,
                instance_id=inst.id,
                hint="register_provider_manager() not called for this provider",
            )

        return inst

    async def _provision_via_manager(self, inst: WarmInstance, manager: Any) -> None:
        """
        Call ProviderManager.deploy() to provision a real instance and
        update the WarmInstance status based on the result.

        On success: instance transitions to READY via mark_ready().
        On failure: instance is removed from the pool (RELEASED) so it
        does not consume budget or block claims.
        """
        try:
            result = await manager.deploy(
                image="ghcr.io/ecodiaos/core:latest",
                env_vars={
                    "SACM_WARM_INSTANCE_ID": inst.id,
                    "SACM_OFFER_ID": inst.offer_id,
                    "SACM_OFFLOAD_CLASS": str(inst.offload_class),
                    "SACM_CPU_VCPU": str(inst.resources.cpu_vcpu),
                    "SACM_MEMORY_GIB": str(inst.resources.memory_gib),
                    "SACM_GPU_UNITS": str(inst.resources.gpu_units),
                },
            )

            if result.success:
                # Persist the provider-assigned endpoint/deployment_id
                async with self._lock:
                    current = self._pool.get(inst.id)
                    if current is not None:
                        current.reason = (
                            f"{inst.reason}:deployment_id={result.deployment_id}"
                        )
                await self.mark_ready(inst.id)
                self._log.info(
                    "warm_instance_provisioned",
                    instance_id=inst.id,
                    provider_id=inst.provider_id,
                    deployment_id=result.deployment_id,
                    endpoint=result.endpoint,
                )
            else:
                self._log.warning(
                    "warm_instance_provision_failed",
                    instance_id=inst.id,
                    provider_id=inst.provider_id,
                    error=result.error,
                )
                await self.release(inst.id)

        except Exception as exc:
            self._log.error(
                "warm_instance_provision_error",
                instance_id=inst.id,
                provider_id=inst.provider_id,
                error=str(exc),
                exc_info=True,
            )
            await self.release(inst.id)

    def _emit_pre_warm_event(self, inst: WarmInstance, reason: str) -> None:
        """Emit SACM_PRE_WARM_PROVISIONED via Synapse so downstream systems can react."""
        if self._synapse is None:
            return
        from systems.synapse.types import SynapseEvent, SynapseEventType

        asyncio.create_task(
            self._synapse.event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.SACM_PRE_WARM_PROVISIONED,
                    source_system="sacm",
                    data={
                        "instance_id": inst.id,
                        "provider_id": inst.provider_id,
                        "offload_class": str(inst.offload_class),
                        "cpu_vcpu": inst.resources.cpu_vcpu,
                        "memory_gib": inst.resources.memory_gib,
                        "gpu_units": inst.resources.gpu_units,
                        "hourly_cost_usd": round(inst.hourly_cost_usd, 6),
                        "reason": reason,
                    },
                )
            ),
            name=f"sacm_pre_warm_{inst.id[:8]}",
        )
