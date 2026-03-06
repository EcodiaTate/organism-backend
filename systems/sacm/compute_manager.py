"""
EcodiaOS — SACM Compute Resource Manager

Arbitrates compute resource allocation across subsystems.  When multiple
subsystems need GPU/CPU time (GRPO training, Simula mutation generation,
Oneiros dream processing, Nova LLM calls), this module:

  1. Resource Inventory  — tracks available local compute capacity
  2. Request Queue       — prioritises pending compute requests
  3. Allocation Policy   — resolves contention via priority + fairness
  4. Federation Offload  — sends work to peer instances when local
                           capacity is exhausted

Integrates with Synapse via set_synapse() to receive COMPUTE_REQUEST_SUBMITTED
events and publish allocation decisions (ALLOCATED / QUEUED / DENIED /
CAPACITY_EXHAUSTED / FEDERATION_OFFLOADED).
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id
from systems.sacm.workload import OffloadClass, ResourceEnvelope, WorkloadPriority

logger = structlog.get_logger("systems.sacm.compute_manager")


# ─── Capacity Model ────────────────────────────────────────────


class ComputeCapacity(EOSBaseModel):
    """Snapshot of available compute resources on a single node."""

    node_id: str = "local"
    cpu_vcpu_total: float = 16.0
    cpu_vcpu_available: float = 16.0
    memory_gib_total: float = 64.0
    memory_gib_available: float = 64.0
    gpu_units_total: float = 0.0
    gpu_units_available: float = 0.0
    gpu_vram_gib_total: float = 0.0
    gpu_vram_gib_available: float = 0.0

    def can_fit(self, envelope: ResourceEnvelope) -> bool:
        """Check whether the requested resources fit within available capacity."""
        return (
            self.cpu_vcpu_available >= envelope.cpu_vcpu
            and self.memory_gib_available >= envelope.memory_gib
            and self.gpu_units_available >= envelope.gpu_units
            and self.gpu_vram_gib_available >= envelope.gpu_vram_gib
        )

    def allocate(self, envelope: ResourceEnvelope) -> None:
        """Subtract requested resources from available pool."""
        self.cpu_vcpu_available -= envelope.cpu_vcpu
        self.memory_gib_available -= envelope.memory_gib
        self.gpu_units_available -= envelope.gpu_units
        self.gpu_vram_gib_available -= envelope.gpu_vram_gib

    def release(self, envelope: ResourceEnvelope) -> None:
        """Return resources to the available pool."""
        self.cpu_vcpu_available = min(
            self.cpu_vcpu_total, self.cpu_vcpu_available + envelope.cpu_vcpu
        )
        self.memory_gib_available = min(
            self.memory_gib_total, self.memory_gib_available + envelope.memory_gib
        )
        self.gpu_units_available = min(
            self.gpu_units_total, self.gpu_units_available + envelope.gpu_units
        )
        self.gpu_vram_gib_available = min(
            self.gpu_vram_gib_total, self.gpu_vram_gib_available + envelope.gpu_vram_gib
        )

    @property
    def utilisation_pct(self) -> float:
        """CPU utilisation as a percentage (0-100)."""
        if self.cpu_vcpu_total == 0:
            return 0.0
        return round(
            (1 - self.cpu_vcpu_available / self.cpu_vcpu_total) * 100, 1
        )


# ─── Compute Request ────────────────────────────────────────────


class ComputeRequest(EOSBaseModel):
    """A request from a subsystem for compute resources."""

    request_id: str = Field(default_factory=new_id)
    source_system: str
    """Which subsystem submitted the request (e.g. 'simula', 'grpo', 'oneiros', 'nova')."""

    resources: ResourceEnvelope = Field(default_factory=ResourceEnvelope)
    priority: WorkloadPriority = WorkloadPriority.NORMAL
    offload_class: OffloadClass = OffloadClass.GENERAL

    estimated_duration_s: float = 300.0
    """How long the requesting subsystem expects to hold the resources."""

    allow_federation: bool = True
    """Whether this request may be offloaded to a federated peer."""

    metadata: dict[str, str] = Field(default_factory=dict)
    submitted_at: float = Field(default_factory=time.monotonic)

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.submitted_at


class AllocationDecision(EOSBaseModel):
    """Result of the allocation policy for a single request."""

    request_id: str
    source_system: str
    outcome: str  # "allocated" | "queued" | "denied" | "federation_offloaded"
    node_id: str = ""
    reason: str = ""
    federation_peer_id: str = ""


class ActiveAllocation(EOSBaseModel):
    """Tracks a currently-held resource allocation."""

    request_id: str
    source_system: str
    resources: ResourceEnvelope
    node_id: str = "local"
    allocated_at: float = Field(default_factory=time.monotonic)
    estimated_duration_s: float = 300.0

    @property
    def held_s(self) -> float:
        return time.monotonic() - self.allocated_at


# ─── Allocation Policy ────────────────────────────────────────────


# Per-subsystem fair-share caps (fraction of total capacity).
# Subsystems not listed get DEFAULT_FAIR_SHARE.
_FAIR_SHARES: dict[str, float] = {
    "nova": 0.30,
    "simula": 0.25,
    "grpo": 0.20,
    "oneiros": 0.15,
}
_DEFAULT_FAIR_SHARE: float = 0.10

# Queue depth limit — requests beyond this are denied outright
_MAX_QUEUE_DEPTH: int = 64

# Stale request timeout — requests waiting longer than this are evicted
_QUEUE_STALE_TIMEOUT_S: float = 600.0


# ─── Compute Resource Manager ─────────────────────────────────────


class ComputeResourceManager:
    """
    Central compute arbitrator for EcodiaOS.

    Accepts resource requests from subsystems (via Synapse events or
    direct calls), applies priority + fair-share allocation, and emits
    decisions back onto the event bus.

    Implements the ManagedSystemProtocol (system_id + health) so it can
    be registered with Synapse for health monitoring.
    """

    system_id: str = "sacm"

    def __init__(
        self,
        capacity: ComputeCapacity | None = None,
    ) -> None:
        self._capacity = capacity or ComputeCapacity()
        self._log = logger.bind(component="sacm.compute_manager")

        # Synapse event bus reference — wired via set_synapse()
        self._synapse: Any = None

        # Request queue sorted by priority (lower value = higher priority)
        self._queue: list[ComputeRequest] = []

        # Active allocations by request_id
        self._active: dict[str, ActiveAllocation] = {}

        # Per-subsystem utilisation tracking (cpu_vcpu held)
        self._held_cpu: dict[str, float] = defaultdict(float)

        # Federation channel reference — wired externally
        self._federation: Any = None

        # Metrics
        self._total_allocated: int = 0
        self._total_queued: int = 0
        self._total_denied: int = 0
        self._total_offloaded: int = 0

    # ─── Synapse Integration ──────────────────────────────────────

    def set_synapse(self, synapse: Any) -> None:
        """
        Wire the SynapseService so SACM can subscribe to compute
        request events and publish allocation decisions.

        Call after Synapse is initialised (step 13 in main.py).
        """
        self._synapse = synapse
        self._subscribe_to_events()
        self._log.info("synapse_wired_to_sacm")

    def _subscribe_to_events(self) -> None:
        """Register SACM's event handlers on the Synapse event bus."""
        if self._synapse is None:
            return

        from systems.synapse.types import SynapseEventType

        try:
            event_bus = self._synapse.event_bus
        except AttributeError:
            self._log.warning("synapse_event_bus_unavailable", subsystem="sacm")
            return

        event_bus.subscribe(
            SynapseEventType.COMPUTE_REQUEST_SUBMITTED,
            self._on_compute_request,
        )

        # Listen to resource pressure to trigger pre-emptive queue drain
        event_bus.subscribe(
            SynapseEventType.RESOURCE_PRESSURE,
            self._on_resource_pressure,
        )

        self._log.info(
            "compute_events_subscribed",
            events=["compute_request_submitted", "resource_pressure"],
        )

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Fire-and-forget emit onto the Synapse event bus."""
        if self._synapse is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        asyncio.create_task(
            self._synapse.event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType(event_type),
                    source_system="sacm",
                    data=data,
                )
            ),
            name=f"sacm_{event_type}_{data.get('request_id', '')[:8]}",
        )

    # ─── Federation Wiring ────────────────────────────────────────

    def set_federation(self, federation: Any) -> None:
        """Wire the FederationService for compute offloading."""
        self._federation = federation
        self._log.info("federation_wired_to_sacm")

    # ─── Event Handlers ───────────────────────────────────────────

    async def _on_compute_request(self, event: Any) -> None:
        """Handle inbound COMPUTE_REQUEST_SUBMITTED from the event bus."""
        data = event.data
        request = ComputeRequest(
            request_id=data.get("request_id", new_id()),
            source_system=data.get("source_system", "unknown"),
            resources=ResourceEnvelope(**data.get("resources", {})),
            priority=WorkloadPriority(data.get("priority", WorkloadPriority.NORMAL.value)),
            offload_class=OffloadClass(data.get("offload_class", OffloadClass.GENERAL.value)),
            estimated_duration_s=float(data.get("estimated_duration_s", 300.0)),
            allow_federation=bool(data.get("allow_federation", True)),
            metadata=data.get("metadata", {}),
        )
        await self.submit_request(request)

    async def _on_resource_pressure(self, event: Any) -> None:
        """On resource pressure events, evict expired allocations to free capacity."""
        released = self._evict_expired_allocations()
        if released > 0:
            self._log.info(
                "resource_pressure_eviction",
                released_count=released,
            )
            # Try to drain queued requests with freed capacity
            await self._drain_queue()

    # ─── Core API ─────────────────────────────────────────────────

    async def submit_request(self, request: ComputeRequest) -> AllocationDecision:
        """
        Submit a compute resource request.

        Allocation flow:
          1. Check fair-share — deny if subsystem exceeds its share
          2. Try local allocation — fits in current capacity?
          3. Try federation offload — peer has capacity?
          4. Try preemption — can we reclaim from a lower-priority holder?
          5. Queue the request — wait for resources to free up
          6. Deny if queue is full
        """
        self._log.info(
            "compute_request_received",
            request_id=request.request_id,
            source=request.source_system,
            priority=request.priority.name,
            cpu=request.resources.cpu_vcpu,
            gpu=request.resources.gpu_units,
        )

        # Evict stale requests first
        self._evict_stale_queue_entries()

        # 1. Fair-share gate
        share_limit = _FAIR_SHARES.get(
            request.source_system, _DEFAULT_FAIR_SHARE
        )
        held_cpu = self._held_cpu[request.source_system]
        share_used = (
            held_cpu / self._capacity.cpu_vcpu_total
            if self._capacity.cpu_vcpu_total > 0
            else 0.0
        )

        if share_used >= share_limit and request.priority > WorkloadPriority.HIGH:
            # Allow CRITICAL and HIGH to bypass fair-share
            decision = AllocationDecision(
                request_id=request.request_id,
                source_system=request.source_system,
                outcome="denied",
                reason=(
                    f"fair_share_exceeded: {request.source_system} "
                    f"using {share_used:.0%} of {share_limit:.0%} cap"
                ),
            )
            self._total_denied += 1
            await self._emit("compute_request_denied", {
                "request_id": request.request_id,
                "source_system": request.source_system,
                "reason": decision.reason,
            })
            return decision

        # 2. Try local allocation
        if self._capacity.can_fit(request.resources):
            return await self._allocate_local(request)

        # 3. Try federation offload (if allowed and available)
        if request.allow_federation and self._federation is not None:
            offload_result = await self._try_federation_offload(request)
            if offload_result is not None:
                return offload_result

        # 4. Preemption check — can we preempt an expired allocation?
        preempted = await self._try_preemption(request)
        if preempted is not None:
            return preempted

        # 5. Queue if room
        if len(self._queue) < _MAX_QUEUE_DEPTH:
            return await self._enqueue(request)

        # 6. Queue full — deny
        decision = AllocationDecision(
            request_id=request.request_id,
            source_system=request.source_system,
            outcome="denied",
            reason="queue_full",
        )
        self._total_denied += 1
        await self._emit("compute_request_denied", {
            "request_id": request.request_id,
            "source_system": request.source_system,
            "reason": "queue_full",
        })
        return decision

    async def release(self, request_id: str) -> None:
        """Release an active allocation, returning resources to the pool."""
        allocation = self._active.pop(request_id, None)
        if allocation is None:
            return

        self._capacity.release(allocation.resources)
        self._held_cpu[allocation.source_system] = max(
            0.0,
            self._held_cpu[allocation.source_system] - allocation.resources.cpu_vcpu,
        )

        self._log.info(
            "allocation_released",
            request_id=request_id,
            source=allocation.source_system,
            held_s=round(allocation.held_s, 1),
        )

        # Drain queued requests now that capacity freed up
        await self._drain_queue()

    # ─── Internal Allocation Helpers ──────────────────────────────

    async def _allocate_local(self, request: ComputeRequest) -> AllocationDecision:
        """Allocate resources from local capacity."""
        self._capacity.allocate(request.resources)
        self._held_cpu[request.source_system] += request.resources.cpu_vcpu

        allocation = ActiveAllocation(
            request_id=request.request_id,
            source_system=request.source_system,
            resources=request.resources,
            node_id=self._capacity.node_id,
            estimated_duration_s=request.estimated_duration_s,
        )
        self._active[request.request_id] = allocation
        self._total_allocated += 1

        decision = AllocationDecision(
            request_id=request.request_id,
            source_system=request.source_system,
            outcome="allocated",
            node_id=self._capacity.node_id,
        )

        await self._emit("compute_request_allocated", {
            "request_id": request.request_id,
            "source_system": request.source_system,
            "node_id": self._capacity.node_id,
            "cpu_vcpu": request.resources.cpu_vcpu,
            "gpu_units": request.resources.gpu_units,
        })

        self._log.info(
            "compute_allocated",
            request_id=request.request_id,
            source=request.source_system,
            node=self._capacity.node_id,
            utilisation_pct=self._capacity.utilisation_pct,
        )
        return decision

    async def _enqueue(self, request: ComputeRequest) -> AllocationDecision:
        """Add request to the priority queue."""
        # Insert sorted by priority (lower value = higher priority)
        inserted = False
        for i, queued in enumerate(self._queue):
            if request.priority < queued.priority:
                self._queue.insert(i, request)
                inserted = True
                break
        if not inserted:
            self._queue.append(request)

        self._total_queued += 1

        decision = AllocationDecision(
            request_id=request.request_id,
            source_system=request.source_system,
            outcome="queued",
            reason=f"queue_position={self._queue.index(request) + 1}",
        )

        await self._emit("compute_request_queued", {
            "request_id": request.request_id,
            "source_system": request.source_system,
            "queue_depth": len(self._queue),
            "priority": request.priority.name,
        })

        # If capacity is fully exhausted, emit alert
        if self._capacity.utilisation_pct >= 95.0:
            await self._emit("compute_capacity_exhausted", {
                "utilisation_pct": self._capacity.utilisation_pct,
                "queue_depth": len(self._queue),
                "active_count": len(self._active),
            })

        return decision

    async def _try_preemption(
        self, request: ComputeRequest
    ) -> AllocationDecision | None:
        """
        Try to preempt an expired active allocation to make room.

        Only CRITICAL requests trigger preemption. Reclaims allocations
        that have exceeded their estimated duration.
        """
        if request.priority > WorkloadPriority.CRITICAL:
            return None

        # Find allocations that overstayed their estimate
        for alloc in list(self._active.values()):
            if alloc.held_s > alloc.estimated_duration_s:
                await self.release(alloc.request_id)
                if self._capacity.can_fit(request.resources):
                    return await self._allocate_local(request)

        return None

    async def _try_federation_offload(
        self, request: ComputeRequest
    ) -> AllocationDecision | None:
        """
        Attempt to offload compute to a federated peer instance.

        Uses the federation service's assistance request channel.
        """
        if self._federation is None:
            return None

        try:
            from primitives.federation import AssistanceRequest

            assistance = AssistanceRequest(
                requesting_instance_id=getattr(
                    self._federation, "instance_id", "local"
                ),
                description=(
                    f"Compute offload: {request.offload_class.value} workload, "
                    f"{request.resources.cpu_vcpu} vCPU, "
                    f"{request.resources.gpu_units} GPU, "
                    f"priority={request.priority.name}"
                ),
                knowledge_domain="compute",
                urgency=1.0 - (request.priority.value / WorkloadPriority.BATCH.value),
                reciprocity_offer="compute_reciprocity",
            )

            response = await self._federation.request_assistance(assistance)

            if response is not None and response.accepted:
                self._total_offloaded += 1
                peer_id = getattr(response, "peer_id", "unknown")
                decision = AllocationDecision(
                    request_id=request.request_id,
                    source_system=request.source_system,
                    outcome="federation_offloaded",
                    federation_peer_id=peer_id,
                    reason="local_capacity_exhausted",
                )

                await self._emit("compute_federation_offloaded", {
                    "request_id": request.request_id,
                    "source_system": request.source_system,
                    "peer_id": peer_id,
                    "cpu_vcpu": request.resources.cpu_vcpu,
                    "gpu_units": request.resources.gpu_units,
                })

                self._log.info(
                    "compute_federation_offloaded",
                    request_id=request.request_id,
                    peer=peer_id,
                )
                return decision

        except Exception as exc:
            self._log.warning(
                "federation_offload_failed",
                request_id=request.request_id,
                error=str(exc),
            )

        return None

    # ─── Queue Maintenance ────────────────────────────────────────

    async def _drain_queue(self) -> None:
        """Try to allocate queued requests with newly freed capacity."""
        remaining: list[ComputeRequest] = []

        for request in self._queue:
            if self._capacity.can_fit(request.resources):
                await self._allocate_local(request)
            else:
                remaining.append(request)

        self._queue = remaining

    def _evict_stale_queue_entries(self) -> None:
        """Remove requests that have been waiting too long."""
        now = time.monotonic()
        before = len(self._queue)
        self._queue = [
            r for r in self._queue
            if (now - r.submitted_at) < _QUEUE_STALE_TIMEOUT_S
        ]
        evicted = before - len(self._queue)
        if evicted > 0:
            self._log.info("stale_requests_evicted", count=evicted)

    def _evict_expired_allocations(self) -> int:
        """Release allocations that have exceeded their estimated duration by 2x."""
        expired_ids: list[str] = []
        for req_id, alloc in self._active.items():
            if alloc.held_s > alloc.estimated_duration_s * 2.0:
                expired_ids.append(req_id)

        for req_id in expired_ids:
            alloc = self._active.pop(req_id)
            self._capacity.release(alloc.resources)
            self._held_cpu[alloc.source_system] = max(
                0.0,
                self._held_cpu[alloc.source_system] - alloc.resources.cpu_vcpu,
            )
            self._log.warning(
                "allocation_expired_evicted",
                request_id=req_id,
                source=alloc.source_system,
                held_s=round(alloc.held_s, 1),
                estimated_s=alloc.estimated_duration_s,
            )

        return len(expired_ids)

    # ─── Inventory & Health ───────────────────────────────────────

    @property
    def inventory(self) -> ComputeCapacity:
        """Current resource inventory."""
        return self._capacity

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def active_count(self) -> int:
        return len(self._active)

    def update_capacity(self, capacity: ComputeCapacity) -> None:
        """Update the resource inventory (e.g. after hardware changes)."""
        self._capacity = capacity
        self._log.info(
            "capacity_updated",
            cpu_total=capacity.cpu_vcpu_total,
            gpu_total=capacity.gpu_units_total,
            mem_total=capacity.memory_gib_total,
        )

    async def health(self) -> dict[str, Any]:
        """ManagedSystemProtocol — health check for Synapse monitoring."""
        return {
            "status": "healthy" if self._capacity.utilisation_pct < 95.0 else "overloaded",
            "utilisation_pct": self._capacity.utilisation_pct,
            "active_allocations": len(self._active),
            "queue_depth": len(self._queue),
            "total_allocated": self._total_allocated,
            "total_queued": self._total_queued,
            "total_denied": self._total_denied,
            "total_offloaded": self._total_offloaded,
            "capacity": {
                "cpu_available": self._capacity.cpu_vcpu_available,
                "cpu_total": self._capacity.cpu_vcpu_total,
                "gpu_available": self._capacity.gpu_units_available,
                "gpu_total": self._capacity.gpu_units_total,
                "memory_available_gib": self._capacity.memory_gib_available,
                "memory_total_gib": self._capacity.memory_gib_total,
            },
            "per_subsystem_cpu": dict(self._held_cpu),
        }

    def snapshot(self) -> dict[str, Any]:
        """Quick summary for API endpoints / admin dashboards."""
        return {
            "utilisation_pct": self._capacity.utilisation_pct,
            "active_allocations": len(self._active),
            "queue_depth": len(self._queue),
            "total_allocated": self._total_allocated,
            "total_denied": self._total_denied,
            "total_offloaded": self._total_offloaded,
            "per_subsystem_cpu_held": dict(self._held_cpu),
        }
