"""
EcodiaOS - Synapse Resource Allocator

Adaptive resource allocation across cognitive systems. Tracks actual
resource consumption and rebalances budgets based on system load.

Uses psutil for process-level resource measurement (graceful fallback
if unavailable). Snapshots taken every ~33 cycles (~5s at 150ms/cycle).
"""

from __future__ import annotations

from typing import Any

import structlog

from systems.synapse.types import (
    BaseResourceAllocator,
    ResourceAllocation,
    ResourceSnapshot,
    SystemBudget,
)

logger = structlog.get_logger("systems.synapse.resources")

# Try to import psutil for resource measurement
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Default per-system budgets (from Synapse spec)
_DEFAULT_BUDGETS: dict[str, SystemBudget] = {
    "atune": SystemBudget(system_id="atune", cpu_share=0.20, memory_mb=512, io_priority=1),
    "memory": SystemBudget(system_id="memory", cpu_share=0.20, memory_mb=2048, io_priority=1),
    "nova": SystemBudget(system_id="nova", cpu_share=0.20, memory_mb=512, io_priority=2),
    "equor": SystemBudget(system_id="equor", cpu_share=0.10, memory_mb=256, io_priority=2),
    "voxis": SystemBudget(system_id="voxis", cpu_share=0.10, memory_mb=256, io_priority=3),
    "axon": SystemBudget(system_id="axon", cpu_share=0.08, memory_mb=256, io_priority=2),
    "evo": SystemBudget(system_id="evo", cpu_share=0.08, memory_mb=512, io_priority=4),
    "simula": SystemBudget(system_id="simula", cpu_share=0.02, memory_mb=128, io_priority=5),
    "synapse": SystemBudget(system_id="synapse", cpu_share=0.02, memory_mb=128, io_priority=1),
}


class ResourceAllocator(BaseResourceAllocator):
    """
    Manages per-system resource budgets and tracks actual utilisation.

    Provides two capabilities:
    1. Snapshot: capture current process-level resource utilisation
    2. Rebalance: adjust per-system allocations based on observed load

    The allocator is passive - it computes allocations but does not enforce them.
    Systems are expected to respect their budgets voluntarily.
    """

    @property
    def allocator_name(self) -> str:
        return "default"

    def __init__(self) -> None:
        self._logger = logger.bind(component="resource_allocator")
        self._budgets: dict[str, SystemBudget] = dict(_DEFAULT_BUDGETS)
        self._allocations: dict[str, ResourceAllocation] = {}
        self._latest_snapshot: ResourceSnapshot | None = None
        self._process: Any = None  # psutil.Process

        # Track per-system observed load (updated externally)
        self._system_loads: dict[str, float] = {}  # system_id → cpu_util [0, 1]

        # Initialise psutil process handle
        if _HAS_PSUTIL:
            try:
                self._process = psutil.Process()
                # Prime CPU measurement (first call always returns 0)
                self._process.cpu_percent(interval=None)
            except Exception:
                self._process = None

        self._logger.info("resource_allocator_initialized", has_psutil=_HAS_PSUTIL)

    # ─── Snapshot ────────────────────────────────────────────────────

    def capture_snapshot(self) -> ResourceSnapshot:
        """
        Capture current resource utilisation.

        Uses psutil for process-level metrics when available,
        falls back to zeros otherwise.
        """
        snapshot = ResourceSnapshot()

        if self._process is not None and _HAS_PSUTIL:
            try:
                # Process CPU (as percentage of all cores)
                snapshot.process_cpu_percent = self._process.cpu_percent(interval=None)
                # Process memory
                mem_info = self._process.memory_info()
                snapshot.process_memory_mb = mem_info.rss / (1024 * 1024)
                # System-wide
                snapshot.total_cpu_percent = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                snapshot.total_memory_mb = mem.used / (1024 * 1024)
                snapshot.total_memory_percent = mem.percent
            except Exception as exc:
                self._logger.debug("psutil_snapshot_failed", error=str(exc))

        self._latest_snapshot = snapshot
        return snapshot

    # ─── Rebalance ───────────────────────────────────────────────────

    def record_system_load(self, system_id: str, cpu_util: float) -> None:
        """Record observed CPU utilisation for a system."""
        self._system_loads[system_id] = max(0.0, min(1.0, cpu_util))

    def rebalance(self, cycle_period_ms: float) -> dict[str, ResourceAllocation]:
        """
        Compute per-system resource allocations based on budgets and observed load.

        Overloaded systems get a priority boost (smoothly, no sudden shifts).
        The total compute budget per cycle is the cycle period itself.

        Returns the new allocations dict.
        """
        allocations: dict[str, ResourceAllocation] = {}

        for sid, budget in self._budgets.items():
            # Base compute budget: share of the cycle period
            compute_ms = budget.cpu_share * cycle_period_ms

            # Burst allowance: overloaded systems get up to 2x their base
            observed = self._system_loads.get(sid, 0.0)
            burst = 1.0
            if observed > 0.8:
                burst = min(2.0, 1.0 + (observed - 0.8) * 5.0)  # Linear ramp

            # Priority boost: systems above 90% load get boosted
            priority_boost = 0.0
            if observed > 0.9:
                priority_boost = min(1.0, (observed - 0.9) * 10.0)

            allocations[sid] = ResourceAllocation(
                system_id=sid,
                compute_ms_per_cycle=round(compute_ms, 2),
                burst_allowance=round(burst, 2),
                priority_boost=round(priority_boost, 2),
            )

        self._allocations = allocations
        return allocations

    # ─── Accessors ───────────────────────────────────────────────────

    def get_budget(self, system_id: str) -> SystemBudget | None:
        return self._budgets.get(system_id)

    def get_allocation(self, system_id: str) -> ResourceAllocation | None:
        return self._allocations.get(system_id)

    @property
    def latest_snapshot(self) -> ResourceSnapshot | None:
        return self._latest_snapshot

    @property
    def stats(self) -> dict[str, Any]:
        snapshot_data = {}
        if self._latest_snapshot:
            snapshot_data = {
                "process_cpu_percent": self._latest_snapshot.process_cpu_percent,
                "process_memory_mb": round(self._latest_snapshot.process_memory_mb, 1),
                "total_cpu_percent": self._latest_snapshot.total_cpu_percent,
            }
        return {
            "has_psutil": _HAS_PSUTIL,
            "budgets": {sid: b.cpu_share for sid, b in self._budgets.items()},
            "system_loads": dict(self._system_loads),
            "snapshot": snapshot_data,
        }
