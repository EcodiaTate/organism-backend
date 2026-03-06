"""
EcodiaOS — SACM Workload Descriptors

Defines the vocabulary for describing offloadable workloads:

  - OffloadClass: coarse resource profile (GPU-heavy, CPU-bound, etc.)
  - WorkloadDescriptor: full specification of what a workload needs
    (resource envelope, latency budget, encryption requirements)
  - WorkloadResult: outcome of executing a workload on a remote substrate

The WorkloadDescriptor is the input to the optimizer — it carries enough
information for the cost function to score every available SubstrateOffer
and for the execution engine to dispatch, encrypt, and verify the result.

Cost model (used by optimizer.py):
  total_cost = (
      cpu_vcpu * duration_s * price_cpu_per_vcpu_s
    + memory_gib * duration_s * price_mem_per_gib_s
    + gpu_units  * duration_s * price_gpu_per_unit_s
    + storage_gib * price_storage_per_gib_s * duration_s
    + egress_gib  * price_egress_per_gib
  )
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

import structlog
from pydantic import Field, field_validator

from primitives.common import (
    EOSBaseModel,
    Identified,
    Timestamped,
)

logger = structlog.get_logger("systems.sacm.workload")


# ─── Offload Classification ──────────────────────────────────────


class OffloadClass(enum.StrEnum):
    """
    Coarse workload profile that determines which substrates are eligible.

    The optimizer filters SubstrateOffers by OffloadClass before scoring,
    so GPU_HEAVY workloads never land on CPU-only nodes.
    """

    CPU_BOUND = "cpu_bound"
    """Compute-intensive but no GPU required (e.g. GRPO reward eval, code analysis)."""

    GPU_HEAVY = "gpu_heavy"
    """Requires dedicated GPU (e.g. model fine-tuning, embedding generation)."""

    MEMORY_INTENSIVE = "memory_intensive"
    """Large working set, moderate CPU (e.g. knowledge graph traversal, vector search)."""

    IO_BOUND = "io_bound"
    """Network/disk-heavy, low CPU (e.g. bulk API calls, data pipeline stages)."""

    GENERAL = "general"
    """No special requirements — can run on any substrate."""


class WorkloadPriority(enum.IntEnum):
    """
    Maps to the Oikos MetabolicPriority cascade.

    Higher numeric values = lower urgency = more price-sensitive.
    The optimizer uses priority to weight latency vs. cost tradeoffs.
    """

    CRITICAL = 0    # survival-tier: must run immediately, cost secondary
    HIGH = 1        # operational: prefer speed, tolerate modest premium
    NORMAL = 2      # growth-tier: balance cost and latency
    LOW = 3         # background: maximise cost savings, latency flexible
    BATCH = 4       # bulk: lowest cost, no latency requirement


class WorkloadStatus(enum.StrEnum):
    """Lifecycle states of a dispatched workload."""

    PENDING = "pending"             # created, not yet dispatched
    DISPATCHED = "dispatched"       # sent to remote substrate
    EXECUTING = "executing"         # confirmed running on substrate
    COMPLETED = "completed"         # result received (not yet verified)
    VERIFIED = "verified"           # passed deterministic/canary verification
    FAILED = "failed"               # execution or verification failed
    CANCELLED = "cancelled"         # withdrawn before completion


# ─── Resource Envelope ────────────────────────────────────────────


class ResourceEnvelope(EOSBaseModel):
    """
    Resource requirements for a single workload execution.

    All values are peak/upper-bound — the substrate must guarantee at
    least this much for the duration of the workload.
    """

    cpu_vcpu: float = 2.0
    """Required vCPUs (fractional ok, e.g. 0.5)."""

    memory_gib: float = 4.0
    """Required memory in GiB."""

    storage_gib: float = 20.0
    """Ephemeral storage in GiB (not persistent across workloads)."""

    gpu_units: float = 0.0
    """GPU units required (0 = no GPU). 1.0 = one full GPU."""

    gpu_vram_gib: float = 0.0
    """Minimum GPU VRAM in GiB (only relevant if gpu_units > 0)."""

    egress_gib: float = 1.0
    """Expected network egress in GiB (for cost estimation)."""

    @field_validator("cpu_vcpu", "memory_gib", "storage_gib", "gpu_units", "gpu_vram_gib", "egress_gib")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Resource value must be >= 0, got {v}")
        return v

    @property
    def requires_gpu(self) -> bool:
        return self.gpu_units > 0


# ─── Workload Descriptor ─────────────────────────────────────────


class WorkloadDescriptor(Identified, Timestamped):
    """
    Complete specification of a workload to be offloaded.

    This is the primary input to the SACM optimizer. It carries:
      - What to run (image, entrypoint, payload_hash)
      - What resources it needs (envelope)
      - How it should be classified (offload_class, priority)
      - Constraints (max latency, max cost, encryption, region)

    The optimizer scores SubstrateOffers against this descriptor
    using compute_total_cost() and returns a ranked placement plan.
    """

    # ── Identity ──
    tenant_id: str = ""
    """Owning organism/instance ID for multi-tenant isolation."""

    name: str = ""
    """Human-readable label (e.g. 'grpo-reward-eval-batch-42')."""

    # ── What to run ──
    image: str = ""
    """Container image URI (e.g. 'ghcr.io/ecodiaos/grpo-worker:latest')."""

    entrypoint: str = ""
    """Override entrypoint if different from image default."""

    env_vars: dict[str, str] = Field(default_factory=dict)
    """Environment variables to inject (secrets handled via encryption)."""

    payload_hash: str = ""
    """SHA-256 of the input payload — used for dedup and verification."""

    payload_size_bytes: int = 0
    """Size of the input payload in bytes (for transfer cost estimation)."""

    # ── Classification ──
    offload_class: OffloadClass = OffloadClass.GENERAL
    priority: WorkloadPriority = WorkloadPriority.NORMAL

    # ── Resource requirements ──
    resources: ResourceEnvelope = Field(default_factory=ResourceEnvelope)

    estimated_duration_s: float = 300.0
    """Estimated wall-clock execution time in seconds."""

    # ── Constraints ──
    max_latency_s: float = 0.0
    """Maximum acceptable dispatch-to-result latency. 0 = no constraint."""

    max_cost_usd: float = 0.0
    """Maximum acceptable total cost in USD. 0 = no constraint."""

    require_encryption: bool = True
    """Whether the payload must be encrypted in transit (default: yes)."""

    require_verification: bool = True
    """Whether the result must pass deterministic replay verification."""

    allowed_regions: list[str] = Field(default_factory=list)
    """Region allowlist. Empty = any region acceptable."""

    blocked_providers: list[str] = Field(default_factory=list)
    """Provider IDs to exclude (e.g. after trust violations)."""

    # ── Payload (raw items for batch execution) ──
    items: list[bytes] = Field(default_factory=list)
    """Serialised input items to be processed remotely.
    Each entry is one unit of work; the execution engine processes
    them independently and returns one output per input."""

    # ── Lifecycle ──
    status: WorkloadStatus = WorkloadStatus.PENDING

    # ── Metadata ──
    metadata: dict[str, str] = Field(default_factory=dict)
    """Arbitrary key-value pairs for operator tagging."""

    @property
    def item_count(self) -> int:
        """Number of payload items in this workload."""
        return len(self.items)

    @property
    def workload_id(self) -> str:
        """Alias for .id — used by the execution engine."""
        return self.id

    @property
    def has_latency_constraint(self) -> bool:
        return self.max_latency_s > 0

    @property
    def has_cost_constraint(self) -> bool:
        return self.max_cost_usd > 0


# ─── Workload Result ──────────────────────────────────────────────


class WorkloadResult(Identified, Timestamped):
    """
    Outcome of executing a workload on a remote substrate.

    Captures the raw output, cost accounting, timing, and verification
    status.  The verification subsystem (deterministic.py, canary audits)
    populates the verification fields after the result is received.
    """

    # ── Link to source ──
    workload_id: str
    """ID of the WorkloadDescriptor this result corresponds to."""

    provider_id: str = ""
    """Which SubstrateProvider executed the workload."""

    offer_id: str = ""
    """Which SubstrateOffer was accepted for placement."""

    # ── Output ──
    output_hash: str = ""
    """SHA-256 of the raw output bytes."""

    output_size_bytes: int = 0
    """Size of the output in bytes."""

    output_payload: bytes = b""
    """Raw output bytes (may be encrypted if require_encryption was set)."""

    # ── Timing ──
    dispatched_at: datetime | None = None
    execution_started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def total_latency_s(self) -> float | None:
        """Wall-clock time from dispatch to completion."""
        if self.dispatched_at is not None and self.completed_at is not None:
            return (self.completed_at - self.dispatched_at).total_seconds()
        return None

    @property
    def execution_duration_s(self) -> float | None:
        """Actual execution time on the substrate."""
        if self.execution_started_at is not None and self.completed_at is not None:
            return (self.completed_at - self.execution_started_at).total_seconds()
        return None

    # ── Cost accounting ──
    actual_cost_usd: float = 0.0
    """Actual cost charged by the provider (may differ from estimate)."""

    cost_breakdown: dict[str, float] = Field(default_factory=dict)
    """Per-resource cost breakdown: {"cpu": 0.12, "memory": 0.04, ...}."""

    # ── Verification ──
    verified: bool = False
    """Whether the result passed verification."""

    verification_method: str = ""
    """Which verification strategy was used (deterministic, canary, consensus)."""

    verification_report_id: str = ""
    """Link to the full VerificationReport."""

    # ── Status ──
    success: bool = False
    """Whether the workload executed successfully (before verification)."""

    error: str = ""
    """Error message if execution failed."""

    # ── Metadata ──
    metadata: dict[str, Any] = Field(default_factory=dict)
