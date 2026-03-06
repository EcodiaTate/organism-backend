"""
EcodiaOS — Axon Scheduler Types

Data types for the energy-aware task scheduling interceptor.

GridReading: a single carbon intensity / energy cost snapshot from any provider.
DeferredTask: a high-compute execution request parked until the grid is clean.
ComputeIntensity: classification of how energy-hungry a task is.
"""

from __future__ import annotations

import enum
from datetime import datetime  # noqa: TC003 — Pydantic requires at runtime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ─── Enums ────────────────────────────────────────────────────────


class ComputeIntensity(enum.StrEnum):
    """How energy-hungry is this task?"""

    LOW = "low"          # Internal bookkeeping, memory writes
    MODERATE = "moderate"  # Single LLM call, short API requests
    HIGH = "high"        # Multi-turn LLM, code generation, GRPO training
    CRITICAL = "critical"  # Model fine-tuning, batch simulation


class DeferralReason(enum.StrEnum):
    """Why was a task deferred?"""

    CARBON_INTENSITY = "carbon_intensity"
    ENERGY_COST = "energy_cost"
    MANUAL_HOLD = "manual_hold"


class DeferralStatus(enum.StrEnum):
    """Lifecycle of a deferred task."""

    WAITING = "waiting"        # In the deferred queue
    RELEASED = "released"      # Grid recovered, task re-queued for execution
    EXPIRED = "expired"        # TTL exceeded, task dropped
    CANCELLED = "cancelled"    # Manually cancelled


# ─── Grid Reading ─────────────────────────────────────────────────


class GridReading(EOSBaseModel):
    """
    A single point-in-time snapshot of grid energy conditions.

    Provider-agnostic: both Electricity Maps and WattTime produce this.
    """

    carbon_intensity_g: float = 0.0
    """Carbon intensity in gCO2eq/kWh."""

    energy_cost_usd_kwh: float | None = None
    """Spot energy price in USD/kWh (optional — not all providers supply this)."""

    zone: str = ""
    """Grid zone identifier (e.g. 'AU-NSW', 'US-CAL-CISO')."""

    provider: str = ""
    """Which API produced this reading."""

    fetched_at: datetime = Field(default_factory=utc_now)
    """When the reading was fetched from the upstream API."""

    raw: dict[str, Any] = Field(default_factory=dict)
    """Raw provider response for debugging."""

    @property
    def is_stale(self) -> bool:
        """True if the reading is older than 30 minutes."""
        age_s = (utc_now() - self.fetched_at).total_seconds()
        return age_s > 1800


# ─── Deferred Task ────────────────────────────────────────────────


class DeferredTask(EOSBaseModel):
    """
    A high-compute execution request parked in the deferred sleep-cycle queue.

    When the grid returns to GREEN_SURPLUS or NORMAL, deferred tasks are
    re-queued for execution in FIFO order.
    """

    id: str = Field(default_factory=new_id)
    intent_id: str
    """The original Intent ID that spawned this execution."""

    execution_request_json: dict[str, Any] = Field(default_factory=dict)
    """Serialised ExecutionRequest — restored when the task is released."""

    compute_intensity: ComputeIntensity = ComputeIntensity.HIGH
    deferral_reason: DeferralReason = DeferralReason.CARBON_INTENSITY
    status: DeferralStatus = DeferralStatus.WAITING

    carbon_intensity_at_deferral: float = 0.0
    """Grid carbon intensity when the task was deferred."""

    deferred_at: datetime = Field(default_factory=utc_now)
    max_defer_seconds: int = 7200
    """Maximum time to hold a deferred task before expiring (default 2 hours)."""

    released_at: datetime | None = None
    release_carbon_intensity: float | None = None

    @property
    def is_expired(self) -> bool:
        """True if the task has exceeded its maximum deferral window."""
        age_s = (utc_now() - self.deferred_at).total_seconds()
        return age_s > self.max_defer_seconds


# ─── Interceptor Decision ────────────────────────────────────────


class InterceptDecision(EOSBaseModel):
    """The interceptor's verdict on whether to execute or defer a task."""

    should_defer: bool = False
    reason: str = ""
    compute_intensity: ComputeIntensity = ComputeIntensity.LOW
    current_carbon_g: float | None = None
    threshold_carbon_g: float | None = None
