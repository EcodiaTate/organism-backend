"""
EcodiaOS - Telemetry Primitives

Metrics, health checks, and observability data types.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from primitives.common import EOSBaseModel, HealthStatus, utc_now


class MetricPoint(EOSBaseModel):
    """A single metric data point."""

    time: datetime = Field(default_factory=utc_now)
    system: str
    metric: str
    value: float
    labels: dict[str, str] = Field(default_factory=dict)


class SystemHealth(EOSBaseModel):
    """Health status of a single cognitive system."""

    status: HealthStatus = HealthStatus.HEALTHY
    latency_ms: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class InstanceHealth(EOSBaseModel):
    """Complete health snapshot of the EOS instance."""

    status: HealthStatus = HealthStatus.HEALTHY
    instance_id: str = ""
    instance_name: str = ""
    uptime_seconds: int = 0
    cycle_count: int = 0
    autonomy_level: int = 1
    systems: dict[str, SystemHealth] = Field(default_factory=dict)
    data_stores: dict[str, SystemHealth] = Field(default_factory=dict)
    affect: dict[str, float] = Field(default_factory=dict)


class SystemHealthSummary(EOSBaseModel):
    """Per-system health summary for ORGANISM_TELEMETRY payload."""

    latency_ema_ms: float = 0.0
    consecutive_misses: int = 0
    restart_count: int = 0
    status: str = "healthy"


class OrganismTelemetry(EOSBaseModel):
    """
    Unified organism state snapshot emitted every 50 cycles by Synapse.

    This is the single most important self-awareness primitive - it bundles
    every vital sign the organism produces into one coherent payload that Nova
    and other consumers can use without subscribing to 8 separate streams.
    """

    # Metabolic state
    burn_rate_usd_per_hour: float = 0.0
    runway_hours: float = float("inf")
    per_provider_cost: dict[str, float] = Field(default_factory=dict)
    infra_cost_usd_per_hour: float = 0.0

    # Coherence state
    phi: float = 0.0
    resonance: float = 0.0
    diversity: float = 0.0
    synchrony: float = 0.0
    coherence_composite: float = 0.0

    # Rhythm
    rhythm_state: str = "normal"
    cycles_in_rhythm_state: int = 0

    # Health: per-system summaries
    health: dict[str, SystemHealthSummary] = Field(default_factory=dict)

    # Resources: per-system CPU utilisation
    cpu_per_system: dict[str, float] = Field(default_factory=dict)
    total_cpu_pct: float = 0.0

    # Emotions: active emotion labels from Soma (empty if unavailable)
    emotions: list[str] = Field(default_factory=list)

    # Interoception: log-derived signals (None if interoception loop not yet run)
    error_rate_per_min: float | None = None
    cascade_pressure: bool = False
    latency_spike_active: bool = False

    # Infrastructure cost context
    infra_resources: dict[str, float] = Field(default_factory=dict)

    # Two-ledger dependency ratio: infra_burn / total_burn (0 = fully self-funded)
    # API costs come from organism's wallet; infra costs billed to human.
    # Target: dependency_ratio → 0 as organism earns its own compute.
    api_burn_rate_usd_per_hour: float = 0.0
    dependency_ratio: float = 0.0  # 0.0 = organism pays all; 1.0 = human pays all

    # Persona - public identity handle so Nova/consumers know the organism's name
    # None until PersonaEngine has generated and sealed the initial persona.
    persona_handle: str | None = None

    # Provenance
    cycle_number: int = 0
    timestamp: datetime = Field(default_factory=utc_now)

    def to_nova_summary(self) -> str:
        """
        One-sentence natural-language summary for injection into Nova's
        deliberation context.  Keeps LLM prompt overhead tiny while giving
        Nova full situational awareness.
        """
        runway = (
            f"{self.runway_hours:.0f}h" if self.runway_hours < 1000 else "inf"
        )
        health_issues = [
            f"{sid} ({s.latency_ema_ms:.0f}ms)"
            for sid, s in self.health.items()
            if s.status not in ("healthy", "starting")
        ]
        health_str = (
            f"{len(health_issues)} degraded ({', '.join(health_issues[:3])})"
            if health_issues
            else "all healthy"
        )
        emotions_str = (
            ", ".join(self.emotions[:3]) if self.emotions else "none"
        )
        cascade_flag = " CASCADE-PRESSURE" if self.cascade_pressure else ""
        dep_flag = f" dep:{self.dependency_ratio:.0%}" if self.dependency_ratio > 0.1 else ""
        return (
            f"Burn: ${self.burn_rate_usd_per_hour:.2f}/hr (api:${self.api_burn_rate_usd_per_hour:.2f}), "
            f"runway: {runway}{dep_flag}. "
            f"Coherence: {self.coherence_composite:.2f}. "
            f"Rhythm: {self.rhythm_state}. "
            f"Emotions: [{emotions_str}]. "
            f"Health: {health_str}. "
            f"CPU: {self.total_cpu_pct:.0f}%.{cascade_flag}"
        )
