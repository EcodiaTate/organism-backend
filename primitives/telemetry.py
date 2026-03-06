"""
EcodiaOS — Telemetry Primitives

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
