"""
EcodiaOS — Benchmark Types

Shared primitives for the benchmarks system.
"""

from __future__ import annotations

from typing import Any
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now

class BenchmarkSnapshot(EOSBaseModel):
    """A single benchmark run result, stored in TimescaleDB."""

    time: datetime = Field(default_factory=utc_now)
    instance_id: str

    # ── Five KPIs ────────────────────────────────────────────────────
    decision_quality: float | None = None
    """% of Nova outcomes rated positive (success). None if no data yet."""

    llm_dependency: float | None = None
    """% of decisions that required an LLM call (slow_path / total). None if no data."""

    economic_ratio: float | None = None
    """Oikos revenue_7d / costs_7d. >1 = metabolically positive. None if costs == 0."""

    learning_rate: float | None = None
    """Evo hypotheses confirmed (supported) in the last benchmark window."""

    mutation_success_rate: float | None = None
    """Simula proposals_approved / proposals_received. None if no proposals."""

    effective_intelligence_ratio: float | None = None
    """Telos effective_I — nominal_I scaled by all four drive multipliers. Master metric."""

    compression_ratio: float | None = None
    """Logos intelligence ratio I = K(reality_modeled) / K(model). >1 means compressive."""

    # ── Diagnostic metadata ───────────────────────────────────────────
    errors: dict[str, str] = Field(default_factory=dict)
    """Per-metric collection errors, keyed by metric name."""

    raw: dict[str, Any] = Field(default_factory=dict)
    """Raw values from health/stats endpoints for debugging."""


class MetricRegression(EOSBaseModel):
    """Describes a regression detected in a single KPI."""

    metric: str
    current_value: float
    rolling_avg: float
    regression_pct: float
    """How much below rolling average (as %, positive = worse)."""

    threshold_pct: float = 20.0


class BenchmarkTrend(EOSBaseModel):
    """Trend data for the dashboard — multiple snapshots for a single KPI."""

    metric: str
    points: list[dict[str, Any]] = Field(default_factory=list)
    """Each point: {time: ISO str, value: float | None}"""

    rolling_avg: float | None = None
    latest: float | None = None
