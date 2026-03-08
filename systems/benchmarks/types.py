"""
EcodiaOS — Benchmark Types

Shared primitives for the benchmarks system.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now
from primitives.evolutionary import BedauPackardStats


class DomainKPI(EOSBaseModel):
    """KPIs specific to a domain specialization.

    Computed daily by DomainKPICalculator from EpisodeOutcome history.
    Emitted on the Synapse bus as DOMAIN_KPI_SNAPSHOT payloads.
    """

    domain: str
    """Domain identifier: 'software_development', 'art', 'trading', 'yield', etc."""

    timestamp: datetime = Field(default_factory=utc_now)

    # ── Task outcomes ─────────────────────────────────────────────────
    attempts: int = 0
    """Total tasks attempted in this domain within the lookback window."""

    successes: int = 0
    """Tasks that completed with outcome == 'success'."""

    success_rate: float = 0.0
    """successes / attempts. 0.0 when attempts == 0."""

    # ── Domain economics ──────────────────────────────────────────────
    revenue_total_usd: Decimal = Decimal(0)
    """Total revenue earned from this domain (sum of EpisodeOutcome.revenue)."""

    cost_total_usd: Decimal = Decimal(0)
    """Total compute/resource cost spent on this domain."""

    net_profit_usd: Decimal = Decimal(0)
    """revenue_total_usd − cost_total_usd."""

    profitability: float = 0.0
    """net_profit / revenue_total when revenue > 0, else 0.0."""

    revenue_per_hour: Decimal = Decimal(0)
    """revenue_total_usd / hours_spent. Key efficiency metric."""

    revenue_per_attempt: Decimal = Decimal(0)
    """revenue_total_usd / attempts."""

    # ── Time ─────────────────────────────────────────────────────────
    hours_spent: float = 0.0
    """Total wall-clock hours spent on domain tasks."""

    tasks_completed: int = 0
    """Alias for successes — counts fully completed tasks."""

    avg_task_duration_hours: float = 0.0
    """hours_spent / attempts."""

    # ── Quality ───────────────────────────────────────────────────────
    customer_satisfaction: float = 0.0
    """[0, 1] — averaged from EpisodeOutcome.custom_metrics['customer_satisfaction']."""

    rework_rate: float = 0.0
    """[0, 1] — fraction of tasks requiring revision (from episode metadata)."""

    # ── Domain-specific custom metrics ────────────────────────────────
    custom_metrics: dict[str, float] = Field(default_factory=dict)
    """
    Domain-specific measurements averaged across episodes.
    Examples:
      software_development: {'code_quality': 0.85, 'deployment_time_hours': 2.3}
      art: {'aesthetic_score': 0.72, 'revision_count': 1.4}
      trading: {'sharpe_ratio': 1.8, 'max_drawdown': 0.05}
    """

    # ── Trend ─────────────────────────────────────────────────────────
    trend_direction: str = "stable"
    """'improving' | 'declining' | 'stable' — based on success_rate vs prior period."""

    trend_magnitude: float = 0.0
    """Absolute change in success_rate vs prior period [0, 1]."""

    # ── Lookback metadata ─────────────────────────────────────────────
    lookback_hours: int = 168
    """Window over which these KPIs were computed (default: 7 days)."""


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

    # ── Evolutionary layer ───────────────────────────────────────────
    bedau_packard: BedauPackardStats | None = None
    """Population-level Bedau-Packard evolutionary activity stats. None until first computation."""

    evolutionary_fitness: dict[str, Any] = Field(default_factory=dict)
    """Aggregated fitness observables from Evo (Loop 6). Keys: hypotheses_evaluated, integrated, schemas_induced, consolidation_count."""

    constitutional_phenotype_divergence: float | None = None
    """
    Variance of drive-weight vectors across the fleet, derived from TELOS_POPULATION_SNAPSHOT.
    Computed as mean of per-drive variances across all fleet instances (coherence, care, growth, honesty).
    Rising trend alongside Bedau-Packard A(t) = adaptive speciation, not just random drift.
    None until first TELOS_POPULATION_SNAPSHOT is received with ≥2 instances.
    """

    # ── Domain specialization KPIs ───────────────────────────────────
    domain_kpis: dict[str, DomainKPI] = Field(default_factory=dict)
    """Per-domain KPI snapshots. Keys are domain names ('software_development', etc.)."""

    primary_domain: str = "generalist"
    """The domain with the highest success_rate. 'generalist' when no episodes yet."""

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
