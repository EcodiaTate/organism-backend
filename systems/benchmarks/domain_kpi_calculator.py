"""
EcodiaOS - Domain KPI Calculator

Computes per-domain KPI snapshots from EpisodeOutcome history.
Used by BenchmarkService to populate BenchmarkSnapshot.domain_kpis daily.

Design
──────
• Stateful deque of EpisodeOutcome objects - systems enrich episodes with
  domain/outcome/revenue/cost/duration_ms/custom_metrics before emitting
  DOMAIN_EPISODE_RECORDED on the Synapse bus.
• calculate_for_domain(domain) slices the deque, computes all DomainKPI
  fields, and compares to the prior half-period for trend detection.
• No LLM calls, no I/O - pure in-process computation.
• Thread-safe: asyncio single-threaded; deque operations are atomic.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import structlog

from systems.benchmarks.types import DomainKPI

logger = structlog.get_logger("systems.benchmarks.domain_kpi")


class EpisodeRecord:
    """Lightweight in-process episode record for domain KPI computation.

    Systems emit DOMAIN_EPISODE_RECORDED events with these fields;
    DomainKPICalculator.record_episode() ingests them.
    """

    __slots__ = (
        "domain",
        "outcome",
        "revenue",
        "cost_usd",
        "duration_ms",
        "custom_metrics",
        "timestamp",
        "recorded_at",
    )

    def __init__(
        self,
        domain: str,
        outcome: str,
        revenue: Decimal,
        cost_usd: Decimal,
        duration_ms: int,
        custom_metrics: dict[str, float],
        timestamp: datetime | None = None,
    ) -> None:
        self.domain = domain
        self.outcome = outcome  # "success" | "partial" | "failure"
        self.revenue = revenue
        self.cost_usd = cost_usd
        self.duration_ms = duration_ms
        self.custom_metrics = custom_metrics
        self.timestamp = timestamp or datetime.now(tz=timezone.utc)
        self.recorded_at = time.time()  # monotonic for fast cutoff checks


class DomainKPICalculator:
    """Computes per-domain KPI snapshots from episode history.

    Usage:
        calc = DomainKPICalculator()
        calc.record_episode(episode_data)
        kpi = calc.calculate_for_domain("software_development", lookback_hours=168)
        domains = calc.active_domains(min_attempts=3, lookback_hours=168)
    """

    def __init__(self, max_history: int = 10_000) -> None:
        self._episodes: deque[EpisodeRecord] = deque(maxlen=max_history)
        self._logger = logger

    # ─── Ingestion ────────────────────────────────────────────────────

    def record_episode(self, data: dict[str, Any]) -> None:
        """Ingest an episode from a DOMAIN_EPISODE_RECORDED event payload."""
        try:
            domain = str(data.get("domain", "generalist"))
            outcome = str(data.get("outcome", "failure"))
            revenue = _to_decimal(data.get("revenue", 0))
            cost_usd = _to_decimal(data.get("cost_usd", 0))
            duration_ms = int(data.get("duration_ms", 0))
            custom_metrics = dict(data.get("custom_metrics", {}))
            ts_raw = data.get("timestamp")
            timestamp = _parse_ts(ts_raw) if ts_raw else None

            ep = EpisodeRecord(
                domain=domain,
                outcome=outcome,
                revenue=revenue,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
                custom_metrics=custom_metrics,
                timestamp=timestamp,
            )
            self._episodes.append(ep)
        except Exception as exc:
            self._logger.warning("domain_episode_record_failed", error=str(exc))

    # ─── Domain discovery ─────────────────────────────────────────────

    def active_domains(
        self,
        min_attempts: int = 1,
        lookback_hours: int = 168,
    ) -> list[str]:
        """Return domain names with ≥ min_attempts episodes in the lookback window."""
        cutoff = time.time() - lookback_hours * 3600
        counts: dict[str, int] = {}
        for ep in self._episodes:
            if ep.recorded_at >= cutoff:
                counts[ep.domain] = counts.get(ep.domain, 0) + 1
        return [d for d, c in counts.items() if c >= min_attempts]

    # ─── KPI computation ──────────────────────────────────────────────

    def calculate_for_domain(
        self,
        domain: str,
        lookback_hours: int = 168,
    ) -> DomainKPI:
        """Calculate a DomainKPI snapshot for the given domain.

        Compares current period to the prior half-period for trend detection:
          - current window: [now - lookback_hours, now]
          - prior window:   [now - 2×lookback_hours, now - lookback_hours]
        """
        now_ts = time.time()
        cutoff = now_ts - lookback_hours * 3600
        prior_cutoff = now_ts - 2 * lookback_hours * 3600

        current_eps: list[EpisodeRecord] = []
        prior_eps: list[EpisodeRecord] = []

        for ep in self._episodes:
            if ep.domain != domain:
                continue
            if ep.recorded_at >= cutoff:
                current_eps.append(ep)
            elif ep.recorded_at >= prior_cutoff:
                prior_eps.append(ep)

        if not current_eps:
            return DomainKPI(
                domain=domain,
                lookback_hours=lookback_hours,
            )

        # ── Outcomes ──────────────────────────────────────────────────
        attempts = len(current_eps)
        successes = sum(1 for ep in current_eps if ep.outcome == "success")
        success_rate = successes / attempts if attempts > 0 else 0.0

        # ── Economics ─────────────────────────────────────────────────
        revenue_total = sum((ep.revenue for ep in current_eps), Decimal(0))
        cost_total = sum((ep.cost_usd for ep in current_eps), Decimal(0))
        net_profit = revenue_total - cost_total
        profitability = (
            float(net_profit / revenue_total)
            if revenue_total > Decimal(0)
            else 0.0
        )
        revenue_per_attempt = (
            revenue_total / attempts if attempts > 0 else Decimal(0)
        )

        # ── Time ──────────────────────────────────────────────────────
        total_ms = sum(ep.duration_ms for ep in current_eps)
        hours_spent = total_ms / 3_600_000
        revenue_per_hour = (
            revenue_total / Decimal(str(hours_spent))
            if hours_spent > 0
            else Decimal(0)
        )
        avg_task_duration_hours = hours_spent / attempts if attempts > 0 else 0.0

        # ── Quality ───────────────────────────────────────────────────
        customer_satisfaction = _avg_custom_metric(current_eps, "customer_satisfaction")
        rework_rate = _avg_custom_metric(current_eps, "rework_rate")

        # ── Custom metrics ────────────────────────────────────────────
        custom_metrics = _aggregate_custom_metrics(current_eps)
        # Remove quality metrics already extracted as first-class fields
        custom_metrics.pop("customer_satisfaction", None)
        custom_metrics.pop("rework_rate", None)

        # ── Trend ─────────────────────────────────────────────────────
        prior_success_rate: float | None = None
        if prior_eps:
            prior_attempts = len(prior_eps)
            prior_successes = sum(1 for ep in prior_eps if ep.outcome == "success")
            prior_success_rate = (
                prior_successes / prior_attempts if prior_attempts > 0 else 0.0
            )

        trend_direction, trend_magnitude = _compute_trend(
            success_rate, prior_success_rate
        )

        return DomainKPI(
            domain=domain,
            timestamp=datetime.now(tz=timezone.utc),
            attempts=attempts,
            successes=successes,
            success_rate=round(success_rate, 4),
            revenue_total_usd=revenue_total,
            cost_total_usd=cost_total,
            net_profit_usd=net_profit,
            profitability=round(profitability, 4),
            revenue_per_hour=revenue_per_hour,
            revenue_per_attempt=revenue_per_attempt,
            hours_spent=round(hours_spent, 4),
            tasks_completed=successes,
            avg_task_duration_hours=round(avg_task_duration_hours, 4),
            customer_satisfaction=round(customer_satisfaction, 4),
            rework_rate=round(rework_rate, 4),
            custom_metrics=custom_metrics,
            trend_direction=trend_direction,
            trend_magnitude=round(trend_magnitude, 4),
            lookback_hours=lookback_hours,
        )

    def calculate_all(
        self,
        lookback_hours: int = 168,
        min_attempts: int = 1,
    ) -> dict[str, DomainKPI]:
        """Calculate KPIs for all active domains."""
        domains = self.active_domains(
            min_attempts=min_attempts, lookback_hours=lookback_hours
        )
        return {
            domain: self.calculate_for_domain(domain, lookback_hours=lookback_hours)
            for domain in domains
        }

    def primary_domain(self, domain_kpis: dict[str, DomainKPI]) -> str:
        """Return the domain with the highest success_rate, or 'generalist'."""
        if not domain_kpis:
            return "generalist"
        return max(domain_kpis.values(), key=lambda k: k.success_rate).domain


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return Decimal(0)


def _parse_ts(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return None


def _avg_custom_metric(episodes: list[EpisodeRecord], key: str) -> float:
    values = [
        float(ep.custom_metrics[key])
        for ep in episodes
        if key in ep.custom_metrics
    ]
    return sum(values) / len(values) if values else 0.0


def _aggregate_custom_metrics(episodes: list[EpisodeRecord]) -> dict[str, float]:
    """Average each custom metric key across all episodes that carry it."""
    buckets: dict[str, list[float]] = {}
    for ep in episodes:
        for k, v in ep.custom_metrics.items():
            try:
                buckets.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                pass
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def _compute_trend(
    current_rate: float,
    prior_rate: float | None,
    threshold: float = 0.05,
) -> tuple[str, float]:
    """Return (direction, magnitude) given current and prior success rates."""
    if prior_rate is None:
        return "stable", 0.0
    delta = current_rate - prior_rate
    magnitude = abs(delta)
    if magnitude < threshold:
        return "stable", magnitude
    return ("improving" if delta > 0 else "declining"), magnitude
