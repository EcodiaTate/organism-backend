"""
EcodiaOS — SACM Cost Accounting

Tracks spend, computes savings, and bridges SACM cost data to the
Soma metabolic regulator so the organism *feels* its compute burn rate.

Main class:

  SACMCostAccounting
    Accumulates per-workload cost records, computes rolling totals and
    savings vs an on-demand baseline, and produces savings reports.
    Optionally holds a reference to SomaService and calls
    inject_external_stress() with a normalised burn-rate scalar on
    every recorded execution.

Metrics emitted:
  sacm.accounting.recorded         — Number of cost records logged
  sacm.accounting.rolling_spend    — Rolling spend window (USD)
  sacm.accounting.savings_ratio    — Cumulative savings / baseline ratio
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, utc_now

if TYPE_CHECKING:
    from collections.abc import Sequence

    from systems.oikos.service import OikosService
    from systems.soma.service import SomaService
logger = structlog.get_logger("systems.sacm.accounting")


# ─── Cost Record ────────────────────────────────────────────────


class CostRecord(Identified, Timestamped):
    """
    Immutable record of the cost incurred by a single workload execution.

    Links back to the workload and provider, captures both actual and
    baseline (on-demand) costs so savings can be derived.
    """

    workload_id: str
    provider_id: str
    offer_id: str = ""

    # Actual cost charged by the substrate provider
    actual_cost_usd: float = 0.0

    # What this workload would have cost on the on-demand baseline
    # (e.g. AWS/GCP spot or on-demand pricing for equivalent resources)
    baseline_cost_usd: float = 0.0

    # Breakdown by resource type
    cost_breakdown: dict[str, float] = Field(default_factory=dict)

    # Execution metadata
    duration_s: float = 0.0
    verified: bool = False
    accepted: bool = False

    @property
    def savings_usd(self) -> float:
        """How much was saved vs the on-demand baseline."""
        return max(0.0, self.baseline_cost_usd - self.actual_cost_usd)

    @property
    def savings_ratio(self) -> float:
        """Fraction saved: 0.0 = no savings, 1.0 = free."""
        if self.baseline_cost_usd <= 0:
            return 0.0
        return self.savings_usd / self.baseline_cost_usd


# ─── Savings Report ─────────────────────────────────────────────


class ProviderSpendSummary(EOSBaseModel):
    """Per-provider spend summary within a savings report."""

    provider_id: str
    total_actual_usd: float = 0.0
    total_baseline_usd: float = 0.0
    total_savings_usd: float = 0.0
    workload_count: int = 0


class SavingsReport(EOSBaseModel):
    """
    Aggregated savings report over a set of cost records.

    Produced by SACMCostAccounting.get_savings_report().
    """

    period_label: str = ""
    record_count: int = 0
    total_actual_usd: float = 0.0
    total_baseline_usd: float = 0.0
    total_savings_usd: float = 0.0
    savings_ratio: float = 0.0
    avg_cost_per_workload_usd: float = 0.0
    avg_savings_per_workload_usd: float = 0.0
    top_providers: list[ProviderSpendSummary] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)


# ─── SACMCostAccounting ─────────────────────────────────────────


# Default on-demand premium: the assumed markup of on-demand cloud
# pricing over SACM substrate pricing.  Used when a workload doesn't
# carry an explicit baseline cost.
_DEFAULT_BASELINE_PREMIUM: float = 1.43  # 43% more expensive on-demand


class SACMCostAccounting:
    """
    Tracks SACM compute spend and produces savings analytics.

    Responsibilities:
      - record_execution(): Log a completed workload's cost.
      - get_savings_report(): Aggregate savings over all or recent records.
      - Report spend to Soma treasury so the organism feels its burn rate.

    The accounting layer sits between the execution pipeline and the
    rest of the organism, ensuring every dollar spent is visible in
    both telemetry and the metabolic substrate.
    """

    def __init__(
        self,
        soma: Any | None = None,
        budget_usd_per_hour: float = 10.0,
        baseline_premium: float = _DEFAULT_BASELINE_PREMIUM,
        rolling_window_s: float = 3600.0,
    ) -> None:
        # soma is SomaService at runtime; typed as Any to avoid circular import
        self._soma: Any | None = soma
        # oikos is OikosService at runtime; typed as Any to avoid circular import
        self._oikos: Any | None = None
        self._baseline_premium = baseline_premium
        self._rolling_window_s = rolling_window_s
        self._budget_usd_per_hour = budget_usd_per_hour

        self._records: list[CostRecord] = []
        # Rolling window for burn-rate calculation
        self._rolling_costs: deque[tuple[float, float]] = deque()  # (timestamp, cost_usd)

        self._log = logger.bind(component="sacm.accounting")
        self._metrics_log = structlog.get_logger("systems.sacm.metrics")

    def wire_soma(self, soma: SomaService) -> None:
        """Attach a SomaService after construction (resolves init-order)."""
        self._soma = soma

    def wire_oikos(self, oikos: OikosService) -> None:
        """Attach an OikosService so every recorded spend is reported to the ledger."""
        self._oikos = oikos

    def record_execution(
        self,
        workload_id: str,
        provider_id: str,
        actual_cost_usd: float,
        baseline_cost_usd: float | None = None,
        cost_breakdown: dict[str, float] | None = None,
        duration_s: float = 0.0,
        verified: bool = False,
        accepted: bool = False,
        offer_id: str = "",
    ) -> CostRecord:
        """
        Record the cost of a completed workload execution.

        If baseline_cost_usd is not provided, it is estimated as
        actual_cost * baseline_premium (the assumed on-demand markup).

        Reports the spend to Soma so the organism's energy dimension
        reflects the real fiat burn.
        """
        if baseline_cost_usd is None:
            baseline_cost_usd = actual_cost_usd * self._baseline_premium

        record = CostRecord(
            workload_id=workload_id,
            provider_id=provider_id,
            offer_id=offer_id,
            actual_cost_usd=actual_cost_usd,
            baseline_cost_usd=baseline_cost_usd,
            cost_breakdown=cost_breakdown or {},
            duration_s=duration_s,
            verified=verified,
            accepted=accepted,
        )

        self._records.append(record)

        # Update rolling window
        now = time.monotonic()
        self._rolling_costs.append((now, actual_cost_usd))
        self._prune_rolling_window(now)

        # Translate burn rate into a normalised [0, 1] stress signal for Soma.
        # After pruning, rolling_burn_rate_usd_per_hour reflects the current window.
        if self._soma is not None:
            burn_rate = self.rolling_burn_rate_usd_per_hour
            stress = min(1.0, burn_rate / max(self._budget_usd_per_hour, 1e-9))
            self._soma.inject_external_stress(stress)

        # Report actual spend to Oikos — the single source of financial truth.
        if self._oikos is not None:
            self._oikos.report_compute_spend(
                workload_id=workload_id,
                provider_id=provider_id,
                actual_cost_usd=actual_cost_usd,
                baseline_cost_usd=baseline_cost_usd or actual_cost_usd * self._baseline_premium,
                cost_breakdown=cost_breakdown or {},
                duration_s=duration_s,
                is_pre_warm=False,
            )

        # Emit accounting metrics
        self._emit_metrics(record)

        self._log.info(
            "execution_recorded",
            workload_id=workload_id,
            provider_id=provider_id,
            actual_usd=round(actual_cost_usd, 6),
            baseline_usd=round(baseline_cost_usd, 6),
            savings_usd=round(record.savings_usd, 6),
            savings_ratio=round(record.savings_ratio, 4),
            total_records=len(self._records),
        )

        return record

    def get_savings_report(self, period_label: str = "all_time") -> SavingsReport:
        """
        Produce an aggregated savings report.

        Summarises total spend, baseline cost, savings, and per-provider
        breakdowns across all recorded executions.
        """
        if not self._records:
            return SavingsReport(period_label=period_label)

        total_actual = sum(r.actual_cost_usd for r in self._records)
        total_baseline = sum(r.baseline_cost_usd for r in self._records)
        total_savings = sum(r.savings_usd for r in self._records)
        count = len(self._records)

        savings_ratio = total_savings / total_baseline if total_baseline > 0 else 0.0

        # Per-provider aggregation
        provider_map: dict[str, list[CostRecord]] = {}
        for r in self._records:
            provider_map.setdefault(r.provider_id, []).append(r)

        top_providers = []
        for pid, records in sorted(provider_map.items(), key=lambda x: -len(x[1])):
            p_actual = sum(r.actual_cost_usd for r in records)
            p_baseline = sum(r.baseline_cost_usd for r in records)
            p_savings = sum(r.savings_usd for r in records)
            top_providers.append(
                ProviderSpendSummary(
                    provider_id=pid,
                    total_actual_usd=round(p_actual, 6),
                    total_baseline_usd=round(p_baseline, 6),
                    total_savings_usd=round(p_savings, 6),
                    workload_count=len(records),
                )
            )

        report = SavingsReport(
            period_label=period_label,
            record_count=count,
            total_actual_usd=round(total_actual, 6),
            total_baseline_usd=round(total_baseline, 6),
            total_savings_usd=round(total_savings, 6),
            savings_ratio=round(savings_ratio, 4),
            avg_cost_per_workload_usd=round(total_actual / count, 6),
            avg_savings_per_workload_usd=round(total_savings / count, 6),
            top_providers=top_providers,
        )

        self._log.info(
            "savings_report_generated",
            period=period_label,
            records=count,
            total_actual_usd=round(total_actual, 6),
            total_savings_usd=round(total_savings, 6),
            savings_ratio=round(savings_ratio, 4),
        )

        return report

    @property
    def rolling_burn_rate_usd_per_hour(self) -> float:
        """
        Current burn rate in USD/hour based on the rolling window.

        Used by Soma to predict energy exhaustion time.
        """
        now = time.monotonic()
        self._prune_rolling_window(now)
        if not self._rolling_costs:
            return 0.0

        window_spend = sum(cost for _, cost in self._rolling_costs)
        # Scale to hourly rate
        return window_spend * (3600.0 / self._rolling_window_s)

    @property
    def total_spend_usd(self) -> float:
        return sum(r.actual_cost_usd for r in self._records)

    @property
    def total_savings_usd(self) -> float:
        return sum(r.savings_usd for r in self._records)

    @property
    def records(self) -> Sequence[CostRecord]:
        """All cost records (read-only)."""
        return self._records

    def _prune_rolling_window(self, now: float) -> None:
        """Remove entries older than the rolling window."""
        cutoff = now - self._rolling_window_s
        while self._rolling_costs and self._rolling_costs[0][0] < cutoff:
            self._rolling_costs.popleft()

    def _emit_metrics(self, record: CostRecord) -> None:
        """Emit structured accounting metrics for the telemetry pipeline."""
        total_actual = sum(r.actual_cost_usd for r in self._records)
        total_baseline = sum(r.baseline_cost_usd for r in self._records)
        total_savings = sum(r.savings_usd for r in self._records)
        savings_ratio = total_savings / total_baseline if total_baseline > 0 else 0.0

        self._metrics_log.info(
            "metric",
            metric="sacm.accounting.recorded",
            value=len(self._records),
            workload_id=record.workload_id,
        )
        self._metrics_log.info(
            "metric",
            metric="sacm.accounting.rolling_spend",
            value=round(self.rolling_burn_rate_usd_per_hour, 6),
        )
        self._metrics_log.info(
            "metric",
            metric="sacm.accounting.savings_ratio",
            value=round(savings_ratio, 4),
        )
        self._metrics_log.info(
            "metric",
            metric="sacm.cost.total_usd",
            value=round(total_actual, 6),
        )
        self._metrics_log.info(
            "metric",
            metric="sacm.cost.savings_usd",
            value=round(total_savings, 6),
        )
