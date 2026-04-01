"""
EcodiaOS - SACM Cost Accounting

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
  sacm.accounting.recorded         - Number of cost records logged
  sacm.accounting.rolling_spend    - Rolling spend window (USD)
  sacm.accounting.savings_ratio    - Cumulative savings / baseline ratio
"""

from __future__ import annotations

import asyncio
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
        budget_usd_per_hour: float = 10.0,
        baseline_premium: float = _DEFAULT_BASELINE_PREMIUM,
        rolling_window_s: float = 3600.0,
    ) -> None:
        # oikos is OikosService at runtime; typed as Any to avoid circular import
        self._oikos: Any | None = None
        # Synapse event bus - wired via set_synapse()
        self._synapse: Any | None = None
        # Neo4j driver - wired via set_neo4j() for EconomicEvent audit trail
        self._neo4j: Any | None = None
        self._baseline_premium = baseline_premium
        self._rolling_window_s = rolling_window_s
        self._budget_usd_per_hour = budget_usd_per_hour

        self._records: list[CostRecord] = []
        # Rolling window for burn-rate calculation
        self._rolling_costs: deque[tuple[float, float]] = deque()  # (timestamp, cost_usd)

        self._log = logger.bind(component="sacm.accounting")
        self._metrics_log = structlog.get_logger("systems.sacm.metrics")

    def set_synapse(self, synapse: Any) -> None:
        """Wire Synapse so stress signals go via event bus instead of direct Soma calls."""
        self._synapse = synapse
        self._log.info("synapse_wired_to_sacm_accounting")

    def wire_oikos(self, oikos: OikosService) -> None:
        """Attach an OikosService so every recorded spend is reported to the ledger."""
        self._oikos = oikos

    def set_neo4j(self, neo4j_driver: Any) -> None:
        """
        Wire a Neo4j async driver so each CostRecord is persisted as an
        (:EconomicEvent) node - immutable audit trail per Spec 27 §10.

        The node is written fire-and-forget; accounting is non-blocking.
        """
        self._neo4j = neo4j_driver
        self._log.info("neo4j_wired_to_sacm_accounting")

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
        estimated_cost_usd: float | None = None,
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

        # Persist to Neo4j as an immutable (:EconomicEvent) node (fire-and-forget)
        if self._neo4j is not None:
            asyncio.create_task(
                self._write_economic_event_neo4j(record),
                name=f"sacm_neo4j_cost_{record.workload_id[:8]}",
            )

        # Update rolling window
        now = time.monotonic()
        self._rolling_costs.append((now, actual_cost_usd))
        self._prune_rolling_window(now)

        # Emit compute stress via Synapse so Soma can feel the burn rate
        # without a direct cross-system import.
        burn_rate = self.rolling_burn_rate_usd_per_hour
        stress = min(1.0, burn_rate / max(self._budget_usd_per_hour, 1e-9))
        if self._synapse is not None and stress > 0:
            self._emit_stress(workload_id, burn_rate, stress)

        # Report actual spend to Oikos - the single source of financial truth.
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

        # Emit provider performance signal to Evo so it can update substrate preference
        # hypotheses. Each verified/accepted execution confirms or refutes hypotheses
        # about the provider's reliability and cost accuracy, enabling Thompson sampling
        # to learn which substrates perform best for which workload types.
        if self._synapse is not None:
            self._emit_provider_performance(
                workload_id=workload_id,
                provider_id=provider_id,
                actual_cost_usd=actual_cost_usd,
                estimated_cost_usd=estimated_cost_usd,
                baseline_cost_usd=baseline_cost_usd,
                duration_s=duration_s,
                verified=verified,
                accepted=accepted,
            )

        # Emit Fovea prediction error if actual cost significantly exceeded estimate.
        # This registers the cost model violation as a prediction error in the Global
        # Workspace so Fovea can tighten precision weighting for future SACM decisions.
        if self._synapse is not None and estimated_cost_usd is not None and estimated_cost_usd > 0:
            ratio = actual_cost_usd / estimated_cost_usd
            if ratio > 1.5:  # ≥50% cost surprise
                self._emit_cost_prediction_error(
                    workload_id=workload_id,
                    provider_id=provider_id,
                    estimated_cost_usd=estimated_cost_usd,
                    actual_cost_usd=actual_cost_usd,
                    surprise_ratio=ratio,
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

    def _emit_stress(
        self,
        workload_id: str,
        burn_rate: float,
        stress: float,
    ) -> None:
        """Fire-and-forget SACM_COMPUTE_STRESS onto Synapse for Soma consumption."""
        if self._synapse is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        asyncio.create_task(
            self._synapse.event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.SACM_COMPUTE_STRESS,
                    source_system="sacm",
                    data={
                        "burn_rate_usd_per_hour": round(burn_rate, 6),
                        "budget_usd_per_hour": self._budget_usd_per_hour,
                        "stress_scalar": round(stress, 4),
                        "workload_id": workload_id,
                    },
                )
            ),
            name=f"sacm_stress_{workload_id[:8]}",
        )

    def _emit_cost_prediction_error(
        self,
        workload_id: str,
        provider_id: str,
        estimated_cost_usd: float,
        actual_cost_usd: float,
        surprise_ratio: float,
    ) -> None:
        """
        Emit FOVEA_INTERNAL_PREDICTION_ERROR when actual compute cost significantly
        exceeds the estimate used for allocation.

        This is the SG5 Fovea attention signal: cost surprises register as prediction
        errors so Fovea can modulate precision weighting on future SACM cost estimates.
        The 'economic' dimension carries the raw surprise; magnitude is the ratio - 1.0.
        """
        if self._synapse is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        magnitude = round(surprise_ratio - 1.0, 4)  # e.g. 0.8 = 80% over estimate

        asyncio.create_task(
            self._synapse.event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.FOVEA_INTERNAL_PREDICTION_ERROR,
                    source_system="sacm",
                    data={
                        "domain": "compute_cost",
                        "workload_id": workload_id,
                        "provider_id": provider_id,
                        "estimated_cost_usd": round(estimated_cost_usd, 6),
                        "actual_cost_usd": round(actual_cost_usd, 6),
                        "surprise_ratio": round(surprise_ratio, 4),
                        # PredictionError.economic dimension - magnitude of surprise
                        "prediction_error": {
                            "economic": magnitude,
                            "source": "sacm.cost_accounting",
                        },
                        # Salience hint: larger cost overruns warrant more attention
                        "salience_hint": min(1.0, magnitude / 2.0),
                    },
                )
            ),
            name=f"sacm_cost_pe_{workload_id[:8]}",
        )

        self._log.warning(
            "cost_prediction_error_emitted",
            workload_id=workload_id,
            provider_id=provider_id,
            estimated_usd=round(estimated_cost_usd, 6),
            actual_usd=round(actual_cost_usd, 6),
            surprise_ratio=round(surprise_ratio, 4),
        )

    async def _write_economic_event_neo4j(self, record: CostRecord) -> None:
        """
        Persist a CostRecord as an (:EconomicEvent) node in Neo4j.

        Node properties follow the EcodiaOS audit trail convention:
          - event_time: ISO8601 wall-clock time of the record (bi-temporal)
          - ingestion_time: time.monotonic() - when it entered the system
          - All cost fields for traceability

        Uses MERGE on record.id so re-runs are idempotent.
        """
        try:
            async with self._neo4j.session() as session:
                await session.run(
                    """
                    MERGE (e:EconomicEvent {id: $id})
                    ON CREATE SET
                        e.event_type      = 'sacm_cost_record',
                        e.system_id       = 'sacm',
                        e.workload_id     = $workload_id,
                        e.provider_id     = $provider_id,
                        e.offer_id        = $offer_id,
                        e.actual_cost_usd = $actual_cost_usd,
                        e.baseline_cost_usd = $baseline_cost_usd,
                        e.savings_usd     = $savings_usd,
                        e.savings_ratio   = $savings_ratio,
                        e.duration_s      = $duration_s,
                        e.verified        = $verified,
                        e.accepted        = $accepted,
                        e.event_time      = $event_time,
                        e.ingestion_time  = datetime()
                    """,
                    id=record.id,
                    workload_id=record.workload_id,
                    provider_id=record.provider_id,
                    offer_id=record.offer_id,
                    actual_cost_usd=record.actual_cost_usd,
                    baseline_cost_usd=record.baseline_cost_usd,
                    savings_usd=record.savings_usd,
                    savings_ratio=record.savings_ratio,
                    duration_s=record.duration_s,
                    verified=record.verified,
                    accepted=record.accepted,
                    event_time=record.created_at.isoformat(),
                )
        except Exception as exc:
            self._log.warning(
                "neo4j_economic_event_write_failed",
                workload_id=record.workload_id,
                error=str(exc),
            )

    def _emit_provider_performance(
        self,
        workload_id: str,
        provider_id: str,
        actual_cost_usd: float,
        estimated_cost_usd: float | None,
        baseline_cost_usd: float,
        duration_s: float,
        verified: bool,
        accepted: bool,
    ) -> None:
        """
        Emit provider performance evidence to Evo so it can update substrate
        preference hypotheses via Thompson sampling.

        Verification success → EVO_HYPOTHESIS_CONFIRMED (provider_reliability hypothesis)
        Verification failure → EVO_HYPOTHESIS_REFUTED
        Cost accuracy evidence is attached as a continuous quality signal.

        Evo uses this to learn which substrates perform best per workload type,
        closing spec gap SG3: static fair-share allocation with no Evo-driven learning.
        """
        if self._synapse is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        # Hypothesis ID follows a stable naming convention so Evo can accumulate
        # evidence across executions without creating duplicate hypotheses.
        hypothesis_id = f"sacm.provider_reliability.{provider_id}"

        # Cost accuracy score: 1.0 = perfectly estimated, lower = more surprise
        cost_accuracy = 1.0
        if estimated_cost_usd is not None and estimated_cost_usd > 0:
            ratio = actual_cost_usd / estimated_cost_usd
            # Score decays from 1.0 as ratio deviates from 1.0 in either direction
            cost_accuracy = max(0.0, 1.0 - abs(ratio - 1.0))

        # Reliability score: verification pass + acceptance weighted
        reliability_score = (0.6 if verified else 0.0) + (0.4 if accepted else 0.0)

        event_type = (
            SynapseEventType.EVO_HYPOTHESIS_CONFIRMED
            if verified and accepted
            else SynapseEventType.EVO_HYPOTHESIS_REFUTED
        )

        asyncio.create_task(
            self._synapse.event_bus.emit(
                SynapseEvent(
                    event_type=event_type,
                    source_system="sacm",
                    data={
                        "hypothesis_id": hypothesis_id,
                        "domain": "compute_substrate",
                        "provider_id": provider_id,
                        "workload_id": workload_id,
                        # Evidence for Thompson sampling weight update
                        "evidence": {
                            "verified": verified,
                            "accepted": accepted,
                            "reliability_score": round(reliability_score, 4),
                            "cost_accuracy": round(cost_accuracy, 4),
                            "actual_cost_usd": round(actual_cost_usd, 6),
                            "baseline_cost_usd": round(baseline_cost_usd, 6),
                            "savings_ratio": round(
                                max(0.0, baseline_cost_usd - actual_cost_usd)
                                / max(baseline_cost_usd, 1e-9),
                                4,
                            ),
                            "duration_s": round(duration_s, 3),
                        },
                        # Composite quality score for Evo's outcome_quality field
                        "quality": round((reliability_score + cost_accuracy) / 2.0, 4),
                    },
                )
            ),
            name=f"sacm_provider_perf_{workload_id[:8]}",
        )
