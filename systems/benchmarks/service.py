"""
EcodiaOS — Benchmark Service

Collects five KPIs from live system health/stats endpoints on a configurable
interval, persists snapshots to TimescaleDB, and fires BENCHMARK_REGRESSION
Synapse events when any metric regresses > threshold% from its rolling average.

Design
──────
• Pulls data from health() / stats endpoints — no direct imports of system internals.
• Each KPI collector is an isolated coroutine that handles its own errors.
• Rolling average is computed from the last N snapshots stored in TimescaleDB.
• Alert fires once per regression, then re-arms when the metric recovers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from systems.benchmarks.types import (
    BenchmarkSnapshot,
    BenchmarkTrend,
    MetricRegression,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from clients.timescaledb import TimescaleDBClient
    from config import BenchmarkConfig

logger = structlog.get_logger("systems.benchmarks")

# All KPI names (original five + two from Telos/Logos)
_KPI_NAMES: tuple[str, ...] = (
    "decision_quality",
    "llm_dependency",
    "economic_ratio",
    "learning_rate",
    "mutation_success_rate",
    "effective_intelligence_ratio",
    "compression_ratio",
)


class BenchmarkService:
    """
    Quantitative measurement layer.

    Wiring
    ──────
      service = BenchmarkService(config=config.benchmarks, tsdb=tsdb_client)
      service.set_nova(nova)
      service.set_evo(evo)
      service.set_oikos(oikos)
      service.set_simula(simula)
      service.set_telos(telos)
      service.set_logos(logos)
      service.set_event_bus(synapse.event_bus)
      await service.initialize()
      # Runs on its own async loop until shutdown()
    """

    system_id: str = "benchmarks"

    def __init__(
        self,
        config: BenchmarkConfig,
        tsdb: TimescaleDBClient,
        instance_id: str = "eos-default",
    ) -> None:
        self._config = config
        self._tsdb = tsdb
        self._instance_id = instance_id
        self._logger = logger.bind(system="benchmarks")

        # Injected dependencies (set via setters after construction)
        self._nova: Any | None = None
        self._evo: Any | None = None
        self._oikos: Any | None = None
        self._simula: Any | None = None
        self._telos: Any | None = None
        self._logos: Any | None = None
        self._event_bus: Any | None = None

        # Runtime state
        self._initialized: bool = False
        self._task: asyncio.Task[None] | None = None

        # In-memory cache of the most recently collected snapshot (for sync stats access)
        self._last_snapshot: BenchmarkSnapshot | None = None

        # Track which metrics are currently in regression so we don't repeat-alert
        self._regressed: set[str] = set()

        # Counters for observability
        self._total_runs: int = 0
        self._total_regressions_fired: int = 0

    # ─── Dependency injection ─────────────────────────────────────────

    def set_nova(self, nova: Any) -> None:
        self._nova = nova

    def set_evo(self, evo: Any) -> None:
        self._evo = evo

    def set_oikos(self, oikos: Any) -> None:
        self._oikos = oikos

    def set_simula(self, simula: Any) -> None:
        self._simula = simula

    def set_telos(self, telos: Any) -> None:
        self._telos = telos

    def set_logos(self, logos: Any) -> None:
        self._logos = logos

    def set_event_bus(self, bus: Any) -> None:
        self._event_bus = bus

    # ─── Lifecycle ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Ensure the benchmarks table exists and start the collection loop."""
        await self._ensure_schema()
        self._initialized = True
        self._task = asyncio.create_task(self._run_loop(), name="benchmarks_loop")
        self._logger.info(
            "benchmarks_started",
            interval_s=self._config.interval_s,
            rolling_window=self._config.rolling_window_snapshots,
            regression_threshold_pct=self._config.regression_threshold_pct,
        )

    async def shutdown(self) -> None:
        """Cancel the background loop gracefully."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._logger.info("benchmarks_stopped")

    # ─── Collection loop ──────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Wait interval_s, collect, store, check regressions. Forever."""
        # Small startup delay so all systems are fully ready
        await asyncio.sleep(10.0)
        while True:
            try:
                await asyncio.sleep(self._config.interval_s)
                snapshot = await self._collect()
                self._last_snapshot = snapshot
                await self._persist(snapshot)
                await self._check_regressions(snapshot)
                self._total_runs += 1
                self._logger.info(
                    "benchmark_run_completed",
                    run=self._total_runs,
                    decision_quality=snapshot.decision_quality,
                    llm_dependency=snapshot.llm_dependency,
                    economic_ratio=snapshot.economic_ratio,
                    learning_rate=snapshot.learning_rate,
                    mutation_success_rate=snapshot.mutation_success_rate,
                    effective_intelligence_ratio=snapshot.effective_intelligence_ratio,
                    compression_ratio=snapshot.compression_ratio,
                    errors=list(snapshot.errors.keys()),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.error("benchmark_loop_error", error=str(exc))

    # ─── KPI collection ──────────────────────────────────────────────

    async def _collect(self) -> BenchmarkSnapshot:
        """Collect all seven KPIs concurrently."""
        results = await asyncio.gather(
            self._collect_decision_quality(),
            self._collect_llm_dependency(),
            self._collect_economic_ratio(),
            self._collect_learning_rate(),
            self._collect_mutation_success_rate(),
            self._collect_effective_intelligence_ratio(),
            self._collect_compression_ratio(),
            return_exceptions=True,
        )

        errors: dict[str, str] = {}
        raw: dict[str, Any] = {}

        def _extract(idx: int, name: str) -> float | None:
            r = results[idx]
            if isinstance(r, Exception):
                errors[name] = str(r)
                return None
            value, raw_data = r  # type: ignore[misc]
            raw[name] = raw_data
            return value

        return BenchmarkSnapshot(
            time=utc_now(),
            instance_id=self._instance_id,
            decision_quality=_extract(0, "decision_quality"),
            llm_dependency=_extract(1, "llm_dependency"),
            economic_ratio=_extract(2, "economic_ratio"),
            learning_rate=_extract(3, "learning_rate"),
            mutation_success_rate=_extract(4, "mutation_success_rate"),
            effective_intelligence_ratio=_extract(5, "effective_intelligence_ratio"),
            compression_ratio=_extract(6, "compression_ratio"),
            errors=errors,
            raw=raw,
        )

    async def _collect_decision_quality(self) -> tuple[float | None, dict[str, Any]]:
        """
        % of Nova outcomes rated positive.

        Nova.health() exposes:
          outcomes_success: int
          outcomes_failure: int
        """
        if self._nova is None:
            return None, {}
        health = await self._nova.health()
        success = int(health.get("outcomes_success", 0))
        failure = int(health.get("outcomes_failure", 0))
        total = success + failure
        raw = {"outcomes_success": success, "outcomes_failure": failure}
        if total == 0:
            return None, raw
        return round(success / total, 4), raw

    async def _collect_llm_dependency(self) -> tuple[float | None, dict[str, Any]]:
        """
        % of decisions that required an LLM call (slow_path / total).

        Nova.health() exposes:
          fast_path_decisions: int
          slow_path_decisions: int
          do_nothing_decisions: int
        """
        if self._nova is None:
            return None, {}
        health = await self._nova.health()
        fast = int(health.get("fast_path_decisions", 0))
        slow = int(health.get("slow_path_decisions", 0))
        do_nothing = int(health.get("do_nothing_decisions", 0))
        total = fast + slow + do_nothing
        raw = {"fast": fast, "slow": slow, "do_nothing": do_nothing, "total": total}
        if total == 0:
            return None, raw
        return round(slow / total, 4), raw

    async def _collect_economic_ratio(self) -> tuple[float | None, dict[str, Any]]:
        """
        Oikos income / expenses = revenue_7d / costs_7d.

        Oikos.stats exposes revenue_7d and costs_7d as Decimal strings.
        """
        if self._oikos is None:
            return None, {}
        stats = self._oikos.stats  # sync property
        rev_raw = stats.get("revenue_7d", "0")
        cost_raw = stats.get("costs_7d", "0")
        raw = {"revenue_7d": rev_raw, "costs_7d": cost_raw}
        try:
            revenue = Decimal(str(rev_raw))
            costs = Decimal(str(cost_raw))
        except InvalidOperation:
            raise ValueError(f"Oikos returned non-numeric values: {raw}")
        if costs == Decimal("0"):
            return None, raw  # No expenses yet — ratio undefined
        return float(round(revenue / costs, 4)), raw

    async def _collect_learning_rate(self) -> tuple[float | None, dict[str, Any]]:
        """
        Number of hypotheses newly confirmed (supported) in this window.

        Evo.stats exposes hypothesis.supported (cumulative).
        We compare to the value stored in the previous snapshot.
        """
        if self._evo is None:
            return None, {}
        stats = self._evo.stats
        hyp = stats.get("hypothesis", {})
        supported_total = int(hyp.get("supported", 0))
        raw = {"supported_total": supported_total}

        # Read previous cumulative value from TimescaleDB
        prev = await self._latest_raw_value("learning_rate_cumulative")
        if prev is None:
            # First run — store baseline, return 0 (no confirmed in this window)
            await self._store_auxiliary("learning_rate_cumulative", float(supported_total))
            return 0.0, raw

        delta = max(0, supported_total - int(prev))
        await self._store_auxiliary("learning_rate_cumulative", float(supported_total))
        raw["delta"] = delta
        return float(delta), raw

    async def _collect_mutation_success_rate(self) -> tuple[float | None, dict[str, Any]]:
        """
        Simula proposals_approved / proposals_received.

        Simula.stats exposes proposals_approved and proposals_received.
        """
        if self._simula is None:
            return None, {}
        stats = self._simula.stats
        approved = int(stats.get("proposals_approved", 0))
        received = int(stats.get("proposals_received", 0))
        raw = {"proposals_approved": approved, "proposals_received": received}
        if received == 0:
            return None, raw
        return round(approved / received, 4), raw

    async def _collect_effective_intelligence_ratio(self) -> tuple[float | None, dict[str, Any]]:
        """
        Telos effective_I — nominal_I scaled by all four drive multipliers.

        Telos.health() exposes last_effective_I directly.
        """
        if self._telos is None:
            return None, {}
        health = await self._telos.health()
        value = health.get("last_effective_I")
        raw: dict[str, Any] = {
            "last_effective_I": value,
            "alignment_gap": health.get("last_alignment_gap"),
        }
        if value is None:
            return None, raw
        return round(float(value), 4), raw

    async def _collect_compression_ratio(self) -> tuple[float | None, dict[str, Any]]:
        """
        Logos intelligence ratio I = K(reality_modeled) / K(model).

        Logos.health() exposes intelligence_ratio directly.
        """
        if self._logos is None:
            return None, {}
        health = await self._logos.health()
        value = health.get("intelligence_ratio")
        raw: dict[str, Any] = {
            "intelligence_ratio": value,
            "cognitive_pressure": health.get("cognitive_pressure"),
            "schwarzschild_met": health.get("schwarzschild_met"),
        }
        if value is None:
            return None, raw
        return round(float(value), 4), raw

    # ─── Persistence ──────────────────────────────────────────────────

    async def _persist(self, snapshot: BenchmarkSnapshot) -> None:
        """Write snapshot to TimescaleDB benchmark_snapshots table."""
        async with self._tsdb.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO benchmark_snapshots (
                    time, instance_id,
                    decision_quality, llm_dependency, economic_ratio,
                    learning_rate, mutation_success_rate,
                    effective_intelligence_ratio, compression_ratio,
                    errors, raw
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                """,
                snapshot.time,
                snapshot.instance_id,
                snapshot.decision_quality,
                snapshot.llm_dependency,
                snapshot.economic_ratio,
                snapshot.learning_rate,
                snapshot.mutation_success_rate,
                snapshot.effective_intelligence_ratio,
                snapshot.compression_ratio,
                json.dumps(snapshot.errors),
                json.dumps(snapshot.raw),
            )

    async def _ensure_schema(self) -> None:
        """Create benchmark_snapshots table (and hypertable if TimescaleDB available)."""
        async with self._tsdb.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_snapshots (
                    time                          TIMESTAMPTZ NOT NULL,
                    instance_id                   TEXT NOT NULL,
                    decision_quality              DOUBLE PRECISION,
                    llm_dependency                DOUBLE PRECISION,
                    economic_ratio                DOUBLE PRECISION,
                    learning_rate                 DOUBLE PRECISION,
                    mutation_success_rate         DOUBLE PRECISION,
                    effective_intelligence_ratio  DOUBLE PRECISION,
                    compression_ratio             DOUBLE PRECISION,
                    errors                        JSONB DEFAULT '{}',
                    raw                           JSONB DEFAULT '{}'
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bm_instance_time
                    ON benchmark_snapshots (instance_id, time DESC)
            """)
            # Auxiliary table for cross-run state (e.g. cumulative learning_rate baseline)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_aux (
                    key        TEXT PRIMARY KEY,
                    value      DOUBLE PRECISION NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            # Promote to hypertable if TimescaleDB extension is available
            has_tsdb = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
            )
            if has_tsdb:
                with contextlib.suppress(Exception):
                    await conn.execute(
                        "SELECT create_hypertable('benchmark_snapshots', 'time', "
                        "if_not_exists => TRUE)"
                    )

    # ─── Rolling average & regression detection ───────────────────────

    async def _rolling_averages(self) -> dict[str, float | None]:
        """
        Compute rolling average of each KPI over the last N snapshots
        stored for this instance.
        """
        n = self._config.rolling_window_snapshots
        averages: dict[str, float | None] = {}
        async with self._tsdb.pool.acquire() as conn:
            for kpi in _KPI_NAMES:
                # Fetch last N non-null values
                rows = await conn.fetch(
                    f"""
                    SELECT {kpi} FROM benchmark_snapshots
                    WHERE instance_id = $1
                      AND {kpi} IS NOT NULL
                    ORDER BY time DESC
                    LIMIT $2
                    """,
                    self._instance_id,
                    n,
                )
                values = [r[kpi] for r in rows]
                averages[kpi] = sum(values) / len(values) if values else None
        return averages

    async def _check_regressions(self, snapshot: BenchmarkSnapshot) -> None:
        """
        For each KPI: if current value is more than threshold% below rolling avg,
        fire a BENCHMARK_REGRESSION Synapse event. Re-arm when it recovers.
        """
        if self._event_bus is None:
            return

        rolling = await self._rolling_averages()
        threshold = self._config.regression_threshold_pct / 100.0

        snapshot_vals: dict[str, float | None] = {
            "decision_quality": snapshot.decision_quality,
            "llm_dependency": snapshot.llm_dependency,
            "economic_ratio": snapshot.economic_ratio,
            "learning_rate": snapshot.learning_rate,
            "mutation_success_rate": snapshot.mutation_success_rate,
            "effective_intelligence_ratio": snapshot.effective_intelligence_ratio,
            "compression_ratio": snapshot.compression_ratio,
        }

        for kpi, current in snapshot_vals.items():
            avg = rolling.get(kpi)

            if current is None or avg is None or avg == 0.0:
                continue

            # For llm_dependency: higher is WORSE (more LLM calls needed),
            # so a regression means the value increased.
            if kpi == "llm_dependency":
                # Regressed if current > avg + threshold * avg
                regression_pct = (current - avg) / avg
                is_regressed = regression_pct > threshold
            else:
                # For all other KPIs: higher is better.
                # Regressed if current < avg - threshold * avg
                regression_pct = (avg - current) / avg
                is_regressed = regression_pct > threshold

            if is_regressed and kpi not in self._regressed:
                self._regressed.add(kpi)
                self._total_regressions_fired += 1
                regression = MetricRegression(
                    metric=kpi,
                    current_value=current,
                    rolling_avg=avg,
                    regression_pct=round(regression_pct * 100, 2),
                    threshold_pct=self._config.regression_threshold_pct,
                )
                await self._fire_regression_event(regression)
            elif not is_regressed and kpi in self._regressed:
                # Metric has recovered — re-arm
                self._regressed.discard(kpi)
                self._logger.info(
                    "benchmark_metric_recovered",
                    metric=kpi,
                    current=current,
                    rolling_avg=avg,
                )

    async def _fire_regression_event(self, regression: MetricRegression) -> None:
        """Emit a BENCHMARK_REGRESSION event via Synapse event bus."""
        event = SynapseEvent(
            event_type=SynapseEventType.BENCHMARK_REGRESSION,
            source_system=self.system_id,
            data={
                "metric": regression.metric,
                "current_value": regression.current_value,
                "rolling_avg": regression.rolling_avg,
                "regression_pct": regression.regression_pct,
                "threshold_pct": regression.threshold_pct,
                "instance_id": self._instance_id,
            },
        )
        await self._event_bus.emit(event)
        self._logger.warning(
            "benchmark_regression_detected",
            metric=regression.metric,
            current=regression.current_value,
            rolling_avg=regression.rolling_avg,
            regression_pct=regression.regression_pct,
        )

    # ─── Auxiliary key-value store ────────────────────────────────────

    async def _store_auxiliary(self, key: str, value: float) -> None:
        """Upsert a single auxiliary float value keyed by name."""
        async with self._tsdb.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO benchmark_aux (key, value, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        updated_at = EXCLUDED.updated_at
                """,
                key,
                value,
            )

    async def _latest_raw_value(self, key: str) -> float | None:
        """Retrieve a single auxiliary float value by key."""
        async with self._tsdb.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM benchmark_aux WHERE key = $1", key
            )
            return float(row["value"]) if row else None

    # ─── Query interface (for the dashboard router) ───────────────────

    async def latest_snapshot(self) -> BenchmarkSnapshot | None:
        """Return the most recent benchmark snapshot."""
        async with self._tsdb.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM benchmark_snapshots
                WHERE instance_id = $1
                ORDER BY time DESC
                LIMIT 1
                """,
                self._instance_id,
            )
        if row is None:
            return None
        return BenchmarkSnapshot(
            time=row["time"],
            instance_id=row["instance_id"],
            decision_quality=row["decision_quality"],
            llm_dependency=row["llm_dependency"],
            economic_ratio=row["economic_ratio"],
            learning_rate=row["learning_rate"],
            mutation_success_rate=row["mutation_success_rate"],
            effective_intelligence_ratio=row["effective_intelligence_ratio"],
            compression_ratio=row["compression_ratio"],
            errors=json.loads(row["errors"] or "{}"),
            raw=json.loads(row["raw"] or "{}"),
        )

    async def trend(
        self,
        metric: str,
        since: datetime | None = None,
        limit: int = 50,
    ) -> BenchmarkTrend:
        """
        Return time-series data for a single KPI.

        Parameters
        ──────────
        metric  — one of the five KPI names
        since   — optional start time (defaults to 7 days ago)
        limit   — max number of points (newest first)
        """
        if metric not in _KPI_NAMES:
            raise ValueError(f"Unknown benchmark metric: {metric!r}. Valid: {_KPI_NAMES}")

        if since is None:
            since = utc_now() - timedelta(days=7)

        async with self._tsdb.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT time, {metric} AS value
                FROM benchmark_snapshots
                WHERE instance_id = $1
                  AND time >= $2
                ORDER BY time DESC
                LIMIT $3
                """,
                self._instance_id,
                since,
                limit,
            )

        points = [
            {"time": r["time"].isoformat(), "value": r["value"]}
            for r in rows
        ]

        non_null = [p["value"] for p in points if p["value"] is not None]
        rolling_avg = sum(non_null) / len(non_null) if non_null else None
        latest = points[0]["value"] if points else None

        return BenchmarkTrend(
            metric=metric,
            points=list(reversed(points)),  # chronological order for charts
            rolling_avg=rolling_avg,
            latest=latest,
        )

    async def all_trends(self, since: datetime | None = None) -> dict[str, BenchmarkTrend]:
        """Return trend data for all five KPIs."""
        trends = await asyncio.gather(
            *[self.trend(kpi, since=since) for kpi in _KPI_NAMES],
            return_exceptions=True,
        )
        result: dict[str, BenchmarkTrend] = {}
        for kpi, t in zip(_KPI_NAMES, trends, strict=False):
            if isinstance(t, Exception):
                self._logger.error("benchmark_trend_error", metric=kpi, error=str(t))
            else:
                result[kpi] = t  # type: ignore[assignment]
        return result

    # ─── Health check (Synapse protocol) ──────────────────────────────

    async def health(self) -> dict[str, Any]:
        latest = await self.latest_snapshot()
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "total_runs": self._total_runs,
            "total_regressions_fired": self._total_regressions_fired,
            "currently_regressed": sorted(self._regressed),
            "latest_snapshot_time": latest.time.isoformat() if latest else None,
            "interval_s": self._config.interval_s,
            "rolling_window": self._config.rolling_window_snapshots,
        }

    @property
    def stats(self) -> dict[str, Any]:
        snap = self._last_snapshot
        return {
            "initialized": self._initialized,
            "total_runs": self._total_runs,
            "total_regressions_fired": self._total_regressions_fired,
            "active_regressions": sorted(self._regressed),
            "interval_s": self._config.interval_s,
            # Current KPI values from last collection cycle
            "decision_quality": snap.decision_quality if snap else None,
            "llm_dependency": snap.llm_dependency if snap else None,
            "economic_ratio": snap.economic_ratio if snap else None,
            "learning_rate": snap.learning_rate if snap else None,
            "mutation_success_rate": snap.mutation_success_rate if snap else None,
            "effective_intelligence_ratio": snap.effective_intelligence_ratio if snap else None,
            "compression_ratio": snap.compression_ratio if snap else None,
        }
