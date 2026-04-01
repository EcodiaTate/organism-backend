"""
EcodiaOS - Telos: Protocol Adapters

Bridges between the actual Logos/Fovea service implementations and the
Telos protocol interfaces (LogosMetrics / FoveaMetrics).

These adapters exist because:
  - LogosService methods are synchronous; the protocol requires async.
  - FoveaService doesn't yet expose the exact methods Telos expects.
  - Neither service returns the precise Telos-typed value objects.

Each adapter wraps the real service and translates to protocol shape.

Additionally, the LogosMetricsAdapter now maintains its OWN time-series
of effective_I values (written to Neo4j) so that GrowthTopologyEngine
can compute dI/dt and d²I/dt² from real historical data.

The FoveaMetricsAdapter buffers high-error experiences from
FOVEA_PREDICTION_ERROR events so that CareTopologyEngine can detect
welfare prediction failures.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from systems.telos.types import (
    CompressionEvent,
    CompressionStats,
    HighErrorExperience,
    TimestampedValue,
)

if TYPE_CHECKING:
    from systems.fovea.service import FoveaService
    from systems.logos.service import LogosService

logger = structlog.get_logger()


# ─── Protocol for world model access ─────────────────────────────────
# Replaces direct getattr calls on LogosService.world_model.


@runtime_checkable
class WorldModelProtocol(Protocol):
    """Minimal protocol for reading world model schema coverage."""

    def get_generative_schemas(self) -> dict[str, Any]: ...


# ─── I-History Neo4j Storage ──────────────────────────────────────────


class IHistoryStore:
    """
    Telos-owned time-series of effective_I values.

    Maintains an in-memory ring buffer for fast access and writes batches
    to Neo4j for long-term persistence. Supports hourly rollups for trend
    analysis.
    """

    _MAX_IN_MEMORY = 1440  # 24h at 1-per-minute

    def __init__(self) -> None:
        self._buffer: deque[TimestampedValue] = deque(maxlen=self._MAX_IN_MEMORY)
        self._neo4j_driver: Any = None
        self._instance_id: str = ""
        self._log = logger.bind(component="telos.i_history")

    def set_neo4j(self, driver: Any, instance_id: str = "") -> None:
        self._neo4j_driver = driver
        self._instance_id = instance_id

    def record(
        self,
        *,
        nominal_I: float,
        effective_I: float,
        care_mult: float,
        coherence_bonus: float,
        honesty_coeff: float,
        growth_score: float,
    ) -> None:
        """Record a new I measurement in the in-memory buffer."""
        entry = TimestampedValue(value=effective_I)
        self._buffer.append(entry)

    async def persist_to_neo4j(
        self,
        *,
        nominal_I: float,
        effective_I: float,
        care_mult: float,
        coherence_bonus: float,
        honesty_coeff: float,
        growth_score: float,
    ) -> None:
        """Batch write to Neo4j - called once per Telos cycle."""
        if self._neo4j_driver is None:
            return
        try:
            query = """
            CREATE (ir:IntelligenceRatio {
                instance_id: $instance_id,
                nominal_I: $nominal_I,
                effective_I: $effective_I,
                care_mult: $care_mult,
                coherence_bonus: $coherence_bonus,
                honesty_coeff: $honesty_coeff,
                growth_score: $growth_score,
                timestamp: datetime($timestamp)
            })
            """
            now = datetime.now(UTC)
            params = {
                "instance_id": self._instance_id,
                "nominal_I": nominal_I,
                "effective_I": effective_I,
                "care_mult": care_mult,
                "coherence_bonus": coherence_bonus,
                "honesty_coeff": honesty_coeff,
                "growth_score": growth_score,
                "timestamp": now.isoformat(),
            }
            async with self._neo4j_driver.session() as session:
                await session.run(query, params)
            self._log.debug("i_history_persisted", effective_I=round(effective_I, 4))
        except Exception as exc:
            self._log.warning("i_history_persist_failed", error=str(exc))

    async def persist_hourly_rollup(self) -> None:
        """Aggregate hourly rollups for long-term trends."""
        if self._neo4j_driver is None or len(self._buffer) < 2:
            return
        now = datetime.now(UTC)
        hour_ago = now - timedelta(hours=1)
        recent = [e for e in self._buffer if e.timestamp >= hour_ago]
        if not recent:
            return
        avg_I = sum(e.value for e in recent) / len(recent)
        min_I = min(e.value for e in recent)
        max_I = max(e.value for e in recent)
        try:
            query = """
            MERGE (r:IntelligenceRatioHourly {
                instance_id: $instance_id,
                hour: datetime($hour)
            })
            SET r.avg_effective_I = $avg_I,
                r.min_effective_I = $min_I,
                r.max_effective_I = $max_I,
                r.sample_count = $count
            """
            hour_key = now.replace(minute=0, second=0, microsecond=0)
            async with self._neo4j_driver.session() as session:
                await session.run(query, {
                    "instance_id": self._instance_id,
                    "hour": hour_key.isoformat(),
                    "avg_I": avg_I,
                    "min_I": min_I,
                    "max_I": max_I,
                    "count": len(recent),
                })
        except Exception as exc:
            self._log.debug("hourly_rollup_failed", error=str(exc))

    def get_history(self, window_hours: float = 24.0) -> list[TimestampedValue]:
        """Return I measurements within the given time window."""
        if not self._buffer:
            return []
        cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
        return [e for e in self._buffer if e.timestamp >= cutoff]

    async def load_from_neo4j(self, window_hours: float = 24.0) -> list[TimestampedValue]:
        """Load historical I measurements from Neo4j on startup."""
        if self._neo4j_driver is None:
            return self.get_history(window_hours)
        try:
            query = """
            MATCH (ir:IntelligenceRatio)
            WHERE ir.instance_id = $instance_id
              AND ir.timestamp >= datetime($cutoff)
            RETURN ir.effective_I AS value, ir.timestamp AS ts
            ORDER BY ir.timestamp ASC
            """
            cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
            async with self._neo4j_driver.session() as session:
                result = await session.run(query, {
                    "instance_id": self._instance_id,
                    "cutoff": cutoff.isoformat(),
                })
                records = await result.data()
            entries = []
            for rec in records:
                ts = rec.get("ts")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                elif not isinstance(ts, datetime):
                    ts = datetime.now(UTC)
                entries.append(TimestampedValue(
                    value=float(rec.get("value", 0.0)),
                    timestamp=ts,
                ))
            # Merge with in-memory buffer (dedup by checking if we already have these)
            if entries:
                existing_ts = {e.timestamp for e in self._buffer}
                for entry in entries:
                    if entry.timestamp not in existing_ts:
                        self._buffer.append(entry)
            return self.get_history(window_hours)
        except Exception as exc:
            self._log.warning("neo4j_i_history_load_failed", error=str(exc))
            return self.get_history(window_hours)


# ─── Fovea Prediction Error Buffer ───────────────────────────────────


class FoveaPredictionErrorBuffer:
    """
    Bounded buffer of high-error experiences from FOVEA_PREDICTION_ERROR
    events. FIFO eviction at max 500 entries.
    """

    _MAX_ENTRIES = 500

    def __init__(self, salience_threshold: float = 0.7) -> None:
        self._buffer: deque[HighErrorExperience] = deque(maxlen=self._MAX_ENTRIES)
        self._salience_threshold = salience_threshold

    def ingest(self, event_data: dict[str, Any]) -> None:
        """Process a FOVEA_PREDICTION_ERROR event, buffer if above threshold."""
        salience = float(event_data.get("precision_weighted_salience", 0.0))
        if salience < self._salience_threshold:
            return

        # Determine domain from the error's dominant type or route
        domain = str(event_data.get("dominant_error_type", ""))
        routes = event_data.get("routes", [])
        if routes and isinstance(routes, list):
            domain = domain or str(routes[0]) if routes else domain

        # Compute aggregate error magnitude from the 6D prediction error
        error_fields = [
            "content_error", "temporal_error", "magnitude_error",
            "source_error", "category_error", "causal_error",
        ]
        errors = [float(event_data.get(f, 0.0)) for f in error_fields]
        prediction_error = max(errors) if errors else salience

        experience = HighErrorExperience(
            id=str(event_data.get("error_id", "")),
            domain=domain,
            prediction_error=prediction_error,
            was_novel_domain=False,
        )
        self._buffer.append(experience)

    def get_recent(self, window_hours: float = 24.0) -> list[HighErrorExperience]:
        """Return buffered experiences within the time window."""
        if not self._buffer:
            return []
        cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
        return [e for e in self._buffer if e.timestamp >= cutoff]

    @property
    def size(self) -> int:
        return len(self._buffer)


# ─── Welfare Keywords ─────────────────────────────────────────────────
# Shared with CareTopologyEngine - kept here to tag experiences at ingest.

_WELFARE_KEYWORDS = (
    "welfare", "care", "harm", "trust", "social", "relationship",
    "safety", "health", "wellbeing", "emotional", "interpersonal",
    "cooperation", "conflict", "consent", "community",
)


# ─── Adapters ─────────────────────────────────────────────────────────


class LogosMetricsAdapter:
    """
    Adapts LogosService to the LogosMetrics protocol required by Telos.

    Maintains its OWN time-series of effective_I values via IHistoryStore
    so that GrowthTopologyEngine can compute dI/dt and d²I/dt² from real
    historical data rather than a single-element list.
    """

    def __init__(self, logos: LogosService) -> None:
        self._logos = logos
        self._i_history = IHistoryStore()

    @property
    def i_history_store(self) -> IHistoryStore:
        """Expose the I-history store for the service to write to."""
        return self._i_history

    def record_measurement(
        self,
        *,
        nominal_I: float,
        effective_I: float,
        care_mult: float,
        coherence_bonus: float,
        honesty_coeff: float,
        growth_score: float,
    ) -> None:
        """Record an I measurement in the in-memory ring buffer."""
        self._i_history.record(
            nominal_I=nominal_I,
            effective_I=effective_I,
            care_mult=care_mult,
            coherence_bonus=coherence_bonus,
            honesty_coeff=honesty_coeff,
            growth_score=growth_score,
        )

    async def persist_measurement(
        self,
        *,
        nominal_I: float,
        effective_I: float,
        care_mult: float,
        coherence_bonus: float,
        honesty_coeff: float,
        growth_score: float,
    ) -> None:
        """Persist the measurement to Neo4j (async, 1 write per cycle)."""
        await self._i_history.persist_to_neo4j(
            nominal_I=nominal_I,
            effective_I=effective_I,
            care_mult=care_mult,
            coherence_bonus=coherence_bonus,
            honesty_coeff=honesty_coeff,
            growth_score=growth_score,
        )

    async def persist_hourly_rollup(self) -> None:
        """Delegate hourly rollup persistence to the I-history store."""
        await self._i_history.persist_hourly_rollup()

    def set_history_neo4j(self, driver: Any, instance_id: str = "") -> None:
        """Inject Neo4j driver into the I-history store."""
        self._i_history.set_neo4j(driver, instance_id)

    # ── Protocol methods (all async) ────────────────────────────────

    async def get_intelligence_ratio(self) -> float:
        return self._logos.get_intelligence_ratio()

    async def get_compression_stats(self) -> CompressionStats:
        raw: dict[str, Any] = self._logos.get_compression_stats()
        wm = getattr(self._logos, "world_model", None)
        domain_count = 0
        if wm is not None:
            schemas = wm.get_generative_schemas() if isinstance(wm, WorldModelProtocol) else getattr(wm, "generative_schemas", {})
            domain_count = len(schemas) if schemas else 0
        return CompressionStats(
            total_description_length=raw.get("world_model_complexity", 0.0),
            reality_covered=raw.get("world_model_coverage", 0.0),
            compression_ratio=max(raw.get("compression_efficiency", 1.0), 0.001),
            domain_count=domain_count,
        )

    async def get_I_history(
        self, window_hours: float = 24.0
    ) -> list[TimestampedValue]:
        """Return Telos-owned I time-series from the IHistoryStore."""
        history = self._i_history.get_history(window_hours)
        if not history:
            # Fallback: seed with current I if no history yet
            current = self._logos.get_intelligence_ratio()
            return [TimestampedValue(value=current)]
        return history

    async def get_domain_coverage_map(self) -> dict[str, float]:
        """Derive from world model generative schemas if available."""
        wm = getattr(self._logos, "world_model", None)
        if wm is None:
            return {}
        schemas = wm.get_generative_schemas() if isinstance(wm, WorldModelProtocol) else getattr(wm, "generative_schemas", {})
        if not schemas:
            return {}
        coverage: dict[str, float] = {}
        for key, schema in schemas.items():
            cov = getattr(schema, "coverage", None)
            if cov is not None:
                coverage[str(key)] = float(cov)
            else:
                coverage[str(key)] = 0.5
        return coverage

    async def get_recent_compression_events(
        self, window_hours: float = 24.0
    ) -> list[CompressionEvent]:
        """No historical compression events tracked yet - return empty."""
        return []


class FoveaMetricsAdapter:
    """
    Adapts FoveaService to the FoveaMetrics protocol required by Telos.

    Buffers high-error experiences from FOVEA_PREDICTION_ERROR events
    so that CareTopologyEngine can detect welfare prediction failures.
    """

    def __init__(
        self,
        fovea: FoveaService,
        salience_threshold: float = 0.7,
    ) -> None:
        self._fovea = fovea
        self._error_buffer = FoveaPredictionErrorBuffer(
            salience_threshold=salience_threshold,
        )

    @property
    def error_buffer(self) -> FoveaPredictionErrorBuffer:
        """Expose buffer for event subscription wiring."""
        return self._error_buffer

    def ingest_prediction_error(self, event_data: dict[str, Any]) -> None:
        """Ingest a FOVEA_PREDICTION_ERROR event payload into the buffer."""
        self._error_buffer.ingest(event_data)

    # ── Protocol methods (all async) ────────────────────────────────

    async def get_prediction_error_rate(self) -> float:
        """Compute from fovea health metrics: errors / total processed."""
        h = await self._fovea.health()
        processed = h.get("errors_processed", 0)
        internal = h.get("internal_errors_generated", 0)
        if processed <= 0:
            return 0.0
        return min(1.0, internal / max(processed, 1))

    async def get_error_distribution(self) -> dict[str, float]:
        """Derive from internal engine error-by-type counts."""
        h = await self._fovea.health()
        errors_by_type: dict[str, int] = h.get("internal_errors_by_type", {})
        total = sum(errors_by_type.values()) or 1
        return {k: v / total for k, v in errors_by_type.items()}

    async def get_prediction_success_rate(self) -> float:
        """Inverse of error rate - fraction of correct predictions."""
        error_rate = await self.get_prediction_error_rate()
        return 1.0 - error_rate

    async def get_recent_high_error_experiences(
        self, window_hours: float = 24.0
    ) -> list[HighErrorExperience]:
        """Return buffered high-error experiences from FOVEA_PREDICTION_ERROR events."""
        return self._error_buffer.get_recent(window_hours)

    async def get_confabulation_rate(self) -> float:
        """
        Confabulation rate from false alarms in weight learner.

        False alarms = predictions that triggered attention but weren't
        followed by actual prediction errors - a proxy for confabulation.
        """
        h = await self._fovea.health()
        false_alarms = h.get("false_alarms", 0)
        total = h.get("internal_predictions_made", 0) or 1
        return min(1.0, false_alarms / total)

    async def get_overclaiming_rate(self) -> float:
        """
        Overclaiming: fraction of domains where predictions are made
        but accuracy is unknown.  Approximated from weight learner state.
        """
        h = await self._fovea.health()
        weights: dict[str, float] = h.get("learned_weights", {})
        if not weights:
            return 0.0
        low_weight_count = sum(1 for w in weights.values() if w < 0.3)
        return min(1.0, low_weight_count / max(len(weights), 1))
