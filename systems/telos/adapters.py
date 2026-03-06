"""
EcodiaOS — Telos: Protocol Adapters

Bridges between the actual Logos/Fovea service implementations and the
Telos protocol interfaces (LogosMetrics / FoveaMetrics).

These adapters exist because:
  - LogosService methods are synchronous; the protocol requires async.
  - FoveaService doesn't yet expose the exact methods Telos expects.
  - Neither service returns the precise Telos-typed value objects.

Each adapter wraps the real service and translates to protocol shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


class LogosMetricsAdapter:
    """
    Adapts LogosService to the LogosMetrics protocol required by Telos.

    LogosService has sync methods and a different return-type signature
    for compression stats.  This adapter normalises both issues.
    """

    def __init__(self, logos: LogosService) -> None:
        self._logos = logos

    # ── Protocol methods (all async) ────────────────────────────────

    async def get_intelligence_ratio(self) -> float:
        return self._logos.get_intelligence_ratio()

    async def get_compression_stats(self) -> CompressionStats:
        raw: dict[str, Any] = self._logos.get_compression_stats()
        return CompressionStats(
            total_description_length=raw.get("world_model_complexity", 0.0),
            reality_covered=raw.get("world_model_coverage", 0.0),
            compression_ratio=max(raw.get("compression_efficiency", 1.0), 0.001),
            domain_count=len(self._logos.world_model.generative_schemas)
            if hasattr(self._logos, "world_model")
            else 0,
        )

    async def get_I_history(
        self, window_hours: float = 24.0
    ) -> list[TimestampedValue]:
        """Logos doesn't maintain a time-series — return singleton with current I."""
        current = self._logos.get_intelligence_ratio()
        return [TimestampedValue(value=current)]

    async def get_domain_coverage_map(self) -> dict[str, float]:
        """Derive from world model generative schemas if available."""
        wm = getattr(self._logos, "world_model", None)
        if wm is None:
            return {}
        schemas = getattr(wm, "generative_schemas", {})
        coverage: dict[str, float] = {}
        for key, schema in schemas.items():
            cov = getattr(schema, "coverage", None)
            if cov is not None:
                coverage[str(key)] = float(cov)
            else:
                coverage[str(key)] = 0.5  # assume moderate coverage
        return coverage

    async def get_recent_compression_events(
        self, window_hours: float = 24.0
    ) -> list[CompressionEvent]:
        """No historical compression events tracked yet — return empty."""
        return []


class FoveaMetricsAdapter:
    """
    Adapts FoveaService to the FoveaMetrics protocol required by Telos.

    FoveaService exists but doesn't expose prediction error rates as
    direct properties.  This adapter extracts what it can from the
    bridge health metrics and internal engine.
    """

    def __init__(self, fovea: FoveaService) -> None:
        self._fovea = fovea

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
        """Inverse of error rate — fraction of correct predictions."""
        error_rate = await self.get_prediction_error_rate()
        return 1.0 - error_rate

    async def get_recent_high_error_experiences(
        self, window_hours: float = 24.0
    ) -> list[HighErrorExperience]:
        """High-error experiences not yet tracked — return empty."""
        return []

    async def get_confabulation_rate(self) -> float:
        """
        Confabulation rate from false alarms in weight learner.

        False alarms = predictions that triggered attention but weren't
        followed by actual prediction errors — a proxy for confabulation.
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
        # Domains with low weight are the ones where fovea is over-confident
        low_weight_count = sum(1 for w in weights.values() if w < 0.3)
        return min(1.0, low_weight_count / max(len(weights), 1))
