"""
EcodiaOS -- Simula Evolution Analytics Engine

Tracks evolution quality metrics over time, enabling Simula to learn
from its own history. All analytics are computed from Neo4j evolution
records -- zero LLM tokens required.

Key metrics:
  - Per-category success/rollback rates
  - Evolution velocity (proposals per day)
  - Rollback pattern analysis (which categories fail most, why)
  - Dynamic caution adjustment (increase risk thresholds for
    categories with high recent rollback rates)

Phase 9 addition:
  - Inspector security analytics integration (vulnerability discovery
    metrics surfaced alongside evolution metrics for unified observability)

Used by:
  - ChangeSimulator: dynamic risk threshold adjustment
  - SimulaService: enhanced stats reporting
  - ProposalIntelligence: cost/risk estimation
  - InspectorService: unified analytics surface (Phase 9)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from systems.simula.evolution_types import (
    CategorySuccessRate,
    CautionAdjustment,
    ChangeCategory,
    EvolutionAnalytics,
    RiskLevel,
)

if TYPE_CHECKING:
    from systems.simula.history import EvolutionHistoryManager
    from systems.simula.inspector.analytics import (
        InspectorAnalyticsStore,
        InspectorAnalyticsView,
    )

logger = structlog.get_logger().bind(system="simula.analytics")

# Rollback rate above this threshold triggers increased caution
_CAUTION_THRESHOLD: float = 0.30

# Window for "recent" rollback rate calculation
_RECENT_WINDOW_DAYS: int = 7

# Risk level to numeric mapping for mean calculation
_RISK_LEVEL_NUMERIC: dict[RiskLevel, float] = {
    RiskLevel.LOW: 0.1,
    RiskLevel.MODERATE: 0.4,
    RiskLevel.HIGH: 0.7,
    RiskLevel.UNACCEPTABLE: 1.0,
}


class EvolutionAnalyticsEngine:
    """
    Tracks evolution quality metrics over time.
    Enables Simula to learn from its own history and dynamically
    adjust risk thresholds based on past performance.

    All computation is from Neo4j records -- no LLM tokens consumed.
    """

    def __init__(
        self,
        history: EvolutionHistoryManager | None = None,
        *,
        inspector_view: InspectorAnalyticsView | None = None,
        inspector_store: InspectorAnalyticsStore | None = None,
    ) -> None:
        self._history = history
        self._log = logger
        self._cached_analytics: EvolutionAnalytics | None = None
        self._cache_ttl_seconds: int = 300  # 5 minutes
        self._last_computed: datetime | None = None

        # Phase 9: Inspector analytics integration
        self._inspector_view = inspector_view
        self._inspector_store = inspector_store

    async def compute_analytics(self) -> EvolutionAnalytics:
        """
        Compute current analytics from the full evolution history.
        Results are cached for 5 minutes to avoid repeated Neo4j queries.
        """
        now = utc_now()
        if (
            self._cached_analytics is not None
            and self._last_computed is not None
            and (now - self._last_computed).total_seconds() < self._cache_ttl_seconds
        ):
            return self._cached_analytics

        if self._history is None:
            return EvolutionAnalytics()

        records = await self._history.get_history(limit=500)

        if not records:
            analytics = EvolutionAnalytics(last_updated=now)
            self._cached_analytics = analytics
            self._last_computed = now
            return analytics

        # Per-category rates
        category_rates: dict[str, CategorySuccessRate] = {}
        total_risk_numeric: float = 0.0
        risk_count: int = 0

        for record in records:
            cat_key = record.category.value
            if cat_key not in category_rates:
                category_rates[cat_key] = CategorySuccessRate(category=record.category)

            rate = category_rates[cat_key]
            rate.total += 1

            if record.rolled_back:
                rate.rolled_back += 1
            else:
                rate.approved += 1

            total_risk_numeric += _RISK_LEVEL_NUMERIC.get(record.simulation_risk, 0.4)
            risk_count += 1

        # Evolution velocity: proposals per day over the record span
        velocity = 0.0
        if len(records) >= 2:
            newest = records[0].created_at
            oldest = records[-1].created_at
            span_days = max(1.0, (newest - oldest).total_seconds() / 86400.0)
            velocity = len(records) / span_days

        # Aggregate rollback rate
        total_rolled_back = sum(r.rolled_back for r in category_rates.values())
        total_proposals = sum(r.total for r in category_rates.values())
        rollback_rate = total_rolled_back / max(1, total_proposals)

        # Mean simulation risk
        mean_risk = total_risk_numeric / max(1, risk_count)

        # Compute recent rollback rates (7-day window) per category
        recent_rollback_rates: dict[str, float] = {}
        cutoff = utc_now() - timedelta(days=_RECENT_WINDOW_DAYS)
        for cat in ChangeCategory:
            recent_records = [
                r for r in records
                if r.category == cat and r.created_at >= cutoff
            ]
            if recent_records:
                recent_rolled_back = sum(1 for r in recent_records if r.rolled_back)
                recent_rollback_rates[cat.value] = round(
                    recent_rolled_back / len(recent_records), 3
                )

        analytics = EvolutionAnalytics(
            category_rates=category_rates,
            total_proposals=total_proposals,
            evolution_velocity=round(velocity, 3),
            mean_simulation_risk=round(mean_risk, 3),
            rollback_rate=round(rollback_rate, 3),
            recent_rollback_rates=recent_rollback_rates,
            last_updated=now,
        )

        self._cached_analytics = analytics
        self._last_computed = now
        self._log.info(
            "analytics_computed",
            total_proposals=total_proposals,
            velocity=analytics.evolution_velocity,
            rollback_rate=analytics.rollback_rate,
            categories=len(category_rates),
        )
        return analytics

    async def get_category_success_rate(self, category: ChangeCategory) -> float:
        """
        Success rate for a specific change category.
        Used by ChangeSimulator for dynamic risk weighting.
        Returns 0.5 (neutral) if no history exists for this category.
        """
        analytics = await self.compute_analytics()
        rate = analytics.category_rates.get(category.value)
        if rate is None or rate.total == 0:
            return 0.5  # no data -- assume neutral
        return rate.success_rate

    async def get_recent_rollback_rate(self, category: ChangeCategory) -> float:
        """
        Rollback rate for a category within the recent window (7 days).
        More responsive to recent trends than the all-time rate.
        """
        if self._history is None:
            return 0.0

        records = await self._history.get_history(limit=200)
        cutoff = utc_now() - timedelta(days=_RECENT_WINDOW_DAYS)

        recent = [
            r for r in records
            if r.category == category and r.created_at >= cutoff
        ]

        if not recent:
            return 0.0

        rolled_back = sum(1 for r in recent if r.rolled_back)
        return rolled_back / len(recent)

    async def get_rollback_patterns(self) -> list[dict[str, Any]]:
        """
        Analyze rollback history for actionable patterns:
        - Which categories roll back most often
        - Common rollback reasons
        - Trend direction (getting better or worse)
        """
        analytics = await self.compute_analytics()
        patterns: list[dict[str, Any]] = []

        for cat_key, rate in analytics.category_rates.items():
            if rate.rolled_back == 0:
                continue
            patterns.append({
                "category": cat_key,
                "rollback_rate": round(rate.rollback_rate, 3),
                "total": rate.total,
                "rolled_back": rate.rolled_back,
                "severity": "high" if rate.rollback_rate > _CAUTION_THRESHOLD else "normal",
            })

        # Sort by rollback rate descending
        patterns.sort(key=lambda p: p["rollback_rate"], reverse=True)
        return patterns

    def should_increase_caution(self, category: ChangeCategory) -> CautionAdjustment:
        """
        Transparent caution adjustment analysis using cached analytics.
        Evaluates multiple factors to determine if simulation should use
        stricter risk thresholds for this category.

        Factors considered:
        - All-time rollback rate (indicates systemic issues)
        - Recent 7-day rollback rate (responsive to recent trends)
        - Data sufficiency (at least 3 proposals needed)

        Returns a CautionAdjustment with full reasoning for observability.
        """
        if self._cached_analytics is None:
            return CautionAdjustment(
                should_adjust=False,
                magnitude=0.0,
                factors={},
                reasoning="No cached analytics available",
            )

        rate = self._cached_analytics.category_rates.get(category.value)
        if rate is None or rate.total < 3:
            return CautionAdjustment(
                should_adjust=False,
                magnitude=0.0,
                factors={},
                reasoning="Insufficient data (< 3 proposals)",
            )

        factors: dict[str, float] = {}

        # Factor 1: All-time rollback rate
        if rate.rollback_rate > _CAUTION_THRESHOLD:
            factors["high_alltime_rollback_rate"] = min(
                0.25, rate.rollback_rate * 0.5
            )

        # Factor 2: Recent 7-day rollback rate
        recent_rate = self._cached_analytics.recent_rollback_rates.get(category.value, 0.0)
        if recent_rate > 0.25:
            factors["high_recent_rollback_rate"] = min(0.20, recent_rate * 0.4)

        total_adjustment = sum(factors.values())

        reasoning_parts = []
        if factors:
            reasoning_parts.append(
                f"Category {category.value}: "
                + ", ".join(f"{k}={v:.2f}" for k, v in factors.items())
            )
        reasoning_parts.append(
            f"All-time: {rate.rollback_rate:.1%}, "
            f"Recent (7d): {recent_rate:.1%}, "
            f"Total: {rate.total} proposals"
        )

        return CautionAdjustment(
            should_adjust=total_adjustment > 0.0,
            magnitude=min(0.5, total_adjustment),
            factors=factors,
            reasoning=" | ".join(reasoning_parts),
        )

    # ── Phase 9: Inspector Integration ──────────────────────────────────────────

    def set_inspector_view(self, view: InspectorAnalyticsView) -> None:
        """Attach a Inspector analytics view for unified querying."""
        self._inspector_view = view

    def set_inspector_store(self, store: InspectorAnalyticsStore) -> None:
        """Attach a Inspector analytics store for durable historical queries."""
        self._inspector_store = store

    def get_inspector_summary(self) -> dict[str, Any]:
        """
        Return in-memory Inspector analytics summary.

        Provides: total vulnerabilities, severity distribution, most common
        classes, patch success rate, weekly trends, rolling windows, and
        throughput metrics.

        Returns empty dict if Inspector analytics view is not attached.
        """
        if self._inspector_view is None:
            return {}
        return self._inspector_view.summary

    async def get_inspector_weekly_trends(
        self,
        *,
        weeks: int = 12,
        target_url: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query weekly vulnerability trends from TimescaleDB.

        Falls back to in-memory view if store is unavailable.

        Args:
            weeks: Number of weeks to look back.
            target_url: Optional filter by target repository.

        Returns:
            List of weekly buckets with vulnerability counts + severity breakdown.
        """
        if self._inspector_store is not None:
            try:
                return await self._inspector_store.get_vulnerabilities_per_week(
                    weeks=weeks, target_url=target_url,
                )
            except Exception as exc:
                self._log.warning(
                    "inspector_store_query_failed",
                    query="weekly_trends",
                    error=str(exc),
                )

        # Fallback to in-memory view
        if self._inspector_view is not None:
            if target_url:
                return self._inspector_view.get_target_weekly_trend(target_url)
            trends = self._inspector_view.summary.get("weekly_trends", [])
            return trends[-weeks:] if isinstance(trends, list) else []

        return []

    async def get_inspector_severity_distribution(
        self,
        *,
        days: int = 30,
        target_url: str | None = None,
    ) -> dict[str, int]:
        """
        Query severity distribution from TimescaleDB over a rolling window.

        Falls back to in-memory view (all-time) if store is unavailable.
        """
        if self._inspector_store is not None:
            try:
                return await self._inspector_store.get_severity_distribution(
                    days=days, target_url=target_url,
                )
            except Exception as exc:
                self._log.warning(
                    "inspector_store_query_failed",
                    query="severity_distribution",
                    error=str(exc),
                )

        if self._inspector_view is not None:
            dist: dict[str, int] = self._inspector_view.summary.get("severity_distribution", {})
            return dist

        return {}

    async def get_inspector_error_summary(self, *, days: int = 7) -> list[dict[str, Any]]:
        """
        Query aggregated pipeline errors from TimescaleDB.

        Only available when a Inspector analytics store is attached.
        """
        if self._inspector_store is not None:
            try:
                return await self._inspector_store.get_error_summary(days=days)
            except Exception as exc:
                self._log.warning(
                    "inspector_store_query_failed",
                    query="error_summary",
                    error=str(exc),
                )
        return []

    async def get_unified_analytics(self) -> dict[str, Any]:
        """
        Return a unified analytics payload combining evolution metrics
        and Inspector security metrics for comprehensive observability.

        This is the single entry point for dashboard consumers that want
        the complete system health picture.
        """
        evolution = await self.compute_analytics()

        result: dict[str, Any] = {
            "evolution": {
                "total_proposals": evolution.total_proposals,
                "evolution_velocity": evolution.evolution_velocity,
                "rollback_rate": evolution.rollback_rate,
                "mean_simulation_risk": evolution.mean_simulation_risk,
                "category_count": len(evolution.category_rates),
                "last_updated": (
                    evolution.last_updated.isoformat()
                    if evolution.last_updated else None
                ),
            },
        }

        # Inspector security analytics
        inspector_summary = self.get_inspector_summary()
        if inspector_summary:
            result["inspector"] = {
                "total_vulnerabilities": inspector_summary.get("total_vulnerabilities", 0),
                "total_hunts": inspector_summary.get("total_hunts", 0),
                "severity_distribution": inspector_summary.get("severity_distribution", {}),
                "patch_success_rate": inspector_summary.get("patch_success_rate", 0),
                "avg_vulns_per_hunt": inspector_summary.get("avg_vulns_per_hunt", 0),
                "rolling_7d": inspector_summary.get("rolling_7d", {}),
                "rolling_30d": inspector_summary.get("rolling_30d", {}),
            }
        else:
            result["inspector"] = None

        return result

    def invalidate_cache(self) -> None:
        """Force recomputation on next analytics request."""
        self._cached_analytics = None
        self._last_computed = None
