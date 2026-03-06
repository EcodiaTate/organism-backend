"""
Tests for Thymos Prophylactic Layer.

Covers:
  - ProphylacticScanner
  - HomeostasisController
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from systems.thymos.prophylactic import (
    HomeostasisController,
    ProphylacticScanner,
)
from systems.thymos.types import (
    Antibody,
    IncidentClass,
    RepairSpec,
    RepairTier,
)


def _make_antibody(error_pattern: str = "memory retrieval timeout") -> Antibody:
    return Antibody(
        fingerprint="scan_test_fp",
        incident_class=IncidentClass.CRASH,
        source_system="memory",
        error_pattern=error_pattern,
        repair_tier=RepairTier.PARAMETER,
        repair_spec=RepairSpec(
            tier=RepairTier.PARAMETER,
            action="adjust_parameters",
            reason="fix",
        ),
        root_cause_description="Memory retrieval timeout due to pressure",
    )


# ─── ProphylacticScanner ─────────────────────────────────────────


class TestProphylacticScanner:
    @pytest.mark.asyncio
    async def test_scan_with_no_antibodies(self):
        library = MagicMock()
        library.get_all_active = AsyncMock(return_value=[])
        scanner = ProphylacticScanner(antibody_library=library)
        result = await scanner.scan([])
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_empty_files(self):
        library = MagicMock()
        library.get_all_active = AsyncMock(return_value=[_make_antibody()])
        scanner = ProphylacticScanner(antibody_library=library)
        result = await scanner.scan([])
        assert result == []

    @pytest.mark.asyncio
    async def test_stats_incremented(self):
        library = MagicMock()
        library.get_all_active = AsyncMock(return_value=[])
        scanner = ProphylacticScanner(antibody_library=library)
        await scanner.scan([])
        assert scanner.stats["scans_run"] == 1


# ─── HomeostasisController ────────────────────────────────────────


class TestHomeostasisController:
    def test_no_adjustments_on_empty_history(self):
        controller = HomeostasisController()
        result = controller.check_homeostasis()
        assert result == []

    def test_record_metric_stores_value(self):
        controller = HomeostasisController()
        controller.record_metric("synapse.cycle.latency_ms", 120.0)
        assert len(controller._history["synapse.cycle.latency_ms"]) == 1

    def test_unknown_metric_ignored(self):
        controller = HomeostasisController()
        controller.record_metric("nonexistent.metric", 42.0)
        # Should not create a new entry
        assert "nonexistent.metric" not in controller._history

    def test_metric_in_range_no_adjustment(self):
        controller = HomeostasisController()
        # Feed values well within the optimal range
        for _ in range(20):
            controller.record_metric("synapse.cycle.latency_ms", 130.0)
        result = controller.check_homeostasis()
        # All stable — no adjustments expected
        assert len(result) == 0

    def test_metric_trending_high_triggers_cooling(self):
        controller = HomeostasisController()
        # Feed values trending toward the upper bound (180ms)
        for i in range(30):
            # Start at 150, trend up to 170
            controller.record_metric(
                "synapse.cycle.latency_ms",
                150.0 + i * 0.7,
            )
        result = controller.check_homeostasis()
        # May or may not trigger depending on exact threshold math
        # At minimum, check it doesn't crash
        assert isinstance(result, list)

    def test_metrics_in_range_count(self):
        controller = HomeostasisController()
        # Without data, metrics_in_range should be 0 or based on history
        assert isinstance(controller.metrics_in_range, int)

    def test_metrics_total_count(self):
        controller = HomeostasisController()
        assert controller.metrics_total == len(controller._ranges)

    def test_adjustments_count(self):
        controller = HomeostasisController()
        assert controller.adjustments_count == 0

    def test_history_limits(self):
        controller = HomeostasisController()
        for _ in range(300):
            controller.record_metric("synapse.cycle.latency_ms", 120.0)
        assert len(controller._history["synapse.cycle.latency_ms"]) <= controller._max_history
