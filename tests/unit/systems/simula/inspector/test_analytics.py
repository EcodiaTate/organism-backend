"""
Unit tests for Inspector Analytics (Phase 9).

Tests the three analytics layers:
  - InspectorAnalyticsEmitter: structured event emission, buffering, stats
  - InspectorAnalyticsView: in-memory aggregation, weekly trends, rolling windows
  - InspectorAnalyticsStore: TimescaleDB persistence (mocked)
  - InspectorEvent: event data model
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from systems.simula.inspector.analytics import (
    HUNTING_VERSION,
    InspectorAnalyticsEmitter,
    InspectorAnalyticsStore,
    InspectorAnalyticsView,
    InspectorEvent,
    _iso_week_start,
    _WeekBucket,
)
from systems.simula.inspector.types import (
    VulnerabilityClass,
    VulnerabilitySeverity,
)

# ── InspectorEvent Tests ──────────────────────────────────────────────────────


class TestInspectorEvent:
    def test_default_construction(self):
        event = InspectorEvent(event_type="hunt_started")
        assert event.event_type == "hunt_started"
        assert event.id  # auto-generated
        assert event.timestamp is not None
        assert event.hunting_version == HUNTING_VERSION
        assert event.target_url == "unknown"
        assert event.payload == {}

    def test_custom_fields(self):
        event = InspectorEvent(
            hunt_id="hunt_123",
            event_type="vulnerability_proved",
            target_url="https://github.com/test/repo",
            payload={"severity": "critical"},
        )
        assert event.hunt_id == "hunt_123"
        assert event.target_url == "https://github.com/test/repo"
        assert event.payload["severity"] == "critical"

    def test_timestamp_auto_set(self):
        """Timestamp should be auto-set when not provided."""
        event = InspectorEvent(event_type="test")
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)


# ── Week Helpers ───────────────────────────────────────────────────────────


class TestWeekHelpers:
    def test_iso_week_start_monday(self):
        """Should return Monday 00:00 UTC of the ISO week."""
        # 2026-02-26 is a Thursday
        thursday = datetime(2026, 2, 26, 14, 30, 0, tzinfo=UTC)
        monday = _iso_week_start(thursday)
        assert monday.weekday() == 0  # Monday
        assert monday.hour == 0
        assert monday.minute == 0
        assert monday.day == 23  # Monday of that week

    def test_iso_week_start_already_monday(self):
        monday = datetime(2026, 2, 23, 10, 0, 0, tzinfo=UTC)
        result = _iso_week_start(monday)
        assert result.day == 23
        assert result.hour == 0

    def test_week_bucket_defaults(self):
        now = datetime.now(UTC)
        bucket = _WeekBucket(now)
        assert bucket.vulnerability_count == 0
        assert bucket.patches_attempted == 0
        assert bucket.patches_succeeded == 0
        assert bucket.surfaces_mapped == 0
        assert bucket.hunts_completed == 0
        assert bucket.total_duration_ms == 0


# ── InspectorAnalyticsEmitter Tests ──────────────────────────────────────────


class TestAnalyticsEmitter:
    def test_construction_without_tsdb(self):
        emitter = InspectorAnalyticsEmitter()
        assert emitter._store is None
        assert emitter._buffer == []
        assert emitter.stats["events_emitted"] == 0

    def test_construction_with_tsdb(self):
        tsdb = MagicMock()
        emitter = InspectorAnalyticsEmitter(tsdb=tsdb)
        assert emitter._store is not None

    def test_emit_hunt_started_increments_counter(self):
        emitter = InspectorAnalyticsEmitter()
        emitter.emit_hunt_started(
            "https://github.com/test/repo", "external_repo",
            hunt_id="h1",
        )
        assert emitter.stats["events_emitted"] == 1
        assert emitter.stats["events_failed"] == 0

    def test_emit_all_event_types(self):
        """All 9 event emitter methods should work without errors."""
        emitter = InspectorAnalyticsEmitter()

        emitter.emit_hunt_started("url", "external_repo", hunt_id="h1")
        emitter.emit_attack_surface_discovered(
            surface_type="api_endpoint",
            entry_point="get_user",
            file_path="routes.py",
            hunt_id="h1",
        )
        emitter.emit_vulnerability_proved(
            vulnerability_class="sql_injection",
            severity="critical",
            z3_time_ms=150,
            hunt_id="h1",
        )
        emitter.emit_poc_generated(
            vuln_id="v1",
            poc_size_bytes=512,
            hunt_id="h1",
        )
        emitter.emit_patch_generated(
            vuln_id="v1",
            repair_time_ms=3000,
            patch_size_bytes=256,
            hunt_id="h1",
        )
        emitter.emit_hunt_completed(
            target_url="url",
            total_surfaces=10,
            total_vulnerabilities=3,
            total_time_ms=5000,
            hunt_id="h1",
        )
        emitter.emit_hunt_error(
            pipeline_stage="ingest",
            error_type="RuntimeError",
            error_message="git clone failed",
            hunt_id="h1",
        )
        emitter.emit_proof_timeout(
            entry_point="get_user",
            attack_goal="SQL injection",
            timeout_s=30,
            hunt_id="h1",
        )
        emitter.emit_surface_mapping_failed(
            error_message="AST parse error",
            file_count=5,
            hunt_id="h1",
        )

        assert emitter.stats["events_emitted"] == 9

    def test_error_message_truncated(self):
        """Error messages should be truncated to 500 chars."""
        emitter = InspectorAnalyticsEmitter()
        long_message = "x" * 1000
        # Should not raise — message is truncated internally
        emitter.emit_hunt_error(
            pipeline_stage="prove",
            error_type="RuntimeError",
            error_message=long_message,
            hunt_id="h1",
        )
        assert emitter.stats["events_emitted"] == 1

    def test_stats_property(self):
        emitter = InspectorAnalyticsEmitter()
        stats = emitter.stats
        assert set(stats.keys()) == {
            "events_emitted",
            "events_failed",
            "events_persisted",
            "events_persist_failed",
            "buffer_size",
        }

    def test_buffer_accumulates_with_tsdb(self):
        tsdb = MagicMock()
        emitter = InspectorAnalyticsEmitter(tsdb=tsdb)
        emitter.emit_hunt_started("url", "external_repo")
        assert emitter.stats["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_flush_persists_buffer(self):
        tsdb = MagicMock()
        emitter = InspectorAnalyticsEmitter(tsdb=tsdb)
        emitter._store = MagicMock()
        emitter._store.write_events = AsyncMock()

        emitter.emit_hunt_started("url", "external_repo")
        emitter.emit_hunt_completed(
            target_url="url",
            total_surfaces=0,
            total_vulnerabilities=0,
            total_time_ms=0,
        )
        assert emitter.stats["buffer_size"] == 2

        await emitter.flush()
        emitter._store.write_events.assert_called_once()
        assert emitter.stats["buffer_size"] == 0
        assert emitter.stats["events_persisted"] == 2

    @pytest.mark.asyncio
    async def test_flush_empty_buffer_noop(self):
        emitter = InspectorAnalyticsEmitter()
        await emitter.flush()  # Should not raise

    @pytest.mark.asyncio
    async def test_flush_failure_updates_counter(self):
        tsdb = MagicMock()
        emitter = InspectorAnalyticsEmitter(tsdb=tsdb)
        emitter._store = MagicMock()
        emitter._store.write_events = AsyncMock(
            side_effect=RuntimeError("TSDB down"),
        )

        emitter.emit_hunt_started("url", "external_repo")
        await emitter.flush()

        assert emitter.stats["events_persist_failed"] == 1
        assert emitter.stats["events_persisted"] == 0

    @pytest.mark.asyncio
    async def test_initialize_creates_schema(self):
        tsdb = MagicMock()
        emitter = InspectorAnalyticsEmitter(tsdb=tsdb)
        emitter._store = MagicMock()
        emitter._store.initialize = AsyncMock()

        await emitter.initialize()
        emitter._store.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_noop_without_tsdb(self):
        emitter = InspectorAnalyticsEmitter()
        await emitter.initialize()  # Should not raise


# ── InspectorAnalyticsView Tests ─────────────────────────────────────────────


def _make_hunt_result(
    *,
    target_url: str = "https://github.com/test/repo",
    vulnerabilities: list | None = None,
    patches: dict | None = None,
    surfaces_mapped: int = 5,
    total_duration_ms: int = 3000,
    completed_at: datetime | None = None,
):
    """Create a mock HuntResult for view ingestion."""
    result = MagicMock()
    result.target_url = target_url
    result.vulnerabilities_found = vulnerabilities or []
    result.generated_patches = patches or {}
    result.surfaces_mapped = surfaces_mapped
    result.total_duration_ms = total_duration_ms
    result.completed_at = completed_at or datetime.now(UTC)
    return result


def _make_mock_vuln(
    severity: VulnerabilitySeverity = VulnerabilitySeverity.HIGH,
    vuln_class: VulnerabilityClass = VulnerabilityClass.SQL_INJECTION,
    vuln_id: str = "v1",
):
    vuln = MagicMock()
    vuln.severity = severity
    vuln.vulnerability_class = vuln_class
    vuln.id = vuln_id
    return vuln


class TestAnalyticsView:
    def test_empty_summary(self):
        view = InspectorAnalyticsView()
        summary = view.summary
        assert summary["total_vulnerabilities"] == 0
        assert summary["total_hunts"] == 0
        assert summary["total_surfaces_mapped"] == 0
        assert summary["severity_distribution"] == {
            "critical": 0, "high": 0, "medium": 0, "low": 0,
        }

    def test_ingest_single_result_no_vulns(self):
        view = InspectorAnalyticsView()
        result = _make_hunt_result(surfaces_mapped=10, total_duration_ms=5000)
        view.ingest_hunt_result(result)

        summary = view.summary
        assert summary["total_hunts"] == 1
        assert summary["total_surfaces_mapped"] == 10
        assert summary["total_duration_ms"] == 5000
        assert summary["total_vulnerabilities"] == 0

    def test_ingest_with_vulnerabilities(self):
        view = InspectorAnalyticsView()

        vulns = [
            _make_mock_vuln(VulnerabilitySeverity.CRITICAL, VulnerabilityClass.SQL_INJECTION, "v1"),
            _make_mock_vuln(VulnerabilitySeverity.HIGH, VulnerabilityClass.XSS, "v2"),
            _make_mock_vuln(VulnerabilitySeverity.LOW, VulnerabilityClass.UNVALIDATED_REDIRECT, "v3"),
        ]
        result = _make_hunt_result(vulnerabilities=vulns, surfaces_mapped=8)
        view.ingest_hunt_result(result)

        summary = view.summary
        assert summary["total_vulnerabilities"] == 3
        assert summary["severity_distribution"]["critical"] == 1
        assert summary["severity_distribution"]["high"] == 1
        assert summary["severity_distribution"]["low"] == 1

    def test_patch_tracking(self):
        view = InspectorAnalyticsView()

        vulns = [
            _make_mock_vuln(vuln_id="v1"),
            _make_mock_vuln(vuln_id="v2"),
        ]
        patches = {"v1": "--- a/f.py\n+++ b/f.py"}
        result = _make_hunt_result(vulnerabilities=vulns, patches=patches)
        view.ingest_hunt_result(result)

        summary = view.summary
        assert summary["total_patches_attempted"] == 2
        assert summary["total_patches_succeeded"] == 1
        assert summary["patch_success_rate"] == 0.5

    def test_multiple_hunt_results_accumulated(self):
        view = InspectorAnalyticsView()

        for i in range(3):
            vulns = [_make_mock_vuln(vuln_id=f"v{i}")]
            view.ingest_hunt_result(
                _make_hunt_result(vulnerabilities=vulns, surfaces_mapped=5),
            )

        summary = view.summary
        assert summary["total_hunts"] == 3
        assert summary["total_vulnerabilities"] == 3
        assert summary["total_surfaces_mapped"] == 15
        assert summary["avg_surfaces_per_hunt"] == 5.0
        assert summary["avg_vulns_per_hunt"] == 1.0

    def test_per_target_counts(self):
        view = InspectorAnalyticsView()

        vulns_a = [_make_mock_vuln(vuln_id="v1")]
        vulns_b = [_make_mock_vuln(vuln_id="v2"), _make_mock_vuln(vuln_id="v3")]

        view.ingest_hunt_result(_make_hunt_result(target_url="repo_a", vulnerabilities=vulns_a))
        view.ingest_hunt_result(_make_hunt_result(target_url="repo_b", vulnerabilities=vulns_b))

        summary = view.summary
        assert summary["vulnerability_counts_by_target"]["repo_a"] == 1
        assert summary["vulnerability_counts_by_target"]["repo_b"] == 2

    def test_weekly_trends(self):
        view = InspectorAnalyticsView()
        now = datetime.now(UTC)

        vulns = [_make_mock_vuln(vuln_id="v1")]
        view.ingest_hunt_result(
            _make_hunt_result(vulnerabilities=vulns, completed_at=now),
        )

        trends = view.summary["weekly_trends"]
        assert len(trends) >= 1
        assert trends[-1]["vulnerabilities"] == 1

    def test_most_common_classes(self):
        view = InspectorAnalyticsView()

        vulns = [
            _make_mock_vuln(vuln_class=VulnerabilityClass.SQL_INJECTION, vuln_id="v1"),
            _make_mock_vuln(vuln_class=VulnerabilityClass.SQL_INJECTION, vuln_id="v2"),
            _make_mock_vuln(vuln_class=VulnerabilityClass.XSS, vuln_id="v3"),
        ]
        view.ingest_hunt_result(_make_hunt_result(vulnerabilities=vulns))

        classes = view.summary["most_common_classes"]
        assert len(classes) >= 1
        assert classes[0][0] == "sql_injection"
        assert classes[0][1] == 2

    def test_rolling_window(self):
        view = InspectorAnalyticsView()
        now = datetime.now(UTC)

        vulns = [_make_mock_vuln(vuln_id="v1")]
        view.ingest_hunt_result(
            _make_hunt_result(
                vulnerabilities=vulns,
                completed_at=now,
            ),
        )

        rolling_7d = view.summary["rolling_7d"]
        assert rolling_7d["days"] == 7
        assert rolling_7d["vulnerabilities"] == 1
        assert rolling_7d["hunts"] == 1

        rolling_30d = view.summary["rolling_30d"]
        assert rolling_30d["days"] == 30
        assert rolling_30d["vulnerabilities"] == 1

    def test_target_weekly_trend(self):
        view = InspectorAnalyticsView()
        now = datetime.now(UTC)

        vulns = [_make_mock_vuln(vuln_id="v1")]
        view.ingest_hunt_result(
            _make_hunt_result(
                target_url="https://github.com/test/repo",
                vulnerabilities=vulns,
                completed_at=now,
            ),
        )

        trend = view.get_target_weekly_trend("https://github.com/test/repo")
        assert len(trend) >= 1
        assert trend[-1]["vulnerabilities"] == 1

    def test_target_weekly_trend_unknown_target(self):
        view = InspectorAnalyticsView()
        trend = view.get_target_weekly_trend("nonexistent")
        assert trend == []

    def test_week_key_format(self):
        view = InspectorAnalyticsView()
        dt = datetime(2026, 2, 26, tzinfo=UTC)
        key = view._get_week_key(dt)
        assert key.startswith("2026-W")
        assert len(key) == 8  # "YYYY-WNN"

    def test_week_bucket_eviction(self):
        """Should evict oldest bucket when capacity exceeded."""
        view = InspectorAnalyticsView()
        view._max_weeks = 3  # Force small capacity

        base = datetime(2026, 1, 5, tzinfo=UTC)
        for i in range(5):
            dt = base + timedelta(weeks=i)
            vulns = [_make_mock_vuln(vuln_id=f"v{i}")]
            view.ingest_hunt_result(
                _make_hunt_result(vulnerabilities=vulns, completed_at=dt),
            )

        assert len(view._weekly_buckets) <= 3

    def test_recent_results_capped(self):
        view = InspectorAnalyticsView()
        view._max_recent = 5

        for i in range(10):
            view.ingest_hunt_result(
                _make_hunt_result(target_url=f"repo_{i}"),
            )

        assert len(view._recent_results) <= 5


# ── InspectorAnalyticsStore Tests ────────────────────────────────────────────


class TestAnalyticsStore:
    def test_construction(self):
        tsdb = MagicMock()
        store = InspectorAnalyticsStore(tsdb)
        assert store._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_creates_schema(self):
        conn = AsyncMock()
        conn.execute = AsyncMock()
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        ))
        tsdb = MagicMock()
        tsdb.pool = pool

        store = InspectorAnalyticsStore(tsdb)
        await store.initialize()

        assert store._initialized is True
        assert conn.execute.call_count > 0

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        conn = AsyncMock()
        conn.execute = AsyncMock()
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        ))
        tsdb = MagicMock()
        tsdb.pool = pool

        store = InspectorAnalyticsStore(tsdb)
        await store.initialize()
        call_count = conn.execute.call_count

        await store.initialize()  # Should be a no-op
        assert conn.execute.call_count == call_count

    @pytest.mark.asyncio
    async def test_write_events_batch(self):
        conn = AsyncMock()
        conn.executemany = AsyncMock()
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        ))
        tsdb = MagicMock()
        tsdb.pool = pool

        store = InspectorAnalyticsStore(tsdb)

        events = [
            InspectorEvent(
                hunt_id="h1",
                event_type="hunt_started",
                target_url="url",
                payload={"workspace_type": "external_repo"},
            ),
            InspectorEvent(
                hunt_id="h1",
                event_type="hunt_completed",
                target_url="url",
                payload={"total_surfaces": 5},
            ),
        ]

        await store.write_events(events)
        conn.executemany.assert_called_once()

        # Verify the rows passed to executemany
        args = conn.executemany.call_args
        rows = args[0][1]
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_write_events_empty_noop(self):
        tsdb = MagicMock()
        store = InspectorAnalyticsStore(tsdb)
        await store.write_events([])  # Should not raise

    @pytest.mark.asyncio
    async def test_get_vulnerabilities_per_week(self):
        """Should query continuous aggregate with fallback to raw."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        ))
        tsdb = MagicMock()
        tsdb.pool = pool

        store = InspectorAnalyticsStore(tsdb)
        result = await store.get_vulnerabilities_per_week(weeks=4)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_severity_distribution(self):
        row1 = {"severity": "critical", "cnt": 5}
        row2 = {"severity": "high", "cnt": 3}
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[row1, row2])
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        ))
        tsdb = MagicMock()
        tsdb.pool = pool

        store = InspectorAnalyticsStore(tsdb)
        dist = await store.get_severity_distribution(days=30)
        assert dist["critical"] == 5
        assert dist["high"] == 3

    @pytest.mark.asyncio
    async def test_get_most_common_classes(self):
        rows = [
            {"vuln_class": "sql_injection", "cnt": 10},
            {"vuln_class": "xss", "cnt": 7},
        ]
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=rows)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        ))
        tsdb = MagicMock()
        tsdb.pool = pool

        store = InspectorAnalyticsStore(tsdb)
        classes = await store.get_most_common_classes(days=30, limit=5)
        assert classes == [("sql_injection", 10), ("xss", 7)]

    @pytest.mark.asyncio
    async def test_get_patch_success_rate(self):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"patches": 3, "vulns": 10})
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        ))
        tsdb = MagicMock()
        tsdb.pool = pool

        store = InspectorAnalyticsStore(tsdb)
        rate = await store.get_patch_success_rate(days=30)
        assert rate["patches_generated"] == 3
        assert rate["vulnerabilities_found"] == 10
        assert rate["patch_rate"] == 0.3

    @pytest.mark.asyncio
    async def test_get_hunt_timeline(self):
        rows = [
            {
                "time": datetime(2026, 2, 26, 10, 0, 0, tzinfo=UTC),
                "event_type": "hunt_started",
                "target_url": "url",
                "payload": '{"workspace_type": "external_repo"}',
            },
        ]
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=rows)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        ))
        tsdb = MagicMock()
        tsdb.pool = pool

        store = InspectorAnalyticsStore(tsdb)
        timeline = await store.get_hunt_timeline("h1")
        assert len(timeline) == 1
        assert timeline[0]["event_type"] == "hunt_started"

    @pytest.mark.asyncio
    async def test_get_target_history(self):
        rows = [
            {
                "time": datetime(2026, 2, 26, tzinfo=UTC),
                "hunt_id": "h1",
                "payload": '{"total_surfaces": 5, "total_vulnerabilities": 2}',
            },
        ]
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=rows)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        ))
        tsdb = MagicMock()
        tsdb.pool = pool

        store = InspectorAnalyticsStore(tsdb)
        history = await store.get_target_history("url")
        assert len(history) == 1
        assert history[0]["hunt_id"] == "h1"
        assert history[0]["total_surfaces"] == 5

    @pytest.mark.asyncio
    async def test_get_error_summary(self):
        rows = [
            {
                "stage": "prove",
                "error_type": "TimeoutError",
                "cnt": 3,
                "last_seen": datetime(2026, 2, 26, tzinfo=UTC),
            },
        ]
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=rows)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=False),
        ))
        tsdb = MagicMock()
        tsdb.pool = pool

        store = InspectorAnalyticsStore(tsdb)
        errors = await store.get_error_summary(days=7)
        assert len(errors) == 1
        assert errors[0]["pipeline_stage"] == "prove"
        assert errors[0]["count"] == 3
