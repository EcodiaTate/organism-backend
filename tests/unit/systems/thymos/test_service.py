"""
Tests for ThymosService — the immune system orchestrator.

Tests the service lifecycle, incident pipeline, and cross-system integration.
All external dependencies (Neo4j, LLM, Synapse, Equor, Evo) are mocked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from config import ThymosConfig
from primitives.common import new_id, utc_now
from systems.synapse.types import SynapseEvent, SynapseEventType
from systems.thymos.service import ThymosService
from systems.thymos.types import (
    Incident,
    IncidentClass,
    IncidentSeverity,
)


def _make_config() -> ThymosConfig:
    return ThymosConfig(
        sentinel_scan_interval_s=999.0,  # Don't scan during tests
        homeostasis_interval_s=999.0,  # Don't run homeostasis during tests
    )


def _make_synapse_mock() -> MagicMock:
    synapse = MagicMock()
    event_bus = MagicMock()
    event_bus.subscribe = MagicMock()
    synapse._event_bus = event_bus
    synapse._health = MagicMock()
    synapse._health.get_record = MagicMock(return_value=None)
    return synapse


def _make_incident(
    fingerprint: str = "svc_test_fp",
    source_system: str = "nova",
    severity: IncidentSeverity = IncidentSeverity.MEDIUM,
    incident_class: IncidentClass = IncidentClass.CRASH,
) -> Incident:
    return Incident(
        id=new_id(),
        timestamp=utc_now(),
        incident_class=incident_class,
        severity=severity,
        fingerprint=fingerprint,
        source_system=source_system,
        error_type="RuntimeError",
        error_message="test error for service",
    )


async def _make_thymos(**kwargs) -> ThymosService:
    """Factory that creates and initializes a ThymosService with mocked deps."""
    config = kwargs.pop("config", _make_config())
    synapse = kwargs.pop("synapse", _make_synapse_mock())

    service = ThymosService(
        config=config,
        synapse=synapse,
        neo4j=kwargs.pop("neo4j", None),
        llm=kwargs.pop("llm", None),
        metrics=kwargs.pop("metrics", None),
    )
    await service.initialize()
    return service


# ─── Lifecycle ────────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initialize(self):
        thymos = await _make_thymos()
        assert thymos._initialized is True
        assert thymos._governor is not None
        assert thymos._antibody_library is not None
        assert thymos._deduplicator is not None
        assert thymos._diagnostic_engine is not None
        assert thymos._prescriber is not None

    @pytest.mark.asyncio
    async def test_double_initialize_is_idempotent(self):
        thymos = await _make_thymos()
        await thymos.initialize()
        assert thymos._initialized is True

    @pytest.mark.asyncio
    async def test_shutdown(self):
        thymos = await _make_thymos()
        await thymos.shutdown()
        assert thymos._sentinel_task is None
        assert thymos._homeostasis_task is None

    @pytest.mark.asyncio
    async def test_subscribes_to_synapse_events(self):
        synapse = _make_synapse_mock()
        await _make_thymos(synapse=synapse)
        # Should have subscribed to multiple event types
        assert synapse._event_bus.subscribe.call_count > 0


# ─── Cross-System Wiring ─────────────────────────────────────────


class TestWiring:
    @pytest.mark.asyncio
    async def test_set_equor(self):
        thymos = await _make_thymos()
        mock_equor = MagicMock()
        thymos.set_equor(mock_equor)
        assert thymos._equor is mock_equor

    @pytest.mark.asyncio
    async def test_set_evo(self):
        thymos = await _make_thymos()
        mock_evo = MagicMock()
        thymos.set_evo(mock_evo)
        assert thymos._evo is mock_evo

    @pytest.mark.asyncio
    async def test_set_atune(self):
        thymos = await _make_thymos()
        mock_atune = MagicMock()
        thymos.set_atune(mock_atune)
        assert thymos._atune is mock_atune


# ─── Incident Pipeline ───────────────────────────────────────────


class TestIncidentPipeline:
    @pytest.mark.asyncio
    async def test_on_incident_tracks_count(self):
        thymos = await _make_thymos()
        incident = _make_incident(fingerprint="unique_1")
        await thymos.on_incident(incident)
        # Allow background task to settle
        await asyncio.sleep(0.1)
        assert thymos._total_incidents >= 1

    @pytest.mark.asyncio
    async def test_duplicate_incident_deduplicated(self):
        thymos = await _make_thymos()
        inc1 = _make_incident(fingerprint="dup_svc")
        inc2 = _make_incident(fingerprint="dup_svc")
        await thymos.on_incident(inc1)
        await thymos.on_incident(inc2)
        await asyncio.sleep(0.1)
        # Only 1 unique incident should be counted
        assert thymos._total_incidents == 1

    @pytest.mark.asyncio
    async def test_on_incident_never_raises(self):
        thymos = await _make_thymos()
        # Even with a bad incident, on_incident should not raise
        incident = _make_incident()
        incident.fingerprint = ""  # Edge case
        await thymos.on_incident(incident)
        # Should not raise — that's the test

    @pytest.mark.asyncio
    async def test_info_severity_gets_accepted(self):
        thymos = await _make_thymos()
        incident = _make_incident(
            fingerprint="info_fp",
            severity=IncidentSeverity.INFO,
        )
        await thymos.on_incident(incident)
        await asyncio.sleep(0.1)
        # INFO severity should be routed to NOOP and accepted
        assert incident.id not in thymos._active_incidents


# ─── Synapse Event Handling ──────────────────────────────────────


class TestSynapseEventHandling:
    @pytest.mark.asyncio
    async def test_system_failed_creates_incident(self):
        thymos = await _make_thymos()
        event = SynapseEvent(
            event_type=SynapseEventType.SYSTEM_FAILED,
            data={"system_id": "nova"},
            source_system="synapse",
        )
        await thymos._on_synapse_event(event)
        await asyncio.sleep(0.1)
        assert thymos._total_incidents >= 1

    @pytest.mark.asyncio
    async def test_system_recovered_resolves_incidents(self):
        thymos = await _make_thymos()
        # Create an active incident for nova
        incident = _make_incident(fingerprint="recover_fp", source_system="nova")
        thymos._active_incidents[incident.id] = incident
        thymos._governor.register_incident(incident)

        # Send recovery event
        event = SynapseEvent(
            event_type=SynapseEventType.SYSTEM_RECOVERED,
            data={"system_id": "nova"},
            source_system="synapse",
        )
        await thymos._on_synapse_event(event)
        assert incident.id not in thymos._active_incidents

    @pytest.mark.asyncio
    async def test_classify_system_failed(self):
        thymos = await _make_thymos()
        event = SynapseEvent(
            event_type=SynapseEventType.SYSTEM_FAILED,
            data={},
        )
        severity, inc_class = thymos._classify_synapse_event(event)
        assert severity == IncidentSeverity.CRITICAL
        assert inc_class == IncidentClass.CRASH

    @pytest.mark.asyncio
    async def test_classify_system_overloaded(self):
        thymos = await _make_thymos()
        event = SynapseEvent(
            event_type=SynapseEventType.SYSTEM_OVERLOADED,
            data={},
        )
        severity, inc_class = thymos._classify_synapse_event(event)
        assert severity == IncidentSeverity.MEDIUM
        assert inc_class == IncidentClass.DEGRADATION


# ─── Public API ──────────────────────────────────────────────────


class TestPublicAPI:
    @pytest.mark.asyncio
    async def test_report_exception(self):
        thymos = await _make_thymos()
        try:
            raise ValueError("test public api")
        except ValueError as exc:
            await thymos.report_exception("nova", exc)
        await asyncio.sleep(0.1)
        assert thymos._total_incidents >= 1

    @pytest.mark.asyncio
    async def test_report_contract_violation(self):
        thymos = await _make_thymos()
        await thymos.report_contract_violation(
            source="atune",
            target="memory",
            operation="store_percept",
            latency_ms=500.0,
            sla_ms=100.0,
        )
        await asyncio.sleep(0.1)
        assert thymos._total_incidents >= 1

    @pytest.mark.asyncio
    async def test_record_metric(self):
        thymos = await _make_thymos()
        # Record a metric — should not raise
        thymos.record_metric("synapse.cycle.latency_ms", 120.0)

    @pytest.mark.asyncio
    async def test_scan_files(self):
        thymos = await _make_thymos()
        result = await thymos.scan_files([])
        assert isinstance(result, list)


# ─── Health ──────────────────────────────────────────────────────


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_dict(self):
        thymos = await _make_thymos()
        result = await thymos.health()
        assert isinstance(result, dict)
        assert result["status"] == "healthy"
        assert result["initialized"] is True

    @pytest.mark.asyncio
    async def test_health_includes_all_sections(self):
        thymos = await _make_thymos()
        result = await thymos.health()
        assert "total_incidents" in result
        assert "total_antibodies" in result
        assert "repairs_attempted" in result
        assert "total_diagnoses" in result
        assert "homeostatic_adjustments" in result
        assert "storm_activations" in result
        assert "prophylactic_scans" in result
        assert "budget" in result
        assert "healing_mode" in result

    @pytest.mark.asyncio
    async def test_stats_property(self):
        thymos = await _make_thymos()
        stats = thymos.stats
        assert stats["initialized"] is True
        assert "total_incidents" in stats
        assert "healing_mode" in stats


# ─── Telemetry ────────────────────────────────────────────────────


class TestTelemetry:
    @pytest.mark.asyncio
    async def test_emit_metric_with_collector(self):
        metrics_mock = MagicMock()
        metrics_mock.record = MagicMock()
        thymos = await _make_thymos(metrics=metrics_mock)

        incident = _make_incident(fingerprint="telemetry_fp")
        await thymos.on_incident(incident)
        await asyncio.sleep(0.1)

        # Should have emitted at least one metric
        assert metrics_mock.record.call_count > 0

    @pytest.mark.asyncio
    async def test_emit_metric_without_collector(self):
        thymos = await _make_thymos(metrics=None)
        # Should not raise when no metrics collector
        thymos._emit_metric("test.metric", 1.0)
