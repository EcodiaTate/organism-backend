"""
Tests for Thymos Diagnostic Layer.

Covers:
  - CausalAnalyzer
  - TemporalCorrelator
  - DiagnosticEngine
"""

from __future__ import annotations

import pytest

from primitives.common import new_id, utc_now
from systems.thymos.diagnosis import (
    CausalAnalyzer,
    DiagnosticEngine,
    TemporalCorrelator,
)
from systems.thymos.types import (
    Antibody,
    CausalChain,
    Incident,
    IncidentClass,
    IncidentSeverity,
    RepairSpec,
    RepairTier,
)


def _make_incident(
    source_system: str = "nova",
    severity: IncidentSeverity = IncidentSeverity.HIGH,
    incident_class: IncidentClass = IncidentClass.CRASH,
    error_type: str = "RuntimeError",
    error_message: str = "something failed",
) -> Incident:
    return Incident(
        id=new_id(),
        timestamp=utc_now(),
        incident_class=incident_class,
        severity=severity,
        fingerprint="test_fp_1234",
        source_system=source_system,
        error_type=error_type,
        error_message=error_message,
    )


def _make_antibody(
    effectiveness: float = 0.95,
    fingerprint: str = "test_fp_1234",
) -> Antibody:
    return Antibody(
        id=new_id(),
        fingerprint=fingerprint,
        incident_class=IncidentClass.CRASH,
        source_system="nova",
        error_pattern="something failed",
        repair_tier=RepairTier.PARAMETER,
        repair_spec=RepairSpec(
            tier=RepairTier.PARAMETER,
            action="adjust_parameters",
            reason="Test fix",
        ),
        root_cause_description="Test root cause",
        effectiveness=effectiveness,
    )


# ─── CausalAnalyzer ──────────────────────────────────────────────


class TestCausalAnalyzer:
    @pytest.mark.asyncio
    async def test_no_upstream_returns_local(self):
        analyzer = CausalAnalyzer()
        incident = _make_incident(source_system="memory")
        chain = await analyzer.trace_root_cause(incident)
        assert chain.root_system == "memory"
        assert chain.confidence > 0.0

    @pytest.mark.asyncio
    async def test_upstream_incident_traced(self):
        analyzer = CausalAnalyzer()
        # Record a memory incident first
        mem_incident = _make_incident(source_system="memory")
        analyzer.record_incident(mem_incident)
        # Now trace nova failure — memory is upstream of nova
        nova_incident = _make_incident(source_system="nova")
        chain = await analyzer.trace_root_cause(nova_incident)
        assert chain.root_system == "memory"
        assert "memory" in chain.chain

    @pytest.mark.asyncio
    async def test_healthy_upstream_returns_local(self):
        analyzer = CausalAnalyzer()
        # No upstream incidents recorded
        nova_incident = _make_incident(source_system="nova")
        chain = await analyzer.trace_root_cause(nova_incident)
        assert chain.root_system == "nova"

    def test_find_common_upstream_with_empty(self):
        analyzer = CausalAnalyzer()
        assert analyzer.find_common_upstream([]) is None

    def test_find_common_upstream(self):
        analyzer = CausalAnalyzer()
        incidents = [
            _make_incident(source_system="nova"),
            _make_incident(source_system="voxis"),
            _make_incident(source_system="axon"),
        ]
        # All depend on memory or equor
        common = analyzer.find_common_upstream(incidents)
        assert common is not None

    def test_record_incident_limits_buffer(self):
        analyzer = CausalAnalyzer()
        for _i in range(100):
            analyzer.record_incident(_make_incident())
        # Should not exceed 50 per system
        assert len(analyzer._recent_incidents.get("nova", [])) <= 50


# ─── TemporalCorrelator ──────────────────────────────────────────


class TestTemporalCorrelator:
    def test_correlate_empty_returns_empty(self):
        correlator = TemporalCorrelator()
        incident = _make_incident()
        result = correlator.correlate(incident)
        assert result == []

    def test_correlate_finds_recent_events(self):
        correlator = TemporalCorrelator()
        # Record an event at the current time
        correlator.record_event(
            event_type="system_restart",
            details="Memory restarted",
            system_id="memory",
        )
        # Create an incident at "now" — the event should correlate
        incident = _make_incident()
        result = correlator.correlate(incident, window_s=60.0)
        assert len(result) >= 1
        assert result[0].type == "system_restart"

    def test_record_metric_anomaly(self):
        correlator = TemporalCorrelator()
        correlator.record_metric_anomaly(
            metric_name="memory.latency_ms",
            value=500.0,
            baseline=100.0,
            z_score=4.0,
        )
        incident = _make_incident()
        result = correlator.correlate(incident, window_s=60.0)
        assert len(result) >= 1
        assert result[0].type == "metric_anomaly"

    def test_event_buffer_limits(self):
        correlator = TemporalCorrelator()
        for i in range(2000):
            correlator.record_event("test", f"event_{i}")
        assert len(correlator._events) <= 1000


# ─── DiagnosticEngine ────────────────────────────────────────────


class TestDiagnosticEngine:
    @pytest.mark.asyncio
    async def test_antibody_match_fast_path(self):
        engine = DiagnosticEngine()
        incident = _make_incident()
        antibody = _make_antibody(effectiveness=0.95)

        chain = CausalChain(
            root_system="nova",
            chain=["nova"],
            confidence=0.5,
        )
        diagnosis = await engine.diagnose(
            incident=incident,
            causal_chain=chain,
            correlations=[],
            antibody_match=antibody,
        )
        assert diagnosis.antibody_id == antibody.id
        assert diagnosis.repair_tier == RepairTier.KNOWN_FIX
        assert diagnosis.confidence >= 0.8

    @pytest.mark.asyncio
    async def test_rule_based_fallback(self):
        """Without LLM, the engine should still produce a diagnosis."""
        engine = DiagnosticEngine(llm_client=None)
        incident = _make_incident()
        chain = CausalChain(
            root_system="nova",
            chain=["nova"],
            confidence=0.5,
        )
        diagnosis = await engine.diagnose(
            incident=incident,
            causal_chain=chain,
            correlations=[],
            antibody_match=None,
        )
        assert diagnosis.root_cause  # Non-empty
        assert 0.0 <= diagnosis.confidence <= 1.0
        assert diagnosis.repair_tier is not None

    @pytest.mark.asyncio
    async def test_low_effectiveness_antibody_not_fast_pathed(self):
        """Antibody with low effectiveness should go through full diagnosis."""
        engine = DiagnosticEngine(llm_client=None)
        incident = _make_incident()
        antibody = _make_antibody(effectiveness=0.3)  # Below 0.8 threshold

        chain = CausalChain(
            root_system="nova",
            chain=["nova"],
            confidence=0.5,
        )
        diagnosis = await engine.diagnose(
            incident=incident,
            causal_chain=chain,
            correlations=[],
            antibody_match=antibody,
        )
        # Should NOT take the fast path
        assert diagnosis.antibody_id is None or diagnosis.confidence < 0.8
