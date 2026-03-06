"""
Tests for Thymos Sentinel Layer.

Covers all five sentinel classes:
  - ExceptionSentinel
  - ContractSentinel
  - FeedbackLoopSentinel
  - DriftSentinel
  - CognitiveStallSentinel
"""

from __future__ import annotations

from systems.thymos.sentinels import (
    CognitiveStallSentinel,
    ContractSentinel,
    DriftSentinel,
    ExceptionSentinel,
    FeedbackLoopSentinel,
)
from systems.thymos.types import (
    ContractSLA,
    DriftConfig,
    FeedbackLoop,
    IncidentClass,
    IncidentSeverity,
    StallConfig,
)

# ─── ExceptionSentinel ────────────────────────────────────────────


class TestExceptionSentinel:
    def test_intercept_creates_incident(self):
        sentinel = ExceptionSentinel()
        try:
            raise ValueError("test error")
        except ValueError as exc:
            incident = sentinel.intercept("nova", "deliberate", exc)

        assert incident.source_system == "nova"
        assert incident.incident_class == IncidentClass.CRASH
        assert incident.error_type == "ValueError"
        assert "test error" in incident.error_message
        assert incident.stack_trace is not None
        assert incident.fingerprint  # Non-empty

    def test_fingerprint_is_stable(self):
        sentinel = ExceptionSentinel()
        try:
            raise TypeError("boom")
        except TypeError as exc:
            fp1 = sentinel.fingerprint("memory", "store", exc)
            fp2 = sentinel.fingerprint("memory", "store", exc)

        assert fp1 == fp2
        assert len(fp1) == 16

    def test_fingerprint_differs_by_system(self):
        sentinel = ExceptionSentinel()
        try:
            raise RuntimeError("x")
        except RuntimeError as exc:
            fp_nova = sentinel.fingerprint("nova", "decide", exc)
            fp_evo = sentinel.fingerprint("evo", "decide", exc)

        assert fp_nova != fp_evo

    def test_critical_system_gets_critical_severity(self):
        sentinel = ExceptionSentinel()
        try:
            raise RuntimeError("fail")
        except RuntimeError as exc:
            incident = sentinel.intercept("equor", "review", exc)

        assert incident.severity == IncidentSeverity.CRITICAL

    def test_non_critical_system_gets_medium_severity(self):
        sentinel = ExceptionSentinel()
        try:
            raise RuntimeError("fail")
        except RuntimeError as exc:
            incident = sentinel.intercept("simula", "evolve", exc)

        assert incident.severity == IncidentSeverity.MEDIUM

    def test_blast_radius_computed_from_downstream(self):
        sentinel = ExceptionSentinel()
        try:
            raise RuntimeError("fail")
        except RuntimeError as exc:
            # memory has many downstream dependencies
            incident = sentinel.intercept("memory", "retrieve", exc)

        assert incident.blast_radius > 0.0
        assert len(incident.affected_systems) > 0

    def test_constitutional_impact_varies_by_system(self):
        sentinel = ExceptionSentinel()
        try:
            raise RuntimeError("fail")
        except RuntimeError as exc:
            equor_incident = sentinel.intercept("equor", "review", exc)
            evo_incident = sentinel.intercept("evo", "learn", exc)

        # Equor should have higher honesty impact
        assert equor_incident.constitutional_impact["honesty"] > evo_incident.constitutional_impact["honesty"]
        # Evo should have higher growth impact
        assert evo_incident.constitutional_impact["growth"] > equor_incident.constitutional_impact["growth"]

    def test_context_passed_through(self):
        sentinel = ExceptionSentinel()
        try:
            raise RuntimeError("fail")
        except RuntimeError as exc:
            incident = sentinel.intercept("nova", "decide", exc, context={"extra": "data"})

        assert incident.context["extra"] == "data"
        assert incident.context["method"] == "decide"


# ─── ContractSentinel ─────────────────────────────────────────────


class TestContractSentinel:
    def test_within_sla_returns_none(self):
        sentinel = ContractSentinel()
        result = sentinel.check_contract(
            source="atune",
            target="memory",
            operation="store_percept",
            latency_ms=50.0,
        )
        assert result is None

    def test_sla_violation_returns_incident(self):
        sentinel = ContractSentinel()
        result = sentinel.check_contract(
            source="atune",
            target="memory",
            operation="store_percept",
            latency_ms=250.0,  # SLA is 100ms
        )
        assert result is not None
        assert result.incident_class == IncidentClass.CONTRACT_VIOLATION
        assert "store_percept" in result.error_message
        assert result.context["actual_ms"] == 250.0

    def test_unknown_contract_returns_none(self):
        sentinel = ContractSentinel()
        result = sentinel.check_contract(
            source="unknown",
            target="unknown",
            operation="unknown",
            latency_ms=99999.0,
        )
        assert result is None

    def test_custom_slas(self):
        custom = [
            ContractSLA(source="a", target="b", operation="op", max_latency_ms=10),
        ]
        sentinel = ContractSentinel(slas=custom)
        result = sentinel.check_contract("a", "b", "op", 20.0)
        assert result is not None


# ─── FeedbackLoopSentinel ─────────────────────────────────────────


class TestFeedbackLoopSentinel:
    def test_all_loops_severed_on_init(self):
        sentinel = FeedbackLoopSentinel()
        incidents = sentinel.check_loops()
        assert len(incidents) > 0
        for inc in incidents:
            assert inc.incident_class == IncidentClass.LOOP_SEVERANCE

    def test_active_loop_not_flagged(self):
        loops = [
            FeedbackLoop(
                name="test_loop",
                source="nova",
                target="atune",
                signal="beliefs",
                check="check_expr",
                description="Test loop",
            ),
        ]
        sentinel = FeedbackLoopSentinel(loops=loops)
        sentinel.report_loop_active("test_loop")
        incidents = sentinel.check_loops(max_staleness_s=60.0)
        assert len(incidents) == 0

    def test_stale_loop_flagged(self):
        loops = [
            FeedbackLoop(
                name="test_loop",
                source="nova",
                target="atune",
                signal="beliefs",
                check="check_expr",
                description="Test loop",
            ),
        ]
        sentinel = FeedbackLoopSentinel(loops=loops)
        # Mark active but with an old timestamp
        sentinel._loop_status["test_loop"] = True
        sentinel._last_check["test_loop"] = 0.0  # epoch = very old
        incidents = sentinel.check_loops(max_staleness_s=1.0)
        assert len(incidents) == 1

    def test_loop_statuses_property(self):
        sentinel = FeedbackLoopSentinel()
        statuses = sentinel.loop_statuses
        assert isinstance(statuses, dict)
        assert all(isinstance(v, bool) for v in statuses.values())


# ─── DriftSentinel ────────────────────────────────────────────────


class TestDriftSentinel:
    def test_no_drift_during_warmup(self):
        sentinel = DriftSentinel(
            metrics={"test.metric": DriftConfig(window=100, sigma_threshold=2.0)}
        )
        # Feed a few values — shouldn't flag during warmup
        for _i in range(10):
            result = sentinel.record_metric("test.metric", 100.0)
        assert result is None

    def test_drift_detected_after_warmup(self):
        sentinel = DriftSentinel(
            metrics={"test.metric": DriftConfig(window=40, sigma_threshold=2.0)}
        )
        # Warm up with stable values
        for _ in range(30):
            sentinel.record_metric("test.metric", 100.0)

        # Spike way above baseline
        result = sentinel.record_metric("test.metric", 500.0)
        assert result is not None
        assert result.incident_class == IncidentClass.DRIFT
        assert "test.metric" in result.error_message

    def test_unknown_metric_ignored(self):
        sentinel = DriftSentinel(metrics={})
        result = sentinel.record_metric("nonexistent", 42.0)
        assert result is None

    def test_directional_drift(self):
        sentinel = DriftSentinel(
            metrics={
                "test.above": DriftConfig(window=40, sigma_threshold=2.0, direction="above"),
            }
        )
        # Warm up
        for _ in range(30):
            sentinel.record_metric("test.above", 100.0)

        # Below baseline — should NOT flag (direction="above")
        result = sentinel.record_metric("test.above", 0.01)
        assert result is None

        # Above baseline — SHOULD flag
        result = sentinel.record_metric("test.above", 500.0)
        assert result is not None

    def test_baselines_property(self):
        sentinel = DriftSentinel()
        baselines = sentinel.baselines
        assert isinstance(baselines, dict)
        for _name, info in baselines.items():
            assert "mean" in info
            assert "std" in info
            assert "warmed_up" in info


# ─── CognitiveStallSentinel ──────────────────────────────────────


class TestCognitiveStallSentinel:
    def test_no_stall_with_active_cycles(self):
        sentinel = CognitiveStallSentinel()
        incidents: list = []
        for _ in range(200):
            result = sentinel.record_cycle(
                had_broadcast=True,
                nova_had_intent=True,
                evo_had_evidence=True,
                atune_had_percept=True,
            )
            incidents.extend(result)
        assert len(incidents) == 0

    def test_stall_detected_with_empty_cycles(self):
        sentinel = CognitiveStallSentinel(
            thresholds={
                "broadcast_ack_rate": StallConfig(min_value=0.3, window_cycles=10),
            }
        )
        incidents: list = []
        for _ in range(20):
            result = sentinel.record_cycle(
                had_broadcast=False,
                nova_had_intent=False,
                evo_had_evidence=False,
                atune_had_percept=False,
            )
            incidents.extend(result)

        assert len(incidents) > 0
        assert incidents[0].incident_class == IncidentClass.COGNITIVE_STALL

    def test_partial_stall(self):
        """Only broadcast stalls, others active."""
        sentinel = CognitiveStallSentinel(
            thresholds={
                "broadcast_ack_rate": StallConfig(min_value=0.3, window_cycles=10),
                "nova_intent_rate": StallConfig(min_value=0.01, window_cycles=10),
            }
        )
        incidents: list = []
        for _ in range(15):
            result = sentinel.record_cycle(
                had_broadcast=False,
                nova_had_intent=True,
                evo_had_evidence=False,
                atune_had_percept=False,
            )
            incidents.extend(result)

        # Broadcast stall should fire, but not nova
        stall_names = [i.context["metric_name"] for i in incidents]
        assert "broadcast_ack_rate" in stall_names
