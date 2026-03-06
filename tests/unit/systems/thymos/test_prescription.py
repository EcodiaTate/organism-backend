"""
Tests for Thymos Prescription Layer.

Covers:
  - RepairPrescriber
  - RepairValidator
"""

from __future__ import annotations

import pytest

from primitives.common import new_id, utc_now
from systems.thymos.prescription import RepairPrescriber, RepairValidator
from systems.thymos.types import (
    Diagnosis,
    Incident,
    IncidentClass,
    IncidentSeverity,
    RepairSpec,
    RepairTier,
)


def _make_incident(
    severity: IncidentSeverity = IncidentSeverity.MEDIUM,
    occurrence_count: int = 1,
    error_type: str = "RuntimeError",
    blast_radius: float = 0.2,
    stack_trace: str | None = "File test.py, line 1",
) -> Incident:
    return Incident(
        id=new_id(),
        timestamp=utc_now(),
        incident_class=IncidentClass.CRASH,
        severity=severity,
        fingerprint="prescribe_fp",
        source_system="nova",
        error_type=error_type,
        error_message="test error",
        blast_radius=blast_radius,
        occurrence_count=occurrence_count,
        stack_trace=stack_trace,
    )


def _make_diagnosis(
    root_cause: str = "memory_pressure",
    confidence: float = 0.8,
    repair_tier: RepairTier = RepairTier.PARAMETER,
    antibody_id: str | None = None,
) -> Diagnosis:
    return Diagnosis(
        root_cause=root_cause,
        confidence=confidence,
        repair_tier=repair_tier,
        antibody_id=antibody_id,
    )


# ─── RepairPrescriber ─────────────────────────────────────────────


class TestRepairPrescriber:
    @pytest.mark.asyncio
    async def test_transient_single_gets_noop(self):
        prescriber = RepairPrescriber()
        incident = _make_incident(
            occurrence_count=1,
            error_type="TimeoutError",
        )
        diagnosis = _make_diagnosis(root_cause="transient timeout")
        repair = await prescriber.prescribe(incident, diagnosis)
        assert repair.tier == RepairTier.NOOP

    @pytest.mark.asyncio
    async def test_antibody_match_gets_known_fix(self):
        prescriber = RepairPrescriber()
        incident = _make_incident()
        diagnosis = _make_diagnosis(antibody_id="ab_123")
        repair = await prescriber.prescribe(incident, diagnosis)
        assert repair.tier == RepairTier.KNOWN_FIX
        assert repair.antibody_id == "ab_123"

    @pytest.mark.asyncio
    async def test_memory_pressure_gets_parameter(self):
        prescriber = RepairPrescriber()
        incident = _make_incident()
        diagnosis = _make_diagnosis(root_cause="memory_pressure causing issues")
        repair = await prescriber.prescribe(incident, diagnosis)
        assert repair.tier == RepairTier.PARAMETER
        assert repair.action == "adjust_parameters"
        assert len(repair.parameter_changes) > 0

    @pytest.mark.asyncio
    async def test_state_corruption_gets_restart(self):
        prescriber = RepairPrescriber()
        incident = _make_incident()
        diagnosis = _make_diagnosis(root_cause="state_corruption in memory store")
        repair = await prescriber.prescribe(incident, diagnosis)
        assert repair.tier == RepairTier.RESTART

    @pytest.mark.asyncio
    async def test_unknown_cause_high_severity_gets_restart(self):
        prescriber = RepairPrescriber()
        incident = _make_incident(severity=IncidentSeverity.CRITICAL)
        diagnosis = _make_diagnosis(root_cause="unknown issue xyz", confidence=0.3)
        repair = await prescriber.prescribe(incident, diagnosis)
        assert repair.tier == RepairTier.RESTART

    @pytest.mark.asyncio
    async def test_unknown_cause_low_severity_gets_escalate(self):
        prescriber = RepairPrescriber()
        incident = _make_incident(severity=IncidentSeverity.LOW)
        diagnosis = _make_diagnosis(root_cause="unknown xyz", confidence=0.3)
        repair = await prescriber.prescribe(incident, diagnosis)
        assert repair.tier == RepairTier.ESCALATE

    @pytest.mark.asyncio
    async def test_high_confidence_with_stack_gets_novel_fix(self):
        prescriber = RepairPrescriber()
        incident = _make_incident(
            severity=IncidentSeverity.HIGH,
            blast_radius=0.3,
            stack_trace="File nova.py, line 42, in deliberate",
        )
        diagnosis = _make_diagnosis(
            root_cause="novel bug in deliberation pipeline",
            confidence=0.9,
        )
        repair = await prescriber.prescribe(incident, diagnosis)
        assert repair.tier == RepairTier.NOVEL_FIX

    @pytest.mark.asyncio
    async def test_no_codegen_for_low_severity(self):
        prescriber = RepairPrescriber()
        incident = _make_incident(severity=IncidentSeverity.LOW)
        diagnosis = _make_diagnosis(
            root_cause="novel bug",
            confidence=0.9,
        )
        repair = await prescriber.prescribe(incident, diagnosis)
        # Should NOT be novel fix for low severity
        assert repair.tier != RepairTier.NOVEL_FIX

    @pytest.mark.asyncio
    async def test_no_codegen_for_high_blast_radius(self):
        prescriber = RepairPrescriber()
        incident = _make_incident(
            severity=IncidentSeverity.HIGH,
            blast_radius=0.8,
            stack_trace="...",
        )
        diagnosis = _make_diagnosis(
            root_cause="novel wide-impact bug",
            confidence=0.9,
        )
        repair = await prescriber.prescribe(incident, diagnosis)
        assert repair.tier != RepairTier.NOVEL_FIX


# ─── RepairValidator ──────────────────────────────────────────────


class TestRepairValidator:
    @pytest.mark.asyncio
    async def test_low_tier_passes_without_equor(self):
        validator = RepairValidator(equor=None)
        incident = _make_incident()
        repair = RepairSpec(
            tier=RepairTier.PARAMETER,
            action="adjust_parameters",
            reason="test",
        )
        result = await validator.validate(incident, repair)
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_high_blast_radius_rejected(self):
        validator = RepairValidator(equor=None)
        incident = _make_incident(blast_radius=0.9)
        repair = RepairSpec(
            tier=RepairTier.KNOWN_FIX,
            action="apply_antibody",
            reason="test",
        )
        result = await validator.validate(incident, repair)
        assert result.approved is False
        assert result.escalate_to == RepairTier.ESCALATE

    @pytest.mark.asyncio
    async def test_noop_always_approved(self):
        validator = RepairValidator()
        incident = _make_incident()
        repair = RepairSpec(
            tier=RepairTier.NOOP,
            action="log_and_monitor",
            reason="transient",
        )
        result = await validator.validate(incident, repair)
        assert result.approved is True
