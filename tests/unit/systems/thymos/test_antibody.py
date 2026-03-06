"""
Tests for Thymos Antibody Library.

Covers:
  - AntibodyLibrary (in-memory mode, no Neo4j)
  - Lookup, creation, outcome recording
  - Effectiveness tracking and retirement
"""

from __future__ import annotations

import pytest

from primitives.common import new_id, utc_now
from systems.thymos.antibody import AntibodyLibrary
from systems.thymos.types import (
    Incident,
    IncidentClass,
    IncidentSeverity,
    RepairSpec,
    RepairTier,
)


def _make_incident(fingerprint: str = "ab_test_fp") -> Incident:
    return Incident(
        id=new_id(),
        timestamp=utc_now(),
        incident_class=IncidentClass.CRASH,
        severity=IncidentSeverity.HIGH,
        fingerprint=fingerprint,
        source_system="nova",
        error_type="RuntimeError",
        error_message="test error for antibody",
    )


def _make_repair(tier: RepairTier = RepairTier.PARAMETER) -> RepairSpec:
    return RepairSpec(
        tier=tier,
        action="adjust_parameters",
        parameter_changes=[{"parameter_path": "test.param", "delta": 10, "reason": "test"}],
        reason="Test repair",
    )


class TestAntibodyLibrary:
    @pytest.mark.asyncio
    async def test_initialize_without_neo4j(self):
        library = AntibodyLibrary(neo4j_client=None)
        await library.initialize()
        assert len(library._all) == 0

    @pytest.mark.asyncio
    async def test_lookup_miss_returns_none(self):
        library = AntibodyLibrary(neo4j_client=None)
        await library.initialize()
        result = await library.lookup("nonexistent_fingerprint")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_from_repair(self):
        library = AntibodyLibrary(neo4j_client=None)
        await library.initialize()
        incident = _make_incident()
        repair = _make_repair()
        antibody = await library.create_from_repair(incident, repair)
        assert antibody.fingerprint == incident.fingerprint
        assert antibody.repair_tier == RepairTier.PARAMETER
        assert antibody.effectiveness == 1.0
        assert antibody.application_count == 0

    @pytest.mark.asyncio
    async def test_lookup_hit_after_creation(self):
        library = AntibodyLibrary(neo4j_client=None)
        await library.initialize()
        incident = _make_incident(fingerprint="findable_fp")
        repair = _make_repair()
        created = await library.create_from_repair(incident, repair)

        found = await library.lookup("findable_fp")
        assert found is not None
        assert found.id == created.id

    @pytest.mark.asyncio
    async def test_record_success_increases_effectiveness(self):
        library = AntibodyLibrary(neo4j_client=None)
        await library.initialize()
        incident = _make_incident()
        repair = _make_repair()
        antibody = await library.create_from_repair(incident, repair)

        await library.record_outcome(antibody.id, success=True)
        assert antibody.success_count == 1
        assert antibody.application_count == 1
        assert antibody.effectiveness == 1.0

    @pytest.mark.asyncio
    async def test_record_failure_decreases_effectiveness(self):
        library = AntibodyLibrary(neo4j_client=None)
        await library.initialize()
        incident = _make_incident()
        repair = _make_repair()
        antibody = await library.create_from_repair(incident, repair)

        await library.record_outcome(antibody.id, success=False)
        assert antibody.failure_count == 1
        assert antibody.effectiveness < 1.0

    @pytest.mark.asyncio
    async def test_retirement_after_sustained_failure(self):
        library = AntibodyLibrary(neo4j_client=None)
        await library.initialize()
        incident = _make_incident()
        repair = _make_repair()
        antibody = await library.create_from_repair(incident, repair)

        # Record enough failures to trigger retirement
        for _ in range(6):
            await library.record_outcome(antibody.id, success=False)

        assert antibody.retired is True

    @pytest.mark.asyncio
    async def test_retired_antibody_not_returned(self):
        library = AntibodyLibrary(neo4j_client=None)
        await library.initialize()
        incident = _make_incident(fingerprint="retire_fp")
        repair = _make_repair()
        antibody = await library.create_from_repair(incident, repair)

        # Force retirement
        antibody.retired = True

        found = await library.lookup("retire_fp")
        assert found is None

    @pytest.mark.asyncio
    async def test_unknown_antibody_outcome_ignored(self):
        library = AntibodyLibrary(neo4j_client=None)
        await library.initialize()
        # Should not raise
        await library.record_outcome("nonexistent_id", success=True)
