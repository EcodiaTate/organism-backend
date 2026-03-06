"""
Tests for Thymos Healing Governor.

Covers:
  - Diagnosis throttling
  - Codegen slot control
  - Storm mode detection and exit
  - Budget tracking (repairs per hour, novel per day)
"""

from __future__ import annotations

from primitives.common import new_id, utc_now
from systems.thymos.governor import HealingGovernor
from systems.thymos.types import (
    HealingMode,
    Incident,
    IncidentClass,
    IncidentSeverity,
    RepairTier,
)


def _make_incident(
    source_system: str = "nova",
    incident_id: str | None = None,
) -> Incident:
    return Incident(
        id=incident_id or new_id(),
        timestamp=utc_now(),
        incident_class=IncidentClass.CRASH,
        severity=IncidentSeverity.HIGH,
        fingerprint=f"gov_{new_id()[:8]}",
        source_system=source_system,
        error_type="RuntimeError",
        error_message="test error",
    )


class TestHealingGovernor:
    def test_starts_in_nominal_mode(self):
        gov = HealingGovernor()
        assert gov.healing_mode == HealingMode.NOMINAL

    def test_should_diagnose_when_under_limit(self):
        gov = HealingGovernor()
        incident = _make_incident()
        assert gov.should_diagnose(incident) is True

    def test_should_diagnose_denied_at_limit(self):
        gov = HealingGovernor()
        gov._active_diagnoses = 3
        incident = _make_incident()
        assert gov.should_diagnose(incident) is False

    def test_begin_end_diagnosis(self):
        gov = HealingGovernor()
        gov.begin_diagnosis()
        assert gov._active_diagnoses == 1
        gov.end_diagnosis()
        assert gov._active_diagnoses == 0

    def test_end_diagnosis_clamps_to_zero(self):
        gov = HealingGovernor()
        gov.end_diagnosis()
        assert gov._active_diagnoses == 0

    def test_should_codegen_when_under_limit(self):
        gov = HealingGovernor()
        assert gov.should_codegen() is True

    def test_should_codegen_denied_at_limit(self):
        gov = HealingGovernor()
        gov._active_codegen = 1
        assert gov.should_codegen() is False

    def test_begin_end_codegen(self):
        gov = HealingGovernor()
        gov.begin_codegen()
        assert gov._active_codegen == 1
        gov.end_codegen()
        assert gov._active_codegen == 0

    def test_storm_mode_entered_on_flood(self):
        gov = HealingGovernor()
        # Register 11 incidents (above threshold of 10)
        for _ in range(11):
            incident = _make_incident()
            gov.register_incident(incident)

        # Now try to diagnose — should enter storm mode
        next_incident = _make_incident()
        gov.should_diagnose(next_incident)
        assert gov.healing_mode == HealingMode.STORM

    def test_storm_focuses_on_most_common_system(self):
        gov = HealingGovernor()
        # 8 incidents from nova, 3 from memory
        for _ in range(8):
            gov.register_incident(_make_incident(source_system="nova"))
        for _ in range(3):
            gov.register_incident(_make_incident(source_system="memory"))

        # Trigger storm check
        gov.should_diagnose(_make_incident())
        assert gov.storm_focus_system == "nova"

    def test_storm_only_diagnoses_first_per_system(self):
        gov = HealingGovernor()
        # Flood to trigger storm
        for _ in range(11):
            gov.register_incident(_make_incident(source_system="nova"))

        # First nova incident should be diagnosable
        inc1 = _make_incident(source_system="nova")
        result1 = gov.should_diagnose(inc1)

        # Second nova incident should be blocked
        inc2 = _make_incident(source_system="nova")
        result2 = gov.should_diagnose(inc2)

        assert result1 is True
        assert result2 is False

    def test_storm_exit_when_rate_drops(self):
        gov = HealingGovernor()
        # Enter storm by registering many
        for _ in range(11):
            gov.register_incident(_make_incident())
        gov.should_diagnose(_make_incident())
        assert gov.healing_mode == HealingMode.STORM

        # Clear the recent incidents to simulate time passing
        gov._recent_incident_times.clear()
        exited = gov.check_storm_exit()
        assert exited is True
        assert gov.healing_mode == HealingMode.NOMINAL

    def test_check_storm_exit_noop_when_not_storm(self):
        gov = HealingGovernor()
        assert gov.check_storm_exit() is False

    def test_record_repair_increments_counter(self):
        gov = HealingGovernor()
        gov.record_repair(RepairTier.PARAMETER)
        assert gov._repairs_this_hour == 1

    def test_record_novel_repair_increments_both(self):
        gov = HealingGovernor()
        gov.record_repair(RepairTier.NOVEL_FIX)
        assert gov._repairs_this_hour == 1
        assert gov._novel_repairs_today == 1

    def test_resolve_incident(self):
        gov = HealingGovernor()
        inc = _make_incident(incident_id="resolve_test")
        gov.register_incident(inc)
        assert "resolve_test" in gov._active_incidents
        gov.resolve_incident("resolve_test")
        assert "resolve_test" not in gov._active_incidents

    def test_budget_state_snapshot(self):
        gov = HealingGovernor()
        gov.record_repair(RepairTier.PARAMETER)
        gov.begin_diagnosis()
        budget = gov.budget_state
        assert budget.repairs_this_hour == 1
        assert budget.active_diagnoses == 1
        assert budget.storm_mode is False

    def test_storm_activations_counter(self):
        gov = HealingGovernor()
        assert gov.storm_activations == 0
        # Trigger storm
        for _ in range(11):
            gov.register_incident(_make_incident())
        gov.should_diagnose(_make_incident())
        assert gov.storm_activations == 1

    # ─── T4 Budget Tests (Cytokine Storm Prevention) ───────────────────────

    def test_can_submit_t4_when_under_limit(self):
        gov = HealingGovernor()
        assert gov.can_submit_t4_proposal() is True

    def test_can_submit_t4_denied_at_concurrent_limit(self):
        gov = HealingGovernor()
        # Use up the 3 concurrent slots
        gov.begin_t4_proposal()
        gov.begin_t4_proposal()
        gov.begin_t4_proposal()
        assert gov.can_submit_t4_proposal() is False

    def test_can_submit_t4_denied_at_hourly_limit(self):
        gov = HealingGovernor()
        # Use up all 5 hourly slots
        for _ in range(5):
            gov.begin_t4_proposal()
        assert gov._t4_proposals_this_hour == 5
        assert gov.can_submit_t4_proposal() is False

    def test_begin_end_t4_proposal(self):
        gov = HealingGovernor()
        gov.begin_t4_proposal()
        assert gov._active_t4_proposal_count == 1
        assert gov._t4_proposals_this_hour == 1
        gov.end_t4_proposal()
        assert gov._active_t4_proposal_count == 0

    def test_end_t4_clamps_to_zero(self):
        gov = HealingGovernor()
        gov.end_t4_proposal()
        assert gov._active_t4_proposal_count == 0

    def test_t4_budget_state_reporting(self):
        gov = HealingGovernor()
        gov.begin_t4_proposal()
        gov.begin_t4_proposal()
        budget = gov.budget_state
        assert budget.active_t4_proposals == 2
        assert budget.t4_proposals_this_hour == 2
        assert budget.max_concurrent_t4_proposals == 3
        assert budget.max_t4_proposals_per_hour == 5
