"""
Unit tests for Simula types.

Verifies the type taxonomy, iron rules, category sets, and model construction.
"""

from __future__ import annotations

from systems.simula.evolution_types import (
    FORBIDDEN,
    FORBIDDEN_WRITE_PATHS,
    GOVERNANCE_REQUIRED,
    SELF_APPLICABLE,
    SIMULA_IRON_RULES,
    ChangeCategory,
    ChangeSpec,
    CodeChangeResult,
    ConfigSnapshot,
    EvolutionProposal,
    FileSnapshot,
    ProposalResult,
    ProposalStatus,
    RiskLevel,
    SimulationResult,
)

# ─── Category Taxonomy ────────────────────────────────────────────────────────


class TestChangeCategory:
    def test_all_categories_accounted_for(self):
        """Every category must be in exactly one of the three sets."""
        all_categories = set(ChangeCategory)
        classified = SELF_APPLICABLE | GOVERNANCE_REQUIRED | FORBIDDEN
        assert all_categories == classified

    def test_sets_are_disjoint(self):
        """No category can be in multiple sets."""
        assert frozenset() == SELF_APPLICABLE & GOVERNANCE_REQUIRED
        assert frozenset() == SELF_APPLICABLE & FORBIDDEN
        assert frozenset() == GOVERNANCE_REQUIRED & FORBIDDEN

    def test_self_applicable_categories(self):
        assert ChangeCategory.ADD_EXECUTOR in SELF_APPLICABLE
        assert ChangeCategory.ADD_INPUT_CHANNEL in SELF_APPLICABLE
        assert ChangeCategory.ADD_PATTERN_DETECTOR in SELF_APPLICABLE
        assert ChangeCategory.ADJUST_BUDGET in SELF_APPLICABLE

    def test_governance_required_categories(self):
        assert ChangeCategory.MODIFY_CONTRACT in GOVERNANCE_REQUIRED
        assert ChangeCategory.ADD_SYSTEM_CAPABILITY in GOVERNANCE_REQUIRED
        assert ChangeCategory.MODIFY_CYCLE_TIMING in GOVERNANCE_REQUIRED
        assert ChangeCategory.CHANGE_CONSOLIDATION in GOVERNANCE_REQUIRED

    def test_forbidden_categories(self):
        assert ChangeCategory.MODIFY_EQUOR in FORBIDDEN
        assert ChangeCategory.MODIFY_CONSTITUTION in FORBIDDEN
        assert ChangeCategory.MODIFY_INVARIANTS in FORBIDDEN
        assert ChangeCategory.MODIFY_SELF_EVOLUTION in FORBIDDEN


# ─── Iron Rules ───────────────────────────────────────────────────────────────


class TestIronRules:
    def test_all_nine_rules_present(self):
        assert len(SIMULA_IRON_RULES) == 9

    def test_equor_protection(self):
        assert any("Equor" in r for r in SIMULA_IRON_RULES)

    def test_self_modification_forbidden(self):
        assert any("own logic" in r for r in SIMULA_IRON_RULES)

    def test_rollback_required(self):
        assert any("rollback" in r for r in SIMULA_IRON_RULES)

    def test_simulate_before_applying(self):
        assert any("simulate" in r.lower() for r in SIMULA_IRON_RULES)


# ─── Forbidden Write Paths ────────────────────────────────────────────────────


class TestForbiddenPaths:
    def test_equor_forbidden(self):
        assert any("equor" in p for p in FORBIDDEN_WRITE_PATHS)

    def test_simula_forbidden(self):
        assert any("simula" in p for p in FORBIDDEN_WRITE_PATHS)

    def test_constitutional_forbidden(self):
        assert any("constitutional" in p for p in FORBIDDEN_WRITE_PATHS)


# ─── Model Construction ──────────────────────────────────────────────────────


class TestChangeSpec:
    def test_empty_spec(self):
        spec = ChangeSpec()
        assert spec.executor_name is None
        assert spec.additional_context == ""
        assert spec.affected_systems == []

    def test_executor_spec(self):
        spec = ChangeSpec(
            executor_name="weather_lookup",
            executor_description="Look up weather data",
            executor_action_type="weather",
        )
        assert spec.executor_name == "weather_lookup"


class TestEvolutionProposal:
    def test_defaults(self):
        spec = ChangeSpec()
        proposal = EvolutionProposal(
            source="evo",
            category=ChangeCategory.ADD_EXECUTOR,
            description="Add weather executor",
            change_spec=spec,
        )
        assert proposal.status == ProposalStatus.PROPOSED
        assert proposal.simulation is None
        assert proposal.evidence == []
        assert proposal.id  # ULID auto-generated

    def test_full_proposal(self):
        proposal = EvolutionProposal(
            source="governance",
            category=ChangeCategory.MODIFY_CONTRACT,
            description="Update Axon-Nova contract",
            change_spec=ChangeSpec(
                contract_changes=["Add timeout field to Intent"],
                affected_systems=["axon", "nova"],
            ),
            evidence=["hyp_001", "hyp_002"],
            expected_benefit="Reduces timeout failures by 30%",
            risk_assessment="Low risk — additive change only",
        )
        assert proposal.category == ChangeCategory.MODIFY_CONTRACT
        assert len(proposal.evidence) == 2


class TestSimulationResult:
    def test_default_low_risk(self):
        result = SimulationResult()
        assert result.risk_level == RiskLevel.LOW
        assert result.episodes_tested == 0

    def test_custom_risk(self):
        result = SimulationResult(
            episodes_tested=200,
            regressions=25,
            risk_level=RiskLevel.UNACCEPTABLE,
            risk_summary="25 regressions out of 200 episodes",
        )
        assert result.risk_level == RiskLevel.UNACCEPTABLE


class TestProposalResult:
    def test_approved(self):
        result = ProposalResult(
            status=ProposalStatus.APPLIED,
            version=5,
            files_changed=["src/systems/axon/executors/weather.py"],
        )
        assert result.version == 5
        assert len(result.files_changed) == 1

    def test_rejected(self):
        result = ProposalResult(
            status=ProposalStatus.REJECTED,
            reason="Forbidden category",
        )
        assert result.reason == "Forbidden category"


class TestConfigSnapshot:
    def test_file_snapshots(self):
        snap = ConfigSnapshot(
            proposal_id="prop_001",
            files=[
                FileSnapshot(path="/a.py", content="old content", existed=True),
                FileSnapshot(path="/b.py", content=None, existed=False),
            ],
            config_version=3,
        )
        assert len(snap.files) == 2
        assert snap.files[1].existed is False


class TestCodeChangeResult:
    def test_success(self):
        result = CodeChangeResult(
            success=True,
            files_written=["src/foo.py"],
            summary="Added foo",
        )
        assert result.lint_passed is True
        assert result.tests_passed is True

    def test_failure(self):
        result = CodeChangeResult(
            success=False,
            error="Syntax error in generated code",
        )
        assert not result.success
