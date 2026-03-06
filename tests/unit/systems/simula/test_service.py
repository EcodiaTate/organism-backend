"""
Unit tests for SimulaService.

Tests the top-level service interface: proposal validation, simulation gating,
governance routing, and the full apply-verify-record pipeline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import SimulaConfig
from systems.simula.evolution_types import (
    ChangeCategory,
    ChangeSpec,
    CodeChangeResult,
    ConfigSnapshot,
    EvolutionProposal,
    HealthCheckResult,
    ProposalStatus,
    RiskLevel,
    SimulationResult,
)
from systems.simula.service import SimulaService

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_config(**kwargs) -> SimulaConfig:
    defaults = {
        "max_simulation_episodes": 200,
        "regression_threshold_unacceptable": 0.10,
        "regression_threshold_high": 0.05,
        "codebase_root": "/tmp/test_eos",
        "max_code_agent_turns": 5,
        "test_command": "pytest",
        "auto_apply_self_applicable": True,
    }
    return SimulaConfig(**{**defaults, **kwargs})


def make_llm() -> MagicMock:
    llm = MagicMock()
    response = MagicMock()
    response.text = "RISK: LOW\nREASONING: No issues.\nBENEFIT: Improved capability."
    llm.generate = AsyncMock(return_value=response)
    llm.evaluate = AsyncMock(return_value=response)
    llm.generate_with_tools = AsyncMock(return_value=MagicMock(
        text="Done",
        tool_calls=[],
        has_tool_calls=False,
        stop_reason="end_turn",
    ))
    return llm


def make_proposal(
    category: ChangeCategory = ChangeCategory.ADD_EXECUTOR,
    source: str = "evo",
) -> EvolutionProposal:
    return EvolutionProposal(
        source=source,
        category=category,
        description="Test proposal",
        change_spec=ChangeSpec(
            executor_name="test_executor",
            executor_description="A test executor",
            executor_action_type="test_action",
        ),
        expected_benefit="Testing",
    )


async def make_service(config: SimulaConfig | None = None) -> SimulaService:
    """Build a SimulaService with all subsystems mocked."""
    config = config or make_config()
    llm = make_llm()
    service = SimulaService(
        config=config,
        llm=llm,
        neo4j=None,
        memory=None,
        codebase_root=Path("/tmp/test_eos"),
    )

    # Mock sub-systems to avoid filesystem / network calls
    service._simulator = MagicMock()
    service._simulator.simulate = AsyncMock(return_value=SimulationResult(
        risk_level=RiskLevel.LOW,
        episodes_tested=0,
    ))

    service._applicator = MagicMock()
    service._applicator.apply = AsyncMock(return_value=(
        CodeChangeResult(
            success=True,
            files_written=["src/test.py"],
            summary="Applied",
        ),
        ConfigSnapshot(proposal_id="test", config_version=0),
    ))

    service._health = MagicMock()
    service._health.check = AsyncMock(return_value=HealthCheckResult(healthy=True))

    service._rollback = MagicMock()
    service._rollback.restore = AsyncMock(return_value=[])

    service._initialized = True
    service._current_version = 0
    return service


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestProposalValidation:
    @pytest.mark.asyncio
    async def test_forbidden_category_rejected(self):
        service = await make_service()
        proposal = make_proposal(category=ChangeCategory.MODIFY_EQUOR)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.REJECTED
        assert "forbidden" in result.reason.lower()
        assert service._proposals_rejected == 1

    @pytest.mark.asyncio
    async def test_modify_constitution_rejected(self):
        service = await make_service()
        proposal = make_proposal(category=ChangeCategory.MODIFY_CONSTITUTION)
        result = await service.process_proposal(proposal)
        assert result.status == ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_modify_invariants_rejected(self):
        service = await make_service()
        proposal = make_proposal(category=ChangeCategory.MODIFY_INVARIANTS)
        result = await service.process_proposal(proposal)
        assert result.status == ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_modify_self_evolution_rejected(self):
        service = await make_service()
        proposal = make_proposal(category=ChangeCategory.MODIFY_SELF_EVOLUTION)
        result = await service.process_proposal(proposal)
        assert result.status == ProposalStatus.REJECTED


class TestGovernanceGating:
    @pytest.mark.asyncio
    async def test_governance_required_routes_to_governance(self):
        service = await make_service()
        proposal = make_proposal(category=ChangeCategory.MODIFY_CONTRACT)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.AWAITING_GOVERNANCE
        assert result.governance_record_id is not None
        assert service._proposals_awaiting_governance == 1

    @pytest.mark.asyncio
    async def test_self_applicable_skips_governance(self):
        service = await make_service()
        proposal = make_proposal(category=ChangeCategory.ADD_EXECUTOR)
        result = await service.process_proposal(proposal)

        assert result.status != ProposalStatus.AWAITING_GOVERNANCE

    def test_requires_governance_check(self):
        service = SimulaService(
            config=make_config(),
            llm=make_llm(),
        )
        assert service.requires_governance(
            make_proposal(category=ChangeCategory.MODIFY_CONTRACT)
        )
        assert service.requires_governance(
            make_proposal(category=ChangeCategory.ADD_SYSTEM_CAPABILITY)
        )
        assert not service.requires_governance(
            make_proposal(category=ChangeCategory.ADD_EXECUTOR)
        )


class TestSimulationGating:
    @pytest.mark.asyncio
    async def test_unacceptable_risk_rejected(self):
        service = await make_service()
        service._simulator.simulate = AsyncMock(return_value=SimulationResult(
            risk_level=RiskLevel.UNACCEPTABLE,
            risk_summary="Too many regressions",
        ))
        proposal = make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.REJECTED
        assert "unacceptable" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_low_risk_proceeds(self):
        service = await make_service()
        proposal = make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.APPLIED


class TestApplyPipeline:
    @pytest.mark.asyncio
    async def test_successful_apply_increments_version(self):
        service = await make_service()
        assert service._current_version == 0

        proposal = make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.APPLIED
        assert result.version == 1
        assert service._current_version == 1
        assert service._proposals_approved == 1

    @pytest.mark.asyncio
    async def test_health_check_failure_triggers_rollback(self):
        service = await make_service()
        service._health.check = AsyncMock(return_value=HealthCheckResult(
            healthy=False,
            issues=["Syntax error in generated code"],
        ))

        proposal = make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.ROLLED_BACK
        assert "health check" in result.reason.lower()
        service._rollback.restore.assert_called_once()
        assert service._proposals_rolled_back == 1

    @pytest.mark.asyncio
    async def test_apply_failure_rolls_back(self):
        service = await make_service()
        service._applicator.apply = AsyncMock(return_value=(
            CodeChangeResult(success=False, error="Code agent failed"),
            ConfigSnapshot(proposal_id="test", config_version=0),
        ))

        proposal = make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.ROLLED_BACK
        assert service._proposals_rolled_back == 1

    @pytest.mark.asyncio
    async def test_files_changed_in_result(self):
        service = await make_service()
        proposal = make_proposal()
        result = await service.process_proposal(proposal)

        assert result.files_changed == ["src/test.py"]


class TestGovernedApproval:
    @pytest.mark.asyncio
    async def test_approve_governed_proposal(self):
        service = await make_service()

        # First, create a governed proposal
        proposal = make_proposal(category=ChangeCategory.MODIFY_CONTRACT)
        result = await service.process_proposal(proposal)
        assert result.status == ProposalStatus.AWAITING_GOVERNANCE

        # Now approve it
        approval = await service.approve_governed_proposal(
            proposal_id=proposal.id,
            governance_record_id=result.governance_record_id or "",
        )
        assert approval.status == ProposalStatus.APPLIED

    @pytest.mark.asyncio
    async def test_approve_unknown_proposal(self):
        service = await make_service()
        result = await service.approve_governed_proposal("nonexistent", "gov_123")
        assert result.status == ProposalStatus.REJECTED
        assert "not found" in result.reason


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_structure(self):
        service = await make_service()
        stats = service.stats
        assert "initialized" in stats
        assert "current_version" in stats
        assert "proposals_received" in stats
        assert "proposals_approved" in stats
        assert "proposals_rejected" in stats
        assert "proposals_rolled_back" in stats
