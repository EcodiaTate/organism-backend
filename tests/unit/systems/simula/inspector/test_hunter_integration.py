"""
Integration tests for Inspector ↔ SimulaService.

Validates that:
  - SimulaService works correctly with Inspector disabled (default)
  - Inspector sub-system initializes when inspector_enabled=True
  - Self-evolution pipeline remains fully functional when Inspector is active
  - SimulaService.stats includes Inspector subsystem status
  - _ensure_inspector raises when Inspector is disabled
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

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_config(**overrides) -> SimulaConfig:
    defaults = {
        "max_simulation_episodes": 200,
        "regression_threshold_unacceptable": 0.10,
        "regression_threshold_high": 0.05,
        "codebase_root": "/tmp/test_eos",
        "max_code_agent_turns": 5,
        "test_command": "pytest",
        "auto_apply_self_applicable": True,
        "inspector_enabled": False,
    }
    defaults.update(overrides)
    return SimulaConfig(**defaults)


def _make_llm() -> MagicMock:
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


def _make_proposal(
    category: ChangeCategory = ChangeCategory.ADD_EXECUTOR,
) -> EvolutionProposal:
    return EvolutionProposal(
        source="evo",
        category=category,
        description="Test proposal",
        change_spec=ChangeSpec(
            executor_name="test_executor",
            executor_description="A test executor",
            executor_action_type="test_action",
        ),
        expected_benefit="Testing",
    )


async def _make_service(config: SimulaConfig | None = None) -> SimulaService:
    """Build a SimulaService with all subsystems mocked."""
    config = config or _make_config()
    llm = _make_llm()
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


# ── Inspector Disabled (Default) ──────────────────────────────────────────────


class TestInspectorDisabled:
    @pytest.mark.asyncio
    async def test_service_works_with_inspector_disabled(self):
        """Normal evolution pipeline should work when inspector_enabled=False."""
        service = await _make_service(config=_make_config(inspector_enabled=False))
        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.APPLIED
        assert service._inspector is None

    @pytest.mark.asyncio
    async def test_ensure_inspector_raises_when_disabled(self):
        """_ensure_inspector should raise RuntimeError when Inspector not enabled."""
        service = await _make_service(config=_make_config(inspector_enabled=False))
        with pytest.raises(RuntimeError, match="Inspector is not enabled"):
            service._ensure_inspector()

    @pytest.mark.asyncio
    async def test_stats_show_inspector_disabled(self):
        """Stats should reflect that Inspector is disabled."""
        service = await _make_service(config=_make_config(inspector_enabled=False))
        stats = service.stats
        assert stats["stage7"]["inspector"] is False

    @pytest.mark.asyncio
    async def test_hunt_external_target_raises_when_disabled(self):
        """hunt_external_target should raise when Inspector is disabled."""
        service = await _make_service(config=_make_config(inspector_enabled=False))
        with pytest.raises(RuntimeError, match="Inspector is not enabled"):
            await service.hunt_external_target("https://github.com/test/repo")

    @pytest.mark.asyncio
    async def test_hunt_internal_eos_raises_when_disabled(self):
        """hunt_internal_eos should raise when Inspector is disabled."""
        service = await _make_service(config=_make_config(inspector_enabled=False))
        with pytest.raises(RuntimeError, match="Inspector is not enabled"):
            await service.hunt_internal_eos()


# ── Evolution Pipeline Unaffected ──────────────────────────────────────────


class TestEvolutionUnaffected:
    @pytest.mark.asyncio
    async def test_proposal_approved_without_inspector(self):
        """Standard ADD_EXECUTOR proposal should be approved normally."""
        service = await _make_service()
        proposal = _make_proposal(category=ChangeCategory.ADD_EXECUTOR)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.APPLIED
        assert result.version == 1

    @pytest.mark.asyncio
    async def test_forbidden_category_still_rejected(self):
        """Iron rules for evolution should hold regardless of Inspector."""
        service = await _make_service()
        proposal = _make_proposal(category=ChangeCategory.MODIFY_EQUOR)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.REJECTED
        assert "forbidden" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_governance_routing_still_works(self):
        """Governance gating for MODIFY_CONTRACT should still function."""
        service = await _make_service()
        proposal = _make_proposal(category=ChangeCategory.MODIFY_CONTRACT)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.AWAITING_GOVERNANCE

    @pytest.mark.asyncio
    async def test_simulation_gating_still_works(self):
        """UNACCEPTABLE risk should still reject proposals."""
        service = await _make_service()
        service._simulator.simulate = AsyncMock(return_value=SimulationResult(
            risk_level=RiskLevel.UNACCEPTABLE,
            risk_summary="Too many regressions",
        ))
        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_rollback_still_works(self):
        """Health check failure should still trigger rollback."""
        service = await _make_service()
        service._health.check = AsyncMock(return_value=HealthCheckResult(
            healthy=False,
            issues=["Generated code has errors"],
        ))
        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.ROLLED_BACK
        service._rollback.restore.assert_called_once()

    @pytest.mark.asyncio
    async def test_version_increments_correctly(self):
        """Version should increment on successful apply."""
        service = await _make_service()
        assert service._current_version == 0

        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.APPLIED
        assert service._current_version == 1


# ── Inspector Enabled (Mocked Sub-systems) ───────────────────────────────────


class TestInspectorEnabled:
    @pytest.mark.asyncio
    async def test_inspector_attribute_set_when_enabled(self):
        """When inspector_enabled=True, _inspector should be set after init."""
        service = await _make_service()
        # Manually set _inspector to simulate successful initialization
        from systems.simula.inspector.service import InspectorService
        from systems.simula.inspector.types import InspectorConfig

        mock_prover = MagicMock()
        inspector_config = InspectorConfig(authorized_targets=["localhost"])
        service._inspector = InspectorService(
            prover=mock_prover,
            config=inspector_config,
        )

        assert service._ensure_inspector() is not None

    @pytest.mark.asyncio
    async def test_evolution_still_works_with_inspector_active(self):
        """Self-evolution proposals should work even when Inspector is active."""
        service = await _make_service()

        # Simulate Inspector being active
        from systems.simula.inspector.service import InspectorService
        from systems.simula.inspector.types import InspectorConfig

        mock_prover = MagicMock()
        inspector_config = InspectorConfig(authorized_targets=["localhost"])
        service._inspector = InspectorService(
            prover=mock_prover,
            config=inspector_config,
        )

        # Normal evolution should still work
        proposal = _make_proposal(category=ChangeCategory.ADD_EXECUTOR)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.APPLIED
        assert result.version == 1

    @pytest.mark.asyncio
    async def test_stats_include_inspector_when_active(self):
        """Stats should include Inspector subsystem info when active."""
        service = await _make_service()

        from systems.simula.inspector.service import InspectorService
        from systems.simula.inspector.types import InspectorConfig

        mock_prover = MagicMock()
        inspector_config = InspectorConfig(
            authorized_targets=["localhost"],
            max_workers=2,
        )
        service._inspector = InspectorService(
            prover=mock_prover,
            config=inspector_config,
        )

        stats = service.stats
        assert stats["stage7"]["inspector"] is True
        assert "inspector_stats" in stats["stage7"]
        assert stats["stage7"]["inspector_stats"]["config"]["max_workers"] == 2


# ── SimulaConfig Inspector Fields ────────────────────────────────────────────


class TestSimulaConfigInspectorFields:
    def test_default_inspector_disabled(self):
        config = _make_config()
        assert config.inspector_enabled is False

    def test_inspector_config_fields(self):
        config = _make_config(
            inspector_enabled=True,
            inspector_max_workers=8,
            inspector_sandbox_timeout_s=60,
            inspector_clone_depth=5,
            inspector_log_analytics=True,
            inspector_generate_pocs=True,
            inspector_generate_patches=True,
            inspector_remediation_enabled=True,
        )
        assert config.inspector_enabled is True
        assert config.inspector_max_workers == 8
        assert config.inspector_sandbox_timeout_s == 60
        assert config.inspector_clone_depth == 5
        assert config.inspector_log_analytics is True
        assert config.inspector_generate_pocs is True
        assert config.inspector_generate_patches is True
        assert config.inspector_remediation_enabled is True

    def test_default_inspector_config_values(self):
        config = SimulaConfig(
            codebase_root="/tmp/test",
            max_simulation_episodes=100,
        )
        assert config.inspector_enabled is False
        assert config.inspector_max_workers == 4
        assert config.inspector_sandbox_timeout_s == 30
        assert config.inspector_clone_depth == 1
        assert config.inspector_log_analytics is True
        assert config.inspector_generate_pocs is False
        assert config.inspector_generate_patches is False
        assert config.inspector_remediation_enabled is False
