"""
Unit tests for Stage 1A - Extended-thinking model routing in SimulaCodeAgent.

Tests:
  - Budget guard routes to thinking model only for governance/high-risk
  - Standard model used for routine additive changes
  - No thinking provider → always standard model
  - KV compression metrics tracked correctly
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from systems.simula.code_agent import SimulaCodeAgent
from systems.simula.evolution_types import (
    ChangeCategory,
    ChangeSpec,
    EvolutionProposal,
    RiskLevel,
    SimulationResult,
)

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_proposal(
    category: ChangeCategory = ChangeCategory.ADD_EXECUTOR,
    risk_level: RiskLevel | None = None,
    **spec_kwargs,
) -> EvolutionProposal:
    spec = ChangeSpec(**spec_kwargs)
    proposal = EvolutionProposal(
        source="evo",
        category=category,
        description="Test proposal for routing",
        change_spec=spec,
    )
    if risk_level is not None:
        proposal.simulation = SimulationResult(
            risk_level=risk_level,
            risk_summary=f"Test simulation risk: {risk_level.value}",
        )
    return proposal


def make_mock_llm() -> MagicMock:
    llm = MagicMock()
    response = MagicMock()
    response.text = "Implementation complete."
    response.has_tool_calls = False
    response.stop_reason = "end_turn"
    response.total_tokens = 100
    response.tool_calls = []
    llm.generate_with_tools = AsyncMock(return_value=response)
    return llm


def make_mock_thinking_llm() -> MagicMock:
    thinking = MagicMock()
    response = MagicMock()
    response.text = "Implementation complete (with thinking)."
    response.has_tool_calls = False
    response.stop_reason = "end_turn"
    response.total_tokens = 200
    response.tool_calls = []
    response.reasoning_tokens = 150
    thinking.generate_with_thinking_and_tools = AsyncMock(return_value=response)
    return thinking


# ─── Model Routing Tests ─────────────────────────────────────────────────────


class TestExtendedThinkingRouting:
    """Verify _should_use_extended_thinking budget guard."""

    def test_no_thinking_provider_always_false(self):
        agent = SimulaCodeAgent(
            llm=make_mock_llm(),
            codebase_root=Path("/tmp"),
            thinking_provider=None,
        )
        proposal = make_proposal(category=ChangeCategory.ADD_EXECUTOR)
        assert agent._should_use_extended_thinking(proposal) is False

    def test_governance_required_uses_thinking(self):
        agent = SimulaCodeAgent(
            llm=make_mock_llm(),
            codebase_root=Path("/tmp"),
            thinking_provider=make_mock_thinking_llm(),
        )
        # MODIFY_CONSTITUTIONAL is in GOVERNANCE_REQUIRED
        proposal = make_proposal(category=ChangeCategory.MODIFY_CONSTITUTIONAL)
        assert agent._should_use_extended_thinking(proposal) is True

    def test_modify_invariant_uses_thinking(self):
        agent = SimulaCodeAgent(
            llm=make_mock_llm(),
            codebase_root=Path("/tmp"),
            thinking_provider=make_mock_thinking_llm(),
        )
        proposal = make_proposal(category=ChangeCategory.MODIFY_INVARIANT)
        assert agent._should_use_extended_thinking(proposal) is True

    def test_high_risk_uses_thinking(self):
        agent = SimulaCodeAgent(
            llm=make_mock_llm(),
            codebase_root=Path("/tmp"),
            thinking_provider=make_mock_thinking_llm(),
        )
        proposal = make_proposal(
            category=ChangeCategory.ADD_EXECUTOR,
            risk_level=RiskLevel.HIGH,
        )
        assert agent._should_use_extended_thinking(proposal) is True

    def test_unacceptable_risk_uses_thinking(self):
        agent = SimulaCodeAgent(
            llm=make_mock_llm(),
            codebase_root=Path("/tmp"),
            thinking_provider=make_mock_thinking_llm(),
        )
        proposal = make_proposal(
            category=ChangeCategory.ADD_EXECUTOR,
            risk_level=RiskLevel.UNACCEPTABLE,
        )
        assert agent._should_use_extended_thinking(proposal) is True

    def test_low_risk_additive_uses_standard(self):
        agent = SimulaCodeAgent(
            llm=make_mock_llm(),
            codebase_root=Path("/tmp"),
            thinking_provider=make_mock_thinking_llm(),
        )
        proposal = make_proposal(
            category=ChangeCategory.ADD_EXECUTOR,
            risk_level=RiskLevel.LOW,
        )
        assert agent._should_use_extended_thinking(proposal) is False

    def test_moderate_risk_uses_standard(self):
        agent = SimulaCodeAgent(
            llm=make_mock_llm(),
            codebase_root=Path("/tmp"),
            thinking_provider=make_mock_thinking_llm(),
        )
        proposal = make_proposal(
            category=ChangeCategory.ADD_EXECUTOR,
            risk_level=RiskLevel.MODERATE,
        )
        assert agent._should_use_extended_thinking(proposal) is False

    def test_no_simulation_result_additive_uses_standard(self):
        """Proposals without simulation results default to standard model."""
        agent = SimulaCodeAgent(
            llm=make_mock_llm(),
            codebase_root=Path("/tmp"),
            thinking_provider=make_mock_thinking_llm(),
        )
        proposal = make_proposal(category=ChangeCategory.ADD_EXECUTOR)
        assert proposal.simulation is None
        assert agent._should_use_extended_thinking(proposal) is False


class TestCodeAgentImplementRouting:
    """Verify that implement() actually routes to the correct LLM."""

    @pytest.mark.asyncio
    async def test_standard_model_called_for_low_risk(self):
        standard_llm = make_mock_llm()
        thinking_llm = make_mock_thinking_llm()

        agent = SimulaCodeAgent(
            llm=standard_llm,
            codebase_root=Path("/tmp"),
            thinking_provider=thinking_llm,
        )

        proposal = make_proposal(
            category=ChangeCategory.ADD_EXECUTOR,
            risk_level=RiskLevel.LOW,
        )
        result = await agent.implement(proposal)

        # Standard LLM should have been called
        standard_llm.generate_with_tools.assert_called()
        # Thinking LLM should NOT have been called
        thinking_llm.generate_with_thinking_and_tools.assert_not_called()
        assert result.used_extended_thinking is False

    @pytest.mark.asyncio
    async def test_thinking_model_called_for_governance(self):
        standard_llm = make_mock_llm()
        thinking_llm = make_mock_thinking_llm()

        agent = SimulaCodeAgent(
            llm=standard_llm,
            codebase_root=Path("/tmp"),
            thinking_provider=thinking_llm,
        )

        proposal = make_proposal(category=ChangeCategory.MODIFY_CONSTITUTIONAL)
        result = await agent.implement(proposal)

        # Thinking LLM should have been called
        thinking_llm.generate_with_thinking_and_tools.assert_called()
        # Standard LLM should NOT have been called
        standard_llm.generate_with_tools.assert_not_called()
        assert result.used_extended_thinking is True

    @pytest.mark.asyncio
    async def test_kv_compression_metrics_returned(self):
        agent = SimulaCodeAgent(
            llm=make_mock_llm(),
            codebase_root=Path("/tmp"),
            kv_compression_ratio=0.3,
            kv_compression_enabled=True,
        )

        proposal = make_proposal(
            category=ChangeCategory.ADD_EXECUTOR,
            risk_level=RiskLevel.LOW,
        )
        result = await agent.implement(proposal)

        # Metrics should be present (may be 0 if only 1 turn)
        assert hasattr(result, "kv_compression_ratio")
        assert hasattr(result, "kv_messages_compressed")
        assert hasattr(result, "kv_original_tokens")
        assert hasattr(result, "kv_compressed_tokens")
