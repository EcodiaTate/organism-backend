"""
Unit tests for Simula ChangeSimulator.

Tests additive change simulation, budget validation, governance LLM simulation,
and name conflict detection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from config import SimulaConfig
from systems.simula.evolution_types import (
    ChangeCategory,
    ChangeSpec,
    EvolutionProposal,
    RiskLevel,
)
from systems.simula.simulation import ChangeSimulator

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_config(**kwargs) -> SimulaConfig:
    defaults = {
        "max_simulation_episodes": 200,
        "regression_threshold_unacceptable": 0.10,
        "regression_threshold_high": 0.05,
    }
    return SimulaConfig(**{**defaults, **kwargs})


def make_llm() -> MagicMock:
    llm = MagicMock()
    response = MagicMock()
    response.text = "RISK: LOW\nREASONING: No issues expected.\nBENEFIT: Improves capability."
    llm.evaluate = AsyncMock(return_value=response)
    return llm


def make_proposal(
    category: ChangeCategory = ChangeCategory.ADD_EXECUTOR,
    **spec_kwargs,
) -> EvolutionProposal:
    spec = ChangeSpec(**spec_kwargs)
    return EvolutionProposal(
        source="evo",
        category=category,
        description="Test simulation",
        change_spec=spec,
    )


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestAdditiveSimulation:
    @pytest.mark.asyncio
    async def test_valid_executor_low_risk(self):
        sim = ChangeSimulator(config=make_config(), llm=make_llm())
        proposal = make_proposal(
            category=ChangeCategory.ADD_EXECUTOR,
            executor_name="weather_lookup",
            executor_description="Look up weather data",
            executor_action_type="weather",
        )
        result = await sim.simulate(proposal)
        assert result.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_missing_executor_name_moderate_risk(self):
        sim = ChangeSimulator(config=make_config(), llm=make_llm())
        proposal = make_proposal(
            category=ChangeCategory.ADD_EXECUTOR,
            executor_description="No name provided",
        )
        result = await sim.simulate(proposal)
        # Should flag incomplete spec as at least moderate risk
        assert result.risk_level in {RiskLevel.MODERATE, RiskLevel.HIGH}

    @pytest.mark.asyncio
    async def test_valid_input_channel(self):
        sim = ChangeSimulator(config=make_config(), llm=make_llm())
        proposal = make_proposal(
            category=ChangeCategory.ADD_INPUT_CHANNEL,
            channel_name="slack_webhook",
            channel_type="webhook",
            channel_description="Slack incoming webhook",
        )
        result = await sim.simulate(proposal)
        assert result.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_valid_pattern_detector(self):
        sim = ChangeSimulator(config=make_config(), llm=make_llm())
        proposal = make_proposal(
            category=ChangeCategory.ADD_PATTERN_DETECTOR,
            detector_name="correlation_detector",
            detector_description="Detects correlations",
        )
        result = await sim.simulate(proposal)
        assert result.risk_level == RiskLevel.LOW


class TestBudgetSimulation:
    @pytest.mark.asyncio
    async def test_small_budget_change_low_risk(self):
        sim = ChangeSimulator(config=make_config(), llm=make_llm())
        proposal = make_proposal(
            category=ChangeCategory.ADJUST_BUDGET,
            budget_parameter="atune.head.novelty.weight",
            budget_old_value=0.20,
            budget_new_value=0.22,
        )
        result = await sim.simulate(proposal)
        assert result.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_large_budget_change_moderate_risk(self):
        sim = ChangeSimulator(config=make_config(), llm=make_llm())
        proposal = make_proposal(
            category=ChangeCategory.ADJUST_BUDGET,
            budget_parameter="atune.head.novelty.weight",
            budget_old_value=0.20,
            budget_new_value=0.38,
        )
        result = await sim.simulate(proposal)
        assert result.risk_level in {RiskLevel.MODERATE, RiskLevel.HIGH}


class TestGovernanceSimulation:
    @pytest.mark.asyncio
    async def test_governance_uses_llm(self):
        llm = make_llm()
        sim = ChangeSimulator(config=make_config(), llm=llm)
        proposal = make_proposal(
            category=ChangeCategory.MODIFY_CONTRACT,
            contract_changes=["Add timeout field"],
            affected_systems=["axon", "nova"],
        )
        result = await sim.simulate(proposal)
        # Should have called the LLM for risk assessment
        llm.evaluate.assert_called_once()
        assert result.risk_level in {RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH}

    @pytest.mark.asyncio
    async def test_governance_high_risk_response(self):
        llm = make_llm()
        response = MagicMock()
        response.text = "RISK: HIGH\nREASONING: May break existing contracts.\nBENEFIT: Minimal."
        llm.evaluate = AsyncMock(return_value=response)

        sim = ChangeSimulator(config=make_config(), llm=llm)
        proposal = make_proposal(category=ChangeCategory.ADD_SYSTEM_CAPABILITY)
        result = await sim.simulate(proposal)
        assert result.risk_level == RiskLevel.HIGH
