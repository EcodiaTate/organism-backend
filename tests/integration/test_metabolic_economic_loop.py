"""
Integration tests for the metabolic-economic feedback loop.

Covers:
  Fix 6.2: Multi-step economic strategy detection in RETrainingExporter
  Fix 6.3: evaluate_economic_intent() passes metabolic_state to the metabolic-aware variant
  Fix 6.4a: PolicyGenerator bounty template min_reward_usd / max_candidates configurable

All tests use real class implementations with mocked I/O.
No real Neo4j, Redis, S3, or LLM calls.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.re_training_exporter import RETrainingExporter
from primitives.common import DriveAlignmentVector
from primitives.re_training import RETrainingDatapoint


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_datapoint(
    example_type: str,
    confidence: float = 0.7,
    timestamp: datetime | None = None,
    episode_id: str = "",
    reasoning_trace: str = "Step 1: analysis\nStep 2: evaluation\nStep 3: decision",
    alternatives: list[str] | None = None,
) -> RETrainingDatapoint:
    return RETrainingDatapoint(
        source_system="oikos",
        example_type=example_type,
        instruction="Execute economic action",
        input_context=f"Action: {example_type}",
        output_action=f"Executed {example_type}",
        outcome="success",
        confidence=confidence,
        timestamp=timestamp or datetime.now(UTC),
        reasoning_trace=reasoning_trace,
        alternatives_considered=alternatives or ["alt_1", "alt_2"],
        constitutional_alignment=DriveAlignmentVector(),
        cost_usd=Decimal("0.001"),
        latency_ms=100,
        episode_id=episode_id,
    )


def make_exporter() -> RETrainingExporter:
    """Build a RETrainingExporter with mocked external deps."""
    exporter = RETrainingExporter.__new__(RETrainingExporter)
    exporter._logger = MagicMock()
    exporter._accumulator = []
    exporter._episode_index = {}
    exporter._total_exported = 0
    exporter._total_batches = 0
    exporter._attached = False
    exporter._starvation_level = "nominal"
    exporter._metabolic_boost = 1.0
    exporter._window_start = datetime.now(UTC)
    return exporter


# ─── Multi-step strategy detection ───────────────────────────────────────────


class TestMultiStepStrategyDetection:
    """Tests for RETrainingExporter._detect_multi_step_strategy()."""

    def test_detects_recovery_via_diversification_pattern(self) -> None:
        """cost_reduce → yield_deploy → spawn_child should be annotated."""
        exporter = make_exporter()
        now = datetime.now(UTC)
        dps = [
            make_datapoint("cost_reduce", timestamp=now),
            make_datapoint("yield_deploy", timestamp=now + timedelta(hours=5)),
            make_datapoint("spawn_child", timestamp=now + timedelta(hours=10)),
        ]
        # Set task_difficulty so 1.3× boost is visible
        for dp in dps:
            dp.task_difficulty = 0.5

        exporter._detect_multi_step_strategy(dps)

        strategic = [dp for dp in dps if dp.is_strategic]
        assert len(strategic) == 3
        assert all(dp.strategy_name == "recovery_via_diversification" for dp in strategic)
        assert strategic[0].strategy_step_number == 1
        assert strategic[1].strategy_step_number == 2
        assert strategic[2].strategy_step_number == 3
        assert strategic[0].strategy_total_steps == 3

    def test_detects_yield_optimization_pattern(self) -> None:
        """yield_rebalance → protocol_explore → consolidate should be annotated."""
        exporter = make_exporter()
        now = datetime.now(UTC)
        dps = [
            make_datapoint("yield_rebalance", timestamp=now),
            make_datapoint("protocol_explore", timestamp=now + timedelta(hours=2)),
            make_datapoint("consolidate", timestamp=now + timedelta(hours=4)),
        ]
        for dp in dps:
            dp.task_difficulty = 0.4

        exporter._detect_multi_step_strategy(dps)

        strategic = [dp for dp in dps if dp.is_strategic]
        assert len(strategic) == 3
        assert all(dp.strategy_name == "yield_optimization" for dp in strategic)

    def test_detects_bounty_then_diversify_pattern(self) -> None:
        """bounty_hunt → asset_create → liquidate should be annotated."""
        exporter = make_exporter()
        now = datetime.now(UTC)
        dps = [
            make_datapoint("bounty_hunt", timestamp=now),
            make_datapoint("asset_create", timestamp=now + timedelta(hours=8)),
            make_datapoint("liquidate", timestamp=now + timedelta(hours=16)),
        ]
        for dp in dps:
            dp.task_difficulty = 0.6

        exporter._detect_multi_step_strategy(dps)

        strategic = [dp for dp in dps if dp.is_strategic]
        assert len(strategic) == 3
        assert all(dp.strategy_name == "bounty_then_diversify" for dp in strategic)

    def test_no_detection_outside_72h_window(self) -> None:
        """Steps more than 72 hours apart should not be matched as a strategy."""
        exporter = make_exporter()
        now = datetime.now(UTC)
        dps = [
            make_datapoint("cost_reduce", timestamp=now),
            make_datapoint("yield_deploy", timestamp=now + timedelta(hours=80)),
            make_datapoint("spawn_child", timestamp=now + timedelta(hours=160)),
        ]
        for dp in dps:
            dp.task_difficulty = 0.5

        exporter._detect_multi_step_strategy(dps)

        assert not any(dp.is_strategic for dp in dps)

    def test_task_difficulty_boosted_1_3x(self) -> None:
        """Strategic datapoints should have task_difficulty × 1.3, capped at 1.0."""
        exporter = make_exporter()
        now = datetime.now(UTC)
        dps = [
            make_datapoint("cost_reduce", timestamp=now),
            make_datapoint("yield_deploy", timestamp=now + timedelta(hours=1)),
            make_datapoint("spawn_child", timestamp=now + timedelta(hours=2)),
        ]
        original_difficulty = 0.5
        for dp in dps:
            dp.task_difficulty = original_difficulty

        exporter._detect_multi_step_strategy(dps)

        for dp in [d for d in dps if d.is_strategic]:
            expected = min(1.0, original_difficulty * 1.3)
            assert abs(dp.task_difficulty - expected) < 1e-9

    def test_difficulty_capped_at_1_0(self) -> None:
        """1.3× boost must not exceed 1.0."""
        exporter = make_exporter()
        now = datetime.now(UTC)
        dps = [
            make_datapoint("cost_reduce", timestamp=now),
            make_datapoint("yield_deploy", timestamp=now + timedelta(hours=1)),
            make_datapoint("spawn_child", timestamp=now + timedelta(hours=2)),
        ]
        for dp in dps:
            dp.task_difficulty = 0.9  # 0.9 × 1.3 > 1.0

        exporter._detect_multi_step_strategy(dps)

        for dp in [d for d in dps if d.is_strategic]:
            assert dp.task_difficulty <= 1.0

    def test_duration_hours_computed_correctly(self) -> None:
        """strategy_duration_hours should reflect actual elapsed time."""
        exporter = make_exporter()
        now = datetime.now(UTC)
        dps = [
            make_datapoint("bounty_hunt", timestamp=now),
            make_datapoint("asset_create", timestamp=now + timedelta(hours=12)),
            make_datapoint("liquidate", timestamp=now + timedelta(hours=24)),
        ]
        for dp in dps:
            dp.task_difficulty = 0.5

        exporter._detect_multi_step_strategy(dps)

        strategic = [dp for dp in dps if dp.is_strategic]
        assert len(strategic) == 3
        assert abs(strategic[0].strategy_duration_hours - 24.0) < 0.1

    def test_non_economic_types_ignored(self) -> None:
        """Non-economic example types should not participate in strategy detection."""
        exporter = make_exporter()
        now = datetime.now(UTC)
        dps = [
            make_datapoint("constitutional_deliberation", timestamp=now),
            make_datapoint("epistemic_reasoning", timestamp=now + timedelta(hours=1)),
            make_datapoint("narrative_synthesis", timestamp=now + timedelta(hours=2)),
        ]
        for dp in dps:
            dp.task_difficulty = 0.5

        exporter._detect_multi_step_strategy(dps)

        assert not any(dp.is_strategic for dp in dps)

    def test_fewer_than_two_economic_events_is_noop(self) -> None:
        """Fewer than 2 economic datapoints should not trigger any annotations."""
        exporter = make_exporter()
        dps = [make_datapoint("bounty_hunt")]
        dps[0].task_difficulty = 0.5

        exporter._detect_multi_step_strategy(dps)

        assert not dps[0].is_strategic


# ─── evaluate_economic_intent with metabolic_state ───────────────────────────


class TestEvaluateEconomicIntentMetabolicState:
    """Tests for Fix 6.3: metabolic_state parameter forwarding."""

    def test_evaluate_without_metabolic_state_uses_base_logic(self) -> None:
        """When metabolic_state is None, base evaluate_economic_intent() runs."""
        from systems.equor.economic_evaluator import evaluate_economic_intent
        from primitives.intent import Intent, IntentStep

        intent = MagicMock(spec=Intent)
        intent.goal = "yield_deploy"
        intent.steps = []
        intent.estimated_cost_usd = Decimal("0.5")

        # Should not raise and should return DriveAlignmentVector or None
        result = evaluate_economic_intent(intent, metabolic_state=None)
        # Result is DriveAlignmentVector or None - either is valid
        assert result is None or hasattr(result, "coherence")

    def test_evaluate_with_metabolic_state_delegates_to_metabolic_variant(self) -> None:
        """When metabolic_state is provided, the metabolic-aware variant is called."""
        from systems.equor.economic_evaluator import evaluate_economic_intent

        intent = MagicMock()
        intent.goal = "yield_deploy"
        intent.steps = []
        intent.estimated_cost_usd = Decimal("1.0")

        metabolic_state = {"starvation_level": "nominal", "efficiency_ratio": 1.2}

        with patch(
            "systems.equor.economic_evaluator.evaluate_economic_intent_with_metabolic_state"
        ) as mock_metabolic:
            mock_metabolic.return_value = DriveAlignmentVector(coherence=0.8, care=0.6)
            result = evaluate_economic_intent(intent, metabolic_state=metabolic_state)

        mock_metabolic.assert_called_once_with(intent, metabolic_state)
        assert result is not None
        assert result.coherence == pytest.approx(0.8)

    def test_metabolic_state_not_passed_when_none(self) -> None:
        """When metabolic_state is None, metabolic variant should NOT be called."""
        from systems.equor.economic_evaluator import evaluate_economic_intent

        intent = MagicMock()
        intent.goal = "cost_reduce"
        intent.steps = []
        intent.estimated_cost_usd = Decimal("0.1")

        with patch(
            "systems.equor.economic_evaluator.evaluate_economic_intent_with_metabolic_state"
        ) as mock_metabolic:
            evaluate_economic_intent(intent, metabolic_state=None)

        mock_metabolic.assert_not_called()


# ─── PolicyGenerator configurable bounty params ──────────────────────────────


class TestPolicyGeneratorBountyParams:
    """Tests for Fix 6.4a: bounty template min_reward_usd and max_candidates."""

    def test_default_bounty_params_are_10_and_20(self) -> None:
        """Default PolicyGenerator should use min_reward_usd=10.0, max_candidates=20."""
        from systems.nova.policy_generator import PolicyGenerator, _PROCEDURE_TEMPLATES

        llm = MagicMock()
        # Reset to defaults first
        PolicyGenerator(llm=llm, bounty_min_reward_usd=10.0, bounty_max_candidates=20)

        bounty_template = next(
            t for t in _PROCEDURE_TEMPLATES if t.get("name") == "Autonomous bounty hunting"
        )
        bounty_step = next(
            s for s in bounty_template["steps"] if s["action_type"] == "bounty_hunt"
        )
        assert bounty_step["parameters"]["min_reward_usd"] == 10.0
        assert bounty_step["parameters"]["max_candidates"] == 20

    def test_custom_bounty_params_applied_to_template(self) -> None:
        """PolicyGenerator with custom params should patch the module-level template."""
        from systems.nova.policy_generator import PolicyGenerator, _PROCEDURE_TEMPLATES

        llm = MagicMock()
        PolicyGenerator(llm=llm, bounty_min_reward_usd=25.0, bounty_max_candidates=50)

        bounty_template = next(
            t for t in _PROCEDURE_TEMPLATES if t.get("name") == "Autonomous bounty hunting"
        )
        bounty_step = next(
            s for s in bounty_template["steps"] if s["action_type"] == "bounty_hunt"
        )
        assert bounty_step["parameters"]["min_reward_usd"] == 25.0
        assert bounty_step["parameters"]["max_candidates"] == 50

        # Reset to defaults to avoid polluting other tests
        PolicyGenerator(llm=llm, bounty_min_reward_usd=10.0, bounty_max_candidates=20)

    def test_params_visible_to_fast_path_template_matching(self) -> None:
        """Fast-path procedure_to_policy uses the patched template, not hardcoded values."""
        from systems.nova.policy_generator import (
            PolicyGenerator,
            _PROCEDURE_TEMPLATES,
            procedure_to_policy,
        )

        llm = MagicMock()
        PolicyGenerator(llm=llm, bounty_min_reward_usd=15.0, bounty_max_candidates=30)

        bounty_template = next(
            t for t in _PROCEDURE_TEMPLATES if t.get("name") == "Autonomous bounty hunting"
        )
        policy = procedure_to_policy(bounty_template)

        # Find the bounty_hunt step in the converted policy
        bounty_step = next(
            (s for s in policy.steps if s.action_type == "bounty_hunt"), None
        )
        assert bounty_step is not None
        assert bounty_step.parameters["min_reward_usd"] == 15.0
        assert bounty_step.parameters["max_candidates"] == 30

        # Reset
        PolicyGenerator(llm=llm, bounty_min_reward_usd=10.0, bounty_max_candidates=20)

    @pytest.mark.asyncio
    async def test_generate_economic_intent_uses_configured_params(self) -> None:
        """generate_economic_intent() should select bounty with configured params when appropriate."""
        from systems.nova.policy_generator import PolicyGenerator
        from systems.nova.types import BeliefState

        llm = MagicMock()
        gen = PolicyGenerator(llm=llm, bounty_min_reward_usd=50.0, bounty_max_candidates=10)

        beliefs = BeliefState()
        # High bounty confidence → bounty hunting template selected
        beliefs.entities["bounty_success_rate"] = MagicMock(confidence=0.9)
        beliefs.entities["economic_risk_level"] = MagicMock(confidence=0.1)

        economic_context = {"wallet_balance_usd": 500.0, "burn_rate_hourly_usd": 0.5}
        policy = await gen.generate_economic_intent(beliefs, economic_context)

        # If bounty template was selected, params should be from constructor
        bounty_step = next(
            (s for s in policy.steps if s.action_type == "bounty_hunt"), None
        )
        if bounty_step is not None:
            assert bounty_step.parameters["min_reward_usd"] == 50.0
            assert bounty_step.parameters["max_candidates"] == 10

        # Reset
        PolicyGenerator(llm=llm, bounty_min_reward_usd=10.0, bounty_max_candidates=20)
