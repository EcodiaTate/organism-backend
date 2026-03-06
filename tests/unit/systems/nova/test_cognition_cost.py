"""
Unit tests for Nova CognitionCostCalculator.

Tests cost computation, budget allocation, importance classification,
inter-budget transfers (debt), and EFE integration with the cognition
cost term.
"""

from __future__ import annotations

import pytest

from systems.nova.cognition_cost import (
    IMPORTANCE_BUDGETS,
    CognitionBudget,
    CognitionCost,
    CognitionCostCalculator,
    CostRates,
    DecisionImportance,
    OperationType,
)

# ─── CognitionCost Type ─────────────────────────────────────────


class TestCognitionCost:
    def test_defaults_are_zero(self) -> None:
        cost = CognitionCost()
        assert cost.cost_usd == 0.0
        assert cost.cost_tokens == 0.0
        assert cost.cost_cpu_seconds == 0.0
        assert cost.annualized_equivalent == 0.0

    def test_annualize_formula(self) -> None:
        # 1000 times/day * 365 days = 365,000x multiplier
        assert CognitionCost.annualize(0.01) == pytest.approx(3650.0)

    def test_addition_sums_fields(self) -> None:
        a = CognitionCost(
            cost_usd=0.10,
            cost_tokens=100.0,
            cost_cpu_seconds=0.5,
            operation_type="llm_call",
            breakdown={"input": 0.06, "output": 0.04},
        )
        b = CognitionCost(
            cost_usd=0.05,
            cost_tokens=50.0,
            cost_cpu_seconds=0.2,
            operation_type="db_query",
            breakdown={"input": 0.01, "db": 0.04},
        )
        combined = a + b
        assert combined.cost_usd == pytest.approx(0.15)
        assert combined.cost_tokens == pytest.approx(150.0)
        assert combined.cost_cpu_seconds == pytest.approx(0.7)
        assert combined.operation_type == "composite"
        # Merged breakdown sums overlapping keys
        assert combined.breakdown["input"] == pytest.approx(0.07)
        assert combined.breakdown["output"] == pytest.approx(0.04)
        assert combined.breakdown["db"] == pytest.approx(0.04)

    def test_addition_updates_annualized(self) -> None:
        a = CognitionCost(cost_usd=0.01)
        b = CognitionCost(cost_usd=0.02)
        combined = a + b
        assert combined.annualized_equivalent == pytest.approx(
            CognitionCost.annualize(0.03)
        )


# ─── CognitionBudget Type ───────────────────────────────────────


class TestCognitionBudget:
    def test_default_budget_is_medium(self) -> None:
        budget = CognitionBudget()
        assert budget.importance == DecisionImportance.MEDIUM
        assert budget.allocated_usd == 0.50

    def test_remaining_usd(self) -> None:
        budget = CognitionBudget(allocated_usd=1.00, spent_usd=0.30)
        assert budget.remaining_usd == pytest.approx(0.70)

    def test_remaining_never_negative(self) -> None:
        budget = CognitionBudget(allocated_usd=0.10, spent_usd=0.50)
        assert budget.remaining_usd == 0.0

    def test_remaining_includes_borrowed(self) -> None:
        budget = CognitionBudget(
            allocated_usd=0.50, spent_usd=0.40, borrowed_usd=0.20
        )
        assert budget.remaining_usd == pytest.approx(0.30)

    def test_utilisation_fraction(self) -> None:
        budget = CognitionBudget(allocated_usd=1.00, spent_usd=0.25)
        assert budget.utilisation == pytest.approx(0.25)

    def test_utilisation_capped_at_one(self) -> None:
        budget = CognitionBudget(allocated_usd=0.10, spent_usd=1.00)
        assert budget.utilisation == 1.0

    def test_utilisation_zero_allocation_returns_one(self) -> None:
        budget = CognitionBudget(allocated_usd=0.0)
        assert budget.utilisation == 1.0

    def test_is_exhausted_when_fully_spent(self) -> None:
        budget = CognitionBudget(allocated_usd=0.10, spent_usd=0.10)
        assert budget.is_exhausted is True

    def test_not_exhausted_when_remaining(self) -> None:
        budget = CognitionBudget(allocated_usd=0.50, spent_usd=0.20)
        assert budget.is_exhausted is False

    def test_charge_increments_spent(self) -> None:
        budget = CognitionBudget(allocated_usd=1.00)
        cost = CognitionCost(cost_usd=0.15)
        budget.charge(cost)
        assert budget.spent_usd == pytest.approx(0.15)
        assert budget.operations_count == 1
        budget.charge(CognitionCost(cost_usd=0.05))
        assert budget.spent_usd == pytest.approx(0.20)
        assert budget.operations_count == 2

    def test_borrow_within_limit(self) -> None:
        budget = CognitionBudget(allocated_usd=0.50)
        actual = budget.borrow(0.30)
        assert actual == pytest.approx(0.30)
        assert budget.borrowed_usd == pytest.approx(0.30)
        assert budget.debt_usd == pytest.approx(0.30)

    def test_borrow_capped_at_2x_allocation(self) -> None:
        budget = CognitionBudget(allocated_usd=0.50)
        # Max borrow = 2x allocation = 1.00
        actual = budget.borrow(2.00)
        assert actual == pytest.approx(1.00)
        assert budget.borrowed_usd == pytest.approx(1.00)

    def test_borrow_successive_respects_cap(self) -> None:
        budget = CognitionBudget(allocated_usd=0.50)
        first = budget.borrow(0.80)
        assert first == pytest.approx(0.80)
        # Only 0.20 remaining of 1.00 cap
        second = budget.borrow(0.50)
        assert second == pytest.approx(0.20)
        assert budget.borrowed_usd == pytest.approx(1.00)

    def test_to_log_dict_has_expected_keys(self) -> None:
        budget = CognitionBudget(allocated_usd=0.50, spent_usd=0.10)
        d = budget.to_log_dict()
        expected_keys = {
            "allocated_usd",
            "spent_usd",
            "remaining_usd",
            "utilisation",
            "borrowed_usd",
            "debt_usd",
            "operations",
        }
        assert set(d.keys()) == expected_keys


# ─── CognitionCostCalculator: LLM Costs ─────────────────────────


class TestLLMCostComputation:
    def setup_method(self) -> None:
        self.rates = CostRates(
            llm_input_per_token=3.00 / 1_000_000,
            llm_output_per_token=15.00 / 1_000_000,
            llm_call_overhead=0.0001,
        )
        self.calc = CognitionCostCalculator(rates=self.rates)

    def test_basic_llm_cost(self) -> None:
        cost = self.calc.compute_llm_cost(
            input_tokens=1000, output_tokens=500
        )
        expected = (
            1000 * self.rates.llm_input_per_token
            + 500 * self.rates.llm_output_per_token
            + self.rates.llm_call_overhead
        )
        assert cost.cost_usd == pytest.approx(expected)
        assert cost.cost_tokens == pytest.approx(1500.0)
        assert cost.operation_type == OperationType.LLM_CALL

    def test_llm_cost_zero_tokens(self) -> None:
        cost = self.calc.compute_llm_cost(input_tokens=0, output_tokens=0)
        # Should still have call overhead
        assert cost.cost_usd == pytest.approx(self.rates.llm_call_overhead)
        assert cost.cost_tokens == 0.0

    def test_llm_cost_records_duration(self) -> None:
        cost = self.calc.compute_llm_cost(
            input_tokens=100, output_tokens=50, duration_ms=500.0
        )
        assert cost.cost_cpu_seconds == pytest.approx(0.5)

    def test_llm_cost_breakdown_keys(self) -> None:
        cost = self.calc.compute_llm_cost(
            input_tokens=1000, output_tokens=200
        )
        assert "input_token_cost" in cost.breakdown
        assert "output_token_cost" in cost.breakdown
        assert "call_overhead" in cost.breakdown

    def test_llm_cost_annualized(self) -> None:
        cost = self.calc.compute_llm_cost(
            input_tokens=1000, output_tokens=500
        )
        assert cost.annualized_equivalent == pytest.approx(
            CognitionCost.annualize(cost.cost_usd)
        )


# ─── CognitionCostCalculator: GPU Costs ─────────────────────────


class TestGPUCostComputation:
    def setup_method(self) -> None:
        self.rates = CostRates(gpu_hourly_rate=1.00)
        self.calc = CognitionCostCalculator(rates=self.rates)

    def test_gpu_cost_one_hour(self) -> None:
        cost = self.calc.compute_gpu_cost(duration_ms=3_600_000.0)
        assert cost.cost_usd == pytest.approx(1.00)

    def test_gpu_cost_one_second(self) -> None:
        cost = self.calc.compute_gpu_cost(duration_ms=1000.0)
        expected = 1.00 / 3600.0
        assert cost.cost_usd == pytest.approx(expected)

    def test_gpu_cost_zero_duration(self) -> None:
        cost = self.calc.compute_gpu_cost(duration_ms=0.0)
        assert cost.cost_usd == 0.0

    def test_gpu_cost_operation_type(self) -> None:
        cost = self.calc.compute_gpu_cost(duration_ms=100.0)
        assert cost.operation_type == OperationType.GPU_COMPUTE


# ─── CognitionCostCalculator: DB Costs ──────────────────────────


class TestDBCostComputation:
    def setup_method(self) -> None:
        self.rates = CostRates(db_cost_per_ms=0.000001)
        self.calc = CognitionCostCalculator(rates=self.rates)

    def test_db_cost_1000ms(self) -> None:
        cost = self.calc.compute_db_cost(query_time_ms=1000.0)
        assert cost.cost_usd == pytest.approx(0.001)

    def test_db_cost_zero(self) -> None:
        cost = self.calc.compute_db_cost(query_time_ms=0.0)
        assert cost.cost_usd == 0.0

    def test_db_cost_operation_type(self) -> None:
        cost = self.calc.compute_db_cost(query_time_ms=50.0)
        assert cost.operation_type == OperationType.DB_QUERY


# ─── CognitionCostCalculator: I/O Costs ─────────────────────────


class TestIOCostComputation:
    def setup_method(self) -> None:
        self.rates = CostRates(io_cost_per_gb=0.09)
        self.calc = CognitionCostCalculator(rates=self.rates)

    def test_io_cost_one_gb(self) -> None:
        one_gb = 1024 ** 3
        cost = self.calc.compute_io_cost(bytes_transferred=one_gb)
        assert cost.cost_usd == pytest.approx(0.09)

    def test_io_cost_one_mb(self) -> None:
        one_mb = 1024 ** 2
        cost = self.calc.compute_io_cost(bytes_transferred=one_mb)
        expected = 0.09 / 1024.0  # 1MB = 1/1024 GB
        assert cost.cost_usd == pytest.approx(expected)

    def test_io_cost_zero_bytes(self) -> None:
        cost = self.calc.compute_io_cost(bytes_transferred=0)
        assert cost.cost_usd == 0.0

    def test_io_cost_operation_type(self) -> None:
        cost = self.calc.compute_io_cost(bytes_transferred=1000)
        assert cost.operation_type == OperationType.NETWORK_IO


# ─── Generic Cost Dispatcher ─────────────────────────────────────


class TestComputeCostDispatcher:
    def setup_method(self) -> None:
        self.calc = CognitionCostCalculator()

    def test_dispatches_llm_call(self) -> None:
        cost = self.calc.compute_cost(
            operation_type=OperationType.LLM_CALL,
            input_tokens=100,
            output_tokens=50,
        )
        assert cost.operation_type == OperationType.LLM_CALL
        assert cost.cost_usd > 0

    def test_dispatches_gpu_compute(self) -> None:
        cost = self.calc.compute_cost(
            operation_type=OperationType.GPU_COMPUTE,
            duration_ms=1000.0,
        )
        assert cost.operation_type == OperationType.GPU_COMPUTE

    def test_dispatches_db_query(self) -> None:
        cost = self.calc.compute_cost(
            operation_type=OperationType.DB_QUERY,
            query_time_ms=50.0,
        )
        assert cost.operation_type == OperationType.DB_QUERY

    def test_dispatches_network_io(self) -> None:
        cost = self.calc.compute_cost(
            operation_type=OperationType.NETWORK_IO,
            bytes_transferred=1024,
        )
        assert cost.operation_type == OperationType.NETWORK_IO

    def test_unknown_type_uses_duration_proxy(self) -> None:
        cost = self.calc.compute_cost(
            operation_type="custom_op",
            duration_ms=100.0,
        )
        assert cost.operation_type == "custom_op"
        assert cost.cost_usd > 0


# ─── Deliberation Cost Estimation ────────────────────────────────


class TestDeliberationCostEstimation:
    def setup_method(self) -> None:
        self.calc = CognitionCostCalculator()

    def test_estimate_scales_with_policies(self) -> None:
        cost_1 = self.calc.estimate_deliberation_cost(num_policies=1)
        cost_5 = self.calc.estimate_deliberation_cost(num_policies=5)
        assert cost_5.cost_usd > cost_1.cost_usd
        # Should be roughly 5x (linear in policies)
        assert cost_5.cost_usd == pytest.approx(cost_1.cost_usd * 5, rel=0.01)

    def test_estimate_cheaper_without_llm(self) -> None:
        full = self.calc.estimate_deliberation_cost(
            num_policies=3, use_llm_pragmatic=True, use_llm_epistemic=True
        )
        no_llm = self.calc.estimate_deliberation_cost(
            num_policies=3, use_llm_pragmatic=False, use_llm_epistemic=False
        )
        assert no_llm.cost_usd < full.cost_usd

    def test_estimate_tokens_populated(self) -> None:
        cost = self.calc.estimate_deliberation_cost(num_policies=2)
        assert cost.cost_tokens > 0

    def test_estimate_breakdown_has_components(self) -> None:
        cost = self.calc.estimate_deliberation_cost(num_policies=1)
        assert "policy_generation" in cost.breakdown
        assert "efe_evaluation" in cost.breakdown
        assert "call_overhead" in cost.breakdown

    def test_marginal_cost_equals_single_policy(self) -> None:
        marginal = self.calc.estimate_policy_marginal_cost()
        single = self.calc.estimate_deliberation_cost(num_policies=1)
        assert marginal == pytest.approx(single.cost_usd)


# ─── Budget Allocation ───────────────────────────────────────────


class TestBudgetAllocation:
    def setup_method(self) -> None:
        self.calc = CognitionCostCalculator()

    def test_low_importance_budget(self) -> None:
        budget = self.calc.allocate_budget(DecisionImportance.LOW)
        assert budget.importance == DecisionImportance.LOW
        assert budget.allocated_usd == pytest.approx(
            IMPORTANCE_BUDGETS[DecisionImportance.LOW]
        )

    def test_medium_importance_budget(self) -> None:
        budget = self.calc.allocate_budget(DecisionImportance.MEDIUM)
        assert budget.allocated_usd == pytest.approx(0.50)

    def test_high_importance_budget(self) -> None:
        budget = self.calc.allocate_budget(DecisionImportance.HIGH)
        assert budget.allocated_usd == pytest.approx(2.00)

    def test_critical_importance_budget(self) -> None:
        budget = self.calc.allocate_budget(DecisionImportance.CRITICAL)
        assert budget.allocated_usd == pytest.approx(5.00)

    def test_custom_budgets_override(self) -> None:
        custom = {
            DecisionImportance.LOW: 0.05,
            DecisionImportance.MEDIUM: 0.25,
            DecisionImportance.HIGH: 1.00,
            DecisionImportance.CRITICAL: 3.00,
        }
        budget = self.calc.allocate_budget(
            DecisionImportance.MEDIUM, custom_budgets=custom
        )
        assert budget.allocated_usd == pytest.approx(0.25)

    def test_debt_reduces_allocation(self) -> None:
        calc = CognitionCostCalculator()
        # Manually set outstanding debt
        calc._outstanding_debt_usd = 0.20

        budget = calc.allocate_budget(DecisionImportance.MEDIUM)
        # Max deduction = 50% of base ($0.50 * 0.5 = $0.25)
        # Actual deduction = min($0.20, $0.25) = $0.20
        assert budget.allocated_usd == pytest.approx(0.30)

    def test_debt_deduction_capped_at_50_percent(self) -> None:
        calc = CognitionCostCalculator()
        # Large debt should only take 50% of allocation
        calc._outstanding_debt_usd = 10.00

        budget = calc.allocate_budget(DecisionImportance.LOW)
        base = IMPORTANCE_BUDGETS[DecisionImportance.LOW]
        assert budget.allocated_usd == pytest.approx(base * 0.50)

    def test_budget_starts_fresh(self) -> None:
        budget = self.calc.allocate_budget(DecisionImportance.MEDIUM)
        assert budget.spent_usd == 0.0
        assert budget.borrowed_usd == 0.0
        assert budget.debt_usd == 0.0
        assert budget.operations_count == 0


# ─── Inter-Budget Debt Settlement ────────────────────────────────


class TestDebtSettlement:
    def test_settle_adds_to_outstanding(self) -> None:
        calc = CognitionCostCalculator()
        budget = CognitionBudget(
            allocated_usd=0.50, borrowed_usd=0.30, debt_usd=0.30
        )
        calc.settle_cycle_debt(budget)
        assert calc.outstanding_debt_usd == pytest.approx(0.30)

    def test_settle_accumulates_across_cycles(self) -> None:
        calc = CognitionCostCalculator()
        budget1 = CognitionBudget(
            allocated_usd=0.50, debt_usd=0.10
        )
        budget2 = CognitionBudget(
            allocated_usd=0.50, debt_usd=0.20
        )
        calc.settle_cycle_debt(budget1)
        calc.settle_cycle_debt(budget2)
        assert calc.outstanding_debt_usd == pytest.approx(0.30)

    def test_settle_no_debt_is_noop(self) -> None:
        calc = CognitionCostCalculator()
        budget = CognitionBudget(allocated_usd=0.50, debt_usd=0.0)
        calc.settle_cycle_debt(budget)
        assert calc.outstanding_debt_usd == 0.0

    def test_debt_flows_through_allocation_cycle(self) -> None:
        """Full cycle: borrow → settle → reduced next allocation."""
        calc = CognitionCostCalculator()

        # Cycle 1: borrow $0.40
        budget1 = calc.allocate_budget(DecisionImportance.MEDIUM)
        budget1.borrow(0.40)
        calc.settle_cycle_debt(budget1)
        assert calc.outstanding_debt_usd == pytest.approx(0.40)

        # Cycle 2: allocation reduced by debt (capped at 50%)
        budget2 = calc.allocate_budget(DecisionImportance.MEDIUM)
        # Debt $0.40 > cap ($0.50*0.5=$0.25), so deduction = $0.25
        assert budget2.allocated_usd == pytest.approx(0.25)
        # Outstanding debt reduced by deduction
        assert calc.outstanding_debt_usd == pytest.approx(0.15)


# ─── Importance Classification ───────────────────────────────────


class TestImportanceClassification:
    def test_low_signals_classify_low(self) -> None:
        imp = CognitionCostCalculator.classify_importance(
            salience_composite=0.1,
            goal_priority=0.1,
            risk_score=0.1,
        )
        assert imp == DecisionImportance.LOW

    def test_medium_signals_classify_medium(self) -> None:
        imp = CognitionCostCalculator.classify_importance(
            salience_composite=0.5,
            goal_priority=0.5,
            risk_score=0.3,
        )
        assert imp == DecisionImportance.MEDIUM

    def test_high_signal_classify_high(self) -> None:
        # importance_signal = 0.4*0.9 + 0.3*0.9 + 0.3*0.4 = 0.36+0.27+0.12 = 0.75 > 0.7
        imp = CognitionCostCalculator.classify_importance(
            salience_composite=0.9,
            goal_priority=0.9,
            risk_score=0.4,
        )
        assert imp == DecisionImportance.HIGH

    def test_external_action_elevates_to_high(self) -> None:
        imp = CognitionCostCalculator.classify_importance(
            salience_composite=0.2,
            goal_priority=0.2,
            risk_score=0.1,
            has_external_action=True,
        )
        assert imp == DecisionImportance.HIGH

    def test_high_risk_elevates_to_critical(self) -> None:
        imp = CognitionCostCalculator.classify_importance(
            salience_composite=0.3,
            goal_priority=0.3,
            risk_score=0.85,
        )
        assert imp == DecisionImportance.CRITICAL

    def test_external_action_with_moderate_risk_is_critical(self) -> None:
        imp = CognitionCostCalculator.classify_importance(
            salience_composite=0.3,
            goal_priority=0.3,
            risk_score=0.6,
            has_external_action=True,
        )
        assert imp == DecisionImportance.CRITICAL

    def test_boundary_medium_to_high(self) -> None:
        """importance_signal > 0.7 → HIGH."""
        # 0.4*0.8 + 0.3*0.8 + 0.3*0.5 = 0.32 + 0.24 + 0.15 = 0.71
        imp = CognitionCostCalculator.classify_importance(
            salience_composite=0.8,
            goal_priority=0.8,
            risk_score=0.5,
        )
        assert imp == DecisionImportance.HIGH

    def test_boundary_low_to_medium(self) -> None:
        """importance_signal > 0.4 → MEDIUM."""
        # 0.4*0.5 + 0.3*0.5 + 0.3*0.3 = 0.20 + 0.15 + 0.09 = 0.44
        imp = CognitionCostCalculator.classify_importance(
            salience_composite=0.5,
            goal_priority=0.5,
            risk_score=0.3,
        )
        assert imp == DecisionImportance.MEDIUM


# ─── Daily Stats & Reset ─────────────────────────────────────────


class TestDailyStats:
    def test_daily_stats_accumulate(self) -> None:
        calc = CognitionCostCalculator()
        calc.compute_llm_cost(input_tokens=1000, output_tokens=500)
        calc.compute_db_cost(query_time_ms=10.0)

        stats = calc.daily_stats
        assert stats["cognition_operations_daily"] == 2
        assert stats["cognition_cost_daily"] > 0

    def test_daily_stats_has_expected_keys(self) -> None:
        calc = CognitionCostCalculator()
        stats = calc.daily_stats
        expected = {
            "cognition_cost_daily",
            "cognition_operations_daily",
            "cognition_cost_hourly",
            "annualized_estimate",
            "outstanding_debt_usd",
        }
        assert set(stats.keys()) == expected

    def test_reset_daily_clears_accumulators(self) -> None:
        calc = CognitionCostCalculator()
        calc.compute_llm_cost(input_tokens=1000, output_tokens=500)
        calc.reset_daily()

        stats = calc.daily_stats
        assert stats["cognition_cost_daily"] == 0.0
        assert stats["cognition_operations_daily"] == 0

    def test_reset_daily_does_not_clear_debt(self) -> None:
        calc = CognitionCostCalculator()
        calc._outstanding_debt_usd = 0.50
        calc.reset_daily()
        assert calc.outstanding_debt_usd == pytest.approx(0.50)


# ─── CostRates ───────────────────────────────────────────────────


class TestCostRates:
    def test_default_rates_are_positive(self) -> None:
        rates = CostRates()
        assert rates.llm_input_per_token > 0
        assert rates.llm_output_per_token > 0
        assert rates.llm_call_overhead > 0
        assert rates.gpu_hourly_rate > 0
        assert rates.db_cost_per_ms > 0
        assert rates.io_cost_per_gb > 0

    def test_output_more_expensive_than_input(self) -> None:
        rates = CostRates()
        assert rates.llm_output_per_token > rates.llm_input_per_token

    def test_custom_rates_override(self) -> None:
        rates = CostRates(llm_input_per_token=0.01, gpu_hourly_rate=5.0)
        assert rates.llm_input_per_token == 0.01
        assert rates.gpu_hourly_rate == 5.0


# ─── EFE Integration (cognition_cost_term in EFEScore) ───────────


class TestEFECognitionCostIntegration:
    """Tests verifying cognition cost integrates correctly with EFE scoring."""

    def setup_method(self) -> None:
        self.calc = CognitionCostCalculator()

    def test_marginal_cost_increases_efe(self) -> None:
        """
        Higher marginal cost → higher cognition_cost_term → worse EFE.

        The term is normalised as cost/budget so a $0.05 cost against a
        $0.10 budget gives term=0.5, while against a $5.00 budget gives
        term=0.01.
        """
        marginal = self.calc.estimate_policy_marginal_cost()

        # Low budget → high normalised term
        low_budget = CognitionBudget(
            importance=DecisionImportance.LOW, allocated_usd=0.10
        )
        term_low = min(1.0, marginal / (low_budget.allocated_usd + low_budget.borrowed_usd))

        # High budget → low normalised term
        high_budget = CognitionBudget(
            importance=DecisionImportance.CRITICAL, allocated_usd=5.00
        )
        term_high = min(1.0, marginal / (high_budget.allocated_usd + high_budget.borrowed_usd))

        assert term_low > term_high

    def test_exhausted_budget_has_max_term(self) -> None:
        """When budget is exhausted, cognition_cost_term should be at max."""
        budget = CognitionBudget(allocated_usd=0.10, spent_usd=0.10)
        assert budget.is_exhausted
        marginal = self.calc.estimate_policy_marginal_cost()
        # With remaining=0, any marginal cost pushes term to 1.0
        term = min(1.0, marginal / max(0.001, budget.remaining_usd))
        assert term == 1.0

    def test_fresh_budget_has_low_term(self) -> None:
        """Large fresh budget should yield a small cognition cost penalty."""
        budget = CognitionBudget(
            importance=DecisionImportance.CRITICAL, allocated_usd=5.00
        )
        marginal = self.calc.estimate_policy_marginal_cost()
        term = min(1.0, marginal / budget.allocated_usd)
        assert term < 0.01  # < 1% of a $5.00 budget

    def test_early_stop_when_budget_tight(self) -> None:
        """Simulate early stop: remaining < marginal cost → exhaustion."""
        budget = self.calc.allocate_budget(DecisionImportance.LOW)
        marginal = self.calc.estimate_policy_marginal_cost()

        # Spend almost everything
        large_cost = CognitionCost(cost_usd=budget.allocated_usd - 0.001)
        budget.charge(large_cost)

        # If remaining < marginal, further deliberation should stop
        should_stop = budget.remaining_usd < marginal
        assert should_stop is True


# ─── OperationType StrEnum ───────────────────────────────────────


class TestOperationType:
    def test_values_are_strings(self) -> None:
        assert OperationType.LLM_CALL == "llm_call"
        assert OperationType.GPU_COMPUTE == "gpu_compute"
        assert OperationType.DB_QUERY == "db_query"
        assert OperationType.NETWORK_IO == "network_io"

    def test_all_types_present(self) -> None:
        expected = {
            "llm_call",
            "gpu_compute",
            "db_query",
            "network_io",
            "policy_generation",
            "efe_evaluation",
            "equor_review",
            "full_deliberation",
        }
        assert {e.value for e in OperationType} == expected
