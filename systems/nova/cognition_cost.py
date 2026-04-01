"""
EcodiaOS - Cognition Cost Calculator

Computes the real-time metabolic cost of cognitive operations so Nova can
factor literal energy budgets into expected-free-energy computations.

Every second of deliberation, every LLM call, every database query has a
real financial cost. This module prices cognition internally so the system
can align cognitive investment with decision importance - spending more
thought on high-stakes decisions and being frugal on routine ones.

Cost model:
  - LLM calls:  base_cost_per_token * tokens + model_premium
  - GPU/Vector:  gpu_hourly_rate * duration_hours
  - DB queries:  query_time_ms * cost_per_ms (marginal)
  - I/O:         network_bytes * cost_per_GB

Budget model:
  - Each cycle gets a USD budget based on decision importance
  - Low (routine): $0.10, Medium: $0.50, Critical (DeFi): $5.00
  - Deliberation stops when budget exhausted
  - Inter-budget transfers allow borrowing from future cycles

Performance: all hot-path methods < 0.1ms (no allocations, no locks).
"""

from __future__ import annotations

import enum
import time
from typing import Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel

logger = structlog.get_logger("systems.nova.cognition_cost")


# ─── Cost Rate Configuration ─────────────────────────────────────


class CostRates(EOSBaseModel):
    """
    Unit costs for different cognitive resource types.

    Defaults based on Anthropic Claude 3.5 Sonnet pricing (Feb 2026)
    and typical cloud compute costs. Override via CognitionCostConfig.
    """

    # LLM token costs (USD per token)
    llm_input_per_token: float = 3.00 / 1_000_000    # $3.00 / 1M input
    llm_output_per_token: float = 15.00 / 1_000_000  # $15.00 / 1M output
    # Model premium: fixed per-call overhead (API latency, connection cost)
    llm_call_overhead: float = 0.0001  # $0.0001 per call

    # GPU / vector compute (for embedding, similarity search)
    gpu_hourly_rate: float = 0.50  # $0.50/hr for lightweight GPU

    # Database query (marginal cost per millisecond of query time)
    db_cost_per_ms: float = 0.000001  # $0.001 per 1000ms of query time

    # Network I/O (per GB transferred)
    io_cost_per_gb: float = 0.09  # $0.09/GB (standard cloud egress)


# ─── Operation Types ─────────────────────────────────────────────


class OperationType(enum.StrEnum):
    """Types of cognitive operations with distinct cost profiles."""

    LLM_CALL = "llm_call"
    GPU_COMPUTE = "gpu_compute"
    DB_QUERY = "db_query"
    NETWORK_IO = "network_io"
    # Composite operations (aggregated from sub-operations)
    POLICY_GENERATION = "policy_generation"
    EFE_EVALUATION = "efe_evaluation"
    EQUOR_REVIEW = "equor_review"
    FULL_DELIBERATION = "full_deliberation"


# ─── CognitionCost (returned per operation) ──────────────────────


class CognitionCost(EOSBaseModel):
    """
    The metabolic cost of a single cognitive operation.

    Returned by CognitionCostCalculator.compute_cost() and accumulated
    per-cycle by the CognitionBudget.
    """

    cost_usd: float = 0.0
    cost_tokens: float = 0.0        # Total tokens (input + output) for LLM ops
    cost_cpu_seconds: float = 0.0    # Wall-clock CPU time consumed
    annualized_equivalent: float = 0.0  # Cost if done 1000x/day for a year
    operation_type: str = ""
    breakdown: dict[str, float] = Field(default_factory=dict)

    @staticmethod
    def annualize(cost_usd: float) -> float:
        """What would this cost per year if done 1,000 times per day?"""
        return cost_usd * 1_000 * 365

    def __add__(self, other: CognitionCost) -> CognitionCost:
        """Accumulate costs across operations."""
        merged_breakdown: dict[str, float] = dict(self.breakdown)
        for k, v in other.breakdown.items():
            merged_breakdown[k] = merged_breakdown.get(k, 0.0) + v
        total_usd = self.cost_usd + other.cost_usd
        return CognitionCost(
            cost_usd=total_usd,
            cost_tokens=self.cost_tokens + other.cost_tokens,
            cost_cpu_seconds=self.cost_cpu_seconds + other.cost_cpu_seconds,
            annualized_equivalent=CognitionCost.annualize(total_usd),
            operation_type="composite",
            breakdown=merged_breakdown,
        )


# ─── Decision Importance ─────────────────────────────────────────


class DecisionImportance(enum.StrEnum):
    """
    Importance tier for budget allocation.

    Determined by Atune salience, goal priority, and risk level.
    """

    LOW = "low"            # Routine: observation, idle checks
    MEDIUM = "medium"      # Standard dialogue, goal pursuit
    HIGH = "high"          # High-stakes: external actions, federation
    CRITICAL = "critical"  # Safety-critical: DeFi, irreversible actions


# Default budgets per importance tier (USD)
IMPORTANCE_BUDGETS: dict[DecisionImportance, float] = {
    DecisionImportance.LOW: 0.10,
    DecisionImportance.MEDIUM: 0.50,
    DecisionImportance.HIGH: 2.00,
    DecisionImportance.CRITICAL: 5.00,
}


# ─── Cognition Budget (per-cycle tracking) ───────────────────────


class CognitionBudget(EOSBaseModel):
    """
    Per-cycle cognitive spending budget.

    Tracks allocated vs. spent USD for a single deliberation cycle.
    When spent exceeds allocated, deliberation should stop generating
    additional policies.
    """

    # Budget parameters
    importance: DecisionImportance = DecisionImportance.MEDIUM
    allocated_usd: float = 0.50
    spent_usd: float = 0.0

    # Inter-budget transfer tracking
    borrowed_usd: float = 0.0   # Borrowed from future cycles
    debt_usd: float = 0.0       # Accumulated debt from past borrows

    # Counters
    operations_count: int = 0
    policies_costed: int = 0

    @property
    def remaining_usd(self) -> float:
        """Available budget including any borrowed amount."""
        return max(0.0, self.allocated_usd + self.borrowed_usd - self.spent_usd)

    @property
    def utilisation(self) -> float:
        """Fraction of budget consumed (0-1)."""
        total = self.allocated_usd + self.borrowed_usd
        if total <= 0:
            return 1.0
        return min(1.0, self.spent_usd / total)

    @property
    def is_exhausted(self) -> bool:
        """True when budget is fully spent."""
        return self.remaining_usd <= 0.0

    def charge(self, cost: CognitionCost) -> None:
        """Record a cognitive expenditure against this cycle's budget."""
        self.spent_usd += cost.cost_usd
        self.operations_count += 1

    def borrow(self, amount_usd: float) -> float:
        """
        Borrow from future cycle budgets. Returns actual amount borrowed.
        Capped at 2x the base allocation to prevent runaway debt.
        """
        max_borrow = self.allocated_usd * 2.0 - self.borrowed_usd
        actual = min(amount_usd, max(0.0, max_borrow))
        self.borrowed_usd += actual
        self.debt_usd += actual
        return actual

    def to_log_dict(self) -> dict[str, float]:
        """Compact dict for structured logging."""
        return {
            "allocated_usd": round(self.allocated_usd, 4),
            "spent_usd": round(self.spent_usd, 4),
            "remaining_usd": round(self.remaining_usd, 4),
            "utilisation": round(self.utilisation, 3),
            "borrowed_usd": round(self.borrowed_usd, 4),
            "debt_usd": round(self.debt_usd, 4),
            "operations": self.operations_count,
        }


# ─── CognitionCostCalculator ────────────────────────────────────


class CognitionCostCalculator:
    """
    Computes the real-time metabolic cost of cognitive operations.

    Inputs: operation type, duration, resources used.
    Output: CognitionCost with USD cost, token cost, CPU seconds,
            and annualized projection.

    Stateless and fast - all state lives in CognitionBudget.

    Usage:
        calc = CognitionCostCalculator()
        cost = calc.compute_llm_cost(input_tokens=1200, output_tokens=350)
        budget.charge(cost)
    """

    def __init__(self, rates: CostRates | None = None) -> None:
        self._rates = rates or CostRates()
        self._logger = logger.bind(component="cognition_cost")

        # Accumulate daily cost for reporting (reset by caller)
        self._daily_total_usd: float = 0.0
        self._daily_operations: int = 0
        self._session_start: float = time.monotonic()

        # Inter-cycle debt ledger: carried forward between cycles
        self._outstanding_debt_usd: float = 0.0

    @property
    def rates(self) -> CostRates:
        return self._rates

    @property
    def outstanding_debt_usd(self) -> float:
        return self._outstanding_debt_usd

    # ─── Per-Operation Cost Computation ──────────────────────────

    def compute_llm_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float = 0.0,
    ) -> CognitionCost:
        """
        Compute cost of a single LLM call.

        cost = input_tokens * input_rate + output_tokens * output_rate + overhead
        """
        r = self._rates
        token_cost = (
            input_tokens * r.llm_input_per_token
            + output_tokens * r.llm_output_per_token
        )
        total = token_cost + r.llm_call_overhead
        total_tokens = float(input_tokens + output_tokens)
        cpu_s = duration_ms / 1000.0

        cost = CognitionCost(
            cost_usd=total,
            cost_tokens=total_tokens,
            cost_cpu_seconds=cpu_s,
            annualized_equivalent=CognitionCost.annualize(total),
            operation_type=OperationType.LLM_CALL,
            breakdown={
                "input_token_cost": input_tokens * r.llm_input_per_token,
                "output_token_cost": output_tokens * r.llm_output_per_token,
                "call_overhead": r.llm_call_overhead,
            },
        )
        self._record(cost)
        return cost

    def compute_gpu_cost(self, duration_ms: float) -> CognitionCost:
        """Compute cost of GPU/vector compute (embeddings, similarity)."""
        hours = duration_ms / 3_600_000.0
        total = hours * self._rates.gpu_hourly_rate

        cost = CognitionCost(
            cost_usd=total,
            cost_cpu_seconds=duration_ms / 1000.0,
            annualized_equivalent=CognitionCost.annualize(total),
            operation_type=OperationType.GPU_COMPUTE,
            breakdown={"gpu_compute": total},
        )
        self._record(cost)
        return cost

    def compute_db_cost(self, query_time_ms: float) -> CognitionCost:
        """Compute marginal cost of a database query."""
        total = query_time_ms * self._rates.db_cost_per_ms

        cost = CognitionCost(
            cost_usd=total,
            cost_cpu_seconds=query_time_ms / 1000.0,
            annualized_equivalent=CognitionCost.annualize(total),
            operation_type=OperationType.DB_QUERY,
            breakdown={"db_query": total},
        )
        self._record(cost)
        return cost

    def compute_io_cost(self, bytes_transferred: int) -> CognitionCost:
        """Compute cost of network I/O."""
        gb = bytes_transferred / (1024 ** 3)
        total = gb * self._rates.io_cost_per_gb

        cost = CognitionCost(
            cost_usd=total,
            annualized_equivalent=CognitionCost.annualize(total),
            operation_type=OperationType.NETWORK_IO,
            breakdown={"network_io": total},
        )
        self._record(cost)
        return cost

    def compute_cost(
        self,
        operation_type: str,
        duration_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        bytes_transferred: int = 0,
        query_time_ms: float = 0.0,
    ) -> CognitionCost:
        """
        Generic cost computation dispatcher.

        Convenience method for callers that don't know the specific operation
        type at compile time (e.g., Axon execution logging).
        """
        try:
            op: OperationType | None = OperationType(operation_type)
        except ValueError:
            op = None

        if op == OperationType.LLM_CALL:
            return self.compute_llm_cost(input_tokens, output_tokens, duration_ms)
        elif op == OperationType.GPU_COMPUTE:
            return self.compute_gpu_cost(duration_ms)
        elif op == OperationType.DB_QUERY:
            return self.compute_db_cost(query_time_ms)
        elif op == OperationType.NETWORK_IO:
            return self.compute_io_cost(bytes_transferred)

        # Composite or unknown: estimate from duration as LLM equivalent
        total = duration_ms * self._rates.db_cost_per_ms * 10  # 10x DB rate as proxy
        cost = CognitionCost(
            cost_usd=total,
            cost_cpu_seconds=duration_ms / 1000.0,
            annualized_equivalent=CognitionCost.annualize(total),
            operation_type=operation_type,
            breakdown={"estimated": total},
        )
        self._record(cost)
        return cost

    # ─── Prospective Cost Estimation ─────────────────────────────

    def estimate_deliberation_cost(
        self,
        num_policies: int,
        use_llm_pragmatic: bool = True,
        use_llm_epistemic: bool = True,
    ) -> CognitionCost:
        """
        Estimate the total cost of deliberating on N policies BEFORE doing it.

        Used by Nova to decide whether generating another policy is worth
        the cognitive investment. This is the "cost of deliberating on(policy)"
        term in the EFE adjustment formula.
        """
        r = self._rates

        # Policy generation: ~500 input + ~300 output tokens per policy
        gen_input = 500 * num_policies
        gen_output = 300 * num_policies
        gen_cost = gen_input * r.llm_input_per_token + gen_output * r.llm_output_per_token

        # EFE evaluation per policy
        eval_cost = 0.0
        per_policy_eval = 0.0
        if use_llm_pragmatic:
            per_policy_eval += 200 * r.llm_input_per_token + 100 * r.llm_output_per_token
        if use_llm_epistemic:
            per_policy_eval += 150 * r.llm_input_per_token + 80 * r.llm_output_per_token
        eval_cost = per_policy_eval * num_policies

        # Call overhead
        calls = num_policies  # generation
        if use_llm_pragmatic:
            calls += num_policies
        if use_llm_epistemic:
            calls += num_policies
        overhead = calls * r.llm_call_overhead

        total = gen_cost + eval_cost + overhead

        return CognitionCost(
            cost_usd=total,
            cost_tokens=float(
                gen_input + gen_output
                + (350 * num_policies if use_llm_pragmatic else 0)
                + (230 * num_policies if use_llm_epistemic else 0)
            ),
            annualized_equivalent=CognitionCost.annualize(total),
            operation_type=OperationType.FULL_DELIBERATION,
            breakdown={
                "policy_generation": gen_cost,
                "efe_evaluation": eval_cost,
                "call_overhead": overhead,
            },
        )

    def estimate_policy_marginal_cost(
        self,
        use_llm_pragmatic: bool = True,
        use_llm_epistemic: bool = True,
    ) -> float:
        """
        Estimate the marginal cost (USD) of generating and evaluating
        one additional policy. Used for early-stop decisions.
        """
        return self.estimate_deliberation_cost(
            num_policies=1,
            use_llm_pragmatic=use_llm_pragmatic,
            use_llm_epistemic=use_llm_epistemic,
        ).cost_usd

    # ─── Budget Allocation ───────────────────────────────────────

    def allocate_budget(
        self,
        importance: DecisionImportance,
        custom_budgets: dict[DecisionImportance, float] | None = None,
    ) -> CognitionBudget:
        """
        Allocate a per-cycle cognition budget based on decision importance.

        If there is outstanding debt from previous cycles, reduce the
        allocation proportionally (up to 50% reduction).
        """
        budgets = custom_budgets or IMPORTANCE_BUDGETS
        base = budgets.get(importance, IMPORTANCE_BUDGETS[DecisionImportance.MEDIUM])

        # Deduct debt from previous over-borrowing (max 50% reduction)
        debt_deduction = min(self._outstanding_debt_usd, base * 0.5)
        self._outstanding_debt_usd -= debt_deduction
        effective = base - debt_deduction

        return CognitionBudget(
            importance=importance,
            allocated_usd=effective,
        )

    def settle_cycle_debt(self, budget: CognitionBudget) -> None:
        """
        After a cycle completes, settle any borrowed funds into the debt ledger.
        Called by DeliberationEngine at end of each deliberation.
        """
        if budget.debt_usd > 0:
            self._outstanding_debt_usd += budget.debt_usd
            self._logger.info(
                "cognition_debt_settled",
                cycle_debt=round(budget.debt_usd, 4),
                outstanding=round(self._outstanding_debt_usd, 4),
            )

    # ─── Importance Classification ───────────────────────────────

    @staticmethod
    def classify_importance(
        salience_composite: float,
        goal_priority: float,
        risk_score: float,
        has_external_action: bool = False,
    ) -> DecisionImportance:
        """
        Determine decision importance from Atune salience and goal context.

        External actions (federation, DeFi, API calls) automatically
        elevate to at least HIGH. Safety-critical actions are CRITICAL.
        """
        # Composite importance signal
        importance_signal = (
            0.4 * salience_composite
            + 0.3 * goal_priority
            + 0.3 * risk_score
        )

        if risk_score > 0.8 or (has_external_action and risk_score > 0.5):
            return DecisionImportance.CRITICAL
        elif has_external_action or importance_signal > 0.7:
            return DecisionImportance.HIGH
        elif importance_signal > 0.4:
            return DecisionImportance.MEDIUM
        else:
            return DecisionImportance.LOW

    # ─── Reporting ───────────────────────────────────────────────

    @property
    def daily_stats(self) -> dict[str, Any]:
        """Daily cost aggregate for monitoring."""
        elapsed_s = time.monotonic() - self._session_start
        elapsed_h = max(elapsed_s / 3600.0, 0.001)
        hourly = self._daily_total_usd / elapsed_h
        return {
            "cognition_cost_daily": round(self._daily_total_usd, 4),
            "cognition_operations_daily": self._daily_operations,
            "cognition_cost_hourly": round(hourly, 4),
            "annualized_estimate": round(hourly * 24 * 365, 2),
            "outstanding_debt_usd": round(self._outstanding_debt_usd, 4),
        }

    def reset_daily(self) -> None:
        """Reset daily accumulators (called by Synapse on 24h boundary)."""
        self._daily_total_usd = 0.0
        self._daily_operations = 0
        self._session_start = time.monotonic()

    # ─── Internal ────────────────────────────────────────────────

    def _record(self, cost: CognitionCost) -> None:
        """Accumulate into daily totals."""
        self._daily_total_usd += cost.cost_usd
        self._daily_operations += 1
