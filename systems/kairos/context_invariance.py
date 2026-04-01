"""
EcodiaOS - Kairos: Context Invariance Testing

This stage separates Kairos from ordinary causal discovery.

Most causal discovery asks: "does this rule hold in my data?"
Kairos asks: "does this rule hold regardless of context?"

hold_rate >= 0.95 -> candidate for Tier 2 or Tier 3 promotion
hold_rate <  0.50 -> context-specific, stays at Tier 1
"""

from __future__ import annotations

import math
from typing import Any

import structlog

from systems.kairos.types import (
    CausalRule,
    ContextTestResult,
    InvarianceTestResult,
    InvarianceVerdict,
    KairosConfig,
    ScopeCondition,
)

logger = structlog.get_logger("kairos.context_invariance")


class ContextInvarianceTester:
    """
    Tests whether a causal rule holds across multiple contexts.

    A rule that survives systematic context variation is a candidate
    for promotion up the causal hierarchy. A rule that fails in some
    contexts reveals its scope conditions (still valuable).
    """

    def __init__(self, config: KairosConfig | None = None) -> None:
        self._config = config or KairosConfig()
        self._tests_run: int = 0

    async def test_invariance(
        self,
        rule: CausalRule,
        observations_by_context: dict[str, list[dict[str, Any]]],
    ) -> InvarianceTestResult:
        """
        Test a causal rule across all available contexts.

        Args:
            rule: The confirmed causal rule from Stage 3 (confounder-free).
            observations_by_context: context_id to list of observation dicts.

        Returns:
            InvarianceTestResult with verdict and scope conditions.
        """
        self._tests_run += 1

        # Filter to contexts that contain both variables
        relevant_contexts: dict[str, list[dict[str, Any]]] = {}
        for ctx_id, obs_list in observations_by_context.items():
            paired = [
                obs for obs in obs_list
                if (
                    obs.get(rule.cause_variable) is not None
                    and obs.get(rule.effect_variable) is not None
                )
            ]
            if len(paired) >= 3:
                relevant_contexts[ctx_id] = paired

        if len(relevant_contexts) < self._config.min_contexts_for_invariance:
            return InvarianceTestResult(
                rule=rule,
                verdict=InvarianceVerdict.INSUFFICIENT_CONTEXTS,
                context_count=len(relevant_contexts),
            )

        # Test the rule in each context
        context_results: list[ContextTestResult] = []
        for ctx_id, obs_list in relevant_contexts.items():
            holds, correlation = self._test_rule_in_context(rule, obs_list)
            context_results.append(
                ContextTestResult(
                    context_id=ctx_id,
                    rule_holds=holds,
                    correlation_in_context=correlation,
                    sample_count=len(obs_list),
                )
            )

        # Compute hold rate
        holds_count = sum(1 for r in context_results if r.rule_holds)
        hold_rate = holds_count / len(context_results)

        # Identify failing contexts and scope conditions
        failing_contexts = [r.context_id for r in context_results if not r.rule_holds]
        scope_conditions = self._identify_scope_conditions(
            context_results, observations_by_context
        )

        # Assign verdict
        if hold_rate >= self._config.strong_invariant_threshold:
            verdict = InvarianceVerdict.STRONG_INVARIANT
        elif hold_rate >= self._config.conditional_invariant_threshold:
            verdict = InvarianceVerdict.CONDITIONAL_INVARIANT
        else:
            verdict = InvarianceVerdict.CONTEXT_SPECIFIC

        result = InvarianceTestResult(
            rule=rule,
            verdict=verdict,
            hold_rate=hold_rate,
            context_count=len(relevant_contexts),
            contexts_tested=context_results,
            failing_contexts=failing_contexts,
            scope_conditions=scope_conditions,
        )

        logger.info(
            "context_invariance_test_complete",
            rule_cause=rule.cause_variable,
            rule_effect=rule.effect_variable,
            verdict=verdict.value,
            hold_rate=round(hold_rate, 3),
            contexts_tested=len(context_results),
            failing_count=len(failing_contexts),
        )

        return result

    # --- Internal ---

    def _test_rule_in_context(
        self,
        rule: CausalRule,
        observations: list[dict[str, Any]],
    ) -> tuple[bool, float]:
        """
        Test whether a causal rule holds in a specific context.

        Returns (holds, correlation).
        A rule "holds" if the correlation has the same sign as the original
        and exceeds a minimum threshold.
        """
        cause_vals: list[float] = []
        effect_vals: list[float] = []
        for obs in observations:
            c = obs.get(rule.cause_variable)
            e = obs.get(rule.effect_variable)
            if isinstance(c, (int, float)) and isinstance(e, (int, float)):
                cause_vals.append(float(c))
                effect_vals.append(float(e))

        if len(cause_vals) < 3:
            return False, 0.0

        r = _pearson(cause_vals, effect_vals)
        if r is None:
            return False, 0.0

        # Rule holds if correlation is non-trivial and in the expected direction
        expected_positive = rule.direction_confidence > 0
        holds = r > 0.15 if expected_positive else r < -0.15

        return holds, r

    def _identify_scope_conditions(
        self,
        context_results: list[ContextTestResult],
        all_observations: dict[str, list[dict[str, Any]]],
    ) -> list[ScopeCondition]:
        """
        Attempt to find distinguishing features between contexts where the rule
        holds vs where it fails. This reveals scope conditions.
        """
        holding_contexts = [r.context_id for r in context_results if r.rule_holds]
        failing_contexts = [r.context_id for r in context_results if not r.rule_holds]

        if not failing_contexts or not holding_contexts:
            return []

        holding_means = _context_group_means(holding_contexts, all_observations)
        failing_means = _context_group_means(failing_contexts, all_observations)

        scope_conditions: list[ScopeCondition] = []

        all_vars = set(holding_means.keys()) & set(failing_means.keys())
        for var in sorted(all_vars):
            h_mean = holding_means[var]
            f_mean = failing_means[var]
            diff = abs(h_mean - f_mean)
            pooled = (abs(h_mean) + abs(f_mean)) / 2

            if pooled > 1e-9 and diff / pooled > 0.5:
                scope_conditions.append(
                    ScopeCondition(
                        condition=f"{var} distinguishes holding vs failing contexts",
                        holds_when=True,
                        distinguishing_feature=var,
                        context_ids=failing_contexts,
                    )
                )

        return scope_conditions[:5]

    @property
    def total_tests_run(self) -> int:
        return self._tests_run


# --- Module-level helpers ---


def _pearson(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True)) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    var_y = sum((yi - mean_y) ** 2 for yi in y) / n

    if var_x < 1e-12 or var_y < 1e-12:
        return None

    return cov / math.sqrt(var_x * var_y)


def _context_group_means(
    context_ids: list[str],
    all_observations: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Compute mean of each variable across a group of contexts."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}

    for ctx_id in context_ids:
        for obs in all_observations.get(ctx_id, []):
            for k, v in obs.items():
                if isinstance(v, (int, float)) and not math.isnan(v):
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1

    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}
