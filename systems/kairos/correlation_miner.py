"""
EcodiaOS — Kairos Stage 1: Correlation Mining

Mines for correlations across CONTEXTS, not just within them.
A correlation that only holds within one context is probably a local artifact.
A correlation that holds across many contexts is a signal worth investigating causally.

Key filter: mean |r| > 0.3 AND cross-context variance < 0.1.
Only consistent cross-context correlations survive to Stage 2.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import structlog

from systems.kairos.types import (
    ContextCorrelation,
    CorrelationCandidate,
    KairosConfig,
)

logger = structlog.get_logger("kairos.correlation_miner")


class CorrelationMiner:
    """
    Stage 1 of the Kairos pipeline.

    Mines for variable pairs whose correlation is consistent across multiple
    contexts. High cross-context variance = confounded or context-specific.
    Low variance + non-trivial magnitude = worth investigating causally.

    Pre-seeding: external systems (Fovea, Oneiros) can inject candidates via
    add_preseed(). These are emitted by drain_preseeds() and merged into the
    candidate list before Stage 2, bypassing the correlation filter (the signal
    source is trusted to have already done that filtering).
    """

    def __init__(self, config: KairosConfig | None = None) -> None:
        self._config = config or KairosConfig()
        self._candidates_found: int = 0
        self._pairs_evaluated: int = 0
        self._preseeds: list[CorrelationCandidate] = []

    def add_preseed(self, candidate: CorrelationCandidate) -> None:
        """
        Inject a pre-seeded correlation candidate from an external signal source
        (e.g. Fovea causal prediction errors, Oneiros cross-domain matches).

        Pre-seeds bypass the correlation filter and are emitted directly in the
        next pipeline run.
        """
        self._preseeds.append(candidate)

    def drain_preseeds(self) -> list[CorrelationCandidate]:
        """Return and clear all queued pre-seeds."""
        seeds = list(self._preseeds)
        self._preseeds.clear()
        return seeds

    async def mine(
        self,
        observations_by_context: dict[str, list[dict[str, Any]]],
    ) -> list[CorrelationCandidate]:
        """
        Mine for cross-context correlations.

        Args:
            observations_by_context: Mapping of context_id to list of observation dicts.
                Each observation dict maps variable names to numeric values.

        Returns:
            List of CorrelationCandidate that passed the consistency filter.
        """
        if len(observations_by_context) < self._config.min_cross_context_count:
            logger.debug(
                "insufficient_contexts",
                context_count=len(observations_by_context),
                min_required=self._config.min_cross_context_count,
            )
            return []

        # Collect all variable names across all contexts
        all_variables: set[str] = set()
        for obs_list in observations_by_context.values():
            for obs in obs_list:
                all_variables.update(
                    k for k, v in obs.items()
                    if isinstance(v, (int, float)) and not math.isnan(v)
                )

        variable_pairs = list(combinations(sorted(all_variables), 2))
        candidates: list[CorrelationCandidate] = []

        for var_a, var_b in variable_pairs:
            self._pairs_evaluated += 1
            context_correlations: list[ContextCorrelation] = []

            for context_id, obs_list in observations_by_context.items():
                # Extract paired values for this context
                values_a: list[float] = []
                values_b: list[float] = []
                for obs in obs_list:
                    a_val = obs.get(var_a)
                    b_val = obs.get(var_b)
                    if (
                        isinstance(a_val, (int, float))
                        and isinstance(b_val, (int, float))
                        and not math.isnan(a_val)
                        and not math.isnan(b_val)
                    ):
                        values_a.append(float(a_val))
                        values_b.append(float(b_val))

                if len(values_a) < 3:
                    continue  # Need at least 3 paired observations

                r = self._pearson_correlation(values_a, values_b)
                if r is not None:
                    context_correlations.append(
                        ContextCorrelation(
                            context_id=context_id,
                            correlation=r,
                            sample_count=len(values_a),
                        )
                    )

            # Need enough contexts to evaluate cross-context consistency
            if len(context_correlations) < self._config.min_cross_context_count:
                continue

            correlations = [cc.correlation for cc in context_correlations]
            mean_r = sum(correlations) / len(correlations)
            variance_r = self._variance(correlations)

            # The filter: consistent and non-trivial
            if (
                abs(mean_r) > self._config.min_abs_mean_correlation
                and variance_r < self._config.max_cross_context_variance
            ):
                candidate = CorrelationCandidate(
                    variable_a=var_a,
                    variable_b=var_b,
                    mean_correlation=mean_r,
                    cross_context_variance=variance_r,
                    context_count=len(context_correlations),
                    context_correlations=context_correlations,
                )
                candidates.append(candidate)
                self._candidates_found += 1

        logger.info(
            "correlation_mining_complete",
            pairs_evaluated=self._pairs_evaluated,
            candidates_found=len(candidates),
            contexts=len(observations_by_context),
            variables=len(all_variables),
        )

        return candidates

    # --- Statistical helpers ---

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float]) -> float | None:
        """
        Compute Pearson correlation coefficient.

        Returns None if variance is zero in either variable (constant).
        """
        n = len(x)
        if n < 3:
            return None

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True)) / n
        var_x = sum((xi - mean_x) ** 2 for xi in x) / n
        var_y = sum((yi - mean_y) ** 2 for yi in y) / n

        if var_x < 1e-12 or var_y < 1e-12:
            return None  # Constant variable

        return cov_xy / math.sqrt(var_x * var_y)

    @staticmethod
    def _variance(values: list[float]) -> float:
        """Compute population variance."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    # --- Metrics ---

    @property
    def total_pairs_evaluated(self) -> int:
        return self._pairs_evaluated

    @property
    def total_candidates_found(self) -> int:
        return self._candidates_found
