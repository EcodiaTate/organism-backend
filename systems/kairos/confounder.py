"""
EcodiaOS — Kairos Stage 3: Confounder Analysis

Full PC (Peter-Clark) algorithm implementation for confounder detection.

Spurious correlations are expensive: they add description length without adding
causal structure. Every confounder found is a net compression gain.

Before: store A->B edge (spurious) + C->A + C->B = 3 edges
After:  store C->A + C->B, delete A->B = 2 edges

The PC algorithm:
1. Start with complete undirected graph over all variables
2. Remove edges where conditional independence holds
3. Orient remaining edges using d-separation rules
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import structlog

from systems.kairos.types import (
    CausalDirectionResult,
    ConfirmedConfounder,
    ConfounderResult,
    KairosConfig,
)

logger = structlog.get_logger("kairos.confounder")


class ConfounderAnalyzer:
    """
    Stage 3 of the Kairos pipeline.

    Implements the PC algorithm for confounder detection.
    Tests conditional independence: are A and B independent given C?
    If yes, C is a confounder and the A-B link is spurious.
    """

    def __init__(self, config: KairosConfig | None = None) -> None:
        self._config = config or KairosConfig()
        self._confounders_found: int = 0
        self._analyses_run: int = 0

    async def analyze(
        self,
        direction_result: CausalDirectionResult,
        all_observations: dict[str, list[dict[str, Any]]],
        candidate_confounders: list[str] | None = None,
    ) -> ConfounderResult:
        """
        Run confounder analysis for a confirmed causal direction.

        Args:
            direction_result: Stage 2 output with accepted direction.
            all_observations: context_id to list of observation dicts (all variables).
            candidate_confounders: Optional list of variable names to test as confounders.
                If None, all other numeric variables are tested.

        Returns:
            ConfounderResult indicating whether the relationship is confounded.
        """
        self._analyses_run += 1
        var_a = direction_result.candidate.variable_a
        var_b = direction_result.candidate.variable_b

        # Flatten all observations into a single dataset for partial correlation
        flat_obs: list[dict[str, float]] = []
        for obs_list in all_observations.values():
            for obs in obs_list:
                numeric = {
                    k: float(v)
                    for k, v in obs.items()
                    if isinstance(v, (int, float)) and not math.isnan(v)
                }
                if var_a in numeric and var_b in numeric:
                    flat_obs.append(numeric)

        if len(flat_obs) < 10:
            logger.debug(
                "insufficient_data_for_confounder_analysis",
                sample_count=len(flat_obs),
            )
            return ConfounderResult(
                original_pair=direction_result,
                is_confounded=False,
                adjusted_correlation=direction_result.candidate.mean_correlation,
            )

        # Identify candidate confounders
        if candidate_confounders is None:
            all_vars: set[str] = set()
            for obs in flat_obs:
                all_vars.update(obs.keys())
            candidate_confounders = sorted(all_vars - {var_a, var_b})

        # Test each candidate confounder
        confirmed: list[ConfirmedConfounder] = []
        original_r = _compute_correlation(flat_obs, var_a, var_b)

        for c_var in candidate_confounders:
            partial_r = _partial_correlation(flat_obs, var_a, var_b, c_var)
            if partial_r is None:
                continue

            # If partial correlation drops significantly, C is confounding
            drop = abs(abs(original_r) - abs(partial_r))
            relative_drop = drop / max(abs(original_r), 1e-9)

            if (
                relative_drop > self._config.min_partial_correlation_drop
                and abs(partial_r) < self._config.conditional_independence_threshold + 0.1
            ):
                mdl_improvement = 50.0 * (1.0 - abs(partial_r))

                confirmed.append(
                    ConfirmedConfounder(
                        variable=c_var,
                        conditional_independence_score=1.0 - abs(partial_r),
                        partial_correlation_residual=partial_r,
                        mdl_improvement=mdl_improvement,
                    )
                )

        is_confounded = len(confirmed) > 0
        if is_confounded:
            self._confounders_found += len(confirmed)
            best = max(confirmed, key=lambda c: c.conditional_independence_score)
            adjusted_r = _partial_correlation(flat_obs, var_a, var_b, best.variable)
            adjusted_correlation = adjusted_r if adjusted_r is not None else 0.0
        else:
            adjusted_correlation = original_r

        # Run simplified PC algorithm on the variable set
        pc_removed, pc_oriented = self._run_pc_skeleton(
            flat_obs, var_a, var_b, candidate_confounders
        )

        total_mdl = sum(c.mdl_improvement for c in confirmed)

        result = ConfounderResult(
            original_pair=direction_result,
            is_confounded=is_confounded,
            confounding_variables=confirmed,
            adjusted_correlation=adjusted_correlation,
            mdl_improvement=total_mdl,
            pc_edges_removed=pc_removed,
            pc_edges_oriented=pc_oriented,
        )

        logger.info(
            "confounder_analysis_complete",
            candidate_id=direction_result.candidate.id,
            is_confounded=is_confounded,
            confounders_found=len(confirmed),
            mdl_improvement=round(total_mdl, 1),
            original_r=round(original_r, 3),
            adjusted_r=round(adjusted_correlation, 3),
        )

        return result

    # --- PC Algorithm ---

    def _run_pc_skeleton(
        self,
        observations: list[dict[str, float]],
        var_a: str,
        var_b: str,
        other_vars: list[str],
    ) -> tuple[int, int]:
        """
        Run simplified PC algorithm skeleton construction.

        Returns (edges_removed, edges_oriented).

        Phase 1: Start with complete graph, remove edges where
        conditional independence holds.
        Phase 2: Orient edges using v-structures (colliders).
        """
        variables = [var_a, var_b] + other_vars[:20]  # Cap for performance
        n_vars = len(variables)

        if n_vars < 3:
            return 0, 0

        # Phase 1: Skeleton construction
        adj: dict[str, set[str]] = {v: set(variables) - {v} for v in variables}
        separation_sets: dict[tuple[str, str], list[str]] = {}
        edges_removed = 0

        for cond_size in range(n_vars - 1):
            for v_i, v_j in combinations(variables, 2):
                if v_j not in adj[v_i]:
                    continue

                neighbors = (adj[v_i] | adj[v_j]) - {v_i, v_j}
                if len(neighbors) < cond_size:
                    continue

                for cond_set in combinations(sorted(neighbors), min(cond_size, len(neighbors))):
                    if not cond_set and cond_size > 0:
                        continue

                    is_independent = self._test_conditional_independence(
                        observations, v_i, v_j, list(cond_set)
                    )

                    if is_independent:
                        adj[v_i].discard(v_j)
                        adj[v_j].discard(v_i)
                        separation_sets[(v_i, v_j)] = list(cond_set)
                        separation_sets[(v_j, v_i)] = list(cond_set)
                        edges_removed += 1
                        break

            if cond_size >= 2:
                break  # Limit conditioning set size

        # Phase 2: V-structure orientation (collider detection)
        edges_oriented = 0
        for v_i, v_j in combinations(variables, 2):
            if v_j in adj[v_i]:
                continue  # Still adjacent

            sep_set = separation_sets.get((v_i, v_j), [])
            common_neighbors = adj[v_i] & adj[v_j]

            for z in common_neighbors:
                if z not in sep_set:
                    edges_oriented += 1

        return edges_removed, edges_oriented

    def _test_conditional_independence(
        self,
        observations: list[dict[str, float]],
        var_x: str,
        var_y: str,
        conditioning: list[str],
    ) -> bool:
        """
        Test whether X is independent of Y given the conditioning set.
        Uses partial correlation.
        """
        if not conditioning:
            r = _compute_correlation(observations, var_x, var_y)
            return abs(r) < self._config.conditional_independence_threshold

        partial_r = _partial_correlation_multivariate(
            observations, var_x, var_y, conditioning
        )
        if partial_r is None:
            return False
        return abs(partial_r) < self._config.conditional_independence_threshold

    # --- Metrics ---

    @property
    def total_analyses_run(self) -> int:
        return self._analyses_run

    @property
    def total_confounders_found(self) -> int:
        return self._confounders_found


# =====================================================================
# Module-level statistical functions (shared across the analyzer)
# =====================================================================


def _compute_correlation(
    observations: list[dict[str, float]],
    var_a: str,
    var_b: str,
) -> float:
    """Compute Pearson correlation from flat observation list."""
    values_a: list[float] = []
    values_b: list[float] = []
    for obs in observations:
        a_val = obs.get(var_a)
        b_val = obs.get(var_b)
        if a_val is not None and b_val is not None:
            values_a.append(a_val)
            values_b.append(b_val)

    n = len(values_a)
    if n < 3:
        return 0.0

    mean_a = sum(values_a) / n
    mean_b = sum(values_b) / n

    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b, strict=True)) / n
    var_a_val = sum((a - mean_a) ** 2 for a in values_a) / n
    var_b_val = sum((b - mean_b) ** 2 for b in values_b) / n

    if var_a_val < 1e-12 or var_b_val < 1e-12:
        return 0.0

    return cov / math.sqrt(var_a_val * var_b_val)


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


def _partial_correlation(
    observations: list[dict[str, float]],
    var_x: str,
    var_y: str,
    var_z: str,
) -> float | None:
    """
    Compute partial correlation r(X, Y | Z).

    Formula: r(X,Y|Z) = (r_xy - r_xz * r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
    """
    triples: list[tuple[float, float, float]] = []
    for obs in observations:
        x = obs.get(var_x)
        y = obs.get(var_y)
        z = obs.get(var_z)
        if x is not None and y is not None and z is not None:
            triples.append((x, y, z))

    if len(triples) < 5:
        return None

    xs = [t[0] for t in triples]
    ys = [t[1] for t in triples]
    zs = [t[2] for t in triples]

    r_xy = _pearson(xs, ys)
    r_xz = _pearson(xs, zs)
    r_yz = _pearson(ys, zs)

    if r_xy is None or r_xz is None or r_yz is None:
        return None

    denom = math.sqrt(max(1 - r_xz**2, 1e-12) * max(1 - r_yz**2, 1e-12))
    if denom < 1e-12:
        return None

    return (r_xy - r_xz * r_yz) / denom


def _partial_correlation_multivariate(
    observations: list[dict[str, float]],
    var_x: str,
    var_y: str,
    conditioning: list[str],
) -> float | None:
    """
    Compute partial correlation controlling for multiple variables.

    Regresses out conditioning variables from both X and Y,
    then correlates the residuals.
    """
    if not conditioning:
        return _compute_correlation(observations, var_x, var_y)

    required = {var_x, var_y} | set(conditioning)
    complete: list[dict[str, float]] = [
        obs for obs in observations
        if all(obs.get(v) is not None for v in required)
    ]

    if len(complete) < max(5, len(required) + 2):
        return None

    x_values = [obs[var_x] for obs in complete]
    y_values = [obs[var_y] for obs in complete]
    cond_matrix = [[obs[c] for c in conditioning] for obs in complete]

    x_residuals = _multivariate_residuals(cond_matrix, x_values)
    y_residuals = _multivariate_residuals(cond_matrix, y_values)

    if x_residuals is None or y_residuals is None:
        return None

    return _pearson(x_residuals, y_residuals)


def _multivariate_residuals(
    predictors: list[list[float]],
    target: list[float],
) -> list[float] | None:
    """Compute residuals of regressing target on predictors using OLS."""
    n = len(target)
    k = len(predictors[0]) if predictors else 0
    if n < k + 2 or k == 0:
        return None

    # Add intercept column
    design = [[1.0] + row for row in predictors]  # noqa: N806
    k_full = k + 1

    # X^T X
    xtx = [[0.0] * k_full for _ in range(k_full)]
    for i in range(k_full):
        for j in range(k_full):
            xtx[i][j] = sum(design[r][i] * design[r][j] for r in range(n))

    # X^T y
    xty = [sum(design[r][i] * target[r] for r in range(n)) for i in range(k_full)]

    beta = _solve_linear(xtx, xty)
    if beta is None:
        return None

    return [target[r] - sum(beta[j] * design[r][j] for j in range(k_full)) for r in range(n)]


def _solve_linear(a_matrix: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a_matrix)]

    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        if abs(aug[col][col]) < 1e-12:
            return None

        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x
