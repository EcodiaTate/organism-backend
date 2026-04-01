"""
EcodiaOS - Soma Ricci Curvature Analyzer

Regions of negative curvature on the state manifold are where small
perturbations amplify - the organism is fragile there. Positive curvature
means perturbations contract - the organism is robust. This enables
preventive intervention before fragility becomes failure.

Implements Ollivier's discrete Ricci curvature (2009):

    kappa(x, y) = 1 - W1(mu_x, mu_y) / d(x, y)

where W1 is the Wasserstein-1 (earth mover's) distance between
neighborhood distributions mu_x and mu_y, and d is the Fisher
geodesic distance.

Designed for the deep analysis path only (~75s interval) because
the Wasserstein computation is O(k^3 log k) per pair.

Dependencies: numpy, scipy (linear_sum_assignment for optimal transport).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import structlog
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from systems.soma.fisher_manifold import FisherManifold

logger = structlog.get_logger("systems.soma.curvature_analyzer")


# ─── Output Types ────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class CurvatureMap:
    """Result of Ricci curvature analysis at a point on the manifold."""

    overall_scalar_curvature: float
    """Mean Ollivier-Ricci curvature across all neighbor pairs.
    Positive = robust, negative = fragile, zero = flat."""

    per_system_curvature: dict[int, float]
    """Per-dimension average curvature. Keyed by dimension index.
    Negative values indicate fragile dimensions."""

    vulnerable_pairs: list[tuple[int, int, float]]
    """(dim_a, dim_b, curvature) triples for the most fragile
    interactions, sorted by curvature ascending (most negative first)."""

    most_vulnerable_region: int
    """Dimension index with the most negative curvature."""


# ─── Curvature Analyzer ─────────────────────────────────────────


class CurvatureAnalyzer:
    """
    Computes Ollivier discrete Ricci curvature on the Fisher manifold.

    Given a point x and its k-nearest neighbors under the Fisher metric,
    the curvature between x and each neighbor y is:

        kappa(x, y) = 1 - W1(mu_x, mu_y) / d(x, y)

    where mu_x and mu_y are the empirical distributions over the
    k-nearest neighborhoods of x and y respectively, and W1 is the
    Wasserstein-1 distance computed via optimal transport.

    The analyzer decomposes curvature by dimension pairs to identify
    which interactions are fragile (negative curvature = small
    perturbations in those dimensions amplify).
    """

    def __init__(
        self,
        k_neighbors: int = 15,
        max_vulnerable_pairs: int = 10,
    ) -> None:
        self._k = k_neighbors
        self._max_vulnerable_pairs = max_vulnerable_pairs

    def analyze(
        self,
        point: np.ndarray,
        manifold: FisherManifold,
    ) -> CurvatureMap | None:
        """
        Compute Ricci curvature at the given point on the manifold.

        Uses the manifold's rolling window as the point cloud and its
        Fisher metric for distance computation.

        Returns None if the manifold has insufficient data.
        """
        point = np.asarray(point, dtype=np.float64).ravel()

        if not manifold.has_fisher or manifold.window_size < self._k * 2:
            return None

        # Materialize the point cloud from the manifold's window
        cloud = np.array(list(manifold.window_vectors))  # (N, D)
        n_points = cloud.shape[0]

        if n_points < self._k + 1:
            return None

        # Compute Fisher distances from point to all cloud vectors
        dists_from_point = np.array([
            manifold.geodesic_distance(point, cloud[i])
            for i in range(n_points)
        ])

        # k-nearest neighbors of point (excluding exact duplicates at dist=0)
        neighbor_indices = np.argsort(dists_from_point)
        neighbor_indices = neighbor_indices[dists_from_point[neighbor_indices] > 1e-12]
        if len(neighbor_indices) < self._k:
            return None
        knn_point = neighbor_indices[: self._k]

        # For each neighbor y, compute kappa(point, y)
        curvatures: list[float] = []
        neighbor_curvatures: list[tuple[int, float]] = []

        for y_idx in knn_point:
            y = cloud[y_idx]
            d_xy = dists_from_point[y_idx]
            if d_xy < 1e-12:
                continue

            # k-nearest neighbors of y
            dists_from_y = np.array([
                manifold.geodesic_distance(y, cloud[j])
                for j in range(n_points)
            ])
            knn_y_indices = np.argsort(dists_from_y)
            knn_y_indices = knn_y_indices[dists_from_y[knn_y_indices] > 1e-12]
            if len(knn_y_indices) < self._k:
                continue
            knn_y = knn_y_indices[: self._k]

            # Wasserstein-1 distance between uniform distributions
            # over knn_point and knn_y neighborhoods
            w1 = self._wasserstein1(cloud, knn_point, knn_y, manifold)

            kappa = 1.0 - w1 / d_xy
            curvatures.append(kappa)
            neighbor_curvatures.append((y_idx, kappa))

        if not curvatures:
            return None

        overall = float(np.mean(curvatures))

        # Decompose curvature by dimension.
        # For each dimension d, compute the average curvature weighted by
        # how much that dimension contributes to the displacement between
        # point and each neighbor.
        dim = point.shape[0]
        dim_curvature_sum: dict[int, float] = {d: 0.0 for d in range(dim)}
        dim_curvature_count: dict[int, float] = {d: 0.0 for d in range(dim)}

        for y_idx, kappa in neighbor_curvatures:
            diff = np.abs(point - cloud[y_idx])
            diff_total = diff.sum()
            if diff_total < 1e-12:
                continue
            weights = diff / diff_total
            for d in range(dim):
                dim_curvature_sum[d] += kappa * weights[d]
                dim_curvature_count[d] += weights[d]

        per_system_curv: dict[int, float] = {}
        for d in range(dim):
            if dim_curvature_count[d] > 1e-12:
                per_system_curv[d] = dim_curvature_sum[d] / dim_curvature_count[d]
            else:
                per_system_curv[d] = 0.0

        # Vulnerable pairs: dimension pairs with most negative joint curvature.
        pair_scores: list[tuple[int, int, float]] = []
        for i in range(dim):
            for j in range(i + 1, dim):
                joint = (per_system_curv[i] + per_system_curv[j]) / 2.0
                pair_scores.append((i, j, joint))

        pair_scores.sort(key=lambda t: t[2])
        vulnerable = pair_scores[: self._max_vulnerable_pairs]

        most_vulnerable = min(per_system_curv, key=lambda d: per_system_curv[d])

        return CurvatureMap(
            overall_scalar_curvature=overall,
            per_system_curvature=per_system_curv,
            vulnerable_pairs=vulnerable,
            most_vulnerable_region=most_vulnerable,
        )

    # ─── Wasserstein-1 via Optimal Transport ─────────────────────

    def _wasserstein1(
        self,
        cloud: np.ndarray,
        indices_a: np.ndarray,
        indices_b: np.ndarray,
        manifold: FisherManifold,
    ) -> float:
        """
        Compute Wasserstein-1 distance between two uniform empirical
        distributions (neighborhoods A and B) using the Fisher metric.

        Uses the Hungarian algorithm (scipy linear_sum_assignment) on the
        cost matrix of pairwise Fisher distances. For equal-sized uniform
        distributions, W1 = min-cost matching / k.
        """
        k_a = len(indices_a)
        k_b = len(indices_b)
        k = min(k_a, k_b)

        if k == 0:
            return 0.0

        cost = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                cost[i, j] = manifold.geodesic_distance(
                    cloud[indices_a[i]], cloud[indices_b[j]],
                )

        row_ind, col_ind = linear_sum_assignment(cost)
        total_cost = cost[row_ind, col_ind].sum()

        return float(total_cost / k)
