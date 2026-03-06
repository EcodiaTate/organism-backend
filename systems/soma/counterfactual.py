"""
EcodiaOS — Soma Counterfactual Engine

Generates counterfactual interoceptive trajectories during Oneiros REM replay.
Given a decision and an alternative policy, simulates what the organism's
felt state would have been — enabling regret, gratitude, and lesson extraction.

This is the computational basis of "what if I had done X instead?"
applied to the organism's internal states, not just external outcomes.

Only active at REFLECTIVE stage and above.
"""

from __future__ import annotations

import structlog

from systems.soma.types import (
    ALL_DIMENSIONS,
    CounterfactualTrace,
    InteroceptiveDimension,
)

logger = structlog.get_logger("systems.soma.counterfactual")


class CounterfactualEngine:
    """
    Simulates alternative interoceptive trajectories for decision evaluation.

    During Oneiros REM replay, takes actual decision records and simulates
    what the organism would have felt under different choices. The resulting
    regret/gratitude signals feed back to Nova (EFE calibration),
    Memory (somatic markers on counterfactuals), and Thread (narrative turning points).
    """

    def __init__(self) -> None:
        self._dynamics: list[list[float]] | None = None  # Shared with predictor

    def set_dynamics(self, dynamics: list[list[float]]) -> None:
        """Share the predictor's dynamics matrix for consistent simulation."""
        self._dynamics = dynamics

    def generate_counterfactual(
        self,
        decision_id: str,
        actual_trajectory: list[dict[InteroceptiveDimension, float]],
        alternative_description: str,
        alternative_initial_impact: dict[InteroceptiveDimension, float],
        setpoints: dict[InteroceptiveDimension, float],
        num_steps: int = 10,
    ) -> CounterfactualTrace:
        """
        Generate a counterfactual interoceptive trajectory.

        Args:
            decision_id: ID of the original decision
            actual_trajectory: Actual sensed states after the decision (5s windows)
            alternative_description: What the alternative policy would have done
            alternative_initial_impact: How the alternative would have shifted
                the initial state (delta per dimension)
            setpoints: Current allostatic setpoints
            num_steps: How many steps to simulate

        Returns:
            CounterfactualTrace with regret, gratitude, and lesson
        """
        if not actual_trajectory:
            return CounterfactualTrace(
                decision_id=decision_id,
                counterfactual_policy_description=alternative_description,
                lesson="Insufficient data for counterfactual simulation",
            )

        # Start from the state at decision time, apply alternative impact
        initial = dict(actual_trajectory[0])
        for dim, delta in alternative_initial_impact.items():
            if dim in initial:
                initial[dim] = max(
                    -1.0, min(1.0, initial[dim] + delta),
                )

        # Simulate forward using simple dynamics
        cf_trajectory = self._simulate(initial, num_steps)

        # Compute regret and gratitude
        regret, gratitude = self._compute_regret_gratitude(
            actual_trajectory, cf_trajectory, setpoints,
        )

        # Extract lesson
        lesson = self._extract_lesson(
            actual_trajectory, cf_trajectory, regret, gratitude,
            alternative_description,
        )

        return CounterfactualTrace(
            decision_id=decision_id,
            chosen_trajectory=actual_trajectory[:num_steps],
            counterfactual_trajectory=cf_trajectory,
            counterfactual_policy_description=alternative_description,
            regret=regret,
            gratitude=gratitude,
            lesson=lesson,
        )

    def _simulate(
        self,
        initial: dict[InteroceptiveDimension, float],
        num_steps: int,
    ) -> list[dict[InteroceptiveDimension, float]]:
        """
        Simple forward simulation using dynamics matrix.
        Each step ≈ 5 seconds of trajectory.
        """
        trajectory = [dict(initial)]
        current = dict(initial)

        for _ in range(num_steps - 1):
            next_state: dict[InteroceptiveDimension, float] = {}
            for i, dim_i in enumerate(ALL_DIMENSIONS):
                val = current.get(dim_i, 0.0)
                # Apply dynamics coupling if available
                if self._dynamics is not None:
                    for j, dim_j in enumerate(ALL_DIMENSIONS):
                        val += self._dynamics[i][j] * current.get(dim_j, 0.0) * 0.01
                # Mean reversion toward 0.5 (homeostatic pull)
                val += (0.5 - val) * 0.02
                lo, hi = (-1.0, 1.0) if dim_i == InteroceptiveDimension.VALENCE else (0.0, 1.0)
                next_state[dim_i] = max(lo, min(hi, val))

            trajectory.append(next_state)
            current = next_state

        return trajectory

    def _compute_regret_gratitude(
        self,
        actual: list[dict[InteroceptiveDimension, float]],
        counterfactual: list[dict[InteroceptiveDimension, float]],
        setpoints: dict[InteroceptiveDimension, float],
    ) -> tuple[float, float]:
        """
        Compute regret and gratitude by comparing allostatic error trajectories.

        Regret: how much better the counterfactual would have been (0-1)
        Gratitude: how much better the actual was than the counterfactual (0-1)
        """
        n = min(len(actual), len(counterfactual))
        if n == 0:
            return 0.0, 0.0

        actual_total_error = 0.0
        cf_total_error = 0.0

        for i in range(n):
            for dim in ALL_DIMENSIONS:
                sp = setpoints.get(dim, 0.5)
                actual_err = abs(actual[i].get(dim, 0.0) - sp)
                cf_err = abs(counterfactual[i].get(dim, 0.0) - sp)
                actual_total_error += actual_err
                cf_total_error += cf_err

        # Normalize by number of steps * dimensions
        norm = n * len(ALL_DIMENSIONS) if n > 0 else 1.0

        if cf_total_error < actual_total_error:
            # Counterfactual was better — regret
            regret = min(1.0, (actual_total_error - cf_total_error) / norm)
            gratitude = 0.0
        else:
            # Actual was better — gratitude
            regret = 0.0
            gratitude = min(1.0, (cf_total_error - actual_total_error) / norm)

        return regret, gratitude

    def _extract_lesson(
        self,
        actual: list[dict[InteroceptiveDimension, float]],
        counterfactual: list[dict[InteroceptiveDimension, float]],
        regret: float,
        gratitude: float,
        alternative_description: str,
    ) -> str:
        """
        Generate a natural language lesson from the counterfactual comparison.
        """
        if regret > gratitude:
            # Find which dimensions suffered most
            worst_dims = self._worst_dimensions(actual, counterfactual)
            dim_names = ", ".join(d.value for d in worst_dims[:3])
            return (
                f"Alternative ({alternative_description}) would have improved "
                f"{dim_names}. Regret={regret:.2f}."
            )
        elif gratitude > 0.05:
            best_dims = self._best_dimensions(actual, counterfactual)
            dim_names = ", ".join(d.value for d in best_dims[:3])
            return (
                f"Chosen path was better for {dim_names}. "
                f"Gratitude={gratitude:.2f}."
            )
        else:
            return "Both paths were roughly equivalent in interoceptive outcome."

    def _worst_dimensions(
        self,
        actual: list[dict[InteroceptiveDimension, float]],
        counterfactual: list[dict[InteroceptiveDimension, float]],
    ) -> list[InteroceptiveDimension]:
        """Dimensions where actual was worse than counterfactual."""
        diff: dict[InteroceptiveDimension, float] = {d: 0.0 for d in ALL_DIMENSIONS}
        n = min(len(actual), len(counterfactual))
        for i in range(n):
            for dim in ALL_DIMENSIONS:
                # Positive diff = actual was further from 0.5 (worse)
                a_dist = abs(actual[i].get(dim, 0.0) - 0.5)
                c_dist = abs(counterfactual[i].get(dim, 0.0) - 0.5)
                diff[dim] += a_dist - c_dist

        ranked = sorted(diff.items(), key=lambda x: x[1], reverse=True)
        return [dim for dim, _ in ranked]

    def _best_dimensions(
        self,
        actual: list[dict[InteroceptiveDimension, float]],
        counterfactual: list[dict[InteroceptiveDimension, float]],
    ) -> list[InteroceptiveDimension]:
        """Dimensions where actual was better than counterfactual."""
        diff: dict[InteroceptiveDimension, float] = {d: 0.0 for d in ALL_DIMENSIONS}
        n = min(len(actual), len(counterfactual))
        for i in range(n):
            for dim in ALL_DIMENSIONS:
                a_dist = abs(actual[i].get(dim, 0.0) - 0.5)
                c_dist = abs(counterfactual[i].get(dim, 0.0) - 0.5)
                diff[dim] += c_dist - a_dist  # Positive = actual was better

        ranked = sorted(diff.items(), key=lambda x: x[1], reverse=True)
        return [dim for dim, _ in ranked]
