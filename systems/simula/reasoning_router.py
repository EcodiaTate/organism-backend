"""
EcodiaOS - Simula Reasoning Router

Thompson sampling router for selecting proof strategies (Z3, Lean, Dafny,
static analysis) based on historical success rates. Persists arm weights
for genome extraction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger("systems.simula.reasoning_router")


@dataclass
class ArmStats:
    """Per-strategy Thompson sampling statistics."""

    successes: float = 1.0  # Beta prior alpha
    failures: float = 1.0  # Beta prior beta

    @property
    def sample(self) -> float:
        return random.betavariate(self.successes, self.failures)

    def update(self, success: bool) -> None:
        if success:
            self.successes += 1.0
        else:
            self.failures += 1.0


class ReasoningRouter:
    """
    Thompson sampling router for proof strategy selection.

    Arms correspond to proof strategies: z3, lean, dafny, static_analysis.
    On each proof request, samples from each arm's Beta posterior and
    selects the strategy with the highest sample.
    """

    DEFAULT_STRATEGIES = ("z3", "lean", "dafny", "static_analysis")

    def __init__(self, strategies: tuple[str, ...] | None = None) -> None:
        self._strategies = strategies or self.DEFAULT_STRATEGIES
        self._arms: dict[str, ArmStats] = {s: ArmStats() for s in self._strategies}
        self._log = logger.bind(component="reasoning_router")

    def select_strategy(self, available: list[str] | None = None) -> str:
        """Select the best proof strategy via Thompson sampling."""
        ranked = self.rank_strategies(available)
        return ranked[0] if ranked else self._strategies[0]

    def rank_strategies(self, available: list[str] | None = None) -> list[str]:
        """Rank all available strategies by Thompson sampling (highest first)."""
        candidates = available or list(self._strategies)
        if not candidates:
            return list(self._strategies)

        samples: list[tuple[str, float]] = []
        for strategy in candidates:
            arm = self._arms.get(strategy)
            if arm is None:
                arm = ArmStats()
                self._arms[strategy] = arm
            samples.append((strategy, arm.sample))

        samples.sort(key=lambda x: x[1], reverse=True)

        self._log.debug(
            "strategies_ranked",
            ranking=[(s, f"{v:.3f}") for s, v in samples],
        )
        return [s for s, _ in samples]

    def update(self, strategy: str, success: bool) -> None:
        """Update the arm's posterior after observing a proof outcome."""
        arm = self._arms.get(strategy)
        if arm is None:
            arm = ArmStats()
            self._arms[strategy] = arm
        arm.update(success)
        self._log.debug(
            "arm_updated",
            strategy=strategy,
            success=success,
            alpha=arm.successes,
            beta=arm.failures,
        )

    def get_weights(self) -> dict[str, dict[str, float]]:
        """Export arm weights for genome extraction."""
        return {
            name: {"successes": arm.successes, "failures": arm.failures}
            for name, arm in self._arms.items()
        }

    def load_weights(self, weights: dict[str, dict[str, float]]) -> None:
        """Load arm weights from genome seeding."""
        for name, w in weights.items():
            self._arms[name] = ArmStats(
                successes=w.get("successes", 1.0),
                failures=w.get("failures", 1.0),
            )
        self._log.info("weights_loaded", strategies=list(weights.keys()))
