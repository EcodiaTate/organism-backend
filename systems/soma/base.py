"""
EcodiaOS - Soma Abstract Base Classes

Strategy interfaces for hot-reloadable Soma components. The
NeuroplasticityBus can evolve these at runtime by swapping in new
subclasses discovered via the ``eos:events:code_evolved`` channel.

Two ABCs:
  BaseSomaPredictor       - multi-horizon interoceptive prediction
  BaseAllostaticRegulator - setpoint management, urgency, signal construction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from systems.soma.types import (
        AllostaticSignal,
        InteroceptiveDimension,
        InteroceptiveState,
    )


class BaseSomaPredictor(ABC):
    """
    Abstract interface for the multi-horizon interoceptive generative model.

    Concrete implementations maintain a trajectory buffer, a dynamics coupling
    matrix, and scheduled events. They must predict future interoceptive states
    at each requested horizon within a 1ms budget.
    """

    @abstractmethod
    def push_state(self, sensed: dict[InteroceptiveDimension, float]) -> None:
        """Buffer a new 9D sensed state into the trajectory."""

    @abstractmethod
    def predict_all_horizons(
        self,
        current: dict[InteroceptiveDimension, float],
        available_horizons: list[str],
    ) -> dict[str, dict[InteroceptiveDimension, float]]:
        """Predict interoceptive state at all available horizons. Budget: <=1ms."""

    @abstractmethod
    def compute_allostatic_errors(
        self,
        predictions: dict[str, dict[InteroceptiveDimension, float]],
        setpoints: dict[InteroceptiveDimension, float],
    ) -> dict[str, dict[InteroceptiveDimension, float]]:
        """Compute predicted - setpoint per horizon per dimension."""

    @abstractmethod
    def compute_error_rates(
        self,
        current_errors: dict[InteroceptiveDimension, float],
    ) -> dict[InteroceptiveDimension, float]:
        """Compute d(error)/dt - is each dimension improving or worsening?"""

    @abstractmethod
    def update_dynamics(self, new_dynamics: list[list[float]]) -> None:
        """Accept an updated 9x9 cross-dimension coupling matrix from Evo."""

    @property
    @abstractmethod
    def trajectory_length(self) -> int:
        """Number of states currently in the trajectory buffer."""

    # ── Non-abstract helpers that the service uses directly ──────────
    # Subclasses MUST expose these for phase-space and counterfactual wiring.

    @property
    @abstractmethod
    def raw_trajectory(self) -> Any:
        """Raw trajectory buffer (deque). Used by PhaseSpaceModel.update()."""

    @abstractmethod
    def compute_velocity(self) -> dict[InteroceptiveDimension, float]:
        """Current velocity vector. Used by PhaseSpaceModel for heading."""

    @property
    @abstractmethod
    def dynamics_matrix(self) -> list[list[float]]:
        """Raw dynamics matrix. Shared with CounterfactualEngine."""


class BaseAllostaticRegulator(ABC):
    """
    Abstract interface for allostatic regulation: setpoint management,
    urgency computation, precision weighting, and signal construction.

    Concrete implementations manage context-adaptive setpoints with EMA
    smoothing and must build the AllostaticSignal within a 0.5ms budget.
    """

    @property
    @abstractmethod
    def setpoints(self) -> dict[InteroceptiveDimension, float]:
        """Current allostatic setpoints per dimension."""

    @property
    @abstractmethod
    def urgency_threshold(self) -> float:
        """Urgency level that triggers Nova allostatic deliberation."""

    @abstractmethod
    def set_context(self, context: str) -> None:
        """Switch allostatic context (conversation, deep_processing, etc.)."""

    @abstractmethod
    def tick_setpoints(self) -> None:
        """EMA-smooth setpoints toward targets. Called every theta cycle."""

    @abstractmethod
    def compute_urgency(
        self,
        errors: dict[str, dict[InteroceptiveDimension, float]],
        error_rates: dict[InteroceptiveDimension, float],
    ) -> float:
        """Compute urgency as a composite need-to-act signal in [0, 1]."""

    @abstractmethod
    def find_dominant_error(
        self,
        errors: dict[str, dict[InteroceptiveDimension, float]],
    ) -> tuple[InteroceptiveDimension, float]:
        """Find the dimension with the largest moment-horizon error."""

    @abstractmethod
    def build_signal(
        self,
        state: InteroceptiveState,
        phase_snapshot: dict[str, Any],
        cycle_number: int,
    ) -> AllostaticSignal:
        """Construct the full AllostaticSignal from state + phase-space."""
