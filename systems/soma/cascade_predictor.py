"""
EcodiaOS — Soma Cascade Predictor

Anticipates systemic cascades by combining the causal flow graph
(transfer entropy) with current interoceptive error signals.

The existing predictor (predictor.py) extrapolates 9D trends. This module
adds a fundamentally different prediction capability: structural
anticipation. It answers "if system X is stressed, which other systems
will feel it next, and how badly?"

This is analogous to how the brain's insular cortex can feel an
impending panic attack before it happens — not by extrapolating heart
rate, but by recognising the causal pattern that precedes one.

Architecture:
  - Reads the CausalFlowMap (transfer entropy matrix) from CausalFlowEngine
  - Reads current per-system state from StateVectorConstructor
  - For each system with above-threshold error/stress, propagates the
    stress through the TE graph using a damped diffusion model
  - Outputs a set of CascadeForecasts: "system X stress → system Y will
    degrade in ~N cycles with magnitude M"

Budget: <1ms per cycle (sparse matrix multiply on at most 29 systems).
Runs every cycle but only produces forecasts when stress is detected.

Novel contribution: This closes the loop between the causal flow
engine (which discovers the real-time causal topology) and the
predictor (which forecasts future states). The existing predictor
extrapolates the 9 interoceptive dimensions independently. This module
propagates stress through the actual measured causal architecture.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from systems.soma.types import (
    InteroceptiveDimension,
    InteroceptiveState,
)

logger = structlog.get_logger("systems.soma.cascade_predictor")


# ─── Output Types ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CascadeForecast:
    """A predicted downstream impact from a stressed system."""

    source_system: str
    target_system: str
    source_stress: float        # Current stress magnitude at source
    predicted_impact: float     # Predicted stress at target [0, 1]
    propagation_hops: int       # Causal path length
    estimated_cycles: int       # Cycles until impact arrives
    confidence: float           # Confidence in this forecast [0, 1]
    causal_chain: list[str]     # The path: [source, intermediate..., target]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_system,
            "target": self.target_system,
            "source_stress": round(self.source_stress, 4),
            "predicted_impact": round(self.predicted_impact, 4),
            "hops": self.propagation_hops,
            "est_cycles": self.estimated_cycles,
            "confidence": round(self.confidence, 3),
            "chain": self.causal_chain,
        }


@dataclass(slots=True)
class CascadeSnapshot:
    """Full cascade prediction for one cycle."""

    cycle_number: int = 0
    forecasts: list[CascadeForecast] = field(default_factory=list)
    total_cascade_risk: float = 0.0  # Aggregate risk score [0, 1]
    epicenter_system: str = ""       # Most causally dangerous stressed system
    at_risk_systems: list[str] = field(default_factory=list)  # Systems about to be hit

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle_number,
            "total_risk": round(self.total_cascade_risk, 4),
            "epicenter": self.epicenter_system,
            "at_risk": self.at_risk_systems,
            "forecast_count": len(self.forecasts),
            "top_forecasts": [f.to_dict() for f in self.forecasts[:5]],
        }


# ─── Constants ───────────────────────────────────────────────────

# Minimum stress level to consider a system as a cascade source
_STRESS_THRESHOLD: float = 0.15

# Damping factor per hop (stress decays through the causal graph)
_DAMPING_PER_HOP: float = 0.6

# Maximum causal hops to trace (avoid exponential blowup)
_MAX_HOPS: int = 3

# Minimum TE value to consider a causal link real
_MIN_TE_FOR_PROPAGATION: float = 0.03

# Estimated cycles per hop (based on typical Synapse propagation delay)
_CYCLES_PER_HOP: int = 5

# Minimum confidence to emit a forecast
_MIN_FORECAST_CONFIDENCE: float = 0.2


class CascadePredictor:
    """
    Predicts systemic cascades by propagating stress through the
    measured causal topology.

    The key insight: transfer entropy tells us not just which systems
    are correlated, but which systems causally influence others. When
    system A is stressed and TE(A→B) is high, system B is likely to
    feel the stress within a few cycles. This is structural prediction,
    not trend extrapolation.
    """

    def __init__(
        self,
        stress_threshold: float = _STRESS_THRESHOLD,
        damping: float = _DAMPING_PER_HOP,
        max_hops: int = _MAX_HOPS,
    ) -> None:
        self._stress_threshold = stress_threshold
        self._damping = damping
        self._max_hops = max_hops

        # Cached TE matrix and system ordering from CausalFlowEngine
        self._te_matrix: np.ndarray | None = None
        self._system_ids: list[str] = []
        self._sys_to_idx: dict[str, int] = {}

        # Per-system stress levels (refreshed each cycle)
        self._system_stress: dict[str, float] = {}

        # Forecast history for validation (did predictions come true?)
        self._forecast_history: deque[CascadeSnapshot] = deque(maxlen=20)
        self._last_snapshot: CascadeSnapshot | None = None

        # Prediction accuracy tracking
        self._predictions_made: int = 0
        self._predictions_validated: int = 0
        self._accuracy_ema: float = 0.5  # Start at coin-flip

    # ─── Causal Graph Update ──────────────────────────────────────

    def update_causal_graph(
        self,
        te_matrix: np.ndarray,
        system_ids: list[str],
    ) -> None:
        """
        Accept a new transfer entropy matrix from CausalFlowEngine.
        Called every ~100 cycles when CausalFlowEngine recomputes.
        """
        self._te_matrix = te_matrix.copy()
        self._system_ids = list(system_ids)
        self._sys_to_idx = {s: i for i, s in enumerate(system_ids)}

    # ─── Per-System Stress Injection ──────────────────────────────

    def update_system_stress(self, system_id: str, stress: float) -> None:
        """
        Update the current stress level for a system.
        Called every cycle from StateVectorConstructor output.

        Stress is derived from the system's error_rate and success_ratio
        in the OrganismStateVector.
        """
        self._system_stress[system_id] = max(0.0, min(1.0, stress))

    def update_system_stresses_from_state(
        self,
        state: InteroceptiveState,
    ) -> None:
        """
        Derive per-interoceptive-dimension stress from allostatic errors
        and inject them as system-level stresses via the dimension→system
        mapping.
        """
        moment_errors = state.errors.get("moment", {})
        error_rates = state.error_rates

        # Map interoceptive dimensions to their primary source systems
        dim_to_system: dict[InteroceptiveDimension, str] = {
            InteroceptiveDimension.ENERGY: "synapse",
            InteroceptiveDimension.AROUSAL: "atune",
            InteroceptiveDimension.VALENCE: "atune",
            InteroceptiveDimension.CONFIDENCE: "atune",
            InteroceptiveDimension.COHERENCE: "synapse",
            InteroceptiveDimension.SOCIAL_CHARGE: "atune",
            InteroceptiveDimension.CURIOSITY_DRIVE: "atune",
            InteroceptiveDimension.INTEGRITY: "thymos",
            InteroceptiveDimension.TEMPORAL_PRESSURE: "nova",
        }

        for dim, system_id in dim_to_system.items():
            error = abs(moment_errors.get(dim, 0.0))
            rate = abs(error_rates.get(dim, 0.0))
            # Stress = error magnitude weighted by rate of change
            stress = min(1.0, error + rate * 0.5)
            # Only update if this is the worst stress for this system
            current = self._system_stress.get(system_id, 0.0)
            if stress > current:
                self._system_stress[system_id] = stress

    # ─── Main Prediction ──────────────────────────────────────────

    def predict(self, cycle_number: int) -> CascadeSnapshot:
        """
        Predict cascades from currently stressed systems.

        Steps:
          1. Identify stressed systems (above threshold)
          2. For each, trace causal paths through TE graph
          3. Compute damped impact at each downstream system
          4. Aggregate and rank forecasts
          5. Validate previous forecasts against current reality

        Budget: <1ms (sparse graph traversal on <=29 nodes).
        """
        # Validate previous predictions
        self._validate_previous()

        forecasts: list[CascadeForecast] = []

        if self._te_matrix is None or len(self._system_ids) < 2:
            snapshot = CascadeSnapshot(cycle_number=cycle_number)
            self._last_snapshot = snapshot
            return snapshot

        # Find stressed systems
        stressed = {
            sid: stress
            for sid, stress in self._system_stress.items()
            if stress > self._stress_threshold and sid in self._sys_to_idx
        }

        if not stressed:
            snapshot = CascadeSnapshot(cycle_number=cycle_number)
            self._last_snapshot = snapshot
            return snapshot

        # Propagate stress through causal graph
        for source_id, source_stress in stressed.items():
            source_forecasts = self._propagate_stress(
                source_id, source_stress,
            )
            forecasts.extend(source_forecasts)

        # Sort by predicted impact (highest first)
        forecasts.sort(key=lambda f: f.predicted_impact, reverse=True)

        # Compute aggregate risk
        total_risk = 0.0
        if forecasts:
            # Risk = max individual impact + 0.2 * sum of others (diminishing)
            total_risk = forecasts[0].predicted_impact
            for f in forecasts[1:]:
                total_risk += f.predicted_impact * 0.2
            total_risk = min(1.0, total_risk)

        # Find epicenter (stressed system with most downstream impact)
        epicenter = ""
        if stressed:
            epicenter_scores: dict[str, float] = {}
            for f in forecasts:
                epicenter_scores[f.source_system] = (
                    epicenter_scores.get(f.source_system, 0.0) + f.predicted_impact
                )
            if epicenter_scores:
                epicenter = max(epicenter_scores, key=epicenter_scores.get)  # type: ignore[arg-type]

        # At-risk systems (targets of forecasts above minimum confidence)
        at_risk = list(dict.fromkeys(
            f.target_system
            for f in forecasts
            if f.confidence >= _MIN_FORECAST_CONFIDENCE
        ))

        snapshot = CascadeSnapshot(
            cycle_number=cycle_number,
            forecasts=forecasts,
            total_cascade_risk=total_risk,
            epicenter_system=epicenter,
            at_risk_systems=at_risk,
        )

        self._forecast_history.append(snapshot)
        self._last_snapshot = snapshot

        if forecasts:
            logger.debug(
                "cascade_forecast",
                cycle=cycle_number,
                risk=round(total_risk, 3),
                epicenter=epicenter,
                at_risk=at_risk[:3],
                n_forecasts=len(forecasts),
            )

        # Clear per-cycle stress accumulator
        self._system_stress.clear()

        return snapshot

    @property
    def last_snapshot(self) -> CascadeSnapshot | None:
        return self._last_snapshot

    @property
    def prediction_accuracy(self) -> float:
        return self._accuracy_ema

    # ─── Internal ──────────────────────────────────────────────────

    def _propagate_stress(
        self,
        source_id: str,
        source_stress: float,
    ) -> list[CascadeForecast]:
        """
        BFS through the TE graph from a stressed system,
        computing damped impact at each reachable downstream system.
        """
        assert self._te_matrix is not None
        forecasts: list[CascadeForecast] = []

        source_idx = self._sys_to_idx.get(source_id)
        if source_idx is None:
            return forecasts

        n = len(self._system_ids)

        # BFS with damping: (system_idx, current_impact, hop_count, path)
        queue: list[tuple[int, float, int, list[str]]] = [
            (source_idx, source_stress, 0, [source_id]),
        ]
        visited: set[int] = {source_idx}

        while queue:
            current_idx, current_impact, hops, path = queue.pop(0)

            if hops >= self._max_hops:
                continue

            # Follow causal outflows from current system
            for target_idx in range(n):
                if target_idx == current_idx or target_idx in visited:
                    continue

                te_value = float(self._te_matrix[current_idx, target_idx])
                if te_value < _MIN_TE_FOR_PROPAGATION:
                    continue

                # Damped impact: stress * TE strength * damping^hops
                propagated = current_impact * te_value * (self._damping ** hops)
                target_id = self._system_ids[target_idx]
                new_path = path + [target_id]

                # Confidence based on TE strength and path length
                confidence = min(1.0, te_value / 0.15) * (0.8 ** hops)

                if propagated > 0.01 and confidence >= _MIN_FORECAST_CONFIDENCE:
                    forecasts.append(CascadeForecast(
                        source_system=source_id,
                        target_system=target_id,
                        source_stress=source_stress,
                        predicted_impact=min(1.0, propagated),
                        propagation_hops=hops + 1,
                        estimated_cycles=(hops + 1) * _CYCLES_PER_HOP,
                        confidence=confidence,
                        causal_chain=new_path,
                    ))

                    # Continue BFS from this target
                    visited.add(target_idx)
                    queue.append((target_idx, propagated, hops + 1, new_path))

        return forecasts

    def _validate_previous(self) -> None:
        """
        Check if previous cascade forecasts came true.
        This feeds back into prediction accuracy tracking.
        """
        if not self._forecast_history:
            return

        # Look at forecasts from 5-15 cycles ago
        for snapshot in self._forecast_history:
            for forecast in snapshot.forecasts:
                if forecast.estimated_cycles > 15:
                    continue  # Too far in the future to validate yet

                target_stress = self._system_stress.get(
                    forecast.target_system, 0.0,
                )

                predicted_hit = forecast.predicted_impact > _STRESS_THRESHOLD
                actual_hit = target_stress > _STRESS_THRESHOLD

                self._predictions_made += 1
                if predicted_hit == actual_hit:
                    self._predictions_validated += 1
                    self._accuracy_ema = self._accuracy_ema * 0.95 + 1.0 * 0.05
                else:
                    self._accuracy_ema = self._accuracy_ema * 0.95 + 0.0 * 0.05

        # Only keep unvalidated forecasts
        # (history is bounded by deque maxlen, so just let it rotate)
