"""
Fovea -- Atune Integration Layer (Phase B)

The merged salience/attention engine. Atune provides the workspace architecture.
Fovea provides the salience computation via prediction error decomposition.

Together they implement: attention = precision-weighted prediction error.

CRITICAL: This module does NOT modify Atune files. It provides a bridge that:
1. Computes salience via prediction error decomposition (content/timing/magnitude/source/category/causal)
2. Replaces independent scoring with prediction_error.precision_weighted_salience
3. Computes a dynamic ignition threshold from the error distribution
4. Feeds WorkspaceCandidates into Atune's GlobalWorkspace
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import structlog

from .habituation import HabituationCompleteInfo, HabituationEngine
from .precision import PrecisionWeightComputer
from .prediction import FoveaPredictionEngine
from .types import (
    ErrorRoute,
    ErrorType,
    FoveaMetrics,
    FoveaPredictionError,
    PerceptContext,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from primitives.percept import Percept
    from systems.atune.types import WorkspaceCandidate
    from systems.atune.workspace import GlobalWorkspace

    from .protocols import LogosWorldModel

logger = structlog.get_logger("systems.fovea.integration")


# ---------------------------------------------------------------------------
# Head mapping: how the seven Atune heads map to prediction error dimensions
#
# Old Atune Head  | Fovea Error Type               | Notes
# ----------------|---------------------------------|----------------------------
# Novelty         | content_error                   | Content prediction error
# Risk            | causal_error                    | Causal prediction error
# Identity        | category_error (self-percepts)  | Self-model category error
# Goal            | causal_error * goal_relevance   | Causal weighted by goal
# Emotional       | magnitude_error (affective)     | Magnitude on affect signals
# Causal          | causal_error (direct)           | Direct causal structure
# Keyword         | source_error + category_error   | Source + category combined
# Economic        | magnitude_error + source_error  | Financial signals
# ---------------------------------------------------------------------------


_HEAD_ERROR_MAPPING: dict[str, dict[str, float]] = {
    "novelty": {ErrorType.CONTENT: 1.0},
    "risk": {ErrorType.CAUSAL: 1.0},
    "identity": {ErrorType.CATEGORY: 1.0},
    "goal": {ErrorType.CAUSAL: 0.7, ErrorType.CONTENT: 0.3},
    "emotional": {ErrorType.MAGNITUDE: 1.0},
    "causal": {ErrorType.CAUSAL: 1.0},
    "keyword": {ErrorType.SOURCE: 0.5, ErrorType.CATEGORY: 0.5},
    "economic": {ErrorType.MAGNITUDE: 0.5, ErrorType.SOURCE: 0.5},
}


# ---------------------------------------------------------------------------
# Dynamic ignition threshold
# ---------------------------------------------------------------------------


class DynamicIgnitionThreshold:
    """
    Computes the workspace ignition threshold from the recent error
    distribution using percentile-based thresholding.

    In low-surprise environments the threshold drops (EOS becomes more
    sensitive). In high-surprise environments it rises (EOS focuses on
    the most extreme errors).
    """

    def __init__(
        self,
        percentile: float = 75.0,
        window_size: int = 100,
        floor: float = 0.15,
        ceiling: float = 0.85,
    ) -> None:
        self._percentile = percentile
        self._window: deque[float] = deque(maxlen=window_size)
        self._floor = floor
        self._ceiling = ceiling
        self._current: float = 0.3  # Bootstrap value

    def record(self, salience: float) -> None:
        """Record an error's habituated salience for threshold computation."""
        self._window.append(salience)

    def compute(self) -> float:
        """Compute the current threshold from the error distribution."""
        if len(self._window) < 5:
            return self._current

        sorted_vals = sorted(self._window)
        idx = int(len(sorted_vals) * self._percentile / 100.0)
        idx = min(idx, len(sorted_vals) - 1)
        raw_threshold = sorted_vals[idx]

        self._current = max(self._floor, min(self._ceiling, raw_threshold))
        return self._current

    @property
    def current(self) -> float:
        return self._current


# ---------------------------------------------------------------------------
# Fovea-Atune Bridge
# ---------------------------------------------------------------------------


class FoveaAtuneBridge:
    """
    The bridge between Fovea's prediction error computation and Atune's
    Global Workspace.

    This is the Phase B integration point. It:
    1. Processes each percept through the full Fovea pipeline
    2. Decomposes the error into per-head scores for compatibility
    3. Computes a dynamic ignition threshold
    4. Feeds qualifying errors into the workspace as WorkspaceCandidates

    It does NOT modify any Atune code. It produces WorkspaceCandidates
    that Atune's existing workspace.enqueue_scored_percept() accepts.
    """

    def __init__(
        self,
        world_model: LogosWorldModel,
        *,
        threshold_percentile: float = 75.0,
        threshold_window: int = 100,
    ) -> None:
        self._prediction_engine = FoveaPredictionEngine(world_model)
        self._precision_computer = PrecisionWeightComputer(world_model)
        self._habituation_engine = HabituationEngine()
        self._dynamic_threshold = DynamicIgnitionThreshold(
            percentile=threshold_percentile,
            window_size=threshold_window,
        )
        self._workspace: GlobalWorkspace | None = None
        self._logger = logger.bind(component="fovea_atune_bridge")

        # Phase C: optional hook to apply learned weights before salience computation
        self._weight_applicator: Callable[[FoveaPredictionError], None] | None = None

        # Transient: last habituation_complete info (consumed by service)
        self._last_habituation_complete: HabituationCompleteInfo | None = None
        # Transient: dishabituation info from last process_percept (consumed by service)
        self._last_dishabituation: dict[str, float] | None = None  # {expected, actual}

        # Metrics
        self._errors_processed: int = 0
        self._workspace_ignitions: int = 0
        self._salience_accumulator: float = 0.0
        self._precision_accumulator: float = 0.0

    def set_workspace(self, workspace: GlobalWorkspace) -> None:
        """Wire the Atune Global Workspace for feeding candidates."""
        self._workspace = workspace

    def set_weight_applicator(
        self, applicator: Callable[[FoveaPredictionError], None]
    ) -> None:
        """Wire a Phase C weight applicator (called before salience computation)."""
        self._weight_applicator = applicator

    @property
    def prediction_engine(self) -> FoveaPredictionEngine:
        return self._prediction_engine

    @property
    def habituation_engine(self) -> HabituationEngine:
        return self._habituation_engine

    @property
    def dynamic_threshold(self) -> DynamicIgnitionThreshold:
        return self._dynamic_threshold

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    async def process_percept(
        self,
        percept: Percept,
    ) -> FoveaPredictionError | None:
        """
        Full Fovea processing pipeline for a single percept.

        1. Extract context
        2. Generate prediction from world model
        3. Compute structured prediction error
        4. Apply precision weighting
        5. Compute precision_weighted_salience
        6. Apply habituation
        7. Record for dynamic threshold
        8. Route: if above threshold, feed into workspace
        9. Return the error for Synapse broadcasting
        """
        # Step 1: Extract context
        context = self._extract_context(percept)

        # Step 2: Generate prediction
        prediction = await self._prediction_engine.generate_prediction(context)

        # Step 3: Compute error
        error = await self._prediction_engine.compute_error(prediction, percept)

        # Step 4: Apply precision weighting (per-component)
        await self._precision_computer.compute_precisions(error, context)

        # Step 4b: Apply learned weights (Phase C) if available
        if self._weight_applicator is not None:
            self._weight_applicator(error)

        # Step 5: Compute precision-weighted salience
        error.compute_precision_weighted_salience()

        # Step 6: Apply habituation
        habituated_salience, habituation_complete, dishabituation_info = (
            self._habituation_engine.apply_habituation(error)
        )
        self._last_habituation_complete = habituation_complete
        self._last_dishabituation = dishabituation_info

        # Step 7: Record for dynamic threshold
        self._dynamic_threshold.record(habituated_salience)
        current_threshold = self._dynamic_threshold.compute()

        # Step 8: Compute routing
        error.compute_routing(current_threshold)

        # Step 9: Feed into workspace if above threshold
        if ErrorRoute.WORKSPACE in error.routes and self._workspace is not None:
            candidate = self._build_workspace_candidate(percept, error)
            self._workspace.enqueue_scored_percept(candidate)
            self._workspace_ignitions += 1

        # Update metrics
        self._errors_processed += 1
        self._salience_accumulator += habituated_salience
        mean_component_precision = (
            sum(error.component_precisions.values()) / len(error.component_precisions)
            if error.component_precisions
            else 0.0
        )
        self._precision_accumulator += mean_component_precision

        self._logger.debug(
            "percept_processed",
            percept_id=percept.id,
            pws=round(error.precision_weighted_salience, 4),
            habituated=round(habituated_salience, 4),
            threshold=round(current_threshold, 4),
            routes=error.routes,
            dominant=error.get_dominant_error_type().value,
        )

        return error

    # ------------------------------------------------------------------
    # Workspace candidate construction
    # ------------------------------------------------------------------

    def _build_workspace_candidate(
        self,
        percept: Percept,
        error: FoveaPredictionError,
    ) -> WorkspaceCandidate:
        """
        Build an Atune-compatible WorkspaceCandidate from a Fovea error.

        The SalienceVector.scores dict maps head names to their re-grounded
        error-based scores, maintaining compatibility with downstream systems
        that inspect per-head scores.
        """
        from systems.atune.types import (
            PredictionError as AtunePredictionError,
        )
        from systems.atune.types import (
            PredictionErrorDirection,
            SalienceVector,
            WorkspaceCandidate,
        )

        # Decompose into per-head scores for Atune compatibility
        head_scores = self._decompose_to_heads(error)

        # Map Fovea's dominant error type to Atune's prediction error direction
        dominant = error.get_dominant_error_type()
        if dominant == ErrorType.CONTENT:
            direction = PredictionErrorDirection.NOVEL
        elif dominant == ErrorType.CATEGORY or error.habituated_salience > 0.6:
            direction = PredictionErrorDirection.CONTRADICTS_BELIEF
        elif error.habituated_salience > 0.3:
            direction = PredictionErrorDirection.NOVEL
        elif error.habituated_salience > 0.1:
            direction = PredictionErrorDirection.CONFIRMS_UNEXPECTED
        else:
            direction = PredictionErrorDirection.EXPECTED

        atune_pe = AtunePredictionError(
            magnitude=min(error.habituated_salience, 1.0),
            direction=direction,
            domain=dominant.value,
        )

        salience = SalienceVector(
            scores=head_scores,
            composite=min(error.habituated_salience, 1.0),
            prediction_error=atune_pe,
        )

        return WorkspaceCandidate(
            content=percept,
            salience=salience,
            source=f"fovea:{percept.source.system}",
            prediction_error=atune_pe,
        )

    @staticmethod
    def _decompose_to_heads(error: FoveaPredictionError) -> dict[str, float]:
        """
        Map the six-dimensional Fovea error back to per-head scores
        for Atune compatibility.
        """
        error_vec = error.get_error_vector()
        head_scores: dict[str, float] = {}

        for head_name, dimension_weights in _HEAD_ERROR_MAPPING.items():
            score = 0.0
            for dim, weight in dimension_weights.items():
                score += error_vec.get(dim, 0.0) * weight
            head_scores[head_name] = min(score, 1.0)

        return head_scores

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_context(percept: Percept) -> PerceptContext:
        """Extract a PerceptContext from a Percept for world model queries."""
        return PerceptContext(
            percept_id=percept.id,
            source_system=percept.source.system,
            channel=percept.source.channel,
            modality=percept.source.modality,
            context_type=f"{percept.source.system}:{percept.source.channel}",
            metadata=percept.metadata,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> FoveaMetrics:
        """Return current Fovea metrics snapshot."""
        mean_salience = (
            self._salience_accumulator / self._errors_processed
            if self._errors_processed > 0
            else 0.0
        )
        mean_precision = (
            self._precision_accumulator / self._errors_processed
            if self._errors_processed > 0
            else 0.0
        )
        return FoveaMetrics(
            errors_processed=self._errors_processed,
            workspace_ignitions=self._workspace_ignitions,
            habituated_count=self._habituation_engine.habituated_count,
            dishabituated_count=self._habituation_engine.dishabituated_count,
            mean_salience=mean_salience,
            mean_precision=mean_precision,
            active_predictions=self._prediction_engine.active_prediction_count,
            habituation_entries=self._habituation_engine.entry_count,
        )
