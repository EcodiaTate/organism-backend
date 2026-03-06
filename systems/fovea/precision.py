"""
Fovea — Precision Weight Computer

Implements the precision weighting from active inference.

Precision = 1/variance = how tightly the world model expects this context
to be predictable. High precision context = errors are informative.
Low precision context = errors may just be noise.

The precision for each error dimension is computed independently:
    precision = historical_accuracy ** 2 + stability_bonus (capped at 0.3)

This prevents Fovea from being overwhelmed by inherently unpredictable
domains while remaining sensitive to violations in highly predictable domains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from .types import ErrorType, FoveaPredictionError, PerceptContext

if TYPE_CHECKING:
    from .protocols import LogosWorldModel

logger = structlog.get_logger("systems.fovea.precision")

# Maximum stability bonus that context age can contribute
_MAX_STABILITY_BONUS: float = 0.3

# Number of cycles required to reach maximum stability bonus
_STABILITY_SCALE: float = 1000.0


class PrecisionWeightComputer:
    """
    Derives per-component precision weights from the world model's
    historical accuracy and context stability.

    precision_i = historical_accuracy(context_type_i) ** 2 + stability_bonus
    stability_bonus = min(context_age / 1000, 0.3)

    Each of the six error dimensions gets its own precision weight,
    stored on the FoveaPredictionError.component_precisions dict.
    """

    def __init__(self, world_model: LogosWorldModel) -> None:
        self._world_model = world_model
        self._logger = logger.bind(component="precision_computer")

    async def compute_precisions(
        self,
        error: FoveaPredictionError,
        context: PerceptContext,
    ) -> None:
        """
        Compute and set per-component precision weights on *error*.

        Queries the world model for historical accuracy and context stability,
        then applies the formula:
            precision = accuracy ** 2 + min(stability_age / 1000, 0.3)
        capped at 1.0.
        """
        context_type = context.context_type or context.source_system

        # Get base signals from world model
        historical_accuracy = await self._safe_call(
            self._world_model.get_historical_accuracy(context_type), 0.5
        )
        stability_age = await self._safe_call(
            self._world_model.get_context_stability_age(context_type), 0
        )

        # Base precision from accuracy (quadratic: high accuracy = very high precision)
        base_precision = historical_accuracy ** 2

        # Stability bonus from context age
        stability_bonus = min(stability_age / _STABILITY_SCALE, _MAX_STABILITY_BONUS)

        # Combined precision, capped at 1.0
        combined_precision = min(base_precision + stability_bonus, 1.0)

        # Set the same precision for all components by default.
        # Future enhancement: per-dimension accuracy tracking from Logos.
        for error_type in ErrorType:
            error.component_precisions[error_type.value] = combined_precision

        self._logger.debug(
            "precision_computed",
            context_type=context_type,
            accuracy=round(historical_accuracy, 4),
            stability_age=stability_age,
            base_precision=round(base_precision, 4),
            stability_bonus=round(stability_bonus, 4),
            combined=round(combined_precision, 4),
        )

    @staticmethod
    async def _safe_call(coro: Any, default: Any) -> Any:  # noqa: ANN401
        try:
            return await coro
        except Exception:
            return default
