"""
Fovea - Precision Weight Computer

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
    Derives per-dimension precision weights from the world model's
    per-dimension historical accuracy and context stability.

    Spec 20 §3.4 - P1: each error dimension gets an independent precision weight:
        precision_i = dim_accuracy(context_type, dim_i) ** 2 + stability_bonus
        stability_bonus = min(context_age / 1000, 0.3)

    Different prediction types have different historical accuracy within the same
    context: content errors may be 70% accurate while causal errors are 30%.
    Uniform precision would over-discount reliable dimensions - per-dimension
    tracking correctly amplifies precise dimensions and dampens noisy ones.
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
        Compute and set per-dimension precision weights on *error*.

        Spec 20 §3.4 (P1 fix 2026-03-07): queries the world model for
        per-dimension accuracy, applies:
            precision_i = accuracy_i ** 2 + stability_bonus
        independently for each of the six error dimensions, capped at 1.0.

        The stability bonus is shared (context-level); per-dimension accuracy
        is what differentiates the six components.
        """
        context_type = context.context_type or context.source_system

        # Stability bonus is context-level (same for all dimensions)
        stability_age = await self._safe_call(
            self._world_model.get_context_stability_age(context_type), 0
        )
        stability_bonus = min(stability_age / _STABILITY_SCALE, _MAX_STABILITY_BONUS)

        # Per-dimension accuracy → independent precision weights
        dim_precisions: dict[str, float] = {}
        for error_type in ErrorType:
            dim_accuracy = await self._safe_call(
                self._world_model.get_dimension_accuracy(context_type, error_type.value),
                0.5,
            )
            precision = min(dim_accuracy ** 2 + stability_bonus, 1.0)
            dim_precisions[error_type.value] = precision
            error.component_precisions[error_type.value] = precision

        self._logger.debug(
            "precision_computed_per_dimension",
            context_type=context_type,
            stability_age=stability_age,
            stability_bonus=round(stability_bonus, 4),
            dim_precisions={k: round(v, 4) for k, v in dim_precisions.items()},
        )

    @staticmethod
    async def _safe_call(coro: Any, default: Any) -> Any:  # noqa: ANN401
        try:
            return await coro
        except Exception:
            return default
