"""
Fovea - World Model Protocol

Defines the interface Fovea needs from Logos (the world model).
Logos is being built simultaneously, so Fovea programs against this protocol.
Dependency injection: the real Logos drops in later.

Also contains the LogosWorldModelAdapter that bridges the real Logos WorldModel
to the Fovea protocol - replacing StubWorldModel with live predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from systems.logos.world_model import WorldModel


@runtime_checkable
class LogosWorldModel(Protocol):
    """
    The interface Fovea requires from the Logos world model.

    Each method corresponds to a specific prediction dimension that Fovea
    needs to compute structured prediction errors. Logos implementations
    must provide all of these, even if some return stubs initially.
    """

    async def predict_content(self, source_system: str, context: dict[str, Any]) -> list[float]:
        """
        Return the expected content embedding for the next percept
        from *source_system* given *context*.

        Returns an empty list if no prediction is available.
        """
        ...

    async def predict_timing(self, source_system: str, context: dict[str, Any]) -> float:
        """
        Return the expected seconds until the next percept from *source_system*.

        Returns 0.0 if no timing prediction is available.
        """
        ...

    async def predict_magnitude(self, source_system: str, context: dict[str, Any]) -> float:
        """
        Return the expected intensity/scale [0, 1] of the next percept.
        """
        ...

    async def predict_source(self, source_system: str, context: dict[str, Any]) -> str:
        """
        Return the expected origin identifier for the next percept.
        """
        ...

    async def predict_category(self, source_system: str, context: dict[str, Any]) -> str:
        """
        Return the expected type/category for the next percept.
        """
        ...

    async def predict_causal_context(
        self, source_system: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Return the expected causal context (cause-effect relationships)
        for the next percept.
        """
        ...

    async def get_prediction_confidence(
        self, source_system: str, context: dict[str, Any]
    ) -> float:
        """
        Return [0, 1] confidence in the predictions for this context.

        Low confidence -> low precision weight -> errors less meaningful.
        """
        ...

    async def get_context_reliability(self, context_type: str) -> float:
        """
        Return [0, 1] reliability of this context type for prediction.

        Used for precision weighting: how much should we trust errors
        from this context?
        """
        ...

    async def get_historical_accuracy(
        self, context_type: str, lookback_window: int = 100
    ) -> float:
        """
        Return [0, 1] historical prediction accuracy for this context type
        over the last *lookback_window* predictions.
        """
        ...

    async def get_context_stability_age(self, context_type: str) -> int:
        """
        Return how many cycles this context type has been stable.

        Used for precision stability bonus calculation.
        """
        ...

    async def get_dimension_accuracy(
        self, context_type: str, dimension: str, lookback_window: int = 100
    ) -> float:
        """
        Return [0, 1] historical prediction accuracy for a specific error dimension
        and context type over the last *lookback_window* predictions.

        Enables per-dimension precision weighting: content errors may be 70% accurate
        while causal errors are only 30% accurate in the same context.

        Returns 0.5 (neutral) if no per-dimension history is available for this context.
        """
        ...

    async def get_compression_score(self) -> float:
        """
        Return the current world model compression score [0, 1].

        Higher = more compressed = better model. Used to track whether
        prediction errors are leading to genuine world model improvement.
        """
        ...


# ---------------------------------------------------------------------------
# Stub implementation for development / testing
# ---------------------------------------------------------------------------


class StubWorldModel:
    """
    A stub world model that returns moderate default predictions.

    Use this until the real Logos is wired in. All predictions return
    neutral/moderate values, ensuring the Fovea infrastructure runs
    end-to-end even without a real world model.
    """

    async def predict_content(
        self, source_system: str, context: dict[str, Any]
    ) -> list[float]:
        return []

    async def predict_timing(
        self, source_system: str, context: dict[str, Any]
    ) -> float:
        return 0.0

    async def predict_magnitude(
        self, source_system: str, context: dict[str, Any]
    ) -> float:
        return 0.5

    async def predict_source(
        self, source_system: str, context: dict[str, Any]
    ) -> str:
        return source_system

    async def predict_category(
        self, source_system: str, context: dict[str, Any]
    ) -> str:
        return ""

    async def predict_causal_context(
        self, source_system: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        return {}

    async def get_prediction_confidence(
        self, source_system: str, context: dict[str, Any]
    ) -> float:
        return 0.5

    async def get_context_reliability(self, context_type: str) -> float:
        return 0.5

    async def get_historical_accuracy(
        self, context_type: str, lookback_window: int = 100
    ) -> float:
        return 0.5

    async def get_dimension_accuracy(
        self, context_type: str, dimension: str, lookback_window: int = 100
    ) -> float:
        return 0.5

    async def get_context_stability_age(self, context_type: str) -> int:
        return 0

    async def get_compression_score(self) -> float:
        return 0.5


# ---------------------------------------------------------------------------
# Live adapter: bridges real Logos WorldModel → Fovea's LogosWorldModel protocol
# ---------------------------------------------------------------------------


class LogosWorldModelAdapter:
    """
    Adapts the real Logos WorldModel to satisfy Fovea's LogosWorldModel protocol.

    Logos WorldModel exposes ``predict(context) → Prediction`` (a single call
    returning expected_content dict + confidence + generating_schemas).

    Fovea needs per-dimension predictions: predict_content(), predict_timing(),
    predict_magnitude(), etc.  This adapter calls ``world_model.predict()`` once
    per percept context and fans the result out to the 6 dimensions Fovea expects.

    This is the LIVE replacement for StubWorldModel - wired by main.py after
    Logos is initialised.
    """

    def __init__(self, world_model: WorldModel) -> None:
        self._wm = world_model

    async def predict_content(
        self, source_system: str, context: dict[str, Any]
    ) -> list[float]:
        """Delegate to Logos world model prediction and extract content embedding."""
        pred = await self._wm.predict({"source_system": source_system, **context})
        # If the world model has generated an expected_content dict, hash it into
        # a rudimentary embedding vector.  Real embeddings come from future Logos
        # upgrades; for now this ensures non-empty content predictions.
        content = pred.expected_content
        if not content:
            return []
        # Produce a lightweight numeric fingerprint from expected_content keys/values.
        values: list[float] = []
        for v in content.values():
            if isinstance(v, (int, float)):
                values.append(float(v))
            elif isinstance(v, str):
                values.append(float(hash(v) % 10000) / 10000.0)
            elif isinstance(v, list):
                values.append(float(len(v)) / 10.0)
        return values or [pred.confidence]

    async def predict_timing(
        self, source_system: str, context: dict[str, Any]
    ) -> float:
        """Estimate expected timing from prior sample counts (higher count = shorter interval)."""
        pred = await self._wm.predict({"source_system": source_system, **context})
        sample_count = pred.expected_content.get("prior_sample_count", 0)
        if isinstance(sample_count, (int, float)) and sample_count > 0:
            # More observations → shorter expected interval
            return max(0.1, 10.0 / float(sample_count))
        return 0.0

    async def predict_magnitude(
        self, source_system: str, context: dict[str, Any]
    ) -> float:
        """Use prediction confidence as expected magnitude."""
        pred = await self._wm.predict({"source_system": source_system, **context})
        return pred.confidence

    async def predict_source(
        self, source_system: str, context: dict[str, Any]
    ) -> str:
        """The expected source is the queried source system."""
        return source_system

    async def predict_category(
        self, source_system: str, context: dict[str, Any]
    ) -> str:
        """Extract expected category from schema pattern if available."""
        pred = await self._wm.predict({"source_system": source_system, **context})
        return str(pred.expected_content.get("domain", ""))

    async def predict_causal_context(
        self, source_system: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract causal expectations from the world model prediction."""
        pred = await self._wm.predict({"source_system": source_system, **context})
        effects = pred.expected_content.get("expected_effects")
        if effects and isinstance(effects, list):
            return {"expected_effects": effects}
        return {}

    async def get_prediction_confidence(
        self, source_system: str, context: dict[str, Any]
    ) -> float:
        """Return the world model's confidence in predictions for this context."""
        pred = await self._wm.predict({"source_system": source_system, **context})
        return pred.confidence

    async def get_context_reliability(self, context_type: str) -> float:
        """Derive reliability from historical accuracy (they're correlated)."""
        return self._wm.get_historical_accuracy(context_type)

    async def get_historical_accuracy(
        self, context_type: str, lookback_window: int = 100
    ) -> float:
        """Delegate to the real Logos world model accuracy tracker."""
        return self._wm.get_historical_accuracy(context_type)

    async def get_dimension_accuracy(
        self, context_type: str, dimension: str, lookback_window: int = 100
    ) -> float:
        """
        Return per-dimension accuracy from Logos if it tracks it, else fall back
        to the global accuracy for this context type.
        """
        # Logos may expose per-dimension accuracy; fall back to global if not.
        fn = getattr(self._wm, "get_dimension_accuracy", None)
        if callable(fn):
            return fn(context_type, dimension)
        return self._wm.get_historical_accuracy(context_type)

    async def get_context_stability_age(self, context_type: str) -> int:
        """Return how many seconds this context prior has been stable."""
        age_s = self._wm.get_context_stability_age(context_type)
        return int(age_s)

    async def get_compression_score(self) -> float:
        """Return the world model's intelligence ratio as a compression score [0, 1]."""
        ir = self._wm.measure_intelligence_ratio()
        # Normalise: intelligence ratio can be >> 1; sigmoid-like squash
        return min(ir / (ir + 1.0), 1.0) if ir > 0 else 0.0
