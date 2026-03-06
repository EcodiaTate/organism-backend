"""
EcodiaOS — Evo Parameter Tuner

Manages the set of tunable system parameters and applies evidence-backed adjustments.

What it does:
  - Maintains current values for all TUNABLE_PARAMETERS in memory
  - Proposes adjustments from supported parameter hypotheses
  - Enforces velocity limits (no lurching personality changes)
  - Persists applied changes to the Memory graph for durability

What it cannot do (EVO_CONSTRAINTS):
  - Touch Equor's evaluation logic
  - Touch constitutional drives
  - Touch invariants
  - Modify its own hypothesis evaluation criteria

Velocity limits (spec Section IX):
  - max_single_parameter_delta = 0.03 (one step)
  - max_total_parameter_delta_per_cycle = 0.15
  - Changes are ALWAYS small — personality doesn't flip

Performance: parameter adjustment application ≤50ms (spec Section X).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.evo.types import (
    PARAMETER_DEFAULTS,
    TUNABLE_PARAMETERS,
    VELOCITY_LIMITS,
    Hypothesis,
    HypothesisCategory,
    HypothesisStatus,
    MutationType,
    ParameterAdjustment,
)

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

logger = structlog.get_logger()


class ParameterTuner:
    """
    Maintains the current values of all tunable parameters and applies
    evidence-backed adjustments with velocity limiting.

    Parameters are initialised from PARAMETER_DEFAULTS at startup.
    Downstream systems (Atune, Nova, Voxis) call get_current_parameter()
    to retrieve the latest value each cycle.

    Applied adjustments are persisted to Memory for durability across restarts.
    """

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="evo.parameter_tuner")

        # Current parameter values (initialised from defaults)
        self._values: dict[str, float] = dict(PARAMETER_DEFAULTS)

        # Adjustment history for this consolidation cycle
        self._cycle_adjustments: list[ParameterAdjustment] = []
        self._total_adjustments: int = 0

    # ─── Query ────────────────────────────────────────────────────────────────

    def get_current_parameter(self, name: str) -> float | None:
        """
        Return the current value of a parameter, or None if unknown.
        Systems should call this each cycle rather than caching.
        """
        return self._values.get(name)

    def get_all_parameters(self) -> dict[str, float]:
        """Return a snapshot of all current parameter values."""
        return dict(self._values)

    # ─── Proposal ─────────────────────────────────────────────────────────────

    def propose_adjustment(
        self,
        hypothesis: Hypothesis,
    ) -> ParameterAdjustment | None:
        """
        Derive a parameter adjustment from a supported parameter hypothesis.

        Returns None if:
          - hypothesis is not a parameter hypothesis
          - hypothesis is not SUPPORTED
          - target parameter is not tunable
          - adjustment would be below meaningful threshold
          - velocity limit would be exceeded
        """
        if hypothesis.category != HypothesisCategory.PARAMETER:
            return None
        if hypothesis.status != HypothesisStatus.SUPPORTED:
            return None

        mutation = hypothesis.proposed_mutation
        if mutation is None or mutation.type != MutationType.PARAMETER_ADJUSTMENT:
            return None

        param_name = mutation.target
        spec = TUNABLE_PARAMETERS.get(param_name)
        if spec is None:
            self._logger.warning(
                "unknown_tunable_parameter",
                parameter=param_name,
                hypothesis_id=hypothesis.id,
            )
            return None

        current = self._values.get(param_name, spec.min_val)
        proposed_delta = mutation.value

        # Clamp to maximum single-step size
        max_step = min(spec.step, VELOCITY_LIMITS["max_single_parameter_delta"])
        clamped_delta = max(-max_step, min(max_step, proposed_delta))

        # Apply and clamp to valid range
        new_value = max(spec.min_val, min(spec.max_val, current + clamped_delta))
        actual_delta = new_value - current

        # Skip if no meaningful change
        if abs(actual_delta) < 0.0001:
            return None

        return ParameterAdjustment(
            parameter=param_name,
            old_value=current,
            new_value=new_value,
            delta=actual_delta,
            hypothesis_id=hypothesis.id,
            evidence_score=hypothesis.evidence_score,
            supporting_count=len(hypothesis.supporting_episodes),
        )

    # ─── Application ──────────────────────────────────────────────────────────

    def check_velocity_limit(
        self,
        adjustments: list[ParameterAdjustment],
    ) -> tuple[bool, str]:
        """
        Check whether this batch of adjustments respects the velocity limit.
        Returns (allowed, reason). reason is empty if allowed.
        """
        total_delta = sum(abs(a.delta) for a in adjustments)
        limit = VELOCITY_LIMITS["max_total_parameter_delta_per_cycle"]
        if total_delta > limit:
            return False, (
                f"Total parameter delta {total_delta:.3f} exceeds "
                f"cycle limit {limit:.3f}"
            )
        return True, ""

    async def apply_adjustment(
        self,
        adjustment: ParameterAdjustment,
    ) -> None:
        """
        Apply a single parameter adjustment.
        Updates the in-memory value and persists to Memory graph.
        Budget: ≤50ms.
        """
        self._values[adjustment.parameter] = adjustment.new_value
        self._cycle_adjustments.append(adjustment)
        self._total_adjustments += 1

        self._logger.info(
            "parameter_adjusted",
            parameter=adjustment.parameter,
            old_value=round(adjustment.old_value, 4),
            new_value=round(adjustment.new_value, 4),
            delta=round(adjustment.delta, 4),
            hypothesis_id=adjustment.hypothesis_id,
            evidence_score=round(adjustment.evidence_score, 2),
        )

        if self._memory is not None:
            await self._persist_adjustment(adjustment)

    def begin_cycle(self) -> None:
        """Called at the start of each consolidation cycle. Resets cycle tracking."""
        self._cycle_adjustments.clear()

    def cycle_delta(self) -> float:
        """Total absolute parameter delta applied in the current cycle."""
        return sum(abs(a.delta) for a in self._cycle_adjustments)

    # ─── Loading ──────────────────────────────────────────────────────────────

    async def load_from_memory(self) -> int:
        """
        Load previously applied parameter values from the Memory graph.
        Call during initialize() to restore state across restarts.
        Returns the count of parameters restored.
        """
        if self._memory is None:
            return 0

        try:
            results = await self._memory._neo4j.execute_read(
                """
                MATCH (p:EvoParameter)
                RETURN p.name AS name, p.current_value AS value
                """
            )
            count = 0
            for row in results:
                name = row.get("name")
                value = row.get("value")
                if name and isinstance(value, (int, float)) and name in TUNABLE_PARAMETERS:
                    spec = TUNABLE_PARAMETERS[name]
                    # Validate against range
                    clamped = max(spec.min_val, min(spec.max_val, float(value)))
                    self._values[name] = clamped
                    count += 1
            self._logger.info("parameters_loaded_from_memory", count=count)
            return count
        except Exception as exc:
            self._logger.warning("parameter_load_failed", error=str(exc))
            return 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_adjustments": self._total_adjustments,
            "cycle_adjustments": len(self._cycle_adjustments),
            "cycle_delta": round(self.cycle_delta(), 4),
            "parameter_count": len(self._values),
        }

    # ─── Private ──────────────────────────────────────────────────────────────

    async def _persist_adjustment(self, adjustment: ParameterAdjustment) -> None:
        """Persist the new parameter value to the Memory graph."""
        try:
            await self._memory._neo4j.execute_write(  # type: ignore[union-attr]
                """
                MERGE (p:EvoParameter {name: $name})
                SET p.current_value = $value,
                    p.last_adjusted = datetime(),
                    p.hypothesis_id = $hypothesis_id,
                    p.evidence_score = $evidence_score
                """,
                {
                    "name": adjustment.parameter,
                    "value": adjustment.new_value,
                    "hypothesis_id": adjustment.hypothesis_id,
                    "evidence_score": adjustment.evidence_score,
                },
            )
        except Exception as exc:
            self._logger.warning(
                "parameter_persist_failed",
                parameter=adjustment.parameter,
                error=str(exc),
            )
