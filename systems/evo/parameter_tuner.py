"""
EcodiaOS - Evo Parameter Tuner

Manages the set of tunable system parameters and applies evidence-backed adjustments.

What it does:
  - Maintains current values for all TUNABLE_PARAMETERS in memory
  - Proposes adjustments from supported parameter hypotheses
  - Enforces velocity limits (no lurching personality changes)
  - Persists applied changes to the Memory graph for durability
  - Captures a baseline KPI snapshot before each adjustment
  - Periodically evaluates whether adjustments improved or degraded metrics
  - Auto-reverts degrading adjustments and feeds outcome back to the hypothesis

What it cannot do (EVO_CONSTRAINTS):
  - Touch Equor's evaluation logic
  - Touch constitutional drives
  - Touch invariants
  - Modify its own hypothesis evaluation criteria

Velocity limits (spec Section IX):
  - max_single_parameter_delta = 0.03 (one step)
  - max_total_parameter_delta_per_cycle = 0.15
  - Changes are ALWAYS small - personality doesn't flip

Performance: parameter adjustment application ≤50ms (spec Section X).

Feedback loop:
  - EVAL_CYCLE_COUNT: evaluate pending adjustments every 500 cycles
  - EVAL_MIN_SECONDS: or at least every 30 minutes (1800s)
  - IMPROVEMENT_THRESHOLD: 1.05 - 5% improvement confirms the adjustment
  - DEGRADATION_THRESHOLD: 0.95 - 5% degradation triggers auto-revert
  - MAX_EVAL_EXTENSIONS: 2 - extend the window up to 2 times before forcing confirm
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    from typing import Any
    from systems.memory.service import MemoryService

logger = structlog.get_logger()

# ─── Feedback-loop constants ───────────────────────────────────────────────────

EVAL_CYCLE_COUNT: int = 500          # Evaluate pending adjustments every N cycles
EVAL_MIN_SECONDS: float = 1800.0     # … or every 30 minutes, whichever first
IMPROVEMENT_THRESHOLD: float = 1.05  # +5% → confirm & archive
DEGRADATION_THRESHOLD: float = 0.95  # −5% → revert
MAX_EVAL_EXTENSIONS: int = 2         # Extend eval window up to 2× before forcing confirm


@dataclass
class ParameterAdjustmentRecord:
    """
    Tracks a single applied parameter adjustment through its evaluation lifecycle.

    Created immediately before `apply_adjustment()` modifies `self._values`.
    Kept in `_pending_adjustments` until confirmed, reverted, or max-extended.
    """
    param_path: str                      # e.g. "atune.head.novelty.weight"
    old_value: float
    new_value: float
    cycle_applied: int
    timestamp_applied: float             # time.monotonic() at application
    hypothesis_id: str
    baseline_metrics: dict[str, float]   # KPI snapshot at time of adjustment
    extensions_used: int = 0             # How many eval-window extensions consumed
    confirmed: bool = False
    reverted: bool = False


class ParameterTuner:
    """
    Maintains the current values of all tunable parameters and applies
    evidence-backed adjustments with velocity limiting.

    Parameters are initialised from PARAMETER_DEFAULTS at startup.
    Downstream systems (Atune, Nova, Voxis) call get_current_parameter()
    to retrieve the latest value each cycle.

    Applied adjustments are persisted to Memory for durability across restarts.

    After application, each adjustment is tracked in `_pending_adjustments`.
    Every EVAL_CYCLE_COUNT cycles (or EVAL_MIN_SECONDS), the tuner compares
    current KPIs against the captured baseline.  Degrading adjustments are
    auto-reverted via `EVO_PARAMETER_REVERTED`; improving ones are archived
    with positive evidence fed back to the originating hypothesis.
    """

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="evo.parameter_tuner")

        # Current parameter values (initialised from defaults)
        self._values: dict[str, float] = dict(PARAMETER_DEFAULTS)

        # Adjustment history for this consolidation cycle
        self._cycle_adjustments: list[ParameterAdjustment] = []
        self._total_adjustments: int = 0

        # Feedback-loop state
        self._pending_adjustments: list[ParameterAdjustmentRecord] = []
        self._cycles_since_eval: int = 0
        self._last_eval_time: float = time.monotonic()

        # Injected references (wired post-init)
        self._event_bus: Any = None
        self._hypothesis_engine: Any = None  # for evidence feedback

    def wire_event_bus(self, event_bus: Any) -> None:
        """Wire the Synapse event bus so parameter changes are pushed, not polled.

        See Spec §IX: downstream systems (Atune, Nova, Voxis) call
        get_current_parameter() each cycle.  With event push they can react
        immediately - no polling lag.
        """
        self._event_bus = event_bus

    def wire_hypothesis_engine(self, hypothesis_engine: Any) -> None:
        """Inject HypothesisEngine so revert/confirm can feed evidence back.

        Called from EvoService.initialize() alongside other sub-system wiring.
        """
        self._hypothesis_engine = hypothesis_engine

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
        current_metrics: dict[str, float] | None = None,
        cycle: int = 0,
    ) -> None:
        """Apply a single parameter adjustment.

        Updates the in-memory value, persists to Memory graph, and emits
        EVO_PARAMETER_ADJUSTED on Synapse so downstream systems (Atune,
        Nova, Voxis) can react immediately rather than waiting for the next
        polling cycle.  See Spec §IX.

        Also captures a ParameterAdjustmentRecord with a baseline KPI snapshot
        so the feedback loop can evaluate the adjustment later.

        Args:
            adjustment: The adjustment to apply.
            current_metrics: KPI snapshot from Benchmarks at time of call.
                             If None, an empty baseline is stored (feedback loop
                             will still run but improvement_ratio will be neutral).
            cycle: Current consolidation cycle number for record-keeping.

        Budget: ≤50ms.
        """
        # ── Step 1: capture baseline before modifying ──────────────────────────
        record = ParameterAdjustmentRecord(
            param_path=adjustment.parameter,
            old_value=adjustment.old_value,
            new_value=adjustment.new_value,
            cycle_applied=cycle,
            timestamp_applied=time.monotonic(),
            hypothesis_id=adjustment.hypothesis_id,
            baseline_metrics=dict(current_metrics) if current_metrics else {},
        )
        self._pending_adjustments.append(record)

        # ── Step 2: apply ──────────────────────────────────────────────────────
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

        await self._emit_parameter_adjusted(adjustment)

    def begin_cycle(self) -> None:
        """Called at the start of each consolidation cycle. Resets cycle tracking."""
        self._cycle_adjustments.clear()

    def cycle_delta(self) -> float:
        """Total absolute parameter delta applied in the current cycle."""
        return sum(abs(a.delta) for a in self._cycle_adjustments)

    # ─── Feedback loop ────────────────────────────────────────────────────────

    async def tick_evaluation(
        self,
        current_metrics: dict[str, float],
        cycle: int = 0,
    ) -> None:
        """Periodic evaluation of pending adjustments.

        Call this every consolidation cycle (from ConsolidationOrchestrator or
        EvoService). The method is a no-op until EVAL_CYCLE_COUNT cycles have
        elapsed since the last evaluation OR EVAL_MIN_SECONDS wall-clock time.

        For each pending adjustment:
          - Compute improvement_ratio per KPI that appears in both baseline and
            current_metrics, then take the geometric mean.
          - improvement_ratio > IMPROVEMENT_THRESHOLD → confirm (positive evidence)
          - improvement_ratio < DEGRADATION_THRESHOLD → revert (negative evidence)
          - Otherwise → extend window (up to MAX_EVAL_EXTENSIONS), then confirm.

        Args:
            current_metrics: Fresh KPI snapshot (same keys as baseline where available).
            cycle: Current consolidation cycle number.
        """
        self._cycles_since_eval += 1
        now = time.monotonic()

        time_elapsed = now - self._last_eval_time
        if (
            self._cycles_since_eval < EVAL_CYCLE_COUNT
            and time_elapsed < EVAL_MIN_SECONDS
        ):
            return

        self._cycles_since_eval = 0
        self._last_eval_time = now

        if not self._pending_adjustments:
            return

        still_pending: list[ParameterAdjustmentRecord] = []

        for record in self._pending_adjustments:
            if record.confirmed or record.reverted:
                continue

            ratio = _compute_improvement_ratio(record.baseline_metrics, current_metrics)

            if ratio < DEGRADATION_THRESHOLD:
                await self._revert_adjustment(record, ratio)
            elif ratio > IMPROVEMENT_THRESHOLD:
                await self._confirm_adjustment(record, ratio)
            else:
                # Neutral - extend or force confirm
                if record.extensions_used < MAX_EVAL_EXTENSIONS:
                    record.extensions_used += 1
                    self._logger.debug(
                        "parameter_eval_extended",
                        param=record.param_path,
                        ratio=round(ratio, 4),
                        extension=record.extensions_used,
                    )
                    still_pending.append(record)
                else:
                    # Max extensions consumed - confirm conservatively
                    await self._confirm_adjustment(record, ratio)

        self._pending_adjustments = still_pending

    async def _revert_adjustment(
        self,
        record: ParameterAdjustmentRecord,
        improvement_ratio: float,
    ) -> None:
        """Apply old_value back, emit EVO_PARAMETER_REVERTED, send negative evidence."""
        record.reverted = True
        self._values[record.param_path] = record.old_value

        self._logger.warning(
            "parameter_reverted",
            param=record.param_path,
            old_value=round(record.old_value, 4),
            new_value=round(record.new_value, 4),
            reverted_to=round(record.old_value, 4),
            improvement_ratio=round(improvement_ratio, 4),
            hypothesis_id=record.hypothesis_id,
            reason="degradation",
        )

        await self._emit_parameter_reverted(record, improvement_ratio)
        await self._feed_hypothesis_evidence(
            hypothesis_id=record.hypothesis_id,
            positive=False,
            improvement_ratio=improvement_ratio,
            param_path=record.param_path,
        )

        # Persist the revert to Memory
        if self._memory is not None:
            await self._persist_revert(record)

    async def _confirm_adjustment(
        self,
        record: ParameterAdjustmentRecord,
        improvement_ratio: float,
    ) -> None:
        """Archive a confirmed adjustment and send positive evidence."""
        record.confirmed = True

        self._logger.info(
            "parameter_confirmed",
            param=record.param_path,
            new_value=round(record.new_value, 4),
            improvement_ratio=round(improvement_ratio, 4),
            hypothesis_id=record.hypothesis_id,
        )

        await self._feed_hypothesis_evidence(
            hypothesis_id=record.hypothesis_id,
            positive=True,
            improvement_ratio=improvement_ratio,
            param_path=record.param_path,
        )

    async def _feed_hypothesis_evidence(
        self,
        hypothesis_id: str,
        positive: bool,
        improvement_ratio: float,
        param_path: str,
    ) -> None:
        """Push outcome evidence back to HypothesisEngine.

        Best-effort - never blocks the tuner if the engine is unavailable.
        """
        if self._hypothesis_engine is None:
            return
        try:
            if positive:
                await self._hypothesis_engine.record_parameter_outcome(
                    hypothesis_id=hypothesis_id,
                    success=True,
                    improvement_ratio=improvement_ratio,
                    param_path=param_path,
                )
            else:
                await self._hypothesis_engine.record_parameter_outcome(
                    hypothesis_id=hypothesis_id,
                    success=False,
                    improvement_ratio=improvement_ratio,
                    param_path=param_path,
                )
        except Exception:
            self._logger.debug("hypothesis_evidence_feed_failed", exc_info=True)

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
            results = await self._memory.execute_read(
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
            "pending_evaluations": len(self._pending_adjustments),
        }

    # ─── Private ──────────────────────────────────────────────────────────────

    async def _emit_parameter_adjusted(self, adjustment: ParameterAdjustment) -> None:
        """Emit EVO_PARAMETER_ADJUSTED on Synapse (Spec §IX push notification).

        Best-effort - failure never blocks the learning loop.
        """
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EVO_PARAMETER_ADJUSTED,
                source_system="evo",
                data={
                    "parameter": adjustment.parameter,
                    "old_value": adjustment.old_value,
                    "new_value": adjustment.new_value,
                    "delta": round(adjustment.delta, 6),
                    "hypothesis_id": adjustment.hypothesis_id,
                    "evidence_score": round(adjustment.evidence_score, 4),
                    "supporting_count": adjustment.supporting_count,
                },
            ))
        except Exception:
            self._logger.debug("parameter_adjusted_emit_failed", exc_info=True)

    async def _emit_parameter_reverted(
        self,
        record: ParameterAdjustmentRecord,
        improvement_ratio: float,
    ) -> None:
        """Emit EVO_PARAMETER_REVERTED on Synapse.

        Best-effort - failure never blocks the revert logic.
        """
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EVO_PARAMETER_REVERTED,
                source_system="evo",
                data={
                    "param_path": record.param_path,
                    "old_value": record.old_value,
                    "new_value": record.new_value,
                    "reverted_to": record.old_value,
                    "hypothesis_id": record.hypothesis_id,
                    "cycle_applied": record.cycle_applied,
                    "improvement_ratio": round(improvement_ratio, 4),
                    "reason": "degradation",
                },
            ))
        except Exception:
            self._logger.debug("parameter_reverted_emit_failed", exc_info=True)

    async def _persist_adjustment(self, adjustment: ParameterAdjustment) -> None:
        """Persist the new parameter value to the Memory graph."""
        try:
            await self._memory.execute_write(  # type: ignore[union-attr]
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

    async def _persist_revert(self, record: ParameterAdjustmentRecord) -> None:
        """Persist the reverted parameter value to the Memory graph."""
        try:
            await self._memory.execute_write(  # type: ignore[union-attr]
                """
                MERGE (p:EvoParameter {name: $name})
                SET p.current_value = $value,
                    p.last_reverted = datetime(),
                    p.reverted_from_hypothesis = $hypothesis_id
                """,
                {
                    "name": record.param_path,
                    "value": record.old_value,
                    "hypothesis_id": record.hypothesis_id,
                },
            )
        except Exception as exc:
            self._logger.warning(
                "parameter_revert_persist_failed",
                parameter=record.param_path,
                error=str(exc),
            )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _compute_improvement_ratio(
    baseline: dict[str, float],
    current: dict[str, float],
) -> float:
    """Geometric mean of per-KPI improvement ratios.

    Only KPIs present in both dicts and with a non-zero baseline contribute.
    Returns 1.0 (neutral) if there are no comparable KPIs.
    """
    if not baseline or not current:
        return 1.0

    import math

    log_sum = 0.0
    count = 0
    for key, base_val in baseline.items():
        cur_val = current.get(key)
        if cur_val is None or base_val == 0.0:
            continue
        ratio = cur_val / base_val
        # Guard against log(0) on a metric that collapsed to exactly 0
        if ratio <= 0:
            ratio = 1e-6
        log_sum += math.log(ratio)
        count += 1

    if count == 0:
        return 1.0

    return math.exp(log_sum / count)
