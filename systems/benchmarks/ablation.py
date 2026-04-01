"""Ablation study framework - Round 5D (Spec 24 §9).

Runs controlled ablation experiments against the continual learning pipeline
to produce the paper's contribution table.  Each ablation mode disables one
component of the learning stack, trains a shadow adapter, evaluates on the
CLadder test set, then reports the delta vs the full-stack baseline.

Design notes:
- AblationOrchestrator.run_all() runs 5 ablation modes sequentially; each
  mode calls run_tier2() on the continual learning orchestrator with
  _ablation_mode set, then evaluates L2/L3 accuracy from the longitudinal
  snapshot returned by run_evaluation_now() on BenchmarkService.
- The original adapter is ALWAYS restored in a finally block - shadow
  training never persists to the live production adapter.
- Neo4j persistence is non-blocking and non-fatal.
- ABLATION_STARTED / ABLATION_COMPLETE emitted on the Synapse bus.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.benchmarks.longitudinal import LongitudinalSnapshot
    from systems.benchmarks.service import BenchmarkService
    from systems.reasoning_engine.continual_learning import ContinualLearningOrchestrator

logger = structlog.get_logger("systems.benchmarks.ablation")


# ─── Ablation modes ──────────────────────────────────────────────────────────


class AblationMode(StrEnum):
    """Which component of the learning stack to disable."""

    STREAM_2_OFF = "stream_2_off"           # Disable failure-example stream
    STREAM_4_OFF = "stream_4_off"           # Disable causal-chain stream
    REPLAY_OFF = "replay_off"               # Disable surprise-prioritized replay
    DPO_OFF = "dpo_off"                     # Disable constitutional DPO pass
    ANTI_FORGETTING_OFF = "anti_forgetting_off"  # Disable full anti-forgetting stack


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class AblationConfig:
    """Runtime configuration for an ablation study."""

    mode: AblationMode
    month: int
    instance_id: str = "eos-default"
    # How long to wait (seconds) for an adapter training run to finish
    train_timeout_s: float = 3600.0
    # How long to wait (seconds) for run_evaluation_now to return
    eval_timeout_s: float = 600.0


@dataclass
class AblationResult:
    """Result of a single ablation run."""

    mode: AblationMode
    month: int
    instance_id: str

    # Performance vs full-stack baseline (positive = improvement, negative = regression)
    l2_delta: float = 0.0   # ΔL2 intervention accuracy (pp)
    l3_delta: float = 0.0   # ΔL3 counterfactual accuracy (pp)

    # Raw scores from baseline and ablated runs
    baseline_l2: float = 0.0
    baseline_l3: float = 0.0
    ablated_l2: float = 0.0
    ablated_l3: float = 0.0

    # Human-readable conclusion for the paper table
    conclusion: str = ""

    # Metadata
    elapsed_s: float = 0.0
    error: str | None = None
    node_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def _derive_conclusion(self) -> str:
        """Auto-derive a one-line paper conclusion from deltas."""
        if self.error:
            return f"Error: {self.error}"
        if abs(self.l2_delta) < 0.5 and abs(self.l3_delta) < 0.5:
            return f"{self.mode}: no significant impact (L2 {self.l2_delta:+.1f}pp, L3 {self.l3_delta:+.1f}pp)"
        direction = "hurts" if self.l2_delta < 0 or self.l3_delta < 0 else "helps"
        return (
            f"{self.mode}: removing this component {direction} performance "
            f"(L2 {self.l2_delta:+.1f}pp, L3 {self.l3_delta:+.1f}pp)"
        )


# ─── Orchestrator ────────────────────────────────────────────────────────────


class AblationOrchestrator:
    """Coordinates full ablation study across all 5 modes.

    Usage::

        orchestrator = AblationOrchestrator(
            cl=continual_learning_orchestrator,
            benchmarks=benchmark_service,
            event_bus=synapse_event_bus,
            memory=memory_service,
        )
        results = await orchestrator.run_all(month=current_month)

    The orchestrator:
    1. Evaluates the full-stack baseline (no ablation).
    2. For each mode, sets _ablation_mode on the CL orchestrator, runs Tier 2,
       evaluates, and restores the original adapter in a finally block.
    3. Computes deltas vs baseline and writes AblationResult to Neo4j.
    4. Emits ABLATION_STARTED / ABLATION_COMPLETE for each mode.
    """

    def __init__(
        self,
        cl: ContinualLearningOrchestrator,
        benchmarks: BenchmarkService,
        event_bus: Any | None = None,
        memory: Any | None = None,
        instance_id: str = "eos-default",
    ) -> None:
        self._cl = cl
        self._benchmarks = benchmarks
        self._event_bus = event_bus
        self._memory = memory
        self._instance_id = instance_id
        self._logger = logger.bind(system="benchmarks.ablation")

    def set_event_bus(self, bus: Any) -> None:
        self._event_bus = bus

    def set_memory(self, memory: Any) -> None:
        self._memory = memory

    # ── Public entry points ───────────────────────────────────────────────────

    async def run_all(self, month: int) -> list[AblationResult]:
        """Run all 5 ablation modes and return results."""
        self._logger.info("ablation_run_all_started", month=month)

        # Step 1: full-stack baseline
        baseline_snap = await self._get_baseline_snapshot(month)

        results: list[AblationResult] = []
        for mode in AblationMode:
            config = AblationConfig(
                mode=mode,
                month=month,
                instance_id=self._instance_id,
            )
            result = await self._run_one(config, baseline_snap)
            results.append(result)

        self._logger.info(
            "ablation_run_all_complete",
            month=month,
            modes_run=len(results),
            errors=[r.mode for r in results if r.error],
        )
        return results

    async def run_one(self, mode: AblationMode, month: int) -> AblationResult:
        """Run a single ablation mode."""
        baseline_snap = await self._get_baseline_snapshot(month)
        config = AblationConfig(mode=mode, month=month, instance_id=self._instance_id)
        return await self._run_one(config, baseline_snap)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _get_baseline_snapshot(self, month: int) -> LongitudinalSnapshot | None:
        """Evaluate current (full-stack) performance as the ablation baseline."""
        try:
            snap = await asyncio.wait_for(
                self._benchmarks.run_evaluation_now(month=month),
                timeout=600.0,
            )
            return snap
        except Exception as exc:
            self._logger.warning("ablation_baseline_eval_failed", error=str(exc))
            return None

    async def _run_one(
        self,
        config: AblationConfig,
        baseline_snap: LongitudinalSnapshot | None,
    ) -> AblationResult:
        """Run a single ablation mode, returning a populated AblationResult."""
        t0 = time.monotonic()
        result = AblationResult(
            mode=config.mode,
            month=config.month,
            instance_id=config.instance_id,
        )
        if baseline_snap is not None:
            result.baseline_l2 = baseline_snap.l2_intervention
            result.baseline_l3 = baseline_snap.l3_counterfactual

        # Emit ABLATION_STARTED
        await self._emit(
            SynapseEventType.ABLATION_STARTED,
            {"mode": str(config.mode), "month": config.month},
        )

        try:
            # Train ablated adapter (restores original in finally inside _train_ablated)
            await self._train_ablated(config)

            # Evaluate ablated performance
            ablated_snap = await asyncio.wait_for(
                self._benchmarks.run_evaluation_now(month=config.month),
                timeout=config.eval_timeout_s,
            )
            result.ablated_l2 = ablated_snap.l2_intervention
            result.ablated_l3 = ablated_snap.l3_counterfactual

        except Exception as exc:
            result.error = str(exc)
            self._logger.warning(
                "ablation_run_failed",
                mode=config.mode,
                error=str(exc),
            )
        finally:
            # Always compute deltas (even partial) and emit ABLATION_COMPLETE
            result.l2_delta = result.ablated_l2 - result.baseline_l2
            result.l3_delta = result.ablated_l3 - result.baseline_l3
            result.conclusion = result._derive_conclusion()
            result.elapsed_s = time.monotonic() - t0
            result.node_id = f"ablation:{config.instance_id}:{config.month}:{config.mode}"

            await self._emit(
                SynapseEventType.ABLATION_COMPLETE,
                {
                    "mode": str(config.mode),
                    "month": config.month,
                    "l2_delta": result.l2_delta,
                    "l3_delta": result.l3_delta,
                    "conclusion": result.conclusion,
                },
            )

            # Persist to Neo4j (non-blocking, non-fatal)
            asyncio.ensure_future(self._persist_result(result))

        return result

    async def _train_ablated(self, config: AblationConfig) -> None:
        """Set ablation mode, run Tier 2, then ALWAYS restore original adapter.

        The _ablation_mode flag is set synchronously so that _execute_tier2()
        sees it before any async yields.  It is cleared in the finally block
        regardless of success or failure.
        """
        original_adapter = getattr(self._cl, "_sure", None)
        original_adapter_path = (
            getattr(original_adapter, "production_adapter_path", None)
            if original_adapter is not None
            else None
        )

        # Set ablation mode synchronously before yielding
        self._cl._ablation_mode = str(config.mode)
        try:
            await asyncio.wait_for(
                self._cl.run_tier2(trigger_reason=f"ablation:{config.mode}"),
                timeout=config.train_timeout_s,
            )
        finally:
            # Restore ablation mode to none (synchronous)
            self._cl._ablation_mode = "none"

            # Restore original production adapter path if we have it
            if original_adapter is not None and original_adapter_path is not None:
                with contextlib.suppress(Exception):
                    original_adapter.production_adapter_path = original_adapter_path

    async def _persist_result(self, result: AblationResult) -> None:
        """Write (:AblationResult) node to Neo4j.  Fire-and-forget, non-fatal."""
        try:
            neo4j = getattr(self._memory, "_neo4j", None) if self._memory else None
            if neo4j is None:
                return
            await neo4j.execute_write(
                """
                MERGE (a:AblationResult {node_id: $node_id})
                SET a.mode = $mode,
                    a.month = $month,
                    a.instance_id = $instance_id,
                    a.l2_delta = $l2_delta,
                    a.l3_delta = $l3_delta,
                    a.baseline_l2 = $baseline_l2,
                    a.baseline_l3 = $baseline_l3,
                    a.ablated_l2 = $ablated_l2,
                    a.ablated_l3 = $ablated_l3,
                    a.conclusion = $conclusion,
                    a.elapsed_s = $elapsed_s,
                    a.error = $error,
                    a.timestamp = datetime()
                RETURN a
                """,
                node_id=result.node_id,
                mode=str(result.mode),
                month=result.month,
                instance_id=result.instance_id,
                l2_delta=result.l2_delta,
                l3_delta=result.l3_delta,
                baseline_l2=result.baseline_l2,
                baseline_l3=result.baseline_l3,
                ablated_l2=result.ablated_l2,
                ablated_l3=result.ablated_l3,
                conclusion=result.conclusion,
                elapsed_s=result.elapsed_s,
                error=result.error or "",
            )
            self._logger.debug("ablation_result_persisted", node_id=result.node_id)
        except Exception as exc:
            self._logger.warning("ablation_neo4j_persist_failed", error=str(exc))

    async def _emit(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        """Emit a Synapse event (fire-and-forget, non-fatal)."""
        if self._event_bus is None:
            return
        try:
            event = SynapseEvent(
                event_type=event_type,
                source_system="benchmarks",
                data=data,
            )
            asyncio.ensure_future(self._event_bus.emit(event))
        except Exception as exc:
            self._logger.warning("ablation_event_emit_failed", error=str(exc))
