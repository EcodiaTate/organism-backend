"""
EcodiaOS - Oneiros v2: Sleep Cycle Engine

Top-level orchestrator for v2 sleep cycles. Drives the four-stage pipeline:
  DESCENT -> SLOW_WAVE -> REM -> EMERGENCE

With optional Lucid Dreaming between REM and EMERGENCE when Simula has
pending mutation proposals.

Each stage transition broadcasts SLEEP_STAGE_TRANSITION on Synapse.
If sleep is interrupted, the system can restore from the Descent checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from primitives.common import utc_now
from systems.oneiros.descent import DescentStage
from systems.oneiros.emergence import EmergenceStage
from systems.oneiros.lucid_stage import (
    ConstitutionalCheckProtocol,
    LucidDreamingStage,
    SimulaProtocol,
)
from systems.oneiros.rem_stage import (
    EvoHypothesisProtocol,
    FoveaErrorDomainProtocol,
    REMStage,
)
from systems.oneiros.scheduler import SleepScheduler
from systems.oneiros.slow_wave import SlowWaveStage
from systems.oneiros.types import (
    STAGE_DURATION_FRACTION,
    SleepCheckpoint,
    SleepCycleV2Report,
    SleepOutcome,
    SleepSchedulerConfig,
    SleepStageV2,
    SleepTrigger,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.logos.service import LogosService
    from systems.oneiros.types import (
        EmergenceReport,
        LucidDreamingReport,
        REMStageReport,
        SlowWaveReport,
    )
    from systems.synapse.event_bus import EventBus

# KPIs queried from Benchmarks before and after sleep
_TRACKED_KPIS = (
    "coherence_composite",
    "hypothesis_avg_confidence",
    "schema_count",
    "re_success_rate",
    "memory_fragmentation",
)

# Adaptive threshold bounds
_THRESHOLD_MIN = 0.75
_THRESHOLD_MAX = 0.95
# Consecutive outcome history window
_OUTCOME_HISTORY_LEN = 5


# ─── Stage Protocols ─────────────────────────────────────────────
# Each stage exposes a typed execute() signature. The engine holds concrete
# instances but these Protocols allow tests and future alternative implementations
# to satisfy the interface contract without inheritance.


@runtime_checkable
class DescentStageProtocol(Protocol):
    """Contract for the Descent stage."""

    async def execute(
        self,
        trigger: SleepTrigger,
        logos: Any,
        target_duration_s: float,
        *,
        active_hypothesis_count: int,
        unprocessed_error_count: int,
    ) -> SleepCheckpoint:
        ...

    @property
    def input_suspended(self) -> bool:
        ...

    def resume_input_channels(self) -> None:
        ...


@runtime_checkable
class SlowWaveStageProtocol(Protocol):
    """Contract for the Slow Wave stage."""

    async def execute(
        self,
        checkpoint: SleepCheckpoint,
        uncompressed_episodes: list[dict[str, Any]] | None,
        active_hypotheses: list[dict[str, Any]] | None,
        causal_observations: list[dict[str, Any]] | None,
    ) -> SlowWaveReport:
        ...


@runtime_checkable
class REMStageProtocol(Protocol):
    """Contract for the REM stage."""

    async def execute(
        self,
        checkpoint: SleepCheckpoint,
        kairos_priority_seeds: list[dict[str, Any]] | None,
    ) -> REMStageReport:
        ...

    @property
    def pre_attention_entries(self) -> list[Any]:
        ...


@runtime_checkable
class LucidDreamingStageProtocol(Protocol):
    """Contract for the Lucid Dreaming stage."""

    async def execute(
        self,
        checkpoint: SleepCheckpoint,
    ) -> LucidDreamingReport:
        ...


@runtime_checkable
class EmergenceStageProtocol(Protocol):
    """Contract for the Emergence stage."""

    async def execute(
        self,
        checkpoint: SleepCheckpoint,
        logos: Any,
        sleep_start_time: float | None,
        slow_wave_report: SlowWaveReport | None,
        rem_report: REMStageReport | None,
        lucid_report: LucidDreamingReport | None,
        pre_attention_entries: list[Any],
        sleep_cycle_id: str,
    ) -> EmergenceReport:
        ...

    @property
    def average_intelligence_improvement(self) -> float:
        ...

logger = structlog.get_logger("oneiros.engine")

_SOURCE = "oneiros"


class SleepCycleEngine:
    """
    v2 Sleep Cycle Engine -- the "batch compiler" orchestrator.

    Full pipeline:
        DESCENT -> SLOW_WAVE -> REM -> (Lucid Dreaming) -> EMERGENCE

    Lifecycle:
        1. Scheduler decides when to sleep (should_sleep + can_sleep_now)
        2. Engine runs the pipeline stages
        3. Each stage transition broadcasts SLEEP_STAGE_TRANSITION
        4. Engine records sleep completion on the scheduler

    Dependencies are injected via setters (same pattern as OneirosService v1)
    to avoid circular imports and allow gradual system wiring.
    """

    def __init__(
        self,
        config: SleepSchedulerConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._config = config or SleepSchedulerConfig()
        self._event_bus = event_bus

        # Sub-stages (typed via stage Protocols)
        self._scheduler = SleepScheduler(config=self._config)
        self._descent: DescentStageProtocol = DescentStage(event_bus=event_bus)
        self._slow_wave: SlowWaveStageProtocol | None = None  # needs logos
        self._rem: REMStageProtocol | None = None  # needs logos
        self._lucid: LucidDreamingStageProtocol | None = None  # needs logos + simula
        self._emergence: EmergenceStageProtocol = EmergenceStage(event_bus=event_bus)

        # Cross-system refs (set via setters)
        self._logos: LogosService | None = None
        self._fovea: FoveaErrorDomainProtocol | None = None
        self._evo: EvoHypothesisProtocol | None = None
        self._simula: SimulaProtocol | None = None
        self._equor: ConstitutionalCheckProtocol | None = None
        self._neo4j: Any = None
        self._benchmarks: Any = None  # optional BenchmarkService for KPI queries

        # State
        self._is_sleeping = False
        self._current_stage: SleepStageV2 | None = None
        self._current_checkpoint: SleepCheckpoint | None = None
        self._sleep_start_mono: float | None = None
        self._creative_goal: str | None = None

        # Interrupted checkpoint pending restoration on next sleep cycle
        self._pending_restore_checkpoint: SleepCheckpoint | None = None

        # ── Performance measurement state ──────────────────────────
        # Captured just before DESCENT
        self._pre_sleep_baseline: dict[str, float] = {}
        self._pre_sleep_cycle_id: str = ""
        self._pre_sleep_timestamp: datetime | None = None
        # Per-sleep stage completion tracking
        self._stages_completed: list[str] = []
        # Ring buffer of recent SleepOutcome verdicts for adaptive threshold
        self._outcome_history: deque[str] = deque(maxlen=_OUTCOME_HISTORY_LEN)

        self._logger = logger.bind(component="engine")

    # -- Cross-System Wiring ----------------------------------------

    def set_logos(self, logos: LogosService) -> None:
        """Wire Logos for world model access across all stages."""
        self._logos = logos
        self._rebuild_stages()

    def set_fovea(self, fovea: FoveaErrorDomainProtocol) -> None:
        """Wire Fovea for error domain tracking in REM dream generation."""
        self._fovea = fovea
        self._rebuild_stages()

    def set_evo(self, evo: EvoHypothesisProtocol) -> None:
        """Wire Evo for hypothesis extraction in REM dream generation."""
        self._evo = evo
        self._rebuild_stages()

    def set_simula(self, simula: SimulaProtocol) -> None:
        """Wire Simula for lucid dreaming mutation testing."""
        self._simula = simula
        self._rebuild_stages()

    def set_equor(self, equor: ConstitutionalCheckProtocol) -> None:
        """Wire Equor for constitutional checks in lucid dreaming."""
        self._equor = equor
        self._rebuild_stages()

    def set_neo4j(self, neo4j: Any) -> None:
        """Wire Neo4j for LucidDreamingStage MetaCognition and DirectedExploration."""
        self._neo4j = neo4j
        self._rebuild_stages()

    def set_benchmarks(self, benchmarks: Any) -> None:
        """Wire BenchmarkService for pre/post-sleep KPI queries."""
        self._benchmarks = benchmarks

    def set_creative_goal(self, goal: str | None) -> None:
        """Pass the current creative goal into LucidDreamingStage."""
        self._creative_goal = goal
        if self._lucid is not None and isinstance(self._lucid, LucidDreamingStage):
            self._lucid._creative_goal = goal

    def _rebuild_stages(self) -> None:
        """Rebuild stages that depend on cross-system refs."""
        self._slow_wave = SlowWaveStage(
            logos=self._logos,
            event_bus=self._event_bus,
        )
        self._rem = REMStage(
            logos=self._logos,
            fovea=self._fovea,
            evo=self._evo,
            event_bus=self._event_bus,
        )
        self._lucid = LucidDreamingStage(
            logos=self._logos,
            simula=self._simula,
            equor=self._equor,
            event_bus=self._event_bus,
            neo4j=self._neo4j,
            creative_goal=self._creative_goal,
        )

    @property
    def scheduler(self) -> SleepScheduler:
        return self._scheduler

    @property
    def is_sleeping(self) -> bool:
        return self._is_sleeping

    @property
    def current_stage(self) -> SleepStageV2 | None:
        return self._current_stage

    @property
    def current_checkpoint(self) -> SleepCheckpoint | None:
        return self._current_checkpoint

    @property
    def emergence(self) -> EmergenceStageProtocol:
        """Access emergence for intelligence improvement tracking."""
        return self._emergence

    # -- Main Execution ---------------------------------------------

    async def run_sleep_cycle(
        self,
        trigger: SleepTrigger,
        *,
        uncompressed_episodes: list[dict[str, Any]] | None = None,
        active_hypotheses: list[dict[str, Any]] | None = None,
        causal_observations: list[dict[str, Any]] | None = None,
        kairos_rem_seeds: list[dict[str, Any]] | None = None,
    ) -> SleepCycleV2Report:
        """
        Execute a complete v2 sleep cycle:
            DESCENT -> SLOW_WAVE -> REM -> (Lucid Dreaming) -> EMERGENCE

        Args:
            trigger: What triggered this sleep cycle.
            uncompressed_episodes: Episodes for the Memory Ladder.
            active_hypotheses: Hypotheses for the Graveyard.
            causal_observations: Observations for Causal Reconstruction.

        Returns:
            Complete sleep cycle report.
        """
        if self._is_sleeping:
            self._logger.warning("sleep_cycle_already_running")
            return SleepCycleV2Report(
                trigger=trigger,
                interrupted=True,
                interrupt_reason="already_sleeping",
            )

        self._is_sleeping = True
        self._sleep_start_mono = time.monotonic()
        target_duration_s = self._config.target_sleep_duration_s

        # Check whether we have an interrupted checkpoint to restore from.
        restore_checkpoint = self._pending_restore_checkpoint
        self._pending_restore_checkpoint = None

        report = SleepCycleV2Report(trigger=trigger)

        # Capture pre-sleep KPI baseline before any stage runs
        await self._capture_pre_sleep_baseline(report.id)

        if restore_checkpoint is not None:
            self._logger.info(
                "sleep_cycle_resuming_from_checkpoint",
                trigger=trigger.value,
                checkpoint_id=restore_checkpoint.id,
                note="skipping DESCENT, resuming at SLOW_WAVE",
            )
        else:
            self._logger.info(
                "sleep_cycle_starting",
                trigger=trigger.value,
                target_duration_s=target_duration_s,
            )

        try:
            if restore_checkpoint is not None:
                # -- Restored from interrupted cycle: skip DESCENT --
                checkpoint = restore_checkpoint
                self._current_checkpoint = checkpoint
                report.checkpoint = checkpoint
            else:
                # -- Stage 1: DESCENT (10%) --
                # Capture pre-sleep counts from Evo and Fovea for the checkpoint.
                active_hypothesis_count = 0
                if self._evo is not None:
                    with contextlib.suppress(Exception):
                        active_hypothesis_count = (
                            await self._evo.get_active_hypothesis_count()
                        )

                unprocessed_error_count = 0
                if self._fovea is not None and hasattr(
                    self._fovea, "get_unprocessed_error_count"
                ):
                    with contextlib.suppress(Exception):
                        unprocessed_error_count = (
                            await self._fovea.get_unprocessed_error_count()
                        )

                await self._transition_to(SleepStageV2.DESCENT)
                checkpoint = await self._descent.execute(
                    trigger=trigger,
                    logos=self._logos,
                    target_duration_s=target_duration_s,
                    active_hypothesis_count=active_hypothesis_count,
                    unprocessed_error_count=unprocessed_error_count,
                )
                self._current_checkpoint = checkpoint
                report.checkpoint = checkpoint

            # -- Stage 2: SLOW_WAVE (50%) --
            await self._transition_to(SleepStageV2.SLOW_WAVE)
            slow_wave_stage = self._slow_wave or SlowWaveStage(
                event_bus=self._event_bus,
            )
            slow_wave_report = await slow_wave_stage.execute(
                checkpoint=checkpoint,
                uncompressed_episodes=uncompressed_episodes,
                active_hypotheses=active_hypotheses,
                causal_observations=causal_observations,
            )
            report.slow_wave = slow_wave_report

            # -- Stage 3: REM (30%) -- Cross-Domain Synthesis --
            await self._transition_to(SleepStageV2.REM)
            rem_stage = self._rem or REMStage(event_bus=self._event_bus)
            rem_report = await rem_stage.execute(
                checkpoint=checkpoint,
                kairos_priority_seeds=kairos_rem_seeds,
            )
            report.rem = rem_report

            # -- Stage 3.5: Lucid Dreaming (within REM budget) --
            lucid_report = None
            if self._lucid is not None:
                lucid_report = await self._lucid.execute(checkpoint=checkpoint)
                if lucid_report.mutations_tested > 0:
                    report.lucid = lucid_report

            # -- Stage 4: EMERGENCE (10%) --
            # Resume input channels now that compilation is complete.
            if hasattr(self._descent, "resume_input_channels"):
                self._descent.resume_input_channels()  # type: ignore[union-attr]

            await self._transition_to(SleepStageV2.EMERGENCE)
            emergence_report = await self._emergence.execute(
                checkpoint=checkpoint,
                logos=self._logos,
                sleep_start_time=self._sleep_start_mono,
                slow_wave_report=slow_wave_report,
                rem_report=rem_report,
                lucid_report=lucid_report,
                pre_attention_entries=rem_stage.pre_attention_entries,
                sleep_cycle_id=report.id,
            )
            report.emergence = emergence_report
            report.intelligence_improvement = emergence_report.intelligence_improvement

            # Finalize
            elapsed = (time.monotonic() - self._sleep_start_mono) * 1000
            report.completed_at = utc_now()
            report.total_duration_ms = elapsed

            # Record sleep completion on scheduler
            self._scheduler.record_sleep_completed()

            # Fire-and-forget: wait 100 cycles then compare KPIs to baseline
            asyncio.ensure_future(self._compute_and_emit_sleep_outcome(report))

            self._logger.info(
                "sleep_cycle_complete",
                cycle_id=report.id,
                trigger=trigger.value,
                intelligence_improvement=round(
                    emergence_report.intelligence_improvement, 4
                ),
                cross_domain_matches=rem_report.cross_domain.strong_matches,
                analogies_found=rem_report.analogies.analogies_found,
                mutations_tested=(
                    lucid_report.mutations_tested if lucid_report else 0
                ),
                duration_ms=round(elapsed, 1),
            )

        except Exception as exc:
            report.interrupted = True
            report.interrupt_reason = str(exc)
            self._logger.error(
                "sleep_cycle_interrupted",
                error=str(exc),
                stage=self._current_stage.value if self._current_stage else "unknown",
            )
        finally:
            self._is_sleeping = False
            self._current_stage = None
            self._current_checkpoint = None
            self._sleep_start_mono = None

        return report

    # -- Stage Transitions ------------------------------------------

    async def _transition_to(self, stage: SleepStageV2) -> None:
        """Transition to a new sleep stage, broadcasting SLEEP_STAGE_TRANSITION."""
        previous = self._current_stage
        self._current_stage = stage
        self._stages_completed.append(stage.value)

        self._logger.info(
            "stage_transition",
            from_stage=previous.value if previous else "awake",
            to_stage=stage.value,
            fraction=STAGE_DURATION_FRACTION[stage],
        )

        await self._broadcast_stage_transition(previous, stage)

    async def _broadcast_stage_transition(
        self,
        from_stage: SleepStageV2 | None,
        to_stage: SleepStageV2,
    ) -> None:
        """Broadcast SLEEP_STAGE_TRANSITION event on Synapse."""
        if self._event_bus is None:
            return

        event = SynapseEvent(
            event_type=SynapseEventType.SLEEP_STAGE_TRANSITION,
            source_system=_SOURCE,
            data={
                "from_stage": from_stage.value if from_stage else None,
                "to_stage": to_stage.value,
                "duration_fraction": STAGE_DURATION_FRACTION[to_stage],
                "checkpoint_id": (
                    self._current_checkpoint.id
                    if self._current_checkpoint
                    else None
                ),
            },
        )
        await self._event_bus.emit(event)

        # Co-emit SLEEP_STAGE_CHANGED - semantic alias consumed by Thread, Benchmarks, etc.
        with contextlib.suppress(Exception):
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.SLEEP_STAGE_CHANGED,
                source_system=_SOURCE,
                data={
                    "from_stage": from_stage.value if from_stage else None,
                    "to_stage": to_stage.value,
                },
            ))

    # -- Performance Measurement ------------------------------------

    async def _query_kpis(self) -> dict[str, float]:
        """
        Query current KPIs from BenchmarkService (best-effort, non-fatal).

        Returns a dict of kpi_name → float for each tracked KPI.
        Missing KPIs are omitted silently.
        """
        kpis: dict[str, float] = {}
        if self._benchmarks is None:
            return kpis
        with contextlib.suppress(Exception):
            if hasattr(self._benchmarks, "get_current_kpis"):
                raw: dict[str, Any] = await self._benchmarks.get_current_kpis(
                    list(_TRACKED_KPIS)
                )
                for key in _TRACKED_KPIS:
                    val = raw.get(key)
                    if isinstance(val, (int, float)):
                        kpis[key] = float(val)
        return kpis

    async def _capture_pre_sleep_baseline(self, cycle_id: str) -> None:
        """Snapshot KPIs just before sleep begins."""
        self._pre_sleep_baseline = await self._query_kpis()
        self._pre_sleep_cycle_id = cycle_id
        self._pre_sleep_timestamp = utc_now()
        self._stages_completed = []
        self._logger.debug(
            "pre_sleep_baseline_captured",
            cycle_id=cycle_id,
            kpis=self._pre_sleep_baseline,
        )

    async def _compute_and_emit_sleep_outcome(
        self,
        report: SleepCycleV2Report,
    ) -> None:
        """
        Wait 100 cycles for metrics to stabilise, then compare KPIs to baseline
        and emit ONEIROS_SLEEP_OUTCOME.  Adapts pressure threshold accordingly.

        This is fire-and-forget - called via asyncio.ensure_future after Emergence.
        """
        # 100-cycle stabilisation: each theta cycle ≈ 150ms → ~15s
        await asyncio.sleep(15.0)

        post_kpis = await self._query_kpis()
        baseline = self._pre_sleep_baseline

        # Compute per-KPI deltas (skip KPIs missing in either snapshot)
        deltas: dict[str, float] = {}
        for key in _TRACKED_KPIS:
            pre = baseline.get(key)
            post = post_kpis.get(key)
            if pre is None or post is None or pre == 0.0:
                continue
            # For memory_fragmentation lower is better - invert the sign
            raw_delta = (post - pre) / abs(pre)
            deltas[key] = -raw_delta if key == "memory_fragmentation" else raw_delta

        positive = [v for v in deltas.values() if v > 0.0]
        negative = [v for v in deltas.values() if v < 0.0]
        net_improvement = sum(positive) / len(positive) if positive else 0.0
        net_degradation = abs(sum(negative) / len(negative)) if negative else 0.0

        if net_improvement > 0.02 and net_improvement >= net_degradation:
            verdict = "beneficial"
        elif net_degradation > 0.02 and net_degradation > net_improvement:
            verdict = "harmful"
        else:
            verdict = "neutral"

        self._outcome_history.append(verdict)

        # Adapt pressure threshold
        old_threshold = self._config.cognitive_pressure_threshold
        new_threshold = self._adapt_threshold(verdict, net_improvement)
        adjusted = new_threshold != old_threshold
        if adjusted:
            self._config.cognitive_pressure_threshold = new_threshold
            self._scheduler._config.cognitive_pressure_threshold = new_threshold
            self._logger.info(
                "sleep_threshold_adapted",
                old=round(old_threshold, 3),
                new=round(new_threshold, 3),
                verdict=verdict,
            )

        outcome = SleepOutcome(
            sleep_cycle_id=report.id,
            sleep_duration_ms=int(report.total_duration_ms),
            stages_completed=self._stages_completed[:],
            kpi_deltas=deltas,
            net_improvement=round(net_improvement, 4),
            net_degradation=round(net_degradation, 4),
            verdict=verdict,
            pressure_threshold_adjusted=adjusted,
            new_pressure_threshold=new_threshold,
        )

        self._logger.info(
            "sleep_outcome_evaluated",
            cycle_id=report.id,
            verdict=verdict,
            net_improvement=outcome.net_improvement,
            net_degradation=outcome.net_degradation,
            kpi_deltas=deltas,
        )

        if self._event_bus is not None:
            with contextlib.suppress(Exception):
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.ONEIROS_SLEEP_OUTCOME,
                    source_system=_SOURCE,
                    data=outcome.model_dump(),
                ))

            # "harmful" → signal Evo to reconsider sleep parameters
            if verdict == "harmful" and self._event_bus is not None:
                with contextlib.suppress(Exception):
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.LEARNING_PRESSURE,
                        source_system=_SOURCE,
                        data={
                            "source_system": _SOURCE,
                            "reason": "harmful_sleep_outcome",
                            "kpi_deltas": deltas,
                            "net_degradation": outcome.net_degradation,
                            "sleep_cycle_id": report.id,
                        },
                    ))

    def _adapt_threshold(self, verdict: str, net_improvement: float) -> float:
        """
        Return the new cognitive_pressure_threshold based on recent outcome history.

        Rules:
        - 3+ consecutive "beneficial" with net_improvement > 10%: decrease to 0.80 (sleep more)
        - 3+ consecutive "beneficial" (any improvement): maintain
        - 2+ consecutive "harmful": increase to 0.90 (sleep less)
        Bounds: [0.75, 0.95]
        """
        current = self._config.cognitive_pressure_threshold
        history = list(self._outcome_history)  # oldest → newest

        if len(history) < 2:
            return current

        last_2 = history[-2:]
        last_3 = history[-3:] if len(history) >= 3 else []

        if all(v == "harmful" for v in last_2):
            return min(_THRESHOLD_MAX, current + 0.05)

        if last_3 and all(v == "beneficial" for v in last_3):
            if net_improvement > 0.10:
                return max(_THRESHOLD_MIN, current - 0.05)
            # 3+ beneficial but modest improvement: maintain
            return current

        return current

    # -- Interruption -----------------------------------------------

    async def interrupt(self, reason: str = "external") -> SleepCheckpoint | None:
        """
        Interrupt the current sleep cycle.

        Captures the Descent checkpoint and stores it so the *next* call to
        run_sleep_cycle() resumes from SLOW_WAVE rather than restarting from
        scratch.  Returns the checkpoint for the caller's awareness.
        """
        if not self._is_sleeping:
            return None

        self._logger.warning(
            "sleep_interrupted",
            reason=reason,
            stage=self._current_stage.value if self._current_stage else "unknown",
        )

        checkpoint = self._current_checkpoint
        # Persist checkpoint so the next sleep cycle can restore from SLOW_WAVE
        if checkpoint is not None:
            self._pending_restore_checkpoint = checkpoint
            self._logger.info(
                "interrupt_checkpoint_stored",
                checkpoint_id=checkpoint.id,
                note="next sleep cycle will resume from SLOW_WAVE",
            )

        self._is_sleeping = False
        self._current_stage = None
        self._current_checkpoint = None
        self._sleep_start_mono = None

        return checkpoint
