"""
EcodiaOS — Oneiros v2: Sleep Cycle Engine

Top-level orchestrator for v2 sleep cycles. Drives the four-stage pipeline:
  DESCENT -> SLOW_WAVE -> REM -> EMERGENCE

With optional Lucid Dreaming between REM and EMERGENCE when Simula has
pending mutation proposals.

Each stage transition broadcasts SLEEP_STAGE_TRANSITION on Synapse.
If sleep is interrupted, the system can restore from the Descent checkpoint.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

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
    SleepSchedulerConfig,
    SleepStageV2,
    SleepTrigger,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.logos.service import LogosService
    from systems.synapse.event_bus import EventBus

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

        # Sub-stages
        self._scheduler = SleepScheduler(config=self._config)
        self._descent = DescentStage(event_bus=event_bus)
        self._slow_wave: SlowWaveStage | None = None  # needs logos
        self._rem: REMStage | None = None  # needs logos
        self._lucid: LucidDreamingStage | None = None  # needs logos + simula
        self._emergence = EmergenceStage(event_bus=event_bus)

        # Cross-system refs (set via setters)
        self._logos: LogosService | None = None
        self._fovea: FoveaErrorDomainProtocol | None = None
        self._evo: EvoHypothesisProtocol | None = None
        self._simula: SimulaProtocol | None = None
        self._equor: ConstitutionalCheckProtocol | None = None

        # State
        self._is_sleeping = False
        self._current_stage: SleepStageV2 | None = None
        self._current_checkpoint: SleepCheckpoint | None = None
        self._sleep_start_mono: float | None = None

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
    def emergence(self) -> EmergenceStage:
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

        report = SleepCycleV2Report(trigger=trigger)

        self._logger.info(
            "sleep_cycle_starting",
            trigger=trigger.value,
            target_duration_s=target_duration_s,
        )

        try:
            # -- Stage 1: DESCENT (10%) --
            # Capture pre-sleep counts from Evo and Fovea for the checkpoint.
            # Both are fetched via duck-typed optional methods to avoid hard deps.
            active_hypothesis_count = 0
            if self._evo is not None and hasattr(self._evo, "get_active_hypothesis_count"):
                with contextlib.suppress(Exception):
                    active_hypothesis_count = await self._evo.get_active_hypothesis_count()

            unprocessed_error_count = 0
            if self._fovea is not None and hasattr(self._fovea, "get_unprocessed_error_count"):
                with contextlib.suppress(Exception):
                    unprocessed_error_count = await self._fovea.get_unprocessed_error_count()

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

    # -- Interruption -----------------------------------------------

    async def interrupt(self, reason: str = "external") -> SleepCheckpoint | None:
        """
        Interrupt the current sleep cycle.

        Returns the checkpoint from Descent so the caller can decide
        whether to restore or discard.
        """
        if not self._is_sleeping:
            return None

        self._logger.warning(
            "sleep_interrupted",
            reason=reason,
            stage=self._current_stage.value if self._current_stage else "unknown",
        )

        checkpoint = self._current_checkpoint
        self._is_sleeping = False
        self._current_stage = None
        self._current_checkpoint = None
        self._sleep_start_mono = None

        return checkpoint
