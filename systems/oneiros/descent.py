"""
EcodiaOS — Oneiros v2: Descent Stage

Graceful shutdown of real-time systems. The critical work is SAFE STATE CAPTURE.
Before any deep processing can occur, the system must capture a consistent
snapshot of its entire cognitive state. This is the commit point.

Steps:
1. Capture intelligence ratio, hypothesis count, error count, world model complexity
2. Suspend input channels (real-time processing pauses)
3. Broadcast SLEEP_INITIATED on Synapse
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from systems.oneiros.types import SleepCheckpoint, SleepTrigger
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.logos.service import LogosService
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oneiros.descent")


class DescentStage:
    """
    Stage 1: Descent (~10% of sleep duration).

    Safe state capture. If sleep is interrupted, the system can
    restore from the checkpoint created here.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._input_suspended = False
        self._logger = logger.bind(stage="descent")

    async def execute(
        self,
        trigger: SleepTrigger,
        logos: LogosService | None = None,
        target_duration_s: float = 7200.0,
        *,
        active_hypothesis_count: int = 0,
        unprocessed_error_count: int = 0,
    ) -> SleepCheckpoint:
        """
        Execute the Descent stage.

        1. Capture consistent system state
        2. Suspend real-time input channels
        3. Broadcast SLEEP_INITIATED

        Args:
            active_hypothesis_count: Current count from Evo (passed by engine).
            unprocessed_error_count: Current backlog count from Fovea (passed by engine).
        """
        self._logger.info("descent_starting", trigger=trigger.value)

        # 1. Capture checkpoint
        intelligence_ratio = 0.0
        world_model_complexity = 0.0
        cognitive_pressure = 0.0

        if logos is not None:
            intelligence_ratio = logos.world_model.measure_intelligence_ratio()
            world_model_complexity = logos.world_model.current_complexity
            cognitive_pressure = logos.budget.state.total_pressure

        checkpoint = SleepCheckpoint(
            intelligence_ratio_at_sleep=intelligence_ratio,
            active_hypothesis_count=active_hypothesis_count,
            unprocessed_error_count=unprocessed_error_count,
            world_model_complexity=world_model_complexity,
            trigger=trigger,
            cognitive_pressure_at_sleep=cognitive_pressure,
        )

        # 2. Suspend input channels
        await self._suspend_input_channels()

        # 3. Broadcast SLEEP_INITIATED
        await self._broadcast_sleep_initiated(checkpoint, target_duration_s)

        self._logger.info(
            "descent_complete",
            checkpoint_id=checkpoint.id,
            intelligence_ratio=round(intelligence_ratio, 4),
            world_model_complexity=round(world_model_complexity, 1),
        )

        return checkpoint

    async def _suspend_input_channels(self) -> None:
        """
        Suspend real-time input processing.

        In the current architecture this is a logical flag — the cognitive
        clock and Atune check this flag to skip perception cycles.
        Full channel suspension is handled by the SleepCycleEngine at a higher level.
        """
        self._input_suspended = True
        self._logger.debug("input_channels_suspended")

    @property
    def input_suspended(self) -> bool:
        return self._input_suspended

    async def _broadcast_sleep_initiated(
        self,
        checkpoint: SleepCheckpoint,
        target_duration_s: float,
    ) -> None:
        """Broadcast SLEEP_INITIATED event on Synapse."""
        if self._event_bus is None:
            return

        event = SynapseEvent(
            event_type=SynapseEventType.SLEEP_INITIATED,
            source_system="oneiros",
            data={
                "trigger": checkpoint.trigger.value,
                "checkpoint_id": checkpoint.id,
                "scheduled_duration_s": target_duration_s,
                "intelligence_ratio_at_sleep": checkpoint.intelligence_ratio_at_sleep,
                "world_model_complexity": checkpoint.world_model_complexity,
            },
        )
        await self._event_bus.emit(event)
