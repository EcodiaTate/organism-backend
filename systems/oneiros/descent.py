"""
EcodiaOS - Oneiros v2: Descent Stage

Graceful shutdown of real-time systems. The critical work is SAFE STATE CAPTURE.
Before any deep processing can occur, the system must capture a consistent
snapshot of its entire cognitive state. This is the commit point.

Steps:
1. Capture intelligence ratio, hypothesis count, error count, world model complexity
2. Suspend input channels (real-time processing pauses)
3. Broadcast SLEEP_INITIATED on Synapse
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        neo4j: Any | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._neo4j = neo4j
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
            intelligence_ratio = logos.get_intelligence_ratio()
            world_model_complexity = logos.get_current_complexity()
            cognitive_pressure = logos.budget.state.total_pressure

        checkpoint = SleepCheckpoint(
            intelligence_ratio_at_sleep=intelligence_ratio,
            active_hypothesis_count=active_hypothesis_count,
            unprocessed_error_count=unprocessed_error_count,
            world_model_complexity=world_model_complexity,
            trigger=trigger,
            cognitive_pressure_at_sleep=cognitive_pressure,
        )

        # 2. Tag recent episodes as uncompressed for the Memory Ladder
        tagged = await self._tag_uncompressed_episodes()

        # 3. Suspend input channels
        await self._suspend_input_channels()

        # 4. Broadcast SLEEP_INITIATED
        await self._broadcast_sleep_initiated(checkpoint, target_duration_s)

        self._logger.info(
            "descent_complete",
            checkpoint_id=checkpoint.id,
            intelligence_ratio=round(intelligence_ratio, 4),
            world_model_complexity=round(world_model_complexity, 1),
        )

        return checkpoint

    async def _tag_uncompressed_episodes(self) -> int:
        """Tag recent Neo4j episodes with `uncompressed: true` label.

        The Memory Ladder uses this label to select episodes for processing
        during Slow Wave. After consolidation, episodes are re-labelled.
        """
        if self._neo4j is None:
            return 0

        try:
            query = """
                MATCH (e:Episode)
                WHERE NOT e:Compressed
                  AND e.event_time > datetime() - duration({hours: 24})
                SET e.uncompressed = true
                RETURN count(e) AS tagged
            """
            result = await self._neo4j.execute_write(query)
            tagged = result[0]["tagged"] if result else 0
            self._logger.debug("episodes_tagged_uncompressed", count=tagged)
            return tagged
        except Exception as exc:
            self._logger.debug("episode_tagging_failed", error=str(exc))
            return 0

    async def _suspend_input_channels(self) -> None:
        """
        Suspend real-time input processing.

        Broadcasts `SLEEP_INITIATED` (already done in execute()) which all
        subscribing systems interpret as a directive to pause new-input
        processing.  The flag `input_suspended` is also exposed so that the
        cognitive loop's `on_cycle()` can short-circuit perception while the
        engine runs the sleep pipeline.
        """
        self._input_suspended = True
        self._logger.debug("input_channels_suspended")

    def resume_input_channels(self) -> None:
        """
        Re-enable real-time input processing after wake.

        Called by SleepCycleEngine during EMERGENCE so that on_cycle() stops
        short-circuiting and normal perception resumes.
        """
        self._input_suspended = False
        self._logger.debug("input_channels_resumed")

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

        # Wire ORGANISM_SLEEP - Axon, Identity, SACM, Simula subscribe to this.
        # SLEEP_INITIATED is Oneiros-internal; ORGANISM_SLEEP is the organism-wide signal.
        organism_sleep_event = SynapseEvent(
            event_type=SynapseEventType.ORGANISM_SLEEP,
            source_system="oneiros",
            data={
                "trigger": checkpoint.trigger.value,
                "checkpoint_id": checkpoint.id,
                "scheduled_duration_s": target_duration_s,
            },
        )
        await self._event_bus.emit(organism_sleep_event)
