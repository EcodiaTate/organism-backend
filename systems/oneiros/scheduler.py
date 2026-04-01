"""
EcodiaOS - Oneiros v2: Sleep Scheduler

Three independent triggers - any one sufficient to initiate sleep:
1. Scheduled: every N hours (configurable, default 6h)
2. Cognitive pressure: Logos budget pressure >= threshold
3. Compression backlog: Fovea unprocessed errors >= threshold

Safety check: can_sleep_now() gates against active commitments,
unsafe suspension state, or active constitutional crisis.

Built against protocols - does not import Nova, Axon, or Equor directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import structlog

from primitives.common import utc_now
from systems.oneiros.types import SleepSchedulerConfig, SleepTrigger

logger = structlog.get_logger("oneiros.scheduler")


# ─── Protocols for systems we check but don't import ─────────────


@runtime_checkable
class CognitiveBudgetProtocol(Protocol):
    """Read cognitive pressure from Logos budget."""

    @property
    def total_pressure(self) -> float:
        """0.0 = empty, 1.0 = full."""
        ...


@runtime_checkable
class NovaProtocol(Protocol):
    """Check if Nova has active commitments."""

    async def has_active_commitments(self) -> bool: ...


@runtime_checkable
class AxonProtocol(Protocol):
    """Check if Axon is safe to suspend."""

    async def is_safe_to_suspend(self) -> bool: ...


@runtime_checkable
class EquorCrisisProtocol(Protocol):
    """Check if Equor has an active crisis."""

    async def is_crisis_active(self) -> bool: ...


@runtime_checkable
class FoveaBacklogProtocol(Protocol):
    """Get unprocessed error count from Fovea."""

    async def get_unprocessed_error_count(self) -> int: ...


# ─── Sleep Scheduler ─────────────────────────────────────────────


class SleepScheduler:
    """
    Determines when EOS should sleep and whether it is safe to do so.

    Three independent triggers - any one is sufficient.
    Safety gate prevents sleep during active commitments or crises.
    """

    def __init__(self, config: SleepSchedulerConfig | None = None) -> None:
        self._config = config or SleepSchedulerConfig()
        # Initialize to epoch so the scheduled trigger fires immediately on first boot,
        # rather than waiting a full interval after construction.
        self._last_sleep: datetime = datetime(1970, 1, 1, tzinfo=UTC)
        self._logger = logger.bind(component="scheduler")

    @property
    def config(self) -> SleepSchedulerConfig:
        return self._config

    @property
    def last_sleep(self) -> datetime:
        return self._last_sleep

    def record_sleep_completed(self) -> None:
        """Mark that a sleep cycle just completed."""
        self._last_sleep = utc_now()

    def hours_since_sleep(self) -> float:
        return float((utc_now() - self._last_sleep).total_seconds()) / 3600.0

    async def should_sleep(
        self,
        budget: CognitiveBudgetProtocol | None = None,
        fovea: FoveaBacklogProtocol | None = None,
    ) -> tuple[bool, SleepTrigger | None]:
        """
        Check whether any sleep trigger is active.

        Returns (should_sleep, trigger_reason).
        Three independent triggers - any one is sufficient.
        """
        # 1. Scheduled
        hours = self.hours_since_sleep()
        if hours >= self._config.scheduled_interval_hours:
            self._logger.info(
                "sleep_trigger_scheduled",
                hours_since=round(hours, 2),
                threshold=self._config.scheduled_interval_hours,
            )
            return True, SleepTrigger.SCHEDULED

        # 2. Cognitive pressure
        if budget is not None:
            pressure = budget.total_pressure
            if pressure >= self._config.cognitive_pressure_threshold:
                self._logger.info(
                    "sleep_trigger_cognitive_pressure",
                    pressure=round(pressure, 3),
                    threshold=self._config.cognitive_pressure_threshold,
                )
                return True, SleepTrigger.COGNITIVE_PRESSURE

        # 3. Compression backlog
        if fovea is not None:
            error_count = await fovea.get_unprocessed_error_count()
            if error_count >= self._config.unprocessed_error_threshold:
                self._logger.info(
                    "sleep_trigger_compression_backlog",
                    error_count=error_count,
                    threshold=self._config.unprocessed_error_threshold,
                )
                return True, SleepTrigger.COMPRESSION_BACKLOG

        return False, None

    async def can_sleep_now(
        self,
        nova: NovaProtocol | None = None,
        axon: AxonProtocol | None = None,
        equor: EquorCrisisProtocol | None = None,
    ) -> bool:
        """
        Even when sleep is needed, certain conditions must be met.
        Cannot sleep during active commitments, unsafe execution state,
        or constitutional crisis.

        If a system reference is None, that check is skipped (assumed safe).
        """
        # Check Nova - no active commitments
        if nova is not None and await nova.has_active_commitments():
            self._logger.debug("sleep_blocked", reason="nova_active_commitments")
            return False

        # Check Axon - safe to suspend
        if axon is not None and not await axon.is_safe_to_suspend():
            self._logger.debug("sleep_blocked", reason="axon_unsafe_to_suspend")
            return False

        # Check Equor - no constitutional crisis
        if equor is not None and await equor.is_crisis_active():
            self._logger.debug("sleep_blocked", reason="equor_crisis_active")
            return False

        return True
