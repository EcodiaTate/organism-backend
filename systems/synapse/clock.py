"""
EcodiaOS - Cognitive Cycle Clock

The beating heart of EOS. Drives Atune's workspace cycle on every theta tick,
with arousal-modulated adaptive timing.

The clock is the most protected component in the organism. It catches every
exception, logs it, backs off, and continues. The clock never dies.

Timing model:
  - Base period: 150ms (~6.7 Hz)
  - High arousal: contracts toward 80ms (more alert, faster processing)
  - Low arousal: expands toward 500ms (reflective, energy-conserving)
  - EMA smoothing (alpha=0.1): arousal spikes take ~1.5s to fully propagate
  - Overrun detection: warns when cycle exceeds budget
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import structlog

from primitives.affect import InteroceptiveDimension
from systems.synapse.types import (
    ClockState,
    CycleResult,
    SomaticCycleState,
    SynapseEvent,
    SynapseEventType,
    SystemLoad,
)

if TYPE_CHECKING:
    from config import SynapseConfig
    from systems.fovea.gateway import AtuneService
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.synapse.clock")

# Callback signature for per-cycle post-processing
CycleCallback = Callable[[CycleResult], Coroutine[Any, Any, None]]

# Error recovery backoff (ms) - clock backs off on failure, never dies
_ERROR_BACKOFF_MS: float = 500.0

# Soma degradation thresholds (spec §15.4)
_SOMA_WARN_MS: float = 10.0     # Log warning when Soma exceeds this
_SOMA_BYPASS_MS: float = 50.0   # Consider bypass when Soma exceeds this
_SOMA_BYPASS_CONSECUTIVE: int = 3  # Consecutive overruns before bypass

# Number of recent cycle times to keep for jitter measurement
_JITTER_WINDOW: int = 50

# Supervision: max restarts before the clock gives up and logs CRITICAL.
_CLOCK_MAX_RESTARTS: int = 3
_CLOCK_BASE_BACKOFF_S: float = 0.5


class CognitiveClock:
    """
    The theta rhythm that drives EOS's stream of consciousness.

    Every tick triggers one workspace cycle in Atune. Arousal from
    the organism's affect state modulates the cycle period - high arousal
    means faster cycles (more alert), low arousal means slower cycles
    (more reflective, energy-conserving).

    The clock is designed to be unkillable. Any exception during a cycle
    is caught, logged, and the clock backs off before continuing.
    """

    def __init__(
        self,
        atune: AtuneService,
        config: SynapseConfig,
    ) -> None:
        self._atune = atune
        self._config = config
        self._logger = logger.bind(component="clock")

        # System references (wired later by SynapseService)
        self._soma = None  # Soma service (step 0 of theta cycle)
        self._event_bus: EventBus | None = None  # Wired for THETA_CYCLE_START/OVERRUN

        # Soma degradation tracking (spec §15.4 / §XVI)
        # If Soma exceeds _SOMA_WARN_MS, log a warning.
        # If Soma exceeds _SOMA_BYPASS_MS for _SOMA_BYPASS_CONSECUTIVE cycles,
        # bypass Soma entirely until it recovers (next successful sub-warn cycle).
        self._soma_consecutive_overruns: int = 0
        self._soma_bypassed: bool = False

        # Timing state
        self._base_period_ms: float = float(config.cycle_period_ms)
        self._min_period_ms: float = float(config.min_cycle_period_ms)
        self._max_period_ms: float = float(config.max_cycle_period_ms)
        self._current_period_ms: float = self._base_period_ms
        self._target_period_ms: float = self._base_period_ms

        # Arousal tracking (EMA smoothed)
        self._current_arousal: float = 0.1
        self._arousal_alpha: float = 0.1  # EMA smoothing factor

        # Coherence drag - when cross-system coherence drops, the clock
        # slows down to give systems time to synchronize. 0.0 = no drag,
        # 1.0 = max drag (push period toward max).
        self._coherence_drag: float = 0.0

        # Cycle counters
        self._cycle_count: int = 0
        self._overrun_count: int = 0
        self._error_count: int = 0
        self._total_elapsed_ms: float = 0.0

        # Jitter measurement (rolling window of actual cycle durations)
        self._recent_periods: deque[float] = deque(maxlen=_JITTER_WINDOW)

        # Control
        self._running: bool = False
        self._paused: bool = False
        self._task: asyncio.Task[None] | None = None

        # Per-cycle callback - set by SynapseService for rhythm/coherence feeding
        self._on_cycle: CycleCallback | None = None

    # ─── Control ─────────────────────────────────────────────────────

    def start(self) -> asyncio.Task[None]:
        """Start the heartbeat with supervision. Returns the background task handle."""
        if self._running:
            raise RuntimeError("Clock is already running")

        self._running = True
        self._paused = False
        self._task = asyncio.create_task(
            self._supervised_run(),
            name="synapse_cognitive_clock",
        )
        self._logger.info(
            "clock_started",
            base_period_ms=self._base_period_ms,
            min_ms=self._min_period_ms,
            max_ms=self._max_period_ms,
        )
        return self._task

    async def _supervised_run(self) -> None:
        """
        Supervision wrapper for _run_loop.

        _run_loop is already unkillable internally (catches every Exception).
        This outer wrapper guards against the rare case where the loop itself
        terminates unexpectedly (e.g., an asyncio internal error or a bug in
        the loop structure).  It respawns with exponential backoff up to
        _CLOCK_MAX_RESTARTS times, then logs CRITICAL and gives up.

        asyncio.CancelledError is always propagated (graceful stop).
        """
        restart_count = 0
        while self._running:
            try:
                await self._run_loop()
                return  # Clean exit (self._running → False via stop())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                restart_count += 1
                backoff_s = _CLOCK_BASE_BACKOFF_S * (2 ** (restart_count - 1))
                self._logger.error(
                    "clock_loop_crashed_restarting",
                    error=str(exc),
                    restart_attempt=restart_count,
                    max_restarts=_CLOCK_MAX_RESTARTS,
                    backoff_s=backoff_s,
                    exc_info=True,
                )
                if restart_count > _CLOCK_MAX_RESTARTS:
                    self._logger.critical(
                        "clock_loop_exhausted_restarts",
                        restart_count=restart_count,
                        note="Cognitive clock permanently stopped - manual intervention required",
                    )
                    return
                if self._running:
                    self._logger.warning(
                        "clock_loop_restart_backoff",
                        backoff_s=backoff_s,
                        attempt=restart_count,
                    )
                    await asyncio.sleep(backoff_s)

    async def stop(self) -> None:
        """Graceful stop. Waits for the current cycle to complete."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._logger.info(
            "clock_stopped",
            total_cycles=self._cycle_count,
            overruns=self._overrun_count,
            errors=self._error_count,
        )

    async def force_stop(self, reason: str = "") -> None:
        """Forceful stop called by VitalityCoordinator during death sequence.

        Unlike stop(), this is designed to be called from OUTSIDE the clock's
        own control flow. The clock cannot prevent or delay this.
        """
        self._logger.critical(
            "clock_force_stopped",
            reason=reason,
            total_cycles=self._cycle_count,
        )
        await self.stop()

    def pause(self) -> None:
        """Pause the clock (e.g., during safe mode). Cycle loop sleeps."""
        self._paused = True
        self._logger.info("clock_paused", cycle=self._cycle_count)
        if self._event_bus is not None:
            asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.CLOCK_PAUSED,
                source_system="synapse",
                data={"cycle": self._cycle_count, "reason": "external_pause"},
            )))

    def resume(self) -> None:
        """Resume after pause."""
        self._paused = False
        self._logger.info("clock_resumed", cycle=self._cycle_count)
        if self._event_bus is not None:
            asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.CLOCK_RESUMED,
                source_system="synapse",
                data={"cycle": self._cycle_count},
            )))

    def set_coherence_drag(self, drag: float) -> None:
        """
        Set coherence drag to slow the clock when systems are desynchronized.

        Called by SynapseService when the CoherenceMonitor detects low
        cross-system coherence. The drag value (0.0 - 1.0) pushes the
        cycle period toward the maximum, giving systems time to resync.
        """
        self._coherence_drag = max(0.0, min(1.0, drag))

    def set_speed(self, hz: float) -> None:
        """
        Override the base cycle frequency (admin API).

        Clamps to a safe range (1–20 Hz) to prevent either starvation or
        runaway loops. The arousal modulation still operates on top of this
        new base - this sets the resting period, not a hard cap.
        """
        hz = max(1.0, min(20.0, hz))
        period_ms = 1000.0 / hz
        self._base_period_ms = period_ms
        self._min_period_ms = max(50.0, period_ms * 0.5)
        self._max_period_ms = min(2000.0, period_ms * 3.0)
        # Snap current period toward the new base immediately
        self._current_period_ms = period_ms
        self._logger.info(
            "clock_speed_set",
            hz=round(hz, 2),
            period_ms=round(period_ms, 2),
        )

    def set_on_cycle(self, callback: CycleCallback) -> None:
        """Register the per-cycle callback (set by SynapseService)."""
        self._on_cycle = callback

    def set_soma(self, soma: Any) -> None:
        """Wire Soma service (step 0 of theta cycle)."""
        self._soma = soma

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire event bus for THETA_CYCLE_START / THETA_CYCLE_OVERRUN emission (Spec 09 §18)."""
        self._event_bus = event_bus

    # ─── State ───────────────────────────────────────────────────────

    @property
    def state(self) -> ClockState:
        """Snapshot of the clock's current state."""
        jitter = self._compute_jitter()
        rate_hz = 1000.0 / self._current_period_ms if self._current_period_ms > 0 else 0.0
        return ClockState(
            running=self._running,
            paused=self._paused,
            cycle_count=self._cycle_count,
            current_period_ms=self._current_period_ms,
            target_period_ms=self._target_period_ms,
            jitter_ms=jitter,
            arousal=self._current_arousal,
            overrun_count=self._overrun_count,
            actual_rate_hz=rate_hz,
        )

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    # ─── The Heartbeat ───────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """
        The main clock loop. Runs indefinitely until stopped.

        On every tick:
        1. Read arousal from Atune's current affect
        2. Run atune.run_cycle() with a SystemLoad estimate
        3. Compute CycleResult
        4. Fire the on_cycle callback
        5. Adapt the period based on arousal
        6. Sleep for the remainder of the period

        The clock NEVER dies. Exceptions are caught, logged, and backed off.
        """
        self._logger.info("clock_loop_starting")

        while self._running:
            # ── Paused - sleep without ticking ──
            if self._paused:
                await asyncio.sleep(0.5)
                continue

            t0 = time.monotonic()

            try:
                # Emit THETA_CYCLE_START before any work so consumers can
                # time the full cycle. Fire-and-forget; never block the clock.
                if self._event_bus is not None:
                    with contextlib.suppress(Exception):
                        asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.THETA_CYCLE_START,
                            source_system="synapse:clock",
                            data={
                                "cycle_number": self._cycle_count + 1,
                                "period_ms": round(self._current_period_ms, 2),
                                "arousal": round(self._current_arousal, 4),
                            },
                        )))

                # 0. RUN SOMA FIRST (step 0 of theta cycle)
                # Soma produces the AllostaticSignal that all downstream
                # systems read.  Degrades gracefully per spec §15.4/§XVI:
                #   >10ms  → warn
                #   >50ms for 3 consecutive → bypass until recovery
                somatic_state: SomaticCycleState | None = None
                if self._soma is not None and not self._soma_bypassed:
                    soma_t0 = time.monotonic()
                    try:
                        await self._soma.run_cycle()
                        soma_elapsed_ms = (time.monotonic() - soma_t0) * 1000.0

                        # ── Degradation bookkeeping ──
                        if soma_elapsed_ms > _SOMA_BYPASS_MS:
                            self._soma_consecutive_overruns += 1
                            if self._soma_consecutive_overruns >= _SOMA_BYPASS_CONSECUTIVE:
                                self._soma_bypassed = True
                                self._logger.warning(
                                    "soma_bypassed",
                                    cycle=self._cycle_count,
                                    consecutive_overruns=self._soma_consecutive_overruns,
                                    last_ms=round(soma_elapsed_ms, 2),
                                )
                        else:
                            if soma_elapsed_ms > _SOMA_WARN_MS:
                                self._logger.warning(
                                    "soma_slow",
                                    cycle=self._cycle_count,
                                    elapsed_ms=round(soma_elapsed_ms, 2),
                                )
                            # Reset consecutive overruns on any sub-bypass cycle
                            self._soma_consecutive_overruns = 0

                        # Snapshot signal immediately after Soma ran
                        signal = self._soma.get_current_signal()
                        sensed = signal.state.sensed
                        somatic_state = SomaticCycleState(
                            urgency=signal.urgency,
                            dominant_error=signal.dominant_error.value
                            if hasattr(signal.dominant_error, "value")
                            else str(signal.dominant_error),
                            arousal_sensed=sensed.get(InteroceptiveDimension.AROUSAL, 0.4),
                            energy_sensed=sensed.get(InteroceptiveDimension.ENERGY, 0.6),
                            precision_weights={
                                k.value if hasattr(k, "value") else str(k): v
                                for k, v in signal.precision_weights.items()
                            },
                            nearest_attractor=signal.nearest_attractor,
                            trajectory_heading=signal.trajectory_heading,
                            soma_cycle_ms=round(soma_elapsed_ms, 3),
                        )
                    except Exception as soma_exc:
                        self._logger.error(
                            "soma_cycle_error",
                            cycle=self._cycle_count,
                            error=str(soma_exc),
                        )
                        # Continue gracefully - Atune runs with default signal
                elif self._soma is not None and self._soma_bypassed:
                    # Periodically probe recovery: every 50 cycles, try once
                    if self._cycle_count % 50 == 0:
                        probe_t0 = time.monotonic()
                        try:
                            await self._soma.run_cycle()
                            probe_ms = (time.monotonic() - probe_t0) * 1000.0
                            if probe_ms <= _SOMA_WARN_MS:
                                self._soma_bypassed = False
                                self._soma_consecutive_overruns = 0
                                self._logger.info(
                                    "soma_recovered",
                                    cycle=self._cycle_count,
                                    probe_ms=round(probe_ms, 2),
                                )
                                # Build somatic state from the successful probe
                                signal = self._soma.get_current_signal()
                                sensed = signal.state.sensed
                                somatic_state = SomaticCycleState(
                                    urgency=signal.urgency,
                                    dominant_error=signal.dominant_error.value
                                    if hasattr(signal.dominant_error, "value")
                                    else str(signal.dominant_error),
                                    arousal_sensed=sensed.get(InteroceptiveDimension.AROUSAL, 0.4),
                                    energy_sensed=sensed.get(InteroceptiveDimension.ENERGY, 0.6),
                                    precision_weights={
                                        k.value if hasattr(k, "value") else str(k): v
                                        for k, v in signal.precision_weights.items()
                                    },
                                    nearest_attractor=signal.nearest_attractor,
                                    trajectory_heading=signal.trajectory_heading,
                                    soma_cycle_ms=round(probe_ms, 3),
                                )
                        except Exception:
                            pass  # Still degraded - stay bypassed

                # 1. Read current arousal from the organism's affect state
                self._update_arousal()

                # 2. Build a lightweight SystemLoad for Atune
                system_load = self._build_system_load()

                # 3. Run the workspace cycle - the core cognitive tick
                # Pass somatic_state so Atune has the fresh signal for this
                # exact tick (precision weights, urgency → threshold modulation).
                broadcast = await self._atune.run_cycle(
                    system_load=system_load,
                    somatic_state=somatic_state,
                )

                # 4. Measure elapsed time
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                self._total_elapsed_ms += elapsed_ms
                self._recent_periods.append(elapsed_ms)
                self._cycle_count += 1

                # 5. Check for overrun
                overrun = elapsed_ms > self._current_period_ms
                if overrun:
                    self._overrun_count += 1
                    if self._overrun_count % 100 == 1:
                        self._logger.warning(
                            "clock_overrun",
                            cycle=self._cycle_count,
                            elapsed_ms=round(elapsed_ms, 2),
                            budget_ms=round(self._current_period_ms, 2),
                        )
                    # Emit THETA_CYCLE_OVERRUN as a bus event (Spec 09 §18 P8)
                    if self._event_bus is not None:
                        with contextlib.suppress(Exception):
                            _overrun_data = {
                                "cycle_number": self._cycle_count,
                                "elapsed_ms": round(elapsed_ms, 2),
                                "budget_ms": round(self._current_period_ms, 2),
                                "overrun_count": self._overrun_count,
                            }
                            asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                                event_type=SynapseEventType.THETA_CYCLE_OVERRUN,
                                source_system="synapse:clock",
                                data=_overrun_data,
                            )))
                            # Co-emit CLOCK_OVERRUN - subscribed by Thymos for immune response
                            asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                                event_type=SynapseEventType.CLOCK_OVERRUN,
                                source_system="synapse:clock",
                                data=_overrun_data,
                            )))

                # 6. Build cycle result
                result = CycleResult(
                    cycle_number=self._cycle_count,
                    elapsed_ms=round(elapsed_ms, 2),
                    budget_ms=round(self._current_period_ms, 2),
                    overrun=overrun,
                    broadcast_id=broadcast.broadcast_id if broadcast else None,
                    had_broadcast=broadcast is not None,
                    arousal=round(self._current_arousal, 4),
                    salience_composite=(
                        broadcast.salience.composite if broadcast else 0.0
                    ),
                    somatic=somatic_state,
                )

                # 7. Fire the on_cycle callback
                if self._on_cycle is not None:
                    try:
                        await self._on_cycle(result)
                    except Exception as cb_exc:
                        self._logger.error(
                            "on_cycle_callback_error",
                            error=str(cb_exc),
                        )

                # 8. Adapt the period based on arousal
                self._adapt_period()

                # 9. Sleep for the remainder
                sleep_ms = max(0.0, self._current_period_ms - elapsed_ms)
                if sleep_ms > 0:
                    await asyncio.sleep(sleep_ms / 1000.0)

            except asyncio.CancelledError:
                self._logger.info("clock_loop_cancelled")
                return
            except Exception as exc:
                self._error_count += 1
                self._logger.error(
                    "clock_cycle_error",
                    cycle=self._cycle_count,
                    error=str(exc),
                    error_count=self._error_count,
                )
                # Back off - but never die
                await asyncio.sleep(_ERROR_BACKOFF_MS / 1000.0)

    # ─── Adaptive Timing ─────────────────────────────────────────────

    def _update_arousal(self) -> None:
        """
        Read the organism's current arousal and smooth with EMA.

        Prefers Soma's interoceptive AROUSAL dimension (sensed directly from
        compute/token throughput) over Atune's affective arousal (derived from
        percept characteristics). Falls back to Atune when Soma hasn't run yet.

        High arousal = faster cycles (more alert, reactive).
        Low arousal = slower cycles (reflective, energy-conserving).
        """
        raw_arousal = 0.1
        _soma_read_ok = False
        # Prefer Soma's sensed arousal (interoceptive ground truth)
        if self._soma is not None:
            with contextlib.suppress(Exception):
                signal = self._soma.get_current_signal()
                if signal.state is not None:
                    raw_arousal = signal.state.sensed.get(
                        InteroceptiveDimension.AROUSAL, 0.1
                    )
                    _soma_read_ok = True
        # Fall back to Atune's affective arousal only when Soma read failed
        if not _soma_read_ok:
            with contextlib.suppress(Exception):
                raw_arousal = self._atune.current_affect.arousal

        # Exponential moving average smoothing
        self._current_arousal = (
            self._arousal_alpha * raw_arousal
            + (1 - self._arousal_alpha) * self._current_arousal
        )

    def _adapt_period(self) -> None:
        """
        Modulate cycle period based on smoothed arousal and coherence.

        Linear interpolation: high arousal → min period, low arousal → max period.
        Coherence drag: low coherence → push period toward max (slow down to resync).
        Smooth transition (EMA with alpha=0.1) to prevent sudden jumps.
        """
        # Target period: linear interpolation from arousal
        # arousal=1.0 → min_period, arousal=0.0 → max_period
        target = self._max_period_ms - self._current_arousal * (
            self._max_period_ms - self._min_period_ms
        )

        # Coherence drag: when systems are desynchronized, slow down
        # to give them time to catch up. Drag pushes 30% of the remaining
        # headroom toward max period.
        if self._coherence_drag > 0:
            headroom = self._max_period_ms - target
            target += headroom * self._coherence_drag * 0.3

        self._target_period_ms = target

        # Smooth transition - don't jump suddenly
        self._current_period_ms = (
            self._current_period_ms * 0.9 + target * 0.1
        )

        # Clamp to configured bounds
        self._current_period_ms = max(
            self._min_period_ms,
            min(self._max_period_ms, self._current_period_ms),
        )

    def _build_system_load(self) -> SystemLoad:
        """Build a lightweight SystemLoad for Atune from cycle telemetry."""
        # CPU utilisation approximation: how much of the budget we're using
        if self._cycle_count > 0 and self._recent_periods:
            avg_elapsed = sum(self._recent_periods) / len(self._recent_periods)
            cpu_util = min(1.0, avg_elapsed / self._current_period_ms)
        else:
            cpu_util = 0.0

        return SystemLoad(
            cpu_utilisation=round(cpu_util, 4),
            memory_utilisation=0.0,  # Will be enriched by ResourceAllocator
            queue_depth=0,
        )

    def _compute_jitter(self) -> float:
        """
        Compute jitter as the standard deviation of recent cycle periods.

        High jitter indicates erratic timing - a signal used by the
        EmergentRhythmDetector to detect stress states.
        """
        if len(self._recent_periods) < 2:
            return 0.0

        periods = list(self._recent_periods)
        n = len(periods)
        mean = sum(periods) / n
        variance = sum((p - mean) ** 2 for p in periods) / n
        return float(variance ** 0.5)
