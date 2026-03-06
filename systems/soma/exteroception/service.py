"""
EcodiaOS — Exteroception Service

Top-level coordinator for the Cross-Modal Synesthesia layer. Manages a
set of exteroceptive adapters, routes their readings through the mapping
engine, and injects the resulting pressure into SomaService.

Architecture:
  ExteroceptionService runs a single background asyncio task that polls
  all registered adapters on a configurable interval (default 120s).
  Raw readings are normalised by each adapter, then fed to the
  ExteroceptiveMappingEngine which produces an ExteroceptivePressure.
  This pressure is injected into SomaService via two channels:

  1. ``inject_external_stress(ambient_stress)`` — the existing scalar
     channel, maintaining backward compatibility with the original
     ExternalVolatilitySensor.

  2. ``inject_exteroceptive_pressure(pressure)`` — a new method on
     SomaService that applies per-dimension pressure deltas to the
     interoceptor's next sense() pass.

  The service also emits SOMA_STATE_SPIKE events on the Synapse event
  bus when ambient_stress changes rapidly, mirroring the existing
  spike detection logic.

Iron Rules:
  - Never called from the theta cycle.
  - All adapter fetches are bounded by per-adapter timeout.
  - Any adapter failure is isolated — other adapters continue.
  - No LLM calls, no DB writes.
  - The entire poll-and-map pass must complete within _OUTER_TIMEOUT_S.

Lifecycle:
  service = ExteroceptionService(config, event_bus)
  service.set_soma(soma_service)
  service.register_adapter(MarketDataAdapter())
  await service.start()
  ...
  await service.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from systems.synapse.types import SynapseEvent, SynapseEventType

from .mapping_engine import ExteroceptiveMappingEngine

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

    from .adapters import BaseExteroceptiveAdapter
    from .types import (
        ExteroceptivePressure,
        ExteroceptiveReading,
        ModalityMapping,
    )

logger = structlog.get_logger("systems.soma.exteroception.service")

# Hard outer timeout for the entire poll-and-map cycle (seconds)
_OUTER_TIMEOUT_S: float = 30.0

# Rolling window for spike detection
_SPIKE_WINDOW_SAMPLES: int = 12


class ExteroceptionService:
    """Coordinates exteroceptive data ingestion and somatic pressure injection.

    This is the "nervous system" for external world sensing: it manages
    adapters (the sensory organs), the mapping engine (the thalamic relay),
    and the injection pathway (the somatic integration).
    """

    def __init__(
        self,
        poll_interval_s: float = 120.0,
        ema_alpha: float = 0.3,
        max_total_pressure: float = 0.25,
        spike_threshold: float = 0.15,
        event_bus: EventBus | None = None,
    ) -> None:
        self._poll_interval_s = poll_interval_s
        self._spike_threshold = spike_threshold
        self._event_bus = event_bus

        # Core sub-systems
        self._mapping_engine = ExteroceptiveMappingEngine(
            ema_alpha=ema_alpha,
            max_total_pressure=max_total_pressure,
        )
        self._adapters: list[BaseExteroceptiveAdapter] = []

        # Soma reference
        self._soma: Any = None

        # State
        self._task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._last_pressure: ExteroceptivePressure | None = None

        # Spike detection history
        self._stress_history: deque[tuple[float, float]] = deque(
            maxlen=_SPIKE_WINDOW_SAMPLES
        )

        self._log = logger.bind(system="exteroception")

    # ─── Wiring ───────────────────────────────────────────────────

    def set_soma(self, soma: Any) -> None:
        """Wire in the SomaService reference for pressure injection."""
        self._soma = soma

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire in the Synapse event bus for spike emission."""
        self._event_bus = event_bus

    def register_adapter(self, adapter: BaseExteroceptiveAdapter) -> None:
        """Register an exteroceptive data adapter."""
        self._adapters.append(adapter)
        self._log.info(
            "adapter_registered",
            adapter=type(adapter).__name__,
            modalities=[m.value for m in adapter.modalities],
        )

    def update_mapping(self, mapping: ModalityMapping) -> None:
        """Hot-update a modality mapping (e.g., from Evo learning)."""
        self._mapping_engine.update_mapping(mapping)

    # ─── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling loop."""
        if not self._adapters:
            self._log.info("exteroception_no_adapters_skip")
            return
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(
            self._poll_loop(),
            name="soma.exteroception_service",
        )
        self._log.info(
            "exteroception_started",
            interval_s=self._poll_interval_s,
            adapters=len(self._adapters),
        )

    async def stop(self) -> None:
        """Gracefully cancel the polling loop."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._log.info("exteroception_stopped")

    # ─── Public Read ──────────────────────────────────────────────

    @property
    def current_pressure(self) -> ExteroceptivePressure | None:
        """Last computed exteroceptive pressure, or None before first poll."""
        return self._last_pressure

    @property
    def mapping_engine(self) -> ExteroceptiveMappingEngine:
        """Expose mapping engine for inspection or Evo updates."""
        return self._mapping_engine

    # ─── Core Loop ────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Main background coroutine. Drift-free polling."""
        while self._running:
            loop_start = time.monotonic()
            try:
                await asyncio.wait_for(
                    self._fetch_map_inject(),
                    timeout=_OUTER_TIMEOUT_S,
                )
            except TimeoutError:
                self._log.warning("exteroception_poll_timeout")
            except Exception as exc:
                self._log.warning("exteroception_poll_error", error=str(exc))

            elapsed = time.monotonic() - loop_start
            sleep_s = max(0.0, self._poll_interval_s - elapsed)
            try:
                await asyncio.sleep(sleep_s)
            except asyncio.CancelledError:
                break

    async def _fetch_map_inject(self) -> None:
        """Fetch from all adapters, map to pressure, inject into Soma."""
        # Step 1: Fetch from all adapters concurrently
        readings = await self._fetch_all()

        if not readings:
            self._log.debug("exteroception_no_readings")
            return

        # Step 2: Map readings to pressure
        pressure = self._mapping_engine.process_readings(readings)
        self._last_pressure = pressure

        # Step 3: Inject into Soma
        self._inject_pressure(pressure)

        # Step 4: Spike detection
        now = time.monotonic()
        self._stress_history.append((now, pressure.ambient_stress))
        await self._maybe_emit_spike(pressure.ambient_stress, now)

        self._log.debug(
            "exteroception_cycle_complete",
            readings=len(readings),
            active_modalities=[m.value for m in pressure.active_modalities],
            ambient_stress=round(pressure.ambient_stress, 4),
            total_pressure=round(pressure.total_absolute_pressure(), 4),
        )

    async def _fetch_all(self) -> list[ExteroceptiveReading]:
        """Fetch from all registered adapters concurrently."""
        if not self._adapters:
            return []

        tasks = [
            asyncio.create_task(adapter.fetch())
            for adapter in self._adapters
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        readings: list[ExteroceptiveReading] = []
        for i, result in enumerate(results):
            if isinstance(result, list):
                readings.extend(result)
            elif isinstance(result, Exception):
                adapter_name = type(self._adapters[i]).__name__
                self._log.debug(
                    "adapter_fetch_failed",
                    adapter=adapter_name,
                    error=str(result),
                )

        return readings

    def _inject_pressure(self, pressure: ExteroceptivePressure) -> None:
        """Push pressure into SomaService via both injection channels."""
        if self._soma is None:
            return

        # Channel 1: Legacy scalar stress (backward compatible)
        self._soma.inject_external_stress(pressure.ambient_stress)

        # Channel 2: Per-dimension pressure (new exteroceptive pathway)
        if hasattr(self._soma, "inject_exteroceptive_pressure"):
            self._soma.inject_exteroceptive_pressure(pressure)

    # ─── Spike Detection ──────────────────────────────────────────

    async def _maybe_emit_spike(self, current: float, now: float) -> None:
        """Emit SOMA_STATE_SPIKE if ambient stress changed rapidly."""
        if self._event_bus is None:
            return
        if len(self._stress_history) < 2:
            return

        one_hour_ago = now - 3600.0
        window_values = [
            v for ts, v in self._stress_history if ts >= one_hour_ago
        ]
        if not window_values:
            return

        delta = current - min(window_values)
        if delta <= self._spike_threshold:
            return

        self._log.info(
            "exteroceptive_spike_detected",
            delta=round(delta, 3),
            current_stress=round(current, 3),
        )

        try:
            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.SOMA_STATE_SPIKE,
                    data={
                        "dimension": "exteroceptive_stress",
                        "value": round(current, 4),
                        "delta_1h": round(delta, 4),
                        "source": "exteroception_service",
                    },
                    source_system="soma.exteroception",
                )
            )
        except Exception as exc:
            self._log.warning("spike_event_emit_failed", error=str(exc))

    # ─── Manual Injection ─────────────────────────────────────────

    def inject_reading(self, reading: ExteroceptiveReading) -> None:
        """Manually inject a reading outside the poll loop.

        Useful for testing or for push-based data sources (WebSocket
        streams, webhooks) that don't fit the pull-based adapter model.
        """
        pressure = self._mapping_engine.process_readings([reading])
        self._last_pressure = pressure
        self._inject_pressure(pressure)
