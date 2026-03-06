"""
EcodiaOS — Grid Metabolism Sensor

Maps physical power-grid carbon intensity into the organism's MetabolicState.
The external energy environment becomes a first-class somatic signal: when the
grid is running on cheap, clean power the organism can afford to think harder;
when the grid is dirty or expensive, it conserves compute.

Architecture:
  GridMetabolismSensor runs a background asyncio task that polls the
  Electricity Maps API (https://api.electricitymap.org/v3/carbon-intensity/latest)
  on a configurable interval (default 15 minutes). The raw ``carbonIntensity``
  value (gCO2eq/kWh) is passed through a three-band classifier to produce a
  MetabolicState. On state transition, a SynapseEvent of type
  GRID_METABOLISM_CHANGED is emitted on the event bus.

Iron Rules (same contract as ExternalVolatilitySensor):
  - Never called from the theta cycle — runs on its own asyncio timer.
  - All HTTP calls are bounded by ``fetch_timeout_s`` with a hard outer cap.
  - Any exception in the fetch or classification path is swallowed and logged;
    the sensor stays running and retries at the next poll interval.
  - No LLM calls, no DB writes, no side-effects beyond event emission.

API reference:
  GET https://api.electricitymap.org/v3/carbon-intensity/latest
      ?lat={lat}&lon={lon}
  Headers: auth-token: <api_key>
  Response: {"carbonIntensity": 123.4, "datetime": "...", "zone": "AU-NSW", ...}

  Free tier: 100 req/month. At 15-minute intervals the sensor uses ≤ 2,976
  req/month — a paid plan is required for production. Set api_key = "" to
  disable without touching enabled flag (sensor will log a warning and skip).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import structlog

from systems.synapse.types import (
    MetabolicState,
    SynapseEvent,
    SynapseEventType,
)

if TYPE_CHECKING:
    from config import EnergyGridConfig
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.soma.grid_sensor")

# Electricity Maps v3 endpoint — lat/lon resolved to nearest grid zone
_ELECTRICITY_MAPS_URL = (
    "https://api.electricitymap.org/v3/carbon-intensity/latest"
    "?lat={lat}&lon={lon}"
)

# Hard outer cap on the entire fetch-and-classify pass (seconds).
_OUTER_TIMEOUT_S: float = 20.0


def _classify(carbon_intensity_g: float, config: EnergyGridConfig) -> MetabolicState:
    """
    Map a carbon intensity reading to a MetabolicState.

    Bands (configurable via EnergyGridConfig):
      < green_threshold_g   → GREEN_SURPLUS
      > conservation_threshold_g → CONSERVATION
      otherwise             → NORMAL
    """
    if carbon_intensity_g < config.green_threshold_g:
        return MetabolicState.GREEN_SURPLUS
    if carbon_intensity_g > config.conservation_threshold_g:
        return MetabolicState.CONSERVATION
    return MetabolicState.NORMAL


class GridMetabolismSensor:
    """
    Background sensor that converts physical power-grid carbon intensity
    into a MetabolicState and emits GRID_METABOLISM_CHANGED on transition.

    Lifecycle:
        sensor = GridMetabolismSensor(config, event_bus)
        await sensor.start()
        ...
        await sensor.stop()
    """

    def __init__(
        self,
        config: EnergyGridConfig,
        event_bus: EventBus | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus

        # Last known state — None until first successful fetch
        self._current_state: MetabolicState | None = None
        # Last raw reading retained for diagnostics
        self._last_carbon_intensity_g: float | None = None

        # Background task handle
        self._task: asyncio.Task[None] | None = None
        self._running: bool = False

        self._log = logger.bind(sensor="grid_metabolism")

    # ─── Public read ─────────────────────────────────────────────────────────

    @property
    def current_state(self) -> MetabolicState | None:
        """Last classified MetabolicState, or None before first successful poll."""
        return self._current_state

    @property
    def last_carbon_intensity_g(self) -> float | None:
        """Last raw carbon intensity reading in gCO2eq/kWh."""
        return self._last_carbon_intensity_g

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling loop."""
        if not self._config.enabled:
            self._log.info("grid_sensor_disabled")
            return
        if not self._config.api_key:
            self._log.warning(
                "grid_sensor_no_api_key",
                hint="Set ECODIAOS_ENERGY_GRID__API_KEY to enable grid polling",
            )
            return
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._poll_loop(),
            name="soma.grid_metabolism_sensor",
        )
        self._log.info(
            "grid_sensor_started",
            lat=self._config.latitude,
            lon=self._config.longitude,
            interval_s=self._config.polling_interval_s,
            green_threshold_g=self._config.green_threshold_g,
            conservation_threshold_g=self._config.conservation_threshold_g,
        )

    async def stop(self) -> None:
        """Gracefully cancel the polling loop."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._log.info("grid_sensor_stopped")

    # ─── Core loop ───────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """
        Main background coroutine.

        Drift-free: sleeps for (interval - elapsed) rather than a fixed
        duration so scheduled polls stay aligned to wall-clock intervals
        even when fetches take variable time.
        """
        while self._running:
            loop_start = time.monotonic()
            try:
                await asyncio.wait_for(
                    self._fetch_and_update(),
                    timeout=_OUTER_TIMEOUT_S,
                )
            except TimeoutError:
                self._log.warning("grid_fetch_timeout_outer")
            except Exception as exc:
                self._log.warning("grid_poll_error", error=str(exc))

            elapsed = time.monotonic() - loop_start
            sleep_s = max(0.0, self._config.polling_interval_s - elapsed)
            try:
                await asyncio.sleep(sleep_s)
            except asyncio.CancelledError:
                break

    # ─── Fetch ───────────────────────────────────────────────────────────────

    async def _fetch_and_update(self) -> None:
        """Fetch carbon intensity, classify, and emit on state change."""
        carbon_g = await self._fetch_carbon_intensity()
        if carbon_g is None:
            # API unavailable — keep existing state, skip update
            self._log.debug("grid_no_data_skip")
            return

        self._last_carbon_intensity_g = carbon_g
        new_state = _classify(carbon_g, self._config)

        self._log.debug(
            "grid_reading",
            carbon_intensity_g=round(carbon_g, 1),
            state=new_state,
            previous_state=self._current_state,
        )

        if new_state == self._current_state:
            return  # No transition — nothing to emit

        previous_state = self._current_state
        self._current_state = new_state

        self._log.info(
            "grid_metabolism_changed",
            previous=previous_state,
            current=new_state,
            carbon_intensity_g=round(carbon_g, 1),
        )

        await self._emit_state_change(new_state, carbon_g, previous_state)

    async def _fetch_carbon_intensity(self) -> float | None:
        """
        Call the Electricity Maps v3 API and return carbonIntensity (gCO2eq/kWh).
        Returns None on any error so the caller can skip gracefully.
        """
        import json as _json
        import urllib.error
        import urllib.request

        url = _ELECTRICITY_MAPS_URL.format(
            lat=self._config.latitude,
            lon=self._config.longitude,
        )
        api_key = self._config.api_key
        timeout = self._config.fetch_timeout_s

        loop = asyncio.get_running_loop()

        def _do_request() -> dict[str, Any]:
            req = urllib.request.Request(
                url,
                headers={
                    "auth-token": api_key,
                    "Accept": "application/json",
                    "User-Agent": "EcodiaOS/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            return _json.loads(body)  # type: ignore[no-any-return]

        try:
            data: dict[str, Any] = await asyncio.wait_for(
                loop.run_in_executor(None, _do_request),
                timeout=timeout + 2.0,
            )
        except urllib.error.HTTPError as exc:
            self._log.warning(
                "grid_http_error", status=exc.code, url=url
            )
            return None
        except (TimeoutError, OSError) as exc:
            self._log.warning("grid_fetch_error", error=str(exc))
            return None

        raw = data.get("carbonIntensity")
        if raw is None:
            self._log.warning("grid_response_missing_field", keys=list(data.keys()))
            return None

        return float(raw)

    # ─── Event emission ──────────────────────────────────────────────────────

    async def _emit_state_change(
        self,
        state: MetabolicState,
        carbon_intensity_g: float,
        previous_state: MetabolicState | None,
    ) -> None:
        """Publish GRID_METABOLISM_CHANGED on the event bus."""
        if self._event_bus is None:
            return
        try:
            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.GRID_METABOLISM_CHANGED,
                    data={
                        "state": state,
                        "previous_state": previous_state,
                        "carbon_intensity_g": round(carbon_intensity_g, 1),
                    },
                    source_system="soma.grid_metabolism_sensor",
                )
            )
        except Exception as exc:
            # Event bus failures are non-fatal — state is already updated locally.
            self._log.warning("grid_event_emit_failed", error=str(exc))
