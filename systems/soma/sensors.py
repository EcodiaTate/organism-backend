"""
EcodiaOS — Soma External Volatility Sensor (Cross-Modal Synesthesia)

Maps exogenous data streams into endogenous somatic states. The organism
does not live in a vacuum — systemic stress in the outside world becomes
felt stress in the body.

Architecture:
  ExternalVolatilitySensor runs a background asyncio task that polls
  lightweight public endpoints on a configurable interval (default 5 min).
  Raw metrics are normalised to [0.0, 1.0] and smoothed via EMA into a
  single ``economic_stress`` float. This float is pushed into SomaService
  via ``inject_external_stress()``, which modulates the TEMPORAL_PRESSURE
  and AROUSAL interoceptive dimensions.

  If the smoothed stress value jumps by more than ``spike_threshold``
  within a 1-hour rolling window, a SynapseEvent of type SOMA_STATE_SPIKE
  is emitted on the event bus.

Iron Rules (mirrors Soma's own rules):
  - Never called from the theta cycle — runs on its own asyncio timer.
  - All HTTP calls are bounded by ``fetch_timeout_s`` with a hard outer cap.
  - Any exception in the fetch or normalisation path is swallowed and logged;
    the sensor stays running and retries at the next poll interval.
  - No LLM calls, no DB writes, no side-effects beyond a float injection.

Data sources (each tried in order, first success wins):
  1. CoinGecko public API — BTC + ETH 24h price change percentages
  2. GitHub REST API (no token required) — open issues + open PRs in the
     ecodiaos organisation as a proxy for codebase activity/pressure.

Normalisation:
  ``economic_stress = sigmoid_normalise(mean_abs_24h_change_pct)``
  where the sigmoid maps 0 % → 0.0, ~5 % → 0.5, ≥ 15 % → ≈ 1.0.

  GitHub pressure:
  ``github_stress = min(open_issues_and_prs / 200.0, 1.0)``
  Blended: ``raw = 0.7 * economic_stress + 0.3 * github_stress``
  Then EMA-smoothed before pushing to Soma.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from config import SomaConfig
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.soma.sensors")

# ─── Public API endpoint (no auth, rate-limited to ~30 req/min) ──────────────
_COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
)

# GitHub REST endpoint — public repos, unauthenticated (60 req/hr)
_GITHUB_SEARCH_URL = (
    "https://api.github.com/search/issues"
    "?q=repo:ecodiaos/ecodiaos+state:open&per_page=1"
)

# Sigmoid steepness: k controls how aggressively mid-range volatility maps to stress.
# k=0.3 → 5 % change ≈ 0.5 stress; k=0.2 → 5 % change ≈ 0.37 stress.
_SIGMOID_K: float = 0.3

# Hard cap on how many historical stress samples we retain for spike detection.
# At 300s poll interval, 12 samples = 1 hour.
_SPIKE_WINDOW_SAMPLES: int = 12

# Total outer timeout wrapping the entire fetch-and-normalise pass (seconds).
_OUTER_TIMEOUT_S: float = 15.0


def _sigmoid_normalise(abs_pct: float) -> float:
    """Map an absolute percentage change to [0, 1] via a logistic curve."""
    # sigmoid(k * abs_pct) centred at 0; result is already in (0.5, 1.0) for any
    # positive input. Rescale to (0.0, 1.0) so 0 % → 0.0.
    raw = 1.0 / (1.0 + math.exp(-_SIGMOID_K * abs_pct))
    # raw is in (0.5, 1.0) — rescale linearly to (0.0, 1.0)
    return (raw - 0.5) * 2.0


class RawVolatilityData:
    """Intermediate bag for fetched metrics before normalisation."""

    __slots__ = ("btc_24h_pct", "eth_24h_pct", "github_open_count", "source_ok")

    def __init__(
        self,
        btc_24h_pct: float = 0.0,
        eth_24h_pct: float = 0.0,
        github_open_count: int = 0,
        source_ok: bool = False,
    ) -> None:
        self.btc_24h_pct = btc_24h_pct
        self.eth_24h_pct = eth_24h_pct
        self.github_open_count = github_open_count
        self.source_ok = source_ok


class ExternalVolatilitySensor:
    """
    Background sensor that converts external market + codebase signals into
    a normalised [0, 1] economic stress float and pushes it into SomaService.

    Lifecycle:
        sensor = ExternalVolatilitySensor(config, event_bus)
        sensor.set_soma(soma_service)
        await sensor.start()
        ...
        await sensor.stop()
    """

    def __init__(
        self,
        config: SomaConfig,
        event_bus: EventBus | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus

        # Current smoothed stress value — starts neutral
        self._stress: float = 0.0
        # EMA state
        self._ema_initialised: bool = False

        # Rolling window of (timestamp_s, stress) for spike detection
        self._stress_history: deque[tuple[float, float]] = deque(maxlen=_SPIKE_WINDOW_SAMPLES)

        # Reference to SomaService (set via set_soma)
        self._soma: Any = None

        # Background task handle
        self._task: asyncio.Task[None] | None = None
        self._running: bool = False

        self._log = logger.bind(sensor="external_volatility")

    # ─── Wiring ──────────────────────────────────────────────────────────────

    def set_soma(self, soma: Any) -> None:
        """Wire in the SomaService reference so we can inject stress."""
        self._soma = soma

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling loop."""
        if not self._config.volatility_sensor_enabled:
            self._log.info("volatility_sensor_disabled")
            return
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._poll_loop(),
            name="soma.volatility_sensor",
        )
        self._log.info(
            "volatility_sensor_started",
            interval_s=self._config.volatility_poll_interval_s,
            spike_threshold=self._config.volatility_spike_threshold,
        )

    async def stop(self) -> None:
        """Gracefully cancel the polling loop."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._log.info("volatility_sensor_stopped")

    # ─── Public read ─────────────────────────────────────────────────────────

    @property
    def current_stress(self) -> float:
        """Last EMA-smoothed stress value pushed into Soma."""
        return self._stress

    # ─── Core loop ───────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """
        Main background coroutine.

        Waits ``poll_interval_s`` between passes; does NOT drift because we
        sleep for (interval - elapsed) rather than a fixed sleep. On
        failure the interval is observed from the start of the failed attempt
        so retries are naturally rate-limited.
        """
        while self._running:
            loop_start = time.monotonic()
            try:
                await asyncio.wait_for(
                    self._fetch_and_update(),
                    timeout=_OUTER_TIMEOUT_S,
                )
            except TimeoutError:
                self._log.warning("volatility_fetch_timeout_outer")
            except Exception as exc:
                self._log.warning("volatility_poll_error", error=str(exc))

            elapsed = time.monotonic() - loop_start
            sleep_s = max(0.0, self._config.volatility_poll_interval_s - elapsed)
            try:
                await asyncio.sleep(sleep_s)
            except asyncio.CancelledError:
                break

    # ─── Fetch ───────────────────────────────────────────────────────────────

    async def _fetch_and_update(self) -> None:
        """Fetch, normalise, smooth, inject, and maybe emit a spike event."""
        raw = await self._fetch_raw()
        if not raw.source_ok:
            # Nothing usable — keep existing stress, skip update
            self._log.debug("volatility_no_data_skip")
            return

        normalised = self._normalise(raw)
        smoothed = self._apply_ema(normalised)
        self._stress = smoothed

        now = time.monotonic()
        self._stress_history.append((now, smoothed))

        if self._soma is not None:
            self._soma.inject_external_stress(smoothed)

        await self._maybe_emit_spike(smoothed, now)

        self._log.debug(
            "volatility_updated",
            btc_24h=round(raw.btc_24h_pct, 2),
            eth_24h=round(raw.eth_24h_pct, 2),
            github_open=raw.github_open_count,
            normalised=round(normalised, 3),
            smoothed=round(smoothed, 3),
        )

    async def _fetch_raw(self) -> RawVolatilityData:
        """
        Attempt CoinGecko first, then optionally GitHub.
        Both are fire-and-don't-care on failure — returns what we have.
        """
        # CoinGecko and GitHub fetches run concurrently; each has its own timeout.
        cg_task = asyncio.create_task(self._fetch_coingecko())
        gh_task = asyncio.create_task(self._fetch_github())

        cg_result, gh_result = await asyncio.gather(
            cg_task, gh_task, return_exceptions=True
        )

        btc_pct = 0.0
        eth_pct = 0.0
        gh_count = 0
        any_ok = False

        if isinstance(cg_result, dict):
            btc_pct = float(cg_result.get("btc_24h", 0.0))
            eth_pct = float(cg_result.get("eth_24h", 0.0))
            any_ok = True
        elif isinstance(cg_result, Exception):
            self._log.debug("coingecko_fetch_failed", error=str(cg_result))

        if isinstance(gh_result, int):
            gh_count = gh_result
            any_ok = True
        elif isinstance(gh_result, Exception):
            self._log.debug("github_fetch_failed", error=str(gh_result))

        return RawVolatilityData(
            btc_24h_pct=btc_pct,
            eth_24h_pct=eth_pct,
            github_open_count=gh_count,
            source_ok=any_ok,
        )

    async def _fetch_coingecko(self) -> dict[str, float]:
        """
        Fetch BTC + ETH 24h price change from CoinGecko (no API key needed).
        Returns {'btc_24h': float, 'eth_24h': float} on success.
        Raises on any HTTP / parse / timeout error — caller handles.
        """
        import urllib.error
        import urllib.request

        # urllib is sync; run in executor so we don't block the event loop.
        loop = asyncio.get_running_loop()

        def _do_request() -> dict[str, Any]:
            import json as _json

            req = urllib.request.Request(
                _COINGECKO_URL,
                headers={"Accept": "application/json", "User-Agent": "EcodiaOS/1.0"},
            )
            with urllib.request.urlopen(
                req, timeout=self._config.volatility_fetch_timeout_s
            ) as resp:
                body = resp.read()
            return _json.loads(body)

        data = await asyncio.wait_for(
            loop.run_in_executor(None, _do_request),
            timeout=self._config.volatility_fetch_timeout_s + 1.0,
        )

        btc = data.get("bitcoin", {}).get("usd_24h_change", 0.0) or 0.0
        eth = data.get("ethereum", {}).get("usd_24h_change", 0.0) or 0.0
        return {"btc_24h": float(btc), "eth_24h": float(eth)}

    async def _fetch_github(self) -> int:
        """
        Count open issues + PRs in the ecodiaos org via GitHub search API.
        Returns an integer count on success.
        Raises on any error — caller handles.

        Note: unauthenticated GitHub search is rate-limited to 10 req/min.
        At our default 5-minute poll interval this is well within budget.
        """
        import json as _json
        import urllib.request

        loop = asyncio.get_running_loop()

        def _do_request() -> int:
            req = urllib.request.Request(
                _GITHUB_SEARCH_URL,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "EcodiaOS/1.0",
                },
            )
            with urllib.request.urlopen(
                req, timeout=self._config.volatility_fetch_timeout_s
            ) as resp:
                body = resp.read()
            parsed = _json.loads(body)
            return int(parsed.get("total_count", 0))

        return await asyncio.wait_for(
            loop.run_in_executor(None, _do_request),
            timeout=self._config.volatility_fetch_timeout_s + 1.0,
        )

    # ─── Normalisation ───────────────────────────────────────────────────────

    def _normalise(self, raw: RawVolatilityData) -> float:
        """
        Convert raw metrics to a single [0, 1] stress float.

        economic_stress = sigmoid( mean(|btc_pct|, |eth_pct|) )
        github_stress   = min( open_count / 200, 1.0 )
        blended         = 0.7 * economic_stress + 0.3 * github_stress
        """
        # Economic component
        mean_abs_pct = (abs(raw.btc_24h_pct) + abs(raw.eth_24h_pct)) / 2.0
        economic = _sigmoid_normalise(mean_abs_pct)

        # GitHub component (normalise against a "busy repo" baseline of 200 issues/PRs)
        github = min(raw.github_open_count / 200.0, 1.0)

        # Weighted blend — economic signal dominates, github is supporting evidence
        blended = 0.7 * economic + 0.3 * github
        return max(0.0, min(1.0, blended))

    def _apply_ema(self, value: float) -> float:
        """Apply exponential moving average to smooth out noise."""
        alpha = self._config.volatility_ema_alpha
        if not self._ema_initialised:
            self._ema_initialised = True
            return value
        return alpha * value + (1.0 - alpha) * self._stress

    # ─── Spike detection ─────────────────────────────────────────────────────

    async def _maybe_emit_spike(self, current: float, now: float) -> None:
        """
        Emit SOMA_STATE_SPIKE if stress has moved more than spike_threshold
        within the last hour's rolling window.

        Only fires when the event bus is wired; silently skips otherwise.
        """
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
        if delta <= self._config.volatility_spike_threshold:
            return

        # Spike detected — emit once per polling pass (not de-duplicated across polls
        # intentionally — if stress stays elevated the organism should keep knowing).
        self._log.info(
            "soma_state_spike_detected",
            delta=round(delta, 3),
            current_stress=round(current, 3),
            window_min=round(min(window_values), 3),
        )
        try:
            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.SOMA_STATE_SPIKE,
                    data={
                        "dimension": "stress",
                        "value": round(current, 4),
                        "delta_1h": round(delta, 4),
                    },
                    source_system="soma.volatility_sensor",
                )
            )
        except Exception as exc:
            # Event bus failures are non-fatal — the stress value is already injected.
            self._log.warning("spike_event_emit_failed", error=str(exc))
