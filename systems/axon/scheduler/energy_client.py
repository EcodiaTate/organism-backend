"""
EcodiaOS - Energy Grid API Client

Provider-agnostic interface to real-time carbon intensity and energy cost APIs.
Ships with two concrete implementations:

  1. ElectricityMapsClient - uses the Electricity Maps v3 API (same as
     GridMetabolismSensor in soma/, but decoupled for modular swapability).
  2. WattTimeClient - uses the WattTime v3 API for MOER-based marginal
     emissions data.

Both conform to the ``EnergyProvider`` protocol so the scheduler interceptor
never couples to a specific upstream.

Iron Rules (inherited from GridMetabolismSensor contract):
  - All HTTP calls are bounded by a configurable timeout + hard outer cap.
  - Any exception is caught and returns None - callers degrade gracefully.
  - No LLM calls, no DB writes, no side-effects beyond the HTTP fetch.
  - Uses stdlib urllib so there are no extra dependencies.
"""

from __future__ import annotations

import abc
import asyncio
import json as _json
import urllib.error
import urllib.request
from typing import Any

import structlog

from systems.axon.scheduler.types import GridReading

logger = structlog.get_logger("systems.axon.scheduler.energy_client")


# ─── Provider Protocol ────────────────────────────────────────────


class EnergyProvider(abc.ABC):
    """
    Abstract interface for grid energy data providers.

    Implement ``fetch()`` to return a ``GridReading`` or ``None`` on failure.
    The scheduler caching layer calls this; implementations must be stateless
    and safe to call from any asyncio task.
    """

    provider_name: str = ""

    @abc.abstractmethod
    async def fetch(self) -> GridReading | None:
        """
        Fetch the latest carbon intensity / energy cost reading.

        Returns None on any error (network, auth, parse). Callers must
        handle the None case gracefully.
        """
        ...


# ─── Electricity Maps ────────────────────────────────────────────


_ELECTRICITY_MAPS_URL = (
    "https://api.electricitymap.org/v3/carbon-intensity/latest"
    "?lat={lat}&lon={lon}"
)


class ElectricityMapsClient(EnergyProvider):
    """
    Electricity Maps v3 API client.

    API: GET /v3/carbon-intensity/latest?lat=...&lon=...
    Auth: ``auth-token`` header.
    Response: ``{"carbonIntensity": 123.4, "zone": "AU-NSW", ...}``
    """

    provider_name = "electricity_maps"

    def __init__(
        self,
        api_key: str,
        latitude: float,
        longitude: float,
        timeout_s: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._lat = latitude
        self._lon = longitude
        self._timeout_s = timeout_s
        self._log = logger.bind(provider="electricity_maps")

    async def fetch(self) -> GridReading | None:
        url = _ELECTRICITY_MAPS_URL.format(lat=self._lat, lon=self._lon)
        data = await _http_get_json(
            url=url,
            headers={"auth-token": self._api_key, "Accept": "application/json"},
            timeout_s=self._timeout_s,
            log=self._log,
        )
        if data is None:
            return None

        raw_carbon = data.get("carbonIntensity")
        if raw_carbon is None:
            self._log.warning("missing_carbon_intensity", keys=list(data.keys()))
            return None

        return GridReading(
            carbon_intensity_g=float(raw_carbon),
            zone=data.get("zone", ""),
            provider=self.provider_name,
            raw=data,
        )


# ─── WattTime ────────────────────────────────────────────────────


_WATTTIME_LOGIN_URL = "https://api.watttime.org/login"
_WATTTIME_INDEX_URL = "https://api.watttime.org/v3/signal-index"


class WattTimeClient(EnergyProvider):
    """
    WattTime v3 API client.

    Auth: HTTP Basic → bearer token (cached for 30 min by the energy cache).
    Signal: ``/v3/signal-index`` returns a MOER (Marginal Operating Emissions
    Rate) value between 0–100 where higher = dirtier.

    We normalise the 0–100 MOER index to an approximate gCO2eq/kWh by
    linear interpolation: 0 → 50g, 100 → 800g. This is a rough heuristic;
    for precise values, use Electricity Maps.
    """

    provider_name = "watttime"

    # Linear mapping: MOER 0 → 50 gCO2/kWh, MOER 100 → 800 gCO2/kWh
    _MOER_MIN_G: float = 50.0
    _MOER_MAX_G: float = 800.0

    def __init__(
        self,
        username: str,
        password: str,
        latitude: float,
        longitude: float,
        timeout_s: float = 10.0,
    ) -> None:
        self._username = username
        self._password = password
        self._lat = latitude
        self._lon = longitude
        self._timeout_s = timeout_s
        self._bearer_token: str | None = None
        self._log = logger.bind(provider="watttime")

    async def fetch(self) -> GridReading | None:
        # Step 1: Authenticate (if no cached token)
        if self._bearer_token is None:
            self._bearer_token = await self._login()
            if self._bearer_token is None:
                return None

        # Step 2: Fetch signal index
        data = await _http_get_json(
            url=_WATTTIME_INDEX_URL,
            headers={
                "Authorization": f"Bearer {self._bearer_token}",
                "Accept": "application/json",
            },
            params={"latitude": str(self._lat), "longitude": str(self._lon)},
            timeout_s=self._timeout_s,
            log=self._log,
        )

        if data is None:
            # Token may have expired - clear so next fetch re-authenticates
            self._bearer_token = None
            return None

        # Extract MOER value from the response
        moer = self._extract_moer(data)
        if moer is None:
            return None

        carbon_g = self._moer_to_carbon_g(moer)

        return GridReading(
            carbon_intensity_g=carbon_g,
            zone=data.get("meta", {}).get("region", ""),
            provider=self.provider_name,
            raw=data,
        )

    async def _login(self) -> str | None:
        """Authenticate with WattTime and return a bearer token."""
        import base64

        credentials = base64.b64encode(
            f"{self._username}:{self._password}".encode()
        ).decode()

        data = await _http_get_json(
            url=_WATTTIME_LOGIN_URL,
            headers={"Authorization": f"Basic {credentials}"},
            timeout_s=self._timeout_s,
            log=self._log,
        )
        if data is None:
            return None

        token = data.get("token")
        if not token:
            self._log.warning("watttime_login_no_token", keys=list(data.keys()))
            return None

        self._log.debug("watttime_authenticated")
        return str(token)

    def _extract_moer(self, data: dict[str, Any]) -> float | None:
        """Extract the MOER value from WattTime v3 signal-index response."""
        # v3 response: {"data": [{"value": 42, ...}], "meta": {...}}
        entries = data.get("data", [])
        if not entries:
            self._log.warning("watttime_no_data_entries")
            return None
        value = entries[0].get("value")
        if value is None:
            self._log.warning("watttime_no_value_in_entry")
            return None
        return float(value)

    def _moer_to_carbon_g(self, moer: float) -> float:
        """Convert MOER index (0–100) to approximate gCO2eq/kWh."""
        clamped = max(0.0, min(100.0, moer))
        fraction = clamped / 100.0
        return self._MOER_MIN_G + fraction * (self._MOER_MAX_G - self._MOER_MIN_G)


# ─── HTTP Helper ─────────────────────────────────────────────────


async def _http_get_json(
    url: str,
    headers: dict[str, str],
    timeout_s: float,
    log: Any,
    params: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """
    stdlib-based async HTTP GET → JSON dict.

    Runs the blocking urllib call in an executor so it doesn't stall
    the event loop. Returns None on any error.
    """
    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{query}"

    headers.setdefault("User-Agent", "EcodiaOS/1.0")

    loop = asyncio.get_running_loop()

    def _do_request() -> dict[str, Any]:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
        return _json.loads(body)  # type: ignore[no-any-return]

    try:
        data: dict[str, Any] = await asyncio.wait_for(
            loop.run_in_executor(None, _do_request),
            timeout=timeout_s + 2.0,  # Outer cap
        )
        return data
    except urllib.error.HTTPError as exc:
        log.warning("http_error", status=exc.code, url=url[:120])
    except (TimeoutError, OSError) as exc:
        log.warning("fetch_error", error=str(exc))
    return None
