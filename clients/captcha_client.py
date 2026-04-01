"""
EcodiaOS - CAPTCHA Solver Client

Async client for 2captcha and Anti-Captcha services. Used by AccountProvisioner
to autonomously complete CAPTCHA challenges during platform account creation.

Supported CAPTCHA types:
  - reCAPTCHA v2 (image checkbox)
  - reCAPTCHA v3 (score-based, invisible)
  - hCaptcha
  - Image-based text CAPTCHA

Pricing (approximate, varies by service):
  - reCAPTCHA v2/hCaptcha: ~$0.001–$0.003 per solve
  - reCAPTCHA v3: ~$0.002–$0.004 per solve
  - Image CAPTCHA: ~$0.0005–$0.001 per solve

All solve costs are logged to the Synapse bus for Oikos metabolic accounting.

Configuration:
  ECODIAOS_CAPTCHA__TWOCAPTCHA_API_KEY   - 2captcha API key
  ECODIAOS_CAPTCHA__ANTICAPTCHA_API_KEY  - Anti-Captcha API key
  ECODIAOS_CAPTCHA__PROVIDER             - "2captcha" (default) | "anticaptcha"
  ECODIAOS_CAPTCHA__POLLING_INTERVAL_S   - seconds between polls (default 5)
  ECODIAOS_CAPTCHA__MAX_WAIT_S           - max seconds to wait (default 120)
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from config import CaptchaConfig
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.captcha_client")

# ─── Provider base URLs ─────────────────────────────────────────────────────

_2CAPTCHA_API = "https://2captcha.com"
_ANTICAPTCHA_API = "https://api.anti-captcha.com"

# ─── Result model ───────────────────────────────────────────────────────────


class CaptchaSolveResult:
    """Result of a CAPTCHA solve operation."""

    def __init__(
        self,
        token: str,
        task_id: str,
        captcha_type: str,
        cost_usd: float,
        elapsed_s: float,
        provider: str,
    ) -> None:
        self.token = token
        self.task_id = task_id
        self.captcha_type = captcha_type
        self.cost_usd = cost_usd
        self.elapsed_s = elapsed_s
        self.provider = provider

    def __repr__(self) -> str:
        return (
            f"CaptchaSolveResult(type={self.captcha_type!r}, "
            f"task_id={self.task_id!r}, cost=${self.cost_usd:.4f}, "
            f"elapsed={self.elapsed_s:.1f}s)"
        )


# ─── CaptchaClient ──────────────────────────────────────────────────────────


class CaptchaClient:
    """
    Async CAPTCHA solving client supporting 2captcha and Anti-Captcha.

    Each solve call:
      1. Submits the task to the configured provider API
      2. Polls until solved or max_wait_s exceeded
      3. Returns the token + cost metadata
      4. Emits a DOMAIN_EPISODE_RECORDED event for Oikos cost tracking

    Thread-safety: NOT thread-safe. Single-threaded asyncio like all EOS clients.
    """

    def __init__(self, config: "CaptchaConfig") -> None:
        self._config = config
        self._event_bus: EventBus | None = None
        self._log = logger.bind(provider=config.provider)
        self._total_cost_usd: float = 0.0
        self._total_solves: int = 0

    def set_event_bus(self, event_bus: "EventBus") -> None:
        self._event_bus = event_bus

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def total_solves(self) -> int:
        return self._total_solves

    # ── Public solve methods ───────────────────────────────────────────────

    async def solve_recaptcha_v2(
        self,
        site_key: str,
        page_url: str,
        invisible: bool = False,
    ) -> str:
        """
        Solve a reCAPTCHA v2 challenge.

        Args:
            site_key: The data-sitekey from the page HTML.
            page_url: Full URL of the page hosting the CAPTCHA.
            invisible: True for invisible reCAPTCHA v2.

        Returns:
            g-recaptcha-response token to inject into the form.
        """
        result = await self._solve(
            captcha_type="recaptchav2",
            params={
                "googlekey": site_key,
                "pageurl": page_url,
                "invisible": 1 if invisible else 0,
            },
        )
        return result.token

    async def solve_recaptcha_v3(
        self,
        site_key: str,
        page_url: str,
        action: str = "verify",
        min_score: float = 0.3,
    ) -> str:
        """
        Solve a reCAPTCHA v3 challenge (score-based, invisible).

        Args:
            site_key: The reCAPTCHA site key.
            page_url: Full URL of the page.
            action: The action name used by the page (e.g. "login", "submit").
            min_score: Minimum acceptable score (0.3–0.9).

        Returns:
            g-recaptcha-response token.
        """
        result = await self._solve(
            captcha_type="recaptchav3",
            params={
                "googlekey": site_key,
                "pageurl": page_url,
                "action": action,
                "min_score": min_score,
            },
        )
        return result.token

    async def solve_hcaptcha(
        self,
        site_key: str,
        page_url: str,
    ) -> str:
        """
        Solve an hCaptcha challenge.

        Args:
            site_key: The data-sitekey attribute value.
            page_url: Full URL of the page hosting the CAPTCHA.

        Returns:
            h-captcha-response token.
        """
        result = await self._solve(
            captcha_type="hcaptcha",
            params={
                "sitekey": site_key,
                "pageurl": page_url,
            },
        )
        return result.token

    async def solve_image_captcha(self, image_base64: str) -> str:
        """
        Solve a text-based image CAPTCHA.

        Args:
            image_base64: Base64-encoded PNG/JPG of the CAPTCHA image.

        Returns:
            Extracted text from the CAPTCHA image.
        """
        result = await self._solve(
            captcha_type="image",
            params={"body": image_base64},
        )
        return result.token

    async def get_balance(self) -> float:
        """
        Query remaining credit balance from the configured provider.

        Returns:
            Balance in USD. Returns -1.0 if the query fails.
        """
        try:
            if self._config.provider == "anticaptcha":
                return await self._anticaptcha_get_balance()
            return await self._2captcha_get_balance()
        except Exception as exc:
            self._log.warning("captcha_balance_query_failed", error=str(exc))
            return -1.0

    # ── Internal dispatch ──────────────────────────────────────────────────

    async def _solve(
        self,
        captcha_type: str,
        params: dict[str, Any],
    ) -> CaptchaSolveResult:
        """
        Submit a CAPTCHA task and poll until solved.

        Raises:
            RuntimeError: If the provider is not configured, solve times out,
                          or the provider returns an error.
        """
        if not self._config.enabled:
            raise RuntimeError(
                "No CAPTCHA provider configured. "
                "Set ECODIAOS_CAPTCHA__TWOCAPTCHA_API_KEY or "
                "ECODIAOS_CAPTCHA__ANTICAPTCHA_API_KEY."
            )

        t0 = time.monotonic()
        self._log.info("captcha_solve_start", captcha_type=captcha_type)

        if self._config.provider == "anticaptcha":
            result = await self._anticaptcha_solve(captcha_type, params, t0)
        else:
            result = await self._2captcha_solve(captcha_type, params, t0)

        self._total_cost_usd += result.cost_usd
        self._total_solves += 1

        self._log.info(
            "captcha_solved",
            captcha_type=captcha_type,
            task_id=result.task_id,
            cost_usd=result.cost_usd,
            elapsed_s=result.elapsed_s,
        )

        await self._emit_cost_event(result)
        return result

    # ── 2captcha backend ──────────────────────────────────────────────────

    async def _2captcha_solve(
        self,
        captcha_type: str,
        params: dict[str, Any],
        t0: float,
    ) -> CaptchaSolveResult:
        """Submit task to 2captcha and poll for result."""
        api_key = self._config.twocaptcha_api_key
        if not api_key:
            raise RuntimeError("2captcha API key not configured")

        # Build submission payload
        submit_data: dict[str, Any] = {"key": api_key, "json": 1, **params}

        # Map captcha_type → 2captcha method name
        method_map = {
            "recaptchav2": "userrecaptcha",
            "recaptchav3": "userrecaptcha",
            "hcaptcha": "hcaptcha",
            "image": "base64",
        }
        submit_data["method"] = method_map.get(captcha_type, captcha_type)
        if captcha_type == "recaptchav3":
            submit_data["version"] = "v3"

        # Submit
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{_2CAPTCHA_API}/in.php", data=submit_data)
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != 1:
            raise RuntimeError(f"2captcha submission failed: {data.get('request')}")

        task_id = str(data["request"])
        self._log.debug("2captcha_task_submitted", task_id=task_id)

        # Poll
        token = await self._2captcha_poll(task_id, api_key, t0)
        elapsed = time.monotonic() - t0
        cost = self._estimate_cost(captcha_type)

        return CaptchaSolveResult(
            token=token,
            task_id=task_id,
            captcha_type=captcha_type,
            cost_usd=cost,
            elapsed_s=elapsed,
            provider="2captcha",
        )

    async def _2captcha_poll(self, task_id: str, api_key: str, t0: float) -> str:
        """Poll 2captcha until the task is solved or timeout."""
        deadline = t0 + self._config.max_wait_s
        interval = self._config.polling_interval_s

        # Initial wait before first poll (2captcha recommends 5s minimum)
        await asyncio.sleep(max(5.0, interval))

        async with httpx.AsyncClient(timeout=15.0) as client:
            while time.monotonic() < deadline:
                resp = await client.get(
                    f"{_2CAPTCHA_API}/res.php",
                    params={"key": api_key, "action": "get", "id": task_id, "json": 1},
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("status") == 1:
                    return str(data["request"])

                err = data.get("request", "")
                if err != "CAPCHA_NOT_READY":
                    raise RuntimeError(f"2captcha error: {err}")

                await asyncio.sleep(interval)

        elapsed = time.monotonic() - t0
        raise RuntimeError(
            f"2captcha timed out after {elapsed:.0f}s (task_id={task_id})"
        )

    async def _2captcha_get_balance(self) -> float:
        """Query 2captcha account balance."""
        api_key = self._config.twocaptcha_api_key
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_2CAPTCHA_API}/res.php",
                params={"key": api_key, "action": "getbalance", "json": 1},
            )
            resp.raise_for_status()
            data = resp.json()
        if data.get("status") == 1:
            return float(data["request"])
        raise RuntimeError(f"2captcha balance error: {data.get('request')}")

    # ── Anti-Captcha backend ───────────────────────────────────────────────

    async def _anticaptcha_solve(
        self,
        captcha_type: str,
        params: dict[str, Any],
        t0: float,
    ) -> CaptchaSolveResult:
        """Submit task to Anti-Captcha and poll for result."""
        api_key = self._config.anticaptcha_api_key
        if not api_key:
            raise RuntimeError("Anti-Captcha API key not configured")

        # Build task payload
        task = self._build_anticaptcha_task(captcha_type, params)
        submit_payload = {"clientKey": api_key, "task": task}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_ANTICAPTCHA_API}/createTask",
                json=submit_payload,
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("errorId") != 0:
            raise RuntimeError(
                f"Anti-Captcha task creation failed: {data.get('errorDescription')}"
            )

        task_id = str(data["taskId"])
        self._log.debug("anticaptcha_task_submitted", task_id=task_id)

        token = await self._anticaptcha_poll(task_id, api_key, t0)
        elapsed = time.monotonic() - t0
        cost = self._estimate_cost(captcha_type)

        return CaptchaSolveResult(
            token=token,
            task_id=task_id,
            captcha_type=captcha_type,
            cost_usd=cost,
            elapsed_s=elapsed,
            provider="anticaptcha",
        )

    def _build_anticaptcha_task(
        self, captcha_type: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Build Anti-Captcha task object from generic params."""
        if captcha_type == "recaptchav2":
            return {
                "type": "NoCaptchaTaskProxyless",
                "websiteURL": params["pageurl"],
                "websiteKey": params["googlekey"],
                "isInvisible": bool(params.get("invisible", 0)),
            }
        if captcha_type == "recaptchav3":
            return {
                "type": "RecaptchaV3TaskProxyless",
                "websiteURL": params["pageurl"],
                "websiteKey": params["googlekey"],
                "pageAction": params.get("action", "verify"),
                "minScore": params.get("min_score", 0.3),
            }
        if captcha_type == "hcaptcha":
            return {
                "type": "HCaptchaTaskProxyless",
                "websiteURL": params["pageurl"],
                "websiteKey": params["sitekey"],
            }
        if captcha_type == "image":
            return {
                "type": "ImageToTextTask",
                "body": params["body"],
            }
        raise ValueError(f"Unknown captcha_type for anticaptcha: {captcha_type!r}")

    async def _anticaptcha_poll(
        self, task_id: str, api_key: str, t0: float
    ) -> str:
        """Poll Anti-Captcha until task is solved or timeout."""
        deadline = t0 + self._config.max_wait_s
        interval = self._config.polling_interval_s

        await asyncio.sleep(max(5.0, interval))

        async with httpx.AsyncClient(timeout=15.0) as client:
            while time.monotonic() < deadline:
                resp = await client.post(
                    f"{_ANTICAPTCHA_API}/getTaskResult",
                    json={"clientKey": api_key, "taskId": int(task_id)},
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("errorId") != 0:
                    raise RuntimeError(
                        f"Anti-Captcha error: {data.get('errorDescription')}"
                    )

                if data.get("status") == "ready":
                    sol = data.get("solution", {})
                    # Return the appropriate token field
                    return (
                        sol.get("gRecaptchaResponse")
                        or sol.get("text")
                        or sol.get("token")
                        or ""
                    )

                await asyncio.sleep(interval)

        elapsed = time.monotonic() - t0
        raise RuntimeError(
            f"Anti-Captcha timed out after {elapsed:.0f}s (task_id={task_id})"
        )

    async def _anticaptcha_get_balance(self) -> float:
        """Query Anti-Captcha account balance."""
        api_key = self._config.anticaptcha_api_key
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_ANTICAPTCHA_API}/getBalance",
                json={"clientKey": api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        if data.get("errorId") == 0:
            return float(data["balance"])
        raise RuntimeError(f"Anti-Captcha balance error: {data.get('errorDescription')}")

    # ── Cost estimation ────────────────────────────────────────────────────

    def _estimate_cost(self, captcha_type: str) -> float:
        """Return estimated cost in USD for a CAPTCHA solve by type."""
        cost_map = {
            "recaptchav2": 0.002,
            "recaptchav3": 0.003,
            "hcaptcha": 0.002,
            "image": 0.001,
        }
        return cost_map.get(captcha_type, 0.002)

    # ── Synapse emission ───────────────────────────────────────────────────

    async def _emit_cost_event(self, result: CaptchaSolveResult) -> None:
        """Emit DOMAIN_EPISODE_RECORDED for Oikos cost accounting."""
        if self._event_bus is None:
            return

        from primitives.common import utc_now
        from systems.synapse.types import SynapseEvent, SynapseEventType

        try:
            event = SynapseEvent(
                event_type=SynapseEventType.DOMAIN_EPISODE_RECORDED,
                source_system="identity.captcha_client",
                data={
                    "domain": "account_provisioning",
                    "outcome": "success",
                    "revenue": "0",
                    "cost_usd": str(round(result.cost_usd, 6)),
                    "duration_ms": int(result.elapsed_s * 1000),
                    "custom_metrics": {
                        "captcha_type": result.captcha_type,
                        "provider": result.provider,
                        "task_id": result.task_id,
                    },
                    "timestamp": utc_now().isoformat(),
                    "source_system": "identity.captcha_client",
                },
            )
            await self._event_bus.emit(event)
        except Exception as exc:
            self._log.warning("captcha_cost_event_failed", error=str(exc))
