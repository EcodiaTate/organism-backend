"""
EcodiaOS - Axon Integration Executors

Integration executors make calls to external systems. They are Level 2-3
because they cross the boundary between EOS and the outside world.

APICallExecutor   - (Level 2) call an external REST API
WebhookExecutor   - (Level 2) trigger a webhook endpoint

These are deliberately generic - specific integrations (GitHub, Slack, IoT sensors)
will be implemented as purpose-built executors in future phases. For now, these
provide a general-purpose HTTP capability with safety constraints.

Safety constraints:
  - All URLs are validated (no local network, no internal services)
  - All responses are capped at 50KB
  - Credentials come from CredentialStore only - no inline secrets
  - All calls are logged with full audit trail
  - Timeouts are strictly enforced (5s default, max 30s)
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

logger = structlog.get_logger()

# Blocked URL patterns - EOS should never call internal network addresses
_BLOCKED_URL_PATTERNS = re.compile(
    r"^https?://(localhost|127\.|0\.0\.0\.0|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.)",
    re.IGNORECASE,
)

_MAX_RESPONSE_BYTES = 50 * 1024  # 50KB cap on API responses
_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def _validate_url(url: str) -> str | None:
    """
    Validate an API URL. Returns error message if invalid, None if valid.
    """
    if not url:
        return "url is required"
    try:
        parsed = urlparse(url)
    except Exception:
        return "url is malformed"
    if parsed.scheme not in ("http", "https"):
        return "url must use http or https"
    if _BLOCKED_URL_PATTERNS.match(url):
        return "url targets a blocked internal network address"
    if not parsed.netloc:
        return "url must include a host"
    return None


# ─── APICallExecutor ──────────────────────────────────────────────


class APICallExecutor(Executor):
    """
    Call an external REST API and return the response.

    The response is returned in data and also fed back as a new observation
    for the workspace (Atune will score its salience next cycle).

    Required params:
      url (str): The API endpoint URL.
      method (str): HTTP method - "GET" | "POST" | "PUT" | "PATCH" | "DELETE".

    Optional params:
      body (dict | str): Request body. Default None (no body).
      headers (dict): Additional request headers. Default {}.
      timeout_s (int): Request timeout in seconds. Default 5, max 30.
      credential_service (str): CredentialStore service key for auth. Default None.
      response_field (str): Extract a specific field from JSON response. Default None.
    """

    action_type = "call_api"
    description = "Call an external REST API (Level 2)"
    required_autonomy = 2
    reversible = False  # HTTP calls can't be generically undone
    max_duration_ms = 30_000
    rate_limit = RateLimit.per_minute(30)

    def __init__(self) -> None:
        self._logger = logger.bind(system="axon.executor.call_api")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        url_error = _validate_url(params.get("url", ""))
        if url_error:
            return ValidationResult.fail(url_error, url=url_error)

        method = str(params.get("method", "")).upper()
        if method not in _ALLOWED_METHODS:
            return ValidationResult.fail(
                f"method must be one of {sorted(_ALLOWED_METHODS)}",
                method="invalid value",
            )

        timeout_s = params.get("timeout_s", 5)
        if not isinstance(timeout_s, (int, float)) or not 0 < float(timeout_s) <= 30:
            return ValidationResult.fail("timeout_s must be between 1 and 30")

        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        url = params["url"]
        method = str(params.get("method", "GET")).upper()
        body = params.get("body")
        extra_headers = dict(params.get("headers", {}))
        timeout_s = float(params.get("timeout_s", 5))
        credential_service = params.get("credential_service")
        response_field = params.get("response_field")

        # Build headers
        headers: dict[str, str] = {
            "User-Agent": "EcodiaOS/1.0 (+https://ecodiaos.org)",
            **extra_headers,
        }

        # Inject auth if credential service specified
        if credential_service:
            token = context.credentials.get(credential_service)
            if token:
                # Extract raw credential from scoped token
                # In Phase 1, credential = raw API key
                # The token format is: expiry:signature:raw_credential
                parts = token.split(":", 2)
                if len(parts) == 3:
                    raw_credential = parts[2]
                    headers["Authorization"] = f"Bearer {raw_credential}"

        self._logger.info(
            "api_call_execute",
            url=url,
            method=method,
            execution_id=context.execution_id,
        )

        try:
            import asyncio
            try:
                import aiohttp  # noqa: F401
                result = await _call_with_aiohttp(
                    url=url,
                    method=method,
                    body=body,
                    headers=headers,
                    timeout_s=timeout_s,
                    response_field=response_field,
                )
            except ImportError:
                # aiohttp not available - use urllib as fallback
                result = await asyncio.to_thread(
                    _call_with_urllib,
                    url=url,
                    method=method,
                    body=body,
                    headers=headers,
                    timeout_s=timeout_s,
                )

            observation = _summarise_response(result, url, method)
            return ExecutionResult(
                success=result.get("status_code", 500) < 400,
                data=result,
                error="" if result.get("status_code", 0) < 400 else (
                    f"HTTP {result.get('status_code')}: {str(result.get('body', ''))[:200]}"
                ),
                side_effects=[f"API call {method} {url} → HTTP {result.get('status_code', '?')}"],
                new_observations=[observation],
            )

        except Exception as exc:
            return ExecutionResult(
                success=False,
                error=f"API call failed: {exc}",
                side_effects=[f"API call {method} {url} failed: {type(exc).__name__}"],
            )


async def _call_with_aiohttp(
    url: str,
    method: str,
    body: Any,
    headers: dict[str, Any],
    timeout_s: float,
    response_field: str | None,
) -> dict[str, Any]:
    import json as _json

    import aiohttp

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        kwargs: dict[str, Any] = {"headers": headers}
        if body is not None:
            if isinstance(body, dict):
                kwargs["json"] = body
            else:
                kwargs["data"] = str(body)

        async with session.request(method, url, **kwargs) as resp:
            raw = await resp.read()
            if len(raw) > _MAX_RESPONSE_BYTES:
                raw = raw[:_MAX_RESPONSE_BYTES]

            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type:
                try:
                    parsed = _json.loads(raw)
                    if response_field and isinstance(parsed, dict):
                        body_data = parsed.get(response_field, parsed)
                    else:
                        body_data = parsed
                except Exception:
                    body_data = raw.decode("utf-8", errors="replace")
            else:
                body_data = raw.decode("utf-8", errors="replace")

            return {
                "status_code": resp.status,
                "body": body_data,
                "content_type": content_type,
                "url": str(resp.url),
            }


def _call_with_urllib(
    url: str,
    method: str,
    body: Any,
    headers: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    import json as _json
    import urllib.error
    import urllib.request

    data = None
    if body is not None:
        if isinstance(body, dict):
            data = _json.dumps(body).encode()
            headers.setdefault("Content-Type", "application/json")
        else:
            data = str(body).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read(_MAX_RESPONSE_BYTES)
            content_type = resp.headers.get("Content-Type", "")
            try:
                body_data = _json.loads(raw)
            except Exception:
                body_data = raw.decode("utf-8", errors="replace")
            return {"status_code": resp.status, "body": body_data, "content_type": content_type, "url": url}
    except urllib.error.HTTPError as e:
        return {"status_code": e.code, "body": str(e.reason), "content_type": "", "url": url}


def _summarise_response(result: dict[str, Any], url: str, method: str) -> str:
    status = result.get("status_code", "?")
    body = result.get("body", "")
    if isinstance(body, dict):
        import json as _json
        body_preview = _json.dumps(body)[:300]
    else:
        body_preview = str(body)[:300]
    return f"API {method} {url} → {status}: {body_preview}"


# ─── WebhookExecutor ──────────────────────────────────────────────


class WebhookExecutor(Executor):
    """
    Trigger a registered webhook endpoint.

    Webhooks are outbound HTTP calls to registered external services -
    GitHub Actions, Zapier, community platform integrations, etc.
    They differ from APICallExecutor in that they use pre-configured
    endpoints from the credential store rather than arbitrary URLs.

    Required params:
      webhook_key (str): Key identifying the webhook in CredentialStore.
      payload (dict): Data to send as the webhook body.

    Optional params:
      event_type (str): Event type header (X-EOS-Event). Default "eos.action".
      timeout_s (int): Request timeout. Default 10.
    """

    action_type = "webhook_trigger"
    description = "Trigger a registered webhook endpoint (Level 2)"
    required_autonomy = 2
    reversible = False
    max_duration_ms = 15_000
    rate_limit = RateLimit.per_minute(10)

    def __init__(self) -> None:
        self._logger = logger.bind(system="axon.executor.webhook")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("webhook_key"):
            return ValidationResult.fail("webhook_key is required")
        if not params.get("payload"):
            return ValidationResult.fail("payload is required and must be non-empty dict")
        if not isinstance(params["payload"], dict):
            return ValidationResult.fail("payload must be a dict")
        timeout_s = params.get("timeout_s", 10)
        if not isinstance(timeout_s, (int, float)) or not 0 < float(timeout_s) <= 30:
            return ValidationResult.fail("timeout_s must be between 1 and 30")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        import hashlib
        import hmac as _hmac
        import json as _json

        webhook_key = params["webhook_key"]
        payload = dict(params["payload"])
        event_type = str(params.get("event_type", "eos.action"))
        timeout_s = float(params.get("timeout_s", 10))

        # Get webhook URL and secret from credentials
        token = context.credentials.get(webhook_key)
        if not token:
            return ExecutionResult(
                success=False,
                error=f"No credential configured for webhook_key '{webhook_key}'",
            )

        # Token format: expiry:signature:url|secret (pipe-separated)
        parts = token.split(":", 2)
        if len(parts) != 3:
            return ExecutionResult(
                success=False,
                error="Webhook credential format invalid",
            )
        raw_credential = parts[2]
        # Credential is "url|secret" or just "url"
        if "|" in raw_credential:
            webhook_url, webhook_secret = raw_credential.split("|", 1)
        else:
            webhook_url = raw_credential
            webhook_secret = ""

        url_error = _validate_url(webhook_url)
        if url_error:
            return ExecutionResult(
                success=False,
                error=f"Webhook URL invalid: {url_error}",
            )

        # Build signed payload
        body = _json.dumps(payload).encode()
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-EOS-Event": event_type,
            "X-EOS-Delivery": context.execution_id,
            "User-Agent": "EcodiaOS/1.0",
        }
        if webhook_secret:
            signature = _hmac.new(
                webhook_secret.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()
            headers["X-EOS-Signature-256"] = f"sha256={signature}"

        self._logger.info(
            "webhook_execute",
            webhook_key=webhook_key,
            event_type=event_type,
            url=webhook_url,
            execution_id=context.execution_id,
        )

        try:
            result = await _call_with_aiohttp(
                url=webhook_url,
                method="POST",
                body=payload,
                headers=headers,
                timeout_s=timeout_s,
                response_field=None,
            )
            success = result.get("status_code", 500) < 400
            return ExecutionResult(
                success=success,
                data={"status_code": result.get("status_code"), "webhook_key": webhook_key},
                error="" if success else f"Webhook returned HTTP {result.get('status_code')}",
                side_effects=[
                    f"Webhook '{webhook_key}' triggered ({event_type}) → HTTP {result.get('status_code')}"
                ],
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error=f"Webhook call failed: {exc}",
            )
