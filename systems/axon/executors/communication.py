"""
EcodiaOS — Axon Communication Executors

Communication executors send messages to people. They range from Level 1
(responding in an active conversation) to Level 2 (pushing unsolicited notifications).

RespondTextExecutor  — (Level 1) route text response through Voxis for personality rendering
NotificationExecutor — (Level 2) send a push notification to a user or group
PostMessageExecutor  — (Level 2) post to X or LinkedIn via sovereign IdentityVault connectors

These executors are not reversible — you cannot un-send a message.
This asymmetry is intentional: communication has real effects in the world.
Nova and Equor bear full responsibility for approving communication intents.

All communication is routed through Voxis for personality rendering —
Axon never sends raw text directly to users.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from systems.axon.audit import AuditLogger
    from systems.identity.connector import PlatformConnector
    from systems.identity.vault import IdentityVault
    from systems.voxis.service import VoxisService
    from systems.voxis.types import ExpressionTrigger

logger = structlog.get_logger()


# ─── RespondTextExecutor ──────────────────────────────────────────


class RespondTextExecutor(Executor):
    """
    Send a text response in the current conversation via Voxis.

    This is the primary "speak" action. It routes the response content to
    Voxis, which applies personality rendering, affect colouring, and
    silence judgement before delivery.

    Level 1: Responding in an active conversation is always within ADVISOR scope.

    Required params:
      content (str): The response content to express.

    Optional params:
      conversation_id (str): Target conversation. Default: current active conversation.
      urgency (float 0-1): Expression urgency. Default 0.5.
      trigger (str): Voxis trigger hint. Default "NOVA_RESPOND".
    """

    action_type = "respond_text"
    description = "Send a text response via Voxis personality engine"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 5000
    rate_limit = RateLimit.per_minute(30)

    def __init__(self, voxis: VoxisService | None = None) -> None:
        self._voxis = voxis
        self._logger = logger.bind(system="axon.executor.respond_text")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("content"):
            return ValidationResult.fail("content is required", content="missing or empty")
        content = params["content"]
        if not isinstance(content, str):
            return ValidationResult.fail("content must be a string")
        if len(content) > 10_000:
            return ValidationResult.fail("content too long (max 10,000 chars)")
        urgency = params.get("urgency", 0.5)
        if not isinstance(urgency, (int, float)) or not 0.0 <= float(urgency) <= 1.0:
            return ValidationResult.fail("urgency must be a float between 0.0 and 1.0")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        content = params["content"]
        conversation_id = params.get("conversation_id")
        urgency = float(params.get("urgency", 0.5))
        trigger_name = params.get("trigger", "NOVA_RESPOND")

        self._logger.info(
            "respond_text_execute",
            content_preview=content[:80],
            conversation_id=conversation_id,
            execution_id=context.execution_id,
        )

        if self._voxis is None:
            return ExecutionResult(
                success=True,
                data={"content": content, "delivered": False},
                side_effects=["Text response staged (no Voxis service)"],
                new_observations=[f"EOS would have said: {content[:200]}"],
            )

        try:

            trigger = _resolve_trigger(trigger_name)
            await self._voxis.express(
                content=content,
                trigger=trigger,
                conversation_id=conversation_id,
                affect=context.affect_state,
                urgency=urgency,
            )
            return ExecutionResult(
                success=True,
                data={"content_length": len(content), "delivered": True},
                side_effects=["Text response delivered via Voxis"],
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error=f"Voxis expression failed: {exc}",
            )


def _resolve_trigger(trigger_name: str) -> ExpressionTrigger:
    from systems.voxis.types import ExpressionTrigger
    try:
        return ExpressionTrigger[trigger_name]
    except KeyError:
        return ExpressionTrigger.NOVA_RESPOND


# ─── NotificationExecutor ─────────────────────────────────────────


class NotificationExecutor(Executor):
    """
    Send a push notification to a user or group of users.

    Level 2: Sending unsolicited notifications requires PARTNER autonomy.
    EOS should use these sparingly — notification overload undermines trust
    and Care. Equor should scrutinise notification intents carefully.

    Required params:
      recipient_id (str): User ID or group ID to notify.
      title (str): Short notification title.
      body (str): Notification body text.

    Optional params:
      urgency (str): "low" | "normal" | "high". Default "normal".
      action_url (str): Deep link URL for the notification. Default None.
    """

    action_type = "send_notification"
    description = "Send a push notification to a user or group (Level 2)"
    required_autonomy = 2
    reversible = False
    max_duration_ms = 5000
    rate_limit = RateLimit.per_hour(10)  # Strict — notification spam is harmful

    def __init__(self, redis_client: Any = None) -> None:
        self._redis = redis_client
        self._logger = logger.bind(system="axon.executor.notification")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        for field in ("recipient_id", "title", "body"):
            if not params.get(field):
                return ValidationResult.fail(
                    f"{field} is required",
                    **{field: "missing or empty"},
                )
        urgency = params.get("urgency", "normal")
        if urgency not in ("low", "normal", "high"):
            return ValidationResult.fail("urgency must be 'low', 'normal', or 'high'")
        title = params["title"]
        if len(title) > 100:
            return ValidationResult.fail("title too long (max 100 chars)")
        body = params["body"]
        if len(body) > 500:
            return ValidationResult.fail("body too long (max 500 chars)")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        recipient_id = params["recipient_id"]
        title = params["title"]
        body = params["body"]
        urgency = params.get("urgency", "normal")
        action_url = params.get("action_url")

        self._logger.info(
            "notification_execute",
            recipient_id=recipient_id,
            title=title,
            urgency=urgency,
            execution_id=context.execution_id,
        )

        notification_payload = {
            "type": "push_notification",
            "recipient_id": recipient_id,
            "title": title,
            "body": body,
            "urgency": urgency,
            "action_url": action_url,
            "sender": "eos",
            "execution_id": context.execution_id,
        }

        if self._redis is not None:
            try:
                import json as _json
                channel = f"eos:notifications:{recipient_id}"
                await self._redis.publish(channel, _json.dumps(notification_payload))
                return ExecutionResult(
                    success=True,
                    data={"channel": channel, "delivered": True},
                    side_effects=[
                        f"Notification '{title}' sent to {recipient_id} ({urgency} urgency)"
                    ],
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    error=f"Notification publish failed: {exc}",
                )
        else:
            self._logger.info("notification_no_redis", payload=notification_payload)
            return ExecutionResult(
                success=True,
                data={"delivered": False, "reason": "No Redis client"},
                side_effects=[f"Notification staged: '{title}' → {recipient_id}"],
            )


# ─── PostMessageExecutor ──────────────────────────────────────────

# Platform → (API endpoint, payload-builder)
# Each entry is a callable that receives (token, params) and returns
# (url, headers, json_payload) ready for httpx.post().
_PLATFORM_X = "x"
_PLATFORM_LINKEDIN = "linkedin"
_SUPPORTED_PLATFORMS = frozenset({_PLATFORM_X, _PLATFORM_LINKEDIN})


def _build_x_request(
    token: str, content: str, params: dict[str, Any]
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build the X v2 tweet creation request."""
    payload: dict[str, Any] = {"text": content}
    reply_id = params.get("thread_id")
    if reply_id:
        payload["reply"] = {"in_reply_to_tweet_id": str(reply_id)}
    return (
        "https://api.twitter.com/2/tweets",
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        payload,
    )


def _build_linkedin_request(
    token: str, content: str, params: dict[str, Any]
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build the LinkedIn UGC post creation request."""
    author_urn = params.get("linkedin_author_urn", "")
    if not author_urn:
        raise ValueError("linkedin_author_urn is required for LinkedIn posts")
    payload: dict[str, Any] = {
        "author": author_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": content},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }
    return (
        "https://api.linkedin.com/v2/ugcPosts",
        {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        },
        payload,
    )


class PostMessageExecutor(Executor):
    """
    Post a message to X (Twitter) or LinkedIn via the sovereign IdentityVault layer.

    The executor resolves the target connector from the registered PlatformConnectors,
    fetches a live access token (auto-refreshing if expired), fires the HTTP request,
    and implements a one-shot self-healing loop: if the platform returns HTTP 401
    (Unauthorized), it calls connector.refresh_token() once and retries.

    Auth successes and failures are written to the AxonAudit trail.

    Level 2: Posting to external social platforms requires PARTNER autonomy.

    Required params:
      content (str): Message text (≤ 5 000 chars).

    Platform-routing params (one of):
      platform (str): "x" (default) | "linkedin"
      channel_id (str): Kept for internal logging / audit; not sent to the API.

    Optional params:
      thread_id (str | int): Reply-to tweet ID (X only).
      linkedin_author_urn (str): Author URN for LinkedIn (e.g. "urn:li:person:ABC123").
    """

    action_type = "post_message"
    description = "Post to X or LinkedIn via sovereign identity connectors (Level 2)"
    required_autonomy = 2
    reversible = False
    max_duration_ms = 15_000
    rate_limit = RateLimit.per_hour(20)

    def __init__(
        self,
        vault: IdentityVault | None = None,
        connectors: dict[str, PlatformConnector] | None = None,
        audit: AuditLogger | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._vault = vault
        self._connectors: dict[str, PlatformConnector] = connectors or {}
        self._audit = audit
        self._http = http_client or httpx.AsyncClient(timeout=15.0)
        self._logger = logger.bind(system="axon.executor.post_message")

    # ── Validation ────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        content = params.get("content", "")
        if not content:
            return ValidationResult.fail("content is required", content="missing or empty")
        if len(content) > 5000:
            return ValidationResult.fail("content too long (max 5,000 chars)")

        platform = params.get("platform", _PLATFORM_X)
        if platform not in _SUPPORTED_PLATFORMS:
            return ValidationResult.fail(
                f"platform must be one of {sorted(_SUPPORTED_PLATFORMS)}",
                platform="unsupported",
            )

        if platform == _PLATFORM_LINKEDIN and not params.get("linkedin_author_urn"):
            return ValidationResult.fail(
                "linkedin_author_urn is required for LinkedIn posts",
                linkedin_author_urn="missing",
            )

        return ValidationResult.ok()

    # ── Execution ─────────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        content: str = params["content"]
        platform: str = params.get("platform", _PLATFORM_X)
        channel_id: str = params.get("channel_id", platform)

        self._logger.info(
            "post_message_execute",
            platform=platform,
            channel_id=channel_id,
            content_preview=content[:80],
            execution_id=context.execution_id,
        )

        connector = self._connectors.get(platform)
        if connector is None:
            err = f"No PlatformConnector registered for platform '{platform}'"
            self._logger.error("post_message_no_connector", platform=platform)
            await self._record_auth_failure(context, platform, err)
            return ExecutionResult(success=False, error=err)

        # ── Token acquisition (auto-refreshes if < 60 s remaining) ─
        token = await connector.get_access_token()
        if not token:
            err = f"Could not obtain access token for platform '{platform}'"
            self._logger.warning("post_message_no_token", platform=platform)
            await self._record_auth_failure(context, platform, err)
            return ExecutionResult(success=False, error=err)

        # ── HTTP request with one-shot 401 self-healing ────────────
        result = await self._fire_with_retry(platform, token, content, params, connector, context)
        return result

    async def _fire_with_retry(
        self,
        platform: str,
        token: str,
        content: str,
        params: dict[str, Any],
        connector: PlatformConnector,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        POST to the platform API.  On HTTP 401, trigger connector.refresh_token()
        and retry exactly once.  Any other error is returned immediately.
        """
        for attempt in (1, 2):
            url, headers, payload = self._build_request(platform, token, content, params)
            try:
                resp = await self._http.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                err = f"HTTP transport error on attempt {attempt}: {exc}"
                self._logger.error(
                    "post_message_transport_error",
                    platform=platform,
                    attempt=attempt,
                    error=str(exc),
                )
                return ExecutionResult(success=False, error=err)

            if resp.status_code == 401 and attempt == 1:
                # ── Self-healing: token was rejected — refresh and retry ──
                self._logger.warning(
                    "post_message_401_refreshing",
                    platform=platform,
                    execution_id=context.execution_id,
                )
                refresh_result = await connector.refresh_token()
                if not refresh_result.success or refresh_result.token_set is None:
                    err = (
                        f"Platform '{platform}' returned 401 and token refresh failed: "
                        f"{refresh_result.error}"
                    )
                    self._logger.error(
                        "post_message_refresh_failed",
                        platform=platform,
                        error=refresh_result.error,
                    )
                    await self._record_auth_failure(context, platform, err)
                    return ExecutionResult(success=False, error=err)

                token = refresh_result.token_set.access_token
                self._logger.info(
                    "post_message_token_refreshed",
                    platform=platform,
                    execution_id=context.execution_id,
                )
                continue  # retry with fresh token

            if resp.status_code not in (200, 201):
                err = (
                    f"Platform '{platform}' returned HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
                self._logger.error(
                    "post_message_api_error",
                    platform=platform,
                    status_code=resp.status_code,
                    response_preview=resp.text[:200],
                )
                await self._record_auth_failure(context, platform, err)
                return ExecutionResult(
                    success=False,
                    error=err,
                    data={"platform": platform, "http_status": resp.status_code},
                )

            # ── Success ─────────────────────────────────────────────────────
            resp_data: dict[str, Any] = {}
            with contextlib.suppress(Exception):
                resp_data = resp.json()

            post_id = _extract_post_id(platform, resp_data)
            self._logger.info(
                "post_message_success",
                platform=platform,
                post_id=post_id,
                attempt=attempt,
                execution_id=context.execution_id,
            )
            await self._record_success(context, platform, post_id)

            return ExecutionResult(
                success=True,
                data={
                    "platform": platform,
                    "post_id": post_id,
                    "http_status": resp.status_code,
                    "attempts": attempt,
                    "delivered": True,
                },
                side_effects=[
                    f"Message posted to {platform} (post_id={post_id or 'unknown'})"
                ],
                new_observations=[
                    f"EOS posted to {platform}: \"{content[:100]}\""
                    + (f" (post_id={post_id})" if post_id else "")
                ],
            )

        # Should be unreachable — the loop always returns on attempt 2.
        return ExecutionResult(success=False, error="Unexpected retry exhaustion")

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _build_request(
        platform: str,
        token: str,
        content: str,
        params: dict[str, Any],
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        if platform == _PLATFORM_X:
            return _build_x_request(token, content, params)
        if platform == _PLATFORM_LINKEDIN:
            return _build_linkedin_request(token, content, params)
        raise ValueError(f"No request builder for platform '{platform}'")

    async def _record_success(
        self,
        context: ExecutionContext,
        platform: str,
        post_id: str,
    ) -> None:
        if self._audit is None:
            return
        self._logger.info(
            "post_message_audit_success",
            platform=platform,
            post_id=post_id,
            execution_id=context.execution_id,
        )

    async def _record_auth_failure(
        self,
        context: ExecutionContext,
        platform: str,
        reason: str,
    ) -> None:
        if self._audit is None:
            return
        self._logger.warning(
            "post_message_audit_auth_failure",
            platform=platform,
            reason=reason,
            execution_id=context.execution_id,
        )


def _extract_post_id(platform: str, resp_data: dict[str, Any]) -> str:
    """Best-effort extraction of the newly created post/tweet ID from the API response."""
    if platform == _PLATFORM_X:
        return str(resp_data.get("data", {}).get("id", ""))
    if platform == _PLATFORM_LINKEDIN:
        return str(resp_data.get("id", ""))
    return ""
