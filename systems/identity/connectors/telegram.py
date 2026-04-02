"""
EcodiaOS - Telegram Bot Connector (Spec 23, §14.x: Telegram Channel)

Telegram uses Bot Token authentication - there is no OAuth2 flow.
The bot token is provisioned once via @BotFather and never expires
(though it can be regenerated via revokeToken or revoked via logOut).

Differences from OAuth2 connectors:
  - No authorization URL or code exchange - token is injected at config time.
  - `authenticate()` validates the token via getMe and stores it in vault.
  - `refresh_token()` re-validates health (tokens don't expire).
  - `revoke()` calls the Telegram logOut API.
  - `check_health()` calls getMe - O(1), no side effects.

Bot token storage:
  vault.encrypt_token_json({"bot_token": token}, platform_id="telegram")

Environment variables:
  ORGANISM_CONNECTORS__TELEGRAM__BOT_TOKEN      - required
  ORGANISM_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID  - optional, for status broadcasts
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from primitives.common import utc_now
from systems.identity.connector import (
    AuthorizationRequest,
    AuthorizationResponse,
    ConnectorHealthReport,
    ConnectorStatus,
    OAuthTokenSet,
    PlatformAuthError,
    PlatformConnector,
    TokenExchangeRequest,
    TokenRefreshResult,
)

if TYPE_CHECKING:
    from systems.identity.connector import OAuthClientConfig
    from systems.identity.vault import IdentityVault

logger = structlog.get_logger("identity.telegram")

_TELEGRAM_API_BASE = "https://api.telegram.org"

# Sentinel OAuthClientConfig fields - Telegram doesn't use OAuth2 URLs,
# but we must still accept the base-class constructor signature.
_TELEGRAM_EMPTY_CONFIG_FIELDS = ("authorize_url", "token_url", "revoke_url")


def _bot_url(token: str, method: str) -> str:
    """Build a Telegram Bot API URL."""
    return f"{_TELEGRAM_API_BASE}/bot{token}/{method}"


class TelegramConnector(PlatformConnector):
    """
    Telegram Bot Token connector.

    Telegram bots authenticate via a static token issued by @BotFather.
    There is no OAuth2 flow - the token is validated once on `authenticate()`,
    stored encrypted in the IdentityVault, and re-validated on `check_health()`.

    Construction pattern:
        connector = TelegramConnector(
            client_config=OAuthClientConfig(
                client_id="",         # unused
                client_secret="",     # unused
                authorize_url="",
                token_url="",
                revoke_url="",
                redirect_uri="",
                scopes=[],
            ),
            vault=vault,
            bot_token=os.environ["ORGANISM_CONNECTORS__TELEGRAM__BOT_TOKEN"],
        )
        connector.set_event_bus(event_bus)
        await connector.authenticate()
    """

    def __init__(
        self,
        client_config: OAuthClientConfig,
        vault: IdentityVault,
        bot_token: str = "",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(client_config, vault)
        # Raw token injected at construction - stored to vault on authenticate().
        self._bot_token: str = bot_token.strip()
        self._http = http_client or httpx.AsyncClient(timeout=15.0)

    @property
    def platform_id(self) -> str:
        return "telegram"

    @property
    def display_name(self) -> str:
        return "Telegram Bot"

    # ─── Token resolution ────────────────────────────────────────

    def _resolve_token(self) -> str:
        """
        Resolve the bot token from vault (preferred) or constructor arg.

        After `authenticate()` the token lives in the vault; before that
        we fall back to `self._bot_token` so callers can validate eagerly.
        """
        if self._credentials and self._credentials.token_envelope_id:
            decrypted = self._decrypt_current_tokens()
            if decrypted and decrypted.access_token:
                return decrypted.access_token
        return self._bot_token

    # ─── Authenticate (replaces build_authorization_url + exchange_code) ──

    async def authenticate(self) -> bool:
        """
        Validate the bot token and store it in the vault.

        Calls GET /getMe - Telegram returns bot info on success (HTTP 200)
        or {"ok": false} on invalid token. Token is stored as the
        `access_token` field of an OAuthTokenSet so the base-class
        infrastructure (credential CRUD, event emissions, health tracking)
        works without modification.

        Returns True on success.
        """
        token = self._bot_token
        if not token:
            raise PlatformAuthError(
                "No bot token configured",
                platform_id=self.platform_id,
            )

        _t0 = self._start_timer()
        try:
            resp = await self._http.get(_bot_url(token, "getMe"))
        except httpx.HTTPError as exc:
            await self._emit_re_training_example("authenticate", "failure", _t0, type(exc).__name__)
            raise PlatformAuthError(
                f"Telegram getMe network error: {exc}",
                platform_id=self.platform_id,
            ) from exc

        if resp.status_code != 200:
            await self._emit_re_training_example(
                "authenticate", "failure", _t0, f"HTTP_{resp.status_code}"
            )
            raise PlatformAuthError(
                f"Telegram getMe failed: {resp.status_code}",
                platform_id=self.platform_id,
                http_status=resp.status_code,
                response_body=resp.text,
            )

        data: dict[str, Any] = resp.json()
        if not data.get("ok"):
            description = data.get("description", "unknown error")
            await self._emit_re_training_example("authenticate", "failure", _t0, description)
            raise PlatformAuthError(
                f"Telegram getMe not ok: {description}",
                platform_id=self.platform_id,
                http_status=resp.status_code,
                response_body=resp.text,
            )

        bot_info: dict[str, Any] = data.get("result", {})
        bot_username = bot_info.get("username", "")

        # Store token as access_token (no refresh token; expires_in=0 means never)
        token_set = OAuthTokenSet(
            access_token=token,
            refresh_token="",
            id_token="",
            token_type="Bot",
            expires_in=0,
            scope="",
        )
        envelope = self._vault.encrypt_token_json(
            token_data=json.loads(token_set.model_dump_json()),
            platform_id=self.platform_id,
        )
        self.set_token_envelope(envelope)
        if self._credentials is not None:
            self._credentials.status = ConnectorStatus.ACTIVE
            self._credentials.metadata["bot_username"] = bot_username
            self._credentials.metadata["bot_id"] = bot_info.get("id")

        self._logger.info(
            "telegram_authenticated",
            bot_username=bot_username,
            envelope_id=envelope.id,
        )
        await self._emit_event(
            "connector_authenticated",
            {"platform_id": self.platform_id, "bot_username": bot_username, "envelope_id": envelope.id},
        )
        await self._emit_re_training_example("authenticate", "success", _t0)
        return True

    # ─── OAuth2 lifecycle stubs ────────────────────────────────────

    async def build_authorization_url(
        self,
        request: AuthorizationRequest,
    ) -> AuthorizationResponse:
        """Telegram bots do not use OAuth2 authorization URLs."""
        raise NotImplementedError(
            "TelegramConnector does not use OAuth2. Call authenticate() directly."
        )

    async def exchange_code(
        self,
        request: TokenExchangeRequest,
    ) -> OAuthTokenSet:
        """Telegram bots do not use authorization codes."""
        raise NotImplementedError(
            "TelegramConnector does not use OAuth2 code exchange. Call authenticate() directly."
        )

    # ─── Refresh token (re-validate health) ───────────────────────

    async def refresh_token(self) -> TokenRefreshResult:
        """
        Re-validate the bot token health.

        Telegram bot tokens do not expire, but a token can be revoked by the
        operator via BotFather. We call getMe to confirm the token is still
        active. No new token is issued - the existing vault entry is preserved.
        """
        token = self._resolve_token()
        if not token:
            return TokenRefreshResult(success=False, error="No bot token in vault")

        _t0 = self._start_timer()
        try:
            resp = await self._http.get(_bot_url(token, "getMe"))
        except httpx.HTTPError as exc:
            if self._credentials is not None:
                self._credentials.refresh_failure_count += 1
                if self._credentials.refresh_failure_count >= 3:
                    self._credentials.status = ConnectorStatus.REFRESH_FAILED
            await self._emit_re_training_example("refresh_token", "failure", _t0, type(exc).__name__)
            return TokenRefreshResult(success=False, error=str(exc))

        if resp.status_code != 200 or not resp.json().get("ok"):
            if self._credentials is not None:
                self._credentials.refresh_failure_count += 1
                if self._credentials.refresh_failure_count >= 3:
                    self._credentials.status = ConnectorStatus.REFRESH_FAILED
            await self._emit_re_training_example(
                "refresh_token", "failure", _t0, f"HTTP_{resp.status_code}"
            )
            return TokenRefreshResult(
                success=False,
                error=f"getMe returned HTTP {resp.status_code}",
            )

        if self._credentials is not None:
            self._credentials.refresh_failure_count = 0
            self._credentials.status = ConnectorStatus.ACTIVE
            self._credentials.last_refresh_at = utc_now()

        await self._emit_event(
            "connector_token_refreshed",
            {"platform_id": self.platform_id},
        )
        await self._emit_re_training_example("refresh_token", "success", _t0)
        return TokenRefreshResult(success=True)

    # ─── Revoke ───────────────────────────────────────────────────

    async def revoke(self) -> bool:
        """
        Revoke the bot session via the Telegram logOut API.

        POST /logOut logs the bot out from the cloud Telegram server.
        After this call the bot can only operate via local webhooks until
        re-authenticated. Returns True on success.
        """
        token = self._resolve_token()
        if not token:
            return True

        _t0 = self._start_timer()
        try:
            resp = await self._http.post(_bot_url(token, "logOut"))
            success = resp.status_code == 200 and resp.json().get("ok", False)
        except httpx.HTTPError as exc:
            self._logger.warning("telegram_logout_failed", error=str(exc))
            await self._emit_re_training_example("revoke", "failure", _t0, type(exc).__name__)
            return False

        if success and self._credentials is not None:
            self._credentials.status = ConnectorStatus.REVOKED

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.CONNECTOR_REVOKED, {"platform_id": self.platform_id})
        await self._emit_re_training_example(
            "revoke",
            "success" if success else "failure",
            _t0,
            "" if success else f"HTTP_{resp.status_code}",
        )
        return success

    # ─── Health ───────────────────────────────────────────────────

    async def check_health(self) -> ConnectorHealthReport:
        """
        Verify bot token validity via GET /getMe.

        Returns ACTIVE status when the bot is reachable and the token is valid.
        Increments consecutive failure counter and emits SYSTEM_DEGRADED after
        3 consecutive failures (via base-class logic).
        """
        report = self.health_report()
        _t0 = self._start_timer()

        token = self._resolve_token()
        if not token:
            self._consecutive_health_failures += 1
            if self._consecutive_health_failures >= 3:
                await self._emit_degraded()
            await self._emit_re_training_example("check_health", "failure", _t0, "no_token")
            return report

        is_healthy = False
        error_type = ""
        try:
            resp = await self._http.get(_bot_url(token, "getMe"))
            if resp.status_code == 200:
                data = resp.json()
                is_healthy = bool(data.get("ok")) and bool(data.get("result", {}).get("id"))
                if is_healthy and self._credentials is not None:
                    bot_info = data["result"]
                    self._credentials.metadata["bot_username"] = bot_info.get("username", "")
            else:
                error_type = f"HTTP_{resp.status_code}"
                self._logger.warning(
                    "telegram_health_bad_status",
                    status=resp.status_code,
                )
        except httpx.HTTPError as exc:
            error_type = type(exc).__name__
            self._logger.warning("telegram_health_check_failed", error=str(exc))

        if is_healthy:
            self._consecutive_health_failures = 0
            report.status = ConnectorStatus.ACTIVE
        else:
            self._consecutive_health_failures += 1
            if self._consecutive_health_failures >= 3:
                await self._emit_degraded()

        await self._emit_re_training_example(
            "check_health", "success" if is_healthy else "failure", _t0, error_type
        )
        self._logger.debug("telegram_health_check", healthy=is_healthy)
        return report

    # ─── Telegram-specific outbound helpers ───────────────────────

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        parse_mode: str = "Markdown",
    ) -> dict[str, Any]:
        """
        Send a text message to a Telegram chat.

        Returns the Telegram Message object on success, or raises on error.
        """
        token = self._resolve_token()
        if not token:
            raise RuntimeError("TelegramConnector: no bot token available")

        resp = await self._http.post(
            _bot_url(token, "sendMessage"),
            json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram sendMessage failed: {data.get('description')}")
        return data.get("result", {})

    async def send_photo(
        self,
        chat_id: int | str,
        photo_url: str,
        caption: str = "",
        parse_mode: str = "Markdown",
    ) -> dict[str, Any]:
        """Send a photo by URL to a Telegram chat."""
        token = self._resolve_token()
        if not token:
            raise RuntimeError("TelegramConnector: no bot token available")

        payload: dict[str, Any] = {"chat_id": chat_id, "photo": photo_url}
        if caption:
            payload["caption"] = caption
            payload["parse_mode"] = parse_mode

        resp = await self._http.post(_bot_url(token, "sendPhoto"), json=payload)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram sendPhoto failed: {data.get('description')}")
        return data.get("result", {})

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str = "",
        show_alert: bool = False,
    ) -> bool:
        """Answer an inline keyboard callback query."""
        token = self._resolve_token()
        if not token:
            raise RuntimeError("TelegramConnector: no bot token available")

        resp = await self._http.post(
            _bot_url(token, "answerCallbackQuery"),
            json={
                "callback_query_id": callback_query_id,
                "text": text,
                "show_alert": show_alert,
            },
        )
        resp.raise_for_status()
        return resp.json().get("ok", False)

    async def set_webhook(
        self,
        url: str,
        secret_token: str = "",
        allowed_updates: list[str] | None = None,
    ) -> bool:
        """
        Register the webhook URL with Telegram.

        Telegram will POST every update to `url`. The `secret_token` is sent
        as X-Telegram-Bot-Api-Secret-Token on every webhook call so we can
        validate authenticity.
        """
        token = self._resolve_token()
        if not token:
            raise RuntimeError("TelegramConnector: no bot token available")

        payload: dict[str, Any] = {"url": url}
        if secret_token:
            payload["secret_token"] = secret_token
        if allowed_updates is not None:
            payload["allowed_updates"] = allowed_updates

        resp = await self._http.post(_bot_url(token, "setWebhook"), json=payload)
        resp.raise_for_status()
        data = resp.json()
        ok = data.get("ok", False)
        if ok:
            self._logger.info("telegram_webhook_registered", url=url)
        else:
            self._logger.warning(
                "telegram_webhook_registration_failed",
                description=data.get("description"),
                url=url,
            )
        return ok

    async def delete_webhook(self) -> bool:
        """Remove the registered webhook (switches bot to polling mode)."""
        token = self._resolve_token()
        if not token:
            return False
        resp = await self._http.post(_bot_url(token, "deleteWebhook"))
        return resp.status_code == 200 and resp.json().get("ok", False)
