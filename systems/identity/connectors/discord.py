"""
EcodiaOS - Discord Bot Connector (Spec 23, §14.x: Discord Channel)

Discord uses Bot Token authentication - there is no OAuth2 flow for bots.
The bot token is provisioned once via Discord Developer Portal and never expires
(though it can be regenerated or revoked via the Portal).

Differences from OAuth2 connectors:
  - No authorization URL or code exchange - token is injected at config time.
  - `authenticate()` validates the token via GET /users/@me and stores it in vault.
  - `refresh_token()` re-validates health (tokens don't expire).
  - `revoke()` is a no-op (Discord doesn't have a logout API for bots).
  - `check_health()` calls GET /users/@me - O(1), no side effects.

Bot token storage:
  vault.encrypt_token_json({"access_token": token}, platform_id="discord")

Environment variables:
  ORGANISM_CONNECTORS__DISCORD__BOT_TOKEN      - required
  ORGANISM_DISCORD_CHANNEL_ID                  - optional, for status broadcasts
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

logger = structlog.get_logger("identity.discord")

_DISCORD_API_BASE = "https://discord.com/api/v10"


class DiscordConnector(PlatformConnector):
    """
    Discord Bot Token connector.

    Discord bots authenticate via a static token issued by Discord Developer Portal.
    There is no OAuth2 flow - the token is validated once on `authenticate()`,
    stored encrypted in the IdentityVault, and re-validated on `check_health()`.

    Construction pattern:
        connector = DiscordConnector(
            client_config=OAuthClientConfig(
                client_id="",
                client_secret="",
                authorize_url="",
                token_url="",
                revoke_url="",
                redirect_uri="",
                scopes=[],
            ),
            vault=vault,
            bot_token=os.environ["ORGANISM_CONNECTORS__DISCORD__BOT_TOKEN"],
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
        return "discord"

    @property
    def display_name(self) -> str:
        return "Discord Bot"

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

        Calls GET /users/@me - Discord returns user info on success (HTTP 200)
        or error JSON on invalid token. Token is stored as the `access_token`
        field of an OAuthTokenSet so the base-class infrastructure works
        without modification.

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
            resp = await self._http.get(
                f"{_DISCORD_API_BASE}/users/@me",
                headers={"Authorization": f"Bot {token}"},
            )
        except httpx.HTTPError as exc:
            await self._emit_re_training_example("authenticate", "failure", _t0, type(exc).__name__)
            raise PlatformAuthError(
                f"Discord /users/@me network error: {exc}",
                platform_id=self.platform_id,
            ) from exc

        if resp.status_code != 200:
            await self._emit_re_training_example(
                "authenticate", "failure", _t0, f"HTTP_{resp.status_code}"
            )
            raise PlatformAuthError(
                f"Discord /users/@me failed: {resp.status_code}",
                platform_id=self.platform_id,
                http_status=resp.status_code,
                response_body=resp.text,
            )

        data: dict[str, Any] = resp.json()
        bot_id = data.get("id", "")
        bot_username = data.get("username", "")

        if not bot_id:
            await self._emit_re_training_example("authenticate", "failure", _t0, "no_bot_id")
            raise PlatformAuthError(
                "Discord /users/@me response missing id field",
                platform_id=self.platform_id,
                http_status=resp.status_code,
                response_body=resp.text,
            )

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
            self._credentials.metadata["bot_id"] = bot_id

        self._logger.info(
            "discord_authenticated",
            bot_username=bot_username,
            bot_id=bot_id,
            envelope_id=envelope.id,
        )
        await self._emit_event(
            "connector_authenticated",
            {"platform_id": self.platform_id, "bot_username": bot_username, "bot_id": bot_id},
        )
        await self._emit_re_training_example("authenticate", "success", _t0)
        return True

    # ─── OAuth2 lifecycle stubs ────────────────────────────────────

    async def build_authorization_url(
        self,
        request: AuthorizationRequest,
    ) -> AuthorizationResponse:
        """Discord bots do not use OAuth2 authorization URLs."""
        raise NotImplementedError(
            "DiscordConnector does not use OAuth2. Call authenticate() directly."
        )

    async def exchange_code(
        self,
        request: TokenExchangeRequest,
    ) -> OAuthTokenSet:
        """Discord bots do not use authorization codes."""
        raise NotImplementedError(
            "DiscordConnector does not use OAuth2 code exchange. Call authenticate() directly."
        )

    # ─── Refresh token (re-validate health) ───────────────────────

    async def refresh_token(self) -> TokenRefreshResult:
        """
        Re-validate the bot token health.

        Discord bot tokens do not expire, but a token can be revoked via the
        Developer Portal. We call GET /users/@me to confirm the token is still
        active. No new token is issued - the existing vault entry is preserved.
        """
        token = self._resolve_token()
        if not token:
            return TokenRefreshResult(success=False, error="No bot token in vault")

        _t0 = self._start_timer()
        try:
            resp = await self._http.get(
                f"{_DISCORD_API_BASE}/users/@me",
                headers={"Authorization": f"Bot {token}"},
            )
        except httpx.HTTPError as exc:
            if self._credentials is not None:
                self._credentials.refresh_failure_count += 1
                if self._credentials.refresh_failure_count >= 3:
                    self._credentials.status = ConnectorStatus.REFRESH_FAILED
            await self._emit_re_training_example("refresh_token", "failure", _t0, type(exc).__name__)
            return TokenRefreshResult(success=False, error=str(exc))

        if resp.status_code != 200:
            if self._credentials is not None:
                self._credentials.refresh_failure_count += 1
                if self._credentials.refresh_failure_count >= 3:
                    self._credentials.status = ConnectorStatus.REFRESH_FAILED
            await self._emit_re_training_example(
                "refresh_token", "failure", _t0, f"HTTP_{resp.status_code}"
            )
            return TokenRefreshResult(
                success=False,
                error=f"GET /users/@me returned HTTP {resp.status_code}",
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

    # ─── Revoke (no-op for Discord) ───────────────────────────────

    async def revoke(self) -> bool:
        """
        Revoke the bot token.

        Discord does not provide a logout API for bot tokens. Revocation must
        be done manually via Discord Developer Portal. This method returns True
        (success assumed) and updates credential status to REVOKED, but performs
        no API call.
        """
        if self._credentials is not None:
            self._credentials.status = ConnectorStatus.REVOKED

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.CONNECTOR_REVOKED, {"platform_id": self.platform_id})
        await self._emit_re_training_example("revoke", "success", self._start_timer())
        return True

    # ─── Health ───────────────────────────────────────────────────

    async def check_health(self) -> ConnectorHealthReport:
        """
        Verify bot token validity via GET /users/@me.

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
            resp = await self._http.get(
                f"{_DISCORD_API_BASE}/users/@me",
                headers={"Authorization": f"Bot {token}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                is_healthy = bool(data.get("id"))
                if is_healthy and self._credentials is not None:
                    self._credentials.metadata["bot_username"] = data.get("username", "")
            else:
                error_type = f"HTTP_{resp.status_code}"
                self._logger.warning(
                    "discord_health_bad_status",
                    status=resp.status_code,
                )
        except httpx.HTTPError as exc:
            error_type = type(exc).__name__
            self._logger.warning("discord_health_check_failed", error=str(exc))

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
        self._logger.debug("discord_health_check", healthy=is_healthy)
        return report

    # ─── Discord-specific outbound helpers ───────────────────────

    async def send_message(
        self,
        channel_id: str | int,
        text: str,
    ) -> dict[str, Any]:
        """
        Send a text message to a Discord channel.

        Returns the Discord Message object on success, or raises on error.
        """
        token = self._resolve_token()
        if not token:
            raise RuntimeError("DiscordConnector: no bot token available")

        resp = await self._http.post(
            f"{_DISCORD_API_BASE}/channels/{channel_id}/messages",
            json={"content": text},
            headers={"Authorization": f"Bot {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data

    async def send_embed(
        self,
        channel_id: str | int,
        title: str = "",
        description: str = "",
        color: int = 0x7289DA,
    ) -> dict[str, Any]:
        """Send an embedded message to a Discord channel."""
        token = self._resolve_token()
        if not token:
            raise RuntimeError("DiscordConnector: no bot token available")

        payload: dict[str, Any] = {
            "embeds": [
                {
                    "title": title,
                    "description": description,
                    "color": color,
                }
            ],
        }

        resp = await self._http.post(
            f"{_DISCORD_API_BASE}/channels/{channel_id}/messages",
            json=payload,
            headers={"Authorization": f"Bot {token}"},
        )
        resp.raise_for_status()
        return resp.json()
