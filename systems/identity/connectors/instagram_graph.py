"""
EcodiaOS — Instagram Graph API Connector

Implements the PlatformConnector ABC for Meta's Instagram Graph API using
the system-user / business OAuth 2.0 flow.

Instagram / Meta specifics:
  - The authorization code flow issues a short-lived user access token (~1 hour).
  - That short-lived token must be exchanged for a long-lived token (60-day TTL)
    via a separate GET request to the Graph API.
  - Long-lived tokens can be refreshed before expiry by calling the refresh
    endpoint; a fresh 60-day window starts from the call time.
  - Meta does not issue a traditional OAuth refresh token — the long-lived token
    itself acts as its own renewal credential.
  - Revocation is handled via the Graph API's DELETE /me/permissions endpoint.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import TYPE_CHECKING

import httpx
import structlog

from primitives.common import utc_now
from systems.identity.connector import (
    AuthorizationRequest,
    AuthorizationResponse,
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

logger = structlog.get_logger("identity.instagram_graph")

_IG_AUTHORIZE_URL = "https://api.instagram.com/oauth/authorize"
_IG_SHORT_TOKEN_URL = "https://api.instagram.com/oauth/access_token"
_IG_LONG_TOKEN_URL = "https://graph.instagram.com/access_token"
_IG_REFRESH_TOKEN_URL = "https://graph.instagram.com/refresh_access_token"
_IG_ME_URL = "https://graph.instagram.com/me"
_IG_REVOKE_URL = "https://graph.facebook.com/me/permissions"

# Long-lived tokens are valid for 60 days; express as seconds.
_IG_LONG_LIVED_TOKEN_TTL = 60 * 24 * 3600  # 5,184,000 seconds


class InstagramConnector(PlatformConnector):
    """
    Instagram Graph API OAuth 2.0 connector.

    Token lifecycle:
      1. build_authorization_url() — standard OAuth authorize redirect.
      2. exchange_code() — POST short-lived token, then GET long-lived token.
      3. refresh_token() — GET a refreshed long-lived token (new 60-day window).
      4. revoke() — DELETE /me/permissions to de-authorise the app.
      5. check_health() — GET /me to verify the token is still valid.

    The connector stores only the long-lived token in the vault; the
    short-lived token is a transient intermediate never persisted.
    """

    def __init__(
        self,
        client_config: OAuthClientConfig,
        vault: IdentityVault,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(client_config, vault)
        self._http = http_client or httpx.AsyncClient(timeout=30.0)

    @property
    def platform_id(self) -> str:
        return "instagram_graph"

    @property
    def display_name(self) -> str:
        return "Instagram (Graph API)"

    # ─── OAuth2 Lifecycle ──────────────────────────────────────────────

    async def build_authorization_url(
        self,
        request: AuthorizationRequest,
    ) -> AuthorizationResponse:
        scopes = request.scopes or self._client_config.scopes
        params: dict[str, str] = {
            "client_id": self._client_config.client_id,
            "redirect_uri": request.extra_params.get(
                "redirect_uri", self._client_config.redirect_uri
            ),
            "scope": ",".join(scopes),  # Instagram uses comma-separated scopes
            "response_type": "code",
            "state": request.state,
        }
        params.update(self._client_config.extra_params)
        params.update({k: v for k, v in request.extra_params.items() if k != "redirect_uri"})

        base = self._client_config.authorize_url or _IG_AUTHORIZE_URL
        url = base + "?" + urllib.parse.urlencode(params)

        self._logger.debug("authorization_url_built", platform=self.platform_id)
        return AuthorizationResponse(url=url, state=request.state)

    async def exchange_code(
        self,
        request: TokenExchangeRequest,
    ) -> OAuthTokenSet:
        """
        Two-step exchange: authorization code → short-lived token → long-lived token.

        Step 1: POST to the basic display token endpoint for the short-lived token.
        Step 2: GET the long-lived token using the short-lived token + app secret.
        """
        # Step 1: short-lived user access token (valid ~1 hour)
        short_resp = await self._http.post(
            self._client_config.token_url or _IG_SHORT_TOKEN_URL,
            data={
                "client_id": self._client_config.client_id,
                "client_secret": self._client_config.client_secret,
                "grant_type": "authorization_code",
                "redirect_uri": request.redirect_uri or self._client_config.redirect_uri,
                "code": request.code,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if short_resp.status_code != 200:
            raise PlatformAuthError(
                f"Instagram short-lived token exchange failed: {short_resp.status_code}",
                platform_id=self.platform_id,
                http_status=short_resp.status_code,
                response_body=short_resp.text,
            )

        short_data = short_resp.json()
        short_lived_token: str = short_data["access_token"]

        # Step 2: long-lived user access token (valid 60 days)
        long_resp = await self._http.get(
            _IG_LONG_TOKEN_URL,
            params={
                "grant_type": "ig_exchange_token",
                "client_secret": self._client_config.client_secret,
                "access_token": short_lived_token,
            },
        )

        if long_resp.status_code != 200:
            raise PlatformAuthError(
                f"Instagram long-lived token exchange failed: {long_resp.status_code}",
                platform_id=self.platform_id,
                http_status=long_resp.status_code,
                response_body=long_resp.text,
            )

        long_data = long_resp.json()
        token_set = OAuthTokenSet(
            # Instagram long-lived tokens are self-refreshing; no separate refresh token.
            access_token=long_data["access_token"],
            refresh_token="",
            token_type=long_data.get("token_type", "bearer"),
            expires_in=long_data.get("expires_in", _IG_LONG_LIVED_TOKEN_TTL),
            scope=long_data.get("scope", ""),
        )

        envelope = self._vault.encrypt_token_json(
            token_data=json.loads(token_set.model_dump_json()),
            platform_id=self.platform_id,
        )
        self.set_token_envelope(envelope)
        if self._credentials is not None:
            self._credentials.status = ConnectorStatus.ACTIVE

        self._logger.info("token_exchanged", platform=self.platform_id)
        await self._emit_event(
            "connector_authenticated",
            {"platform_id": self.platform_id, "envelope_id": envelope.id},
        )
        return token_set

    async def refresh_token(self) -> TokenRefreshResult:
        """
        Refresh the long-lived token, resetting its 60-day window.

        Meta requires the token be at least 24 hours old before a refresh
        will extend the window. Calling early is harmless — the existing
        token is returned unchanged if too recent.
        """
        if self._credentials is None or not self._credentials.token_envelope_id:
            return TokenRefreshResult(success=False, error="No stored credentials")

        current = self._decrypt_current_tokens()
        if current is None or not current.access_token:
            return TokenRefreshResult(success=False, error="No access token in vault")

        try:
            resp = await self._http.get(
                _IG_REFRESH_TOKEN_URL,
                params={
                    "grant_type": "ig_refresh_token",
                    "access_token": current.access_token,
                },
            )
        except httpx.HTTPError as exc:
            self._credentials.refresh_failure_count += 1
            if self._credentials.refresh_failure_count >= 3:
                self._credentials.status = ConnectorStatus.REFRESH_FAILED
            return TokenRefreshResult(success=False, error=str(exc))

        if resp.status_code != 200:
            self._credentials.refresh_failure_count += 1
            if self._credentials.refresh_failure_count >= 3:
                self._credentials.status = ConnectorStatus.REFRESH_FAILED
            return TokenRefreshResult(
                success=False,
                error=f"HTTP {resp.status_code}: {resp.text}",
            )

        data = resp.json()
        new_token_set = OAuthTokenSet(
            access_token=data["access_token"],
            refresh_token="",
            token_type=data.get("token_type", "bearer"),
            expires_in=data.get("expires_in", _IG_LONG_LIVED_TOKEN_TTL),
            scope=current.scope,
        )

        envelope = self._vault.encrypt_token_json(
            token_data=json.loads(new_token_set.model_dump_json()),
            platform_id=self.platform_id,
        )
        self.set_token_envelope(envelope)
        self._credentials.refresh_failure_count = 0
        self._credentials.status = ConnectorStatus.ACTIVE
        self._credentials.last_refresh_at = utc_now()

        self._logger.info("token_refreshed", platform=self.platform_id)
        await self._emit_event(
            "connector_token_refreshed",
            {"platform_id": self.platform_id, "envelope_id": envelope.id},
        )
        return TokenRefreshResult(success=True, token_set=new_token_set)

    async def revoke(self) -> bool:
        """
        De-authorise the app by deleting all permissions via the Graph API.

        DELETE https://graph.facebook.com/me/permissions?access_token=...
        """
        current = self._decrypt_current_tokens()
        if current is None:
            return True

        try:
            resp = await self._http.delete(
                _IG_REVOKE_URL,
                params={"access_token": current.access_token},
            )
            success = resp.status_code in (200, 204)
        except httpx.HTTPError as exc:
            self._logger.warning("revoke_failed", platform=self.platform_id, error=str(exc))
            return False

        if success and self._credentials is not None:
            self._credentials.status = ConnectorStatus.REVOKED
        await self._emit_event("connector_revoked", {"platform_id": self.platform_id})
        return success

    # ─── Health ───────────────────────────────────────────────────────

    async def check_health(self) -> bool:
        """
        Verify token validity by fetching the authenticated user's profile.

        GET https://graph.instagram.com/me?fields=id,name&access_token=...

        Returns True if the API responds with HTTP 200 and a user id.
        """
        token = await self.get_access_token()
        if not token:
            return False

        try:
            resp = await self._http.get(
                _IG_ME_URL,
                params={"fields": "id,name", "access_token": token},
            )
        except httpx.HTTPError as exc:
            self._logger.warning("health_check_failed", platform=self.platform_id, error=str(exc))
            return False

        if resp.status_code != 200:
            self._logger.warning(
                "health_check_bad_status",
                platform=self.platform_id,
                status=resp.status_code,
            )
            return False

        data = resp.json()
        valid = bool(data.get("id"))
        self._logger.debug("health_check_complete", platform=self.platform_id, valid=valid)
        return valid
