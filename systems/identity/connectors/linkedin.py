"""
EcodiaOS — LinkedIn v2 Connector

Implements the PlatformConnector ABC for LinkedIn OAuth 2.0.

LinkedIn specifics:
  - Access tokens expire in 60 days (5,184,000 seconds).
  - Refresh tokens are issued alongside access tokens (~365 day TTL).
  - LinkedIn rotates both tokens on each refresh call.
  - Confidential client flow — no PKCE.
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

logger = structlog.get_logger("identity.linkedin")

_LINKEDIN_ACCESS_TOKEN_TTL = 60 * 24 * 3600  # 60 days in seconds


class LinkedInConnector(PlatformConnector):
    """
    LinkedIn v2 OAuth 2.0 connector (confidential client, no PKCE).

    Typical scopes: openid, profile, email, w_member_social
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
        return "linkedin"

    @property
    def display_name(self) -> str:
        return "LinkedIn"

    async def build_authorization_url(
        self,
        request: AuthorizationRequest,
    ) -> AuthorizationResponse:
        scopes = request.scopes or self._client_config.scopes
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self._client_config.client_id,
            "redirect_uri": request.extra_params.get(
                "redirect_uri", self._client_config.redirect_uri
            ),
            "state": request.state,
            "scope": " ".join(scopes),
        }
        params.update(self._client_config.extra_params)
        params.update({k: v for k, v in request.extra_params.items() if k != "redirect_uri"})

        url = self._client_config.authorize_url + "?" + urllib.parse.urlencode(params)
        self._logger.debug("authorization_url_built", platform=self.platform_id)
        return AuthorizationResponse(url=url, state=request.state)

    async def exchange_code(
        self,
        request: TokenExchangeRequest,
    ) -> OAuthTokenSet:
        payload = {
            "grant_type": "authorization_code",
            "code": request.code,
            "redirect_uri": request.redirect_uri or self._client_config.redirect_uri,
            "client_id": self._client_config.client_id,
            "client_secret": self._client_config.client_secret,
        }

        resp = await self._http.post(
            self._client_config.token_url,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if resp.status_code != 200:
            raise PlatformAuthError(
                f"LinkedIn token exchange failed: {resp.status_code}",
                platform_id=self.platform_id,
                http_status=resp.status_code,
                response_body=resp.text,
            )

        data = resp.json()
        token_set = OAuthTokenSet(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", _LINKEDIN_ACCESS_TOKEN_TTL),
            scope=data.get("scope", ""),
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
        if self._credentials is None or not self._credentials.token_envelope_id:
            return TokenRefreshResult(success=False, error="No stored credentials")

        # Refresh token must be retrieved from envelope via CRUD before calling this.
        # The connector receives it pre-populated via set_credentials().
        current = self._decrypt_current_tokens()
        if current is None or not current.refresh_token:
            return TokenRefreshResult(success=False, error="No refresh token available")

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": current.refresh_token,
            "client_id": self._client_config.client_id,
            "client_secret": self._client_config.client_secret,
        }

        try:
            resp = await self._http.post(
                self._client_config.token_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
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
            # LinkedIn rotates the refresh token — fall back to old one if absent.
            refresh_token=data.get("refresh_token", current.refresh_token),
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", _LINKEDIN_ACCESS_TOKEN_TTL),
            scope=data.get("scope", current.scope),
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
        if not self._client_config.revoke_url:
            return True

        current = self._decrypt_current_tokens()
        if current is None:
            return True

        try:
            resp = await self._http.post(
                self._client_config.revoke_url,
                data={
                    "token": current.access_token,
                    "client_id": self._client_config.client_id,
                    "client_secret": self._client_config.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            success = resp.status_code in (200, 204)
        except httpx.HTTPError as exc:
            self._logger.warning("revoke_failed", platform=self.platform_id, error=str(exc))
            return False

        if success and self._credentials is not None:
            self._credentials.status = ConnectorStatus.REVOKED
        await self._emit_event("connector_revoked", {"platform_id": self.platform_id})
        return success
