"""
EcodiaOS — X (Twitter) OAuth 2.0 PKCE Connector

Implements the PlatformConnector ABC for X using raw httpx.

X specifics:
  - Public client flow — PKCE (S256) required.
  - Access tokens expire in approximately 2 hours.
  - Token endpoint uses HTTP Basic auth (client_id:client_secret).
  - Refresh tokens are long-lived but require offline.access scope.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
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

logger = structlog.get_logger("identity.x")

_X_AUTHORIZE_URL = "https://twitter.com/i/oauth2/authorize"
_X_TOKEN_URL = "https://api.twitter.com/2/oauth2/token"
_X_REVOKE_URL = "https://api.twitter.com/2/oauth2/revoke"
_X_ACCESS_TOKEN_TTL = 7200  # seconds


def _pkce_challenge(verifier: str) -> str:
    """Compute S256 code challenge from a PKCE verifier string."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _basic_auth(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}"
    return "Basic " + base64.b64encode(raw.encode()).decode("ascii")


class XConnector(PlatformConnector):
    """
    X (Twitter) OAuth 2.0 PKCE connector implemented with raw httpx.

    The caller must supply a code_verifier in AuthorizationRequest
    (or one will be generated) and pass the same verifier to exchange_code().
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
        return "x"

    @property
    def display_name(self) -> str:
        return "X (Twitter)"

    async def build_authorization_url(
        self,
        request: AuthorizationRequest,
    ) -> AuthorizationResponse:
        verifier = request.code_verifier or secrets.token_urlsafe(64)
        challenge = _pkce_challenge(verifier)

        scopes = request.scopes or self._client_config.scopes
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self._client_config.client_id,
            "redirect_uri": request.extra_params.get(
                "redirect_uri", self._client_config.redirect_uri
            ),
            "scope": " ".join(scopes),
            "state": request.state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        params.update({k: v for k, v in request.extra_params.items() if k != "redirect_uri"})

        base = self._client_config.authorize_url or _X_AUTHORIZE_URL
        url = base + "?" + urllib.parse.urlencode(params)

        self._logger.debug("authorization_url_built", platform=self.platform_id)
        return AuthorizationResponse(url=url, state=request.state, code_challenge=challenge)

    async def exchange_code(
        self,
        request: TokenExchangeRequest,
    ) -> OAuthTokenSet:
        resp = await self._http.post(
            self._client_config.token_url or _X_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": request.code,
                "redirect_uri": request.redirect_uri or self._client_config.redirect_uri,
                "code_verifier": request.code_verifier,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": _basic_auth(
                    self._client_config.client_id, self._client_config.client_secret
                ),
            },
        )

        if resp.status_code != 200:
            raise PlatformAuthError(
                f"X token exchange failed: {resp.status_code}",
                platform_id=self.platform_id,
                http_status=resp.status_code,
                response_body=resp.text,
            )

        data = resp.json()
        token_set = OAuthTokenSet(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            token_type=data.get("token_type", "bearer"),
            expires_in=data.get("expires_in", _X_ACCESS_TOKEN_TTL),
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

        current = self._decrypt_current_tokens()
        if current is None or not current.refresh_token:
            return TokenRefreshResult(success=False, error="No refresh token available")

        try:
            resp = await self._http.post(
                self._client_config.token_url or _X_TOKEN_URL,
                data={"grant_type": "refresh_token", "refresh_token": current.refresh_token},
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": _basic_auth(
                        self._client_config.client_id, self._client_config.client_secret
                    ),
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
            return TokenRefreshResult(success=False, error=f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        new_set = OAuthTokenSet(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", current.refresh_token),
            token_type=data.get("token_type", "bearer"),
            expires_in=data.get("expires_in", _X_ACCESS_TOKEN_TTL),
            scope=data.get("scope", current.scope),
        )

        envelope = self._vault.encrypt_token_json(
            token_data=json.loads(new_set.model_dump_json()),
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
        return TokenRefreshResult(success=True, token_set=new_set)

    async def revoke(self) -> bool:
        current = self._decrypt_current_tokens()
        if current is None:
            return True

        revoke_url = self._client_config.revoke_url or _X_REVOKE_URL
        try:
            resp = await self._http.post(
                revoke_url,
                data={"token": current.access_token, "token_type_hint": "access_token"},
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": _basic_auth(
                        self._client_config.client_id, self._client_config.client_secret
                    ),
                },
            )
            success = resp.status_code in (200, 204)
        except httpx.HTTPError as exc:
            self._logger.warning("revoke_failed", platform=self.platform_id, error=str(exc))
            return False

        if success and self._credentials is not None:
            self._credentials.status = ConnectorStatus.REVOKED
        await self._emit_event("connector_revoked", {"platform_id": self.platform_id})
        return success
