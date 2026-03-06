"""
EcodiaOS — Canva Connect API Connector

Implements the PlatformConnector ABC for Canva's OAuth 2.0 PKCE flow
using the Canva Connect API.

Canva specifics:
  - Public client — PKCE (S256) is required.
  - Access tokens expire in 4 hours (14,400 seconds).
  - Refresh tokens are long-lived and do not expire as long as they are used.
  - Canva token requests use JSON body, not form-encoded.
  - Required scopes for asset creation and design uploading:
      asset:write          — upload and manage assets
      design:content:read  — read design content (required alongside write ops)
  - Revocation via the standard OAuth 2.0 revocation endpoint (RFC 7009).
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

logger = structlog.get_logger("identity.canva")

_CANVA_AUTHORIZE_URL = "https://www.canva.com/api/oauth/authorize"
_CANVA_TOKEN_URL = "https://api.canva.com/rest/v1/oauth/token"
_CANVA_REVOKE_URL = "https://api.canva.com/rest/v1/oauth/token"  # RFC 7009 DELETE
_CANVA_USER_URL = "https://api.canva.com/rest/v1/users/me"

_CANVA_ACCESS_TOKEN_TTL = 4 * 3600  # 4 hours in seconds

# Scopes required for asset creation and design content reading.
_CANVA_DEFAULT_SCOPES = ["asset:write", "design:content:read"]


def _pkce_challenge(verifier: str) -> str:
    """Compute S256 code challenge from a PKCE verifier string."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class CanvaConnector(PlatformConnector):
    """
    Canva Connect API OAuth 2.0 PKCE connector.

    Requests asset:write and design:content:read scopes by default to
    support asset uploads and design content reading.

    The caller must supply a code_verifier in AuthorizationRequest (or one
    will be generated) and pass the same verifier to exchange_code(). Canva
    uses S256 PKCE for all public client flows.
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
        return "canva"

    @property
    def display_name(self) -> str:
        return "Canva"

    # ─── OAuth2 Lifecycle ──────────────────────────────────────────────

    async def build_authorization_url(
        self,
        request: AuthorizationRequest,
    ) -> AuthorizationResponse:
        verifier = request.code_verifier or secrets.token_urlsafe(64)
        challenge = _pkce_challenge(verifier)

        # Merge default scopes with any caller-supplied scopes, preserving order.
        requested = request.scopes or self._client_config.scopes or _CANVA_DEFAULT_SCOPES
        # Ensure the two required scopes are always present.
        scope_set = list(dict.fromkeys(requested + _CANVA_DEFAULT_SCOPES))

        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self._client_config.client_id,
            "redirect_uri": request.extra_params.get(
                "redirect_uri", self._client_config.redirect_uri
            ),
            "scope": " ".join(scope_set),
            "state": request.state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        params.update(self._client_config.extra_params)
        params.update({k: v for k, v in request.extra_params.items() if k != "redirect_uri"})

        base = self._client_config.authorize_url or _CANVA_AUTHORIZE_URL
        url = base + "?" + urllib.parse.urlencode(params)

        self._logger.debug("authorization_url_built", platform=self.platform_id)
        return AuthorizationResponse(url=url, state=request.state, code_challenge=challenge)

    async def exchange_code(
        self,
        request: TokenExchangeRequest,
    ) -> OAuthTokenSet:
        """
        Exchange an authorization code for a Canva token set.

        Canva expects a JSON body for token requests, not form-encoded.
        """
        payload: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": request.code,
            "redirect_uri": request.redirect_uri or self._client_config.redirect_uri,
            "client_id": self._client_config.client_id,
            "code_verifier": request.code_verifier,
        }

        resp = await self._http.post(
            self._client_config.token_url or _CANVA_TOKEN_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if resp.status_code != 200:
            raise PlatformAuthError(
                f"Canva token exchange failed: {resp.status_code}",
                platform_id=self.platform_id,
                http_status=resp.status_code,
                response_body=resp.text,
            )

        data = resp.json()
        token_set = OAuthTokenSet(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", _CANVA_ACCESS_TOKEN_TTL),
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
        """
        Refresh the Canva access token using the stored refresh token.

        Canva issues a new access token (and optionally a new refresh token)
        in exchange for the current refresh token.
        """
        if self._credentials is None or not self._credentials.token_envelope_id:
            return TokenRefreshResult(success=False, error="No stored credentials")

        current = self._decrypt_current_tokens()
        if current is None or not current.refresh_token:
            return TokenRefreshResult(success=False, error="No refresh token available")

        payload: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": current.refresh_token,
            "client_id": self._client_config.client_id,
        }

        try:
            resp = await self._http.post(
                self._client_config.token_url or _CANVA_TOKEN_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
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
            # Canva may rotate the refresh token; fall back to the existing one if absent.
            refresh_token=data.get("refresh_token", current.refresh_token),
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", _CANVA_ACCESS_TOKEN_TTL),
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
        """
        Revoke the current access token via RFC 7009.

        Canva accepts the token as a JSON body on the same /token endpoint
        with the DELETE method.
        """
        current = self._decrypt_current_tokens()
        if current is None:
            return True

        revoke_url = self._client_config.revoke_url or _CANVA_REVOKE_URL
        try:
            resp = await self._http.request(
                "DELETE",
                revoke_url,
                json={
                    "token": current.access_token,
                    "token_type_hint": "access_token",
                },
                headers={"Content-Type": "application/json"},
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

        GET https://api.canva.com/rest/v1/users/me

        Returns True if the API responds with HTTP 200 and a user ID is present.
        """
        token = await self.get_access_token()
        if not token:
            return False

        try:
            resp = await self._http.get(
                _CANVA_USER_URL,
                headers={"Authorization": f"Bearer {token}"},
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
        # Canva returns {"user": {"id": "...", "display_name": "..."}} shape.
        valid = bool(data.get("user", {}).get("id"))
        self._logger.debug("health_check_complete", platform=self.platform_id, valid=valid)
        return valid
