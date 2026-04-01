"""
EcodiaOS - Google OAuth2 Connector (Spec 23, §14.1 + §14.2)

Implements the PlatformConnector ABC for Google Workspace / Google APIs using
the OAuth 2.0 Authorization Code flow with PKCE (S256).

Google specifics:
  - Public client - PKCE (S256) recommended for web and mobile apps.
  - Access tokens expire in 3,600 seconds (1 hour).
  - Refresh tokens are long-lived until explicitly revoked or 6 months of
    inactivity. Google issues a refresh token only on the first authorization
    or when access_type=offline is requested.
  - Token revocation via POST to https://oauth2.googleapis.com/revoke.
  - Scopes use full URLs (e.g. https://www.googleapis.com/auth/userinfo.email).
  - The token response includes an id_token (JWT) when openid is in scope.

Authentication type:
  Authorization Code + PKCE (S256) - same pattern as CanvaConnector.
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

logger = structlog.get_logger("identity.google")

_GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

_GOOGLE_ACCESS_TOKEN_TTL = 3600  # seconds - Google standard

# Minimum scopes required for userinfo health check.
_GOOGLE_DEFAULT_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


def _pkce_challenge(verifier: str) -> str:
    """Compute S256 code challenge from a PKCE verifier string."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class GoogleConnector(PlatformConnector):
    """
    Google Workspace OAuth2 PKCE connector.

    Implements the full PlatformConnector lifecycle for Google APIs:
      1. build_authorization_url() - consent URL with PKCE S256 challenge.
      2. exchange_code()           - POST code + verifier → token set.
      3. refresh_token()           - POST refresh_token → new access token.
      4. revoke()                  - POST token to Google's revocation endpoint.
      5. check_health()            - GET /userinfo to verify token validity.

    Construction pattern:
        connector = GoogleConnector(
            client_config=OAuthClientConfig(
                client_id="...",
                client_secret="...",
                authorize_url=_GOOGLE_AUTHORIZE_URL,
                token_url=_GOOGLE_TOKEN_URL,
                revoke_url=_GOOGLE_REVOKE_URL,
                redirect_uri="https://eos.example.com/oauth/google/callback",
                scopes=_GOOGLE_DEFAULT_SCOPES,
            ),
            vault=vault,
        )
        connector.set_event_bus(event_bus)
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
        return "google"

    @property
    def display_name(self) -> str:
        return "Google Workspace"

    # ─── OAuth2 Lifecycle ──────────────────────────────────────────────

    async def build_authorization_url(
        self,
        request: AuthorizationRequest,
    ) -> AuthorizationResponse:
        """
        Build the Google OAuth2 PKCE authorization URL.

        Generates a fresh PKCE verifier if none is supplied in the request.
        Always requests access_type=offline and prompt=consent to ensure a
        refresh token is included in the token response.
        """
        verifier = request.code_verifier or secrets.token_urlsafe(64)
        challenge = _pkce_challenge(verifier)

        # Merge default scopes with caller-supplied scopes, preserving order.
        requested = request.scopes or self._client_config.scopes or _GOOGLE_DEFAULT_SCOPES
        scope_set = list(dict.fromkeys(requested + _GOOGLE_DEFAULT_SCOPES))

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
            # Request offline access so Google issues a refresh token.
            "access_type": "offline",
            # Force consent screen so refresh token is always issued.
            "prompt": "consent",
        }
        params.update(self._client_config.extra_params)
        params.update({k: v for k, v in request.extra_params.items() if k != "redirect_uri"})

        base = self._client_config.authorize_url or _GOOGLE_AUTHORIZE_URL
        url = base + "?" + urllib.parse.urlencode(params)

        self._logger.debug("authorization_url_built", platform=self.platform_id)
        return AuthorizationResponse(url=url, state=request.state, code_challenge=challenge)

    async def exchange_code(
        self,
        request: TokenExchangeRequest,
    ) -> OAuthTokenSet:
        """
        Exchange an authorization code for a Google token set.

        Google uses form-encoded POST bodies for token requests (not JSON).
        On success, encrypts the token set via the vault and updates credentials.
        """
        _t0 = self._start_timer()
        payload: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": request.code,
            "redirect_uri": request.redirect_uri or self._client_config.redirect_uri,
            "client_id": self._client_config.client_id,
            "client_secret": self._client_config.client_secret,
            "code_verifier": request.code_verifier,
        }

        resp = await self._http.post(
            self._client_config.token_url or _GOOGLE_TOKEN_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if resp.status_code != 200:
            await self._emit_re_training_example(
                "exchange_code", "failure", _t0, f"HTTP_{resp.status_code}"
            )
            raise PlatformAuthError(
                f"Google token exchange failed: {resp.status_code}",
                platform_id=self.platform_id,
                http_status=resp.status_code,
                response_body=resp.text,
            )

        data = resp.json()
        token_set = OAuthTokenSet(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", ""),
            id_token=data.get("id_token", ""),
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", _GOOGLE_ACCESS_TOKEN_TTL),
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
        await self._emit_re_training_example("exchange_code", "success", _t0)
        return token_set

    async def refresh_token(self) -> TokenRefreshResult:
        """
        Refresh the Google access token using the stored refresh token.

        Google form-encodes refresh requests. On success, Google returns a new
        access token but does NOT rotate the refresh token - the existing one
        remains valid and must be preserved in the new token set.
        """
        if self._credentials is None or not self._credentials.token_envelope_id:
            return TokenRefreshResult(success=False, error="No stored credentials")

        current = self._decrypt_current_tokens()
        if current is None or not current.refresh_token:
            return TokenRefreshResult(success=False, error="No refresh token available")

        _t0 = self._start_timer()
        payload: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": current.refresh_token,
            "client_id": self._client_config.client_id,
            "client_secret": self._client_config.client_secret,
        }

        try:
            resp = await self._http.post(
                self._client_config.token_url or _GOOGLE_TOKEN_URL,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
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
                error=f"HTTP {resp.status_code}: {resp.text}",
            )

        data = resp.json()
        new_token_set = OAuthTokenSet(
            access_token=data["access_token"],
            # Google does not rotate refresh tokens; preserve the original.
            refresh_token=data.get("refresh_token", current.refresh_token),
            id_token=data.get("id_token", current.id_token),
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", _GOOGLE_ACCESS_TOKEN_TTL),
            scope=data.get("scope", current.scope),
        )

        envelope = self._vault.encrypt_token_json(
            token_data=json.loads(new_token_set.model_dump_json()),
            platform_id=self.platform_id,
        )
        self.set_token_envelope(envelope)
        if self._credentials is not None:
            self._credentials.refresh_failure_count = 0
            self._credentials.status = ConnectorStatus.ACTIVE
            self._credentials.last_refresh_at = utc_now()

        self._logger.info("token_refreshed", platform=self.platform_id)
        await self._emit_event(
            "connector_token_refreshed",
            {"platform_id": self.platform_id, "envelope_id": envelope.id},
        )
        await self._emit_re_training_example("refresh_token", "success", _t0)
        return TokenRefreshResult(success=True, token_set=new_token_set)

    async def revoke(self) -> bool:
        """
        Revoke the current access token via Google's revocation endpoint.

        POST https://oauth2.googleapis.com/revoke?token=...

        Google accepts both access tokens and refresh tokens at this endpoint.
        Revoking the refresh token invalidates all derived access tokens.
        """
        current = self._decrypt_current_tokens()
        if current is None:
            return True

        # Prefer revoking the refresh token to invalidate the entire grant.
        token_to_revoke = current.refresh_token or current.access_token
        revoke_url = self._client_config.revoke_url or _GOOGLE_REVOKE_URL
        _t0 = self._start_timer()

        try:
            resp = await self._http.post(
                revoke_url,
                params={"token": token_to_revoke},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            success = resp.status_code in (200, 204)
        except httpx.HTTPError as exc:
            self._logger.warning("revoke_failed", platform=self.platform_id, error=str(exc))
            await self._emit_re_training_example("revoke", "failure", _t0, type(exc).__name__)
            return False

        if success and self._credentials is not None:
            self._credentials.status = ConnectorStatus.REVOKED
        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.CONNECTOR_REVOKED, {"platform_id": self.platform_id})
        await self._emit_re_training_example(
            "revoke", "success" if success else "failure", _t0,
            "" if success else f"HTTP_{resp.status_code}",
        )
        return success

    # ─── Health ───────────────────────────────────────────────────────

    async def check_health(self) -> ConnectorHealthReport:
        """
        Verify token validity by fetching the authenticated user's profile.

        GET https://www.googleapis.com/oauth2/v3/userinfo

        Updates the consecutive health failure counter and emits SYSTEM_DEGRADED
        after 3 consecutive failures (via the base class logic).
        """
        report = self.health_report()
        _t0 = self._start_timer()

        token = await self.get_access_token()
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
                _GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                # Google userinfo returns {"sub": "...", "email": "..."} shape.
                is_healthy = bool(data.get("sub") or data.get("email"))
                if is_healthy and self._credentials is not None:
                    email = data.get("email", "")
                    if email:
                        self._credentials.metadata["email"] = email
            else:
                error_type = f"HTTP_{resp.status_code}"
                self._logger.warning(
                    "health_check_bad_status",
                    platform=self.platform_id,
                    status=resp.status_code,
                )
        except httpx.HTTPError as exc:
            error_type = type(exc).__name__
            self._logger.warning("health_check_failed", platform=self.platform_id, error=str(exc))

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
        self._logger.debug("health_check_complete", platform=self.platform_id, healthy=is_healthy)
        return report
