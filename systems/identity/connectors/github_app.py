"""
EcodiaOS - GitHub App Connector

JWT to Installation Access Token flow.

GitHub App specifics:
  - Apps authenticate via RS256 JWT signed with their private key (max 10 min).
  - JWT is exchanged for an Installation Access Token (IAT) that expires in 1 hour.
  - No traditional refresh - a fresh JWT + IAT is minted on each refresh call.
  - client_id  = GitHub App ID
  - client_secret = RS256 private key PEM
  - extra_params["installation_id"] = numeric installation ID
"""

from __future__ import annotations

import json
import time
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

logger = structlog.get_logger("identity.github_app")

_GITHUB_API_BASE = "https://api.github.com"
_JWT_VALIDITY_SECONDS = 540   # 9 min - GitHub max is 10
_IAT_TTL_SECONDS = 3600       # Installation Access Tokens expire in 1 hour


def _build_github_jwt(app_id: str, private_key_pem: str) -> str:
    """
    Build a GitHub App JWT signed with RS256.

    Prefers PyJWT if installed; falls back to the cryptography library
    so there is no hard dependency on pyjwt.
    """
    now = int(time.time())
    payload = {
        "iat": now - 60,  # skew back 60s for clock drift tolerance
        "exp": now + _JWT_VALIDITY_SECONDS,
        "iss": app_id,
    }

    try:
        import jwt as pyjwt
        return pyjwt.encode(payload, private_key_pem, algorithm="RS256")
    except ImportError:
        pass

    # Fallback: manual RS256 using cryptography
    import base64

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    body = _b64url(json.dumps(payload).encode())
    message = f"{header}.{body}".encode("ascii")

    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(), password=None
    )
    sig = private_key.sign(message, padding.PKCS1v15(), hashes.SHA256())  # type: ignore[call-arg]
    return f"{header}.{body}.{_b64url(sig)}"


class GitHubAppConnector(PlatformConnector):
    """
    GitHub App connector - JWT to Installation Access Token.

    install_id must be provided via extra_params["installation_id"] in
    the client config or in TokenExchangeRequest.extra_params.
    """

    def __init__(
        self,
        client_config: OAuthClientConfig,
        vault: IdentityVault,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(client_config, vault)
        self._http = http_client or httpx.AsyncClient(
            base_url=_GITHUB_API_BASE,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    @property
    def platform_id(self) -> str:
        return "github_app"

    @property
    def display_name(self) -> str:
        return "GitHub App"

    @property
    def _installation_id(self) -> str:
        return self._client_config.extra_params.get("installation_id", "")

    async def build_authorization_url(
        self,
        request: AuthorizationRequest,
    ) -> AuthorizationResponse:
        """
        Return the GitHub App installation page URL.

        GitHub Apps use JWT-based authentication, not OAuth authorization codes,
        so this method provides the installation URL for operator convenience.
        """
        app_id = self._client_config.client_id
        url = f"https://github.com/apps/{app_id}/installations/new"
        return AuthorizationResponse(url=url, state=request.state)

    async def exchange_code(
        self,
        request: TokenExchangeRequest,
    ) -> OAuthTokenSet:
        """
        Mint a JWT and obtain an Installation Access Token.

        The authorization code is unused - GitHub App auth is entirely JWT-based.
        """
        installation_id = (
            request.extra_params.get("installation_id") or self._installation_id
        )
        if not installation_id:
            raise PlatformAuthError(
                "installation_id required in request.extra_params or client_config.extra_params",
                platform_id=self.platform_id,
            )

        token_set = await self._fetch_iat(installation_id)

        envelope = self._vault.encrypt_token_json(
            token_data=json.loads(token_set.model_dump_json()),
            platform_id=self.platform_id,
        )
        self.set_token_envelope(envelope)
        if self._credentials is not None:
            self._credentials.status = ConnectorStatus.ACTIVE

        self._logger.info("iat_obtained", platform=self.platform_id)
        await self._emit_event(
            "connector_authenticated",
            {"platform_id": self.platform_id, "envelope_id": envelope.id},
        )
        return token_set

    async def refresh_token(self) -> TokenRefreshResult:
        """
        Obtain a fresh IAT by minting a new JWT.

        GitHub Apps have no long-lived refresh token - the private key IS the
        long-term credential. Each refresh re-derives a JWT from config.
        """
        if self._credentials is None:
            return TokenRefreshResult(success=False, error="No credentials configured")

        install_id = self._installation_id
        if not install_id:
            return TokenRefreshResult(success=False, error="installation_id not configured")

        try:
            token_set = await self._fetch_iat(install_id)
        except (PlatformAuthError, httpx.HTTPError) as exc:
            self._credentials.refresh_failure_count += 1
            if self._credentials.refresh_failure_count >= 3:
                self._credentials.status = ConnectorStatus.REFRESH_FAILED
            return TokenRefreshResult(success=False, error=str(exc))

        envelope = self._vault.encrypt_token_json(
            token_data=json.loads(token_set.model_dump_json()),
            platform_id=self.platform_id,
        )
        self.set_token_envelope(envelope)
        self._credentials.refresh_failure_count = 0
        self._credentials.status = ConnectorStatus.ACTIVE
        self._credentials.last_refresh_at = utc_now()

        self._logger.info("iat_refreshed", platform=self.platform_id)
        await self._emit_event(
            "connector_token_refreshed",
            {"platform_id": self.platform_id, "envelope_id": envelope.id},
        )
        return TokenRefreshResult(success=True, token_set=token_set)

    async def revoke(self) -> bool:
        """
        Revoke the current Installation Access Token via DELETE /installation/token.

        The IAT naturally expires in 1 hour; this provides early revocation.
        """
        current = self._decrypt_current_tokens()
        if current is None:
            return True

        try:
            resp = await self._http.delete(
                "/installation/token",
                headers={"Authorization": f"Bearer {current.access_token}"},
            )
            # 204 = revoked; 401 = already expired - both are acceptable outcomes
            success = resp.status_code in (204, 401)
        except httpx.HTTPError as exc:
            self._logger.warning("revoke_failed", platform=self.platform_id, error=str(exc))
            return False

        if success and self._credentials is not None:
            self._credentials.status = ConnectorStatus.REVOKED
        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_event(_SET.CONNECTOR_REVOKED, {"platform_id": self.platform_id})
        return success

    async def _fetch_iat(self, installation_id: str) -> OAuthTokenSet:
        """Build JWT, POST to GitHub, return OAuthTokenSet."""
        jwt = _build_github_jwt(
            app_id=self._client_config.client_id,
            private_key_pem=self._client_config.client_secret,
        )
        resp = await self._http.post(
            f"/app/installations/{installation_id}/access_tokens",
            headers={"Authorization": f"Bearer {jwt}"},
        )
        if resp.status_code != 201:
            raise PlatformAuthError(
                f"GitHub IAT request failed: {resp.status_code}",
                platform_id=self.platform_id,
                http_status=resp.status_code,
                response_body=resp.text,
            )

        data = resp.json()
        permissions: dict[str, str] = data.get("permissions") or {}
        scope = ",".join(f"{k}:{v}" for k, v in permissions.items())

        return OAuthTokenSet(
            access_token=data["token"],
            refresh_token="",  # IATs have no refresh token; JWT re-auth is used
            token_type="bearer",
            expires_in=_IAT_TTL_SECONDS,
            scope=scope,
        )
