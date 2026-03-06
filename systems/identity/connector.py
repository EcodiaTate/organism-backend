"""
EcodiaOS — Platform Connector ABCs (Phase 16h: External Identity Layer)

Abstract base classes and Pydantic schemas that standardise how EcodiaOS
authenticates with external platforms (Google, GitHub, X/Twitter, etc.).

Every connector implements the same OAuth2 lifecycle:
  1. authorize_url()   — Build the consent URL for the human operator.
  2. exchange_code()   — Swap the authorization code for tokens.
  3. refresh_token()   — Refresh an expired access token.
  4. revoke()          — Revoke a token (best-effort).

Connectors never touch raw secrets directly — they receive an IdentityVault
reference and operate on SealedEnvelopes. The connector ABC is deliberately
transport-agnostic: HTTP client injection happens at construction time so
concrete implementations can use httpx, aiohttp, or any async client.

Event integration: connectors emit Synapse events for token refresh,
expiry, and revocation so downstream systems (Nova, Thymos) can react.

Thread-safety: NOT thread-safe. Single-threaded asyncio like all EOS.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
from datetime import datetime
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, utc_now

if TYPE_CHECKING:

    from clients.redis import RedisClient
    from systems.identity.vault import IdentityVault, SealedEnvelope
    from systems.synapse.event_bus import EventBus
logger = structlog.get_logger("identity.connector")

# One asyncio.Lock per platform_id — shared across all connector instances in the process.
# Prevents concurrent refresh storms: only one coroutine executes the OAuth refresh
# while others wait for the result already stored in the cache / credentials.
_REFRESH_LOCKS: dict[str, asyncio.Lock] = {}

# Redis TTL guard: evict cache entries 60 s before token expiry so callers
# always get a token with at least 60 s of remaining lifetime.
_CACHE_EXPIRY_GUARD_S: int = 60

# Health check: emit SYSTEM_DEGRADED after this many consecutive failures.
_HEALTH_FAILURE_THRESHOLD: int = 3


# ─── Enums ──────────────────────────────────────────────────────────────


class OAuthGrantType(enum.StrEnum):
    """Supported OAuth2 grant types."""

    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"
    DEVICE_CODE = "urn:ietf:params:oauth:grant-type:device_code"


class ConnectorStatus(enum.StrEnum):
    """Lifecycle state of a platform connector."""

    UNCONFIGURED = "unconfigured"
    AWAITING_AUTH = "awaiting_auth"
    ACTIVE = "active"
    TOKEN_EXPIRED = "token_expired"
    REFRESH_FAILED = "refresh_failed"
    REVOKED = "revoked"
    ERROR = "error"


class TokenType(enum.StrEnum):
    """OAuth token classification."""

    ACCESS = "access"
    REFRESH = "refresh"
    ID_TOKEN = "id_token"


# ─── Pydantic Schemas ──────────────────────────────────────────────────


class OAuthClientConfig(EOSBaseModel):
    """
    OAuth2 client registration for a specific platform.

    Stored in config (YAML / env vars), never in the database.
    """

    client_id: str
    client_secret: str
    authorize_url: str
    """Platform's authorization endpoint (e.g. https://accounts.google.com/o/oauth2/v2/auth)."""

    token_url: str
    """Platform's token exchange endpoint."""

    revoke_url: str = ""
    """Platform's token revocation endpoint (optional — not all platforms support it)."""

    redirect_uri: str = ""
    """OAuth redirect URI registered with the platform."""

    scopes: list[str] = Field(default_factory=list)
    """Default scopes to request during authorization."""

    grant_type: OAuthGrantType = OAuthGrantType.AUTHORIZATION_CODE
    """Primary grant type for this platform."""

    extra_params: dict[str, str] = Field(default_factory=dict)
    """Platform-specific extra parameters for the authorize request."""


class OAuthTokenSet(EOSBaseModel):
    """
    The token set returned by a successful OAuth2 exchange or refresh.

    This is the plaintext representation — it is encrypted into a
    SealedEnvelope before storage and never persisted raw.
    """

    access_token: str
    refresh_token: str = ""
    id_token: str = ""
    token_type: str = "Bearer"
    expires_in: int = 3600
    """Token lifetime in seconds (as reported by the provider)."""

    scope: str = ""
    """Space-separated scopes actually granted by the provider."""

    issued_at: datetime = Field(default_factory=utc_now)
    """When EOS received this token set."""

    @property
    def expires_at(self) -> datetime:
        from datetime import timedelta
        return self.issued_at + timedelta(seconds=self.expires_in)

    @property
    def is_expired(self) -> bool:
        return utc_now() >= self.expires_at

    @property
    def remaining_seconds(self) -> float:
        delta = self.expires_at - utc_now()
        return max(0.0, delta.total_seconds())


class ConnectorCredentials(Identified, Timestamped):
    """
    Per-platform credential record. Stored in the database with encrypted
    envelopes — never raw tokens.

    The connector_id is a stable identifier for this particular OAuth
    connection (e.g. 'google:user@example.com' or 'github:org-bot').
    """

    connector_id: str
    """Stable identifier for this connection instance."""

    platform_id: str
    """Platform identifier (e.g. 'google', 'github', 'twitter')."""

    status: ConnectorStatus = ConnectorStatus.UNCONFIGURED
    """Current lifecycle state."""

    token_envelope_id: str = ""
    """ID of the SealedEnvelope containing the encrypted OAuthTokenSet."""

    totp_envelope_id: str = ""
    """ID of the SealedEnvelope containing the encrypted TOTP base32 secret (if 2FA required)."""

    cookie_envelope_id: str = ""
    """ID of the SealedEnvelope containing encrypted browser cookie state (if session-based)."""

    last_refresh_at: datetime | None = None
    """When the token was last successfully refreshed."""

    refresh_failure_count: int = 0
    """Consecutive refresh failures (reset on success)."""

    metadata: dict[str, str] = Field(default_factory=dict)
    """Platform-specific metadata (e.g. account email, user ID)."""


class AuthorizationRequest(EOSBaseModel):
    """Parameters for building an OAuth2 authorization URL."""

    state: str
    """CSRF state token (opaque, random, verified on callback)."""

    scopes: list[str] = Field(default_factory=list)
    """Scopes to request (overrides client config defaults if non-empty)."""

    code_verifier: str = ""
    """PKCE code verifier (required for public clients)."""

    extra_params: dict[str, str] = Field(default_factory=dict)
    """Additional query parameters for the authorize request."""


class AuthorizationResponse(EOSBaseModel):
    """Result of building an authorization URL."""

    url: str
    """The full authorization URL to present to the human operator."""

    state: str
    """The CSRF state token (must be verified on callback)."""

    code_challenge: str = ""
    """PKCE code challenge (if code_verifier was provided)."""


class TokenExchangeRequest(EOSBaseModel):
    """Parameters for exchanging an authorization code for tokens."""

    code: str
    """The authorization code from the callback."""

    state: str
    """The CSRF state token (for verification)."""

    redirect_uri: str = ""
    """The redirect URI used in the authorization request."""

    code_verifier: str = ""
    """PKCE code verifier (must match the challenge sent during authorization)."""


class TokenRefreshResult(EOSBaseModel):
    """Result of a token refresh attempt."""

    success: bool
    token_set: OAuthTokenSet | None = None
    error: str = ""
    retry_after_seconds: int = 0
    """Provider-suggested retry delay (0 = immediate retry OK)."""


class ConnectorHealthReport(EOSBaseModel):
    """Health snapshot for a connector, used in system health checks."""

    connector_id: str
    platform_id: str
    status: ConnectorStatus
    token_remaining_seconds: float = 0.0
    refresh_failure_count: int = 0
    last_refresh_at: datetime | None = None
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Abstract Base Class ───────────────────────────────────────────────


class PlatformConnector(ABC):
    """
    Abstract base class for OAuth2 platform connectors.

    Concrete implementations (GoogleConnector, GitHubConnector, etc.)
    provide platform-specific HTTP calls. The ABC standardises:
      - The OAuth2 authorization → exchange → refresh → revoke lifecycle.
      - Token encryption/decryption via IdentityVault.
      - Synapse event emission for lifecycle transitions.
      - Health reporting for Thymos monitoring.

    Construction pattern:
        connector = GoogleConnector(
            client_config=OAuthClientConfig(...),
            vault=vault,
            http_client=httpx.AsyncClient(),
        )
        connector.set_event_bus(event_bus)
    """

    def __init__(
        self,
        client_config: OAuthClientConfig,
        vault: IdentityVault,
    ) -> None:
        self._client_config = client_config
        self._vault = vault
        self._event_bus: EventBus | None = None
        self._redis: RedisClient | None = None
        self._credentials: ConnectorCredentials | None = None
        # In-memory envelope cache — populated whenever a token is encrypted so
        # _decrypt_current_tokens() can access the SealedEnvelope without a DB
        # round-trip.  Persists for the lifetime of the process; the DB is the
        # authoritative store on restart (callers must call set_token_envelope).
        self._token_envelope: SealedEnvelope | None = None
        # Consecutive health-check failures for Thymos immune wiring.
        self._consecutive_health_failures: int = 0
        self._logger = logger.bind(
            component="platform_connector",
            platform=self.platform_id,
        )

    # ─── Identity ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def platform_id(self) -> str:
        """Stable platform identifier (e.g. 'google', 'github')."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable platform name (e.g. 'Google Workspace')."""
        ...

    @property
    def status(self) -> ConnectorStatus:
        if self._credentials is None:
            return ConnectorStatus.UNCONFIGURED
        return self._credentials.status

    @property
    def credentials(self) -> ConnectorCredentials | None:
        return self._credentials

    # ─── Wiring ──────────────────────────────────────────────────────

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire Synapse event bus for connector lifecycle events."""
        self._event_bus = event_bus

    def set_redis_cache(self, redis: RedisClient) -> None:
        """Wire a Redis client for decrypted-token caching."""
        self._redis = redis

    def set_credentials(self, credentials: ConnectorCredentials) -> None:
        """Load persisted credentials (called at startup from database)."""
        self._credentials = credentials

    def set_token_envelope(self, envelope: SealedEnvelope) -> None:
        """
        Cache a SealedEnvelope for in-memory token decryption.

        Call this whenever a new envelope is produced (after exchange_code or
        refresh_token) so _decrypt_current_tokens() can access the ciphertext
        without a database round-trip.  On process restart, callers must
        re-populate the envelope from DB before the first get_access_token().
        """
        self._token_envelope = envelope
        if self._credentials is not None:
            self._credentials.token_envelope_id = envelope.id

    # ─── OAuth2 Lifecycle (abstract) ─────────────────────────────────

    @abstractmethod
    async def build_authorization_url(
        self,
        request: AuthorizationRequest,
    ) -> AuthorizationResponse:
        """
        Build the OAuth2 authorization URL for the human operator.

        The implementation must:
          1. Construct the URL with client_id, redirect_uri, scopes, state.
          2. If PKCE is required, compute the code_challenge from code_verifier.
          3. Return the full URL.
        """
        ...

    @abstractmethod
    async def exchange_code(
        self,
        request: TokenExchangeRequest,
    ) -> OAuthTokenSet:
        """
        Exchange an authorization code for a token set.

        The implementation must:
          1. POST to the platform's token_url with the code + client credentials.
          2. Parse the response into an OAuthTokenSet.
          3. Encrypt the token set via self._vault and store the envelope.
          4. Update self._credentials status to ACTIVE.

        Raises:
            PlatformAuthError: If the exchange fails.
        """
        ...

    @abstractmethod
    async def refresh_token(self) -> TokenRefreshResult:
        """
        Refresh the current access token using the stored refresh token.

        The implementation must:
          1. Decrypt the current token set from the vault.
          2. POST to the platform's token_url with grant_type=refresh_token.
          3. Encrypt the new token set and update the envelope.
          4. Update self._credentials.last_refresh_at.
          5. Return a TokenRefreshResult.

        On failure, increment refresh_failure_count and transition status
        to REFRESH_FAILED after 3 consecutive failures.
        """
        ...

    @abstractmethod
    async def revoke(self) -> bool:
        """
        Revoke the current token with the platform (best-effort).

        Returns True if revocation succeeded or the platform doesn't
        support revocation. Returns False on error.
        """
        ...

    # ─── Token Access (concrete helpers) ─────────────────────────────

    async def get_access_token(self) -> str | None:
        """
        Return the current access token, refreshing if expired.

        Hot path (Redis cache hit):
          1. Check Redis for a cached OAuthTokenSet with >60 s remaining.
          2. Return access_token immediately — no DB query, no decryption.

        Warm path (cache miss or near-expiry):
          1. Decrypt from vault (DB query + Fernet).
          2. If remaining < 60 s, acquire the per-platform refresh lock so
             only one coroutine performs the OAuth refresh while others wait.
          3. After refresh, write the new token back to Redis.

        Returns None if no credentials are configured or refresh fails.
        """
        if self._credentials is None or not self._credentials.token_envelope_id:
            return None

        # ── 1. Redis cache hit ──────────────────────────────────────
        cached = await self._read_token_cache()
        if cached is not None and cached.remaining_seconds >= _CACHE_EXPIRY_GUARD_S:
            return cached.access_token

        # ── 2. Decrypt from vault ───────────────────────────────────
        token_set = self._decrypt_current_tokens()
        if token_set is None:
            return None

        # ── 3. Refresh if needed (with per-platform lock) ───────────
        if token_set.remaining_seconds < _CACHE_EXPIRY_GUARD_S:
            lock = self._get_refresh_lock()
            async with lock:
                # Re-check cache inside lock — another waiter may have refreshed
                cached_after_wait = await self._read_token_cache()
                if (
                    cached_after_wait is not None
                    and cached_after_wait.remaining_seconds >= _CACHE_EXPIRY_GUARD_S
                ):
                    return cached_after_wait.access_token

                result = await self.refresh_token()
                if not result.success or result.token_set is None:
                    self._logger.warning("auto_refresh_failed", error=result.error)
                    return None
                token_set = result.token_set

        # ── 4. Populate / refresh Redis cache ──────────────────────
        await self._write_token_cache(token_set)
        return token_set.access_token

    def _decrypt_current_tokens(self) -> OAuthTokenSet | None:
        """
        Decrypt the cached token envelope back to an OAuthTokenSet.

        Uses the in-memory SealedEnvelope set via set_token_envelope().  If the
        envelope has not been loaded (e.g. after a cold process restart) callers
        must call set_token_envelope() with the DB-fetched envelope first.
        """
        if self._token_envelope is None:
            return None
        try:
            token_data = self._vault.decrypt_token_json(self._token_envelope)
            return OAuthTokenSet.model_validate(token_data)
        except Exception as exc:
            self._logger.warning(
                "decrypt_current_tokens_failed",
                platform_id=self.platform_id,
                envelope_id=self._token_envelope.id,
                error=str(exc),
            )
            return None

    # ─── Cache Helpers ────────────────────────────────────────────────

    def _cache_key(self) -> str:
        """Redis key for this connector's cached token set."""
        return f"identity:token:{self.platform_id}"

    async def _read_token_cache(self) -> OAuthTokenSet | None:
        """Read a cached OAuthTokenSet from Redis. Returns None on miss or error."""
        if self._redis is None:
            return None
        try:
            data = await self._redis.get_json(self._cache_key())
            if data is None:
                return None
            return OAuthTokenSet.model_validate(data)
        except Exception as exc:
            self._logger.debug("token_cache_read_error", error=str(exc))
            return None

    async def _write_token_cache(self, token_set: OAuthTokenSet) -> None:
        """Write an OAuthTokenSet to Redis with TTL = remaining_seconds - guard."""
        if self._redis is None:
            return
        ttl = max(1, int(token_set.remaining_seconds) - _CACHE_EXPIRY_GUARD_S)
        try:
            await self._redis.set_json(
                self._cache_key(),
                token_set.model_dump(mode="json"),
                ttl=ttl,
            )
        except Exception as exc:
            # Cache write failure is non-fatal — the vault is the source of truth.
            self._logger.debug("token_cache_write_error", error=str(exc))

    async def _invalidate_token_cache(self) -> None:
        """Remove this connector's cached token (call after revoke / error)."""
        if self._redis is None:
            return
        with contextlib.suppress(Exception):
            await self._redis.delete(self._cache_key())

    # ─── Refresh Lock ─────────────────────────────────────────────────

    def _get_refresh_lock(self) -> asyncio.Lock:
        """
        Return (or create) the process-wide asyncio.Lock for this platform_id.

        Keyed by platform_id so concurrent connectors for the same platform
        share one lock, but different platforms don't block each other.
        """
        if self.platform_id not in _REFRESH_LOCKS:
            _REFRESH_LOCKS[self.platform_id] = asyncio.Lock()
        return _REFRESH_LOCKS[self.platform_id]

    # ─── Health ──────────────────────────────────────────────────────

    def health_report(self) -> ConnectorHealthReport:
        """Generate a health snapshot for Thymos/monitoring."""
        creds = self._credentials
        return ConnectorHealthReport(
            connector_id=creds.connector_id if creds else f"{self.platform_id}:unconfigured",
            platform_id=self.platform_id,
            status=self.status,
            refresh_failure_count=creds.refresh_failure_count if creds else 0,
            last_refresh_at=creds.last_refresh_at if creds else None,
        )

    async def check_health(self) -> ConnectorHealthReport:
        """
        Perform a live health probe and update consecutive failure count.

        A connector is considered healthy if:
          - It has active credentials, AND
          - The stored token has ≥ 60 s remaining lifetime OR can be refreshed.

        After _HEALTH_FAILURE_THRESHOLD (3) consecutive failures, emits a
        SYSTEM_DEGRADED SynapseEvent so Thymos can quarantine this connector
        and alert the human operator.

        On recovery (any success), the counter resets and a CONNECTOR_TOKEN_REFRESHED
        event is emitted.
        """
        report = self.health_report()

        # Determine liveness: can we provide a valid token right now?
        token = await self.get_access_token()
        is_healthy = token is not None

        if is_healthy:
            self._consecutive_health_failures = 0
            report.status = ConnectorStatus.ACTIVE
        else:
            self._consecutive_health_failures += 1
            self._logger.warning(
                "connector_health_check_failed",
                platform_id=self.platform_id,
                consecutive_failures=self._consecutive_health_failures,
            )
            if self._consecutive_health_failures >= _HEALTH_FAILURE_THRESHOLD:
                await self._emit_degraded()

        return report

    async def _emit_degraded(self) -> None:
        """Emit SYSTEM_DEGRADED so Thymos can quarantine this connector."""
        creds = self._credentials
        connector_id = creds.connector_id if creds else f"{self.platform_id}:unconfigured"

        self._logger.error(
            "connector_degraded",
            connector_id=connector_id,
            platform_id=self.platform_id,
            consecutive_failures=self._consecutive_health_failures,
        )

        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            event = SynapseEvent(
                event_type=SynapseEventType.SYSTEM_DEGRADED,
                source_system="identity",
                data={
                    "connector_id": connector_id,
                    "platform_id": self.platform_id,
                    "consecutive_health_failures": self._consecutive_health_failures,
                    "action": "quarantine_requested",
                },
            )
            await self._event_bus.emit(event)
        except Exception as exc:
            self._logger.warning("degraded_event_emit_failed", error=str(exc))

    # ─── Event Emission ──────────────────────────────────────────────

    async def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a Synapse event if the event bus is wired."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            type_map: dict[str, SynapseEventType] = {
                "connector_authenticated": SynapseEventType.CONNECTOR_AUTHENTICATED,
                "connector_token_refreshed": SynapseEventType.CONNECTOR_TOKEN_REFRESHED,
                "connector_token_expired": SynapseEventType.CONNECTOR_TOKEN_EXPIRED,
                "connector_revoked": SynapseEventType.CONNECTOR_REVOKED,
                "connector_error": SynapseEventType.CONNECTOR_ERROR,
                "system_degraded": SynapseEventType.SYSTEM_DEGRADED,
            }
            evt_type = type_map.get(event_type)
            if evt_type is None:
                return

            event = SynapseEvent(
                event_type=evt_type,
                source_system="identity",
                data=data,
            )
            await self._event_bus.emit(event)
        except Exception as exc:
            self._logger.warning("event_emit_failed", event=event_type, error=str(exc))


# ─── Error Types ────────────────────────────────────────────────────────


class PlatformAuthError(Exception):
    """Raised when a platform authentication operation fails."""

    def __init__(
        self,
        message: str,
        platform_id: str = "",
        http_status: int = 0,
        response_body: str = "",
    ) -> None:
        super().__init__(message)
        self.platform_id = platform_id
        self.http_status = http_status
        self.response_body = response_body
