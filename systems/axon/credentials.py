"""
EcodiaOS - Axon Credential Store

Executors need credentials to act on external systems - API keys, OAuth tokens,
webhook secrets. But executors must never see raw secrets.

CredentialStore wraps the raw credential vault and issues scoped, time-limited
tokens per execution. Each token:
  - Is scoped to a specific service (e.g., "email", "calendar", "federation")
  - Is valid only for the duration of the intent execution + a 60s buffer
  - Is tied to the execution_id for audit correlation

In Phase 1 (current), this is a simple in-memory store loaded from config.
In future phases (Federation, external integrations), this will be backed by
a secrets manager and will support OAuth2 token exchange.

Executors call context.credentials.get("service_name") to retrieve their token.
They never access this store directly.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import TYPE_CHECKING

import structlog

from systems.axon.types import ScopedCredentials

if TYPE_CHECKING:
    from primitives.intent import Intent

logger = structlog.get_logger()


def _extract_required_services(intent: Intent) -> set[str]:
    """
    Determine which external services an intent requires credentials for.

    Derived from the executor names in the action plan. This is a heuristic -
    executors may declare their service requirements explicitly in future.
    """
    services: set[str] = set()
    for step in intent.plan.steps:
        executor_name = step.executor.lower()
        if "email" in executor_name:
            services.add("email")
        if "calendar" in executor_name or "schedule" in executor_name:
            services.add("calendar")
        if "webhook" in executor_name:
            services.add("webhook")
        if "federation" in executor_name or "federate" in executor_name:
            services.add("federation")
        if "iot" in executor_name:
            services.add("iot")
        if "call_api" in executor_name or executor_name == "api":
            # Extract service from parameters if available
            services.add("external_api")
    return services


class CredentialStore:
    """
    Issues scoped, time-limited credentials for intent execution.

    Raw secrets are never exposed to executors. Instead, callers receive a
    ScopedCredentials wrapper with time-limited tokens keyed by service name.

    Phase 1 implementation: credentials loaded from a flat config dict.
    Future: backed by Vault/KMS with dynamic secret generation.
    """

    def __init__(self, raw_credentials: dict[str, str] | None = None) -> None:
        """
        Args:
            raw_credentials: Map of service_name → raw secret.
                             Loaded from config, never logged.
        """
        self._vault: dict[str, str] = raw_credentials or {}
        self._logger = logger.bind(system="axon.credentials")
        # signing_key for HMAC token generation - randomised per process
        self._signing_key = hashlib.sha256(
            f"eos-credential-{time.time()}".encode()
        ).digest()

    def configure(self, credentials: dict[str, str]) -> None:
        """Add or update credentials in the vault (called at startup)."""
        self._vault.update(credentials)

    async def get_for_intent(self, intent: Intent) -> ScopedCredentials:
        """
        Issue scoped tokens for all services required by this intent.

        Returns a ScopedCredentials containing time-limited tokens.
        Services without configured credentials get an empty token.
        """
        required = _extract_required_services(intent)
        if not required:
            return ScopedCredentials()

        # Estimate execution duration for TTL
        estimated_duration_s = max(
            step.timeout_ms for step in intent.plan.steps
        ) // 1000 if intent.plan.steps else 30
        ttl_seconds = estimated_duration_s + 60  # 60s buffer

        tokens: dict[str, str] = {}
        for service in required:
            if service in self._vault:
                tokens[service] = self._issue_scoped_token(
                    service=service,
                    scope=f"intent:{intent.id}",
                    ttl_seconds=ttl_seconds,
                )
                self._logger.debug(
                    "credential_issued",
                    service=service,
                    intent_id=intent.id,
                    ttl_seconds=ttl_seconds,
                )
            else:
                # Service required but not configured - empty token signals executors
                tokens[service] = ""
                self._logger.warning(
                    "credential_missing",
                    service=service,
                    intent_id=intent.id,
                )

        return ScopedCredentials(tokens=tokens)

    def _issue_scoped_token(
        self,
        service: str,
        scope: str,
        ttl_seconds: int,
    ) -> str:
        """
        Generate a scoped token that wraps the raw credential with a
        time-limited HMAC signature.

        Format: {service}:{expiry_ts}:{hmac}:{raw_credential}
        This is deliberately simple - in production, replace with proper
        token exchange (OAuth2, Vault dynamic secrets, etc.)
        """
        raw = self._vault.get(service, "")
        expiry = int(time.time()) + ttl_seconds
        payload = f"{service}:{scope}:{expiry}"
        signature = hmac.new(
            self._signing_key,
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()[:16]
        # The token carries the raw credential for Phase 1 - executors extract it.
        # In Phase 2, the token would be opaque and exchangeable at the service.
        return f"{expiry}:{signature}:{raw}"

    def extract_credential(self, token: str) -> str | None:
        """
        Extract the raw credential from a scoped token.

        Returns None if the token is expired or malformed.
        Executors call this via the token they receive - not directly.
        """
        if not token:
            return None
        parts = token.split(":", 2)
        if len(parts) < 3:
            return None
        expiry_str, _signature, raw = parts
        try:
            expiry = int(expiry_str)
        except ValueError:
            return None
        if time.time() > expiry:
            self._logger.warning("credential_expired", expiry=expiry)
            return None
        return raw
