"""
EcodiaOS - Identity Vault (Phase 16h: External Identity Layer)

Encrypts sensitive credential material at rest using Fernet (AES-128-CBC +
HMAC-SHA256). Every secret stored through this vault - OAuth tokens, TOTP
base32 seeds, serialised browser cookie jars - is encrypted before it
leaves process memory.

Design decisions:
  - Fernet chosen over raw AES because it bundles authentication (HMAC),
    versioning, and timestamping, making misuse harder.
  - The master key is derived from a high-entropy passphrase via PBKDF2
    (600 000 iterations, SHA-256) so the system can be bootstrapped from a
    single env var (ECODIAOS_VAULT_PASSPHRASE).
  - Each encrypt() call produces a fresh IV - ciphertext is never
    deterministic, defeating frequency analysis.
  - The vault emits Synapse events on encrypt/decrypt failures and key
    rotation so downstream monitors (Thymos, audit) can react.

Thread-safety: NOT thread-safe. Single-threaded asyncio like all EOS.
"""

from __future__ import annotations

import asyncio
import base64
import secrets
from typing import TYPE_CHECKING, Any
from datetime import datetime

import structlog
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, utc_now

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus
    from systems.synapse.types import SynapseEvent, SynapseEventType

logger = structlog.get_logger("identity.vault")

# Vault ID used in event payloads when no per-instance ID is defined
_VAULT_SINGLETON_ID = "identity_vault"


# ─── Sealed Envelope (encrypted-at-rest container) ───────────────────────


class SealedEnvelope(Identified, Timestamped):
    """
    An encrypted credential blob with metadata.

    The ciphertext is Fernet-encrypted (base64-urlsafe output). The
    purpose and platform fields are plaintext metadata so the vault can
    index and retrieve envelopes without decrypting them.
    """

    platform_id: str
    """Which platform connector owns this secret (e.g. 'google', 'github')."""

    purpose: str
    """What kind of secret this is: 'oauth_token', 'totp_secret', 'cookie_state'."""

    ciphertext: str
    """Fernet-encrypted, base64-urlsafe-encoded secret material."""

    key_version: int = 1
    """Which master-key generation encrypted this envelope."""

    last_accessed_at: datetime | None = None
    """Timestamp of last successful decrypt (for staleness detection)."""


# ─── Vault Configuration ────────────────────────────────────────────────


class VaultConfig(EOSBaseModel):
    """Configuration for the IdentityVault."""

    pbkdf2_iterations: int = Field(default=600_000, ge=100_000)
    """PBKDF2 iteration count. 600k is OWASP 2024 recommendation for SHA-256."""

    salt_bytes: int = Field(default=16, ge=16)
    """Salt length in bytes. 16 = 128-bit, matching Fernet's AES-128."""


# ─── Vault Event (Synapse payload model) ────────────────────────────────


class VaultEvent(EOSBaseModel):
    """Structured payload for vault Synapse events."""

    event_type: str
    """One of: decrypt_failure | key_rotation_started | key_rotation_complete | key_rotation_failed"""

    timestamp: datetime
    """UTC time the event occurred."""

    vault_id: str
    """Identifier of the vault instance."""

    error_type: str = ""
    """For decrypt_failure: key_mismatch | tampered | unknown. Empty for rotation events."""

    severity: str = "medium"
    """low | medium | high | critical"""


# ─── The Vault ──────────────────────────────────────────────────────────


class IdentityVault:
    """
    Encrypts and decrypts sensitive credential material using Fernet.

    The vault derives a 256-bit Fernet key from a passphrase + salt via
    PBKDF2-HMAC-SHA256. Callers store SealedEnvelopes (containing the
    ciphertext) in the database - the vault never touches persistence
    directly.

    Lifecycle:
      1. Construct with passphrase (from env/config) and optional salt.
      2. Call encrypt() to seal plaintext → SealedEnvelope.
      3. Call decrypt() to unseal SealedEnvelope → plaintext bytes.
      4. Optionally call rotate_key() to re-encrypt all envelopes under
         a new passphrase.
    """

    def __init__(
        self,
        passphrase: str,
        salt: bytes | None = None,
        config: VaultConfig | None = None,
    ) -> None:
        if not passphrase:
            raise ValueError("Vault passphrase must not be empty")

        self._config = config or VaultConfig()
        self._salt = salt or secrets.token_bytes(self._config.salt_bytes)
        self._key_version: int = 1
        self._event_bus: EventBus | None = None
        self._logger = logger.bind(component="identity_vault")

        # Derive the Fernet key from passphrase + salt
        self._fernet = self._derive_fernet(passphrase, self._salt)

        self._logger.info(
            "vault_initialized",
            pbkdf2_iterations=self._config.pbkdf2_iterations,
            key_version=self._key_version,
        )

    # ─── Public API ──────────────────────────────────────────────────

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire Synapse event bus for vault lifecycle events."""
        self._event_bus = event_bus

    # ─── Internal Event Helpers ───────────────────────────────────────

    def _fire_event(self, event_type_name: str, payload: dict[str, Any]) -> None:
        """Fire-and-forget a Synapse event from a synchronous context."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            event = SynapseEvent(
                type=SynapseEventType[event_type_name],
                data=payload,
                source="identity_vault",
            )
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._event_bus.emit(event))
        except Exception:
            # Never let event emission crash a crypto operation
            self._logger.warning("vault_event_emission_failed", event_type=event_type_name)

    @property
    def salt(self) -> bytes:
        """The PBKDF2 salt. Must be persisted alongside the ciphertext."""
        return self._salt

    @property
    def key_version(self) -> int:
        return self._key_version

    def encrypt(
        self,
        plaintext: bytes,
        platform_id: str,
        purpose: str,
    ) -> SealedEnvelope:
        """
        Encrypt plaintext bytes and return a SealedEnvelope.

        Args:
            plaintext: The raw secret material (token JSON, base32 seed,
                       serialised cookie jar, etc.)
            platform_id: Owning platform connector identifier.
            purpose: Category - 'oauth_token', 'totp_secret', 'cookie_state'.

        Returns:
            A SealedEnvelope ready for database persistence.
        """
        ciphertext_bytes = self._fernet.encrypt(plaintext)
        ciphertext_str = ciphertext_bytes.decode("ascii")

        envelope = SealedEnvelope(
            platform_id=platform_id,
            purpose=purpose,
            ciphertext=ciphertext_str,
            key_version=self._key_version,
        )

        self._logger.debug(
            "secret_encrypted",
            platform_id=platform_id,
            purpose=purpose,
            envelope_id=envelope.id,
        )
        return envelope

    def decrypt(self, envelope: SealedEnvelope) -> bytes:
        """
        Decrypt a SealedEnvelope and return the plaintext bytes.

        Raises:
            InvalidToken: If the ciphertext was tampered with, the key is
                          wrong, or the Fernet token is malformed.
        """
        try:
            plaintext = self._fernet.decrypt(envelope.ciphertext.encode("ascii"))
        except InvalidToken as exc:
            # Distinguish key-version mismatch from general tamper/malform
            error_type = (
                "key_mismatch"
                if envelope.key_version != self._key_version
                else "tampered"
            )
            self._logger.error(
                "decrypt_failed",
                envelope_id=envelope.id,
                platform_id=envelope.platform_id,
                purpose=envelope.purpose,
                error_type=error_type,
            )
            self._fire_event(
                "VAULT_DECRYPT_FAILED",
                {
                    "vault_id": _VAULT_SINGLETON_ID,
                    "envelope_id": envelope.id,
                    "platform_id": envelope.platform_id,
                    "error_type": error_type,
                    "key_version": envelope.key_version,
                    "error": str(exc) or "InvalidToken",
                },
            )
            raise

        envelope.last_accessed_at = utc_now()

        self._logger.debug(
            "secret_decrypted",
            platform_id=envelope.platform_id,
            purpose=envelope.purpose,
            envelope_id=envelope.id,
        )
        return plaintext

    def rotate_key(
        self,
        new_passphrase: str,
        envelopes: list[SealedEnvelope],
        new_salt: bytes | None = None,
    ) -> list[SealedEnvelope]:
        """
        Re-encrypt all envelopes under a new passphrase.

        1. Decrypt each envelope with the current key.
        2. Derive a new Fernet key from the new passphrase.
        3. Re-encrypt each plaintext with the new key.

        Returns a list of new SealedEnvelopes (same IDs, new ciphertext +
        incremented key_version). The caller is responsible for persisting
        the updated envelopes atomically.

        Raises InvalidToken if any envelope cannot be decrypted.
        """
        previous_version = self._key_version

        self._fire_event(
            "VAULT_KEY_ROTATION_STARTED",
            {
                "vault_id": _VAULT_SINGLETON_ID,
                "previous_key_version": previous_version,
                "envelope_count": len(envelopes),
                "timestamp": utc_now().isoformat(),
            },
        )

        try:
            # Phase 1: decrypt all under current key
            plaintexts: list[tuple[SealedEnvelope, bytes]] = []
            for env in envelopes:
                pt = self.decrypt(env)
                plaintexts.append((env, pt))

            # Phase 2: derive new key
            salt = new_salt or secrets.token_bytes(self._config.salt_bytes)
            new_fernet = self._derive_fernet(new_passphrase, salt)
            new_version = self._key_version + 1

            # Phase 3: re-encrypt
            rotated: list[SealedEnvelope] = []
            for old_env, pt in plaintexts:
                ct = new_fernet.encrypt(pt).decode("ascii")
                new_env = SealedEnvelope(
                    id=old_env.id,
                    platform_id=old_env.platform_id,
                    purpose=old_env.purpose,
                    ciphertext=ct,
                    key_version=new_version,
                    created_at=old_env.created_at,
                )
                rotated.append(new_env)

            # Phase 4: swap to new key
            self._fernet = new_fernet
            self._salt = salt
            self._key_version = new_version

            self._logger.info(
                "vault_key_rotated",
                key_version=new_version,
                envelopes_rotated=len(rotated),
            )
            self._fire_event(
                "VAULT_KEY_ROTATION_COMPLETE",
                {
                    "vault_id": _VAULT_SINGLETON_ID,
                    "new_key_version": new_version,
                    "envelopes_rotated": len(rotated),
                    "timestamp": utc_now().isoformat(),
                },
            )
            return rotated

        except Exception as exc:
            self._logger.error(
                "vault_key_rotation_failed",
                previous_key_version=previous_version,
                error=str(exc),
            )
            self._fire_event(
                "VAULT_KEY_ROTATION_FAILED",
                {
                    "vault_id": _VAULT_SINGLETON_ID,
                    "previous_key_version": previous_version,
                    "error": str(exc),
                    "timestamp": utc_now().isoformat(),
                },
            )
            raise

    # ─── Convenience Methods ─────────────────────────────────────────

    def encrypt_token_json(
        self,
        token_data: dict[str, Any],
        platform_id: str,
    ) -> SealedEnvelope:
        """Encrypt an OAuth token dict (access_token, refresh_token, etc.)."""
        import json

        raw = json.dumps(token_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self.encrypt(raw, platform_id=platform_id, purpose="oauth_token")

    def decrypt_token_json(self, envelope: SealedEnvelope) -> dict[str, Any]:
        """Decrypt a SealedEnvelope back to an OAuth token dict."""
        import json

        raw = self.decrypt(envelope)
        result: dict[str, Any] = json.loads(raw)
        return result

    def encrypt_totp_secret(
        self,
        base32_secret: str,
        platform_id: str,
    ) -> SealedEnvelope:
        """Encrypt a TOTP base32 seed."""
        return self.encrypt(
            base32_secret.encode("ascii"),
            platform_id=platform_id,
            purpose="totp_secret",
        )

    def decrypt_totp_secret(self, envelope: SealedEnvelope) -> str:
        """Decrypt a SealedEnvelope back to a TOTP base32 seed string."""
        return self.decrypt(envelope).decode("ascii")

    def encrypt_cookie_state(
        self,
        serialised_cookies: bytes,
        platform_id: str,
    ) -> SealedEnvelope:
        """Encrypt a serialised browser cookie jar (JSON bytes)."""
        return self.encrypt(
            serialised_cookies,
            platform_id=platform_id,
            purpose="cookie_state",
        )

    def decrypt_cookie_state(self, envelope: SealedEnvelope) -> bytes:
        """Decrypt a SealedEnvelope back to raw cookie state bytes."""
        return self.decrypt(envelope)

    # ─── Internal ────────────────────────────────────────────────────

    def _derive_fernet(self, passphrase: str, salt: bytes) -> Fernet:
        """Derive a Fernet key from passphrase + salt via PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self._config.pbkdf2_iterations,
        )
        key_material = kdf.derive(passphrase.encode("utf-8"))
        # Fernet requires a 32-byte key encoded as urlsafe base64
        fernet_key = base64.urlsafe_b64encode(key_material)
        return Fernet(fernet_key)
