"""
EcodiaOS - TOTP Generator (Phase 16h: External Identity Layer)

RFC 6238-compliant Time-Based One-Time Password generator for automated
2FA authentication against external platforms.

This is a native implementation - no dependency on pyotp or similar
libraries. The algorithm is straightforward (HMAC-SHA1 over a time-based
counter, truncated to 6 digits) and small enough to own entirely.

Usage:
    from systems.identity.totp import TOTPGenerator

    gen = TOTPGenerator(base32_secret="JBSWY3DPEHPK3PXP")
    code = gen.now()           # "482731"
    valid = gen.verify("482731")  # True (within window)

Integration with IdentityVault:
    The base32 secret is stored encrypted via vault.encrypt_totp_secret().
    At 2FA time, decrypt and construct a TOTPGenerator:

        secret = vault.decrypt_totp_secret(envelope)
        code = TOTPGenerator(secret).now()

Event integration:
    TOTPGenerator itself is stateless and emits no events. The calling
    connector emits CONNECTOR_AUTHENTICATED after successful 2FA.

Thread-safety: Fully thread-safe (stateless computation from inputs).
"""

from __future__ import annotations

import base64
import hmac
import struct
import time

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel

logger = structlog.get_logger("identity.totp")


# ─── Configuration ──────────────────────────────────────────────────────


class TOTPConfig(EOSBaseModel):
    """TOTP generation parameters (RFC 6238 defaults)."""

    digits: int = Field(default=6, ge=6, le=8)
    """Number of digits in the OTP (6 or 8)."""

    period: int = Field(default=30, ge=15, le=60)
    """Time step in seconds."""

    algorithm: str = "sha1"
    """HMAC algorithm. SHA-1 is the RFC 6238 default and what most
    providers use. SHA-256 and SHA-512 are supported but rare."""

    skew: int = Field(default=1, ge=0, le=2)
    """Number of time steps to check before/after current time for
    clock drift tolerance during verify()."""


# ─── Generator ──────────────────────────────────────────────────────────


class TOTPGenerator:
    """
    RFC 6238 TOTP generator and verifier.

    Stateless: takes a base32-encoded secret and produces time-based
    one-time passwords. Each call to now() computes fresh from the
    current wall-clock time.
    """

    def __init__(
        self,
        base32_secret: str,
        config: TOTPConfig | None = None,
    ) -> None:
        self._config = config or TOTPConfig()

        # Decode the base32 secret (strip whitespace and padding)
        cleaned = base32_secret.strip().upper().replace(" ", "")
        # Add padding if necessary (base32 requires length % 8 == 0)
        padding = (8 - len(cleaned) % 8) % 8
        cleaned += "=" * padding
        self._secret = base64.b32decode(cleaned)

        self._hash_func = self._resolve_hash(self._config.algorithm)

    def now(self) -> str:
        """Generate the current TOTP code."""
        return self.at(time.time())

    def at(self, timestamp: float) -> str:
        """Generate the TOTP code for a specific Unix timestamp."""
        counter = int(timestamp) // self._config.period
        return self._generate(counter)

    def verify(
        self,
        code: str,
        timestamp: float | None = None,
    ) -> bool:
        """
        Verify a TOTP code against the current time (or a given timestamp).

        Checks the current time step and `skew` steps before/after to
        tolerate clock drift between EOS and the authenticating platform.

        Returns True if the code matches any valid window.
        """
        ts = timestamp if timestamp is not None else time.time()
        counter = int(ts) // self._config.period

        for offset in range(-self._config.skew, self._config.skew + 1):
            expected = self._generate(counter + offset)
            if hmac.compare_digest(code, expected):
                return True

        return False

    def remaining_seconds(self) -> int:
        """Seconds until the current code expires."""
        return self._config.period - (int(time.time()) % self._config.period)

    # ─── Internal ────────────────────────────────────────────────────

    def _generate(self, counter: int) -> str:
        """
        Core HOTP generation (RFC 4226 section 5.3).

        1. Pack the counter as a big-endian 64-bit integer.
        2. HMAC-SHA1 (or SHA-256/512) over the secret + counter.
        3. Dynamic truncation to extract a 31-bit integer.
        4. Modulo 10^digits to get the final OTP.
        """
        # Step 1: counter to 8-byte big-endian
        counter_bytes = struct.pack(">Q", counter)

        # Step 2: HMAC
        mac = hmac.new(self._secret, counter_bytes, self._hash_func)
        digest = mac.digest()

        # Step 3: dynamic truncation (RFC 4226 section 5.4)
        offset = digest[-1] & 0x0F
        truncated = struct.unpack(">I", digest[offset : offset + 4])[0]
        truncated &= 0x7FFFFFFF  # Mask to 31 bits (remove sign bit)

        # Step 4: modulo reduction
        otp = truncated % (10 ** self._config.digits)

        # Zero-pad to the configured digit count
        return str(otp).zfill(self._config.digits)

    @staticmethod
    def _resolve_hash(algorithm: str) -> str:
        """Map algorithm name to hashlib name."""
        mapping = {
            "sha1": "sha1",
            "sha256": "sha256",
            "sha512": "sha512",
        }
        resolved = mapping.get(algorithm.lower())
        if resolved is None:
            raise ValueError(
                f"Unsupported TOTP algorithm: {algorithm!r}. "
                f"Supported: {', '.join(mapping.keys())}"
            )
        return resolved


# ─── Convenience Functions ──────────────────────────────────────────────


def generate_totp(base32_secret: str) -> str:
    """One-shot: generate the current 6-digit TOTP code from a base32 secret."""
    return TOTPGenerator(base32_secret).now()


def verify_totp(base32_secret: str, code: str) -> bool:
    """One-shot: verify a TOTP code against a base32 secret."""
    return TOTPGenerator(base32_secret).verify(code)
