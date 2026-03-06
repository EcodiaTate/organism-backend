"""
EcodiaOS — SACM Encryption Utilities

Provides end-to-end encryption for workload payloads dispatched to
untrusted remote compute providers.  The protocol:

  1. Each party generates an ephemeral X25519 keypair.
  2. ECDH derives a shared secret from (our_private, their_public).
  3. The shared secret is fed through HKDF-SHA256 to derive a
     256-bit AES-GCM key.
  4. AES-256-GCM encrypts the payload with a random 96-bit nonce.
  5. The encrypted envelope bundles (nonce ‖ ciphertext ‖ tag) plus
     the ephemeral public key so the recipient can derive the same
     shared secret.

Why X25519+AES-GCM:
  - X25519 gives forward secrecy per workload (ephemeral keys).
  - AES-GCM provides authenticated encryption (confidentiality + integrity).
  - Standard `cryptography` library — FIPS-capable, no exotic deps.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import structlog
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from primitives.common import EOSBaseModel

logger = structlog.get_logger("systems.sacm.encryption")

# ─── Constants ──────────────────────────────────────────────────

_AES_KEY_BYTES = 32        # AES-256
_NONCE_BYTES = 12          # 96-bit GCM nonce
_HKDF_INFO = b"sacm-workload-encryption-v1"


# ─── Key Material ───────────────────────────────────────────────


class X25519KeyPair(NamedTuple):
    """Ephemeral X25519 keypair for a single workload exchange."""

    private_key: X25519PrivateKey
    public_key: X25519PublicKey
    public_bytes: bytes  # 32-byte raw public key


def generate_keypair() -> X25519KeyPair:
    """Generate a fresh ephemeral X25519 keypair."""
    private = X25519PrivateKey.generate()
    public = private.public_key()
    raw_public = public.public_bytes_raw()
    return X25519KeyPair(
        private_key=private,
        public_key=public,
        public_bytes=raw_public,
    )


def public_key_from_bytes(raw: bytes) -> X25519PublicKey:
    """Reconstruct an X25519 public key from 32 raw bytes."""
    if len(raw) != 32:
        raise ValueError(f"X25519 public key must be 32 bytes, got {len(raw)}")
    return X25519PublicKey.from_public_bytes(raw)


# ─── Key Derivation ────────────────────────────────────────────


def derive_shared_key(
    our_private: X25519PrivateKey,
    their_public: X25519PublicKey,
    salt: bytes | None = None,
) -> bytes:
    """
    Derive a 256-bit AES key from an X25519 ECDH shared secret.

    Uses HKDF-SHA256 with an optional salt (defaults to empty).
    The info parameter is fixed to scope the derived key to SACM
    workload encryption.
    """
    shared_secret = our_private.exchange(their_public)
    hkdf = HKDF(
        algorithm=SHA256(),
        length=_AES_KEY_BYTES,
        salt=salt,
        info=_HKDF_INFO,
    )
    return hkdf.derive(shared_secret)


# ─── Encrypted Envelope ────────────────────────────────────────


class EncryptedEnvelope(EOSBaseModel):
    """
    Wire format for an encrypted workload payload.

    Carries everything the recipient needs to decrypt:
      - sender_public_key: 32-byte X25519 ephemeral public key
      - nonce: 12-byte AES-GCM nonce
      - ciphertext: AES-256-GCM encrypted payload (includes 16-byte auth tag)
      - aad_hex: hex-encoded additional authenticated data (if any)
    """

    sender_public_key: bytes
    nonce: bytes
    ciphertext: bytes
    aad_hex: str = ""

    model_config = {"arbitrary_types_allowed": True}


class EncryptionMeta(EOSBaseModel):
    """Metadata about an encryption operation, for audit logging."""

    plaintext_size_bytes: int = 0
    ciphertext_size_bytes: int = 0
    algorithm: str = "X25519-HKDF-SHA256/AES-256-GCM"
    key_derivation: str = "HKDF-SHA256"
    nonce_bytes: int = _NONCE_BYTES
    aad_present: bool = False


# ─── Encrypt / Decrypt ──────────────────────────────────────────


class EncryptionResult(NamedTuple):
    """Returned by encrypt_payload: the envelope plus audit metadata."""

    envelope: EncryptedEnvelope
    meta: EncryptionMeta


def encrypt_payload(
    plaintext: bytes,
    recipient_public: X25519PublicKey,
    aad: bytes | None = None,
    salt: bytes | None = None,
) -> EncryptionResult:
    """
    Encrypt a workload payload for a remote provider.

    Generates an ephemeral X25519 keypair, derives a shared AES-256
    key via HKDF, and seals the payload with AES-GCM.

    Args:
        plaintext:        Raw workload bytes to encrypt.
        recipient_public: The remote provider's X25519 public key.
        aad:              Optional additional authenticated data
                          (e.g. workload ID) — authenticated but not
                          encrypted.  Must be supplied at decrypt time.
        salt:             Optional HKDF salt for domain separation.

    Returns:
        EncryptionResult with the sealed envelope and audit metadata.
    """
    if not plaintext:
        raise ValueError("Cannot encrypt empty payload")

    # Ephemeral keypair — fresh per workload for forward secrecy
    ephemeral = generate_keypair()

    # Derive symmetric key
    aes_key = derive_shared_key(ephemeral.private_key, recipient_public, salt)

    # Encrypt with AES-256-GCM
    nonce = os.urandom(_NONCE_BYTES)
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

    envelope = EncryptedEnvelope(
        sender_public_key=ephemeral.public_bytes,
        nonce=nonce,
        ciphertext=ciphertext,
        aad_hex=aad.hex() if aad else "",
    )
    meta = EncryptionMeta(
        plaintext_size_bytes=len(plaintext),
        ciphertext_size_bytes=len(ciphertext),
        aad_present=aad is not None,
    )

    logger.debug(
        "payload_encrypted",
        plaintext_bytes=len(plaintext),
        ciphertext_bytes=len(ciphertext),
        aad_present=aad is not None,
    )
    return EncryptionResult(envelope=envelope, meta=meta)


def decrypt_payload(
    envelope: EncryptedEnvelope,
    recipient_private: X25519PrivateKey,
    aad: bytes | None = None,
    salt: bytes | None = None,
) -> bytes:
    """
    Decrypt a workload payload received from a remote provider.

    Reconstructs the shared secret from (our_private, sender_public),
    derives the same AES key, and opens the GCM ciphertext.

    Args:
        envelope:          The EncryptedEnvelope from the sender.
        recipient_private: Our X25519 private key.
        aad:               The same AAD used at encryption time
                           (or None if none was used).
        salt:              The same HKDF salt used at encryption time.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        cryptography.exceptions.InvalidTag: if the ciphertext was
            tampered with or the wrong key/AAD was used.
    """
    sender_public = public_key_from_bytes(envelope.sender_public_key)
    aes_key = derive_shared_key(recipient_private, sender_public, salt)
    aesgcm = AESGCM(aes_key)
    plaintext = aesgcm.decrypt(envelope.nonce, envelope.ciphertext, aad)

    logger.debug(
        "payload_decrypted",
        ciphertext_bytes=len(envelope.ciphertext),
        plaintext_bytes=len(plaintext),
    )
    return plaintext


# ─── Convenience: Symmetric-only (for local canary verification) ──


class SymmetricSealedBox(EOSBaseModel):
    """Lightweight sealed box for local symmetric encryption (canary payloads)."""

    nonce: bytes
    ciphertext: bytes
    key_id: str = ""

    model_config = {"arbitrary_types_allowed": True}


def symmetric_encrypt(plaintext: bytes, key: bytes) -> SymmetricSealedBox:
    """Encrypt with a pre-shared 256-bit key (AES-256-GCM). Used for canary sealing."""
    if len(key) != _AES_KEY_BYTES:
        raise ValueError(f"Key must be {_AES_KEY_BYTES} bytes, got {len(key)}")
    nonce = os.urandom(_NONCE_BYTES)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return SymmetricSealedBox(nonce=nonce, ciphertext=ct)


def symmetric_decrypt(box: SymmetricSealedBox, key: bytes) -> bytes:
    """Decrypt a SymmetricSealedBox with the same 256-bit key."""
    if len(key) != _AES_KEY_BYTES:
        raise ValueError(f"Key must be {_AES_KEY_BYTES} bytes, got {len(key)}")
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(box.nonce, box.ciphertext, None)


def generate_symmetric_key() -> bytes:
    """Generate a random 256-bit AES key."""
    return AESGCM.generate_key(bit_length=256)
