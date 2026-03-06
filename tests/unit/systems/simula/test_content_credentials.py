"""
Unit tests for Simula ContentCredentialManager (Stage 6A.2).

Tests C2PA content credential signing, verification, batch verification,
unsigned-file handling, and manifest structure.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from systems.simula.audit.content_credentials import ContentCredentialManager
from systems.simula.verification.types import (
    ContentCredential,
    ContentCredentialResult,
    ContentCredentialStatus,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_file(tmp_path: Path, rel_path: str, content: str = "hello world") -> None:
    """Write a file at codebase_root / rel_path with the given content."""
    full = tmp_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")


def _make_ed25519_keypair():
    """Generate a fresh Ed25519 key pair using the cryptography library."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def _manager_with_key() -> ContentCredentialManager:
    """Return a ContentCredentialManager with an in-memory Ed25519 key pair."""
    private_key, public_key = _make_ed25519_keypair()
    mgr = ContentCredentialManager()
    mgr._private_key = private_key
    mgr._public_key = public_key
    return mgr


def _manager_without_key() -> ContentCredentialManager:
    """Return a ContentCredentialManager with no signing key."""
    return ContentCredentialManager()


# ── File Signing With Key ────────────────────────────────────────────────────


class TestSignFilesWithKey:
    @pytest.mark.asyncio
    async def test_sign_single_file(self, tmp_path: Path):
        _write_file(tmp_path, "src/main.py", "print('hello')")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["src/main.py"], tmp_path)

        assert isinstance(result, ContentCredentialResult)
        assert result.status == ContentCredentialStatus.SIGNED
        assert len(result.credentials) == 1
        assert result.unsigned_files == []
        assert result.verified_count == 0
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_signed_credential_has_nonempty_signature(self, tmp_path: Path):
        _write_file(tmp_path, "src/main.py", "print('hello')")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["src/main.py"], tmp_path)
        cred = result.credentials[0]

        assert cred.signature != ""
        assert len(cred.signature) > 0

    @pytest.mark.asyncio
    async def test_signed_credential_has_correct_hash(self, tmp_path: Path):
        content = "def add(a, b): return a + b"
        _write_file(tmp_path, "lib.py", content)
        mgr = _manager_with_key()

        result = await mgr.sign_files(["lib.py"], tmp_path)
        cred = result.credentials[0]

        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert cred.content_hash == expected_hash

    @pytest.mark.asyncio
    async def test_signed_credential_metadata(self, tmp_path: Path):
        _write_file(tmp_path, "app.py", "x = 1")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["app.py"], tmp_path)
        cred = result.credentials[0]

        assert cred.file_path == "app.py"
        assert cred.issuer == "EcodiaOS Simula"
        assert cred.algorithm == "Ed25519"
        assert cred.created_at is not None

    @pytest.mark.asyncio
    async def test_sign_multiple_files(self, tmp_path: Path):
        _write_file(tmp_path, "a.py", "a = 1")
        _write_file(tmp_path, "b.py", "b = 2")
        _write_file(tmp_path, "c.py", "c = 3")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["a.py", "b.py", "c.py"], tmp_path)

        assert result.status == ContentCredentialStatus.SIGNED
        assert len(result.credentials) == 3
        assert result.unsigned_files == []
        paths = [c.file_path for c in result.credentials]
        assert paths == ["a.py", "b.py", "c.py"]

    @pytest.mark.asyncio
    async def test_sign_with_custom_issuer(self, tmp_path: Path):
        _write_file(tmp_path, "f.py", "pass")
        private_key, public_key = _make_ed25519_keypair()
        mgr = ContentCredentialManager(issuer_name="CustomCorp CI")
        mgr._private_key = private_key
        mgr._public_key = public_key

        result = await mgr.sign_files(["f.py"], tmp_path)
        cred = result.credentials[0]

        assert cred.issuer == "CustomCorp CI"


# ── File Signing Without Key ─────────────────────────────────────────────────


class TestSignFilesWithoutKey:
    @pytest.mark.asyncio
    async def test_unsigned_when_no_key(self, tmp_path: Path):
        _write_file(tmp_path, "src/main.py", "print('hello')")
        mgr = _manager_without_key()

        result = await mgr.sign_files(["src/main.py"], tmp_path)

        assert result.status == ContentCredentialStatus.SIGNED
        assert len(result.credentials) == 1
        cred = result.credentials[0]
        assert cred.signature == ""

    @pytest.mark.asyncio
    async def test_unsigned_credential_still_has_hash(self, tmp_path: Path):
        content = "x = 42"
        _write_file(tmp_path, "mod.py", content)
        mgr = _manager_without_key()

        result = await mgr.sign_files(["mod.py"], tmp_path)
        cred = result.credentials[0]

        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert cred.content_hash == expected_hash
        assert cred.signature == ""

    @pytest.mark.asyncio
    async def test_unsigned_credential_has_manifest(self, tmp_path: Path):
        _write_file(tmp_path, "mod.py", "pass")
        mgr = _manager_without_key()

        result = await mgr.sign_files(["mod.py"], tmp_path)
        cred = result.credentials[0]

        assert cred.c2pa_manifest_json != ""
        manifest = json.loads(cred.c2pa_manifest_json)
        assert manifest["signature_info"]["algorithm"] == "Ed25519"


# ── Missing / Nonexistent Files ──────────────────────────────────────────────


class TestMissingFiles:
    @pytest.mark.asyncio
    async def test_missing_file_goes_to_unsigned(self, tmp_path: Path):
        mgr = _manager_with_key()

        result = await mgr.sign_files(["nonexistent.py"], tmp_path)

        assert result.status == ContentCredentialStatus.UNSIGNED
        assert len(result.credentials) == 0
        assert result.unsigned_files == ["nonexistent.py"]

    @pytest.mark.asyncio
    async def test_mix_existing_and_missing(self, tmp_path: Path):
        _write_file(tmp_path, "exists.py", "ok")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["exists.py", "gone.py"], tmp_path)

        assert result.status == ContentCredentialStatus.SIGNED
        assert len(result.credentials) == 1
        assert result.credentials[0].file_path == "exists.py"
        assert result.unsigned_files == ["gone.py"]

    @pytest.mark.asyncio
    async def test_empty_file_list(self, tmp_path: Path):
        mgr = _manager_with_key()

        result = await mgr.sign_files([], tmp_path)

        assert result.status == ContentCredentialStatus.UNSIGNED
        assert len(result.credentials) == 0
        assert result.unsigned_files == []

    @pytest.mark.asyncio
    async def test_all_files_missing_status_unsigned(self, tmp_path: Path):
        mgr = _manager_with_key()

        result = await mgr.sign_files(["a.py", "b.py"], tmp_path)

        assert result.status == ContentCredentialStatus.UNSIGNED
        assert len(result.credentials) == 0
        assert set(result.unsigned_files) == {"a.py", "b.py"}


# ── Signature Verification ───────────────────────────────────────────────────


class TestVerifyFile:
    @pytest.mark.asyncio
    async def test_verify_valid_signature(self, tmp_path: Path):
        content = "verified content"
        _write_file(tmp_path, "v.py", content)
        mgr = _manager_with_key()

        sign_result = await mgr.sign_files(["v.py"], tmp_path)
        cred = sign_result.credentials[0]

        verified = await mgr.verify_file("v.py", cred, tmp_path)
        assert verified is True

    @pytest.mark.asyncio
    async def test_verify_detects_tampered_content(self, tmp_path: Path):
        _write_file(tmp_path, "v.py", "original content")
        mgr = _manager_with_key()

        sign_result = await mgr.sign_files(["v.py"], tmp_path)
        cred = sign_result.credentials[0]

        # Tamper with the file after signing
        _write_file(tmp_path, "v.py", "tampered content")

        verified = await mgr.verify_file("v.py", cred, tmp_path)
        assert verified is False

    @pytest.mark.asyncio
    async def test_verify_detects_invalid_signature(self, tmp_path: Path):
        content = "some code"
        _write_file(tmp_path, "v.py", content)
        mgr = _manager_with_key()

        sign_result = await mgr.sign_files(["v.py"], tmp_path)
        cred = sign_result.credentials[0]

        # Forge a bad signature (same length hex, but wrong bytes)
        bad_sig = "aa" * (len(cred.signature) // 2)
        tampered_cred = ContentCredential(
            file_path=cred.file_path,
            content_hash=cred.content_hash,
            issuer=cred.issuer,
            signature=bad_sig,
            algorithm=cred.algorithm,
            c2pa_manifest_json=cred.c2pa_manifest_json,
            created_at=cred.created_at,
        )

        verified = await mgr.verify_file("v.py", tampered_cred, tmp_path)
        assert verified is False

    @pytest.mark.asyncio
    async def test_verify_missing_file_returns_false(self, tmp_path: Path):
        mgr = _manager_with_key()
        cred = ContentCredential(
            file_path="missing.py",
            content_hash="abc123",
            issuer="EcodiaOS Simula",
        )

        verified = await mgr.verify_file("missing.py", cred, tmp_path)
        assert verified is False

    @pytest.mark.asyncio
    async def test_verify_without_public_key_skips_sig_check(self, tmp_path: Path):
        """When there is no public key, verification checks only the hash."""
        content = "hash only check"
        _write_file(tmp_path, "h.py", content)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        mgr = _manager_without_key()
        cred = ContentCredential(
            file_path="h.py",
            content_hash=content_hash,
            issuer="EcodiaOS Simula",
            signature="deadbeef",
        )

        verified = await mgr.verify_file("h.py", cred, tmp_path)
        assert verified is True

    @pytest.mark.asyncio
    async def test_verify_hash_mismatch_without_key(self, tmp_path: Path):
        _write_file(tmp_path, "h.py", "real content")
        mgr = _manager_without_key()
        cred = ContentCredential(
            file_path="h.py",
            content_hash="0000000000000000000000000000000000000000000000000000000000000000",
            issuer="EcodiaOS Simula",
        )

        verified = await mgr.verify_file("h.py", cred, tmp_path)
        assert verified is False


# ── Batch Verification ───────────────────────────────────────────────────────


class TestVerifyBatch:
    @pytest.mark.asyncio
    async def test_batch_all_valid(self, tmp_path: Path):
        _write_file(tmp_path, "a.py", "aaa")
        _write_file(tmp_path, "b.py", "bbb")
        mgr = _manager_with_key()

        sign_result = await mgr.sign_files(["a.py", "b.py"], tmp_path)

        batch_result = await mgr.verify_batch(sign_result.credentials, tmp_path)

        assert batch_result.status == ContentCredentialStatus.VERIFIED
        assert batch_result.verified_count == 2
        assert batch_result.invalid_count == 0
        assert batch_result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_batch_some_invalid(self, tmp_path: Path):
        _write_file(tmp_path, "a.py", "aaa")
        _write_file(tmp_path, "b.py", "bbb")
        mgr = _manager_with_key()

        sign_result = await mgr.sign_files(["a.py", "b.py"], tmp_path)

        # Tamper with file b after signing
        _write_file(tmp_path, "b.py", "TAMPERED")

        batch_result = await mgr.verify_batch(sign_result.credentials, tmp_path)

        assert batch_result.status == ContentCredentialStatus.INVALID
        assert batch_result.verified_count == 1
        assert batch_result.invalid_count == 1

    @pytest.mark.asyncio
    async def test_batch_all_invalid(self, tmp_path: Path):
        _write_file(tmp_path, "x.py", "x")
        mgr = _manager_with_key()

        sign_result = await mgr.sign_files(["x.py"], tmp_path)

        _write_file(tmp_path, "x.py", "CHANGED")

        batch_result = await mgr.verify_batch(sign_result.credentials, tmp_path)

        assert batch_result.status == ContentCredentialStatus.INVALID
        assert batch_result.verified_count == 0
        assert batch_result.invalid_count == 1

    @pytest.mark.asyncio
    async def test_batch_empty_credentials(self, tmp_path: Path):
        mgr = _manager_with_key()

        batch_result = await mgr.verify_batch([], tmp_path)

        assert batch_result.status == ContentCredentialStatus.UNSIGNED
        assert batch_result.verified_count == 0
        assert batch_result.invalid_count == 0

    @pytest.mark.asyncio
    async def test_batch_with_missing_file(self, tmp_path: Path):
        _write_file(tmp_path, "ok.py", "fine")
        mgr = _manager_with_key()

        sign_result = await mgr.sign_files(["ok.py"], tmp_path)

        # Delete the file after signing
        (tmp_path / "ok.py").unlink()

        batch_result = await mgr.verify_batch(sign_result.credentials, tmp_path)

        assert batch_result.status == ContentCredentialStatus.INVALID
        assert batch_result.invalid_count == 1


# ── C2PA Manifest Structure ──────────────────────────────────────────────────


class TestC2PAManifest:
    @pytest.mark.asyncio
    async def test_manifest_is_valid_json(self, tmp_path: Path):
        _write_file(tmp_path, "m.py", "manifest test")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["m.py"], tmp_path)
        cred = result.credentials[0]

        manifest = json.loads(cred.c2pa_manifest_json)
        assert isinstance(manifest, dict)

    @pytest.mark.asyncio
    async def test_manifest_claim_generator(self, tmp_path: Path):
        _write_file(tmp_path, "m.py", "cg")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["m.py"], tmp_path)
        manifest = json.loads(result.credentials[0].c2pa_manifest_json)

        assert manifest["claim_generator"] == "EcodiaOS Simula/1.0"
        assert manifest["claim_generator_info"] == [
            {"name": "EcodiaOS", "version": "1.0"},
        ]

    @pytest.mark.asyncio
    async def test_manifest_title_matches_file_path(self, tmp_path: Path):
        _write_file(tmp_path, "src/deep/file.py", "nested")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["src/deep/file.py"], tmp_path)
        manifest = json.loads(result.credentials[0].c2pa_manifest_json)

        assert manifest["title"] == "src/deep/file.py"

    @pytest.mark.asyncio
    async def test_manifest_has_instance_id(self, tmp_path: Path):
        _write_file(tmp_path, "m.py", "id")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["m.py"], tmp_path)
        manifest = json.loads(result.credentials[0].c2pa_manifest_json)

        assert manifest["instance_id"].startswith("urn:uuid:")

    @pytest.mark.asyncio
    async def test_manifest_hash_assertion(self, tmp_path: Path):
        content = "hash assertion content"
        _write_file(tmp_path, "m.py", content)
        mgr = _manager_with_key()

        result = await mgr.sign_files(["m.py"], tmp_path)
        manifest = json.loads(result.credentials[0].c2pa_manifest_json)

        assertions = manifest["assertions"]
        hash_assertion = next(a for a in assertions if a["label"] == "c2pa.hash.sha256")
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        assert hash_assertion["data"]["hash"] == expected_hash
        assert hash_assertion["data"]["algorithm"] == "SHA-256"

    @pytest.mark.asyncio
    async def test_manifest_action_assertion(self, tmp_path: Path):
        _write_file(tmp_path, "m.py", "action")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["m.py"], tmp_path)
        manifest = json.loads(result.credentials[0].c2pa_manifest_json)

        assertions = manifest["assertions"]
        action_assertion = next(a for a in assertions if a["label"] == "c2pa.actions")
        actions = action_assertion["data"]["actions"]

        assert len(actions) == 1
        assert actions[0]["action"] == "c2pa.created"
        assert actions[0]["softwareAgent"] == "EcodiaOS Simula Code Agent"
        assert "when" in actions[0]

    @pytest.mark.asyncio
    async def test_manifest_signature_info(self, tmp_path: Path):
        _write_file(tmp_path, "m.py", "siginfo")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["m.py"], tmp_path)
        manifest = json.loads(result.credentials[0].c2pa_manifest_json)

        sig_info = manifest["signature_info"]
        assert sig_info["issuer"] == "EcodiaOS Simula"
        assert sig_info["algorithm"] == "Ed25519"
        assert "time" in sig_info

    @pytest.mark.asyncio
    async def test_manifest_dc_format(self, tmp_path: Path):
        _write_file(tmp_path, "m.py", "fmt")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["m.py"], tmp_path)
        manifest = json.loads(result.credentials[0].c2pa_manifest_json)

        assert manifest["dc:format"] == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_manifests_have_unique_instance_ids(self, tmp_path: Path):
        _write_file(tmp_path, "a.py", "aaa")
        _write_file(tmp_path, "b.py", "bbb")
        mgr = _manager_with_key()

        result = await mgr.sign_files(["a.py", "b.py"], tmp_path)

        manifest_a = json.loads(result.credentials[0].c2pa_manifest_json)
        manifest_b = json.loads(result.credentials[1].c2pa_manifest_json)

        assert manifest_a["instance_id"] != manifest_b["instance_id"]


# ── Key Loading ──────────────────────────────────────────────────────────────


class TestKeyLoading:
    def test_load_valid_ed25519_key(self, tmp_path: Path):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
        )

        private_key = Ed25519PrivateKey.generate()
        pem_data = private_key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption(),
        )
        key_file = tmp_path / "signing.pem"
        key_file.write_bytes(pem_data)

        mgr = ContentCredentialManager(signing_key_path=str(key_file))

        assert mgr._private_key is not None
        assert mgr._public_key is not None

    def test_load_missing_key_file(self, tmp_path: Path):
        mgr = ContentCredentialManager(
            signing_key_path=str(tmp_path / "does_not_exist.pem"),
        )

        assert mgr._private_key is None
        assert mgr._public_key is None

    def test_load_empty_path_no_key(self):
        mgr = ContentCredentialManager(signing_key_path="")

        assert mgr._private_key is None
        assert mgr._public_key is None

    def test_load_non_ed25519_key_rejected(self, tmp_path: Path):
        from cryptography.hazmat.primitives.asymmetric.ec import (
            SECP256R1,
            generate_private_key,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
        )

        ec_key = generate_private_key(SECP256R1())
        pem_data = ec_key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption(),
        )
        key_file = tmp_path / "ec_key.pem"
        key_file.write_bytes(pem_data)

        mgr = ContentCredentialManager(signing_key_path=str(key_file))

        assert mgr._private_key is None
        assert mgr._public_key is None

    def test_load_corrupted_key_file(self, tmp_path: Path):
        key_file = tmp_path / "bad.pem"
        key_file.write_text("not a valid pem file", encoding="utf-8")

        mgr = ContentCredentialManager(signing_key_path=str(key_file))

        assert mgr._private_key is None
        assert mgr._public_key is None


# ── Error Handling During Signing ────────────────────────────────────────────


class TestSigningErrorHandling:
    @pytest.mark.asyncio
    async def test_read_error_marks_file_unsigned(self, tmp_path: Path):
        """If reading a file raises an unexpected error, the file goes to unsigned."""
        _write_file(tmp_path, "err.py", "ok")
        mgr = _manager_with_key()

        with patch.object(
            type(tmp_path / "err.py"), "read_bytes",
            side_effect=PermissionError("access denied"),
        ):
            result = await mgr.sign_files(["err.py"], tmp_path)

        assert "err.py" in result.unsigned_files

    @pytest.mark.asyncio
    async def test_signing_key_error_marks_file_unsigned(self, tmp_path: Path):
        """If the private key signing raises, the file goes to unsigned."""
        _write_file(tmp_path, "s.py", "sign me")
        mgr = _manager_with_key()

        mgr._private_key = MagicMock()
        mgr._private_key.sign.side_effect = RuntimeError("key hardware failure")

        result = await mgr.sign_files(["s.py"], tmp_path)

        assert "s.py" in result.unsigned_files
        assert len(result.credentials) == 0


# ── Round-Trip Integration ───────────────────────────────────────────────────


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_sign_then_verify_roundtrip(self, tmp_path: Path):
        """Full round-trip: sign files, then batch-verify them."""
        _write_file(tmp_path, "alpha.py", "alpha code")
        _write_file(tmp_path, "beta.py", "beta code")
        _write_file(tmp_path, "gamma.py", "gamma code")
        mgr = _manager_with_key()

        sign_result = await mgr.sign_files(
            ["alpha.py", "beta.py", "gamma.py"], tmp_path,
        )
        assert sign_result.status == ContentCredentialStatus.SIGNED
        assert len(sign_result.credentials) == 3

        verify_result = await mgr.verify_batch(sign_result.credentials, tmp_path)
        assert verify_result.status == ContentCredentialStatus.VERIFIED
        assert verify_result.verified_count == 3
        assert verify_result.invalid_count == 0

    @pytest.mark.asyncio
    async def test_sign_verify_tamper_verify(self, tmp_path: Path):
        """Sign, verify OK, tamper one file, verify fails."""
        _write_file(tmp_path, "safe.py", "safe")
        _write_file(tmp_path, "target.py", "original")
        mgr = _manager_with_key()

        sign_result = await mgr.sign_files(["safe.py", "target.py"], tmp_path)

        verify_ok = await mgr.verify_batch(sign_result.credentials, tmp_path)
        assert verify_ok.status == ContentCredentialStatus.VERIFIED

        _write_file(tmp_path, "target.py", "INJECTED MALICIOUS CODE")

        verify_bad = await mgr.verify_batch(sign_result.credentials, tmp_path)
        assert verify_bad.status == ContentCredentialStatus.INVALID
        assert verify_bad.verified_count == 1
        assert verify_bad.invalid_count == 1

    @pytest.mark.asyncio
    async def test_binary_file_roundtrip(self, tmp_path: Path):
        """Ensure binary file content is signed and verified correctly."""
        binary_content = bytes(range(256))
        bin_path = tmp_path / "data.bin"
        bin_path.write_bytes(binary_content)
        mgr = _manager_with_key()

        sign_result = await mgr.sign_files(["data.bin"], tmp_path)
        assert len(sign_result.credentials) == 1

        expected_hash = hashlib.sha256(binary_content).hexdigest()
        assert sign_result.credentials[0].content_hash == expected_hash

        verify_result = await mgr.verify_batch(sign_result.credentials, tmp_path)
        assert verify_result.status == ContentCredentialStatus.VERIFIED
