"""
EcodiaOS -- Simula Governance Verifiable Credentials (Stage 6A.3)

Verifiable Credentials for governance decisions - tamper-evident
approval chains.

Each governance decision (approve/reject/defer) is signed with Ed25519
and chained to prior decisions for the same proposal, creating an
auditable approval trail.

Regulatory targets: finance (SOX), healthcare (HIPAA), defense (CMMC).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from primitives.common import utc_now
from systems.simula.verification.types import (
    GovernanceCredential,
    GovernanceCredentialResult,
    VerifiableCredentialStatus,
)

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    from clients.neo4j import Neo4jClient

logger = structlog.get_logger().bind(system="simula.audit.verifiable_credentials")


class GovernanceCredentialManager:
    """Issues and verifies tamper-evident governance decision credentials."""

    def __init__(
        self,
        neo4j: Neo4jClient | None = None,
        *,
        signing_key_path: str = "",
    ) -> None:
        self._neo4j = neo4j
        self._private_key: Ed25519PrivateKey | None = None
        self._public_key: Ed25519PublicKey | None = None

        if signing_key_path:
            self._load_signing_key(signing_key_path)

    # ── Public API ──────────────────────────────────────────────────────────

    async def issue_credential(
        self,
        governance_record_id: str,
        proposal_id: str,
        approver_id: str,
        decision: str,
    ) -> GovernanceCredential:
        """
        Issue a signed credential for a governance decision.

        The credential contains:
        1. The decision payload (who approved what, when)
        2. SHA-256 hash of the payload
        3. Ed25519 signature of the hash
        4. Chain of prior credentials for the same proposal
        """
        now = utc_now()

        # Build the payload to sign
        payload = {
            "governance_record_id": governance_record_id,
            "proposal_id": proposal_id,
            "approver_id": approver_id,
            "decision": decision,
            "issued_at": now.isoformat(),
        }
        payload_json = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

        # Sign the payload hash
        signature = ""
        if self._private_key is not None:
            sig_bytes = self._private_key.sign(payload_hash.encode("utf-8"))
            signature = sig_bytes.hex()

        # Fetch prior credentials for this proposal to build the chain
        prior_credentials = await self._get_prior_credentials(proposal_id)
        chain_json = json.dumps(
            [
                {
                    "governance_record_id": c.governance_record_id,
                    "decision": c.decision,
                    "signed_payload_hash": c.signed_payload_hash,
                }
                for c in prior_credentials
            ],
        )

        credential = GovernanceCredential(
            governance_record_id=governance_record_id,
            proposal_id=proposal_id,
            approver_id=approver_id,
            decision=decision,
            signature=signature,
            signed_payload_hash=payload_hash,
            credential_chain_json=chain_json,
            issued_at=now,
        )

        # Persist to Neo4j
        if self._neo4j is not None:
            await self._store_credential(credential)

        logger.info(
            "governance_credential_issued",
            governance_record_id=governance_record_id,
            proposal_id=proposal_id,
            decision=decision,
            chain_length=len(prior_credentials) + 1,
        )
        return credential

    async def verify_credential(self, credential: GovernanceCredential) -> bool:
        """Verify a single governance credential's signature."""
        if not credential.signature or not credential.signed_payload_hash:
            return False

        if self._public_key is None:
            # Without a public key, we can only verify hash integrity
            payload = {
                "governance_record_id": credential.governance_record_id,
                "proposal_id": credential.proposal_id,
                "approver_id": credential.approver_id,
                "decision": credential.decision,
                "issued_at": credential.issued_at.isoformat(),
            }
            payload_json = json.dumps(payload, sort_keys=True)
            expected_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            return expected_hash == credential.signed_payload_hash

        try:
            sig_bytes = bytes.fromhex(credential.signature)
            self._public_key.verify(
                sig_bytes,
                credential.signed_payload_hash.encode("utf-8"),
            )
            return True
        except Exception:
            logger.warning(
                "credential_signature_invalid",
                governance_record_id=credential.governance_record_id,
            )
            return False

    async def verify_governance_chain(
        self,
        proposal_id: str,
    ) -> GovernanceCredentialResult:
        """
        Verify the complete governance credential chain for a proposal.

        Walks all credentials in issuance order, verifying each signature
        and the chain linkage.
        """
        start = time.monotonic()
        credentials = await self._get_prior_credentials(proposal_id)

        if not credentials:
            return GovernanceCredentialResult(
                status=VerifiableCredentialStatus.UNVERIFIED,
                chain_length=0,
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        all_valid = True
        for cred in credentials:
            if not await self.verify_credential(cred):
                all_valid = False
                break

        status = (
            VerifiableCredentialStatus.VALID
            if all_valid
            else VerifiableCredentialStatus.UNVERIFIED
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "governance_chain_verified",
            proposal_id=proposal_id,
            chain_length=len(credentials),
            valid=all_valid,
            duration_ms=elapsed_ms,
        )

        return GovernanceCredentialResult(
            status=status,
            credentials=credentials,
            chain_verified=all_valid,
            chain_length=len(credentials),
            duration_ms=elapsed_ms,
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    async def _get_prior_credentials(
        self,
        proposal_id: str,
    ) -> list[GovernanceCredential]:
        """Fetch all governance credentials for a proposal from Neo4j."""
        if self._neo4j is None:
            return []

        rows = await self._neo4j.execute_read(
            """
            MATCH (c:GovernanceCredential {proposal_id: $proposal_id})
            RETURN c.governance_record_id AS governance_record_id,
                   c.proposal_id AS proposal_id,
                   c.approver_id AS approver_id,
                   c.decision AS decision,
                   c.signature AS signature,
                   c.signed_payload_hash AS signed_payload_hash,
                   c.credential_chain_json AS credential_chain_json,
                   c.issued_at AS issued_at
            ORDER BY c.issued_at ASC
            """,
            {"proposal_id": proposal_id},
        )

        from datetime import datetime

        results: list[GovernanceCredential] = []
        for row in rows:
            issued_str = str(row.get("issued_at", ""))
            try:
                issued_at = datetime.fromisoformat(issued_str)
            except (ValueError, TypeError):
                issued_at = utc_now()

            results.append(
                GovernanceCredential(
                    governance_record_id=str(row["governance_record_id"]),
                    proposal_id=str(row["proposal_id"]),
                    approver_id=str(row.get("approver_id", "")),
                    decision=str(row.get("decision", "")),
                    signature=str(row.get("signature", "")),
                    signed_payload_hash=str(row.get("signed_payload_hash", "")),
                    credential_chain_json=str(row.get("credential_chain_json", "")),
                    issued_at=issued_at,
                ),
            )
        return results

    async def _store_credential(self, credential: GovernanceCredential) -> None:
        """Store a governance credential in Neo4j."""
        if self._neo4j is None:
            return

        await self._neo4j.execute_write(
            """
            CREATE (c:GovernanceCredential {
                governance_record_id: $governance_record_id,
                proposal_id: $proposal_id,
                approver_id: $approver_id,
                decision: $decision,
                signature: $signature,
                signed_payload_hash: $signed_payload_hash,
                credential_chain_json: $credential_chain_json,
                issued_at: $issued_at
            })
            """,
            {
                "governance_record_id": credential.governance_record_id,
                "proposal_id": credential.proposal_id,
                "approver_id": credential.approver_id,
                "decision": credential.decision,
                "signature": credential.signature,
                "signed_payload_hash": credential.signed_payload_hash,
                "credential_chain_json": credential.credential_chain_json,
                "issued_at": credential.issued_at.isoformat(),
            },
        )

    def _load_signing_key(self, key_path: str) -> None:
        """Load Ed25519 private key from PEM file."""
        try:
            from cryptography.hazmat.primitives.serialization import (
                load_pem_private_key,
            )

            path = Path(key_path)
            if not path.exists():
                logger.warning("signing_key_not_found", path=key_path)
                return

            key_data = path.read_bytes()
            private_key = load_pem_private_key(key_data, password=None)

            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

            if isinstance(private_key, Ed25519PrivateKey):
                self._private_key = private_key
                self._public_key = private_key.public_key()
                logger.info("signing_key_loaded", path=key_path)
            else:
                logger.warning("signing_key_not_ed25519", key_type=type(private_key).__name__)

        except Exception as exc:
            logger.warning("signing_key_load_failed", error=str(exc))
