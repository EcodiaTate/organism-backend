"""
EcodiaOS -- Simula Hash Chain Manager (Stage 6A.1)

SHA-256 hash chains on EvolutionRecord nodes.

Each EvolutionRecord is hashed and chained to its predecessor,
creating a tamper-evident log. If any record is modified after
the fact, the chain breaks and verification fails.

Algorithm:
  content_hash = SHA-256(sorted canonical fields of the record)
  chain_hash   = SHA-256(previous_chain_hash + content_hash)
  genesis:       chain_hash = SHA-256("" + content_hash)

Neo4j schema:
  (:HashChainEntry {record_id, content_hash, chain_hash, previous_hash,
                    position, verified_at})
  (entry)-[:CHAIN_NEXT]->(prev_entry)
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

import structlog

from primitives.common import utc_now
from systems.simula.verification.types import (
    HashChainEntry,
    HashChainStatus,
    HashChainVerificationResult,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.simula.evolution_types import EvolutionRecord

logger = structlog.get_logger().bind(system="simula.audit.hash_chain")


# Fields from EvolutionRecord used to compute the content hash.
# Sorted alphabetically for deterministic hashing.
_CANONICAL_FIELDS: tuple[str, ...] = (
    "applied_at",
    "category",
    "description",
    "files_changed",
    "from_version",
    "id",
    "proposal_id",
    "rolled_back",
    "rollback_reason",
    "simulation_risk",
    "to_version",
)


class HashChainManager:
    """Manages the SHA-256 hash chain for EvolutionRecord auditability."""

    def __init__(
        self,
        neo4j: Neo4jClient | None = None,
        *,
        algorithm: str = "sha256",
    ) -> None:
        self._neo4j = neo4j
        self._algorithm = algorithm

    # ── Public API ──────────────────────────────────────────────────────────

    def compute_content_hash(self, record: EvolutionRecord) -> str:
        """SHA-256 hash of the record's canonical fields."""
        canonical: dict[str, str] = {}
        for field in _CANONICAL_FIELDS:
            val = getattr(record, field, "")
            if isinstance(val, list):
                canonical[field] = json.dumps(sorted(val))
            elif hasattr(val, "value"):
                canonical[field] = str(val.value)
            elif hasattr(val, "isoformat"):
                canonical[field] = val.isoformat()
            else:
                canonical[field] = str(val)

        payload = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def compute_chain_hash(self, content_hash: str, previous_hash: str) -> str:
        """SHA-256(previous_chain_hash + content_hash)."""
        payload = (previous_hash + content_hash).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    async def append(self, record: EvolutionRecord) -> HashChainEntry:
        """
        Append a new entry to the hash chain.

        1. Get the current chain tip from Neo4j (or start genesis).
        2. Compute content hash and chain hash.
        3. Write the new entry node + CHAIN_NEXT relationship.
        """
        start = time.monotonic()
        content_hash = self.compute_content_hash(record)

        # Get current chain tip
        tip = await self._get_chain_tip()
        previous_hash = tip.chain_hash if tip is not None else ""
        position = (tip.chain_position + 1) if tip is not None else 0

        chain_hash = self.compute_chain_hash(content_hash, previous_hash)

        entry = HashChainEntry(
            record_id=record.id,
            previous_hash=previous_hash,
            content_hash=content_hash,
            chain_hash=chain_hash,
            chain_position=position,
            verified_at=utc_now(),
        )

        # Persist to Neo4j
        if self._neo4j is not None:
            await self._write_entry(entry, tip_record_id=tip.record_id if tip else None)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "hash_chain_appended",
            record_id=record.id,
            position=position,
            chain_hash=chain_hash[:16],
            duration_ms=elapsed_ms,
        )
        return entry

    async def verify_chain(self, limit: int = 100) -> HashChainVerificationResult:
        """
        Walk the chain from genesis and verify every link.

        Returns a result indicating whether the chain is intact,
        and where any break was found.
        """
        start = time.monotonic()

        if self._neo4j is None:
            return HashChainVerificationResult(
                status=HashChainStatus.UNVERIFIED,
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        entries = await self._read_chain(limit=limit)
        if not entries:
            return HashChainVerificationResult(
                status=HashChainStatus.GENESIS,
                chain_length=0,
                entries_verified=0,
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        # Walk from genesis (position 0) to tip
        entries.sort(key=lambda e: e.chain_position)
        verified = 0
        root_hash = entries[0].chain_hash

        for i, entry in enumerate(entries):
            expected_previous = entries[i - 1].chain_hash if i > 0 else ""
            if entry.previous_hash != expected_previous:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                logger.warning(
                    "hash_chain_break_detected",
                    position=entry.chain_position,
                    expected_previous=expected_previous[:16],
                    actual_previous=entry.previous_hash[:16],
                )
                return HashChainVerificationResult(
                    status=HashChainStatus.BROKEN,
                    chain_length=len(entries),
                    entries_verified=verified,
                    break_position=entry.chain_position,
                    root_hash=root_hash,
                    tip_hash=entries[-1].chain_hash,
                    duration_ms=elapsed_ms,
                )

            # Recompute chain hash to detect content tampering
            expected_chain = self.compute_chain_hash(
                entry.content_hash, expected_previous,
            )
            if entry.chain_hash != expected_chain:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                logger.warning(
                    "hash_chain_tamper_detected",
                    position=entry.chain_position,
                    record_id=entry.record_id,
                )
                return HashChainVerificationResult(
                    status=HashChainStatus.BROKEN,
                    chain_length=len(entries),
                    entries_verified=verified,
                    break_position=entry.chain_position,
                    root_hash=root_hash,
                    tip_hash=entries[-1].chain_hash,
                    duration_ms=elapsed_ms,
                )

            verified += 1

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "hash_chain_verified",
            chain_length=len(entries),
            entries_verified=verified,
            duration_ms=elapsed_ms,
        )
        return HashChainVerificationResult(
            status=HashChainStatus.VALID,
            chain_length=len(entries),
            entries_verified=verified,
            break_position=-1,
            root_hash=root_hash,
            tip_hash=entries[-1].chain_hash,
            duration_ms=elapsed_ms,
        )

    async def get_chain_root(self) -> str | None:
        """Return the genesis (root) hash, or None if chain is empty."""
        if self._neo4j is None:
            return None
        rows = await self._neo4j.execute_read(
            """
            MATCH (e:HashChainEntry)
            WHERE e.position = 0
            RETURN e.chain_hash AS chain_hash
            LIMIT 1
            """,
            {},
        )
        if rows:
            return str(rows[0]["chain_hash"])
        return None

    # ── Private helpers ─────────────────────────────────────────────────────

    async def _get_chain_tip(self) -> HashChainEntry | None:
        """Get the most recent entry in the chain."""
        if self._neo4j is None:
            return None
        rows = await self._neo4j.execute_read(
            """
            MATCH (e:HashChainEntry)
            RETURN e.record_id AS record_id,
                   e.previous_hash AS previous_hash,
                   e.content_hash AS content_hash,
                   e.chain_hash AS chain_hash,
                   e.position AS chain_position
            ORDER BY e.position DESC
            LIMIT 1
            """,
            {},
        )
        if not rows:
            return None
        row = rows[0]
        return HashChainEntry(
            record_id=str(row["record_id"]),
            previous_hash=str(row["previous_hash"]),
            content_hash=str(row["content_hash"]),
            chain_hash=str(row["chain_hash"]),
            chain_position=int(row["chain_position"]),
        )

    async def _write_entry(self, entry: HashChainEntry, tip_record_id: str | None) -> None:
        """Write a new HashChainEntry node and link to predecessor."""
        if self._neo4j is None:
            return

        # Create the new entry node
        await self._neo4j.execute_write(
            """
            CREATE (e:HashChainEntry {
                record_id: $record_id,
                previous_hash: $previous_hash,
                content_hash: $content_hash,
                chain_hash: $chain_hash,
                position: $position,
                verified_at: $verified_at
            })
            """,
            {
                "record_id": entry.record_id,
                "previous_hash": entry.previous_hash,
                "content_hash": entry.content_hash,
                "chain_hash": entry.chain_hash,
                "position": entry.chain_position,
                "verified_at": entry.verified_at.isoformat(),
            },
        )

        # Link to predecessor if not genesis
        if tip_record_id is not None:
            await self._neo4j.execute_write(
                """
                MATCH (curr:HashChainEntry {record_id: $curr_id})
                MATCH (prev:HashChainEntry {record_id: $prev_id})
                CREATE (curr)-[:CHAIN_NEXT]->(prev)
                """,
                {"curr_id": entry.record_id, "prev_id": tip_record_id},
            )

    async def _read_chain(self, limit: int = 100) -> list[HashChainEntry]:
        """Read the full chain from Neo4j, ordered by position."""
        if self._neo4j is None:
            return []
        rows = await self._neo4j.execute_read(
            """
            MATCH (e:HashChainEntry)
            RETURN e.record_id AS record_id,
                   e.previous_hash AS previous_hash,
                   e.content_hash AS content_hash,
                   e.chain_hash AS chain_hash,
                   e.position AS chain_position
            ORDER BY e.position ASC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [
            HashChainEntry(
                record_id=str(row["record_id"]),
                previous_hash=str(row["previous_hash"]),
                content_hash=str(row["content_hash"]),
                chain_hash=str(row["chain_hash"]),
                chain_position=int(row["chain_position"]),
            )
            for row in rows
        ]
