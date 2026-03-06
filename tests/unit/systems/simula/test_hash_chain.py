"""
Unit tests for Simula HashChainManager (Stage 6A.1).

Tests SHA-256 hash chain append, verification, content hashing
determinism, chain hash chaining, genesis record handling, broken
chain detection, and graceful degradation when Neo4j is None.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from systems.simula.audit.hash_chain import (
    _CANONICAL_FIELDS,
    HashChainManager,
)
from systems.simula.evolution_types import (
    ChangeCategory,
    EvolutionRecord,
    RiskLevel,
)
from systems.simula.verification.types import (
    HashChainEntry,
    HashChainStatus,
    HashChainVerificationResult,
)

# Resolve deferred annotations so Pydantic can construct models at runtime.
# The types module imports `datetime` under TYPE_CHECKING; we need it resolved.
HashChainEntry.model_rebuild()
HashChainVerificationResult.model_rebuild()
EvolutionRecord.model_rebuild()


# ── Helpers ──────────────────────────────────────────────────────────────────


_FIXED_DT = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def _make_record(
    *,
    record_id: str = "rec-001",
    proposal_id: str = "prop-001",
    category: ChangeCategory = ChangeCategory.ADD_EXECUTOR,
    description: str = "Add a new executor",
    from_version: int = 1,
    to_version: int = 2,
    files_changed: list[str] | None = None,
    simulation_risk: RiskLevel = RiskLevel.LOW,
    rolled_back: bool = False,
    rollback_reason: str = "",
    applied_at: datetime | None = None,
) -> EvolutionRecord:
    """Build a minimal EvolutionRecord for testing."""
    return EvolutionRecord(
        id=record_id,
        proposal_id=proposal_id,
        category=category,
        description=description,
        from_version=from_version,
        to_version=to_version,
        files_changed=files_changed or ["src/demo.py"],
        simulation_risk=simulation_risk,
        rolled_back=rolled_back,
        rollback_reason=rollback_reason,
        applied_at=applied_at or _FIXED_DT,
        created_at=_FIXED_DT,
    )


def _make_entry(
    *,
    record_id: str = "rec-001",
    content_hash: str = "aaa",
    chain_hash: str = "bbb",
    previous_hash: str = "",
    chain_position: int = 0,
) -> HashChainEntry:
    """Build a HashChainEntry for testing."""
    return HashChainEntry(
        record_id=record_id,
        content_hash=content_hash,
        chain_hash=chain_hash,
        previous_hash=previous_hash,
        chain_position=chain_position,
    )


def _mock_neo4j() -> AsyncMock:
    """Return an AsyncMock that stands in for Neo4jClient."""
    neo4j = AsyncMock()
    neo4j.execute_read = AsyncMock(return_value=[])
    neo4j.execute_write = AsyncMock(return_value=None)
    return neo4j


# ── TestComputeContentHash ───────────────────────────────────────────────────


class TestComputeContentHash:
    """Tests for compute_content_hash determinism and canonical field handling."""

    def test_deterministic_same_record(self):
        """Same record always produces the same content hash."""
        manager = HashChainManager(neo4j=None)
        record = _make_record()

        hash1 = manager.compute_content_hash(record)
        hash2 = manager.compute_content_hash(record)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_different_records_produce_different_hashes(self):
        """Records with different content produce different hashes."""
        manager = HashChainManager(neo4j=None)
        rec_a = _make_record(record_id="rec-001", description="First change")
        rec_b = _make_record(record_id="rec-002", description="Second change")

        hash_a = manager.compute_content_hash(rec_a)
        hash_b = manager.compute_content_hash(rec_b)

        assert hash_a != hash_b

    def test_uses_canonical_fields(self):
        """Content hash should be derived from the defined _CANONICAL_FIELDS."""
        manager = HashChainManager(neo4j=None)
        record = _make_record()

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
        expected = hashlib.sha256(payload).hexdigest()

        assert manager.compute_content_hash(record) == expected

    def test_list_field_sorted_for_determinism(self):
        """files_changed list is sorted before hashing, ensuring order independence."""
        manager = HashChainManager(neo4j=None)
        rec_a = _make_record(files_changed=["b.py", "a.py"])
        rec_b = _make_record(files_changed=["a.py", "b.py"])

        assert manager.compute_content_hash(rec_a) == manager.compute_content_hash(rec_b)

    def test_enum_field_uses_value(self):
        """Enum fields should be hashed by their .value string."""
        manager = HashChainManager(neo4j=None)
        record = _make_record(category=ChangeCategory.MODIFY_CONTRACT)

        content_hash = manager.compute_content_hash(record)
        assert isinstance(content_hash, str)
        assert len(content_hash) == 64

    def test_datetime_field_uses_isoformat(self):
        """datetime fields should be hashed via .isoformat()."""
        manager = HashChainManager(neo4j=None)
        record = _make_record()

        # applied_at is a datetime and should be isoformat-serialized
        content_hash = manager.compute_content_hash(record)
        assert isinstance(content_hash, str)
        assert len(content_hash) == 64


# ── TestComputeChainHash ─────────────────────────────────────────────────────


class TestComputeChainHash:
    """Tests for compute_chain_hash chaining logic."""

    def test_chain_hash_deterministic(self):
        """Same inputs always produce the same chain hash."""
        manager = HashChainManager(neo4j=None)

        result1 = manager.compute_chain_hash("content_abc", "prev_xyz")
        result2 = manager.compute_chain_hash("content_abc", "prev_xyz")

        assert result1 == result2
        assert len(result1) == 64

    def test_chain_hash_matches_manual_sha256(self):
        """Chain hash should equal SHA-256(previous_hash + content_hash)."""
        manager = HashChainManager(neo4j=None)
        content = "deadbeef"
        previous = "cafebabe"

        expected = hashlib.sha256((previous + content).encode("utf-8")).hexdigest()
        assert manager.compute_chain_hash(content, previous) == expected

    def test_genesis_chain_hash_uses_empty_previous(self):
        """Genesis entry uses empty string as previous_hash."""
        manager = HashChainManager(neo4j=None)
        content = "first_content_hash"

        genesis_hash = manager.compute_chain_hash(content, "")
        expected = hashlib.sha256(("" + content).encode("utf-8")).hexdigest()

        assert genesis_hash == expected

    def test_different_previous_produces_different_chain_hash(self):
        """Changing the previous hash changes the chain hash."""
        manager = HashChainManager(neo4j=None)
        content = "same_content"

        hash1 = manager.compute_chain_hash(content, "prev_a")
        hash2 = manager.compute_chain_hash(content, "prev_b")

        assert hash1 != hash2

    def test_different_content_produces_different_chain_hash(self):
        """Changing the content hash changes the chain hash."""
        manager = HashChainManager(neo4j=None)
        previous = "same_previous"

        hash1 = manager.compute_chain_hash("content_a", previous)
        hash2 = manager.compute_chain_hash("content_b", previous)

        assert hash1 != hash2


# ── TestHashChainManager ─────────────────────────────────────────────────────


class TestHashChainManager:
    """Tests for the append and chain tip logic."""

    @pytest.mark.asyncio
    async def test_append_genesis_record_no_neo4j(self):
        """Appending with no Neo4j should return a valid genesis entry."""
        manager = HashChainManager(neo4j=None)
        record = _make_record()

        entry = await manager.append(record)

        assert isinstance(entry, HashChainEntry)
        assert entry.record_id == record.id
        assert entry.chain_position == 0
        assert entry.previous_hash == ""
        assert len(entry.content_hash) == 64
        assert len(entry.chain_hash) == 64

    @pytest.mark.asyncio
    async def test_append_genesis_chain_hash_correct(self):
        """Genesis chain_hash should equal SHA-256('' + content_hash)."""
        manager = HashChainManager(neo4j=None)
        record = _make_record()

        entry = await manager.append(record)

        expected_chain = manager.compute_chain_hash(entry.content_hash, "")
        assert entry.chain_hash == expected_chain

    @pytest.mark.asyncio
    async def test_append_with_existing_tip(self):
        """When a tip exists, new entry chains from it."""
        neo4j = _mock_neo4j()
        tip = _make_entry(
            record_id="rec-000",
            content_hash="tip_content",
            chain_hash="tip_chain_hash_abc",
            previous_hash="",
            chain_position=0,
        )
        neo4j.execute_read = AsyncMock(
            return_value=[
                {
                    "record_id": tip.record_id,
                    "previous_hash": tip.previous_hash,
                    "content_hash": tip.content_hash,
                    "chain_hash": tip.chain_hash,
                    "chain_position": tip.chain_position,
                },
            ],
        )

        manager = HashChainManager(neo4j=neo4j)
        record = _make_record(record_id="rec-001")

        entry = await manager.append(record)

        assert entry.chain_position == 1
        assert entry.previous_hash == "tip_chain_hash_abc"
        expected_chain = manager.compute_chain_hash(entry.content_hash, "tip_chain_hash_abc")
        assert entry.chain_hash == expected_chain

    @pytest.mark.asyncio
    async def test_append_writes_to_neo4j(self):
        """append should call execute_write on Neo4j to persist the entry."""
        neo4j = _mock_neo4j()
        manager = HashChainManager(neo4j=neo4j)
        record = _make_record()

        await manager.append(record)

        # Genesis: one write for the node, no link write (no predecessor)
        assert neo4j.execute_write.call_count == 1

    @pytest.mark.asyncio
    async def test_append_writes_link_when_tip_exists(self):
        """When a tip exists, append should write both node and CHAIN_NEXT link."""
        neo4j = _mock_neo4j()
        tip = _make_entry(record_id="rec-000", chain_hash="prev_hash", chain_position=0)
        neo4j.execute_read = AsyncMock(
            return_value=[
                {
                    "record_id": tip.record_id,
                    "previous_hash": tip.previous_hash,
                    "content_hash": tip.content_hash,
                    "chain_hash": tip.chain_hash,
                    "chain_position": tip.chain_position,
                },
            ],
        )

        manager = HashChainManager(neo4j=neo4j)
        record = _make_record(record_id="rec-001")

        await manager.append(record)

        # One write for the node, one write for the CHAIN_NEXT relationship
        assert neo4j.execute_write.call_count == 2


# ── TestChainVerification ────────────────────────────────────────────────────


class TestChainVerification:
    """Tests for verify_chain: valid chains, broken chains, empty chains."""

    @pytest.mark.asyncio
    async def test_verify_no_neo4j_returns_unverified(self):
        """Without Neo4j, verification returns UNVERIFIED status."""
        manager = HashChainManager(neo4j=None)

        result = await manager.verify_chain()

        assert isinstance(result, HashChainVerificationResult)
        assert result.status == HashChainStatus.UNVERIFIED

    @pytest.mark.asyncio
    async def test_verify_empty_chain_returns_genesis(self):
        """Empty chain (no entries) returns GENESIS status."""
        neo4j = _mock_neo4j()
        neo4j.execute_read = AsyncMock(return_value=[])

        manager = HashChainManager(neo4j=neo4j)
        result = await manager.verify_chain()

        assert result.status == HashChainStatus.GENESIS
        assert result.chain_length == 0
        assert result.entries_verified == 0

    @pytest.mark.asyncio
    async def test_verify_valid_single_entry_chain(self):
        """A single genesis entry should verify as VALID."""
        neo4j = _mock_neo4j()
        manager = HashChainManager(neo4j=neo4j)

        content_hash = "a1b2c3d4e5f6"
        chain_hash = manager.compute_chain_hash(content_hash, "")

        neo4j.execute_read = AsyncMock(
            return_value=[
                {
                    "record_id": "rec-001",
                    "previous_hash": "",
                    "content_hash": content_hash,
                    "chain_hash": chain_hash,
                    "chain_position": 0,
                },
            ],
        )

        result = await manager.verify_chain()

        assert result.status == HashChainStatus.VALID
        assert result.chain_length == 1
        assert result.entries_verified == 1
        assert result.break_position == -1
        assert result.root_hash == chain_hash
        assert result.tip_hash == chain_hash

    @pytest.mark.asyncio
    async def test_verify_valid_multi_entry_chain(self):
        """A properly linked multi-entry chain should verify as VALID."""
        neo4j = _mock_neo4j()
        manager = HashChainManager(neo4j=neo4j)

        # Build a 3-entry chain manually
        content_0 = "content_hash_0"
        chain_0 = manager.compute_chain_hash(content_0, "")

        content_1 = "content_hash_1"
        chain_1 = manager.compute_chain_hash(content_1, chain_0)

        content_2 = "content_hash_2"
        chain_2 = manager.compute_chain_hash(content_2, chain_1)

        neo4j.execute_read = AsyncMock(
            return_value=[
                {
                    "record_id": "rec-0",
                    "previous_hash": "",
                    "content_hash": content_0,
                    "chain_hash": chain_0,
                    "chain_position": 0,
                },
                {
                    "record_id": "rec-1",
                    "previous_hash": chain_0,
                    "content_hash": content_1,
                    "chain_hash": chain_1,
                    "chain_position": 1,
                },
                {
                    "record_id": "rec-2",
                    "previous_hash": chain_1,
                    "content_hash": content_2,
                    "chain_hash": chain_2,
                    "chain_position": 2,
                },
            ],
        )

        result = await manager.verify_chain()

        assert result.status == HashChainStatus.VALID
        assert result.chain_length == 3
        assert result.entries_verified == 3
        assert result.break_position == -1
        assert result.root_hash == chain_0
        assert result.tip_hash == chain_2

    @pytest.mark.asyncio
    async def test_verify_broken_chain_wrong_previous_hash(self):
        """A chain with an incorrect previous_hash link should be BROKEN."""
        neo4j = _mock_neo4j()
        manager = HashChainManager(neo4j=neo4j)

        content_0 = "content_hash_0"
        chain_0 = manager.compute_chain_hash(content_0, "")

        content_1 = "content_hash_1"
        # Correct chain_hash for entry 1 using the real previous
        chain_1 = manager.compute_chain_hash(content_1, chain_0)

        # Entry 1 claims a wrong previous_hash
        neo4j.execute_read = AsyncMock(
            return_value=[
                {
                    "record_id": "rec-0",
                    "previous_hash": "",
                    "content_hash": content_0,
                    "chain_hash": chain_0,
                    "chain_position": 0,
                },
                {
                    "record_id": "rec-1",
                    "previous_hash": "WRONG_PREVIOUS_HASH",
                    "content_hash": content_1,
                    "chain_hash": chain_1,
                    "chain_position": 1,
                },
            ],
        )

        result = await manager.verify_chain()

        assert result.status == HashChainStatus.BROKEN
        assert result.chain_length == 2
        assert result.entries_verified == 1  # only genesis verified
        assert result.break_position == 1

    @pytest.mark.asyncio
    async def test_verify_broken_chain_tampered_content(self):
        """A chain with a tampered chain_hash should be BROKEN."""
        neo4j = _mock_neo4j()
        manager = HashChainManager(neo4j=neo4j)

        content_0 = "content_hash_0"
        chain_0 = manager.compute_chain_hash(content_0, "")

        content_1 = "content_hash_1"

        # Entry 1 has the correct previous_hash but a tampered chain_hash
        neo4j.execute_read = AsyncMock(
            return_value=[
                {
                    "record_id": "rec-0",
                    "previous_hash": "",
                    "content_hash": content_0,
                    "chain_hash": chain_0,
                    "chain_position": 0,
                },
                {
                    "record_id": "rec-1",
                    "previous_hash": chain_0,
                    "content_hash": content_1,
                    "chain_hash": "TAMPERED_CHAIN_HASH",
                    "chain_position": 1,
                },
            ],
        )

        result = await manager.verify_chain()

        assert result.status == HashChainStatus.BROKEN
        assert result.break_position == 1
        assert result.entries_verified == 1

    @pytest.mark.asyncio
    async def test_verify_broken_genesis_tampered(self):
        """Even a tampered genesis entry should be detected."""
        neo4j = _mock_neo4j()
        manager = HashChainManager(neo4j=neo4j)

        neo4j.execute_read = AsyncMock(
            return_value=[
                {
                    "record_id": "rec-0",
                    "previous_hash": "",
                    "content_hash": "real_content",
                    "chain_hash": "FAKE_GENESIS_CHAIN_HASH",
                    "chain_position": 0,
                },
            ],
        )

        result = await manager.verify_chain()

        assert result.status == HashChainStatus.BROKEN
        assert result.break_position == 0
        assert result.entries_verified == 0

    @pytest.mark.asyncio
    async def test_verify_chain_walk_sorts_by_position(self):
        """Entries returned out of order should still verify correctly."""
        neo4j = _mock_neo4j()
        manager = HashChainManager(neo4j=neo4j)

        content_0 = "content_0"
        chain_0 = manager.compute_chain_hash(content_0, "")

        content_1 = "content_1"
        chain_1 = manager.compute_chain_hash(content_1, chain_0)

        # Return out of order: position 1 first, then position 0
        neo4j.execute_read = AsyncMock(
            return_value=[
                {
                    "record_id": "rec-1",
                    "previous_hash": chain_0,
                    "content_hash": content_1,
                    "chain_hash": chain_1,
                    "chain_position": 1,
                },
                {
                    "record_id": "rec-0",
                    "previous_hash": "",
                    "content_hash": content_0,
                    "chain_hash": chain_0,
                    "chain_position": 0,
                },
            ],
        )

        result = await manager.verify_chain()

        assert result.status == HashChainStatus.VALID
        assert result.entries_verified == 2

    @pytest.mark.asyncio
    async def test_verify_chain_limit_parameter(self):
        """The limit parameter should be forwarded to _read_chain."""
        neo4j = _mock_neo4j()
        neo4j.execute_read = AsyncMock(return_value=[])

        manager = HashChainManager(neo4j=neo4j)
        await manager.verify_chain(limit=50)

        # The _read_chain call should pass limit=50
        call_args = neo4j.execute_read.call_args
        assert call_args[0][1]["limit"] == 50

    @pytest.mark.asyncio
    async def test_verify_reports_duration_ms(self):
        """Result should include a non-negative duration_ms."""
        neo4j = _mock_neo4j()
        neo4j.execute_read = AsyncMock(return_value=[])

        manager = HashChainManager(neo4j=neo4j)
        result = await manager.verify_chain()

        assert result.duration_ms >= 0


# ── TestChainWalk ────────────────────────────────────────────────────────────


class TestChainWalk:
    """End-to-end chain walk: append multiple records, then verify."""

    @pytest.mark.asyncio
    async def test_append_then_verify_without_neo4j(self):
        """Without Neo4j, append returns entries but verify returns UNVERIFIED."""
        manager = HashChainManager(neo4j=None)
        rec1 = _make_record(record_id="rec-001")
        rec2 = _make_record(record_id="rec-002", description="Second change")

        entry1 = await manager.append(rec1)
        entry2 = await manager.append(rec2)

        # Both are genesis because there is no stored tip without Neo4j
        assert entry1.chain_position == 0
        assert entry2.chain_position == 0

        result = await manager.verify_chain()
        assert result.status == HashChainStatus.UNVERIFIED

    @pytest.mark.asyncio
    async def test_full_chain_append_and_verify(self):
        """Build a 3-entry chain via mocked Neo4j and verify it end-to-end."""
        neo4j = _mock_neo4j()
        manager = HashChainManager(neo4j=neo4j)

        # Append entry 0 (genesis)
        neo4j.execute_read = AsyncMock(return_value=[])
        rec0 = _make_record(record_id="rec-000", description="Genesis")
        entry0 = await manager.append(rec0)

        assert entry0.chain_position == 0
        assert entry0.previous_hash == ""

        # Append entry 1 (chains from entry 0)
        neo4j.execute_read = AsyncMock(
            return_value=[
                {
                    "record_id": entry0.record_id,
                    "previous_hash": entry0.previous_hash,
                    "content_hash": entry0.content_hash,
                    "chain_hash": entry0.chain_hash,
                    "chain_position": entry0.chain_position,
                },
            ],
        )
        rec1 = _make_record(record_id="rec-001", description="Second")
        entry1 = await manager.append(rec1)

        assert entry1.chain_position == 1
        assert entry1.previous_hash == entry0.chain_hash

        # Append entry 2 (chains from entry 1)
        neo4j.execute_read = AsyncMock(
            return_value=[
                {
                    "record_id": entry1.record_id,
                    "previous_hash": entry1.previous_hash,
                    "content_hash": entry1.content_hash,
                    "chain_hash": entry1.chain_hash,
                    "chain_position": entry1.chain_position,
                },
            ],
        )
        rec2 = _make_record(record_id="rec-002", description="Third")
        entry2 = await manager.append(rec2)

        assert entry2.chain_position == 2
        assert entry2.previous_hash == entry1.chain_hash

        # Now verify the full chain
        neo4j.execute_read = AsyncMock(
            return_value=[
                {
                    "record_id": entry0.record_id,
                    "previous_hash": entry0.previous_hash,
                    "content_hash": entry0.content_hash,
                    "chain_hash": entry0.chain_hash,
                    "chain_position": entry0.chain_position,
                },
                {
                    "record_id": entry1.record_id,
                    "previous_hash": entry1.previous_hash,
                    "content_hash": entry1.content_hash,
                    "chain_hash": entry1.chain_hash,
                    "chain_position": entry1.chain_position,
                },
                {
                    "record_id": entry2.record_id,
                    "previous_hash": entry2.previous_hash,
                    "content_hash": entry2.content_hash,
                    "chain_hash": entry2.chain_hash,
                    "chain_position": entry2.chain_position,
                },
            ],
        )

        result = await manager.verify_chain()

        assert result.status == HashChainStatus.VALID
        assert result.chain_length == 3
        assert result.entries_verified == 3
        assert result.root_hash == entry0.chain_hash
        assert result.tip_hash == entry2.chain_hash


# ── TestGetChainRoot ─────────────────────────────────────────────────────────


class TestGetChainRoot:
    """Tests for get_chain_root helper."""

    @pytest.mark.asyncio
    async def test_no_neo4j_returns_none(self):
        """Without Neo4j, get_chain_root returns None."""
        manager = HashChainManager(neo4j=None)
        result = await manager.get_chain_root()
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_chain_returns_none(self):
        """Empty chain should return None."""
        neo4j = _mock_neo4j()
        neo4j.execute_read = AsyncMock(return_value=[])

        manager = HashChainManager(neo4j=neo4j)
        result = await manager.get_chain_root()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_genesis_hash(self):
        """Should return the chain_hash of position 0."""
        neo4j = _mock_neo4j()
        neo4j.execute_read = AsyncMock(
            return_value=[{"chain_hash": "genesis_root_hash_abc"}],
        )

        manager = HashChainManager(neo4j=neo4j)
        result = await manager.get_chain_root()
        assert result == "genesis_root_hash_abc"


# ── TestNeo4jNone ────────────────────────────────────────────────────────────


class TestNeo4jNone:
    """When Neo4j is None, all methods should return empty/default results."""

    @pytest.mark.asyncio
    async def test_get_chain_tip_returns_none(self):
        manager = HashChainManager(neo4j=None)
        result = await manager._get_chain_tip()
        assert result is None

    @pytest.mark.asyncio
    async def test_write_entry_is_noop(self):
        """_write_entry should not raise when Neo4j is None."""
        manager = HashChainManager(neo4j=None)
        entry = _make_entry()
        # Should not raise
        await manager._write_entry(entry, tip_record_id=None)

    @pytest.mark.asyncio
    async def test_read_chain_returns_empty(self):
        manager = HashChainManager(neo4j=None)
        result = await manager._read_chain()
        assert result == []

    @pytest.mark.asyncio
    async def test_verify_chain_returns_unverified(self):
        manager = HashChainManager(neo4j=None)
        result = await manager.verify_chain()
        assert result.status == HashChainStatus.UNVERIFIED

    @pytest.mark.asyncio
    async def test_get_chain_root_returns_none(self):
        manager = HashChainManager(neo4j=None)
        result = await manager.get_chain_root()
        assert result is None

    @pytest.mark.asyncio
    async def test_append_still_returns_entry(self):
        """Even without Neo4j, append should compute and return a valid entry."""
        manager = HashChainManager(neo4j=None)
        record = _make_record()

        entry = await manager.append(record)

        assert isinstance(entry, HashChainEntry)
        assert entry.record_id == record.id
        assert len(entry.content_hash) == 64
        assert len(entry.chain_hash) == 64
