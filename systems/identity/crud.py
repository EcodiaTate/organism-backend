"""
EcodiaOS — Identity CRUD

Async asyncpg queries for SealedEnvelope persistence.

Table schema (created by ensure_table()):

    CREATE TABLE IF NOT EXISTS sealed_envelopes (
        id                TEXT PRIMARY KEY,
        platform_id       TEXT NOT NULL,
        purpose           TEXT NOT NULL,
        ciphertext        TEXT NOT NULL,
        key_version       INTEGER NOT NULL DEFAULT 1,
        created_at        TIMESTAMPTZ NOT NULL,
        last_accessed_at  TIMESTAMPTZ,
        updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

All functions accept an asyncpg.Connection so callers can participate
in an outer transaction if needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from datetime import datetime

import structlog

from systems.identity.vault import SealedEnvelope

if TYPE_CHECKING:

    import asyncpg
logger = structlog.get_logger("identity.crud")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sealed_envelopes (
    id                TEXT PRIMARY KEY,
    platform_id       TEXT NOT NULL,
    purpose           TEXT NOT NULL,
    ciphertext        TEXT NOT NULL,
    key_version       INTEGER NOT NULL DEFAULT 1,
    created_at        TIMESTAMPTZ NOT NULL,
    last_accessed_at  TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sealed_envelopes_platform_id
    ON sealed_envelopes (platform_id);
"""


async def ensure_table(conn: asyncpg.Connection) -> None:  # type: ignore[type-arg]
    """Create sealed_envelopes table and index if not present."""
    for stmt in _CREATE_TABLE_SQL.split(";"):
        s = stmt.strip()
        if s:
            await conn.execute(s)
    logger.debug("sealed_envelopes_table_ensured")


# ─── Insert ──────────────────────────────────────────────────────────────────


async def insert_envelope(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    envelope: SealedEnvelope,
) -> SealedEnvelope:
    """Insert a new SealedEnvelope. Raises UniqueViolationError on duplicate ID."""
    await conn.execute(
        """
        INSERT INTO sealed_envelopes
            (id, platform_id, purpose, ciphertext, key_version, created_at, last_accessed_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        envelope.id,
        envelope.platform_id,
        envelope.purpose,
        envelope.ciphertext,
        envelope.key_version,
        envelope.created_at,
        envelope.last_accessed_at,
    )
    logger.debug("envelope_inserted", envelope_id=envelope.id, platform_id=envelope.platform_id)
    return envelope


# ─── Retrieve ────────────────────────────────────────────────────────────────


async def get_envelope_by_id(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    envelope_id: str,
) -> SealedEnvelope | None:
    """Fetch by primary key. Returns None if not found."""
    row = await conn.fetchrow(
        """
        SELECT id, platform_id, purpose, ciphertext, key_version, created_at, last_accessed_at
        FROM sealed_envelopes
        WHERE id = $1
        """,
        envelope_id,
    )
    return _row_to_envelope(row) if row else None


async def get_envelopes_by_platform(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    platform_id: str,
) -> list[SealedEnvelope]:
    """Return all envelopes for a platform, ordered by created_at ascending."""
    rows = await conn.fetch(
        """
        SELECT id, platform_id, purpose, ciphertext, key_version, created_at, last_accessed_at
        FROM sealed_envelopes
        WHERE platform_id = $1
        ORDER BY created_at ASC
        """,
        platform_id,
    )
    return [_row_to_envelope(r) for r in rows]


async def get_all_envelopes(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
) -> list[SealedEnvelope]:
    """Return all envelopes across all platforms, ordered by platform_id then created_at."""
    rows = await conn.fetch(
        """
        SELECT id, platform_id, purpose, ciphertext, key_version, created_at, last_accessed_at
        FROM sealed_envelopes
        ORDER BY platform_id ASC, created_at ASC
        """,
    )
    return [_row_to_envelope(r) for r in rows]


async def count_envelopes(conn: asyncpg.Connection) -> int:  # type: ignore[type-arg]
    """Return total number of sealed envelopes."""
    row = await conn.fetchrow("SELECT COUNT(*) AS n FROM sealed_envelopes")
    return int(row["n"]) if row else 0


async def max_key_version(conn: asyncpg.Connection) -> int:  # type: ignore[type-arg]
    """Return the highest key_version present, or 1 if table is empty."""
    row = await conn.fetchrow("SELECT MAX(key_version) AS v FROM sealed_envelopes")
    v = row["v"] if row else None
    return int(v) if v is not None else 1


async def get_envelope_by_platform_and_purpose(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    platform_id: str,
    purpose: str,
) -> SealedEnvelope | None:
    """
    Fetch the most recent envelope for a platform + purpose pair.

    Returns None if no match.
    """
    row = await conn.fetchrow(
        """
        SELECT id, platform_id, purpose, ciphertext, key_version, created_at, last_accessed_at
        FROM sealed_envelopes
        WHERE platform_id = $1 AND purpose = $2
        ORDER BY created_at DESC
        LIMIT 1
        """,
        platform_id,
        purpose,
    )
    return _row_to_envelope(row) if row else None


# ─── Update ──────────────────────────────────────────────────────────────────


async def update_envelope(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    envelope: SealedEnvelope,
) -> bool:
    """
    Update ciphertext, key_version, and last_accessed_at for an existing row.

    Returns True if a row was matched, False if the ID was not found.
    """
    result = await conn.execute(
        """
        UPDATE sealed_envelopes
        SET ciphertext       = $2,
            key_version      = $3,
            last_accessed_at = $4,
            updated_at       = NOW()
        WHERE id = $1
        """,
        envelope.id,
        envelope.ciphertext,
        envelope.key_version,
        envelope.last_accessed_at,
    )
    updated = result.endswith("1")
    if updated:
        logger.debug("envelope_updated", envelope_id=envelope.id)
    else:
        logger.warning("envelope_update_not_found", envelope_id=envelope.id)
    return updated


async def upsert_envelope(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    envelope: SealedEnvelope,
) -> SealedEnvelope:
    """Insert or update a SealedEnvelope atomically (ON CONFLICT DO UPDATE)."""
    await conn.execute(
        """
        INSERT INTO sealed_envelopes
            (id, platform_id, purpose, ciphertext, key_version, created_at, last_accessed_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (id) DO UPDATE
            SET ciphertext       = EXCLUDED.ciphertext,
                key_version      = EXCLUDED.key_version,
                last_accessed_at = EXCLUDED.last_accessed_at,
                updated_at       = NOW()
        """,
        envelope.id,
        envelope.platform_id,
        envelope.purpose,
        envelope.ciphertext,
        envelope.key_version,
        envelope.created_at,
        envelope.last_accessed_at,
    )
    logger.debug("envelope_upserted", envelope_id=envelope.id, platform_id=envelope.platform_id)
    return envelope


# ─── Delete ──────────────────────────────────────────────────────────────────


async def delete_envelope_by_id(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    envelope_id: str,
) -> bool:
    """
    Hard-delete a SealedEnvelope by ID.

    Envelopes are intentionally hard-deleted — ciphertext encrypted under a
    rotated key is unrecoverable and should be purged rather than soft-deleted.
    Returns True if a row was deleted.
    """
    result = await conn.execute(
        "DELETE FROM sealed_envelopes WHERE id = $1",
        envelope_id,
    )
    deleted = result.endswith("1")
    if deleted:
        logger.info("envelope_deleted", envelope_id=envelope_id)
    else:
        logger.warning("envelope_delete_not_found", envelope_id=envelope_id)
    return deleted


async def delete_envelopes_by_platform(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    platform_id: str,
) -> int:
    """
    Delete all envelopes for a platform (e.g. on connector decommission).

    Returns the count of deleted rows.
    """
    result = await conn.execute(
        "DELETE FROM sealed_envelopes WHERE platform_id = $1",
        platform_id,
    )
    try:
        count = int(result.split()[-1])
    except (ValueError, IndexError):
        count = 0
    logger.info("envelopes_deleted_for_platform", platform_id=platform_id, count=count)
    return count


# ─── Internal ────────────────────────────────────────────────────────────────


def _row_to_envelope(row: asyncpg.Record) -> SealedEnvelope:  # type: ignore[type-arg]
    last_accessed: datetime | None = row["last_accessed_at"]
    return SealedEnvelope(
        id=row["id"],
        platform_id=row["platform_id"],
        purpose=row["purpose"],
        ciphertext=row["ciphertext"],
        key_version=row["key_version"],
        created_at=row["created_at"],
        last_accessed_at=last_accessed,
    )
