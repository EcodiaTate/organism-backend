"""
EcodiaOS - Identity CRUD

Async asyncpg queries for SealedEnvelope and ConnectorCredentials persistence.

Table schemas (created by ensure_table()):

    CREATE TABLE IF NOT EXISTS sealed_envelopes (
        id                TEXT PRIMARY KEY,
        platform_id       TEXT NOT NULL,
        purpose           TEXT NOT NULL,
        ciphertext        TEXT NOT NULL,
        key_version       INTEGER NOT NULL DEFAULT 1,
        created_at        TIMESTAMPTZ NOT NULL,
        last_accessed_at  TIMESTAMPTZ,
        updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        deleted_at        TIMESTAMPTZ
    );

    CREATE TABLE IF NOT EXISTS connector_credentials (
        id                      TEXT PRIMARY KEY,
        connector_id            TEXT NOT NULL UNIQUE,
        platform_id             TEXT NOT NULL,
        status                  TEXT NOT NULL DEFAULT 'unconfigured',
        token_envelope_id       TEXT NOT NULL DEFAULT '',
        totp_envelope_id        TEXT NOT NULL DEFAULT '',
        cookie_envelope_id      TEXT NOT NULL DEFAULT '',
        last_refresh_at         TIMESTAMPTZ,
        refresh_failure_count   INTEGER NOT NULL DEFAULT 0,
        metadata                JSONB NOT NULL DEFAULT '{}',
        created_at              TIMESTAMPTZ NOT NULL,
        updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        deleted_at              TIMESTAMPTZ
    );

All functions accept an asyncpg.Connection so callers can participate
in an outer transaction if needed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from datetime import datetime

import structlog

from systems.identity.connector import ConnectorCredentials, ConnectorStatus
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
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at        TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sealed_envelopes_platform_id
    ON sealed_envelopes (platform_id);
CREATE TABLE IF NOT EXISTS connector_credentials (
    id                      TEXT PRIMARY KEY,
    connector_id            TEXT NOT NULL,
    platform_id             TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'unconfigured',
    token_envelope_id       TEXT NOT NULL DEFAULT '',
    totp_envelope_id        TEXT NOT NULL DEFAULT '',
    cookie_envelope_id      TEXT NOT NULL DEFAULT '',
    last_refresh_at         TIMESTAMPTZ,
    refresh_failure_count   INTEGER NOT NULL DEFAULT 0,
    metadata                JSONB NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ NOT NULL,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at              TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_connector_credentials_connector_id
    ON connector_credentials (connector_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_connector_credentials_platform_id
    ON connector_credentials (platform_id) WHERE deleted_at IS NULL;
"""


async def ensure_table(conn: asyncpg.Connection) -> None:  # type: ignore[type-arg]
    """Create sealed_envelopes and connector_credentials tables if not present."""
    for stmt in _CREATE_TABLE_SQL.split(";"):
        s = stmt.strip()
        if s:
            await conn.execute(s)
    logger.debug("identity_tables_ensured")


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
    """Return all non-deleted envelopes for a platform, ordered by created_at ascending."""
    rows = await conn.fetch(
        """
        SELECT id, platform_id, purpose, ciphertext, key_version, created_at, last_accessed_at
        FROM sealed_envelopes
        WHERE platform_id = $1 AND deleted_at IS NULL
        ORDER BY created_at ASC
        """,
        platform_id,
    )
    return [_row_to_envelope(r) for r in rows]


async def get_all_envelopes(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
) -> list[SealedEnvelope]:
    """Return all non-deleted envelopes across all platforms, ordered by platform_id then created_at."""
    rows = await conn.fetch(
        """
        SELECT id, platform_id, purpose, ciphertext, key_version, created_at, last_accessed_at
        FROM sealed_envelopes
        WHERE deleted_at IS NULL
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
    Fetch the most recent non-deleted envelope for a platform + purpose pair.

    Returns None if no match.
    """
    row = await conn.fetchrow(
        """
        SELECT id, platform_id, purpose, ciphertext, key_version, created_at, last_accessed_at
        FROM sealed_envelopes
        WHERE platform_id = $1 AND purpose = $2 AND deleted_at IS NULL
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
    Soft-delete a SealedEnvelope by ID (sets deleted_at = NOW()).

    Soft-delete preserves the audit trail in Neo4j and complies with the
    EcodiaOS soft-delete rule. Returns True if a row was marked deleted.
    """
    result = await conn.execute(
        "UPDATE sealed_envelopes SET deleted_at = NOW() WHERE id = $1 AND deleted_at IS NULL",
        envelope_id,
    )
    deleted = result.endswith("1")
    if deleted:
        logger.info("envelope_soft_deleted", envelope_id=envelope_id)
    else:
        logger.warning("envelope_delete_not_found", envelope_id=envelope_id)
    return deleted


async def delete_envelopes_by_platform(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    platform_id: str,
) -> int:
    """
    Soft-delete all envelopes for a platform (e.g. on connector decommission).

    Returns the count of rows marked deleted.
    """
    result = await conn.execute(
        "UPDATE sealed_envelopes SET deleted_at = NOW() WHERE platform_id = $1 AND deleted_at IS NULL",
        platform_id,
    )
    try:
        count = int(result.split()[-1])
    except (ValueError, IndexError):
        count = 0
    logger.info("envelopes_soft_deleted_for_platform", platform_id=platform_id, count=count)
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


# ─── ConnectorCredentials CRUD ───────────────────────────────────────────────


async def get_all_credentials(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
) -> list[ConnectorCredentials]:
    """
    Return all non-deleted ConnectorCredentials, ordered by platform_id then created_at.

    Called at boot to restore all connector token state from the database.
    Without this, connector state (status, envelope IDs, failure counts) is lost
    on process restart.
    """
    rows = await conn.fetch(
        """
        SELECT id, connector_id, platform_id, status, token_envelope_id, totp_envelope_id,
               cookie_envelope_id, last_refresh_at, refresh_failure_count, metadata, created_at
        FROM connector_credentials
        WHERE deleted_at IS NULL
        ORDER BY platform_id ASC, created_at ASC
        """,
    )
    return [_row_to_credentials(r) for r in rows]


async def get_credentials_by_connector_id(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    connector_id: str,
) -> ConnectorCredentials | None:
    """Fetch a single non-deleted ConnectorCredentials by connector_id."""
    row = await conn.fetchrow(
        """
        SELECT id, connector_id, platform_id, status, token_envelope_id, totp_envelope_id,
               cookie_envelope_id, last_refresh_at, refresh_failure_count, metadata, created_at
        FROM connector_credentials
        WHERE connector_id = $1 AND deleted_at IS NULL
        """,
        connector_id,
    )
    return _row_to_credentials(row) if row else None


async def upsert_credential(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    cred: ConnectorCredentials,
) -> ConnectorCredentials:
    """
    Insert or update a ConnectorCredentials record atomically.

    Uses connector_id as the natural key. On conflict, updates all mutable
    fields. The deleted_at column is reset to NULL so a previously soft-deleted
    credential is effectively restored.
    """
    metadata_json = json.dumps(cred.metadata, sort_keys=True)
    await conn.execute(
        """
        INSERT INTO connector_credentials
            (id, connector_id, platform_id, status, token_envelope_id, totp_envelope_id,
             cookie_envelope_id, last_refresh_at, refresh_failure_count, metadata, created_at,
             updated_at, deleted_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, NOW(), NULL)
        ON CONFLICT (connector_id) DO UPDATE
            SET status                = EXCLUDED.status,
                token_envelope_id     = EXCLUDED.token_envelope_id,
                totp_envelope_id      = EXCLUDED.totp_envelope_id,
                cookie_envelope_id    = EXCLUDED.cookie_envelope_id,
                last_refresh_at       = EXCLUDED.last_refresh_at,
                refresh_failure_count = EXCLUDED.refresh_failure_count,
                metadata              = EXCLUDED.metadata,
                updated_at            = NOW(),
                deleted_at            = NULL
        """,
        cred.id,
        cred.connector_id,
        cred.platform_id,
        cred.status.value,
        cred.token_envelope_id,
        cred.totp_envelope_id,
        cred.cookie_envelope_id,
        cred.last_refresh_at,
        cred.refresh_failure_count,
        metadata_json,
        cred.created_at,
    )
    logger.debug(
        "connector_credentials_upserted",
        connector_id=cred.connector_id,
        platform_id=cred.platform_id,
    )
    return cred


async def delete_credential(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    connector_id: str,
) -> bool:
    """
    Soft-delete ConnectorCredentials by connector_id (sets deleted_at = NOW()).

    Returns True if a row was marked deleted.
    """
    result = await conn.execute(
        """
        UPDATE connector_credentials
        SET deleted_at = NOW(), updated_at = NOW()
        WHERE connector_id = $1 AND deleted_at IS NULL
        """,
        connector_id,
    )
    deleted = result.endswith("1")
    if deleted:
        logger.info("connector_credentials_soft_deleted", connector_id=connector_id)
    else:
        logger.warning("connector_credentials_delete_not_found", connector_id=connector_id)
    return deleted


def _row_to_credentials(row: asyncpg.Record) -> ConnectorCredentials:  # type: ignore[type-arg]
    raw_meta = row["metadata"]
    metadata: dict[str, str] = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})
    return ConnectorCredentials(
        id=row["id"],
        connector_id=row["connector_id"],
        platform_id=row["platform_id"],
        status=ConnectorStatus(row["status"]),
        token_envelope_id=row["token_envelope_id"] or "",
        totp_envelope_id=row["totp_envelope_id"] or "",
        cookie_envelope_id=row["cookie_envelope_id"] or "",
        last_refresh_at=row["last_refresh_at"],
        refresh_failure_count=row["refresh_failure_count"],
        metadata=metadata,
        created_at=row["created_at"],
    )
