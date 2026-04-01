"""
EcodiaOS - Identity API Router

Admin endpoints for managing SealedEnvelope credential records and
querying vault, certificate, and connector health.

Envelope endpoints:
  POST   /api/v1/identity/envelopes              - insert a new envelope
  GET    /api/v1/identity/envelopes/{id}          - fetch envelope by ID
  GET    /api/v1/identity/envelopes               - list (all or by platform_id)
  PUT    /api/v1/identity/envelopes/{id}          - replace ciphertext / key_version
  DELETE /api/v1/identity/envelopes/{id}          - hard-delete a single envelope
  DELETE /api/v1/identity/envelopes/platform/{id} - bulk-delete all for a platform

Identity dashboard endpoints:
  GET    /api/v1/identity/health                        - overall system health
  GET    /api/v1/identity/certificate                   - current EcodianCertificate
  GET    /api/v1/identity/connectors                    - all connector credential records
  POST   /api/v1/identity/connectors/{id}/refresh       - trigger token refresh
  POST   /api/v1/identity/connectors/{id}/revoke        - revoke OAuth token
  GET    /api/v1/identity/vault/status                  - vault metadata (no ciphertext)
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic needs at runtime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now
from systems.identity import crud
from systems.identity.certificate import CertificateStatus, CertificateType
from systems.identity.connector import ConnectorStatus
from systems.identity.vault import SealedEnvelope, VaultConfig

logger = structlog.get_logger("api.identity")

router = APIRouter(prefix="/api/v1/identity", tags=["identity"])


# ─── Schemas ─────────────────────────────────────────────────────────────────


class EnvelopeCreateRequest(EOSBaseModel):
    platform_id: str = Field(..., description="Platform identifier e.g. 'linkedin'")
    purpose: str = Field(..., description="'oauth_token' | 'totp_secret' | 'cookie_state'")
    ciphertext: str = Field(..., description="Fernet-encrypted ciphertext (base64 urlsafe)")
    key_version: int = Field(default=1, ge=1)
    id: str = Field(default_factory=new_id)


class EnvelopeUpdateRequest(EOSBaseModel):
    ciphertext: str = Field(..., description="Replacement Fernet-encrypted ciphertext")
    key_version: int = Field(..., ge=1)


class EnvelopeResponse(EOSBaseModel):
    id: str
    platform_id: str
    purpose: str
    ciphertext: str
    key_version: int
    created_at: datetime
    last_accessed_at: datetime | None


def _to_response(env: SealedEnvelope) -> dict[str, Any]:
    return EnvelopeResponse(
        id=env.id,
        platform_id=env.platform_id,
        purpose=env.purpose,
        ciphertext=env.ciphertext,
        key_version=env.key_version,
        created_at=env.created_at,
        last_accessed_at=env.last_accessed_at,
    ).model_dump(mode="json")


def _to_envelope_item(env: SealedEnvelope) -> dict[str, Any]:
    """Envelope summary for dashboard - omits ciphertext."""
    return {
        "id": env.id,
        "platform_id": env.platform_id,
        "purpose": env.purpose,
        "key_version": env.key_version,
        "created_at": env.created_at.isoformat(),
        "last_accessed_at": env.last_accessed_at.isoformat() if env.last_accessed_at else None,
    }


def _pool(request: Request):  # type: ignore[return]
    """Resolve the asyncpg pool from app.state.tsdb (shared Postgres pool)."""
    tsdb = getattr(request.app.state, "tsdb", None)
    if tsdb is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return tsdb.pool


# ─── Routes ──────────────────────────────────────────────────────────────────


@router.post("/envelopes", status_code=201)
async def create_envelope(body: EnvelopeCreateRequest, request: Request) -> JSONResponse:
    """
    Inject an initial SealedEnvelope.

    The ciphertext must already be encrypted by IdentityVault - this endpoint
    does not handle plaintext secrets.
    """
    pool = _pool(request)
    envelope = SealedEnvelope(
        id=body.id,
        platform_id=body.platform_id,
        purpose=body.purpose,
        ciphertext=body.ciphertext,
        key_version=body.key_version,
        created_at=utc_now(),
    )

    try:
        async with pool.acquire() as conn:
            await crud.ensure_table(conn)
            inserted = await crud.insert_envelope(conn, envelope)
    except Exception as exc:
        err = str(exc).lower()
        if "unique" in err or "duplicate" in err:
            raise HTTPException(status_code=409, detail=f"Envelope '{body.id}' already exists")
        logger.error("envelope_create_failed", error=str(exc), platform_id=body.platform_id)
        raise HTTPException(status_code=500, detail="Failed to persist envelope")

    logger.info("envelope_created_via_api", envelope_id=inserted.id, platform_id=inserted.platform_id)
    return JSONResponse(status_code=201, content=_to_response(inserted))


@router.get("/envelopes/{envelope_id}")
async def get_envelope(envelope_id: str, request: Request) -> JSONResponse:
    """Fetch a SealedEnvelope by its ULID."""
    pool = _pool(request)
    async with pool.acquire() as conn:
        await crud.ensure_table(conn)
        envelope = await crud.get_envelope_by_id(conn, envelope_id)

    if envelope is None:
        raise HTTPException(status_code=404, detail=f"Envelope '{envelope_id}' not found")
    return JSONResponse(content=_to_response(envelope))


@router.get("/envelopes")
async def list_envelopes(
    request: Request,
    platform_id: str | None = Query(None, description="Platform identifier to filter by (omit for all)"),
    purpose: str | None = Query(None, description="Optional purpose to narrow results (requires platform_id)"),
) -> JSONResponse:
    """
    List SealedEnvelopes.

    - No platform_id: returns all envelopes across all platforms.
    - With platform_id + purpose: returns at most one record (the most recent match).
    - With platform_id only: returns all envelopes for the platform.

    Response shape: { envelopes: [...], total: int }
    """
    pool = _pool(request)
    async with pool.acquire() as conn:
        await crud.ensure_table(conn)
        if platform_id is None:
            envelopes = await crud.get_all_envelopes(conn)
        elif purpose is not None:
            single = await crud.get_envelope_by_platform_and_purpose(conn, platform_id, purpose)
            envelopes = [single] if single else []
        else:
            envelopes = await crud.get_envelopes_by_platform(conn, platform_id)

    items = [_to_envelope_item(e) for e in envelopes]
    return JSONResponse(content={"envelopes": items, "total": len(items)})


@router.put("/envelopes/{envelope_id}")
async def update_envelope(
    envelope_id: str,
    body: EnvelopeUpdateRequest,
    request: Request,
) -> JSONResponse:
    """
    Replace the ciphertext of an existing envelope.

    Intended for use after vault key rotation - the caller re-encrypts
    the secret under the new key and pushes the new ciphertext here.
    """
    pool = _pool(request)
    async with pool.acquire() as conn:
        await crud.ensure_table(conn)
        existing = await crud.get_envelope_by_id(conn, envelope_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Envelope '{envelope_id}' not found")

        existing.ciphertext = body.ciphertext
        existing.key_version = body.key_version
        existing.last_accessed_at = utc_now()
        ok = await crud.update_envelope(conn, existing)

    if not ok:
        raise HTTPException(status_code=500, detail="Update command matched no rows")

    logger.info("envelope_updated_via_api", envelope_id=envelope_id)
    return JSONResponse(content=_to_response(existing))


@router.delete("/envelopes/{envelope_id}", status_code=204)
async def delete_envelope(envelope_id: str, request: Request) -> JSONResponse:
    """Hard-delete a single SealedEnvelope."""
    pool = _pool(request)
    async with pool.acquire() as conn:
        await crud.ensure_table(conn)
        deleted = await crud.delete_envelope_by_id(conn, envelope_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Envelope '{envelope_id}' not found")

    logger.info("envelope_deleted_via_api", envelope_id=envelope_id)
    return JSONResponse(status_code=204, content=None)


@router.delete("/envelopes/platform/{platform_id}")
async def delete_platform_envelopes(platform_id: str, request: Request) -> JSONResponse:
    """Bulk-delete all envelopes for a platform (e.g. on connector decommission)."""
    pool = _pool(request)
    async with pool.acquire() as conn:
        await crud.ensure_table(conn)
        count = await crud.delete_envelopes_by_platform(conn, platform_id)

    logger.info("platform_envelopes_deleted_via_api", platform_id=platform_id, count=count)
    return JSONResponse(content={"deleted": count, "platform_id": platform_id})


# ─── Dashboard: Pydantic Response Models ─────────────────────────────────────


class VaultHealthInfo(EOSBaseModel):
    initialized: bool
    passphrase_configured: bool
    envelope_count: int


class CertificateHealthInfo(EOSBaseModel):
    status: CertificateStatus
    days_remaining: float
    type: CertificateType


class ConnectorCounts(EOSBaseModel):
    total: int
    active: int
    degraded: int


class IdentityHealthResponse(EOSBaseModel):
    status: str  # "healthy" | "degraded" | "error"
    vault: VaultHealthInfo
    certificate: CertificateHealthInfo | None
    connectors: ConnectorCounts


class IdentityCertificateResponse(EOSBaseModel):
    certificate_id: str
    instance_id: str
    certificate_type: CertificateType
    issuer_instance_id: str
    issued_at: datetime
    expires_at: datetime
    validity_days: int
    renewal_count: int
    status: CertificateStatus
    days_remaining: float
    lineage_hash: str
    constitutional_hash: str
    protocol_version: str


class ConnectorItem(EOSBaseModel):
    connector_id: str
    platform_id: str
    status: ConnectorStatus
    last_refresh_at: datetime | None
    refresh_failure_count: int
    metadata: dict[str, str]
    token_expires_at: datetime | None
    token_remaining_seconds: float | None


class IdentityConnectorsResponse(EOSBaseModel):
    connectors: list[ConnectorItem]
    total: int
    active: int
    degraded: int


class ConnectorActionResponse(EOSBaseModel):
    connector_id: str
    success: bool
    message: str


class VaultStatusResponse(EOSBaseModel):
    initialized: bool
    envelope_count: int
    key_version: int
    pbkdf2_iterations: int
    passphrase_configured: bool


# ─── Dashboard: Helper ───────────────────────────────────────────────────────


def _passphrase_configured() -> bool:
    """Return True if ECODIAOS_VAULT_PASSPHRASE env var is set and non-empty."""
    import os as _os
    return bool(_os.environ.get("ECODIAOS_VAULT_PASSPHRASE", ""))


# ─── Dashboard: Routes ───────────────────────────────────────────────────────


@router.get("/health")
async def identity_health(request: Request) -> JSONResponse:
    """
    Overall identity system health.

    Aggregates vault passphrase status, certificate validity, and
    connector health derived from sealed envelope records.
    """
    pool = _pool(request)

    # Vault info
    passphrase_ok = _passphrase_configured()
    async with pool.acquire() as conn:
        await crud.ensure_table(conn)
        envelope_count = await crud.count_envelopes(conn)

    vault_info = VaultHealthInfo(
        initialized=passphrase_ok,
        passphrase_configured=passphrase_ok,
        envelope_count=envelope_count,
    )

    # Certificate info
    cert_info: CertificateHealthInfo | None = None
    cert_manager = getattr(request.app.state, "certificate_manager", None)
    if cert_manager is not None:
        cert = cert_manager.certificate
        if cert is not None:
            cert_info = CertificateHealthInfo(
                status=cert.status,
                days_remaining=cert.remaining_days,
                type=cert.certificate_type,
            )

    # Connector counts from in-process registry.
    _conn_map: dict[str, Any] = getattr(request.app.state, "connectors", {}) or {}
    _conn_list = [c for c in _conn_map.values() if c.credentials is not None]
    connector_counts = ConnectorCounts(
        total=len(_conn_list),
        active=sum(1 for c in _conn_list if c.status == ConnectorStatus.ACTIVE),
        degraded=sum(
            1 for c in _conn_list
            if c.status in (ConnectorStatus.REFRESH_FAILED, ConnectorStatus.ERROR)
        ),
    )

    # Determine overall status
    overall = "healthy"
    if not passphrase_ok or cert_info is not None and cert_info.status == CertificateStatus.EXPIRED:
        overall = "error"
    elif cert_info is not None and cert_info.status == CertificateStatus.EXPIRING_SOON:
        overall = "degraded"

    response = IdentityHealthResponse(
        status=overall,
        vault=vault_info,
        certificate=cert_info,
        connectors=connector_counts,
    )
    return JSONResponse(content=response.model_dump(mode="json"))


@router.get("/certificate")
async def identity_certificate(request: Request) -> JSONResponse:
    """
    Return the current EcodianCertificate.

    Returns 503 if the certificate manager is not initialized (Oikos disabled).
    Returns 404 if no certificate is loaded.
    """
    cert_manager = getattr(request.app.state, "certificate_manager", None)
    if cert_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Certificate manager not initialized (Oikos may be disabled)",
        )

    cert = cert_manager.certificate
    if cert is None:
        raise HTTPException(status_code=404, detail="No certificate loaded")

    response = IdentityCertificateResponse(
        certificate_id=cert.certificate_id,
        instance_id=cert.instance_id,
        certificate_type=cert.certificate_type,
        issuer_instance_id=cert.issuer_instance_id,
        issued_at=cert.issued_at,
        expires_at=cert.expires_at,
        validity_days=cert.validity_days,
        renewal_count=cert.renewal_count,
        status=cert.status,
        days_remaining=cert.remaining_days,
        lineage_hash=cert.lineage_hash,
        constitutional_hash=cert.constitutional_hash,
        protocol_version=cert.protocol_version,
    )
    return JSONResponse(content=response.model_dump(mode="json"))


@router.get("/connectors")
async def list_connectors(request: Request) -> JSONResponse:
    """
    Return all registered connector credential records.

    Reads from app.state.connectors populated during the lifespan phase 15g.
    Returns an empty list if no connectors are configured.
    """
    connectors_map: dict[str, Any] = getattr(request.app.state, "connectors", {}) or {}
    items: list[ConnectorItem] = []

    for connector in connectors_map.values():
        creds = connector.credentials
        if creds is None:
            continue

        # Derive token expiry from Redis cache (cheap) or leave null.
        token_expires_at: datetime | None = None
        token_remaining_seconds: float | None = None
        try:
            cached = await connector._read_token_cache()
            if cached is not None:
                token_expires_at = cached.expires_at
                token_remaining_seconds = cached.remaining_seconds
        except Exception:
            pass

        items.append(ConnectorItem(
            connector_id=creds.connector_id,
            platform_id=connector.platform_id,
            status=connector.status,
            last_refresh_at=creds.last_refresh_at,
            refresh_failure_count=creds.refresh_failure_count,
            metadata=creds.metadata,
            token_expires_at=token_expires_at,
            token_remaining_seconds=token_remaining_seconds,
        ))

    response = IdentityConnectorsResponse(
        connectors=items,
        total=len(items),
        active=sum(1 for c in items if c.status == ConnectorStatus.ACTIVE),
        degraded=sum(
            1
            for c in items
            if c.status in (ConnectorStatus.REFRESH_FAILED, ConnectorStatus.ERROR)
        ),
    )
    return JSONResponse(content=response.model_dump(mode="json"))


@router.post("/connectors/{connector_id}/refresh")
async def refresh_connector(connector_id: str, request: Request) -> JSONResponse:
    """
    Trigger a token refresh for the named connector.

    Requires the connector to be available in the in-process connector
    registry (app.state). In the current architecture connectors run in
    the standalone skia_worker - this endpoint returns 404 when not found.
    """
    # Attempt to locate connector in app.state (future extensibility)
    connectors_map: dict[str, Any] = getattr(request.app.state, "connectors", {}) or {}
    connector = connectors_map.get(connector_id)

    if connector is None:
        raise HTTPException(
            status_code=404,
            detail=f"Connector '{connector_id}' not found in process registry",
        )

    try:
        token = await connector.get_access_token()
        success = token is not None
        message = "Token refreshed successfully" if success else "Refresh failed - no token returned"
    except Exception as exc:
        logger.error("connector_refresh_failed", connector_id=connector_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Refresh failed: {exc}") from exc

    response = ConnectorActionResponse(
        connector_id=connector_id,
        success=success,
        message=message,
    )
    return JSONResponse(content=response.model_dump(mode="json"))


@router.post("/connectors/{connector_id}/revoke")
async def revoke_connector(connector_id: str, request: Request) -> JSONResponse:
    """
    Revoke the OAuth token for the named connector.

    Requires the connector to be in the in-process registry.
    Returns 404 when not found.
    """
    connectors_map: dict[str, Any] = getattr(request.app.state, "connectors", {}) or {}
    connector = connectors_map.get(connector_id)

    if connector is None:
        raise HTTPException(
            status_code=404,
            detail=f"Connector '{connector_id}' not found in process registry",
        )

    try:
        ok = await connector.revoke()
    except Exception as exc:
        logger.error("connector_revoke_failed", connector_id=connector_id, error=str(exc))
        ok = False

    response = ConnectorActionResponse(
        connector_id=connector_id,
        success=ok,
        message="Token revoked" if ok else "Revocation failed or not supported",
    )
    return JSONResponse(content=response.model_dump(mode="json"))


@router.get("/vault/status")
async def vault_status(request: Request) -> JSONResponse:
    """
    Return vault metadata.

    Never returns ciphertext or decrypted secrets - metadata only.
    """
    pool = _pool(request)
    passphrase_ok = _passphrase_configured()

    async with pool.acquire() as conn:
        await crud.ensure_table(conn)
        envelope_count = await crud.count_envelopes(conn)
        key_ver = await crud.max_key_version(conn)

    default_config = VaultConfig()

    response = VaultStatusResponse(
        initialized=passphrase_ok,
        envelope_count=envelope_count,
        key_version=key_ver,
        pbkdf2_iterations=default_config.pbkdf2_iterations,
        passphrase_configured=passphrase_ok,
    )
    return JSONResponse(content=response.model_dump(mode="json"))
