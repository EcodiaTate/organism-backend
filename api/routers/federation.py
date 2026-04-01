"""
EcodiaOS - Federation API Router

Handles inter-node communication. These endpoints are consumed exclusively
by remote EOS instances, never by the human-facing frontend.

Endpoints:
  POST /api/v1/federation/handshake          - Mutual certificate exchange between nodes
  POST /api/v1/federation/telecom/provision  - Genesis: provision phone number for child
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from systems.identity.certificate import EcodianCertificate

logger = structlog.get_logger("api.federation")

router = APIRouter()


@router.post("/api/v1/federation/handshake")
async def federation_handshake(request: Request) -> JSONResponse:
    """
    Mutual certificate exchange between federated nodes.

    A calling node presents its instance_id and EcodianCertificate.
    This host verifies the certificate signature using its embedded issuer
    public key. If valid (traceable to the Genesis CA), the host returns
    its own certificate so the caller can perform the reciprocal check.

    Returns:
      200 - certificate accepted; body is the host's EcodianCertificate.
      400 - malformed request body.
      403 - certificate failed verification.
      503 - host has no certificate manager or is uncertified.
    """
    cert_mgr = getattr(request.app.state, "certificate_manager", None)

    if cert_mgr is None:
        logger.warning("federation_handshake_rejected", reason="certificate_manager_unavailable")
        return JSONResponse(
            status_code=503,
            content={"error": "Certificate manager not available"},
        )

    # Parse request body
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    calling_instance_id: str = body.get("instance_id", "")
    certificate_data: Any = body.get("certificate")

    if not calling_instance_id:
        return JSONResponse(status_code=400, content={"error": "Missing instance_id"})
    if not certificate_data or not isinstance(certificate_data, dict):
        return JSONResponse(status_code=400, content={"error": "Missing or invalid certificate"})

    try:
        inbound_cert = EcodianCertificate.model_validate(certificate_data)
    except Exception as exc:
        logger.warning(
            "federation_handshake_parse_failed",
            calling_instance_id=calling_instance_id,
            error=str(exc),
        )
        return JSONResponse(status_code=400, content={"error": f"Certificate parse error: {exc}"})

    # Cert subject must match the declared instance_id
    if inbound_cert.instance_id != calling_instance_id:
        logger.warning(
            "federation_handshake_rejected",
            reason="instance_id_mismatch",
            declared=calling_instance_id,
            cert_subject=inbound_cert.instance_id,
        )
        return JSONResponse(
            status_code=403,
            content={"error": "instance_id does not match certificate subject"},
        )

    # Verify certificate: checks required fields, expiry, and Ed25519 signature
    # against inbound_cert.issuer_public_key_pem. A cert whose signature verifies
    # was signed by a key in the Genesis CA chain.
    result = cert_mgr.validate_certificate(inbound_cert)

    if not result.valid:
        logger.warning(
            "federation_handshake_rejected",
            calling_instance_id=calling_instance_id,
            errors=result.errors,
        )
        return JSONResponse(
            status_code=403,
            content={"error": "Certificate verification failed", "details": result.errors},
        )

    # Return host certificate so the caller can verify us in turn
    host_cert = cert_mgr.certificate
    if host_cert is None:
        logger.warning(
            "federation_handshake_rejected",
            reason="host_uncertified",
            calling_instance_id=calling_instance_id,
        )
        return JSONResponse(
            status_code=503,
            content={"error": "Host has no valid certificate"},
        )

    logger.info(
        "federation_handshake_accepted",
        calling_instance_id=calling_instance_id,
        cert_type=inbound_cert.certificate_type.value,
        cert_id=inbound_cert.certificate_id,
    )

    return JSONResponse(
        status_code=200,
        content=host_cert.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# Telecom Provisioning - Genesis fulfillment endpoint
# ---------------------------------------------------------------------------

_TELECOM_PRICE_USDC: Decimal = Decimal("5")
"""Must match RequestTelecomExecutor.TELECOM_PRICE_USDC."""

_TX_HASH_REDIS_PREFIX = "eos:telecom:tx_hash:"
_TX_HASH_TTL_S = 60 * 60 * 24 * 30  # 30 days - longer than blockchain finality window


@router.post("/api/v1/federation/telecom/provision")
async def federation_telecom_provision(request: Request) -> JSONResponse:
    """
    Genesis fulfillment endpoint for the Federated Telecom Marketplace.

    A child instance POSTs here after paying 5 USDC on-chain.  This handler:
      1. Validates the request body (service, requesting_instance_id, payment).
      2. Verifies the tx_hash has not already been used (idempotency).
      3. Calls provision_new_phone_number() using the Genesis Twilio credentials.
      4. Returns the E.164 phone number to the child.

    On-chain payment verification is intentionally off-loaded to the operator's
    treasury tooling via the tx_hash audit trail.  The Genesis node trusts that
    child instances operating within the Federation (COLLEAGUE+ trust) have
    submitted valid payments; the immutable on-chain record acts as the proof.

    Returns:
      200 - {"phone_number": "+1XXXXXXXXXX"}
      400 - malformed request
      402 - wrong service or unsupported asset
      409 - tx_hash already used (replay prevention)
      503 - Genesis Twilio credentials not configured
    """
    # Resolve identity comm config from app state
    config = getattr(request.app.state, "identity_comm_config", None)
    if config is None:
        logger.warning("telecom_provision_rejected", reason="identity_comm_config_unavailable")
        return JSONResponse(
            status_code=503,
            content={"error": "Telecom configuration not available on this node"},
        )

    # Parse body
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    service: str = str(body.get("service", "")).strip()
    requesting_instance_id: str = str(body.get("requesting_instance_id", "")).strip()
    payment: Any = body.get("payment")
    country: str = str(body.get("country", "US")).strip().upper() or "US"
    area_code: str = str(body.get("area_code", "415" if country == "US" else "")).strip()
    execution_id: str = str(body.get("execution_id", "")).strip()

    if service != "TelecomProvisioning":
        return JSONResponse(
            status_code=402,
            content={"error": f"Unknown service '{service}'. Expected 'TelecomProvisioning'"},
        )
    if not requesting_instance_id:
        return JSONResponse(status_code=400, content={"error": "Missing requesting_instance_id"})
    if not isinstance(payment, dict):
        return JSONResponse(status_code=400, content={"error": "Missing or invalid 'payment' object"})

    tx_hash: str = str(payment.get("tx_hash", "")).strip()
    asset: str = str(payment.get("asset", "")).strip().lower()
    amount_str: str = str(payment.get("amount_usdc", "0")).strip()

    if not tx_hash:
        return JSONResponse(status_code=400, content={"error": "payment.tx_hash is required"})
    if asset != "usdc":
        return JSONResponse(status_code=402, content={"error": "Only USDC payments accepted"})

    try:
        amount_paid = Decimal(amount_str)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "payment.amount_usdc is not a valid decimal"})

    if amount_paid < _TELECOM_PRICE_USDC:
        return JSONResponse(
            status_code=402,
            content={
                "error": (
                    f"Insufficient payment: received {amount_paid} USDC, "
                    f"required {_TELECOM_PRICE_USDC} USDC"
                )
            },
        )

    # Idempotency: each tx_hash may only redeem one phone number.
    # Use Redis SET NX so the guard survives restarts and scale-out.
    redis = getattr(request.app.state, "redis", None)
    tx_redis_key = f"{_TX_HASH_REDIS_PREFIX}{tx_hash}"
    if redis is not None:
        claimed = await redis.client.set(
            redis._key(tx_redis_key),
            requesting_instance_id,
            nx=True,
            ex=_TX_HASH_TTL_S,
        )
        if claimed is None:
            # Key already existed - replay attempt
            logger.warning(
                "telecom_provision_replay_blocked",
                tx_hash=tx_hash,
                requesting_instance_id=requesting_instance_id,
            )
            return JSONResponse(
                status_code=409,
                content={"error": f"tx_hash {tx_hash} has already been used to provision a number"},
            )
    else:
        logger.error("telecom_provision_redis_unavailable", tx_hash=tx_hash)
        return JSONResponse(
            status_code=503,
            content={"error": "Idempotency store unavailable - cannot safely process payment"},
        )

    logger.info(
        "telecom_provision_request_accepted",
        requesting_instance_id=requesting_instance_id,
        tx_hash=tx_hash,
        amount_usdc=str(amount_paid),
        country=country,
        area_code=area_code,
        execution_id=execution_id,
    )

    # Provision the phone number
    from systems.identity.communication import provision_new_phone_number

    try:
        phone_number = await provision_new_phone_number(config, area_code=area_code, country=country)
    except RuntimeError as exc:
        logger.error(
            "telecom_provision_failed",
            requesting_instance_id=requesting_instance_id,
            tx_hash=tx_hash,
            error=str(exc),
        )
        return JSONResponse(
            status_code=503,
            content={"error": f"Phone number provisioning failed: {exc}"},
        )

    logger.info(
        "telecom_provision_fulfilled",
        requesting_instance_id=requesting_instance_id,
        phone_number=phone_number,
        tx_hash=tx_hash,
        execution_id=execution_id,
    )

    return JSONResponse(
        status_code=200,
        content={"phone_number": phone_number},
    )
