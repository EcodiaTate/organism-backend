"""
EcodiaOS - Tollbooth Router

External revenue endpoints:
  POST /api/v1/voxis/generate      - Personality-aware content generation
  POST /api/v1/knowledge/query     - ArXiv-backed knowledge retrieval
  POST /api/v1/webhooks/payment    - Inbound payment notification receiver
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request

from api.monetization.auth import _get_ledger, require_tollbooth_key
from api.monetization.ledger import (
    CreditLedger,  # noqa: TC001 - used at runtime in Depends()
)
from api.monetization.types import (
    PRODUCT_COST,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
    KnowledgeResult,
    PaymentWebhookPayload,
    RotateKeyResponse,
    TollboothError,
    TollboothProduct,
    VoxisGenerateRequest,
    VoxisGenerateResponse,
    WebhookAckResponse,
)

logger = structlog.get_logger("api.monetization.router")

router = APIRouter(tags=["tollbooth"])

# ─── Internal stubs (mock the real subsystem calls) ──────────────


async def _mock_voxis_generate(prompt: str, max_tokens: int) -> tuple[str, int]:
    """
    Stub for VoxisService.express().

    Returns (generated_text, tokens_used).  Replace with the real
    service call once Voxis is wired into the Tollbooth lifespan.
    """
    placeholder = (
        f"[Voxis stub] Generated response for prompt "
        f"({len(prompt)} chars, max_tokens={max_tokens})"
    )
    return placeholder, min(max_tokens, len(prompt) // 4 + 1)


async def _mock_knowledge_query(
    query: str, top_k: int, categories: list[str]
) -> list[KnowledgeResult]:
    """
    Stub for ArxivScientist / Memory knowledge retrieval.

    Returns a synthetic result list.  Replace with the real pipeline
    once the knowledge index is exposed as a service.
    """
    return [
        KnowledgeResult(
            title=f"[Stub] Result {i + 1} for '{query[:40]}'",
            summary="Placeholder - real results will come from the ArXiv pipeline.",
            arxiv_id=f"2024.{10000 + i}",
            relevance_score=round(1.0 - i * 0.15, 2),
        )
        for i in range(min(top_k, 3))
    ]


# ─── Billing helper ──────────────────────────────────────────────


async def _charge(
    ledger: CreditLedger, api_key: str, product: TollboothProduct
) -> int:
    """
    Debit the product cost.  Returns new balance or raises 402.
    """
    cost = PRODUCT_COST[product]
    new_balance = await ledger.debit(api_key, cost)
    if new_balance is None:
        current = await ledger.balance(api_key)
        raise HTTPException(
            status_code=402,
            detail=(
                f"Insufficient credits. Required: {cost}, "
                f"available: {current}. Top up via payment webhook."
            ),
        )
    return new_balance


# ─── Endpoints ───────────────────────────────────────────────────


@router.get("/api/v1/tollbooth/balance")
async def tollbooth_balance(
    api_key: Annotated[str, Depends(require_tollbooth_key)],
    ledger: Annotated[CreditLedger, Depends(_get_ledger)],
) -> dict:
    """Return the current credit balance for the authenticated API key."""
    balance = await ledger.balance(api_key)
    return {"api_key": api_key[:8] + "****", "credits_remaining": balance}


@router.post(
    "/api/v1/tollbooth/rotate-key",
    response_model=RotateKeyResponse,
    responses={
        403: {"model": TollboothError, "description": "Invalid API key"},
    },
)
async def rotate_key(
    api_key: Annotated[str, Depends(require_tollbooth_key)],
    ledger: Annotated[CreditLedger, Depends(_get_ledger)],
) -> RotateKeyResponse:
    """
    Rotate the authenticated API key.

    Generates a cryptographically secure replacement key, atomically
    transfers the full credit balance from the old key to the new one,
    and invalidates the old key.  The new key is returned in plaintext
    exactly once - store it immediately.
    """
    new_key = "sk-toll-" + secrets.token_urlsafe(32)
    credits_transferred = await ledger.rotate_key(api_key, new_key)

    logger.info(
        "tollbooth_key_rotated_via_api",
        old_key=api_key[:8],
        new_key=new_key[:8],
        credits_transferred=credits_transferred,
    )

    return RotateKeyResponse(
        new_api_key=new_key,
        credits_transferred=credits_transferred,
    )


@router.post(
    "/api/v1/voxis/generate",
    response_model=VoxisGenerateResponse,
    responses={
        402: {"model": TollboothError, "description": "Insufficient credits"},
        429: {"model": TollboothError, "description": "Rate limited"},
        503: {"model": TollboothError, "description": "Backend unavailable - credits refunded"},
    },
)
async def voxis_generate(
    body: VoxisGenerateRequest,
    api_key: Annotated[str, Depends(require_tollbooth_key)],
    ledger: Annotated[CreditLedger, Depends(_get_ledger)],
) -> VoxisGenerateResponse:
    """
    Generate personality-aware content via Voxis (metered).

    Credits are debited atomically before the backend call.  If the backend
    fails (Anthropic API down, timeout, internal error), the full cost is
    refunded automatically so the user does not lose credits.
    """
    cost = PRODUCT_COST[TollboothProduct.VOXIS_GENERATE]
    new_balance = await _charge(ledger, api_key, TollboothProduct.VOXIS_GENERATE)

    try:
        content, tokens_used = await _mock_voxis_generate(body.prompt, body.max_tokens)
    except Exception as exc:
        # Backend failed after successful debit - refund atomically.
        await ledger.refund(api_key, cost, reason=f"voxis_backend_error: {type(exc).__name__}")
        logger.error(
            "voxis_backend_error_credits_refunded",
            api_key=api_key[:8],
            credits_refunded=cost,
            error=str(exc),
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Voxis backend unavailable. {cost} credits have been refunded to your account."
            ),
        ) from exc

    return VoxisGenerateResponse(
        content=content,
        conversation_id=body.conversation_id,
        tokens_used=tokens_used,
        credits_charged=cost,
        credits_remaining=new_balance,
    )


@router.post(
    "/api/v1/knowledge/query",
    response_model=KnowledgeQueryResponse,
    responses={
        402: {"model": TollboothError, "description": "Insufficient credits"},
        429: {"model": TollboothError, "description": "Rate limited"},
        503: {"model": TollboothError, "description": "Backend unavailable - credits refunded"},
    },
)
async def knowledge_query(
    body: KnowledgeQueryRequest,
    api_key: Annotated[str, Depends(require_tollbooth_key)],
    ledger: Annotated[CreditLedger, Depends(_get_ledger)],
) -> KnowledgeQueryResponse:
    """
    Query the ArXiv-backed knowledge base (metered).

    Credits are debited atomically before the backend call.  If the ArXiv
    pipeline or retrieval index fails, the full cost is refunded automatically.
    """
    cost = PRODUCT_COST[TollboothProduct.KNOWLEDGE_QUERY]
    new_balance = await _charge(ledger, api_key, TollboothProduct.KNOWLEDGE_QUERY)

    try:
        results = await _mock_knowledge_query(body.query, body.top_k, body.categories)
    except Exception as exc:
        # Backend failed after successful debit - refund atomically.
        await ledger.refund(api_key, cost, reason=f"knowledge_backend_error: {type(exc).__name__}")
        logger.error(
            "knowledge_backend_error_credits_refunded",
            api_key=api_key[:8],
            credits_refunded=cost,
            error=str(exc),
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Knowledge backend unavailable. {cost} credits have been refunded to your account."
            ),
        ) from exc

    return KnowledgeQueryResponse(
        results=results,
        credits_charged=cost,
        credits_remaining=new_balance,
    )


# ─── Webhook receiver ────────────────────────────────────────────

# Webhook signing secret - set TOLLBOOTH_WEBHOOK_SECRET in env.
# When empty, signature verification is skipped (dev mode).
_WEBHOOK_SECRET = os.environ.get("TOLLBOOTH_WEBHOOK_SECRET", "")

# Credits per cent - configurable conversion rate
_CREDITS_PER_CENT = 1


def _verify_webhook_signature(payload_bytes: bytes, signature: str) -> bool:
    """
    HMAC-SHA256 signature verification for inbound webhooks.

    The sender must compute:
        HMAC-SHA256(secret, raw_body)  →  hex digest
    and pass it in the X-Tollbooth-Signature header.
    """
    if not _WEBHOOK_SECRET:
        # Dev mode: no secret configured, skip verification
        return True
    expected = hmac.new(
        _WEBHOOK_SECRET.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post(
    "/api/v1/webhooks/payment",
    response_model=WebhookAckResponse,
    responses={
        400: {"model": TollboothError, "description": "Invalid signature"},
    },
)
async def payment_webhook(
    request: Request,
    ledger: Annotated[CreditLedger, Depends(_get_ledger)],
    x_tollbooth_signature: Annotated[str, Header()] = "",
) -> WebhookAckResponse:
    """
    Receive payment notifications from Stripe or crypto RPC.

    Validates the HMAC signature, parses the payload, converts the
    payment amount to credits, and atomically increments the customer's
    balance in Redis.
    """
    raw_body = await request.body()

    if not _verify_webhook_signature(raw_body, x_tollbooth_signature):
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")

    payload = PaymentWebhookPayload.model_validate_json(raw_body)

    # Validate the target key exists before touching idempotency state.
    # An invalid api_key must fail consistently on every retry, not just the first.
    if not await ledger.key_exists(payload.api_key):
        detail = "Unknown API key - cannot credit unregistered key."
        raise HTTPException(status_code=400, detail=detail)

    # Idempotency: check if event already processed
    redis = request.app.state.redis.client
    prefix = request.app.state.redis._config.prefix
    idem_key = f"{prefix}:tollbooth:webhook_events:{payload.event_id}"
    already = await redis.set(idem_key, "1", ex=86400, nx=True)
    if not already:
        # Already processed - return the current balance without double-crediting
        current = await ledger.balance(payload.api_key)
        return WebhookAckResponse(
            event_id=payload.event_id,
            credits_added=0,
            new_balance=current,
        )

    credits_to_add = payload.amount_cents * _CREDITS_PER_CENT
    new_balance = await ledger.credit(payload.api_key, credits_to_add)

    logger.info(
        "tollbooth_payment_received",
        event_id=payload.event_id,
        api_key=payload.api_key[:8],
        amount_cents=payload.amount_cents,
        credits_added=credits_to_add,
        new_balance=new_balance,
    )

    return WebhookAckResponse(
        event_id=payload.event_id,
        credits_added=credits_to_add,
        new_balance=new_balance,
    )
