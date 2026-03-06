"""
EcodiaOS — Tollbooth Types

Request/response schemas for the external monetization API.
"""

from __future__ import annotations

import enum
from datetime import datetime  # noqa: TC003 — used at runtime by Pydantic

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ─── Enums ────────────────────────────────────────────────────────


class TollboothProduct(enum.StrEnum):
    """Billable product lines exposed by the Tollbooth."""

    VOXIS_GENERATE = "voxis_generate"
    KNOWLEDGE_QUERY = "knowledge_query"


# ─── Cost table (credits per request) ────────────────────────────

PRODUCT_COST: dict[TollboothProduct, int] = {
    TollboothProduct.VOXIS_GENERATE: 10,
    TollboothProduct.KNOWLEDGE_QUERY: 5,
}


# ─── Request schemas ─────────────────────────────────────────────


class VoxisGenerateRequest(EOSBaseModel):
    """External request to generate personality-aware content."""

    prompt: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation thread to continue.",
    )
    max_tokens: int = Field(default=512, ge=1, le=4096)


class KnowledgeQueryRequest(EOSBaseModel):
    """External request to query the ArXiv-backed knowledge base."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    categories: list[str] = Field(
        default_factory=list,
        description="Optional arXiv category filter (e.g. 'cs.AI').",
    )


# ─── Response schemas ────────────────────────────────────────────


class VoxisGenerateResponse(EOSBaseModel):
    request_id: str = Field(default_factory=new_id)
    content: str
    conversation_id: str | None = None
    tokens_used: int = 0
    credits_charged: int = 0
    credits_remaining: int = 0


class KnowledgeQueryResponse(EOSBaseModel):
    request_id: str = Field(default_factory=new_id)
    results: list[KnowledgeResult]
    credits_charged: int = 0
    credits_remaining: int = 0


class KnowledgeResult(EOSBaseModel):
    title: str
    summary: str
    arxiv_id: str = ""
    relevance_score: float = 0.0


# ─── Webhook schemas ─────────────────────────────────────────────


class PaymentWebhookPayload(EOSBaseModel):
    """Inbound payment notification (Stripe / crypto RPC stub)."""

    event_id: str = Field(default_factory=new_id)
    event_type: str = Field(
        ...,
        description="e.g. 'payment_intent.succeeded', 'crypto.confirmed'",
    )
    api_key: str = Field(..., description="Customer API key to credit.")
    amount_cents: int = Field(..., ge=1, description="Payment amount in cents.")
    currency: str = Field(default="usd")
    timestamp: datetime = Field(default_factory=utc_now)
    metadata: dict[str, str] = Field(default_factory=dict)


class WebhookAckResponse(EOSBaseModel):
    event_id: str
    credits_added: int
    new_balance: int


# ─── Key rotation schemas ─────────────────────────────────────────


class RotateKeyResponse(EOSBaseModel):
    new_api_key: str
    credits_transferred: int


# ─── Error schemas ───────────────────────────────────────────────


class TollboothError(EOSBaseModel):
    error: str
    detail: str | None = None
    request_id: str = Field(default_factory=new_id)


# Forward-ref rebuild (KnowledgeQueryResponse references KnowledgeResult)
KnowledgeQueryResponse.model_rebuild()
