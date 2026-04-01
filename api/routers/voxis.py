"""
Voxis API Router - Expression / Communication / Personality
============================================================
Exposes observability, control, and configuration endpoints for the
Voxis expression engine. The backend VoxisService holds all state;
these routes surface it for the UI dashboard.

Existing endpoints (in main.py, not moved):
  GET  /api/v1/voxis/personality
  GET  /api/v1/voxis/health

New endpoints here:
  GET  /api/v1/voxis/metrics          - counters, rates, breakdowns
  GET  /api/v1/voxis/queue            - expression queue state + items
  GET  /api/v1/voxis/diversity        - diversity tracker state
  GET  /api/v1/voxis/reception        - reception engine metrics
  GET  /api/v1/voxis/dynamics         - conversation dynamics metrics
  GET  /api/v1/voxis/voice            - current voice parameter snapshot
  GET  /api/v1/voxis/conversations    - active conversation list/summary
  GET  /api/v1/voxis/config           - current Voxis config values
  POST /api/v1/voxis/personality      - adjust personality dimension(s)
  POST /api/v1/voxis/config           - update runtime config thresholds
  POST /api/v1/voxis/queue/drain      - manually trigger queue drain
  DELETE /api/v1/voxis/conversations/{id} - close a conversation
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from primitives.common import EOSBaseModel

router = APIRouter(prefix="/api/v1/voxis", tags=["voxis"])


# ─── Response Models ─────────────────────────────────────────────────────────

class VoxisMetricsResponse(EOSBaseModel):
    initialized: bool = False
    total_expressions: int = 0
    total_silence: int = 0
    total_speak: int = 0
    total_queued: int = 0
    total_queue_delivered: int = 0
    honesty_rejections: int = 0
    diversity_rejections: int = 0
    background_task_failures: int = 0
    silence_rate: float = 0.0
    expressions_by_trigger: dict[str, int] = Field(default_factory=dict)
    expressions_by_channel: dict[str, int] = Field(default_factory=dict)
    instance_name: str = ""


class QueuedExpressionItem(EOSBaseModel):
    intent_id: str
    trigger: str
    initial_relevance: float
    current_relevance: float
    queued_at_seconds: float
    halflife_seconds: float


class VoxisQueueResponse(EOSBaseModel):
    initialized: bool = False
    queue_size: int = 0
    max_size: int = 0
    total_enqueued: int = 0
    total_delivered: int = 0
    total_expired: int = 0
    total_evicted: int = 0
    highest_relevance: float = 0.0
    delivery_threshold: float = 0.3
    items: list[QueuedExpressionItem] = Field(default_factory=list)


class VoxisDiversityResponse(EOSBaseModel):
    initialized: bool = False
    window_size: int = 0
    threshold: float = 0.4
    recent_expressions_tracked: int = 0
    total_diversity_rejections: int = 0
    last_composite_score: float | None = None
    last_ngram_score: float | None = None
    last_semantic_score: float | None = None
    last_opener_score: float | None = None


class VoxisReceptionResponse(EOSBaseModel):
    initialized: bool = False
    total_correlated: int = 0
    total_expired: int = 0
    pending_count: int = 0
    avg_understood: float | None = None
    avg_emotional_impact: float | None = None
    avg_engagement: float | None = None
    avg_satisfaction: float | None = None


class VoxisDynamicsResponse(EOSBaseModel):
    initialized: bool = False
    total_turns: int = 0
    avg_response_time_s: float | None = None
    avg_user_word_count: float | None = None
    repair_mode: bool = False
    repair_signal_count: int = 0
    coherence_breaks: int = 0
    emotional_trajectory_valence: float | None = None
    emotional_trajectory_volatility: float | None = None


class VoxisVoiceResponse(EOSBaseModel):
    initialized: bool = False
    base_voice: str = ""
    speed: float = 1.0
    pitch_shift: float = 0.0
    emphasis: float = 1.0
    pause_frequency: float = 0.5
    last_personality_warmth: float | None = None
    last_personality_directness: float | None = None


class ConversationSummaryItem(EOSBaseModel):
    conversation_id: str
    participant_count: int
    message_count: int
    last_speaker: str | None = None
    last_message_preview: str | None = None
    dominant_topics: list[str] = Field(default_factory=list)
    emotional_arc_latest: float | None = None
    created_at: str | None = None


class VoxisConversationsResponse(EOSBaseModel):
    initialized: bool = False
    active_conversations: list[ConversationSummaryItem] = Field(default_factory=list)
    total_active: int = 0
    max_active: int = 0


class VoxisConfigResponse(EOSBaseModel):
    max_expression_length: int = 2000
    min_expression_interval_minutes: float = 1.0
    voice_synthesis_enabled: bool = False
    insight_expression_threshold: float = 0.6
    conversation_history_window: int = 50
    context_window_max_tokens: int = 4000
    conversation_summary_threshold: int = 10
    feedback_enabled: bool = True
    honesty_check_enabled: bool = True
    temperature_base: float = 0.7
    max_active_conversations: int = 50


class PersonalityUpdateRequest(EOSBaseModel):
    delta: dict[str, float] = Field(
        ...,
        description=(
            "Partial personality update. Keys are dimension names (warmth, directness, "
            "verbosity, formality, curiosity_expression, humour, empathy_expression, "
            "confidence_display, metaphor_use). Values are absolute deltas (capped at "
            "0.03 per dimension by VoxisService)."
        ),
    )


class PersonalityUpdateResponse(EOSBaseModel):
    previous: dict[str, float]
    updated: dict[str, float]
    applied_delta: dict[str, float]


class ConfigUpdateRequest(EOSBaseModel):
    max_expression_length: int | None = None
    min_expression_interval_minutes: float | None = None
    insight_expression_threshold: float | None = None
    honesty_check_enabled: bool | None = None
    temperature_base: float | None = None


class QueueDrainResponse(EOSBaseModel):
    drained_count: int
    delivered: list[str]


# ─── Routes ──────────────────────────────────────────────────────────────────


def _get_voxis(request: Request) -> Any:
    voxis = getattr(request.app.state, "voxis", None)
    if voxis is None:
        raise HTTPException(status_code=503, detail="Voxis service not initialized")
    return voxis


@router.get("/metrics", response_model=VoxisMetricsResponse)
async def get_voxis_metrics(request: Request) -> VoxisMetricsResponse:
    """Aggregate Voxis expression counters and rates."""
    voxis = getattr(request.app.state, "voxis", None)
    if voxis is None:
        return VoxisMetricsResponse(initialized=False)

    total_speak = getattr(voxis, "_total_speak", 0)
    total_silence = getattr(voxis, "_total_silence", 0)
    total = total_speak + total_silence
    silence_rate = (total_silence / total) if total > 0 else 0.0

    return VoxisMetricsResponse(
        initialized=True,
        total_expressions=getattr(voxis, "_total_expressions", 0),
        total_silence=total_silence,
        total_speak=total_speak,
        total_queued=getattr(voxis, "_total_queued", 0),
        total_queue_delivered=getattr(voxis, "_total_queue_delivered", 0),
        honesty_rejections=getattr(voxis, "_honesty_rejections", 0),
        diversity_rejections=getattr(voxis, "_diversity_rejections", 0),
        background_task_failures=getattr(voxis, "_background_task_failures", 0),
        silence_rate=round(silence_rate, 4),
        expressions_by_trigger=dict(getattr(voxis, "_expressions_by_trigger", {})),
        expressions_by_channel=dict(getattr(voxis, "_expressions_by_channel", {})),
        instance_name=getattr(voxis, "_instance_name", ""),
    )


@router.get("/queue", response_model=VoxisQueueResponse)
async def get_voxis_queue(request: Request) -> VoxisQueueResponse:
    """Expression queue state: pending items with relevance decay."""
    voxis = getattr(request.app.state, "voxis", None)
    if voxis is None:
        return VoxisQueueResponse(initialized=False)

    eq = getattr(voxis, "_expression_queue", None)
    if eq is None:
        return VoxisQueueResponse(initialized=True)

    m = eq.metrics() if hasattr(eq, "metrics") else {}

    # Try to get actual queued items for display
    items: list[QueuedExpressionItem] = []
    raw_items = getattr(eq, "_queue", [])
    for item in raw_items:
        try:
            import time

            queued_at = getattr(item, "queued_at", 0.0)
            elapsed = time.monotonic() - queued_at
            halflife = getattr(item, "halflife_seconds", 300.0)
            initial = getattr(item, "initial_relevance", 0.5)
            current = initial * (0.5 ** (elapsed / halflife)) if halflife > 0 else 0.0
            intent = getattr(item, "intent", None)
            trigger = (
                getattr(getattr(intent, "trigger", None), "value", "unknown")
                if intent
                else "unknown"
            )
            intent_id = getattr(intent, "id", str(id(item)))
            items.append(
                QueuedExpressionItem(
                    intent_id=intent_id,
                    trigger=trigger,
                    initial_relevance=round(initial, 4),
                    current_relevance=round(current, 4),
                    queued_at_seconds=round(elapsed, 1),
                    halflife_seconds=halflife,
                )
            )
        except Exception:
            pass

    getattr(voxis, "_config", None)
    delivery_threshold = getattr(eq, "_delivery_threshold", 0.3)

    return VoxisQueueResponse(
        initialized=True,
        queue_size=m.get("queue_size", len(raw_items)),
        max_size=getattr(eq, "_max_size", 20),
        total_enqueued=m.get("total_enqueued", 0),
        total_delivered=m.get("total_delivered", 0),
        total_expired=m.get("total_expired", 0),
        total_evicted=m.get("total_evicted", 0),
        highest_relevance=round(m.get("highest_relevance", 0.0), 4),
        delivery_threshold=round(delivery_threshold, 4),
        items=items,
    )


@router.get("/diversity", response_model=VoxisDiversityResponse)
async def get_voxis_diversity(request: Request) -> VoxisDiversityResponse:
    """Diversity tracker state - repetition detection metrics."""
    voxis = getattr(request.app.state, "voxis", None)
    if voxis is None:
        return VoxisDiversityResponse(initialized=False)

    dt = getattr(voxis, "_diversity_tracker", None)
    if dt is None:
        return VoxisDiversityResponse(initialized=True)

    dt.metrics() if hasattr(dt, "metrics") else {}
    recent = getattr(dt, "_recent_expressions", [])

    # Try to get last computed scores
    last_scores = getattr(dt, "_last_scores", None)

    return VoxisDiversityResponse(
        initialized=True,
        window_size=getattr(dt, "_window_size", 20),
        threshold=getattr(dt, "_threshold", 0.4),
        recent_expressions_tracked=len(recent),
        total_diversity_rejections=getattr(voxis, "_diversity_rejections", 0),
        last_composite_score=(
            round(last_scores.get("composite", 0.0), 4) if last_scores else None
        ),
        last_ngram_score=(
            round(last_scores.get("ngram", 0.0), 4) if last_scores else None
        ),
        last_semantic_score=(
            round(last_scores.get("semantic", 0.0), 4) if last_scores else None
        ),
        last_opener_score=(
            round(last_scores.get("opener", 0.0), 4) if last_scores else None
        ),
    )


@router.get("/reception", response_model=VoxisReceptionResponse)
async def get_voxis_reception(request: Request) -> VoxisReceptionResponse:
    """Reception engine metrics - feedback loop quality."""
    voxis = getattr(request.app.state, "voxis", None)
    if voxis is None:
        return VoxisReceptionResponse(initialized=False)

    re = getattr(voxis, "_reception_engine", None)
    if re is None:
        return VoxisReceptionResponse(initialized=True)

    m = re.metrics() if hasattr(re, "metrics") else {}
    pending = getattr(re, "_pending_expressions", {})

    return VoxisReceptionResponse(
        initialized=True,
        total_correlated=m.get("total_correlated", 0),
        total_expired=m.get("total_expired", 0),
        pending_count=len(pending),
        avg_understood=m.get("avg_understood"),
        avg_emotional_impact=m.get("avg_emotional_impact"),
        avg_engagement=m.get("avg_engagement"),
        avg_satisfaction=m.get("avg_satisfaction"),
    )


@router.get("/dynamics", response_model=VoxisDynamicsResponse)
async def get_voxis_dynamics(request: Request) -> VoxisDynamicsResponse:
    """Conversation dynamics metrics - pacing, repair, emotional trajectory."""
    voxis = getattr(request.app.state, "voxis", None)
    if voxis is None:
        return VoxisDynamicsResponse(initialized=False)

    de = getattr(voxis, "_dynamics_engine", None)
    if de is None:
        return VoxisDynamicsResponse(initialized=True)

    m = de.metrics() if hasattr(de, "metrics") else {}

    return VoxisDynamicsResponse(
        initialized=True,
        total_turns=m.get("total_turns", 0),
        avg_response_time_s=m.get("avg_response_time_s"),
        avg_user_word_count=m.get("avg_user_word_count"),
        repair_mode=m.get("repair_mode", False),
        repair_signal_count=m.get("repair_signal_count", 0),
        coherence_breaks=m.get("coherence_breaks", 0),
        emotional_trajectory_valence=m.get("emotional_trajectory_valence"),
        emotional_trajectory_volatility=m.get("emotional_trajectory_volatility"),
    )


@router.get("/voice", response_model=VoxisVoiceResponse)
async def get_voxis_voice(request: Request) -> VoxisVoiceResponse:
    """Current TTS voice parameter snapshot derived from personality + affect."""
    voxis = getattr(request.app.state, "voxis", None)
    if voxis is None:
        return VoxisVoiceResponse(initialized=False)

    ve = getattr(voxis, "_voice_engine", None)
    if ve is None:
        return VoxisVoiceResponse(initialized=True)

    # Read last derived params if available
    last = getattr(ve, "_last_params", None)
    personality = getattr(voxis, "_personality_engine", None)
    pv = getattr(personality, "_vector", None) if personality else None

    return VoxisVoiceResponse(
        initialized=True,
        base_voice=getattr(last, "base_voice", getattr(ve, "_base_voice", "nova")),
        speed=round(getattr(last, "speed", 1.0), 3),
        pitch_shift=round(getattr(last, "pitch_shift", 0.0), 3),
        emphasis=round(getattr(last, "emphasis", 1.0), 3),
        pause_frequency=round(getattr(last, "pause_frequency", 0.5), 3),
        last_personality_warmth=round(pv.warmth, 4) if pv else None,
        last_personality_directness=round(pv.directness, 4) if pv else None,
    )


@router.get("/conversations", response_model=VoxisConversationsResponse)
async def get_voxis_conversations(request: Request) -> VoxisConversationsResponse:
    """Active conversation states managed by ConversationManager."""
    voxis = getattr(request.app.state, "voxis", None)
    if voxis is None:
        return VoxisConversationsResponse(initialized=False)

    cm = getattr(voxis, "_conversation_manager", None)
    if cm is None:
        return VoxisConversationsResponse(initialized=True)

    # ConversationManager stores active states in _active_states dict
    active_states: dict[str, Any] = getattr(cm, "_active_states", {})
    config = getattr(voxis, "_config", None)
    max_active = getattr(config, "max_active_conversations", 50)

    items: list[ConversationSummaryItem] = []
    for conv_id, state in active_states.items():
        messages = getattr(state, "messages", [])
        last_msg = messages[-1] if messages else None
        last_speaker = getattr(last_msg, "role", None) if last_msg else None
        preview = None
        if last_msg:
            raw_content = getattr(last_msg, "content", "")
            preview = raw_content[:80] + "…" if len(raw_content) > 80 else raw_content

        arc: list[float] = getattr(state, "emotional_arc", [])
        arc_latest = round(arc[-1], 4) if arc else None

        topics: list[str] = getattr(state, "topics", [])
        participants: list[str] = getattr(state, "participant_ids", [])
        created_at = getattr(state, "created_at", None)

        items.append(
            ConversationSummaryItem(
                conversation_id=conv_id,
                participant_count=len(participants),
                message_count=len(messages),
                last_speaker=last_speaker,
                last_message_preview=preview,
                dominant_topics=topics[:3],
                emotional_arc_latest=arc_latest,
                created_at=str(created_at) if created_at else None,
            )
        )

    return VoxisConversationsResponse(
        initialized=True,
        active_conversations=items,
        total_active=len(items),
        max_active=max_active,
    )


@router.get("/config", response_model=VoxisConfigResponse)
async def get_voxis_config(request: Request) -> VoxisConfigResponse:
    """Current Voxis runtime configuration values."""
    voxis = getattr(request.app.state, "voxis", None)
    if voxis is None:
        return VoxisConfigResponse()

    cfg = getattr(voxis, "_config", None)
    if cfg is None:
        return VoxisConfigResponse()

    return VoxisConfigResponse(
        max_expression_length=getattr(cfg, "max_expression_length", 2000),
        min_expression_interval_minutes=getattr(
            cfg, "min_expression_interval_minutes", 1.0
        ),
        voice_synthesis_enabled=getattr(cfg, "voice_synthesis_enabled", False),
        insight_expression_threshold=getattr(
            cfg, "insight_expression_threshold", 0.6
        ),
        conversation_history_window=getattr(cfg, "conversation_history_window", 50),
        context_window_max_tokens=getattr(cfg, "context_window_max_tokens", 4000),
        conversation_summary_threshold=getattr(
            cfg, "conversation_summary_threshold", 10
        ),
        feedback_enabled=getattr(cfg, "feedback_enabled", True),
        honesty_check_enabled=getattr(cfg, "honesty_check_enabled", True),
        temperature_base=getattr(cfg, "temperature_base", 0.7),
        max_active_conversations=getattr(cfg, "max_active_conversations", 50),
    )


@router.post("/personality", response_model=PersonalityUpdateResponse)
async def update_voxis_personality(
    request: Request, body: PersonalityUpdateRequest
) -> PersonalityUpdateResponse:
    """
    Apply a personality delta. Adjustments are capped at ±0.03 per dimension
    by VoxisService (MAX_PERSONALITY_DELTA). Values are clamped to [0, 1].
    """
    voxis = _get_voxis(request)

    personality_engine = getattr(voxis, "_personality_engine", None)
    if personality_engine is None:
        raise HTTPException(status_code=503, detail="Personality engine not available")

    current_vector = getattr(personality_engine, "_vector", None)
    if current_vector is None:
        raise HTTPException(
            status_code=503, detail="Personality vector not initialized"
        )

    _personality_dims = [
        "warmth",
        "directness",
        "verbosity",
        "formality",
        "curiosity_expression",
        "humour",
        "empathy_expression",
        "confidence_display",
        "metaphor_use",
    ]

    # Snapshot before
    previous = {d: round(getattr(current_vector, d, 0.0), 4) for d in _personality_dims}

    # Apply via VoxisService.update_personality (respects MAX_PERSONALITY_DELTA)
    updated_vector = voxis.update_personality(body.delta)

    # Snapshot after
    updated = {d: round(getattr(updated_vector, d, 0.0), 4) for d in _personality_dims}
    applied_delta = {d: round(updated[d] - previous[d], 4) for d in _personality_dims if d in body.delta}

    return PersonalityUpdateResponse(
        previous=previous,
        updated=updated,
        applied_delta=applied_delta,
    )


@router.post("/config", response_model=VoxisConfigResponse)
async def update_voxis_config(
    request: Request, body: ConfigUpdateRequest
) -> VoxisConfigResponse:
    """
    Update runtime config thresholds without restarting.
    Only provided fields are changed.
    """
    voxis = _get_voxis(request)
    cfg = getattr(voxis, "_config", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Voxis config not available")

    if body.max_expression_length is not None:
        cfg.max_expression_length = max(100, body.max_expression_length)

    if body.min_expression_interval_minutes is not None:
        cfg.min_expression_interval_minutes = max(
            0.0, body.min_expression_interval_minutes
        )

    if body.insight_expression_threshold is not None:
        cfg.insight_expression_threshold = max(
            0.0, min(1.0, body.insight_expression_threshold)
        )

    if body.honesty_check_enabled is not None:
        cfg.honesty_check_enabled = body.honesty_check_enabled
        # Propagate to renderer if accessible
        renderer = getattr(voxis, "_renderer", None)
        if renderer is not None:
            with __import__("contextlib").suppress(Exception):
                renderer._honesty_check_enabled = body.honesty_check_enabled

    if body.temperature_base is not None:
        cfg.temperature_base = max(0.0, min(2.0, body.temperature_base))

    return VoxisConfigResponse(
        max_expression_length=cfg.max_expression_length,
        min_expression_interval_minutes=cfg.min_expression_interval_minutes,
        voice_synthesis_enabled=getattr(cfg, "voice_synthesis_enabled", False),
        insight_expression_threshold=cfg.insight_expression_threshold,
        conversation_history_window=getattr(cfg, "conversation_history_window", 50),
        context_window_max_tokens=getattr(cfg, "context_window_max_tokens", 4000),
        conversation_summary_threshold=getattr(
            cfg, "conversation_summary_threshold", 10
        ),
        feedback_enabled=getattr(cfg, "feedback_enabled", True),
        honesty_check_enabled=cfg.honesty_check_enabled,
        temperature_base=cfg.temperature_base,
        max_active_conversations=getattr(cfg, "max_active_conversations", 50),
    )


@router.post("/queue/drain", response_model=QueueDrainResponse)
async def drain_expression_queue(request: Request) -> QueueDrainResponse:
    """
    Manually trigger a queue drain cycle. Delivers up to 3 queued expressions
    whose relevance is above the delivery threshold.
    """
    voxis = _get_voxis(request)

    eq = getattr(voxis, "_expression_queue", None)
    if eq is None:
        return QueueDrainResponse(drained_count=0, delivered=[])

    drained = eq.drain(max_items=3) if hasattr(eq, "drain") else []
    delivered_ids: list[str] = []

    for queued_expr in drained:
        try:
            intent = getattr(queued_expr, "intent", None)
            affect = getattr(queued_expr, "affect", None)
            if intent is None:
                continue

            expression = await voxis.express(
                content=getattr(intent, "content_to_express", ""),
                trigger=getattr(intent, "trigger", None),
                conversation_id=getattr(intent, "conversation_id", None),
                addressee_id=getattr(intent, "addressee_id", None),
                affect=affect,
                urgency=getattr(intent, "urgency", 0.5),
                insight_value=getattr(intent, "insight_value", 0.5),
            )
            delivered_ids.append(expression.id)
            getattr(voxis, "_total_queue_delivered", 0)  # counter updated inside
        except Exception:
            pass

    return QueueDrainResponse(
        drained_count=len(delivered_ids),
        delivered=delivered_ids,
    )


@router.delete("/conversations/{conversation_id}")
async def close_conversation(
    request: Request, conversation_id: str
) -> JSONResponse:
    """Close and remove a conversation state from ConversationManager."""
    voxis = _get_voxis(request)
    cm = getattr(voxis, "_conversation_manager", None)
    if cm is None:
        raise HTTPException(
            status_code=503, detail="ConversationManager not available"
        )

    try:
        await cm.close_conversation(conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        content={"status": "closed", "conversation_id": conversation_id}
    )
