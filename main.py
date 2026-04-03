"""
EcodiaOS - Application Entry Point

FastAPI application with the startup sequence defined in the
Infrastructure Architecture specification.

`docker compose up` → uvicorn ecodiaos.main:app

The startup/shutdown lifecycle is orchestrated by ``core.registry.SystemRegistry``.
This module defines:
  1. A thin ``lifespan()`` that delegates to the registry
  2. The FastAPI app, CORS, middleware
  3. Router includes
  4. All API endpoints (inline - to be migrated to api/routers/ over time)
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env file before any configuration is loaded
load_dotenv()

from clients.redis import RedisClient
from core.registry import SystemRegistry

logger = structlog.get_logger()
_chat_logger = structlog.get_logger("chat")

# ─── Registry (single instance for the process lifetime) ─────────
_registry = SystemRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown sequence.
    Delegates entirely to SystemRegistry - see core/registry.py.
    """
    await _registry.startup(app)

    yield

    await _registry.shutdown(app)


# ─── Helpers (kept for backward compat with API endpoints) ────────
# The startup helper classes have been moved to core/helpers.py.
# API endpoints below reference app.state.* which is populated by
# the SystemRegistry during startup.


# Metrics publisher function (used by API endpoint and registry)
# Moved to telemetry/publisher.py but re-exported here for the
# command_center_metrics_stream endpoint.
from telemetry.publisher import METRICS_CHANNEL as _METRICS_CHANNEL  # noqa: E402

# ─── FastAPI Application ─────────────────────────────────────────

app = FastAPI(
    title="EcodiaOS",
    description="A living digital organism - API surface",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS for frontend
_cors_origins = [
    "http://localhost:3000",
    "https://ecodiaos-frontend-929871567697.australia-southeast1.run.app",
]
# Allow additional origins via env var (comma-separated)
_extra_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")
if _extra_origins:
    _cors_origins.extend(o.strip() for o in _extra_origins.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Oikos & Identity Router ─────────────────────────────────────
from api.routers.oikos import router as oikos_router

app.include_router(oikos_router)

# ─── Federation Router ───────────────────────────────────────────
from api.routers.federation import router as federation_router

app.include_router(federation_router)

# ─── Identity Envelope Router ────────────────────────────────────
from api.routers.identity import router as identity_router

app.include_router(identity_router)

# ─── Tollbooth (External Monetization API) ──────────────────────
from api.monetization.router import router as tollbooth_router

app.include_router(tollbooth_router)

# ─── Legal Entity Provisioning Router ───────────────────────────
from api.routers.legal import router as legal_router

app.include_router(legal_router)

# ─── EIS (Epistemic Immune System) Router ────────────────────────
from api.routers.eis import router as eis_router

app.include_router(eis_router)

# ─── SACM (Substrate-Arbitrage Compute Mesh) Router ─────────────
from api.routers.sacm import router as sacm_router

app.include_router(sacm_router)

# ─── Axon (Action Executor / Reflex Arc) Router ──────────────────
from api.routers.axon import router as axon_router

app.include_router(axon_router)

# ─── Arbitrage Reflex Arc (Fast-Path) Router ─────────────────────
from api.routers.fast_path import router as fast_path_router

app.include_router(fast_path_router)

# ─── Benchmarks Router ───────────────────────────────────────────
from api.routers.benchmarks import router as benchmarks_router

app.include_router(benchmarks_router)

# ─── Memory System Router ────────────────────────────────────────
from api.routers.memory import router as memory_router

app.include_router(memory_router)

# ─── Nova Router ─────────────────────────────────────────────────
from api.routers.nova import router as nova_router

app.include_router(nova_router)

# ─── Thymos (Immune System) Router ───────────────────────────────
from api.routers.thymos import router as thymos_router

app.include_router(thymos_router)

# ─── Telos (Drives as Intelligence Topology) Router ──────────────
from api.routers.telos import router as telos_router

app.include_router(telos_router)

# ─── Mitosis (Self-Replication) Router ───────────────────────────
from api.routers.mitosis import router as mitosis_router

app.include_router(mitosis_router)

# ─── Phantom Liquidity (Phase 16q) Router ────────────────────────
from api.routers.phantom_liquidity import router as phantom_liquidity_router

app.include_router(phantom_liquidity_router)

# ─── Simula (Self-Evolution + Inspector) Router ───────────────────
from api.routers.simula import router as simula_router

app.include_router(simula_router)

# ─── Skia (Shadow Infrastructure / DR) Router ────────────────────
from api.routers.skia import router as skia_router

app.include_router(skia_router)

from api.routers.voxis import router as voxis_router

app.include_router(voxis_router)

# ─── Kairos (Causal Invariant Mining) Router ─────────────────────
from api.routers.kairos import router as kairos_router

app.include_router(kairos_router)

# ─── Logs & Diagnostics Router ──────────────────────────────────
from api.routers.logs import router as logs_router

app.include_router(logs_router)

# ─── Equor (Constitutional Ethics Engine) Router ─────────────────
from api.routers.equor import router as equor_router

app.include_router(equor_router)

# ─── Observatory Router ─────────────────────────────────────────
from api.routers.observatory import router as observatory_router

app.include_router(observatory_router)

# ─── Symbridge (EcodiaOS Factory Bridge) ────────────────────────
from api.routers.symbridge import router as symbridge_router

app.include_router(symbridge_router)

# ─── Corpus Knowledge Ingestion Router ──────────────────────────
from api.routers.corpus_ingestion import router as corpus_ingestion_router

app.include_router(corpus_ingestion_router)

# ─── Alive WebSocket on port 8000 (for Cloud Run) ────────────────
# Cloud Run only exposes one port per container. The standalone ws_server
# on port 8001 is unreachable, so we add a FastAPI WebSocket route here
# that taps into the same Atune + Redis data streams.

import asyncio as _ws_asyncio

import orjson as _ws_orjson
from fastapi import WebSocket, WebSocketDisconnect


def _ws_json(data: dict[str, Any]) -> str:
    return _ws_orjson.dumps(data).decode()


@app.websocket("/ws/alive")
async def alive_websocket(ws: WebSocket):
    """
    Alive visualization WebSocket - Cloud Run single-port alternative (port 8000).

    This endpoint is a **distinct protocol** from the standalone AliveWebSocketServer
    on port 8001. It is NOT governed by Spec 11a and serves different consumers
    (perception / decisions dashboard pages running in Cloud Run environments where
    only one port is exposed).

    Protocol schema:
      {"stream": "affect",    "payload": {...}}  - 6D AffectState from Atune ~10Hz
      {"stream": "synapse",   "payload": {...}}  - raw Synapse events (Redis pub/sub)
      {"stream": "workspace", "payload": {...}}  - Atune workspace snapshot ~1Hz
      {"stream": "outcomes",  "payload": {...}}  - Axon execution outcomes ~0.5Hz

    Affect payload: valence, arousal, dominance, curiosity, care_activation,
                    coherence_stress, ts

    Do NOT merge this with the standalone server. See the protocol note in
    ``systems/alive/ws_server.py`` for the rationale.
    """
    await ws.accept()

    atune: AtuneService = app.state.atune
    redis_client: RedisClient = app.state.redis

    # Send initial affect snapshot so the client renders immediately
    affect = atune.current_affect
    await ws.send_text(_ws_json({
        "stream": "affect",
        "payload": {
            "valence": round(affect.valence, 4),
            "arousal": round(affect.arousal, 4),
            "dominance": round(affect.dominance, 4),
            "curiosity": round(affect.curiosity, 4),
            "care_activation": round(affect.care_activation, 4),
            "coherence_stress": round(affect.coherence_stress, 4),
            "ts": affect.timestamp.isoformat() if affect.timestamp else None,
        },
    }))

    # Channel for background tasks to push messages into
    queue: _ws_asyncio.Queue[str] = _ws_asyncio.Queue(maxsize=256)
    running = True

    async def _affect_poller() -> None:
        """Poll Atune affect at ~10 Hz."""
        while running:
            a = atune.current_affect
            msg = _ws_json({
                "stream": "affect",
                "payload": {
                    "valence": round(a.valence, 4),
                    "arousal": round(a.arousal, 4),
                    "dominance": round(a.dominance, 4),
                    "curiosity": round(a.curiosity, 4),
                    "care_activation": round(a.care_activation, 4),
                    "coherence_stress": round(a.coherence_stress, 4),
                    "ts": a.timestamp.isoformat() if a.timestamp else None,
                },
            })
            try:
                queue.put_nowait(msg)
            except _ws_asyncio.QueueFull:
                pass  # Drop if client is lagging
            await _ws_asyncio.sleep(0.1)

    async def _redis_subscriber() -> None:
        """Subscribe to synapse events from Redis and enqueue."""
        redis = redis_client.client
        prefix = redis_client._config.prefix
        channel = f"{prefix}:channel:synapse_events"
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            while running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.1,
                )
                if message and message["type"] == "message":
                    raw = message["data"]
                    payload = _ws_orjson.loads(raw) if isinstance(raw, (str, bytes)) else raw
                    msg = _ws_json({"stream": "synapse", "payload": payload})
                    with suppress(_ws_asyncio.QueueFull):
                        queue.put_nowait(msg)
        except _ws_asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()  # type: ignore[no-untyped-call]

    async def _sender() -> None:
        """Drain the queue and send to the WebSocket client."""
        while running:
            msg = await queue.get()
            await ws.send_text(msg)

    async def _workspace_poller() -> None:
        """Poll Atune workspace snapshot at ~1 Hz for the perception page."""
        while running:
            try:
                ws_snapshot = atune.workspace_snapshot
                affect = atune.current_affect
                workspace_items = []
                for b in ws_snapshot.recent_broadcasts:
                    workspace_items.append({
                        "broadcast_id": b.broadcast_id,
                        "salience": round(b.salience, 4),
                        "ts": b.timestamp.isoformat() if hasattr(b, "timestamp") and b.timestamp else None,
                    })
                msg = _ws_json({
                    "stream": "workspace",
                    "payload": {
                        "cycle_count": atune.cycle_count,
                        "dynamic_threshold": round(atune.workspace_threshold, 4),
                        "meta_attention_mode": atune.meta_attention_mode,
                        "recent_broadcasts": workspace_items,
                        "affect": {
                            "valence": round(affect.valence, 4),
                            "arousal": round(affect.arousal, 4),
                            "curiosity": round(affect.curiosity, 4),
                            "coherence_stress": round(affect.coherence_stress, 4),
                        },
                    },
                })
                with suppress(_ws_asyncio.QueueFull):
                    queue.put_nowait(msg)
            except Exception:
                pass
            await _ws_asyncio.sleep(1.0)  # 1 Hz - workspace state changes slowly

    async def _outcomes_poller() -> None:
        """Poll Axon recent outcomes at ~0.5 Hz for the decisions page."""
        last_count = 0
        while running:
            try:
                axon_svc = app.state.axon
                total = getattr(axon_svc, "_total_executions", 0)
                if total != last_count:
                    last_count = total
                    outcomes = axon_svc.recent_outcomes[:10]
                    msg = _ws_json({
                        "stream": "outcomes",
                        "payload": {
                            "outcomes": [
                                {
                                    "execution_id": o.execution_id,
                                    "intent_id": o.intent_id,
                                    "success": o.success,
                                    "partial": o.partial,
                                    "status": o.status.value,
                                    "failure_reason": o.failure_reason or None,
                                    "duration_ms": o.duration_ms,
                                    "steps": [
                                        {
                                            "action_type": s.action_type,
                                            "description": s.description[:80],
                                            "success": s.result.success,
                                        }
                                        for s in o.step_outcomes
                                    ],
                                    "world_state_changes": o.world_state_changes[:3],
                                }
                                for o in outcomes
                            ],
                            "total": total,
                            "successful": getattr(axon_svc, "_successful_executions", 0),
                            "failed": getattr(axon_svc, "_failed_executions", 0),
                        },
                    })
                    with suppress(_ws_asyncio.QueueFull):
                        queue.put_nowait(msg)
            except Exception:
                pass
            await _ws_asyncio.sleep(2.0)  # 0.5 Hz - poll on change

    poller_task = _ws_asyncio.create_task(_affect_poller())
    subscriber_task = _ws_asyncio.create_task(_redis_subscriber())
    workspace_task = _ws_asyncio.create_task(_workspace_poller())
    outcomes_task = _ws_asyncio.create_task(_outcomes_poller())
    sender_task = _ws_asyncio.create_task(_sender())

    try:
        # Keep alive - we don't expect client messages
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        running = False
        poller_task.cancel()
        subscriber_task.cancel()
        workspace_task.cancel()
        outcomes_task.cancel()
        sender_task.cancel()
        # Await all tasks so their finally-blocks (e.g. pubsub.aclose()) run
        # before the next connection reuses the same Redis connection pool.
        await _ws_asyncio.gather(
            poller_task,
            subscriber_task,
            workspace_task,
            outcomes_task,
            sender_task,
            return_exceptions=True,
        )


# ─── API Key Authentication Middleware ─────────────────────────────
# Protects all /api/v1/* endpoints. /health is always public.
# When no API keys are configured (dev mode), all requests pass through.

from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from starlette.requests import Request

    from clients.file_watcher import FileWatcher
    from clients.scheduler import PerceptionScheduler
    from systems.simula.distributed_shield import FleetShieldManager as _Fleet


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates API key from X-EOS-API-Key header or Authorization Bearer token.

    Protected paths: /api/v1/*
    Public paths: /health, /docs, /openapi.json, /api/v1/federation/identity
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Public endpoints - no auth required
        if path in (
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/v1/admin/llm/metrics",
            "/api/v1/admin/llm/summary",
            "/api/v1/eis/health",
            "/api/v1/eis/stats",
            "/api/v1/sacm/metrics",
            "/api/v1/sacm/savings",
        ):
            return await call_next(request)
        # Federation endpoints accessible to peers (trust-gated at service level)
        if path in (
            "/api/v1/federation/identity",
            "/api/v1/federation/threat-advisory",
            "/api/v1/federation/handshake",
            "/api/v1/federation/handshake/confirm",
        ):
            return await call_next(request)

        # Only protect /api/v1/* paths
        if not path.startswith("/api/v1/"):
            return await call_next(request)

        # Check if auth is configured
        config = getattr(request.app.state, "config", None)
        if config is None or not config.server.api_keys:
            # Dev mode: no keys configured, allow all
            return await call_next(request)

        # Extract API key from header or Authorization bearer
        api_key = request.headers.get(config.server.api_key_header, "")
        if not api_key:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]

        if not api_key or api_key not in config.server.api_keys:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key"},
            )

        return await call_next(request)


app.add_middleware(APIKeyMiddleware)

# ─── Error Capture Middleware ─────────────────────────────────────
# Outermost middleware: intercepts all HTTP errors and unhandled exceptions,
# converts them to Thymos Incidents for automatic immune-system diagnosis.
# Registered after APIKeyMiddleware so it wraps the auth layer too.
from api.middleware.error_capture import ErrorCaptureMiddleware  # noqa: E402

app.add_middleware(ErrorCaptureMiddleware)


# ─── Health & Status Endpoints ────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, Any]:
    """System health check.

    Every sub-check is individually guarded so that a single system
    failure never crashes the entire endpoint — the caller always
    gets a JSON response indicating which subsystems are degraded.
    """
    _NOT_INIT: dict[str, str] = {"status": "not_initialized"}

    async def _safe_health(name: str, attr: str, method: str = "health") -> dict[str, Any]:
        """Call ``app.state.<attr>.<method>()`` with full error isolation and timeout."""
        if not hasattr(app.state, attr):
            return _NOT_INIT
        try:
            obj = getattr(app.state, attr)
            return await asyncio.wait_for(getattr(obj, method)(), timeout=5.0)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("health_check_timeout", system=name)
            return {"status": "timeout", "error": f"{name} health check timed out"}
        except Exception as exc:
            error_msg = str(exc) or f"{type(exc).__name__} (no message)"
            logger.warning("health_check_failed", system=name, error=error_msg)
            return {"status": "error", "error": error_msg}

    def _safe_sync_stats(attr: str) -> dict[str, Any]:
        """Read a sync ``.stats`` property with full error isolation."""
        if not hasattr(app.state, attr):
            return _NOT_INIT
        try:
            return getattr(app.state, attr).stats
        except Exception as exc:
            error_msg = str(exc) or f"{type(exc).__name__} (no message)"
            logger.warning("health_check_failed", system=attr, error=error_msg)
            return {"status": "error", "error": error_msg}

    # Gather all health probes concurrently — each is individually guarded
    (
        memory_health,
        equor_health,
        voxis_health,
        nova_health,
        synapse_health,
        thymos_health,
        oneiros_health,
        thread_health,
        neo4j_health,
        redis_health,
        federation_health,
    ) = await asyncio.gather(
        _safe_health("memory", "memory"),
        _safe_health("equor", "equor"),
        _safe_health("voxis", "voxis"),
        _safe_health("nova", "nova"),
        _safe_health("synapse", "synapse"),
        _safe_health("thymos", "thymos"),
        _safe_health("oneiros", "oneiros"),
        _safe_health("thread", "thread"),
        _safe_health("neo4j", "neo4j", "health_check"),
        _safe_health("redis", "redis", "health_check"),
        _safe_health("federation", "federation"),
    )

    # TimescaleDB is optional — may be None
    if hasattr(app.state, "tsdb") and app.state.tsdb is not None:
        try:
            tsdb_health = await asyncio.wait_for(app.state.tsdb.health_check(), timeout=5.0)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("health_check_timeout", system="timescaledb")
            tsdb_health = {"status": "timeout", "error": "timescaledb health check timed out"}
        except Exception as exc:
            error_msg = str(exc) or f"{type(exc).__name__} (no message)"
            logger.warning("health_check_failed", system="timescaledb", error=error_msg)
            tsdb_health = {"status": "error", "error": error_msg}
    else:
        tsdb_health = {"status": "not_configured"}

    # Determine overall status
    overall = "healthy"
    if any(
        h.get("status") not in ("connected", "healthy", "running")
        for h in [neo4j_health, tsdb_health, redis_health]
        if h.get("status") != "not_configured"
    ):
        overall = "degraded"
    if equor_health.get("safe_mode"):
        overall = "degraded"
    if synapse_health.get("safe_mode"):
        overall = "safe_mode"

    # Instance identity (guarded with timeout)
    instance_name = "unborn"
    try:
        if hasattr(app.state, "memory"):
            instance = await asyncio.wait_for(app.state.memory.get_self(), timeout=5.0)
            if instance:
                instance_name = instance.name
    except Exception:
        pass

    # Atune telemetry (guarded)
    atune_health: dict[str, Any] = _NOT_INIT
    if hasattr(app.state, "atune"):
        try:
            atune: AtuneService = app.state.atune
            atune_health = {
                "status": "running",
                "cycle_count": atune.cycle_count,
                "workspace_threshold": round(atune.workspace_threshold, 4),
                "meta_attention_mode": atune.meta_attention_mode,
                "affect": {
                    "valence": round(atune.current_affect.valence, 4),
                    "arousal": round(atune.current_affect.arousal, 4),
                    "curiosity": round(atune.current_affect.curiosity, 4),
                    "care_activation": round(atune.current_affect.care_activation, 4),
                    "coherence_stress": round(atune.current_affect.coherence_stress, 4),
                },
            }
        except Exception as exc:
            error_msg = str(exc) or f"{type(exc).__name__} (no message)"
            atune_health = {"status": "error", "error": error_msg}

    instance_id = getattr(getattr(app.state, "config", None), "instance_id", "unknown")

    return {
        "status": overall,
        "instance_id": instance_id,
        "instance_name": instance_name,
        "phase": "19_metrics_publisher",
        "systems": {
            "memory": memory_health,
            "equor": equor_health,
            "nova": nova_health,
            "axon": _safe_sync_stats("axon"),
            "evo": _safe_sync_stats("evo"),
            "simula": _safe_sync_stats("simula"),
            "atune": atune_health,
            "voxis": voxis_health,
            "synapse": synapse_health,
            "thymos": thymos_health,
            "oneiros": oneiros_health,
            "thread": thread_health,
            "federation": federation_health,
        },
        "data_stores": {
            "neo4j": neo4j_health,
            "timescaledb": tsdb_health,
            "redis": redis_health,
        },
    }


@app.get("/api/v1/admin/llm/metrics")
async def get_llm_metrics():
    """
    LLM cost optimization dashboard.

    Returns token spend, cache hit rate, budget tier, latency,
    and per-system breakdowns.
    """
    llm_metrics: LLMMetricsCollector = app.state.llm_metrics
    token_budget: TokenBudget = app.state.token_budget
    llm: OptimizedLLMProvider = app.state.llm

    dashboard = llm_metrics.get_dashboard_data()
    budget_status = await token_budget.get_status()
    cache_stats = llm.cache.get_stats() if llm.cache else {"hit_rate": 0.0}

    import math

    def _safe_float(v: float) -> float | None:
        """Return None for inf/nan so the JSON encoder doesn't crash."""
        if math.isfinite(v):
            return round(v, 2)
        return None

    return {
        "status": "ok",
        "budget": {
            "tier": budget_status.tier.value,
            "tokens_used": budget_status.tokens_used,
            "tokens_remaining": budget_status.tokens_remaining,
            "calls_made": budget_status.calls_made,
            "calls_remaining": budget_status.calls_remaining,
            "burn_rate_tokens_per_sec": _safe_float(budget_status.tokens_per_sec),
            "hours_until_exhausted": _safe_float(budget_status.hours_until_exhausted),
            "warning": budget_status.warning_message,
        },
        "cache": cache_stats,
        "dashboard": dashboard,
    }


@app.get("/api/v1/admin/llm/summary")
async def get_llm_summary():
    """Human-readable LLM cost summary."""
    llm_metrics: LLMMetricsCollector = app.state.llm_metrics
    return {"summary": llm_metrics.summary()}


@app.get("/api/v1/admin/instance")
async def get_instance():
    """Get instance information."""
    instance = await app.state.memory.get_self()
    if instance is None:
        return {"status": "unborn", "message": "No instance has been born yet."}
    return instance.model_dump()


@app.get("/api/v1/admin/memory/stats")
async def get_memory_stats():
    """Get memory graph statistics."""
    return await app.state.memory.stats()


@app.get("/api/v1/governance/constitution")
async def get_constitution():
    """View the current constitution."""
    constitution = await app.state.memory.get_constitution()
    if constitution is None:
        return {"status": "not_found"}
    return constitution


@app.get("/api/v1/admin/health")
async def full_health():
    """Alias for /health with full detail."""
    return await health()


# ─── Phase 3: Perception via Atune ───────────────────────────────


@app.post("/api/v1/perceive/event")
async def perceive_event(body: dict[str, Any]):
    """
    Ingest a percept through Atune's full perception pipeline.

    The input is normalised, evaluated against world model predictions
    (salience = precision-weighted prediction error via Fovea), and if it
    passes the workspace ignition threshold, broadcast to all systems.

    Body: {text, channel?, metadata?}
    """
    text = body.get("text", body.get("content", ""))
    if not text:
        return {"error": "No text/content provided"}

    channel_str = body.get("channel", "text_chat")
    try:
        channel = InputChannel(channel_str)
    except ValueError:
        channel = InputChannel.TEXT_CHAT

    raw = RawInput(
        data=text,
        channel_id=body.get("channel_id", ""),
        metadata=body.get("metadata", {}),
    )

    # Ingest through Atune (normalise → predict → score → enqueue)
    atune: AtuneService = app.state.atune
    percept_id = await atune.ingest(raw, channel)

    if percept_id is None:
        return {"percept_id": None, "accepted": False, "reason": "queue_full"}

    # Percept is enqueued - Synapse clock will pick it up on the next tick.
    return {
        "percept_id": percept_id,
        "accepted": True,
        "queued": True,
        "salience_threshold": round(atune.workspace_threshold, 4),
        "affect": {
            "valence": round(atune.current_affect.valence, 4),
            "arousal": round(atune.current_affect.arousal, 4),
            "curiosity": round(atune.current_affect.curiosity, 4),
            "care_activation": round(atune.current_affect.care_activation, 4),
            "coherence_stress": round(atune.current_affect.coherence_stress, 4),
        },
    }


@app.get("/api/v1/atune/affect")
async def get_affect_state():
    """Get Atune's current affective state."""
    affect = app.state.atune.current_affect
    return {
        "valence": round(affect.valence, 4),
        "arousal": round(affect.arousal, 4),
        "dominance": round(affect.dominance, 4),
        "curiosity": round(affect.curiosity, 4),
        "care_activation": round(affect.care_activation, 4),
        "coherence_stress": round(affect.coherence_stress, 4),
        "timestamp": affect.timestamp.isoformat() if affect.timestamp else None,
    }


@app.get("/api/v1/atune/workspace")
async def get_workspace_state():
    """Get workspace state - threshold, recent broadcasts, meta-attention mode."""
    atune: AtuneService = app.state.atune
    recent = atune.recent_broadcasts
    return {
        "cycle_count": atune.cycle_count,
        "dynamic_threshold": round(atune.workspace_threshold, 4),
        "meta_attention_mode": atune.meta_attention_mode,
        "recent_broadcasts": [
            {
                "broadcast_id": b.broadcast_id,
                "salience": round(b.salience.composite, 4),
                "timestamp": b.timestamp.isoformat() if b.timestamp else None,
            }
            for b in recent[-10:]
        ],
    }


# ─── Phase 1: Memory Test Endpoints (kept for backwards compat) ──


@app.post("/api/v1/memory/retrieve")
async def retrieve_memory(body: dict[str, Any]):
    """
    Query memory (temporary test endpoint).
    In later phases, retrieval is triggered by the cognitive cycle.
    """
    query = body.get("query", "")
    if not query:
        return {"error": "No query provided"}

    response = await app.state.memory.retrieve(
        query_text=query,
        max_results=body.get("max_results", 10),
    )
    return response.model_dump()


# ─── Phase 2: Equor Endpoints ────────────────────────────────────


@app.post("/api/v1/equor/review")
async def review_intent(body: dict[str, Any]):
    """
    Submit an Intent for constitutional review (test endpoint).
    In later phases, Nova calls this automatically.

    Body: {goal, steps?, reasoning?, alternatives?, domain?, expected_free_energy?}
    """
    from primitives.intent import (
        Action,
        ActionSequence,
        DecisionTrace,
        GoalDescriptor,
        Intent,
    )

    goal_text = body.get("goal", "")
    if not goal_text:
        return {"error": "No goal provided"}

    steps = []
    for s in body.get("steps", []):
        steps.append(Action(
            executor=s.get("executor", ""),
            parameters=s.get("parameters", {}),
        ))

    intent = Intent(
        goal=GoalDescriptor(
            description=goal_text,
            target_domain=body.get("domain", ""),
        ),
        plan=ActionSequence(steps=steps),
        expected_free_energy=body.get("expected_free_energy", 0.0),
        decision_trace=DecisionTrace(
            reasoning=body.get("reasoning", ""),
            alternatives_considered=body.get("alternatives", []),
        ),
    )

    check = await app.state.equor.review(intent)
    return check.model_dump()


@app.get("/api/v1/equor/invariants")
async def get_invariants():
    """List all active invariants (hardcoded + community)."""
    return await app.state.equor.get_invariants()


@app.get("/api/v1/equor/drift")
async def get_drift():
    """Get the current drift report."""
    return await app.state.equor.get_drift_report()


@app.get("/api/v1/equor/autonomy")
async def get_autonomy():
    """Get the current autonomy level and promotion eligibility."""
    level = await app.state.equor.get_autonomy_level()
    next_level = level + 1 if level < 3 else None
    eligibility = None
    if next_level:
        eligibility = await app.state.equor.check_promotion(next_level)
    return {
        "current_level": level,
        "level_name": {1: "Advisor", 2: "Partner", 3: "Steward"}.get(level, "unknown"),
        "promotion_eligibility": eligibility,
    }


@app.get("/api/v1/governance/history")
async def governance_history():
    """View governance event history."""
    return await app.state.equor.get_governance_history()


@app.get("/api/v1/governance/reviews")
async def recent_reviews():
    """View recent constitutional reviews."""
    return await app.state.equor.get_recent_reviews()


@app.post("/api/v1/governance/amendments")
async def propose_amendment_endpoint(body: dict[str, Any]):
    """
    Propose a constitutional amendment.
    Body: {proposed_drives: {coherence, care, growth, honesty}, title, description, proposer_id}
    """
    required = ["proposed_drives", "title", "description", "proposer_id"]
    for field in required:
        if field not in body:
            return {"error": f"Missing required field: {field}"}

    return await app.state.equor.propose_amendment(
        proposed_drives=body["proposed_drives"],
        title=body["title"],
        description=body["description"],
        proposer_id=body["proposer_id"],
    )


# ─── Phase 4: Chat & Expression via Voxis ────────────────────────


@app.post("/api/v1/chat/message")
async def chat_message(body: dict[str, Any]):
    """
    Send a message to EOS and receive an expression in response.

    The full Voxis pipeline runs: silence check → policy selection via EFE
    → personality + affect colouring → audience adaptation → LLM generation
    → honesty check → response.

    Body: {message, conversation_id?, speaker_id?, speaker_name?}
    Returns: {expression_id, content, is_silence, silence_reason?, conversation_id,
              policy_class, efe_score, affect_snapshot, generation_trace?}
    """
    message = body.get("message", body.get("content", ""))
    if not message:
        return {"error": "No message provided"}

    voxis: VoxisService = app.state.voxis
    atune: AtuneService = app.state.atune
    current_affect = atune.current_affect

    conversation_id = body.get("conversation_id")
    speaker_id = body.get("speaker_id")
    speaker_name = body.get("speaker_name")

    try:
        # Record user message into conversation state first
        conversation_id = await voxis.ingest_user_message(
            message=message,
            conversation_id=conversation_id,
            speaker_id=speaker_id,
        )

        # Also feed through Atune (updates affect, workspace state).
        # No manual run_cycle() needed - Synapse clock picks up the percept.
        try:
            raw = RawInput(data=message, channel_id=conversation_id or "", metadata={})
            await atune.ingest(raw, InputChannel.TEXT_CHAT)
        except Exception as atune_err:
            # Atune ingestion is non-critical for chat - log and continue
            _chat_logger.warning("chat_atune_ingest_failed", error=str(atune_err))

        # Generate expression via Voxis.
        # NOTE: Do NOT pass the raw user message as content here - ingest_user_message()
        # already appended it to conversation history, and express() appends content as
        # an additional user turn. Passing message again would duplicate it in the LLM
        # context, causing the model to see the same prompt twice and loop on the same
        # defensive response. Pass a response directive instead.
        expression = await voxis.express(
            content="Respond to the conversation.",
            trigger=ExpressionTrigger.NOVA_RESPOND,
            conversation_id=conversation_id,
            addressee_id=speaker_id,
            addressee_name=speaker_name,
            affect=current_affect,
            urgency=0.6,
        )
    except Exception as exc:
        _chat_logger.error("chat_expression_failed", error=str(exc), exc_info=True)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={
                "error": "expression_pipeline_failed",
                "detail": str(exc),
                "stage": "voxis_express",
            },
        )

    # Include Thread identity context in response (P2.9)
    identity_context = ""
    if hasattr(app.state, "thread"):
        with suppress(Exception):
            identity_context = app.state.thread.get_identity_context()

    response: dict[str, Any] = {
        "expression_id": expression.id,
        "conversation_id": expression.conversation_id,
        "content": expression.content,
        "is_silence": expression.is_silence,
        "identity_context": identity_context,
    }

    if expression.is_silence:
        response["silence_reason"] = expression.silence_reason
    else:
        response["channel"] = expression.channel
        response["affect_snapshot"] = {
            "valence": round(expression.affect_valence, 4),
            "arousal": round(expression.affect_arousal, 4),
            "curiosity": round(expression.affect_curiosity, 4),
            "care_activation": round(expression.affect_care_activation, 4),
            "coherence_stress": round(expression.affect_coherence_stress, 4),
        }
        if expression.generation_trace:
            response["generation"] = {
                "model": expression.generation_trace.model,
                "temperature": expression.generation_trace.temperature,
                "latency_ms": expression.generation_trace.latency_ms,
                "honesty_check_passed": expression.generation_trace.honesty_check_passed,
            }

    return response


@app.get("/api/v1/voxis/personality")
async def get_voxis_personality():
    """Get the current personality vector and its dimensions."""
    voxis: VoxisService = app.state.voxis
    p = voxis.current_personality
    return {
        "warmth": p.warmth,
        "directness": p.directness,
        "verbosity": p.verbosity,
        "formality": p.formality,
        "curiosity_expression": p.curiosity_expression,
        "humour": p.humour,
        "empathy_expression": p.empathy_expression,
        "confidence_display": p.confidence_display,
        "metaphor_use": p.metaphor_use,
        "vocabulary_affinities": p.vocabulary_affinities,
        "thematic_references": p.thematic_references,
    }


@app.get("/api/v1/voxis/health")
async def get_voxis_health():
    """Voxis system health and observability metrics."""
    return await app.state.voxis.health()


# ─── Nova Endpoints ───────────────────────────────────────────────


@app.get("/api/v1/nova/health")
async def get_nova_health():
    """Nova decision system health and observability metrics."""
    return await app.state.nova.health()


@app.get("/api/v1/nova/goals")
async def get_nova_goals():
    """Active goals in Nova's goal manager."""
    nova: NovaService = app.state.nova
    goals = nova._goal_manager.active_goals if nova._goal_manager else []
    return {
        "active_goals": [
            {
                "id": g.id,
                "description": g.description,
                "source": g.source.value,
                "priority": round(g.priority, 4),
                "urgency": round(g.urgency, 4),
                "progress": round(g.progress, 4),
                "status": g.status.value,
            }
            for g in goals
        ],
        "total_active": len(goals),
    }


# ─── Phase 7: Evo Endpoints ───────────────────────────────────────


@app.get("/api/v1/evo/stats")
async def get_evo_stats():
    """Evo learning system stats - hypotheses, parameters, consolidation."""
    evo: EvoService = app.state.evo
    return evo.stats


@app.post("/api/v1/evo/consolidate")
async def trigger_consolidation():
    """Manually trigger an Evo consolidation cycle (sleep mode)."""
    evo: EvoService = app.state.evo
    result = await evo.run_consolidation()
    if result is None:
        return {
            "status": "skipped",
            "duration_ms": 0,
            "hypotheses_evaluated": 0,
            "hypotheses_integrated": 0,
            "procedures_extracted": 0,
            "parameters_adjusted": 0,
            "total_parameter_delta": 0.0,
        }

    # Get the result as dict and add status field
    result_dict = result.model_dump()
    result_dict["status"] = "completed"
    return result_dict


@app.get("/api/v1/evo/parameters")
async def get_evo_parameters():
    """Get all current Evo-tuned parameter values."""
    evo: EvoService = app.state.evo
    return evo.get_all_parameters()


@app.get("/api/v1/evo/health")
async def get_evo_health():
    """Detailed Evo health: broadcasts, consolidations, pending candidates, arXiv scanner."""
    evo: EvoService = app.state.evo
    return await evo.health()


@app.get("/api/v1/evo/hypotheses")
async def get_evo_hypotheses():
    """All active hypotheses with full detail: status, evidence score, volatility, mutation."""
    evo: EvoService = app.state.evo
    engine = evo._hypothesis_engine
    if engine is None:
        return {"hypotheses": [], "total": 0}
    hypotheses = engine.get_all_active()
    return {
        "hypotheses": [
            {
                "id": h.id,
                "category": h.category,
                "statement": h.statement,
                "formal_test": h.formal_test,
                "status": h.status,
                "evidence_score": round(h.evidence_score, 4),
                "supporting_count": len(h.supporting_episodes),
                "contradicting_count": len(h.contradicting_episodes),
                "complexity_penalty": h.complexity_penalty,
                "volatility_flag": h.volatility_flag,
                "volatility_weight": h.volatility_weight,
                "confidence_oscillations": h.confidence_oscillations,
                "proposed_mutation": h.proposed_mutation.model_dump() if h.proposed_mutation else None,
                "created_at": h.created_at.isoformat() if hasattr(h, "created_at") else None,
                "last_evidence_at": h.last_evidence_at.isoformat() if hasattr(h, "last_evidence_at") else None,
            }
            for h in hypotheses
        ],
        "total": len(hypotheses),
    }


@app.get("/api/v1/evo/tournaments")
async def get_evo_tournaments():
    """Active and recent hypothesis tournaments (Thompson sampling A/B tests)."""
    evo: EvoService = app.state.evo
    engine = evo._tournament_engine
    if engine is None:
        return {"tournaments": [], "stats": {}}
    all_tournaments = engine.get_all_tournaments()
    return {
        "tournaments": [
            {
                "id": t.id,
                "stage": t.stage,
                "sample_count": t.sample_count,
                "winner_id": t.winner_id,
                "convergence_threshold": t.convergence_threshold,
                "burn_in_trials": t.burn_in_trials,
                "hypotheses": [
                    {
                        "id": ref.hypothesis_id,
                        "statement": ref.statement,
                        "alpha": t.beta_parameters[ref.hypothesis_id].alpha if ref.hypothesis_id in t.beta_parameters else 1.0,
                        "beta": t.beta_parameters[ref.hypothesis_id].beta if ref.hypothesis_id in t.beta_parameters else 1.0,
                        "mean": t.beta_parameters[ref.hypothesis_id].mean if ref.hypothesis_id in t.beta_parameters else 0.5,
                        "sample_count": t.beta_parameters[ref.hypothesis_id].sample_count if ref.hypothesis_id in t.beta_parameters else 0,
                    }
                    for ref in t.hypotheses
                    if ref.hypothesis_id in t.beta_parameters
                ],
                "is_running": t.is_running,
                "is_converged": t.is_converged,
            }
            for t in all_tournaments
        ],
        "stats": engine.stats,
    }


@app.get("/api/v1/evo/self-model")
async def get_evo_self_model():
    """Current self-model statistics: capability scores, regret, constitutional alignment."""
    evo: EvoService = app.state.evo
    self_model = evo._self_model
    if self_model is None:
        return {"available": False}
    current = self_model.get_current()
    if current is None:
        return {"available": False}
    return {
        "available": True,
        "success_rate": current.success_rate,
        "mean_alignment": current.mean_alignment,
        "total_outcomes_evaluated": current.total_outcomes_evaluated,
        "mean_regret": current.regret.mean_regret,
        "high_regret_count": current.regret.high_regret_count,
        "total_regret_resolved": current.regret.total_resolved,
        "capability_scores": {
            k: {"success_rate": v.rate, "sample_count": v.total_count}
            for k, v in current.capability_scores.items()
        },
        "regret_by_policy_type": current.regret.regret_by_policy_type,
        "regret_by_goal_domain": current.regret.regret_by_goal_domain,
        "updated_at": current.updated_at.isoformat(),
    }


@app.get("/api/v1/evo/stale-beliefs")
async def get_evo_stale_beliefs():
    """Beliefs that have decayed past the staleness threshold (age_factor < 0.5)."""
    evo: EvoService = app.state.evo
    stale = await evo.get_stale_beliefs()
    return {"beliefs": stale, "total": len(stale)}


@app.get("/api/v1/evo/patterns")
async def get_evo_patterns():
    """Current in-flight pattern context: cooccurrences, sequences, temporal bins, pending candidates."""
    evo: EvoService = app.state.evo
    ctx = evo._pattern_context
    candidates = evo._pending_candidates
    return {
        "episodes_scanned": ctx.episodes_scanned,
        "cooccurrence_count": len(ctx.cooccurrence_counts),
        "top_cooccurrences": sorted(
            ctx.cooccurrence_counts.items(), key=lambda x: x[1], reverse=True
        )[:10],
        "sequence_count": len(ctx.sequence_counts),
        "top_sequences": sorted(
            ctx.sequence_counts.items(), key=lambda x: x[1], reverse=True
        )[:5],
        "temporal_bin_count": len(ctx.temporal_bins),
        "affect_pattern_count": len(ctx.affect_responses),
        "pending_candidates": len(candidates),
        "candidate_types": {
            t: sum(1 for c in candidates if c.type == t)
            for t in set(c.type for c in candidates)
        },
    }


# ─── Thread: Narrative Identity Endpoints ──────────────────────────


@app.get("/api/v1/thread/who-am-i")
async def thread_who_am_i():
    """The organism's current identity summary - who it thinks it is."""
    thread: ThreadService = app.state.thread
    return thread.who_am_i()


@app.get("/api/v1/thread/health")
async def thread_health_endpoint():
    """Thread system health and identity metrics."""
    thread: ThreadService = app.state.thread
    return await thread.health()


@app.get("/api/v1/thread/identity")
async def thread_identity():
    """Complete narrative identity snapshot."""
    thread: ThreadService = app.state.thread

    # Gather schemas by strength
    core_schemas = []
    established_schemas = []
    developing_schemas = []
    nascent_schemas = []
    total_idem = 0.0

    for s in thread._schemas:
        schema_data = {
            "id": s.id,
            "statement": s.statement,
            "strength": getattr(s, 'strength', 'nascent').value if hasattr(getattr(s, 'strength', None), 'value') else 'nascent',  # type: ignore[union-attr]
            "valence": getattr(s, 'valence', 'ambivalent'),
            "confirmation_count": getattr(s, 'confirmation_count', 0),
            "disconfirmation_count": getattr(s, 'disconfirmation_count', 0),
            "evidence_ratio": round(getattr(s, 'evidence_ratio', 0.5), 3),
            "trigger_contexts": getattr(s, 'trigger_contexts', []),
            "behavioral_tendency": getattr(s, 'behavioral_tendency', None),
        }

        strength = schema_data["strength"]
        if strength == "core":
            core_schemas.append(schema_data)
        elif strength == "established":
            established_schemas.append(schema_data)
        elif strength == "developing":
            developing_schemas.append(schema_data)
        else:
            nascent_schemas.append(schema_data)

        confidence = s.confidence if hasattr(s, 'confidence') else 0.5
        total_idem += confidence

    idem_score = total_idem / len(thread._schemas) if thread._schemas else 0.0

    # Gather active commitments
    active_commitments = []
    total_ipse = 0.0

    for c in thread._commitments:
        commitment_data = {
            "id": c.id,
            "statement": c.statement,
            "source": c.drive_source or "explicit_declaration",
            "status": c.type.value if hasattr(c, 'type') else "active",
            "tests_faced": c.test_count if hasattr(c, 'test_count') else 0,
            "tests_held": max(0, c.test_count - 1) if hasattr(c, 'test_count') and c.test_count > 0 else 0,
            "fidelity": round(c.fidelity, 3),
            "made_at": c.created_at.isoformat() if hasattr(c, 'created_at') else None,
            "last_tested": None,
        }
        active_commitments.append(commitment_data)
        total_ipse += c.fidelity

    ipse_score = total_ipse / len(thread._commitments) if thread._commitments else 0.0

    # Get current chapter
    current_chapter_title = None
    current_chapter_theme = None
    if hasattr(thread, '_chapters') and thread._chapters:
        latest_ch = thread._chapters[-1] if thread._chapters else None
        if latest_ch:
            current_chapter_title = latest_ch.title if hasattr(latest_ch, 'title') else "Forming..."
            current_chapter_theme = latest_ch.theme if hasattr(latest_ch, 'theme') else None

    # Get turning points (from latest chapter if available)
    recent_turning_points: list[dict[str, Any]] = []
    if hasattr(thread, '_chapters') and thread._chapters:
        latest_ch = thread._chapters[-1]
        # Attempt to extract turning points from the chapter
        if hasattr(latest_ch, 'turning_points'):
            for tp in latest_ch.turning_points:
                recent_turning_points.append({
                    "id": getattr(tp, 'id', f"tp_{len(recent_turning_points)}"),
                    "type": getattr(tp, 'type', 'growth'),
                    "description": getattr(tp, 'description', str(tp)),
                    "surprise_magnitude": getattr(tp, 'surprise_magnitude', 0.5),
                    "narrative_weight": getattr(tp, 'narrative_weight', 0.5),
                })

    # Get personality traits
    key_personality_traits = {}
    if hasattr(thread, '_personality_snapshot'):
        key_personality_traits = thread._personality_snapshot if isinstance(thread._personality_snapshot, dict) else {}

    # Get life story
    life_story_summary = None
    if hasattr(thread, '_life_story') and thread._life_story:
        life_story_summary = getattr(thread._life_story, 'summary', None)

    # Determine narrative coherence
    narrative_coherence = "integrated"  # Default to integrated
    if hasattr(thread, '_coherence_status'):
        narrative_coherence = thread._coherence_status

    return {
        "core_schemas": core_schemas,
        "established_schemas": established_schemas,
        "active_commitments": active_commitments,
        "current_chapter_title": current_chapter_title,
        "current_chapter_theme": current_chapter_theme,
        "life_story_summary": life_story_summary,
        "key_personality_traits": key_personality_traits,
        "recent_turning_points": recent_turning_points,
        "narrative_coherence": narrative_coherence,
        "idem_score": round(idem_score, 3),
        "ipse_score": round(ipse_score, 3),
    }


@app.get("/api/v1/thread/commitments")
async def thread_commitments():
    """All identity commitments - constitutional and emergent."""
    thread: ThreadService = app.state.thread

    commitments = []
    strained = []
    resolved = []
    total_fidelity = 0.0

    for c in thread._commitments:
        status = c.type.value if hasattr(c, 'type') else "active"
        commitment_data = {
            "id": c.id,
            "statement": c.statement,
            "source": c.drive_source or "explicit_declaration",
            "status": status,
            "tests_faced": c.test_count if hasattr(c, 'test_count') else 0,
            "tests_held": max(0, c.test_count - 1) if hasattr(c, 'test_count') and c.test_count > 0 else 0,
            "fidelity": round(c.fidelity, 3),
            "made_at": c.created_at.isoformat() if hasattr(c, 'created_at') else None,
            "last_tested": None,
        }
        commitments.append(commitment_data)
        total_fidelity += c.fidelity

        if status == "resolved":
            resolved.append(commitment_data)
        elif c.fidelity < 0.6:
            strained.append(commitment_data)

    ipse_score = total_fidelity / len(commitments) if commitments else 0.0

    return {
        "commitments": commitments,
        "total": len(commitments),
        "ipse_score": round(ipse_score, 3),
        "strained": strained,
        "resolved": resolved,
    }


@app.get("/api/v1/thread/schemas")
async def thread_schemas():
    """All identity schemas - the organism's self-understanding."""
    thread: ThreadService = app.state.thread

    # Categorize schemas by strength
    core_schemas = []
    established_schemas = []
    developing_schemas = []
    nascent_schemas = []
    total_confidence = 0.0

    for s in thread._schemas:
        schema_data = {
            "id": s.id,
            "statement": s.statement,
            "strength": getattr(s, 'strength', 'nascent').value if hasattr(getattr(s, 'strength', None), 'value') else 'nascent',  # type: ignore[union-attr]
            "valence": getattr(s, 'valence', 'ambivalent'),
            "confirmation_count": getattr(s, 'confirmation_count', 0),
            "disconfirmation_count": getattr(s, 'disconfirmation_count', 0),
            "evidence_ratio": round(getattr(s, 'evidence_ratio', s.confidence if hasattr(s, 'confidence') else 0.5), 3),
            "trigger_contexts": getattr(s, 'trigger_contexts', []),
            "behavioral_tendency": getattr(s, 'behavioral_tendency', None),
        }

        strength = schema_data["strength"]
        if strength == "core":
            core_schemas.append(schema_data)
        elif strength == "established":
            established_schemas.append(schema_data)
        elif strength == "developing":
            developing_schemas.append(schema_data)
        else:
            nascent_schemas.append(schema_data)

        confidence = s.confidence if hasattr(s, 'confidence') else 0.5
        total_confidence += confidence

    idem_score = total_confidence / len(thread._schemas) if thread._schemas else 0.0

    return {
        "schemas": {
            "core": core_schemas,
            "established": established_schemas,
            "developing": developing_schemas,
            "nascent": nascent_schemas,
        },
        "total": len(thread._schemas),
        "idem_score": round(idem_score, 3),
    }


@app.get("/api/v1/thread/coherence")
async def thread_coherence():
    """Diachronic coherence - behavioral fingerprints over time with drift analysis."""
    thread: ThreadService = app.state.thread

    # Get recent fingerprints (last 50)
    recent_fps = list(thread._fingerprints[-50:]) if hasattr(thread, '_fingerprints') else []

    def _classify_distance(dist: float) -> str:
        if dist < 0.05:
            return "stable"
        elif dist < 0.15:
            return "growth"
        elif dist < 0.25:
            return "transition"
        else:
            return "drift"

    fingerprints = []
    for i, fp in enumerate(recent_fps):
        # Distance from previous fingerprint
        distance_from_prev: float | None = None
        drift_classification: str | None = None
        if i > 0:
            prev = recent_fps[i - 1]
            prev_vec = list(getattr(prev, 'vector', []))
            curr_vec = list(getattr(fp, 'vector', []))
            target_len = 29
            while len(prev_vec) < target_len:
                prev_vec.append(0.0)
            while len(curr_vec) < target_len:
                curr_vec.append(0.0)
            if prev_vec and curr_vec:
                distance_from_prev = round(
                    sum(abs(a - b) for a, b in zip(curr_vec, prev_vec, strict=False)) / len(curr_vec), 4
                )
                drift_classification = _classify_distance(distance_from_prev)

        ts = int(fp.created_at.timestamp()) if hasattr(fp, 'created_at') and fp.created_at else 0
        fingerprints.append({
            "id": fp.id,
            "epoch": len(recent_fps) - i - 1,
            "cycle_number": getattr(fp, 'cycle_number', 0),
            "window_start": ts,
            "window_end": ts + 3600,
            "created_at": fp.created_at.isoformat() if hasattr(fp, 'created_at') and fp.created_at else None,
            "personality": list(getattr(fp, 'personality', [0.0] * 9)),
            "drive_alignment": list(getattr(fp, 'drive_alignment', [0.0] * 4)),
            "affect": list(getattr(fp, 'affect', [0.0] * 6)),
            "goal_profile": list(getattr(fp, 'goal_profile', [0.0] * 5)),
            "interaction_profile": list(getattr(fp, 'interaction_profile', [0.0] * 5)),
            "distance_from_prev": distance_from_prev,
            "drift_classification": drift_classification,
        })

    # Current drift: compare latest vs oldest available
    current_drift: str | None = None
    current_distance: float | None = None
    if len(recent_fps) >= 2:
        latest_vec = list(getattr(recent_fps[-1], 'vector', []))
        earliest_vec = list(getattr(recent_fps[0], 'vector', []))
        target_len = 29
        while len(latest_vec) < target_len:
            latest_vec.append(0.0)
        while len(earliest_vec) < target_len:
            earliest_vec.append(0.0)
        if latest_vec and earliest_vec:
            current_distance = round(
                sum(abs(a - b) for a, b in zip(latest_vec, earliest_vec, strict=False)) / len(latest_vec), 4
            )
            current_drift = _classify_distance(current_distance)

    return {
        "fingerprint_count": len(fingerprints),
        "recent_fingerprints": fingerprints,
        "current_drift": current_drift,
        "current_distance": current_distance,
    }


@app.get("/api/v1/thread/fingerprints")
async def thread_fingerprints():
    """Recent identity fingerprints (29D snapshots over time)."""
    thread: ThreadService = app.state.thread
    return [
        {
            "id": fp.id,
            "cycle_number": fp.cycle_number,
            "personality": fp.personality,
            "drive_alignment": fp.drive_alignment,
            "affect": fp.affect,
            "goal_profile": fp.goal_profile,
            "interaction_profile": fp.interaction_profile,
            "created_at": fp.created_at.isoformat(),
        }
        for fp in thread._fingerprints[-50:]  # Last 50
    ]


@app.get("/api/v1/thread/chapters")
async def thread_chapters():
    """Narrative chapters - the organism's life phases."""
    thread: ThreadService = app.state.thread

    chapters = []
    for ch in thread._chapters:
        created_at_iso = ch.created_at.isoformat() if hasattr(ch, 'created_at') and ch.created_at else None
        started_at_iso = (
            ch.started_at.isoformat() if hasattr(ch, 'started_at') and ch.started_at else created_at_iso
        )
        ended_at_iso = (
            ch.ended_at.isoformat() if hasattr(ch, 'ended_at') and ch.ended_at else None
        )
        theme = getattr(ch, 'theme', '') or ''
        status_val = ch.status.value if hasattr(ch.status, 'value') else str(getattr(ch, 'status', 'active'))
        chapters.append({
            "id": ch.id,
            "title": getattr(ch, 'title', '') or '',
            "theme": theme,
            "themes": [theme] if theme else [],
            "status": status_val,
            "opened_at_cycle": getattr(ch, 'opened_at_cycle', 0),
            "closed_at_cycle": getattr(ch, 'closed_at_cycle', None),
            "summary": getattr(ch, 'summary', None) or None,
            "created_at": created_at_iso,
            "began_at": started_at_iso,
            "ended_at": ended_at_iso,
        })

    return {"chapters": chapters}


@app.get("/api/v1/thread/chapters/current")
async def thread_current_chapter():
    """Current narrative chapter context."""
    thread: ThreadService = app.state.thread

    # Get the latest chapter
    current_ch = None
    if hasattr(thread, '_chapters') and thread._chapters:
        current_ch = thread._chapters[-1]

    if not current_ch:
        return {
            "title": None,
            "theme": None,
            "arc_type": "unknown",
            "episode_count": 0,
            "scenes": [],
            "turning_points": [],
            "status": "forming",
        }

    # Extract scenes and turning points
    scenes = []
    turning_points = []

    if hasattr(current_ch, 'scenes'):
        scenes = [str(scene) for scene in current_ch.scenes] if isinstance(current_ch.scenes, list) else []

    if hasattr(current_ch, 'turning_points'):
        turning_points = [str(tp) for tp in current_ch.turning_points] if isinstance(current_ch.turning_points, list) else []

    return {
        "title": getattr(current_ch, 'title', "Forming..."),
        "theme": getattr(current_ch, 'theme', None),
        "arc_type": getattr(current_ch, 'arc_type', "unknown"),
        "episode_count": 1,  # Placeholder
        "scenes": scenes,
        "turning_points": turning_points,
        "status": getattr(current_ch, 'status', 'forming').value if hasattr(getattr(current_ch, 'status', None), 'value') else 'forming',  # type: ignore[union-attr]
    }


@app.get("/api/v1/thread/past-self")
async def thread_past_self(cycle: int = 0):
    """
    View the organism's identity at a past point in time.
    Query param: cycle=N (0 = earliest available).
    """
    thread: ThreadService = app.state.thread
    return thread.get_past_self(cycle_reference=cycle)


@app.get("/api/v1/thread/life-story")
async def thread_life_story():
    """The organism's latest autobiographical synthesis."""
    thread: ThreadService = app.state.thread
    if thread._life_story is None:
        return {"status": "not_yet_synthesised", "message": "Life story not yet generated"}
    return thread._life_story.model_dump()


@app.get("/api/v1/thread/conflicts")
async def thread_conflicts():
    """Detected schema conflicts - contradictions in self-understanding."""
    thread: ThreadService = app.state.thread

    def _severity_from_similarity(sim: float) -> str:
        # High cosine similarity between conflicting beliefs = more severe
        if sim >= 0.8:
            return "CRITICAL"
        elif sim >= 0.65:
            return "HIGH"
        elif sim >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    conflicts = []
    for c in thread._conflicts:
        sim = getattr(c, 'cosine_similarity', 0.5)
        stmt_a = getattr(c, 'schema_a_statement', '')
        stmt_b = getattr(c, 'schema_b_statement', '')
        conflicts.append({
            "id": c.id,
            "description": f'"{stmt_a}" contradicts "{stmt_b}"',
            "schema_a_statement": stmt_a,
            "schema_b_statement": stmt_b,
            "drives_in_tension": [
                stmt_a[:50] + "…" if len(stmt_a) > 50 else stmt_a,
                stmt_b[:50] + "…" if len(stmt_b) > 50 else stmt_b,
            ],
            "severity": _severity_from_similarity(sim),
            "since": c.created_at.isoformat() if hasattr(c, 'created_at') and c.created_at else None,
            "cosine_similarity": round(sim, 3),
            "resolved": getattr(c, 'resolved', False),
            "created_at": c.created_at.isoformat() if hasattr(c, 'created_at') and c.created_at else None,
        })

    return {"conflicts": conflicts}


@app.get("/api/v1/thread/identity-context")
async def thread_identity_context():
    """Brief identity context string (injected into Voxis expressions)."""
    thread: ThreadService = app.state.thread
    return {"context": thread.get_identity_context()}


# ─── Phase 8: Simula Endpoints ────────────────────────────────────


@app.get("/api/v1/simula/stats")
async def get_simula_stats():
    """Simula self-evolution system stats."""
    simula: SimulaService = app.state.simula
    return simula.stats


@app.post("/api/v1/simula/proposals")
async def submit_evolution_proposal(body: dict[str, Any]):
    """
    Submit an evolution proposal to Simula.

    Body: {
      source: "evo" | "governance",
      category: ChangeCategory value,
      description: str,
      change_spec: {... see ChangeSpec fields},
      evidence: [list of hypothesis/episode IDs],
      expected_benefit: str,
      risk_assessment: str
    }
    """
    from systems.simula.evolution_types import (
        ChangeCategory,
        ChangeSpec,
        EvolutionProposal,
    )

    try:
        category = ChangeCategory(body.get("category", ""))
    except ValueError:
        return {"error": f"Unknown category: {body.get('category')}"}

    spec_data = body.get("change_spec", {})
    try:
        change_spec = ChangeSpec(**spec_data)
    except Exception as exc:
        return {"error": f"Invalid change_spec: {exc}"}

    proposal = EvolutionProposal(
        source=body.get("source", "governance"),
        category=category,
        description=body.get("description", ""),
        change_spec=change_spec,
        evidence=body.get("evidence", []),
        expected_benefit=body.get("expected_benefit", ""),
        risk_assessment=body.get("risk_assessment", ""),
    )

    simula: SimulaService = app.state.simula
    result = await simula.process_proposal(proposal)
    return {
        "proposal_id": proposal.id,
        "result": result.model_dump(),
    }


@app.get("/api/v1/simula/history")
async def get_evolution_history(limit: int = 50):
    """Get the evolution history - all structural changes applied."""
    simula: SimulaService = app.state.simula
    records = await simula.get_history(limit=limit)
    return {
        "records": [r.model_dump() for r in records],
        "current_version": await simula.get_current_version(),
    }


@app.get("/api/v1/simula/version")
async def get_simula_version():
    """Get the current config version and version chain."""
    simula: SimulaService = app.state.simula
    chain = await simula.get_version_chain()
    return {
        "current_version": await simula.get_current_version(),
        "version_chain": [v.model_dump() for v in chain],
    }


@app.post("/api/v1/simula/proposals/{proposal_id}/approve")
async def approve_governed_proposal(proposal_id: str, body: dict[str, Any]):
    """
    Approve a governed proposal after community governance.
    Body: {governance_record_id: str}
    """
    governance_record_id = body.get("governance_record_id", "")
    if not governance_record_id:
        return {"error": "governance_record_id required"}

    simula: SimulaService = app.state.simula
    result = await simula.approve_governed_proposal(proposal_id, governance_record_id)
    return result.model_dump()


@app.get("/api/v1/simula/proposals")
async def get_active_proposals():
    """Get all proposals currently active in the Simula pipeline."""
    simula: SimulaService = app.state.simula
    proposals = simula.get_active_proposals()
    return {
        "proposals": [
            {
                "id": p.id,
                "category": p.category.value,
                "description": p.description,
                "status": p.status.value,
                "source": p.source,
                "created_at": p.created_at.isoformat(),
            }
            for p in proposals
        ],
        "total": len(proposals),
    }


@app.get("/api/v1/nova/beliefs")
async def get_nova_beliefs():
    """Nova's current belief state summary."""
    nova: NovaService = app.state.nova
    beliefs = nova.beliefs
    return {
        "overall_confidence": round(beliefs.overall_confidence, 4),
        "free_energy": round(beliefs.free_energy, 4),
        "entity_count": len(beliefs.entities),
        "individual_count": len(beliefs.individual_beliefs),
        "context": {
            "summary": beliefs.current_context.summary[:200],
            "domain": beliefs.current_context.domain,
            "is_active_dialogue": beliefs.current_context.is_active_dialogue,
            "confidence": round(beliefs.current_context.confidence, 4),
        },
        "self_belief": {
            "epistemic_confidence": round(beliefs.self_belief.epistemic_confidence, 4),
            "cognitive_load": round(beliefs.self_belief.cognitive_load, 4),
        },
        "last_updated": beliefs.last_updated.isoformat(),
    }


# ─── Phase 9: Synapse Endpoints ──────────────────────────────────


@app.get("/api/v1/admin/synapse/cycle")
async def get_synapse_cycle():
    """
    Synapse cognitive cycle telemetry.

    Returns cycle count, period, latency, jitter, rhythm state,
    and coherence - the pulse of the organism.
    """
    synapse: SynapseService = app.state.synapse
    clock = synapse.clock_state
    rhythm = synapse.rhythm_snapshot
    coherence = synapse.coherence_snapshot

    return {
        "cycle_count": clock.cycle_count,
        "current_period_ms": round(clock.current_period_ms, 2),
        "target_period_ms": round(clock.target_period_ms, 2),
        "actual_rate_hz": round(clock.actual_rate_hz, 2),
        "jitter_ms": round(clock.jitter_ms, 2),
        "arousal": round(clock.arousal, 4),
        "overrun_count": clock.overrun_count,
        "running": clock.running,
        "paused": clock.paused,
        "rhythm": {
            "state": rhythm.state.value,
            "confidence": rhythm.confidence,
            "broadcast_density": rhythm.broadcast_density,
            "salience_trend": rhythm.salience_trend,
            "salience_mean": rhythm.salience_mean,
            "rhythm_stability": rhythm.rhythm_stability,
            "cycles_in_state": rhythm.cycles_in_state,
        },
        "coherence": {
            "composite": coherence.composite,
            "phi": coherence.phi_approximation,
            "resonance": coherence.system_resonance,
            "diversity": coherence.broadcast_diversity,
            "synchrony": coherence.response_synchrony,
        },
    }


@app.get("/api/v1/admin/synapse/budget")
async def get_synapse_budget():
    """Resource utilisation and budget allocation."""
    synapse: SynapseService = app.state.synapse
    return synapse.stats.get("resources", {})


@app.post("/api/v1/admin/synapse/safe-mode")
async def toggle_safe_mode(body: dict[str, Any]):
    """
    Manually toggle safe mode.

    Body: {enabled: bool, reason?: str}
    """
    enabled = body.get("enabled", False)
    reason = body.get("reason", "")
    synapse: SynapseService = app.state.synapse
    await synapse.set_safe_mode(enabled, reason)
    return {
        "safe_mode": synapse.is_safe_mode,
        "reason": reason if enabled else "",
    }


@app.get("/api/v1/admin/synapse/stats")
async def get_synapse_stats():
    """Full Synapse system statistics."""
    synapse: SynapseService = app.state.synapse
    return synapse.stats


@app.post("/api/v1/admin/clock/pause")
async def pause_clock():
    """Pause the cognitive cycle clock."""
    synapse: SynapseService = app.state.synapse
    synapse.pause_clock()
    return {"paused": True, "cycle_count": synapse.clock_state.cycle_count}


@app.post("/api/v1/admin/clock/resume")
async def resume_clock():
    """Resume the cognitive cycle clock after a pause."""
    synapse: SynapseService = app.state.synapse
    synapse.resume_clock()
    return {"paused": False, "cycle_count": synapse.clock_state.cycle_count}


@app.post("/api/v1/admin/clock/speed")
async def set_clock_speed(body: dict[str, Any]):
    """
    Set the base clock frequency.

    Body: {hz: float}  - clamped to 1–20 Hz.
    Arousal modulation still operates on top of this base frequency.
    """
    hz = body.get("hz")
    if hz is None:
        return {"error": "hz required"}
    try:
        hz = float(hz)
    except (TypeError, ValueError):
        return {"error": "hz must be a number"}
    synapse: SynapseService = app.state.synapse
    synapse.set_clock_speed(hz)
    state = synapse.clock_state
    return {
        "hz_requested": hz,
        "period_ms": round(state.current_period_ms, 2),
        "actual_rate_hz": round(state.actual_rate_hz, 2),
    }


@app.get("/api/v1/admin/synapse/health")
async def get_synapse_health():
    """
    Health monitor state: per-system health records, safe mode status,
    degradation level, and aggregate failure statistics.
    """
    synapse: SynapseService = app.state.synapse
    health_stats = synapse.stats.get("health", {})
    degradation_stats = synapse.stats.get("degradation", {})

    # Build full health records from live health monitor
    all_records = synapse._health.get_all_records()
    systems = {}
    for sid, rec in all_records.items():
        systems[sid] = {
            "system_id": rec.system_id,
            "status": rec.status.value,
            "consecutive_misses": rec.consecutive_misses,
            "consecutive_successes": rec.consecutive_successes,
            "total_checks": rec.total_checks,
            "total_failures": rec.total_failures,
            "last_check_time": rec.last_check_time.isoformat() if rec.last_check_time else None,
            "last_success_time": rec.last_success_time.isoformat() if rec.last_success_time else None,
            "last_failure_time": rec.last_failure_time.isoformat() if rec.last_failure_time else None,
            "latency_ema_ms": round(rec.latency_ema_ms, 2),
            "latency_peak_ms": round(rec.latency_peak_ms, 2),
            "restart_count": rec.restart_count,
            "is_critical": rec.is_critical,
        }

    return {
        "safe_mode": synapse.is_safe_mode,
        "safe_mode_reason": health_stats.get("safe_mode_reason", ""),
        "degradation_level": degradation_stats.get("level", "nominal"),
        "systems": systems,
        "total_checks": health_stats.get("total_checks", 0),
        "total_failures_detected": health_stats.get("failures_detected", 0),
        "total_recoveries": health_stats.get("recoveries", 0),
    }


@app.get("/api/v1/admin/synapse/resources")
async def get_synapse_resources():
    """
    Resource allocator state: current CPU/memory snapshot,
    per-system allocations, and configured budgets.
    """
    synapse: SynapseService = app.state.synapse
    synapse.stats.get("resources", {})

    # Build allocations dict from live allocator
    allocations: dict[str, Any] = {}
    for sid, _budget in synapse._resources._budgets.items():
        alloc = synapse._resources.get_allocation(sid)
        allocations[sid] = {
            "system_id": sid,
            "compute_ms_per_cycle": round(alloc.compute_ms_per_cycle, 2) if alloc else 0.0,
            "burst_allowance": round(alloc.burst_allowance, 3) if alloc else 1.0,
            "priority_boost": round(alloc.priority_boost, 3) if alloc else 0.0,
        }

    # Build budgets dict
    budgets: dict[str, Any] = {
        sid: {
            "cpu_share": b.cpu_share,
            "memory_mb": b.memory_mb,
            "io_priority": b.io_priority,
        }
        for sid, b in synapse._resources._budgets.items()
    }

    # Build snapshot
    snapshot_out = None
    snap = synapse._resources.latest_snapshot
    if snap:
        snapshot_out = {
            "total_cpu_percent": snap.total_cpu_percent,
            "total_memory_mb": round(snap.total_memory_mb, 1),
            "total_memory_percent": snap.total_memory_percent,
            "process_cpu_percent": snap.process_cpu_percent,
            "process_memory_mb": round(snap.process_memory_mb, 1),
            "timestamp": snap.timestamp.isoformat(),
        }

    return {
        "snapshot": snapshot_out,
        "allocations": allocations,
        "budgets": budgets,
    }


@app.get("/api/v1/admin/synapse/metabolism")
async def get_synapse_metabolism():
    """
    Metabolic tracker state: burn rate, rolling deficit, per-system
    cost breakdown, token usage, and depletion projection.
    """
    synapse: SynapseService = app.state.synapse
    snap = synapse.metabolic_snapshot
    return {
        "rolling_deficit_usd": round(snap.rolling_deficit_usd, 6),
        "window_cost_usd": round(snap.window_cost_usd, 6),
        "per_system_cost_usd": {k: round(v, 6) for k, v in snap.per_system_cost_usd.items()},
        "burn_rate_usd_per_sec": round(snap.burn_rate_usd_per_sec, 8),
        "burn_rate_usd_per_hour": round(snap.burn_rate_usd_per_hour, 6),
        "total_input_tokens": snap.total_input_tokens,
        "total_output_tokens": snap.total_output_tokens,
        "total_calls": snap.total_calls,
        "hours_until_depleted": snap.hours_until_depleted if snap.hours_until_depleted != float("inf") else -1,
        "timestamp": snap.timestamp.isoformat(),
    }


@app.get("/api/v1/admin/synapse/degradation")
async def get_synapse_degradation():
    """
    Degradation manager state: current level, per-system strategies,
    and restart attempt counts.
    """
    # import at call-site to avoid circular import issues
    from systems.synapse.degradation import _STRATEGIES as _DEG_STRATS

    synapse: SynapseService = app.state.synapse
    degradation_stats = synapse.stats.get("degradation", {})

    strategies = {
        sid: {
            "critical": s.triggers_safe_mode,
            "fallback": s.fallback_behavior,
            "auto_restart": s.auto_restart,
            "max_attempts": s.max_restart_attempts,
        }
        for sid, s in _DEG_STRATS.items()
    }

    return {
        "level": degradation_stats.get("level", "nominal"),
        "strategies": strategies,
        "restart_attempts": degradation_stats.get("restart_attempts", {}),
        "active_restart_tasks": degradation_stats.get("active_restart_tasks", []),
    }


@app.post("/api/v1/admin/synapse/inject-revenue")
async def inject_synapse_revenue(body: dict[str, Any]):
    """
    Inject revenue to offset the metabolic deficit.

    Body: {amount_usd: float, source?: str}
    """
    amount_usd = body.get("amount_usd")
    if amount_usd is None:
        return {"error": "amount_usd required"}
    try:
        amount_usd = float(amount_usd)
    except (TypeError, ValueError):
        return {"error": "amount_usd must be a number"}
    if amount_usd <= 0:
        return {"error": "amount_usd must be positive"}
    source = str(body.get("source", "manual"))
    synapse: SynapseService = app.state.synapse
    await synapse.inject_revenue(amount_usd, source=source)
    return {
        "deficit_usd": round(synapse.metabolic_deficit, 6),
        "total_revenue_usd": round(synapse.metabolic_snapshot.rolling_deficit_usd + amount_usd, 6),
    }


@app.get("/api/v1/debug/cycle-status")
async def get_cycle_status():
    """
    Lightweight cycle health snapshot for development and monitoring.

    Returns cycle count, current Hz, paused state, and whether the clock
    is running - the minimum needed to confirm the continuous cycle is live.
    """
    synapse: SynapseService = app.state.synapse
    clock = synapse.clock_state
    return {
        "running": clock.running,
        "paused": clock.paused,
        "cycle_count": clock.cycle_count,
        "hz": round(clock.actual_rate_hz, 2),
        "period_ms": round(clock.current_period_ms, 2),
        "jitter_ms": round(clock.jitter_ms, 2),
        "overrun_count": clock.overrun_count,
        "arousal": round(clock.arousal, 4),
    }


# ─── Phase 2: Multi-Channel Perception Endpoints ─────────────────


@app.get("/api/v1/perception/file-watcher")
async def get_file_watcher_status():
    """
    File watcher status - directory being watched, ingestion counts.
    """

    watcher: FileWatcher = app.state.file_watcher
    return watcher.stats


@app.get("/api/v1/perception/scheduler")
async def get_scheduler_status():
    """
    Perception scheduler status - registered tasks, run counts, intervals.
    """

    sched: PerceptionScheduler = app.state.scheduler
    return sched.stats


@app.post("/api/v1/perception/scheduler/register")
async def register_scheduler_task(body: dict[str, Any]):
    """
    Dynamically register a built-in named scheduler task at runtime.

    Body: {name: str, task: str}

    Currently supported built-in tasks:
    - "self_clock" - injects a periodic time-awareness percept (every 300s)
    """
    from systems.fovea.types import InputChannel

    name = body.get("name")
    task_key = body.get("task")
    if not name or not task_key:
        return {"error": "name and task required"}

    sched: PerceptionScheduler = app.state.scheduler

    # ── Built-in tasks ────────────────────────────────────────────
    if task_key == "self_clock":
        import datetime as _dt

        async def _self_clock_fn() -> str:
            now = _dt.datetime.now(_dt.UTC)
            return (
                f"The current time is {now.strftime('%Y-%m-%d %H:%M UTC')}. "
                f"I am aware of the passage of time."
            )

        sched.register(
            name=name,
            interval_seconds=300,
            channel=InputChannel.SYSTEM_EVENT,
            fn=_self_clock_fn,
            metadata={"built_in": "self_clock"},
        )
        return {"registered": True, "name": name, "task": task_key}

    return {"error": f"Unknown built-in task: {task_key!r}"}


# ─── Phase 3: Frontend Data Integration Endpoints ─────────────────


@app.get("/api/v1/axon/outcomes")
async def get_axon_outcomes(limit: int = 20):
    """
    Recent Axon execution outcomes for the Decisions page.

    Returns the last N action verdicts (newest first): intent_id, success,
    status, action types executed, duration_ms, and world_state_changes.
    This is the observable footprint of the Nova→Equor→Axon pipeline.
    """

    axon: AxonService = app.state.axon
    outcomes = axon.recent_outcomes[:limit]

    return {
        "outcomes": [
            {
                "execution_id": o.execution_id,
                "intent_id": o.intent_id,
                "success": o.success,
                "partial": o.partial,
                "status": o.status.value,
                "failure_reason": o.failure_reason or None,
                "duration_ms": o.duration_ms,
                "steps": [
                    {
                        "action_type": s.action_type,
                        "description": s.description[:120],
                        "success": s.result.success,
                        "duration_ms": s.duration_ms,
                    }
                    for s in o.step_outcomes
                ],
                "world_state_changes": o.world_state_changes[:5],
                "new_observations": o.new_observations[:3],
            }
            for o in outcomes
        ],
        "total": axon._total_executions,
        "successful": axon._successful_executions,
        "failed": axon._failed_executions,
    }


@app.get("/api/v1/atune/workspace-detail")
async def get_workspace_detail():
    """
    Detailed workspace state - ignited percepts with content, salience, and channel.

    Used by the /perception page WorkspaceStream component to show what the
    organism is currently considering at this cognitive moment.
    """
    atune: AtuneService = app.state.atune

    # Collect ignited percepts from the workspace buffer
    workspace_items: list[dict[str, Any]] = []
    try:
        # Access the workspace buffer directly (ring buffer of recent broadcasts)
        broadcasts = getattr(atune, "_workspace_broadcasts", []) or []
        for b in list(broadcasts)[-20:]:  # Last 20
            workspace_items.append({
                "broadcast_id": getattr(b, "broadcast_id", str(id(b))),
                "content": getattr(b, "content", "")[:200] if hasattr(b, "content") else "",
                "salience": round(getattr(b, "salience", 0.0), 4),
                "channel": getattr(b, "channel", "unknown"),
                "timestamp": b.timestamp.isoformat() if hasattr(b, "timestamp") and b.timestamp else None,
                "source": getattr(b, "source", "unknown"),
            })
    except Exception:
        pass

    # If workspace_broadcasts not available, fall back to recent_broadcasts from workspace
    if not workspace_items:
        try:
            ws_snapshot = atune.workspace_snapshot
            for b in ws_snapshot.recent_broadcasts:
                workspace_items.append({
                    "broadcast_id": b.broadcast_id,
                    "content": "",
                    "salience": round(b.salience, 4),
                    "channel": "unknown",
                    "timestamp": b.timestamp.isoformat() if hasattr(b, "timestamp") else None,
                    "source": "workspace",
                })
        except Exception:
            pass

    affect = atune.current_affect
    return {
        "cycle_count": atune.cycle_count,
        "dynamic_threshold": round(atune.workspace_threshold, 4),
        "meta_attention_mode": atune.meta_attention_mode,
        "workspace_items": workspace_items,
        "affect": {
            "valence": round(affect.valence, 4),
            "arousal": round(affect.arousal, 4),
            "curiosity": round(affect.curiosity, 4),
            "coherence_stress": round(affect.coherence_stress, 4),
        },
    }


# ─── Atune Extended Observability Endpoints ──────────────────────


@app.get("/api/v1/atune/salience-heads")
async def get_salience_heads():
    """
    Salience is computed by Fovea via prediction error decomposition
    (per-dimension precision-weighted errors). This endpoint returns
    Fovea's error dimension weights and the current attention mode.
    """
    fovea = getattr(app.state, "fovea", None)
    if fovea is None:
        return {"error": "fovea_not_wired", "heads": [], "meta_attention_mode": "fovea_driven"}

    metrics = fovea.get_metrics() if fovea is not None else None

    from systems.fovea.types import DEFAULT_ERROR_WEIGHTS
    dimension_weights = {k: round(v, 4) for k, v in DEFAULT_ERROR_WEIGHTS.items()}

    return {
        "note": "Salience is now prediction-error magnitude (Fovea). Seven-head engine removed.",
        "error_dimension_weights": dimension_weights,
        "meta_attention_mode": "fovea_driven",
        "fovea_metrics": {
            "errors_processed": metrics.errors_processed if metrics else 0,
            "workspace_ignitions": metrics.workspace_ignitions if metrics else 0,
            "mean_salience": round(metrics.mean_salience, 4) if metrics else 0.0,
            "mean_precision": round(metrics.mean_precision, 4) if metrics else 0.0,
        } if metrics else None,
    }


@app.get("/api/v1/atune/momentum")
async def get_salience_momentum():
    """
    Momentum tracking has moved to Fovea's HabituationEngine.
    Returns Fovea's habituation statistics instead.
    """
    fovea = getattr(app.state, "fovea", None)
    atune: AtuneService = app.state.atune
    if fovea is None:
        return {"note": "fovea_not_wired", "cycle_count": atune.cycle_count}

    metrics = fovea.get_metrics()
    return {
        "note": "Per-head momentum replaced by Fovea HabituationEngine.",
        "habituation": {
            "habituated_count": metrics.habituated_count,
            "dishabituated_count": metrics.dishabituated_count,
            "active_entries": metrics.habituation_entries,
        },
        "cycle_count": atune.cycle_count,
    }


@app.get("/api/v1/atune/bias-model")
async def get_salience_bias():
    """
    Perceptual learning bias is now owned by Fovea's AttentionWeightLearner.
    Returns Fovea's learned weight adjustments per error dimension instead.
    """
    fovea = getattr(app.state, "fovea", None)
    if fovea is None:
        return {"note": "fovea_not_wired", "biases": {}}

    learner = fovea.weight_learner

    weights = getattr(learner, "current_weights", {})
    return {
        "note": "Bias model replaced by Fovea AttentionWeightLearner.",
        "learned_dimension_weights": {k: round(v, 4) for k, v in weights.items()},
        "total_dimensions_tracked": len(weights),
    }


@app.get("/api/v1/atune/mood")
async def get_mood_state():
    """
    Affect state is now owned by Soma (allostatic regulation).
    Returns the current interoceptive signal from Soma.
    """
    atune: AtuneService = app.state.atune
    affect = atune.current_affect  # reads from Soma via _get_affect_for_broadcast

    return {
        "note": "Affect state now owned by Soma. Reading via Atune compatibility accessor.",
        "current_affect": {
            "valence": round(getattr(affect, "valence", 0.0), 4),
            "arousal": round(getattr(affect, "arousal", 0.0), 4),
            "dominance": round(getattr(affect, "dominance", 0.0), 4),
            "curiosity": round(getattr(affect, "curiosity", 0.0), 4),
            "care_activation": round(getattr(affect, "care_activation", 0.0), 4),
            "coherence_stress": round(getattr(affect, "coherence_stress", 0.0), 4),
            "timestamp": affect.timestamp.isoformat() if hasattr(affect, "timestamp") and affect.timestamp else None,
        },
    }


@app.get("/api/v1/atune/config")
async def get_atune_config():
    """
    Atune configuration - thresholds, buffer sizes, cache refresh intervals.
    """
    atune: AtuneService = app.state.atune
    cfg = atune._config
    ws = atune._workspace

    return {
        "ignition_threshold": "(dynamic - set by Fovea error distribution percentile)",
        "workspace_buffer_size": cfg.workspace_buffer_size,
        "spontaneous_recall_base_probability": cfg.spontaneous_recall_base_probability,
        "max_percept_queue_size": cfg.max_percept_queue_size,
        "cache_refresh_cycles": {
            "identity": cfg.cache_identity_refresh_cycles,
            "risk": cfg.cache_risk_refresh_cycles,
            "vocab": cfg.cache_vocab_refresh_cycles,
            "alerts": cfg.cache_alert_refresh_cycles,
        },
        "workspace": {
            "dynamic_threshold": round(ws.dynamic_threshold, 4),
            "cycle_count": ws.cycle_count,
            "habituation_sources": len(ws._habituation),
            "percept_queue_size": len(ws._percept_queue),
            "contribution_queue_size": len(ws._contribution_queue),
        },
    }


@app.get("/api/v1/atune/broadcasts")
async def get_recent_broadcasts(limit: int = 20):
    """
    Recent workspace broadcasts with full salience vector breakdown.

    Returns per-head scores, composite, meta-attention mode, affect snapshot,
    and source channel for each ignited percept.
    """
    atune: AtuneService = app.state.atune
    broadcasts = atune.recent_broadcasts[-limit:]

    result = []
    for b in reversed(broadcasts):
        per_head_scores: dict[str, float] = {}
        momentum_info: dict[str, dict] = {}
        composite = 0.0
        threat_traj = "unknown"

        if b.salience is not None:
            sv = b.salience
            composite = round(sv.composite, 4)
            per_head_scores = {k: round(v, 4) for k, v in (sv.scores or {}).items()}
            threat_traj = sv.threat_trajectory.value if hasattr(sv.threat_trajectory, "value") else str(sv.threat_trajectory)
            for head_name, hm in (sv.momentum or {}).items():
                momentum_info[head_name] = {
                    "trajectory": hm.trajectory.value if hasattr(hm.trajectory, "value") else str(hm.trajectory),
                    "bonus": round(hm.momentum_bonus, 4),
                }

        affect_snap: dict = {}
        if b.affect is not None:
            affect_snap = {
                "valence": round(b.affect.valence, 3),
                "arousal": round(b.affect.arousal, 3),
                "curiosity": round(b.affect.curiosity, 3),
            }

        content_str = ""
        content_raw = b.content
        if isinstance(content_raw, str):
            content_str = content_raw[:300]
        elif hasattr(content_raw, "content"):
            content_str = str(content_raw.content)[:300]
        elif hasattr(content_raw, "data"):
            content_str = str(content_raw.data)[:300]
        else:
            content_str = str(content_raw)[:300]

        result.append({
            "broadcast_id": b.broadcast_id,
            "timestamp": b.timestamp.isoformat() if b.timestamp else None,
            "source": b.source,
            "composite_salience": composite,
            "per_head_scores": per_head_scores,
            "momentum": momentum_info,
            "threat_trajectory": threat_traj,
            "affect_snapshot": affect_snap,
            "precision": round(b.precision, 4) if b.precision is not None else None,
            "content_preview": content_str,
        })

    return {
        "broadcasts": result,
        "total_returned": len(result),
        "cycle_count": atune.cycle_count,
        "dynamic_threshold": round(atune.workspace_threshold, 4),
    }


# ─── Phase 12: Thymos (Immune System) Endpoints ──────────────────


@app.get("/api/v1/thymos/health")
async def get_thymos_health():
    """Thymos immune system health and observability metrics."""
    thymos: ThymosService = app.state.thymos
    return await thymos.health()


@app.get("/api/v1/thymos/incidents")
async def get_thymos_incidents(limit: int = 50):
    """Recent incidents from the immune system."""
    thymos: ThymosService = app.state.thymos
    incidents = list(thymos._incident_buffer)[-limit:]
    return [
        {
            "id": i.id,
            "timestamp": i.timestamp.isoformat(),
            "source_system": i.source_system,
            "incident_class": i.incident_class.value,
            "severity": i.severity.value,
            "error_type": i.error_type,
            "error_message": i.error_message[:200],
            "repair_status": i.repair_status.value,
            "repair_tier": i.repair_tier.name if i.repair_tier else None,
            "repair_successful": i.repair_successful,
            "resolution_time_ms": i.resolution_time_ms,
            "root_cause": i.root_cause_hypothesis,
            "antibody_id": i.antibody_id,
        }
        for i in reversed(incidents)
    ]


@app.get("/api/v1/thymos/antibodies")
async def get_thymos_antibodies():
    """All active antibodies in the immune memory."""
    thymos: ThymosService = app.state.thymos
    if thymos._antibody_library is None:
        return []
    return [
        {
            "id": a.id,
            "fingerprint": a.fingerprint,
            "source_system": a.source_system,
            "incident_class": a.incident_class.value,
            "repair_tier": a.repair_tier.name,
            "effectiveness": round(a.effectiveness, 3),
            "application_count": a.application_count,
            "success_count": a.success_count,
            "failure_count": a.failure_count,
            "root_cause": a.root_cause_description,
            "created_at": a.created_at.isoformat(),
            "last_applied": a.last_applied.isoformat() if a.last_applied else None,
            "retired": a.retired,
            "generation": a.generation,
        }
        for a in thymos._antibody_library._all.values()
    ]


@app.get("/api/v1/thymos/stats")
async def get_thymos_stats():
    """Thymos aggregate stats for monitoring."""
    thymos: ThymosService = app.state.thymos
    return thymos.stats


@app.get("/api/v1/thymos/repairs")
async def get_thymos_repairs(limit: int = 50):
    """Recent repairs and their outcomes."""
    thymos: ThymosService = app.state.thymos
    incidents = list(thymos._incident_buffer)[-limit:]
    repairs = [
        {
            "incident_id": i.id,
            "timestamp": i.timestamp.isoformat(),
            "source_system": i.source_system,
            "repair_tier": i.repair_tier.name if i.repair_tier else None,
            "repair_status": i.repair_status.value,
            "repair_successful": i.repair_successful,
            "resolution_time_ms": i.resolution_time_ms,
            "incident_class": i.incident_class.value,
            "severity": i.severity.value,
            "antibody_id": i.antibody_id,
        }
        for i in reversed(incidents)
        if i.repair_tier is not None
    ]
    return repairs


@app.get("/api/v1/thymos/homeostasis")
async def get_thymos_homeostasis():
    """Current homeostasis metrics status."""
    thymos: ThymosService = app.state.thymos
    health_data = await thymos.health()
    return {
        "metrics_in_range": health_data.get("metrics_in_range", 0),
        "homeostatic_adjustments": health_data.get("homeostatic_adjustments", 0),
        "healing_mode": health_data.get("healing_mode", "normal"),
        "storm_activations": health_data.get("storm_activations", 0),
    }


# ─── Oneiros (Dream Engine) ───────────────────────────────────────


@app.get("/api/v1/oneiros/health")
async def get_oneiros_health():
    oneiros: OneirosService = app.state.oneiros
    return await oneiros.health()


@app.get("/api/v1/oneiros/stats")
async def get_oneiros_stats():
    oneiros: OneirosService = app.state.oneiros
    return oneiros.stats


@app.get("/api/v1/oneiros/dreams")
async def get_oneiros_dreams(limit: int = 50):
    oneiros: OneirosService = app.state.oneiros
    dreams = list(oneiros._journal._dream_buffer)[-limit:]
    return [
        {
            "id": d.id,
            "dream_type": d.dream_type.value,
            "coherence_score": round(d.coherence_score, 3),
            "coherence_class": d.coherence_class.value,
            "bridge_narrative": d.bridge_narrative,
            "affect_valence": round(d.affect_valence, 3),
            "affect_arousal": round(d.affect_arousal, 3),
            "themes": d.themes,
            "summary": d.summary,
            "timestamp": d.timestamp.isoformat(),
        }
        for d in reversed(dreams)
    ]


@app.get("/api/v1/oneiros/insights")
async def get_oneiros_insights(limit: int = 50):
    oneiros: OneirosService = app.state.oneiros
    all_insights = list(oneiros._journal._all_insights.values())
    sorted_insights = sorted(all_insights, key=lambda i: i.created_at, reverse=True)[:limit]
    return [
        {
            "id": i.id,
            "insight_text": i.insight_text,
            "coherence_score": round(i.coherence_score, 3),
            "domain": i.domain,
            "status": i.status.value,
            "wake_applications": i.wake_applications,
            "created_at": i.created_at.isoformat(),
        }
        for i in sorted_insights
    ]


@app.get("/api/v1/oneiros/sleep-cycles")
async def get_oneiros_sleep_cycles(limit: int = 20):
    oneiros: OneirosService = app.state.oneiros
    cycles = list(oneiros._recent_cycles)[-limit:]
    return [
        {
            "id": c.id,
            "started_at": c.started_at.isoformat(),
            "completed_at": c.completed_at.isoformat() if c.completed_at else None,
            "quality": c.quality.value,
            "episodes_replayed": c.episodes_replayed,
            "dreams_generated": c.dreams_generated,
            "insights_discovered": c.insights_discovered,
            "pressure_before": round(c.pressure_before, 3),
            "pressure_after": round(c.pressure_after, 3),
        }
        for c in reversed(cycles)
    ]


@app.get("/api/v1/oneiros/circadian")
async def get_oneiros_circadian():
    """Circadian clock state - pressure breakdown by source and wake degradation detail."""
    oneiros: OneirosService = app.state.oneiros
    clock = oneiros._clock
    pressure = clock.pressure
    degradation = clock.degradation
    phase = clock.phase

    max_cycles = getattr(clock, "_max_wake_cycles", 528000)
    affect_cap = getattr(clock, "_affect_capacity", 50.0)
    episode_cap = getattr(clock, "_episode_capacity", 500)
    hyp_cap = getattr(clock, "_hypothesis_capacity", 50)
    w_cycles = getattr(clock, "_w_cycles", 0.40)
    w_affect = getattr(clock, "_w_affect", 0.25)
    w_episodes = getattr(clock, "_w_episodes", 0.20)
    w_hypotheses = getattr(clock, "_w_hypotheses", 0.15)

    cycle_contrib = w_cycles * min(1.0, pressure.cycles_since_sleep / max(max_cycles, 1))
    affect_contrib = w_affect * min(1.0, pressure.unprocessed_affect_residue / max(affect_cap, 0.01))
    episode_contrib = w_episodes * min(1.0, pressure.unconsolidated_episode_count / max(episode_cap, 1))
    hyp_contrib = w_hypotheses * min(1.0, pressure.hypothesis_backlog / max(hyp_cap, 1))

    return {
        "pressure": {
            "composite": round(pressure.composite_pressure, 4),
            "threshold": round(pressure.threshold, 4),
            "critical_threshold": round(pressure.critical_threshold, 4),
            "contributions": {
                "cycles": round(cycle_contrib, 4),
                "affect": round(affect_contrib, 4),
                "episodes": round(episode_contrib, 4),
                "hypotheses": round(hyp_contrib, 4),
            },
            "raw_counts": {
                "cycles_since_sleep": pressure.cycles_since_sleep,
                "unprocessed_affect_residue": round(pressure.unprocessed_affect_residue, 3),
                "unconsolidated_episodes": pressure.unconsolidated_episode_count,
                "hypothesis_backlog": pressure.hypothesis_backlog,
            },
        },
        "degradation": {
            "composite_impairment": round(degradation.composite_impairment, 4),
            "salience_noise": round(degradation.salience_noise, 4),
            "efe_precision_loss": round(degradation.efe_precision_loss, 4),
            "expression_flatness": round(degradation.expression_flatness, 4),
            "learning_rate_reduction": round(degradation.learning_rate_reduction, 4),
        },
        "phase": {
            "current_stage": oneiros._stage_controller.current_stage.value,
            "total_cycles_completed": phase.total_cycles_completed,
            "wake_duration_target_s": phase.wake_duration_target_s,
            "sleep_duration_target_s": phase.sleep_duration_target_s,
        },
        "last_sleep_completed": pressure.last_sleep_completed.isoformat() if pressure.last_sleep_completed else None,
    }


@app.get("/api/v1/oneiros/worker-metrics")
async def get_oneiros_worker_metrics():
    """Last-cycle metrics from v2 sleep stages (Slow Wave, REM, Lucid Dreaming)."""
    oneiros: OneirosService = app.state.oneiros
    recent_cycles = list(oneiros._recent_cycles)
    if not recent_cycles:
        return {"slow_wave": None, "rem": None, "lucid": None, "current_cycle": None}

    last = recent_cycles[-1]
    current = oneiros._current_cycle

    return {
        "slow_wave": {
            "memories_processed": last.episodes_replayed,
            "semantic_nodes_created": last.semantic_nodes_created,
            "schemas_created": last.beliefs_compressed,
            "hypotheses_retired": last.hypotheses_pruned,
            "hypotheses_confirmed": last.hypotheses_promoted,
        },
        "rem": {
            "scenarios_generated": last.dreams_generated,
            "analogies_found": last.insights_discovered,
        },
        "lucid": {
            "mutations_tested": last.lucid_proposals_submitted,
            "mutations_applied": last.lucid_proposals_accepted,
            "mutations_rejected": last.lucid_proposals_rejected,
        },
        "current_cycle": {
            "id": current.id,
            "started_at": current.started_at.isoformat(),
            "quality": current.quality.value if current.quality else None,
            "interrupted": current.interrupted,
        } if current else None,
    }


@app.get("/api/v1/oneiros/insight-lifecycle")
async def get_oneiros_insight_lifecycle():
    """Insight status distribution and domain breakdown."""
    oneiros: OneirosService = app.state.oneiros
    all_insights = list(oneiros._journal._all_insights.values())

    by_status: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    top_insights: list[dict] = []

    for i in all_insights:
        status_key = i.status.value
        by_status[status_key] = by_status.get(status_key, 0) + 1
        domain_key = i.domain or "unknown"
        by_domain[domain_key] = by_domain.get(domain_key, 0) + 1

    sorted_insights = sorted(all_insights, key=lambda x: x.wake_applications, reverse=True)[:10]
    for i in sorted_insights:
        top_insights.append({
            "id": i.id,
            "insight_text": i.insight_text,
            "domain": i.domain,
            "status": i.status.value,
            "coherence_score": round(i.coherence_score, 3),
            "wake_applications": i.wake_applications,
            "created_at": i.created_at.isoformat(),
        })

    return {
        "total": len(all_insights),
        "by_status": by_status,
        "by_domain": dict(sorted(by_domain.items(), key=lambda x: -x[1])),
        "top_applied": top_insights,
        "lifetime": {
            "validated": oneiros._insights_validated,
            "invalidated": oneiros._insights_invalidated,
            "integrated": oneiros._insights_integrated,
        },
    }


# ─── Phase 11: Federation Endpoints ─────────────────────────────


@app.get("/api/v1/federation/identity")
async def get_federation_identity():
    """
    This instance's public identity card.

    Used by other instances during link establishment to verify
    identity and check compatibility.
    """
    federation: FederationService = app.state.federation
    card = federation.identity_card
    if card is None:
        return {"status": "disabled", "message": "Federation is not enabled"}
    return card.model_dump(mode="json")


@app.get("/api/v1/federation/links")
async def get_federation_links():
    """List all federation links with trust levels and status."""
    federation: FederationService = app.state.federation
    links = federation.active_links
    return {
        "links": [
            {
                "id": lnk.id,
                "remote_instance_id": lnk.remote_instance_id,
                "remote_name": lnk.remote_name,
                "remote_endpoint": lnk.remote_endpoint,
                "trust_level": lnk.trust_level.name,
                "trust_score": round(lnk.trust_score, 2),
                "status": lnk.status.value,
                "established_at": lnk.established_at.isoformat(),
                "last_communication": lnk.last_communication.isoformat()
                if lnk.last_communication else None,
                "shared_knowledge_count": lnk.shared_knowledge_count,
                "received_knowledge_count": lnk.received_knowledge_count,
                "successful_interactions": lnk.successful_interactions,
                "failed_interactions": lnk.failed_interactions,
            }
            for lnk in links
        ],
        "total_active": len(links),
    }


@app.post("/api/v1/federation/links")
async def establish_federation_link(body: dict[str, Any]):
    """
    Establish a new federation link with a remote instance.

    Body: {endpoint: str}

    The full link establishment protocol runs:
      1. Fetch remote identity card
      2. Verify identity (Ed25519 + certificate fingerprint)
      3. Equor constitutional review
      4. Create link at NONE trust
      5. Open mTLS channel

    Performance target: ≤3000ms
    """
    endpoint = body.get("endpoint", "")
    if not endpoint:
        return {"error": "No endpoint provided"}

    federation: FederationService = app.state.federation
    return await federation.establish_link(endpoint)


@app.delete("/api/v1/federation/links/{link_id}")
async def withdraw_federation_link(link_id: str):
    """
    Withdraw from a federation link.

    Withdrawal is always free - any instance can disconnect at any
    time with no penalty.
    """
    federation: FederationService = app.state.federation
    return await federation.withdraw_link(link_id)


@app.post("/api/v1/federation/knowledge/request")
async def handle_federation_knowledge_request(body: dict[str, Any]):
    """
    Handle an inbound knowledge request from a federated instance.

    Body: {requesting_instance_id, knowledge_type, query?, domain?, max_results?}

    The full knowledge exchange protocol runs:
      1. Trust level check (is this knowledge type permitted?)
      2. Equor constitutional review
      3. Knowledge retrieval from memory
      4. Privacy filter (PII removal, consent enforcement)
      5. Return filtered knowledge

    Performance target: ≤2000ms
    """
    from primitives.federation import KnowledgeRequest, KnowledgeType

    try:
        knowledge_type = KnowledgeType(body.get("knowledge_type", ""))
    except ValueError:
        return {"error": f"Unknown knowledge type: {body.get('knowledge_type')}"}

    request = KnowledgeRequest(
        requesting_instance_id=body.get("requesting_instance_id", ""),
        knowledge_type=knowledge_type,
        query=body.get("query", ""),
        domain=body.get("domain", ""),
        max_results=body.get("max_results", 10),
    )

    federation: FederationService = app.state.federation
    response = await federation.handle_knowledge_request(request)
    return response.model_dump(mode="json")


@app.post("/api/v1/federation/knowledge/share")
async def request_knowledge_from_remote(body: dict[str, Any]):
    """
    Request knowledge from a linked remote instance.

    Body: {link_id, knowledge_type, query?, max_results?}
    """
    from primitives.federation import KnowledgeType

    link_id = body.get("link_id", "")
    if not link_id:
        return {"error": "No link_id provided"}

    try:
        knowledge_type = KnowledgeType(body.get("knowledge_type", ""))
    except ValueError:
        return {"error": f"Unknown knowledge type: {body.get('knowledge_type')}"}

    federation: FederationService = app.state.federation
    response = await federation.request_knowledge(
        link_id=link_id,
        knowledge_type=knowledge_type,
        query=body.get("query", ""),
        max_results=body.get("max_results", 10),
    )

    if response is None:
        return {"error": "Failed to request knowledge (link not found or channel error)"}

    return response.model_dump(mode="json")


@app.post("/api/v1/federation/assistance/request")
async def handle_federation_assistance_request(body: dict[str, Any]):
    """
    Handle an inbound assistance request from a federated instance.

    Body: {requesting_instance_id, description, knowledge_domain?, urgency?, reciprocity_offer?}

    Requires COLLEAGUE trust level or higher.
    """
    from primitives.federation import AssistanceRequest

    request = AssistanceRequest(
        requesting_instance_id=body.get("requesting_instance_id", ""),
        description=body.get("description", ""),
        knowledge_domain=body.get("knowledge_domain", ""),
        urgency=body.get("urgency", 0.5),
        reciprocity_offer=body.get("reciprocity_offer"),
    )

    federation: FederationService = app.state.federation
    response = await federation.handle_assistance_request(request)
    return response.model_dump(mode="json")


@app.post("/api/v1/federation/assistance/respond")
async def request_assistance_from_remote(body: dict[str, Any]):
    """
    Request assistance from a linked remote instance.

    Body: {link_id, description, knowledge_domain?, urgency?}
    """
    link_id = body.get("link_id", "")
    if not link_id:
        return {"error": "No link_id provided"}

    federation: FederationService = app.state.federation
    response = await federation.request_assistance(
        link_id=link_id,
        description=body.get("description", ""),
        knowledge_domain=body.get("knowledge_domain", ""),
        urgency=body.get("urgency", 0.5),
    )

    if response is None:
        return {"error": "Failed to request assistance (link not found or channel error)"}

    return response.model_dump(mode="json")


@app.get("/api/v1/federation/stats")
async def get_federation_stats():
    """Full federation system statistics."""
    federation: FederationService = app.state.federation
    return federation.stats


@app.get("/api/v1/federation/trust/{link_id}")
async def get_federation_trust(link_id: str):
    """Get trust details for a specific federation link."""
    federation: FederationService = app.state.federation
    link = federation.get_link(link_id)
    if link is None:
        return {"error": "Link not found"}

    from primitives.federation import SHARING_PERMISSIONS

    return {
        "link_id": link.id,
        "remote_instance_id": link.remote_instance_id,
        "remote_name": link.remote_name,
        "trust_level": link.trust_level.name,
        "trust_score": round(link.trust_score, 2),
        "permitted_knowledge_types": [
            kt.value for kt in SHARING_PERMISSIONS.get(link.trust_level, [])
        ],
        "can_coordinate": link.trust_level.value >= 2,  # COLLEAGUE+
        "successful_interactions": link.successful_interactions,
        "failed_interactions": link.failed_interactions,
        "violation_count": link.violation_count,
    }


@app.post("/api/v1/federation/threat-advisory")
async def receive_threat_advisory(body: dict[str, Any]):
    """
    Receive a threat advisory from a federated peer.

    Layer 4 of the Economic Immune System: trust-gated threat
    intelligence sharing between instances.

    Body: ThreatAdvisory payload (source_instance_id, threat_type,
          severity, affected_protocols, affected_addresses, etc.)
    """
    from primitives.federation import ThreatAdvisory as ThreatAdvisoryModel

    try:
        advisory = ThreatAdvisoryModel.model_validate(body)
    except Exception as exc:
        return {"accepted": False, "reason": f"Invalid advisory format: {exc}"}

    federation: FederationService = app.state.federation
    accepted, reason = federation.handle_threat_advisory(
        advisory, advisory.source_instance_id
    )

    return {"accepted": accepted, "reason": reason}


@app.post("/api/v1/federation/handshake")
async def handle_federation_handshake(body: dict[str, Any]):
    """
    Federation handshake endpoint (responder side, Phases 2-3).

    Receives a HandshakeRequest from a remote instance, verifies
    identity, certificate, and constitutional alignment, then returns
    a HandshakeResponse with this instance's credentials and a signed
    challenge.

    This endpoint is public (no API key required) - authentication
    happens via the handshake protocol itself.

    Body: HandshakeRequest payload (initiator identity card, certificate,
          constitutional hash, nonce, capabilities)
    """
    federation: FederationService = app.state.federation
    return await federation.handle_handshake(body)


@app.post("/api/v1/federation/handshake/confirm")
async def handle_federation_handshake_confirm(body: dict[str, Any]):
    """
    Federation handshake confirmation (responder side, Phase 4).

    The initiator sends this after verifying our handshake response.
    Contains the initiator's signature over our nonce, completing
    mutual authentication. On success, the responder creates its
    side of the federation link.

    Body: HandshakeConfirmation (handshake_id, initiator_instance_id,
          responder_nonce_signature)
    """
    federation: FederationService = app.state.federation
    return await federation.handle_handshake_confirmation(body)


@app.get("/api/v1/federation/interactions")
async def get_federation_interactions(limit: int = 50) -> dict[str, Any]:
    """Recent federation interaction history (ring buffer, newest first)."""
    federation: FederationService = app.state.federation
    safe_limit = max(1, min(limit, 200))
    history = list(reversed(federation._interaction_history[-safe_limit:]))
    return {
        "interactions": [
            {
                "id": str(i.id),
                "link_id": str(i.link_id),
                "interaction_type": i.interaction_type.value
                if hasattr(i.interaction_type, "value")
                else str(i.interaction_type),
                "direction": i.direction.value
                if hasattr(i.direction, "value")
                else str(i.direction),
                "outcome": i.outcome.value
                if hasattr(i.outcome, "value")
                else str(i.outcome),
                "violation_type": i.violation_type.value
                if i.violation_type and hasattr(i.violation_type, "value")
                else (str(i.violation_type) if i.violation_type else None),
                "trust_value_change": round(i.trust_value_change, 3),
                "latency_ms": i.latency_ms,
                "timestamp": i.timestamp.isoformat(),
                "metadata": i.metadata or {},
            }
            for i in history
        ],
        "total": len(federation._interaction_history),
    }


@app.post("/api/v1/federation/iiep/push")
async def iiep_push_exchange(body: dict[str, Any]) -> dict[str, Any]:
    """Push IIEP knowledge payloads to a linked peer."""
    from primitives.federation import ExchangePayloadKind

    link_id: str = body.get("link_id", "")
    raw_kinds: list[str] = body.get("payload_kinds", [])
    max_items: int = int(body.get("max_items_per_kind", 5))

    if not link_id:
        return {"error": "link_id is required"}
    if not raw_kinds:
        return {"error": "payload_kinds is required"}

    kinds: list[ExchangePayloadKind] = []
    for k in raw_kinds:
        try:
            kinds.append(ExchangePayloadKind(k))
        except ValueError:
            return {"error": f"Unknown payload kind: {k}"}

    federation: FederationService = app.state.federation
    link = federation.get_link(link_id)
    if link is None:
        return {"error": f"Link not found: {link_id}"}

    from systems.federation.exchange import ExchangeProtocol

    exchange: ExchangeProtocol | None = federation._exchange
    if exchange is None:
        return {"error": "Exchange protocol not initialized"}

    payloads = []
    for kind in kinds:
        if kind == ExchangePayloadKind.HYPOTHESIS and federation._evo:
            payloads.extend(
                await ExchangeProtocol.collect_hypotheses(
                    federation._evo, federation._instance_id or "", max_items=max_items
                )
            )
        elif kind == ExchangePayloadKind.PROCEDURE and federation._evo:
            payloads.extend(
                await ExchangeProtocol.collect_procedures(
                    federation._evo, federation._instance_id or "", max_items=max_items
                )
            )
        elif kind == ExchangePayloadKind.MUTATION_PATTERN and federation._simula:
            payloads.extend(
                await ExchangeProtocol.collect_mutation_patterns(
                    federation._simula, federation._instance_id or "", max_items=max_items
                )
            )
        elif kind == ExchangePayloadKind.ECONOMIC_INTEL and federation._oikos:
            payloads.extend(
                await ExchangeProtocol.collect_economic_intel(
                    federation._oikos, federation._instance_id or "", max_items=max_items
                )
            )

    if not payloads:
        return {"error": "No payloads available for the requested kinds"}

    result = await federation.push_exchange(link, payloads)
    if result is None:
        return {"error": "Push exchange failed (channel error or no envelope produced)"}

    verdicts = [
        {
            "payload_index": v.payload_index,
            "verdict": v.verdict.value if hasattr(v.verdict, "value") else str(v.verdict),
            "reason": v.reason,
        }
        for v in result.verdicts
    ]
    return {
        "payloads_sent": len(payloads),
        "verdicts": verdicts,
        "accepted": sum(1 for v in verdicts if v["verdict"] == "ACCEPTED"),
        "rejected": sum(1 for v in verdicts if v["verdict"] == "REJECTED"),
        "quarantined": sum(1 for v in verdicts if v["verdict"] == "QUARANTINED"),
    }


@app.post("/api/v1/federation/threat-advisory/broadcast")
async def broadcast_threat_advisory_endpoint(body: dict[str, Any]) -> dict[str, Any]:
    """Broadcast a threat advisory to all ACQUAINTANCE+ peers."""
    from primitives.federation import ThreatAdvisory as ThreatAdvisoryModel

    federation: FederationService = app.state.federation
    if not federation._identity:
        return {"error": "Federation identity not initialized"}

    try:
        advisory = ThreatAdvisoryModel(
            source_instance_id=federation._instance_id or "",
            threat_type=body.get("threat_type", ""),
            severity=float(body.get("severity", 0.5)),
            description=body.get("description", ""),
            affected_protocols=body.get("affected_protocols", []),
            affected_addresses=body.get("affected_addresses", []),
            chain_id=body.get("chain_id"),
            evidence=body.get("evidence"),
            recommended_action=body.get("recommended_action"),
        )
    except Exception as exc:
        return {"error": f"Invalid advisory: {exc}"}

    delivery = await federation.broadcast_threat_advisory(advisory, federation.active_links)
    return {
        "broadcast_to": len(delivery),
        "delivered": sum(1 for v in delivery.values() if v),
        "failed": sum(1 for v in delivery.values() if not v),
        "results": {k: v for k, v in delivery.items()},
    }


# ── Soma (Interoceptive Predictive Substrate) ──────────────────────


@app.get("/api/v1/soma/health")
async def get_soma_health():
    """Soma system health report."""
    soma: SomaService = app.state.soma
    return await soma.health()


@app.get("/api/v1/soma/state")
async def get_soma_state():
    """Current interoceptive state - the organism's felt sense of its own viability."""
    soma: SomaService = app.state.soma
    state = soma.get_current_state()
    if state is None:
        return {"dimensions": [], "overall_urgency": 0.0, "status": "no_state"}
    signal = soma.get_current_signal()
    moment_errors = state.errors.get("moment", {})
    precision = signal.precision_weights if signal else {}
    dimensions = []
    for d, sensed_val in state.sensed.items():
        err = moment_errors.get(d, 0.0)
        err_rate = state.error_rates.get(d, 0.0)
        prec = precision.get(d, 0.0) if precision else 0.0
        sp = state.setpoints.get(d, sensed_val)
        urgency_val = min(1.0, abs(err) * abs(err_rate)) if err_rate != 0 else abs(err) * 0.5
        dimensions.append({
            "name": d.value,
            "sensed": round(sensed_val, 4),
            "setpoint": round(sp, 4),
            "error": round(err, 4),
            "error_rate": round(err_rate, 4),
            "urgency": round(urgency_val, 4),
            "temporal_dissonance": round(state.temporal_dissonance.get(d, 0.0), 4),
            "precision": round(prec, 4),
        })
    overall_urgency = state.urgency
    if overall_urgency >= 0.8:
        urgency_classification = "critical"
    elif overall_urgency >= 0.5:
        urgency_classification = "warning"
    else:
        urgency_classification = "nominal"
    return {
        "dimensions": dimensions,
        "overall_urgency": round(overall_urgency, 4),
        "dominant_error": state.dominant_error.value,
        "max_error_magnitude": round(state.max_error_magnitude, 4),
        "timestamp": state.timestamp.isoformat(),
        "urgency_classification": urgency_classification,
    }


def _sanitize_float(v: float | None, default: float | None = None) -> float | None:
    """Replace non-finite floats with a JSON-safe default."""
    if v is None:
        return default
    if not isinstance(v, float):
        return v
    import math
    return default if (math.isinf(v) or math.isnan(v)) else v


@app.get("/api/v1/soma/signal")
async def get_soma_signal():
    """Current allostatic signal - the output all other systems consume."""
    soma: SomaService = app.state.soma
    signal = soma.get_current_signal()
    return {
        "signal_strength": round(signal.urgency, 4),
        "direction": signal.trajectory_heading or "stable",
        "dominant_dimension": signal.dominant_error.value,
        "urgency": round(signal.urgency, 4),
        "dominant_error": signal.dominant_error.value,
        "dominant_error_magnitude": round(signal.dominant_error_magnitude, 4),
        "dominant_error_rate": round(signal.dominant_error_rate, 4),
        "precision_weights": {d.value: round(v, 4) for d, v in signal.precision_weights.items()},
        "max_temporal_dissonance": round(signal.max_temporal_dissonance, 4),
        "dissonant_dimension": signal.dissonant_dimension.value if signal.dissonant_dimension else None,
        "nearest_attractor": signal.nearest_attractor,
        "distance_to_bifurcation": _sanitize_float(signal.distance_to_bifurcation),
        "trajectory_heading": signal.trajectory_heading,
        "energy_burn_rate": round(signal.energy_burn_rate, 4),
        "predicted_energy_exhaustion_s": _sanitize_float(signal.predicted_energy_exhaustion_s),
        "cycle_number": signal.cycle_number,
        "timestamp": signal.timestamp.isoformat(),
    }


@app.get("/api/v1/soma/phase-space")
async def get_soma_phase_space():
    """Phase-space navigation - attractors, bifurcations, trajectory heading."""
    import math

    from systems.soma.types import DIMENSION_RANGES

    def _safe(v: float) -> float:
        return 0.0 if (math.isinf(v) or math.isnan(v)) else round(v, 4)

    soma: SomaService = app.state.soma
    position = soma.get_phase_position()
    attractors = soma._phase_space.attractors
    bifurcations = soma._phase_space.bifurcations

    current_attractor = position.get("nearest_attractor") or "unknown"
    trajectory = position.get("trajectory_heading") or "stable"

    bif_warnings: list[str] = []
    for b in bifurcations:
        bif_warnings.append(f"{b.pre_regime} → {b.post_regime} ({b.crossing_count} crossings)")

    # Build per-dimension bounds from DIMENSION_RANGES so the frontend can
    # render axes correctly without assuming a universal [-1, 1] range.
    # Most dimensions are [0, 1]; valence is [-1, 1].
    bounds: dict[str, list[float]] = {
        dim.value: list(rng)
        for dim, rng in DIMENSION_RANGES.items()
    }

    return {
        "current_attractor": current_attractor,
        "trajectory": trajectory,
        "attractors": [
            {
                "name": a.label,
                "stability": _safe(a.stability),
            }
            for a in attractors
        ],
        "bifurcations": bif_warnings,
        "position": {
            k: (_safe(v) if isinstance(v, float) else v)
            for k, v in position.items()
        },
        "raw_attractors": [
            {
                "label": a.label,
                "basin_radius": _safe(a.basin_radius),
                "stability": _safe(a.stability),
                "stability_label": (
                    "stable" if a.stability >= 0.8
                    else "meta-stable" if a.stability >= 0.5
                    else "unstable"
                ),
                "valence": _safe(a.valence),
                "visits": a.visits,
                "mean_dwell_time_s": _safe(a.mean_dwell_time_s),
            }
            for a in attractors
        ],
        "bounds": bounds,
    }


@app.get("/api/v1/soma/developmental")
async def get_soma_developmental():
    """Developmental stage and maturation progress."""
    soma: SomaService = app.state.soma
    stage = soma.get_developmental_stage()
    stage_order = ["reflexive", "associative", "deliberative", "reflective", "generative"]
    stage_val = stage.value if hasattr(stage, "value") else str(stage)
    stage_idx = stage_order.index(stage_val) if stage_val in stage_order else 0
    # Maturation progress within current stage based on cycle_count thresholds
    # Rough: reflexive=0-100, associative=100-1000, deliberative=1k-10k, etc.
    stage_cycle_thresholds = [0, 100, 1000, 10_000, 100_000]
    lo = stage_cycle_thresholds[stage_idx]
    hi = stage_cycle_thresholds[stage_idx + 1] if stage_idx + 1 < len(stage_cycle_thresholds) else lo + 100_000
    cycle = soma.cycle_count
    progress = min(1.0, max(0.0, (cycle - lo) / (hi - lo))) if hi > lo else 1.0
    caps = soma._developmental.capabilities if hasattr(soma._developmental, "capabilities") else []
    horizons = soma._temporal_depth.available_horizons if hasattr(soma._temporal_depth, "available_horizons") else []
    return {
        "stage": stage_idx,
        "stage_name": stage_val,
        "maturation_progress": round(progress, 4),
        "unlocked_capabilities": list(caps),
        "cycle_count": cycle,
        "available_horizons": list(horizons),
    }


@app.get("/api/v1/soma/errors")
async def get_soma_errors():
    """Allostatic errors per horizon per dimension."""
    soma: SomaService = app.state.soma
    errors = soma.get_errors()
    horizons = []
    for horizon, dims in errors.items():
        errors_list = [
            {"dimension": d.value, "magnitude": round(abs(v), 4)}
            for d, v in dims.items()
            if abs(v) > 0.0001
        ]
        errors_list.sort(key=lambda x: x["magnitude"], reverse=True)
        horizons.append({"horizon": horizon, "errors": errors_list})
    horizons.sort(key=lambda h: max((e["magnitude"] for e in h["errors"]), default=0), reverse=True)
    return {"horizons": horizons}


@app.get("/api/v1/soma/exteroception")
async def get_soma_exteroception():
    """Exteroceptive pressure - the organism's felt sense of external weather."""
    soma: SomaService = app.state.soma
    pressure = soma.exteroceptive_pressure
    if pressure is None:
        return {"stress_level": 0.0, "sources": [], "status": "no_data"}
    # Build named sources from per-dimension pressures + modalities
    sources = [
        {"name": dim.value, "pressure": round(abs(val), 4)}
        for dim, val in pressure.pressures.items()
        if abs(val) > 0.0001
    ]
    sources.sort(key=lambda s: s["pressure"], reverse=True)
    return {
        "stress_level": round(pressure.ambient_stress, 4),
        "sources": sources,
        "active_modalities": [m.value for m in pressure.active_modalities],
        "reading_count": pressure.reading_count,
        "total_absolute_pressure": round(pressure.total_absolute_pressure(), 4),
        "external_stress_scalar": round(soma.external_stress, 4),
    }




def _soma_sf(v):
    import math as _math
    if v is None:
        return None
    if not isinstance(v, float):
        return v
    return None if (_math.isinf(v) or _math.isnan(v)) else round(v, 4)


@app.get("/api/v1/soma/vulnerability")
async def get_soma_vulnerability():
    soma: SomaService = app.state.soma
    vuln = soma.vulnerability_map()
    return {
        "fragile_dimensions": {k: _soma_sf(v) for k, v in vuln.get("fragile_dimensions", {}).items()},
        "vulnerable_pairs": [
            {"source": src, "target": tgt, "curvature": _soma_sf(cur)}
            for src, tgt, cur in vuln.get("vulnerable_pairs", [])
        ],
        "unexpected_influences": [
            {"source": item.get("source"), "target": item.get("target"), "te": _soma_sf(item.get("te"))}
            for item in vuln.get("unexpected_influences", [])
        ],
        "missing_influences": [
            {"source": item.get("source"), "target": item.get("target"),
             "expected": _soma_sf(item.get("expected")), "actual": _soma_sf(item.get("actual"))}
            for item in vuln.get("missing_influences", [])
        ],
        "topological_breaches": vuln.get("topological_breaches", 0),
        "topological_fractures": vuln.get("topological_fractures", 0),
        "novel_cycles": vuln.get("novel_cycles", 0),
        "chaotic_metrics": [
            {"metric": item.get("metric"), "lyapunov": _soma_sf(item.get("lyapunov")), "horizon": item.get("horizon")}
            for item in vuln.get("chaotic_metrics", [])
        ],
    }


@app.get("/api/v1/soma/analysis")
async def get_soma_analysis():
    soma: SomaService = app.state.soma
    result = soma.latest_analysis()
    if not result:
        return {"status": "pending", "message": "Deep analysis has not completed yet"}
    geo = result.get("geodesic_deviation") or {}
    emergence = result.get("emergence") or {}
    causal = result.get("causal_flow") or {}
    rg = result.get("renormalization") or {}
    topo = result.get("topology") or {}
    curv = result.get("curvature") or {}
    psr = result.get("phase_space") or {}
    return {
        "status": "ok",
        "geodesic_deviation": {
            "scalar": _soma_sf(geo.get("scalar")),
            "percentile": _soma_sf(geo.get("percentile")),
            "dominant_systems": geo.get("dominant_systems") or [],
        },
        "emergence": {
            "causal_emergence": _soma_sf(emergence.get("causal_emergence") if isinstance(emergence, dict) else None),
            "macro_states": emergence.get("macro_states") if isinstance(emergence, dict) else None,
        },
        "causal_flow": {
            "max_te": _soma_sf(causal.get("max_te") if isinstance(causal, dict) else None),
            "mean_te": _soma_sf(causal.get("mean_te") if isinstance(causal, dict) else None),
            "dominant_pair": causal.get("dominant_pair") if isinstance(causal, dict) else None,
        },
        "renormalization": {
            "anomaly_scale": rg.get("anomaly_scale"),
            "interpretation": rg.get("interpretation"),
            "fixed_point_drift": _soma_sf(rg.get("fixed_point_drift")),
            "n_fixed_points": rg.get("n_fixed_points"),
        },
        "topology": {
            "betti_numbers": topo.get("betti_numbers") if isinstance(topo, dict) else None,
            "n_breaches": topo.get("n_breaches") if isinstance(topo, dict) else None,
        },
        "curvature": {
            "overall": _soma_sf(curv.get("overall")),
            "most_vulnerable_region": curv.get("most_vulnerable_region"),
            "n_vulnerable_pairs": curv.get("n_vulnerable_pairs"),
        },
        "phase_space_reconstruction": {
            "n_diagnosed": psr.get("n_diagnosed"),
            "n_skipped": psr.get("n_skipped"),
            "chaotic_metrics": psr.get("chaotic_metrics") or [],
        },
    }


@app.get("/api/v1/soma/manifold")
async def get_soma_manifold():
    soma: SomaService = app.state.soma
    sv = soma.get_organism_state_vector()
    deriv = soma.get_derivatives()
    percept = soma.get_interoceptive_percept()
    state_vector: dict = {}
    if sv is not None:
        state_vector = {
            "timestamp": sv.timestamp,
            "cycle_number": sv.cycle_number,
            "systems": {
                sys_name: {
                    "call_rate": _soma_sf(slice_.call_rate),
                    "error_rate": _soma_sf(slice_.error_rate),
                    "mean_latency_ms": _soma_sf(slice_.mean_latency_ms),
                    "latency_variance": _soma_sf(slice_.latency_variance),
                    "success_ratio": _soma_sf(slice_.success_ratio),
                    "resource_rate": _soma_sf(slice_.resource_rate),
                    "event_entropy": _soma_sf(slice_.event_entropy),
                }
                for sys_name, slice_ in sv.systems.items()
            },
        }
    derivatives: dict = {}
    if deriv is not None:
        derivatives = {
            "organism_velocity_norm": {k: _soma_sf(v) for k, v in (deriv.organism_velocity_norm or {}).items()},
            "organism_acceleration_norm": {k: _soma_sf(v) for k, v in (deriv.organism_acceleration_norm or {}).items()},
            "organism_jerk_norm": {k: _soma_sf(v) for k, v in (deriv.organism_jerk_norm or {}).items()},
            "dominant_system_velocity": deriv.dominant_system_velocity or {},
            "dominant_system_acceleration": deriv.dominant_system_acceleration or {},
            "dominant_system_jerk": deriv.dominant_system_jerk or {},
        }
    last_percept: dict | None = None
    if percept is not None:
        last_percept = {
            "sensation_type": percept.sensation_type.value if hasattr(percept.sensation_type, "value") else str(percept.sensation_type),
            "recommended_action": percept.recommended_action.value if hasattr(percept.recommended_action, "value") else str(percept.recommended_action),
            "magnitude": _soma_sf(percept.magnitude),
            "source_systems": list(percept.source_systems) if percept.source_systems else [],
            "timestamp": percept.timestamp.isoformat() if hasattr(percept.timestamp, "isoformat") else str(percept.timestamp),
        }
    return {"state_vector": state_vector, "derivatives": derivatives, "last_percept": last_percept}


@app.get("/api/v1/soma/financial")
async def get_soma_financial():
    soma: SomaService = app.state.soma
    signal = soma.get_current_signal()
    ttd = _soma_sf(signal.financial_ttd_days)
    cfg = soma._config
    regime = "unknown"
    if ttd is not None:
        if ttd >= cfg.temporal_depth_secure_days:
            regime = "secure"
        elif ttd >= cfg.temporal_depth_comfortable_days:
            regime = "comfortable"
        elif ttd >= cfg.temporal_depth_cautious_days:
            regime = "cautious"
        elif ttd >= cfg.temporal_depth_anxious_days:
            regime = "anxious"
        else:
            regime = "critical"
    affect_bias = signal.financial_affect_bias or {}
    return {
        "ttd_days": ttd,
        "regime": regime,
        "affect_bias": {
            (d.value if hasattr(d, "value") else str(d)): _soma_sf(v)
            for d, v in affect_bias.items()
        },
        "thresholds": {
            "secure_days": cfg.temporal_depth_secure_days,
            "comfortable_days": cfg.temporal_depth_comfortable_days,
            "cautious_days": cfg.temporal_depth_cautious_days,
            "anxious_days": cfg.temporal_depth_anxious_days,
            "critical_days": cfg.temporal_depth_critical_days,
        },
        "predicted_energy_exhaustion_s": _soma_sf(signal.predicted_energy_exhaustion_s),
        "energy_burn_rate": _soma_sf(signal.energy_burn_rate),
    }


@app.post("/api/v1/soma/context")
async def set_soma_context(payload: dict):
    soma: SomaService = app.state.soma
    context = payload.get("context", "")
    valid = {"conversation", "deep_processing", "recovery", "exploration"}
    if context not in valid:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Invalid context. Must be one of: {sorted(valid)}")
    soma.set_context(context)
    return {"ok": True, "context": context}


@app.post("/api/v1/soma/inject-stress")
async def inject_soma_stress(payload: dict):
    soma: SomaService = app.state.soma
    stress = payload.get("stress", 0.0)
    if not isinstance(stress, (int, float)) or stress < 0 or stress > 1:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="stress must be a float in [0, 1]")
    soma.inject_external_stress(float(stress))
    return {"ok": True, "stress": float(stress)}


@app.get("/api/v1/soma/emotions")
async def get_soma_emotions():
    """Emergent emotion regions active in the current allostatic error space."""
    soma: SomaService = app.state.soma
    state = soma.get_current_state()
    if state is None:
        return {"emotions": [], "status": "no_state"}
    emotions = soma._emotion_detector.detect(state)
    return {
        "emotions": [e.to_dict() for e in emotions],
        "urgency": round(state.urgency, 4),
        "timestamp": state.timestamp.isoformat(),
    }


@app.get("/api/v1/soma/predictions")
async def get_soma_predictions():
    """Multi-horizon predictions for all 9 interoceptive dimensions."""
    import math as _m

    def _sf(v):
        if v is None:
            return None
        if not isinstance(v, float):
            return v
        return None if (_m.isinf(v) or _m.isnan(v)) else round(v, 4)

    soma: SomaService = app.state.soma
    state = soma.get_current_state()
    if state is None:
        return {"horizons": [], "temporal_dissonance": {}, "status": "no_state"}

    horizons = []
    for horizon_name, dim_predictions in (state.predicted or {}).items():
        predictions = []
        for d in state.sensed:
            predicted_val = dim_predictions.get(d)
            sensed_val = state.sensed.get(d, 0.0)
            setpoint_val = state.setpoints.get(d, sensed_val)
            error_at_horizon = (predicted_val - setpoint_val) if predicted_val is not None else None
            predictions.append({
                "dimension": d.value,
                "sensed": _sf(sensed_val),
                "predicted": _sf(predicted_val),
                "setpoint": _sf(setpoint_val),
                "error_at_horizon": _sf(error_at_horizon),
            })
        horizons.append({"horizon": horizon_name, "predictions": predictions})

    dissonance = {
        d.value: _sf(v) for d, v in (state.temporal_dissonance or {}).items()
        if v is not None and abs(v) > 0.001
    }
    return {
        "horizons": horizons,
        "temporal_dissonance": dissonance,
        "max_dissonance": _sf(max(abs(v) for v in state.temporal_dissonance.values()) if state.temporal_dissonance else 0.0),
        "timestamp": state.timestamp.isoformat(),
    }


@app.get("/api/v1/soma/markers")
async def get_soma_markers():
    """Recent somatic markers and current 19D marker vector."""
    soma: SomaService = app.state.soma
    marker = soma.get_somatic_marker()
    if marker is None:
        return {"current_marker": None, "marker_vector": [], "status": "no_marker"}
    from systems.soma.types import ALL_DIMENSIONS
    vector = marker.to_vector()
    snapshot = {d.value: round(v, 4) for d, v in marker.interoceptive_snapshot.items()}
    error_snapshot = {d.value: round(v, 4) for d, v in marker.allostatic_error_snapshot.items()}
    return {
        "current_marker": {
            "interoceptive_snapshot": snapshot,
            "allostatic_error_snapshot": error_snapshot,
            "prediction_error_at_encoding": round(marker.prediction_error_at_encoding, 4),
            "allostatic_context": marker.allostatic_context,
        },
        "marker_vector": [round(v, 4) for v in vector],
        "dimension_labels": [d.value for d in ALL_DIMENSIONS] + [d.value + "_err" for d in ALL_DIMENSIONS] + ["pe"],
        "status": "ok",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Command Center - Phantom + Inspector pipeline (SSE streaming)
# ─────────────────────────────────────────────────────────────────────────────

import json as _cc_json
import re as _cc_re
import signal as _cc_signal
import sys as _cc_sys
import uuid as _cc_uuid
from pathlib import Path as _CCPath

from fastapi.responses import StreamingResponse as _CCStreamingResponse
from pydantic import BaseModel as _CCBaseModel


def _cc_try_parse_json(s: str | None) -> dict:
    """Parse a Z3 counterexample string as JSON, falling back to raw string."""
    if not s:
        return {}
    try:
        return _cc_json.loads(s)
    except (_cc_json.JSONDecodeError, ValueError):
        return {"raw": s}

_CC_BACKEND_DIR = _CCPath(__file__).parent.parent  # ecodiaos/../../ = backend/
_CC_PHANTOM = _CC_BACKEND_DIR / "phantom_recon.py"
_CC_INSPECTOR = _CCPath(__file__).parent / "systems" / "simula" / "run_inspector.py"
_CC_FILE_RE = _cc_re.compile(r"file:///tmp/[^\s]+")

# ── Active subprocess registry (task_id → asyncio.subprocess.Process) ─────
# Only tracks child processes spawned by the pipeline, never the parent uvicorn.
_cc_active_procs: dict[str, asyncio.subprocess.Process] = {}


class _CCEngagePayload(_CCBaseModel):
    target_url: str


def _cc_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {_cc_json.dumps(data)}\n\n"


async def _cc_stream_proc(
    cmd: list[str],
    stdin_data: bytes | None = None,
    task_id: str | None = None,
):
    """Async subprocess stream - yields ``(source, line)``.  No ``shell=True``.

    When *task_id* is given the child process is registered in
    ``_cc_active_procs`` so it can be terminated via
    ``POST /api/v1/command-center/terminate/{task_id}``.
    Only the spawned child is tracked - never the parent uvicorn process.
    """
    import asyncio as _aio
    import os as _os

    proc = await _aio.create_subprocess_exec(
        *cmd,
        stdin=_aio.subprocess.PIPE if stdin_data is not None else None,
        stdout=_aio.subprocess.PIPE,
        stderr=_aio.subprocess.PIPE,
        env=_os.environ.copy(),
    )

    # Register so terminate endpoint can reach this child.
    if task_id is not None:
        _cc_active_procs[task_id] = proc

    # Feed stdin then close it so input() unblocks
    if stdin_data is not None and proc.stdin is not None:
        proc.stdin.write(stdin_data)
        await proc.stdin.drain()
        proc.stdin.close()

    q: _aio.Queue = _aio.Queue()

    async def _feed(stream, label: str) -> None:
        while True:
            raw = await stream.readline()
            if not raw:
                break
            await q.put((label, raw.decode(errors="replace").rstrip()))
        await q.put(None)

    assert proc.stdout and proc.stderr
    tasks = [
        _aio.create_task(_feed(proc.stdout, "stdout")),
        _aio.create_task(_feed(proc.stderr, "stderr")),
    ]

    done = 0
    while done < 2:
        item = await q.get()
        if item is None:
            done += 1
        else:
            yield item

    await _aio.gather(*tasks)
    await proc.wait()

    # Unregister once the child exits naturally.
    if task_id is not None:
        _cc_active_procs.pop(task_id, None)


async def _cc_pipeline(target_url: str, task_id: str):
    def log(phase: str, text: str) -> str:
        return _cc_sse("log", {"phase": phase, "text": text})

    def phase_ev(name: str, status: str) -> str:
        return _cc_sse("phase", {"name": name, "status": status})

    # Emit task_id to the client so the UI can target this run for termination.
    yield _cc_sse("task_id", {"task_id": task_id})

    for script, label in [(_CC_PHANTOM, "phantom_recon.py"), (_CC_INSPECTOR, "run_inspector.py")]:
        if not script.exists():
            yield _cc_sse("error", {"message": f"Missing script: {label} (expected at {script})"})
            yield _cc_sse("done", {"success": False, "message": "Aborted - missing scripts"})
            return

    # ── Phase 1: Phantom Harvester ─────────────────────────────────────────
    yield phase_ev("phantom", "started")
    yield log("phantom", f"[PHANTOM] Initiating black-box recon on {target_url}")

    repo_path = None
    phantom_ok = True

    try:
        async for src, line in _cc_stream_proc([_cc_sys.executable, str(_CC_PHANTOM)], stdin_data=(target_url + chr(10)).encode(), task_id=task_id):
            pfx = "[PHANTOM] " if src == "stdout" else "[PHANTOM·ERR] "
            yield log("phantom", f"{pfx}{line}")

            # Try to parse JSON output from phantom_recon.py
            if src == "stdout" and repo_path is None:
                try:
                    payload = _cc_json.loads(line)
                    if isinstance(payload, dict) and "repo_path" in payload:
                        repo_path = payload.get("repo_path")
                        if repo_path:
                            yield _cc_sse("result", {"repo_path": repo_path})
                            yield log("phantom", f"[PHANTOM] Repo path extracted → {repo_path}")
                except (_cc_json.JSONDecodeError, ValueError):
                    # Not JSON, continue streaming
                    pass
    except Exception as exc:
        yield _cc_sse("error", {"message": str(exc)})
        phantom_ok = False

    if not phantom_ok or repo_path is None:
        yield phase_ev("phantom", "failed")
        yield _cc_sse("done", {"success": False, "message": "Phantom recon failed or produced invalid output"})
        return

    yield phase_ev("phantom", "completed")

    # ── Phase 2: Inspector (AST + Z3 + XDP) ──────────────────────────────────
    yield phase_ev("inspector", "started")
    yield log("inspector", f"[INSPECTOR] Loading analysis target: {repo_path}")

    inspector_ok = True
    _boundary_evidence: list[dict] = []

    # Branch: proxy mode offloads Z3 to the worker; local mode spawns
    # run_inspector.py as a subprocess on this process.
    _iproxy = getattr(app.state, "inspector_proxy", None)

    if _iproxy is not None:
        # ── PROXY PATH: non-blocking call to the worker ───────────
        yield log("inspector", "[INSPECTOR] Routing hunt to Simula worker via Redis proxy")
        yield phase_ev("ast", "started")

        try:
            _hunt_result = await _iproxy.hunt_external_repo(
                repo_path,
                generate_pocs=True,
                generate_patches=True,
            )
        except Exception as exc:
            yield _cc_sse("error", {"message": str(exc)})
            inspector_ok = False
            _hunt_result = None

        if inspector_ok and _hunt_result is not None:
            yield phase_ev("ast", "completed")
            yield phase_ev("z3", "started")

            # Synthesise boundary_test SSE events from the InspectionResult
            # so the UI and Phase 3 receive the same data shape.
            for vuln in _hunt_result.vulnerabilities_found:
                try:
                    bt = {
                        "status": "sat",
                        "details": {
                            "vuln_id": vuln.id,
                            "endpoint": vuln.attack_surface.entry_point,
                            "file_path": vuln.attack_surface.file_path,
                            "line_number": vuln.attack_surface.line_number,
                            "vulnerability_class": vuln.vulnerability_class.value,
                            "severity": vuln.severity.value,
                            "attack_goal": vuln.attack_goal,
                            "edge_case_input": _cc_try_parse_json(vuln.z3_counterexample),
                            "surface_type": vuln.attack_surface.surface_type.value,
                            "context_code": vuln.attack_surface.context_code,
                            "z3_constraints": vuln.z3_constraints_code,
                        },
                    }
                except Exception as _bt_err:
                    yield log("inspector", f"[Z3·EVIDENCE] Failed to build boundary_test for {vuln.id}: {_bt_err}")
                    continue
                _boundary_evidence.append(bt)
                yield _cc_sse("boundary_test", bt)
                yield log(
                    "inspector",
                    f"[Z3·EVIDENCE] {vuln.vulnerability_class.value.upper()} "
                    f"at {vuln.attack_surface.file_path}:{vuln.attack_surface.line_number}",
                )

            yield phase_ev("z3", "completed")
            yield log(
                "inspector",
                f"[INSPECTOR] Hunt complete - {_hunt_result.surfaces_mapped} surfaces, "
                f"{len(_hunt_result.vulnerabilities_found)} vulns, "
                f"{_hunt_result.total_duration_ms}ms",
            )
            yield phase_ev("inspector", "completed")
        else:
            yield phase_ev("inspector", "failed")
            yield _cc_sse("done", {"success": False, "message": "Inspector proxy hunt failed"})
            return

    else:
        # ── LOCAL PATH: subprocess (original behavior) ────────────
        ast_seen = z3_seen = xdp_seen = False

        try:
            async for src, line in _cc_stream_proc([_cc_sys.executable, str(_CC_INSPECTOR), "--target", repo_path], task_id=task_id):
                pfx = "[INSPECTOR] " if src == "stdout" else "[INSPECTOR·ERR] "
                ll = line.lower()

                if src == "stdout":
                    try:
                        _parsed = _cc_json.loads(line)
                        if isinstance(_parsed, dict) and "boundary_test" in _parsed:
                            bt = _parsed["boundary_test"]
                            _boundary_evidence.append(bt)
                            yield _cc_sse("boundary_test", bt)
                            yield log("inspector", f"[Z3·EVIDENCE] Boundary test result emitted for {bt.get('details', {}).get('endpoint', 'unknown')}")
                            continue
                    except (_cc_json.JSONDecodeError, ValueError):
                        pass

                yield log("inspector", f"{pfx}{line}")

                if not ast_seen and any(k in ll for k in ("ast", "slic", "pars", "context")):
                    ast_seen = True
                    yield phase_ev("ast", "started")

                if not z3_seen and any(k in ll for k in ("z3", "smt", "prov", "satisf", "constraint", "formal")):
                    z3_seen = True
                    if ast_seen:
                        yield phase_ev("ast", "completed")
                    yield phase_ev("z3", "started")

                if not xdp_seen and any(k in ll for k in ("xdp", "ebpf", "bpf", "shield", "kernel", "layer 2")):
                    xdp_seen = True
                    if z3_seen:
                        yield phase_ev("z3", "completed")
                    elif ast_seen:
                        yield phase_ev("ast", "completed")
                    yield phase_ev("xdp", "started")

        except Exception as exc:
            yield _cc_sse("error", {"message": str(exc)})
            inspector_ok = False

        if inspector_ok:
            for name, seen in [("ast", ast_seen), ("z3", z3_seen), ("xdp", xdp_seen)]:
                if seen:
                    yield phase_ev(name, "completed")
            yield phase_ev("inspector", "completed")
        else:
            yield phase_ev("inspector", "failed")
            yield _cc_sse("done", {"success": False, "message": "Inspector engine crashed"})
            return

    # ── Phase 3: Deterministic Shield Deployment ─────────────────────────────
    # Generate a verifier-compliant XDP filter from the boundary_test evidence,
    # attach it live, and stream kernel telemetry back over this SSE connection.

    if not _boundary_evidence:
        yield log("shield", "[SHIELD] No boundary_test evidence collected - skipping filter deployment")
        yield _cc_sse("done", {"success": True, "message": "Pipeline complete - no exploits proven, shield skipped"})
        return

    yield phase_ev("shield", "started")

    # 3a. Extract edge_case_input from the first evidence payload and generate C.
    try:
        from systems.simula.filter_generator import generate_xdp_filter as _gen_filter

        _edge_input = _boundary_evidence[0].get("details", {}).get("edge_case_input", {})
        if not _edge_input:
            yield log("shield", "[SHIELD] boundary_test evidence has no edge_case_input - skipping")
            yield phase_ev("shield", "skipped")
            yield _cc_sse("done", {"success": True, "message": "Pipeline complete - edge_case_input empty"})
            return

        _generated_c = _gen_filter(_edge_input)
        yield log("shield", f"[SHIELD] Deterministic XDP filter generated ({len(_generated_c)} bytes, {len(_boundary_evidence)} evidence payloads)")
        yield _cc_sse("filter_generated", {
            "code_size": len(_generated_c),
            "evidence_count": len(_boundary_evidence),
        })

    except Exception as exc:
        yield _cc_sse("error", {"message": f"Filter generation failed: {exc}"})
        yield phase_ev("shield", "failed")
        yield _cc_sse("done", {"success": False, "message": "Filter generation failed"})
        return

    # 3b. Broadcast the filter to the entire fleet via Redis Pub/Sub.
    #     The local node is also subscribed, so it will receive its own
    #     broadcast and deploy the filter - unifying local and remote logic.
    try:

        _fleet: _Fleet = app.state.fleet_shield
        _receivers = await _fleet.broadcast_filter(_generated_c)
        yield log("shield", f"[SHIELD] Filter broadcast to fleet ({_receivers} subscriber(s))")
        yield _cc_sse("shield_broadcast", {
            "receivers": _receivers,
            "code_bytes": len(_generated_c),
        })
        yield phase_ev("shield", "deployed")

    except Exception as exc:
        yield _cc_sse("error", {"message": f"Fleet broadcast failed: {exc}"})
        yield phase_ev("shield", "failed")
        yield _cc_sse("done", {"success": False, "message": "Fleet broadcast failed"})
        return

    yield phase_ev("shield", "completed")

    # ── Phase 4: Automated Remediation ───────────────────────────────────────
    # Launch the RepairAgent against each boundary test evidence payload.
    # The agent generates a code patch, re-verifies it via Z3, and streams
    # the verified diff back as a `verified_patch` SSE event.

    yield phase_ev("remediation", "started")
    yield log("remediation", "[REPAIR] Phase 4 - Automated Remediation initiated")

    _remediation_count = 0
    _remediation_failures = 0

    try:
        import difflib as _difflib

        from clients.llm import create_llm_provider as _create_llm
        from config import LLMConfig as _LLMConfig
        from systems.simula.inspector.prover import VulnerabilityProver as _Prover
        from systems.simula.inspector.remediation import RepairAgent as _RepairAgent
        from systems.simula.inspector.types import (
            AttackSurface as _AttackSurface,
        )
        from systems.simula.inspector.types import (
            AttackSurfaceType as _AttackSurfaceType,
        )
        from systems.simula.inspector.types import (
            VulnerabilityClass as _VulnClass,
        )
        from systems.simula.inspector.types import (
            VulnerabilityReport as _VulnReport,
        )
        from systems.simula.inspector.types import (
            VulnerabilitySeverity as _VulnSeverity,
        )
        from systems.simula.verification.z3_bridge import Z3Bridge as _Z3Bridge

        _llm_cfg = _LLMConfig(
            provider=os.environ.get("ORGANISM_LLM__PROVIDER", "bedrock"),
            model=os.environ.get("ORGANISM_LLM__MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        )
        _repair_llm = _create_llm(_llm_cfg)
        _z3 = _Z3Bridge(check_timeout_ms=10_000)
        _prover = _Prover(z3_bridge=_z3, llm=_repair_llm)
        _repair_agent = _RepairAgent(llm=_repair_llm, prover=_prover, max_retries=3)

        for _idx, _ev in enumerate(_boundary_evidence):
            _details = _ev.get("details", {})
            _file_path = _details.get("file_path", "")
            _vuln_id = _details.get("vuln_id", f"unknown_{_idx}")

            if not _file_path:
                yield log("remediation", f"[REPAIR] Evidence #{_idx} has no file_path - skipping")
                continue

            yield log("remediation", f"[REPAIR] Processing {_vuln_id} - {_file_path}")

            # Reconstruct a VulnerabilityReport from the boundary evidence
            # so the RepairAgent can consume it.
            try:
                _surface = _AttackSurface(
                    entry_point=_details.get("entry_point", "unknown"),
                    surface_type=_AttackSurfaceType(_details.get("surface_type", "api_endpoint")),
                    file_path=_file_path,
                    line_number=_details.get("line_number"),
                    context_code=_details.get("context_code", ""),
                )
                _report = _VulnReport(
                    id=_vuln_id,
                    target_url=target_url,
                    vulnerability_class=_VulnClass(_details.get("vulnerability_class", "other")),
                    severity=_VulnSeverity(_details.get("severity", "medium")),
                    attack_surface=_surface,
                    attack_goal=_details.get("attack_goal", ""),
                    z3_counterexample=_cc_json.dumps(_details.get("edge_case_input", {})),
                    z3_constraints_code=_details.get("z3_constraints", ""),
                )
            except Exception as _build_err:
                yield log("remediation", f"[REPAIR] Failed to build report for {_vuln_id}: {_build_err}")
                _remediation_failures += 1
                continue

            # Run the RepairAgent - generate patch + Z3 re-verification
            try:
                _patched_code = await _repair_agent.generate_and_verify_patch(_report)
            except Exception as _repair_err:
                yield log("remediation", f"[REPAIR] RepairAgent error for {_vuln_id}: {_repair_err}")
                _remediation_failures += 1
                continue

            if _patched_code is None:
                yield log("remediation", f"[REPAIR] RepairAgent exhausted retries for {_vuln_id} - no verified patch")
                _remediation_failures += 1
                continue

            # Generate unified diff
            _original_lines = (_surface.context_code or "").splitlines(keepends=True)
            _patched_lines = _patched_code.splitlines(keepends=True)
            _diff = "".join(_difflib.unified_diff(
                _original_lines,
                _patched_lines,
                fromfile=f"a/{_file_path}",
                tofile=f"b/{_file_path}",
                lineterm="",
            ))

            yield _cc_sse("verified_patch", {
                "vuln_id": _vuln_id,
                "file_path": _file_path,
                "diff": _diff,
                "patched_code": _patched_code,
                "vulnerability_class": _details.get("vulnerability_class", ""),
                "severity": _details.get("severity", ""),
            })
            yield log("remediation", f"[REPAIR] Verified patch emitted for {_vuln_id} - {_file_path}")
            _remediation_count += 1

        # Cleanup LLM client
        with suppress(Exception):
            await _repair_llm.close()

    except ImportError as _imp_err:
        yield _cc_sse("error", {"message": f"Remediation import error: {_imp_err}"})
        yield log("remediation", f"[REPAIR] Import failed - skipping Phase 4: {_imp_err}")
    except Exception as _rem_err:
        yield _cc_sse("error", {"message": f"Remediation error: {_rem_err}"})
        yield log("remediation", f"[REPAIR] Unexpected error: {_rem_err}")

    if _remediation_count > 0:
        yield log("remediation", f"[REPAIR] Phase 4 complete - {_remediation_count} verified patch(es), {_remediation_failures} failure(s)")
        yield phase_ev("remediation", "completed")
    elif _remediation_failures > 0:
        yield log("remediation", f"[REPAIR] Phase 4 failed - 0 patches verified, {_remediation_failures} failure(s)")
        yield phase_ev("remediation", "failed")
    else:
        yield log("remediation", "[REPAIR] Phase 4 - no evidence to remediate")
        yield phase_ev("remediation", "completed")

    yield _cc_sse("done", {
        "success": True,
        "message": f"Pipeline complete - shield deployed, {_remediation_count} verified patch(es)",
    })


@app.post("/api/v1/command-center/engage")
async def command_center_engage(payload: _CCEngagePayload) -> _CCStreamingResponse:
    """Stream the Phantom + Inspector pipeline as Server-Sent Events."""
    task_id = str(_cc_uuid.uuid4())

    async def gen():
        try:
            async for chunk in _cc_pipeline(payload.target_url, task_id=task_id):
                yield chunk.encode()
        except Exception as _gen_err:
            # Emit a done event so the UI knows the pipeline crashed
            # instead of hanging indefinitely.
            yield _cc_sse("error", {"message": str(_gen_err)}).encode()
            yield _cc_sse("done", {"success": False, "message": f"Pipeline error: {_gen_err}"}).encode()
        finally:
            # Guarantee cleanup even if the client disconnects mid-stream.
            _cc_active_procs.pop(task_id, None)

    return _CCStreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/v1/command-center/terminate/{task_id}")
async def command_center_terminate(task_id: str) -> dict:
    """Gracefully terminate a running Command Center subprocess.

    Sends SIGINT first (allowing cleanup), then SIGKILL after 5 s
    if the child is still alive.  Only targets child processes in
    ``_cc_active_procs`` - never the parent uvicorn server.
    """
    import asyncio as _aio

    proc = _cc_active_procs.get(task_id)
    if proc is None:
        return {"status": "not_found", "message": f"No active task with id {task_id}"}

    if proc.returncode is not None:
        _cc_active_procs.pop(task_id, None)
        return {"status": "already_exited", "returncode": proc.returncode}

    # Phase 1: graceful - SIGINT (Ctrl-C) lets the child clean up.
    try:
        proc.send_signal(_cc_signal.SIGINT)
    except (OSError, ProcessLookupError):
        _cc_active_procs.pop(task_id, None)
        return {"status": "already_exited", "returncode": proc.returncode}

    # Wait up to 5 s for a clean exit.
    try:
        await _aio.wait_for(proc.wait(), timeout=5.0)
    except TimeoutError:
        # Phase 2: forceful - SIGKILL.
        try:
            proc.kill()
            await proc.wait()
        except (OSError, ProcessLookupError):
            pass

    _cc_active_procs.pop(task_id, None)
    return {
        "status": "terminated",
        "returncode": proc.returncode,
        "task_id": task_id,
    }


@app.get("/api/v1/command-center/health")
async def command_center_health() -> dict:
    return {
        "phantom": _CC_PHANTOM.exists(),
        "inspector": _CC_INSPECTOR.exists(),
    }


# ─── Deploy Patch ─────────────────────────────────────────────────────────────

import pathlib as _pathlib
import shutil as _shutil
import tempfile as _tempfile

from pydantic import BaseModel as _DeployBase

_ORGANISM_ROOT = _pathlib.Path(__file__).resolve().parent.parent


class _DeployPatchPayload(_DeployBase):
    vuln_id: str
    file_path: str
    patched_code: str


class _DeployPatchResponse(_DeployBase):
    success: bool
    applied_file: str = ""
    backup_path: str = ""
    error: str | None = None


@app.post("/api/v1/command-center/deploy-patch", response_model=_DeployPatchResponse)
async def command_center_deploy_patch(payload: _DeployPatchPayload) -> _DeployPatchResponse:
    """
    Write a verified RepairAgent patch to disk.

    Security: *file_path* is resolved relative to the EcodiaOS project root
    and must stay within it - path traversal returns 400.
    """
    # Resolve and validate the target path is within the project root.
    try:
        target = (_ORGANISM_ROOT / payload.file_path).resolve()
        target.relative_to(_ORGANISM_ROOT)
    except ValueError:
        return _DeployPatchResponse(
            success=False,
            error=f"Path traversal rejected: {payload.file_path!r} is outside project root.",
        )

    # Back up the existing file (if present) before overwriting.
    backup_path = ""
    if target.exists():
        backup = _pathlib.Path(_tempfile.mktemp(suffix=".bak", prefix=target.name + "_"))
        try:
            _shutil.copy2(target, backup)
            backup_path = str(backup)
        except OSError as exc:
            return _DeployPatchResponse(
                success=False,
                error=f"Could not create backup: {exc}",
            )

    # Write patched content, creating parent dirs if needed.
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload.patched_code, encoding="utf-8")
    except OSError as exc:
        # Restore from backup on write failure.
        if backup_path and _pathlib.Path(backup_path).exists():
            with suppress(OSError):
                _shutil.copy2(backup_path, target)
        return _DeployPatchResponse(
            success=False,
            error=f"Write failed: {exc}",
        )

    logger.info(
        "command_center_patch_applied",
        vuln_id=payload.vuln_id,
        applied_file=str(target),
        backup_path=backup_path,
    )

    return _DeployPatchResponse(
        success=True,
        applied_file=str(target),
        backup_path=backup_path,
    )


# ─── Log Stream SSE ──────────────────────────────────────────────────────────

import json as _json

from fastapi.responses import StreamingResponse as _LogStreamingResponse
from telemetry.logging import subscribe_logs, unsubscribe_logs


@app.get("/api/v1/admin/logs/stream")
async def stream_logs():
    """
    Server-Sent Events stream of all structlog/stdlib log records.

    Each event is: ``data: <json>\\n\\n``

    Fields: ts, level, logger, event, + any extra structlog context keys.
    Connect from the browser with EventSource or a fetch+ReadableStream.
    The endpoint sends a heartbeat comment every 15 s to keep proxies alive.
    """
    q = subscribe_logs()

    async def generate():
        try:
            # Initial ping so the browser knows the stream is open
            yield b": connected\n\n"
            heartbeat_interval = 15.0
            while True:
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=heartbeat_interval)
                    yield f"data: {_json.dumps(entry)}\n\n".encode()
                except TimeoutError:
                    # SSE keep-alive comment
                    yield b": heartbeat\n\n"
        finally:
            unsubscribe_logs(q)

    return _LogStreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─── Metrics Pub/Sub + SSE ───────────────────────────────────────────────────
# _METRICS_CHANNEL and publish_metrics_loop are imported from telemetry.publisher


@app.get("/api/v1/command-center/metrics")
async def command_center_metrics_stream():
    """
    Server-Sent Events stream of system metrics published to the
    ecodiaos:system:metrics Redis channel.
    """
    redis = app.state.redis.client

    async def generate():
        async with redis.pubsub() as pubsub:
            await pubsub.subscribe(_METRICS_CHANNEL)
            try:
                yield b": connected\n\n"
                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue
                    data = message["data"]
                    try:
                        _json.loads(data)  # validate before forwarding
                        yield f"data: {data}\n\n".encode()
                    except (_json.JSONDecodeError, TypeError):
                        logger.warning("metrics_invalid_payload")
            except asyncio.CancelledError:
                pass
            finally:
                await pubsub.unsubscribe(_METRICS_CHANNEL)

    return _LogStreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─── Oikos (Economic Engine) ───────────────────────────────────────


@app.get("/api/v1/oikos/status")
async def oikos_status() -> dict[str, Any]:
    """Full Oikos economic snapshot - the organism's financial truth."""
    from fastapi.responses import JSONResponse

    oikos = app.state.oikos
    s = oikos.snapshot()

    cert_mgr = getattr(oikos, "_certificate_manager", None)
    cert = cert_mgr.certificate if cert_mgr else None

    return JSONResponse(
        content={
            "total_net_worth": str(s.total_net_worth),
            "liquid_balance": str(s.liquid_balance),
            "survival_reserve": str(s.survival_reserve),
            "survival_reserve_target": str(s.survival_reserve_target),
            "total_deployed": str(s.total_deployed),
            "total_receivables": str(s.total_receivables),
            "total_asset_value": str(s.total_asset_value),
            "total_fleet_equity": str(s.total_fleet_equity),
            "bmr_usd_per_day": str(s.basal_metabolic_rate.usd_per_day),
            "burn_rate_usd_per_day": str(s.current_burn_rate.usd_per_day),
            "runway_days": str(s.runway_days),
            "starvation_level": s.starvation_level.value,
            "metabolic_efficiency": str(s.metabolic_efficiency),
            "is_metabolically_positive": s.is_metabolically_positive,
            "revenue_24h": str(s.revenue_24h),
            "revenue_7d": str(s.revenue_7d),
            "costs_24h": str(s.costs_24h),
            "costs_7d": str(s.costs_7d),
            "net_income_24h": str(s.net_income_24h),
            "net_income_7d": str(s.net_income_7d),
            "survival_probability_30d": str(s.survival_probability_30d),
            "certificate": {
                "status": cert.status.value if cert else "none",
                "type": cert.certificate_type.value if cert else None,
                "issued_at": cert.issued_at.isoformat() if cert else None,
                "expires_at": cert.expires_at.isoformat() if cert else None,
                "remaining_days": round(oikos.certificate_validity_days, 1),
                "lineage_hash": cert.lineage_hash if cert else None,
                "instance_id": cert.instance_id if cert else None,
            },
            "timestamp": s.timestamp.isoformat(),
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/v1/oikos/organs")
async def oikos_organs() -> dict[str, Any]:
    """Economic morphogenesis - active organs and their lifecycle states."""
    from fastapi.responses import JSONResponse

    oikos = app.state.oikos
    organs = oikos._morphogenesis.all_organs

    return JSONResponse(
        content={
            "organs": [
                {
                    "organ_id": o.organ_id,
                    "category": o.category.value,
                    "specialisation": o.specialisation,
                    "maturity": o.maturity.value,
                    "resource_allocation_pct": str(o.resource_allocation_pct),
                    "efficiency": str(o.efficiency),
                    "revenue_30d": str(o.revenue_30d),
                    "cost_30d": str(o.cost_30d),
                    "days_since_last_revenue": o.days_since_last_revenue,
                    "is_active": o.is_active,
                    "created_at": o.created_at.isoformat(),
                }
                for o in organs
            ],
            "active_count": len([o for o in organs if o.is_active]),
            "total_count": len(organs),
            "stats": oikos._morphogenesis.stats,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/v1/oikos/assets")
async def oikos_assets() -> dict[str, Any]:
    """Owned autonomous assets and child fleet positions."""
    from fastapi.responses import JSONResponse

    oikos = app.state.oikos
    s = oikos.snapshot()

    return JSONResponse(
        content={
            "owned_assets": [
                {
                    "asset_id": a.asset_id,
                    "name": a.name,
                    "description": a.description,
                    "asset_type": a.asset_type,
                    "status": a.status.value,
                    "monthly_revenue_usd": str(a.monthly_revenue_usd),
                    "monthly_cost_usd": str(a.monthly_cost_usd),
                    "total_revenue_usd": str(a.total_revenue_usd),
                    "development_cost_usd": str(a.development_cost_usd),
                    "break_even_reached": a.break_even_reached,
                    "projected_break_even_days": a.projected_break_even_days,
                    "days_since_deployment": a.days_since_deployment,
                    "is_profitable": a.is_profitable,
                    "deployed_at": a.deployed_at.isoformat() if a.deployed_at else None,
                    "compute_provider": a.compute_provider,
                }
                for a in s.owned_assets
            ],
            "child_instances": [
                {
                    "instance_id": c.instance_id,
                    "niche": c.niche,
                    "status": c.status.value,
                    "seed_capital_usd": str(c.seed_capital_usd),
                    "current_net_worth_usd": str(c.current_net_worth_usd),
                    "current_runway_days": str(c.current_runway_days),
                    "current_efficiency": str(c.current_efficiency),
                    "dividend_rate": str(c.dividend_rate),
                    "total_dividends_paid_usd": str(c.total_dividends_paid_usd),
                    "is_independent": c.is_independent,
                    "spawned_at": c.spawned_at.isoformat(),
                }
                for c in s.child_instances
            ],
            "total_asset_value": str(s.total_asset_value),
            "total_fleet_equity": str(s.total_fleet_equity),
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/v1/oikos/genesis-spark")
async def oikos_genesis_spark() -> dict[str, Any]:
    """Inject the Genesis Trigger into the live organism, waking its metabolism."""
    from genesis_trigger import inject_into_live_organism

    atune = app.state.atune
    evo = app.state.evo
    oikos = app.state.oikos
    synapse = app.state.synapse
    nova = app.state.nova
    axon = app.state.axon
    certificate_manager = getattr(app.state, "certificate_manager", None)

    try:
        result = await inject_into_live_organism(
            atune=atune,
            evo=evo,
            oikos=oikos,
            synapse=synapse,
            nova=nova,
            axon=axon,
            certificate_manager=certificate_manager,
            skip_dream=False,
        )
        phases = result.get("phases", {})
        passed = sum(1 for v in phases.values() if v)
        total = len(phases)
        return {
            "status": "ok",
            "message": f"Genesis complete: {passed}/{total} phases succeeded",
            "phases": phases,
        }
    except Exception as e:
        logger.error("genesis_spark_failed", error=str(e))
        return {
            "status": "error",
            "message": f"Genesis failed: {e}",
            "phases": {},
        }
