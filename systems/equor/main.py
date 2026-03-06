"""
EcodiaOS — Application Entry Point

FastAPI application with the startup sequence defined in the
Infrastructure Architecture specification.

`docker compose up` → uvicorn ecodiaos.main:app
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from clients.embedding import create_embedding_client
from clients.llm import create_llm_provider
from clients.neo4j import Neo4jClient
from clients.redis import RedisClient
from clients.timescaledb import TimescaleDBClient
from config import load_config, load_seed
from systems.equor.service import EquorService
from systems.memory.service import MemoryService
from telemetry.logging import setup_logging
from telemetry.metrics import MetricCollector

logger = structlog.get_logger()


# ─── Application State ───────────────────────────────────────────
# These are set during startup and accessible via app.state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown sequence.
    Follows the Infrastructure Architecture spec section 3.2.
    """
    # ── 1. Load configuration ─────────────────────────────────
    config_path = os.environ.get("ECODIAOS_CONFIG_PATH", "config/default.yaml")
    config = load_config(config_path)
    app.state.config = config

    # ── 2. Set up logging ─────────────────────────────────────
    setup_logging(config.logging, instance_id=config.instance_id)
    logger.info(
        "ecodiaos_starting",
        instance_id=config.instance_id,
        config_path=config_path,
    )

    # ── 3. Connect to data stores ─────────────────────────────
    neo4j_client = Neo4jClient(config.neo4j)
    await neo4j_client.connect()
    app.state.neo4j = neo4j_client

    tsdb_client = TimescaleDBClient(config.timescaledb)
    await tsdb_client.connect()
    app.state.tsdb = tsdb_client

    redis_client = RedisClient(config.redis)
    await redis_client.connect()
    app.state.redis = redis_client

    # ── 4. Initialize LLM and embedding clients ───────────────
    llm_client = create_llm_provider(config.llm)
    app.state.llm = llm_client

    embedding_client = create_embedding_client(config.embedding)
    app.state.embedding = embedding_client

    # ── 5. Initialize Memory service ──────────────────────────
    memory = MemoryService(neo4j_client, embedding_client)
    await memory.initialize()
    app.state.memory = memory

    # ── 6. Initialize Equor (Constitution & Ethics) ───────────
    governance_config = _resolve_governance_config(config)
    equor = EquorService(
        neo4j=neo4j_client,
        llm=llm_client,
        config=config.equor,
        governance_config=governance_config,
    )
    await equor.initialize()
    app.state.equor = equor

    # ── 7. Initialize telemetry ───────────────────────────────
    metrics = MetricCollector(tsdb_client)
    await metrics.start_writer()
    app.state.metrics = metrics

    # ── 8. Check for existing instance or birth new one ───────
    instance = await memory.get_self()
    if instance is None:
        seed_path = os.environ.get("ECODIAOS_SEED_PATH", "config/seeds/example_seed.yaml")
        try:
            seed = load_seed(seed_path)
            birth_result = await memory.birth(seed, config.instance_id)
            logger.info("instance_born", **birth_result)
            # Re-seed invariants after birth (constitution now exists)
            await equor.initialize()
        except FileNotFoundError:
            logger.warning(
                "no_seed_found",
                seed_path=seed_path,
                message="Instance not born. Provide a seed config to create one.",
            )
    else:
        logger.info(
            "instance_loaded",
            name=instance.name,
            instance_id=instance.instance_id,
            cycle_count=instance.cycle_count,
            episodes=instance.total_episodes,
            entities=instance.total_entities,
        )

    # ── 9. Phase 2 complete ───────────────────────────────────
    # Future phases will add:
    #   atune = AtuneService(memory, embedding_client, config.atune)
    #   voxis = VoxisService(memory, llm_client, config.voxis)
    #   nova = NovaService(memory, equor, llm_client, config.nova)
    #   axon = AxonService(memory, config.axon)
    #   evo = EvoService(memory, llm_client, config.evo)
    #   simula = SimulaService(memory, config.simula)
    #   synapse = SynapseService(...)
    #   asyncio.create_task(synapse.start_clock())

    logger.info("ecodiaos_ready", phase="2_constitution_ethics")

    yield

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("ecodiaos_shutting_down")
    await metrics.stop()
    await embedding_client.close()
    await llm_client.close()
    await redis_client.close()
    await tsdb_client.close()
    await neo4j_client.close()
    logger.info("ecodiaos_shutdown_complete")


def _resolve_governance_config(config: Any) -> Any:
    """Resolve governance config from seed or use defaults."""
    from config import GovernanceConfig
    try:
        seed_path = os.environ.get("ECODIAOS_SEED_PATH", "config/seeds/example_seed.yaml")
        seed = load_seed(seed_path)
        return seed.constitution.governance
    except Exception:
        return GovernanceConfig()


# ─── FastAPI Application ─────────────────────────────────────────

app = FastAPI(
    title="EcodiaOS",
    description="A living digital organism — API surface",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health & Status Endpoints ────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, Any]:
    """System health check."""
    memory_health = await app.state.memory.health()
    equor_health = await app.state.equor.health()
    neo4j_health = await app.state.neo4j.health_check()
    tsdb_health = await app.state.tsdb.health_check()
    redis_health = await app.state.redis.health_check()

    overall = "healthy"
    if any(
        h.get("status") != "connected"
        for h in [neo4j_health, tsdb_health, redis_health]
    ):
        overall = "degraded"
    if equor_health.get("safe_mode"):
        overall = "degraded"

    instance = await app.state.memory.get_self()

    return {
        "status": overall,
        "instance_id": app.state.config.instance_id,
        "instance_name": instance.name if instance else "unborn",
        "phase": "2_constitution_ethics",
        "systems": {
            "memory": memory_health,
            "equor": equor_health,
        },
        "data_stores": {
            "neo4j": neo4j_health,
            "timescaledb": tsdb_health,
            "redis": redis_health,
        },
    }


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


# ─── Phase 1: Memory Test Endpoints ──────────────────────────────


@app.post("/api/v1/perceive/event")
async def perceive_event(body: dict[str, Any]):
    """
    Ingest a percept (temporary test endpoint).
    In later phases, this goes through Atune's full pipeline.
    """
    from primitives import Percept

    text = body.get("text", body.get("content", ""))
    if not text:
        return {"error": "No text/content provided"}

    percept = Percept.from_user_message(text)
    episode_id = await app.state.memory.store_percept(percept)

    return {
        "episode_id": episode_id,
        "stored": True,
    }


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
    Propose a constitutional amendment (legacy endpoint).
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


# ─── Amendment Pipeline Endpoints ──────────────────────────────────


@app.post("/api/v1/governance/amendments/submit")
async def submit_amendment_endpoint(body: dict[str, Any]):
    """
    Submit a constitutional amendment with evidence requirements.

    Body: {
        proposed_drives: {coherence, care, growth, honesty},
        title, description, rationale, proposer_id,
        evidence_hypothesis_ids: [str]
    }
    """
    required = [
        "proposed_drives", "title", "description", "rationale",
        "proposer_id", "evidence_hypothesis_ids",
    ]
    for field in required:
        if field not in body:
            return {"error": f"Missing required field: {field}"}

    return await app.state.equor.submit_amendment_proposal(
        proposed_drives=body["proposed_drives"],
        title=body["title"],
        description=body["description"],
        rationale=body["rationale"],
        proposer_id=body["proposer_id"],
        evidence_hypothesis_ids=body["evidence_hypothesis_ids"],
    )


@app.post("/api/v1/governance/amendments/{proposal_id}/shadow")
async def start_shadow_endpoint(proposal_id: str):
    """Start the shadow period for an amendment (after deliberation ends)."""
    return await app.state.equor.start_amendment_shadow(proposal_id)


@app.get("/api/v1/governance/amendments/shadow/status")
async def shadow_status_endpoint():
    """Get the current shadow period status."""
    status = await app.state.equor.get_shadow_status()
    if status is None:
        return {"active": False}
    return {"active": True, **status}


@app.post("/api/v1/governance/amendments/{proposal_id}/vote")
async def cast_vote_endpoint(proposal_id: str, body: dict[str, Any]):
    """
    Cast a vote on an amendment.
    Body: {voter_id, vote: "for"|"against"|"abstain"}
    """
    if "voter_id" not in body or "vote" not in body:
        return {"error": "Missing required fields: voter_id, vote"}

    return await app.state.equor.cast_amendment_vote(
        proposal_id=proposal_id,
        voter_id=body["voter_id"],
        vote=body["vote"],
    )


@app.post("/api/v1/governance/amendments/{proposal_id}/open-voting")
async def open_voting_endpoint(proposal_id: str):
    """Open an amendment for community voting (after shadow passes)."""
    return await app.state.equor.open_amendment_voting(proposal_id)


@app.post("/api/v1/governance/amendments/{proposal_id}/tally")
async def tally_votes_endpoint(proposal_id: str, body: dict[str, Any]):
    """
    Tally votes and determine pass/fail.
    Body: {total_eligible_voters: int}
    """
    if "total_eligible_voters" not in body:
        return {"error": "Missing required field: total_eligible_voters"}

    return await app.state.equor.tally_amendment_votes(
        proposal_id=proposal_id,
        total_eligible_voters=body["total_eligible_voters"],
    )


@app.post("/api/v1/governance/amendments/{proposal_id}/adopt")
async def adopt_amendment_endpoint(proposal_id: str):
    """Adopt a passed amendment into the constitution."""
    return await app.state.equor.adopt_passed_amendment(proposal_id)


@app.get("/api/v1/governance/amendments/{proposal_id}/status")
async def amendment_status_endpoint(proposal_id: str):
    """Get the full pipeline status for an amendment."""
    status = await app.state.equor.get_amendment_pipeline_status(proposal_id)
    if status is None:
        return {"error": "Proposal not found"}
    return status
