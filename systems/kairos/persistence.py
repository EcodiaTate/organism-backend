"""
EcodiaOS — Kairos: Neo4j Persistence

Persists CausalInvariant nodes and CausalNode/CAUSES relationships to Neo4j.
Restores invariants on startup so the hierarchy survives process restarts.

Schema:
  (:CausalInvariant {id, cause, effect, abstract_form, confidence, tier,
                     hold_rate, validated, direction, recency_weight, active,
                     distilled, is_minimal, is_tautological, created_at})
  (:CausalNode {name})-[:CAUSES {confidence, validated, invariant_id}]->(:CausalNode)

All writes are batched via UNWIND to avoid per-invariant transaction overhead.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives.causal import (
    ApplicableDomain,
    CausalInvariant,
    CausalInvariantTier,
    ScopeCondition,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger("kairos.persistence")

# ─── Schema ───────────────────────────────────────────────────────────

CONSTRAINTS = [
    "CREATE CONSTRAINT kairos_invariant_id IF NOT EXISTS "
    "FOR (ci:CausalInvariant) REQUIRE ci.id IS UNIQUE",
    "CREATE CONSTRAINT kairos_causal_node_name IF NOT EXISTS "
    "FOR (cn:CausalNode) REQUIRE cn.name IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX kairos_invariant_tier IF NOT EXISTS "
    "FOR (ci:CausalInvariant) ON (ci.tier)",
    "CREATE INDEX kairos_invariant_active IF NOT EXISTS "
    "FOR (ci:CausalInvariant) ON (ci.active)",
]


async def ensure_schema(neo4j: Neo4jClient) -> None:
    """Idempotent schema setup on startup."""
    for statement in CONSTRAINTS + INDEXES:
        try:
            await neo4j.execute_write(statement)
        except Exception as e:
            if "already exists" in str(e).lower():
                continue
            logger.warning("kairos_schema_statement_warning", statement=statement[:80])


# ─── Serialization ────────────────────────────────────────────────────


def _invariant_to_props(inv: CausalInvariant) -> dict[str, Any]:
    """Flatten a CausalInvariant to Neo4j-safe scalar properties."""
    # Parse cause/effect from abstract form
    cause, effect = "", ""
    parts = inv.abstract_form.split(" causes ")
    if len(parts) == 2:
        cause = parts[0].strip()
        effect = parts[1].strip()

    return {
        "id": inv.id,
        "tier": inv.tier.value,
        "abstract_form": inv.abstract_form,
        "cause": cause,
        "effect": effect,
        "invariance_hold_rate": inv.invariance_hold_rate,
        "confidence": inv.invariance_hold_rate,
        "direction": inv.direction,
        "validated": inv.validated,
        "active": inv.active,
        "recency_weight": inv.recency_weight,
        "distilled": inv.distilled,
        "is_minimal": inv.is_minimal,
        "is_tautological": inv.is_tautological,
        "intelligence_ratio_contribution": inv.intelligence_ratio_contribution,
        "description_length_bits": inv.description_length_bits,
        "source_rule_id": inv.source_rule_id,
        "violation_count": inv.violation_count,
        "refined_scope": inv.refined_scope,
        "concrete_instances_json": json.dumps(inv.concrete_instances),
        "applicable_domains_json": json.dumps(
            [d.model_dump() for d in inv.applicable_domains]
        ),
        "scope_conditions_json": json.dumps(
            [sc.model_dump() for sc in inv.scope_conditions]
        ),
        "variable_roles_json": json.dumps(inv.variable_roles),
        "untested_domains_json": json.dumps(inv.untested_domains),
        "created_at": inv.created_at.isoformat() if inv.created_at else None,
    }


def _row_to_invariant(row: dict[str, Any]) -> CausalInvariant | None:
    """Reconstruct a CausalInvariant from a Neo4j row."""
    try:
        g: dict[str, Any] = row.get("ci", row)

        # Parse JSON fields
        concrete_instances = json.loads(g.get("concrete_instances_json", "[]") or "[]")
        applicable_domains_raw = json.loads(
            g.get("applicable_domains_json", "[]") or "[]"
        )
        applicable_domains = [ApplicableDomain(**d) for d in applicable_domains_raw]
        scope_conditions_raw = json.loads(
            g.get("scope_conditions_json", "[]") or "[]"
        )
        scope_conditions = [ScopeCondition(**sc) for sc in scope_conditions_raw]
        variable_roles = json.loads(g.get("variable_roles_json", "{}") or "{}")
        untested_domains = json.loads(g.get("untested_domains_json", "[]") or "[]")

        # Parse created_at
        created_at = None
        raw_ts = g.get("created_at")
        if raw_ts is not None:
            if hasattr(raw_ts, "to_native"):
                created_at = raw_ts.to_native()
            elif isinstance(raw_ts, str):
                from datetime import datetime

                created_at = datetime.fromisoformat(raw_ts)

        tier_val = int(g.get("tier", 1))

        return CausalInvariant(
            id=str(g["id"]),
            tier=CausalInvariantTier(tier_val),
            abstract_form=g.get("abstract_form", ""),
            concrete_instances=concrete_instances,
            applicable_domains=applicable_domains,
            invariance_hold_rate=float(g.get("invariance_hold_rate", 0.0)),
            scope_conditions=scope_conditions,
            intelligence_ratio_contribution=float(
                g.get("intelligence_ratio_contribution", 0.0)
            ),
            description_length_bits=float(g.get("description_length_bits", 0.0)),
            source_rule_id=g.get("source_rule_id", ""),
            direction=g.get("direction", ""),
            validated=bool(g.get("validated", False)),
            active=bool(g.get("active", True)),
            recency_weight=float(g.get("recency_weight", 1.0)),
            distilled=bool(g.get("distilled", False)),
            is_minimal=bool(g.get("is_minimal", False)),
            is_tautological=bool(g.get("is_tautological", False)),
            variable_roles=variable_roles,
            untested_domains=untested_domains,
            violation_count=int(g.get("violation_count", 0)),
            refined_scope=g.get("refined_scope", ""),
            **({"created_at": created_at} if created_at else {}),
        )
    except Exception as exc:
        logger.warning(
            "kairos_invariant_row_parse_failed",
            error=str(exc),
            row_id=row.get("ci", row).get("id", "?"),
        )
        return None


# ─── Batch Write ──────────────────────────────────────────────────────


async def persist_invariants_batch(
    neo4j: Neo4jClient,
    invariants: list[CausalInvariant],
) -> int:
    """
    Batch-persist invariants to Neo4j using UNWIND.

    Also creates CausalNode/CAUSES relationships for validated invariants.
    Returns count of invariants persisted.
    """
    if not invariants:
        return 0

    props_list = [_invariant_to_props(inv) for inv in invariants]

    # Batch MERGE invariant nodes
    await neo4j.execute_write(
        """
        UNWIND $batch AS props
        MERGE (ci:CausalInvariant {id: props.id})
        SET ci += props, ci.persisted_at = datetime()
        """,
        {"batch": props_list},
    )

    # Create CausalNode + CAUSES relationships for validated invariants
    causal_edges = []
    for inv in invariants:
        parts = inv.abstract_form.split(" causes ")
        if len(parts) != 2:
            continue
        cause_name = parts[0].strip()
        effect_name = parts[1].strip()
        if cause_name and effect_name:
            causal_edges.append(
                {
                    "cause": cause_name,
                    "effect": effect_name,
                    "confidence": inv.invariance_hold_rate,
                    "validated": inv.validated,
                    "invariant_id": inv.id,
                }
            )

    if causal_edges:
        await neo4j.execute_write(
            """
            UNWIND $edges AS e
            MERGE (c:CausalNode {name: e.cause})
            MERGE (ef:CausalNode {name: e.effect})
            MERGE (c)-[r:CAUSES {invariant_id: e.invariant_id}]->(ef)
            SET r.confidence = e.confidence, r.validated = e.validated
            """,
            {"edges": causal_edges},
        )

    logger.info(
        "kairos_invariants_persisted",
        count=len(invariants),
        causal_edges=len(causal_edges),
    )

    return len(invariants)


# ─── Restore on Startup ──────────────────────────────────────────────


async def restore_invariants(neo4j: Neo4jClient) -> list[CausalInvariant]:
    """
    Restore all active invariants from Neo4j on startup.

    Must complete before the first pipeline cycle.
    """
    rows = await neo4j.execute_read(
        "MATCH (ci:CausalInvariant) WHERE ci.active = true RETURN ci ORDER BY ci.tier DESC",
        {},
    )

    invariants: list[CausalInvariant] = []
    for row in rows:
        inv = _row_to_invariant(row)
        if inv is not None:
            invariants.append(inv)

    logger.info(
        "kairos_invariants_restored",
        count=len(invariants),
        tier1=sum(1 for i in invariants if i.tier == CausalInvariantTier.TIER_1_DOMAIN),
        tier2=sum(
            1
            for i in invariants
            if i.tier == CausalInvariantTier.TIER_2_CROSS_DOMAIN
        ),
        tier3=sum(
            1
            for i in invariants
            if i.tier == CausalInvariantTier.TIER_3_SUBSTRATE
        ),
    )

    return invariants
