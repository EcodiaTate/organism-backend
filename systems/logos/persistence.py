"""
EcodiaOS - Logos: Neo4j Persistence Layer

Persists world model state (generative schemas, causal edges, domain priors,
MDL scores) to Neo4j so intelligence survives restarts.

Node types:
  (:LogosGenerativeSchema {id, name, domain, description, pattern, ...})
  (:LogosCausalEdge {key, cause_id, effect_id, strength, domain, ...})
  (:LogosWorldModelPrior {context_key, variance, sample_count, ...})

All writes are batched via UNWIND for efficiency.
Max 1 transaction per update cycle (constraint).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.logos.world_model import WorldModel

logger = structlog.get_logger("logos.persistence")


class LogosPersistence:
    """
    Neo4j persistence for the Logos world model.

    All writes are batched - call persist_world_model() after each
    WORLD_MODEL_UPDATED emission, not per-schema change.
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j

    # ─── Write: Full world model batch persist ────────────────────

    async def persist_world_model(self, world_model: WorldModel) -> int:
        """Batch-persist all world model state in a single transaction.

        Returns total node count written.
        """
        total = 0
        total += await self._persist_schemas(world_model)
        total += await self._persist_causal_edges(world_model)
        total += await self._persist_priors(world_model)
        total += await self._persist_invariants(world_model)
        total += await self._persist_metrics(world_model)
        logger.info("world_model_persisted", total_nodes=total)
        return total

    async def _persist_schemas(self, world_model: WorldModel) -> int:
        """Batch-upsert generative schemas."""
        schemas = world_model.generative_schemas
        if not schemas:
            return 0

        rows = [
            {
                "id": s.id,
                "name": s.name,
                "domain": s.domain,
                "description": s.description,
                "pattern": str(s.pattern),
                "instance_count": s.instance_count,
                "compression_ratio": s.compression_ratio,
                "created_at": s.created_at.isoformat(),
                "updated_at": utc_now().isoformat(),
            }
            for s in schemas.values()
        ]

        query = """
        UNWIND $rows AS r
        MERGE (s:LogosGenerativeSchema {id: r.id})
        SET s.name = r.name,
            s.domain = r.domain,
            s.description = r.description,
            s.pattern = r.pattern,
            s.instance_count = r.instance_count,
            s.compression_ratio = r.compression_ratio,
            s.created_at = r.created_at,
            s.updated_at = r.updated_at
        """
        await self._neo4j.execute_write(query, {"rows": rows})
        return len(rows)

    async def _persist_causal_edges(self, world_model: WorldModel) -> int:
        """Batch-upsert causal graph edges."""
        links = world_model.causal_structure.links
        if not links:
            return 0

        rows = [
            {
                "key": key,
                "cause_id": link.cause_id,
                "effect_id": link.effect_id,
                "strength": link.strength,
                "domain": link.domain,
                "observations": link.observations,
                "last_observed": link.last_observed.isoformat(),
            }
            for key, link in links.items()
        ]

        query = """
        UNWIND $rows AS r
        MERGE (e:LogosCausalEdge {key: r.key})
        SET e.cause_id = r.cause_id,
            e.effect_id = r.effect_id,
            e.strength = r.strength,
            e.domain = r.domain,
            e.observations = r.observations,
            e.last_observed = r.last_observed
        """
        await self._neo4j.execute_write(query, {"rows": rows})
        return len(rows)

    async def _persist_priors(self, world_model: WorldModel) -> int:
        """Batch-upsert predictive priors."""
        priors = world_model.predictive_priors
        if not priors:
            return 0

        rows = [
            {
                "context_key": prior.context_key,
                "variance": prior.variance,
                "sample_count": prior.sample_count,
                "last_updated": prior.last_updated.isoformat(),
            }
            for prior in priors.values()
        ]

        query = """
        UNWIND $rows AS r
        MERGE (p:LogosWorldModelPrior {context_key: r.context_key})
        SET p.variance = r.variance,
            p.sample_count = r.sample_count,
            p.last_updated = r.last_updated
        """
        await self._neo4j.execute_write(query, {"rows": rows})
        return len(rows)

    async def _persist_invariants(self, world_model: WorldModel) -> int:
        """Batch-upsert empirical invariants."""
        invariants = world_model.empirical_invariants
        if not invariants:
            return 0

        rows = [
            {
                "id": inv.id,
                "statement": inv.statement,
                "domain": inv.domain,
                "observation_count": inv.observation_count,
                "confidence": inv.confidence,
                "created_at": inv.created_at.isoformat(),
                "last_tested": inv.last_tested.isoformat(),
                "source": inv.source,
            }
            for inv in invariants
        ]

        query = """
        UNWIND $rows AS r
        MERGE (i:LogosEmpiricalInvariant {id: r.id})
        SET i.statement = r.statement,
            i.domain = r.domain,
            i.observation_count = r.observation_count,
            i.confidence = r.confidence,
            i.created_at = r.created_at,
            i.last_tested = r.last_tested,
            i.source = r.source
        """
        await self._neo4j.execute_write(query, {"rows": rows})
        return len(rows)

    async def _persist_metrics(self, world_model: WorldModel) -> int:
        """Persist world model metrics snapshot (complexity, coverage, intelligence ratio)."""
        row = {
            "id": "logos_world_model_metrics",
            "current_complexity": world_model.current_complexity,
            "coverage": world_model.coverage,
            "intelligence_ratio": world_model.measure_intelligence_ratio(),
            "schema_count": len(world_model.generative_schemas),
            "causal_link_count": world_model.causal_structure.link_count,
            "prior_count": len(world_model.predictive_priors),
            "invariant_count": len(world_model.empirical_invariants),
            "updated_at": utc_now().isoformat(),
        }

        query = """
        MERGE (m:LogosWorldModelMetrics {id: $row.id})
        SET m.current_complexity = $row.current_complexity,
            m.coverage = $row.coverage,
            m.intelligence_ratio = $row.intelligence_ratio,
            m.schema_count = $row.schema_count,
            m.causal_link_count = $row.causal_link_count,
            m.prior_count = $row.prior_count,
            m.invariant_count = $row.invariant_count,
            m.updated_at = $row.updated_at
        """
        await self._neo4j.execute_write(query, {"row": row})
        return 1

    # ─── Read: Restore world model on startup ─────────────────────

    async def restore_world_model(self, world_model: WorldModel) -> int:
        """Restore world model state from Neo4j on startup.

        Returns total nodes restored.
        """
        total = 0
        total += await self._restore_schemas(world_model)
        total += await self._restore_causal_edges(world_model)
        total += await self._restore_priors(world_model)
        total += await self._restore_invariants(world_model)
        logger.info("world_model_restored", total_nodes=total)
        return total

    async def _restore_schemas(self, world_model: WorldModel) -> int:
        """Restore generative schemas from Neo4j."""
        from systems.logos.types import GenerativeSchema
        from datetime import datetime, UTC

        rows = await self._neo4j.execute_read(
            "MATCH (s:LogosGenerativeSchema) RETURN s"
        )
        for row in rows:
            node = row["s"]
            schema = GenerativeSchema(
                id=node["id"],
                name=node.get("name", ""),
                domain=node.get("domain", ""),
                description=node.get("description", ""),
                instance_count=node.get("instance_count", 0),
                compression_ratio=node.get("compression_ratio", 0.0),
            )
            world_model.generative_schemas[schema.id] = schema

        return len(rows)

    async def _restore_causal_edges(self, world_model: WorldModel) -> int:
        """Restore causal graph edges from Neo4j."""
        from systems.logos.types import CausalLink

        rows = await self._neo4j.execute_read(
            "MATCH (e:LogosCausalEdge) RETURN e"
        )
        for row in rows:
            node = row["e"]
            link = CausalLink(
                cause_id=node["cause_id"],
                effect_id=node["effect_id"],
                strength=node.get("strength", 0.5),
                domain=node.get("domain", ""),
                observations=node.get("observations", 0),
            )
            world_model.causal_structure.add_link(link)

        return len(rows)

    async def _restore_priors(self, world_model: WorldModel) -> int:
        """Restore predictive priors from Neo4j."""
        from systems.logos.types import PriorDistribution

        rows = await self._neo4j.execute_read(
            "MATCH (p:LogosWorldModelPrior) RETURN p"
        )
        for row in rows:
            node = row["p"]
            prior = PriorDistribution(
                context_key=node["context_key"],
                variance=node.get("variance", 1.0),
                sample_count=node.get("sample_count", 0),
            )
            world_model.predictive_priors[prior.context_key] = prior

        return len(rows)

    async def _restore_invariants(self, world_model: WorldModel) -> int:
        """Restore empirical invariants from Neo4j."""
        from systems.logos.types import EmpiricalInvariant

        rows = await self._neo4j.execute_read(
            "MATCH (i:LogosEmpiricalInvariant) RETURN i"
        )
        for row in rows:
            node = row["i"]
            invariant = EmpiricalInvariant(
                id=node["id"],
                statement=node.get("statement", ""),
                domain=node.get("domain", ""),
                observation_count=node.get("observation_count", 0),
                confidence=node.get("confidence", 1.0),
                source=node.get("source", ""),
            )
            world_model.empirical_invariants.append(invariant)

        return len(rows)

    # ─── Fitness Time-Series (SG3/SG4: Mitosis + evolutionary activity) ─────────

    async def persist_fitness_record(
        self,
        record: "Any",
    ) -> None:
        """Append an immutable fitness snapshot to the time-series.

        Each call creates a new (:LogosFitnessTimeSeries) node - records are
        never updated.  This enables Mitosis to rank instances by IR trajectory
        and Benchmarks to compute Bedau-Packard evolutionary activity.

        Spec 21 SG3/SG4: population-level compression fitness via Neo4j append log.
        """
        query = """
        CREATE (f:LogosFitnessTimeSeries {
            instance_id:            $instance_id,
            intelligence_ratio:     $intelligence_ratio,
            compression_efficiency: $compression_efficiency,
            world_model_coverage:   $world_model_coverage,
            cognitive_pressure:     $cognitive_pressure,
            schema_count:           $schema_count,
            anchor_count:           $anchor_count,
            schwarzschild_met:      $schwarzschild_met,
            timestamp:              $timestamp
        })
        """
        await self._neo4j.execute_write(query, {
            "instance_id":            record.instance_id,
            "intelligence_ratio":     record.intelligence_ratio,
            "compression_efficiency": record.compression_efficiency,
            "world_model_coverage":   record.world_model_coverage,
            "cognitive_pressure":     record.cognitive_pressure,
            "schema_count":           record.schema_count,
            "anchor_count":           record.anchor_count,
            "schwarzschild_met":      record.schwarzschild_met,
            "timestamp":              record.timestamp.isoformat(),
        })

    # ─── WorldModel Integration: (:WorldModel) node + [:COMPRESSES] ────────────

    async def persist_integration(
        self,
        update: "Any",
        delta: "Any",
        *,
        source_episode_ids: list[str] | None = None,
        source_semantic_ids: list[str] | None = None,
    ) -> str:
        """Write a (:WorldModel) node for one integration event.

        Links source Episode/Semantic nodes via [:COMPRESSES] relationships and
        updates the self-prediction index singleton so Fovea and Schwarzschild
        can read prediction accuracy without a full world-model scan.

        Returns the new WorldModel node ID.

        Spec 21 §CRITICAL-1: the Neo4j half of the Compression Cascade.
        """
        from primitives.common import new_id

        wm_node_id = new_id()
        now = utc_now().isoformat()

        # 1. Create the (:WorldModel) event node
        await self._neo4j.execute_write("""
        CREATE (wm:WorldModel {
            id:                  $id,
            update_type:         $update_type,
            schemas_added:       $schemas_added,
            schemas_extended:    $schemas_extended,
            priors_updated:      $priors_updated,
            causal_links_added:  $causal_links_added,
            invariants_violated: $invariants_violated,
            complexity_delta:    $complexity_delta,
            coverage_delta:      $coverage_delta,
            information_content: $information_content,
            created_at:          $created_at
        })
        """, {
            "id":                  wm_node_id,
            "update_type":         update.update_type.value,
            "schemas_added":       update.schemas_added,
            "schemas_extended":    update.schemas_extended,
            "priors_updated":      update.priors_updated,
            "causal_links_added":  update.causal_links_added,
            "invariants_violated": update.invariants_violated,
            "complexity_delta":    update.complexity_delta,
            "coverage_delta":      update.coverage_delta,
            "information_content": getattr(delta, "information_content", 0.0),
            "created_at":          now,
        })

        # 2. [:COMPRESSES] → source Episode nodes
        if source_episode_ids:
            await self._neo4j.execute_write("""
            UNWIND $ids AS src_id
            MATCH (ep:Episode {id: src_id})
            MATCH (wm:WorldModel {id: $wm_id})
            MERGE (wm)-[:COMPRESSES {created_at: $now}]->(ep)
            """, {"ids": source_episode_ids, "wm_id": wm_node_id, "now": now})

        # 3. [:COMPRESSES] → source Semantic nodes
        if source_semantic_ids:
            await self._neo4j.execute_write("""
            UNWIND $ids AS src_id
            MATCH (sn:SemanticNode {id: src_id})
            MATCH (wm:WorldModel {id: $wm_id})
            MERGE (wm)-[:COMPRESSES {created_at: $now}]->(sn)
            """, {"ids": source_semantic_ids, "wm_id": wm_node_id, "now": now})

        # 4. Upsert the self-prediction index singleton - O(1) read for
        #    Fovea and Schwarzschild without traversing the full world model.
        await self._neo4j.execute_write("""
        MERGE (idx:LogosSelfPredictionIndex {id: 'singleton'})
        SET idx.last_wm_node_id       = $wm_id,
            idx.last_update_type      = $update_type,
            idx.last_coverage_delta   = $coverage_delta,
            idx.last_complexity_delta = $complexity_delta,
            idx.updated_at            = $now
        """, {
            "wm_id":            wm_node_id,
            "update_type":      update.update_type.value,
            "coverage_delta":   update.coverage_delta,
            "complexity_delta": update.complexity_delta,
            "now":              now,
        })

        logger.info(
            "world_model_integration_persisted",
            wm_node_id=wm_node_id,
            update_type=update.update_type.value,
            source_episodes=len(source_episode_ids or []),
            source_semantics=len(source_semantic_ids or []),
        )
        return wm_node_id

    # ─── Eviction Audit (immutable, append-only) ──────────────────

    async def log_eviction(
        self,
        item_id: str,
        reason: str,
        bits_freed: float,
        trigger: str = "decay_cycle",
    ) -> None:
        """SG2: Immutable eviction audit log (append-only Neo4j node, never updated)."""
        query = """
        CREATE (e:LogosEvictionAudit {
            item_id: $item_id,
            reason: $reason,
            bits_freed: $bits_freed,
            trigger: $trigger,
            timestamp: $timestamp
        })
        """
        await self._neo4j.execute_write(query, {
            "item_id": item_id,
            "reason": reason,
            "bits_freed": bits_freed,
            "trigger": trigger,
            "timestamp": utc_now().isoformat(),
        })
