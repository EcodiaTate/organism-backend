"""
EcodiaOS - Nova Genome Extraction & Seeding

Implements GenomeExtractionProtocol for the Nova decision-and-planning system.

Heritable state:
  1. Persistent beliefs - high-confidence entity beliefs that represent
     well-evidenced world knowledge (confidence > 0.7, top 200).
  2. Goal priors - per-domain completion statistics showing which kinds of
     goals tend to succeed, informing goal priority bootstrapping.
  3. Planning heuristics - learned action preferences derived from decision
     records (which policy types were selected and succeeded).

Payload format:
    {
        "beliefs": [
            {"entity_id": str, "name": str, "entity_type": str,
             "confidence": float, "properties": dict},
            ...
        ],
        "goal_priors": [
            {"target_domain": str, "source": str, "total": int,
             "achieved": int, "abandoned": int, "achievement_rate": float},
            ...
        ],
        "planning_heuristics": [
            {"policy_name": str, "path": str, "times_selected": int,
             "times_dispatched": int, "dispatch_rate": float},
            ...
        ],
    }

Integration:
    MitosisEngine calls extract_genome_segment() before spawning a child.
    birth.py calls seed_from_genome_segment() after child instance starts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# ─── Extraction Limits ───────────────────────────────────────────────────────

_MAX_BELIEFS: int = 200
_MIN_BELIEF_CONFIDENCE: float = 0.7
_MAX_GOAL_PRIOR_DOMAINS: int = 100
_MAX_HEURISTIC_ENTRIES: int = 100

# Confidence discount applied to seeded beliefs (child starts slightly less certain)
_SEEDED_BELIEF_CONFIDENCE_DISCOUNT: float = 0.85


class NovaGenomeExtractor:
    """
    Extracts and seeds Nova's heritable state via Neo4j.

    Implements GenomeExtractionProtocol:
        extract_genome_segment() -> OrganGenomeSegment
        seed_from_genome_segment(segment) -> bool
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._log = logger.bind(subsystem="nova.genome")

    # ─── GenomeExtractionProtocol ────────────────────────────────────────────

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """
        Serialise Nova's heritable state into an OrganGenomeSegment.

        Queries Neo4j for:
          - High-confidence entity beliefs (confidence > 0.7, top 200)
          - Goal completion statistics grouped by target_domain + source
          - Decision records showing selected policy preferences

        Returns an empty segment (version=0) if no beliefs exist.
        """
        beliefs = await self._extract_beliefs()
        goal_priors = await self._extract_goal_priors()
        planning_heuristics = await self._extract_planning_heuristics()
        efe_weights = await self._extract_efe_weights()
        world_model_summary = await self._extract_world_model_summary()

        if not beliefs:
            self._log.info("nova_genome_empty", reason="no_high_confidence_beliefs")
            return build_segment(
                system_id=SystemID.NOVA,
                payload={
                    "beliefs": [],
                    "goal_priors": [],
                    "planning_heuristics": [],
                    "efe_weights": efe_weights,
                    "world_model_summary": world_model_summary,
                },
                version=0,
            )

        payload: dict[str, Any] = {
            "beliefs": beliefs,
            "goal_priors": goal_priors,
            "planning_heuristics": planning_heuristics,
            "efe_weights": efe_weights,
            "world_model_summary": world_model_summary,
        }

        segment = build_segment(
            system_id=SystemID.NOVA,
            payload=payload,
            version=2,
        )

        self._log.info(
            "nova_genome_extracted",
            beliefs_count=len(beliefs),
            goal_prior_domains=len(goal_priors),
            heuristic_entries=len(planning_heuristics),
            has_efe_weights=bool(efe_weights),
            size_bytes=segment.size_bytes,
        )

        return segment

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """
        Pre-populate Nova's Neo4j state from a parent's genome segment.

        Steps:
          1. Verify payload_hash integrity.
          2. Check schema_version compatibility.
          3. Seed entity belief nodes (confidence discounted by 0.85x).
          4. Seed goal prior nodes for priority bootstrapping.
          5. Seed EFE weights (parent's learned policy selection preferences).
          6. Seed world model summary (parent's cognitive state overview).

        Returns True on success, False on any validation or persistence failure.
        """
        if not verify_segment(segment):
            self._log.error("nova_genome_seed_hash_mismatch", system_id=segment.system_id)
            return False

        if not check_schema_version(segment):
            self._log.error(
                "nova_genome_seed_schema_incompatible",
                schema_version=segment.schema_version,
            )
            return False

        payload = segment.payload
        beliefs: list[dict[str, Any]] = payload.get("beliefs", [])
        goal_priors: list[dict[str, Any]] = payload.get("goal_priors", [])
        efe_weights: dict[str, float] = payload.get("efe_weights", {})
        world_model_summary: dict[str, Any] = payload.get("world_model_summary", {})

        beliefs_seeded = await self._seed_beliefs(beliefs)
        priors_seeded = await self._seed_goal_priors(goal_priors)
        efe_seeded = await self._seed_efe_weights(efe_weights)
        wm_seeded = await self._seed_world_model_summary(world_model_summary)

        self._log.info(
            "nova_genome_seeded",
            beliefs_seeded=beliefs_seeded,
            priors_seeded=priors_seeded,
            efe_weights_seeded=efe_seeded,
            world_model_seeded=wm_seeded,
            segment_version=segment.version,
        )

        return True

    # ─── Extraction Helpers ──────────────────────────────────────────────────

    async def _extract_beliefs(self) -> list[dict[str, Any]]:
        """
        Query high-confidence entity beliefs from Neo4j.

        Returns dicts with entity_id, name, entity_type, confidence, properties.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (b:EntityBelief)
                WHERE b.confidence > $min_confidence
                RETURN b.entity_id AS entity_id,
                       b.name AS name,
                       b.entity_type AS entity_type,
                       b.confidence AS confidence,
                       b.properties_json AS properties_json
                ORDER BY b.confidence DESC
                LIMIT $limit
                """,
                {"min_confidence": _MIN_BELIEF_CONFIDENCE, "limit": _MAX_BELIEFS},
            )
        except Exception as exc:
            self._log.warning("nova_genome_extract_beliefs_failed", error=str(exc))
            return []

        results: list[dict[str, Any]] = []
        for row in rows:
            props_raw = row.get("properties_json", "{}")
            properties: dict[str, Any] = {}
            if isinstance(props_raw, str):
                try:
                    import json

                    properties = json.loads(props_raw)
                except (ValueError, TypeError):
                    properties = {}
            elif isinstance(props_raw, dict):
                properties = props_raw

            results.append({
                "entity_id": str(row.get("entity_id", "")),
                "name": str(row.get("name", "")),
                "entity_type": str(row.get("entity_type", "")),
                "confidence": float(row.get("confidence", 0.0)),
                "properties": properties,
            })

        return results

    async def _extract_goal_priors(self) -> list[dict[str, Any]]:
        """
        Aggregate goal completion stats grouped by target_domain and source.

        Returns per-domain achievement rates so the child knows which goal
        categories tend to succeed in this organism's niche.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (g:Goal)
                WHERE g.status IN ['achieved', 'abandoned', 'active', 'suspended']
                WITH g.target_domain AS domain,
                     g.source AS source,
                     count(g) AS total,
                     sum(CASE WHEN g.status = 'achieved' THEN 1 ELSE 0 END) AS achieved,
                     sum(CASE WHEN g.status = 'abandoned' THEN 1 ELSE 0 END) AS abandoned
                WHERE total >= 2
                RETURN domain, source, total, achieved, abandoned
                ORDER BY total DESC
                LIMIT $limit
                """,
                {"limit": _MAX_GOAL_PRIOR_DOMAINS},
            )
        except Exception as exc:
            self._log.warning("nova_genome_extract_goal_priors_failed", error=str(exc))
            return []

        results: list[dict[str, Any]] = []
        for row in rows:
            total = int(row.get("total", 0))
            achieved = int(row.get("achieved", 0))
            abandoned = int(row.get("abandoned", 0))
            achievement_rate = achieved / total if total > 0 else 0.0

            results.append({
                "target_domain": str(row.get("domain", "")),
                "source": str(row.get("source", "")),
                "total": total,
                "achieved": achieved,
                "abandoned": abandoned,
                "achievement_rate": round(achievement_rate, 4),
            })

        return results

    async def _extract_planning_heuristics(self) -> list[dict[str, Any]]:
        """
        Extract action preferences from persisted decision records.

        Groups by selected_policy_name and deliberation path to reveal which
        policy types the organism prefers and how often they lead to dispatch.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (d:DecisionRecord)
                WHERE d.selected_policy_name IS NOT NULL
                  AND d.selected_policy_name <> ''
                WITH d.selected_policy_name AS policy_name,
                     d.path AS path,
                     count(d) AS times_selected,
                     sum(CASE WHEN d.intent_dispatched = true THEN 1 ELSE 0 END) AS times_dispatched
                WHERE times_selected >= 2
                RETURN policy_name, path, times_selected, times_dispatched
                ORDER BY times_selected DESC
                LIMIT $limit
                """,
                {"limit": _MAX_HEURISTIC_ENTRIES},
            )
        except Exception as exc:
            self._log.warning("nova_genome_extract_heuristics_failed", error=str(exc))
            return []

        results: list[dict[str, Any]] = []
        for row in rows:
            times_selected = int(row.get("times_selected", 0))
            times_dispatched = int(row.get("times_dispatched", 0))
            dispatch_rate = (
                times_dispatched / times_selected if times_selected > 0 else 0.0
            )

            results.append({
                "policy_name": str(row.get("policy_name", "")),
                "path": str(row.get("path", "")),
                "times_selected": times_selected,
                "times_dispatched": times_dispatched,
                "dispatch_rate": round(dispatch_rate, 4),
            })

        return results

    async def _extract_efe_weights(self) -> dict[str, float]:
        """
        Extract current EFE weight vector from persisted decision records.

        Falls back to default weights if no data is available. These weights
        represent the organism's learned preference for how to balance
        pragmatic, epistemic, constitutional, feasibility, risk, and
        cognition cost when selecting policies.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (d:DecisionRecord)
                WHERE d.efe_pragmatic IS NOT NULL
                WITH avg(d.efe_pragmatic) AS pragmatic,
                     avg(d.efe_epistemic) AS epistemic,
                     avg(d.efe_constitutional) AS constitutional,
                     avg(d.efe_feasibility) AS feasibility,
                     avg(d.efe_risk) AS risk,
                     avg(d.efe_cognition_cost) AS cognition_cost
                RETURN pragmatic, epistemic, constitutional,
                       feasibility, risk, cognition_cost
                """,
                {},
            )
            if rows:
                row = rows[0]
                return {
                    "pragmatic": float(row.get("pragmatic", 0.35) or 0.35),
                    "epistemic": float(row.get("epistemic", 0.20) or 0.20),
                    "constitutional": float(row.get("constitutional", 0.20) or 0.20),
                    "feasibility": float(row.get("feasibility", 0.15) or 0.15),
                    "risk": float(row.get("risk", 0.10) or 0.10),
                    "cognition_cost": float(row.get("cognition_cost", 0.10) or 0.10),
                }
        except Exception as exc:
            self._log.debug("nova_genome_extract_efe_weights_failed", error=str(exc))

        # Default weights
        return {
            "pragmatic": 0.35,
            "epistemic": 0.20,
            "constitutional": 0.20,
            "feasibility": 0.15,
            "risk": 0.10,
            "cognition_cost": 0.10,
        }

    async def _extract_world_model_summary(self) -> dict[str, Any]:
        """
        Extract a summary of the current world model state for genome.

        Includes aggregate statistics rather than raw beliefs (which are
        exported separately). This gives the child an overview of the
        parent's cognitive state at birth.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (b:EntityBelief)
                WITH count(b) AS total,
                     avg(b.confidence) AS mean_confidence,
                     collect(DISTINCT b.entity_type) AS entity_types
                RETURN total, mean_confidence, entity_types
                """,
                {},
            )
            if rows:
                row = rows[0]
                return {
                    "total_beliefs": int(row.get("total", 0)),
                    "mean_confidence": round(float(row.get("mean_confidence", 0.0) or 0.0), 4),
                    "entity_types": list(row.get("entity_types", [])),
                }
        except Exception as exc:
            self._log.debug("nova_genome_extract_world_model_failed", error=str(exc))

        return {"total_beliefs": 0, "mean_confidence": 0.0, "entity_types": []}

    # ─── Seeding Helpers ─────────────────────────────────────────────────────

    async def _seed_beliefs(self, beliefs: list[dict[str, Any]]) -> int:
        """
        Pre-populate EntityBelief nodes in the child's Neo4j.

        Confidence is discounted by _SEEDED_BELIEF_CONFIDENCE_DISCOUNT so the
        child treats inherited beliefs as slightly less certain than the parent did.
        """
        import json

        from primitives.common import new_id, utc_now

        seeded = 0
        now_iso = utc_now().isoformat()

        for belief in beliefs:
            try:
                raw_confidence = float(belief.get("confidence", 0.5))
                discounted = round(raw_confidence * _SEEDED_BELIEF_CONFIDENCE_DISCOUNT, 4)
                properties = belief.get("properties", {})
                properties_json = json.dumps(properties, sort_keys=True, default=str)

                await self._neo4j.execute_write(
                    """
                    CREATE (:EntityBelief {
                        id: $id,
                        entity_id: $entity_id,
                        name: $name,
                        entity_type: $entity_type,
                        confidence: $confidence,
                        properties_json: $properties_json,
                        source: "parent_genome",
                        created_at: datetime($now)
                    })
                    """,
                    {
                        "id": new_id(),
                        "entity_id": str(belief.get("entity_id", "")),
                        "name": str(belief.get("name", "")),
                        "entity_type": str(belief.get("entity_type", "")),
                        "confidence": discounted,
                        "properties_json": properties_json,
                        "now": now_iso,
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug(
                    "nova_seed_belief_failed",
                    entity_id=belief.get("entity_id", ""),
                    error=str(exc),
                )

        return seeded

    async def _seed_goal_priors(self, goal_priors: list[dict[str, Any]]) -> int:
        """
        Pre-populate GoalPrior nodes so the child's goal manager can bootstrap
        priority estimates from inherited domain success rates.
        """
        from primitives.common import new_id, utc_now

        seeded = 0
        now_iso = utc_now().isoformat()

        for prior in goal_priors:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (:GoalPrior {
                        id: $id,
                        target_domain: $target_domain,
                        source: $source,
                        total: $total,
                        achieved: $achieved,
                        abandoned: $abandoned,
                        achievement_rate: $achievement_rate,
                        origin: "parent_genome",
                        created_at: datetime($now)
                    })
                    """,
                    {
                        "id": new_id(),
                        "target_domain": str(prior.get("target_domain", "")),
                        "source": str(prior.get("source", "")),
                        "total": int(prior.get("total", 0)),
                        "achieved": int(prior.get("achieved", 0)),
                        "abandoned": int(prior.get("abandoned", 0)),
                        "achievement_rate": float(prior.get("achievement_rate", 0.0)),
                        "now": now_iso,
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug(
                    "nova_seed_goal_prior_failed",
                    domain=prior.get("target_domain", ""),
                    error=str(exc),
                )

        return seeded

    async def _seed_efe_weights(self, efe_weights: dict[str, float]) -> bool:
        """
        Persist the parent's learned EFE weight vector so the child boots with
        inherited policy selection preferences rather than defaults.
        """
        if not efe_weights:
            return False

        from primitives.common import new_id, utc_now

        try:
            await self._neo4j.execute_write(
                """
                CREATE (:EFEWeights {
                    id: $id,
                    pragmatic: $pragmatic,
                    epistemic: $epistemic,
                    constitutional: $constitutional,
                    feasibility: $feasibility,
                    risk: $risk,
                    cognition_cost: $cognition_cost,
                    origin: "parent_genome",
                    created_at: datetime($now)
                })
                """,
                {
                    "id": new_id(),
                    "pragmatic": float(efe_weights.get("pragmatic", 0.35)),
                    "epistemic": float(efe_weights.get("epistemic", 0.20)),
                    "constitutional": float(efe_weights.get("constitutional", 0.20)),
                    "feasibility": float(efe_weights.get("feasibility", 0.15)),
                    "risk": float(efe_weights.get("risk", 0.10)),
                    "cognition_cost": float(efe_weights.get("cognition_cost", 0.10)),
                    "now": utc_now().isoformat(),
                },
            )
            return True
        except Exception as exc:
            self._log.debug("nova_seed_efe_weights_failed", error=str(exc))
            return False

    async def _seed_world_model_summary(self, summary: dict[str, Any]) -> bool:
        """
        Persist the parent's world model summary so the child has context about
        the parent's cognitive state at birth.
        """
        if not summary:
            return False

        import json

        from primitives.common import new_id, utc_now

        try:
            await self._neo4j.execute_write(
                """
                CREATE (:WorldModelSummary {
                    id: $id,
                    total_beliefs: $total_beliefs,
                    mean_confidence: $mean_confidence,
                    entity_types_json: $entity_types_json,
                    origin: "parent_genome",
                    created_at: datetime($now)
                })
                """,
                {
                    "id": new_id(),
                    "total_beliefs": int(summary.get("total_beliefs", 0)),
                    "mean_confidence": float(summary.get("mean_confidence", 0.0)),
                    "entity_types_json": json.dumps(
                        list(summary.get("entity_types", [])), default=str
                    ),
                    "now": utc_now().isoformat(),
                },
            )
            return True
        except Exception as exc:
            self._log.debug("nova_seed_world_model_failed", error=str(exc))
            return False
