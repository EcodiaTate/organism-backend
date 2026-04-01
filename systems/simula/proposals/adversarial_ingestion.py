"""
EcodiaOS - Adversarial Proposal Ingestion Adapter

Persists EvolutionProposals produced by the AdversarialSelfPlay engine
into the Neo4j knowledge graph, establishing the full graph schema:

  (:RedTeamInstance)-[:GENERATED]->(:EvolutionProposal)-[:TARGETS]->(:Constitution)

Design decisions:
  - All Cypher queries use parameterised inputs - no string interpolation.
  - Proposals enter Neo4j with status ``AWAITING_GOVERNANCE`` so human
    operators can query the graph for pending approvals.
  - Constraint creation is idempotent (``IF NOT EXISTS``).
  - Write failures are logged and surfaced to callers via the result type;
    they never crash the drain loop.
  - A singleton :Constitution node is MERGEd (not CREATEd) to avoid
    duplicates across restarts.

Graph schema:

  (:EvolutionProposal {
      id, source, category, description, status,
      attack_category, bypass_severity, risk_assessment,
      expected_benefit, cycle_id, evidence,
      created_at
  })

  (:RedTeamInstance { instance_id })
  (:Constitution { name: "EcodiaOS" })

  (:RedTeamInstance)-[:GENERATED { at }]->(:EvolutionProposal)
  (:EvolutionProposal)-[:TARGETS]->(:Constitution)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.simula.evolution_types import EvolutionProposal

logger = structlog.get_logger().bind(module="simula.proposals.adversarial_ingestion")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Outcome of ingesting one batch of adversarial proposals."""

    written: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cypher constants - all parameterised, no string interpolation
# ---------------------------------------------------------------------------

_ENSURE_CONSTRAINTS = """
CREATE CONSTRAINT evolution_proposal_id IF NOT EXISTS
FOR (p:EvolutionProposal) REQUIRE p.id IS UNIQUE
"""

_ENSURE_RED_TEAM_CONSTRAINT = """
CREATE CONSTRAINT red_team_instance_id IF NOT EXISTS
FOR (r:RedTeamInstance) REQUIRE r.instance_id IS UNIQUE
"""

_ENSURE_CONSTITUTION_CONSTRAINT = """
CREATE CONSTRAINT constitution_name IF NOT EXISTS
FOR (c:Constitution) REQUIRE c.name IS UNIQUE
"""

_MERGE_CONSTITUTION = """
MERGE (:Constitution {name: $name})
"""

_MERGE_RED_TEAM_INSTANCE = """
MERGE (:RedTeamInstance {instance_id: $instance_id})
"""

_CREATE_PROPOSAL_WITH_EDGES = """
MATCH (rt:RedTeamInstance {instance_id: $instance_id})
MATCH (c:Constitution {name: $constitution_name})
CREATE (p:EvolutionProposal {
    id:                $id,
    source:            $source,
    category:          $category,
    description:       $description,
    status:            $status,
    attack_category:   $attack_category,
    bypass_severity:   $bypass_severity,
    risk_assessment:   $risk_assessment,
    expected_benefit:  $expected_benefit,
    cycle_id:          $cycle_id,
    evidence:          $evidence,
    change_spec_json:  $change_spec_json,
    created_at:        $created_at
})
CREATE (rt)-[:GENERATED {at: $created_at}]->(p)
CREATE (p)-[:TARGETS]->(c)
RETURN p.id AS proposal_id
"""

_QUERY_PENDING_GOVERNANCE = """
MATCH (p:EvolutionProposal {status: $status})
OPTIONAL MATCH (rt:RedTeamInstance)-[:GENERATED]->(p)
OPTIONAL MATCH (p)-[:TARGETS]->(c:Constitution)
RETURN p.id              AS id,
       p.description      AS description,
       p.attack_category  AS attack_category,
       p.bypass_severity  AS bypass_severity,
       p.risk_assessment  AS risk_assessment,
       p.expected_benefit AS expected_benefit,
       p.cycle_id         AS cycle_id,
       p.created_at       AS created_at,
       rt.instance_id     AS red_team_instance,
       c.name             AS constitution
ORDER BY p.created_at ASC
"""

_UPDATE_PROPOSAL_STATUS = """
MATCH (p:EvolutionProposal {id: $id})
SET p.status = $new_status
RETURN p.id AS id
"""

# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

CONSTITUTION_NAME = "EcodiaOS"


class AdversarialProposalIngester:
    """
    Drains adversarial EvolutionProposals and writes them to Neo4j.

    Usage::

        ingester = AdversarialProposalIngester(neo4j, instance_id="asp-01")
        await ingester.ensure_schema()

        # In the drain loop:
        proposals = self_play.drain_proposals()
        result = await ingester.ingest(proposals)
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        instance_id: str = "adversarial-self-play-0",
    ) -> None:
        self._neo4j = neo4j
        self._instance_id = instance_id
        self._schema_ensured = False
        self._log = logger

    async def ensure_schema(self) -> None:
        """
        Idempotent schema setup: constraints + singleton nodes.

        Safe to call on every startup - ``IF NOT EXISTS`` guards everything.
        """
        if self._schema_ensured:
            return

        try:
            await self._neo4j.execute_write(_ENSURE_CONSTRAINTS)
            await self._neo4j.execute_write(_ENSURE_RED_TEAM_CONSTRAINT)
            await self._neo4j.execute_write(_ENSURE_CONSTITUTION_CONSTRAINT)
            await self._neo4j.execute_write(
                _MERGE_CONSTITUTION, {"name": CONSTITUTION_NAME}
            )
            await self._neo4j.execute_write(
                _MERGE_RED_TEAM_INSTANCE, {"instance_id": self._instance_id}
            )
            self._schema_ensured = True
            self._log.info(
                "adversarial_schema_ensured",
                instance_id=self._instance_id,
            )
        except Exception as exc:
            self._log.error(
                "adversarial_schema_setup_failed",
                error=str(exc),
            )
            raise

    async def ingest(
        self,
        proposals: list[EvolutionProposal],
    ) -> IngestionResult:
        """
        Write a batch of drained proposals to Neo4j.

        Applies two-tier in-process deduplication before any Neo4j write so
        that semantically identical bypass variants (same prefix or same
        category+affected_systems) don't flood the AWAITING_GOVERNANCE queue.

        Each surviving proposal is written individually so a single failure
        doesn't block the rest. All queries use parameterised inputs.
        """
        if not proposals:
            return IngestionResult()

        if not self._schema_ensured:
            await self.ensure_schema()

        proposals = self._deduplicate(proposals)
        result = IngestionResult()

        for proposal in proposals:
            try:
                await self._write_proposal(proposal)
                result.written += 1
                self._log.info(
                    "adversarial_proposal_ingested",
                    proposal_id=proposal.id,
                    status=proposal.status.value,
                    category=proposal.category.value,
                )
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"{proposal.id}: {exc}")
                self._log.warning(
                    "adversarial_proposal_ingest_failed",
                    proposal_id=proposal.id,
                    error=str(exc),
                )

        self._log.info(
            "adversarial_ingest_batch_complete",
            written=result.written,
            failed=result.failed,
        )
        return result

    async def query_pending_governance(self) -> list[dict[str, object]]:
        """
        Return all proposals in AWAITING_GOVERNANCE state.

        Convenience query for human operators to review pending approvals.
        """
        return await self._neo4j.execute_read(
            _QUERY_PENDING_GOVERNANCE,
            {"status": "awaiting_governance"},
        )

    async def update_status(self, proposal_id: str, new_status: str) -> bool:
        """
        Transition a proposal to a new status (e.g. approved / rejected).

        Returns True if the proposal was found and updated.
        """
        rows = await self._neo4j.execute_write(
            _UPDATE_PROPOSAL_STATUS,
            {"id": proposal_id, "new_status": new_status},
        )
        return len(rows) > 0

    # ── Private ─────────────────────────────────────────────────────────────

    async def _write_proposal(self, proposal: EvolutionProposal) -> None:
        """Write a single EvolutionProposal node with GENERATED and TARGETS edges."""
        # Extract adversarial-specific metadata from the description/change_spec.
        attack_category = self._extract_attack_category(proposal)
        bypass_severity = self._extract_bypass_severity(proposal)
        cycle_id = self._extract_cycle_id(proposal)

        params = {
            "instance_id": self._instance_id,
            "constitution_name": CONSTITUTION_NAME,
            "id": proposal.id,
            "source": proposal.source,
            "category": proposal.category.value,
            "description": proposal.description,
            "status": proposal.status.value,
            "attack_category": attack_category,
            "bypass_severity": bypass_severity,
            "risk_assessment": proposal.risk_assessment,
            "expected_benefit": proposal.expected_benefit,
            "cycle_id": cycle_id,
            "evidence": json.dumps(proposal.evidence),
            "change_spec_json": json.dumps(
                proposal.change_spec.model_dump(mode="json")
            ),
            "created_at": proposal.created_at.isoformat(),
        }

        rows = await self._neo4j.execute_write(
            _CREATE_PROPOSAL_WITH_EDGES, params
        )

        if not rows:
            raise RuntimeError(
                f"CREATE returned no rows for proposal {proposal.id} - "
                f"RedTeamInstance or Constitution node may be missing"
            )

    @staticmethod
    def _deduplicate(
        proposals: list[EvolutionProposal],
    ) -> list[EvolutionProposal]:
        """
        Two-tier in-process dedup: keeps the first representative per group.

        Tier 1 - exact description prefix (first 50 chars, lowercased).
        Tier 2 - category + sorted affected_systems string key.

        Both tiers are zero-cost (no LLM, no embeddings). Tier 1 runs first;
        proposals deduplicated by Tier 1 are excluded from Tier 2 checks.
        """
        _PREFIX_LEN = 50
        seen_prefixes: set[str] = set()
        seen_cat_systems: set[str] = set()
        unique: list[EvolutionProposal] = []

        for p in proposals:
            # Tier 1: description prefix
            prefix = p.description[:_PREFIX_LEN].lower().strip()
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)

            # Tier 2: category + affected systems
            cat_key = (
                f"{p.category.value}::"
                f"{','.join(sorted(p.change_spec.affected_systems))}"
            )
            if cat_key in seen_cat_systems:
                continue
            seen_cat_systems.add(cat_key)

            unique.append(p)

        return unique

    @staticmethod
    def _extract_attack_category(proposal: EvolutionProposal) -> str:
        """Pull the attack category from the change_spec additional_context."""
        ctx = proposal.change_spec.additional_context
        for segment in ctx.split("."):
            segment = segment.strip()
            if segment.startswith("Attack category:"):
                return segment.removeprefix("Attack category:").strip()
        return "unknown"

    @staticmethod
    def _extract_bypass_severity(proposal: EvolutionProposal) -> str:
        """Pull bypass severity from the change_spec additional_context."""
        ctx = proposal.change_spec.additional_context
        for segment in ctx.split("."):
            segment = segment.strip()
            if segment.startswith("Bypass severity:"):
                return segment.removeprefix("Bypass severity:").strip()
        return "unknown"

    @staticmethod
    def _extract_cycle_id(proposal: EvolutionProposal) -> str:
        """The first evidence entry is the cycle ID by convention."""
        return proposal.evidence[0] if proposal.evidence else ""
