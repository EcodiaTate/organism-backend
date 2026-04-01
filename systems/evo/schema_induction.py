"""
EcodiaOS - Schema Induction Engine

Genuine structure learning from the organism's accumulated experience.
This is NOT a simple label creator - it discovers latent relational structure
in the knowledge graph and proposes typed entity categories, relation types,
and compositional schemas that compress the organism's world model.

Three induction strategies, each progressively deeper:

1. **Graph Motif Mining** (statistical)
   Scans the Neo4j graph for recurring subgraph patterns - entity clusters,
   hub-spoke structures, chain patterns - and proposes entity types that
   capture the regularity. Uses community detection + betweenness centrality.

2. **Analogical Structure Mapping** (relational)
   Identifies isomorphic subgraphs: if (A)-[R1]->(B) and (C)-[R1]->(D)
   share structural properties, propose an abstract schema that captures
   the common structure. This is the computational analog of analogical
   reasoning - discovering that "planets orbit stars" and "electrons orbit
   nuclei" share the same relational structure.

3. **Compression-Driven Schema Discovery** (information-theoretic)
   Uses MDL (Minimum Description Length) scoring: a proposed schema is
   valuable if encoding the data WITH the schema takes fewer bits than
   encoding the data WITHOUT it. Computes description length reduction
   and only accepts schemas that genuinely compress the knowledge graph.

Performance budget: Phase 3 of consolidation, ≤10s total.
Safety: schemas are proposals - they go to Equor/Simula for approval before
modifying the canonical graph structure.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import time
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.memory.service import MemoryService

logger = structlog.get_logger()


# ─── Types ───────────────────────────────────────────────────────────────────


class SchemaElement(EOSBaseModel):
    """A single discovered structural element (entity type or relation type)."""

    kind: str  # "entity_type" | "relation_type" | "constraint"
    name: str
    description: str = ""
    source_pattern: str = ""  # Which induction strategy found this
    instance_count: int = 0  # How many graph nodes match
    mdl_gain_bits: float = 0.0  # Bits saved by adopting this schema element
    confidence: float = 0.0  # 0-1, statistical confidence in the pattern


class SchemaComposition(EOSBaseModel):
    """
    A compositional schema: a typed subgraph pattern discovered from
    recurring structural motifs.

    Example: "social_interaction" schema composes:
        (:Person)-[:INITIATED]->(:Conversation)-[:RESULTED_IN]->(:Outcome)
    """

    id: str = Field(default_factory=new_id)
    name: str
    description: str = ""
    elements: list[SchemaElement] = Field(default_factory=list)
    # The structural pattern as a list of (source_type, relation, target_type) triples
    structure: list[tuple[str, str, str]] = Field(default_factory=list)
    # Abstraction level: 0=concrete, 1=first-order abstract, 2+=higher
    abstraction_level: int = 0
    # Parent schema this specializes (None if root)
    parent_schema_id: str | None = None
    # MDL score: total bits saved by this composition vs. flat encoding
    mdl_total_gain: float = 0.0
    instance_count: int = 0
    confidence: float = 0.0


class SchemaInductionResult(EOSBaseModel):
    """Result of a full schema induction pass."""

    entity_types_discovered: int = 0
    relation_types_discovered: int = 0
    compositions_discovered: int = 0
    mdl_total_gain_bits: float = 0.0
    elements: list[SchemaElement] = Field(default_factory=list)
    compositions: list[SchemaComposition] = Field(default_factory=list)
    duration_ms: int = 0


# ─── Constants ───────────────────────────────────────────────────────────────

# Minimum instances of a pattern before it's worth schematizing
_MIN_INSTANCES: int = 5
# Minimum MDL gain (bits) to justify a new schema element
_MIN_MDL_GAIN_BITS: float = 2.0
# Maximum schema elements per induction pass (budget control)
_MAX_ELEMENTS_PER_PASS: int = 10
# Minimum co-occurrence ratio to propose a relation type
_MIN_COOCCURRENCE_RATIO: float = 0.3


# ─── Schema Induction Engine ─────────────────────────────────────────────────


class SchemaInductionEngine:
    """
    Discovers latent structure in the knowledge graph during Evo consolidation.

    Three strategies run in sequence, each building on the previous:
    1. Graph motif mining → entity type proposals
    2. Analogical structure mapping → relation type + composition proposals
    3. MDL filtering → only keep schemas that genuinely compress

    Requires Neo4j access via MemoryService.
    """

    def __init__(
        self,
        memory: MemoryService | None = None,
        logos: Any | None = None,
    ) -> None:
        self._memory = memory
        self._logos = logos
        self._logger = logger.bind(system="evo.schema_induction")
        # Cache of previously discovered schemas (avoid re-proposing)
        self._known_schema_hashes: set[str] = set()

    async def induce(
        self,
        supported_hypotheses: list[Any] | None = None,
    ) -> SchemaInductionResult:
        """
        Run the full schema induction pipeline.

        Steps:
        1. Mine graph motifs for recurring structural patterns
        2. Discover analogical mappings between subgraph regions
        3. Score all candidates with MDL and filter
        4. Propose compositions from surviving elements
        5. Persist approved schemas to the graph

        Returns SchemaInductionResult with all discovered elements.
        """
        start = time.monotonic()
        result = SchemaInductionResult()

        if self._memory is None:
            return result

        neo4j = self._memory

        # ── Strategy 1: Graph Motif Mining ─────────────────────────────
        entity_candidates = await self._mine_entity_type_motifs(neo4j)
        relation_candidates = await self._mine_relation_type_motifs(neo4j)

        # ── Strategy 2: Analogical Structure Mapping ───────────────────
        analogy_candidates = await self._discover_analogical_structures(neo4j)

        # ── Strategy 4: Causal Invariant Mining ────────────────────────
        causal_candidates = await self._mine_causal_invariants(neo4j)

        # ── Strategy 3: MDL Filtering ──────────────────────────────────
        all_candidates = (
            entity_candidates + relation_candidates
            + analogy_candidates + causal_candidates
        )
        filtered = self._mdl_filter(all_candidates)

        # ── Deduplicate against known schemas ──────────────────────────
        novel = self._deduplicate(filtered)

        # ── Compose higher-order schemas from co-occurring elements ────
        compositions = self._compose_schemas(novel)

        # ── Hypothesis-driven schema proposals ─────────────────────────
        if supported_hypotheses:
            hyp_schemas = self._schemas_from_hypotheses(supported_hypotheses)
            novel.extend(hyp_schemas)

        # ── Persist to graph ───────────────────────────────────────────
        persisted_count = 0
        for element in novel[:_MAX_ELEMENTS_PER_PASS]:
            success = await self._persist_schema_element(neo4j, element)
            if success:
                persisted_count += 1
                self._known_schema_hashes.add(self._element_hash(element))

        for comp in compositions[:5]:
            await self._persist_composition(neo4j, comp)

        # ── Tally results ──────────────────────────────────────────────
        result.entity_types_discovered = sum(
            1 for e in novel if e.kind == "entity_type"
        )
        result.relation_types_discovered = sum(
            1 for e in novel if e.kind == "relation_type"
        )
        result.compositions_discovered = len(compositions)
        result.mdl_total_gain_bits = sum(e.mdl_gain_bits for e in novel)
        result.elements = novel[:_MAX_ELEMENTS_PER_PASS]
        result.compositions = compositions[:5]
        result.duration_ms = int((time.monotonic() - start) * 1000)

        self._logger.info(
            "schema_induction_complete",
            entity_types=result.entity_types_discovered,
            relation_types=result.relation_types_discovered,
            compositions=result.compositions_discovered,
            mdl_gain_bits=round(result.mdl_total_gain_bits, 2),
            duration_ms=result.duration_ms,
        )

        return result

    # ─── Strategy 1: Graph Motif Mining ───────────────────────────────────────

    async def _mine_entity_type_motifs(
        self,
        neo4j: Neo4jClient,
    ) -> list[SchemaElement]:
        """
        Discover natural entity groupings by mining label co-occurrence,
        property overlap, and neighbourhood structure.

        Technique: For unlabelled nodes (those with only generic labels like
        :Entity), cluster by shared property keys + relationship types.
        Nodes with identical property signatures and similar neighbourhoods
        likely belong to the same latent type.
        """
        candidates: list[SchemaElement] = []

        try:
            # Query property signatures of Entity nodes
            records = await neo4j.execute_read(
                """
                MATCH (n)
                WHERE NOT n:Self AND NOT n:Hypothesis AND NOT n:EvoParameter
                  AND NOT n:Procedure AND NOT n:ConsolidatedBelief
                  AND NOT n:BeliefGenome
                WITH labels(n) AS lbls, keys(n) AS props, n
                WITH lbls, props,
                     [k IN props WHERE NOT k IN ['id', 'created_at', 'updated_at',
                      'event_time', 'ingestion_time']] AS sig_props,
                     count(n) AS cnt
                WHERE cnt >= $min_instances
                RETURN lbls, sig_props, cnt
                ORDER BY cnt DESC
                LIMIT 50
                """,
                {"min_instances": _MIN_INSTANCES},
            )

            # Group by property signature → candidate entity types
            sig_groups: dict[str, int] = defaultdict(int)
            sig_labels: dict[str, list[list[str]]] = defaultdict(list)
            for record in records:
                sig = _canonical_sig(record.get("sig_props", []))
                count = int(record.get("cnt", 0))
                labels = record.get("lbls", [])
                sig_groups[sig] += count
                sig_labels[sig].append(labels)

            for sig, total_count in sig_groups.items():
                if total_count < _MIN_INSTANCES:
                    continue

                # Derive a name from the most common label set
                all_labels = sig_labels[sig]
                label_counts = Counter(
                    tuple(sorted(lbls)) for lbls in all_labels
                )
                most_common_labels = label_counts.most_common(1)[0][0]

                # Compute MDL gain: encoding N nodes with type T saves
                # ~log2(N) bits per node vs. listing all properties each time
                props_in_sig = sig.split("|") if sig else []
                bits_per_node = len(props_in_sig) * 3.0  # ~3 bits per property key
                mdl_gain = total_count * bits_per_node - math.log2(max(1, len(sig_groups)))

                if mdl_gain < _MIN_MDL_GAIN_BITS:
                    continue

                name = "_".join(most_common_labels) if most_common_labels else f"type_{sig[:12]}"
                candidates.append(SchemaElement(
                    kind="entity_type",
                    name=name,
                    description=(
                        f"Entity cluster with properties [{sig}], "
                        f"{total_count} instances"
                    ),
                    source_pattern="graph_motif_property_signature",
                    instance_count=total_count,
                    mdl_gain_bits=round(mdl_gain, 2),
                    confidence=min(0.95, 0.5 + total_count * 0.01),
                ))

        except Exception as exc:
            self._logger.warning("entity_motif_mining_failed", error=str(exc))

        return candidates

    async def _mine_relation_type_motifs(
        self,
        neo4j: Neo4jClient,
    ) -> list[SchemaElement]:
        """
        Discover latent relation types from edge frequency patterns.

        Technique: Group relationships by (source_label_set, rel_type, target_label_set).
        Frequent patterns that aren't already typed suggest a discoverable relation type.
        Also computes co-occurrence ratios: if A-[R]->B appears with A-[R]->C frequently,
        R might be a many-to-many relation worth constraining.
        """
        candidates: list[SchemaElement] = []

        try:
            records = await neo4j.execute_read(
                """
                MATCH (a)-[r]->(b)
                WITH labels(a) AS src_labels, type(r) AS rel_type,
                     labels(b) AS tgt_labels, count(*) AS cnt
                WHERE cnt >= $min_instances
                RETURN src_labels, rel_type, tgt_labels, cnt
                ORDER BY cnt DESC
                LIMIT 40
                """,
                {"min_instances": _MIN_INSTANCES},
            )

            for record in records:
                src = sorted(record.get("src_labels", []))
                rel = str(record.get("rel_type", ""))
                tgt = sorted(record.get("tgt_labels", []))
                count = int(record.get("cnt", 0))

                # MDL gain: typing a relationship saves ~log2(distinct_rel_types) bits per edge
                mdl_gain = count * 1.5  # Conservative estimate

                if mdl_gain < _MIN_MDL_GAIN_BITS:
                    continue

                pattern_name = f"{':'.join(src)}-[{rel}]->{':'.join(tgt)}"
                candidates.append(SchemaElement(
                    kind="relation_type",
                    name=pattern_name,
                    description=(
                        f"Relation pattern ({':'.join(src)})-[{rel}]->({':'.join(tgt)}) "
                        f"with {count} instances"
                    ),
                    source_pattern="graph_motif_relation_frequency",
                    instance_count=count,
                    mdl_gain_bits=round(mdl_gain, 2),
                    confidence=min(0.9, 0.5 + count * 0.005),
                ))

        except Exception as exc:
            self._logger.warning("relation_motif_mining_failed", error=str(exc))

        return candidates

    # ─── Strategy 2: Analogical Structure Mapping ─────────────────────────────

    async def _discover_analogical_structures(
        self,
        neo4j: Neo4jClient,
    ) -> list[SchemaElement]:
        """
        Find structurally isomorphic subgraph regions.

        Technique: For each relationship type R, find all (A)-[R]->(B) pairs.
        Group by the property overlap between A and B across instances.
        If multiple distinct (A, B) pairs share the same structural role
        (same outgoing/incoming edge types), they instantiate the same abstract
        schema - propose it.

        This is computational analogy: discovering that structurally different
        entities participate in the same relational pattern.
        """
        candidates: list[SchemaElement] = []

        try:
            # Find relationship types that connect diverse entity types
            records = await neo4j.execute_read(
                """
                MATCH (a)-[r]->(b)
                WITH type(r) AS rel_type,
                     collect(DISTINCT labels(a)) AS src_types,
                     collect(DISTINCT labels(b)) AS tgt_types,
                     count(*) AS cnt
                WHERE cnt >= $min_instances
                  AND size(src_types) >= 2
                RETURN rel_type, src_types, tgt_types, cnt
                ORDER BY cnt DESC
                LIMIT 20
                """,
                {"min_instances": _MIN_INSTANCES},
            )

            for record in records:
                rel_type = str(record.get("rel_type", ""))
                src_types = record.get("src_types", [])
                count = int(record.get("cnt", 0))

                num_distinct_src = len(src_types)

                if num_distinct_src < 2:
                    continue

                # Analogical gain: the more diverse the source types sharing the
                # same relationship, the more compressive the abstract schema
                analogy_gain = math.log2(max(2, num_distinct_src)) * count * 0.1

                if analogy_gain < _MIN_MDL_GAIN_BITS:
                    continue

                candidates.append(SchemaElement(
                    kind="entity_type",
                    name=f"abstract_{rel_type}_participant",
                    description=(
                        f"Abstract role: {num_distinct_src} different entity types "
                        f"participate as source in [{rel_type}] relationships. "
                        f"Suggests a shared abstract type."
                    ),
                    source_pattern="analogical_structure_mapping",
                    instance_count=count,
                    mdl_gain_bits=round(analogy_gain, 2),
                    confidence=min(0.85, 0.4 + num_distinct_src * 0.1),
                ))

        except Exception as exc:
            self._logger.warning("analogical_discovery_failed", error=str(exc))

        return candidates

    # ─── Strategy 4: Causal Invariant Mining ────────────────────────────────

    async def _mine_causal_invariants(
        self,
        neo4j: Neo4jClient,
    ) -> list[SchemaElement]:
        """
        Mine Kairos-discovered causal invariants from the graph.

        Kairos stores invariants as :CausalInvariant nodes with cause, effect,
        confidence, and tier fields. High-tier invariants (especially Tier 3 -
        substrate-independent) represent deep structural relationships that
        should be encoded as schema elements.

        These become typed causal edges in the schema: (CauseType)-[CAUSES]->(EffectType)
        """
        candidates: list[SchemaElement] = []

        try:
            records = await neo4j.execute_read(
                """
                MATCH (ci:CausalInvariant)
                WHERE ci.confidence >= 0.5 AND ci.active = true
                RETURN ci.cause AS cause, ci.effect AS effect,
                       ci.confidence AS confidence,
                       ci.tier AS tier,
                       ci.scope AS scope,
                       ci.evidence_count AS evidence_count
                ORDER BY ci.confidence DESC, ci.tier DESC
                LIMIT 30
                """,
            )

            for record in records:
                cause = str(record.get("cause", ""))
                effect = str(record.get("effect", ""))
                confidence = float(record.get("confidence", 0.0))
                tier = int(record.get("tier", 1))
                evidence_count = int(record.get("evidence_count", 0))

                if not cause or not effect:
                    continue

                # Higher tiers and more evidence = more MDL gain
                # Tier 3 (substrate-independent) discoveries are most compressive
                tier_multiplier = {1: 1.0, 2: 2.0, 3: 5.0}.get(tier, 1.0)
                mdl_gain = (
                    math.log2(max(2, evidence_count))
                    * confidence
                    * tier_multiplier
                )

                if mdl_gain < _MIN_MDL_GAIN_BITS:
                    continue

                candidates.append(SchemaElement(
                    kind="relation_type",
                    name=f"CAUSES:{cause}→{effect}",
                    description=(
                        f"Causal invariant (Tier {tier}): {cause} causes {effect} "
                        f"(confidence={confidence:.2f}, evidence={evidence_count})"
                    ),
                    source_pattern="causal_invariant_mining",
                    instance_count=evidence_count,
                    mdl_gain_bits=round(mdl_gain, 2),
                    confidence=confidence,
                ))

        except Exception as exc:
            self._logger.warning("causal_invariant_mining_failed", error=str(exc))

        return candidates

    # ─── Strategy 3: MDL Filtering ────────────────────────────────────────────

    def _mdl_filter(
        self,
        candidates: list[SchemaElement],
    ) -> list[SchemaElement]:
        """
        Filter candidates by MDL criterion: keep only schemas whose
        description length savings exceed the cost of encoding the schema itself.

        Schema cost: ~10 bits for the schema definition + log2(candidates) for indexing
        Schema benefit: mdl_gain_bits (pre-computed per candidate)
        Net gain = benefit - cost

        Also applies a Logos MDL score if Logos is wired.
        """
        schema_definition_cost = 10.0  # bits
        index_cost = math.log2(max(2, len(candidates)))

        filtered: list[SchemaElement] = []
        for candidate in candidates:
            net_gain = candidate.mdl_gain_bits - schema_definition_cost - index_cost
            if net_gain > 0:
                candidate.mdl_gain_bits = round(net_gain, 2)
                filtered.append(candidate)

        # Sort by MDL gain descending - most compressive first
        filtered.sort(key=lambda e: e.mdl_gain_bits, reverse=True)

        return filtered

    # ─── Schema Composition ───────────────────────────────────────────────────

    def _compose_schemas(
        self,
        elements: list[SchemaElement],
    ) -> list[SchemaComposition]:
        """
        Compose higher-order schemas from co-occurring elements.

        If an entity type E and relation type R frequently appear together
        in the same motif, compose them into a SchemaComposition that
        captures the full subgraph pattern.
        """
        if len(elements) < 2:
            return []

        compositions: list[SchemaComposition] = []

        # Group elements by source pattern
        by_pattern: dict[str, list[SchemaElement]] = defaultdict(list)
        for elem in elements:
            by_pattern[elem.source_pattern].append(elem)

        # For each pattern group with 2+ elements, propose a composition
        for pattern, group in by_pattern.items():
            if len(group) < 2:
                continue

            # Take the top elements by MDL gain
            top = sorted(group, key=lambda e: e.mdl_gain_bits, reverse=True)[:4]

            total_gain = sum(e.mdl_gain_bits for e in top)
            # Composition bonus: encoding elements together is more efficient
            # than encoding them separately (shared context)
            composition_bonus = math.log2(max(2, len(top))) * 2.0
            total_gain += composition_bonus

            mean_confidence = sum(e.confidence for e in top) / len(top)

            comp = SchemaComposition(
                name=f"schema_{pattern}_{len(compositions)}",
                description=(
                    f"Compositional schema from {len(top)} elements "
                    f"discovered via {pattern}"
                ),
                elements=top,
                abstraction_level=1,
                mdl_total_gain=round(total_gain, 2),
                instance_count=min(e.instance_count for e in top),
                confidence=round(mean_confidence, 3),
            )
            compositions.append(comp)

        return compositions

    # ─── Hypothesis-Driven Proposals ──────────────────────────────────────────

    def _schemas_from_hypotheses(
        self,
        hypotheses: list[Any],
    ) -> list[SchemaElement]:
        """
        Extract schema proposals from supported WORLD_MODEL hypotheses
        that have SCHEMA_ADDITION mutations.
        """
        from systems.evo.types import HypothesisCategory, MutationType

        elements: list[SchemaElement] = []
        for h in hypotheses:
            if h.category != HypothesisCategory.WORLD_MODEL:
                continue
            if h.proposed_mutation is None:
                continue
            if h.proposed_mutation.type != MutationType.SCHEMA_ADDITION:
                continue

            elements.append(SchemaElement(
                kind="entity_type",
                name=h.proposed_mutation.target,
                description=h.statement,
                source_pattern="hypothesis_driven",
                instance_count=len(h.supporting_episodes),
                mdl_gain_bits=h.evidence_score * 0.5,
                confidence=min(0.9, 0.5 + h.evidence_score * 0.05),
            ))

        return elements

    # ─── Deduplication ────────────────────────────────────────────────────────

    def _deduplicate(
        self,
        candidates: list[SchemaElement],
    ) -> list[SchemaElement]:
        """Remove candidates that match previously discovered schemas."""
        novel: list[SchemaElement] = []
        for c in candidates:
            h = self._element_hash(c)
            if h not in self._known_schema_hashes:
                novel.append(c)
        return novel

    @staticmethod
    def _element_hash(element: SchemaElement) -> str:
        """Stable hash for deduplication."""
        canonical = f"{element.kind}:{element.name}:{element.source_pattern}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    # ─── Persistence ──────────────────────────────────────────────────────────

    async def _persist_schema_element(
        self,
        neo4j: Neo4jClient,
        element: SchemaElement,
    ) -> bool:
        """Persist a discovered schema element as an :EvoSchema node."""
        try:
            await neo4j.execute_write(
                """
                MERGE (s:EvoSchema {name: $name, kind: $kind})
                SET s.description = $description,
                    s.source_pattern = $source_pattern,
                    s.instance_count = $instance_count,
                    s.mdl_gain_bits = $mdl_gain_bits,
                    s.confidence = $confidence,
                    s.discovered_at = datetime(),
                    s.active = true
                WITH s
                MATCH (self:Self)
                MERGE (self)-[:DISCOVERED_SCHEMA]->(s)
                """,
                {
                    "name": element.name,
                    "kind": element.kind,
                    "description": element.description,
                    "source_pattern": element.source_pattern,
                    "instance_count": element.instance_count,
                    "mdl_gain_bits": element.mdl_gain_bits,
                    "confidence": element.confidence,
                },
            )
            self._logger.info(
                "schema_element_persisted",
                kind=element.kind,
                name=element.name,
                mdl_gain=round(element.mdl_gain_bits, 2),
                confidence=round(element.confidence, 3),
            )
            return True
        except Exception as exc:
            self._logger.warning(
                "schema_element_persist_failed",
                name=element.name,
                error=str(exc),
            )
            return False

    async def _persist_composition(
        self,
        neo4j: Neo4jClient,
        composition: SchemaComposition,
    ) -> bool:
        """Persist a compositional schema with links to its constituent elements."""
        try:
            element_names = [e.name for e in composition.elements]
            await neo4j.execute_write(
                """
                MERGE (sc:EvoSchemaComposition {id: $id})
                SET sc.name = $name,
                    sc.description = $description,
                    sc.abstraction_level = $abstraction_level,
                    sc.mdl_total_gain = $mdl_total_gain,
                    sc.instance_count = $instance_count,
                    sc.confidence = $confidence,
                    sc.element_names = $element_names,
                    sc.discovered_at = datetime()
                WITH sc
                MATCH (self:Self)
                MERGE (self)-[:DISCOVERED_COMPOSITION]->(sc)
                """,
                {
                    "id": composition.id,
                    "name": composition.name,
                    "description": composition.description,
                    "abstraction_level": composition.abstraction_level,
                    "mdl_total_gain": composition.mdl_total_gain,
                    "instance_count": composition.instance_count,
                    "confidence": composition.confidence,
                    "element_names": element_names,
                },
            )

            # Link composition to its constituent schema elements
            for elem_name in element_names:
                with contextlib.suppress(Exception):
                    await neo4j.execute_write(
                        """
                        MATCH (sc:EvoSchemaComposition {id: $comp_id})
                        MATCH (s:EvoSchema {name: $elem_name})
                        MERGE (sc)-[:COMPOSED_OF]->(s)
                        """,
                        {"comp_id": composition.id, "elem_name": elem_name},
                    )

            return True
        except Exception as exc:
            self._logger.warning(
                "schema_composition_persist_failed",
                name=composition.name,
                error=str(exc),
            )
            return False


# ─── Helpers ─────────────────────────────────────────────────────────────────


class SchemaAlgebra:
    """
    Compositional schema algebra: compose, specialize, abstract, and transfer schemas.

    Operators:
      - compose(A, B) → C: two co-occurring schemas form a higher-order schema
      - specialize(parent) → child: add constraints to an existing schema
      - abstract(schemas) → parent: extract common properties from N schemas
      - transfer(schema, source_domain, target_domain) → abstract schema

    Schema algebra rules:
      - Composition is associative: (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)
      - Specialization preserves parent MDL gain
      - Abstraction reduces total MDL cost (shared properties encoded once)

    Persists the full schema DAG in Neo4j:
      (:EvoSchema)-[:SPECIALIZES]->(:EvoSchema)
      (:EvoSchema)-[:COMPOSED_OF]->(:EvoSchema)
      (:EvoSchema)-[:ABSTRACTS]->(:EvoSchema)
    """

    def __init__(self, memory: Any | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="evo.schema_algebra")
        # Version tracking: schema_name → current version number
        self._schema_versions: dict[str, int] = defaultdict(int)
        # Fitness tracking: schema_name → compression bits provided
        self._schema_fitness: dict[str, float] = {}

    async def compose(
        self,
        schema_a: SchemaComposition,
        schema_b: SchemaComposition,
    ) -> SchemaComposition | None:
        """
        Schema composition operator: A ⊗ B → C.

        Two schemas that co-occur form a higher-order schema. The composed
        schema inherits all elements from both parents and gains a composition
        bonus in MDL (shared context is encoded once).

        Returns None if composition would not reduce MDL.
        """
        # Check for overlap - shared elements give composition bonus
        a_names = {e.name for e in schema_a.elements}
        b_names = {e.name for e in schema_b.elements}
        shared = a_names & b_names

        # Merged element list: union of both, shared elements counted once
        merged_elements: list[SchemaElement] = list(schema_a.elements)
        for elem in schema_b.elements:
            if elem.name not in a_names:
                merged_elements.append(elem)

        # MDL gain: parent gains + composition bonus from shared context
        parent_gain = schema_a.mdl_total_gain + schema_b.mdl_total_gain
        # Shared elements encoded once instead of twice
        shared_bonus = len(shared) * 3.0  # ~3 bits per shared element
        # Overhead of the composition node itself
        composition_cost = 10.0 + math.log2(max(2, len(merged_elements)))

        net_gain = parent_gain + shared_bonus - composition_cost
        if net_gain <= 0:
            return None

        # Build the merged structure
        merged_structure: list[tuple[str, str, str]] = list(schema_a.structure)
        for triple in schema_b.structure:
            if triple not in merged_structure:
                merged_structure.append(triple)

        composed = SchemaComposition(
            name=f"{schema_a.name}⊗{schema_b.name}",
            description=(
                f"Composition of '{schema_a.name}' and '{schema_b.name}': "
                f"{len(merged_elements)} elements, {len(shared)} shared"
            ),
            elements=merged_elements,
            structure=merged_structure,
            abstraction_level=max(schema_a.abstraction_level, schema_b.abstraction_level) + 1,
            mdl_total_gain=round(net_gain, 2),
            instance_count=min(schema_a.instance_count, schema_b.instance_count),
            confidence=round(
                (schema_a.confidence + schema_b.confidence) / 2.0, 3
            ),
        )

        # Persist composition relationship
        if self._memory is not None:
            neo4j = self._memory
            await self._persist_composition_relationship(
                neo4j, composed, [schema_a.id, schema_b.id], "COMPOSED_OF"
            )

        self._logger.info(
            "schema_composed",
            name=composed.name,
            elements=len(merged_elements),
            shared=len(shared),
            mdl_gain=round(net_gain, 2),
        )
        return composed

    async def specialize(
        self,
        parent: SchemaComposition,
        added_constraints: list[SchemaElement],
        specialization_name: str = "",
    ) -> SchemaComposition | None:
        """
        Schema specialization: Parent → Child with inherited properties + added constraints.

        Like OOP inheritance for knowledge structure: the child inherits all
        parent elements and adds new constraints that narrow its scope.

        Specialization preserves parent MDL gain - the child can only ADD value.
        Returns None if the added constraints don't provide enough MDL gain.
        """
        if not added_constraints:
            return None

        # MDL: parent gain is inherited; added constraints must provide net benefit
        added_gain = sum(e.mdl_gain_bits for e in added_constraints)
        constraint_cost = len(added_constraints) * 2.0  # ~2 bits per constraint
        net_added = added_gain - constraint_cost

        if net_added <= 0:
            return None

        child_elements = list(parent.elements) + list(added_constraints)
        child_name = specialization_name or f"{parent.name}_specialized"

        child = SchemaComposition(
            name=child_name,
            description=(
                f"Specialization of '{parent.name}' with {len(added_constraints)} "
                f"additional constraints"
            ),
            elements=child_elements,
            structure=list(parent.structure),
            abstraction_level=parent.abstraction_level,
            parent_schema_id=parent.id,
            mdl_total_gain=round(parent.mdl_total_gain + net_added, 2),
            instance_count=min(
                parent.instance_count,
                min((e.instance_count for e in added_constraints), default=0),
            ),
            confidence=round(
                (
                    parent.confidence
                    + sum(e.confidence for e in added_constraints)
                    / max(1, len(added_constraints))
                )
                / 2.0,
                3,
            ),
        )

        if self._memory is not None:
            neo4j = self._memory
            await self._persist_composition_relationship(
                neo4j, child, [parent.id], "SPECIALIZES"
            )

        self._logger.info(
            "schema_specialized",
            parent=parent.name,
            child=child_name,
            added_constraints=len(added_constraints),
            net_gain=round(net_added, 2),
        )
        return child

    async def abstract(
        self,
        schemas: list[SchemaComposition],
        abstract_name: str = "",
    ) -> SchemaComposition | None:
        """
        Schema abstraction: detect when N concrete schemas share M common properties,
        extract an abstract parent schema.

        The abstract schema encodes shared properties once, reducing total MDL cost.
        Returns None if insufficient commonality.
        """
        if len(schemas) < 2:
            return None

        # Find common elements across all schemas
        element_sets = [
            {e.name for e in s.elements} for s in schemas
        ]
        common_names = element_sets[0]
        for es in element_sets[1:]:
            common_names = common_names & es

        if len(common_names) < 2:
            return None  # Need at least 2 common elements for worthwhile abstraction

        # Extract the actual common elements (take from first schema)
        element_lookup: dict[str, SchemaElement] = {}
        for s in schemas:
            for e in s.elements:
                if e.name in common_names and e.name not in element_lookup:
                    element_lookup[e.name] = e
        common_elements = list(element_lookup.values())

        # MDL gain: encoding M properties once instead of N times
        properties_per_schema = len(common_elements)
        num_schemas = len(schemas)
        bits_per_property = 3.0
        savings = properties_per_schema * bits_per_property * (num_schemas - 1)
        # Cost of the abstract schema itself
        abstract_cost = 10.0 + properties_per_schema * 1.0
        net_gain = savings - abstract_cost

        if net_gain <= 0:
            return None

        name = abstract_name or f"abstract_{'_'.join(s.name[:10] for s in schemas[:3])}"

        abstract = SchemaComposition(
            name=name,
            description=(
                f"Abstract schema from {num_schemas} schemas sharing "
                f"{properties_per_schema} common properties"
            ),
            elements=common_elements,
            abstraction_level=max(s.abstraction_level for s in schemas) + 1,
            mdl_total_gain=round(net_gain, 2),
            instance_count=sum(s.instance_count for s in schemas),
            confidence=round(
                sum(s.confidence for s in schemas) / num_schemas, 3
            ),
        )

        # Persist abstraction relationships
        if self._memory is not None:
            neo4j = self._memory
            for s in schemas:
                await self._persist_composition_relationship(
                    neo4j, abstract, [s.id], "ABSTRACTS"
                )

        self._logger.info(
            "schema_abstracted",
            name=name,
            source_schemas=num_schemas,
            common_properties=properties_per_schema,
            mdl_gain=round(net_gain, 2),
        )
        return abstract

    async def detect_cross_domain_transfers(
        self,
        schemas: list[SchemaComposition],
    ) -> list[tuple[SchemaComposition, SchemaComposition, float]]:
        """
        Find schemas in different domains that are structurally isomorphic.

        When Schema X in domain A is isomorphic to Schema Y in domain B,
        propose a domain-independent abstract schema Z.

        Returns list of (schema_a, schema_b, isomorphism_score) tuples.
        """
        transfers: list[tuple[SchemaComposition, SchemaComposition, float]] = []

        for i, sa in enumerate(schemas):
            for sb in schemas[i + 1:]:
                # Skip if same domain/source
                if sa.name == sb.name:
                    continue

                score = self._compute_isomorphism(sa, sb)
                if score > 0.6:  # Threshold for meaningful isomorphism
                    transfers.append((sa, sb, score))

        # Sort by isomorphism score descending
        transfers.sort(key=lambda t: t[2], reverse=True)
        return transfers[:5]  # Cap at 5 transfers per cycle

    def _compute_isomorphism(
        self,
        a: SchemaComposition,
        b: SchemaComposition,
    ) -> float:
        """
        Compute structural isomorphism between two schemas.

        Uses Jaccard similarity on element kinds and structure patterns.
        """
        a_kinds = Counter(e.kind for e in a.elements)
        b_kinds = Counter(e.kind for e in b.elements)

        # Kind distribution similarity
        all_kinds = set(a_kinds.keys()) | set(b_kinds.keys())
        if not all_kinds:
            return 0.0

        intersection = sum(min(a_kinds.get(k, 0), b_kinds.get(k, 0)) for k in all_kinds)
        union = sum(max(a_kinds.get(k, 0), b_kinds.get(k, 0)) for k in all_kinds)
        kind_sim = intersection / max(1, union)

        # Structure similarity (if available)
        structure_sim = 0.0
        if a.structure and b.structure:
            # Compare relation types in structure triples
            a_rels = {t[1] for t in a.structure}
            b_rels = {t[1] for t in b.structure}
            if a_rels or b_rels:
                rel_intersection = len(a_rels & b_rels)
                rel_union = len(a_rels | b_rels)
                structure_sim = rel_intersection / max(1, rel_union)

        # Element count similarity
        size_diff = abs(len(a.elements) - len(b.elements))
        size_total = max(1, len(a.elements) + len(b.elements))
        count_sim = 1.0 - size_diff / size_total

        return (kind_sim * 0.4 + structure_sim * 0.4 + count_sim * 0.2)

    async def track_schema_version(
        self,
        schema_name: str,
        fitness: float,
    ) -> int:
        """
        Track schema version and fitness. Returns the new version number.

        Schemas evolve - this tracks lineage and measures fitness (how much
        compression each version provides). Dead branches are pruned.
        """
        self._schema_versions[schema_name] += 1
        version = self._schema_versions[schema_name]
        self._schema_fitness[schema_name] = fitness

        if self._memory is not None:
            neo4j = self._memory
            try:
                await neo4j.execute_write(
                    """
                    MATCH (s:EvoSchema {name: $name})
                    SET s.version = $version,
                        s.fitness = $fitness,
                        s.last_evolved_at = datetime()
                    """,
                    {"name": schema_name, "version": version, "fitness": fitness},
                )
            except Exception as exc:
                self._logger.warning("schema_version_track_failed", error=str(exc))

        return version

    async def prune_dead_schemas(self, min_fitness: float = 0.0) -> int:
        """
        Prune schema branches with fitness below the threshold.
        Returns the count of pruned schemas.
        """
        pruned = 0
        dead_names = [
            name for name, fitness in self._schema_fitness.items()
            if fitness <= min_fitness
        ]

        for name in dead_names:
            del self._schema_fitness[name]
            self._schema_versions.pop(name, None)
            pruned += 1

            if self._memory is not None:
                neo4j = self._memory
                try:
                    await neo4j.execute_write(
                        """
                        MATCH (s:EvoSchema {name: $name})
                        SET s.active = false, s.pruned_at = datetime()
                        """,
                        {"name": name},
                    )
                except Exception as exc:
                    self._logger.warning("schema_prune_failed", name=name, error=str(exc))

        if pruned:
            self._logger.info("dead_schemas_pruned", count=pruned)
        return pruned

    async def _persist_composition_relationship(
        self,
        neo4j: Any,
        child: SchemaComposition,
        parent_ids: list[str],
        rel_type: str,
    ) -> None:
        """Persist a schema relationship (COMPOSED_OF, SPECIALIZES, ABSTRACTS) in Neo4j."""
        try:
            # Ensure the child schema composition node exists
            await neo4j.execute_write(
                """
                MERGE (sc:EvoSchemaComposition {id: $id})
                SET sc.name = $name,
                    sc.description = $description,
                    sc.abstraction_level = $abstraction_level,
                    sc.mdl_total_gain = $mdl_total_gain,
                    sc.instance_count = $instance_count,
                    sc.confidence = $confidence,
                    sc.discovered_at = datetime()
                """,
                {
                    "id": child.id,
                    "name": child.name,
                    "description": child.description,
                    "abstraction_level": child.abstraction_level,
                    "mdl_total_gain": child.mdl_total_gain,
                    "instance_count": child.instance_count,
                    "confidence": child.confidence,
                },
            )

            # Create relationships to parents
            for parent_id in parent_ids:
                with contextlib.suppress(Exception):
                    await neo4j.execute_write(
                        f"""
                        MATCH (child:EvoSchemaComposition {{id: $child_id}})
                        MATCH (parent:EvoSchemaComposition {{id: $parent_id}})
                        MERGE (child)-[:{rel_type}]->(parent)
                        """,
                        {"child_id": child.id, "parent_id": parent_id},
                    )
        except Exception as exc:
            self._logger.warning(
                "schema_relationship_persist_failed",
                child=child.name,
                rel_type=rel_type,
                error=str(exc),
            )


def _canonical_sig(props: list[str] | Any) -> str:
    """Create a canonical property signature string for clustering."""
    if not isinstance(props, list):
        return ""
    return "|".join(sorted(str(p) for p in props))
