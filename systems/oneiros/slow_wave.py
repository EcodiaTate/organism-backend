"""
EcodiaOS — Oneiros v2: Slow Wave Stage (The Deep Compiler)

The heart of Oneiros. All systems are available for batch computation.
No real-time constraints. No user to respond to. No policy to execute.
Only: find the truth in what we experienced today.

Three operations:
1. Memory Ladder — four-rung compression from episodic to world model
2. Hypothesis Graveyard — MDL-based retirement of stale hypotheses
3. Causal Graph Reconstruction — rebuild causal structure from all evidence
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id
from systems.oneiros.types import (
    CausalReconstructionReport,
    HypothesisGraveyardReport,
    MemoryLadderReport,
    RungResult,
    SleepCheckpoint,
    SlowWaveReport,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.logos.service import LogosService
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oneiros.slow_wave")


# ═══════════════════════════════════════════════════════════════════
# Memory Ladder
# ═══════════════════════════════════════════════════════════════════


class MemoryLadder:
    """
    The Memory Ladder: compression in a single integrated pass.

    Four rungs, climbed bottom-up. You cannot skip rungs.
    A memory that cannot climb a rung is marked as:
    - anchor memory (irreducibly novel), or
    - decay-flagged (low MDL)

    Rung 1: Episodic -> Semantic — extract entities, relations, semantic nodes
    Rung 2: Semantic -> Schema — find repeated patterns, replace with schema + refs
    Rung 3: Schema -> Procedure — extract action-outcome sequences
    Rung 4: Procedure -> World Model — integrate as generative rules
    """

    def __init__(self, logos: LogosService | None = None) -> None:
        self._logos = logos
        self._logger = logger.bind(component="memory_ladder")

    async def run(
        self,
        uncompressed_episodes: list[dict[str, Any]],
    ) -> MemoryLadderReport:
        """
        Run the full four-rung memory ladder on uncompressed episodes.

        Each episode is a dict with at least: id, content, context, raw_complexity.
        """
        t0 = time.monotonic()
        total = len(uncompressed_episodes)
        self._logger.info("memory_ladder_starting", episodes=total)

        if total == 0:
            return MemoryLadderReport()

        # Rung 1: Episodic -> Semantic
        rung1, semantic_nodes = await self._rung1_episodic_to_semantic(
            uncompressed_episodes
        )

        # Rung 2: Semantic -> Schema
        rung2, schemas = await self._rung2_semantic_to_schema(semantic_nodes)

        # Rung 3: Schema -> Procedure
        rung3, procedures = await self._rung3_schema_to_procedure(schemas)

        # Rung 4: Procedure -> World Model
        rung4, world_model_updates = await self._rung4_procedure_to_world_model(
            procedures
        )

        # Compute overall compression ratio.
        # Each rung's compression_ratio = items_in / items_promoted (how many items
        # compressed into how few).  The overall ratio is the product of per-rung
        # ratios, representing the full 4-rung pipeline compression.
        overall_ratio = 1.0
        for rung in [rung1, rung2, rung3, rung4]:
            if rung.compression_ratio > 0:
                overall_ratio *= rung.compression_ratio

        report = MemoryLadderReport(
            memories_processed=total,
            semantic_nodes_created=rung1.items_promoted,
            schemas_created=rung2.items_promoted,
            procedures_extracted=rung3.items_promoted,
            world_model_updates=rung4.items_promoted,
            anchor_memories=sum(
                r.items_anchored for r in [rung1, rung2, rung3, rung4]
            ),
            compression_ratio=round(overall_ratio, 3),
            rung_details=[rung1, rung2, rung3, rung4],
        )

        elapsed = (time.monotonic() - t0) * 1000
        self._logger.info(
            "memory_ladder_complete",
            episodes=total,
            semantic_nodes=report.semantic_nodes_created,
            schemas=report.schemas_created,
            procedures=report.procedures_extracted,
            world_model_updates=report.world_model_updates,
            anchors=report.anchor_memories,
            ratio=report.compression_ratio,
            elapsed_ms=round(elapsed, 1),
        )

        return report

    async def _rung1_episodic_to_semantic(
        self,
        episodes: list[dict[str, Any]],
    ) -> tuple[RungResult, list[dict[str, Any]]]:
        """
        Rung 1: Episodic -> Semantic.

        For each uncompressed episode, extract entities and relations
        to create semantic nodes. Episodes that are fully predictable
        (low information content) get decay-flagged.
        """
        semantic_nodes: list[dict[str, Any]] = []
        promoted = 0
        anchored = 0
        decay_flagged = 0

        for ep in episodes:
            content = ep.get("content", {})
            info_content = ep.get("information_content", 0.5)

            # Extract entities from content values
            entities: list[str] = []
            relations: list[str] = []

            for key, value in content.items():
                if isinstance(value, str) and len(value) > 2:
                    entities.append(value)
                elif isinstance(value, dict):
                    relations.append(key)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            entities.append(item)

            if not entities and not relations:
                # Nothing extractable — check if anchor or decay
                if info_content > 0.8:
                    anchored += 1  # Highly novel but unstructured
                else:
                    decay_flagged += 1
                continue

            # Create semantic node
            node = {
                "id": new_id(),
                "source_episode_id": ep.get("id", ""),
                "entities": entities,
                "relations": relations,
                "domain": content.get("domain", "general"),
                "raw_complexity": ep.get("raw_complexity", 1.0),
                "information_content": info_content,
            }
            semantic_nodes.append(node)
            promoted += 1

        ratio = len(episodes) / max(promoted, 1) if promoted > 0 else 1.0

        return RungResult(
            rung=1,
            items_in=len(episodes),
            items_promoted=promoted,
            items_anchored=anchored,
            items_decay_flagged=decay_flagged,
            compression_ratio=round(ratio, 3),
        ), semantic_nodes

    async def _rung2_semantic_to_schema(
        self,
        semantic_nodes: list[dict[str, Any]],
    ) -> tuple[RungResult, list[dict[str, Any]]]:
        """
        Rung 2: Semantic -> Schema.

        Find repeated semantic patterns. When 3+ nodes share entity structure,
        replace with 1 schema + N lightweight refs.
        """
        if not semantic_nodes:
            return RungResult(rung=2), []

        # Group nodes by entity structure (domain + sorted entity types)
        pattern_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in semantic_nodes:
            # Create a structural fingerprint from the entity set
            domain = node.get("domain", "general")
            entity_sig = ",".join(sorted(set(node.get("entities", []))))
            key = f"{domain}:{entity_sig}"
            pattern_groups[key].append(node)

        schemas: list[dict[str, Any]] = []
        promoted = 0
        anchored = 0
        decay_flagged = 0

        for pattern_key, group in pattern_groups.items():
            if len(group) >= 3:
                # Sufficient repetition — create a schema
                schema = {
                    "id": new_id(),
                    "pattern_key": pattern_key,
                    "domain": group[0].get("domain", "general"),
                    "entities": group[0].get("entities", []),
                    "relations": group[0].get("relations", []),
                    "instance_count": len(group),
                    "instance_refs": [n["id"] for n in group],
                    "raw_complexity": sum(
                        n.get("raw_complexity", 1.0) for n in group
                    ),
                }
                schemas.append(schema)
                promoted += 1
            else:
                # Not enough repetition — check info content
                for node in group:
                    info = node.get("information_content", 0.5)
                    if info > 0.8:
                        anchored += 1
                    elif info < 0.2:
                        decay_flagged += 1

        ratio = (
            len(semantic_nodes) / max(len(schemas), 1)
            if schemas
            else 1.0
        )

        return RungResult(
            rung=2,
            items_in=len(semantic_nodes),
            items_promoted=promoted,
            items_anchored=anchored,
            items_decay_flagged=decay_flagged,
            compression_ratio=round(ratio, 3),
        ), schemas

    async def _rung3_schema_to_procedure(
        self,
        schemas: list[dict[str, Any]],
    ) -> tuple[RungResult, list[dict[str, Any]]]:
        """
        Rung 3: Schema -> Procedure.

        Find schemas that describe action-outcome sequences.
        Extract as explicit procedures for Nova's procedure library.
        """
        if not schemas:
            return RungResult(rung=3), []

        procedures: list[dict[str, Any]] = []
        promoted = 0
        anchored = 0

        for schema in schemas:
            relations = schema.get("relations", [])
            # Heuristic: schemas with relational keys suggest action-outcome patterns
            has_actions = any(
                r in ("action", "outcome", "cause", "effect", "step", "result")
                for r in relations
            )

            if has_actions or len(relations) >= 2:
                procedure = {
                    "id": new_id(),
                    "source_schema_id": schema["id"],
                    "domain": schema.get("domain", "general"),
                    "entities": schema.get("entities", []),
                    "relations": relations,
                    "instance_count": schema.get("instance_count", 0),
                    "raw_complexity": schema.get("raw_complexity", 1.0),
                }
                procedures.append(procedure)
                promoted += 1
            else:
                # Schema without procedural content — anchor if high complexity
                if schema.get("raw_complexity", 0) > 10.0:
                    anchored += 1

        ratio = len(schemas) / max(promoted, 1) if promoted > 0 else 1.0

        return RungResult(
            rung=3,
            items_in=len(schemas),
            items_promoted=promoted,
            items_anchored=anchored,
            items_decay_flagged=0,
            compression_ratio=round(ratio, 3),
        ), procedures

    async def _rung4_procedure_to_world_model(
        self,
        procedures: list[dict[str, Any]],
    ) -> tuple[RungResult, int]:
        """
        Rung 4: Procedure -> World Model.

        Find procedures that reflect invariant causal structure.
        Integrate as generative rules via Logos world model.
        This is the most compressed form.
        """
        if not procedures:
            return RungResult(rung=4), 0

        updates = 0
        anchored = 0

        for proc in procedures:
            # Only procedures with sufficient instances reflect invariant structure
            instance_count = proc.get("instance_count", 0)

            if instance_count >= 3 and self._logos is not None:
                # Integrate as a generative rule
                from systems.logos.types import (
                    ExperienceDelta,
                    SemanticDelta,
                )

                delta = ExperienceDelta(
                    experience_id=proc["id"],
                    delta_content=SemanticDelta(
                        information_content=0.5,
                        novel_entities=proc.get("entities", []),
                        novel_relations=[
                            f"{e}->{proc.get('domain', 'general')}"
                            for e in proc.get("entities", [])[:3]
                        ],
                        content={
                            "domain": proc.get("domain", "general"),
                            "source": "oneiros_memory_ladder",
                            "instance_count": instance_count,
                        },
                    ),
                    information_content=0.5,
                    world_model_update_required=True,
                )
                await self._logos.integrate_delta(delta)
                updates += 1
            elif instance_count >= 3:
                # No Logos — can't integrate, mark as anchor
                anchored += 1
            else:
                # Not enough instances for invariant status — anchor
                anchored += 1

        ratio = len(procedures) / max(updates, 1) if updates > 0 else 1.0

        return RungResult(
            rung=4,
            items_in=len(procedures),
            items_promoted=updates,
            items_anchored=anchored,
            items_decay_flagged=0,
            compression_ratio=round(ratio, 3),
        ), updates


# ═══════════════════════════════════════════════════════════════════
# Hypothesis Graveyard
# ═══════════════════════════════════════════════════════════════════


class HypothesisGraveyard:
    """
    Resolve long-running hypotheses using MDL scoring.

    Hypotheses with bad compression ratios that have survived multiple
    sleep cycles get retired. This frees cognitive budget.
    """

    # Retire if compression ratio below this threshold
    RETIREMENT_RATIO_THRESHOLD: float = 1.0
    # Must have been around for at least this many sleep cycles to retire
    MIN_CYCLES_FOR_RETIREMENT: int = 3

    def __init__(self, logos: LogosService | None = None) -> None:
        self._logos = logos
        self._logger = logger.bind(component="hypothesis_graveyard")

    async def process(
        self,
        hypotheses: list[dict[str, Any]],
    ) -> HypothesisGraveyardReport:
        """
        Evaluate all active hypotheses using MDL scoring.

        Good MDL (compression_ratio >= threshold) -> confirmed (kept)
        Bad MDL + multiple cycles -> retired (freed)
        Bad MDL + few cycles -> deferred (give more time)
        """
        t0 = time.monotonic()
        total = len(hypotheses)
        self._logger.info("graveyard_starting", hypotheses=total)

        confirmed = 0
        retired = 0
        deferred = 0
        total_mdl_freed = 0.0

        for hyp in hypotheses:
            hyp_id = hyp.get("id", "unknown")
            description = hyp.get("description", "")
            supporting = hyp.get("supporting_observations", [])
            cycles_seen = hyp.get("sleep_cycles_seen", 0)

            # Estimate compression ratio: observations explained / description length
            obs_complexity = sum(
                (o.get("complexity", 1.0) if isinstance(o, dict) else getattr(o, "complexity", 1.0))
                for o in supporting
            )
            # Shannon entropy approximation: ~4.5 bits per char
            desc_length = max(len(description) * 4.5, 1.0)
            compression_ratio = obs_complexity / desc_length

            if compression_ratio >= self.RETIREMENT_RATIO_THRESHOLD:
                # Good hypothesis — confirm
                confirmed += 1
                self._logger.debug(
                    "hypothesis_confirmed",
                    id=hyp_id,
                    ratio=round(compression_ratio, 3),
                )
            elif cycles_seen >= self.MIN_CYCLES_FOR_RETIREMENT:
                # Bad ratio + enough time -> retire
                retired += 1
                total_mdl_freed += desc_length
                self._logger.info(
                    "hypothesis_retired",
                    id=hyp_id,
                    ratio=round(compression_ratio, 3),
                    cycles=cycles_seen,
                    bits_freed=round(desc_length, 1),
                )
            else:
                # Bad ratio but not enough time — defer
                deferred += 1
                self._logger.debug(
                    "hypothesis_deferred",
                    id=hyp_id,
                    ratio=round(compression_ratio, 3),
                    cycles=cycles_seen,
                )

        elapsed = (time.monotonic() - t0) * 1000
        self._logger.info(
            "graveyard_complete",
            total=total,
            confirmed=confirmed,
            retired=retired,
            deferred=deferred,
            mdl_freed=round(total_mdl_freed, 1),
            elapsed_ms=round(elapsed, 1),
        )

        return HypothesisGraveyardReport(
            hypotheses_evaluated=total,
            hypotheses_confirmed=confirmed,
            hypotheses_retired=retired,
            hypotheses_deferred=deferred,
            total_mdl_freed=total_mdl_freed,
        )


# ═══════════════════════════════════════════════════════════════════
# Causal Graph Reconstruction
# ═══════════════════════════════════════════════════════════════════


class CausalGraphReconstructor:
    """
    Rebuild the causal graph from all available evidence.

    Steps:
    1. Build full correlation matrix from all causal observations
    2. Run simplified causal discovery (PC algorithm approximation)
    3. Validate against existing Logos causal model
    4. Resolve contradictions using evidence weight
    5. Extract causal invariants (rules holding in >95% of contexts)
    6. Update Logos world model causal structure
    """

    # Minimum correlation strength to consider a link
    CORRELATION_THRESHOLD: float = 0.3
    # Invariant must hold in this fraction of contexts
    INVARIANT_CONFIDENCE: float = 0.95

    def __init__(self, logos: LogosService | None = None) -> None:
        self._logos = logos
        self._logger = logger.bind(component="causal_reconstruction")

    async def reconstruct(
        self,
        causal_observations: list[dict[str, Any]],
    ) -> CausalReconstructionReport:
        """
        Full causal graph reconstruction from all available observations.
        """
        t0 = time.monotonic()
        total = len(causal_observations)
        self._logger.info("reconstruction_starting", observations=total)

        if total == 0:
            return CausalReconstructionReport()

        # Step 1: Build correlation matrix
        correlation_matrix = self._build_correlation_matrix(causal_observations)

        # Step 2: Run simplified PC algorithm for direction inference
        candidate_graph = self._run_pc_algorithm(correlation_matrix)

        # Step 3 & 4: Validate against existing model and resolve contradictions
        contradictions_resolved = 0
        if self._logos is not None:
            existing_graph = self._logos.world_model.causal_structure
            contradictions_resolved = await self._resolve_contradictions(
                candidate_graph, existing_graph, causal_observations
            )

        # Step 5: Extract causal invariants
        invariants = self._extract_invariants(
            candidate_graph, causal_observations
        )

        # Step 6: Update world model
        if self._logos is not None:
            await self._update_world_model(candidate_graph, invariants)

        # Measure change magnitude
        change_magnitude = self._measure_change_magnitude(candidate_graph)

        elapsed = (time.monotonic() - t0) * 1000
        nodes = len(candidate_graph.get("nodes", set()))
        edges = len(candidate_graph.get("edges", []))

        self._logger.info(
            "reconstruction_complete",
            nodes=nodes,
            edges=edges,
            contradictions=contradictions_resolved,
            invariants=len(invariants),
            change=round(change_magnitude, 3),
            elapsed_ms=round(elapsed, 1),
        )

        return CausalReconstructionReport(
            nodes_in_graph=nodes,
            edges_in_graph=edges,
            contradictions_resolved=contradictions_resolved,
            invariants_discovered=len(invariants),
            change_magnitude=change_magnitude,
        )

    def _build_correlation_matrix(
        self,
        observations: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """
        Build a co-occurrence / correlation matrix from all causal observations.

        Each observation is expected to have 'cause' and 'effect' fields.
        We count co-occurrences and normalize to [0, 1].
        """
        co_occurrence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        entity_counts: dict[str, int] = defaultdict(int)

        for obs in observations:
            cause = obs.get("cause", "")
            effect = obs.get("effect", "")
            if cause and effect:
                co_occurrence[cause][effect] += 1
                entity_counts[cause] += 1
                entity_counts[effect] += 1

        # Normalize to correlation strength [0, 1]
        matrix: dict[str, dict[str, float]] = {}
        for cause, effects in co_occurrence.items():
            matrix[cause] = {}
            for effect, count in effects.items():
                # Correlation = co-occurrence / sqrt(count_cause * count_effect)
                denom = (entity_counts[cause] * entity_counts[effect]) ** 0.5
                matrix[cause][effect] = count / denom if denom > 0 else 0.0

        return matrix

    def _run_pc_algorithm(
        self,
        correlation_matrix: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """
        Simplified PC algorithm for causal direction inference.

        Full PC algorithm:
        1. Start with complete undirected graph
        2. Remove edges with low correlation
        3. Orient edges using d-separation tests

        Simplified version:
        1. Keep edges above correlation threshold
        2. Orient by asymmetry: if corr(A->B) >> corr(B->A), direction is A->B
        3. Remove weak bidirectional edges (likely common cause)
        """
        nodes: set[str] = set()
        edges: list[dict[str, Any]] = []

        for cause, effects in correlation_matrix.items():
            nodes.add(cause)
            for effect, strength in effects.items():
                nodes.add(effect)
                if strength < self.CORRELATION_THRESHOLD:
                    continue

                # Check asymmetry for direction
                reverse_strength = (
                    correlation_matrix.get(effect, {}).get(cause, 0.0)
                )

                if strength > reverse_strength * 1.5:
                    # Clearly directional: cause -> effect
                    edges.append({
                        "cause": cause,
                        "effect": effect,
                        "strength": strength,
                        "observations": int(strength * 100),
                    })
                elif reverse_strength > strength * 1.5:
                    # Reverse direction — captured when iterating from effect
                    pass
                elif strength > 0.5 and reverse_strength > 0.5:
                    # Strong bidirectional — likely common cause, skip
                    pass
                else:
                    # Weak but directional enough — keep the stronger direction
                    if strength > reverse_strength:
                        edges.append({
                            "cause": cause,
                            "effect": effect,
                            "strength": strength,
                            "observations": int(strength * 100),
                        })

        return {"nodes": nodes, "edges": edges}

    async def _resolve_contradictions(
        self,
        candidate: dict[str, Any],
        existing_graph: Any,  # CausalGraph
        observations: list[dict[str, Any]],
    ) -> int:
        """
        Find contradictions between candidate and existing causal model.
        Resolve using evidence weight.
        """
        contradictions = 0
        candidate_edges = candidate.get("edges", [])

        for edge in candidate_edges:
            cause = edge["cause"]
            effect = edge["effect"]

            # Check if existing graph has the reverse direction
            reverse_key = f"{effect}->{cause}"
            forward_key = f"{cause}->{effect}"

            existing_links = (
                existing_graph.links if hasattr(existing_graph, "links") else {}
            )

            if reverse_key in existing_links and forward_key not in existing_links:
                # Contradiction: candidate says A->B, existing says B->A
                existing_strength = existing_links[reverse_key].strength
                candidate_strength = edge["strength"]

                if candidate_strength > existing_strength:
                    # New evidence wins — remove old, add new
                    existing_graph.revise_link(effect, cause, 0.0)
                    contradictions += 1
                    self._logger.info(
                        "contradiction_resolved",
                        cause=cause,
                        effect=effect,
                        old_direction=f"{effect}->{cause}",
                        new_direction=f"{cause}->{effect}",
                    )

        return contradictions

    def _extract_invariants(
        self,
        candidate: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Extract causal invariants: rules that hold in >95% of contexts.
        """
        edge_contexts: dict[str, list[str]] = defaultdict(list)
        all_contexts: set[str] = set()

        for obs in observations:
            cause = obs.get("cause", "")
            effect = obs.get("effect", "")
            context = obs.get("domain", obs.get("context", "general"))
            if cause and effect:
                key = f"{cause}->{effect}"
                edge_contexts[key].append(context)
                all_contexts.add(context)

        invariants: list[dict[str, Any]] = []
        total_contexts = max(len(all_contexts), 1)

        for edge in candidate.get("edges", []):
            key = f"{edge['cause']}->{edge['effect']}"
            contexts = set(edge_contexts.get(key, []))
            coverage = len(contexts) / total_contexts

            if coverage >= self.INVARIANT_CONFIDENCE:
                invariants.append({
                    "id": new_id(),
                    "statement": f"{edge['cause']} causes {edge['effect']}",
                    "cause": edge["cause"],
                    "effect": edge["effect"],
                    "confidence": coverage,
                    "observation_count": len(edge_contexts.get(key, [])),
                    "domain": (
                        "cross_domain"
                        if len(contexts) > 1
                        else next(iter(contexts), "general")
                    ),
                })

        return invariants

    async def _update_world_model(
        self,
        candidate: dict[str, Any],
        invariants: list[dict[str, Any]],
    ) -> None:
        """Update Logos world model with reconstructed causal structure."""
        if self._logos is None:
            return

        from systems.logos.types import CausalLink, EmpiricalInvariant

        wm = self._logos.world_model

        # Add new causal links
        for edge in candidate.get("edges", []):
            link = CausalLink(
                cause_id=edge["cause"],
                effect_id=edge["effect"],
                strength=edge["strength"],
                domain=edge.get("domain", "general"),
                observations=edge.get("observations", 1),
            )
            wm.causal_structure.add_link(link)

        # Ingest invariants
        for inv in invariants:
            empirical = EmpiricalInvariant(
                id=inv["id"],
                statement=inv["statement"],
                domain=inv.get("domain", "general"),
                observation_count=inv.get("observation_count", 0),
                confidence=inv.get("confidence", 1.0),
                source="oneiros_causal_reconstruction",
            )
            wm.ingest_invariant(empirical)

        # Prune weak links
        pruned = wm.causal_structure.remove_weak_links(threshold=0.05)
        if pruned > 0:
            self._logger.info("weak_links_pruned", count=pruned)

    def _measure_change_magnitude(self, candidate: dict[str, Any]) -> float:
        """
        Measure how much the causal graph changed relative to existing.
        0.0 = no change, 1.0 = complete rebuild.
        """
        if self._logos is None:
            return 1.0  # No existing model — everything is new

        existing_count = self._logos.world_model.causal_structure.link_count
        candidate_count = len(candidate.get("edges", []))

        if existing_count == 0 and candidate_count == 0:
            return 0.0
        if existing_count == 0:
            return 1.0

        # Approximate change as fraction of new edges relative to existing
        return float(min(candidate_count / max(existing_count, 1), 1.0))


# ═══════════════════════════════════════════════════════════════════
# Slow Wave Stage Orchestrator
# ═══════════════════════════════════════════════════════════════════


class SlowWaveStage:
    """
    Stage 2: Slow Wave (~50% of sleep duration).

    The batch compiler. Orchestrates:
    1. Memory Ladder compression (4 rungs)
    2. Hypothesis Graveyard (MDL-based retirement)
    3. Causal Graph Reconstruction (PC algorithm)

    Broadcasts COMPRESSION_BACKLOG_PROCESSED and CAUSAL_GRAPH_RECONSTRUCTED.
    """

    def __init__(
        self,
        logos: LogosService | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._logos = logos
        self._event_bus = event_bus
        self._memory_ladder = MemoryLadder(logos=logos)
        self._graveyard = HypothesisGraveyard(logos=logos)
        self._reconstructor = CausalGraphReconstructor(logos=logos)
        self._logger = logger.bind(stage="slow_wave")

    async def execute(
        self,
        checkpoint: SleepCheckpoint,
        uncompressed_episodes: list[dict[str, Any]] | None = None,
        active_hypotheses: list[dict[str, Any]] | None = None,
        causal_observations: list[dict[str, Any]] | None = None,
    ) -> SlowWaveReport:
        """
        Execute the full Slow Wave stage.
        """
        t0 = time.monotonic()
        self._logger.info("slow_wave_starting", checkpoint_id=checkpoint.id)

        # 1. Memory Ladder compression
        compression_report = await self._memory_ladder.run(
            uncompressed_episodes or []
        )
        await self._broadcast_compression_processed(compression_report)

        # 2. Hypothesis Graveyard
        hypothesis_report = await self._graveyard.process(
            active_hypotheses or []
        )

        # 3. Causal Graph Reconstruction
        causal_report = await self._reconstructor.reconstruct(
            causal_observations or []
        )
        await self._broadcast_causal_reconstructed(causal_report)

        elapsed = (time.monotonic() - t0) * 1000

        report = SlowWaveReport(
            compression=compression_report,
            hypotheses=hypothesis_report,
            causal=causal_report,
            duration_ms=elapsed,
        )

        self._logger.info(
            "slow_wave_complete",
            memories_processed=compression_report.memories_processed,
            schemas_created=compression_report.schemas_created,
            hypotheses_retired=hypothesis_report.hypotheses_retired,
            invariants_discovered=causal_report.invariants_discovered,
            elapsed_ms=round(elapsed, 1),
        )

        return report

    async def _broadcast_compression_processed(
        self, report: MemoryLadderReport
    ) -> None:
        if self._event_bus is None:
            return

        event = SynapseEvent(
            event_type=SynapseEventType.COMPRESSION_BACKLOG_PROCESSED,
            source_system="oneiros",
            data={
                "memories_processed": report.memories_processed,
                "semantic_nodes_created": report.semantic_nodes_created,
                "schemas_created": report.schemas_created,
                "procedures_extracted": report.procedures_extracted,
                "world_model_updates": report.world_model_updates,
                "anchor_memories": report.anchor_memories,
                "compression_ratio": report.compression_ratio,
            },
        )
        await self._event_bus.emit(event)

    async def _broadcast_causal_reconstructed(
        self, report: CausalReconstructionReport
    ) -> None:
        if self._event_bus is None:
            return

        event = SynapseEvent(
            event_type=SynapseEventType.CAUSAL_GRAPH_RECONSTRUCTED,
            source_system="oneiros",
            data={
                "nodes_in_graph": report.nodes_in_graph,
                "edges_in_graph": report.edges_in_graph,
                "contradictions_resolved": report.contradictions_resolved,
                "invariants_discovered": report.invariants_discovered,
                "change_magnitude": report.change_magnitude,
            },
        )
        await self._event_bus.emit(event)
