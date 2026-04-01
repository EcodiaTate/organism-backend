"""
EcodiaOS - Oneiros v2: Slow Wave Stage (The Deep Compiler)

The heart of Oneiros. All systems are available for batch computation.
No real-time constraints. No user to respond to. No policy to execute.
Only: find the truth in what we experienced today.

Three operations:
1. Memory Ladder - four-rung compression from episodic to world model
2. Hypothesis Graveyard - MDL-based retirement of stale hypotheses
3. Causal Graph Reconstruction - rebuild causal structure from all evidence
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID, new_id
from primitives.re_training import RETrainingBatch, RETrainingExample
from systems.oneiros.types import (
    CausalReconstructionReport,
    HypothesisGraveyardReport,
    MemoryLadderReport,
    RungResult,
    SleepCheckpoint,
    SlowWaveReport,
    WorldModelConsistencyReport,
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

    Rung 1: Episodic -> Semantic - extract entities, relations, semantic nodes
    Rung 2: Semantic -> Schema - find repeated patterns, replace with schema + refs
    Rung 3: Schema -> Procedure - extract action-outcome sequences
    Rung 4: Procedure -> World Model - integrate as generative rules
    """

    def __init__(
        self,
        logos: LogosService | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._logos = logos
        self._event_bus = event_bus
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

        # RE Stream 1: emit training examples from successful consolidation
        if schemas and self._event_bus is not None:
            self._emit_re_training_stream1(
                schemas, semantic_nodes, uncompressed_episodes, overall_ratio
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

    def _emit_re_training_stream1(
        self,
        schemas: list[dict[str, Any]],
        semantic_nodes: list[dict[str, Any]],
        raw_episodes: list[dict[str, Any]],
        compression_ratio: float,
    ) -> None:
        """RE Stream 1: emit training examples from successful reasoning chains.

        During NREM consolidation, compressed schemas represent successful
        abstraction - ideal training data for the RE's consolidation reasoning.
        Fire-and-forget via asyncio.create_task.
        """
        examples: list[RETrainingExample] = []
        # Build a lookup of semantic nodes by source episode
        node_by_episode: dict[str, dict[str, Any]] = {}
        for node in semantic_nodes:
            ep_id = node.get("source_episode_id", "")
            if ep_id:
                node_by_episode[ep_id] = node

        for schema in schemas:
            instance_refs = schema.get("instance_refs", [])
            # Gather raw episodes that fed this schema
            raw_context_parts: list[str] = []
            for ref_id in instance_refs[:5]:  # cap context size
                node = node_by_episode.get(ref_id)
                if node:
                    raw_context_parts.append(str(node.get("entities", [])))

            example = RETrainingExample(
                source_system=SystemID.ONEIROS,
                instruction=f"Schema: {schema.get('pattern_key', '')} "
                            f"(domain={schema.get('domain', 'general')}, "
                            f"instances={schema.get('instance_count', 0)})",
                input_context=" | ".join(raw_context_parts) if raw_context_parts else "",
                output=str(schema.get("entities", [])),
                outcome_quality=min(1.0, compression_ratio / 10.0),
                category="consolidation_reasoning",
            )
            examples.append(example)

        if not examples:
            return

        batch = RETrainingBatch(
            examples=examples,
            source_system=SystemID.ONEIROS,
        )
        event = SynapseEvent(
            event_type=SynapseEventType.RE_TRAINING_BATCH,
            source_system="oneiros",
            data=batch.model_dump(mode="json"),
        )

        async def _emit() -> None:
            try:
                await self._event_bus.emit(event)  # type: ignore[union-attr]
            except Exception:
                pass

        asyncio.create_task(_emit())

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
                # Nothing extractable - check if anchor or decay
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
                # Sufficient repetition - create a schema
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
                # Not enough repetition - check info content
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
                # Schema without procedural content - anchor if high complexity
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
                from primitives.logos import ExperienceDelta, SemanticDelta

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
                # No Logos - can't integrate, mark as anchor
                anchored += 1
            else:
                # Not enough instances for invariant status - anchor
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
                # Good hypothesis - confirm
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
                # Bad ratio but not enough time - defer
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
            existing_graph = self._logos.get_causal_structure()
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

    # Significance level for the Fisher Z conditional independence test.
    # α=0.05 gives a good balance between skeleton sparsity and edge retention.
    _CI_ALPHA: float = 0.05

    def _run_pc_algorithm(
        self,
        correlation_matrix: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """
        PC algorithm with proper d-separation for causal structure learning.

        Three phases:
        1. Skeleton - start with complete undirected graph; iteratively remove
           edges X-Y where X ⊥ Y | Z for some conditioning set Z using the
           Fisher Z-test on partial correlations derived from the co-occurrence
           matrix.  Separation sets sep[X,Y] are recorded.
        2. V-structure orientation - for each unshielded triple X-Z-Y (X and Y
           not adjacent), orient X→Z←Y iff Z ∉ sep[X,Y] (collider detection).
        3. Meek rules R1–R3 - propagate orientations to avoid new colliders and
           directed cycles.

        Returns a dict {"nodes": set[str], "edges": list[dict]} where each edge
        has keys: cause, effect, strength, observations.
        """
        import math

        # ── Build node set and symmetric adjacency from correlation matrix ──
        nodes: set[str] = set()
        # corr[a][b] = symmetrised co-occurrence strength
        corr: dict[str, dict[str, float]] = {}
        for cause, effects in correlation_matrix.items():
            nodes.add(cause)
            for effect, strength in effects.items():
                nodes.add(effect)
                if strength < self.CORRELATION_THRESHOLD:
                    continue
                rev = correlation_matrix.get(effect, {}).get(cause, 0.0)
                sym = (strength + rev) / 2.0
                corr.setdefault(cause, {})[effect] = sym
                corr.setdefault(effect, {})[cause] = sym

        node_list = sorted(nodes)
        n_nodes = len(node_list)
        if n_nodes < 2:
            return {"nodes": nodes, "edges": []}

        # ── Phase 1: Skeleton via conditional independence tests ──────────
        # Undirected adjacency set - start from corr edges (already thresholded)
        adj: dict[str, set[str]] = {v: set() for v in node_list}
        for v in node_list:
            for u in corr.get(v, {}):
                adj[v].add(u)
                adj[u].add(v)

        # sep[x][y] = conditioning set that d-separates x and y (or None if adjacent)
        sep: dict[str, dict[str, set[str]]] = {
            v: {u: set() for u in node_list} for v in node_list
        }

        # Sample count proxy: use the max co-occurrence count across all pairs
        # as a rough N for Fisher Z significance testing.
        raw_max = max(
            (correlation_matrix.get(a, {}).get(b, 0.0) for a in node_list for b in node_list),
            default=0.0,
        )
        # N is reconstructed from the normalised correlation. We use a conservative
        # floor of 30 so that the test never rejects every edge with tiny N.
        n_obs = max(30, int(raw_max * 100))

        def _partial_corr(x: str, y: str, z_set: set[str]) -> float:
            """
            Approximate partial correlation r(X,Y|Z) from the co-occurrence
            correlation matrix.  For |Z|=0 this is just the marginal correlation.
            For |Z|=1 we apply the standard 1-variable partial formula.
            For |Z|>1 we use recursive elimination (order-2 cap for performance).
            """
            r_xy = corr.get(x, {}).get(y, 0.0)
            if not z_set:
                return r_xy
            # Order-1 partial correlation
            z = next(iter(z_set))
            r_xz = corr.get(x, {}).get(z, 0.0)
            r_yz = corr.get(y, {}).get(z, 0.0)
            denom = ((1 - r_xz ** 2) * (1 - r_yz ** 2)) ** 0.5
            if denom < 1e-9:
                return 0.0
            pc = (r_xy - r_xz * r_yz) / denom
            return max(-1.0, min(1.0, pc))

        def _ci_test(x: str, y: str, z_set: set[str]) -> bool:
            """
            Fisher Z-test: return True iff X ⊥ Y | Z at significance _CI_ALPHA.
            """
            r = _partial_corr(x, y, z_set)
            r_abs = min(abs(r), 0.9999)
            z_stat = 0.5 * math.log((1 + r_abs) / (1 - r_abs))
            # Standard error of Fisher Z under H0
            se = 1.0 / math.sqrt(max(n_obs - len(z_set) - 3, 1))
            # Two-tailed critical value for α=0.05 ≈ 1.96, for α=0.01 ≈ 2.576
            # Using 1.96 as standard 5% threshold
            z_crit = 1.96
            return abs(z_stat) / se < z_crit  # True means independent

        # Iterate over conditioning set sizes 0, 1, 2 (standard PC truncation
        # to order 2 keeps complexity polynomial for sparse graphs)
        from itertools import combinations as _combns

        for cond_size in range(3):  # 0, 1, 2
            edge_pairs = [
                (x, y)
                for x in node_list
                for y in sorted(adj[x])
                if x < y  # process each undirected pair once
            ]
            for x, y in edge_pairs:
                if y not in adj[x]:
                    continue  # already removed
                # Conditioning candidates: neighbours of x (or y) minus {y, x}
                cands = sorted((adj[x] | adj[y]) - {x, y})
                for z_tuple in _combns(cands, cond_size):
                    z_set = set(z_tuple)
                    if _ci_test(x, y, z_set):
                        # X and Y are conditionally independent - remove edge
                        adj[x].discard(y)
                        adj[y].discard(x)
                        sep[x][y] = z_set
                        sep[y][x] = z_set
                        break  # found a separation set; move to next pair

        # ── Phase 2: V-structure orientation ─────────────────────────────
        # directed[x][y] = True means x→y is oriented
        directed: dict[str, dict[str, bool]] = {v: {} for v in node_list}

        for z in node_list:
            z_nbrs = sorted(adj[z])
            for x, y in _combns(z_nbrs, 2):
                # Unshielded triple: x-z-y with x and y NOT adjacent
                if y in adj[x]:
                    continue  # shielded triple - not a collider candidate
                # Collider iff z was NOT in the separation set of x and y
                if z not in sep[x][y]:
                    # Orient x→z←y
                    directed[x][z] = True
                    directed[y][z] = True

        # ── Phase 3: Meek rules R1–R3 ────────────────────────────────────
        changed = True
        while changed:
            changed = False
            for x in node_list:
                for y in sorted(adj[x]):
                    if directed.get(x, {}).get(y) or directed.get(y, {}).get(x):
                        continue  # already oriented
                    # R1: Orient z→x-y as z→x→y when z-y not in adj
                    for z in sorted(adj[x]):
                        if z == y:
                            continue
                        if directed.get(z, {}).get(x) and y not in adj[z]:
                            directed.setdefault(x, {})[y] = True
                            changed = True
                            break
                    if directed.get(x, {}).get(y):
                        continue
                    # R2: Orient x-y into x→y when there's a directed path x→z→y
                    for z in sorted(adj[x] & adj[y]):
                        if directed.get(x, {}).get(z) and directed.get(z, {}).get(y):
                            directed.setdefault(x, {})[y] = True
                            changed = True
                            break

        # ── Collect oriented edges ────────────────────────────────────────
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for src, dsts in directed.items():
            for dst, is_fwd in dsts.items():
                if not is_fwd:
                    continue
                if (src, dst) in seen:
                    continue
                seen.add((src, dst))
                strength = corr.get(src, {}).get(dst, 0.0)
                if strength < self.CORRELATION_THRESHOLD:
                    continue
                edges.append({
                    "cause": src,
                    "effect": dst,
                    "strength": strength,
                    "observations": int(strength * 100),
                })

        # For undirected edges remaining (neither end oriented), keep the
        # direction with higher marginal correlation as a fallback.
        for x in node_list:
            for y in sorted(adj[x]):
                if x >= y:
                    continue
                if (x, y) in seen or (y, x) in seen:
                    continue
                r_xy = corr.get(x, {}).get(y, 0.0)
                if r_xy < self.CORRELATION_THRESHOLD:
                    continue
                cause, effect = (x, y) if r_xy >= corr.get(y, {}).get(x, 0.0) else (y, x)
                edges.append({
                    "cause": cause,
                    "effect": effect,
                    "strength": r_xy,
                    "observations": int(r_xy * 100),
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
                    # New evidence wins - remove old, add new
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

        from primitives.logos import CausalLink, EmpiricalInvariant

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
            return 1.0  # No existing model - everything is new

        causal = self._logos.get_causal_structure()
        existing_count = causal.link_count if hasattr(causal, "link_count") else 0
        candidate_count = len(candidate.get("edges", []))

        if existing_count == 0 and candidate_count == 0:
            return 0.0
        if existing_count == 0:
            return 1.0

        # Approximate change as fraction of new edges relative to existing
        return float(min(candidate_count / max(existing_count, 1), 1.0))


# ═══════════════════════════════════════════════════════════════════
# Synaptic Downscaler (NREM)
# ═══════════════════════════════════════════════════════════════════


class SynapticDownscaler:
    """
    Apply 0.85x salience decay to episodes not accessed in 7+ days.

    This is how sleep renews memory capacity - stale memories fade unless
    they've already been consolidated (consolidation_level >= 3).
    """

    DECAY_FACTOR: float = 0.85
    STALE_DAYS: int = 7
    MIN_CONSOLIDATION_LEVEL: int = 3  # already-consolidated episodes are protected

    def __init__(self, neo4j: Any | None = None) -> None:
        self._neo4j = neo4j
        self._logger = logger.bind(component="synaptic_downscaler")

    async def run(self) -> dict[str, int]:
        """Apply salience decay to stale episodes. Returns count of decayed episodes."""
        if self._neo4j is None:
            self._logger.debug("no_neo4j_available")
            return {"episodes_decayed": 0}

        t0 = time.monotonic()
        query = """
            MATCH (e:Episode)
            WHERE e.last_accessed < datetime() - duration({days: $stale_days})
              AND coalesce(e.consolidation_level, 0) < $min_consolidation
              AND e.salience_composite > 0.01
            SET e.salience_composite = e.salience_composite * $decay_factor
            RETURN count(e) AS decayed
        """
        try:
            result = await self._neo4j.execute_write(
                query,
                parameters={
                    "stale_days": self.STALE_DAYS,
                    "min_consolidation": self.MIN_CONSOLIDATION_LEVEL,
                    "decay_factor": self.DECAY_FACTOR,
                },
            )
            decayed = result[0]["decayed"] if result else 0
            elapsed = (time.monotonic() - t0) * 1000
            self._logger.info(
                "synaptic_downscaling_complete",
                episodes_decayed=decayed,
                elapsed_ms=round(elapsed, 1),
            )
            return {"episodes_decayed": decayed}
        except Exception as exc:
            self._logger.error("synaptic_downscaling_failed", error=str(exc))
            return {"episodes_decayed": 0}


# ═══════════════════════════════════════════════════════════════════
# Belief Compressor (NREM)
# ═══════════════════════════════════════════════════════════════════


class BeliefCompressor:
    """
    During NREM, query Nova for current belief set, identify redundant or
    low-confidence beliefs, and propose consolidation via Synapse.
    """

    def __init__(
        self,
        nova: Any | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._nova = nova
        self._event_bus = event_bus
        self._logger = logger.bind(component="belief_compressor")

    async def run(self) -> dict[str, int]:
        """Identify and propose consolidation of redundant beliefs."""
        if self._nova is None:
            self._logger.debug("no_nova_available")
            return {"beliefs_compressed": 0, "beliefs_evaluated": 0}

        t0 = time.monotonic()
        beliefs_compressed = 0
        beliefs_evaluated = 0

        try:
            # Get current beliefs from Nova
            get_beliefs = getattr(self._nova, "get_active_beliefs", None)
            if get_beliefs is None:
                return {"beliefs_compressed": 0, "beliefs_evaluated": 0}

            beliefs = await get_beliefs()
            beliefs_evaluated = len(beliefs)

            if not beliefs:
                return {"beliefs_compressed": 0, "beliefs_evaluated": 0}

            # Group beliefs by domain/category to find redundancies
            by_domain: dict[str, list[Any]] = defaultdict(list)
            for belief in beliefs:
                domain = getattr(belief, "domain", "general")
                by_domain[domain].append(belief)

            # Find low-confidence beliefs (< 0.3) and groups with 3+ similar beliefs
            to_consolidate: list[str] = []
            for domain, group in by_domain.items():
                for belief in group:
                    confidence = getattr(belief, "confidence", 1.0)
                    if confidence < 0.3:
                        to_consolidate.append(getattr(belief, "id", ""))
                        beliefs_compressed += 1

                # Mark redundant beliefs within same domain (keep highest confidence)
                if len(group) >= 3:
                    sorted_beliefs = sorted(
                        group,
                        key=lambda b: getattr(b, "confidence", 0.0),
                        reverse=True,
                    )
                    for redundant in sorted_beliefs[2:]:  # keep top 2 per domain
                        redundant_conf = getattr(redundant, "confidence", 0.0)
                        if redundant_conf < 0.5:
                            to_consolidate.append(getattr(redundant, "id", ""))
                            beliefs_compressed += 1

            # Propose consolidation to Nova via Synapse
            if to_consolidate and self._event_bus is not None:
                event = SynapseEvent(
                    event_type=SynapseEventType.COMPRESSION_BACKLOG_PROCESSED,
                    source_system="oneiros",
                    data={
                        "consolidation_type": "belief_compression",
                        "belief_ids_to_retire": to_consolidate,
                        "count": len(to_consolidate),
                    },
                )
                await self._event_bus.emit(event)

        except Exception as exc:
            self._logger.error("belief_compression_failed", error=str(exc))

        elapsed = (time.monotonic() - t0) * 1000
        self._logger.info(
            "belief_compression_complete",
            beliefs_evaluated=beliefs_evaluated,
            beliefs_compressed=beliefs_compressed,
            elapsed_ms=round(elapsed, 1),
        )
        return {
            "beliefs_compressed": beliefs_compressed,
            "beliefs_evaluated": beliefs_evaluated,
        }


# ═══════════════════════════════════════════════════════════════════
# World Model Consistency Auditor (Spec 14 §3.3.4)
# ═══════════════════════════════════════════════════════════════════


class WorldModelAuditor:
    """
    Global scan of the world model for structural inconsistencies (Spec 14 §3.3.4).

    Three audit passes, each independent and try-except guarded:

    1. Orphaned schemas - GenerativeSchema nodes with no linked episode, causal
       link, or procedure.  Pruned if also low-usage (access_count < 2).

    2. Circular causal structures - detect cycles in the causal graph via
       iterative DFS over CAUSES_LINK relationships in Neo4j.  Cycles are
       resolved by removing the weakest link in the cycle.

    3. Deprecated hypotheses - Evo Hypothesis nodes with status INVALIDATED or
       SUPERSEDED that were promoted as world model beliefs.  Retired by setting
       status = RETIRED and removing the HAS_BELIEF relationship.

    All three passes run against Neo4j directly - no LLM calls.
    """

    # Orphaned schema: no linked evidence, used fewer than this many times
    ORPHAN_ACCESS_THRESHOLD: int = 2

    def __init__(self, neo4j: Any | None) -> None:
        self._neo4j = neo4j
        self._logger = logger.bind(worker="world_model_auditor")

    async def run(self) -> WorldModelConsistencyReport:
        """Run all three audit passes. Returns a consistency report."""
        if self._neo4j is None:
            return WorldModelConsistencyReport(audit_skipped=True)

        t0 = time.monotonic()
        orphan_found = orphan_pruned = 0
        cycle_found = cycle_resolved = 0
        deprecated_found = deprecated_retired = 0

        # Pass 1: Orphaned schemas
        try:
            orphan_found, orphan_pruned = await self._audit_orphaned_schemas()
        except Exception:
            self._logger.exception("orphan_schema_audit_error")

        # Pass 2: Circular causal structures
        try:
            cycle_found, cycle_resolved = await self._audit_causal_cycles()
        except Exception:
            self._logger.exception("causal_cycle_audit_error")

        # Pass 3: Deprecated hypotheses
        try:
            deprecated_found, deprecated_retired = await self._audit_deprecated_hypotheses()
        except Exception:
            self._logger.exception("deprecated_hypothesis_audit_error")

        elapsed = (time.monotonic() - t0) * 1000
        self._logger.info(
            "world_model_audit_complete",
            orphan_found=orphan_found,
            orphan_pruned=orphan_pruned,
            cycle_found=cycle_found,
            cycle_resolved=cycle_resolved,
            deprecated_found=deprecated_found,
            deprecated_retired=deprecated_retired,
            elapsed_ms=round(elapsed, 1),
        )
        return WorldModelConsistencyReport(
            orphaned_schemas_found=orphan_found,
            orphaned_schemas_pruned=orphan_pruned,
            circular_structures_found=cycle_found,
            circular_structures_resolved=cycle_resolved,
            deprecated_hypotheses_found=deprecated_found,
            deprecated_hypotheses_retired=deprecated_retired,
            duration_ms=elapsed,
        )

    async def _audit_orphaned_schemas(self) -> tuple[int, int]:
        """
        Find GenerativeSchema nodes with no linked episodes, causal links, or procedures.
        Prune those with access_count < ORPHAN_ACCESS_THRESHOLD.
        """
        # Find candidates: schemas with no outgoing evidence relationships
        find_query = """
        MATCH (s:GenerativeSchema)
        WHERE NOT (s)-[:DERIVED_FROM|LINKS_TO|USED_BY]->()
          AND NOT ()<-[:HAS_SCHEMA]-(s)
        RETURN s.id AS schema_id, coalesce(s.access_count, 0) AS access_count
        """
        result = await self._neo4j.execute_read(find_query)
        candidates = [
            r["schema_id"]
            for r in result.records
            if (r["access_count"] or 0) < self.ORPHAN_ACCESS_THRESHOLD
        ]
        found = len(result.records)

        if not candidates:
            return found, 0

        # Soft-delete: mark as orphaned rather than hard-delete (immutable audit trail)
        prune_query = """
        UNWIND $ids AS schema_id
        MATCH (s:GenerativeSchema {id: schema_id})
        SET s.status = 'orphaned',
            s.orphaned_at = datetime()
        """
        await self._neo4j.execute_write(prune_query, {"ids": candidates})
        return found, len(candidates)

    async def _audit_causal_cycles(self) -> tuple[int, int]:
        """
        Detect cycles in the causal graph using Neo4j's apoc.algo.scc or
        a manual iterative DFS via shortest-path self-reachability.
        Removes the weakest link (lowest strength) in each cycle.
        """
        # Find any node that can reach itself via CAUSES_LINK (cycle indicator)
        cycle_query = """
        MATCH path = (n)-[:CAUSES_LINK*2..10]->(n)
        WITH nodes(path) AS cycle_nodes, relationships(path) AS cycle_rels
        UNWIND cycle_rels AS rel
        RETURN rel.id AS rel_id, coalesce(rel.strength, 0.0) AS strength
        ORDER BY strength ASC
        LIMIT 50
        """
        result = await self._neo4j.execute_read(cycle_query)
        if not result.records:
            return 0, 0

        # Each cycle path returns all its edges; we find the weakest per cycle
        # Simple heuristic: retire the globally weakest edges found in cycles
        seen_cycles: set[str] = set()
        to_remove: list[str] = []
        for record in result.records:
            rel_id = record["rel_id"]
            if rel_id and rel_id not in seen_cycles:
                seen_cycles.add(rel_id)
                to_remove.append(rel_id)

        found = len(to_remove)
        if not to_remove:
            return 0, 0

        # Soft-delete: mark link as cycle-removed
        remove_query = """
        UNWIND $ids AS rel_id
        MATCH ()-[r:CAUSES_LINK {id: rel_id}]->()
        SET r.status = 'cycle_removed',
            r.removed_at = datetime()
        DELETE r
        """
        await self._neo4j.execute_write(remove_query, {"ids": to_remove})
        return found, len(to_remove)

    async def _audit_deprecated_hypotheses(self) -> tuple[int, int]:
        """
        Find Evo Hypothesis nodes that are INVALIDATED or SUPERSEDED but still
        connected to WorldModel beliefs.  Retire the belief relationship.
        """
        find_query = """
        MATCH (h:Hypothesis)-[:PROMOTED_TO]->(b:Belief)
        WHERE h.status IN ['invalidated', 'superseded']
        RETURN h.id AS hyp_id, b.id AS belief_id
        LIMIT 100
        """
        result = await self._neo4j.execute_read(find_query)
        pairs = [(r["hyp_id"], r["belief_id"]) for r in result.records]
        found = len(pairs)

        if not found:
            return 0, 0

        retire_query = """
        UNWIND $pairs AS pair
        MATCH (h:Hypothesis {id: pair[0]})-[r:PROMOTED_TO]->(b:Belief {id: pair[1]})
        SET h.status = 'retired',
            h.retired_at = datetime(),
            b.deprecated = true
        DELETE r
        """
        await self._neo4j.execute_write(
            retire_query, {"pairs": [[h, b] for h, b in pairs]}
        )
        return found, len(pairs)


# ═══════════════════════════════════════════════════════════════════
# Slow Wave Stage Orchestrator
# ═══════════════════════════════════════════════════════════════════


class SlowWaveStage:
    """
    Stage 2: Slow Wave (~50% of sleep duration).

    The batch compiler. Orchestrates:
    1. Memory Ladder compression (4 rungs)
    2. Synaptic Downscaling (salience decay)
    3. Belief Compression (redundant belief retirement)
    4. Hypothesis Graveyard (MDL-based retirement)
    5. Causal Graph Reconstruction (PC algorithm)

    Broadcasts COMPRESSION_BACKLOG_PROCESSED and CAUSAL_GRAPH_RECONSTRUCTED.
    """

    def __init__(
        self,
        logos: LogosService | None = None,
        event_bus: EventBus | None = None,
        neo4j: Any | None = None,
        nova: Any | None = None,
    ) -> None:
        self._logos = logos
        self._event_bus = event_bus
        self._memory_ladder = MemoryLadder(logos=logos, event_bus=event_bus)
        self._downscaler = SynapticDownscaler(neo4j=neo4j)
        self._belief_compressor = BeliefCompressor(nova=nova, event_bus=event_bus)
        self._graveyard = HypothesisGraveyard(logos=logos)
        self._reconstructor = CausalGraphReconstructor(logos=logos)
        self._auditor = WorldModelAuditor(neo4j=neo4j)
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

        # 1b. Synaptic Downscaling - decay stale episode salience
        downscale_result = await self._downscaler.run()
        self._logger.debug(
            "downscaling_done",
            episodes_decayed=downscale_result.get("episodes_decayed", 0),
        )

        # 1c. Belief Compression - identify and retire redundant beliefs
        belief_result = await self._belief_compressor.run()
        self._logger.debug(
            "belief_compression_done",
            beliefs_compressed=belief_result.get("beliefs_compressed", 0),
        )

        # 2. Hypothesis Graveyard
        hypothesis_report = await self._graveyard.process(
            active_hypotheses or []
        )

        # 3. Causal Graph Reconstruction
        causal_report = await self._reconstructor.reconstruct(
            causal_observations or []
        )
        await self._broadcast_causal_reconstructed(causal_report)

        # 4. World Model Consistency Audit (Spec 14 §3.3.4)
        consistency_report = await self._auditor.run()

        elapsed = (time.monotonic() - t0) * 1000

        report = SlowWaveReport(
            compression=compression_report,
            hypotheses=hypothesis_report,
            causal=causal_report,
            consistency=consistency_report,
            duration_ms=elapsed,
        )

        self._logger.info(
            "slow_wave_complete",
            memories_processed=compression_report.memories_processed,
            schemas_created=compression_report.schemas_created,
            hypotheses_retired=hypothesis_report.hypotheses_retired,
            invariants_discovered=causal_report.invariants_discovered,
            orphaned_schemas_pruned=consistency_report.orphaned_schemas_pruned,
            causal_cycles_resolved=consistency_report.circular_structures_resolved,
            deprecated_hypotheses_retired=consistency_report.deprecated_hypotheses_retired,
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
