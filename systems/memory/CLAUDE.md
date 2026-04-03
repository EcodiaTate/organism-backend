# Memory System - CLAUDE.md

**Spec**: `.claude/EcodiaOS_Spec_01_Memory.md` (v1.2) - read before editing.
**Last audit**: 2026-03-08 (autonomy audit - 6 bugs fixed, SLEEP_INITIATED wired, graph health KPI added)
**Role**: Neo4j knowledge graph substrate. Every system reads/writes here. If you lose the graph, you lose the self.

---

## Architecture

**Graph layers**: Identity (`Self`, `Constitution`) · Episodic (`Episode`, `Intent`, `Outcome`) · Semantic (`Entity`, `SemanticRelation`) · Communities (`Community`, `ProjectionBasis`) · Evolution (`Hypothesis`, `Experiment`, `ExperimentResult`) · Causal (`CausalNode`)

**Key relationships** (code reality - spec §2.2 now matches):
- `MENTIONS {role, confidence, span}` - Episode → Entity
- `RELATES_TO {type, strength, confidence}` - Entity ↔ Entity
- `FOLLOWED_BY {gap_seconds, causal_strength}` - Episode → Episode (≤1h, gap based on `event_time` delta - P4 fixed 2026-03-07); failure → `INCIDENT_DETECTED` (Thymos)
- `BELONGS_TO` - Entity → Community
- `COMPRESSED_VIA` - Community → ProjectionBasis (no `CompressedEpisode` node; compression is in-place on Episode)
- `GENERATED` → `RESULTED_IN` - Episode → Intent → Outcome
- `ALTERNATIVE_TO` - Counterfactual → Episode
- `SUPPORTED_BY` - Belief → Episode
- `DERIVED_FROM` - Belief → Hypothesis
- `CORE_CONCEPT` - Self → Entity

**Public API** (`MemoryService`):
- `store_percept()` - main ingestion; stamps somatic marker, links temporal chain (`event_time`-based gap), emits `EPISODE_STORED`; chain failure → `INCIDENT_DETECTED` (Thymos) **[P9b resolved]**
- `store_expression_episode()` - **AV3 resolved** - Voxis expression storage via service layer (somatic stamp + FOLLOWED_BY + EPISODE_STORED)
- `store_counterfactual_episode()` / `resolve_counterfactual()` / `link_counterfactual_to_outcome()` - **AV4 resolved** - Nova counterfactual path via service layer
- `get_recent_episodes()` / `get_episode()` / `get_entity()` - **AV6 resolved** - read queries for API router via service layer
- `store_intent()` / `store_outcome()` - RE training chain
- `retrieve()` - 4-leg parallel hybrid: vector (0.35) + bm25 (0.20) + graph spreading-activation (0.20) + salience (0.25)
- `consolidate()` - salience decay → Louvain community detection (+ centroid/radius/level) → PCA/SVD compression → entity dedup → belief half-life → belief promotion → re-embedding **[all gaps resolved]**
- `reembed_pending_nodes(batch_size)` - **P9 resolved** - re-embeds episodes/entities with `needs_reembedding=true`; called from `consolidate()`
- `update_personality_from_evo(hypothesis_outcome)` - **SG7 resolved** - updates `Self.personality_vector` from Evo/Equor drive deltas; records to `drive_weight_history`
- `export_training_batch(stream_id, since, limit)` - RE training data extraction (all 5 streams)
- `export_genome()` - **SG1 resolved** - genome export for Mitosis child spawning (delegates to `MemoryGenomeExtractor` v2: includes drive_weight_history + floor_threshold_snapshots)
- `update_affect()` - writes 6 affect scalars to Self node; emits `SELF_AFFECT_UPDATED`
- `update_conscience_fields(last_conscience_activation, compliance_score)` - - writes `last_conscience_activation` (datetime) and `avg_compliance_score` (EMA α=0.05 rolling 24h mean) to Self node; called by Equor after every constitutional review
- `set_soma()` / `set_event_bus()` - dependency injection points

**Do not call internal modules directly.** All writes must go through `MemoryService`. Cross-system callers that bypass the service layer are architecture violations (see AV3–AV6 below).

---

## Synapse Events

**Emitted**: `EPISODE_STORED` · `BELIEF_CONSOLIDATED` (with real `beliefs_created` count from promotion) · `SELF_AFFECT_UPDATED` · `MEMORY_PRESSURE` (episode_count > 10k or unconsolidated lag > 500; lag is now a real count of consolidation_level=0 episodes) · `SELF_STATE_DRIFTED` (contradictions > 5 during consolidation, or compliance drop > 0.1 from EQUOR_CONSTITUTIONAL_SNAPSHOT) · `MEMORY_EPISODES_DECAYED` (degradation soft-deletes) · `EVOLUTIONARY_OBSERVABLE` (entity_discovered, relation_formed, consolidation_pattern, community_emerged, genome_exported, **memory_graph_utilization** - graph health KPI for Benchmarks) · `INCIDENT_DETECTED` (temporal chain integrity failure → Thymos) · `SYSTEM_MODULATION_ACK`

**Self node conscience fields**:
- `last_conscience_activation` - datetime of last Equor review
- `avg_compliance_score` - EMA (α=0.05) of `composite_alignment` across all reviews; proxy for constitutional health over time

**Consumed**:
- `SLEEP_INITIATED` → `_on_sleep_initiated()`: **NEW 2026-03-08** - auto-triggers `consolidate()` at every sleep cycle; closes spec §18 gap M5
- `MEMORY_DEGRADATION` → `_on_memory_degradation()`: batch-decays salience on unconsolidated episodes older than `affected_episode_age_hours` by factor `(1 - fidelity_loss_rate)`; soft-deletes (sets `decayed=true`) episodes where salience falls below 0.01; emits `MEMORY_EPISODES_DECAYED` if any deleted. **This is a real graph mutation - not advisory.**
- `WALLET_TRANSFER_CONFIRMED` · `REVENUE_INJECTED` → `FinancialEncoder` creates salience=1.0 episodes directly
- `EQUOR_CONSTITUTIONAL_SNAPSHOT` → `_on_equor_constitutional_snapshot()`: writes `(:ConstitutionalSnapshot)-[:SNAPSHOT_OF]->(:Self)` to Neo4j; emits `SELF_STATE_DRIFTED` if compliance drops >0.1 from previous snapshot
- `METABOLIC_PRESSURE` → `_on_metabolic_pressure()`: updates `_starvation_level`
- `SYSTEM_MODULATION` → `_on_system_modulation()`: applies Skia austerity directives; emits ACK

**Constitutional Snapshot tracking**: `_last_compliance_score: float | None` on `MemoryService` tracks the previous snapshot's compliance for drift detection. Persists `(:ConstitutionalSnapshot)` nodes so Thread can walk constitutional evolution history.

**Graph health KPI**: `emit_graph_health_kpi()` emits `EVOLUTIONARY_OBSERVABLE` with `observable_type="memory_graph_utilization"` at the end of every consolidation cycle. Benchmarks subscribes to `EVOLUTIONARY_OBSERVABLE` and can track episode/entity/node counts as population fitness metrics.

---

## Key Implementation Details

- **Salience decay**: exponential λ = ln(2)/72h; floor 0.01; core identity floor 0.5
- **Novelty score**: `1.0 - max_cosine_sim` to recent episodes, computed in `store_percept()` via vector index
- **Somatic reranking**: applied in `retrieve()` when Soma is wired - state-congruent recall
- **FinancialEncoder**: separate sequence state; must sync with MemoryService via `set_sequence_state/get_sequence_state`
- **Belief half-life enforcement**: `BeliefAgingScanner` is in `systems/evo/belief_halflife.py`, not Memory - expiry is Evo's responsibility
- **Genome import**: `seed_from_parent_genome()` (GenomeSeeder) + `seed_simula_from_parent_genome()` in `birth.py` - deferred cross-system imports (violation, but tolerated until refactored)
- **Community detection**: Louvain (pure Cypher, not Leiden). Community nodes use `id` property (not `community_id`).

---

## Open Gaps (as of 2026-03-08 v1.3)

### Critical (remaining)
| # | Gap | Location |
|---|-----|----------|
| AV5 | Simula writes entities directly via `semantic.create_entity` | `simula/proposals/paper_memory.py:24` - no MemoryService wrapper added yet |
| AV2 | `birth.py` imports from Simula + Evo (deferred, inside function body) | Genome seeding should be orchestrated at startup |

### Resolved (2026-03-08 audit)
| # | Resolution |
|---|------------|
| **BUG-1** | `_on_memory_degradation()` Cypher `{hours: }` missing `$age_hours` param - now `{hours: $age_hours}`; also fixed `ep.salience` → `ep.salience_composite` |
| **BUG-2** | `semantic_compression.py` queried `Community {community_id: $cid}` - Community nodes use `id`, not `community_id`; three query sites fixed |
| **BUG-3** | `genome.py` extracted/seeded relations via `[:SEMANTIC_RELATION]` - schema uses `[:RELATES_TO]`; extraction now matches `RELATES_TO` with correct property names (`type`, `strength`); seeding uses `MERGE … [:RELATES_TO]` |
| **BUG-4** | `_check_memory_pressure()` used `consolidation_lag = ep_count` (always wrong) - now counts actual episodes with `consolidation_level=0` |
| **BUG-5** | `BELIEF_CONSOLIDATED` event hardcoded `beliefs_created: 0` - now reads `steps.belief_promotion.promoted` from consolidation result |
| **GAP-M5** | `SLEEP_INITIATED` subscription added - `_on_sleep_initiated()` auto-triggers `consolidate()` at every Oneiros sleep cycle |
| **GAP-KPI** | `emit_graph_health_kpi()` added - emits `EVOLUTIONARY_OBSERVABLE` (`memory_graph_utilization`) at end of every consolidation; Benchmarks can now observe memory load |

### Resolved
| # | Resolution |
|---|------------|
| AV3 | `store_expression_episode()` added to MemoryService |
| AV4 | `store_counterfactual_episode()`, `resolve_counterfactual()`, `link_counterfactual_to_outcome()` added |
| AV6 | Read queries routed through MemoryService in API router |
| SG1 | `export_genome()` added - `MemoryGenomeExtractor` v2 with drive_weight_history + floor_threshold_snapshots |
| SG7 | `update_personality_from_evo()` - personality vector updated from Evo/Equor drive deltas |
| P4 | `FOLLOWED_BY` uses `event_time` delta |
| P5 | `consolidate_high_confidence_beliefs()` - `:Belief` → `:ConsolidatedBelief` at precision ≥ 0.85 |
| P7 | Community centroid, radius, level=1 during `_materialize_community_nodes()` |
| P9 | `reembed_pending_nodes()` - batch 50/cycle |
| M17 | Temporal chain failure → `INCIDENT_DETECTED` (Thymos) |

### Medium (remaining)
| # | Gap | Notes |
|---|-----|-------|
| AV6-partial | `get_entity_neighbours()` still direct in API router | One call site, tracked for cleanup |
| SG4 | No fitness signal to Oikos/Mitosis | Memory retrieval speed unmeasured |
| THETA | No `COGNITIVE_CYCLE_COMPLETE` heartbeat subscription | Consolidation still needs explicit trigger beyond SLEEP_INITIATED |

### Dead Code
- `project_query_embedding()`, `decompress_embedding()`, `get_all_basis_ids()` - `semantic_compression.py` (no callers found)

---

## Schema Constraints (key)

```
UNIQUE: Episode.id, Entity.id, Community.id, Self.instance_id
UNIQUE: Intent.id, Outcome.id, CausalNode.name, Experiment.id
Vector indexes: episode_embedding (768D), entity_embedding (768D), community_embedding (768D), episode_somatic (19D)
```

Community nodes must use `id` property (not `community_id`) - constraint enforced.

---

## Performance SLAs

| Operation | Target |
|-----------|--------|
| `store_percept()` | ≤50ms |
| `retrieve()` | ≤200ms |
| `resolve_and_create_entity()` | ≤100ms |
| `consolidate()` (full) | ≤10s (async, sleep cycles) |

---

## Speciation Role

Memory is the genome of each instance and the substrate of organismal identity. The Self node is the singular reference point for affect, personality, cycle count, and constitution. Soft-delete is an architectural invariant - hard delete is prohibited. The consolidation pipeline is compression for retrieval efficiency, not biological aging.
