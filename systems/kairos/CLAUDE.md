# Kairos - Causal Invariant Mining (Spec 22)

## What It Does

Kairos mines the hierarchy of causal knowledge: correlations → causal rules → context-invariant rules → substrate-independent invariants. Each level up is exponentially more compressed and more generative. A single Tier 3 invariant generates predictions across every domain it touches.

**Self-causality (new - 8 Mar 2026)**: Kairos also runs a lightweight internal causal tracker alongside the external pipeline. It samples 6 organism-internal variables each pipeline run, mines pairwise causal relationships within them, and emits `KAIROS_INTERNAL_INVARIANT` when a self-causal law is discovered (e.g. "prediction_error_rate increases coherence_decrease [lag: 1 pipeline run]"). Nova + Thread subscribe. This is how the organism discovers causal laws about *itself*, not just about the external world.

## Architecture

### 7-Stage Pipeline

| Stage | Module | What It Does |
|-------|--------|-------------|
| 1 | `correlation_miner.py` | O(n²) cross-context correlation mining. Filter: mean |r| > 0.3, variance < 0.1 |
| 2 | `causal_direction.py` | Three tests: temporal precedence, intervention asymmetry (Axon logs), additive noise model |
| 3 | `confounder.py` | PC algorithm for confounder detection with partial correlation |
| 4 | `mechanism_extractor.py` | BFS over CausalNode graph in Memory; populates `CausalRule.mechanism` before Stage 5 |
| 5 | `context_invariance.py` | Tests rules across contexts. Computes hold_rate, identifies scope conditions |
| 6 | `invariant_distiller.py` | Phase C: variable abstraction, tautology test, minimality test, domain mapping |
| 7 | `counter_invariant.py` | Phase D: violation scanning, clustering, scope refinement |

### Supporting Modules

| Module | What It Does |
|--------|-------------|
| `hierarchy.py` | 3-tier invariant hierarchy (Domain → Cross-Domain → Substrate-Independent) |
| `intelligence_ledger.py` | Per-invariant I-ratio accounting, historical tracking, trend analysis, counterfactual estimation |
| `pipeline.py` | Main orchestrator - event subscriptions, pipeline execution, all feedback loop emissions |
| `persistence.py` | Neo4j persistence - batched UNWIND writes, startup restoration, schema/index management |
| `types.py` | Event payloads, config. Re-exports `CausalInvariant` etc. from `primitives.causal` |

### Shared Primitives

`CausalInvariant`, `CausalInvariantTier`, `ApplicableDomain`, `ScopeCondition` live in `primitives/causal.py` - the canonical location. `kairos/types.py` re-exports them for backward compatibility.

Key fields added to `CausalInvariant`:
- `direction: Literal["positive", "negative", ""]` - populated from correlation sign in Stage 2
- `validated: bool` - marks externally validated invariants
- `recency_weight: float` - decays 0.95× per pipeline cycle; demote < 0.3, archive < 0.1
- `active: bool` - False for archived invariants (excluded from Neo4j restore)

### 3-Tier Hierarchy

- **Tier 1 (Domain)**: Holds within a single domain. Default tier for new invariants.
- **Tier 2 (Cross-Domain)**: Holds across 2+ distinct domains. High transfer value.
- **Tier 3 (Substrate-Independent)**: 4+ domains, 3+ substrates, distilled + minimal + not tautological + hold_rate >= 0.95. Finding one is an architectural event.

## Neo4j Persistence (`persistence.py`)

- `ensure_schema()` - idempotent uniqueness constraint on `CausalInvariant.id`, indexes on `tier`, `active`, `invariance_hold_rate`
- `persist_invariants_batch(neo4j, invariants)` - batched UNWIND MERGE for invariant nodes + `CausalNode`/`CAUSES` relationships
- `restore_invariants(neo4j)` - loads all `active=true` invariants on startup, ordered by tier DESC
- Called from `pipeline.initialize()` (restore) and end of `run_pipeline()` (persist)

## Integration Surface

### Events Emitted (18)

| Event | Target | Purpose |
|-------|--------|---------|
| `KAIROS_CAUSAL_CANDIDATE_GENERATED` | Organism | Stage 1 output |
| `KAIROS_CAUSAL_DIRECTION_ACCEPTED` | Organism | Stage 2 confirmed direction |
| `KAIROS_CONFOUNDER_DISCOVERED` | Organism | Stage 3 spurious edge found |
| `KAIROS_INVARIANT_CANDIDATE` | Organism | Stage 5 strong/conditional invariant |
| `KAIROS_INVARIANT_DISTILLED` | Nova, Evo, **Thread** | Phase C distillation complete; Thread caches for causal attribution |
| `KAIROS_TIER3_INVARIANT_DISCOVERED` | Nexus, Oneiros, Logos | Highest-priority event |
| `KAIROS_COUNTER_INVARIANT_FOUND` | Organism | Phase D scope refinement |
| `KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE` | Telos | I-ratio jump detected |
| `KAIROS_VALIDATED_CAUSAL_STRUCTURE` | Evo | Thompson sampler reward signal |
| `KAIROS_SPURIOUS_HYPOTHESIS_CLASS` | Evo | Thompson sampler penalty signal |
| `KAIROS_INVARIANT_ABSORPTION_REQUESTED` | Fovea | World model integration request |
| `KAIROS_CAUSAL_NOVELTY_DETECTED` | Organism | Novel causal structure found |
| `KAIROS_HEALTH_DEGRADED` | Thymos | System health incident |
| `KAIROS_VIOLATION_ESCALATION` | Thymos | Repeated invariant violations |
| `CONSTITUTIONAL_REVIEW_REQUESTED` | Equor | Tier 3 constitutional review gate |
| `NARRATIVE_MILESTONE` | Thread | Tier 3 discovery as narrative event |
| `WORLD_MODEL_UPDATED` | Nova | Tier 3 policy-relevant world model update |
| `KAIROS_INTERNAL_INVARIANT` | Thread, Nova | **NEW (8 Mar 2026)** Self-causal law discovered within organism's own dynamics |

### Events Subscribed (16)

| Event | Source | Purpose |
|-------|--------|---------|
| `FOVEA_PREDICTION_ERROR` | Fovea | Primary input: causal surprises become pre-seeded candidates |
| `EVOLUTION_CANDIDATE` | Evo | Causal hypotheses feed into Stage 1 |
| `EPISODE_STORED` | Memory | New observations with causal annotations |
| `CROSS_DOMAIN_MATCH_FOUND` | Oneiros | REM cross-domain structural matches |
| `WORLD_MODEL_UPDATED` | Logos | Bidirectional: re-evaluate invariants on model changes |
| `COMPRESSION_BACKLOG_PROCESSED` | Oneiros | Consolidation feedback: cross-domain patterns |
| `FEDERATION_INVARIANT_RECEIVED` | Federation/Nexus | Inbound invariant from peer - validated locally before acceptance |
| `OIKOS_ECONOMIC_EPISODE` | Oikos | Economic causal data stream for EconomicCausalMiner |
| `FOVEA_INTERNAL_PREDICTION_ERROR` | Fovea/SACM | Economic domain prediction errors as high-salience causal signals |
| `PHANTOM_PRICE_OBSERVATION` | Phantom Liquidity | Market price time-series for economic correlation mining |
| `GROUND_TRUTH_CANDIDATE` | Nexus | Level 3+ epistemic fragments seeded as causal candidates |
| `EMPIRICAL_INVARIANT_CONFIRMED` | Nexus | Level 4 confirmation reinforces matching invariants, prevents decay |
| `PERCEPT_ARRIVED` | Perception Gateway | Raw percept stream: high-salience percepts become causal candidates |
| `EQUOR_DRIVE_WEIGHTS_UPDATED` | Equor | Constitutional drive shifts mined for causal structure |
| `SOMATIC_DRIVE_VECTOR` | Soma | Organism drive state buffered as cross-context observations |

### Direct Wiring (set_* methods)

- `set_event_bus(bus)` - Synapse pub/sub (called at Phase 6)
- `set_logos(logos_ingest)` - Logos KairosInvariantProtocol for world model ingestion (called at Phase 6)
- `set_memory(memory)` - Memory system for `query_observations_for_testing()` and Stage 4 CausalNode BFS (called post-Phase 9)
- `set_neo4j(neo4j)` - Neo4j client for persistence; triggers `initialize()` for startup restoration (called post-Phase 9)
- `set_nexus(nexus_share)` - Nexus fragment sharing for Tier 3 federation (called post-Phase 9)
- `set_oneiros(oneiros)` - Oneiros REM seed injection on Tier 3 discoveries (called at Phase 6 via `oneiros.set_kairos(kairos)`)

## Feedback Loops

1. **Fovea → Kairos → Fovea**: Prediction errors seed mining; confirmed invariants request absorption into world model
2. **Evo → Kairos → Evo**: Causal hypotheses tested (parsed via `_parse_causal_statement`); validated structures reward Thompson sampler; confounded structures penalize it
3. **Oneiros ↔ Kairos**: Cross-domain matches seed mining; Tier 3 discoveries become priority REM seeds
4. **Logos ↔ Kairos**: Kairos ingests invariants to Logos; Tier 3 also triggers deep structural reorganization; Logos model updates trigger re-evaluation
5. **Kairos → Thymos**: Health degradation and violation escalation trigger immune response
6. **Kairos → Telos**: I-ratio step changes (cross-run + counterfactual) signal intelligence geometry shifts
7. **Kairos → Equor/Thread/Nova**: Tier 3 cascade - constitutional review, narrative milestone, policy update
8. **Federation → Kairos**: Inbound invariants validated against local counter-invariants before acceptance
9. **Kairos → RE**: Validated causal chains emit `RE_TRAINING_EXAMPLE` (Stream 4) for reasoning engine training

## Invariant Decay

Every `run_pipeline()` cycle applies decay to all active invariants:
- `recency_weight *= 0.95` per cycle
- Invariants with `recency_weight < 0.3` are demoted one tier
- Invariants with `recency_weight < 0.1` are archived (`active = False`)
- `recency_weight` resets to 1.0 when an invariant is re-confirmed by new observations

## Evolutionary Metrics

Emitted as `EvolutionaryObservable` at the end of each pipeline run:
- **Mean I-ratio**: Average intelligence ratio contribution across active invariants
- **Tier 3 discovery rate**: `tier3_discoveries / pipeline_runs`
- **Invariant overlap coefficient**: Jaccard similarity of variables across active invariants (measures compression)

## Tier 3 Promotion Cascade

When `CausalHierarchy` promotes an invariant to Tier 3, a sync callback queues the async cascade:
1. Broadcast `KAIROS_TIER3_INVARIANT_DISCOVERED` on Synapse
2. Share with Nexus for federation broadcasting
3. Inject Oneiros REM seed with `priority: True` (M8) - prompts dream cycle cross-domain search
4. Request Equor constitutional review (`CONSTITUTIONAL_REVIEW_REQUESTED`)
5. Emit Thread narrative milestone (`NARRATIVE_MILESTONE`)
6. Notify Nova of policy-relevant world model update (`WORLD_MODEL_UPDATED`)
7. Signal Logos for deep structural reorganization with `reorganize: True` (M7)

Async reliability: sync callback uses `loop.create_task()` with error-logging done callback. If no loop is running, invariants are queued in `_deferred_tier3` and drained at the start of the next `run_pipeline()` call.

## Health Monitoring

Diagnosed every pipeline run. Checks:
- **Discovery stall**: Not enough invariants per candidate (< 0.1 rate)
- **Confounder inflation**: Too many correlations turn out confounded (> 0.5 rate)
- **Causal surprise / corruption**: Violation rate exceeds threshold (> 0.4)
- **Ledger drift**: Intelligence contributions shifting rapidly (> 0.3)

Emits `KAIROS_HEALTH_DEGRADED` to Thymos on degraded/critical status.

## Intelligence Ledger

Per-invariant accounting with:
- `observations_covered`, `description_savings`, `intelligence_ratio_contribution`
- **Historical tracking**: Point-in-time snapshots per invariant (max 50)
- **Trend analysis**: Linear regression on ratio contribution over time (increasing/decreasing/stable)
- **Counterfactual**: `estimate_counterfactual_i_without(id)` - what I-ratio would be without this invariant
- **Drift detection**: `get_ledger_drift()` - average contribution change between consecutive computations

## Implemented (2026-03-07 gap closure)

- **P5**: `_extract_observation_pairs` now pools pairs across ALL contexts (not first-context-wins)
- **P6**: Step-change detection uses `_prev_i_ratios` to compare across pipeline runs, not within-run counterfactuals
- **AV4**: `_hierarchy._find()` replaced with public `find_invariant()` throughout pipeline
- **D3**: `tier3_min_contexts` renamed to `tier3_min_observations` - semantics now match usage
- **D4**: Counterfactual I-ratio values with non-trivial weight emit a separate `KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE` with `cause="counterfactual_removal"`
- **D5**: `_discovered_patterns` capped at 200 entries; count exposed in `health()`
- **M2**: Multi-level abstraction loop in `InvariantDistiller.distill()` - up to 5 raising iterations, stops at tautology boundary. Added `_raise_abstraction_level()` with 3-level ladder (domain-role → process-role → causal-role)
- **M7**: `_logos_tier3_structural_reorganize()` added to Tier 3 cascade - sends `reorganize: True` payload to Logos for deep world model restructuring
- **M8**: Oneiros REM seed now carries `priority: True` for Tier 3 injections
- **M10**: Phase D sorted by ledger `rank_by_value()` - highest-value invariants scanned first for violations
- **SG4**: `_emit_re_training_example()` fires after each validated causal chain (Stream 4, RE training)
- **P2** (variable abstraction): `_abstract_variables()` now also parses scope condition tokens and domain substrate identifiers

## Implemented (2026-03-07 interface gap closure)

- **Stage 4 IMPLEMENTED**: `mechanism_extractor.py` - `MechanismExtractor` BFS-traverses the `CausalNode` graph in Memory (up to 4 hops, max 8 branch per node). Populates `CausalRule.mechanism` before Stage 5. Falls back to direct-influence placeholder when Memory unavailable or no path found. Wired into `pipeline.py` via `_mechanism_extractor` field; receives Memory reference through `set_memory()`.
- **`_loses_predictive_power()` IMPLEMENTED**: `invariant_distiller.py` - checks whether raising abstraction level collapses the cause/effect variable roles into the same abstract role (directional information lost) or reduces unique roles below 2 (prediction becomes trivial). Used as a second stop condition in the M2 abstraction-raising loop.
- **`LogosEngineProtocol` ADDED**: `logos/protocols.py` - `count_observations_explained_by(invariant_id)` and `estimate_description_length_without(invariant_id)` as `@runtime_checkable Protocol`. Kairos can now depend on this interface without importing LogosService directly.
- **`WorldModel` satisfies `LogosEngineProtocol`**: `logos/world_model.py` - both methods implemented. `count_observations_explained_by` returns `EmpiricalInvariant.observation_count`. `estimate_description_length_without` computes `current_complexity - 80 + observation_count * 50` (removes rule cost, adds back raw observation bits).
- **`ActionLog` primitive ADDED** (previous session): `primitives/causal.py` - canonical type for Axon intervention logs queryable by Kairos Stage 2.
- **`CausalHierarchyLevel` primitive ADDED** (previous session): `primitives/causal.py` - tier metadata type promoted from Kairos-internal to shared primitives.
- **`MemoryService.get_all_observations_with_context()` ADDED** (previous session): `memory/service.py` - returns `list[tuple[Episode, [Entity], [CausalNode ref]]]` for the last N cycles; satisfies Stage 1 CorrelationMiner input requirement.

## Pipeline Loop (2026-03-07 - root-cause fix)

**Root cause of 0% event coverage**: `run_pipeline()` was never called. All 14 Synapse event emissions are correctly wired inside `run_pipeline()`, but nothing triggered it. Event handlers (`_on_fovea_prediction_error`, `_on_episode_stored`, etc.) only buffer data - they do not trigger a pipeline run.

**Fix**: `start_pipeline_loop()` + `_pipeline_loop()` added to `KairosPipeline`:
- `start_pipeline_loop()` - creates a supervised asyncio Task (idempotent if already running)
- `_pipeline_loop()` - 15s warm-up delay, then calls `run_pipeline()` every `mining_interval_s` (default 300s)
- `core/registry.py._init_kairos()` - calls `kairos.start_pipeline_loop()` after wiring

All 14 events now fire on every pipeline cycle when sufficient data is available.

## Implemented (2026-03-08 autonomy audit)

### Critical Wiring Fixes
- **Registry: memory/neo4j/nexus wiring** - `_init_kairos()` previously only wired `event_bus` and `logos`. Added post-Phase-9 wiring block in `registry.py` (after nexus creation): `kairos.set_memory(memory)`, `kairos.set_neo4j(infra.neo4j)`, `await kairos.initialize()` (restores invariants from Neo4j), `kairos.set_nexus(nexus)`. Without this, Kairos ran with zero observation data and no persistence.

### New Subscriptions (5 added, now 16 total)
- **`GROUND_TRUTH_CANDIDATE`** (Nexus SG-NEXUS): Handler `_on_ground_truth_candidate` - pre-seeds correlation miner with Level 3+ epistemic fragments as causal candidates. Closes Nexus→Kairos ground truth feedback loop.
- **`EMPIRICAL_INVARIANT_CONFIRMED`** (Nexus SG-NEXUS): Handler `_on_empirical_invariant_confirmed` - reinforces matching invariant (`recency_weight=1.0`, `validated=True`), triggers tier re-evaluation. Level 4 confirmation prevents decay and accelerates promotion.
- **`PERCEPT_ARRIVED`** (SG-PERCEPT): Handler `_on_percept_arrived` - converts high-salience percepts into correlation candidates. Closes the raw observation stream path specified in Spec XI as `PERCEPT_BROADCAST`.
- **`EQUOR_DRIVE_WEIGHTS_UPDATED`** (SG-DRIVE): Handler `_on_drive_weights_updated` - converts significant drive weight deltas (≥0.05) into causal candidates. Mines constitutional causality.
- **`SOMATIC_DRIVE_VECTOR`** (SG-SOMA): Handler `_on_somatic_drive_vector` - buffers organism drive state as cross-context observations (capped at 300). Merges into `observations_by_context` at pipeline start. Enables discovery of inter-drive causal relationships (e.g. low Coherence causes elevated Care response).

### Bug Fixes
- **`query_observations_for_testing`**: Replaced private `self._hierarchy._find()` with public `self._hierarchy.find_invariant()` (AV4 consistency fix).

## Implemented (2026-03-08 self-causality)

### Self-Causality Tracker - NEW
- **`_run_self_causal_tracking()`** - called at end of every `run_pipeline()` after evolutionary metrics. Samples 6 internal variables and mines pairwise causal laws.
- **6 tracked variables**: `prediction_error_rate` (causal_surprise_rate from health), `coherence` (mean I-ratio from ledger), `hypothesis_count` (hierarchy.total_count), `re_success_rate` (tier3/total_discoveries), `sleep_frequency` (consolidation count), `consolidation_depth` (events per pipeline run).
- **Algorithm**: Rolling history (window=100), Pearson r at lag=1 (cause[:-1] → effect[1:]), threshold |r| > 0.35. Consistency test: 4 sub-windows, ≥50% must hold at r > 0.25 → accept as causal. Hold_rate update: emit `KAIROS_INTERNAL_INVARIANT` on discovery or hold_rate change > 0.1.
- **`_emit_internal_invariant(entry)`** - emits `KAIROS_INTERNAL_INVARIANT` (new SynapseEventType). Payload: `invariant_id`, `cause_variable`, `effect_variable`, `direction`, `lag_cycles`, `hold_rate`, `correlation`, `abstract_form`, `discovery_run`, `sample_count`.
- **`_internal_var_history`** - `dict[str, list[float]]`, rolling window per variable, max 100.
- **`_internal_invariants`** - `dict[(cause, effect), dict]`, accepted internal causal laws.
- **`IntelligenceContributionLedger.mean_i_ratio()`** - new method, mean intelligence ratio contribution across all tracked invariants.
- **Health report updated**: `self_causality` section added to `health()` response - `variables_tracked`, `invariants_discovered`, `active_invariants`, `sample_window`.
- **Nova subscribes** to `KAIROS_INTERNAL_INVARIANT` - organism learns its own causal dynamics; Thread subscribes for causal attribution cache.

### Known Issues / Remaining Work

- Memory `query_episodes()` API assumed but not yet confirmed against Memory system interface (used via `hasattr` guard in pipeline)
- Curriculum learning and strategic priority scheduling not yet implemented - needs cognitive budget integration with Synapse
- Abstract structure extraction is basic - only detects bidirectional and cross-domain novelty, not full pattern library
- `SKIA_HEARTBEAT` / `SKIA_SNAPSHOT_COMPLETED` not subscribed - no direct Skia integration; Kairos relies on Thymos for survival signaling
- RE-integrated abstraction raising (spec XII) not yet implemented - `_raise_abstraction_level()` uses heuristic keyword taxonomy rather than RE routing
- Self-causality: `consolidation_depth` and `sleep_frequency` variables are raw event counters (ever-increasing), not rate-normalized - correlation mining may find spurious trends. Future: normalize to per-window counts.

## Config (KairosConfig)

All thresholds in `types.py::KairosConfig`. Key fields:
- Correlation: `min_abs_mean_correlation=0.3`, `max_cross_context_variance=0.1`
- Direction: `min_direction_confidence=0.6`
- Tier promotion: `tier2_min_domains=2`, `tier3_min_domains=4`, `tier3_min_substrates=3`
- Health: `discovery_stall_threshold=0.1`, `confounder_inflation_threshold=0.5`, `corruption_surprise_threshold=0.4`
- Decay: `recency_weight` decays 0.95× per cycle, demote threshold 0.3, archive threshold 0.1
- Minimality: `minimality_hold_rate_tolerance=0.02` - scope conditions removable if impact ≤ 2%
- Tautology: `tautology_min_variables=2` - fewer unique roles → likely tautological
- Pipeline timing: `mining_interval_s=300.0`
- **D3 rename**: `tier3_min_observations=5` (was `tier3_min_contexts`) - total observation count threshold for Tier 3 eligibility
