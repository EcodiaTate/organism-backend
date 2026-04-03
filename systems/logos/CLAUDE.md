# Logos - CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_21_Logos.md`
**System ID:** `logos`
**Role:** Universal Compression Engine - cognitive budget, MDL scoring, four-stage compression cascade, entropic decay, Schwarzschild threshold detection, world model persistence, and intelligence metrics.

---

## What Is Implemented

**Phase A - Budget + MDL (complete)**
- `CognitiveBudgetManager` - hard capacity ceiling, tier-based allocation, pressure/urgency curves
- `MDLEstimator` - episode, hypothesis, schema, and generic scoring
- `COGNITIVE_PRESSURE` broadcast every 30s with `tier_utilization` payload
- `BUDGET_EMERGENCY` emission at >= 0.90 utilization (debounced 30s)
- Critical eviction at >= 0.95 utilization (synchronous, bypasses 300s timer)

**Phase B - Holographic Encoding + World Model (complete)**
- `HolographicEncoder` - delta computation between prediction and reality
- `WorldModel` - generative schemas, causal graph, predictive priors, empirical invariants
- `WorldModel.snapshot()` / `restore_from_snapshot()` - Mitosis genome support
- `WorldModel.register_schema()` - duplicate-safe schema registration

**Phase C - Compression Cascade + Entropic Decay (complete)**
- `CompressionCascade` - 4-stage pipeline (holographic → episodic → semantic → world model)
- Stage 3 schemas properly registered in world model (P5 fix)
- `EntropicDecayEngine` - access decay, compression decay, contradiction decay
- `record_contradiction()` method wired from `HYPOTHESIS_REJECTED` events
- Anchor memory creation with RE training data emission

**Phase D - Schwarzschild Cognition (complete)**
- `SchwarzchildCognitionDetector` - 5 indicators, self-prediction loop
- Fires `SCHWARZSCHILD_THRESHOLD_MET` once when threshold met

**Ecosystem Integration (complete)**
- 9 inbound Synapse subscriptions (previously only 1):
  - `FOVEA_PREDICTION_ERROR`, `MEMORY_CONSOLIDATED`, `EVO_HYPOTHESIS_CONFIRMED`,
  - `EVO_HYPOTHESIS_REFUTED`, `SCHEMA_INDUCED`, `KAIROS_TIER3_INVARIANT_DISCOVERED`,
  - `SLEEP_INITIATED`, `WAKE_ONSET`, `INSTANCE_SPAWNED`
- Neo4j persistence via `LogosPersistence` - schemas, causal edges, priors, invariants, metrics
- Eviction audit logs (immutable, append-only Neo4j nodes)
- World model restored from Neo4j on startup
- `LogosGenomeExtractor` - genome extraction and seeding for Mitosis
- RE training examples emitted on anchor creation and high-MDL compressions

---

## Synapse Events

**Emitted:**
- `COGNITIVE_PRESSURE` - every 30s, includes `tier_utilization`
- `BUDGET_EMERGENCY` - when utilization >= 0.90 (debounced 30s)
- `WORLD_MODEL_UPDATED` - on structural integration, includes `coverage_delta`, `complexity_delta`, `invariants_tested`, `invariants_violated`
- `LOGOS_INVARIANT_VIOLATED` - dedicated event when world model invariants are contradicted (Fix 1)
- `ANCHOR_MEMORY_CREATED` - includes `reason` field
- `INTELLIGENCE_METRICS` - every 60s
- `COMPRESSION_CYCLE_COMPLETE` - after decay cycles; includes `evicted_item_ids` + `evicted_items` (Fix 2)
- `SCHWARZSCHILD_THRESHOLD_MET` - once, ever
- `LOGOS_SCHWARZSCHILD_APPROACHING` - at 80% of any threshold indicator, once per lifetime (Fix 3)
- `LOGOS_BUDGET_ADMISSION_DENIED` - when `try_admit()` rejects a KU; includes tier, KU amounts, pressure (Fix 4)
- `RE_TRAINING_EXAMPLE` - on anchor creation and high-MDL compressions

**Subscribed:**
- `FOVEA_PREDICTION_ERROR` → feed prediction error into cascade
- `MEMORY_CONSOLIDATED` → update coverage metrics
- `EVO_HYPOTHESIS_CONFIRMED` → reinforce matching schemas
- `EVO_HYPOTHESIS_REFUTED` → record contradictions on related items
- `SCHEMA_INDUCED` → score via MDL, integrate if ratio > 1.0
- `KAIROS_TIER3_INVARIANT_DISCOVERED` → immediate world model integration
- `SLEEP_INITIATED` → pause real-time compression
- `WAKE_ONSET` → resume compression
- `INSTANCE_SPAWNED` → snapshot world model for child
- `INSTANCE_RETIRED` → prune `WorldModel.generative_schemas` where `source_system == retired_instance_id`
- `SYSTEM_MODULATION` → pause compression via `_sleep_active` gate; emit `SYSTEM_MODULATION_ACK`

---

## Integration Surface

**DI setters:**
- `set_synapse(SynapseService)` - event bus + subscriptions
- `set_memory(Any)` - Memory system (wired but not yet consumed)
- `set_memory_store(MemoryStoreProtocol)` - decay engine memory access
- `set_neo4j(Neo4jClient)` - world model persistence

**Key public methods:**
- `process_experience(RawExperience)` - full 4-stage cascade
- `encode_experience(RawExperience)` → `ExperienceDelta`
- `integrate_delta(ExperienceDelta)` → `WorldModelUpdate`
- `predict(context)` → `Prediction` (FoveaPredictionProtocol)
- `run_batch_compression()` - OneirosCompressionHooks
- `mark_anchor(item_id)` - protect from eviction
- `ingest_invariant(EmpiricalInvariant)` - KairosInvariantProtocol

---

## Known Remaining Issues

1. **Event handler `Any` types** - Event handlers use `Any` for the event parameter to match Synapse bus callback signature. Could be typed as `SynapseEvent` if bus guarantees that type.
2. **Neo4j restore is best-effort** - If Neo4j is unavailable on startup, world model starts empty with a warning. No retry mechanism.
3. **A5: Dual wiring path for `kairos.set_logos()`** - Both `core/registry.py` and `core/wiring.py` inject Logos into Kairos. Needs consolidation to canonical `wiring.py`.
4. **M9: Anchor review cycle not implemented** - Spec §XIV Conflict 3: anchor memories should be periodically reviewed for re-compression eligibility. Flag-only protection with no age-out mechanism.
5. **P8: Self-prediction first-cycle accuracy is always 0.0** - First record has no prior to compare against; the second cycle evaluates the first. Working as designed but may confuse dashboards.
6. **EXTERNAL - M6**: Benchmarks time-series wiring (blocked on Benchmarks).
7. **EXTERNAL - M7**: Atune salience integration (Atune not yet implemented).
8. **EXTERNAL - M8**: Nova EFE / world model generative coupling.
9. **EXTERNAL - SG6**: VitalitySystem hooks (blocked on VitalitySystem).

## Resolved Since Last Audit (07 March 2026 → v1.2)

- **P4 RESOLVED** - `_semantic_distance()` now uses value-aware `_value_distance()` (numeric relative error, string char-Jaccard, dict key-overlap). Captures large numeric novelty correctly.
- **P7 RESOLVED** - `run_batch_compression()` and `_trigger_critical_eviction()` now decrement the correct budget tier per evicted item via `evicted_item_types` in `DecayReport`.
- **A2 RESOLVED** - `_on_fovea_prediction_error()` now logs `ERROR` level with `exc_info=True`, `exc_type`, and `percept_id`. Not re-raised (Synapse bus must not crash).
- **A3 RESOLVED** - Anchor emit and RE training emit tasks tracked in `self._fire_forget_tasks`, cancelled cleanly in `shutdown()`.
- **D1/A4 RESOLVED** - `set_memory(Any)` and `self._memory: Any` removed entirely. Dead weight gone.
- **Sleep gate RESOLVED** - `_sleep_active` now gates `process_experience()`: experiences during Oneiros sleep return immediately with a `HOLOGRAPHIC_ENCODING`-stage result.
- **SG3/SG4 RESOLVED** - `LogosFitnessRecord` type added; `LogosPersistence.persist_fitness_record()` appends immutable `(:LogosFitnessTimeSeries)` nodes every 60s alongside `INTELLIGENCE_METRICS` broadcast. `set_instance_id()` DI method added.

## Autonomy Gap Closure - 08 March 2026 (v1.3)

- **Fix 1: LOGOS_INVARIANT_VIOLATED** - `integrate_delta()` + `process_experience()` now emit a dedicated `LOGOS_INVARIANT_VIOLATED` event (not just a WARNING log) when `wm.invariants_violated > 0`. `WORLD_MODEL_UPDATED` payload enriched with `invariants_tested` + `invariants_violated`. Kairos/Equor/Thymos can now react.
- **Fix 2: Eviction visibility** - `COMPRESSION_CYCLE_COMPLETE` payload now includes `evicted_item_ids` (up to 20), `evicted_items` (id+type pairs), `total_evicted_this_cycle`, `eviction_truncated` flag. Organism knows exactly what was discarded and why.
- **Fix 3: LOGOS_SCHWARZSCHILD_APPROACHING** - Added `_schwarzschild_approaching_emitted` guard + progressive warning in `_schwarzschild_loop()` at 80% of any threshold indicator. Organism gets foresight before cognitive reorganization, not binary surprise.
- **Fix 4: LOGOS_BUDGET_ADMISSION_DENIED** - `try_admit()` now fire-and-forget emits `LOGOS_BUDGET_ADMISSION_DENIED` via `asyncio.create_task()` when `CognitiveBudgetManager.increment()` returns False. Payload: tier, requested KU, tier used/limit, utilization%, total pressure, urgency, recommendation. No event bus injection needed in `budget.py`.

## Gap Closure - 07 March 2026 (v1.2)

- **CRITICAL: WorldModel integration persistence** - `LogosPersistence.persist_integration()` writes `(:WorldModel)` event node (all update fields), `[:COMPRESSES]` relationships to source `(:Episode)` and `(:SemanticNode)` nodes, and upserts `LogosSelfPredictionIndex` singleton. Called from `integrate_delta()` and `process_experience()` in service.py.
- **HIGH: MDL `raw_complexity` formula** - `MDLEstimator.compute_raw_complexity(content)` implemented as `byte_length(JSON(content)) × log2(unique_token_count)`. Used as fallback in `score_episode()` when `raw_complexity=0`. Exposed as `@staticmethod`.
- **HIGH: CognitiveBudget write path** - `CognitiveBudgetManager.record_compression_operation(cost_ku, gain_ku, tier)` added. Increments/decrements per-tier utilization on every compression operation. `current_utilization_state` property returns Synapse-readable `dict[str, float]`. Wired in `process_experience()`.
- **HIGH: Decay scheduling via theta clock** - `THETA_CYCLE_START` subscription added. `_on_theta_cycle()` counter: decay runs every 100 cycles, Schwarzschild self-prediction every 50. No polling loop needed - piggybacks on existing Synapse theta heartbeat.
- **MEDIUM: COGNITIVE_PRESSURE → Nova** - Nova subscribes to `COGNITIVE_PRESSURE`; `_on_cognitive_pressure()` calls `modulate_policy_k_from_pressure()` at thresholds 0.85 (40% reduction) and 0.95 (80% reduction).
- **MEDIUM: COGNITIVE_PRESSURE → Evo** - Evo subscribes to `COGNITIVE_PRESSURE`; `_cognitive_pressure_high` flag pauses `_generate_hypotheses_safe()` at ≥0.85 load (hysteresis: resumes below 0.75). Prevents LLM calls during compression pressure.
- **MEDIUM: COGNITIVE_PRESSURE → Simula** - Simula subscribes to `COGNITIVE_PRESSURE`; `_on_cognitive_pressure()` sets `_health._shallow_verification_mode=True` at ≥0.85 (skips Dafny/Lean/Z3). Hysteresis restores full verification below 0.75.
- **MEDIUM: Oikos coupling** - Oikos subscribes to `COGNITIVE_PRESSURE` in `attach()`. `_on_cognitive_pressure()` sets `_cognitive_load_high=True` at ≥0.90 (hysteresis: clears below 0.80). `check_metabolic_gate()` now denies any `priority ≥ GROWTH` action when cognitive load is high.
- **MEDIUM: Schwarzschild self-prediction scheduling** - `_on_theta_cycle()` (every 50 cycles) calls `run_self_prediction_cycle()`. `_latest_schwarzschild.self_prediction_accuracy` stored on service; surfaced in `INTELLIGENCE_METRICS` for Benchmarks.
