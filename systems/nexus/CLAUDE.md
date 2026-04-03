# Nexus - CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_19_Nexus.md`
**System ID:** `nexus`
**Role:** Epistemic triangulation across the federation - instances share compressed world model structure, not raw beliefs. Convergence across maximally diverse compression paths is the primary evidence for ground truth.

---

## What Is Implemented

**Phase A - Fragment Infrastructure (complete)**
- `ShareableWorldModelFragment` with `SleepCertification`, `CompressionPath`, `TriangulationMetadata`
- `AbstractStructure` - formally typed Pydantic model replacing `dict[str, Any]`: `node_count`, `edge_count`, `schema_type`, `compression_ratio`, `provenance_hash`, `sleep_certified`, `node_type_distribution`, `edge_type_distribution`, `symmetry`, `invariants`, `nodes`, `edges`; `to_legacy_dict()` for backward compat
- `CompressionPath` - formally typed: `source_system`, `target_system`, `compression_steps: list[str]`, `fidelity_score`, `created_at`
- `ConvergenceDetector` - **WL-1 (Weisfeiler-Lehman 1D) graph isomorphism** primary path; iterative colour refinement (SHA-256, configurable iterations); canonical histogram comparison; 90% WL-1 + 10% size ratio blend; falls back to legacy heuristics when node lists absent
- `NexusService.extract_fragment`, `share_fragment`, `receive_fragment` - full fragment lifecycle
- `WorldModelFragmentShare` - extended with `sleep_certified: bool`, `consolidation_cycle_id: str`
- `WorldModelFragmentShareResponse` - extended with `wl1_used: bool`
- Telos fragment quality gate in `share_fragment`
- **Sleep certification gate**: fragment sharing blocked until `sleep_certified=True` on `AbstractStructure`; set by `ONEIROS_CONSOLIDATION_COMPLETE` handler

**Phase B - Divergence and Incentives (complete)**
- `InstanceDivergenceMeasurer` - five dimensions: domain, structural, attentional, hypothesis, temporal
- `DivergenceIncentiveEngine` - triangulation weight + divergence pressure → Thymos + Fovea; pressure threshold now read from `NexusConfig.divergence_pressure_threshold` (runtime-adjustable via genome mutation) rather than module-level constant `_PRESSURE_THRESHOLD = 0.4`
- Periodic `_divergence_loop` (5 min default) background task
- Evolutionary observable metrics emitted after each divergence cycle (Gap 5 fix)
- **Oikos metabolic coupling**: `NEXUS_CONVERGENCE_METABOLIC_SIGNAL` emitted after each `receive_fragment()`; convergence → `metabolic_signal="bonus"` (magnitude `convergence_growth_bonus=0.15`); persistent divergence (≥5 consecutive cycles without convergence) → `metabolic_signal="penalty"` (magnitude `-0.10`); `_consecutive_divergence_cycles` counter on `NexusService`
- `TRIANGULATION_WEIGHT_UPDATE` payload now includes `previous_weight` and `weight_delta` fields (spec §XI compliance)

**Phase C - Speciation (complete)**
- `SpeciationDetector` - threshold 0.8 triggers `SpeciationEvent`
- `SpeciationRegistry` - cognitive kind taxonomy, bridge pair tracking
- `InvariantBridge` - cross-speciation causal invariant exchange
  - `_are_abstractly_equivalent` now falls back to topology+domain+edge similarity when invariants empty (Gap 3 fix)
  - Threshold > 0.7 on weighted score; original Jaccard path unchanged when invariants populated
- Post-speciation fragment sharing block in `receive_fragment`
- `_on_speciation_event` auto-triggers bridge exchange
- **Neo4j persistence** via `NexusPersistence` - speciation events (now with `genome_distance: float` + `is_new_species: bool` properties), cognitive kinds, epistemic levels, bridge survivors (Gap 1 fix)
- `SpeciationEvent` type extended with `genome_distance: float = 0.0` and `is_new_species: bool = False`

**Phase D - Ground Truth Promotion (complete)**
- `GroundTruthPromotionPipeline` - Level 0→4 with Oneiros adversarial + Evo competition gates
- Equor constitutional protection request on Level 4
- Synapse events emitted on Level 3 and Level 4 promotions
- RE training examples emitted on Level 3+ promotions (Gap 4 fix)
- Promotion state persisted to Neo4j

**Phase E - Full Neo4j Persistence (complete, 2026-03-07)**
- `NexusPersistence.persist_fragments()` / `load_fragments()` - `ShareableWorldModelFragment` persisted on extract + certification update + triangulation update. Restored on `initialize()`.
- `NexusPersistence.persist_converged_invariants()` / `load_converged_invariants()` - `ConvergedInvariant` objects from `InvariantBridge` exchange persisted immediately. Previously created but never written (R1 closed).
- `NexusPersistence.persist_divergence_profiles()` / `load_divergence_profiles()` - `InstanceDivergenceProfile` persisted on every `measure_divergence()` call and on `_on_instance_spawned`. Restored on `initialize()`. Enables cross-session divergence tracking (R5 closed).
- `restore_full_state()` extended to return 7-tuple (adds fragments, invariants, profiles).
- Node types added: `(:NexusFragment)`, `(:NexusConvergedInvariant)`, `(:NexusDivergenceProfile)`.

**Synapse subscriptions active:** `WAKE_INITIATED`, `EMPIRICAL_INVARIANT_CONFIRMED`, `SPECIATION_EVENT`, `KAIROS_TIER3_INVARIANT_DISCOVERED`, `FEDERATION_SESSION_STARTED`, `ONEIROS_CONSOLIDATION_COMPLETE`, `EVO_HYPOTHESIS_CONFIRMED`, `EVO_HYPOTHESIS_REFUTED`, `INSTANCE_SPAWNED`, `INSTANCE_RETIRED`, `INCIDENT_RESOLVED`, `ONEIROS_THREAT_SCENARIO`

**Synapse events emitted:**
- `NEXUS_CONVERGENCE_METABOLIC_SIGNAL` - payload: `convergence_tier`, `economic_reward_usd`, `convergence_score`, `source_diversity`, `wl1_used`, `fragment_a/b_id`, `metabolic_signal` ("bonus"/"penalty"), `magnitude`, `consecutive_divergence_cycles`. **Subscribers: Oikos** (credits reward + triggers yield deployment signal at tier ≥ 2). **Benchmarks** indirectly via REVENUE_INJECTED re-broadcast.
- `NEXUS_EPISTEMIC_VALUE` - emitted 2× per divergence cycle. Payload is `EvolutionaryObservable.model_dump()`: `observable_type` ("federation_mean_divergence" | "speciation_event_count" | "epistemic_promotion_rate" | "local_epistemic_state"), `value`, `metadata` (counts, triangulation_weight). **Subscriber: Benchmarks** - `_on_nexus_epistemic_value` accumulates per-type rolling totals; on `local_epistemic_state` (end-of-cycle sentinel) emits `DOMAIN_KPI_SNAPSHOT` with `domain="nexus_epistemic"`, `epistemic_value_per_cycle`, `schema_quality_trend`, and full per-type means.
- `WORLD_MODEL_FRAGMENT_SHARE` - emitted per share with IIEP session context: `session_id`, `convergence_round`, full fragment metadata
- `NEXUS_CERTIFIED_FOR_FEDERATION` - emitted immediately after `ONEIROS_CONSOLIDATION_COMPLETE` when schemas are sleep-certified. Payload: `instance_id`, `schema_ids`, `consolidation_cycle_id`, `certified_fragment_count`. **Subscriber: Federation** - `_on_nexus_certified_for_federation` marks schemas as sleep-certified, emits `FEDERATION_KNOWLEDGE_SHARED`, and fires per-link fragment shares to COLLEAGUE+ peers.
- `DIVERGENCE_PRESSURE` - also emitted when `ONEIROS_THREAT_SCENARIO` reveals epistemic instability (convergence_score ≥ 0.4 against adversarial world state)

**Adapters available:** `LogosWorldModelAdapter`, `EvoHypothesisSourceAdapter`, `ThymosNexusSinkAdapter`, `KairosCausalSourceAdapter`

---

**New types:** `IIEPMessage` (wire envelope for all federation epistemic exchange), `IIEPFragmentType` (enum: WORLD_MODEL_FRAGMENT / CAUSAL_INVARIANT / CONVERGENCE_SUMMARY / SESSION_OPEN / SESSION_CLOSE). `NexusConfig.convergence_economic_reward_usdc_per_tier = 0.001`.

**IIEP session tracking:** `NexusService._iiep_sessions` dict maps `session_id → {initiator_id, fragment_id, started_at, convergence_round}`. Every `share_fragment` opens a session.

---

## Event Coverage Fix

**Root cause of 0% event coverage**: `_emit_divergence_observables(scores)` had an early return `if not scores: return`. In single-instance deployments with no federation peers, `measure_all_divergences()` returns `{}` (no active link IDs), so `NEXUS_EPISTEMIC_VALUE`, `DIVERGENCE_PRESSURE`, and `TRIANGULATION_WEIGHT_UPDATE` never fired. `FRAGMENT_SHARED` only fired when broadcast got ACCEPTED responses from peers - impossible without federation.

**Fix 1** - `_divergence_loop()` now calls `await self._emit_local_epistemic_value()` unconditionally after the existing observables call.

**Fix 2** - New `_emit_local_epistemic_value()` method emits `NEXUS_EPISTEMIC_VALUE` from local fragment state regardless of peer count. Payload includes: `local_fragment_count`, `remote_fragment_count`, `convergence_count`, `speciation_count`, `ground_truth_count`, `triangulation_weight`. Fires every 5 minutes (divergence loop interval).

## Autonomy Audit Fixes (8 Mar 2026)

**CRITICAL - Dead wiring resolved:**
- `nexus.set_neo4j(infra.neo4j)` now called in `core/registry.py` (post-init wiring block after Kairos). Previously: all `NexusPersistence` calls silently no-ops because `self._persistence` was always `None`. All Neo4j writes (speciation events, epistemic promotions, fragments, bridge survivors, converged invariants, divergence profiles) now actually persist.
- `nexus.set_kairos(KairosCausalSourceAdapter(kairos))` now called in same block. Previously: `sync_kairos_tier3()` always returned 0 and the bidirectional Kairos↔Nexus Tier 3 invariant loop was fully broken.

**Bug fix - AttributeError on TriangulationMetadata:**
- `TriangulationMetadata` had no `source_diversity` attribute (only `source_diversity_score`). Service accessed `fragment.triangulation.source_diversity` in `get_divergence_weighted_fragments()`, `_emit_re_training_example()`, and `_emit_local_epistemic_value()`. Fixed by adding `source_diversity` as an alias property.

**Static threshold → runtime-adjustable:**
- `DivergenceIncentiveEngine` hardcoded pressure threshold to `_PRESSURE_THRESHOLD = 0.4` module constant. Now reads `NexusConfig.divergence_pressure_threshold` (default 0.4) so it can be adjusted at runtime via genome mutation.

**Invisible telemetry fixed:**
- `TRIANGULATION_WEIGHT_UPDATE` now includes `previous_weight` and `weight_delta` fields (spec §XI).
- `INCIDENT_RESOLVED` handler now actually discounts triangulation confidence (×0.8) on fragments from the affected source system, then emits `DIVERGENCE_PRESSURE` to notify the bus. Previously the handler only logged.

---

## Known Issues / Remaining

1. **No RE integration (partial)** - Level 3/4 fragments emit RE training examples but are not routed to the RE training queue. Nexus is supposed to be the organism's long-term curriculum designer for the Reasoning Engine (Spec §XII). Queue routing not yet implemented. RE inference server not running.

2. ~~**No Benchmarks integration (partial)**~~ - **RESOLVED (8 Mar 2026)**: Benchmarks now subscribes to `NEXUS_EPISTEMIC_VALUE`. Handler `_on_nexus_epistemic_value` accumulates per-observable-type rolling totals; on `local_epistemic_state` sentinel emits `DOMAIN_KPI_SNAPSHOT` (domain=nexus_epistemic) with `epistemic_value_per_cycle` and `schema_quality_trend`.

3. **Sleep certification partially event-driven** - `ONEIROS_CONSOLIDATION_COMPLETE` handler does per-schema certification when `schema_ids` is in the event payload. Falls back to blanket certification via `WAKE_INITIATED` for backward compatibility. Full per-schema cert requires Oneiros to populate `schema_ids`.

4. **Cross-system import in adapters.py** - `EmpiricalInvariant` import is a sanctioned adapter bridge (lazy runtime import, not module-level). Documented as approved exception.

5. **Mitosis coupling (partial)** - `get_divergence_weighted_fragments()` exposed for genome inheritance. Not yet consumed by Mitosis - requires Mitosis to call during division.

6. ~~**Oikos coupling (partial)**~~ - **RESOLVED (8 Mar 2026)**: Oikos already subscribes to `NEXUS_CONVERGENCE_METABOLIC_SIGNAL` (NEXUS-ECON-1: credits `economic_reward_usd` to reserves + re-broadcasts as `REVENUE_INJECTED`). Extended (NEXUS-ECON-2): at convergence_tier ≥ 2, triggers a `YIELD_DEPLOYMENT_REQUEST` (5% of liquid_balance, 10–100 USDC) after metabolic gate check (YIELD priority). Logs to economic episode. Federation also now subscribes to `NEXUS_CERTIFIED_FOR_FEDERATION` (see #10 for IIEP unwrap gap).

7. **RE as ConvergenceDetector backend** - `ConvergenceDetector` still uses heuristic graph comparison. RE should be the inference backend for structural isomorphism (Spec §XII point 2). Blocked until RE inference server is live.

8. **RE domain suggestions in divergence pressure** - `compute_divergence_pressure` selects frontier_domains heuristically; RE should suggest domains (Spec §XII point 3).

9. **`ConvergenceDetector` WL-1 not perfectly GI-correct** - WL-1 colour refinement cannot distinguish all non-isomorphic graphs (e.g., certain regular graphs). Rare false-positives possible; acceptable for epistemic triangulation where probabilistic convergence is sufficient. Upgrade path: RE inference backend for full graph isomorphism (blocked until RE inference server live).

10. **IIEP responder-side unwrap not implemented** - `receive_fragment` accepts `WorldModelFragmentShare` directly; callers that send raw `IIEPMessage` envelopes need to unwrap before calling. Synapse event-driven inbound path (when Federation emits `WORLD_MODEL_FRAGMENT_SHARE`) needs a subscriber that calls `receive_fragment`. Currently only direct method calls are used.

---

## Integration Surface

**Injects into:** `NexusService` via `set_*` methods:
- `set_synapse(SynapseService)` - required for all events
- `set_world_model(LogosWorldModelProtocol)` - fragment extraction + divergence
- `set_fovea(FoveaAttentionProtocol)` - attentional diversity + divergence pressure signal
- `set_federation(FederationFragmentProtocol)` - fragment transport
- `set_thymos(ThymosDriveSinkProtocol)` - divergence pressure routing
- `set_oneiros(OneirosAdversarialProtocol)` - Level 4 adversarial test
- `set_evo(EvoCompetitionProtocol)` - Level 4 hypothesis competition
- `set_evo_hypothesis_source(EvoHypothesisSourceProtocol)` - divergence measurement
- `set_equor(EquorProtectionProtocol)` - constitutional protection of Level 4
- `set_telos(TelosFragmentGateProtocol)` - fragment quality gate
- `set_logos_adapter(LogosWriteBackProtocol)` - write-back to Logos world model
- `set_kairos(KairosCausalSourceProtocol)` - Tier 3 invariant pull sync
- `set_neo4j(Neo4jClient)` - persistent state (speciation, promotions, bridges)

**Lifecycle:** `await nexus.initialize(instance_id)` → `await nexus.start_background_loops()` → `nexus.subscribe_to_synapse_events()`

**Key public methods:**
- `extract_fragment(schema_id, abstract_structure, ...)` - wrap a Logos schema for sharing
- `share_fragment(fragment_id)` - broadcast to federation
- `receive_fragment(WorldModelFragmentShare)` - inbound from peer
- `measure_all_divergences()` - five-dimensional divergence across all active links
- `evaluate_all_promotions()` - run promotion pipeline over all local fragments
- `exchange_invariants_across_bridge(remote_logos, remote_instance_id)` - post-speciation exchange
- `get_federation_confidence(schema_id)` - on-demand triangulation confidence for Logos
- `get_divergence_weighted_fragments()` - Mitosis genome inheritance weighting
- `get_epistemic_metabolic_value()` - Oikos metabolic representation of epistemic assets

---

## Architecture Notes

- Nexus decides WHAT to share. Federation handles HOW (encryption, auth, transport).
- All external system dependencies are protocol-injected - never import Logos/Fovea/Thymos directly.
- Fragment store is bounded (`max_stored_fragments=1000`). Lowest quality-score fragments evicted.
- Circuit breaker per federation link (`_max_consecutive_failures=5`).
- `_divergence_loop` runs every 5 minutes by default (`NexusConfig.divergence_measurement_interval_s`).
- `NexusConfig` WL-1 + metabolic fields: `wl1_iterations=3`, `wl1_min_nodes_to_activate=3`, `convergence_growth_bonus=0.15`, `divergence_penalty_threshold=5`, `divergence_metabolic_penalty=-0.10`.
- Neo4j persistence uses batched UNWIND writes. Node labels: `NexusSpeciationEvent`, `NexusCognitiveKind`, `NexusEpistemicPromotion`, `NexusBridgeSurvivor`. `(:NexusSpeciationEvent)` now stores `genome_distance` and `is_new_species`.
- State restore on `initialize()` - graceful degradation if Neo4j unavailable.
- `federation_confidence` field on `GenerativeSchema` separates Nexus epistemic confidence from Logos `compression_ratio`.
