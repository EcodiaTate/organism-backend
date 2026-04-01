# Fovea - System CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_20_Fovea.md` (Spec 20)
**System ID:** `fovea` (`SystemID.FOVEA` in `primitives/common.py`)

Attention = prediction error. Fovea computes how reality diverges from the world model and routes the delta as salience. It does not allocate attention - it measures surprise.

---

## What's Implemented

### Phase A - Error Computation
- 7-dimensional error decomposition: content, temporal, magnitude, source, category, causal, **economic**
- `FoveaPredictionEngine` with `generate_prediction()` and `compute_error()`
- `PrecisionWeightComputer` - **per-dimension precision** (2026-03-07): each of the 6 error dimensions gets an independent precision weight from `get_dimension_accuracy(context_type, dimension)` - no longer uniform
- `LogosWorldModel` protocol (now includes `get_dimension_accuracy()`) + `StubWorldModel` fallback + `LogosWorldModelAdapter`
- **`WorldModelAdapter`** (2026-03-07) - Memory-backed live implementation of `LogosWorldModel` protocol. Replaces `StubWorldModel` as the default world model. Queries Memory's Episode graph for content predictions (centroid of top-k matches), FOLLOWED_BY edge statistics for timing predictions, and confidence as (matching/total) episodes clamped [0.1, 0.9]. Caches with 2s TTL. Feed with `adapter.set_memory(memory_svc)` and inject with `fovea_service.set_world_model(adapter)`.
- **Distance helpers** in `world_model_adapter.py`: `_semantic_distance` (cosine on 768-D embeddings), `_timing_distance` (normalised delta / 60s), `_source_distance` (binary exact-match)

### Phase B - Atune Integration
- `FoveaAtuneBridge`: full pipeline (context → prediction → error → precision → weights → salience → habituation → threshold → routing → workspace)
- **18ms latency budget** (2026-03-07): `FoveaAtuneBridge.process_percept()` wraps `generate_prediction()` in `asyncio.wait_for(timeout=0.018)`. On timeout, falls back to a neutral `FoveaPredictionError(percept_id=...)` with zero errors - workspace routing still fires with neutral salience.
- `DynamicIgnitionThreshold` (percentile-based, floor=0.15, ceiling=0.85)
- `PerceptionGateway` (formerly AtuneService): EIS → Fovea → Workspace pipeline
- `GlobalWorkspace` with competitive selection

### Phase C - Learning and Habituation
- `AttentionWeightLearner`: reinforcement on world model update correlation, false-alarm decay, weight normalisation, Neo4j persistence
- `HabituationEngine`: signature-based grouping, incremental habituation (0.05/event, 90% max), dis-habituation on magnitude surprise, Neo4j persistence
- `HabituationCompleteInfo` with stochastic/learning_failure diagnosis

### Phase D - Self-Attention
- `InternalPredictionError` with 3x precision multiplier
- `InternalPredictionEngine`: predict → resolve → error generation for 4 internal error types (constitutional, competency, behavioral, affective)

### Cross-Cutting
- **Synapse events emitted:** `PREDICTION_ERROR`, `HABITUATION_DECAY`, `DISHABITUATION`, `INTERNAL_PREDICTION_ERROR`, `ATTENTION_PROFILE_UPDATE`, `WORKSPACE_IGNITION`, `HABITUATION_COMPLETE`, `FOVEA_ATTENTIONAL_DIVERGENCE`, `FOVEA_DIAGNOSTIC_REPORT` (every 50 errors - precision weights, habituation stats, economic trends, backlog, all adjustable params), `FOVEA_BACKPRESSURE_WARNING` (when unresolved backlog > 150)
- **Synapse events consumed:** `WORLD_MODEL_UPDATED`, `FOVEA_PREDICTION_ERROR`, `SELF_AFFECT_UPDATED`, `SLEEP_STAGE_TRANSITION`, `EVO_HYPOTHESIS_CONFIRMED`, `EVO_HYPOTHESIS_REFUTED`, `PERCEPT_ARRIVED` (timing gaps → WorldModelAdapter), `FOVEA_ATTENTION_PROFILE_UPDATE` (fleet sibling weight samples → KL divergence), `AXON_EXECUTION_REQUEST` (→ `_on_axon_execution_request()`: `_internal_engine.predict()` competency self-model; prediction_id stored in `_axon_prediction_ids[intent_id]`), `AXON_EXECUTION_RESULT` (→ `_on_axon_execution_result()`: `_internal_engine.resolve()` computes competency error delta), `ECONOMIC_VITALITY` (→ `_on_economic_vitality()`: feeds `EconomicPredictionModel`; emits ECONOMIC prediction errors when composite > 0.3), `FOVEA_PARAMETER_ADJUSTMENT` (→ `_on_parameter_adjustment()`: runtime adjustment of routing thresholds, threshold percentile, habituation speed - organism can now influence Fovea sensitivity without restart)
- **Neo4j persistence:** `FoveaWeights` and `FoveaHabituation` nodes (batched writes, max 1 per 10 changes; force on sleep/shutdown; restore on startup)
- **Genome:** `GenomeExtractionProtocol` - exports/imports learned weights, learning_rate, false_alarm_decay
- **Fitness:** `EvolutionaryObservable` emission with `attention_calibration` TPR metric
- **RE training:** `RETrainingExample` emission for (PredictionError, WorldModelUpdate) pairs (Stream 6)
- **Config:** `config/default.yaml` → `fovea.*` (habituation_increment, max_habituation_cap, learning_rate, false_alarm_decay, error_weights)
- **Public API (2026-03-07):** `FoveaService.get_metrics()` and `.weight_learner` property eliminate all `_bridge` private access from gateway.py and main.py
- **Identity fix (2026-03-07):** `normalisation.py` unknown-channel fallback changed from `SystemID.ATUNE` → `SystemID.FOVEA`
- **Dead types documented (2026-03-07):** `AttentionContext`, `MetaContext`, `SystemLoad` in `types.py` marked with deprecation/status docstrings

---

## Key Files

| File | Role |
|------|------|
| `service.py` | Main orchestrator - lifecycle, Synapse wiring, genome, fitness, RE emission, KL divergence |
| `gateway.py` | `PerceptionGateway` - EIS gate, normalisation, workspace orchestration |
| `integration.py` | `FoveaAtuneBridge`, `DynamicIgnitionThreshold`, head mapping |
| `prediction.py` | `FoveaPredictionEngine` - world model queries, error computation |
| `precision.py` | `PrecisionWeightComputer` - accuracy + stability → precision |
| `habituation.py` | `HabituationEngine` - signature tracking, dis-habituation |
| `learning.py` | `AttentionWeightLearner` - online weight learning, Neo4j persistence |
| `workspace.py` | `GlobalWorkspace` - competitive selection, broadcast |
| `internal.py` | `InternalPredictionEngine` - self-model violation detection |
| `types.py` | All Fovea types: `FoveaPredictionError`, `ErrorType`, weights, routing |
| `economic_model.py` | `EconomicPredictionModel` - EMA revenue/cost predictions, per-source trackers, trend velocity, composite error |
| `normalisation.py` | Input normalisation for 11 channel types |
| `protocols.py` | `LogosWorldModel` protocol, stub, adapter |
| `world_model_adapter.py` | `WorldModelAdapter` - Memory-backed `LogosWorldModel`; distance helpers; TTL cache |
| `extraction.py` | Entity/relation extraction (LLM-backed, post-broadcast) |
| `gradient.py` | Gradient attention (analytical Jacobian + numerical fallback) |
| `block_competition.py` | On-chain MEV-aware timing |

---

## Autonomy Gap Closure - Calibration Alerts + Threshold Persistence (2026-03-08)

### Part A: Calibration Alert (Fovea → Evo)

**Problem:** Fovea knows its attention is degrading (low TPR, high false alarms) but can't tell anyone.

**Solution:** Track consecutive poor cycles and emit `FOVEA_CALIBRATION_ALERT` for Evo to generate targeted tuning hypotheses.

**Implementation:**
- **New SynapseEventType:** `FOVEA_CALIBRATION_ALERT` (added to `synapse/types.py`)
  - Payload: `alert_type` ("low_tpr" or "high_false_alarm"), `current_value` (metric), `consecutive_cycles` (count), `threshold_params` (dict with percentile/floor/ceiling)
- **FoveaService tracking:** Two new instance fields added during init:
  - `_consecutive_poor_tpr: int` - incremented when TPR < 0.6, reset to 0 when TPR ≥ 0.6
  - `_consecutive_high_false_alarm: int` - incremented when false_alarm_rate > 0.4, reset to 0 when false_alarm_rate ≤ 0.4
- **Emission logic:** `_check_calibration_alert()` called every 50 errors (diagnostic report cadence):
  - Computes TPR and false alarm rate from `_weight_learner` stats
  - Checks both counters; if either reaches 5, emits `FOVEA_CALIBRATION_ALERT` and resets counter
  - Non-fatal: missing event bus gracefully skips emission
- **Evo subscription:** `_on_fovea_calibration_alert()` handler in `evo/service.py`:
  - Generates a `PatternCandidate` with confidence boosted by persistence (0.5 + 0.05×consecutive_cycles, capped at 0.8)
  - Queues candidate in `_pending_candidates` for next hypothesis generation pass
  - Evidence dict includes alert details (alert_type, current_value, threshold_params, recommendation)

### Part B: Threshold Persistence (Neo4j)

**Problem:** Evo-tuned ignition thresholds are lost on restart.

**Solution:** Persist `DynamicIgnitionThreshold` params to Neo4j on every mutation; restore on startup.

**Implementation:**
- **DynamicIgnitionThreshold enhancements** (`integration.py`):
  - New instance fields: `_neo4j_driver: Any`, `_instance_id: str`
  - `set_neo4j_driver(driver, instance_id)` - wire Neo4j post-construction
  - `adjust(delta)` - schedules `asyncio.ensure_future(_persist_state())` after mutation
  - `async _persist_state()` - `async with driver.session()` MERGE `(:FoveaThresholdState)` with percentile/floor/ceiling + last 500 samples (capped to keep node bounded)
  - `async persist_params()` - public force-flush, called after direct param mutation
  - `async restore_state_from_neo4j()` - `async with driver.session()` reads back all fields and seeds distribution window on startup
- **FoveaService wiring** (`service.py`):
  - `set_neo4j_driver()` propagates to `dynamic_threshold.set_neo4j_driver(driver, instance_id)`
  - `startup()` calls `await dynamic_threshold.restore_state_from_neo4j()`
  - `adjust_threshold_param()` calls `asyncio.ensure_future(dt.persist_params())` after every direct `_percentile`/`_floor`/`_ceiling` mutation - closes the gap where Evo ADJUST_BUDGET calls bypassed persistence
- **Non-fatal design:** All methods guard with `if self._neo4j_driver is None: return`; failures warn-log and don't raise

**Neo4j Schema:**
```cypher
(:FoveaThresholdState {
  instance_id: string,     // PK for MERGE
  percentile: float,       // clamped [10.0, 99.0]
  floor: float,            // clamped [0.01, 0.5]
  ceiling: float,          // clamped [0.5, 1.0]
  distribution_samples: list[float],  // last ≤500 salience values
  updated_at: datetime
})
```

## Autonomy Gap Closure - Previous (2026-03-08)

- **`FOVEA_DIAGNOSTIC_REPORT`** emitted every 50 errors (same cadence as fitness signal). Surfaces previously invisible internal state: per-dimension precision weights (learned accuracy profile), habituation engine stats (entry count, current increment), economic per-source trend data (worst source, revenue/efficiency errors), unresolved error backlog count, all adjustable parameters, weight learning state (reinforcements/decays/false alarms). Nova/Evo/RE subscribe to reason about attention calibration at planning time. New `SynapseEventType.FOVEA_DIAGNOSTIC_REPORT` added.
- **`FOVEA_PARAMETER_ADJUSTMENT`** subscription added. Any system (Equor/Nova) can now tune Fovea's routing thresholds and sensitivity at runtime without restart: `routing_threshold_equor` (default 0.3), `routing_threshold_oneiros` (default 0.5), `economic_route_threshold` (default 0.3), `economic_workspace_threshold` (default 0.5), `threshold_percentile` (clamped 30–95), `habituation_speed` (0.5–2.0× multiplier). All changes are clamped to safe ranges. Fovea emits `FOVEA_DIAGNOSTIC_REPORT` immediately after applying so requester can confirm. New `SynapseEventType.FOVEA_PARAMETER_ADJUSTMENT` added.
- **`FOVEA_BACKPRESSURE_WARNING`** emitted when unresolved error backlog exceeds 150 entries. Surfaces Oneiros processing pressure. Includes top 5 error domains by salience and actionable recommendation. De-duplicated by count (won't re-warn at same level). New `SynapseEventType.FOVEA_BACKPRESSURE_WARNING` added.
- **Runtime-adjustable routing thresholds**: module-level constants `_CONSTITUTIONAL_EQUOR_THRESHOLD`, `_CONSTITUTIONAL_ONEIROS_THRESHOLD`, `_ECONOMIC_ROUTE_THRESHOLD`, `_ECONOMIC_WORKSPACE_THRESHOLD` initialized to original hardcoded values. Each `FoveaService` instance stores its own adjusted values in `_constitutional_equor_threshold` etc. - organism can now shift routing sensitivity per Equor policy.

## What's Missing / Known Gaps

1. ~~**Economic prediction error dimension** (revenue-side blind spot)~~ - **RESOLVED (2026-03-08)**: `ErrorType.ECONOMIC` + `economic_error` field + `EconomicPredictionModel` + `ECONOMIC_VITALITY` subscription in `service.py`. Routes to `ErrorRoute.OIKOS` + `EVO` at threshold > 0.3.
1. ~~**Attentional divergence metric** (M4/SG2)~~ - **RESOLVED (2026-03-07)**: `_emit_attentional_divergence()` in `service.py` emits `FOVEA_ATTENTIONAL_DIVERGENCE` every 100 errors with KL(P||Q) between this instance's weight vector and fleet mean from sibling `FOVEA_ATTENTION_PROFILE_UPDATE` samples
2. **RE backend for semantic/causal distance** (M5) - cosine distance and key-overlap heuristic used; RE invocation deferred
3. **Neo4j graph writes for RE training pairs** - Stream 6 flows via Synapse only, not persisted to graph
4. ~~**No-op gateway methods**~~ - **RESOLVED (2026-03-08)**: All 7 stubs implemented. See `atune/CLAUDE.md` §Cross-System Modulation API.
5. **`SystemLoad` fields not read** (D3) - `run_cycle()` accepts `SystemLoad` but never reads `cpu_utilisation`, `memory_utilisation`, `queue_depth`
6. **Routing thresholds in `types.py` `compute_routing()`** - the hardcoded `> 0.3` / `> 0.5` routing logic in `FoveaPredictionError.compute_routing()` is not yet wired to the instance-level adjustable thresholds. The current fix adjusts instance state; the next step is to pass adjusted thresholds into `compute_routing()` at call sites in `service.py`.

## Event Emission Audit (2026-03-07)

All 6 spec_checker-required Fovea events are implemented and wired in `service.py`:
- `FOVEA_HABITUATION_DECAY` - fires when `error.habituation_level > 0.0 and dishabituation_info is None` per prediction error cycle
- `FOVEA_DISHABITUATION` - fires when `bridge.consume_dishabituation()` returns a result (sudden magnitude change detected)
- `FOVEA_WORKSPACE_IGNITION` - fires when `ErrorRoute.WORKSPACE in error.routes` (threshold crossed); also fires from `resolve_self()` for internal errors
- `FOVEA_ATTENTION_PROFILE_UPDATE` - fires on stale error flush (`flush_stale_errors()` returns non-empty) OR on weight learning update from world model events
- `FOVEA_HABITUATION_COMPLETE` - fires when `bridge.consume_habituation_complete()` returns a result (level > 0.8 without world model update)
- `FOVEA_INTERNAL_PREDICTION_ERROR` - fires from `resolve_self()` when self-model is violated (constitutional/competency/behavioral/affective errors)

All emit functions guard with `if self._event_bus is None: return` and wrap `try/except (ValueError, ImportError)` - missing enum value degrades silently. All 6 `SynapseEventType` values confirmed registered in `synapse/types.py`.

---

## Integration Surface

**Gives to the organism:**
- **Economic prediction errors** → Oikos (early warning before starvation), Evo (hypothesis about revenue failure source)
- Salience signal (prediction error magnitude) → workspace ignition → all systems
- Error routing → Evo (hypotheses), Logos (world model updates), Kairos (causal patterns)
- **Constitutional mismatch routing** → Equor (`constitutional_mismatch > 0.3`), Oneiros (`constitutional_mismatch > 0.5`)
- Habituation/dis-habituation signals → Soma (arousal), Thymos (anomaly)
- Fitness signal → Evo (selection pressure on attention calibration)
- RE training data → RE (Stream 6: novelty vs noise discrimination)
- Genome segment → Mitosis (heritable attentional phenotype)
- **Attentional divergence** → Benchmarks (`FOVEA_ATTENTIONAL_DIVERGENCE`: KL divergence vs fleet mean, speciation signal)

**Receives from the organism:**
- **Economic actuals** → Oikos (`ECONOMIC_VITALITY`) - revenue_24h, costs_24h, metabolic_efficiency, revenue_by_source
- World model predictions → Memory Episode graph (via `WorldModelAdapter`, replaces stub Logos)
- World model update correlation → Logos (`WORLD_MODEL_UPDATED`)
- Affect state → Soma (`SELF_AFFECT_UPDATED`) - τ coupling
- Sleep transitions → Oneiros (`SLEEP_STAGE_TRANSITION`)
- Hypothesis outcomes → Evo (`EVO_HYPOTHESIS_CONFIRMED/REFUTED`)
- External error signals → Simula/Synapse (`FOVEA_PREDICTION_ERROR`)
- **Live percept timing** → Synapse (`PERCEPT_ARRIVED`) - feeds `WorldModelAdapter.record_timing_observation()`
- **Fleet weight samples** → sibling Fovea instances (`FOVEA_ATTENTION_PROFILE_UPDATE`) - rolling window for KL divergence

---

### Resolved (2026-03-08) - Autonomy Audit

- **Learnable habituation parameters** - All 6 hardcoded constants in `habituation.py` (`_HABITUATION_INCREMENT`, `_MAX_HABITUATION`, `_DISHABITUATION_THRESHOLD`, `_DISHABITUATION_AMPLIFICATION`, `_HISTORY_WINDOW`, `_HABITUATION_COMPLETE_THRESHOLD`) replaced with instance-level `self._*` fields. Full API: `HabituationEngine.adjust_param()`, `get_learnable_params()`, `export_learnable_params()`, `import_learnable_params()`. Evo ADJUST_BUDGET compatible; genome-heritable.
- **Learnable weight learner parameters** - All 4 hardcoded constants in `learning.py` (`_CORRELATION_WINDOW_S`, `_WEIGHT_FLOOR`, `_WEIGHT_CEILING`, `_LEARNING_SALIENCE_THRESHOLD`) replaced with instance-level fields. Full API: `AttentionWeightLearner.adjust_param()`, `get_learnable_params()`, `export_learnable_params()`, `import_learnable_params()`.
- **Learnable economic model parameters** - `EconomicPredictionModel` EMA alpha, min_observations, trend_window, source_stale_cycles, and 4 composite weights all parameterized as instance-level fields. Full API: `adjust_param()`, `get_learnable_params()`, `export_learnable_params()`, `import_learnable_params()`. Per-source trackers receive model-level params at creation.
- **Enriched genome export** - `extract_genome_segment()` now includes `learnable_params` key with all subsystem params (learner + habituation + economic + threshold). `seed_from_genome_segment()` imports full learnable params with legacy fallback. Coverage: 5 → 25+ heritable params.
- **Self-introspection** - `FoveaService.introspect_autonomy()` returns learner_params, habituation_params, economic_model_params, threshold_config, effectiveness metrics (TPR, false alarms, reinforcements), and fleet_divergence. Exposed in `health()["autonomy"]`.
- **Unified learnable parameter API** - `FoveaService.adjust_learner_param()`, `adjust_habituation_param()`, `adjust_economic_param()`, `adjust_threshold_param()`, `get_all_learnable_params()`, `export_learnable_params()`, `import_learnable_params()`. All Evo ADJUST_BUDGET compatible.

## What's Missing / Known Gaps

1. ~~**Fovea autonomy communication** (M1/SG2)~~ - **RESOLVED (2026-03-08)**: `FOVEA_CALIBRATION_ALERT` event emitted when TPR or false alarm rate stays poor for 5+ cycles. Evo `_on_fovea_calibration_alert()` generates targeted attention-tuning hypotheses.
2. ~~**Threshold persistence on restart** (M2/SG4)~~ - **RESOLVED (2026-03-08)**: `DynamicIgnitionThreshold._persist_state()` writes `(:FoveaThresholdState)` to Neo4j on every adjust(); `restore_state_from_neo4j()` restores on startup.
3. **RE backend for semantic/causal distance** (M5) - cosine distance and key-overlap heuristic used; RE invocation deferred
4. **Neo4j graph writes for RE training pairs** - Stream 6 flows via Synapse only, not persisted to graph
5. ~~**No-op gateway methods**~~ - **RESOLVED (2026-03-08)**: All 7 gateway stubs now apply real coupling. See `atune/CLAUDE.md` §Cross-System Modulation API for full spec.
6. ~~**`SystemLoad` fields not read** (D3)~~ - **RESOLVED (2026-03-08)**: `run_cycle()` now reads `cpu_utilisation`/`memory_utilisation` (raises threshold up to +0.05 when >75%) and `queue_depth` (dampens arousal when queue >80% full to shrink workspace buffers).
7. ~~**Routing thresholds not wired**~~ - **RESOLVED (2026-03-08)**: `compute_routing()` on both `FoveaPredictionError` and `InternalPredictionError` now accepts `constitutional_equor_threshold`, `constitutional_oneiros_threshold`, `economic_route_threshold` params. All call sites in `service.py` pass the instance-level adjustable values. **Also fixed critical sequencing bug**: `_inject_constitutional_mismatch()` was called AFTER `compute_routing()` inside the bridge, meaning constitutional routing (EQUOR, ONEIROS) never fired. Now `service.py` re-runs `compute_routing()` post-injection.
8. ~~**`FoveaService.set_neo4j_driver()` dead wiring**~~ - **RESOLVED (2026-03-08)**: `fovea.set_neo4j_driver(infra.neo4j, config.instance_id)` now called in `registry.py` after `_init_fovea()`. Threshold persistence, weight learner persistence, and habituation persistence all have live Neo4j drivers.
9. ~~**All `PerceptionGateway` modulation methods dead-wired**~~ - **RESOLVED (2026-03-08)**: `set_belief_state(nova)` direct call + `RHYTHM_STATE_CHANGED` → `set_rhythm_state()` + `FEDERATION_PEER_CONNECTED` → `set_community_size()` + `EVO_HYPOTHESIS_CREATED` → `set_pending_hypothesis_count()` + `EPISODE_STORED` → `set_last_episode_id()` - all subscribed in `wire_intelligence_loops()` in `core/wiring.py`.

_Last updated: 2026-03-08 (Autonomy audit - dead wiring, routing bug, SystemLoad, threshold params all resolved)_
