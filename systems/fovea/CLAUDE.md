# Fovea — System CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_20_Fovea.md` (Spec 20)
**System ID:** `fovea` (`SystemID.FOVEA` in `primitives/common.py`)

Attention = prediction error. Fovea computes how reality diverges from the world model and routes the delta as salience. It does not allocate attention — it measures surprise.

---

## What's Implemented

### Phase A — Error Computation
- 7-dimensional error decomposition: content, temporal, magnitude, source, category, causal, **economic**
- `FoveaPredictionEngine` with `generate_prediction()` and `compute_error()`
- `PrecisionWeightComputer` — **per-dimension precision** (2026-03-07): each of the 6 error dimensions gets an independent precision weight from `get_dimension_accuracy(context_type, dimension)` — no longer uniform
- `LogosWorldModel` protocol (now includes `get_dimension_accuracy()`) + `StubWorldModel` fallback + `LogosWorldModelAdapter`
- **`WorldModelAdapter`** (2026-03-07) — Memory-backed live implementation of `LogosWorldModel` protocol. Replaces `StubWorldModel` as the default world model. Queries Memory's Episode graph for content predictions (centroid of top-k matches), FOLLOWED_BY edge statistics for timing predictions, and confidence as (matching/total) episodes clamped [0.1, 0.9]. Caches with 2s TTL. Feed with `adapter.set_memory(memory_svc)` and inject with `fovea_service.set_world_model(adapter)`.
- **Distance helpers** in `world_model_adapter.py`: `_semantic_distance` (cosine on 768-D embeddings), `_timing_distance` (normalised delta / 60s), `_source_distance` (binary exact-match)

### Phase B — Atune Integration
- `FoveaAtuneBridge`: full pipeline (context → prediction → error → precision → weights → salience → habituation → threshold → routing → workspace)
- **18ms latency budget** (2026-03-07): `FoveaAtuneBridge.process_percept()` wraps `generate_prediction()` in `asyncio.wait_for(timeout=0.018)`. On timeout, falls back to a neutral `FoveaPredictionError(percept_id=...)` with zero errors — workspace routing still fires with neutral salience.
- `DynamicIgnitionThreshold` (percentile-based, floor=0.15, ceiling=0.85)
- `PerceptionGateway` (formerly AtuneService): EIS → Fovea → Workspace pipeline
- `GlobalWorkspace` with competitive selection

### Phase C — Learning and Habituation
- `AttentionWeightLearner`: reinforcement on world model update correlation, false-alarm decay, weight normalisation, Neo4j persistence
- `HabituationEngine`: signature-based grouping, incremental habituation (0.05/event, 90% max), dis-habituation on magnitude surprise, Neo4j persistence
- `HabituationCompleteInfo` with stochastic/learning_failure diagnosis

### Phase D — Self-Attention
- `InternalPredictionError` with 3x precision multiplier
- `InternalPredictionEngine`: predict → resolve → error generation for 4 internal error types (constitutional, competency, behavioral, affective)

### Cross-Cutting
- **Synapse events emitted:** `PREDICTION_ERROR`, `HABITUATION_DECAY`, `DISHABITUATION`, `INTERNAL_PREDICTION_ERROR`, `ATTENTION_PROFILE_UPDATE`, `WORKSPACE_IGNITION`, `HABITUATION_COMPLETE`, `FOVEA_ATTENTIONAL_DIVERGENCE`
- **Synapse events consumed:** `WORLD_MODEL_UPDATED`, `FOVEA_PREDICTION_ERROR`, `SELF_AFFECT_UPDATED`, `SLEEP_STAGE_TRANSITION`, `EVO_HYPOTHESIS_CONFIRMED`, `EVO_HYPOTHESIS_REFUTED`, `PERCEPT_ARRIVED` (timing gaps → WorldModelAdapter), `FOVEA_ATTENTION_PROFILE_UPDATE` (fleet sibling weight samples → KL divergence), `AXON_EXECUTION_REQUEST` (→ `_on_axon_execution_request()`: `_internal_engine.predict()` competency self-model; prediction_id stored in `_axon_prediction_ids[intent_id]`), `AXON_EXECUTION_RESULT` (→ `_on_axon_execution_result()`: `_internal_engine.resolve()` computes competency error delta), `ECONOMIC_VITALITY` (→ `_on_economic_vitality()`: feeds `EconomicPredictionModel`; emits ECONOMIC prediction errors when composite > 0.3)
- **Neo4j persistence:** `FoveaWeights` and `FoveaHabituation` nodes (batched writes, max 1 per 10 changes; force on sleep/shutdown; restore on startup)
- **Genome:** `GenomeExtractionProtocol` — exports/imports learned weights, learning_rate, false_alarm_decay
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
| `service.py` | Main orchestrator — lifecycle, Synapse wiring, genome, fitness, RE emission, KL divergence |
| `gateway.py` | `PerceptionGateway` — EIS gate, normalisation, workspace orchestration |
| `integration.py` | `FoveaAtuneBridge`, `DynamicIgnitionThreshold`, head mapping |
| `prediction.py` | `FoveaPredictionEngine` — world model queries, error computation |
| `precision.py` | `PrecisionWeightComputer` — accuracy + stability → precision |
| `habituation.py` | `HabituationEngine` — signature tracking, dis-habituation |
| `learning.py` | `AttentionWeightLearner` — online weight learning, Neo4j persistence |
| `workspace.py` | `GlobalWorkspace` — competitive selection, broadcast |
| `internal.py` | `InternalPredictionEngine` — self-model violation detection |
| `types.py` | All Fovea types: `FoveaPredictionError`, `ErrorType`, weights, routing |
| `economic_model.py` | `EconomicPredictionModel` — EMA revenue/cost predictions, per-source trackers, trend velocity, composite error |
| `normalisation.py` | Input normalisation for 11 channel types |
| `protocols.py` | `LogosWorldModel` protocol, stub, adapter |
| `world_model_adapter.py` | `WorldModelAdapter` — Memory-backed `LogosWorldModel`; distance helpers; TTL cache |
| `extraction.py` | Entity/relation extraction (LLM-backed, post-broadcast) |
| `gradient.py` | Gradient attention (analytical Jacobian + numerical fallback) |
| `block_competition.py` | On-chain MEV-aware timing |

---

## What's Missing / Known Gaps

1. ~~**Economic prediction error dimension** (revenue-side blind spot)~~ — **RESOLVED (2026-03-08)**: `ErrorType.ECONOMIC` + `economic_error` field + `EconomicPredictionModel` + `ECONOMIC_VITALITY` subscription in `service.py`. Routes to `ErrorRoute.OIKOS` + `EVO` at threshold > 0.3.
1. ~~**Attentional divergence metric** (M4/SG2)~~ — **RESOLVED (2026-03-07)**: `_emit_attentional_divergence()` in `service.py` emits `FOVEA_ATTENTIONAL_DIVERGENCE` every 100 errors with KL(P||Q) between this instance's weight vector and fleet mean from sibling `FOVEA_ATTENTION_PROFILE_UPDATE` samples
2. **RE backend for semantic/causal distance** (M5) — cosine distance and key-overlap heuristic used; RE invocation deferred
3. **Neo4j graph writes for RE training pairs** — Stream 6 flows via Synapse only, not persisted to graph
4. **No-op gateway methods** — 7 stub methods retained for backward compatibility with external callers (`set_belief_state`, `set_community_size`, `set_rhythm_state`, `nudge_dominance`, `nudge_valence`, `apply_evo_adjustments`, `receive_belief_feedback`)
5. **`SystemLoad` fields not read** (D3) — `run_cycle()` accepts `SystemLoad` but never reads `cpu_utilisation`, `memory_utilisation`, `queue_depth`

## Event Emission Audit (2026-03-07)

All 6 spec_checker-required Fovea events are implemented and wired in `service.py`:
- `FOVEA_HABITUATION_DECAY` — fires when `error.habituation_level > 0.0 and dishabituation_info is None` per prediction error cycle
- `FOVEA_DISHABITUATION` — fires when `bridge.consume_dishabituation()` returns a result (sudden magnitude change detected)
- `FOVEA_WORKSPACE_IGNITION` — fires when `ErrorRoute.WORKSPACE in error.routes` (threshold crossed); also fires from `resolve_self()` for internal errors
- `FOVEA_ATTENTION_PROFILE_UPDATE` — fires on stale error flush (`flush_stale_errors()` returns non-empty) OR on weight learning update from world model events
- `FOVEA_HABITUATION_COMPLETE` — fires when `bridge.consume_habituation_complete()` returns a result (level > 0.8 without world model update)
- `FOVEA_INTERNAL_PREDICTION_ERROR` — fires from `resolve_self()` when self-model is violated (constitutional/competency/behavioral/affective errors)

All emit functions guard with `if self._event_bus is None: return` and wrap `try/except (ValueError, ImportError)` — missing enum value degrades silently. All 6 `SynapseEventType` values confirmed registered in `synapse/types.py`.

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
- **Economic actuals** → Oikos (`ECONOMIC_VITALITY`) — revenue_24h, costs_24h, metabolic_efficiency, revenue_by_source
- World model predictions → Memory Episode graph (via `WorldModelAdapter`, replaces stub Logos)
- World model update correlation → Logos (`WORLD_MODEL_UPDATED`)
- Affect state → Soma (`SELF_AFFECT_UPDATED`) — τ coupling
- Sleep transitions → Oneiros (`SLEEP_STAGE_TRANSITION`)
- Hypothesis outcomes → Evo (`EVO_HYPOTHESIS_CONFIRMED/REFUTED`)
- External error signals → Simula/Synapse (`FOVEA_PREDICTION_ERROR`)
- **Live percept timing** → Synapse (`PERCEPT_ARRIVED`) — feeds `WorldModelAdapter.record_timing_observation()`
- **Fleet weight samples** → sibling Fovea instances (`FOVEA_ATTENTION_PROFILE_UPDATE`) — rolling window for KL divergence

---

_Last updated: 2026-03-08 (Economic prediction error dimension — ErrorType.ECONOMIC, EconomicPredictionModel, ECONOMIC_VITALITY subscription, OIKOS ErrorRoute, per-source revenue tracking, trend velocity error, RE training on revenue divergence)_
