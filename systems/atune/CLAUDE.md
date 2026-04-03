# Atune - CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_03_DISCONTINUED.md`
**System ID:** `atune`
**Role:** Sensory cortex & Global Workspace. Receives all input (text, voice, sensor, events), scores percepts via 7-head salience, and broadcasts the winner to all cognitive systems. If Memory is selfhood and Equor is conscience, Atune is awareness.

---

## Architecture

**Pipeline (per theta cycle, ≤150ms):**
```
RawInput → Normalise → EIS Gate → Prediction Error → 7-Head Salience
→ Workspace Competition → Winner Broadcast → Memory Enrichment
→ Async Entity Extraction → Affect Persistence
```

**Core modules:** `service.py` (orchestrator), `normalisation.py`, `salience.py` (7 heads), `workspace.py` (Global Workspace), `prediction.py`, `momentum.py`, `affect.py`, `meta.py`, `extraction.py`, `market_pattern.py`

**Input channels:** `TEXT_CHAT`, `VOICE`, `GESTURE`, `SENSOR_IOT`, `CALENDAR`, `EXTERNAL_API`, `SYSTEM_EVENT`, `MEMORY_BUBBLE`, `AFFECT_SHIFT`, `EVO_INSIGHT`, `FEDERATION_MSG`

---

## Seven Salience Heads

| Head | Weight | Basis |
|------|--------|-------|
| Novelty | 0.20 | Prediction error × (1 − habituation × 0.5); contradiction bonus ×1.3 |
| Risk | 0.18 | Embedding similarity to known threats (Memory-backed) |
| Goal | 0.15 | Cosine similarity to active goal embeddings |
| Identity | 0.15 | Relevance to core self entities |
| Consequence | 0.12 | Temporal proximity × consequence scope |
| Social | 0.10 | Agent mentions + sentiment + conflict |
| Economic | 0.10 | Market keyword matching via Evo patterns |

All scores precision-weighted by `AffectState` (Friston 2010: `precision ~ 1/uncertainty`) then meta-weighted by `MetaAttentionController`. Momentum tracking adds first/second derivatives per head; ACCELERATING heads trigger arousal nudge.

**Ignition threshold:** 0.3 (configurable). Winner broadcast to all Synapse subscribers.

---

## Key Types

```python
class AtuneConfig:
    ignition_threshold: float = 0.3
    workspace_buffer_size: int = 32
    spontaneous_recall_base_probability: float = 0.02
    max_percept_queue_size: int = 100
    affect_persist_interval: int = 10
    cache_identity_refresh_cycles: int = 1000
    cache_risk_refresh_cycles: int = 500

class SalienceVector:
    scores: dict[str, float]       # per-head
    composite: float
    prediction_error: PredictionError
    gradient_attention: dict[str, GradientAttentionVector]  # token-level attribution
    momentum: dict[str, HeadMomentum]
    threat_trajectory: ThreatTrajectory
```

---

## EIS Integration

Every percept passes through `eis_service.eis_gate()` before salience scoring:
- `BLOCK` → reject, log, return `None`
- `ATTENUATE` → accept with reduced salience
- `PASS` → continue

EIS result stored in `percept.metadata["eis_result"]` - RiskHead reads this directly.

---

## Memory Integration

**Retrieval** (before broadcast): `memory_client.retrieve_context(embedding, text, max_results=10)`
**Storage** (after broadcast): `memory_client.store_percept_with_broadcast(percept, salience, affect)` - stores Episode, emits `EPISODE_STORED`
**Entity extraction** (async, background): LLM extract → resolve entities → `MENTIONED_IN` edges on Neo4j

Temporal causality: episodes linked via `FOLLOWED_BY` edge if gap ≤1h.

---

## Wiring (Startup Order)

```python
atune.set_eis(eis_service)           # before startup
atune.set_memory_service(memory)     # before startup
atune.set_synapse(synapse)           # during startup
atune.set_belief_state(nova_reader)  # after startup (fallback available)
atune.set_soma(soma_service)
atune.set_market_pattern_detector(templates, axon)  # fast-path reflex arc
```

**Fast-path:** `MarketPatternDetector` detects pre-approved patterns → `FastPathIntent` → `axon.execute_fast_path()` directly (bypasses Nova/Equor, ≤50ms budget).

---

## What's Implemented

### PERCEPT_ARRIVED
`PerceptionGateway.ingest()` (`fovea/gateway.py`) emits `PERCEPT_ARRIVED` with `source_system="atune"` immediately after a percept clears the EIS gate. Fovea's `WorldModelAdapter` subscribes to this event for inter-event timing statistics. The spec_checker credits this event to the "atune" system. Payload: `percept_id`, `source_system`, `channel`, `timestamp_iso`, `modality`.

Note: `PerceptionGateway` (in `fovea/`) is the live implementation of what Spec 03 calls "Atune". It is aliased as `AtuneService` for backward compatibility.

### ATUNE_REPAIR_VALIDATION
Emitted by Thymos (`service.py:_broadcast_repair_completed()`) with `source_system="thymos"` 60 cycles after a repair, checking whether the incident fingerprint re-fires. Thymos also subscribes to it. The event type exists in `SynapseEventType` and fires as part of the repair validation pipeline.

### Arousal-Scaled Buffer Sizes
`GlobalWorkspace` (`fovea/workspace.py`) now sizes its three deques dynamically based on Soma's arousal signal:

```python
@staticmethod
def _compute_buffer_sizes(arousal: float) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, arousal))
    percept_q   = int(50 + 350 * clamped)   # 50–400
    contrib_q   = int(20 + 80  * clamped)   # 20–100
    broadcast_h = int(10 + 30  * clamped)   # 10–40
    return percept_q, contrib_q, broadcast_h
```

`update_arousal(arousal)` is called on every `ALLOSTATIC_SIGNAL` event (subscribed in `PerceptionGateway.set_synapse()`). Resizing preserves existing items - Python `deque(existing, maxlen=new_size)` drops oldest entries from the left when shrinking.

When `ingest()` finds the percept queue full, it emits **`PERCEPT_DROPPED`** (SynapseEventType added 2026-03-08):
```python
# Payload
{
    "percept_id": str,
    "dropped_salience": float,
    "queue_size": int,
    "arousal": float,
    "channel": str,
}
```

### Evolvable Curiosity Parameters
Three curiosity parameters are registered with Evo's parameter tuner and are heritable via `AtuneGenomeFragment`:

| Evo path | Default | Range | Step | Workspace field |
|----------|---------|-------|------|-----------------|
| `atune.workspace.base_prob` | 0.02 | 0.005–0.10 | 0.003 | `_spontaneous_base_prob` |
| `atune.workspace.cooldown_cycles` | 20.0 | 5–50 | 1.0 | `_cooldown_cycles` |
| `atune.workspace.curiosity_boost` | 0.03 | 0.01–0.10 | 0.005 | `_curiosity_boost` |

`GlobalWorkspace` exposes:
- `adjust_param(name, delta)` - clamps within spec bounds
- `get_learnable_params()` / `export_learnable_params()` / `import_learnable_params(params, jitter=True)` - full round-trip with ±5% Gaussian jitter on import
- `record_curiosity_outcome(percept_id, positive)` - marks a previously pending spontaneous recall outcome
- `curiosity_hit_rate` - rolling hit rate over recorded outcomes (used by Evo fitness scoring)

`FoveaService` wires these into the rest of the organism:
- `_on_evo_workspace_param_adjusted()` - handles `EVO_PARAMETER_ADJUSTED` for `atune.workspace.*`, converts absolute Evo value to delta from default, calls `workspace.adjust_param()`
- `export_atune_genome(instance_id, generation)` - returns `AtuneGenomeFragment` from live workspace state
- `_apply_inherited_atune_genome_if_child()` - reads `ORGANISM_ATUNE_GENOME_PAYLOAD` env var at startup, applies parent curiosity params with jitter, emits `GENOME_INHERITED`
- `_on_curiosity_positive_signal()` - subscribes to `EVO_HYPOTHESIS_CREATED` and `COHERENCE_SHIFT`; resolves all pending (`-1`) curiosity outcomes as positive

### AtuneGenomeFragment (primitives/genome_inheritance.py)
```python
class AtuneGenomeFragment(EOSBaseModel):
    genome_id: str
    instance_id: str
    generation: int
    extracted_at: datetime
    curiosity_params: dict[str, float]
    # {"base_prob": 0.02, "cooldown_cycles": 20.0, "curiosity_boost": 0.03}
    buffer_scale_arousal: float   # arousal at extraction time
    curiosity_hit_rate: float     # rolling hit rate at extraction time
```

Exported from `primitives/__init__.py`.

**Status (Atune standalone): Not yet implemented** - the Atune system directory contains only this CLAUDE.md. No `.py` files exist.

The spec (Spec 03) fully defines the interface. Key things to implement:
- `AtuneService` with `ingest()`, `run_cycle()`, `contribute()`, `receive_belief_feedback()`
- All 7 salience heads as separate scoring functions
- `GlobalWorkspace` with ignition + broadcast
- `MarketPatternDetector` + `AtuneCache` with staggered refresh cycles

---

## What's Missing (All of it - system unimplemented as standalone)

1. No `ingest()` / `run_cycle()` implementation (lives in `fovea/`)
2. No 7-head salience engine (lives in `fovea/`)
3. No Global Workspace competition logic (lives in `fovea/workspace.py`)
4. No momentum tracking or gradient attention vectors
5. No EIS integration
6. No `MarketPatternDetector` fast-path trigger
7. No async entity extraction pipeline
8. No `AtuneCache` with refresh cycles

---

## Key Constraints

- Total cycle: ≤150ms; normalisation ≤5ms; EIS ≤30ms; 7-head scoring ≤40ms
- Workspace queue bounded at 100 percepts; overflow drops oldest
- Feedback loop protection: bias clamp ±0.40, inertia decay 0.95, per-source history 20
- Entity extraction is non-blocking - `asyncio.create_task()`, loop continues immediately
- All inter-system communication via Synapse bus - no direct system imports

## Cross-System Modulation API (2026-03-08 - formerly no-op stubs)

All seven methods on `PerceptionGateway` now apply real coupling.
State is stored in five gateway fields (all read by `health()["modulation"]`):

| Field | Default | Updated by |
|---|---|---|
| `_current_belief_confidence` | 0.5 | `set_belief_state()` |
| `_current_community_size` | 1 | `set_community_size()` |
| `_rhythm_state` | "NEUTRAL" | `set_rhythm_state()` |
| `_affect_dominance` | 0.5 | `nudge_dominance()` |
| `_affect_valence` | 0.5 | `nudge_valence()` |

### `set_belief_state(reader)` - Precision modulation (Nova)
Queries `reader.get_current_beliefs()` for average confidence.
Maps confidence → `learning_salience_threshold` shift (±10% nudge):
- High confidence (≈1.0) → lower threshold by up to 0.005 (confirming percepts cost less to attend)
- Low confidence (≈0.0) → raise threshold by up to 0.005 (surprises need more scrutiny)

### `set_community_size(n)` - Social scaling
- `n ≤ 1` (solo): attenuates `source` error weight by 20% to suppress federation noise
- `n ≥ 10` (large community): amplifies `source` weight by 15% to boost convergence percepts
- `1 < n < 10`: linear interpolation; always re-normalises the weight vector

### `set_rhythm_state(state)` - Processing mode (Synapse)
- `FLOW` → `DynamicIgnitionThreshold.adjust(+0.06)` + mild arousal drop (narrow focus)
- `STRESS` → `adjust(-0.08)` + arousal boost (widen aperture, let everything through)
- `BOREDOM` → content error weight ×1.2, spontaneous recall `base_prob` ×1.3
- `DEEP_PROCESSING` → `adjust(+0.12)`, reset `base_prob` to default (current workspace dominates)

### `nudge_dominance(delta)` - Affect coupling (Soma)
Accumulates delta onto `_affect_dominance` ∈ [0, 1].
- `d > 0.7` (agency): economic weight ×(1 + 0.15×strength), causal weight ×(1 + 0.10×strength)
- `d < 0.3` (threat): causal weight ×(1 + 0.20×strength), category weight ×(1 + 0.10×strength)
Re-normalises weights after every call.

### `nudge_valence(delta)` - Affect coupling (Soma)
Accumulates delta onto `_affect_valence` ∈ [0, 1].
- `v > 0.7` (positive): attenuation of causal/category weights + small threshold raise (+0.02×strength)
- `v < 0.3` (negative): amplification of causal/category weights + threshold lower (−0.04×strength)

### `apply_evo_adjustments(params)` - Evo parameter feed
Strips `atune.` / `fovea.` / `workspace.` / `threshold.` / `habituation.` prefixes,
then routes to:
- `GlobalWorkspace.adjust_param()` for `base_prob`, `cooldown_cycles`, `curiosity_boost`
- `AttentionWeightLearner.adjust_param()` for learner hyperparams
- `FoveaService.adjust_threshold_param()` for `percentile`, `floor`, `ceiling`
- `FoveaService.adjust_habituation_param()` for habituation hyperparams
Logs every applied key via `gateway_info`.

### `receive_belief_feedback(feedback)` - Attention learning signal (Nova)
Expects `{percept_id, outcome, dominant_error_type}` (dict or object).
- Positive outcome (`"good"` / `"positive"` / `"confirmed"`):
  - `workspace.record_curiosity_outcome(percept_id, positive=True)`
  - Reinforces `dominant_error_type` weight by `learning_rate × 0.5`
- Negative outcome (`"bad"` / `"negative"` / `"refuted"`):
  - `workspace.record_curiosity_outcome(percept_id, positive=False)`
  - Suppresses `dominant_error_type` weight by `false_alarm_decay × 2.0`
All weight mutations call `_normalise_weights()` to maintain a valid distribution.

---

## Autonomy Gap Closure (2026-03-08 - Dead Wiring + Routing Fixes)

### Dead Wiring Resolved

All formerly dead `set_X()` methods on `PerceptionGateway` (gateway.py) are now called from `core/wiring.py`:

| Method | Was | Now | How |
|--------|-----|-----|-----|
| `set_belief_state(nova)` | never called | called in `wire_intelligence_loops()` | direct call after nova is available |
| `set_rhythm_state(state)` | never called | subscribed in `wire_intelligence_loops()` | `RHYTHM_STATE_CHANGED` event handler |
| `set_community_size(n)` | never called | subscribed in `wire_intelligence_loops()` | `FEDERATION_PEER_CONNECTED` event handler reads `peer_count` |
| `set_pending_hypothesis_count(n)` | never called | subscribed in `wire_intelligence_loops()` | `EVO_HYPOTHESIS_CREATED` handler reads `hypothesis_count` |
| `set_last_episode_id(id)` | never called | subscribed in `wire_intelligence_loops()` | `EPISODE_STORED` handler reads `episode_id` - fixes entity extraction MENTIONED_IN edges |

`wire_intelligence_loops()` now accepts `synapse` as an optional kwarg. The call in `registry.py` passes `synapse=synapse`.

### FoveaService.set_neo4j_driver() - Dead Wiring Resolved

`fovea.set_neo4j_driver(infra.neo4j, config.instance_id)` now called in `registry.py` after `_init_fovea()`. Enables:
- `DynamicIgnitionThreshold` threshold persistence/restore across restarts (Part B gap)
- `AttentionWeightLearner` Neo4j persistence
- `HabituationEngine` Neo4j persistence

### Constitutional Routing - Sequencing Bug Fixed

`_inject_constitutional_mismatch()` in `service.py` set `error.constitutional_mismatch` AFTER `compute_routing()` had already run inside the bridge - meaning EQUOR and ONEIROS routing never fired for constitutional errors. Fixed by re-running `compute_routing()` in `service.py` after mismatch injection, now using the instance-level adjustable thresholds (`_constitutional_equor_threshold`, `_constitutional_oneiros_threshold`, `_economic_route_threshold`).

### `compute_routing()` - Adjustable Thresholds Wired

`FoveaPredictionError.compute_routing()` and `InternalPredictionError.compute_routing()` in `types.py` now accept:
- `constitutional_equor_threshold` (default 0.3)
- `constitutional_oneiros_threshold` (default 0.5)
- `economic_route_threshold` (default 0.3)

All three call sites in `service.py` now pass the instance-level adjustable thresholds. Evo ADJUST_BUDGET tuning via `FOVEA_PARAMETER_ADJUSTMENT` now actually changes routing behaviour rather than updating dead state.

### SystemLoad Fields - D3 Gap Closure

`run_cycle(system_load)` in `gateway.py` previously accepted `SystemLoad` but ignored `cpu_utilisation`, `memory_utilisation`, and `queue_depth`. Now:
- `cpu_utilisation > 0.75` OR `memory_utilisation > 0.75`: raises ignition threshold proportionally (up to +0.05) to shed load
- `queue_depth > 80% of max_percept_queue_size`: nudges arousal down so `GlobalWorkspace._compute_buffer_sizes()` returns smaller deque maxlens

---

## Integration Surface

| System | Direction | Purpose |
|--------|-----------|---------|
| EIS | ← | Epistemic threat screening per percept |
| Memory | ↔ | Context retrieval (pre-broadcast) + episode storage (post-broadcast) |
| Soma | ← | Precision weights + `ALLOSTATIC_SIGNAL` (buffer sizing) + `nudge_dominance/valence` (affect-coupled weights) |
| Nova | ← | `set_belief_state()` (precision modulation) + `receive_belief_feedback()` (attention learning signal) |
| Synapse | ← | `set_rhythm_state()` via `RHYTHM_STATE_CHANGED` event (processing mode adaptation) |
| Evo | ↔ | `apply_evo_adjustments()` (param feed) + curiosity outcomes; `EVO_PARAMETER_ADJUSTED` tunes params; `EVO_HYPOTHESIS_CREATED` → `set_pending_hypothesis_count()` |
| Federation | ← | `set_community_size()` via `FEDERATION_PEER_CONNECTED` event (social scaling) |
| Memory | ← | `set_last_episode_id()` via `EPISODE_STORED` event (entity extraction MENTIONED_IN edges) |
| Axon | → | Fast-path dispatch via MarketPatternDetector |
| Logos | ← | `COGNITIVE_PRESSURE` → raises ignition threshold_percentile (75 at ≥0.85, 85 at ≥0.95, restore 60 below 0.75) |
| Skia | ← | `SYSTEM_MODULATION` → workspace arousal drops to 0.1 (minimum throughput); emits `SYSTEM_MODULATION_ACK` |
| All systems | → | Workspace broadcast (winner percept) |

---

## Synapse Subscriptions in `PerceptionGateway.set_synapse()`

| Event | Handler | Purpose |
|-------|---------|---------|
| `ALLOSTATIC_SIGNAL` | `_on_allostatic_signal` | Arousal-scaled workspace buffer sizing |
| `COGNITIVE_PRESSURE` | `_on_cognitive_pressure` | Raise ignition threshold under Logos budget pressure |
| `SYSTEM_MODULATION` | `_on_system_modulation` | VitalityCoordinator austerity - throttle + ACK |
