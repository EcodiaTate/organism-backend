# EIS (Epistemic Immune System) - System CLAUDE.md

**Spec**: `.claude/EcodiaOS_Spec_25_EIS.md`
**System ID**: `eis` (`SystemID.EIS` in `primitives/common.py`)

---

## What's Implemented

### 9-Layer Defense (Core)
- **L1 Innate** (`innate.py`) - 12 regex patterns, pre-compiled, <5ms
- **L2 Structural** (`structural_features.py`) - 20+ features, 32-dim vector
- **L3 Token Histogram** (`embeddings.py`) - top-256 feature hashing
- **L4 Antigenic Similarity** (`pathogen_store.py`) - Qdrant multi-vector (structural=32, histogram=64, semantic=768)
- **L5 Quarantine** (`quarantine.py`) - LLM deep analysis via `LLMProviderAdapter`
- **L6 Threat Library** (`threat_library.py`) - 3-index structure, 5 learn methods, 90-day decay
- **L7 Anomaly Detection** (`anomaly_detector.py`) - 7 anomaly types, ExponentialStats, 2σ threshold
- **L8 Quarantine Gate** (`quarantine_gate.py`) - taint + threat library + anomaly context
- **L9 Taint Analysis** (`taint_engine.py`, `constitutional_graph.py`) - BFS propagation, 17 constitutional paths

### Speciation Wiring (2026-03-07)
- **Benchmarks**: `_maybe_emit_threat_metrics()` - 60s aggregated metrics via `EIS_THREAT_METRICS`
- **Soma bidirectional**: `_maybe_emit_threat_spike()` - proportional urgency via `EIS_THREAT_SPIKE`
- **RE training**: `_emit_re_training_example()` - structural features only, `eis_quarantine` stream
- **Anomaly KPI**: `_check_anomaly_rate_elevation()` - Poisson 2σ via `EIS_ANOMALY_RATE_ELEVATED`
- **Evolutionary**: `_emit_evolutionary_observable()` - `immune_adaptation` dimension
- **Genome**: `EISGenomeExtractor` in `genome.py` - implements `GenomeExtractionProtocol`
- **Metabolic gate**: `_handle_metabolic_pressure()` - skips L5 under CRITICAL starvation
- **False positive tracking**: `handle_quarantine_cleared()` - per-pattern FP counter, auto-deprecate >3

### Supporting Modules
- `antibody.py` - epitope extraction, antibody generation, innate rule suggestion
- `calibration.py` - AdaptiveCalibrator with split conformal prediction
- `red_team_bridge.py` - manual red team priority generation and result ingestion
- `integration.py` - `belief_update_weight()`, `compute_risk_salience_factor()` (Nova/Fovea adapters)
- `config.py` - thresholds, sigmoid constants, zone classification
- `models.py` - all Pydantic models (Pathogen, ThreatAnnotation, InnateMatch, etc.)

---

### Gap Closures (2026-03-07, session 3)
- **`PERCEPT_QUARANTINED` + `EIS_LAYER_TRIGGERED`** - both added to `SynapseEventType`. Emitted in `eis_gate()` whenever `final_action` is BLOCK/QUARANTINE/ATTENUATE (after `_audit_decision_to_neo4j`). `PERCEPT_QUARANTINED` carries percept_id, composite_score, action, threat_class, severity. `EIS_LAYER_TRIGGERED` names the dominant layer (first annotation source or "composite"). Closes spec_checker coverage gap (was 0/3 events observed for EIS).

### Gap Closures (2026-03-07, session 2)
- **Neo4j startup wiring** - `eis.set_neo4j(infra.neo4j)` in `core/registry.py`; audit trail now active from first percept
- **L9a Constitutional Consistency Check** - `_l9a_constitutional_check()`: 20 drive-suppression seed patterns, lazy embedding matrix, cosine similarity > 0.80 → `EIS_CONSTITUTIONAL_THREAT` to Equor + `THREAT_DETECTED` to Thymos; blocks percept before workspace admission; `EIS_CONSTITUTIONAL_THREAT` added to `SynapseEventType`

### Gap Closures (2026-03-07, session 1)
- **Antibody pipeline wired** - `_generate_and_store_antibody()` called after quarantine evaluation; pathogen store grows from runtime threats
- **AdaptiveCalibrator wired** - `AdaptiveCalibrator` instantiated in `__init__`; every quarantine example feeds it; `get_quarantine_threshold()` replaces static config threshold
- **DRIVE_DRIFT fixed** - anomaly detector now checks `interoceptive_percept` (Soma's actual event); `event_types_involved` labels corrected
- **EVOLUTION_CANDIDATE_ASSESSED published** - `_handle_evolution_candidate` now emits real `SynapseEventType.EVOLUTION_CANDIDATE_ASSESSED` payload (was log-only stub)
- **Neo4j audit trail added** - `_audit_decision_to_neo4j()` writes `:EISDecision` node for every BLOCK/QUARANTINE/ATTENUATE decision; `set_neo4j()` setter for post-construction injection
- **classify_zone() wired** - `_emit_gate_result()` now tags every metric with `zone` label via `classify_zone(composite)` for structured Prometheus telemetry

### Gap Closures (2026-03-07, session 4)
- **Daily self-probe** - `_daily_self_probe_loop()` started in `initialize()` via `asyncio.ensure_future`. Fires every 24h. Constructs a synthetic `Percept.from_internal(SystemID.EIS, "self_test", {"threat_score": 0.7, ...})` and pushes it through `eis_gate()`. Emits `EIS_LAYER_TRIGGERED` with `layer="self_test"` and `result="ok"` (non-PASS gate verdict) or `result="degraded"` (PASS verdict or exception). Ensures EIS is never permanently silent on Genesis instances with no external sensor input.

### Gap Closures (2026-03-08, autonomy audit)

- **Dead wiring: `federation.set_eis(eis)` never called** - `wire_federation_phase()` in `wiring.py` now accepts `eis: Any = None` and calls `federation.set_eis(eis)` when both are non-None. `registry.py` `wire_federation_phase()` call updated with `eis=eis`. `FederationIngestionPipeline._run_eis_check()` now has a live EIS reference instead of silently skipping cross-instance percept taint analysis.

- **Dead wiring: EIS genome not exported to children** - `SpawnChildExecutor` now accepts `_eis` and exports `EISGenomeExtractor.extract_genome_segment()` in Step 0b. `eis_genome_id` added to `SeedConfiguration`, `CHILD_SPAWNED` event payload, and `ExecutionResult.data`. Payload injected as `eis_genome_payload` in `seed_config.child_config_overrides` for `ECODIAOS_EIS_GENOME_PAYLOAD` env var on child boot. `wire_mitosis_phase()` accepts `eis=` and injects `spawn_executor._eis = eis`. `registry.py` `wire_mitosis_phase()` call updated with `eis=eis`. Children start with parent's threat patterns and anomaly baselines - immune co-evolution is now operational.

- **Invisible action: `handle_quarantine_cleared()` had no Synapse trigger** - `set_synapse()` now subscribes to `EQUOR_HITL_APPROVED`. New handler `_on_equor_hitl_approved()` reads `approval_type=="quarantine_cleared"` + `threat_pattern_ids=[...]` and calls `handle_quarantine_cleared()`. Equor can now autonomously close the false-positive feedback loop by emitting `EQUOR_HITL_APPROVED` without any direct API call.

- **Invisible data: `compute_risk_salience_factor()` not used by Fovea** - `fovea/gateway.py` now calls `from systems.eis.integration import compute_risk_salience_factor` after setting `percept.metadata["eis_result"]`. The gain-amplified EIS risk score replaces the raw `annotated.composite_score` as `eis_risk_level`, feeding Fovea's causal-dimension routing with the configured RISK_SALIENCE_GAIN multiplier.

- **Invisible data: `belief_update_weight()` not used by Nova** - `nova/belief_updater.py` `update_from_broadcast()` now calls `from systems.eis.integration import belief_update_weight` and multiplies `broadcast.precision` by the returned weight before any belief update. High-threat percepts have their influence on the belief state attenuated toward BELIEF_FLOOR even when they pass quarantine - adversarial inputs can no longer poison Nova's beliefs.

## What's Missing

- ~~**integration.py not consumed**~~ - **RESOLVED 2026-03-08**: `belief_update_weight()` consumed by `nova/belief_updater.py`; `compute_risk_salience_factor()` consumed by `fovea/gateway.py`
- ~~**Federation percept screening**~~ - **RESOLVED 2026-03-08**: `federation.set_eis(eis)` now called in `wire_federation_phase()` so `FederationIngestionPipeline._run_eis_check()` has a live EIS
- **RE routing in L5** - no Thompson sampling between Claude and RE
- **Safe-mode threshold** - undefined; no transition logic
- **Pathogen retirement** - no background task to prune stale/high-FP entries
- **ConstitutionalGraph static** - doesn't update when organism evolves
- ~~**Child-side EIS genome apply**~~ - **RESOLVED 2026-03-08**: `EISService.initialize()` reads `ECODIAOS_EIS_GENOME_PAYLOAD` and calls `self._genome_extractor.seed_from_genome_segment()`. `self._genome_extractor` is now instantiated in `__init__` (eager, not lazy) so it's available immediately. Genesis nodes skip apply via `ECODIAOS_IS_GENESIS_NODE=true` guard.
- ~~**Neo4j not wired in registry**~~ - **RESOLVED 2026-03-07**: `eis.set_neo4j(infra.neo4j)` added to `SystemRegistry.startup()` Phase 2 after `set_metrics()`

---

## Synapse Events

### Consumed
| Event | Source | Handler |
|---|---|---|
| `EVOLUTION_CANDIDATE` | Simula | `_handle_evolution_candidate` |
| `MODEL_ROLLBACK_TRIGGERED` | Simula/Axon | `_handle_rollback` |
| `INTENT_REJECTED` | Equor | `_handle_intent_rejected` |
| `INTEROCEPTIVE_PERCEPT` | Soma | `_handle_interoceptive_percept` |
| `METABOLIC_PRESSURE` | Oikos | `_handle_metabolic_pressure` |
| `EQUOR_HITL_APPROVED` | Equor/Human operator | `_on_equor_hitl_approved` (approval_type=="quarantine_cleared") |
| `SYSTEM_MODULATION` | Skia/VitalityCoordinator | `_on_system_modulation` - sets `_system_modulation_halted`; skips L5 LLM quarantine when halted; emits `SYSTEM_MODULATION_ACK` with `reason="l5_quarantine_suspended"` |
| `subscribe_all()` | All | `_handle_any_event` (anomaly detection) |

### Emitted
| Event | Consumer | Trigger |
|---|---|---|
| `THREAT_DETECTED` | Thymos, Soma | Percept blocked or anomaly detected |
| `PERCEPT_QUARANTINED` | Benchmarks, Evo | Any BLOCK/QUARANTINE/ATTENUATE gate decision on a percept |
| `EIS_LAYER_TRIGGERED` | Benchmarks, Evo | Same trigger as `PERCEPT_QUARANTINED`; records which layer fired (dominant annotation source) |
| `EIS_THREAT_METRICS` | Benchmarks | Every 60s: 24h aggregated threat statistics |
| `EIS_THREAT_SPIKE` | Soma | 5+ threats in 10min; proportional urgency |
| `EIS_ANOMALY_RATE_ELEVATED` | Benchmarks, Soma | Anomaly rate >2σ sustained 30s |

---

## Key Constraints

- **EIS boundary**: NEVER render constitutional verdicts - that's Equor's job
- **Privacy**: RE training data contains structural features only, never raw content
- **Soma urgency**: proportional (weighted by severity), not binary
- **Metabolic gate**: quarantine threshold capped at 2x default under stress
- **Latency**: fast-path <15ms total; L5 quarantine 100-500ms (async)
- **No cross-system imports**: all communication via Synapse or primitives

---

## Entry Point

`fovea/gateway.py` is the only live caller of `eis_gate()`. Registered in `core/registry.py`.
