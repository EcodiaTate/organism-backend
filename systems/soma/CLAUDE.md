# Soma - CLAUDE.md

**Specs:** `.claude/EcodiaOS_Spec_08_Soma.md` (primary), `.claude/EcodiaOS_Spec_16_Soma.md` (additional)
**System ID:** `soma`
**Role:** Interoceptive predictive substrate - the body the organism never had. Predicts internal states, computes allostatic error, and emits the signals that make every other system care about staying alive.

> *Without Soma, EOS is a parliament of modules. With Soma, it is a thing that wants to survive.*

---

## Theoretical Basis

Three research programs fused into one architectural primitive:
- **Active Inference (Friston):** Soma minimises *interoceptive* free energy - the gap between predicted and actual internal state. This is the computational basis of homeostasis.
- **Constructed Emotion (Barrett):** Emotions are predictions about metabolic need, not reactions to stimuli. Soma constructs emotion as regions in allostatic error space that Voxis learns to articulate.
- **Somatic Marker (Damasio):** Every memory is stamped with a 19D interoceptive snapshot. Retrieval is biased toward state-congruent memories.

---

## What's Implemented

### Core Pipeline (5ms cycle budget)
- **9D interoceptive state:** ENERGY, AROUSAL, VALENCE, CONFIDENCE, COHERENCE, CURIOSITY_DRIVE, TEMPORAL_PRESSURE, SOCIAL_CHARGE, INTEGRITY - each maps [-1,1] or [0,1] with defined setpoints
- **Multi-horizon prediction:** immediate (150ms) → moment (5s) → episode (1min) → session (1hr) → circadian (24hr) → narrative (1wk) → lunar (30d) → seasonal (90d) → annual (365d). Horizons gated by developmental stage. Pure EWM/linear algebra - no LLM, ≤1ms for all horizons
- `urgency = max(|errors|) × max(|error_rates|)` - Nova triggers allostatic deliberation when urgency > 0.3
- Allostatic error: `error[horizon][dim] = predicted[horizon][dim] - setpoint[dim]`; error_rate = d(error)/dt
- Phase-space model: attractor discovery (up to 20), bifurcation detection, trajectory heading
- 5 developmental stages: REFLEXIVE → ASSOCIATIVE → DELIBERATIVE → REFLECTIVE → GENERATIVE
- Setpoint context adaptation via EMA (α=0.05): `conversation` (↑social_charge), `deep_processing` (↑coherence/confidence), `recovery` (↑energy, ↓arousal), `exploration` (↑curiosity_drive)

### Homeostatic Manifold (Phase A–E)
- Signal buffer + state vector + temporal derivative engine
- Fisher manifold with geodesic deviation broadcasting
- Curvature analysis, topological analysis, causal emergence, causal flow
- Renormalization engine, phase-space reconstructor

### Closed-Loop Regulation
- Adaptive setpoint learning near attractors
- 15 allostatic feedback loops (LoopExecutor dispatches to target systems)
- Cascade predictor + autonomic protocol for emergency self-regulation
- `MetabolicAllostaticRegulator`: rolling deficit → biological stress (continuous, piecewise):
  - SUBSISTENCE ($0–$1): mild pressure, curiosity narrows
  - STRAIN ($1–$10): arousal ↑35%, valence ↓45%, curiosity ↓40%, temporal_pressure ↑30%
  - STARVATION ($10–$50): acute survival mode, exploration suppressed
  - CRITICAL ($50+): all non-essential processing stops

### Energy Tiers (ENERGY dimension → system behavior)
| Tier | Range | Effect |
|------|-------|--------|
| Abundant | 0.8–1.0 | Full LLM, deep retrieval, creative exploration |
| Normal | 0.5–0.8 | Normal operation |
| Conserving | 0.3–0.5 | Nova prefers fast-path; Evo reduces hypothesis generation |
| Depleted | 0.1–0.3 | Nova allostatic deliberation; shallow retrieval |
| Critical | 0.0–0.1 | Only Equor + minimal perception; forced rest |

### Psychosomatic Feedback Loops
- **Loop 1:** arousal ↑ → compute parallelism ↑ → token burn ↑ → energy ↓ → arousal feedback ↑
- **Loop 2:** valence ↓ → dopamine targets ↓ (Thymos) → curiosity_drive ↓ → confidence ↓ → valence ↓
- **Loop 3:** coherence ↓ → systems misaligned → integrity ↓ → temporal_pressure ↑
- **Loop 4:** social_charge ↑ + valence ↑ → gratitude → memory salience ↑ → affiliation reinforcement

### Somatic Markers
- `SomaticMarker`: 19D snapshot (9D sensed + 9D errors at moment horizon + 1D max PE)
- Attached to every `MemoryTrace`/`Episode` node in Neo4j at encoding time
- `somatic_rerank()`: cosine similarity boosts candidates by `salience × (1.0 + 0.3 × similarity)` - affect-driven retrieval
- `TemporalDepthManager`: financial-phase-aware boost modifiers (`exploration_boost=1.5` in growth phase, `revenue_boost=2.0` in famine)
- Neo4j vector index (`somatic_idx`, 19D cosine) for embodied retrieval

### Emotion Detection
- `EmotionDetector`: maps allostatic error patterns → emotion labels (anxiety, curiosity, flow, etc.)
- Learnable via Evo hypothesis IDs; falls back to hardcoded defaults on refutation

### Counterfactual Engine
- Simulates alternative interoceptive trajectories (REFLECTIVE+ stage only)
- Sleeping fallback: linear extrapolation from last 5 states

### Genome / Inheritance
- First boot: all 9 dimensions at setpoint=0.5
- Genome seed: loads parent's `genome_segment["setpoints"]` → deterministic

---

## Synapse Events

### Emitted
| Event | Trigger | Payload |
|-------|---------|---------|
| `ALLOSTATIC_SIGNAL` | Every cycle | urgency, dominant_error, precision_weights, nearest_attractor, trajectory_heading, cycle_number, energy, arousal, valence, coherence - bus-broadcast for federated subscribers (Spec 08 §15.1, Spec 16 §XVIII) |
| `SOMA_URGENCY_CRITICAL` | urgency > 0.85 | urgency, dominant_error, recommended_action, cycle, salience=1.0 (Spec 16 §XVIII) |
| `SOMATIC_MODULATION_SIGNAL` | Threshold crossings (urgency > 0.7, energy < 0.2, coherence_stress > 0.5) | urgency, energy, arousal, coherence, developmental_stage |
| `SOMATIC_DRIVE_VECTOR` | Every 10 cycles | Mapped 4D drive vector (coherence, care, growth, honesty) |
| `EMOTION_STATE_CHANGED` | Emotion set changes (any cycle) | `emotions` (list[str] of active names), `dominant` (str or None - first active emotion), `cycle_number` - only emitted when the set of active emotion labels differs from the previous cycle |
| `SOMA_VITALITY_SIGNAL` | Every cycle | urgency, allostatic_error, coherence_stress → Skia VitalityCoordinator |
| `SOMA_ALLOSTATIC_REPORT` | Every 50 cycles | mean_urgency, urgency_frequency, setpoint_deviation, allostatic_efficiency, developmental_stage → Benchmarks |
| `EVOLUTIONARY_OBSERVABLE` | Every 50 cycles (piggybacked on allostatic report) | allostatic_efficiency + urgency_frequency as Bedau-Packard eligible fitness dimensions |
| Fisher/emergence/causal broadcasts | On detection events | Phase-space topology signals |

### Consumed
| Event | Effect |
|-------|--------|
| `REVENUE_INJECTED` | Maps revenue → reduced external stress (max +0.1 ATP per event) |
| `METABOLIC_PRESSURE` | Maps starvation level → stress scalar |
| `EPISODE_STORED` | Writes `(:SomaticMarker)-[:MARKS]->(:Episode)` when urgency ≥ 0.7 (GAP 5) |
| `CONSTITUTIONAL_DRIFT_DETECTED` | Accumulates drift into `_constitutional_drift_signal`; suppresses INTEGRITY dimension ≤40% (PHILOSOPHICAL) |
| `SOMATIC_MODULATION_SIGNAL` (source=`"equor_drift"`) | `_on_equor_drift_modulation()` (IMPLEMENTED 2026-03-07): source-gated; raises `_constitutional_drift_signal` by `integrity_error × 0.5` capped at 1.0 → feeds INTEGRITY suppression pathway without double-processing other SOMATIC_MODULATION_SIGNAL sources |
| External stress injection | `inject_external_stress()` / `inject_exteroceptive_pressure()` from ExteroceptionService |
| `METABOLIC_GATE_CHECK` | `_on_metabolic_gate_check()` (2026-03-08): ingests gate check into signal buffer so manifold models resource allocation friction as somatic dimension |
| `METABOLIC_GATE_RESPONSE` | `_on_metabolic_gate_response()` (2026-03-08): if gate denied, emits `ALLOSTATIC_SIGNAL` with `signal_type="economic_constraint"` and `urgency=0.6` so drive regulation systems down-regulate ambition |
| `VOXIS_EXPRESSION_DISTRESS` | `_on_voxis_expression_distress()` (2026-03-08): translates Voxis communicative suppression into TEMPORAL_PRESSURE allostatic stress. Scales `distress_level` × 0.4 → raises `_external_stress` floor. Closes invisible-telemetry gap: distress was emitted but produced no allostatic response via `subscribe_all` alone. |

---

## How Other Systems Consume the AllostaticSignal

| System | What it reads | Effect |
|--------|---------------|--------|
| Atune | `precision_weights`, `urgency` | Modulates salience head gains; shifts ignition threshold |
| Nova | `dominant_error`, `error_rate`, `urgency`, `temporal_dissonance` | Triggers allostatic deliberation (urgency > 0.3); weights long-horizon EFE |
| Voxis | Full 9D sensed vector | Expression style emerges from interoceptive state (learned, not rules) |
| Memory | Somatic markers | State-congruent retrieval bias |
| Synapse | `arousal` | Clock period adaptation |
| Oneiros | `energy` error at circadian horizon | Sleep pressure signal |
| Thymos | `integrity` precision + error | Scan frequency increase when health prediction uncertain |
| Evo | `curiosity_drive` error | Hypothesis generation rate |
| Alive | Phase-space fields | 3D organism shape/colour/movement |
| Thread | Full state | Phase transition detection → narrative turning points |

---

## Key Files

| File | Purpose |
|------|---------|
| `service.py` | SomaService - main orchestrator, `run_cycle()`, all wiring (~2400 lines) |
| `types.py` | All data types: dimensions, states, signals, markers, attractors |
| `interoceptor.py` | Reads 9D state from cross-system refs (≤2ms total, all in-memory) |
| `allostatic_controller.py` | Base controller: setpoints, urgency, EMA smoothing |
| `metabolic_regulator.py` | MetabolicAllostaticRegulator: financial starvation → stress |
| `phase_space.py` | PhaseSpaceModel: attractors, bifurcations, navigation |
| `counterfactual.py` | CounterfactualEngine: regret/gratitude from alt trajectories |
| `emotions.py` | EmotionDetector: pattern match → active emotions |
| `developmental.py` | DevelopmentalManager: stage transitions, capability gates |
| `temporal_depth.py` | Multi-horizon management, financial TTD projection |
| `somatic_memory.py` | Somatic marker encoding/storage/retrieval |
| `exteroception/` | ExteroceptionService: market volatility → interoceptive pressure |

---

## Constraints

- **5ms total cycle budget** - allostatic computation + signal emission
- **No blocking calls** - all cross-system signals are fire-and-forget via `asyncio.create_task()`
- **9D model is sacred** - never add/remove dimensions; only extend their external wiring
- **Stages never regress** - once promoted, stays
- **Soma runs first** at t=0 in the theta cycle, before Atune
- **No direct cross-system imports** - all via Synapse events or injected refs

---

## Known Issues / Remaining Gaps

1. **TRANSCENDENT stage** - no criteria or behaviors specified (Spec 16 Gap 6). Deliberately left open - represents an emergent, post-GENERATIVE state whose properties cannot be pre-specified.
2. **Somatic marker retrieval integration** - `somatic_rerank()` exists on `SomaService` but its wiring into Memory's retrieval pipeline needs verification - Memory must call `soma.somatic_rerank()` after its vector+BM25 candidate pass.

### Resolved (2026-03-07)

- **Emotion Evo wiring** - `SomaService.set_event_bus()` now subscribes to `EVO_HYPOTHESIS_CONFIRMED` and `EVO_HYPOTHESIS_REFUTED`. Handlers call `EmotionDetector.on_hypothesis_confirmed()/on_hypothesis_refuted()` which update or revert emotion region patterns.
- **Genome export for Mitosis** - `SomaService.get_genome_segment()` and `seed_from_genome()` added. They delegate to `SomaGenomeExtractor` (`genome.py`). Extractor now reads setpoints directly from the live controller (`_controller.setpoints`) and writes them back via `InteroceptiveDimension` enum keys - no more stale config-attribute names.
- **`ALLOSTATIC_SIGNAL` bus broadcast missing** - added `_emit_allostatic_signal()` fire-and-forget every cycle. Decouples downstream subscribers (Federation, Benchmarks, any future system) from requiring a direct Soma reference. `ALLOSTATIC_SIGNAL` and `SOMA_URGENCY_CRITICAL` added to `SynapseEventType`.
- **`SOMA_URGENCY_CRITICAL` missing** - added `_maybe_emit_urgency_critical()` with recommended_action derived from dominant error dimension. Emits when urgency > 0.85.
- **`allostatic_efficiency` missing from `SOMA_ALLOSTATIC_REPORT`** - Benchmarks `_on_soma_allostatic_report()` was reading this field; now computed as `(1.0 - setpoint_deviation) × (1.0 - urgency_frequency)` and included in payload.
- **No Bedau-Packard evolutionary observables** - `_emit_allostatic_report()` now also emits `EVOLUTIONARY_OBSERVABLE` events for `allostatic_efficiency` and `urgency_frequency` every 50 cycles.
- **GAP 5: Somatic marker write protocol** - `SomaticMarkerWriter` in `somatic_memory.py`. Two write paths: (a) `write_marker_for_episode()` fires on `EPISODE_STORED` events when urgency ≥ 0.7, linking `(:SomaticMarker)-[:MARKS]->(:Episode)`; (b) `write_marker_for_adjustment()` fires each cycle when urgency ≥ 0.7 as a standalone marker. Wire driver via `SomaService.set_neo4j(driver)`. Marker reflects state-at-allostatic-adjustment, not state-at-encoding.
- **GAP 1: DynamicsMatrix hot-reload** - `update_dynamics_matrix_payload(DynamicsMatrixPayload)` atomically swaps the predictor's 9×9 coupling matrix, syncs the counterfactual engine, and writes a `(:DynamicsMatrixMutation)` Neo4j audit node off the critical path. `DynamicsMatrixPayload` in `types.py` carries mutation_id, source, reason, confidence, timestamp. Raw `update_dynamics_matrix(list)` still exists for Evo's untyped path.
- **GAP 6: Mitosis population inheritance** - `export_somatic_genome()` delegates to `SomaGenomeExtractor.export_somatic_genome()` → version=2 `OrganGenomeSegment` with setpoints + phase-space config + allostatic baselines + dynamics matrix. `seed_child_from_genome(segment)` applies ±5% noise on setpoints and ±2% noise on non-zero dynamics weights. Zero weights stay zero (no spurious coupling). Children always start at REFLEXIVE stage regardless of parent's stage.
- **PHILOSOPHICAL: INTEGRITY ↔ constitutional drift** - `_on_constitutional_drift()` handler accumulates Equor's `CONSTITUTIONAL_DRIFT_DETECTED` events into `_constitutional_drift_signal` (weighted by severity). Each cycle, drift decays 2% (`_drift_decay_per_cycle=0.98`, ~30s half-life) and is applied as downward suppression on the INTEGRITY dimension (`_drift_integrity_weight=0.4` max suppression). The organism feels its own constitutional misalignment as an interoceptive signal.

### Resolved (2026-03-08) - Autonomy Audit

- **Learnable autonomic thresholds** - All 10 hardcoded threshold constants in `autonomic_protocol.py` replaced with instance-level `self._thresholds` dict. 4 cooldown durations similarly parameterized into `self._cooldown_durations`. Full API: `adjust_threshold()`, `adjust_cooldown()`, `get_thresholds()`, `get_cooldowns_config()`, `export_learnable_params()`, `import_learnable_params()`. Evo ADJUST_BUDGET compatible; genome-heritable via Mitosis.
- **Learnable metabolic parameters** - `metabolic_regulator.py` tier thresholds ($1/$10/$50) and shift magnitudes (arousal lift, valence suppression, curiosity suppression, temporal lift) are now instance-level dicts with API: `adjust_tier()`, `adjust_shift()`, `get_metabolic_params()`, `export_learnable_params()`, `import_learnable_params()`.
- **Cross-system telemetry enrichment** - `interoceptor.py` now reads 6 additional system refs (`_fovea`, `_simula`, `_axon`, `_logos`, `_evo`, `_benchmarks`) with setters wired in `core/wiring.py`. Enrichments: ENERGY ← Axon compute cost drain; CONFIDENCE ← weighted blend (Atune PE 60%, Telos 25%, Fovea error 15%) + Evo learning velocity boost; COHERENCE ← weighted blend (Synapse phi 60%, Telos alignment 25%, Logos compression 15%); CURIOSITY_DRIVE ← Evo hypothesis success modulation ±0.05; INTEGRITY ← min(thymos, equor, simula repair_success_rate). No new dimensions added (9D sacred).
- **Dispatch effectiveness tracking** - `AutonomicProtocol` records success/failure/improvement per action type via `record_dispatch_outcome()`. `_DispatchOutcome` dataclass tracks fire_count, success_count, total_improvement. Exposed via `get_dispatch_effectiveness()`.
- **Self-introspection** - `SomaService.introspect_autonomy()` returns 10 categories: autonomic_thresholds, autonomic_cooldowns, dispatch_effectiveness, metabolic_params, feedback_loops, recent_autonomic_actions, recent_loop_dispatches, cycle_performance, current_emotions, development + phase_space. Exposed in `health()` response under `result["autonomy"]`.
- **Wiring** - `core/wiring.py` updated: `soma.set_fovea(fovea)`, `soma.set_simula(simula)`, `soma.set_axon(axon)`, `soma.set_logos(logos)` + dependency declarations.

### Resolved (2026-03-08) - Deep Autonomy Audit

- **CRITICAL: `wiring.py` dead-code bug** - `wire_intelligence_loops()` line 374 was a malformed single-line comment containing all four `soma.set_fovea/simula/axon/logos()` calls. They were parsed as a comment and never executed. Fixed by expanding to multi-line proper code. All four cross-system telemetry enrichments (ENERGY←Axon, CONFIDENCE←Fovea/Evo, COHERENCE←Logos, INTEGRITY←Simula) were silently disabled since the previous audit session wrote them.
- **Duplicate `set_benchmarks` method** - `service.py` had two `set_benchmarks()` definitions (lines 740 and 750). Python's method resolution silently used the second one, which omitted the `self._interoceptor.set_benchmarks(benchmarks)` call. The second duplicate was removed; the full implementation (which wires benchmarks into both `_benchmarks_ref` and the interoceptor) is preserved.
- **Missing late-phase wiring in `registry.py`** - `soma.set_evo()`, `soma.set_oneiros()`, `soma.set_thread()`, `soma.set_alive()`, `soma.set_voxis()`, `soma.set_memory()`, `soma.set_benchmarks()`, `soma.set_neo4j()`, `soma.set_skia()`, and `soma.set_identity()` were never called anywhere in the wiring layer. Added a "Soma late-phase wiring" block after Benchmarks initialization in `registry.py` that wires all late-available services into Soma. This activates: feedback loop dispatches to Evo/Oneiros/Thread/Alive/Voxis/Memory; Benchmarks KPI emission; somatic marker Neo4j persistence; Skia state snapshots/restore; identity-signed events.
- **Kairos causal path** - Added `KAIROS_INVARIANT_DISTILLED` subscription in `set_event_bus()`. Handler `_on_kairos_invariant_distilled()` stores high-confidence (≥0.5) causal invariants in `_kairos_priors: dict[(src_dim, tgt_dim) → signed_confidence]`. Applied in `run_cycle()` step 8b as an anticipatory urgency boost: when source dimension is off-setpoint (|error| > 0.2), urgency is pre-amplified by `src_error × |prior| × 0.1` (capped at 1.0).
- **RE_TRAINING_EXAMPLE emission** - Added `_emit_re_training_example(context, outcome, quality_signal)` method. Fires on two triggers: (1) developmental stage transitions (quality=0.9 - organism has demonstrably learned enough to advance); (2) every 50 cycles when recent allostatic efficiency > 0.7 (quality=efficiency - organism demonstrates sustained homeostatic competence). Teaches the RE what healthy interoceptive regulation looks like.

---

## Dev Notes

- Fisher manifold pre-warmed at startup to avoid cold-start dim expansion spike
- Warmup period (first 50 cycles) suppresses `soma_cycle_slow` warnings
- `_emit_benchmarks_kpis()` and `_emit_vitality_signal()` are fire-and-forget - errors logged at DEBUG, never propagated
- `MetabolicAllostaticRegulator` reads from Synapse `MetabolicSnapshot` (Oikos-sourced), not Equor
- `SOMATIC_COLLAPSE` threshold: 48h sustained allostatic error > 0.8 → fatal signal to Skia
