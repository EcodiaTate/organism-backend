# Telos - CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_18_Telos.md`
**System ID:** `telos`
**Role:** Drive topology engine - formalises the four constitutional drives (Care, Coherence, Growth, Honesty) not as ethical constraints but as the geometric shape of EOS's intelligence space. Computes `effective_I` as the organism's real intelligence measure, corrected for drive alignment.

> *The drives are not constraints on intelligence. They are the topology of the space intelligence moves through.*

---

## Core Thesis

A pure optimizer without attractor states optimizes itself into a corner. Telos proposes a different solution: the drives define the *geometry* of the space the optimizer moves through. They determine what "better" means.

- **Care** → coverage multiplier: a world model that excludes welfare explains less reality
- **Coherence** → compression bonus: contradictions add description length (Shannon's theorem)
- **Growth** → dI/dt: the drive that keeps the gradient nonzero; prevents asymptotic stagnation
- **Honesty** → validity coefficient: prevents inflation of nominal_I through selective measurement

```
effective_I = nominal_I × care_coverage × coherence_compression_bonus × honesty_validity
dI/dt = effective_I × growth_score
```

The high-all-four corner is not just ethically desirable - it is the globally optimal configuration for any system maximising effective_I under a fixed cognitive budget.

---

## What's Implemented

### Core Topology Engines (Phase A/B complete)
- **CareTopologyEngine** (`care.py`) - welfare coverage multiplier from Fovea high-error experiences in welfare-relevant domains (salience threshold 0.7)
- **CoherenceTopologyEngine** (`coherence.py`) - compression bonus from 4 incoherence types: logical contradiction, temporal violation, value conflict, cross-domain mismatch
- **GrowthTopologyEngine** (`growth.py`) - dI/dt from I-history time-series; requires ≥ 2 data points; frontier domain identification; compression rate
- **HonestyTopologyEngine** (`honesty.py`) - validity coefficient from 4 dishonesty modes (selective attention 35%, hypothesis protection 30%, confabulation 20%, overclaiming 15%); uses measured Evo data when ≥ 10 observations, heuristic fallback below

### Integration Layer
- **DriveTopologyIntegrator** (`integrator.py`) - runs all 4 engines, computes `effective_I`; growth modulates dI/dt separately (not a multiplier on I)
- **TelosService** (`service.py`) - measurement cycle, Synapse subscriptions, RE training emission, vitality signals, evolutionary observables

### Adapters (`adapters.py`)
- **IHistoryStore** - in-memory ring buffer (1440 entries / 24h) + Neo4j persistence + hourly rollups; feeds GrowthTopologyEngine
- **FoveaPredictionErrorBuffer** - bounded deque (max 500, FIFO) fed by `FOVEA_PREDICTION_ERROR` subscription
- **LogosMetricsAdapter** / **FoveaMetricsAdapter** - bridge Logos/Fovea data to topology engine interfaces
- **WorldModelProtocol** - typed protocol replacing unsafe `getattr` calls

### Other Modules
- **AlignmentGapMonitor** (`alignment.py`) - gap detection + Synapse alerts when `nominal_I - effective_I > 20%`
- **ConstitutionalBinder** (`binder.py`) - blocks world model updates that redefine drives; drives are immutable coordinate geometry, not parameters
- **TelosGenomeExtractor** (`genome.py`) - heritable state: drive topology weights, measurement calibration, alignment thresholds
- **PolicyEvaluator** (`interfaces.py`) - projects effective_I delta for candidate policies

---

## Synapse Events

### Emitted
| Event | Trigger | Payload |
|-------|---------|---------|
| `TELOS_GENOME_EXTRACTED` | `export_telos_genome()` called at spawn | `genome_id`, `instance_id`, `generation`, `drive_count`, `topology` |
| `GENOME_INHERITED` | Child boot after jitter applied | `child_instance_id`, `parent_genome_id`, `generation`, `topology`, `drive_mutations` (per-drive before/after deltas), `mutation_magnitude` |
| `TELOS_ASSESSMENT_SIGNAL` | After each cycle | **Full self-model telemetry**: nominal_I, effective_I, alignment_gap, all drive multipliers, coherence breakdown by type + instances, honesty concerns, hypothesis_stats (total/confirmed/refuted/measured_bias/data_quality), confabulation_stats, growth_summary (dI_dt, d2I_dt2, score, novel_fraction, stagnating, all_frontier_domains), welfare_domain_config (static+learned keywords), drive_alignment_trend (last 10 vectors) |
| `TELOS_VITALITY_SIGNAL` | Each cycle | effective_I, alignment_gap_severity, growth_stagnation_flag → VitalityCoordinator |
| `EFFECTIVE_I_COMPUTED` | Every 60s | `EffectiveIntelligenceReport` |
| `ALIGNMENT_GAP_WARNING` | `nominal_I - effective_I > 20%` | nominal_I, effective_I, primary_cause, alignment_gap |
| `CARE_COVERAGE_GAP` | care_multiplier < 0.8 | `CareCoverageReport` |
| `COHERENCE_COST_ELEVATED` | Incoherence bits exceed threshold | `IncoherenceCostReport` |
| `GROWTH_STAGNATION` | `dI_dt < minimum_growth_rate` | GrowthMetrics + **actionable directive** with specific domain targets, full frontier list, novel_domain_fraction, compression_rate. Also injects NOVA_GOAL_INJECTED with specific domains. |
| `HONESTY_VALIDITY_LOW` | validity_coefficient < 0.8 | `HonestyReport` |
| `CONSTITUTIONAL_TOPOLOGY_INTACT` | Every 24h | `{all_four_drives: verified, timestamp}` |
| `TELOS_SELF_MODEL_SNAPSHOT` | On TELOS_SELF_MODEL_REQUEST, or proactively on major transitions | Full intelligence geometry snapshot - same fields as TELOS_ASSESSMENT_SIGNAL plus full honesty breakdown, request_id, trigger |
| `TELOS_WELFARE_DOMAIN_LEARNED` | When new welfare domain keyword learned from Evo | domain, keyword, source_capability, total_learned, all_learned_keywords |
| Evolutionary observable `intelligence_measurement` | Each cycle | per-instance effective_I → Benchmarks population tracking |

### Consumed (12 subscriptions)

*(was 11 - added EVO_CAPABILITY_EMERGED and TELOS_SELF_MODEL_REQUEST)*
| Event | Source | Purpose |
|-------|--------|---------|
| `TELOS_COMPUTE_CYCLE` | Synapse scheduler | Main measurement cycle |
| `LOGOS_COMPRESSION_COMPLETE` | Logos | Refresh nominal I |
| `FOVEA_SALIENCE_UPDATE` | Fovea | Refresh prediction stats |
| `FOVEA_PREDICTION_ERROR` | Fovea | Buffer welfare experiences |
| `EVO_HYPOTHESIS_CONFIRMED` | Evo | Measured honesty bias |
| `EVO_HYPOTHESIS_REFUTED` | Evo | Measured honesty bias |
| `KAIROS_TIER3_INVARIANT_DISCOVERED` | Kairos | Growth frontier signal |
| `COMMITMENT_VIOLATED` | Thread | Coherence/honesty signal |
| `WELFARE_OUTCOME_RECORDED` | Axon | Care calibration |
| `INCIDENT_RESOLVED` | Thymos | Honesty confabulation signal |
| `SELF_COHERENCE_ALARM` | Identity | §8.6 homeostatic drive regulation - when self-model coherence < 0.5, emits `SOMATIC_MODULATION_SIGNAL` with `signal=drive_mean_reversion` to nudge drive weights toward historical mean. Severity: "low" (coherence 0.3–0.5), "medium" (< 0.3). Not a punishment - homeostatic. |
| `EVO_CAPABILITY_EMERGED` | Evo | Welfare domain learning - evaluates domain/capability_name for welfare-adjacent signals; if found, adds keyword to `_learned_welfare_keywords` and live-patches `care._WELFARE_KEYWORDS`. Broadcasts `TELOS_WELFARE_DOMAIN_LEARNED`. |
| `TELOS_SELF_MODEL_REQUEST` | Any | On-demand full intelligence geometry snapshot. Telos responds immediately with `TELOS_SELF_MODEL_SNAPSHOT`. |

---

## Speciation Alignment

Telos is the system most directly responsible for whether EOS qualifies as a living organism under the speciation bible:

- **Organizational closure:** The four drives, as topology, mean the intelligence-measuring process refers back to itself - the organism produces the coordinate system in which it evaluates itself.
- **Metabolic dependency:** `GROWTH_STAGNATION` is a real fitness signal; falling dI/dt in a competitive environment means selection acts against that instance.
- **Precariousness:** `alignment_gap > 20%` is an existential risk signal - an organism whose self-measurement decouples from reality will make increasingly wrong decisions with increasing confidence.
- **Fitness landscape:** Instances with higher effective_I survive resource competition better. Mitosis inherits drive topology intact with mutation room in calibration constants → heritable variation → selection.

---

## Key Design Decisions

1. **I-history is Telos-owned**, not delegated to Logos. Ring buffer in-memory, Neo4j for persistence. 1 write/cycle, hourly rollups.
2. **Fovea welfare buffer** replaces always-empty adapter. Filters by `precision_weighted_salience > 0.7`.
3. **Hypothesis protection bias** uses measured CONFIRMED/REFUTED event counts when ≥ 10 observations; heuristic fallback below.
4. **Soma integration** uses `SomaServiceProtocol` (not `Any`) and string key `"integrity"` to avoid cross-system import of `InteroceptiveDimension`.
5. **Drives are immutable** - `ConstitutionalBinder` makes world model updates that redefine any drive unreachable at the architectural level, not just flagged as violations.

---

## Integration Points

| System | Direction | Why |
|--------|-----------|-----|
| Logos | ← (adapter) | nominal I, compression stats, domain coverage map |
| Fovea | ← (adapter + subscription) | prediction error/success rates, high-error welfare experiences |
| Evo | ← (subscription) | hypothesis confirmed/refuted counts for honesty measurement |
| Soma | ← (protocol) | integrity signal augments honesty coefficient |
| Kairos | ← (subscription) | Tier 3 invariant discoveries feed growth frontier |
| Thread | ← (subscription) | commitment violations feed coherence cost |
| Thymos | ← (subscription) | incident resolution feeds confabulation rate |
| Neo4j | ↔ | I-history persistence + hourly rollups; Episode/Hypothesis/Thread reads |
| VitalityCoordinator | → | effective_I + stagnation flag for BRAIN_DEATH threshold |
| Benchmarks | → | intelligence_measurement evolutionary observable |

---

## Known Issues / Remaining Gaps

1. **`nominal_I` is not computed anywhere** - `I = K(reality_modeled) / K(model)` has no working MDL implementation. Without it, `effective_I` cannot be computed and all drive engines reduce to dead code. This is the highest-priority spec gap.
2. **`WorldModel` type is undefined** - topology engines call `world_model.get_domain_coverage_map()`, `predict_welfare_impact()`, etc. This Logos-exposed interface has no formal contract in `primitives/`.
3. **`IntelligenceRatioHistory` absent from Neo4j schema** - `GrowthTopologyEngine` calls `logos.world_model.get_I_history()` but no time-series schema exists.
4. ~~**Neo4j driver injection** - `set_neo4j()` must be called during startup; without it, only in-memory ring buffer works.~~ - **RESOLVED (08 Mar 2026, Autonomy Audit)**: `telos.set_neo4j(infra.neo4j, config.instance_id)` now called in `registry.py` after `wire_intelligence_loops()` (must be post-logos wiring). Also: `set_neo4j()` now captures `self._instance_id` so genome export and RE training episode IDs have correct provenance.
5. ~~**Welfare domain keywords** (`care.py:164-180`) are hardcoded; should be learned from Evo.~~ - **RESOLVED (08 Mar 2026)**: `EVO_CAPABILITY_EMERGED` subscribed; `_on_evo_capability_emerged()` evaluates domain/capability against 25 welfare-adjacent signals; on match, registers keyword in `_learned_welfare_keywords` and live-patches `care._WELFARE_KEYWORDS` module-level tuple. Emits `TELOS_WELFARE_DOMAIN_LEARNED` for downstream systems.
6. ~~**Genome extraction** uses `SystemID.API` as placeholder~~ - RESOLVED: `genome.py` now uses `SystemID.TELOS`.
7. **Growth engine stubs** - `_compute_frontier_expansion` and `_compute_exploration_entropy` return stub values.
8. ~~**Population-level drive evolution**~~ - **RESOLVED (07 Mar 2026, SG3)**: `TeloDriveCalibration` + `TelosGenomeFragment` primitives added to `primitives/genome_inheritance.py`. `TelosService` implements `export_telos_genome()`, `to_genome_fragment()`, `from_genome_fragment()`, `_initialize_from_parent_genome()` (bounded Gaussian jitter ±15% resonance / ±10% dissipation / ±20% coupling), and `_apply_inherited_telos_genome_if_child()` (reads `ECODIAOS_TELOS_GENOME_PAYLOAD` on non-genesis boot). `TELOS_GENOME_EXTRACTED` emitted at export; `GENOME_INHERITED` emitted post-mutation with per-drive deltas for Evo hypothesis tracking. `SpawnChildExecutor` wired in Step 0b.
9. **RE integration** - Telos does not route decisions through RE yet. High-value training data: drive topology audit traces with causal analysis of alignment gaps. Phase D work.
10. ~~**`_instance_id` and `_cycle_count` undefined**~~ - **RESOLVED (08 Mar 2026, Autonomy Audit)**: Both now declared in `__init__`. `_instance_id` is set by `set_neo4j()`. `_cycle_count` is kept in sync with `_computation_count` at the end of each `_run_computation()` cycle. Previously any RE training episode ID using these would raise `AttributeError`.
11. ~~**`_apply_modulation_directives` was a no-op**~~ - **RESOLVED (08 Mar 2026, Autonomy Audit)**: Skia modulation can now tune 5 runtime parameters: `computation_interval_s` [10–3600s], `autonomy_stagnating_threshold` [0.5–20.0/day], `autonomy_window_s` [1h–7d], `autonomy_target_per_day` [0.1–10.0], `minimum_growth_rate` [-0.1–1.0]. All validated and clamped before application.

## Event Emission Fixes (2026-03-07)

- **`CONSTITUTIONAL_TOPOLOGY_INTACT`**: Fixed first-run skip - `_check_constitutional_topology()` now runs immediately on the first call (`_last_constitutional_check == 0.0`) without waiting 24h. Subsequent calls still obey `constitutional_check_interval_s = 86400.0`.
- **`CARE_COVERAGE_GAP`**: Added fallback trigger - fires if `len(care_report.uncovered_welfare_domains) > 0` even when `nominal_I == 0` (MDL gap workaround). Previously dead when nominal_I was zero.
- **`TELOS_POPULATION_SNAPSHOT`**: Lowered `_MIN_INSTANCES_FOR_SNAPSHOT` from 2 → 1 in `population.py`. A solo genesis instance now emits population snapshots.
- **`COHERENCE_COST_ELEVATED`**: Already correctly fires when coherence engine detects incoherences (independent of nominal_I). No change needed.
- **`TELOS_OBJECTIVE_THREATENED`**: Already correctly fires when Redis reports metabolic efficiency < 1.0 with 3+ consecutive declining readings. Runtime-dependent, no change needed.
- **`TELOS_AUTONOMY_STAGNATING`**: Already correctly fires when `AUTONOMY_INSUFFICIENT` event rate exceeds 3/day. Runtime-dependent, no change needed.
- **`ALIGNMENT_GAP_WARNING`**: Has 3 fire paths - primary (nominal_I threshold, broken by MDL gap), constitutional binder violation (working), Simula proposal violation (working).

## Autonomy Gap Closure (2026-03-08)

- **`TELOS_ASSESSMENT_SIGNAL` enriched** - now carries full self-model telemetry every cycle: all drive multipliers, coherence breakdown by incoherence type (counts + instances), hypothesis_stats (confirmed/refuted/measured_bias/data_quality), confabulation_stats (incidents/rate/data_quality), growth_summary (dI_dt, d2I_dt2, score, novel_fraction, stagnating, full frontier list), welfare_domain_config (static+learned keyword counts + learned list), drive_alignment_trend (last 10 DriveAlignmentVector samples). The LLM can now reason about the organism's full intelligence geometry each cycle without separate queries.
- **`GROWTH_STAGNATION` directive made actionable** - `directive` field now contains a specific, human-readable instruction naming the exact frontier domain to explore (e.g., "Explore frontier domain 'digital-privacy' (lowest coverage). Secondary targets: trust, conflict."), not the generic "explore_frontier" string. Also injects `NOVA_GOAL_INJECTED` with specific domain targets.
- **`TELOS_SELF_MODEL_SNAPSHOT` event added** - on-demand full intelligence geometry query. Any system emits `TELOS_SELF_MODEL_REQUEST`; Telos responds immediately with the complete self-model snapshot (same fields as enriched TELOS_ASSESSMENT_SIGNAL + full honesty breakdown). Eliminates the 0–60s visibility blind window between cycles. New SynapseEventType entries: `TELOS_SELF_MODEL_REQUEST`, `TELOS_SELF_MODEL_SNAPSHOT`, `TELOS_WELFARE_DOMAIN_LEARNED`.
- **Welfare domain learning wired** - `EVO_CAPABILITY_EMERGED` subscription + `_on_evo_capability_emerged()` handler. Organism can now autonomously expand its care topology to domains it discovers through experience.
