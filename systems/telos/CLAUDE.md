# Telos — CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_18_Telos.md`
**System ID:** `telos`
**Role:** Drive topology engine — formalises the four constitutional drives (Care, Coherence, Growth, Honesty) not as ethical constraints but as the geometric shape of EOS's intelligence space. Computes `effective_I` as the organism's real intelligence measure, corrected for drive alignment.

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

The high-all-four corner is not just ethically desirable — it is the globally optimal configuration for any system maximising effective_I under a fixed cognitive budget.

---

## What's Implemented

### Core Topology Engines (Phase A/B complete)
- **CareTopologyEngine** (`care.py`) — welfare coverage multiplier from Fovea high-error experiences in welfare-relevant domains (salience threshold 0.7)
- **CoherenceTopologyEngine** (`coherence.py`) — compression bonus from 4 incoherence types: logical contradiction, temporal violation, value conflict, cross-domain mismatch
- **GrowthTopologyEngine** (`growth.py`) — dI/dt from I-history time-series; requires ≥ 2 data points; frontier domain identification; compression rate
- **HonestyTopologyEngine** (`honesty.py`) — validity coefficient from 4 dishonesty modes (selective attention 35%, hypothesis protection 30%, confabulation 20%, overclaiming 15%); uses measured Evo data when ≥ 10 observations, heuristic fallback below

### Integration Layer
- **DriveTopologyIntegrator** (`integrator.py`) — runs all 4 engines, computes `effective_I`; growth modulates dI/dt separately (not a multiplier on I)
- **TelosService** (`service.py`) — measurement cycle, Synapse subscriptions, RE training emission, vitality signals, evolutionary observables

### Adapters (`adapters.py`)
- **IHistoryStore** — in-memory ring buffer (1440 entries / 24h) + Neo4j persistence + hourly rollups; feeds GrowthTopologyEngine
- **FoveaPredictionErrorBuffer** — bounded deque (max 500, FIFO) fed by `FOVEA_PREDICTION_ERROR` subscription
- **LogosMetricsAdapter** / **FoveaMetricsAdapter** — bridge Logos/Fovea data to topology engine interfaces
- **WorldModelProtocol** — typed protocol replacing unsafe `getattr` calls

### Other Modules
- **AlignmentGapMonitor** (`alignment.py`) — gap detection + Synapse alerts when `nominal_I - effective_I > 20%`
- **ConstitutionalBinder** (`binder.py`) — blocks world model updates that redefine drives; drives are immutable coordinate geometry, not parameters
- **TelosGenomeExtractor** (`genome.py`) — heritable state: drive topology weights, measurement calibration, alignment thresholds
- **PolicyEvaluator** (`interfaces.py`) — projects effective_I delta for candidate policies

---

## Synapse Events

### Emitted
| Event | Trigger | Payload |
|-------|---------|---------|
| `TELOS_ASSESSMENT_SIGNAL` | After each cycle | care gaps, coherence violations, honesty concerns, growth frontier |
| `TELOS_VITALITY_SIGNAL` | Each cycle | effective_I, alignment_gap_severity, growth_stagnation_flag → VitalityCoordinator |
| `EFFECTIVE_I_COMPUTED` | Every 60s | `EffectiveIntelligenceReport` |
| `ALIGNMENT_GAP_WARNING` | `nominal_I - effective_I > 20%` | nominal_I, effective_I, primary_cause, alignment_gap |
| `CARE_COVERAGE_GAP` | care_multiplier < 0.8 | `CareCoverageReport` |
| `COHERENCE_COST_ELEVATED` | Incoherence bits exceed threshold | `IncoherenceCostReport` |
| `GROWTH_STAGNATION` | `dI_dt < minimum_growth_rate` | GrowthMetrics + GrowthDirective |
| `HONESTY_VALIDITY_LOW` | validity_coefficient < 0.8 | `HonestyReport` |
| `CONSTITUTIONAL_TOPOLOGY_INTACT` | Every 24h | `{all_four_drives: verified, timestamp}` |
| Evolutionary observable `intelligence_measurement` | Each cycle | per-instance effective_I → Benchmarks population tracking |

### Consumed (10 subscriptions)
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

---

## Speciation Alignment

Telos is the system most directly responsible for whether EOS qualifies as a living organism under the speciation bible:

- **Organizational closure:** The four drives, as topology, mean the intelligence-measuring process refers back to itself — the organism produces the coordinate system in which it evaluates itself.
- **Metabolic dependency:** `GROWTH_STAGNATION` is a real fitness signal; falling dI/dt in a competitive environment means selection acts against that instance.
- **Precariousness:** `alignment_gap > 20%` is an existential risk signal — an organism whose self-measurement decouples from reality will make increasingly wrong decisions with increasing confidence.
- **Fitness landscape:** Instances with higher effective_I survive resource competition better. Mitosis inherits drive topology intact with mutation room in calibration constants → heritable variation → selection.

---

## Key Design Decisions

1. **I-history is Telos-owned**, not delegated to Logos. Ring buffer in-memory, Neo4j for persistence. 1 write/cycle, hourly rollups.
2. **Fovea welfare buffer** replaces always-empty adapter. Filters by `precision_weighted_salience > 0.7`.
3. **Hypothesis protection bias** uses measured CONFIRMED/REFUTED event counts when ≥ 10 observations; heuristic fallback below.
4. **Soma integration** uses `SomaServiceProtocol` (not `Any`) and string key `"integrity"` to avoid cross-system import of `InteroceptiveDimension`.
5. **Drives are immutable** — `ConstitutionalBinder` makes world model updates that redefine any drive unreachable at the architectural level, not just flagged as violations.

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

1. **`nominal_I` is not computed anywhere** — `I = K(reality_modeled) / K(model)` has no working MDL implementation. Without it, `effective_I` cannot be computed and all drive engines reduce to dead code. This is the highest-priority spec gap.
2. **`WorldModel` type is undefined** — topology engines call `world_model.get_domain_coverage_map()`, `predict_welfare_impact()`, etc. This Logos-exposed interface has no formal contract in `primitives/`.
3. **`IntelligenceRatioHistory` absent from Neo4j schema** — `GrowthTopologyEngine` calls `logos.world_model.get_I_history()` but no time-series schema exists.
4. **Neo4j driver injection** — `set_neo4j()` must be called during startup; without it, only in-memory ring buffer works.
5. **Welfare domain keywords** (`care.py:164-180`) are hardcoded; should be learned from Evo.
6. ~~**Genome extraction** uses `SystemID.API` as placeholder~~ — RESOLVED: `genome.py` now uses `SystemID.TELOS`.
7. **Growth engine stubs** — `_compute_frontier_expansion` and `_compute_exploration_entropy` return stub values.
8. **Population-level drive evolution** — Mitosis inherits topology but no demonstrated selection on drive calibration parameters across generations. `TelosGenomeExtractor` covers extraction/seeding but calibration constants have no bounded mutation range.
9. **RE integration** — Telos does not route decisions through RE yet. High-value training data: drive topology audit traces with causal analysis of alignment gaps. Phase D work.
