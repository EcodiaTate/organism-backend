# Oneiros - Sleep as Batch Compiler

**Specs**: `.claude/EcodiaOS_Spec_13_Oneiros.md` (v1, circadian/dream architecture), `.claude/EcodiaOS_Spec_14_Oneiros.md` (v2, batch compiler framing - primary)
**SystemID**: `oneiros`

## What Oneiros Does

Oneiros is the organism's offline compilation engine. While the wake-state cognitive cycle operates as an interpreter (incremental, single-domain, real-time), Oneiros runs in an offline mode that enables three structurally incompatible operations: cross-domain pattern finding (holding multiple domains in working memory simultaneously), global causal graph reconstruction (seeing all evidence at once to resolve contradictions and transitive chains), and constructive simulation (exploring edges of the world model via dream hypothesis stress-testing). The intelligence ratio improvement per sleep cycle compounds: better predictions → smaller deltas → faster compression → more cycles → higher fitness.

Sleep is not recovery. It is a chosen architectural mode in which the deepest compression occurs.

## Architecture

```
OneirosService (orchestrator)
├── SleepScheduler          - 3 independent triggers: scheduled, cognitive_pressure (≥0.85), compression_backlog
├── CircadianClock           - sleep pressure tracking (4-component formula), DROWSY/forced-sleep thresholds
├── SleepStageController     - v1 state machine (WAKE/HYPNAGOGIA/NREM/REM/LUCID/HYPNOPOMPIA); drives is_sleeping
├── SleepCycleEngine         - v2 executor (DESCENT→SLOW_WAVE→REM→EMERGENCE); drives actual work
│   ├── DescentStage         - checkpoint capture, tag uncompressed episodes in Neo4j (24h window)
│   ├── SlowWaveStage        - Memory Ladder (4 rungs) + causal graph reconstruction + SynapticDownscaler + BeliefCompressor
│   ├── REMStage             - CrossDomainSynthesizer + dream cycles + AffectProcessor + EthicalDigestion
│   └── EmergenceStage       - pre-attention cache, genome update, sleep narrative for Thread, wake broadcast
├── LucidDreamingStage       - mutation simulation (Simula proposals via SimulaProtocol pull pattern)
├── DreamJournal             - Neo4j: Dream / DreamInsight / SleepCycle nodes
└── SleepDebtSystem          - WakeDegradation multipliers applied to Atune/Nova/Evo/Voxis (real, not simulated)
```

**Dual-spec coexistence**: v2 engine (SleepCycleEngine) drives execution. v1 controller (SleepStageController) drives the `is_sleeping` state machine and v1 event names (SLEEP_ONSET, WAKE_ONSET, etc.). Both `SleepStage` and `SleepStageV2` enums coexist in `types.py`.

## Key Files

| File | Role |
|------|------|
| `service.py` | Orchestrator - lifecycle, Synapse wiring, emergency wake, pressure tracking |
| `engine.py` | SleepCycleEngine - DESCENT→SLOW_WAVE→REM→EMERGENCE execution, interrupt/checkpoint |
| `scheduler.py` | SleepScheduler - 3-trigger logic, `can_sleep_now()` guard |
| `circadian.py` | CircadianClock - pressure formula, stage controller (v1), DROWSY/critical thresholds |
| `slow_wave.py` | MemoryLadder (4 rungs), CausalGraphReconstructor, SynapticDownscaler, BeliefCompressor |
| `rem_stage.py` | CrossDomainSynthesizer, dream cycles, AffectProcessor, EthicalDigestion |
| `emergence.py` | EmergenceStage - pre-attention cache, OrganGenomeSegment, sleep narrative, WAKE_INITIATED, FOVEA_PREATTENTION_CACHE_READY |
| `descent.py` | DescentStage - checkpoint, `_tag_uncompressed_episodes()` Neo4j tagging |
| `lucid_stage.py` | LucidDreamingStage - mutation simulation via shadow world model fork |
| `journal.py` | DreamJournal - Neo4j Dream/DreamInsight/SleepCycle persistence |
| `types.py` | All types: SleepStage (v1+v2), SleepPressure, Dream, DreamInsight, SleepCycle, WakeDegradation, config |

## What's Implemented (as of 8 March 2026)

### Fully Operational
- **SleepCycleEngine**: DESCENT→SLOW_WAVE→REM→EMERGENCE pipeline runs end-to-end
- **SleepScheduler**: all 3 triggers (scheduled, cognitive pressure ≥0.85, compression backlog)
- **Sleep pressure**: 4-component formula (cycles 40%, affect 25%, episodes 20%, hypotheses 15%); polling-based via `CircadianClock.tick()` on each theta cycle
- **Emergency wake**: subscribes to `SYSTEM_FAILED` and `SAFE_MODE_ENTERED`; calls `_stage_controller.emergency_wake()`, **captures checkpoint via `_v2_engine.interrupt()`** (fixed 8 Mar), then cancels sleep task
- **SynapticDownscaler**: Neo4j batch 0.85× salience decay on episodes not accessed 7+ days (protects consolidation_level ≥ 3)
- **BeliefCompressor**: queries Nova active beliefs, identifies low-confidence (<0.3) and redundant beliefs per domain, proposes consolidation via Synapse
- **AffectProcessor**: Neo4j batch dampens affect_arousal 20% for episodes with arousal > 0.7, updates Soma coherence_stress
- **EthicalDigestion**: queries DEFERRED/ESCALATE Equor verdicts, proposes fast-path heuristics, emits RE Stream 3 (constitutional_deliberation) training examples
- **DescentStage memory tagging**: `_tag_uncompressed_episodes()` Neo4j Cypher tags 24h episodes with `uncompressed: true`
- **EmergenceStage genome**: `_prepare_genome_update()` extracts schemas/invariants/causal links/improvement history into `OrganGenomeSegment` (SHA256 hash), emits `ONEIROS_GENOME_READY` for Mitosis
- **ONEIROS_SLEEP_CYCLE_SUMMARY**: emitted after wake onset with full cycle metrics for Benchmarks
- **RE training**: Stream 1 (consolidation reasoning) from MemoryLadder; Stream 3 (constitutional deliberation) from EthicalDigestion
- **LucidDreamingStage**: Simula mutations queued via `SimulaProtocol.get_pending_mutations()`, skips gracefully if none pending
- **MetaCognition** (`lucid_stage.py`): Runs every lucid stage - clusters recurring Dream themes by Jaccard similarity over 30-day window, promotes high-frequency clusters (≥3 dreams) to `(:CONCEPT {is_core_identity: true})` Neo4j nodes. No LLM. Results in `LucidDreamingReport.concepts_discovered`. **Neo4j now correctly wired** (fixed 8 Mar 2026).
- **DirectedExploration** (`lucid_stage.py`): Takes `creative_goal` (from `OneirosService._creative_goal`, now passed through) and high-coherence DreamInsights (coherence ≥ 0.85). Applies 4 operators - domain transfer, negation, amplification, constraint - and stores each variation as a `DreamInsight` node (status=PENDING). Results in `LucidDreamingReport.variations_generated`. **Neo4j now correctly wired** (fixed 8 Mar 2026).
- **ThreatSimulator** (`rem_stage.py`): Seeds from Thymos incidents + Evo concerning hypotheses + Nova high-uncertainty beliefs (3 independent Neo4j queries). Synthesises up to 15 threat scenarios, derives heuristic response plans, stores as `(:Procedure)` nodes, emits `ONEIROS_THREAT_SCENARIO` for Thymos prophylactic antibody generation. No LLM. Runs as step 6 of `REMStage.execute()`.
- **WorldModelAuditor** (`slow_wave.py`): Three-pass consistency audit (Spec 14 §3.3.4): orphaned schema detection + soft-prune, causal cycle detection via Neo4j path query + weakest-link removal, deprecated hypothesis retirement. Results in `SlowWaveReport.consistency` (`WorldModelConsistencyReport`). Runs as step 4 of `SlowWaveStage.execute()`.
- **Architecture clean**: no direct Oikos import (duck-typed via `oikos.get_dream_worker()`), no private Evo access (uses `get_active_hypothesis_count()` public API)
- **Pre-attention cache → Fovea delivery** (**Gap 2 CLOSED**, 8 Mar 2026): `EmergenceStage._broadcast_wake_initiated()` now emits `FOVEA_PREATTENTION_CACHE_READY` (new `SynapseEventType`) with the full serialized `entries` list when `total_predictions > 0`. `WAKE_INITIATED` retains `pre_attention_cache_size` (int) for lightweight consumers. Fovea must add a handler.
- **Pre/post-sleep performance measurement** (8 Mar 2026): `SleepCycleEngine` now captures KPI baseline before DESCENT and emits `ONEIROS_SLEEP_OUTCOME` after a 100-cycle (~15s) stabilisation window post-Emergence. See details below.

### Memory Ladder (Slow Wave)
4 rungs, must climb in order - cannot skip:
1. **Episodic → Semantic**: cluster episodes by pattern, extract SemanticNode (LLM), reduce episode salience 30%, mark INTEGRATED
2. **Semantic → Schema**: find shared structure across semantic nodes, create schema with delta references (5:1 compression target)
3. **Schema → Procedure**: extract action-outcome schemas into reusable procedure templates for Nova
4. **Procedure → World Model**: integrate invariant causal procedures as generative rules (deepest compression)

Episodes that cannot climb a rung are marked as **anchor memories** (irreducibly novel) or decay-flagged (low MDL) - never deleted.

## Neo4j Wiring (Critical Path)

Neo4j flows to `LucidDreamingStage` through three layers:

```
registry._init_oneiros()
  └── OneirosService.__init__(neo4j=infra.neo4j)
        └── SleepCycleEngine.set_neo4j(neo4j)   ← forwarded in constructor
              └── LucidDreamingStage(neo4j=neo4j)
                    ├── MetaCognition(neo4j=neo4j)
                    └── DirectedExploration(neo4j=neo4j)

registry._init_oneiros() also calls:
  └── oneiros.set_neo4j(infra.neo4j)            ← explicit setter (idempotent, visible)
```

`set_neo4j()` is available for late injection and re-wiring.

## Sleep Performance Measurement (8 Mar 2026)

### Architecture
`SleepCycleEngine` tracks whether each sleep cycle actually improved cognition.

**Step 1 - Pre-sleep baseline** (`_capture_pre_sleep_baseline()`):
Called at the start of `run_sleep_cycle()`, before any stage. Queries BenchmarkService for 5 KPIs:
`coherence_composite`, `hypothesis_avg_confidence`, `schema_count`, `re_success_rate`, `memory_fragmentation`.
Stored in `_pre_sleep_baseline: dict[str, float]`. Requires `set_benchmarks(benchmarks)` to be wired.
If Benchmarks is unavailable the method returns an empty dict (non-fatal - no outcome is emitted).

**Step 2 - Post-sleep comparison** (`_compute_and_emit_sleep_outcome()`):
Fire-and-forget coroutine launched via `asyncio.ensure_future` after `_scheduler.record_sleep_completed()`.
Waits 15 seconds (~100 theta cycles) for metrics to stabilise, then queries the same KPIs.
Per-KPI delta = `(post − pre) / |pre|`; `memory_fragmentation` delta is inverted (lower = better).

**`SleepOutcome` model** (`oneiros/types.py`):
```python
class SleepOutcome(EOSBaseModel):
    sleep_cycle_id: str
    sleep_duration_ms: int
    stages_completed: list[str]
    kpi_deltas: dict[str, float]       # positive = improvement
    net_improvement: float             # mean of positive deltas
    net_degradation: float             # |mean of negative deltas|
    verdict: str                       # "beneficial" | "neutral" | "harmful"
    pressure_threshold_adjusted: bool
    new_pressure_threshold: float
```

**Verdict rules**:
- `beneficial`: net_improvement > 2% AND net_improvement ≥ net_degradation
- `harmful`: net_degradation > 2% AND net_degradation > net_improvement
- `neutral`: otherwise

**Step 3 - Emit**:
- Always emits `ONEIROS_SLEEP_OUTCOME` with `SleepOutcome.model_dump()` payload.
- If `verdict == "harmful"`: also emits `LEARNING_PRESSURE` (source=oneiros, reason=harmful_sleep_outcome).

**Step 4 - Adaptive threshold** (`_adapt_threshold()`):
Maintains a ring buffer of last 5 outcomes (`_outcome_history: deque[str]`).
- 2+ consecutive "harmful" → `cognitive_pressure_threshold += 0.05` (sleep less often)
- 3+ consecutive "beneficial" with net_improvement > 10% → `threshold -= 0.05` (sleep more often)
- Bounds: `[0.75, 0.95]`. Adjusted in-place on `self._config` and `self._scheduler._config`.

**Wiring**:
- `OneirosService.set_benchmarks(benchmarks)` - forwards to `SleepCycleEngine.set_benchmarks()`. Called from `registry.py` Phase 11 late-wiring block (alongside `soma.set_benchmarks()`), after Benchmarks is fully initialised. If Benchmarks is unavailable at startup the engine silently skips outcome measurement (non-fatal).
- `_stages_completed` list is populated in `_transition_to()` so the outcome always knows which stages ran.

**Evo integration** (`evo/service.py`):
- Subscribes to `ONEIROS_SLEEP_OUTCOME` in `register_on_synapse()`.
- "harmful": queues a `PatternType.TEMPORAL` `PatternCandidate` (domain=`oneiros.sleep_parameters`, confidence up to 0.80) for next consolidation.
- "beneficial" with net_improvement > 5%: queues a `PatternType.COOCCURRENCE` `PatternCandidate` (confidence up to 0.90) reinforcing the current sleep schedule.

## Not Yet Implemented

| Gap | Description |
|-----|-------------|
| **PC algorithm correctness** | `CausalGraphReconstructor._run_pc_algorithm()` is a correlation-asymmetry heuristic - no d-separation, no FCI for latent confounds |
| **CrossDomainSynthesizer scaling** | Pairwise schema comparison across domains will scale quadratically; mitigated by MAX_SCHEMAS_PER_DOMAIN=20, MAX_DOMAIN_PAIRS=45 caps |
| **Federation sleep coordination** | `FEDERATION_SLEEP_SYNC` is subscribed; no peer-coordination logic beyond timing self-sleep |
| **Logos atomic graph APIs** | `find_contradictions()` and `replace_causal_structure()` not yet called - reconstructor uses finer-grained `revise_link()` / `remove_weak_links()` |
| **Fovea `load_preattention_cache()`** | Oneiros now emits `FOVEA_PREATTENTION_CACHE_READY` - Fovea must add a subscription handler and implement the cache load |

## Integration Points

### Emits
- `SLEEP_ONSET` / `SLEEP_INITIATED` - entering sleep (pressure, cycle_id, trigger)
- `ORGANISM_SLEEP` - **NEW (9 Mar 2026)** organism-wide sleep signal emitted immediately after `SLEEP_INITIATED`; Axon, Identity, SACM, Simula all subscribe to this (not to SLEEP_INITIATED)
- `SLEEP_STAGE_CHANGED` / `SLEEP_STAGE_TRANSITION` - between stages (from/to, elapsed_s, stage_report)
- `COMPRESSION_BACKLOG_PROCESSED` - end of Slow Wave (MemoryLadderReport)
- `CAUSAL_GRAPH_RECONSTRUCTED` - end of causal reconstruction
- `CROSS_DOMAIN_MATCH_FOUND` - during REM (CrossDomainMatch)
- `ANALOGY_DISCOVERED` - during REM (Analogy)
- `DREAM_HYPOTHESES_GENERATED` - during REM dream cycles
- `DREAM_INSIGHT` - REM DreamGenerator coherence ≥ 0.70
- `WAKE_ONSET` / `WAKE_INITIATED` - sleep ends (cycle_id, quality, insights_count, intelligence_improvement)
- `ORGANISM_WAKE` - **NEW (9 Mar 2026)** organism-wide wake signal emitted immediately after `WAKE_INITIATED`; Axon, SACM, Simula subscribe to this (not to WAKE_INITIATED)
- `FOVEA_PREATTENTION_CACHE_READY` - **NEW** full PreAttentionCache for Fovea (entries, domains_covered, total_predictions, sleep_cycle_id)
- `SLEEP_PRESSURE_WARNING` - pressure > 0.70
- `SLEEP_FORCED` - critical threshold (0.95) auto-sleep
- `EMERGENCY_WAKE` - Thymos/system critical interrupted sleep
- `LUCID_DREAM_RESULT` - mutation simulation result
- `ONEIROS_GENOME_READY` - OrganGenomeSegment ready for Mitosis
- `ONEIROS_SLEEP_CYCLE_SUMMARY` - full cycle metrics for Benchmarks
- `ONEIROS_ECONOMIC_INSIGHT` - ruin_probability > 0.2 economic dream result
- `ONEIROS_THREAT_SCENARIO` - ThreatSimulator scenario for Thymos
- `ONEIROS_CONSOLIDATION_COMPLETE` - Federation sleep certification (sleep_certified, certified_invariant_ids)
- `ONEIROS_SLEEP_OUTCOME` - **NEW** post-sleep KPI delta verdict (SleepOutcome payload), emitted ~15s after Emergence
- `RE_TRAINING_BATCH` - fire-and-forget training examples (Streams 1 + 3)
- `INTELLIGENCE_IMPROVEMENT_DECLINING` - signal to Telos Growth for new domain exposure

### Consumes
- `SYSTEM_FAILED`, `SAFE_MODE_ENTERED` - emergency wake
- `THETA_CYCLE_COMPLETE` - increment cycles_since_sleep
- `MUTATION_PROPOSAL_READY` (Simula) - queue for next LucidDreamingStage
- `FEDERATION_ALERT` - emergency wake evaluation
- `KAIROS_TIER3_INVARIANT_DISCOVERED` - queue as priority REM seed (Loop 5)
- `GRID_METABOLISM_CHANGED` - GREEN_SURPLUS triggers opportunistic sleep; CONSERVATION cancels
- `EVOLUTION_APPLIED` - increment consolidation pressure (structural change episode)
- `METABOLIC_PRESSURE` - starvation gates: EMERGENCY/CRITICAL block; AUSTERITY halves frequency
- `BOUNTY_PAID`, `REVENUE_INJECTED`, `ASSET_BREAK_EVEN`, `CHILD_INDEPENDENT` - set economic creative_goal
- `FEDERATION_SLEEP_SYNC` - coordinate sleep timing across federation peers
- `SYSTEM_MODULATION` - Skia modulation halt/resume

### Memory Reads
Episode retrieval (RAW consolidation level, salience, affect), Nova active beliefs, Evo PROPOSED/TESTING hypotheses, Equor DEFERRED/ESCALATE verdicts, Schema/CausalInvariant nodes (REM cross-domain), Fovea error counts (compression backlog trigger)

### Memory Writes
SemanticNode, Dream, DreamInsight, SleepCycle, Procedure, WorldModel generative rules, Analogy nodes, episode salience updates (downscaler), belief archive/merge/flag, hypothesis retire/promote/merge, BeliefGenome (OrganGenomeSegment via Mitosis event), CONCEPT nodes (MetaCognition core identity), DreamInsight variations (DirectedExploration)

## Iron Rules

1. Sleep cannot be permanently disabled - debt accumulates, WakeDegradation multipliers are real
2. Emergency wake always possible (Thymos CRITICAL or SYSTEM_FAILED)
3. Consolidation irreversibility is by design (MDL: once pattern extracted to SemanticNode, retaining full episode salience is redundant)
4. Dream content cannot be fabricated - emerges from real episodes and genuine random activation
5. WakeDegradation multipliers are real, not simulated: salience_noise (+15%), EFE precision loss (-20%), expression flatness (-25%), learning rate reduction (-30%)
6. Sleep duration (22h/2h default) is a chosen parameter, not a biological constant - Evo should tune `intelligence_ratio_improvement_per_sleep_cycle` vs frequency

## Sleep Pressure Formula

```
pressure = 0.40 * (cycles_since_sleep / max_wake_cycles)
         + 0.25 * (unprocessed_affect / affect_capacity)
         + 0.20 * (unconsolidated_episodes / episode_capacity)
         + 0.15 * (hypothesis_backlog / max_hypotheses)

threshold = 0.70 → SLEEP_PRESSURE_WARNING
critical  = 0.95 → SLEEP_FORCED
```

Pressure updated via `CircadianClock.tick()` per theta cycle (polling, not push). `hypothesis_backlog` queried via `Evo.get_active_hypothesis_count()` public API every 100 cycles.

## RE Integration (Current State)

All LLM calls currently target Claude API. Two RE training streams already wired:
- **Stream 1** (consolidation reasoning): from MemoryLadder schema creation during Slow Wave
- **Stream 3** (constitutional deliberation): from EthicalDigestion during REM

Highest-priority RE tasks (not yet wired for RE routing):
- EpisodicReplay pattern extraction - most repetitive structured task, ~100-200 training pairs per sleep cycle
- EthicalDigestion - constitutional edge cases, most alignment-critical RE task
- DreamGenerator bridge narrative - needs 6+ months of dream data for RE to match Claude quality

Thompson sampling between Claude and RE is the correct integration pattern - route to RE when posterior > 0.75, fall back to Claude otherwise.

## Known Issues

1. Spec 13 and Spec 14 use different Synapse event name strings for equivalent concepts (e.g. `SLEEP_ONSET` vs `SLEEP_INITIATED`). Both are emitted; canonical set should be reconciled.
2. `SleepStageV2` and `SleepStage` enums coexist - v2 drives execution, v1 drives `is_sleeping` state; cleanup deferred.
3. `LogosEngine.find_contradictions()` and `replace_causal_structure()` not yet used - causal reconstruction uses finer-grained `revise_link()` / `remove_weak_links()` calls instead of atomic graph replacement.
4. Fovea does not yet subscribe to `FOVEA_PREATTENTION_CACHE_READY` - Oneiros now emits it (8 Mar 2026) but Fovea consumer is a Fovea-side gap.
