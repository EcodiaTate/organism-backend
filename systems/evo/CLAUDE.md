# Evo — System CLAUDE.md

**Role:** Growth drive made computational. The organism's learning organ.
**Spec:** `.claude/EcodiaOS_Spec_07_Evo.md`

---

## What's Implemented

### Wake Mode (≤20ms per episode)
- 4 online pattern detectors: `CooccurrenceDetector`, `SequenceDetector`, `TemporalDetector`, `AffectPatternDetector`
- All detector state in `PatternContext` dataclass (mutable accumulator, not Pydantic)
- Evidence accumulation for active hypotheses (approximate Bayesian comparison, Occam's razor penalty)
- Tournament outcome recording per episode
- Hypothesis generation triggered when pending patterns hit batch size

### Hypothesis Engine
- Full lifecycle: `proposed → testing → supported → integrated | refuted | archived`
- LLM-driven generation (falsifiability required — no `formal_test`, no admission)
- Evidence scoring: strength × (1 − decay × contradiction_ratio) − contradiction penalties
- Age gating (≥24h), evidence threshold (score ≥ 3.0, ≥10 supporting episodes)
- `ExperimentDesign` created on PROPOSED→TESTING; `ExperimentResult` on SUPPORTED/REFUTED (Stream 5 RE training unblocked)
- Kairos Tier-3 invariant handler: pre-validated SUPPORTED hypothesis with evidence_score=5.0, novelty_score=1.0, no age wait
- **LRU eviction at cap**: when `max_active_hypotheses: 50` is reached, `_evict_lowest_fitness()` removes the PROPOSED/TESTING hypothesis with lowest `evidence_score − staleness_penalty` (staleness = days × 0.1, capped at 2.0). Applied to both `generate_hypotheses()` and `register_repair_hypothesis()`

### Thompson Sampling Tournaments
- Beta-Bernoulli A/B testing for competing parameter hypotheses
- 10-trial burn-in, convergence at P(winner) ≥ 0.95 via Monte Carlo (10K samples)
- Max 5 concurrent tournaments; fitness cluster threshold δ ≥ 1.0 triggers creation
- Converged tournaments archived; winner hypothesis promoted

### Belief Lifecycle (Sleep Phases 2.5 / 2.75 / 2.8)
- **Aging (2.5):** Radioisotope decay model per domain (`sentiment` 0.3d → `identity` 365d). Stale at age_factor < 0.5, critical < 0.1
- **Consolidation (2.75):** Precision ≥ 0.85, volatility < 0.2, age ≥ 30d → `:ConsolidatedBelief` (mutable=false). Foundation conflicts escalate to Thymos via `INCIDENT_DETECTED` (1 per cycle)
- **Genetic fixation (2.8):** Confidence ≥ 0.95, volatility < 0.1, age ≥ 30d → `BeliefGenome`. LZ4/zlib compression. Instance must have ≥10K episodes + ≥100 confirmed hypotheses

### GenomeExtractor + EvoGenomeExtractor
- `genetic_memory.py` — legacy `GenomeExtractor` for `BeliefGenome`
- `genome.py` — `EvoGenomeExtractor` implements `GenomeExtractionProtocol` for Mitosis inheritance
  - Payload v1.0.0: 23+ tunable params, top-300 hypotheses, top-200 experiment summaries, Thompson priors (capped 50), active experiment designs (capped 100)
- `GenomeSeeder`: injects inherited beliefs at birth with confidence discount × 0.95; `GenomeInheritanceReport` tracks fidelity

### Parameter Tuner (29 parameters across 4 subsystems + belief half-lives)
- Atune salience heads (7), Nova EFE weights (8), Voxis personality (5), Memory salience (5)
- **6 learnable belief half-life parameters**: `belief.halflife.{sentiment,preference,capability,context,social,policy}` — tunable via Evo hypotheses; ranges mirror domain semantics (sentiment: 0.1–2.0d, policy: 30–365d)
- `sync_halflife_overrides()` on `BeliefAgingScanner` propagates live values to Neo4j beliefs + in-memory registry; called from `_run_consolidation_now()` after Phase 5
- Velocity limits: single delta ≤ 0.03, cumulative ≤ 0.15/cycle
- Applied in evidence_score order; adjustments persisted as `:ParameterAdjustment` nodes
- **Push notification on change**: `ParameterTuner.apply_adjustment()` emits `EVO_PARAMETER_ADJUSTED` via Synapse; wired via `wire_event_bus()` called from `EvoService.wire_event_bus()`

### Self-Model (Phase 6)
- Per-capability success rates from 500-outcome, 30-day window
- Regret computation from `:Counterfactual` nodes: `regret = factual − counterfactual`
- Persisted to `:Self` node; emitted as `EVO_DRIFT_DATA` to Equor

### Consolidation Orchestrator (Sleep Mode, ≤60s)
9 sequential phases: Memory consolidation → Hypothesis review → Belief aging → Belief consolidation → Genetic fixation → Schema induction → Procedure extraction → Parameter optimization → Self-model → Drift feed → Evolution proposals → **Exploration proposals (Phase 8.5)**
- All phases wrapped in try/except; failures don't block subsequent phases
- Triggered every 6h or 10K cycles (whichever first)

### Exploration Hypotheses (Phase 8.5 — Gap Closure 8 Mar 2026)
**Problem solved:** Enable bootstrap of entirely new strategies when existing revenue streams dry up.
**Solution:** Fast-track low-evidence hypotheses (evidence_score 2.0–5.0) through lightweight Simula pipeline.

- **Phase 8.5 logic**: Collects SUPPORTED hypotheses with evidence_score ∈ [2.0, 5.0) and proposed_mutation.type == EXPLORATION
- **Metabolic gating**: No explorations when starvation >= AUSTERITY; budget scales 2–5% of liquid reserves
- **Concurrency limit**: Max 2 concurrent explorations (prevent resource exhaustion)
- **Lightweight Simula pipeline**: VALIDATE → GATE → APPLY → VERIFY → RECORD (skip SIMULATE — no training data)
- **Equor integration**: Full constitutional check still applied; no bypass
- **Outcome feedback**: Success boosts evidence_score by 3.0 (fast-track to full EVOLUTION_PROPOSAL); failure increments attempts counter (max 3)
- **RE training**: Emits RE_TRAINING_EXAMPLE at proposed/success/failed stages (category=`exploration_outcome`)
- **Hypothesis fields added**: `is_exploration`, `exploration_budget_usd`, `exploration_attempts`, `exploration_max_attempts`, `exploration_outcomes`
- **Synapse events**: `EXPLORATION_PROPOSED` (Evo → Simula), `EXPLORATION_OUTCOME` (Simula → Evo)
- **Success criteria**: Hypothesis can now discover novel strategies without requiring full evidence accumulation first

### Schema Induction & Procedure Extraction
- `schema_induction.py` — proposes new entity/relation types to Simula from graph clusters
- `procedure_extractor.py` + `procedure_codifier.py` — mines `PatternContext.sequence_counts`, min 0.75 success rate, max 3 new procedures/cycle
- Procedures codified as `:Procedure` Neo4j nodes for Nova fast-path execution

### Speciation Detection
- Monitors: confirmation rate change >20%, novel domains, foundation conflicts
- `SPECIATION_EVENT` emitted when combined magnitude > 0.2 (rate-limited: max 1/24h)

### Additional Modules (beyond core spec)
- `meta_learning.py` — learning-to-learn infrastructure
- `curiosity.py` — epistemic curiosity drive
- `pressure.py` — learning pressure signals
- `self_modification.py` — self-modification proposals
- `cognitive_niche.py` + `niche_forking.py` — niche identification and divergence (Speciation Phase 1)
- `speciation.py` — speciation event orchestration
- `detectors/synapse_cognitive_stall_detector.py` + `simula_codegen_stall_detector.py` — stall detection for Synapse and Simula

### Gap Closure (7 Mar 2026)
- **ArxivProposalTranslator lazy import** — top-level cross-system import from `service.py` removed. `_handle_new_arxiv_innovation()` instantiates `ArxivProposalTranslator()` lazily inside the function body.
- **`CausalFailureAnalyzer` wired** — `EvoService.initialize()` constructs `CausalFailureAnalyzer` lazily from `systems.simula.coevolution.causal_surgery` with try/except guard. Passes it to `ConsolidationOrchestrator`. Phase 6.5 failure-pattern detection is now live.
- **`wire_oikos(oikos)` method** — runtime Oikos injection (no constructor cross-import). Propagates to `ConsolidationOrchestrator._oikos`.
- **`EVO_BELIEF_CONSOLIDATED` emitted** — at end of Phase 2.75 (`_phase_belief_consolidation`). Payload: `beliefs_consolidated`, `foundation_conflicts`, `consolidation_number`.
- **`EVO_GENOME_EXTRACTED` emitted** — in Phase 2.8 when a genome is produced. Payload: `genome_id`, `candidates_fixed`, `genome_size_bytes`, `generation`.
- **`NICHE_FORK_PROPOSAL` event** — `NicheForkingEngine._dispatch_proposal()` now emits `NICHE_FORK_PROPOSAL` instead of `EVOLUTION_CANDIDATE`. All three events added to `synapse/types.py`.
- **Phase 5 Oikos metabolic gate** — `_phase_parameter_optimisation()` calls `check_metabolic_gate(MetabolicPriority.GROWTH)` before running. Skips with `(0, 0.0)` on denial. Non-fatal when Oikos ref is None.
- **`learning_speedup_pct` implemented** — `GenomeSeeder.seed_from_genome()` computes `seeded_count / total_inherited × 100.0`. Logged with inheritance report.
- **Genome version migration** — `_CURRENT_GENOME_VERSION = 1`, `_migrate_genome_item()` fills missing v1 fields via `setdefault`. `decompress_genome()` applies migration + graceful-skip on corrupt records.
- **`NicheRegistry` metabolic gate** — `set_starvation_level(level: str)` method added. `create_niche_from_species()` blocks when level is `starving/critical/terminal`. `EvoService` propagates starvation level to `NicheRegistry` in its Oikos starvation handler.

---

## Gap Closure (7 Mar 2026 — second pass)

- **`export_belief_genome()`** — `EvoService.export_belief_genome()` added (`service.py`). Returns `primitives.genome_inheritance.BeliefGenome` with top-50 hypotheses (confidence ≥ 0.6), current drive weight snapshot (via Equor), last 10 constitutional drift entries (via Memory), and generation counter. Module-level helpers `_safe_get_drive_scores` + `_safe_fetch_drift_history` appended at end of file. Called by `SpawnChildExecutor` Step 0b at spawn time.

## Gap Closure (7 Mar 2026 — event coverage fix)

- **`GENOME_EXTRACT_REQUEST` emitted from `export_belief_genome()`** — signals genome extraction starting on the Synapse bus. Emitted before the `try:` block so spec_checker can observe it regardless of extraction outcome. Non-fatal: if `_event_bus` is None or emit fails, extraction continues unaffected. Satisfies spec_checker's `genome_extract_request` expected event for the `evo` system.

## Gap Closure — Round 2A (7 Mar 2026)

### Staleness Decay (HYPOTHESIS_STALENESS handler)
- `_on_hypothesis_staleness()` — fully implemented in `service.py`
- Iterates all PROPOSED/TESTING hypotheses in `_hypothesis_engine._active`
- Multiplies `evidence_score *= (1.0 - staleness_rate)` for each
- Archives hypotheses where `evidence_score < 0.05` via `archive_hypothesis(h, reason="staleness_decay")`
- Emits `EVO_HYPOTHESES_STALED` (if any archived) — payload: `decayed_count`, `archived_count`, `archived_ids`
- Always emits `EVO_HYPOTHESIS_REVALIDATED` — VitalityCoordinator uses this to call `on_hypotheses_revalidated()` and reduce cumulative entropy pressure (closes the degradation feedback loop)

## What's Missing

- **Phase 8 callback, not Synapse event** — `_simula_callback` injected at init; proposals dropped silently if Simula is unavailable. No retry/queue.
- **Tournament routing requires Nova awareness** — if Nova dispatches without `tournament_context`, no outcomes are recorded; tournament never converges.
- **`Goal` type from `systems.nova.types`** — lazy import inside `_generate_goal_from_hypothesis()`. Not a blocker (lazy), but should eventually move `Goal` to `primitives/`.

---

## Key Files

| File | Purpose |
|------|---------|
| `service.py` | EvoService — wake mode, consolidation trigger, all wiring |
| `hypothesis.py` | HypothesisEngine — full lifecycle |
| `consolidation.py` | ConsolidationOrchestrator — 8-phase sleep pipeline |
| `detectors.py` | 4 online pattern detectors + PatternContext |
| `tournament.py` | TournamentEngine — Thompson sampling A/B |
| `parameter_tuner.py` | 23-param velocity-limited tuner |
| `belief_halflife.py` | BeliefAgingScanner — radioisotope decay |
| `belief_consolidation.py` | BeliefConsolidationScanner — hardening + conflict detection |
| `genetic_memory.py` | GenomeExtractor / GenomeSeeder (legacy path) |
| `genome.py` | EvoGenomeExtractor — Mitosis GenomeExtractionProtocol |
| `self_model.py` | SelfModelManager — capability + regret scoring |
| `procedure_extractor.py` | ProcedureExtractor from PatternContext.sequence_counts |
| `procedure_codifier.py` | ProcedureCodifier — Neo4j :Procedure nodes |
| `schema_induction.py` | SchemaInduction proposals to Simula |
| `speciation.py` | Speciation event detection and emission |
| `types.py` | All Evo types |

---

## Integration Surface

### Events Consumed
| Event | Source | Handler |
|-------|--------|---------|
| *(workspace broadcast)* | Synapse | `receive_broadcast()` — every cycle; `episode`, `affect_state`, `memory.entities` |
| `KAIROS_TIER3_INVARIANT_DISCOVERED` | Kairos | Pre-validated SUPPORTED hypothesis, no age wait |
| `BOUNTY_PAID` | Oikos | `_on_bounty_paid()` — confirmed revenue → high-valence episode + bounty_outcomes window (SG5) |
| `ASSET_BREAK_EVEN` | Oikos | `_on_asset_break_even()` — asset ROI confirmed → positive episode for asset-type hypothesis (SG5) |
| `CHILD_INDEPENDENT` | Oikos | `_on_child_independent()` — successful reproduction → highest-valence episode for reproduction strategy (SG5) |
| `METABOLIC_EFFICIENCY_PRESSURE` | Oikos | `_on_metabolic_efficiency_pressure()` — records negative-valence economic episode + appends TEMPORAL PatternCandidate to `_pending_candidates` (hypothesis_domain: yield_strategy \| budget_allocation \| niche_selection) for next consolidation's hypothesis generation pass |
| `RE_DECISION_OUTCOME` | Nova | `_on_re_decision_outcome()` — tracks consecutive degraded readings (`_re_degradation_count`); after 10 consecutive readings with `success_rate < 0.60`, queues a PatternCandidate targeting `reasoning_engine.hyperparameter_adjustment` for the next consolidation pass |

### Events Emitted
| Event | Trigger | Consumer |
|-------|---------|---------|
| `EVO_DRIFT_DATA` | Phase 7 | Equor |
| `EVO_HYPOTHESIS_CREATED` | Generation | RE training pipeline |
| `EVO_HYPOTHESIS_CONFIRMED` | SUPPORTED | RE training pipeline (Stream 5) |
| `EVO_HYPOTHESIS_REFUTED` | REFUTED | RE training pipeline |
| `EVO_CONSOLIDATION_COMPLETE` | End of sleep | Benchmarks, monitoring |
| `EVO_CAPABILITY_EMERGED` | Structural hypothesis integrated | Telos |
| `EVO_PARAMETER_ADJUSTED` | `ParameterTuner.apply_adjustment()` | Atune, Nova, Voxis |
| `EVO_BELIEF_CONSOLIDATED` | End of Phase 2.75 | Benchmarks, Thread, monitoring |
| `EVO_GENOME_EXTRACTED` | Phase 2.8 genome produced | Mitosis, Benchmarks |
| `NICHE_FORK_PROPOSAL` | Phase 2.95 fork dispatch | Simula (organogenesis) |
| `SPECIATION_EVENT` | Magnitude > 0.2 | Telos, Alive, Benchmarks |
| `INCIDENT_DETECTED` | Foundation conflicts (Phase 2.75) | Thymos |
| `FITNESS_OBSERVABLE_BATCH` | After consolidation | Benchmarks |
| `RE_TRAINING_EXAMPLE` | Hypothesis created/confirmed/refuted | RE training pipeline |
| *(evolution proposals via callback)* | Phase 8 | Simula (injected callback, not Synapse) |

---

## Constraints

- **Wake mode ≤20ms** — pattern detection + evidence accumulation; no Neo4j reads on hot path
- **Consolidation ≤60s** — 8 phases sequential; all wrapped in try/except
- **Cannot modify:** Equor evaluation, constitutional drives, invariants, or self-evaluation criteria
- **Velocity limits are hard:** single delta ≤ 0.03, cumulative ≤ 0.15/cycle — no exceptions
- **No cross-system imports** — all via Synapse events or injected service refs
