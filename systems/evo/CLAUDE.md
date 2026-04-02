# Evo - System CLAUDE.md

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
- LLM-driven generation (falsifiability required - no `formal_test`, no admission)
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
- `genetic_memory.py` - legacy `GenomeExtractor` for `BeliefGenome`
- `genome.py` - `EvoGenomeExtractor` implements `GenomeExtractionProtocol` for Mitosis inheritance
  - Payload v1.0.0: 23+ tunable params, top-300 hypotheses, top-200 experiment summaries, Thompson priors (capped 50), active experiment designs (capped 100)
- `GenomeSeeder`: injects inherited beliefs at birth with confidence discount × 0.95; `GenomeInheritanceReport` tracks fidelity

### Parameter Tuner (29 parameters across 4 subsystems + belief half-lives)
- Atune salience heads (7), Nova EFE weights (8), Voxis personality (5), Memory salience (5)
- **6 learnable belief half-life parameters**: `belief.halflife.{sentiment,preference,capability,context,social,policy}` - tunable via Evo hypotheses; ranges mirror domain semantics (sentiment: 0.1–2.0d, policy: 30–365d)
- `sync_halflife_overrides()` on `BeliefAgingScanner` propagates live values to Neo4j beliefs + in-memory registry; called from `_run_consolidation_now()` after Phase 5
- Velocity limits: single delta ≤ 0.03, cumulative ≤ 0.15/cycle
- Applied in evidence_score order; adjustments persisted as `:ParameterAdjustment` nodes
- **Push notification on change**: `ParameterTuner.apply_adjustment()` emits `EVO_PARAMETER_ADJUSTED` via Synapse; wired via `wire_event_bus()` called from `EvoService.wire_event_bus()`
- **Feedback loop (8 Mar 2026)**: Every consolidation cycle, `ParameterTuner.tick_evaluation()` compares pending adjustments against current KPIs (geometric mean of per-KPI ratios). Improvement ≥ 5% → confirm + positive hypothesis evidence. Degradation ≥ 5% → auto-revert to `old_value` + `EVO_PARAMETER_REVERTED` event + negative hypothesis evidence. Neutral → extend eval window (max `MAX_EVAL_EXTENSIONS=2`), then confirm.
  - `ParameterAdjustmentRecord` captures `param_path`, `old/new_value`, `cycle_applied`, `baseline_metrics`, `hypothesis_id`
  - KPI baseline sourced from `_last_benchmark_kpis` cache (updated on each `BENCHMARK_REGRESSION` event)
  - `wire_hypothesis_engine()` wires `HypothesisEngine` into tuner for evidence callbacks
  - `_persist_revert()` writes reverted value back to `:EvoParameter` Neo4j node

### Self-Model (Phase 6)
- Per-capability success rates from 500-outcome, 30-day window
- Regret computation from `:Counterfactual` nodes: `regret = factual − counterfactual`
- Persisted to `:Self` node; emitted as `EVO_DRIFT_DATA` to Equor

### Consolidation Orchestrator (Sleep Mode, ≤60s)
9 sequential phases: Memory consolidation → Hypothesis review → Belief aging → Belief consolidation → Genetic fixation → Schema induction → Procedure extraction → Parameter optimization → Self-model → Drift feed → Evolution proposals → **Exploration proposals (Phase 8.5)**
- All phases wrapped in try/except; failures don't block subsequent phases
- Triggered every 6h or 10K cycles (whichever first) - **now adaptive** (see Adaptive Timing below)

### Adaptive Consolidation Timing (8 Mar 2026)
**Problem solved:** Fixed 6h interval consolidates at the same rate regardless of learning velocity, wasting cycles when the organism is learning fast and under-reacting to regressions.

**Solution:** `_learning_pressure: float` accumulator (0.0 – 1.0) on `ConsolidationOrchestrator`:
- `+0.1` per new HIGH-confidence hypothesis (evidence_score ≥ 8.0, i.e. confidence ≥ 0.9)
- `+0.2` per `BENCHMARK_REGRESSION` event
- `+0.15` per `FOVEA_CALIBRATION_ALERT` event
- `+0.3` per `LEARNING_PRESSURE` event (also continues to set early-consolidation flag)
- `−0.05` per 100 cycles (natural cooldown); decay called each 60s loop poll
- Capped at 1.0; **reset to 0.0 after every consolidation**

**Dynamic interval:** `effective_interval_hours = max(1.0, 6.0 × (1.0 − pressure))`
- At pressure=0.0: 6 hours (default, unchanged)
- At pressure=0.5: 3 hours
- At pressure=1.0: 1 hour (minimum - never consolidates more than once/hour)

**Stall detection** uses `_consolidation_expected_interval_s` which is now updated post-consolidation to the current effective interval, so stall alarms stay proportional.

**Pre/post quality snapshot + `EVO_CONSOLIDATION_QUALITY`** - emitted after every consolidation:
- Snapshot keys: `hypothesis_count`, `avg_confidence`, `schema_count`, `re_success_rate`
- `improvement_delta` = `post − pre` per KPI
- Payload: `consolidation_number`, `consolidation_duration_ms`, `hypotheses_promoted`, `hypotheses_pruned`, `improvement_delta`, `pre_snapshot`, `post_snapshot`
- New `SynapseEventType`: `EVO_CONSOLIDATION_QUALITY = "evo_consolidation_quality"`

**Public API on `ConsolidationOrchestrator`:**
- `add_pressure(delta)` - increment pressure (non-fatal)
- `decay_pressure(cycles_elapsed)` - apply cooldown (called by service loop)
- `reset_pressure()` - zero out after consolidation
- `learning_pressure` property - read current scalar
- `_effective_interval_hours()` - computed interval (private, read by service for stall threshold)

### Exploration Hypotheses (Phase 8.5 - Gap Closure 8 Mar 2026)
**Problem solved:** Enable bootstrap of entirely new strategies when existing revenue streams dry up.
**Solution:** Fast-track low-evidence hypotheses (evidence_score 2.0–5.0) through lightweight Simula pipeline.

- **Phase 8.5 logic**: Collects SUPPORTED hypotheses with evidence_score ∈ [2.0, 5.0) and proposed_mutation.type == EXPLORATION
- **Metabolic gating**: No explorations when starvation >= AUSTERITY; budget scales 2–5% of liquid reserves
- **Concurrency limit**: Max 2 concurrent explorations (prevent resource exhaustion)
- **Lightweight Simula pipeline**: VALIDATE → GATE → APPLY → VERIFY → RECORD (skip SIMULATE - no training data)
- **Equor integration**: Full constitutional check still applied; no bypass
- **Outcome feedback**: Success boosts evidence_score by 3.0 (fast-track to full EVOLUTION_PROPOSAL); failure increments attempts counter (max 3)
- **RE training**: Emits RE_TRAINING_EXAMPLE at proposed/success/failed stages (category=`exploration_outcome`)
- **Hypothesis fields added**: `is_exploration`, `exploration_budget_usd`, `exploration_attempts`, `exploration_max_attempts`, `exploration_outcomes`
- **Synapse events**: `EXPLORATION_PROPOSED` (Evo → Simula), `EXPLORATION_OUTCOME` (Simula → Evo)
- **Success criteria**: Hypothesis can now discover novel strategies without requiring full evidence accumulation first

### Schema Induction & Procedure Extraction
- `schema_induction.py` - proposes new entity/relation types to Simula from graph clusters
- `procedure_extractor.py` + `procedure_codifier.py` - mines `PatternContext.sequence_counts`, min 0.75 success rate, max 3 new procedures/cycle
- Procedures codified as `:Procedure` Neo4j nodes for Nova fast-path execution

### Speciation Detection
- Monitors: confirmation rate change >20%, novel domains, foundation conflicts
- `SPECIATION_EVENT` emitted when combined magnitude > 0.2 (rate-limited: max 1/24h)

### Additional Modules (beyond core spec)
- `meta_learning.py` - learning-to-learn infrastructure
- `curiosity.py` - epistemic curiosity drive
- `pressure.py` - learning pressure signals
- `self_modification.py` - self-modification proposals
- `cognitive_niche.py` + `niche_forking.py` - niche identification and divergence (Speciation Phase 1)
- `speciation.py` - speciation event orchestration
- `detectors/synapse_cognitive_stall_detector.py` + `simula_codegen_stall_detector.py` - stall detection for Synapse and Simula

### Gap Closure (7 Mar 2026)
- **ArxivProposalTranslator lazy import** - top-level cross-system import from `service.py` removed. `_handle_new_arxiv_innovation()` instantiates `ArxivProposalTranslator()` lazily inside the function body.
- **`CausalFailureAnalyzer` wired** - `EvoService.initialize()` constructs `CausalFailureAnalyzer` lazily from `systems.simula.coevolution.causal_surgery` with try/except guard. Passes it to `ConsolidationOrchestrator`. Phase 6.5 failure-pattern detection is now live.
- **`wire_oikos(oikos)` method** - runtime Oikos injection (no constructor cross-import). Propagates to `ConsolidationOrchestrator._oikos`.
- **`EVO_BELIEF_CONSOLIDATED` emitted** - at end of Phase 2.75 (`_phase_belief_consolidation`). Payload: `beliefs_consolidated`, `foundation_conflicts`, `consolidation_number`.
- **`EVO_GENOME_EXTRACTED` emitted** - in Phase 2.8 when a genome is produced. Payload: `genome_id`, `candidates_fixed`, `genome_size_bytes`, `generation`.
- **`NICHE_FORK_PROPOSAL` event** - `NicheForkingEngine._dispatch_proposal()` now emits `NICHE_FORK_PROPOSAL` instead of `EVOLUTION_CANDIDATE`. All three events added to `synapse/types.py`.
- **Phase 5 Oikos metabolic gate** - `_phase_parameter_optimisation()` calls `check_metabolic_gate(MetabolicPriority.GROWTH)` before running. Skips with `(0, 0.0)` on denial. Non-fatal when Oikos ref is None.
- **`learning_speedup_pct` implemented** - `GenomeSeeder.seed_from_genome()` computes `seeded_count / total_inherited × 100.0`. Logged with inheritance report.
- **Genome version migration** - `_CURRENT_GENOME_VERSION = 1`, `_migrate_genome_item()` fills missing v1 fields via `setdefault`. `decompress_genome()` applies migration + graceful-skip on corrupt records.
- **`NicheRegistry` metabolic gate** - `set_starvation_level(level: str)` method added. `create_niche_from_species()` blocks when level is `starving/critical/terminal`. `EvoService` propagates starvation level to `NicheRegistry` in its Oikos starvation handler.

---

## Gap Closure (7 Mar 2026 - second pass)

- **`export_belief_genome()`** - `EvoService.export_belief_genome()` added (`service.py`). Returns `primitives.genome_inheritance.BeliefGenome` with top-50 hypotheses (confidence ≥ 0.6), current drive weight snapshot (via Equor), last 10 constitutional drift entries (via Memory), and generation counter. Module-level helpers `_safe_get_drive_scores` + `_safe_fetch_drift_history` appended at end of file. Called by `SpawnChildExecutor` Step 0b at spawn time.

## Gap Closure (7 Mar 2026 - event coverage fix)

- **`GENOME_EXTRACT_REQUEST` emitted from `export_belief_genome()`** - signals genome extraction starting on the Synapse bus. Emitted before the `try:` block so spec_checker can observe it regardless of extraction outcome. Non-fatal: if `_event_bus` is None or emit fails, extraction continues unaffected. Satisfies spec_checker's `genome_extract_request` expected event for the `evo` system.

## Gap Closure - Round 2A (7 Mar 2026)

### Staleness Decay (HYPOTHESIS_STALENESS handler)
- `_on_hypothesis_staleness()` - fully implemented in `service.py`
- Iterates all PROPOSED/TESTING hypotheses in `_hypothesis_engine._active`
- Multiplies `evidence_score *= (1.0 - staleness_rate)` for each
- Archives hypotheses where `evidence_score < 0.05` via `archive_hypothesis(h, reason="staleness_decay")`
- Emits `EVO_HYPOTHESES_STALED` (if any archived) - payload: `decayed_count`, `archived_count`, `archived_ids`
- Always emits `EVO_HYPOTHESIS_REVALIDATED` - VitalityCoordinator uses this to call `on_hypotheses_revalidated()` and reduce cumulative entropy pressure (closes the degradation feedback loop)

## Gap Closure (8 Mar 2026 - Parameter Adjustment Feedback Loop)

### Closed: No revert mechanism for degrading parameter adjustments
**Problem:** Evo tuned parameters (`atune.head.novelty.weight`, etc.) via `EVO_PARAMETER_ADJUSTED` but never checked whether the adjustment helped or hurt. Bad adjustments persisted silently.

**Fix:**
- `ParameterAdjustmentRecord` dataclass - captures `param_path`, `old/new_value`, `cycle_applied`, `timestamp_applied`, `hypothesis_id`, `baseline_metrics`, `extensions_used`
- `apply_adjustment(adj, current_metrics, cycle)` - captures baseline KPI snapshot into a `ParameterAdjustmentRecord` before modifying `self._values`. Record queued in `_pending_adjustments`.
- `tick_evaluation(current_metrics, cycle)` - periodic evaluation (every 500 cycles OR 30 min). Computes geometric mean of per-KPI improvement ratios. Three outcomes:
  - ratio > 1.05 → `_confirm_adjustment()` + positive hypothesis evidence
  - ratio < 0.95 → `_revert_adjustment()`: restores `old_value`, emits `EVO_PARAMETER_REVERTED`, sends negative hypothesis evidence
  - neutral → extend eval window (up to 2×), then force-confirm
- `wire_hypothesis_engine(engine)` - injects `HypothesisEngine` for `record_parameter_outcome()` callbacks
- `_feed_hypothesis_evidence()` - calls `hypothesis_engine.record_parameter_outcome(hypothesis_id, success, improvement_ratio, param_path)` (best-effort, non-fatal)
- `_persist_revert()` - writes reverted value back to `:EvoParameter` Neo4j node
- `_compute_improvement_ratio()` - module-level helper; geometric mean over matching KPI keys; returns 1.0 (neutral) when no comparable KPIs

**Wiring (EvoService):**
- `_last_benchmark_kpis: dict[str, float]` cache - updated in `_on_benchmark_regression` from each `BENCHMARK_REGRESSION` event
- `wire_hypothesis_engine()` called in `initialize()` immediately after `ParameterTuner` is constructed
- `_run_consolidation_now()` passes `current_metrics=self._last_benchmark_kpis` to `_orchestrator.run()` and calls `tick_evaluation` on the tuner after the result

**Wiring (ConsolidationOrchestrator):**
- `run(pattern_context, current_metrics=None)` - stores `current_metrics` on `self._current_metrics` for Phase 5
- `_phase_parameter_optimisation()` - each `apply_adjustment()` call now passes `current_metrics=self._current_metrics` and `cycle=self._total_runs`

**New SynapseEventType:** `EVO_PARAMETER_REVERTED = "evo_parameter_reverted"` in `synapse/types.py`

---

## Gap Closure (8 Mar 2026 - Belief Genome Child Inheritance)

### `_apply_inherited_belief_genome_if_child()` (NEW)
**Problem:** `EvoService.export_belief_genome()` existed and was called by `SpawnChildExecutor`, but
there was no corresponding child-side method to *receive and apply* that genome. Children booted
with no hypothesis priors regardless of what the parent exported.

**Fix:**
- `_apply_inherited_belief_genome_if_child()` added to `EvoService`
- Reads `ORGANISM_BELIEF_GENOME_PAYLOAD` (JSON-encoded `BeliefGenome`)
- Skips silently if `ORGANISM_IS_GENESIS_NODE=true` or env var absent
- Each inherited hypothesis is added to `_pending_candidates` as a `PatternCandidate` with:
  - `pattern_type=COOCCURRENCE`, elements include `"inherited:{category}"` + statement (first 120 chars)
  - `confidence = parent_confidence × 0.95` (5% discount per generation)
  - `extra["inherited"]=True`, `extra["parent_genome_id"]`
- Drive weight snapshot stored as `self._inherited_drive_weights` (Equor may override at runtime)
- Drift history stored as `self._inherited_drift_history` for analytics
- Emits `GENOME_INHERITED` (system="evo") on Synapse bus if event_bus is wired
- Called from `initialize()` after hypothesis engine is built; wrapped in try/except (non-fatal)

## Gap Closure (9 Mar 2026 - Genome Integrity, Evidence Threshold, Tournament Seeding, KPI Freshness)

### Belief Genome Checksum Verification (Gap 2b)
**Problem:** Corrupted or tampered `BeliefGenome` payloads were applied silently. No integrity check.

**Fix:**
- `BeliefGenome` in `primitives/genome_inheritance.py` gained `genome_checksum: str`, `_compute_checksum()`, `seal()`, `verify()` methods
- `_apply_inherited_belief_genome_if_child()` calls `parent_genome.verify()` before applying. Mismatch → skip with `belief_genome_checksum_mismatch` log. Legacy genomes without checksums are trusted (backward-compatible: `verify()` returns True when `genome_checksum` is empty).
- Checksum = `SHA-256(genome_id|instance_id|generation|len(hypotheses)|len(tournament_priors))`

### Evidence Threshold for Inherited Hypotheses (Gap 2b / Gap 5)
**Problem:** N=1 inherited hypotheses carried more noise than signal. All were applied regardless of evidence.

**Fix:**
- `_MIN_SUPPORTING_COUNT = 3` - hypotheses with `supporting_count < 3` are skipped at apply time
- Complementary filter: export side (`export_belief_genome()`) already required min 5 samples for `tournament_beta_priors`
- Both sides independently guard against low-evidence inheritance

### Tournament Beta Prior Seeding (Gap 2b)
**Problem:** `TournamentEngine` was reset to uniform Beta(1,1) priors on every child boot. Parent's Thompson sampling history was discarded.

**Fix:**
- `TournamentEngine.seed_inherited_prior(tournament_id, hypothesis_id, alpha, beta)` added to `tournament.py`
  - If tournament already exists: updates `beta_parameters[hypothesis_id]` directly
  - If tournament not yet created: stores in `_pending_priors` (lazy dict, no attr set if unused)
  - `_create_tournament()` now applies any pending priors for matching `(tournament_id, hypothesis_id)` pairs immediately after registering
- `_apply_inherited_belief_genome_if_child()` iterates `parent_genome.tournament_beta_priors` and calls `seed_inherited_prior()` for each valid entry
- Non-fatal: any exception per-prior is silently skipped; `tournament_priors_seeded` count is logged

### KPI Freshness between Regression Cycles (Gap 2c)
**Problem:** `_last_benchmark_kpis` was only updated on `BENCHMARK_REGRESSION` events. Between regressions (which may be hours apart), `ParameterTuner.tick_evaluation()` ran against stale KPI baselines.

**Fix:**
- New handler `_on_domain_kpi_snapshot()` in `EvoService` - subscribed to `DOMAIN_KPI_SNAPSHOT` (already emitted frequently by Benchmarks: every Nexus epistemic cycle, every RE export, every economic deferral)
- Updates `_last_benchmark_kpis` for any known float KPI keys: `decision_quality`, `llm_dependency`, `economic_ratio`, `learning_rate`, `mutation_success_rate`, `effective_intelligence_ratio`, `compression_ratio`, `re_success_rate`, `re_usage_pct`, `epistemic_value_per_cycle`, `schema_quality_trend`, `success_rate`, `profitability`
- `DOMAIN_KPI_SNAPSHOT` subscription registered in `register_on_synapse()` with `hasattr` guard (non-fatal if event type absent)
- KPI cache is now continuously warm; `tick_evaluation()` always sees near-real-time baselines

## What's Missing

- **Phase 8 callback, not Synapse event** - `_simula_callback` injected at init; proposals dropped silently if Simula is unavailable. No retry/queue.
- **Tournament routing requires Nova awareness** - if Nova dispatches without `tournament_context`, no outcomes are recorded; tournament never converges.
- **`Goal` type from `systems.nova.types`** - lazy import inside `_generate_goal_from_hypothesis()`. Not a blocker (lazy), but should eventually move `Goal` to `primitives/`.

---

## Key Files

| File | Purpose |
|------|---------|
| `service.py` | EvoService - wake mode, consolidation trigger, all wiring |
| `hypothesis.py` | HypothesisEngine - full lifecycle |
| `consolidation.py` | ConsolidationOrchestrator - 8-phase sleep pipeline |
| `detectors.py` | 4 online pattern detectors + PatternContext |
| `tournament.py` | TournamentEngine - Thompson sampling A/B |
| `parameter_tuner.py` | 23-param velocity-limited tuner |
| `belief_halflife.py` | BeliefAgingScanner - radioisotope decay |
| `belief_consolidation.py` | BeliefConsolidationScanner - hardening + conflict detection |
| `genetic_memory.py` | GenomeExtractor / GenomeSeeder (legacy path) |
| `genome.py` | EvoGenomeExtractor - Mitosis GenomeExtractionProtocol |
| `self_model.py` | SelfModelManager - capability + regret scoring |
| `procedure_extractor.py` | ProcedureExtractor from PatternContext.sequence_counts |
| `procedure_codifier.py` | ProcedureCodifier - Neo4j :Procedure nodes |
| `schema_induction.py` | SchemaInduction proposals to Simula |
| `speciation.py` | Speciation event detection and emission |
| `types.py` | All Evo types |

---

## Integration Surface

### Events Consumed
| Event | Source | Handler |
|-------|--------|---------|
| `DOMAIN_KPI_SNAPSHOT` | Benchmarks | `_on_domain_kpi_snapshot()` (9 Mar 2026) - keeps `_last_benchmark_kpis` fresh between regression cycles; updates 13 known float KPI keys |
| *(workspace broadcast)* | Synapse | `receive_broadcast()` - every cycle; `episode`, `affect_state`, `memory.entities` |
| `KAIROS_TIER3_INVARIANT_DISCOVERED` | Kairos | Pre-validated SUPPORTED hypothesis, no age wait |
| `KAIROS_INVARIANT_DISTILLED` | Kairos | `_on_kairos_invariant()` - hybrid tier routing: Tier 3 + confidence ≥ 0.8 → direct `Hypothesis(status=SUPPORTED, evidence_score=confidence×5, source="kairos_invariant")`; Tier 2 + confidence 0.6–0.79 → boosted `PatternCandidate(confidence+0.10)`; Tier 1 → standard `PatternCandidate`. Tier-3 hypotheses tagged `source="kairos_invariant"` to distinguish from Evo-native (prevents auto-revert). Also boosts existing aligned hypotheses by +2.0 evidence_score. |
| `BOUNTY_PAID` | Oikos | `_on_bounty_paid()` - confirmed revenue → high-valence episode + bounty_outcomes window (SG5) |
| `ASSET_BREAK_EVEN` | Oikos | `_on_asset_break_even()` - asset ROI confirmed → positive episode for asset-type hypothesis (SG5) |
| `CHILD_INDEPENDENT` | Oikos | `_on_child_independent()` - successful reproduction → highest-valence episode for reproduction strategy (SG5) |
| `METABOLIC_EFFICIENCY_PRESSURE` | Oikos | `_on_metabolic_efficiency_pressure()` - records negative-valence economic episode + appends TEMPORAL PatternCandidate to `_pending_candidates` (hypothesis_domain: yield_strategy \| budget_allocation \| niche_selection) for next consolidation's hypothesis generation pass |
| `RE_DECISION_OUTCOME` | Nova | `_on_re_decision_outcome()` - tracks consecutive degraded readings (`_re_degradation_count`); after 10 consecutive readings with `success_rate < 0.60`, queues a PatternCandidate targeting `reasoning_engine.hyperparameter_adjustment` for the next consolidation pass |
| `OIKOS_ECONOMIC_EPISODE` | Oikos | `_on_oikos_economic_episode()` (2026-03-08) - consumes action_type, success, roi_pct, protocol, causal_applicable_domains; creates COOCCURRENCE `PatternCandidate` in `_pending_candidates`; confidence = 0.20 (failure) or 0.40–0.80 scaled by roi_pct (success) |
| `METABOLIC_GATE_RESPONSE` | Oikos | `_on_metabolic_gate_response()` (2026-03-08) - denied gates only; creates TEMPORAL `PatternCandidate` with elements `[metabolic_gate_denied::{action_type}, starvation_level::{level}, economic_constraint]`; confidence=0.35 |
| `EVO_THOMPSON_QUERY` | Nova | `_on_thompson_query()` (2026-03-08, arch fix) - Nova requests arm weights for a domain; calls `get_thompson_arm_weights(domain)` internally; emits `EVO_THOMPSON_RESPONSE` with request_id correlation; non-fatal if event bus unavailable |
| `BENCHMARK_REGRESSION` | Benchmarks | `_on_benchmark_regression()` (2026-03-08, Path 4 added 2026-03-09) - four paths: (1) severity==critical → emergency PatternCandidate queued; (2) 3+ consecutive regressions for same KPI → `LEARNING_PRESSURE` emitted (rate-limited 1/hour) + `_orchestrator.on_learning_pressure()` called; (3) RE-related KPI → `RE_TRAINING_REQUESTED` emitted; **(4) pending unevaluated ParameterTuner adjustments exist → rollback PatternCandidate queued with suspect param + old/new values for hypothesis engine to evaluate causation** |
| `BELIEF_CONSOLIDATED` | Memory | `_on_belief_consolidated()` (9 Mar 2026) - keyword-matches consolidated belief text against active hypothesis statements; hypotheses with ≥2 keyword hits get a boost of `belief_confidence × min(0.8, overlap × 0.15) × volatility_weight` |
| `SCHEMA_FORMED` | Thread | `_on_schema_formed()` (9 Mar 2026) - creates a SELF_MODEL `Hypothesis` with `id=f"schema_{schema_id}"`, `evidence_score = min(2.0, supporting_count × 0.2)`, `status=TESTING` - seeds Evo with Thread's crystallised identity schemas |
| `SCHEMA_EVOLVED` | Thread | `_on_schema_evolved()` (9 Mar 2026) - locates existing schema hypothesis by id; boosts `evidence_score` by 0.5 (`new_strength < 0.5`) or 1.0 (`new_strength ≥ 0.5`) |
| `CONVERGENCE_DETECTED` | Nexus | `_on_convergence_detected()` (9 Mar 2026) - keyword-matches convergence concepts against hypotheses; effective_confidence = geomean(`convergence_score`, `triangulation_confidence`) × diversity_bonus; diversity_bonus = 1.0 + `0.1 × (unique_instances - 2)` |
| `DREAM_INSIGHT` | Oneiros | `_on_dream_insight()` (9 Mar 2026) - coherence ≥ 0.70 gate; if source hypothesis ids provided, boosts each by `coherence × 0.3`; otherwise creates a WORLD_MODEL hypothesis with `evidence_score = coherence × 1.5`, `min_age_hours = 1.0` |
| `TRIANGULATION_WEIGHT_UPDATE` | Nexus | `_on_triangulation_weight_update()` (9 Mar 2026) - caches `self._triangulation_weight` (initialised 0.5); weight used to scale convergence boosts in `_on_convergence_detected()` |
| `DIVERGENCE_PRESSURE` | Nexus | `_on_divergence_pressure()` (9 Mar 2026) - `triangulation_weight < 0.4` gate; queues ≤5 COOCCURRENCE `PatternCandidate`s (one per saturated domain from payload), confidence=0.55, to stimulate cross-domain hypotheses during federated divergence |
| `TURNING_POINT_DETECTED` | Thread | `_on_turning_point_detected()` (9 Mar 2026) - `surprise_magnitude ≥ 0.4` gate; queues a TEMPORAL `PatternCandidate` with `confidence = min(0.75, 0.3 + surprise_magnitude × 0.5)` - narrative inflection → Evo exploration |

### Events Emitted
| Event | Trigger | Consumer |
|-------|---------|---------|
| `EVO_DRIFT_DATA` | Phase 7 | Equor |
| `EVO_HYPOTHESIS_CREATED` | Generation | RE training pipeline |
| `EVO_HYPOTHESIS_CONFIRMED` | SUPPORTED | RE training pipeline (Stream 5) |
| `EVO_HYPOTHESIS_REFUTED` | REFUTED | RE training pipeline |
| `EVO_CONSOLIDATION_COMPLETE` | End of sleep | Benchmarks, monitoring |
| `EVO_CONSOLIDATION_QUALITY` | After every consolidation | Benchmarks, Telos, Thread |
| `SCHEMA_INDUCED` | `_phase_schema_induction()` - per element in engine path; per success in legacy path | Thread (`_on_schema_induced`), Logos (`_on_schema_induced`) |
| `EVO_CAPABILITY_EMERGED` | Structural hypothesis integrated | Telos |
| `EVO_PARAMETER_ADJUSTED` | `ParameterTuner.apply_adjustment()` | Atune, Nova, Voxis |
| `EVO_PARAMETER_REVERTED` | `ParameterTuner._revert_adjustment()` - degradation detected | Atune, Nova, Voxis (re-apply old value) |
| `EVO_BELIEF_CONSOLIDATED` | End of Phase 2.75 | Benchmarks, Thread, monitoring |
| `EVO_GENOME_EXTRACTED` | Phase 2.8 genome produced | Mitosis, Benchmarks |
| `NICHE_FORK_PROPOSAL` | Phase 2.95 fork dispatch | Simula (organogenesis) |
| `SPECIATION_EVENT` | Magnitude > 0.2 | Telos, Alive, Benchmarks |
| `INCIDENT_DETECTED` | Foundation conflicts (Phase 2.75) | Thymos |
| `FITNESS_OBSERVABLE_BATCH` | After consolidation | Benchmarks |
| `RE_TRAINING_EXAMPLE` | Hypothesis created/confirmed/refuted | RE training pipeline |
| `EVO_THOMPSON_RESPONSE` | In response to `EVO_THOMPSON_QUERY` | Nova - resolves `_thompson_futures[request_id]` Future |
| `LEARNING_PRESSURE` | 3+ consecutive regressions for a KPI (rate-limited 1/hour) | Evo consolidation loop - `_orchestrator.on_learning_pressure()` enables early consolidation after ≥1h cooldown |
| `RE_TRAINING_REQUESTED` | RE-related KPI regression (re_success_rate, decision_quality) | ContinualLearningOrchestrator - lowers min_examples to 50 and fires Tier 2 urgently |
| `BENCHMARK_THRESHOLD_UPDATE` | Phase 5.5 of every consolidation cycle (`_emit_benchmark_threshold_calibration`) | Benchmarks - adjusts `re_progress_min_improvement_pct` (3.0/5.0/7.0% based on learning_rate) and `metabolic_degradation_fraction` (0.07/0.10/0.15 based on economic_ratio); deduplicated (only emits when value shifts ≥0.5% or ≥0.01 fraction); payload includes `learning_rate`, `economic_ratio`, `adj_count`, `consolidation_number` |
| *(evolution proposals via callback)* | Phase 8 | Simula (injected callback, not Synapse) |

---

## Constraints

- **Wake mode ≤20ms** - pattern detection + evidence accumulation; no Neo4j reads on hot path
- **Consolidation ≤60s** - 8 phases sequential; all wrapped in try/except
- **Cannot modify:** Equor evaluation, constitutional drives, invariants, or self-evaluation criteria
- **Velocity limits are hard:** single delta ≤ 0.03, cumulative ≤ 0.15/cycle - no exceptions
- **No cross-system imports** - all via Synapse events or injected service refs
