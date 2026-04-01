# Thread - Narrative Identity & Temporal Self-Continuity

**Spec**: `.claude/EcodiaOS_Spec_15_Thread.md` (v1.3, updated 8 March 2026 causal grounding)
**SystemID**: `thread`

## What Thread Does

Thread maintains the organism's autobiographical self - who it is, what it's committed to, how it has changed, and what chapter of its life it's living. Implements Ricoeur's narrative identity: **idem** (structural sameness via IdentitySchemas) and **ipse** (ethical selfhood via Commitments). McAdams Level 3 Life Story Model provides the chapter/scene/turning-point structure. Friston's self-evidencing runs at identity level: SelfEvidencingLoop generates predictions from schemas, collects per-episode evidence, and updates the self-model from prediction error. DiachronicCoherenceMonitor distinguishes drift (unexplained behavioural change) from growth (change explained by narrative context) via Wasserstein distance on 29D behavioural fingerprints.

Thread is the continuity organ - it makes EOS a persistent individual rather than a sequence of disconnected inference calls.

## Architecture

```
ThreadService (orchestrator)
├── SelfEvidencingLoop        - active inference for identity; schema predictions → per-episode evidence → identity surprise
├── ChapterDetector            - Bayesian surprise accumulator; 5-factor weighted boundary detection; ≤10ms per episode
├── IdentitySchemaEngine       - core self-beliefs; evidence tracking; idem score; velocity-limited promotions/decay
├── CommitmentKeeper           - promise tracking (ipse); fidelity testing; RUPTURE TurningPoint on broken commitments
├── NarrativeSynthesizer       - LLM: scene/chapter/life-story composition; arc detection; hot-reloadable
├── NarrativeRetriever         - Neo4j: who_am_i, schema_relevant, chapter_context, past_self (no LLM)
└── DiachronicCoherenceMonitor - Wasserstein distance on 29D fingerprints; narrative-contextualized growth/drift/transition
```

## What's Implemented (as of 8 March 2026)

### Causal Grounding - NEW (8 March 2026)
- **35 inbound Synapse subscriptions** (+4 causal grounding Mar 8, +4 learning trajectory Mar 9, +2 goal events): `kairos_invariant_distilled`, `kairos_internal_invariant`, `evo_parameter_adjusted`, `equor_amendment_auto_adopted`, `crash_pattern_confirmed`, `benchmark_re_progress`, `thymos_repair_requested`, `thymos_repair_complete`
- **Causal attribution on TurningPoints**: `_get_causal_attribution(context_keywords, limit, max_cycle_age)` scans the rolling invariant cache and returns up to N natural-language attribution strings for embedding in `turning_point_detected` payload. Used by Kairos Tier 3 handler and Equor amendment handler.
- **`_cached_kairos_invariants`** (rolling window, max 50): stores both external world invariants (`KAIROS_INVARIANT_DISTILLED`) and internal self-causal laws (`KAIROS_INTERNAL_INVARIANT`). Each entry tagged with `source="external"` or `source="internal"`.
- **`_cached_evo_adjustments`** (rolling window, max 30): Evo parameter adjustments with delta, system_id, reason.
- **Causal chapter boundaries**: `_pending_causal_theme` (str) set when a causal regime change is detected:
  - `EVO_PARAMETER_ADJUSTED` with |delta| ≥ 0.15 → sets `_pending_causal_theme`
  - `EQUOR_AMENDMENT_AUTO_ADOPTED` → always sets `_pending_causal_theme`, also emits REVELATION TurningPoint with causal attribution
  - On next chapter open: `causal_theme` field added to `chapter_opened` payload; `_pending_causal_theme` consumed (one per chapter).
- **Internal invariant caching**: `_on_kairos_internal_invariant()` stores Kairos self-causal laws for attribution. These describe the organism's own dynamics (e.g. "prediction_error_rate increases coherence_decrease [lag=1 pipeline run]").

### Fully Operational (as of 7 March 2026 gap closure + 8 March 2026 causal grounding)
- **All 13 Synapse events emitted**: `chapter_closed/opened`, `turning_point_detected`, `schema_formed/evolved/challenged`, `identity_shift_detected/dissonance/crisis`, `commitment_made/tested/strain`, `narrative_coherence_shift` - all via `_emit_event()`
- **25 core inbound subscriptions** (was 16, +9 added 8 March 2026): `episode_stored`, `fovea_internal_prediction_error`, `wake_initiated`, `voxis_personality_shifted`, `somatic_drive_vector`, `self_affect_updated`, `action_completed`, `schema_induced`, `kairos_tier3_invariant_discovered`, `goal_achieved`, `goal_abandoned`, `nova_goal_injected`, `lucid_dream_result`, `oneiros_consolidation_complete`, `self_model_updated`, `evo_belief_consolidated`, and **Economic & Domain Milestone**: `domain_mastery_detected`, `domain_performance_declining`, `asset_break_even`, `child_independent`, `revenue_injected`, `bounty_paid`, `equor_economic_permit`, `evo_hypothesis_created`, `evo_belief_consolidated`
- **Rich chapter events (HIGH gap)**: `chapter_closed` and `chapter_opened` include `narrative_theme`, `dominant_drive`, `start_episode_id`, `constitutional_snapshot`, `trigger`, and now `causal_theme` (causal force that opened this chapter, if any).
- **Drive-drift chapter trigger (HIGH gap)**: Slow EMA (α=0.05) on `_cached_drive_alignment`; sustained drift >0.2 across 10 episodes triggers a chapter boundary with `trigger="identity_shift"`.
- **Goal-domain chapter trigger (HIGH gap)**: `_infer_goal_domain()` extracts coarse domain from episode text (6 labels). Domain transition → chapter boundary with `trigger="goal_domain_began"`.
- **Constitutional snapshot helper**: `_build_constitutional_snapshot()` returns up to 8 core schemas, 6 commitments, drive vector, personality, idem/ipse, coherence - used in chapter events.
- **Kairos Tier 3 narrative milestone (MEDIUM gap)**: `_on_kairos_tier3_invariant()` creates a `REVELATION` TurningPoint (narrative_weight=0.9) and now includes `causal_attribution` list extracted from the invariant cache.
- **SelfEvidencingLoop**: instantiated in `initialize()`, `tick()` in `on_cycle()`, `collect_evidence()` + `classify_surprise()` in `process_episode()`; emits `identity_dissonance` (surprise ≥ 0.5) and `identity_crisis` (surprise ≥ 0.8)
- **Chapter lifecycle**: full 8-step closure pipeline - mark CLOSED, snapshot personality, detect arc, compose narrative (NarrativeSynthesizer), create successor with `PRECEDED_BY`, reset accumulator, emit events, persist to Neo4j
- **Zero direct cross-system imports**: all cross-system state via Synapse event caching (`_cached_personality` 9D, `_cached_drive_alignment` 4D, `_cached_affect` 6D)
- **RE training**: Stream 6 `thread_narrative_reasoning` - `_emit_re_training_trace()` fires on every event emission
- **GenomeExtractionProtocol**: `extract_genome_segment()` / `seed_from_genome_segment()` for Mitosis
- **NarrativeRetriever**: wired for `who_am_i_full()` - assembles `NarrativeIdentitySummary` from Neo4j without LLM (≤500ms)
- **Neo4j schema**: 6 node labels with constraints, performance indexes, 4 vector indexes (chapter, schema, turning_point, commitment), `PRECEDED_BY` chaining on chapters
- **IdentitySchemaEngine**: full CRUD, fast-path (cosine) + slow-path (LLM) evidence evaluation, velocity-limited promotions (NASCENT→DEVELOPING→ESTABLISHED→CORE), inactive decay, `compute_idem_score()` 4-component formula
- **CommitmentKeeper**: all 4 formation sources, LLM fidelity testing with embedding gate, Iron Rule #4 (fidelity < 0.4 over 5+ tests → BROKEN + RUPTURE TurningPoint), ipse_score computation
- **NarrativeSynthesizer**: scene (≤2s), chapter (≤5s), life-story (≤15s), arc detection - all implemented and hot-reloadable via `BaseNarrativeSynthesizer` ABC
- **DiachronicCoherenceMonitor**: instantiated in `initialize()` when neo4j+llm available; fed via `_compute_fingerprint()` every 100 cycles; `assess_change()` called in `on_cycle()` - narrative-contextualized growth/drift/transition/stable classification drives `identity_shift_detected` and `identity_crisis` events. Falls back to simple L1 if monitor unavailable.
- **NarrativeScene creation**: `_episode_scene_buffer` accumulates episode summaries; `compose_scene()` called every `_SCENE_EPISODE_THRESHOLD` (20) episodes; scene persisted to Neo4j via `_persist_scene()` with `(:NarrativeChapter)-[:CONTAINS]->(:NarrativeScene)` link; buffer reset on chapter close.
- **Self node identity scores**: `autobiography_summary`, `idem_score`, `ipse_score`, `current_life_theme` written to `Self` node in both `_persist_state_to_graph()` (every 500 cycles) and immediately in `integrate_life_story()` so `NarrativeRetriever.who_am_i_full()` reads current data.
- **CURRENT_CHAPTER relationship**: already written in chapter closure pipeline when new chapter opens.

### Not Yet Wired
- Inbound subscriptions missing: `pattern_detected` (Evo - event not yet defined), `rem_metacognition_observation`, `constitutional_drift_detected`, `incident_resolved`
- Population fingerprint divergence across fleet (Bedau-Packard speciation signal)
- `identity_relevance` signal to Atune salience (not specified in Atune's spec)
- Commitment-goal priority boost in Nova `drive_resonance` (not specified in Nova's spec)
- `NarrativeRetriever.get_reasoning_context()` not implemented (RE context injection)

### Fixed (2026-03-08 - autonomy audit)
- **CRITICAL: `set_llm()` never called from `registry.py`** - `CommitmentKeeper`, `IdentitySchemaEngine`, `NarrativeRetriever`, `DiachronicCoherenceMonitor`, and `NarrativeSynthesizer` were never instantiated. Thread ran in permanently degraded mode: no schema evaluation, no commitment fidelity testing, no LLM narrative synthesis, no diachronic coherence classification. Fixed: `registry.py:_init_thread()` now calls `thread.set_llm(infra.llm)` before `initialize()`.
- **CRITICAL: RE training emitted on wrong event type** - `_emit_re_training_trace()` used `SynapseEventType.SCHEMA_INDUCED` instead of `RE_TRAINING_EXAMPLE`. All Thread RE training examples (schema formation, commitment detection, chapter reasoning, self-evidencing) were being routed to schema handlers instead of the RE exporter. Fixed: `service.py:2354`.
- **CRITICAL: `NarrativeSynthesizer` never instantiated at boot** - even with LLM wired, the synthesizer was only created via NeuroplasticityBus hot-reload (which happens lazily). Scene composition and chapter narrative were silently skipped at every boundary. Fixed: `initialize()` now directly instantiates `NarrativeSynthesizer(llm=..., config=..., organism_name=...)` alongside the other sub-systems. `set_llm()` post-init also creates it if absent.
- **Invisible telemetry: `identity_drift_detected` not emitted to bus** - fine-grained fingerprint delta was logged internally but invisible to all other systems (Benchmarks, Fovea, Telos). Fixed: `_compute_fingerprint()` now emits `identity_shift_detected` event for drifts > 0.05 - classification `"drift"` above `wasserstein_major_threshold`, else `"growth"`.
- **Invisible telemetry: schema conflicts never emitted** - `_detect_schema_conflicts()` found contradictory schema pairs every 1000 cycles but never emitted them. Oneiros lucid processing could not route conflicts for resolution. Fixed: `_detect_schema_conflicts()` now emits `schema_challenged` with `conflict_type="schema_contradiction"` and both schema IDs/statements.
- **Invisible telemetry: `integrate_life_story()` never broadcast its result** - life story snapshot computed every 5000 cycles but not visible to the bus. Fixed: `integrate_life_story()` now emits `narrative_coherence_shift` with synthesis excerpt, coherence, and chapter count after each synthesis.
- **Bug: `_current_chapter` attribute used in `_on_equor_economic_permit()`** - attribute doesn't exist on `ThreadService`; only `_get_active_chapter()` exists. Would raise `AttributeError` on any EQUOR_ECONOMIC_PERMIT event. Fixed: replaced with `self._get_active_chapter()`.
- **Bug: `TurningPoint(significance=0.65)` invalid field** - `TurningPoint` has no `significance` field; the constructor used in `_on_equor_economic_permit` would silently discard it (Pydantic v2 raises `ValidationError` in strict mode). Fixed: changed to `surprise_magnitude=0.65`.

### Fixed (2026-03-07)
- **`narrative_milestone` now emitted** - was logged but never broadcast. 4 call sites wired: `kairos_tier3` (causal_discovery), `nova_goal_achieved` (goal_achieved), `nova_goal_abandoned` (goal_abandoned), `oneiros_lucid_dream` (lucid_dream_simulation). Each payload includes `milestone_type`, `source`, `chapter_id`, and context fields.

## Key Files

| File | Lines | Role |
|------|-------|------|
| `service.py` | ~1100 | Orchestrator - lifecycle, Synapse wiring, event emission, chapter closure, `process_episode()` pipeline |
| `types.py` | ~600 | All types: IdentitySchema, NarrativeChapter, Commitment, IdentityFingerprint, ThreadConfig |
| `self_evidencing.py` | ~255 | SelfEvidencingLoop: predictions from schemas, evidence collection, 4-tier surprise classification |
| `chapter_detector.py` | ~250 | ChapterDetector: 5-factor Bayesian surprise, spike/sustained/goal-resolution triggers |
| `identity_schema_engine.py` | ~700 | Schema CRUD, evidence evaluation, velocity-limited promotions, decay, idem_score, Neo4j persistence |
| `commitment_keeper.py` | ~420 | Commitment formation, fidelity testing, RUPTURE enforcement, ipse_score, Neo4j persistence |
| `narrative_synthesizer.py` | ~400 | LLM: scene composition, chapter narrative, life-story integration, arc detection |
| `narrative_retriever.py` | ~480 | Neo4j queries: who_am_i_full, schema_relevant (vector search), chapter_context, past_self |
| `diachronic_coherence.py` | ~350 | Wasserstein distance, fingerprint computation, growth/drift/transition classification (wired) |
| `schema.py` | ~120 | Neo4j schema setup (constraints, indexes, vector indexes) |
| `processors.py` | ~170 | ABCs for hot-reloadable NarrativeSynthesizer + ChapterDetector via NeuroplasticityBus |

## Integration Points

### Emits (14 events)
- `chapter_closed`, `chapter_opened` - chapter lifecycle
- `turning_point_detected` - narrative inflection (CRISIS, REVELATION, COMMITMENT, LOSS, ACHIEVEMENT, ENCOUNTER, RUPTURE, GROWTH, RESILIENCE)
- `schema_formed`, `schema_evolved`, `schema_challenged` - identity beliefs
- `identity_shift_detected` (W-dist 0.25–0.49), `identity_dissonance` (surprise 0.5–0.79), `identity_crisis` (surprise ≥ 0.8 or W-dist ≥ 0.50)
- `commitment_made`, `commitment_tested`, `commitment_strain` (ipse_score < 0.6)
- `narrative_coherence_shift`
- `narrative_milestone` - significant autobiographical moment (causal_discovery / goal_achieved / goal_abandoned / lucid_dream_simulation)

### Consumes
**Wired (29 total)**:

Core (16): `episode_stored`, `fovea_internal_prediction_error`, `wake_initiated`, `voxis_personality_shifted`, `somatic_drive_vector`, `self_affect_updated`, `action_completed`, `schema_induced`, `kairos_tier3_invariant_discovered`, `goal_achieved`, `goal_abandoned`, `nova_goal_injected`, `lucid_dream_result`, `oneiros_consolidation_complete`, `self_model_updated`, `evo_belief_consolidated`

Economic & Domain Milestone (9): `domain_mastery_detected`, `domain_performance_declining`, `asset_break_even`, `child_independent`, `revenue_injected`, `bounty_paid`, `equor_economic_permit`, `evo_hypothesis_created`, `evo_belief_consolidated`

**Causal Grounding (4 - new 8 Mar 2026)**: `kairos_invariant_distilled`, `kairos_internal_invariant`, `evo_parameter_adjusted`, `equor_amendment_auto_adopted`

**Learning Trajectory (4 - new 9 Mar 2026)**: `crash_pattern_confirmed`, `benchmark_re_progress` (filtered to `re_model.health_score`), `thymos_repair_requested`, `thymos_repair_complete`

**Planned, not wired**: `pattern_detected` (Evo - event not yet defined), `rem_metacognition_observation`, `constitutional_drift_detected`, `incident_resolved`

### Learning Trajectory Handlers - NEW (2026-03-09)

Four handlers give Thread self-awareness of the organism's learning journey:

**`_on_crash_pattern_confirmed`** - `CRASH_PATTERN_CONFIRMED`:
- Creates a `GROWTH` TurningPoint: `"Identified recurring failure pattern: {lesson}"`
- `surprise_magnitude = narrative_weight = confidence`; tags: `["learning","crash_pattern","self_knowledge"]`
- If `example_count >= 5` (deeply established pattern): sets `_pending_causal_theme = f"Post-{pattern_id[:8]} era"` and calls `_close_current_chapter_and_open_new()`

**`_on_re_model_improved`** - `BENCHMARK_RE_PROGRESS` (kpi_name=`re_model.health_score`, delta > 0.05):
- Creates a `GROWTH` TurningPoint: `f"Reasoning Engine improved to {value:.0%} health - organism is learning"`
- `significance = min(delta * 5, 1.0)` - small improvements noted quietly; large leaps become chapter milestones
- Tags: `["learning","reasoning_engine","capability_growth"]`

**`_on_thymos_repair_requested`** - `THYMOS_REPAIR_REQUESTED`:
- Caches non-preventive `NOVEL_FIX` repairs in `_pending_coma_repairs: dict[str, dict]` (keyed by incident_id)
- Only tracks repairs where `preventive=False` AND `repair_tier.upper() == "NOVEL_FIX"`

**`_on_thymos_repair_complete`** - `THYMOS_REPAIR_COMPLETE` (success=True):
- If `incident_id` in `_pending_coma_repairs`: pops entry, creates `RESILIENCE` TurningPoint (new type)
- Description: `"Survived crash and self-repaired - organism demonstrated resilience"`
- `surprise_magnitude = narrative_weight = 0.9`; tags: `["resilience","self_repair","survival","novel_fix"]`
- Always triggers a chapter boundary: `_pending_causal_theme = f"Post-incident survival: {class} in {system}"`

**New `TurningPointType.RESILIENCE`** added to `types.py`: `RESILIENCE = "resilience"` - survival from a crash and novel self-repair.

**New state**: `self._pending_coma_repairs: dict[str, dict[str, Any]] = {}` - in-flight coma repair tracker.

### SELF_MODEL_UPDATED handler (`_on_self_model_updated`) - NEW (2026-03-07, §8.6)
- Subscribed in `register_on_synapse()` - subscription count is now 29
- Creates a `REVELATION` TurningPoint when `month <= 1` (initial self-assessment) OR `coherence < 0.7` (significant identity shift)
- Stable self-models (month > 1, coherence >= 0.7) are silently logged - no TurningPoint created
- `significance = 1.0 - coherence`; this becomes `surprise_magnitude` and `narrative_weight` on the TurningPoint
- Emits `turning_point_detected` with `source="self_model_updated"`, `self_coherence`, `month`

### Memory Reads (Neo4j)
Episodes (by ID, by CONFIRMED_BY schema), Self node (personality, chapter, autobiography), active IdentitySchemas/Commitments, closed NarrativeChapters, BehavioralFingerprint chain, TurningPoints

### Memory Writes (Neo4j)
Writes only to its 6 node labels. Does **not** mutate Episode nodes - only adds `CONTAINS`, `CONFIRMED_BY`, `CHALLENGED_BY` relationships. Updates Self node: `autobiography_summary`, `current_life_theme` (not yet wired).

## Key Algorithms

**Chapter boundary detection** (≤10ms, no LLM) - 4 independent triggers:
```
1. Bayesian surprise:
   surprise = 0.25*affect_delta + 0.25*goal_event + 0.20*context_shift + 0.15*new_entity + 0.15*schema_challenge
   Boundary if (surprise > 3×EMA spike OR EMA > 2×baseline sustained OR goal resolution) AND min_episodes met

2. Drive-drift (new - 7 Mar 2026):
   drive_ema[d] = 0.05 * current[d] + 0.95 * drive_ema[d]  (slow EMA per drive)
   Boundary if any drive_ema[d] deviates > 0.2 from chapter-open baseline for ≥ 10 consecutive episodes

3. Goal-domain (new - 7 Mar 2026):
   domain = _infer_goal_domain(episode)  # heuristic keyword match → 6 coarse labels
   Boundary if domain != current_goal_domain AND current_goal_domain != ""

4. Causal regime change (new - 8 Mar 2026):
   _pending_causal_theme set by: EVO_PARAMETER_ADJUSTED (|delta| ≥ 0.15) or EQUOR_AMENDMENT_AUTO_ADOPTED
   On next chapter boundary: causal_theme injected into chapter_opened payload; consumed (one per chapter)
   EQUOR_AMENDMENT_AUTO_ADOPTED also emits REVELATION TurningPoint (narrative_weight=0.95) immediately.
```

**Causal attribution** (`_get_causal_attribution(keywords, limit, max_cycle_age)`):
```
Scans _cached_kairos_invariants (rolling 50) for entries matching context keywords.
Returns list[str] attribution strings - "Internal causal law: X [lag=1, hold=0.82]" or
"Kairos invariant: X [N domains]".
Injected into: REVELATION TurningPoints (Tier 3, Equor amendment), future handlers.
```

**Idem score** (structural sameness - target 0.6–0.85, not 1.0):
```
idem = 0.40*schema_stability + 0.30*personality_stability + 0.20*behavioral_consistency + 0.10*memory_accessibility
```

**Ipse score** (promise-keeping): `mean(commitment.fidelity for commitments with ≥3 tests)`

**Schema velocity limits**: max 1 formation per 48h; max 1 promotion per 24h; CORE requires 50+ confirmations AND 180+ days of age; CORE schemas never deleted (only MALADAPTIVE) - by design, not limitation.

**Fingerprint (29D)**: personality centroid 9D (weight 0.35) + drive alignment 4D (0.25) + affect centroid 6D (0.20) + goal source 6D (0.10) + interaction style 4D (0.10). Computed every 1000 cycles.

## RE Integration

All LLM calls use `claude-sonnet-4-6` (hardcoded in `ThreadConfig.llm_model`). RE Stream 6 training data already wired via `_emit_re_training_trace()` on every event emission (schema crystallization, evidence evaluation, commitment detection examples).

RE-suitable operations (not yet routed): schema crystallization, schema evidence evaluation slow path, commitment detection, drift classification LLM fallback. Keep on Claude: scene composition, chapter narrative, life-story integration.

`NarrativeRetriever.get_reasoning_context()` (not implemented) should inject active schema and commitment context into RE inference prompts - makes RE reasoning identity-coherent.

## Known Issues

1. `self._memory._neo4j` direct access (private attribute of Memory) - needs Memory public query API
2. Config defaults diverge from spec §9 (e.g. `chapter_min_episodes=10` vs spec's `50`) - intentional tuning, not bugs
3. Schema 48h temporal span check not enforced in `form_schema_from_pattern()` - requires Neo4j query on episode timestamps
4. `promote_schema()` is sync, emits async `SCHEMA_EVOLVED` via `asyncio.create_task()` fire-and-forget
5. `retrieve_past_self()` does not parse natural language dates - only "beginning", "chapter N", "last chapter"
6. `Evo.pattern_detected` event does not exist - schema auto-formation from Evo patterns is dead code until Evo emits it
7. Oneiros inbound events now confirmed: `ONEIROS_CONSOLIDATION_COMPLETE` (verified in oneiros/service.py) and `LUCID_DREAM_RESULT` (verified in oneiros/lucid_stage.py) - both correctly subscribed
8. `DiachronicCoherenceMonitor._classify_change()` Neo4j checks (`_check_schema_alignment`, `_check_turning_point_context`) are coarse heuristics - schema alignment check counts strong schemas rather than checking vector direction of change
