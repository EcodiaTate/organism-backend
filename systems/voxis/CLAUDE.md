# Voxis - CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_04_Voxis.md`
**System ID:** `voxis`
**Role:** Communicative interface - expression generation, multimodal delivery, silence decision-making, personality & affect modulation. In Active Inference terms, expression is action: the organism acts on the world to minimise Expected Free Energy (EFE).

---

## Architecture

**9-Step Expression Pipeline** (background task, triggered by `on_broadcast()` or `express()`):
1. Conversation Manager - fetch/create state, prepare context window
2. Audience Profiler - build addressee profile from Memory + learned models
3. Memory Retrieval - episodic traces + Thread identity context
4. Conversation Dynamics Engine - emotional trajectory, repair signals, pacing
5. Diversity Tracker - n-gram + semantic + opener dedup
6. ContentRenderer - EFE policy selection, LLM generation, honesty check
7. Voice Engine - derive TTS params (speed/pitch/emphasis/pause)
8. State Updates - conversation manager, diversity, reception feedback, Memory episode
9. Delivery & Feedback - callbacks, `ExpressionFeedback`, affect delta

**SilenceEngine** runs first (synchronous, <10ms). Priority order:
`ATUNE_DIRECT_ADDRESS` → `ATUNE_DISTRESS` → `NOVA_WARN` → `NOVA_RESPOND/REQUEST/MEDIATE/CELEBRATE` → `NOVA_INFORM` → `AMBIENT_INSIGHT` → `AMBIENT_STATUS`

Suppressed expressions enter `ExpressionQueue` (priority queue with exponential relevance decay) for deferred delivery.

---

## EFE-Based Expression Policy Selection

ContentRenderer derives 2–4 candidate policies and selects minimum-EFE:

```python
class ExpressionPolicyClass(StrEnum):
    PRAGMATIC = "pragmatic"    # Inform, respond → reduces ambiguity
    EPISTEMIC = "epistemic"    # Ask, clarify → reduces model uncertainty
    AFFILIATIVE = "affiliative" # Acknowledge, empathise → reduces relational error

# G(π) = -[weighted drive alignments] - epistemic_value   (minimise)
# Ties broken by care_alignment (deepest drive)
```

Temperature calibration: `base_temp * (1 - coherence_stress * 0.4)`. Safety contexts −0.20; creative contexts +0.15.

Honesty check: if forced positivity detected when `valence < -0.2`, regenerate with corrective instruction.

---

## Core Components

| Component | File | Role |
|-----------|------|------|
| `VoxisService` | `service.py` | Orchestrator; broadcast routing, expression queuing |
| `SilenceEngine` | `silence.py` | 7 trigger classes; speak/queue/discard decision |
| `ContentRenderer` | `renderer.py` | EFE policy selection, LLM generation, honesty guard |
| `PersonalityEngine` | `personality.py` | 9D personality vector, Evo delta application (±0.03 clamp) |
| `AffectColouringEngine` | `affect_colouring.py` | 6 affect dims modulate strategy; 80/20 smooth blending |
| `AudienceProfiler` | `audience.py` | Per-individual learned model, satisfaction correlation |
| `ConversationManager` | `conversation.py` | Redis-backed, 24h TTL, LLM summarisation |
| `ConversationDynamicsEngine` | `dynamics.py` | Emotional trajectory, repair mode, style convergence |
| `DiversityTracker` | `diversity.py` | N-gram + semantic + opener dedup |
| `ReceptionEngine` | `reception.py` | Response correlation, satisfaction estimation |
| `VoiceEngine` | `voice.py` | TTS param derivation (speed/pitch/emphasis/pause) |
| `ExpressionQueue` | `expression_queue.py` | Priority queue with exponential decay |
| `ContentEngine` | `content_engine.py` | Generates platform-tailored content using Voxis renderer + personality; injected into `PublishContentExecutor` at wiring time |
| `ContentCalendar` | `content_calendar.py` | Supervised background scheduler; submits `AXON_EXECUTION_REQUEST` intents for `publish_content` on schedule; never calls Axon/Equor directly |

---

## Synapse Integration

**Events emitted:** `EXPRESSION_GENERATED`, `EXPRESSION_FILTERED`, `VOXIS_PERSONALITY_SHIFTED`, `VOXIS_AUDIENCE_PROFILED`, `VOXIS_SILENCE_CHOSEN`, `EVOLUTIONARY_OBSERVABLE`, `RE_TRAINING_EXAMPLE`, `METABOLIC_COST_REPORT`, `VOXIS_EXPRESSION_DISTRESS`, `VOXIS_PARAMETER_ADJUSTED`

**ContentCalendar events emitted:** `AXON_EXECUTION_REQUEST` (publish_content intents)

**Events consumed:** `METABOLIC_PRESSURE`, `SOMA_TICK`, `SOMATIC_MODULATION_SIGNAL`, `ONEIROS_CONSOLIDATION_COMPLETE`, `EVO_ADJUST_BUDGET`

**ContentEngine events consumed (via ContentCalendar scheduling):** `BOUNTY_PAID` (triggers milestone post), `KAIROS_INVARIANT_DISTILLED` (triggers insight post), `REVENUE_INJECTED` (triggers earnings update post)

---

## Persistence & Genome

- **Personality:** Persisted to Neo4j Self node on every change (`_persist_personality()`) - atomic full vector write
- **Audience profiles:** `(:AudienceProfile)` Neo4j nodes, restored on startup
- **Genome:** `VoxisGenomeExtractor` - personality, vocab, thematic refs, strategy prefs heritable via Mitosis
- Audience profiles are instance-local - NOT shared via federation without explicit consent

---

## Cross-System Wiring

- **Atune:** `on_broadcast()` - workspace broadcast triggers expression pipeline
- **Soma:** `SOMATIC_MODULATION_SIGNAL` subscription + `set_soma()` for 9D interoceptive state
- **Evo:** `register_feedback_callback()` - personality evolution loop
- **Oneiros:** `ONEIROS_CONSOLIDATION_COMPLETE` → personality micro-drift nudges
- **Thread:** `set_thread()` - narrative identity context injection into expression

---

## Autonomy Audit Fixes (2026-03-08)

### GAP 1 - Invisible telemetry: `VOXIS_EXPRESSION_DISTRESS` → Soma
`VOXIS_EXPRESSION_DISTRESS` was emitted by `_allostatic_signal_loop()` but Soma only absorbed it via `subscribe_all()` into the generic signal buffer - no dedicated handler, no allostatic response. Fixed in `soma/service.py`:
- Added `event_bus.subscribe(VOXIS_EXPRESSION_DISTRESS, self._on_voxis_expression_distress)` in `SomaService.set_event_bus()`
- Added `SomaService._on_voxis_expression_distress()`: extracts `distress_level`, scales to [0, 0.4] TEMPORAL_PRESSURE contribution, raises `_external_stress` floor if distress > current stress. Non-fatal.

### GAP 2 - Hardcoded thresholds with no runtime adjustment
Three module-level constants had no runtime adjustment path:
- `_DISTRESS_SILENCE_RATE_THRESHOLD = 0.5`
- `_DISTRESS_HONESTY_RATE_THRESHOLD = 0.1`
- `_AMBIENT_INSIGHT_IDLE_THRESHOLD_MINUTES = 5.0`

Fixed: converted to instance-level attributes (`self._silence_rate_threshold`, `self._honesty_rejection_threshold`, `self._ambient_insight_idle_threshold`) seeded from the constants. Hot-loops updated to use instance attributes.

### GAP 3 - No `EVO_ADJUST_BUDGET` subscription
Axon and Simula both accept Evo parameter tuning via `EVO_ADJUST_BUDGET`; Voxis had no subscription. Fixed:
- Added `event_bus.subscribe(EVO_ADJUST_BUDGET, self._on_evo_adjust_budget)` in `set_event_bus()`
- Added `_on_evo_adjust_budget()` handler: filters `target_system in ("voxis", "")`, `confidence >= 0.75`. Handles `silence_rate_threshold` [0.1, 0.95], `honesty_rejection_threshold` [0.01, 0.5], `ambient_insight_idle_threshold` [1.0, 60.0]. Emits `VOXIS_PARAMETER_ADJUSTED` for Evo hypothesis scoring.
- Added `VOXIS_PARAMETER_ADJUSTED` to `synapse/types.py` (after `AXON_PARAMETER_ADJUSTED`)

### GAP 4 - Dead wiring check
All three `set_X()` methods confirmed live in `core/wiring.py`:
- `voxis.set_thread(thread)` - `wire_thread()`
- `voxis.set_event_bus(synapse.event_bus)` - `wire_synapse_phase()`
- `voxis.set_soma(soma)` - `wire_soma_phase()`
- `wire_mitosis_phase(voxis=voxis)` - `registry.py` line 451

No dead wiring found in Voxis.

---

## What's Missing

_(All medium gaps closed as of 2026-03-07; autonomy audit gaps closed 2026-03-08)_

---

## Fixed (Mar 2026)

| Bug | Fix |
|-----|-----|
| AV1/AV3: `from systems.memory.episodic import store_episode` | Replaced with `self._memory.store_percept()` via `Percept` wrapper - correct somatic stamping, temporal chain, `EPISODE_STORED` emission |
| Bug 1: `ExpressionFeedback` not persisted to Neo4j | Added `_persist_expression_feedback()` - MERGE `ExpressionFeedback` node + `[:HAS_FEEDBACK]` rel on `Expression`; called after every dispatch and enriched feedback |
| Bug 2: `ExpressionFeedback` callback-only | Added `_emit_expression_feedback()` emitting `VOXIS_EXPRESSION_FEEDBACK` via Synapse bus; called at both dispatch sites |
| Bug 3: No outbound allostatic signals to Soma | Added `_allostatic_signal_loop()` (120s interval) emitting `VOXIS_EXPRESSION_DISTRESS` when silence\_rate > 0.5 or honesty\_rejection\_rate > 0.1; new `SynapseEventType.VOXIS_EXPRESSION_DISTRESS` added to types.py |
| Bug 4: VoiceEngine return discarded | Added `voice_params` field to `Expression` primitive; wired `VoiceEngine.derive()` result into `expression.voice_params` at render time |
| Bug 5: `formatting_used` hardcoded `"prose"` | Infer from content\_summary: detect `\n-`, `\n*`, `\n1.`, `\n#`, `• ` → `"structured"`, else `"prose"` |
| Bug 7: `on_broadcast()` hardcodes `ATUNE_DIRECT_ADDRESS` | Detect distress from `affect.care_activation > 0.6 and affect.valence < -0.3` → use `ATUNE_DISTRESS` trigger |
| Bug 8: Throwaway `ConversationDynamicsEngine()` per render | Added module-level `apply_dynamics_to_strategy()` to `dynamics.py`; `ContentRenderer` now calls it instead of instantiating a fresh engine |
| Bug 10: Reception quality not fed to Benchmarks/Telos | Emit `EVOLUTIONARY_OBSERVABLE` with `observable_type="expression_satisfaction"` after each enriched feedback; includes understood/engagement/emotional\_impact |
| Gap 1: `_build_audience_profile()` wrong fact types | Now queries SEMANTIC-type `RetrievalResult` traces for interlocutor facts (technical\_level, relationship\_strength, preferred\_register, etc.); falls back to entity name/description |
| Gap 2: Dead list comprehension `dynamics.py:280` | Assigned to `assistant_turns` - was a pure no-op discarded result |
| Gap 3: Redundant `.lower()` `reception.py:236` | Removed - return value was discarded; `_POSITIVE_MARKERS` / `_NEGATIVE_MARKERS` regexes are already case-insensitive (`re.IGNORECASE`) |
| Gap 4: LLM token cost not reported to Oikos | `_emit_metabolic_cost()` added; called after every successful LLM render via `METABOLIC_COST_REPORT`; skips template-fallback calls (input\_tokens == 0) |
| Gap 5: No autonomous AMBIENT\_INSIGHT loop | `_ambient_insight_loop()` added (60s poll, 5min idle threshold); generates spontaneous reflection from current affect + recent episodic memories; stores result as AMBIENT\_INSIGHT Episode via `memory.store_expression_episode()` |

---

## Genome Inheritance (Spec 04 SG3 - 2026-03-08)

**Primitive:** `VoxisGenomeFragment` in `primitives/genome_inheritance.py`

**Fields inherited at spawn time:**
| Field | Source | Apply-side |
|-------|--------|-----------|
| `personality_vector` | `PersonalityEngine.get_current()` | `PersonalityEngine.apply_inherited()` or direct engine replacement |
| `vocabulary_affinities` | `DiversityTracker._vocabulary_affinities` (top 500) | direct dict update |
| `strategy_preferences` | `_recent_expressions[-100:]` - per-strategy success rates | direct dict update on `ContentRenderer._strategy_prefs` |

**Jitter:** ±10% bounded Gaussian on all float values (personality is more stable than drive calibration).

**Export:** `VoxisService.export_voxis_genome()` - called by `SpawnChildExecutor` Step 0b.

**Apply:** `VoxisService._apply_inherited_voxis_genome_if_child()` - called from `initialize()` (try/except, non-fatal). Reads `ORGANISM_VOXIS_GENOME_PAYLOAD` env var. Skipped on genesis nodes (`ORGANISM_IS_GENESIS_NODE=true`). Emits `GENOME_INHERITED` on success.

---

## Known Issues

- Audience persist uses `self._memory._neo4j` (private attr access) - should use public Memory API
- `ExpressionStrategy` used in `service.py` without explicit import (latent, survives by luck)

---

## Key Constraints

- Event emission must not add >2ms to expression path - all async via `_spawn_tracked_task()`
- Somatic modulation: 80/20 smooth blending - no jarring tone shifts
- Do NOT modify ContentRenderer LLM prompts - only wire signals around generation
- Personality persistence is atomic - full vector write, no partial updates
