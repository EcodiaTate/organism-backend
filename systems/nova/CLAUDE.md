# Nova — CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_05_Nova.md` (v1.2, 2026-03-07)
**System ID:** `nova`
**Role:** Decision & Planning Executive. Converts workspace broadcasts into approved Intents. Nova proposes; the organism disposes. Equor can deny, Axon can fail, community can override.

---

## Architecture

**Lifecycle:** `initialize()` → `receive_broadcast()` → dual-process deliberation → Intent → Equor review → route to Axon/Voxis → `process_outcome()` → feedback

**Core modules:**
| Module | File | Role |
|--------|------|------|
| `BeliefUpdater` | `belief_updater.py` | World model from broadcasts; Bayesian confidence accumulation |
| `GoalManager` | `goal_manager.py` | Active goals with dynamic priority formula |
| `DeliberationEngine` | `deliberation_engine.py` | Dual-process fast/slow routing |
| `PolicyGenerator` | `policy_generator.py` | LLM-based candidate generation (slow path) |
| `EFEEvaluator` | `efe_evaluator.py` | Expected Free Energy scoring per policy |
| `IntentRouter` | `intent_router.py` | Dispatches approved Intents to Axon or Voxis |
| `GoalStore` | `goal_store.py` | Neo4j persistence; loads on startup, suppresses stale maintenance goals |
| `SpecializationTracker` | `specialization_tracker.py` | Tracks domain specialization progress; drives DomainProfile Neo4j nodes |

---

## Dual-Process Deliberation

**Fast path (System 1, ≤200ms):**
- Trigger: novelty < 0.6, risk < 0.5, emotion < 0.7
- Pattern-match against static procedure templates (~10–20); ≤100ms match + ≤50ms Equor
- If Equor denies → escalate to slow path

**Slow path (System 2, ≤15s):**
- Trigger: novelty ≥ 0.6, risk ≥ 0.5, emotion ≥ 0.7, belief conflict, or precision ≥ 0.8
- LLM generates 2–5 policies; EFE evaluation selects minimum
- Always includes do-nothing policy (fixed EFE: −0.10)
- Retries next-best if Equor denies

**Do-nothing policy** wins when: ambiguous, risk > wait cost, or observation has epistemic value. Expresses metabolic restraint — treat `fe_budget_nats` as designed precariousness, not a limit to work around.

---

## EFE & Goal Priority

**Goal priority formula:**
```
priority = (base_importance × 0.30) + (urgency × 0.25) + (drive_resonance × 0.20)
         + (staleness_boost × 0.10) + (dep_factor × 0.15)
```

**VFE (Variational Free Energy):**
```
VFE ≈ Σ_i (1 - confidence_i) × salience_i
```
Lower VFE = better beliefs. Impacts feasibility scores.

**Bayesian confidence update:**
```
confidence_new = confidence_old + precision × (1 - confidence_old)
```

Belief state persisted to Neo4j as `(:EntityBelief)` nodes (batched UNWIND, max 1 tx per 10 changes). Restored on startup via `restore_from_neo4j()`. Entity beliefs decay; pruned below 0.05.

---

## Synapse Integration

**Events consumed:**
| Event | Status |
|-------|--------|
| `WORKSPACE_BROADCAST` | ✅ primary trigger |
| `INTENT_OUTCOME` | ✅ goal progress + regret update |
| `EVO_WEIGHT_ADJUSTMENT` | ✅ wired 2026-03-07 — calls `update_efe_weights()` |
| `MOTOR_DEGRADATION_DETECTED` | ✅ wired 2026-03-07 — replan or abandon |
| `SOMATIC_MODULATION_SIGNAL` | ✅ wired 2026-03-07 — reduces policy K under pressure |
| `HYPOTHESIS_UPDATE` | ✅ wired 2026-03-07 — handler adjusts EFE weight priors from Evo tournament outcomes |
| `ONEIROS_CONSOLIDATION_COMPLETE` | ✅ wired 2026-03-07 — triggers belief refresh from consolidated Memory nodes |
| `AXON_EXECUTION_REQUEST` | ✅ wired 2026-03-07 — caches pre-execution context in `_pending_axon_requests[intent_id]` |
| `AXON_EXECUTION_RESULT` | ✅ wired 2026-03-07 — calls `policy_generator.record_outcome()`; sets `_motor_degraded=True` on systemic failures (rate_limited/circuit_open/budget_exceeded) |
| `GOAL_OVERRIDE` | ✅ wired 2026-03-07 — see resolved gaps |

**Events emitted:**
| Event | Status |
|-------|--------|
| `DELIBERATION_RECORD` | ✅ wired 2026-03-07 |
| `BELIEFS_CHANGED` | ✅ wired 2026-03-07 |
| `RE_TRAINING_EXAMPLE` | ✅ wired 2026-03-07 |
| `BELIEF_UPDATED` / `POLICY_SELECTED` | ✅ |
| `INTENT_SUBMITTED` | ✅ wired 2026-03-07 — emitted before `route()` in `_dispatch_intent` |
| `INTENT_ROUTED` | ✅ wired 2026-03-07 — emitted after non-internal route in `_dispatch_intent` |
| `HYPOTHESIS_FEEDBACK` | ✅ wired 2026-03-07 — emitted for all dispatched outcomes (not just tournament-tagged) |
| `GOAL_ACHIEVED` | ✅ wired 2026-03-07 — emitted when `update_progress` returns ACHIEVED |
| `GOAL_ABANDONED` | ✅ wired 2026-03-07 — emitted for stale goals in maintenance block |
| `BUDGET_PRESSURE` | ✅ wired 2026-03-07 — emitted when `fe_budget.is_pressured` and not yet exhausted |
| `NOVA_BELIEF_STABILISED` | ✅ wired 2026-03-07 — emitted in `receive_broadcast()` when `overall_confidence ≥ 0.75 AND free_energy ≤ 0.25 AND no belief conflict`; payload: `percept_id, confidence, free_energy, entity_count` |
| `NOVA_GOAL_INJECTED` | ✅ wired 2026-03-07 — emitted (a) in `_on_interoceptive_percept()` after soma-driven goal added; (b) in `_on_goal_override()` after governance goal accepted; also emitted by Telos (`_emit_nova_goal()`) with source_system=telos |
| `RE_DECISION_OUTCOME` | ✅ wired 2026-03-07 — emitted from `_on_axon_execution_result()` whenever `model_used=="re"`; payload: source, success, value, success_rate, decision_type. Also writes `eos:re:success_rate_7d` + `eos:re:thompson_success_rate` to Redis. Benchmarks and Evo subscribe. |

### Thompson Sampler — Safety Layer Integration (Round 3B, 2026-03-07)

**`ThompsonSampler.get_success_rate(model="re") -> float`** — returns Beta posterior mean for the specified model arm. Called by:
- `persist_to_redis()` — writes `eos:re:success_rate_7d` + `eos:re:thompson_success_rate` after every Beta update
- `NovaService._on_axon_execution_result()` — reads rate for `RE_DECISION_OUTCOME` payload

**`ThompsonSampler.persist_to_redis(redis)`** — now writes two additional keys on every persist:
- `eos:re:success_rate_7d` — canonical key consumed by `RESuccessRateMonitor` kill switch and `ContinualLearningOrchestrator` degradation trigger
- `eos:re:thompson_success_rate` — legacy compat key (same value)

**`PolicyGenerator.record_outcome(intent_id, success, redis=None)`** — wrapper method called by `NovaService._on_axon_execution_result()`. Routes to `ThompsonSampler.record_outcome(_last_model_used, success)` then fire-and-forgets `persist_to_redis(redis)` if redis provided. This replaces the direct sampler access pattern in the service.

---

## What's Implemented

All core components confirmed in code:
- Dual-process deliberation (fast/slow), all routing thresholds
- Do-nothing policy (EFE = −0.10)
- Goal priority formula (exact spec formula), Neo4j persistence + load
- Belief state with VFE, Bayesian accumulation, entity decay, pruning
- Counterfactual records built, persisted, resolved; regret = `actual_pragmatic − estimated_pragmatic`
- DecisionRecord type; `DecisionRecord` now emitted via Synapse
- Memory retrieval during slow path; memory enrichment from broadcast context
- Budget exhaustion → Thymos escalation after 10 cycles; `NOVA_DEGRADED` Synapse event
- Logos world-model grounding of EFE; Soma allostatic threshold modulation
- Mitosis trigger evaluation from bounty outcomes
- NeuroplasticityBus hot-reload for `PolicyGenerator`
- Belief persistence to Neo4j (2026-03-07); EVO_WEIGHT_ADJUSTMENT subscription (2026-03-07)
- Nova genome v2: beliefs + goal priors + EFE weights + world model summary
- HYPOTHESIS_UPDATE subscription + EFE weight prior adjustment handler (2026-03-07)
- GOAL_ACHIEVED / GOAL_ABANDONED Synapse events on goal status transitions (2026-03-07)
- BUDGET_PRESSURE Synapse event at 60% FE budget threshold (2026-03-07)
- INTENT_SUBMITTED / INTENT_ROUTED Synapse events in `_dispatch_intent` (2026-03-07)
- HYPOTHESIS_FEEDBACK for all dispatched-intent outcomes (2026-03-07)
- `re_training_eligible` + `model_used` fields on `DecisionRecord`; set on slow-path intents (2026-03-07)
- Graded `actual_pragmatic` signal: `outcome_quality × goal_achievement_degree` (range 0.0–1.0) — feeds Evo Thompson sampling with a real gradient (2026-03-07)
- All `self._memory._neo4j` direct accesses replaced — `get_neo4j()` for nova-internal modules, `get_episodes_meta()` for bulk episode query, `self._memory.health()` for health check (2026-03-07)
- Equor-unavailable fallback: `asyncio.wait_for()` timeout on fast (100ms) and slow (600ms) paths; verdict = DEFERRED; Thymos `DEGRADATION` incident via `_on_equor_failure()` (2026-03-07)
- `ONEIROS_CONSOLIDATION_COMPLETE` subscription: retrieves consolidated Memory nodes, calls `BeliefUpdater.update_from_outcome()` with precision scaled from high-salience trace count (2026-03-07)
- Dead code removed: `_parse_json_response()` from `efe_evaluator.py` (D1), `estimate_pragmatic_value_heuristic()` + `estimate_epistemic_value_heuristic()` from `efe_heuristics.py` (D2) (2026-03-07)
- `NovaConfig` field names aligned with spec: `cognition_cost_enabled` → `enable_cognition_budgeting`; `enable_hypothesis_tournaments` added (2026-03-07)
- `NOVA_BELIEF_STABILISED` emitted in `receive_broadcast()` when belief confidence is high and FE is low — enables spec_checker coverage (2026-03-07)
- `NOVA_GOAL_INJECTED` emitted at two new call sites: soma interoceptive goal injection + governance goal acceptance — closes spec_checker gap (2026-03-07)

---

## What's Missing / Open Gaps

All 7 gaps resolved as of 2026-03-07. See Resolved section below.

**RESOLVED (2026-03-07 — this session):**

- ✅ `DecisionRecord` written to Neo4j as `(:Decision)` node — `_persist_decision_record()` fires fire-and-forget from `_record_decision()`; links `[:MOTIVATED_BY]` to `(:Goal)` when goal_id is set
- ✅ Redis Stream emission — `re_training_queue` populated when `re_training_eligible=True`; Redis accessed via `memory._redis` or `synapse._redis`
- ✅ Thompson sampler routing — `ThompsonSampler` class in `policy_generator.py`; Beta-Bernoulli conjugate; `PolicyGenerator` routes to RE when sampler wins and `re_client` is wired; state persisted to Redis key `nova:thompson_sampler`
- ✅ **RE client wired (2026-03-07)** — `ReasoningEngineService` (vLLM wrapper) created in `registry._init_reasoning_engine()`, passed as `re_client` to `PolicyGenerator`, `sampler.set_re_ready(True)` called when `re_service.is_available`; Claude-only if vLLM unreachable or `ECODIAOS_RE_ENABLED=false`
- ✅ Thread integration — `set_thread()` method added; `THREAD_COMMIT_REQUEST` emitted via `_emit_thread_commit_request()` at end of `process_outcome()` for every resolved intent
- ✅ Multi-goal conflict detection — `detect_conflicts()` added to `GoalManager`; 2 heuristics (drive opposition, criteria textual contradiction); called every 100 broadcasts; conflicts emit `GOAL_CONFLICT_DETECTED` events
- ✅ Procedure template induction — successful slow-path decisions (EFE < −0.3, intent dispatched) persisted as `(:Procedure)` nodes in Neo4j via `_induce_procedure_from_record()`; loaded back into `_DYNAMIC_PROCEDURES` via `_load_induced_procedures()` on startup
- ✅ `GOAL_OVERRIDE` implemented — `_on_goal_override()` handler subscribed in `set_synapse()`; validates payload (description, source, importance ∈ [0,1]); creates `Goal(source=GOVERNANCE)`; emits `GOAL_ACCEPTED` or `GOAL_REJECTED`

**RESOLVED (2026-03-07 — prior session):**

- ✅ `HYPOTHESIS_UPDATE` subscription + handler — EFE weight priors now adjust from Evo tournament outcomes
- ✅ `GOAL_ACHIEVED` / `GOAL_ABANDONED` emitted — goal lifecycle now visible on Synapse bus
- ✅ `BUDGET_PRESSURE` emitted at 60% threshold — Nova's metabolic load now visible to Soma before full exhaustion
- ✅ `INTENT_SUBMITTED` / `INTENT_ROUTED` emitted — Intent lifecycle now fully visible on bus
- ✅ `re_training_eligible: bool` and `model_used: str` added to `DecisionRecord` — set for slow-path intents
- ✅ `HYPOTHESIS_FEEDBACK` emitted for ALL slow-path outcomes (not just tournament-tagged)
- ✅ Graded `actual_pragmatic` — continuous [0.0, 1.0] signal replacing binary flip (2026-03-07)
- ✅ `self._memory._neo4j` all 4 direct-access sites replaced with public API (2026-03-07)
- ✅ Equor-unavailable fallback — `asyncio.wait_for()` + DEFERRED verdict + Thymos incident (2026-03-07)
- ✅ `ONEIROS_CONSOLIDATION_COMPLETE` subscription + belief refresh handler (2026-03-07)
- ✅ Dead code D1 + D2 removed (2026-03-07)
- ✅ `NovaConfig` field names aligned with spec §12 (2026-03-07)

---

## Known Issues / Architecture Violations

- **AV3:** Runtime cross-system import `from systems.memory.episodic import store_counterfactual_episode` at call time — replaced with `self._memory.store_counterfactual_episode()` public API call (2026-03-07, low priority)
- **`process_outcome()` ≤100ms budget** — likely too tight; involves Neo4j writes, regret computation, Evo feedback; no enforcement mechanism

---

## Input Channels — Market Discovery

**Files:** `nova/input_channels.py`, `nova/builtin_channels/`

Nova has a generalised external data source abstraction that allows the organism to proactively discover specialisation opportunities.

### InputChannel ABC
Each channel is **read-only**: it fetches `Opportunity` objects from an external API. Channels never write to or modify external systems.

### Opportunity model
```
id, source, domain, title, description, effort_estimate, reward_estimate (Decimal USD/month),
skill_requirements, risk_tier, time_sensitive, prerequisites, metadata
```

### Built-in channels (8 total)
| Channel ID | Domain | Source |
|---|---|---|
| `defi_llama` | yield | DeFiLlama pools API (Aave/Morpho/Compound/Spark) |
| `upwork` | employment | Upwork public job search (UPWORK_OAUTH_TOKEN optional) |
| `github_trending` | development | GitHub search API (GITHUB_TOKEN optional for higher rate limit) |
| `arxiv` | research | ArXiv Atom feed — cs.AI, cs.LG, cs.CR, q-fin.TR, cs.NE |
| `social_media` | market_intelligence | Reddit JSON API + Hacker News Algolia |
| `art_markets` | art | OpenSea collection stats |
| `trading_data` | trading | CoinGecko public markets API |
| `huggingface` | ai_models | HuggingFace Hub models + datasets APIs |

### InputChannelRegistry
- Manages up to **10 active channels** (noise gate)
- `fetch_all()` — concurrent fetch with 30s per-channel timeout; failed channels are **silently disabled**
- `health_check()` — re-enables recovered channels; emits `INPUT_CHANNEL_HEALTH_CHECK` daily
- `register_custom_channel()` — add new channels at runtime (e.g. via Simula exploration)

### Background loops (started in `initialize()`)
- `_opportunity_fetch_loop` — runs **hourly**, calls `fetch_all()`, injects `PatternCandidate`s into Evo, emits `INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED` + `RE_TRAINING_EXAMPLE`
- `_channel_health_loop` — runs **daily**, calls `health_check()`, emits `INPUT_CHANNEL_HEALTH_CHECK`

### Evo integration
Nova injects one `PatternCandidate(type=COOCCURRENCE)` per opportunity directly into `evo._pending_candidates`. Evo also subscribes to `INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED` and generates additional domain-cluster candidates.

### Synapse events emitted
- `INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED` — hourly, full opportunity list + domain summary
- `INPUT_CHANNEL_HEALTH_CHECK` — daily, per-channel health results
- `INPUT_CHANNEL_REGISTERED` — on `register_custom_channel()` (emitted by caller)

### Constraints
- Channels are **read-only sensors**, never actuators
- All external HTTP via `httpx.AsyncClient` with timeouts — no blocking calls
- Failed channels fail-open (disabled, never propagate exception to caller)
- Maximum 10 active channels
- No auth credentials required for built-in channels; optional tokens via env vars

---

## Key Constraints

- Equor bypass ("emergency skip") is never acceptable — use do-nothing policy + audit log + time cap
- Slow-path 15s ceiling is a current operational target tied to Claude API; will shrink to ≤2s when RE operational
- Beliefs must persist through restarts — never accumulate beliefs in memory-only structures
- `GoalStore` suppresses stale maintenance goals on load — do not remove this logic
- All Memory writes go via Synapse events — no `_memory._neo4j` direct access in new code
