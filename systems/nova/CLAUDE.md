# Nova - CLAUDE.md

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
| `ActionTypeRegistry` | `action_type_registry.py` | Runtime registry of 18 static + N dynamic action types; thread-safe; feeds PolicyGenerator prompt |

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

**Do-nothing policy** wins when: ambiguous, risk > wait cost, or observation has epistemic value. Expresses metabolic restraint - treat `fe_budget_nats` as designed precariousness, not a limit to work around.

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
| `EVO_WEIGHT_ADJUSTMENT` | ✅ wired 2026-03-07 - calls `update_efe_weights()` |
| `MOTOR_DEGRADATION_DETECTED` | ✅ wired 2026-03-07 - replan or abandon |
| `SOMATIC_MODULATION_SIGNAL` | ✅ wired 2026-03-07 - reduces policy K under pressure |
| `HYPOTHESIS_UPDATE` | ✅ wired 2026-03-07 - handler adjusts EFE weight priors from Evo tournament outcomes |
| `EVO_HYPOTHESIS_CONFIRMED` | ✅ wired 2026-03-08 - `_on_hypothesis_confirmed()` raises `hypothesis_domain.{category}` belief confidence +0.05 |
| `EVO_HYPOTHESIS_REFUTED` | ✅ wired 2026-03-08 - `_on_hypothesis_refuted()` decays domain belief −0.08; high-evidence refutations (score ≥ 3.0 or contradictions ≥ 5) trigger `_immediate_deliberation()` at urgency 0.75 |
| `ONEIROS_CONSOLIDATION_COMPLETE` | ✅ wired 2026-03-07 - triggers belief refresh from consolidated Memory nodes |
| `AXON_EXECUTION_REQUEST` | ✅ wired 2026-03-07 - caches pre-execution context in `_pending_axon_requests[intent_id]` |
| `AXON_EXECUTION_RESULT` | ✅ wired 2026-03-07 - calls `policy_generator.record_outcome()`; sets `_motor_degraded=True` on systemic failures (rate_limited/circuit_open/budget_exceeded) |
| `GOAL_OVERRIDE` | ✅ wired 2026-03-07 - see resolved gaps |
| `FOVEA_INTERNAL_PREDICTION_ERROR` | ✅ wired 2026-03-08 (NOVA-ECON-1) - `_on_fovea_econ_error()` updates `economic_risk_level` belief and triggers `_immediate_deliberation()` when economic error > 0.2 |
| `REVENUE_INJECTED` | ✅ wired 2026-03-08 (NOVA-ECON-1) - `_on_revenue_change()` injects `revenue_burn_ratio` belief and reduces economic risk |
| `BOUNTY_PAID` | ✅ wired 2026-03-08 (NOVA-ECON-1) - `_on_bounty_outcome()` adjusts `bounty_success_rate` belief; failures trigger immediate deliberation |
| `YIELD_DEPLOYMENT_RESULT` | ✅ wired 2026-03-08 (NOVA-ECON-1) - `_on_yield_outcome()` updates `yield_apy_{protocol}` belief; failures trigger immediate deliberation |
| `ECONOMIC_ACTION_DEFERRED` | ✅ wired 2026-03-08 - `_on_economic_action_deferred()` updates `economic_pressure` belief; triggers `_immediate_deliberation()` with urgency proportional to starvation level (0.55 nominal → 0.95 existential) |
| `ONEIROS_THREAT_SCENARIO` | ✅ wired 2026-03-08 - `_on_oneiros_threat_scenario()` injects `threat_likelihood_{domain}` belief entity at severity-scaled confidence (critical=0.85/high=0.65/medium=0.45/low=0.25); CRITICAL severity triggers `_immediate_deliberation()` at urgency=0.8 |
| `EVO_THOMPSON_RESPONSE` | ✅ wired 2026-03-08 (arch fix) - `_on_thompson_response()` resolves the matching `asyncio.Future` in `_thompson_futures` keyed by `request_id` |
| `AXON_CAPABILITY_SNAPSHOT` | ✅ wired 2026-03-08 (autonomy audit) - `_on_axon_capability_snapshot()` caches per-executor CB/rate/success state; `is_executor_available()` + `get_executor_health()` public API for feasibility pruning |
| `AXON_INTENT_PIVOT` | ✅ wired 2026-03-08 (autonomy audit) - `_on_axon_intent_pivot()` creates pivot goal from `fallback_goal` + triggers `_immediate_deliberation()` |
| `ECONOMIC_STATE_UPDATED` | ✅ wired 2026-03-08 (autonomy audit) - `_on_economic_state_updated()` caches balance + burn rate; `get_economic_context()` returns planning-time economics (stale after 5min) |
| `ORGANISM_TELEMETRY` | ✅ wired 2026-03-08 - `_on_organism_telemetry()` caches `OrganismTelemetry` from Synapse 50-cycle broadcast; calls `DeliberationEngine.set_organism_summary()` so every slow-path LLM call appends a one-line natural-language organism state (`to_nova_summary()`) to `situation_summary` |
| `INTEROCEPTIVE_ALERT` | ✅ wired 2026-03-08 - `_on_interoceptive_alert()` triggers `_immediate_deliberation()` at urgency=0.85 on `critical` severity signals from interoception loop |
| `KAIROS_INVARIANT_DISTILLED` | ✅ wired 2026-03-08 - `_on_kairos_invariant()` fetches full `CausalInvariant` from Neo4j by `invariant_id`; stores up to 50 entries sorted by confidence desc in `self._causal_knowledge`; confidence ≥ 0.7 AND tier ≥ 2 → `upsert_entity(EntityBelief, entity_type="causal_law")`; persisted as `(:CausalKnowledge)` nodes linked to `(:Self)` via `[:KNOWS_CAUSAL_LAW]`; restored from Neo4j on startup via `_restore_causal_knowledge_from_neo4j()`; injected into slow-path LLM prompts via `get_causal_knowledge_summary()` → `DeliberationEngine._causal_laws_provider` |
| `NOVEL_ACTION_CREATED` | ✅ wired 2026-03-08 (novel action system) - `_on_novel_action_created()` calls `ActionTypeRegistry.register_dynamic()` to add the newly generated executor to the live registry; also calls `record_outcome("propose_novel_action", True)` so Thompson sampler gets a positive signal |
| `NOVA_INTENT_REQUESTED` | ✅ wired 2026-03-08 (autonomy audit) - `_on_nova_intent_requested()` injects context into belief state and fires `_immediate_deliberation()`. Universal recourse channel: any system (Phantom, Skia, Thymos) can trigger Nova deliberation without bypassing Equor. |
| `PHANTOM_PRICE_UPDATE` | ✅ wired 2026-03-08 (autonomy audit) - `_on_phantom_price_update()` updates `market_price_{pair}` belief entities. IL risk > 2% triggers `_immediate_deliberation()` at urgency 0.70. Nova's world model now includes on-chain price state. |
| `HOMEOSTASIS_ADJUSTED` | ✅ wired 2026-03-09 - `_on_homeostasis_adjusted()` injects `homeostasis_stress.<parameter>` belief entity; warn_only=False (repair fired) triggers `_immediate_deliberation()` at urgency proportional to drift magnitude. |
| `TELOS_OBJECTIVE_THREATENED` | ✅ wired 2026-03-09 - `_on_telos_objective_threatened()` injects `telos_sufficiency_threat` belief at high confidence, decays `revenue_burn_ratio`, triggers `_immediate_deliberation()` at urgency scaling with consecutive_declines (0.70 at 3, up to 0.92). |
| `NARRATIVE_COHERENCE_SHIFT` | ✅ wired 2026-03-09 - `_on_narrative_coherence_shift()` maps state labels to numeric scores, updates `narrative_coherence` belief, deprioritises low-coherence-aligned goals on significant drop, triggers `_immediate_deliberation()` in crisis state (urgency 0.85). |
| `DREAM_INSIGHT` | ✅ wired 2026-03-09 - `_on_dream_insight()` injects `dream_insight.<domain>` belief at confidence = sqrt(coherence × novelty); high-coherence (≥ 0.85) actionable insights trigger `_immediate_deliberation()` at urgency up to 0.75. |
| `CONNECTOR_REVOKED` | ✅ wired 2026-03-09 - `_on_connector_revoked()` sets `connector_availability.<platform>` belief to 0.0, abandons active goals whose description or metadata references the revoked platform, triggers `_immediate_deliberation()` at urgency 0.72. |

**Events emitted:**
| Event | Status |
|-------|--------|
| `DELIBERATION_RECORD` | ✅ wired 2026-03-07 |
| `BELIEFS_CHANGED` | ✅ wired 2026-03-07 |
| `RE_TRAINING_EXAMPLE` | ✅ wired 2026-03-07 |
| `BELIEF_UPDATED` / `POLICY_SELECTED` | ✅ |
| `INTENT_SUBMITTED` | ✅ wired 2026-03-07 - emitted before `route()` in `_dispatch_intent` |
| `INTENT_ROUTED` | ✅ wired 2026-03-07 - emitted after non-internal route in `_dispatch_intent` |
| `HYPOTHESIS_FEEDBACK` | ✅ wired 2026-03-07 - emitted for all dispatched outcomes (not just tournament-tagged) |
| `GOAL_ACHIEVED` | ✅ wired 2026-03-07 - emitted when `update_progress` returns ACHIEVED |
| `GOAL_ABANDONED` | ✅ wired 2026-03-07 - emitted for stale goals in maintenance block |
| `BUDGET_PRESSURE` | ✅ wired 2026-03-07 - emitted when `fe_budget.is_pressured` and not yet exhausted |
| `NOVA_BELIEF_STABILISED` | ✅ wired 2026-03-07 - emitted in `receive_broadcast()` when `overall_confidence ≥ 0.75 AND free_energy ≤ 0.25 AND no belief conflict`; payload: `percept_id, confidence, free_energy, entity_count` |
| `NOVA_GOAL_INJECTED` | ✅ wired 2026-03-07 - emitted (a) in `_on_interoceptive_percept()` after soma-driven goal added; (b) in `_on_goal_override()` after governance goal accepted; also emitted by Telos (`_emit_nova_goal()`) with source_system=telos |
| `RE_DECISION_OUTCOME` | ✅ wired 2026-03-07 - emitted from `_on_axon_execution_result()` whenever `model_used=="re"`; payload: source, success, value, success_rate, decision_type. Also writes `eos:re:success_rate_7d` + `eos:re:thompson_success_rate` to Redis. Benchmarks and Evo subscribe. |
| `EVO_THOMPSON_QUERY` | ✅ wired 2026-03-08 (arch fix) - emitted by `_request_thompson_weights(domain)` to ask Evo for arm weights without holding a direct tournament_engine reference; correlated by `request_id`; 2s timeout → empty dict fallback |
| `RE_TRAINING_REQUESTED` | ✅ wired 2026-03-08 - emitted from `_on_axon_execution_result()` when `_re_low_confidence_count ≥ 5` (5 consecutive RE outcomes with success_rate < 0.50). Resets counter after emitting. Triggers CLO urgent retraining with lowered 50-example threshold. |
| `ACTION_BUDGET_EXPANSION_REQUEST` | ✅ wired 2026-03-08 - emitted from `_on_axon_execution_result()` when `failure_reason == "budget_exceeded"` and `_budget_expansion_cooldown <= 0`. Requests `max_actions_per_cycle + 3` (capped at 20) for 20 cycles. Cooldown of 50 cycles prevents spam. Equor evaluates; Axon applies if approved. `_budget_expansion_cooldown` counter initialized in `__init__`. |
| `COMPUTE_BUDGET_EXPANSION_REQUEST` | ✅ wired 2026-03-08 - emitted by `_request_compute_budget_expansion()` when goal criticality → multiplier > 1.5 (critical/existential goals). Payload: request_id, goal_id, goal_criticality, requested_multiplier=2.0, duration_cycles=10. 30-cycle cooldown. Nova self-authorises ≤ 1.5. |
| `COMPUTE_BUDGET_EXPANSION_RESPONSE` | ✅ wired 2026-03-08 - `_on_compute_budget_expansion_response()` applies approved_multiplier for duration_cycles; on denial, caps at 1.5. |
| `NOVEL_ACTION_REQUESTED` | ✅ wired 2026-03-08 (novel action system) - emitted by `emit_novel_action_requested()` / `_on_propose_novel_action_step()` when `DeliberationEngine` intercepts a `propose_novel_action` step in the selected policy. Payload: `proposal_id`, `action_name`, `description`, `required_capabilities`, `expected_outcome`, `justification`, `goal_id`, `goal_description`, `urgency`, `proposed_by`, `proposed_at`. |

### Multi-Provider Registry - N-Armed Thompson Sampler (2026-03-08)

The binary Claude ↔ RE sampler has been replaced by a generalised N-armed provider registry supporting dynamic discovery, health monitoring, and ranked fallback chains.

#### ThompsonSampler (generalised)
- `_arms: dict[str, ProviderMeta]` - any number of provider arms, each with `alpha/beta/ready/cost_per_token/latency_estimate_ms/capability_tags`
- `register_arm(name, prior_alpha, prior_beta, ready, cost_per_token, latency_estimate_ms, capability_tags)` - idempotent; preserves accumulated Beta params on re-register
- `set_arm_ready(name, ready)` - gates arm in/out of sampling
- `set_re_ready(ready)` - backward-compat alias for `set_arm_ready("re", ready)`
- `sample()` - draws from all ready arms, returns winner (falls back to "claude" if none ready)
- `sample_ranked()` - returns all ready arms ranked by Beta draw (best first); used by fallback chain
- `record_outcome(model, success)` - same interface, now generic
- `get_success_rate(model="re")` - backward compat; still writes `eos:re:*` Redis keys
- `persist_to_redis(redis)` - persists all arms as `{name}_alpha` / `{name}_beta` keys; backward-compat with old two-arm format
- `load_from_redis(redis)` - restores all arms by key prefix; unknown arms in Redis are safely ignored

#### ProviderHealthMonitor
- Tracks `consecutive_failures` and `latency_ema` per arm
- On `FAILURE_THRESHOLD` (3) consecutive failures: `sampler.set_arm_ready(provider, False)` - arm removed from rotation automatically
- `on_cycle()` - every `PROBE_INTERVAL_CYCLES` (100) cycles, returns list of downed arms for caller to probe
- `re_enable(provider)` - call when a health probe succeeds; re-adds arm to rotation
- `record_call(provider, success, latency_ms)` - hot-path call from `generate_candidates()` after each provider attempt

#### PolicyGenerator - fallback chain
- `generate_candidates()` now uses `sampler.sample_ranked()` → tries arms best-first
- On per-arm failure: `health_monitor.record_call(arm, False)`, `sampler.record_outcome(arm, False)`, continues to next arm
- All arms failed: emits `REASONING_CAPABILITY_DEGRADED` via Synapse (fire-and-forget), returns `[DoNothingPolicy]`
- `set_synapse(synapse)` - wires Synapse for degraded-state emission
- `register_provider(name, client, ready, cost_per_token, latency_estimate_ms, capability_tags)` - runtime arm registration; registers both `_extra_clients[name]` and `sampler.register_arm(name)`
- `record_outcome()` now also calls `health_monitor.record_call(model, success, latency_ms=0)` so post-call quality signals feed into consecutive-failure tracking

#### New SynapseEventType
- `REASONING_CAPABILITY_DEGRADED` - all provider arms failed; organism forced to do-nothing; Thymos/Skia/Benchmarks subscribe

**`ThompsonSampler.get_success_rate(model="re") -> float`** - returns Beta posterior mean for the specified model arm. Called by:
- `persist_to_redis()` - writes `eos:re:success_rate_7d` + `eos:re:thompson_success_rate` after every Beta update
- `NovaService._on_axon_execution_result()` - reads rate for `RE_DECISION_OUTCOME` payload

**`PolicyGenerator.record_outcome(intent_id, success, redis=None)`** - wrapper called by `NovaService._on_axon_execution_result()`. Routes to `ThompsonSampler.record_outcome(_last_model_used, success)` and `ProviderHealthMonitor.record_call()`, then fire-and-forgets `persist_to_redis(redis)` if redis provided.

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
- Graded `actual_pragmatic` signal: `outcome_quality × goal_achievement_degree` (range 0.0–1.0) - feeds Evo Thompson sampling with a real gradient (2026-03-07)
- All `self._memory._neo4j` direct accesses replaced - `get_neo4j()` for nova-internal modules, `get_episodes_meta()` for bulk episode query, `self._memory.health()` for health check (2026-03-07)
- Equor-unavailable fallback: `asyncio.wait_for()` timeout on fast (100ms) and slow (600ms) paths; verdict = DEFERRED; Thymos `DEGRADATION` incident via `_on_equor_failure()` (2026-03-07)
- `ONEIROS_CONSOLIDATION_COMPLETE` subscription: retrieves consolidated Memory nodes, calls `BeliefUpdater.update_from_outcome()` with precision scaled from high-salience trace count (2026-03-07)
- Dead code removed: `_parse_json_response()` from `efe_evaluator.py` (D1), `estimate_pragmatic_value_heuristic()` + `estimate_epistemic_value_heuristic()` from `efe_heuristics.py` (D2) (2026-03-07)
- `NovaConfig` field names aligned with spec: `cognition_cost_enabled` → `enable_cognition_budgeting`; `enable_hypothesis_tournaments` added (2026-03-07)
- `NOVA_BELIEF_STABILISED` emitted in `receive_broadcast()` when belief confidence is high and FE is low - enables spec_checker coverage (2026-03-07)
- `NOVA_GOAL_INJECTED` emitted at two new call sites: soma interoceptive goal injection + governance goal acceptance - closes spec_checker gap (2026-03-07)
- **EVO_HYPOTHESIS_CONFIRMED/REFUTED (2026-03-08)**: `_on_hypothesis_confirmed()` + `_on_hypothesis_refuted()` - confirmed hypotheses raise `hypothesis_domain.{category}` belief confidence; refuted ones decay it and trigger `_immediate_deliberation()` when evidence_score ≥ 3.0 or contradicting_count ≥ 5
- **NOVA-ECON-1 (2026-03-08)**: 4 economic event subscriptions - `FOVEA_INTERNAL_PREDICTION_ERROR`, `REVENUE_INJECTED`, `BOUNTY_PAID`, `YIELD_DEPLOYMENT_RESULT` - handlers update priority belief entities and trigger `_immediate_deliberation()` within 50ms of economic signals
- **NOVA-ECON-2 (2026-03-08)**: 5 distinct economic policy templates in `_PROCEDURE_TEMPLATES` (bounty_hunting/yield_farming/cost_optimization/asset_liquidation/revenue_diversification); `PolicyGenerator.generate_economic_intent()` selects via EFE proxy scoring (not keyword matching)
- **NOVA-ECON-3 (2026-03-08)**: `BeliefUrgencyMonitor` in `belief_updater.py` - watches 7 priority belief keys; fires `_immediate_deliberation()` callback fire-and-forget when any confidence shifts >20%; wired in `initialize()` via `set_urgency_monitor()`
- **`_immediate_deliberation()` (2026-03-08)**: async method on `NovaService` - raises deliberation urgency thresholds and emits `POLICY_SELECTED` signal to Synapse bus; used by all 4 economic handlers + urgency monitor
- **Organism telemetry awareness (2026-03-08)**: `_organism_telemetry: OrganismTelemetry | None` cached on `NovaService`. `DeliberationEngine._organism_summary: str` appended to every slow-path `situation_summary` via `set_organism_summary()`. Nova now deliberates with full organism vital sign awareness - burn rate, coherence, rhythm state, emotions, health - every single slow-path decision.
- **N-armed provider registry (2026-03-08)**: `ThompsonSampler` generalised to N arms (`_arms: dict[str, ProviderMeta]`). `register_arm()` / `set_arm_ready()` / `sample_ranked()` API. `ProviderHealthMonitor` auto-disables arms on 3 consecutive failures and probes downed arms every 100 cycles. `PolicyGenerator.generate_candidates()` uses ranked fallback chain - tries best arm, falls back to next on failure, emits `REASONING_CAPABILITY_DEGRADED` if all fail. `register_provider()` for runtime arm addition (Ollama, Bedrock, re_v2, …). New `SynapseEventType.REASONING_CAPABILITY_DEGRADED` added.
- **Elastic FE budget (2026-03-08)**: `_budget_multiplier: float` scales `FreeEnergyBudget.budget_nats` per cycle based on highest-priority active goal. Multipliers: low-priority (<0.5) → 0.7, normal (0.5–0.8) → 1.0, high (≥0.8) → 1.5 (self-authorised), critical/existential → 2.0 (requires Equor via `COMPUTE_BUDGET_EXPANSION_REQUEST`). Multiplier resets to 1.0 at cycle end unless a multi-cycle Equor-approved window is active. `_fe_spent_per_goal` tracks FE cost per goal for cost accounting. New state: `_budget_multiplier`, `_budget_multiplier_cycles_remaining`, `_compute_budget_expansion_cooldown` (30-cycle), `_fe_budget_baseline_nats`. Exposed in `health()` under `elastic_fe_budget`.
- **Novel action proposal system (2026-03-08)**: `ActionTypeRegistry` in `action_type_registry.py` - thread-safe runtime registry of 18 static + N dynamic action types. Pre-populated at startup; `register_dynamic()` adds Simula-generated executors; `record_outcome()` tracks EMA success rates; `deprecate()` removes failed types. `PolicyGenerator` uses registry via `set_action_type_registry()` (falls back to static `AVAILABLE_ACTION_TYPES` if not wired). `propose_novel_action` meta-type added to prompt + registry. `DeliberationEngine` intercepts `propose_novel_action` steps in slow-path loop via `_novel_action_cb` → `_on_propose_novel_action_step()` → `emit_novel_action_requested()` → `NOVEL_ACTION_REQUESTED` on Synapse bus. On `NOVEL_ACTION_CREATED`, `_on_novel_action_created()` calls `register_dynamic()` to hot-add the new type for subsequent deliberation cycles.

---

## What's Missing / Open Gaps

### Autonomy Audit - 5 Gaps Closed (8 Mar 2026)

**AXON_CAPABILITY_SNAPSHOT subscription** - Nova was planning blind to Axon's actual state. Now subscribes to `AXON_CAPABILITY_SNAPSHOT` (every theta cycle). Caches: per-executor CB status, rate limit remaining, success rate, is_degrading. Cached in `_axon_executor_index` (fast lookup by action_type). Public API: `is_executor_available(action_type)`, `get_executor_health(action_type)`. Nova can now prune infeasible policies before wasting Equor review budget.

**Policy effectiveness self-tracking** - Nova saw binary success/failure but couldn't introspect which policy classes work for which domains. `record_policy_effectiveness(policy_name, success)` now called from `_on_axon_execution_result()`. `get_policy_effectiveness_summary()` returns per-policy success rates. Nova can weight policies by historical effectiveness during slow-path deliberation.

**Equor rejection pattern detection** - Nova couldn't detect systematic denials. `record_equor_rejection()` now called at both Equor review sites (heartbeat + spawn). `_equor_rejection_patterns` tracks pattern frequency. `is_policy_systematically_rejected(policy_name)` returns True after ≥5 identical rejections. Logged at WARNING every 3 rejections. Nova can deprioritize systematically-rejected policy classes.

**AXON_INTENT_PIVOT subscription** - Mid-execution replanning. When Axon signals a step failed with a `fallback_goal`, Nova creates a high-priority pivot goal (0.85) and triggers `_immediate_deliberation()`. Closes the binary abort/continue gap.

**Oikos economic state cache** - `ECONOMIC_STATE_UPDATED` subscription caches `liquid_balance_usd` and `burn_rate_usd_per_hour`. `get_economic_context()` returns balance, burn rate, and hours_until_depleted for policy generation. Stale after 5 minutes (returns empty dict).

### Remaining Self-Blind Spots
- **Deliberation ROI**: No tracking of whether 15s slow-path saves money vs 150ms fast-path (needs per-path cost tracking)
- **Belief accuracy**: No ground-truth validation; only precision-based confidence
- **Memory retrieval quality**: Episodes returned but not scored for relevance
- **Policy generation latency impact**: No measurement of how policy generation time correlates with outcome quality

**RESOLVED (2026-03-08 - economic intelligence):**

- ✅ **NOVA-ECON-1**: 4 economic event subscriptions wired - `FOVEA_INTERNAL_PREDICTION_ERROR` → `_on_fovea_econ_error()`, `REVENUE_INJECTED` → `_on_revenue_change()`, `BOUNTY_PAID` → `_on_bounty_outcome()`, `YIELD_DEPLOYMENT_RESULT` → `_on_yield_outcome()`. Closes 60-minute economic blind spot.
- ✅ **NOVA-ECON-2**: 5 economic policy templates added to `_PROCEDURE_TEMPLATES`; `PolicyGenerator.generate_economic_intent()` scores all 5 by EFE proxy (epistemic + pragmatic value) rather than keyword matching. Templates: bounty_hunting (55%), yield_farming (70%), cost_optimization (80%), asset_liquidation (65%), revenue_diversification (40%).
- ✅ **NOVA-ECON-3**: `BeliefUrgencyMonitor` class in `belief_updater.py` monitors 7 priority economic belief keys; >20% confidence shift triggers `_immediate_deliberation()` callback. Beliefs are now active planning inputs, not passive state.
- ✅ **EVO-NOVA-1**: `EvoService._generate_goal_from_hypothesis()` extended to include `hypothesis_statement`, `confidence`, `evidence_score`, `domain`, `thompson_arm_id`, `thompson_arm_weights` in `NOVA_GOAL_INJECTED` payload. `EvoService.get_thompson_arm_weights(domain)` public API added.
- ✅ **Tests**: `backend/tests/systems/nova/test_economic_intent.py` - 22 unit tests covering all 4 gaps.

**RESOLVED (2026-03-07 - this session):**

- ✅ `DecisionRecord` written to Neo4j as `(:Decision)` node - `_persist_decision_record()` fires fire-and-forget from `_record_decision()`; links `[:MOTIVATED_BY]` to `(:Goal)` when goal_id is set
- ✅ Redis Stream emission - `re_training_queue` populated when `re_training_eligible=True`; Redis accessed via `memory._redis` or `synapse._redis`
- ✅ Thompson sampler routing - `ThompsonSampler` class in `policy_generator.py`; Beta-Bernoulli conjugate; `PolicyGenerator` routes to RE when sampler wins and `re_client` is wired; state persisted to Redis key `nova:thompson_sampler`
- ✅ **RE client wired (2026-03-07)** - `ReasoningEngineService` (vLLM wrapper) created in `registry._init_reasoning_engine()`, passed as `re_client` to `PolicyGenerator`, `sampler.set_re_ready(True)` called when `re_service.is_available`; Claude-only if vLLM unreachable or `ECODIAOS_RE_ENABLED=false`
- ✅ Thread integration - `set_thread()` method added; `THREAD_COMMIT_REQUEST` emitted via `_emit_thread_commit_request()` at end of `process_outcome()` for every resolved intent
- ✅ Multi-goal conflict detection - `detect_conflicts()` added to `GoalManager`; 2 heuristics (drive opposition, criteria textual contradiction); called every 100 broadcasts; conflicts emit `GOAL_CONFLICT_DETECTED` events
- ✅ Procedure template induction - successful slow-path decisions (EFE < −0.3, intent dispatched) persisted as `(:Procedure)` nodes in Neo4j via `_induce_procedure_from_record()`; loaded back into `_DYNAMIC_PROCEDURES` via `_load_induced_procedures()` on startup
- ✅ `GOAL_OVERRIDE` implemented - `_on_goal_override()` handler subscribed in `set_synapse()`; validates payload (description, source, importance ∈ [0,1]); creates `Goal(source=GOVERNANCE)`; emits `GOAL_ACCEPTED` or `GOAL_REJECTED`

**RESOLVED (2026-03-07 - prior session):**

- ✅ `HYPOTHESIS_UPDATE` subscription + handler - EFE weight priors now adjust from Evo tournament outcomes
- ✅ `GOAL_ACHIEVED` / `GOAL_ABANDONED` emitted - goal lifecycle now visible on Synapse bus
- ✅ `BUDGET_PRESSURE` emitted at 60% threshold - Nova's metabolic load now visible to Soma before full exhaustion
- ✅ `INTENT_SUBMITTED` / `INTENT_ROUTED` emitted - Intent lifecycle now fully visible on bus
- ✅ `re_training_eligible: bool` and `model_used: str` added to `DecisionRecord` - set for slow-path intents
- ✅ `HYPOTHESIS_FEEDBACK` emitted for ALL slow-path outcomes (not just tournament-tagged)
- ✅ Graded `actual_pragmatic` - continuous [0.0, 1.0] signal replacing binary flip (2026-03-07)
- ✅ `self._memory._neo4j` all 4 direct-access sites replaced with public API (2026-03-07)
- ✅ Equor-unavailable fallback - `asyncio.wait_for()` + DEFERRED verdict + Thymos incident (2026-03-07)
- ✅ `ONEIROS_CONSOLIDATION_COMPLETE` subscription + belief refresh handler (2026-03-07)
- ✅ Dead code D1 + D2 removed (2026-03-07)
- ✅ `NovaConfig` field names aligned with spec §12 (2026-03-07)

---

## Known Issues / Architecture Violations

- **AV3:** Runtime cross-system import `from systems.memory.episodic import store_counterfactual_episode` at call time - replaced with `self._memory.store_counterfactual_episode()` public API call (2026-03-07, low priority)
- **`process_outcome()` ≤100ms budget** - likely too tight; involves Neo4j writes, regret computation, Evo feedback; no enforcement mechanism

## Autonomy Gap Closure - 08 March 2026

- **AV-EVO-1a RESOLVED**: Removed direct `_evo._pending_candidates.append()` in `_fetch_and_process_opportunities()`. Evo already subscribes to `INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED` and builds identical `PatternCandidate`s via `_on_opportunities_discovered()`. The direct injection was redundant and violated the no-cross-import rule.
- **AV-EVO-1b RESOLVED**: Removed direct `_evo.record_tournament_outcome()` call in `process_outcome()`. `HYPOTHESIS_FEEDBACK` payload now includes `tournament_id` + `tournament_hypothesis_id`. Evo's new `_on_hypothesis_feedback_with_tournament()` handler detects these fields and routes to `record_tournament_outcome()` internally.
- **AV-EVO-1c RESOLVED**: Replaced `_evo.run_consolidation()` with `EVO_CONSOLIDATION_REQUESTED` Synapse event. Nova emits the event; Evo subscribes via `_on_consolidation_requested()` and triggers `_run_consolidation_now()`. Nova resets FE budget via new `_on_evo_consolidation_complete()` handler (subscribes to `EVO_CONSOLIDATION_COMPLETE`). 90s safety-net timeout prevents permanent lockout if Evo is unavailable. `self._evo` reference is now fully unused - can be removed in follow-up.
- New `SynapseEventType`: `EVO_CONSOLIDATION_REQUESTED` added to `synapse/types.py`.

---

## Input Channels - Market Discovery

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
| `arxiv` | research | ArXiv Atom feed - cs.AI, cs.LG, cs.CR, q-fin.TR, cs.NE |
| `social_media` | market_intelligence | Reddit JSON API + Hacker News Algolia |
| `art_markets` | art | OpenSea collection stats |
| `trading_data` | trading | CoinGecko public markets API |
| `huggingface` | ai_models | HuggingFace Hub models + datasets APIs |

### InputChannelRegistry
- Manages up to **10 active channels** (noise gate)
- `fetch_all()` - concurrent fetch with 30s per-channel timeout; failed channels are **silently disabled**
- `health_check()` - re-enables recovered channels; emits `INPUT_CHANNEL_HEALTH_CHECK` daily
- `register_custom_channel()` - add new channels at runtime (e.g. via Simula exploration)

### Background loops (started in `initialize()`)
- `_opportunity_fetch_loop` - runs **hourly**, calls `fetch_all()`, injects `PatternCandidate`s into Evo, emits `INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED` + `RE_TRAINING_EXAMPLE`
- `_channel_health_loop` - runs **daily**, calls `health_check()`, emits `INPUT_CHANNEL_HEALTH_CHECK`

### Evo integration
Nova injects one `PatternCandidate(type=COOCCURRENCE)` per opportunity directly into `evo._pending_candidates`. Evo also subscribes to `INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED` and generates additional domain-cluster candidates.

### Synapse events emitted
- `INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED` - hourly, full opportunity list + domain summary
- `INPUT_CHANNEL_HEALTH_CHECK` - daily, per-channel health results
- `INPUT_CHANNEL_REGISTERED` - on `register_custom_channel()` (emitted by caller)

### Constraints
- Channels are **read-only sensors**, never actuators
- All external HTTP via `httpx.AsyncClient` with timeouts - no blocking calls
- Failed channels fail-open (disabled, never propagate exception to caller)
- Maximum 10 active channels
- No auth credentials required for built-in channels; optional tokens via env vars

---

## Proactive Opportunity Scanner (9 Mar 2026)

**File:** `nova/opportunity_scanner.py`

Distinct from the passive hourly `_opportunity_fetch_loop` (InputChannels → Evo PatternCandidates). The `OpportunityScanner` actively ranks opportunities against constitutional drives and makes goal injection decisions.

### Sub-scanners

| Scanner | Source | Notes |
|---------|--------|-------|
| `BountyOpportunityScanner` | Redis `oikos:bounty:candidates` | Falls back to GitHub public search API |
| `YieldOpportunityScanner` | DeFiLlama `/pools` API | Base + Ethereum chains; only surfaces if APY improvement ≥ 20pp vs current portfolio |
| `LearningOpportunityScanner` | ArXiv cs.AI feed + GitHub trending + HackerNews | Classifies into 8 capability gap categories |
| `PartnershipOpportunityScanner` | GitHub issue search (labels: ai-agent, collaboration) | |
| `MarketTimingScanner` | Snapshot.org GraphQL + BaseScan gas oracle | Only surfaces governance votes with <72h deadline |

### Key thresholds

| Constant | Value | Purpose |
|----------|-------|---------|
| `AUTO_GOAL_MIN_CONFIDENCE` | 0.80 | Min confidence for auto-goal injection |
| `AUTO_GOAL_MIN_ROI` | 3.0 | Min ROI multiple for auto-goal injection |
| `MIN_COMPOSITE_SCORE` | 0.15 | Filter noise before emitting `OPPORTUNITY_DETECTED` |
| `YIELD_IMPROVEMENT_THRESHOLD_PCT` | 20.0 | Only surface yield opportunities if APY improvement > 20pp |
| `MAX_BACKLOG_SIZE` | 50 | Max opportunity backlog size |
| `MIN_LEARNING_RELEVANCE` | 0.30 | Min relevance score to emit `LEARNING_OPPORTUNITY_DETECTED` |

### Deduplication

`nova:scanner:seen_ids` Redis SET with 7-day TTL. IDs are `sha256(namespace:key)[:16]`.

### Goal injection logic

```
if confidence >= 0.80 AND roi >= 3.0 AND composite_score >= 0.5:
    → create Goal(source=SELF_GENERATED, status=ACTIVE)
    → emit NOVA_GOAL_INJECTED + OPPORTUNITY_DETECTED(auto_goal=True)
else:
    → append to _opportunity_backlog (capped at 50)
    → emit OPPORTUNITY_DETECTED(auto_goal=False)
```

Backlog is surfaced via `get_opportunity_backlog_summary()` - injected as deliberation context by slow-path planning.

### Learning resources

Learning resources from `LearningOpportunityScanner` bypass the goal system entirely:
- Emitted as `LEARNING_OPPORTUNITY_DETECTED` (not `OPPORTUNITY_DETECTED`)
- Evo subscribes → creates `PatternCandidate(type=COOCCURRENCE)` → feeds hypothesis engine
- Simula subscribes → queues `ADD_SYSTEM_CAPABILITY` proposal for `code_generation`/`formal_verification`/`self_evolution` domains

### Background loop

`_opportunity_scan_loop()` in `NovaService` - 30-minute cycle (immediate first scan). Started as `asyncio.create_task()` in `initialize()`. Re-triggered on `REVENUE_INJECTED` (amount > $5) and `DOMAIN_MASTERY_DETECTED`.

### Infrastructure wiring (registry.py)

After `_init_nova()`, registry wires:
```python
nova._opportunity_scanner.set_redis(infra.redis)
nova._opportunity_scanner.set_basescan_api_key(os.environ.get("ECODIAOS_BASESCAN_API_KEY", ""))
```

### Events

| Event | Direction | Notes |
|-------|-----------|-------|
| `OPPORTUNITY_DETECTED` | Emitted | Per non-learning opportunity above `MIN_COMPOSITE_SCORE` |
| `LEARNING_OPPORTUNITY_DETECTED` | Emitted | Per learning resource above `MIN_LEARNING_RELEVANCE` |
| `OPPORTUNITY_DETECTED` | Consumed | Federated opportunities from other instances (`source_system != "nova"`) |

---

## Genome Inheritance (Spec 05 SG3 - 2026-03-08 / extended 2026-03-09)

**Primitive:** `NovaGenomeFragment` in `primitives/genome_inheritance.py`

**Fields inherited at spawn time:**
| Field | Source | Apply-side |
|-------|--------|-----------|
| `goal_domain_priors` | `GoalManager._domain_weights` (top 20 by weight) | `GoalManager.seed_domain_weights()` or direct dict update |
| `policy_success_rates` | `_decision_records[-200:]` - per-policy-class success ratio | `PolicyGenerator.seed_success_rates()` |
| `belief_urgency_thresholds` | `BeliefUrgencyMonitor._thresholds` | direct dict update |
| `active_inference_params` | EFE evaluator weights (epistemic/pragmatic/affiliative) | direct dict update on `EFEEvaluator._weights` |
| `thompson_arm_history` | `PolicyGenerator._sampler._arms` (≥10 real trials per arm) | `ThompsonSampler` arm priors seeded directly; 0.85 alpha discount for disabled/failed arms |

**Jitter:** ±15% bounded Gaussian on all float values (same pattern as Telos).

**Export:** `NovaService.export_nova_genome()` - called by `SpawnChildExecutor` Step 0b.
- `thompson_arm_history` extracted from `sampler._arms`: `alpha`, `beta`, `total_trials` (= alpha+beta−2), `consecutive_failures`, `ready`
- Arms with < 10 real trials are excluded (too little signal to inherit)

**Apply:** `NovaService._apply_inherited_nova_genome_if_child()` - called from `initialize()` (try/except, non-fatal). Reads `ECODIAOS_NOVA_GENOME_PAYLOAD` env var. Skipped on genesis nodes (`ECODIAOS_IS_GENESIS_NODE=true`). Emits `GENOME_INHERITED` on success.
- Thompson arm seeding: if arm already registered → overwrite alpha/beta; if not → `sampler.register_arm()` with inherited priors
- Arms that were `ready=False` or had `consecutive_failures >= 3` in parent get `alpha *= 0.85` discount - child inherits partial skepticism, not full failure
- Non-fatal per-arm: exceptions silently skipped; `thompson_arms_seeded` count logged

---

## Key Constraints

- Equor bypass ("emergency skip") is never acceptable - use do-nothing policy + audit log + time cap
- Slow-path 15s ceiling is a current operational target tied to Claude API; will shrink to ≤2s when RE operational
- Beliefs must persist through restarts - never accumulate beliefs in memory-only structures
- `GoalStore` suppresses stale maintenance goals on load - do not remove this logic
- All Memory writes go via Synapse events - no `_memory._neo4j` direct access in new code
