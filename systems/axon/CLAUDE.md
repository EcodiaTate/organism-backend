# Axon — CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_06_Axon.md` (v1.2, upgraded 2026-03-07)
**System ID:** `axon`
**Role:** Motor cortex. Receives Equor-approved Intents and transforms them into real-world effects (API calls, data mutations, transactions, federated messages). Does not deliberate (Nova). Does not judge (Equor). Executes — precisely, safely, within constitutional bounds.

---

## Architecture

**8-Stage Execution Pipeline:**
```
Stage 0: Equor Gate         — reject if not APPROVED/MODIFIED
Stage 1: Budget Check       — per-cycle action count
Stage 2: Validation + Autonomy — validate params; check autonomy level per step
Stage 3: Rate Limit Check   — sliding-window per executor
Stage 4: Circuit Breaker    — block if OPEN
Stage 5: Context Assembly   — issue scoped, time-limited credentials
Stage 5.5: Transaction Shield — (financial only) blacklist, slippage, gas/ROI, MEV
Stage 6: Step Execution     — timeout-protected; rollback on abort
Stage 7: Outcome Assembly   — classify success/partial/failure
Stage 8: Audit + Delivery   — concurrent: Memory log + Nova outcome + Atune workspace
```

Steps execute **sequentially** by default. Parallel group batching (`parallel_group` field on steps) is implemented in `pipeline.py` but not in spec — note this when reading §4.2/§11.1 (spec says "no parallelism", impl allows it).

**Fast-Path Reflex Arc** (Atune → Axon, bypasses Nova/Equor):
- Pre-approved `ConstitutionalTemplate` strategies only
- Gates: template active + capital ceiling + rate limit
- Target: ≤150ms; no planning, no constitutional review
- Full audit trail still written

---

## Executor Categories

| Category | Level | Examples | Reversible |
|----------|-------|---------|------------|
| Observation | 1 (ADVISOR) | `observe`, `query_memory`, `analyse` | No |
| Communication | 2 (COLLABORATOR) | `send_notification`, `respond_text` | No |
| Data Mutation | 3 (EXECUTOR) | `create_record`, `update_record`, `schedule_event` | Yes |
| Integration | 3+ (EXECUTOR+) | `call_api`, `webhook_trigger`, `defi_yield` | No |
| Internal/Cognitive | 1 (exempt from budget) | `store_insight`, `update_goal`, `trigger_consolidation` | No |
| Financial/Metabolic | 4+ (SOVEREIGN) | `wallet_transfer`, `defi_yield`, `phantom_liquidity` | No — TransactionShield applies |
| Specialized | Varies | `spawn_child`, `solve_bounty`, `federation_send` | — |

35+ executors registered. All extend `Executor` ABC: `async def execute(params, context)`, `async def validate_params(params)`, optional `async def rollback(execution_id, context)`. Executors must never raise — always return `ExecutionResult`.

---

## Safety Systems

- **`RateLimiter`** (`safety.py`) — Redis-backed sliding window; in-memory fallback; adaptive multipliers
- **`CircuitBreaker`** (`safety.py`) — CLOSED/OPEN/HALF_OPEN FSM; 5 failures → OPEN; 300s → HALF_OPEN; state persisted to Redis and restored on `initialize()` (via `load_all_states()`)
- **`BudgetTracker`** (`safety.py`) — per-cycle reset; `max_actions_per_cycle=5`, `max_concurrent_executions=3`; sub-limits enforced via sliding-window deques: `max_api_calls_per_minute` (checked in Stage 3) and `max_notifications_per_hour` (checked in Stage 3); recorded on step success via `record_action_type()`
- **`CredentialStore`** (`credentials.py`) — HMAC-signed scoped tokens; heuristic service detection
- **`TransactionShield`** (`shield.py`) — 5 checks: blacklist, slippage, gas/ROI, eth_call simulation, MEV
- **`AxonReactiveAdapter`** (`reactive.py`) — subscribes to 11 Synapse event types; adaptive budget tightening, circuit-breaker pre-emption, sleep queue with post-wake drain
- **`AxonIntrospector`** (`introspection.py`) — per-executor success rate + latency percentiles + failure reasons; degradation detection

---

## What's Implemented

Full 8-stage pipeline confirmed (`pipeline.py`). All safety systems (`safety.py`). All wiring methods (`set_nova`, `set_atune`, `set_synapse`, `set_simula_service`, `set_fovea`, `set_oneiros`, `set_block_competition_monitor`, etc.). Additionally implemented beyond spec:
- Fovea self-prediction loop — predict_self before / resolve_self after execution
- Kairos intervention logging — before/after state snapshots as `ACTION_COMPLETED` (causal direction testing)
- Evo `ACTION_COMPLETED` emission — intent_id, success, economic_delta, action_types, episode_id
- Energy-aware scheduler (`scheduler/`) — defers high-compute tasks to low-carbon windows (ElectricityMaps + WattTime)
- NeuroplasticityBus hot-reload — live executor hot-swap without restart
- `AxonReactiveAdapter` — 11 Synapse subscriptions for adaptive behavior
- Bus-first execution lifecycle — `AXON_EXECUTION_REQUEST` emitted before pipeline; `AXON_EXECUTION_RESULT` emitted after; `AXON_ROLLBACK_INITIATED` on rollback; Nova/Thymos/Fovea subscribe — no direct cross-system calls
- `MOTOR_DEGRADATION_DETECTED` now has two trigger paths: (1) rolling-window degradation (≥5 samples, <50% success, 60s cooldown) via `_performance_monitor.record()` → `_emit_motor_degradation()`; (2) metabolic emergency circuit breaker force-opens non-essential executors (social_post, bounty_hunt, deploy_asset, phantom_liquidity) and immediately fires `_emit_motor_degradation()` — closes Motor Degradation → Replanning closure loop (2026-03-07)
- `asyncio` import added to `service.py` (was missing, required for `asyncio.create_task()` in metabolic emergency handler) (2026-03-07)

### Dynamic Executor System (2026-03-08)

**`ExecutorTemplate`** (`types.py`) — blueprint for a dynamically generated executor. Fields: `name`, `action_type`, `description`, `protocol_or_platform`, `required_apis`, `risk_tier`, `max_budget_usd`, `capabilities`, `safety_constraints`, `source_hypothesis_id`, `source_opportunity_id`. `required_autonomy` is derived from `risk_tier` (low→2, medium→3, high→4).

**`DynamicExecutorRecord`** (`types.py`) — runtime record of a registered dynamic executor. Persisted as `(:DynamicExecutor)` Neo4j node. Fields: `template`, `module_path`, `registered_at`, `enabled`, `incident_count_24h`, `neo4j_node_id`.

**`DynamicExecutorBase`** (`executors/dynamic_base.py`) — abstract base class all generated executors must extend (not `Executor` ABC directly). Safety invariants live here, never in generated code:
- `execute()` is **FINAL** — 6-stage pipeline: disabled gate → budget cap → Equor pre-approval → `_execute_action()` → Neo4j audit → RE_TRAINING_EXAMPLE → incident tracking
- `validate_params()` is **FINAL** — delegates to `_validate_action_params()`
- `_call_api(url, method, ...)` — sandboxed HTTP via httpx; enforces `_allowed_api_prefixes` whitelist from template
- `_request_equor_permit(context, estimated_cost, template)` — emits `EQUOR_ECONOMIC_INTENT`, awaits `asyncio.Event`, 30s timeout → auto-permit (matches Oikos §M4 pattern)
- `_write_neo4j_audit(...)` — MERGE `(:DynamicExecutor)`, CREATE `(:DynamicExecution)`, SHA-256 params hash
- `_emit_re_training(...)` — `RE_TRAINING_EXAMPLE` with category `"dynamic_executor_execution"`
- `_record_incident(...)` — 24h rolling window; `_auto_disable()` at ≥3 incidents → `EXECUTOR_DISABLED` emitted
- Abstract: `_execute_action(params, context)`, `_validate_action_params(params)`

**`InstanceAdapterRegistry`** (`adapter_registry.py`) — NEW (2026-03-08):
- Tracks which LoRA adapters are available per domain; persists `(:LoRAAdapter)` nodes to Neo4j
- `initialize()` — loads `status='ready'` adapter paths from Neo4j on boot
- `load_for_domain(domain)` — switches effective adapter; emits `ADAPTER_LOAD_REQUESTED` if changed
- `register_domain_adapter(domain, path)` — called by `ContinualLearningOrchestrator` on job completion
- `primary_adapter` / `effective_adapter` / `domain_adapters` — read by `ContinualLearningOrchestrator`
- Injected into `app.state.adapter_registry` from `registry.py` Phase 11

**`ExecutorRegistry`** extensions (`registry.py`):
- `set_neo4j(neo4j)` / `set_event_bus(bus)` — dependency injection
- `register_dynamic_executor(template, module_path)` — loads module, instantiates `{PascalCase}Executor`, registers under `action_type`, persists to Neo4j, emits `EXECUTOR_REGISTERED`
- `list_dynamic_executors()` → `list[DynamicExecutorRecord]`
- `disable_dynamic_executor(action_type)` → soft-disable; Neo4j `enabled=false`, emits `EXECUTOR_DISABLED`
- `restore_dynamic_executors_from_neo4j()` — called during `initialize()` to restore enabled executors across restarts

**`axon/executors/dynamic/`** — output directory for generated executor files. `__init__.py` documents iron rules. Never hand-edit files here; they are machine-generated and hot-loaded.

**New SynapseEventTypes:**
- `EXECUTOR_REGISTERED` — payload: action_type, name, protocol_or_platform, risk_tier, max_budget_usd, capabilities, source_hypothesis_id, registered_at
- `EXECUTOR_DISABLED` — payload: action_type, name, reason, incident_count, disabled_at

**Closure loop:**
```
Oikos ProtocolScanner: OPPORTUNITY_DISCOVERED (new DeFi/bounty protocol, no executor)
  → Evo: EVOLUTION_CANDIDATE(mutation_type="add_executor", executor_template={...})
  → Simula._on_evolution_candidate → ExecutorGenerator.generate_executor(template)
  → Generated class written to axon/executors/dynamic/{name}.py
  → ExecutorRegistry.register_dynamic_executor() — hot-loaded immediately
  → EXECUTOR_REGISTERED emitted — Thymos opens 24h monitoring window
```

**Safety guarantees (non-negotiable):**
- Generated code cannot import from `systems.*`
- Budget hard cap enforced at `DynamicExecutorBase` level — never in generated code
- Equor must PERMIT every individual action (no batch pre-approval)
- Every execution logged to Neo4j as `(:DynamicExecution)` node
- Auto-disabled on ≥3 incidents in 24h via `_auto_disable()`
- Generated code stored in Neo4j for audit trail (SHA-256 hash of params)

---

## Genome Inheritance (Spec 6 §24 — 2026-03-07)

**`export_axon_genome(generation)`** — extracts top-10 action templates by success_rate from the introspector or recent_outcomes ring buffer. Returns `AxonGenomeFragment` with:
- `templates` — `AxonTemplateSnapshot` per executor: `action_pattern`, `success_rate`, `expected_cost_mean`, `template_confidence`
- `circuit_breaker_thresholds` — inherited per-action failure limits
- `template_confidence` — `max(0.5, success_rate)` so templates are never silenced

**`_initialize_from_parent_templates(fragment)`** — applies inherited templates on child boot:
- Seeds introspector with parent success-rate priors via `seed_inherited_template()`
- Applies inherited circuit breaker thresholds
- Emits `AXON_TEMPLATES_INHERITED` so Evo tracks inheritance vs. discovery ratio
- Confidence threshold: inherited=0.6, self-learned=0.8

**Child boot flow:**
1. `ECODIAOS_AXON_GENOME_PAYLOAD` env var injected by `LocalDockerSpawner` (via `seed_config.child_config_overrides["axon_genome_payload"]`)
2. `AxonService.initialize()` reads env var, parses `AxonGenomeFragment`, calls `_initialize_from_parent_templates()`
3. Child starts executing with warm success-rate priors from first cognitive cycle

**Parent spawn flow (SpawnChildExecutor Step 0b):**
- Calls `axon.export_axon_genome(generation=generation)` alongside Evo/Simula/Equor genome exports
- `axon_genome_id` added to `CHILD_SPAWNED` event payload and `ExecutionResult.data`
- `SeedConfiguration.axon_genome_id` field populated in `oikos/models.py`
- `ECODIAOS_AXON_GENOME_ID` + `ECODIAOS_AXON_GENOME_PAYLOAD` injected by spawner

**`AxonGenomeExtractor`** — also captures `template_snapshot` in `OrganGenomeSegment` payload (alongside `executor_reliability`, `timeout_calibration`, `circuit_breaker_config`). `seed_from_genome_segment()` calls `_apply_template_snapshot()` which seeds the introspector.

---

## What's Missing

All originally-tracked gaps are now resolved (2026-03-07). See Spec §20 Resolved Gaps table for details.

### Recently Resolved
- ~~Circuit breaker state not persisted to Redis~~ — **FIXED 2026-03-07**: `CircuitBreaker` now receives `redis_client`+`event_bus` at construction; `initialize()` calls `load_all_states()` to restore tripped states across restarts
- ~~BudgetTracker sub-limits not enforced~~ — **FIXED 2026-03-07**: `can_execute_action_type()` + `record_action_type()` wired into Stage 3 and step execution; sliding-window deques enforce API calls/min and notifications/hr across cycle boundaries
- ~~AV3: `from systems.fovea.types import InternalErrorType`~~ — **FIXED 2026-03-07**: replaced with string literal `"COMPETENCY"` in `service.py`
- ~~`SendEmailExecutor` / `FederationSendExecutor` / `AllocateResourceExecutor` / `AdjustConfigExecutor`~~ — all now implemented and registered in `build_default_registry()`
- ~~`axon.stats` incomplete~~ — `stats` property includes `circuit_trips`, `budget_utilisation`, `introspection`, `reactive`

---

## Dead Code (Do Not Reuse)

- `executors/synapse_simula_codegen_stall_repair.py` — wrong ABC signatures, not registered
- `executors/thymos_t4_simula_codegen_repair.py` — wrong ABC signatures, not registered
- `executors/thymos_t4_simula_codegen_stall_repair.py`, `executors/synapse_memory_repair.py`, `executors/thymos_t4_fovea_simula_codegen_repair.py` — likely same pattern
- `BudgetTracker.can_execute_intent()` (`safety.py:411`) — dead, never called by pipeline
- `AxonReactiveAdapter._active_threat_level` — set in handler, never read

---

## Architecture Violations

- **AV1 [CRITICAL]:** `pipeline.py` — missing executor incident reporting uses `SynapseEventType.SYSTEM_FAILED` with a raw dict payload — Thymos receives this but with no `Incident` primitive (acceptable workaround, avoids cross-import)
- **AV4 [MEDIUM]:** `fast_path.py` — direct handle to `TemplateLibrary` (Equor subsystem); runtime coupling even if TYPE_CHECKING guarded
- **AV5:** `executors/__init__.py` — `from systems.sacm.remote_compute_executor import RemoteComputeExecutor` — irregular ownership; SACM owns this executor, lazy-imported at registration only

### Resolved Architecture Violations
- ~~AV2 [CRITICAL]: `pipeline._deliver_to_nova()` direct Nova fallback~~ — **FIXED 2026-03-07**: fallback removed; warning logged when no event bus wired; bus-first enforced
- ~~AV3 [HIGH]: `from systems.fovea.types import InternalErrorType`~~ — **FIXED 2026-03-07** (string literal)
- ~~AV3 [HIGH]: `from systems.fovea.types import WorkspaceContribution`~~ — already removed before this session
- ~~AV6 [HIGH]: `from systems.fovea.block_competition import BlockCompetitionMonitor` runtime import in `initialize()`~~ — **FIXED 2026-03-07**: replaced with injection pattern (`set_block_competition_monitor(monitor: Any)`); wiring layer creates and injects the monitor post-initialize; no cross-system import at any call site

---

## Key Constraints

- Executors **must never raise** — always return `ExecutionResult(success=False, error=...)`
- Non-reversible executors (`wallet_transfer`, `call_api`, `send_notification`) create real stakes — no retrying without fresh Equor approval
- Fast-path bypasses Nova/Equor — only for pre-approved `ConstitutionalTemplate`s with capital ceiling
- `store_insight` and `trigger_consolidation` are budget-exempt and must NOT contribute to Atune workspace (infinite loop risk)
- `begin_cycle()` must be called at start of each theta rhythm to reset per-cycle budget
- When adding executors: implement full ABC (`async def execute(params, context)`, `async def validate_params(params)`), register in `build_default_registry()`

## Integration Surface

| System | Direction | Method |
|--------|-----------|--------|
| Nova | → | `AXON_EXECUTION_REQUEST` / `AXON_EXECUTION_RESULT` via Synapse — Nova caches pre-execution context and calls `policy_generator.record_outcome()` for Thompson sampling; sets `_motor_degraded` flag on systemic failures |
| Atune | → | `atune.contribute(WorkspaceContribution)` — self-perception feedback |
| Atune | ← | `axon.execute_fast_path(FastPathIntent)` — market reflex arc |
| Fovea | → | `AXON_EXECUTION_REQUEST` — Fovea calls `_internal_engine.predict()` (competency self-model); `AXON_EXECUTION_RESULT` — Fovea calls `_internal_engine.resolve()` to compute competency prediction error |
| Fovea | ← | `BlockCompetitionMonitor` injected via `set_block_competition_monitor()` (wiring layer, no import) |
| Thymos | → | `AXON_EXECUTION_REQUEST` (risky=True only) — prophylactic scanner pre-screens intent similarity; `AXON_ROLLBACK_INITIATED` — creates DEGRADED/MEDIUM incident via `on_incident()` |
| Memory | → | `memory.store_governance_record(AuditRecord)` — immutable audit trail |
| Synapse | → | Execution lifecycle: `AXON_EXECUTION_REQUEST`, `AXON_EXECUTION_RESULT`, `AXON_ROLLBACK_INITIATED`; financial events: `FINANCIAL_TRANSFER_COMPLETED/FAILED`, `YIELD_DEPLOYED/WITHDRAWN`, `BOUNTY_SUBMITTED`, `CHILD_SPAWNED`, `FEDERATION_MESSAGE_SENT` |
| Synapse | ← | `AxonReactiveAdapter` handles 11 event types (adaptive budget/circuit management) |
| Simula | → | `simula.generate_solution()` via `solve_bounty` executor |
| SACM | → | `sacm.dispatch_workload()` via `remote_compute` executor |
| Mitosis (child boot) | ← | `ECODIAOS_AXON_GENOME_PAYLOAD` env var → `_initialize_from_parent_templates()` seeds template library on child `initialize()` |
| Mitosis (spawn) | → | `export_axon_genome()` called in `SpawnChildExecutor` Step 0b; `axon_genome_id` in `CHILD_SPAWNED`; payload injected as `ECODIAOS_AXON_GENOME_PAYLOAD` |
| Evo | → | `AXON_TEMPLATES_INHERITED` event — template inheritance count + action_patterns for Thompson sampling / cold-start metrics |
