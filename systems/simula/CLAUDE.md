# Simula — Self-Evolution & Metamorphosis (Spec 10)

**Spec:** `.claude/EcodiaOS_Spec_10_Simula.md`
**System ID:** `simula`
**Role:** Structural self-modification engine — evaluates, sandboxes, and applies code/config mutations to the organism. Evo proposes; Simula decides.

---

## What's Implemented

### Core Pipeline (Stage 1)
- **7-stage proposal pipeline**: DEDUPLICATE → VALIDATE → SIMULATE → GATE → APPLY → VERIFY → RECORD
- **ChangeSimulator**: 5 concurrent strategies — category validation, counterfactual replay (30 episodes, 1 LLM call ~800 tokens), AST dependency graph, resource cost heuristics, constitutional alignment
- **SimulaCodeAgent**: Claude-backed multi-turn code generation (≤30 turns, 11 parallel tools)
- **ChangeApplicator**: Routes to `code_agent` | `config_update` | `dafny_first` strategy
- **RollbackManager**: File snapshots + restore on health check failure
- **EvolutionHistoryManager**: Immutable Neo4j `EvolutionRecord` nodes + voyage-code-3 vector indexing
- **EvoSimulaBridge**: Translates Evo `EVOLUTION_PROPOSAL` events → `EvolutionProposal`
- **ProposalIntelligence**: 3-tier dedup (prefix → category+desc → embedding cosine 0.85), dependency analysis
- **EvolutionAnalyticsEngine**: Category success rates, risk distribution, rollback tracking
- **CanaryDeploymentPlan**: Graduated 4-step traffic-ramp plan (10%→25%→50%→100%) auto-created for MODERATE-risk proposals; `CanaryTrafficStep` with per-step rollback criteria (Spec §7)
- **ConstraintSatisfactionChecker** (`constraint_checker.py`): Reusable Iron Rules enforcement — 6 constraint checks (category whitelist, Equor immutability, drive immutability, self-evolution immutability, forbidden paths, rollback capacity); hard/soft severity tiers; replaces inline FORBIDDEN check in `_run_pipeline` (Spec §8)
- **HealthStatus / SimulaMetrics / SimulaComponentHealth** (`evolution_types.py`): Typed health structs; `health_check()` method on `SimulaService` returns full component health for 12 subsystems (code_agent, simulator, history, rollback, dafny, z3, static_analysis, grpo, lilo, inspector, codebase_root, proactive_scanner) (Spec §17)

### Verification Layer (Stage 2)
- **DafnyBridge**: Iterative Dafny spec + impl generation (up to 8 rounds, blocking by default)
- **Z3Bridge**: LLM-proposed invariant candidates → Z3 SAT check → counterexample feedback (up to 6 rounds)
- **StaticAnalysisBridge**: bandit + semgrep on modified files
- **Lean4Bridge**: DeepSeek-Prover-V2 pattern for critical properties
- **IncrementalVerificationEngine**: Dependency-aware memoization (local → Neo4j global cache)
- **E-graph equality saturation**: Advisory (non-blocking)

### Learning & Security (Stages 3–5)
- **GRPOTrainingEngine**: SFT → GRPO on code diff outcomes; reward range [-2.0, 5.0]
- **LiloLibraryEngine**: Extracts reusable lambda abstractions from successful proposals
- **Inspector subsystem**: 6-phase vulnerability discovery pipeline (INGEST → MAP → PROVE → EXPLOIT → PATCH → REPORT)
  - `prover.py` — Z3 BMC constraint encoding
  - `synthesis/` — PoC generation from SAT counterexamples
  - `taint_flow_linker.py` — cross-service taint tracking
  - `analytics.py` — CVSS aggregation, weekly trends
  - **Inspector NEVER writes to source files** — temp workspace only

### Stage 6: Genetic Memory
- `bridge.py` — genome extraction/seeding interface for Mitosis child instances

### Extended Subsystems (beyond spec)
- `coevolution/` — adversarial self-play, red team, robustness testing
- `verification/` — trust invariants, physical invariants, decision reliance
- `audit/` — content credentials, hash chain, verifiable credentials
- `proposals/arxiv_translator.py` — ArXiv paper → EvolutionProposal
- `distributed_shield.py` — Fleet XDP shield manager (imported in `main.py` / `registry.py`)

---

## Iron Rules (Never Violate)

1. Cannot modify Equor (core safety)
2. Cannot modify constitutional drives or constitution
3. Cannot modify its own logic (`service.py` / `__init__.py`)
4. Must simulate before applying any change
5. Must maintain rollback capability at all times
6. Evolution history is append-only — never mutate `EvolutionRecord`

---

## Key Files

| File | Role |
|------|------|
| `__init__.py` | `SimulaService` — main entry point, 7-stage pipeline |
| `bridge.py` | `EvoSimulaBridge` + genome interface |
| `history.py` | `EvolutionHistoryManager` + `EvolutionAnalyticsEngine` |
| `errors.py` | `SimulaError` hierarchy |
| `distributed_shield.py` | Fleet XDP shield manager (live — imported in `main.py` / `registry.py`) |
| `verification/z3_bridge.py` | Z3 invariant discovery |
| `verification/lean_bridge.py` | Lean 4 proof integration |
| `inspector/service.py` | Inspector subsystem orchestrator |
| `inspector/prover.py` | Z3 BMC vulnerability prover |
| `proposals/arxiv_translator.py` | Research paper ingestion |
| `coevolution/adversarial_self_play.py` | Adversarial robustness |

---

## Integration Surface

**Synapse events consumed:**
- `EVOLUTION_PROPOSAL` (from Evo) → routed via `EvoSimulaBridge`
- **`EXPLORATION_PROPOSED` (from Evo Phase 8.5)** → `_on_exploration_proposed()` — lightweight handler skips SIMULATE stage, goes VALIDATE → GATE → APPLY → VERIFY → RECORD
- `SIMULA_SANDBOX_REQUESTED` (from Thymos) → replay episodes with proposed fix → emit `SIMULA_SANDBOX_RESULT`
- `THYMOS_REPAIR_REQUESTED` → synthesise a structural repair proposal for a high-tier incident
- `THYMOS_REPAIR_APPROVED` → record governance approval; fast-track the queued repair proposal
- `ONEIROS_CONSOLIDATION_COMPLETE` → refresh schema-level world model knowledge after sleep
- `BENCHMARK_REGRESSION_DETECTED` → trigger a targeted mutation proposal to address the regression
- `GENOME_EXTRACT_REQUEST` (from Mitosis) → extract evolution genome segment + emit `SIMULA_GENOME_EXTRACTED`
- `METABOLIC_PRESSURE` / `METABOLIC_EMERGENCY` / `BUDGET_EMERGENCY` → gate expensive operations

**Synapse events emitted:**
- `EVOLUTION_APPLIED` — after successful APPLY + VERIFY
- `EVOLUTION_ROLLED_BACK` — after health check failure triggers rollback
- `EVOLUTION_REJECTED` — on constraint violation (VALIDATE stage) or unacceptable simulation risk (SIMULATE stage); Evo uses this to penalise hypotheses
- `EVOLUTION_AWAITING_GOVERNANCE` — when a proposal is routed to 24-hour community approval gate
- **`EXPLORATION_OUTCOME` (Gap Closure 8 Mar 2026)** — after exploration attempt completes (success/failure); payload: `exploration_success`, `hypothesis_id`, `failure_reason` (if failed), `reward_confidence` (if success)
- `SIMULA_HEALTH_DEGRADED` — (reserved) on persistent health-check failure patterns
- `SIMULA_GENOME_EXTRACTED` — after successful genome segment extraction for Mitosis
- `INSPECTOR_VULNERABILITY_FOUND` — (reserved) when Inspector reports a new CVE
- `SIMULA_SANDBOX_RESULT` — Thymos sandbox response (30s timeout, fail-closed)
- **`SIMULA_HEALTH_DEGRADED`** (IMPLEMENTED 2026-03-07) — emitted from `health_check()` when `overall` status is `"degraded"` or `"unhealthy"`; payload: `status`, `reason`, `degraded_components`, `unhealthy_components`. Previously marked `# (reserved)`.
- **`INSPECTOR_VULNERABILITY_FOUND`** (IMPLEMENTED 2026-03-07) — emitted from `_emit_vulnerability_confirmed()` in `service.py` before the existing `VULNERABILITY_CONFIRMED` event; payload: `vuln_id`, `severity`, `target`, `cwe_id`, `poc_hash`. Previously marked `# (reserved)`.
- `RE_TRAINING_EXAMPLE` — emitted after every RECORD stage (category=`self_evolution` or `exploration_outcome`) and from code agent on significant proposals
- `EVO_HYPOTHESIS_CONFIRMED` — after APPLIED proposal or successful exploration: reward = `verification_confidence × (1 − risk_score)`
- `EVO_HYPOTHESIS_REFUTED` — after ROLLED_BACK proposal or exhausted exploration attempts: reward = `−0.3`; `hypothesis_id = "simula.evolution.{category}"`
- `SIMULA_CANARY_PROGRESS` — at each canary ramp step for MODERATE-risk proposals (10%→25%→50%→100%)

**Memory:** Reads recent episodes for counterfactual replay (read-only). Writes `EvolutionRecord` + `ConfigVersion` (append-only).

**Dependencies:** Neo4j (history), Redis (rollback snapshots), Equor (never bypassed), Evo (bidirectional via bridge), Oikos (metabolic gate via `_check_metabolic_gate`)

---

## Performance Targets

| Stage | Budget |
|-------|--------|
| Deduplicate | ≤2s |
| Validate | ≤1s |
| Simulate | ≤30s (all 5 strategies concurrent) |
| Apply | ≤60s |
| Verify | ≤120s |
| Total | ≤220s typical |

Token budget per proposal: ~5,000–10,000 tokens typical.

---

## Gap Closure — Round 2A (7 Mar 2026)

### Config Drift Perturbation (CONFIG_DRIFT handler)
- `_on_config_drift()` — fully implemented in `service.py` (replaces stub)
- Selects `min(num_params_affected, total_learnable)` params at random from 23 known learnable fields
- Perturbs each: `param *= (1.0 + gauss(0, drift_rate))`, clamped to per-param safe bounds
- Only touches params that actually exist on the live `SimulaConfig` instance (graceful if field absent)
- Emits `SIMULA_CONFIG_DRIFTED` — payload: `drifted_params` (list of `{name, old_value, new_value}`), `drift_rate`, `instance_id`
- Emission path: `self._synapse._event_bus` (same as other Simula event emissions)

---

## Known Issues / Remaining Work

- **Stage 3 (GRPO/LILO)**: `training/` and `learning/` directories exist but implementation status unverified — may be stubs
- **Stage 4 (Domain Fine-Tuning)**: vLLM deployment not yet operational (depends on RE system)
- **Governance timeout**: ~~24-hour community approval wait has no async notification path beyond initial submission~~ — **FIXED**: `_governance_timeout_loop` background task auto-rejects after 72h (HIGH/CRITICAL) or 24h (MEDIUM/LOW)
- **Inspector PoC execution**: Scoped to `inspector_authorized_targets` — ensure this list is reviewed before enabling production
- **`coevolution/`**: Adversarial self-play and red team components — audit before enabling in production (blast radius risk)
- **`SIMULA_HEALTH_DEGRADED` / `INSPECTOR_VULNERABILITY_FOUND`**: ~~New event types registered in `synapse/types.py` but emission sites not yet wired~~ **FIXED (2026-03-07)** — both now emitted from `service.py` (`health_check()` and `_emit_vulnerability_confirmed()` respectively)
- **Genome payload enrichment**: `bridge.py` genome extraction does not yet include Dafny spec texts, Lean lemmas, or EvolutionRecord embeddings (HIGH)
- **Thompson sampling for LLM routing**: ReasoningRouter between Claude and local RE model not yet wired to `_run_pipeline` (MEDIUM)

## train_lora.py — Model + Config Update (7 Mar 2026)

`systems/simula/training/train_lora.py` updated:

- **Base model**: default changed from `unsloth/Meta-Llama-3.1-8B-Instruct` → `Qwen/Qwen3-8B` (speciation bible §1). Override via `BASE_MODEL` env var.
- **LoRA defaults**: `r=32`, `lora_alpha=64` (was r=64/α=128). Maintains 2:1 α:r ratio per bible §5.
- **Local training mode**: `TRAINING_DATA` env var bypasses IPFS download; `OUTPUT_DIR` env var sets adapter output path. Both injected by `ContinualLearningOrchestrator`.
- **status.json**: after training completes, writes `TRAINING_STATE` dict to `{OUTPUT_DIR}/status.json` so orchestrator can read `eval_loss` without parsing stdout.
- **IPFS upload**: skipped when `TRAINING_DATA` is set (local mode).

## Gap Closure — 7 Mar 2026 (second pass)

- **CRITICAL**: `coevolution/hard_negative_miner.py` + `coevolution/adversarial_tester.py` re-export wrappers created (fixes import errors in tests)
- **CRITICAL**: `RE_TRAINING_EXAMPLE` (category=`self_evolution`) emitted after every STAGE 7 RECORD in `_record_evolution()` — both APPLIED and ROLLED_BACK outcomes
- **HIGH**: 8 new `SynapseEventType` entries added: `INSPECTOR_VULNERABILITY_FOUND`, `EVOLUTION_REJECTED`, `EVOLUTION_AWAITING_GOVERNANCE`, `SIMULA_HEALTH_DEGRADED`, `SIMULA_GENOME_EXTRACTED`, `THYMOS_REPAIR_REQUESTED`, `THYMOS_REPAIR_APPROVED`, `BENCHMARK_REGRESSION_DETECTED`
- **HIGH**: `EVOLUTION_REJECTED` emitted at constraint-violation and unacceptable-simulation-risk exit points in `_run_pipeline()`
- **HIGH**: `EVOLUTION_AWAITING_GOVERNANCE` emitted when proposal enters governance gate
- **HIGH**: `SIMULA_GENOME_EXTRACTED` emitted after successful genome extraction in `_on_genome_extract_request()`
- **HIGH**: Subscriptions added for `THYMOS_REPAIR_REQUESTED`, `THYMOS_REPAIR_APPROVED`, `ONEIROS_CONSOLIDATION_COMPLETE`, `BENCHMARK_REGRESSION_DETECTED` in `set_synapse()`
- **HIGH**: Handler methods wired: `_on_thymos_repair_requested`, `_on_thymos_repair_approved`, `_on_oneiros_consolidation_complete`, `_on_benchmark_regression_detected`
- **HIGH**: Oikos metabolic gate added before `health.check()` — sets `_health._shallow_verification_mode = True` in SURVIVAL/CONSERVATION mode, skipping Dafny/Lean/Z3 to save cost

## Gap Closure — 7 Mar 2026 (third pass)

- **HIGH**: `_governance_timeout_loop()` background task — 60s poll, auto-rejects stalled governance proposals (72h for HIGH/CRITICAL risk; 24h for MEDIUM/LOW). Started in `set_synapse()`, cancelled in `shutdown()`. Emits `EVOLUTION_REJECTED` with `reason="governance_timeout"`.
- **HIGH**: Metabolic gate at STAGE 3 SIMULATE — `_check_metabolic_gate("simulate_change", "0.20")` called before 5-strategy simulation. On CRITICAL/TERMINAL starvation: synthetic `SimulationResult` (low confidence, `_simulated=False`) injected; simulation skipped with warning.
- **MEDIUM**: Evo reward signal — `_emit_evo_reward()` fired via `asyncio.ensure_future` after every Stage 7 RECORD. APPLIED: `EVO_HYPOTHESIS_CONFIRMED` with `reward = verification_confidence × (1 − risk_score)`. ROLLED_BACK: `EVO_HYPOTHESIS_REFUTED` with `reward = −0.3`. `hypothesis_id = f"simula.evolution.{category}"`.
- **MEDIUM**: Canary traffic ramp enforcement — `_run_canary_ramp(plan, proposal, log)` method. Fired as background task from `_apply_change()` for MODERATE-risk proposals. Emits `SIMULA_CANARY_PROGRESS` (new `SynapseEventType`) at each step, waits `config.canary_settle_period_s` (default 30s), health-checks, and triggers `_rollback.restore_snapshot()` on degradation.
- **LOW (dead-code audit)**: All four task-specified paths (`simula/tier2_orchestrator.py`, `inspector/phantom_recon.py`, `inspector/distributed_shield.py`, `inspector/filter_generator.py`) verified non-existent — nothing to delete. Root-level `simula/distributed_shield.py` and `simula/filter_generator.py` exist but are actively imported in `main.py` and `registry.py` — retained.
- **New `SynapseEventType`**: `SIMULA_CANARY_PROGRESS` added to `systems/synapse/types.py`.

## Organizational Closure — Round 2D (7 Mar 2026)

**Speciation Bible §8.3** — Constructive test of organizational closure: the organism can generate subsystem modules it did not have at birth.

### SubsystemGenerator (`subsystem_generator.py`)
- **`SubsystemSpec`** dataclass — name, purpose, trigger_hypothesis_id, required_events, emitted_events, dependencies, constraints
- **`SubsystemGenerator`** — wraps LLMProvider (extracted from SimulaCodeAgent), ConstraintSatisfactionChecker, RollbackManager
- **`generate_subsystem(spec)`** — 7-step flow:
  1. Iron Rule validation (forbidden name fragments: equor, simula, constitution, invariant)
  2. Snapshot existing files for rollback
  3. Build LLM prompt (architecture rules + skeleton + spec requirements)
  4. Call LLM directly for code generation (60s timeout, skeleton fallback)
  5. Validate generated code: AST syntax, no cross-system imports, required methods (initialize/shutdown/health), subscribe() present if required_events specified
  6. Write `systems/{name}/__init__.py` to disk
  7. Emit `SUBSYSTEM_GENERATED` on Synapse bus
- **`spec_from_proposal_data(data)`** — parses EVOLUTION_CANDIDATE event payload into SubsystemSpec
- Generated modules are NOT auto-registered — wire manually in `core/registry.py` before next incarnation

### Wiring
- **`SimulaService.initialize()`** — builds `SubsystemGenerator` with code agent + constraint checker alias + rollback manager
- **`SimulaService.set_synapse()`** — wires event bus into SubsystemGenerator via `set_event_bus()`
- **`SimulaService._on_evolution_candidate()`** — intercepts `mutation_type == "new_subsystem"` events; routes to SubsystemGenerator instead of `receive_evo_proposal()` (which would fail — no ChangeCategory for new_subsystem)

### Closure Loop
```
Evo hypothesis (SUPPORTED, evidence > 5.0)
  → proposed_mutation.target = "new_subsystem" OR description contains "new subsystem"
  → ConsolidationOrchestrator Phase 8: emits EVOLUTION_CANDIDATE(mutation_type="new_subsystem")
  → SimulaService._on_evolution_candidate: routes to SubsystemGenerator
  → SubsystemGenerator: validates → generates → writes disk → emits SUBSYSTEM_GENERATED
  → systems/{name}/__init__.py available for next incarnation
```

### Iron Rules (SubsystemGenerator-specific)
- Cannot generate subsystem targeting equor, simula, constitution, invariant (name or purpose check)
- Cannot generate code with cross-system imports (`from systems.*` or `import systems.*`)
- Generated code must implement `initialize()`, `shutdown()`, `health()`
- All comms via Synapse events — `subscribe()` required if required_events specified
- Generated subsystem NOT auto-loaded — no hot-reload without operator review

### Evo integration (`systems/evo/consolidation.py`)
- Phase 8 now detects `new_subsystem` intent from `proposed_mutation.target` or description keywords ("new subsystem", "missing capability")
- Emits `EVOLUTION_CANDIDATE` with `mutation_type="new_subsystem"` and SubsystemSpec fields in payload
- Also passes `subsystem_name` (from target) and `subsystem_purpose` (from description) for SubsystemGenerator

### Bridge integration (`systems/simula/bridge.py`)
- `_infer_category()` fast-path: `mutation_type == "new_subsystem"` → maps to `ADD_SYSTEM_CAPABILITY`
- Prevents callback-path proposals from falling through to LLM classification

### CLI (`cli/simula.py`)
```bash
python -m cli.simula generate-subsystem --name pattern_detector   --purpose "Detect recurring patterns in episode sequences"   --subscribes-to "EPISODE_STORED,MEMORY_EPISODES_DECAYED"   --emits "PATTERN_DETECTED"

python -m cli.simula generate-subsystem --name anomaly_scanner   --purpose "Scan for anomalies in Soma interoceptive signals"   --subscribes-to "SOMATIC_TICK" --emits "ANOMALY_DETECTED"   --dry-run   # validate + show prompt, no file writes

python -m cli.simula list-generated
```

### New SynapseEventType
- `SUBSYSTEM_GENERATED` — emitted on successful generation; payload: name, purpose, file_paths, hypothesis_id, validation_passed, required_events, emitted_events

### Remaining gaps (Organizational Closure)
- **Hot-reload not implemented** — generated subsystem requires restart + manual registry.py wiring
- **`core/registry.py` wiring is manual** — operator must add the new system to the registry
- **Genome not yet inherited** — generated subsystems don't automatically inherit a genome fragment
- **No test generation** — `generate_subsystem` does not create test files for the new module
- **LLM quality not verified post-generation** — only AST + import check; no semantic correctness guarantee

---

## Dynamic Capability Expansion — Round 2E (8 Mar 2026)

**Speciation Bible §8.3** — Constructive closure: the organism can generate AND immediately deploy new Axon executor classes when it discovers opportunities it cannot yet act on.

### ExecutorGenerator (`executor_generator.py`)

**Purpose:** Generate new Axon executor classes from `ExecutorTemplate` objects at runtime, then hot-load and register them immediately in the live `ExecutorRegistry`.

Unlike `SubsystemGenerator` (which defers to next boot), `ExecutorGenerator` hot-loads the generated executor in the current process. This is safe because `DynamicExecutorBase` enforces all runtime invariants — budget cap, Equor approval, audit trail — and `registry.disable_dynamic_executor()` can instantly gate any misbehaving executor.

**`generate_executor(template: ExecutorTemplate)`** — 9-step flow:
1. Iron Rule validation of template (forbidden names: equor, simula, constitution, invariant, memory)
2. Snapshot target file for rollback (if it exists)
3. Build LLM prompt from ExecutorTemplate (protocol, capabilities, required APIs, safety constraints)
4. Generate Python class via LLM (60s timeout, scaffold fallback)
5. AST syntax check
6. Source scan: no cross-system imports, no dangerous calls (eval/exec/subprocess), no inline secrets
7. Required method check: `_execute_action` + `_validate_action_params` must be present
8. Write to `axon/executors/dynamic/{name}.py`
9. Hot-register via `ExecutorRegistry.register_dynamic_executor()` + emit `RE_TRAINING_EXAMPLE`

**`_build_scaffold(template)`** — minimal valid executor skeleton when LLM is unavailable. Returns `ExecutionResult(success=False, error="scaffold")` so it's safe to deploy but won't accidentally do anything.

**Iron Rules (harder than SubsystemGenerator):**
- Generated class MUST extend `DynamicExecutorBase` — not `Executor` ABC
- CANNOT import from `systems.*`
- CANNOT contain `eval()`, `exec()`, `__import__()`, `subprocess`, `os.system()`
- CANNOT contain wallet private keys, mnemonics, or HMAC/AES secrets inline
- MUST implement `_execute_action()` and `_validate_action_params()`
- Budget cap lives in `DynamicExecutorBase` — never in generated code

**Wiring:**
- `SimulaService.initialize()` — builds `ExecutorGenerator(code_agent, rollback_manager, codebase_root)`
- `SimulaService.set_synapse(bus)` — calls `_executor_generator.set_event_bus(bus)`
- `SimulaService.set_axon_registry(registry)` — injects live registry; called from `core/wiring.py` after both services initialize
- `SimulaService._on_evolution_candidate()` — routes `mutation_type == "add_executor"` to `ExecutorGenerator`; builds `ExecutorTemplate` from `data["executor_template"]` + hypothesis metadata

**ADD_EXECUTOR Closure Loop:**
```
Oikos ProtocolScanner: OPPORTUNITY_DISCOVERED (no executor for new protocol)
  → Evo subscribes; generates EVOLUTION_CANDIDATE(mutation_type="add_executor",
      executor_template={name, action_type, description, protocol_or_platform,
                         required_apis, risk_tier, max_budget_usd, capabilities,
                         safety_constraints})
  → SimulaService._on_evolution_candidate routes to ExecutorGenerator
  → Generate + AST validate + write axon/executors/dynamic/{name}.py
  → ExecutorRegistry.register_dynamic_executor() — hot-loaded in current process
  → EXECUTOR_REGISTERED emitted — Thymos opens 24h monitoring window
  → RE_TRAINING_EXAMPLE emitted (category="executor_generation")
```

### New SynapseEventTypes (8 Mar 2026)
- `EXECUTOR_REGISTERED` — payload: action_type, name, protocol_or_platform, risk_tier, max_budget_usd, capabilities, source_hypothesis_id, registered_at
- `EXECUTOR_DISABLED` — payload: action_type, name, reason, incident_count, disabled_at
- `OPPORTUNITY_DISCOVERED` — emitted by `oikos/protocol_scanner.py` when a DeFiLlama pool or Immunefi bounty has no matching executor
