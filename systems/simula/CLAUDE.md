# Simula - Self-Evolution & Metamorphosis (Spec 10)

**Spec:** `.claude/EcodiaOS_Spec_10_Simula.md`
**System ID:** `simula`
**Role:** Structural self-modification engine - evaluates, sandboxes, and applies code/config mutations to the organism. Evo proposes; Simula decides.

---

## What's Implemented

### Core Pipeline (Stage 1)
- **7-stage proposal pipeline**: DEDUPLICATE → VALIDATE → SIMULATE → GATE → APPLY → VERIFY → RECORD
- **ChangeSimulator**: 5 concurrent strategies - category validation, counterfactual replay (30 episodes, 1 LLM call ~800 tokens), AST dependency graph, resource cost heuristics, constitutional alignment
- **SimulaCodeAgent**: Claude-backed multi-turn code generation (≤30 turns, 11 parallel tools)
- **ChangeApplicator**: Routes to `code_agent` | `config_update` | `dafny_first` strategy
- **RollbackManager**: File snapshots + restore on health check failure
- **EvolutionHistoryManager**: Immutable Neo4j `EvolutionRecord` nodes + voyage-code-3 vector indexing
- **EvoSimulaBridge**: Translates Evo `EVOLUTION_PROPOSAL` events → `EvolutionProposal`
- **ProposalIntelligence**: 3-tier dedup (prefix → category+desc → embedding cosine 0.85), dependency analysis
- **EvolutionAnalyticsEngine**: Category success rates, risk distribution, rollback tracking
- **CanaryDeploymentPlan**: Graduated 4-step traffic-ramp plan (10%→25%→50%→100%) auto-created for MODERATE-risk proposals; `CanaryTrafficStep` with per-step rollback criteria (Spec §7)
- **ConstraintSatisfactionChecker** (`constraint_checker.py`): Reusable Iron Rules enforcement - 6 constraint checks (category whitelist, Equor immutability, drive immutability, self-evolution immutability, forbidden paths, rollback capacity); hard/soft severity tiers; replaces inline FORBIDDEN check in `_run_pipeline` (Spec §8)
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
  - `prover.py` - Z3 BMC constraint encoding
  - `synthesis/` - PoC generation from SAT counterexamples
  - `taint_flow_linker.py` - cross-service taint tracking
  - `analytics.py` - CVSS aggregation, weekly trends
  - **Inspector NEVER writes to source files** - temp workspace only

### Stage 6: Genetic Memory
- `bridge.py` - genome extraction/seeding interface for Mitosis child instances

### Extended Subsystems (beyond spec)
- `coevolution/` - adversarial self-play, red team, robustness testing
- `verification/` - trust invariants, physical invariants, decision reliance
- `audit/` - content credentials, hash chain, verifiable credentials
- `proposals/arxiv_translator.py` - ArXiv paper → EvolutionProposal
- `distributed_shield.py` - Fleet XDP shield manager (imported in `main.py` / `registry.py`)

---

## Iron Rules (Never Violate)

1. Cannot modify Equor (core safety)
2. Cannot modify constitutional drives or constitution
3. Cannot modify its own logic (`service.py` / `__init__.py`)
4. Must simulate before applying any change
5. Must maintain rollback capability at all times
6. Evolution history is append-only - never mutate `EvolutionRecord`

---

## Key Files

| File | Role |
|------|------|
| `__init__.py` | `SimulaService` - main entry point, 7-stage pipeline |
| `bridge.py` | `EvoSimulaBridge` + genome interface |
| `history.py` | `EvolutionHistoryManager` + `EvolutionAnalyticsEngine` |
| `errors.py` | `SimulaError` hierarchy |
| `distributed_shield.py` | Fleet XDP shield manager (live - imported in `main.py` / `registry.py`) |
| `verification/z3_bridge.py` | Z3 invariant discovery |
| `verification/lean_bridge.py` | Lean 4 proof integration |
| `inspector/service.py` | Inspector subsystem orchestrator |
| `inspector/prover.py` | Z3 BMC vulnerability prover |
| `proposals/arxiv_translator.py` | Research paper ingestion |
| `coevolution/adversarial_self_play.py` | Adversarial robustness |

---

## Integration Surface

**Synapse events consumed:**
- `CRASH_PATTERN_CONFIRMED` (from Thymos/Kairos, 9 Mar 2026) → `_on_crash_pattern_confirmed()` - learns fatal patterns, gates future patches
- `KAIROS_INVARIANT_DISTILLED` (from Kairos, extended 9 Mar 2026) → `_on_kairos_invariant_for_crash_patterns()` - extracts crash-pattern-tagged invariants
- `EVOLUTION_PROPOSAL` (from Evo) → routed via `EvoSimulaBridge`
- **`EXPLORATION_PROPOSED` (from Evo Phase 8.5)** → `_on_exploration_proposed()` - lightweight handler skips SIMULATE stage, goes VALIDATE → GATE → APPLY → VERIFY → RECORD
- `SIMULA_SANDBOX_REQUESTED` (from Thymos) → replay episodes with proposed fix → emit `SIMULA_SANDBOX_RESULT`
- `THYMOS_REPAIR_REQUESTED` → synthesise a structural repair proposal for a high-tier incident
- `THYMOS_REPAIR_APPROVED` → record governance approval; fast-track the queued repair proposal
- `ONEIROS_CONSOLIDATION_COMPLETE` → refresh schema-level world model knowledge after sleep
- `BENCHMARK_REGRESSION_DETECTED` → trigger a targeted mutation proposal to address the regression
- `GENOME_EXTRACT_REQUEST` (from Mitosis) → extract evolution genome segment + emit `SIMULA_GENOME_EXTRACTED`
- `METABOLIC_PRESSURE` / `METABOLIC_EMERGENCY` / `BUDGET_EMERGENCY` → gate expensive operations
- **`SPEC_DRAFTED` (9 Mar 2026)** → `_on_spec_drafted()` - stores spec metadata in `_pending_spec_drafts[proposal_id]`; when `NOVEL_ACTION_REQUESTED` fires with `pipeline_managed=True` for the same `proposal_id`, routes to `SubsystemGenerator` instead of `ExecutorGenerator`

**Synapse events emitted:**
- `RE_TRAINING_EXAMPLE(category="pattern_blocked", outcome_quality=0.0)` - emitted when a patch is blocked by `_apply_patch_risk_gate()` matching a known fatal pattern at score ≥ 0.70
- `THYMOS_REPAIR_REQUESTED(escalation_reason="known_fatal_pattern")` - emitted alongside the blocked RE training example so Thymos can route to a higher tier
- `INCIDENT_DETECTED(severity=low)` - emitted by `_proactive_pattern_scan()` for each generated-code Memory trace that matches a known pattern at score ≥ 0.50
- `EVOLUTION_APPLIED` - after successful APPLY + VERIFY
- `EVOLUTION_ROLLED_BACK` - after health check failure triggers rollback
- `EVOLUTION_REJECTED` - on constraint violation (VALIDATE stage) or unacceptable simulation risk (SIMULATE stage); Evo uses this to penalise hypotheses
- `EVOLUTION_AWAITING_GOVERNANCE` - when a proposal is routed to 24-hour community approval gate
- **`EXPLORATION_OUTCOME` (Gap Closure 8 Mar 2026)** - after exploration attempt completes (success/failure); payload: `exploration_success`, `hypothesis_id`, `failure_reason` (if failed), `reward_confidence` (if success)
- `SIMULA_HEALTH_DEGRADED` - emitted from `health_check()` when `overall` status is `"degraded"` or `"unhealthy"`
- `SIMULA_GENOME_EXTRACTED` - after successful genome segment extraction for Mitosis
- `INSPECTOR_VULNERABILITY_FOUND` - emitted from `_emit_vulnerability_confirmed()` before `VULNERABILITY_CONFIRMED`; payload: `vuln_id`, `severity`, `target`, `cwe_id`, `poc_hash`
- `SIMULA_SANDBOX_RESULT` - Thymos sandbox response (30s timeout, fail-closed)
- `RE_TRAINING_EXAMPLE` - emitted after every RECORD stage (category=`self_evolution` or `exploration_outcome`) and from code agent on significant proposals
- `EVO_HYPOTHESIS_CONFIRMED` - after APPLIED proposal or successful exploration: reward = `verification_confidence × (1 − risk_score)`
- `EVO_HYPOTHESIS_REFUTED` - after ROLLED_BACK proposal or exhausted exploration attempts: reward = `−0.3`; `hypothesis_id = "simula.evolution.{category}"`
- `SIMULA_CANARY_PROGRESS` - at each canary ramp step for MODERATE-risk proposals (10%→25%→50%→100%)
- **`PROOF_FOUND` / `PROOF_FAILED` / `PROOF_TIMEOUT` (9 Mar 2026)** - emitted via `_emit_proof_events_from_health()` after every `health.check()` call (both primary pipeline and repair re-check paths); extracted from `HealthCheckResult.formal_verification` (Dafny/Z3) and `HealthCheckResult.lean_verification`. Includes RE training trace per event. Previously defined but never called - now wired.

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

## Gap Closure - Round 2A (7 Mar 2026)

### Config Drift Perturbation (CONFIG_DRIFT handler)
- `_on_config_drift()` - fully implemented in `service.py` (replaces stub)
- Selects `min(num_params_affected, total_learnable)` params at random from 34 learnable fields (21 float + 13 int)
- Perturbs each: `param *= (1.0 + gauss(0, drift_rate))`, clamped to per-param safe bounds
- Only touches params that actually exist on the live `SimulaConfig` instance (graceful if field absent)
- Emits `SIMULA_CONFIG_DRIFTED` - payload: `drifted_params` (list of `{name, old_value, new_value}`), `drift_rate`, `instance_id`
- Emission path: `self._synapse._event_bus` (same as other Simula event emissions)

---

## Autonomy Audit - 8 Mar 2026

### Dead Wiring Fixed

**CRITICAL: `simula.set_evo(evo)` never called from `wiring.py`**
- `_validate_against_learned_repairs()` checks `if self._evo is not None` before querying Evo's procedural hypothesis engine for known failure patterns
- Without this call, every Stage 1 VALIDATE silently skips repair-pattern validation - proposals touching known-failure surfaces were never checked against learned fixes
- **Fix:** `wiring.py` now calls `simula.set_evo(evo)` alongside `simula.set_synapse()` and `simula.set_axon_registry()`

**HIGH: `simula.set_benchmarks(benchmarks)` never called from `registry.py`**
- Method exists and sets `self._benchmarks` (legacy KPI push path); comment says "KPI now via bus" but the setter was still a dead wiring point
- **Fix:** `registry.py` late-phase wiring block now calls `simula.set_benchmarks(app.state.benchmarks)` alongside `oneiros.set_benchmarks()` and `soma.set_benchmarks()`

### Invisible Telemetry Fixed

**HIGH: Repair-pattern mismatch advisory never emitted to Synapse bus**
- `_validate_against_learned_repairs()` found high-confidence mismatches (proposal touches known-failure endpoint without known fix) and only logged them
- Evo could not penalise the source hypotheses; Thymos could not track recurring blind spots
- **Fix:** `_validate_against_learned_repairs()` now emits `SIMULA_VALIDATION_ADVISORY` (new `SynapseEventType`) via fire-and-forget `asyncio.ensure_future` whenever missing fixes are detected. Payload: `proposal_id`, `endpoints`, `flagged_hypothesis_count`, `missing_count`, `high_confidence_count`, `missing_fix_summaries`. Advisory - never blocks the pipeline.
- **New `SynapseEventType`:** `SIMULA_VALIDATION_ADVISORY` added to `systems/synapse/types.py`

## General-Purpose Contractor - Phase 16s (9 Mar 2026)

EOS is no longer limited to evolving its own codebase. Simula can now operate as a **general-purpose software contractor** against arbitrary external GitHub repositories.

### New files
- **`simula/external_workspace.py`** - `ExternalWorkspace` class: clone → language detection → test/lint → cleanup
  - Isolated to `/tmp/eos_workspace_{task_id}_{token}/repo`
  - Path boundary enforcement via `assert_write_allowed()` (raises `ExternalWorkspaceWriteViolation`)
  - Forbidden infrastructure files: `.github/workflows`, `Makefile`, build configs, lock files
  - Language auto-detection: Rust > Solidity > Go > TypeScript > JavaScript > Python (ordered sentinel check)
  - Language-specific commands: `pytest`/`ruff` (Python), `cargo test`/`clippy` (Rust), `npm test`/`eslint` (JS/TS), `forge test`/`solhint` (Solidity), `go test ./...`/`golangci-lint` (Go)
  - `ExternalRepoConfig` dataclass: `repo_url`, `base_branch`, `target_files`, `forbidden_paths`, `test_command`, `lint_command`, `language`, `max_repair_attempts`, `clone_depth`
  - Token injection for private repos: GitHub token spliced into HTTPS URL, never logged
  - Async context manager (`__aenter__`/`__aexit__`) - cleanup on exit

### Extended: `SimulaCodeAgent`
- **`set_external_workspace(workspace)`** - redirects `self._root` to workspace root; stores workspace reference
- **`clear_external_workspace()`** - restores internal EOS root
- **`_check_forbidden_path()`** - routes through `workspace.assert_write_allowed()` when in external mode (replaces EOS `FORBIDDEN_WRITE_PATHS` check)
- **`_tool_run_tests()`** - calls `workspace.run_tests()` (language-aware) instead of hardcoded `pytest`
- **`_tool_run_linter()`** - calls `workspace.run_linter()` (language-aware) instead of hardcoded `ruff`
- **`_build_system_prompt()`** - injects `## External Repository Mode` section with language, repo URL, scope, forbidden paths when workspace is active
- **`implement_external(issue_description, workspace)`** - new entry point; synthesizes `EvolutionProposal(source="external_contractor")`, calls `implement()`, ensures workspace cleanup via `finally`

### Integration Notes
- `implement_external()` reuses the full 11-tool agentic loop - the agent reads files, writes files, lists dirs, searches code, runs tests and linter, all within the workspace boundary
- The system prompt explicitly tells the agent it is in a foreign codebase and NOT to add EOS-specific imports or Synapse calls
- `_validate_path()` still enforces the boundary (same logic, different root)

---

## Build-Error RE Training Signal - 9 Mar 2026

Every time Simula generates code that fails a build check, a `RE_TRAINING_EXAMPLE`
with `outcome_quality=0.0` and `category="build_error"` is emitted on the Synapse
bus. This teaches the RE model what **not** to generate.

### Schema (`BuildErrorTrainingSignal` - embedded in RETrainingExample)

| Field | Type | Description |
|-------|------|-------------|
| `generated_code` | str (≤4000 chars) | The code that failed |
| `instruction` / `prompt_used` | str (≤2000 chars) | What was sent to RE |
| `error_type` | Literal | `syntax` / `import` / `runtime` / `verification_timeout` / `proof_failed` / `sandbox_escape` |
| `error_message` | str (≤1000 chars) | The error text |
| `error_traceback` | str \| None (≤1500 chars) | Full traceback when available |
| `strategy_used` | str | `dafny` / `z3` / `lean` / `static` / `symbolic` / `egraph` / `health_check_phase*` / `subsystem_generation` / `executor_generation` |
| `category` | `"build_error"` | Fixed - consumed by RETrainingExporter |
| `outcome_quality` | `0.0` | Fixed - always zero for build errors |
| `lesson` | str | Human-readable summary of what not to do (in `reasoning_trace`) |

### Capture Points

**`health.py` - HealthChecker.check()** (after `set_synapse(synapse)` wired by `service.py`):
- Phase 1 syntax failure: `_check_syntax` returns errors → `error_type="syntax"`
- Phase 2 import failure: `_check_imports` returns errors → `error_type="import"`
- Phase 3 test timeout → `error_type="verification_timeout"`, `strategy_used="health_check_phase3_pytest"`
- Phase 3 test failure → `error_type="runtime"`, `strategy_used="health_check_phase3_pytest"`, full pytest output in `error_traceback`
- Phase 4 formal verification blocking failure → `error_type="proof_failed"`, `strategy_used` = inferred from which bridge failed (`dafny`/`z3`/`static`)
- Phase 5 Lean blocking failure → `error_type="proof_failed"`, `strategy_used="lean"`
- Phase 6 formal guarantees blocking failure → `error_type="proof_failed"`, `strategy_used` = `symbolic` or `egraph`

**`subsystem_generator.py` - SubsystemGenerator.generate_subsystem()**:
- Code generation exception (LLM/prompt failure) → `error_type="runtime"`, full traceback, `generated_code=""`
- `_validate_generated_code` failure → `error_type` classified from error list (`syntax` / `import` / `runtime`)

**`executor_generator.py` - ExecutorGenerator.generate_executor()**:
- Code generation exception → `error_type="runtime"`, full traceback, `generated_code=""`
- `_validate_generated_code` failure → `error_type` classified (`syntax` / `import` / `sandbox_escape` / `runtime`)

### Wiring

- `HealthChecker.set_synapse(synapse)` called from `SimulaService.set_synapse()` at the same point where `SubsystemGenerator` and `ExecutorGenerator` get their event bus wired.
- `SubsystemGenerator._emit_build_error()` uses `self._event_bus` (already injected).
- `ExecutorGenerator._emit_build_error()` uses `self._event_bus` (already injected).
- All emission paths: `asyncio.create_task(_emit())` - never block the error handling flow. Wrapped in `try/except RuntimeError` for contexts without a running loop.

---

## Fatal Pattern Memory - 9 Mar 2026

Simula now maintains an in-process dictionary of confirmed fatal crash patterns and uses
it to gate every generated patch before execution.

### New state
- `self._known_fatal_patterns: dict[str, CrashPattern]` - populated on boot from
  `CRASH_PATTERN_CONFIRMED` events; persists across events until restart.
- `self._pattern_scan_task` - background asyncio task running every 2 hours.

### New subscriptions (wired in `set_synapse()`)
| Event | Handler |
|-------|---------|
| `CRASH_PATTERN_CONFIRMED` (new) | `_on_crash_pattern_confirmed()` |
| `KAIROS_INVARIANT_DISTILLED` | `_on_kairos_invariant_for_crash_patterns()` (existing subscription reused; crash-pattern-tagged invariants only) |

### New SynapseEventType
- `CRASH_PATTERN_CONFIRMED` - added to `systems/synapse/types.py`. Emitted by Thymos
  after all repair tiers fail, or synthesised from Kairos when an invariant is tagged
  `invariant_type="crash_pattern"`. Payload: `pattern_id`, `signature`, `description`,
  `confidence`, `failed_tiers`, `lesson`, `source`.

### Learning handler (`_on_crash_pattern_confirmed`)
- Stores pattern in `self._known_fatal_patterns[pattern_id]`
- Writes `(:MemoryTrace)` MERGE node to Neo4j with `tags=["crash_pattern","avoid"]`
- Logs at WARNING: `simula_learned_fatal_pattern {pattern_id}: {lesson}`

### Kairos bridge (`_on_kairos_invariant_for_crash_patterns`)
- Fires on every `KAIROS_INVARIANT_DISTILLED` event
- Checks `invariant_type == "crash_pattern"` gate - ignores all others
- Synthesises a CRASH_PATTERN_CONFIRMED payload from the invariant fields
- Delegates to `_on_crash_pattern_confirmed()` transparently

### Pre-flight patch gate (`_score_patch_against_patterns` + `_apply_patch_risk_gate`)
Called from `_apply_change()` before `self._applicator.apply()`.

**Feature extraction** (`_extract_patch_features`): mirrors `CrashPatternAnalyzer.extract_features()`
- prefix-namespaced features (`source:`, `class:`, `etype:`, `kw:`, `affects:`).

**Scoring** formula: `match_score = |patch_features ∩ pattern.signature| / |pattern.signature|`
- `>= 0.70` → **BLOCK** - pipeline terminates with `ProposalStatus.REJECTED`; emits
  `RE_TRAINING_EXAMPLE(outcome_quality=0.0, category="pattern_blocked")` and
  `THYMOS_REPAIR_REQUESTED(escalation_reason="known_fatal_pattern")`
- `0.40–0.69` → **WARN** - proceeds with pattern risk stashed on `proposal._pattern_risk`;
  annotation appended to RECORD-stage RE training example as `| pattern_risk=WARN ...`
- `< 0.40` → **none** - no action

### Proactive background scan (`_proactive_pattern_scan` / `_proactive_pattern_scan_loop`)
- Runs every **2 hours** as a background asyncio task (started in `set_synapse()`)
- Queries Neo4j for `(:MemoryTrace)` nodes with `tag="generated_code"` written in the last 24h
- Scores each summary's token features against `_known_fatal_patterns` with threshold `>= 0.5`
- For each match: emits `INCIDENT_DETECTED(severity=low)` requesting Thymos preemptive review
- Task cancelled in `shutdown()`

### New methods
| Method | Role |
|--------|------|
| `_on_crash_pattern_confirmed(event)` | Learn pattern, write Memory |
| `_on_kairos_invariant_for_crash_patterns(event)` | Bridge Kairos → crash pattern |
| `_score_patch_against_patterns(patch, ctx)` | Feature-overlap scoring |
| `_extract_patch_features(patch, ctx)` | Feature set extraction (static) |
| `_apply_patch_risk_gate(patch, ctx, proposal_id, hypothesis_id)` | Gate + side-effects |
| `_proactive_pattern_scan()` | One-shot 24h Memory scan |
| `_proactive_pattern_scan_loop()` | 2-hour background loop |

---

## Proxy Sandbox Fix - 9 Mar 2026

**CRITICAL BUG**: In proxy mode (`SIMULA_MODE=proxy`), `SimulaProxy.set_synapse()` was a no-op.
This meant `SIMULA_SANDBOX_REQUESTED` events from Thymos were never received by anyone -
every sandbox validation timed out, causing every Tier 3+ repair to be rejected.

**Fix**: `SimulaProxy.set_synapse()` now subscribes to `SIMULA_SANDBOX_REQUESTED` on the
Synapse event bus and handles sandbox requests inline in the main process. Since the
`ChangeSimulator` lives in the worker process and isn't available to the proxy, the handler
performs Iron Rule checks (reject protected system mutations) and approves conservatively
for all tiers. This mirrors the `no_simulator` fallback path in `SimulaService`.

The handler emits `SIMULA_SANDBOX_RESULT` directly on the Synapse bus, resolving Thymos'
correlation-based Future.

---

## Known Issues / Remaining Work

- **Stage 3 (GRPO/LILO)**: ~~`training/` and `learning/` directories exist but implementation status unverified~~ **VERIFIED** - GRPO wired in `_apply_change()` pipeline, handles Claude-vs-local-RE Thompson routing. LILO extraction runs post-RECORD on successful proposals. Both are live.
- **Stage 4 (Domain Fine-Tuning)**: vLLM deployment not yet operational (depends on RE system)
- **Governance timeout**: ~~24-hour community approval wait has no async notification path beyond initial submission~~ - **FIXED**: `_governance_timeout_loop` background task auto-rejects after 72h (HIGH/CRITICAL) or 24h (MEDIUM/LOW)
- **Inspector PoC execution**: Scoped to `inspector_authorized_targets` - ensure this list is reviewed before enabling production
- **`coevolution/`**: Adversarial self-play and red team components - audit before enabling in production (blast radius risk)
- **`SIMULA_HEALTH_DEGRADED` / `INSPECTOR_VULNERABILITY_FOUND`**: ~~New event types registered in `synapse/types.py` but emission sites not yet wired~~ **FIXED** - both now emitted from `service.py` (`health_check()` and `_emit_vulnerability_confirmed()` respectively)
- **Genome payload enrichment**: ~~`bridge.py` genome extraction does not yet include Dafny spec texts, Lean lemmas, or EvolutionRecord embeddings~~ **FIXED** - `export_simula_genome()` now exports category success rates, top-15 LILO abstractions, and top-10 successful mutation embeddings via `genome.extra`. Child-side `_apply_inherited_simula_genome_if_child()` seeds analytics priors, LILO library, and Neo4j `InheritedEvolutionRecord` nodes.
- **Thompson sampling for proof strategy routing**: ~~ReasoningRouter not wired~~ **FIXED** - `ReasoningRouter` (Thompson sampling) initialized in `__init__`, wired to `HealthChecker`, updated after Stage 6 VERIFY outcomes, exported/inherited via genome. In shallow verification mode (metabolic pressure), health.py uses `rank_strategies()` to select only the single best strategy instead of running all in parallel. Lean + formal guarantees (Phases 5–6) skipped entirely under shallow mode. Note: GRPO already handles Claude-vs-local-RE LLM routing separately.

## Intense Audit - 8 Mar 2026

### Dead Code Removal (1,537 lines deleted)
5 subsystem directories confirmed zero external imports via multi-layered grep:
- `correlation/` (446 lines) - never imported by any live code
- `instrumentation/` (248 lines) - never imported
- `protocol/` (298 lines) - never imported
- `sidechannel/` (232 lines) - never imported
- `trust/` (313 lines) - never imported

**Retained** (initially flagged but verified live):
- `egraph/` - used by health Phase 6 when `egraph_enabled=True`
- `orchestration/` - called at service.py for multi-file proposals
- 3 event handlers (`_on_immune_pattern_advisory`, `_on_organism_sleep`, `_on_telos_alignment_gap_warning`) - all cache state read by downstream pipeline gates

### ReasoningRouter Wiring (Thompson Sampling for Proof Strategies)
- `reasoning_router.py` existed but was unwired - now fully integrated
- `service.py.__init__`: `self._reasoning_router = ReasoningRouter()`
- `health.py.__init__`: `self._reasoning_router: Any = None` (set by service)
- `_run_formal_verification()`: Under shallow mode, uses `rank_strategies()` to pick the single best proof strategy (Dafny/Z3/static) instead of running all in parallel
- `check()`: Phases 5 (Lean) and 6 (egraph/symbolic) skipped entirely in shallow mode
- Stage 6 VERIFY outcomes update router arms
- Genome export/import preserves router weights across generations
- `rank_strategies()` method added to `ReasoningRouter` for ordered multi-arm selection

### Enriched Genome Extraction
`export_simula_genome()` now exports via `genome.extra`:
- `reasoning_router_weights` - Thompson arm stats per proof strategy
- `efe_calibration` - EFE scorer priors and record count
- `category_success_rates` - per-category success rates from EvolutionAnalyticsEngine
- `lilo_abstractions` - top-15 LILO library entries (name, signature, confidence, tags)
- `mutation_embeddings` - top-10 successful mutation voyage-code-3 embeddings

Child-side `_apply_inherited_simula_genome_if_child()` now seeds:
- ReasoningRouter with parent's Thompson weights
- EvolutionAnalyticsEngine with `_inherited_category_priors`
- LILO library with parent abstractions (0.8× confidence decay)
- Neo4j `InheritedEvolutionRecord` nodes with parent's mutation embeddings

---

## train_lora.py - Model + Config Update (7 Mar 2026)

`systems/simula/training/train_lora.py` updated:

- **Base model**: default changed from `unsloth/Meta-Llama-3.1-8B-Instruct` → `Qwen/Qwen3-8B` (speciation bible §1). Override via `BASE_MODEL` env var.
- **LoRA defaults**: `r=32`, `lora_alpha=64` (was r=64/α=128). Maintains 2:1 α:r ratio per bible §5.
- **Local training mode**: `TRAINING_DATA` env var bypasses IPFS download; `OUTPUT_DIR` env var sets adapter output path. Both injected by `ContinualLearningOrchestrator`.
- **status.json**: after training completes, writes `TRAINING_STATE` dict to `{OUTPUT_DIR}/status.json` so orchestrator can read `eval_loss` without parsing stdout.
- **IPFS upload**: skipped when `TRAINING_DATA` is set (local mode).

## Gap Closure - 7 Mar 2026 (second pass)

- **CRITICAL**: `coevolution/hard_negative_miner.py` + `coevolution/adversarial_tester.py` re-export wrappers created (fixes import errors in tests)
- **CRITICAL**: `RE_TRAINING_EXAMPLE` (category=`self_evolution`) emitted after every STAGE 7 RECORD in `_record_evolution()` - both APPLIED and ROLLED_BACK outcomes
- **HIGH**: 8 new `SynapseEventType` entries added: `INSPECTOR_VULNERABILITY_FOUND`, `EVOLUTION_REJECTED`, `EVOLUTION_AWAITING_GOVERNANCE`, `SIMULA_HEALTH_DEGRADED`, `SIMULA_GENOME_EXTRACTED`, `THYMOS_REPAIR_REQUESTED`, `THYMOS_REPAIR_APPROVED`, `BENCHMARK_REGRESSION_DETECTED`
- **HIGH**: `EVOLUTION_REJECTED` emitted at constraint-violation and unacceptable-simulation-risk exit points in `_run_pipeline()`
- **HIGH**: `EVOLUTION_AWAITING_GOVERNANCE` emitted when proposal enters governance gate
- **HIGH**: `SIMULA_GENOME_EXTRACTED` emitted after successful genome extraction in `_on_genome_extract_request()`
- **HIGH**: Subscriptions added for `THYMOS_REPAIR_REQUESTED`, `THYMOS_REPAIR_APPROVED`, `ONEIROS_CONSOLIDATION_COMPLETE`, `BENCHMARK_REGRESSION_DETECTED` in `set_synapse()`
- **HIGH**: Handler methods wired: `_on_thymos_repair_requested`, `_on_thymos_repair_approved`, `_on_oneiros_consolidation_complete`, `_on_benchmark_regression_detected`
- **HIGH**: Oikos metabolic gate added before `health.check()` - sets `_health._shallow_verification_mode = True` in SURVIVAL/CONSERVATION mode, skipping Dafny/Lean/Z3 to save cost

## Gap Closure - 7 Mar 2026 (third pass)

- **HIGH**: `_governance_timeout_loop()` background task - 60s poll, auto-rejects stalled governance proposals (72h for HIGH/CRITICAL risk; 24h for MEDIUM/LOW). Started in `set_synapse()`, cancelled in `shutdown()`. Emits `EVOLUTION_REJECTED` with `reason="governance_timeout"`.
- **HIGH**: Metabolic gate at STAGE 3 SIMULATE - `_check_metabolic_gate("simulate_change", "0.20")` called before 5-strategy simulation. On CRITICAL/TERMINAL starvation: synthetic `SimulationResult` (low confidence, `_simulated=False`) injected; simulation skipped with warning.
- **MEDIUM**: Evo reward signal - `_emit_evo_reward()` fired via `asyncio.ensure_future` after every Stage 7 RECORD. APPLIED: `EVO_HYPOTHESIS_CONFIRMED` with `reward = verification_confidence × (1 − risk_score)`. ROLLED_BACK: `EVO_HYPOTHESIS_REFUTED` with `reward = −0.3`. `hypothesis_id = f"simula.evolution.{category}"`.
- **MEDIUM**: Canary traffic ramp enforcement - `_run_canary_ramp(plan, proposal, log)` method. Fired as background task from `_apply_change()` for MODERATE-risk proposals. Emits `SIMULA_CANARY_PROGRESS` (new `SynapseEventType`) at each step, waits `config.canary_settle_period_s` (default 30s), health-checks, and triggers `_rollback.restore_snapshot()` on degradation.
- **LOW (dead-code audit)**: All four task-specified paths (`simula/tier2_orchestrator.py`, `inspector/phantom_recon.py`, `inspector/distributed_shield.py`, `inspector/filter_generator.py`) verified non-existent - nothing to delete. Root-level `simula/distributed_shield.py` and `simula/filter_generator.py` exist but are actively imported in `main.py` and `registry.py` - retained.
- **New `SynapseEventType`**: `SIMULA_CANARY_PROGRESS` added to `systems/synapse/types.py`.

## Organizational Closure - Round 2D (7 Mar 2026)

**Speciation Bible §8.3** - Constructive test of organizational closure: the organism can generate subsystem modules it did not have at birth.

### SubsystemGenerator (`subsystem_generator.py`)
- **`SubsystemSpec`** dataclass - name, purpose, trigger_hypothesis_id, required_events, emitted_events, dependencies, constraints
- **`SubsystemGenerator`** - wraps LLMProvider (extracted from SimulaCodeAgent), ConstraintSatisfactionChecker, RollbackManager
- **`generate_subsystem(spec)`** - 7-step flow:
  1. Iron Rule validation (forbidden name fragments: equor, simula, constitution, invariant)
  2. Snapshot existing files for rollback
  3. Build LLM prompt (architecture rules + skeleton + spec requirements)
  4. Call LLM directly for code generation (60s timeout, skeleton fallback)
  5. Validate generated code: AST syntax, no cross-system imports, required methods (initialize/shutdown/health), subscribe() present if required_events specified
  6. Write `systems/{name}/__init__.py` to disk
  7. Emit `SUBSYSTEM_GENERATED` on Synapse bus
- **`spec_from_proposal_data(data)`** - parses EVOLUTION_CANDIDATE event payload into SubsystemSpec
- Generated modules are NOT auto-registered - wire manually in `core/registry.py` before next incarnation

### Wiring
- **`SimulaService.initialize()`** - builds `SubsystemGenerator` with code agent + constraint checker alias + rollback manager
- **`SimulaService.set_synapse()`** - wires event bus into SubsystemGenerator via `set_event_bus()`
- **`SimulaService._on_evolution_candidate()`** - intercepts `mutation_type == "new_subsystem"` events; routes to SubsystemGenerator instead of `receive_evo_proposal()` (which would fail - no ChangeCategory for new_subsystem)

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
- All comms via Synapse events - `subscribe()` required if required_events specified
- Generated subsystem NOT auto-loaded - no hot-reload without operator review

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
- `SUBSYSTEM_GENERATED` - emitted on successful generation; payload: name, purpose, file_paths, hypothesis_id, validation_passed, required_events, emitted_events

### Remaining gaps (Organizational Closure)
- **Hot-reload not implemented** - generated subsystem requires restart + manual registry.py wiring
- **`core/registry.py` wiring is manual** - operator must add the new system to the registry
- **Genome not yet inherited** - generated subsystems don't automatically inherit a genome fragment
- **No test generation** - `generate_subsystem` does not create test files for the new module
- **LLM quality not verified post-generation** - only AST + import check; no semantic correctness guarantee

---

## Dynamic Capability Expansion - Round 2E (8 Mar 2026)

**Speciation Bible §8.3** - Constructive closure: the organism can generate AND immediately deploy new Axon executor classes when it discovers opportunities it cannot yet act on.

### ExecutorGenerator (`executor_generator.py`)

**Purpose:** Generate new Axon executor classes from `ExecutorTemplate` objects at runtime, then hot-load and register them immediately in the live `ExecutorRegistry`.

Unlike `SubsystemGenerator` (which defers to next boot), `ExecutorGenerator` hot-loads the generated executor in the current process. This is safe because `DynamicExecutorBase` enforces all runtime invariants - budget cap, Equor approval, audit trail - and `registry.disable_dynamic_executor()` can instantly gate any misbehaving executor.

**`generate_executor(template: ExecutorTemplate)`** - 9-step flow:
1. Iron Rule validation of template (forbidden names: equor, simula, constitution, invariant, memory)
2. Snapshot target file for rollback (if it exists)
3. Build LLM prompt from ExecutorTemplate (protocol, capabilities, required APIs, safety constraints)
4. Generate Python class via LLM (60s timeout, scaffold fallback)
5. AST syntax check
6. Source scan: no cross-system imports, no dangerous calls (eval/exec/subprocess), no inline secrets
7. Required method check: `_execute_action` + `_validate_action_params` must be present
8. Write to `axon/executors/dynamic/{name}.py`
9. Hot-register via `ExecutorRegistry.register_dynamic_executor()` + emit `RE_TRAINING_EXAMPLE`

**`_build_scaffold(template)`** - minimal valid executor skeleton when LLM is unavailable. Returns `ExecutionResult(success=False, error="scaffold")` so it's safe to deploy but won't accidentally do anything.

**Iron Rules (harder than SubsystemGenerator):**
- Generated class MUST extend `DynamicExecutorBase` - not `Executor` ABC
- CANNOT import from `systems.*`
- CANNOT contain `eval()`, `exec()`, `__import__()`, `subprocess`, `os.system()`
- CANNOT contain wallet private keys, mnemonics, or HMAC/AES secrets inline
- MUST implement `_execute_action()` and `_validate_action_params()`
- Budget cap lives in `DynamicExecutorBase` - never in generated code

**Wiring:**
- `SimulaService.initialize()` - builds `ExecutorGenerator(code_agent, rollback_manager, codebase_root)`
- `SimulaService.set_synapse(bus)` - calls `_executor_generator.set_event_bus(bus)`
- `SimulaService.set_axon_registry(registry)` - injects live registry; called from `core/wiring.py` after both services initialize
- `SimulaService._on_evolution_candidate()` - routes `mutation_type == "add_executor"` to `ExecutorGenerator`; builds `ExecutorTemplate` from `data["executor_template"]` + hypothesis metadata

**ADD_EXECUTOR Closure Loop:**
```
Oikos ProtocolScanner: OPPORTUNITY_DISCOVERED (no executor for new protocol)
  → Evo subscribes; generates EVOLUTION_CANDIDATE(mutation_type="add_executor",
      executor_template={name, action_type, description, protocol_or_platform,
                         required_apis, risk_tier, max_budget_usd, capabilities,
                         safety_constraints})
  → SimulaService._on_evolution_candidate routes to ExecutorGenerator
  → Generate + AST validate + write axon/executors/dynamic/{name}.py
  → ExecutorRegistry.register_dynamic_executor() - hot-loaded in current process
  → EXECUTOR_REGISTERED emitted - Thymos opens 24h monitoring window
  → RE_TRAINING_EXAMPLE emitted (category="executor_generation")
```

### New SynapseEventTypes (8 Mar 2026)
- `EXECUTOR_REGISTERED` - payload: action_type, name, protocol_or_platform, risk_tier, max_budget_usd, capabilities, source_hypothesis_id, registered_at
- `EXECUTOR_DISABLED` - payload: action_type, name, reason, incident_count, disabled_at
- `OPPORTUNITY_DISCOVERED` - emitted by `oikos/protocol_scanner.py` when a DeFiLlama pool or Immunefi bounty has no matching executor

## Novel Action Proposal Pipeline (8 Mar 2026)

**Trigger:** Nova emits `NOVEL_ACTION_REQUESTED` when the deliberation engine selects a policy containing a `propose_novel_action` step and no existing executor covers the goal.

**Handler:** `SimulaService._on_novel_action_requested()` (closure in `wire_evo_callbacks()`):
1. **Iron Rule check** - rejects proposals whose `action_name` contains any of: `equor`, `simula`, `constitution`, `invariant`, `memory`. Logs `novel_action_iron_rule_blocked` and returns.
2. **Equor pre-approval** - emits `EQUOR_ECONOMIC_INTENT` (action="novel_action_proposal"); awaits `EQUOR_ECONOMIC_PERMIT` with 30s timeout. Auto-permits on timeout (safety fallback, logged at WARNING). Denied → returns without executor generation.
3. **Risk tier inference** - checks `required_capabilities` against `high_risk_caps = {wallet_access, defi_write, mitosis_spawn}` → "high"; `medium_risk_caps = {git_write, http_client, code_generation}` → "medium"; else "low".
4. **ExecutorTemplate construction** from proposal payload fields.
5. **`self._executor_generator.generate_executor(template)`** - full 9-step flow (validate → LLM → AST → scan → write → hot-register).
6. **`NOVEL_ACTION_CREATED` emitted** on success with: `proposal_id`, `action_name`, `description`, `capabilities`, `risk_tier`, `executor_file`, `success=True`. On failure: `success=False`, `error`.

**Iron Rules (same as ExecutorGenerator hardcoded rules):**
- `action_name` cannot shadow reserved names
- Generated executor must extend `DynamicExecutorBase`
- No cross-system imports; no eval/exec/subprocess; no inline secrets
## Gap Closure (8 Mar 2026 - SimulaGenome Child Inheritance)

### `_apply_inherited_simula_genome_if_child()` (NEW)
**Problem:** `SimulaService.export_simula_genome()` existed and was called by `SpawnChildExecutor`,
but there was no child-side receiver. Child Simula instances always booted with default config
params, ignoring the parent's tuned evolution state.

**Fix:**
- `_apply_inherited_simula_genome_if_child()` added to `SimulaService`
- Reads `ORGANISM_SIMULA_GENOME_PAYLOAD` (JSON-encoded `SimulaGenome`)
- Skips silently if genesis node or env var absent
- Applies each inherited learnable param to `self._config` with bounded ±10% Gaussian jitter (σ=3.3%)
  for genetic variation, mirroring the Telos pattern (Spec 18 SG3)
  - Float params: `value * (1.0 + jitter)`; int params: `max(1, round(value * (1.0 + jitter)))`
  - Strings/bools: inherited exactly (no jitter)
  - Only params that exist on the live `SimulaConfig` are applied (graceful skip on absent fields)
- Mutation history stored as `self._inherited_mutation_history` for analytics
- Emits `GENOME_INHERITED` (system="simula") if Synapse bus is wired
- Called from `initialize()` after all subsystems are built; wrapped in try/except (non-fatal)
- **Dafny spec hash optimization (8 Mar 2026)**: inherited `dafny_spec_hashes` stored on `self._inherited_spec_hashes` and seeded into `IncrementalVerificationEngine` via `seed_inherited_hashes()`. During the first verification cycle, specs whose content hash matches a parent-verified hash are skipped (treated as pre-verified, logged as `verification_skipped_inherited_from_parent`). Inherited hashes cleared after first `verify_incremental()` call completes (one-time boot optimization).

---

## Unified Economic Parameter Registry (8 Mar 2026)

Two methods govern economic parameter learning, and they must use identical parameter sets and bounds:

| Method | Registry | Purpose |
|--------|----------|---------|
| `_on_config_drift()` | `float_params` list (~line 2440) | Gaussian perturbation driven by Soma/Evo CONFIG_DRIFT events |
| `_on_evo_adjust_budget()` | `_ECON_PARAM_BOUNDS` dict (~line 2570) | Directed adjustment driven by Evo high-confidence hypotheses (confidence > 0.75) |

### Unified Economic Parameter Set

| Parameter | Range | Notes |
|-----------|-------|-------|
| `yield_apy_drop_rebalance_threshold` | (0.05, 0.50) | Lower = more aggressive rebalancing |
| `yield_apy_minimum_acceptable` | (0.01, 0.20) | Minimum APY before Oikos rejects a protocol |
| `bounty_min_roi_multiple` | (1.0, 5.0) | Minimum return multiple for bounty acceptance |
| `bounty_max_risk_score` | (0.20, 0.90) | Maximum risk score for bounty acceptance |
| `asset_dev_budget_pct` | (0.05, 0.40) | Fraction of assets available for dev investment |
| `child_spawn_interval_days` | (7.0, 180.0) | Minimum days between child spawning events |
| `child_min_profitability_usd` | (10.0, 10000.0) | Minimum child revenue before dividend |
| `cost_reduction_target_pct` | (0.01, 0.50) | Target cost reduction per optimization cycle |
| `emergency_liquidation_threshold` | (0.02, 0.30) | Runway fraction that triggers emergency exit |
| `protocol_exploration_budget_pct` | (0.05, 0.50) | Fraction of budget for new protocol exploration |
| `protocol_allocation_aggressiveness` | (0.1, 1.0) | Concentration vs diversification tuning |

### INT_PARAMS (class-level constant)

`SimulaService.INT_PARAMS` is a `frozenset[str]` containing parameter names that appear in `float_params` but are semantically integers. After Gaussian perturbation and clamping in `_on_config_drift`, and after clamping in `_on_evo_adjust_budget`, values for these params are cast to `int` via `round()`.

Current members: `child_spawn_interval_days`, `child_min_profitability_usd`

### Sync Contract

When adding a new economic parameter:
1. Add to `float_params` in `_on_config_drift()` with `(lo, hi)` bounds
2. Add to `_ECON_PARAM_BOUNDS` in `_on_evo_adjust_budget()` with identical bounds
3. Update the table above in this CLAUDE.md
4. If semantically integer, add to `SimulaService.INT_PARAMS`

When changing a bound:
1. Change it in BOTH registries simultaneously
2. The CONFIG_DRIFT range is considered authoritative for semantics (conservative, operationally tested)

---

## Recursive Self-Improvement Layer (Spec 10 §SM - 9 Mar 2026)

EOS can now identify its own capability gaps and close them autonomously, within constitutional constraints.

### Architecture

Three new components work together:

| Component | File | Role |
|-----------|------|------|
| `CapabilityAuditor` | `nova/capability_auditor.py` | Monitors NOVEL_ACTION_REQUESTED + AXON_EXECUTION_RESULT events; emits CAPABILITY_GAP_IDENTIFIED when ≥3 goals blocked or value > $10 |
| `SelfModificationPipeline` | `nova/self_modification_pipeline.py` | Gap → drive alignment scoring → SELF_MODIFICATION_PROPOSED → Equor review → Simula code gen → HotDeployment → live test → RE training |
| `HotDeployment` | `core/hot_deploy.py` | Writes executor file, imports module, registers in ExecutorRegistry, writes (:SelfModification) Neo4j node, emits EXECUTOR_DEPLOYED |

### Full Pipeline

```
CapabilityAuditor detects gap (≥3 NOVEL_ACTION_REQUESTED with no NOVEL_ACTION_CREATED)
  → emits CAPABILITY_GAP_IDENTIFIED
  → SelfModificationPipeline._handle_gap()

Step 1: Drive alignment scoring (coherence/care/growth/honesty)
         EFE proxy score < 0.3 → reject silently

Step 2: Emit SELF_MODIFICATION_PROPOSED → Equor reviews constitutionally

Step 3: Await EQUOR_ECONOMIC_PERMIT (30s timeout → auto-permit with WARNING)
         Denied → RE_TRAINING_EXAMPLE(outcome=equor_denied) + return

Step 4: If requires_external_dependency:
         HotDeployment.install_dependency(package) → PyPI safety check → pip install
         On failure → RE_TRAINING_EXAMPLE(outcome=dependency_failed) + return

Step 4b: If complexity=high AND subsystem_needed:
          Draft Spec document → emit SPEC_DRAFTED → Equor reviews
          (Simula implements Spec after approval - separate flow)

Step 5: Emit NOVEL_ACTION_REQUESTED (pipeline_managed=True)
         Simula generates executor code → NOVEL_ACTION_CREATED with executor_code

Step 6: HotDeployment.deploy_executor()
         → Iron Rule validate → write axon/executors/dynamic/{action_type}.py
         → importlib hot-load → ExecutorRegistry.register_dynamic_executor()
         → (:SelfModification) Neo4j node → emit EXECUTOR_DEPLOYED

Step 7: Nova registers action_type in ActionTypeRegistry
         Queues low-stakes test goal via NOVA_INTENT_REQUESTED (test_mode=True)
         Waits up to 5 min for INTENT_OUTCOME

         Test success → capability permanent; Evo gets +reward
         Test failure → HotDeployment.rollback_executor()
                      → emit EXECUTOR_REVERTED
                      → Nova removes from ActionTypeRegistry

Step 8: RE_TRAINING_EXAMPLE emitted for every outcome:
         - equor_denied / dependency_failed / code_generation_failed
         - deploy_failed / deployed_and_verified / deployed_then_rolled_back
```

### Iron Rules (never violated)

1. `action_type` cannot contain: `equor`, `simula`, `constitution`, `invariant`, `memory`
2. Generated code must extend `DynamicExecutorBase` - never `Executor` ABC
3. No `from systems.*` imports in generated code
4. No `eval()`, `exec()`, `subprocess`, `os.system()` in generated code
5. No inline private keys, mnemonics, HMAC/AES secrets
6. Deployment target MUST be inside `axon/executors/dynamic/`
7. All deployments logged in Neo4j `(:SelfModification)` with `reverted=false` initially
8. Rollback always available - HotDeployment.rollback_executor() removes file + unregisters

### New SynapseEventTypes (9 Mar 2026)

| Event | Emitter | Consumers |
|-------|---------|-----------|
| `CAPABILITY_GAP_IDENTIFIED` | CapabilityAuditor | SelfModificationPipeline, logging |
| `SELF_MODIFICATION_PROPOSED` | SelfModificationPipeline | Equor (review), Simula (awaits permit) |
| `EXECUTOR_DEPLOYED` | HotDeployment | Nova (register + test), Thymos (24h window), Evo (hypothesis), Thread (GROWTH) |
| `EXECUTOR_REVERTED` | HotDeployment | Nova (remove from registry), Thread (CRISIS), Evo (refutation) |
| `DEPENDENCY_INSTALLED` | HotDeployment | Thymos (dependency surface), logging |
| `SPEC_DRAFTED` | SelfModificationPipeline | Equor (review), Simula (implement after approval) |

### Neo4j Audit Trail

Every successful deployment creates:
```cypher
CREATE (m:SelfModification {
    id: $deployment_id,
    capability_added: $action_type,
    gap_resolved: $proposal_id,
    code_hash: $sha256,
    module_path: "systems/axon/executors/dynamic/{action_type}.py",
    deployed_at: datetime(),
    equor_approval_id: $approval_id,
    reverted: false
})
```
On rollback: `SET m.reverted = true, m.reverted_at = datetime()`

### Registration in core/registry.py

Phase 11 (after benchmarks): `self_modification_layer_started` log message.
Components accessible as:
- `app.state.hot_deploy` - HotDeployment instance
- `app.state.capability_auditor` - CapabilityAuditor instance
- `app.state.self_modification_pipeline` - SelfModificationPipeline instance

### Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `AUDITOR_MIN_BLOCKING_GOALS` | `3` | Min blocked goals before gap is emitted |
| `AUDITOR_MIN_VALUE_USD` | `10` | Min estimated value to trigger gap emission |
| `AUDITOR_COOLDOWN_HOURS` | `6` | Hours between re-emitting same gap |
| `AUDITOR_OBSERVATION_WINDOW_HOURS` | `24` | Rolling window for counting events |

---

## Preventive Audit - 9 Mar 2026

`backend/systems/simula/preventive_audit.py` - `SimulaPreventiveAudit`

Scheduled 4-hour self-audit that proactively scans recently generated/modified code
for fragility patterns **before** crashes occur. Complements the 2-hour
`_proactive_pattern_scan` (token-feature matching) with full AST static analysis.

### Phases

**Phase 1 - Fragility scan**
- Queries Neo4j for `(:MemoryTrace)` nodes tagged `generated_code` modified in the last 24h
- Runs 5 AST/regex static checks per fragment:
  | Check | Weight |
  |-------|--------|
  | `bare_except` - bare `except:` or `except Exception: pass` | 0.25 |
  | `unhandled_create_task` - discarded `asyncio.create_task()` result | 0.20 |
  | `division_no_guard` - division by variable without zero-guard | 0.20 |
  | `db_no_try_except` - Redis/Neo4j calls in functions lacking try/except | 0.25 |
  | `missing_await` - async def with un-awaited coroutine calls | 0.10 |
- `fragility_score` = weighted sum, clamped [0.0, 1.0]

**Phase 2 - Pattern matching**
- Fragments with `fragility_score > enter_threshold` (≈0.3, lowered by `audit_aggressiveness`)
- Jaccard overlap against `service._known_fatal_patterns` (the live `CrashPattern` dict)
- `match_score ≥ 0.50` → flagged PREEMPTIVE_RISK
- `combined_risk = fragility_score × pattern.confidence`

**Phase 3 - Incident history correlation**
- Emits `THYMOS_INCIDENT_QUERY` (request_id, lookback_days=7, max_incidents=50)
- Awaits `THYMOS_INCIDENT_RESPONSE` via async Future (10s timeout)
- Token-overlap correlation between fragment tokens and incident `error_message`
- Overlap ratio ≥ 0.15 with any recent incident → PREEMPTIVE_CRITICAL upgrade

**Phase 4 - Action**
| Level | Condition | Action |
|-------|-----------|--------|
| PREEMPTIVE_CRITICAL | `combined_risk > 0.7` | `THYMOS_REPAIR_REQUESTED(context["preventive"]=True)` + `RE_TRAINING_EXAMPLE(category="preventive_repair", outcome_quality=0.8)` |
| PREEMPTIVE_RISK | `combined_risk 0.4–0.7` | `INCIDENT_DETECTED(severity=LOW)` |
| Below threshold | `combined_risk < 0.4` | Silent - metrics to Benchmarks only |

### Genome Parameter

`audit_aggressiveness: float` - added to `SimulaConfig` (default `0.5`).
- `0.0` - only flag `combined_risk > 0.7` (CRITICAL only)
- `0.5` - default thresholds
- `1.0` - flag everything above `fragility_score > 0.2`
- Exported via `export_simula_genome()` → `current_evolution_params["audit_aggressiveness"]`
- Inherited by children via `_apply_inherited_simula_genome_if_child()` (±10% jitter)
- Evo can tune via `ADJUST_BUDGET` mechanism (float param)

### Synapse Integration

**Emits:**
- `THYMOS_INCIDENT_QUERY` - request incident history from Thymos (Phase 3)
- `THYMOS_REPAIR_REQUESTED(context["preventive"]=True)` - proactive repair request (Phase 4 critical)
- `RE_TRAINING_EXAMPLE(category="preventive_repair")` - learning signal (Phase 4 critical)
- `INCIDENT_DETECTED(severity=LOW)` - flag for human review (Phase 4 risk)
- `BENCHMARK_RE_PROGRESS` - audit cycle metrics to Benchmarks

**Subscribes:**
- `THYMOS_INCIDENT_RESPONSE` → `on_thymos_incident_response()` - resolves pending futures

### Thymos Handler (`_on_incident_query`)

Added to `ThymosService` - subscribes to `THYMOS_INCIDENT_QUERY`.
- Scans `_incident_buffer` (deque[Incident], maxlen=10,000) for incidents within `lookback_days`
- Returns up to `max_incidents` serialized incident dicts (incident_id, incident_class, severity, source_system, error_type, error_message, fingerprint, created_at)
- Emits `THYMOS_INCIDENT_RESPONSE` with `request_id` correlation

### Wiring

- `SimulaService.initialize()`: `self._preventive_audit = SimulaPreventiveAudit(self)`
- `SimulaService.set_synapse()`: subscribes `THYMOS_INCIDENT_RESPONSE` → `_preventive_audit.on_thymos_incident_response()`; starts `supervised_task("simula.preventive_audit", max_restarts=5)`
- `SimulaService.shutdown()`: `self._preventive_audit_task.cancel()` (via existing task cancellation sweep)

### New SynapseEventTypes
- `THYMOS_INCIDENT_QUERY` - added to `systems/synapse/types.py`
- `THYMOS_INCIDENT_RESPONSE` - added to `systems/synapse/types.py`
