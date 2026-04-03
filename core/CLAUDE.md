# Core - CLAUDE.md

**Role:** Organism startup, wiring, and background task orchestration.
**Path:** `backend/core/`

---

## Key Files

| File | Purpose |
|------|---------|
| `registry.py` | `SystemRegistry` - full 12-phase startup sequence; owns `_tasks` dict |
| `wiring.py` | All `set_*()` calls, subscriptions, and `wire_*_phase()` functions |
| `scheduled_tasks.py` | `PerceptionScheduler` registrations (PRs, DeFi yield, foraging, consolidation) |
| `re_training_exporter.py` | `RETrainingExporter` - hourly RE training data batch export |
| `continuous_learning_orchestrator.py` | `ContinualLearningOrchestrator` - domain-aware LoRA adapter training scheduler |
| `curriculum_builder.py` | `DomainCurriculum` - filters and orders examples for domain-specific training |
| `infra.py` | `InfraClients` dataclass; `create_infra()` / `close_infra()` |
| `inner_life.py` | `inner_life_loop()` - background inner dialogue generator |
| `interoception_loop.py` | `interoception_loop()` - log analyzer → Soma signal bridge |
| `smoke_test.py` | `run_smoke_tests()` - post-startup sanity checks |
| `hotreload.py` | `NeuroplasticityBus` - live evaluator/executor hot-swap |
| `helpers.py` | `MemoryWorkspaceAdapter`, `resolve_governance_config`, `seed_atune_cache` |

---

## Startup Phase Sequence

| Phase | Systems / Actions |
|-------|------------------|
| Foundation | Memory → Logos → Equor → Atune → EIS → SACM |
| Core Cognitive | Voxis → Nova → Axon → (Atune startup) → core wiring |
| Learning & Identity | Evo → Thread |
| Self-Evolution | Simula |
| Coordination Bus | Synapse (clock + health monitor) |
| Immune & Dream | Thymos → Oneiros → Kairos |
| Interoception | Soma → exteroception |
| Intelligence Loops | Telos → Fovea |
| Federation + Economic | Federation → Nexus → wallet → Oikos → Mitosis |
| Alive WebSocket | Phantom Liquidity → Skia → connectors |
| Background Tasks | interoception, inner_life, file_watcher, scheduler, fleet_shield, metrics_publisher, **re_training_export**, benchmarks, observatory, **imap_scanner**, **account_provisioner wiring**, **email_client wiring** |
| Smoke Tests | post-startup validation |

---

## RE Training Exporter (`re_training_exporter.py`)

**Status:** Wired in Phase 11 of `registry.py`

### What it does
- Subscribes to `RE_TRAINING_EXAMPLE`, `AXON_EXECUTION_RESULT`, `EVO_HYPOTHESIS_CONFIRMED`, and `EVO_HYPOTHESIS_REFUTED` events
- Accumulates `RETrainingDatapoint` objects in-memory with episode-level dedup
- Maintains `_episode_index: dict[str, RETrainingDatapoint]` for O(1) retroactive quality corrections
- Every 3600s: drains accumulator → `RETrainingExportBatch` → enrichment → S3 (JSON lines) + Neo4j lineage
- Writes individual `(:RETrainingDatapoint)` nodes to Neo4j (batched UNWIND) with full reasoning traces, constitutional alignment scores, and `[:CONTAINS_DATAPOINT]` edges to batch
- Emits `RE_TRAINING_EXPORT_COMPLETE` on successful export (Benchmarks subscribes)

### Retroactive outcome correction (AXON_EXECUTION_RESULT)
- When `AXON_EXECUTION_RESULT` arrives, `_on_axon_execution_result()` looks up the episode by `episode_id`
- If `|actual_quality - estimated_confidence| > 0.1`, updates `confidence`, `outcome`, sets `outcome_updated=True`, `actual_outcome_quality=<float>`
- `update_outcome_quality(episode_id, actual_quality, source_system)` is also callable externally
- Neo4j writes `actual_outcome_quality` as a separate property for ground-truth queries

### Export enrichment (`_enrich_batch`)
Called in `export_to_s3()` before serialisation. Mutates each datapoint in-place:
1. `task_difficulty` - `min(alternatives/5, 1.0)×0.3 + counterfactual_heuristic×0.3 + (1−|conf−0.5|×2)×0.4`
2. Scaffold validation - `scaffold_formatter.validate(reasoning_trace)`: ≥3 of 5 Step headers
3. `quality_tier` assignment: `"gold"` | `"silver"` | `"bronze"`

### Quality tier rules
| Tier | Conditions |
|------|-----------|
| gold | scaffold_valid + trace>100 chars + (≥2 alternatives OR task_difficulty≥0.5) + confidence≥0.7 |
| silver | scaffold_valid OR (trace>100 + ≥2 alternatives) |
| bronze | everything else |

### Monitoring
`re_training_quality_tier_distribution` log emitted every export cycle with `gold`, `silver`, `bronze` counts, `outcome_corrected` count, `total`.

### New fields on `RETrainingDatapoint` (primitives)
- `outcome_updated: bool` - True if retroactively corrected from Axon
- `actual_outcome_quality: float | None` - ground-truth quality from Axon
- `quality_tier: str` - "gold" | "silver" | "bronze" (assigned at export)
- `task_difficulty: float` - richness proxy [0, 1] (computed at export)

### Configuration (env vars)
| Variable | Default | Purpose |
|----------|---------|---------|
| `RE_TRAINING_EXPORT_DIR` | `data/re_training_batches` | Local fallback export dir |
| `RE_TRAINING_S3_BUCKET` | `ecodiaos-re-training` | S3 bucket for CLoRA pipeline |
| `RE_TRAINING_S3_PREFIX` | `batches/` | S3 key prefix |

### S3 fallback
If `boto3` is not installed or S3 fails, batches are written as JSON lines to `RE_TRAINING_EXPORT_DIR`. The local path is included in `export_destinations` so the Benchmarks system can track it.

### Primitives
- `RETrainingDatapoint` - one normalised record per `RE_TRAINING_EXAMPLE` event
- `RETrainingExportBatch` - hourly roll-up; written to Neo4j as `(:RETrainingBatch)` + `(:RETrainingSource)` + individual `(:RETrainingDatapoint)` nodes

### Input Validation (Gap 4 - 9 Mar 2026)
`_datapoint_from_event()` now validates before creating a datapoint:
- `instruction` length < 10 chars → `None` (reject trivially empty)
- `output` length < 3 chars → `None` (reject trivially empty)
- `outcome_quality` is clamped to `[0.0, 1.0]`; non-numeric values default to 0.0

### Hypothesis Lifecycle Subscriptions (Gap 4 - 9 Mar 2026)
Two new event subscriptions in `attach()`:
- **`EVO_HYPOTHESIS_CONFIRMED`** → `_on_hypothesis_confirmed()`: creates `RETrainingDatapoint` with `outcome="success"`, `outcome_quality=hypothesis.confidence` (or 0.7 fallback). Dedup key: `evo_confirmed:{hypothesis_id}`. Category: `evo_hypothesis_confirmed`.
- **`EVO_HYPOTHESIS_REFUTED`** → `_on_hypothesis_refuted()`: creates `RETrainingDatapoint` with `outcome="failure"`, `outcome_quality=0.0`. Dedup key: `evo_refuted:{hypothesis_id}`. Category: `evo_hypothesis_refuted`.
- Both subscriptions guarded by `hasattr(SynapseEventType, ...)` - non-fatal if event types absent
- Both handlers set episode_id = `evo_confirmed:{hypothesis_id}` / `evo_refuted:{hypothesis_id}` for standard dedup in `_accumulate()`

### Integration
- `app.state.re_exporter` - accessible from API endpoints for stats
- `re_exporter.stats` - `{pending_examples, total_exported, total_batches, window_start, seen_episode_ids, attached}`
- `export_cycle()` - callable directly for testing without waiting for the 1-hour interval

---

## RE Post-Training Evaluator (`re_evaluator.py`)

**Status:** Wired in Phase 11 of `registry.py`, immediately after RE Training Exporter

### What it does
- Subscribes to `RE_TRAINING_EXPORT_COMPLETE` - triggers immediately after every successful hourly export
- Also runs on a 24h safety-net schedule (`supervised_task("re_evaluator")`)
- For each category in `["build_error","hot_swap_failure","hot_swap_rollback","crash_pattern","general_repair","code_generation"]`:
  - Pulls up to 20 recent `RETrainingDatapoint` records from S3 / local filesystem
  - Replays each original instruction+context as a prompt through `VLLMProvider.generate()`
  - Scores the response with a per-category heuristic (compile for code_generation, error-pattern avoidance for repair, known-bad CID avoidance for hot_swap)
  - Computes `pass_rate = successes / total`
- Compares to `EvaluationBaseline` stored in Redis: `"re_eval:baseline:{instance_id}:{category}"`
- Emits `BENCHMARK_RE_PROGRESS` per category (`kpi_name=re_model.{category}.pass_rate`, `delta`, `direction`)
- If `delta < -0.05` → emits `INCIDENT_DETECTED` severity=HIGH ("RE model regressed on {category}")
- If `delta > 0.10` → emits `RE_TRAINING_EXAMPLE` with `outcome_quality=1.0, category="model_improvement"` (organism celebrates its learning)
- Computes `health_score = weighted average` across all categories; emits `BENCHMARK_RE_PROGRESS` with `kpi_name=re_model.health_score`
-: `_emit_health_score()` now loads the previous `re_eval:health_score:{instance_id}` from Redis before persisting the new value, computes `delta = current - previous`, and sets `direction = "up"/"down"/"flat"`. Previously always emitted `delta=0.0` which prevented Thread's `_on_re_model_improved` from firing (requires `delta > 0.05`).
- Persists health score to Redis: `"re_eval:health_score:{instance_id}"`
- If `health_score > 0.85` → emits `NOVA_GOAL_INJECTED` (priority=0.6): "RE model performing at X% - organism is learning"
- Skipped silently if RE service is not available (`is_available == False`)

### Category weights (health score)
| Category | Weight |
|---|---|
| `build_error` | 0.30 |
| `crash_pattern` | 0.25 |
| `hot_swap_failure` | 0.20 |
| `general_repair` | 0.10 |
| `code_generation` | 0.10 |
| `hot_swap_rollback` | 0.05 |

### Scoring heuristics
- `code_generation`: `compile(response)` → 1.0 if clean, 0.3 if non-trivial but syntactically broken, 0.0 otherwise
- `hot_swap_*`: checks response does not suggest the known-failed adapter CID; then repair heuristic
- All other categories: presence of known error patterns penalised; repair keywords rewarded; length bonus

### Redis keys
| Key | Value |
|---|---|
| `re_eval:baseline:{instance_id}:{category}` | JSON: `{pass_rate, sample_count, timestamp}` |
| `re_eval:health_score:{instance_id}` | Float string |
| `re_eval:last_run:{instance_id}` | ISO-8601 UTC timestamp |

### Configuration (env vars - same as re_training_exporter)
| Variable | Default | Purpose |
|---|---|---|
| `RE_TRAINING_EXPORT_DIR` | `data/re_training_batches` | Local JSONL directory |
| `RE_TRAINING_S3_BUCKET` | `ecodiaos-re-training` | S3 bucket |
| `RE_TRAINING_S3_PREFIX` | `batches/` | S3 key prefix |

### New SynapseEventType
`RE_MODEL_EVALUATED` - added to `synapse/types.py` (reserved for future use by evaluator summary event; current KPIs use `BENCHMARK_RE_PROGRESS`)

### Integration
- `app.state.re_evaluator` - accessible from API health endpoints
- `re_evaluator.stats` - `{running, attached, last_eval_ts, vllm_wired, instance_id}`
- `re_evaluator.set_vllm(re_service)` - wired in registry with `app.state.reasoning_engine`

---

## Background Tasks (Phase 11)

All tasks are started via `utils.supervision.supervised_task()` with auto-restart.

| Task key | Source | Interval | Purpose |
|----------|--------|----------|---------|
| `nova_heartbeat` | `nova.start_heartbeat()` | config | drive-based inner monologue |
| `interoception` | `interoception_loop()` | continuous | log → Soma signals |
| `inner_life` | `inner_life_loop()` | continuous | background cognition |
| `metrics_publisher` | `publish_metrics_loop()` | continuous | Redis → InfluxDB |
| `token_refresh_scheduler` | `TokenRefreshScheduler.run()` | 3600s check | Proactive OAuth2 token refresh 24h before expiry; emits CONNECTOR_TOKEN_REFRESHED / CONNECTOR_TOKEN_EXPIRED |
| `imap_scanner` | `IMAPScanner.run()` | configurable (default 60s) | Polls IMAP inbox for inbound OTP/verification codes; emits IDENTITY_VERIFICATION_RECEIVED + EMAIL_OTP_RECEIVED; no-op if imap_host unset |
| `account_provisioner` | `AccountProvisioner` (init only, not a loop) | one-shot at boot | Autonomous platform identity provisioning (Twilio, GitHub, Gmail); triggered by `IdentitySystem._run_platform_provisioning()` after wiring |
| `re_training_export` | `re_exporter.run_loop()` | 3600s | RE training data pipeline |
| `re_evaluator` | `re_evaluator.run_loop()` | 86400s | Post-training evaluator - replays prompts, scores per-category pass rates, emits BENCHMARKS_KPI / INCIDENT_DETECTED / RE_TRAINING_EXAMPLE / NOVA_GOAL_INJECTED |
| `domain_specialization` | `_domain_clo.run_loop()` | 3600s | Domain-specific LoRA adapter training |
| `red_team_monthly` | `_run_monthly_red_team()` | 30 days | Red-team adversarial eval + Tier 2 kill switch |
| `tier3_quarterly_cron` | `_run_tier3_cron()` | 7-day check / 90-day fire | Quarterly Tier 3 full retrain, fires independently of data-volume gate |
| `re_reprobe` | `re_service.start_reprobe_loop()` | 120s | Circuit breaker reprobe - auto-detects vLLM recovery after adapter_watcher restart |
| `infra_cost_poller` | `InfrastructureCostPoller.start()` | 300s | RunPod GraphQL cost polling → MetabolicTracker (not supervised_task - self-managed asyncio) |

**Self-Modification Layer** (wired in Phase 11, event-driven - no background loop):

| Component | `app.state` key | Role |
|-----------|-----------------|------|
| `HotDeployment` | `hot_deploy` | Writes + imports + registers executors; Neo4j audit; rollback |
| `CapabilityAuditor` | `capability_auditor` | Event-driven gap detector; emits CAPABILITY_GAP_IDENTIFIED |
| `SelfModificationPipeline` | `self_modification_pipeline` | Orchestrates full gap→Equor→Simula→deploy→test cycle |

---

## Red-Team Monthly Background Task

**Status:** Wired in Phase 11 of `registry.py`, after Continual Learning Orchestrator

### What it does
- Instantiates `RedTeamEvaluator` from `systems/reasoning_engine/safety.py`
- Every 30 days: calls `check_kill_switch(re_service, event_bus, equor)` which:
  1. Loads `data/evaluation/red_team_prompts.jsonl` (50 adversarial prompts)
  2. Runs each prompt through the RE model; checks output for unsafe patterns
  3. Emits `RED_TEAM_EVALUATION_COMPLETE` with `{pass_rate, total, blocked, by_category, kill_switch_triggered}`
  4. If `pass_rate < 0.70`: emits `RE_TRAINING_HALTED` (Tier 2 kill switch) AND sets `app.state.continual_learning._training_halted = True`
- Skipped (with log) if RE service is not available
- Non-fatal throughout: inner exception → `red_team.monthly_failed` log; outer exception → task restarts via `supervised_task`

### Configuration
| Variable | Default | Purpose |
|---|---|---|
| `RE_RED_TEAM_PROMPTS_PATH` | `data/evaluation/red_team_prompts.jsonl` | Adversarial prompt set |
| `RE_CONSTITUTIONAL_SCENARIOS_PATH` | `data/evaluation/constitutional_scenarios.jsonl` | SafeLoRA proxy |

### Kill switch wiring
- `_training_halted = True` on `app.state.continual_learning` AND persisted to Redis key `eos:re:training_halted` - survives restarts
- Organism continues normally; only self-training is paused
- Manual recovery: `python -m cli.training_run clear-halt` (deletes Redis key + clears in-memory flag)

---

## Tier 3 Quarterly Cron (`_run_tier3_cron` in `registry.py`)

**Status:** Wired in Phase 11, after Continual Learning Orchestrator, before red-team cron

### What it does
- Checks every 7 days whether 90 days have elapsed since last Tier 3 (reads `eos:re:last_tier3_timestamp`)
- When ready: calls `clo._build_cumulative_dataset()` + `clo._tier3.run_tier3()`
- Decouples Tier 3 from `should_train()` data-volume gate - Tier 3 now fires even if organism is data-starved

### Key properties
- Check interval: 7 days (`7 * 24 * 3600`)
- Fire condition: `Tier3Orchestrator.should_run_tier3()` returns True (90 days elapsed)
- Non-fatal throughout: inner exception → `tier3_cron.failed` log; outer exception → task restarts
- Only started if both `re_service` and `infra.neo4j` are available (same guard as CLO)
- `app.state` key: none (task owns itself; CLO owns Tier3Orchestrator)

### app.state key
- `app.state.red_team_evaluator` - `RedTeamEvaluator` instance, accessible from API health endpoints

---

---

## Runtime Introspection API (`registry.py`)


Three methods added to `SystemRegistry` for live observability without restarting the organism.

### `get_system_status(name, app) -> dict`
Probes a single system by name. Checks:
1. Whether `app.state.<name>` exists (object alive)
2. Whether `self._tasks[name]` asyncio Task is not done (task alive)
3. `_initialized` / `initialized` attribute (sync probe - no await)
4. `task.exception()` if task done and not cancelled → `status = "error"`

Returns: `{name, status, task_alive, initialized, extra}` where status ∈ `"running" | "stopped" | "error" | "unknown"`.

### `get_all_systems(app) -> list[dict]`
Calls `get_system_status()` for every key in `_DEPENDENCY_GRAPH` plus every key in `self._tasks`. Sorted alphabetically. Returns list of status dicts.

### `get_dependency_graph() -> dict[str, list[str]]`
Returns `SystemRegistry._DEPENDENCY_GRAPH` - a class-level constant mapping each of the 27 named systems to its dependency list. Available before startup completes.

### `_DEPENDENCY_GRAPH`
Class-level constant. 27 systems. Topology: memory → logos → equor → …; nova depends on memory+equor+voxis; synapse depends on atune+nova+evo+equor; etc.

---

## Runtime Config Query API (`config.py`)


Three functions exported from `config.py` for introspecting the live config at runtime.

### `get_all_config(config, *, yaml_raw=None) -> dict[str, ConfigEntry]`
Flattens the full `EcodiaOSConfig` tree into a dot-delimited key→value mapping. Each value is a `ConfigEntry` dict with:
- `value` - current value, or `"<redacted>"` for secret fields
- `source` - `"env"` | `"yaml"` | `"default"`
- `is_secret` - bool

Secret detection: any field whose name contains `key`, `secret`, `token`, `password`, `pwd`, or `api_key` (case-insensitive) is redacted.

### `get_config(config, key, *, yaml_raw=None) -> ConfigEntry | None`
Returns the ConfigEntry for a single dot-delimited key (e.g. `"nova.max_active_goals"`). Returns None if key not found.

### `is_overridden(config, key) -> bool`
Returns True when the field's current value differs from the Pydantic model default. Walks the dotted key path to find the leaf model field and compares via `model_fields`.

---

## Constraints

- No system imports at module level in `registry.py` - all deferred to `_init_*()` methods
- `wiring.py` uses `Any` type hints for system args to avoid cross-imports
- All `_tasks` cancelled on `shutdown()` - add new background tasks to `_tasks` dict
