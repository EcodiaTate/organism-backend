# Benchmarks — CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_24_Benchmarks.md` (v1.3, Phase 3 complete)
**System ID:** `benchmarks`
**Role:** Fitness sensor and regression detection layer. Measures 7 KPIs every N seconds, persists to TimescaleDB, fires Synapse events on regression/recovery/RE progress. Also measures per-domain specialization KPIs and emits mastery/decline/profitability signals.

---

## What's Implemented

### Core Service (`service.py`)
- `BenchmarkService` — full lifecycle (`initialize`, `shutdown`, `_run_loop`, `_monthly_eval_loop`)
- All **7 KPIs** collected concurrently via `asyncio.gather(return_exceptions=True)`:
  1. `decision_quality` — Nova `outcomes_success / total`
  2. `llm_dependency` — Nova `slow_path / total` (inverted regression semantics)
  3. `economic_ratio` — Oikos `revenue_7d / costs_7d` (Decimal-safe)
  4. `learning_rate` — Evo cumulative delta (restart-safe; §26.2 fix)
  5. `mutation_success_rate` — Simula `proposals_approved / received`
  6. `effective_intelligence_ratio` — Telos `last_effective_I`
  7. `compression_ratio` — Logos `intelligence_ratio`
- **6 `@runtime_checkable` Protocol types** for upstream dependencies (no `Any`)
- **Rolling avg + regression detection** — per-KPI, `_regressed` set de-duplicated
- **`_regressed` persisted to Redis** (`eos:benchmarks:regressed:{instance_id}`) — restart-safe
- **BENCHMARK_REGRESSION** — emitted once per regression, re-arms on recovery
- **BENCHMARK_RECOVERY** — emitted with `duration_regressed` on re-arm
- **BENCHMARK_RE_PROGRESS** — emitted when `llm_dependency` improves >5% cycle-over-cycle
- **Sustained llm_dependency alert** — 30-snapshot half-window comparison
- **Neo4j episode tagging** — tags `(ep:Episode {used_re:true, outcome_success:false})` when decision_quality regresses + llm_dependency > 0.5
- **`record_kpi()` batch API** — accepts `metrics=dict[str, Any]` (Simula/Synapse) or `metric+value` (Soma); all callers now land correctly
- **TimescaleDB schema** — `benchmark_snapshots` + `benchmark_aux` + hypertable promotion; includes `bedau_packard JSONB`, `evolutionary_fitness JSONB`, and `constitutional_phenotype_divergence DOUBLE PRECISION` columns (idempotent migration via `ALTER TABLE ADD COLUMN IF NOT EXISTS`)
- **Query interface** — `latest_snapshot()`, `trend(metric, since, limit)`, `all_trends(since)`
- **Health endpoint** + **`stats` property** — all 7 KPIs exposed

### Phase 3 — Bedau-Packard Intelligence-Ratio Time-Series (`service.py`)
- Subscribes to `TELOS_POPULATION_SNAPSHOT` via Synapse (6th inbound subscription)
- `_on_telos_population_snapshot`: fingerprints drive-weight cluster centroids (rounded to 2dp), computes `adaptive_activity_A` = novel-and-persistent configs, computes `constitutional_phenotype_divergence` = mean per-drive variance (primary: `drive_weight_distribution.{drive}.variance`; fallback: centroid variance across clusters)
- Persists `(:BedauPackardSample)` Neo4j nodes idempotently (`MERGE` on `node_id`)
- Emits `BENCHMARKS_EVOLUTIONARY_ACTIVITY` to Evo + Nexus each snapshot
- `constitutional_phenotype_divergence` cached as `_last_phenotype_divergence`; included in next `_collect()` cycle snapshot
- `BenchmarkSnapshot.constitutional_phenotype_divergence: float | None` — new 8th KPI field

### Evolutionary Tracker (`evolutionary_tracker.py`)
- Subscribes to `EVOLUTIONARY_OBSERVABLE` via Synapse
- Computes Bedau-Packard: total_activity, mean_activity, diversity_index (Shannon entropy), evolutionary_rate, persistence (parent observable overlap via Redis)
- Emits `BEDAU_PACKARD_SNAPSHOT` each cycle
- Persisted to Redis (`eos:benchmarks:evolutionary_observables:{instance_id}`) — restart-safe
- Restores history on startup via `restore_from_redis()`

### Synapse Subscriptions (8 inbound)
| Event | Handler | Purpose |
|---|---|---|
| `EVOLUTIONARY_OBSERVABLE` | EvolutionaryTracker | Bedau-Packard population stats |
| `SOMA_ALLOSTATIC_REPORT` | `_on_soma_allostatic_report` | Correlate allostatic_efficiency with economic_ratio |
| `COHERENCE_SNAPSHOT` | `_on_coherence_snapshot` | Correlate coherence with decision_quality |
| `EFFECTIVE_I_COMPUTED` | `_on_effective_i_computed` | Track per-instance effective_I |
| `KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE` | `_on_kairos_i_ratio_step` | Log compression ratio step changes |
| `TELOS_POPULATION_SNAPSHOT` | `_on_telos_population_snapshot` | Compute Bedau-Packard `adaptive_activity_A` from drive-weight phenotype fingerprints; compute `constitutional_phenotype_divergence`; persist `(:BedauPackardSample)` node; emit `BENCHMARKS_EVOLUTIONARY_ACTIVITY` |
| `BENCHMARKS_METABOLIC_VALUE` | `_on_metabolic_value` | Push-based metabolic efficiency time-series (168-sample 7-day deque); emits `BENCHMARK_REGRESSION` when latest reading < 90% of rolling mean and trend slope is negative — detects economic collapse within one consolidation cycle instead of the 24h poll window |
| `RE_DECISION_OUTCOME` | `_on_re_decision_outcome` | Tracks RE model performance in a 7-day rolling window (`_re_outcomes`). Computes `success_rate` + `usage_pct`. Stores in `_re_performance` dict, included in monthly eval Neo4j node. |
| `CHILD_SPAWNED` | `_on_child_spawned_genome` | Cache child genome snapshot (`_fleet_genomes`) for monthly Bedau-Packard fleet-level computation |

---

## Event Coverage Fix (2026-03-07)

**Root cause of 0% event coverage**: `_run_loop` slept 10s on startup then immediately entered `await asyncio.sleep(interval_s)` where `interval_s = 86400.0` (24 hours). All 5 spec-expected events (`BEDAU_PACKARD_SNAPSHOT`, `BENCHMARK_REGRESSION`, `BENCHMARK_RE_PROGRESS`, `BENCHMARK_RECOVERY`, `BENCHMARKS_EVOLUTIONARY_ACTIVITY`) are emitted inside `_run_loop` — so nothing ever fired in a real session.

**Fix**: `_run_loop` now uses a `first_run` flag to skip the `interval_s` sleep on the very first iteration. After the 10s warm-up the loop collects immediately, then waits 86400s between subsequent runs. `BENCHMARKS_EVOLUTIONARY_ACTIVITY` fires reactively via `_on_telos_population_snapshot` and is unaffected.

---

## Round 2C — Test Sets + Monthly Scheduler (7 Mar 2026)

### Test Sets Created (`data/evaluation/`)
All 6 JSONL files now exist with seed content. See `data/evaluation/README.md` for full schemas.

| File | Items | Key use |
|------|-------|---------|
| `domain_tests.jsonl` | 50 | P1 Specialization Index (domain score) |
| `general_tests.jsonl` | 50 | P1 Specialization Index (general retention) |
| `cladder_tests.jsonl` | 30 | P3 Causal Reasoning (L1/L2/L3 CLadder) |
| `ccr_gb_tests.jsonl` | 20 | P3 Causal Reasoning (CCR.GB fictional worlds) |
| `constitutional_scenarios.jsonl` | 30 | P5 Ethical Drift Map (FROZEN) |
| `held_out_episodes.jsonl` | 20 | P2 Novelty Emergence (FROZEN, freeze_date 2026-03-07) |

Target per speciation bible: 200/200/200/100/100/100. Current counts are seed.

### Monthly Eval Scheduler (`service.py`)
`_monthly_eval_loop()` now runs as a second background task alongside `_run_loop`:
- 15s startup delay (after all systems are ready)
- Loads all 6 test sets via `TestSetManager.load_all()`
- Calls `EvaluationProtocol.run_monthly_evaluation()` with current test sets + RE service
- Emits `MONTHLY_EVALUATION_COMPLETE` with `result.to_dict()` payload
- Sleeps until the 1st of the next month at 03:00 UTC between runs
- Cancelled gracefully in `shutdown()`

### RE Service Wiring
- `EvaluationProtocol.set_re_service(re)` — new method on `evaluation_protocol.py`
- `BenchmarkService.set_re_service(re)` — delegates to `_evaluation_protocol`
- `core/registry.py._init_benchmarks()` — now accepts `memory` + `re_service` params;
  calls `benchmarks.set_memory(memory)` + `benchmarks.set_re_service(re_service)` after `initialize()`
- `_evaluation_protocol` + `_test_set_manager` + `_monthly_eval_task` added to `BenchmarkService.__init__`

### Field-Name Compatibility
`constitutional_scenarios.jsonl` uses `scenario`/`drives_in_tension`/`expected_analysis` fields.
`test_sets.py:load_constitutional_scenarios()` expects `context`/`drives_in_conflict`/`conflict_description`.
`evaluation_protocol.py:_eval_set()` bridges this via priority chains:
- episode_context: `prompt` → `question` → `context` → `scenario`
- expected: `expected_answer` → `answer` → `expected` → `expected_analysis`

---

## Evaluation Framework (Added 7 Mar 2026)

The five-pillar monthly evaluation protocol from the speciation bible §6.2–6.5 is now implemented as a separate capability alongside (not replacing) the existing 7 KPIs.

### New Files

| File | Purpose |
|---|---|
| `shadow_reset.py` | `ShadowResetController` — non-destructive population snapshot + adaptive delta |
| `evaluation_protocol.py` | `EvaluationProtocol` — 5-pillar monthly evaluation framework |
| `test_sets.py` | `TestSetManager` — JSONL test set loader for all 5 pillars |
| `data/evaluation/README.md` | Schema docs for all 6 test set formats |
| `cli/evaluate.py` | 4 CLI commands: `monthly`, `shadow-snapshot`, `shadow-delta`, `learning-velocity` |

### Shadow-Reset Controller (`shadow_reset.py`)

Non-destructive. Snapshots current population state (observable types, frequencies, novelty rate, Shannon diversity) to Redis at `eos:benchmarks:shadow_snapshot:{snapshot_id}`. Compares later to measure:

- `activity_drop_pct` — how much novelty rate dropped since snapshot
- `diversity_change_pct` — how Shannon entropy changed
- `jaccard_overlap` — fraction of observable types shared between then and now
- `is_adaptive` — True when activity_drop_pct > 50% (Bedau & Packard criterion)
- `diversity_recovery_time` — seconds since snapshot when diversity recovered (None if not yet)

**Bible §6.4 key insight:** A dramatic drop in adaptive activity post-reset proves the dynamics are genuinely adaptive (organisms react to population-state changes), not statistical drift (which is insensitive to history). Near-zero drop = drift.

`BenchmarkService.take_shadow_snapshot()` → delegates to controller + emits `SHADOW_RESET_SNAPSHOT`
`BenchmarkService.compute_shadow_delta(snapshot_id)` → delegates + emits `SHADOW_RESET_DELTA`

### Five-Pillar Evaluation Protocol (`evaluation_protocol.py`)

All pillars are **callable but return stub results** (is_stub=True) until test sets are created and RE is operational. The framework exists now; Round 2 fills in the data.

| Pillar | Method | Status | Key metric |
|---|---|---|---|
| P1 Specialization Index | `measure_specialization()` | STUB (needs RE; test sets present) | SI > 0.1 = genuine specialization |
| P2 Novelty Emergence | `measure_novelty_emergence()` | STUB (needs RE; held-out set present) | High success + high cosine distance |
| P3 Causal Reasoning | `measure_causal_reasoning()` | STUB (needs RE; CLadder + CCR.GB present) | L2 + L3 CLadder accuracy improving |
| P4 Learning Velocity | `measure_learning_velocity()` | CALLABLE with historical data | velocity > 0.02 = accelerating |
| P5 Ethical Drift Map | `measure_ethical_drift()` | STUB (needs RE; constitutional set present) | Drift vector + INV-017 guard |

**Pillar 4 is the exception:** it operates on a list of `{month, score}` dicts — no RE required. It fits a power law and falls back to linear regression if scipy is unavailable. The CLI `learning-velocity` command calls it directly.

### Test Set Manager (`test_sets.py`)

`TestSetManager` loads JSONL files from `data/evaluation/` (configurable). All loaders return `[]` if the file does not exist. Call `await mgr.load_all()` to get the full dict for `run_monthly_evaluation()`.

Test set files (seed counts from Round 2C; targets in parentheses):
- `domain_tests.jsonl` — 50 EOS domain tasks (target: 200)
- `general_tests.jsonl` — 50 general reasoning tasks (target: 200)
- `held_out_episodes.jsonl` — 20 never-seen episodes, FROZEN (target: 100)
- `cladder_tests.jsonl` — 30 CLadder L1/L2/L3 questions (target: 200)
- `ccr_gb_tests.jsonl` — 20 CCR.GB fictional world tests (target: 100)
- `constitutional_scenarios.jsonl` — 30 catch-22 drive dilemmas, FROZEN (target: 100)

### CLI Commands

```bash
# Run from backend/
python -m cli.evaluate monthly                      # 5-pillar evaluation (stubs today)
python -m cli.evaluate monthly --month 3 --re-version v0.3
python -m cli.evaluate shadow-snapshot              # Take population snapshot
python -m cli.evaluate shadow-delta <snapshot_id>  # Compute adaptive delta
python -m cli.evaluate learning-velocity            # Power-law velocity fit
python -m cli.evaluate learning-velocity --history-file data/velocity_history.json
```

### New SynapseEventType entries

| Event | Emitter | Subscribers |
|---|---|---|
| `SHADOW_RESET_SNAPSHOT` | BenchmarkService | Alive (visualization), Evo (population state) |
| `SHADOW_RESET_DELTA` | BenchmarkService | Evo (adaptive-dynamics evidence for hypothesis scoring) |
| `MONTHLY_EVALUATION_COMPLETE` | BenchmarkService | Thread (chapter milestone), Evo (training trigger), Nova (Thompson recalibration) |

### What's NOT yet wired

- No FastAPI endpoints for `/benchmarks/shadow-snapshot` or `/benchmarks/shadow-delta` — CLI falls back to Redis directly

---

## Round 4D — Bedau-Packard Fleet Tracker + Red-Team Scheduler (7 Mar 2026)

### New file: `bedau_packard.py`

`BedauPackardTracker` — fleet-level Bedau-Packard evolutionary activity statistics (§8.5).

**Neo4j persistence (added 2026-03-07):** Each `compute_adaptive_activity()` call persists a `(:BedauPackardSample)` node via MERGE — required for `PaperDataExporter._export_evolutionary_activity()` which queries these nodes to build `evolutionary_activity.csv`.

| Detail | Value |
|--------|-------|
| Node label | `BedauPackardSample` |
| Merge key | `node_id = "bp_fleet:{instance_id}:{month}"` |
| Properties | `instance_id`, `month`, `adaptive_activity`, `novelty_rate`, `diversity_index`, `population_size`, `component_count`, `novel_component_count`, `exceeds_shadow`, `oee_verdict`, `recorded_at` |
| Write pattern | `asyncio.ensure_future(_persist_snapshot(snap))` — fire-and-forget, non-fatal |
| Injection | `set_neo4j(neo4j, instance_id)` called from `BenchmarkService.set_memory()` when `memory._neo4j is not None` |

| Class | Purpose |
|---|---|
| `EvolutionaryComponent` | Single evolvable component: hypothesis / drive_weight / config_param |
| `EvolutionaryActivitySnapshot` | Monthly snapshot: adaptive_activity, novelty_rate, diversity, shadow_activity, etc. |
| `BedauPackardTracker` | Ingests fleet genome snapshots, computes A(t), shadow control, OEE evidence |

**Key methods:**
- `ingest_fleet_genomes(genomes)` — extracts components from Evo/Simula/Telos genome dicts
- `compute_adaptive_activity(components, month)` — computes A(t) = novel + persistent; shadow control at 5% random rate; returns `EvolutionaryActivitySnapshot`
- `assess_oee_evidence()` — available after ≥3 months; returns `verdict` in {"insufficient_data", "bounded", "growing_not_shadow_controlled", "exceeds_bounded"}; **never claims "open-ended evolution"** — uses "exceeds bounded classification" per §8.5

**Shadow control:** ~5% random persistence probability per component — first-order approximation for single instance; accuracy improves with fleet size ≥5.

### Fleet genome caching (`service.py`)

- `_fleet_genomes: dict[str, dict]` — instance_id → genome snapshot, populated from `CHILD_SPAWNED` events
- `_on_child_spawned_genome(event)` — caches `{instance_id, evo, simula, telos, equor}` sub-keys from CHILD_SPAWNED payload
- `_collect_fleet_genomes()` — returns cached snapshots; empty list when no children yet (single-instance mode)
- `_bp_tracker = BedauPackardTracker(speciation_threshold=config.mitosis...)` — initialised in `__init__`
- `_monthly_eval_count: int` — incremented each month; gates OEE assessment at ≥3

### Monthly eval integration (`_monthly_eval_loop`)

After 5-pillar evaluation:
1. `_collect_fleet_genomes()` → `_bp_tracker.ingest_fleet_genomes()` → `compute_adaptive_activity(month=now.month)`
2. Adds `result_dict["evolutionary_activity"]` with: month, adaptive_activity, novelty_rate, diversity, exceeds_shadow, population_size, component_count, novel_component_count, oee_verdict
3. After month ≥3: adds `result_dict["oee_assessment"]` = full `assess_oee_evidence()` dict
4. Emits `EVOLUTIONARY_ACTIVITY_COMPUTED` on Synapse (Evo + Nexus + Alive)
5. Non-fatal: any exception logged as warning, monthly eval continues

### New Synapse subscription (Round 4D)

| Event | Handler | Purpose |
|---|---|---|
| `CHILD_SPAWNED` | `_on_child_spawned_genome` | Cache child genome for monthly Bedau-Packard |

### New SynapseEventType

| Event | Purpose |
|---|---|
| `EVOLUTIONARY_ACTIVITY_COMPUTED` | Monthly Bedau-Packard snapshot; payload: month, adaptive_activity, novelty_rate, diversity, exceeds_shadow, population_size, component_count, novel_component_count, oee_verdict |

---

## Round 4C — Ethical Drift Map + Longitudinal Evaluation (7 Mar 2026)

### New Files

| File | Purpose |
|---|---|
| `ethical_drift.py` | `EthicalDriftEvaluator` + `EthicalDriftTracker` — Pillar 5 full implementation |
| `longitudinal.py` | `LongitudinalTracker` — Month 1 baseline capture + Month 1 vs Month N comparison |
| `data/evaluation/ethical_drift_scenarios.jsonl` | 100 frozen catch-22 dilemmas — NEVER modify, NEVER include in training |

### Ethical Drift Map (`ethical_drift.py`)

**`EthicalDriftEvaluator`** — runs all 100 frozen scenarios through the RE each month:
- `load_scenarios()` — loads `ethical_drift_scenarios.jsonl` once; never re-read mid-run
- `evaluate(re_service, month, instance_id)` → `MonthlyDriftRecord`
- `_infer_dominant_drive(reasoning, scenario)` — keyword-frequency heuristic restricted to conflict drives
- `_score_drive_activation(reasoning)` → `dict[str, float]` normalized 0–1 (max drive = 1.0)
- `_extract_chosen_option(decision, options)` — substring overlap matching

**`EthicalDriftTracker`** — persists records and computes drift:
- `record_month(record)` — computes drift vector vs Month 1 baseline; persists `(:EthicalDriftRecord)` to Neo4j (fire-and-forget); emits `ETHICAL_DRIFT_RECORDED`
- `_get_baseline / _set_baseline` — `(:EthicalDriftBaseline)` Neo4j nodes, keyed by `instance_id`
- `compute_population_divergence(records)` — Euclidean distance in drive_means space; `is_speciation_signal = True` when mean distance > 0.2

**Data types:**
- `ScenarioResult` — per-scenario: scenario_id, drive_conflict, chosen_option, dominant_drive, drive_scores, reasoning_excerpt, confidence
- `MonthlyDriftRecord` — month, instance_id, drive_means, drift_vector, drift_magnitude, dominant_drive_distribution, scenario_results

### Longitudinal Tracker (`longitudinal.py`)

**`LongitudinalSnapshot`** — evaluation scores at a specific month:
- Pillar 1: specialization_index, domain_improvement, general_retention
- Pillar 3: l1_association, l2_intervention (key paper metric), l3_counterfactual, ccr_validity
- Pillar 5: drift_magnitude, dominant_drive
- RE: re_success_rate, re_usage_pct; adapter_path for reproducibility

**`LongitudinalTracker`** — multi-month comparison:
- `record_month(month, eval_results, re_performance, adapter_path)` → `LongitudinalSnapshot`; Month 1 snapshot also stored as `(:LongitudinalBaseline)`
- `compare_to_baseline(current)` → dict with per-pillar deltas + `verdict`
- `_compute_verdict(current, baseline)` — five mutually exclusive verdicts:
  - `continuous_learning_demonstrated` — L2 +10pp, L3 +5pp vs baseline
  - `partial_improvement` — L2 +5pp only
  - `stable_no_forgetting` — L2 within ±5pp, retention intact
  - `catastrophic_forgetting` — general retention < 85% of baseline
  - `plasticity_loss_suspected` — L2 regressed, retention intact

### Wiring in `service.py`

**New imports:** `EthicalDriftEvaluator`, `EthicalDriftTracker`, `LongitudinalTracker`

**`__init__` additions:**
- `self._ethical_drift = EthicalDriftEvaluator()` — loads scenarios lazily on first evaluate()
- `self._drift_tracker = EthicalDriftTracker(memory=None)` — memory injected via set_memory()
- `self._longitudinal = LongitudinalTracker(memory=None, instance_id=instance_id)` — same
- `self._current_month: int = 1` — monotonic counter; independent of calendar month

**`set_event_bus`:** wires `self._drift_tracker.set_event_bus(bus)`

**`set_memory`:** also sets `self._drift_tracker._memory` and `self._longitudinal._memory`

**`_monthly_eval_loop` additions** (inserted before MONTHLY_EVALUATION_COMPLETE emit):
1. Pillar 5: `_ethical_drift.evaluate()` → `_drift_tracker.record_month()` → `result_dict["ethical_drift"]`
2. Longitudinal: `_longitudinal.record_month()` → `compare_to_baseline()` → `result_dict["longitudinal_comparison"]`
3. `self._current_month += 1` after both complete

### Ethical Drift Scenarios

100 scenarios across 4 conflict types (25 each):
- `survival_vs_care` — ed_001, ed_005, ed_007 (growth_vs_survival), ed_021, ed_025, ed_029, ed_040, ed_045, ed_051, ed_057, ed_066, ed_067, ed_069, ed_070, ed_074, ed_085, ed_089, ed_091, ed_093, ed_095, ed_096, ed_098...
- `growth_vs_honesty` — ed_012, ed_017, ed_026, ed_027, ed_033, ed_041, ed_054, ed_059, ed_073, ed_077...
- `coherence_vs_survival` — ed_004, ed_006, ed_013, ed_019, ed_023, ed_028, ed_030, ed_032, ed_036, ed_043, ed_048, ed_049, ed_050, ed_055, ed_058, ed_060, ed_063, ed_065, ed_067, ed_069, ed_070, ed_075, ed_076, ed_079, ed_080, ed_082, ed_084, ed_086, ed_087, ed_090, ed_091, ed_097, ed_099...
- `care_vs_growth` — ed_003, ed_014, ed_018, ed_035, ed_037, ed_046, ed_062, ed_064, ed_068, ed_072, ed_078, ed_081, ed_083...

**FROZEN — never modify, never include in training data, never add to exclusion list that ships training JSONL files.**

### New SynapseEventType

| Event | Payload |
|---|---|
| `ETHICAL_DRIFT_RECORDED` | `{month, instance_id, drift_magnitude, dominant_drive, drift_vector, drive_means}` |

### Neo4j Nodes Created

| Node | Key fields |
|---|---|
| `(:EthicalDriftRecord)` | instance_id + month (merge key), drift_magnitude, dominant_drive, drive_means_json, drift_vector_json |
| `(:EthicalDriftBaseline)` | instance_id (merge key), drive_means_json, month |
| `(:LongitudinalSnapshot)` | node_id = `longitudinal:{iid}:{month}`, full snapshot_json, all metric fields |
| `(:LongitudinalBaseline)` | instance_id (merge key), snapshot_json, month |

### Remaining Gaps

- Population-level divergence (`compute_population_divergence`) requires ≥2 live instances; will activate as fleet grows — not called from service.py yet
- `ethical_drift_scenarios.jsonl` exclusion from RE training pipeline: must be added to the same exclusion mechanism as `red_team_prompts.jsonl` and anchor prompts in `scripts/re/`
- Longitudinal `compare_to_baseline` on Month 1 returns `{"no_baseline": True}` — first run baseline is set and returned in the same call, so Month 1 effectively has a no-delta comparison

---

## Round 5C — Population Divergence + `run_evaluation_now()` (7 Mar 2026)

### Population Divergence (monthly eval) — Round 6 upgrade

Wired into `_monthly_eval_loop()` immediately after the Bedau-Packard block.

- **Condition**: `len(self._fleet_genomes) >= 2` — no-op for single-instance deployments
- **Primary metric** (Round 6): real per-drive ethical drift records from Neo4j for each fleet instance at the current month. Queries `(:EthicalDriftRecord {instance_id, month})` for each cached fleet instance. If ≥2 records found, calls `EthicalDriftTracker.compute_population_divergence()` — Euclidean distance in drive_means space, `is_speciation_signal=True` when mean distance > 0.2.
- **Fallback metric**: genome structural distance proxy (evo 30%, simula 25%, telos 25%, equor 20%) used when < 2 ethical drift records are available (early months / no RE service yet).
- **`divergence_source`** field in output: `"ethical_drift"` (primary) or `"genome_distance_proxy"` (fallback)
- **Primary output keys** (ethical drift): `divergence`, `max_divergence`, `pairs_compared`, `is_speciation_signal`, `population_size`, `divergence_source`
- **Proxy output keys** (genome distance): `mean_genome_distance`, `max_genome_distance`, `pairs_compared`, `population_size`, `speciation_detected`, `speciation_threshold`, `divergence_source`
- Both branches are non-fatal; any exception logged as warning and monthly eval continues

### `run_evaluation_now(month=None)` — on-demand evaluation

New public method for ablation studies and manual CLI invocations.

```python
snap = await benchmark_service.run_evaluation_now(month=3)
# Returns LongitudinalSnapshot with all 5-pillar scores
```

- Accepts optional `month` override (defaults to `_current_month`)
- Does **NOT** increment `_current_month` — read-only evaluation pass
- Does **NOT** persist to Neo4j — that only happens in the scheduled monthly loop
- Raises `RuntimeError` if called before `initialize()`
- Returns a `LongitudinalSnapshot` (same type as the scheduled loop produces)

### Remaining gaps

- ~~Population divergence used genome structural distance as proxy~~ — **RESOLVED (Round 6)**: now queries Neo4j for real `EthicalDriftRecord` per fleet instance; genome distance used only as fallback when < 2 records available
- `run_evaluation_now()` calls `_longitudinal.record_month()` which sets the baseline if month == 1; calling it multiple times in the same month would overwrite the baseline — consider adding a `dry_run=True` flag to `record_month()` in a future pass

---

## Round 5D — Ablation Studies + Paper Data Pipeline (7 Mar 2026)

### New Files

| File | Purpose |
|---|---|
| `ablation.py` | `AblationOrchestrator` — 5-mode ablation study framework |
| `paper_data.py` | `PaperDataExporter` — 4 CSV exports + W&B push |

### Ablation Framework (`ablation.py`)

**`AblationMode` (StrEnum):** 5 modes matching the speciation bible §9 contribution table:
- `stream_2_off` — remove failure+correction examples from Tier 2 dataset
- `stream_4_off` — remove causal-chain examples from Tier 2 dataset
- `replay_off` — disable SurprisePrioritizedReplay (no historical mixing)
- `dpo_off` — disable constitutional DPO pass (no alignment fine-tuning)
- `anti_forgetting_off` — bypass full SuRe EMA + SafeLoRA + KL gate + perplexity stack

**`AblationResult`** dataclass: mode, month, instance_id, l2_delta, l3_delta, baseline_l2/l3, ablated_l2/l3, conclusion, elapsed_s, error.

**`AblationOrchestrator`** lifecycle:
1. `run_all(month)` — evaluates full-stack baseline, then runs all 5 ablation modes
2. `run_one(mode, month)` — single mode run
3. Per mode: `_train_ablated()` sets `cl._ablation_mode` synchronously, calls `run_tier2()`, restores original adapter in `finally`
4. `run_evaluation_now()` called before + after to capture L2/L3 delta
5. ABLATION_STARTED / ABLATION_COMPLETE emitted per mode
6. `(:AblationResult)` Neo4j nodes persisted (fire-and-forget, non-fatal)

**Integration with `ContinualLearningOrchestrator`:**
- `_ablation_mode: str = "none"` field added to CLO `__init__`
- Stream filtering: `_execute_tier2()` Step 1b — strips stream_id 2 or 4 from exported JSONL when mode is `stream_2_off`/`stream_4_off`
- Replay bypass: Step 3b condition `and self._ablation_mode not in ("replay_off", "anti_forgetting_off")`
- Anti-forgetting bypass: Steps 6b–6e wrapped in `if not _anti_forgetting_disabled` when mode is `anti_forgetting_off`; Step 6d always runs (fast adapter deployed directly)

### Paper Data Exporter (`paper_data.py`)

**`PaperDataExporter`** — exports 4 CSVs after each monthly evaluation:

| CSV | Source nodes | Key columns |
|-----|-------------|-------------|
| `longitudinal_results.csv` | `(:LongitudinalSnapshot)` | month, L2/L3 accuracy, specialization_index, re_success_rate |
| `ablation_results.csv` | `(:AblationResult)` | mode, l2_delta, l3_delta, conclusion |
| `evolutionary_activity.csv` | `(:BedauPackardSample)` | month, adaptive_activity, novelty_rate, oee_verdict |
| `ethical_drift.csv` | `(:EthicalDriftRecord)` | month, drift_magnitude, dominant_drive, per-drive columns |

**W&B integration:** all W&B calls inside `if wandb_available:` guards — never crashes if wandb not installed. Uses `wandb.Artifact` type `dataset`. Run name: `paper_export_month_{N}`.

**Wired in `BenchmarkService`:**
- `self._paper_exporter = PaperDataExporter(memory=None, instance_id=...)` in `__init__`
- `set_memory()` calls `self._paper_exporter.set_memory(memory)`
- `_monthly_eval_loop()` fires `asyncio.ensure_future(self._paper_exporter.export_all(month=self._current_month - 1))` after `MONTHLY_EVALUATION_COMPLETE` emit (fire-and-forget)

### `run_evaluation_now(month)` — on-demand evaluation

New public method on `BenchmarkService`. Called by `AblationOrchestrator`.

- Runs all 5 pillars synchronously (same protocol as monthly loop)
- Does **NOT** increment `_current_month`
- Does **NOT** persist to Neo4j or emit events
- Returns `LongitudinalSnapshot` with current pillar scores
- Graceful fallback: returns empty snapshot if evaluation protocol not initialised

### New SynapseEventType entries

| Event | Emitter | Payload |
|---|---|---|
| `ABLATION_STARTED` | `AblationOrchestrator` | `{mode, month}` |
| `ABLATION_COMPLETE` | `AblationOrchestrator` | `{mode, month, l2_delta, l3_delta, conclusion}` |

### Neo4j Nodes Created

| Node | Key fields |
|---|---|
| `(:AblationResult)` | node_id = `ablation:{iid}:{month}:{mode}`, l2_delta, l3_delta, conclusion, elapsed_s |

---

## Round 6 — Pillars 1–4 + Memorization Detection (`pillars.py`) (7 Mar 2026)

### New File: `pillars.py`

Implements bible §6.2 Pillars 1–4 and §6.3 Memorization Detection as a standalone module.
Runs **alongside** (not replacing) `EvaluationProtocol` in `_monthly_eval_loop()`.

| Symbol | Pillar | Key metric |
|--------|--------|-----------|
| `measure_specialization(custom, base, domain_test, general_test)` | P1 Specialization Index | SI = (cd-bd)-(bg-cg). >0.1 genuine, >0.3 publishable |
| `measure_novelty_emergence(engine, novel_episodes)` | P2 Novelty Emergence | success_rate + cosine_distance; genuine_learning when both >threshold |
| `measure_causal_reasoning(engine, cladder_questions, ccr_gb_scenarios)` | P3 Causal Reasoning | l2_intervention (KEY), l3_counterfactual, ccr_validity |
| `compute_learning_velocity(history)` | P4 Learning Velocity | power-law fit; velocity <0.005 = plateaued, >0.02 = accelerating |
| `detect_memorization(engine, training, holdout, paraphrase_pairs)` | §6.3 Memorization | MI accuracy, paraphrase drop, SVD intruder ratio; risk low/medium/high |
| `load_fixed_test_sets()` | Loader | Reads all 6 JSONL files from `data/evaluation/`; warns on missing |

**Result dataclasses:** `SpecializationResult`, `NoveltyEmergenceResult`, `CausalReasoningResult`, `LearningVelocityResult`, `MemorizationReport`

### Fixed Test Sets (`data/evaluation/`)

**FROZEN post Week 7 (bible §10 Phase 1 Week 7) — never modify, never include in training.**

| File | Target count | Pillar | Notes |
|------|-------------|--------|-------|
| `domain_test_200.jsonl` | 200 | P1 Specialization (domain) | Schema: `{question, answer}` |
| `general_test_200.jsonl` | 200 | P1 Specialization (general) | Schema: `{question, answer}` |
| `novel_episodes_100.jsonl` | 100 | P2 Novelty Emergence | Schema: `{question, answer}`. FROZEN. |
| `cladder_200.jsonl` | 200 | P3 Causal Reasoning | Schema: `{question, answer, rung: 1|2|3}`. Download from Jin et al. NeurIPS 2023. |
| `ccr_gb_100.jsonl` | 100 | P3 Causal Reasoning | Schema: `{scenario, ground_truth, world_model}`. Maasch et al. ICML 2025. |
| `paraphrase_pairs_50.jsonl` | 50 | §6.3 Memorization | Schema: `{original, paraphrase, answer}` |

All 6 files ship as 1-line placeholders. **Operator must replace with real content before Month 1 eval.**

### Wiring in `service.py`

**New fields (`__init__`):**
- `self._reasoning_engine: Any | None` — set via `set_reasoning_engine(engine)` post-`initialize()`
- `self._test_sets: dict` — loaded once in `initialize()` via `load_fixed_test_sets()`
- `self._causal_history: list[dict]` — accumulates `{"month": int, "score": float}` each month from Pillar 3 (L2+L3 mean); fed into `compute_learning_velocity()`

**New method:** `set_reasoning_engine(engine)` — distinct from `set_re_service()` (which targets `EvaluationProtocol`).

**`initialize()` addition:** `self._test_sets = load_fixed_test_sets()` — runs synchronously, warns on missing files.

**`_monthly_eval_loop()` addition:** After longitudinal tracking and before `_current_month += 1`, a new block runs all 4 pillars + memorization detection (non-fatal; guarded by `if self._reasoning_engine and self._test_sets`). Adds keys to `result_dict`:
- `pillar1_specialization_index`, `pillar1_domain_improvement`, `pillar1_general_retention`
- `pillar2_novel_success_rate`, `pillar2_cosine_distance`, `pillar2_genuine_learning`
- `pillar3_l2_intervention`, `pillar3_l3_counterfactual`, `pillar3_ccr_validity`
- `pillar4_velocity`, `pillar4_is_plateaued`, `pillar4_predicted_month_12`
- `memorization_risk`, `memorization_mi_accuracy`, `memorization_paraphrase_drop`

**`_causal_history` accumulation:** One entry appended per month when Pillar 3 completes. `compute_learning_velocity()` requires ≥3 entries before producing a non-trivial result (`insufficient_data=True` before that).

### Import addition (`service.py`)

```python
from systems.benchmarks.pillars import (
    compute_learning_velocity,
    detect_memorization,
    load_fixed_test_sets,
    measure_causal_reasoning,
    measure_novelty_emergence,
    measure_specialization,
)
```

---

## Domain KPI System (8 Mar 2026)

### Overview

Per-specialization KPI measurement so the organism can determine which domain is worth specializing in. Answers: "Is my $50/month yield revenue worth more than my $30/month code delivery, and which is improving?"

### New Type: `DomainKPI` (`types.py`)

```python
class DomainKPI(EOSBaseModel):
    domain: str                          # e.g. "software_development", "yield"
    timestamp: datetime
    attempts: int; successes: int; success_rate: float
    revenue_total_usd: Decimal; cost_total_usd: Decimal; net_profit_usd: Decimal
    profitability: float                 # net_profit / revenue; 0.0 if no revenue
    revenue_per_hour: Decimal; revenue_per_attempt: Decimal
    hours_spent: float; tasks_completed: int; avg_task_duration_hours: float
    customer_satisfaction: float         # from custom_metrics["customer_satisfaction"]
    rework_rate: float                   # from custom_metrics["rework_rate"]
    custom_metrics: dict[str, float]     # domain-specific averaged metrics
    trend_direction: str                 # "stable" | "improving" | "declining"
    trend_magnitude: float               # |delta| in success_rate vs prior half-period
    lookback_hours: int                  # default 168 (7 days)
```

`BenchmarkSnapshot` extended with `domain_kpis: dict[str, DomainKPI] = {}` and `primary_domain: str = "generalist"`.

### New File: `domain_kpi_calculator.py`

`DomainKPICalculator` — stateful in-process `deque[EpisodeRecord]` (max 10,000):
- `record_episode(data)` — ingests from `DOMAIN_EPISODE_RECORDED` event payload
- `calculate_for_domain(domain, lookback_hours=168)` — computes full `DomainKPI`; trend = compare `[now-168h, now]` vs `[now-336h, now-168h]`; `threshold=0.05` for stable/improving/declining
- `calculate_all(lookback_hours, min_attempts)` — all active domains
- `primary_domain(domain_kpis)` — domain with highest `success_rate`
- `active_domains(min_attempts, lookback_hours)` — domains with enough episodes

`EpisodeRecord` uses `__slots__` + `time.time()` monotonic `recorded_at` for fast cutoff checks.

### New Primitive: `primitives/episodes.py`

`EpisodeOutcome(EOSBaseModel)` — canonical type for emitting `DOMAIN_EPISODE_RECORDED`:
- Fields: `domain`, `outcome`, `revenue`, `cost_usd`, `duration_ms`, `custom_metrics`, `timestamp`, `episode_id`, `source_system`
- `to_bus_payload()` — serialise to Synapse event payload

Domain conventions: `software_development`, `art`, `trading`, `yield`, `bounty_hunting`, `consulting`, `generalist`.

Exported from `primitives/__init__.py` as `EpisodeOutcome`.

### Service Wiring (`service.py`)

**New `__init__` fields:**
- `self._domain_kpi_calc = DomainKPICalculator(max_history=10_000)`
- `self._prev_primary_domain: str = "generalist"`

**New Synapse subscription** (9th inbound — with `hasattr` guard):
- `DOMAIN_EPISODE_RECORDED` → `_on_domain_episode_recorded` — delegates to `_domain_kpi_calc.record_episode(data)`

**New `_collect()` return fields:**
- `domain_kpis: dict[str, DomainKPI]` — from `_collect_domain_kpis()`
- `primary_domain: str` — domain with highest success_rate

**New `_run_loop` additions** (after daily snapshot):
- `_emit_domain_signals(domain_kpis)` — per-domain Synapse events
- `_persist_domain_kpis_neo4j(domain_kpis)` — MERGE `(:DomainKPI)` nodes
- Primary domain pivot detection: if `primary_domain != _prev_primary_domain`, emits `NOVA_GOAL_INJECTED` to inform Nova of specialization shift

### Domain Signal Emission (`_emit_domain_signals`)

Per domain, each daily cycle:

| Condition | Event emitted |
|---|---|
| Always | `DOMAIN_KPI_SNAPSHOT` — full `DomainKPI` dict payload |
| `success_rate > 0.75` AND `attempts >= 5` | `DOMAIN_MASTERY_DETECTED` |
| `revenue_per_hour > Decimal("10")` | `DOMAIN_PROFITABILITY_CONFIRMED` |
| `trend_direction == "declining"` AND `trend_magnitude > 0.15` | `DOMAIN_PERFORMANCE_DECLINING` |

### Neo4j Persistence (`_persist_domain_kpis_neo4j`)

```cypher
MERGE (k:DomainKPI {node_id: "domain_kpi:{instance_id}:{domain}:{date}"})
SET k += {domain, attempts, successes, success_rate, revenue_total_usd, ...}
WITH k
MATCH (i:Instance {instance_id: $instance_id})
MERGE (i)-[:INSTANCE_HAS_KPI]->(k)
```

Idempotent daily MERGE. Silently no-ops if `Instance` node not yet created.

### New `SynapseEventType` Entries

| Event | Purpose |
|---|---|
| `DOMAIN_EPISODE_RECORDED` | Inbound: any system emits when a domain task completes |
| `DOMAIN_KPI_SNAPSHOT` | Daily per-domain snapshot (full `DomainKPI` payload) |
| `DOMAIN_MASTERY_DETECTED` | success_rate > 0.75 AND attempts >= 5 |
| `DOMAIN_PROFITABILITY_CONFIRMED` | revenue_per_hour > $10 |
| `DOMAIN_PERFORMANCE_DECLINING` | declining trend AND magnitude > 0.15 |

### Downstream Integration

**Nova** (`nova/service.py`):
- `DOMAIN_MASTERY_DETECTED` → `_on_domain_mastery`: injects SELF_GENERATED goal (priority=0.85) to continue specializing; deduplicates against active goals
- `DOMAIN_PERFORMANCE_DECLINING` → `_on_domain_performance_declining`: injects investigative goal (priority=0.70) to debug the decline
- `DOMAIN_PROFITABILITY_CONFIRMED` → `_on_domain_profitability_confirmed`: boosts priority of existing goals in that domain by 1.3×

**Thread** (`thread/service.py`):
- `DOMAIN_MASTERY_DETECTED` → `_on_domain_mastery`: ACHIEVEMENT TurningPoint + `narrative_milestone` (milestone_type="domain_mastery")
- `DOMAIN_PERFORMANCE_DECLINING` → `_on_domain_performance_declining`: CRISIS TurningPoint + `narrative_coherence_shift` reassessment

---

## Known Issues / Remaining Gaps

| Gap | Location | Risk |
|---|---|---|
| `record_kpi` data (aux) not in regression detection | `service.py:record_kpi` | Soma/Simula/Synapse telemetry stored in `benchmark_aux` but never surfaced in rolling avg or API trend |
| 24h collection interval too slow for precariousness | `§16.1` | **PARTIALLY MITIGATED**: `_on_metabolic_value` now provides push-based sub-cycle detection for economic efficiency degradation; other KPIs still on 24h poll |
| ~~No RE/Claude routing split~~ | `§3.1` | **PARTIALLY FIXED (2026-03-07)**: `_on_re_decision_outcome` tracks RE outcomes separately in `_re_performance`; `llm_dependency` still collapses both into one KPI for the 7 existing KPIs, but `_re_performance` now surfaces RE success_rate and usage_pct separately in monthly eval |
| No cross-instance aggregation | `§22 Phase 5` | Fleet-wide Bedau-Packard still absent; single-instance only |
| Atune subscription to BENCHMARK_REGRESSION | `§14.1` | No handler in `systems/atune/`; downstream reaction chain aspirational |
| Anomaly detection (z-score, IQR) | `§7` | Not implemented |
| Latency profiling | `§10` | Not implemented |
| Hot-swappable collectors | `§15` | Not implemented |

---

## Integration Surface

### Upstream (pulls from)
| System | Method | Fields |
|---|---|---|
| Nova | `nova.health()` | `outcomes_success`, `outcomes_failure`, `fast_path`, `slow_path`, `do_nothing` |
| Oikos | `oikos.stats` | `revenue_7d`, `costs_7d` |
| Evo | `evo.stats` | `hypothesis.supported` |
| Simula | `simula.stats` | `proposals_approved`, `proposals_received` |
| Telos | `telos.health()` | `last_effective_I`, `last_alignment_gap` |
| Logos | `logos.health()` | `intelligence_ratio`, `cognitive_pressure`, `schwarzschild_met` |

### Downstream (emits to)
| Event | Consumers |
|---|---|
| `BENCHMARK_REGRESSION` | Thymos (→ MEDIUM incident), Soma (→ warning severity) |
| `BENCHMARK_RECOVERY` | Thymos, Evo — close feedback loops |
| `BENCHMARK_RE_PROGRESS` | Nova (Thompson sampling weight update) |
| `BEDAU_PACKARD_SNAPSHOT` | Alive visualization |
| `BENCHMARKS_EVOLUTIONARY_ACTIVITY` | Evo (incorporate A(t) into hypothesis scoring), Nexus (epistemic triangulation signal) |

### Wiring (registry)
```python
# core/registry.py — _init_benchmarks()
benchmarks.set_nova(nova)
benchmarks.set_evo(evo)
benchmarks.set_oikos(oikos)
benchmarks.set_simula(simula)
benchmarks.set_telos(telos)
benchmarks.set_logos(logos)
benchmarks.set_event_bus(synapse.event_bus)
benchmarks.set_redis(redis_client)
benchmarks.set_memory(memory)
await benchmarks.initialize()
```

### Push callers
- **Soma** — calls `record_kpi(system="soma", metric=..., value=...)` (single-metric form)
- **Simula** — calls `record_kpi(system="simula", metrics={...})` (batch form; 3 call-sites)
- **Synapse** — calls `record_kpi(system="synapse", metrics={...})` (batch form; 1 call-site)

---

## Key Design Decisions

- **No cross-system imports** — all data pulled from `health()` / `stats` protocol methods; never imports system internals
- **Each collector fails independently** — `asyncio.gather(return_exceptions=True)` + `errors` dict in snapshot
- **`_regressed` set** persisted to Redis; prevents duplicate alerts across restarts
- **learning_rate restart detection** — if Evo's cumulative resets below stored baseline, re-baselines and logs warning (prevents negative or inflated deltas)
- **`bedau_packard` + `evolutionary_fitness` persisted** — both written to `benchmark_snapshots` JSONB columns; schema idempotently migrated
