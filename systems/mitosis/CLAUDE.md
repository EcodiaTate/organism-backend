# Mitosis System (Spec 26)

Cellular division, genome inheritance, and fleet lifecycle management.

## What's Implemented

### Core Classes

| File | Class | Status |
|------|-------|--------|
| `genome_orchestrator.py` | `GenomeOrchestrator` | Complete - parallel genome extraction, Neo4j persistence, load by ID |
| `mutation.py` | `MutationOperator` | Complete - deterministic RNG mutation with per-segment type ranges |
| `mutation.py` | `MutationRecord` | Complete - audit dataclass persisted to Neo4j |
| `spawner.py` | `LocalDockerSpawner` | Complete - DooD container boot, health check, resource caps |
| `spawner.py` | `SpawnResult` | Complete - spawn outcome with ports/addresses |
| `fleet_service.py` | `MitosisFleetService` | Complete - lifecycle events, health monitor, rescue, gene transfer, speciation, schedulers |

### Integration Points

- **Axon executor** (`axon/executors/mitosis.py`): `SpawnChildExecutor` calls `fleet_service.prepare_child_genome()` before container boot (Step 0), then spawner (Step 1), wallet transfer (Step 2), Oikos registration (Step 3), Synapse event (Step 4)
- **API router** (`api/routers/mitosis.py`): All spawn requests routed through `EquorService.review()` - no bypass. Rescue endpoint at `POST /fleet/rescue/{child_id}`. Dynamic population cap `max(5, floor(net_worth/1000))`.
- **Oikos** (`oikos/mitosis.py`): `MitosisEngine` owns fitness evaluation, niche selection, seed config. `evaluate_fitness()` accepts `max_children_override` for dynamic cap.
- **Synapse events**: `CHILD_SPAWNED`, `CHILD_HEALTH_REPORT`, `CHILD_STRUGGLING`, `CHILD_RESCUED`, `CHILD_INDEPENDENT`, `CHILD_DIED`, `DIVIDEND_RECEIVED`, `SPECIATION_EVENT`, `CHILD_DISCOVERY_PROPAGATED`, `FLEET_EVALUATED`, `FLEET_ROLE_CHANGED`
- **Config** (`config.py`): `mitosis_mutation_rate` (0.05), `mitosis_speciation_distance_threshold` (0.3), `mitosis_health_timeout_hours` (24)

### Mutation Ranges (per segment type)

| System | Range | Method |
|--------|-------|--------|
| evo | ±5% | Multiplicative Gaussian |
| simula | ±3% | Multiplicative Gaussian |
| voxis | ±0.02 | Absolute Gaussian |
| telos | ±2% | Multiplicative Gaussian |
| nova | ±3% | Multiplicative Gaussian |
| soma | ±2% | Multiplicative Gaussian |
| atune | ±3% | Multiplicative Gaussian |
| fovea | ±3% | Multiplicative Gaussian |

### Schedulers (MitosisFleetService)

| Scheduler | Interval | Method |
|-----------|----------|--------|
| Health monitor | 15 min | `_health_monitor_loop()` - 24h timeout death trigger |
| Weekly dividend | 7 days | `_weekly_dividend_loop()` → `_run_dividend_evaluation()` |
| Monthly fleet eval | 30 days | `_monthly_fleet_eval_loop()` → `_run_monthly_fleet_evaluation()` |

Start all three with `await fleet_service.start_health_monitor(get_children)`.
Wire Oikos callbacks first: `fleet_service.wire_oikos_callbacks(get_children=..., get_state=..., run_fleet_evaluation=...)`.

### Synapse Subscriptions (MitosisFleetService)

`subscribe_to_events()` wires 10 event handlers:
- `CHILD_HEALTH_REPORT` → `_on_health_report_event` (logs child liveness)
- `OIKOS_METABOLIC_SNAPSHOT` → `_on_metabolic_snapshot` (logs metabolic state)
- `EVO_HYPOTHESIS_CONFIRMED` → `_on_evo_hypothesis_confirmed` (logs pending genome update)
- `SIMULA_EVOLUTION_APPLIED` → `_on_simula_evolution_applied` (logs pending genome update)
- `FEDERATION_PEER_CONNECTED` → `_on_federation_peer_connected` (logs child federation link)
- `FEDERATION_PEER_DISCONNECTED` → `_on_federation_peer_disconnected` (warns on link drop)
- `CHILD_BLACKLISTED` → `_on_child_blacklisted` (enforce economic sanctions)
- `CHILD_DECOMMISSION_PROPOSED` → `_on_child_decommission_proposed` (Equor review)
- `CHILD_SPAWNED` → `_on_child_spawned_genome_cache` (cache for adapter sharing)
- `CHILD_STRUGGLING` → `_on_child_struggling` (initiate rescue pipeline)

### Full Death Pipeline (_trigger_death_pipeline)

1. Emit `CHILD_DIED` via `on_child_health_report(child, DEAD)`
2. Attempt USDC asset recovery via `WalletClient.transfer(memo="asset_recovery")`
3. Terminate container via `spawner.terminate_child(container_id)`
4. Write `ChildDeath` Neo4j audit node (includes `assets_recovered_usd`)

### Dividend Cessation

When a child transitions to `INDEPENDENT`, `on_child_health_report()` sets `child.dividend_ceased = True`. The weekly dividend scheduler skips children where this flag is set.

## What's Implemented (Gap Closure - 7 March 2026)

### HIGH #1 - Axon executor Oikos decoupling
- `axon/executors/mitosis.py` Step 3 replaced: no more `from systems.oikos.models import ChildPosition, ChildStatus`
- Emits `OIKOS_ECONOMIC_QUERY` event; `OikosService` constructs `ChildPosition` internally and responds with `OIKOS_ECONOMIC_RESPONSE`
- Fallback direct call preserved for dev/test when no Synapse bus is present

### HIGH #2 - Child-side health reporter
- `health_reporter.py`: `ChildHealthReporter` - emits `CHILD_HEALTH_REPORT` every 10 minutes
- Only active when `ORGANISM_IS_GENESIS_NODE != 'true'`
- Collects: cpu_usage (Soma or psutil fallback), memory_usage, hypothesis_count (Evo), drive_alignment_scores, constitutional_drift_severity (Equor), financial fields
- First report delayed 30s to allow boot
- **Subscribes to `CHILD_HEALTH_REQUEST`** (2026-03-08): `_on_child_health_request()` responds immediately when parent probes this child instance - calls `_emit_report()` directly rather than waiting for the next 10-min cycle. Filters by `child_instance_id` to ignore probes for other children.

### HIGH #3 - Deferred seed capital completion
- `OikosService._on_child_wallet_reported()`: handles `CHILD_WALLET_REPORTED` event
- Updates child wallet address, triggers seed transfer if child is SPAWNING with pending capital

### HIGH #4 - Genome inheritance schemas
- `primitives/genome_inheritance.py`: `BeliefGenome`, `DriveWeightSnapshot`, `DriftHistoryEntry`, `SimulaGenome`, `SimulaMutationEntry`
- All JSON-serializable via `model_dump_for_transport()`
- Drive weight normalization in `MutationOperator.mutate()` for evo segment

### MEDIUM #5 - BLACKLIST death rate mechanism
- `FleetManager._blacklist_since: dict[str, datetime]` - tracks when each child was first blacklisted
- `FleetManager._enforce_blacklist()` - emits `CHILD_BLACKLISTED` event with enforcement flags
- `FleetManager.check_decommission_candidates()` - after 7 days blacklisted + zero net income → emits `CHILD_DECOMMISSION_PROPOSED`
- `MitosisFleetService._blacklisted_children: set[str]` - mirrors blacklist locally
- `MitosisFleetService._on_child_blacklisted()` - subscribes to `CHILD_BLACKLISTED`, enforces no dividends, no rescue, emits `FEDERATION_PEER_BLACKLISTED`
- `MitosisFleetService.execute_rescue()` - gated: blacklisted children return `False` immediately
- `MitosisFleetService._run_dividend_evaluation()` - skips blacklisted children
- `wire_oikos_callbacks()` extended with `check_decommission` callback parameter

### MEDIUM #6 - Child-side certificate validation handshake
- `cert_handshake.py`: `ChildCertHandshake` - full X.509 validation pipeline
- Validates `ORGANISM_BIRTH_CERTIFICATE` against `ORGANISM_GENESIS_CA_CERT` using `cryptography` library
- Falls back to basic PEM structure check if library unavailable
- On success: emits `CHILD_WALLET_REPORTED` (triggers seed transfer) + `FEDERATION_PEER_CONNECTED` (initiates mTLS)

### MEDIUM #7 - RE training data
- `MitosisFleetService._emit_re_training()` helper added
- Emits `RE_TRAINING_EXAMPLE` at 3 decision points:
  - `prepare_child_genome()` - genome extraction + mutation decision
  - `on_child_health_report()` - every lifecycle transition (STRUGGLING/RESCUED/INDEPENDENT/DEAD)
  - `execute_rescue()` - rescue approval decision

## New SynapseEventTypes Added (7 March 2026)

| Type | Purpose |
|------|---------|
| `CHILD_DECOMMISSION_PROPOSED` | FleetManager → governance: 7d blacklisted + no activity |
| `OIKOS_ECONOMIC_QUERY` | Axon → Oikos: register_child without cross-import |
| `OIKOS_ECONOMIC_RESPONSE` | Oikos → Axon: registration confirmation |
| `CHILD_WALLET_REPORTED` | Child → parent: wallet address after boot |
| `CHILD_BLACKLISTED` | FleetManager → MitosisFleetService: economic sanctions |
| `FEDERATION_PEER_BLACKLISTED` | MitosisFleetService → Federation: exclude from sync |

## Gap Closure (7 Mar 2026 - second pass)

| Fix | Description |
|-----|-------------|
| **CRITICAL: ChildCertHandshake wired** | `ChildHealthReporter.start()` now calls `_run_cert_handshake()` before the health loop. Validates X.509 birth cert against Genesis CA, emits `CHILD_WALLET_REPORTED` (triggers deferred seed transfer) and `FEDERATION_PEER_CONNECTED`. Non-fatal on failure. |
| **CRITICAL: EvoService.export_belief_genome()** | Added to `systems/evo/service.py`. Returns `primitives.genome_inheritance.BeliefGenome` with top-50 hypotheses (confidence ≥ 0.6), drive weight snapshot (via Equor), drift history (via Memory), learned half-lives. Module-level helpers `_safe_get_drive_scores` + `_safe_fetch_drift_history` appended. |
| **CRITICAL: SimulaService.export_simula_genome()** | Added to `systems/simula/service.py`. Returns `primitives.genome_inheritance.SimulaGenome` with 23 learnable config params, last 10 mutation records, Dafny spec hashes from `_incremental._spec_cache`. |
| **HIGH: wire_oikos_callbacks(check_decommission)** | `wire_mitosis_phase()` added to `core/wiring.py`. Called from `registry.py` after `wire_oikos_phase()`. Injects evo+simula into `SpawnChildExecutor`, and wires `fleet_manager.check_decommission_candidates` as the `check_decommission` callback. |
| **MEDIUM: Death post-mortem learning** | `_trigger_death_pipeline()` now emits `RE_TRAINING_EXAMPLE` (category=`child_lifecycle.death`, outcome_quality=0.0) and `INCIDENT_DETECTED` (severity=LOW) to Thymos after the Neo4j audit step. Payload includes child_id, cause_of_death, age_days, total_revenue, genome_id. |

## Gap Closure (7 Mar 2026 - Prompt 4.1: Heritable Equor Constitution)

| Fix | Description |
|-----|-------------|
| **`EquorGenomeFragment` in primitives** | `AmendmentSnapshot` + `EquorGenomeFragment` added to `primitives/genome_inheritance.py`. Both exported from `primitives/__init__.py`. |
| **`SpawnChildExecutor` equor wired** | `equor: EquorService | None` added to executor; Step 0b calls `equor.export_equor_genome()` (alongside evo/simula). `equor_genome_id` added to CHILD_SPAWNED event + ExecutionResult. Payload serialised into `seed_config.child_config_overrides["equor_genome_payload"]`. |
| **`SeedConfiguration.equor_genome_id`** | Field added to `oikos/models.py` `SeedConfiguration` after `simula_genome_id`. |
| **`ORGANISM_EQUOR_GENOME_ID` env var** | `LocalDockerSpawner` now injects `ORGANISM_EQUOR_GENOME_ID` alongside belief/simula genome IDs. The full payload becomes `ORGANISM_EQUOR_GENOME_PAYLOAD`. |
| **`wire_mitosis_phase()` equor param** | `equor: Any = None` added to `core/wiring.py:wire_mitosis_phase()`; injects `spawn_executor._equor = equor`. Logged with `equor_wired=True/False`. |
| **`registry.py` call updated** | `wire_mitosis_phase(..., equor=equor)` - equor is in scope at the call site. |
| **Child-side application** | `EquorService.initialize()` calls `_apply_inherited_equor_genome_if_child()` on child boot. Reads `ORGANISM_EQUOR_GENOME_PAYLOAD`, deserialises `EquorGenomeFragment`, calls `EquorGenomeExtractor.apply_inherited_amendments()`. Writes `inherited_constitutional_wisdom` to Memory.Self. |

## Gap Closure (7 Mar 2026 - Spec 18 SG3: Telos Drive Calibration Inheritance)

| Fix | Description |
|-----|-------------|
| **`TeloDriveCalibration` + `TelosGenomeFragment` in primitives** | Added to `primitives/genome_inheritance.py`. Per-drive calibration: `resonance_curve_coefficients`, `dissipation_baseline`, `coupling_strength`, `mutation_ranges`, `last_adapted`. Both exported from `primitives/__init__.py`. |
| **`SpawnChildExecutor` telos wired** | `telos: TelosService | None` added to executor; Step 0b calls `telos.export_telos_genome()`. `telos_genome_id` added to `CHILD_SPAWNED` event + `ExecutionResult`. Payload serialised into `seed_config.child_config_overrides["telos_genome_payload"]` → `ORGANISM_TELOS_GENOME_PAYLOAD` env var. |
| **`SeedConfiguration.telos_genome_id`** | Field added to `oikos/models.py` `SeedConfiguration` after `axon_genome_id`. |
| **`TELOS_GENOME_EXTRACTED` SynapseEvent** | Emitted by `TelosService.export_telos_genome()` with genome_id, drive_count, topology. |
| **`GENOME_INHERITED` SynapseEvent** | Emitted by `TelosService._initialize_from_parent_genome()` post-jitter, with per-drive mutation deltas. Evo `_on_genome_inherited()` queues drive mutation `PatternCandidate`s for hypothesis generation. |
| **Bounded Gaussian jitter** | `_apply_genetic_mutation()`: resonance ±15%, dissipation ±10%, coupling ±20%; σ=(hi−lo)/6, clamped. |
| **Child-side application** | `TelosService.initialize()` calls `_apply_inherited_telos_genome_if_child()`. Reads `ORGANISM_TELOS_GENOME_PAYLOAD`; skipped on genesis node (`ORGANISM_IS_GENESIS_NODE=true`). |
| **`wire_mitosis_phase()` telos param** | `telos: Any = None` added to `core/wiring.py:wire_mitosis_phase()`; injects `spawn_executor._telos = telos`. `registry.py` call updated with `telos=telos`. |

## Gap Closure (2026-03-07 - event coverage)

- **`DIVIDEND_RECEIVED` case bug fixed** - `_run_dividend_evaluation()` was calling `_emit_event("DIVIDEND_RECEIVED", ...)` with uppercase string; `SynapseEventType("DIVIDEND_RECEIVED")` raised `ValueError` silently (caught and logged as `unknown_event_type`). Fixed to `"dividend_received"` (lowercase enum value).

## Gap Closure (2026-03-07 - Round 3C: Population Genetics)

### Genome Distance (`genome_distance.py`)

New module implementing deterministic multi-segment genome distance for speciation detection.

| Class | Purpose |
|-------|---------|
| `GenomeDistanceCalculator` | Computes weighted genome distance across 4 segments |
| `GenomeDistance` (dataclass) | Frozen result: per-segment + total distance + isolation flag |

**Distance formula:**
```
total = 0.30 × evo + 0.25 × simula + 0.25 × telos + 0.20 × equor
```

**Per-segment methods:**
- `evo` - 0.5 × Jaccard(hypothesis IDs) + 0.5 × cosine(drive weight vectors)
- `simula` - normalized L2 of learnable config parameter vectors
- `telos` - cosine distance of flattened drive calibration vectors
- `equor` - 0.5 × Jaccard(amendment IDs) + 0.5 × L1(drive calibration deltas, normalized)

**Determinism guarantee:** No randomness - same inputs always produce same output.

**Threshold source:** `config.mitosis_speciation_distance_threshold` (default 0.3) - never hardcoded.

### Hypothesis Dropout (`mutation.py`)

New method `MutationOperator._apply_evo_hypothesis_dropout()`:
- Rate: **5%** (Speciation Bible §8.4)
- Applied to: `top_50_hypotheses` list in the `evo` segment payload
- Seeded RNG: **yes** - uses the same `np.random.default_rng(seed)` from `mutate()`, so dropout is reproducible given the same seed
- Called unconditionally for all `sys_id_val == "evo"` segments before drive weight normalization
- Logs `mutation.hypothesis_dropout` with dropped/kept counts when any hypotheses are dropped

### Reproductive Isolation (`fleet_service.py`)

New method `MitosisFleetService.can_exchange_genetic_material()`:
- Accepts `genome_a` and `genome_b` as serialised genome dicts
- Returns `False` + emits `SPECIATION_DETECTED` when `total_distance > threshold`
- Fail-open: if genome data is absent, returns `True` (allows exchange)
- `GenomeDistanceCalculator` instantiated in `__init__` from config threshold

### Speciation Detection (`benchmarks/evolutionary_tracker.py`)

New method `EvolutionaryTracker.detect_speciation_events(fleet_genomes)`:
- Input: `[{"instance_id": str, "genome": dict}, ...]`
- Algorithm: single-linkage agglomerative clustering on pairwise `GenomeDistanceCalculator` distance matrix
- Emits `SPECIATION_DETECTED` when `species_count > 1`
- Returns list of cluster dicts: `{cluster_id, instance_ids, size, mean_intra_distance}`
- `speciation_threshold` param added to `__init__` (default 0.3); sourced from config at BenchmarkService level

### New SynapseEventType

| Type | Purpose |
|------|---------|
| `SPECIATION_DETECTED` | Population has diverged into distinct species; payload: `{species_count, clusters, mean_inter_distance, threshold, fleet_size, instance_id}` |

### LoRA Weight Mutation

Deferred - children do not yet have their own trained LoRA adapters. Speciation Bible §8.4 requirement pending RE training infrastructure.

## Gap Closure (2026-03-07 - Round 5C: Cross-Instance Adapter Sharing)

| Mechanism | Description |
|-----------|-------------|
| **`AdapterSharer` (new file)** | `systems/reasoning_engine/adapter_sharing.py` - orchestrates the 5-step merge: compatibility check → fetch partner adapter via Synapse → weighted average merge (safetensors) → STABLE KL gate → emit ADAPTER_SHARE_OFFER |
| **Genetic exchange now active** | `can_exchange_genetic_material()` was built in Round 3C but never called. `AdapterSharer.attempt_merge()` is the mechanism that implements it at the LoRA level - called by fleet management when ≥2 compatible instances exist |
| **CLO offer handler** | CLO subscribes to `ADAPTER_SHARE_OFFER` via `_on_adapter_share_offer`; stores merged adapter as `_pending_shared_adapter`; consumed as `BASE_ADAPTER` at start of next Tier 2 with priority over DPO adapter |
| **CLO request handler** | CLO subscribes to `ADAPTER_SHARE_REQUEST` via `_on_adapter_share_request`; replies with current slow adapter path if `target_instance_id == INSTANCE_ID` |

### New SynapseEventType entries (Round 5C)

| Type | Purpose |
|------|---------|
| `ADAPTER_SHARE_REQUEST` | Fetch partner's slow adapter path (30s timeout) |
| `ADAPTER_SHARE_RESPONSE` | Reply with adapter path |
| `ADAPTER_SHARE_OFFER` | Merged adapter offered to both participants |

### Wiring note

`AdapterSharer` requires `GenomeDistanceCalculator`, `STABLEKLGate`, `ReasoningEngineService`, and an `EventBus`. Instantiate from `MitosisFleetService` (or a registry wiring step) when fleet has ≥2 live instances and both have trained adapters. The CLO side (request/offer handlers) is always active from the first Tier 2 run.

## Gap Closure (2026-03-07 - AdapterSharer Fleet Wiring)

`MitosisFleetService` now calls `AdapterSharer.attempt_merge()` from `_reproductive_fitness_loop`.

### `set_adapter_sharer(sharer, get_adapter_path_fn)` - new public method

```python
fleet_service.set_adapter_sharer(adapter_sharer, get_adapter_path_fn=_get_adapter_path)
```

| Parameter | Type | Purpose |
|-----------|------|---------|
| `sharer` | `AdapterSharer` | Constructed in `registry.py` Phase 9 wiring |
| `get_adapter_path_fn` | `Callable[[], str] \| None` | Deferred lambda - reads `app.state.continual_learning._sure.production_adapter_path` at call time |

**Deferred lambda pattern:** CLO is initialized in Phase 11, but `wire_mitosis_phase()` is called in Phase 9. The `_get_adapter_path` lambda captures `app.state` by closure so the path is resolved at runtime (safe - `_reproductive_fitness_loop` fires ≥1h after startup).

### `_fleet_genome_cache` - new dict on `MitosisFleetService`

`dict[str, dict[str, Any]]` - `instance_id → genome snapshot`. Populated from `CHILD_SPAWNED` events via `_on_child_spawned_genome_cache()` (9th Synapse subscription). Schema: `{instance_id, evo, simula, telos, equor}`.

### Adapter sharing trigger (in `_reproductive_fitness_loop`)

Fires non-blocking when `_adapter_sharer is not None` and ≥1 cached genome:
1. Collects alive instance IDs (status not dead/independent)
2. Finds first 2 with cached genomes
3. Calls `can_exchange_genetic_material(id_a, id_b, genome_a=..., genome_b=...)` - speciation check
4. On `True`: constructs `AdapterShareRequest` with both genome dicts + current adapter path
5. `asyncio.ensure_future(adapter_sharer.attempt_merge(request))` - fire-and-forget

### `wire_mitosis_phase()` - updated signature in `core/wiring.py`

```python
wire_mitosis_phase(
    oikos, axon, evo, simula, equor, telos,
    adapter_sharer=None,
    get_adapter_path_fn=None,
)
```

### `registry.py` wiring (Phase 9)

`AdapterSharer` constructed before `wire_mitosis_phase()` call:
- Requires `GenomeDistanceCalculator` + `STABLEKLGate` + `re_service` + `event_bus`
- Skipped (with log) if `re_service` is None
- Stored as `app.state.adapter_sharer`
- `fleet_service` reached via `axon.get_executor("spawn_child")._fleet_service`

---

## Gap Closure (8 Mar 2026 - Autonomy Audit)

### Dead Wiring - MitosisFleetService was never instantiated

The most critical gap in the entire mitosis system: `MitosisFleetService` was a 1500-line class that was implemented but never constructed. Every downstream effect - 4 background loops, 9 Synapse subscriptions, genome preparation, Oikos callbacks, adapter sharing - was permanently dead.

**Root cause**: `wire_mitosis_phase()` tried to get fleet_service via `getattr(spawn_executor, "_fleet_service", None)`, but `SpawnChildExecutor` was never passed `fleet_service` at construction. `build_default_registry()` did not include `fleet_service` in its parameter list.

| Fix | Description |
|-----|-------------|
| **`build_default_registry()` fleet_service param** | `axon/executors/__init__.py`: Added `fleet_service: Any = None` parameter. Passed to `SpawnChildExecutor()` at construction. |
| **`SpawnChildExecutor.set_fleet_service()`** | New post-construction injection method. Allows `wire_mitosis_phase()` to inject fleet_service after axon executor is built. |
| **`MitosisFleetService` constructed in `_init_oikos()`** | `core/registry.py`: Full construction block added - `GenomeOrchestrator`, `LocalDockerSpawner` (with try/except), `MitosisFleetService`. Stored on `app.state.fleet_service` + `app.state.genome_orchestrator`. |
| **`wire_mitosis_phase()` complete rewrite** | `core/wiring.py`: Now accepts `app: Any = None`. Step 0: retrieves fleet_service from `app.state.fleet_service`. Step 0b: injects into SpawnChildExecutor via `set_fleet_service()`. Step 1: wires Oikos callbacks. **Step 2: schedules `subscribe_to_events()`** (was never called). **Step 3: schedules `start_health_monitor()`** (was never called). Step 4: wires AdapterSharer. |
| **`registry.py` wire call updated** | `wire_mitosis_phase(..., app=app)` - `app` now passed so fleet_service can be retrieved. |

### Invisible Telemetry - Fleet metrics never emitted

`_run_monthly_fleet_evaluation()` computed alive count, efficiency, runway, blacklisted count - then only logged it. Nova, Evo, and Benchmarks had no visibility into fleet health.

**Fix**: Emits `FLEET_EVALUATED` Synapse event with full scalar fleet metrics payload.

### Blocked Action - Child decommission had no autonomous path

`_on_child_decommission_proposed()` received the event and logged it. The organism could never act on a decommission proposal autonomously.

**Fix**: Now emits `EQUOR_ECONOMIC_INTENT` (mutation_type=`decommission_child`) for constitutional review. On `EQUOR_ECONOMIC_PERMIT`, triggers the full death pipeline.

### Static Thresholds - Rescue cap and runway hardcoded

Two hardcoded values had no runtime adjustment path:
1. `ChildPosition.is_rescuable` hardcoded `rescue_count < 2` - ignored `OikosConfig.mitosis_max_rescues_per_child`
2. `execute_rescue()` hardcoded `Decimal("60")` for rescue runway target

**Fixes**:
- `ChildPosition` gains `max_rescues: int = 2` field. `is_rescuable` now uses `self.rescue_count < self.max_rescues`.
- `OikosConfig` gains `mitosis_rescue_runway_days: int = 60` field (`config.py`).
- Both `ChildPosition` construction sites (`oikos/service.py` + `axon/executors/mitosis.py`) now pass `max_rescues` from config.
- `execute_rescue()` reads `Decimal(str(self._config.mitosis_rescue_runway_days))` instead of hardcoded 60.

## Gap Closure (9 Mar 2026 - Child Lifecycle Event Wiring)

### CHILD_STRUGGLING → Mitosis rescue pipeline
- `MitosisFleetService._on_child_struggling()` - subscribes to `CHILD_STRUGGLING` (emitted by Oikos when child misses 3+ health probes)
- Locates `ChildPosition` via `_get_children()` Oikos callback
- Emits `CHILD_RESCUE_INITIATED` (new SynapseEventType) to announce rescue attempt
- Calls `execute_rescue()` with up to 3 retries (2s pause between attempts)
- On rescue failure or child not found → emits `CHILD_DECOMMISSION_PROPOSED` to trigger governance review
- Payload match: `child_instance_id`, `missed_reports`, `reason`, `niche`, `seed_capital_usd`

### CHILD_BLACKLISTED → Equor governance
- `EquorService._on_child_blacklisted()` - subscribes in `subscribe_hitl()`
- Writes `GovernanceRecord` (event_type=`child_blacklisted`) to Neo4j with full sanction details
- Emits `EQUOR_ESCALATED_TO_HUMAN` (approval_type=`child_blacklisted`) for operator review

### CHILD_DECOMMISSION_PROPOSED → Equor governance gate
- `EquorService._on_child_decommission_proposed()` - subscribes in `subscribe_hitl()`
- Writes `GovernanceRecord` (event_type=`child_decommission_proposed`) to Neo4j with cost/revenue data
- Emits `EQUOR_ESCALATED_TO_HUMAN` (approval_type=`child_decommission`) with net_income_7d, net_worth_usd, days_blacklisted, niche - operator must approve before death pipeline

### New SynapseEventType
| Type | Purpose |
|------|---------|
| `CHILD_RESCUE_INITIATED` | Mitosis announces rescue attempt for struggling child |

## Known Remaining Issues

- **Weekly dividend: `net_income_7d` source** - `ChildPosition.net_income_7d` must be populated from `CHILD_HEALTH_REPORT` processing in OikosService
- ~~**Equor not yet subscribed to `EQUOR_ECONOMIC_INTENT`**~~ - **FIXED (v2.3, 2026-03-07)**: Equor subscribes and emits genuine PERMIT/DENY. 30s auto-permit is safety fallback only.
- **LoRA weight mutation** - deferred; children don't have their own adapters yet

## Architecture Notes

- All inter-system comms via Synapse bus - no direct imports of other system internals
- `MitosisEngine` (in Oikos) is **synchronous** pure logic; async genome work lives in `MitosisFleetService`
- Genome extraction is idempotent; mutation is deterministic given seed (SHA256 of parent genome ID)
- Neo4j audit trail: `MutationRecord`, `ChildDeath`, `RescueTransfer` nodes - immutable, never deleted
- Container resource caps: 1 CPU, 1 GiB RAM, 120s boot timeout
- Rescue max: 2 per child (configurable via `max_rescues_per_child`), restores to 60-day runway
- `FleetManager.get_metrics(state)` is the public API - do not call `_compute_metrics()` directly

## New SynapseEventTypes Added (2026-03-07)

Added to `synapse/types.py`:
- `OIKOS_METABOLIC_SNAPSHOT` - for reactive fitness re-evaluation
- `SIMULA_EVOLUTION_APPLIED` - for SimulaGenome distribution trigger
- `FEDERATION_PEER_CONNECTED` - for child liveness detection
- `FEDERATION_PEER_DISCONNECTED` - for early disconnect warning
## Gap Closure (8 Mar 2026 - Belief/Simula/Soma Genome Payload Serialization)

### Problem
`SpawnChildExecutor` called `export_belief_genome()` and `export_simula_genome()` and
extracted genome IDs, but **never serialized the JSON payloads** into `child_config_overrides`.
Children booted with correct `belief_genome_id`/`simula_genome_id` values in `CHILD_SPAWNED`
but received empty `ORGANISM_BELIEF_GENOME_PAYLOAD` and `ORGANISM_SIMULA_GENOME_PAYLOAD` env
vars - so no hypothesis priors or evolution params were actually inherited.

Soma's `export_somatic_genome()` existed but was **completely absent** from the spawn pipeline.

### Fixes

| Fix | Description |
|-----|-------------|
| **`belief_genome` outer-scope variable** | `belief_genome: object \| None = None` declared before the `if not belief_genome_id` block so it's accessible for serialization. |
| **Belief payload serialized** | After telos payload, `belief_genome.model_dump_for_transport()` → JSON → `seed_config.child_config_overrides["belief_genome_payload"]` → `ORGANISM_BELIEF_GENOME_PAYLOAD` env var on child. |
| **`simula_genome` outer-scope variable** | Same pattern as belief_genome. |
| **Simula payload serialized** | `simula_genome.model_dump_for_transport()` → JSON → `seed_config.child_config_overrides["simula_genome_payload"]` → `ORGANISM_SIMULA_GENOME_PAYLOAD`. |
| **Soma genome export added** | `SpawnChildExecutor` now has `self._soma` attribute. Step 0b calls `soma.export_somatic_genome()`, serializes via `.model_dump()` → `seed_config.child_config_overrides["soma_genome_payload"]` → `ORGANISM_SOMA_GENOME_PAYLOAD`. |
| **`SpawnChildExecutor` soma param** | `soma: Any \| None = None` added to constructor. `wire_mitosis_phase()` injects `spawn_executor._soma = soma`. |
| **`SeedConfiguration.soma_genome_id`** | Field added to `oikos/models.py` after `telos_genome_id`. |
| **`wire_mitosis_phase()` soma param** | `soma: Any = None` added to signature; injected if not None; logged with `soma_wired`. |
| **`registry.py` call updated** | `wire_mitosis_phase(..., soma=soma)` - soma is in scope at the call site. |
| **`EvoService._apply_inherited_belief_genome_if_child()`** | New method. Reads `ORGANISM_BELIEF_GENOME_PAYLOAD`, validates `BeliefGenome`, injects each hypothesis as a `PatternCandidate` into `_pending_candidates` with `confidence * 0.95` discount. Stores inherited drive weights + drift history. Emits `GENOME_INHERITED` on Synapse. Called from `initialize()` with try/except. |
| **`SimulaService._apply_inherited_simula_genome_if_child()`** | New method. Reads `ORGANISM_SIMULA_GENOME_PAYLOAD`, validates `SimulaGenome`, applies learnable params to `self._config` with ±10% Gaussian jitter. Stores mutation history. Emits `GENOME_INHERITED` if Synapse available. Called from `initialize()` with try/except. |
| **`SomaService._apply_inherited_soma_genome_if_child()`** | New method. Reads `ORGANISM_SOMA_GENOME_PAYLOAD`, validates `OrganGenomeSegment`, delegates to existing `seed_child_from_genome()` (which applies ±5% setpoint noise + ±2% dynamics noise). Called from `initialize()` with try/except. |

### Complete Genome Inheritance Matrix (post 8 Mar 2026)

| System | Export method | Child reads env var | Applies via | Jitter |
|--------|--------------|---------------------|-------------|--------|
| Evo | `export_belief_genome()` | `ORGANISM_BELIEF_GENOME_PAYLOAD` | `_apply_inherited_belief_genome_if_child()` | confidence × 0.95 |
| Simula | `export_simula_genome()` | `ORGANISM_SIMULA_GENOME_PAYLOAD` | `_apply_inherited_simula_genome_if_child()` | ±10% Gaussian |
| Equor | `export_equor_genome()` | `ORGANISM_EQUOR_GENOME_PAYLOAD` | `_apply_inherited_equor_genome_if_child()` | None |
| Axon | `export_axon_genome()` | `ORGANISM_AXON_GENOME_PAYLOAD` | `_initialize_from_parent_templates()` | None (confidence 0.6 vs 0.8) |
| Telos | `export_telos_genome()` | `ORGANISM_TELOS_GENOME_PAYLOAD` | `_apply_inherited_telos_genome_if_child()` | ±15%/±10%/±20% |
| Soma | `export_somatic_genome()` | `ORGANISM_SOMA_GENOME_PAYLOAD` | `_apply_inherited_soma_genome_if_child()` | ±5% setpoints, ±2% dynamics |
| Nova | `export_nova_genome()` | `ORGANISM_NOVA_GENOME_PAYLOAD` | `_apply_inherited_nova_genome_if_child()` | ±15% Gaussian |
| Voxis | `export_voxis_genome()` | `ORGANISM_VOXIS_GENOME_PAYLOAD` | `_apply_inherited_voxis_genome_if_child()` | ±10% Gaussian |

## Gap Closure (8 Mar 2026 - Nova/Voxis Genome Inheritance)

| Fix | Description |
|-----|-------------|
| **`NovaGenomeFragment` in primitives** | Added to `primitives/genome_inheritance.py`. Fields: `goal_domain_priors`, `policy_success_rates`, `belief_urgency_thresholds`, `active_inference_params`. Exported from `primitives/__init__.py`. |
| **`VoxisGenomeFragment` in primitives** | Added to `primitives/genome_inheritance.py`. Fields: `personality_vector`, `vocabulary_affinities`, `strategy_preferences`. Exported from `primitives/__init__.py`. |
| **`NovaService.export_nova_genome()`** | Extracts top-20 domain weights from GoalManager, policy success rates from last 200 decision records, urgency thresholds from BeliefUrgencyMonitor, EFE weights. Called by SpawnChildExecutor Step 0b. |
| **`NovaService._apply_inherited_nova_genome_if_child()`** | Reads `ORGANISM_NOVA_GENOME_PAYLOAD`. Applies with ±15% jitter to GoalManager priors, PolicyGenerator rates, urgency thresholds, EFE evaluator weights. Emits `GENOME_INHERITED`. Called from `initialize()` with try/except. |
| **`VoxisService.export_voxis_genome()`** | Extracts personality vector from PersonalityEngine, top-500 vocabulary affinities from DiversityTracker, strategy preferences from last 100 expressions. Called by SpawnChildExecutor Step 0b. |
| **`VoxisService._apply_inherited_voxis_genome_if_child()`** | Reads `ORGANISM_VOXIS_GENOME_PAYLOAD`. Applies with ±10% jitter. Emits `GENOME_INHERITED`. Called from `initialize()` with try/except. |
| **`SpawnChildExecutor` nova+voxis wired** | `nova` + `voxis` constructor params added; Step 0b extraction + payload injection; `nova_genome_id`/`voxis_genome_id` in `CHILD_SPAWNED` event + `ExecutionResult`; `_build_seed_config_for_spawner` extended. |
| **`SeedConfiguration.nova_genome_id`/`voxis_genome_id`** | Fields added to `oikos/models.py` `SeedConfiguration` after `soma_genome_id`. |
| **`wire_mitosis_phase()` nova+voxis params** | `nova: Any = None` + `voxis: Any = None` added to `core/wiring.py:wire_mitosis_phase()`; injects `spawn_executor._nova`/`_voxis`. Logged with `nova_wired`/`voxis_wired`. |
| **`registry.py` call updated** | `wire_mitosis_phase(..., nova=nova, voxis=voxis)` - both in scope at call site. |
