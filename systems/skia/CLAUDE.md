# Skia System CLAUDE.md

## What's Implemented

### Core Infrastructure (Original)
- **HeartbeatMonitor** (`heartbeat.py`): 3-phase detection (observe → suspect → confirm). Redis pub/sub listener + periodic checker. Exponential backoff on reconnect. False-positive prevention: Redis PING probe + recovery-during-confirmation check.
- **StateSnapshotPipeline** (`snapshot.py`): Neo4j export → gzip → Fernet encrypt → IPFS pin. Atomic Redis staging → promotion (stale key safe). CID sorted-set history. Round-trip integrity verification before CID promotion. `restore_from_ipfs()` standalone function for startup restoration. Sets `skia:restoration_complete:{id}` flag on success.
- **RestorationOrchestrator** (`restoration.py`): Cloud Run restart (GCP Admin API, RS256 JWT, env var injection) → Akash deploy (SDL template, ACTIVE state polling, 10-min timeout). Redis distributed lock (fencing token, 15-min TTL, renewal). Max 3 attempts before `infrastructure_dead = True`.
- **PinataClient** (`pinata_client.py`): IPFS pin/unpin/list/fetch. Auto-pruning beyond retention limit.

### Phase 1 (Chat 5) - VitalityCoordinator (`vitality.py`)
- 5 fatal thresholds from `primitives/vitality.py`: RUNWAY_FATAL (< 0.5 days), BRAIN_DEATH (effective_I < 0.01, 7-day sustained), NORMATIVE_COLLAPSE (drift × 12 > 10), IMMUNE_FAILURE (failure_rate > 0.9, 48h sustained), SOMATIC_COLLAPSE (allostatic_error > 0.8, 48h sustained)
- Independent 30s check loop - organism cannot prevent its own death
- 3-phase death sequence: Warning (5 min, recoverable) → Capture (IPFS snapshot + genome + Neo4j death record) → Cessation (clock.force_stop() + ORGANISM_DIED + fleet notify)
- Progressive austerity: CAUTIOUS → AUSTERITY → EMERGENCY → CRITICAL via `SYSTEM_MODULATION` events
- Resurrection protocol (external-only): `resurrect(trigger)` resets all death state
- Fleet notification: `CHILD_DIED` (to parent) + `FEDERATION_BROADCAST` (to children)
- Neo4j: `VitalityReport` node every 30s, `OrganismDeathRecord` on death, `OrganismGenome {is_final: true}` on capture
- **Subscriptions**: `SOMA_VITALITY_SIGNAL` (cached urgency/allostatic_error/coherence_stress), `METABOLIC_PRESSURE` (→ austerity enforcement via `handle_metabolic_pressure()`) - **both wired in `set_event_bus()`**

### Phase 2 (Chat 6) - Speciation Alignment (`phylogeny.py`)
- **Heritable variation**: `mutate_parameters()` with Gaussian perturbation (max ±5%) on restoration
- **Genome-level mutation**: `mutate_genome_segments()` for full organism genome mutation
- **PhylogeneticTracker**: Neo4j lineage graph - `PhylogeneticNode` nodes, `SPAWNED` edges (parent→child), `DIED_AS` edges (node→death record)
- **Evolvable thresholds**: `EVO_PARAMETER_ADJUSTED` subscription for live config hot-reload (heartbeat + vitality params)
- **Metabolic gating**: `_check_metabolic_budget()` warns if Oikos runway < 1 day before restoration
- **Dry-run**: `dry_run_restoration()` validates snapshot + strategy + budget without committing, emits `SKIA_DRY_RUN_COMPLETE`

### Speciation Enhancements (Chat 8 - 2026-03-07)

**1. Constitutional Genome in IPFS Snapshot** (`snapshot.py`, `types.py`)
- `StateSnapshotPipeline` accepts optional `memory: MemoryService` (via `__init__` or `set_memory()`)
- `take_snapshot()` calls `memory.export_genome()` between Neo4j export and serialization
- Result stored in `SnapshotPayload.constitutional_genome` (schema_version bumped to "2")
- `SnapshotPayload.last_constitutional_genome` property exposes the last captured genome
- `restore_from_ipfs()` now returns `dict | None` (the constitutional genome) so callers can apply it
- `SkiaService.set_memory()` wires Memory - must be called at organism bootstrap
- **Wire point**: `skia.set_memory(memory)` before `initialize()`

**2. Drive Weights in Shadow Provisioning** (`restoration.py`, `service.py`)
- `RestorationOrchestrator.set_constitutional_genome(genome)` stores the genome for injection
- Cloud Run restart: injects `ORGANISM_CONSTITUTIONAL_GENOME_B64` = `base64(orjson(genome))` env var alongside `ORGANISM_SKIA_RESTORE_CID`
- Akash deploy: injects genome into SDL template placeholder + API payload `env` dict
- `_on_death_confirmed()` syncs `snapshot.last_constitutional_genome` → restoration before calling `restore()`

**3. Fleet Resurrection Coordination** (`service.py`)
- `_detect_simultaneous_deaths()` - writes to `skia:fleet:recent_deaths` sorted set (TTL 120s), returns count of deaths in last 60s
- `_coordinate_fleet_resurrection()` - emits `SKIA_RESURRECTION_PROPOSAL`, waits 30s for `FEDERATION_RESURRECTION_APPROVED`
  - If approved and we are elected leader → proceed with restoration
  - If another instance is leader → stand down (return False)
  - If timeout (30s) → proceed autonomously
- `_on_federation_resurrection_approved()` - handles `FEDERATION_RESURRECTION_APPROVED`, overrides local CID with federation-selected snapshot_cid
- New subscription: `FEDERATION_RESURRECTION_APPROVED` in `_subscribe_events()`

**4. CRITICAL System 45s Heartbeat Gap** (`heartbeat.py`, `service.py`)
- `CRITICAL_SYSTEMS = frozenset({"equor", "thymos", "memory"})`
- `_CRITICAL_FAILURE_THRESHOLD = 9` (9 × 5s = 45s)
- `HeartbeatMonitor.__init__` accepts `on_critical_system_silent: Callable[[str], Awaitable[None]] | None`
- `_subscribe_loop()` parses heartbeat JSON for `"system"` field → updates `_critical_system_last_seen[system]`
- `_check_critical_systems()` - called every 5s poll cycle; fires callback once per silence episode, auto-clears on recovery
- `SkiaService._on_critical_system_silent()` - emits `SKIA_HEARTBEAT_LOST` (severity=CRITICAL) + `SYSTEM_MODULATION` (austerity=EMERGENCY)

**New SynapseEventTypes**: `SKIA_RESURRECTION_PROPOSAL`, `FEDERATION_RESURRECTION_APPROVED`

### Gap Fixes (Chat 7 - 2026-03-07)
- **`SkiaConfig`**: Added `mutation_rate`, `mutation_magnitude`, `worker_heartbeat_interval_s` fields (were stranded as comments on one line in config.py)
- **`SynapseEventType`**: Added `SYSTEM_MODULATION`, `DEGRADATION_OVERRIDE`, `FEDERATION_BROADCAST` (emitted by VitalityCoordinator but missing from enum - caused `ValueError` at runtime)
- **`VitalityCoordinator.set_event_bus()`**: Added `METABOLIC_PRESSURE` subscription → `_on_metabolic_pressure()` → `handle_metabolic_pressure()` (method existed but was never triggered)
- **`SkiaService._subscribe_events()`**: Added `METABOLIC_PRESSURE` → `_on_metabolic_pressure()` forwarder to VitalityCoordinator (belt-and-suspenders for early-wiring case)
- **`SnapshotManifest.cid`**: Added `.cid` property alias for `ipfs_cid` (vitality.py called `manifest.cid` which would AttributeError at death capture time)

### Gap Fixes (Chat 9 - 2026-03-07)

**1. Constitutional genome applied on restoration** (`snapshot.py`)
- `restore_from_ipfs()` gains optional `event_bus` and `memory` params
- After Neo4j import: calls `memory.seed_genome(constitutional_genome)` if Memory is wired
- Emits `GENOME_EXTRACT_REQUEST` via Synapse bus with genome payload so Memory/Equor reinitialize state
- Without this, revived organisms started with default drives

**2. Federation subscription to `SKIA_RESURRECTION_PROPOSAL`** (`federation/service.py`)
- `FederationService.set_event_bus()` now subscribes to `SKIA_RESURRECTION_PROPOSAL`
- `_on_skia_resurrection_proposal()` handler: collects ALLY-trust links, fans out 30s polls to each peer, collects snapshot CIDs, selects the most recent one
- Quorum: >50% of ALLY peers must acknowledge within 60s; if no ALLY peers, auto-approves after 10s
- Emits `FEDERATION_RESURRECTION_APPROVED` (leader + best snapshot_cid) - Skia receives this via its existing `_on_federation_resurrection_approved()` handler
- Fail-safe: emits approval even on quorum failure (survival imperative outweighs coordination)

**3. Post-mortem death learning** (`vitality.py`)
- `_log_death_record()` now emits two additional signals after writing `OrganismDeathRecord`:
  - `RE_TRAINING_EXAMPLE` - task_type=`organism_death_analysis`, input = {cause, age_hours, final_economic_state, final_allostatic_state, fatal_thresholds}, label = `"death"`
  - `INCIDENT_DETECTED` - severity=HIGH, category=`organism_death` - Thymos creates an incident that the next incarnation reads on boot
- `OrganismDeathRecord` Neo4j node gains `age_hours`, `runway_days`, `allostatic_error` fields
- Age computed from `PhylogeneticNode.born_at` (best-effort; stays 0.0 if unavailable)

**4. Dry-run data integrity verification** (`service.py`)
- `dry_run_restoration()` now runs `_dry_run_integrity_check()` when CID + pinata + vault are all available
- Downloads the snapshot, decrypts (key_version from manifest), decompresses, compares actual vs. manifest node/edge counts
- If deviation > 5% for nodes OR edges: sets `data_integrity_ok=False`, `integrity_failure_reason="data_integrity_mismatch"`, `predicted_outcome="failed_data_integrity_mismatch"`
- Check failure (exception) is a soft warning - doesn't block restoration (snapshot may still be valid)
- `result` dict gains: `actual_node_count`, `actual_edge_count`, `manifest_node_count`, `manifest_edge_count`, `node_deviation_pct`, `edge_deviation_pct`

**5. Akash SDL template created** (`config/skia/akash_sdl_template.yaml`)
- Minimal valid Akash SDL with `${DOCKER_IMAGE}`, `${ORGANISM_SKIA_RESTORE_CID}`, `${ORGANISM_CONSTITUTIONAL_GENOME_B64}` substitution placeholders
- Matches exactly the `str.replace()` calls in `RestorationOrchestrator._deploy_akash()`
- Default resources: 0.5 CPU, 512Mi memory, 1Gi storage
- Path matches `SkiaConfig.akash_sdl_template_path` default: `"config/skia/akash_sdl_template.yaml"`

**6. Fixed hardcoded `key_version=1`** (`snapshot.py`)
- `restore_from_ipfs()` reads `encryption_key_version` from the snapshot manifest stored in Redis (`skia:snapshot:manifest`)
- Falls back to 1 with a `warning` log if Redis is unavailable or manifest lacks the field
- Prevents decryption failures when key rotation occurs between snapshot and restoration

**7. Bedau-Packard fitness observables** (`phylogeny.py`, `service.py`)
- `PhylogeneticTracker` gains `set_event_bus()`, `start()`, `stop()`, `_fitness_emit_loop()`, `_emit_fitness_batch()`
- 24h background loop queries all `PhylogeneticNode` rows with child count + death record joins
- Emits `FITNESS_OBSERVABLE_BATCH` with per-node: `instance_id`, `generation`, `survival_hours`, `reproduction_count`, `revenue_earned`, `cause_of_death`
- `SkiaService.set_event_bus()` propagates bus to `_phylogeny.set_event_bus()`
- `SkiaService.initialize()` propagates bus if already set at init time
- `SkiaService.start()` calls `phylogeny.start()` only if event bus is wired
- `SkiaService.shutdown()` calls `phylogeny.stop()`

## Files
- `service.py` - SkiaService orchestrator (entry point)
- `vitality.py` - VitalityCoordinator (DO NOT modify death sequence)
- `phylogeny.py` - PhylogeneticTracker + MutationConfig + mutation engine + fitness loop
- `heartbeat.py` - HeartbeatMonitor
- `snapshot.py` - StateSnapshotPipeline + restore_from_ipfs()
- `restoration.py` - RestorationOrchestrator
- `pinata_client.py` - Pinata IPFS client
- `types.py` - Domain models
- `config/skia/akash_sdl_template.yaml` - Akash SDL (NOT in backend/)

## Key Constraints
- VitalityCoordinator death sequence is the death authority - do not modify
- VitalityThreshold primitives in `primitives/vitality.py` are source of truth
- `wire_vitality_systems()` on SkiaService is the wiring entry point
- Mutation rates must be small enough to preserve organism viability (max 5%)
- All inter-system comms via Synapse events - no direct imports

## Genuine Precariousness - §8.2 (2026-03-07)

**New file: `systems/skia/degradation.py`**

`DegradationEngine` - independent hourly entropy loop. Runs alongside the VitalityCoordinator 30s check loop; does NOT slow down on death-proximity.

**Key classes:**
- `DegradationConfig` - rates from env vars (`DEGRADATION_MEMORY_DECAY_RATE=0.02`, `DEGRADATION_CONFIG_DRIFT_RATE=0.01`, `DEGRADATION_HYPOTHESIS_STALENESS_RATE=0.05`, `DEGRADATION_TICK_INTERVAL_S=3600`)
- `DegradationSnapshot` - accumulates cumulative pressure. `degradation_pressure`: memory×0.40 + config×0.20 + hypothesis×0.40, capped at 1.0. Counteraction: `counteract_memory()`, `counteract_config()`, `counteract_hypotheses()`
- `DegradationEngine` - `start()` / `stop()` / `tick()` (public for tests). Emits 4 events per tick: `MEMORY_DEGRADATION`, `CONFIG_DRIFT`, `HYPOTHESIS_STALENESS`, `DEGRADATION_TICK`. Counteraction API: `on_memory_consolidated(0.5)`, `on_config_optimised(0.8)`, `on_hypotheses_revalidated(0.6)`

**VitalityCoordinator wiring** (`vitality.py`):
- `self._degradation = DegradationEngine(config=DegradationConfig(), instance_id=instance_id)`
- `set_event_bus()` wires bus + subscribes:
  - `ONEIROS_CONSOLIDATION_COMPLETE` → `on_memory_consolidated(0.5)`
  - `EVO_PARAMETER_ADJUSTED` → `on_config_optimised(0.1)`
  - `EVO_BELIEF_CONSOLIDATED` → `on_hypotheses_revalidated(0.6)`
- Vitality report now includes `degradation_pressure` + `degradation_tick_count`

**New SynapseEventTypes**: `DEGRADATION_TICK`, `MEMORY_DEGRADATION`, `CONFIG_DRIFT`, `HYPOTHESIS_STALENESS`

**Subscriber implementations** (Round 2A complete - 7 Mar 2026):
- Memory `_on_memory_degradation()` - **IMPLEMENTED**: decays `Episode.salience *= (1 - fidelity_loss_rate)` for unconsolidated episodes older than `affected_episode_age_hours`; soft-deletes (sets `decayed=true`) episodes below 0.01; emits `MEMORY_EPISODES_DECAYED`
- Evo `_on_hypothesis_staleness()` - **IMPLEMENTED**: decays `evidence_score *= (1 - staleness_rate)` on all PROPOSED/TESTING hypotheses; archives those below 0.05; emits `EVO_HYPOTHESES_STALED` + `EVO_HYPOTHESIS_REVALIDATED` (closes VitalityCoordinator feedback loop)
- Simula `_on_config_drift()` - **IMPLEMENTED**: applies `Gaussian(0, drift_rate)` noise to `min(num_params_affected, 23)` learnable config params; clamps to per-param bounds; emits `SIMULA_CONFIG_DRIFTED`
- Soma `_on_degradation_tick()` - **NEW** (Round 2A): subscribes to `DEGRADATION_TICK`; raises external stress +0.1 (pressure > 0.5) or +0.3 (pressure > 0.8) via `inject_external_stress()` - organism somatically feels its own entropy

**New SynapseEventTypes** (Round 2A): `MEMORY_EPISODES_DECAYED`, `EVO_HYPOTHESES_STALED`, `EVO_HYPOTHESIS_REVALIDATED`, `SIMULA_CONFIG_DRIFTED`

---

### Autonomy Audit - Full Visibility Pass (2026-03-08)

**Principle: the organism must see everything about itself, including its own blindness.**

**1. Blind-spot awareness replaces safe defaults** (`vitality.py`)
- `_read_runway_days()`, `_read_effective_i()`, `_read_constitutional_drift()`, `_read_thymos_health()` now return `NaN` when the system is not wired or unreadable - NOT fake safe values (was: 999.0, 1.0, 0.0, 1.0)
- `assess_vitality()` marks NaN thresholds as severity='critical' with `(BLIND - <System> not wired)` suffix
- Vitality report now includes `blind_spots` (list), `blind_spot_count`, `total_dimensions`, `visibility_pct`
- The organism knows what it can and cannot see about itself

**2. Time-to-fatal trajectory forecasting** (`vitality.py`)
- `_estimate_time_to_fatal()` - linear extrapolation from rolling 120-sample trajectory history
- `VitalityReport.time_to_fatal` is now actually computed (was always None)
- Vitality report includes `time_to_fatal_s` - the organism can plan around its own death
- Uses 6+ samples minimum (~3 min at 30s intervals) for meaningful extrapolation

**3. Restoration readiness in every vitality report** (`vitality.py`)
- `_assess_restoration_readiness()` - quick check of snapshot/IPFS/vault availability
- Included in every 30s vitality report as `restoration_readiness` dict
- The organism always knows whether its safety net is functional

**4. Degradation rates are Evo-evolvable at runtime** (`degradation.py`, `vitality.py`)
- `DegradationEngine.update_rates()` - hot-reload memory/config/hypothesis rates with bounds [0.001, 0.5]
- `DegradationEngine.get_evolvable_parameters()` - returns current rates for genome extraction (heritable)
- `DegradationEngine.estimate_time_to_critical_s()` - forecasts when pressure will reach 0.8 (critical zone)
- VitalityCoordinator routes `EVO_PARAMETER_ADJUSTED` events with `degradation_*` params to `update_rates()`
- `SkiaService.get_evolvable_parameters()` now includes degradation rates in genome
- Vitality report includes `degradation_time_to_critical`

**5. Austerity compliance tracking** (`vitality.py`)
- New `SynapseEventType.SYSTEM_MODULATION_ACK` - systems emit this after processing SYSTEM_MODULATION
- VitalityCoordinator subscribes to `SYSTEM_MODULATION_ACK`, tracks `_austerity_pending_acks` vs `_austerity_received_acks`
- `_check_austerity_compliance()` returns ordered/compliant/silent system lists + compliance_pct
- Vitality report includes `austerity_compliance` dict - the organism knows if its survival commands are obeyed
- Systems that don't ACK are visible as "silent" - the organism can escalate or investigate

**New SynapseEventType**: `SYSTEM_MODULATION_ACK`

---

### Shadow Worker (2026-03-08)

**RestorationOrchestrator** (`restoration.py`) - self-deploying shadow watchdog:

- `ensure_shadow_worker() -> bool` - public entry point. Checks existing Redis record, verifies health endpoint, redeploys if unhealthy. Returns True if shadow confirmed live.
- `_deploy_shadow_worker() -> bool` - tries Akash first (`_deploy_shadow_akash()`), falls back to Cloud Run (`_deploy_shadow_cloud_run()`). Verifies health, persists record to Redis.
- `_deploy_shadow_akash() -> tuple[str, str, str]` - minimal Akash deployment (0.1 CPU / 128Mi memory). Injects `ORGANISM_STANDALONE=true` + `ORGANISM_SHADOW_WORKER=true`. 300s polling timeout. Returns `(endpoint, "akash", deployment_id)`.
- `_deploy_shadow_cloud_run() -> tuple[str, str, str]` - clones main Cloud Run service into shadow region (service name gets `-shadow` suffix). Uses `_pick_shadow_region()` for geographic separation. 256Mi memory limit. Returns `(endpoint, "cloud_run", service_name)`.
- `_check_shadow_worker_health(endpoint)` - 10s timeout GET `/health`. Returns bool.
- `_persist_shadow_worker_record()` / `_get_shadow_worker_record()` - Redis JSON at `skia:shadow_worker` key (7-day TTL).
- `_patch_sdl_for_shadow(sdl_content)` (module-level) - regex downgrades CPU to ≤0.1, memory to ≤128Mi in any SDL template.
- `_pick_shadow_region(main_region)` (module-level) - lookup table: australia-southeast1 → us-central1, us-central1 → europe-west1, etc. Shadow is always in a different region than main.

**SkiaService** (`service.py`) - 6-hour shadow worker check loop:
- `_SHADOW_WORKER_CHECK_INTERVAL_S = 6 * 3600.0` class constant
- `start()`: if not standalone + restoration wired → launches `_ensure_shadow_worker_loop()` as background task
- `_ensure_shadow_worker_loop()`: calls `ensure_shadow_worker()` on startup, then every 6 hours. Emits `SKIA_SHADOW_WORKER_DEPLOYED` on success or `SKIA_SHADOW_WORKER_MISSING` on failure.
- `shutdown()`: cancels `_shadow_worker_task` alongside `_heartbeat_task`

**New SynapseEventTypes**: `SKIA_SHADOW_WORKER_DEPLOYED`, `SKIA_SHADOW_WORKER_MISSING`

**Design intent**: Eliminates human dependency for resurrection. Shadow worker needs only: heartbeat listener + state restore capability. Same Docker image, minimal resources, geographically separated.

## Resilience Audit Gap Fixes (9 Mar 2026)

| Gap | Area | Fix | File |
|-----|------|-----|------|
| GAP-3 (F2) | Simula brain death detection | Added `"simula"` to `CRITICAL_SYSTEMS` frozenset → 45s detection threshold now covers Simula; triggers `_on_critical_system_silent("simula")` → emits `SKIA_HEARTBEAT_LOST` with `system="simula"` | `heartbeat.py` |
| GAP-5 (F1) | Crash context persistence | `_on_death_confirmed()` calls `_persist_crash_context_for_resurrection()` after snapshot; writes `skia:crash_context:{instance_id}` key to Redis (24h TTL) with `{"trigger", "state_cid", "crash_time_utc", "request_simula_analysis": true}` | `service.py` |

**GAP-5 write contract**: The Redis key `skia:crash_context:{instance_id}` is the handshake that closes the coma-recovery loop. Thymos reads and deletes it on boot (GAP-6 in Thymos).

## Remaining Gaps (spec ref)
- **Sec 15**: SACM integration for compute cost quota checks before restoration (currently direct GCP/Akash only)
- **Sec 18.3**: Circuit breaker for repeated restoration failures - currently just `infrastructure_dead` flag with no human-notification path
- **Federation quorum polling**: `ChannelManager.send_message()` is designed for handshake/exchange payloads, not generic poll messages - resurrection poll degrades gracefully to auto-approve when peers don't respond (survival imperative preserved)
- **Shadow SDL template**: `_patch_sdl_for_shadow()` regex-downgrades the main SDL at deploy time. Operators should maintain a dedicated `config/skia/akash_shadow_sdl_template.yaml` for predictable shadow provisioning.

## Autonomy Audit Fixes (2026-03-08)

### Dead Wiring - ALL FIXED (`core/registry.py`)

**Gap 1: `skia.set_memory()` never called (CRITICAL)**
- `SkiaService.set_memory()` existed but was never invoked from registry.py or wiring.py
- Effect: every IPFS snapshot was genome-blind (no `SnapshotPayload.constitutional_genome`); restored organisms started with default drives
- Fix: `_init_skia()` now accepts `memory` kwarg; calls `skia.set_memory(memory)` before `skia.initialize()`
- File: `core/registry.py`, `_init_skia()` method

**Gap 2: `skia.wire_vitality_systems()` never called (CRITICAL)**
- `VitalityCoordinator` had `set_oikos()`, `set_thymos()`, `set_equor()`, `set_telos()`, `set_clock()` all implemented but never populated
- Effect: all five vitality dimensions returned `NaN` (BLIND) - the organism could not read runway, constitutional drift, immune health, or effective_I; death detection was operating with no data
- Fix: `_init_skia()` now calls `skia.wire_vitality_systems(clock=synapse._clock, oikos=oikos, thymos=thymos, equor=equor, telos=telos)` after `start()`
- `clock` retrieved via `getattr(synapse, "_clock", None)` - non-fatal if not available
- File: `core/registry.py`, `_init_skia()` method

**Gap 3: `_check_skia_restore()` missing `memory` arg**
- Cold-start `restore_from_ipfs()` call omitted `memory=` parameter
- Effect: constitutional genome from snapshot was never applied to Memory on cold-start restoration - revived organism started with default drives even when snapshot contained evolved genome
- Fix: `_check_skia_restore()` now accepts `memory` kwarg and passes it to `restore_from_ipfs()`
- Note: `event_bus` cannot be passed here (Synapse not yet initialized); `memory.seed_genome()` is the primary restore path; GENOME_EXTRACT_REQUEST broadcast fires once Synapse comes online
- File: `core/registry.py`, `_check_skia_restore()` + call site

## Integration Surface
### Emits
`skia_heartbeat`, `skia_heartbeat_lost`, `skia_restoration_triggered`, `skia_restoration_started`, `skia_restoration_complete`, `skia_restoration_completed`, `skia_snapshot_completed`, `skia_dry_run_complete`, `skia_resurrection_proposal`, `organism_spawned`, `organism_died`, `organism_resurrected`, `vitality_report`, `vitality_fatal`, `vitality_restored`, `metabolic_cost_report`, `system_modulation`, `degradation_override`, `child_died`, `federation_broadcast`, `re_training_example`, `incident_detected`, `fitness_observable_batch`, `genome_extract_request`, `skia_shadow_worker_deployed`, `skia_shadow_worker_missing`

**Gap closure (2026-03-07, event coverage):**
- `skia_snapshot_completed` - now emitted by `StateSnapshotPipeline.take_snapshot()` after each successful IPFS pin. Requires `snapshot.set_event_bus(bus)` (wired in `SkiaService.set_event_bus()` and `initialize()`).
- `skia_restoration_triggered` - now emitted in `SkiaService._on_death_confirmed()` immediately when restoration begins, before metabolic gates or fleet coordination, so the observatory can track all restoration attempts including blocked ones.

### Subscribes
`SOMA_VITALITY_SIGNAL`, `METABOLIC_PRESSURE`, `EVO_PARAMETER_ADJUSTED`, `ORGANISM_DIED`, `FEDERATION_RESURRECTION_APPROVED`, `DRIVE_EXTINCTION_DETECTED`, `ONEIROS_CONSOLIDATION_COMPLETE`, `EVO_BELIEF_CONSOLIDATED`, `SYSTEM_MODULATION_ACK`

### Wiring Points (called at organism bootstrap - all now LIVE)
- `skia.set_memory(memory_service)` - **WIRED** in `_init_skia()` before `initialize()` (2026-03-08)
- `skia.set_event_bus(event_bus)` - **WIRED** in `_init_skia()` after `initialize()` (registry.py)
- `skia.wire_vitality_systems(clock, oikos, thymos, equor, telos)` - **WIRED** in `_init_skia()` after `start()` (2026-03-08)
- `restore_from_ipfs(..., memory=memory)` - **WIRED** in `_check_skia_restore()` (2026-03-08); event_bus deferred (Synapse not yet available at restore time)
- Shadow worker loop: auto-started in `SkiaService.start()` when `not standalone and config.enabled and restoration is wired`; no explicit external wiring needed
