# SACM - CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_27_SACM.md`
**System ID:** `sacm`
**Role:** Substrate-Arbitrage Compute Mesh - distributed compute orchestration across local, Cloud Run, and Akash. Offloads expensive workloads, arbitrages pricing, pre-warms capacity, and enforces metabolic budgets.

---

## What Is Implemented

**Core Modules:**
- `compute_manager.py` - `SACMComputeManager`: main orchestrator, fair-share allocation, queue management, lifecycle event handling, GenomeExtractionProtocol
- `service.py` - `SACMService`: top-level service facade, Redis-backed workload history persistence
- `migrator.py` - `MigrationExecutor` + `CostTriggeredMigrationMonitor`: **NEW** - atomic provider migration (Cloud Run ↔ Akash) with Equor constitutional gate, 24h circuit breaker, distributed lock, rollback on failure. `CostTriggeredMigrationMonitor` subscribes to `ORGANISM_TELEMETRY` and proposes migration when infra cost is >20% above cheapest alternative for >6 consecutive hours.
- `accounting.py` - `SACMAccounting`: budget tracking, burn rate, Synapse stress emission (decoupled from Soma)
- `pre_warming.py` - `PreWarmingEngine`: EMA demand prediction, warm pool management, price-opportunity detection, provisioning event emission
- `oracle.py` - `ComputeMarketOracle`: pricing surface snapshots from substrate providers
- `optimizer.py` - cost estimation and allocation optimization
- `workload.py` - workload types, `OffloadClass`, `ResourceEnvelope`, `ComputeRequest`, `AllocationDecision`
- `config.py` - all SACM configuration (budgets, timing, pre-warm thresholds)
- `partition.py` - workload partitioning for distributed execution
- `encryption.py` - encrypted compute envelopes (Ed25519/X25519+AES-256-GCM)
- `remote_executor.py` / `remote_compute_executor.py` - remote execution management
- `providers/akash.py` - Akash Network substrate provider
- `verification/` - consensus, deterministic, and probabilistic result verification

**Synapse Integration:**
- Subscribed events: `COMPUTE_OFFLOAD_SUBMITTED`, `COMPUTE_OFFLOAD_RESULT`, `PRICING_SURFACE_UPDATED`, `ORGANISM_SLEEP`, `ORGANISM_WAKE`, `METABOLIC_EMERGENCY`, `RESOURCE_PRESSURE` (emitted by Thymos), `GENOME_EXTRACT_REQUEST`, `ORGANISM_TELEMETRY` (CostTriggeredMigrationMonitor - cost monitoring), `EQUOR_ECONOMIC_PERMIT` (MigrationExecutor - migration approval gate)
- Emitted events: `SACM_COMPUTE_STRESS` (burn rate → Soma), `SACM_PRE_WARM_PROVISIONED` (warm instance creation), `GENOME_EXTRACT_RESPONSE`, `RE_TRAINING_EXAMPLE` (compute allocation decisions), `FOVEA_INTERNAL_PREDICTION_ERROR` (cost surprises ≥50% over estimate → Fovea), `EVO_HYPOTHESIS_CONFIRMED` / `EVO_HYPOTHESIS_REFUTED` (provider reliability evidence → Evo), `ALLOCATION_RELEASED` (capacity freed after workload), `SACM_DRAINING` (graceful shutdown signal), `COMPUTE_MIGRATION_STARTED` / `COMPUTE_MIGRATION_COMPLETED` / `COMPUTE_MIGRATION_FAILED` (migration lifecycle), `COMPUTE_ARBITRAGE_DETECTED` (cost threshold breach detected), `ORGANISM_SHUTDOWN_REQUESTED` (signals old instance to shut down post-migration)

**Lifecycle Handling:**
- `ORGANISM_SLEEP`: downgrades non-CRITICAL queue to BATCH, pauses pre-warming
- `ORGANISM_WAKE`: clears sleep flag, restarts pre-warming
- `METABOLIC_EMERGENCY`: drains non-critical queue, stops pre-warming, gates `submit_request()` to deny all non-CRITICAL

**GenomeExtractionProtocol:**
- Heritable state: fair-share weights, capacity config, pre-warming config
- `extract_genome_segment()` / `seed_from_genome_segment()` on `SACMComputeManager`

**RE Training:**
- Every compute allocation decision (allocated, denied-fair-share, denied-queue-full) emits `RETrainingExample` via Synapse

**Workload History:**
- Redis sorted set (`sacm:workload_history`), scored by `submitted_at`, auto-trimmed to 500 entries
- Falls back to in-memory when Redis unavailable

**Migration Safety (MigrationExecutor):**
- **Equor gate**: every migration proposal goes through `EQUOR_ECONOMIC_INTENT` → `EQUOR_ECONOMIC_PERMIT` event pair (30s timeout, auto-permit as survival fallback)
- **24h circuit breaker**: Redis sorted set `sacm:migration:history` tracks recent migrations; `ComputeArbitrageConfig.max_migrations_per_24h` (default 1) enforced before acquiring lock
- **Distributed lock**: `eos:sacm:migration:lock` (SETNX, 15-min TTL) prevents concurrent migrations
- **Atomic 8-step sequence**: approve → circuit-break → lock → snapshot → provision → verify → switch routing → shutdown old
- **Rollback guarantee**: any step failure calls `_rollback_new_instance()` and emits `COMPUTE_MIGRATION_FAILED`; old instance remains active
- **Routing switch**: Redis keys `sacm:current_provider_endpoint` + `sacm:current_provider` updated atomically
- **Neo4j audit**: `MigrationRecord` persisted to Redis sorted set (max 100 entries), `(:EconomicEvent)` node written via `_write_migration_neo4j()`
- **Health verification**: `handoff_healthy_threshold` (default 3) consecutive `/health` 200s required within `handoff_timeout_s` (default 300s)

---

## Known Issues / Remaining

1. **Verification modules are stubs** - consensus, deterministic, and probabilistic verifiers exist but may not be fully integrated into the execution pipeline.
2. **Router private attribute access** - `api/routers/sacm.py` reads `_metrics`, `_pool`, `_held_cpu`, `_predictor` etc. directly; encapsulation violation.
3. **PlacementDecision location** - still lives in `remote_executor.py` rather than `optimizer.py`; stale circular-import comment.
4. **`oracle.snapshot` property vs method** - spec examples show `oracle.snapshot()` call; implementation is `@property`.
5. ~~**MigrationExecutor not wired in registry.py**~~ - **FIXED**: `MigrationExecutor` and `CostTriggeredMigrationMonitor` now instantiated in `core/registry.py._init_sacm()` and event-bus-wired in `core/wiring.py.wire_synapse_phase()`.
6. ~~**`_write_migration_neo4j()` stub**~~ - **FIXED**: `set_neo4j()` added to `MigrationExecutor`; `_write_migration_neo4j()` fully implemented as idempotent MERGE; driver injected from `registry.py`.

## Dead Wiring Closed (8 Mar 2026 Autonomy Audit)

| Setter | Location | Impact of fix |
|--------|----------|---------------|
| `SACMCostAccounting.set_synapse()` | `wiring.py wire_synapse_phase` | Enables `SACM_COMPUTE_STRESS`, `EVO_HYPOTHESIS_CONFIRMED/REFUTED`, `FOVEA_INTERNAL_PREDICTION_ERROR` emissions |
| `SACMCostAccounting.set_neo4j()` | `registry.py` Phase 1 | Enables `CostRecord → (:EconomicEvent)` Neo4j audit trail |
| `SACMWorkloadHistoryStore.set_redis()` + `load_from_redis()` | `registry.py` Phase 1 | Workload history persists across restarts |
| `PreWarmingEngine.set_synapse()` | `wiring.py wire_synapse_phase` | Enables `SACM_PRE_WARM_PROVISIONED` Synapse emission |
| `ComputeResourceManager.set_pre_warming()` | `wiring.py wire_synapse_phase` | `ORGANISM_SLEEP/WAKE/METABOLIC_EMERGENCY` can pause/resume pre-warm loop |
| `MigrationExecutor.set_event_bus()` | `wiring.py wire_synapse_phase` | `EQUOR_ECONOMIC_PERMIT` subscription live; migrations can receive approval |
| `CostTriggeredMigrationMonitor.set_event_bus()` | `wiring.py wire_synapse_phase` | `ORGANISM_TELEMETRY` subscription live; cost arbitrage detection active |
| `MigrationExecutor.set_neo4j()` (new method) | `registry.py` Phase 1 | `MigrationRecord → (:EconomicEvent)` Neo4j audit trail |

## Architecture Violations Fixed (this session)

| ID | Fix |
|----|-----|
| SG2 | **Oikos metabolic gate wired** - `PreWarmingEngine._create_warm_instance()` now calls `await self._oikos.check_metabolic_gate(...)` with `MetabolicPriority.GROWTH` before provisioning. Denied gate returns a `WarmInstance(status=RELEASED)` stub so the calling loop skips gracefully. Non-fatal if Oikos unavailable. |
| SG4 | **Evo provider performance signals** - `SACMCostAccounting.record_execution()` now emits `EVO_HYPOTHESIS_CONFIRMED` / `EVO_HYPOTHESIS_REFUTED` after each execution with `hypothesis_id=sacm.provider_reliability.<provider_id>`, reliability score (verification + acceptance), cost accuracy, and composite quality. Evo can apply Thompson sampling to learn substrate preferences per workload type. |
| SG5 | **Fovea cost-surprise signals** - `SACMCostAccounting.record_execution()` now accepts `estimated_cost_usd` and emits `FOVEA_INTERNAL_PREDICTION_ERROR` when actual cost ≥ 1.5× estimate. Payload includes `prediction_error.economic = ratio - 1.0` and `salience_hint = min(1.0, magnitude / 2.0)`. |
| P1 | **ALLOCATION_RELEASED event** - `ComputeResourceManager.release()` now emits `ALLOCATION_RELEASED` via `_emit()` with full capacity payload (request_id, source_system, cpu/gpu/memory released, held_s, node_id, available, utilisation_pct). `ALLOCATION_RELEASED` added to `SynapseEventType`. |
| P2 | **Keypair bug fixed** - `remote_executor.py` keypair now generated in Phase 2 (before encrypt). `our_keypair.public_bytes.hex()` passed as `response_public_key` in dispatch metadata. Phase 4 reuses `our_keypair.private_key` - decryption now succeeds. |
| P3 | **Oracle UNREACHABLE recovery** - `ComputeMarketOracle.start()` / `stop()` lifecycle methods launch a background `_recovery_loop` that re-health-checks UNREACHABLE providers every 5 minutes and restores them to AVAILABLE on success. |
| P4 | **Pre-warming actually provisions** - `PreWarmingEngine.register_provider_manager(provider_id, manager)` wires infrastructure layer. `_create_warm_instance()` launches `_provision_via_manager(inst, manager)` as background task; calls `manager.deploy()`, transitions to READY on success or RELEASED on failure. |
| M3 | **CostRecord → Neo4j** - `SACMCostAccounting.record_execution()` fires `_write_economic_event_neo4j(record)` as background task. Writes `(:EconomicEvent)` node via idempotent MERGE. `set_neo4j(driver)` for injection. |
| M4 | **Secondary provider retry** - `SACMService._execute_and_resolve(plan)` iterates `plan.feasible_placements()[1:]` up to `max_retries` on primary failure. `_scored_placement_to_decision()` converts each candidate. |
| M5 | **Shutdown drain** - `SACMService.shutdown(drain_timeout_s=30.0)` emits `SACM_DRAINING`, then `asyncio.wait_for(_drain_pending(), timeout)`. `SACM_DRAINING` added to `SynapseEventType`. |
| M8 | **Akash exchange rate fallback** - `SACMAkashProvider._fallback_offers()` returns single DEGRADED CPU offer with hardcoded USD/second rates on CoinGecko/API failure. `trust_score=0.60`, `metadata.pricing_source="fallback"`. |

---

## Integration Surface

| System | Direction | Mechanism |
|--------|-----------|-----------|
| Soma | → | `SACM_COMPUTE_STRESS` event (Soma subscribes for allostatic signal) |
| Oikos | ← | Budget reads + metabolic gate via `wire_oikos()` on PreWarmingEngine; gate called before every pre-warm provisioning |
| Equor | ← | `EQUOR_ECONOMIC_PERMIT` - constitutional gate for all provider migrations |
| Evo | → | `EVO_HYPOTHESIS_CONFIRMED` / `EVO_HYPOTHESIS_REFUTED` per execution - provider reliability evidence for Thompson sampling |
| Fovea | → | `FOVEA_INTERNAL_PREDICTION_ERROR` on ≥50% cost surprises - economic prediction error for precision weighting |
| Mitosis | ↔ | `GENOME_EXTRACT_REQUEST/RESPONSE` for heritable compute state |
| Skia | ← | `StateSnapshotPipeline` - MigrationExecutor forces snapshot before migration to capture current state CID |
| Synapse | ↔ | All event pub/sub, lifecycle coordination |
| RE | → | `RE_TRAINING_EXAMPLE` events for compute allocation training data |
