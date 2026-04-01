# Thymos - Immune System (Spec 12)

## What's Implemented

### Core Pipeline
- **6-tier repair**: NOOP → PARAMETER → RESTART → KNOWN_FIX → NOVEL_FIX → ESCALATE
- **5 sentinel types**: Exception (fingerprint = system+exc_type+first-local-frame), Contract (SLA monitoring), FeedbackLoop (15 severed-loop checks), Drift (8 metrics, σ-threshold SPC), CognitiveStall (broadcast ack rate, nova intent rate, etc.)
- **Triage**: composite severity scoring (blast radius 0.25 + recurrence velocity 0.20 + constitutional impact 0.25 + user visibility 0.15 + healing potential 0.15), fingerprint dedup with class-specific windows
- **`DiagnosticEngine`** with LLM-powered hypothesis generation (do NOT modify prompts) - max 3 hypotheses, testable via `DIAGNOSTIC_TEST_REGISTRY`
- **`CausalAnalyzer`**: Neo4j `(System)-[:DEPENDS_ON]->(System)` traversal (5-min cache, hardcoded fallback)
- **`TemporalCorrelator`**: TimescaleDB metric anomaly + system event queries (30s window)
- **`AntibodyLibrary`**: fingerprint lookup, effectiveness tracking (refinement <0.6, retirement <0.3 after 5+ apps), generation lineage in Neo4j
- **Repair validation gates**: Equor review (Tier 3+), blast radius check (>0.5 → escalate), Simula sandbox (Tier 4), rate limits (5/hour, 3 novel/day)
- **`HealingGovernor`**: rate limiting, storm detection, hysteresis-based storm exit
- **`HomeostasisController`**: adaptive baselines (7-day rolling median ± 25%)
- **`DriftSentinel`**: configurable σ-thresholds per metric
- **Embedding-based prophylactic scanner (P2, IMPLEMENTED 2026-03-07)**: 768-dim sentence-transformer cosine similarity >0.85 threshold; antibody fingerprint embeddings cached in `_fingerprint_store`; keyword fallback when embedder unavailable; `check_intent_similarity()` for intent-time gating; `add_fingerprints_from_procedures()` for Oneiros ingestion
- `ThymosService.set_embedding_client()` - hot-swap wiring from main.py; also wires into scanner if already initialized
- **INV-017 Drive Extinction handler (IMPLEMENTED 2026-03-07)**: `_on_drive_extinction()` subscribes to `DRIVE_EXTINCTION_DETECTED` (from Equor). Creates CRITICAL / `IncidentClass.DRIVE_EXTINCTION` incident with `blast_radius=1.0`, `user_visible=True`. No autonomous repair: blast_radius > 0.5 triggers validation gate escalation at Tier 3+ (`prescription.py:467`). Requires human/federation governance review.

### Synapse Integration (AV1 migration)
- All sub-components use `_on_event` callbacks bridged to Synapse by `ThymosService`
- Soma/Oikos health reads migrated from direct calls to cached subscription state
- Sandbox validation via correlation-based SIMULA_SANDBOX_REQUESTED/RESULT (60s timeout, fail-closed) - raised from 30s on 2026-03-09; Simula's internal ChangeSimulator runs up to 20s + event bus overhead was causing consistent timeouts at 30s
- 6+ lifecycle events emitted (INCIDENT_CREATED, REPAIR_APPLIED, REPAIR_ROLLED_BACK, etc.)
- Pydantic v2 validation on all 27+ subscribed events (`event_payloads.py`, non-blocking)

### Telemetry Loops
- `_vitality_loop()` - VITALITY_SIGNAL every 60s (antibody count, repair success rate, MTTH, novel ratio, storm status)
- `_drive_pressure_loop()` - THYMOS_DRIVE_PRESSURE every 30s (4 constitutional drives)

### Speciation Features
- Federation antibody sync: `export_for_federation()` / `import_from_federation()` with trust gating (ally/bonded/kin)
- RE training data: full repair episodes emitted as RE_TRAINING_DATA with measured outcome_quality
- Deepened SPECIATION_EVENT handler: antibody cross-ref, DriftSentinel tightening (×0.7), federation quarantine

## Pattern-Aware Routing (9 Mar 2026)

### CrashPatternAnalyzer (`backend/core/crash_pattern_analyzer.py`)
- Redis-backed `CrashPattern` store; key prefix `crash_pattern:*`
- `CrashPattern`: `id` (SHA-256 12-char from sorted signature), `signature` (feature list),
  `confidence` (0.0–0.98), `failed_tiers` (list), `highest_resolved_tier`, `occurrence_count`, `resolution_count`
- `extract_features()` - static; produces `frozenset[str]` from `source:*`, `class:*`, `etype:*`, `kw:*`, `affects:*` tokens
- `score_incident()` - `match_score = |incident ∩ signature| / |signature|`
- `load_all_patterns()` - SCAN cursor iteration over `crash_pattern:*` keys (non-blocking, count=100 batches)
- `update_on_success(pattern_id, repair_tier)` - decays confidence by 0.12 (floor 0.10); records `highest_resolved_tier`
- `update_on_failure(pattern_id, repair_tier)` - raises confidence by 0.08 (cap 0.98); appends tier to `failed_tiers`
- `register_pattern(features, description, initial_confidence, failed_tiers)` - idempotent; returns existing if already present

### PatternAwareRouter (`backend/systems/thymos/pattern_router.py`)
- Queried in `process_incident()` as **Step 2f**, after history-driven tier escalation, before Step 3 (Prescribe)
- Also consulted in **`_on_incident_inner` re-emission path** (Gap 6) - overrides tier on T4 recurrence escalations before `_process_incident_safe` runs
- Match thresholds: `match_score >= 0.7` AND `pattern.confidence >= 0.6`
- Returns `PatternRouteResult` with: `matched`, `pattern_id`, `pattern_confidence`, `match_score`, `tier_override`, `tier_skip_reason`, `skipped_tiers`, `federation_escalate`
- If all local tiers exhausted by `failed_tiers` → `federation_escalate=True` → ThymosService emits `INCIDENT_ESCALATED` (federation_broadcast=True, pattern_id attached) and returns without attempting local repair
- If pattern matched + tiers remain → `tier_override` = lowest available tier above highest failed → diagnosis overridden to skip wasted attempts
- **Auto-seeding (Gap 1)**: `_learn_from_failure()` registers a new `CrashPattern` when `matched_pattern_id is None` AND ≥2 distinct tiers have failed - initial_confidence=0.5, failed_tiers pre-populated
- **RE training timing (Gap 2)**: supplementary `anomaly_detection` RE example emitted after Step 2f when a pattern matched, capturing pattern_id + skipped_tiers (the `_on_incident_inner` example fires before pattern data is available)
- **Cross-instance confidence sync (Gap 4)**: subscribes to `CRASH_PATTERN_RESOLVED` + `CRASH_PATTERN_REINFORCED` on the Synapse bus; calls `update_on_success`/`update_on_failure` on local Redis to propagate peer repair experience

### Incident fields added
- `matched_pattern_id: str | None` - ID of the best-matching CrashPattern (None if no match)
- `pattern_confidence: float` - confidence of matched pattern at routing time
- `tier_skip_reason: str | None` - human-readable reason tiers were skipped

### Events
| Event | When emitted | Key payload |
|-------|-------------|-------------|
| `CRASH_PATTERN_RESOLVED` | Repair succeeded on pattern-matched incident | `pattern_id`, `repair_tier`, `strategy_used`, `time_to_resolve_ms`, `incident_id`, `confidence_before` |
| `CRASH_PATTERN_REINFORCED` | Repair failed/rolled-back on pattern-matched incident | `pattern_id`, `repair_tier`, `failure_reason`, `incident_id`, `confidence_before` |
| `CRASH_PATTERN_CONFIRMED` | Pattern confidence crosses 0.70 threshold (from below) on reinforcement; OR new pattern auto-seeded in `_learn_from_failure` | `pattern_id`, `confidence`, `lesson`, `example_count`, `failed_tiers`, `incident_id`, `source` ("thymos_reinforcement" / "thymos_rollback_reinforcement" / "thymos_auto_seed") |

**CRASH_PATTERN_CONFIRMED emission logic (2026-03-09):**
- Auto-seed path (`_learn_from_failure`): fires immediately when a new pattern is registered (confidence=0.5, example_count=len(repair_history))
- Reinforcement paths (both application failure and rollback paths): fires when `confidence_before < 0.70` AND `confidence_before + 0.08 >= 0.70` (i.e., this update crosses the confirmation threshold)
- Subscribers: `BenchmarkService._on_crash_pattern_confirmed`, `ThreadService._on_crash_pattern_confirmed`, `SimulaService._on_crash_pattern_confirmed`
- `CrashPattern` now imported alongside `CrashPatternAnalyzer` in `service.py` for `CrashPattern.make_id()` call at auto-seed site

### RE Training
- `repair_strategy` category RE examples now include `pattern_id`, `pattern_confidence`, and `tier_skip_reason` in `input_context` when a pattern match occurred

### set_redis()
- New `ThymosService.set_redis(redis)` method - hot-swaps Redis client into both `NotificationDispatcher` and `CrashPatternAnalyzer`

## What's Missing

| ID | Item | Priority |
|----|------|----------|
| SG1 | Mitosis antibody inheritance (federation sync done) | BLOCKED on Spec 26 |

## Completed (9 Mar 2026 - Resilience Audit)

### Resilience / RE Training Gaps (9 Mar 2026)

| Gap | Area | Fix | File |
|-----|------|-----|------|
| GAP-4 (F2) | Brain death SOS | `_on_skia_heartbeat_lost()` extended: when `system_id == "simula"`, emits `INCIDENT_ESCALATED` with `federation_broadcast=True`, `sos=True`, `capabilities_lost=["tier4_novel_fix","tier5_codegen","sandbox_validation"]` | `service.py` |
| GAP-6 (F1) | Coma recovery | `_check_pre_resurrection_crash_context()` - boot-time boot reads `skia:crash_context:{instance_id}` from Redis; if present and `request_simula_analysis=True`, creates `PreviousIncarnationCrash` CRASH incident with `repair_tier=NOVEL_FIX` and IPFS CID in context, then deletes the key. Fired as a deferred `asyncio.create_task` at end of `initialize()`. | `service.py` |
| Infrastructure | Crash context Redis access | `self._redis = redis` stored in `__init__` (was only passed to `NotificationDispatcher`, not retained) | `service.py` |

**GAP-6 closes the coma-recovery loop:**
`Skia detects death → writes skia:crash_context:{id} to Redis → restarts organism → Thymos reads context on boot → creates NOVEL_FIX incident → Simula analyses crash → generates fix → Thymos applies it`

### Events

| Event | Direction | Handler | Notes |
|-------|-----------|---------|-------|
| `PHANTOM_POOL_STALE` | SUBSCRIBE | `_on_phantom_pool_stale()` | DEGRADATION/MEDIUM - stale price oracle |
| `PHANTOM_POSITION_CRITICAL` | SUBSCRIBE | `_on_phantom_position_critical()` | RESOURCE_EXHAUSTION/HIGH - IL + range exit risk |
| `PHANTOM_RESOURCE_EXHAUSTED` | SUBSCRIBE | `_on_phantom_resource_exhausted()` | RESOURCE_EXHAUSTION/CRITICAL - oracle blind |
| `PHANTOM_IL_DETECTED` | SUBSCRIBE | `_on_phantom_il_detected()` | DEGRADATION/MEDIUM or LOW - IL level-gated |
| `RE_TRAINING_FAILED` | SUBSCRIBE | `_on_re_training_failed()` | DEGRADATION/HIGH - growth arrested |
| `RE_TRAINING_HALTED` | SUBSCRIBE | `_on_re_training_halted()` | SECURITY/HIGH - kill switch tripped |
| `INV_017_VIOLATED` | SUBSCRIBE | `_on_inv017_violated()` | DRIVE_EXTINCTION/CRITICAL - formal proof |
| `IDENTITY_CRISIS` | SUBSCRIBE | `_on_identity_crisis()` | DRIFT/CRITICAL - narrative shift ≥ 0.50 |
| `COMPUTE_REQUEST_DENIED` | SUBSCRIBE | `_on_compute_request_denied()` | RESOURCE_EXHAUSTION/HIGH - SACM denied |
| `DEPENDENCY_INSTALLED` | SUBSCRIBE | `_on_dependency_installed()` | Auto-resolves ImportError incidents for target system |
| `FOVEA_DISHABITUATION` | SUBSCRIBE | `_on_fovea_dishabituation()` | Re-sensitizes DriftSentinel σ for relevant system metrics (×0.8 toward 1.5σ floor) |
| `ACCOUNT_PROVISIONING_FAILED` | SUBSCRIBE | `_on_account_provisioning_failed()` | SECURITY/HIGH - Identity provisioning failure |
| `AFFECT_STATE_CHANGED` | SUBSCRIBE | `_on_affect_state_changed()` | Modulates DriftSentinel σ ±5–10% based on arousal×valence stress signal |
| `VULNERABILITY_CONFIRMED` | SUBSCRIBE | `_on_vulnerability_confirmed()` | SECURITY/CRITICAL - formally proved vuln from Simula |
| `EMERGENCY_WAKE` | EMIT | `_on_incident_inner()` Step 4a | Emitted on every CRITICAL incident; interrupts Oneiros sleep cycle |

## Completed (7 Mar 2026)

| ID | Item | Where |
|----|------|-------|
| M7 | Persist `(:Repair)` nodes with `[:REPAIRED_WITH]` edge for all outcomes | `service.py::_persist_repair_node()` |
| M8 | `HomeostasisController.check_drift_warnings()` - broadcasts HOMEOSTASIS_ADJUSTED (warn_only=True) in pre-repair zone | `prophylactic.py` + `service.py::_homeostasis_loop` |
| SG4 | `_try_federation_escalation()` - broadcast INCIDENT_ESCALATED to peers, wait 45s for FEDERATION_ASSISTANCE_ACCEPTED before human escalation | `service.py` |
| SG7 | Subscribe to KAIROS_INVARIANT_DISTILLED → `_on_kairos_invariant()` injects edges into CausalAnalyzer._graph_deps | `service.py` |
| P8 | Per-sentinel try/except → `_raise_sentinel_internal_incident()` creates MEDIUM DEGRADATION incident on sentinel crash | `service.py::_sentinel_scan_loop` |
| P8 | Version-rollback guard → `_request_model_version_rollback()` emits MODEL_ROLLBACK_TRIGGERED on MODEL_HOT_SWAP_FAILED | `service.py` |
| P2 | Upgrade prophylactic scanner to 768-dim embedding cosine similarity (>0.85 threshold); keyword fallback; fingerprint store cache; `check_intent_similarity()`; `set_embedding_client()` | `prophylactic.py` + `service.py` |
| SG8 | Subscribe to ONEIROS_CONSOLIDATION_COMPLETE → query `(:Procedure {thymos_repair: true})` → batch-embed → inject into prophylactic fingerprint store | `service.py::_on_oneiros_consolidation()` |
| Nova feedback | Emit THYMOS_REPAIR_VALIDATED on Tier 3+ verified success so Nova can strengthen recoverability priors | `service.py` + `synapse/types.py` |
| Identity #8 | Subscribe to `VAULT_DECRYPT_FAILED` → MEDIUM `SECURITY` incident; `VAULT_KEY_ROTATION_FAILED` → CRITICAL `SECURITY` incident. `_on_vault_decrypt_failed()` + `_on_vault_key_rotation_failed()` handlers added. New `IncidentClass.SECURITY` added to `primitives/incident.py`. | `service.py` |
| Oneiros threat scenarios | `ONEIROS_THREAT_SCENARIO` subscription (2026-03-08): `_on_oneiros_threat_scenario()` stores scenario in `_threat_scenario_cache`; for HIGH/CRITICAL severity, calls `_prophylactic_scanner.add_fingerprints_from_procedures()` to pre-arm the scenario as a prophylactic fingerprint so triage is faster when the real incident fires. | `service.py` |

## Known Issues
- AV1 migration is incremental - some direct cross-system calls may remain
- AV4: background tasks start before all subscriptions confirmed (race window)
- D1/D3/D4: dead code candidates (`SimulationResult`, `AddressBlacklistEntry`, `record_metric_anomaly`)
- SG4: `_event_bus.subscribe/unsubscribe` signature varies by Synapse implementation; handler may not deregister cleanly on all bus variants

## Key Files
- `service.py` - main ThymosService (~6100+ lines)
- `diagnosis.py` - CausalAnalyzer, DiagnosticEngine
- `antibody.py` - AntibodyLibrary with federation sync
- `governor.py` - HealingGovernor with storm hysteresis
- `prophylactic.py` - HomeostasisController with adaptive baselines + drift warnings
- `event_payloads.py` - 35+ Pydantic v2 payload models

## Integration Surface
- **Emits:** SYSTEM_HEALED, THYMOS_STORM_ENTERED/EXITED, RE_TRAINING_DATA, VITALITY_SIGNAL, THYMOS_DRIVE_PRESSURE, FEDERATION_ANTIBODY_SHARED, FEDERATION_TRUST_UPDATED, SIMULA_SANDBOX_REQUESTED, HOMEOSTASIS_ADJUSTED (warn_only), INCIDENT_ESCALATED (federation_broadcast), MODEL_ROLLBACK_TRIGGERED, THYMOS_REPAIR_VALIDATED (Tier 3+ success → Nova recoverability priors), IMMUNE_CYCLE_COMPLETE (end of every sentinel scan loop), THYMOS_REPAIR_APPROVED (Tier 5 Equor auto-approval → Simula resume), **EMERGENCY_WAKE** (2026-03-09) - emitted in `_on_incident_inner()` Step 4a on every CRITICAL incident to interrupt Oneiros sleep; payload: `incident_id`, `incident_class`, `source_system`, `reason`, `triggered_by`, **CRASH_PATTERN_RESOLVED** (2026-03-09) - repair succeeded on pattern-matched incident; **CRASH_PATTERN_REINFORCED** (2026-03-09) - repair failed on pattern-matched incident.
- **Consumes:** SYSTEM_FAILED, SYSTEM_DEGRADED, SOMATIC_MODULATION, SPECIATION_EVENT, FEDERATION_KNOWLEDGE_RECEIVED, SIMULA_SANDBOX_RESULT, KAIROS_INVARIANT_DISTILLED, FEDERATION_ASSISTANCE_ACCEPTED, ONEIROS_CONSOLIDATION_COMPLETE, ONEIROS_THREAT_SCENARIO, DRIVE_EXTINCTION_DETECTED, AXON_EXECUTION_REQUEST, AXON_ROLLBACK_INITIATED, VAULT_DECRYPT_FAILED, VAULT_KEY_ROTATION_FAILED, **PHANTOM_POOL_STALE**, **PHANTOM_POSITION_CRITICAL**, **PHANTOM_RESOURCE_EXHAUSTED**, **PHANTOM_IL_DETECTED**, **RE_TRAINING_FAILED**, **RE_TRAINING_HALTED**, **INV_017_VIOLATED**, **IDENTITY_CRISIS**, **COMPUTE_REQUEST_DENIED**, **DEPENDENCY_INSTALLED**, **FOVEA_DISHABITUATION**, **ACCOUNT_PROVISIONING_FAILED**, **AFFECT_STATE_CHANGED**, **VULNERABILITY_CONFIRMED** (all 2026-03-09), + 20 more (see event_payloads.py _PAYLOAD_MODELS)
- **Neo4j:** reads `(System)-[:DEPENDS_ON]->(System)` for causal analysis; writes `(:Repair)-[:REPAIRED_WITH]->(:Incident)`
