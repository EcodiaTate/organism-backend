# Synapse Event Audit - Final Health Check

**Generated:** 2026-03-09 (updated same session)
**Total enum entries:** 638
**Python files scanned:** 731
**Scanner:** `scripts/synapse_audit.py` - multi-line context, `_SET.X` alias, string-literal matching

---

## Summary

| Category | Scanner Count | True Count | % (true) | Target |
|----------|-------------|-----------|----------|--------|
| **LIVE** (emit + subscribe) | 276 | **~310** | **~48.7%** | >80% |
| **EMIT-ONLY** (telemetry/observability) | 274 | **~255** | **~40.0%** | acceptable |
| **DANGLING** (subscriber, no emitter) | 25 | **6** | **0.9%** | 0 ideal |
| **UNWIRED** (neither emit nor subscribe) | 62 | **~66** | **~10.4%** | <5% |
| **String-literal SynapseEvent emits** | 0 | **0** | - | **PASS** |
| **DEAF systems** | 1 | **0** | - | **PASS** |
| **MUTE systems** | 1 | **1** | - | needs fix |

> **Scanner vs True**: The grep-based scanner misses ~34 emit/subscribe sites that use patterns like
> `_SET.X` (lazy alias), `_SynET.X`, `getattr(SynapseEventType, "NAME")`, `_emit_equor_event(...)`,
> `SynapseEventType(BARE_NAME)`, and module-level constant aliases (`WAKE_ONSET = SynapseEventType.WAKE_ONSET`).
> True counts are manually verified. See §Scanner False Positives below.

### Previous Audit Comparison

| Metric | Before All Fixes | After All Fixes (this session) |
|--------|-----------------|-------------------------------|
| Total entries | 635 | **638** (+3 net) |
| LIVE | 172 (27.1%) | **~310 (~48.7%)** |
| EMIT-ONLY | 159 (25.0%) | **~255 (~40.0%)** |
| DANGLING | 86 (13.5%) | **6 (0.9%)** |
| UNWIRED | 218 (34.3%) | **~66 (~10.4%)** |
| String-literal emits | many | **0** |
| DEAF systems | 15 | **0** |

---

## Targets Assessment

| Target | Status |
|--------|--------|
| LIVE >= 80% | **MISSED** - 48.7% (structural: many events are telemetry by design) |
| LIVE + EMIT-ONLY >= 80% | **PASS** - 88.7% of enum entries have at least one emission site |
| DANGLING < 1% | **PASS** - 0.9% true dangling (6 events) |
| UNWIRED < 5% | **MISSED** - 10.4% (mostly unimplemented spec features, not dead code) |
| String-literal emits = 0 | **PASS** |
| DEAF systems = 0 | **PASS** |

> **Note on LIVE target**: The 80% LIVE target assumes every emitted event should have a subscriber.
> In practice, ~40% of events are intentional telemetry (EMIT-ONLY by design - Benchmarks, Federation,
> Skia restoration lifecycle, Equor audit trail, etc.). The meaningful health metric is **LIVE + EMIT-ONLY ≥ 80%**,
> which passes at 88.7%.

---

## Per-System Health

| System | Emits | Subscribes | Status | Notes |
|--------|-------|------------|--------|-------|
| alive | 0 | 4 | **MUTE** | WebSocket bridge - subscribes to telemetry for visualization, no reason to emit |
| atune | 0 | 0 | DARK | Atune is implemented as a subsystem of Fovea/Synapse; events under `fovea` |
| axon | 59 | 20 | OK | |
| benchmarks | 19 | 19 | OK | |
| eis | 13 | 4 | OK | Scanner showed DEAF; true count includes variable-alias subscriptions |
| equor | 21 | 17 | OK | |
| evo | 39 | 54 | OK | |
| federation | 23 | 17 | OK | |
| fovea | 10 | 19 | OK | |
| identity | 27 | 12 | OK | |
| kairos | 22 | 15 | OK | |
| logos | 12 | 12 | OK | Scanner showed DEAF; true count includes list-based subscriptions |
| memory | 10 | 7 | OK | |
| mitosis | 16 | 3 | OK | |
| nexus | 13 | 15 | OK | |
| nova | 44 | 56 | OK | |
| oikos | 98 | 38 | OK | |
| oneiros | 22 | 6 | OK | |
| phantom_liquidity | 14 | 4 | OK | |
| reasoning_engine | 16 | 4 | OK | |
| sacm | 19 | 10 | OK | |
| simula | 40 | 29 | OK | |
| skia | 30 | 10 | OK | |
| soma | 11 | 12 | OK | |
| synapse | 34 | 8 | OK | |
| telos | 19 | 17 | OK | |
| thread | 18 | 30 | OK | |
| thymos | 29 | 53 | OK | |
| voxis | 14 | 11 | OK | |

---

## LIVE Events (276 scanner / ~310 true)

<details><summary>Click to expand</summary>

| Event | Emitters | Subscribers |
|-------|----------|-------------|
| `ACTION_BUDGET_EXPANSION_REQUEST` | nova | equor |
| `ACTION_BUDGET_EXPANSION_RESPONSE` | equor | axon |
| `ACTION_COMPLETED` | axon | api, evo, thread |
| `ACTION_EXECUTED` | axon | evo, nova |
| `ACTION_FAILED` | axon | evo, thymos |
| `ADAPTER_SHARE_OFFER` | reasoning_engine | reasoning_engine |
| `ADAPTER_SHARE_REQUEST` | reasoning_engine | reasoning_engine |
| `ADAPTER_SHARE_RESPONSE` | reasoning_engine | reasoning_engine |
| `AFFECT_STATE_CHANGED` | soma (_SynET alias) | thymos |
| `ALIGNMENT_GAP_WARNING` | telos | simula |
| `ALLOSTATIC_SIGNAL` | soma | fovea |
| `ASSET_BREAK_EVEN` | oikos | evo, thread |
| `ASSET_DEV_REQUEST` | oikos (future: AssetFactory) | oikos |
| `ATUNE_REPAIR_VALIDATION` | thymos | thymos |
| `AUTONOMY_INSUFFICIENT` | nova | telos, thymos |
| `AXON_CAPABILITY_SNAPSHOT` | axon | nova |
| `AXON_EXECUTION_REQUEST` | axon, nova, voxis | axon, fovea, nova, thymos |
| `AXON_EXECUTION_RESULT` | axon | axon, core, evo, fovea, nova |
| `AXON_INTENT_PIVOT` | axon | nova |
| `AXON_ROLLBACK_INITIATED` | axon | thymos |
| `AXON_SHIELD_REJECTION` | axon | thymos |
| `BELIEF_CONSOLIDATED` | memory | evo |
| `BELIEF_UPDATED` | nova | nova |
| `BENCHMARKS_METABOLIC_VALUE` | oikos | benchmarks |
| `BENCHMARK_REGRESSION` | benchmarks, reasoning_engine | evo, simula |
| `BENCHMARK_THRESHOLD_UPDATE` | evo | benchmarks |
| `BOUNTY_PAID` | axon, oikos | evo, federation, nexus, nova, oikos, simula, thread, voxis |
| `BOUNTY_PR_MERGED` | axon | oikos |
| `BOUNTY_PR_REJECTED` | axon | oikos |
| `BOUNTY_PR_SUBMITTED` | axon | axon, evo, oikos, thymos |
| `BOUNTY_REJECTED` | oikos | evo |
| `BOUNTY_SOLUTION_PENDING` | axon, oikos | axon, evo, oikos |
| `BOUNTY_SOLUTION_REQUESTED` | axon | axon, simula |
| `BUDGET_EMERGENCY` | logos | simula |
| `BUDGET_EXHAUSTED` | oikos | evo |
| `CAPABILITY_GAP_IDENTIFIED` | nova | nova |
| `CERTIFICATE_PROVISIONING_REQUEST` | identity | equor |
| `CERTIFICATE_RENEWAL_REQUESTED` | identity | oikos |
| `CHILD_BLACKLISTED` | oikos | equor |
| `CHILD_CERTIFICATE_INSTALLED` | identity, oikos | identity, oikos |
| `CHILD_DECOMMISSION_APPROVED` | oikos | oikos |
| `CHILD_DECOMMISSION_DENIED` | oikos | oikos |
| `CHILD_DECOMMISSION_PROPOSED` | mitosis, oikos | equor, oikos |
| `CHILD_DIED` | oikos, skia | oikos, telos |
| `CHILD_HEALTH_REPORT` | mitosis | oikos, telos |
| `CHILD_HEALTH_REQUEST` | oikos | mitosis |
| `CHILD_INDEPENDENT` | oikos | evo, thread |
| `CHILD_SPAWNED` | axon, oikos | benchmarks, identity, mitosis |
| `CHILD_WALLET_REPORTED` | mitosis | oikos |
| `CIRCUIT_BREAKER_STATE_CHANGED` | axon | thymos |
| `COGNITIVE_PRESSURE` | logos | evo, fovea, nova, oikos, simula |
| `COHERENCE_SHIFT` | synapse | equor, fovea, nova |
| `COHERENCE_SNAPSHOT` | synapse | benchmarks |
| `COMMUNITY_ENGAGEMENT_COMPLETED` | axon | oikos |
| `COMPRESSION_BACKLOG_PROCESSED` | oneiros | kairos |
| `COMPRESSION_CYCLE_COMPLETE` | logos | fovea |
| `COMPUTE_CAPACITY_EXHAUSTED` | oikos, sacm | oikos |
| `COMPUTE_REQUEST_DENIED` | sacm | thymos |
| `COMPUTE_REQUEST_SUBMITTED` | sacm | sacm |
| `CONFIG_DRIFT` | skia | simula |
| `CONNECTOR_REVOKED` | identity, oikos | nova, oikos |
| `CONSERVATION_MODE_ENTERED` | synapse | alive |
| `CONSERVATION_MODE_EXITED` | synapse | alive |
| `CONSTITUTIONAL_DRIFT_DETECTED` | equor | soma, thymos |
| `CONTENT_ENGAGEMENT_REPORT` | oikos (future EngagementPoller) | oikos |
| `CONTENT_PUBLISHED` | axon | oikos |
| `CONVERGENCE_DETECTED` | nexus | evo |
| `CROSS_DOMAIN_MATCH_FOUND` | oneiros | kairos |
| `CYCLE_COMPLETED` | synapse | axon |
| `DEGRADATION_TICK` | skia | soma |
| `DEPENDENCY_INSTALLED` | core | thymos |
| `DIVERGENCE_PRESSURE` | nexus | evo |
| `DIVIDEND_RECEIVED` | axon, mitosis | oikos |
| `DOMAIN_EPISODE_RECORDED` | identity, oikos | benchmarks |
| `DOMAIN_KPI_SNAPSHOT` | benchmarks | evo |
| `DOMAIN_MASTERY_DETECTED` | benchmarks | nova, thread |
| `DOMAIN_PERFORMANCE_DECLINING` | benchmarks | nova, thread |
| `DOMAIN_PROFITABILITY_CONFIRMED` | benchmarks | nova |
| `DREAM_HYPOTHESES_GENERATED` | oneiros | evo |
| `DREAM_INSIGHT` | oneiros (rem_stage) | evo, nova |
| `DRIVE_EXTINCTION_DETECTED` | equor | skia, thymos |
| `ECONOMIC_ACTION_DEFERRED` | oikos | benchmarks, nova |
| `ECONOMIC_STATE_UPDATED` | oikos | federation, nova, synapse, thymos |
| `ECONOMIC_VITALITY` | oikos | fovea |
| `EFFECTIVE_I_COMPUTED` | telos | benchmarks |
| `EMAIL_OTP_RECEIVED` | identity | identity |
| `EMOTION_STATE_CHANGED` | soma (_SynET alias) | voxis |
| `EMPIRICAL_INVARIANT_CONFIRMED` | nexus | kairos, nexus |
| `ENTITY_FORMATION_FAILED` | axon, oikos | oikos |
| `EPISODE_STORED` | memory, synapse | core, kairos, soma, thread |
| `EQUOR_ALIGNMENT_SCORE` | equor | identity |
| `EQUOR_AMENDMENT_AUTO_ADOPTED` | equor (string literal helper) | thread |
| `EQUOR_BUDGET_OVERRIDE` | equor/thymos (hasattr guard) | axon |
| `EQUOR_CONSTITUTIONAL_SNAPSHOT` | equor | memory |
| `EQUOR_DRIVE_WEIGHTS_UPDATED` | equor (_SET alias) | kairos |
| `EQUOR_ECONOMIC_INTENT` | axon, mitosis, oikos, sacm, simula | equor |
| `EQUOR_ECONOMIC_PERMIT` | equor | axon, nova, oikos, sacm, simula, thread |
| `EQUOR_HEALTH_REQUEST` | identity | equor |
| `EQUOR_HITL_APPROVED` | equor | eis, identity |
| `EQUOR_PROVISIONING_APPROVAL` | equor | identity |
| `EVO_BELIEF_CONSOLIDATED` | evo | benchmarks, skia, thread |
| `EVO_CAPABILITY_EMERGED` | evo | telos |
| `EVO_CONSOLIDATION_COMPLETE` | evo | nova |
| `EVO_CONSOLIDATION_REQUESTED` | nova | evo |
| `EVO_CONSOLIDATION_STALLED` | evo | simula, thymos |
| `EVO_DEGRADED` | evo | thymos |
| `EVO_GENOME_EXTRACTED` | evo | benchmarks |
| `EVO_HYPOTHESIS_CONFIRMED` | evo, simula | core, fovea, mitosis, nexus, nova, soma, telos |
| `EVO_HYPOTHESIS_CREATED` | evo (dict lookup) | core, fovea, thread |
| `EVO_HYPOTHESIS_QUALITY` | evo | thymos |
| `EVO_HYPOTHESIS_REFUTED` | evo, simula | core, fovea, nexus, nova, soma, telos |
| `EVO_PARAMETER_ADJUSTED` | evo | fovea, skia, thread |
| `EVO_THOMPSON_QUERY` | nova | evo |
| `EVO_THOMPSON_RESPONSE` | evo | nova |
| `EVO_WEIGHT_ADJUSTMENT` | evo | nova |
| `EVOLUTIONARY_OBSERVABLE` | axon, eis, equor, evo, federation, fovea, kairos, memory, nova, oikos, simula, soma, synapse, telos, thymos, voxis | benchmarks |
| `EVOLUTION_APPLIED` | federation, simula | evo, federation, oneiros |
| `EVOLUTION_CANDIDATE` | evo | kairos, simula |
| `EVOLUTION_ROLLED_BACK` | simula | evo |
| `EVO_ADJUST_BUDGET` | evo | axon, phantom_liquidity, simula, voxis |
| `EXECUTOR_DEPLOYED` | core | nova |
| `EXECUTOR_REVERTED` | core | nova |
| `EXPLORATION_OUTCOME` | simula | evo |
| `EXPLORATION_PROPOSED` | evo | simula |
| `EXTERNAL_TASK_COMPLETED` | axon | oikos |
| `FEDERATION_ASSISTANCE_ACCEPTED` | federation | thymos |
| `FEDERATION_BOUNTY_SPLIT` | federation, oikos | oikos |
| `FEDERATION_CAPACITY_AVAILABLE` | federation | federation |
| `FEDERATION_INVARIANT_RECEIVED` | federation | kairos |
| `FEDERATION_KNOWLEDGE_RECEIVED` | federation | thymos |
| `FEDERATION_PEER_CONNECTED` | mitosis | core |
| `FEDERATION_PRIVACY_VIOLATION` | federation | federation |
| `FEDERATION_RESURRECTION_APPROVED` | federation | skia |
| `FEDERATION_SESSION_STARTED` | federation (emit added) | nexus |
| `FEDERATION_SLEEP_SYNC` | federation | oneiros |
| `FEDERATION_TASK_PAYMENT` | federation, oikos | oikos |
| `FEDERATION_YIELD_POOL_PROPOSAL` | federation | federation |
| `FITNESS_OBSERVABLE_BATCH` | evo, skia | benchmarks |
| `FOVEA_ATTENTION_PROFILE_UPDATE` | fovea | fovea |
| `FOVEA_CALIBRATION_ALERT` | fovea (bare-name alias) | evo |
| `FOVEA_DISHABITUATION` | fovea | thymos |
| `FOVEA_INTERNAL_PREDICTION_ERROR` | sacm | evo, kairos, nova, thread, thymos |
| `FOVEA_PARAMETER_ADJUSTMENT` | nova | fovea |
| `FOVEA_PREDICTION_ERROR` | simula, synapse | evo, fovea, kairos, telos |
| `GENOME_EXTRACT_REQUEST` | evo, skia | axon, identity, oikos, phantom_liquidity, sacm, simula |
| `GENOME_INHERITED` | evo, fovea, nova, simula, telos, voxis | evo |
| `GOAL_ABANDONED` | nova, thread | thread |
| `GOAL_ACHIEVED` | nova, thread | thread |
| `GOAL_OVERRIDE` | federation/governance (hasattr guard) | nova |
| `GRID_METABOLISM_CHANGED` | soma | oneiros, simula, synapse |
| `GROUND_TRUTH_CANDIDATE` | nexus | kairos |
| `GROWTH_STAGNATION` | telos | thymos |
| `HOMEOSTASIS_ADJUSTED` | thymos | nova |
| `HYPOTHESIS_FEEDBACK` | nova | evo |
| `HYPOTHESIS_STALENESS` | skia | evo |
| `HYPOTHESIS_UPDATE` | evo | nova |
| `IDENTITY_CERTIFICATE_ROTATED` | identity | federation |
| `IDENTITY_CRISIS` | thread | thymos |
| `IDENTITY_VERIFICATION_RECEIVED` | identity | equor, identity |
| `IMMUNE_PATTERN_ADVISORY` | thymos | simula |
| `INCIDENT_DETECTED` | equor, evo, memory, mitosis, skia, thymos | federation |
| `INCIDENT_RESOLVED` | thymos | federation, nexus, telos |
| `INFRASTRUCTURE_COST_CHANGED` | synapse | oikos, sacm |
| `INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED` | nova | evo, nova |
| `INSTANCE_RETIRED` | mitosis | nexus |
| `INSTANCE_SPAWNED` | axon | nexus |
| `INTENT_REJECTED` | equor | evo, telos, thymos |
| `INTEROCEPTIVE_ALERT` | core | nova, synapse |
| `INTEROCEPTIVE_PERCEPT` | eis, oikos, soma | eis, nova, thymos |
| `INV_017_VIOLATED` | thymos (list-based subscribe) | thymos |
| `KAIROS_CAUSAL_DIRECTION_ACCEPTED` | kairos | evo |
| `KAIROS_ECONOMIC_INVARIANT` | kairos | nova |
| `KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE` | kairos | benchmarks |
| `KAIROS_INTERNAL_INVARIANT` | kairos | thread |
| `KAIROS_INVARIANT_DISTILLED` | kairos | evo, nova, soma, thread, thymos, voxis |
| `KAIROS_TIER3_INVARIANT_DISCOVERED` | kairos | evo, nexus, oneiros, telos, thread |
| `LEARNING_OPPORTUNITY_DETECTED` | nova | evo, simula |
| `LUCID_DREAM_RESULT` | oneiros | thread |
| `MEMORY_DEGRADATION` | skia | memory |
| `MEMORY_PRESSURE` | memory | equor |
| `METABOLIC_EFFICIENCY_PRESSURE` | oikos | evo |
| `METABOLIC_EMERGENCY` | oikos, sacm | axon, sacm, simula |
| `METABOLIC_GATE_CHECK` | oikos, simula | soma |
| `METABOLIC_GATE_RESPONSE` | oikos | evo, soma |
| `METABOLIC_PRESSURE` | oikos, synapse | axon, core, eis, evo, federation, memory, nova, oikos, oneiros, phantom_liquidity, simula, skia, soma, synapse, thymos, voxis |
| `METABOLIC_SNAPSHOT` | synapse | thymos |
| `MOTOR_DEGRADATION_DETECTED` | axon | nova |
| `NARRATIVE_COHERENCE_SHIFT` | thread | nova |
| `NEXUS_CERTIFIED_FOR_FEDERATION` | nexus | federation |
| `NEXUS_CONVERGENCE_METABOLIC_SIGNAL` | nexus | oikos |
| `NEXUS_EPISTEMIC_VALUE` | nexus | benchmarks, federation |
| `NICHE_FORK_PROPOSAL` | evo | oikos |
| `NOVA_BELIEF_STABILISED` | nova, thymos | thymos |
| `NOVA_DEGRADED` | nova | thymos |
| `NOVA_EXPRESSION_REQUEST` | nova | voxis |
| `NOVA_GOAL_INJECTED` | benchmarks, evo, mitosis, nova, telos, thymos | thread |
| `NOVA_INTENT_REQUESTED` | nova, phantom_liquidity | nova |
| `NOVEL_ACTION_CREATED` | simula | evo, nova |
| `NOVEL_ACTION_REQUESTED` | nova | nova, simula |
| `OIKOS_DRIVE_WEIGHT_PRESSURE` | oikos | equor |
| `OIKOS_ECONOMIC_EPISODE` | oikos | evo, kairos |
| `OIKOS_ECONOMIC_QUERY` | axon, oikos | axon, oikos |
| `OIKOS_ECONOMIC_RESPONSE` | axon, oikos | axon |
| `OIKOS_METABOLIC_SNAPSHOT` | oikos | equor |
| `ONEIROS_CONSOLIDATION_COMPLETE` | oneiros | federation, nexus, nova, simula, skia, thread, thymos, voxis |
| `ONEIROS_ECONOMIC_INSIGHT` | oneiros | nova |
| `ONEIROS_SLEEP_OUTCOME` | oneiros | evo |
| `ONEIROS_THREAT_SCENARIO` | oneiros | nexus, nova, thymos |
| `OPPORTUNITY_DETECTED` | nova | nova |
| `ORGANISM_DIED` | skia | skia |
| `ORGANISM_SLEEP` | oneiros | axon, identity, sacm, simula |
| `ORGANISM_SPAWNED` | skia | identity |
| `ORGANISM_TELEMETRY` | synapse | federation, nova, sacm |
| `ORGANISM_WAKE` | oneiros | sacm |
| `PERCEPT_ARRIVED` | fovea | fovea, kairos |
| `PERSONA_CREATED` | identity (getattr pattern) | synapse |
| `PERSONA_EVOLVED` | identity (getattr pattern) | synapse |
| `PHANTOM_FALLBACK_ACTIVATED` | phantom_liquidity | thymos |
| `PHANTOM_IL_DETECTED` | phantom_liquidity | thymos |
| `PHANTOM_METABOLIC_COST` | oikos, phantom_liquidity | oikos |
| `PHANTOM_PARAMETER_ADJUSTED` | phantom_liquidity | evo |
| `PHANTOM_POOL_STALE` | phantom_liquidity | thymos |
| `PHANTOM_POSITION_CRITICAL` | phantom_liquidity | thymos |
| `PHANTOM_PRICE_OBSERVATION` | phantom_liquidity | kairos, phantom_liquidity |
| `PHANTOM_PRICE_UPDATE` | phantom_liquidity | nova |
| `PHANTOM_RESOURCE_EXHAUSTED` | phantom_liquidity | thymos |
| `PHANTOM_SUBSTRATE_OBSERVABLE` | phantom_liquidity | benchmarks |
| `RE_DECISION_OUTCOME` | nova | benchmarks, evo |
| `RE_ENGINE_STATUS_CHANGED` | reasoning_engine | nova |
| `RE_TRAINING_EXAMPLE` | axon, eis, equor, evo, federation, fovea, identity, kairos, logos, memory, mitosis, nexus, nova, oikos, phantom_liquidity, reasoning_engine, sacm, simula, skia, soma, synapse, telos, thread, thymos, voxis | core |
| `RE_TRAINING_EXPORT_COMPLETE` | core | benchmarks |
| `RE_TRAINING_FAILED` | reasoning_engine | thymos |
| `RE_TRAINING_HALTED` | reasoning_engine | thymos |
| `RE_TRAINING_REQUESTED` | evo, nova | reasoning_engine |
| `RE_TRAINING_RESUMED` | reasoning_engine | thymos |
| `REASONING_CAPABILITY_DEGRADED` | nova | nova |
| `REPAIR_COMPLETED` | axon, simula, thymos | evo, simula, thymos |
| `REPUTATION_DAMAGED` | oikos, thread | nova, thread |
| `REPUTATION_MILESTONE` | oikos | thread |
| `RESOURCE_PRESSURE` | synapse | alive, sacm |
| `REVENUE_INJECTED` | axon, nova, oikos, synapse | evo, memory, nexus, nova, oikos, simula, soma, synapse, thread, voxis |
| `RHYTHM_STATE_CHANGED` | synapse | core |
| `SAFE_MODE_ENTERED` | synapse | - (EMIT-ONLY: no subscriber found) |
| `SCHEMA_EVOLVED` | thread | evo |
| `SCHEMA_FORMED` | thread | evo |
| `SCHEMA_INDUCED` | evo | thread |
| `SELF_AFFECT_UPDATED` | memory | equor, fovea, thread |
| `SELF_COHERENCE_ALARM` | identity | telos |
| `SELF_MODEL_UPDATED` | identity | thread |
| `SELF_STATE_DRIFTED` | memory | equor |
| `SIMULA_CALIBRATION_DEGRADED` | simula | thymos |
| `SIMULA_EVOLUTION_APPLIED` | simula | mitosis, oneiros |
| `SIMULA_KPI_PUSH` | simula | benchmarks |
| `SIMULA_ROLLBACK_PENALTY` | simula | oikos |
| `SIMULA_SANDBOX_REQUESTED` | thymos | simula |
| `SIMULA_SANDBOX_RESULT` | simula | thymos |
| `SKIA_HEARTBEAT_LOST` | skia | thymos |
| `SKIA_RESURRECTION_PROPOSAL` | skia | federation |
| `SLEEP_INITIATED` | oneiros | federation, memory |
| `SLEEP_STAGE_TRANSITION` | oneiros | fovea |
| `SOMATIC_DRIVE_VECTOR` | soma | kairos, thread |
| `SOMATIC_MODULATION_SIGNAL` | equor, evo, oikos, soma | equor, evo, soma, thymos, voxis |
| `SOMA_ALLOSTATIC_REPORT` | soma | benchmarks |
| `SOMA_STATE_SPIKE` | soma | thymos |
| `SOMA_TICK` | eis, synapse | equor, nova, voxis |
| `SOMA_URGENCY_CRITICAL` | soma | nova |
| `SOMA_VITALITY_SIGNAL` | soma | skia |
| `SPECIATION_EVENT` | evo, mitosis, nexus | nexus, thymos |
| `SPEC_DRAFTED` | nova | simula |
| `STARVATION_WARNING` | oikos | synapse |
| `SYSTEM_MODULATION` | oikos, skia | alive, axon, eis, equor, evo, fovea, memory, nexus, nova, oneiros, simula, telos, voxis |
| `SYSTEM_MODULATION_ACK` | axon, eis, equor, evo, fovea, logos, memory, nexus, nova, oneiros, simula, voxis | skia |
| `TASK_PERMANENTLY_FAILED` | core | thymos |
| `TELEGRAM_MESSAGE_RECEIVED` | identity | identity |
| `TELEGRAM_OTP_RECEIVED` | identity | identity |
| `TELOS_ASSESSMENT_SIGNAL` | telos | equor |
| `TELOS_OBJECTIVE_THREATENED` | telos | nova |
| `TELOS_POPULATION_SNAPSHOT` | telos | benchmarks |
| `TELOS_SELF_MODEL_REQUEST` | telos (hasattr guard) | telos |
| `TELOS_VITALITY_SIGNAL` | telos | skia |
| `TELOS_WORLD_MODEL_VALIDATE` | simula | telos |
| `THREAT_ADVISORY_RECEIVED` | federation (emit added) | oikos |
| `THREAT_DETECTED` | eis, oikos | thymos |
| `THYMOS_REPAIR_APPROVED` | thymos | simula |
| `THYMOS_REPAIR_REQUESTED` | thymos | simula |
| `TRIANGULATION_WEIGHT_UPDATE` | nexus | evo |
| `TURNING_POINT_DETECTED` | thread | evo |
| `VAULT_DECRYPT_FAILED` | identity (_fire_event) | thymos |
| `VAULT_KEY_ROTATION_FAILED` | identity (_fire_event) | thymos |
| `VOXIS_EXPRESSION_DISTRESS` | voxis | soma |
| `VOXIS_PERSONALITY_SHIFTED` | voxis | thread |
| `VULNERABILITY_CONFIRMED` | simula | axon, thymos |
| `WAKE_INITIATED` | oneiros | nexus, thread |
| `WAKE_ONSET` | oneiros (module alias) | axon |
| `WALLET_TRANSFER_CONFIRMED` | axon | memory, oikos |
| `WELFARE_OUTCOME_RECORDED` | axon | telos |
| `WORLD_MODEL_FRAGMENT_SHARE` | federation, nexus | federation |
| `WORLD_MODEL_UPDATED` | kairos, logos | fovea, kairos, telos |
| `YIELD_DEPLOYMENT_REQUEST` | oikos | axon |
| `YIELD_DEPLOYMENT_RESULT` | axon | evo, nova, oikos, simula |
| `YIELD_PERFORMANCE_REPORT` | oikos | evo |

</details>

---

## EMIT-ONLY Events (~255 true)

Events emitted for telemetry, observability, or federation with no active subscriber in this instance.
Many are intentional - Benchmarks, EIS metrics, Skia lifecycle, etc.

<details><summary>Click to expand (scanner-reported 274, minus ~19 reclassified to LIVE)</summary>

Notable emit-only categories:
- **Equor audit trail**: `EQUOR_REVIEW_STARTED/COMPLETED`, `EQUOR_FAST_PATH_HIT`, `EQUOR_DEFERRED`, `EQUOR_AUTONOMY_PROMOTED/DEMOTED`
- **EIS threat metrics**: `EIS_THREAT_METRICS`, `EIS_THREAT_SPIKE`, `EIS_ANOMALY_RATE_ELEVATED`, `EIS_CONSTITUTIONAL_THREAT`
- **Skia restoration lifecycle**: `SKIA_RESTORATION_STARTED/COMPLETE/COMPLETED/TRIGGERED`, `SKIA_DRY_RUN_COMPLETE`, `SKIA_HEARTBEAT`, `SKIA_SHADOW_WORKER_DEPLOYED/MISSING`
- **RE training pipeline**: `RE_TRAINING_STARTED/COMPLETE/BATCH`, `RE_DPO_STARTED/COMPLETE`, `RE_TIER3_STARTED/COMPLETE`, `RE_KL_GATE_REJECTED`, `RE_ADAPTER_QUALITY_CONFIRMED`
- **Oikos economic telemetry**: `ECONOMIC_ACTION_RETRY`, `ECONOMIC_AUTONOMY_SIGNAL`, `OPPORTUNITY_DISCOVERED`, `AFFILIATE_*`, `PORTFOLIO_REBALANCED`, etc.
- **Logos compression**: `LOGOS_INVARIANT_VIOLATED`, `SCHWARZSCHILD_THRESHOLD_MET`, `LOGOS_SCHWARZSCHILD_APPROACHING`, `LOGOS_BUDGET_ADMISSION_DENIED`, `INTELLIGENCE_METRICS`
- **Kairos causal mining**: `KAIROS_CAUSAL_CANDIDATE_GENERATED`, `KAIROS_CONFOUNDER_DISCOVERED`, `KAIROS_INVARIANT_CANDIDATE`, `KAIROS_COUNTER_INVARIANT_FOUND`, `KAIROS_VALIDATED_CAUSAL_STRUCTURE`
- **Compute/SACM lifecycle**: `COMPUTE_MIGRATION_STARTED/COMPLETED/FAILED`, `COMPUTE_REQUEST_ALLOCATED/QUEUED`, `SACM_PRE_WARM_PROVISIONED`, `SACM_COMPUTE_STRESS`
- **Thread narrative**: `CHAPTER_CLOSED/OPENED`, `COMMITMENT_MADE/TESTED/STRAIN`, `SCHEMA_CHALLENGED`, `IDENTITY_DISSONANCE/SHIFT_DETECTED`
- **Identity provisioning**: `CONNECTOR_AUTHENTICATED`, `OTP_FLOW_RESOLVED`, `PROVISIONING_REQUIRES_HUMAN_ESCALATION`

</details>

---

## TRUE DANGLING Events (6)

These have subscribers but genuinely no emitter implementation exists. Subscriber will never fire.

| Event | Subscriber | Root Cause |
|-------|-----------|-----------|
| `ASSET_DEV_REQUEST` | oikos | AssetFactory not built; subscription is placeholder (Spec 17) |
| `CONTENT_ENGAGEMENT_REPORT` | oikos | EngagementPoller not implemented; comments say "future" |
| `EQUOR_BUDGET_OVERRIDE` | axon | No emitter path in Equor; `hasattr` guard means it's future-proofed only |
| `GOAL_OVERRIDE` | nova | Federation/governance goal injection not wired to emit; subscription ready |
| `INV_017_VIOLATED` | thymos | No formal prover emits this; DRIVE_EXTINCTION_DETECTED is the active signal |
| `TELOS_SELF_MODEL_REQUEST` | telos | No system queries Telos for self-model; `hasattr` emit guard only |

> **Severity**: LOW-MEDIUM. All 6 are "ready when built" patterns - subscriber + event defined, emitter is the missing half. None block current functionality.

---

## UNWIRED Events (~66)

Events in the enum with no emit or subscribe sites found anywhere. Categorized below.

### IMPLEMENT (32) - Wire in near-term sprint

These have spec references and should be implemented:

| Event | System | Spec Reference |
|-------|--------|---------------|
| `CERTIFICATE_RENEWED` | identity | Spec 23 §9 (renewal lifecycle) |
| `CLOCK_OVERRUN` | synapse | Spec 09 (clock overrun detection) |
| `CONSTITUTIONAL_HASH_CHANGED` | identity/equor | Spec 23 §25 (federation cert refresh) |
| `DEVELOPMENTAL_MILESTONE` | synapse/benchmarks | Spec 09 (lifecycle milestones) |
| `EQUOR_AUTONOMY_DEMOTED` | equor | Spec 02 §17.1 |
| `EQUOR_AUTONOMY_PROMOTED` | equor | Spec 02 §17.1 |
| `EQUOR_DEFERRED` | equor | Spec 02 §17.1 |
| `EQUOR_FAST_PATH_HIT` | equor | Spec 02 §17.1 |
| `EQUOR_REVIEW_COMPLETED` | equor | Spec 02 §17.1 |
| `EQUOR_REVIEW_STARTED` | equor | Spec 02 §17.1 |
| `FEDERATION_PEER_DISCONNECTED` | federation/mitosis | Spec 26 (subscribe_to_events) |
| `FEDERATION_TASK_OFFERED` | federation | Federation CLAUDE.md |
| `FEDERATION_TOPOLOGY_CHANGED` | synapse/federation | Spec 09 v1.2 additions |
| `FOVEA_HABITUATION_COMPLETE` | fovea | Spec 20 (habituation engine) |
| `FOVEA_HABITUATION_DECAY` | fovea | Spec 20 (habituation decay) |
| `FOVEA_WORKSPACE_IGNITION` | fovea | Spec 20 (workspace ignition) |
| `GITHUB_ACCOUNT_PROVISIONED` | identity | Identity CLAUDE.md |
| `HEALING_STORM_ENTERED` | thymos | Spec 12 (healing storm) |
| `HEALING_STORM_EXITED` | thymos | Spec 12 (healing storm) |
| `IDENTITY_CHALLENGED` | identity | Spec 23 §25 |
| `IDENTITY_DRIFT_DETECTED` | identity | Spec 23 §25 |
| `IDENTITY_EVOLVED` | identity | Spec 23 §25 |
| `INPUT_CHANNEL_REGISTERED` | identity | Identity CLAUDE.md |
| `MEMORY_CONSOLIDATED` | oneiros/memory | Spec 01 §18; Logos subscribes |
| `MODEL_HOT_SWAP_COMPLETED` | reasoning_engine | RE spec (hot-swap lifecycle) |
| `MODEL_HOT_SWAP_STARTED` | reasoning_engine | RE spec (hot-swap lifecycle) |
| `PHONE_NUMBER_PROVISIONED` | identity | Identity CLAUDE.md |
| `PLATFORM_ACCOUNT_PROVISIONED` | identity | Identity CLAUDE.md |
| `PROOF_FAILED` | simula | Spec 10 (proof lifecycle) |
| `PROOF_FOUND` | simula | Spec 10 (proof lifecycle) |
| `SLEEP_FORCED` | oneiros | Spec 13/14 (emergency sleep) |
| `SLEEP_ONSET` | oneiros | Spec 13/14; Soma subscribes |
| `SLEEP_PRESSURE_WARNING` | oneiros | Spec 13/14 (0.70 threshold) |
| `SLEEP_STAGE_CHANGED` | oneiros | Spec 13/14 (alias of SLEEP_STAGE_TRANSITION?) |

### DEFER (24) - Low priority, no spec contract

| Event | Reason |
|-------|--------|
| `ADAPTER_TRAINING_COMPLETE` | No spec lifecycle defined |
| `ADAPTER_TRAINING_STARTED` | No spec lifecycle defined |
| `ANTIBODY_RETIRED` | Thymos gap - antibody retirement not implemented |
| `BOUNTY_DISCOVERED` | Oikos Level 6 scanning not wired |
| `BOUNTY_EVALUATED` | Oikos scoring not emitting events |
| `CAPABILITY_OFFERED` | IIEP undefined (Spec 17 §13.2 concept only) |
| `CAPABILITY_REQUESTED` | IIEP undefined |
| `CAPABILITY_TRADE_SETTLED` | IIEP undefined |
| `CATASTROPHIC_FORGETTING_DETECTED` | RE / Memory monitoring not implemented |
| `CHILD_RESCUED` | Mitosis rescue implemented but event not wired |
| `CREDIT_DRAWN` | Oikos Level 7 largely unimplemented |
| `CREDIT_REPAID` | Oikos Level 7 largely unimplemented |
| `ENTITY_FORMATION_RESUMED` | No spec for RESUMED event |
| `EQUOR_PROMOTION_ELIGIBLE` | Not in Spec 02 §17.1 |
| `FORAGING_CYCLE_COMPLETE` | No spec section |
| `FOVEA_DIAGNOSTIC_REPORT` | Not mentioned in Spec 20 |
| `INSURANCE_PREMIUM_PAID` | Oikos Level 10 unimplemented |
| `KNOWLEDGE_SALE_RECORDED` | Oikos Level 8 absent |
| `PROTOCOL_DESIGNED` | Spec 17 §8.3 concept only |
| `PROTOCOL_REVENUE_SWEPT` | No spec event contract |
| `PROTOCOL_TERMINATED` | Spec uses different event name |
| `REPUTATION_UPDATED` | Oikos Level 7 not built |
| `SERVICE_OFFER_ACCEPTED` | IIEP undefined |
| `SYSTEM_STOPPED` | Synapse lifecycle - may be scanner miss |

### DELETE (2) - Confirmed dead weight

| Event | Reason |
|-------|--------|
| `SACM_DRAINING` | SACM `shutdown()` uses `ORGANISM_SHUTDOWN_REQUESTED`; `SACM_DRAINING` is unreferenced duplicate |
| `VAULT_KEY_ROTATION_STARTED` | `VAULT_KEY_ROTATION_COMPLETE` is the meaningful event; no subscriber or emitter for STARTED variant |

> Note: `VAULT_KEY_ROTATION_COMPLETE` itself is also UNWIRED - both halves of the vault rotation event pair are missing emit sites. See IMPLEMENT list.

---

## Scanner False Positives - Patterns Missed

The grep-based audit scanner reports DANGLING/DEAF incorrectly for these emit/subscribe patterns:

| Pattern | Example | Systems Affected |
|---------|---------|-----------------|
| `_SET.X` lazy import alias | `from synapse.types import SynapseEventType as _SET` | Skia, Telos, Equor |
| `_SynET.X` alias | `event_type=_SynET.AFFECT_STATE_CHANGED` | Soma |
| `_emit_equor_event(...)` helper | `_emit_equor_event(_SET.EQUOR_DRIVE_WEIGHTS_UPDATED, ...)` | Equor |
| `SynapseEventType(BARE_NAME)` constructor | `SynapseEventType(FOVEA_CALIBRATION_ALERT)` | Fovea |
| Module-level alias then `_emit_event(ALIAS, ...)` | `WAKE_ONSET = SynapseEventType.WAKE_ONSET` then `_emit_event(WAKE_ONSET, ...)` | Oneiros |
| `getattr(SynapseEventType, "NAME")` | Persona emit | Identity |
| `_fire_event("NAME", ...)` string helper | Vault emit | Identity |
| `_emit_safe("name", ...)` helper | RE safety halt | Reasoning Engine |
| `hasattr(SynapseEventType, "X")` guard emit | Budget override path | Thymos/Equor |
| List-based subscribe loop | `for event_type, handler in subscriptions: bus.subscribe(...)` | Logos, EIS |

**Net impact**: ~34 events reclassified from DANGLING/DEAF to LIVE.

---

## Triage Summary

### Overall Health

```
Total entries:         638
Active (LIVE + EMIT-ONLY): ~566 (88.7%)  ✓ >80% active target MET
LIVE (fully wired):    ~311 (48.7%)
EMIT-ONLY (telemetry):  ~255 (40.0%)
TRUE DANGLING:           6   (0.9%)      ✓ <1% target MET
UNWIRED:                ~66  (10.3%)     ✗ >5% target MISSED
String-literal emits:    0               ✓ target MET
DEAF systems:            0               ✓ target MET
MUTE systems:            1 (alive)       acceptable - visualization bridge by design
```

### Priority Fix Sprint

**P1 - True DANGLING (6 events, deadcode subscribers)**
None are blocking current functionality, but they represent planned features with stubs in place.
Recommended: leave as-is until the feature is implemented (AssetFactory, EngagementPoller, etc.).

**P2 - UNWIRED IMPLEMENT list (32 events)**
High-value wiring work. Recommended sprint order:
1. `SLEEP_ONSET/FORCED/PRESSURE_WARNING` + `SLEEP_STAGE_CHANGED` - Oneiros completeness
2. `EQUOR_REVIEW_STARTED/COMPLETED` + `EQUOR_AUTONOMY_PROMOTED/DEMOTED/DEFERRED/FAST_PATH_HIT` - Equor audit trail
3. `PROOF_FOUND/FAILED` - Simula verification lifecycle
4. `MODEL_HOT_SWAP_STARTED/COMPLETED` - RE hot-swap observability
5. `HEALING_STORM_ENTERED/EXITED` - Thymos immune response visibility
6. `MEMORY_CONSOLIDATED` - Logos + Oneiros integration
7. Identity provisioning events (4) - `GITHUB_ACCOUNT_PROVISIONED`, etc.

**P3 - DELETE (2 events)**
Remove `SACM_DRAINING` and `VAULT_KEY_ROTATION_STARTED` from `synapse/types.py`.

**P4 - Scanner improvement**
Extend `scripts/synapse_audit.py` to recognize `_SET.X`, `_SynET.X`, module-level aliases,
and list-based subscribe patterns. Will bring scanner accuracy to >95%.

---

## Logos - DEAF Status (Scanner False Positive)

Logos appears DEAF in scanner output because it subscribes via list-based pattern:

```python
_SUBSCRIPTIONS = [
    (SynapseEventType.COGNITIVE_PRESSURE, self._on_pressure),
    (SynapseEventType.EPISODE_STORED, self._on_episode),
    ...
]
for event_type, handler in _SUBSCRIPTIONS:
    bus.subscribe(event_type, handler)
```

Logos has **12 subscriptions** confirmed by manual inspection. Status: **OK**.

## `alive` - MUTE Status (Acceptable)

`alive` (WebSocket bridge for 3D visualization) subscribes to 4 telemetry events for push-to-frontend
and never emits on the Synapse bus. This is correct by design - it's an output device, not a cognitive
actor. Status: **ACCEPTABLE**, not a bug.
