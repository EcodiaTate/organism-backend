# EcodiaOS Synapse Event Triage - 2026-03-09

> **Scope**: Every UNWIRED enum entry (218 events - in enum, neither emitted nor subscribed) and every DANGLING subscriber (86 events - subscribed but nothing emits via enum). Each classified as **IMPLEMENT**, **DELETE**, or **DEFER** with spec evidence.

---

## 1. UNWIRED Events (in enum, neither emitted nor subscribed)

### RE Training Pipeline

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| ABLATION_COMPLETE | IMPLEMENT | Spec'd in Round 5 prompts for monthly ablation studies | Round 5 Prompts (ablation study) | RE, Benchmarks |
| ABLATION_STARTED | IMPLEMENT | Spec'd in Round 5 prompts for ablation study lifecycle | Round 5 Prompts (ablation study) | RE, Benchmarks |
| ADAPTER_SHARE_OFFER | IMPLEMENT | Year 2 cross-instance adapter merging; fully designed | Round 5/6 Prompts, SPECIATION_GAPS_FINAL | RE, Federation |
| ADAPTER_SHARE_REQUEST | IMPLEMENT | Request-reply pair for adapter path fetching between instances | Round 5 Prompts (adapter sharing) | RE, Federation |
| ADAPTER_SHARE_RESPONSE | IMPLEMENT | Response side of adapter sharing request-reply | Round 5 Prompts (adapter sharing) | RE, Federation |
| ADAPTER_TRAINING_COMPLETE | DEFER | No spec section defines payload or lifecycle | None found | RE |
| ADAPTER_TRAINING_STARTED | DEFER | No spec section defines payload or lifecycle | None found | RE |
| RE_ADAPTER_QUALITY_CONFIRMED | IMPLEMENT | Post-deploy quality check, emitted when post_rate > pre_rate × 1.05 | RE CLAUDE.md (deployment quality) | RE, Benchmarks |
| RE_DPO_COMPLETE | IMPLEMENT | Fully specified with payload schema | Round 4 Prompts (DPO training) | RE |
| RE_DPO_STARTED | IMPLEMENT | Fully specified with payload schema | Round 4 Prompts (DPO training) | RE |
| RE_ENGINE_STATUS_CHANGED | IMPLEMENT | Circuit breaker state transitions; Benchmarks subscribes for llm_dependency KPI | Round 1 Prompts, RE CLAUDE.md | RE, Benchmarks |
| RE_KL_GATE_REJECTED | IMPLEMENT | Emitted when STABLE KL gate blocks adapter update | Round 3 Prompts (STABLE KL gate) | RE, Benchmarks |
| RE_TIER3_COMPLETE | IMPLEMENT | Quarterly retrain completion with payload | Round 4 Prompts (quarterly retrain) | RE |
| RE_TIER3_STARTED | IMPLEMENT | Quarterly retrain start with payload | Round 4 Prompts (quarterly retrain) | RE |
| RE_TRAINING_COMPLETE | IMPLEMENT | Training lifecycle event with payload schema | Round 2 Prompts (training lifecycle) | RE, Benchmarks |
| RE_TRAINING_FAILED | IMPLEMENT | Training failure event with payload | Round 2/3 Prompts (training lifecycle) | RE, Thymos |
| RE_TRAINING_HALTED | IMPLEMENT | Tier 2 kill switch event | Round 3 Prompts (kill switch) | RE, Thymos |
| RE_TRAINING_STARTED | IMPLEMENT | Training lifecycle start with payload | Round 2 Prompts (training lifecycle) | RE, Benchmarks |
| INV_017_VIOLATED | IMPLEMENT | Drive extinction detection; constitutional invariant | Round 3 Prompts (constitutional invariant) | RE, Equor, Thymos |
| RED_TEAM_EVALUATION_COMPLETE | IMPLEMENT | Monthly red-team evaluation results | Round 3/4 Prompts | RE, Benchmarks |

### Identity / Provisioning

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| ACCOUNT_PROVISIONING_FAILED | IMPLEMENT | Provisioning pipeline needs failure signaling | Identity CLAUDE.md (provisioning) | Identity, Thymos |
| AFFECT_STATE_CHANGED | IMPLEMENT | Needed for Voxis affect coloring; emitted by Thymos/Soma | synapse/types.py comment | Soma/Thymos, Voxis |
| CONNECTOR_REVOKED | IMPLEMENT | Event table lists payload and consumers (Nova, Oikos) | Spec 23 §15 event table | Identity, Nova, Oikos |
| CONSTITUTIONAL_HASH_CHANGED | IMPLEMENT | Federation needs it for mTLS cert refresh | Spec 23 §25 | Identity, Federation |
| GITHUB_ACCOUNT_PROVISIONED | IMPLEMENT | Emission from GitHub provisioning flow documented | Identity CLAUDE.md | Identity |
| IDENTITY_CHALLENGED | IMPLEMENT | Event table specifies payload and semantics | Spec 23 §25 | Identity, Federation |
| IDENTITY_CRISIS | IMPLEMENT | Emitted on fingerprint shift ≥ 0.50; Thymos subscribes | Spec 15 §4.6.3 | Thread, Thymos |
| IDENTITY_DISSONANCE | IMPLEMENT | Emitted from SelfEvidencingLoop on significant surprise | Spec 15 §4.6.3 | Thread |
| IDENTITY_DRIFT_DETECTED | IMPLEMENT | Emitted when coherence < 0.7; Evo subscribes for diversity metrics | Spec 23 §25 | Identity, Evo |
| IDENTITY_EVOLVED | IMPLEMENT | Event table specifies payload (old_hash, new_hash, generation) | Spec 23 §25 | Identity, Federation |
| IDENTITY_SHIFT_DETECTED | IMPLEMENT | Emitted on fingerprint shift 0.25–0.49 | Spec 15 §4.6.3 | Thread |
| IDENTITY_VERIFIED | IMPLEMENT | Federation uses for membership confirmation | Spec 23 §25 | Identity, Federation |
| IMMUNE_CYCLE_COMPLETE | IMPLEMENT | AxonReactiveAdapter subscribes to it | Spec 06 (AxonReactiveAdapter) | Thymos, Axon |
| PHONE_NUMBER_PROVISIONED | IMPLEMENT | Emission from Twilio provisioning documented | Identity CLAUDE.md | Identity |
| PLATFORM_ACCOUNT_PROVISIONED | IMPLEMENT | Emission from generic platform provisioning documented | Identity CLAUDE.md | Identity |
| PROVISIONING_REQUIRES_HUMAN_ESCALATION | IMPLEMENT | Emitted from CertificateManager documented | GAP_REMEDIATION_PROMPTS, Identity CLAUDE.md | Identity, Equor |

### Federation Tasks / Collaboration

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| FEDERATION_ASSISTANCE_DECLINED | IMPLEMENT | Event table specifies payload and semantics | Spec 11 event table | Federation |
| FEDERATION_BOUNTY_SPLIT | IMPLEMENT | Listed as emitted event; code in bounty_splitting.py | Federation CLAUDE.md | Federation, Oikos |
| FEDERATION_BROADCAST | IMPLEMENT | VitalityCoordinator emits it; added to SynapseEventType | Spec 29 (VitalityCoordinator) | Skia, Federation |
| FEDERATION_KNOWLEDGE_SHARED | IMPLEMENT | Event table specifies payload with novelty_score | Spec 11 event table | Federation, Benchmarks |
| FEDERATION_LINK_DROPPED | IMPLEMENT | Emitted in withdraw_link() and starvation suspension | Spec 11 event table | Federation |
| FEDERATION_LINK_ESTABLISHED | IMPLEMENT | Emitted after mTLS handshake + Equor PERMIT | Spec 11 event table | Federation, Telos |
| FEDERATION_PEER_BLACKLISTED | IMPLEMENT | Mitosis CLAUDE.md documents emission for fleet blacklist | Spec 26 (Mitosis blacklist) | Mitosis, Federation |
| FEDERATION_PEER_DISCONNECTED | IMPLEMENT | Mitosis spec lists it as subscribed event with handler | Spec 26 (subscribe_to_events) | Federation, Mitosis |
| FEDERATION_TASK_ACCEPTED | IMPLEMENT | Code in task_delegation.py:205 (string literal - needs enum fix) | Federation CLAUDE.md | Federation |
| FEDERATION_TASK_COMPLETED | IMPLEMENT | Code in task_delegation.py:276 (string literal - needs enum fix) | Federation CLAUDE.md | Federation |
| FEDERATION_TASK_DECLINED | IMPLEMENT | Code in task_delegation.py:237 (string literal - needs enum fix) | Federation CLAUDE.md | Federation |
| FEDERATION_TASK_OFFERED | IMPLEMENT | Listed as emitted event | Federation CLAUDE.md | Federation |
| FEDERATION_TASK_PAYMENT | IMPLEMENT | Code in task_delegation.py:368 (string literal - needs enum fix) | Federation CLAUDE.md | Federation, Oikos |
| FEDERATION_TOPOLOGY_CHANGED | IMPLEMENT | Listed as new SynapseEventType entry | Spec 09 (v1.2 additions) | Synapse, Federation |
| FEDERATION_TRUST_UPDATED | IMPLEMENT | Thymos emits on speciation event quarantine | Spec 11 event table | Federation, Thymos |
| FEDERATION_WORK_ROUTED | IMPLEMENT | Code in work_router.py:156 (string literal - needs enum fix) | Federation CLAUDE.md | Federation |
| FORAGING_CYCLE_COMPLETE | DEFER | No spec section describes payload or lifecycle | None found | Oikos |

### Skia / Vitality / Degradation

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| COMPUTE_ARBITRAGE_DETECTED | DEFER | Not mentioned in any spec | None found | SACM |
| COMPUTE_CAPACITY_EXHAUSTED | IMPLEMENT | Emitted at utilisation ≥ 95% | Spec 27 §12 event table | SACM, Soma, Oikos |
| COMPUTE_FEDERATION_OFFLOADED | IMPLEMENT | Emitted on peer offload | Spec 27 §12 event table | SACM, Federation |
| COMPUTE_MIGRATION_COMPLETED | IMPLEMENT | MigrationExecutor lifecycle documented | SACM CLAUDE.md | SACM |
| COMPUTE_MIGRATION_FAILED | IMPLEMENT | Rollback emits this event | SACM CLAUDE.md | SACM |
| COMPUTE_MIGRATION_STARTED | IMPLEMENT | Migration lifecycle documented | SACM CLAUDE.md | SACM |
| COMPUTE_REQUEST_ALLOCATED | IMPLEMENT | Event table + integration table specify payload | Spec 27 §12 | SACM, Soma |
| COMPUTE_REQUEST_DENIED | IMPLEMENT | Thymos subscribes | Spec 27 §12 | SACM, Thymos |
| COMPUTE_REQUEST_QUEUED | IMPLEMENT | Event table specifies payload | Spec 27 §12 | SACM |
| DEGRADATION_OVERRIDE | IMPLEMENT | Added to SynapseEventType (was causing ValueError) | Spec 29 (VitalityCoordinator) | Skia |
| DEPENDENCY_INSTALLED | IMPLEMENT | Emitted from HotDeployment; Thymos should track | Simula CLAUDE.md | Simula, Thymos |
| MEMORY_CONSOLIDATED | IMPLEMENT | Logos subscribes for world model coverage metric | Spec 21 (Logos subscriptions) | Oneiros, Logos |
| ORGANISM_RESURRECTED | IMPLEMENT | Emitted from VitalityCoordinator.resurrect() | Spec 09 v1.2, Spec 29 §21.5.5 | Skia |
| ORGANISM_SHUTDOWN_REQUESTED | IMPLEMENT | Emitted by SACM MigrationExecutor after new instance confirmed | synapse/types.py, sacm/migrator.py | SACM |
| SACM_DRAINING | IMPLEMENT | Emitted from SACMService.shutdown() | Spec 27 (shutdown drain) | SACM |
| SKIA_DRY_RUN_COMPLETE | IMPLEMENT | dry_run_restoration() emits this | Spec 29 (dry-run restoration) | Skia |
| SKIA_HEARTBEAT | IMPLEMENT | New emission from _worker_heartbeat_loop() | Spec 29 (Phase 2 additions) | Skia |
| SKIA_RESTORATION_COMPLETE | IMPLEMENT | Listed in Phase 2 additions; distinct from COMPLETED variant | Spec 29 (Phase 2 additions) | Skia |
| SKIA_RESTORATION_COMPLETED | IMPLEMENT | Event table specifies payload (outcome, strategy, state_cid, duration_ms) | Spec 29 §13.2 | Skia |
| SKIA_RESTORATION_STARTED | IMPLEMENT | Listed in Phase 2 additions | Spec 29 (Phase 2 additions) | Skia |
| SKIA_RESTORATION_TRIGGERED | IMPLEMENT | Event table specifies payload | Spec 29 §13.2 | Skia |
| SKIA_SHADOW_WORKER_DEPLOYED | IMPLEMENT | Emitted from _ensure_shadow_worker_loop() | Skia CLAUDE.md | Skia |
| SKIA_SHADOW_WORKER_MISSING | IMPLEMENT | Emitted on shadow worker provisioning failure | Skia CLAUDE.md | Skia |
| VITALITY_FATAL | IMPLEMENT | Emitted during death sequence | Spec 29 §21.5 | Skia |
| VITALITY_REPORT | DEFER | No spec section describes payload or lifecycle | None found | Skia |
| VITALITY_RESTORED | IMPLEMENT | Emitted when fatal threshold recovers during warning | Spec 09 v1.2, Spec 29 §21.5 | Skia |

### Equor

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| EQUOR_AUTONOMY_DEMOTED | IMPLEMENT | Emitted from apply_autonomy_change() and drift-triggered demotions | Spec 02 §17.1 | Equor |
| EQUOR_AUTONOMY_PROMOTED | IMPLEMENT | Emitted when autonomy level increases | Spec 02 §17.1 | Equor |
| EQUOR_DEFERRED | IMPLEMENT | Emitted when verdict is DEFERRED including timeout | Spec 02 §17.1 | Equor |
| EQUOR_ESCALATED_TO_HUMAN | IMPLEMENT | Emitted during HITL suspension | Spec 02 §17.1 | Equor |
| EQUOR_FAST_PATH_HIT | IMPLEMENT | Emitted when review_critical() returns | Spec 02 §17.1 | Equor |
| EQUOR_PROMOTION_ELIGIBLE | DEFER | Not listed in Spec 02 §17.1 emitted events | None found | Equor |
| EQUOR_REVIEW_COMPLETED | IMPLEMENT | Emitted at end of _review_inner() | Spec 02 §17.1 | Equor |
| EQUOR_REVIEW_STARTED | IMPLEMENT | Emitted at start of review() | Spec 02 §17.1 | Equor |
| EQUOR_SAFE_MODE_ENTERED | IMPLEMENT | Emitted on first transition into safe mode | Spec 02 §17.1 | Equor |

### Telos / Drive

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| CARE_COVERAGE_GAP | IMPLEMENT | Emitted when care_coverage_multiplier < 0.8 | Spec 18 §X, §XIII | Telos |
| COHERENCE_COST_ELEVATED | IMPLEMENT | Emitted when incoherence > threshold | Spec 18 §X, §XIII | Telos |
| CONSTITUTIONAL_TOPOLOGY_INTACT | IMPLEMENT | Every 24h routine verification of all four drives | Spec 18 §X, §XIII | Telos |
| HONESTY_VALIDITY_LOW | IMPLEMENT | Emitted when validity_coefficient < 0.8 | Spec 18 §X, §XIII | Telos |
| TELOS_ASSESSMENT_SIGNAL | DEFER | Not listed in Spec 18 emitted or consumed events | None found | Telos |
| TELOS_AUTONOMY_STAGNATING | IMPLEMENT | Emitted when avg AUTONOMY_INSUFFICIENT events/day > 3 over 7d | Spec 18 audit §1 (Task 2) | Telos |
| TELOS_GENOME_EXTRACTED | IMPLEMENT | Emitted when Telos genome extracted for Mitosis inheritance | Spec 18 §XII (SG3) | Telos, Mitosis |
| TELOS_OBJECTIVE_THREATENED | IMPLEMENT | Emitted on 3 consecutive metabolic_efficiency declines | Spec 18 audit §1 (Task 1) | Telos, Nova |
| TELOS_SELF_MODEL_SNAPSHOT | DEFER | Not in Spec 18 emitted events table | None found | Telos |
| TELOS_VITALITY_SIGNAL | DEFER | VitalitySystem hooks flagged as gap (Nova SG6); not formally specified | Spec 05 SG6 (gap) | Telos |
| TELOS_WELFARE_DOMAIN_LEARNED | DEFER | No explicit event by this name in Spec 18 | None found | Telos |
| THYMOS_DRIVE_PRESSURE | DEFER | Not found in Spec 18; Thymos is Spec 12 | None in read specs | Thymos |
| THYMOS_REPAIR_VALIDATED | DEFER | Not found in read specs; Thymos is Spec 12 | None in read specs | Thymos |
| THYMOS_VITALITY_SIGNAL | DEFER | VitalitySystem flagged as undefined (Evo SG8) | Spec 07 audit SG8 | Thymos |

### Narrative / Thread

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| BELIEF_CONSOLIDATED | IMPLEMENT | Memory emits after consolidate(); Thread/Evo/Nexus should subscribe | Spec 01 §18; Spec 07 | Memory, Evo, Thread, Nexus |
| CHAPTER_CLOSED | IMPLEMENT | ChapterDetector closes chapter; already emitted per gap closure | Spec 15 §14 | Thread |
| CHAPTER_OPENED | IMPLEMENT | Emitted at chapter boundary | Spec 15 §14 | Thread |
| COMMITMENT_MADE | IMPLEMENT | Emitted when constitutional commitments seeded at birth | Spec 15 | Thread |
| COMMITMENT_STRAIN | IMPLEMENT | Emitted when ipse_score < 0.6; triggers Equor review + Oneiros routing | Spec 15 §14 | Thread, Equor, Oneiros |
| COMMITMENT_TESTED | IMPLEMENT | Part of CommitmentKeeper subsystem | Spec 15 | Thread |
| CONVERGENCE_DETECTED | IMPLEMENT | Nexus core event from ConvergenceDetector.compare_fragments | Spec 19 | Nexus, Evo, Kairos, Thread |
| DEVELOPMENTAL_MILESTONE | IMPLEMENT | Synapse emits lifecycle milestones | Spec 09 | Synapse, Benchmarks, Thread |
| DIVERGENCE_PRESSURE | IMPLEMENT | Emitted when instance triangulation weight falls below 0.4 | Spec 19 | Nexus, Evo, Thread |
| DREAM_INSIGHT | IMPLEMENT | REM DreamGenerator produces coherence ≥ 0.70 | Spec 13 §12; Spec 14 | Oneiros, Thread, Evo, Nova |
| FRAGMENT_SHARED | IMPLEMENT | Nexus core event emitted during federation sessions | Spec 19 | Nexus, Federation |
| NARRATIVE_CHAPTER_CLOSED | DELETE | Duplicate/alias of CHAPTER_CLOSED; Spec 09 added separately from Spec 15's canonical name | Spec 09 vs Spec 15 - naming collision | Thread |
| NARRATIVE_COHERENCE_SHIFT | IMPLEMENT | DiachronicCoherenceMonitor detects coherence state change | Spec 15 | Thread, Soma, Nova |
| REPUTATION_SNAPSHOT | DEFER | No spec section defines event contract; Oikos Level 7 unimplemented | None found; Oikos Spec 17 Level 7 | Oikos |
| REPUTATION_UPDATED | DEFER | No spec section defines it; Oikos Level 7 not yet built | None found; Oikos Spec 17 Level 7 | Oikos |
| SCHEMA_CHALLENGED | IMPLEMENT | Part of IdentitySchemaEngine - disconfirming evidence | Spec 15 | Thread, Evo, Oneiros |
| SCHEMA_EVOLVED | IMPLEMENT | Emitted via promote_schema() fire-and-forget | Spec 15 | Thread, Evo, Nexus |
| SCHEMA_FORMED | IMPLEMENT | Emitted by form_schema_from_pattern() | Spec 15 | Thread, Evo |
| SELF_MODIFICATION_PROPOSED | IMPLEMENT | Nova approving gap closure aligned with drives | types.py; links to Spec 10 | Nova, Simula, Equor |
| SPEC_DRAFTED | IMPLEMENT | Nova (via SelfModificationPipeline) drafted a new Spec | types.py; links to Spec 10 | Nova/Simula, Equor |
| TRIANGULATION_WEIGHT_UPDATE | IMPLEMENT | Nexus core event emitted after each federation session | Spec 19 | Nexus, Evo, Federation |
| TURNING_POINT_DETECTED | IMPLEMENT | NarrativeSynthesizer turning point detection | Spec 15 | Thread, Oneiros, Evo |

### Fovea / Perception

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| FOVEA_BACKPRESSURE_WARNING | DEFER | Not mentioned in Spec 20; no spec contract exists | None found | Fovea |
| FOVEA_DIAGNOSTIC_REPORT | DEFER | Not mentioned in Spec 20; no payload defined | None found | Fovea |
| FOVEA_DISHABITUATION | IMPLEMENT | Core Fovea event; already emitted by service.py | Spec 20 (DISHABITUATION) | Fovea, Atune, Thymos |
| FOVEA_HABITUATION_COMPLETE | IMPLEMENT | Part of habituation engine; already emitted | Spec 20 | Fovea, Evo |
| FOVEA_HABITUATION_DECAY | IMPLEMENT | Core event; already emitted | Spec 20 (HABITUATION_DECAY) | Fovea, Evo, Soma |
| FOVEA_WORKSPACE_IGNITION | IMPLEMENT | Emitted when prediction error exceeds ignition threshold | Spec 20 (WORKSPACE_IGNITION) | Fovea, All systems |
| PERCEPT_QUARANTINED | DEFER | EIS describes quarantine as internal action, not a Synapse event | Spec 25 §2.2 (internal only) | EIS |

### EIS (Epistemic Immune System)

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| EIS_ANOMALY_RATE_ELEVATED | IMPLEMENT | Emitted when anomaly rate exceeds 2-sigma Poisson baseline for 30s | Spec 25 §2.3, §22 | EIS, Benchmarks, Soma |
| EIS_CONSTITUTIONAL_THREAT | IMPLEMENT | L9a consistency check when cosine similarity > 0.80 to drive-suppression vector | Spec 25 | EIS, Equor |
| EIS_LAYER_TRIGGERED | DEFER | Not in Spec 25 event table; no consumers defined | None in Spec 25 | EIS |
| EIS_THREAT_METRICS | IMPLEMENT | Emitted every 60s with aggregated threat stats; Benchmarks consumes | Spec 25 §2.3, §22 | EIS, Benchmarks |
| EIS_THREAT_SPIKE | IMPLEMENT | Emitted on 5+ threats in 10min with proportional urgency; Soma consumes | Spec 25 §2.3, §22 | EIS, Soma |

### Sleep / Oneiros

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| EMERGENCY_WAKE | IMPLEMENT | Triggered by Thymos CRITICAL incident interrupting sleep | Spec 13, Spec 14 | Oneiros, All systems |
| ENTITY_FORMATION_FAILED | IMPLEMENT | Part of legal entity formation workflow; Soma signal_buffer maps to "error" | types.py; Axon establish_entity.py | Axon, Soma, Oikos |
| ENTITY_FORMATION_RESUMED | DEFER | No spec describes a RESUMED event or when it fires | None found | Axon |
| ENTITY_FORMATION_STARTED | IMPLEMENT | Part of entity formation pipeline alongside COMPLETED/FAILED | types.py; Axon establish_entity.py | Axon, Soma, Thread |
| SLEEP_FORCED | IMPLEMENT | Emitted when critical pressure threshold (0.95) triggers automatic sleep | Spec 13, Spec 14 | Oneiros, All systems |
| SLEEP_ONSET | IMPLEMENT | Also consumed by Soma (Spec 08) | Spec 13, Spec 14, Spec 08 | Oneiros, Soma, Evo, Fovea |
| SLEEP_PRESSURE_WARNING | IMPLEMENT | Emitted when pressure crosses 0.70 threshold | Spec 13, Spec 14 | Oneiros, Nova, Soma |
| SLEEP_STAGE_CHANGED | IMPLEMENT | Fovea already subscribes as SLEEP_STAGE_TRANSITION; name variant | Spec 13, Spec 14 | Oneiros, Atune, Nova, Axon, Evo, Fovea |

### Economic / Revenue (Oikos)

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| AFFILIATE_MEMBERSHIP_APPLIED | DEFER | No spec section defines this event | None found | Oikos |
| AFFILIATE_PROGRAM_DISCOVERED | DEFER | No spec section defines this event | None found | Oikos |
| AFFILIATE_REVENUE_RECORDED | DEFER | No spec section defines this event | None found | Oikos |
| ALLOCATION_RELEASED | IMPLEMENT | Emitted by ComputeResourceManager.release() | Spec 27 | SACM |
| API_RESELL_PAYMENT_RECEIVED | DEFER | Oikos Level 8 not fully implemented | None found | Oikos |
| API_RESELL_REQUEST_SERVED | DEFER | Oikos Level 8 not fully implemented | None found | Oikos |
| CAPABILITY_GAP_IDENTIFIED | DEFER | Oikos Level 10 (IIEP) unimplemented; schema undefined (gap 8) | Spec 17 gap 8 | Oikos, Federation |
| CAPABILITY_OFFERED | DEFER | IIEP message schema explicitly undefined | Spec 17 §13.2 (concept only) | Oikos, Federation |
| CAPABILITY_REQUESTED | DEFER | IIEP message schema explicitly undefined | Spec 17 §13.2 (concept only) | Oikos, Federation |
| CAPABILITY_TRADE_SETTLED | DEFER | IIEP undefined | Spec 17 §13.2 (concept only) | Oikos, Federation |
| CONTENT_MONETIZATION_MILESTONE | DEFER | No spec section defines this event | None found | Oikos |
| CONTENT_REVENUE_RECORDED | DEFER | No spec section defines this event | None found | Oikos |
| CREDIT_DRAWN | DEFER | Oikos Level 7 largely unimplemented | Spec 17 Level 7 | Oikos |
| CREDIT_REPAID | DEFER | No event contract in spec; Level 7 conceptual only | Spec 17 Level 7 | Oikos |
| CROSS_CHAIN_OPPORTUNITY | DEFER | No spec section defines this event | None found | Oikos |
| GOVERNANCE_VOTE_CAST | DEFER | Governance mechanism explicitly undefined (gap 7) | Spec 17 gap 7 | Equor, Oikos |
| INSURANCE_CLAIM_APPROVED | DEFER | Level 10 mutual insurance pool unimplemented | Spec 17 §13 (concept only) | Oikos, Federation |
| INSURANCE_CLAIM_FILED | DEFER | Same as above | Spec 17 §13 (concept only) | Oikos, Federation |
| INSURANCE_PREMIUM_PAID | DEFER | Config param exists but no implementation | Spec 17 §13 | Oikos, Federation |
| INTELLIGENCE_UPDATE | DEFER | Ambiguous name; no spec defines this event | None found | Unknown |
| KNOWLEDGE_SALE_RECORDED | DEFER | Level 8 ERC-20 and secondary market absent | Spec 17 Level 8 | Oikos |
| PORTFOLIO_REBALANCED | DEFER | No spec section; Oikos uses different event names | None found | Oikos |
| PROTOCOL_DEPLOYED | DEFER | Level 5 blocked by undefined governance (gap 7) | Spec 17 Level 5 | Oikos |
| PROTOCOL_DESIGNED | DEFER | No event for design phase in spec | Spec 17 §8.3 (concept only) | Oikos |
| PROTOCOL_REVENUE_SWEPT | DEFER | No spec section; no event contract | None found | Oikos |
| PROTOCOL_TERMINATED | DEFER | Spec uses OIKOS_ASSET_TERMINATED instead | Spec 17 (different event name) | Oikos |
| SERVICE_OFFER_ACCEPTED | DEFER | IIEP-related; undefined (gap 8) | Spec 17 gap 8 | Oikos, Federation |
| SERVICE_OFFER_DRAFTED | DEFER | IIEP-related; undefined (gap 8) | Spec 17 gap 8 | Oikos, Federation |
| SOCIAL_GRAPH_UPDATED | DEFER | No spec section defines this event | None found | Unknown |
| TREASURY_REBALANCED | DEFER | No spec event for rebalancing | None found | Oikos |
| YIELD_REINVESTED | DEFER | Oikos uses different event names for yield ops | None found | Oikos |

### Phantom / DeFi

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| PHANTOM_FALLBACK_ACTIVATED | IMPLEMENT | Event table specifies payload; marked as implemented | Spec 28 §22 event table | Phantom |
| PHANTOM_IL_DETECTED | IMPLEMENT | Emitted in maintenance_cycle() | Spec 28 §22 event table | Phantom, Thymos |
| PHANTOM_METABOLIC_COST | IMPLEMENT | Payload: gas_cost_usd, fee_earned_usd, net_pnl_usd | Spec 28 §22 event table | Phantom, Oikos |
| PHANTOM_PARAMETER_ADJUSTED | IMPLEMENT | Emitted after Evo parameter adjustment | Spec 28 (A6 gap closure) | Phantom, Evo |
| PHANTOM_POOL_STALE | IMPLEMENT | Thymos subscribes | Spec 28 §22 event table | Phantom, Thymos |
| PHANTOM_POSITION_CRITICAL | IMPLEMENT | Emitted when IL > threshold | Spec 28 §22 event table | Phantom, Thymos |
| PHANTOM_RESOURCE_EXHAUSTED | IMPLEMENT | Emitted on EMERGENCY/CRITICAL metabolic pressure | Spec 28 §22 event table | Phantom, Nova, Thymos |
| PHANTOM_SUBSTRATE_OBSERVABLE | IMPLEMENT | Bedau-Packard observables for Benchmarks | Spec 28 §22 event table | Phantom, Benchmarks |

### Compute / SACM

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| PROOF_FAILED | IMPLEMENT | Emitted by Simula service.py for proof lifecycle | Simula service.py | Simula |
| PROOF_FOUND | IMPLEMENT | Emitted by Simula service.py for proof lifecycle | Simula service.py | Simula |
| PROOF_TIMEOUT | IMPLEMENT | Emitted by Simula service.py for proof lifecycle | Simula service.py | Simula |
| SIMULA_EVOLUTION_APPLIED | DELETE | Duplicate/alias of EVOLUTION_APPLIED; Spec 10 §21 canonical name | Spec 10 §21 - alias | Simula |
| SIMULA_SANDBOX_REQUESTED | IMPLEMENT | Thymos correlation-based request/reply with 30s timeout, fail-closed | Spec 12 (sandbox validation gate) | Thymos, Simula |
| VULNERABILITY_CONFIRMED | IMPLEMENT | Emitted by simula/service.py:1549; Axon/Thymos should subscribe | Simula CLAUDE.md | Simula, Axon, Thymos |

### Misc UNWIRED

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| ANTIBODY_CREATED | DEFER | Thymos Spec 12; part of immune system antibodies | CLAUDE.md glossary | Thymos |
| ANTIBODY_RETIRED | DEFER | Thymos Spec 12 | None in read specs | Thymos |
| BOUNTY_DISCOVERED | DEFER | Oikos Spec 17 economic bounty system | None in read specs | Oikos |
| BOUNTY_EVALUATED | DEFER | Oikos Spec 17 | None in read specs | Oikos |
| BUDGET_PRESSURE | IMPLEMENT | Emitted when fe_budget.is_pressured (60% threshold) | Spec 05 §20 | Nova, Soma |
| CHILD_DISCOVERY_PROPAGATED | DEFER | Likely Mitosis or Federation | None in read specs | Mitosis, Federation |
| CHILD_RESCUED | DEFER | Mitosis Spec 26 | None in read specs | Mitosis |
| CHILD_RESCUE_INITIATED | DEFER | Mitosis Spec 26 | None in read specs | Mitosis |
| CLOCK_OVERRUN | DEFER | Likely Synapse Spec 09 theta cycle monitoring | None in read specs | Synapse |
| EMOTION_STATE_CHANGED | DEFER | Likely Soma Spec 08 or Atune | None in read specs | Soma |
| EVOLUTION_CANDIDATE_ASSESSED | DEFER | _ASSESSED variant not in spec | None in read specs | Evo, Simula |
| EXECUTOR_DISABLED | DEFER | Likely Axon Spec 06 | None in read specs | Axon |
| EXECUTOR_REGISTERED | DEFER | Likely Axon Spec 06 | None in read specs | Axon |
| EXTERNAL_CODE_REPUTATION_UPDATED | DEFER | Likely Identity or Federation | None in read specs | Identity, Federation |
| EXTERNAL_TASK_CONSTITUTIONAL_VETO | DEFER | Not specified in Spec 02 | None in read specs | Equor, Axon |
| EXTERNAL_TASK_FAILED | DEFER | Likely Axon Spec 06 or Oikos | None in read specs | Axon, Oikos |
| EXTERNAL_TASK_STARTED | DEFER | Likely Axon Spec 06 or Oikos | None in read specs | Axon, Oikos |
| HEALING_STORM_ENTERED | DEFER | Thymos Spec 12 | None in read specs | Thymos |
| HEALING_STORM_EXITED | DEFER | Thymos Spec 12 | None in read specs | Thymos |
| HOMEOSTASIS_ADJUSTED | IMPLEMENT | Thymos M8 - HomeostasisController.check_drift_warnings() emits with warn_only=True | MEMORY.md Thymos M8 | Thymos, Nova, Telos |
| INCIDENT_ESCALATED | IMPLEMENT | Thymos SG4 - _try_federation_escalation() emits with federation_broadcast=True | MEMORY.md Thymos SG4 | Thymos, Federation |
| INPUT_CHANNEL_REGISTERED | DEFER | Likely Atune Spec 03 or Axon | None in read specs | Atune, Axon |
| NICHE_ASSIGNED | DEFER | NicheRegistry exists but event not listed as Synapse event | None in Spec 07 | Evo |
| ONEIROS_GENOME_READY | DEFER | Oneiros Spec 13/14 | None in read specs | Oneiros |
| ONEIROS_SLEEP_CYCLE_SUMMARY | DEFER | Oneiros Spec 13/14 | None in read specs | Oneiros |
| OPPORTUNITY_DISCOVERED | DEFER | Likely Oikos economic opportunity detection | None in read specs | Oikos |
| SYSTEM_STOPPED | DEFER | Generic lifecycle event; likely Synapse Spec 09 | None in read specs | Core, Synapse |

---

## 2. DANGLING Events (subscribed but never emitted via enum)

### Identity

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| CHILD_CERTIFICATE_INSTALLED | IMPLEMENT | CertificateManager emits; Oikos subscribes for citizenship tax | Spec 23 §2.2, Oikos CLAUDE.md | Identity, Oikos |
| EQUOR_HITL_APPROVED | IMPLEMENT | Equor emits (string literal not enum - BUG-1); Identity/EIS subscribe | Spec 02 v1.3, Spec 23 | Equor, Identity, EIS, Axon |
| IDENTITY_CERTIFICATE_ROTATED | IMPLEMENT | Federation must refresh mTLS channel credentials | Spec 11 event table | Identity, Federation |
| VAULT_DECRYPT_FAILED | IMPLEMENT | Vault _fire_event() emits; Spec 23 marks RESOLVED | Spec 23 (vault events), GAP_REMEDIATION_PROMPTS | Identity |
| VAULT_KEY_ROTATION_FAILED | IMPLEMENT | Vault rotation path emits | GAP_REMEDIATION_PROMPTS | Identity |
| ORGANISM_SPAWNED | IMPLEMENT | Spec 29 emission with lineage metadata; Identity subscribes | Spec 29 (Phase 2), Spec 23 | Skia, Identity |
| PERSONA_CREATED | IMPLEMENT | Emitted from generate_initial_persona(); Synapse subscribes | Identity CLAUDE.md | Identity, Synapse |
| PERSONA_EVOLVED | IMPLEMENT | Emitted from evolve_persona(); Synapse subscribes | Identity CLAUDE.md | Identity, Synapse |

### Federation

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| FEDERATION_ASSISTANCE_ACCEPTED | IMPLEMENT | Thymos _try_federation_escalation() waits 45s for this before human fallback | Thymos SG4 (MEMORY.md) | Federation, Thymos |
| FEDERATION_CAPACITY_AVAILABLE | IMPLEMENT | ResourceSharingManager emits; WorkRouter subscribes | Federation CLAUDE.md | Federation |
| FEDERATION_INVARIANT_RECEIVED | IMPLEMENT | Kairos fully specifies handler with counter-invariant validation | Spec 22 (P8/M9) | Federation, Kairos |
| FEDERATION_KNOWLEDGE_RECEIVED | IMPLEMENT | Thymos subscribes for antibody sync | Spec 11 event table, Spec 12 | Federation, Thymos |
| FEDERATION_RESURRECTION_APPROVED | IMPLEMENT | Fleet resurrection coordination; Federation emits after quorum | Spec 29 §20 | Federation, Skia |
| FEDERATION_SESSION_STARTED | IMPLEMENT | Nexus subscribes; triggers fragment exchange with new peers | Spec 19 (inbound subscriptions) | Federation, Nexus |
| FEDERATION_SLEEP_SYNC | IMPLEMENT | Oneiros subscribes; nudges sleep pressure on peer sync | Spec 13 §21, Spec 14 | Federation, Oneiros |
| FEDERATION_YIELD_POOL_PROPOSAL | IMPLEMENT | YieldPoolManager emits; peer pool caching subscription | Federation CLAUDE.md | Federation |
| WORLD_MODEL_FRAGMENT_SHARE | IMPLEMENT | Full payload specified; Nexus uses for IIEP | Spec 11 (HIGH-4), Spec 19 | Federation, Nexus |

### Skia / Degradation

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| DEGRADATION_TICK | IMPLEMENT | Hourly degradation pulse; Soma subscribes | Round 1 Prompts (degradation model) | Skia, Soma |
| CONFIG_DRIFT | IMPLEMENT | Simula handles random small perturbation | Round 1 Prompts (degradation model) | Skia, Simula |
| HYPOTHESIS_STALENESS | IMPLEMENT | Evo decays unvalidated hypothesis confidence | Round 1 Prompts (degradation model) | Skia, Evo |
| MEMORY_DEGRADATION | IMPLEMENT | Memory reduces fidelity on old episodes | Round 1 Prompts (degradation model) | Skia, Memory |
| ORGANISM_DIED | IMPLEMENT | Emitted with cause, report, genome_id, snapshot_cid; Skia PhylogeneticTracker subscribes | Spec 09 v1.2, Spec 29 §21.5.6 | Skia |
| ORGANISM_SLEEP | IMPLEMENT | Identity persists state to Neo4j; SACM downgrades queue | Spec 23, Spec 27 §12 | Oneiros, Identity, SACM |
| ORGANISM_WAKE | IMPLEMENT | SACM clears sleep flag, restarts pre-warming | Spec 27 §12 | Oneiros, SACM |
| SKIA_HEARTBEAT_LOST | IMPLEMENT | Thymos, Soma, Synapse subscribe | Spec 29 §3.3, §6.3, §13.2 | Skia, Thymos, Soma, Synapse |
| SKIA_RESURRECTION_PROPOSAL | IMPLEMENT | Fleet resurrection coordination; emitted by surviving Skia | Spec 29 §20 | Skia, Federation |

### SACM

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| COMPUTE_BUDGET_EXPANSION_RESPONSE | IMPLEMENT | Nova handler applies approved_multiplier | Nova CLAUDE.md | Equor, Nova |
| COMPUTE_REQUEST_SUBMITTED | IMPLEMENT | SACM subscribes via _on_compute_request() | Spec 27 §12 | SACMClient, SACM |
| METABOLIC_EMERGENCY | IMPLEMENT | SACM drains non-critical queue, stops pre-warming | Spec 27 §12 | Oikos, SACM |

### Phantom

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| PHANTOM_PRICE_OBSERVATION | IMPLEMENT | Fleet consensus uses 2σ median aggregation (string literal emit - BUG-1) | Spec 28 §22, §24 gap 7 | Phantom |
| PHANTOM_PRICE_UPDATE | IMPLEMENT | Core price distribution; Nova subscribes (string literal emit - BUG-1) | Spec 28 §6.2, §22 | Phantom, Atune, Nova |

### Evo

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| EVO_ADJUST_BUDGET | IMPLEMENT | 4 subscribers (Axon, Phantom, Simula, Voxis); needs emitting or subscriber migration to EVO_PARAMETER_ADJUSTED | Spec 07 §9 (parameter tuning) | Evo, Axon, Simula, Voxis, Phantom |
| EVO_HYPOTHESIS_CREATED | IMPLEMENT | Emitted after generate_hypotheses(); resolved in audit | Spec 07 Integration Surface | Evo, Fovea, Thread |
| EVO_WEIGHT_ADJUSTMENT | IMPLEMENT | Nova subscribes to adjust EFE weights; must be emitted by Evo | Spec 05 §20 | Evo, Nova |
| FOVEA_CALIBRATION_ALERT | DEFER | Not found in read specs; Fovea Spec 20 | None in read specs | Fovea, Evo |
| FOVEA_PARAMETER_ADJUSTMENT | DEFER | Not found in read specs; Fovea Spec 20 | None in read specs | Fovea |
| YIELD_PERFORMANCE_REPORT | DEFER | Not found in read specs; Oikos Spec 17 | None in read specs | Oikos, Evo |

### Nova

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| COMPUTE_BUDGET_EXPANSION_RESPONSE | IMPLEMENT | (See SACM section above) | Nova CLAUDE.md | Equor, Nova |
| EXECUTOR_DEPLOYED | DEFER | Not found in Spec 05 or Spec 10 | None found | Axon/Simula, Nova |
| EXECUTOR_REVERTED | DEFER | Not found in Spec 05 or Spec 10 | None found | Axon/Simula, Nova |
| GOAL_ABANDONED | IMPLEMENT | Spec 05 §20 Nova-emitted; Thread subscribes; conditional assignment pattern (audit false positive) | Spec 05 §20 | Nova, Thread |
| GOAL_ACHIEVED | IMPLEMENT | Spec 05 §20 Nova-emitted; Thread subscribes; conditional assignment pattern (audit false positive) | Spec 05 §20 | Nova, Thread |
| GOAL_OVERRIDE | IMPLEMENT | External goal injection consumed by Nova | Spec 05 §20, §22 #4 | Governance/Federation, Nova |
| HYPOTHESIS_UPDATE | IMPLEMENT | Tournament results - update EFE weight priors | Spec 05 §20 | Evo, Nova |
| NOVEL_ACTION_CREATED | DEFER | No spec reference for this event | None in read specs | Evo, Nova |
| NOVEL_ACTION_REQUESTED | DEFER | Not in Spec 10 consumed events | None in Spec 10 | Nova, Simula |
| NOVA_INTENT_REQUESTED | DEFER | Not in Spec 05 consumed events | None in Spec 05 | Nova |
| ONEIROS_ECONOMIC_INSIGHT | DEFER | Oneiros Spec 13/14 not referenced | None in read specs | Oneiros, Nova |

### Equor

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| EQUOR_AMENDMENT_AUTO_ADOPTED | IMPLEMENT | Per-drive drift auto-proposals auto-approved at confidence ≥ 0.8; may alias DRIVE_AMENDMENT_APPLIED - verify | Spec 02 §19 SG5 | Equor, Thread |
| EQUOR_BUDGET_OVERRIDE | DEFER | Not found in Spec 02 §17.1 | None in Spec 02 | Equor, Axon |
| EQUOR_DRIVE_WEIGHTS_UPDATED | IMPLEMENT | Emitted after amendment applied; Kairos subscribes | Spec 02 §17.1 | Equor, Kairos |
| MEMORY_PRESSURE | DEFER | Not found in any read spec | None found | Memory, Equor |
| THYMOS_REPAIR_APPROVED | IMPLEMENT | Simula consumed event: ChangeApplicator executes the approved repair | Spec 10 §21 | Thymos, Simula |

### Telos

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| ALIGNMENT_GAP_WARNING | IMPLEMENT | Emitted when nominal − effective > 20%; Simula subscribes | Spec 18 §X, §XIII | Telos, Simula |
| COMMITMENT_VIOLATED | IMPLEMENT | Source: Thread. Triggers coherence cost re-computation | Spec 18 §XIII | Thread, Telos |
| EFFECTIVE_I_COMPUTED | IMPLEMENT | Every 60s with EffectiveIntelligenceReport; Benchmarks subscribes | Spec 18 §X, §XIII | Telos, Benchmarks |
| GROWTH_STAGNATION | IMPLEMENT | Emitted when dI/dt < minimum; Thymos subscribes | Spec 18 §X, §XIII | Telos, Thymos |
| SELF_COHERENCE_ALARM | DEFER | Not in Spec 18 emitted or consumed tables | None in Spec 18 | Telos |
| TELOS_POPULATION_SNAPSHOT | IMPLEMENT | Every 60s with population intelligence data; Benchmarks subscribes | Spec 18 §X, §XIII | Telos, Benchmarks |
| TELOS_SELF_MODEL_REQUEST | DEFER | Not in Spec 18 consumed events | None in Spec 18 | Telos |
| WELFARE_OUTCOME_RECORDED | IMPLEMENT | Source: Axon. Feeds CareTopologyEngine | Spec 18 §XIII | Axon, Telos |

### Thread / Narrative

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| REPUTATION_DAMAGED | DEFER | Not found in read specs; Thread Spec 15 | None in read specs | Thread, Nova |
| REPUTATION_MILESTONE | DEFER | Not found in read specs | None in read specs | Thread |
| SCHEMA_INDUCED | DEFER | Evo Phase 3 produces schema inductions but event not listed by this name | None in Spec 07 | Evo, Thread |
| SELF_AFFECT_UPDATED | DEFER | Likely Soma Spec 08; not found in read specs | None in read specs | Soma, Equor, Fovea, Thread |
| SELF_MODEL_UPDATED | DEFER | Not in Spec 18; Telos emits EFFECTIVE_I_COMPUTED instead | None in read specs | Evo/Telos, Thread |

### Misc Dangling

| Event | Category | Reason | Spec Reference | System(s) Involved |
|-------|----------|--------|----------------|---------------------|
| ASSET_DEV_REQUEST | DEFER | Not found in read specs; Oikos Spec 17 | None in read specs | Oikos |
| BENCHMARK_REGRESSION_DETECTED | IMPLEMENT | Simula subscribes per Spec 10 §21; name mismatch with BENCHMARK_REGRESSION - needs rename fix | Spec 10 §21 (name mismatch) | Benchmarks, Simula |
| CERTIFICATE_RENEWAL_REQUESTED | IMPLEMENT | Added to SynapseEventType; Oikos subscribes | MEMORY.md Identity gap closure | Identity, Oikos |
| CHILD_DIED | IMPLEMENT | Core Mitosis lifecycle event; Telos + Oikos + Skia consume | Spec 26 §7, §12, §14; Spec 18; Spec 29 | Mitosis, Telos, Oikos, Skia |
| COMMUNITY_ENGAGEMENT_COMPLETED | DEFER | Not found in read specs | None in read specs | Oikos |
| CONTENT_ENGAGEMENT_REPORT | DEFER | Not found in read specs | None in read specs | Oikos |
| CONTENT_PUBLISHED | DEFER | Not found in read specs | None in read specs | Oikos |
| EMPIRICAL_INVARIANT_CONFIRMED | DEFER | Likely Kairos Spec 22 | None in read specs | Kairos, Nexus |
| EQUOR_AMENDMENT_PROPOSED | IMPLEMENT | severity ≥ 0.9 for 3 consecutive drift checks → auto-proposal; emitted but no subscriber - acceptable as audit trail | Spec 02 §19 SG5 | Equor |
| EQUOR_DRIFT_WARNING | IMPLEMENT | Moderate drift (0.2 ≤ severity < 0.5); emitted but no subscriber - acceptable as telemetry | Spec 02 §17.1, §8.3 | Equor |
| EQUOR_HITL_ESCALATED | DEFER | Not in Spec 02 §17.1 emitted events; possible alias of EQUOR_ESCALATED_TO_HUMAN | Not in Spec 02 emitted list | Equor |
| ETHICAL_DRIFT_RECORDED | DEFER | Not in Spec 02; Equor uses CONSTITUTIONAL_DRIFT_DETECTED instead | None in Spec 02 | Equor |
| EVOLUTION_AWAITING_GOVERNANCE | IMPLEMENT | Emitted when proposal routed to community vote; no subscriber - acceptable as audit event | Spec 10 §21, §23 | Simula |
| EVOLUTION_REJECTED | IMPLEMENT | Emitted on validation/simulation/governance rejection; no subscriber - acceptable as audit event | Spec 10 §21, §23 | Simula |
| EXTERNAL_TASK_COMPLETED | DEFER | Not found in read specs; likely Axon or Oikos | None in read specs | Axon, Oikos |
| GROUND_TRUTH_CANDIDATE | DEFER | Likely Nexus Spec 19 or Kairos Spec 22 | None in read specs | Nexus, Kairos |
| INCIDENT_RESOLVED | IMPLEMENT | Telos subscribes for confabulation rate metric; Federation + Nexus also subscribe. Thymos emits via string literal - needs enum fix | Spec 18 §XIII | Thymos, Telos, Federation, Nexus |
| INSTANCE_RETIRED | IMPLEMENT | Nexus garbage-collects remote profiles; Logos also consumes | Spec 19; Spec 21 | Mitosis, Nexus, Logos |
| INSTANCE_SPAWNED | IMPLEMENT | Nexus creates InstanceDivergenceProfile; Logos triggers world model snapshot | Spec 19; Spec 21 | Mitosis, Nexus, Logos |
| NEXUS_CERTIFIED_FOR_FEDERATION | DEFER | Nexus Spec 19 | None in read specs | Nexus, Federation |
| NEXUS_CONVERGENCE_METABOLIC_SIGNAL | DEFER | Nexus Spec 19 | None in read specs | Nexus, Oikos |
| NEXUS_EPISTEMIC_VALUE | IMPLEMENT | Benchmarks evolutionary observables; Benchmarks + Federation subscribe | MEMORY.md Nexus overhaul | Nexus, Benchmarks, Federation |
| ONEIROS_CONSOLIDATION_COMPLETE | IMPLEMENT | **CRITICAL**: 8 subscribers, emitted only as string literal - needs enum fix | Spec 05 §20, Spec 10 §21 | Oneiros → Nova, Simula + 6 others |
| SIMULA_SANDBOX_RESULT | DEFER | Not in Spec 10 §21 | None in Spec 10 | Simula, Thymos |
| THREAT_ADVISORY_RECEIVED | DEFER | Likely EIS Spec 25 or Federation | None in read specs | EIS, Oikos |
| WAKE_ONSET | DEFER | Likely Oneiros Spec 13/14 or Synapse | None in read specs | Oneiros, Axon |

---

## 3. Summary Totals

| Category | Count | Description |
|----------|-------|-------------|
| **IMPLEMENT** | 180 | Spec requires this event in a working loop; needs wiring |
| **DELETE** | 2 | Dead weight - duplicate or alias of another event |
| **DEFER** | 122 | Spec mentions concept but system not built yet, or no spec evidence found |
| **Total** | 304 | 218 UNWIRED + 86 DANGLING |

### DELETE candidates

| Event | Reason |
|-------|--------|
| `SIMULA_EVOLUTION_APPLIED` | Alias of `EVOLUTION_APPLIED` (Spec 10 §21 canonical name) |
| `NARRATIVE_CHAPTER_CLOSED` | Duplicate of `CHAPTER_CLOSED` (Spec 09 added separately from Spec 15's canonical name) |

### Top-priority IMPLEMENT items (dangling - subscribed but deaf)

| Priority | Event | Subscribers | Fix |
|----------|-------|-------------|-----|
| P0 | `ONEIROS_CONSOLIDATION_COMPLETE` | 8 systems | Convert string literal emit to enum in Oneiros |
| P1 | `EVO_ADJUST_BUDGET` | 4 systems | Either emit from Evo or migrate subscribers to `EVO_PARAMETER_ADJUSTED` |
| P1 | `INCIDENT_RESOLVED` | 3 systems | Convert Thymos string literal to enum |
| P1 | `CHILD_DIED` | 3 systems | Convert Mitosis string literal to enum |
| P1 | `PHANTOM_PRICE_UPDATE` | Nova | Convert Phantom string literal to enum |
| P1 | `PHANTOM_PRICE_OBSERVATION` | Kairos, Phantom | Convert Phantom string literal to enum |
| P2 | `ORGANISM_SLEEP` / `ORGANISM_WAKE` | 4 systems | Synapse should relay from SLEEP_INITIATED / WAKE_INITIATED |
| P2 | `EFFECTIVE_I_COMPUTED` | Benchmarks | Telos must emit every 60s |
| P2 | `TELOS_POPULATION_SNAPSHOT` | Benchmarks | Telos must emit every 60s |
| P2 | `BENCHMARK_REGRESSION_DETECTED` | Simula | Rename subscriber to `BENCHMARK_REGRESSION` (name mismatch) |

---

*Generated 2026-03-09 by spec-driven triage of SYNAPSE_AUDIT.md against `.claude/EcodiaOS_Spec_*.md`.*
