# Federation System - CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_11_Federation.md` (Spec 11b)
**Status:** Core + population dynamics + peer discovery + RE quality scoring + Active Work Pooling complete (2026-03-09)

---

## What's Implemented

### Core Protocol
- **Identity**: Ed25519 keypair, `InstanceIdentityCard`, dynamic constitutional hash from drive weights
- **Handshake**: 4-phase mTLS protocol (HELLO → CHALLENGE → ACCEPT → CONFIRM), nonce replay prevention
- **Trust**: 5 levels (NONE → ALLY), score thresholds (5/20/50/100), 3x violation penalty, privacy breach instant reset, time-based decay (24h grace)
- **Knowledge Exchange**: `KnowledgeExchangeManager` + IIEP `ExchangeProtocol` (dual paths), trust-gated sharing permissions
- **Privacy Filter**: PII key stripping, email/phone regex, PRIVATE always removed, COMMUNITY_ONLY at COLLEAGUE+
- **Coordination**: Assistance request/response with trust gate (COLLEAGUE+), Equor review
- **Channels**: mTLS via httpx, `ChannelManager` with SSL context, dev mode fallback
- **Link Persistence**: Redis primary + local file fallback
- **IIEP**: Exchange protocol, ingestion pipeline (6-stage + 2 new stages), threat intelligence, reputation staking
- **Certificate Validation**: `_validate_sender_certificate()` on all inbound operations

### Active Work Pooling (2026-03-09)
- **TaskDelegationManager** (`task_delegation.py`): offer/accept/settle discrete tasks between trusted peers; COLLEAGUE+ trust gate (normalised ≥ `required_trust_level`, default 0.7); USDC payment via WalletClient on completion; full lifecycle tracking (OFFERED→ACCEPTED→COMPLETED/FAILED)
- **BountySplitter** (`bounty_splitting.py`): decompose large bounties into N sub-tasks; Simula-assisted task boundary generation with naive fallback; round-robin offer to PARTNER+ peers; `on_bounty_paid()` distributes 90% of total to contributors proportionally; orchestrator retains 10%
- **ResourceSharingManager** (`resource_sharing.py`): publish spare compute capacity as `CapacityOffer` via `FEDERATION_CAPACITY_AVAILABLE`; auto-retract when CPU > 85%, auto-republish when CPU < 40% (via `ORGANISM_TELEMETRY`); `best_peer_for_offload()` selects lowest-cost matching specialisation; fee payment via `TaskDelegationManager.settle_payment()`
- **YieldPoolManager** (`yield_pool.py`): propose/join/fund/deploy/settle multi-instance yield pools; trust ≥ 0.9 (ALLY threshold); proportional yield distribution on settlement; subscribes to `FEDERATION_YIELD_POOL_PROPOSAL` for peer pool caching
- **FederationMarketplace** (`marketplace.py`): post listings, receive bids, award best bidder (trust × price_ratio × specialisation_match scoring), rate results; Redis pub/sub on `eos:federation:marketplace`; ratings fed back as `FEDERATION_TRUST_UPDATED` signals
- **WorkRouter** (`work_router.py`): Nexus-aware specialisation routing; updates from `NEXUS_EPISTEMIC_VALUE` (epistemic depth) + `FEDERATION_CAPACITY_AVAILABLE` (declared specialisations) + empirical task outcomes; composite score = depth×0.5 + declared×0.3 + empirical×0.2, scaled by normalised trust

**New SynapseEventTypes (9)**:
`FEDERATION_TASK_OFFERED`, `FEDERATION_TASK_ACCEPTED`, `FEDERATION_TASK_COMPLETED`, `FEDERATION_TASK_PAYMENT`, `FEDERATION_CAPACITY_AVAILABLE`, `FEDERATION_YIELD_POOL_PROPOSAL`, `FEDERATION_TASK_DECLINED`, `FEDERATION_BOUNTY_SPLIT`, `FEDERATION_WORK_ROUTED`

**New primitive types (15)** in `primitives/federation.py`:
`TaskType`, `TaskStatus`, `TaskDelegation`, `DelegationResult`, `CapacityOffer`, `OffloadRequest`, `PoolParticipant`, `YieldPoolStatus`, `YieldPoolProposal`, `MarketplaceListingStatus`, `MarketplaceListing`, `MarketplaceBid`, `MarketplaceRating`

**Wiring in service.py**:
- All 6 work pooling managers built in `initialize()` after IIEP setup
- `set_event_bus()` propagates bus to all 6 managers
- `set_simula()` propagates to `BountySplitter`
- `set_oikos()` propagates to `YieldPoolManager`
- `_on_bounty_paid()` handler distributes sub-rewards on `BOUNTY_PAID` event
- Public API: `task_delegation`, `bounty_splitter`, `resource_sharing`, `yield_pool`, `marketplace` properties + `route_work()` + `record_task_outcome()`

### Population Dynamics (P2-2 - 2026-03-07)
- **Synapse Events**: 8 outbound events emitted (`_emit_synapse_event` helper), 12 `SynapseEventType` entries (including new `FEDERATION_PRIVACY_VIOLATION`)
- **Synapse Subscriptions**: 5 inbound handlers via `set_event_bus()` - economic state, cert rotation, incidents, sleep consolidation, privacy violations
- **Metabolic Gate**: Trust upgrades blocked when `metabolic_efficiency < 0.1`
- **Sleep Certification**: Knowledge sharing gated on `ONEIROS_CONSOLIDATION_COMPLETE` (bypasses THREAT_ADVISORY)
- **Constitutional Hash**: Dynamic from injectable drive weights via `_compute_constitutional_hash()`, `set_equor()` pulls live weights
- **Novelty Score**: `_compute_novelty_score()` via content hash tracking, included in event payloads
- **Jaccard Fix**: `ReputationBond.claim_content` stores original content for real word-set extraction
- **ExchangeEnvelope Fix**: `send_fragment()` and `request_divergence_profile()` use correct field names
- **RE Training**: `RE_TRAINING_EXAMPLE` events emitted on assistance decisions
- **RE_ADAPTER_DIFF**: `KnowledgeType` added, ALLY trust gate - transport type only, no LoRA logic yet

### Peer Discovery (P2-3 - 2026-03-07)
- **`seed_peers`** field in `FederationConfig` - list of endpoint URLs to auto-connect on startup
- **`seed_retry_interval_seconds`** - configurable retry interval for failed seed connections (default 300s)
- **`_connect_seed_peers()`** - async, fire-and-forget, skips already-linked endpoints, logs success/failure
- **`_retry_seed_peers()`** - single-pass retry loop after interval, converges to empty when all connected
- Resolves Spec §XIII gap #5 (no discovery mechanism). Manual `establish_link(endpoint)` still works.

### Knowledge Stub Consolidation (P2-3 - 2026-03-07)
- **`_retrieve_procedures()`** now delegates to `collect_procedures()` from `exchange.py` (single source of truth)
- **`_retrieve_hypotheses()`** now delegates to `collect_hypotheses()` from `exchange.py`
- `KnowledgeExchangeManager` gains `evo` and `instance_id` constructor params
- `set_evo()` propagates to both `_ingestion._evo` and `_knowledge._evo`
- Resolves Spec §VIII Fix #11 (dead knowledge.py stubs)

### Privacy Violation Detection (P2-3 - 2026-03-07)
- **Ingestion Stage 3.5**: `_detect_privacy_violation()` scans inbound payload content for 24 PII key patterns
- **`_emit_privacy_violation()`**: emits `FEDERATION_PRIVACY_VIOLATION` Synapse event + logs warning
- **`_on_privacy_violation()`** in FederationService: receives the event, records PRIVACY_BREACH interaction, triggers trust zero-reset via `_update_trust_and_emit()`
- `IngestionPipeline` gains `event_bus` constructor param; propagated from `set_event_bus()`
- Resolves Spec §XI gap (`FEDERATION_PRIVACY_VIOLATION` event not yet emitted)

### RE Semantic Quality Scoring (P2-3 - 2026-03-07)
- **Ingestion Stage 4.5**: `_run_re_quality_check()` scores HYPOTHESIS payloads at PARTNER+ trust
- Scores three dimensions via RE/Claude: coherence, novelty, constitutional safety
- Uses harmonic mean - any single weak dimension degrades the composite score
- Below `_RE_QUALITY_THRESHOLD` (0.35): DEFERRED (not rejected - local evidence may validate later)
- `set_re(re)` wirer on `FederationService` propagates to `_ingestion._re`
- Fail-open: no RE wired → all payloads pass Stage 4.5
- Resolves Spec §XII §2 (RE-assisted semantic quality scoring)

---

## Key Files

| File | Role |
|------|------|
| `service.py` | Main orchestrator - wires all subsystems, Synapse integration, link lifecycle, peer discovery, work pooling |
| `identity.py` | `IdentityManager` - keypair, identity card, constitutional hash |
| `trust.py` | `TrustManager` - scoring, level transitions, decay |
| `knowledge.py` | `KnowledgeExchangeManager` - knowledge exchange, now delegates procedures/hypotheses to IIEP collectors |
| `privacy.py` | `PrivacyFilter` - PII removal, consent enforcement |
| `coordination.py` | `CoordinationManager` - assistance request/response |
| `channel.py` | `ChannelManager` - mTLS channel lifecycle |
| `handshake.py` | 4-phase handshake protocol |
| `exchange.py` | IIEP `ExchangeProtocol` - push/pull with provenance; canonical collectors for hypotheses/procedures |
| `ingestion.py` | IIEP ingestion pipeline - 7 stages: dedup, loop, privacy scan, EIS, RE quality, Equor, routing |
| `iiep.py` | Economic coordination - `CapabilityMarketplace`, `MutualInsurancePool` |
| `reputation_staking.py` | Bond create/forfeit/recover, contradiction detection |
| `threat_intelligence.py` | Signed advisory broadcast, trust-gated receipt |
| `task_delegation.py` | `TaskDelegationManager` - offer/accept/settle discrete tasks; USDC payment |
| `bounty_splitting.py` | `BountySplitter` - large bounty decomposition + multi-instance co-solving |
| `resource_sharing.py` | `ResourceSharingManager` - compute offloading, capacity advertisement |
| `yield_pool.py` | `YieldPoolManager` - federated capital pooling for yield positions |
| `marketplace.py` | `FederationMarketplace` - task marketplace with bidding and ratings |
| `work_router.py` | `WorkRouter` - Nexus specialisation-aware task routing |

---

## Known Issues / Remaining Work

- **[MEDIUM]** `systems.nexus.types` still inline lazy imports in `service.py` - should move to shared primitives
- **[MEDIUM]** LoRA diff collection/application for `RE_ADAPTER_DIFF` not implemented (requires RE system)
- **[LOW]** Privacy filter is regex-based, not k-anonymity/differential privacy
- **[LOW]** Nova alignment check is a pass-through flag, not a real Nova call
- **[LOW]** Trust decay contradicts spec philosophy (biological decay vs. digital persistence) - undocumented design choice
- **[MEDIUM]** YieldPoolManager escrow is simulated via USDC transfer to placeholder address - real smart contract escrow needed for capital safety
- **[MEDIUM]** `_execute_offload()` in ResourceSharingManager is a stub - real execution should dispatch via Axon `execute_action()`
- **[LOW]** Marketplace Redis subscriber (`_redis_subscriber`) requires `RedisClient` to expose an async generator `subscribe(channel)` - verify interface matches
- **[LOW]** TaskDelegationManager does not persist task state to Neo4j - task records lost on restart

---

## Integration Points

**Emits (Synapse):** `FEDERATION_LINK_ESTABLISHED`, `FEDERATION_LINK_DROPPED`, `FEDERATION_TRUST_UPDATED`, `FEDERATION_KNOWLEDGE_SHARED`, `FEDERATION_KNOWLEDGE_RECEIVED`, `FEDERATION_INVARIANT_RECEIVED`, `WORLD_MODEL_FRAGMENT_SHARE`, `FEDERATION_ASSISTANCE_ACCEPTED`, `FEDERATION_ASSISTANCE_DECLINED`, `FEDERATION_PRIVACY_VIOLATION`, `RE_TRAINING_EXAMPLE`, `FEDERATION_TASK_OFFERED`, `FEDERATION_TASK_ACCEPTED`, `FEDERATION_TASK_COMPLETED`, `FEDERATION_TASK_PAYMENT`, `FEDERATION_TASK_DECLINED`, `FEDERATION_CAPACITY_AVAILABLE`, `FEDERATION_YIELD_POOL_PROPOSAL`, `FEDERATION_BOUNTY_SPLIT`, `FEDERATION_WORK_ROUTED`, `FEDERATION_SLEEP_SYNC` (on `SLEEP_INITIATED` - 2026-03-09)

**Gap closure (2026-03-07, event coverage):**
- `WORLD_MODEL_FRAGMENT_SHARE` - now emitted in `FederationService.send_fragment()` when a world model fragment is accepted by the peer (non-REJECTED receipt verdict). Data: `link_id`, `remote_instance_id`, `message_id`, `fragment_type`.
- `FEDERATION_INVARIANT_RECEIVED` - now emitted in `FederationService.handle_exchange_envelope()` when inbound PUSH payloads include at least one accepted hypothesis with confidence ≥ 0.9 (Kairos-distilled causal invariants travel as high-confidence hypotheses). Data: `link_id`, `remote_instance_id`, `invariant_count`, `min_confidence`.

**Gap closure (2026-03-08, world model fragment ingest):**
- `WORLD_MODEL_FRAGMENT_SHARE` subscription added in `set_event_bus()`. Handler `_on_world_model_fragment_share`:
  1. Self-loop guard: skips if `sender_instance_id == self._instance_id`
  2. Calls `self._knowledge.record_received_fragment()` if available (non-fatal on missing method)
  3. Fire-and-forgets `FEDERATION_KNOWLEDGE_RECEIVED` with `knowledge_type="world_model_fragment"`, `fragment_id`, `sleep_certified`, `domain_labels`, `item_count=1`
  - Closes Nexus CLAUDE.md known issue #10 (`WORLD_MODEL_FRAGMENT_SHARE` had no subscriber)

**Gap closure (2026-03-08, Nexus integration):**
- `NEXUS_CERTIFIED_FOR_FEDERATION` subscription added in `set_event_bus()`. Handler `_on_nexus_certified_for_federation`:
  1. Marks all `schema_ids` as sleep-certified in `_sleep_certified_knowledge`
  2. Emits `FEDERATION_KNOWLEDGE_SHARED` with trigger=nexus_certified
  3. Fires `_share_certified_schema_to_link()` (fire-and-forget) for each COLLEAGUE+ link (trust_score ≥ 20) × each schema_id
  4. `_share_certified_schema_to_link()` delegates to `self._knowledge.send_certified_schema()` if available; logs deferred otherwise (non-fatal)

**Gap closure (2026-03-08, orphan event wiring):**
- `ECONOMIC_STATE_UPDATED` - now emitted by OikosService every consolidation cycle (5–15 min). Federation caches `metabolic_efficiency` + `liquid_balance_usd` for metabolic gating.
- `IDENTITY_CERTIFICATE_ROTATED` - now emitted by CertificateManager on every `install_certificate()`. Federation updates cached `certificate_fingerprint`.

**Subscribes (Synapse):** `ECONOMIC_STATE_UPDATED` (Oikos), `IDENTITY_CERTIFICATE_ROTATED` (Identity), `INCIDENT_DETECTED` (Thymos), `INCIDENT_RESOLVED` (Thymos - 8 Mar 2026), `ONEIROS_CONSOLIDATION_COMPLETE` (Oneiros), `FEDERATION_PRIVACY_VIOLATION` (self - triggers trust reset), `SKIA_RESURRECTION_PROPOSAL` (Skia), `NEXUS_CERTIFIED_FOR_FEDERATION` (Nexus - 8 Mar 2026), `WORLD_MODEL_FRAGMENT_SHARE` (Nexus/Federation - 8 Mar 2026), `BOUNTY_PAID` (Oikos - 9 Mar 2026, BountySplitter payment distribution), `ORGANISM_TELEMETRY` (Synapse - 9 Mar 2026, ResourceSharingManager CPU monitoring), `NEXUS_EPISTEMIC_VALUE` (Nexus - 9 Mar 2026, WorkRouter specialisation scoring), `FEDERATION_CAPACITY_AVAILABLE` (self - 9 Mar 2026, peer capacity caching), `FEDERATION_YIELD_POOL_PROPOSAL` (peers - 9 Mar 2026, YieldPoolManager pool discovery), `SLEEP_INITIATED` (Oneiros - 9 Mar 2026, triggers FEDERATION_SLEEP_SYNC broadcast to active peers)`

**Post-init wirers:** `set_equor()`, `set_evo()`, `set_oikos()`, `set_simula()`, `set_eis()`, `set_re()`, `set_atune()`, `set_certificate_manager()`, `set_event_bus()`

**Runtime-adjustable thresholds (Autonomy Audit - 8 Mar 2026):**
- `set_metabolic_gate_threshold(float)` - below this `metabolic_efficiency`, trust upgrades block. Default: 0.1.
- `set_colleague_trust_floor(float)` - minimum trust score for COLLEAGUE+-gated schema sharing. Default: 20.0.
- `set_re_quality_threshold(float)` - ingestion Stage 4.5 minimum quality score. Default: 0.35.

**Dependencies:** Equor (constitutional review), Memory (knowledge retrieval), Redis (link persistence), Identity (certificates - via lazy imports), Nexus (fragment protocol - via lazy imports)

---

## Autonomy Audit Gap Closure (8 Mar 2026)

### Dead Wiring Fixed
- **`set_evo()`**: now called from `wire_federation_phase()` in `core/wiring.py` - enables HYPOTHESIS/PROCEDURE collection for `push_knowledge()` and ingestion routing to Evo.
- **`set_simula()`**: now called from `wire_federation_phase()` - enables MUTATION_PATTERN kind for cross-instance push.
- **`set_re()`**: now called from `wire_federation_phase()` - activates Stage 4.5 RE semantic quality scoring; was permanently fail-open before.
- **`set_oikos()`**: wired in `registry.py` immediately after `wire_oikos_phase()` (Phase 9b) - Oikos initializes after federation so it can't be in `wire_federation_phase()`.

### Invisible Telemetry Fixed
- **Ingestion pipeline stats**: `_total_processed/_accepted/_quarantined/_rejected/_deferred` now emitted as `EVOLUTIONARY_OBSERVABLE` on every processed envelope. Nova/Benchmarks can observe pipeline health in real-time.

### Blocked Actions Fixed
- **`_incident_heightened_scrutiny` never cleared**: subscribed to `INCIDENT_RESOLVED`; `_on_incident_resolved()` resets the flag so trust upgrades resume after incidents close. Previously blocked permanently after any incident.

### Static Thresholds Made Runtime-Adjustable
- **Metabolic gate threshold** (`0.1`): now `_metabolic_gate_threshold` instance var, adjustable via `set_metabolic_gate_threshold()`.
- **COLLEAGUE+ trust floor** (`20`): now `_colleague_trust_floor` instance var, adjustable via `set_colleague_trust_floor()`.
- **RE quality threshold** (`0.35`): now `_re_quality_threshold` instance var in IngestionPipeline, adjustable via `service.set_re_quality_threshold()`.

---

## Architecture Rules

- No module-level cross-system imports - use `TYPE_CHECKING` guard + lazy wrappers or inline function-level imports
- All trust updates go through `_update_trust_and_emit()` - never call `self._trust.update_trust()` directly
- All Synapse emissions go through `_emit_synapse_event()` helper (fire-and-forget, logged on error)
- Constitutional hash must stay deterministic (6-decimal precision, sorted drive keys)
- Ingestion stages are ordered: dedup(1) → loop(2) → privacy scan(3.5) → EIS(4) → RE quality(4.5) → Equor(5) → routing(6)
- Peer discovery is fire-and-forget from `initialize()` - never blocks startup
- Oikos is wired to federation post-init in Phase 9b of registry (after `wire_oikos_phase`) - not in `wire_federation_phase`
