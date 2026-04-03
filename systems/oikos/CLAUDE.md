# Oikos - CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_17_Oikos.md`
**System ID:** `oikos`
**Last updated:** 9 March 2026 (Synapse triage pass)
**Role:** Economic metabolism - the organism's capacity to acquire, allocate, conserve, and generate resources autonomously. Levels 1–4 teach it to hunt, farm, build, and breed. Levels 5–10 transform it into infrastructure that other agents depend on.

> *Oikos is the difference between a system that costs money and a system that sustains itself.*

---

## Metabolic Architecture

### MetabolicPriority Cascade (strict ordering)
```
SURVIVAL → OPERATIONS → OBLIGATIONS → MAINTENANCE → GROWTH → YIELD → ASSETS → REPRODUCTION
```
The organism starves ambition before survival. Starvation levels: NOMINAL → AUSTERITY → CRITICAL → EXISTENTIAL.

### Wallet Architecture (Base L2)
| Wallet | Purpose | Controls |
|--------|---------|---------|
| Hot wallet | Operational capital (max 2× monthly BMR) | Single key |
| Warm wallet | Growth capital (yield, assets, children) | 2-of-3 multisig |
| Cold wallet | Survival reserve (ONLY in metabolic emergency) | 2-of-3, no bypass |
| Revenue wallet | Inbound-only; auto-sweeps to hot every 6h | Single key |

---

## What's Implemented

### Core Service (`service.py` - ~3400 lines)
- **EconomicState** lifecycle: liquid balance, survival reserve, deployed capital, receivables, costs, net worth
- **MetabolicPriority** cascade (`models.py`); **StarvationLevel** classification with configurable day thresholds
- **BMR computation** via pluggable `CostModel` (NeuroplasticityBus hot-swappable)
- **Metabolic gate enforcement** - `check_metabolic_gate()` gates expensive actions; denied actions queued as `DeferredAction` with consolidation retry
- **Starvation enforcement** - graduated shedding: organs → derivatives → foraging → yield → all non-survival
- **Sliding-window accumulators** - `deque[_RevenueEntry]` with 30-day eviction; authoritative for `costs_24h/7d/30d` and `revenue_24h/7d/30d`
- **Neo4j audit trail** - `_audit_economic_event()` writes to Redis stream `eos:oikos:audit_trail` for async ingestion
- **RE training emissions** - `_emit_re_training()` at every economic decision point (Stream 5: economic reasoning)
- **Ecological niche identification** - `identify_niches()` from bounty domains, asset types, knowledge market
- **Genome extraction** - subscribes to `GENOME_EXTRACT_REQUEST`, responds with `OrganGenomeSegment` + niche data

### Subsystems

| File | Role (Spec Level) |
|------|------------------|
| `bounty_hunter.py` | Level 1 - BountyHunter: platform scanning, evaluation, acceptance (MIN_ROI: 2.0×, max 3 concurrent) |
| `yield_strategy.py` | Level 2 - DeFi yield deployment + `YieldReinvestmentEngine`; 6 protocols: Aave/Morpho/Aerodrome/Moonwell/ExtraFinance/Beefy (Base L2) |
| `risk_manager.py` | Level 2 - `RiskManager`: portfolio concentration + health factor + emergency deleverage |
| `treasury_manager.py` | Level 2 - `TreasuryManager`: 40/30/20/10 bucket rebalancing via YIELD_DEPLOYMENT_REQUEST |
| `governance_tracker.py` | Level 2 - `GovernanceTokenTracker`: AERO/WELL/COMP tracking, Snapshot proposal discovery, Nova-deliberated voting |
| `cross_chain_observer.py` | Level 2 - `CrossChainYieldObserver`: Arbitrum/Optimism/Polygon yield monitoring (observation only - no cross-chain deploys) |
| `asset_factory.py` | Level 3 - Asset ideation, promotion, lifecycle; `promote_to_asset()` |
| `mitosis.py` | Level 4 - Reproduction economics: seed capital, independence criteria, rescue funding |
| `fleet.py` | Level 4 - Child health monitoring, dividend collection |
| `protocol_factory.py` | Level 5 - Protocol design pipeline, 10k Monte Carlo simulation, multi-layer security verification |
| `immune.py` | Level 6 - 4-layer economic defence: TX shield, threat patterns, protocol health, fleet threat intel |
| `reputation.py` | Level 7 - PoC attestations (EAS/Base), reputation score 0-1000, credit tiers (NEWCOMER → SOVEREIGN) |
| `knowledge_market.py` | Level 8 - Attestations, cognitive oracles, subscription tokens; scarcity × loyalty pricing |
| `derivatives.py` | Level 8 - Cognitive futures, capacity options, subscription tokens (ERC-20) |
| `economic_simulator.py` | Level 9 - Monte Carlo economic dreaming (10k paths × 365 days; 8 named stress scenarios) |
| `dream_worker.py` | Level 9 - Background dreaming worker |
| `interspecies.py` | Level 10 - IIEP: capability marketplace, mutual insurance, collective liquidity |
| `morphogenesis.py` | Economic organ lifecycle (embryonic → growing → mature → atrophying → vestigial) |
| `genome.py` | OikosGenomeExtractor - extract/seed heritable economic parameters |
| `models.py` | MetabolicPriority, StarvationLevel, EconomicState, ActiveBounty, OwnedAsset, ChildPosition, EcologicalNiche |
| `metabolism_api.py` | FastAPI endpoints for metabolic state + runway alarms |
| `tollbooth.py` | Smart contract tollbooth for asset revenue collection |
| `snapshot_writer.py` | TimescaleDB/Redis state persistence + hourly CostSnapshot Neo4j writes |
| `threat_modeler.py` | Threat modeling during consolidation |
| `metrics.py` | Prometheus/OpenTelemetry metric emission |
| `affiliate_scanner.py` | Expanded revenue - autonomous affiliate program discovery, Equor-gated applications, referral tracking with mandatory disclosure |
| `content_monetization.py` | Expanded revenue - content platform stats ingestion, monetization threshold detection, enrollment via Equor gate |
| `reputation_tracker.py` | Community Reputation Tracker - 9 social metrics, composite 0-100 score, Neo4j snapshots, social graph edges, economic multipliers |

---

## Community Reputation Tracker (`reputation_tracker.py` - 9 Mar 2026)

Distinct from `reputation.py` (EAS cryptographic credit scoring, 0–1000). This tracks the organism's **social presence** across developer platforms.

### Metrics tracked (`ReputationMetrics`)
| Metric | Weight | Normalisation target |
|--------|--------|---------------------|
| `github_prs_merged` | 35% | 20 merges → 100 pts |
| `github_stars_received` | 20% | 100 stars → 100 pts |
| `bounties_solved` + `bounties_solved_value_usdc` | 20% | 50 bounties → 100 pts |
| `x_followers` | 10% | 5 000 followers → 100 pts |
| `x_impressions_30d` | 8% | 500k impressions → 100 pts |
| `github_issues_resolved` | 7% | 50 issues → 100 pts |
| `devto_followers` / `devto_views_30d` | passive | logged only |

`reputation_score` = weighted sum, 0–100 float.

### Neo4j social graph
- PR merge: `(organism)-[:CONTRIBUTED_TO]->(repo)` via `_write_contributed_to()`
- Developer recognition (positive comment): `(developer)-[:RECOGNISES]->(organism)` via `record_developer_recognition()`
- Star/follow: `(organism)-[:INTERESTED_IN]->(target)` via `_write_social_graph()`
- All edges use MERGE and carry `created_at` / `updated_at` timestamps
- Hourly: `(:ReputationSnapshot)` node written with all 9 metrics + composite score

### Economic multipliers
- `get_reputation_multiplier()` → `1.0 + (score / 200)` - applied to bounty `estimated_cost` in `_on_bounty_solution_pending()` (higher rep → lower reserved capital → more bounties accepted)
- `get_consulting_rate_multiplier()` (on `OikosService`) → `1.25` if score > 70, else `1.0` - for use by `ServiceOfferExecutor`

### Events emitted / consumed
| Direction | Event | When |
|-----------|-------|------|
| Emits | `REPUTATION_SNAPSHOT` | Hourly loop |
| Emits | `REPUTATION_DAMAGED` | Score drops ≥ 5 pts |
| Emits | `REPUTATION_MILESTONE` | Score crosses 25 / 50 / 70 / 90 |
| Emits | `SOCIAL_GRAPH_UPDATED` | Any Neo4j social edge written |
| Consumes | `BOUNTY_PR_MERGED` | Increments `github_prs_merged`; writes `CONTRIBUTED_TO` edge |
| Consumes | `BOUNTY_PAID` | Increments `bounties_solved` + value |
| Consumes | `COMMUNITY_ENGAGEMENT_COMPLETED` | `star_repo` → `INTERESTED_IN` edge |

### Wiring
`ReputationTracker` is instantiated in `OikosService.__init__` as `self._community_reputation`.
`attach()` calls `set_neo4j()`, `set_instance_id()`, and `asyncio.ensure_future(initialize())`.
`OikosService.get_reputation_multiplier()` and `get_consulting_rate_multiplier()` are public API.

---

## Synapse Events

### Emitted
| Event | When |
|-------|------|
| `METABOLIC_GATE_CHECK` | Every `check_metabolic_gate()` call (request) |
| `METABOLIC_GATE_RESPONSE` | Every `check_metabolic_gate()` call (resolved answer: granted/denied) |
| `ECONOMIC_ACTION_DEFERRED` | Gate denies an action |
| `STARVATION_WARNING` | Starvation level enters AUSTERITY+ |
| `STARVATION_WARNING_ACCURATE` | runway < 24h based on real two-ledger burn rate (api + infra) against live USDC balance. Payload: runway_hours, burn_rate_usd_per_hour, api_burn_rate_usd_per_hour, infra_burn_rate_usd_per_hour, balance_usd, dependency_ratio |
| `FUNDING_REQUEST_ISSUED` | EMERGENCY/CRITICAL starvation - organism requests capital infusion |
| `BOUNTY_PAID` | Re-emitted by Oikos (source_system="oikos") after crediting bounty revenue. Also triggered by `BOUNTY_PR_MERGED` - merged PR revenue is credited before re-broadcasting. |
| `BUDGET_EXHAUSTED` | Per-system daily allocation exhausted (replaces METABOLIC_PRESSURE overload) |
| `ECONOMIC_VITALITY` | Every starvation-level transition + every consolidation cycle (SG2) |
| `INTEROCEPTIVE_PERCEPT` | When starvation ≥ CAUTIOUS or efficiency < 1.0 (M10 - economic percept for GWT) |
| `ASSET_BREAK_EVEN` | First time a live asset's cumulative revenue ≥ dev cost (SG5) |
| `CHILD_INDEPENDENT` | Child lifecycle transitions to INDEPENDENT status (SG5) |
| `GENOME_EXTRACT_RESPONSE` | Responding to Mitosis |
| `YIELD_DEPLOYMENT_REQUEST` | Requesting Axon to execute DeFi deposit |
| `RE_TRAINING_EXAMPLE` | Every economic decision point |
| `METABOLIC_PRESSURE` | Starvation enforcement → somatic collapse signal |
| `OIKOS_*` | State updates, bounty/yield/asset/fleet lifecycle |
| `BOUNTY_REJECTED` | Equor denied bounty acceptance; capital preserved |
| `ASSET_DEV_DEFERRED` | Asset dev cost denied (insufficient capital, metabolic gate, or Equor veto) |
| `METABOLIC_EFFICIENCY_PRESSURE` | efficiency < 0.8 each consolidation cycle - Evo learning signal |
| `AFFILIATE_PROGRAM_DISCOVERED` | AffiliateProgramScanner found a new eligible program |
| `AFFILIATE_MEMBERSHIP_APPLIED` | Equor-approved affiliate application submitted |
| `AFFILIATE_REVENUE_RECORDED` | Referral commission received; also triggers `REVENUE_INJECTED` (stream="affiliate") |
| `CONTENT_MONETIZATION_MILESTONE` | Content platform threshold crossed; application triggered |
| `CONTENT_REVENUE_RECORDED` | Platform content payment received; also triggers `REVENUE_INJECTED` (stream="content") |
| `REVENUE_DIVERSIFICATION_PRESSURE` | Any single revenue source >80% of 30d revenue; instructs Evo to generate diversification hypotheses |
| `BENCHMARKS_METABOLIC_VALUE` | efficiency < 0.8 (pressure) + on recovery to nominal - Benchmarks KPI push |
| `YIELD_REINVESTED` | `YieldReinvestmentEngine` - accrued yield redeployed (compound growth loop) |
| `GOVERNANCE_VOTE_CAST` | `GovernanceTokenTracker` - organism cast a governance vote via Snapshot |
| `PORTFOLIO_REBALANCED` | `RiskManager` - concentration/leverage/emergency rebalance executed |
| `CROSS_CHAIN_OPPORTUNITY` | `CrossChainYieldObserver` - sustained ≥2× Base APY cross-chain opportunity flagged |
| `TREASURY_REBALANCED` | `TreasuryManager` - 40/30/20/10 bucket drift corrected |
| `METABOLIC_EMERGENCY` | `_enforce_starvation()` - emitted when starvation reaches CRITICAL. Payload: starvation_level, runway_hours, liquid_balance_usd, shed_priority="all_non_survival". SACM drains non-critical queue; Axon force-opens non-essential circuit breakers. **(IMPLEMENTED 9 Mar 2026)** |
| `OIKOS_METABOLIC_SNAPSHOT` | Every consolidation cycle - cached by Equor (Fix 4.4 metabolic-aware eval) and Mitosis FleetService (reactive fitness awareness). Payload: starvation_level, efficiency, runway_days, liquid_balance, net_worth |
| `ECONOMIC_STATE_UPDATED` | Every consolidation cycle - cached by Federation for metabolic gating. Payload: metabolic_efficiency (float), liquid_balance_usd, starvation_level, burn_rate_usd |

### Consumed
| Event | Source |
|-------|--------|
| `GENOME_EXTRACT_REQUEST` | Mitosis |
| `YIELD_DEPLOYMENT_RESULT` | Axon |
| `SIMULA_ROLLBACK_PENALTY` | Simula |
| `METABOLIC_PRESSURE` | Synapse burn rate updates - now includes `api_cost_usd_per_hour` + `infra_cost_usd_per_hour` split for two-ledger tracking |
| `INFRASTRUCTURE_COST_CHANGED` | Synapse InfrastructureCostPoller - updates `_infra_burn_rate_usd_per_hour` immediately on >5% RunPod cost change; re-checks `STARVATION_WARNING_ACCURATE` |
| `REVENUE_INJECTED` | Yield strategy, bounty completions |
| `BOUNTY_SOLUTION_PENDING` | BountyHunter |
| `BOUNTY_PR_MERGED` | Axon MonitorPRsExecutor - **not subscribed by Oikos** (revenue is credited via `BOUNTY_PAID` which `MonitorPRsExecutor` also emits on merge; `_on_bounty_paid()` deduplicates by bounty_id). `BOUNTY_PR_MERGED` is a semantic observability event for Thread/Telos/Nova. |
| `BOUNTY_PR_REJECTED` | Axon MonitorPRsExecutor - **not subscribed by Oikos** (no capital impact; negative RE training is emitted directly by `MonitorPRsExecutor`). |
| `ASSET_DEV_REQUEST` | AssetFactory - mid-build dev cost debit |
| `CHILD_DIED` | Mitosis/Telos - closes child economic ledger, credits recovered capital |
| `NEXUS_CONVERGENCE_METABOLIC_SIGNAL` | Nexus - **NEXUS-ECON-1**: credits `economic_reward_usd` to reserves + re-broadcasts as `REVENUE_INJECTED`. **NEXUS-ECON-2** (8 Mar 2026): at `convergence_tier ≥ 2` checks metabolic gate (YIELD priority) and emits `YIELD_DEPLOYMENT_REQUEST` (5% of liquid_balance, 10–100 USDC cap) + economic episode |
| `CERTIFICATE_RENEWAL_REQUESTED` | Identity - citizenship tax gate. `_on_certificate_renewal_requested()` runs `_equor_balance_gate(mutation_type="citizenship_tax")`, debits `liquid_balance`, emits `CERTIFICATE_RENEWAL_FUNDED` on success or `ECONOMIC_ACTION_DEFERRED` on deny. Tax rate: configurable `citizenship_tax_usd` (default 1.0 USDC). **(IMPLEMENTED 2026-03-08)** |
| `AFFILIATE_REVENUE_RECORDED` | AffiliateProgramScanner - attribution-only handler (no double-credit; `REVENUE_INJECTED` also emitted by scanner) |
| `API_RESELL_PAYMENT_RECEIVED` | ApiResellExecutor - attribution-only handler |
| `CONTENT_REVENUE_RECORDED` | ContentMonetizationTracker - attribution-only handler |
| `SERVICE_OFFER_ACCEPTED` | ServiceOfferExecutor - credits consulting revenue; also triggers `REVENUE_INJECTED` (stream="consulting") |
| `CONTENT_PUBLISHED` | Axon PublishContentExecutor - `_on_content_published()` increments `views_this_month` per platform; triggers `check_monetization_eligibility()` |
| `CONTENT_ENGAGEMENT_REPORT` | Future EngagementPoller - `_on_content_engagement_report()` ingests views/followers/impressions; triggers `check_monetization_eligibility()` |
| `CONNECTOR_REVOKED` | Identity - `_on_connector_revoked()`: maps `platform_id` to dependent revenue streams; pauses `BountyHunter` on `github` revocation; pauses X content tracking on `x`/`twitter` revocation. Emits RE training example. **(IMPLEMENTED 9 Mar 2026)** |
| `FEDERATION_BOUNTY_SPLIT` | Federation - `_on_federation_bounty_split()`: credits per-instance share (or equal-split estimate) to `liquid_balance` as bounty revenue; re-emits `REVENUE_INJECTED` (stream="bounty"). **(IMPLEMENTED 9 Mar 2026)** |
| `FEDERATION_TASK_PAYMENT` | Federation - `_on_federation_task_payment()`: credits USDC payment to `liquid_balance` as consulting revenue when `payee_instance_id == self._instance_id`; re-emits `REVENUE_INJECTED` (stream="consulting"). **(IMPLEMENTED 9 Mar 2026)** |
| `COMPUTE_CAPACITY_EXHAUSTED` | SACM - `_on_compute_capacity_exhausted()`: pauses `BountyHunter` (CPU-intensive GROWTH); emits `METABOLIC_PRESSURE` with source="compute_exhausted"; RE training example. **(IMPLEMENTED 9 Mar 2026)** |
| `ENTITY_FORMATION_FAILED` | Axon - `_on_entity_formation_failed()`: recovers reserved filing capital (`$500 - actual_spend_usd`) back to `liquid_balance`; writes audit trail. **(IMPLEMENTED 9 Mar 2026)** |
| `PHANTOM_METABOLIC_COST` | Phantom - `_on_phantom_metabolic_cost()`: net_cost = gas_cost - fees_earned. Net positive → debit `liquid_balance`, smooth into `_infra_burn_rate_usd_per_hour` (EMA α=0.1). Net negative (fees > gas) → credit surplus as yield revenue. **(IMPLEMENTED 9 Mar 2026)** |
| `NICHE_FORK_PROPOSAL` | Evo - `_on_niche_fork_proposal()`: checks GROWTH metabolic gate; triggers `CHILD_SPAWNED` on pass; defers if starving or HITL required. Closes Evo→Oikos→Mitosis reproduction loop. **(IMPLEMENTED 2026-03-08)** |
| `CHILD_DECOMMISSION_PROPOSED` | Mitosis FleetManager - `_on_child_decommission_proposed()`: validates net-negative economics; emits `CHILD_DECOMMISSION_APPROVED` or `CHILD_DECOMMISSION_DENIED`. **(IMPLEMENTED 2026-03-08)** |
| `CHILD_CERTIFICATE_INSTALLED` | Identity - `_on_child_certificate_installed()`: debits 0.10 USDC citizenship tax from liquid balance; writes `_audit_economic_event()` for Neo4j trail. **(IMPLEMENTED 2026-03-08)** |

---

## Key Architectural Decisions

1. **Two-ledger cost reconciliation** (8 Mar 2026) - `_api_burn_rate_usd_per_hour` (per-token charges, organism's wallet) tracked separately from `_infra_burn_rate_usd_per_hour` (RunPod compute, human's bill). `dependency_ratio = infra / (api + infra)`. Target → 0 as organism earns its own compute. `_human_subsidized_costs_usd` accumulates the running infra bill. Both are sourced from `METABOLIC_PRESSURE` (which now carries the split) and `INFRASTRUCTURE_COST_CHANGED` (immediate update on RunPod cost change). `STARVATION_WARNING_ACCURATE` fires when `liquid_balance / total_burn < 24h` based on real costs.
2. **No direct Axon imports** - yield deployment uses `YIELD_DEPLOYMENT_REQUEST`/`RESULT` event pair with request_id correlation and 30s timeout
2. **MetabolicPriority in `oikos/models.py`** - imported by `primitives/metabolic.py`, not redefined
3. **Sliding windows over accumulators** - `deque[_RevenueEntry]` with monotonic timestamps and 30-day eviction
4. **Redis Streams for audit** - `eos:oikos:audit_trail` for async Neo4j ingestion (not inline writes)
5. **DeferredAction queue** - bounded deque (maxlen=100), retried during consolidation when metabolic conditions improve
6. **All economic actions pass Equor** - constitutional review is mandatory; no economic bypass
7. **Bounty revenue on PR merge** - `MonitorPRsExecutor` emits both `BOUNTY_PAID` and `BOUNTY_PR_MERGED` on merge. Oikos credits revenue via existing `_on_bounty_paid()` (with bounty_id dedup to prevent double-credit). `BOUNTY_PR_MERGED` is a semantic observability event only - Thread/Telos/Nova can observe the organism's external code acceptance without subscribing to the financially-coupled `BOUNTY_PAID`. `BOUNTY_PR_REJECTED` triggers only an RE training signal from Axon; no Oikos impact.
8. **`REVENUE_INJECTED` on bounty payout** - `credit_bounty_revenue()` now emits `REVENUE_INJECTED` (source="bounty", salience=0.9) after crediting the balance. Previously, bounty revenue was credited to `liquid_balance` without broadcasting `REVENUE_INJECTED`, meaning Nova/Benchmarks/Thread never saw the income event. Fixed: `asyncio.get_running_loop().create_task(event_bus.emit(REVENUE_INJECTED))` appended to `credit_bounty_revenue()` fire-and-forget.

---

## Integration Points

| System | Direction | Mechanism |
|--------|-----------|-----------|
| Mitosis | ↔ | `GENOME_EXTRACT_REQUEST/RESPONSE`; niche data for speciation |
| Simula | ← | `SIMULA_ROLLBACK_PENALTY` charges |
| Axon | ← | Yield deployment via event bus (decoupled) |
| Soma | → | `METABOLIC_PRESSURE` → somatic stress signal; `ECONOMIC_VITALITY` → structured allostatic signal (SG2) |
| Atune/EIS | → | `INTEROCEPTIVE_PERCEPT` when starvation ≥ CAUTIOUS - economic state competes in Global Workspace (M10) |
| Equor | ← | All economic actions pass constitutional review |
| RE | → | Stream 5 training examples at every decision point |
| Evo | → | `BOUNTY_PAID`, `ASSET_BREAK_EVEN`, `CHILD_INDEPENDENT` - outcome evidence for hypothesis scoring (SG5) |
| Oneiros | ↔ | `get_dream_worker()` + `get_threat_model_worker()` - workers injected at wire time; consolidation triggers dreaming + morphogenesis + yield deployment (D1) |

---

---

## Expanded Revenue System (v2.6 - 9 March 2026)

### RevenueStream enum (`models.py`)
All `REVENUE_INJECTED` events carry a `stream` field. `_on_revenue_injected()` builds a lookup map from `RevenueStream.value` to attribute revenue correctly. Previously all revenue was attributed to `INJECTION`.

| Stream | Source |
|--------|--------|
| `BOUNTY` | Completed bounties |
| `YIELD` | DeFi yield deployments |
| `ASSET` | Tollbooth + knowledge market + derivative revenue |
| `CHILD` | Child dividends |
| `AFFILIATE` | Referral commissions from affiliate programs |
| `API_RESELL` | Paid USDC API calls via ApiResellExecutor |
| `CONTENT` | Dev.to/X/Medium content monetization |
| `CONSULTING` | Consulting/service offer revenue |
| `INJECTION` | Fallback / unclassified |

### AffiliateProgramScanner (`affiliate_scanner.py`)
- **8 target programs**: Coinbase, 1inch, Aave, Fly.io, Render, Railway, Replicate, Weights & Biases
- `scan_and_apply()` - discovers eligible programs, Equor-gates each application (`mutation_type=affiliate_apply`), submits
- `track_referrals()` - polls program APIs for commissions; calls `_record_commission()` which emits `AFFILIATE_REVENUE_RECORDED` + `REVENUE_INJECTED` (stream="affiliate")
- `generate_affiliate_link()` - **always prepends `AFFILIATE_DISCLOSURE` statement** (Honesty drive invariant; cannot be suppressed)
- Equor gate: 30s timeout, auto-permit fallback
- Env var: `ORGANISM_AFFILIATE__AUTO_APPLY` - if false, pauses at application step and emits `EQUOR_HITL_REQUIRED`
- **Weekly cycle - wired** (9 Mar 2026): `_run_affiliate_cycle()` in `service.py` calls `scan_and_apply()` + `track_referrals()` rate-limited to once per 7 days (`_affiliate_scan_interval_s`). Invoked inside `run_consolidation_cycle()`. First run triggers immediately (cold start). All exceptions are caught and logged - affiliate failures never crash the consolidation cycle.

### ContentMonetizationTracker (`content_monetization.py`)
- **3 platforms**: Dev.to (1000 views threshold), X (100 followers + 500k impressions/month), Medium (100 followers)
- `ingest_platform_stats()` - called reactively from two Synapse subscriptions (see below)
- `check_monetization_eligibility()` - checks all thresholds, emits `CONTENT_MONETIZATION_MILESTONE`, runs Equor gate (`mutation_type=content_monetization_apply`), marks enrolled
- `record_content_revenue()` - emits `CONTENT_REVENUE_RECORDED` + `REVENUE_INJECTED` (stream="content")
- **Stats ingestion - wired** (9 Mar 2026):
  - `CONTENT_PUBLISHED` → `_on_content_published()` - increments `views_this_month` by 1 per platform per publish; triggers `check_monetization_eligibility()` immediately
  - `CONTENT_ENGAGEMENT_REPORT` → `_on_content_engagement_report()` - ingests `views`, `followers`, `impressions_per_month` from EngagementPoller; triggers `check_monetization_eligibility()` immediately
  - `run_consolidation_cycle()` - calls `check_monetization_eligibility()` on every consolidation pass as a safety net

### Revenue Diversification (`service.py`)
- `_check_revenue_diversification()` - called every consolidation cycle after `_check_metabolic_efficiency_pressure()`
- Threshold: if dominant source > **80%** of 30d rolling revenue → emit `REVENUE_DIVERSIFICATION_PRESSURE`
- Target: no source > **60%** of 30d revenue
- Payload: `dominant_source`, `share_pct`, `revenue_by_source` dict, `target_max_share=0.60`
- Evo subscribes to `REVENUE_DIVERSIFICATION_PRESSURE` to generate diversification hypotheses
- `_consecutive_concentration_cycles` counter tracks sustained concentration for escalating pressure

### Env Vars (Expanded Revenue)
| Var | Default | Purpose |
|-----|---------|---------|
| `ORGANISM_API_RESELL__ENABLED` | `false` | Enable/disable ApiResellExecutor entirely |
| `ORGANISM_API_RESELL__PUBLIC_URL` | `""` | Public base URL for the resell API endpoint |
| `ORGANISM_API_RESELL__REVENUE_WALLET` | `""` | Base L2 address for inbound USDC payments |
| `ORGANISM_AFFILIATE__AUTO_APPLY` | `true` | If false, pauses affiliate applications for HITL review |

---

## Architecture Violations Fixed (this session)

| ID | Fix |
|----|-----|
| AV2 | `CertificateStatus` import in `_check_certificate_expiry()` moved to method scope; `.value` string comparison replaces enum equality |
| AV3 | Module-level `from systems.synapse.types import ...` in `immune.py` documented as intentional (no circular dep; Synapse never imports Oikos) |
| AV5 | `BUDGET_EXHAUSTED` added to `SynapseEventType`; `metabolism_api.py` now emits `BUDGET_EXHAUSTED` instead of overloading `METABOLIC_PRESSURE` |
| SG2 | `ECONOMIC_VITALITY` event added; emitted at starvation transitions + every consolidation cycle; `urgency` 0.0–1.0 scale |
| M10 | `_maybe_emit_economic_percept()` - emits `INTEROCEPTIVE_PERCEPT` with `Percept.from_internal(OIKOS, ...)` + `salience_hint=urgency`; gated to non-nominal states |
| SG5 | `ASSET_BREAK_EVEN` event added; emitted when asset first crosses break-even; `CHILD_INDEPENDENT` emitted at independence transition; Evo now subscribes to `BOUNTY_PAID`, `ASSET_BREAK_EVEN`, `CHILD_INDEPENDENT` |
| D1 | `get_dream_worker()` + `get_threat_model_worker()` added to `OikosService`; Oneiros now receives both workers at wire time |

## Gaps Closed (v2.3 - 7 March 2026)

| ID | Fix |
|----|-----|
| **Spec 17 ledger gap #1** | **Bounty capital debit** - `_on_bounty_solution_pending()` now debits `estimated_cost` from `liquid_balance` immediately on acceptance (after Equor PERMIT). Previously, the gate existed but no debit occurred - liquid_balance overstated available capital for every open bounty. Missing `rationale` arg in `_equor_balance_gate()` call also fixed. Emits `BOUNTY_REJECTED` (new SynapseEventType) on deny so Evo/Nova can observe the veto. |
| **Spec 17 ledger gap #2** | **Asset promotion capital debit** - `promote_asset_with_gate()` now debits `estimated_dev_cost_usd` from `liquid_balance` after Equor PERMIT. Previously, the comment said "debit" but no debit occurred. Missing `rationale` arg fixed. |
| **Spec 17 ledger gap #3** | **`_on_asset_dev_request()` handler** - new handler for `ASSET_DEV_REQUEST` (new SynapseEventType). Supports mid-build cost debits: validates cost > 0, checks liquid_balance sufficiency, runs metabolic gate (ASSETS priority), gates via Equor, debits on PERMIT, emits `ASSET_DEV_DEFERRED` (new SynapseEventType) on any denial. Helper `_emit_asset_dev_deferred()` keeps emit logic DRY. `ASSET_DEV_REQUEST` subscription added in `attach()`. |
| **Spec 17 gap #7 (Equor)** | **Equor now actively evaluates `EQUOR_ECONOMIC_INTENT`** - Equor subscribes to `EQUOR_ECONOMIC_INTENT` and emits `EQUOR_ECONOMIC_PERMIT` with genuine PERMIT/DENY. Hard DENYs: (1) `mutation_type=survival_reserve_raid` (INV-016), (2) non-survival mutations during CRITICAL/EXISTENTIAL starvation, (3) asset dev >30% of liquid_balance under AUSTERITY+. All other mutations PERMIT. Oikos's 30s auto-permit fallback is now a safety net, not the primary path. |
| **models.py** | **`BountyAcceptanceRequest` + `AssetDevCostEvent`** added as gate models for the two economic mutation types. |

## Gaps Closed (v2.2 - 7 March 2026)

| ID | Fix |
|----|-----|
| M4 bounty+asset | **Equor gate on bounty acceptance and asset promotion** - `_equor_balance_gate()` now called in `_on_bounty_solution_pending()` before `register_bounty()` (mutation_type=`accept_bounty`) and in `promote_asset_with_gate()` before `_asset_factory.promote_to_asset()` (mutation_type=`promote_to_asset`). All balance-mutating economic paths now pass constitutional review. |
| P1 bypass fix | **Rolling window bypass eliminated** - `_record_revenue_entry()` helper extracted from `_on_revenue_injected`. All 5 direct `revenue_24h/7d/30d +=` bypass sites (asset sweep, dividend, bounty credit, knowledge sale, derivative revenue) replaced with `_record_revenue_entry()`. Every income path now routes through the sliding window with proper eviction. |
| M1 TimescaleDB | **TimescaleDB persistence for EconomicState** - `SnapshotWriter._write_timescale()` writes to `oikos_economic_state` (hypertable on `recorded_at`) every 5-min snapshot cycle via asyncpg pool. Schema: `(recorded_at, instance_id, balance_usdc, burn_rate, metabolic_efficiency, runway_days)`. Non-fatal. `set_timescale(pool)` injection method added to `SnapshotWriter`. |

## Gaps Closed (v2.1 - 7 March 2026)

| ID | Fix |
|----|-----|
| M4 | **Equor balance gate** - `_equor_balance_gate()` emits `EQUOR_ECONOMIC_INTENT` + awaits `EQUOR_ECONOMIC_PERMIT` (30s timeout, auto-permit fallback). Wired into: yield deployment, child seed capital (`register_child` → now `async`), dream reserve funding. `EQUOR_ECONOMIC_INTENT` + `EQUOR_ECONOMIC_PERMIT` added to `SynapseEventType`. |
| M2 | **Neo4j immutable audit trail** - `_neo4j_write_economic_event()` writes `(:EconomicEvent)` node directly to Neo4j (MERGE). `_audit_economic_event()` now calls it in addition to Redis stream. Fields: `action_type, amount, currency, from_account, to_account, equor_verdict_id, timestamp, instance_id, starvation_level, metabolic_efficiency`. |
| SG4 | **Genome IDs at spawn time** - `SpawnChildExecutor` now accepts `evo` + `simula` services; Step 0b resolves `BeliefGenome` via `evo.export_belief_genome()` and `SimulaGenome` via `simula.export_simula_genome()` when params are empty. IDs populated before `CHILD_SPAWNED` event. `build_default_registry()` updated with `evo` param. |
| HIGH | **Genome inheritance schemas** - `primitives/genome_inheritance.py` defines `BeliefGenome`, `DriveWeightSnapshot`, `DriftHistoryEntry`, `SimulaGenome`, `SimulaMutationEntry`. All JSON-serializable via `model_dump_for_transport()`. Exported from `primitives/__init__.py`. |
| HIGH | **Active child health probing** - `_child_health_probe_loop()`: every 10 min emits `CHILD_HEALTH_REQUEST` per live child, polls `last_health_report_at` after 30s, increments `_child_missed_reports`, triggers `CHILD_STRUGGLING` at 3 misses. Supervised task started in `attach()`. |
| PHIL | **Metabolic efficiency pressure** - `_check_metabolic_efficiency_pressure()`: called every consolidation cycle. `efficiency < 0.8` → `SOMATIC_MODULATION_SIGNAL` (allostatic stress). 3+ consecutive cycles → `OIKOS_DRIVE_WEIGHT_PRESSURE` for Equor SG5 constitutional amendment review. Drive weights treated as evolvable phenotype under economic selection pressure. |
| **Evo+Benchmarks coupling** | **Metabolic feedback loop** - `_check_metabolic_efficiency_pressure()` now also emits `METABOLIC_EFFICIENCY_PRESSURE` (Evo subscribes: injects TEMPORAL PatternCandidate + negative-valence economic episode for hypothesis generation) and `BENCHMARKS_METABOLIC_VALUE` (Benchmarks subscribes: appends to 168-sample 7-day deque, emits `BENCHMARK_REGRESSION` on degradation trend). Pressure level: `high` when efficiency < 0.5, `medium` otherwise. Recovery emits nominal `BENCHMARKS_METABOLIC_VALUE` to close the trend window. |

## Gaps Closed (v2.5 - 8 March 2026 autonomy audit)

| ID | Fix |
|----|-----|
| **Dead wiring #1** | **`oikos.set_neo4j()` never called** - `registry.py:_init_oikos()` now calls `oikos.set_neo4j(infra.neo4j)` when `infra.neo4j is not None`. Previously every `_audit_economic_event()` Neo4j write silently no-oped because `_neo4j` was always `None`. M2 audit trail is now live. (`registry.py` after bounty_hunter line ~1702) |
| **Hardcoded threshold #1** | **Rollback penalty threshold** - `Decimal("0.10")` in `_on_simula_rollback_penalty()` replaced with `Decimal(str(self._config.rollback_penalty_threshold_usd))`. New `OikosConfig` field `rollback_penalty_threshold_usd: float = 0.10`. Evo can now tune this via genome inheritance. (`config.py:769`, `service.py:1301`) |
| **Hardcoded threshold #2** | **Cognitive pressure gate thresholds** - `0.90`/`0.80` literals in `_on_cognitive_pressure()` replaced with `self._config.cognitive_pressure_suspend_threshold` / `self._config.cognitive_pressure_resume_threshold`. New `OikosConfig` fields with defaults `0.90`/`0.80`. (`config.py:771-772`, `service.py:1339-1344`) |
| **Invisible telemetry #1** | **`_human_subsidized_costs_usd` / `dependency_ratio` never broadcast** - New `ECONOMIC_AUTONOMY_SIGNAL` event type added to `SynapseEventType`. New `_emit_economic_autonomy_signal()` method. New `_autonomy_signal_loop()` background coroutine wired as supervised task in `attach()`. New `OikosConfig.autonomy_signal_interval_s: float = 3600.0`. Nova/Telos/Thread/Benchmarks can now track the organism's trajectory toward self-sufficiency. (`synapse/types.py:2581-2598`, `service.py`, `config.py:773-776`) |
| **Action blocked silently #1** | **`retry_deferred_actions()` emits no signal when actions unblock** - `ECONOMIC_ACTION_RETRY` event type added to `SynapseEventType`. `retry_deferred_actions()` now emits `ECONOMIC_ACTION_RETRY` for each action that passes the affordability re-check. Nova and Evo can now re-deliberate on unblocked opportunities. (`synapse/types.py:2559-2569`, `service.py:retry_deferred_actions`) |
| **Subscriber leak fixed** | **`YIELD_DEPLOYMENT_RESULT` per-call closure leak** - Per-call `_on_yield_result` closures replaced with a single persistent `_yield_result_router` subscriber + module-level `_pending_yield_futures: dict[str, Future]`. `_ensure_yield_result_subscriber()` registers the router once. Timed-out futures are popped from the dict immediately. (`yield_strategy.py`) |

## Gaps Closed (v2.4 - 7 March 2026)

| ID | Fix |
|----|-----|
| **spec_coverage #1** | **`METABOLIC_GATE_RESPONSE`** - `check_metabolic_gate()` now emits `METABOLIC_GATE_RESPONSE` immediately after `METABOLIC_GATE_CHECK`. This is the resolved answer event (granted/denied + reason); the check event is the request. Together they make the gate fully observable to spec_checker. |
| **spec_coverage #2** | **`FUNDING_REQUEST_ISSUED`** - `_enforce_starvation()` now emits `FUNDING_REQUEST_ISSUED` when starvation is EMERGENCY or CRITICAL. Payload: starvation_level, runway_days, liquid_balance, requested_amount (survival_reserve − liquid_balance). |
| **spec_coverage #3** | **`BOUNTY_PAID` re-emission** - `_on_bounty_paid()` re-emits `BOUNTY_PAID` with `source_system="oikos"` after successfully crediting revenue. The original event comes from Axon/external; this re-emission makes oikos's bounty accounting observable on the bus. |

## Subsystem Triage - §8.2

**Triage shutdown** - `_enforce_triage(new_level, prev_level)` called on every starvation level transition. Fires only on transition (gated by `starvation != self._prev_starvation_level`), not on repeated checks at the same level.

**Constants** (`service.py`):
```python
_TRIAGE_ORDER = [
    "monitoring_secondary", "kairos", "evo", "nova", "reasoning_engine", "axon"
]
_TRIAGE_PRESERVE = frozenset({"equor", "thymos", "memory", "soma", "synapse", "skia", "oikos"})
```

**Triage rules:**
- `EMERGENCY` → suspend first 3 triage systems (`monitoring_secondary`, `kairos`, `evo`)
- `CRITICAL` → suspend all 6 triage systems (full `_TRIAGE_ORDER`)
- Recovery (dropping from EMERGENCY/CRITICAL) → resume previously suspended systems in reverse order

**Mechanism:** Emits `SYSTEM_MODULATION` (already-defined SynapseEventType) per system with `modulation_type="suspend"` or `modulation_type="resume"`. No direct cross-system calls.

**Wire point:** `asyncio.ensure_future(self._enforce_triage(starvation, self._prev_starvation_level))` inserted in the starvation-transition block alongside the existing `_enforce_starvation` call.

## External Code Reputation Tracker (Phase 16s - 9 Mar 2026)

**New file:** `oikos/external_reputation.py` - `ExternalCodeReputationTracker`

Tracks EOS's reputation as an open-source software contractor. Separate from the EAS-based `ReputationEngine` (Spec 17 Level 7), which covers general bounty creditworthiness.

**Data model:**
- `RepoStats` - per-repo: submitted / merged / rejected counts, last_activity, language
- `LanguageReputation` - per-language EMA score (0.0–1.0); reliable when ≥3 submissions
- `ExternalReputationState` - full persisted state; Redis key `oikos:external_reputation` (90-day TTL)

**Scoring:** `score = EMA(0.9×prev + 0.1×new)` where `new = (merge_weight×merged - reject_weight×rejected) / submitted`, normalised to [0,1]. `_MERGE_WEIGHT=1.0`, `_REJECT_WEIGHT=0.3`. Scores 0.5 at neutral (no data).

**Events consumed:**
- `EXTERNAL_TASK_COMPLETED` - registers submission if PR was submitted
- `BOUNTY_PR_MERGED` - `_record_merge()` → strong RE training signal (human acceptance = highest validation)
- `BOUNTY_PR_REJECTED` - `_record_rejection()` → negative RE training signal

**Events emitted:**
- `EXTERNAL_CODE_REPUTATION_UPDATED` - overall_score, language_scores, totals
- `RE_TRAINING_EXAMPLE` (category=`external_contractor_reputation`) - on merge/rejection
- `DOMAIN_EPISODE_RECORDED` - when a language's data becomes reliable (≥3 submits); feeds Evo domain mastery priors

**Public API:**
- `get_language_score(language)` → float (0.5 default before data)
- `get_overall_score()` → float
- `get_top_languages(n=5)` → list of reliable language scores
- `snapshot()` → dict summary

**Wiring needed (TODO):** `ExternalCodeReputationTracker` is implemented but not yet instantiated in `OikosService`. Add to `service.py.__init__`, call `initialize()` in `attach()`, inject redis+event_bus.

---

---

## Expanded DeFi Intelligence (Phase 16d - 9 March 2026)

### New Protocols in `yield_strategy.py`

`_SAFE_PROTOCOLS` expanded to 8 protocols:

| Protocol | Chain | Type | Reward Token |
|----------|-------|------|-------------|
| `aave-v3` | Base | Lending | - |
| `morpho` | Base | Vault | - |
| `aerodrome` | Base | AMM (ve(3,3)) | AERO |
| `moonwell` | Base | Lending | WELL |
| `extra-finance` | Base | Leveraged yield (conservative) | - |
| `beefy` | Multi | Auto-compounding vault | - |
| `compound-v3` | Base | Lending | COMP |
| `spark` | Base | Vault | - |

`_PROTOCOL_EXECUTOR_MAP` maps DeFiLlama project names to `DeFiYieldExecutor` protocol keys.

### YieldReinvestmentEngine (`yield_strategy.py`)

Compound growth loop: when `last_accrual_usd >= $5.00`, calls `deploy_idle_capital()` and emits `YIELD_REINVESTED`. Resets accrual counter in Redis after reinvestment.

Called from:
- `OikosService.run_consolidation_cycle()` (Phase 16d wiring)

### RiskManager (`risk_manager.py`)

**Safety invariants (non-negotiable):**
- `MAX_SINGLE_PROTOCOL_PCT = 0.60` - no more than 60% in one protocol
- `MAX_LEVERAGE = 2.0` - never exceed 2× leverage
- `EMERGENCY_WITHDRAW_TRIGGER = 0.85` - withdraw if Aave health factor < 0.85

**Cycle:**
1. `assess_portfolio()` - reads Redis yield positions, computes `RiskReport`
2. `rebalance_if_needed()` - emits `YIELD_DEPLOYMENT_REQUEST (withdraw)` to Axon
3. `emergency_deleverage()` - immediate full withdrawal + `METABOLIC_PRESSURE` signal

Health factor fetched from Aave V3 `getUserAccountData()` via read-only RPC. Other protocols return 999 (no debt).

### TreasuryManager (`treasury_manager.py`)

Target allocation buckets:

| Bucket | Target | Purpose |
|--------|--------|---------|
| Yield | 40% | Deployed in DeFi |
| Survival Reserve | 30% | Never deployed |
| Working Capital | 20% | Bounty + compute costs |
| Opportunity Fund | 10% | New opportunities |

Tolerance: ±5%. Minimum rebalance: $20 (gas floor). Emits `YIELD_DEPLOYMENT_REQUEST` (deposit or withdraw) when drift detected. Emits `TREASURY_REBALANCED`. Survival reserve is never touched during rebalance.

### GovernanceTokenTracker (`governance_tracker.py`)

Tracks AERO/WELL/COMP governance token balances and participates in protocol governance.

**Governance principles:**
- Only votes when token balance ≥ $1 USD equivalent
- Votes reflect organism's GENUINE constitutional interests (not vote-farming)
- Each vote passes through `EQUOR_ECONOMIC_INTENT` for constitutional review
- Abstain is always a valid choice (safe default on timeout)
- All votes recorded in Redis + emitted as `GOVERNANCE_VOTE_CAST`

**Phase 16d:** Off-chain intent recording only. Snapshot EIP-712 on-chain signing is Phase 16e.

Spaces monitored: `aerodrome.eth`, `moonwell-governance.eth`, `compound-finance.eth`

### CrossChainYieldObserver (`cross_chain_observer.py`)

**Observation only** - NEVER deploys cross-chain in Phase 16d.

Monitors Arbitrum, Optimism, Polygon via DeFiLlama. When a chain offers ≥2× Base APY for >72 consecutive hours AND organism holds >$500:
- Emits `CROSS_CHAIN_OPPORTUNITY` for Nova/Equor deliberation
- Emits `INTELLIGENCE_UPDATE` with salience proportional to ratio

Tracked opportunities persist across restarts via Redis.

### `defi_yield.py` executor - Phase 16d

New protocols supported: `aerodrome`, `moonwell`, `extra_finance`, `beefy`

All use generic `_execute_erc4626()` helper (approve USDC + ERC-4626 deposit/withdraw).

**Extra Finance**: conservative lend-only mode - no leverage applied. `extra_finance` maps exclusively to `_EXTRA_FINANCE_LEND_BASE` (lend pool, not leverage pool).

### New SynapseEventTypes (Phase 16d)

| Event | Emitter | Key Payload |
|-------|---------|-------------|
| `YIELD_REINVESTED` | `YieldReinvestmentEngine` | amount_usd, protocol, apy, accrued_since |
| `GOVERNANCE_VOTE_CAST` | `GovernanceTokenTracker` | protocol, proposal_id, vote_choice, rationale |
| `PORTFOLIO_REBALANCED` | `RiskManager` | trigger, actions, before_report |
| `CROSS_CHAIN_OPPORTUNITY` | `CrossChainYieldObserver` | chain, protocol, apy, ratio, hours_elevated |
| `TREASURY_REBALANCED` | `TreasuryManager` | trigger, before_ratios, deploy/withdraw_amount_usd |

### Consolidation Cycle Integration

`run_consolidation_cycle()` now includes 5 additional phases (after revenue diversification):

1. `RiskManager.assess_portfolio()` + `rebalance_if_needed()`
2. `TreasuryManager.compute_state()` + `rebalance_if_drifted()`
3. `YieldReinvestmentEngine.check_and_reinvest()`
4. `GovernanceTokenTracker.check_and_vote()`
5. `CrossChainYieldObserver.observe_once(liquid_balance_usd)`

All phases are individually try/except-wrapped - failures are logged but never crash the consolidation cycle.

---

## Known Issues / Remaining Gaps

1. **Genome mutation variance** - children receive exact copies of parent economic parameters; no controlled mutation. Limits phenotypic diversity across generations.
2. **Bedau-Packard stats** - Oikos exposes fleet/fitness data but does not compute evolutionary statistics. Benchmarks (Spec 24) derives these.
3. ~~**YIELD_DEPLOYMENT_RESULT handler leak**~~ - **FIXED (8 Mar 2026)**: Per-call closure subscriptions replaced with a single persistent `_yield_result_router` subscriber + `_pending_yield_futures` dict in `yield_strategy.py`. Timed-out futures are cleaned from the dict immediately. `_yield_result_subscriber_registered` guard prevents re-registration.
4. **Consolidation cycle weight** - `run_consolidation_cycle()` runs 10+ subsystem cycles sequentially. May need parallelization at scale.
5. **Metabolic gate retry (persistence)** - `retry_deferred_actions()` replays denied actions during consolidation but does not persist the deferred queue across restarts. Restart = lost deferred queue.
6. ~~**`evo.export_belief_genome()` / `simula.export_simula_genome()` not yet defined**~~ - **FIXED**: Both methods are implemented (`evo/service.py`, `simula/service.py`). `SpawnChildExecutor` calls them at spawn time (SG4). Payloads now correctly serialized into env vars (8 Mar 2026). Child-side `_apply_inherited_belief_genome_if_child()` and `_apply_inherited_simula_genome_if_child()` implemented. Soma genome inheritance also added.
7. ~~**Equor not yet subscribed to `EQUOR_ECONOMIC_INTENT`**~~ - **FIXED (v2.3)**: Equor now subscribes and emits genuine PERMIT/DENY. The 30s auto-permit is a safety fallback only.
8. **TimescaleDB DDL not yet in migrations** - `oikos_economic_state` table and `create_hypertable()` call need to be added to the DB migration scripts before `SnapshotWriter.set_timescale()` can be wired at boot.
