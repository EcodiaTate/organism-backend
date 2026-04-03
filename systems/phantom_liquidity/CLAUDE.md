# Phantom Liquidity - System CLAUDE.md

**Spec**: `.claude/EcodiaOS_Spec_28_PhantomLiquidity.md`
**System ID**: `phantom_liquidity`
**Last Updated**: 2026-03-08 (v1.3 - autonomy audit gaps closed)

---

## What's Implemented

### Core Modules

| Module | Status | Notes |
|---|---|---|
| `types.py` | ✅ Complete | `PoolHealth`, `PhantomLiquidityPool`, `PhantomPriceFeed`, `PoolSelectionCandidate` - all use `EOSBaseModel` |
| `pool_selector.py` | ✅ Complete | Static 5-pool curated list; `select_pools()` with TVL filter + budget cap; `compute_tick_range()` with ±80%/±50% spread; `get_static_pools()` public API |
| `price_listener.py` | ✅ Complete | `eth_getLogs` polling; `_decode_swap_data` (all 5 Swap event fields, signed int256/int24); `sqrt_price_x96_to_price` formula; graceful degradation |
| `executor.py` | ✅ Complete | `mint_position()` approve×2 + mint + receipt parse; `burn_position()` decreaseLiquidity → collect → burn; `_parse_mint_receipt` uses **full 32-byte** `IncreaseLiquidity` topic |
| `service.py` | ✅ Complete | Full lifecycle; `get_price()` staleness-aware; `get_price_with_fallback()` CoinGecko; `maintenance_cycle()` staleness + IL; Synapse emission; `get_candidates()` public API |

### Synapse Events

All 7 `PHANTOM_*` events implemented and registered in `synapse/types.py`:
- `PHANTOM_PRICE_UPDATE` - per Swap event
- `PHANTOM_POOL_STALE` - no swaps > `staleness_threshold_s`
- `PHANTOM_POSITION_CRITICAL` - IL > threshold
- `PHANTOM_IL_DETECTED` - IL change during maintenance
- `PHANTOM_FALLBACK_ACTIVATED` - CoinGecko fallback used
- `PHANTOM_RESOURCE_EXHAUSTED` - EMERGENCY/CRITICAL metabolic pressure
- `PHANTOM_METABOLIC_COST` - per maintenance cycle (gas, fees, net P&L)

### Synapse Subscriptions

- `METABOLIC_PRESSURE` → `_on_metabolic_pressure()` - AUSTERITY warns + emits `PHANTOM_RESOURCE_EXHAUSTED`; EMERGENCY/CRITICAL also emits `NOVA_INTENT_REQUESTED` to autonomously withdraw highest-IL pool (v1.3)
- `GENOME_EXTRACT_REQUEST` → `_on_genome_extract_request()` → `GENOME_EXTRACT_RESPONSE` with pool configs + thresholds
- `EVO_ADJUST_BUDGET` → `_on_evo_adjust_budget()` - Evo can tune `il_rebalance_threshold`, `staleness_threshold_s`, `consensus_window_s`, `swap_poll_interval_s` at runtime; bounds enforced; emits `PHANTOM_PARAMETER_ADJUSTED` (v1.3)

### Genome Extraction Protocol (Mitosis)

`extract_genome_segment()` / `seed_from_genome_segment()` implemented. Exports: pool addresses, pair configs, fee tiers, tick ranges, capital allocation, staleness/IL/poll thresholds + SHA-256 payload hash.

### Oikos Integration

Direct call (acceptable per Spec §9): `register_phantom_position()`, `update_phantom_position()`, `remove_phantom_position()`. Uses `YieldPosition` from `systems.oikos.models`.

### TimescaleDB Persistence

`write_phantom_price` → `phantom_price_history` table. `get_phantom_price_history` for REST API.

### REST API

12 endpoints in `api/routers/phantom_liquidity.py`, wired in `main.py`. Bonus: `/defillama-pools`, `/tick-range`.

---

## What's Missing (Known Gaps)

### MEDIUM

- **Two executors with no clear division**: `systems/phantom_liquidity/executor.py` (used by service/router) vs `axon/executors/phantom_liquidity.py` (Axon Intent-driven, currently zero runtime callers). Both use the correct NPM address. Not blocking but creates maintenance surface.

- **LP key rotation**: `store_lp_key()` seals the key at provisioning time but does not implement periodic rotation or re-sealing under a new vault key version. Key rotation would require calling `vault.rotate_key()` and updating `_lp_key_envelope`.

### Closed Gaps (2026-03-08 - Autonomy Audit)

- ~~**EVO_ADJUST_BUDGET not subscribed**~~ - `_on_evo_adjust_budget()` wired in `attach()`; 4 runtime-tunable parameters with bounds: `il_rebalance_threshold` [0.5%–10%], `staleness_threshold_s` [60s–3600s], `consensus_window_s` [10s–120s], `swap_poll_interval_s` [1s–30s]. Emits `PHANTOM_PARAMETER_ADJUSTED` for Evo feedback.
- ~~**Hard block on IL breach with no recourse**~~ - `_on_price_update()` now detects IL threshold breach on every swap (not just hourly maintenance); immediately emits `NOVA_INTENT_REQUESTED` for autonomous withdrawal. EMERGENCY/CRITICAL metabolic pressure handler also emits `NOVA_INTENT_REQUESTED`.
- ~~**RE training annotation missing**~~ - `RE_TRAINING_EXAMPLE` emitted on every IL breach event with full causal context (pair, entry/current price, IL %, capital at risk, threshold). Closes Spec §23 annotation gap.
- ~~**PHANTOM_PRICE_UPDATE invisible to Nova/Atune**~~ - Nova now subscribes to `PHANTOM_PRICE_UPDATE` (via `set_synapse()`) and updates market-price beliefs. IL risk > 2% in price update triggers `_immediate_deliberation()`.
- ~~**NOVA_INTENT_REQUESTED event type missing**~~ - Added to `SynapseEventType` in `synapse/types.py`. Nova subscribes via `_on_nova_intent_requested()`; fires `_immediate_deliberation()` so any system can trigger Nova deliberation without bypassing Equor.
- ~~**PHANTOM_PARAMETER_ADJUSTED event type missing**~~ - Added to `SynapseEventType` in `synapse/types.py`. Evo subscribes to confirm hypothesis outcomes.

### Closed Gaps

- ~~Identity/wallet key management~~ - `store_lp_key()` / `retrieve_lp_key()` via `IdentityVault`. Never in config/env.
- ~~TimescaleDB → Memory bridge~~ - `_write_price_observation_to_neo4j()` writes `(:PriceObservation)` nodes per swap event.
- ~~Multi-instance price consensus~~ - `PHANTOM_PRICE_OBSERVATION` + `_compute_consensus_price()` (2σ median, 30s window).
- ~~Genome round-trip validation~~ - 4 pytest tests in `tests/unit/systems/phantom_liquidity/test_genome_roundtrip.py`.
- ~~Bedau-Packard contribution~~ - `PHANTOM_SUBSTRATE_OBSERVABLE` emitted per maintenance cycle.

---

## Architecture Notes

- **No cross-system imports at runtime** - all inter-system comms via Synapse events or direct Oikos method call (explicitly permitted in Spec §9)
- `from systems.oikos.models import YieldPosition` in `service.py:register_pool()` creates a cross-system type dependency - acceptable per spec but worth monitoring
- Router no longer imports private `_STATIC_POOLS` - uses `svc.get_candidates()` or `PoolSelector.get_static_pools()`
- `PoolHealth` comparisons in `pool_selector.py` now use enum values, not raw strings

---

## Integration Map

| System | Channel | Direction | What |
|---|---|---|---|
| Synapse | `PHANTOM_PRICE_UPDATE` | → emit | Every swap event; Atune/Nova/Oikos subscribe |
| Synapse | `PHANTOM_POOL_STALE` | → emit | Pool health degraded; Thymos may subscribe |
| Synapse | `PHANTOM_POSITION_CRITICAL` | → emit | IL > 2%; Thymos/Nova may subscribe |
| Synapse | `PHANTOM_RESOURCE_EXHAUSTED` | → emit | EMERGENCY metabolic pressure |
| Synapse | `PHANTOM_METABOLIC_COST` | → emit | Hourly cost/fee report |
| Synapse | `PHANTOM_PRICE_OBSERVATION` | → emit | Raw swap observation for fleet consensus |
| Synapse | `PHANTOM_SUBSTRATE_OBSERVABLE` | → emit | Bedau-Packard evolutionary metrics per maintenance cycle |
| Synapse | `METABOLIC_PRESSURE` | ← consume | Oikos sends; AUSTERITY warns; EMERGENCY/CRITICAL proposes withdrawal via `NOVA_INTENT_REQUESTED` |
| Synapse | `GENOME_EXTRACT_REQUEST` | ← consume | Mitosis requests; responds with `GENOME_EXTRACT_RESPONSE` |
| Synapse | `PHANTOM_PRICE_OBSERVATION` | ← consume | Peer observations; aggregated for fleet consensus |
| Synapse | `EVO_ADJUST_BUDGET` | ← consume | Evo tunes thresholds at runtime; emits `PHANTOM_PARAMETER_ADJUSTED` |
| Synapse | `NOVA_INTENT_REQUESTED` | → emit | IL breach + EMERGENCY pressure - asks Nova to autonomously withdraw |
| Synapse | `PHANTOM_PARAMETER_ADJUSTED` | → emit | After Evo parameter adjustment (Evo feedback) |
| Synapse | `RE_TRAINING_EXAMPLE` | → emit | On IL breach - price/IL causal context for RE training |
| Oikos | `register_phantom_position()` | → direct call | Position lifecycle tracking |
| TimescaleDB | `phantom_price_history` | → write | Per swap event persistence |
| Neo4j | `(:PriceObservation)` nodes | → write | Per swap event - Memory bridge for Kairos/Memory |
| IdentityVault | `store_lp_key()` / `retrieve_lp_key()` | → call | LP wallet key sealed at rest; decrypted per on-chain op |
