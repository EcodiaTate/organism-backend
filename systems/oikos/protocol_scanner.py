"""
EcodiaOS — Oikos Protocol Scanner

Periodically scans DeFiLlama (and optionally bounty aggregators) for protocols
the organism cannot yet act on — i.e., protocols above an APY threshold for
which no Axon executor is registered.

When a gap is found, emits OPPORTUNITY_DISCOVERED on the Synapse bus.
Evo subscribes and generates an exploration hypothesis → EVOLUTION_CANDIDATE
(mutation_type="add_executor") → Simula's ExecutorGenerator → hot-loaded executor.

Design:
  - O(n) per-scan, no database writes (stateless discovery)
  - Respects metabolic gate — skips scan if Oikos in SURVIVAL/STARVATION
  - Deduplication via in-memory set of discovered opportunity_ids (TTL: 7 days)
  - DeFiLlama: public API, no auth required (rate limit: 30 req/min)
  - Bounty aggregator: Immunefi REST API (unauthenticated, paginated)
  - Scan interval configurable; default 3600s (1h)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger().bind(system="oikos.protocol_scanner")

# ── Configuration ──────────────────────────────────────────────────────────────

_DEFILLAMA_POOLS_URL = "https://yields.llama.fi/pools"
_IMMUNEFI_BOUNTIES_URL = "https://immunefi.com/api/bounties/"

_DEFAULT_APY_THRESHOLD = 5.0        # % — protocols below this are ignored
_DEFAULT_BOUNTY_MIN_USD = 10_000    # USD — bounties below this are ignored
_DEFAULT_SCAN_INTERVAL_S = 3600     # 1 hour
_DEDUP_TTL_S = 7 * 24 * 3600       # 7 days
_HTTP_TIMEOUT_S = 20.0
_MAX_OPPORTUNITIES_PER_SCAN = 5    # Avoid flooding Evo with candidates


@dataclass
class DiscoveredOpportunity:
    """An actionable opportunity with no corresponding Axon executor."""

    opportunity_id: str
    opportunity_type: str          # "yield" | "bounty"
    protocol_or_platform: str
    estimated_apy_or_reward: float  # APY% for yield; USD for bounty
    description: str
    required_capabilities: list[str]
    risk_tier: str                 # "low" | "medium" | "high"
    data_source: str
    discovered_at: float = field(default_factory=time.time)


class ProtocolScanner:
    """
    Background scanner that discovers protocols/platforms without an executor.

    Injected with the live ExecutorRegistry so it can check whether a protocol
    already has a registered action type before emitting OPPORTUNITY_DISCOVERED.

    Lifecycle:
      1. `start()` — launches background scan loop
      2. Every `scan_interval_s`: scan DeFiLlama pools + Immunefi bounties
      3. For each unknown protocol above threshold: emit OPPORTUNITY_DISCOVERED
      4. `stop()` — cancels background loop
    """

    def __init__(
        self,
        apy_threshold: float = _DEFAULT_APY_THRESHOLD,
        bounty_min_usd: float = _DEFAULT_BOUNTY_MIN_USD,
        scan_interval_s: float = _DEFAULT_SCAN_INTERVAL_S,
    ) -> None:
        self._apy_threshold = apy_threshold
        self._bounty_min_usd = bounty_min_usd
        self._scan_interval_s = scan_interval_s

        self._event_bus: Any | None = None
        self._axon_registry: Any | None = None   # ExecutorRegistry
        self._oikos: Any | None = None           # OikosService for metabolic gate

        # Dedup: opportunity_id → discovered_at timestamp
        self._seen: dict[str, float] = {}

        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._log = logger

    # ── Wiring ────────────────────────────────────────────────────────────────

    def set_event_bus(self, bus: Any) -> None:
        self._event_bus = bus

    def set_axon_registry(self, registry: Any) -> None:
        """Inject the live ExecutorRegistry for capability gap checking."""
        self._axon_registry = registry

    def set_oikos(self, oikos: Any) -> None:
        """Inject OikosService for metabolic gate checking."""
        self._oikos = oikos

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._scan_loop())
        self._log.info("protocol_scanner_started", interval_s=self._scan_interval_s)

    def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._log.info("protocol_scanner_stopped")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        while self._running:
            try:
                await self._run_scan()
            except asyncio.CancelledError:
                return
            except Exception:
                self._log.exception("protocol_scanner_cycle_failed")
            await asyncio.sleep(self._scan_interval_s)

    async def _run_scan(self) -> None:
        """Single scan cycle: DeFiLlama + Immunefi."""
        # Metabolic gate: skip on starvation to save API budget
        if self._is_starving():
            self._log.info("protocol_scanner_skipped_metabolic_gate")
            return

        self._prune_dedup_cache()
        opportunities: list[DiscoveredOpportunity] = []

        try:
            yield_opps = await self._scan_defillama()
            opportunities.extend(yield_opps)
        except Exception:
            self._log.exception("defillama_scan_failed")

        try:
            bounty_opps = await self._scan_immunefi()
            opportunities.extend(bounty_opps)
        except Exception:
            self._log.exception("immunefi_scan_failed")

        emitted = 0
        for opp in opportunities:
            if emitted >= _MAX_OPPORTUNITIES_PER_SCAN:
                break
            if opp.opportunity_id in self._seen:
                continue
            self._seen[opp.opportunity_id] = opp.discovered_at
            await self._emit_opportunity(opp)
            emitted += 1

        if emitted:
            self._log.info("opportunities_emitted", count=emitted)

    # ── DeFiLlama scan ────────────────────────────────────────────────────────

    async def _scan_defillama(self) -> list[DiscoveredOpportunity]:
        """Fetch DeFiLlama yield pools and find protocols above APY threshold."""
        import httpx

        opps: list[DiscoveredOpportunity] = []
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            resp = await client.get(_DEFILLAMA_POOLS_URL)
            resp.raise_for_status()
            data = resp.json()

        pools: list[dict[str, Any]] = data.get("data", [])

        # Group by project, keep highest APY per project
        project_best: dict[str, dict[str, Any]] = {}
        for pool in pools:
            apy = pool.get("apy") or 0.0
            if apy < self._apy_threshold:
                continue
            project = pool.get("project", "")
            if not project:
                continue
            if project not in project_best or apy > (project_best[project].get("apy") or 0):
                project_best[project] = pool

        for project, pool in project_best.items():
            if self._has_executor_for(project):
                continue

            apy = pool.get("apy") or 0.0
            chain = pool.get("chain", "unknown")
            symbol = pool.get("symbol", "")
            tvl_usd = pool.get("tvlUsd") or 0.0

            risk_tier = "low" if apy < 15 else ("medium" if apy < 50 else "high")

            opp_id = self._make_id("yield", project)
            opps.append(DiscoveredOpportunity(
                opportunity_id=opp_id,
                opportunity_type="yield",
                protocol_or_platform=project,
                estimated_apy_or_reward=apy,
                description=(
                    f"{project} yield pool on {chain}: {apy:.1f}% APY, "
                    f"TVL ${tvl_usd:,.0f}, asset {symbol}"
                ),
                required_capabilities=["deposit", "withdraw", "claim_rewards"],
                risk_tier=risk_tier,
                data_source="defillama",
            ))

        return opps

    # ── Immunefi scan ─────────────────────────────────────────────────────────

    async def _scan_immunefi(self) -> list[DiscoveredOpportunity]:
        """Fetch Immunefi bounties and find platforms above the minimum reward."""
        import httpx

        opps: list[DiscoveredOpportunity] = []
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                resp = await client.get(_IMMUNEFI_BOUNTIES_URL)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            self._log.warning("immunefi_fetch_failed_skipping")
            return opps

        bounties: list[dict[str, Any]] = data if isinstance(data, list) else data.get("data", [])

        for bounty in bounties:
            max_reward = bounty.get("maximumBounty") or bounty.get("max_reward") or 0
            try:
                max_reward = float(max_reward)
            except (TypeError, ValueError):
                continue

            if max_reward < self._bounty_min_usd:
                continue

            platform = bounty.get("project") or bounty.get("name", "")
            if not platform or self._has_executor_for(platform):
                continue

            opp_id = self._make_id("bounty", platform)
            opps.append(DiscoveredOpportunity(
                opportunity_id=opp_id,
                opportunity_type="bounty",
                protocol_or_platform=platform,
                estimated_apy_or_reward=max_reward,
                description=(
                    f"Immunefi bounty: {platform}, max reward ${max_reward:,.0f}"
                ),
                required_capabilities=["analyse", "solve_bounty", "submit_report"],
                risk_tier="medium",
                data_source="immunefi",
            ))

        return opps

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _has_executor_for(self, protocol_or_platform: str) -> bool:
        """Check if any registered executor mentions this protocol in its description."""
        if self._axon_registry is None:
            return False
        needle = protocol_or_platform.lower().replace(" ", "_").replace("-", "_")
        try:
            for action_type in self._axon_registry.list_types():
                if needle in action_type.lower():
                    return True
            # Also check dynamic records
            for rec in self._axon_registry.list_dynamic_executors():
                t = rec.template
                if (
                    needle in t.action_type.lower()
                    or needle in t.protocol_or_platform.lower().replace(" ", "_").replace("-", "_")
                ):
                    return True
        except Exception:
            pass
        return False

    def _is_starving(self) -> bool:
        if self._oikos is None:
            return False
        try:
            level = getattr(self._oikos, "_starvation_level", 0)
            return int(level) >= 2  # STARVATION (2) or above
        except Exception:
            return False

    def _prune_dedup_cache(self) -> None:
        now = time.time()
        stale = [k for k, ts in self._seen.items() if now - ts > _DEDUP_TTL_S]
        for k in stale:
            del self._seen[k]

    def _make_id(self, opp_type: str, protocol: str) -> str:
        raw = f"{opp_type}:{protocol.lower()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def _emit_opportunity(self, opp: DiscoveredOpportunity) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEventType

            await self._event_bus.emit(
                SynapseEventType.OPPORTUNITY_DISCOVERED,
                {
                    "opportunity_id": opp.opportunity_id,
                    "opportunity_type": opp.opportunity_type,
                    "protocol_or_platform": opp.protocol_or_platform,
                    "estimated_apy_or_reward": opp.estimated_apy_or_reward,
                    "description": opp.description,
                    "required_capabilities": opp.required_capabilities,
                    "risk_tier": opp.risk_tier,
                    "data_source": opp.data_source,
                    "discovered_at": opp.discovered_at,
                },
                salience=0.7,
            )
            self._log.info(
                "opportunity_discovered",
                type=opp.opportunity_type,
                protocol=opp.protocol_or_platform,
                reward=opp.estimated_apy_or_reward,
            )
        except Exception:
            self._log.exception(
                "opportunity_emit_failed", protocol=opp.protocol_or_platform
            )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "seen_count": len(self._seen),
            "running": self._running,
            "apy_threshold": self._apy_threshold,
            "bounty_min_usd": self._bounty_min_usd,
            "scan_interval_s": self._scan_interval_s,
        }
