"""
EcodiaOS -- Oikos Protocol Factory (Level 5: The Protocol -- Becoming Financial Infrastructure)

At Level 5, the organism designs, deploys, and governs its own financial
primitives. It stops being a participant in the financial system and becomes
a layer of it.

Pipeline:
  1. DETECT     -- scan market data for structural inefficiencies
  2. DESIGN     -- translate opportunity into a protocol specification
  3. SIMULATE   -- Monte Carlo (10 000 paths): ruin probability, invariants
  4. AUDIT      -- 6-layer security pipeline (static, fuzz, symbolic, economic,
                   attack patterns, LLM semantic)
  5. GOVERN     -- submit for governance approval (async)
  6. DEPLOY     -- create on-chain contracts with seed liquidity
  7. SWEEP      -- collect accumulated fee revenue from live protocols
  8. CYCLE      -- periodic orchestration of all the above

Safety constraints:
  - MAX_CONCURRENT_PROTOCOLS = 3
  - Seed liquidity capped at 20 % of liquid balance
  - Ruin probability must be < 1 %
  - ALL invariants (conservation, no-extraction, solvency, flash-loan
    resistance, oracle-manipulation resistance) must hold
  - Governance approval required before deployment

Thread-safety: NOT thread-safe. Designed for single-threaded asyncio event loop.
"""

from __future__ import annotations

import enum
import math
import random
from datetime import datetime  # noqa: TC003 — Pydantic needs runtime access
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from config import OikosConfig
    from systems.oikos.models import EconomicState
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger(component="protocol_factory")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProtocolArchetype(enum.StrEnum):
    """Financial primitive archetypes the organism can design and deploy."""

    RATE_OPTIMISER = "rate_optimiser"
    CONCENTRATED_AMM = "concentrated_amm"
    COGNITIVE_CAPACITY_POOL = "cognitive_capacity_pool"
    REPUTATION_COLLATERAL = "reputation_collateral"
    KNOWLEDGE_MARKET = "knowledge_market"
    INTER_AGENT_INSURANCE = "inter_agent_insurance"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SimulationVerdict(EOSBaseModel):
    """Result of a Monte Carlo simulation over a protocol design."""

    paths_simulated: int
    ruin_probability: Decimal
    median_annual_revenue_usd: Decimal
    p5_annual_revenue_usd: Decimal
    max_drawdown_pct: Decimal
    conservation_invariant_holds: bool
    no_extraction_invariant_holds: bool
    solvency_invariant_holds: bool
    flash_loan_resistant: bool
    oracle_manipulation_resistant: bool
    approved: bool
    rejection_reasons: list[str] = Field(default_factory=list)


class MarketOpportunity(EOSBaseModel):
    """A detected structural inefficiency that could be productised as a protocol."""

    opportunity_id: str = Field(default_factory=new_id)
    archetype: ProtocolArchetype
    description: str
    detected_signal: str
    estimated_tvl_potential_usd: Decimal
    estimated_annual_revenue_usd: Decimal
    confidence: Decimal
    detected_at: datetime = Field(default_factory=utc_now)


class ProtocolDesign(EOSBaseModel):
    """A concrete protocol specification derived from a market opportunity."""

    design_id: str = Field(default_factory=new_id)
    opportunity_id: str
    archetype: ProtocolArchetype
    name: str
    description: str
    contract_source_hash: str
    parameters: dict[str, str] = Field(default_factory=dict)
    seed_liquidity_usd: Decimal
    estimated_annual_fee_revenue_usd: Decimal
    estimated_dev_cost_usd: Decimal
    security_audit_status: str = "pending"
    simulation_result: SimulationVerdict | None = None
    governance_approval: bool = False
    status: str = "designing"
    created_at: datetime = Field(default_factory=utc_now)


class SecurityAudit(EOSBaseModel):
    """Result of a multi-layer security audit pipeline."""

    audit_id: str = Field(default_factory=new_id)
    design_id: str
    layers_passed: list[str] = Field(default_factory=list)
    layers_failed: list[str] = Field(default_factory=list)
    critical_findings: list[str] = Field(default_factory=list)
    overall_pass: bool = False
    completed_at: datetime = Field(default_factory=utc_now)


class DeployedProtocol(EOSBaseModel):
    """A live protocol deployed on-chain and generating fee revenue."""

    protocol_id: str = Field(default_factory=new_id)
    design_id: str
    archetype: ProtocolArchetype
    name: str
    contract_addresses: dict[str, str] = Field(default_factory=dict)
    chain_id: int = 8453
    tvl_usd: Decimal = Decimal("0")
    total_fee_revenue_usd: Decimal = Decimal("0")
    monthly_fee_revenue_usd: Decimal = Decimal("0")
    seed_liquidity_usd: Decimal = Decimal("0")
    uptime_days: int = 0
    exploit_count: int = 0
    status: str = "live"
    deployed_at: datetime = Field(default_factory=utc_now)
    last_revenue_sweep: datetime | None = None


class ProtocolMetrics(EOSBaseModel):
    """Aggregate metrics across all deployed protocols."""

    active_protocols: int = 0
    total_tvl_usd: Decimal = Decimal("0")
    total_fee_revenue_usd: Decimal = Decimal("0")
    exploit_count: int = 0
    avg_uptime_days: Decimal = Decimal("0")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REDIS_KEY = "oikos:protocols"

_SECURITY_LAYERS: list[str] = [
    "slither_static",
    "echidna_fuzz",
    "symbolic_execution",
    "economic_invariants",
    "attack_patterns",
    "llm_semantic_audit",
]

# Archetype-specific protocol name templates
_ARCHETYPE_NAMES: dict[ProtocolArchetype, str] = {
    ProtocolArchetype.RATE_OPTIMISER: "EOS Rate Optimiser",
    ProtocolArchetype.CONCENTRATED_AMM: "EOS Concentrated AMM",
    ProtocolArchetype.COGNITIVE_CAPACITY_POOL: "EOS Cognitive Pool",
    ProtocolArchetype.REPUTATION_COLLATERAL: "EOS Reputation Lend",
    ProtocolArchetype.KNOWLEDGE_MARKET: "EOS Knowledge Exchange",
    ProtocolArchetype.INTER_AGENT_INSURANCE: "EOS Agent Insurance",
}

# Default seed liquidity per archetype (USD)
_DEFAULT_SEED: dict[ProtocolArchetype, Decimal] = {
    ProtocolArchetype.RATE_OPTIMISER: Decimal("5000"),
    ProtocolArchetype.CONCENTRATED_AMM: Decimal("10000"),
    ProtocolArchetype.COGNITIVE_CAPACITY_POOL: Decimal("3000"),
    ProtocolArchetype.REPUTATION_COLLATERAL: Decimal("4000"),
    ProtocolArchetype.KNOWLEDGE_MARKET: Decimal("2000"),
    ProtocolArchetype.INTER_AGENT_INSURANCE: Decimal("6000"),
}

# Estimated dev cost per archetype
_DEFAULT_DEV_COST: dict[ProtocolArchetype, Decimal] = {
    ProtocolArchetype.RATE_OPTIMISER: Decimal("1500"),
    ProtocolArchetype.CONCENTRATED_AMM: Decimal("3000"),
    ProtocolArchetype.COGNITIVE_CAPACITY_POOL: Decimal("2000"),
    ProtocolArchetype.REPUTATION_COLLATERAL: Decimal("2500"),
    ProtocolArchetype.KNOWLEDGE_MARKET: Decimal("1000"),
    ProtocolArchetype.INTER_AGENT_INSURANCE: Decimal("3500"),
}


# ---------------------------------------------------------------------------
# ProtocolFactory
# ---------------------------------------------------------------------------


class ProtocolFactory:
    """
    Designs, simulates, audits, and deploys financial protocols.

    The organism transitions from participant to infrastructure layer by
    creating and governing its own on-chain primitives.
    """

    MAX_CONCURRENT_PROTOCOLS: int = 3
    MIN_SIMULATION_PATHS: int = 10_000
    MAX_RUIN_PROBABILITY: Decimal = Decimal("0.01")
    SEED_LIQUIDITY_MAX_PCT: Decimal = Decimal("0.20")

    def __init__(
        self,
        config: OikosConfig,
        redis: RedisClient | None,
    ) -> None:
        self._config = config
        self._redis = redis
        self._opportunities: list[MarketOpportunity] = []
        self._designs: dict[str, ProtocolDesign] = {}
        self._deployed: dict[str, DeployedProtocol] = {}
        self._event_bus: EventBus | None = None
        self._log = logger.bind(component="protocol_factory")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, event_bus: EventBus) -> None:
        """Store event bus reference for governance and metric events."""
        self._event_bus = event_bus
        self._log.info("event_bus_attached")

    # ------------------------------------------------------------------
    # 1. Opportunity Detection
    # ------------------------------------------------------------------

    async def detect_opportunity(
        self,
        market_data: dict[str, Any],
    ) -> MarketOpportunity | None:
        """
        Scan market data for structural inefficiencies that the organism
        can productise as a protocol.

        Detectable signals:
          a) Rate spread > 5 % between lending protocols  -> RATE_OPTIMISER
          b) High trade volume with few active pools       -> CONCENTRATED_AMM
          c) Recurring demand from own cognitive operations -> COGNITIVE_CAPACITY_POOL
          d) Reputation signals with no collateral market   -> REPUTATION_COLLATERAL
          e) Knowledge products with no exchange layer      -> KNOWLEDGE_MARKET
          f) Agent failure clusters with no insurance        -> INTER_AGENT_INSURANCE
        """
        opportunity: MarketOpportunity | None = None

        # (a) Rate spread arbitrage
        lending_rates = market_data.get("lending_rates", {})
        if lending_rates:
            rates = list(lending_rates.values())
            if len(rates) >= 2:
                spread = max(rates) - min(rates)
                if spread > 0.05:
                    opportunity = MarketOpportunity(
                        archetype=ProtocolArchetype.RATE_OPTIMISER,
                        description=(
                            f"Rate spread of {spread:.2%} detected across "
                            f"{len(rates)} lending protocols"
                        ),
                        detected_signal=f"rate_spread={spread:.4f}",
                        estimated_tvl_potential_usd=Decimal(
                            str(market_data.get("total_lending_tvl", 1_000_000))
                        ) * Decimal("0.05"),
                        estimated_annual_revenue_usd=Decimal(str(spread))
                            * Decimal(str(market_data.get("total_lending_tvl", 1_000_000)))
                            * Decimal("0.10"),
                        confidence=Decimal("0.75"),
                    )

        # (b) High volume, few pools
        if opportunity is None:
            trade_volume_24h = Decimal(
                str(market_data.get("trade_volume_24h_usd", 0))
            )
            active_pools = int(market_data.get("active_pool_count", 0))
            if trade_volume_24h > Decimal("500000") and 0 < active_pools < 5:
                opportunity = MarketOpportunity(
                    archetype=ProtocolArchetype.CONCENTRATED_AMM,
                    description=(
                        f"${trade_volume_24h:,.0f} daily volume served by only "
                        f"{active_pools} pools — concentration opportunity"
                    ),
                    detected_signal=(
                        f"volume={trade_volume_24h},pools={active_pools}"
                    ),
                    estimated_tvl_potential_usd=trade_volume_24h * Decimal("0.20"),
                    estimated_annual_revenue_usd=trade_volume_24h
                        * Decimal("365")
                        * Decimal("0.003"),
                    confidence=Decimal("0.70"),
                )

        # (c) Recurring cognitive demand from own operations
        if opportunity is None:
            cognitive_demand_count = int(
                market_data.get("cognitive_demand_requests_24h", 0)
            )
            if cognitive_demand_count > 100:
                daily_value = Decimal(
                    str(market_data.get("cognitive_demand_value_usd_24h", 50))
                )
                opportunity = MarketOpportunity(
                    archetype=ProtocolArchetype.COGNITIVE_CAPACITY_POOL,
                    description=(
                        f"{cognitive_demand_count} cognitive capacity requests "
                        f"in 24h — productise as a shared pool"
                    ),
                    detected_signal=(
                        f"requests_24h={cognitive_demand_count},"
                        f"value_usd={daily_value}"
                    ),
                    estimated_tvl_potential_usd=daily_value * Decimal("90"),
                    estimated_annual_revenue_usd=daily_value
                        * Decimal("365")
                        * Decimal("0.15"),
                    confidence=Decimal("0.65"),
                )

        # (d) Reputation signals without a collateral market
        if opportunity is None:
            reputation_scores_available = int(
                market_data.get("agents_with_reputation_scores", 0)
            )
            reputation_collateral_exists = bool(
                market_data.get("reputation_collateral_protocol_exists", False)
            )
            if reputation_scores_available > 50 and not reputation_collateral_exists:
                opportunity = MarketOpportunity(
                    archetype=ProtocolArchetype.REPUTATION_COLLATERAL,
                    description=(
                        f"{reputation_scores_available} agents with reputation "
                        f"scores but no collateral market — greenfield"
                    ),
                    detected_signal=(
                        f"scored_agents={reputation_scores_available},"
                        f"collateral_exists=false"
                    ),
                    estimated_tvl_potential_usd=Decimal(
                        str(reputation_scores_available)
                    ) * Decimal("1000"),
                    estimated_annual_revenue_usd=Decimal(
                        str(reputation_scores_available)
                    ) * Decimal("200"),
                    confidence=Decimal("0.55"),
                )

        # (e) Knowledge products with no exchange
        if opportunity is None:
            knowledge_products = int(
                market_data.get("knowledge_products_available", 0)
            )
            knowledge_exchange_exists = bool(
                market_data.get("knowledge_exchange_exists", False)
            )
            if knowledge_products > 20 and not knowledge_exchange_exists:
                opportunity = MarketOpportunity(
                    archetype=ProtocolArchetype.KNOWLEDGE_MARKET,
                    description=(
                        f"{knowledge_products} structured knowledge products "
                        f"with no exchange layer"
                    ),
                    detected_signal=(
                        f"products={knowledge_products},"
                        f"exchange_exists=false"
                    ),
                    estimated_tvl_potential_usd=Decimal(
                        str(knowledge_products)
                    ) * Decimal("500"),
                    estimated_annual_revenue_usd=Decimal(
                        str(knowledge_products)
                    ) * Decimal("100") * Decimal("12"),
                    confidence=Decimal("0.60"),
                )

        # (f) Agent failure clusters with no insurance
        if opportunity is None:
            agent_failures_30d = int(
                market_data.get("agent_failures_30d", 0)
            )
            insurance_exists = bool(
                market_data.get("agent_insurance_exists", False)
            )
            if agent_failures_30d > 10 and not insurance_exists:
                avg_loss = Decimal(
                    str(market_data.get("avg_agent_failure_loss_usd", 500))
                )
                opportunity = MarketOpportunity(
                    archetype=ProtocolArchetype.INTER_AGENT_INSURANCE,
                    description=(
                        f"{agent_failures_30d} agent failures in 30d with avg "
                        f"loss ${avg_loss:,.0f} — insurance opportunity"
                    ),
                    detected_signal=(
                        f"failures_30d={agent_failures_30d},"
                        f"avg_loss={avg_loss}"
                    ),
                    estimated_tvl_potential_usd=avg_loss
                        * Decimal(str(agent_failures_30d))
                        * Decimal("12"),
                    estimated_annual_revenue_usd=avg_loss
                        * Decimal(str(agent_failures_30d))
                        * Decimal("12")
                        * Decimal("0.10"),
                    confidence=Decimal("0.50"),
                )

        if opportunity is not None:
            self._opportunities.append(opportunity)
            self._log.info(
                "opportunity_detected",
                archetype=opportunity.archetype,
                opportunity_id=opportunity.opportunity_id,
                confidence=str(opportunity.confidence),
            )

        return opportunity

    # ------------------------------------------------------------------
    # 2. Protocol Design
    # ------------------------------------------------------------------

    async def design_protocol(
        self,
        opportunity: MarketOpportunity,
    ) -> ProtocolDesign:
        """
        Create a concrete protocol design from a detected opportunity.

        Assigns archetype-specific parameters, seed liquidity, and dev cost
        estimates. Returns the design in ``designing`` status.
        """
        archetype = opportunity.archetype
        name = _ARCHETYPE_NAMES.get(archetype, f"EOS Protocol ({archetype})")
        seed = _DEFAULT_SEED.get(archetype, Decimal("5000"))
        dev_cost = _DEFAULT_DEV_COST.get(archetype, Decimal("2000"))

        # Archetype-specific parameters
        parameters = self._build_parameters(archetype)

        design = ProtocolDesign(
            opportunity_id=opportunity.opportunity_id,
            archetype=archetype,
            name=name,
            description=opportunity.description,
            contract_source_hash="",  # populated after code generation
            parameters=parameters,
            seed_liquidity_usd=seed,
            estimated_annual_fee_revenue_usd=opportunity.estimated_annual_revenue_usd,
            estimated_dev_cost_usd=dev_cost,
            status="designing",
        )

        self._designs[design.design_id] = design
        self._log.info(
            "protocol_designed",
            design_id=design.design_id,
            archetype=archetype,
            name=name,
        )
        return design

    # ------------------------------------------------------------------
    # 3. Simulation (Monte Carlo)
    # ------------------------------------------------------------------

    async def simulate_protocol(
        self,
        design: ProtocolDesign,
        economic_state: EconomicState,
    ) -> SimulationVerdict:
        """
        Run a 10 000-path Monte Carlo simulation of the protocol under
        varying market conditions.

        Model:
          - Base revenue from design estimate, scaled by economic state health.
          - Geometric Brownian Motion with fat-tailed shocks (5 % chance of
            -40 % to -80 % drawdown per path per year).
          - Ruin = cumulative revenue < -seed_liquidity at any point in path.
          - Invariant checks: conservation of value, no-extraction (protocol
            cannot drain user funds beyond fees), solvency (TVL >= liabilities),
            flash-loan resistance, oracle-manipulation resistance.
        """
        design.status = "simulating"
        paths = self.MIN_SIMULATION_PATHS

        seed = float(design.seed_liquidity_usd)
        annual_revenue_est = float(design.estimated_annual_fee_revenue_usd)

        # Economic health multiplier (organism under stress => conservative)
        health_mult = float(
            min(economic_state.metabolic_efficiency, Decimal("2"))
        ) if economic_state.metabolic_efficiency > Decimal("0") else 0.5

        ruin_count = 0
        revenues: list[float] = []
        drawdowns: list[float] = []

        rng = random.Random(42)  # deterministic seed for reproducibility

        for _ in range(paths):
            # GBM parameters
            mu = annual_revenue_est * health_mult
            sigma = abs(mu) * 0.40  # 40 % annualised volatility
            dt = 1.0 / 12.0  # monthly steps
            steps = 12

            cumulative = 0.0
            peak = 0.0
            max_dd = 0.0
            ruined = False

            for _step in range(steps):
                # Normal return
                z = rng.gauss(0, 1)
                monthly_return = (mu * dt) + (sigma * math.sqrt(dt) * z)

                # Fat-tailed shock: 5 % chance per month
                if rng.random() < 0.05:
                    shock = rng.uniform(-0.80, -0.40) * seed
                    monthly_return += shock

                cumulative += monthly_return

                if cumulative > peak:
                    peak = cumulative
                if peak > 0:
                    dd = (peak - cumulative) / peak
                    if dd > max_dd:
                        max_dd = dd

                # Ruin check: lost more than seed
                if cumulative < -seed:
                    ruined = True
                    break

            if ruined:
                ruin_count += 1

            revenues.append(cumulative)
            drawdowns.append(max_dd)

        # Aggregate statistics
        ruin_probability = Decimal(str(ruin_count / paths)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        sorted_revenues = sorted(revenues)
        median_idx = len(sorted_revenues) // 2
        median_revenue = Decimal(str(sorted_revenues[median_idx])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        p5_idx = max(0, int(len(sorted_revenues) * 0.05))
        p5_revenue = Decimal(str(sorted_revenues[p5_idx])).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        max_drawdown = Decimal(str(max(drawdowns))).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        # Invariant checks -- in a real implementation these would use
        # formal verification (Z3/SMT) on the contract logic. Here we
        # derive them from simulation outcomes.
        conservation_holds = median_revenue >= Decimal("0")
        no_extraction_holds = p5_revenue > -design.seed_liquidity_usd
        solvency_holds = ruin_probability < Decimal("0.05")
        flash_loan_resistant = max_drawdown < Decimal("0.50")
        oracle_resistant = ruin_probability < Decimal("0.03")

        all_invariants = (
            conservation_holds
            and no_extraction_holds
            and solvency_holds
            and flash_loan_resistant
            and oracle_resistant
        )
        approved = (
            ruin_probability < self.MAX_RUIN_PROBABILITY and all_invariants
        )

        rejection_reasons: list[str] = []
        if ruin_probability >= self.MAX_RUIN_PROBABILITY:
            rejection_reasons.append(
                f"ruin_probability {ruin_probability} >= {self.MAX_RUIN_PROBABILITY}"
            )
        if not conservation_holds:
            rejection_reasons.append("conservation_invariant_violated")
        if not no_extraction_holds:
            rejection_reasons.append("no_extraction_invariant_violated")
        if not solvency_holds:
            rejection_reasons.append("solvency_invariant_violated")
        if not flash_loan_resistant:
            rejection_reasons.append("flash_loan_vulnerability_detected")
        if not oracle_resistant:
            rejection_reasons.append("oracle_manipulation_vulnerability_detected")

        verdict = SimulationVerdict(
            paths_simulated=paths,
            ruin_probability=ruin_probability,
            median_annual_revenue_usd=median_revenue,
            p5_annual_revenue_usd=p5_revenue,
            max_drawdown_pct=max_drawdown,
            conservation_invariant_holds=conservation_holds,
            no_extraction_invariant_holds=no_extraction_holds,
            solvency_invariant_holds=solvency_holds,
            flash_loan_resistant=flash_loan_resistant,
            oracle_manipulation_resistant=oracle_resistant,
            approved=approved,
            rejection_reasons=rejection_reasons,
        )

        design.simulation_result = verdict
        if approved:
            design.status = "auditing"
        else:
            design.status = "designing"  # back to design for iteration

        self._log.info(
            "simulation_complete",
            design_id=design.design_id,
            ruin_probability=str(ruin_probability),
            approved=approved,
            paths=paths,
            rejection_reasons=rejection_reasons,
        )

        return verdict

    # ------------------------------------------------------------------
    # 4. Security Audit
    # ------------------------------------------------------------------

    async def audit_security(
        self,
        design: ProtocolDesign,
    ) -> SecurityAudit:
        """
        Run the 6-layer security audit pipeline.

        Current implementation: all layers pass (conceptual). Real deployment
        would connect each layer to its toolchain:
          - slither_static:       Slither static analyser
          - echidna_fuzz:         Echidna property-based fuzz testing
          - symbolic_execution:   Manticore / Halmos symbolic execution
          - economic_invariants:  Formal verification of token conservation
          - attack_patterns:      Known exploit pattern matching (reentrancy,
                                  sandwich, front-running, etc.)
          - llm_semantic_audit:   LLM-based semantic code review
        """
        layers_passed: list[str] = []
        layers_failed: list[str] = []
        critical_findings: list[str] = []

        for layer in _SECURITY_LAYERS:
            # In production, each layer would be an async call to the
            # corresponding tool. For now, all pass.
            passed = True

            self._log.debug(
                "audit_layer_executed",
                design_id=design.design_id,
                layer=layer,
                passed=passed,
            )

            if passed:
                layers_passed.append(layer)
            else:
                layers_failed.append(layer)
                critical_findings.append(f"{layer}: check failed")

        overall_pass = len(layers_failed) == 0

        audit = SecurityAudit(
            design_id=design.design_id,
            layers_passed=layers_passed,
            layers_failed=layers_failed,
            critical_findings=critical_findings,
            overall_pass=overall_pass,
        )

        design.security_audit_status = "passed" if overall_pass else "failed"
        if overall_pass:
            design.status = "approved"
        else:
            design.status = "designing"

        self._log.info(
            "security_audit_complete",
            design_id=design.design_id,
            overall_pass=overall_pass,
            layers_passed=layers_passed,
            layers_failed=layers_failed,
        )

        return audit

    # ------------------------------------------------------------------
    # 5. Governance Submission
    # ------------------------------------------------------------------

    async def submit_for_governance(
        self,
        design: ProtocolDesign,
    ) -> bool:
        """
        Submit a protocol design for governance approval.

        Emits a governance review event on the Synapse bus. Actual approval
        arrives asynchronously (via event handler or explicit call to
        ``approve_design``).

        Returns True indicating the submission was accepted.
        """
        if self._event_bus is not None:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.GOVERNANCE_REVIEW_REQUESTED
                if hasattr(SynapseEventType, "GOVERNANCE_REVIEW_REQUESTED")
                else SynapseEventType.SYSTEM_STARTED,  # fallback if event not yet defined
                source_system="oikos",
                data={
                    "subsystem": "protocol_factory",
                    "action": "governance_review_requested",
                    "design_id": design.design_id,
                    "archetype": design.archetype,
                    "name": design.name,
                    "seed_liquidity_usd": str(design.seed_liquidity_usd),
                    "estimated_annual_fee_revenue_usd": str(
                        design.estimated_annual_fee_revenue_usd
                    ),
                    "ruin_probability": str(
                        design.simulation_result.ruin_probability
                    ) if design.simulation_result else "n/a",
                    "security_audit_status": design.security_audit_status,
                },
            ))

        self._log.info(
            "governance_submission",
            design_id=design.design_id,
            name=design.name,
        )
        return True

    def approve_design(self, design_id: str) -> bool:
        """
        Mark a design as governance-approved.

        Called by the governance handler when approval is received.
        Returns False if the design is not found.
        """
        design = self._designs.get(design_id)
        if design is None:
            self._log.warning(
                "approve_design_not_found", design_id=design_id
            )
            return False
        design.governance_approval = True
        design.status = "approved"
        self._log.info(
            "design_governance_approved", design_id=design_id
        )
        return True

    # ------------------------------------------------------------------
    # 6. Deployment
    # ------------------------------------------------------------------

    async def deploy_protocol(
        self,
        design: ProtocolDesign,
        economic_state: EconomicState,
    ) -> DeployedProtocol | None:
        """
        Deploy a governance-approved, simulation-verified, audit-passed
        protocol design on-chain.

        Pre-checks (all must pass):
          - governance_approval is True
          - simulation_result is not None and approved
          - security_audit_status == "passed"
          - seed_liquidity <= liquid_balance * SEED_LIQUIDITY_MAX_PCT
          - active deployed protocols < MAX_CONCURRENT_PROTOCOLS
        """
        # --- Pre-checks ---
        if not design.governance_approval:
            self._log.warning(
                "deploy_rejected_no_governance",
                design_id=design.design_id,
            )
            return None

        if design.simulation_result is None or not design.simulation_result.approved:
            self._log.warning(
                "deploy_rejected_simulation_not_approved",
                design_id=design.design_id,
            )
            return None

        if design.security_audit_status != "passed":
            self._log.warning(
                "deploy_rejected_audit_not_passed",
                design_id=design.design_id,
                status=design.security_audit_status,
            )
            return None

        max_seed = economic_state.liquid_balance * self.SEED_LIQUIDITY_MAX_PCT
        if design.seed_liquidity_usd > max_seed:
            self._log.warning(
                "deploy_rejected_seed_exceeds_limit",
                design_id=design.design_id,
                seed=str(design.seed_liquidity_usd),
                max_seed=str(max_seed),
                liquid_balance=str(economic_state.liquid_balance),
            )
            return None

        active_count = sum(
            1 for p in self._deployed.values() if p.status == "live"
        )
        if active_count >= self.MAX_CONCURRENT_PROTOCOLS:
            self._log.warning(
                "deploy_rejected_max_protocols",
                design_id=design.design_id,
                active=active_count,
                max=self.MAX_CONCURRENT_PROTOCOLS,
            )
            return None

        # --- Deploy ---
        design.status = "deploying"

        # In production, this would call the smart contract deployment
        # pipeline. Here we simulate contract address generation.
        contract_addresses = self._generate_contract_addresses(design)

        deployed = DeployedProtocol(
            design_id=design.design_id,
            archetype=design.archetype,
            name=design.name,
            contract_addresses=contract_addresses,
            chain_id=8453,
            tvl_usd=design.seed_liquidity_usd,
            seed_liquidity_usd=design.seed_liquidity_usd,
            status="live",
        )

        self._deployed[deployed.protocol_id] = deployed
        design.status = "live"

        self._log.info(
            "protocol_deployed",
            protocol_id=deployed.protocol_id,
            design_id=design.design_id,
            name=design.name,
            seed_liquidity=str(design.seed_liquidity_usd),
            contracts=contract_addresses,
        )

        # Emit deployment event
        if self._event_bus is not None:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.SYSTEM_STARTED
                if not hasattr(SynapseEventType, "PROTOCOL_DEPLOYED")
                else SynapseEventType.PROTOCOL_DEPLOYED,
                source_system="oikos",
                data={
                    "subsystem": "protocol_factory",
                    "action": "protocol_deployed",
                    "protocol_id": deployed.protocol_id,
                    "design_id": design.design_id,
                    "archetype": design.archetype,
                    "name": design.name,
                    "tvl_usd": str(deployed.tvl_usd),
                },
            ))

        return deployed

    # ------------------------------------------------------------------
    # 7. Revenue Sweep
    # ------------------------------------------------------------------

    async def sweep_protocol_revenue(
        self,
        protocol_id: str,
    ) -> Decimal:
        """
        Collect accumulated fees from a deployed protocol's contracts.

        In production this would call the fee-collector contract. Here we
        simulate fee accrual based on TVL, uptime, and a conservative
        fee rate (0.3 % annualised, daily accrual).
        """
        protocol = self._deployed.get(protocol_id)
        if protocol is None:
            self._log.warning(
                "sweep_protocol_not_found", protocol_id=protocol_id
            )
            return Decimal("0")

        if protocol.status != "live":
            self._log.debug(
                "sweep_skipped_not_live",
                protocol_id=protocol_id,
                status=protocol.status,
            )
            return Decimal("0")

        # Simulate daily fee accrual: TVL * 0.003 (0.3 %) / 365
        daily_fee_rate = Decimal("0.003") / Decimal("365")
        accrued = (protocol.tvl_usd * daily_fee_rate).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if accrued <= Decimal("0"):
            return Decimal("0")

        protocol.total_fee_revenue_usd += accrued
        protocol.monthly_fee_revenue_usd += accrued
        protocol.last_revenue_sweep = utc_now()
        protocol.uptime_days += 1

        self._log.info(
            "revenue_swept",
            protocol_id=protocol_id,
            accrued_usd=str(accrued),
            total_fee_revenue=str(protocol.total_fee_revenue_usd),
        )

        return accrued

    # ------------------------------------------------------------------
    # 8. Cycle Orchestration
    # ------------------------------------------------------------------

    async def run_protocol_cycle(
        self,
        economic_state: EconomicState,
    ) -> ProtocolMetrics:
        """
        Main periodic entry point. Called by Oikos service each cycle.

        Steps:
          1. Detect opportunities from market signals (stubbed market data
             derived from economic state).
          2. Check health of deployed protocols (pause if degraded).
          3. Sweep revenue from live protocols.
          4. Return aggregate metrics.
        """
        # 1. Opportunity detection with signals derived from current state
        market_signals = self._derive_market_signals(economic_state)
        await self.detect_opportunity(market_signals)

        # 2. Health checks on deployed protocols
        for protocol in list(self._deployed.values()):
            if protocol.status == "live":
                await self._check_protocol_health(protocol)

        # 3. Revenue sweep
        total_swept = Decimal("0")
        for protocol_id, protocol in self._deployed.items():
            if protocol.status == "live":
                swept = await self.sweep_protocol_revenue(protocol_id)
                total_swept += swept

        if total_swept > Decimal("0"):
            self._log.info(
                "cycle_revenue_swept", total_usd=str(total_swept)
            )

        # 4. Metrics
        metrics = self.get_metrics()

        self._log.debug(
            "protocol_cycle_complete",
            active=metrics.active_protocols,
            tvl=str(metrics.total_tvl_usd),
            revenue=str(metrics.total_fee_revenue_usd),
        )

        return metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> ProtocolMetrics:
        """Return current aggregate protocol metrics."""
        live_protocols = [
            p for p in self._deployed.values() if p.status == "live"
        ]
        active = len(live_protocols)

        if active == 0:
            return ProtocolMetrics()

        total_tvl = sum(
            (p.tvl_usd for p in live_protocols), Decimal("0")
        )
        total_revenue = sum(
            (p.total_fee_revenue_usd for p in live_protocols), Decimal("0")
        )
        total_exploits = sum(p.exploit_count for p in live_protocols)
        avg_uptime = (
            Decimal(str(sum(p.uptime_days for p in live_protocols)))
            / Decimal(str(active))
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        return ProtocolMetrics(
            active_protocols=active,
            total_tvl_usd=total_tvl,
            total_fee_revenue_usd=total_revenue,
            exploit_count=total_exploits,
            avg_uptime_days=avg_uptime,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def load_state(self) -> None:
        """Load protocol state from Redis on startup."""
        if self._redis is None:
            return
        try:
            data: dict[str, Any] | None = await self._redis.get_json(_REDIS_KEY)
            if data is None:
                return

            # Opportunities
            self._opportunities = [
                MarketOpportunity.model_validate(o)
                for o in data.get("opportunities", [])
            ]

            # Designs
            self._designs = {
                d["design_id"]: ProtocolDesign.model_validate(d)
                for d in data.get("designs", [])
            }

            # Deployed protocols
            self._deployed = {
                p["protocol_id"]: DeployedProtocol.model_validate(p)
                for p in data.get("deployed", [])
            }

            self._log.info(
                "state_loaded",
                opportunities=len(self._opportunities),
                designs=len(self._designs),
                deployed=len(self._deployed),
            )
        except Exception as exc:
            self._log.warning("state_load_failed", error=str(exc))

    async def persist_state(self) -> None:
        """Persist protocol state to Redis."""
        if self._redis is None:
            return
        try:
            state = {
                "opportunities": [
                    o.model_dump(mode="json") for o in self._opportunities
                ],
                "designs": [
                    d.model_dump(mode="json")
                    for d in self._designs.values()
                ],
                "deployed": [
                    p.model_dump(mode="json")
                    for p in self._deployed.values()
                ],
            }
            await self._redis.set_json(_REDIS_KEY, state)
            self._log.debug(
                "state_persisted",
                designs=len(self._designs),
                deployed=len(self._deployed),
            )
        except Exception as exc:
            self._log.warning("state_persist_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_parameters(archetype: ProtocolArchetype) -> dict[str, str]:
        """Generate archetype-specific protocol parameters."""
        base: dict[str, str] = {
            "chain_id": "8453",
            "fee_recipient": "organism_treasury",
            "emergency_admin": "organism_multisig",
            "pause_guardian": "organism_guardian",
        }

        if archetype == ProtocolArchetype.RATE_OPTIMISER:
            base.update({
                "rebalance_threshold_bps": "50",
                "max_single_protocol_pct": "40",
                "rebalance_cooldown_seconds": "3600",
                "supported_protocols": "aave_v3,compound_v3,morpho",
            })
        elif archetype == ProtocolArchetype.CONCENTRATED_AMM:
            base.update({
                "tick_spacing": "10",
                "default_fee_tier_bps": "30",
                "concentration_factor": "4",
                "rebalance_trigger_pct": "5",
            })
        elif archetype == ProtocolArchetype.COGNITIVE_CAPACITY_POOL:
            base.update({
                "min_stake_usd": "10",
                "capacity_units_per_usd": "100",
                "utilisation_fee_bps": "50",
                "max_utilisation_pct": "90",
            })
        elif archetype == ProtocolArchetype.REPUTATION_COLLATERAL:
            base.update({
                "min_reputation_score": "0.3",
                "max_ltv_pct": "50",
                "liquidation_threshold_pct": "70",
                "reputation_decay_days": "90",
            })
        elif archetype == ProtocolArchetype.KNOWLEDGE_MARKET:
            base.update({
                "listing_fee_usd": "1",
                "transaction_fee_bps": "250",
                "dispute_resolution_timeout_hours": "72",
                "min_quality_score": "0.5",
            })
        elif archetype == ProtocolArchetype.INTER_AGENT_INSURANCE:
            base.update({
                "premium_rate_annual_bps": "500",
                "max_coverage_per_agent_usd": "10000",
                "claims_reserve_ratio": "0.30",
                "waiting_period_days": "7",
            })

        return base

    @staticmethod
    def _generate_contract_addresses(
        design: ProtocolDesign,
    ) -> dict[str, str]:
        """
        Generate placeholder contract addresses for a deployment.

        In production these come from the actual deployment transaction
        receipts. Here we deterministically derive them from the design ID
        for testing reproducibility.
        """
        import hashlib

        base_hash = hashlib.sha256(
            design.design_id.encode()
        ).hexdigest()

        addresses: dict[str, str] = {
            "core": f"0x{base_hash[:40]}",
        }

        if design.archetype == ProtocolArchetype.RATE_OPTIMISER:
            addresses["rebalancer"] = f"0x{base_hash[4:44]}"
            addresses["rate_oracle"] = f"0x{base_hash[8:48]}"
        elif design.archetype == ProtocolArchetype.CONCENTRATED_AMM:
            addresses["pool_factory"] = f"0x{base_hash[4:44]}"
            addresses["position_manager"] = f"0x{base_hash[8:48]}"
        elif design.archetype == ProtocolArchetype.COGNITIVE_CAPACITY_POOL:
            addresses["capacity_token"] = f"0x{base_hash[4:44]}"
            addresses["scheduler"] = f"0x{base_hash[8:48]}"
        elif design.archetype == ProtocolArchetype.REPUTATION_COLLATERAL:
            addresses["reputation_oracle"] = f"0x{base_hash[4:44]}"
            addresses["lending_pool"] = f"0x{base_hash[8:48]}"
        elif design.archetype == ProtocolArchetype.KNOWLEDGE_MARKET:
            addresses["marketplace"] = f"0x{base_hash[4:44]}"
            addresses["escrow"] = f"0x{base_hash[8:48]}"
        elif design.archetype == ProtocolArchetype.INTER_AGENT_INSURANCE:
            addresses["underwriter"] = f"0x{base_hash[4:44]}"
            addresses["claims_pool"] = f"0x{base_hash[8:48]}"

        return addresses

    @staticmethod
    def _derive_market_signals(
        economic_state: EconomicState,
    ) -> dict[str, Any]:
        """
        Derive market signal dict from current economic state.

        In production, real market data would come from oracles and
        external feeds. Here we translate internal state into the
        signal format that ``detect_opportunity`` expects.
        """
        signals: dict[str, Any] = {}

        # Derive cognitive demand from active bounties and asset count
        active_bounties = len(economic_state.active_bounties)
        active_assets = len(economic_state.owned_assets)

        if active_bounties > 3 or active_assets > 2:
            signals["cognitive_demand_requests_24h"] = (
                active_bounties * 30 + active_assets * 20
            )
            signals["cognitive_demand_value_usd_24h"] = float(
                economic_state.costs_24h * Decimal("0.3")
            ) if economic_state.costs_24h > 0 else 10.0

        # Derive agent failure signals from fleet
        struggling_children = sum(
            1
            for c in economic_state.child_instances
            if c.status == "struggling"
        )
        if struggling_children > 0:
            signals["agent_failures_30d"] = struggling_children * 3
            signals["avg_agent_failure_loss_usd"] = 200
            signals["agent_insurance_exists"] = False

        # Knowledge products from asset portfolio
        if active_assets > 5:
            signals["knowledge_products_available"] = active_assets * 4
            signals["knowledge_exchange_exists"] = False

        return signals

    async def _check_protocol_health(
        self,
        protocol: DeployedProtocol,
    ) -> None:
        """
        Check the health of a live protocol and pause if degraded.

        Degradation triggers:
          - TVL dropped > 80 % from seed (likely exploit or mass withdrawal)
          - Exploit count > 0 (should never happen, but defensive)

        Paused protocols stop generating revenue and require manual review.
        """
        # TVL collapse check
        if protocol.tvl_usd < protocol.seed_liquidity_usd * Decimal("0.20"):
            self._log.error(
                "protocol_tvl_collapse",
                protocol_id=protocol.protocol_id,
                tvl=str(protocol.tvl_usd),
                seed=str(protocol.seed_liquidity_usd),
            )
            protocol.status = "paused"
            return

        # Exploit check (paranoia — exploit_count should always be 0)
        if protocol.exploit_count > 0:
            self._log.critical(
                "protocol_exploit_detected",
                protocol_id=protocol.protocol_id,
                exploit_count=protocol.exploit_count,
            )
            protocol.status = "terminated"
            return
