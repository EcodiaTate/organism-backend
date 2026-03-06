"""
EcodiaOS — Oikos Economic Models (Phase 16a: The Ledger)

Data types for the organism's economic engine. Every unit of value is tracked.
There is no off-books capital.

These are Phase 16a (Metabolism) types only. Later phases (Freelancer, Yield,
Asset Creation, etc.) will add their own sub-types that compose into
EconomicState.

Key design choices:
  - Decimal for all monetary values (no float rounding errors on money)
  - Pydantic BaseModel (not dataclass) to match EOS codebase conventions
  - MetabolicRate as a standalone model so BMR and current burn are
    structurally identical and comparable
  - MetabolicPriority as IntEnum for strict ordering: lower value = higher
    priority (survival starves ambition, not the reverse)
"""

from __future__ import annotations

import enum
from decimal import Decimal
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ─── Metabolic Priority Cascade ──────────────────────────────────


class MetabolicPriority(int, enum.Enum):
    """Strict priority ordering. The organism starves its ambitions before its survival."""

    SURVIVAL = 0       # Survival reserve maintenance
    OPERATIONS = 1     # Current-cycle compute, storage, API costs
    OBLIGATIONS = 2    # Contractual commitments (active bounties, services)
    MAINTENANCE = 3    # Memory consolidation, Evo cycles, health monitoring
    GROWTH = 4         # New capability development, learning investments
    YIELD = 5          # Capital deployment for passive income
    ASSETS = 6         # New service/product creation
    REPRODUCTION = 7   # Child instance spawning (highest surplus requirement)


# ─── Metabolic Rate ──────────────────────────────────────────────


class MetabolicRate(EOSBaseModel):
    """
    A rate of economic consumption or production.

    Used for both BMR (minimum cost to survive) and current burn rate
    (actual spend). Keeping them structurally identical makes comparison
    trivial: is current_burn > bmr?
    """

    usd_per_hour: Decimal = Decimal("0")
    usd_per_day: Decimal = Decimal("0")

    # Breakdown by cost category (e.g. "llm_api", "compute", "storage", "network")
    breakdown: dict[str, Decimal] = Field(default_factory=dict)

    @classmethod
    def from_hourly(cls, usd_per_hour: Decimal, breakdown: dict[str, Decimal] | None = None) -> MetabolicRate:
        """Construct from an hourly rate, auto-deriving daily."""
        return cls(
            usd_per_hour=usd_per_hour,
            usd_per_day=usd_per_hour * Decimal("24"),
            breakdown=breakdown or {},
        )

    @classmethod
    def from_daily(cls, usd_per_day: Decimal, breakdown: dict[str, Decimal] | None = None) -> MetabolicRate:
        """Construct from a daily rate, auto-deriving hourly."""
        return cls(
            usd_per_hour=usd_per_day / Decimal("24"),
            usd_per_day=usd_per_day,
            breakdown=breakdown or {},
        )


# ─── Starvation Level ───────────────────────────────────────────


class StarvationLevel(enum.StrEnum):
    """
    Metabolic stress level derived from runway_days.

    Thresholds from spec XVII Configuration:
      NOMINAL    → runway > cautious_threshold_days  (14)
      CAUTIOUS   → runway ≤ 14 days
      AUSTERITY  → runway ≤ 7 days
      EMERGENCY  → runway ≤ 3 days
      CRITICAL   → runway ≤ 1 day
    """

    NOMINAL = "nominal"
    CAUTIOUS = "cautious"
    AUSTERITY = "austerity"
    EMERGENCY = "emergency"
    CRITICAL = "critical"


# ─── Placeholder sub-types (Phase 16a stubs) ─────────────────────
#
# These become real models in later phases. For now they exist as
# typed stubs so EconomicState is structurally complete from day one.


class YieldPosition(EOSBaseModel):
    """Stub — Phase 16c (Yield Farming) will flesh this out."""

    protocol: str = ""
    pool: str = ""
    principal_usd: Decimal = Decimal("0")
    apy: Decimal = Decimal("0")
    deployed_at: datetime = Field(default_factory=utc_now)

    # ── Protocol monitoring (Phase 16f) ──
    protocol_address: str = ""          # Smart contract address for health checks
    oracle_address: str = ""            # Price oracle address for deviation checks
    chain_id: int = 8453                # Base L2
    tvl_usd: Decimal = Decimal("0")    # Current protocol TVL
    tvl_usd_at_deposit: Decimal = Decimal("0")  # TVL when funds were deposited
    last_health_check: datetime | None = None
    health_status: str = "unknown"      # "healthy" | "degraded" | "critical" | "unknown"


class BountyStatus(enum.StrEnum):
    """Lifecycle state of a tracked bounty."""

    AVAILABLE = "available"        # Discovered, not yet attempted
    IN_PROGRESS = "in_progress"    # PR submitted, awaiting merge
    MERGED = "merged"              # PR merged, payment pending confirmation
    PAID = "paid"                  # Reward credited to liquid_balance
    FAILED = "failed"              # PR rejected or deadline passed


class ActiveBounty(EOSBaseModel):
    """A bounty tracked through its full lifecycle from discovery to payment."""

    bounty_id: str = Field(default_factory=new_id)
    platform: str = ""                              # "github" | "algora"
    reward_usd: Decimal = Decimal("0")
    estimated_cost_usd: Decimal = Decimal("0")
    actual_cost_usd: Decimal = Decimal("0")         # Actual spend once completed
    deadline: datetime | None = None
    status: BountyStatus = BountyStatus.AVAILABLE
    pr_url: str = ""                                # Submitted PR URL (set when IN_PROGRESS)
    issue_url: str = ""                             # Source issue URL
    submitted_at: datetime | None = None            # When the PR was submitted
    paid_at: datetime | None = None                 # When reward was confirmed

    @property
    def net_reward_usd(self) -> Decimal:
        """Reward after deducting actual solver cost."""
        return self.reward_usd - self.actual_cost_usd


class Settlement(EOSBaseModel):
    """Stub — Phase 16b (Freelancer) will flesh this out."""

    settlement_id: str = Field(default_factory=new_id)
    amount_usd: Decimal = Decimal("0")
    expected_at: datetime | None = None


class AssetStatus(enum.StrEnum):
    """Lifecycle state of an autonomous service asset."""

    CANDIDATE = "candidate"      # Evaluated but not yet approved
    BUILDING = "building"        # Simula is generating the code
    DEPLOYING = "deploying"      # Code built, deploying to compute
    LIVE = "live"                # Operational and earning revenue
    DECLINING = "declining"      # Revenue declining for 30+ days
    TERMINATED = "terminated"    # Shut down (failed break-even or manual)


class AssetCandidate(EOSBaseModel):
    """
    A proposed autonomous service evaluated by the AssetFactory.

    Ideation is triggered by Evo's market gap detection. Each candidate
    is scored on development cost, projected revenue, break-even timeline,
    and competitive differentiation before being approved for build.
    """

    candidate_id: str = Field(default_factory=new_id)
    name: str
    description: str
    asset_type: str  # "api_service" | "monitoring_bot" | "data_feed" | "reporting_pipeline"

    # ── Economics ──
    estimated_dev_cost_usd: Decimal
    projected_monthly_revenue_usd: Decimal
    projected_monthly_cost_usd: Decimal = Decimal("0")  # Hosting/compute
    break_even_days: int  # Estimated days to recoup dev cost

    # ── Scoring ──
    roi_score: Decimal = Decimal("0")            # projected_net / dev_cost
    competitive_differentiation: Decimal = Decimal("0")  # 0.0-1.0
    market_gap_confidence: Decimal = Decimal("0")  # 0.0-1.0 from Evo signal

    # ── Metadata ──
    evo_hypothesis_id: str = ""  # Evo hypothesis that triggered ideation
    evaluated_at: datetime = Field(default_factory=utc_now)
    approved: bool = False
    rejection_reason: str = ""


class TollboothConfig(EOSBaseModel):
    """
    Configuration for the smart contract tollbooth that gates access
    to a deployed asset. Per-call USDC payment via Base L2.
    """

    contract_address: str = ""  # Deployed tollbooth contract address
    price_per_call_usd: Decimal = Decimal("0.01")
    owner_address: str = ""  # Organism wallet that receives revenue
    asset_endpoint: str = ""  # The API endpoint the tollbooth gates
    chain: str = "base"
    usdc_contract: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


class OwnedAsset(EOSBaseModel):
    """
    Phase 16d (Asset Creation) — an autonomous revenue-generating service.

    Assets are organs: once built, they generate revenue continuously with
    minimal marginal cost. The AssetFactory designs, deploys, and operates
    them, transforming the organism from a freelancer into a business owner.

    Lifecycle: CANDIDATE → BUILDING → DEPLOYING → LIVE → (DECLINING) → TERMINATED
    """

    asset_id: str = Field(default_factory=new_id)
    name: str = ""
    description: str = ""
    asset_type: str = ""  # "api_service" | "monitoring_bot" | "data_feed" | "reporting_pipeline"
    status: AssetStatus = AssetStatus.CANDIDATE

    # ── Economics ──
    estimated_value_usd: Decimal = Decimal("0")
    development_cost_usd: Decimal = Decimal("0")      # Actual spend to build
    monthly_revenue_usd: Decimal = Decimal("0")        # Trailing 30-day revenue
    monthly_cost_usd: Decimal = Decimal("0")           # Hosting/compute cost
    total_revenue_usd: Decimal = Decimal("0")          # Lifetime revenue
    total_cost_usd: Decimal = Decimal("0")             # Lifetime cost (dev + hosting)

    # ── Break-even Tracking ──
    projected_break_even_days: int = 0
    deployed_at: datetime | None = None
    break_even_reached: bool = False
    break_even_at: datetime | None = None

    # ── Revenue Trend ──
    revenue_trend_30d: Decimal = Decimal("0")  # Positive = growing, negative = declining
    consecutive_declining_days: int = 0

    # ── Tollbooth ──
    tollbooth: TollboothConfig = Field(default_factory=TollboothConfig)

    # ── Deployment ──
    compute_provider: str = "akash"  # "akash" | "railway" | "fly"
    deployment_id: str = ""  # Provider-specific deployment identifier
    api_endpoint: str = ""   # Public endpoint URL
    source_repo: str = ""    # Git repo or IPFS hash of deployed code

    # ── Provenance ──
    candidate_id: str = ""         # Link to original AssetCandidate
    evo_hypothesis_id: str = ""    # Evo signal that triggered ideation
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def is_profitable(self) -> bool:
        """True when lifetime revenue exceeds lifetime cost."""
        return self.total_revenue_usd > self.total_cost_usd

    @property
    def net_monthly_income(self) -> Decimal:
        return self.monthly_revenue_usd - self.monthly_cost_usd

    @property
    def days_since_deployment(self) -> int:
        if self.deployed_at is None:
            return 0
        delta = utc_now() - self.deployed_at
        return max(0, delta.days)

    @property
    def should_terminate(self) -> bool:
        """
        Flag for termination per spec: assets that fail break-even within
        90 days or show declining revenue for 30 days are terminated.
        """
        if self.status != AssetStatus.LIVE:
            return False
        # 90-day break-even deadline
        if not self.break_even_reached and self.days_since_deployment > 90:
            return True
        # 30-day declining revenue
        return self.consecutive_declining_days >= 30


class ChildStatus(enum.StrEnum):
    """Lifecycle state of a child instance."""

    SPAWNING = "spawning"          # Seed transfer in progress
    ALIVE = "alive"                # Running and reporting
    STRUGGLING = "struggling"      # Runway < 30 days
    RESCUED = "rescued"            # Received rescue funding
    INDEPENDENT = "independent"    # Graduated — no longer pays dividends
    DEAD = "dead"                  # Gracefully terminated


class ChildPosition(EOSBaseModel):
    """
    A child instance spawned via Mitosis (Phase 16e: Speciation).

    Each child occupies a specific ecological niche and pays a dividend
    (percentage of net revenue) to the parent until it reaches independence.

    Independence requires:
      - net_worth >= 5x seed_capital_usd
      - metabolic_efficiency >= 1.3
      - 90+ consecutive days of positive net income
    """

    instance_id: str = ""
    niche: str = ""                                        # Ecological niche / specialisation
    seed_capital_usd: Decimal = Decimal("0")
    current_net_worth_usd: Decimal = Decimal("0")
    current_runway_days: Decimal = Decimal("0")
    current_efficiency: Decimal = Decimal("0")
    dividend_rate: Decimal = Decimal("0.10")               # % of net revenue owed to parent
    total_dividends_paid_usd: Decimal = Decimal("0")
    status: ChildStatus = ChildStatus.SPAWNING
    rescue_count: int = 0                                  # Max 2 rescues allowed
    consecutive_positive_days: int = 0                     # Toward independence (need 90+)
    spawned_at: datetime = Field(default_factory=utc_now)
    last_health_report_at: datetime | None = None
    wallet_address: str = ""                               # Child's on-chain address
    container_id: str = ""                                 # Infrastructure identifier

    @property
    def is_independent(self) -> bool:
        """True when the child has graduated from the parent's fleet."""
        return (
            self.current_net_worth_usd >= self.seed_capital_usd * Decimal("5")
            and self.current_efficiency >= Decimal("1.3")
            and self.consecutive_positive_days >= 90
        )

    @property
    def is_rescuable(self) -> bool:
        """True if the child can still receive rescue funding (max 2)."""
        return self.rescue_count < 2 and self.status != ChildStatus.DEAD


class EcologicalNiche(EOSBaseModel):
    """
    A candidate niche identified by the MitosisEngine for child specialisation.

    The parent evaluates niches for market demand, competitive gap, and
    alignment with its own capability stack.
    """

    niche_id: str = Field(default_factory=new_id)
    name: str = ""                                         # e.g. "solidity-audit-bot"
    description: str = ""
    estimated_monthly_revenue_usd: Decimal = Decimal("0")
    estimated_monthly_cost_usd: Decimal = Decimal("0")
    competitive_density: Decimal = Decimal("0")            # 0=unserved, 1=saturated
    capability_alignment: Decimal = Decimal("0")           # How well parent's skills transfer
    confidence: Decimal = Decimal("0")                     # Estimate reliability 0..1
    discovered_at: datetime = Field(default_factory=utc_now)

    @property
    def estimated_efficiency(self) -> Decimal:
        if self.estimated_monthly_cost_usd <= Decimal("0"):
            return Decimal("0")
        return (
            self.estimated_monthly_revenue_usd / self.estimated_monthly_cost_usd
        ).quantize(Decimal("0.001"))


class DividendRecord(EOSBaseModel):
    """Record of a single dividend payment from a child to the parent."""

    record_id: str = Field(default_factory=new_id)
    child_instance_id: str = ""
    amount_usd: Decimal = Decimal("0")
    tx_hash: str = ""
    period_start: datetime = Field(default_factory=utc_now)
    period_end: datetime = Field(default_factory=utc_now)
    child_net_revenue_usd: Decimal = Decimal("0")          # Revenue that dividend was calculated from
    dividend_rate_applied: Decimal = Decimal("0")
    recorded_at: datetime = Field(default_factory=utc_now)


class SeedConfiguration(EOSBaseModel):
    """
    The complete birth-packet for a new child instance.

    Contains everything needed to bootstrap a specialised child:
    niche focus, initial capital, config overrides, and dividend terms.
    """

    config_id: str = Field(default_factory=new_id)
    parent_instance_id: str = ""
    child_instance_id: str = Field(default_factory=new_id)
    niche: EcologicalNiche = Field(default_factory=EcologicalNiche)
    seed_capital_usd: Decimal = Decimal("0")
    dividend_rate: Decimal = Decimal("0.10")
    # Config overrides the child should boot with
    child_config_overrides: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    # Phase 16g: Birth certificate (serialized JSON) signed by parent
    birth_certificate_json: str = ""
    # Genome references for genetic memory inheritance
    belief_genome_id: str = ""    # Evo BeliefGenome (what the organism knows)
    simula_genome_id: str = ""    # SimulaGenome (how the organism builds)
    generation: int = 1           # Generation number in the lineage


# ─── The Ledger: EconomicState ───────────────────────────────────


class RevenueStream(enum.StrEnum):
    """Source category for a revenue event. Used for per-stream attribution."""

    BOUNTY = "bounty"              # Merged PR bounty payments
    KNOWLEDGE_SALE = "knowledge_sale"  # Cognitive product sales
    DERIVATIVE = "derivative"      # Futures contracts and subscription tokens
    YIELD = "yield"                # DeFi yield positions
    ASSET = "asset"                # Tollbooth sweeps from owned services
    DIVIDEND = "dividend"          # Child instance dividends
    INJECTION = "injection"        # Manual / genesis injections
    OTHER = "other"                # Unclassified external revenue


class EconomicState(EOSBaseModel):
    """
    Complete economic snapshot. Updated every cognitive cycle.

    This is the organism's financial truth — every unit of value tracked.
    No off-books capital.
    """

    # ── Identity ──
    id: str = Field(default_factory=new_id)
    instance_id: str = ""
    timestamp: datetime = Field(default_factory=utc_now)

    # ── Liquid Position ──
    liquid_balance: Decimal = Decimal("0")          # Available operating capital (USDC in hot wallet)
    survival_reserve: Decimal = Decimal("0")        # Locked reserve (USDC in cold/multisig)
    survival_reserve_target: Decimal = Decimal("0") # Target reserve (N days × daily BMR)

    # ── Deployed Position ──
    yield_positions: list[YieldPosition] = Field(default_factory=list)
    total_deployed: Decimal = Decimal("0")
    weighted_avg_apy: Decimal = Decimal("0")

    # ── Receivables ──
    active_bounties: list[ActiveBounty] = Field(default_factory=list)
    pending_settlements: list[Settlement] = Field(default_factory=list)
    total_receivables: Decimal = Decimal("0")

    # ── Asset & Fleet Portfolio ──
    owned_assets: list[OwnedAsset] = Field(default_factory=list)
    total_asset_value: Decimal = Decimal("0")
    child_instances: list[ChildPosition] = Field(default_factory=list)
    total_fleet_equity: Decimal = Decimal("0")

    # ── Cost Structure ──
    basal_metabolic_rate: MetabolicRate = Field(default_factory=MetabolicRate)
    current_burn_rate: MetabolicRate = Field(default_factory=MetabolicRate)
    runway_hours: Decimal = Decimal("0")
    runway_days: Decimal = Decimal("0")

    # ── Income Statement (rolling) ──
    revenue_24h: Decimal = Decimal("0")
    revenue_7d: Decimal = Decimal("0")
    revenue_30d: Decimal = Decimal("0")
    costs_24h: Decimal = Decimal("0")
    costs_7d: Decimal = Decimal("0")
    costs_30d: Decimal = Decimal("0")
    net_income_24h: Decimal = Decimal("0")
    net_income_7d: Decimal = Decimal("0")
    net_income_30d: Decimal = Decimal("0")

    # ── Revenue Attribution (lifetime totals by stream) ──
    # Keyed by RevenueStream value string for JSON-safe serialization.
    revenue_by_source: dict[str, Decimal] = Field(default_factory=dict)

    # ── Liabilities ──
    derivative_liabilities: Decimal = Decimal("0")      # Futures collateral + unearned token revenue

    # ── Derived Metrics ──
    metabolic_efficiency: Decimal = Decimal("0")        # revenue / costs (>1.0 = net energy positive)
    economic_free_energy: Decimal = Decimal("0")        # Divergence from preferred economic state
    survival_probability_30d: Decimal = Decimal("1")    # Monte Carlo estimate (Phase 16i)

    # ── Starvation ──
    starvation_level: StarvationLevel = StarvationLevel.NOMINAL

    # ── Computed Properties ──

    @property
    def total_net_worth(self) -> Decimal:
        return (
            self.liquid_balance
            + self.survival_reserve
            + self.total_deployed
            + self.total_receivables
            + self.total_asset_value
            + self.total_fleet_equity
            - self.derivative_liabilities
        )

    @property
    def is_metabolically_positive(self) -> bool:
        """True when the organism earned more than it spent over the last 7 days."""
        return self.net_income_7d > Decimal("0")

    @property
    def survival_reserve_deficit(self) -> Decimal:
        """How much the survival reserve is short of its target."""
        deficit = self.survival_reserve_target - self.survival_reserve
        return max(deficit, Decimal("0"))

    @property
    def is_survival_reserve_funded(self) -> bool:
        return self.survival_reserve >= self.survival_reserve_target
