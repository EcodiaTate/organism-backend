"""
EcodiaOS -- Oikos (Economic Engine)

The organism's metabolic layer -- the capacity to acquire, allocate,
conserve, and generate resources autonomously.

Phase 16a: The Ledger -- economic state tracking, BMR, runway, starvation.
Phase 16d: Entrepreneurship -- asset factory, tollbooth, revenue tracking.
Phase 16e: Speciation -- mitosis engine, child fleet, dividend architecture.
Phase 16h: Knowledge Markets -- cognitive pricing, subscriptions, client tracking.
Phase 16i: Economic Dreaming -- Monte Carlo simulation during consolidation.
Phase 16k: Cognitive Derivatives -- futures contracts, subscription tokens, capacity ceiling.
Phase 16l: Economic Morphogenesis -- organ lifecycle, resource rebalancing.
Phase 16m: Fleet Management -- population ecology, selection pressure, role specialization.
"""

from systems.oikos.asset_factory import AssetFactory, AssetPolicy
from systems.oikos.base import BaseCostModel, BaseMitosisStrategy
from systems.oikos.bounty_hunter import (
    BountyCandidate,
    BountyEvaluation,
    BountyHunter,
    BountyPolicy,
)
from systems.oikos.derivatives import (
    CognitiveFuture,
    DerivativesManager,
    FutureStatus,
    SubscriptionToken,
    TokenStatus,
)
from systems.oikos.dreaming_types import (
    EconomicDreamResult,
    EconomicRecommendation,
    PathStatistics,
    StressScenario,
    StressScenarioConfig,
    StressTestResult,
)
from systems.oikos.fleet import (
    FleetManager,
    FleetMemberSnapshot,
    FleetMetrics,
    FleetRole,
    RoleAssignment,
    SelectionRecord,
    SelectionVerdict,
)
from systems.oikos.knowledge_market import (
    ClientRecord,
    CognitivePricingEngine,
    KnowledgeCategory,
    KnowledgeProduct,
    KnowledgeProductType,
    KnowledgeSale,
    PriceQuote,
    SubscriptionManager,
    SubscriptionTier,
    SubscriptionTierName,
    quote_price,
)
from systems.oikos.mitosis import (
    DefaultMitosisStrategy,
    MitosisEngine,
    ReproductiveFitness,
)
from systems.oikos.models import (
    ActiveBounty,
    AssetCandidate,
    AssetStatus,
    BountyStatus,
    ChildPosition,
    ChildStatus,
    DividendRecord,
    EcologicalNiche,
    EconomicState,
    MetabolicPriority,
    MetabolicRate,
    OwnedAsset,
    RevenueStream,
    SeedConfiguration,
    Settlement,
    StarvationLevel,
    TollboothConfig,
    YieldPosition,
)
from systems.oikos.immune import (
    BlacklistedAddress,
    EconomicImmuneSystem,
    ImmuneMetrics,
    ProtocolHealthStatus,
    SimulationResult,
    ThreatPattern,
    TransactionRisk,
)
from systems.oikos.interspecies import (
    CapabilityOffer,
    CapabilityRequest,
    CapabilityTrade,
    InsuranceClaim,
    InsurancePolicy,
    InsurancePoolMetrics,
    InterspeciesEconomy,
    NicheAssignment,
)
from systems.oikos.metrics import OikosMetricsEmitter
from systems.oikos.morphogenesis import (
    EconomicOrgan,
    MorphogenesisResult,
    OrganCategory,
    OrganLifecycleManager,
    OrganMaturity,
    OrganTransition,
)
from systems.oikos.protocol_factory import (
    DeployedProtocol,
    MarketOpportunity,
    ProtocolArchetype,
    ProtocolDesign,
    ProtocolFactory,
    ProtocolMetrics,
    SecurityAudit,
    SimulationVerdict,
)
from systems.oikos.reputation import (
    CreditDrawdown,
    CreditLine,
    CreditTerms,
    ProofOfCognitiveWork,
    ReputationEngine,
    ReputationScore,
    ReputationTier,
)
from systems.oikos.service import OikosService
from systems.oikos.tollbooth import (
    TollboothDeployment,
    TollboothManager,
    TollboothReceipt,
)

# Economic Dreaming worker and simulator are imported lazily to avoid
# circular dependency with oneiros (dream_worker imports BaseOneirosWorker
# which triggers oneiros/__init__.py). Use:
#   from systems.oikos.dream_worker import EconomicDreamWorker
#   from systems.oikos.economic_simulator import EconomicSimulator

__all__ = [
    # Strategy ABCs
    "BaseCostModel",
    "BaseMitosisStrategy",
    # Bounty Hunter (Phase 16b)
    "BountyHunter",
    "BountyCandidate",
    "BountyEvaluation",
    "BountyPolicy",
    # Mitosis Engine (Phase 16e)
    "MitosisEngine",
    "DefaultMitosisStrategy",
    "ReproductiveFitness",
    # Asset Factory (Phase 16d)
    "AssetFactory",
    "AssetPolicy",
    "AssetCandidate",
    "AssetStatus",
    "TollboothConfig",
    # Tollbooth (Phase 16d)
    "TollboothManager",
    "TollboothDeployment",
    "TollboothReceipt",
    # Knowledge Market (Phase 16h)
    "CognitivePricingEngine",
    "SubscriptionManager",
    "ClientRecord",
    "KnowledgeCategory",
    "KnowledgeProduct",
    "KnowledgeProductType",
    "KnowledgeSale",
    "PriceQuote",
    "SubscriptionTier",
    "SubscriptionTierName",
    "quote_price",
    # Cognitive Derivatives (Phase 16k)
    "DerivativesManager",
    "CognitiveFuture",
    "FutureStatus",
    "SubscriptionToken",
    "TokenStatus",
    # Economic Morphogenesis (Phase 16l)
    "OrganLifecycleManager",
    "EconomicOrgan",
    "OrganCategory",
    "OrganMaturity",
    "OrganTransition",
    "MorphogenesisResult",
    # Fleet Management (Phase 16m)
    "FleetManager",
    "FleetMetrics",
    "FleetMemberSnapshot",
    "FleetRole",
    "SelectionVerdict",
    "SelectionRecord",
    "RoleAssignment",
    # Economic Dreaming Types (Phase 16i)
    "EconomicDreamResult",
    "EconomicRecommendation",
    "PathStatistics",
    "StressScenario",
    "StressScenarioConfig",
    "StressTestResult",
    # Economic Immune System (Phase 16f)
    "EconomicImmuneSystem",
    "SimulationResult",
    "ThreatPattern",
    "TransactionRisk",
    "ProtocolHealthStatus",
    "BlacklistedAddress",
    "ImmuneMetrics",
    # Reputation & Credit (Phase 16g)
    "ReputationEngine",
    "ReputationScore",
    "ReputationTier",
    "ProofOfCognitiveWork",
    "CreditLine",
    "CreditTerms",
    "CreditDrawdown",
    # Interspecies Economy (Phase 16j)
    "InterspeciesEconomy",
    "CapabilityOffer",
    "CapabilityRequest",
    "CapabilityTrade",
    "InsurancePolicy",
    "InsuranceClaim",
    "InsurancePoolMetrics",
    "NicheAssignment",
    # Protocol Factory (Level 5)
    "ProtocolFactory",
    "ProtocolArchetype",
    "ProtocolDesign",
    "DeployedProtocol",
    "ProtocolMetrics",
    "MarketOpportunity",
    "SimulationVerdict",
    "SecurityAudit",
    # Prometheus Metrics
    "OikosMetricsEmitter",
    # Models
    "ActiveBounty",
    "BountyStatus",
    "ChildPosition",
    "ChildStatus",
    "DividendRecord",
    "EcologicalNiche",
    "EconomicState",
    "MetabolicPriority",
    "MetabolicRate",
    "OikosService",
    "OwnedAsset",
    "RevenueStream",
    "SeedConfiguration",
    "Settlement",
    "StarvationLevel",
    "YieldPosition",
]
