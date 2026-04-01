"""
EcodiaOS - Evo (Learning & Hypothesis System)

Evo is the Growth drive made computational. It observes patterns across
experience, forms hypotheses, accumulates evidence, and when evidence is
sufficient, adjusts the organism's parameters, codifies procedures, and
proposes structural evolution.

Evo operates in two modes:
  WAKE - lightweight online pattern detection during each cognitive cycle
  SLEEP - deep offline consolidation: schema induction, procedure extraction,
           parameter optimisation, self-model update, drift monitoring,
           belief half-life aging

Guard rails:
  - Cannot touch Equor evaluation logic or constitutional drives
  - All parameter changes are small (velocity-limited)
  - Hypotheses must be falsifiable
  - Evolution proposals go to Simula for gating, not applied directly

Public interface:
  EvoService          - main service class
  ConsolidationResult - result of a consolidation cycle
  SelfModelStats      - self-assessment metrics
  ParameterTuner      - tunable parameter management (for direct access)
  BeliefHalfLife      - half-life metadata for knowledge freshness
"""

from systems.evo.belief_halflife import (
    BeliefAgingResult,
    BeliefAgingScanner,
    BeliefHalfLife,
    StaleBelief,
    compute_age_factor,
    compute_halflife_metadata,
    get_halflife_for_domain,
    stamp_belief_halflife,
)
from systems.evo.cognitive_niche import CognitiveNiche, NicheRegistry
from systems.evo.curiosity import CuriosityEngine
from systems.evo.genetic_memory import (
    GenomeExtractor,
    GenomeInheritanceMonitor,
    GenomeSeeder,
    compress_genome,
    decompress_genome,
)
from systems.evo.hypothesis import StructuralHypothesisGenerator
from systems.evo.meta_learning import (
    DetectorStats,
    HypothesisQualityScore,
    LearningRateState,
    MetaLearningEngine,
    MetaLearningReport,
)
from systems.evo.niche_forking import (
    ConsolidationStrategyFork,
    DetectorFork,
    EvidenceFunctionFork,
    ForkProposal,
    NicheForkingEngine,
    NicheForkingResult,
    SchemaTopologyFork,
    WorldviewFork,
)
from systems.evo.pressure import EvolutionaryPressureSystem
from systems.evo.schema_induction import (
    SchemaAlgebra,
    SchemaComposition,
    SchemaElement,
    SchemaInductionEngine,
    SchemaInductionResult,
)
from systems.evo.self_modification import SelfModificationEngine
from systems.evo.service import EvoService
from systems.evo.speciation import (
    RingSpecies,
    SpeciationEngine,
    SpeciationEvent,
    SpeciationResult,
)
from systems.evo.tournament import TournamentEngine
from systems.evo.types import (
    TUNABLE_PARAMETERS,
    VELOCITY_LIMITS,
    BeliefGenome,
    BetaDistribution,
    CognitiveSpecies,
    ConsolidationResult,
    ConsolidationSchedule,
    CrossDomainTransfer,
    CuriosityState,
    DetectorReplacementProposal,
    EpistemicIntent,
    FitnessScore,
    GenomeExtractionResult,
    GenomeInheritanceReport,
    Hypothesis,
    HypothesisCategory,
    HypothesisRef,
    HypothesisStatus,
    HypothesisTournament,
    InheritedHypothesisRecord,
    LearningArchitectureProposal,
    ParameterAdjustment,
    PatternCandidate,
    PatternType,
    PressureState,
    Procedure,
    SchemaRelationship,
    SchemaVersion,
    SelectionEvent,
    SelfModelStats,
    SelfModificationRecord,
    TournamentContext,
    TournamentOutcome,
    TournamentStage,
)

__all__ = [
    "EvoService",
    "ConsolidationResult",
    "Hypothesis",
    "HypothesisCategory",
    "HypothesisStatus",
    "ParameterAdjustment",
    "PatternCandidate",
    "PatternType",
    "Procedure",
    "SelfModelStats",
    "TUNABLE_PARAMETERS",
    "VELOCITY_LIMITS",
    # Belief half-life
    "BeliefHalfLife",
    "BeliefAgingResult",
    "BeliefAgingScanner",
    "StaleBelief",
    "compute_age_factor",
    "compute_halflife_metadata",
    "get_halflife_for_domain",
    "stamp_belief_halflife",
    # Hypothesis tournaments
    "TournamentEngine",
    "BetaDistribution",
    "HypothesisRef",
    "HypothesisTournament",
    "TournamentContext",
    "TournamentOutcome",
    "TournamentStage",
    # Genetic memory (belief inheritance)
    "BeliefGenome",
    "GenomeExtractionResult",
    "GenomeExtractor",
    "GenomeInheritanceMonitor",
    "GenomeInheritanceReport",
    "GenomeSeeder",
    "InheritedHypothesisRecord",
    "compress_genome",
    "decompress_genome",
    # Schema induction
    "SchemaInductionEngine",
    "SchemaElement",
    "SchemaComposition",
    "SchemaInductionResult",
    # Meta-learning
    "MetaLearningEngine",
    "MetaLearningReport",
    "DetectorStats",
    "LearningRateState",
    "HypothesisQualityScore",
    # Curiosity-driven exploration
    "CuriosityEngine",
    "CuriosityState",
    "EpistemicIntent",
    # Evolutionary pressure
    "EvolutionaryPressureSystem",
    "FitnessScore",
    "SelectionEvent",
    "CognitiveSpecies",
    "PressureState",
    # Self-modification
    "SelfModificationEngine",
    "DetectorReplacementProposal",
    "LearningArchitectureProposal",
    "SelfModificationRecord",
    "ConsolidationSchedule",
    # Schema algebra
    "SchemaAlgebra",
    "SchemaVersion",
    "SchemaRelationship",
    "CrossDomainTransfer",
    # Structural hypothesis generation
    "StructuralHypothesisGenerator",
    # Cognitive speciation
    "CognitiveNiche",
    "NicheRegistry",
    "SpeciationEngine",
    "SpeciationEvent",
    "SpeciationResult",
    "RingSpecies",
    # Niche forking (cognitive organogenesis)
    "NicheForkingEngine",
    "NicheForkingResult",
    "ForkProposal",
    "DetectorFork",
    "EvidenceFunctionFork",
    "ConsolidationStrategyFork",
    "SchemaTopologyFork",
    "WorldviewFork",
]
