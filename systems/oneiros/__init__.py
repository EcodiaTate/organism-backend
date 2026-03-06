"""
EcodiaOS -- Oneiros: The Dream Engine & Circadian Architecture

Thymos gave the organism a will to live. Oneiros gives it an inner life.

v2: Sleep as Batch Compiler -- Descent -> Slow Wave -> REM -> Emergence.
    Phase C: REM cross-domain synthesis, dream generation, analogy discovery.
    Phase D: Lucid dreaming (mutation testing), full emergence with wake prep.
"""

from systems.oneiros.descent import DescentStage
from systems.oneiros.emergence import EmergenceStage

# v2 modules -- Phase A/B
from systems.oneiros.engine import SleepCycleEngine

# v2 modules -- Phase D (Lucid Dreaming)
from systems.oneiros.lucid_stage import LucidDreamingStage

# v2 modules -- Phase C (REM)
from systems.oneiros.rem_stage import (
    AnalogyDiscoverer,
    CrossDomainSynthesizer,
    DreamGenerator,
    REMStage,
)
from systems.oneiros.scheduler import SleepScheduler
from systems.oneiros.service import OneirosService
from systems.oneiros.slow_wave import (
    CausalGraphReconstructor,
    HypothesisGraveyard,
    MemoryLadder,
    SlowWaveStage,
)
from systems.oneiros.types import (
    STAGE_DURATION_FRACTION,
    # v2 types -- Phase C (REM)
    AbstractStructure,
    AnalogicalTransfer,
    AnalogyDiscoveryReport,
    # v2 types -- Phase A/B
    CausalReconstructionReport,
    # v1 types (preserved for journal compatibility and health snapshot)
    CircadianPhase,
    CrossDomainMatch,
    CrossDomainSynthesisReport,
    Dream,
    DreamCoherence,
    DreamCycleResult,
    DreamGenerationReport,
    DreamInsight,
    DreamScenario,
    DreamType,
    EmergenceReport,
    HypothesisDisposition,
    HypothesisGraveyardReport,
    InsightStatus,
    # v2 types -- Phase D (Lucid Dreaming + Emergence)
    LucidDreamingReport,
    LucidResult,
    MemoryClassification,
    MemoryLadderReport,
    MutationSimulationReport,
    MutationTestResult,
    NREMConsolidationResult,
    OneirosHealthSnapshot,
    PreAttentionCache,
    PreAttentionEntry,
    REMDreamResult,
    REMStageReport,
    RungResult,
    SleepCheckpoint,
    SleepCycle,
    SleepCycleV2Report,
    SleepNarrative,
    SleepPressure,
    SleepQuality,
    SleepSchedulerConfig,
    SleepStage,
    SleepStageV2,
    SleepTrigger,
    SlowWaveReport,
    WakeDegradation,
    WakeStatePreparation,
)

__all__ = [
    # Service
    "OneirosService",
    # v1 types (preserved for journal / health snapshot compatibility)
    "SleepStage",
    "DreamType",
    "DreamCoherence",
    "InsightStatus",
    "SleepQuality",
    "SleepPressure",
    "CircadianPhase",
    "Dream",
    "DreamInsight",
    "SleepCycle",
    "NREMConsolidationResult",
    "REMDreamResult",
    "LucidResult",
    "DreamCycleResult",
    "WakeDegradation",
    "OneirosHealthSnapshot",
    # v2 -- Phase A/B: Sleep as Batch Compiler
    "SleepCycleEngine",
    "SleepScheduler",
    "DescentStage",
    "SlowWaveStage",
    "MemoryLadder",
    "HypothesisGraveyard",
    "CausalGraphReconstructor",
    "EmergenceStage",
    "SleepStageV2",
    "SleepTrigger",
    "MemoryClassification",
    "HypothesisDisposition",
    "SleepCheckpoint",
    "SleepSchedulerConfig",
    "SleepCycleV2Report",
    "SlowWaveReport",
    "EmergenceReport",
    "MemoryLadderReport",
    "RungResult",
    "HypothesisGraveyardReport",
    "CausalReconstructionReport",
    "STAGE_DURATION_FRACTION",
    # v2 -- Phase C: REM (Cross-Domain Synthesis)
    "REMStage",
    "CrossDomainSynthesizer",
    "DreamGenerator",
    "AnalogyDiscoverer",
    "AbstractStructure",
    "CrossDomainMatch",
    "DreamScenario",
    "PreAttentionEntry",
    "AnalogicalTransfer",
    "CrossDomainSynthesisReport",
    "DreamGenerationReport",
    "AnalogyDiscoveryReport",
    "REMStageReport",
    # v2 -- Phase D: Lucid Dreaming + Full Emergence
    "LucidDreamingStage",
    "MutationTestResult",
    "MutationSimulationReport",
    "LucidDreamingReport",
    "PreAttentionCache",
    "SleepNarrative",
    "WakeStatePreparation",
]
