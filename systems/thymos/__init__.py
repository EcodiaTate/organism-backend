"""
EcodiaOS — Thymos (System #12)

The immune system. Detects failures, diagnoses root causes, prescribes
repairs, maintains an antibody library of learned fixes, and prevents
future errors through prophylactic scanning and homeostatic regulation.

Every error becomes an Incident — the organism feels it break and heals itself.
"""

from systems.thymos.antibody import AntibodyLibrary
from systems.thymos.diagnosis import (
    CausalAnalyzer,
    DiagnosticEngine,
    TemporalCorrelator,
)
from systems.thymos.governor import HealingGovernor
from systems.thymos.prescription import RepairPrescriber, RepairValidator
from systems.thymos.prophylactic import HomeostasisController, ProphylacticScanner
from systems.thymos.sentinels import (
    BankruptcySentinel,
    BaseThymosSentinel,
    CognitiveStallSentinel,
    ContractSentinel,
    DriftSentinel,
    ExceptionSentinel,
    FeedbackLoopSentinel,
)
from systems.thymos.service import ThymosService
from systems.thymos.triage import (
    IncidentDeduplicator,
    ResponseRouter,
    SeverityScorer,
)
from systems.thymos.types import (
    Antibody,
    CausalChain,
    ContractSLA,
    Diagnosis,
    DiagnosticEvidence,
    DiagnosticHypothesis,
    DiagnosticTestResult,
    DriftConfig,
    FeedbackLoop,
    HealingBudgetState,
    HealingMode,
    Incident,
    IncidentClass,
    IncidentSeverity,
    ParameterAdjustment,
    ParameterFix,
    ProphylacticWarning,
    RepairSpec,
    RepairStatus,
    RepairTier,
    StallConfig,
    TemporalCorrelation,
    ThymosHealthSnapshot,
    ValidationResult,
)

__all__ = [
    # Service
    "ThymosService",
    # Sub-systems
    "AntibodyLibrary",
    "BankruptcySentinel",
    "BaseThymosSentinel",
    "CausalAnalyzer",
    "CognitiveStallSentinel",
    "ContractSentinel",
    "DiagnosticEngine",
    "DriftSentinel",
    "ExceptionSentinel",
    "FeedbackLoopSentinel",
    "HealingGovernor",
    "HomeostasisController",
    "IncidentDeduplicator",
    "ProphylacticScanner",
    "RepairPrescriber",
    "RepairValidator",
    "ResponseRouter",
    "SeverityScorer",
    "TemporalCorrelator",
    # Types — Enums
    "HealingMode",
    "IncidentClass",
    "IncidentSeverity",
    "RepairStatus",
    "RepairTier",
    # Types — Models
    "Antibody",
    "CausalChain",
    "ContractSLA",
    "Diagnosis",
    "DiagnosticEvidence",
    "DiagnosticHypothesis",
    "DiagnosticTestResult",
    "DriftConfig",
    "FeedbackLoop",
    "HealingBudgetState",
    "Incident",
    "ParameterAdjustment",
    "ParameterFix",
    "ProphylacticWarning",
    "RepairSpec",
    "StallConfig",
    "TemporalCorrelation",
    "ThymosHealthSnapshot",
    "ValidationResult",
]
