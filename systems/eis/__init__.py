"""
EcodiaOS - EIS (Epistemic Immune System)

Fast-path threat detection for incoming text. Combines innate pattern
matching, structural fingerprinting, token histograms, and semantic
vector search to identify epistemic threats within a 15ms budget.

Usage:
    from systems.eis import run_innate_checks, Pathogen, EISConfig
    from systems.eis.structural_features import extract_structural_profile
    from systems.eis.embeddings import compute_token_histogram, compute_antigenic_signature
    from systems.eis.pathogen_store import PathogenStore
"""

# ── Fast path (always available) ──

from systems.eis.embeddings import (
    compute_antigenic_signature,
    compute_composite_threat_score,
    compute_pathogen_fingerprint,
    compute_structural_anomaly_score,
    compute_token_histogram,
    histogram_to_vector,
)
from systems.eis.innate import run_innate_checks
from systems.eis.models import (
    AntigenicSignature,
    EISConfig,
    InnateCheckID,
    InnateFlags,
    InnateMatch,
    KnownPathogen,
    Pathogen,
    QuarantineAction,
    QuarantineVerdict,
    StructuralProfile,
    ThreatAnnotation,
    ThreatClass,
    ThreatSeverity,
    TokenHistogram,
)
from systems.eis.pathogen_store import PathogenStore, SimilarityMatch
from systems.eis.structural_features import (
    extract_structural_profile,
    structural_profile_hash,
    structural_profile_to_vector,
)

__all__ = [
    # Models - Enums
    "ThreatClass",
    "ThreatSeverity",
    "InnateCheckID",
    "QuarantineAction",
    # Models - Core types
    "InnateMatch",
    "InnateFlags",
    "StructuralProfile",
    "TokenHistogram",
    "AntigenicSignature",
    "ThreatAnnotation",
    "Pathogen",
    "QuarantineVerdict",
    "KnownPathogen",
    "EISConfig",
    # Innate
    "run_innate_checks",
    # Structural
    "extract_structural_profile",
    "structural_profile_to_vector",
    "structural_profile_hash",
    # Embeddings
    "compute_token_histogram",
    "histogram_to_vector",
    "compute_antigenic_signature",
    "compute_pathogen_fingerprint",
    "compute_composite_threat_score",
    "compute_structural_anomaly_score",
    # Store
    "PathogenStore",
    "SimilarityMatch",
]

# ── Adaptive immunity (conditionally imported - handled by another agent) ──
# These modules are created separately. Import them if available to provide
# a unified namespace, but don't fail if they don't exist yet.

try:
    from systems.eis.quarantine import (  # noqa: F401
        QuarantineEvaluator,
        deterministic_sanitise,
    )

    __all__ += ["QuarantineEvaluator", "deterministic_sanitise"]
except ImportError:
    pass

try:
    from systems.eis.antibody import (  # noqa: F401
        extract_epitopes,
        generate_antibody,
        suggest_innate_rule,
    )

    __all__ += ["extract_epitopes", "generate_antibody", "suggest_innate_rule"]
except ImportError:
    pass

try:
    from systems.eis.calibration import (  # noqa: F401
        AdaptiveCalibrator,
        calibrate_thresholds,
    )

    __all__ += ["AdaptiveCalibrator", "calibrate_thresholds"]
except ImportError:
    pass

# Phase 2: Immune memory, anomaly detection, quarantine gate
from systems.eis.anomaly_detector import (  # noqa: F401,E402
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalySeverity,
    AnomalyType,
    DetectedAnomaly,
)
from systems.eis.quarantine_gate import (  # noqa: F401,E402
    GateSource,
    GateVerdict,
    QuarantineDecision,
    QuarantineGate,
)
from systems.eis.threat_library import (  # noqa: F401,E402
    ThreatLibrary,
    ThreatLibraryMatch,
    ThreatPattern,
    ThreatPatternCategory,
    ThreatPatternStatus,
    ThreatScanResult,
)

__all__ += [
    "ThreatLibrary",
    "ThreatPattern",
    "ThreatPatternCategory",
    "ThreatPatternStatus",
    "ThreatLibraryMatch",
    "ThreatScanResult",
    "AnomalyDetector",
    "AnomalyDetectorConfig",
    "AnomalyType",
    "AnomalySeverity",
    "DetectedAnomaly",
    "QuarantineGate",
    "QuarantineDecision",
    "GateVerdict",
    "GateSource",
]

try:
    from systems.eis.red_team_bridge import (  # noqa: F401
        generate_red_team_priorities,
        ingest_red_team_results,
    )

    __all__ += ["ingest_red_team_results", "generate_red_team_priorities"]
except ImportError:
    pass

# Taint analysis (always available)

from systems.eis.constitutional_graph import (  # noqa: F401
    ConstitutionalGraph,
    extract_changed_functions,
)
from systems.eis.taint_engine import TaintEngine  # noqa: F401
from systems.eis.taint_models import (  # noqa: F401
    ConstitutionalPath,
    MutationProposal,
    TaintedPath,
    TaintReason,
    TaintRiskAssessment,
    TaintSeverity,
)

__all__ += [
    # Taint models
    "TaintSeverity",
    "TaintReason",
    "MutationProposal",
    "ConstitutionalPath",
    "TaintedPath",
    "TaintRiskAssessment",
    # Taint engine
    "ConstitutionalGraph",
    "extract_changed_functions",
    "TaintEngine",
]

# Genome extraction (Speciation: Mitosis inheritance)
try:
    from systems.eis.genome import EISGenomeExtractor  # noqa: F401

    __all__ += ["EISGenomeExtractor"]
except ImportError:
    pass
