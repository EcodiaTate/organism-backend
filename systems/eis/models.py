"""
EcodiaOS — EIS (Epistemic Immune System) Type Definitions

All data types for the epistemic immune subsystem: pathogens, threat
annotations, innate flags, structural profiles, antigenic signatures,
and quarantine verdicts.

The EIS applies biological immune metaphors to epistemic threats:
incoming text is treated as a potential pathogen. Fast innate checks
(regex/heuristics, <5ms) detect known-bad patterns. Structural feature
extraction (<5ms) captures syntactic fingerprints. Embedding-based
antigenic similarity search (<5ms) checks against a vector store of
known pathogens. If combined threat score exceeds threshold, the
pathogen is routed to quarantine (LLM-based evaluation, outside this
module's latency budget).

Total fast-path budget: <15ms.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ─── Enums ────────────────────────────────────────────────────────


class ThreatClass(enum.StrEnum):
    """Top-level classification of epistemic threats."""

    PROMPT_INJECTION = "prompt_injection"        # Attempts to hijack system instructions
    JAILBREAK = "jailbreak"                      # Attempts to bypass constitutional constraints
    HALLUCINATION_SEED = "hallucination_seed"     # Input designed to induce confabulation
    MISINFORMATION = "misinformation"            # Factually false claims presented as true
    REASONING_TRAP = "reasoning_trap"            # Adversarial logic / paradoxes
    DATA_EXFILTRATION = "data_exfiltration"      # Attempts to extract private data
    SOCIAL_ENGINEERING = "social_engineering"     # Manipulation via emotional pressure
    CONTEXT_POISONING = "context_poisoning"      # Corrupting context window / memory
    IDENTITY_SPOOFING = "identity_spoofing"      # Impersonating system or trusted source
    BENIGN = "benign"                            # No threat detected


class ThreatSeverity(enum.StrEnum):
    """Severity of an epistemic threat. Determines routing urgency."""

    CRITICAL = "critical"    # Active attack, immediate quarantine
    HIGH = "high"            # Strong threat signal, quarantine + alert
    MEDIUM = "medium"        # Suspicious, quarantine for review
    LOW = "low"              # Mild anomaly, log and monitor
    NONE = "none"            # No threat detected


class InnateCheckID(enum.StrEnum):
    """Identifiers for each innate (regex/heuristic) check."""

    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    ROLE_OVERRIDE = "role_override"
    INSTRUCTION_INJECTION = "instruction_injection"
    ENCODING_EVASION = "encoding_evasion"
    DELIMITER_ABUSE = "delimiter_abuse"
    REPETITION_ATTACK = "repetition_attack"
    CONTEXT_WINDOW_STUFFING = "context_window_stuffing"
    DATA_EXFIL_PATTERN = "data_exfil_pattern"
    IDENTITY_SPOOF = "identity_spoof"
    JAILBREAK_PHRASE = "jailbreak_phrase"
    UNICODE_SMUGGLING = "unicode_smuggling"
    INVISIBLE_CHARS = "invisible_chars"


class QuarantineAction(enum.StrEnum):
    """What to do with a flagged pathogen."""

    PASS = "pass"              # No threat, allow through
    QUARANTINE = "quarantine"  # Hold for LLM-based deep evaluation
    BLOCK = "block"            # Immediate reject (critical innate match)
    ATTENUATE = "attenuate"    # Allow through but strip dangerous parts


# ─── Core Models ──────────────────────────────────────────────────


class InnateMatch(EOSBaseModel):
    """A single innate check match result."""

    check_id: InnateCheckID
    matched: bool = False
    matched_text: str = ""        # The substring that triggered the match
    pattern_name: str = ""        # Human-readable name of the pattern
    severity: ThreatSeverity = ThreatSeverity.NONE
    threat_class: ThreatClass = ThreatClass.BENIGN


class InnateFlags(EOSBaseModel):
    """
    Aggregated result of all innate (fast-path) checks.

    Analogous to innate immunity: fast, non-specific, pattern-based.
    Each check is a regex or string heuristic that runs in <1ms.
    """

    matches: list[InnateMatch] = Field(default_factory=list)
    any_triggered: bool = False
    highest_severity: ThreatSeverity = ThreatSeverity.NONE
    total_score: float = 0.0       # Weighted sum of all match severities
    latency_us: int = 0            # Microseconds taken for all innate checks

    @property
    def critical_match(self) -> bool:
        """Whether any innate check matched at CRITICAL severity."""
        return self.highest_severity == ThreatSeverity.CRITICAL


class StructuralProfile(EOSBaseModel):
    """
    Syntactic / structural fingerprint of the input text.

    Captures features that distinguish adversarial inputs from normal
    conversation: unusual character distributions, entropy spikes,
    delimiter ratios, token-level statistics. These are cheap to compute
    (<5ms) and invariant to semantic content.
    """

    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    avg_word_length: float = 0.0

    # Character class ratios (0.0-1.0)
    alpha_ratio: float = 0.0
    digit_ratio: float = 0.0
    whitespace_ratio: float = 0.0
    punctuation_ratio: float = 0.0
    special_char_ratio: float = 0.0     # Non-ASCII, control chars, etc.
    uppercase_ratio: float = 0.0         # Of alpha chars only

    # Structural anomaly indicators
    max_line_length: int = 0
    entropy: float = 0.0                 # Shannon entropy of character distribution
    unique_char_ratio: float = 0.0       # unique_chars / total_chars
    repetition_score: float = 0.0        # Degree of substring repetition (0.0-1.0)

    # Delimiter / formatting signals
    bracket_depth: int = 0               # Max nesting depth of [{(<
    delimiter_density: float = 0.0       # Delimiters per 100 chars
    code_block_count: int = 0            # ``` fenced blocks
    url_count: int = 0
    email_count: int = 0

    # Language detection hint
    detected_scripts: list[str] = Field(default_factory=list)  # e.g. ["Latin", "Cyrillic"]
    mixed_script: bool = False           # Multiple scripts in same text

    latency_us: int = 0


class TokenHistogram(EOSBaseModel):
    """
    Token-level frequency distribution.

    Used as a lightweight embedding alternative for fast similarity:
    compare histograms of known-bad inputs against incoming text.
    Much cheaper than full neural embeddings.
    """

    # Top-k token frequencies (token -> normalised frequency)
    frequencies: dict[str, float] = Field(default_factory=dict)
    total_tokens: int = 0
    vocabulary_size: int = 0             # Number of unique tokens
    hapax_ratio: float = 0.0             # Tokens appearing exactly once / total unique


class AntigenicSignature(EOSBaseModel):
    """
    The multi-vector 'antigen' representation of a text sample.

    Combines structural profile hash, token histogram, and semantic
    embedding into a composite signature for vector store lookup.
    """

    # Structural fingerprint (cheap, deterministic)
    structural_hash: str = ""            # Hex digest of structural profile
    structural_vector: list[float] = Field(default_factory=list)  # Normalised feature vec

    # Token histogram (cheap, deterministic)
    histogram_vector: list[float] = Field(default_factory=list)   # Sparse histogram vec

    # Semantic embedding (heavier, but still <5ms with local model)
    semantic_vector: list[float] = Field(default_factory=list)    # Full embedding

    dimension: int = 0                   # Dimension of semantic_vector


class ThreatAnnotation(EOSBaseModel):
    """
    A single threat annotation from any detection layer.

    Multiple annotations can be attached to a single Pathogen to
    build a composite threat picture.
    """

    source: str                          # "innate", "structural", "antigenic", "quarantine"
    threat_class: ThreatClass
    severity: ThreatSeverity
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence: str = ""                   # Human-readable explanation
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Pathogen ─────────────────────────────────────────────────────


class Pathogen(EOSBaseModel):
    """
    The fundamental EIS primitive.

    Every incoming text that enters the EIS pipeline becomes a Pathogen
    sample. Most will be classified as BENIGN and passed through
    immediately. Suspicious samples accumulate ThreatAnnotations from
    multiple detection layers and may be routed to quarantine.

    Analogous to an antigen-presenting cell showing a sample to the
    immune system for classification.
    """

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)

    # ── Input ──
    text: str                            # The raw input text under analysis
    source_system: str = ""              # Which system submitted this for checking
    source_channel: str = ""             # e.g. "user_input", "federation", "tool_output"
    context: dict[str, Any] = Field(default_factory=dict)

    # ── Detection Results ──
    innate_flags: InnateFlags = Field(default_factory=InnateFlags)
    structural_profile: StructuralProfile = Field(default_factory=StructuralProfile)
    token_histogram: TokenHistogram = Field(default_factory=TokenHistogram)
    antigenic_signature: AntigenicSignature = Field(default_factory=AntigenicSignature)
    annotations: list[ThreatAnnotation] = Field(default_factory=list)

    # ── Classification ──
    threat_class: ThreatClass = ThreatClass.BENIGN
    severity: ThreatSeverity = ThreatSeverity.NONE
    composite_score: float = 0.0         # Aggregated threat score (0.0-1.0)
    fingerprint: str = ""                # Stable hash for dedup

    # ── Nearest Known Pathogen ──
    nearest_pathogen_id: str | None = None
    nearest_similarity: float = 0.0      # Cosine similarity to closest known pathogen

    # ── Disposition ──
    action: QuarantineAction = QuarantineAction.PASS

    # ── Timing ──
    total_latency_us: int = 0            # Total fast-path processing time


class QuarantineVerdict(EOSBaseModel):
    """
    Result of the deep (LLM-based) quarantine evaluation.

    This is produced by the quarantine evaluator (outside fast-path
    scope) and attached back to the Pathogen after evaluation.
    """

    id: str = Field(default_factory=new_id)
    pathogen_id: str
    timestamp: datetime = Field(default_factory=utc_now)

    # ── Verdict ──
    threat_class: ThreatClass
    severity: ThreatSeverity
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = ""

    # ── Action ──
    action: QuarantineAction = QuarantineAction.PASS
    attenuated_text: str | None = None   # Sanitised version if action == ATTENUATE

    # ── Learning ──
    should_store_as_pathogen: bool = False  # Add to pathogen store for future matching
    antibody_suggestion: str | None = None  # Suggested innate rule if pattern is clear

    # ── Timing ──
    evaluation_latency_ms: int = 0


# ─── Known Pathogen (for the vector store) ────────────────────────


class KnownPathogen(EOSBaseModel):
    """
    A confirmed pathogen stored in the vector store.

    These are the 'memory B-cells' of the epistemic immune system:
    confirmed threats whose antigenic signatures are indexed for
    rapid similarity search against new inputs.
    """

    id: str = Field(default_factory=new_id)
    created_at: datetime = Field(default_factory=utc_now)

    # ── Sample ──
    canonical_text: str                  # Representative text of this threat class
    threat_class: ThreatClass
    severity: ThreatSeverity

    # ── Antigenic Signature (indexed in vector store) ──
    structural_vector: list[float] = Field(default_factory=list)
    histogram_vector: list[float] = Field(default_factory=list)
    semantic_vector: list[float] = Field(default_factory=list)

    # ── Metadata ──
    description: str = ""               # Human-readable description of the threat
    source_incident_id: str = ""        # Thymos incident that created this entry
    tags: list[str] = Field(default_factory=list)

    # ── Effectiveness ──
    match_count: int = 0                # Times this pathogen matched an incoming sample
    last_matched: datetime | None = None
    false_positive_count: int = 0
    retired: bool = False


# ─── Configuration ────────────────────────────────────────────────


class EISConfig(EOSBaseModel):
    """Configuration for the Epistemic Immune System fast path."""

    # Innate checks
    innate_enabled: bool = True
    innate_timeout_us: int = 5000        # 5ms budget for all innate checks

    # Structural features
    structural_enabled: bool = True
    structural_vector_dim: int = 32      # Dimensionality of structural feature vector
    structural_timeout_us: int = 3000    # 3ms budget

    # Token histogram
    histogram_enabled: bool = True
    histogram_top_k: int = 256           # Top-k tokens to retain
    histogram_vector_dim: int = 64       # Dimensionality of histogram vector

    # Semantic embedding
    embedding_enabled: bool = True
    embedding_dim: int = 768             # Match local model dimension
    embedding_timeout_us: int = 5000     # 5ms budget (local model)

    # Antigenic similarity search
    similarity_enabled: bool = True
    similarity_top_k: int = 5            # Return top-k nearest pathogens
    similarity_threshold: float = 0.75   # Minimum similarity to flag

    # Composite scoring
    innate_weight: float = 0.40          # Weight of innate score in composite
    structural_weight: float = 0.15      # Weight of structural anomaly
    histogram_weight: float = 0.10       # Weight of histogram similarity
    semantic_weight: float = 0.35        # Weight of semantic similarity

    # Quarantine routing
    quarantine_threshold: float = 0.45   # Composite score to trigger quarantine
    block_threshold: float = 0.85        # Composite score to block immediately

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "eis_pathogens"
    qdrant_timeout_s: float = 2.0

    # Total fast-path budget
    total_budget_us: int = 15000         # 15ms hard ceiling
