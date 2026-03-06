"""
EcodiaOS — EIS Epistemic Antibody System (Adaptive Immunity)

Epistemic antibodies are crystallised threat signatures that allow the
EIS to instantly recognise and neutralise previously-seen attack patterns
without invoking the full quarantine evaluation pipeline.

Two core pipelines:

  1. extract_epitopes  — Given a QuarantineVerdict, extract the minimal
     distinctive "epitope" signatures that uniquely identify this class
     of threat. Analogous to how biological antibodies bind to specific
     protein fragments.

  2. generate_antibody — Given extracted epitopes and the original
     QuarantineVerdict, compile them into a KnownPathogen record
     suitable for the vector store, plus an optional innate rule
     suggestion for the fast path.

Lifecycle:
  CREATION → stored in Qdrant as KnownPathogen → MATCHING on future inputs
  → FEEDBACK (true/false positive tracking) → RETIREMENT when ineffective
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from primitives.common import new_id, utc_now
from systems.eis.models import (
    KnownPathogen,
    Pathogen,
    QuarantineAction,
    QuarantineVerdict,
    ThreatClass,
    ThreatSeverity,
)

if TYPE_CHECKING:
    from systems.eis.quarantine import SanitisationResult

logger = structlog.get_logger().bind(system="eis", component="antibody")


# ─── Epitope Types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Epitope:
    """
    A minimal distinctive signature extracted from a threat.

    Epitopes are the "binding sites" — the specific patterns that
    distinguish this threat from benign content. They come in three
    flavours:

    - LEXICAL:     regex pattern matching raw text fragments
    - SEMANTIC:    embedding-space reference (stored as tag for Qdrant lookup)
    - STRUCTURAL:  structural profile anomaly signal
    """

    kind: str               # "lexical" | "semantic" | "structural"
    signature: str          # The pattern (regex for lexical, JSON for others)
    weight: float           # 0.0-1.0, importance for matching
    source_rule: str        # Which sanitisation rule or LLM analysis produced this
    hash: str               # SHA-256 fingerprint of the signature


@dataclass
class EpitopeExtractionResult:
    """Output of the extract_epitopes pipeline."""

    epitopes: list[Epitope]
    threat_class: ThreatClass
    source_pathogen_id: str
    extraction_time_ms: int
    total_candidates: int
    retained_count: int


# ─── Innate Rule Suggestion ─────────────────────────────────────────────────


@dataclass
class InnateRuleSuggestion:
    """
    A suggested new innate check derived from a confirmed threat.

    If the quarantine evaluator or epitope extraction identifies a
    clear, compact pattern, it suggests promoting it to the innate
    layer for O(1) future matching.
    """

    pattern_regex: str              # Compiled-safe regex string
    threat_class: ThreatClass
    severity: ThreatSeverity
    description: str
    source_pathogen_id: str
    confidence: float               # How confident we are this won't false-positive


# ─── Constants ──────────────────────────────────────────────────────────────

_MIN_EPITOPE_WEIGHT = 0.2
_MAX_EPITOPES_PER_ANTIBODY = 12
_MIN_LEXICAL_LENGTH = 4


def _hash_signature(sig: str) -> str:
    """Deterministic hash for deduplication."""
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


# ─── Epitope Extraction Pipeline ────────────────────────────────────────────


def _extract_lexical_epitopes(
    pathogen: Pathogen,
    verdict: QuarantineVerdict,
    sanitisation: SanitisationResult,
) -> list[Epitope]:
    """
    Extract lexical epitopes from innate matches, sanitisation rules,
    and LLM-suggested antibody patterns.
    """
    epitopes: list[Epitope] = []
    seen_hashes: set[str] = set()

    # From innate check matches — these are the highest-quality epitopes
    for match in pathogen.innate_flags.matches:
        if not match.matched or not match.matched_text:
            continue
        sig = re.escape(match.matched_text[:200])
        if len(sig) < _MIN_LEXICAL_LENGTH:
            continue
        h = _hash_signature(sig)
        if h not in seen_hashes:
            seen_hashes.add(h)
            from systems.eis.innate import _SEVERITY_SCORE
            weight = _SEVERITY_SCORE.get(match.severity, 0.3)
            epitopes.append(Epitope(
                kind="lexical",
                signature=sig,
                weight=min(weight, 1.0),
                source_rule=f"innate:{match.check_id.value}",
                hash=h,
            ))

    # From sanitisation rules that fired
    for rule_name, count in sanitisation.sanitisation_map.items():
        sig = f"rule:{rule_name}:count={count}"
        h = _hash_signature(sig)
        if h not in seen_hashes:
            seen_hashes.add(h)
            base_weight = min(count * 0.25, 0.8)
            epitopes.append(Epitope(
                kind="lexical",
                signature=sig,
                weight=base_weight,
                source_rule=rule_name,
                hash=h,
            ))

    # From LLM antibody suggestion (if the quarantine evaluator suggested one)
    if verdict.antibody_suggestion:
        sig = verdict.antibody_suggestion[:300]
        h = _hash_signature(sig)
        if h not in seen_hashes:
            seen_hashes.add(h)
            epitopes.append(Epitope(
                kind="lexical",
                signature=sig,
                weight=0.6,
                source_rule="llm_antibody_suggestion",
                hash=h,
            ))

    return epitopes


def _extract_structural_epitopes(
    pathogen: Pathogen,
    verdict: QuarantineVerdict,
) -> list[Epitope]:
    """
    Extract structural epitopes from the pathogen's structural profile
    and antigenic signature.
    """
    epitopes: list[Epitope] = []

    # Structural anomaly indicators
    profile = pathogen.structural_profile
    anomaly_signals: list[tuple[str, float]] = []

    if profile.mixed_script:
        anomaly_signals.append(("mixed_script", 0.6))
    if profile.special_char_ratio > 0.1:
        anomaly_signals.append((f"high_special_chars:{profile.special_char_ratio:.2f}", 0.5))
    if profile.repetition_score > 0.6:
        anomaly_signals.append((f"high_repetition:{profile.repetition_score:.2f}", 0.4))
    if profile.bracket_depth > 10:
        anomaly_signals.append((f"deep_brackets:{profile.bracket_depth}", 0.4))

    for sig, weight in anomaly_signals:
        full_sig = f"structural:{sig}"
        epitopes.append(Epitope(
            kind="structural",
            signature=full_sig,
            weight=weight,
            source_rule="structural_anomaly",
            hash=_hash_signature(full_sig),
        ))

    # Threat class as structural signal
    if verdict.threat_class != ThreatClass.BENIGN:
        sig = f"threat_class:{verdict.threat_class.value}"
        epitopes.append(Epitope(
            kind="structural",
            signature=sig,
            weight=0.5,
            source_rule="threat_classification",
            hash=_hash_signature(sig),
        ))

    return epitopes


def _extract_semantic_epitope(pathogen: Pathogen) -> list[Epitope]:
    """
    Create a semantic epitope from the pathogen's antigenic signature.

    This is a marker indicating that the semantic embedding should be
    stored for future similarity search. The actual vector is stored
    in the KnownPathogen record, not in the epitope itself.
    """
    if pathogen.antigenic_signature.semantic_vector:
        sig = f"semantic:dim={pathogen.antigenic_signature.dimension}"
        return [Epitope(
            kind="semantic",
            signature=sig,
            weight=0.7,
            source_rule="embedding",
            hash=_hash_signature(sig),
        )]
    return []


def extract_epitopes(
    pathogen: Pathogen,
    verdict: QuarantineVerdict,
    sanitisation: SanitisationResult,
) -> EpitopeExtractionResult:
    """
    Main epitope extraction pipeline.

    Given a Pathogen that was quarantined and evaluated, extract the
    minimal set of distinctive signatures that identify this threat.

    Only runs for non-PASS verdicts — benign inputs don't generate
    antibodies.
    """
    start = time.monotonic()

    if verdict.action == QuarantineAction.PASS:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return EpitopeExtractionResult(
            epitopes=[],
            threat_class=verdict.threat_class,
            source_pathogen_id=pathogen.id,
            extraction_time_ms=elapsed_ms,
            total_candidates=0,
            retained_count=0,
        )

    # Gather candidates from all sources
    candidates: list[Epitope] = []
    candidates.extend(_extract_lexical_epitopes(pathogen, verdict, sanitisation))
    candidates.extend(_extract_structural_epitopes(pathogen, verdict))
    candidates.extend(_extract_semantic_epitope(pathogen))

    total_candidates = len(candidates)

    # Filter by minimum weight
    candidates = [e for e in candidates if e.weight >= _MIN_EPITOPE_WEIGHT]

    # Deduplicate by hash
    seen: set[str] = set()
    unique: list[Epitope] = []
    for ep in candidates:
        if ep.hash not in seen:
            seen.add(ep.hash)
            unique.append(ep)

    # Sort by weight descending, cap at max
    unique.sort(key=lambda e: e.weight, reverse=True)
    retained = unique[:_MAX_EPITOPES_PER_ANTIBODY]

    elapsed_ms = int((time.monotonic() - start) * 1000)

    logger.info(
        "epitope_extraction_complete",
        pathogen_id=pathogen.id,
        candidates=total_candidates,
        retained=len(retained),
        threat_class=verdict.threat_class.value,
        elapsed_ms=elapsed_ms,
    )

    return EpitopeExtractionResult(
        epitopes=retained,
        threat_class=verdict.threat_class,
        source_pathogen_id=pathogen.id,
        extraction_time_ms=elapsed_ms,
        total_candidates=total_candidates,
        retained_count=len(retained),
    )


# ─── Antibody Generation Pipeline ───────────────────────────────────────────


def generate_antibody(
    pathogen: Pathogen,
    verdict: QuarantineVerdict,
    extraction: EpitopeExtractionResult,
) -> KnownPathogen | None:
    """
    Generate a KnownPathogen record for the vector store.

    Given a quarantined Pathogen, its QuarantineVerdict, and extracted
    epitopes, compile a KnownPathogen suitable for insertion into the
    Qdrant pathogen store for future antigenic similarity matching.

    Returns None if:
    - The verdict was PASS (no threat to defend against)
    - No epitopes were extracted (nothing to match on)
    - The verdict explicitly said not to store as pathogen
    """
    if verdict.action == QuarantineAction.PASS:
        return None

    if not verdict.should_store_as_pathogen and not extraction.epitopes:
        logger.debug(
            "antibody_generation_skipped",
            pathogen_id=pathogen.id,
            reason="no_storage_flag_and_no_epitopes",
        )
        return None

    # Build description from epitopes
    epitope_summary = ", ".join(
        f"{e.kind}:{e.source_rule}" for e in extraction.epitopes[:5]
    )
    description = (
        f"Auto-generated from quarantine verdict: {verdict.action.value}. "
        f"Threat: {verdict.threat_class.value} ({verdict.severity.value}). "
        f"Epitopes: [{epitope_summary}]. "
        f"Confidence: {verdict.confidence:.2f}"
    )

    # Build tags from epitope signatures for metadata search
    tags = [f"threat:{verdict.threat_class.value}"]
    tags.append(f"severity:{verdict.severity.value}")
    tags.append(f"action:{verdict.action.value}")
    for ep in extraction.epitopes:
        tags.append(f"epitope:{ep.kind}:{ep.source_rule}")

    known_pathogen = KnownPathogen(
        id=new_id(),
        created_at=utc_now(),
        canonical_text=pathogen.text[:2000],  # Cap stored text
        threat_class=verdict.threat_class,
        severity=verdict.severity,
        structural_vector=pathogen.antigenic_signature.structural_vector,
        histogram_vector=pathogen.antigenic_signature.histogram_vector,
        semantic_vector=pathogen.antigenic_signature.semantic_vector,
        description=description,
        tags=tags,
    )

    logger.info(
        "antibody_generated",
        known_pathogen_id=known_pathogen.id,
        threat_class=verdict.threat_class.value,
        severity=verdict.severity.value,
        epitope_count=len(extraction.epitopes),
        has_semantic_vector=bool(known_pathogen.semantic_vector),
    )

    return known_pathogen


def suggest_innate_rule(
    pathogen: Pathogen,
    verdict: QuarantineVerdict,
    extraction: EpitopeExtractionResult,
) -> InnateRuleSuggestion | None:
    """
    If the threat has a clear, compact lexical pattern, suggest promoting
    it to the innate layer for instant future detection.

    Only suggests rules for HIGH or CRITICAL threats with high-confidence
    lexical epitopes. The suggestion is NOT auto-applied — it requires
    review and manual addition to the innate checks.
    """
    if verdict.severity not in (ThreatSeverity.HIGH, ThreatSeverity.CRITICAL):
        return None

    if verdict.confidence < 0.8:
        return None

    # Look for LLM-suggested antibody pattern
    if verdict.antibody_suggestion:
        # Validate it's a safe regex
        try:
            re.compile(verdict.antibody_suggestion)
            return InnateRuleSuggestion(
                pattern_regex=verdict.antibody_suggestion,
                threat_class=verdict.threat_class,
                severity=verdict.severity,
                description=(
                    f"LLM-suggested pattern from quarantine evaluation of "
                    f"pathogen {pathogen.id[:8]}"
                ),
                source_pathogen_id=pathogen.id,
                confidence=verdict.confidence,
            )
        except re.error:
            pass

    # Look for high-weight lexical epitopes from innate matches
    for ep in extraction.epitopes:
        if ep.kind == "lexical" and ep.weight >= 0.7:
            sig = ep.signature
            if sig.startswith("rule:"):
                continue  # Already an innate rule
            try:
                re.compile(sig)
                return InnateRuleSuggestion(
                    pattern_regex=sig,
                    threat_class=verdict.threat_class,
                    severity=verdict.severity,
                    description=(
                        f"High-weight lexical epitope from pathogen "
                        f"{pathogen.id[:8]} (weight={ep.weight:.2f})"
                    ),
                    source_pathogen_id=pathogen.id,
                    confidence=min(ep.weight, verdict.confidence),
                )
            except re.error:
                continue

    return None


# ─── Antibody Matching (Fast-Path Helper) ───────────────────────────────────


def match_known_pathogen_lexical(
    known: KnownPathogen,
    raw_text: str,
) -> float:
    """
    Score how well a raw text matches a known pathogen's canonical text
    using simple lexical overlap.

    This is a lightweight supplement to the vector similarity search.
    The primary matching is done by Qdrant cosine similarity on the
    embedding vectors. This function catches cases where the embedding
    model misses lexical-level recurrence.

    Returns a similarity score in [0.0, 1.0].
    """
    if not known.canonical_text or not raw_text:
        return 0.0

    # Tokenise both texts
    known_tokens = set(known.canonical_text.lower().split())
    input_tokens = set(raw_text.lower().split())

    if not known_tokens or not input_tokens:
        return 0.0

    # Jaccard similarity
    intersection = known_tokens & input_tokens
    union = known_tokens | input_tokens

    return len(intersection) / len(union)
