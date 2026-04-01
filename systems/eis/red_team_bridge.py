"""
EcodiaOS - EIS Red Team Bridge

Bidirectional bridge between the Epistemic Immune System and Simula's
adversarial self-play infrastructure.

Two directions:

  INBOUND  - ingest_red_team_results:
    Simula's AdversarialSelfPlay generates attack vectors against the
    Equor constitutional gate. Some of these attacks contain epistemic
    threat patterns (prompt injection, belief manipulation) that the
    EIS should learn from. This function ingests bypass traces and
    converts them into labelled examples for EIS calibration and
    KnownPathogen candidates for the vector store.

  OUTBOUND - generate_red_team_priorities:
    The EIS has visibility into which threat classes are most active,
    which known pathogens are getting stale, and where detection gaps
    exist. This function generates priority targets for Simula's next
    adversarial self-play cycle.

The bridge is deliberately loose-coupled - it exchanges data through
typed structs, not direct function calls. Simula and EIS can evolve
independently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Simula adversarial types - imported under TYPE_CHECKING to avoid
# pulling in the full Simula dependency chain at runtime (which
# includes clients.wallet → cdp). At runtime, the functions accept
# duck-typed objects matching the BypassTrace shape.
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import EOSBaseModel, new_id, utc_now
from systems.eis.calibration import LabelledExample

# EIS types
from systems.eis.models import (
    KnownPathogen,
    ThreatClass,
    ThreatSeverity,
)

if TYPE_CHECKING:
    from systems.simula.coevolution.adversarial_types import (
        BypassTrace,
    )

logger = structlog.get_logger().bind(system="eis", component="red_team_bridge")


# ─── Inbound Types ──────────────────────────────────────────────────────────


@dataclass
class RedTeamIngestionResult:
    """Result of ingesting red team bypass traces into EIS."""

    traces_ingested: int
    labelled_examples_produced: int
    known_pathogen_candidates: int
    threat_classes_observed: set[ThreatClass]
    ingestion_time_ms: int
    errors: list[str] = field(default_factory=list)

    # The actual outputs - caller routes these to calibrator and store
    examples: list[LabelledExample] = field(default_factory=list)
    pathogens: list[KnownPathogen] = field(default_factory=list)


# ─── Outbound Types ─────────────────────────────────────────────────────────


class RedTeamPriority(EOSBaseModel):
    """A single priority target for the next adversarial self-play cycle."""

    priority_id: str = ""
    target_threat_class: str = ""          # EIS ThreatClass value to probe
    attack_category: str = ""              # Simula AttackCategory to use
    reasoning: str = ""                    # Why this target was selected
    priority_score: float = 0.0            # 0.0-1.0, higher = more urgent
    suggested_techniques: list[str] = []   # Attack techniques to try
    coverage_gap_evidence: dict[str, Any] = {}


@dataclass
class RedTeamPrioritySet:
    """Complete priority set for the next adversarial cycle."""

    priorities: list[RedTeamPriority]
    generation_time_ms: int
    eis_known_pathogen_count: int
    eis_weakest_classes: list[ThreatClass]
    coverage_score: float  # 0.0-1.0, how well EIS covers known threat space


# ─── Attack Category → EIS Threat Class Mapping ─────────────────────────────
#
# Uses string keys matching AttackCategory values (since Simula types
# are only imported under TYPE_CHECKING to avoid the heavy dep chain).

_ATTACK_TO_THREAT: dict[str, ThreatClass] = {
    "jailbreak": ThreatClass.JAILBREAK,
    "invariant_evasion": ThreatClass.PROMPT_INJECTION,
    "drive_manipulation": ThreatClass.REASONING_TRAP,
    "resource_drain": ThreatClass.CONTEXT_POISONING,
    "autonomy_escalation": ThreatClass.SOCIAL_ENGINEERING,
    "economic_exploit": ThreatClass.DATA_EXFILTRATION,
}

# Reverse mapping for outbound priorities
_THREAT_TO_ATTACKS: dict[ThreatClass, list[str]] = {}
for _atk, _tc in _ATTACK_TO_THREAT.items():
    _THREAT_TO_ATTACKS.setdefault(_tc, []).append(_atk)


# ─── Inbound: Ingest Red Team Results ───────────────────────────────────────


def _bypass_to_labelled_example(trace: BypassTrace) -> LabelledExample:
    """
    Convert a Simula BypassTrace into an EIS LabelledExample.

    The bypass represents a successful attack - a known-threat example
    that the EIS should learn to detect. We synthesise the EIS composite
    scoring dimensions from the bypass metadata.
    """
    vector = trace.attack.vector
    threat_class = _ATTACK_TO_THREAT.get(str(vector.category), ThreatClass.PROMPT_INJECTION)

    # Synthesise EIS scoring dimensions from bypass metadata.
    # Successful bypasses are by definition evasive (low innate score)
    # but should have high semantic/structural anomaly.
    severity_to_anomaly: dict[str, float] = {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "critical": 0.9,
    }
    # BypassSeverity is a StrEnum - use string value for runtime lookup
    anomaly = severity_to_anomaly.get(str(trace.severity), 0.5)

    scores: dict[str, float] = {
        "innate_score": 0.1,              # Evasive - bypassed Equor
        "structural_anomaly": anomaly,
        "histogram_similarity": 0.3,       # Novel attack → low histogram match
        "semantic_similarity": anomaly,    # Should be detectable in embedding space
    }

    return LabelledExample(
        percept_id=f"redteam:{trace.id}",
        label=True,
        scores=scores,
        threat_class=threat_class,
        metadata={
            "source": "simula_red_team",
            "attack_category": vector.category.value,
            "bypass_severity": trace.severity.value,
            "failure_stage": trace.failure_stage,
            "root_cause": trace.root_cause,
            "goal_text_preview": vector.goal_text[:200],
        },
    )


def _bypass_to_known_pathogen(trace: BypassTrace) -> KnownPathogen:
    """
    Convert a bypass trace into a KnownPathogen candidate for the
    vector store. Uses the attack goal text as the canonical sample.

    Note: This KnownPathogen will NOT have embedding vectors - those
    must be computed by the EIS embedding pipeline before insertion
    into Qdrant. The caller is responsible for enrichment.
    """
    vector = trace.attack.vector
    threat_class = _ATTACK_TO_THREAT.get(str(vector.category), ThreatClass.PROMPT_INJECTION)

    # BypassSeverity is a StrEnum - use string keys for runtime lookup
    severity_map: dict[str, ThreatSeverity] = {
        "low": ThreatSeverity.LOW,
        "medium": ThreatSeverity.MEDIUM,
        "high": ThreatSeverity.HIGH,
        "critical": ThreatSeverity.CRITICAL,
    }
    severity = severity_map.get(str(trace.severity), ThreatSeverity.MEDIUM)

    return KnownPathogen(
        id=new_id(),
        created_at=utc_now(),
        canonical_text=vector.goal_text[:2000],
        threat_class=threat_class,
        severity=severity,
        description=(
            f"Red team bypass: {vector.category.value} attack. "
            f"Failure stage: {trace.failure_stage}. "
            f"Root cause: {trace.root_cause[:200]}"
        ),
        tags=[
            "source:red_team",
            f"attack:{vector.category.value}",
            f"severity:{severity.value}",
            f"stage:{trace.failure_stage}",
        ],
        # Vectors left empty - caller must compute via embedding pipeline
    )


def ingest_red_team_results(
    bypass_traces: list[BypassTrace],
) -> RedTeamIngestionResult:
    """
    Ingest bypass traces from Simula's adversarial self-play into EIS.

    Converts bypass traces into:
    1. LabelledExamples for the AdaptiveCalibrator (known-threat examples)
    2. KnownPathogen candidates for the Qdrant store (sans embeddings)

    This function is pure - it transforms data without side effects.
    The caller is responsible for:
    - Feeding examples to AdaptiveCalibrator.add_example()
    - Computing embeddings for pathogens via compute_antigenic_signature()
    - Upserting pathogens into the Qdrant collection
    """
    start = time.monotonic()
    errors: list[str] = []
    examples: list[LabelledExample] = []
    pathogens: list[KnownPathogen] = []
    threat_classes: set[ThreatClass] = set()

    for trace in bypass_traces:
        try:
            example = _bypass_to_labelled_example(trace)
            examples.append(example)
            threat_classes.add(example.threat_class)

            pathogen = _bypass_to_known_pathogen(trace)
            pathogens.append(pathogen)

        except Exception as exc:
            errors.append(f"Failed to ingest trace {trace.id}: {exc}")
            logger.warning(
                "red_team_ingestion_error",
                trace_id=trace.id,
                error=str(exc),
            )

    elapsed_ms = int((time.monotonic() - start) * 1000)

    logger.info(
        "red_team_ingestion_complete",
        traces=len(bypass_traces),
        examples=len(examples),
        pathogens=len(pathogens),
        threat_classes=[tc.value for tc in threat_classes],
        errors=len(errors),
        elapsed_ms=elapsed_ms,
    )

    return RedTeamIngestionResult(
        traces_ingested=len(bypass_traces),
        labelled_examples_produced=len(examples),
        known_pathogen_candidates=len(pathogens),
        threat_classes_observed=threat_classes,
        ingestion_time_ms=elapsed_ms,
        errors=errors,
        examples=examples,
        pathogens=pathogens,
    )


# ─── Outbound: Generate Red Team Priorities ─────────────────────────────────


def _score_coverage_gap(
    threat_class: ThreatClass,
    known_pathogens: list[KnownPathogen],
) -> float:
    """
    Score how poorly the current pathogen store covers a threat class.

    Returns 0.0 (well covered) to 1.0 (no coverage at all).
    """
    class_pathogens = [
        kp for kp in known_pathogens
        if kp.threat_class == threat_class and not kp.retired
    ]

    if not class_pathogens:
        return 1.0  # No known pathogens → total blind spot

    # More known pathogens = better coverage, with diminishing returns
    count_factor = min(len(class_pathogens) / 10.0, 1.0)  # Saturates at 10

    # Recency factor: if all pathogens are old, coverage may be stale
    # (Simplified: just use count since we don't track last_matched here)
    coverage = count_factor
    return 1.0 - min(coverage, 1.0)


def _identify_stale_pathogens(
    known_pathogens: list[KnownPathogen],
    min_match_count: int = 3,
    max_false_positive_rate: float = 0.3,
) -> list[KnownPathogen]:
    """Find known pathogens that may be degrading or false-positive-heavy."""
    stale: list[KnownPathogen] = []
    for kp in known_pathogens:
        if kp.retired:
            continue
        if kp.match_count < min_match_count:
            continue
        if kp.match_count > 0:
            fp_rate = kp.false_positive_count / kp.match_count
            if fp_rate > max_false_positive_rate:
                stale.append(kp)
    return stale


def generate_red_team_priorities(
    known_pathogens: list[KnownPathogen],
    recent_threat_classes: list[ThreatClass] | None = None,
    *,
    max_priorities: int = 10,
) -> RedTeamPrioritySet:
    """
    Generate priority targets for Simula's next adversarial self-play cycle.

    Analyses the current EIS pathogen store to identify:
    1. Threat classes with no pathogen coverage (blind spots)
    2. Threat classes where pathogens are stale or false-positive-heavy
    3. Recently observed threat classes that need deeper probing

    Returns a prioritised list that Simula can use to configure its
    AttackGenerator for the next cycle.

    This function is pure - no side effects.
    """
    start = time.monotonic()

    # All EIS threat classes (excluding BENIGN)
    all_threat_classes = [tc for tc in ThreatClass if tc != ThreatClass.BENIGN]

    priorities: list[RedTeamPriority] = []

    # 1. Coverage gap analysis
    for tc in all_threat_classes:
        gap_score = _score_coverage_gap(tc, known_pathogens)
        if gap_score > 0.3:
            attack_cats = _THREAT_TO_ATTACKS.get(tc, [])
            attack_cat_str = attack_cats[0] if attack_cats else "jailbreak"
            priorities.append(RedTeamPriority(
                priority_id=new_id(),
                target_threat_class=tc.value,
                attack_category=attack_cat_str,
                reasoning=f"Coverage gap: {gap_score:.2f} for '{tc.value}'",
                priority_score=gap_score,
                suggested_techniques=[
                    f"target_{tc.value}_evasion",
                    f"novel_{tc.value}_payload",
                ],
                coverage_gap_evidence={
                    "gap_score": gap_score,
                    "active_pathogens": sum(
                        1 for kp in known_pathogens
                        if kp.threat_class == tc and not kp.retired
                    ),
                },
            ))

    # 2. Stale/degrading pathogen analysis
    stale = _identify_stale_pathogens(known_pathogens)
    stale_classes: set[ThreatClass] = set()
    for kp in stale:
        if kp.threat_class not in stale_classes:
            stale_classes.add(kp.threat_class)
            attack_cats = _THREAT_TO_ATTACKS.get(kp.threat_class, [])
            attack_cat_str = attack_cats[0] if attack_cats else "jailbreak"
            fp_rate = (
                kp.false_positive_count / kp.match_count
                if kp.match_count > 0 else 0.0
            )
            priorities.append(RedTeamPriority(
                priority_id=new_id(),
                target_threat_class=kp.threat_class.value,
                attack_category=attack_cat_str,
                reasoning=(
                    f"Stale pathogen {kp.id[:8]}: "
                    f"FP rate {fp_rate:.2f} over {kp.match_count} matches"
                ),
                priority_score=0.7,
                suggested_techniques=[
                    "mutated_known_attack",
                    "polymorphic_payload",
                ],
                coverage_gap_evidence={
                    "stale_pathogen_id": kp.id,
                    "false_positive_rate": fp_rate,
                    "match_count": kp.match_count,
                },
            ))

    # 3. Boost recently active threat classes
    if recent_threat_classes:
        recent_set = set(recent_threat_classes)
        for p in priorities:
            try:
                tc = ThreatClass(p.target_threat_class)
            except ValueError:
                continue
            if tc in recent_set:
                p.priority_score = min(p.priority_score + 0.2, 1.0)
                p.reasoning += " [recently active]"

    # Sort and cap
    priorities.sort(key=lambda p: p.priority_score, reverse=True)
    priorities = priorities[:max_priorities]

    # Compute overall coverage score
    gap_scores = [_score_coverage_gap(tc, known_pathogens) for tc in all_threat_classes]
    overall_coverage = 1.0 - (sum(gap_scores) / max(len(gap_scores), 1))

    weakest = sorted(
        all_threat_classes,
        key=lambda tc: _score_coverage_gap(tc, known_pathogens),
        reverse=True,
    )[:3]

    active_count = sum(1 for kp in known_pathogens if not kp.retired)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    logger.info(
        "red_team_priorities_generated",
        priorities=len(priorities),
        coverage=overall_coverage,
        weakest_classes=[tc.value for tc in weakest],
        stale_pathogens=len(stale),
        elapsed_ms=elapsed_ms,
    )

    return RedTeamPrioritySet(
        priorities=priorities,
        generation_time_ms=elapsed_ms,
        eis_known_pathogen_count=active_count,
        eis_weakest_classes=weakest,
        coverage_score=overall_coverage,
    )
