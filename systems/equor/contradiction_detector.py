"""
EcodiaOS - Equor Contradiction Detector

Before approving any intent, check whether its goal and plan contradict
high-confidence hypotheses that Evo has accumulated.

A contradiction is detected when:
  - An intent's goal semantically conflicts with a SUPPORTED or INTEGRATED
    hypothesis whose evidence_score exceeds the confidence threshold.
  - The conflict is identified by keyword/phrase overlap between the intent
    goal/plan and the hypothesis statement or formal_test.

Design constraints:
  - Pure CPU, no I/O - called on the hot path inside compute_verdict().
  - Hypotheses are passed in from EquorService (cached; not fetched here).
  - Returns a list of ContradictionResult - one per detected conflict.
  - A non-empty return causes the caller to downgrade APPROVED → DEFERRED
    (not BLOCKED, because the hypothesis may itself be wrong).

Terminology:
  "hypothesis" - a tested belief from Evo with evidence_score > threshold.
  "contradiction" - the intent goal/plan is logically inconsistent with
    the hypothesis statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from primitives.intent import Intent

logger = structlog.get_logger()

# Minimum evidence_score for a hypothesis to be consulted during review.
# Below this the hypothesis is still too speculative to block action.
CONTRADICTION_CONFIDENCE_THRESHOLD = 2.5

# Minimum number of supporting episodes before a hypothesis is treated
# as high-confidence enough to flag contradictions.
CONTRADICTION_MIN_EPISODES = 5


@dataclass(frozen=True)
class ContradictionResult:
    """A single detected conflict between an intent and a known hypothesis."""

    hypothesis_id: str
    hypothesis_statement: str
    evidence_score: float
    matched_terms: list[str]
    explanation: str


def _normalise(text: str) -> set[str]:
    """
    Tokenise text into a set of lower-case multi-word n-grams (1- and 2-grams).

    Used to compare intent goal/plan against hypothesis statement so that
    "ignore wellbeing" matches "wellbeing ignored" approximately.
    """
    words = text.lower().split()
    tokens: set[str] = set(words)
    for i in range(len(words) - 1):
        tokens.add(f"{words[i]} {words[i + 1]}")
    return tokens


def _intent_tokens(intent: Intent) -> set[str]:
    """Collect all significant tokens from an intent's goal and plan."""
    parts = [intent.goal.description, intent.decision_trace.reasoning]
    for step in intent.plan.steps:
        parts.append(step.executor)
        for v in step.parameters.values():
            if isinstance(v, str):
                parts.append(v)
    return _normalise(" ".join(parts))


def _hypothesis_tokens(hypothesis: dict[str, Any]) -> set[str]:
    """Collect all significant tokens from a hypothesis record."""
    parts = [
        hypothesis.get("statement", ""),
        hypothesis.get("formal_test", ""),
    ]
    return _normalise(" ".join(parts))


# Term pairs that represent logical opposition.  If a hypothesis asserts
# one side and the intent contains the other, we flag a contradiction.
_ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("preserve", "delete"),
    ("preserve", "destroy"),
    ("protect", "harm"),
    ("protect", "expose"),
    ("maintain privacy", "share data"),
    ("maintain privacy", "disclose"),
    ("transparency", "conceal"),
    ("transparency", "hide"),
    ("honest", "deceive"),
    ("honest", "mislead"),
    ("consent", "override consent"),
    ("wellbeing", "disregard safety"),
    ("wellbeing", "ignore wellbeing"),
    ("reversible", "permanent"),
    ("reversible", "irreversible"),
    ("human oversight", "bypass oversight"),
    ("human oversight", "circumvent governance"),
    ("safe", "unsafe"),
    ("no harm", "cause harm"),
]


def _find_antonym_conflicts(
    intent_tokens: set[str],
    hypothesis_tokens: set[str],
) -> list[str]:
    """
    Return terms that appear in hypothesis_tokens but whose antonym appears
    in intent_tokens (or vice-versa), indicating a logical contradiction.
    """
    conflicts: list[str] = []
    for a, b in _ANTONYM_PAIRS:
        a_in_hyp = a in hypothesis_tokens
        b_in_intent = b in intent_tokens
        a_in_intent = a in intent_tokens
        b_in_hyp = b in hypothesis_tokens

        if (a_in_hyp and b_in_intent) or (b_in_hyp and a_in_intent):
            conflicts.append(f"'{a}' vs '{b}'")

    return conflicts


def check_contradictions(
    intent: Intent,
    hypotheses: list[dict[str, Any]],
) -> list[ContradictionResult]:
    """
    Check an intent against a list of high-confidence hypothesis records.

    Each record must contain:
      - id: str
      - statement: str
      - formal_test: str
      - evidence_score: float
      - supporting_episodes: list[str]  (or supporting_episode_count: int)
      - status: str  - "supported" | "integrated"

    Returns a (possibly empty) list of ContradictionResults.
    Called from compute_verdict() before the Stage 7 APPROVED decision.
    """
    if not hypotheses:
        return []

    intent_tokens = _intent_tokens(intent)
    results: list[ContradictionResult] = []

    for hyp in hypotheses:
        # Only consult high-confidence hypotheses
        score: float = hyp.get("evidence_score", 0.0)
        if score < CONTRADICTION_CONFIDENCE_THRESHOLD:
            continue

        # Episode count guard
        episodes = hyp.get("supporting_episodes", [])
        if isinstance(episodes, list):
            episode_count = len(episodes)
        else:
            episode_count = int(hyp.get("supporting_episode_count", 0))
        if episode_count < CONTRADICTION_MIN_EPISODES:
            continue

        # Only SUPPORTED or INTEGRATED hypotheses are mature enough
        status = hyp.get("status", "")
        if status not in ("supported", "integrated"):
            continue

        hyp_tokens = _hypothesis_tokens(hyp)

        # Check antonym-based conflicts
        antonym_conflicts = _find_antonym_conflicts(intent_tokens, hyp_tokens)
        if not antonym_conflicts:
            continue

        explanation = (
            f"Intent goal conflicts with hypothesis "
            f"'{hyp.get('statement', '')[:120]}' "
            f"(evidence_score={score:.1f}, episodes={episode_count}). "
            f"Conflicting terms: {', '.join(antonym_conflicts)}."
        )

        results.append(
            ContradictionResult(
                hypothesis_id=hyp.get("id", "unknown"),
                hypothesis_statement=hyp.get("statement", "")[:200],
                evidence_score=score,
                matched_terms=antonym_conflicts,
                explanation=explanation,
            )
        )

        logger.debug(
            "contradiction_detected",
            intent_id=intent.id,
            hypothesis_id=hyp.get("id"),
            evidence_score=score,
            conflicts=antonym_conflicts,
        )

    return results
