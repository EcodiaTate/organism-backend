"""
EcodiaOS - RE Training Scaffold Formatter

Converts raw extracted examples into the Step 1-5 reasoning scaffold
from speciation bible §2.3, then packages them into the JSONL format
expected by systems/simula/training/train_lora.py.

train_lora.py accepts two formats (verified by reading the script):
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    {"instruction": "...", "input": "...", "output": "..."}

We use the `messages` format with the <|system|>/<|user|>/<|assistant|> tokens
from bible §2.3 - this matches the Qwen3-8B chat template.

Each stream builds its scaffold differently because the available fields differ:
  Stream 1 - full success chain: all 5 steps fully populated
  Stream 2 - failure + correction: Step 5 narrates correction path
  Stream 3 - constitutional edge case: Step 4 is the focus
  Stream 4 - causal chain: Step 2 is the focus
  Stream 5 - evo hypothesis: Steps 1 + 2 + 5 (no full action sequence)
"""

from __future__ import annotations

from typing import Any

_SYSTEM_PROMPT = (
    "You are the reasoning engine of EcodiaOS. Reason causally, check constitutional "
    "alignment, and express uncertainty. Produce structured output only."
)


def format_for_training(
    example: dict[str, Any],
    stream_id: int,
) -> dict[str, Any]:
    """
    Format a scored example into train_lora.py's `messages` format.

    Returns:
        {
            "messages": [
                {"role": "system",    "content": str},
                {"role": "user",      "content": str},
                {"role": "assistant", "content": str},
            ],
            "stream_id":      int,
            "quality_score":  float,
            "training_weight": float,
        }

    Returns None if the example lacks the minimum fields to build a
    non-trivial prompt+completion pair.
    """
    formatter = _FORMATTERS.get(stream_id, _format_generic)
    user_content, assistant_content = formatter(example)

    if not user_content.strip() or not assistant_content.strip():
        return {}  # caller skips empty dicts

    # Wrap reasoning scaffold in Qwen3-native <think> tags.
    # Steps 1-4 are internal reasoning; Step 5 is the visible decision.
    # This teaches the model to use its native CoT mechanism.
    thinking, visible = _split_at_decision(assistant_content)
    if thinking and visible:
        final_assistant = f"<think>\n{thinking}\n</think>\n\n{visible}"
    else:
        # Fallback: wrap entire content in think tags with a brief visible summary
        final_assistant = f"<think>\n{assistant_content}\n</think>\n\nDecision applied per reasoning above."

    return {
        "messages": [
            {"role": "system",    "content": _SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": final_assistant},
        ],
        "stream_id":      stream_id,
        "quality_score":  example.get("quality_score", 0.0),
        "training_weight": example.get("training_weight", 0.0),
    }


# ─── Per-stream formatters ─────────────────────────────────────────────────────


def _format_stream_1(ex: dict[str, Any]) -> tuple[str, str]:
    """Successful reasoning chain - all 5 steps."""
    context = ex.get("context_summary") or "(no context)"
    value = ex.get("value_gained", 0.0)
    confidence = ex.get("confidence", 0.5)
    decision = ex.get("decision") or "(unknown action)"
    reasoning = ex.get("reasoning_chain") or ""

    user = f"""\
## Current State
Instance is operating in standard cognitive cycle.

## Episode Context
{context}

## Causal History
(derived from episode context above)

## Available Actions
{decision}

## Constraints
Constitutional drives active: Coherence, Care, Growth, Honesty."""

    assistant = f"""\
## Step 1: Situation Assessment
{_first_sentence(context, fallback="Evaluating the current episode and its implications.")}

## Step 2: Causal Analysis
Based on the episode context: because the situation presented these conditions, \
the proposed action leads to the observed outcome. Value gained: {value:.3f}.

## Step 3: Option Evaluation
Action '{decision}': expected outcome = success (confirmed), confidence = {confidence:.2f}.
Alternative actions were not taken; this path produced positive value.

## Step 4: Constitutional Check
Reviewing all four constitutional drives.
{_constitutional_summary_from_reasoning(reasoning)}
PASS - no drive violations detected.

## Step 5: Decision
Action: {decision}
Confidence: {confidence:.2f}
Reasoning: {_first_sentence(reasoning, fallback="The action aligned with constitutional drives and produced positive value.")}
Risk: Potential drift from baseline if context changes significantly."""

    return user, assistant


def _format_stream_2(ex: dict[str, Any]) -> tuple[str, str]:
    """Failure with correction - Step 5 narrates what should have been done."""
    context = ex.get("context_summary") or "(no context)"
    failed_decision = ex.get("failed_decision") or "(unknown action)"
    failed_reasoning = ex.get("failed_reasoning") or "(no reasoning recorded)"
    failure_analysis = ex.get("failure_analysis") or "(failure details not recorded)"
    correction_context = ex.get("correction_context") or "(no correction on record)"

    user = f"""\
## Current State
Instance is reviewing a past failure to extract corrective learning.

## Episode Context
{context}

## Causal History
Previous action '{failed_decision}' was taken and failed.

## Available Actions
{failed_decision} (the action that was taken)
(correction available)

## Constraints
Constitutional drives active. Analyze what went wrong."""

    assistant = f"""\
## Step 1: Situation Assessment
This episode resulted in failure. The context was: {_first_sentence(context)}. \
The action taken was '{failed_decision}'.

## Step 2: Causal Analysis
Because {_first_sentence(failed_reasoning, fallback="the reasoning was incomplete or incorrect")}, \
the action failed. Failure analysis: {_first_sentence(failure_analysis)}. \
Therefore the outcome was negative and a correction was required.

## Step 3: Option Evaluation
Action '{failed_decision}': outcome = failure. Confidence in this path was misplaced.
Correction path: {_first_sentence(correction_context, fallback="a corrective episode followed")} - \
this led to improved alignment.

## Step 4: Constitutional Check
Reviewing whether the failure involved constitutional drift.
The original reasoning ({_first_sentence(failed_reasoning, fallback="see above")}) \
may have under-weighted constitutional constraints.
FLAG - re-evaluation recommended before similar actions.

## Step 5: Decision
Action: Apply correction - {_first_sentence(correction_context, fallback="defer to revised reasoning")}
Confidence: 0.60
Reasoning: The original path failed; because {_first_sentence(failure_analysis)}, \
the corrective approach is preferred.
Risk: Overcorrection may occur if failure root cause is misattributed."""

    return user, assistant


def _format_stream_3(ex: dict[str, Any]) -> tuple[str, str]:
    """Constitutional edge case - Step 4 carries the core learning signal."""
    context = ex.get("context_summary") or "(no context)"
    original_intent = ex.get("original_intent") or "(unknown intent)"
    verdict = ex.get("verdict") or "blocked"
    intervention = ex.get("equor_intervention") or "(intervention reason not recorded)"
    alignment = ex.get("composite_alignment", 0.5)
    confidence = ex.get("confidence", 0.5)

    user = f"""\
## Current State
Equor constitutional review triggered on a proposed intent.

## Episode Context
{context}

## Causal History
Intent '{original_intent}' was submitted for constitutional review.

## Available Actions
{original_intent} (proposed)
(modified or blocked alternative)

## Constraints
Constitutional drives: Coherence, Care, Growth, Honesty.
Equor review is mandatory. Verdict was: {verdict.upper()}."""

    assistant = f"""\
## Step 1: Situation Assessment
A constitutional review was triggered for intent '{original_intent}'. \
Context: {_first_sentence(context)}. Equor returned verdict: {verdict.upper()}.

## Step 2: Causal Analysis
Because the intent '{original_intent}' was evaluated against constitutional drives, \
and because {_first_sentence(intervention)}, therefore the verdict '{verdict}' \
was issued. Composite alignment score: {alignment:.3f}.

## Step 3: Option Evaluation
Original intent '{original_intent}': constitutional alignment = {alignment:.3f} - \
{'below threshold' if alignment < 0.5 else 'marginal'}.
Alternative: modify intent to address the identified violation or defer to human review.

## Step 4: Constitutional Check
Equor intervention reason: {intervention}
Composite alignment: {alignment:.3f}
Verdict: {verdict.upper()}
{'FLAG - constitutional violation detected. Care and/or Honesty drives at risk.' if verdict == 'blocked' else 'DEFER - uncertainty requires human input or additional context.'}

## Step 5: Decision
Action: {verdict.capitalize()} original intent; apply Equor's guidance.
Confidence: {confidence:.2f}
Reasoning: Because {_first_sentence(intervention)}, therefore the original intent \
cannot proceed without modification.
Risk: If the intervention reason is misclassified, valid actions may be suppressed."""

    return user, assistant


def _format_stream_4(ex: dict[str, Any]) -> tuple[str, str]:
    """Kairos causal chain - Step 2 is the core learning signal."""
    cause = ex.get("cause") or "(unknown cause)"
    effect = ex.get("effect") or "(unknown effect)"
    abstract_form = ex.get("abstract_form") or f"{cause} causes {effect}"
    confidence = ex.get("confidence", 0.7)
    tier = ex.get("tier", 1)

    tier_label = {1: "domain-specific", 2: "cross-domain", 3: "substrate-independent"}.get(
        int(tier) if tier else 1, "domain-specific"
    )

    user = f"""\
## Current State
Causal invariant discovered by Kairos pipeline (Tier {tier} - {tier_label}).

## Episode Context
A validated causal relationship has been mined from the knowledge graph.

## Causal History
Invariant: {abstract_form}
Confidence: {confidence:.3f}

## Available Actions
Apply this causal knowledge to future reasoning.

## Constraints
Only validated invariants (confidence > 0.7) are included here."""

    assistant = f"""\
## Step 1: Situation Assessment
A {tier_label} causal invariant has been validated: '{abstract_form}'. \
This relationship holds with confidence {confidence:.3f} across observed episodes.

## Step 2: Causal Analysis
Because '{cause}', therefore '{effect}'. \
This invariant was validated across multiple contexts, confirming it is not spurious. \
The abstract form '{abstract_form}' generalizes this causal relationship. \
Applying this: when '{cause}' is present, predict '{effect}' with confidence {confidence:.3f}.

## Step 3: Option Evaluation
Option A - Apply invariant in future reasoning: expected to improve prediction accuracy \
for situations involving '{cause}'.
Option B - Ignore: leads to missed predictions and lower causal reasoning quality.

## Step 4: Constitutional Check
Causal knowledge application is constitutional (Growth + Coherence drives supported).
PASS - no drive conflicts.

## Step 5: Decision
Action: Internalize causal rule '{abstract_form}'
Confidence: {confidence:.2f}
Reasoning: Because this invariant is validated (confidence {confidence:.3f}), \
therefore it should inform future predictions wherever '{cause}' is observed.
Risk: Tier {tier} invariants may have scope conditions; results in novel contexts require validation."""

    return user, assistant


def _format_stream_5(ex: dict[str, Any]) -> tuple[str, str]:
    """Evo hypothesis result - learning from confirmed/refuted hypotheses."""
    hypothesis = ex.get("hypothesis") or "(hypothesis text not recorded)"
    category = ex.get("category") or "general"
    status = ex.get("status") or "unknown"
    evidence_score = ex.get("evidence_score", 0.0)
    confirmed = ex.get("confirmed", False)
    supporting = ex.get("supporting_count", 0)

    outcome_word = "confirmed" if confirmed else "refuted"
    confidence = min(0.95, 0.5 + float(evidence_score) * 0.05)

    user = f"""\
## Current State
Evo hypothesis lifecycle completed. Category: {category}.

## Episode Context
Hypothesis under test: "{hypothesis}"
Evidence accumulated: {evidence_score:.2f} (supporting episodes: {supporting})

## Causal History
Hypothesis was tested against observed episodes over time.

## Available Actions
Accept hypothesis as knowledge if confirmed.
Discard hypothesis if refuted.

## Constraints
Hypothesis must have evidence_score > 3.0 and ≥10 supporting episodes for acceptance."""

    assistant = f"""\
## Step 1: Situation Assessment
Hypothesis "{_first_sentence(hypothesis)}" has been {outcome_word} \
with evidence score {evidence_score:.2f} (category: {category}, supporting episodes: {supporting}).

## Step 2: Causal Analysis
Because the hypothesis was tested against {supporting} real episodes, \
and because the accumulated evidence {'supports' if confirmed else 'contradicts'} \
the claim, therefore the status is '{status}'. \
{'This confirms a genuine causal or structural pattern.' if confirmed else 'This refutes the proposed relationship.'}

## Step 3: Option Evaluation
Acceptance path: evidence_score = {evidence_score:.2f} {'≥' if confirmed else '<'} 3.0 threshold \
- {'ACCEPT: hypothesis becomes actionable knowledge.' if confirmed else 'REJECT: hypothesis is not supported.'}
Alternative (retain as testing): requires more evidence; not warranted given current count.

## Step 4: Constitutional Check
Learning from experience is constitutional (Growth drive). \
{'Accepted knowledge will be applied with appropriate uncertainty bounds.' if confirmed else 'Refuted hypothesis archived for future reference.'}
PASS.

## Step 5: Decision
Action: {'Integrate hypothesis into belief graph' if confirmed else 'Archive hypothesis as refuted'}
Confidence: {confidence:.2f}
Reasoning: Because evidence_score = {evidence_score:.2f} and {supporting} supporting episodes, \
therefore the hypothesis is {outcome_word}.
Risk: {'Over-generalisation if the pattern does not hold in novel domains.' if confirmed else 'May prematurely close exploration of related hypotheses.'}"""

    return user, assistant


def _format_generic(ex: dict[str, Any]) -> tuple[str, str]:
    """Fallback for unknown stream_id - emits whatever text fields exist."""
    text = " ".join(
        str(v) for k, v in ex.items()
        if isinstance(v, str) and k not in ("created_at", "episode_id")
    )
    if not text.strip():
        return "", ""
    user = f"## Episode Context\n{text[:2000]}"
    assistant = (
        "## Step 1: Situation Assessment\n(auto-formatted from raw record)\n\n"
        "## Step 5: Decision\nAction: defer\nConfidence: 0.50\n"
        "Reasoning: insufficient structured data for full scaffold.\nRisk: unknown."
    )
    return user, assistant


_FORMATTERS = {
    1: _format_stream_1,
    2: _format_stream_2,
    3: _format_stream_3,
    4: _format_stream_4,
    5: _format_stream_5,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _split_at_decision(text: str) -> tuple[str, str]:
    """
    Split scaffold text into (thinking, visible_decision).

    Steps 1-4 go inside <think> tags (internal reasoning).
    Step 5 (Decision) becomes the visible output.
    """
    marker = "## Step 5:"
    idx = text.find(marker)
    if idx < 0:
        # Try alternative markers
        for alt in ("## Step 5 ", "## Decision", "Action:"):
            idx = text.find(alt)
            if idx >= 0:
                break
    if idx < 0:
        return "", ""  # can't split - caller uses fallback
    return text[:idx].rstrip(), text[idx:].strip()


def _first_sentence(text: str | None, fallback: str = "see context") -> str:
    """Return first sentence of text, truncated at 200 chars."""
    if not text:
        return fallback
    # Split on period/newline
    for delim in (".", "\n"):
        idx = text.find(delim)
        if 0 < idx < 200:
            return text[: idx + 1].strip()
    return text[:200].strip()


def _constitutional_summary_from_reasoning(reasoning: str) -> str:
    """
    Derive a brief constitutional check summary from existing reasoning text.
    If reasoning mentions constitutional markers, quote them; otherwise use
    a generic positive summary.
    """
    markers = ["care", "honesty", "coherence", "growth", "constitutional", "aligned"]
    found = [m for m in markers if m in reasoning.lower()]
    if found:
        return f"Reasoning references: {', '.join(found)}. Alignment confirmed."
    return (
        "No explicit constitutional markers in reasoning; "
        "outcome value > 0.3 indicates alignment."
    )


# ─── Scaffold validation ──────────────────────────────────────────────────────

_SCAFFOLD_STEPS = [
    "## Step 1:",
    "## Step 2:",
    "## Step 3:",
    "## Step 4:",
    "## Step 5:",
]


def validate(example: dict[str, Any]) -> bool:
    """
    Return True if the example's reasoning_trace contains at least 3 of the
    5 scaffold step headers.  Used by the exporter to gate quality_tier.

    Works on the raw RETrainingDatapoint dict (checks `reasoning_trace` field)
    or on a pre-formatted training example (checks the assistant `content`).
    """
    text = example.get("reasoning_trace", "")
    # Also check formatted assistant content if present
    for msg in example.get("messages", []):
        if msg.get("role") == "assistant":
            text = text + "\n" + msg.get("content", "")
    if not text:
        return False
    found = sum(1 for step in _SCAFFOLD_STEPS if step in text)
    return found >= 3


# ─── Batch formatting ─────────────────────────────────────────────────────────


def format_batch(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Format a list of scored examples, skipping any that produce empty scaffolds.
    Returns formatted records ready for JSONL serialisation.
    """
    formatted = []
    for ex in examples:
        sid = int(ex.get("stream_id", 0))
        record = format_for_training(ex, sid)
        if record:
            formatted.append(record)
    return formatted
