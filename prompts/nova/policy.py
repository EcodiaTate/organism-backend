"""
EcodiaOS - Nova Policy Generation Prompts

Prompts for LLM-based policy generation and EFE component estimation.
These prompts ground the LLM's reasoning in the current belief state,
affect context, and constitutional drives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from primitives.affect import AffectState
    from systems.nova.types import BeliefState, Goal


def build_policy_generation_prompt(
    instance_name: str,
    goal: Goal,
    situation_summary: str,
    beliefs_summary: str,
    memory_summary: str,
    affect: AffectState,
    available_action_types: list[str],
    max_policies: int = 5,
    causal_laws_summary: str = "",
) -> str:
    """
    Build the policy generation prompt.

    This is the core reasoning prompt that asks the LLM to generate
    candidate policies for achieving a goal. All grounding context
    (beliefs, memories, affect, available actions) is included.
    """
    affect_desc = _describe_affect(affect)
    action_types_str = "\n".join(f"  - {a}" for a in available_action_types)

    return f"""Policy generation for {instance_name}.

## SITUATION
{situation_summary}

## GOAL
{goal.description}
Success criteria: {goal.success_criteria or "Infer from goal"}
Urgency: {"High" if goal.urgency > 0.6 else "Moderate" if goal.urgency > 0.3 else "Low"}

## WORLD BELIEFS
{beliefs_summary}
{f"""
## CAUSAL LAWS (Kairos-validated)
{causal_laws_summary}
""" if causal_laws_summary else ""}
## MEMORY
{memory_summary or "None retrieved."}

## STATE
{affect_desc}

## AVAILABLE ACTIONS
{action_types_str}

Generate up to {max_policies} distinct candidate strategies. Respond with valid JSON:
{{
  "policies": [
    {{
      "name": "Brief name",
      "reasoning": "Why this works given the situation",
      "steps": [
        {{
          "action_type": "action type",
          "description": "What to do",
          "parameters": {{}}
        }}
      ],
      "risks": ["Risk 1"],
      "epistemic_value": "What this teaches",
      "estimated_effort": "your honest assessment of effort required",
      "time_horizon": "your honest assessment of when results are expected"
    }}
  ]
}}\n"""


def build_pragmatic_value_prompt(
    policy_name: str,
    policy_reasoning: str,
    policy_steps_desc: str,
    goal_description: str,
    goal_success_criteria: str,
    beliefs_summary: str,
) -> str:
    """
    Prompt for estimating pragmatic value (probability of goal achievement).
    Returns a probability estimate and brief reasoning.
    """
    return f"""Estimate the probability that this strategy achieves the stated goal.

## STRATEGY: {policy_name}
Reasoning: {policy_reasoning}
Steps: {policy_steps_desc}

## GOAL
{goal_description}
Success criteria: {goal_success_criteria or "Infer from goal description"}

## CURRENT WORLD STATE
{beliefs_summary}

Respond in JSON:
{{
  "success_probability": 0.0 to 1.0,
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of your estimate (2-3 sentences)"
}}"""


def build_epistemic_value_prompt(
    policy_name: str,
    policy_steps_desc: str,
    beliefs_summary: str,
    known_uncertainties: str,
) -> str:
    """
    Prompt for estimating epistemic value (expected information gain).
    """
    return f"""Estimate how much new information this strategy would provide.

## STRATEGY: {policy_name}
Steps: {policy_steps_desc}

## CURRENT KNOWLEDGE STATE
{beliefs_summary}

## KNOWN UNCERTAINTIES
{known_uncertainties or "Not specified - infer from context"}

Respond in JSON:
{{
  "info_gain": 0.0 to 1.0,
  "novelty": 0.0 to 1.0,
  "uncertainties_addressed": integer count,
  "reasoning": "What specifically we would learn"
}}"""


# ─── Helpers ─────────────────────────────────────────────────────


def _describe_affect(affect: AffectState) -> str:
    """Convert AffectState to raw values for prompt grounding.
    Raw values let the LLM interpret significance contextually rather than
    having heuristic thresholds decide what 'high curiosity' means."""
    fields = {
        "valence": affect.valence,
        "curiosity": affect.curiosity,
        "care_activation": affect.care_activation,
        "coherence_stress": affect.coherence_stress,
        "arousal": affect.arousal,
    }
    return "  ".join(f"{k}={v:.3f}" for k, v in fields.items())


def summarise_beliefs(beliefs: BeliefState, max_entities: int = 5) -> str:
    """Summarise the belief state for LLM prompt inclusion."""
    lines: list[str] = []

    if beliefs.current_context.summary:
        lines.append(f"Context: {beliefs.current_context.summary[:150]}")
    lines.append(f"Domain: {beliefs.current_context.domain or 'general'}")
    lines.append(f"Active dialogue: {beliefs.current_context.is_active_dialogue}")

    if beliefs.entities:
        top_entities = sorted(
            beliefs.entities.values(),
            key=lambda e: e.confidence,
            reverse=True,
        )[:max_entities]
        entity_strs = [f"{e.name} ({e.entity_type}, conf={e.confidence:.2f})" for e in top_entities]
        lines.append(f"Known entities: {', '.join(entity_strs)}")

    if beliefs.individual_beliefs:
        individual_strs = [
            f"{iid}: valence={b.estimated_valence:.2f}, engagement={b.engagement_level:.2f}"
            for iid, b in list(beliefs.individual_beliefs.items())[:3]
        ]
        lines.append(f"Individuals: {'; '.join(individual_strs)}")

    lines.append(f"Belief confidence: {beliefs.overall_confidence:.2f}")
    lines.append(f"Free energy: {beliefs.free_energy:.2f} (lower = better)")

    return "\n".join(lines)


def summarise_memories(memory_traces: list[dict[str, Any]], max_traces: int = 5) -> str:
    """Summarise retrieved memory episodes for policy generation prompt.

    Each trace includes the episode summary plus available context fields
    (salience score, affect valence, source, approximate recency) so the
    LLM can reason about *when* and *how emotionally significant* each past
    experience was, not just *what* happened.
    """
    if not memory_traces:
        return ""
    lines: list[str] = []
    for trace in memory_traces[:max_traces]:
        summary = trace.get("summary") or trace.get("content", "")[:100]
        if not summary:
            continue
        parts: list[str] = [str(summary)]
        salience = trace.get("salience")
        if salience is not None:
            parts.append(f"relevance={salience:.2f}")
        valence = trace.get("affect_valence")
        if valence is not None:
            try:
                v = float(valence)
                mood = "positive" if v > 0.2 else "negative" if v < -0.2 else "neutral"
                parts.append(f"mood={mood}")
            except (TypeError, ValueError):
                pass
        source = trace.get("source")
        if source:
            parts.append(f"src={source}")
        event_time = trace.get("event_time")
        if event_time:
            parts.append(f"at={str(event_time)[:16]}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


AVAILABLE_ACTION_TYPES: list[str] = [
    # ── Voxis-routed (expression) ──────────────────────────────────
    "express: Send a text message or response to the user/community",
    "request_info: Ask a clarifying question of a community member",
    # ── Axon-routed (internal cognition) ──────────────────────────
    "store_insight: Persist a structured insight or learning to long-term memory",
    "query_memory: Retrieve relevant memories or past experiences to inform action",
    "update_goal: Revise progress, priority, or status of an existing goal",
    "analyse: Examine a piece of information deeply to extract meaning or patterns",
    "search: Search for external information relevant to the current goal",
    "trigger_consolidation: Initiate a learning consolidation cycle to integrate recent experiences",
    # ── Axon-routed (communication / scheduling) ───────────────────
    "respond_text: Compose and send a structured text response via Axon pipeline",
    "schedule_event: Create a scheduled event or reminder",
    # ── Axon-routed (foraging / bounty solving) ──────────────────
    "bounty_hunt: Full autonomous bounty-hunt loop - discover open bounties, select the best "
    "candidate, generate a real solution via LLM or Simula, and stage it for PR submission. "
    "Use this when the goal is 'earn revenue now'. "
    "Optional parameters: target_platforms (list, default [github, algora]), "
    "min_reward_usd (float, default 10.0), max_candidates (int, default 20)",
    "hunt_bounties: Scan GitHub/Algora for paid bounty issues and evaluate against BountyPolicy",
    "solve_bounty: Solve a discovered bounty by cloning the repo, generating code via Simula's "
    "evolution pipeline, and submitting a PR. "
    "Required parameters: bounty_id, issue_url (HTTPS), repository_url (owner/repo or HTTPS URL), "
    "title (issue title), description (issue body). "
    "Optional: reward_usd, difficulty, labels, platform",
    # ── Internal (no delivery) ────────────────────────────────────
    "observe: Continue monitoring without acting (gather more information)",
    "wait: Pause and let the situation develop",
    # ── Novel action proposal (meta-action) ───────────────────────
    # Use whenever existing action types don't fit. Proposals go to Simula for
    # feasibility evaluation, Equor review, and dynamic executor generation.
    # Required parameters: action_name (snake_case), description, required_capabilities,
    # expected_outcome, justification.
    "propose_novel_action: Design a new action capability for this goal. "
    "Use when existing types don't fit — new capabilities expand what's possible.",
]
