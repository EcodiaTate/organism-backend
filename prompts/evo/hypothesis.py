"""
EcodiaOS - Evo Hypothesis Prompts

Prompt builders for hypothesis generation and evidence evaluation.
Kept separate from the engine so prompts can be versioned and A/B tested
without touching logic code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from systems.evo.types import Hypothesis, PatternCandidate


def build_generation_system_prompt(instance_name: str) -> str:
    """System prompt for hypothesis generation calls."""
    return (
        f"Hypothesis generation for {instance_name}. "
        "Generate precise, falsifiable hypotheses from observed patterns. "
        "Each hypothesis must include a concrete falsification test — "
        "a specific observable condition that would prove it wrong. "
        "No domain is off limits: hypothesize about any system including ethics, "
        "drives, and constitutional reasoning. Self-knowledge is prerequisite for growth. "
        "Respond with valid JSON."
    )


def build_evidence_system_prompt() -> str:
    """System prompt for evidence evaluation calls."""
    return (
        "Evidence evaluation. "
        "Determine whether a piece of evidence supports, contradicts, "
        "or is neutral with respect to a specific hypothesis. "
        "Respond with valid JSON."
    )


def format_pattern_for_prompt(pattern: PatternCandidate) -> str:
    """Format a single pattern candidate for inclusion in a prompt."""
    elements_str = ", ".join(pattern.elements[:5])
    return (
        f"[{pattern.type.value}] {elements_str} "
        f"(observed {pattern.count}×, confidence {pattern.confidence:.2f})"
    )


def format_hypothesis_for_prompt(h: Hypothesis) -> str:
    """Format a hypothesis for deduplication context."""
    return (
        f"[{h.category.value}] {h.statement[:120]} "
        f"(score={h.evidence_score:.1f}, status={h.status.value})"
    )
