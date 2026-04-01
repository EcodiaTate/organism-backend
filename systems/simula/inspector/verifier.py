"""
EcodiaOS - Inspector Adversarial Verifier (Phase 2)

Multi-agent adversarial debate to eliminate false positives.

Agent Red (the Tracer/Prover) claims a vulnerability is real.
Agent Blue (AdversarialVerifier) is a principal AppSec engineer who
looks for reasons the finding is unexploitable in practice.

Only findings with a PoC are debated - the prover already filters those
without a counterexample, so unarmed reports skip straight through.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.types import (
    VulnerabilityReport,
    VulnerabilitySeverity,
)

if TYPE_CHECKING:
    import logging

    from clients.llm import LLMProvider

logger = structlog.get_logger().bind(system="simula.inspector.verifier")

_DEFENDER_SYSTEM_PROMPT = """\
You are a Principal Application Security Engineer (Agent Blue).
A junior researcher (Agent Red) claims to have found a vulnerability. \
They have provided the source code context, their Z3 mathematical proof, \
and a reproduction script.

Your job is to find the flaw in their attack. Look for:
- Missed sanitization functions or input validation already in the code
- Framework-level mitigations (ORM escaping, CSRF tokens, parameterised queries)
- Strict typing or schema validation that blocks the PoC payload
- Logical impossibilities that make the counterexample unexploitable in reality
- Runtime environment constraints (sandboxing, WAF, network policy) implied by context

Output MUST be valid JSON with exactly two keys:
  "is_valid": boolean - true if the vulnerability is real and exploitable
  "justification": string - one concise paragraph explaining your conclusion

Do not include any text outside the JSON object."""

# LLM timeout for a single verification call (seconds).
_VERIFY_TIMEOUT_S = 60


class AdversarialVerifier:
    """
    Agent Blue: refutes false positives via adversarial LLM debate.

    Only reports that carry a proof-of-concept script are sent to the
    defender - uninvestigated surfaces pass through unchanged.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._log = logger

    async def verify_finding(self, report: VulnerabilityReport) -> VulnerabilityReport:
        """
        Challenge a VulnerabilityReport with Agent Blue.

        If the report has no PoC it is returned unchanged - we only debate
        findings that Agent Red has fully substantiated.

        If Agent Blue determines the finding is not exploitable, the report
        is mutated in-place:
          - severity  → VulnerabilitySeverity.FALSE_POSITIVE
          - proof_of_concept_code → "" (PoC retracted)
          - defender_notes → Agent Blue's justification

        Args:
            report: The VulnerabilityReport to verify.

        Returns:
            The (possibly mutated) VulnerabilityReport.
        """
        if not report.has_poc:
            # No PoC → nothing to debate
            return report

        log = self._log.bind(
            vulnerability_id=report.id,
            vulnerability_class=report.vulnerability_class.value,
            severity=report.severity.value,
            entry_point=report.attack_surface.entry_point,
        )

        log.info("defender_debate_started")

        debate_prompt = _build_debate_prompt(report)

        try:
            from clients.llm import Message  # local import avoids cycle

            response = await asyncio.wait_for(
                self._llm.generate(
                    system_prompt=_DEFENDER_SYSTEM_PROMPT,
                    messages=[Message(role="user", content=debate_prompt)],
                    max_tokens=512,
                    temperature=0.2,  # low temp for analytical refutation
                ),
                timeout=_VERIFY_TIMEOUT_S,
            )
        except TimeoutError:
            log.warning(
                "defender_timeout",
                timeout_s=_VERIFY_TIMEOUT_S,
                action="passing_finding_as_valid",
            )
            return report
        except Exception as exc:
            log.warning(
                "defender_llm_error",
                error=str(exc),
                action="passing_finding_as_valid",
            )
            return report

        verdict = _parse_defender_response(response.text, log)
        if verdict is None:
            # Unparseable response - trust Agent Red, don't discard
            log.warning(
                "defender_parse_failed",
                raw=response.text[:300],
                action="passing_finding_as_valid",
            )
            return report

        is_valid: bool = verdict["is_valid"]
        justification: str = verdict["justification"]

        if is_valid:
            log.info(
                "defender_confirmed_valid",
                justification=justification[:200],
            )
        else:
            log.info(
                "vulnerability_refuted_by_defender",
                justification=justification[:200],
            )
            report.severity = VulnerabilitySeverity.FALSE_POSITIVE
            report.proof_of_concept_code = ""
            report.defender_notes = justification

        return report


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_debate_prompt(report: VulnerabilityReport) -> str:
    """Assemble the user-turn message Agent Blue receives."""
    lines = [
        "## Agent Red's Claim",
        f"Vulnerability class : {report.vulnerability_class.value}",
        f"Severity            : {report.severity.value}",
        f"Attack goal         : {report.attack_goal}",
        "",
        "## Source Code Context",
        "```",
        report.attack_surface.context_code or "(no context extracted)",
        "```",
        "",
        "## Z3 Counterexample (mathematical proof)",
        report.z3_counterexample or "(none)",
        "",
        "## Proof-of-Concept Script",
        "```python",
        report.proof_of_concept_code,
        "```",
        "",
        "Is this vulnerability real and exploitable in practice? Respond in JSON.",
    ]
    return "\n".join(lines)


def _parse_defender_response(
    raw: str,
    log: logging.Logger | structlog.BoundLogger,
) -> dict[str, bool | str] | None:
    """
    Extract the JSON verdict from Agent Blue's response.

    Tries two strategies:
      1. Direct json.loads on the stripped response.
      2. Locate the first '{' … '}' block and parse that.

    Returns None on parse failure so callers can fall back safely.
    """
    text = raw.strip()

    # Strategy 1: whole response is JSON
    try:
        data = json.loads(text)
        if _is_valid_verdict(data):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if _is_valid_verdict(data):
                return data
        except json.JSONDecodeError:
            pass

    return None


def _is_valid_verdict(data: object) -> bool:
    """Confirm the parsed object has the expected shape."""
    return (
        isinstance(data, dict)
        and "is_valid" in data
        and "justification" in data
        and isinstance(data["is_valid"], bool)
        and isinstance(data["justification"], str)
    )
