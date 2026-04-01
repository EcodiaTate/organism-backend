"""
EcodiaOS - Simula Immune Advisory Filter

Maintains a blocklist of patterns Thymos has flagged as dangerous.
Before Simula submits a mutation proposal, it checks against this
filter to avoid re-introducing known-bad patterns.

Closure Loop 4: Thymos → Simula (Antibody Learning → Mutation Avoidance)
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger("systems.simula.immune_filter")


class ImmuneAdvisoryFilter:
    """Blocklist of patterns Thymos has flagged as dangerous."""

    def __init__(self) -> None:
        self._advisories: dict[str, dict[str, Any]] = {}  # fingerprint → advisory data

    def ingest_advisory(self, advisory_data: dict[str, Any]) -> None:
        """Add or reinforce an advisory."""
        fp = advisory_data.get("pattern_fingerprint", "")
        if not fp:
            return
        self._advisories[fp] = advisory_data
        logger.info(
            "immune_advisory_ingested",
            fingerprint=fp[:16],
            total_advisories=len(self._advisories),
        )

    def check_proposal(
        self,
        proposal_description: str,
        affected_files: list[str],
    ) -> tuple[bool, str]:
        """
        Returns (is_safe, reason).

        Blocks proposals that touch files associated with known-bad patterns.
        """
        if not self._advisories:
            return True, ""

        affected_set = set(affected_files)
        for fp, adv in self._advisories.items():
            adv_files = set(adv.get("affected_files", []))
            if affected_set & adv_files:
                reason = (
                    f"Blocked by immune advisory {fp[:16]}: "
                    f"{adv.get('description', 'known-bad pattern')[:100]}"
                )
                logger.warning(
                    "proposal_blocked_by_immune_advisory",
                    fingerprint=fp[:16],
                    overlapping_files=list(affected_set & adv_files),
                )
                return False, reason

        return True, ""

    @property
    def advisory_count(self) -> int:
        return len(self._advisories)
