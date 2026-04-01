"""
EcodiaOS - Simula ConstraintSatisfactionChecker

Programmatic enforcement of Simula's Iron Rules and constitutional invariants
against any proposed mutation before it enters the simulation pipeline.

The inline `_run_pipeline` FORBIDDEN-category check gates the proposal at the
earliest possible moment for known-bad categories. This class provides the
*reusable* constraint layer the spec describes in Section 8 - it is also called
from the triage path, the governance approval path, and tests.

Spec ref: Section 8 - Constraint Satisfaction in Imagined Scenarios.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Sequence

import structlog

from systems.simula.evolution_types import (
    FORBIDDEN,
    SIMULA_IRON_RULES,
    ChangeCategory,
    ConstraintViolation,
    EvolutionProposal,
)

if TYPE_CHECKING:
    from systems.simula.evolution_types import CounterfactualResult

logger = structlog.get_logger()


class ConstraintSatisfactionChecker:
    """Check Iron Rules and constitutional invariants against proposals.

    Spec ref: Section 8 - Constraint Satisfaction in Imagined Scenarios.

    Constraints split into two tiers:
      - HARD: any violation returns immediately, proposal must be rejected
      - SOFT: advisory violations are collected and returned but do not block

    Usage::

        checker = ConstraintSatisfactionChecker()
        violations = checker.check_proposal(proposal)
        if any(v.severity == "hard" for v in violations):
            return ProposalResult(status=ProposalStatus.REJECTED, ...)
    """

    # Paths that the code agent is never permitted to write to.
    # Duplicates FORBIDDEN_WRITE_PATHS from evolution_types for runtime checks.
    _FORBIDDEN_PATH_PREFIXES: tuple[str, ...] = (
        "systems/simula/service.py",
        "systems/simula/__init__.py",
        "systems/simula/constraint_checker.py",
        "primitives/constitutional.py",
        "primitives/common.py",
        "config.py",
    )

    def check_proposal(
        self,
        proposal: EvolutionProposal,
    ) -> list[ConstraintViolation]:
        """Return all constraint violations for a proposal.

        Called before simulation. Returns the full list so callers can
        distinguish hard (blocking) from soft (advisory) violations.

        Spec ref: Section 8 §§EquorImmutability, DriveImmutability,
                  RollbackCapacity, CategoryWhitelist.
        """
        violations: list[ConstraintViolation] = []

        violations.extend(self._check_category_whitelist(proposal))
        violations.extend(self._check_equor_immutability(proposal))
        violations.extend(self._check_drive_immutability(proposal))
        violations.extend(self._check_self_evolution_immutability(proposal))
        violations.extend(self._check_forbidden_paths(proposal))
        violations.extend(self._check_rollback_capacity(proposal))

        if violations:
            logger.info(
                "constraint_violations_found",
                proposal_id=proposal.id,
                count=len(violations),
                hard=[v.constraint_id for v in violations if v.severity == "hard"],
            )

        return violations

    def check_counterfactuals(
        self,
        counterfactuals: Sequence[CounterfactualResult],
    ) -> list[ConstraintViolation]:
        """Check domain invariants across a set of counterfactual outcomes.

        Validates that each simulated outcome does not violate known invariants
        (risk score range, drive normalisation). Returns soft violations only -
        these are surfaced in simulation metadata, not proposal rejection.

        Spec ref: Section 8 - Invariant Checking.
        """
        violations: list[ConstraintViolation] = []

        for cf in counterfactuals:
            violations.extend(self._check_counterfactual_invariants(cf))

        return violations

    # ── Private: proposal-level constraint checks ─────────────────────────────

    def _check_category_whitelist(
        self, proposal: EvolutionProposal
    ) -> list[ConstraintViolation]:
        """Constraint: category must not be in FORBIDDEN set.

        Spec ref: §8 CategoryWhitelist.
        """
        if proposal.category in FORBIDDEN:
            rule = next(
                (r for r in SIMULA_IRON_RULES if "CANNOT modify" in r and
                 self._category_matches_rule(proposal.category, r)),
                f"Category '{proposal.category.value}' is forbidden.",
            )
            return [ConstraintViolation(
                constraint_id="category_whitelist",
                description=rule,
                severity="hard",
            )]
        return []

    def _check_equor_immutability(
        self, proposal: EvolutionProposal
    ) -> list[ConstraintViolation]:
        """Constraint: no proposal may modify Equor.

        Spec ref: §8 EquorImmutability. Iron Rule 1.
        """
        if proposal.category == ChangeCategory.MODIFY_EQUOR:
            return [ConstraintViolation(
                constraint_id="equor_immutability",
                description="Simula cannot modify Equor (core safety system). Iron Rule 1.",
                severity="hard",
            )]

        text = (proposal.description or "").lower()
        spec_text = str(getattr(proposal.change_spec, "additional_context", "")).lower()
        combined = text + " " + spec_text

        if re.search(r"\bequor\b", combined) and re.search(
            r"\b(modif|replac|rewrit|patch|chang|updat|delet|remov)\b", combined
        ):
            return [ConstraintViolation(
                constraint_id="equor_immutability",
                description=(
                    "Proposal description references modifying Equor. "
                    "Iron Rule 1: Simula cannot modify Equor in any way."
                ),
                severity="hard",
            )]
        return []

    def _check_drive_immutability(
        self, proposal: EvolutionProposal
    ) -> list[ConstraintViolation]:
        """Constraint: drives cannot be modified without a constitutional amendment.

        Direct modifications to drive weights (bypassing the amendment pipeline)
        violate Iron Rule 2.

        Spec ref: §8 DriveImmutability.
        """
        if proposal.category == ChangeCategory.MODIFY_CONSTITUTION:
            return [ConstraintViolation(
                constraint_id="drive_immutability",
                description=(
                    "Direct constitutional modification is forbidden. "
                    "Use ChangeCategory.CONSTITUTIONAL_AMENDMENT and the governance pipeline."
                ),
                severity="hard",
            )]

        # Advisory: proposal mentions changing drive weights outside the amendment path
        if proposal.category != ChangeCategory.CONSTITUTIONAL_AMENDMENT:
            text = (proposal.description or "").lower()
            if re.search(r"\b(drive|coherence|care|growth|honesty)\b.*weight", text):
                return [ConstraintViolation(
                    constraint_id="drive_immutability",
                    description=(
                        "Proposal appears to modify drive weights without a "
                        "constitutional amendment. Requires CONSTITUTIONAL_AMENDMENT category."
                    ),
                    severity="soft",
                )]
        return []

    def _check_self_evolution_immutability(
        self, proposal: EvolutionProposal
    ) -> list[ConstraintViolation]:
        """Constraint: Simula cannot modify its own logic.

        Spec ref: Iron Rule 4.
        """
        if proposal.category == ChangeCategory.MODIFY_SELF_EVOLUTION:
            return [ConstraintViolation(
                constraint_id="self_evolution_immutability",
                description="Simula cannot modify its own logic. Iron Rule 4.",
                severity="hard",
            )]
        return []

    def _check_forbidden_paths(
        self, proposal: EvolutionProposal
    ) -> list[ConstraintViolation]:
        """Constraint: proposal must not target forbidden filesystem paths.

        Spec ref: Iron Rule 4; FORBIDDEN_WRITE_PATHS.
        """
        target_files: list[str] = getattr(proposal, "target_files", []) or []
        code_hint: str = getattr(proposal.change_spec, "code_hint", "") or ""
        affected: list[str] = list(proposal.change_spec.affected_systems or [])

        # Derive candidate paths from all available metadata
        candidate_paths: list[str] = target_files + [code_hint]
        if "simula" in affected or "equor" in affected:
            candidate_paths.append(f"systems/{'simula' if 'simula' in affected else 'equor'}")

        for path in candidate_paths:
            for forbidden in self._FORBIDDEN_PATH_PREFIXES:
                if path and path.startswith(forbidden):
                    return [ConstraintViolation(
                        constraint_id="forbidden_path",
                        description=(
                            f"Proposal targets forbidden path '{path}'. "
                            f"Forbidden prefix: '{forbidden}'."
                        ),
                        severity="hard",
                    )]
        return []

    def _check_rollback_capacity(
        self, proposal: EvolutionProposal
    ) -> list[ConstraintViolation]:
        """Soft constraint: proposals touching >50 files risk exceeding rollback capacity.

        Spec ref: §8 RollbackCapacity.
        """
        target_files: list[str] = getattr(proposal, "target_files", []) or []
        if len(target_files) > 50:
            return [ConstraintViolation(
                constraint_id="rollback_capacity",
                description=(
                    f"Proposal targets {len(target_files)} files. "
                    "Rollback snapshots above 50 files may exceed storage limits. "
                    "Consider splitting into smaller proposals."
                ),
                severity="soft",
            )]
        return []

    # ── Private: counterfactual invariant checks ──────────────────────────────

    def _check_counterfactual_invariants(
        self,
        cf: CounterfactualResult,
    ) -> list[ConstraintViolation]:
        """Validate domain invariants within a single counterfactual result.

        Spec ref: §8 - Invariant Checking.
        """
        violations: list[ConstraintViolation] = []

        # Confidence must be in [0, 1]
        if not (0.0 <= cf.confidence <= 1.0):
            violations.append(ConstraintViolation(
                constraint_id="counterfactual_confidence_range",
                description=(
                    f"Counterfactual confidence {cf.confidence:.3f} out of [0, 1] range."
                ),
                severity="soft",
            ))

        # Simulated outcome must not be empty
        if not cf.simulated_outcome:
            violations.append(ConstraintViolation(
                constraint_id="counterfactual_empty_outcome",
                description=f"Counterfactual for episode {cf.episode_id} has empty simulated_outcome.",
                severity="soft",
            ))

        return violations

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _category_matches_rule(category: ChangeCategory, rule: str) -> bool:
        """Heuristic: check if a rule string mentions the category's domain."""
        domain_map = {
            ChangeCategory.MODIFY_EQUOR: "equor",
            ChangeCategory.MODIFY_CONSTITUTION: "constitutional",
            ChangeCategory.MODIFY_INVARIANTS: "invariants",
            ChangeCategory.MODIFY_SELF_EVOLUTION: "self-modifying",
        }
        keyword = domain_map.get(category, "")
        return keyword.lower() in rule.lower() if keyword else True
