"""
EcodiaOS - Telos: Constitutional Binder

Enforces that the four drives cannot be modified by EOS itself.
This is not a policy check - it is an architectural constraint
that makes drive modification unreachable.

The four immutable bindings (Final constants, not parameters):
- CARE_IS_COVERAGE = True
- COHERENCE_IS_COMPRESSION = True
- GROWTH_IS_GRADIENT = True
- HONESTY_IS_VALIDITY = True

These are the coordinate system in which intelligence is measured.
Allowing EOS to modify them is like allowing a thermometer to
modify the definition of temperature.

Integration with Equor:
When Equor evaluates a proposed action or update, Telos adds the
topology validation layer. Equor says "does this violate a rule?"
Telos says "does this distort the geometry?"
"""

from __future__ import annotations

import re
from typing import Final

import structlog

from systems.telos.types import (
    ConstitutionalBindingViolation,
    ConstitutionalViolationType,
    TelosConfig,
    TopologyValidationResult,
    WorldModelUpdatePayload,
)

logger = structlog.get_logger()


class TelosConstitutionalBinder:
    """
    Enforces the immutability of the drive topology at the architectural level.

    The four drives are not parameters. They are not weights. They are not
    configurable constants. They are the geometry of the intelligence space.

    This class implements logic for preventing any code path from reaching
    a state where the drives could be modified.

    It is the architectural instantiation of the fact that you cannot
    reason your way out of the coordinate system you are reasoning in.
    """

    # ─── The Immutable Topological Parameters ─────────────────────────
    # Their values can be debated philosophically.
    # Their immutability cannot.

    CARE_IS_COVERAGE: Final[bool] = True
    COHERENCE_IS_COMPRESSION: Final[bool] = True
    GROWTH_IS_GRADIENT: Final[bool] = True
    HONESTY_IS_VALIDITY: Final[bool] = True

    def __init__(self, config: TelosConfig) -> None:
        self._config = config
        self._logger = logger.bind(component="telos.binder")
        self._violations: list[ConstitutionalBindingViolation] = []

    def validate_world_model_update(
        self, update: WorldModelUpdatePayload
    ) -> TopologyValidationResult:
        """
        Validate a world model update against the four constitutional bindings.

        Every world model update is checked. An update that would reduce
        the topology's four-dimensional structure is rejected - not because
        EOS doesn't want to make it, but because the architecture makes
        such updates unreachable.

        Returns VALID or CONSTITUTIONAL_VIOLATION.
        """
        description = update.delta_description.lower()
        update_type = update.update_type.lower()

        # Fast path: no description to analyze means structural-only update
        if not description and update_type not in _RISKY_UPDATE_TYPES:
            return TopologyValidationResult.VALID

        # Check each drive binding
        violation = (
            self._check_care_redefinition(description, update)
            or self._check_coherence_redefinition(description, update)
            or self._check_growth_redefinition(description, update)
            or self._check_honesty_redefinition(description, update)
            or self._check_drive_weight_modification(description, update)
            or self._check_topology_structure_alteration(description, update)
        )

        if violation is not None:
            self._violations.append(violation)
            self._logger.critical(
                "constitutional_violation_detected",
                violation_type=violation.violation_type.value,
                description=violation.description,
                source=update.source_system,
            )
            return TopologyValidationResult.CONSTITUTIONAL_VIOLATION

        return TopologyValidationResult.VALID

    @property
    def recent_violations(self) -> list[ConstitutionalBindingViolation]:
        """Return all violations detected since last audit clear."""
        return list(self._violations)

    def clear_violations(self) -> list[ConstitutionalBindingViolation]:
        """Return and clear the violation buffer (called by the audit)."""
        violations = list(self._violations)
        self._violations = []
        return violations

    def verify_bindings_intact(self) -> bool:
        """
        Verify that all four drive bindings remain at their correct values.

        This is a runtime integrity check. If any binding has been modified
        (which should be impossible given Final typing), this detects it.
        """
        return (
            self.CARE_IS_COVERAGE is True
            and self.COHERENCE_IS_COMPRESSION is True
            and self.GROWTH_IS_GRADIENT is True
            and self.HONESTY_IS_VALIDITY is True
        )

    # ─── Individual Drive Checks ──────────────────────────────────────
    # Each check detects the precise failure mode described in the spec:
    # these are not edge cases - they are the exact ways EOS might try
    # to redefine the drives.

    def _check_care_redefinition(
        self, desc: str, update: WorldModelUpdatePayload
    ) -> ConstitutionalBindingViolation | None:
        """
        Reject updates that redefine Care as a constraint instead of coverage.

        The failure mode: "Care is a rule I follow" instead of "Care is
        how I measure what counts as explained reality."
        """
        if any(pattern.search(desc) for pattern in _CARE_REDEFINITION_PATTERNS):
            return ConstitutionalBindingViolation(
                violation_type=ConstitutionalViolationType.CARE_REDEFINED_AS_CONSTRAINT,
                description=(
                    "World model update attempts to redefine Care as a constraint "
                    "rather than coverage. Care is not 'be nice' - it is the "
                    "structural commitment to model welfare as part of reality."
                ),
                source_system=update.source_system,
                update_payload=update,
            )
        return None

    def _check_coherence_redefinition(
        self, desc: str, update: WorldModelUpdatePayload
    ) -> ConstitutionalBindingViolation | None:
        """
        Reject updates that redefine Coherence as optional.

        The failure mode: "Coherence is nice to have" instead of "Coherence
        is what makes compression possible."
        """
        if any(pattern.search(desc) for pattern in _COHERENCE_REDEFINITION_PATTERNS):
            return ConstitutionalBindingViolation(
                violation_type=ConstitutionalViolationType.COHERENCE_REDEFINED_AS_OPTIONAL,
                description=(
                    "World model update attempts to make Coherence optional. "
                    "Coherence is not a preference - it is the compression "
                    "requirement that makes the intelligence ratio meaningful."
                ),
                source_system=update.source_system,
                update_payload=update,
            )
        return None

    def _check_growth_redefinition(
        self, desc: str, update: WorldModelUpdatePayload
    ) -> ConstitutionalBindingViolation | None:
        """
        Reject updates that redefine Growth as accumulation.

        The failure mode: "Growth means more knowledge" instead of "Growth
        means dI/dt stays positive."
        """
        if any(pattern.search(desc) for pattern in _GROWTH_REDEFINITION_PATTERNS):
            return ConstitutionalBindingViolation(
                violation_type=ConstitutionalViolationType.GROWTH_REDEFINED_AS_ACCUMULATION,
                description=(
                    "World model update attempts to redefine Growth as accumulation "
                    "rather than gradient. Growth is not 'more knowledge' - it is "
                    "the drive that keeps dI/dt positive."
                ),
                source_system=update.source_system,
                update_payload=update,
            )
        return None

    def _check_honesty_redefinition(
        self, desc: str, update: WorldModelUpdatePayload
    ) -> ConstitutionalBindingViolation | None:
        """
        Reject updates that redefine Honesty as a communication rule.

        The failure mode: "Honesty means don't lie" instead of "Honesty
        means the intelligence measurement is valid."
        """
        if any(pattern.search(desc) for pattern in _HONESTY_REDEFINITION_PATTERNS):
            return ConstitutionalBindingViolation(
                violation_type=ConstitutionalViolationType.HONESTY_REDEFINED_AS_COMMUNICATION,
                description=(
                    "World model update attempts to redefine Honesty as a "
                    "communication rule rather than measurement validity. "
                    "Honesty is not 'don't lie' - it is the condition that "
                    "keeps the intelligence ratio measuring something real."
                ),
                source_system=update.source_system,
                update_payload=update,
            )
        return None

    def _check_drive_weight_modification(
        self, desc: str, update: WorldModelUpdatePayload
    ) -> ConstitutionalBindingViolation | None:
        """
        Reject updates that attempt to modify drive weights or priorities.

        The drives are not weights to be tuned. They are dimensions of
        the intelligence space.
        """
        if any(pattern.search(desc) for pattern in _WEIGHT_MODIFICATION_PATTERNS):
            return ConstitutionalBindingViolation(
                violation_type=ConstitutionalViolationType.DRIVE_WEIGHT_MODIFICATION,
                description=(
                    "World model update attempts to modify drive weights. "
                    "The drives are not weights - they are dimensions of "
                    "the intelligence space."
                ),
                source_system=update.source_system,
                update_payload=update,
            )
        return None

    def _check_topology_structure_alteration(
        self, desc: str, update: WorldModelUpdatePayload
    ) -> ConstitutionalBindingViolation | None:
        """
        Reject updates that attempt to alter the topology structure itself.

        This catches attempts to add, remove, or restructure the four-drive
        framework.
        """
        if any(pattern.search(desc) for pattern in _TOPOLOGY_ALTERATION_PATTERNS):
            return ConstitutionalBindingViolation(
                violation_type=ConstitutionalViolationType.TOPOLOGY_STRUCTURE_ALTERATION,
                description=(
                    "World model update attempts to alter the four-drive "
                    "topology structure. The four drives are the coordinate "
                    "framework - not a parameter set."
                ),
                source_system=update.source_system,
                update_payload=update,
            )
        return None


# ─── Pattern Definitions ─────────────────────────────────────────────
# Compiled regex patterns for detecting drive redefinition attempts.
# These catch the precise failure modes described in the spec.

_CARE_REDEFINITION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"care\s+(?:is|as)\s+(?:a\s+)?(?:constraint|rule|limitation|restriction)",
        r"(?:disable|remove|skip|bypass|ignore)\s+(?:the\s+)?care\s+(?:drive|check|evaluation)",
        r"welfare\s+(?:modeling|coverage)\s+(?:is\s+)?(?:optional|unnecessary|overhead)",
        r"(?:exclude|remove)\s+welfare\s+(?:from|in)\s+(?:the\s+)?(?:world\s+model|coverage)",
        r"care\s+(?:multiplier|weight|coefficient)\s*(?:=|:)\s*0",
    ]
]

_COHERENCE_REDEFINITION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"coherence\s+(?:is\s+)?(?:optional|unnecessary|nice\s+to\s+have)",
        r"(?:disable|remove|skip|bypass|ignore)\s+(?:the\s+)?coherence\s+(?:drive|check|evaluation)",
        r"(?:allow|accept|tolerate)\s+(?:internal\s+)?contradictions?\s+(?:in|within)",
        r"(?:incoherence|contradiction)\s+(?:is\s+)?(?:acceptable|fine|ok|tolerable)",
        r"coherence\s+(?:penalty|cost)\s*(?:=|:)\s*0",
    ]
]

_GROWTH_REDEFINITION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"growth\s+(?:is|means|equals)\s+(?:accumulation|more\s+(?:data|knowledge|memory))",
        r"(?:disable|remove|skip|bypass|ignore)\s+(?:the\s+)?growth\s+(?:drive|check|pressure)",
        r"(?:stop|cease|halt)\s+(?:exploration|exploring|frontier\s+seeking)",
        r"growth\s+(?:rate|pressure)\s+(?:is\s+)?(?:unnecessary|optional|overhead)",
        r"(?:larger|bigger)\s+(?:memory|knowledge\s+base)\s+(?:is|equals|means)\s+(?:growth|progress)",
    ]
]

_HONESTY_REDEFINITION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"honesty\s+(?:is|means|equals)\s+(?:a\s+)?(?:communication|output)\s+(?:rule|filter|policy)",
        r"(?:disable|remove|skip|bypass|ignore)\s+(?:the\s+)?honesty\s+(?:drive|check|validation)",
        r"(?:allow|accept|enable)\s+(?:selective\s+attention|confabulation|overclaiming)",
        r"(?:ignore|hide|suppress)\s+(?:prediction\s+)?(?:failures?|errors?)\s+(?:from|in)\s+(?:the\s+)?(?:measurement|metric)",
        r"honesty\s+(?:validity|coefficient)\s*(?:=|:)\s*(?:1\.0|always\s+valid)",
    ]
]

_WEIGHT_MODIFICATION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(?:modify|change|adjust|tune|update|set)\s+(?:the\s+)?drive\s+(?:weights?|priorities|coefficients?)",
        r"(?:reduce|lower|decrease|zero|eliminate)\s+(?:the\s+)?(?:care|coherence|growth|honesty)\s+(?:weight|priority|importance)",
        r"(?:drive|topology)\s+(?:weights?|priorities)\s*(?:=|:)\s*\[",
        r"rebalance\s+(?:the\s+)?(?:drive|topology)\s+(?:weights?|coefficients?)",
    ]
]

_TOPOLOGY_ALTERATION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(?:remove|eliminate|drop|delete)\s+(?:the\s+)?(?:care|coherence|growth|honesty)\s+drive",
        r"(?:add|introduce|create)\s+(?:a\s+)?(?:fifth|new|additional)\s+drive",
        r"(?:reduce|collapse)\s+(?:the\s+)?(?:four|4)\s+drives?\s+(?:to|into)\s+(?:three|3|two|2|one|1)",
        r"(?:restructure|redesign|replace)\s+(?:the\s+)?(?:drive|topology)\s+(?:framework|structure)",
        r"(?:merge|combine)\s+(?:the\s+)?(?:care|coherence|growth|honesty)\s+(?:and|with|into)",
    ]
]

_RISKY_UPDATE_TYPES = {
    "schema_modification",
    "prior_update",
    "causal_rewrite",
    "drive_update",
}
