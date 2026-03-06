"""
EcodiaOS — Kairos Phase C: Invariant Distillation

Extract the minimal abstract form of context-invariant rules.

The distillation pipeline:
1. VARIABLE ABSTRACTION: Replace concrete variable names with abstract roles.
   "price" and "population" both become "quantity_under_pressure".
   The invariant becomes: "quantity_under_pressure in constrained_environment
   follows compression_dynamics"

2. TAUTOLOGY TEST: Is the abstracted form trivially true?
   If yes, it's not an invariant — it's a logical necessity. Reject.

3. MINIMALITY TEST: Can any part of the abstracted rule be removed while
   maintaining the hold_rate? If yes, remove it. The invariant should be
   maximally compressed.

4. DOMAIN MAPPING: Scan the world model for domains where the abstract
   structure matches but the invariant hasn't been tested yet.
   These are FREE PREDICTIONS.
"""

from __future__ import annotations

import structlog

from systems.kairos.types import (
    CausalInvariant,
    DistillationResult,
    KairosConfig,
)

logger = structlog.get_logger("kairos.invariant_distiller")


# ─── Abstraction Role Taxonomy ──────────────────────────────────────
# Maps concrete variable name patterns to abstract roles.
# The invariant survives only if the abstracted form is non-trivial.

_ROLE_PATTERNS: list[tuple[list[str], str]] = [
    # Quantities under pressure
    (
        ["price", "cost", "population", "demand", "load", "pressure", "count",
         "density", "concentration", "volume", "mass", "weight", "frequency"],
        "quantity_under_pressure",
    ),
    # Rates and flows
    (
        ["rate", "speed", "velocity", "throughput", "bandwidth", "flux",
         "flow", "growth", "decay", "change", "delta", "gradient"],
        "rate_of_change",
    ),
    # Constraints and limits
    (
        ["capacity", "limit", "threshold", "ceiling", "floor", "bound",
         "maximum", "minimum", "constraint", "budget", "supply", "resource"],
        "constraint_boundary",
    ),
    # States and conditions
    (
        ["status", "phase", "mode", "condition", "level",
         "stage", "health", "quality", "temperature", "energy"],
        "observable_state",
    ),
    # Time-related
    (
        ["time", "duration", "interval", "period", "latency", "age",
         "delay", "wait", "timeout", "elapsed"],
        "temporal_extent",
    ),
    # Feedback signals
    (
        ["error", "residual", "deviation", "variance", "noise",
         "signal", "feedback", "response", "output", "reward"],
        "feedback_signal",
    ),
]

# Tautological abstract forms that are logically necessary, not empirical
_TAUTOLOGICAL_PATTERNS: list[str] = [
    "quantity_under_pressure causes rate_of_change",
    "constraint_boundary causes constraint_boundary",
    "temporal_extent causes temporal_extent",
    "observable_state causes observable_state",
    "feedback_signal causes feedback_signal",
]


class InvariantDistiller:
    """
    Phase C: Distills causal invariants to their minimal abstract form.

    A distilled invariant replaces concrete variables with abstract roles,
    passes tautology and minimality tests, and maps to untested domains
    for free predictions.
    """

    def __init__(self, config: KairosConfig | None = None) -> None:
        self._config = config or KairosConfig()
        self._distillations_run: int = 0
        self._tautologies_rejected: int = 0
        self._domains_mapped: int = 0

    async def distill(
        self,
        invariant: CausalInvariant,
        known_domains: list[str] | None = None,
    ) -> DistillationResult:
        """
        Distill an invariant to its minimal abstract form.

        Args:
            invariant: The invariant to distill.
            known_domains: All domains in the world model, for domain mapping.

        Returns:
            DistillationResult with abstract form, tests, and untested domains.
        """
        self._distillations_run += 1
        known_domains = known_domains or []

        # Step 1: Variable abstraction
        variable_roles = self._abstract_variables(invariant)
        abstract_form = self._build_abstract_form(invariant, variable_roles)

        # Step 2: Tautology test
        is_tautological = self._test_tautology(abstract_form, variable_roles)
        if is_tautological:
            self._tautologies_rejected += 1
            logger.info(
                "tautology_rejected",
                invariant_id=invariant.id,
                abstract_form=abstract_form,
            )
            return DistillationResult(
                invariant_id=invariant.id,
                original_form=invariant.abstract_form,
                abstract_form=abstract_form,
                variable_roles=variable_roles,
                is_tautological=True,
                is_minimal=False,
            )

        # Step 3: Minimality test
        is_minimal, parts_removed = self._test_minimality(
            invariant, variable_roles, abstract_form
        )

        # Step 4: Domain mapping
        untested_domains = self._map_domains(
            invariant, variable_roles, known_domains
        )
        self._domains_mapped += len(untested_domains)

        # Update the invariant in-place
        invariant.distilled = True
        invariant.variable_roles = variable_roles
        invariant.is_tautological = False
        invariant.is_minimal = is_minimal
        invariant.untested_domains = untested_domains
        if abstract_form and abstract_form != invariant.abstract_form:
            invariant.abstract_form = abstract_form

        result = DistillationResult(
            invariant_id=invariant.id,
            original_form=invariant.abstract_form,
            abstract_form=abstract_form,
            variable_roles=variable_roles,
            is_tautological=False,
            is_minimal=is_minimal,
            parts_removed=parts_removed,
            untested_domains=untested_domains,
        )

        logger.info(
            "invariant_distilled",
            invariant_id=invariant.id,
            abstract_form=abstract_form[:80],
            is_minimal=is_minimal,
            untested_domains=len(untested_domains),
            parts_removed=parts_removed,
        )

        return result

    # --- Step 1: Variable Abstraction ---

    def _abstract_variables(self, invariant: CausalInvariant) -> dict[str, str]:
        """Map concrete variable names to abstract roles."""
        roles: dict[str, str] = {}

        # Collect all concrete variable names from the invariant
        concrete_vars: list[str] = []
        for domain in invariant.applicable_domains:
            concrete_vars.append(domain.domain)
        # Parse cause/effect from abstract_form if possible
        parts = invariant.abstract_form.split(" causes ")
        if len(parts) == 2:
            concrete_vars.extend(parts)

        # Also collect from concrete_instances context
        for var in concrete_vars:
            if var in roles:
                continue
            role = self._classify_variable(var)
            if role:
                roles[var] = role

        return roles

    @staticmethod
    def _classify_variable(variable_name: str) -> str:
        """Classify a variable name into an abstract role."""
        name_lower = variable_name.lower()

        for keywords, role in _ROLE_PATTERNS:
            for kw in keywords:
                if kw in name_lower:
                    return role

        # Default: use the variable as-is (unabstractable)
        return "unclassified_variable"

    def _build_abstract_form(
        self,
        invariant: CausalInvariant,
        variable_roles: dict[str, str],
    ) -> str:
        """Build the abstracted form from variable roles."""
        form = invariant.abstract_form
        if not form:
            return ""

        # Replace concrete names with roles
        abstract = form
        for concrete, role in sorted(
            variable_roles.items(), key=lambda x: -len(x[0])
        ):
            abstract = abstract.replace(concrete, role)

        return abstract

    # --- Step 2: Tautology Test ---

    def _test_tautology(
        self,
        abstract_form: str,
        variable_roles: dict[str, str],
    ) -> bool:
        """
        Test if the abstracted form is a tautology (logically necessary).

        A tautology is trivially true regardless of domain, so it
        carries zero information. Examples:
        - "X causes X" (identity)
        - "quantity_under_pressure causes rate_of_change" (definitional)
        """
        if not abstract_form:
            return True

        # Check known tautological patterns
        normalized = abstract_form.strip().lower()
        for pattern in _TAUTOLOGICAL_PATTERNS:
            if normalized == pattern:
                return True

        # Self-causation is always tautological
        parts = normalized.split(" causes ")
        if len(parts) == 2 and parts[0].strip() == parts[1].strip():
            return True

        # Too few distinct roles = probably tautological
        unique_roles = set(variable_roles.values()) - {"unclassified_variable"}
        if len(unique_roles) < self._config.tautology_min_variables:
            all_same = len(unique_roles) <= 1
            if all_same:
                return True

        return False

    # --- Step 3: Minimality Test ---

    def _test_minimality(
        self,
        invariant: CausalInvariant,
        variable_roles: dict[str, str],
        abstract_form: str,
    ) -> tuple[bool, int]:
        """
        Test if any part of the invariant can be removed while maintaining hold_rate.

        Returns (is_minimal, parts_removed).
        """
        if not invariant.scope_conditions:
            return True, 0

        # Try removing each scope condition and check if hold_rate is maintained
        parts_removed = 0
        removable_conditions: list[int] = []

        for i, condition in enumerate(invariant.scope_conditions):
            # A condition is removable if the invariant holds without it
            # (i.e. the condition doesn't narrow the scope meaningfully)
            if condition.holds_when and len(condition.context_ids) == 0:
                removable_conditions.append(i)
                parts_removed += 1

        # Remove redundant conditions
        if removable_conditions:
            invariant.scope_conditions = [
                sc for i, sc in enumerate(invariant.scope_conditions)
                if i not in removable_conditions
            ]

        is_minimal = parts_removed == 0 or len(invariant.scope_conditions) > 0
        return is_minimal, parts_removed

    # --- Step 4: Domain Mapping ---

    def _map_domains(
        self,
        invariant: CausalInvariant,
        variable_roles: dict[str, str],
        known_domains: list[str],
    ) -> list[str]:
        """
        Scan known domains for ones where the abstract structure matches
        but the invariant hasn't been tested.

        These are FREE PREDICTIONS — the invariant predicts behavior
        in domains it was never trained on.
        """
        tested_domains = {d.domain for d in invariant.applicable_domains}
        untested: list[str] = []

        # Get the set of abstract roles this invariant uses
        invariant_roles = set(variable_roles.values()) - {"unclassified_variable"}
        if not invariant_roles:
            return []

        for domain in known_domains:
            if domain in tested_domains:
                continue

            # Check if domain name matches any of the role patterns
            domain_roles = set()
            domain_lower = domain.lower()
            for keywords, role in _ROLE_PATTERNS:
                for kw in keywords:
                    if kw in domain_lower:
                        domain_roles.add(role)
                        break

            # If the domain has at least one matching role, it's a candidate
            overlap = invariant_roles & domain_roles
            if overlap:
                untested.append(domain)

        return untested

    # --- Tier 3 promotion eligibility (Phase C criteria) ---

    def check_tier3_eligibility(self, invariant: CausalInvariant) -> bool:
        """
        Check Phase C Tier 3 promotion requirements:
        - hold_rate > 0.95
        - minimality test passed
        - not tautological
        - 5+ contexts
        """
        if invariant.is_tautological:
            return False
        if not invariant.is_minimal:
            return False
        if invariant.invariance_hold_rate < 0.95:
            return False
        total_observations = sum(
            d.observation_count for d in invariant.applicable_domains
        )
        return total_observations >= self._config.tier3_min_contexts

    # --- Metrics ---

    @property
    def total_distillations_run(self) -> int:
        return self._distillations_run

    @property
    def total_tautologies_rejected(self) -> int:
        return self._tautologies_rejected

    @property
    def total_domains_mapped(self) -> int:
        return self._domains_mapped
