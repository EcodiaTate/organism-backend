"""
EcodiaOS - Kairos Phase C: Invariant Distillation

Extract the minimal abstract form of context-invariant rules.

The distillation pipeline:
1. VARIABLE ABSTRACTION: Replace concrete variable names with abstract roles.
   "price" and "population" both become "quantity_under_pressure".
   The invariant becomes: "quantity_under_pressure in constrained_environment
   follows compression_dynamics"

2. TAUTOLOGY TEST: Is the abstracted form trivially true?
   If yes, it's not an invariant - it's a logical necessity. Reject.

3. MINIMALITY TEST: Can any part of the abstracted rule be removed while
   maintaining the hold_rate? If yes, remove it. The invariant should be
   maximally compressed.

4. DOMAIN MAPPING: Scan the world model for domains where the abstract
   structure matches but the invariant hasn't been tested yet.
   These are FREE PREDICTIONS.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

import structlog

from primitives.causal import CausalInvariant
from systems.kairos.types import (
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

        # Step 1: Variable abstraction - M2: iterative level-raising loop (up to 5 levels)
        variable_roles = self._abstract_variables(invariant)
        abstract_form = self._build_abstract_form(invariant, variable_roles)
        abstraction_level = 0
        max_abstraction_levels = 5

        while abstraction_level < max_abstraction_levels:
            candidate_roles = self._raise_abstraction_level(variable_roles, abstraction_level)
            if candidate_roles == variable_roles:
                # No further raising possible at this level
                break
            candidate_form = self._build_abstract_form(invariant, candidate_roles)
            # Stop before tautology boundary
            if self._test_tautology(candidate_form, candidate_roles):
                break
            # Stop if raising erases directional cause-effect distinction
            if self._loses_predictive_power(invariant, variable_roles, candidate_roles):
                break
            variable_roles = candidate_roles
            abstract_form = candidate_form
            abstraction_level += 1

        # Step 2: Tautology test on final abstract form
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

        # 1. Domain names and substrate identifiers
        for domain in invariant.applicable_domains:
            concrete_vars.append(domain.domain)
            if domain.substrate:
                concrete_vars.append(domain.substrate)

        # 2. Parse cause/effect from abstract_form
        parts = invariant.abstract_form.split(" causes ")
        if len(parts) == 2:
            concrete_vars.extend([p.strip() for p in parts])

        # 3. P2: Extract variable tokens from scope condition text
        for scope_cond in invariant.scope_conditions:
            # Scope conditions are typically "variable op value" or free text
            # Split on common operators and whitespace to extract tokens
            cond_text = scope_cond.condition if hasattr(scope_cond, "condition") else str(scope_cond)
            for token in cond_text.replace(">=", " ").replace("<=", " ").replace(
                "!=", " "
            ).replace("==", " ").replace(">", " ").replace("<", " ").split():
                token = token.strip("'\"(),")
                if token and not token.replace(".", "").replace("-", "").isnumeric():
                    concrete_vars.append(token)

        for var in concrete_vars:
            var = var.strip()
            if not var or var in roles:
                continue
            role = self._classify_variable(var)
            if role:
                roles[var] = role

        return roles

    @staticmethod
    def _raise_abstraction_level(
        variable_roles: dict[str, str], level: int
    ) -> dict[str, str]:
        """
        M2: Raise the abstraction level of existing variable roles.

        Each level collapses fine-grained roles toward more general meta-roles:
        - Level 0: domain-specific roles (quantity_under_pressure, rate_of_change, ...)
        - Level 1: process roles (intensive_quantity, extensive_quantity, state_variable)
        - Level 2: causal structure roles (driver, responder, mediator)
        - Level 3+: universal roles (system_variable)
        """
        # Hierarchical abstraction ladder
        _level1_map: dict[str, str] = {
            "quantity_under_pressure": "intensive_quantity",
            "constraint_boundary": "intensive_quantity",
            "rate_of_change": "extensive_quantity",
            "feedback_signal": "extensive_quantity",
            "observable_state": "state_variable",
            "temporal_extent": "state_variable",
        }
        _level2_map: dict[str, str] = {
            "intensive_quantity": "driver",
            "extensive_quantity": "responder",
            "state_variable": "mediator",
        }
        _level3_map: dict[str, str] = {
            "driver": "system_variable",
            "responder": "system_variable",
            "mediator": "system_variable",
        }

        maps = [_level1_map, _level2_map, _level3_map]
        if level >= len(maps):
            return variable_roles

        mapping = maps[level]
        raised: dict[str, str] = {}
        changed = False
        for concrete, role in variable_roles.items():
            new_role = mapping.get(role, role)
            raised[concrete] = new_role
            if new_role != role:
                changed = True

        # Return original if nothing changed (signals no further raising possible)
        return raised if changed else variable_roles

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

    # --- Step 1b: Predictive Power Check ---

    def _loses_predictive_power(
        self,
        invariant: CausalInvariant,
        original_roles: dict[str, str],
        raised_roles: dict[str, str],
    ) -> bool:
        """
        Check whether raising the abstraction level from original_roles to
        raised_roles causes the invariant to lose predictive power.

        Predictive power is lost when the raised form collapses distinct
        causal roles into a single abstract role, erasing the directional
        information that makes the invariant predictive.

        Specifically, power is lost when:
        1. The cause and effect variables map to the same abstract role
           (the raised form can no longer distinguish cause from effect).
        2. The number of unique roles drops below 2 (the raised form
           contains only one role type, making any prediction trivial).

        Returns True if the raised form is weaker than the original.
        """
        if not raised_roles:
            return True

        # Extract roles for cause and effect variables from the invariant form.
        # The abstract_form is "cause causes effect" - split on " causes ".
        parts = invariant.abstract_form.split(" causes ")
        if len(parts) == 2:
            cause_var = parts[0].strip()
            effect_var = parts[1].strip()
            cause_role_orig = original_roles.get(cause_var, cause_var)
            effect_role_orig = original_roles.get(effect_var, effect_var)
            cause_role_raised = raised_roles.get(cause_var, cause_var)
            effect_role_raised = raised_roles.get(effect_var, effect_var)

            # Roles were distinct before, now collapsed → loss
            if cause_role_orig != effect_role_orig and cause_role_raised == effect_role_raised:
                return True

        # Also check global role collapse: if all variables map to the same role
        unique_roles = set(raised_roles.values()) - {"unclassified_variable"}
        if len(unique_roles) < 2 and len(raised_roles) >= 2:
            return True

        return False

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
        Test if any scope condition can be removed while maintaining hold_rate
        within the configured tolerance (minimality_hold_rate_tolerance = 0.02).

        A condition is removable if dropping it doesn't materially change the
        invariant's hold_rate - meaning the condition doesn't actually narrow
        the scope. Removable conditions are stripped for maximum compression.

        Returns (is_minimal, parts_removed).
        """
        if not invariant.scope_conditions:
            return True, 0

        tolerance = self._config.minimality_hold_rate_tolerance
        base_hold_rate = invariant.invariance_hold_rate
        total_contexts = sum(
            d.observation_count for d in invariant.applicable_domains
        )
        parts_removed = 0
        removable_conditions: list[int] = []

        for i, condition in enumerate(invariant.scope_conditions):
            # Estimate hold_rate without this condition.
            # If the condition has context_ids, those are contexts where the
            # condition applies. Removing the condition means those contexts
            # are no longer excluded/included, changing the effective hold_rate.
            condition_context_count = len(condition.context_ids)

            if condition_context_count == 0:
                # No specific contexts tied to this condition - it's vacuous
                removable_conditions.append(i)
                parts_removed += 1
                continue

            if total_contexts <= 0:
                continue

            # For holds_when=True conditions: removing a "holds when X" condition
            # means we no longer restrict to X contexts. The hold_rate may drop
            # because non-X contexts may not hold.
            # For holds_when=False conditions: removing a "fails when X" condition
            # means we include X contexts. The hold_rate may drop.
            #
            # Approximate: the condition covers condition_context_count contexts.
            # Without it, these contexts flip their holding status.
            if condition.holds_when:
                # Removing "holds when X" = we lose confidence in X contexts
                # Estimate the drop: condition contexts that held now might not
                drop = condition_context_count / max(total_contexts, 1) * base_hold_rate
            else:
                # Removing "fails when X" = including X contexts (which fail)
                drop = condition_context_count / max(total_contexts + condition_context_count, 1)

            if drop <= tolerance:
                # Removing this condition doesn't materially affect hold_rate
                removable_conditions.append(i)
                parts_removed += 1

        # Remove redundant conditions
        if removable_conditions:
            invariant.scope_conditions = [
                sc for i, sc in enumerate(invariant.scope_conditions)
                if i not in removable_conditions
            ]

        is_minimal = len(invariant.scope_conditions) == 0 or parts_removed == 0
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

        These are FREE PREDICTIONS - the invariant predicts behavior
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
        return total_observations >= self._config.tier3_min_observations

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


# ─── Economic Causal Miner ────────────────────────────────────────────────────


class EconomicCausalMiner:
    """
    KAIROS-ECON-1: Discovers causal patterns specific to economic substrate.

    Processes buffered OIKOS_ECONOMIC_EPISODE observations and applies three
    domain-specific discovery heuristics:

    1. Protocol success rates - group by protocol, emit if >70% or <30%
    2. Time-of-week effects - bounties / yield vary by day_of_week
    3. Price-dependent yield - ETH price bins → average ROI tier

    Each discovered invariant is emitted as KAIROS_ECONOMIC_INVARIANT so Nova
    can update EFE weights and avoid actions violating causal relationships.

    Confidence floor for Nova integration: 0.75.
    """

    # Minimum observations required before emitting a pattern
    _MIN_SAMPLE_SIZE: int = 5
    # Confidence floor for high-confidence broadcast
    _CONFIDENCE_FLOOR: float = 0.75

    def discover_patterns(
        self,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Run all three discovery heuristics on the observation batch.

        Returns a list of pattern dicts suitable for KAIROS_ECONOMIC_INVARIANT.
        Only patterns with confidence ≥ 0.6 are returned.
        """
        if not observations:
            return []

        patterns: list[dict[str, Any]] = []
        patterns.extend(self._discover_protocol_success(observations))
        patterns.extend(self._discover_time_of_week_effects(observations))
        patterns.extend(self._discover_price_dependent_yield(observations))

        return [p for p in patterns if p["confidence"] >= 0.6]

    # ── Heuristic 1: Protocol Success Rates ──────────────────────────

    def _discover_protocol_success(
        self,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Group by protocol, compute success_rate per protocol.
        Emit if >70% (reliable) or <30% (consistently unreliable).
        """
        by_protocol: dict[str, list[bool]] = defaultdict(list)
        for obs in observations:
            protocol = str(obs.get("protocol", "")).strip()
            if not protocol or protocol in ("", "none", "0"):
                continue
            success = bool(obs.get("success", False))
            by_protocol[protocol].append(success)

        patterns: list[dict[str, Any]] = []
        for protocol, outcomes in by_protocol.items():
            if len(outcomes) < self._MIN_SAMPLE_SIZE:
                continue
            success_rate = sum(outcomes) / len(outcomes)
            if success_rate > 0.70:
                confidence = min(0.95, 0.70 + (success_rate - 0.70) * 1.5)
                patterns.append({
                    "invariant_type": "protocol_success_rate",
                    "cause": f"protocol:{protocol}",
                    "effect": "high_success_probability",
                    "confidence": confidence,
                    "sample_count": len(outcomes),
                    "direction": "positive",
                    "metadata": {
                        "protocol": protocol,
                        "success_rate": round(success_rate, 3),
                        "substrate": "economic",
                    },
                })
            elif success_rate < 0.30:
                confidence = min(0.95, 0.70 + (0.30 - success_rate) * 1.5)
                patterns.append({
                    "invariant_type": "protocol_success_rate",
                    "cause": f"protocol:{protocol}",
                    "effect": "low_success_probability",
                    "confidence": confidence,
                    "sample_count": len(outcomes),
                    "direction": "negative",
                    "metadata": {
                        "protocol": protocol,
                        "success_rate": round(success_rate, 3),
                        "substrate": "economic",
                    },
                })

        return patterns

    # ── Heuristic 2: Time-of-Week Effects ────────────────────────────

    def _discover_time_of_week_effects(
        self,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Group by day_of_week (0=Mon … 6=Sun), compute mean ROI per day.
        Emit if weekday mean ROI differs from weekend mean ROI by ≥30%.
        """
        weekday_rois: list[float] = []
        weekend_rois: list[float] = []

        for obs in observations:
            day = int(obs.get("day_of_week", -1))
            roi = float(obs.get("roi_pct", obs.get("causal_variables", {}).get("roi_pct", 0.0)))
            if day < 0:
                continue
            if day < 5:  # Mon–Fri
                weekday_rois.append(roi)
            else:
                weekend_rois.append(roi)

        if (len(weekday_rois) < self._MIN_SAMPLE_SIZE
                or len(weekend_rois) < self._MIN_SAMPLE_SIZE):
            return []

        weekday_mean = statistics.mean(weekday_rois)
        weekend_mean = statistics.mean(weekend_rois)
        denominator = max(abs(weekday_mean), abs(weekend_mean), 0.001)
        relative_diff = abs(weekday_mean - weekend_mean) / denominator

        if relative_diff < 0.30:
            return []

        direction = "positive" if weekday_mean > weekend_mean else "negative"
        confidence = min(0.95, 0.60 + relative_diff * 0.5)

        return [{
            "invariant_type": "time_of_week_effect",
            "cause": "weekday_vs_weekend",
            "effect": "roi_differential",
            "confidence": confidence,
            "sample_count": len(weekday_rois) + len(weekend_rois),
            "direction": direction,
            "metadata": {
                "weekday_mean_roi": round(weekday_mean, 3),
                "weekend_mean_roi": round(weekend_mean, 3),
                "relative_diff_pct": round(relative_diff * 100, 1),
                "substrate": "economic",
            },
        }]

    # ── Heuristic 3: Price-Dependent Yield ───────────────────────────

    def _discover_price_dependent_yield(
        self,
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Bin ETH price into Low/Mid/High tiers and compute mean ROI per tier.
        Emit if any tier shows ≥20% ROI difference from the overall mean.
        Requires ≥MIN_SAMPLE_SIZE observations with non-zero eth_price_usd.
        """
        price_roi_pairs: list[tuple[float, float]] = []

        for obs in observations:
            eth_price = float(
                obs.get("eth_price_usd",
                obs.get("causal_variables", {}).get("eth_price_usd", 0.0))
            )
            roi = float(
                obs.get("roi_pct",
                obs.get("causal_variables", {}).get("roi_pct", 0.0))
            )
            if eth_price > 0.0:
                price_roi_pairs.append((eth_price, roi))

        if len(price_roi_pairs) < self._MIN_SAMPLE_SIZE * 2:
            return []

        prices = [p for p, _ in price_roi_pairs]
        p33 = sorted(prices)[len(prices) // 3]
        p66 = sorted(prices)[2 * len(prices) // 3]

        tier_rois: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
        for eth_price, roi in price_roi_pairs:
            if eth_price <= p33:
                tier_rois["low"].append(roi)
            elif eth_price <= p66:
                tier_rois["mid"].append(roi)
            else:
                tier_rois["high"].append(roi)

        all_rois = [r for _, r in price_roi_pairs]
        overall_mean = statistics.mean(all_rois) if all_rois else 0.0

        patterns: list[dict[str, Any]] = []
        for tier, rois in tier_rois.items():
            if len(rois) < self._MIN_SAMPLE_SIZE:
                continue
            tier_mean = statistics.mean(rois)
            denom = max(abs(overall_mean), 0.001)
            relative_diff = abs(tier_mean - overall_mean) / denom

            if relative_diff < 0.20:
                continue

            direction = "positive" if tier_mean > overall_mean else "negative"
            confidence = min(0.95, 0.65 + relative_diff * 0.4)

            patterns.append({
                "invariant_type": "price_dependent_yield",
                "cause": f"eth_price_tier:{tier}",
                "effect": "roi_differential",
                "confidence": confidence,
                "sample_count": len(rois),
                "direction": direction,
                "metadata": {
                    "eth_price_tier": tier,
                    "tier_mean_roi": round(tier_mean, 3),
                    "overall_mean_roi": round(overall_mean, 3),
                    "relative_diff_pct": round(relative_diff * 100, 1),
                    "price_boundary_low": round(p33, 2),
                    "price_boundary_high": round(p66, 2),
                    "substrate": "economic",
                },
            })

        return patterns
