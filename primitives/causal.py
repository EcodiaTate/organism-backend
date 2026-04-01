"""
EcodiaOS - Causal Primitives

Shared causal invariant types used across systems (Kairos, Logos, Nexus, Nova, Axon).
Moved from systems/kairos/types.py so other systems can import without
cross-system violations.
"""

from __future__ import annotations

import enum
from datetime import datetime  # noqa: TC003 - Pydantic needs this at runtime
from typing import Any, Literal

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, utc_now


class CausalInvariantTier(enum.IntEnum):
    """Three tiers of causal invariant generality."""

    TIER_1_DOMAIN = 1        # Holds within a single domain
    TIER_2_CROSS_DOMAIN = 2  # Holds across multiple distinct domains
    TIER_3_SUBSTRATE = 3     # Holds regardless of substrate - deepest layer


class ScopeCondition(EOSBaseModel):
    """A condition under which a causal rule holds or fails."""

    condition: str  # Natural language description
    holds_when: bool = True  # True = rule holds when condition is met
    distinguishing_feature: str = ""
    context_ids: list[str] = Field(default_factory=list)


class ApplicableDomain(EOSBaseModel):
    """A domain where a causal invariant has been tested and holds."""

    domain: str
    substrate: str = ""  # physical, biological, computational, social, economic
    hold_rate: float = 0.0
    observation_count: int = 0


class CausalInvariant(Identified, Timestamped):
    """
    A confirmed causal invariant - the most compressed form of causal knowledge.

    One invariant generates predictions across every domain it touches.
    Tier 3 invariants are architectural events for the world model.
    """

    tier: CausalInvariantTier = CausalInvariantTier.TIER_1_DOMAIN
    abstract_form: str = ""  # The invariant statement in its most abstract form
    concrete_instances: list[str] = Field(default_factory=list)  # CausalRule IDs
    applicable_domains: list[ApplicableDomain] = Field(default_factory=list)
    invariance_hold_rate: float = 0.0
    scope_conditions: list[ScopeCondition] = Field(default_factory=list)
    intelligence_ratio_contribution: float = 0.0
    description_length_bits: float = 0.0
    source_rule_id: str = ""  # ID of the CausalRule that was promoted

    # Causal direction - used by counter-invariant violation predicate
    direction: Literal["positive", "negative", ""] = ""

    # Neo4j persistence - marks invariants validated for RE training export
    validated: bool = False

    # Phase C: Distillation fields
    distilled: bool = False  # True after InvariantDistiller has processed this
    variable_roles: dict[str, str] = Field(default_factory=dict)
    """Maps concrete variable names to abstract roles (e.g. "price" -> "quantity")."""
    is_tautological: bool = False  # Set by tautology test
    is_minimal: bool = False  # Set by minimality test
    untested_domains: list[str] = Field(default_factory=list)
    """Domains where the abstract form matches but the invariant hasn't been tested."""

    # Phase D: Counter-invariant tracking
    violation_count: int = 0
    refined_scope: str = ""  # Natural language boundary condition
    last_violation_check: datetime | None = None

    # Invariant decay - decayed by 0.95× per pipeline cycle when not reinforced
    recency_weight: float = 1.0
    active: bool = True  # False = archived (recency_weight < 0.1)

    @property
    def domain_count(self) -> int:
        return len(self.applicable_domains)

    @property
    def substrate_count(self) -> int:
        return len({d.substrate for d in self.applicable_domains if d.substrate})

    @property
    def cross_domain_transfer_value(self) -> float:
        """Number of untested domains × invariant coverage."""
        return len(self.untested_domains) * self.invariance_hold_rate


# ─── ActionLog ────────────────────────────────────────────────────────────────


class ActionLog(Identified, Timestamped):
    """
    A single logged Axon action, queryable by Kairos for Stage 2 intervention
    asymmetry testing.

    Axon writes one ActionLog per completed intent execution.  Kairos reads
    these to determine: when EOS intervened on variable X, did variable Y
    change?  A consistent asymmetry (do(X) changes Y but do(Y) doesn't change
    X) is strong evidence for X → Y causation.

    Fields
    ------
    action_id : str
        Stable ULID - same as the AuditRecord.execution_id written by Axon.
    action_type : str
        Executor name (e.g. "adjust_config", "wallet_transfer", "defi_yield").
    intent_id : str
        The Intent that triggered this action.
    outcome : str
        "success" | "failure" | "rolled_back" | "partial".
    axon_executor : str
        Primary executor used (first step executor, mirrors Axon audit).
    rollback_occurred : bool
        True if the step triggered a rollback.
    target_variable : str
        The primary variable this action targeted (used by Kairos for do(X) matching).
    outcome_changes : list[str]
        Variables observably changed after the action (used by Kairos for
        intervention asymmetry: did B change after do(A)?).
    before_state : dict[str, Any]
        Snapshot of relevant state before the action.
    after_state : dict[str, Any]
        Snapshot of relevant state after the action.
    """

    action_id: str = ""
    action_type: str = ""
    intent_id: str = ""
    outcome: str = ""
    axon_executor: str = ""
    rollback_occurred: bool = False
    target_variable: str = ""
    outcome_changes: list[str] = Field(default_factory=list)
    before_state: dict[str, Any] = Field(default_factory=dict)
    after_state: dict[str, Any] = Field(default_factory=dict)


# ─── CausalHierarchyLevel ─────────────────────────────────────────────────────


class CausalHierarchyLevel(EOSBaseModel):
    """
    A single tier in the causal invariant knowledge hierarchy.

    Used by Kairos CausalHierarchy to tag where each invariant sits in the
    compression stack.  Promoted to primitives so other systems (Nexus,
    Logos, Nova) can interpret hierarchy metadata without importing Kairos.

    tier : CausalInvariantTier
        Which level this entry represents.
    invariant_id : str
        The invariant placed at this tier.
    scope : str
        Natural language description of the scope/domain covered.
    confidence : float
        Hold rate at promotion time (0–1).
    substrate_independent : bool
        True only for Tier 3 invariants - holds across ALL substrates.
    """

    tier: CausalInvariantTier = CausalInvariantTier.TIER_1_DOMAIN
    invariant_id: str = ""
    scope: str = ""
    confidence: float = 0.0
    substrate_independent: bool = False
