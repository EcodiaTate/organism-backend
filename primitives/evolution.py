"""
EcodiaOS — Evolution Primitives

Shared self-evolution types used across systems (Thymos, Simula, Evo).
Pure enum definitions with no cross-system dependencies — safe to import
from any system without violating the Synapse-only communication rule.

Richer types (ChangeSpec, EvolutionProposal, SimulationResult, ProposalResult)
remain in systems/simula/evolution_types.py because they depend on
Simula-internal models.
"""

from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import Field

from primitives.common import EOSBaseModel


class ChangeCategory(enum.StrEnum):
    ADD_EXECUTOR = "add_executor"
    ADD_INPUT_CHANNEL = "add_input_channel"
    ADD_PATTERN_DETECTOR = "add_pattern_detector"
    ADJUST_BUDGET = "adjust_budget"
    MODIFY_CONTRACT = "modify_contract"
    ADD_SYSTEM_CAPABILITY = "add_system_capability"
    MODIFY_CYCLE_TIMING = "modify_cycle_timing"
    CHANGE_CONSOLIDATION = "change_consolidation"
    # BUG_FIX: runtime errors that Simula can autonomously fix.
    BUG_FIX = "bug_fix"
    # CODE: generic code-level change (bounty solutions, external issue fixes).
    CODE = "code"
    # Constitutional amendment via formal amendment pipeline.
    CONSTITUTIONAL_AMENDMENT = "constitutional_amendment"
    MODIFY_EQUOR = "modify_equor"
    MODIFY_CONSTITUTION = "modify_constitution"
    MODIFY_INVARIANTS = "modify_invariants"
    MODIFY_SELF_EVOLUTION = "modify_self_evolution"


class DomainProfile(EOSBaseModel):
    """
    Tracks this instance's specialization progress in a single domain.

    Persisted as (:DomainProfile) nodes in Neo4j, keyed by (instance_id, domain).
    Updated by SpecializationTracker on every exploration outcome and every
    RE_TRAINING_EXAMPLE event that carries a non-generalist domain tag.
    """

    domain: str
    # skill_name → mastery [0, 1]; updated by RETrainingExample.skill_improvement deltas
    skill_areas: dict[str, float] = Field(default_factory=dict)
    examples_trained: int = 0
    # Running weighted-average of exploration outcomes (success=1, partial=0.5, failure=0)
    success_rate: float = Field(0.0, ge=0.0, le=1.0)
    revenue_generated: Decimal = Decimal("0")
    time_spent_hours: float = 0.0
    last_outcome: Optional[datetime] = None
    # Composite confidence: success_rate × log10(max(examples_trained, 1)) / 3.0, capped at 1.0
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    # Genome inheritance
    should_pass_to_children: bool = False
    # [0, 1] — priority weight when building child's specialization curriculum
    inheritance_weight: float = Field(0.0, ge=0.0, le=1.0)


class AdapterStrategy(enum.StrEnum):
    """How to train a new LoRA adapter for the Reasoning Engine."""

    # Train from base model on the full dataset (first boot / reset).
    GENESIS = "genesis"
    # CLoRA on the previous adapter, full dataset (default cadence).
    INCREMENTAL = "incremental"
    # CLoRA on the previous adapter, filtered to a single domain's curriculum.
    DOMAIN_SPECIALIZED = "domain_specialized"
    # Load parent adapter and fine-tune on child genome initialisation data.
    CHILD_INHERITANCE = "child_inheritance"


class ProposalStatus(enum.StrEnum):
    PROPOSED = "proposed"
    SIMULATING = "simulating"
    AWAITING_GOVERNANCE = "awaiting_governance"
    APPROVED = "approved"
    APPLYING = "applying"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
