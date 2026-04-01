"""
EcodiaOS - Niche Forking Engine: Cognitive Organogenesis

When a cognitive niche reaches sufficient maturity and coherence, it can
propose FORKING THE COGNITIVE ARCHITECTURE ITSELF - requesting that
Simula create a new processing pathway, detector, evidence function, or
consolidation strategy specialized for that niche.

This is the organism growing new cognitive organs.

Five fork types:

1. **Detector Fork** - The niche proposes a new PatternDetector that's
   specialized for its domain. Instead of using the general-purpose
   CooccurrenceDetector, the niche wants a detector tuned to its
   specific pattern vocabulary. Dispatched to Simula for synthesis.

2. **Evidence Function Fork** - The niche has learned that the standard
   Bayesian evidence scoring doesn't work for its domain. It proposes
   a custom evidence function: different priors, different decay rates,
   different complexity penalties. Applied as a niche-local override.

3. **Consolidation Strategy Fork** - The niche has its own optimal
   consolidation rhythm. Instead of the global 6-hour cycle, it might
   need 1-hour rapid iteration or 24-hour deep reflection. The niche
   gets its own consolidation schedule with its own phase ordering.

4. **Schema Topology Fork** - The niche has discovered that its domain
   needs a fundamentally different graph structure. Instead of the
   standard entity→relation→entity pattern, it proposes hierarchical,
   temporal, or multi-relational schemas. Dispatched to Simula for
   graph migration.

5. **Worldview Fork** - The nuclear option. The niche has diverged so
   far that it proposes becoming a separate cognitive subsystem entirely.
   This triggers Mitosis-level genome extraction: the niche's hypotheses,
   schemas, procedures, and forked processing components are packaged as
   a CognitiveGenome that can be inherited by child instances or
   integrated as a permanent cognitive module.

Safety:
  - All forks go through Equor constitutional review
  - Worldview forks require HITL approval
  - Detector and evidence function forks are sandboxed in Simula
  - Consolidation strategy forks are velocity-limited
  - The organism can NEVER fork Equor itself

Integration:
  - Runs during consolidation Phase 2.95 (after speciation, before schema induction)
  - Consumes mature niches from NicheRegistry
  - Dispatches to Simula via NICHE_FORK_PROPOSAL Synapse event
  - Results feed back into NicheRegistry and ConsolidationOrchestrator
"""

from __future__ import annotations

import statistics
from datetime import datetime  # noqa: TC003 - Pydantic needs at runtime
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, utc_now

if TYPE_CHECKING:
    from systems.evo.cognitive_niche import CognitiveNiche, NicheRegistry
    from systems.evo.types import Hypothesis
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger()


# ─── Constants ──────────────────────────────────────────────────────────────────

_MIN_MATURITY_FOR_FORK: float = 0.75    # Niche maturity required
_MIN_POPULATION_FOR_FORK: int = 5       # Minimum niche population
_MAX_FORKS_PER_CYCLE: int = 2           # Max fork proposals per consolidation
_WORLDVIEW_FORK_MATURITY: float = 0.9   # Higher bar for worldview forks
_WORLDVIEW_FORK_MIN_POP: int = 10       # Larger population for worldview forks
_FORK_COOLDOWN_HOURS: float = 24.0      # Min time between fork proposals for same niche

# Forbidden fork targets
_FORBIDDEN_FORK_TARGETS = frozenset({"equor", "constitutional", "invariant", "safety", "drive"})


# ─── Fork Types ─────────────────────────────────────────────────────────────────


class ForkType(EOSBaseModel):
    """Base for all fork types."""

    # "detector" | "evidence_function" | "consolidation" | "schema_topology" | "worldview"
    fork_kind: str
    niche_id: str
    niche_name: str = ""
    rationale: str = ""
    expected_improvement: str = ""
    success_probability: float = 0.5
    requires_hitl: bool = False
    requires_simula: bool = True


class DetectorFork(ForkType):
    """Proposal to synthesize a niche-specialized pattern detector."""

    fork_kind: str = "detector"
    target_patterns: list[str] = Field(default_factory=list)
    detector_specification: str = ""
    current_detector_effectiveness: float = 0.0
    target_effectiveness: float = 0.5


class EvidenceFunctionFork(ForkType):
    """Proposal for a niche-local evidence scoring override."""

    fork_kind: str = "evidence_function"
    requires_simula: bool = False  # Applied locally
    custom_priors: dict[str, float] = Field(default_factory=dict)
    custom_decay_rate: float | None = None
    custom_complexity_penalty: float | None = None
    calibration_data: list[tuple[float, float]] = Field(default_factory=list)  # (predicted, actual)


class ConsolidationStrategyFork(ForkType):
    """Proposal for a niche-local consolidation schedule."""

    fork_kind: str = "consolidation"
    requires_simula: bool = False
    proposed_interval_hours: float = 6.0
    proposed_phase_order: list[str] = Field(default_factory=list)
    skip_phases: list[str] = Field(default_factory=list)
    reason: str = ""


class SchemaTopologyFork(ForkType):
    """Proposal for a domain-specific graph structure."""

    fork_kind: str = "schema_topology"
    current_schema_ids: list[str] = Field(default_factory=list)
    proposed_topology: str = ""  # "hierarchical" | "temporal" | "multi_relational" | "hypergraph"
    topology_specification: str = ""
    mdl_gain_estimate: float = 0.0


class WorldviewFork(ForkType):
    """
    The nuclear option - niche becomes a separate cognitive subsystem.

    Packages the niche's entire knowledge state into a CognitiveGenome
    that can be inherited by child instances or integrated as a
    permanent module.
    """

    fork_kind: str = "worldview"
    requires_hitl: bool = True  # Always requires human approval
    hypothesis_ids: list[str] = Field(default_factory=list)
    schema_ids: list[str] = Field(default_factory=list)
    procedure_ids: list[str] = Field(default_factory=list)
    forked_detector_specs: list[str] = Field(default_factory=list)
    forked_evidence_overrides: dict[str, float] = Field(default_factory=dict)
    genome_description: str = ""


# ─── Fork Result ────────────────────────────────────────────────────────────────


class ForkProposal(Identified, Timestamped):
    """A concrete fork proposal ready for dispatch."""

    fork: ForkType
    status: str = "proposed"  # "proposed" | "dispatched" | "approved" | "rejected" | "applied"
    dispatched_at: datetime | None = None
    resolved_at: datetime | None = None
    resolution_reason: str = ""


class NicheForkingResult(EOSBaseModel):
    """Summary of one niche forking phase during consolidation."""

    proposals_generated: int = 0
    detector_forks: int = 0
    evidence_forks: int = 0
    consolidation_forks: int = 0
    schema_forks: int = 0
    worldview_forks: int = 0
    proposals: list[ForkProposal] = Field(default_factory=list)
    duration_ms: int = 0


# ─── Engine ─────────────────────────────────────────────────────────────────────


class NicheForkingEngine:
    """
    Cognitive organogenesis - the organism growing new cognitive organs.

    Examines mature niches and proposes architectural forks that
    specialize processing for each niche's domain. The forks are
    dispatched to Simula for synthesis (detectors, schemas) or
    applied locally (evidence functions, consolidation schedules).
    """

    def __init__(
        self,
        niche_registry: NicheRegistry,
        event_bus: EventBus | None = None,
    ) -> None:
        self._registry = niche_registry
        self._event_bus = event_bus
        self._logger = logger.bind(system="evo.niche_forking")
        self._proposals: list[ForkProposal] = []
        self._last_fork_time: dict[str, datetime] = {}  # niche_id → last fork time
        self._total_forks_proposed: int = 0

    # ─── Main Entry Point ───────────────────────────────────────────────────

    async def run_forking_cycle(
        self,
        hypotheses: list[Hypothesis],
        hypothesis_outcomes: dict[str, list[tuple[float, float]]] | None = None,
    ) -> NicheForkingResult:
        """
        Run a niche forking cycle during consolidation.

        Examines all fork-eligible niches and proposes appropriate
        architectural forks based on each niche's characteristics.
        """
        import time
        t0 = time.monotonic()

        eligible = self._registry.get_fork_eligible_niches()
        proposals: list[ForkProposal] = []
        forks_this_cycle = 0

        for niche in eligible:
            if forks_this_cycle >= _MAX_FORKS_PER_CYCLE:
                break

            # Cooldown check
            last = self._last_fork_time.get(niche.id)
            if last and (utc_now() - last).total_seconds() < _FORK_COOLDOWN_HOURS * 3600:
                continue

            # Constitutional guard
            if any(f in niche.primary_domain.lower() for f in _FORBIDDEN_FORK_TARGETS):
                self._logger.warning(
                    "fork_blocked_constitutional",
                    niche=niche.name, domain=niche.primary_domain,
                )
                continue

            niche_hyps = [h for h in hypotheses if h.id in niche.hypothesis_ids]

            # Determine which fork type is most appropriate
            fork = self._determine_fork_type(niche, niche_hyps, hypothesis_outcomes)
            if fork is None:
                continue

            proposal = ForkProposal(fork=fork, status="proposed")
            proposals.append(proposal)
            self._proposals.append(proposal)
            self._last_fork_time[niche.id] = utc_now()
            niche.fork_proposals_submitted += 1
            forks_this_cycle += 1

            self._logger.info(
                "niche_fork_proposed",
                niche_id=niche.id,
                niche_name=niche.name,
                fork_kind=fork.fork_kind,
                rationale=fork.rationale[:200],
            )

        # Dispatch proposals
        for proposal in proposals:
            await self._dispatch_proposal(proposal)

        self._total_forks_proposed += len(proposals)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        return NicheForkingResult(
            proposals_generated=len(proposals),
            detector_forks=sum(1 for p in proposals if p.fork.fork_kind == "detector"),
            evidence_forks=sum(1 for p in proposals if p.fork.fork_kind == "evidence_function"),
            consolidation_forks=sum(1 for p in proposals if p.fork.fork_kind == "consolidation"),
            schema_forks=sum(1 for p in proposals if p.fork.fork_kind == "schema_topology"),
            worldview_forks=sum(1 for p in proposals if p.fork.fork_kind == "worldview"),
            proposals=proposals,
            duration_ms=elapsed_ms,
        )

    # ─── Fork Type Selection ────────────────────────────────────────────────

    def _determine_fork_type(
        self,
        niche: CognitiveNiche,
        niche_hyps: list[Hypothesis],
        hypothesis_outcomes: dict[str, list[tuple[float, float]]] | None = None,
    ) -> ForkType | None:
        """
        Determine the most appropriate fork type for a mature niche.

        Priority order (most specific first):
          1. Worldview fork - if niche is extremely mature and large
          2. Detector fork - if niche has poor detection coverage
          3. Evidence function fork - if calibration is poor
          4. Consolidation strategy fork - if niche learning velocity is abnormal
          5. Schema topology fork - if niche schemas don't compress well
        """
        # 1. Worldview fork (nuclear option)
        if (
            niche.maturity_score >= _WORLDVIEW_FORK_MATURITY
            and niche.population_size >= _WORLDVIEW_FORK_MIN_POP
            and niche.is_fully_isolated
        ):
            return self._propose_worldview_fork(niche, niche_hyps)

        # 2. Detector fork - poor pattern detection in this niche
        if niche.detector_affinities and self._has_poor_detection(niche, niche_hyps):
            return self._propose_detector_fork(niche, niche_hyps)

        # 3. Evidence function fork - poor calibration
        if hypothesis_outcomes:
            niche_outcomes = {
                cat: outcomes
                for cat, outcomes in hypothesis_outcomes.items()
                if cat == niche.category.value
            }
            if niche_outcomes and self._has_poor_calibration(niche_outcomes):
                return self._propose_evidence_fork(niche, niche_outcomes)

        # 4. Consolidation strategy fork - abnormal learning velocity
        if self._has_abnormal_velocity(niche):
            return self._propose_consolidation_fork(niche)

        # 5. Schema topology fork - poor compression
        if niche.schema_ids and len(niche.schema_ids) > 3:
            return self._propose_schema_fork(niche)

        return None

    # ─── Fork Proposals ─────────────────────────────────────────────────────

    def _propose_worldview_fork(
        self,
        niche: CognitiveNiche,
        niche_hyps: list[Hypothesis],
    ) -> WorldviewFork:
        """Propose the niche becomes a separate cognitive subsystem."""
        return WorldviewFork(
            niche_id=niche.id,
            niche_name=niche.name,
            rationale=(
                f"Niche '{niche.name}' has reached extreme maturity "
                f"(score={niche.maturity_score:.3f}), full reproductive isolation "
                f"(isolation={niche.reproductive_isolation:.3f}), and population "
                f"{niche.population_size}. It has developed a coherent worldview "
                f"incompatible with the general population. Proposing cognitive "
                f"subsystem extraction."
            ),
            expected_improvement=(
                f"Dedicated processing pathway for domain '{niche.primary_domain}' "
                f"with specialized detectors, evidence functions, and consolidation. "
                f"Expected 2-3x prediction accuracy improvement in-domain."
            ),
            success_probability=0.6,
            hypothesis_ids=[h.id for h in niche_hyps],
            schema_ids=list(niche.schema_ids),
            procedure_ids=list(niche.procedure_ids),
            genome_description=(
                f"CognitiveGenome for domain '{niche.primary_domain}': "
                f"{len(niche_hyps)} hypotheses, {len(niche.schema_ids)} schemas, "
                f"{len(niche.procedure_ids)} procedures. "
                f"Generation {niche.genealogy.generation}."
            ),
        )

    def _propose_detector_fork(
        self,
        niche: CognitiveNiche,
        niche_hyps: list[Hypothesis],
    ) -> DetectorFork:
        """Propose a niche-specialized pattern detector."""
        # Find which patterns the niche needs
        pattern_vocab: list[str] = []
        for h in niche_hyps:
            if h.source_detector:
                pattern_vocab.append(h.source_detector)

        most_common_detector = (
            max(set(pattern_vocab), key=pattern_vocab.count)
            if pattern_vocab else "unknown"
        )

        return DetectorFork(
            niche_id=niche.id,
            niche_name=niche.name,
            rationale=(
                f"Niche '{niche.name}' relies primarily on detector "
                f"'{most_common_detector}' but needs domain-specific patterns. "
                f"Proposing specialized detector for domain '{niche.primary_domain}'."
            ),
            expected_improvement=(
                f"Hypothesis generation rate increase for domain '{niche.primary_domain}' "
                f"with higher survival rate."
            ),
            target_patterns=[niche.primary_domain, niche.category.value],
            detector_specification=(
                f"PatternDetector specialized for domain '{niche.primary_domain}'. "
                f"Must implement async scan() method. Should detect patterns relevant "
                f"to {niche.category.value} hypotheses with emphasis on "
                f"{niche.primary_domain}-specific regularities. "
                f"Target: >30% hypothesis survival rate."
            ),
            current_detector_effectiveness=0.2,
            target_effectiveness=0.5,
            success_probability=0.5,
        )

    def _propose_evidence_fork(
        self,
        niche: CognitiveNiche,
        niche_outcomes: dict[str, list[tuple[float, float]]],
    ) -> EvidenceFunctionFork:
        """Propose a niche-local evidence scoring override."""
        calibration_data: list[tuple[float, float]] = []
        for outcomes in niche_outcomes.values():
            calibration_data.extend(outcomes)

        # Compute calibration direction
        if calibration_data:
            predicted = [d[0] for d in calibration_data]
            actual = [d[1] for d in calibration_data]
            mean_predicted = statistics.mean(predicted)
            mean_actual = statistics.mean(actual)
            overconfident = mean_predicted > mean_actual
        else:
            overconfident = False

        return EvidenceFunctionFork(
            niche_id=niche.id,
            niche_name=niche.name,
            rationale=(
                f"Niche '{niche.name}' evidence scoring is poorly calibrated "
                f"({'overconfident' if overconfident else 'underconfident'}). "
                f"Proposing niche-local evidence function with adapted parameters."
            ),
            expected_improvement="Better calibrated confidence estimates in-niche.",
            custom_complexity_penalty=(
                niche.thresholds.complexity_penalty * 1.5 if overconfident
                else niche.thresholds.complexity_penalty * 0.7
            ),
            calibration_data=calibration_data[:50],
            success_probability=0.6,
        )

    def _propose_consolidation_fork(
        self,
        niche: CognitiveNiche,
    ) -> ConsolidationStrategyFork:
        """Propose a niche-local consolidation schedule."""
        # Fast-learning niche → shorter intervals
        # Slow, deep niche → longer intervals
        eff = niche.metabolism.metabolic_efficiency
        if eff > 0.5:
            proposed = 2.0  # Fast niche → 2h consolidation
            reason = "High metabolic efficiency → rapid consolidation cycle"
        else:
            proposed = 12.0  # Slow niche → deep reflection
            reason = "Low metabolic efficiency → deep consolidation for schema discovery"

        return ConsolidationStrategyFork(
            niche_id=niche.id,
            niche_name=niche.name,
            rationale=(
                f"Niche '{niche.name}' has abnormal learning velocity "
                f"(efficiency={eff:.3f}). Standard 6h consolidation cycle is "
                f"suboptimal. Proposing {proposed}h niche-local cycle."
            ),
            expected_improvement=(
                f"Optimized learning velocity for domain "
                f"'{niche.primary_domain}'."
            ),
            proposed_interval_hours=proposed,
            reason=reason,
            success_probability=0.7,
        )

    def _propose_schema_fork(
        self,
        niche: CognitiveNiche,
    ) -> SchemaTopologyFork:
        """Propose a domain-specific graph structure."""
        return SchemaTopologyFork(
            niche_id=niche.id,
            niche_name=niche.name,
            rationale=(
                f"Niche '{niche.name}' has {len(niche.schema_ids)} schemas but "
                f"they may not compress optimally with the standard graph topology. "
                f"Proposing domain-specific topology for '{niche.primary_domain}'."
            ),
            expected_improvement="Better MDL compression and faster traversal for niche queries.",
            current_schema_ids=list(niche.schema_ids),
            proposed_topology="hierarchical",  # Default proposal
            topology_specification=(
                f"Hierarchical schema structure for domain '{niche.primary_domain}'. "
                f"Parent→child entity relationships with inherited properties. "
                f"Optimized for the {niche.category.value} hypothesis category."
            ),
            mdl_gain_estimate=10.0,
            success_probability=0.4,
        )

    # ─── Condition Checks ───────────────────────────────────────────────────

    def _has_poor_detection(self, niche: CognitiveNiche, niche_hyps: list[Hypothesis]) -> bool:
        """Check if the niche has poor pattern detection coverage."""
        if not niche_hyps:
            return True
        recent_hyps = [h for h in niche_hyps if h.status in ("proposed", "testing")]
        return len(recent_hyps) < 2  # Very few new hypotheses → poor detection

    def _has_poor_calibration(self, outcomes: dict[str, list[tuple[float, float]]]) -> bool:
        """Check if evidence scoring is poorly calibrated."""
        all_outcomes: list[tuple[float, float]] = []
        for v in outcomes.values():
            all_outcomes.extend(v)
        if len(all_outcomes) < 5:
            return False
        calibration_error = statistics.mean(abs(p - a) for p, a in all_outcomes)
        return calibration_error > 0.2

    def _has_abnormal_velocity(self, niche: CognitiveNiche) -> bool:
        """Check if niche learning velocity is abnormally fast or slow."""
        v = niche.thresholds.integration_velocity
        return v > 1.5 or v < 0.6

    # ─── Dispatch ───────────────────────────────────────────────────────────

    async def _dispatch_proposal(self, proposal: ForkProposal) -> None:
        """Dispatch a fork proposal via Synapse event bus."""
        proposal.status = "dispatched"
        proposal.dispatched_at = utc_now()

        if self._event_bus is not None:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Use NICHE_FORK_PROPOSAL (not EVOLUTION_CANDIDATE) - fork proposals
            # request cognitive organogenesis, not structural code changes.
            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.NICHE_FORK_PROPOSAL,
                    source_system="evo.niche_forking",
                    data={
                        "proposal_id": proposal.id,
                        "fork_kind": proposal.fork.fork_kind,
                        "niche_id": proposal.fork.niche_id,
                        "niche_name": proposal.fork.niche_name,
                        "rationale": proposal.fork.rationale,
                        "requires_hitl": proposal.fork.requires_hitl,
                        "requires_simula": proposal.fork.requires_simula,
                        "success_probability": proposal.fork.success_probability,
                    },
                )
            )

        self._logger.info(
            "fork_proposal_dispatched",
            proposal_id=proposal.id,
            fork_kind=proposal.fork.fork_kind,
            niche=proposal.fork.niche_name,
            requires_hitl=proposal.fork.requires_hitl,
        )

    # ─── Outcome Tracking ───────────────────────────────────────────────────

    def record_fork_outcome(
        self,
        proposal_id: str,
        approved: bool,
        reason: str = "",
    ) -> None:
        """Record whether a fork proposal was approved or rejected."""
        for p in self._proposals:
            if p.id == proposal_id:
                p.status = "approved" if approved else "rejected"
                p.resolved_at = utc_now()
                p.resolution_reason = reason
                self._logger.info(
                    "fork_outcome_recorded",
                    proposal_id=proposal_id,
                    fork_kind=p.fork.fork_kind,
                    approved=approved,
                    reason=reason,
                )
                return

    # ─── State ──────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_forks_proposed": self._total_forks_proposed,
            "pending": sum(1 for p in self._proposals if p.status in ("proposed", "dispatched")),
            "approved": sum(1 for p in self._proposals if p.status == "approved"),
            "rejected": sum(1 for p in self._proposals if p.status == "rejected"),
            "by_kind": {
                kind: sum(1 for p in self._proposals if p.fork.fork_kind == kind)
                for kind in (
                    "detector", "evidence_function", "consolidation",
                    "schema_topology", "worldview",
                )
            },
        }
