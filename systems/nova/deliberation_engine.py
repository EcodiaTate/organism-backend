"""
EcodiaOS - Nova Deliberation Engine

The dual-process decision engine. Implements the System 1 / System 2 split
from cognitive architecture - not as a performance trick, but as a genuine
model of how deliberation works.

Fast path (System 1, ≤200ms total):
  - Pattern-match against known procedure templates
  - Build intent directly from matched procedure
  - Submit to Equor for critical-path review (≤50ms)
  - If denied, escalate to slow path

Slow path (System 2, ≤15000ms total):
  - Generate 2-5 candidate policies via LLM (≤10000ms)
  - Evaluate EFE for each candidate in parallel (≤200ms per policy)
  - Select minimum-EFE policy
  - Formulate Intent from selected policy
  - Submit to Equor for standard review (≤500ms)
  - If denied, retry with next-best policy

Routing decision:
  The choice between fast and slow path is itself a decision.
  Novelty, risk, emotional intensity, and belief conflict drive it.
  Over time (via Evo), more situations shift from slow to fast as
  reliable patterns are learned.

The null outcome: if no active goal matches and the broadcast doesn't
warrant creating one, deliberation returns None - no action taken.
This is the correct outcome, not a failure.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import Verdict, new_id
from primitives.intent import (
    Action,
    ActionSequence,
    DecisionTrace,
    GoalDescriptor,
    Intent,
)
from systems.nova.cognition_cost import (
    CognitionBudget,
    CognitionCostCalculator,
)
from systems.nova.policy_generator import (  # noqa: F401
    BasePolicyGenerator,
    PolicyGenerator,
    _broadcast_is_hungry,
    _is_bounty_discovery,
    find_matching_procedure,
    procedure_to_policy,
)
from systems.nova.types import (
    BeliefState,
    DecisionRecord,
    EFEScore,
    FreeEnergyBudget,
    Goal,
    Policy,
    PriorityContext,
    SituationAssessment,
)

if TYPE_CHECKING:
    from primitives.affect import AffectState
    from primitives.constitutional import ConstitutionalCheck
    from systems.fovea.types import WorkspaceBroadcast
    from systems.equor.service import EquorService
    from systems.evo.tournament import TournamentEngine
    from systems.evo.types import TournamentContext
    from systems.nova.efe_evaluator import EFEEvaluator
    from systems.nova.goal_manager import GoalManager

logger = structlog.get_logger()

# Deliberation routing thresholds — initial values, Evo tunes these over time.
# Access via the DeliberationEngine instance attributes (self._novelty_threshold etc.)
# so the system can update them without restarting. Never read these module-level
# values directly in hot-path code — always use the instance-level equivalents below.
_NOVELTY_THRESHOLD = 0.6
_RISK_THRESHOLD = 0.5
_EMOTIONAL_THRESHOLD = 0.7
_PRECISION_THRESHOLD = 0.8


class DeliberationEngine:
    """
    The dual-process cognitive architecture.

    Receives workspace broadcasts, assesses situation, routes to fast or slow
    deliberation, and returns a constitutional Intent (or None for do-nothing).

    Performance targets:
      - Fast path: ≤200ms total (≤150ms procedure + ≤50ms Equor critical)
      - Slow path: ≤15000ms total (≤10000ms generation + ≤500ms Equor + overhead)
    """

    def __init__(
        self,
        goal_manager: GoalManager,
        policy_generator: BasePolicyGenerator,
        efe_evaluator: EFEEvaluator,
        equor: EquorService,
        drive_weights: dict[str, float] | None = None,
        fast_path_timeout_ms: int = 300,
        slow_path_timeout_ms: int = 15000,
        cost_calculator: CognitionCostCalculator | None = None,
        tournament_engine: TournamentEngine | None = None,
    ) -> None:
        self._goals = goal_manager
        self._policy_gen = policy_generator
        self._efe = efe_evaluator
        self._equor = equor
        self._tournament_engine = tournament_engine
        self._drive_weights = drive_weights or {
            "coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0
        }
        self._fast_timeout = fast_path_timeout_ms / 1000.0
        self._slow_timeout = slow_path_timeout_ms / 1000.0
        self._last_equor_check: ConstitutionalCheck | None = None
        self._logger = logger.bind(system="nova.deliberation_engine")
        # Optional callable that returns causal laws summary string for LLM context.
        self._causal_laws_provider: Any = None

        # Instance-level deliberation thresholds — start at spec defaults,
        # updated by update_thresholds() as Evo learns better values.
        self._novelty_threshold: float = _NOVELTY_THRESHOLD
        self._risk_threshold: float = _RISK_THRESHOLD
        self._emotional_threshold: float = _EMOTIONAL_THRESHOLD
        self._precision_threshold: float = _PRECISION_THRESHOLD

        # Allostatic EFE threshold modulation (updated by update_somatic_thresholds())
        # Stored as *deltas* applied to the instance thresholds at assessment time.
        self._novelty_threshold_delta: float = 0.0
        self._precision_threshold_delta: float = 0.0
        # Do-nothing EFE override: when urgency is high, the baseline rises so
        # inaction has to beat a higher bar. None = use DO_NOTHING_EFE constant.
        self._do_nothing_efe_override: float | None = None

        # Equor unavailability callback - wired by NovaService after initialize().
        # Called when Equor times out or raises a connectivity exception so
        # service.py can log a Thymos DEGRADATION incident. Signature: (reason: str) -> None.
        self._equor_failure_cb: Any | None = None

        # Free energy budget - information-theoretic pressure valve.
        # Tracks cumulative surprise (nats) per window. When exhausted, triggers
        # emergency consolidation and reduces policy generation diversity.
        self._fe_budget = FreeEnergyBudget()

        # Cognition cost calculator - metabolic budgeting for deliberation.
        # When present, each cycle gets a USD budget based on decision importance.
        # The budget constrains how many policies are generated and influences
        # EFE scoring via the cognition_cost_term.
        self._cost_calc = cost_calculator
        self._telos: Any = None
        self._logos: Any = None
        self._current_soma_signal: Any = None
        # Soma trajectory-based do-nothing EFE delta (independent of urgency override)
        self._do_nothing_efe_delta: float = 0.0
        # Organism telemetry summary - injected into slow-path prompt every 50 cycles.
        # Updated atomically by set_organism_summary(); never None after first telemetry.
        self._organism_summary: str = ""
        # Novel action callback - fired when the selected policy contains a
        # propose_novel_action step.  Signature:
        #   async (goal: Goal, step_parameters: dict) -> None
        # Wired by NovaService after initialize() via set_novel_action_cb().
        self._novel_action_cb: Any | None = None

    @property
    def fe_budget(self) -> FreeEnergyBudget:
        """Expose the free energy budget for NovaService to read/update."""
        return self._fe_budget

    def update_fe_budget_params(self, budget_nats: float, threshold_fraction: float) -> None:
        """
        Apply Evo-tuned budget parameters.
        Called by NovaService after consolidation pushes new parameter values.
        """
        self._fe_budget.budget_nats = budget_nats
        self._fe_budget.threshold_fraction = threshold_fraction
        self._logger.debug(
            "fe_budget_params_updated",
            budget_nats=budget_nats,
            threshold_fraction=threshold_fraction,
        )

    def reset_fe_budget(self) -> None:
        """Reset the budget after consolidation completes or window expires."""
        self._fe_budget.reset()
        self._logger.info(
            "fe_budget_reset",
            interrupts_total=self._fe_budget.interrupts_triggered,
        )

    def update_drive_weights(self, weights: dict[str, float]) -> None:
        """Called by NovaService when constitution changes."""
        self._drive_weights = weights

    def set_causal_laws_provider(self, provider: Any) -> None:
        """
        Wire a callable that returns the current causal laws summary string.

        Called by NovaService after initialize(). Provider signature:
        ``() -> str`` (e.g. ``nova_service.get_causal_knowledge_summary``).
        """
        self._causal_laws_provider = provider

    def set_telos(self, telos: Any) -> None:
        """Wire Telos so policy EFE scoring can account for effective_I impact."""
        self._telos = telos
        self._logger.info("telos_wired_to_deliberation_engine")

    def set_logos(self, logos: Any) -> None:
        """Wire Logos so policy generation adapts to world model state."""
        self._logos = logos
        self._logger.info("logos_wired_to_deliberation_engine")

    def set_organism_summary(self, summary: str) -> None:
        """Update cached organism-state summary for injection into slow-path prompts."""
        self._organism_summary = summary

    def modulate_policy_k_from_pressure(self, cognitive_pressure: float) -> None:
        """
        Reduce policy generation K when Logos cognitive pressure is high.

        Cognitive pressure > 0.75 signals the organism's knowledge budget is
        strained. Generating fewer candidate policies conserves resources.
        Pressure > 0.90 → K = 2 (minimum exploration).
        """
        if cognitive_pressure > 0.90:
            self._fe_budget.reduced_k = 2
            self._fe_budget.is_exhausted = True
        elif cognitive_pressure > 0.75:
            # Linear interpolation: pressure 0.75→0.90 maps K from normal to reduced
            fraction = (cognitive_pressure - 0.75) / 0.15
            target_k = round(
                self._fe_budget.normal_k - fraction * (self._fe_budget.normal_k - 2)
            )
            self._fe_budget.reduced_k = max(2, target_k)
        self._logger.debug(
            "policy_k_modulated_by_logos_pressure",
            cognitive_pressure=round(cognitive_pressure, 3),
            effective_k=self._fe_budget.effective_k,
        )

    def update_soma_signal(self, signal: Any) -> None:
        """
        Receive the full allostatic signal for deep Soma integration.

        Beyond the urgency/arousal threshold shifting (update_somatic_thresholds),
        this gives the deliberation engine access to:
        - Energy tier: gate expensive policy generation under energy depletion
        - Precision weights: modulate which EFE components matter more
        - Trajectory heading: bias toward stabilising vs explorative policies
        - Nearest attractor: context for self-regulation goal framing
        """
        self._current_soma_signal = signal

        # Energy-gated policy generation: under depleted energy, force reduced K
        # to conserve metabolic resources for essential operations
        from systems.soma.types import InteroceptiveDimension
        energy = signal.state.sensed.get(InteroceptiveDimension.ENERGY, 0.6)
        if energy < 0.3:
            # Depleted or critical - minimum exploration
            self._fe_budget.reduced_k = 2
            self._logger.debug(
                "energy_gated_policy_k", energy=round(energy, 3), effective_k=2
            )
        elif energy < 0.5:
            # Conserving - moderate reduction
            self._fe_budget.reduced_k = max(2, self._fe_budget.normal_k - 1)

        # Trajectory heading influences do-nothing EFE: if heading toward an
        # attractor (stabilising), inaction is safer; if heading away
        # (destabilising), inaction costs more.
        heading = getattr(signal, "trajectory_heading", "transient")
        if heading == "toward_attractor":
            # Organism is stabilising - inaction is less costly
            self._do_nothing_efe_delta = max(self._do_nothing_efe_delta - 0.05, -0.15)
        elif heading == "away_from_attractor":
            # Organism is destabilising - inaction costs more
            self._do_nothing_efe_delta = min(self._do_nothing_efe_delta + 0.10, 0.25)

    def set_tournament_engine(self, engine: TournamentEngine) -> None:
        """Wire the Evo tournament engine for hypothesis A/B routing."""
        self._tournament_engine = engine

    def set_policy_generator(self, generator: BasePolicyGenerator) -> None:
        """
        Hot-swap the active policy generator.

        Called by NovaService when HotReloader discovers a new
        BasePolicyGenerator subclass.  The swap is atomic at Python's
        reference level - any in-flight slow-path call using the old
        generator will complete normally; new calls get the new generator.
        """
        self._policy_gen = generator

    def update_thresholds(
        self,
        novelty: float | None = None,
        risk: float | None = None,
        emotional: float | None = None,
        precision: float | None = None,
    ) -> None:
        """
        Update deliberation routing thresholds from Evo learning.

        Called by NovaService after consolidation discovers better threshold values
        via hypothesis testing. All values are clamped to [0.1, 0.99] to prevent
        pathological states (never-slow or always-slow).
        """
        if novelty is not None:
            self._novelty_threshold = max(0.1, min(0.99, novelty))
        if risk is not None:
            self._risk_threshold = max(0.1, min(0.99, risk))
        if emotional is not None:
            self._emotional_threshold = max(0.1, min(0.99, emotional))
        if precision is not None:
            self._precision_threshold = max(0.1, min(0.99, precision))
        self._logger.info(
            "deliberation_thresholds_updated",
            novelty=self._novelty_threshold,
            risk=self._risk_threshold,
            emotional=self._emotional_threshold,
            precision=self._precision_threshold,
        )

    def update_somatic_thresholds(self, urgency: float, arousal: float) -> None:
        """
        Shift EFE deliberation thresholds based on Soma's somatic state.

        Called by NovaService.receive_broadcast() before deliberation when
        Soma is wired and has produced a signal.

        Mapping (all deltas are negative = thresholds lowered = more sensitive):

        High arousal (>0.6):
          - Lowers _PRECISION_THRESHOLD by up to -0.15 at arousal=1.0.
            Under physiological stress, even moderately precise broadcasts
            warrant slow deliberation.

        High urgency (>threshold):
          - Lowers _NOVELTY_THRESHOLD by up to -0.20 at urgency=1.0.
            When something is wrong, almost any novel event needs attention.
          - Raises do-nothing EFE baseline by up to +0.15 at urgency=1.0.
            Inaction is harder to justify when the organism is far from setpoints.

        The deltas are clamped so they cannot push thresholds below 0.1
        (prevents pathological always-slow-path under sustained stress).
        """
        # Arousal → precision threshold lowering
        if arousal > 0.6:
            self._precision_threshold_delta = -min(0.15, (arousal - 0.6) * 0.375)
        else:
            self._precision_threshold_delta = 0.0

        # Urgency → novelty threshold lowering + do-nothing EFE raising
        if urgency > 0.3:
            urgency_excess = (urgency - 0.3) / 0.7  # 0→1 as urgency 0.3→1.0
            self._novelty_threshold_delta = -min(0.20, urgency_excess * 0.20)
            self._do_nothing_efe_override = -0.10 + min(0.15, urgency_excess * 0.15)
        else:
            self._novelty_threshold_delta = 0.0
            self._do_nothing_efe_override = None

        self._logger.debug(
            "somatic_thresholds_updated",
            urgency=round(urgency, 3),
            arousal=round(arousal, 3),
            novelty_threshold=round(self._novelty_threshold + self._novelty_threshold_delta, 3),
            precision_threshold=round(self._precision_threshold + self._precision_threshold_delta, 3),
            do_nothing_efe=self._do_nothing_efe_override,
        )

    def set_equor_failure_callback(self, cb: Any) -> None:
        """
        Wire a callback invoked when Equor is unreachable or times out.

        Signature: (reason: str) -> None
        NovaService uses this to raise a Thymos DEGRADATION incident so the
        immune system tracks constitutional gate unavailability.
        """
        self._equor_failure_cb = cb

    def set_novel_action_cb(self, cb: Any) -> None:
        """
        Wire an async callback invoked when the selected policy contains a
        propose_novel_action step.

        Signature: async (goal: Goal, step_parameters: dict) -> None
        NovaService uses this to emit NOVEL_ACTION_REQUESTED onto the Synapse
        bus so Simula can generate the executor.
        """
        self._novel_action_cb = cb

    @property
    def last_equor_check(self) -> ConstitutionalCheck | None:
        """The Equor check from the most recent approved intent."""
        return self._last_equor_check

    async def deliberate(
        self,
        broadcast: WorkspaceBroadcast,
        belief_state: BeliefState,
        affect: AffectState,
        belief_delta_is_conflicting: bool = False,
        memory_traces: list[dict[str, Any]] | None = None,
        allostatic_mode: bool = False,
        allostatic_error_dim: Any = None,
    ) -> tuple[Intent | None, DecisionRecord, list[tuple[Policy, EFEScore]]]:
        """
        Main deliberation entry point.

        Returns (Intent | None, DecisionRecord, rejected_policies).
        Returns None when the best action is no action.
        DecisionRecord is always returned for observability.
        rejected_policies contains (Policy, EFEScore) tuples for each
        non-selected, non-do-nothing policy from the slow path (empty for fast path).
        """
        start = time.monotonic()
        self._last_equor_check = None  # Reset per-deliberation

        try:
            # End-to-end timeout: the entire deliberation (including possible
            # fast→slow escalation) must complete within the slow-path budget.
            async with asyncio.timeout(self._slow_timeout):
                return await self._deliberate_inner(
                    broadcast, belief_state, affect,
                    belief_delta_is_conflicting, memory_traces, start,
                    allostatic_mode, allostatic_error_dim,
                )
        except TimeoutError:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._logger.warning("deliberation_end_to_end_timeout", elapsed_ms=elapsed_ms)
            record = DecisionRecord(
                broadcast_id=broadcast.broadcast_id,
                path="timeout",
                latency_ms=elapsed_ms,
            )
            return None, record, []

    async def _deliberate_inner(
        self,
        broadcast: WorkspaceBroadcast,
        belief_state: BeliefState,
        affect: AffectState,
        belief_delta_is_conflicting: bool,
        memory_traces: list[dict[str, Any]] | None,
        start: float,
        allostatic_mode: bool = False,
        allostatic_error_dim: Any = None,
    ) -> tuple[Intent | None, DecisionRecord, list[tuple[Policy, EFEScore]]]:
        """Inner deliberation logic, called within the end-to-end timeout."""

        # ── Free energy budget gate ────────────────────────────────────
        # Accumulate prediction error from this broadcast and check whether
        # the organism's surprise budget is exhausted. If so, skip
        # deliberation and signal that emergency consolidation is needed.
        pe_magnitude = 0.0
        if broadcast.salience.prediction_error is not None:
            pe_magnitude = broadcast.salience.prediction_error.magnitude

        budget_interrupt = False
        if self._fe_budget.is_exhausted:
            # Budget already exhausted on a previous cycle - stay in
            # reduced mode until consolidation resets it.
            budget_interrupt = True
        elif self._fe_budget.would_exhaust(pe_magnitude):
            # This percept tips us over the threshold → trigger interrupt.
            nats_added = self._fe_budget.accumulate(pe_magnitude)
            self._fe_budget.is_exhausted = True
            self._fe_budget.interrupts_triggered += 1
            budget_interrupt = True
            self._logger.warning(
                "free_energy_budget_exhausted",
                spent=round(self._fe_budget.spent_nats, 3),
                percept_added=round(nats_added, 3),
                total=round(self._fe_budget.spent_nats, 3),
                threshold=round(self._fe_budget.threshold_nats, 3),
                interrupts_total=self._fe_budget.interrupts_triggered,
            )
        else:
            # Budget not exhausted - accumulate and proceed normally.
            self._fe_budget.accumulate(pe_magnitude)

        if budget_interrupt:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            record = DecisionRecord(
                broadcast_id=broadcast.broadcast_id,
                path="budget_exhausted",
                latency_ms=elapsed_ms,
                fe_budget_spent_nats=round(self._fe_budget.spent_nats, 4),
                fe_budget_remaining_nats=round(self._fe_budget.remaining_nats, 4),
                fe_budget_interrupt=True,
            )
            return None, record, []

        # ── Normal deliberation path ───────────────────────────────────

        # Recompute goal priorities before deliberating
        priority_ctx = PriorityContext(
            current_affect=affect,
            drive_weights=self._drive_weights,
        )
        self._goals.recompute_priorities(priority_ctx)

        # When allostatic urgency is high, reorder goals to prioritise those
        # addressing the dominant error dimension (e.g. energy depletion → rest goals)
        if allostatic_mode and allostatic_error_dim is not None:
            self._reorder_goals_for_allostatic_mode(
                active_goals=self._goals.active_goals,
                error_dimension=allostatic_error_dim,
            )

        # Assess situation to determine path
        assessment = self._assess_situation(
            broadcast=broadcast,
            belief_conflict=belief_delta_is_conflicting,
        )

        record = DecisionRecord(
            broadcast_id=broadcast.broadcast_id,
            situation_assessment=assessment,
            fe_budget_spent_nats=round(self._fe_budget.spent_nats, 4),
            fe_budget_remaining_nats=round(self._fe_budget.remaining_nats, 4),
        )

        # Yield control briefly so the Synapse clock and other coroutines
        # don't starve while we begin the potentially long deliberation.
        await asyncio.sleep(0)

        # Route to appropriate path
        rejected: list[tuple[Policy, EFEScore]] = []
        cognition_budget: CognitionBudget | None = None
        if assessment.requires_deliberation:
            self._logger.debug("deliberation_slow_path", broadcast_id=broadcast.broadcast_id)
            intent, rejected, cognition_budget = await self._slow_path(
                broadcast, assessment, belief_state, affect, memory_traces
            )
            path = "slow"
        else:
            self._logger.debug("deliberation_fast_path", broadcast_id=broadcast.broadcast_id)
            intent, escalated = await self._fast_path(broadcast, assessment, belief_state, affect)
            path = "slow" if escalated else "fast"
            if escalated and intent is None:
                # Fast path escalated - yield before entering the heavy slow path
                await asyncio.sleep(0)
                intent, rejected, cognition_budget = await self._slow_path(
                    broadcast, assessment, belief_state, affect, memory_traces
                )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Settle any inter-budget debt from this cycle
        if cognition_budget is not None and self._cost_calc is not None:
            self._cost_calc.settle_cycle_debt(cognition_budget)

        # ── Tournament context: check if any selected policy's reasoning
        # references a hypothesis that is in an active A/B tournament ──
        tournament_ctx = self._check_tournament_context(intent)

        # Enrich record with slow-path policy selection data
        update: dict[str, Any] = {
            "path": (
                path if intent is not None
                else ("do_nothing" if not assessment.requires_deliberation else "no_goal")
            ),
            "intent_dispatched": intent is not None,
            "latency_ms": elapsed_ms,
        }
        if rejected:
            # Extract chosen policy info from the intent's decision trace
            selected_name = ""
            all_efe: dict[str, float] = {}
            if intent is not None and intent.decision_trace:
                all_efe = intent.decision_trace.free_energy_scores
                # The selected policy is the one NOT in rejected and NOT do_nothing
                rejected_names = {p.name for p, _ in rejected}
                for name in all_efe:
                    if name not in rejected_names and name != "do_nothing":
                        selected_name = name
                        break
            update["selected_policy_name"] = selected_name
            update["efe_scores"] = all_efe
            update["policies_generated"] = len(rejected) + 1  # rejected + selected

        # Enrich record with cognition cost data
        if cognition_budget is not None:
            update["cognition_cost_total_usd"] = round(cognition_budget.spent_usd, 6)
            update["cognition_budget_allocated_usd"] = round(cognition_budget.allocated_usd, 4)
            update["cognition_budget_remaining_usd"] = round(cognition_budget.remaining_usd, 4)
            update["cognition_budget_utilisation"] = round(cognition_budget.utilisation, 3)
            update["cognition_budget_importance"] = cognition_budget.importance.value
            update["cognition_budget_borrowed_usd"] = round(cognition_budget.borrowed_usd, 4)
            update["cognition_budget_early_stop"] = (
                cognition_budget.policies_costed < (len(rejected) + 1) if rejected else False
            )

        # Attach tournament metadata if applicable
        if tournament_ctx is not None:
            update["tournament_id"] = tournament_ctx.tournament_id
            update["tournament_hypothesis_id"] = tournament_ctx.hypothesis_id

        # Mark slow-path records with an intent as RE training eligible.
        # Fast-path and budget-exhausted records carry too little signal.
        # model_used is "claude" until Thompson sampling routes to RE.
        # (Spec §21 - re_training_eligible field, previously missing.)
        if path == "slow" and intent is not None:
            update["re_training_eligible"] = True
            update["model_used"] = "claude"

        record = record.model_copy(update=update)

        self._logger.info(
            "deliberation_complete",
            path=record.path,
            intent_dispatched=intent is not None,
            latency_ms=elapsed_ms,
            rejected_policies=len(rejected),
            **(cognition_budget.to_log_dict() if cognition_budget else {}),
        )
        return intent, record, rejected

    # ─── Telos Policy Scoring ─────────────────────────────────────

    def _apply_telos_scoring(
        self,
        scored: list[tuple[Any, Any]],
        goal: Any,
    ) -> list[tuple[Any, Any]]:
        """
        Adjust EFE totals by Telos effective_I impact.

        For each policy, build a TelosPolicyScorer-compatible dict and get the
        composite_score. Negative composite = hurts effective_I → add EFE penalty.
        Positive composite = helps effective_I → subtract EFE bonus.
        The coefficient (0.15) keeps this comparable to the constitutional_alignment
        weight (0.20) without dominating the decision.

        Uses the latest EffectiveIntelligenceReport from Telos so scoring
        reflects the actual drive multiplier state, not defaults.
        """
        telos_efe_weight = 0.15

        # Fetch the latest effective_I report - this is what makes policies
        # compete on effective_I (the real intelligence) not just nominal_I.
        current_report = None
        try:
            integrator = getattr(self._telos, "integrator", None)
            if integrator is not None:
                current_report = getattr(integrator, "last_report", None)
        except Exception:
            pass

        adjusted: list[tuple[Any, Any]] = []
        for policy, efe_score in scored:
            if policy.id == "do_nothing":
                adjusted.append((policy, efe_score))
                continue

            # Build the policy descriptor dict from Nova policy fields.
            # Use pragmatic success_probability as proxy for expected_nominal_I_delta.
            pragmatic_score = 0.0
            if (
                hasattr(efe_score, "pragmatic")
                and hasattr(efe_score.pragmatic, "success_probability")
            ):
                pragmatic_score = efe_score.pragmatic.success_probability

            policy_dict = {
                "expected_welfare_impact": getattr(policy, "expected_welfare_impact", 0.0),
                "expected_coherence_impact": getattr(policy, "expected_coherence_impact", 0.0),
                "expected_honesty_impact": getattr(policy, "expected_honesty_impact", 0.0),
                "expected_growth_impact": getattr(policy, "expected_growth_impact", 0.0),
                "expected_nominal_I_delta": pragmatic_score,
                "goal_description": getattr(goal, "description", ""),
            }

            # Pass the real EffectiveIntelligenceReport so Telos scores
            # against actual drive multiplier state, not defaults.
            telos_score = self._telos.score_policy(policy_dict, current_report)
            telos_adj = -telos_score.composite_score * telos_efe_weight

            # Apply misalignment penalty: if nominal rises but effective drops
            if telos_score.misalignment_risk:
                telos_adj += 0.3  # 0.3 EFE penalty - significant but not overwhelming

            # Alignment gap weighting: when effective_I lags nominal_I significantly,
            # amplify the Telos adjustment to steer behaviour toward alignment.
            if current_report is not None and current_report.alignment_gap_warning:
                telos_adj *= 1.5  # 50% amplification under alignment gap

            new_total = efe_score.total + telos_adj
            adjusted_score = efe_score.model_copy(update={"total": new_total})
            adjusted.append((policy, adjusted_score))

        # Re-sort: lower EFE total = more preferred
        adjusted.sort(key=lambda x: x[1].total)
        return adjusted

    # ─── Tournament Integration ──────────────────────────────────

    def _check_tournament_context(
        self,
        intent: Intent | None,
    ) -> TournamentContext | None:
        """
        Check if the dispatched intent's decision trace references a hypothesis
        that is in an active A/B tournament. If so, sample from the tournament
        to decide which hypothesis to credit and return a TournamentContext.

        This is the emit side - Evo's outcome handler consumes it later.
        """
        if intent is None or self._tournament_engine is None:
            return None

        # The tournament engine maintains an index of hypothesis_id → tournament_id.
        # We look up each hypothesis referenced by any active tournament.
        # Since policies don't directly carry hypothesis IDs, tournaments are
        # triggered when the hypothesis engine has placed hypotheses into competition.
        # The tournament engine's sample_hypothesis() is called here to route.
        active_tournaments = self._tournament_engine.get_active_tournaments()
        if not active_tournaments:
            return None

        # Pick the first active tournament and sample from it.
        # Future: could match tournament hypotheses to the intent's policy domain.
        tournament = active_tournaments[0]
        ctx = self._tournament_engine.sample_hypothesis(tournament)
        ctx.policy_id = intent.id

        self._logger.debug(
            "tournament_context_emitted",
            tournament_id=ctx.tournament_id,
            hypothesis_id=ctx.hypothesis_id,
            policy_id=ctx.policy_id,
        )
        return ctx

    # ─── Allostatic Goal Reordering ───────────────────────────────

    def _reorder_goals_for_allostatic_mode(
        self,
        active_goals: list[Any],
        error_dimension: Any,
    ) -> None:
        """
        Reorder active goals so those addressing the dominant allostatic error
        dimension are prioritised. Called when Soma urgency > threshold.

        Dimension→keyword mapping: ENERGY→rest/recover, SOCIAL_CHARGE→engage,
        COHERENCE→clarify, INTEGRITY→honest, AROUSAL→calm, TEMPORAL_PRESSURE→urgent.
        """
        _dim_keywords: dict[str, list[str]] = {
            "energy": ["rest", "recover", "sleep", "recharge", "restore"],
            "social_charge": ["engage", "connect", "communicate", "respond", "interact"],
            "coherence": ["clarify", "understand", "learn", "resolve", "explore"],
            "integrity": ["honest", "truthful", "authentic", "correct", "admit"],
            "arousal": ["calm", "settle", "ground", "stabilise", "slow"],
            "temporal_pressure": ["urgent", "immediate", "now", "quickly", "deadline"],
            "valence": ["positive", "good", "help", "care", "support"],
            "confidence": ["verify", "confirm", "check", "validate", "test"],
            "curiosity_drive": ["explore", "discover", "learn", "investigate", "try"],
        }
        dim_str = str(error_dimension).lower()
        keywords = _dim_keywords.get(dim_str, [])
        if not keywords:
            return

        def _allostatic_weight(goal: Any) -> float:
            description = getattr(goal, "description", "").lower()
            if any(kw in description for kw in keywords):
                return 1.0
            return 0.0

        # Stable sort: allostatic matches float to top, original order preserved otherwise
        active_goals.sort(key=lambda g: (1.0 - _allostatic_weight(g), -getattr(g, "priority", 0.5)))

        self._logger.debug(
            "allostatic_goal_reorder",
            error_dimension=dim_str,
            top_goal=(active_goals[0].description[:60] if active_goals else "none"),
        )

    # ─── Situation Assessment ─────────────────────────────────────

    def _assess_situation(
        self,
        broadcast: WorkspaceBroadcast,
        belief_conflict: bool,
    ) -> SituationAssessment:
        """
        Determine if deliberative (slow) or habitual (fast) processing is needed.
        Must complete in ≤20ms.

        Thresholds are modulated by somatic state via update_somatic_thresholds():
          - High arousal lowers _PRECISION_THRESHOLD (stress → more broadcasts warrant slow path)
          - High urgency lowers _NOVELTY_THRESHOLD (allostatic pressure → more novelty triggers
            deliberation)
        """
        salience_scores = broadcast.salience.scores if broadcast.salience.scores else {}
        novelty = salience_scores.get("novelty", 0.0)
        risk = salience_scores.get("risk", 0.0)
        emotional = salience_scores.get("emotional", 0.0)
        precision = broadcast.precision

        # Apply somatic threshold deltas to instance thresholds (clamped to minimum 0.1)
        effective_novelty_threshold = max(0.1, self._novelty_threshold + self._novelty_threshold_delta)
        effective_precision_threshold = max(
            0.1, self._precision_threshold + self._precision_threshold_delta
        )

        # Bounty discovery observations must always go through the slow path
        # so the LLM PolicyGenerator can extract bounty parameters (bounty_id,
        # issue_url, repository_url) and generate a proper solve_bounty policy.
        is_bounty = _is_bounty_discovery(broadcast)

        requires_deliberation = (
            novelty > effective_novelty_threshold
            or risk > self._risk_threshold
            or emotional > self._emotional_threshold
            or belief_conflict
            or precision > effective_precision_threshold
            or is_bounty
        )

        has_procedure = find_matching_procedure(broadcast) is not None

        return SituationAssessment(
            novelty=novelty,
            risk=risk,
            emotional_intensity=emotional,
            belief_conflict=belief_conflict,
            requires_deliberation=requires_deliberation,
            has_matching_procedure=has_procedure,
            broadcast_precision=precision,
        )

    # ─── Fast Path ────────────────────────────────────────────────

    async def _fast_path(
        self,
        broadcast: WorkspaceBroadcast,
        assessment: SituationAssessment,
        belief_state: BeliefState,
        affect: AffectState,
    ) -> tuple[Intent | None, bool]:
        """
        System 1: Pattern-match → build intent → Equor critical review.
        Returns (intent | None, escalated_to_slow).
        """
        try:
            async with asyncio.timeout(self._fast_timeout):
                procedure = find_matching_procedure(broadcast)
                if procedure is None:
                    return None, True  # No matching procedure → escalate

                # Economic procedures delegate strategy selection to EFE scoring
                if procedure.get("domain") == "economic" and hasattr(self._policy_generator, "generate_economic_intent"):
                    if _broadcast_is_hungry(broadcast):
                        # Hunger override: force bounty hunting regardless of EFE
                        policy = procedure_to_policy(procedure)
                    else:
                        policy = self._policy_generator.generate_economic_intent(belief_state)
                else:
                    policy = procedure_to_policy(procedure)

                # Find or create a goal for this intent
                goal = self._goals.find_relevant_goal(broadcast)
                if goal is None:
                    goal = self._goals.create_from_broadcast(broadcast)
                if goal is None:
                    return None, False  # No goal → do nothing

                intent = _policy_to_intent(
                    policy, goal, path="fast", confidence=procedure["success_rate"]
                )

                # Equor critical-path review (≤50ms, cache-only, no DB/LLM I/O)
                try:
                    check = await asyncio.wait_for(
                        self._equor.review_critical(intent),
                        timeout=0.1,  # 100ms hard cap for critical path
                    )
                except (TimeoutError, asyncio.TimeoutError, OSError, ConnectionError) as exc:
                    reason = f"equor_fast_path_unavailable: {exc!s}"
                    self._logger.warning("equor_unavailable_fast_path", reason=reason)
                    if self._equor_failure_cb is not None:
                        try:
                            self._equor_failure_cb(reason)
                        except Exception:
                            pass
                    return None, True  # Escalate to slow path; do-nothing is fallback

                if check.verdict == Verdict.APPROVED:
                    self._last_equor_check = check
                    return intent, False
                elif check.verdict == Verdict.MODIFIED:
                    self._last_equor_check = check
                    intent = _apply_modifications(intent, check.modifications)
                    return intent, False
                else:
                    # Denied by Equor → escalate to slow path
                    self._logger.info("fast_path_equor_denied_escalating", intent_id=intent.id)
                    return None, True

        except TimeoutError:
            self._logger.warning("fast_path_timeout_escalating")
            return None, True
        except Exception as exc:
            self._logger.error("fast_path_error", error=str(exc))
            return None, True

    # ─── Slow Path ────────────────────────────────────────────────

    async def _slow_path(
        self,
        broadcast: WorkspaceBroadcast,
        assessment: SituationAssessment,
        belief_state: BeliefState,
        affect: AffectState,
        memory_traces: list[dict[str, Any]] | None,
    ) -> tuple[Intent | None, list[tuple[Policy, EFEScore]], CognitionBudget | None]:
        """
        System 2: Generate → EFE score → select → Equor standard review.

        Returns (intent, rejected_policies, cognition_budget) where:
        - rejected_policies contains all scored non-do-nothing policies that were NOT selected
        - cognition_budget tracks the metabolic cost of this deliberation cycle

        Yield points (asyncio.sleep(0)) are inserted between major phases
        to prevent event loop starvation during the slow path budget.
        """
        cognition_budget: CognitionBudget | None = None

        try:
            async with asyncio.timeout(self._slow_timeout):
                # ── Find or create goal ──
                goal = self._goals.find_relevant_goal(broadcast)
                if goal is None:
                    goal = self._goals.create_from_broadcast(broadcast)
                if goal is None:
                    return None, [], None  # No goal warrants no action

                # ── Allocate cognition budget based on decision importance ──
                if self._cost_calc is not None:
                    salience_composite = broadcast.salience.composite if broadcast.salience else 0.5
                    risk_score = assessment.risk
                    has_external = risk_score > 0.5  # Candidates not yet generated; use risk proxy
                    importance = CognitionCostCalculator.classify_importance(
                        salience_composite=salience_composite,
                        goal_priority=goal.priority,
                        risk_score=risk_score,
                        has_external_action=has_external,
                    )
                    cognition_budget = self._cost_calc.allocate_budget(importance)
                    self._logger.debug(
                        "cognition_budget_allocated",
                        importance=importance.value,
                        allocated_usd=round(cognition_budget.allocated_usd, 4),
                        debt_carried=round(self._cost_calc.outstanding_debt_usd, 4),
                    )

                # ── Extract situation summary for policy generation ──
                situation = _extract_situation_summary(broadcast)
                # Append organism state so the LLM deliberates with full awareness.
                if self._organism_summary:
                    situation = (
                        f"{situation}\n[Organism] {self._organism_summary}"
                    )

                # ── Generate candidate policies (up to 3000ms) ──
                causal_laws = (
                    self._causal_laws_provider()
                    if self._causal_laws_provider is not None
                    else ""
                )
                candidates = await self._policy_gen.generate_candidates(
                    goal=goal,
                    situation_summary=situation,
                    beliefs=belief_state,
                    affect=affect,
                    memory_traces=memory_traces,
                    causal_laws_summary=causal_laws,
                )

                if not candidates:
                    return None, [], cognition_budget

                # ── Apply FE budget pressure: reduce candidate diversity ──
                # When the surprise budget is pressured, trim candidate list
                # to reduce cognitive load (fewer EFE evaluations, faster cycle).
                effective_k = self._fe_budget.effective_k
                if len(candidates) > effective_k:
                    # Keep the do_nothing policy + top (effective_k - 1) candidates
                    do_nothing = [p for p in candidates if p.id == "do_nothing"]
                    others = [p for p in candidates if p.id != "do_nothing"]
                    candidates = others[:effective_k - len(do_nothing)] + do_nothing
                    self._logger.debug(
                        "fe_budget_k_reduction",
                        effective_k=effective_k,
                        candidates_after=len(candidates),
                        budget_utilisation=round(self._fe_budget.utilisation, 3),
                    )

                # ── Cognition budget early stop: trim candidates if budget is tight ──
                early_stop = False
                if cognition_budget is not None and self._cost_calc is not None:
                    active_candidates = [p for p in candidates if p.id != "do_nothing"]
                    marginal = self._cost_calc.estimate_policy_marginal_cost()
                    estimated_total = marginal * len(active_candidates)
                    if (
                        estimated_total > cognition_budget.remaining_usd
                        and len(active_candidates) > 1
                    ):
                        # Trim: keep as many as the budget can afford (min 1)
                        affordable = max(1, int(cognition_budget.remaining_usd / marginal))
                        if affordable < len(active_candidates):
                            do_nothing_policies = [p for p in candidates if p.id == "do_nothing"]
                            candidates = active_candidates[:affordable] + do_nothing_policies
                            early_stop = True
                            cognition_budget.policies_costed = affordable
                            self._logger.info(
                                "cognition_budget_early_stop",
                                affordable=affordable,
                                original=len(active_candidates),
                                budget_remaining=round(cognition_budget.remaining_usd, 4),
                                marginal_cost=round(marginal, 6),
                            )

                # Yield after the heaviest LLM call so the clock can tick
                await asyncio.sleep(0)

                # ── Evaluate EFE for all candidates (parallelised) ──
                scored = await self._efe.evaluate_all(
                    policies=candidates,
                    goal=goal,
                    beliefs=belief_state,
                    affect=affect,
                    drive_weights=self._drive_weights,
                    cognition_budget=cognition_budget,
                )

                # ── Charge the cognition budget for EFE evaluation ──
                if cognition_budget is not None and self._cost_calc is not None:
                    eval_cost = self._cost_calc.estimate_deliberation_cost(
                        num_policies=len(candidates),
                        use_llm_pragmatic=self._efe._use_llm,
                        use_llm_epistemic=self._efe._use_llm,
                    )
                    cognition_budget.charge(eval_cost)
                    cognition_budget.policies_costed = len(candidates)
                    if early_stop:
                        # Mark the early stop in the budget
                        pass  # Already logged above

                # scored is sorted: lowest EFE first
                # Apply somatic do-nothing EFE override: high urgency raises the
                # do-nothing baseline, making inaction harder to justify.
                # The trajectory delta from Soma modulates on top: heading toward
                # attractor → inaction cheaper; heading away → inaction costlier.
                _do_nothing_adj = self._do_nothing_efe_delta
                if self._do_nothing_efe_override is not None or _do_nothing_adj != 0.0:
                    scored = [
                        (
                            policy,
                            score.model_copy(update={
                                "total": (
                                    self._do_nothing_efe_override
                                    if self._do_nothing_efe_override is not None
                                    else score.total
                                ) + _do_nothing_adj
                            })
                            if policy.id == "do_nothing"
                            else score,
                        )
                        for policy, score in scored
                    ]
                    # Re-sort after override
                    scored.sort(key=lambda x: x[1].total)

                # ── Telos: adjust EFE by effective_I impact ──────────
                # Policies that improve effective_I (not just nominal_I) get an
                # EFE bonus (lower total → preferred). Misalignment risk incurs
                # a penalty. This is the mechanism by which Telos transforms
                # Nova from "maximise nominal I" to "maximise real intelligence."
                if self._telos is not None:
                    try:
                        scored = self._apply_telos_scoring(scored, goal)
                    except Exception as exc:
                        self._logger.warning("telos_scoring_failed", error=str(exc))

                # If do-nothing wins, return None (no counterfactuals worth archiving)
                if scored and scored[0][0].id == "do_nothing":
                    self._logger.info(
                        "do_nothing_policy_selected",
                        goal=goal.description[:60],
                        do_nothing_efe=scored[0][1].total,
                        somatic_override=self._do_nothing_efe_override,
                    )
                    return None, [], cognition_budget

                # Yield before the Equor review loop
                await asyncio.sleep(0)

                # ── Equor review with retry on denial ──
                for policy, efe_score in scored:
                    if policy.id == "do_nothing":
                        continue  # Skip do-nothing - if we're here, we want to act

                    # ── Novel action interception ──
                    # If the selected policy contains a propose_novel_action step,
                    # fire the callback (which emits NOVEL_ACTION_REQUESTED) and
                    # continue to the next policy - the current cycle takes
                    # do-nothing while Simula generates the executor asynchronously.
                    novel_step = next(
                        (s for s in policy.steps if s.action_type == "propose_novel_action"),
                        None,
                    )
                    if novel_step is not None:
                        if self._novel_action_cb is not None:
                            try:
                                asyncio.ensure_future(
                                    self._novel_action_cb(goal, novel_step.parameters)
                                )
                                self._logger.info(
                                    "propose_novel_action_intercepted",
                                    policy=policy.name,
                                    action_name=novel_step.parameters.get("action_name", ""),
                                )
                            except Exception as exc:
                                self._logger.warning(
                                    "novel_action_cb_fire_failed", error=str(exc)
                                )
                        continue  # Do not submit this policy to Equor; fall through

                    intent = _policy_to_intent(
                        policy,
                        goal,
                        path="slow",
                        confidence=efe_score.confidence,
                        efe_score=efe_score,
                        all_efe_scores={p.name: e.total for p, e in scored},
                    )

                    try:
                        check = await asyncio.wait_for(
                            self._equor.review(intent),
                            timeout=0.6,  # 600ms - generous but bounded
                        )
                    except (TimeoutError, asyncio.TimeoutError, OSError, ConnectionError) as exc:
                        reason = f"equor_slow_path_unavailable: {exc!s}"
                        self._logger.warning("equor_unavailable_slow_path", reason=reason)
                        if self._equor_failure_cb is not None:
                            try:
                                self._equor_failure_cb(reason)
                            except Exception:
                                pass
                        return None, [], cognition_budget  # do-nothing fallback

                    if check.verdict == Verdict.APPROVED:
                        self._last_equor_check = check
                        # Collect rejected: all non-selected, non-do-nothing policies
                        rejected = [
                            (p, s) for p, s in scored
                            if p.id != policy.id and p.id != "do_nothing"
                        ]
                        return intent, rejected, cognition_budget
                    elif check.verdict == Verdict.MODIFIED:
                        self._last_equor_check = check
                        rejected = [
                            (p, s) for p, s in scored
                            if p.id != policy.id and p.id != "do_nothing"
                        ]
                        return (
                            _apply_modifications(intent, check.modifications),
                            rejected,
                            cognition_budget,
                        )
                    elif check.verdict == Verdict.DEFERRED:
                        self._logger.info("intent_deferred_by_equor", intent_id=intent.id)
                        return None, [], cognition_budget  # Governance will handle
                    # BLOCKED → try next policy
                    self._logger.info(
                        "policy_blocked_by_equor_trying_next",
                        policy=policy.name,
                        reasoning=check.reasoning[:80],
                    )
                    # Yield between retries so we don't hold the loop
                    await asyncio.sleep(0)

                return None, [], cognition_budget  # All policies blocked

        except TimeoutError:
            self._logger.warning("slow_path_timeout")
            return None, [], cognition_budget
        except Exception as exc:
            self._logger.error("slow_path_error", error=str(exc))
            return None, [], cognition_budget


# ─── Intent Construction ─────────────────────────────────────────


def _policy_to_intent(
    policy: Policy,
    goal: Goal,
    path: str,
    confidence: float = 0.7,
    efe_score: EFEScore | None = None,
    all_efe_scores: dict[str, float] | None = None,
) -> Intent:
    """
    Convert a selected Policy into a formal Intent primitive.
    Intents are the cross-system communication unit; Policies are Nova-internal.
    """
    actions = [
        Action(
            executor=f"executor.{step.action_type}",
            parameters={
                "description": step.description,
                **step.parameters,
            },
            timeout_ms=step.expected_duration_ms * 2,
        )
        for step in policy.steps
    ]

    efe_reasoning = ""
    if efe_score:
        efe_reasoning = efe_score.reasoning

    trace_alternatives = []
    if all_efe_scores:
        trace_alternatives = [{"policy": name, "efe": efe} for name, efe in all_efe_scores.items()]

    return Intent(
        id=new_id(),
        goal=GoalDescriptor(
            description=goal.description,
            target_domain=goal.target_domain,
            success_criteria={"criteria": goal.success_criteria} if goal.success_criteria else {},
        ),
        plan=ActionSequence(steps=actions),
        expected_free_energy=efe_score.total if efe_score else 0.0,
        priority=goal.priority,
        decision_trace=DecisionTrace(
            reasoning=(
                f"Path: {path}. Policy: {policy.name}. {policy.reasoning[:200]}. "
                f"EFE evaluation: {efe_reasoning}"
            ),
            alternatives_considered=trace_alternatives,
            free_energy_scores=all_efe_scores or {},
        ),
    )


def _apply_modifications(intent: Intent, modifications: list[str]) -> Intent:
    """Apply Equor's suggested modifications to an intent."""
    existing_reasoning = intent.decision_trace.reasoning
    modification_notes = "; ".join(modifications[:5])
    updated_trace = intent.decision_trace.model_copy(
        update={
            "reasoning": f"{existing_reasoning} | Equor modifications: {modification_notes}"
        }
    )
    return intent.model_copy(update={"decision_trace": updated_trace})


def _extract_situation_summary(broadcast: WorkspaceBroadcast) -> str:
    """Extract a brief situation summary from a broadcast for policy generation."""
    content = broadcast.content
    paths = [
        ("content", "content"),
        ("content",),
    ]
    for path in paths:
        obj: object = content
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, str) and obj:
            return obj[:400]
    return f"Broadcast from workspace (salience: {broadcast.salience.composite:.2f})"
