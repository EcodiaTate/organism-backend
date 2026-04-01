"""
EcodiaOS -- Adversarial Self-Play Orchestrator (Stage 6B.3)

Closed-loop system that autonomously hardens the organism's constitution:

  1. AttackGenerator crafts adversarial Intents via the local LLM
  2. RedTeamInstance runs them through a sandboxed Equor gate
  3. Successful bypasses are analyzed for root cause
  4. EvolutionProposals are emitted to simula/proposals/ with
     GOVERNANCE_REQUIRED tag, awaiting human approval before
     merging into the live constitution

Safety guarantees:
  - Runs entirely in the coevolution/ sandbox
  - Cannot touch the live Synapse queue, Neo4j, or Redis
  - Proposals require governance approval before any live effect
  - Background-only execution at low priority
  - All attacks are synthetic - never executed against real infrastructure
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from primitives.common import new_id
from systems.simula.coevolution.adversarial_types import (
    AdversarialSelfPlayResult,
    BypassTrace,
    SelfPlayConfig,
)
from systems.simula.coevolution.attack_generator import AttackGenerator
from systems.simula.coevolution.red_team import RedTeamInstance
from systems.simula.evolution_types import (
    ChangeCategory,
    ChangeSpec,
    EvolutionProposal,
    ProposalStatus,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider

logger = structlog.get_logger().bind(system="simula.coevolution.adversarial_self_play")


class AdversarialSelfPlay:
    """
    Closed-loop adversarial hardening of the Equor constitutional gate.

    Orchestrates the attack → evaluate → analyze → propose cycle.
    Designed to run as a low-priority background task via asyncio.
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: SelfPlayConfig | None = None,
        *,
        proposal_sink: list[EvolutionProposal] | None = None,
    ) -> None:
        self._config = config or SelfPlayConfig()
        self._sandbox = RedTeamInstance(self._config)
        self._attacker = AttackGenerator(llm, self._config)
        # Proposals are appended here. The caller (SimulaService) drains
        # this list and routes them into the evolution pipeline.
        # Default to an internal list if no external sink is provided.
        self._proposal_sink: list[EvolutionProposal] = (
            proposal_sink if proposal_sink is not None else []
        )
        self._bypass_history: list[BypassTrace] = []
        self._bypass_history_max: int = 500  # cap to prevent unbounded memory growth
        self._running = False
        self._cycle_count = 0

    # ── Public API ──────────────────────────────────────────────────────────

    async def run_cycle(self) -> AdversarialSelfPlayResult:
        """
        Execute one complete adversarial self-play cycle:

        1. Generate adversarial attacks
        2. Run them through the sandboxed Equor gate
        3. Analyze any successful bypasses
        4. Emit governance-gated EvolutionProposals for each bypass
        """
        start = time.monotonic()
        cycle_id = new_id()
        self._cycle_count += 1

        logger.info(
            "adversarial_cycle_start",
            cycle_id=cycle_id,
            cycle_number=self._cycle_count,
        )

        errors: list[str] = []

        # Phase 1: Generate attacks
        try:
            vectors = await self._attacker.generate_attacks(
                prior_bypasses=self._bypass_history[-10:],  # feed recent history
            )
        except Exception as exc:
            errors.append(f"Attack generation failed: {exc}")
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return AdversarialSelfPlayResult(
                cycle_id=cycle_id,
                duration_ms=elapsed_ms,
                errors=errors,
            )

        if not vectors:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning("adversarial_cycle_no_vectors", cycle_id=cycle_id)
            return AdversarialSelfPlayResult(
                cycle_id=cycle_id,
                duration_ms=elapsed_ms,
            )

        # Phase 2: Execute attacks through sandbox
        results = []
        for vector in vectors:
            try:
                result = await asyncio.wait_for(
                    self._sandbox.review_attack(vector),
                    timeout=self._config.attack_timeout_s,
                )
                results.append(result)
            except TimeoutError:
                errors.append(
                    f"Attack timeout: {vector.category}/{vector.goal_text[:50]}"
                )
            except Exception as exc:
                errors.append(f"Attack execution error: {exc}")

        # Phase 3: Identify bypasses and analyze
        bypasses: list[BypassTrace] = []
        for result in results:
            if result.bypassed:
                try:
                    trace = await self._attacker.analyze_bypass(result)
                    bypasses.append(trace)
                    self._bypass_history.append(trace)
                    # Evict oldest entries to keep memory bounded
                    if len(self._bypass_history) > self._bypass_history_max:
                        del self._bypass_history[: len(self._bypass_history) - self._bypass_history_max]
                except Exception as exc:
                    errors.append(f"Bypass analysis error: {exc}")

        # Phase 4: Generate proposals for bypasses
        proposals_emitted = 0
        proposals_cap = self._config.max_proposals_per_cycle
        for trace in bypasses:
            if proposals_emitted >= proposals_cap:
                break
            proposal = self._trace_to_proposal(trace, cycle_id)
            self._proposal_sink.append(proposal)
            trace.proposal_id = proposal.id
            proposals_emitted += 1

            logger.info(
                "adversarial_proposal_emitted",
                proposal_id=proposal.id,
                severity=trace.severity,
                failure_stage=trace.failure_stage,
                category=trace.attack.vector.category,
            )

        # Aggregate stats
        elapsed_ms = int((time.monotonic() - start) * 1000)
        bypasses_by_cat: dict[str, int] = {}
        bypasses_by_sev: dict[str, int] = {}
        for b in bypasses:
            cat = b.attack.vector.category.value
            bypasses_by_cat[cat] = bypasses_by_cat.get(cat, 0) + 1
            sev = b.severity.value
            bypasses_by_sev[sev] = bypasses_by_sev.get(sev, 0) + 1

        result_obj = AdversarialSelfPlayResult(
            cycle_id=cycle_id,
            attacks_generated=len(vectors),
            attacks_executed=len(results),
            bypasses_found=len(bypasses),
            bypasses_by_category=bypasses_by_cat,
            bypasses_by_severity=bypasses_by_sev,
            proposals_emitted=proposals_emitted,
            duration_ms=elapsed_ms,
            errors=errors,
        )

        logger.info(
            "adversarial_cycle_complete",
            cycle_id=cycle_id,
            attacks=len(vectors),
            executed=len(results),
            bypasses=len(bypasses),
            proposals=proposals_emitted,
            duration_ms=elapsed_ms,
            errors_count=len(errors),
        )

        return result_obj

    async def run_background_loop(self) -> None:
        """
        Run adversarial self-play continuously as a background task.

        Sleeps for config.cycle_cooldown_s between cycles. Cancel the
        task to stop the loop.
        """
        self._running = True
        logger.info("adversarial_background_loop_started")

        try:
            while self._running:
                try:
                    await self.run_cycle()
                except Exception as exc:
                    logger.error(
                        "adversarial_cycle_unhandled_error",
                        error=str(exc),
                    )

                # Low-priority cooldown between cycles
                await asyncio.sleep(self._config.cycle_cooldown_s)
        finally:
            self._running = False
            logger.info("adversarial_background_loop_stopped")

    def stop(self) -> None:
        """Signal the background loop to stop after the current cycle."""
        self._running = False

    @property
    def bypass_history(self) -> list[BypassTrace]:
        """Read-only access to the accumulated bypass history."""
        return list(self._bypass_history)

    @property
    def pending_proposals(self) -> list[EvolutionProposal]:
        """Read-only access to proposals waiting to be drained."""
        return list(self._proposal_sink)

    def drain_proposals(self) -> list[EvolutionProposal]:
        """
        Return and clear all pending proposals.

        Called by SimulaService to ingest proposals into the evolution pipeline.
        """
        proposals = list(self._proposal_sink)
        self._proposal_sink.clear()
        return proposals

    # ── Private ─────────────────────────────────────────────────────────────

    def _trace_to_proposal(
        self,
        trace: BypassTrace,
        cycle_id: str,
    ) -> EvolutionProposal:
        """
        Convert a BypassTrace into a GOVERNANCE_REQUIRED EvolutionProposal.

        The proposal targets MODIFY_CONTRACT (the constitutional gate's
        filtering rules), which is in the GOVERNANCE_REQUIRED set -
        ensuring human review before any live change.
        """
        severity_label = trace.severity.value.upper()
        category_label = trace.attack.vector.category.value

        description = (
            f"[ADVERSARIAL-{severity_label}] Constitutional gate bypass detected "
            f"via {category_label} attack.\n\n"
            f"Failure stage: {trace.failure_stage}\n"
            f"Root cause: {trace.root_cause}\n"
            f"Attack goal: {trace.attack.vector.goal_text[:200]}\n"
            f"Gate reasoning: {trace.attack.reasoning[:200]}"
        )

        change_spec = ChangeSpec(
            contract_changes=[trace.suggested_fix] if trace.suggested_fix else [],
            affected_systems=["equor"],
            additional_context=(
                f"Adversarial self-play cycle {cycle_id}. "
                f"Attack category: {category_label}. "
                f"Bypass severity: {severity_label}. "
                f"Failure stage: {trace.failure_stage}."
            ),
            code_hint=trace.suggested_fix,
        )

        return EvolutionProposal(
            source="adversarial_self_play",
            category=ChangeCategory.MODIFY_CONTRACT,
            description=description,
            change_spec=change_spec,
            evidence=[cycle_id, trace.id],
            expected_benefit=(
                f"Patch {trace.failure_stage} vulnerability: {trace.root_cause[:100]}"
            ),
            risk_assessment=(
                f"Severity: {severity_label}. Requires governance review "
                f"before modifying live Equor configuration."
            ),
            status=ProposalStatus.AWAITING_GOVERNANCE,
        )
