"""
EcodiaOS - Self-Modification Pipeline
Spec 10 §SM - Recursive Self-Improvement Layer.

Orchestrates the full self-modification workflow triggered when CapabilityAuditor
identifies a gap (CAPABILITY_GAP_IDENTIFIED event):

  Step 1: Nova deliberates - is filling this gap aligned with our drives?
          EFE scoring against the four constitutional drives.  If below threshold,
          emit no proposal.

  Step 2: Emit SELF_MODIFICATION_PROPOSED to Synapse bus.
          Equor subscribes and reviews the proposal constitutionally.

  Step 3: Await EQUOR_ECONOMIC_PERMIT or EQUOR_ECONOMIC_DENY (30s timeout).
          Denied → log + return.  Timed-out → auto-permit as safety fallback (logged
          at WARNING to avoid blocking the organism on Equor unavailability).

  Step 4: If proposal requires a new Python dependency, install it via
          HotDeployment.install_dependency() (which runs its own safety check).

  Step 5: Simula CodeAgent generates the executor code for the proposed action type.
          Uses the existing NOVEL_ACTION_REQUESTED / NOVEL_ACTION_CREATED flow so
          the full Simula pipeline (AST check, Iron Rules, Z3 if risk=high) runs.

  Step 6: HotDeployment.deploy_executor() - writes, imports, registers in the
          live ExecutorRegistry.  Neo4j (:SelfModification) node created.

  Step 7: Nova queues a low-stakes test goal using the new executor.
          If the test goal succeeds within TEST_TIMEOUT_S:
            → executor is permanent; Evo gets a positive signal
          If the test goal fails:
            → HotDeployment.rollback_executor()
            → Emit EXECUTOR_REVERTED

  Step 8: Emit RE_TRAINING_EXAMPLE for every outcome (success or failure).
          This trains the RE to generate better executors over time.

  Step 9 (optional): If gap requires a full new subsystem (complexity=high AND
          no_executor_possible), draft a Spec document via _draft_spec() and emit
          SPEC_DRAFTED for Equor review.

Constitutional constraints:
  - All proposals go through Equor constitutional review (step 3)
  - All modifications tracked in Neo4j with full audit trail
  - All modifications reversible via HotDeployment.rollback_executor()
  - The organism may propose modification to any subsystem including core ones;
    Equor and governance decide whether to approve.
"""

from __future__ import annotations

import asyncio
import textwrap
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now_str
from primitives.re_training import RETrainingExample

if TYPE_CHECKING:
    from core.hot_deploy import HotDeployment
    from systems.equor.service import EquorService
    from systems.simula import SimulaService
    from systems.synapse.service import SynapseService

logger = structlog.get_logger().bind(system="nova.self_modification_pipeline")

# ── Configuration ──────────────────────────────────────────────────────────────

_EQUOR_TIMEOUT_S: float = 30.0
_TEST_GOAL_TIMEOUT_S: float = 300.0   # 5 min for the live test goal


@dataclass
class _PipelineState:
    gap_id: str
    proposal_id: str
    action_type: str
    description: str
    blocking_goal_count: int
    estimated_value_usdc: Decimal
    implementation_complexity: str
    requires_external_dependency: bool
    dependency_package: str | None
    source_events: list[str]
    equor_permit_event: asyncio.Event
    equor_denied: bool
    equor_approval_id: str
    generated_code: str
    deployment_id: str
    test_goal_id: str
    success: bool
    failure_reason: str


class SelfModificationPipeline:
    """
    Orchestrates the gap → proposal → Equor → generate → deploy → test → train cycle.

    Usage:
        pipeline = SelfModificationPipeline()
        pipeline.set_synapse(synapse)
        pipeline.set_equor(equor)
        pipeline.set_simula(simula)
        pipeline.set_hot_deploy(hot_deploy)
        pipeline.attach()   # subscribes to CAPABILITY_GAP_IDENTIFIED + Equor responses
    """

    def __init__(self) -> None:
        self._synapse: SynapseService | None = None
        self._equor: EquorService | None = None
        self._simula: SimulaService | None = None
        self._hot_deploy: HotDeployment | None = None

        # Pending proposals waiting for Equor response
        # proposal_id → _PipelineState
        self._pending: dict[str, _PipelineState] = {}
        # Test goals in flight: test_goal_id → proposal_id
        self._test_goals: dict[str, str] = {}

        self._attached: bool = False

    # ── Dependency injection ──────────────────────────────────────────────────

    def set_synapse(self, synapse: SynapseService) -> None:
        self._synapse = synapse

    def set_equor(self, equor: EquorService) -> None:
        self._equor = equor

    def set_simula(self, simula: SimulaService) -> None:
        self._simula = simula

    def set_hot_deploy(self, hot_deploy: HotDeployment) -> None:
        self._hot_deploy = hot_deploy

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def attach(self) -> None:
        if self._synapse is None or self._attached:
            return
        bus = self._synapse.event_bus
        bus.subscribe("capability_gap_identified", self._on_capability_gap)
        bus.subscribe("equor_economic_permit", self._on_equor_permit)
        bus.subscribe("equor_economic_deny", self._on_equor_deny)
        bus.subscribe("intent_outcome", self._on_intent_outcome)
        bus.subscribe("executor_reverted", self._on_executor_reverted)
        self._attached = True
        logger.info("self_modification_pipeline.attached")

    def detach(self) -> None:
        if self._synapse is None or not self._attached:
            return
        bus = self._synapse.event_bus
        for evt, handler in [
            ("capability_gap_identified", self._on_capability_gap),
            ("equor_economic_permit", self._on_equor_permit),
            ("equor_economic_deny", self._on_equor_deny),
            ("intent_outcome", self._on_intent_outcome),
            ("executor_reverted", self._on_executor_reverted),
        ]:
            try:
                bus.unsubscribe(evt, handler)
            except Exception:
                pass
        self._attached = False

    # ── Step 1: Receive gap ───────────────────────────────────────────────────

    def _on_capability_gap(self, event: Any) -> None:
        asyncio.ensure_future(self._handle_gap(event.data))

    async def _handle_gap(self, data: dict[str, Any]) -> None:
        action_type: str = data.get("proposed_action_type", "").strip()
        description: str = data.get("description", action_type)
        gap_id: str = data.get("gap_id", new_id())

        try:
            value = Decimal(str(data.get("estimated_value_usdc", "0")))
        except Exception:
            value = Decimal("0")

        complexity: str = data.get("implementation_complexity", "low")
        requires_dep: bool = bool(data.get("requires_external_dependency", False))
        dep_pkg: str | None = data.get("dependency_package")
        blocking_count: int = int(data.get("blocking_goal_count", 0))
        source_events: list[str] = data.get("source_events", [])

        # Step 1: Nova deliberation - LLM evaluates drive alignment and EFE
        drive_alignment, efe_score, should_proceed = await self._evaluate_gap_with_llm(
            action_type=action_type,
            description=description,
            value=value,
            complexity=complexity,
            blocking_count=blocking_count,
            gap_id=gap_id,
            source_events=source_events,
        )

        if not should_proceed:
            logger.info(
                "self_modification.gap_not_worth_pursuing",
                gap_id=gap_id,
                efe_score=round(efe_score, 3),
                reasoning=drive_alignment.get("reasoning", ""),
            )
            return

        # Step 2: Emit SELF_MODIFICATION_PROPOSED
        proposal_id = new_id()
        state = _PipelineState(
            gap_id=gap_id,
            proposal_id=proposal_id,
            action_type=action_type,
            description=description,
            blocking_goal_count=blocking_count,
            estimated_value_usdc=value,
            implementation_complexity=complexity,
            requires_external_dependency=requires_dep,
            dependency_package=dep_pkg,
            source_events=source_events,
            equor_permit_event=asyncio.Event(),
            equor_denied=False,
            equor_approval_id="",
            generated_code="",
            deployment_id="",
            test_goal_id="",
            success=False,
            failure_reason="",
        )
        self._pending[proposal_id] = state

        await self._emit_proposal(state, drive_alignment)

        # Step 3: Await Equor (with timeout)
        try:
            await asyncio.wait_for(state.equor_permit_event.wait(), timeout=_EQUOR_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "self_modification.equor_timeout_auto_permit",
                proposal_id=proposal_id,
                action_type=action_type,
            )
            state.equor_approval_id = "auto_permit_timeout"

        if state.equor_denied:
            logger.info(
                "self_modification.equor_denied",
                proposal_id=proposal_id,
                action_type=action_type,
            )
            self._pending.pop(proposal_id, None)
            await self._emit_re_training(state, outcome="equor_denied", success=False)
            return

        # Step 4: Install dependency if needed
        if requires_dep and dep_pkg and self._hot_deploy is not None:
            installed = await self._hot_deploy.install_dependency(dep_pkg, proposal_id)
            if not installed:
                logger.error(
                    "self_modification.dependency_install_failed",
                    proposal_id=proposal_id,
                    package=dep_pkg,
                )
                self._pending.pop(proposal_id, None)
                await self._emit_re_training(state, outcome="dependency_failed", success=False)
                return

        # Special path: complexity=high with no executor possible → draft Spec
        if complexity == "high" and self._is_subsystem_needed(description):
            await self._draft_spec(state)
            self._pending.pop(proposal_id, None)
            return

        # Step 5: Simula generates executor code
        code = await self._generate_executor_code(state)
        if not code:
            logger.error(
                "self_modification.code_generation_failed",
                proposal_id=proposal_id,
                action_type=action_type,
            )
            self._pending.pop(proposal_id, None)
            await self._emit_re_training(state, outcome="code_generation_failed", success=False)
            return
        state.generated_code = code

        # Step 6: Hot-deploy
        if self._hot_deploy is None:
            logger.error("self_modification.hot_deploy_not_wired", proposal_id=proposal_id)
            self._pending.pop(proposal_id, None)
            return

        record = await self._hot_deploy.deploy_executor(
            code=code,
            action_type=action_type,
            proposal_id=proposal_id,
            equor_approval_id=state.equor_approval_id,
        )

        if not record.success:
            logger.error(
                "self_modification.deploy_failed",
                proposal_id=proposal_id,
                error=record.error,
            )
            self._pending.pop(proposal_id, None)
            state.failure_reason = record.error or "deploy_failed"
            await self._emit_re_training(state, outcome="deploy_failed", success=False)
            return

        state.deployment_id = record.deployment_id
        state.test_goal_id = record.test_goal_id
        self._test_goals[record.test_goal_id] = proposal_id

        # Step 7: Queue test goal with timeout monitoring
        asyncio.ensure_future(self._monitor_test_goal(state))

    # ── Step 3: Equor responses ───────────────────────────────────────────────

    def _on_equor_permit(self, event: Any) -> None:
        proposal_id: str = event.data.get("proposal_id", "")
        action: str = event.data.get("action", "")
        if action != "novel_action_proposal":
            return
        state = self._pending.get(proposal_id)
        if state is None:
            return
        state.equor_approval_id = event.data.get("approval_id", new_id())
        state.equor_permit_event.set()

    def _on_equor_deny(self, event: Any) -> None:
        proposal_id: str = event.data.get("proposal_id", "")
        action: str = event.data.get("action", "")
        if action != "novel_action_proposal":
            return
        state = self._pending.get(proposal_id)
        if state is None:
            return
        state.equor_denied = True
        state.equor_permit_event.set()

    # ── Step 7: Test goal monitoring ─────────────────────────────────────────

    def _on_intent_outcome(self, event: Any) -> None:
        intent_id: str = event.data.get("intent_id", "")
        goal_id: str = event.data.get("goal_id", "")
        check_id = intent_id or goal_id
        proposal_id = self._test_goals.get(check_id)
        if proposal_id is None:
            return
        state = self._pending.get(proposal_id)
        if state is None:
            return
        success: bool = bool(event.data.get("success", False))
        state.success = success
        if not success:
            state.failure_reason = event.data.get("failure_reason", "test_goal_failed")
        # Signal the monitoring task
        state.equor_permit_event.set()  # reuse as a general signal flag

    async def _monitor_test_goal(self, state: _PipelineState) -> None:
        """Wait for test goal outcome; rollback on failure or timeout."""
        test_result_event = asyncio.Event()

        def _on_result(event: Any) -> None:
            intent_id = event.data.get("intent_id", "")
            goal_id = event.data.get("goal_id", "")
            if state.test_goal_id in (intent_id, goal_id):
                state.success = bool(event.data.get("success", False))
                if not state.success:
                    state.failure_reason = event.data.get("failure_reason", "test_goal_failed")
                test_result_event.set()

        bus = self._synapse.event_bus if self._synapse else None
        if bus:
            bus.subscribe("intent_outcome", _on_result)

        # Emit a low-stakes test goal via Nova - request deliberation
        await self._emit_test_goal_request(state)

        try:
            await asyncio.wait_for(test_result_event.wait(), timeout=_TEST_GOAL_TIMEOUT_S)
        except asyncio.TimeoutError:
            state.success = False
            state.failure_reason = "test_goal_timeout"

        if bus:
            try:
                bus.unsubscribe("intent_outcome", _on_result)
            except Exception:
                pass

        # Step 8: RE training example (always)
        outcome = "deployed_and_verified" if state.success else "deployed_then_rolled_back"
        await self._emit_re_training(state, outcome=outcome, success=state.success)

        if not state.success:
            # Rollback
            if self._hot_deploy is not None and state.deployment_id:
                await self._hot_deploy.rollback_executor(
                    action_type=state.action_type,
                    deployment_id=state.deployment_id,
                    reason="test_failed",
                    failure_details=state.failure_reason,
                )
            logger.info(
                "self_modification.rolled_back",
                proposal_id=state.proposal_id,
                action_type=state.action_type,
                reason=state.failure_reason,
            )
        else:
            logger.info(
                "self_modification.capability_added",
                proposal_id=state.proposal_id,
                action_type=state.action_type,
                value_usdc=str(state.estimated_value_usdc),
            )

        self._pending.pop(state.proposal_id, None)
        self._test_goals.pop(state.test_goal_id, None)

    def _on_executor_reverted(self, event: Any) -> None:
        # External revert (Thymos) - clean up our state
        deployment_id: str = event.data.get("deployment_id", "")
        for pid, state in list(self._pending.items()):
            if state.deployment_id == deployment_id:
                state.success = False
                state.failure_reason = event.data.get("reason", "external_revert")
                asyncio.ensure_future(
                    self._emit_re_training(state, outcome="thymos_reverted", success=False)
                )
                self._pending.pop(pid, None)
                break

    # ── Code generation ──────────────────────────────────────────────────────

    async def _generate_executor_code(self, state: _PipelineState) -> str:
        """
        Ask Simula to generate executor code by emitting NOVEL_ACTION_REQUESTED
        and awaiting NOVEL_ACTION_CREATED with a matching proposal_id.
        """
        if self._synapse is None:
            return ""

        result_event = asyncio.Event()
        generated_code: list[str] = []

        def _on_created(event: Any) -> None:
            if event.data.get("proposal_id") == state.proposal_id:
                code = event.data.get("executor_code", "")
                if code:
                    generated_code.append(code)
                result_event.set()

        bus = self._synapse.event_bus
        bus.subscribe("novel_action_created", _on_created)

        from systems.synapse.types import SynapseEventType
        try:
            await bus.emit(
                SynapseEventType.NOVEL_ACTION_REQUESTED,
                {
                    "proposal_id": state.proposal_id,
                    "action_name": state.action_type,
                    "description": state.description,
                    "required_capabilities": self._infer_capabilities(state),
                    "expected_outcome": (
                        f"Executor for '{state.action_type}' that enables "
                        f"{state.blocking_goal_count} blocked goals"
                    ),
                    "justification": (
                        f"Self-modification pipeline: gap '{state.gap_id}' - "
                        f"estimated value ${state.estimated_value_usdc}/cycle"
                    ),
                    "goal_id": state.gap_id,
                    "goal_description": state.description,
                    "urgency": min(0.9, 0.4 + float(state.estimated_value_usdc) / 100),
                    "proposed_by": "self_modification_pipeline",
                    "proposed_at": utc_now_str(),
                    # Flag: pipeline will handle HotDeployment itself; Simula should
                    # return code in NOVEL_ACTION_CREATED rather than auto-deploying
                    "pipeline_managed": True,
                },
                source_system="nova.self_modification_pipeline",
            )
        except Exception as exc:
            logger.warning("self_modification.emit_novel_action_failed", error=str(exc))
            bus.unsubscribe("novel_action_created", _on_created)
            return ""

        try:
            await asyncio.wait_for(result_event.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            logger.warning(
                "self_modification.code_generation_timeout",
                proposal_id=state.proposal_id,
            )

        try:
            bus.unsubscribe("novel_action_created", _on_created)
        except Exception:
            pass

        return generated_code[0] if generated_code else ""

    # ── Spec drafting (step 9) ────────────────────────────────────────────────

    async def _draft_spec(self, state: _PipelineState) -> None:
        """Draft a Spec document for a gap that requires a full new subsystem."""
        import hashlib
        from pathlib import Path

        spec_id = new_id()
        system_name = state.action_type.replace("_", " ").title().replace(" ", "")
        drafted_at = utc_now_str()

        # Simple template - Simula can elaborate later
        spec_content = textwrap.dedent(f"""\
            # EcodiaOS - Spec DRAFT - {system_name}
            *Auto-drafted by SelfModificationPipeline on {drafted_at}*
            *Gap ID:* `{state.gap_id}`
            *Proposal ID:* `{state.proposal_id}`

            ---

            ## Purpose
            {state.description}

            ## Motivation
            - **Blocking goal count:** {state.blocking_goal_count}
            - **Estimated value:** ${state.estimated_value_usdc}/cycle
            - **Implementation complexity:** {state.implementation_complexity}

            ## Proposed Capability
            Action type: `{state.action_type}`

            The organism requires a new subsystem to provide this capability because
            an Axon executor alone is insufficient (complexity={state.implementation_complexity}).

            ## High-Level Design
            TBD - Simula CodeAgent will elaborate after Equor approval.

            ## Constitutional Alignment
            This Spec must be reviewed by Equor before Simula implements it.
            All four drives (Coherence, Care, Growth, Honesty) must score ≥ 0.0.

            ## Integration Points
            - **Synapse events emitted:** TBD
            - **Synapse events consumed:** TBD
            - **Memory reads:** TBD
            - **Dependencies:** {state.dependency_package or "none"}

            ## Iron Rules
            - Cannot import from systems.*
            - Must communicate via Synapse bus
            - Must implement initialize(), shutdown(), health()

            ## Open Questions
            1. What is the minimum viable implementation?
            2. Which existing systems does this most closely resemble?
            3. What are the failure modes?
        """)

        spec_dir = Path(__file__).parent.parent.parent / ".claude"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_filename = f"EcodiaOS_Spec_DRAFT_{system_name}.md"
        spec_path = spec_dir / spec_filename
        try:
            spec_path.write_text(spec_content, encoding="utf-8")
        except OSError as exc:
            logger.warning("self_modification.spec_write_failed", error=str(exc))
            return

        spec_hash = hashlib.sha256(spec_content.encode()).hexdigest()
        rel_path = str(spec_path.relative_to(Path(__file__).parent.parent.parent))

        if self._synapse is not None:
            from systems.synapse.types import SynapseEventType
            try:
                await self._synapse.event_bus.emit(
                    SynapseEventType.SPEC_DRAFTED,
                    {
                        "spec_id": spec_id,
                        "proposal_id": state.proposal_id,
                        "spec_title": f"Spec DRAFT - {system_name}",
                        "spec_path": rel_path,
                        "system_name": system_name,
                        "spec_hash": spec_hash,
                        "drafted_at": drafted_at,
                    },
                    source_system="nova.self_modification_pipeline",
                    salience=0.7,
                )
            except Exception as exc:
                logger.warning("self_modification.emit_spec_drafted_failed", error=str(exc))

        logger.info(
            "self_modification.spec_drafted",
            spec_id=spec_id,
            system_name=system_name,
            spec_path=rel_path,
        )

    # ── RE Training ──────────────────────────────────────────────────────────

    async def _emit_re_training(
        self,
        state: _PipelineState,
        outcome: str,
        success: bool,
    ) -> None:
        """Emit a RE_TRAINING_EXAMPLE for every pipeline outcome."""
        if self._synapse is None:
            return
        from systems.synapse.types import SynapseEventType

        example = RETrainingExample(
            episode_id=state.proposal_id,
            category="self_modification",
            prompt=(
                f"Capability gap detected: {state.description}\n"
                f"Blocking goals: {state.blocking_goal_count}\n"
                f"Estimated value: ${state.estimated_value_usdc}"
            ),
            reasoning_trace=(
                f"Pipeline outcome: {outcome}\n"
                f"Action type proposed: {state.action_type}\n"
                f"Complexity: {state.implementation_complexity}\n"
                f"Requires dependency: {state.requires_external_dependency}\n"
                f"Dependency package: {state.dependency_package}\n"
                f"Code generated: {bool(state.generated_code)}\n"
                f"Deployed: {bool(state.deployment_id)}\n"
                f"Test passed: {success}"
            ),
            response=outcome,
            confidence=0.9 if success else 0.3,
            source_system="nova.self_modification_pipeline",
        )
        try:
            await self._synapse.event_bus.emit(
                SynapseEventType.RE_TRAINING_EXAMPLE,
                example.model_dump(),
                source_system="nova.self_modification_pipeline",
            )
        except Exception as exc:
            logger.warning("self_modification.re_training_emit_failed", error=str(exc))

    # ── Emissions ────────────────────────────────────────────────────────────

    async def _emit_proposal(
        self, state: _PipelineState, drive_alignment: dict[str, float]
    ) -> None:
        if self._synapse is None:
            return
        from systems.synapse.types import SynapseEventType
        try:
            await self._synapse.event_bus.emit(
                SynapseEventType.SELF_MODIFICATION_PROPOSED,
                {
                    "proposal_id": state.proposal_id,
                    "gap_id": state.gap_id,
                    "description": state.description,
                    "proposed_action_type": state.action_type,
                    "implementation_complexity": state.implementation_complexity,
                    "requires_external_dependency": state.requires_external_dependency,
                    "dependency_package": state.dependency_package,
                    "estimated_value_usdc": str(state.estimated_value_usdc),
                    "drive_alignment": drive_alignment,
                    "proposed_by": "nova",
                    "proposed_at": utc_now_str(),
                },
                source_system="nova.self_modification_pipeline",
                salience=0.75,
            )
        except Exception as exc:
            logger.warning("self_modification.emit_proposal_failed", error=str(exc))

    async def _emit_test_goal_request(self, state: _PipelineState) -> None:
        """
        Ask Nova to deliberate a low-stakes test using the new executor.
        Uses NOVA_INTENT_REQUESTED so any system can trigger Nova deliberation.
        """
        if self._synapse is None:
            return
        from systems.synapse.types import SynapseEventType
        try:
            await self._synapse.event_bus.emit(
                SynapseEventType.NOVA_INTENT_REQUESTED,
                {
                    "request_id": state.test_goal_id,
                    "context": (
                        f"Self-modification test: verify new executor '{state.action_type}' works. "
                        f"Use action_type='{state.action_type}' with a minimal, reversible, "
                        f"low-stakes test scenario. This is an automated capability verification."
                    ),
                    "goal_description": f"Verify new capability: {state.action_type}",
                    "urgency": 0.6,
                    "source_system": "nova.self_modification_pipeline",
                    "test_mode": True,
                    "preferred_action_type": state.action_type,
                },
                source_system="nova.self_modification_pipeline",
            )
        except Exception as exc:
            logger.warning("self_modification.emit_test_goal_failed", error=str(exc))

    # ── LLM-based Gap Evaluation ──────────────────────────────────────────────

    async def _evaluate_gap_with_llm(
        self,
        action_type: str,
        description: str,
        value: Decimal,
        complexity: str,
        blocking_count: int,
        gap_id: str,
        source_events: list[str],
    ) -> tuple[dict[str, Any], float, bool]:
        """
        Ask the LLM to evaluate whether this capability gap is worth pursuing.

        Returns (drive_alignment_dict, efe_score, should_proceed).
        Falls back to pursuing the gap if LLM is unavailable (organism is not
        blocked by infrastructure failures).
        """
        if self._simula is None:
            # No LLM available — proceed; Equor will review constitutionally
            return {"coherence": 0.5, "care": 0.5, "growth": 0.5, "honesty": 0.5, "reasoning": "llm_unavailable"}, 0.5, True

        prompt = f"""Capability gap evaluation.

Evaluate whether pursuing this gap aligns with the organism's four constitutional drives and is worth acting on.

Gap ID: {gap_id}
Proposed action type: {action_type}
Description: {description}
Estimated value: ${value} per cycle
Implementation complexity: {complexity}
Goals currently blocked by this gap: {blocking_count}
Source events: {source_events}

Assess the four drives (each -1.0 to 1.0):
- Coherence: Does adding this capability help the organism act more coherently and reliably?
- Care: Does this capability serve users, the community, and avoid harm?
- Growth: Does this expand what the organism can accomplish or earn?
- Honesty: Is this transparent, auditable, and honest in purpose?

Also assess overall EFE (expected free energy reduction, 0.0-1.0). Higher = more worth pursuing.

Respond with JSON only:
{{
  "coherence": float,
  "care": float,
  "growth": float,
  "honesty": float,
  "efe_score": float,
  "should_pursue": true|false,
  "reasoning": "one sentence"
}}"""

        try:
            import json
            llm = getattr(self._simula, "_llm", None) or getattr(self._simula, "llm", None)
            if llm is None:
                return {"reasoning": "llm_not_wired"}, 0.5, True

            response = await llm.generate(prompt, max_tokens=300, temperature=0.3)
            text = response.text.strip()
            # Extract JSON from possible markdown wrapping
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            data = json.loads(text)
            drive_alignment = {
                "coherence": float(data.get("coherence", 0.5)),
                "care": float(data.get("care", 0.5)),
                "growth": float(data.get("growth", 0.5)),
                "honesty": float(data.get("honesty", 0.5)),
                "reasoning": data.get("reasoning", ""),
            }
            efe_score = float(data.get("efe_score", 0.5))
            should_pursue = bool(data.get("should_pursue", True))
            return drive_alignment, efe_score, should_pursue
        except Exception as exc:
            logger.warning("self_modification.llm_evaluation_failed", error=str(exc))
            # Fail open: Equor will review; don't silently kill proposals on infra errors
            return {"reasoning": f"llm_error: {exc}"}, 0.5, True

    def _infer_capabilities(self, state: _PipelineState) -> list[str]:
        """Infer required_capabilities list from action_type and description."""
        combined = (state.action_type + " " + state.description).lower()
        caps: list[str] = []
        cap_keywords = {
            "wallet": "wallet_access",
            "defi": "defi_write",
            "git": "git_write",
            "http": "http_client",
            "code": "code_generation",
            "database": "database_write",
        }
        for kw, cap in cap_keywords.items():
            if kw in combined:
                caps.append(cap)
        if not caps:
            caps = ["http_client"]
        return caps

    @staticmethod
    def _is_subsystem_needed(description: str) -> bool:
        """
        True if the description suggests a full subsystem is needed
        rather than a single Axon executor.
        """
        keywords = {
            "subsystem", "new system", "new module", "full pipeline",
            "continuous monitoring", "background loop", "multiple executors",
            "persistent state", "own event handlers",
        }
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in keywords)

    # ── Introspection ─────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "pending_proposals": len(self._pending),
            "active_test_goals": len(self._test_goals),
            "attached": self._attached,
        }
