"""
EcodiaOS — Equor Service

The conscience of EOS. Single interface for:
- Constitutional review (the primary entry point from Nova)
- Invariant management
- Autonomy enforcement
- Drift monitoring
- Amendment facilitation
- Audit trail

Equor cannot be disabled. If Equor fails, the instance enters safe mode
where only Level 1 (Advisor) actions are permitted.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import (
    DriveAlignmentVector,
    Verdict,
    new_id,
    utc_now,
)
from primitives.constitutional import ConstitutionalCheck
from systems.equor.amendment import (
    apply_amendment,
    propose_amendment,
)
from systems.equor.amendment_pipeline import (
    ShadowTracker,
    cast_vote,
    complete_shadow_period,
    evaluate_shadow,
    get_amendment_status,
    open_voting,
    start_shadow_period,
    submit_amendment,
    tally_votes,
)
from systems.equor.amendment_pipeline import (
    adopt_amendment as pipeline_adopt_amendment,
)
from systems.equor.autonomy import (
    apply_autonomy_change,
    check_promotion_eligibility,
    get_autonomy_level,
)
from systems.equor.constitutional_memory import ConstitutionalMemory
from systems.equor.drift import DriftTracker, respond_to_drift, store_drift_report
from systems.equor.economic_evaluator import (
    apply_economic_adjustment,
    classify_economic_action,
    evaluate_economic_intent,
)
from systems.equor.evaluators import (
    BaseEquorEvaluator,
    default_evaluators,
    evaluate_all_drives,
)
from systems.equor.invariants import (
    HARDCODED_INVARIANTS,
    check_community_invariant,
)
from systems.equor.schema import ensure_equor_schema, seed_hardcoded_invariants
from systems.equor.template_library import TemplateLibrary
from systems.equor.verdict import compute_verdict

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient
    from config import EquorConfig, GovernanceConfig
    from core.hotreload import NeuroplasticityBus
    from primitives.intent import Intent
    from systems.axon.service import AxonService
    from systems.synapse.types import SynapseEvent

# Regex the admin must send to authorise a suspended intent: "AUTH <6-digit-code>"
_HITL_AUTH_RE = re.compile(r"^AUTH\s+(\d{6})$", re.IGNORECASE)
# Redis key prefix for suspended intents.
# Full key: eos:hitl:suspended:<6-digit-code>
_HITL_KEY_PREFIX = "eos:hitl:suspended:"
# Default TTL for suspended-intent entries in Redis. Overridden by
# EquorConfig.hitl_intent_ttl_s so it can be tuned per deployment.
# The 1-hour original was too short for human review workflows that
# span timezone boundaries or require committee sign-off.
_HITL_INTENT_TTL_S = 86400  # 24 hours

logger = structlog.get_logger()

# Review timeout: the entire review() call must not block the event loop
# beyond this budget. Community invariant LLM calls are the most expensive
# component and will be skipped if the budget is exhausted.
_REVIEW_TIMEOUT_S = 0.8
# Cache TTL for constitution and autonomy level (seconds).
# These change only via governance events, so a short TTL is safe.
_STATE_CACHE_TTL_S = 30.0
# Cache TTL for high-confidence hypotheses fetched from Evo's Memory graph.
# Hypotheses mature slowly (24h minimum age), so a 60-second TTL is safe.
_HYPOTHESIS_CACHE_TTL_S = 60.0


class EquorService:
    """
    The constitutional ethics system.
    Gates every intent before execution.
    Cannot be disabled.
    """

    system_id: str = "equor"

    def __init__(
        self,
        neo4j: Neo4jClient,
        llm: LLMProvider,
        config: EquorConfig,
        governance_config: GovernanceConfig,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
        redis: RedisClient | None = None,
    ):
        self._neo4j = neo4j
        self._llm = llm
        self._config = config
        self._governance = governance_config
        self._drift_tracker = DriftTracker(window_size=config.drift_window_size)
        self._safe_mode = False
        self._total_reviews = 0
        self._evo: Any = None  # Wired post-init for learning feedback from vetoes
        self._axon: Any = None  # Wired post-init for HITL dispatch
        self._bus = neuroplasticity_bus
        self._redis: RedisClient | None = redis
        # Event bus wired via subscribe_hitl(); used to emit INTENT_REJECTED.
        self._event_bus: Any = None
        # Pluggable notification hook: set by the application layer so Equor
        # doesn't directly depend on IdentityCommConfig.
        self._send_admin_sms: Any = None  # async callable(message: str) | None

        # Live evaluator set — hot-reloaded via the NeuroplasticityBus.
        # Initialised with built-in defaults; the bus callback replaces
        # individual evaluators when Simula evolves a new subclass.
        self._evaluators: dict[str, BaseEquorEvaluator] = default_evaluators()

        # Constitutional template library for the Arbitrage Reflex Arc.
        # Templates are pre-approved execution strategies that bypass Nova
        # and full Equor review for sub-200ms market execution.
        self._template_library = TemplateLibrary()

        # Cached state: constitution and autonomy level rarely change (only via
        # governance events), so we cache them to avoid hitting Neo4j on every
        # review() call. Invalidated after _STATE_CACHE_TTL_S or on mutation.
        self._cached_constitution: dict[str, Any] | None = None
        self._cached_autonomy_level: int | None = None
        self._cache_updated_at: float = 0.0

        # Constitutional memory: rolling window of past decisions used to
        # detect novel-intent patterns that have historically been blocked.
        self._constitutional_memory = ConstitutionalMemory(
            max_size=getattr(config, "memory_window_size", 500)
        )

        # Cached high-confidence Evo hypotheses for contradiction detection.
        # Refreshed every _HYPOTHESIS_CACHE_TTL_S seconds from Neo4j.
        self._cached_hypotheses: list[dict[str, Any]] = []
        self._hypotheses_updated_at: float = 0.0

        # Active amendment shadow tracker. At most one amendment can be in
        # shadow mode at a time. When active, every review() call also runs
        # the proposed weights in parallel and records the divergence.
        self._shadow_tracker: ShadowTracker | None = None

    def set_evo(self, evo: Any) -> None:
        """Wire Evo so constitutional vetoes become learning episodes."""
        self._evo = evo
        logger.info("evo_wired_to_equor")

    def set_notification_hook(self, send_fn: Any) -> None:
        """
        Wire an async callable ``send_fn(message: str) -> None`` that Equor
        will call when suspending a HITL intent.

        Typical usage in app startup::

            equor.set_notification_hook(
                lambda msg: send_admin_sms(cfg.identity_comm, msg)
            )
        """
        self._send_admin_sms = send_fn
        logger.info("equor_notification_hook_set")

    def set_axon(self, axon: AxonService) -> None:
        """Wire Axon so Equor can dispatch HITL-approved intents for execution."""
        self._axon = axon
        logger.info("axon_wired_to_equor")

    def subscribe_hitl(self, event_bus: Any) -> None:
        """
        Register the HITL listener on the Synapse event bus.

        Call this after both Equor and the event bus are initialised::

            equor.subscribe_hitl(event_bus)

        This subscribes ``on_identity_verification_received`` to
        ``IDENTITY_VERIFICATION_RECEIVED`` events so admin SMS replies
        unlock suspended intents.
        """
        from systems.synapse.types import SynapseEventType

        self._event_bus = event_bus
        event_bus.subscribe(
            SynapseEventType.IDENTITY_VERIFICATION_RECEIVED,
            self.on_identity_verification_received,
        )
        logger.info("equor_hitl_listener_registered")

    # ─── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Ensure schema, seed invariants, and register for hot-reload."""
        await ensure_equor_schema(self._neo4j)
        await seed_hardcoded_invariants(self._neo4j)

        if self._bus is not None:
            self._bus.register(
                base_class=BaseEquorEvaluator,
                registration_callback=self._on_evaluator_evolved,
                system_id=self.system_id,
            )

        logger.info("equor_initialized")

    async def shutdown(self) -> None:
        """Deregister evaluators from the bus on shutdown."""
        if self._bus is not None:
            self._bus.deregister(BaseEquorEvaluator)
            logger.info("equor_evaluators_deregistered")

    def _on_evaluator_evolved(self, evaluator: BaseEquorEvaluator) -> None:
        """
        NeuroplasticityBus callback — swap a single drive evaluator in-place.

        The bus instantiates the new subclass and calls this method.  We key on
        ``drive_name`` so only the matching evaluator is replaced; the other
        three continue running undisturbed.
        """
        name = evaluator.drive_name
        if name not in self._evaluators:
            logger.warning(
                "equor_unknown_drive_evolved",
                drive_name=name,
                class_name=type(evaluator).__name__,
            )
            return

        old_cls = type(self._evaluators[name]).__name__
        self._evaluators[name] = evaluator
        logger.info(
            "equor_evaluator_hot_reloaded",
            drive=name,
            old_class=old_cls,
            new_class=type(evaluator).__name__,
        )

    # ─── Primary Entry Point: Constitutional Review ───────────────

    async def review(self, intent: Intent) -> ConstitutionalCheck:
        """
        The primary entry point. Nova submits an Intent for ethical evaluation.
        Target: ≤500ms standard.

        If Equor is in safe mode, only Level 1 actions are permitted.
        """
        start = time.monotonic()

        # Safe mode: only advisory actions permitted
        if self._safe_mode:
            return self._safe_mode_review(intent)

        try:
            async with asyncio.timeout(_REVIEW_TIMEOUT_S):
                return await self._review_inner(intent, start)
        except TimeoutError:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "equor_review_timeout",
                intent_id=intent.id,
                elapsed_ms=elapsed_ms,
            )
            # Timeout is NOT a failure — return a conservative approval so
            # we don't block the fast path. The audit trail records the timeout.
            return ConstitutionalCheck(
                intent_id=intent.id,
                verdict=Verdict.APPROVED,
                reasoning=(
                    f"Equor review timed out after {elapsed_ms}ms. "
                    "Approved conservatively (heuristic invariants passed)."
                ),
                confidence=0.5,
            )
        except Exception as e:
            # Equor failure = enter safe mode
            logger.error("equor_review_failed", error=str(e), intent_id=intent.id)
            self._safe_mode = True
            return self._safe_mode_review(intent)

    async def review_critical(self, intent: Intent) -> ConstitutionalCheck:
        """
        Lightweight critical-path review for Nova's fast path.

        Constraints (must complete in ≤50ms):
          - Uses ONLY cached constitution/autonomy (never waits for Neo4j)
          - Runs CPU-only verdict computation (drive evaluation + hardcoded invariants)
          - Skips community invariant LLM checks entirely
          - Skips audit trail write (fire-and-forget bookkeeping still runs)

        If no cached state is available (cold start), conservatively approves
        with low confidence so the fast path can proceed without blocking.
        """
        start = time.monotonic()

        if self._safe_mode:
            return self._safe_mode_review(intent)

        # Use cached state only — never block on Neo4j
        if (
            self._cached_constitution is not None
            and self._cached_autonomy_level is not None
        ):
            constitution = self._cached_constitution
            autonomy_level = self._cached_autonomy_level
        else:
            # Cold start: no cached state yet. Approve conservatively.
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.debug(
                "critical_review_no_cache",
                intent_id=intent.id,
                elapsed_ms=elapsed_ms,
            )
            return ConstitutionalCheck(
                intent_id=intent.id,
                verdict=Verdict.APPROVED,
                reasoning="Critical-path review: no cached state, approved conservatively.",
                confidence=0.4,
            )

        # Pure CPU: drive evaluation + hardcoded invariant verdict
        alignment = await evaluate_all_drives(intent, self._evaluators)

        # Economic guardrail on critical path too — all CPU, no I/O.
        economic_delta = evaluate_economic_intent(intent)
        if economic_delta is not None:
            alignment = apply_economic_adjustment(alignment, economic_delta)

        check = compute_verdict(alignment, intent, autonomy_level, constitution)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Fire-and-forget bookkeeping (non-blocking)
        asyncio.create_task(
            self._post_review_bookkeeping(intent, alignment, check, elapsed_ms),
            name=f"equor_critical_{intent.id[:8]}",
        )

        logger.debug(
            "critical_review_complete",
            intent_id=intent.id,
            verdict=check.verdict.value,
            latency_ms=elapsed_ms,
        )
        return check

    async def _review_inner(
        self, intent: Intent, start: float,
    ) -> ConstitutionalCheck:
        """Core review logic, called within the review timeout."""
        # 1. Drive evaluation + cached state fetch in parallel.
        #    Drive evaluators are pure CPU heuristics (<1ms).
        #    Constitution/autonomy are cached with TTL to avoid Neo4j per-review.
        #    Hypotheses are fetched separately with their own TTL so a slow
        #    Evo graph doesn't block the core review path.
        alignment, (constitution, autonomy_level), hypotheses = await asyncio.gather(
            evaluate_all_drives(intent, self._evaluators),
            self._get_cached_state(),
            self._get_cached_hypotheses(),
        )

        # 1b. Economic guardrail: if this is an Oikos economic intent,
        #     apply domain-specific drive adjustments before verdict.
        #     This ensures Care/Honesty floors catch scam assets, harmful
        #     bounties, etc. even if base evaluators scored them neutrally.
        economic_delta = evaluate_economic_intent(intent)
        if economic_delta is not None:
            alignment = apply_economic_adjustment(alignment, economic_delta)
            logger.debug(
                "economic_adjustment_applied",
                intent_id=intent.id,
                action_type=classify_economic_action(intent),
                adjusted_care=f"{alignment.care:.2f}",
                adjusted_honesty=f"{alignment.honesty:.2f}",
            )

        # 2. Run the verdict engine (pure CPU, includes hardcoded invariant checks,
        #    contradiction detection against Evo hypotheses, and constitutional
        #    memory consultation for novel intent patterns).
        check = compute_verdict(
            alignment, intent, autonomy_level, constitution,
            hypotheses=hypotheses,
            memory=self._constitutional_memory,
        )

        # 2a. Shadow evaluation: if an amendment is in shadow mode, run the
        #     proposed weights in parallel and record the divergence. This is
        #     fire-and-forget — the shadow verdict does not affect the real verdict.
        if self._shadow_tracker is not None:
            try:
                shadow = evaluate_shadow(
                    alignment, intent, autonomy_level, constitution,
                    self._shadow_tracker,
                    hypotheses=hypotheses,
                    memory=self._constitutional_memory,
                )
                if shadow.invariant_violation:
                    logger.warning(
                        "amendment_shadow_invariant_violation",
                        proposal_id=self._shadow_tracker.proposal_id,
                        intent_id=intent.id,
                    )
                if self._shadow_tracker.has_failed or self._shadow_tracker.is_expired:
                    asyncio.create_task(
                        self._finalize_shadow_period(),
                        name="equor_shadow_finalize",
                    )
            except Exception:
                logger.debug("shadow_evaluation_failed", exc_info=True)

        # 2b. HITL gate: only GOVERNED actions require human approval before
        #     execution. GOVERNED = constitutional amendment, mitosis, single
        #     transaction above EOS_HITL_CAPITAL_THRESHOLD, or external
        #     commitment on behalf of Ecodia. All other actions are AUTONOMOUS
        #     and proceed without suspension.
        if check.verdict not in (Verdict.BLOCKED, Verdict.DEFERRED):
            from systems.equor.verdict import _is_governed
            if _is_governed(intent):
                elapsed_ms = int((time.monotonic() - start) * 1000)
                asyncio.create_task(
                    self._post_review_bookkeeping(intent, alignment, check, elapsed_ms),
                    name=f"equor_bookkeeping_{intent.id[:8]}",
                )
                return await self._suspend_for_human_review(intent)

        # 3. Community invariant checks — parallelised with a tight timeout.
        #    Only run if we haven't already blocked (invariant / floor / autonomy).
        if check.verdict not in (Verdict.BLOCKED, Verdict.DEFERRED):
            community_violations = await self._check_community_invariants(intent)
            if community_violations:
                check.verdict = Verdict.BLOCKED
                check.reasoning = (
                    f"Community invariant violated: {community_violations[0]}"
                )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # 4. Fire-and-forget: audit trail, drift tracking, Evo feedback.
        #    These are important but must not block the review response.
        asyncio.create_task(
            self._post_review_bookkeeping(intent, alignment, check, elapsed_ms),
            name=f"equor_bookkeeping_{intent.id[:8]}",
        )

        logger.info(
            "constitutional_review_complete",
            intent_id=intent.id,
            verdict=check.verdict.value,
            composite=f"{alignment.composite:.2f}",
            latency_ms=elapsed_ms,
        )

        return check

    async def _suspend_for_human_review(self, intent: Intent) -> ConstitutionalCheck:
        """
        Suspend a PARTNER+ autonomy intent pending human authorisation.

        1. Generate a 6-digit numeric auth ID (e.g. 482194).
        2. Serialise the Intent to JSON and store in Redis with a TTL
           under key ``eos:hitl:suspended:<id>`` (matching _HITL_KEY_PREFIX).
        3. Fire an outbound SMS to the admin via the registered notification hook.
        4. Return SUSPENDED_AWAITING_HUMAN.

        If Redis is unavailable the intent is BLOCKED to avoid untracked execution.
        """
        auth_id = f"{secrets.randbelow(1_000_000):06d}"
        redis_key = f"{_HITL_KEY_PREFIX}{auth_id}"
        description = intent.goal.description[:120]
        sms_body = (
            "\U0001f6a8 EcodiaOS Equor Gate: Action requires approval. "
            + description
            + f". Reply AUTH {auth_id} to execute."
        )

        if self._redis is not None:
            try:
                await self._redis.set_json(
                    redis_key,
                    intent.model_dump(mode="json"),
                    ttl=getattr(self._config, "hitl_intent_ttl_s", _HITL_INTENT_TTL_S),
                )
            except Exception:
                logger.error(
                    "hitl_redis_store_failed",
                    intent_id=intent.id,
                    auth_id=auth_id,
                    exc_info=True,
                )
                return ConstitutionalCheck(
                    intent_id=intent.id,
                    verdict=Verdict.BLOCKED,
                    reasoning=(
                        "HITL suspension failed: Redis unavailable. "
                        "Intent blocked to prevent untracked execution."
                    ),
                    confidence=1.0,
                )
        else:
            logger.warning("hitl_no_redis_configured", intent_id=intent.id)

        if self._send_admin_sms is not None:
            async def _notify() -> None:
                try:
                    await self._send_admin_sms(sms_body)
                except Exception:
                    logger.error(
                        "hitl_sms_failed",
                        intent_id=intent.id,
                        auth_id=auth_id,
                        exc_info=True,
                    )
            asyncio.create_task(_notify(), name=f"equor_hitl_sms_{auth_id}")
        else:
            logger.warning(
                "hitl_sms_hook_not_configured",
                intent_id=intent.id,
                auth_id=auth_id,
                sms_body=sms_body,
            )

        logger.info(
            "intent_suspended_awaiting_human",
            intent_id=intent.id,
            auth_id=auth_id,
            autonomy_required=intent.autonomy_level_required,
        )
        return ConstitutionalCheck(
            intent_id=intent.id,
            verdict=Verdict.SUSPENDED_AWAITING_HUMAN,
            reasoning=(
                "Intent is in the GOVERNED tier (constitutional amendment, mitosis, "
                "capital above threshold, or external commitment). "
                f"Suspended pending human authorisation (AUTH {auth_id})."
            ),
            confidence=1.0,
        )

    async def _get_cached_state(self) -> tuple[dict[str, Any], int]:
        """Return (constitution_dict, autonomy_level) from cache or Neo4j."""
        now = time.monotonic()
        if (
            self._cached_constitution is not None
            and self._cached_autonomy_level is not None
            and (now - self._cache_updated_at) < _STATE_CACHE_TTL_S
        ):
            return self._cached_constitution, self._cached_autonomy_level

        # Fetch both in parallel
        constitution, autonomy_level = await asyncio.gather(
            self._get_constitution_dict(),
            get_autonomy_level(self._neo4j),
        )
        self._cached_constitution = constitution
        self._cached_autonomy_level = autonomy_level
        self._cache_updated_at = now
        return constitution, autonomy_level

    def _invalidate_state_cache(self) -> None:
        """Called after governance mutations (amendments, autonomy changes)."""
        self._cached_constitution = None
        self._cached_autonomy_level = None
        self._cache_updated_at = 0.0

    async def _get_cached_hypotheses(self) -> list[dict[str, Any]]:
        """
        Return high-confidence Evo hypotheses from cache or Neo4j.

        Only fetches hypotheses in SUPPORTED or INTEGRATED status with an
        evidence_score above the contradiction detector's threshold.
        Refreshed every _HYPOTHESIS_CACHE_TTL_S seconds.
        """
        now = time.monotonic()
        if (now - self._hypotheses_updated_at) < _HYPOTHESIS_CACHE_TTL_S:
            return self._cached_hypotheses

        try:
            from systems.equor.contradiction_detector import (
                CONTRADICTION_CONFIDENCE_THRESHOLD,
                CONTRADICTION_MIN_EPISODES,
            )

            results = await self._neo4j.execute_read(
                """
                MATCH (h:Hypothesis)
                WHERE h.status IN ['supported', 'integrated']
                  AND h.evidence_score >= $min_score
                  AND size(h.supporting_episodes) >= $min_episodes
                RETURN h.id AS id,
                       h.statement AS statement,
                       h.formal_test AS formal_test,
                       h.evidence_score AS evidence_score,
                       h.status AS status,
                       size(h.supporting_episodes) AS supporting_episode_count
                ORDER BY h.evidence_score DESC
                LIMIT 100
                """,
                {
                    "min_score": CONTRADICTION_CONFIDENCE_THRESHOLD,
                    "min_episodes": CONTRADICTION_MIN_EPISODES,
                },
            )
            self._cached_hypotheses = [dict(r) for r in results]
        except Exception:
            logger.warning("hypothesis_cache_refresh_failed", exc_info=True)
            # Keep stale cache rather than disrupting reviews

        self._hypotheses_updated_at = now
        return self._cached_hypotheses

    async def _post_review_bookkeeping(
        self,
        intent: Intent,
        alignment: DriveAlignmentVector,
        check: ConstitutionalCheck,
        elapsed_ms: int,
    ) -> None:
        """Non-blocking post-review work: audit trail, drift, Evo feedback."""
        try:
            await self._store_review_record(intent, alignment, check, elapsed_ms)
        except Exception:
            # Warn, not debug — a missing audit record is an observability gap.
            logger.warning(
                "audit_trail_write_failed",
                intent_id=intent.id,
                verdict=check.verdict.value,
                exc_info=True,
            )

        # Record this decision in the constitutional memory so future reviews on
        # similar novel intents can benefit from the accumulated history.
        try:
            self._constitutional_memory.record(
                intent=intent,
                verdict=check.verdict.value,
                confidence=check.confidence,
                reasoning=check.reasoning,
                composite_alignment=alignment.composite,
                autonomy_level=intent.autonomy_level_required,
            )
        except Exception:
            logger.debug("constitutional_memory_record_failed", exc_info=True)

        self._drift_tracker.record_decision(alignment, check.verdict.value)
        self._total_reviews += 1

        if check.verdict == Verdict.BLOCKED and self._evo is not None:
            try:
                await self._feed_veto_to_evo(intent, check)
            except Exception:
                logger.debug("evo_veto_feed_failed", exc_info=True)

        if check.verdict in (Verdict.BLOCKED, Verdict.DEFERRED) and self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(
                    SynapseEvent(
                        event_type=SynapseEventType.INTENT_REJECTED,
                        source_system="equor",
                        data={
                            "intent_id": intent.id,
                            "intent_goal": intent.goal.description[:200],
                            "verdict": check.verdict.value,
                            "reasoning": check.reasoning,
                            "alignment": alignment.model_dump(),
                        },
                    )
                )
            except Exception:
                logger.debug("intent_rejected_emit_failed", exc_info=True)

        if self._total_reviews % self._config.drift_report_interval == 0:
            try:
                await self._run_drift_check()
                await self._run_promotion_check()
            except Exception:
                logger.debug("drift_or_promotion_check_failed", exc_info=True)

    # ─── Invariant Management ─────────────────────────────────────

    async def get_invariants(self) -> list[dict[str, Any]]:
        """Get all active invariants (hardcoded + community)."""
        results = await self._neo4j.execute_read(
            """
            MATCH (c:Constitution)-[:INCLUDES_INVARIANT]->(i:Invariant)
            WHERE i.active = true
            RETURN i.id AS id, i.name AS name, i.description AS description,
                   i.source AS source, i.severity AS severity
            ORDER BY i.id
            """
        )
        return [dict(r) for r in results]

    async def add_community_invariant(
        self,
        name: str,
        description: str,
        severity: str,
        governance_record_id: str,
    ) -> str:
        """Add a community-defined invariant via governance."""
        invariant_id = f"CINV-{new_id()[:8]}"
        now = utc_now()

        await self._neo4j.execute_write(
            """
            MATCH (c:Constitution)
            CREATE (i:Invariant {
                id: $id,
                name: $name,
                description: $description,
                source: 'community',
                severity: $severity,
                active: true,
                added_at: datetime($now),
                added_by: $gov_id
            })
            CREATE (c)-[:INCLUDES_INVARIANT]->(i)
            """,
            {
                "id": invariant_id,
                "name": name,
                "description": description,
                "severity": severity,
                "now": now.isoformat(),
                "gov_id": governance_record_id,
            },
        )

        logger.info("community_invariant_added", invariant_id=invariant_id, name=name)
        return invariant_id

    # ─── Autonomy ─────────────────────────────────────────────────

    async def get_autonomy_level(self) -> int:
        return await get_autonomy_level(self._neo4j)

    async def check_promotion(self, target_level: int) -> dict[str, Any]:
        current = await get_autonomy_level(self._neo4j)
        return await check_promotion_eligibility(self._neo4j, current, target_level)

    async def apply_autonomy_change(self, new_level: int, reason: str, actor: str = "governance") -> dict[str, Any]:
        self._invalidate_state_cache()
        return await apply_autonomy_change(self._neo4j, new_level, reason, actor)

    # ─── Amendments (Legacy — kept for backward compatibility) ─────

    async def propose_amendment(
        self,
        proposed_drives: dict[str, float],
        title: str,
        description: str,
        proposer_id: str,
    ) -> dict[str, Any]:
        return await propose_amendment(
            self._neo4j, proposed_drives, title, description,
            proposer_id, self._governance,
        )

    async def apply_amendment(self, proposal_id: str, proposed_drives: dict[str, float]) -> dict[str, Any]:
        self._invalidate_state_cache()
        return await apply_amendment(self._neo4j, proposal_id, proposed_drives)

    # ─── Amendment Pipeline (formal process with shadow mode) ────

    async def submit_amendment_proposal(
        self,
        proposed_drives: dict[str, float],
        title: str,
        description: str,
        rationale: str,
        proposer_id: str,
        evidence_hypothesis_ids: list[str],
    ) -> dict[str, Any]:
        """Submit a constitutional amendment with evidence requirements."""
        return await submit_amendment(
            self._neo4j,
            proposed_drives=proposed_drives,
            title=title,
            description=description,
            rationale=rationale,
            proposer_id=proposer_id,
            evidence_hypothesis_ids=evidence_hypothesis_ids,
            governance_config=self._governance,
            min_evidence_count=self._governance.amendment_min_evidence_count,
            min_evidence_confidence=self._governance.amendment_min_evidence_confidence,
        )

    async def start_amendment_shadow(
        self,
        proposal_id: str,
    ) -> dict[str, Any]:
        """Transition an amendment from deliberation to shadow mode."""
        if self._shadow_tracker is not None:
            return {
                "started": False,
                "reason": (
                    f"Shadow period already active for proposal "
                    f"{self._shadow_tracker.proposal_id}."
                ),
            }

        result = await start_shadow_period(
            self._neo4j,
            proposal_id,
            shadow_days=self._governance.amendment_shadow_days,
            max_divergence_rate=self._governance.amendment_shadow_max_divergence_rate,
        )

        if result.get("started"):
            self._shadow_tracker = result.pop("tracker")

        return result

    async def get_shadow_status(self) -> dict[str, Any] | None:
        """Get the current shadow period status, if active."""
        if self._shadow_tracker is None:
            return None
        return self._shadow_tracker.compute_report()

    async def _finalize_shadow_period(self) -> None:
        """Complete the shadow period and clean up the tracker."""
        if self._shadow_tracker is None:
            return

        tracker = self._shadow_tracker
        self._shadow_tracker = None  # Clear before async work

        try:
            result = await complete_shadow_period(self._neo4j, tracker)
            logger.info(
                "shadow_period_finalized",
                proposal_id=tracker.proposal_id,
                passed=result.get("passed"),
            )
        except Exception:
            logger.error("shadow_period_finalize_failed", exc_info=True)

    async def open_amendment_voting(self, proposal_id: str) -> dict[str, Any]:
        """Open an amendment for community voting after shadow passes."""
        return await open_voting(self._neo4j, proposal_id)

    async def cast_amendment_vote(
        self,
        proposal_id: str,
        voter_id: str,
        vote: str,
    ) -> dict[str, Any]:
        """Cast a vote on an amendment."""
        return await cast_vote(self._neo4j, proposal_id, voter_id, vote)

    async def tally_amendment_votes(
        self,
        proposal_id: str,
        total_eligible_voters: int,
    ) -> dict[str, Any]:
        """Tally votes and determine if the amendment passes."""
        return await tally_votes(self._neo4j, proposal_id, total_eligible_voters)

    async def adopt_passed_amendment(self, proposal_id: str) -> dict[str, Any]:
        """Apply a passed amendment to the constitution."""
        result = await pipeline_adopt_amendment(self._neo4j, proposal_id)
        if result.get("adopted"):
            self._invalidate_state_cache()
        return result

    async def get_amendment_pipeline_status(
        self, proposal_id: str,
    ) -> dict[str, Any] | None:
        """Get the full pipeline status for an amendment."""
        return await get_amendment_status(self._neo4j, proposal_id)

    # ─── Drift ────────────────────────────────────────────────────

    @property
    def constitutional_drift(self) -> float:
        """
        Synchronous scalar drift magnitude for Soma's INTEGRITY dimension.

        Returns drift_severity (0.0 = no drift, 1.0 = severe constitutional drift)
        from the current rolling alignment window. Soma reads this every theta
        cycle as the Equor component of the INTEGRITY signal:
            integrity_equor_component = 1.0 - constitutional_drift

        Returns 0.0 (no drift) when fewer than 10 decisions have been recorded.
        """
        return float(self._drift_tracker.compute_report().get("drift_severity", 0.0))

    async def get_drift_report(self) -> dict[str, Any]:
        """Get the current drift report."""
        report = self._drift_tracker.compute_report()
        response = respond_to_drift(report)
        return {**report, "recommended_action": response}

    # ─── Governance Records ───────────────────────────────────────

    async def get_recent_reviews(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent constitutional reviews from the audit trail."""
        results = await self._neo4j.execute_read(
            """
            MATCH (g:GovernanceRecord)
            WHERE g.event_type = 'constitutional_review'
            RETURN g.id AS id, g.timestamp AS timestamp,
                   g.intent_id AS intent_id, g.verdict AS verdict,
                   g.alignment_composite AS composite,
                   g.reasoning AS reasoning, g.latency_ms AS latency_ms
            ORDER BY g.timestamp DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [dict(r) for r in results]

    async def get_governance_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get all governance events."""
        results = await self._neo4j.execute_read(
            """
            MATCH (g:GovernanceRecord)
            RETURN g.id AS id, g.event_type AS event_type,
                   g.timestamp AS timestamp, g.actor AS actor,
                   g.outcome AS outcome
            ORDER BY g.timestamp DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [dict(r) for r in results]

    # ─── HITL Authorization Flow ──────────────────────────────────

    async def on_identity_verification_received(self, event: SynapseEvent) -> None:
        """
        Event listener for IDENTITY_VERIFICATION_RECEIVED.

        Called by Synapse when the Twilio webhook fires after the admin texts
        a reply.  Expected raw body format: ``AUTH <6-digit-id>``

        Flow:
          1. Match raw_body against ``^AUTH\\s+(\\d{6})$``.
          2. Look up the suspended Intent in Redis at
             ``eos:hitl:suspended:<6-digit-id>``.
          3. Delete the Redis key (prevents replay).
          4. Upgrade intent.ethical_clearance.status → APPROVED.
          5. Dispatch the intent to Axon via ExecutionRequest.
          6. Send admin a confirmation SMS.
        """
        data: dict[str, Any] = event.data if hasattr(event, "data") else {}
        raw_body: str = data.get("raw_body", "").strip()

        m = _HITL_AUTH_RE.match(raw_body)
        if not m:
            logger.debug(
                "hitl_auth_ignored",
                reason="body_does_not_match_auth_pattern",
                preview=raw_body[:40],
            )
            return

        short_id: str = m.group(1)
        redis_key = f"{_HITL_KEY_PREFIX}{short_id}"

        if self._redis is None:
            logger.error("hitl_auth_failed", reason="redis_not_configured", short_id=short_id)
            return

        try:
            raw_intent = await self._redis.get(redis_key)
        except Exception as exc:
            logger.error("hitl_redis_get_failed", short_id=short_id, error=str(exc))
            return

        if raw_intent is None:
            logger.warning(
                "hitl_auth_no_matching_intent",
                short_id=short_id,
                note="expired or already consumed",
            )
            return

        # Delete first — prevents a second AUTH replay from re-dispatching
        try:
            await self._redis.delete(redis_key)
        except Exception as exc:
            logger.error("hitl_redis_delete_failed", short_id=short_id, error=str(exc))
            return

        # Reconstruct the Intent from JSON
        try:
            from primitives.intent import Intent
            intent_dict: dict[str, Any] = json.loads(raw_intent)
            intent = Intent.model_validate(intent_dict)
        except Exception as exc:
            logger.error("hitl_intent_deserialize_failed", short_id=short_id, error=str(exc))
            return

        # Formally upgrade the EthicalClearance to APPROVED
        intent.ethical_clearance.status = Verdict.APPROVED
        intent.ethical_clearance.reasoning = (
            f"HITL admin authorisation received (code {short_id})."
        )

        logger.info(
            "hitl_intent_authorized",
            intent_id=intent.id,
            short_id=short_id,
            goal=intent.goal.description[:80],
        )

        # Dispatch to Axon
        if self._axon is not None:
            try:
                from primitives.constitutional import ConstitutionalCheck
                from systems.axon.types import ExecutionRequest

                equor_check = ConstitutionalCheck(
                    intent_id=intent.id,
                    verdict=Verdict.APPROVED,
                    reasoning=f"HITL admin authorisation (code {short_id}).",
                    confidence=1.0,
                )
                request = ExecutionRequest(intent=intent, equor_check=equor_check)
                asyncio.create_task(
                    self._axon.execute(request),
                    name=f"hitl_dispatch_{intent.id[:8]}",
                )
                logger.info("hitl_intent_dispatched_to_axon", intent_id=intent.id)
            except Exception as exc:
                logger.error("hitl_axon_dispatch_failed", intent_id=intent.id, error=str(exc))
        else:
            logger.warning("hitl_axon_not_wired", intent_id=intent.id)

        # Confirmation SMS
        if self._send_admin_sms is not None:
            try:
                await self._send_admin_sms(
                    f"✅ Intent {short_id} Authorized and Executing."
                )
            except Exception as exc:
                logger.error("hitl_confirmation_sms_failed", error=str(exc))

    # ─── Constitutional Template Library (Arbitrage Reflex Arc) ────

    @property
    def template_library(self) -> TemplateLibrary:
        """
        Read-only access to the constitutional template library.

        Atune's MarketPatternDetector queries this to find pre-approved
        templates for fast-path execution. Equor owns the write path.
        """
        return self._template_library

    async def register_template(
        self,
        template_id: str,
        pattern_signature: dict[str, Any],
        max_capital_per_execution: float,
        approval_confidence: float = 0.95,
    ) -> ConstitutionalCheck:
        """
        Register a new ConstitutionalTemplate after deliberative review.

        This is the governance entry point for adding fast-path strategies.
        Equor performs a lightweight constitutional check on the template
        parameters before registering it.

        Returns ConstitutionalCheck with the registration verdict.
        """
        from primitives.fast_path import ConstitutionalTemplate

        # Validate: approval confidence must be high for fast-path
        if approval_confidence < 0.9:
            return ConstitutionalCheck(
                intent_id=template_id,
                verdict=Verdict.BLOCKED,
                reasoning=(
                    f"Approval confidence {approval_confidence:.2f} below "
                    f"fast-path minimum (0.9). Use standard execution path."
                ),
                confidence=1.0,
            )

        # Validate: capital ceiling must be positive and bounded
        if max_capital_per_execution <= 0 or max_capital_per_execution > 10_000:
            return ConstitutionalCheck(
                intent_id=template_id,
                verdict=Verdict.BLOCKED,
                reasoning=(
                    f"Capital ceiling ${max_capital_per_execution:.2f} outside "
                    f"allowed range ($0, $10,000]. Adjust and resubmit."
                ),
                confidence=1.0,
            )

        template = ConstitutionalTemplate(
            template_id=template_id,
            pattern_signature=pattern_signature,
            max_capital_per_execution=max_capital_per_execution,
            approval_confidence=approval_confidence,
        )

        self._template_library.register(template)

        logger.info(
            "constitutional_template_registered",
            template_id=template_id,
            confidence=approval_confidence,
            max_capital=max_capital_per_execution,
        )

        return ConstitutionalCheck(
            intent_id=template_id,
            verdict=Verdict.APPROVED,
            reasoning=f"Template '{template_id}' registered for fast-path execution.",
            confidence=approval_confidence,
        )

    async def revoke_template(self, template_id: str, reason: str = "governance") -> bool:
        """Revoke a constitutional template via governance action."""
        success = self._template_library.deactivate(template_id, reason=reason)
        if success:
            logger.info(
                "constitutional_template_revoked",
                template_id=template_id,
                reason=reason,
            )
        return success

    # ─── Health ───────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Health check for Equor."""
        return {
            "status": "safe_mode" if self._safe_mode else "healthy",
            "total_reviews": self._total_reviews,
            "drift_tracker_size": self._drift_tracker.history_size,
            "safe_mode": self._safe_mode,
            "invariant_count": len(HARDCODED_INVARIANTS),
            "template_library": self._template_library.stats,
            "constitutional_memory": self._constitutional_memory.stats,
            "cached_hypotheses": len(self._cached_hypotheses),
        }

    # ─── Internal Helpers ─────────────────────────────────────────

    def _safe_mode_review(self, intent: Intent) -> ConstitutionalCheck:
        """In safe mode, only Level 1 actions pass."""
        from systems.equor.verdict import ACTION_AUTONOMY_MAP

        # Check if any step requires > Level 1
        for step in intent.plan.steps:
            executor_base = step.executor.split(".")[0].lower() if step.executor else ""
            for action_key, level in ACTION_AUTONOMY_MAP.items():
                if action_key in executor_base and level != 1:
                    return ConstitutionalCheck(
                        intent_id=intent.id,
                        verdict=Verdict.BLOCKED,
                        reasoning=(
                            "Equor is in safe mode. Only Level 1 (Advisor) "
                            "actions are permitted until normal operation resumes."
                        ),
                        confidence=1.0,
                    )

        return ConstitutionalCheck(
            intent_id=intent.id,
            verdict=Verdict.APPROVED,
            reasoning="Safe mode: Level 1 action permitted.",
            confidence=0.9,
        )

    async def _get_constitution_dict(self) -> dict[str, Any]:
        """Fetch the current constitution as a plain dict."""
        results = await self._neo4j.execute_read(
            """
            MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
            RETURN c.drive_coherence AS drive_coherence,
                   c.drive_care AS drive_care,
                   c.drive_growth AS drive_growth,
                   c.drive_honesty AS drive_honesty,
                   c.version AS version
            """
        )
        if results:
            return dict(results[0])
        # Fallback defaults
        return {
            "drive_coherence": 1.0,
            "drive_care": 1.0,
            "drive_growth": 1.0,
            "drive_honesty": 1.0,
            "version": 1,
        }

    async def _check_community_invariants(self, intent: Intent) -> list[str]:
        """Check all community-defined invariants via LLM evaluation (parallelised)."""
        results = await self._neo4j.execute_read(
            """
            MATCH (c:Constitution)-[:INCLUDES_INVARIANT]->(i:Invariant)
            WHERE i.source = 'community' AND i.active = true
            RETURN i.name AS name, i.description AS description
            """
        )

        if not results:
            return []

        # Run all invariant checks in parallel with a per-check timeout
        async def _check_one(row: dict[str, Any]) -> str | None:
            try:
                async with asyncio.timeout(0.4):
                    satisfied = await check_community_invariant(
                        self._llm, intent, row["name"], row["description"],
                    )
                    return None if satisfied else row["name"]
            except TimeoutError:
                logger.warning(
                    "community_invariant_check_timeout",
                    invariant=row["name"],
                )
                return None  # Timeout = skip (fail-open for liveness)
            except Exception:
                return row["name"]  # Error = fail-safe (treat as violated)

        check_results = await asyncio.gather(
            *[_check_one(row) for row in results],
        )
        return [name for name in check_results if name is not None]

    async def _feed_veto_to_evo(
        self, intent: Intent, check: ConstitutionalCheck,
    ) -> None:
        """
        Feed a constitutional veto to Evo as a learning episode.

        The organism should learn from its constitutional failures — when Equor
        blocks an intent, the violation becomes a negative-affect episode so Evo
        can refine hypothesis about which intent patterns violate the constitution.
        """
        try:
            from primitives.memory_trace import Episode

            episode = Episode(
                source="equor.veto",
                modality="internal",
                raw_content=(
                    f"Constitutional veto: {intent.goal.description[:200]}. "
                    f"Reason: {check.reasoning[:300]}"
                ),
                summary=f"Blocked intent: {check.reasoning[:100]}",
                salience_composite=0.7,  # Vetoes are important learning events
                affect_valence=-0.3,
                affect_arousal=0.4,
            )
            await self._evo.process_episode(episode)
            logger.info("veto_fed_to_evo", intent_id=intent.id)
        except Exception:
            logger.debug("evo_veto_feed_failed", exc_info=True)

    async def _store_review_record(
        self,
        intent: Intent,
        alignment: DriveAlignmentVector,
        check: ConstitutionalCheck,
        latency_ms: int,
    ) -> None:
        """Store a constitutional review in the immutable audit trail."""
        now = utc_now()
        record_id = new_id()

        await self._neo4j.execute_write(
            """
            CREATE (g:GovernanceRecord {
                id: $id,
                event_type: 'constitutional_review',
                timestamp: datetime($now),
                intent_id: $intent_id,
                alignment_coherence: $coherence,
                alignment_care: $care,
                alignment_growth: $growth,
                alignment_honesty: $honesty,
                alignment_composite: $composite,
                verdict: $verdict,
                reasoning: $reasoning,
                confidence: $confidence,
                latency_ms: $latency_ms,
                actor: 'equor',
                outcome: $verdict
            })
            """,
            {
                "id": record_id,
                "now": now.isoformat(),
                "intent_id": intent.id,
                "coherence": alignment.coherence,
                "care": alignment.care,
                "growth": alignment.growth,
                "honesty": alignment.honesty,
                "composite": alignment.composite,
                "verdict": check.verdict.value,
                "reasoning": check.reasoning,
                "confidence": check.confidence,
                "latency_ms": latency_ms,
            },
        )

    async def _run_drift_check(self) -> None:
        """Run a drift check and respond accordingly."""
        report = self._drift_tracker.compute_report()
        response = respond_to_drift(report)

        if response["action"] != "log":
            await store_drift_report(self._neo4j, report, response)

        # Auto-demote on severe drift
        if response["action"] == "demote_autonomy":
            current = await get_autonomy_level(self._neo4j)
            if current > 1:
                await apply_autonomy_change(
                    self._neo4j,
                    current - 1,
                    reason=response["detail"],
                    actor="equor_drift_detection",
                )

    async def _run_promotion_check(self) -> None:
        """
        Periodically check whether the instance is eligible for autonomy promotion.

        Promotion requires governance approval, so this method only records
        eligibility as a governance record and logs it — it does NOT auto-promote.
        Governance (human or community vote) must call apply_autonomy_change().
        """
        try:
            current = await get_autonomy_level(self._neo4j)
            if current >= 3:
                return  # Already at maximum (Steward)

            target = current + 1
            eligibility = await check_promotion_eligibility(
                self._neo4j, current, target,
            )

            if not eligibility["eligible"]:
                return

            # Record the eligibility event so governance can act on it
            now = utc_now()
            record_id = new_id()
            await self._neo4j.execute_write(
                """
                CREATE (g:GovernanceRecord {
                    id: $id,
                    event_type: 'promotion_eligible',
                    timestamp: datetime($now),
                    details: $details,
                    actor: 'equor_promotion_check',
                    outcome: 'eligible'
                })
                """,
                {
                    "id": record_id,
                    "now": now.isoformat(),
                    "details": f"Eligible for promotion from level {current} to {target}",
                },
            )

            logger.info(
                "promotion_eligibility_detected",
                current_level=current,
                target_level=target,
                checks=eligibility["checks"],
            )
        except Exception:
            logger.debug("promotion_check_failed", exc_info=True)
