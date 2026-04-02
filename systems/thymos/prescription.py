"""
EcodiaOS - Thymos Prescription Layer (Repair Strategy)

Based on diagnosis, Thymos prescribes the least invasive effective repair.
This follows the principle of minimal intervention: try rest before
antibiotics before surgery.

Two components:
  1. RepairPrescriber   - selects the repair tier and generates RepairSpec
  2. RepairValidator    - gates repairs through constitutional review + safety
"""

from __future__ import annotations

from typing import Any

import structlog

from primitives.common import utc_now
from systems.thymos.types import (
    ApiErrorContext,
    Diagnosis,
    Incident,
    IncidentSeverity,
    ParameterFix,
    RepairSpec,
    RepairTier,
    ValidationResult,
)

logger = structlog.get_logger()


# ─── API Context Extraction ──────────────────────────────────────


def _extract_api_context(incident: Incident) -> ApiErrorContext | None:
    """Extract API context from typed ApiErrorContext or plain dict."""
    if isinstance(incident.context, ApiErrorContext):
        return incident.context
    if isinstance(incident.context, dict):
        ctx = incident.context
        if "http_path" in ctx:
            return ApiErrorContext(
                endpoint=ctx["http_path"],
                method=ctx.get("http_method", "GET"),
                status_code=ctx.get("http_status", 0),
                request_id=ctx.get("request_id", ""),
                remote_addr=ctx.get("remote_addr", "unknown"),
                latency_ms=ctx.get("latency_ms", 0.0),
                user_agent=ctx.get("user_agent", ""),
            )
    return None


# ─── Parameter Fix Registry ─────────────────────────────────────


# Root cause → parameter adjustments that might resolve it
PARAMETER_FIXES: dict[str, list[ParameterFix]] = {
    "memory_pressure": [
        ParameterFix(
            parameter_path="synapse.resources.memory.evo",
            delta=-128,
            reason="Reduce Evo memory to relieve pressure",
        ),
        ParameterFix(
            parameter_path="synapse.resources.memory.simula",
            delta=-64,
            reason="Reduce Simula memory",
        ),
    ],
    "retrieval_timeout": [
        ParameterFix(
            parameter_path="memory.retrieval.timeout_ms",
            delta=50,
            reason="Give Memory more time for retrieval",
        ),
    ],
    "workspace_contention": [
        ParameterFix(
            parameter_path="synapse.clock.current_period_ms",
            delta=20,
            reason="Slow the cycle to reduce workspace contention",
        ),
    ],
    "llm_rate_limit": [
        ParameterFix(
            parameter_path="voxis.generation.max_concurrent",
            delta=-1,
            reason="Reduce concurrent LLM calls",
        ),
        ParameterFix(
            parameter_path="evo.hypothesis.batch_size",
            delta=-1,
            reason="Reduce Evo LLM usage",
        ),
    ],
    "Resource pressure causing latency increase": [
        ParameterFix(
            parameter_path="synapse.clock.current_period_ms",
            delta=30,
            reason="Slow cycle to reduce resource pressure",
        ),
    ],
    # API timeout / pool fixes (Tier 1, spec §1 - 5xx PARAMETER path)
    "request timeout or pool exhaustion": [
        ParameterFix(
            parameter_path="config.api.request_timeout_ms",
            delta=2000,
            reason="Increase API request timeout to reduce 5xx gateway errors",
        ),
        ParameterFix(
            parameter_path="config.api.read_timeout_ms",
            delta=1000,
            reason="Increase read timeout for slow downstream responses",
        ),
        ParameterFix(
            parameter_path="config.api.pool_size",
            delta=5,
            reason="Grow connection pool to handle concurrent API load",
        ),
    ],
}

# ── API-specific parameter fixes keyed by status-code range ─────
#
# These are consulted by _check_api_parameter_fixes() when the incident
# carries ApiErrorContext and the standard PARAMETER_FIXES table doesn't
# match.  Keyed by HTTP status prefix (5 = 5xx, 4 = 404, etc.).

_API_STATUS_PARAMETER_FIXES: dict[int, list[ParameterFix]] = {
    # 5xx: timeout / pool expansion
    5: [
        ParameterFix(
            parameter_path="config.api.request_timeout_ms",
            delta=2000,
            reason="Increase API request timeout to reduce 5xx gateway errors",
        ),
        ParameterFix(
            parameter_path="config.api.read_timeout_ms",
            delta=1000,
            reason="Increase read timeout for slow downstream responses",
        ),
        ParameterFix(
            parameter_path="config.api.pool_size",
            delta=5,
            reason="Grow connection pool to handle concurrent API load",
        ),
    ],
    # 404: route registration nudge (increase handler scan depth)
    4: [
        ParameterFix(
            parameter_path="config.api.router_scan_depth",
            delta=1,
            reason="Increase router scan depth to pick up unregistered handlers",
        ),
    ],
}


# ─── Repair Prescriber ──────────────────────────────────────────


class RepairPrescriber:
    """
    Prescribes repairs following the principle of minimal intervention:
    the least invasive fix that resolves the issue.

    Tier 0: No-op - transient, already resolved
    Tier 1: Parameter tweak - adjustable without code changes
    Tier 2: System restart - bad state but code is fine
    Tier 3: Known fix - apply antibody from the library
    Tier 4: Novel fix - generate via Simula Code Agent (local)
    Tier 5: Factory repair - dispatch to EcodiaOS Factory CC engine (full autonomy)
    Tier 6: Human escalation - cannot auto-resolve
    """

    def __init__(self) -> None:
        self._logger = logger.bind(system="thymos", component="prescriber")

    async def prescribe(
        self,
        incident: Incident,
        diagnosis: Diagnosis,
    ) -> RepairSpec:
        """Generate a repair specification based on diagnosis."""

        # ── TIER 4 override: diagnosis explicitly requests NOVEL_FIX ──
        # When the triage router force-escalated to T4 (masking-loop detection),
        # the diagnosis tier is set to NOVEL_FIX. Honor it - don't let lower-tier
        # heuristics short-circuit the codegen path.
        if diagnosis.repair_tier == RepairTier.NOVEL_FIX:
            return RepairSpec(
                tier=RepairTier.NOVEL_FIX,
                action="simula_codegen",
                target_system=incident.source_system,
                reason=f"Forced novel repair: {diagnosis.root_cause}",
            )

        # ── TIER 0: No-op ──
        if incident.occurrence_count == 1 and self._is_likely_transient(incident):
            return RepairSpec(
                tier=RepairTier.NOOP,
                action="log_and_monitor",
                reason="Transient single occurrence - monitoring",
            )

        # ── TIER 3: Known Fix (Antibody) ── (check before parameter/restart)
        if diagnosis.antibody_id is not None:
            return RepairSpec(
                tier=RepairTier.KNOWN_FIX,
                action="apply_antibody",
                antibody_id=diagnosis.antibody_id,
                reason=f"Antibody match: {diagnosis.root_cause}",
            )

        # ── TIER 1: Parameter Tweak ──
        param_fix = self._check_parameter_fixes(diagnosis)
        if param_fix is not None:
            return param_fix

        # ── TIER 2: System Restart ──
        restart_causes = {
            "state_corruption",
            "resource_leak",
            "deadlock",
            "memory_leak",
            "unbounded growth",
        }
        if any(cause in diagnosis.root_cause.lower() for cause in restart_causes):
            return RepairSpec(
                tier=RepairTier.RESTART,
                action="restart_system",
                target_system=incident.source_system,
                reason=f"State issue: {diagnosis.root_cause}",
            )

        # ── API-specific: Tier 1 parameter fix for 5xx / 404 ──
        api_param_fix = self._check_api_parameter_fixes(incident, diagnosis)
        if api_param_fix is not None:
            return api_param_fix

        # ── API-specific: Tier 3 antibody search for 404 route-missing ──
        api_antibody = self._check_api_404_antibody(incident, diagnosis)
        if api_antibody is not None:
            return api_antibody

        # ── TIER 4: Novel Fix (Codegen) ──
        # Confidence threshold is 0.4 (not 0.6) because novel errors by definition
        # have low diagnosis confidence - if we knew the root cause with high
        # confidence we'd have an antibody. Simula's code agent can investigate
        # further with its own tools.
        if diagnosis.confidence > 0.4 and self._is_codegen_appropriate(incident):
            return RepairSpec(
                tier=RepairTier.NOVEL_FIX,
                action="simula_codegen",
                target_system=incident.source_system,
                reason=f"Novel repair needed: {diagnosis.root_cause}",
            )

        # ── TIER 2: Restart as fallback ──
        if incident.severity in (IncidentSeverity.CRITICAL, IncidentSeverity.HIGH):
            return RepairSpec(
                tier=RepairTier.RESTART,
                action="restart_system",
                target_system=incident.source_system,
                reason=f"High severity, no specific fix: {diagnosis.root_cause}",
            )

        # ── TIER 5: Factory Repair (EcodiaOS CC Engine) ──
        # When local Simula can't fix it but the incident is code-related,
        # dispatch to the Factory — full Claude Code autonomy across all codebases.
        if self._is_factory_appropriate(incident, diagnosis):
            return RepairSpec(
                tier=RepairTier.FACTORY_REPAIR,
                action="factory_dispatch",
                target_system=incident.source_system,
                reason=(
                    f"Factory repair: {diagnosis.root_cause} "
                    f"(local codegen insufficient or cross-codebase fix needed)"
                ),
            )

        # ── TIER 6: Human Escalation ──
        return RepairSpec(
            tier=RepairTier.ESCALATE,
            action="alert_operator",
            reason=(
                f"Cannot auto-resolve: {diagnosis.root_cause} "
                f"(confidence: {diagnosis.confidence:.2f})"
            ),
        )

    def _is_factory_appropriate(
        self, incident: Incident, diagnosis: Diagnosis
    ) -> bool:
        """Check if incident warrants dispatching to EcodiaOS Factory.

        Factory is appropriate when:
        - The incident is code-related (bug, regression, missing feature)
        - Simula's local code agent has failed or isn't suitable
        - The fix may span multiple codebases
        - The incident involves an externally deployed system (Vercel, VPS)
        """
        code_indicators = {
            "bug", "regression", "error", "exception", "crash", "failed",
            "broken", "missing", "undefined", "null", "type error",
            "syntax", "import", "dependency", "build", "deploy",
            "test failure", "lint", "compilation",
        }
        cause_lower = (diagnosis.root_cause or "").lower()
        return any(ind in cause_lower for ind in code_indicators)

    def _is_likely_transient(self, incident: Incident) -> bool:
        """Check if an incident is likely transient (network hiccup, etc.)."""
        transient_types = {
            "TimeoutError",
            "ConnectionError",
            "ConnectionResetError",
            "ConnectionRefusedError",
        }
        return incident.error_type in transient_types

    def _check_parameter_fixes(self, diagnosis: Diagnosis) -> RepairSpec | None:
        """Check if a parameter adjustment can resolve the issue."""
        # Check root cause against known parameter fix patterns
        for pattern, fixes in PARAMETER_FIXES.items():
            if pattern.lower() in diagnosis.root_cause.lower():
                return RepairSpec(
                    tier=RepairTier.PARAMETER,
                    action="adjust_parameters",
                    parameter_changes=[f.model_dump() for f in fixes],
                    reason=f"Parameter adjustment for: {pattern}",
                )
        return None

    def _check_api_parameter_fixes(
        self,
        incident: Incident,
        diagnosis: Diagnosis,
    ) -> RepairSpec | None:
        """
        For API incidents with 5xx or 404, prescribe Tier 1 parameter fixes
        targeting timeout and pool configuration when the diagnosis points to
        a timeout/pool issue.

        Only fires when:
          - incident carries API context (typed or plain dict)
          - diagnosis repair_tier is PARAMETER (hypothesis selected Tier 1)
          - status_code is 5xx or 404
        """
        api_ctx = _extract_api_context(incident)
        if api_ctx is None:
            return None
        if diagnosis.repair_tier != RepairTier.PARAMETER:
            return None

        sc = api_ctx.status_code
        prefix = sc // 100  # 5 for 5xx, 4 for 4xx

        fixes = _API_STATUS_PARAMETER_FIXES.get(prefix)
        if not fixes:
            return None

        endpoint = api_ctx.endpoint
        return RepairSpec(
            tier=RepairTier.PARAMETER,
            action="adjust_parameters",
            parameter_changes=[f.model_dump() for f in fixes],
            reason=(
                f"API {sc} on {endpoint}: adjusting timeout/pool parameters "
                f"(Tier 1)"
            ),
        )

    def _check_api_404_antibody(
        self,
        incident: Incident,
        diagnosis: Diagnosis,
    ) -> RepairSpec | None:
        """
        For 404 incidents where the diagnosis suggests a known-fix tier (Tier 3),
        prescribe an antibody search targeting route-registration patterns.

        The antibody library lookup is performed by DiagnosticEngine upstream; here
        we prescribe the KNOWN_FIX action so the validator and apply pipeline
        route it correctly.
        """
        api_ctx = _extract_api_context(incident)
        if api_ctx is None:
            return None
        if api_ctx.status_code != 404:
            return None
        if diagnosis.repair_tier != RepairTier.KNOWN_FIX:
            return None
        if diagnosis.antibody_id is None:
            # No antibody matched - fall through to Tier 4 or escalation
            return None

        return RepairSpec(
            tier=RepairTier.KNOWN_FIX,
            action="apply_antibody",
            antibody_id=diagnosis.antibody_id,
            reason=(
                f"404 on {api_ctx.endpoint}: applying known route-fix "
                f"antibody {diagnosis.antibody_id} (Tier 3)"
            ),
        )

    def _is_codegen_appropriate(self, incident: Incident) -> bool:
        """Should we attempt codegen repair?"""
        # Don't codegen for transient or low-severity issues - UNLESS the
        # incident has been force-escalated to T4 via recurrence detection.
        # Recurring low-severity issues that won't self-resolve ARE worth
        # fixing structurally via Simula.
        recurrence_escalated = (
            incident.repair_tier == RepairTier.NOVEL_FIX
            and incident.occurrence_count > 5
        )
        low_severity = incident.severity in (IncidentSeverity.LOW, IncidentSeverity.INFO)
        if low_severity and not recurrence_escalated:
            return False
        # Don't codegen for system-wide issues
        if incident.blast_radius > 0.5:
            return False
        # API incidents are eligible - the endpoint + error message gives Simula
        # enough context to generate a targeted patch.  Check both structured
        # ApiErrorContext and the plain dict emitted by ErrorCaptureMiddleware.
        if isinstance(incident.context, ApiErrorContext):
            return True
        if isinstance(incident.context, dict) and "http_path" in incident.context:
            return True
        # Stack trace gives Simula the most to work with
        if incident.stack_trace is not None:
            return True
        # Even without a stack trace, a detailed error message (>30 chars) plus
        # a known error type gives Simula enough to search the codebase and
        # generate a fix.  This unblocks codegen for degradation/stall incidents
        # that don't produce tracebacks.
        if len(incident.error_message) > 30 and incident.error_type:
            return True
        return False


# ─── Repair Validator ────────────────────────────────────────────


class RepairValidator:
    """
    Validates a proposed repair before application.

    Gate 1: Equor constitutional review (Tier 3+)
    Gate 2: Blast radius check (reject > 0.5 for auto-repair)
    Gate 3: Rate limit check (prevent healing storms)

    Simula sandbox validation (Gate 3 from spec) is handled by
    Simula's own simulation pipeline when Tier 4 repairs route through it.
    """

    MAX_REPAIRS_PER_HOUR = 50
    MAX_NOVEL_REPAIRS_PER_DAY = 20

    def __init__(self, equor: Any = None) -> None:
        """
        Args:
            equor: The EquorService for constitutional review.
        """
        self._equor = equor
        self._recent_repairs: list[float] = []  # timestamps of recent repairs
        self._recent_novel: list[float] = []  # timestamps of recent novel repairs
        self._logger = logger.bind(system="thymos", component="repair_validator")

    async def validate(
        self,
        incident: Incident,
        repair: RepairSpec,
    ) -> ValidationResult:
        """Run the full validation gate on a proposed repair."""

        # Gate 1: Constitutional review for Tier 3+
        if repair.tier >= RepairTier.KNOWN_FIX and self._equor is not None:
            try:
                review = await self._constitutional_review(incident, repair)
                if not review.approved:
                    return review
            except Exception as exc:
                self._logger.warning(
                    "equor_review_failed",
                    error=str(exc),
                    tier=repair.tier.name,
                )
                # Equor failure for high-tier repairs → escalate
                if repair.tier >= RepairTier.NOVEL_FIX:
                    return ValidationResult(
                        approved=False,
                        reason=f"Equor review failed: {exc}",
                        escalate_to=RepairTier.ESCALATE,
                    )

        # Gate 2: Blast radius for Tier 3+
        if repair.tier >= RepairTier.KNOWN_FIX and incident.blast_radius > 0.5:
            return ValidationResult(
                approved=False,
                reason=(
                    f"Blast radius too high ({incident.blast_radius:.2f}) "
                    f"for automated repair"
                ),
                escalate_to=RepairTier.ESCALATE,
            )

        # Gate 3: Rate limiting
        now_ts = utc_now().timestamp()
        hour_ago = now_ts - 3600.0
        day_ago = now_ts - 86400.0

        recent_count = sum(1 for ts in self._recent_repairs if ts > hour_ago)
        if recent_count >= self.MAX_REPAIRS_PER_HOUR:
            return ValidationResult(
                approved=False,
                reason=(
                    f"Healing budget exceeded: {recent_count} repairs in last hour "
                    f"(max: {self.MAX_REPAIRS_PER_HOUR})"
                ),
                escalate_to=RepairTier.ESCALATE,
            )

        if repair.tier == RepairTier.NOVEL_FIX:
            novel_count = sum(1 for ts in self._recent_novel if ts > day_ago)
            if novel_count >= self.MAX_NOVEL_REPAIRS_PER_DAY:
                return ValidationResult(
                    approved=False,
                    reason=(
                        f"Novel repair budget exceeded: {novel_count} in 24h "
                        f"(max: {self.MAX_NOVEL_REPAIRS_PER_DAY})"
                    ),
                    escalate_to=RepairTier.ESCALATE,
                )

        return ValidationResult(approved=True)

    def record_repair(self, repair: RepairSpec) -> None:
        """Record that a repair was applied (for rate limiting)."""
        now_ts = utc_now().timestamp()
        self._recent_repairs.append(now_ts)
        if repair.tier == RepairTier.NOVEL_FIX:
            self._recent_novel.append(now_ts)

        # Prune old entries
        hour_ago = now_ts - 3600.0
        self._recent_repairs = [ts for ts in self._recent_repairs if ts > hour_ago]
        day_ago = now_ts - 86400.0
        self._recent_novel = [ts for ts in self._recent_novel if ts > day_ago]

    async def _constitutional_review(
        self,
        incident: Incident,
        repair: RepairSpec,
    ) -> ValidationResult:
        """Submit repair to Equor as an Intent for constitutional review."""
        from primitives.common import SystemID
        from primitives.intent import (
            Action,
            ActionSequence,
            DecisionTrace,
            GoalDescriptor,
            Intent,
        )

        intent = Intent(
            goal=GoalDescriptor(
                description=f"Immune repair: {repair.reason}",
                target_domain=repair.target_system or incident.source_system,
            ),
            plan=ActionSequence(
                steps=[
                    Action(
                        executor=f"thymos.{repair.action}",
                        parameters={
                            "tier": repair.tier.name,
                            "target": repair.target_system or incident.source_system,
                            "incident_id": incident.id,
                        },
                    )
                ]
            ),
            expected_free_energy=0.0,
            created_by=SystemID.THYMOS,
            priority=0.8 if incident.severity == IncidentSeverity.CRITICAL else 0.6,
            decision_trace=DecisionTrace(
                reasoning=f"Thymos immune repair: {repair.reason}",
                alternatives_considered=[],
            ),
        )

        review = await self._equor.review(intent)

        if review.verdict.value == "approved":
            return ValidationResult(approved=True)
        elif review.verdict.value == "modified":
            return ValidationResult(
                approved=True,
                modifications={"equor_modifications": review.reasoning},
            )
        else:
            # Embed alignment scores so the caller (ThymosService) can feed them
            # into the drive pressure accumulator without re-querying Equor.
            alignment_snapshot = {
                "coherence": review.drive_alignment.coherence,
                "care": review.drive_alignment.care,
                "growth": review.drive_alignment.growth,
                "honesty": review.drive_alignment.honesty,
            }
            return ValidationResult(
                approved=False,
                reason=f"Equor denied repair: {review.reasoning}",
                escalate_to=RepairTier.ESCALATE,
                modifications={"equor_alignment": alignment_snapshot},
            )

    @property
    def repairs_this_hour(self) -> int:
        now_ts = utc_now().timestamp()
        hour_ago = now_ts - 3600.0
        return sum(1 for ts in self._recent_repairs if ts > hour_ago)

    @property
    def novel_repairs_today(self) -> int:
        now_ts = utc_now().timestamp()
        day_ago = now_ts - 86400.0
        return sum(1 for ts in self._recent_novel if ts > day_ago)
