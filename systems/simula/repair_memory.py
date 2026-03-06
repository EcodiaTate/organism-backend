"""
EcodiaOS — Simula Repair Memory  (Tasks 2, 4, 5)

Structured learning from every proposal outcome.

────────────────────────────────────────────────────────────────────────────
Task 2 — RepairMemory
────────────────────────────────────────────────────────────────────────────
After every proposal completes (APPLIED, ROLLED_BACK, or REJECTED):
  • Store an outcome record in Neo4j as (:RepairOutcome) node.
  • Build a success-rate model per (change_category, target_system).
  • Adjust risk thresholds dynamically based on historical success rate:
      success_rate > 80%  → lower simulation scrutiny (scrutiny_factor < 1.0)
      success_rate < 40%  → raise scrutiny, require extra verification
  • Expose summary via get_repair_memory_summary().

Neo4j schema
  (:RepairOutcome {
      proposal_id:          str   — ID of the EvolutionProposal
      change_category:      str   — ChangeCategory.value
      target_system:        str   — primary affected system name (may be "")
      risk_level_predicted: str   — RiskLevel.value from simulation
      risk_level_actual:    str   — "low"|"moderate"|"high" inferred from outcome
      systems_affected:     list  — all affected system names
      verification_passed:  bool  — whether post-apply health check passed
      rollback_needed:      bool  — whether automatic rollback was triggered
      time_to_verify_ms:    int   — wall-clock ms from apply-start to verify-end
      recorded_at:          str   — ISO-8601 UTC
  })

────────────────────────────────────────────────────────────────────────────
Task 4 — PostMortemHypothesis
────────────────────────────────────────────────────────────────────────────
After every rollback, analyse the failure and write a structured hypothesis
into Neo4j as (:PostMortemHypothesis) node, then emit EVO_REPAIR_POSTMORTEM
on the Synapse bus so Evo can treat it as high-confidence negative evidence.

Neo4j schema
  (:PostMortemHypothesis {
      id:                str   — new_id()
      proposal_id:       str
      change_category:   str
      target_system:     str
      failure_mode:      str   — "health_check_failed"|"wrong_category"|
                                 "risk_prediction_inaccurate"|"unknown"
      what_was_tried:    str   — brief description of the attempted change
      why_it_failed:     str   — structured diagnosis
      next_time_do:      str   — corrective recommendation
      rollback_reason:   str   — raw reason string from rollback
      confidence:        float — 0.0-1.0, how certain the diagnosis is
      recorded_at:       str
  })

────────────────────────────────────────────────────────────────────────────
Task 5 — Simulation Confidence Calibration
────────────────────────────────────────────────────────────────────────────
After verification completes, compare predicted_risk vs actual_outcome.
Track a calibration score over the last 20 proposals.

  calibration_score > 90% → INFO  (predictions trustworthy)
  calibration_score < 70% → emit SIMULA_CALIBRATION_DEGRADED event
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from systems.simula.evolution_types import RiskLevel

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.simula.evolution_types import EvolutionProposal
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("simula.repair_memory")

# Calibration window — last N proposals considered
_CALIBRATION_WINDOW = 20

# Dynamic threshold control
_HIGH_SUCCESS_RATE = 0.80  # above this → lower scrutiny
_LOW_SUCCESS_RATE = 0.40   # below this → raise scrutiny + extra verification

# Calibration alert thresholds
_CALIBRATION_DEGRADED_THRESHOLD = 0.70
_CALIBRATION_TRUSTWORTHY_THRESHOLD = 0.90


class RepairMemory:
    """
    Learns from every proposal outcome.

    Thread-safe: all mutable state is protected by self._lock.
    Neo4j is optional — degrades gracefully when unavailable.
    """

    def __init__(
        self,
        neo4j: Neo4jClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._event_bus = event_bus
        self._log = logger

        # Sentinel for escalating learning pipeline failures to Thymos
        from systems.synapse.sentinel import ErrorSentinel
        self._sentinel = ErrorSentinel("simula.repair_memory")

        # In-memory caches — rebuilt on first access or after each record
        # Key: (change_category, target_system) → [bool]  (True = success)
        self._outcome_cache: dict[tuple[str, str], list[bool]] = {}

        # Calibration window: list of (predicted_risk, was_accurate) booleans
        self._calibration_window: list[bool] = []

        self._record_count: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()

    # ─── Public API ────────────────────────────────────────────────────────────

    async def record_outcome(
        self,
        proposal: EvolutionProposal,
        *,
        verification_passed: bool,
        rollback_needed: bool,
        time_to_verify_ms: int = 0,
    ) -> None:
        """
        Store the outcome of a completed proposal.

        Called by SimulaService after every APPLIED, ROLLED_BACK, or REJECTED
        proposal (except governance-rejected ones where no change was attempted).
        """
        category = proposal.category.value
        affected = list(proposal.change_spec.affected_systems or [])
        target_system = affected[0] if affected else ""

        predicted_risk = (
            proposal.simulation.risk_level.value
            if proposal.simulation
            else RiskLevel.LOW.value
        )
        actual_risk = self._infer_actual_risk(
            predicted=predicted_risk,
            rollback_needed=rollback_needed,
            verification_passed=verification_passed,
        )

        async with self._lock:
            # Update in-memory success-rate cache
            key = (category, target_system)
            if key not in self._outcome_cache:
                self._outcome_cache[key] = []
            self._outcome_cache[key].append(verification_passed and not rollback_needed)
            # Keep last 100 outcomes per key (memory guard)
            if len(self._outcome_cache[key]) > 100:
                self._outcome_cache[key] = self._outcome_cache[key][-100:]

            # Update calibration window
            accurate = self._is_prediction_accurate(predicted_risk, rollback_needed)
            self._calibration_window.append(accurate)
            if len(self._calibration_window) > _CALIBRATION_WINDOW:
                self._calibration_window = self._calibration_window[-_CALIBRATION_WINDOW:]

            self._record_count += 1
            calibration_score = self._compute_calibration_score()

        self._log.info(
            "repair_outcome_recorded",
            proposal_id=proposal.id,
            category=category,
            target_system=target_system,
            verification_passed=verification_passed,
            rollback_needed=rollback_needed,
            predicted_risk=predicted_risk,
            actual_risk=actual_risk,
            time_to_verify_ms=time_to_verify_ms,
            calibration_score=round(calibration_score, 3),
        )

        # Write to Neo4j (non-blocking, best-effort)
        if self._neo4j is not None:
            try:
                await self._write_outcome_to_neo4j(
                    proposal_id=proposal.id,
                    change_category=category,
                    target_system=target_system,
                    risk_level_predicted=predicted_risk,
                    risk_level_actual=actual_risk,
                    systems_affected=affected,
                    verification_passed=verification_passed,
                    rollback_needed=rollback_needed,
                    time_to_verify_ms=time_to_verify_ms,
                )
            except Exception as exc:
                self._log.warning("repair_outcome_neo4j_write_failed", error=str(exc))
                await self._sentinel.report(
                    exc,
                    context={"operation": "record_outcome", "proposal_id": proposal.id},
                )

        # Calibration health checks
        await self._check_calibration(calibration_score, proposal.id)

    async def record_postmortem(
        self,
        proposal: EvolutionProposal,
        rollback_reason: str,
        verification_passed: bool,
    ) -> str:
        """
        Analyse a rollback failure and write a PostMortemHypothesis.

        Returns the postmortem ID.  Emits EVO_REPAIR_POSTMORTEM on the bus.
        """
        category = proposal.category.value
        affected = list(proposal.change_spec.affected_systems or [])
        target_system = affected[0] if affected else ""

        failure_mode, why_failed, next_time, confidence = self._diagnose_failure(
            proposal=proposal,
            rollback_reason=rollback_reason,
            verification_passed=verification_passed,
        )

        postmortem_id = new_id()
        recorded_at = utc_now().isoformat()

        self._log.info(
            "repair_postmortem_generated",
            postmortem_id=postmortem_id,
            proposal_id=proposal.id,
            category=category,
            target_system=target_system,
            failure_mode=failure_mode,
            confidence=confidence,
        )

        if self._neo4j is not None:
            try:
                await self._write_postmortem_to_neo4j(
                    postmortem_id=postmortem_id,
                    proposal_id=proposal.id,
                    change_category=category,
                    target_system=target_system,
                    failure_mode=failure_mode,
                    what_was_tried=proposal.description[:400],
                    why_it_failed=why_failed,
                    next_time_do=next_time,
                    rollback_reason=rollback_reason[:400],
                    confidence=confidence,
                    recorded_at=recorded_at,
                )
            except Exception as exc:
                self._log.warning("postmortem_neo4j_write_failed", error=str(exc))
                await self._sentinel.report(
                    exc,
                    context={"operation": "record_postmortem", "proposal_id": proposal.id},
                )

        # Emit EVO_REPAIR_POSTMORTEM on the Synapse bus
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                # EVO_REPAIR_POSTMORTEM is a new event type — add it to the
                # SynapseEventType enum in synapse/types.py (see wiring section).
                await self._event_bus.emit(
                    SynapseEvent(
                        event_type=SynapseEventType.EVO_REPAIR_POSTMORTEM,  # type: ignore[attr-defined]
                        source_system="simula",
                        data={
                            "postmortem_id": postmortem_id,
                            "proposal_id": proposal.id,
                            "change_category": category,
                            "target_system": target_system,
                            "failure_mode": failure_mode,
                            "why_it_failed": why_failed,
                            "next_time_do": next_time,
                            "confidence": confidence,
                        },
                    )
                )
            except Exception as exc:
                self._log.warning("postmortem_bus_emit_failed", error=str(exc))

        return postmortem_id

    async def record_axon_repair(
        self,
        *,
        source: str,
        success: bool,
        fix_type: str,
        incident_class: str,
        elapsed_ms: int,
    ) -> None:
        """Record an Axon executor repair outcome for calibration learning.

        Axon repair executors (cognitive_stall_repair, simula_codegen_repair,
        etc.) emit REPAIR_COMPLETED events.  Recording their outcomes lets
        Simula adjust scrutiny when proposing changes that target the same
        systems Axon is also repairing.
        """
        category = "axon_repair"
        target_system = incident_class or fix_type

        async with self._lock:
            key = (category, target_system)
            if key not in self._outcome_cache:
                self._outcome_cache[key] = []
            self._outcome_cache[key].append(success)
            if len(self._outcome_cache[key]) > 100:
                self._outcome_cache[key] = self._outcome_cache[key][-100:]

        self._log.info(
            "axon_repair_outcome_recorded",
            source=source,
            success=success,
            fix_type=fix_type,
            incident_class=incident_class,
            elapsed_ms=elapsed_ms,
        )

    def get_scrutiny_factor(self, category: str, target_system: str) -> float:
        """
        Return a scrutiny multiplier for the given (category, target_system).

        > 1.0  → increase simulation scrutiny (failed category/system)
        = 1.0  → baseline
        < 1.0  → lower scrutiny (proven-reliable category/system)

        Used by SimulaService to adjust simulation thresholds per Task 2.
        """
        key = (category, target_system)
        outcomes = self._outcome_cache.get(key, [])
        if len(outcomes) < 5:
            # Not enough data — baseline
            return 1.0

        success_rate = sum(outcomes) / len(outcomes)
        if success_rate > _HIGH_SUCCESS_RATE:
            # 80–100% success → reduce scrutiny (e.g. 0.7 at 100%)
            return max(0.7, 1.0 - (success_rate - _HIGH_SUCCESS_RATE) * 1.5)
        if success_rate < _LOW_SUCCESS_RATE:
            # <40% success → raise scrutiny (up to 1.5 at 0%)
            return min(1.5, 1.0 + (_LOW_SUCCESS_RATE - success_rate) * 1.25)
        return 1.0

    def get_calibration_score(self) -> float:
        """Return current prediction accuracy score (0.0–1.0)."""
        return self._compute_calibration_score()

    async def get_repair_memory_summary(self) -> dict[str, Any]:
        """
        Build the summary payload for GET /simula/repair-memory.

        Queries Neo4j for totals; falls back to in-memory cache on failure.
        """
        total_proposals = 0
        total_rolled_back = 0

        if self._neo4j is not None:
            try:
                result = await self._neo4j.execute_read(
                    """
                    MATCH (r:RepairOutcome)
                    RETURN count(r) AS total,
                           sum(CASE WHEN r.rollback_needed THEN 1 ELSE 0 END) AS rolled_back
                    """
                )
                if result:
                    row = result[0]
                    total_proposals = int(row.get("total") or 0)
                    total_rolled_back = int(row.get("rolled_back") or 0)
            except Exception as exc:
                self._log.warning("repair_memory_neo4j_read_failed", error=str(exc))
                total_proposals = self._record_count

        rollback_rate = (
            total_rolled_back / total_proposals if total_proposals > 0 else 0.0
        )

        # Build per-(category, system) success rates
        async with self._lock:
            success_rates: dict[str, float] = {}
            for (cat, sys), outcomes in self._outcome_cache.items():
                if outcomes:
                    key_str = f"{cat}:{sys}" if sys else cat
                    success_rates[key_str] = round(sum(outcomes) / len(outcomes), 3)

        most_reliable = max(success_rates, key=lambda k: success_rates[k], default="")
        most_risky = min(success_rates, key=lambda k: success_rates[k], default="")

        return {
            "success_rates_by_category": success_rates,
            "total_proposals": total_proposals,
            "rollback_rate": round(rollback_rate, 4),
            "most_reliable_change_type": most_reliable,
            "most_risky_change_type": most_risky,
            "calibration_score": round(self.get_calibration_score(), 3),
            "calibration_window_size": len(self._calibration_window),
        }

    @property
    def record_count(self) -> int:
        """Total outcomes recorded (in-memory count)."""
        return self._record_count

    # ─── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _infer_actual_risk(
        predicted: str,
        rollback_needed: bool,
        verification_passed: bool,
    ) -> str:
        """Infer actual risk level from the outcome."""
        if rollback_needed:
            return RiskLevel.HIGH.value
        if not verification_passed:
            return RiskLevel.MODERATE.value
        return RiskLevel.LOW.value

    @staticmethod
    def _is_prediction_accurate(predicted_risk: str, rollback_needed: bool) -> bool:
        """
        Was the risk prediction accurate?

        LOW predicted + no rollback → accurate.
        HIGH/UNACCEPTABLE predicted + rollback → accurate (correct caution).
        LOW predicted + rollback → inaccurate (under-predicted risk).
        HIGH predicted + no rollback → inaccurate (over-predicted risk).
        MODERATE is flexible — accurate either way.
        """
        if predicted_risk == RiskLevel.MODERATE.value:
            return True  # Moderate is always "acceptable" prediction
        if rollback_needed:
            return predicted_risk in (RiskLevel.HIGH.value, RiskLevel.UNACCEPTABLE.value)
        return predicted_risk == RiskLevel.LOW.value

    def _compute_calibration_score(self) -> float:
        """Fraction of accurate predictions in the calibration window."""
        if not self._calibration_window:
            return 1.0  # No data — assume calibrated
        return sum(self._calibration_window) / len(self._calibration_window)

    @staticmethod
    def _diagnose_failure(
        proposal: EvolutionProposal,
        rollback_reason: str,
        verification_passed: bool,
    ) -> tuple[str, str, str, float]:
        """
        Return (failure_mode, why_failed, next_time_do, confidence).
        """
        reason_lower = rollback_reason.lower()

        if not verification_passed or "health" in reason_lower:
            return (
                "health_check_failed",
                (
                    f"Post-apply health check failed for "
                    f"'{proposal.category.value}' on "
                    f"{proposal.change_spec.affected_systems}. "
                    f"Rollback reason: {rollback_reason[:200]}"
                ),
                (
                    "Before applying this category again on the same system, "
                    "run an isolated dry-run in a shadow environment and "
                    "expand the verification checks."
                ),
                0.80,
            )

        if "risk" in reason_lower or "unacceptable" in reason_lower:
            predicted = (
                proposal.simulation.risk_level.value if proposal.simulation else "unknown"
            )
            return (
                "risk_prediction_inaccurate",
                (
                    f"Predicted risk was '{predicted}' but the actual outcome "
                    f"required rollback. The simulation model may have "
                    f"underestimated blast radius for "
                    f"'{proposal.category.value}'."
                ),
                (
                    "Increase counterfactual episode count for this category. "
                    "Require formal verification (Dafny/Z3) before applying."
                ),
                0.65,
            )

        if "category" in reason_lower or "forbidden" in reason_lower:
            return (
                "wrong_category",
                (
                    f"Proposal category '{proposal.category.value}' was "
                    f"inappropriate for the target change. "
                    f"Rollback reason: {rollback_reason[:200]}"
                ),
                (
                    "Re-classify the change under the correct ChangeCategory "
                    "before re-proposing. Review the GOVERNANCE_REQUIRED set."
                ),
                0.75,
            )

        return (
            "unknown",
            (
                f"Rollback triggered with reason: {rollback_reason[:300]}. "
                f"Manual inspection required for proposal {proposal.id}."
            ),
            "Investigate the rollback logs and add a specific failure mode "
            "to RepairMemory._diagnose_failure() once the root cause is clear.",
            0.40,
        )

    async def _check_calibration(
        self, calibration_score: float, proposal_id: str
    ) -> None:
        """Emit calibration health events based on current score."""
        if calibration_score < _CALIBRATION_DEGRADED_THRESHOLD:
            self._log.warning(
                "simula_calibration_degraded",
                calibration_score=round(calibration_score, 3),
                window_size=len(self._calibration_window),
                threshold=_CALIBRATION_DEGRADED_THRESHOLD,
                proposal_id=proposal_id,
            )
            if self._event_bus is not None:
                try:
                    from systems.synapse.types import SynapseEvent, SynapseEventType

                    await self._event_bus.emit(
                        SynapseEvent(
                            event_type=SynapseEventType.SIMULA_CALIBRATION_DEGRADED,  # type: ignore[attr-defined]
                            source_system="simula",
                            data={
                                "calibration_score": calibration_score,
                                "window_size": len(self._calibration_window),
                                "threshold": _CALIBRATION_DEGRADED_THRESHOLD,
                                "recent_proposal_id": proposal_id,
                            },
                        )
                    )
                except Exception as exc:
                    self._log.warning("calibration_event_emit_failed", error=str(exc))
        elif calibration_score > _CALIBRATION_TRUSTWORTHY_THRESHOLD:
            self._log.info(
                "simula_calibration_trustworthy",
                calibration_score=round(calibration_score, 3),
                window_size=len(self._calibration_window),
            )

    async def get_repair_guidance_prompt(
        self,
        category: str,
        target_system: str,
    ) -> str:
        """
        Build a prompt section describing past repair outcomes for injection
        into the code agent's system prompt.

        Includes:
          - Success rate for this (category, system) pair
          - Recent postmortem lessons (what failed and why)
          - Scrutiny factor guidance
        """
        if self._neo4j is None:
            return ""

        lines: list[str] = []

        try:
            # Query recent successful repairs for this category+system
            successes = await self._neo4j.execute_read(
                """
                MATCH (r:RepairOutcome)
                WHERE r.change_category = $category
                  AND ($system = "" OR r.target_system = $system)
                  AND r.verification_passed = true
                  AND r.rollback_needed = false
                RETURN r.proposal_id AS pid, r.recorded_at AS ts
                ORDER BY r.recorded_at DESC
                LIMIT 5
                """,
                parameters={"category": category, "system": target_system},
            )

            # Query recent postmortems (failures) for this category+system
            postmortems = await self._neo4j.execute_read(
                """
                MATCH (p:PostMortemHypothesis)
                WHERE p.change_category = $category
                  AND ($system = "" OR p.target_system = $system)
                  AND p.confidence >= 0.5
                RETURN p.what_was_tried AS tried,
                       p.why_it_failed AS failed,
                       p.next_time_do AS recommendation,
                       p.confidence AS confidence
                ORDER BY p.recorded_at DESC
                LIMIT 5
                """,
                parameters={"category": category, "system": target_system},
            )

            if not successes and not postmortems:
                return ""

            lines.append("## Repair Memory — Lessons from Past Proposals")
            lines.append("")

            # Scrutiny factor
            scrutiny = self.get_scrutiny_factor(category, target_system)
            if scrutiny != 1.0:
                direction = "elevated" if scrutiny > 1.0 else "reduced"
                lines.append(
                    f"**Scrutiny level**: {direction} ({scrutiny:.2f}x) — "
                    f"{'be extra careful, past repairs have failed often' if scrutiny > 1.0 else 'this category has a good track record'}."
                )
                lines.append("")

            # Success context
            if successes:
                lines.append(
                    f"**{len(successes)} recent successful repairs** for "
                    f"`{category}` on `{target_system or 'any system'}`. "
                    f"This category is known to work."
                )
                lines.append("")

            # Postmortem lessons — the most valuable part
            if postmortems:
                lines.append("**Past failures — DO NOT repeat these mistakes:**")
                for pm in postmortems:
                    tried = pm.get("tried", "unknown")
                    failed = pm.get("failed", "unknown")
                    recommendation = pm.get("recommendation", "")
                    confidence = pm.get("confidence", 0.0)
                    lines.append(
                        f"- [{confidence:.0%} confidence] Tried: {tried[:120]}. "
                        f"Failed because: {failed[:120]}. "
                    )
                    if recommendation:
                        lines.append(f"  **Instead**: {recommendation[:150]}")
                lines.append("")

        except Exception as exc:
            self._log.warning(
                "repair_guidance_prompt_failed",
                error=str(exc),
                category=category,
                target_system=target_system,
            )

        return "\n".join(lines)

    async def _write_outcome_to_neo4j(
        self,
        *,
        proposal_id: str,
        change_category: str,
        target_system: str,
        risk_level_predicted: str,
        risk_level_actual: str,
        systems_affected: list[str],
        verification_passed: bool,
        rollback_needed: bool,
        time_to_verify_ms: int,
    ) -> None:
        """
        Neo4j write — (:RepairOutcome) node.

        Schema:
          (:RepairOutcome {
              proposal_id, change_category, target_system,
              risk_level_predicted, risk_level_actual,
              systems_affected (JSON list as string),
              verification_passed, rollback_needed,
              time_to_verify_ms, recorded_at
          })
        """
        assert self._neo4j is not None
        import json as _json

        await self._neo4j.execute_write(
            """
            CREATE (r:RepairOutcome {
                proposal_id:          $proposal_id,
                change_category:      $change_category,
                target_system:        $target_system,
                risk_level_predicted: $risk_level_predicted,
                risk_level_actual:    $risk_level_actual,
                systems_affected:     $systems_affected,
                verification_passed:  $verification_passed,
                rollback_needed:      $rollback_needed,
                time_to_verify_ms:    $time_to_verify_ms,
                recorded_at:          $recorded_at
            })
            """,
            parameters={
                "proposal_id": proposal_id,
                "change_category": change_category,
                "target_system": target_system,
                "risk_level_predicted": risk_level_predicted,
                "risk_level_actual": risk_level_actual,
                "systems_affected": _json.dumps(systems_affected),
                "verification_passed": verification_passed,
                "rollback_needed": rollback_needed,
                "time_to_verify_ms": time_to_verify_ms,
                "recorded_at": utc_now().isoformat(),
            },
        )

    async def _write_postmortem_to_neo4j(
        self,
        *,
        postmortem_id: str,
        proposal_id: str,
        change_category: str,
        target_system: str,
        failure_mode: str,
        what_was_tried: str,
        why_it_failed: str,
        next_time_do: str,
        rollback_reason: str,
        confidence: float,
        recorded_at: str,
    ) -> None:
        """
        Neo4j write — (:PostMortemHypothesis) node.

        Schema:
          (:PostMortemHypothesis {
              id, proposal_id, change_category, target_system,
              failure_mode, what_was_tried, why_it_failed,
              next_time_do, rollback_reason, confidence, recorded_at
          })
        Linked to the originating proposal via:
          (:EvolutionRecord)-[:HAD_POSTMORTEM]->(:PostMortemHypothesis)
        (link written here if the record already exists)
        """
        assert self._neo4j is not None

        await self._neo4j.execute_write(
            """
            CREATE (p:PostMortemHypothesis {
                id:              $id,
                proposal_id:     $proposal_id,
                change_category: $change_category,
                target_system:   $target_system,
                failure_mode:    $failure_mode,
                what_was_tried:  $what_was_tried,
                why_it_failed:   $why_it_failed,
                next_time_do:    $next_time_do,
                rollback_reason: $rollback_reason,
                confidence:      $confidence,
                recorded_at:     $recorded_at
            })
            WITH p
            MATCH (r:EvolutionRecord {proposal_id: $proposal_id})
            MERGE (r)-[:HAD_POSTMORTEM]->(p)
            """,
            parameters={
                "id": postmortem_id,
                "proposal_id": proposal_id,
                "change_category": change_category,
                "target_system": target_system,
                "failure_mode": failure_mode,
                "what_was_tried": what_was_tried,
                "why_it_failed": why_it_failed,
                "next_time_do": next_time_do,
                "rollback_reason": rollback_reason,
                "confidence": confidence,
                "recorded_at": recorded_at,
            },
        )
