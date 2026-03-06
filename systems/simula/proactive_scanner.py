"""
EcodiaOS — Simula Proactive Scanner  (Tasks 1 & 3)

────────────────────────────────────────────────────────────────────────────
Task 1 — ProactiveScanner
────────────────────────────────────────────────────────────────────────────
Simula hunts for pathology rather than waiting for Thymos to escalate to T4.

Runs every 5 minutes (configurable via SIMULA_PROACTIVE_INTERVAL_S env var).
On each scan cycle:
  1. Reads the last 50 Thymos (:Incident) nodes from Neo4j.
  2. Detects three pattern types:
       REPEAT_FAILURE   — same source_system appears ≥3 times in 50 incidents
       RECURRING_ERROR  — same incident_class appears ≥4 times in 50 incidents
       FALSE_RESOLUTION — same fingerprint was "resolved" T1/T2 but re-appeared
  3. For each detected pattern, generates an EvolutionProposal with
     source="simula_proactive" and submits it to the normal pipeline
     (process_proposal is passed in as a callable).
  4. Respects SIMULA_PROACTIVE_DRY_RUN=true — logs proposals but never submits.
  5. Logs every scan: patterns_detected, proposals_generated,
     proposals_approved, proposals_applied.

────────────────────────────────────────────────────────────────────────────
Task 3 — GoalAuditor
────────────────────────────────────────────────────────────────────────────
Nova accumulates "Monitor system recovery" maintenance goals indefinitely.
The GoalAuditor cleans them up every 10 minutes.

  • Queries (:Goal) nodes where source="maintenance" older than 30 minutes.
  • Checks whether the monitored system is currently healthy (reads the last
    Incident for that system — if the most recent is repair_successful=true
    and >10 min ago, the system is considered healthy).
  • Marks stale maintenance goals ACHIEVED in Neo4j.
  • Emits GOAL_HYGIENE_COMPLETE on the Synapse bus.
  • Never touches goals with source != "maintenance".

────────────────────────────────────────────────────────────────────────────
Configuration (env vars)
────────────────────────────────────────────────────────────────────────────
SIMULA_PROACTIVE_DRY_RUN=true    — log proposals but do not submit (default: false)
SIMULA_PROACTIVE_INTERVAL_S=300  — seconds between ProactiveScanner runs (default: 300)
SIMULA_GOAL_AUDIT_INTERVAL_S=600 — seconds between GoalAuditor runs (default: 600)
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from systems.simula.evolution_types import (
    ChangeCategory,
    ChangeSpec,
    EvolutionProposal,
    ProposalResult,
    ProposalStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from clients.neo4j import Neo4jClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("simula.proactive_scanner")

# ── Tunables ──────────────────────────────────────────────────────────────────

_INCIDENT_WINDOW = 50                    # How many recent incidents to inspect
_REPEAT_FAILURE_MIN = 3                  # Min occurrences for REPEAT_FAILURE pattern
_RECURRING_ERROR_MIN = 4                 # Min occurrences for RECURRING_ERROR pattern
_MAINTENANCE_GOAL_AGE_MINUTES = 30       # Goals older than this are candidates
_SYSTEM_HEALTHY_QUIESCE_MINUTES = 10     # No failure for this long = system healthy

# ── Pattern detection result ──────────────────────────────────────────────────

class _Pattern:
    """Internal DTO for a detected pathology pattern."""

    __slots__ = (
        "pattern_type", "affected_system", "incident_class",
        "occurrence_count", "description", "evidence"
    )

    def __init__(
        self,
        pattern_type: str,          # "REPEAT_FAILURE" | "RECURRING_ERROR" | "FALSE_RESOLUTION"
        affected_system: str,
        incident_class: str,
        occurrence_count: int,
        description: str,
        evidence: list[str],        # incident IDs
    ) -> None:
        self.pattern_type = pattern_type
        self.affected_system = affected_system
        self.incident_class = incident_class
        self.occurrence_count = occurrence_count
        self.description = description
        self.evidence = evidence


class ProactiveScannerStats:
    """Mutable stats for the last scan and cumulative totals."""

    def __init__(self) -> None:
        self.last_scan_at: datetime | None = None
        self.last_patterns_detected: int = 0
        self.last_proposals_generated: int = 0
        self.last_proposals_approved: int = 0
        self.last_proposals_applied: int = 0
        self.total_scans: int = 0
        self.total_patterns_detected: int = 0
        self.total_proposals_generated: int = 0
        self.scanner_alive: bool = False
        self.dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_scan_at": self.last_scan_at.isoformat() if self.last_scan_at else None,
            "last_patterns_detected": self.last_patterns_detected,
            "last_proposals_generated": self.last_proposals_generated,
            "last_proposals_approved": self.last_proposals_approved,
            "last_proposals_applied": self.last_proposals_applied,
            "total_scans": self.total_scans,
            "total_patterns_detected": self.total_patterns_detected,
            "total_proposals_generated": self.total_proposals_generated,
            "scanner_alive": self.scanner_alive,
            "dry_run": self.dry_run,
        }


class ProactiveScanner:
    """
    Self-healing engine: hunts for systemic pathology every N seconds.

    Pass the SimulaService.process_proposal coroutine factory as
    ``process_proposal_fn``.  The scanner calls it for each generated
    proposal, so it goes through the normal 7-stage pipeline.

    The GoalAuditor is embedded here (Task 3) — it runs on a separate
    (slower) timer within the same supervision loop.
    """

    def __init__(
        self,
        neo4j: Neo4jClient | None,
        event_bus: EventBus | None,
        process_proposal_fn: Callable[[EvolutionProposal], Coroutine[Any, Any, ProposalResult]],
        *,
        scan_interval_s: float | None = None,
        goal_audit_interval_s: float | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._event_bus = event_bus
        self._process = process_proposal_fn
        self._log = logger

        self._scan_interval_s: float = float(
            scan_interval_s
            or os.environ.get("SIMULA_PROACTIVE_INTERVAL_S", "300")
        )
        self._goal_audit_interval_s: float = float(
            goal_audit_interval_s
            or os.environ.get("SIMULA_GOAL_AUDIT_INTERVAL_S", "600")
        )
        self._dry_run: bool = (
            os.environ.get("SIMULA_PROACTIVE_DRY_RUN", "false").lower() == "true"
        )

        self.stats = ProactiveScannerStats()
        self.stats.dry_run = self._dry_run

        # Track which (source_system, incident_class) pairs we already
        # proposed for — avoid flooding the pipeline with identical proposals
        # across consecutive scans.
        self._proposed_patterns: set[str] = set()
        self._pattern_proposal_lock: asyncio.Lock = asyncio.Lock()

        # GoalAuditor timer state
        self._last_goal_audit_at: datetime | None = None

    # ─── Main supervision loop ────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """
        Infinite loop: scan every N seconds; audit goals every M seconds.

        Designed to be wrapped in supervised_task() — the outer supervisor
        handles restarts, so this loop exits cleanly on CancelledError.
        """
        self.stats.scanner_alive = True
        self._log.info(
            "proactive_scanner_started",
            interval_s=self._scan_interval_s,
            goal_audit_interval_s=self._goal_audit_interval_s,
            dry_run=self._dry_run,
        )

        try:
            while True:
                # ProactiveScanner — Task 1
                await self._run_scan()

                # GoalAuditor — Task 3 (runs every goal_audit_interval_s)
                now = utc_now()
                if (
                    self._last_goal_audit_at is None
                    or (now - self._last_goal_audit_at).total_seconds()
                    >= self._goal_audit_interval_s
                ):
                    await self._run_goal_audit()
                    self._last_goal_audit_at = utc_now()

                await asyncio.sleep(self._scan_interval_s)
        except asyncio.CancelledError:
            self.stats.scanner_alive = False
            self._log.info("proactive_scanner_stopped")
            raise
        except Exception as exc:
            # Unexpected exception in proactive scanner — emit to Thymos
            # The immune system must be able to heal itself
            self.stats.scanner_alive = False
            self._log.error(
                "proactive_scanner_exception",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            await self._emit_scanner_incident(
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            # Re-raise so supervisor can restart the scanner
            raise

    # ─── Task 1: Proactive scan ───────────────────────────────────────────────

    async def _run_scan(self) -> None:
        """Single scan cycle — detect patterns, generate proposals."""
        if self._neo4j is None:
            self._log.warning("proactive_scanner_no_neo4j_skipping")
            return

        incidents = await self._fetch_recent_incidents()
        if not incidents:
            self._log.debug("proactive_scanner_no_incidents")
            return

        patterns = self._detect_patterns(incidents)

        proposals_generated = 0
        proposals_approved = 0
        proposals_applied = 0

        for pattern in patterns:
            pattern_key = (
                f"{pattern.pattern_type}:{pattern.affected_system}:{pattern.incident_class}"
            )
            async with self._pattern_proposal_lock:
                if pattern_key in self._proposed_patterns:
                    self._log.debug(
                        "proactive_scanner_pattern_already_proposed",
                        pattern_key=pattern_key,
                    )
                    continue
                self._proposed_patterns.add(pattern_key)

            proposal = self._build_proposal(pattern)
            proposals_generated += 1

            if self._dry_run:
                self._log.info(
                    "proactive_scanner_dry_run_proposal",
                    proposal_id=proposal.id,
                    category=proposal.category.value,
                    description=proposal.description[:120],
                    pattern_type=pattern.pattern_type,
                    affected_system=pattern.affected_system,
                    occurrences=pattern.occurrence_count,
                )
                continue

            try:
                result = await self._process(proposal)
                if result.status == ProposalStatus.APPLIED:
                    proposals_applied += 1
                    proposals_approved += 1
                elif result.status in (
                    ProposalStatus.APPROVED,
                    ProposalStatus.AWAITING_GOVERNANCE,
                ):
                    proposals_approved += 1

                self._log.info(
                    "proactive_scanner_proposal_submitted",
                    proposal_id=proposal.id,
                    result_status=result.status.value,
                    pattern_type=pattern.pattern_type,
                )
            except Exception as exc:
                self._log.warning(
                    "proactive_scanner_proposal_failed",
                    error=str(exc),
                    pattern_type=pattern.pattern_type,
                    affected_system=pattern.affected_system,
                )
                # Remove from cache so it can be retried next scan
                async with self._pattern_proposal_lock:
                    self._proposed_patterns.discard(pattern_key)

        # Prune the pattern cache to avoid unbounded growth
        # (keep only the 200 most recent patterns)
        async with self._pattern_proposal_lock:
            if len(self._proposed_patterns) > 200:
                # Discard the oldest half (set has no ordering, so just shrink)
                excess = list(self._proposed_patterns)[:100]
                for k in excess:
                    self._proposed_patterns.discard(k)

        # Update stats
        now = utc_now()
        self.stats.last_scan_at = now
        self.stats.last_patterns_detected = len(patterns)
        self.stats.last_proposals_generated = proposals_generated
        self.stats.last_proposals_approved = proposals_approved
        self.stats.last_proposals_applied = proposals_applied
        self.stats.total_scans += 1
        self.stats.total_patterns_detected += len(patterns)
        self.stats.total_proposals_generated += proposals_generated

        self._log.info(
            "proactive_scan_complete",
            patterns_detected=len(patterns),
            proposals_generated=proposals_generated,
            proposals_approved=proposals_approved,
            proposals_applied=proposals_applied,
            dry_run=self._dry_run,
        )

    async def _fetch_recent_incidents(self) -> list[dict[str, Any]]:
        """
        Read the last _INCIDENT_WINDOW Thymos (:Incident) nodes from Neo4j.

        Returns list of dicts with keys:
          id, source_system, incident_class, fingerprint,
          repair_successful, repair_tier, timestamp
        """
        assert self._neo4j is not None
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (i:Incident)
                WHERE i.timestamp IS NOT NULL
                RETURN i.id              AS id,
                       i.source_system   AS source_system,
                       i.incident_class  AS incident_class,
                       i.fingerprint     AS fingerprint,
                       i.repair_tier     AS repair_tier,
                       i.repair_successful AS repair_successful,
                       i.timestamp       AS timestamp
                ORDER BY i.timestamp DESC
                LIMIT $limit
                """,
                parameters={"limit": _INCIDENT_WINDOW},
            )
            return [dict(r) for r in rows]
        except Exception as exc:
            self._log.warning("proactive_scanner_fetch_incidents_failed", error=str(exc))
            return []

    @staticmethod
    def _detect_patterns(incidents: list[dict[str, Any]]) -> list[_Pattern]:
        """
        Detect pathology patterns from the incident list.

        Three pattern types:
          REPEAT_FAILURE   — same source_system fails ≥ threshold
          RECURRING_ERROR  — same incident_class appears ≥ threshold
          FALSE_RESOLUTION — same fingerprint was "resolved" T1/T2 but re-appeared
        """
        patterns: list[_Pattern] = []
        by_system: Counter[str] = Counter()
        by_class: Counter[str] = Counter()
        fingerprint_tiers: dict[str, list[str]] = {}  # fingerprint → [tier, ...]

        system_incidents: dict[str, list[str]] = {}   # system → [incident_id, ...]
        class_incidents: dict[str, list[str]] = {}    # class → [incident_id, ...]

        for inc in incidents:
            sys = inc.get("source_system") or ""
            cls = inc.get("incident_class") or ""
            fp = inc.get("fingerprint") or ""
            tier = inc.get("repair_tier") or "unknown"
            inc_id = inc.get("id") or ""

            if sys:
                by_system[sys] += 1
                system_incidents.setdefault(sys, []).append(inc_id)
            if cls:
                by_class[cls] += 1
                class_incidents.setdefault(cls, []).append(inc_id)
            if fp:
                fingerprint_tiers.setdefault(fp, []).append(tier)

        # Pattern 1: REPEAT_FAILURE
        for sys, count in by_system.items():
            if count >= _REPEAT_FAILURE_MIN:
                patterns.append(_Pattern(
                    pattern_type="REPEAT_FAILURE",
                    affected_system=sys,
                    incident_class="",
                    occurrence_count=count,
                    description=(
                        f"System '{sys}' has failed {count} times in the last "
                        f"{_INCIDENT_WINDOW} incidents. Repeated failures suggest "
                        f"a structural defect that self-repair has not resolved."
                    ),
                    evidence=system_incidents.get(sys, [])[:10],
                ))

        # Pattern 2: RECURRING_ERROR
        for cls, count in by_class.items():
            if count >= _RECURRING_ERROR_MIN:
                # Find which system is most associated with this class
                affected = Counter(
                    inc.get("source_system", "")
                    for inc in incidents
                    if inc.get("incident_class") == cls
                )
                primary_sys = affected.most_common(1)[0][0] if affected else ""
                patterns.append(_Pattern(
                    pattern_type="RECURRING_ERROR",
                    affected_system=primary_sys,
                    incident_class=cls,
                    occurrence_count=count,
                    description=(
                        f"Incident class '{cls}' has recurred {count} times "
                        f"in the last {_INCIDENT_WINDOW} incidents (primary system: "
                        f"'{primary_sys}'). A recurring error class signals a "
                        f"systemic root cause not addressed by tier repair."
                    ),
                    evidence=class_incidents.get(cls, [])[:10],
                ))

        # Pattern 3: FALSE_RESOLUTION
        # A fingerprint is "falsely resolved" if it appeared with a T1/T2 tier
        # and then appeared again (meaning the resolution didn't stick).
        low_tier_names = {"NOOP", "PARAMETER", "RESTART", "KNOWN_FIX", "T1", "T2"}
        for fp, tiers in fingerprint_tiers.items():
            # Need at least 2 occurrences, with at least one being a low tier
            if len(tiers) >= 2 and any(t.upper() in low_tier_names for t in tiers):
                # Find matching incidents
                fp_incidents = [
                    inc.get("id", "") for inc in incidents
                    if inc.get("fingerprint") == fp
                ]
                # Infer a system from the first matched incident
                first_match = next(
                    (inc for inc in incidents if inc.get("fingerprint") == fp), {}
                )
                sys = first_match.get("source_system", "")
                cls = first_match.get("incident_class", "")
                patterns.append(_Pattern(
                    pattern_type="FALSE_RESOLUTION",
                    affected_system=sys,
                    incident_class=cls,
                    occurrence_count=len(tiers),
                    description=(
                        f"Fingerprint '{fp[:16]}...' (system: '{sys}', class: '{cls}') "
                        f"was marked resolved at tier {tiers[0]} but has recurred "
                        f"{len(tiers)} times. The tier repair did not address the "
                        f"root cause — structural intervention is needed."
                    ),
                    evidence=fp_incidents[:10],
                ))

        return patterns

    @staticmethod
    def _build_proposal(pattern: _Pattern) -> EvolutionProposal:
        """
        Construct an EvolutionProposal for a detected pathology pattern.

        Selects the most appropriate ChangeCategory based on pattern type.
        """
        # Map pattern → most appropriate change category
        category_map = {
            "REPEAT_FAILURE": ChangeCategory.ADD_SYSTEM_CAPABILITY,
            "RECURRING_ERROR": ChangeCategory.ADD_PATTERN_DETECTOR,
            "FALSE_RESOLUTION": ChangeCategory.MODIFY_CONTRACT,
        }
        category = category_map.get(pattern.pattern_type, ChangeCategory.ADD_SYSTEM_CAPABILITY)

        affected = [pattern.affected_system] if pattern.affected_system else []

        is_capability = category == ChangeCategory.ADD_SYSTEM_CAPABILITY
        is_detector = category == ChangeCategory.ADD_PATTERN_DETECTOR
        spec = ChangeSpec(
            capability_description=pattern.description if is_capability else None,
            detector_description=pattern.description if is_detector else None,
            affected_systems=affected,
            additional_context=(
                f"Pattern: {pattern.pattern_type}. "
                f"Occurrences: {pattern.occurrence_count}. "
                f"Evidence incident IDs: {', '.join(pattern.evidence[:5])}."
            ),
        )

        return EvolutionProposal(
            id=new_id(),
            source="simula_proactive",
            category=category,
            description=pattern.description[:500],
            change_spec=spec,
            evidence=pattern.evidence[:10],
            expected_benefit=(
                f"Eliminating recurring {pattern.pattern_type.lower().replace('_', ' ')} "
                f"in system '{pattern.affected_system}' will reduce Thymos T4 escalations "
                f"and restore self-healing reliability."
            ),
            risk_assessment=(
                f"Proactively generated by SimulaScanner from {pattern.occurrence_count} "
                f"observed incidents. Simulation scrutiny should be standard."
            ),
        )

    # ─── Task 3: Goal Auditor ────────────────────────────────────────────────

    async def _run_goal_audit(self) -> None:
        """
        Find and retire stale maintenance goals in Nova's goal store.

        Never touches goals with source != "maintenance".
        """
        if self._neo4j is None:
            self._log.warning("goal_auditor_no_neo4j_skipping")
            return

        cutoff = (
            utc_now() - timedelta(minutes=_MAINTENANCE_GOAL_AGE_MINUTES)
        ).isoformat()

        try:
            # Fetch maintenance goals older than the cutoff that are still ACTIVE
            rows = await self._neo4j.execute_read(
                """
                MATCH (g:Goal)
                WHERE g.source = 'maintenance'
                  AND g.status = 'active'
                  AND g.created_at < $cutoff
                RETURN g.id          AS id,
                       g.description AS description,
                       g.created_at  AS created_at
                """,
                parameters={"cutoff": cutoff},
            )
        except Exception as exc:
            self._log.warning("goal_auditor_fetch_failed", error=str(exc))
            return

        if not rows:
            self._log.debug("goal_auditor_no_stale_goals")
            return

        removed = 0
        for row in rows:
            goal_id = row.get("id") or ""
            description = row.get("description") or ""
            if not goal_id:
                continue

            # Extract the monitored system from the description.
            # Thymos writes: "Monitor system recovery: <system> after <tier> repair"
            monitored_system = self._extract_system_from_maintenance_goal(description)

            if monitored_system and not await self._is_system_healthy(monitored_system):
                # System still unhealthy — leave the goal active
                continue

            # Mark goal ACHIEVED in Neo4j
            try:
                await self._neo4j.execute_write(
                    """
                    MATCH (g:Goal {id: $id})
                    SET g.status = 'achieved',
                        g.updated_at = $now
                    """,
                    parameters={"id": goal_id, "now": utc_now().isoformat()},
                )
                removed += 1
                self._log.debug(
                    "goal_auditor_retired_stale_goal",
                    goal_id=goal_id,
                    monitored_system=monitored_system,
                    description=description[:80],
                )
            except Exception as exc:
                self._log.warning(
                    "goal_auditor_retire_failed",
                    goal_id=goal_id,
                    error=str(exc),
                )

        # Count remaining active goals for the event payload
        try:
            remaining_rows = await self._neo4j.execute_read(
                "MATCH (g:Goal) WHERE g.status = 'active' RETURN count(g) AS n"
            )
            remaining = int(remaining_rows[0].get("n") or 0) if remaining_rows else 0
        except Exception:
            remaining = -1

        self._log.info(
            "goal_hygiene_complete",
            stale_goals_removed=removed,
            active_goals_remaining=remaining,
        )

        if self._event_bus is not None and removed > 0:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(
                    SynapseEvent(
                        event_type=SynapseEventType.GOAL_HYGIENE_COMPLETE,  # type: ignore[attr-defined]
                        source_system="simula",
                        data={
                            "stale_goals_removed": removed,
                            "active_goals_remaining": remaining,
                        },
                    )
                )
            except Exception as exc:
                self._log.warning("goal_hygiene_event_emit_failed", error=str(exc))

    async def _emit_scanner_incident(
        self,
        error_type: str,
        error_message: str,
    ) -> None:
        """
        Emit an Incident to Thymos when the proactive scanner encounters an error.

        The immune system must be able to heal itself. Scanner errors that go
        unreported prevent the organism from self-healing.
        """
        if self._event_bus is None:
            self._log.warning(
                "scanner_incident_not_emitted_no_bus",
                error_type=error_type,
            )
            return

        try:
            import hashlib

            from systems.synapse.types import SynapseEvent, SynapseEventType
            from systems.thymos.types import Incident, IncidentClass, IncidentSeverity

            incident = Incident(
                incident_class=IncidentClass.CRASH,
                severity=IncidentSeverity.HIGH,
                fingerprint=hashlib.md5(
                    f"proactive_scanner_{error_type}".encode()
                ).hexdigest(),
                source_system="simula",
                error_type=error_type,
                error_message=error_message,
                context={"component": "proactive_scanner"},
                affected_systems=["simula"],
                blast_radius=0.5,
                user_visible=False,
            )

            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_FAILED,
                    source_system="simula",
                    data={"incident": incident.model_dump()},
                )
            )
            self._log.debug(
                "scanner_incident_emitted",
                error_type=error_type,
            )
        except Exception as emit_exc:
            self._log.warning(
                "failed_to_emit_scanner_incident",
                error=str(emit_exc),
            )

    async def _is_system_healthy(self, system_name: str) -> bool:
        """
        Check if the named system has been quiet (no incidents) for
        at least _SYSTEM_HEALTHY_QUIESCE_MINUTES minutes.

        Returns True (healthy) when there are no recent incidents.
        Returns False (still recovering) when in-doubt.
        """
        if self._neo4j is None:
            return True  # No Neo4j → can't verify → safe to retire goal

        quiesce_cutoff = (
            utc_now() - timedelta(minutes=_SYSTEM_HEALTHY_QUIESCE_MINUTES)
        ).isoformat()

        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (i:Incident)
                WHERE i.source_system = $system
                  AND i.timestamp > $cutoff
                RETURN count(i) AS n
                """,
                parameters={"system": system_name, "cutoff": quiesce_cutoff},
            )
            recent_count = int(rows[0].get("n") or 0) if rows else 0
            return recent_count == 0
        except Exception as exc:
            self._log.warning(
                "goal_auditor_health_check_failed",
                system=system_name,
                error=str(exc),
            )
            return False  # Be conservative — don't retire if we can't verify

    @staticmethod
    def _extract_system_from_maintenance_goal(description: str) -> str:
        """
        Parse the monitored system from a Thymos-generated maintenance goal.

        Thymos writes:
          "Monitor system recovery: <system_name> after <tier> repair"

        Returns "" if the description doesn't match the expected format.
        """
        import re
        m = re.search(r"Monitor system recovery:\s*(\S+)", description)
        if m:
            return m.group(1)
        return ""
