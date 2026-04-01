"""
EcodiaOS - Thymos Healing Governor (Cytokine Storm Prevention)

The immune system itself can be the problem. If 50 errors fire
simultaneously, Thymos must not try to diagnose and fix all 50.
That would consume all system resources and make things worse.

Biological parallel: a cytokine storm is when the immune response
is more damaging than the infection itself. In software: spending
100% of CPU diagnosing errors means 0% for actual cognitive work.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from systems.thymos.types import (
    HealingBudgetState,
    HealingMode,
    Incident,
    RepairTier,
)

if TYPE_CHECKING:
    from systems.thymos.diagnosis import CausalAnalyzer

logger = structlog.get_logger()


class HealingGovernor:
    """
    Prevents the immune system from overwhelming the organism.

    Rules:
    1. Max 3 concurrent diagnoses
    2. Max 1 concurrent codegen repair
    3. If >10 incidents in 60 seconds, switch to ROOT CAUSE FIRST mode:
       - Stop diagnosing individual incidents
       - Identify the common upstream cause
       - Fix that ONE thing
       - Wait for cascading failures to resolve
    4. Total immune system CPU budget: 10% of organism
    """

    MAX_CONCURRENT_DIAGNOSES = 5
    MAX_CONCURRENT_CODEGEN = 3
    MAX_CONCURRENT_T4_PROPOSALS = 5   # Cytokine storm: max concurrent T4 repairs
    MAX_T4_PROPOSALS_PER_HOUR = 20    # Cytokine storm: max T4 per hour
    STORM_THRESHOLD = 50              # incidents per minute before storm triggers
    STORM_EXIT_RATIO = 0.5            # exit at 50% below entry threshold (hysteresis)
    STORM_EXIT_SUSTAINED_S = 300.0    # 5 minutes below exit threshold before exiting
    STORM_WINDOW_S = 60.0
    CPU_BUDGET_FRACTION = 0.25

    def __init__(self) -> None:
        self._active_diagnoses: int = 0
        self._active_codegen: int = 0
        self._healing_mode: HealingMode = HealingMode.NOMINAL
        self._storm_focus_system: str | None = None
        self._storm_diagnosed_systems: set[str] = set()
        self._causal_analyzer: CausalAnalyzer | None = None

        # Ring buffer of recent incident timestamps for storm detection
        self._recent_incident_times: list[float] = []
        # All active (unresolved) incidents
        self._active_incidents: dict[str, Incident] = {}

        # T4 budget tracking (Tier 4 = novel repairs via Simula)
        self._t4_proposals_this_hour: int = 0
        self._t4_hour_start: float = utc_now().timestamp()
        self._active_t4_proposal_count: int = 0

        # Repair budget tracking
        self._repairs_this_hour: int = 0
        self._novel_repairs_today: int = 0
        self._hour_start: float = utc_now().timestamp()
        self._day_start: float = utc_now().timestamp()

        self._storm_activations: int = 0
        self._storm_entered_at: float = 0.0
        self._incidents_during_storm: int = 0
        self._storm_exit_candidate_since: float = 0.0  # hysteresis: when rate first dropped below exit threshold

        self._logger = logger.bind(system="thymos", component="healing_governor")
        # Optional event callback - set by ThymosService for lifecycle events
        self._on_event: Callable[[str, dict[str, Any]], None] | None = None

    def set_causal_analyzer(self, analyzer: CausalAnalyzer) -> None:
        """Inject causal analyzer for storm mode root cause detection."""
        self._causal_analyzer = analyzer

    def register_incident(self, incident: Incident) -> None:
        """Register an incident for storm detection and tracking."""
        now = utc_now().timestamp()
        self._recent_incident_times.append(now)
        self._active_incidents[incident.id] = incident
        # Prune old entries
        cutoff = now - self.STORM_WINDOW_S
        self._recent_incident_times = [
            t for t in self._recent_incident_times if t > cutoff
        ]

    def resolve_incident(self, incident_id: str) -> None:
        """Remove a resolved incident from tracking."""
        self._active_incidents.pop(incident_id, None)

    def should_diagnose(self, incident: Incident) -> bool:
        """Check if we have budget to diagnose this incident."""
        if self._active_diagnoses >= self.MAX_CONCURRENT_DIAGNOSES:
            self._logger.debug(
                "diagnosis_throttled",
                active=self._active_diagnoses,
                max=self.MAX_CONCURRENT_DIAGNOSES,
            )
            return False

        # Check for storm mode
        if self._is_storm():
            if self._healing_mode != HealingMode.STORM:
                self._enter_storm_mode()
            # In storm mode, only diagnose FIRST incident per source system
            if incident.source_system in self._storm_diagnosed_systems:
                return False
            self._storm_diagnosed_systems.add(incident.source_system)

        return True

    def should_codegen(self) -> bool:
        """Check if we can run a codegen (Tier 4) repair."""
        return self._active_codegen < self.MAX_CONCURRENT_CODEGEN

    def begin_diagnosis(self) -> None:
        """Acquire a diagnosis slot."""
        self._active_diagnoses += 1

    def end_diagnosis(self) -> None:
        """Release a diagnosis slot."""
        self._active_diagnoses = max(0, self._active_diagnoses - 1)

    def begin_codegen(self) -> None:
        """Acquire a codegen slot."""
        self._active_codegen += 1

    def end_codegen(self) -> None:
        """Release a codegen slot."""
        self._active_codegen = max(0, self._active_codegen - 1)

    def can_submit_t4_proposal(self) -> bool:
        """
        Cytokine storm prevention for T4 (Tier 4 novel repair) proposals.

        Rules:
        1. Max 3 concurrent T4 proposals in flight at once
        2. Max 5 T4 proposals per hour regardless of dedup

        Returns True if both budgets allow submission, False otherwise.
        """
        now = utc_now().timestamp()

        # Roll over hourly T4 counter
        if now - self._t4_hour_start > 3600.0:
            self._t4_proposals_this_hour = 0
            self._t4_hour_start = now

        # Check both budgets
        if self._active_t4_proposal_count >= self.MAX_CONCURRENT_T4_PROPOSALS:
            return False
        return not self._t4_proposals_this_hour >= self.MAX_T4_PROPOSALS_PER_HOUR

    def begin_t4_proposal(self) -> None:
        """Record start of a T4 proposal submission."""
        now = utc_now().timestamp()

        # Roll over hourly counter if needed
        if now - self._t4_hour_start > 3600.0:
            self._t4_proposals_this_hour = 0
            self._t4_hour_start = now

        self._t4_proposals_this_hour += 1
        self._active_t4_proposal_count += 1

    def end_t4_proposal(self) -> None:
        """Record completion of a T4 proposal (success or failure)."""
        self._active_t4_proposal_count = max(0, self._active_t4_proposal_count - 1)

    MAX_REPAIRS_PER_HOUR = 50
    MAX_NOVEL_REPAIRS_PER_DAY = 20

    def record_repair(self, tier: RepairTier) -> None:
        """Record a repair for budget tracking. Enters DEGRADED mode on exhaustion."""
        now = utc_now().timestamp()

        # Roll over hour
        if now - self._hour_start > 3600.0:
            self._repairs_this_hour = 0
            self._hour_start = now
            # Budget reset may exit degraded mode
            if self._healing_mode == HealingMode.DEGRADED:
                self._exit_degraded_mode()

        # Roll over day
        if now - self._day_start > 86400.0:
            self._novel_repairs_today = 0
            self._day_start = now
            if self._healing_mode == HealingMode.DEGRADED:
                self._exit_degraded_mode()

        self._repairs_this_hour += 1
        if tier == RepairTier.NOVEL_FIX:
            self._novel_repairs_today += 1

        # Check if budget is now exhausted
        if (
            self._healing_mode not in (HealingMode.STORM, HealingMode.DEGRADED)
            and self.is_budget_exhausted
        ):
            self._enter_degraded_mode()

    @property
    def is_budget_exhausted(self) -> bool:
        """True if the hourly repair budget or daily novel repair budget is used up."""
        return (
            self._repairs_this_hour >= self.MAX_REPAIRS_PER_HOUR
            or self._novel_repairs_today >= self.MAX_NOVEL_REPAIRS_PER_DAY
        )

    @property
    def degraded_reason(self) -> str | None:
        """Human-readable reason for degraded mode, or None if not degraded."""
        if self._healing_mode != HealingMode.DEGRADED:
            return None
        reasons: list[str] = []
        if self._repairs_this_hour >= self.MAX_REPAIRS_PER_HOUR:
            reasons.append(
                f"hourly repair budget exhausted "
                f"({self._repairs_this_hour}/{self.MAX_REPAIRS_PER_HOUR})"
            )
        if self._novel_repairs_today >= self.MAX_NOVEL_REPAIRS_PER_DAY:
            reasons.append(
                f"daily novel repair budget exhausted "
                f"({self._novel_repairs_today}/{self.MAX_NOVEL_REPAIRS_PER_DAY})"
            )
        return "; ".join(reasons) or "budget limits reached"

    def _enter_degraded_mode(self) -> None:
        """Enter degraded healing mode - repair budget exhausted."""
        self._healing_mode = HealingMode.DEGRADED
        self._logger.warning(
            "degraded_healing_mode_entered",
            repairs_this_hour=self._repairs_this_hour,
            max_repairs_per_hour=self.MAX_REPAIRS_PER_HOUR,
            novel_repairs_today=self._novel_repairs_today,
            max_novel_repairs_per_day=self.MAX_NOVEL_REPAIRS_PER_DAY,
        )

    def _exit_degraded_mode(self) -> None:
        """Exit degraded mode when budget resets."""
        self._healing_mode = HealingMode.NOMINAL
        self._logger.info("degraded_healing_mode_exited")

    def check_storm_exit(self) -> bool:
        """
        Check if storm conditions have subsided with hysteresis.

        Exit requires the incident rate to stay below the exit threshold
        (50% of entry threshold) for STORM_EXIT_SUSTAINED_S consecutive
        seconds (default 5 minutes). This prevents oscillation between
        storm and nominal modes.
        """
        if self._healing_mode != HealingMode.STORM:
            return False

        now = utc_now().timestamp()
        exit_threshold = self.STORM_THRESHOLD * self.STORM_EXIT_RATIO
        current_rate = self._current_incident_rate()

        if current_rate < exit_threshold:
            # Below exit threshold - start or continue the hysteresis timer
            if self._storm_exit_candidate_since == 0.0:
                self._storm_exit_candidate_since = now
                self._logger.debug(
                    "storm_exit_candidate_started",
                    current_rate=current_rate,
                    exit_threshold=exit_threshold,
                )
            elif now - self._storm_exit_candidate_since >= self.STORM_EXIT_SUSTAINED_S:
                # Sustained below exit threshold for required duration - safe to exit
                self._storm_exit_candidate_since = 0.0
                self._exit_storm_mode()
                return True
        else:
            # Rate spiked above exit threshold - reset the hysteresis timer
            if self._storm_exit_candidate_since > 0.0:
                self._logger.debug(
                    "storm_exit_candidate_reset",
                    current_rate=current_rate,
                    exit_threshold=exit_threshold,
                )
            self._storm_exit_candidate_since = 0.0

        return False

    def _is_storm(self) -> bool:
        """Check if incident rate exceeds storm threshold."""
        now = utc_now().timestamp()
        cutoff = now - self.STORM_WINDOW_S
        recent_count = sum(
            1 for t in self._recent_incident_times if t > cutoff
        )
        return recent_count >= self.STORM_THRESHOLD

    def _enter_storm_mode(self) -> None:
        """
        Storm mode: too many incidents firing. Focus on root cause.

        1. Pause individual diagnoses
        2. Use causal graph traversal to find the upstream root cause
           (not just the most common symptom system)
        3. Focus ALL diagnostic effort on that one system
        4. Exit storm mode when incident rate drops below threshold
        """
        self._healing_mode = HealingMode.STORM
        self._storm_activations += 1
        self._storm_entered_at = utc_now().timestamp()
        self._incidents_during_storm = len(self._active_incidents)
        self._storm_diagnosed_systems.clear()

        active_list = list(self._active_incidents.values())

        # Use causal analyzer if available for real upstream root detection
        if self._causal_analyzer is not None and active_list:
            root = self._causal_analyzer.find_common_upstream(active_list)
            if root is not None:
                self._storm_focus_system = root
                self._logger.critical(
                    "storm_mode_entered",
                    incident_rate=len(self._recent_incident_times),
                    active_incidents=len(self._active_incidents),
                    focus_system=self._storm_focus_system,
                    root_cause_method="causal_graph",
                )
                self._fire_storm_event("healing_storm_entered")
                return

        # Fallback: most common source system (no causal analyzer wired)
        system_counts: Counter[str] = Counter(
            i.source_system for i in active_list
        )
        if system_counts:
            self._storm_focus_system = system_counts.most_common(1)[0][0]
        else:
            self._storm_focus_system = None

        self._logger.critical(
            "storm_mode_entered",
            incident_rate=len(self._recent_incident_times),
            active_incidents=len(self._active_incidents),
            focus_system=self._storm_focus_system,
            root_cause_method="frequency_fallback",
        )
        self._fire_storm_event("healing_storm_entered")

    def _exit_storm_mode(self) -> None:
        """Exit storm mode - return to normal healing."""
        now = utc_now().timestamp()
        duration_s = now - self._storm_entered_at if self._storm_entered_at > 0 else 0.0
        exit_rate = self._current_incident_rate()

        self._healing_mode = HealingMode.NOMINAL
        self._storm_focus_system = None
        self._storm_diagnosed_systems.clear()

        self._logger.info(
            "storm_mode_exited",
            active_incidents=len(self._active_incidents),
            duration_s=round(duration_s, 1),
        )

        if self._on_event is not None:
            self._on_event("healing_storm_exited", {
                "duration_s": round(duration_s, 1),
                "incidents_during_storm": self._incidents_during_storm,
                "exit_rate": exit_rate,
                "timestamp": utc_now().isoformat(),
            })

    def _current_incident_rate(self) -> float:
        """Current incident rate (incidents/minute)."""
        now = utc_now().timestamp()
        cutoff = now - self.STORM_WINDOW_S
        return float(sum(1 for t in self._recent_incident_times if t > cutoff))

    def _fire_storm_event(self, event_name: str) -> None:
        """Emit storm lifecycle event via callback."""
        if self._on_event is not None:
            self._on_event(event_name, {
                "incident_rate": self._current_incident_rate(),
                "threshold": float(self.STORM_THRESHOLD),
                "active_incidents": len(self._active_incidents),
                "timestamp": utc_now().isoformat(),
            })

    @property
    def healing_mode(self) -> HealingMode:
        return self._healing_mode

    @property
    def storm_focus_system(self) -> str | None:
        return self._storm_focus_system

    @property
    def budget_state(self) -> HealingBudgetState:
        return HealingBudgetState(
            repairs_this_hour=self._repairs_this_hour,
            novel_repairs_today=self._novel_repairs_today,
            max_repairs_per_hour=self.MAX_REPAIRS_PER_HOUR,
            max_novel_repairs_per_day=self.MAX_NOVEL_REPAIRS_PER_DAY,
            active_diagnoses=self._active_diagnoses,
            max_concurrent_diagnoses=self.MAX_CONCURRENT_DIAGNOSES,
            active_codegen=self._active_codegen,
            max_concurrent_codegen=self.MAX_CONCURRENT_CODEGEN,
            t4_proposals_this_hour=self._t4_proposals_this_hour,
            max_t4_proposals_per_hour=self.MAX_T4_PROPOSALS_PER_HOUR,
            active_t4_proposals=self._active_t4_proposal_count,
            max_concurrent_t4_proposals=self.MAX_CONCURRENT_T4_PROPOSALS,
            storm_mode=self._healing_mode == HealingMode.STORM,
            storm_focus_system=self._storm_focus_system,
            cpu_budget_fraction=self.CPU_BUDGET_FRACTION,
        )

    @property
    def storm_activations(self) -> int:
        return self._storm_activations
