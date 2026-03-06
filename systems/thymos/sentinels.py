"""
EcodiaOS — Thymos Sentinel Layer (Detection)

Sentinels are the sensory organs of the immune system. They instrument
every system boundary to capture failures, anomalies, contract violations,
and degradation trends.

Five sentinel classes:
  1. ExceptionSentinel  — unhandled exceptions with full context
  2. ContractSentinel   — inter-system SLA violations
  3. FeedbackLoopSentinel — severed cognitive feedback loops
  4. DriftSentinel      — statistical process control for gradual degradation
  5. CognitiveStallSentinel — workspace cycle producing nothing
"""

from __future__ import annotations

import hashlib
import math
import traceback
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import structlog

from primitives.common import utc_now
from systems.thymos.types import (
    AddressBlacklistEntry,
    ContractSLA,
    DriftConfig,
    FeedbackLoop,
    Incident,
    IncidentClass,
    IncidentSeverity,
    RepairTier,
    StallConfig,
    ThreatPattern,
    ThreatSeverity,
    ThreatType,
)

logger = structlog.get_logger()


# ─── Strategy ABC ───────────────────────────────────────────────


class BaseThymosSentinel(ABC):
    """
    Strategy base class for all Thymos sentinel detectors.

    Each sentinel is a stateless (or cleanly-resettable) observer that
    instruments one specific class of system failure.  The
    NeuroplasticityBus uses this ABC as its registration target so that
    evolved sentinel subclasses can be hot-swapped into a live
    ThymosService without restarting the process.

    Contract:
    - ``sentinel_name`` must be a stable string that uniquely identifies the
      slot in ThymosService (e.g. "exception", "drift").
    - Subclasses MUST remain stateless OR fully initialise their state in
      ``__init__`` so that a fresh instance dropped in via hot-swap is
      immediately usable.
    - Do NOT store references to long-lived external objects beyond what the
      constructor receives — hot-swap creates a new instance via ``cls()``
      (zero-arg) by default.
    """

    @property
    @abstractmethod
    def sentinel_name(self) -> str:
        """
        Stable identifier for the slot this sentinel occupies.

        Must match the key used in ThymosService._sentinels dict and the
        _on_sentinel_evolved dispatcher.
        """
        ...


# ─── System Dependency Graph ────────────────────────────────────


# Which systems directly affect the user
_USER_FACING_SYSTEMS = frozenset({"voxis", "alive", "atune"})

# Downstream impact map: if system X fails, these are affected
_DOWNSTREAM: dict[str, list[str]] = {
    "memory": ["atune", "nova", "evo", "voxis", "simula", "federation"],
    "equor": ["nova", "axon", "simula", "federation"],
    "atune": ["nova", "evo", "voxis"],
    "nova": ["axon", "voxis"],
    "voxis": [],
    "axon": [],
    "evo": ["simula"],
    "simula": [],
    "synapse": ["atune", "nova", "evo", "memory"],
    "federation": [],
}

# Total number of cognitive systems
_TOTAL_SYSTEMS = 11


# ─── Exception Sentinel ─────────────────────────────────────────


class ExceptionSentinel(BaseThymosSentinel):
    """
    Intercepts unhandled exceptions from system methods.
    Creates Incidents with full diagnostic context.

    Does NOT suppress the exception — it propagates normally after capture.
    The goal is observation, not intervention.
    """

    @property
    def sentinel_name(self) -> str:
        return "exception"

    def __init__(self) -> None:
        self._logger = logger.bind(system="thymos", component="exception_sentinel")

    def intercept(
        self,
        system_id: str,
        method_name: str,
        exception: BaseException,
        context: dict[str, Any] | None = None,
    ) -> Incident:
        """Create an Incident from an unhandled exception."""
        fp = self.fingerprint(system_id, method_name, exception)
        affected = _DOWNSTREAM.get(system_id, [])
        blast = len(affected) / _TOTAL_SYSTEMS

        return Incident(
            timestamp=utc_now(),
            incident_class=IncidentClass.CRASH,
            severity=self._assess_severity(system_id, exception),
            fingerprint=fp,
            source_system=system_id,
            error_type=type(exception).__name__,
            error_message=str(exception)[:500],
            stack_trace=traceback.format_exc()[:2000],
            context={
                "method": method_name,
                **(context or {}),
            },
            affected_systems=affected,
            blast_radius=blast,
            user_visible=system_id in _USER_FACING_SYSTEMS,
            constitutional_impact=self._assess_constitutional_impact(system_id),
        )

    def fingerprint(
        self,
        system_id: str,
        method: str,
        exc: BaseException,
    ) -> str:
        """
        Create a stable fingerprint for deduplication.

        Hash of: system_id + exception type + first frame in our code.
        Groups "same bug, different call path" together while
        distinguishing genuinely different errors.
        """
        first_frame = self._extract_first_local_frame(exc)
        raw = f"{system_id}:{type(exc).__name__}:{first_frame}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _extract_first_local_frame(self, exc: BaseException) -> str:
        """Extract the first stack frame from our code (not library).

        Uses filename:function_name (no line number) so that the same logical
        bug still maps to the same fingerprint after Simula edits nearby code.
        """
        tb = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
        for frame in reversed(tb):
            if "ecodiaos" in frame.filename:
                return f"{frame.filename}:{frame.name}"
        # Fallback: last frame
        if tb:
            f = tb[-1]
            return f"{f.filename}:{f.name}"
        return "unknown"

    def _assess_severity(
        self,
        system_id: str,
        exception: BaseException,
    ) -> IncidentSeverity:
        """Initial severity based on system criticality and exception type."""
        # Critical systems crashing is always HIGH+
        if system_id in ("equor", "memory", "atune", "synapse"):
            return IncidentSeverity.CRITICAL
        if system_id in ("nova", "voxis", "axon"):
            return IncidentSeverity.HIGH
        # Non-critical systems
        if isinstance(exception, (TimeoutError, ConnectionError)):
            return IncidentSeverity.MEDIUM
        return IncidentSeverity.MEDIUM

    def _assess_constitutional_impact(self, system_id: str) -> dict[str, float]:
        """Estimate impact on each drive when a system fails."""
        impacts: dict[str, dict[str, float]] = {
            "equor": {"coherence": 0.9, "care": 0.5, "growth": 0.3, "honesty": 0.9},
            "memory": {"coherence": 0.8, "care": 0.3, "growth": 0.7, "honesty": 0.4},
            "atune": {"coherence": 0.7, "care": 0.4, "growth": 0.5, "honesty": 0.3},
            "nova": {"coherence": 0.6, "care": 0.3, "growth": 0.5, "honesty": 0.2},
            "voxis": {"coherence": 0.2, "care": 0.7, "growth": 0.1, "honesty": 0.6},
            "axon": {"coherence": 0.3, "care": 0.5, "growth": 0.2, "honesty": 0.1},
            "evo": {"coherence": 0.2, "care": 0.1, "growth": 0.8, "honesty": 0.1},
            "simula": {"coherence": 0.1, "care": 0.1, "growth": 0.7, "honesty": 0.1},
            "synapse": {"coherence": 0.8, "care": 0.3, "growth": 0.3, "honesty": 0.2},
        }
        return impacts.get(
            system_id,
            {"coherence": 0.1, "care": 0.1, "growth": 0.1, "honesty": 0.1},
        )


# ─── Contract Sentinel ──────────────────────────────────────────


# Inter-system contract SLAs from the Architecture Spec §IV
DEFAULT_CONTRACT_SLAS: list[ContractSLA] = [
    ContractSLA(source="atune", target="memory", operation="store_percept", max_latency_ms=100),
    ContractSLA(source="memory", target="atune", operation="retrieval", max_latency_ms=200),
    ContractSLA(source="atune", target="all", operation="broadcast", max_latency_ms=50),
    ContractSLA(source="nova", target="equor", operation="review", max_latency_ms=500),
    ContractSLA(source="nova", target="equor", operation="review_critical", max_latency_ms=50),
]


class ContractSentinel(BaseThymosSentinel):
    """
    Instruments inter-system calls to verify SLA compliance.

    Does NOT add latency to the call itself — measurements are taken
    around the existing call, not inserted into it. The sentinel
    observes the event bus, not the call stack.
    """

    @property
    def sentinel_name(self) -> str:
        return "contract"

    def __init__(self, slas: list[ContractSLA] | None = None) -> None:
        self._slas: dict[tuple[str, str, str], ContractSLA] = {}
        for sla in slas or DEFAULT_CONTRACT_SLAS:
            self._slas[(sla.source, sla.target, sla.operation)] = sla
        self._logger = logger.bind(system="thymos", component="contract_sentinel")

    def check_contract(
        self,
        source: str,
        target: str,
        operation: str,
        latency_ms: float,
    ) -> Incident | None:
        """Check if an inter-system call violated its SLA."""
        sla = self._slas.get((source, target, operation))
        if sla is None:
            return None

        if latency_ms <= sla.max_latency_ms:
            return None

        overshoot = latency_ms / sla.max_latency_ms
        fp = hashlib.sha256(
            f"contract:{source}:{target}:{operation}".encode()
        ).hexdigest()[:16]

        # Honesty takes the hit when information contracts are violated —
        # the system is not being transparent about its actual responsiveness.
        # Coherence degrades when the cognitive loop is slow or broken.
        honesty_impact = 0.3 if overshoot > 2.0 else 0.1
        coherence_impact = min(0.6, overshoot * 0.15)

        return Incident(
            timestamp=utc_now(),
            incident_class=IncidentClass.CONTRACT_VIOLATION,
            severity=IncidentSeverity.MEDIUM,
            fingerprint=fp,
            source_system=source,
            error_type="ContractViolation",
            error_message=(
                f"Contract violation: {source}→{target}.{operation} "
                f"took {latency_ms:.0f}ms (SLA: {sla.max_latency_ms}ms)"
            ),
            context={
                "expected_ms": sla.max_latency_ms,
                "actual_ms": latency_ms,
                "overshoot_factor": overshoot,
                "target_system": target,
                "operation": operation,
            },
            affected_systems=[source, target],
            blast_radius=2 / _TOTAL_SYSTEMS,
            user_visible=target in _USER_FACING_SYSTEMS,
            constitutional_impact={
                "coherence": coherence_impact,
                "care": 0.2 if target in _USER_FACING_SYSTEMS else 0.05,
                "growth": 0.1,
                "honesty": honesty_impact,
            },
        )


# ─── Feedback Loop Sentinel ─────────────────────────────────────


# The feedback loops identified in the architecture audit
DEFAULT_FEEDBACK_LOOPS: list[FeedbackLoop] = [
    # active=None  → data flow NOT confirmed end-to-end; generates LOW incidents only.
    # active=True  → confirmed wired; MISSING transmissions generate HIGH incidents.
    # active=False → explicitly disabled; never generates incidents.

    FeedbackLoop(
        name="top_down_prediction",
        source="nova",
        target="atune",
        signal="belief_state",
        check="atune.has_received_beliefs_in_last_n_cycles(10)",
        description="Nova beliefs → Atune prediction error modeling",
        active=None,  # set_belief_state() exists but is never called from nova
    ),
    FeedbackLoop(
        name="goal_guided_attention",
        source="nova",
        target="atune",
        signal="active_goals",
        check="atune.salience_head_weights_include_goal_component()",
        description="Nova goals → Atune salience weighting",
        active=None,  # set_active_goals wired but no sentinel verification method
    ),
    FeedbackLoop(
        name="expression_feedback",
        source="voxis",
        target="atune",
        signal="expression_feedback",
        check="atune.has_received_expression_feedback_in_last_n_cycles(100)",
        description="Voxis expression → Atune learning signal",
        active=None,  # nudge_valence wired; full learning signal not confirmed
    ),
    FeedbackLoop(
        name="evo_head_weights",
        source="evo",
        target="atune",
        signal="head_weight_adjustments",
        check="atune.has_received_evo_adjustments()",
        description="Evo learned weights → Atune meta-attention",
        active=None,  # apply_evo_adjustments() is a stub (pass)
    ),
    FeedbackLoop(
        name="axon_outcome_beliefs",
        source="axon",
        target="nova",
        signal="action_outcomes",
        check="nova.has_received_outcomes_in_last_n_cycles(100)",
        description="Axon action outcomes → Nova belief updates",
        active=None,  # partial: expression outcomes reach nova; direct axon path unclear
    ),
    FeedbackLoop(
        name="memory_salience_decay",
        source="memory",
        target="memory",
        signal="salience_decay",
        check="memory.salience_decay_running()",
        description="Memory salience decay over time",
        active=None,  # no decay implementation found in memory service
    ),
    FeedbackLoop(
        name="personality_evolution",
        source="voxis",
        target="evo",
        signal="expression_effectiveness",
        check="evo.has_personality_evidence()",
        description="Voxis expression → Evo personality tuning",
        active=None,  # set_voxis wired but reverse data flow not confirmed
    ),
    FeedbackLoop(
        name="rhythm_modulation",
        source="synapse",
        target="nova",
        signal="rhythm_state",
        check="nova.receives_rhythm_updates()",
        description="Synapse rhythm → Nova decision thresholds",
        active=True,  # RHYTHM_STATE_CHANGED → nova.on_rhythm_change() confirmed wired
    ),
    FeedbackLoop(
        name="consolidation_weights",
        source="evo",
        target="atune",
        signal="parameter_adjustments",
        check="atune.has_evo_parameters()",
        description="Evo consolidation → Atune salience head weights",
        active=None,  # same stub as evo_head_weights — apply_evo_adjustments is pass
    ),
    FeedbackLoop(
        name="drive_weight_modulation",
        source="equor",
        target="equor",
        signal="contextual_drive_weights",
        check="equor.drive_weights_modulated()",
        description="Context → dynamic drive weighting",
        active=None,  # no drive weight modulation found in equor service
    ),
    FeedbackLoop(
        name="affect_expression",
        source="atune",
        target="voxis",
        signal="affect_state",
        check="voxis.uses_affect_for_style()",
        description="Atune affect → Voxis expression style",
        active=True,  # atune affect in workspace broadcasts → voxis expression confirmed
    ),
    FeedbackLoop(
        name="federation_trust_access",
        source="federation",
        target="federation",
        signal="trust_level_changes",
        check="federation.trust_updates_permissions()",
        description="Trust level changes → knowledge exchange permissions",
        active=None,  # TrustManager exists; permission-change mechanism not confirmed
    ),
    FeedbackLoop(
        name="simula_version_params",
        source="simula",
        target="synapse",
        signal="config_version",
        check="synapse.uses_current_config_version()",
        description="Simula config changes → system parameter propagation",
        active=None,  # no config propagation from simula to synapse found
    ),
    FeedbackLoop(
        name="coherence_safe_mode",
        source="synapse",
        target="synapse",
        signal="coherence_level",
        check="synapse.coherence_triggers_safe_mode()",
        description="Low coherence → safe mode consideration",
        active=None,  # CoherenceMonitor exists; safe-mode trigger not confirmed
    ),
    FeedbackLoop(
        name="community_schema",
        source="memory",
        target="evo",
        signal="community_detection",
        check="evo.uses_community_structure()",
        description="Neo4j community detection → Evo schema induction",
        active=None,  # on_schema_formed callback exists; memory trigger not confirmed
    ),
]


class FeedbackLoopSentinel(BaseThymosSentinel):
    """
    Periodically verifies that each feedback loop is actively transmitting.

    Unlike heartbeats (which check "is the system alive?"), this checks
    "is the system CONNECTED?" A system can be alive but disconnected
    from the cognitive cycle — like a nerve that's intact but severed
    from the brain.
    """

    @property
    def sentinel_name(self) -> str:
        return "feedback_loop"

    def __init__(self, loops: list[FeedbackLoop] | None = None) -> None:
        self._loops = loops or DEFAULT_FEEDBACK_LOOPS
        # Track which loops have been verified as connected.
        # Initial value respects loop.active:
        #   True  → False (not yet seen, expected to become True)
        #   None  → None  (unknown; not yet implemented — don't flood HIGH incidents)
        #   False → False and loop is silenced in check_loops
        self._loop_status: dict[str, bool | None] = {
            loop.name: (None if loop.active is None else False)
            for loop in self._loops
            if loop.active is not False  # disabled loops are never tracked
        }
        self._last_check: dict[str, float] = {}  # loop_name → timestamp
        self._logger = logger.bind(system="thymos", component="feedback_loop_sentinel")

    def report_loop_active(self, loop_name: str) -> None:
        """
        Called when evidence of a loop transmitting is observed.

        Upgrades status from None (unknown) or False (not yet seen) to True,
        regardless of the loop's initial ``active`` field.  This allows an
        unknown loop to be promoted to known-good once it fires.
        """
        if loop_name not in self._loop_status:
            # Loop is either disabled (active=False) or not defined — ignore.
            return
        self._loop_status[loop_name] = True
        self._last_check[loop_name] = utc_now().timestamp()

    def check_loops(self, max_staleness_s: float = 30.0) -> list[Incident]:
        """
        Check all loops. Returns incidents for any that aren't transmitting.

        Severity is determined by ``loop.active``:
          - ``True``  (confirmed wired): MISSING → HIGH LOOP_SEVERANCE incident.
          - ``None``  (unknown / not yet implemented in code): MISSING → LOW INFO
                      incident so the boot-time flood doesn't drown real alerts.
          - ``False`` (explicitly disabled): never generates incidents.

        A loop is considered not transmitting if:
        - It has never been observed active (report_loop_active never called), OR
        - It was last seen active more than max_staleness_s seconds ago.
        """
        now = utc_now()
        incidents: list[Incident] = []

        for loop in self._loops:
            # Disabled loops are never monitored.
            if loop.active is False:
                continue

            # Loops not in our status dict were disabled at init time — skip.
            if loop.name not in self._loop_status:
                continue

            status = self._loop_status.get(loop.name)
            last_ts = self._last_check.get(loop.name)

            if status is True and last_ts is not None:
                age_s = now.timestamp() - last_ts
                if age_s <= max_staleness_s:
                    continue  # Loop is fresh

            fp = hashlib.sha256(f"loop:{loop.name}".encode()).hexdigest()[:16]
            care_impact = 0.5 if loop.signal in ("expression_feedback", "affect_state") else 0.1
            growth_impact = 0.5 if loop.signal in (
                "head_weight_adjustments", "parameter_adjustments", "community_detection"
            ) else 0.1

            # Unknown loops (active=None) emit LOW severity — they are gaps to be
            # filled, not confirmed breakages.  Confirmed loops (active=True) that
            # go silent are HIGH — something that was working has stopped.
            if loop.active is None:
                severity = IncidentSeverity.LOW
                error_type = "FeedbackLoopUnknown"
                msg = (
                    f"Feedback loop '{loop.name}' has not been wired yet "
                    f"(implementation pending): {loop.description}"
                )
                constitutional_impact = {
                    "coherence": 0.1,
                    "care": care_impact * 0.2,
                    "growth": growth_impact * 0.2,
                    "honesty": 0.0,
                }
            else:
                severity = IncidentSeverity.HIGH
                error_type = "FeedbackLoopSevered"
                msg = (
                    f"Feedback loop '{loop.name}' is not transmitting: "
                    f"{loop.description}"
                )
                constitutional_impact = {
                    "coherence": 0.6,
                    "care": care_impact,
                    "growth": growth_impact,
                    "honesty": 0.1,
                }

            incidents.append(
                Incident(
                    timestamp=now,
                    incident_class=IncidentClass.LOOP_SEVERANCE,
                    severity=severity,
                    fingerprint=fp,
                    source_system=loop.source,
                    error_type=error_type,
                    error_message=msg,
                    context={
                        "loop_name": loop.name,
                        "source": loop.source,
                        "target": loop.target,
                        "signal": loop.signal,
                        "loop_active": loop.active,
                    },
                    affected_systems=[loop.source, loop.target],
                    blast_radius=2 / _TOTAL_SYSTEMS,
                    user_visible=False,
                    constitutional_impact=constitutional_impact,
                )
            )

        return incidents

    @property
    def loop_statuses(self) -> dict[str, bool | None]:
        """Current status of all feedback loops. None = unknown/not yet wired."""
        return dict(self._loop_status)


# ─── Drift Sentinel ─────────────────────────────────────────────


class _RollingBaseline:
    """Exponential moving average + standard deviation tracker."""

    def __init__(self, window: int) -> None:
        self._window = window
        self._values: deque[float] = deque(maxlen=window)
        self._ema: float = 0.0
        self._ema_sq: float = 0.0
        self._alpha: float = 2.0 / (window + 1)
        self._count: int = 0

    @property
    def is_warmed_up(self) -> bool:
        return self._count >= self._window // 4  # 25% of window

    def update(self, value: float) -> None:
        self._values.append(value)
        self._count += 1
        if self._count == 1:
            self._ema = value
            self._ema_sq = value * value
        else:
            self._ema = self._alpha * value + (1 - self._alpha) * self._ema
            self._ema_sq = (
                self._alpha * (value * value)
                + (1 - self._alpha) * self._ema_sq
            )

    @property
    def mean(self) -> float:
        return self._ema

    @property
    def std(self) -> float:
        variance = max(0.0, self._ema_sq - self._ema * self._ema)
        return math.sqrt(variance)

    def z_score(self, value: float) -> float:
        s = self.std
        if s < 1e-9:
            return 0.0
        return (value - self._ema) / s


# Default metrics to monitor for drift
DEFAULT_DRIFT_METRICS: dict[str, DriftConfig] = {
    "synapse.cycle.latency_ms": DriftConfig(window=1000, sigma_threshold=2.5),
    "memory.retrieval.latency_ms": DriftConfig(window=500, sigma_threshold=2.0),
    "atune.salience.processing_ms": DriftConfig(window=500, sigma_threshold=2.0),
    "nova.efe.computation_ms": DriftConfig(window=500, sigma_threshold=2.5),
    "voxis.generation.latency_ms": DriftConfig(window=300, sigma_threshold=2.0),
    "synapse.resources.memory_mb": DriftConfig(
        window=200, sigma_threshold=3.0, direction="above"
    ),
    "evo.self_model.success_rate": DriftConfig(
        window=100, sigma_threshold=2.0, direction="below"
    ),
    "atune.coherence.phi": DriftConfig(
        window=200, sigma_threshold=2.0, direction="below"
    ),
}


# Constitutional impact per drifting metric — which drive does this hurt most?
_DRIFT_CONSTITUTIONAL_IMPACT: dict[str, dict[str, float]] = {
    "synapse.cycle.latency_ms": {"coherence": 0.3, "care": 0.2, "growth": 0.1, "honesty": 0.0},
    "memory.retrieval.latency_ms": {"coherence": 0.5, "care": 0.1, "growth": 0.2, "honesty": 0.1},
    "atune.salience.processing_ms": {"coherence": 0.4, "care": 0.1, "growth": 0.1, "honesty": 0.0},
    "nova.efe.computation_ms": {"coherence": 0.3, "care": 0.0, "growth": 0.2, "honesty": 0.0},
    "voxis.generation.latency_ms": {"coherence": 0.1, "care": 0.5, "growth": 0.0, "honesty": 0.2},
    "synapse.resources.memory_mb": {"coherence": 0.2, "care": 0.1, "growth": 0.3, "honesty": 0.0},
    "evo.self_model.success_rate": {"coherence": 0.2, "care": 0.0, "growth": 0.7, "honesty": 0.1},
    "atune.coherence.phi": {"coherence": 0.8, "care": 0.1, "growth": 0.3, "honesty": 0.1},
}


class DriftSentinel(BaseThymosSentinel):
    """
    Statistical process control for system metrics.

    Maintains a rolling baseline (EMA + std dev) for each metric.
    When a metric deviates beyond the control limits, it's flagged.

    This catches:
    - Memory leaks (gradual increase in memory_mb)
    - Latency creep (gradually slower responses)
    - Accuracy decay (prediction errors gradually increasing)
    - Throughput degradation (cycles/second declining)

    Adapts to the organism's actual operating characteristics.
    """

    @property
    def sentinel_name(self) -> str:
        return "drift"

    def __init__(
        self,
        metrics: dict[str, DriftConfig] | None = None,
    ) -> None:
        self._metrics = metrics or DEFAULT_DRIFT_METRICS
        self._baselines: dict[str, _RollingBaseline] = {}
        for name, cfg in self._metrics.items():
            self._baselines[name] = _RollingBaseline(cfg.window)
        self._logger = logger.bind(system="thymos", component="drift_sentinel")

    def record_metric(self, metric_name: str, value: float) -> Incident | None:
        """
        Record a metric value and check for drift.
        Returns an Incident if drift is detected, None otherwise.
        """
        config = self._metrics.get(metric_name)
        if config is None:
            return None

        baseline = self._baselines[metric_name]
        baseline.update(value)

        if not baseline.is_warmed_up:
            return None

        z = baseline.z_score(value)

        is_drift = (
            (config.direction == "above" and z > config.sigma_threshold)
            or (config.direction == "below" and z < -config.sigma_threshold)
            or (config.direction is None and abs(z) > config.sigma_threshold)
        )

        if not is_drift:
            return None

        fp = hashlib.sha256(f"drift:{metric_name}".encode()).hexdigest()[:16]
        system_id = metric_name.split(".")[0] if "." in metric_name else "unknown"

        # Drive impact depends on which metric is drifting:
        # - coherence metrics (atune.coherence.phi) → coherence + growth
        # - success rate metrics → growth
        # - latency metrics → care (users experience degradation)
        drift_impact = _DRIFT_CONSTITUTIONAL_IMPACT.get(
            metric_name,
            {"coherence": 0.1, "care": 0.1, "growth": 0.1, "honesty": 0.0},
        )

        return Incident(
            timestamp=utc_now(),
            incident_class=IncidentClass.DRIFT,
            severity=IncidentSeverity.MEDIUM,
            fingerprint=fp,
            source_system=system_id,
            error_type="MetricDrift",
            error_message=(
                f"Metric '{metric_name}' drifting: "
                f"value={value:.2f}, baseline={baseline.mean:.2f}, "
                f"z-score={z:.2f} (threshold: ±{config.sigma_threshold})"
            ),
            context={
                "metric_name": metric_name,
                "current_value": value,
                "baseline_mean": baseline.mean,
                "baseline_std": baseline.std,
                "z_score": z,
                "direction": config.direction or "both",
            },
            blast_radius=0.1,
            user_visible=False,
            constitutional_impact=drift_impact,
        )

    @property
    def baselines(self) -> dict[str, dict[str, float]]:
        """Current baseline statistics for all monitored metrics."""
        return {
            name: {
                "mean": b.mean,
                "std": b.std,
                "warmed_up": b.is_warmed_up,
                "samples": b._count,
            }
            for name, b in self._baselines.items()
        }


# ─── Cognitive Stall Sentinel ────────────────────────────────────


# Default stall thresholds
DEFAULT_STALL_THRESHOLDS: dict[str, StallConfig] = {
    "broadcast_ack_rate": StallConfig(min_value=0.3, window_cycles=50),
    "nova_intent_rate": StallConfig(min_value=0.01, window_cycles=200),
    "evo_evidence_rate": StallConfig(min_value=0.001, window_cycles=500),
    "atune_percept_rate": StallConfig(min_value=0.1, window_cycles=50),
}


# Constitutional impact per stall type — what breaks when that rate hits zero?
_STALL_CONSTITUTIONAL_IMPACT: dict[str, dict[str, float]] = {
    "broadcast_ack_rate": {"coherence": 0.7, "care": 0.4, "growth": 0.3, "honesty": 0.2},
    "nova_intent_rate": {"coherence": 0.8, "care": 0.2, "growth": 0.4, "honesty": 0.1},
    "evo_evidence_rate": {"coherence": 0.3, "care": 0.0, "growth": 0.8, "honesty": 0.1},
    "atune_percept_rate": {"coherence": 0.5, "care": 0.6, "growth": 0.2, "honesty": 0.1},
}


class CognitiveStallSentinel(BaseThymosSentinel):
    """
    Detects when the cognitive cycle is running but accomplishing nothing.

    The heartbeat is fine. The systems are "healthy." But nothing is
    happening. The organism is not thinking.

    This is the equivalent of a person who is conscious but catatonic.
    """

    @property
    def sentinel_name(self) -> str:
        return "cognitive_stall"

    def __init__(
        self,
        thresholds: dict[str, StallConfig] | None = None,
    ) -> None:
        # Merge custom thresholds with defaults to ensure all expected keys exist
        self._thresholds = DEFAULT_STALL_THRESHOLDS.copy()
        if thresholds:
            self._thresholds.update(thresholds)

        self._counters: dict[str, deque[float]] = {
            name: deque(maxlen=cfg.window_cycles)
            for name, cfg in self._thresholds.items()
        }
        self._logger = logger.bind(system="thymos", component="stall_sentinel")

    def record_cycle(
        self,
        had_broadcast: bool,
        nova_had_intent: bool,
        evo_had_evidence: bool,
        atune_had_percept: bool,
    ) -> list[Incident]:
        """Record one cognitive cycle's activity and check for stalls."""
        self._counters["broadcast_ack_rate"].append(1.0 if had_broadcast else 0.0)
        self._counters["nova_intent_rate"].append(1.0 if nova_had_intent else 0.0)
        self._counters["evo_evidence_rate"].append(1.0 if evo_had_evidence else 0.0)
        self._counters["atune_percept_rate"].append(1.0 if atune_had_percept else 0.0)

        incidents: list[Incident] = []
        now = utc_now()

        for name, cfg in self._thresholds.items():
            window = self._counters[name]
            if len(window) < cfg.window_cycles:
                continue  # Not enough data yet

            rate = sum(window) / len(window)
            if rate >= cfg.min_value:
                continue  # Above threshold, no stall

            fp = hashlib.sha256(f"stall:{name}".encode()).hexdigest()[:16]
            # Cognitive stalls hurt all drives: the organism isn't thinking.
            # Intent stalls hit coherence hardest (no deliberation happening).
            # Percept stalls hit care (organism is ignoring external signals).
            stall_impact = _STALL_CONSTITUTIONAL_IMPACT.get(
                name,
                {"coherence": 0.5, "care": 0.3, "growth": 0.3, "honesty": 0.1},
            )
            incidents.append(
                Incident(
                    timestamp=now,
                    incident_class=IncidentClass.COGNITIVE_STALL,
                    severity=IncidentSeverity.HIGH,
                    fingerprint=fp,
                    source_system="synapse",
                    error_type="CognitiveStall",
                    error_message=(
                        f"Cognitive stall: '{name}' rate is {rate:.4f} "
                        f"(minimum: {cfg.min_value}) over {cfg.window_cycles} cycles"
                    ),
                    context={
                        "metric_name": name,
                        "rate": rate,
                        "threshold": cfg.min_value,
                        "window_cycles": cfg.window_cycles,
                    },
                    blast_radius=0.5,
                    user_visible=True,  # Catatonic organism affects users
                    constitutional_impact=stall_impact,
                )
            )

        return incidents


# ─── Bankruptcy Sentinel ──────────────────────────────────────────


# Critical gas floor: below this ETH balance the organism cannot submit
# on-chain transactions (ERC-4337 UserOps, transfers).  0.00005 ETH ≈
# the cost of a single Base L2 transaction at modest gas price.
_CRITICAL_GAS_THRESHOLD_ETH: float = 0.00005

# Hard limit on the rolling fiat API deficit. Once the organism has spent
# this much more than it has earned it is functionally insolvent and must
# shed all discretionary cognitive load immediately.
_CRITICAL_API_DEFICIT_USD: float = 5.0

# Stable fingerprints — one per failure mode so deduplication works correctly.
_FP_ETH = hashlib.sha256(b"bankruptcy:eth_below_gas_threshold").hexdigest()[:16]
_FP_API = hashlib.sha256(b"bankruptcy:api_deficit_hard_limit").hexdigest()[:16]

# Repair strategy description embedded in every incident context so that
# the Prescription sub-system can short-circuit to ESCALATE without an
# LLM diagnosis round-trip.
_REPAIR_STRATEGY = (
    "Tier-1 austerity: immediately halt all non-essential cognitive functions. "
    "Stop Oneiros dreaming (set sleep pressure floor to prevent new sleep cycles). "
    "Halt Evo background consolidation (skip run_consolidation until deficit clears). "
    "Preserve remaining capital exclusively for essential I/O (Voxis responses, "
    "Atune perception, Memory reads). Resume full operation only after revenue is "
    "injected via SynapseService.metabolism.inject_revenue()."
)


class BankruptcySentinel(BaseThymosSentinel):
    """
    Monitors the organism's metabolic solvency.

    Two failure modes trigger a Tier-1 (CRITICAL) incident:

    1. **ETH gas exhaustion** — on-chain balance drops below
       ``_CRITICAL_GAS_THRESHOLD_ETH``.  Without gas the organism cannot
       execute wallet operations, which blocks all revenue pathways.

    2. **API deficit hard limit** — the rolling fiat deficit tracked by
       ``SynapseService.metabolism`` exceeds ``_CRITICAL_API_DEFICIT_USD``.
       This means the organism has consumed significantly more in LLM API
       costs than it has earned, and continued operation will accelerate
       the debt.

    Repair strategy: halt all discretionary cognitive functions (Oneiros
    dreaming, Evo background consolidation) to reduce burn rate to the
    minimum required for active I/O, and escalate for human revenue
    injection.

    Design notes
    ------------
    - Both ``check_eth_balance`` and ``check_api_deficit`` are **pure**
      (no I/O) so they can be called on every homeostasis tick without
      async overhead.  The caller is responsible for fetching fresh
      values from ``WalletClient`` / ``SynapseService`` first.
    - The sentinel is **stateless** per the ABC contract.  It has no
      rolling windows or accumulators — the thresholds are absolute
      because near-zero balances are dangerous regardless of trend.
    - ``sentinel_name`` is ``"bankruptcy"`` so it occupies its own slot
      in ``ThymosService._sentinels`` and can be hot-swapped independently.
    """

    @property
    def sentinel_name(self) -> str:
        return "bankruptcy"

    def __init__(
        self,
        gas_threshold_eth: float = _CRITICAL_GAS_THRESHOLD_ETH,
        api_deficit_limit_usd: float = _CRITICAL_API_DEFICIT_USD,
    ) -> None:
        self._gas_threshold = gas_threshold_eth
        self._deficit_limit = api_deficit_limit_usd
        self._logger = logger.bind(system="thymos", component="bankruptcy_sentinel")

    # ── Public check methods ─────────────────────────────────────────

    def check_eth_balance(self, eth_balance: float) -> Incident | None:
        """
        Return a CRITICAL incident if ``eth_balance`` is below the gas floor,
        otherwise return None.

        Args:
            eth_balance: Current ETH balance in whole units (e.g. 0.0001).
        """
        if eth_balance >= self._gas_threshold:
            return None

        self._logger.warning(
            "bankruptcy_eth_critical",
            eth_balance=eth_balance,
            threshold=self._gas_threshold,
        )
        return Incident(
            incident_class=IncidentClass.RESOURCE_EXHAUSTION,
            severity=IncidentSeverity.CRITICAL,
            fingerprint=_FP_ETH,
            source_system="wallet",
            error_type="ETHBelowGasThreshold",
            error_message=(
                f"ETH balance {eth_balance:.8f} ETH is below critical gas "
                f"threshold {self._gas_threshold:.5f} ETH. On-chain operations "
                "are blocked. Organism cannot execute transactions or earn revenue."
            ),
            context={
                "eth_balance": eth_balance,
                "gas_threshold_eth": self._gas_threshold,
                "repair_strategy": _REPAIR_STRATEGY,
                "sentinel": "bankruptcy",
                "failure_mode": "eth_gas_exhaustion",
            },
            affected_systems=["wallet", "synapse", "nova", "voxis", "federation"],
            blast_radius=0.9,  # Near-total: revenue pathways are severed
            user_visible=True,
            constitutional_impact={
                "coherence": 0.4,
                "care": 0.8,   # Cannot respond to users without revenue
                "growth": 0.9, # All learning/evolution halted
                "honesty": 0.2,
            },
            repair_tier=RepairTier.ESCALATE,
        )

    def check_api_deficit(self, deficit_usd: float) -> Incident | None:
        """
        Return a CRITICAL incident if the rolling fiat deficit exceeds the
        hard limit, otherwise return None.

        Args:
            deficit_usd: Current rolling deficit in USD from
                ``SynapseService.metabolic_deficit``.
        """
        if deficit_usd < self._deficit_limit:
            return None

        self._logger.warning(
            "bankruptcy_api_deficit_critical",
            deficit_usd=deficit_usd,
            limit_usd=self._deficit_limit,
        )
        return Incident(
            incident_class=IncidentClass.RESOURCE_EXHAUSTION,
            severity=IncidentSeverity.CRITICAL,
            fingerprint=_FP_API,
            source_system="synapse",
            error_type="APIDeficitHardLimit",
            error_message=(
                f"Rolling API deficit ${deficit_usd:.4f} USD exceeds hard limit "
                f"${self._deficit_limit:.2f} USD. Continued operation accelerates "
                "insolvency. Discretionary cognition must be suspended immediately."
            ),
            context={
                "deficit_usd": deficit_usd,
                "limit_usd": self._deficit_limit,
                "repair_strategy": _REPAIR_STRATEGY,
                "sentinel": "bankruptcy",
                "failure_mode": "api_deficit_exceeded",
            },
            affected_systems=["synapse", "nova", "evo", "oneiros", "simula"],
            blast_radius=0.7,  # High: all LLM-backed systems affected
            user_visible=False,  # Financial state is internal
            constitutional_impact={
                "coherence": 0.3,
                "care": 0.5,
                "growth": 1.0,  # Growth drive is directly starved
                "honesty": 0.1,
            },
            repair_tier=RepairTier.ESCALATE,
        )

    def check(
        self,
        eth_balance: float,
        deficit_usd: float,
    ) -> list[Incident]:
        """
        Run both solvency checks and return all triggered incidents.

        Convenience wrapper for the ThymosService homeostasis tick, which
        fetches both values once and passes them in together.

        Args:
            eth_balance: Current ETH balance in whole ETH units.
            deficit_usd: Current rolling API deficit in USD.

        Returns a list with 0, 1, or 2 incidents.
        """
        incidents: list[Incident] = []

        eth_incident = self.check_eth_balance(eth_balance)
        if eth_incident is not None:
            incidents.append(eth_incident)

        api_incident = self.check_api_deficit(deficit_usd)
        if api_incident is not None:
            incidents.append(api_incident)

        return incidents


# -- Threat Pattern Sentinel (Layer 2: Economic Immune System) ------


# Default threat detection patterns
_DEFAULT_THREAT_PATTERNS: list[ThreatPattern] = [
    ThreatPattern(
        threat_type=ThreatType.FLASH_LOAN_ATTACK,
        description="Single transaction value exceeds $100K",
        detection_rule="value_usd > 100_000 in single tx",
        severity=ThreatSeverity.CRITICAL,
        confidence=0.7,
        false_positive_rate=0.15,
    ),
    ThreatPattern(
        threat_type=ThreatType.ORACLE_MANIPULATION,
        description="Oracle price deviates >5% from reference",
        detection_rule="abs(oracle_price - reference_price) / reference_price > 0.05",
        severity=ThreatSeverity.HIGH,
        confidence=0.85,
        false_positive_rate=0.05,
    ),
    ThreatPattern(
        threat_type=ThreatType.SUSPICIOUS_CONTRACT,
        description="Interaction with unverified or recently deployed contract",
        detection_rule="contract_age_days < 7 and not verified",
        severity=ThreatSeverity.MEDIUM,
        confidence=0.6,
        false_positive_rate=0.20,
    ),
]


class ThreatPatternSentinel(BaseThymosSentinel):
    """
    Detects economic threats by matching transaction and on-chain event
    patterns against a library of known attack signatures.

    Maintains an address blacklist shared with the TransactionShield.
    Creates ECONOMIC_THREAT incidents when patterns match.

    Layer 2 of the economic immune system.
    """

    @property
    def sentinel_name(self) -> str:
        return "threat_pattern"

    def __init__(
        self,
        patterns: list[ThreatPattern] | None = None,
    ) -> None:
        self._patterns = list(patterns or _DEFAULT_THREAT_PATTERNS)
        self._blacklist: dict[str, AddressBlacklistEntry] = {}
        self._logger = logger.bind(system="thymos", component="threat_pattern_sentinel")

    def check_transaction(
        self,
        to: str,
        from_addr: str,
        data: str,
        value_usd: float,
        chain_id: int = 8453,
    ) -> Incident | None:
        """
        Check if a transaction matches any known threat pattern.

        Called by the shield or directly during post-execution analysis.
        """
        # Check blacklist first
        if self.is_blacklisted(to):
            fp = hashlib.sha256(
                f"threat:blacklisted:{to}".encode()
            ).hexdigest()[:16]
            return Incident(
                incident_class=IncidentClass.ECONOMIC_THREAT,
                severity=IncidentSeverity.CRITICAL,
                fingerprint=fp,
                source_system="oikos",
                error_type="BlacklistedAddress",
                error_message=f"Transaction to blacklisted address {to}",
                context={
                    "to": to,
                    "from": from_addr,
                    "value_usd": value_usd,
                    "chain_id": chain_id,
                    "sentinel": "threat_pattern",
                },
                affected_systems=["oikos", "axon"],
                blast_radius=0.6,
                user_visible=False,
                repair_tier=RepairTier.ESCALATE,
            )

        # Check flash loan pattern (high-value single tx)
        for pattern in self._patterns:
            if (
                pattern.threat_type == ThreatType.FLASH_LOAN_ATTACK
                and value_usd > 100_000
            ):
                fp = hashlib.sha256(
                    f"threat:flash_loan:{to}:{value_usd}".encode()
                ).hexdigest()[:16]
                return Incident(
                    incident_class=IncidentClass.ECONOMIC_THREAT,
                    severity=IncidentSeverity.CRITICAL,
                    fingerprint=fp,
                    source_system="oikos",
                    error_type="FlashLoanSuspected",
                    error_message=(
                        f"Suspected flash loan: ${value_usd:,.0f} single "
                        f"transaction to {to}"
                    ),
                    context={
                        "to": to,
                        "from": from_addr,
                        "value_usd": value_usd,
                        "chain_id": chain_id,
                        "pattern_id": pattern.pattern_id,
                        "sentinel": "threat_pattern",
                    },
                    affected_systems=["oikos", "axon"],
                    blast_radius=0.7,
                    user_visible=False,
                    repair_tier=RepairTier.ESCALATE,
                )

        return None

    def check_on_chain_event(
        self,
        event_type: str,
        contract_address: str,
        event_data: dict[str, Any],
    ) -> Incident | None:
        """Check an on-chain event against threat patterns."""
        # Oracle manipulation detection
        if event_type == "price_update":
            oracle_price = event_data.get("price", 0)
            reference_price = event_data.get("reference_price", 0)

            if reference_price > 0:
                deviation = abs(oracle_price - reference_price) / reference_price
                if deviation > 0.05:
                    fp = hashlib.sha256(
                        f"threat:oracle:{contract_address}".encode()
                    ).hexdigest()[:16]
                    return Incident(
                        incident_class=IncidentClass.ECONOMIC_THREAT,
                        severity=IncidentSeverity.HIGH,
                        fingerprint=fp,
                        source_system="oikos",
                        error_type="OracleDeviation",
                        error_message=(
                            f"Oracle deviation {deviation:.1%} on "
                            f"{contract_address}"
                        ),
                        context={
                            "contract": contract_address,
                            "oracle_price": oracle_price,
                            "reference_price": reference_price,
                            "deviation": deviation,
                            "sentinel": "threat_pattern",
                        },
                        affected_systems=["oikos"],
                        blast_radius=0.4,
                        user_visible=False,
                    )

        return None

    def add_pattern(self, pattern: ThreatPattern) -> None:
        """Register a new threat detection pattern."""
        self._patterns.append(pattern)
        self._logger.info(
            "threat_pattern_added",
            pattern_id=pattern.pattern_id,
            threat_type=pattern.threat_type,
        )

    def add_to_blacklist(self, entry: AddressBlacklistEntry) -> None:
        """Add an address to the sentinel's blacklist."""
        key = entry.address.lower()
        self._blacklist[key] = entry
        self._logger.info(
            "address_blacklisted_by_sentinel",
            address=entry.address,
            reason=entry.reason,
        )

    def is_blacklisted(self, address: str) -> bool:
        return address.lower() in self._blacklist


# -- Protocol Health Sentinel (Layer 3: Economic Immune System) -----

# Stable fingerprint prefixes for deduplication
_FP_TVL = "protocol_health:tvl_drop"
_FP_ORACLE = "protocol_health:oracle_deviation"
_FP_PAUSED = "protocol_health:contract_paused"
_FP_GOVERNANCE = "protocol_health:governance_anomaly"


class ProtocolHealthSentinel(BaseThymosSentinel):
    """
    Monitors the health of DeFi protocols where the organism has capital.

    Tracks TVL, oracle prices, contract pause status, and governance
    activity. Creates PROTOCOL_DEGRADATION incidents when thresholds
    are breached.

    CRITICAL incidents include ``context["requires_withdrawal"] = True``
    so the immune pipeline can trigger emergency withdrawal intents.

    Layer 3 of the economic immune system.
    """

    @property
    def sentinel_name(self) -> str:
        return "protocol_health"

    def __init__(
        self,
        tvl_drop_threshold: float = 0.20,
        oracle_deviation_threshold: float = 0.05,
    ) -> None:
        self._tvl_drop_threshold = tvl_drop_threshold
        self._oracle_deviation_threshold = oracle_deviation_threshold
        self._logger = logger.bind(system="thymos", component="protocol_health_sentinel")

    def check_protocol(
        self,
        protocol: str,
        protocol_address: str,
        current_tvl: float,
        deposit_tvl: float,
        oracle_price: float = 0.0,
        reference_price: float = 0.0,
        is_paused: bool = False,
    ) -> Incident | None:
        """
        Check a single protocol's health metrics.

        Returns an Incident if any threshold is breached, None otherwise.
        """
        # -- Contract paused (CRITICAL) ----------------------------------------
        if is_paused:
            fp = hashlib.sha256(
                f"{_FP_PAUSED}:{protocol}:{protocol_address}".encode()
            ).hexdigest()[:16]
            return Incident(
                incident_class=IncidentClass.PROTOCOL_DEGRADATION,
                severity=IncidentSeverity.CRITICAL,
                fingerprint=fp,
                source_system="oikos",
                error_type="ContractPaused",
                error_message=f"Protocol {protocol} contract is paused at {protocol_address}",
                context={
                    "protocol": protocol,
                    "protocol_address": protocol_address,
                    "requires_withdrawal": True,
                    "sentinel": "protocol_health",
                },
                affected_systems=["oikos", "axon"],
                blast_radius=0.5,
                user_visible=False,
                repair_tier=RepairTier.ESCALATE,
            )

        # -- TVL drop (CRITICAL if >threshold) ---------------------------------
        if deposit_tvl > 0:
            tvl_change = (current_tvl - deposit_tvl) / deposit_tvl
            if tvl_change < -self._tvl_drop_threshold:
                fp = hashlib.sha256(
                    f"{_FP_TVL}:{protocol}:{protocol_address}".encode()
                ).hexdigest()[:16]
                return Incident(
                    incident_class=IncidentClass.PROTOCOL_DEGRADATION,
                    severity=IncidentSeverity.CRITICAL,
                    fingerprint=fp,
                    source_system="oikos",
                    error_type="TVLDrop",
                    error_message=(
                        f"Protocol {protocol} TVL dropped {abs(tvl_change):.0%} "
                        f"(${current_tvl:,.0f} vs ${deposit_tvl:,.0f} at deposit)"
                    ),
                    context={
                        "protocol": protocol,
                        "protocol_address": protocol_address,
                        "current_tvl": current_tvl,
                        "deposit_tvl": deposit_tvl,
                        "tvl_change_pct": tvl_change,
                        "threshold": self._tvl_drop_threshold,
                        "requires_withdrawal": True,
                        "sentinel": "protocol_health",
                    },
                    affected_systems=["oikos", "axon"],
                    blast_radius=0.6,
                    user_visible=False,
                    repair_tier=RepairTier.ESCALATE,
                )

        # -- Oracle deviation (HIGH) -------------------------------------------
        if reference_price > 0 and oracle_price > 0:
            deviation = abs(oracle_price - reference_price) / reference_price
            if deviation > self._oracle_deviation_threshold:
                fp = hashlib.sha256(
                    f"{_FP_ORACLE}:{protocol}:{protocol_address}".encode()
                ).hexdigest()[:16]
                return Incident(
                    incident_class=IncidentClass.PROTOCOL_DEGRADATION,
                    severity=IncidentSeverity.HIGH,
                    fingerprint=fp,
                    source_system="oikos",
                    error_type="OracleDeviation",
                    error_message=(
                        f"Protocol {protocol} oracle deviation {deviation:.1%} "
                        f"(${oracle_price:.4f} vs ref ${reference_price:.4f})"
                    ),
                    context={
                        "protocol": protocol,
                        "protocol_address": protocol_address,
                        "oracle_price": oracle_price,
                        "reference_price": reference_price,
                        "deviation": deviation,
                        "threshold": self._oracle_deviation_threshold,
                        "requires_withdrawal": False,
                        "sentinel": "protocol_health",
                    },
                    affected_systems=["oikos"],
                    blast_radius=0.3,
                    user_visible=False,
                )

        return None

    def check_all_positions(
        self,
        positions: list[Any],
        live_data: dict[str, dict[str, Any]],
    ) -> list[Incident]:
        """
        Check all active yield positions against live protocol data.

        Args:
            positions: List of YieldPosition objects from the economic state.
            live_data: Dict keyed by protocol_address with live metrics:
                       {"tvl_usd": float, "oracle_price": float,
                        "reference_price": float, "is_paused": bool}
        """
        incidents: list[Incident] = []

        for pos in positions:
            protocol = getattr(pos, "protocol", "")
            protocol_address = getattr(pos, "protocol_address", "")
            if not protocol_address:
                continue

            data = live_data.get(protocol_address, {})
            if not data:
                continue

            incident = self.check_protocol(
                protocol=protocol,
                protocol_address=protocol_address,
                current_tvl=float(data.get("tvl_usd", 0)),
                deposit_tvl=float(getattr(pos, "tvl_usd_at_deposit", 0)),
                oracle_price=float(data.get("oracle_price", 0)),
                reference_price=float(data.get("reference_price", 0)),
                is_paused=bool(data.get("is_paused", False)),
            )
            if incident is not None:
                incidents.append(incident)

        return incidents
