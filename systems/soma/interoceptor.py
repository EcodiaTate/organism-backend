"""
EcodiaOS - Soma Interoceptor

Reads from all cognitive systems every theta cycle to compose the
9-dimensional sensed interoceptive state. All reads are in-memory -
no database, no LLM, no network calls. Total sensing budget: 2ms.

Each interoceptor maps a system reference to one dimension with a
transform function and fallback value for when the source is unavailable.
"""

from __future__ import annotations

from typing import Any

import structlog

from systems.soma.types import (
    ALL_DIMENSIONS,
    InteroceptiveDimension,
    _clamp_dimension,
)

logger = structlog.get_logger("systems.soma.interoceptor")


# ─── Fallback Values ──────────────────────────────────────────────

FALLBACK_VALUES: dict[InteroceptiveDimension, float] = {
    InteroceptiveDimension.ENERGY: 0.5,
    InteroceptiveDimension.AROUSAL: 0.4,
    InteroceptiveDimension.VALENCE: 0.0,
    InteroceptiveDimension.CONFIDENCE: 0.5,
    InteroceptiveDimension.COHERENCE: 0.5,
    InteroceptiveDimension.SOCIAL_CHARGE: 0.3,
    InteroceptiveDimension.CURIOSITY_DRIVE: 0.5,
    InteroceptiveDimension.INTEGRITY: 0.8,
    InteroceptiveDimension.TEMPORAL_PRESSURE: 0.15,
}


class Interoceptor:
    """
    Composes the 9D sensed interoceptive state by reading from system references.

    All system references are set via set_*() methods during initialization.
    Missing references gracefully degrade to fallback values.
    """

    def __init__(self) -> None:
        # System references - set during wiring
        self._atune: Any = None
        self._synapse: Any = None
        self._nova: Any = None
        self._thymos: Any = None
        self._equor: Any = None
        self._token_budget: Any = None  # Synapse resource manager
        # Cross-modal synesthesia: SomaService reference for external stress read-back
        self._soma: Any = None
        # Telos reference for Loop 6: confidence + coherence augmentation
        self._telos: Any = None

        # ── AUTONOMY: Cross-system telemetry refs ──
        # These give the organism visibility into systems that were previously
        # invisible to somatic sensing, closing critical blind spots.
        self._fovea: Any = None     # Prediction error by domain
        self._simula: Any = None    # Self-modification repair latency
        self._axon: Any = None      # Compute cost / execution stats (SACM)
        self._logos: Any = None     # Compression quality / cognitive pressure
        self._evo: Any = None       # Learning velocity / hypothesis success
        self._benchmarks: Any = None  # Performance regression detection

    def set_atune(self, atune: Any) -> None:
        self._atune = atune

    def set_synapse(self, synapse: Any) -> None:
        self._synapse = synapse

    def set_nova(self, nova: Any) -> None:
        self._nova = nova

    def set_thymos(self, thymos: Any) -> None:
        self._thymos = thymos

    def set_equor(self, equor: Any) -> None:
        self._equor = equor

    def set_token_budget(self, budget: Any) -> None:
        self._token_budget = budget

    def set_soma(self, soma: Any) -> None:
        """Back-reference to SomaService for reading external_stress."""
        self._soma = soma

    def set_telos(self, telos: Any) -> None:
        """Wire Telos for confidence + coherence augmentation (Loop 6)."""
        self._telos = telos

    # ── AUTONOMY: Cross-system telemetry wiring ──

    def set_fovea(self, fovea: Any) -> None:
        """Wire Fovea for prediction error domain visibility."""
        self._fovea = fovea

    def set_simula(self, simula: Any) -> None:
        """Wire Simula for self-modification repair latency visibility."""
        self._simula = simula

    def set_axon(self, axon: Any) -> None:
        """Wire Axon for compute cost / execution stats (SACM proxy)."""
        self._axon = axon

    def set_logos(self, logos: Any) -> None:
        """Wire Logos for compression quality / cognitive pressure."""
        self._logos = logos

    def set_evo(self, evo: Any) -> None:
        """Wire Evo for learning velocity / hypothesis success rate."""
        self._evo = evo

    def set_benchmarks(self, benchmarks: Any) -> None:
        """Wire Benchmarks for performance regression detection."""
        self._benchmarks = benchmarks

    def sense(self) -> dict[InteroceptiveDimension, float]:
        """
        Read all 9 interoceptive dimensions from system references.

        Returns a dict mapping each dimension to its current sensed value,
        clamped to valid ranges. Total budget: <=2ms.
        """
        state: dict[InteroceptiveDimension, float] = {}

        state[InteroceptiveDimension.ENERGY] = self._sense_energy()
        state[InteroceptiveDimension.AROUSAL] = self._sense_arousal()
        state[InteroceptiveDimension.VALENCE] = self._sense_valence()
        state[InteroceptiveDimension.CONFIDENCE] = self._sense_confidence()
        state[InteroceptiveDimension.COHERENCE] = self._sense_coherence()
        state[InteroceptiveDimension.SOCIAL_CHARGE] = self._sense_social_charge()
        state[InteroceptiveDimension.CURIOSITY_DRIVE] = self._sense_curiosity_drive()
        state[InteroceptiveDimension.INTEGRITY] = self._sense_integrity()
        state[InteroceptiveDimension.TEMPORAL_PRESSURE] = self._sense_temporal_pressure()

        # Ensure all dimensions clamped to valid ranges
        for dim in ALL_DIMENSIONS:
            state[dim] = _clamp_dimension(dim, state[dim])

        return state

    # ─── Per-Dimension Readers ────────────────────────────────────

    def _sense_energy(self) -> float:
        """
        ENERGY: 1.0 - token_budget.utilization, enriched with compute cost awareness.

        AUTONOMY: Now also reads Axon execution stats (mean_cost_usd) so the
        organism feels compute expense as energy drain, not just token usage.
        """
        base: float | None = None
        try:
            if self._token_budget is not None:
                status = (
                    self._token_budget.cached_status
                    if hasattr(self._token_budget, "cached_status")
                    else None
                )
                if status is None and hasattr(self._token_budget, "get_status"):
                    status = self._token_budget.get_status()
                if status is not None:
                    if hasattr(status, "utilization"):
                        base = 1.0 - float(status.utilization)
                    elif isinstance(status, dict):
                        base = 1.0 - float(status.get("utilization", 0.5))
                    elif hasattr(status, "tokens_used") and hasattr(status, "tokens_remaining"):
                        total = status.tokens_used + status.tokens_remaining
                        base = 1.0 - (status.tokens_used / total) if total > 0 else 0.5
            if base is None and self._synapse is not None and hasattr(self._synapse, "_resources"):
                resources = self._synapse._resources
                if hasattr(resources, "get_status"):
                    status = resources.get_status()
                    if hasattr(status, "utilization"):
                        base = 1.0 - float(status.utilization)
        except Exception:
            pass

        # AUTONOMY: Axon compute cost enrichment - high cost per action drains energy
        compute_drain: float = 0.0
        try:
            if self._axon is not None and hasattr(self._axon, "executor_stats"):
                stats = self._axon.executor_stats
                if stats is not None:
                    mean_cost = float(stats.get("mean_cost_usd", 0.0) if isinstance(stats, dict) else getattr(stats, "mean_cost_usd", 0.0))
                    # $0.01/action = 0.0 drain; $0.10/action = 0.15 drain (significant)
                    compute_drain = min(0.2, mean_cost * 1.5)
        except Exception:
            pass

        if base is not None:
            return max(0.0, base - compute_drain)
        return FALLBACK_VALUES[InteroceptiveDimension.ENERGY]

    def _sense_arousal(self) -> float:
        """AROUSAL: direct from Atune affect_manager.current_affect.arousal."""
        try:
            if self._atune is not None:
                affect = self._get_current_affect()
                if affect is not None:
                    return float(affect.arousal)
        except Exception:
            pass
        return FALLBACK_VALUES[InteroceptiveDimension.AROUSAL]

    def _sense_valence(self) -> float:
        """VALENCE: direct from Atune affect_manager.current_affect.valence."""
        try:
            if self._atune is not None:
                affect = self._get_current_affect()
                if affect is not None:
                    return float(affect.valence)
        except Exception:
            pass
        return FALLBACK_VALUES[InteroceptiveDimension.VALENCE]

    def _sense_confidence(self) -> float:
        """CONFIDENCE: 1.0 - clamp(mean_prediction_error, 0, 1).

        Augmented with:
        - Telos effective_I (Loop 6, 20% weight)
        - Fovea per-domain error distribution (AUTONOMY, 10% weight)
        - Evo learning velocity (AUTONOMY, blended into base)

        The organism feels confident when: prediction errors are low, drives
        are aligned, error distribution is narrow, and learning is progressing.
        """
        base: float | None = None
        try:
            if self._atune is not None:
                if hasattr(self._atune, "mean_prediction_error"):
                    mpe = float(self._atune.mean_prediction_error)
                    base = 1.0 - max(0.0, min(1.0, mpe))
                else:
                    affect = self._get_current_affect()
                    if affect is not None:
                        base = 1.0 - max(0.0, min(1.0, float(affect.coherence_stress)))
        except Exception:
            pass

        # Loop 6 augmentation: blend in Telos effective_I
        telos_confidence: float | None = None
        try:
            if self._telos is not None:
                report = self._telos.last_report
                if report is not None:
                    nominal = max(report.nominal_I, 0.001)
                    telos_confidence = max(0.0, min(1.0, report.effective_I / nominal))
        except Exception:
            pass

        # AUTONOMY: Fovea prediction error spread - narrow spread = confident
        fovea_confidence: float | None = None
        try:
            if self._fovea is not None:
                # Read cached error count - sync-safe, no async
                if hasattr(self._fovea, "cached_error_stats"):
                    stats = self._fovea.cached_error_stats
                    if stats is not None and isinstance(stats, dict):
                        # High unprocessed error count = low confidence
                        count = float(stats.get("unprocessed_count", 0))
                        fovea_confidence = max(0.0, 1.0 - min(1.0, count / 50.0))
                elif hasattr(self._fovea, "error_count"):
                    count = float(self._fovea.error_count)
                    fovea_confidence = max(0.0, 1.0 - min(1.0, count / 50.0))
        except Exception:
            pass

        # AUTONOMY: Evo learning velocity - fast learning = confident
        evo_boost: float = 0.0
        try:
            if self._evo is not None and hasattr(self._evo, "hypothesis_success_rate"):
                rate = float(self._evo.hypothesis_success_rate)
                # Success rate > 0.5 → slight confidence boost
                evo_boost = max(0.0, (rate - 0.5) * 0.1)
        except Exception:
            pass

        # Weighted blend: base 55%, telos 20%, fovea 10%, evo 5% (rest = fallback)
        components: list[tuple[float, float]] = []  # (value, weight)
        if base is not None:
            components.append((base + evo_boost, 0.60))
        if telos_confidence is not None:
            components.append((telos_confidence, 0.25))
        if fovea_confidence is not None:
            components.append((fovea_confidence, 0.15))

        if components:
            total_weight = sum(w for _, w in components)
            return sum(v * w for v, w in components) / total_weight
        return FALLBACK_VALUES[InteroceptiveDimension.CONFIDENCE]

    def _sense_coherence(self) -> float:
        """COHERENCE: synapse.coherence_monitor.current_phi (already 0-1).

        Augmented with:
        - Telos alignment_gap (Loop 6, 20% weight)
        - Logos compression quality (AUTONOMY, 10% weight) - high compression
          fidelity = internal consistency

        The organism feels coherent when: systems are integrated (phi), drives
        are aligned (telos), and knowledge compresses cleanly (logos).
        """
        base: float | None = None
        try:
            if self._synapse is not None:
                if hasattr(self._synapse, "_coherence"):
                    coherence = self._synapse._coherence
                    if hasattr(coherence, "current_phi"):
                        phi = coherence.current_phi
                        if phi is not None:
                            base = float(phi)
                if base is None and hasattr(self._synapse, "coherence_snapshot"):
                    snap = self._synapse.coherence_snapshot
                    if hasattr(snap, "phi_approximation"):
                        base = float(snap.phi_approximation)
                    elif hasattr(snap, "phi"):
                        base = float(snap.phi)
        except Exception:
            pass

        # Loop 6 augmentation: narrowing alignment gap → coherence improvement
        telos_coherence: float | None = None
        try:
            if self._telos is not None:
                report = self._telos.last_report
                if report is not None:
                    nominal = max(report.nominal_I, 0.001)
                    gap_fraction = max(0.0, min(1.0, report.alignment_gap / nominal))
                    telos_coherence = 1.0 - gap_fraction
        except Exception:
            pass

        # AUTONOMY: Logos compression quality - clean compression = coherent knowledge
        logos_coherence: float | None = None
        try:
            if self._logos is not None:
                if hasattr(self._logos, "compression_fidelity"):
                    fidelity = float(self._logos.compression_fidelity)
                    logos_coherence = max(0.0, min(1.0, fidelity))
                elif hasattr(self._logos, "cognitive_pressure"):
                    # High cognitive pressure = low coherence (approaching Schwarzschild)
                    pressure = float(self._logos.cognitive_pressure)
                    logos_coherence = max(0.0, 1.0 - min(1.0, pressure))
        except Exception:
            pass

        # Weighted blend
        components: list[tuple[float, float]] = []
        if base is not None:
            components.append((base, 0.60))
        if telos_coherence is not None:
            components.append((telos_coherence, 0.25))
        if logos_coherence is not None:
            components.append((logos_coherence, 0.15))

        if components:
            total_weight = sum(w for _, w in components)
            return sum(v * w for v, w in components) / total_weight
        return FALLBACK_VALUES[InteroceptiveDimension.COHERENCE]

    def _sense_social_charge(self) -> float:
        """SOCIAL_CHARGE: atune.affect_manager.current_affect.care_activation."""
        try:
            if self._atune is not None:
                affect = self._get_current_affect()
                if affect is not None:
                    return float(affect.care_activation)
        except Exception:
            pass
        return FALLBACK_VALUES[InteroceptiveDimension.SOCIAL_CHARGE]

    def _sense_curiosity_drive(self) -> float:
        """CURIOSITY_DRIVE: atune.affect_manager.current_affect.curiosity.

        AUTONOMY: Enriched with Evo hypothesis success rate. When hypotheses
        are succeeding at high rates, curiosity is well-channeled → slight
        boost. When success rate is very low, curiosity may be misdirected
        → slight damping to signal "explore differently, not more".
        """
        base: float | None = None
        try:
            if self._atune is not None:
                affect = self._get_current_affect()
                if affect is not None:
                    base = float(affect.curiosity)
        except Exception:
            pass

        # AUTONOMY: Evo learning progress modulates curiosity signal
        evo_modulation: float = 0.0
        try:
            if self._evo is not None:
                if hasattr(self._evo, "hypothesis_success_rate"):
                    rate = float(self._evo.hypothesis_success_rate)
                    # Success rate 0.3-0.7 is ideal; outside that range, modulate
                    if rate > 0.6:
                        evo_modulation = 0.05  # Learning well → slight curiosity boost
                    elif rate < 0.15:
                        evo_modulation = -0.05  # Misdirected → slight damping
        except Exception:
            pass

        if base is not None:
            return max(0.0, min(1.0, base + evo_modulation))
        return FALLBACK_VALUES[InteroceptiveDimension.CURIOSITY_DRIVE]

    def _sense_integrity(self) -> float:
        """INTEGRITY: min(thymos_health, 1.0 - equor_drift).

        Thymos is the single source for immune health (ThymosService.current_health_score).
        Equor is the single source for constitutional drift (EquorService.constitutional_drift).
        Soma combines both into one unified integrity signal that the rest of the organism reads.

        AUTONOMY: Enriched with Simula self-repair effectiveness. When Simula
        repairs are failing or latency is high, the organism feels structurally
        compromised even if Thymos health looks nominal.
        """
        thymos_health = 1.0
        equor_component = 1.0

        try:
            if self._thymos is not None:
                score = self._thymos.current_health_score
                if score is not None:
                    thymos_health = float(score)
        except Exception:
            pass

        try:
            if self._equor is not None:
                drift = self._equor.constitutional_drift
                if drift is not None:
                    equor_component = 1.0 - max(0.0, min(1.0, float(drift)))
        except Exception:
            pass

        # AUTONOMY: Simula repair effectiveness - failed repairs = structural concern
        simula_integrity: float = 1.0
        try:
            if self._simula is not None:
                if hasattr(self._simula, "repair_success_rate"):
                    rate = float(self._simula.repair_success_rate)
                    # Low repair success = integrity concern (max 0.15 reduction)
                    simula_integrity = 1.0 - max(0.0, min(0.15, (1.0 - rate) * 0.2))
                elif hasattr(self._simula, "overall_performance_delta"):
                    delta = float(self._simula.overall_performance_delta)
                    # Negative delta = performance declining = integrity concern
                    if delta < 0:
                        simula_integrity = max(0.85, 1.0 + delta * 0.15)
        except Exception:
            pass

        return min(thymos_health, equor_component, simula_integrity)

    def _sense_temporal_pressure(self) -> float:
        """
        TEMPORAL_PRESSURE: nova.goal_urgency_max + arousal * 0.3
        + external_stress * 0.3, clamped 0-1.

        The external_stress component is injected by ExternalVolatilitySensor
        and represents cross-modal synesthesia - market / codebase volatility
        translated into felt time pressure.
        """
        goal_urgency = 0.0
        arousal_boost = 0.0
        external_boost = 0.0

        try:
            if self._nova is not None:
                if hasattr(self._nova, "goal_urgency_max"):
                    gu = self._nova.goal_urgency_max
                    if gu is not None:
                        goal_urgency = float(gu)
                elif hasattr(self._nova, "_goal_manager"):
                    gm = self._nova._goal_manager
                    if gm is not None and hasattr(gm, "max_urgency"):
                        goal_urgency = float(gm.max_urgency or 0.0)
        except Exception:
            pass

        try:
            if self._atune is not None:
                affect = self._get_current_affect()
                if affect is not None:
                    arousal_boost = affect.arousal * 0.3
        except Exception:
            pass

        try:
            if self._soma is not None and hasattr(self._soma, "_external_stress"):
                external_boost = float(self._soma._external_stress) * 0.3
        except Exception:
            pass

        return max(0.0, min(1.0, goal_urgency + arousal_boost + external_boost))

    # ─── Helpers ──────────────────────────────────────────────────

    def _get_current_affect(self) -> Any:
        """Get current AffectState from Atune, trying known attribute paths."""
        if self._atune is None:
            return None
        # Path 1: direct attribute
        if hasattr(self._atune, "_affect_mgr"):
            mgr = self._atune._affect_mgr
            if hasattr(mgr, "current"):
                return mgr.current
        # Path 2: public method
        if hasattr(self._atune, "current_affect"):
            ca = self._atune.current_affect
            if callable(ca):
                return ca()
            return ca
        return None
