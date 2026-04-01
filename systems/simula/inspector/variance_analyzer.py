"""
EcodiaOS - Inspector Phase 7: Variance Analyzer (Orchestrator)

Single-entry orchestrator for the Phase 7 execution variance pipeline:

  MeasurementPlanBuilder  → MeasurementPlan   (derived from Phase 4/5/6 artifacts)
  VarianceMeasurer        → list[TrialBatch]   (raw timing observations)
  VarianceProfiler        → list[VarianceProfile] + list[DistinguishabilityResult]
  ChannelSignatureMapper  → list[ChannelSignature]
  → Phase7Result

Usage
-----
  # Full pipeline (Phase 6 → 7):
  analyzer = VarianceAnalyzer()
  result = analyzer.analyze(
      phase6_result=phase6_result,
      phase5_result=phase5_result,   # optional
      phase4_result=phase4_result,   # optional
  )

  # Phase 3 only (no FSM context):
  result = analyzer.analyze(phase3_result=phase3_result)

  # Summary:
  summary = analyzer.model_summary(result)

  # Explain a specific distinguishable channel:
  detail = analyzer.explain_channel(result, result_id)

Exit criterion
--------------
Phase7Result.exit_criterion_met = True when:
  - ≥1 DistinguishabilityResult.is_distinguishable = True
  - ≥1 ChannelSignature maps that result to a higher-layer event
  - claim_supported is set (True or False, not None)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.variance_engine import (
    ChannelSignatureMapper,
    VarianceMeasurer,
    VarianceProfiler,
)
from systems.simula.inspector.variance_types import (
    ChannelKind,
    DistinguishabilityResult,
    DistinguishabilityVerdict,
    IsolationStrategy,
    MeasurementPlan,
    NoiseLevel,
    OperationClass,
    OperationSpec,
    Phase7Result,
)

if TYPE_CHECKING:
    from systems.simula.inspector.constraint_types import Phase4Result
    from systems.simula.inspector.protocol_types import Phase6Result
    from systems.simula.inspector.static_types import Phase3Result
    from systems.simula.inspector.trust_types import Phase5Result

logger = structlog.get_logger().bind(system="simula.inspector.variance_analyzer")


# ── MeasurementPlanBuilder ──────────────────────────────────────────────────────


class MeasurementPlanBuilder:
    """
    Derives a MeasurementPlan from available Phase 4/5/6 artifacts.

    Priority order for operation selection:
    1. Phase 6 boundary states → PROTOCOL_GUARD operations (TIMING_FINE)
    2. Phase 5 trust corridors → AUTH_DECISION operations (TIMING_COARSE)
    3. Phase 4 steerable regions → CONTROL_FLOW operations (TIMING_FINE)
    4. Phase 3 fragments → CRYPTO_BRANCH / SERIALISATION operations (TIMING_COARSE)
    5. Synthetic baseline operations (fallback when no artifact input available)

    Parameters
    ----------
    max_operations    - cap on operations per plan (default 12)
    trials_per_class  - trials per input class per operation (default 1000)
    isolation         - isolation strategy (default CPU_AFFINITY)
    """

    def __init__(
        self,
        max_operations: int = 12,
        trials_per_class: int = 1000,
        isolation: IsolationStrategy = IsolationStrategy.CPU_AFFINITY,
    ) -> None:
        self._max_ops = max_operations
        self._trials = trials_per_class
        self._isolation = isolation

    def build(
        self,
        target_id: str,
        phase3_result: Phase3Result | None = None,
        phase4_result: Phase4Result | None = None,
        phase5_result: Phase5Result | None = None,
        phase6_result: Phase6Result | None = None,
    ) -> MeasurementPlan:
        ops: list[OperationSpec] = []

        # Phase 6: boundary states → PROTOCOL_GUARD measurements
        if phase6_result:
            for fsm in phase6_result.fsms:
                for state in fsm.boundary_states()[:3]:
                    ops.append(OperationSpec(
                        name=f"guard_at_{state.name}",
                        operation_class=OperationClass.PROTOCOL_GUARD,
                        channel_kind=ChannelKind.TIMING_FINE,
                        source_fsm_state_id=state.state_id,
                        hypothesis=(
                            f"Boundary state '{state.name}' guard check runs longer "
                            f"when counter/timer is at boundary value vs nominal value."
                        ),
                    ))
            # Phase 6 auth-window failures → AUTH_DECISION measurements
            for failure in phase6_result.boundary_failures[:2]:
                if "auth" in failure.boundary_kind.value.lower():
                    ops.append(OperationSpec(
                        name=f"auth_decision_{failure.failure_id[:8]}",
                        operation_class=OperationClass.AUTH_DECISION,
                        channel_kind=ChannelKind.TIMING_COARSE,
                        source_fsm_state_id=failure.failing_state_id,
                        hypothesis=(
                            f"Auth-window boundary failure ({failure.boundary_kind.value}) "
                            f"produces timing difference between correct and expired credential."
                        ),
                    ))

        # Phase 5: trust corridors → AUTH_DECISION measurements
        if phase5_result:
            corridors = getattr(phase5_result, "corridors", [])
            for corridor in corridors[:3]:
                c_id = getattr(corridor, "corridor_id", "")
                desc = getattr(corridor, "description", f"trust corridor {c_id}")
                ops.append(OperationSpec(
                    name=f"trust_corridor_{c_id[:8]}",
                    operation_class=OperationClass.AUTH_DECISION,
                    channel_kind=ChannelKind.TIMING_COARSE,
                    source_corridor_id=c_id,
                    hypothesis=(
                        f"Trust propagation via corridor '{desc[:60]}' "
                        f"produces timing difference at privilege boundary."
                    ),
                ))

        # Phase 4: steerable regions → CONTROL_FLOW measurements
        if phase4_result:
            regions = getattr(phase4_result, "steerable_regions", {})
            for region_id, _region in list(regions.items())[:3]:
                ops.append(OperationSpec(
                    name=f"branch_at_{region_id[:8]}",
                    operation_class=OperationClass.CONTROL_FLOW,
                    channel_kind=ChannelKind.TIMING_FINE,
                    source_region_id=region_id,
                    hypothesis=(
                        f"Steerable branch in region {region_id[:8]} "
                        f"produces timing difference between taken and not-taken paths."
                    ),
                ))

        # Phase 3: fragments → CRYPTO_BRANCH / SERIALISATION measurements
        if phase3_result:
            catalog = getattr(phase3_result, "fragment_catalog", None)
            if catalog:
                fragments = getattr(catalog, "fragments", {})
                for frag_id, frag in list(fragments.items())[:2]:
                    sem = getattr(frag, "semantics", None)
                    if sem and "crypto" in str(sem).lower():
                        ops.append(OperationSpec(
                            name=f"crypto_{frag_id[:8]}",
                            operation_class=OperationClass.CRYPTO_BRANCH,
                            channel_kind=ChannelKind.TIMING_FINE,
                            hypothesis=(
                                f"Crypto fragment {frag_id[:8]} may have "
                                f"variable-time comparison path."
                            ),
                        ))

        # Fallback synthetic operations when no artifacts are available
        if not ops:
            ops = _default_synthetic_operations()

        # Trim to max_operations
        ops = ops[: self._max_ops]

        return MeasurementPlan(
            target_id=target_id,
            operations=ops,
            trials_per_class=self._trials,
            warmup_trials=max(20, self._trials // 20),
            isolation_strategy=self._isolation,
            noise_characterisation_trials=min(300, self._trials // 3),
        )


def _default_synthetic_operations() -> list[OperationSpec]:
    """Baseline operations used when no Phase 4/5/6 context is available."""
    return [
        OperationSpec(
            name="hmac_verify_correct_vs_incorrect",
            operation_class=OperationClass.CRYPTO_BRANCH,
            channel_kind=ChannelKind.TIMING_FINE,
            hypothesis=(
                "HMAC verification runs longer for incorrect tokens "
                "due to variable-time string comparison."
            ),
        ),
        OperationSpec(
            name="auth_accept_vs_reject",
            operation_class=OperationClass.AUTH_DECISION,
            channel_kind=ChannelKind.TIMING_COARSE,
            hypothesis=(
                "Authentication rejection path executes different code "
                "than acceptance path, producing measurable latency delta."
            ),
        ),
        OperationSpec(
            name="cache_cold_vs_warm_lookup",
            operation_class=OperationClass.MEMORY_ACCESS,
            channel_kind=ChannelKind.CACHE_L1,
            hypothesis=(
                "Cache-cold lookup of sensitive table entry is "
                "2–5× slower than cache-warm lookup."
            ),
        ),
        OperationSpec(
            name="branch_taken_vs_not_taken",
            operation_class=OperationClass.CONTROL_FLOW,
            channel_kind=ChannelKind.TIMING_FINE,
            hypothesis=(
                "Branch misprediction on secret-dependent conditional "
                "produces measurable pipeline flush overhead."
            ),
        ),
    ]


# ── VarianceAnalyzer ────────────────────────────────────────────────────────────


class VarianceAnalyzer:
    """
    Phase 7 orchestrator - builds a Phase7Result for a target.

    The result combines:
    - A MeasurementPlan derived from Phase 4/5/6 artifacts
    - VarianceProfiles (one per operation)
    - DistinguishabilityResults (verdict per operation)
    - ChannelSignatures (grouped + linked to higher-layer events)
    - Aggregate statistics and exit criterion flag

    Parameters
    ----------
    max_operations      - cap on operations in the plan (default 12)
    trials_per_class    - measurement trials per input class (default 1000)
    significance_level  - p-value threshold (default 0.01)
    min_effect_size_d   - minimum Cohen's d for practical significance (default 0.2)
    isolation_strategy  - scheduling isolation approach (default CPU_AFFINITY)
    seed                - RNG seed for reproducibility (default 0)
    """

    def __init__(
        self,
        max_operations: int = 12,
        trials_per_class: int = 1000,
        significance_level: float = 0.01,
        min_effect_size_d: float = 0.2,
        isolation_strategy: IsolationStrategy = IsolationStrategy.CPU_AFFINITY,
        seed: int = 0,
    ) -> None:
        self._plan_builder = MeasurementPlanBuilder(
            max_operations=max_operations,
            trials_per_class=trials_per_class,
            isolation=isolation_strategy,
        )
        self._measurer = VarianceMeasurer(seed=seed)
        self._profiler = VarianceProfiler(
            significance_level=significance_level,
            min_effect_size_d=min_effect_size_d,
        )
        self._log = logger

    # ── Primary entry point ───────────────────────────────────────────────────

    def analyze(
        self,
        phase3_result: Phase3Result | None = None,
        phase4_result: Phase4Result | None = None,
        phase5_result: Phase5Result | None = None,
        phase6_result: Phase6Result | None = None,
    ) -> Phase7Result:
        """
        Build a Phase7Result from available upstream phase outputs.

        At least one of phase3/4/5/6_result should be provided; if none are
        provided, a synthetic baseline plan is generated.

        Args:
            phase3_result: Phase 3 static analysis output (fragment catalog).
            phase4_result: Phase 4 steerability model (steerable regions).
            phase5_result: Phase 5 trust graph (corridors).
            phase6_result: Phase 6 protocol FSM stress (boundary failures).

        Returns:
            Phase7Result with profiles, distinguishability results,
            channel signatures, and exit criterion flag.
        """
        target_id = (
            getattr(phase6_result, "target_id", None)
            or getattr(phase5_result, "target_id", None)
            or getattr(phase4_result, "target_id", None)
            or getattr(phase3_result, "target_id", None)
            or "unknown"
        )
        log = self._log.bind(target_id=target_id)
        log.info("variance_analysis_started")

        # 1. Build measurement plan
        plan = self._plan_builder.build(
            target_id=target_id,
            phase3_result=phase3_result,
            phase4_result=phase4_result,
            phase5_result=phase5_result,
            phase6_result=phase6_result,
        )

        # 2. Collect raw observations
        batches = self._measurer.collect(plan)

        # 3. Profile: compute stats + distinguishability
        profiles, dr_list = self._profiler.profile_all(batches, plan)

        # 4. Map to channel signatures
        sig_mapper = ChannelSignatureMapper(
            phase4_result=phase4_result,
            phase5_result=phase5_result,
            phase6_result=phase6_result,
        )
        signatures = sig_mapper.map(dr_list, target_id)

        # 5. Aggregate statistics
        total_dist    = sum(1 for dr in dr_list if dr.verdict == DistinguishabilityVerdict.DISTINGUISHABLE)
        total_not     = sum(1 for dr in dr_list if dr.verdict == DistinguishabilityVerdict.NOT_DISTINGUISHABLE)
        total_marg    = sum(1 for dr in dr_list if dr.verdict == DistinguishabilityVerdict.MARGINAL)
        total_incon   = sum(1 for dr in dr_list if dr.verdict == DistinguishabilityVerdict.INCONCLUSIVE)
        sec_channels  = sum(1 for dr in dr_list if dr.is_security_relevant)

        # Phase-artifact linkage counts
        p4_linked = sum(1 for dr in dr_list if dr.profile.source_region_id)
        p5_linked = sum(1 for dr in dr_list if dr.profile.source_corridor_id)
        p6_linked = sum(1 for dr in dr_list if dr.profile.source_fsm_state_id)

        # Dominant noise level
        noise_levels = [dr.noise_level for dr in dr_list]
        dominant_noise = (
            max(set(noise_levels), key=noise_levels.count)
            if noise_levels else NoiseLevel.UNKNOWN
        )

        # 6. Claim verdict
        claim_supported: bool | None
        if total_dist > 0:
            claim_supported = True
        elif total_incon > total_not:
            claim_supported = None   # inconclusive
        else:
            claim_supported = False

        claim_summary = self._build_claim_summary(
            dr_list, total_dist, total_not, total_marg, total_incon, dominant_noise
        )

        # 7. Exit criterion
        any(
            sig.higher_layer_events and dr.is_distinguishable
            for sig in signatures
            for dr in dr_list
            if dr.result_id in sig.distinguishability_result_ids
        )
        exit_met = total_dist >= 1 and len(signatures) >= 1 and claim_supported is not None

        result = Phase7Result(
            target_id=target_id,
            measurement_plan=plan,
            variance_profiles=profiles,
            distinguishability_results=dr_list,
            channel_signatures=signatures,
            total_operations_measured=len(profiles),
            total_distinguishable=total_dist,
            total_not_distinguishable=total_not,
            total_marginal=total_marg,
            total_inconclusive=total_incon,
            total_channel_signatures=len(signatures),
            security_relevant_channels=sec_channels,
            phase4_regions_linked=p4_linked,
            phase5_corridors_linked=p5_linked,
            phase6_failures_linked=p6_linked,
            dominant_noise_level=dominant_noise,
            claim_supported=claim_supported,
            claim_evidence_summary=claim_summary,
            exit_criterion_met=exit_met,
        )

        log.info(
            "variance_analysis_complete",
            operations=len(profiles),
            distinguishable=total_dist,
            not_distinguishable=total_not,
            marginal=total_marg,
            inconclusive=total_incon,
            signatures=len(signatures),
            security_relevant=sec_channels,
            claim_supported=claim_supported,
            exit_criterion_met=exit_met,
        )

        return result

    # ── Targeted queries ──────────────────────────────────────────────────────

    def explain_channel(
        self,
        result: Phase7Result,
        result_id: str,
    ) -> dict:
        """
        Return a structured explanation dict for a specific DistinguishabilityResult.

        Returns a dict with:
          result_id, op_name, operation_class, channel_kind, verdict,
          is_distinguishable, mean_delta_ns, max_effect_size_d,
          noise_level, evidence_summary, confirming_tests, security_relevant,
          linked_signatures, higher_layer_events
        """
        dr = next(
            (dr for dr in result.distinguishability_results if dr.result_id == result_id),
            None,
        )
        if not dr:
            return {"error": f"result '{result_id}' not found"}

        # Find all signatures that reference this result
        linked_sigs = [
            sig for sig in result.channel_signatures
            if result_id in sig.distinguishability_result_ids
        ]
        events = []
        for sig in linked_sigs:
            for e in sig.higher_layer_events:
                events.append({
                    "layer": e.layer,
                    "description": e.description,
                    "sensitivity": e.sensitivity,
                    "phase4_region_id": e.phase4_region_id,
                    "phase5_corridor_id": e.phase5_corridor_id,
                    "phase6_fsm_state_id": e.phase6_fsm_state_id,
                    "phase6_boundary_failure_id": e.phase6_boundary_failure_id,
                })

        p = dr.profile
        return {
            "result_id":           dr.result_id,
            "op_name":             dr.op_name,
            "operation_class":     dr.operation_class.value,
            "channel_kind":        dr.channel_kind.value,
            "verdict":             dr.verdict.value,
            "is_distinguishable":  dr.is_distinguishable,
            "mean_delta_ns":       dr.mean_delta_ns,
            "mean_delta_relative_pct": round(p.mean_delta_relative * 100, 2),
            "class_a_mean_ns":     p.class_a_stats.mean_ns,
            "class_b_mean_ns":     p.class_b_stats.mean_ns,
            "class_a_std_ns":      p.class_a_stats.std_ns,
            "class_b_std_ns":      p.class_b_stats.std_ns,
            "max_effect_size_d":   dr.max_effect_size_d,
            "noise_level":         dr.noise_level.value,
            "noise_cv":            round(p.noise_model.baseline_cv, 4),
            "evidence_summary":    dr.evidence_summary,
            "confirming_tests":    [t.value for t in dr.confirming_tests],
            "non_significant_tests": [t.value for t in dr.non_significant_tests],
            "is_security_relevant": dr.is_security_relevant,
            "security_reason":     dr.security_relevance_reason,
            "linked_signatures":   [sig.signature_id for sig in linked_sigs],
            "higher_layer_events": events,
        }

    def model_summary(self, result: Phase7Result) -> dict:
        """
        Return a concise reporting dict for the Phase7Result.
        """
        # Top distinguishable results
        top_results = sorted(
            result.distinguishability_results,
            key=lambda dr: (dr.is_distinguishable, dr.max_effect_size_d),
            reverse=True,
        )[:5]
        result_summaries = [
            {
                "result_id":         dr.result_id,
                "op_name":           dr.op_name,
                "operation_class":   dr.operation_class.value,
                "channel_kind":      dr.channel_kind.value,
                "verdict":           dr.verdict.value,
                "is_distinguishable": dr.is_distinguishable,
                "mean_delta_ns":     round(dr.mean_delta_ns, 1),
                "max_effect_size_d": round(dr.max_effect_size_d, 3),
                "noise_level":       dr.noise_level.value,
                "security_relevant": dr.is_security_relevant,
            }
            for dr in top_results
        ]

        # Signature summaries
        sig_summaries = [
            {
                "signature_id":    sig.signature_id,
                "operation_class": sig.operation_class.value,
                "channel_kind":    sig.channel_kind.value,
                "n_results":       len(sig.distinguishability_result_ids),
                "n_events":        len(sig.higher_layer_events),
                "confidence":      round(sig.confidence, 3),
                "security_relevant": sig.is_security_relevant,
                "narrative":       sig.signature_narrative[:200],
            }
            for sig in result.channel_signatures
        ]

        # Verdict breakdown
        verdicts: dict[str, int] = {}
        for dr in result.distinguishability_results:
            verdicts[dr.verdict.value] = verdicts.get(dr.verdict.value, 0) + 1

        return {
            "target_id":                  result.target_id,
            "exit_criterion_met":         result.exit_criterion_met,
            "claim_supported":            result.claim_supported,
            "claim_evidence_summary":     result.claim_evidence_summary[:300],
            "total_operations_measured":  result.total_operations_measured,
            "total_distinguishable":      result.total_distinguishable,
            "total_not_distinguishable":  result.total_not_distinguishable,
            "total_marginal":             result.total_marginal,
            "total_inconclusive":         result.total_inconclusive,
            "total_channel_signatures":   result.total_channel_signatures,
            "security_relevant_channels": result.security_relevant_channels,
            "dominant_noise_level":       result.dominant_noise_level.value,
            "phase4_regions_linked":      result.phase4_regions_linked,
            "phase5_corridors_linked":    result.phase5_corridors_linked,
            "phase6_failures_linked":     result.phase6_failures_linked,
            "verdict_breakdown":          verdicts,
            "top_results":                result_summaries,
            "signatures":                 sig_summaries,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_claim_summary(
        dr_list: list[DistinguishabilityResult],
        total_dist: int,
        total_not: int,
        total_marg: int,
        total_incon: int,
        dominant_noise: NoiseLevel,
    ) -> str:
        total = len(dr_list)
        if total == 0:
            return "No operations were measured; claim cannot be assessed."

        if total_dist > 0:
            dist_ops = [dr.op_name for dr in dr_list if dr.is_distinguishable][:3]
            sec_ops  = [dr.op_name for dr in dr_list if dr.is_security_relevant][:2]
            sec_note = (
                f" Security-relevant distinguishable channels: {', '.join(sec_ops)}."
                if sec_ops else ""
            )
            return (
                f"SUPPORTED: {total_dist}/{total} operations are distinguishable "
                f"under {dominant_noise.value} noise. "
                f"Distinguishable operations include: {', '.join(dist_ops)}.{sec_note} "
                f"Marginal: {total_marg}, not distinguishable: {total_not}, "
                f"inconclusive: {total_incon}."
            )
        elif total_not >= total / 2:
            return (
                f"FALSIFIED: {total_not}/{total} operations show no significant "
                f"timing difference under {dominant_noise.value} noise. "
                f"Marginal: {total_marg}, inconclusive: {total_incon}."
            )
        else:
            return (
                f"INCONCLUSIVE: {total_incon}/{total} operations produced marginal "
                f"or insufficient data under {dominant_noise.value} noise. "
                f"Tighter isolation or more trials may resolve. "
                f"Marginal: {total_marg}, not distinguishable: {total_not}."
            )
