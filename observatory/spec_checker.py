"""
EcodiaOS - Spec Compliance Checker

Compares the full set of SynapseEventType enum values against
what the EventTracer has actually observed. Reports:
- Events defined but never seen ("missing")
- Events seen but not in the enum ("unknown" - shouldn't happen)
- Per-system coverage (how many of its spec'd events actually fire)
"""

from __future__ import annotations

from typing import Any

from systems.synapse.types import SynapseEventType

from observatory.tracer import EventTracer

# Map each system to the event types it should emit.
# This is the source of truth for "what should we see?"
# Populated from the spec - event types whose source_system is known.
SYSTEM_EXPECTED_EVENTS: dict[str, list[str]] = {
    "synapse": [
        "cycle_completed", "soma_tick", "clock_started", "clock_stopped",
        "clock_paused", "clock_resumed", "clock_overrun",
        "coherence_shift", "coherence_snapshot",
        "resource_rebalanced", "resource_pressure",
        "rhythm_state_changed", "system_overloaded", "system_degraded",
        "safe_mode_entered", "safe_mode_exited",
    ],
    "soma": [
        "interoceptive_percept", "somatic_drive_vector",
        "somatic_modulation_signal", "soma_vitality_signal",
        "soma_allostatic_report", "soma_state_spike",
    ],
    "oikos": [
        "metabolic_pressure", "metabolic_snapshot",
        "budget_exhausted", "funding_request_issued",
        "revenue_injected", "bounty_paid",
        "metabolic_gate_response",
        "economic_action_deferred", "starvation_warning",
        "yield_deployment_request",
    ],
    "thymos": [
        "repair_completed", "system_failed", "system_recovered",
        "system_restarting", "immune_pattern_advisory",
        "tier5_auto_approval", "immune_cycle_complete",
        "incident_detected",
    ],
    "nova": [
        "belief_updated", "policy_selected",
        "nova_degraded", "nova_belief_stabilised",
        "nova_goal_injected", "autonomy_insufficient",
    ],
    "axon": [
        "action_completed", "axon_shield_rejection",
        "motor_degradation_detected",
    ],
    "equor": [
        "equor_review_started", "equor_review_completed",
        "equor_drift_warning", "equor_drive_weights_updated",
        "equor_alignment_score", "equor_fast_path_hit",
        "equor_escalated_to_human", "equor_deferred",
        "equor_constitutional_snapshot",
        "constitutional_drift_detected", "intent_rejected",
        # Autonomy lifecycle (v1.3+)
        "equor_autonomy_promoted", "equor_autonomy_demoted",
        "equor_safe_mode_entered", "equor_hitl_approved",
        # Amendment self-proposal (SG5 - expected once implemented)
        "equor_amendment_proposed",
    ],
    "simula": [
        "evolution_applied", "evolution_rolled_back",
        "evo_repair_postmortem", "simula_calibration_degraded",
        "goal_hygiene_complete", "simula_rollback_penalty",
        # Missing events being added (gap closure)
        "evolution_rejected", "evolution_awaiting_governance",
        "simula_health_degraded", "inspector_vulnerability_found",
        "simula_genome_extracted",
        # RE training signal
        "re_training_example",
    ],
    "evo": [
        "evolution_candidate", "evo_hypothesis_created",
        "evo_hypothesis_confirmed", "evo_hypothesis_refuted",
        "evo_consolidation_complete", "evo_capability_emerged",
        "evo_drift_data", "evo_degraded", "evo_consolidation_stalled",
        "evo_hypothesis_quality", "evo_epistemic_intent_proposed",
        "evolutionary_observable", "fitness_observable_batch",
        "oikos_param_adjust",
        "genome_extract_request",
    ],
    "telos": [
        "effective_i_computed", "alignment_gap_warning",
        "care_coverage_gap", "coherence_cost_elevated",
        "growth_stagnation", "honesty_validity_low",
        "constitutional_topology_intact",
        "telos_objective_threatened", "telos_autonomy_stagnating",
        # Population-level speciation signal (gap closure)
        "telos_population_snapshot",
    ],
    "fovea": [
        "fovea_prediction_error", "fovea_habituation_decay",
        "fovea_dishabituation", "fovea_workspace_ignition",
        "fovea_attention_profile_update", "fovea_habituation_complete",
        "fovea_internal_prediction_error",
    ],
    "atune": [
        "atune_repair_validation",
        "percept_arrived",
    ],
    "logos": [
        "cognitive_pressure", "intelligence_metrics",
        "compression_cycle_complete", "anchor_memory_created",
        "world_model_updated",
    ],
    "kairos": [
        "kairos_causal_candidate_generated",
        "kairos_causal_direction_accepted",
        "kairos_confounder_discovered",
        "kairos_invariant_candidate",
        "kairos_invariant_distilled",
        "kairos_tier3_invariant_discovered",
        "kairos_counter_invariant_found",
        "kairos_intelligence_ratio_step_change",
        "kairos_validated_causal_structure",
        "kairos_spurious_hypothesis_class",
        "kairos_invariant_absorption_requested",
        "kairos_causal_novelty_detected",
        "kairos_health_degraded",
        "kairos_violation_escalation",
    ],
    "oneiros": [
        "sleep_initiated", "sleep_stage_transition",
        "compression_backlog_processed", "causal_graph_reconstructed",
        "cross_domain_match_found", "analogy_discovered",
        "dream_hypotheses_generated", "lucid_dream_result",
        "wake_initiated", "oneiros_genome_ready",
        "oneiros_sleep_cycle_summary",
        # Threat simulation (gap closure)
        "oneiros_threat_scenario",
    ],
    "nexus": [
        "fragment_shared", "convergence_detected",
        "divergence_pressure", "triangulation_weight_update",
        "speciation_event", "ground_truth_candidate",
        "empirical_invariant_confirmed",
        "nexus_epistemic_value",
    ],
    "memory": [
        "episode_stored", "belief_consolidated",
        "self_affect_updated", "memory_pressure",
        "self_state_drifted",
    ],
    "skia": [
        "skia_heartbeat", "skia_heartbeat_lost",
        "skia_snapshot_completed", "skia_restoration_triggered",
        "skia_restoration_started", "skia_restoration_complete",
        "organism_spawned", "vitality_report", "vitality_fatal",
        "vitality_restored", "organism_died", "organism_resurrected",
    ],
    "thread": [
        "narrative_milestone",
        # Chapter lifecycle (gap closure)
        "chapter_opened", "chapter_closed",
    ],
    "benchmarks": [
        "benchmark_regression", "benchmark_re_progress",
        "benchmark_recovery", "bedau_packard_snapshot",
        # Evolutionary activity statistics
        "benchmarks_evolutionary_activity",
    ],
    "identity": [
        "certificate_expiring", "certificate_expired",
        "connector_authenticated", "connector_token_refreshed",
        "connector_token_expired", "connector_revoked",
        "connector_error",
    ],
    "federation": [
        "federation_link_established", "federation_link_dropped",
        "federation_trust_updated", "federation_knowledge_shared",
        "federation_knowledge_received", "federation_invariant_received",
        "world_model_fragment_share",
    ],
    "mitosis": [
        "child_spawned", "child_health_report", "child_struggling",
        "child_rescued", "child_independent", "child_died",
        "dividend_received",
    ],
    "phantom": [
        "phantom_price_observation",
        "phantom_substrate_observable",
    ],
    "sacm": [
        "compute_request_allocated", "compute_request_queued",
        "compute_request_denied", "compute_capacity_exhausted",
        "allocation_released",
    ],
    "voxis": [],
    # EIS - epistemic immune system
    "eis": [
        "threat_detected",
        "percept_quarantined",
        "eis_layer_triggered",
    ],
    # Alive - WebSocket telemetry bridge
    "alive": [],
}


class SpecComplianceChecker:
    """
    Compares spec-defined event types against observed events.
    """

    def __init__(self, tracer: EventTracer) -> None:
        self._tracer = tracer

    def check(self) -> dict[str, Any]:
        """Run full compliance check."""
        all_enum_values = {e.value for e in SynapseEventType}
        observed = set(self._tracer._per_type.keys())

        missing = sorted(all_enum_values - observed)
        unknown = sorted(observed - all_enum_values)

        # Per-system coverage
        system_coverage: list[dict[str, Any]] = []
        for sys_id, expected in sorted(SYSTEM_EXPECTED_EVENTS.items()):
            if not expected:
                system_coverage.append({
                    "system": sys_id,
                    "expected": 0,
                    "observed": 0,
                    "coverage_pct": None,
                    "missing": [],
                })
                continue

            seen = [e for e in expected if e in observed]
            not_seen = [e for e in expected if e not in observed]
            pct = round(100 * len(seen) / len(expected), 1) if expected else 0

            system_coverage.append({
                "system": sys_id,
                "expected": len(expected),
                "observed": len(seen),
                "coverage_pct": pct,
                "missing": not_seen,
            })

        # Sort by coverage ascending (worst first)
        system_coverage.sort(key=lambda x: x.get("coverage_pct") or 0)

        return {
            "total_defined": len(all_enum_values),
            "total_observed": len(observed),
            "missing_count": len(missing),
            "unknown_count": len(unknown),
            "missing_events": missing,
            "unknown_events": unknown,
            "per_system_coverage": system_coverage,
        }
