"""
EcodiaOS - Thymos Event Payload Validation Models (AV6)

Pydantic v2 models for all Synapse event payloads that Thymos subscribes to.
Validation is NON-BLOCKING: on failure, the raw dict is passed through with
a warning log. This prevents a single malformed event from breaking the
immune system.

Usage in service.py:
    data = validate_event_payload(event, logger)
    # data is either the validated model dict or the raw event.data
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel, Field

from systems.synapse.types import SynapseEvent, SynapseEventType

logger = structlog.get_logger()


# ─── Base ────────────────────────────────────────────────────────────


class _Lenient(BaseModel):
    """All payload models inherit this - extra fields are allowed."""

    model_config = {"extra": "allow"}


# ─── Generic System Events (_SUBSCRIBED_EVENTS) ─────────────────────


class SystemFailedPayload(_Lenient):
    system_id: str
    error: str = ""
    reason: str = ""


class SystemRecoveredPayload(_Lenient):
    system_id: str


class SystemRestartingPayload(_Lenient):
    system_id: str
    reason: str = ""


class SystemOverloadedPayload(_Lenient):
    system_id: str
    load: float = 0.0


class ClockOverrunPayload(_Lenient):
    cycle_number: int = 0
    overrun_ms: float = 0.0


class ResourcePressurePayload(_Lenient):
    resource: str = ""
    utilization: float = 0.0


class TransactionShieldedPayload(_Lenient):
    executor: str = ""
    rejection_reason: str = ""
    check_type: str = ""


class ProtocolAlertPayload(_Lenient):
    protocol: str = ""
    alert_type: str = ""
    severity: str = ""


class CertificateExpiringPayload(_Lenient):
    certificate_id: str = ""
    expires_at: str = ""


class ConnectorErrorPayload(_Lenient):
    connector: str = ""
    error: str = ""


class ModelHotSwapFailedPayload(_Lenient):
    model_id: str = ""
    reason: str = ""


class CatastrophicForgettingPayload(_Lenient):
    model_id: str = ""
    metric: str = ""
    degradation: float = 0.0


# ─── Dedicated Handler Events ───────────────────────────────────────


class IntentRejectedPayload(_Lenient):
    intent_id: str = ""
    reason: str = ""
    drive_scores: dict[str, float] = Field(default_factory=dict)


class InteroceptivePerceptPayload(_Lenient):
    signal_type: str = ""
    magnitude: float = 0.0


class GrowthStagnationPayload(_Lenient):
    di_dt: float = 0.0
    threshold: float = 0.0


class AutonomyInsufficientPayload(_Lenient):
    required_level: str = ""
    current_level: str = ""
    goal_description: str = ""


class MetabolicPressurePayload(_Lenient):
    economic_stress: bool = False
    runway_days: float = 0.0


class TaskPermanentlyFailedPayload(_Lenient):
    task_name: str = ""
    error: str = ""
    restarts: int = 0


class BountyPRSubmittedPayload(_Lenient):
    bounty_id: str = ""
    pr_url: str = ""
    estimated_value_usd: float = 0.0


class SimulaCalibrationDegradedPayload(_Lenient):
    accuracy: float = 0.0
    threshold: float = 0.0


# ─── Sentinel Events ────────────────────────────────────────────────


class RepairCompletedPayload(_Lenient):
    incident_id: str = ""
    repair_tier: str = ""
    success: bool = False


class EvoConsolidationStalledPayload(_Lenient):
    stall_duration_s: float = 0.0


class NovaDegradedPayload(_Lenient):
    metric: str = ""
    value: float = 0.0


class FoveaPredictionErrorPayload(_Lenient):
    metric_name: str = ""
    prediction_error: float = 0.0


class SomaStateSpikePayload(_Lenient):
    signal: str = ""
    value: float = 0.0


class SkiaHeartbeatLostPayload(_Lenient):
    instance_id: str = ""
    last_seen: str = ""


class ThreatDetectedPayload(_Lenient):
    threat_class: str = ""
    severity: str = ""
    source: str = ""


class SpeciationEventPayload(_Lenient):
    peer_id: str = ""
    divergence_score: float = 0.0
    dimensions: dict[str, float] = Field(default_factory=dict)


# ─── Closure / Feedback Events ──────────────────────────────────────


class ConstitutionalDriftPayload(_Lenient):
    drive: str = ""
    drift_magnitude: float = 0.0


class AxonShieldRejectionPayload(_Lenient):
    executor: str = ""
    rejection_reason: str = ""
    check_type: str = ""
    intent_id: str = ""


class AtuneRepairValidationPayload(_Lenient):
    incident_id: str = ""
    validation_result: str = ""


class EvoHypothesisQualityPayload(_Lenient):
    hypothesis_id: str = ""
    quality: float = 0.0


class NovaBeliefStabilisedPayload(_Lenient):
    belief_id: str = ""
    stability: float = 0.0


# ─── Cross-system Cache Events ──────────────────────────────────────


class SomaticModulationPayload(_Lenient):
    precision_weights: dict[str, float] = Field(default_factory=dict)
    coherence: float = 1.0
    coherence_signal: float = 1.0


class EconomicStatePayload(_Lenient):
    runway_days: float = 0.0


class FederationKnowledgeReceivedPayload(_Lenient):
    remote_instance_id: str = ""
    knowledge_type: str = ""


class SandboxResultPayload(_Lenient):
    correlation_id: str
    approved: bool = False
    reason: str = ""


class OneirosConsolidationCompletePayload(_Lenient):
    cycle_id: str = ""
    episodes_consolidated: int = 0
    schemas_updated: int = 0
    duration_s: float = 0.0


# ─── Registry ───────────────────────────────────────────────────────

_PAYLOAD_MODELS: dict[SynapseEventType, type[BaseModel]] = {
    # Generic system events
    SynapseEventType.SYSTEM_FAILED: SystemFailedPayload,
    SynapseEventType.SYSTEM_RECOVERED: SystemRecoveredPayload,
    SynapseEventType.SYSTEM_RESTARTING: SystemRestartingPayload,
    SynapseEventType.SYSTEM_OVERLOADED: SystemOverloadedPayload,
    SynapseEventType.CLOCK_OVERRUN: ClockOverrunPayload,
    SynapseEventType.RESOURCE_PRESSURE: ResourcePressurePayload,
    SynapseEventType.TRANSACTION_SHIELDED: TransactionShieldedPayload,
    SynapseEventType.PROTOCOL_ALERT: ProtocolAlertPayload,
    SynapseEventType.CERTIFICATE_EXPIRING: CertificateExpiringPayload,
    SynapseEventType.CERTIFICATE_EXPIRED: CertificateExpiringPayload,
    SynapseEventType.SYSTEM_DEGRADED: SystemFailedPayload,
    SynapseEventType.CONNECTOR_ERROR: ConnectorErrorPayload,
    SynapseEventType.CONNECTOR_TOKEN_EXPIRED: ConnectorErrorPayload,
    SynapseEventType.MODEL_HOT_SWAP_FAILED: ModelHotSwapFailedPayload,
    SynapseEventType.CATASTROPHIC_FORGETTING_DETECTED: CatastrophicForgettingPayload,
    SynapseEventType.MODEL_ROLLBACK_TRIGGERED: ModelHotSwapFailedPayload,
    # Dedicated handlers
    SynapseEventType.INTENT_REJECTED: IntentRejectedPayload,
    SynapseEventType.INTEROCEPTIVE_PERCEPT: InteroceptivePerceptPayload,
    SynapseEventType.GROWTH_STAGNATION: GrowthStagnationPayload,
    SynapseEventType.AUTONOMY_INSUFFICIENT: AutonomyInsufficientPayload,
    SynapseEventType.METABOLIC_PRESSURE: MetabolicPressurePayload,
    SynapseEventType.TASK_PERMANENTLY_FAILED: TaskPermanentlyFailedPayload,
    SynapseEventType.BOUNTY_PR_SUBMITTED: BountyPRSubmittedPayload,
    SynapseEventType.SIMULA_CALIBRATION_DEGRADED: SimulaCalibrationDegradedPayload,
    # Sentinels
    SynapseEventType.REPAIR_COMPLETED: RepairCompletedPayload,
    SynapseEventType.EVO_CONSOLIDATION_STALLED: EvoConsolidationStalledPayload,
    SynapseEventType.NOVA_DEGRADED: NovaDegradedPayload,
    SynapseEventType.FOVEA_INTERNAL_PREDICTION_ERROR: FoveaPredictionErrorPayload,
    SynapseEventType.SOMA_STATE_SPIKE: SomaStateSpikePayload,
    SynapseEventType.SKIA_HEARTBEAT_LOST: SkiaHeartbeatLostPayload,
    SynapseEventType.THREAT_DETECTED: ThreatDetectedPayload,
    SynapseEventType.SPECIATION_EVENT: SpeciationEventPayload,
    # Closure / feedback
    SynapseEventType.CONSTITUTIONAL_DRIFT_DETECTED: ConstitutionalDriftPayload,
    SynapseEventType.AXON_SHIELD_REJECTION: AxonShieldRejectionPayload,
    SynapseEventType.ATUNE_REPAIR_VALIDATION: AtuneRepairValidationPayload,
    SynapseEventType.EVO_HYPOTHESIS_QUALITY: EvoHypothesisQualityPayload,
    SynapseEventType.NOVA_BELIEF_STABILISED: NovaBeliefStabilisedPayload,
    # Cross-system caches
    SynapseEventType.SOMATIC_MODULATION_SIGNAL: SomaticModulationPayload,
    SynapseEventType.ECONOMIC_STATE_UPDATED: EconomicStatePayload,
    SynapseEventType.METABOLIC_SNAPSHOT: EconomicStatePayload,
    SynapseEventType.FEDERATION_KNOWLEDGE_RECEIVED: FederationKnowledgeReceivedPayload,
    SynapseEventType.SIMULA_SANDBOX_RESULT: SandboxResultPayload,
    # Oneiros consolidation (SG8)
    SynapseEventType.ONEIROS_CONSOLIDATION_COMPLETE: OneirosConsolidationCompletePayload,
}


def validate_event_payload(
    event: SynapseEvent,
    log: structlog.stdlib.BoundLogger | None = None,
) -> dict[str, Any]:
    """
    Validate event.data against the registered Pydantic model.

    Non-blocking: on validation failure, logs a warning and returns
    the raw event.data unchanged so the handler can still proceed.
    """
    model_cls = _PAYLOAD_MODELS.get(event.event_type)
    if model_cls is None:
        return event.data

    try:
        validated = model_cls.model_validate(event.data)
        return validated.model_dump()
    except Exception as exc:  # noqa: BLE001
        _log = log or logger
        _log.warning(
            "event_payload_validation_failed",
            event_type=event.event_type.value,
            error=str(exc)[:200],
            source_system=event.source_system,
        )
        return event.data
