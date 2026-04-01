"""
EcodiaOS - Incident Primitives

IncidentClass and IncidentSeverity are organism-wide classification enums
that any system (including Synapse's ErrorSentinel) needs without importing
the full Thymos type graph.

The full Incident model (with RepairTier, RepairAttempt, AntibodyLibrary refs)
lives in systems.thymos.types and imports from here.

Spec 09 §AV9 - moved here so sentinel.py has no cross-system dependency.
"""

from __future__ import annotations

import enum


class IncidentSeverity(enum.StrEnum):
    """How bad is the failure?"""

    CRITICAL = "critical"  # System down, user impact, drives affected
    HIGH = "high"          # System degraded, partial user impact
    MEDIUM = "medium"      # Anomaly detected, no immediate user impact
    LOW = "low"            # Cosmetic, informational, or transient
    INFO = "info"          # Normal variance logged for pattern detection


class IncidentClass(enum.StrEnum):
    """What kind of failure is this?"""

    CRASH = "crash"                           # Unhandled exception, system death
    DEGRADATION = "degradation"               # Slow or incorrect responses
    CONTRACT_VIOLATION = "contract_violation" # Inter-system SLA breach
    LOOP_SEVERANCE = "loop_severance"         # Feedback loop not transmitting
    DRIFT = "drift"                           # Gradual metric deviation from baseline
    PREDICTION_FAILURE = "prediction_failure" # Active inference errors elevated
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Budget exceeded
    COGNITIVE_STALL = "cognitive_stall"       # Workspace cycle blocked or empty
    ECONOMIC_THREAT = "economic_threat"       # Malicious on-chain activity detected
    PROTOCOL_DEGRADATION = "protocol_degradation"  # DeFi protocol health declining
    DRIVE_EXTINCTION = "drive_extinction"     # INV-017: constitutional drive mean < 0.01
    SECURITY = "security"                     # Cryptographic or identity security event (vault, cert, tamper)
