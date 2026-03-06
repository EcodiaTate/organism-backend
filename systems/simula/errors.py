"""
EcodiaOS -- Simula Evolution Error Hierarchy

All exceptions raised within the Simula self-evolution pipeline.

Namespace: systems.simula.errors
Distinct from: systems.simula.inspector.errors  (hunt/proof errors)

These two namespaces must never be mixed:
  SimulaError subclasses  -> proposal lifecycle failures -> Thymos Incidents
  InspectorError subclasses -> hunt failures -> InspectorAnalyticsEmitter

Severity guide:
  ApplicationError  HIGH     -- change could not be applied; rollback triggered
  RollbackError     CRITICAL -- rollback itself failed; filesystem may be inconsistent
"""

from __future__ import annotations


class SimulaError(RuntimeError):
    """Base for all Simula self-evolution pipeline errors."""


class ApplicationError(SimulaError):
    """
    An approved evolution proposal could not be applied.

    Severity: HIGH
    Recovery: RollbackManager restores the pre-change snapshot,
    SimulaService records the proposal as ROLLED_BACK.
    Thymos: IncidentClass.DEGRADATION, repair_tier KNOWN_FIX.
    """


class RollbackError(SimulaError):
    """
    Restoring files to their pre-change state failed.

    Severity: CRITICAL
    Recovery: manual intervention required; filesystem state unknown.
    Thymos: IncidentClass.CRASH, repair_tier ESCALATE.
    """
