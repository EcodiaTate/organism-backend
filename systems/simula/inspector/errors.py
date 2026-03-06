"""
EcodiaOS -- Inspector Error Hierarchy

All exceptions raised within the Inspector vulnerability-discovery pipeline.

Namespace: systems.simula.inspector.errors
Distinct from: systems.simula.errors  (self-evolution errors)

Inspector errors are NEVER routed to Thymos or the evolution pipeline.
Failures are recorded via InspectorAnalyticsEmitter (hunt_error event)
and structlog. The hunt continues or aborts cleanly; the EOS cognitive
cycle is never disrupted.
"""

from __future__ import annotations


class InspectorError(RuntimeError):
    """Base for all Inspector vulnerability-discovery pipeline errors."""


class HuntError(InspectorError):
    """Fatal error aborting an entire hunt. Recorded as hunt_error event."""


class SurfaceMappingError(InspectorError):
    """Attack-surface discovery failed for one or more targets."""


class ProofTimeoutError(InspectorError):
    """Z3 or Dafny proof exceeded the configured time budget."""


class PoCGenerationError(InspectorError):
    """Z3 SAT counterexample could not be translated into a PoC script."""


class SandboxError(InspectorError):
    """Detonation chamber rejected or crashed during PoC execution."""


class RemediationError(InspectorError):
    """Autonomous patch generation or post-patch re-verification failed."""


class IngestError(InspectorError):
    """Repository clone or workspace initialisation failed."""
