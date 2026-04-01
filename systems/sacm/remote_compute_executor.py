"""
EcodiaOS - SACM Remote Compute Executor (re-export shim)

The executor has moved to systems.axon.executors.remote_compute.
This module re-exports RemoteComputeExecutor for backward compatibility
with any callers that reference the old path.
"""

from systems.axon.executors.remote_compute import RemoteComputeExecutor as RemoteComputeExecutor  # noqa: F401

__all__ = ["RemoteComputeExecutor"]
