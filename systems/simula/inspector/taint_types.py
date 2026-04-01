"""
EcodiaOS - Inspector Taint-Aware Types

Data models for cross-service taint tracking. These types bridge the gap
between kernel-level eBPF observations and the Z3 formal verification engine.

The pipeline flow:
  eBPF ring buffer → taint_collector daemon → TaintCollectorClient →
  TaintGraph → VulnerabilityProver (cross-service Z3 encoding)

Phase 1 Observability additions (Cross-Layer Observability Substrate):
  CorrelationContext - UUID threading through all observability phases
  KernelEventType / KernelEvent - typed kernel/syscall events from eBPF
  InteractionGraph - per-proposal process-to-service interaction topology

All models extend EOSBaseModel for consistency with the rest of EOS.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ── Enums ────────────────────────────────────────────────────────────────────


class TaintLevel(enum.StrEnum):
    """Classification of how data became tainted."""

    UNTAINTED = "untainted"
    USER_INPUT = "user_input"
    PROPAGATED = "propagated"
    SANITIZED = "sanitized"


class FlowType(enum.StrEnum):
    """Transport mechanism for a cross-service data flow."""

    NETWORK = "network"
    FILE = "file"
    IPC = "ipc"
    DATABASE = "database"


class SinkType(enum.StrEnum):
    """Classification of where tainted data terminates."""

    SQL_QUERY = "sql_query"
    COMMAND_EXEC = "command_exec"
    FILE_WRITE = "file_write"
    HTTP_RESPONSE = "http_response"
    LOG_OUTPUT = "log_output"
    EXTERNAL_REQUEST = "external_request"


# ── Taint Source / Sink ──────────────────────────────────────────────────────


class TaintSource(EOSBaseModel):
    """An entry point where untrusted data enters the system."""

    id: str = Field(default_factory=new_id)
    variable_name: str = Field(
        ...,
        description="Name of the variable carrying tainted data",
    )
    source_service: str = Field(
        ...,
        description="Compose service where the taint originates",
    )
    entry_point: str = Field(
        ...,
        description="Route or function where taint enters (e.g. 'POST /api/users')",
    )
    taint_level: TaintLevel = TaintLevel.USER_INPUT
    http_method: str | None = None
    route_pattern: str | None = None
    observed_at: datetime = Field(default_factory=utc_now)


class TaintSink(EOSBaseModel):
    """A termination point where tainted data is consumed dangerously."""

    id: str = Field(default_factory=new_id)
    variable_name: str = Field(
        ...,
        description="Name of the variable at the sink",
    )
    sink_service: str = Field(
        ...,
        description="Compose service containing the sink",
    )
    sink_type: SinkType
    function_name: str = Field(
        ...,
        description="Function containing the sink call",
    )
    file_path: str = ""
    line_number: int | None = None
    is_sanitized: bool = Field(
        default=False,
        description="Whether taint was sanitized before reaching this sink",
    )
    sanitizer_name: str | None = Field(
        default=None,
        description="Name of the sanitizer applied (e.g. 'parameterized_query', 'html_escape')",
    )
    observed_at: datetime = Field(default_factory=utc_now)


# ── Taint Flow ──────────────────────────────────────────────────────────────


class TaintFlow(EOSBaseModel):
    """A correlated data flow between two services, observed by eBPF."""

    id: str = Field(default_factory=new_id)
    from_service: str
    to_service: str
    flow_type: FlowType = FlowType.NETWORK
    payload_signature: str = Field(
        default="",
        description="Hex of FNV-1a payload hash from eBPF observation",
    )
    payload_size: int = 0
    event_count: int = 1
    first_observed: datetime = Field(default_factory=utc_now)
    last_observed: datetime = Field(default_factory=utc_now)


# ── Taint Graph ─────────────────────────────────────────────────────────────


class TaintGraphNode(EOSBaseModel):
    """A service node in the taint propagation graph."""

    service_name: str
    service_type: str = Field(
        default="",
        description="Inferred service role: 'web', 'api', 'database', 'cache', etc.",
    )
    pid: int | None = None
    container_id: str | None = None


class TaintGraph(EOSBaseModel):
    """
    Full taint propagation graph across a docker-compose topology.

    Built by the taint collector sidecar from eBPF observations, then
    consumed by the VulnerabilityProver to generate cross-service Z3
    constraints.
    """

    id: str = Field(default_factory=new_id)
    nodes: list[TaintGraphNode] = Field(default_factory=list)
    edges: list[TaintFlow] = Field(default_factory=list)
    sources: list[TaintSource] = Field(default_factory=list)
    sinks: list[TaintSink] = Field(default_factory=list)
    collected_at: datetime = Field(default_factory=utc_now)

    @property
    def service_names(self) -> list[str]:
        """All service names in the graph."""
        return [n.service_name for n in self.nodes]

    @property
    def has_unsanitized_path(self) -> bool:
        """True if any sink has not been sanitized."""
        return any(not s.is_sanitized for s in self.sinks)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def flows_from(self, service: str) -> list[TaintFlow]:
        """All flows originating from the given service."""
        return [e for e in self.edges if e.from_service == service]

    def flows_to(self, service: str) -> list[TaintFlow]:
        """All flows terminating at the given service."""
        return [e for e in self.edges if e.to_service == service]


# ── Cross-Service Attack Surface ────────────────────────────────────────────


class CrossServiceAttackSurface(EOSBaseModel):
    """
    Links a traditional single-service AttackSurface with eBPF-observed
    taint data flowing across service boundaries.

    Used by the taint-augmented VulnerabilityProver to generate cross-service
    Z3 constraints and multi-step PoCs.
    """

    id: str = Field(default_factory=new_id)
    primary_surface: Any = Field(
        ...,
        description="The original single-service AttackSurface (from types.py)",
    )
    service_name: str = Field(
        ...,
        description="Compose service hosting the primary surface",
    )
    taint_sources: list[TaintSource] = Field(default_factory=list)
    taint_sinks: list[TaintSink] = Field(default_factory=list)
    cross_service_flows: list[TaintFlow] = Field(default_factory=list)
    cross_service_context_code: str = Field(
        default="",
        description="Multi-service stitched source with taint annotations",
    )
    involved_services: list[str] = Field(
        default_factory=list,
        description="All services in the taint chain for this surface",
    )
    discovered_at: datetime = Field(default_factory=utc_now)


# ── Collector Status ────────────────────────────────────────────────────────


class TaintCollectorStatus(EOSBaseModel):
    """Status response from the taint collector sidecar."""

    status: str = Field(
        default="unknown",
        description="'ready', 'degraded', 'loading', or 'error'",
    )
    programs_loaded: int = 0
    events_collected: int = 0
    flows_correlated: int = 0
    buffer_drops: int = 0
    uptime_seconds: float = 0.0
    degraded_reason: str | None = None


# ── Phase 1: Cross-Layer Observability Substrate ─────────────────────────────


class CorrelationContext(EOSBaseModel):
    """
    UUID threading through all observability phases (1–8).

    Set at the proposal entry point and attached to every emitted event so
    Phase 8 can assemble end-to-end story graphs from heterogeneous sources.
    """

    correlation_id: str = Field(default_factory=new_id)
    phase: int = Field(
        default=1,
        description="Which observability phase generated this event (1-8)",
    )
    timestamp: datetime = Field(default_factory=utc_now)
    tenant_id: str = ""
    proposal_id: str = Field(
        default="",
        description="Which Simula proposal spawned this observation run",
    )


class KernelEventType(enum.StrEnum):
    """Kernel-level event types observable via eBPF / procfs."""

    PROCESS_FORK = "process_fork"
    PROCESS_EXIT = "process_exit"
    THREAD_CREATE = "thread_create"
    THREAD_EXIT = "thread_exit"
    SYSCALL_ENTER = "syscall_enter"
    SYSCALL_EXIT = "syscall_exit"
    FILE_OPEN = "file_open"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_CLOSE = "file_close"
    SOCKET_CONNECT = "socket_connect"
    SOCKET_SEND = "socket_send"
    SOCKET_RECV = "socket_recv"
    SOCKET_CLOSE = "socket_close"


class KernelEvent(EOSBaseModel):
    """
    Low-level kernel event from the eBPF observer or procfs poller.

    Extends the taint pipeline's ObserverEvent with structured fields
    and a CorrelationContext for Phase 8 assembly.
    """

    id: str = Field(default_factory=new_id)
    event_type: KernelEventType
    correlation_context: CorrelationContext

    # Process / thread
    pid: int
    ppid: int = 0
    comm: str = ""

    # Syscall details - populated for SYSCALL_ENTER / SYSCALL_EXIT
    syscall_name: str | None = None
    syscall_args: dict[str, Any] | None = None
    syscall_retval: int | None = None

    # File details - populated for FILE_* events
    fd: int | None = None
    path: str | None = None

    # Network details - populated for SOCKET_* events
    remote_ip: str | None = None
    remote_port: int | None = None
    local_port: int | None = None

    # CPU / scheduling
    cpu_id: int | None = None
    duration_ns: int | None = None

    observed_at: datetime = Field(default_factory=utc_now)


class InteractionEdge(EOSBaseModel):
    """A single directed interaction between two nodes in the topology."""

    source: str  # e.g. "process_main", "service_auth"
    target: str  # e.g. "service_db"
    interaction_type: str  # "http_call", "db_query", "file_io", "fork"
    count: int = 1  # Observed occurrence count in this proposal run


class InteractionGraph(EOSBaseModel):
    """
    Process-to-service interaction topology for a single proposal run.

    Nodes are service names or process labels; edges carry the interaction
    type and multiplicity.  Stored in Neo4j via EvolutionHistoryManager.
    """

    id: str = Field(default_factory=new_id)
    correlation_context: CorrelationContext
    nodes: list[str] = Field(default_factory=list)
    edges: list[InteractionEdge] = Field(default_factory=list)
    interaction_types: set[str] = Field(default_factory=set)
    captured_at: datetime = Field(default_factory=utc_now)

    def add_edge(
        self,
        source: str,
        target: str,
        interaction_type: str,
    ) -> None:
        """Record an interaction, merging duplicates by incrementing count."""
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)
        self.interaction_types.add(interaction_type)

        for edge in self.edges:
            if (
                edge.source == source
                and edge.target == target
                and edge.interaction_type == interaction_type
            ):
                edge.count += 1
                return

        self.edges.append(
            InteractionEdge(
                source=source,
                target=target,
                interaction_type=interaction_type,
            )
        )
