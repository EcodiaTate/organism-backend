"""
EcodiaOS — Inspector Taint Flow Linker

User-space observer component that links inbound taint observations to
outbound taint observations by application-level token identity.

While TaintCorrelator operates on raw payload hashes (FNV-1a of bytes),
TaintFlowLinker operates on structured token strings (e.g. ``TAINT=<uuid>;``)
that are visible in payload content — bridging the gap between kernel-level
byte observations and application-level request chains.

Architecture
------------

TaintRegistry
    key: (pid, token) or (service_id, token)
    value: TaintEntryPoint — when/where the token was first seen, socket tuple

TaintFlowLinker
    Ingests observations (inbound and outbound) carrying tokens.
    When an outbound observation contains a token already in the registry,
    creates a directed edge: source_event → outbound_event with full context.

TaintChainGraph
    Holds the directed graph of token-linked edges.
    Emits JSON edge lists and writes to SQLite (or JSONL) for replay.

Output example::

    [
      {"from": "A", "to": "B", "token": "TAINT=abc123;",
       "src_ts": 1700000000.1, "dst_ts": 1700000000.2,
       "src_tuple": ("10.0.0.1", 52000, "10.0.0.2", 8080),
       "dst_tuple": ("10.0.0.2", 8080, "10.0.0.3", 9090)},
      ...
    ]

Usage::

    linker = TaintFlowLinker()

    # Service A sends a request containing the token
    linker.observe_inbound("service_a", pid=100, token="TAINT=abc;",
                           src_tuple=("external", 0, "10.0.0.1", 80),
                           timestamp=time.time())

    # Service B receives from A and forwards to C — outbound carries same token
    linker.observe_outbound("service_b", pid=200, token="TAINT=abc;",
                            dst_tuple=("10.0.0.2", 8080, "10.0.0.3", 9090),
                            timestamp=time.time())

    chain = linker.chain_for_token("TAINT=abc;")
    # → [A → B, B → C]

    linker.emit_json()          # stdout JSON edges
    linker.write_sqlite("taint_flows.db")  # persist for replay
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Final

import structlog

logger: Final = structlog.get_logger().bind(system="simula.inspector.taint_flow_linker")

# Socket tuple: (src_ip, src_port, dst_ip, dst_port)
SocketTuple = tuple[str, int, str, int]

_UNKNOWN_TUPLE: SocketTuple = ("", 0, "", 0)

# ── Sensitive Sink Ports ──────────────────────────────────────────────────────

#: Destination ports considered sensitive data sinks.
#: Connections to these ports trigger a CRITICAL_DATA_FLOW_BREACH alert.
SENSITIVE_SINK_PORTS: frozenset[int] = frozenset({
    5432,   # PostgreSQL
    3306,   # MySQL / MariaDB
    1433,   # MSSQL
    1521,   # Oracle
    6379,   # Redis
    6380,   # Redis TLS
    9200,   # Elasticsearch HTTP
    9300,   # Elasticsearch transport
    27017,  # MongoDB
    27018,  # MongoDB shard
    5984,   # CouchDB
    8087,   # Riak
    9042,   # Cassandra CQL
    2181,   # ZooKeeper
    5672,   # RabbitMQ AMQP
    4222,   # NATS
})

# ── Sink Alert ────────────────────────────────────────────────────────────────


@dataclass
class SinkAlert:
    """
    Emitted when tainted data is observed flowing into a sensitive sink port.

    Logged as a CRITICAL_DATA_FLOW_BREACH event by TaintFlowLinker.
    """

    token: str
    sink_port: int
    sink_ip: str
    ingress_src_tuple: SocketTuple    # original entry point (where taint arrived)
    egress_dst_tuple: SocketTuple     # socket headed to the sink
    service_id: str                   # service making the outbound call
    pid: int
    timestamp: float
    alert_id: str = ""

    def __post_init__(self) -> None:
        if not self.alert_id:
            raw = f"{self.token}|{self.service_id}|{self.sink_ip}:{self.sink_port}|{self.timestamp:.3f}"
            self.alert_id = f"sa-{hash(raw) & 0xFFFFFFFF:08x}"

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "event": "CRITICAL_DATA_FLOW_BREACH",
            "token": self.token,
            "sink_ip": self.sink_ip,
            "sink_port": self.sink_port,
            "service_id": self.service_id,
            "pid": self.pid,
            "ingress_src_tuple": list(self.ingress_src_tuple),
            "egress_dst_tuple": list(self.egress_dst_tuple),
            "timestamp": self.timestamp,
        }


# ── Per-Connection Rolling Stream Buffer ──────────────────────────────────────

_STREAM_BUFFER_MAX_BYTES: int = 4096   # truncate per-connection buffer at 4 KB
_STREAM_BUFFER_MAX_CONNS: int = 8192   # LRU capacity (max live connections tracked)
_STREAM_BUFFER_TTL_S: float = 120.0   # evict idle connections after 2 minutes

_TOKEN_RE = re.compile(rb"TAINT=([^;]+);")


class StreamBuffer:
    """
    Per-connection TCP reassembly buffer for split-packet token scanning.

    Design goals
    ------------
    * Correctness — token strings that arrive split across two or more TCP
      segments are found by concatenating payloads before scanning.
    * Bounded memory — each connection buffer is capped at
      ``_STREAM_BUFFER_MAX_BYTES`` (4 KB); the oldest bytes are discarded
      when the cap is exceeded so scanning can still find tokens in the tail.
    * Bounded connections — an LRU dict with ``_STREAM_BUFFER_MAX_CONNS``
      capacity evicts the least-recently-used connection entry automatically.
    * Automatic eviction — connections idle longer than
      ``_STREAM_BUFFER_TTL_S`` are reaped on every ``feed()`` call.

    Thread-safety
    -------------
    A single ``threading.Lock`` protects the internal state; callers must not
    hold their own locks around ``feed()`` / ``close()`` calls.

    Usage::

        buf = StreamBuffer()
        tokens = buf.feed(conn_key, payload_bytes)
        # tokens: list[str] — all "TAINT=<uuid>;" strings found in the
        #         reassembled stream since the last scan, de-duplicated
        #         across segment boundaries.

        buf.close(conn_key)   # on TCP FIN / RST
    """

    def __init__(
        self,
        max_bytes: int = _STREAM_BUFFER_MAX_BYTES,
        max_conns: int = _STREAM_BUFFER_MAX_CONNS,
        ttl_s: float = _STREAM_BUFFER_TTL_S,
    ) -> None:
        self._max_bytes = max_bytes
        self._max_conns = max_conns
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        # OrderedDict as LRU: key = SocketTuple, value = (bytearray, last_seen_ts)
        self._buffers: OrderedDict[SocketTuple, tuple[bytearray, float]] = OrderedDict()
        # TTL reaping: track last reap time to avoid per-call overhead
        self._last_reap: float = time.monotonic()

    # ── Public API ────────────────────────────────────────────────────────

    def feed(self, conn_key: SocketTuple, payload: bytes) -> list[str]:
        """
        Append *payload* to the buffer for *conn_key* and return any
        ``TAINT=…;`` tokens found in the combined stream.

        The returned token strings include the ``TAINT=`` prefix and ``;``
        suffix, matching the output of ``extract_tokens_from_payload()``.
        """
        if not payload:
            return []

        now = time.monotonic()
        with self._lock:
            self._maybe_reap(now)
            buf, _ = self._get_or_create(conn_key, now)

            # Append new bytes, then cap at max_bytes (keep the tail)
            buf.extend(payload)
            if len(buf) > self._max_bytes:
                del buf[: len(buf) - self._max_bytes]

            # Scan concatenated buffer for tokens
            found = [
                m.group(0).decode("utf-8", errors="replace")
                for m in _TOKEN_RE.finditer(buf)
            ]

            # Move to end (most-recently-used)
            self._buffers.move_to_end(conn_key)
            self._buffers[conn_key] = (buf, now)

        return found

    def close(self, conn_key: SocketTuple) -> None:
        """Discard the buffer for *conn_key* (call on TCP FIN or RST)."""
        with self._lock:
            self._buffers.pop(conn_key, None)

    @property
    def active_connections(self) -> int:
        with self._lock:
            return len(self._buffers)

    # ── Internals ─────────────────────────────────────────────────────────

    def _get_or_create(
        self, key: SocketTuple, now: float
    ) -> tuple[bytearray, float]:
        """Return (buf, last_seen); create a new entry if absent. Evicts LRU if full."""
        if key in self._buffers:
            return self._buffers[key]

        # Evict LRU when at capacity
        if len(self._buffers) >= self._max_conns:
            evicted_key, _ = self._buffers.popitem(last=False)  # FIFO = LRU
            logger.debug("stream_buffer_lru_evict", conn=evicted_key)

        entry: tuple[bytearray, float] = (bytearray(), now)
        self._buffers[key] = entry
        return entry

    def _maybe_reap(self, now: float) -> None:
        """Evict connections idle longer than _ttl_s (called under lock)."""
        if now - self._last_reap < 10.0:
            return  # Reap at most every 10 seconds
        self._last_reap = now
        cutoff = now - self._ttl_s
        stale = [k for k, (_, ts) in self._buffers.items() if ts < cutoff]
        for k in stale:
            del self._buffers[k]
        if stale:
            logger.debug("stream_buffer_ttl_evict", count=len(stale))


# ── Entry Point ──────────────────────────────────────────────────────────────


@dataclass
class TaintEntryPoint:
    """
    Metadata captured the first time a token is seen entering a service
    (inbound observation).
    """

    token: str
    service_id: str
    pid: int
    first_seen_ts: float              # unix timestamp (seconds)
    entry_tuple: SocketTuple          # (src_ip, src_port, dst_ip, dst_port)
    entry_type: str = "inbound"       # "inbound" | "injected"
    extra: dict = field(default_factory=dict)


# ── Graph Edge ───────────────────────────────────────────────────────────────


@dataclass
class TaintEdge:
    """
    A directed edge in the taint propagation graph.

    Represents: ``src_service`` sent token ``token`` → ``dst_service``
    (observed when an outbound payload from src_service carrying the token
    arrives at dst_service).
    """

    token: str
    src_service: str
    dst_service: str
    src_pid: int
    dst_pid: int
    src_ts: float                     # when outbound payload was observed leaving src
    dst_ts: float | None              # when inbound arrival was observed at dst (may lag)
    src_tuple: SocketTuple
    dst_tuple: SocketTuple
    hop_index: int = 0                # 0 = A→B, 1 = B→C, ...
    edge_id: str = ""

    def __post_init__(self) -> None:
        if not self.edge_id:
            # Deterministic ID so duplicate edges can be detected
            raw = f"{self.token}|{self.src_service}|{self.dst_service}|{self.src_ts:.3f}"
            self.edge_id = f"{hash(raw) & 0xFFFFFFFF:08x}"

    def to_json_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "token": self.token,
            "from": self.src_service,
            "to": self.dst_service,
            "src_pid": self.src_pid,
            "dst_pid": self.dst_pid,
            "src_ts": self.src_ts,
            "dst_ts": self.dst_ts,
            "src_tuple": list(self.src_tuple),
            "dst_tuple": list(self.dst_tuple),
            "hop_index": self.hop_index,
        }


# ── Registry ─────────────────────────────────────────────────────────────────


class TaintRegistry:
    """
    Taint token registry.

    key:   (service_id, token) or (pid, token) — both are supported
    value: TaintEntryPoint — first-seen metadata (when, where, how it entered)

    Thread-safe.  Entries are never evicted (the graph must remain complete
    for replay); callers can call ``clear()`` between test runs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Primary index: (service_id, token) → TaintEntryPoint
        self._by_service: dict[tuple[str, str], TaintEntryPoint] = {}
        # Secondary index: (pid, token) → TaintEntryPoint  (same objects)
        self._by_pid: dict[tuple[int, str], TaintEntryPoint] = {}

    # ── Write ──────────────────────────────────────────────────────────────

    def register(
        self,
        token: str,
        *,
        service_id: str,
        pid: int,
        timestamp: float | None = None,
        entry_tuple: SocketTuple = _UNKNOWN_TUPLE,
        entry_type: str = "inbound",
        extra: dict | None = None,
    ) -> TaintEntryPoint:
        """
        Record first-seen metadata for *token*.

        If the (service_id, token) pair was already registered, returns the
        existing entry without overwriting (first-writer wins — this preserves
        the true entry point if the same token passes through multiple services).
        """
        svc_key = (service_id, token)
        pid_key = (pid, token)

        with self._lock:
            if svc_key in self._by_service:
                return self._by_service[svc_key]

            entry = TaintEntryPoint(
                token=token,
                service_id=service_id,
                pid=pid,
                first_seen_ts=timestamp if timestamp is not None else time.time(),
                entry_tuple=entry_tuple,
                entry_type=entry_type,
                extra=extra or {},
            )
            self._by_service[svc_key] = entry
            self._by_pid[pid_key] = entry
            logger.debug(
                "taint_registered",
                token=token,
                service_id=service_id,
                pid=pid,
            )
            return entry

    # ── Read ───────────────────────────────────────────────────────────────

    def lookup_by_service(self, service_id: str, token: str) -> TaintEntryPoint | None:
        with self._lock:
            return self._by_service.get((service_id, token))

    def lookup_by_pid(self, pid: int, token: str) -> TaintEntryPoint | None:
        with self._lock:
            return self._by_pid.get((pid, token))

    def all_entries_for_token(self, token: str) -> list[TaintEntryPoint]:
        """Return all entry points that saw this token (across all services)."""
        with self._lock:
            return [ep for (_, t), ep in self._by_service.items() if t == token]

    def known_tokens(self) -> set[str]:
        with self._lock:
            return {t for (_, t) in self._by_service}

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_service)

    def clear(self) -> None:
        with self._lock:
            self._by_service.clear()
            self._by_pid.clear()


# ── Chain Graph ───────────────────────────────────────────────────────────────


class TaintChainGraph:
    """
    Directed graph of token-linked taint edges.

    Edges are stored in insertion order per token so ``chain_for_token``
    returns them in hop order (A→B before B→C).

    Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # All edges, keyed by edge_id (dedup)
        self._edges: dict[str, TaintEdge] = {}
        # Per-token ordered list of edge_ids
        self._token_edges: dict[str, list[str]] = {}

    def add_edge(self, edge: TaintEdge) -> bool:
        """
        Insert edge.  Returns False if a duplicate edge_id already exists.
        """
        with self._lock:
            if edge.edge_id in self._edges:
                return False
            self._edges[edge.edge_id] = edge
            self._token_edges.setdefault(edge.token, []).append(edge.edge_id)
            return True

    def chain_for_token(self, token: str) -> list[TaintEdge]:
        """
        Return the ordered edge chain for *token*, sorted by src_ts.

        Example: token "TAINT=abc;" → [A→B, B→C]
        """
        with self._lock:
            ids = self._token_edges.get(token, [])
            edges = [self._edges[eid] for eid in ids if eid in self._edges]
        return sorted(edges, key=lambda e: e.src_ts)

    def all_edges(self) -> list[TaintEdge]:
        with self._lock:
            return list(self._edges.values())

    def known_tokens(self) -> set[str]:
        with self._lock:
            return set(self._token_edges)

    def to_json(self, *, token: str | None = None, indent: int | None = 2) -> str:
        """
        Emit graph as a JSON string.

        If *token* is given, emits only the chain for that token.
        Otherwise emits all edges grouped by token.
        """
        if token is not None:
            edges = self.chain_for_token(token)
            payload: dict | list = [e.to_json_dict() for e in edges]
        else:
            payload = {}
            for t in sorted(self.known_tokens()):
                payload[t] = [e.to_json_dict() for e in self.chain_for_token(t)]  # type: ignore[index]
        return json.dumps(payload, indent=indent)

    def print_chains(self, *, token: str | None = None) -> None:
        """
        Human-readable chain summary to stdout.

        Format::

            [TAINT=abc123;]
              hop 0: A → B  (ts=1700000000.100 → 1700000000.200)
              hop 1: B → C  (ts=1700000000.201 → 1700000000.310)
        """
        tokens = [token] if token else sorted(self.known_tokens())
        for tok in tokens:
            chain = self.chain_for_token(tok)
            if not chain:
                continue
            print(f"\n[{tok}]")
            for edge in chain:
                src_ts = f"{edge.src_ts:.3f}"
                dst_ts = f"{edge.dst_ts:.3f}" if edge.dst_ts else "?"
                print(
                    f"  hop {edge.hop_index}: {edge.src_service} → {edge.dst_service}"
                    f"  (ts={src_ts} → {dst_ts})"
                    f"  src_pid={edge.src_pid}"
                    f"  dst_tuple={edge.dst_tuple[2]}:{edge.dst_tuple[3]}"
                )

    # ── SQLite persistence ─────────────────────────────────────────────────

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS taint_edges (
            edge_id     TEXT PRIMARY KEY,
            token       TEXT NOT NULL,
            src_service TEXT NOT NULL,
            dst_service TEXT NOT NULL,
            src_pid     INTEGER,
            dst_pid     INTEGER,
            src_ts      REAL NOT NULL,
            dst_ts      REAL,
            src_ip      TEXT,
            src_port    INTEGER,
            dst_ip      TEXT,
            dst_port    INTEGER,
            hop_index   INTEGER DEFAULT 0,
            recorded_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_te_token ON taint_edges (token);
    """

    def write_sqlite(self, db_path: str) -> int:
        """
        Persist all edges to *db_path* (SQLite).  Upserts by edge_id.
        Returns number of rows written.
        """
        edges = self.all_edges()
        if not edges:
            return 0

        now = time.time()
        con = sqlite3.connect(db_path)
        try:
            con.executescript(self._CREATE_TABLE)
            rows = [
                (
                    e.edge_id, e.token, e.src_service, e.dst_service,
                    e.src_pid, e.dst_pid, e.src_ts, e.dst_ts,
                    e.src_tuple[0], e.src_tuple[1],
                    e.dst_tuple[2], e.dst_tuple[3],
                    e.hop_index, now,
                )
                for e in edges
            ]
            con.executemany(
                """
                INSERT OR REPLACE INTO taint_edges
                  (edge_id, token, src_service, dst_service,
                   src_pid, dst_pid, src_ts, dst_ts,
                   src_ip, src_port, dst_ip, dst_port,
                   hop_index, recorded_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            con.commit()
        finally:
            con.close()

        logger.info("taint_sqlite_written", db_path=db_path, rows=len(rows))
        return len(rows)

    def write_jsonl(self, log_path: str, *, append: bool = True) -> int:
        """
        Append all edges to *log_path* as newline-delimited JSON (JSONL).
        Returns number of lines written.
        """
        edges = self.all_edges()
        if not edges:
            return 0

        mode = "a" if append else "w"
        with open(log_path, mode, encoding="utf-8") as fh:
            for edge in edges:
                fh.write(json.dumps(edge.to_json_dict()) + "\n")

        logger.info("taint_jsonl_written", log_path=log_path, lines=len(edges))
        return len(edges)

    @classmethod
    def load_sqlite(cls, db_path: str) -> TaintChainGraph:
        """Reload a graph from a previously written SQLite file (for replay)."""
        graph = cls()
        try:
            con = sqlite3.connect(db_path)
            con.row_factory = sqlite3.Row
            try:
                rows = con.execute("SELECT * FROM taint_edges ORDER BY src_ts").fetchall()
            finally:
                con.close()
        except sqlite3.OperationalError:
            return graph  # empty DB or missing table

        for row in rows:
            edge = TaintEdge(
                token=row["token"],
                src_service=row["src_service"],
                dst_service=row["dst_service"],
                src_pid=row["src_pid"] or 0,
                dst_pid=row["dst_pid"] or 0,
                src_ts=row["src_ts"],
                dst_ts=row["dst_ts"],
                src_tuple=(row["src_ip"] or "", row["src_port"] or 0, "", 0),
                dst_tuple=("", 0, row["dst_ip"] or "", row["dst_port"] or 0),
                hop_index=row["hop_index"] or 0,
                edge_id=row["edge_id"],
            )
            graph.add_edge(edge)

        return graph


# ── Flow Linker ───────────────────────────────────────────────────────────────


class TaintFlowLinker:
    """
    Links inbound taint observations to outbound taint observations.

    Workflow
    --------
    1. Call ``observe_inbound`` when a token arrives at a service.  The
       payload bytes are fed through ``StreamBuffer`` for split-packet
       reassembly before token extraction.
    2. Call ``observe_outbound`` when an outbound payload leaving a service
       contains a known token.  The linker:
         a. Feeds the raw payload through ``StreamBuffer`` to find tokens
            across TCP segment boundaries.
         b. Looks up the registry to find the entry point for (service, token).
         c. Creates an edge: entry_service → current_service (delivering hop).
         d. Creates an edge: current_service → dst_service (forwarding hop).
         e. Checks whether ``dst_tuple[3]`` (destination port) is in
            ``sink_ports``; if so, emits a ``CRITICAL_DATA_FLOW_BREACH``
            ``SinkAlert`` via structlog.
    3. Call ``chain_for_token`` / ``emit_json`` / ``write_sqlite`` on the
       graph to inspect or persist results.

    The linker is intentionally stateless beyond its registry + graph — it
    does not buffer pending observations or speculate about future hops.

    Parameters
    ----------
    registry:
        Shared ``TaintRegistry`` instance.  One is created if not provided.
    graph:
        Shared ``TaintChainGraph`` instance.  One is created if not provided.
    stream_buffer:
        ``StreamBuffer`` used for per-connection TCP reassembly.  A default
        instance is created if not provided.
    sink_ports:
        Set of destination ports considered sensitive sinks.  Defaults to
        ``SENSITIVE_SINK_PORTS``.  Pass an empty set to disable sink alerts.
    """

    def __init__(
        self,
        registry: TaintRegistry | None = None,
        graph: TaintChainGraph | None = None,
        stream_buffer: StreamBuffer | None = None,
        sink_ports: frozenset[int] | None = None,
    ) -> None:
        self.registry = registry or TaintRegistry()
        self.graph = graph or TaintChainGraph()
        self.stream_buffer = stream_buffer or StreamBuffer()
        self.sink_ports: frozenset[int] = (
            sink_ports if sink_ports is not None else SENSITIVE_SINK_PORTS
        )
        # Deduplicate alerts: track (token, sink_ip, sink_port) tuples already fired
        self._fired_alerts: set[tuple[str, str, int]] = set()
        self._fired_lock = threading.Lock()

    # ── Ingest ─────────────────────────────────────────────────────────────

    def observe_inbound(
        self,
        service_id: str,
        *,
        token: str,
        pid: int = 0,
        timestamp: float | None = None,
        src_tuple: SocketTuple = _UNKNOWN_TUPLE,
        extra: dict | None = None,
        raw_payload: bytes | None = None,
    ) -> TaintEntryPoint:
        """
        Record that *token* arrived at *service_id* (inbound observation).

        If *raw_payload* is provided it is fed into the ``StreamBuffer`` for
        the connection identified by *src_tuple* so that tokens split across
        TCP segments are found.  The explicit *token* argument still takes
        effect regardless (e.g. when the caller already extracted the token
        via another mechanism).

        This registers the entry point in the registry.  If the token was
        already seen at this service, the existing entry is returned unchanged.
        """
        if raw_payload:
            reassembled = self.stream_buffer.feed(src_tuple, raw_payload)
            if reassembled:
                logger.debug(
                    "stream_buffer_inbound_tokens",
                    service_id=service_id,
                    conn=src_tuple,
                    tokens=reassembled,
                )

        return self.registry.register(
            token,
            service_id=service_id,
            pid=pid,
            timestamp=timestamp,
            entry_tuple=src_tuple,
            entry_type="inbound",
            extra=extra,
        )

    def observe_outbound(
        self,
        service_id: str,
        *,
        token: str,
        pid: int = 0,
        timestamp: float | None = None,
        dst_tuple: SocketTuple = _UNKNOWN_TUPLE,
        dst_service_id: str | None = None,
        extra: dict | None = None,
        raw_payload: bytes | None = None,
    ) -> list[TaintEdge]:
        """
        Record that *service_id* is emitting an outbound payload containing
        *token*, destined for *dst_tuple* (or the named *dst_service_id*).

        Behaviour
        ---------
        - If *raw_payload* is given, it is fed into the ``StreamBuffer`` for
          the connection identified by *dst_tuple* so that tokens split across
          TCP segments are found in reassembled data.
        - If the token is not in the registry, it is registered here as an
          "observed_outbound" entry (the inbound side may be missing if the
          chain entry was before observation started).
        - Locates the most recent inbound entry for *token* at *service_id*
          (or any service if none found at this one).
        - Creates one edge: inbound_service → service_id (the delivering hop)
          if the two services differ.
        - Creates one edge: service_id → dst_service  (the forwarding hop).
        - Both edges are inserted into the graph and returned.
        - **Sink detection**: if *dst_tuple*'s destination port is in
          ``self.sink_ports``, a ``SinkAlert`` is emitted via structlog at
          CRITICAL level as a ``CRITICAL_DATA_FLOW_BREACH`` event.

        Returns the list of new edges added (0–2 edges).
        """
        ts = timestamp if timestamp is not None else time.time()
        dst_svc = dst_service_id or _tuple_to_service_label(dst_tuple)

        # ── TCP reassembly via StreamBuffer ────────────────────────────────
        if raw_payload:
            reassembled_tokens = self.stream_buffer.feed(dst_tuple, raw_payload)
            if reassembled_tokens:
                logger.debug(
                    "stream_buffer_outbound_tokens",
                    service_id=service_id,
                    conn=dst_tuple,
                    tokens=reassembled_tokens,
                )

        # Ensure outbound service is in the registry as an observer
        outbound_entry = self.registry.lookup_by_service(service_id, token)
        if outbound_entry is None:
            # Token not yet seen inbound at this service — register it now
            # so downstream hops can link back to it
            outbound_entry = self.registry.register(
                token,
                service_id=service_id,
                pid=pid,
                timestamp=ts,
                entry_tuple=_UNKNOWN_TUPLE,
                entry_type="observed_outbound",
                extra=extra,
            )

        new_edges: list[TaintEdge] = []

        # ── Edge 1: delivering hop (inbound_service → service_id) ──────────
        # Find where the token originally arrived — the service that received
        # it before the current one forwarded it.
        all_entries = self.registry.all_entries_for_token(token)
        # Choose the entry at *this* service (the one currently forwarding)
        # or the earliest entry overall as the source of the delivering hop.
        inbound_at_this = outbound_entry
        earliest = min(all_entries, key=lambda e: e.first_seen_ts, default=None)
        source_entry = inbound_at_this if inbound_at_this.entry_type == "inbound" else earliest

        if source_entry and source_entry.service_id != service_id:
            # Determine hop index: count existing edges for this token
            existing_chain = self.graph.chain_for_token(token)
            hop = len(existing_chain)

            delivering_edge = TaintEdge(
                token=token,
                src_service=source_entry.service_id,
                dst_service=service_id,
                src_pid=source_entry.pid,
                dst_pid=pid,
                src_ts=source_entry.first_seen_ts,
                dst_ts=ts,
                src_tuple=source_entry.entry_tuple,
                dst_tuple=outbound_entry.entry_tuple,
                hop_index=hop,
            )
            if self.graph.add_edge(delivering_edge):
                new_edges.append(delivering_edge)
                logger.info(
                    "taint_edge_added",
                    token=token,
                    src=delivering_edge.src_service,
                    dst=delivering_edge.dst_service,
                    hop=delivering_edge.hop_index,
                )

        # ── Edge 2: forwarding hop (service_id → dst_service) ──────────────
        existing_chain = self.graph.chain_for_token(token)
        hop = len(existing_chain)

        forwarding_edge = TaintEdge(
            token=token,
            src_service=service_id,
            dst_service=dst_svc,
            src_pid=pid,
            dst_pid=0,                  # unknown until dst observe_inbound fires
            src_ts=ts,
            dst_ts=None,
            src_tuple=outbound_entry.entry_tuple,
            dst_tuple=dst_tuple,
            hop_index=hop,
        )
        if self.graph.add_edge(forwarding_edge):
            new_edges.append(forwarding_edge)
            logger.info(
                "taint_edge_added",
                token=token,
                src=forwarding_edge.src_service,
                dst=forwarding_edge.dst_service,
                hop=forwarding_edge.hop_index,
            )

        # ── Sink detection ──────────────────────────────────────────────────
        dst_port = dst_tuple[3]
        if dst_port and dst_port in self.sink_ports:
            self._maybe_emit_sink_alert(
                token=token,
                service_id=service_id,
                pid=pid,
                timestamp=ts,
                ingress_src_tuple=outbound_entry.entry_tuple,
                egress_dst_tuple=dst_tuple,
            )

        return new_edges

    def _maybe_emit_sink_alert(
        self,
        *,
        token: str,
        service_id: str,
        pid: int,
        timestamp: float,
        ingress_src_tuple: SocketTuple,
        egress_dst_tuple: SocketTuple,
    ) -> SinkAlert | None:
        """
        Emit a ``CRITICAL_DATA_FLOW_BREACH`` structlog event if this
        (token, sink_ip, sink_port) combination has not already been alerted.

        Returns the ``SinkAlert`` that was emitted, or ``None`` if it was
        a duplicate.
        """
        sink_ip = egress_dst_tuple[2]
        sink_port = egress_dst_tuple[3]
        dedup_key = (token, sink_ip, sink_port)

        with self._fired_lock:
            if dedup_key in self._fired_alerts:
                return None
            self._fired_alerts.add(dedup_key)

        alert = SinkAlert(
            token=token,
            sink_port=sink_port,
            sink_ip=sink_ip,
            ingress_src_tuple=ingress_src_tuple,
            egress_dst_tuple=egress_dst_tuple,
            service_id=service_id,
            pid=pid,
            timestamp=timestamp,
        )
        logger.critical(
            "CRITICAL_DATA_FLOW_BREACH",
            **alert.to_dict(),
        )
        return alert

    def observe_arrival(
        self,
        service_id: str,
        *,
        token: str,
        pid: int = 0,
        timestamp: float | None = None,
        src_tuple: SocketTuple = _UNKNOWN_TUPLE,
        raw_payload: bytes | None = None,
    ) -> None:
        """
        Back-fill the dst_ts and dst_pid on the most recent pending edge
        for *token* whose dst_service matches *service_id*.

        Call this when an inbound observation at *service_id* confirms that a
        previously emitted forwarding edge actually arrived.

        If *raw_payload* is given it is fed into the ``StreamBuffer`` for
        *src_tuple* so tokens split across TCP segments are found.  Also
        checks whether *src_tuple*'s destination port is a sensitive sink and
        emits a ``CRITICAL_DATA_FLOW_BREACH`` alert if so.
        """
        ts = timestamp if timestamp is not None else time.time()

        if raw_payload:
            reassembled_tokens = self.stream_buffer.feed(src_tuple, raw_payload)
            if reassembled_tokens:
                logger.debug(
                    "stream_buffer_arrival_tokens",
                    service_id=service_id,
                    conn=src_tuple,
                    tokens=reassembled_tokens,
                )

        # Sink check on arrival: the local port (dst of the tuple) may be a sink
        local_port = src_tuple[3]
        if local_port and local_port in self.sink_ports:
            entry = self.registry.lookup_by_service(service_id, token)
            ingress = entry.entry_tuple if entry else _UNKNOWN_TUPLE
            self._maybe_emit_sink_alert(
                token=token,
                service_id=service_id,
                pid=pid,
                timestamp=ts,
                ingress_src_tuple=ingress,
                egress_dst_tuple=src_tuple,
            )

        chain = self.graph.chain_for_token(token)
        for edge in reversed(chain):
            if edge.dst_service == service_id and edge.dst_ts is None:
                edge.dst_ts = ts
                edge.dst_pid = pid
                break

    def close_connection(self, conn_key: SocketTuple) -> None:
        """
        Discard the ``StreamBuffer`` for *conn_key*.

        Call this on TCP FIN / RST to free per-connection reassembly memory
        immediately instead of waiting for the TTL reaper.
        """
        self.stream_buffer.close(conn_key)

    # ── Convenience accessors ──────────────────────────────────────────────

    def chain_for_token(self, token: str) -> list[TaintEdge]:
        """Return ordered edge chain for *token*."""
        return self.graph.chain_for_token(token)

    def emit_json(self, *, token: str | None = None, indent: int | None = 2) -> str:
        """Serialize the graph (or a single token chain) as JSON."""
        return self.graph.to_json(token=token, indent=indent)

    def print_chains(self, *, token: str | None = None) -> None:
        """Print human-readable chain summary to stdout."""
        self.graph.print_chains(token=token)

    def write_sqlite(self, db_path: str) -> int:
        """Persist all edges to SQLite.  Returns rows written."""
        return self.graph.write_sqlite(db_path)

    def write_jsonl(self, log_path: str, *, append: bool = True) -> int:
        """Append all edges to a JSONL log.  Returns lines written."""
        return self.graph.write_jsonl(log_path, append=append)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _tuple_to_service_label(t: SocketTuple) -> str:
    """
    Convert a socket tuple to a human-readable service label.

    Falls back to ``ip:port`` notation when no symbolic name is available.
    """
    dst_ip, dst_port = t[2], t[3]
    if dst_ip and dst_port:
        return f"{dst_ip}:{dst_port}"
    if dst_ip:
        return dst_ip
    return "unknown"


def extract_tokens_from_payload(
    payload: str | bytes,
    *,
    prefix: str = "TAINT=",
    suffix: str = ";",
) -> list[str]:
    """
    Extract all taint tokens from a raw payload string or bytes.

    Default pattern matches the testbed format: ``TAINT=<uuid>;``

    Returns a list of full token strings including prefix and suffix,
    e.g. ``["TAINT=abc123-...-xyz;"]``.
    """
    import re

    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8", errors="replace")
        except Exception:
            return []
    else:
        text = payload

    escaped_prefix = re.escape(prefix)
    escaped_suffix = re.escape(suffix)
    pattern = re.compile(rf"{escaped_prefix}([^{re.escape(suffix[0])}]+){escaped_suffix}")
    return [m.group(0) for m in pattern.finditer(text)]
