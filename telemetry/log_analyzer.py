"""
EcodiaOS — Log Analyzer & Aggregator

Intelligent log ingestion and analysis. Goes beyond simple heuristics:
- Dependency-aware cascade detection (knows system topology)
- Temporal correlation (what happened before the failure?)
- LLM-powered root cause diagnosis via Thymos DiagnosticEngine
- Redis Streams for durable, time-windowed log queries
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.redis import RedisClient

logger = structlog.get_logger()


# ─── System Dependency Graph ──────────────────────────────────────
# Mirrors core/wiring.py's declare_dependencies — kept here as a
# lightweight lookup so the analyzer doesn't need Synapse at runtime.

_UPSTREAM_DEPS: dict[str, list[str]] = {
    "nova": ["memory", "equor", "axon", "evo", "thymos", "soma", "telos", "logos", "oikos"],
    "axon": ["nova", "atune", "simula", "fovea", "oneiros"],
    "equor": ["evo", "axon"],
    "evo": [
        "atune", "nova", "voxis", "simula", "thymos",
        "soma", "telos", "kairos", "fovea", "logos",
    ],
    "voxis": ["thread", "soma"],
    "thread": ["voxis", "equor", "atune", "evo", "nova", "fovea", "oneiros"],
    "atune": ["soma", "eis", "fovea"],
    "thymos": ["equor", "evo", "atune", "nova", "simula", "soma", "telos", "oikos"],
    "oneiros": [
        "equor", "evo", "nova", "atune", "thymos", "memory",
        "simula", "soma", "kairos", "logos", "fovea", "oikos",
    ],
    "soma": ["atune", "nova", "thymos", "equor", "telos", "oikos"],
    "logos": ["memory"],
    "simula": ["telos"],
    # Leaf systems (no upstream within cognitive layer)
    "memory": [],
    "synapse": [],
    "eis": [],
    "fovea": [],
    "kairos": [],
    "telos": [],
    "oikos": [],
}

# Invert: for each system, which systems depend on it?
_DOWNSTREAM: dict[str, list[str]] = defaultdict(list)
for _sys, _deps in _UPSTREAM_DEPS.items():
    for _dep in _deps:
        _DOWNSTREAM[_dep].append(_sys)


def _ms_epoch(dt: datetime) -> int:
    """Convert a datetime to millisecond epoch (Redis Stream ID format)."""
    return int(dt.timestamp() * 1000)


def _parse_stream_entries(raw: list[Any]) -> list[dict[str, str]]:
    """Convert redis-py xrange results to flat dicts.

    With ``decode_responses=True`` the driver returns:
        [(id_str, {field: value, ...}), ...]
    """
    return [{"id": entry_id, **fields} for entry_id, fields in raw]


class LogAnalyzer:
    """Intelligent log aggregation with dependency-aware analysis."""

    def __init__(self, redis_client: RedisClient) -> None:
        self._redis = redis_client
        self._stream_main = "eos:logs:main"
        self._error_stream = "eos:logs:errors"
        self._latency_stream = "eos:logs:latency"
        self._maxlen = 10_000

    # ── Ingestion ─────────────────────────────────────────────────

    async def ingest_log(self, **fields: Any) -> None:
        """Add a structured log entry to the appropriate streams."""
        ts = datetime.now(UTC).isoformat()
        entry: dict[str, str] = {"ts": ts}
        for k, v in fields.items():
            entry[k] = str(v)

        r = self._redis.client
        try:
            await r.xadd(self._stream_main, entry, maxlen=self._maxlen)  # type: ignore[arg-type]

            if entry.get("level") in ("error", "critical", "exception"):
                await r.xadd(self._error_stream, entry, maxlen=self._maxlen)  # type: ignore[arg-type]

            if "latency_ms" in entry:
                await r.xadd(self._latency_stream, entry, maxlen=self._maxlen)  # type: ignore[arg-type]
        except Exception as e:
            # Never block the logging path
            logger.warning("log_ingest_failed", error=str(e))

    # ── Basic Queries ─────────────────────────────────────────────

    async def get_recent_logs(
        self, minutes: int = 5, limit: int = 100,
    ) -> list[dict[str, str]]:
        """Fetch recent logs from the main stream."""
        try:
            cutoff_ms = _ms_epoch(datetime.now(UTC)) - minutes * 60_000
            raw = await self._redis.client.xrange(
                self._stream_main, min=str(cutoff_ms), max="+", count=limit,
            )
            return _parse_stream_entries(raw) if raw else []
        except Exception as e:
            logger.warning("log_fetch_failed", error=str(e))
            return []

    async def get_error_logs(
        self, minutes: int = 5, limit: int = 50,
    ) -> list[dict[str, str]]:
        """Fetch recent errors only."""
        try:
            cutoff_ms = _ms_epoch(datetime.now(UTC)) - minutes * 60_000
            raw = await self._redis.client.xrange(
                self._error_stream, min=str(cutoff_ms), max="+", count=limit,
            )
            return _parse_stream_entries(raw) if raw else []
        except Exception as e:
            logger.warning("error_fetch_failed", error=str(e))
            return []

    async def get_latency_summary(
        self, minutes: int = 5,
    ) -> dict[str, dict[str, Any]]:
        """Per-system latency stats: min, max, avg, p95."""
        try:
            cutoff_ms = _ms_epoch(datetime.now(UTC)) - minutes * 60_000
            raw = await self._redis.client.xrange(
                self._latency_stream, min=str(cutoff_ms), max="+",
            )
            if not raw:
                return {}

            by_system: dict[str, list[float]] = defaultdict(list)
            for _, fields in raw:
                system = fields.get("system", "unknown")
                try:
                    by_system[system].append(float(fields.get("latency_ms", "0")))
                except (ValueError, TypeError):
                    continue

            summary: dict[str, dict[str, Any]] = {}
            for system, lats in by_system.items():
                s = sorted(lats)
                n = len(s)
                summary[system] = {
                    "count": n,
                    "min_ms": round(s[0], 2),
                    "max_ms": round(s[-1], 2),
                    "avg_ms": round(sum(s) / n, 2),
                    "p95_ms": round(s[min(int(n * 0.95), n - 1)], 2),
                }
            return summary
        except Exception as e:
            logger.warning("latency_summary_failed", error=str(e))
            return {}

    # ── Intelligent Analysis ──────────────────────────────────────

    async def detect_cascades(self) -> list[dict[str, Any]]:
        """Dependency-aware cascade detection.

        Instead of naively grouping by exact error message, this:
        1. Groups errors by system
        2. Checks the dependency graph for upstream root causes
        3. Identifies whether a "cascade" is really N downstream
           systems failing because one upstream system is unhealthy
        """
        errors = await self.get_error_logs(minutes=5, limit=500)
        if not errors:
            return []

        # Group errors by system
        errors_by_system: dict[str, list[dict[str, str]]] = defaultdict(list)
        for err in errors:
            system = err.get("system", "unknown")
            errors_by_system[system].append(err)

        # For each system with errors, check if its upstream deps also have errors.
        # Walk the graph to find root cause systems.
        failing_systems = set(errors_by_system.keys())
        cascades: list[dict[str, Any]] = []
        visited_roots: set[str] = set()

        for system in failing_systems:
            root = self._find_root_cause(system, failing_systems)
            if root in visited_roots:
                continue
            visited_roots.add(root)

            # Collect all downstream systems that are failing because of this root
            affected = self._collect_downstream_failures(root, failing_systems)
            if len(affected) < 2:
                continue  # Not a cascade — single system failure

            # Collect sample error messages from affected systems
            sample_errors: list[dict[str, str]] = []
            for sys in affected:
                for err in errors_by_system.get(sys, [])[:2]:
                    sample_errors.append({
                        "system": sys,
                        "message": err.get("message", err.get("event", "?")),
                        "ts": err.get("ts", ""),
                    })

            cascades.append({
                "root_cause_system": root,
                "affected_systems": sorted(affected),
                "system_count": len(affected),
                "severity": (
                    "critical" if len(affected) >= 5
                    else "high" if len(affected) >= 3
                    else "medium"
                ),
                "total_errors": sum(len(errors_by_system.get(s, [])) for s in affected),
                "sample_errors": sample_errors[:6],
                "reasoning": (
                    f"{root} appears to be the root cause: "
                    f"{len(affected)} downstream systems are affected "
                    f"({', '.join(sorted(affected - {root}))})"
                ),
            })

        # Also detect non-topology cascades: same error message across 3+ systems
        msg_groups: dict[str, set[str]] = defaultdict(set)
        for err in errors:
            msg = err.get("message", err.get("event", ""))
            system = err.get("system", "unknown")
            if msg:
                msg_groups[msg].add(system)

        for msg, systems in msg_groups.items():
            if len(systems) >= 3:
                # Check if already covered by a topology cascade
                already_covered = any(
                    systems.issubset(set(c["affected_systems"]))
                    for c in cascades
                )
                if not already_covered:
                    cascades.append({
                        "root_cause_system": "unknown",
                        "affected_systems": sorted(systems),
                        "system_count": len(systems),
                        "severity": "high" if len(systems) >= 5 else "medium",
                        "total_errors": sum(
                            1 for e in errors
                            if e.get("message", e.get("event", "")) == msg
                        ),
                        "error_pattern": msg[:200],
                        "reasoning": (
                            f"Same error across {len(systems)} systems — "
                            f"possible shared dependency or global issue"
                        ),
                    })

        return sorted(cascades, key=lambda x: x["system_count"], reverse=True)

    async def get_temporal_context(
        self,
        system: str,
        minutes_before: int = 5,
        limit: int = 50,
    ) -> dict[str, Any]:
        """What happened in the N minutes before errors started in a system?

        Answers: "What was going on right before nova crashed?"
        Returns logs from all systems in the time window, grouped by system,
        with upstream dependency systems highlighted.
        """
        errors = await self.get_error_logs(minutes=minutes_before, limit=200)
        system_errors = [e for e in errors if e.get("system") == system]
        if not system_errors:
            return {
                "system": system,
                "status": "no_recent_errors",
                "errors_in_window": 0,
            }

        # Find the earliest error timestamp for this system
        first_error_ts = system_errors[0].get("ts", "")

        # Get ALL logs from the window for context
        all_logs = await self.get_recent_logs(minutes=minutes_before, limit=500)

        # Split into "before first error" and "after"
        before: list[dict[str, str]] = []
        after: list[dict[str, str]] = []
        for log in all_logs:
            if log.get("ts", "") < first_error_ts:
                before.append(log)
            else:
                after.append(log)

        # Group "before" logs by system for readability
        before_by_system: dict[str, list[dict[str, str]]] = defaultdict(list)
        for log in before[-limit:]:
            before_by_system[log.get("system", "unknown")].append(log)

        upstream = _UPSTREAM_DEPS.get(system, [])
        upstream_activity = {
            dep: before_by_system.get(dep, [])
            for dep in upstream
            if dep in before_by_system
        }

        return {
            "system": system,
            "first_error_ts": first_error_ts,
            "error_count": len(system_errors),
            "upstream_dependencies": upstream,
            "upstream_activity_before_failure": {
                dep: logs[-5:] for dep, logs in upstream_activity.items()
            },
            "other_activity_before_failure": {
                sys: logs[-3:]
                for sys, logs in before_by_system.items()
                if sys not in upstream and sys != system
            },
            "errors_after": [
                e for e in after
                if e.get("level") in ("error", "critical", "exception")
            ][:20],
        }

    def get_dependency_graph(self) -> dict[str, Any]:
        """Return the system dependency graph for visualization/debugging."""
        return {
            "upstream": dict(_UPSTREAM_DEPS),
            "downstream": dict(_DOWNSTREAM),
            "leaf_systems": [s for s, deps in _UPSTREAM_DEPS.items() if not deps],
            "most_depended_on": sorted(
                _DOWNSTREAM.keys(),
                key=lambda s: len(_DOWNSTREAM[s]),
                reverse=True,
            )[:10],
        }

    async def compute_interoceptive_signals(
        self, minutes: int = 5,
    ) -> list[dict[str, Any]]:
        """Compute aggregate interoceptive signals for Soma consumption.

        These represent the organism's internal health state derived from
        log patterns. Soma injects these as periodic signals into its
        allostatic controller to modulate urgency/arousal.

        Returns:
          - error_rate_signal: errors per minute, categorized as critical/high/low
          - cascade_pressure: number of active cascades × system count
          - latency_pressure: max p95 latency across all systems
          - error_concentration: how concentrated errors are (distributed = healthy)
        """
        errors = await self.get_error_logs(minutes=minutes, limit=1000)
        latency = await self.get_latency_summary(minutes=minutes)
        cascades = await self.detect_cascades()

        signals: list[dict[str, Any]] = []

        # Signal 1: Error rate trend
        error_count = len(errors)
        error_rate = error_count / max(minutes, 1)
        signals.append({
            "signal_type": "error_rate",
            "value": error_rate,
            "unit": "errors_per_minute",
            "severity": (
                "critical" if error_rate > 5.0
                else "high" if error_rate > 2.0
                else "low"
            ),
            "interpretation": f"{error_count} errors in {minutes}min = {error_rate:.2f}/min",
        })

        # Signal 2: Cascade pressure (systemic health)
        cascade_pressure = sum(c.get("system_count", 0) for c in cascades)
        signals.append({
            "signal_type": "cascade_pressure",
            "value": cascade_pressure,
            "unit": "affected_systems",
            "severity": (
                "critical" if cascade_pressure > 10
                else "high" if cascade_pressure > 3
                else "low"
            ),
            "interpretation": (
                f"{len(cascades)} cascades affecting {cascade_pressure} systems"
            ),
            "cascades": cascades[:3],
        })

        # Signal 3: Latency pressure (bottleneck detection)
        max_p95 = max(
            (s.get("p95_ms", 0) for s in latency.values()),
            default=0,
        )
        slowest_sys = max(
            latency.items(),
            key=lambda x: x[1].get("p95_ms", 0),
            default=(None, {}),
        )[0]
        signals.append({
            "signal_type": "latency_pressure",
            "value": max_p95,
            "unit": "milliseconds",
            "severity": (
                "critical" if max_p95 > 500
                else "high" if max_p95 > 200
                else "low"
            ),
            "interpretation": (
                f"Slowest system: {slowest_sys} at p95={max_p95:.1f}ms"
            ),
        })

        # Signal 4: Error distribution (concentration = risk)
        errors_by_system: dict[str, int] = {}
        for err in errors:
            sys = err.get("system", "unknown")
            errors_by_system[sys] = errors_by_system.get(sys, 0) + 1

        if errors_by_system:
            total = sum(errors_by_system.values())
            max_concentration = max(errors_by_system.values()) / total
            signals.append({
                "signal_type": "error_concentration",
                "value": max_concentration,
                "unit": "fraction",
                "severity": (
                    "high" if max_concentration > 0.7 else "medium"
                    if max_concentration > 0.5 else "low"
                ),
                "interpretation": (
                    f"Errors concentrated in {max_concentration*100:.0f}% "
                    f"of systems (good = distributed)"
                ),
                "affected_systems": sorted(
                    errors_by_system.keys(),
                    key=lambda s: errors_by_system[s],
                    reverse=True,
                )[:5],
            })

        return signals

    # ── Private helpers ───────────────────────────────────────────

    def _find_root_cause(
        self, system: str, failing_systems: set[str],
    ) -> str:
        """Walk upstream through the dependency graph to find the deepest
        failing ancestor. If no upstream is failing, the system itself is root."""
        visited: set[str] = set()
        current = system
        while True:
            visited.add(current)
            upstream = _UPSTREAM_DEPS.get(current, [])
            # Find the first upstream dep that is also failing
            deeper = None
            for dep in upstream:
                if dep in failing_systems and dep not in visited:
                    deeper = dep
                    break
            if deeper is None:
                return current
            current = deeper

    def _collect_downstream_failures(
        self, root: str, failing_systems: set[str],
    ) -> set[str]:
        """BFS from root through downstream edges, collecting all failing systems."""
        collected: set[str] = {root} if root in failing_systems else set()
        queue = [root]
        visited: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for downstream in _DOWNSTREAM.get(current, []):
                if downstream in failing_systems:
                    collected.add(downstream)
                    queue.append(downstream)
        return collected


# ── Global instance & init ────────────────────────────────────────

_analyzer: LogAnalyzer | None = None


async def initialize_analyzer(redis_client: RedisClient) -> LogAnalyzer:
    """Initialize the global log analyzer."""
    global _analyzer
    _analyzer = LogAnalyzer(redis_client)
    return _analyzer


def get_analyzer() -> LogAnalyzer:
    """Get the global log analyzer instance (raises if not initialized)."""
    if _analyzer is None:
        raise RuntimeError("Log analyzer not initialized")
    return _analyzer
