#!/usr/bin/env python3
"""
EcodiaOS Observatory CLI

A terminal-first diagnostic tool for observing the organism's nervous system.

Usage:
    python -m cli.observatory snapshot          # Full diagnostic report
    python -m cli.observatory watch             # Live event stream (via Redis)
    python -m cli.observatory watch --system soma --type interoceptive_percept
    python -m cli.observatory flow              # Event flow graph
    python -m cli.observatory dead-letters      # Dead letter queue
    python -m cli.observatory missing           # Spec-defined events never observed
    python -m cli.observatory closures          # Closure loop health
    python -m cli.observatory vitality          # Vitality thresholds
    python -m cli.observatory drives            # Live constitutional drive state + weights
    python -m cli.observatory drift             # Constitutional drift history + severity
    python -m cli.observatory genome            # Genome snapshot (drives, weights, mutations)
    python -m cli.observatory incidents         # Active Thymos incidents
    python -m cli.observatory metabolic         # Oikos metabolic state + survival cascade

Requires the organism to be running (hits http://localhost:8000).
The 'watch' command connects directly to Redis pub/sub for real-time streaming.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import httpx

API_BASE = os.environ.get("ORGANISM_API_URL", "http://localhost:8000")


def _build_redis_url() -> str:
    url = os.environ.get("ORGANISM_REDIS__URL", os.environ.get("REDIS_URL", "redis://localhost:6379")).strip()
    pw = os.environ.get("ORGANISM_REDIS_PASSWORD", "").strip()
    if pw and "://" in url and "@" not in url:
        scheme, rest = url.split("://", 1)
        url = f"{scheme}://:{pw}@{rest}"
    return url


REDIS_URL = _build_redis_url()

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"

# System -> color mapping for visual distinction
_SYSTEM_COLORS: dict[str, str] = {
    "synapse": _CYAN,
    "soma": _MAGENTA,
    "oikos": _GREEN,
    "thymos": _RED,
    "nova": _BLUE,
    "axon": _YELLOW,
    "equor": _WHITE,
    "simula": _CYAN,
    "evo": _GREEN,
    "telos": _MAGENTA,
    "fovea": _BLUE,
    "atune": _BLUE,
    "logos": _WHITE,
    "kairos": _YELLOW,
    "oneiros": _MAGENTA,
    "nexus": _CYAN,
    "memory": _GREEN,
    "skia": _RED,
    "thread": _YELLOW,
    "benchmarks": _WHITE,
    "identity": _CYAN,
    "federation": _GREEN,
    "mitosis": _MAGENTA,
    "phantom": _RED,
    "sacm": _YELLOW,
    "voxis": _BLUE,
}


def _color(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def _system_color(system: str) -> str:
    return _SYSTEM_COLORS.get(system, _WHITE)


def _status_icon(status: str) -> str:
    if status == "ACTIVE":
        return _color("OK", _GREEN)
    if status == "STALE":
        return _color("STALE", _YELLOW)
    if status == "NEVER_FIRED":
        return _color("NEVER", _RED)
    return status


def _format_ago(seconds: float | None) -> str:
    if seconds is None:
        return "never"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms ago"
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


# ─── API Helpers ──────────────────────────────────────────────────


async def _get(path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=API_BASE, timeout=10) as client:
        r = await client.get(path)
        r.raise_for_status()
        return r.json()


# ─── Commands ─────────────────────────────────────────────────────


async def cmd_snapshot() -> None:
    """Full diagnostic snapshot."""
    data = await _get("/api/v1/observatory/snapshot")

    events = data.get("events", {})
    closures_data = data.get("closure_loops", [])
    spec = data.get("spec_compliance", {})
    bus = data.get("bus_stats", {})
    dlq = data.get("dead_letters", {})

    print()
    print(_color("=== EcodiaOS Observatory Snapshot ===", _BOLD))
    print()

    # Uptime & throughput
    uptime = events.get("uptime_s", 0)
    total = events.get("total_events", 0)
    eps = events.get("events_per_second", 0)
    print(f"  Uptime: {_format_duration(uptime)}")
    print(f"  Events: {total:,} total ({eps:.1f}/s)")
    print(f"  Types seen: {events.get('unique_types_seen', 0)} / {spec.get('total_defined', '?')}")
    print(f"  Systems active: {events.get('unique_systems_seen', 0)}")
    print()

    # Bus health
    print(_color("  Bus Health:", _BOLD))
    print(f"    Subscribers: {bus.get('subscriber_count', '?')}")
    print(f"    Callback timeouts: {bus.get('callback_timeouts', 0)}")
    print(f"    Callback failures: {bus.get('callback_failures', 0)}")
    print(f"    Dead letters: {bus.get('dead_lettered', 0)}")
    print(f"    Rate limited: {bus.get('rate_limited', 0)}")
    print(f"    Redis failures: {bus.get('redis_failures', 0)}")
    print()

    # Top emitters
    top = events.get("top_emitters", [])
    if top:
        print(_color("  Top Emitters (5m):", _BOLD))
        for entry in top[:8]:
            sys_id = entry.get("system", "?")
            count_5m = entry.get("5m", 0)
            color = _system_color(sys_id)
            bar = "#" * min(count_5m // 10, 40)
            print(f"    {_color(f'{sys_id:>12s}', color)} {count_5m:>6d}  {_color(bar, _DIM)}")
        print()

    # Closure loops
    if closures_data:
        print(_color("  Closure Loops:", _BOLD))
        for loop in closures_data:
            status = _status_icon(loop["status"])
            name = loop["name"]
            critical = " [CRITICAL]" if loop["is_critical"] else ""
            trigger_ago = _format_ago(loop.get("last_trigger_ago_s"))
            latency = loop.get("avg_latency_ms")
            lat_str = f" ({latency:.0f}ms avg)" if latency else ""
            print(f"    {status:>20s}  {name}{critical} (last: {trigger_ago}){lat_str}")
        print()

    # Missing events summary
    missing_count = spec.get("missing_count", 0)
    if missing_count > 0:
        print(_color(f"  Missing Events: {missing_count} spec-defined events never observed", _YELLOW))

        # Per-system coverage (worst first)
        coverage = spec.get("per_system_coverage", [])
        worst = [c for c in coverage if c.get("coverage_pct") is not None and c["coverage_pct"] < 100]
        if worst:
            for entry in worst[:10]:
                sys_id = entry["system"]
                pct = entry["coverage_pct"]
                obs = entry["observed"]
                exp = entry["expected"]
                color = _RED if pct < 30 else (_YELLOW if pct < 70 else _GREEN)
                print(f"    {_color(f'{sys_id:>12s}', _system_color(sys_id))} {_color(f'{pct:.0f}%', color)} ({obs}/{exp})")
        print()

    # Dead letters
    dlq_count = dlq.get("count", 0)
    if dlq_count > 0:
        print(_color(f"  Dead Letters: {dlq_count}", _RED))
        for item in dlq.get("recent", [])[-5:]:
            print(f"    {item['event_type']} from {item['source']}: {item['reason'][:80]}")
        print()

    # Silent systems
    per_sys = events.get("per_system", {})
    all_known = list(_SYSTEM_COLORS.keys())
    silent = [s for s in all_known if s not in per_sys]
    if silent:
        print(_color(f"  Silent Systems: {', '.join(silent)}", _RED))
        print()


async def cmd_watch(system: str | None = None, event_type: str | None = None) -> None:
    """Live event stream via Redis pub/sub."""
    try:
        import redis.asyncio as aioredis
    except ImportError:
        print("Error: redis[async] package required. Install with: pip install redis")
        sys.exit(1)

    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = r.pubsub()

    # Determine the channel - check if there's a prefix
    # Try the standard channel first
    channel = "synapse_events"
    # Check for prefix pattern used by RedisClient
    prefix_env = os.environ.get("REDIS_KEY_PREFIX", "")
    if prefix_env:
        channel = f"{prefix_env}:channel:synapse_events"
    else:
        # Try both patterns
        channel = "*:channel:synapse_events"

    if "*" in channel:
        await pubsub.psubscribe(channel)
        print(f"Subscribed to pattern: {channel}")
    else:
        await pubsub.subscribe(channel)
        print(f"Subscribed to: {channel}")

    filters = []
    if system:
        filters.append(f"system={system}")
    if event_type:
        filters.append(f"type={event_type}")
    if filters:
        print(f"Filters: {', '.join(filters)}")

    print(f"Listening for events... (Ctrl+C to stop)")
    print()

    try:
        async for message in pubsub.listen():
            if message["type"] not in ("message", "pmessage"):
                continue

            try:
                raw = message["data"]
                if isinstance(raw, str):
                    data = json.loads(raw)
                elif isinstance(raw, bytes):
                    data = json.loads(raw.decode())
                else:
                    continue
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            source = data.get("source", "?")
            etype = data.get("type", "?")
            ts = data.get("ts", "")
            payload = data.get("data", {})

            # Apply filters
            if system and source != system:
                continue
            if event_type and etype != event_type:
                continue

            # Format timestamp
            try:
                dt = datetime.fromisoformat(ts)
                ts_str = dt.strftime("%H:%M:%S.%f")[:-3]
            except (ValueError, TypeError):
                ts_str = ts[:12] if ts else "??:??:??"

            color = _system_color(source)
            # Compact payload display
            payload_str = ""
            if payload:
                # Show first 3 keys with values truncated
                items = list(payload.items())[:3]
                parts = []
                for k, v in items:
                    v_str = str(v)
                    if len(v_str) > 30:
                        v_str = v_str[:27] + "..."
                    parts.append(f"{k}={v_str}")
                payload_str = " " + " ".join(parts)
                if len(payload) > 3:
                    payload_str += f" (+{len(payload) - 3})"

            print(f"{_DIM}{ts_str}{_RESET} {_color(f'{source:>12s}', color)} {etype}{_DIM}{payload_str}{_RESET}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        await pubsub.unsubscribe()
        await r.aclose()


async def cmd_flow() -> None:
    """Event flow graph."""
    data = await _get("/api/v1/observatory/flow")
    flow = data.get("flow", {})

    print()
    print(_color("=== Event Flow Graph ===", _BOLD))
    print()

    if not flow:
        print("  No events observed yet.")
        return

    for sys_id, etypes in sorted(flow.items()):
        color = _system_color(sys_id)
        print(f"  {_color(sys_id, color)}")
        for etype in etypes:
            print(f"    -> {etype}")
        print()


async def cmd_dead_letters() -> None:
    """Dead letter queue."""
    data = await _get("/api/v1/observatory/dead-letters")
    total = data.get("total", 0)
    items = data.get("items", [])

    print()
    print(_color(f"=== Dead Letter Queue ({total} total) ===", _BOLD))
    print()

    if not items:
        print("  Queue is empty.")
        return

    for item in items:
        source = item.get("source", "?")
        etype = item.get("event_type", "?")
        reason = item.get("reason", "?")
        ts = item.get("timestamp", "")
        color = _system_color(source)
        print(f"  {_DIM}{ts[:19]}{_RESET} {_color(f'{source:>12s}', color)} {etype}")
        print(f"    Reason: {reason}")
        print()


async def cmd_missing() -> None:
    """Events defined in spec but never observed."""
    data = await _get("/api/v1/observatory/missing")

    print()
    print(_color("=== Spec Compliance Report ===", _BOLD))
    print()

    total_def = data.get("total_defined", 0)
    total_obs = data.get("total_observed", 0)
    missing = data.get("missing_count", 0)

    print(f"  Defined: {total_def}")
    print(f"  Observed: {total_obs}")
    print(f"  Missing: {_color(str(missing), _RED if missing > 20 else _YELLOW)}")
    print()

    # Per-system coverage
    coverage = data.get("per_system_coverage", [])
    if coverage:
        print(_color("  Per-System Coverage:", _BOLD))
        for entry in coverage:
            sys_id = entry["system"]
            pct = entry.get("coverage_pct")
            obs = entry["observed"]
            exp = entry["expected"]

            if pct is None:
                print(f"    {_color(f'{sys_id:>12s}', _system_color(sys_id))} (no events expected)")
                continue

            color = _RED if pct < 30 else (_YELLOW if pct < 70 else _GREEN)
            bar_len = int(pct / 5)
            bar = _color("#" * bar_len, color) + _DIM + "-" * (20 - bar_len) + _RESET

            print(f"    {_color(f'{sys_id:>12s}', _system_color(sys_id))} [{bar}] {_color(f'{pct:.0f}%', color)} ({obs}/{exp})")

            # Show missing events for this system
            if entry.get("missing"):
                for evt in entry["missing"]:
                    print(f"    {'':>12s}   {_DIM}x {evt}{_RESET}")
        print()


async def cmd_closures() -> None:
    """Closure loop health."""
    data = await _get("/api/v1/observatory/closures")
    summary = data.get("summary", {})
    loops = data.get("loops", [])

    print()
    print(_color("=== Closure Loop Health ===", _BOLD))
    print()
    print(f"  Total: {summary.get('total', 0)}")
    print(f"  Active: {_color(str(summary.get('active', 0)), _GREEN)}")
    print(f"  Stale: {_color(str(summary.get('stale', 0)), _YELLOW)}")
    print(f"  Never fired: {_color(str(summary.get('never_fired', 0)), _RED)}")
    crit = summary.get("critical_unhealthy", 0)
    if crit:
        print(f"  {_color(f'CRITICAL UNHEALTHY: {crit}', _RED)}")
    print()

    for loop in loops:
        status = _status_icon(loop["status"])
        name = loop["name"]
        src = loop["source"]
        sink = loop["sink"]
        trigger = loop["trigger_event"]
        response = loop["response_event"]
        critical = _color(" [CRITICAL]", _RED) if loop["is_critical"] else ""
        latency = loop.get("avg_latency_ms")

        print(f"  {status:>20s}  {name}{critical}")
        print(f"    {'':>20s}  {_color(src, _system_color(src))} --[{trigger}]--> {_color(sink, _system_color(sink))} --[{response}]-->")
        print(f"    {'':>20s}  Triggers: {loop['trigger_count']}, Responses: {loop['response_count']}", end="")
        if latency:
            print(f", Avg latency: {latency:.0f}ms", end="")
        print()
        print()


async def cmd_vitality() -> None:
    """Vitality thresholds."""
    data = await _get("/api/v1/observatory/vitality")
    vitality = data.get("vitality")

    print()
    print(_color("=== Vitality Status ===", _BOLD))
    print()

    if isinstance(vitality, str):
        print(f"  {vitality}")
    elif isinstance(vitality, dict):
        for key, val in vitality.items():
            # Highlight thresholds near breach
            val_str = str(val)
            try:
                f = float(val)
                color = _GREEN if f > 0.5 else (_YELLOW if f > 0.2 else _RED)
                val_str = _color(f"{f:.3f}", color)
            except (ValueError, TypeError):
                pass
            print(f"  {key:40s} {val_str}")
    else:
        print(f"  {vitality}")
    print()


async def cmd_drives() -> None:
    """Live constitutional drive state - the most important debug surface."""
    data = await _get("/api/v1/equor/health")

    print()
    print(_color("=== Constitutional Drive State ===", _BOLD))
    print()

    # Try to surface drive scores from health endpoint
    drift_severity = data.get("drift_severity", None)
    autonomy_level = data.get("autonomy_level", "?")
    constitution_version = data.get("constitution_version", "?")
    amendments_active = data.get("amendments_active", 0)
    safe_mode = data.get("safe_mode", False)

    if safe_mode:
        print(_color("  !! SAFE MODE ACTIVE - only Level 1 actions permitted !!", _RED))
        print()

    print(f"  Autonomy Level:        {_color(str(autonomy_level), _CYAN)}")
    print(f"  Constitution Version:  {constitution_version}")
    print(f"  Active Amendments:     {amendments_active}")

    if drift_severity is not None:
        try:
            ds = float(drift_severity)
            color = _GREEN if ds < 0.3 else (_YELLOW if ds < 0.7 else _RED)
            bar_len = int(ds * 20)
            bar = _color("#" * bar_len, color) + _DIM + "-" * (20 - bar_len) + _RESET
            print(f"  Drift Severity:        [{bar}] {_color(f'{ds:.3f}', color)}")
            if ds >= 0.9:
                print(_color("  !! CRITICAL DRIFT - immune response should be active (SG1) !!", _RED))
            elif ds >= 0.7:
                print(_color("  WARNING: Thymos incident should have fired", _YELLOW))
        except (ValueError, TypeError):
            print(f"  Drift Severity:        {drift_severity}")
    print()

    # Per-drive scores from last alignment score event (may not be in health endpoint)
    alignment = data.get("last_alignment_scores", {})
    if alignment:
        print(_color("  Drive Alignment Scores:", _BOLD))
        drives = [
            ("Coherence", "coherence", _CYAN),
            ("Care",      "care",      _MAGENTA),
            ("Growth",    "growth",    _GREEN),
            ("Honesty",   "honesty",   _WHITE),
        ]
        for label, key, color in drives:
            score = alignment.get(key)
            if score is not None:
                try:
                    s = float(score)
                    bar_len = int((s + 1.0) * 10)  # -1..+1 → 0..20
                    bar = _color("#" * bar_len, color) + _DIM + "-" * (20 - bar_len) + _RESET
                    status = ""
                    if key in ("care", "honesty") and s < -0.105:
                        status = _color(" !! BELOW FLOOR", _RED)
                    elif s < 0:
                        status = _color(" (negative)", _YELLOW)
                    print(f"    {label:10s} [{bar}] {_color(f'{s:+.3f}', color)}{status}")
                except (ValueError, TypeError):
                    print(f"    {label:10s} {score}")
        print()

    # Drive weights (evolvable phenotype)
    weights = data.get("drive_weights", {})
    if weights:
        print(_color("  Drive Weights (evolvable phenotype):", _BOLD))
        total = sum(float(v) for v in weights.values() if v)
        for drive, w in weights.items():
            try:
                pct = float(w) / total * 100 if total else 0
                print(f"    {drive:10s} {float(w):.3f}  ({pct:.1f}% of total)")
            except (ValueError, TypeError):
                print(f"    {drive:10s} {w}")
        print()

    # INV-017 sentinel
    violations = data.get("invariant_violations_detected", 0)
    if violations:
        print(_color(f"  Invariant Violations (recent): {violations}", _RED))
        print()


async def cmd_drift() -> None:
    """Constitutional drift history and immune response status."""
    data = await _get("/api/v1/equor/drift")

    print()
    print(_color("=== Constitutional Drift History ===", _BOLD))
    print()

    current = data.get("current_severity", data.get("drift_severity", None))
    window_size = data.get("window_size", "?")
    samples = data.get("recent_samples", [])
    immune_active = data.get("immune_response_active", data.get("recommended_action") == "immune_response")
    amendment_proposals = data.get("pending_amendment_proposals", 0)

    if immune_active:
        print(_color("  IMMUNE RESPONSE ACTIVE: Soma signal + Thymos incident in flight", _YELLOW))
        print()

    if current is not None:
        try:
            ds = float(current)
            color = _GREEN if ds < 0.3 else (_YELLOW if ds < 0.7 else _RED)
            print(f"  Current severity:   {_color(f'{ds:.3f}', color)}")
        except (ValueError, TypeError):
            print(f"  Current severity:   {current}")

    print(f"  Tracking window:    {window_size} reviews")
    print(f"  Pending amendments: {amendment_proposals}")
    print()

    if samples:
        print(_color("  Recent drift samples (newest last):", _BOLD))
        for s in samples[-20:]:
            ts = s.get("timestamp", "")[:19]
            sev = s.get("severity", 0)
            drive = s.get("drifting_drive", "")
            try:
                sev_f = float(sev)
                color = _GREEN if sev_f < 0.3 else (_YELLOW if sev_f < 0.7 else _RED)
                sev_str = _color(f"{sev_f:.3f}", color)
            except (ValueError, TypeError):
                sev_str = str(sev)
            drive_str = f"  [{drive}]" if drive else ""
            print(f"    {_DIM}{ts}{_RESET} severity={sev_str}{drive_str}")
        print()
    else:
        print("  No drift history yet.")
        print()


async def cmd_genome() -> None:
    """Genome snapshot - heritable constitutional phenotype."""
    data = await _get("/api/v1/equor/genome")

    print()
    print(_color("=== Constitutional Genome Snapshot ===", _BOLD))
    print()

    genome = data.get("genome", data)  # some endpoints wrap, some don't

    version = genome.get("constitution_version", "?")
    extracted_at = genome.get("extracted_at", "?")
    print(f"  Version:          {version}")
    print(f"  Extracted:        {str(extracted_at)[:19]}")
    print()

    # Drive weights
    weights = genome.get("drive_weights", {})
    if weights:
        print(_color("  Drive Weights:", _BOLD))
        for drive, w in weights.items():
            print(f"    {drive:10s} {float(w):.4f}")
        print()

    # Floor thresholds (evolvable)
    floors = genome.get("floor_thresholds", {})
    if floors:
        print(_color("  Floor Thresholds (evolvable):", _BOLD))
        for k, v in floors.items():
            print(f"    {k:30s} {v}")
        print()

    # Drift history summary
    drift_history = genome.get("drift_history", [])
    if drift_history:
        avg_drift = sum(d.get("severity", 0) for d in drift_history) / len(drift_history)
        print(f"  Drift history:    {len(drift_history)} records, avg severity={avg_drift:.3f}")
        print()

    # Active amendments
    amendments = genome.get("amendments", [])
    if amendments:
        print(_color(f"  Active Amendments ({len(amendments)}):", _BOLD))
        for a in amendments[:5]:
            print(f"    [{a.get('status', '?')}] {a.get('description', '?')[:70]}")
        print()

    # Mutation count from Simula
    mutations = genome.get("simula_mutation_count", None)
    if mutations is not None:
        print(f"  Simula mutations: {mutations}")
        print()


async def cmd_incidents() -> None:
    """Active Thymos incidents - what the immune system is currently fighting."""
    data = await _get("/api/v1/thymos/incidents")

    if isinstance(data, list):
        incidents = data
    else:
        incidents = data.get("active_incidents", data.get("incidents", []))
    total = len(incidents) if isinstance(data, list) else data.get("total_active", len(incidents))

    print()
    print(_color(f"=== Active Thymos Incidents ({total}) ===", _BOLD))
    print()

    if not incidents:
        print("  No active incidents.")
        print()
        return

    for inc in incidents:
        severity = inc.get("severity", "?")
        tier = inc.get("repair_tier", "?")
        cls = inc.get("incident_class", "?")
        system = inc.get("source_system", "?")
        status = inc.get("status", "?")
        occurrence = inc.get("occurrence_count", 1)
        first_seen = inc.get("first_seen", "")[:19]
        description = inc.get("description", "")[:80]

        sev_color = _RED if severity in ("CRITICAL", "HIGH") else (_YELLOW if severity == "MEDIUM" else _WHITE)
        sys_color = _system_color(system)

        print(f"  {_color(severity, sev_color)} | Tier {tier} | {_color(system, sys_color)} | {cls}")
        print(f"    Status: {status}  |  Occurrences: {occurrence}  |  First: {first_seen}")
        if description:
            print(f"    {_DIM}{description}{_RESET}")
        print()


async def cmd_metabolic() -> None:
    """Oikos metabolic state - survival cascade and economic health."""
    data = await _get("/api/v1/oikos/state")

    print()
    print(_color("=== Metabolic State ===", _BOLD))
    print()

    # /api/v1/oikos/state wraps payload under "data"
    state = data.get("data", data.get("metabolic_state", data))

    # Core survival metrics
    efficiency = state.get("metabolic_efficiency", None)
    liquid = state.get("liquid_balance", None)
    burn = state.get("bmr_usd_per_day", None)
    runway = state.get("runway_days", None)
    survival_reserve = state.get("survival_reserve", None)

    if efficiency is not None:
        try:
            eff = float(efficiency)
            color = _GREEN if eff >= 1.0 else (_YELLOW if eff >= 0.5 else _RED)
            status = "SELF-SUSTAINING" if eff >= 1.0 else ("DEGRADED" if eff >= 0.5 else "CRITICAL - below survival")
            print(f"  Metabolic efficiency: {_color(f'{eff:.3f}', color)}  ({status})")
        except (ValueError, TypeError):
            print(f"  Metabolic efficiency: {efficiency}")

    if liquid is not None:
        print(f"  Liquid balance:       ${float(liquid):.2f}")
    if burn is not None:
        print(f"  Burn rate:            ${float(burn):.2f}/day")
    if runway is not None:
        try:
            r = float(runway)
            color = _GREEN if r > 30 else (_YELLOW if r > 7 else _RED)
            print(f"  Runway:               {_color(f'{r:.1f} days', color)}")
        except (ValueError, TypeError):
            print(f"  Runway:               {runway}")
    if survival_reserve is not None:
        print(f"  Survival reserve:     ${float(survival_reserve):.2f}")
    print()

    # Metabolic cascade
    cascade = state.get("cascade_status", {})
    if cascade:
        print(_color("  Metabolic Cascade:", _BOLD))
        tiers = [
            "survival", "operations", "obligations",
            "maintenance", "growth", "yield", "assets", "reproduction",
        ]
        for tier in tiers:
            tier_state = cascade.get(tier, "?")
            color = _GREEN if tier_state == "funded" else (_YELLOW if tier_state == "partial" else _RED)
            print(f"    {tier:12s} {_color(str(tier_state), color)}")
        print()

    # Active bounties
    bounties = state.get("active_bounty_count", None)
    revenue = state.get("total_revenue_usd", None)
    if bounties is not None:
        print(f"  Active bounties:      {bounties}")
    if revenue is not None:
        print(f"  Total revenue:        ${float(revenue):.2f}")
    print()


# ─── Main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EcodiaOS Observatory - organism diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    sub.add_parser("all", help="Run all diagnostic commands in sequence")
    sub.add_parser("snapshot", help="Full diagnostic report")

    watch_p = sub.add_parser("watch", help="Live event stream via Redis")
    watch_p.add_argument("--system", "-s", help="Filter by source system")
    watch_p.add_argument("--type", "-t", dest="event_type", help="Filter by event type")

    sub.add_parser("flow", help="Event flow graph")
    sub.add_parser("dead-letters", help="Dead letter queue")
    sub.add_parser("missing", help="Spec compliance - missing events")
    sub.add_parser("closures", help="Closure loop health")
    sub.add_parser("vitality", help="Vitality thresholds")
    sub.add_parser("drives", help="Live constitutional drive state + weights")
    sub.add_parser("drift", help="Constitutional drift history + severity")
    sub.add_parser("genome", help="Genome snapshot (drives, weights, mutations)")
    sub.add_parser("incidents", help="Active Thymos incidents")
    sub.add_parser("metabolic", help="Oikos metabolic state + survival cascade")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "all":
        async def cmd_all() -> None:
            for cmd in [
                cmd_missing, cmd_genome, cmd_snapshot, cmd_drives,
                cmd_drift, cmd_closures, cmd_metabolic, cmd_incidents, cmd_vitality,
            ]:
                try:
                    await cmd()
                except Exception as e:
                    print(_color(f"  [error in {cmd.__name__}] {e}", _RED))
                    print()
        asyncio.run(cmd_all())
    elif args.command == "snapshot":
        asyncio.run(cmd_snapshot())
    elif args.command == "watch":
        asyncio.run(cmd_watch(system=args.system, event_type=args.event_type))
    elif args.command == "flow":
        asyncio.run(cmd_flow())
    elif args.command == "dead-letters":
        asyncio.run(cmd_dead_letters())
    elif args.command == "missing":
        asyncio.run(cmd_missing())
    elif args.command == "closures":
        asyncio.run(cmd_closures())
    elif args.command == "vitality":
        asyncio.run(cmd_vitality())
    elif args.command == "drives":
        asyncio.run(cmd_drives())
    elif args.command == "drift":
        asyncio.run(cmd_drift())
    elif args.command == "genome":
        asyncio.run(cmd_genome())
    elif args.command == "incidents":
        asyncio.run(cmd_incidents())
    elif args.command == "metabolic":
        asyncio.run(cmd_metabolic())


if __name__ == "__main__":
    main()
