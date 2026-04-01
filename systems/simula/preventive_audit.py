"""
EcodiaOS - Simula Preventive Audit

Scheduled self-audit (every 4 hours) that scans recently generated/modified
code against known crash patterns, static fragility heuristics, and the
organism's own incident history - then proactively proposes fixes before
anything breaks.

Four-phase pipeline per audit cycle:
  Phase 1 - Fragility scan
    Query Memory for MemoryTrace entries tagged "generated_code" modified in
    the last 24 hours.  For each fragment, run five static checks:
      * Bare except clauses
      * asyncio.create_task without result capture
      * Division without zero-guard (denominator from external data)
      * Redis/Neo4j I/O calls without try/except
      * Missing await on coroutines (heuristic)
    Assigns fragility_score 0.0–1.0 per fragment.

  Phase 2 - Pattern matching
    For fragments with fragility_score > 0.3: match against self._known_fatal_patterns
    (the shared CrashPattern dict maintained by SimulaService).  Any match_score ≥ 0.5
    is flagged PREEMPTIVE_RISK.  combined_risk = fragility_score × pattern.confidence.

  Phase 3 - Incident history correlation
    Emit THYMOS_INCIDENT_QUERY, await THYMOS_INCIDENT_RESPONSE (10 s timeout).
    For each PREEMPTIVE_RISK item: if a fingerprint-similar incident occurred in the
    last 7 days, upgrade to PREEMPTIVE_CRITICAL.

  Phase 4 - Action
    PREEMPTIVE_CRITICAL (combined_risk > 0.7):
      • Generate a repair patch via SimulaService.process_proposal()
      • Emit THYMOS_REPAIR_REQUESTED with context["preventive"]=True
      • Emit RE_TRAINING_EXAMPLE category="preventive_repair" (outcome_quality=0.8)
    PREEMPTIVE_RISK (0.4–0.7):
      • Emit INCIDENT_DETECTED severity=LOW - flag for human review
    Below threshold: silent, metrics to Benchmarks only

Genome parameter:
  audit_aggressiveness: float [0.0, 1.0]
    0.0 - only flag combined_risk > 0.7 (CRITICAL only)
    1.0 - flag everything above the 0.2 threshold (maximum sensitivity)
  Evo can tune this via ADJUST_BUDGET mechanism.
"""

from __future__ import annotations

import ast
import asyncio
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.memory.service import MemoryService
    from systems.simula.service import SimulaService

logger = structlog.get_logger().bind(system="simula.preventive_audit")

# ─── Fragility check thresholds ─────────────────────────────────────────────

# Minimum fragility score to enter pattern-matching phase
_FRAGILITY_ENTER_THRESHOLD: float = 0.30

# Patch weights: how much each check contributes to fragility_score
_CHECK_WEIGHTS: dict[str, float] = {
    "bare_except": 0.25,
    "unhandled_create_task": 0.20,
    "division_no_guard": 0.20,
    "db_no_try_except": 0.25,
    "missing_await": 0.10,
}

# Pattern match score required to flag PREEMPTIVE_RISK
_PATTERN_MATCH_THRESHOLD: float = 0.50

# combined_risk > this → PREEMPTIVE_CRITICAL
_CRITICAL_THRESHOLD: float = 0.70

# Thymos incident query timeout (seconds)
_INCIDENT_QUERY_TIMEOUT_S: float = 10.0

# How many hours back to look for generated code traces
_GENERATED_CODE_LOOKBACK_H: int = 24

# Lookback window for incident history correlation (days)
_INCIDENT_LOOKBACK_DAYS: int = 7


# ─── Static fragility checks ─────────────────────────────────────────────────


def _check_bare_except(source: str) -> int:
    """Count bare `except:` or `except Exception:` clauses that re-raise nothing."""
    count = 0
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    count += 1
                elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    # Check body - if it's just `pass`, that's fragile
                    if all(
                        isinstance(stmt, (ast.Pass, ast.Expr))
                        for stmt in node.body
                    ):
                        count += 1
    except SyntaxError:
        pass
    return count


def _check_unhandled_create_task(source: str) -> int:
    """Count asyncio.create_task() calls whose return value is not captured."""
    count = 0
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            # Look for expression statements (result discarded) that call create_task
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                # asyncio.create_task(...)  or  ensure_future(...)
                if isinstance(call.func, ast.Attribute):
                    if call.func.attr in ("create_task", "ensure_future"):
                        count += 1
                elif isinstance(call.func, ast.Name):
                    if call.func.id in ("create_task", "ensure_future"):
                        count += 1
    except SyntaxError:
        pass
    return count


_DIVISION_PATTERN = re.compile(
    r"\b(\w+)\s*/\s*(\w+)",
)
_EXTERNAL_DATA_PATTERNS = re.compile(
    r"data\.get|event\.get|\[.+\]|int\(|float\(|payload",
    re.IGNORECASE,
)


def _check_division_no_guard(source: str) -> int:
    """
    Heuristic: count division expressions where denominator appears to come
    from external data (dict.get, event fields, casts from external input).
    True zero-guard analysis would require full data-flow; this is a fast proxy.
    """
    count = 0
    lines = source.splitlines()
    for line in lines:
        if "/" in line and _EXTERNAL_DATA_PATTERNS.search(line):
            # Basic skip if already guarded
            if "if " not in line and "or 1" not in line and "max(" not in line:
                count += 1
    return count


_DB_CALL_PATTERN = re.compile(
    r"\b(redis|neo4j|execute_read|execute_write|hget|hset|lpush|rpush|xadd"
    r"|set_json|get_json|execute)\b",
    re.IGNORECASE,
)
_TRY_EXCEPT_LINES = re.compile(r"\btry\b|\bexcept\b")


def _check_db_no_try_except(source: str) -> int:
    """
    Count DB/cache call lines that are not enclosed in any try/except block.
    Uses a simple line-level heuristic: if a DB call appears in a function
    that has zero try/except blocks, flag it.
    """
    count = 0
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if any try/except in this function
                has_try = any(
                    isinstance(child, ast.Try)
                    for child in ast.walk(node)
                )
                if not has_try:
                    # Count DB calls in the function source lines
                    func_lines = ast.get_source_segment(source, node) or ""
                    db_calls = len(_DB_CALL_PATTERN.findall(func_lines))
                    count += db_calls
    except SyntaxError:
        pass
    return count


_COROUTINE_PATTERN = re.compile(
    r"^\s*(?!await\b)(?!return\b)(?!result\s*=)(?!_\s*=)"
    r"([a-z_]\w+)\s*\(",
    re.MULTILINE,
)
_ASYNC_DEF_PATTERN = re.compile(r"\basync\s+def\s+(\w+)")


def _check_missing_await(source: str) -> int:
    """
    Heuristic: look for calls to known-async functions (those defined with
    `async def` in the same fragment) whose call sites lack `await`.
    This is a very loose proxy - avoids complex AST type inference.
    """
    count = 0
    async_fns = set(_ASYNC_DEF_PATTERN.findall(source))
    if not async_fns:
        return 0

    lines = source.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("await") or stripped.startswith("return"):
            continue
        if "=" in stripped:
            # assignment - fine
            continue
        for fn in async_fns:
            if re.search(rf"\b{re.escape(fn)}\s*\(", stripped):
                count += 1
    return count


# ─── Fragility scorer ─────────────────────────────────────────────────────────


def compute_fragility_score(source: str) -> tuple[float, dict[str, int]]:
    """
    Run all five checks on a source fragment and return
    (fragility_score: float [0, 1], hit_counts: dict[check_name, count]).

    Score is capped at 1.0 and weighted by _CHECK_WEIGHTS.
    """
    hits: dict[str, int] = {
        "bare_except": _check_bare_except(source),
        "unhandled_create_task": _check_unhandled_create_task(source),
        "division_no_guard": _check_division_no_guard(source),
        "db_no_try_except": _check_db_no_try_except(source),
        "missing_await": _check_missing_await(source),
    }

    raw_score = 0.0
    for check, cnt in hits.items():
        if cnt > 0:
            # Each additional hit adds diminishing returns (cap contribution at 3)
            raw_score += _CHECK_WEIGHTS[check] * min(cnt, 3) / 3
    return min(raw_score, 1.0), hits


# ─── FragilityResult ──────────────────────────────────────────────────────────


class FragilityResult:
    """Intermediate result for a single code fragment."""

    __slots__ = (
        "trace_id", "source_ref", "source", "fragility_score", "hit_counts",
        "pattern_id", "pattern_confidence", "match_score", "combined_risk",
        "risk_level", "incident_match",
    )

    def __init__(
        self,
        trace_id: str,
        source_ref: str,
        source: str,
        fragility_score: float,
        hit_counts: dict[str, int],
    ) -> None:
        self.trace_id = trace_id
        self.source_ref = source_ref
        self.source = source
        self.fragility_score = fragility_score
        self.hit_counts = hit_counts
        self.pattern_id: str | None = None
        self.pattern_confidence: float = 0.0
        self.match_score: float = 0.0
        self.combined_risk: float = 0.0
        self.risk_level: str = "below_threshold"  # "below_threshold" | "preemptive_risk" | "preemptive_critical"
        self.incident_match: bool = False


# ─── SimulaPreventiveAudit ────────────────────────────────────────────────────


class SimulaPreventiveAudit:
    """
    Periodic 4-hour audit that scans generated code for fragility and
    proactively issues repair proposals or incident flags before crashes occur.

    Wired into SimulaService after initialize():
      service._preventive_audit = SimulaPreventiveAudit(service)
      service._preventive_audit_task = supervised_task(
          service._preventive_audit.run_loop(), ...
      )

    Reads:
      service._known_fatal_patterns   - dict[pattern_id, CrashPattern]
      service._memory                 - MemoryService reference
      service._synapse                - event bus
      service._config.audit_aggressiveness - float [0, 1]
    """

    _INTERVAL_S: float = 4 * 3600  # 4 hours

    def __init__(self, service: SimulaService) -> None:
        self._service = service
        self._logger = logger
        # Pending THYMOS_INCIDENT_RESPONSE futures: request_id → Future
        self._incident_query_futures: dict[str, asyncio.Future[list[dict[str, Any]]]] = {}
        # Audit cycle counter (for metrics / logging)
        self._cycle: int = 0
        # Last audit timestamp
        self._last_run_at: float = 0.0

    # ── Event handler (wired by set_synapse) ──────────────────────────────

    async def on_thymos_incident_response(self, event: Any) -> None:
        """Resolve a pending THYMOS_INCIDENT_QUERY future."""
        data = getattr(event, "data", {}) or {}
        request_id = str(data.get("request_id", ""))
        incidents = data.get("incidents", [])
        fut = self._incident_query_futures.pop(request_id, None)
        if fut is not None and not fut.done():
            fut.set_result(list(incidents))

    # ── Main loop ─────────────────────────────────────────────────────────

    async def run_loop(self) -> None:
        """Background loop: sleep _INTERVAL_S then run one audit cycle."""
        # Stagger first run by half-interval so it doesn't pile up at boot
        try:
            await asyncio.sleep(self._INTERVAL_S / 2)
        except asyncio.CancelledError:
            return
        while True:
            try:
                await self._run_audit_cycle()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._logger.warning("preventive_audit_cycle_error", error=str(exc))
            try:
                await asyncio.sleep(self._INTERVAL_S)
            except asyncio.CancelledError:
                return

    # ── Audit cycle ───────────────────────────────────────────────────────

    async def _run_audit_cycle(self) -> None:
        self._cycle += 1
        self._last_run_at = time.monotonic()
        cycle = self._cycle

        self._logger.info("preventive_audit_started", cycle=cycle)

        # ── Phase 1: Fragility scan ────────────────────────────────────────
        fragments = await self._fetch_generated_code_traces()
        if not fragments:
            self._logger.debug("preventive_audit_no_fragments", cycle=cycle)
            return

        results: list[FragilityResult] = []
        for trace_id, source_ref, source in fragments:
            if not source or not source.strip():
                continue
            score, hits = compute_fragility_score(source)
            results.append(FragilityResult(
                trace_id=trace_id,
                source_ref=source_ref,
                source=source,
                fragility_score=score,
                hit_counts=hits,
            ))

        # ── Phase 2: Pattern matching ──────────────────────────────────────
        aggressiveness = self._audit_aggressiveness
        enter_threshold = max(0.2, _FRAGILITY_ENTER_THRESHOLD * (1.0 - aggressiveness * 0.5))

        candidates: list[FragilityResult] = [
            r for r in results if r.fragility_score > enter_threshold
        ]

        for result in candidates:
            self._match_crash_patterns(result)

        # ── Phase 3: Incident history correlation ─────────────────────────
        if candidates:
            incident_history = await self._query_thymos_incidents()
            for result in candidates:
                if result.risk_level == "preemptive_risk":
                    self._correlate_incidents(result, incident_history)

        # ── Phase 4: Actions ───────────────────────────────────────────────
        criticals = [r for r in candidates if r.risk_level == "preemptive_critical"]
        risks = [r for r in candidates if r.risk_level == "preemptive_risk"]
        below = [r for r in results if r.risk_level == "below_threshold"]

        for result in criticals:
            await self._act_on_critical(result)

        for result in risks:
            await self._act_on_risk(result)

        # Metrics for below-threshold items
        await self._emit_audit_metrics(
            cycle=cycle,
            total_fragments=len(results),
            criticals=len(criticals),
            risks=len(risks),
            below=len(below),
        )

        self._logger.info(
            "preventive_audit_complete",
            cycle=cycle,
            total_fragments=len(results),
            candidates=len(candidates),
            criticals=len(criticals),
            risks=len(risks),
        )

    # ── Phase 1 helpers ──────────────────────────────────────────────────

    async def _fetch_generated_code_traces(
        self,
    ) -> list[tuple[str, str, str]]:
        """
        Query Memory for MemoryTrace entries tagged "generated_code" modified
        in the last 24 hours.

        Returns list of (trace_id, source_ref, source_code) triples.
        Falls back to empty list on any error.
        """
        memory: MemoryService | None = getattr(self._service, "_memory", None)
        if memory is None:
            return []

        try:
            neo4j = getattr(memory, "_neo4j", None) or getattr(self._service, "_neo4j", None)
            if neo4j is None:
                return []

            rows = await neo4j.execute_read(
                """
                MATCH (m:MemoryTrace)
                WHERE "generated_code" IN coalesce(m.tags, [])
                  AND m.ingestion_time >= datetime() - duration({hours: $hours})
                RETURN m.id AS trace_id,
                       coalesce(m.source, m.id) AS source_ref,
                       coalesce(m.raw_content, m.summary, '') AS source
                ORDER BY m.ingestion_time DESC
                LIMIT 200
                """,
                {"hours": _GENERATED_CODE_LOOKBACK_H},
            )
            result: list[tuple[str, str, str]] = []
            for row in rows:
                tid = str(row.get("trace_id", "") or "")
                ref = str(row.get("source_ref", "") or "")
                src = str(row.get("source", "") or "")
                if src:
                    result.append((tid, ref, src))
            return result

        except Exception as exc:
            self._logger.debug("preventive_audit_fetch_traces_error", error=str(exc))
            return []

    # ── Phase 2 helpers ──────────────────────────────────────────────────

    @property
    def _audit_aggressiveness(self) -> float:
        config = getattr(self._service, "_config", None)
        if config is None:
            return 0.5
        return float(getattr(config, "audit_aggressiveness", 0.5))

    def _match_crash_patterns(self, result: FragilityResult) -> None:
        """
        Match result.source against known CrashPatterns.
        Sets result.pattern_id, result.match_score, result.combined_risk,
        and result.risk_level.
        """
        known: dict[str, Any] = getattr(self._service, "_known_fatal_patterns", {})
        if not known:
            return

        # Build feature set from this code fragment (reuse the check hits)
        fragment_features: frozenset[str] = frozenset(
            k for k, v in result.hit_counts.items() if v > 0
        )

        best_score = 0.0
        best_pattern_id = ""
        best_confidence = 0.0

        for pattern_id, pattern in known.items():
            sig = getattr(pattern, "signature", [])
            if not sig:
                continue
            intersection = fragment_features & frozenset(sig)
            if not sig:
                continue
            match_score = len(intersection) / len(sig)
            confidence = float(getattr(pattern, "confidence", 0.0))

            if match_score > best_score:
                best_score = match_score
                best_pattern_id = pattern_id
                best_confidence = confidence

        result.match_score = best_score
        result.pattern_id = best_pattern_id if best_score >= _PATTERN_MATCH_THRESHOLD else None
        result.pattern_confidence = best_confidence

        combined = result.fragility_score * max(best_confidence, 0.1)
        result.combined_risk = combined

        aggressiveness = self._audit_aggressiveness
        risk_threshold = max(0.2, _FRAGILITY_ENTER_THRESHOLD * (1.0 - aggressiveness * 0.5))

        if best_score >= _PATTERN_MATCH_THRESHOLD:
            result.risk_level = "preemptive_risk"
        elif result.fragility_score > risk_threshold:
            result.risk_level = "preemptive_risk"

    # ── Phase 3 helpers ──────────────────────────────────────────────────

    async def _query_thymos_incidents(self) -> list[dict[str, Any]]:
        """
        Emit THYMOS_INCIDENT_QUERY and await THYMOS_INCIDENT_RESPONSE.
        Returns the incident list or empty list on timeout/error.
        """
        bus = self._event_bus
        if bus is None:
            return []

        request_id = str(uuid.uuid4())
        fut: asyncio.Future[list[dict[str, Any]]] = asyncio.get_event_loop().create_future()
        self._incident_query_futures[request_id] = fut

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.THYMOS_INCIDENT_QUERY,
                source_system="simula",
                data={
                    "request_id": request_id,
                    "instance_id": getattr(self._service, "_instance_name", ""),
                    "lookback_days": _INCIDENT_LOOKBACK_DAYS,
                    "max_incidents": 200,
                },
            ))

            try:
                return await asyncio.wait_for(fut, timeout=_INCIDENT_QUERY_TIMEOUT_S)
            except asyncio.TimeoutError:
                self._incident_query_futures.pop(request_id, None)
                self._logger.debug("preventive_audit_incident_query_timeout")
                return []
        except Exception as exc:
            self._incident_query_futures.pop(request_id, None)
            self._logger.debug("preventive_audit_incident_query_error", error=str(exc))
            return []

    def _correlate_incidents(
        self,
        result: FragilityResult,
        incident_history: list[dict[str, Any]],
    ) -> None:
        """
        Check whether a recent incident has similar characteristics to this
        fragility result.  If found, upgrades risk_level to PREEMPTIVE_CRITICAL.
        """
        if not incident_history:
            return

        # Use a bag-of-words overlap between hit_counts keys and incident error strings
        result_tokens = set(result.hit_counts.keys()) | {
            w.lower() for w in re.split(r"\W+", result.source_ref)
            if len(w) > 3
        }

        for incident in incident_history:
            err_type = str(incident.get("error_type", "")).lower()
            err_msg = str(incident.get("error_message", "")).lower()
            inc_tokens = set(re.split(r"\W+", err_type + " " + err_msg))

            overlap = result_tokens & inc_tokens
            if len(overlap) >= 2:
                result.incident_match = True
                combined_upgraded = max(result.combined_risk, _CRITICAL_THRESHOLD + 0.01)
                result.combined_risk = combined_upgraded
                result.risk_level = "preemptive_critical"
                self._logger.debug(
                    "preventive_audit_incident_correlation",
                    source_ref=result.source_ref,
                    incident_id=incident.get("incident_id", ""),
                    overlap=list(overlap)[:5],
                )
                break  # one match is enough to upgrade

    # ── Phase 4 helpers ──────────────────────────────────────────────────

    async def _act_on_critical(self, result: FragilityResult) -> None:
        """
        PREEMPTIVE_CRITICAL: generate a repair patch and emit events.
        """
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            bus = self._event_bus

            # 1. Request Simula repair pipeline
            if bus is not None:
                await bus.emit(SynapseEvent(
                    event_type=SynapseEventType.THYMOS_REPAIR_REQUESTED,
                    source_system="simula",
                    data={
                        "incident_id": f"preventive:{result.trace_id}",
                        "incident_class": "fragile_code",
                        "severity": "medium",
                        "description": (
                            f"Preventive audit flagged fragile code in {result.source_ref}: "
                            f"fragility_score={result.fragility_score:.2f}, "
                            f"combined_risk={result.combined_risk:.2f}"
                        ),
                        "affected_system": result.source_ref,
                        "repair_tier": 4,
                        "context": {
                            "preventive": True,
                            "fragility_score": result.fragility_score,
                            "combined_risk": result.combined_risk,
                            "hit_counts": result.hit_counts,
                            "matched_pattern": result.pattern_id or "",
                            "incident_correlated": result.incident_match,
                            "source_ref": result.source_ref,
                        },
                    },
                ))

            # 2. RE training example
            await self._emit_preventive_re_training(result, action="repair_requested")

            self._logger.info(
                "preventive_audit_critical_action",
                source_ref=result.source_ref,
                fragility_score=result.fragility_score,
                combined_risk=result.combined_risk,
                pattern_id=result.pattern_id,
                incident_match=result.incident_match,
            )

        except Exception as exc:
            self._logger.warning("preventive_audit_critical_action_error", error=str(exc))

    async def _act_on_risk(self, result: FragilityResult) -> None:
        """
        PREEMPTIVE_RISK: emit INCIDENT_DETECTED for human review via Equor.
        """
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            bus = self._event_bus
            if bus is None:
                return

            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.INCIDENT_DETECTED,
                source_system="simula",
                data={
                    "incident_id": f"preventive_risk:{result.trace_id}",
                    "incident_class": "fragile_code",
                    "severity": "low",
                    "source_system": "simula",
                    "description": (
                        f"Fragile code detected pre-crash in {result.source_ref}: "
                        f"fragility_score={result.fragility_score:.2f}"
                    ),
                    "fragility_score": result.fragility_score,
                    "matched_patterns": [result.pattern_id] if result.pattern_id else [],
                    "hit_counts": result.hit_counts,
                    "combined_risk": result.combined_risk,
                    "source_ref": result.source_ref,
                    "preventive": True,
                },
            ))

            self._logger.info(
                "preventive_audit_risk_flagged",
                source_ref=result.source_ref,
                fragility_score=result.fragility_score,
                combined_risk=result.combined_risk,
            )

        except Exception as exc:
            self._logger.warning("preventive_audit_risk_action_error", error=str(exc))

    async def _emit_preventive_re_training(
        self,
        result: FragilityResult,
        action: str,
    ) -> None:
        """Fire RE_TRAINING_EXAMPLE for a preventive repair action."""
        try:
            from decimal import Decimal
            from primitives.common import DriveAlignmentVector, SystemID
            from primitives.re_training import RETrainingExample
            from systems.synapse.types import SynapseEvent, SynapseEventType

            bus = self._event_bus
            if bus is None:
                return

            example = RETrainingExample(
                source_system=SystemID.SIMULA,
                category="preventive_repair",
                instruction=(
                    f"Detect and repair fragile code pre-crash. "
                    f"Source: {result.source_ref}"
                ),
                input_context=(
                    f"fragility_score={result.fragility_score:.3f}, "
                    f"combined_risk={result.combined_risk:.3f}, "
                    f"checks={result.hit_counts}, "
                    f"pattern={result.pattern_id or 'none'}"
                ),
                output=f"preventive_action={action}",
                outcome_quality=0.8,  # Tentative - updated when repair verified
                reasoning_trace=(
                    f"Fragility checks: {result.hit_counts}. "
                    f"Pattern match: {result.pattern_id}. "
                    f"Incident correlated: {result.incident_match}. "
                    f"Combined risk: {result.combined_risk:.3f}. "
                    f"Action: {action}."
                ),
                constitutional_alignment=DriveAlignmentVector(
                    coherence=0.7, care=0.8, growth=0.6, honesty=0.9,
                ),
                cost_usd=Decimal("0.001"),
            )

            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="simula",
                data=example.model_dump(mode="json"),
            ))
        except Exception:
            pass  # Never block the audit pipeline

    async def _emit_audit_metrics(
        self,
        cycle: int,
        total_fragments: int,
        criticals: int,
        risks: int,
        below: int,
    ) -> None:
        """Send audit summary metrics to Benchmarks via Synapse."""
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            bus = self._event_bus
            if bus is None:
                return

            # Emit as BENCHMARK_RE_PROGRESS so Benchmarks can track preventive coverage
            if hasattr(SynapseEventType, "BENCHMARK_RE_PROGRESS"):
                await bus.emit(SynapseEvent(
                    event_type=SynapseEventType.BENCHMARK_RE_PROGRESS,
                    source_system="simula",
                    data={
                        "kpi_name": "simula.preventive_audit",
                        "cycle": cycle,
                        "total_fragments": total_fragments,
                        "criticals": criticals,
                        "risks": risks,
                        "below_threshold": below,
                        "audit_aggressiveness": self._audit_aggressiveness,
                    },
                ))
        except Exception:
            pass

    # ── Utility ───────────────────────────────────────────────────────────

    @property
    def _event_bus(self) -> Any:
        synapse = getattr(self._service, "_synapse", None)
        if synapse is None:
            return None
        return getattr(synapse, "_event_bus", None)
