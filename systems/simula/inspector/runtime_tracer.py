"""
EcodiaOS — Inspector Phase 2: Runtime Tracer

Instruments Python target modules/functions for:
  1. Function-level tracing  — call / return / exception events
  2. Basic-block signals     — coarse control-flow coverage via line-granularity
  3. Branch observations     — taken/not-taken signals from conditional lines

Instrumentation strategy
------------------------
Python 3.12+ exposes ``sys.monitoring`` (PEP 669), which is the preferred
backend: it's selective (per-tool, per-event), low-overhead, and does not
require CPython internals.  On older runtimes we fall back to ``sys.settrace``.

For subprocess targets (native/compiled code) the tracer records crash signals
via subprocess return codes and ``signal.Signals`` enumeration; actual
assembly-level BB tracing is out of scope for this Python runtime layer — the
eBPF observer sidecar (observer/observer.py) handles that at the kernel level.

Iron Rules
----------
- The tracer NEVER modifies the target code or writes to the workspace.
- All collected events are purely in-memory until explicitly serialised.
- Tracing overhead must not silently mask bugs — the tracer catches its own
  exceptions and degrades gracefully rather than interfering with the target.
- Subprocess targets are executed with a hard timeout (configurable).
"""

from __future__ import annotations

import asyncio
import inspect
import signal
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from systems.simula.inspector.runtime_types import (
    BasicBlockTrace,
    ControlFlowTrace,
    FaultClass,
    FaultObservation,
    RunCategory,
    TraceDataset,
    TraceEvent,
    TraceEventKind,
)

if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Generator

logger = structlog.get_logger().bind(system="simula.inspector.runtime_tracer")

# Maximum events to buffer before dropping (prevent OOM on tight loops).
_MAX_EVENTS: int = 50_000

# Default subprocess timeout seconds.
_DEFAULT_TIMEOUT_S: float = 30.0

# Python 3.12+ monitoring API availability.
_HAS_SYS_MONITORING: bool = hasattr(sys, "monitoring")

# Tool ID for sys.monitoring (must be unique per tool in a process; 1–5 are reserved).
_MONITORING_TOOL_ID: int = 6


# ── Per-run tracing state ─────────────────────────────────────────────────────


class _RunState:
    """
    Mutable tracing state accumulated during a single instrumented run.

    Deliberately not a Pydantic model — it's a hot-path accumulator.
    Conversion to immutable types happens in _finalise().
    """

    __slots__ = (
        "run_id",
        "run_category",
        "events",
        "call_sequence",
        "functions_visited",
        "exception_exit_functions",
        "bb_hits",
        "branch_total",
        "branch_taken_count",
        "max_stack_depth",
        "start_ns",
        "fault_observations",
    )

    def __init__(self, run_id: str, run_category: RunCategory) -> None:
        self.run_id = run_id
        self.run_category = run_category
        self.events: list[TraceEvent] = []
        self.call_sequence: list[tuple[str, str]] = []
        self.functions_visited: set[str] = set()
        self.exception_exit_functions: list[str] = []
        self.bb_hits: dict[str, int] = {}
        self.branch_total: int = 0
        self.branch_taken_count: int = 0
        self.max_stack_depth: int = 0
        self.start_ns: int = time.monotonic_ns()
        self.fault_observations: list[FaultObservation] = []

    def record(self, event: TraceEvent) -> None:
        if len(self.events) < _MAX_EVENTS:
            self.events.append(event)


def _finalise(state: _RunState) -> tuple[ControlFlowTrace, list[FaultObservation]]:
    """Convert mutable _RunState into immutable dataclasses."""
    elapsed_ns = time.monotonic_ns() - state.start_ns

    # Basic-block aggregation
    sum(state.bb_hits.values())
    unique_blocks = len(state.bb_hits)
    branch_diversity = 0.0
    if state.branch_total > 0:
        taken_frac = state.branch_taken_count / state.branch_total
        # diversity = how far from "always taken" or "never taken" (0 = pure, 0.5 = max)
        branch_diversity = min(taken_frac, 1.0 - taken_frac) * 2.0

    bb_trace = BasicBlockTrace(
        run_id=state.run_id,
        hits=dict(state.bb_hits),
        unique_blocks=unique_blocks,
        branch_observations=state.branch_total,
        branch_diversity=branch_diversity,
    )

    trace = ControlFlowTrace(
        run_id=state.run_id,
        run_category=state.run_category,
        call_sequence=list(state.call_sequence),
        functions_visited=sorted(state.functions_visited),
        exception_exit_functions=list(dict.fromkeys(state.exception_exit_functions)),
        bb_trace=bb_trace,
        max_stack_depth=state.max_stack_depth,
        total_events=len(state.events),
        duration_ms=elapsed_ns / 1_000_000,
    )

    return trace, list(state.fault_observations)


# ── sys.settrace backend ──────────────────────────────────────────────────────


def _make_settrace_handler(
    state: _RunState,
    scope_prefix: str,
) -> Callable:
    """
    Return a ``sys.settrace``-compatible trace function that filters to the
    target scope (files whose path contains *scope_prefix*).
    """
    seq = [0]

    def local_trace(frame: types.FrameType, event: str, arg: Any) -> Callable | None:
        """Line-level trace (returned from global_trace for in-scope frames)."""
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name

        bb_id = f"{filename}:{lineno}"
        state.bb_hits[bb_id] = state.bb_hits.get(bb_id, 0) + 1

        now = time.monotonic_ns()
        seq[0] += 1

        if event == "line":
            # Each distinct line-entry within a function body is a pseudo-BB.
            te = TraceEvent(
                run_id=state.run_id,
                seq=seq[0],
                kind=TraceEventKind.BASIC_BLOCK,
                timestamp_ns=now,
                file=filename,
                line=lineno,
                func_name=func_name,
                bb_id=bb_id,
                stack_depth=len(inspect.stack(context=0)),
            )
            state.record(te)
        elif event == "exception":
            exc_type, exc_val, _ = arg
            exc_name = exc_type.__name__ if exc_type else "UnknownException"
            exc_msg = str(exc_val)[:512] if exc_val else ""
            state.exception_exit_functions.append(func_name)
            te = TraceEvent(
                run_id=state.run_id,
                seq=seq[0],
                kind=TraceEventKind.EXCEPTION,
                timestamp_ns=now,
                file=filename,
                line=lineno,
                func_name=func_name,
                exception_type=exc_name,
                exception_message=exc_msg,
                stack_depth=len(inspect.stack(context=0)),
            )
            state.record(te)

        return local_trace

    def global_trace(frame: types.FrameType, event: str, arg: Any) -> Callable | None:
        """Global trace — called for every function entry/exit."""
        filename = frame.f_code.co_filename

        # Filter to target scope only
        if scope_prefix and scope_prefix not in filename:
            return None

        func_name = frame.f_code.co_name
        lineno = frame.f_lineno
        now = time.monotonic_ns()
        seq[0] += 1

        depth = len(inspect.stack(context=0))
        if depth > state.max_stack_depth:
            state.max_stack_depth = depth

        if event == "call":
            caller = ""
            caller_frame = frame.f_back
            if caller_frame:
                caller = caller_frame.f_code.co_name
            state.functions_visited.add(func_name)
            state.call_sequence.append((caller, func_name))
            te = TraceEvent(
                run_id=state.run_id,
                seq=seq[0],
                kind=TraceEventKind.CALL,
                timestamp_ns=now,
                file=filename,
                line=lineno,
                func_name=func_name,
                caller=caller,
                callee=func_name,
                stack_depth=depth,
            )
            state.record(te)
            return local_trace  # enable line-level tracing for this frame

        elif event == "return":
            ret_type = type(arg).__name__ if arg is not None else "None"
            te = TraceEvent(
                run_id=state.run_id,
                seq=seq[0],
                kind=TraceEventKind.RETURN,
                timestamp_ns=now,
                file=filename,
                line=lineno,
                func_name=func_name,
                return_value_type=ret_type,
                stack_depth=depth,
            )
            state.record(te)

        return None

    return global_trace


# ── sys.monitoring backend (Python 3.12+) ────────────────────────────────────


def _install_monitoring_hooks(state: _RunState, scope_prefix: str) -> None:
    """
    Register ``sys.monitoring`` event hooks for the target scope.

    Hooks are registered globally for the process; the scope_prefix filter
    is applied inside each hook to avoid overhead on unrelated frames.

    This is only called when ``_HAS_SYS_MONITORING`` is True.
    """
    mon = sys.monitoring  # type: ignore[attr-defined]
    tool_id = _MONITORING_TOOL_ID
    seq = [0]

    try:
        mon.use_tool_id(tool_id, "simula.inspector.runtime_tracer")
    except ValueError:
        # Already registered by a previous (non-finalised) run; free and re-register.
        mon.free_tool_id(tool_id)
        mon.use_tool_id(tool_id, "simula.inspector.runtime_tracer")

    def on_call(code: types.CodeType, instruction_offset: int) -> object:
        if scope_prefix and scope_prefix not in code.co_filename:
            return mon.DISABLE
        func_name = code.co_name
        now = time.monotonic_ns()
        seq[0] += 1
        state.functions_visited.add(func_name)
        # Caller info not directly available in monitoring API — leave empty.
        state.call_sequence.append(("", func_name))
        te = TraceEvent(
            run_id=state.run_id,
            seq=seq[0],
            kind=TraceEventKind.CALL,
            timestamp_ns=now,
            file=code.co_filename,
            func_name=func_name,
            callee=func_name,
            stack_depth=0,  # monitoring doesn't expose depth directly
        )
        state.record(te)
        return None

    def on_return(
        code: types.CodeType,
        instruction_offset: int,
        retval: object,
    ) -> object:
        if scope_prefix and scope_prefix not in code.co_filename:
            return mon.DISABLE
        now = time.monotonic_ns()
        seq[0] += 1
        ret_type = type(retval).__name__ if retval is not None else "None"
        te = TraceEvent(
            run_id=state.run_id,
            seq=seq[0],
            kind=TraceEventKind.RETURN,
            timestamp_ns=now,
            file=code.co_filename,
            func_name=code.co_name,
            return_value_type=ret_type,
            stack_depth=0,
        )
        state.record(te)
        return None

    def on_exception(
        code: types.CodeType,
        instruction_offset: int,
        exc: BaseException,
    ) -> object:
        if scope_prefix and scope_prefix not in code.co_filename:
            return mon.DISABLE
        exc_name = type(exc).__name__
        exc_msg = str(exc)[:512]
        now = time.monotonic_ns()
        seq[0] += 1
        state.exception_exit_functions.append(code.co_name)
        te = TraceEvent(
            run_id=state.run_id,
            seq=seq[0],
            kind=TraceEventKind.EXCEPTION,
            timestamp_ns=now,
            file=code.co_filename,
            func_name=code.co_name,
            exception_type=exc_name,
            exception_message=exc_msg,
            stack_depth=0,
        )
        state.record(te)
        return None

    def on_line(
        code: types.CodeType,
        line_number: int,
    ) -> object:
        if scope_prefix and scope_prefix not in code.co_filename:
            return mon.DISABLE
        bb_id = f"{code.co_filename}:{line_number}"
        state.bb_hits[bb_id] = state.bb_hits.get(bb_id, 0) + 1
        return None

    mon.set_events(
        tool_id,
        mon.events.PY_START  # call
        | mon.events.PY_RETURN
        | mon.events.RAISE
        | mon.events.LINE,
    )
    mon.register_callback(tool_id, mon.events.PY_START, on_call)
    mon.register_callback(tool_id, mon.events.PY_RETURN, on_return)
    mon.register_callback(tool_id, mon.events.RAISE, on_exception)
    mon.register_callback(tool_id, mon.events.LINE, on_line)


def _uninstall_monitoring_hooks() -> None:
    """Remove all sys.monitoring hooks registered for this tool."""
    if not _HAS_SYS_MONITORING:
        return
    mon = sys.monitoring  # type: ignore[attr-defined]
    try:
        mon.free_tool_id(_MONITORING_TOOL_ID)
    except Exception:  # noqa: BLE001
        pass


# ── Main tracer context manager ───────────────────────────────────────────────


@contextmanager
def _tracing_context(
    run_id: str,
    run_category: RunCategory,
    scope_prefix: str,
    use_monitoring: bool,
) -> Generator[_RunState, None, None]:
    """
    Context manager that activates trace hooks, yields the mutable state, then
    deactivates hooks on exit (normal or exception).
    """
    state = _RunState(run_id=run_id, run_category=run_category)

    if use_monitoring and _HAS_SYS_MONITORING:
        _install_monitoring_hooks(state, scope_prefix)
    else:
        old_trace = sys.gettrace()
        sys.settrace(_make_settrace_handler(state, scope_prefix))

    try:
        yield state
    finally:
        if use_monitoring and _HAS_SYS_MONITORING:
            _uninstall_monitoring_hooks()
        else:
            sys.settrace(old_trace)  # type: ignore[arg-type]


# ── RuntimeTracer public API ──────────────────────────────────────────────────


class RuntimeTracer:
    """
    Instruments callable Python targets (functions, test suites, or subprocess
    commands) and returns a ControlFlowTrace + any FaultObservations.

    Usage — in-process callable::

        tracer = RuntimeTracer(scope_prefix="mypackage/")
        trace, faults = await tracer.trace_callable(
            run_id="run-001",
            run_category=RunCategory.NORMAL,
            target=my_function,
            args=(...,),
            kwargs={...},
        )

    Usage — subprocess::

        trace, faults = await tracer.trace_subprocess(
            run_id="run-002",
            run_category=RunCategory.FAILURE,
            cmd=["python", "-m", "mymodule", "--bad-input", "fuzz"],
            cwd=Path("/path/to/workspace"),
        )
    """

    def __init__(
        self,
        scope_prefix: str = "",
        use_monitoring: bool = True,
        subprocess_timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        """
        Args:
            scope_prefix:        Only trace frames whose filename contains this
                                 string.  Empty string traces everything (expensive).
            use_monitoring:      Prefer sys.monitoring when available (Python 3.12+).
            subprocess_timeout_s: Hard timeout for traced subprocess invocations.
        """
        self._scope_prefix = scope_prefix
        self._use_monitoring = use_monitoring
        self._subprocess_timeout_s = subprocess_timeout_s
        self._log = logger

    # ── In-process callable tracing ───────────────────────────────────────────

    async def trace_callable(
        self,
        run_id: str,
        run_category: RunCategory,
        target: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> tuple[ControlFlowTrace, list[FaultObservation]]:
        """
        Execute *target(*args, **kwargs)* under instrumentation.

        The callable is run in the current thread (on the event loop's executor
        for async-safe blocking I/O).  Exceptions raised by the target are
        caught, recorded as fault observations, and re-raised after finalisation
        so the caller can inspect them.
        """
        kwargs = kwargs or {}
        log = self._log.bind(run_id=run_id, target=getattr(target, "__qualname__", str(target)))

        caught_exc: BaseException | None = None

        with _tracing_context(
            run_id=run_id,
            run_category=run_category,
            scope_prefix=self._scope_prefix,
            use_monitoring=self._use_monitoring,
        ) as state:
            try:
                result = target(*args, **kwargs)
                # Handle awaitables transparently
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:  # noqa: BLE001
                caught_exc = exc
                log.debug("callable_raised", exc_type=type(exc).__name__, exc=str(exc)[:256])

                # Build a fault observation from the exception
                tb_lines = traceback.format_tb(exc.__traceback__)
                fault = _classify_python_exception(
                    run_id=run_id,
                    exc=exc,
                    stack_trace=tb_lines,
                    call_sequence=state.call_sequence,
                )
                state.fault_observations.append(fault)

                # Update run category if it looks like a crash
                if isinstance(exc, (MemoryError, SystemError, RecursionError)):
                    state.run_category = RunCategory.CRASH

        trace, faults = _finalise(state)
        log.debug(
            "trace_callable_complete",
            events=trace.total_events,
            funcs=len(trace.functions_visited),
            faults=len(faults),
            duration_ms=trace.duration_ms,
        )

        # Re-raise so callers can optionally handle it
        if caught_exc is not None and not isinstance(caught_exc, Exception):
            raise caught_exc  # Re-raise BaseException subclasses (KeyboardInterrupt etc.)

        return trace, faults

    # ── Subprocess tracing ────────────────────────────────────────────────────

    async def trace_subprocess(
        self,
        run_id: str,
        run_category: RunCategory,
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[ControlFlowTrace, list[FaultObservation]]:
        """
        Run *cmd* as a subprocess and observe its exit status and stderr output
        for crash signals.

        Because we cannot install Python-level trace hooks inside a subprocess,
        the ControlFlowTrace will have an empty call_sequence — only the crash
        signal / exit code is recorded.  Callers that need deeper visibility
        should run the target in-process (trace_callable) or use the eBPF
        observer sidecar for native targets.

        The fault classification uses signal number, exit code, and any
        exception-like patterns found in stderr.
        """
        log = self._log.bind(run_id=run_id, cmd=" ".join(cmd[:4]))
        state = _RunState(run_id=run_id, run_category=run_category)

        stdout_data = b""
        stderr_data = b""
        returncode: int = 0

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._subprocess_timeout_s,
                )
                returncode = proc.returncode or 0
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                # Use SIGKILL if available (POSIX), otherwise use -1 to indicate timeout kill
                returncode = -getattr(signal, 'SIGKILL', 1)
                log.warning("subprocess_timeout", timeout_s=self._subprocess_timeout_s)
        except OSError as exc:
            log.warning("subprocess_launch_failed", error=str(exc))
            returncode = -1

        # Record a SIGNAL event if the process was killed by a signal
        if returncode < 0:
            sig_num = abs(returncode)
            sig_event = TraceEvent(
                run_id=run_id,
                seq=1,
                kind=TraceEventKind.SIGNAL,
                timestamp_ns=time.monotonic_ns(),
                signal_number=sig_num,
                fault_class=_signal_to_fault_class(sig_num),
            )
            state.record(sig_event)
            state.run_category = RunCategory.CRASH

        # Parse stderr for exception/panic patterns
        stderr_text = stderr_data.decode("utf-8", errors="replace")
        fault = _classify_subprocess_output(
            run_id=run_id,
            returncode=returncode,
            stderr=stderr_text,
            call_sequence=[],
        )
        if fault is not None:
            state.fault_observations.append(fault)
            if run_category == RunCategory.NORMAL and fault.fault_class != FaultClass.UNKNOWN:
                state.run_category = RunCategory.FAILURE

        trace, faults = _finalise(state)
        log.debug(
            "trace_subprocess_complete",
            returncode=returncode,
            faults=len(faults),
            duration_ms=trace.duration_ms,
        )
        return trace, faults

    # ── Batch tracing ─────────────────────────────────────────────────────────

    async def trace_many_callables(
        self,
        runs: list[dict],
    ) -> TraceDataset:
        """
        Trace multiple callable runs and collect them into a TraceDataset.

        Each item in *runs* must be a dict with keys:
            run_id:       str (auto-generated if absent)
            run_category: RunCategory
            target:       Callable
            args:         tuple (optional)
            kwargs:       dict (optional)
            target_id:    str (used for dataset.target_id if all items share one)

        Returns a fully populated TraceDataset.
        """
        target_ids = {r.get("target_id", "") for r in runs}
        dataset_target_id = next(iter(target_ids)) if len(target_ids) == 1 else "multi"
        dataset = TraceDataset(target_id=dataset_target_id)

        for run in runs:
            rid = run.get("run_id") or str(uuid.uuid4())
            category = run.get("run_category", RunCategory.NORMAL)
            target = run["target"]
            args = run.get("args", ())
            kwargs = run.get("kwargs", {})

            try:
                trace, faults = await self.trace_callable(
                    run_id=rid,
                    run_category=category,
                    target=target,
                    args=args,
                    kwargs=kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                self._log.warning("batch_run_failed", run_id=rid, error=str(exc))
                # Still register a minimal failure trace
                trace = ControlFlowTrace(
                    run_id=rid,
                    run_category=RunCategory.CRASH,
                    total_events=0,
                )
                faults = []

            dataset.add_trace(trace)
            for fault in faults:
                dataset.add_fault(fault)

        return dataset

    async def trace_many_subprocesses(
        self,
        runs: list[dict],
    ) -> TraceDataset:
        """
        Trace multiple subprocess runs.

        Each item in *runs* must be a dict with keys:
            run_id:       str (auto-generated if absent)
            run_category: RunCategory
            cmd:          list[str]
            cwd:          Path (optional)
            env:          dict (optional)
            target_id:    str (optional)
        """
        target_ids = {r.get("target_id", "") for r in runs}
        dataset_target_id = next(iter(target_ids)) if len(target_ids) == 1 else "multi"
        dataset = TraceDataset(target_id=dataset_target_id)

        for run in runs:
            rid = run.get("run_id") or str(uuid.uuid4())
            category = run.get("run_category", RunCategory.NORMAL)
            cmd = run["cmd"]
            cwd = run.get("cwd")
            env = run.get("env")

            try:
                trace, faults = await self.trace_subprocess(
                    run_id=rid,
                    run_category=category,
                    cmd=cmd,
                    cwd=Path(cwd) if cwd else None,
                    env=env,
                )
            except Exception as exc:  # noqa: BLE001
                self._log.warning("batch_subprocess_failed", run_id=rid, error=str(exc))
                trace = ControlFlowTrace(
                    run_id=rid,
                    run_category=RunCategory.CRASH,
                    total_events=0,
                )
                faults = []

            dataset.add_trace(trace)
            for fault in faults:
                dataset.add_fault(fault)

        return dataset


# ── Classification helpers ────────────────────────────────────────────────────


def _signal_to_fault_class(sig_num: int) -> FaultClass:
    """Map a POSIX signal number to a FaultClass label."""
    _MAP: dict[int, FaultClass] = {
        signal.SIGSEGV: FaultClass.SIGNAL_SEGV,
        signal.SIGABRT: FaultClass.SIGNAL_ABORT,
        signal.SIGFPE:  FaultClass.SIGNAL_FPE,
    }
    # Add SIGBUS if available (not on Windows)
    if hasattr(signal, 'SIGBUS'):
        _MAP[signal.SIGBUS] = FaultClass.SIGNAL_BUS

    return _MAP.get(sig_num, FaultClass.SIGNAL_OTHER)


def _classify_python_exception(
    run_id: str,
    exc: BaseException,
    stack_trace: list[str],
    call_sequence: list[tuple[str, str]],
) -> FaultObservation:
    """
    Map a Python exception to a FaultObservation with a preliminary FaultClass.

    The classification is heuristic — it uses the exception type name and
    message, not dynamic memory analysis.
    """
    exc_type_name = type(exc).__name__
    exc_msg = str(exc)[:512]

    fault_class = FaultClass.UNKNOWN
    confidence = 0.5

    # Index / buffer / sequence errors → OOB
    if exc_type_name in ("IndexError", "BufferError", "OverflowError"):
        fault_class = FaultClass.OOB
        confidence = 0.8

    # Attribute errors on None / deleted objects → dangling reference heuristic
    elif exc_type_name == "AttributeError" and (
        "NoneType" in exc_msg or "deleted" in exc_msg.lower() or "object has no attribute" in exc_msg
    ):
        fault_class = FaultClass.UAF
        confidence = 0.55

    # TypeError — covers type confusion patterns
    elif exc_type_name == "TypeError":
        fault_class = FaultClass.TYPE
        confidence = 0.7

    # RuntimeError / StopIteration from exhausted generators → lifetime
    elif exc_type_name in ("RuntimeError", "StopIteration") and (
        "generator" in exc_msg.lower() or "deque" in exc_msg.lower()
    ):
        fault_class = FaultClass.LIFETIME
        confidence = 0.6

    # AssertionError → logic invariant
    elif exc_type_name == "AssertionError":
        fault_class = FaultClass.LOGIC
        confidence = 0.9

    # Memory / recursion → may be OOB-induced
    elif exc_type_name in ("MemoryError", "RecursionError"):
        fault_class = FaultClass.OOB
        confidence = 0.4

    # Generic unhandled exception
    elif exc_type_name not in ("KeyboardInterrupt", "SystemExit"):
        fault_class = FaultClass.UNHANDLED_EXC
        confidence = 0.7

    # Last structured function = second-to-last caller in the call sequence
    last_structured = ""
    if len(call_sequence) >= 2:
        last_structured = call_sequence[-2][1]  # callee of penultimate call

    divergence_depth = len(call_sequence)

    return FaultObservation(
        run_id=run_id,
        fault_class=fault_class,
        confidence=confidence,
        fault_at_func=call_sequence[-1][1] if call_sequence else "",
        exception_type=exc_type_name,
        exception_message=exc_msg,
        stack_trace=stack_trace[-10:],  # keep last 10 frames
        last_structured_func=last_structured,
        divergence_depth=divergence_depth,
    )


_STDERR_PATTERNS: list[tuple[str, FaultClass, float]] = [
    # OOB / segfault patterns in C extensions / ctypes
    ("segmentation fault", FaultClass.SIGNAL_SEGV, 0.9),
    ("sigsegv", FaultClass.SIGNAL_SEGV, 0.9),
    ("sigabrt", FaultClass.SIGNAL_ABORT, 0.9),
    ("abort", FaultClass.SIGNAL_ABORT, 0.7),
    ("stack smashing detected", FaultClass.OOB, 0.95),
    ("buffer overflow", FaultClass.OOB, 0.85),
    ("heap-buffer-overflow", FaultClass.OOB, 0.95),  # ASan
    ("stack-buffer-overflow", FaultClass.OOB, 0.95),  # ASan
    ("heap-use-after-free", FaultClass.UAF, 0.95),   # ASan
    ("use-after-free", FaultClass.UAF, 0.9),
    ("use-after-poison", FaultClass.UAF, 0.9),
    ("invalid free", FaultClass.UAF, 0.85),
    ("double free", FaultClass.UAF, 0.9),
    ("memory error", FaultClass.OOB, 0.6),
    ("type error", FaultClass.TYPE, 0.55),
    ("traceback", FaultClass.UNHANDLED_EXC, 0.5),
    ("panic", FaultClass.LOGIC, 0.6),
    ("assertion failed", FaultClass.LOGIC, 0.85),
    ("assert failed", FaultClass.LOGIC, 0.85),
]


def _classify_subprocess_output(
    run_id: str,
    returncode: int,
    stderr: str,
    call_sequence: list[tuple[str, str]],
) -> FaultObservation | None:
    """
    Classify a subprocess exit by return code and stderr content.

    Returns None for clean exits (returncode 0, no fault patterns in stderr).
    """
    stderr_lower = stderr.lower()

    fault_class = FaultClass.UNKNOWN
    confidence = 0.0

    # Signal-killed
    if returncode < 0:
        fault_class = _signal_to_fault_class(abs(returncode))
        confidence = 0.9
    else:
        # Scan stderr for known fault patterns
        for pattern, fc, conf in _STDERR_PATTERNS:
            if pattern in stderr_lower and conf > confidence:
                fault_class = fc
                confidence = conf

    # No evidence of fault
    if returncode == 0 and confidence == 0.0:
        return None

    # Extract the last stack-trace-like lines from stderr (last 15 non-empty lines)
    stderr_lines = [l for l in stderr.splitlines() if l.strip()][-15:]

    # Synthesise a minimal observation
    return FaultObservation(
        run_id=run_id,
        fault_class=fault_class,
        confidence=confidence,
        exception_message=f"subprocess exit code {returncode}",
        stack_trace=stderr_lines,
        last_structured_func="(subprocess)",
        divergence_depth=0,
    )
