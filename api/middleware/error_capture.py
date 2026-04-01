"""
EcodiaOS - API Error Capture Middleware

Intercepts HTTP errors, request validation failures, and unhandled exceptions
at the ASGI layer and feeds them into Thymos as Incidents.

Design notes:
- Never blocks the response path - Thymos reporting is fire-and-forget.
- Never modifies the response; client receives the original error as-is.
- Deduplication is handled by Thymos' built-in deduplicator.
- Thymos access is via app.state.thymos; if not yet initialised, errors are
  logged but silently dropped (startup window safety).
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import traceback
import uuid
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import Request, Response  # noqa: TC002 - used at runtime in dispatch signature
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from systems.thymos.types import Incident, IncidentClass, IncidentSeverity

if TYPE_CHECKING:
    from starlette.types import ASGIApp

    from systems.thymos.service import ThymosService

logger = structlog.get_logger().bind(system="api.error_capture")

# ─── Classification tables ────────────────────────────────────────

# (severity, incident_class) keyed by HTTP status code.
# Codes not listed fall through to _classify_status_range().
_STATUS_MAP: dict[int, tuple[IncidentSeverity, IncidentClass]] = {
    400: (IncidentSeverity.LOW, IncidentClass.PROTOCOL_DEGRADATION),
    401: (IncidentSeverity.INFO, IncidentClass.PROTOCOL_DEGRADATION),
    403: (IncidentSeverity.INFO, IncidentClass.PROTOCOL_DEGRADATION),
    404: (IncidentSeverity.LOW, IncidentClass.PROTOCOL_DEGRADATION),
    405: (IncidentSeverity.LOW, IncidentClass.PROTOCOL_DEGRADATION),
    422: (IncidentSeverity.LOW, IncidentClass.DEGRADATION),
    429: (IncidentSeverity.MEDIUM, IncidentClass.RESOURCE_EXHAUSTION),
    500: (IncidentSeverity.HIGH, IncidentClass.DEGRADATION),
    502: (IncidentSeverity.HIGH, IncidentClass.DEGRADATION),
    503: (IncidentSeverity.HIGH, IncidentClass.DEGRADATION),
    504: (IncidentSeverity.HIGH, IncidentClass.DEGRADATION),
}

# Statuses below this threshold are not reported (2xx / 3xx are healthy).
_REPORT_THRESHOLD = 400

# Statuses we intentionally skip - high-volume, low-signal noise.
_SKIP_STATUSES: frozenset[int] = frozenset()


def _classify_status_range(
    status: int,
) -> tuple[IncidentSeverity, IncidentClass]:
    """Fallback classification for unlisted status codes."""
    if 400 <= status < 500:
        return IncidentSeverity.LOW, IncidentClass.PROTOCOL_DEGRADATION
    if 500 <= status < 600:
        return IncidentSeverity.HIGH, IncidentClass.DEGRADATION
    # Should not reach here given _REPORT_THRESHOLD guard.
    return IncidentSeverity.INFO, IncidentClass.DEGRADATION


def _classify_status(
    status: int,
) -> tuple[IncidentSeverity, IncidentClass]:
    return _STATUS_MAP.get(status) or _classify_status_range(status)


def _get_root_app(request: Request) -> Any:
    """
    Access the root FastAPI app via the request scope.

    The request scope contains a reference to the app instance set by FastAPI.
    """
    return request.app


# ─── Fingerprinting ───────────────────────────────────────────────


def _fingerprint(endpoint: str, error_type: str) -> str:
    """
    Stable fingerprint for deduplication.

    Groups: same endpoint × same error class, regardless of request params.
    A 404 on /api/v1/foo and a 404 on /api/v1/bar are distinct incidents
    because the endpoint (path template) differs.
    """
    raw = f"api:{endpoint}:{error_type}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ─── Incident builders ────────────────────────────────────────────


def _build_incident(
    *,
    status: int,
    method: str,
    path: str,
    error_type: str,
    error_message: str,
    stack_trace: str | None = None,
    extra_context: dict[str, Any] | None = None,
    request_id: str = "",
) -> Incident:
    severity, incident_class = _classify_status(status)

    context: dict[str, Any] = {
        "http_method": method,
        "http_path": path,
        "http_status": status,
        "request_id": request_id,
    }
    if extra_context:
        context.update(extra_context)

    return Incident(
        incident_class=incident_class,
        severity=severity,
        fingerprint=_fingerprint(path, error_type),
        source_system="api",
        error_type=error_type,
        error_message=error_message[:500],
        stack_trace=stack_trace,
        context=context,
        user_visible=True,  # API errors always reach the client
    )


# ─── Middleware ───────────────────────────────────────────────────


class ErrorCaptureMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that observes every request/response cycle.

    On any error response (4xx/5xx) or unhandled exception, it constructs a
    Thymos Incident and fires it asynchronously without touching the response.

    Thymos is fetched lazily from app.state on each call so that the
    middleware can be registered before the startup lifespan completes.
    """

    def __init__(self, app: ASGIApp | Any) -> None:
        """Initialize the middleware."""
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
        path = request.url.path
        method = request.method
        start = time.monotonic()

        log = logger.bind(
            request_id=request_id,
            path=path,
            method=method,
        )
        log.debug("request_start")

        # ── Fast exit: only inspect paths that matter ──────────────
        # Health probe endpoints emit a lot of traffic; skip them.
        if path in ("/health", "/metrics"):
            return await call_next(request)

        response: Response | None = None
        captured_exc: BaseException | None = None

        try:
            response = await call_next(request)
        except RequestValidationError as exc:
            captured_exc = exc
            response = JSONResponse(
                status_code=422,
                content={"detail": exc.errors()},
            )
            self._fire_validation_incident(exc, method, path, request_id, request)
        except TimeoutError as exc:
            captured_exc = exc
            response = JSONResponse(
                status_code=504,
                content={"detail": "Gateway timeout"},
            )
            self._fire_timeout_incident(exc, method, path, request_id, request)
        except Exception as exc:
            captured_exc = exc
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )
            self._fire_unhandled_incident(exc, method, path, request_id, request)

        latency_ms = (time.monotonic() - start) * 1000
        status = response.status_code

        if captured_exc is not None:
            log.warning(
                "request_error",
                status_code=status,
                error=str(captured_exc)[:200],
                latency_ms=f"{latency_ms:.1f}",
            )
        elif status >= _REPORT_THRESHOLD and status not in _SKIP_STATUSES:
            log.info(
                "request_end",
                status_code=status,
                latency_ms=f"{latency_ms:.1f}",
            )
            # HTTPException raised inside routers - response is already formed,
            # we just observe it here.
            self._fire_http_incident(status, method, path, request_id, request)
        else:
            log.debug(
                "request_end",
                status_code=status,
                latency_ms=f"{latency_ms:.1f}",
            )

        return response

    # ─── Incident reporters ───────────────────────────────────────

    def _fire_http_incident(
        self,
        status: int,
        method: str,
        path: str,
        request_id: str,
        request: Request,
    ) -> None:
        """HTTPException already returned a response - observe the status only."""
        incident = _build_incident(
            status=status,
            method=method,
            path=path,
            error_type=f"HTTPException{status}",
            error_message=f"HTTP {status} on {method} {path}",
            request_id=request_id,
        )
        self._report(incident, request)

    def _fire_validation_incident(
        self,
        exc: RequestValidationError,
        method: str,
        path: str,
        request_id: str,
        request: Request,
    ) -> None:
        """Pydantic/FastAPI request body or query param validation failure."""
        errors_summary = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()[:3]
        )
        incident = _build_incident(
            status=422,
            method=method,
            path=path,
            error_type="RequestValidationError",
            error_message=f"Validation failed on {method} {path}: {errors_summary}",
            extra_context={"validation_errors": exc.errors()[:10]},
            request_id=request_id,
        )
        self._report(incident, request)

    def _fire_timeout_incident(
        self,
        exc: TimeoutError,
        method: str,
        path: str,
        request_id: str,
        request: Request,
    ) -> None:
        severity = IncidentSeverity.HIGH
        incident = Incident(
            incident_class=IncidentClass.DEGRADATION,
            severity=severity,
            fingerprint=_fingerprint(path, "TimeoutError"),
            source_system="api",
            error_type="TimeoutError",
            error_message=f"Request timed out: {method} {path}",
            stack_trace=traceback.format_exc()[:2000],
            context={
                "http_method": method,
                "http_path": path,
                "http_status": 504,
                "request_id": request_id,
            },
            user_visible=True,
        )
        self._report(incident, request)

    def _fire_unhandled_incident(
        self,
        exc: BaseException,
        method: str,
        path: str,
        request_id: str,
        request: Request,
    ) -> None:
        """Unhandled exception - CRITICAL, includes full stack trace."""
        incident = Incident(
            incident_class=IncidentClass.CRASH,
            severity=IncidentSeverity.CRITICAL,
            fingerprint=_fingerprint(path, type(exc).__name__),
            source_system="api",
            error_type=type(exc).__name__,
            error_message=f"Unhandled {type(exc).__name__} on {method} {path}: {exc!s}"[:500],
            stack_trace=traceback.format_exc()[:2000],
            context={
                "http_method": method,
                "http_path": path,
                "http_status": 500,
                "request_id": request_id,
            },
            user_visible=True,
        )
        self._report(incident, request)

    # ─── Async fire-and-forget ─────────────────────────────────────

    def _report(self, incident: Incident, request: Request) -> None:
        """
        Non-blocking report to Thymos and LogAnalyzer.

        Fetches Thymos and LogAnalyzer from app.state at call time so we
        handle the startup window gracefully (middleware is registered
        before lifespan completes, so app.state may not exist yet).
        """
        root_app = _get_root_app(request)
        thymos: ThymosService | None = getattr(root_app.state, "thymos", None)  # type: ignore[attr-defined]
        if thymos is None:
            logger.debug(
                "thymos_not_ready",
                incident_fingerprint=incident.fingerprint,
                incident_class=incident.incident_class,
            )
            return

        asyncio.create_task(
            _report_safe(thymos, incident, root_app),
            name=f"thymos_api_err_{incident.fingerprint}",
        )


async def _report_safe(
    thymos: ThymosService, incident: Incident, app: Any,
) -> None:
    """Fire-and-forget report to Thymos and LogAnalyzer."""
    try:
        await thymos.on_incident(incident)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "thymos_report_failed",
            error=str(exc),
            incident_fingerprint=incident.fingerprint,
        )

    # Also feed to LogAnalyzer for cascade detection
    try:
        analyzer = getattr(app.state, "log_analyzer", None)
        if analyzer is not None:
            await analyzer.ingest_log(
                level="error",
                system="api",
                message=incident.error_message,
                error_type=incident.error_type,
                incident_class=incident.incident_class.value,
                incident_id=incident.id,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("log_analyzer_ingest_failed", error=str(exc))
