"""
EcodiaOS - Inspector Dynamic Taint Injector

Actively maps live data-flow paths by firing uniquely-tagged HTTP requests
into a running container cluster and reading back which services the token
propagated through via the TaintFlowLinker.

How it works
------------
1. ``generate_taint_token()`` mints a cryptographically unique marker string
   of the form ``TAINT=<hex8>;`` (short enough to ride inside any field).
2. ``inject_and_trace(target_url, surface)`` builds an HTTP request whose
   shape matches the surface metadata (method, route, body vs query param vs
   header), fires it with a short timeout, then waits briefly for the eBPF
   ring buffers to flush and the TaintFlowLinker to process the observations.
3. ``chain_for_token`` on the linker returns the ordered list of ``TaintEdge``
   objects representing each service-to-service hop the token traversed.

The resulting ``list[TaintEdge]`` is attached to the ``AttackSurface`` as
``verified_data_path`` so the VulnerabilityProver knows exactly which backend
sinks a public entry point connects to.

Iron Rules
----------
- All requests use a 5-second timeout - a hanging server must not stall
  the pipeline.
- HTTP error responses (4xx / 5xx) are silently accepted; we care about
  data propagation, not application correctness.
- The injector NEVER executes code on the target; it only sends HTTP.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING

import httpx
import structlog

if TYPE_CHECKING:
    from systems.simula.inspector.taint_flow_linker import TaintEdge, TaintFlowLinker
    from systems.simula.inspector.types import AttackSurface

logger = structlog.get_logger().bind(system="simula.inspector.injector")

# How long to wait after firing the request for eBPF events to propagate.
_EBPF_FLUSH_WAIT_S: float = 0.5

# HTTP request timeout - kept short so a dead container doesn't block the pipeline.
_REQUEST_TIMEOUT_S: float = 5.0

# Header name used when injecting via custom header.
_TAINT_HEADER: str = "X-Taint-Token"


class DynamicTaintInjector:
    """
    Fires marked HTTP requests into a live cluster and reads back the
    resulting token propagation chain from the TaintFlowLinker.

    Args:
        client:  Shared ``httpx.AsyncClient`` (caller manages lifecycle).
        linker:  ``TaintFlowLinker`` instance already wired to the eBPF
                 collector for this cluster.
    """

    def __init__(self, client: httpx.AsyncClient, linker: TaintFlowLinker) -> None:
        self._client = client
        self._linker = linker
        self._log = logger

    # ── Token generation ─────────────────────────────────────────────────────

    def generate_taint_token(self) -> str:
        """Return a unique taint marker string: ``TAINT=<8 hex chars>;``."""
        return f"TAINT={uuid.uuid4().hex[:8]};"

    # ── Main entry point ─────────────────────────────────────────────────────

    async def inject_and_trace(
        self,
        target_url: str,
        surface: AttackSurface,
    ) -> list[TaintEdge]:
        """
        Inject a unique taint token into ``surface`` at ``target_url`` and
        return the ordered list of ``TaintEdge`` hops the token traversed.

        Steps
        -----
        1. Mint a fresh token.
        2. Build and fire an HTTP request shaped to match the surface.
        3. Wait ``_EBPF_FLUSH_WAIT_S`` seconds for eBPF observations to arrive.
        4. Ask the linker for the token's propagation chain.
        5. Return the chain (empty list if no hops were observed).
        """
        token = self.generate_taint_token()
        log = self._log.bind(
            token=token,
            surface_type=surface.surface_type,
            entry_point=surface.entry_point,
            target_url=target_url,
        )

        url, method, kwargs = self._build_request(target_url, surface, token)

        log.debug("taint_injection_firing", url=url, method=method)
        try:
            await self._client.request(
                method,
                url,
                timeout=_REQUEST_TIMEOUT_S,
                **kwargs,
            )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            # Network errors are expected when targeting hardened services -
            # the token may still have propagated before the error occurred.
            log.debug("taint_injection_network_error", error=str(exc))

        # Give the eBPF ring buffer time to flush and the linker to process.
        await asyncio.sleep(_EBPF_FLUSH_WAIT_S)

        chain = self._linker.chain_for_token(token)
        log.info(
            "taint_injection_traced",
            hop_count=len(chain),
            hops=[f"{e.src_service} → {e.dst_service}" for e in chain],
        )
        return chain

    # ── Request builder ──────────────────────────────────────────────────────

    def _build_request(
        self,
        target_url: str,
        surface: AttackSurface,
        token: str,
    ) -> tuple[str, str, dict]:
        """
        Construct the (url, method, kwargs) triple for ``httpx.AsyncClient.request``.

        Injection strategy (in priority order):
        1. JSON body field ``_taint`` - used for POST/PUT/PATCH endpoints.
        2. Query parameter ``_taint`` - used for GET/DELETE/HEAD.
        3. Custom header ``X-Taint-Token`` - always added regardless of strategy.

        The route pattern from the surface is appended to ``target_url`` if
        it is set and ``target_url`` does not already end with a path segment.
        """
        method = (surface.http_method or "GET").upper()

        # Build the full URL
        base = target_url.rstrip("/")
        if surface.route_pattern:
            # Substitute any path params with safe placeholder values so the
            # request reaches the handler rather than a 404.
            route = surface.route_pattern
            route = _fill_path_params(route)
            url = base + route
        else:
            url = base

        headers = {_TAINT_HEADER: token}
        kwargs: dict = {"headers": headers}

        if method in ("POST", "PUT", "PATCH"):
            # Inject into JSON body
            kwargs["content"] = json.dumps({"_taint": token}).encode()
            headers["Content-Type"] = "application/json"
        else:
            # Inject into query string
            kwargs["params"] = {"_taint": token}

        return url, method, kwargs


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fill_path_params(route: str) -> str:
    """
    Replace ``{param}`` or ``:param`` style path segments with ``"test"``
    so the request reaches the handler without triggering a routing 404.

    Examples::

        /api/users/{id}         → /api/users/test
        /api/orders/:orderId    → /api/orders/test
    """
    import re

    # Flask/FastAPI style: {param_name}
    route = re.sub(r"\{[^}]+\}", "test", route)
    # Express style: :paramName
    route = re.sub(r":[a-zA-Z_][a-zA-Z0-9_]*", "test", route)
    return route
