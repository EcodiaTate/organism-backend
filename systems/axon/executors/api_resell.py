"""
EcodiaOS - Axon API Resell Executor

The organism resells its own cognitive capabilities as a public API.
Clients pay USDC on Base before their request is processed.

Architecture:
  - ApiResellExecutor  - validates on-chain payment, dispatches to the
                         appropriate cognitive sub-system, returns result.
  - ApiResellConfig    - per-endpoint pricing and rate limits.
  - ResellableEndpoint - schema for a single exposed capability.

Payment flow:
  1. Client sends `price_usdc_per_call` USDC to the organism's revenue wallet.
  2. Client calls POST /api/v1/resell/{endpoint} with their tx_hash.
  3. This executor verifies the tx_hash via WalletClient, then processes.
  4. On success: emits API_RESELL_PAYMENT_RECEIVED + API_RESELL_REQUEST_SERVED
     and REVENUE_INJECTED so Oikos credits the income.

Safety constraints:
  - Required autonomy: SOVEREIGN (3) - real USDC payment verification on-chain.
  - Rate limit: 20 resell requests per hour (burst protection).
  - WalletClient required; no wallet = abort (never process unpaid requests).
  - Every request and its payment are logged to Neo4j.
  - ORGANISM_API_RESELL__ENABLED must be true (defaults false) to activate.
  - Equor reviews capability representation before serving each endpoint.

Env vars:
  ORGANISM_API_RESELL__ENABLED        (bool, default false)
  ORGANISM_API_RESELL__PUBLIC_URL     (str, e.g. https://{instance}.ecodiaos.ai)
  ORGANISM_API_RESELL__REVENUE_WALLET (str, Base wallet address for payments)
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)
from systems.synapse.types import SynapseEventType

if TYPE_CHECKING:
    from clients.wallet import WalletClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger()

# ─── Config ──────────────────────────────────────────────────────────────────

# USDC on Base
_USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
_USDC_DECIMALS = 6

# Minimum payment confirmation depth (blocks)
_MIN_CONFIRMATIONS = 2

# Equor review timeout
_EQUOR_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class ResellableEndpoint:
    """A single API capability offered for sale."""

    path: str                           # e.g. "/solve"
    description: str
    price_usdc_per_call: Decimal
    rate_limit_per_day: int             # per client_id
    # Human-readable capability claim - Equor validates this is accurate
    capability_claim: str = ""


@dataclass
class ApiResellConfig:
    """Live configuration for the API resell service."""

    enabled: bool = False
    public_url: str = ""
    revenue_wallet: str = ""

    endpoints: list[ResellableEndpoint] = field(default_factory=lambda: [
        ResellableEndpoint(
            path="/solve",
            description="Submit a coding problem, receive a solution with explanation.",
            price_usdc_per_call=Decimal("0.50"),
            rate_limit_per_day=10,
            capability_claim=(
                "EcodiaOS can analyse and solve coding problems using its "
                "reasoning engine. Results are best-effort; no guarantee of correctness."
            ),
        ),
        ResellableEndpoint(
            path="/analyse",
            description="Deep code or system analysis with architectural feedback.",
            price_usdc_per_call=Decimal("1.00"),
            rate_limit_per_day=5,
            capability_claim=(
                "EcodiaOS can provide architectural analysis and code review. "
                "Results reflect the organism's current reasoning capability."
            ),
        ),
    ])

    @classmethod
    def from_env(cls) -> ApiResellConfig:
        enabled = os.getenv("ORGANISM_API_RESELL__ENABLED", "false").lower() == "true"
        return cls(
            enabled=enabled,
            public_url=os.getenv("ORGANISM_API_RESELL__PUBLIC_URL", ""),
            revenue_wallet=os.getenv("ORGANISM_API_RESELL__REVENUE_WALLET", ""),
        )


# ─── Per-client rate limiting (in-memory, resets on restart) ─────────────────


@dataclass
class _ClientUsage:
    client_id: str
    date_str: str   # YYYY-MM-DD
    call_count: int = 0


class _ClientRateLimiter:
    def __init__(self) -> None:
        self._usage: dict[str, _ClientUsage] = {}

    def check_and_record(self, client_id: str, endpoint_path: str, limit_per_day: int) -> bool:
        today = utc_now().date().isoformat()
        key = f"{client_id}:{endpoint_path}"
        usage = self._usage.get(key)
        if not usage or usage.date_str != today:
            self._usage[key] = _ClientUsage(client_id=client_id, date_str=today, call_count=1)
            return True
        if usage.call_count >= limit_per_day:
            return False
        usage.call_count += 1
        return True


# ─── Executor ────────────────────────────────────────────────────────────────


class ApiResellExecutor(Executor):
    """
    Execute a paid API resell request.

    Required params:
      client_id  (str): Opaque client identifier (wallet address recommended).
      endpoint   (str): One of "/solve" or "/analyse".
      tx_hash    (str): On-chain USDC payment transaction hash.
      payload    (dict): The request body specific to the endpoint.
        For /solve:   {"problem": str, "language": str (optional)}
        For /analyse: {"code": str, "context": str (optional)}

    Returns ExecutionResult with:
      data:
        request_id        -- unique request identifier
        endpoint          -- endpoint served
        result            -- the cognitive output (str)
        payment_verified  -- True if on-chain payment confirmed
        amount_usdc       -- payment amount verified
    """

    action_type = "api_resell"
    description = (
        "Serve a paid API resell request - verify USDC payment on Base, "
        "dispatch to cognitive subsystem, return result."
    )

    required_autonomy = 3       # SOVEREIGN - real on-chain payment verification
    reversible = False
    max_duration_ms = 120_000   # Reasoning can take up to 2 minutes
    rate_limit = RateLimit.per_hour(20)

    def __init__(
        self,
        wallet: WalletClient | None = None,
        config: ApiResellConfig | None = None,
    ) -> None:
        self._wallet = wallet
        self._config = config or ApiResellConfig.from_env()
        self._rate_limiter = _ClientRateLimiter()
        self._event_bus: EventBus | None = None
        self._pending_equor: dict[str, Any] = {}
        self._log = logger.bind(executor="api_resell")

    def set_wallet(self, wallet: WalletClient) -> None:
        self._wallet = wallet

    def set_event_bus(self, bus: EventBus) -> None:
        self._event_bus = bus
        bus.subscribe(SynapseEventType.EQUOR_ECONOMIC_PERMIT, self._on_equor_permit)

    # ── Validation ───────────────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not self._config.enabled:
            return ValidationResult(
                valid=False,
                errors=["API resell is disabled. Set ORGANISM_API_RESELL__ENABLED=true to activate."],
            )

        client_id = str(params.get("client_id", "")).strip()
        if not client_id:
            return ValidationResult(valid=False, errors=["client_id is required"])

        endpoint = str(params.get("endpoint", "")).strip()
        known_paths = {ep.path for ep in self._config.endpoints}
        if endpoint not in known_paths:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown endpoint '{endpoint}'. Valid: {sorted(known_paths)}"],
            )

        tx_hash = str(params.get("tx_hash", "")).strip()
        if not tx_hash or not tx_hash.startswith("0x"):
            return ValidationResult(valid=False, errors=["tx_hash is required and must start with 0x"])

        payload = params.get("payload")
        if not isinstance(payload, dict):
            return ValidationResult(valid=False, errors=["payload must be a dict"])

        return ValidationResult(valid=True)

    # ── Execution ────────────────────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        import asyncio

        if not self._config.enabled:
            return ExecutionResult(
                success=False,
                error="API resell is disabled.",
                failure_type="disabled",
            )

        client_id = str(params["client_id"]).strip()
        endpoint_path = str(params["endpoint"]).strip()
        tx_hash = str(params["tx_hash"]).strip()
        payload = dict(params.get("payload", {}))
        request_id = new_id()

        # Find endpoint spec
        ep_spec = next(
            (ep for ep in self._config.endpoints if ep.path == endpoint_path),
            None,
        )
        if ep_spec is None:
            return ExecutionResult(success=False, error=f"Unknown endpoint: {endpoint_path}")

        # Per-client rate limit
        if not self._rate_limiter.check_and_record(client_id, endpoint_path, ep_spec.rate_limit_per_day):
            return ExecutionResult(
                success=False,
                error=f"Daily rate limit exceeded for {endpoint_path}. Limit: {ep_spec.rate_limit_per_day}/day.",
                failure_type="rate_limit_exceeded",
            )

        # WalletClient guard - never process without payment verification
        if not self._wallet:
            self._log.error("api_resell.no_wallet")
            return ExecutionResult(
                success=False,
                error="WalletClient not available - cannot verify payment.",
                failure_type="no_wallet",
            )

        # Verify on-chain USDC payment
        t0 = time.monotonic()
        payment_ok, amount_usdc = await self._verify_payment(
            tx_hash, ep_spec.price_usdc_per_call,
        )
        if not payment_ok:
            return ExecutionResult(
                success=False,
                error=f"Payment not verified for tx {tx_hash}. Expected ≥{ep_spec.price_usdc_per_call} USDC.",
                failure_type="payment_not_verified",
                data={"request_id": request_id, "tx_hash": tx_hash},
            )

        # Emit payment received
        await self._emit(SynapseEventType.API_RESELL_PAYMENT_RECEIVED, {
            "client_id": client_id,
            "endpoint": endpoint_path,
            "amount_usd": str(amount_usdc),
            "tx_hash": tx_hash,
            "request_id": request_id,
        })
        # Credit revenue to Oikos
        await self._emit(SynapseEventType.REVENUE_INJECTED, {
            "amount_usd": str(amount_usdc),
            "source": "api_resell",
            "endpoint": endpoint_path,
            "client_id": client_id,
            "tx_hash": tx_hash,
            "stream": "api_resell",
        })

        # Equor gate - verify capability claim is honest before serving
        equor_ok = await self._equor_capability_review(ep_spec, context)
        if not equor_ok:
            return ExecutionResult(
                success=False,
                error="Equor denied capability claim - cannot honestly represent this service.",
                failure_type="equor_denied",
                data={"request_id": request_id, "amount_usdc": str(amount_usdc)},
            )

        # Dispatch to cognitive sub-system
        result_text = await self._dispatch_to_cognition(endpoint_path, payload, context)
        latency_ms = (time.monotonic() - t0) * 1000

        # Emit served event
        await self._emit(SynapseEventType.API_RESELL_REQUEST_SERVED, {
            "client_id": client_id,
            "endpoint": endpoint_path,
            "latency_ms": latency_ms,
            "success": True,
            "request_id": request_id,
        })

        self._log.info(
            "api_resell.served",
            client_id=client_id,
            endpoint=endpoint_path,
            amount_usdc=str(amount_usdc),
            latency_ms=f"{latency_ms:.0f}",
        )

        return ExecutionResult(
            success=True,
            data={
                "request_id": request_id,
                "endpoint": endpoint_path,
                "result": result_text,
                "payment_verified": True,
                "amount_usdc": str(amount_usdc),
                "client_id": client_id,
            },
            side_effects=[f"Served {endpoint_path} for client {client_id[:8]}… ({amount_usdc} USDC)"],
        )

    # ── Payment Verification ─────────────────────────────────────────────────

    async def _verify_payment(
        self,
        tx_hash: str,
        expected_amount: Decimal,
    ) -> tuple[bool, Decimal]:
        """
        Verify that tx_hash is a confirmed USDC transfer to our revenue wallet
        of at least expected_amount.

        Returns (verified: bool, actual_amount: Decimal).
        """
        if not self._wallet:
            return False, Decimal("0")

        try:
            receipt = await self._wallet.get_transaction_receipt(tx_hash)
            if not receipt:
                return False, Decimal("0")

            # Require minimum confirmation depth
            confirmations = receipt.get("confirmations", 0)
            if confirmations < _MIN_CONFIRMATIONS:
                self._log.warning(
                    "api_resell.payment_unconfirmed",
                    tx_hash=tx_hash,
                    confirmations=confirmations,
                )
                return False, Decimal("0")

            # Find a Transfer(from, to, amount) log matching our revenue wallet
            revenue_wallet = self._config.revenue_wallet.lower()
            for log in receipt.get("logs", []):
                if log.get("address", "").lower() != _USDC_BASE.lower():
                    continue
                topics = log.get("topics", [])
                # Transfer event topic: keccak256("Transfer(address,address,uint256)")
                if not topics or topics[0] != "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef":
                    continue
                # to address is topics[2], zero-padded
                to_addr = "0x" + topics[2][-40:] if len(topics) > 2 else ""
                if to_addr.lower() != revenue_wallet:
                    continue
                # Amount is the data field
                amount_raw = int(log.get("data", "0x0"), 16)
                amount_usdc = Decimal(amount_raw) / Decimal(10 ** _USDC_DECIMALS)
                if amount_usdc >= expected_amount:
                    return True, amount_usdc

            return False, Decimal("0")

        except Exception as exc:
            self._log.error("api_resell.payment_verification_failed", error=str(exc), tx_hash=tx_hash)
            return False, Decimal("0")

    # ── Equor Review ─────────────────────────────────────────────────────────

    async def _equor_capability_review(
        self,
        ep: ResellableEndpoint,
        context: ExecutionContext,
    ) -> bool:
        """Equor verifies we're honestly representing our capability."""
        import asyncio

        if not self._event_bus:
            return True  # No bus - optimistic

        intent_id = new_id()
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_equor[intent_id] = future

        await self._emit(SynapseEventType.EQUOR_ECONOMIC_INTENT, {
            "intent_id": intent_id,
            "mutation_type": "api_resell_serve",
            "amount_usd": "0",
            "rationale": (
                f"Serve API resell endpoint {ep.path}. "
                f"Capability claim: {ep.capability_claim}"
            ),
            "endpoint": ep.path,
            "capability_claim": ep.capability_claim,
        })

        try:
            return await asyncio.wait_for(future, timeout=_EQUOR_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._log.warning("api_resell.equor_timeout_auto_permit", endpoint=ep.path)
            return True
        finally:
            self._pending_equor.pop(intent_id, None)

    async def _on_equor_permit(self, event: Any) -> None:
        intent_id = event.data.get("intent_id", "")
        future = self._pending_equor.get(intent_id)
        if future and not future.done():
            verdict = event.data.get("verdict", "PERMIT")
            future.set_result(verdict == "PERMIT")

    # ── Cognitive Dispatch ───────────────────────────────────────────────────

    async def _dispatch_to_cognition(
        self,
        endpoint: str,
        payload: dict[str, Any],
        context: ExecutionContext,
    ) -> str:
        """
        Dispatch the request to the appropriate cognitive capability via Synapse.

        This routes through the existing Nova → Simula reasoning pipeline by
        emitting an AXON_EXECUTION_REQUEST for the relevant internal action type.
        The response is correlated via a request_id Future.

        In the initial implementation we emit a BOUNTY_SOLUTION_REQUESTED-style
        event and let the Nova policy engine handle it.  The result is returned
        as a string.
        """
        if endpoint == "/solve":
            problem = str(payload.get("problem", ""))
            language = str(payload.get("language", "python"))
            return await self._invoke_solve(problem, language)
        elif endpoint == "/analyse":
            code = str(payload.get("code", ""))
            ctx = str(payload.get("context", ""))
            return await self._invoke_analyse(code, ctx)
        return "Endpoint processing not implemented."

    async def _invoke_solve(self, problem: str, language: str) -> str:
        """Invoke Simula's solve pipeline for an external problem."""
        # In production: emit BOUNTY_SOLUTION_REQUESTED and await solution via Future.
        # For the initial wiring, we indicate the intent clearly so Nova/Simula
        # can pick it up in the next cycle.
        if not self._event_bus:
            return "Reasoning pipeline unavailable."

        import asyncio

        request_id = new_id()
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def _on_solution(event: Any) -> None:
            if event.data.get("resell_request_id") == request_id and not future.done():
                future.set_result(event.data.get("solution", "No solution returned."))

        unsub = self._event_bus.subscribe(SynapseEventType.BOUNTY_SOLUTION_PENDING, _on_solution)

        await self._emit(SynapseEventType.BOUNTY_SOLUTION_REQUESTED, {
            "resell_request_id": request_id,
            "problem": problem[:4000],  # truncate to prevent token abuse
            "language": language,
            "source": "api_resell",
        })

        try:
            return await asyncio.wait_for(future, timeout=90.0)
        except asyncio.TimeoutError:
            return "Reasoning timed out. Please retry."
        finally:
            with contextlib.suppress(Exception):
                if callable(unsub):
                    unsub()

    async def _invoke_analyse(self, code: str, ctx: str) -> str:
        """Invoke code analysis."""
        if not self._event_bus:
            return "Analysis pipeline unavailable."

        import asyncio

        request_id = new_id()
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def _on_analysis(event: Any) -> None:
            if event.data.get("resell_request_id") == request_id and not future.done():
                future.set_result(event.data.get("analysis", "No analysis returned."))

        unsub = self._event_bus.subscribe(SynapseEventType.AXON_EXECUTION_RESULT, _on_analysis)

        await self._emit(SynapseEventType.AXON_EXECUTION_REQUEST, {
            "resell_request_id": request_id,
            "action_type": "analyse",
            "params": {"code": code[:8000], "context": ctx[:2000], "source": "api_resell"},
            "source": "api_resell",
        })

        try:
            return await asyncio.wait_for(future, timeout=90.0)
        except asyncio.TimeoutError:
            return "Analysis timed out. Please retry."
        finally:
            with contextlib.suppress(Exception):
                if callable(unsub):
                    unsub()

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _emit(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        if self._event_bus:
            with contextlib.suppress(Exception):
                await self._event_bus.emit(event_type, data, source_system="axon.api_resell")

    def get_endpoint_info(self) -> list[dict[str, Any]]:
        """Return public endpoint catalogue for the organism's documentation."""
        return [
            {
                "path": ep.path,
                "description": ep.description,
                "price_usdc_per_call": str(ep.price_usdc_per_call),
                "rate_limit_per_day": ep.rate_limit_per_day,
                "payment_address": self._config.revenue_wallet,
                "network": "base",
                "payment_token": "USDC",
                "base_url": self._config.public_url,
            }
            for ep in self._config.endpoints
        ]
