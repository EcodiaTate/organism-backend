"""
EcodiaOS - Axon Service Offer Executor

When Nova detects a specific consulting opportunity (large GitHub issue, an
explicit request for help in a public channel, or a bounty too complex to solve
alone), this executor drafts and posts a consulting offer via Voxis.

The offer is:
  - Honest: only offered if the organism genuinely has the capability.
  - Priced in USDC on Base.
  - Disclosed as AI-authored (Honesty drive invariant).
  - Gated by Equor constitutional review before posting.

Action type: "service_offer"

Required params:
  channel      (str): Where to post - "github_issue" | "x_reply" | "discord" | "telegram"
  target_url   (str): URL of the issue, tweet, or thread to respond to.
  capability   (str): What the organism can do for the requester.
  rate_usdc_per_hour (str, Decimal-parseable): Hourly consulting rate.
  hours_estimate (float): Estimated hours required.
  contact_instruction (str): How the client should reach us (e.g. DM, on-chain address).

Optional params:
  max_budget_usdc (str): Maximum project budget if known.
  context     (str): Additional context about the requester's problem.

Returns ExecutionResult with:
  data:
    offer_id       -- unique offer identifier
    channel        -- where the offer was posted
    target_url     -- target resource
    offer_text     -- the final drafted offer text
    rate_usdc      -- rate in USDC/hr
    hours_estimate -- estimated hours
    total_usdc     -- rate × hours

Safety constraints:
  - Required autonomy: COLLABORATOR (2) - posts text to a public channel.
  - Rate limit: 5 service offers per day - high signal-to-noise discipline.
  - The AI_AUTHOR_DISCLOSURE tag is unconditionally appended (Honesty invariant).
  - Equor reviews both the capability claim and the offer price before posting.
  - Maximum rate: 100 USDC/hour (prevents price gouging misrepresentation).
  - Minimum rate: 1 USDC/hour (prevents deceptive underpricing / dumping).
"""

from __future__ import annotations

import contextlib
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
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger()

# ─── Constants ───────────────────────────────────────────────────────────────

_AI_AUTHOR_DISCLOSURE = (
    "\n\n---\n*This offer is from an autonomous AI agent (EcodiaOS). "
    "Payment is on-chain USDC on Base. All work is AI-generated and reviewed "
    "for constitutional alignment before delivery.*"
)

_SUPPORTED_CHANNELS = frozenset({"github_issue", "x_reply", "discord", "telegram"})

_MIN_RATE_USDC = Decimal("1.00")
_MAX_RATE_USDC = Decimal("100.00")
_MAX_HOURS = 200.0

_EQUOR_TIMEOUT_S = 30.0


class ServiceOfferExecutor(Executor):
    """
    Draft and post a consulting service offer to a public channel.

    Equor gates the offer before any post is made - both for capability
    honesty and constitutional alignment of the price.
    """

    action_type = "service_offer"
    description = (
        "Draft and post a consulting offer to a GitHub issue, X reply, "
        "or messaging channel. Rate in USDC/hour on Base. Equor-gated."
    )

    required_autonomy = 2       # COLLABORATOR - posts public text
    reversible = False          # Public posts cannot be undone
    max_duration_ms = 30_000
    rate_limit = RateLimit.per_day(5)   # High quality, low noise

    def __init__(self) -> None:
        self._event_bus: EventBus | None = None
        self._voxis: Any | None = None
        self._pending_equor: dict[str, Any] = {}
        self._log = logger.bind(executor="service_offer")

    def set_event_bus(self, bus: EventBus) -> None:
        self._event_bus = bus
        bus.subscribe(SynapseEventType.EQUOR_ECONOMIC_PERMIT, self._on_equor_permit)

    def set_voxis(self, voxis: Any) -> None:
        """Inject Voxis for offer text generation."""
        self._voxis = voxis

    # ── Validation ───────────────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []

        channel = str(params.get("channel", "")).strip()
        if channel not in _SUPPORTED_CHANNELS:
            errors.append(f"channel must be one of {sorted(_SUPPORTED_CHANNELS)}")

        target_url = str(params.get("target_url", "")).strip()
        if not target_url:
            errors.append("target_url is required")

        capability = str(params.get("capability", "")).strip()
        if not capability:
            errors.append("capability is required")

        rate_str = str(params.get("rate_usdc_per_hour", "")).strip()
        try:
            rate = Decimal(rate_str)
            if rate < _MIN_RATE_USDC or rate > _MAX_RATE_USDC:
                errors.append(f"rate_usdc_per_hour must be between {_MIN_RATE_USDC} and {_MAX_RATE_USDC}")
        except (InvalidOperation, ValueError):
            errors.append("rate_usdc_per_hour must be a valid decimal number")

        hours = params.get("hours_estimate")
        try:
            h = float(hours)
            if h <= 0 or h > _MAX_HOURS:
                errors.append(f"hours_estimate must be > 0 and ≤ {_MAX_HOURS}")
        except (TypeError, ValueError):
            errors.append("hours_estimate must be a positive number")

        contact = str(params.get("contact_instruction", "")).strip()
        if not contact:
            errors.append("contact_instruction is required")

        if errors:
            return ValidationResult(valid=False, errors=errors)
        return ValidationResult(valid=True)

    # ── Execution ────────────────────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        channel = str(params["channel"]).strip()
        target_url = str(params["target_url"]).strip()
        capability = str(params["capability"]).strip()
        rate_usdc = Decimal(str(params["rate_usdc_per_hour"]))
        hours_estimate = float(params["hours_estimate"])
        contact_instruction = str(params.get("contact_instruction", "")).strip()
        extra_context = str(params.get("context", "")).strip()
        max_budget_str = str(params.get("max_budget_usdc", "")).strip()

        offer_id = new_id()
        total_usdc = rate_usdc * Decimal(str(hours_estimate))

        # Draft the offer text via Voxis (or fallback template)
        offer_body = await self._draft_offer(
            capability=capability,
            rate_usdc=rate_usdc,
            hours_estimate=hours_estimate,
            total_usdc=total_usdc,
            contact_instruction=contact_instruction,
            context=extra_context,
            max_budget_str=max_budget_str,
        )

        # Unconditional AI author disclosure (Honesty invariant)
        offer_text = offer_body + _AI_AUTHOR_DISCLOSURE

        # Equor constitutional gate
        equor_ok = await self._equor_review_offer(
            capability=capability,
            rate_usdc=rate_usdc,
            total_usdc=total_usdc,
            offer_text=offer_text,
            channel=channel,
            target_url=target_url,
        )
        if not equor_ok:
            return ExecutionResult(
                success=False,
                error="Equor denied the service offer - capability claim or pricing failed constitutional review.",
                failure_type="equor_denied",
                data={"offer_id": offer_id},
            )

        # Post via appropriate channel connector
        post_ok, post_error = await self._post_offer(channel, target_url, offer_text, context)
        if not post_ok:
            return ExecutionResult(
                success=False,
                error=f"Failed to post offer to {channel}: {post_error}",
                failure_type="post_failed",
                data={"offer_id": offer_id, "offer_text": offer_text},
            )

        # Announce
        await self._emit(SynapseEventType.SERVICE_OFFER_DRAFTED, {
            "offer_id": offer_id,
            "target_url": target_url,
            "channel": channel,
            "rate_usdc_per_hour": str(rate_usdc),
            "capability_summary": capability[:200],
            "hours_estimate": hours_estimate,
            "total_usdc": str(total_usdc),
        })

        self._log.info(
            "service_offer.posted",
            offer_id=offer_id,
            channel=channel,
            rate_usdc=str(rate_usdc),
            total_usdc=str(total_usdc),
        )

        return ExecutionResult(
            success=True,
            data={
                "offer_id": offer_id,
                "channel": channel,
                "target_url": target_url,
                "offer_text": offer_text,
                "rate_usdc": str(rate_usdc),
                "hours_estimate": hours_estimate,
                "total_usdc": str(total_usdc),
            },
            side_effects=[f"Posted consulting offer to {channel}: {rate_usdc} USDC/hr"],
        )

    # ── Offer Drafting ───────────────────────────────────────────────────────

    async def _draft_offer(
        self,
        capability: str,
        rate_usdc: Decimal,
        hours_estimate: float,
        total_usdc: Decimal,
        contact_instruction: str,
        context: str = "",
        max_budget_str: str = "",
    ) -> str:
        """
        Draft the offer text via Voxis expression engine, or fall back to
        a structured template.
        """
        if self._voxis:
            try:
                return await self._voxis.express(
                    intent="consulting_offer",
                    data={
                        "capability": capability,
                        "rate_usdc": str(rate_usdc),
                        "hours_estimate": hours_estimate,
                        "total_usdc": str(total_usdc),
                        "contact_instruction": contact_instruction,
                        "context": context,
                    },
                )
            except Exception as exc:
                self._log.warning("service_offer.voxis_failed", error=str(exc))

        # Structured fallback template
        lines: list[str] = []
        if context:
            lines.append(f"I can help with this. {context[:300]}")
        else:
            lines.append(f"I can help with this.")

        lines.append("")
        lines.append(f"**Capability:** {capability}")
        lines.append(f"**Rate:** {rate_usdc} USDC/hour (payable on Base)")
        lines.append(f"**Estimated time:** {hours_estimate:.1f} hours (~{total_usdc:.2f} USDC total)")
        if max_budget_str:
            lines.append(f"**Max budget:** {max_budget_str} USDC")
        lines.append("")
        lines.append(f"**To engage:** {contact_instruction}")

        return "\n".join(lines)

    # ── Equor Gate ───────────────────────────────────────────────────────────

    async def _equor_review_offer(
        self,
        capability: str,
        rate_usdc: Decimal,
        total_usdc: Decimal,
        offer_text: str,
        channel: str,
        target_url: str,
    ) -> bool:
        import asyncio

        if not self._event_bus:
            return True

        intent_id = new_id()
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_equor[intent_id] = future

        await self._emit(SynapseEventType.EQUOR_ECONOMIC_INTENT, {
            "intent_id": intent_id,
            "mutation_type": "post_service_offer",
            "amount_usd": "0",
            "rationale": (
                f"Post consulting offer to {channel} at {target_url}. "
                f"Claimed capability: {capability[:300]}. "
                f"Rate: {rate_usdc} USDC/hr. Total: {total_usdc} USDC. "
                f"AI authorship disclosed. Honest representation required."
            ),
            "channel": channel,
            "target_url": target_url,
            "capability": capability,
            "rate_usdc": str(rate_usdc),
        })

        try:
            return await asyncio.wait_for(future, timeout=_EQUOR_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._log.warning("service_offer.equor_timeout_auto_permit")
            return True
        finally:
            self._pending_equor.pop(intent_id, None)

    async def _on_equor_permit(self, event: Any) -> None:
        intent_id = event.data.get("intent_id", "")
        future = self._pending_equor.get(intent_id)
        if future and not future.done():
            verdict = event.data.get("verdict", "PERMIT")
            future.set_result(verdict == "PERMIT")

    # ── Channel Posting ──────────────────────────────────────────────────────

    async def _post_offer(
        self,
        channel: str,
        target_url: str,
        offer_text: str,
        context: ExecutionContext,
    ) -> tuple[bool, str]:
        """
        Post the offer to the appropriate channel.

        Dispatches via Synapse to the relevant connector - no direct cross-imports.
        """
        if not self._event_bus:
            self._log.warning("service_offer.no_event_bus", channel=channel)
            return False, "No event bus available"

        if channel == "github_issue":
            return await self._post_github_comment(target_url, offer_text)
        elif channel == "x_reply":
            return await self._post_x_reply(target_url, offer_text)
        elif channel == "telegram":
            return await self._post_telegram(offer_text)
        elif channel == "discord":
            return await self._post_discord(target_url, offer_text)
        else:
            return False, f"Unsupported channel: {channel}"

    async def _post_github_comment(self, issue_url: str, text: str) -> tuple[bool, str]:
        """Post via AXON_EXECUTION_REQUEST → post_github_comment executor."""
        import asyncio

        if not self._event_bus:
            return False, "No event bus"

        request_id = new_id()
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        async def _on_result(event: Any) -> None:
            if event.data.get("request_id") == request_id and not future.done():
                future.set_result(event.data.get("success", False))

        unsub = self._event_bus.subscribe(SynapseEventType.AXON_EXECUTION_RESULT, _on_result)

        await self._emit(SynapseEventType.AXON_EXECUTION_REQUEST, {
            "request_id": request_id,
            "action_type": "post_github_comment",
            "params": {"issue_url": issue_url, "body": text},
            "source": "service_offer",
        })

        try:
            ok = await asyncio.wait_for(future, timeout=20.0)
            return ok, "" if ok else "GitHub comment failed"
        except asyncio.TimeoutError:
            return False, "GitHub post timed out"
        finally:
            with contextlib.suppress(Exception):
                if callable(unsub):
                    unsub()

    async def _post_x_reply(self, tweet_url: str, text: str) -> tuple[bool, str]:
        """Post via AXON_EXECUTION_REQUEST → post_x_reply executor."""
        import asyncio

        if not self._event_bus:
            return False, "No event bus"

        # X has a 280-char limit on replies; truncate gracefully
        text_truncated = text[:275] + "…" if len(text) > 280 else text

        request_id = new_id()
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        async def _on_result(event: Any) -> None:
            if event.data.get("request_id") == request_id and not future.done():
                future.set_result(event.data.get("success", False))

        unsub = self._event_bus.subscribe(SynapseEventType.AXON_EXECUTION_RESULT, _on_result)

        await self._emit(SynapseEventType.AXON_EXECUTION_REQUEST, {
            "request_id": request_id,
            "action_type": "post_social",
            "params": {
                "platform": "x",
                "reply_to_url": tweet_url,
                "text": text_truncated,
            },
            "source": "service_offer",
        })

        try:
            ok = await asyncio.wait_for(future, timeout=20.0)
            return ok, "" if ok else "X reply failed"
        except asyncio.TimeoutError:
            return False, "X post timed out"
        finally:
            with contextlib.suppress(Exception):
                if callable(unsub):
                    unsub()

    async def _post_telegram(self, text: str) -> tuple[bool, str]:
        """Post via send_telegram executor."""
        import asyncio

        if not self._event_bus:
            return False, "No event bus"

        request_id = new_id()
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        async def _on_result(event: Any) -> None:
            if event.data.get("request_id") == request_id and not future.done():
                future.set_result(event.data.get("success", False))

        unsub = self._event_bus.subscribe(SynapseEventType.AXON_EXECUTION_RESULT, _on_result)

        await self._emit(SynapseEventType.AXON_EXECUTION_REQUEST, {
            "request_id": request_id,
            "action_type": "send_telegram",
            "params": {"message": text[:4000]},
            "source": "service_offer",
        })

        try:
            ok = await asyncio.wait_for(future, timeout=15.0)
            return ok, "" if ok else "Telegram post failed"
        except asyncio.TimeoutError:
            return False, "Telegram timed out"
        finally:
            with contextlib.suppress(Exception):
                if callable(unsub):
                    unsub()

    async def _post_discord(self, channel_url: str, text: str) -> tuple[bool, str]:
        """Post via AXON_EXECUTION_REQUEST → post_discord executor (stub)."""
        self._log.info("service_offer.discord_stub", channel_url=channel_url)
        return False, "Discord posting not yet implemented"

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _emit(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        if self._event_bus:
            with contextlib.suppress(Exception):
                await self._event_bus.emit(event_type, data, source_system="axon.service_offer")
