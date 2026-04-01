"""
EcodiaOS - Send Email Executor

Sends email notifications via configured SMTP or API-based email provider.
Emits ACTION_EXECUTED / ACTION_FAILED events and RE training traces.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import ExecutionContext, ExecutionResult, RateLimit, ValidationResult
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class SendEmailExecutor(Executor):
    action_type = "send_email"
    description = "Send an email notification via configured provider"
    required_autonomy = 2
    reversible = False
    max_duration_ms = 30_000
    rate_limit = RateLimit.per_minute(10)

    def __init__(self, email_client: Any = None, event_bus: Any = None) -> None:
        self._email_client = email_client
        self._event_bus = event_bus

    def set_email_client(self, client: Any) -> None:
        """Inject the EmailClient after construction (registry wiring pattern)."""
        self._email_client = client

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        to = params.get("to")
        if not to or not isinstance(to, str):
            return ValidationResult.fail("'to' is required and must be a string")
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", to):
            return ValidationResult.fail(f"Invalid email address: {to}")
        subject = params.get("subject")
        if not subject or not isinstance(subject, str):
            return ValidationResult.fail("'subject' is required and must be a string")
        body = params.get("body")
        if not body or not isinstance(body, str):
            return ValidationResult.fail("'body' is required and must be a string")
        return ValidationResult.ok()

    async def execute(self, params: dict[str, Any], context: ExecutionContext) -> ExecutionResult:
        to = params["to"]
        subject = params["subject"]
        body = params["body"]
        from_addr = params.get("from", "noreply@ecodiaos.org")

        try:
            if self._email_client is None:
                logger.warning("send_email_no_client", to=to, subject=subject)
                return ExecutionResult(
                    success=False,
                    error="No email client configured",
                )

            result = await self._email_client.send(
                to=to,
                from_addr=from_addr,
                subject=subject,
                body=body,
            )

            # RE trace only - AXON_EXECUTION_RESULT (emitted by service.py) is
            # the canonical aggregate event; per-executor ACTION_EXECUTED would
            # create duplicate signals with no dedup strategy.
            await self._emit_re_trace(context, params, success=True)

            return ExecutionResult(
                success=True,
                data={"message_id": result.get("message_id", ""), "to": to},
                side_effects=[f"Email sent to {to}: {subject}"],
            )
        except Exception as exc:
            await self._emit_re_trace(context, params, success=False, error=str(exc))
            return ExecutionResult(success=False, error=str(exc))

    async def _emit_event(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=event_type,
                source_system="axon",
                data=data,
            ))
        except Exception:
            pass

    async def _emit_re_trace(
        self,
        context: ExecutionContext,
        params: dict[str, Any],
        success: bool,
        error: str = "",
    ) -> None:
        if self._event_bus is None:
            return
        try:
            import json as _json
            from primitives.common import DriveAlignmentVector, SystemID
            from primitives.re_training import RETrainingExample

            to = params.get("to", "unknown")
            subject = params.get("subject", "")
            body_len = len(params.get("body", ""))
            from_addr = params.get("from", "noreply@ecodiaos.org")

            reasoning_trace = "\n".join([
                f"1. VALIDATE: recipient={to!r}, subject present={bool(subject)}, body_length={body_len}",
                f"2. CLIENT CHECK: email_client={'configured' if self._email_client is not None else 'MISSING'}",
                f"3. SEND: from={from_addr!r} → to={to!r}, subject={subject[:80]!r}",
                f"4. OUTCOME: success={success}" + (f", error={error[:120]!r}" if error else ""),
            ])

            alternatives = [
                "Alternative: queue email for async retry if provider is temporarily unavailable",
                "Alternative: use notification executor instead if email delivery confirmation is not required",
            ]
            if not success:
                alternatives.append(
                    "Alternative: escalate to human-in-the-loop if email delivery is critical for the intent"
                )

            counterfactual = ""
            if not success:
                counterfactual = (
                    f"If the email had been sent successfully, the intent's communication goal would be "
                    f"fulfilled. Failure leaves the recipient '{to}' uninformed - downstream steps that "
                    f"depend on the recipient's response cannot proceed."
                )

            equor_alignment = getattr(context.equor_check, "drive_alignment", None) if context.equor_check is not None else None

            trace = RETrainingExample(
                source_system=SystemID.AXON,
                episode_id=context.execution_id,
                instruction=f"Send email from {from_addr!r} to {to!r}: subject={subject[:80]!r}",
                input_context=_json.dumps({
                    "to": to,
                    "from": from_addr,
                    "subject": subject[:200],
                    "body_length": body_len,
                }),
                output=_json.dumps({
                    "success": success,
                    "error": error[:200] if error else None,
                }),
                outcome_quality=1.0 if success else 0.0,
                category="email_delivery",
                constitutional_alignment=equor_alignment or DriveAlignmentVector(),
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives,
                counterfactual=counterfactual,
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="axon",
                data=trace.model_dump(mode="json"),
            ))
        except Exception:
            pass
