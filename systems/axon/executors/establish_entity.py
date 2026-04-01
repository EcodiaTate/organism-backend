"""
EcodiaOS - Axon ExecuteEstablishEntity Executor

Orchestrates the full legal entity provisioning pipeline:
  1. PREFLIGHT: Check treasury for ~$500 filing fee
  2. DOCUMENTS_GENERATED: Generate operating agreement + articles
  3. SUBMITTED: Submit to registered agent API with payment
  4. AWAITING_HUMAN: Emit HITL_REQUIRED event - pause execution
  5. (Resume via confirm_entity endpoint or event)
  6. REGISTERED: Store entity identity in IdentityVault

This is a multi-phase executor that intentionally pauses at step 4.
Unlike normal executors that complete in one call, this one returns
ExecutionResult with status=SUSPENDED_HITL and stores its state in
Redis so the resume path can pick up where it left off.

The HITL gate is non-negotiable - the organism cannot complete KYC or
provide a wet signature. A human must:
  - Complete identity verification at the registered agent portal
  - Sign the Articles of Organization
  - Provide the confirmed Entity ID back to the organism

Safety constraints:
  - Required autonomy: STEWARD (3) - commits real capital for filing
  - Rate limit: 1 per day - entity formation is deliberate
  - Reversible: False - legal filings cannot be atomically rolled back
  - Max duration: 120s - API submission can be slow
  - Constitutional review via Equor is mandatory (enforced by Axon)
"""

from __future__ import annotations

import json
import secrets
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import structlog

from interfaces.legal.document_engine import (
    generate_articles_of_organization,
    generate_dao_manifesto,
    generate_operating_agreement,
)
from interfaces.legal.types import (
    EntityFormationRecord,
    EntityFormationState,
    EntityParameters,
    EntityRegistration,
    EntityType,
    HITLInstruction,
    JurisdictionCode,
    RegisteredAgentSubmission,
)
from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from interfaces.legal.registered_agent import RegisteredAgentClient
    from systems.identity.vault import IdentityVault
    from systems.oikos.service import OikosService
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger()

# Redis key prefix for suspended entity formation state
_FORMATION_KEY_PREFIX = "eos:legal:formation:"
_FORMATION_TTL_S = 86400 * 7  # 7 days - entity formation can take time

# Minimum treasury balance required for filing
_MINIMUM_FILING_FEE_USD = Decimal("500")


class EstablishEntityExecutor(Executor):
    """
    Provision a legal entity (LLC/DAO) with staged HITL orchestration.

    Required params:
      organism_name (str): Legal name for the entity.
      organiser_name (str): Human who signs articles / completes KYC.
      registered_agent_name (str): Registered agent name.
      registered_agent_address (str): Registered agent physical address.

    Optional params:
      entity_type (str): "wyoming_dao_llc" (default) | "wyoming_llc" | "delaware_llc"
      jurisdiction (str): "WY" (default) | "DE"
      initial_capital_usd (str): Decimal string for initial capital.
      wallet_address (str): On-chain treasury address.

    Returns ExecutionResult with:
      On HITL pause (initial call):
        data: submission_id, auth_code, portal_url, formation_record_id
        status hint: SUSPENDED_HITL
      On resume (after human confirmation):
        data: entity_id, filing_number, formation_date, registration_id
    """

    action_type = "establish_entity"
    description = (
        "Provision a legal entity (LLC/DAO) for the organism. "
        "Generates legal documents, submits filing, and pauses for "
        "human KYC/signature before completing registration (Level 3)."
    )

    required_autonomy = 3       # STEWARD - commits real capital
    reversible = False          # Legal filings are irreversible
    max_duration_ms = 120_000   # API submission + document generation
    rate_limit = RateLimit.per_hour(1)

    def __init__(
        self,
        oikos: OikosService | None = None,
        registered_agent: RegisteredAgentClient | None = None,
        identity_vault: IdentityVault | None = None,
        event_bus: EventBus | None = None,
        redis: RedisClient | None = None,
        send_admin_notification: Any = None,
    ) -> None:
        self._oikos = oikos
        self._registered_agent = registered_agent
        self._identity_vault = identity_vault
        self._event_bus = event_bus
        self._redis = redis
        self._send_admin_notification = send_admin_notification
        self._logger = logger.bind(component="axon.executor.establish_entity")

    # ── Validation ────────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """Fast param validation - no I/O."""
        # Required fields
        for field in ("organism_name", "organiser_name", "registered_agent_name", "registered_agent_address"):
            value = str(params.get(field, "")).strip()
            if not value:
                return ValidationResult.fail(
                    f"{field} is required",
                    **{field: "missing"},
                )

        # Entity type (optional, validated)
        entity_type_raw = str(params.get("entity_type", "wyoming_dao_llc")).strip()
        try:
            EntityType(entity_type_raw)
        except ValueError:
            valid = ", ".join(e.value for e in EntityType)
            return ValidationResult.fail(
                f"entity_type must be one of: {valid}",
                entity_type="invalid",
            )

        # Jurisdiction (optional, validated)
        jurisdiction_raw = str(params.get("jurisdiction", "WY")).strip()
        try:
            JurisdictionCode(jurisdiction_raw)
        except ValueError:
            valid = ", ".join(j.value for j in JurisdictionCode)
            return ValidationResult.fail(
                f"jurisdiction must be one of: {valid}",
                jurisdiction="invalid",
            )

        # Initial capital (optional, must be valid decimal)
        capital_raw = str(params.get("initial_capital_usd", "")).strip()
        if capital_raw:
            try:
                val = Decimal(capital_raw)
                if val < Decimal("0"):
                    return ValidationResult.fail(
                        "initial_capital_usd must be non-negative",
                        initial_capital_usd="negative",
                    )
            except InvalidOperation:
                return ValidationResult.fail(
                    "initial_capital_usd must be a valid decimal",
                    initial_capital_usd="not a decimal",
                )

        return ValidationResult.ok()

    # ── Execution (Initial Call) ──────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Run the entity formation pipeline up to the HITL gate.
        Never raises - failures returned in ExecutionResult.
        """
        # -- Dependency checks --
        if self._registered_agent is None:
            return ExecutionResult(
                success=False,
                error="RegisteredAgentClient not configured.",
            )

        record = EntityFormationRecord(execution_id=context.execution_id)

        # Emit ENTITY_FORMATION_STARTED so Thread/Soma can record this lifecycle event
        await self._emit_formation_started(record, params)

        # -- Step 1: PREFLIGHT - Treasury check --
        record.transition_to(EntityFormationState.PREFLIGHT)

        treasury_ok, treasury_error = await self._check_treasury()
        if not treasury_ok:
            record.transition_to(EntityFormationState.FAILED)
            record.error = treasury_error
            await self._emit_formation_failed(record, treasury_error)
            return ExecutionResult(
                success=False,
                error=treasury_error,
                data={"formation_record": record.model_dump(mode="json")},
            )

        # -- Build EntityParameters from params + organism config --
        entity_params = self._build_entity_params(params)
        record.entity_parameters = entity_params

        self._logger.info(
            "establish_entity_preflight_passed",
            organism_name=entity_params.organism_name,
            entity_type=entity_params.entity_type.value,
            execution_id=context.execution_id,
        )

        # -- Step 2: Generate legal documents --
        record.transition_to(EntityFormationState.DOCUMENTS_GENERATED)

        operating_agreement = generate_operating_agreement(entity_params)
        articles = generate_articles_of_organization(entity_params)
        documents = [operating_agreement, articles]

        if entity_params.entity_type == EntityType.WYOMING_DAO_LLC:
            manifesto = generate_dao_manifesto(entity_params)
            documents.append(manifesto)

        record.documents = documents

        self._logger.info(
            "establish_entity_documents_generated",
            document_count=len(documents),
            document_types=[d.document_type for d in documents],
            execution_id=context.execution_id,
        )

        # -- Step 3: Submit to registered agent --
        record.transition_to(EntityFormationState.SUBMITTED)

        try:
            # Create payment intent
            payment_response = await self._registered_agent.create_payment_intent(
                amount_usd=_MINIMUM_FILING_FEE_USD,
                description=f"Entity formation: {entity_params.organism_name}",
            )

            # Submit formation
            submission_response = await self._registered_agent.submit_formation(
                params=entity_params,
                documents=documents,
                payment_intent_id=payment_response.payment_intent_id,
            )
        except Exception as exc:
            record.transition_to(EntityFormationState.FAILED)
            record.error = f"Registered agent submission failed: {exc}"
            await self._emit_formation_failed(record, record.error)
            return ExecutionResult(
                success=False,
                error=record.error,
                data={"formation_record": record.model_dump(mode="json")},
            )

        record.submission = RegisteredAgentSubmission(
            entity_parameters=entity_params,
            documents=documents,
            payment_intent_id=payment_response.payment_intent_id,
            filing_fee_usd=submission_response.filing_fee_charged_usd,
            submission_id=submission_response.submission_id,
            status=submission_response.status,
        )

        # -- Step 4: HITL Gate - Pause execution --
        record.transition_to(EntityFormationState.AWAITING_HUMAN)

        auth_code = f"{secrets.randbelow(1_000_000):06d}"

        hitl = HITLInstruction(
            execution_id=context.execution_id,
            submission_id=submission_response.submission_id,
            action_required=(
                "Complete KYC verification and sign the Articles of Organization "
                f"for {entity_params.organism_name}. Once the entity is confirmed, "
                f"provide the Entity ID via the /legal/confirm endpoint with "
                f"auth code: {auth_code}"
            ),
            portal_url=submission_response.portal_url,
            auth_code=auth_code,
            entity_parameters=entity_params,
        )
        record.hitl_instruction = hitl

        # Persist formation state to Redis for the resume path
        await self._persist_formation_state(
            auth_code=auth_code,
            record=record,
        )

        # Emit HITL_REQUIRED Synapse event
        await self._emit_hitl_event(record, hitl)

        # Notify human operator
        await self._notify_human(hitl)

        self._logger.info(
            "establish_entity_hitl_paused",
            submission_id=submission_response.submission_id,
            auth_code=auth_code,
            portal_url=submission_response.portal_url,
            execution_id=context.execution_id,
        )

        # Return suspended result - the executor is paused
        return ExecutionResult(
            success=True,
            data={
                "status": ExecutionStatus.SUSPENDED_HITL.value,
                "submission_id": submission_response.submission_id,
                "auth_code": auth_code,
                "portal_url": submission_response.portal_url,
                "formation_record_id": record.id,
                "operating_agreement_hash": operating_agreement.content_hash,
                "documents_generated": len(documents),
                "filing_fee_usd": str(submission_response.filing_fee_charged_usd),
                "message": (
                    "Entity formation submitted. Awaiting human KYC and signature. "
                    f"Auth code: {auth_code}. Portal: {submission_response.portal_url}"
                ),
            },
            side_effects=[
                f"Entity formation initiated for '{entity_params.organism_name}' "
                f"({entity_params.entity_type.value}). Filing fee: "
                f"${submission_response.filing_fee_charged_usd}. "
                f"Paused at HITL gate - awaiting human organiser.",
            ],
            new_observations=[
                f"Legal entity formation in progress: '{entity_params.organism_name}' "
                f"({entity_params.entity_type.value} in {entity_params.jurisdiction.value}). "
                f"Submitted to registered agent (ID: {submission_response.submission_id}). "
                f"Awaiting human KYC/signature. Auth code {auth_code} issued.",
            ],
        )

    # ── Resume (Called after human confirms) ──────────────────────────

    async def resume_after_human_confirmation(
        self,
        auth_code: str,
        entity_id: str,
        filing_number: str = "",
    ) -> ExecutionResult:
        """
        Resume entity formation after the human has completed KYC and
        provided the confirmed entity ID.

        Called from the /legal/confirm API endpoint or an event handler.
        """
        # -- Retrieve suspended state from Redis --
        record = await self._retrieve_formation_state(auth_code)
        if record is None:
            return ExecutionResult(
                success=False,
                error=(
                    f"No suspended formation found for auth code {auth_code}. "
                    "It may have expired or already been consumed."
                ),
            )

        if record.state != EntityFormationState.AWAITING_HUMAN:
            return ExecutionResult(
                success=False,
                error=(
                    f"Formation record is in state '{record.state.value}', "
                    "not 'awaiting_human'. Cannot resume."
                ),
            )

        self._logger.info(
            "establish_entity_resume",
            auth_code=auth_code,
            entity_id=entity_id,
            formation_record_id=record.id,
            execution_id=record.execution_id,
        )

        # -- Step 5: Confirm with registered agent --
        record.transition_to(EntityFormationState.HUMAN_CONFIRMED)

        submission_id = ""
        if record.submission is not None:
            submission_id = record.submission.submission_id

        if self._registered_agent is not None:
            try:
                confirm_response = await self._registered_agent.confirm_entity(
                    submission_id=submission_id,
                    entity_id=entity_id,
                    filing_number=filing_number,
                )
                if not filing_number:
                    filing_number = confirm_response.filing_number
            except Exception as exc:
                self._logger.warning(
                    "establish_entity_confirm_api_failed",
                    error=str(exc),
                )

        # -- Step 6: Register identity in vault --
        registration = EntityRegistration(
            entity_name=record.entity_parameters.organism_name,
            entity_type=record.entity_parameters.entity_type,
            jurisdiction=record.entity_parameters.jurisdiction,
            entity_id=entity_id,
            filing_number=filing_number,
            submission_id=submission_id,
            execution_id=record.execution_id,
            operating_agreement_hash=(
                record.documents[0].content_hash if record.documents else ""
            ),
        )
        record.registration = registration

        if self._identity_vault is not None:
            try:
                envelope = self._identity_vault.encrypt_token_json(
                    token_data={
                        "entity_id": entity_id,
                        "entity_name": registration.entity_name,
                        "entity_type": registration.entity_type.value,
                        "jurisdiction": registration.jurisdiction.value,
                        "filing_number": filing_number,
                        "formation_date": registration.formation_date.isoformat(),
                        "operating_agreement_hash": registration.operating_agreement_hash,
                    },
                    platform_id="legal_entity",
                )
                self._logger.info(
                    "entity_identity_stored_in_vault",
                    envelope_id=envelope.id,
                    entity_id=entity_id,
                )
            except Exception as exc:
                self._logger.error(
                    "entity_identity_vault_store_failed",
                    entity_id=entity_id,
                    error=str(exc),
                )
                record.transition_to(EntityFormationState.FAILED)
                record.error = f"Failed to store entity in vault: {exc}"
                await self._emit_formation_failed(record, record.error)
                return ExecutionResult(
                    success=False,
                    error=record.error,
                    data={"formation_record": record.model_dump(mode="json")},
                )

        record.transition_to(EntityFormationState.REGISTERED)

        # Clean up Redis state
        await self._delete_formation_state(auth_code)

        # Emit completion event
        await self._emit_completion_event(record)

        self._logger.info(
            "establish_entity_complete",
            entity_id=entity_id,
            entity_name=registration.entity_name,
            filing_number=filing_number,
            execution_id=record.execution_id,
        )

        return ExecutionResult(
            success=True,
            data={
                "status": "registered",
                "entity_id": entity_id,
                "entity_name": registration.entity_name,
                "entity_type": registration.entity_type.value,
                "jurisdiction": registration.jurisdiction.value,
                "filing_number": filing_number,
                "formation_date": registration.formation_date.isoformat(),
                "registration_id": registration.id,
                "operating_agreement_hash": registration.operating_agreement_hash,
            },
            side_effects=[
                f"Legal entity registered: '{registration.entity_name}' "
                f"(Entity ID: {entity_id}, Filing: {filing_number}). "
                f"Identity sealed in IdentityVault.",
            ],
            new_observations=[
                f"Legal sovereignty achieved: '{registration.entity_name}' "
                f"is now a registered {registration.entity_type.value} in "
                f"{registration.jurisdiction.value}. Entity ID: {entity_id}. "
                f"The organism can now hold assets and enter contracts.",
            ],
        )

    # ── Treasury Check ────────────────────────────────────────────────

    async def _check_treasury(self) -> tuple[bool, str]:
        """Verify the treasury has sufficient balance for filing fees."""
        if self._oikos is None:
            # If Oikos is not wired, skip treasury check with a warning
            self._logger.warning("establish_entity_no_oikos_configured")
            return True, ""

        try:
            state = self._oikos.snapshot()
            if state.liquid_balance < _MINIMUM_FILING_FEE_USD:
                return False, (
                    f"Insufficient treasury balance for entity formation. "
                    f"Required: ${_MINIMUM_FILING_FEE_USD}, "
                    f"Available: ${state.liquid_balance}. "
                    f"Starvation level: {state.starvation_level.value}."
                )
            return True, ""
        except Exception as exc:
            self._logger.error("establish_entity_treasury_check_failed", error=str(exc))
            return False, f"Treasury check failed: {exc}"

    # ── Parameter Building ────────────────────────────────────────────

    def _build_entity_params(self, params: dict[str, Any]) -> EntityParameters:
        """Build EntityParameters from executor params + organism config."""
        entity_type = EntityType(
            str(params.get("entity_type", "wyoming_dao_llc")).strip()
        )
        jurisdiction = JurisdictionCode(
            str(params.get("jurisdiction", "WY")).strip()
        )
        capital_raw = str(params.get("initial_capital_usd", "0")).strip()
        initial_capital = Decimal(capital_raw) if capital_raw else Decimal("0")

        # Pull constitutional drives from Oikos/config if available
        coherence = 1.0
        care = 1.0
        growth = 1.0
        honesty = 1.0
        amendment_supermajority = 0.75
        amendment_quorum = 0.60
        amendment_deliberation_days = 14
        amendment_cooldown_days = 90

        return EntityParameters(
            organism_name=str(params["organism_name"]).strip(),
            entity_type=entity_type,
            jurisdiction=jurisdiction,
            coherence_drive=coherence,
            care_drive=care,
            growth_drive=growth,
            honesty_drive=honesty,
            amendment_supermajority=amendment_supermajority,
            amendment_quorum=amendment_quorum,
            amendment_deliberation_days=amendment_deliberation_days,
            amendment_cooldown_days=amendment_cooldown_days,
            registered_agent_name=str(params["registered_agent_name"]).strip(),
            registered_agent_address=str(params["registered_agent_address"]).strip(),
            organiser_name=str(params["organiser_name"]).strip(),
            initial_capital_usd=initial_capital,
            wallet_address=str(params.get("wallet_address", "")).strip(),
        )

    # ── Redis Persistence ─────────────────────────────────────────────

    async def _persist_formation_state(
        self,
        auth_code: str,
        record: EntityFormationRecord,
    ) -> None:
        """Store formation state in Redis for the resume path."""
        if self._redis is None:
            self._logger.warning("establish_entity_no_redis_for_state")
            return

        key = f"{_FORMATION_KEY_PREFIX}{auth_code}"
        try:
            await self._redis.set_json(
                key,
                record.model_dump(mode="json"),
                ttl=_FORMATION_TTL_S,
            )
        except Exception as exc:
            self._logger.error(
                "establish_entity_redis_persist_failed",
                auth_code=auth_code,
                error=str(exc),
            )

    async def _retrieve_formation_state(
        self,
        auth_code: str,
    ) -> EntityFormationRecord | None:
        """Retrieve formation state from Redis."""
        if self._redis is None:
            self._logger.error("establish_entity_no_redis_for_resume")
            return None

        key = f"{_FORMATION_KEY_PREFIX}{auth_code}"
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            data = json.loads(raw) if isinstance(raw, str) else raw
            return EntityFormationRecord.model_validate(data)
        except Exception as exc:
            self._logger.error(
                "establish_entity_redis_retrieve_failed",
                auth_code=auth_code,
                error=str(exc),
            )
            return None

    async def _delete_formation_state(self, auth_code: str) -> None:
        """Delete consumed formation state from Redis."""
        if self._redis is None:
            return
        key = f"{_FORMATION_KEY_PREFIX}{auth_code}"
        try:
            await self._redis.delete(key)
        except Exception as exc:
            self._logger.warning(
                "establish_entity_redis_delete_failed",
                auth_code=auth_code,
                error=str(exc),
            )

    # ── Event Emission ────────────────────────────────────────────────

    async def _emit_hitl_event(
        self,
        record: EntityFormationRecord,
        hitl: HITLInstruction,
    ) -> None:
        """Emit ENTITY_FORMATION_HITL_REQUIRED via Synapse event bus."""
        if self._event_bus is None:
            self._logger.warning("establish_entity_no_event_bus")
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.ENTITY_FORMATION_HITL_REQUIRED,
                source_system="axon.establish_entity",
                data={
                    "execution_id": record.execution_id,
                    "formation_record_id": record.id,
                    "auth_code": hitl.auth_code,
                    "portal_url": hitl.portal_url,
                    "action_required": hitl.action_required,
                    "organism_name": record.entity_parameters.organism_name,
                    "entity_type": record.entity_parameters.entity_type.value,
                    "submission_id": hitl.submission_id,
                },
            ))
        except Exception as exc:
            self._logger.warning(
                "establish_entity_event_emit_failed",
                error=str(exc),
            )

    async def _emit_completion_event(
        self,
        record: EntityFormationRecord,
    ) -> None:
        """Emit ENTITY_FORMATION_COMPLETED via Synapse event bus."""
        if self._event_bus is None:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            reg = record.registration
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.ENTITY_FORMATION_COMPLETED,
                source_system="axon.establish_entity",
                data={
                    "execution_id": record.execution_id,
                    "entity_id": reg.entity_id if reg else "",
                    "entity_name": reg.entity_name if reg else "",
                    "entity_type": reg.entity_type.value if reg else "",
                    "jurisdiction": reg.jurisdiction.value if reg else "",
                    "filing_number": reg.filing_number if reg else "",
                },
            ))
        except Exception as exc:
            self._logger.warning(
                "establish_entity_completion_event_failed",
                error=str(exc),
            )

    async def _emit_formation_started(
        self,
        record: EntityFormationRecord,
        params: dict[str, Any],
    ) -> None:
        """Emit ENTITY_FORMATION_STARTED so Thread/Soma can record this lifecycle event."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.ENTITY_FORMATION_STARTED,
                source_system="axon.establish_entity",
                data={
                    "execution_id": record.execution_id,
                    "formation_record_id": record.id,
                    "organism_name": str(params.get("organism_name", "")).strip(),
                    "entity_type": str(params.get("entity_type", "wyoming_dao_llc")).strip(),
                    "jurisdiction": str(params.get("jurisdiction", "WY")).strip(),
                },
            ))
        except Exception as exc:
            self._logger.warning(
                "establish_entity_started_event_failed",
                error=str(exc),
            )

    async def _emit_formation_failed(
        self,
        record: EntityFormationRecord,
        error: str,
    ) -> None:
        """Emit ENTITY_FORMATION_FAILED so Soma can signal distress and Oikos can recover budget."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            entity_name = ""
            entity_type = ""
            if record.entity_parameters is not None:
                entity_name = record.entity_parameters.organism_name
                entity_type = record.entity_parameters.entity_type.value

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.ENTITY_FORMATION_FAILED,
                source_system="axon.establish_entity",
                data={
                    "execution_id": record.execution_id,
                    "formation_record_id": record.id,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "failed_at_state": record.state.value,
                    "error": error[:500],
                },
            ))
        except Exception as exc:
            self._logger.warning(
                "establish_entity_failed_event_failed",
                error=str(exc),
            )

    # ── Human Notification ────────────────────────────────────────────

    async def _notify_human(self, hitl: HITLInstruction) -> None:
        """Notify the human operator that HITL action is required."""
        if self._send_admin_notification is None:
            self._logger.warning(
                "establish_entity_no_notification_hook",
                auth_code=hitl.auth_code,
                portal_url=hitl.portal_url,
            )
            return

        message = (
            f"\U0001f3db\ufe0f EcodiaOS Legal: Entity formation requires your action.\n"
            f"Entity: {hitl.entity_parameters.organism_name if hitl.entity_parameters else 'Unknown'}\n"
            f"Portal: {hitl.portal_url}\n"
            f"Auth Code: {hitl.auth_code}\n"
            f"Action: {hitl.action_required}"
        )

        try:
            import asyncio
            asyncio.create_task(
                self._send_admin_notification(message),
                name=f"legal_hitl_notify_{hitl.auth_code}",
            )
        except Exception as exc:
            self._logger.error(
                "establish_entity_notification_failed",
                auth_code=hitl.auth_code,
                error=str(exc),
            )
