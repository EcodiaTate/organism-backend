"""
EcodiaOS — RequestTelecomExecutor  (Phase 16j: Federated Telecom Marketplace)

Runs on a **child** EcodiaOS instance.  Acquires a Twilio phone number from
the Genesis instance over the Federation network in three phases:

  1. Locate the Genesis node via the active FederationService links.
  2. Pay 5 USDC to the Genesis wallet address via the CDP-managed wallet.
  3. POST a KnowledgeRequest for service=TelecomProvisioning to the Genesis
     federation endpoint, carrying the on-chain tx_hash as proof of payment.

On success the executor receives the provisioned E.164 phone number in the
response body and writes it back to IdentityCommConfig.twilio_from_number so
the instance can immediately send and receive SMS.

Safety constraints:
  - Required autonomy: SOVEREIGN (3) — moves real USDC on-chain
  - Rate limit: 1 per day — phone numbers are persistent resources
  - WalletClient and FederationService injected at construction; never globals
  - Payment is non-refundable; Genesis fulfillment failures are logged and
    returned as ExecutionResult(success=False) for human review
  - Config mutation is in-process only; the env var is NOT changed.  The
    caller is responsible for persisting the new number to durable storage
    (e.g. GCP Secret Manager) if required across restarts.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from clients.wallet import WalletClient
    from config import IdentityCommConfig
    from systems.federation.service import FederationService

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TELECOM_SERVICE_NAME: str = "TelecomProvisioning"
TELECOM_PRICE_USDC: Decimal = Decimal("5")

# The federation endpoint path on the Genesis node that handles telecom requests
_GENESIS_TELECOM_PATH: str = "/api/v1/federation/telecom/provision"

# E.164 validation — all countries (+CC followed by 7-14 digits)
_E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")

# ISO 3166-1 alpha-2 country codes supported by provision_new_phone_number()
_SUPPORTED_COUNTRIES = frozenset({
    "US", "AU", "GB", "CA", "NZ", "IE", "ZA", "IN", "SG",
    "DE", "FR", "JP", "BR", "MX", "SE", "NO", "FI", "DK",
    "NL", "BE", "CH", "AT", "IT", "ES", "PT", "PL", "CZ",
})

# HTTP timeout for the remote Genesis call (provisioning can take ~15 s)
_HTTP_TIMEOUT_S: float = 30.0


def _is_genesis(link: Any) -> bool:
    """Return True if this FederationLink connects to the Genesis instance."""
    remote_id: str = getattr(link, "remote_instance_id", "") or ""
    # Genesis nodes self-identify with the canonical prefix "genesis"
    return remote_id.lower().startswith("genesis")


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class RequestTelecomExecutor(Executor):
    """
    Acquire a new phone number from the Genesis Federated Telecom Marketplace.

    Required params:
      (none) — all configuration is resolved from injected services.

    Optional params:
      country (str): ISO 3166-1 alpha-2 country code (default "US").
          Examples: "US", "AU", "GB", "CA", "NZ". Determines which
          Twilio number pool is searched.
      area_code (str): Area code hint forwarded to Genesis (default "415"
          for US; for AU use 2-digit codes like "02", "03", "07", "08").
      genesis_wallet_address (str): 0x-prefixed EVM address of the Genesis
          wallet.  If omitted the executor reads it from the first active
          federation link whose remote_instance_id starts with "genesis".

    Returns ExecutionResult with:
      data:
        phone_number         -- E.164 number provisioned by Genesis
        tx_hash              -- on-chain USDC payment tx hash
        genesis_instance_id  -- Genesis node that fulfilled the request
      side_effects:
        -- Human-readable description for the world-state log
      new_observations:
        -- Observation fed back into Atune for cognitive awareness
    """

    action_type = "request_telecom"
    description = (
        "Acquire a Twilio phone number from the Genesis instance via the "
        "Federated Telecom Marketplace (5 USDC). Supports any country Twilio "
        "offers Local numbers in (US, AU, GB, CA, etc.). Updates "
        "IdentityCommConfig so the instance can immediately send and receive SMS."
    )

    required_autonomy = 3       # SOVEREIGN — moves USDC on-chain
    reversible = False          # Phone purchase + USDC transfer both irreversible
    max_duration_ms = 60_000    # Genesis provisioning can take up to ~30 s
    rate_limit = RateLimit.per_day(1)   # One number per day maximum

    def __init__(
        self,
        wallet: WalletClient | None = None,
        federation: FederationService | None = None,
        identity_comm_config: IdentityCommConfig | None = None,
    ) -> None:
        self._wallet = wallet
        self._federation = federation
        self._identity_comm_config = identity_comm_config
        self._log = logger.bind(system="axon.executor.request_telecom")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if self._wallet is None:
            return ValidationResult.fail(
                "WalletClient not configured — cannot pay for telecom provisioning.",
                wallet="missing",
            )
        if self._federation is None:
            return ValidationResult.fail(
                "FederationService not configured — cannot reach Genesis node.",
                federation="missing",
            )

        country = str(params.get("country", "US")).strip().upper()
        if country not in _SUPPORTED_COUNTRIES:
            return ValidationResult.fail(
                f"country must be an ISO 3166-1 alpha-2 code from the supported list: "
                f"{sorted(_SUPPORTED_COUNTRIES)}",
                country="unsupported",
            )

        area_code = str(params.get("area_code", "415" if country == "US" else "")).strip()
        if area_code and not area_code.isdigit():
            return ValidationResult.fail(
                "area_code must be a numeric string (e.g. '415' for US, '02' for AU)",
                area_code="invalid",
            )

        genesis_addr = str(params.get("genesis_wallet_address", "")).strip()
        if genesis_addr and not re.match(r"^0x[0-9a-fA-F]{40}$", genesis_addr):
            return ValidationResult.fail(
                "genesis_wallet_address must be a 0x-prefixed 40-hex-char EVM address",
                genesis_wallet_address="invalid format",
            )

        return ValidationResult.ok()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Three-phase execution:
          1. Locate Genesis node.
          2. Pay 5 USDC on-chain.
          3. Submit KnowledgeRequest → receive phone number → update config.
        """
        country = str(params.get("country", "US")).strip().upper()
        area_code = str(params.get("area_code", "415" if country == "US" else "")).strip()
        genesis_wallet_address = str(params.get("genesis_wallet_address", "")).strip()

        # ── Phase 1: Locate Genesis federation link ────────────────────────
        genesis_link = self._find_genesis_link()
        if genesis_link is None:
            return ExecutionResult(
                success=False,
                error=(
                    "No active federation link to a Genesis instance found. "
                    "Establish a federation link with the Genesis node first."
                ),
            )

        genesis_instance_id: str = genesis_link.remote_instance_id
        genesis_endpoint: str = genesis_link.remote_endpoint

        self._log.info(
            "telecom_request_genesis_located",
            genesis_instance_id=genesis_instance_id,
            genesis_endpoint=genesis_endpoint,
            execution_id=context.execution_id,
        )

        # Resolve Genesis wallet address: prefer explicit param, then link metadata
        if not genesis_wallet_address:
            genesis_wallet_address = getattr(
                genesis_link.remote_identity, "wallet_address", ""
            ) or ""
        if not genesis_wallet_address:
            return ExecutionResult(
                success=False,
                error=(
                    "Genesis wallet address unknown. Pass genesis_wallet_address= "
                    "param or ensure the FederationLink.remote_identity has a "
                    "wallet_address attribute."
                ),
            )

        # ── Phase 2: Pay 5 USDC on-chain ──────────────────────────────────
        self._log.info(
            "telecom_request_paying",
            amount=str(TELECOM_PRICE_USDC),
            destination=genesis_wallet_address,
            execution_id=context.execution_id,
        )

        try:
            transfer_result = await self._wallet.transfer(  # type: ignore[union-attr]
                amount=str(TELECOM_PRICE_USDC),
                destination_address=genesis_wallet_address,
                asset="usdc",
            )
        except Exception as exc:
            error_str = str(exc)
            self._log.error(
                "telecom_request_payment_failed",
                error=error_str,
                execution_id=context.execution_id,
            )
            insufficient = any(
                phrase in error_str.lower()
                for phrase in ("insufficient funds", "insufficient balance", "exceeds balance")
            )
            return ExecutionResult(
                success=False,
                error=(
                    f"INSUFFICIENT_FUNDS: wallet lacks {TELECOM_PRICE_USDC} USDC "
                    f"to pay Genesis for telecom provisioning."
                    if insufficient
                    else f"USDC payment to Genesis failed: {error_str}"
                ),
                data={"failure_type": "insufficient_funds" if insufficient else "payment_error"},
            )

        tx_hash: str = transfer_result.tx_hash
        self._log.info(
            "telecom_request_payment_confirmed",
            tx_hash=tx_hash,
            genesis_instance_id=genesis_instance_id,
            execution_id=context.execution_id,
        )

        # ── Phase 3: Submit KnowledgeRequest to Genesis ────────────────────
        phone_number = await self._submit_provision_request(
            genesis_endpoint=genesis_endpoint,
            genesis_instance_id=genesis_instance_id,
            tx_hash=tx_hash,
            area_code=area_code,
            country=country,
            execution_id=context.execution_id,
        )

        if phone_number is None:
            return ExecutionResult(
                success=False,
                error=(
                    f"Genesis accepted payment (tx: {tx_hash}) but failed to "
                    f"provision a phone number. Contact Genesis operator for refund. "
                    f"tx_hash={tx_hash}, genesis={genesis_instance_id}"
                ),
                data={
                    "tx_hash": tx_hash,
                    "genesis_instance_id": genesis_instance_id,
                    "failure_type": "fulfillment_failed",
                },
            )

        # ── Self-configuration: update IdentityCommConfig in-process ───────
        if self._identity_comm_config is not None:
            try:
                object.__setattr__(
                    self._identity_comm_config,
                    "twilio_from_number",
                    phone_number,
                )
                self._log.info(
                    "telecom_self_config_updated",
                    twilio_from_number=phone_number,
                    execution_id=context.execution_id,
                )
            except Exception as exc:
                # Non-fatal: number is provisioned, just log the config failure
                self._log.warning(
                    "telecom_self_config_failed",
                    error=str(exc),
                    phone_number=phone_number,
                    execution_id=context.execution_id,
                )

        self._log.info(
            "telecom_provisioning_complete",
            phone_number=phone_number,
            tx_hash=tx_hash,
            genesis_instance_id=genesis_instance_id,
            execution_id=context.execution_id,
        )

        return ExecutionResult(
            success=True,
            data={
                "phone_number": phone_number,
                "tx_hash": tx_hash,
                "genesis_instance_id": genesis_instance_id,
            },
            side_effects=[
                f"Provisioned Twilio number {phone_number} (country={country}) via Genesis "
                f"({genesis_instance_id}). Paid {TELECOM_PRICE_USDC} USDC "
                f"(tx: {tx_hash}). IdentityCommConfig.twilio_from_number updated."
            ],
            new_observations=[
                f"New phone number acquired from Genesis Telecom Marketplace: "
                f"{phone_number} (country={country}). The instance can now send and receive SMS."
            ],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_genesis_link(self) -> Any | None:
        """Return the first active FederationLink whose remote is a Genesis node."""
        if self._federation is None:
            return None
        links: dict[str, Any] = getattr(self._federation, "_links", {})
        for link in links.values():
            status = getattr(link, "status", None)
            status_value = getattr(status, "value", str(status))
            if status_value == "active" and _is_genesis(link):
                return link
        return None

    async def _submit_provision_request(
        self,
        genesis_endpoint: str,
        genesis_instance_id: str,
        tx_hash: str,
        area_code: str,
        country: str,
        execution_id: str,
    ) -> str | None:
        """
        POST a TelecomProvisioning KnowledgeRequest to the Genesis node.

        Returns the provisioned E.164 phone number, or None on failure.
        """
        our_instance_id: str = (
            getattr(self._federation, "_instance_id", "unknown")
            if self._federation is not None
            else "unknown"
        )

        payload: dict[str, Any] = {
            "service": TELECOM_SERVICE_NAME,
            "requesting_instance_id": our_instance_id,
            "payment": {
                "tx_hash": tx_hash,
                "amount_usdc": str(TELECOM_PRICE_USDC),
                "asset": "usdc",
            },
            "country": country,
            "area_code": area_code,
            "execution_id": execution_id,
        }

        url = genesis_endpoint.rstrip("/") + _GENESIS_TELECOM_PATH

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                response = await client.post(url, json=payload)
        except Exception as exc:
            self._log.error(
                "telecom_provision_request_network_error",
                genesis_endpoint=url,
                error=str(exc),
                execution_id=execution_id,
            )
            return None

        if response.status_code != 200:
            self._log.error(
                "telecom_provision_request_rejected",
                status=response.status_code,
                body=response.text[:500],
                genesis_instance_id=genesis_instance_id,
                execution_id=execution_id,
            )
            return None

        try:
            data: dict[str, Any] = response.json()
        except Exception:
            self._log.error(
                "telecom_provision_response_not_json",
                body=response.text[:200],
                execution_id=execution_id,
            )
            return None

        phone_number: str = str(data.get("phone_number", "")).strip()
        if not _E164_RE.match(phone_number):
            self._log.error(
                "telecom_provision_invalid_number",
                phone_number=phone_number,
                execution_id=execution_id,
            )
            return None

        return phone_number
