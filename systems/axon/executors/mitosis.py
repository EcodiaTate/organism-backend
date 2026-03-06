"""
EcodiaOS — Axon Mitosis Executor (Phase 16e: Speciation)

SpawnChildExecutor orchestrates the full child-spawning pipeline:
  1. Validate the SeedConfiguration from MitosisEngine
  2. Transfer seed USDC to the child's wallet via WalletClient
  3. Register the child as a ChildPosition in OikosService
  4. Emit CHILD_SPAWNED event via Synapse

DividendCollectorExecutor handles inbound dividend payments:
  1. Validate the dividend report from a child instance
  2. Record the dividend in MitosisEngine's history
  3. Update the child's ChildPosition with cumulative totals
  4. Emit DIVIDEND_RECEIVED event

Safety constraints:
  - SpawnChildExecutor: SOVEREIGN autonomy (3) — moves real funds
  - Rate limit: 1 spawn per hour (reproduction is rare and deliberate)
  - All parameters validated before any on-chain transfer
  - WalletClient injected at construction; never resolved from globals
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

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
    from systems.mitosis.spawner import LocalDockerSpawner
    from systems.oikos.service import OikosService
    from systems.synapse.service import SynapseService

logger = structlog.get_logger()

_EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


class SpawnChildExecutor(Executor):
    """
    Orchestrate the birth of a new child instance.

    This executor handles the irreversible financial act of transferring
    seed capital, then registers the child in Oikos and broadcasts the
    spawn event via Synapse.

    Required params:
      child_instance_id (str): Unique ID for the new child.
      child_wallet_address (str): 0x-prefixed EVM address for the child.
      seed_capital_usd (str): Amount of USDC to transfer as seed capital.
      niche_name (str): Ecological niche the child will specialise in.
      niche_description (str): Description of the niche.
      dividend_rate (str): Decimal rate (e.g. "0.10" for 10%).

    Optional params:
      container_id (str): Infrastructure ID for the child container.
      config_overrides (dict): Config overrides for the child.
      belief_genome_id (str): Evo BeliefGenome ID for genetic inheritance.
      simula_genome_id (str): SimulaGenome ID for evolution inheritance.
      generation (int): Generation number in the lineage (default 1).
    """

    action_type = "spawn_child"
    description = (
        "Spawn a specialised child instance: transfer seed USDC, register "
        "in Oikos fleet, and broadcast birth event (Level 3)"
    )

    required_autonomy = 3       # SOVEREIGN — moves funds on-chain
    reversible = False          # Blockchain transfer + container spin-up
    max_duration_ms = 120_000   # 2 minutes — wallet transfer can be slow
    rate_limit = RateLimit.per_hour(1)  # Reproduction is rare

    def __init__(
        self,
        wallet: WalletClient | None = None,
        oikos: OikosService | None = None,
        synapse: SynapseService | None = None,
        spawner: LocalDockerSpawner | None = None,
    ) -> None:
        self._wallet = wallet
        self._oikos = oikos
        self._synapse = synapse
        self._spawner = spawner
        self._logger = logger.bind(system="axon.executor.spawn_child")

    # ── Validation ─────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        child_id = str(params.get("child_instance_id", "")).strip()
        if not child_id:
            return ValidationResult.fail("child_instance_id is required", child_instance_id="missing")

        wallet_addr = str(params.get("child_wallet_address", "")).strip()
        if wallet_addr:
            # If provided, must be a valid EVM address
            if not _EVM_ADDRESS_RE.match(wallet_addr):
                return ValidationResult.fail(
                    "child_wallet_address must be a 0x-prefixed 40-hex-character EVM address",
                    child_wallet_address="invalid format",
                )
        elif self._spawner is None:
            # No wallet address AND no spawner to assign one — fatal
            return ValidationResult.fail(
                "child_wallet_address is required when no LocalDockerSpawner is configured",
                child_wallet_address="missing",
            )

        seed_raw = str(params.get("seed_capital_usd", "")).strip()
        if not seed_raw:
            return ValidationResult.fail("seed_capital_usd is required", seed_capital_usd="missing")
        try:
            seed = Decimal(seed_raw)
        except InvalidOperation:
            return ValidationResult.fail(
                "seed_capital_usd must be a valid decimal",
                seed_capital_usd="not a decimal",
            )
        if seed <= Decimal("0"):
            return ValidationResult.fail(
                "seed_capital_usd must be positive",
                seed_capital_usd="must be positive",
            )

        niche_name = str(params.get("niche_name", "")).strip()
        if not niche_name:
            return ValidationResult.fail("niche_name is required", niche_name="missing")

        div_raw = str(params.get("dividend_rate", "")).strip()
        if not div_raw:
            return ValidationResult.fail("dividend_rate is required", dividend_rate="missing")
        try:
            div = Decimal(div_raw)
        except InvalidOperation:
            return ValidationResult.fail(
                "dividend_rate must be a valid decimal",
                dividend_rate="not a decimal",
            )
        if div < Decimal("0") or div > Decimal("1"):
            return ValidationResult.fail(
                "dividend_rate must be between 0 and 1",
                dividend_rate="out of range",
            )

        return ValidationResult.ok()

    # ── Execution ──────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Full spawn pipeline:
          1. Boot child container via LocalDockerSpawner (if configured)
          2. Transfer seed USDC to child wallet (if wallet address known)
          3. Register child in OikosService
          4. Emit CHILD_SPAWNED via Synapse

        When triggered by mitosis (no wallet address provided), the container
        is booted first. Seed capital transfer is deferred — the child will
        request funding via ``request_funding`` once it boots and discovers
        its wallet address, closing the loop via the existing financial flow.
        """
        child_id = str(params["child_instance_id"]).strip()
        wallet_addr = str(params.get("child_wallet_address", "")).strip()
        seed_amount = str(params["seed_capital_usd"]).strip()
        niche_name = str(params["niche_name"]).strip()
        niche_desc = str(params.get("niche_description", "")).strip()
        dividend_rate = str(params["dividend_rate"]).strip()
        container_id = str(params.get("container_id", "")).strip()
        belief_genome_id = str(params.get("belief_genome_id", "")).strip()
        simula_genome_id = str(params.get("simula_genome_id", "")).strip()
        generation = int(params.get("generation", 1))

        self._logger.info(
            "spawn_child_starting",
            child_id=child_id,
            wallet=wallet_addr or "(deferred — spawner-assigned)",
            seed=seed_amount,
            niche=niche_name,
            execution_id=context.execution_id,
        )

        # ── Step 1: Boot child container via LocalDockerSpawner ──
        spawn_container_id = container_id
        spawn_federation_addr = ""

        if self._spawner is not None:
            # Build a minimal SeedConfiguration for the spawner.
            seed_config = self._build_seed_config_for_spawner(
                child_id=child_id,
                seed_amount=seed_amount,
                niche_name=niche_name,
                niche_desc=niche_desc,
                dividend_rate=dividend_rate,
                config_overrides=params.get("config_overrides", {}),
                belief_genome_id=belief_genome_id,
                simula_genome_id=simula_genome_id,
                generation=generation,
            )

            if seed_config is not None:
                # Get parent certificate for federation handshake
                parent_cert = ""
                if self._oikos is not None:
                    try:
                        cert_mgr = getattr(self._oikos, "_certificate_manager", None)
                        if cert_mgr is not None:
                            cert = getattr(cert_mgr, "certificate", None)
                            if cert is not None:
                                parent_cert = cert.model_dump_json()
                    except Exception:
                        pass

                try:
                    spawn_result = await self._spawner.spawn_child(
                        child_config=seed_config,
                        parent_cert=parent_cert,
                    )

                    if spawn_result.success:
                        spawn_container_id = spawn_result.container_id
                        spawn_federation_addr = spawn_result.federation_address
                        self._logger.info(
                            "spawn_child_container_booted",
                            child_id=child_id,
                            container_id=spawn_result.container_id[:12],
                            http=spawn_result.http_address,
                            federation=spawn_federation_addr,
                        )
                    else:
                        self._logger.error(
                            "spawn_child_container_failed",
                            child_id=child_id,
                            error=spawn_result.error,
                        )
                        return ExecutionResult(
                            success=False,
                            error=f"Container boot failed: {spawn_result.error}",
                            data={
                                "child_instance_id": child_id,
                                "stage": "container_boot",
                            },
                        )
                except Exception as exc:
                    self._logger.error(
                        "spawn_child_container_exception",
                        child_id=child_id,
                        error=str(exc),
                    )
                    return ExecutionResult(
                        success=False,
                        error=f"Container spawn exception: {exc}",
                        data={
                            "child_instance_id": child_id,
                            "stage": "container_boot",
                        },
                    )
        else:
            self._logger.debug(
                "spawn_child_no_spawner",
                child_id=child_id,
                note="No LocalDockerSpawner configured — skipping container boot",
            )

        # ── Step 2: Transfer seed USDC (if wallet address known) ──
        # When wallet_addr is empty (mitosis-triggered spawn), the child will
        # request funding via ``request_funding`` once it boots and reports
        # its wallet address through the federation handshake.
        tx_hash = ""
        network = ""
        seed_transferred = False

        if wallet_addr:
            if self._wallet is None:
                return ExecutionResult(
                    success=False,
                    error="WalletClient not configured — cannot transfer seed capital.",
                    data={
                        "child_instance_id": child_id,
                        "stage": "transfer",
                        "container_id": spawn_container_id,
                    },
                )
            try:
                transfer_result = await self._wallet.transfer(
                    amount=seed_amount,
                    destination_address=wallet_addr,
                    asset="usdc",
                )
                tx_hash = transfer_result.tx_hash
                network = transfer_result.network
                seed_transferred = True

                self._logger.info(
                    "spawn_child_funded",
                    child_id=child_id,
                    tx_hash=tx_hash,
                    seed=seed_amount,
                    network=network,
                )
            except Exception as exc:
                error_str = str(exc)
                self._logger.error(
                    "spawn_child_transfer_failed",
                    child_id=child_id,
                    error=error_str,
                    execution_id=context.execution_id,
                )
                return ExecutionResult(
                    success=False,
                    error=f"Seed capital transfer failed: {error_str}",
                    data={
                        "child_instance_id": child_id,
                        "stage": "transfer",
                        "container_id": spawn_container_id,
                    },
                )
        else:
            self._logger.info(
                "spawn_child_seed_deferred",
                child_id=child_id,
                note="No wallet address — child will request funding via federation",
            )

        # ── Step 3: Register child in Oikos ──
        from systems.oikos.models import ChildPosition, ChildStatus

        child_position = ChildPosition(
            instance_id=child_id,
            niche=niche_name,
            seed_capital_usd=Decimal(seed_amount) if seed_transferred else Decimal("0"),
            current_net_worth_usd=Decimal(seed_amount) if seed_transferred else Decimal("0"),
            dividend_rate=Decimal(dividend_rate),
            status=ChildStatus.ALIVE if seed_transferred else ChildStatus.SPAWNING,
            wallet_address=wallet_addr,
            container_id=spawn_container_id,
        )

        if self._oikos is not None:
            self._oikos.register_child(child_position)

        # ── Step 4: Emit CHILD_SPAWNED event ──
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.CHILD_SPAWNED,
                    source_system="axon.spawn_child",
                    data={
                        "child_instance_id": child_id,
                        "niche": niche_name,
                        "seed_capital_usd": seed_amount if seed_transferred else "0",
                        "seed_transfer_deferred": not seed_transferred,
                        "dividend_rate": dividend_rate,
                        "wallet_address": wallet_addr,
                        "tx_hash": tx_hash,
                        "network": network,
                        "container_id": spawn_container_id,
                        "federation_address": spawn_federation_addr,
                        "execution_id": context.execution_id,
                        "belief_genome_id": belief_genome_id,
                        "simula_genome_id": simula_genome_id,
                        "generation": generation,
                    },
                ))
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_event_failed",
                    error=str(exc),
                )

        if seed_transferred:
            side_effect = (
                f"Spawned child '{child_id}' in niche '{niche_name}': "
                f"transferred {seed_amount} USDC to {wallet_addr} "
                f"(tx: {tx_hash}, network: {network})"
            )
        else:
            side_effect = (
                f"Spawned child '{child_id}' in niche '{niche_name}': "
                f"container booted, seed capital deferred until federation handshake"
            )
        if spawn_container_id:
            side_effect += f", container: {spawn_container_id[:12]}"

        observation = (
            f"Child instance born: {child_id} specialising in '{niche_name}'. "
            f"Seed capital: {'$' + seed_amount + ' transferred' if seed_transferred else 'deferred'}. "
            f"Dividend rate: {dividend_rate}. "
            f"Container: {spawn_container_id or 'pending'}."
        )

        return ExecutionResult(
            success=True,
            data={
                "child_instance_id": child_id,
                "niche": niche_name,
                "seed_capital_usd": seed_amount if seed_transferred else "0",
                "seed_transfer_deferred": not seed_transferred,
                "dividend_rate": dividend_rate,
                "tx_hash": tx_hash,
                "wallet_address": wallet_addr,
                "network": network,
                "container_id": spawn_container_id,
                "federation_address": spawn_federation_addr,
                "belief_genome_id": belief_genome_id,
                "simula_genome_id": simula_genome_id,
                "generation": generation,
            },
            side_effects=[side_effect],
            new_observations=[observation],
        )

    def _build_seed_config_for_spawner(
        self,
        *,
        child_id: str,
        seed_amount: str,
        niche_name: str,
        niche_desc: str,
        dividend_rate: str,
        config_overrides: dict[str, str],
        belief_genome_id: str = "",
        simula_genome_id: str = "",
        generation: int = 1,
    ) -> Any:
        """Build a SeedConfiguration from executor params for the spawner."""
        try:
            from systems.oikos.models import EcologicalNiche, SeedConfiguration

            niche = EcologicalNiche(
                name=niche_name,
                description=niche_desc or niche_name,
            )
            parent_id = (
                getattr(self._oikos, "_instance_id", None) or "eos-default"
            ) if self._oikos is not None else "eos-default"
            return SeedConfiguration(
                parent_instance_id=parent_id,
                child_instance_id=child_id,
                niche=niche,
                seed_capital_usd=Decimal(seed_amount),
                dividend_rate=Decimal(dividend_rate),
                child_config_overrides=config_overrides or {
                    "specialisation": niche_name,
                    "niche_description": niche_desc or niche_name,
                },
                belief_genome_id=belief_genome_id,
                simula_genome_id=simula_genome_id,
                generation=generation,
            )
        except Exception as exc:
            self._logger.warning(
                "seed_config_build_failed",
                error=str(exc),
            )
            return None


class DividendCollectorExecutor(Executor):
    """
    Record an inbound dividend payment from a child instance.

    This executor is triggered when a child reports its periodic revenue
    and the dividend is confirmed on-chain. It updates OikosService's
    child position and the MitosisEngine's dividend history.

    Required params:
      child_instance_id (str): ID of the child paying the dividend.
      amount_usd (str): Dividend amount received.
      child_net_revenue_usd (str): Child's net revenue for the period.
      tx_hash (str): On-chain transaction hash confirming the payment.

    Optional params:
      period_days (int): Number of days this dividend covers (default 30).
    """

    action_type = "collect_dividend"
    description = (
        "Record a dividend payment received from a child instance and "
        "update fleet economics (Level 1)"
    )

    required_autonomy = 1       # AWARE — recording, no funds moved
    reversible = False
    max_duration_ms = 5_000
    rate_limit = RateLimit.per_hour(20)

    def __init__(
        self,
        oikos: OikosService | None = None,
        synapse: SynapseService | None = None,
    ) -> None:
        self._oikos = oikos
        self._synapse = synapse
        self._logger = logger.bind(system="axon.executor.collect_dividend")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        child_id = str(params.get("child_instance_id", "")).strip()
        if not child_id:
            return ValidationResult.fail("child_instance_id is required", child_instance_id="missing")

        amount_raw = str(params.get("amount_usd", "")).strip()
        if not amount_raw:
            return ValidationResult.fail("amount_usd is required", amount_usd="missing")
        try:
            amount = Decimal(amount_raw)
        except InvalidOperation:
            return ValidationResult.fail("amount_usd must be a valid decimal", amount_usd="not a decimal")
        if amount <= Decimal("0"):
            return ValidationResult.fail("amount_usd must be positive", amount_usd="must be positive")

        tx = str(params.get("tx_hash", "")).strip()
        if not tx:
            return ValidationResult.fail("tx_hash is required", tx_hash="missing")

        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        from datetime import timedelta

        from primitives.common import utc_now
        from systems.oikos.models import DividendRecord

        child_id = str(params["child_instance_id"]).strip()
        amount = Decimal(str(params["amount_usd"]).strip())
        child_revenue = Decimal(str(params.get("child_net_revenue_usd", "0")).strip())
        tx_hash = str(params["tx_hash"]).strip()
        period_days = int(params.get("period_days", 30))

        now = utc_now()
        record = DividendRecord(
            child_instance_id=child_id,
            amount_usd=amount,
            tx_hash=tx_hash,
            period_start=now - timedelta(days=period_days),
            period_end=now,
            child_net_revenue_usd=child_revenue,
            dividend_rate_applied=Decimal(str(params.get("dividend_rate", "0.10"))),
        )

        # Record in OikosService
        if self._oikos is not None:
            self._oikos.record_dividend(record)

        # Emit event
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.DIVIDEND_RECEIVED,
                    source_system="axon.collect_dividend",
                    data={
                        "child_instance_id": child_id,
                        "amount_usd": str(amount),
                        "tx_hash": tx_hash,
                        "child_net_revenue_usd": str(child_revenue),
                        "execution_id": context.execution_id,
                    },
                ))
            except Exception as exc:
                self._logger.warning("dividend_event_failed", error=str(exc))

        self._logger.info(
            "dividend_collected",
            child=child_id,
            amount=str(amount),
            tx_hash=tx_hash,
        )

        return ExecutionResult(
            success=True,
            data={
                "child_instance_id": child_id,
                "amount_usd": str(amount),
                "tx_hash": tx_hash,
            },
            side_effects=[
                f"Dividend ${amount} received from child '{child_id}' (tx: {tx_hash})"
            ],
            new_observations=[
                f"Child '{child_id}' paid ${amount} dividend on ${child_revenue} net revenue."
            ],
        )
