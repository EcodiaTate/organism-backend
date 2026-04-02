"""
EcodiaOS - Axon Mitosis Executor (Phase 16e: Speciation)

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
  - SpawnChildExecutor: SOVEREIGN autonomy (3) - moves real funds
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
    from systems.axon.service import AxonService
    from systems.equor.service import EquorService
    from systems.evo.service import EvoService
    from systems.memory.service import MemoryService
    from systems.mitosis.fleet_service import MitosisFleetService
    from systems.mitosis.spawner import LocalDockerSpawner
    from systems.nova.service import NovaService
    from systems.oikos.service import OikosService
    from systems.simula.service import SimulaService
    from systems.synapse.service import SynapseService
    from systems.telos.service import TelosService
    from systems.voxis.service import VoxisService

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
      equor_genome_id (str): Equor constitutional genome ID (amendment history).
      axon_genome_id (str): Axon template genome ID (execution strategy inheritance).
      generation (int): Generation number in the lineage (default 1).
    """

    action_type = "spawn_child"
    description = (
        "Spawn a specialised child instance: transfer seed USDC, register "
        "in Oikos fleet, and broadcast birth event (Level 3)"
    )

    required_autonomy = 3       # SOVEREIGN - moves funds on-chain
    reversible = False          # Blockchain transfer + container spin-up
    max_duration_ms = 120_000   # 2 minutes - wallet transfer can be slow
    rate_limit = RateLimit.per_hour(1)  # Reproduction is rare

    def __init__(
        self,
        wallet: WalletClient | None = None,
        oikos: OikosService | None = None,
        synapse: SynapseService | None = None,
        spawner: LocalDockerSpawner | None = None,
        fleet_service: MitosisFleetService | None = None,
        memory: MemoryService | None = None,
        evo: EvoService | None = None,
        simula: SimulaService | None = None,
        equor: EquorService | None = None,
        axon: AxonService | None = None,
        telos: TelosService | None = None,
        soma: Any | None = None,
        nova: NovaService | None = None,
        voxis: VoxisService | None = None,
        eis: Any | None = None,
    ) -> None:
        self._wallet = wallet
        self._oikos = oikos
        self._synapse = synapse
        self._spawner = spawner
        self._fleet_service = fleet_service
        self._memory = memory
        self._evo = evo
        self._simula = simula
        self._equor = equor
        self._axon = axon
        self._telos = telos
        self._soma = soma
        self._nova = nova
        self._voxis = voxis
        self._eis = eis
        self._logger = logger.bind(system="axon.executor.spawn_child")

    def set_fleet_service(self, fleet_service: Any) -> None:
        """Inject MitosisFleetService after construction (used by wire_mitosis_phase)."""
        self._fleet_service = fleet_service  # type: ignore[attr-defined]

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
            # No wallet address AND no spawner to assign one - fatal
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
        is booted first. Seed capital transfer is deferred - the child will
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
        equor_genome_id = str(params.get("equor_genome_id", "")).strip()
        axon_genome_id = str(params.get("axon_genome_id", "")).strip()
        telos_genome_id = str(params.get("telos_genome_id", "")).strip()
        soma_genome_id = str(params.get("soma_genome_id", "")).strip()
        nova_genome_id = str(params.get("nova_genome_id", "")).strip()
        voxis_genome_id = str(params.get("voxis_genome_id", "")).strip()
        eis_genome_id = str(params.get("eis_genome_id", "")).strip()
        generation = int(params.get("generation", 1))

        self._logger.info(
            "spawn_child_starting",
            child_id=child_id,
            wallet=wallet_addr or "(deferred - spawner-assigned)",
            seed=seed_amount,
            niche=niche_name,
            execution_id=context.execution_id,
        )

        # ── Step 0: Prepare child genome (extract → mutate → persist) ──
        organism_genome_id = str(params.get("organism_genome_id", "")).strip()
        if not organism_genome_id and self._fleet_service is not None:
            try:
                child_genome = await self._fleet_service.prepare_child_genome(
                    parent_instance_id=(
                        getattr(self._oikos, "_instance_id", None) or "eos-default"
                    ) if self._oikos is not None else "eos-default",
                    generation=generation,
                )
                if child_genome is not None:
                    organism_genome_id = child_genome.id
                    self._logger.info(
                        "spawn_child_genome_prepared",
                        child_id=child_id,
                        genome_id=organism_genome_id,
                        generation=child_genome.generation,
                    )
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_genome_failed",
                    child_id=child_id,
                    error=str(exc),
                    note="Proceeding without genome - child will use default",
                )

        # ── Step 0b: SG4 - Resolve BeliefGenome and SimulaGenome IDs at spawn time ──
        # belief_genome_id: snapshot of parent's hypothesis set, drive weights, drift history
        # simula_genome_id: snapshot of Simula's current evolution params + last 10 mutations
        # Both are resolved NOW so the child inherits the live parent state, not a stale param.
        belief_genome: object | None = None
        simula_genome: object | None = None
        if not belief_genome_id and self._evo is not None:
            try:
                belief_genome = await self._evo.export_belief_genome()
                if belief_genome is not None:
                    belief_genome_id = str(belief_genome.genome_id)
                    self._logger.info(
                        "spawn_child_belief_genome_resolved",
                        child_id=child_id,
                        belief_genome_id=belief_genome_id,
                        generation=getattr(belief_genome, "generation", generation),
                    )
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_belief_genome_failed",
                    child_id=child_id,
                    error=str(exc),
                    note="Proceeding with empty belief_genome_id - child inherits no belief priors",
                )

        if not simula_genome_id and self._simula is not None:
            try:
                simula_genome = await self._simula.export_simula_genome()
                if simula_genome is not None:
                    simula_genome_id = str(simula_genome.genome_id)
                    self._logger.info(
                        "spawn_child_simula_genome_resolved",
                        child_id=child_id,
                        simula_genome_id=simula_genome_id,
                        generation=getattr(simula_genome, "generation", generation),
                    )
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_simula_genome_failed",
                    child_id=child_id,
                    error=str(exc),
                    note="Proceeding with empty simula_genome_id - child inherits no evolution state",
                )

        # equor_genome_id: snapshot of parent's constitutional amendment history,
        # drive calibration deltas, and constitution hash. Children apply inherited
        # amendments during Equor init so wisdom accumulates across generations.
        equor_genome: object | None = None
        if not equor_genome_id and self._equor is not None:
            try:
                equor_genome = await self._equor.export_equor_genome()
                if equor_genome is not None:
                    equor_genome_id = str(getattr(equor_genome, "genome_id", ""))
                    self._logger.info(
                        "spawn_child_equor_genome_resolved",
                        child_id=child_id,
                        equor_genome_id=equor_genome_id,
                        amendment_count=len(getattr(equor_genome, "top_amendments", [])),
                        total_adopted=getattr(equor_genome, "total_amendments_adopted", 0),
                    )
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_equor_genome_failed",
                    child_id=child_id,
                    error=str(exc),
                    note="Proceeding with empty equor_genome_id - child inherits no constitutional wisdom",
                )

        # axon_genome_id: snapshot of parent's top-10 action templates, circuit breaker
        # thresholds, and per-pattern success rates. Children apply inherited templates
        # during Axon init so execution quality is high from the first cognitive cycle.
        axon_genome: object | None = None
        if not axon_genome_id and self._axon is not None:
            try:
                axon_genome = await self._axon.export_axon_genome(generation=generation)
                if axon_genome is not None:
                    axon_genome_id = str(getattr(axon_genome, "genome_id", ""))
                    self._logger.info(
                        "spawn_child_axon_genome_resolved",
                        child_id=child_id,
                        axon_genome_id=axon_genome_id,
                        template_count=len(getattr(axon_genome, "templates", [])),
                    )
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_axon_genome_failed",
                    child_id=child_id,
                    error=str(exc),
                    note="Proceeding with empty axon_genome_id - child starts with default execution templates",
                )

        # telos_genome_id: snapshot of parent's drive calibration constants (resonance
        # curves, dissipation baseline, inter-drive coupling). Child Telos applies these
        # with bounded mutation jitter so drive geometry evolves across generations.
        telos_genome: object | None = None
        if not telos_genome_id and self._telos is not None:
            try:
                telos_genome = await self._telos.export_telos_genome()
                if telos_genome is not None:
                    telos_genome_id = str(getattr(telos_genome, "genome_id", ""))
                    self._logger.info(
                        "spawn_child_telos_genome_resolved",
                        child_id=child_id,
                        telos_genome_id=telos_genome_id,
                        topology=getattr(telos_genome, "topology", ""),
                        drive_count=len(getattr(telos_genome, "drive_calibrations", {})),
                    )
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_telos_genome_failed",
                    child_id=child_id,
                    error=str(exc),
                    note="Proceeding with empty telos_genome_id - child uses default drive calibrations",
                )

        # soma_genome_id: snapshot of parent's interoceptive setpoints, dynamics matrix,
        # and allostatic baselines. Child Soma applies these with ±5% noise so the child
        # inherits calibrated homeostatic geometry from birth.
        soma_genome: object | None = None
        if not soma_genome_id and self._soma is not None:
            try:
                soma_genome = await self._soma.export_somatic_genome()
                if soma_genome is not None:
                    soma_genome_id = str(getattr(soma_genome, "id", getattr(soma_genome, "system_id", "soma")))
                    self._logger.info(
                        "spawn_child_soma_genome_resolved",
                        child_id=child_id,
                        soma_genome_id=soma_genome_id,
                        version=getattr(soma_genome, "version", 0),
                    )
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_soma_genome_failed",
                    child_id=child_id,
                    error=str(exc),
                    note="Proceeding without soma genome - child uses default homeostatic setpoints",
                )

        # nova_genome_id: snapshot of parent's goal-domain priors, policy success rates,
        # belief urgency thresholds, and active inference EFE weights. Children apply
        # these with ±15% jitter so deliberation quality is high from birth.
        nova_genome: object | None = None
        if not nova_genome_id and self._nova is not None:
            try:
                nova_genome = await self._nova.export_nova_genome()
                if nova_genome is not None:
                    nova_genome_id = str(getattr(nova_genome, "genome_id", ""))
                    self._logger.info(
                        "spawn_child_nova_genome_resolved",
                        child_id=child_id,
                        nova_genome_id=nova_genome_id,
                        domain_count=len(getattr(nova_genome, "goal_domain_priors", {})),
                        policy_count=len(getattr(nova_genome, "policy_success_rates", {})),
                    )
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_nova_genome_failed",
                    child_id=child_id,
                    error=str(exc),
                    note="Proceeding with empty nova_genome_id - child inherits no deliberation priors",
                )

        # voxis_genome_id: snapshot of parent's personality vector, vocabulary affinities,
        # and strategy preferences. Children apply these with ±10% jitter so they speak
        # with their parent's voice from the first expression cycle.
        voxis_genome: object | None = None
        if not voxis_genome_id and self._voxis is not None:
            try:
                voxis_genome = await self._voxis.export_voxis_genome()
                if voxis_genome is not None:
                    voxis_genome_id = str(getattr(voxis_genome, "genome_id", ""))
                    self._logger.info(
                        "spawn_child_voxis_genome_resolved",
                        child_id=child_id,
                        voxis_genome_id=voxis_genome_id,
                        personality_dims=len(getattr(voxis_genome, "personality_vector", {})),
                        vocab_count=len(getattr(voxis_genome, "vocabulary_affinities", {})),
                    )
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_voxis_genome_failed",
                    child_id=child_id,
                    error=str(exc),
                    note="Proceeding with empty voxis_genome_id - child starts with neutral personality",
                )

        # eis_genome_id: snapshot of parent's immune memory (threat patterns, anomaly
        # baselines, quarantine thresholds). Children apply these via EISGenomeExtractor
        # so they recognise known attack signatures from the first percept cycle.
        eis_genome: object | None = None
        if not eis_genome_id and self._eis is not None:
            try:
                genome_extractor = getattr(self._eis, "_genome_extractor", None)
                if genome_extractor is not None:
                    eis_genome = await genome_extractor.extract_genome_segment()
                if eis_genome is not None:
                    eis_genome_id = str(getattr(eis_genome, "id", getattr(eis_genome, "system_id", "eis")))
                    self._logger.info(
                        "spawn_child_eis_genome_resolved",
                        child_id=child_id,
                        eis_genome_id=eis_genome_id,
                        patterns=len(getattr(eis_genome, "payload", {}).get("threat_patterns", [])),
                        baselines=len(getattr(eis_genome, "payload", {}).get("anomaly_baselines", {})),
                    )
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_eis_genome_failed",
                    child_id=child_id,
                    error=str(exc),
                    note="Proceeding without EIS genome - child starts with empty immune memory",
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
                organism_genome_id=organism_genome_id,
                belief_genome_id=belief_genome_id,
                simula_genome_id=simula_genome_id,
                equor_genome_id=equor_genome_id,
                axon_genome_id=axon_genome_id,
                telos_genome_id=telos_genome_id,
                soma_genome_id=soma_genome_id,
                nova_genome_id=nova_genome_id,
                voxis_genome_id=voxis_genome_id,
                eis_genome_id=eis_genome_id,
                generation=generation,
            )

            if seed_config is not None:
                # Attach EquorGenomeFragment payload to seed config for env injection
                if equor_genome is not None:
                    try:
                        equor_payload = equor_genome.model_dump_for_transport()  # type: ignore[union-attr]
                        seed_config.child_config_overrides["equor_genome_payload"] = str(
                            __import__("json").dumps(equor_payload)
                        )
                    except Exception:
                        pass

                # Attach AxonGenomeFragment payload to seed config for env injection
                # Child AxonService reads ORGANISM_AXON_GENOME_PAYLOAD on initialize()
                if axon_genome is not None:
                    try:
                        axon_payload = axon_genome.model_dump_for_transport()  # type: ignore[union-attr]
                        seed_config.child_config_overrides["axon_genome_payload"] = str(
                            __import__("json").dumps(axon_payload)
                        )
                    except Exception:
                        pass

                # Attach TelosGenomeFragment payload to seed config for env injection
                # Child TelosService reads ORGANISM_TELOS_GENOME_PAYLOAD on initialize()
                if telos_genome is not None:
                    try:
                        telos_payload = telos_genome.model_dump_for_transport()  # type: ignore[union-attr]
                        seed_config.child_config_overrides["telos_genome_payload"] = str(
                            __import__("json").dumps(telos_payload)
                        )
                    except Exception:
                        pass

                # Attach BeliefGenome payload to seed config for env injection
                # Child EvoService reads ORGANISM_BELIEF_GENOME_PAYLOAD on initialize()
                if belief_genome is not None:
                    try:
                        belief_payload = belief_genome.model_dump_for_transport()  # type: ignore[union-attr]
                        seed_config.child_config_overrides["belief_genome_payload"] = str(
                            __import__("json").dumps(belief_payload)
                        )
                    except Exception:
                        pass

                # Attach SimulaGenome payload to seed config for env injection
                # Child SimulaService reads ORGANISM_SIMULA_GENOME_PAYLOAD on initialize()
                if simula_genome is not None:
                    try:
                        simula_payload = simula_genome.model_dump_for_transport()  # type: ignore[union-attr]
                        seed_config.child_config_overrides["simula_genome_payload"] = str(
                            __import__("json").dumps(simula_payload)
                        )
                    except Exception:
                        pass

                # Attach Soma OrganGenomeSegment payload to seed config for env injection
                # Child SomaService reads ORGANISM_SOMA_GENOME_PAYLOAD on initialize()
                if soma_genome is not None:
                    try:
                        soma_payload = soma_genome.model_dump()  # type: ignore[union-attr]
                        seed_config.child_config_overrides["soma_genome_payload"] = str(
                            __import__("json").dumps(soma_payload, default=str)
                        )
                    except Exception:
                        pass

                # Attach NovaGenomeFragment payload to seed config for env injection
                # Child NovaService reads ORGANISM_NOVA_GENOME_PAYLOAD on initialize()
                if nova_genome is not None:
                    try:
                        nova_payload = nova_genome.model_dump_for_transport()  # type: ignore[union-attr]
                        seed_config.child_config_overrides["nova_genome_payload"] = str(
                            __import__("json").dumps(nova_payload)
                        )
                    except Exception:
                        pass

                # Attach VoxisGenomeFragment payload to seed config for env injection
                # Child VoxisService reads ORGANISM_VOXIS_GENOME_PAYLOAD on initialize()
                if voxis_genome is not None:
                    try:
                        voxis_payload = voxis_genome.model_dump_for_transport()  # type: ignore[union-attr]
                        seed_config.child_config_overrides["voxis_genome_payload"] = str(
                            __import__("json").dumps(voxis_payload)
                        )
                    except Exception:
                        pass

                # Attach EIS OrganGenomeSegment payload to seed config for env injection
                # Child EISService reads ORGANISM_EIS_GENOME_PAYLOAD on initialize() and
                # passes it to EISGenomeExtractor.seed_from_genome_segment() so the child
                # starts with the parent's immune memory (threat patterns + anomaly baselines).
                if eis_genome is not None:
                    try:
                        eis_payload = eis_genome.model_dump(mode="json")  # type: ignore[union-attr]
                        seed_config.child_config_overrides["eis_genome_payload"] = str(
                            __import__("json").dumps(eis_payload, default=str)
                        )
                    except Exception:
                        pass

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
                note="No LocalDockerSpawner configured - skipping container boot",
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
                    error="WalletClient not configured - cannot transfer seed capital.",
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
                note="No wallet address - child will request funding via federation",
            )

        # ── Step 3: Register child in Oikos via OIKOS_ECONOMIC_QUERY event ──
        # Architecture rule: no cross-system imports. Axon emits
        # OIKOS_ECONOMIC_QUERY{action="register_child"} on the Synapse bus;
        # OikosService subscribes, constructs ChildPosition, and responds with
        # OIKOS_ECONOMIC_RESPONSE. Fallback to direct call only when bus is absent.
        _child_registered = False
        if self._synapse is not None:
            import asyncio as _asyncio

            from systems.synapse.types import SynapseEvent, SynapseEventType

            _reg_request_id = f"spawn_reg_{child_id}"
            _reg_future: _asyncio.Future[dict] = _asyncio.get_event_loop().create_future()

            async def _on_reg_response(ev: Any) -> None:
                ev_data = ev.data if hasattr(ev, "data") else {}
                if ev_data.get("request_id") == _reg_request_id and not _reg_future.done():
                    _reg_future.set_result(ev_data)

            self._synapse.event_bus.subscribe(SynapseEventType.OIKOS_ECONOMIC_RESPONSE, _on_reg_response)
            try:
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.OIKOS_ECONOMIC_QUERY,
                    source_system="axon.spawn_child",
                    data={
                        "request_id": _reg_request_id,
                        "action": "register_child",
                        "child_data": {
                            "instance_id": child_id,
                            "niche": niche_name,
                            "seed_capital_usd": seed_amount if seed_transferred else "0",
                            "current_net_worth_usd": seed_amount if seed_transferred else "0",
                            "dividend_rate": dividend_rate,
                            "status": "alive" if seed_transferred else "spawning",
                            "wallet_address": wallet_addr,
                            "container_id": spawn_container_id,
                        },
                    },
                ))
                try:
                    reg_result = await _asyncio.wait_for(_reg_future, timeout=5.0)
                    _child_registered = reg_result.get("success", False)
                    if not _child_registered:
                        self._logger.warning(
                            "spawn_child_oikos_registration_failed",
                            child_id=child_id,
                            error=reg_result.get("error", "unknown"),
                        )
                except _asyncio.TimeoutError:
                    self._logger.warning(
                        "spawn_child_oikos_registration_timeout",
                        child_id=child_id,
                        note="OikosService did not respond in 5s - registration unconfirmed",
                    )
            except Exception as exc:
                self._logger.warning("spawn_child_oikos_query_failed", child_id=child_id, error=str(exc))
        elif self._oikos is not None:
            # No Synapse bus - fallback direct call (dev/test only, bus unavailable)
            from systems.oikos.models import ChildPosition, ChildStatus  # noqa: PLC0415

            _oikos_config = getattr(self._oikos, "_config", None)
            _max_rescues: int = getattr(_oikos_config, "mitosis_max_rescues_per_child", 2)
            child_position = ChildPosition(
                instance_id=child_id,
                niche=niche_name,
                seed_capital_usd=Decimal(seed_amount) if seed_transferred else Decimal("0"),
                current_net_worth_usd=Decimal(seed_amount) if seed_transferred else Decimal("0"),
                dividend_rate=Decimal(dividend_rate),
                status=ChildStatus.ALIVE if seed_transferred else ChildStatus.SPAWNING,
                wallet_address=wallet_addr,
                container_id=spawn_container_id,
                max_rescues=_max_rescues,
            )
            await self._oikos.register_child(child_position)
            _child_registered = True

        # ── Step 3.5: Export parent genome payload for child inheritance ──
        # Mitosis requires the full genome payload (not just the ID) so the
        # child can reconstruct its starting belief weights, schema priors, and
        # procedure library without needing a round-trip back to the parent.
        parent_genome_payload: dict[str, Any] = {}
        if self._memory is not None and hasattr(self._memory, "export_genome"):
            try:
                parent_genome_payload = await self._memory.export_genome() or {}
                self._logger.info(
                    "spawn_child_genome_exported",
                    child_id=child_id,
                    genome_keys=list(parent_genome_payload.keys()),
                )
            except Exception as exc:
                self._logger.warning(
                    "spawn_child_genome_export_failed",
                    child_id=child_id,
                    error=str(exc),
                    note="Proceeding without genome payload - child will use defaults",
                )

        # ── Step 4: Emit CHILD_SPAWNED + INSTANCE_SPAWNED events ──
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                _spawn_data = {
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
                    "organism_genome_id": organism_genome_id,
                    "belief_genome_id": belief_genome_id,
                    "simula_genome_id": simula_genome_id,
                    "equor_genome_id": equor_genome_id,
                    "axon_genome_id": axon_genome_id,
                    "telos_genome_id": telos_genome_id,
                    "nova_genome_id": nova_genome_id,
                    "voxis_genome_id": voxis_genome_id,
                    "eis_genome_id": eis_genome_id,
                    "generation": generation,
                    # Full genome payload - Mitosis seeds the child from this.
                    # Empty dict when Memory unavailable; child falls back to defaults.
                    "parent_genome_payload": parent_genome_payload,
                }
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.CHILD_SPAWNED,
                    source_system="axon.spawn_child",
                    data=_spawn_data,
                ))
                # INSTANCE_SPAWNED - Nexus/Logos need this to register divergence
                # profiles and snapshot the world model for the new instance.
                await self._synapse.event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.INSTANCE_SPAWNED,
                    source_system="axon.spawn_child",
                    data={
                        "instance_id": child_id,
                        "niche": niche_name,
                        "generation": generation,
                        "container_id": spawn_container_id,
                        "federation_address": spawn_federation_addr,
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
                "organism_genome_id": organism_genome_id,
                "belief_genome_id": belief_genome_id,
                "simula_genome_id": simula_genome_id,
                "equor_genome_id": equor_genome_id,
                "axon_genome_id": axon_genome_id,
                "telos_genome_id": telos_genome_id,
                "nova_genome_id": nova_genome_id,
                "voxis_genome_id": voxis_genome_id,
                "eis_genome_id": eis_genome_id,
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
        organism_genome_id: str = "",
        belief_genome_id: str = "",
        simula_genome_id: str = "",
        equor_genome_id: str = "",
        axon_genome_id: str = "",
        telos_genome_id: str = "",
        soma_genome_id: str = "",
        nova_genome_id: str = "",
        voxis_genome_id: str = "",
        eis_genome_id: str = "",
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
                organism_genome_id=organism_genome_id,
                belief_genome_id=belief_genome_id,
                simula_genome_id=simula_genome_id,
                equor_genome_id=equor_genome_id,
                axon_genome_id=axon_genome_id,
                telos_genome_id=telos_genome_id,
                soma_genome_id=soma_genome_id,
                nova_genome_id=nova_genome_id,
                voxis_genome_id=voxis_genome_id,
                eis_genome_id=eis_genome_id,
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

    required_autonomy = 1       # AWARE - recording, no funds moved
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
