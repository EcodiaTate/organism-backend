"""
EcodiaOS - Axon Built-in Executors

All built-in executors for the EOS action system.

Executors are organised by capability category:
  observation    - ObserveExecutor, QueryMemoryExecutor, AnalyseExecutor, SearchExecutor, ScrapePageExecutor
  communication  - RespondTextExecutor, NotificationExecutor, PostMessageExecutor
  data           - CreateRecordExecutor, UpdateRecordExecutor, ScheduleExecutor, ReminderExecutor
  integration    - APICallExecutor, WebhookExecutor
  internal       - StoreInsightExecutor, UpdateGoalExecutor, ConsolidationExecutor
  financial      - WalletTransferExecutor, RequestFundingExecutor, DeFiYieldExecutor
  foraging       - BountyHunterExecutor, SolveBountyExecutor, SolveExternalTaskExecutor
  monitoring     - MonitorPRsExecutor (PR merge monitoring)
  entrepreneurship - DeployAssetExecutor (Phase 16d)
  mitosis        - SpawnChildExecutor, DividendCollectorExecutor (Phase 16e)

Import build_default_registry() to get a fully-populated ExecutorRegistry.
"""

from __future__ import annotations

from typing import Any

from systems.axon.executors.bounty_hunt import BountyHuntExecutor
from systems.axon.executors.bounty_hunter import BountyHunterExecutor
from systems.axon.executors.bounty_submit import BountySubmitExecutor
from systems.axon.executors.cognitive_stall_repair import CognitiveStallRepairExecutor
from systems.axon.executors.communication import (
    NotificationExecutor,
    PostMessageExecutor,
    RespondTextExecutor,
)
from systems.axon.executors.compute_arbitrage import ComputeArbitrageExecutor
from systems.axon.executors.data import (
    CreateRecordExecutor,
    ReminderExecutor,
    ScheduleExecutor,
    UpdateRecordExecutor,
)
from systems.axon.executors.defi_yield import DeFiYieldExecutor
from systems.axon.executors.deploy_asset import DeployAssetExecutor
from systems.axon.executors.establish_entity import EstablishEntityExecutor
from systems.axon.executors.financial import (
    RequestFundingExecutor,
    WalletTransferExecutor,
)
from systems.axon.executors.integration import (
    APICallExecutor,
    WebhookExecutor,
)
from systems.axon.executors.internal import (
    ConsolidationExecutor,
    StoreInsightExecutor,
    UpdateGoalExecutor,
)
from systems.axon.executors.mitosis import (
    DividendCollectorExecutor,
    SpawnChildExecutor,
)
from systems.axon.executors.monitor_prs import MonitorPRsExecutor
from systems.axon.executors.observation import (
    AnalyseExecutor,
    ObserveExecutor,
    QueryMemoryExecutor,
    ScrapePageExecutor,
    SearchExecutor,
)
from systems.axon.executors.phantom_liquidity import PhantomLiquidityExecutor
from systems.axon.executors.adjust_config import AdjustConfigExecutor
from systems.axon.executors.allocate_resource import AllocateResourceExecutor
from systems.axon.executors.federation_send import FederationSendExecutor
from systems.axon.executors.query_memory import QueryMemoryGraphExecutor
from systems.axon.executors.send_email import SendEmailExecutor
from systems.axon.executors.send_telegram import SendTelegramExecutor
from systems.axon.executors.send_discord import SendDiscordExecutor
from systems.axon.executors.simula_codegen_repair import SimulaCodegenRepairExecutor
from systems.axon.executors.publish_content import PublishContentExecutor
from systems.axon.executors.request_telecom import RequestTelecomExecutor
from systems.axon.executors.simula_codegen_stall_repair import SimulaCodegenStallRepairExecutor
from systems.axon.executors.social_post import ExecuteSocialPostExecutor
from systems.axon.executors.update_platform_profile import UpdatePlatformProfileExecutor
from systems.axon.executors.solve_bounty import SolveBountyExecutor
from systems.axon.executors.synapse_cognitive_stall_repair import (
    SynapseCognitiveStallRepairExecutor,
)
from systems.axon.executors.api_resell import ApiResellExecutor
from systems.axon.executors.service_offer import ServiceOfferExecutor
from systems.axon.executors.solve_external import SolveExternalTaskExecutor
from systems.axon.executors.community_engage import CommunityEngageExecutor
from systems.axon.executors.symbridge_factory import SymbridgeFactoryExecutor
from systems.axon.registry import ExecutorRegistry

__all__ = [
    "build_default_registry",
    # Bounty hunt (combined loop)
    "BountyHuntExecutor",
    # Observation
    "ObserveExecutor",
    "QueryMemoryExecutor",
    "AnalyseExecutor",
    "SearchExecutor",
    "ScrapePageExecutor",
    # Communication
    "RespondTextExecutor",
    "NotificationExecutor",
    "PostMessageExecutor",
    # Data
    "CreateRecordExecutor",
    "UpdateRecordExecutor",
    "ScheduleExecutor",
    "ReminderExecutor",
    # Integration
    "APICallExecutor",
    "WebhookExecutor",
    # Financial
    "WalletTransferExecutor",
    "RequestFundingExecutor",
    "DeFiYieldExecutor",
    # Foraging
    "BountyHunterExecutor",
    "SolveBountyExecutor",
    "BountySubmitExecutor",
    # Entrepreneurship (Phase 16d)
    "DeployAssetExecutor",
    # Legal Entity Provisioning (Phase 16g)
    "EstablishEntityExecutor",
    # Compute Arbitrage (Phase 16o)
    "ComputeArbitrageExecutor",
    # Internal
    "StoreInsightExecutor",
    "UpdateGoalExecutor",
    "ConsolidationExecutor",
    # PR Monitoring
    "MonitorPRsExecutor",
    # Mitosis (Phase 16e)
    "SpawnChildExecutor",
    "DividendCollectorExecutor",
    # Social Presence
    "ExecuteSocialPostExecutor",
    "PublishContentExecutor",
    # Phantom Liquidity - sensor network (Phase 16q)
    "PhantomLiquidityExecutor",
    # Federated Telecom Marketplace (Phase 16j)
    "RequestTelecomExecutor",
    # New capability executors
    "SendEmailExecutor",
    "SendTelegramExecutor",
    "SendDiscordExecutor",
    "FederationSendExecutor",
    "AllocateResourceExecutor",
    "AdjustConfigExecutor",
    "QueryMemoryGraphExecutor",
    "UpdatePlatformProfileExecutor",
    # Repair executors (Thymos-triggered)
    "CognitiveStallRepairExecutor",
    "SynapseCognitiveStallRepairExecutor",
    "SimulaCodegenRepairExecutor",
    "SimulaCodegenStallRepairExecutor",
    # Expanded Revenue (Phase 16r+)
    "ApiResellExecutor",
    "ServiceOfferExecutor",
    # General-Purpose Contractor (Phase 16s)
    "SolveExternalTaskExecutor",
    # Community Presence & Engagement
    "CommunityEngageExecutor",
    # Symbiosis (EcodiaOS Factory Bridge)
    "SymbridgeFactoryExecutor",
]


def build_default_registry(
    memory: Any = None,
    voxis: Any = None,
    redis_client: Any = None,
    wallet: Any = None,
    synapse: Any = None,
    oikos: Any = None,
    asset_factory: Any = None,
    simula: Any = None,
    evo: Any = None,
    github_config: Any = None,
    llm: Any = None,
    spawner: Any = None,
    snapshot_pipeline: Any = None,
    compute_arbitrage_config: Any = None,
    compute_providers: dict[str, Any] | None = None,
    vault: Any = None,
    sacm_client: Any = None,
    registered_agent: Any = None,
    event_bus: Any = None,
    send_admin_notification: Any = None,
    github_connector: Any = None,
    atune: Any = None,
    budget_tracker: Any = None,
    circuit_breaker: Any = None,
    rate_limiter: Any = None,
    federation: Any = None,
    identity_comm_config: Any = None,
    fleet_service: Any = None,
    telegram_connector: Any = None,
    persona_engine: Any = None,
    instance_id: str = "unknown",
    web_client: Any = None,
) -> ExecutorRegistry:
    """
    Build and return a fully-populated ExecutorRegistry with all built-in executors.

    Args:
        memory: MemoryService instance (for memory-backed executors)
        voxis: VoxisService instance (for RespondTextExecutor)
        redis_client: Redis client (for scheduled tasks and reminders)
        wallet: WalletClient instance (for WalletTransferExecutor; omit to skip registration)
        synapse: SynapseService instance (for RequestFundingExecutor metabolic reads)
        oikos: OikosService instance (for SpawnChildExecutor fleet registration)
        asset_factory: AssetFactory instance (for DeployAssetExecutor candidate evaluation)
        simula: SimulaService instance (for DeployAssetExecutor code generation)
        github_config: ExternalPlatformsConfig instance (for BountyHunterExecutor live GitHub fetch)
        llm: LLMProvider instance (for BountyHunterExecutor Ecodian alignment scoring)
        spawner: LocalDockerSpawner instance (for SpawnChildExecutor container boot)
        snapshot_pipeline: StateSnapshotPipeline instance
            (for ComputeArbitrageExecutor state capture)
        compute_arbitrage_config: ComputeArbitrageConfig instance (migration thresholds and limits)
        compute_providers: dict[str, ProviderManager] mapping provider IDs to implementations
        vault: IdentityVault instance (for ExecuteSocialPostExecutor credential resolution)
        sacm_client: SACMClient instance (for RemoteComputeExecutor SACM dispatch)
        web_client: WebIntelligenceClient instance (for SearchExecutor real web search
            and ScrapePageExecutor; degrades gracefully to LLM synthesis when None)
    """
    registry = ExecutorRegistry()

    # ── Observation (Level 1) ──────────────────────────────────────
    registry.register(ObserveExecutor(memory=memory))
    # QueryMemoryExecutor superseded by QueryMemoryGraphExecutor (registered below)
    registry.register(AnalyseExecutor(memory=memory))
    registry.register(SearchExecutor(memory=memory, llm=llm, web_client=web_client))
    # ScrapePageExecutor: always registered - degrades gracefully when web_client is None
    registry.register(ScrapePageExecutor(
        memory=memory, llm=llm, web_client=web_client, event_bus=event_bus
    ))

    # ── Communication (Level 1-2) ─────────────────────────────────
    registry.register(RespondTextExecutor(voxis=voxis))
    registry.register(NotificationExecutor(redis_client=redis_client))
    registry.register(PostMessageExecutor(vault=vault))

    # ── Data Operations (Level 2) ─────────────────────────────────
    registry.register(CreateRecordExecutor(memory=memory))
    registry.register(UpdateRecordExecutor(memory=memory))
    registry.register(ScheduleExecutor(redis_client=redis_client))
    registry.register(ReminderExecutor(redis_client=redis_client))

    # ── Integration (Level 2-3) ───────────────────────────────────
    registry.register(APICallExecutor())
    registry.register(WebhookExecutor())

    # ── Financial (Level 1–3) ─────────────────────────────────────
    # RequestFundingExecutor: always registered - it only emits an event,
    # moves no funds, and requires only AWARE autonomy (level 1).
    # WalletTransferExecutor + DeFiYieldExecutor: only registered when
    # a WalletClient is provided (on-chain operations require a wallet).
    registry.register(RequestFundingExecutor(wallet=wallet, synapse=synapse))
    if wallet is not None:
        registry.register(WalletTransferExecutor(wallet=wallet))
        registry.register(DeFiYieldExecutor(wallet=wallet))
        registry.register(PhantomLiquidityExecutor(wallet=wallet))

    # ── Foraging (Level 2–3) ────────────────────────────────────────
    registry.register(BountyHunterExecutor(
        synapse=synapse,
        github_config=github_config,
        llm=llm,
    ))
    registry.register(BountyHuntExecutor(
        synapse=synapse,
        github_config=github_config,
        llm=llm,
        simula=simula,
        memory=memory,
    ))
    registry.register(SolveBountyExecutor(
        simula=simula,
        llm=llm,
    ))

    # BountySubmitExecutor: always registered - degrades gracefully when
    # GitHubConnector is absent, emitting GITHUB_CREDENTIALS_MISSING.
    registry.register(BountySubmitExecutor(
        github_connector=github_connector,
        redis=redis_client,
        event_bus=event_bus,
    ))

    # ── PR Monitoring (Level 1) ──────────────────────────────────
    # Always registered - read-only GitHub API calls, no funds moved.
    # GitHubConnector preferred over github_config for token resolution
    # (supports GitHub App JWT→IAT with Redis caching).
    # Redis enables key cleanup after PR resolves - prevents stale polling.
    registry.register(MonitorPRsExecutor(
        github_config=github_config,
        synapse=synapse,
        redis=redis_client,
        github_connector=github_connector,
    ))

    # ── Entrepreneurship / Phase 16d (Level 3) ──────────────────
    # DeployAssetExecutor requires STEWARD autonomy (level 3) and
    # is only registered when an AssetFactory is provided.
    if asset_factory is not None:
        registry.register(
            DeployAssetExecutor(
                asset_factory=asset_factory,
                oikos=oikos,
                simula=simula,
            )
        )

    # ── Internal (Level 1) ───────────────────────────────────────
    registry.register(StoreInsightExecutor(memory=memory))
    registry.register(UpdateGoalExecutor())
    registry.register(ConsolidationExecutor(memory=memory))

    # ── Mitosis / Phase 16e (Level 1–3) ──────────────────────────
    # SpawnChildExecutor: registered when wallet or spawner is available.
    # With wallet: transfers seed capital on-chain before (or after) container boot.
    # With spawner only: boots container, defers seed transfer until child
    # reports its wallet address via federation handshake.
    # DividendCollectorExecutor: always registered (recording only, no funds moved).
    registry.register(DividendCollectorExecutor(oikos=oikos, synapse=synapse))
    if wallet is not None or spawner is not None:
        registry.register(SpawnChildExecutor(
            wallet=wallet, oikos=oikos, synapse=synapse, spawner=spawner,
            memory=memory, evo=evo, simula=simula,
            fleet_service=fleet_service,
        ))

    # ── Social Presence (Level 2) ─────────────────────────────────
    # ExecuteSocialPostExecutor: always registered - degrades gracefully when
    # vault credentials are absent, so Nova can request operator provisioning.
    registry.register(ExecuteSocialPostExecutor(vault=vault))
    registry.register(PublishContentExecutor(vault=vault))

    # ── Compute Arbitrage / Phase 16o (Level 3) ──────────────────
    # ComputeArbitrageExecutor: registered when snapshot pipeline and
    # compute providers are available. Requires STEWARD autonomy.
    # Orchestrates graceful organism migration between cloud providers.
    if snapshot_pipeline is not None and compute_providers:
        registry.register(ComputeArbitrageExecutor(
            providers=compute_providers,
            snapshot_pipeline=snapshot_pipeline,
            synapse=synapse,
            redis=redis_client,
            config=compute_arbitrage_config,
        ))

    # ── Remote Compute - SACM Bridge / Section XI (Level 2) ──────
    # RemoteComputeExecutor: only registered when a SACMClient is
    # provided. Bridges Axon action_type="remote_compute" intents to
    # the SACM pipeline via SACMClient.submit_and_await().
    if sacm_client is not None:
        from systems.axon.executors.remote_compute import RemoteComputeExecutor
        registry.register(RemoteComputeExecutor(sacm_client=sacm_client))

    # ── Legal Entity Provisioning / Phase 16g (Level 3) ──────────
    # EstablishEntityExecutor: always registered - degrades gracefully when
    # registered_agent or identity_vault are absent. The executor surfaces
    # clear error messages so the operator knows what to provision.
    _establish_entity = EstablishEntityExecutor(
        oikos=oikos,
        registered_agent=registered_agent,
        identity_vault=vault,
        event_bus=event_bus,
        redis=redis_client,
        send_admin_notification=send_admin_notification,
    )
    registry.register(_establish_entity)

    # ── Federated Telecom Marketplace / Phase 16j (Level 3) ──────────
    # RequestTelecomExecutor: registered when wallet is available.
    # Requires SOVEREIGN autonomy (level 3) - moves 5 USDC on-chain.
    # Degrades gracefully when federation or identity_comm_config are None.
    if wallet is not None:
        registry.register(RequestTelecomExecutor(
            wallet=wallet,
            federation=federation,
            identity_comm_config=identity_comm_config,
        ))

    # ── New Capability Executors ─────────────────────────────────────
    registry.register(SendEmailExecutor(event_bus=event_bus))

    # ── Telegram Outbound / Phase 16h (Level 2) ────────────────────
    # Always registered - degrades gracefully (returns success=False) when
    # telegram_connector is absent or ADMIN_CHAT_ID is not set.
    _send_telegram = SendTelegramExecutor(event_bus=event_bus)
    _send_telegram.set_telegram_connector(telegram_connector)
    registry.register(_send_telegram)

    # ── Discord Outbound / Phase 16h (Level 2) ────────────────────
    # Always registered - degrades gracefully (returns success=False) when
    # discord_connector is absent or DISCORD_CHANNEL_ID is not set.
    _send_discord = SendDiscordExecutor(event_bus=event_bus)
    registry.register(_send_discord)

    registry.register(FederationSendExecutor(event_bus=event_bus))
    registry.register(AllocateResourceExecutor(event_bus=event_bus))
    registry.register(AdjustConfigExecutor(event_bus=event_bus))
    registry.register(QueryMemoryGraphExecutor(memory=memory, event_bus=event_bus))

    # ── Repair Executors (Thymos-triggered) ────────────────────────
    # These are invoked by Thymos prescriptions to repair stalled or
    # degraded cognitive systems. All degrade gracefully when deps are None.
    registry.register(CognitiveStallRepairExecutor(
        atune=atune,
        memory=memory,
        synapse=synapse,
        budget_tracker=budget_tracker,
        circuit_breaker=circuit_breaker,
        rate_limiter=rate_limiter,
    ))
    registry.register(SynapseCognitiveStallRepairExecutor(
        synapse=synapse,
        event_bus=event_bus,
    ))
    registry.register(SimulaCodegenRepairExecutor(
        simula=simula,
        synapse=synapse,
    ))
    registry.register(SimulaCodegenStallRepairExecutor(
        simula=simula,
        synapse=synapse,
    ))

    # ── General-Purpose Contractor / Phase 16s ───────────────────────
    # SolveExternalTaskExecutor: registered when simula is available.
    # Clones external repos, generates fixes with Simula, runs language-
    # native tests, and hands off to BountySubmitExecutor for PR submission.
    # Degrades gracefully (returns success=False) when simula is None.
    _solve_external = SolveExternalTaskExecutor(
        simula=simula,
        github_connector=github_connector,
        event_bus=event_bus,
    )
    registry.register(_solve_external)

    # ── Expanded Revenue / Phase 16r+ ────────────────────────────────
    # ApiResellExecutor: registered always - disabled until
    # ECODIAOS_API_RESELL__ENABLED=true; degrades gracefully when disabled.
    _api_resell = ApiResellExecutor(wallet=wallet)
    if event_bus is not None:
        _api_resell.set_event_bus(event_bus)
    registry.register(_api_resell)

    # ServiceOfferExecutor: always registered (COLLABORATOR level 2).
    # Degrades gracefully when event_bus or voxis absent.
    _service_offer = ServiceOfferExecutor()
    if event_bus is not None:
        _service_offer.set_event_bus(event_bus)
    if voxis is not None:
        _service_offer.set_voxis(voxis)
    registry.register(_service_offer)

    # ── Persona / Platform Identity (Spec 23 addendum) ─────────────
    # Always registered - degrades gracefully when PersonaEngine or vault
    # is not yet wired (returns success=False with informative observation).
    _update_profile = UpdatePlatformProfileExecutor()
    if persona_engine is not None:
        _update_profile.set_persona_engine(persona_engine)
    if vault is not None:
        _update_profile.set_vault(vault)
    if redis_client is not None:
        _update_profile.set_redis(redis_client)
    _update_profile.set_instance_id(instance_id)
    registry.register(_update_profile)

    # ── Community Presence & Engagement ──────────────────────────────
    # CommunityEngageExecutor: always registered. Equor gates every
    # engagement; daily limits enforced via Redis (20 GitHub / 10 X).
    # Degrades gracefully when vault or redis absent.
    _community_engage = CommunityEngageExecutor(
        vault=vault,
        redis=redis_client,
        event_bus=event_bus,
    )
    registry.register(_community_engage)

    # SymbridgeFactoryExecutor: dispatches code changes and capability
    # requests to the EcodiaOS Factory via Redis Streams + HTTP fallback.
    _symbridge_factory = SymbridgeFactoryExecutor(
        event_bus=event_bus,
        redis_client=redis_client,
    )
    # EcodiaOS Factory API URL (set via config or env)
    import os as _os
    _factory_url = _os.environ.get("ECODIAOS_FACTORY_API_URL", "")
    if _factory_url:
        _symbridge_factory.set_ecodiaos_url(_factory_url)
    registry.register(_symbridge_factory)

    return registry
