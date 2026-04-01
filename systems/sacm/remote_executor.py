"""
EcodiaOS - SACM Remote Execution Manager

The orchestrator for the Execution & Verification subsystem.
Coordinates the full lifecycle of dispatching a workload to a remote
compute provider with end-to-end encryption and dual-strategy
verification.

Pipeline:
  1. PREPARE - Serialize workload, generate canaries, mix into batch.
  2. ENCRYPT - Seal the mixed batch with X25519/AES-256-GCM for the
     target provider.
  3. DISPATCH - Send encrypted payload to the remote provider.
  4. RECEIVE - Collect and decrypt the result batch.
  5. VERIFY - Run consensus verification (deterministic replay +
     probabilistic canary audit).
  6. ACCEPT/REJECT - If accepted, extract and return real results.
     If rejected, mark provider, optionally retry on a different
     provider.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, new_id
from systems.sacm.encryption import (
    EncryptedEnvelope,
    EncryptionMeta,
    decrypt_payload,
    encrypt_payload,
    generate_keypair,
    public_key_from_bytes,
)
from systems.sacm.verification.consensus import (
    ConsensusOutcome,
    ConsensusReport,
    ConsensusVerifier,
    ConsensusWeights,
)
from systems.sacm.verification.deterministic import (
    DeterministicReplayVerifier,
    MatchKind,
    ReplayItem,
    ToleranceSpec,
)
from systems.sacm.verification.probabilistic import (
    CanaryPreparer,
    MixedBatch,
    ProbabilisticAuditVerifier,
    extract_real_results,
    mix_canaries_into_batch,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from systems.sacm.workload import WorkloadDescriptor

logger = structlog.get_logger("systems.sacm.remote_executor")


# ─── Placement Decision ────────────────────────────────────────
# Defined here until optimizer.py is built and owns it.  Once the
# optimizer agent creates optimizer.py, move this class there and
# re-export or import from that module.


class PlacementDecision(Identified, Timestamped):
    """
    The optimizer's output: which provider should execute a workload.

    Carries the chosen provider ID, its X25519 public key (for
    payload encryption), the endpoint URL to submit work to, and
    the estimated cost.
    """

    provider_id: str
    """Stable provider identifier (e.g. 'akash', 'gcp', 'render')."""

    provider_public_key: bytes
    """32-byte X25519 public key for encrypting the workload payload."""

    provider_endpoint: str
    """URL or address to submit the encrypted workload to."""

    estimated_cost_usd: float = 0.0
    """Estimated total cost for this placement (from the cost model)."""

    offer_id: str = ""
    """The SubstrateOffer.id that was selected."""

    model_config = {"arbitrary_types_allowed": True}


# ─── Configuration ──────────────────────────────────────────────


class RemoteExecutionConfig(EOSBaseModel):
    """Configuration for the RemoteExecutionManager."""

    # Canary injection
    canary_count: int = 10
    canary_ratio: float = 0.05   # fallback: 5% of batch size if canary_count=0

    # Deterministic replay
    max_defect_rate: float = 0.01
    replay_alpha: float = 0.05
    replay_min_samples: int = 1
    replay_tolerance: ToleranceSpec = Field(
        default_factory=lambda: ToleranceSpec(match_kind=MatchKind.EXACT),
    )

    # Consensus weights
    consensus_weights: ConsensusWeights = Field(
        default_factory=ConsensusWeights,
    )

    # Dispatch
    dispatch_timeout_s: float = 300.0  # 5 minutes
    max_retries: int = 1               # retry on different provider
    retry_delay_s: float = 2.0

    # Encryption
    encrypt_aad: bool = True  # include workload_id as AAD


# ─── Execution Lifecycle Types ──────────────────────────────────


class ExecutionPhase(enum.StrEnum):
    PREPARE = "prepare"
    ENCRYPT = "encrypt"
    DISPATCH = "dispatch"
    RECEIVE = "receive"
    VERIFY = "verify"
    COMPLETE = "complete"
    FAILED = "failed"


class RemoteExecutionResult(Identified, Timestamped):
    """Final result of a remote execution lifecycle."""

    workload_id: str = ""
    provider_id: str = ""
    phase: ExecutionPhase = ExecutionPhase.COMPLETE
    accepted: bool = False
    results: list[bytes] = Field(default_factory=list)
    consensus_report: ConsensusReport | None = None
    encryption_meta: EncryptionMeta | None = None
    total_duration_ms: float = 0.0
    batch_size: int = 0
    canary_count: int = 0
    error: str = ""


# ─── Provider Transport ABC ────────────────────────────────────


class RemoteProviderTransport(ABC):
    """
    Interface for communicating with a remote compute provider.

    Implementations handle the actual network transport (HTTP, gRPC,
    Akash SDL deployment, etc.).  The RemoteExecutionManager is
    transport-agnostic.
    """

    @abstractmethod
    async def submit_workload(
        self,
        provider_id: str,
        endpoint: str,
        encrypted_payload: bytes,
        metadata: dict[str, str],
    ) -> bytes:
        """
        Submit an encrypted workload batch and return the encrypted
        result batch.

        The provider processes the items and returns encrypted results
        in the same order.
        """
        ...


# ─── Orchestrator ───────────────────────────────────────────────


class RemoteExecutionManager:
    """
    Orchestrates the full remote execution lifecycle.

    Ties together encryption, canary injection, remote dispatch,
    and consensus verification into a single coherent pipeline.

    Usage:
        manager = RemoteExecutionManager(
            config=RemoteExecutionConfig(),
            transport=my_transport,
            canary_generator=my_canary_gen,
            local_executor=my_local_exec,
        )
        result = await manager.execute(workload, placement)
        if result.accepted:
            process(result.results)
    """

    def __init__(
        self,
        config: RemoteExecutionConfig,
        transport: RemoteProviderTransport,
        canary_generator: Callable[[int], Awaitable[list[bytes]]],
        local_executor: Callable[[bytes], Awaitable[bytes]],
        shared_consensus_verifier: ConsensusVerifier | None = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._canary_gen = canary_generator
        self._local_exec = local_executor
        self._shared_verifier = shared_consensus_verifier
        self._log = logger.bind(component="sacm.remote_executor")

    async def execute(
        self,
        workload: WorkloadDescriptor,
        placement: PlacementDecision,
    ) -> RemoteExecutionResult:
        """
        Execute a workload on a remote provider with full encryption
        and verification.

        Returns RemoteExecutionResult with:
          - accepted=True + results: real output bytes in original order
          - accepted=False + error: explanation of failure
        """
        t0 = time.monotonic()
        execution_id = new_id()

        self._log.info(
            "remote_execution_start",
            execution_id=execution_id,
            workload_id=workload.workload_id,
            provider_id=placement.provider_id,
            item_count=workload.item_count,
        )

        try:
            return await self._execute_pipeline(
                workload, placement, execution_id, t0,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            self._log.error(
                "remote_execution_unhandled",
                execution_id=execution_id,
                error=str(exc),
                exc_info=True,
            )
            return RemoteExecutionResult(
                id=execution_id,
                workload_id=workload.workload_id,
                provider_id=placement.provider_id,
                phase=ExecutionPhase.FAILED,
                accepted=False,
                total_duration_ms=duration_ms,
                batch_size=workload.item_count,
                error=f"unhandled error: {exc}",
            )

    async def _execute_pipeline(
        self,
        workload: WorkloadDescriptor,
        placement: PlacementDecision,
        execution_id: str,
        t0: float,
    ) -> RemoteExecutionResult:
        """Internal pipeline - called by execute(), never raises to caller."""
        real_inputs = list(workload.items)

        # ── Phase 1: PREPARE - canary generation + batch mixing ──

        canary_count = self._config.canary_count
        if canary_count == 0:
            canary_count = max(1, int(len(real_inputs) * self._config.canary_ratio))

        preparer = CanaryPreparer(
            canary_generator=self._canary_gen,
            local_executor=self._local_exec,
        )
        canaries = await preparer.prepare(canary_count)

        mixed_batch = mix_canaries_into_batch(real_inputs, canaries)

        self._log.info(
            "batch_prepared",
            execution_id=execution_id,
            real_items=len(real_inputs),
            canaries=len(canaries),
            mixed_total=mixed_batch.total_size,
        )

        # ── Phase 2: ENCRYPT - seal the mixed batch ─────────────

        provider_public = public_key_from_bytes(placement.provider_public_key)
        aad = (
            workload.workload_id.encode()
            if self._config.encrypt_aad
            else None
        )

        # Generate our ephemeral keypair for this workload exchange.
        # The provider needs our public key to encrypt their response back to us.
        # We MUST use this same private key in Phase 4 to decrypt that response.
        # Never generate a new keypair for decryption.
        our_keypair = generate_keypair()

        serialized_batch = self._serialize_batch(mixed_batch.items)
        encryption_result = encrypt_payload(
            plaintext=serialized_batch,
            recipient_public=provider_public,
            aad=aad,
        )

        self._log.info(
            "batch_encrypted",
            execution_id=execution_id,
            plaintext_bytes=encryption_result.meta.plaintext_size_bytes,
            ciphertext_bytes=encryption_result.meta.ciphertext_size_bytes,
        )

        # ── Phase 3: DISPATCH - send to remote provider ─────────

        encrypted_result_bytes = await asyncio.wait_for(
            self._transport.submit_workload(
                provider_id=placement.provider_id,
                endpoint=placement.provider_endpoint,
                encrypted_payload=self._envelope_to_wire(encryption_result.envelope),
                metadata={
                    "workload_id": workload.workload_id,
                    "execution_id": execution_id,
                    "item_count": str(mixed_batch.total_size),
                    # Pass our public key so the provider can encrypt their
                    # response back to us using ECDH (X25519).
                    "response_public_key": our_keypair.public_bytes.hex(),
                },
            ),
            timeout=self._config.dispatch_timeout_s,
        )

        self._log.info(
            "results_received",
            execution_id=execution_id,
            result_bytes=len(encrypted_result_bytes),
        )

        # ── Phase 4: RECEIVE - decrypt result batch ──────────────
        # Use the same keypair from Phase 2 - our_keypair.private_key matches
        # the public key we advertised to the provider for their response encryption.

        result_envelope = self._wire_to_envelope(encrypted_result_bytes)

        decrypted_results = decrypt_payload(
            envelope=result_envelope,
            recipient_private=our_keypair.private_key,
            aad=aad,
        )
        remote_results = self._deserialize_batch(decrypted_results)

        if len(remote_results) != mixed_batch.total_size:
            return RemoteExecutionResult(
                id=execution_id,
                workload_id=workload.workload_id,
                provider_id=placement.provider_id,
                phase=ExecutionPhase.RECEIVE,
                accepted=False,
                total_duration_ms=(time.monotonic() - t0) * 1000,
                batch_size=len(real_inputs),
                canary_count=len(canaries),
                error=(
                    f"result count mismatch: expected {mixed_batch.total_size}, "
                    f"got {len(remote_results)}"
                ),
            )

        # ── Phase 5: VERIFY - consensus verification ────────────

        replay_items = self._build_replay_items(mixed_batch, remote_results)

        det_verifier = DeterministicReplayVerifier(
            replay_fn=self._local_exec,
            tolerance=self._config.replay_tolerance,
            max_defect_rate=self._config.max_defect_rate,
            alpha=self._config.replay_alpha,
            min_samples=self._config.replay_min_samples,
        )
        prob_verifier = ProbabilisticAuditVerifier(
            seal_key=preparer.seal_key,
        )
        if self._shared_verifier is not None:
            # Re-use the shared verifier's trust store; swap per-execution sub-verifiers
            self._shared_verifier._det = det_verifier
            self._shared_verifier._prob = prob_verifier
            consensus = self._shared_verifier
        else:
            consensus = ConsensusVerifier(
                deterministic=det_verifier,
                probabilistic=prob_verifier,
                weights=self._config.consensus_weights,
            )

        consensus_report = await consensus.verify(
            replay_items=replay_items,
            canaries=canaries,
            remote_results=remote_results,
            provider_id=placement.provider_id,
        )

        # ── Phase 6: ACCEPT/REJECT ──────────────────────────────

        duration_ms = (time.monotonic() - t0) * 1000
        accepted = consensus_report.outcome in (
            ConsensusOutcome.ACCEPTED,
            ConsensusOutcome.DEGRADED_ACCEPT,
        )

        if accepted:
            real_results = extract_real_results(remote_results, mixed_batch)
            self._log.info(
                "execution_accepted",
                execution_id=execution_id,
                provider_id=placement.provider_id,
                outcome=consensus_report.outcome.value,
                duration_ms=round(duration_ms, 1),
            )
        else:
            real_results = []
            self._log.warning(
                "execution_rejected",
                execution_id=execution_id,
                provider_id=placement.provider_id,
                outcome=consensus_report.outcome.value,
                error=consensus_report.error,
                duration_ms=round(duration_ms, 1),
            )

        return RemoteExecutionResult(
            id=execution_id,
            workload_id=workload.workload_id,
            provider_id=placement.provider_id,
            phase=ExecutionPhase.COMPLETE if accepted else ExecutionPhase.FAILED,
            accepted=accepted,
            results=real_results,
            consensus_report=consensus_report,
            encryption_meta=encryption_result.meta,
            total_duration_ms=duration_ms,
            batch_size=len(real_inputs),
            canary_count=len(canaries),
            error="" if accepted else consensus_report.error,
        )

    # ── Serialisation ───────────────────────────────────────────

    @staticmethod
    def _serialize_batch(items: Sequence[bytes]) -> bytes:
        """
        Serialize a list of byte items into a single payload.

        Format: [4-byte big-endian length][item bytes] repeated.
        Prefixed with a 4-byte item count.
        """
        parts: list[bytes] = []
        parts.append(len(items).to_bytes(4, "big"))
        for item in items:
            parts.append(len(item).to_bytes(4, "big"))
            parts.append(item)
        return b"".join(parts)

    @staticmethod
    def _deserialize_batch(data: bytes) -> list[bytes]:
        """Deserialize a batch payload back into individual items."""
        if len(data) < 4:
            return []
        count = int.from_bytes(data[:4], "big")
        items: list[bytes] = []
        offset = 4
        for _ in range(count):
            if offset + 4 > len(data):
                break
            item_len = int.from_bytes(data[offset : offset + 4], "big")
            offset += 4
            if offset + item_len > len(data):
                break
            items.append(data[offset : offset + item_len])
            offset += item_len
        return items

    @staticmethod
    def _envelope_to_wire(envelope: EncryptedEnvelope) -> bytes:
        """
        Serialize an EncryptedEnvelope to wire bytes.

        Format:
          [32 bytes sender_public_key]
          [12 bytes nonce]
          [remaining: ciphertext]
        """
        return envelope.sender_public_key + envelope.nonce + envelope.ciphertext

    @staticmethod
    def _wire_to_envelope(data: bytes) -> EncryptedEnvelope:
        """Deserialize wire bytes back into an EncryptedEnvelope."""
        if len(data) < 44:  # 32 (key) + 12 (nonce) + 0 (min ciphertext)
            raise ValueError(
                f"Wire data too short for envelope: {len(data)} bytes "
                f"(minimum 44)"
            )
        return EncryptedEnvelope(
            sender_public_key=data[:32],
            nonce=data[32:44],
            ciphertext=data[44:],
        )

    # ── Replay Item Construction ────────────────────────────────

    @staticmethod
    def _build_replay_items(
        mixed_batch: MixedBatch,
        remote_results: Sequence[bytes],
    ) -> list[ReplayItem]:
        """
        Build ReplayItems from real (non-canary) positions
        in the mixed batch.

        These are used by the DeterministicReplayVerifier to
        select a sample and compare remote vs local execution.
        """
        replay_items: list[ReplayItem] = []
        for pos in mixed_batch.real_positions:
            if pos < len(mixed_batch.items) and pos < len(remote_results):
                input_data = mixed_batch.items[pos]
                replay_items.append(
                    ReplayItem(
                        input_data=input_data,
                        remote_output=remote_results[pos],
                        input_hash=hashlib.sha256(input_data).hexdigest(),
                    )
                )
        return replay_items
