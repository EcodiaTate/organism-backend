"""
EcodiaOS — SACM Probabilistic Audit Verifier

Verifies remote execution by injecting "canary" inputs with
pre-computed known-good outputs into the workload batch before
sending it to the remote provider.  After execution, the canary
outputs are checked against the expected values.

Why canaries:
  - The remote provider cannot distinguish canaries from real work.
  - If the provider returns correct answers for canaries, it
    provides probabilistic evidence that it executed honestly.
  - If any canary fails, the entire batch is rejected.

Canary preparation:
  1. Generate synthetic inputs that are indistinguishable from real
     inputs (same schema, size distribution, value ranges).
  2. Execute each canary locally to obtain the known-good output.
  3. Encrypt the expected outputs so they cannot leak to the provider
     (the provider never sees expected answers).
  4. Mix canary items into the real workload at random positions.

Verification:
  1. After receiving remote results, extract canary positions.
  2. Decrypt the expected outputs.
  3. Compare each canary's remote output to its expected output.
  4. If all canaries match: batch accepted.
  5. If any canary fails: batch rejected, flag provider.

Statistical model:
  If the provider cheats on fraction `q` of items randomly, the
  probability of all `k` canaries passing is (1-q)^k.
  For k=10 canaries and q=5% cheating: P(pass) = 0.95^10 ≈ 0.60.
  So 10 canaries catch a 5%-cheating provider ~40% of the time per
  batch.  Over multiple batches the detection compounds.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable, Sequence

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, new_id
from systems.sacm.encryption import (
    SymmetricSealedBox,
    generate_symmetric_key,
    symmetric_decrypt,
    symmetric_encrypt,
)

logger = structlog.get_logger("systems.sacm.verification.probabilistic")


# ─── Types ──────────────────────────────────────────────────────


class CanaryItem(EOSBaseModel):
    """
    A canary input with its sealed expected output.

    The `expected_output_sealed` is encrypted with a key held only
    by the verifier — the remote provider never sees the expected
    answer.  `position` is assigned after mixing into the batch.
    """

    canary_id: str = Field(default_factory=new_id)
    input_data: bytes = b""
    expected_output: bytes = b""              # plaintext — never sent to provider
    expected_output_sealed: SymmetricSealedBox | None = None
    position: int = -1                        # index in the mixed batch
    tag: str = ""                             # optional label for categorisation


class CanaryVerificationResult(EOSBaseModel):
    """Result of verifying a single canary."""

    canary_id: str
    position: int = -1
    passed: bool = False
    expected_hash: str = ""
    actual_hash: str = ""
    mismatch_detail: str = ""


class ProbabilisticAuditReport(Identified, Timestamped):
    """Aggregate report from a probabilistic canary audit."""

    total_canaries: int = 0
    passed_canaries: int = 0
    failed_canaries: int = 0
    accepted: bool = False
    detection_probability: float = 0.0        # P(catch cheater) at assumed cheat rate
    assumed_cheat_rate: float = 0.05
    canary_results: list[CanaryVerificationResult] = Field(default_factory=list)
    error: str = ""


# ─── Canary Generator Protocol ──────────────────────────────────

# A CanaryGenerator produces synthetic inputs indistinguishable
# from real workload inputs.  It's provided by the caller because
# only the workload owner knows the input schema.

CanaryGeneratorFn = Callable[[int], Awaitable[list[bytes]]]
"""Async callable: (count: int) -> list[bytes] of synthetic inputs."""

LocalExecutorFn = Callable[[bytes], Awaitable[bytes]]
"""Async callable: (input_data: bytes) -> output bytes."""


# ─── Canary Preparation ────────────────────────────────────────


class CanaryPreparer:
    """
    Prepares canary items for injection into a workload batch.

    1. Generates synthetic inputs via the canary_generator.
    2. Executes each locally to get known-good outputs.
    3. Seals expected outputs with a random symmetric key.
    """

    def __init__(
        self,
        canary_generator: CanaryGeneratorFn,
        local_executor: LocalExecutorFn,
    ) -> None:
        self._generate = canary_generator
        self._execute = local_executor
        self._seal_key: bytes = generate_symmetric_key()
        self._log = logger.bind(component="sacm.canary.preparer")

    @property
    def seal_key(self) -> bytes:
        """The symmetric key used to seal expected outputs."""
        return self._seal_key

    async def prepare(self, count: int) -> list[CanaryItem]:
        """
        Generate and prepare `count` canary items.

        Each item has:
          - input_data: synthetic input
          - expected_output: known-good output (computed locally)
          - expected_output_sealed: encrypted expected output
        """
        if count <= 0:
            return []

        # Step 1: generate synthetic inputs
        inputs = await self._generate(count)
        if len(inputs) < count:
            self._log.warning(
                "canary_generator_shortfall",
                requested=count,
                received=len(inputs),
            )

        # Step 2+3: execute locally and seal expected outputs
        canaries: list[CanaryItem] = []
        for i, input_data in enumerate(inputs):
            try:
                expected = await self._execute(input_data)
            except Exception as exc:
                self._log.error(
                    "canary_local_exec_failed",
                    index=i,
                    error=str(exc),
                )
                continue

            sealed = symmetric_encrypt(expected, self._seal_key)

            canary = CanaryItem(
                input_data=input_data,
                expected_output=expected,
                expected_output_sealed=sealed,
            )
            canaries.append(canary)

        self._log.info(
            "canaries_prepared",
            requested=count,
            prepared=len(canaries),
        )
        return canaries


# ─── Batch Mixing ───────────────────────────────────────────────


class MixedBatch(EOSBaseModel):
    """
    A workload batch with canaries mixed in at random positions.

    The `items` list contains both real and canary inputs in the
    mixed order.  `canary_positions` maps canary_id → index for
    extraction after remote execution.
    """

    items: list[bytes] = Field(default_factory=list)
    canary_positions: dict[str, int] = Field(default_factory=dict)
    real_positions: list[int] = Field(default_factory=list)
    total_size: int = 0


def mix_canaries_into_batch(
    real_inputs: Sequence[bytes],
    canaries: Sequence[CanaryItem],
    seed: int | None = None,
) -> MixedBatch:
    """
    Shuffle canary inputs into the real workload batch.

    Canaries are inserted at random positions so the remote
    provider cannot identify them by position.

    Returns a MixedBatch with the combined item list and
    position mappings for later extraction.
    """
    rng = secrets.SystemRandom() if seed is None else __import__("random").Random(seed)

    total = len(real_inputs) + len(canaries)
    if total == 0:
        return MixedBatch()

    # Assign random unique positions to canaries
    all_positions = list(range(total))
    rng.shuffle(all_positions)

    canary_pos_map: dict[str, int] = {}
    canary_data_map: dict[int, bytes] = {}

    for i, canary in enumerate(canaries):
        pos = all_positions[i]
        canary_pos_map[canary.canary_id] = pos
        canary_data_map[pos] = canary.input_data
        # Mutate the canary's position field
        canary.position = pos

    # Remaining positions go to real inputs
    real_positions_list: list[int] = sorted(all_positions[len(canaries):])
    real_data_map: dict[int, bytes] = {}
    for pos, data in zip(real_positions_list, real_inputs, strict=False):
        real_data_map[pos] = data

    # Build the mixed item list in order
    items: list[bytes] = []
    for pos in range(total):
        if pos in canary_data_map:
            items.append(canary_data_map[pos])
        elif pos in real_data_map:
            items.append(real_data_map[pos])
        else:
            # Should not happen, but defensive
            items.append(b"")

    return MixedBatch(
        items=items,
        canary_positions=canary_pos_map,
        real_positions=real_positions_list,
        total_size=total,
    )


# ─── Verifier ───────────────────────────────────────────────────


class ProbabilisticAuditVerifier:
    """
    Verify remote execution by checking canary outputs.

    Usage:
        preparer = CanaryPreparer(generator_fn, local_exec_fn)
        canaries = await preparer.prepare(count=10)
        mixed = mix_canaries_into_batch(real_inputs, canaries)

        # ... send mixed.items to remote provider, get results ...

        verifier = ProbabilisticAuditVerifier(seal_key=preparer.seal_key)
        report = await verifier.verify(canaries, remote_results)
        if report.accepted:
            real_results = extract_real_results(remote_results, mixed)
    """

    def __init__(
        self,
        seal_key: bytes,
        assumed_cheat_rate: float = 0.05,
    ) -> None:
        """
        Args:
            seal_key:           Key to unseal expected canary outputs.
            assumed_cheat_rate: Assumed fraction of items the provider
                                might cheat on.  Used to compute
                                detection_probability in the report.
        """
        self._seal_key = seal_key
        self._assumed_cheat_rate = assumed_cheat_rate
        self._log = logger.bind(component="sacm.verification.probabilistic")

    async def verify(
        self,
        canaries: Sequence[CanaryItem],
        remote_results: Sequence[bytes],
    ) -> ProbabilisticAuditReport:
        """
        Check that remote results at canary positions match expected outputs.

        Args:
            canaries:       The canary items (with .position set).
            remote_results: Full result list from the remote provider
                            (same length as the mixed batch).

        Returns:
            ProbabilisticAuditReport with per-canary results.
        """
        if not canaries:
            return ProbabilisticAuditReport(
                accepted=True,
                error="no canaries — vacuously accepted",
            )

        results: list[CanaryVerificationResult] = []
        passed = 0
        failed = 0

        for canary in canaries:
            result = self._verify_single(canary, remote_results)
            results.append(result)
            if result.passed:
                passed += 1
            else:
                failed += 1
                self._log.warning(
                    "canary_failed",
                    canary_id=canary.canary_id,
                    position=canary.position,
                    detail=result.mismatch_detail,
                )

        # Detection probability: P(catch) = 1 - (1-q)^k
        k = len(canaries)
        q = self._assumed_cheat_rate
        detection_prob = 1.0 - ((1.0 - q) ** k)

        accepted = failed == 0

        report = ProbabilisticAuditReport(
            total_canaries=k,
            passed_canaries=passed,
            failed_canaries=failed,
            accepted=accepted,
            detection_probability=detection_prob,
            assumed_cheat_rate=q,
            canary_results=results,
        )

        self._log.info(
            "probabilistic_audit_complete",
            total=k,
            passed=passed,
            failed=failed,
            accepted=accepted,
            detection_prob=round(detection_prob, 4),
        )
        return report

    def _verify_single(
        self,
        canary: CanaryItem,
        remote_results: Sequence[bytes],
    ) -> CanaryVerificationResult:
        """Check a single canary against its expected output."""
        import hashlib

        if canary.position < 0 or canary.position >= len(remote_results):
            return CanaryVerificationResult(
                canary_id=canary.canary_id,
                position=canary.position,
                passed=False,
                mismatch_detail=(
                    f"canary position {canary.position} out of range "
                    f"(batch size {len(remote_results)})"
                ),
            )

        remote_output = remote_results[canary.position]

        # Unseal the expected output
        if canary.expected_output_sealed is not None:
            try:
                expected = symmetric_decrypt(
                    canary.expected_output_sealed, self._seal_key,
                )
            except Exception as exc:
                return CanaryVerificationResult(
                    canary_id=canary.canary_id,
                    position=canary.position,
                    passed=False,
                    mismatch_detail=f"failed to unseal expected output: {exc}",
                )
        else:
            # Fallback: use the plaintext expected_output
            expected = canary.expected_output

        expected_hash = hashlib.sha256(expected).hexdigest()[:16]
        actual_hash = hashlib.sha256(remote_output).hexdigest()[:16]

        if remote_output == expected:
            return CanaryVerificationResult(
                canary_id=canary.canary_id,
                position=canary.position,
                passed=True,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
            )

        return CanaryVerificationResult(
            canary_id=canary.canary_id,
            position=canary.position,
            passed=False,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
            mismatch_detail=(
                f"output mismatch: expected {len(expected)} bytes "
                f"(sha={expected_hash}…), got {len(remote_output)} bytes "
                f"(sha={actual_hash}…)"
            ),
        )


# ─── Utility: extract real results from mixed batch ─────────────


def extract_real_results(
    remote_results: Sequence[bytes],
    mixed_batch: MixedBatch,
) -> list[bytes]:
    """
    Extract only the real (non-canary) results from the remote
    output, preserving original order.
    """
    return [remote_results[pos] for pos in mixed_batch.real_positions]
