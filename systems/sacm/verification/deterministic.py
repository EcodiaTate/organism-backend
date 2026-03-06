"""
EcodiaOS — SACM Deterministic Replay Verifier

Verifies remote execution results by re-running a statistically
significant sample of the workload locally and comparing outputs.

The core idea: if a remote provider returns results for N items,
we re-execute a random sample of size `n` locally and check that
each local result matches the remote result within a configurable
tolerance.  If the match rate exceeds the acceptance threshold,
the full batch is accepted.

Sample size math:
  Uses the hypergeometric-approximation formula for attribute
  sampling (ISO 2859-style).  Given:
    - N = batch size (total items sent to remote)
    - p = maximum tolerable defect rate (e.g. 0.01 = 1%)
    - α = false-positive risk (probability of accepting a bad batch)

  Sample size n ≈ ceil( ln(α) / ln(1 − p) )

  This is the "zero-defect" plan: if we observe zero mismatches in
  n samples and the true defect rate is ≥ p, the probability of
  seeing zero defects is ≤ α.  For finite populations (N < 10×n)
  we apply a finite-population correction:

    n_corrected = ceil( n / (1 + (n − 1) / N) )

Tolerance matching:
  - For numeric outputs: |remote − local| ≤ atol + rtol × |local|
    (numpy-style allclose semantics)
  - For bytes/string outputs: exact equality after normalisation
  - For structured outputs: recursive field comparison with per-field
    tolerance overrides
"""

from __future__ import annotations

import enum
import hashlib
import math
import random
from collections.abc import Sequence
from typing import Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped, new_id

logger = structlog.get_logger("systems.sacm.verification.deterministic")


# ─── Types ──────────────────────────────────────────────────────


class MatchKind(enum.StrEnum):
    """How two outputs should be compared."""

    EXACT = "exact"
    NUMERIC = "numeric"
    HASH = "hash"
    STRUCTURAL = "structural"


class ToleranceSpec(EOSBaseModel):
    """Tolerance parameters for numeric comparison."""

    atol: float = 1e-8          # absolute tolerance
    rtol: float = 1e-5          # relative tolerance
    match_kind: MatchKind = MatchKind.EXACT

    # For STRUCTURAL matching: per-field overrides
    # e.g. {"score": ToleranceSpec(atol=0.01, match_kind="numeric")}
    field_overrides: dict[str, Any] = Field(default_factory=dict)


class ReplayItem(EOSBaseModel):
    """A single input→output pair from the remote batch."""

    item_id: str = Field(default_factory=new_id)
    input_data: bytes = b""
    remote_output: bytes = b""
    input_hash: str = ""       # SHA-256 of input_data for dedup


class ReplayResult(Identified, Timestamped):
    """Result of replaying a single item locally."""

    item_id: str
    local_output: bytes = b""
    matched: bool = False
    mismatch_detail: str = ""


class VerificationReport(Identified, Timestamped):
    """Aggregate report from a deterministic replay verification run."""

    batch_size: int = 0
    sample_size: int = 0
    matches: int = 0
    mismatches: int = 0
    match_rate: float = 0.0
    accepted: bool = False
    defect_rate_threshold: float = 0.0
    confidence_alpha: float = 0.0
    tolerance: ToleranceSpec = Field(default_factory=ToleranceSpec)
    mismatch_item_ids: list[str] = Field(default_factory=list)
    error: str = ""


# ─── Sample Size Calculation ────────────────────────────────────


def compute_sample_size(
    batch_size: int,
    max_defect_rate: float = 0.01,
    alpha: float = 0.05,
    min_samples: int = 1,
    max_samples: int | None = None,
) -> int:
    """
    Compute the number of items to replay locally.

    Uses the zero-defect acceptance sampling formula with
    finite-population correction.

    Args:
        batch_size:      Total items in the remote batch (N).
        max_defect_rate: Maximum tolerable defect fraction (p).
                         0.01 = accept if ≤1% defective.
        alpha:           False-positive risk.  0.05 = 5% chance of
                         accepting a batch with defect rate ≥ p.
        min_samples:     Floor on sample size (≥1).
        max_samples:     Ceiling on sample size (defaults to batch_size).

    Returns:
        Number of items to sample and replay.
    """
    if batch_size <= 0:
        return 0
    if max_defect_rate <= 0.0 or max_defect_rate >= 1.0:
        raise ValueError(f"max_defect_rate must be in (0, 1), got {max_defect_rate}")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Infinite-population sample size (zero-defect plan)
    # P(0 defects | n samples, defect rate p) = (1-p)^n ≤ α
    # ⟹ n ≥ ln(α) / ln(1-p)
    n_infinite = math.ceil(math.log(alpha) / math.log(1.0 - max_defect_rate))

    # Finite-population correction (FPC)
    # n_corrected = n / (1 + (n-1)/N)
    if batch_size < 10 * n_infinite:
        n_corrected = math.ceil(
            n_infinite / (1.0 + (n_infinite - 1.0) / batch_size)
        )
    else:
        n_corrected = n_infinite

    # Clamp
    cap = max_samples if max_samples is not None else batch_size
    return max(min_samples, min(n_corrected, cap))


# ─── Output Comparison ──────────────────────────────────────────


def outputs_match(
    remote: bytes,
    local: bytes,
    tolerance: ToleranceSpec,
) -> tuple[bool, str]:
    """
    Compare a remote output to a locally replayed output.

    Returns (matched, detail) where detail explains any mismatch.
    """
    if tolerance.match_kind == MatchKind.EXACT:
        if remote == local:
            return True, ""
        return False, (
            f"exact mismatch: remote {len(remote)} bytes vs local {len(local)} bytes"
        )

    if tolerance.match_kind == MatchKind.HASH:
        rh = hashlib.sha256(remote).hexdigest()
        lh = hashlib.sha256(local).hexdigest()
        if rh == lh:
            return True, ""
        return False, f"hash mismatch: remote={rh[:16]}… local={lh[:16]}…"

    if tolerance.match_kind == MatchKind.NUMERIC:
        return _numeric_match(remote, local, tolerance)

    if tolerance.match_kind == MatchKind.STRUCTURAL:
        return _structural_match(remote, local, tolerance)

    return False, f"unknown match_kind: {tolerance.match_kind}"


def _numeric_match(
    remote: bytes,
    local: bytes,
    tolerance: ToleranceSpec,
) -> tuple[bool, str]:
    """Compare two byte-encoded floats with atol/rtol tolerance."""
    try:
        r_val = float(remote)
        l_val = float(local)
    except (ValueError, UnicodeDecodeError):
        return False, "cannot parse as float for numeric comparison"

    # numpy allclose semantics: |a-b| <= atol + rtol * |b|
    diff = abs(r_val - l_val)
    threshold = tolerance.atol + tolerance.rtol * abs(l_val)
    if diff <= threshold:
        return True, ""
    return False, (
        f"numeric mismatch: remote={r_val}, local={l_val}, "
        f"diff={diff:.2e}, threshold={threshold:.2e}"
    )


def _structural_match(
    remote: bytes,
    local: bytes,
    tolerance: ToleranceSpec,
) -> tuple[bool, str]:
    """
    Compare two JSON-encoded structures field by field.

    Falls back to exact comparison if JSON parsing fails.
    Per-field tolerance overrides can be specified in
    tolerance.field_overrides.
    """
    import orjson

    try:
        r_obj = orjson.loads(remote)
        l_obj = orjson.loads(local)
    except Exception:
        # Fall back to exact comparison
        if remote == local:
            return True, ""
        return False, "structural: JSON parse failed, exact mismatch"

    mismatches: list[str] = []
    _compare_recursive(r_obj, l_obj, tolerance, "", mismatches)

    if not mismatches:
        return True, ""
    return False, f"structural mismatches: {'; '.join(mismatches[:5])}"


def _compare_recursive(
    remote_val: Any,
    local_val: Any,
    tolerance: ToleranceSpec,
    path: str,
    mismatches: list[str],
) -> None:
    """Recursively compare two values, collecting mismatch descriptions."""
    # Check for per-field override
    field_name = path.rsplit(".", 1)[-1] if path else ""
    field_tol_raw = tolerance.field_overrides.get(field_name)
    if isinstance(field_tol_raw, dict):
        field_tol = ToleranceSpec(**field_tol_raw)
    elif isinstance(field_tol_raw, ToleranceSpec):
        field_tol = field_tol_raw
    else:
        field_tol = tolerance

    if isinstance(remote_val, dict) and isinstance(local_val, dict):
        all_keys = set(remote_val.keys()) | set(local_val.keys())
        for key in sorted(all_keys):
            sub_path = f"{path}.{key}" if path else key
            if key not in remote_val:
                mismatches.append(f"{sub_path}: missing in remote")
            elif key not in local_val:
                mismatches.append(f"{sub_path}: missing in local")
            else:
                _compare_recursive(
                    remote_val[key], local_val[key], field_tol, sub_path, mismatches,
                )
    elif isinstance(remote_val, list) and isinstance(local_val, list):
        if len(remote_val) != len(local_val):
            mismatches.append(
                f"{path}: list length remote={len(remote_val)} local={len(local_val)}"
            )
        else:
            for i, (rv, lv) in enumerate(zip(remote_val, local_val, strict=False)):
                _compare_recursive(rv, lv, field_tol, f"{path}[{i}]", mismatches)
    elif isinstance(remote_val, (int, float)) and isinstance(local_val, (int, float)):
        diff = abs(float(remote_val) - float(local_val))
        threshold = field_tol.atol + field_tol.rtol * abs(float(local_val))
        if diff > threshold:
            mismatches.append(
                f"{path}: {remote_val} vs {local_val} (diff={diff:.2e})"
            )
    elif remote_val != local_val:
        mismatches.append(f"{path}: {remote_val!r} vs {local_val!r}")


# ─── Verifier ───────────────────────────────────────────────────


class DeterministicReplayVerifier:
    """
    Verify remote execution by replaying a sample locally.

    Usage:
        verifier = DeterministicReplayVerifier(
            replay_fn=my_local_executor,
            tolerance=ToleranceSpec(match_kind=MatchKind.NUMERIC, rtol=1e-4),
        )
        report = await verifier.verify(batch_items)
        if report.accepted:
            # safe to use remote results
    """

    def __init__(
        self,
        replay_fn: ReplayFunction,
        tolerance: ToleranceSpec | None = None,
        max_defect_rate: float = 0.01,
        alpha: float = 0.05,
        min_samples: int = 1,
        max_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            replay_fn:       Callable(input_data: bytes) -> bytes that
                             re-executes a single workload item locally.
            tolerance:       How to compare remote vs local outputs.
            max_defect_rate: Maximum tolerable defect fraction for sample
                             size calculation.
            alpha:           Significance level for sample size calculation.
            min_samples:     Minimum items to replay regardless of math.
            max_samples:     Cap on sample size.
            seed:            RNG seed for reproducible sample selection.
        """
        self._replay_fn = replay_fn
        self._tolerance = tolerance or ToleranceSpec()
        self._max_defect_rate = max_defect_rate
        self._alpha = alpha
        self._min_samples = min_samples
        self._max_samples = max_samples
        self._rng = random.Random(seed)
        self._log = logger.bind(component="sacm.verification.deterministic")

    async def verify(
        self,
        items: Sequence[ReplayItem],
    ) -> VerificationReport:
        """
        Run deterministic replay verification on a batch.

        1. Compute sample size from batch size and statistical params.
        2. Randomly select sample items.
        3. Replay each selected item locally via replay_fn.
        4. Compare local output to remote output per tolerance spec.
        5. Accept if zero mismatches observed in the sample.

        Returns a VerificationReport with full audit trail.
        """
        batch_size = len(items)

        if batch_size == 0:
            return VerificationReport(
                accepted=True,
                error="empty batch — vacuously accepted",
            )

        # Step 1: sample size
        sample_size = compute_sample_size(
            batch_size=batch_size,
            max_defect_rate=self._max_defect_rate,
            alpha=self._alpha,
            min_samples=self._min_samples,
            max_samples=self._max_samples,
        )

        self._log.info(
            "replay_sample_computed",
            batch_size=batch_size,
            sample_size=sample_size,
            max_defect_rate=self._max_defect_rate,
            alpha=self._alpha,
        )

        # Step 2: random sample
        indices = list(range(batch_size))
        sampled_indices = sorted(self._rng.sample(indices, min(sample_size, batch_size)))
        sampled_items = [items[i] for i in sampled_indices]

        # Step 3+4: replay and compare
        results: list[ReplayResult] = []
        matches = 0
        mismatches = 0
        mismatch_ids: list[str] = []

        for item in sampled_items:
            try:
                local_output = await self._replay_fn(item.input_data)
            except Exception as exc:
                result = ReplayResult(
                    item_id=item.item_id,
                    matched=False,
                    mismatch_detail=f"replay_fn raised: {exc}",
                )
                results.append(result)
                mismatches += 1
                mismatch_ids.append(item.item_id)
                self._log.warning(
                    "replay_fn_error",
                    item_id=item.item_id,
                    error=str(exc),
                )
                continue

            matched, detail = outputs_match(
                item.remote_output, local_output, self._tolerance,
            )

            result = ReplayResult(
                item_id=item.item_id,
                local_output=local_output,
                matched=matched,
                mismatch_detail=detail,
            )
            results.append(result)

            if matched:
                matches += 1
            else:
                mismatches += 1
                mismatch_ids.append(item.item_id)
                self._log.warning(
                    "replay_mismatch",
                    item_id=item.item_id,
                    detail=detail,
                )

        # Step 5: accept if zero defects (zero-defect sampling plan)
        actual_sample = len(results)
        match_rate = matches / actual_sample if actual_sample > 0 else 0.0
        accepted = mismatches == 0

        report = VerificationReport(
            batch_size=batch_size,
            sample_size=actual_sample,
            matches=matches,
            mismatches=mismatches,
            match_rate=match_rate,
            accepted=accepted,
            defect_rate_threshold=self._max_defect_rate,
            confidence_alpha=self._alpha,
            tolerance=self._tolerance,
            mismatch_item_ids=mismatch_ids,
        )

        self._log.info(
            "replay_verification_complete",
            batch_size=batch_size,
            sample_size=actual_sample,
            matches=matches,
            mismatches=mismatches,
            accepted=accepted,
        )
        return report


# ─── Type alias for the replay callable ─────────────────────────

# The replay function takes raw input bytes and returns output bytes.
# It must be an async callable.
from collections.abc import Awaitable, Callable

ReplayFunction = Callable[[bytes], Awaitable[bytes]]
