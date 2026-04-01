"""
EcodiaOS - Fast-Path Primitives

Types for the Arbitrage Reflex Arc: a sub-200ms execution channel that
bypasses Nova's full deliberation for pre-approved constitutional templates.

Design rationale:
  DeFi opportunities decay in milliseconds. The standard cycle
  (Atune → Nova → Equor → Axon) takes 1-15 seconds. For templated
  strategies that Equor has pre-approved as constitutionally safe,
  we skip Nova's policy generation and Equor's full review, routing
  directly from Atune's pattern detection to Axon's executor.

Trust model:
  ConstitutionalTemplate captures Equor's pre-approval: a specific
  pattern signature, confidence threshold, and capital limit. FastPathIntent
  is NOT a Policy (Nova's output) - it is a lightweight execution directive
  that carries the template's pre-approved trust.

Latency budget:
  Pattern detection:    ≤30ms  (hash + lookup)
  Template matching:    ≤20ms  (signature comparison)
  FastPath execution:   ≤150ms (on-chain call)
  Total:               ≤200ms
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from primitives.common import (
    EOSBaseModel,
    Identified,
    Timestamped,
    new_id,
    utc_now,
)


class ConstitutionalTemplate(Identified, Timestamped):
    """
    A pre-approved execution strategy that Equor has deemed constitutionally safe.

    Templates are the output of Equor's deliberative review process - they encode
    trust so the reflex arc doesn't need to re-derive it under time pressure.

    Immutable after creation. Revocation is via deactivation, not mutation.
    """

    template_id: str  # e.g., "arb_triangular_uniswap"
    pattern_signature: dict[str, Any] = Field(default_factory=dict)
    """
    The market feature vector this template matches.
    Keys and types are strategy-specific:
      - {"spread_threshold": 0.5, "pools": ["USDC-ETH", "ETH-WBTC", "WBTC-USDC"]}
      - {"token_pair": "ETH-USDC", "min_spread_bps": 30}
    """
    max_capital_per_execution: float = 0.0
    """Risk ceiling in USD. Axon enforces this before deploying capital."""
    approval_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    """Equor's confidence that this strategy is constitutionally safe. Range: 0–1."""
    last_approved_at: datetime = Field(default_factory=utc_now)
    """When Equor last reviewed and approved this template."""

    # ── Operational state (mutable by the reflex arc) ──────────────
    active: bool = True
    """False when circuit breaker disables the template."""
    consecutive_failures: int = 0
    """Failure counter for the circuit breaker. Reset on success."""
    total_executions: int = 0
    total_capital_deployed: float = 0.0
    last_executed_at: datetime | None = None

    @property
    def signature_hash(self) -> str:
        """
        Deterministic hash of the pattern signature for O(1) lookup.
        Uses canonical JSON serialisation → SHA-256 → hex digest.
        """
        canonical = json.dumps(self.pattern_signature, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def matches(self, candidate_signature: dict[str, Any], tolerance: float = 0.1) -> bool:
        """
        Approximate signature match. Checks:
          1. All template keys present in candidate
          2. Numeric values within tolerance (relative)
          3. List/set values overlap ≥ 80%
          4. String values match exactly
        """
        for key, template_val in self.pattern_signature.items():
            candidate_val = candidate_signature.get(key)
            if candidate_val is None:
                return False

            if isinstance(template_val, (int, float)) and isinstance(candidate_val, (int, float)):
                if template_val == 0:
                    if candidate_val != 0:
                        return False
                elif abs(candidate_val - template_val) / abs(template_val) > tolerance:
                    return False

            elif isinstance(template_val, list) and isinstance(candidate_val, list):
                template_set = set(str(v) for v in template_val)
                candidate_set = set(str(v) for v in candidate_val)
                if not template_set:
                    continue
                overlap = len(template_set & candidate_set) / len(template_set)
                if overlap < 0.8:
                    return False

            elif isinstance(template_val, str) and template_val != candidate_val:
                return False

        return True


class FastPathIntent(Identified, Timestamped):
    """
    A lightweight execution directive for the reflex arc.

    NOT a Policy (Nova's output). NOT an Intent (Nova + Equor's output).
    This is a pre-approved execution ticket that carries the template's trust
    directly to Axon without passing through Nova or Equor.

    Axon's fast_path_executor recognises this type and routes accordingly.
    """

    template_id: str
    """The ConstitutionalTemplate that authorised this execution."""
    pattern_signature: dict[str, Any] = Field(default_factory=dict)
    """The actual market feature vector that triggered the match."""
    executor_type: str = ""
    """The Axon executor to invoke (e.g., "defi_yield", "wallet_transfer")."""
    execution_params: dict[str, Any] = Field(default_factory=dict)
    """Parameters to pass to the executor."""
    max_capital: float = 0.0
    """Capital ceiling copied from the template. Enforced by Axon."""
    approval_confidence: float = 0.0
    """Copied from the template for audit trail."""
    source_percept_id: str = ""
    """The Atune percept that triggered this fast-path intent."""

    # ── Timing metadata ────────────────────────────────────────────
    percept_received_at: datetime = Field(default_factory=utc_now)
    """When Atune first received the market data percept."""
    template_matched_at: datetime | None = None
    """When the template library returned a match."""


class FastPathOutcome(EOSBaseModel):
    """
    Execution outcome for the fast-path reflex arc.

    Captures latency breakdown for performance monitoring and
    circuit breaker state for the template library.
    """

    intent_id: str
    template_id: str
    execution_id: str = Field(default_factory=new_id)
    success: bool = False
    error: str = ""

    # ── Latency breakdown ──────────────────────────────────────────
    total_latency_ms: int = 0
    pattern_match_ms: int = 0
    execution_ms: int = 0

    # ── Capital tracking ───────────────────────────────────────────
    capital_deployed: float = 0.0
    capital_returned: float = 0.0

    # ── Execution metadata ─────────────────────────────────────────
    executor_type: str = ""
    execution_data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)
