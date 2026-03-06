"""
EcodiaOS — Logos: Integration Protocols

Clean protocol interfaces for systems that don't exist yet but will
consume Logos data. Protocol-based dependency injection ensures Logos
doesn't import future systems — they import these protocols instead.

Protocols exposed:
- FoveaPredictionProtocol: Fovea queries the world model for predictions
- TelosMetricsProtocol: Telos reads the intelligence ratio and compression stats
- OneirosCompressionHooks: Oneiros runs batch compression during sleep cycles
- KairosInvariantProtocol: Kairos feeds causal invariants back into the world model
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from systems.logos.types import (
        CompressionCycleReport,
        EmpiricalInvariant,
        ExperienceDelta,
        IntelligenceMetrics,
        Prediction,
        RawExperience,
        WorldModelUpdate,
    )


# ─── Fovea: Prediction Interface ────────────────────────────────


@runtime_checkable
class FoveaPredictionProtocol(Protocol):
    """
    Protocol for Fovea to query the world model for predictions.

    Fovea is the attentional spotlight — it needs to know what the
    world model expects so it can compute prediction errors and
    direct attention to surprising stimuli.
    """

    async def predict(self, context: dict[str, Any]) -> Prediction:
        """Generate a prediction for what will happen in this context."""
        ...

    def get_historical_accuracy(self, domain: str | None = None) -> float:
        """
        Historical prediction accuracy, optionally filtered by domain.
        Fovea uses this to calibrate how much to trust predictions.
        """
        ...

    def get_context_stability_age(self, context_key: str) -> float:
        """
        How long (seconds) a context's prior has been stable.
        Stable priors = confident predictions = less attention needed.
        """
        ...


# ─── Telos: Intelligence Metrics Interface ──────────────────────


@runtime_checkable
class TelosMetricsProtocol(Protocol):
    """
    Protocol for Telos to read the intelligence ratio and compression stats.

    Telos is the drive system — it modulates the intelligence ratio
    with drive multipliers to steer what the system prioritises learning.
    """

    def get_intelligence_ratio(self) -> float:
        """
        I = K(reality_modeled) / K(model)

        The primary AGI progress metric. Telos reads this to
        determine whether the system is getting smarter.
        """
        ...

    def get_compression_stats(self) -> dict[str, float]:
        """
        Compression statistics for Telos drive modulation.

        Returns dict with keys:
        - cognitive_pressure: 0-1 how full the budget is
        - compression_urgency: non-linear urgency signal
        - compression_efficiency: fraction of knowledge with MDL > 1.0
        - world_model_coverage: fraction of episodes predictable
        - world_model_complexity: bits of the model
        """
        ...

    def get_latest_metrics(self) -> IntelligenceMetrics:
        """Full intelligence metrics snapshot."""
        ...


# ─── Oneiros: Offline Compression Hooks ─────────────────────────


@runtime_checkable
class OneirosCompressionHooks(Protocol):
    """
    Protocol for Oneiros to run batch compression during sleep.

    During sleep (especially NREM), Oneiros triggers aggressive
    offline compression passes. This is analogous to memory
    consolidation during biological sleep.
    """

    async def run_batch_compression(
        self,
        *,
        force: bool = False,
        max_items: int = 100,
    ) -> CompressionCycleReport:
        """
        Run a batch compression cycle.

        During sleep, Oneiros calls this with force=True to bypass
        the normal pressure threshold check. max_items limits the
        scope of each pass (Oneiros calls this multiple times per
        sleep cycle with increasing aggressiveness).
        """
        ...

    async def encode_experience(
        self, raw_experience: RawExperience
    ) -> ExperienceDelta:
        """
        Holographically encode a raw experience.

        Oneiros may replay consolidated episodes through the encoder
        to check whether they're still surprising relative to the
        updated world model (dream replay = re-encoding check).
        """
        ...

    async def integrate_delta(self, delta: ExperienceDelta) -> WorldModelUpdate:
        """
        Integrate a delta into the world model.

        During NREM, Oneiros feeds accumulated deltas into the world
        model in bulk, allowing the model to absorb patterns that
        weren't integrated during waking due to time pressure.
        """
        ...


# ─── Kairos: Causal Invariant Ingestion ─────────────────────────


@runtime_checkable
class KairosInvariantProtocol(Protocol):
    """
    Protocol for Kairos to feed causal invariants back into the world model.

    Kairos discovers causal invariants through temporal analysis.
    These invariants are the most compressed knowledge: a single rule
    that covers infinite future instances.
    """

    def ingest_invariant(self, invariant: EmpiricalInvariant) -> None:
        """
        Ingest a causal invariant into the world model.

        Invariants are never-violated rules discovered by Kairos.
        They reduce the world model's complexity by replacing many
        specific observations with a single general rule.
        """
        ...
