"""
SACM Pre-Warming & Partition Configuration.

Central configuration for the pre-warming engine and partition recovery
subsystems. All thresholds, budget limits, and timing constants live here
so operators can tune behaviour from a single module.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from primitives.common import EOSBaseModel
from systems.sacm.workload import OffloadClass

# ── Pre-Warming Budget ─────────────────────────────────────────────────


class PreWarmBudget(EOSBaseModel):
    """Budget caps for the pre-warming pool.

    0 = unlimited (metabolism gates cost, not hard caps).
    Set non-zero values via config to restrict.
    """

    max_pre_warm_budget_usd_per_hour: float = Field(
        default=0.0,
        ge=0.0,
        description="0 = unlimited. Maximum hourly spend across all warm instances.",
    )
    max_warm_instances: int = Field(
        default=0,
        ge=0,
        description="0 = unlimited. Hard cap on total warm instances across all classes.",
    )
    max_warm_instances_per_class: int = Field(
        default=0,
        ge=0,
        description="0 = unlimited. Per-offload-class cap on warm instances.",
    )
    max_single_instance_cost_usd_per_hour: float = Field(
        default=0.0,
        ge=0.0,
        description="0 = unlimited. Reject any single instance costing more than this per hour.",
    )


# ── Pre-Warming Timing ────────────────────────────────────────────────


class PreWarmTiming(EOSBaseModel):
    """Intervals and TTLs that govern the pre-warm loop."""

    loop_interval_s: float = Field(
        default=30.0,
        gt=0.0,
        description="How often the pre-warm loop re-evaluates the pool.",
    )
    warm_instance_ttl_s: float = Field(
        default=300.0,
        gt=0.0,
        description=(
            "Seconds a warm instance stays alive without being claimed. "
            "After this the engine releases it to save cost."
        ),
    )
    price_check_interval_s: float = Field(
        default=60.0,
        gt=0.0,
        description="How often the engine polls the oracle for price opportunities.",
    )
    prediction_horizon_s: float = Field(
        default=600.0,
        gt=0.0,
        description="How far ahead the workload predictor looks (seconds).",
    )


# ── Workload Prediction ──────────────────────────────────────────────


class PredictionConfig(EOSBaseModel):
    """Tuning knobs for the exponential-moving-average workload predictor."""

    ema_alpha: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description=(
            "Smoothing factor for the EMA predictor. "
            "Higher = more weight on recent observations."
        ),
    )
    min_history_samples: int = Field(
        default=5,
        ge=1,
        description="Minimum observations before the predictor produces a forecast.",
    )
    burst_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        description=(
            "Multiplier applied to predicted demand when a burst is detected "
            "(rolling stddev > 2× rolling mean)."
        ),
    )
    burst_stddev_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="StdDev-to-mean ratio above which we declare a burst.",
    )
    default_classes: list[OffloadClass] = Field(
        default_factory=lambda: [OffloadClass.GENERAL, OffloadClass.GPU_HEAVY],
        description="Offload classes the predictor tracks by default.",
    )


# ── Price Opportunity ─────────────────────────────────────────────────


class PriceOpportunityConfig(EOSBaseModel):
    """Controls when the engine pre-positions capacity ahead of a price rise."""

    price_increase_threshold_pct: float = Field(
        default=15.0,
        gt=0.0,
        le=100.0,
        description=(
            "Predicted price increase percentage that triggers pre-positioning. "
            "E.g. 15.0 means 'act when the oracle forecasts a ≥15 % rise'."
        ),
    )
    opportunity_budget_fraction: float = Field(
        default=0.4,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of max_pre_warm_budget_usd_per_hour that may be spent "
            "specifically on price-opportunity pre-positioning."
        ),
    )
    max_opportunity_instances: int = Field(
        default=3,
        ge=1,
        description="Most instances to spin up in a single opportunity window.",
    )


# ── Partition Recovery ────────────────────────────────────────────────


class PartitionRecoveryConfig(EOSBaseModel):
    """Thresholds that drive the PartitionRecoveryPolicy decision tree."""

    wait_budget_s: float = Field(
        default=30.0,
        ge=0.0,
        description=(
            "If the remaining deadline is more than this many seconds past "
            "the estimated reprovision time, the policy will wait for the "
            "original provider to come back."
        ),
    )
    reprovision_overhead_s: float = Field(
        default=45.0,
        ge=0.0,
        description="Estimated wall-clock cost of reprovisioning on a new provider.",
    )
    local_fallback_max_duration_s: float = Field(
        default=120.0,
        ge=0.0,
        description=(
            "Only attempt local fallback if the workload's estimated "
            "duration is below this threshold."
        ),
    )
    abort_grace_s: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "If the remaining deadline is less than this, abort immediately "
            "rather than attempting recovery."
        ),
    )
    max_reprovision_attempts: int = Field(
        default=2,
        ge=1,
        description="Maximum reprovision attempts before escalating to local/abort.",
    )


# ── Aggregate Config ──────────────────────────────────────────────────


class SACMPreWarmConfig(EOSBaseModel):
    """Top-level configuration consumed by PreWarmingEngine and PartitionRecoveryPolicy."""

    budget: PreWarmBudget = Field(default_factory=PreWarmBudget)
    timing: PreWarmTiming = Field(default_factory=PreWarmTiming)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    price_opportunity: PriceOpportunityConfig = Field(
        default_factory=PriceOpportunityConfig,
    )
    partition_recovery: PartitionRecoveryConfig = Field(
        default_factory=PartitionRecoveryConfig,
    )

    @field_validator("budget", mode="after")
    @classmethod
    def _instance_cost_within_hourly(cls, v: PreWarmBudget) -> PreWarmBudget:
        # 0 = unlimited — skip validation when either is unlimited
        if (
            v.max_single_instance_cost_usd_per_hour > 0
            and v.max_pre_warm_budget_usd_per_hour > 0
            and v.max_single_instance_cost_usd_per_hour > v.max_pre_warm_budget_usd_per_hour
        ):
            msg = (
                "max_single_instance_cost_usd_per_hour "
                f"({v.max_single_instance_cost_usd_per_hour}) must not exceed "
                f"max_pre_warm_budget_usd_per_hour ({v.max_pre_warm_budget_usd_per_hour})"
            )
            raise ValueError(msg)
        return v
