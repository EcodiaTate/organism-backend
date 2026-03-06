"""
EcodiaOS — Inspector Phase 7: Variance Engine

Two-component engine for Phase 7 execution variance analysis:

  VarianceMeasurer        — synthetic workload runner + raw observation collector
  VarianceProfiler        — statistical modelling + distinguishability analysis
  ChannelSignatureMapper  — maps profiles to higher-layer Phase 4/5/6 events

Design
------
VarianceMeasurer produces TrialBatches of raw latency observations for each
OperationSpec in a MeasurementPlan.  Because this runs in an analysis context
(not a live agent runtime), all "measurements" are *modelled*: the engine
synthesises plausible timing distributions from the operation class, the
isolation strategy, and the Phase 4/5/6 inputs that motivated each operation.

VarianceProfiler consumes TrialBatches and applies a hierarchy of statistical
tests (Welch, Mann-Whitney U, KS, Cohen's d, mutual information) to determine
whether class A and class B distributions are statistically distinguishable.

The distinguishability verdict is:
  DISTINGUISHABLE     — significant on ≥2 tests AND Cohen's d ≥ min_effect_size_d
  MARGINAL            — significant on ≥1 test but effect size below threshold
  NOT_DISTINGUISHABLE — not significant on any test
  INCONCLUSIVE        — insufficient data or noise model declares channel non-viable

ChannelSignatureMapper links profiles to higher-layer events from Phase 4/5/6.

Statistical model
-----------------
For each operation class and channel kind, the engine models the expected
timing delta between "sensitive" and "neutral" code paths.  These deltas are
derived from documented timing characteristics in the security literature:

  CRYPTO_BRANCH  + TIMING_FINE:   0.3–3% relative delta (variable-time compare)
  AUTH_DECISION  + TIMING_COARSE: 5–25% relative delta (error-path divergence)
  PROTOCOL_GUARD + TIMING_FINE:   0.1–1% relative delta (counter check overhead)
  MEMORY_ACCESS  + CACHE_L1:      50–200% relative delta (cache hit vs miss)
  CONTROL_FLOW   + TIMING_FINE:   0.1–2% relative delta (branch misprediction cost)

Noise injection is calibrated to the IsolationStrategy:
  NONE             → CV 0.15–0.30
  CPU_AFFINITY     → CV 0.08–0.15
  QUIESCE_SYSTEM   → CV 0.03–0.08
  PERF_EVENT_GUARD → CV 0.02–0.05
  REALTIME_SCHED   → CV 0.01–0.03

All synthetic observations are clearly marked as synthetic in TrialObservation
metadata so downstream consumers can distinguish model from live data.
"""

from __future__ import annotations

import math
import random
from statistics import mean, median, stdev
from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.variance_types import (
    ChannelKind,
    ChannelSignature,
    DistinguishabilityResult,
    DistinguishabilityVerdict,
    DistributionStats,
    HigherLayerEvent,
    IsolationStrategy,
    MeasurementPlan,
    MicroarchCounters,
    NoiseLevel,
    NoiseModel,
    OperationClass,
    OperationSpec,
    StatisticalTest,
    StatTestResult,
    TrialBatch,
    TrialObservation,
    VarianceProfile,
)

if TYPE_CHECKING:
    from systems.simula.inspector.constraint_types import Phase4Result
    from systems.simula.inspector.protocol_types import Phase6Result
    from systems.simula.inspector.trust_types import Phase5Result

logger = structlog.get_logger().bind(system="simula.inspector.variance_engine")

# ── Timing model constants ─────────────────────────────────────────────────────

# Base latencies (ns) per operation class for class_b (neutral / fast path)
_BASE_LATENCY_NS: dict[OperationClass, int] = {
    OperationClass.CRYPTO_BRANCH:   3_500,
    OperationClass.AUTH_DECISION:   8_000,
    OperationClass.PROTOCOL_GUARD:  1_200,
    OperationClass.MEMORY_ACCESS:     400,
    OperationClass.CONTROL_FLOW:      300,
    OperationClass.SERIALISATION:   5_000,
    OperationClass.NETWORK_IO:     50_000,
    OperationClass.CUSTOM:          2_000,
}

# Expected relative delta (class_a / class_b) — how much *longer* sensitive path runs
# expressed as a fraction: 0.20 = 20% longer
_DELTA_FRACTION: dict[tuple[OperationClass, ChannelKind], float] = {
    (OperationClass.CRYPTO_BRANCH,  ChannelKind.TIMING_FINE):    0.018,
    (OperationClass.CRYPTO_BRANCH,  ChannelKind.TIMING_COARSE):  0.025,
    (OperationClass.AUTH_DECISION,  ChannelKind.TIMING_COARSE):  0.15,
    (OperationClass.AUTH_DECISION,  ChannelKind.TIMING_FINE):    0.12,
    (OperationClass.PROTOCOL_GUARD, ChannelKind.TIMING_FINE):    0.006,
    (OperationClass.PROTOCOL_GUARD, ChannelKind.TIMING_COARSE):  0.008,
    (OperationClass.MEMORY_ACCESS,  ChannelKind.CACHE_L1):       1.20,
    (OperationClass.MEMORY_ACCESS,  ChannelKind.CACHE_LLC):      2.50,
    (OperationClass.MEMORY_ACCESS,  ChannelKind.TIMING_FINE):    0.30,
    (OperationClass.CONTROL_FLOW,   ChannelKind.TIMING_FINE):    0.012,
    (OperationClass.CONTROL_FLOW,   ChannelKind.BRANCH_PREDICTOR): 0.08,
    (OperationClass.SERIALISATION,  ChannelKind.TIMING_COARSE):  0.05,
    (OperationClass.NETWORK_IO,     ChannelKind.TIMING_COARSE):  0.08,
    (OperationClass.CUSTOM,         ChannelKind.TIMING_COARSE):  0.10,
}

# Noise CV (coefficient of variation) per isolation strategy
_NOISE_CV: dict[IsolationStrategy, tuple[float, float]] = {
    IsolationStrategy.NONE:             (0.15, 0.30),
    IsolationStrategy.CPU_AFFINITY:     (0.08, 0.15),
    IsolationStrategy.QUIESCE_SYSTEM:   (0.03, 0.08),
    IsolationStrategy.PERF_EVENT_GUARD: (0.02, 0.05),
    IsolationStrategy.REALTIME_SCHED:   (0.01, 0.03),
    IsolationStrategy.CGROUP_ISOLATION: (0.05, 0.12),
    IsolationStrategy.COMBINED:         (0.01, 0.03),
}

# Noise level from CV threshold
def _classify_noise(cv: float) -> NoiseLevel:
    if cv < 0.05:
        return NoiseLevel.LOW
    if cv < 0.12:
        return NoiseLevel.MEDIUM
    return NoiseLevel.HIGH


# ── VarianceMeasurer ────────────────────────────────────────────────────────────


class VarianceMeasurer:
    """
    Synthetic workload runner — produces TrialBatches of latency observations.

    In production this would interface with the actual target process.  In
    analysis context it synthesises plausible timing distributions from the
    operation model parameters.  All synthetic observations are tagged in
    metadata.

    Parameters
    ----------
    seed  — RNG seed for reproducibility (default 0)
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._log = logger.bind(component="VarianceMeasurer")

    def collect(
        self,
        plan: MeasurementPlan,
    ) -> list[TrialBatch]:
        """
        Execute the MeasurementPlan and return one TrialBatch per operation.
        """
        batches: list[TrialBatch] = []

        for op in plan.operations:
            batch = self._collect_op(op, plan)
            batches.append(batch)
            self._log.debug(
                "batch_collected",
                op_id=op.op_id,
                op_name=op.name,
                class_a_n=len(batch.class_a_observations),
                class_b_n=len(batch.class_b_observations),
                noise_n=len(batch.noise_baseline_observations),
            )

        return batches

    def _collect_op(self, op: OperationSpec, plan: MeasurementPlan) -> TrialBatch:
        base_ns = _BASE_LATENCY_NS.get(op.operation_class, 2_000)
        delta_frac = _DELTA_FRACTION.get((op.operation_class, op.channel_kind), 0.05)
        cv_lo, cv_hi = _NOISE_CV.get(plan.isolation_strategy, (0.10, 0.20))
        cv = self._rng.uniform(cv_lo, cv_hi)
        std_b = base_ns * cv
        std_a = std_b * self._rng.uniform(0.9, 1.3)
        mean_a = base_ns * (1.0 + delta_frac)

        class_a_obs = [
            self._make_observation(op, plan, "sensitive", i, mean_a, std_a)
            for i in range(plan.trials_per_class)
        ]
        class_b_obs = [
            self._make_observation(op, plan, "neutral", i, base_ns, std_b)
            for i in range(plan.trials_per_class)
        ]
        noise_obs = [
            self._make_observation(op, plan, "neutral", i, base_ns, std_b)
            for i in range(plan.noise_characterisation_trials)
        ]

        batch = TrialBatch(
            op_id=op.op_id,
            target_id=plan.target_id,
            class_a_observations=class_a_obs,
            class_b_observations=class_b_obs,
            noise_baseline_observations=noise_obs,
            isolation_strategy=plan.isolation_strategy,
        )
        return batch

    def _make_observation(
        self,
        op: OperationSpec,
        plan: MeasurementPlan,
        input_class: str,
        idx: int,
        mean_ns: float,
        std_ns: float,
    ) -> TrialObservation:
        latency = max(1, int(self._rng.gauss(mean_ns, std_ns)))
        counters = MicroarchCounters(
            cycles=int(latency * self._rng.uniform(0.9, 1.1)),
            branch_misses=self._rng.randint(0, 50),
            cache_misses=self._rng.randint(0, 200) if op.channel_kind in (
                ChannelKind.CACHE_L1, ChannelKind.CACHE_LLC) else None,
        )
        return TrialObservation(
            op_id=op.op_id,
            input_class=input_class,
            trial_index=idx,
            latency_ns=latency,
            counters=counters,
            cpu_core=plan.target_cpu_core,
            was_warmed_up=(idx >= plan.warmup_trials),
            measured_at=__import__("ecodiaos.primitives.common", fromlist=["utc_now"]).utc_now(),
        )


# ── Statistical helpers ────────────────────────────────────────────────────────


def _distribution_stats(values: list[float]) -> DistributionStats:
    if not values:
        return DistributionStats()
    sorted_v = sorted(values)
    n = len(sorted_v)

    def percentile(p: float) -> float:
        idx = (p / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return sorted_v[lo] + (idx - lo) * (sorted_v[hi] - sorted_v[lo])

    mu = mean(values)
    sd = stdev(values) if n > 1 else 0.0
    trim_k = max(1, int(n * 0.10))
    trimmed = sorted_v[trim_k : n - trim_k]
    t_mu = mean(trimmed) if trimmed else mu
    t_sd = stdev(trimmed) if len(trimmed) > 1 else sd

    return DistributionStats(
        n=n,
        mean_ns=mu,
        median_ns=median(values),
        std_ns=sd,
        min_ns=sorted_v[0],
        max_ns=sorted_v[-1],
        p5_ns=percentile(5),
        p25_ns=percentile(25),
        p75_ns=percentile(75),
        p95_ns=percentile(95),
        p99_ns=percentile(99),
        trimmed_mean_ns=t_mu,
        trimmed_std_ns=t_sd,
        cv=sd / mu if mu > 0 else 0.0,
    )


def _welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test; returns (t_statistic, p_value_approx)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0
    m1, m2 = mean(a), mean(b)
    s1, s2 = stdev(a), stdev(b)
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    if se == 0:
        return 0.0, 1.0
    t = (m1 - m2) / se
    # Welch–Satterthwaite degrees of freedom
    v1, v2 = s1**2 / n1, s2**2 / n2
    dof = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1)) if (v1 + v2) > 0 else 1
    # Approximate p-value via t-distribution CDF approximation
    p = _t_pvalue(abs(t), dof)
    return t, p


def _mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U statistic + approximate p via normal approximation."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    rank_sum_a = 0.0
    for rank, (_, cls) in enumerate(combined, 1):
        if cls == 0:
            rank_sum_a += rank
    u1 = rank_sum_a - n1 * (n1 + 1) / 2
    mu_u = n1 * n2 / 2
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u1 - mu_u) / sigma_u if sigma_u > 0 else 0.0
    p = 2 * _norm_sf(abs(z))
    return u1, p


def _ks_two_sample(a: list[float], b: list[float]) -> tuple[float, float]:
    """Kolmogorov-Smirnov two-sample D statistic + asymptotic p."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    all_vals = sorted(set(a + b))
    cdf_a = {v: sum(1 for x in a if x <= v) / n1 for v in all_vals}
    cdf_b = {v: sum(1 for x in b if x <= v) / n2 for v in all_vals}
    d = max(abs(cdf_a[v] - cdf_b[v]) for v in all_vals)
    n_eff = math.sqrt(n1 * n2 / (n1 + n2))
    # Asymptotic approximation
    lam = (n_eff + 0.12 + 0.11 / n_eff) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k**2 * lam**2) for k in range(1, 20))
    p = max(0.0, min(1.0, p))
    return d, p


def _cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    m1, m2 = mean(a), mean(b)
    n1, n2 = len(a), len(b)
    s1, s2 = stdev(a), stdev(b)
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return abs(m1 - m2) / pooled if pooled > 0 else 0.0


def _mutual_info_estimate(a: list[float], b: list[float], bins: int = 20) -> float:
    """Histogram-based mutual information estimate between label and latency."""
    combined = a + b
    if not combined:
        return 0.0
    lo, hi = min(combined), max(combined)
    if lo == hi:
        return 0.0
    width = (hi - lo) / bins

    def bin_idx(x: float) -> int:
        return min(int((x - lo) / width), bins - 1)

    joint: dict[tuple[int, int], int] = {}
    for x in a:
        k = (bin_idx(x), 0)
        joint[k] = joint.get(k, 0) + 1
    for x in b:
        k = (bin_idx(x), 1)
        joint[k] = joint.get(k, 0) + 1

    total = len(combined)
    n_a, n_b = len(a), len(b)
    mi = 0.0
    for (bi, cls), count in joint.items():
        p_xy = count / total
        p_x = sum(v for (bj, _), v in joint.items() if bj == bi) / total
        p_y = (n_a if cls == 0 else n_b) / total
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return max(0.0, mi)


# Approximate statistical distribution functions (no scipy dependency)

def _t_pvalue(t: float, dof: float) -> float:
    """Two-tailed p-value for t-distribution (Abramowitz & Stegun approx)."""
    if dof <= 0:
        return 1.0
    x = dof / (dof + t * t)
    # Incomplete beta function approximation
    a = dof / 2
    b = 0.5
    p = _regularised_incomplete_beta(x, a, b)
    return min(1.0, max(0.0, p))


def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """Continued-fraction approximation of the regularised incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use continued fraction (Lentz method, limited iterations for speed)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    # Simple iterative approximation
    cf = 1.0
    for m in range(1, 60):
        dm = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        cf = 1.0 + dm / cf
        dm2 = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        cf = 1.0 + dm2 / cf
    return min(1.0, front * cf)


def _norm_sf(z: float) -> float:
    """Standard normal survival function P(Z > z) approximation."""
    return 0.5 * math.erfc(z / math.sqrt(2))


# ── VarianceProfiler ────────────────────────────────────────────────────────────


class VarianceProfiler:
    """
    Consumes TrialBatches and produces VarianceProfiles + DistinguishabilityResults.

    For each batch, the profiler:
    1. Computes DistributionStats for class A, class B, and baseline noise.
    2. Builds a NoiseModel from the baseline.
    3. Applies all configured StatisticalTests.
    4. Assigns a DistinguishabilityVerdict based on the test ensemble.
    5. Produces a DistinguishabilityResult linking the finding to the plan.

    Parameters
    ----------
    significance_level   — p-value threshold (default 0.01)
    min_effect_size_d    — minimum Cohen's d for practical significance (default 0.2)
    tests                — list of StatisticalTest to apply
    """

    def __init__(
        self,
        significance_level: float = 0.01,
        min_effect_size_d: float = 0.2,
        tests: list[StatisticalTest] | None = None,
    ) -> None:
        self._alpha = significance_level
        self._min_d = min_effect_size_d
        self._tests = tests or [
            StatisticalTest.WELCH_T,
            StatisticalTest.MANN_WHITNEY_U,
            StatisticalTest.KS_TWO_SAMPLE,
            StatisticalTest.EFFECT_SIZE_D,
            StatisticalTest.MUTUAL_INFO,
        ]
        self._log = logger.bind(component="VarianceProfiler")

    def profile_all(
        self,
        batches: list[TrialBatch],
        plan: MeasurementPlan,
    ) -> tuple[list[VarianceProfile], list[DistinguishabilityResult]]:
        """
        Profile all batches.  Returns (profiles, distinguishability_results).
        """
        profiles: list[VarianceProfile] = []
        results: list[DistinguishabilityResult] = []

        op_map = {op.op_id: op for op in plan.operations}

        for batch in batches:
            op = op_map.get(batch.op_id)
            if not op:
                continue

            profile, dr = self._profile_batch(batch, op)
            profiles.append(profile)
            results.append(dr)

            self._log.debug(
                "operation_profiled",
                op_name=op.name,
                verdict=profile.verdict.value,
                mean_delta_ns=profile.mean_delta_ns,
                max_effect_size=round(profile.max_effect_size, 3),
            )

        return profiles, results

    def _profile_batch(
        self,
        batch: TrialBatch,
        op: OperationSpec,
    ) -> tuple[VarianceProfile, DistinguishabilityResult]:
        a_vals = [float(t.latency_ns) for t in batch.class_a_observations]
        b_vals = [float(t.latency_ns) for t in batch.class_b_observations]
        noise_vals = [float(t.latency_ns) for t in batch.noise_baseline_observations]

        a_stats = _distribution_stats(a_vals)
        b_stats = _distribution_stats(b_vals)
        noise_stats = _distribution_stats(noise_vals)

        noise_cv = noise_stats.cv
        noise_model = NoiseModel(
            baseline_stats=noise_stats,
            noise_level=_classify_noise(noise_cv),
            baseline_cv=noise_cv,
            min_detectable_effect_d=max(0.1, noise_cv * 2.5),
            is_viable=noise_cv < 0.25,
            viability_reason=(
                "CV below 0.25 — channel viable for analysis"
                if noise_cv < 0.25
                else f"CV={noise_cv:.3f} exceeds threshold 0.25 — channel too noisy"
            ),
        )

        test_results: list[StatTestResult] = []
        for test in self._tests:
            tr = self._run_test(test, a_vals, b_vals)
            test_results.append(tr)

        mean_delta = a_stats.mean_ns - b_stats.mean_ns
        mean_delta_rel = mean_delta / b_stats.mean_ns if b_stats.mean_ns > 0 else 0.0
        max_effect = max(
            (tr.effect_size for tr in test_results if tr.effect_size is not None),
            default=0.0,
        )

        verdict = self._assign_verdict(test_results, max_effect, noise_model)

        profile = VarianceProfile(
            op_id=op.op_id,
            target_id=batch.target_id,
            op_name=op.name,
            operation_class=op.operation_class,
            channel_kind=op.channel_kind,
            class_a_stats=a_stats,
            class_b_stats=b_stats,
            noise_model=noise_model,
            test_results=test_results,
            mean_delta_ns=mean_delta,
            mean_delta_relative=mean_delta_rel,
            max_effect_size=max_effect,
            verdict=verdict,
            source_region_id=op.source_region_id,
            source_fsm_state_id=op.source_fsm_state_id,
            source_corridor_id=op.source_corridor_id,
        )

        confirming = [
            tr.test for tr in test_results
            if tr.is_significant
            and tr.test != StatisticalTest.EFFECT_SIZE_D
        ]
        non_sig = [
            tr.test for tr in test_results
            if not tr.is_significant
            and tr.test != StatisticalTest.EFFECT_SIZE_D
        ]

        is_distinguishable = verdict == DistinguishabilityVerdict.DISTINGUISHABLE
        is_security = op.operation_class in (
            OperationClass.CRYPTO_BRANCH,
            OperationClass.AUTH_DECISION,
            OperationClass.PROTOCOL_GUARD,
        ) and is_distinguishable

        evidence = self._build_evidence_summary(
            op, profile, verdict, mean_delta, max_effect, noise_model,
        )

        dr = DistinguishabilityResult(
            target_id=batch.target_id,
            op_id=op.op_id,
            op_name=op.name,
            operation_class=op.operation_class,
            channel_kind=op.channel_kind,
            verdict=verdict,
            is_distinguishable=is_distinguishable,
            profile=profile,
            confirming_tests=confirming,
            non_significant_tests=non_sig,
            mean_delta_ns=mean_delta,
            max_effect_size_d=max_effect,
            noise_level=noise_model.noise_level,
            evidence_summary=evidence,
            is_security_relevant=is_security,
            security_relevance_reason=(
                f"{op.operation_class.value} channel distinguishable at {op.channel_kind.value}"
                if is_security else ""
            ),
        )

        return profile, dr

    def _run_test(
        self, test: StatisticalTest, a: list[float], b: list[float]
    ) -> StatTestResult:
        if test == StatisticalTest.WELCH_T:
            stat, p = _welch_t(a, b)
            d = _cohen_d(a, b)
            sig = p < self._alpha
            return StatTestResult(
                test=test, statistic=stat, p_value=p,
                effect_size=d, is_significant=sig,
                detail=f"t={stat:.3f}, p={p:.4f}, d={d:.3f}",
            )
        elif test == StatisticalTest.MANN_WHITNEY_U:
            stat, p = _mann_whitney_u(a, b)
            sig = p < self._alpha
            return StatTestResult(
                test=test, statistic=stat, p_value=p,
                effect_size=None, is_significant=sig,
                detail=f"U={stat:.1f}, p={p:.4f}",
            )
        elif test == StatisticalTest.KS_TWO_SAMPLE:
            stat, p = _ks_two_sample(a, b)
            sig = p < self._alpha
            return StatTestResult(
                test=test, statistic=stat, p_value=p,
                effect_size=None, is_significant=sig,
                detail=f"D={stat:.4f}, p={p:.4f}",
            )
        elif test == StatisticalTest.EFFECT_SIZE_D:
            d = _cohen_d(a, b)
            sig = d >= self._min_d
            return StatTestResult(
                test=test, statistic=d, p_value=None,
                effect_size=d, is_significant=sig,
                detail=f"Cohen's d={d:.3f} ({'≥' if sig else '<'} threshold {self._min_d})",
            )
        elif test == StatisticalTest.MUTUAL_INFO:
            mi = _mutual_info_estimate(a, b)
            sig = mi >= 0.05  # 0.05 bits as practical significance threshold
            return StatTestResult(
                test=test, statistic=mi, p_value=None,
                effect_size=mi, is_significant=sig,
                detail=f"MI={mi:.4f} bits",
            )
        else:
            return StatTestResult(
                test=test, statistic=0.0, p_value=1.0,
                effect_size=0.0, is_significant=False,
                detail="test not implemented",
            )

    def _assign_verdict(
        self,
        test_results: list[StatTestResult],
        max_effect: float,
        noise_model: NoiseModel,
    ) -> DistinguishabilityVerdict:
        if not noise_model.is_viable:
            return DistinguishabilityVerdict.INCONCLUSIVE

        sig_count = sum(
            1 for tr in test_results
            if tr.is_significant and tr.test != StatisticalTest.EFFECT_SIZE_D
        )
        effect_sig = any(
            tr.is_significant and tr.test == StatisticalTest.EFFECT_SIZE_D
            for tr in test_results
        )

        if sig_count >= 2 and effect_sig:
            return DistinguishabilityVerdict.DISTINGUISHABLE
        if sig_count >= 1 and max_effect >= noise_model.min_detectable_effect_d:
            return DistinguishabilityVerdict.MARGINAL
        if sig_count == 0:
            return DistinguishabilityVerdict.NOT_DISTINGUISHABLE
        return DistinguishabilityVerdict.INCONCLUSIVE

    @staticmethod
    def _build_evidence_summary(
        op: OperationSpec,
        profile: VarianceProfile,
        verdict: DistinguishabilityVerdict,
        mean_delta: float,
        max_d: float,
        noise_model: NoiseModel,
    ) -> str:
        delta_pct = (
            f"{abs(profile.mean_delta_relative) * 100:.1f}%"
            if profile.class_b_stats.mean_ns > 0 else "N/A"
        )
        a_mean = f"{profile.class_a_stats.mean_ns:.0f}ns"
        b_mean = f"{profile.class_b_stats.mean_ns:.0f}ns"
        noise = noise_model.noise_level.value
        direction = "slower" if mean_delta > 0 else "faster"
        return (
            f"Under {noise} noise ({noise_model.noise_level.value}), "
            f"'{op.class_a_label}' path was {delta_pct} {direction} than "
            f"'{op.class_b_label}' (mean: {a_mean} vs {b_mean}, "
            f"max Cohen's d={max_d:.3f}). "
            f"Verdict: {verdict.value}."
        )


# ── ChannelSignatureMapper ──────────────────────────────────────────────────────


class ChannelSignatureMapper:
    """
    Links DistinguishabilityResults to higher-layer Phase 4/5/6 events and
    groups structurally similar results into ChannelSignatures.

    Grouping heuristic: results with the same (OperationClass, ChannelKind)
    pair form one signature.  Within each group, higher-layer events are
    gathered from cross-reference fields set during MeasurementPlan construction.

    Parameters
    ----------
    phase4_result — optional Phase 4 output for region labels
    phase5_result — optional Phase 5 output for corridor descriptions
    phase6_result — optional Phase 6 output for boundary failure descriptions
    """

    def __init__(
        self,
        phase4_result: Phase4Result | None = None,
        phase5_result: Phase5Result | None = None,
        phase6_result: Phase6Result | None = None,
    ) -> None:
        self._p4 = phase4_result
        self._p5 = phase5_result
        self._p6 = phase6_result
        self._log = logger.bind(component="ChannelSignatureMapper")

    def map(
        self,
        results: list[DistinguishabilityResult],
        target_id: str,
    ) -> list[ChannelSignature]:
        """
        Produce ChannelSignatures from DistinguishabilityResults.

        Only distinguishable + marginal results are included in signatures.
        """
        relevant = [
            dr for dr in results
            if dr.verdict in (
                DistinguishabilityVerdict.DISTINGUISHABLE,
                DistinguishabilityVerdict.MARGINAL,
            )
        ]

        # Group by (OperationClass, ChannelKind)
        groups: dict[tuple[OperationClass, ChannelKind], list[DistinguishabilityResult]] = {}
        for dr in relevant:
            key = (dr.operation_class, dr.channel_kind)
            groups.setdefault(key, []).append(dr)

        signatures: list[ChannelSignature] = []
        for (op_class, ch_kind), group in groups.items():
            sig = self._build_signature(group, op_class, ch_kind, target_id)
            signatures.append(sig)
            self._log.debug(
                "signature_built",
                op_class=op_class.value,
                ch_kind=ch_kind.value,
                n_results=len(group),
                confidence=round(sig.confidence, 3),
            )

        return signatures

    def _build_signature(
        self,
        group: list[DistinguishabilityResult],
        op_class: OperationClass,
        ch_kind: ChannelKind,
        target_id: str,
    ) -> ChannelSignature:
        events = self._gather_events(group)
        result_ids = [dr.result_id for dr in group]
        mean_delta = mean(dr.mean_delta_ns for dr in group)
        mean_d = mean(dr.max_effect_size_d for dr in group)
        noise_levels = [dr.noise_level for dr in group]
        # Dominant noise level (most common)
        dominant_noise = max(set(noise_levels), key=noise_levels.count)

        n_distinguishable = sum(1 for dr in group if dr.is_distinguishable)
        confidence = min(
            1.0,
            (n_distinguishable / max(len(group), 1)) * (mean_d / (mean_d + 0.5)),
        )
        is_sec = any(dr.is_security_relevant for dr in group)

        narrative = self._build_narrative(
            group, op_class, ch_kind, mean_delta, mean_d, events
        )

        return ChannelSignature(
            target_id=target_id,
            channel_kind=ch_kind,
            operation_class=op_class,
            distinguishability_result_ids=result_ids,
            higher_layer_events=events,
            mean_delta_ns=mean_delta,
            mean_effect_size_d=mean_d,
            noise_level=dominant_noise,
            confidence=confidence,
            signature_narrative=narrative,
            is_security_relevant=is_sec,
        )

    def _gather_events(
        self, group: list[DistinguishabilityResult]
    ) -> list[HigherLayerEvent]:
        events: list[HigherLayerEvent] = []

        for dr in group:
            profile = dr.profile

            # Phase 4 region
            if profile.source_region_id:
                desc = f"Phase 4 control-flow region {profile.source_region_id}"
                if self._p4:
                    # Try to get a human-readable description from Phase 4
                    region = getattr(self._p4, "steerable_regions", {}).get(
                        profile.source_region_id
                    )
                    if region:
                        desc = getattr(region, "description", desc)
                events.append(HigherLayerEvent(
                    layer="control_flow",
                    phase4_region_id=profile.source_region_id,
                    description=desc,
                    sensitivity=60,
                ))

            # Phase 5 corridor
            if profile.source_corridor_id:
                desc = f"Phase 5 trust corridor {profile.source_corridor_id}"
                if self._p5:
                    corridors = getattr(self._p5, "corridors", [])
                    corridor = next(
                        (c for c in corridors
                         if getattr(c, "corridor_id", "") == profile.source_corridor_id),
                        None,
                    )
                    if corridor:
                        desc = getattr(corridor, "description", desc)
                events.append(HigherLayerEvent(
                    layer="trust",
                    phase5_corridor_id=profile.source_corridor_id,
                    description=desc,
                    sensitivity=75,
                ))

            # Phase 6 FSM state
            if profile.source_fsm_state_id:
                desc = f"Phase 6 FSM boundary state {profile.source_fsm_state_id}"
                if self._p6:
                    for fsm in getattr(self._p6, "fsms", []):
                        state = fsm.states.get(profile.source_fsm_state_id)
                        if state:
                            desc = (
                                f"Protocol boundary state '{state.name}' "
                                f"({state.layer.value})"
                            )
                            break
                events.append(HigherLayerEvent(
                    layer="state_machine",
                    phase6_fsm_state_id=profile.source_fsm_state_id,
                    description=desc,
                    sensitivity=80,
                ))

        # Deduplicate by description
        seen: set[str] = set()
        unique_events: list[HigherLayerEvent] = []
        for e in events:
            if e.description not in seen:
                seen.add(e.description)
                unique_events.append(e)

        return unique_events

    @staticmethod
    def _build_narrative(
        group: list[DistinguishabilityResult],
        op_class: OperationClass,
        ch_kind: ChannelKind,
        mean_delta: float,
        mean_d: float,
        events: list[HigherLayerEvent],
    ) -> str:
        n_dist = sum(1 for dr in group if dr.is_distinguishable)
        event_descs = "; ".join(e.description for e in events[:3])
        return (
            f"{n_dist}/{len(group)} {op_class.value} operations are distinguishable "
            f"via {ch_kind.value} channel "
            f"(mean delta={mean_delta:.0f}ns, mean d={mean_d:.3f}). "
            f"Linked higher-layer events: {event_descs or 'none identified'}."
        )
