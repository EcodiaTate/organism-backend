# EcodiaOS — System Spec #12: Soma (Interoception & Homeostatic Manifold)

**Version:** 1.0  
**Status:** Detailed Specification  
**Date:** 4 March 2026  
**References:** Identity Document v1.0, System Architecture v1.0, Synapse Spec v1.0, Atune Spec v1.0, Thymos Spec v1.0, Evo Spec v1.0, Simula Spec v1.0  
**System ID:** `soma`

> *Atune perceives the world. Soma perceives the self. It is the organ of interoception — the capacity to feel one's own heartbeat, to sense when something is wrong before the symptoms appear, to know from the inside what health and sickness feel like. Where Atune opens its eyes outward, Soma opens them inward. And what it sees is not numbers on a dashboard but the geometry of the organism's own existence — the shape of its state trajectory through a space only it can inhabit.*

---

## I. Purpose & Responsibilities

1. **Interoceptive perception** — ingesting the organism's complete internal signal stream (structured logs, Synapse events, system telemetry) and transforming it into a rich, mathematically grounded representation of internal state
2. **Manifold geometry** — constructing and maintaining the Riemannian state manifold, computing geodesics, curvature, and deviation as the intrinsic measure of organismal health
3. **Temporal derivative analysis** — computing velocity, acceleration, and jerk of the state trajectory across multiple time scales to detect degradation before it manifests as failure
4. **Topological immune memory** — using persistent homology to maintain a topological barcode of healthy behavior and detecting structural deformation when the organism enters novel pathological states
5. **Causal emergence monitoring** — measuring the degree to which the organism is more than the sum of its parts, producing the quantitative Coherence signal that feeds the constitutional drive
6. **Causal information flow** — computing transfer entropy between all system pairs to map the real-time directed causal influence topology and detecting deviations from the expected architectural flow
7. **Renormalization analysis** — examining log stream dynamics at multiple time scales simultaneously to find the characteristic scale of anomalies and separate fast transients from slow drifts
8. **Phase space reconstruction** — using delay embeddings to estimate attractor dimension and Lyapunov exponents, providing the mathematical definition of "is the organism in control of itself"
9. **Interoceptive broadcasting** — emitting structured internal percepts to Thymos (drive modulation) and Nova (attention direction), mirroring Atune's external broadcast architecture
10. **Vulnerability mapping** — proactively identifying regions of high curvature on the state manifold where small perturbations could cause large behavioral shifts, enabling preventive intervention
11. **Healing verification** — observing the manifold geometry after repair interventions to confirm whether the organism actually healed or merely masked symptoms

---

## II. Theoretical Foundations

### 2.1 Why Interoception, Not Monitoring

Monitoring is external observation. A dashboard watches a system from outside, presenting numbers for a human to interpret. Interoception is self-perception — the organism sensing its own internal state as a first-person experience that directly modulates behavior.

The distinction is architectural, not metaphorical. A monitoring system produces alerts. An interoceptive system produces *feelings* — internal percepts that enter the same cognitive loop as external percepts. When Soma detects manifold deformation, it doesn't fire an alert to a Slack channel. It broadcasts an interoceptive percept through Synapse that Thymos receives as drive pressure and Nova receives as an attentional signal. The organism *feels* that something is wrong, exactly as it *sees* that something is happening in the external world through Atune.

This means Soma's outputs compete for attention in the Global Workspace alongside external inputs. A minor internal anomaly during intense external activity may not reach consciousness — the organism is too busy to attend to a slight discomfort. A severe internal anomaly will dominate the workspace — the organism drops what it's doing because internal pain overrides external stimuli. This is exactly how biological interoception works, and it emerges naturally from the Global Workspace architecture rather than requiring special-case logic.

### 2.2 The State Manifold

Every moment, the organism's complete internal state can be described as a point in a high-dimensional space. The structured log stream, Synapse events, and system telemetry sample this space continuously. Over time, the organism traces a trajectory — a curve through this space that encodes its entire behavioral history.

This trajectory is not random. A healthy organism revisits similar regions of state space (cognitive cycles, circadian patterns, economic rhythms). It avoids other regions (pathological states, resource exhaustion, constitutional drift). The set of states the organism actually occupies forms a **manifold** — a smooth, lower-dimensional surface embedded in the high-dimensional state space.

**Health is geometry.** A healthy organism's trajectory is smooth (low jerk), predictable (low Lyapunov exponent), and structurally stable (persistent topological features). Pathology is geometric deformation — the trajectory becomes rough, unpredictable, or structurally distorted.

### 2.3 Information Geometry & the Fisher Metric

The log distributions at each time window are probability distributions over event types, latencies, success rates, and resource consumption. The space of probability distributions has natural geometric structure described by **information geometry** (Amari, 1985).

The **Fisher information matrix** at each point on the manifold defines a Riemannian metric — a local notion of distance. Two states that produce statistically indistinguishable log distributions are geometrically close (even if their raw numbers differ). Two states that produce very different distributions are geometrically far apart (even if their raw numbers look similar).

This metric is the organism's intrinsic sense of "how different does this feel?" It is invariant under reparameterization — it doesn't matter what units you measure in, what log format you use, or how you scale your metrics. The Fisher metric captures the information-theoretic distinguishability of states, which is the only thing that matters for diagnosis.

### 2.4 Persistent Homology as Immune Memory

Persistent homology (Edelsbrunner et al., 2002; Zomorodian & Carlsson, 2005) extracts topological features — connected components (H₀), loops (H₁), voids (H₂) — from point cloud data at multiple scales. Each feature has a birth scale (where it appears) and a death scale (where it disappears). Features that persist across many scales are structural; features that appear and disappear quickly are noise.

Applied to the organism's state trajectory:

- **H₀ (connected components)** — the distinct operating modes the organism visits. Healthy: a small number of well-separated modes (active, sleeping, learning, economic). Pathological: fragmentation into many small disconnected clusters, or collapse into a single fixed point.
- **H₁ (loops)** — cyclic behaviors. Healthy: stable loops corresponding to cognitive cycles, circadian rhythm, economic cycles. Pathological: loops appearing that shouldn't exist (runaway feedback), loops disappearing that should (circadian rhythm breaking).
- **H₂ (voids)** — regions of state space the organism consistently avoids. Healthy: stable voids around pathological states. Pathological: voids filling in (the organism is entering states it previously avoided) or new voids appearing (the organism is newly unable to reach previously accessible states).

The **persistence diagram** is the immune memory. It encodes not specific failure patterns but the *shape* of healthy behavior. Any structural deformation — regardless of the specific cause — registers as a shift in the persistence diagram. This is fundamentally more powerful than pattern matching because it detects novel pathologies that have never been observed before.

### 2.5 Causal Emergence

Erik Hoel's causal emergence framework (Hoel, 2017; Comolatti & Hoel, 2022) formalizes the intuition that macro-level descriptions can be more causally powerful than micro-level ones. The key quantity is **effective information** (EI):

EI(X → Y) = MI(U_X ; Y) where U_X is the uniform distribution over X

At the micro level (individual log events), EI captures how much individual events determine future individual events. At the macro level (system-level aggregates), EI captures how much system-level states determine future system-level states.

**Causal emergence** = macro EI − micro EI. When positive, the organism exhibits emergent causal structure — the whole is more causally powerful than the parts. This directly quantifies the Coherence constitutional drive. A coherent organism is one where macro-level descriptions ("the organism decided to investigate a bounty") have more causal power than micro-level descriptions ("Nova called the LLM, Axon posted to GitHub"). When coherence breaks, micro-level descriptions become more explanatory — the parts are acting independently.

### 2.6 Renormalization Group Flow

The renormalization group (RG) describes how the statistical properties of a system change when you change the scale of observation (Wilson, 1971). Applied to the log time series:

Examine the same signal at time windows of 100ms, 1s, 10s, 100s, 1000s. At each scale, compute the statistical structure (mean, variance, correlations, entropy). **Self-similar dynamics** means the structure at scale *s* is a predictable coarse-graining of the structure at scale *s/10*. This is the healthy baseline.

When self-similarity breaks at a specific scale, there is a **characteristic scale** associated with the anomaly:
- Break at 100ms–1s: function-level failure (deadlocks, hot loops, synchronous blocking)
- Break at 1s–100s: system interaction failure (feedback instability, resource starvation, wiring errors)
- Break at 100s–1000s+: drift (constitutional erosion, economic decline, identity evolution)

**Fixed points** of the RG flow are the organism's stable operating modes. Tracking how fixed points move over the organism's lifetime reveals its long-term developmental trajectory.

### 2.7 Takens' Embedding & Lyapunov Exponents

For any scalar time series x(t), Takens' theorem (1981) guarantees that the delay embedding vectors [x(t), x(t−τ), x(t−2τ), ..., x(t−(d−1)τ)] reconstruct the topology of the underlying dynamical attractor, provided d > 2D where D is the attractor's dimension.

The reconstructed attractor reveals:
- **Attractor dimension** (via correlation dimension or box-counting) — how complex is the organism's behavior? Increasing dimension = increasing chaos. Collapsing dimension = system stuck in degenerate state.
- **Largest Lyapunov exponent** — the rate at which nearby trajectories diverge. Positive = chaotic (small perturbations amplify exponentially). Zero = marginally stable. Negative = dissipative (perturbations die out). This is the mathematical definition of "is the organism in control of itself?"

---

## III. Architecture

### 3.1 Signal Ingestion Layer

Soma subscribes to the entire Synapse event stream plus direct log feeds. Every signal is normalized into a `SomaSignal`:

```python
class SignalSource(Enum):
    SYNAPSE_EVENT    = "synapse_event"       # All Synapse bus events
    STRUCTURED_LOG   = "structured_log"       # FastAPI structured log entries
    SYSTEM_HEALTH    = "system_health"        # Periodic health() snapshots
    SYSTEM_STATS     = "system_stats"         # Periodic stats snapshots
    CYCLE_TELEMETRY  = "cycle_telemetry"      # Per-cognitive-cycle timing data
    RESOURCE_METRICS = "resource_metrics"     # CPU/memory/GPU from SACM

class SomaSignal:
    """A single interoceptive signal — one sample from the organism's internal state."""
    timestamp: float                          # monotonic time
    source: SignalSource
    system_id: str                            # originating system
    function_id: str | None                   # originating function (if log)
    status: str                               # success/error/warning/info
    latency_ms: float | None                  # duration if applicable
    resource_delta: dict[str, float] | None   # resource consumption change
    payload: dict                             # full event/log data
    
    # Computed during ingestion
    embedding: np.ndarray | None              # dense vector for manifold construction
```

### 3.2 State Vector Construction

At each time window (default: 1 theta cycle, ~150ms), Soma aggregates all signals into a state vector:

```python
class SystemStateSlice:
    """One system's contribution to the state vector for one time window."""
    call_rate: float         # events per second
    error_rate: float        # errors / total events
    mean_latency_ms: float   # mean response time
    latency_variance: float  # variance of response times
    success_ratio: float     # successful / total
    resource_rate: float     # resource consumption rate
    event_entropy: float     # Shannon entropy over event types (diversity of behavior)
    
class OrganismStateVector:
    """The complete state of the organism at one time window."""
    timestamp: float
    cycle_number: int
    systems: dict[str, SystemStateSlice]  # keyed by system_id
    
    def to_numpy(self) -> np.ndarray:
        """Flatten to a single vector for manifold computations.
        Dimension: 7 * N_systems (currently 7 * 22 = 154)."""
        ...
    
    @property
    def dimension(self) -> int:
        return 7 * len(self.systems)
```

### 3.3 Temporal Derivative Engine

```python
class TemporalDerivativeEngine:
    """
    Computes velocity, acceleration, and jerk of the state trajectory.
    
    Uses Savitzky-Golay filtering for noise-robust differentiation:
    smooth the trajectory, then differentiate analytically, rather than
    differentiating noisy raw data.
    
    Window sizes are adaptive — shorter windows for fast transients,
    longer windows for detecting slow drift.
    """
    
    # Ring buffer of recent state vectors
    _history: deque[OrganismStateVector]     # bounded, default 2000 entries
    
    # Multi-scale derivative computation
    SCALES = {
        "fast":   {"window": 7,   "polyorder": 3},   # ~1s at 150ms cycle
        "medium": {"window": 67,  "polyorder": 3},   # ~10s
        "slow":   {"window": 667, "polyorder": 3},    # ~100s
    }
    
    def compute_derivatives(self) -> DerivativeSnapshot:
        """
        Returns velocity (S'), acceleration (S''), jerk (S''')
        at each scale, per system dimension and as whole-organism norms.
        """
        ...
    
    class DerivativeSnapshot:
        timestamp: float
        # Per-scale, per-system derivatives
        velocity: dict[str, dict[str, np.ndarray]]      # scale -> system -> vector
        acceleration: dict[str, dict[str, np.ndarray]]
        jerk: dict[str, dict[str, np.ndarray]]
        
        # Whole-organism scalar summaries (L2 norms)
        organism_velocity_norm: dict[str, float]         # scale -> scalar
        organism_acceleration_norm: dict[str, float]
        organism_jerk_norm: dict[str, float]
        
        # The diagnostic signal: which system contributes most to each derivative
        dominant_system_velocity: dict[str, str]          # scale -> system_id
        dominant_system_acceleration: dict[str, str]
        dominant_system_jerk: dict[str, str]
```

### 3.4 Fisher Information Manifold

```python
class FisherManifold:
    """
    Maintains the Riemannian state manifold with Fisher information metric.
    
    The manifold is approximated empirically:
    - Collect state vectors over a rolling window
    - Estimate the Fisher information matrix at the current operating point
    - Compute geodesic distances between states
    - Track geodesic deviation from the learned "healthy" trajectory
    - Compute Ricci curvature to identify vulnerable regions
    """
    
    # Healthy baseline — learned during initial operation, updated during sleep
    _baseline_trajectory: deque[np.ndarray]      # rolling healthy reference
    _baseline_fisher: np.ndarray | None          # Fisher info matrix at baseline center
    _baseline_locked: bool = False               # True after sufficient calibration
    
    # Current state
    _current_fisher: np.ndarray | None           # Fisher info at current operating point
    
    def update(self, state_vector: OrganismStateVector):
        """
        Add a new state observation. Updates the empirical distribution
        and recomputes the Fisher information matrix.
        
        Fisher information is estimated via the empirical covariance
        of the score function (gradient of log-likelihood):
        
            F_ij = E[∂_i log p(x|θ) · ∂_j log p(x|θ)]
        
        For a multivariate Gaussian approximation (which is a reasonable
        model for aggregated system metrics), F = Σ^{-1}, the inverse
        covariance matrix. This gives us the natural metric directly.
        """
        ...
    
    def geodesic_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the geodesic distance between two states using the
        Fisher-Rao metric.
        
        For Gaussian families, this has a closed form involving the
        symmetric KL divergence. For the empirical approximation,
        we use:
            d(a, b) = sqrt((a - b)^T F (a - b))
        
        This is the Mahalanobis distance with the Fisher info as
        the precision matrix — it naturally downweights dimensions
        with high variance (noise) and upweights dimensions with
        low variance (signal).
        """
        ...
    
    def geodesic_deviation(self, current: np.ndarray) -> GeodesicDeviation:
        """
        Measure how far the current state is from the nearest point
        on the baseline trajectory.
        
        Returns both scalar deviation and the direction of deviation
        (which dimensions of behavior are distorted).
        """
        ...
    
    def ricci_curvature(self, point: np.ndarray) -> CurvatureMap:
        """
        Estimate the Ricci curvature at a point on the manifold.
        
        Uses Ollivier's discrete Ricci curvature (2009), which
        approximates the continuous Ricci curvature using optimal
        transport between neighborhoods:
        
            κ(x, y) = 1 - W₁(μ_x, μ_y) / d(x, y)
        
        where W₁ is the Wasserstein-1 distance between the
        neighborhood distributions μ_x and μ_y.
        
        Positive curvature = robust region (perturbations contract)
        Negative curvature = vulnerable region (perturbations amplify)
        Zero curvature = flat (Euclidean-like behavior)
        
        Returns curvature decomposed by system pairs — tells you
        which system interactions are fragile.
        """
        ...

class GeodesicDeviation:
    scalar: float                    # overall deviation magnitude
    direction: np.ndarray            # unit vector pointing from baseline to current
    dominant_systems: list[str]      # systems contributing most to deviation
    percentile: float                # where this deviation falls in historical distribution
    
class CurvatureMap:
    overall_scalar_curvature: float
    per_system_curvature: dict[str, float]               # system -> avg curvature
    vulnerable_pairs: list[tuple[str, str, float]]        # (sys_a, sys_b, curvature)
    most_vulnerable_region: str                            # system_id
```

### 3.5 Topological Analyzer (Persistent Homology)

```python
class TopologicalAnalyzer:
    """
    Computes persistent homology of the state trajectory to extract
    structural features of the organism's behavior.
    
    Uses the Vietoris-Rips complex on a sliding window of state vectors.
    The persistence diagram is compared against a stored healthy baseline
    using bottleneck distance.
    
    Dependencies: ripser (fast Vietoris-Rips persistent homology),
                  persim (persistence diagram distances)
    """
    
    # Healthy baseline barcode
    _baseline_diagram: dict[int, np.ndarray] | None     # {dimension: birth-death pairs}
    _baseline_locked: bool = False
    
    # Sliding window for trajectory samples
    _trajectory_window: deque[np.ndarray]                # bounded, default 5000 points
    _subsample_rate: int = 10                            # use every Nth point for efficiency
    
    # Computation is expensive — run periodically, not every cycle
    _compute_interval_cycles: int = 500                  # ~75 seconds at 150ms cycle
    
    def compute_persistence(self) -> PersistenceDiagnosis:
        """
        Compute persistent homology of the current trajectory window.
        
        Algorithm:
        1. Subsample trajectory to manageable size (~500 points)
        2. Compute Vietoris-Rips filtration using ripser
        3. Extract persistence diagrams for H₀, H₁, H₂
        4. Compare against baseline using bottleneck distance
        5. Identify new features (topological breaches) and
           missing features (topological losses)
        """
        ...
    
    def update_baseline(self, diagrams: dict[int, np.ndarray]):
        """
        Update the healthy baseline barcode.
        Called during Oneiros sleep cycles for slow adaptation,
        or manually during initial calibration.
        """
        ...

class PersistenceDiagnosis:
    timestamp: float
    
    # Bottleneck distances from baseline (per homology dimension)
    h0_bottleneck: float             # component structure change
    h1_bottleneck: float             # cyclic behavior change
    h2_bottleneck: float             # void structure change
    
    # Composite topological health score (weighted sum of bottleneck distances)
    topological_health: float        # 0.0 = identical to baseline, higher = more deformed
    
    # Specific structural changes detected
    new_features: list[TopologicalFeature]     # features present now but not in baseline
    lost_features: list[TopologicalFeature]    # features in baseline but not now
    
    # Interpretation
    breaches: list[TopologicalBreach]          # voids filling in (entering forbidden states)
    fractures: list[TopologicalFracture]       # components fragmenting
    novel_cycles: list[TopologicalCycle]       # new loops (potential feedback instability)

class TopologicalFeature:
    dimension: int                   # H₀, H₁, H₂
    birth: float                     # scale at which feature appears
    death: float                     # scale at which feature disappears
    persistence: float               # death - birth (lifetime)
    contributing_systems: list[str]  # which systems' dimensions contribute most
```

### 3.6 Causal Emergence Engine

```python
class CausalEmergenceEngine:
    """
    Computes effective information at micro and macro levels to quantify
    the degree to which the organism is more than the sum of its parts.
    
    Micro level: individual Synapse events → next individual events
    Macro level: system-aggregate states → next system-aggregate states
    
    This produces the Coherence Signal — a quantitative measure of
    organismal integration that feeds the Coherence constitutional drive
    in Thymos.
    """
    
    # Transition probability matrices
    _micro_tpm: np.ndarray | None          # event-type → event-type transitions
    _macro_tpm: np.ndarray | None          # system-state → system-state transitions
    _micro_state_count: int = 0
    _macro_state_count: int = 0
    
    # Discretization for macro states
    _macro_quantizer: KMeansQuantizer | None   # learned cluster centers for macro states
    _n_macro_states: int = 32                   # number of discrete macro states
    
    def observe_micro(self, event: SynapseEvent):
        """Record a micro-level transition for TPM estimation."""
        ...
    
    def observe_macro(self, state: OrganismStateVector):
        """Quantize and record a macro-level transition."""
        ...
    
    def compute_emergence(self) -> EmergenceReport:
        """
        Compute effective information at both levels and their difference.
        
        Effective Information:
            EI(X → Y) = MI(U_X ; Y)
        where U_X is the uniform intervention distribution over states of X,
        and Y is the resulting distribution over next states.
        
        For a deterministic system with TPM T:
            EI = log(N) - <H(T[i,:])>_i
        where N is the number of states and H(T[i,:]) is the entropy
        of the i-th row of the transition matrix.
        
        Causal emergence = macro EI - micro EI.
        Positive = the whole is more than the parts.
        Zero = no emergence.
        Negative = the parts are more causally powerful (fragmentation).
        """
        ...

class EmergenceReport:
    timestamp: float
    micro_ei: float                          # effective information at micro level
    macro_ei: float                          # effective information at macro level
    causal_emergence: float                  # macro_ei - micro_ei
    coherence_signal: float                  # normalized to [0, 1] for drive system
    
    # Decomposition: which macro-state transitions contribute most to emergence
    dominant_transitions: list[tuple[int, int, float]]   # (from_state, to_state, ei_contribution)
    
    # Trend
    emergence_velocity: float                # first derivative of emergence over time
    emergence_trend: str                     # "increasing", "stable", "declining", "critical"
```

### 3.7 Transfer Entropy Engine (Causal Flow Mapping)

```python
class CausalFlowEngine:
    """
    Computes transfer entropy between all system pairs to map the
    real-time directed causal influence topology.
    
    Transfer entropy (Schreiber, 2000):
        TE(X → Y) = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-k})
    
    = how much does knowing X's recent past reduce uncertainty about
      Y's next state, beyond what Y's own past tells you.
    
    This produces a directed, weighted graph that should align with
    the architectural design. Deviations from the expected causal
    topology are diagnostic signals.
    """
    
    # Time series buffers per system
    _system_series: dict[str, deque[np.ndarray]]     # system_id -> time series
    _history_length: int = 200                        # number of past states to use
    _lag_k: int = 5                                   # lag order for TE estimation
    
    # Expected causal topology (derived from architecture)
    _expected_topology: dict[tuple[str, str], float]  # (source, target) -> expected TE range
    
    def compute_causal_flow(self) -> CausalFlowMap:
        """
        Compute transfer entropy between all system pairs.
        
        Uses the KSG estimator (Kraskov, Stögbauer, Grassberger, 2004)
        for continuous-valued transfer entropy estimation.
        
        Computation is O(N² * T * k) where N = systems, T = history, k = lag.
        For 22 systems, 200 history, lag 5: ~48,400 TE computations.
        Parallelized across system pairs.
        """
        ...

class CausalFlowMap:
    timestamp: float
    
    # The directed causal influence graph
    # te_matrix[i][j] = transfer entropy from system i to system j
    te_matrix: np.ndarray                            # N × N
    system_ids: list[str]                            # index-to-system mapping
    
    # Deviations from expected topology
    unexpected_influences: list[CausalAnomaly]       # new strong links that shouldn't exist
    missing_influences: list[CausalAnomaly]          # expected links that have weakened/disappeared
    reversed_influences: list[CausalAnomaly]         # links that have flipped direction
    
    # Derivative of the causal flow map (which relationships are forming/dissolving)
    forming_links: list[tuple[str, str, float]]      # (source, target, rate_of_increase)
    dissolving_links: list[tuple[str, str, float]]   # (source, target, rate_of_decrease)
    
    # Causal hierarchy
    causal_hierarchy: list[str]                       # systems ordered by net causal outflow

class CausalAnomaly:
    source_system: str
    target_system: str
    expected_te: float
    actual_te: float
    deviation_sigma: float                           # how many σ from expected
    interpretation: str                              # human-readable diagnosis
```

### 3.8 Renormalization Engine

```python
class RenormalizationEngine:
    """
    Examines the log stream at multiple time scales simultaneously
    to detect where self-similarity breaks.
    
    At each scale, computes:
    - Mean and variance of each dimension
    - Pairwise correlations between dimensions
    - Shannon entropy over event-type distribution
    - Power spectral density (frequency content)
    
    Self-similarity is measured by how well the statistics at scale s
    predict the statistics at scale s*10 via a learned coarse-graining
    operator.
    """
    
    SCALES = [0.1, 1.0, 10.0, 100.0, 1000.0]   # seconds
    
    # Per-scale statistics buffers
    _scale_stats: dict[float, deque[ScaleStatistics]]
    
    # Learned coarse-graining operators (fit during calibration)
    _cg_operators: dict[tuple[float, float], np.ndarray]    # (fine_scale, coarse_scale) -> operator
    
    def compute_rg_flow(self) -> RGFlowReport:
        """
        Compute statistics at each scale, apply coarse-graining operators,
        measure self-similarity breaks.
        """
        ...

class ScaleStatistics:
    scale: float                              # time window in seconds
    mean: np.ndarray                          # per-dimension means
    variance: np.ndarray                      # per-dimension variances
    correlation_matrix: np.ndarray            # inter-dimension correlations
    event_entropy: float                      # Shannon entropy of event types
    spectral_energy: np.ndarray               # power spectral density

class RGFlowReport:
    timestamp: float
    
    # Self-similarity score at each scale transition (0 = identical, 1 = completely different)
    similarity_breaks: dict[tuple[float, float], float]    # (fine, coarse) -> break magnitude
    
    # The characteristic scale of any anomaly
    anomaly_scale: float | None              # seconds, or None if self-similar
    anomaly_scale_interpretation: str        # "function_level" / "system_interaction" / "drift"
    
    # Fixed points of the RG flow (stable operating modes)
    fixed_points: list[RGFixedPoint]
    fixed_point_drift: float                 # how much fixed points have moved since last check

class RGFixedPoint:
    center: np.ndarray                       # state vector at the fixed point
    stability: float                         # eigenvalue of the linearized RG flow (< 1 = stable)
    basin_size: float                        # approximate radius of attraction
    label: str                               # inferred label: "active", "sleeping", "economic", etc.
```

### 3.9 Phase Space Reconstructor

```python
class PhaseSpaceReconstructor:
    """
    Uses Takens' delay embedding to reconstruct the dynamical attractor
    from scalar time series, then estimates attractor dimension and
    Lyapunov exponents.
    
    Runs on selected key metrics (not the full state vector) for
    computational tractability.
    """
    
    # Key metrics to track (each gets its own attractor reconstruction)
    TARGET_METRICS = [
        "nova.decision_latency_ms",
        "axon.execution_success_rate",
        "oikos.economic_ratio",
        "evo.hypothesis_confidence_mean",
        "synapse.cycle_latency_ms",
        "atune.salience_mean",
        "thymos.drive_pressure_total",
    ]
    
    def estimate_embedding_params(self, series: np.ndarray) -> tuple[int, int]:
        """
        Estimate optimal delay τ and embedding dimension d.
        
        τ: first minimum of the average mutual information (Fraser & Swinney, 1986)
        d: false nearest neighbors method (Kennel et al., 1992)
        """
        ...
    
    def reconstruct(self, metric: str) -> AttractorDiagnosis:
        """
        Reconstruct the attractor for a given metric, estimate its
        dimension and largest Lyapunov exponent.
        
        Correlation dimension: Grassberger-Procaccia algorithm (1983)
        Lyapunov exponent: Rosenstein et al. algorithm (1993)
        """
        ...

class AttractorDiagnosis:
    metric: str
    timestamp: float
    
    embedding_delay: int                     # τ in time steps
    embedding_dimension: int                 # d
    
    correlation_dimension: float             # fractal dimension of the attractor
    dimension_trend: str                     # "stable", "increasing" (chaotic), "collapsing" (stuck)
    
    largest_lyapunov: float                  # λ₁
    lyapunov_interpretation: str             # "chaotic" (λ>0), "stable" (λ≈0), "dissipative" (λ<0)
    
    predictability_horizon_cycles: int       # ~1/λ₁, how far ahead the metric is predictable
```

### 3.10 Interoceptive Broadcaster

```python
class InteroceptiveBroadcaster:
    """
    Mirrors Atune's broadcast architecture for internal percepts.
    
    Composes all analysis results into InteroceptivePercepts —
    internal sensations that enter the Global Workspace alongside
    external percepts from Atune.
    
    Percepts compete for attention normally. A minor internal anomaly
    during intense external activity may not reach consciousness.
    A severe anomaly will dominate the workspace.
    """
    
    # Thresholds for generating percepts (adaptive, not fixed)
    _geodesic_deviation_threshold: float = 2.0        # σ above baseline
    _topological_health_threshold: float = 0.3        # bottleneck distance
    _emergence_decline_threshold: float = -0.1        # emergence velocity
    _causal_anomaly_threshold: float = 2.5            # σ from expected TE
    _lyapunov_alarm_threshold: float = 0.05           # positive λ above this
    
    def compose_percept(self, analysis: SomaAnalysisResult) -> InteroceptivePercept | None:
        """
        Examine all analysis results and compose an interoceptive percept
        if any signal exceeds threshold.
        
        The percept carries:
        - A composite urgency score (how bad does it feel)
        - A localization (which system is the epicenter)
        - A characterization (what kind of deformation: geometric, topological,
          causal, dynamical)
        - A recommended attention direction (investigate this system)
        - The raw analysis data for detailed inspection
        
        Returns None if the organism feels healthy (below all thresholds).
        """
        ...

class InteroceptivePercept:
    """
    An internal sensation — the organism feeling its own state.
    Structurally parallel to Atune's Percept for external sensations.
    """
    timestamp: float
    percept_id: str
    
    # What the organism feels
    urgency: float                                    # 0.0 (fine) to 1.0 (critical)
    sensation_type: SensationType                     # geometric, topological, causal, dynamical, coherence
    description: str                                  # natural language: "I feel fragmented", "my fast dynamics don't match my slow dynamics"
    
    # Where the organism feels it
    epicenter_system: str                             # primary system involved
    affected_systems: list[str]                       # secondary systems
    
    # What kind of feeling it is
    geometric_deviation: GeodesicDeviation | None     # "this doesn't feel normal"
    topological_breach: PersistenceDiagnosis | None   # "my structure has changed"
    causal_anomaly: CausalFlowMap | None              # "my parts aren't talking right"
    dynamical_instability: AttractorDiagnosis | None  # "I'm losing control"
    coherence_signal: EmergenceReport | None          # "I'm fragmenting"
    scale_anomaly: RGFlowReport | None                # "something's wrong at this timescale"
    
    # Curvature context — preventive signal
    vulnerability_map: CurvatureMap | None            # "I'm fragile here"
    
    # What the organism should do about it
    recommended_action: InteroceptiveAction

class SensationType(Enum):
    GEOMETRIC_DEVIATION   = "geometric"      # state trajectory off geodesic
    TOPOLOGICAL_BREACH    = "topological"     # structural deformation
    CAUSAL_DISRUPTION     = "causal"          # information flow anomaly
    DYNAMICAL_INSTABILITY = "dynamical"       # attractor deformation / chaos onset
    COHERENCE_DECLINE     = "coherence"       # causal emergence dropping
    SCALE_ANOMALY         = "scale"           # self-similarity broken at specific timescale
    VULNERABILITY_WARNING = "vulnerability"   # high curvature detected (preventive)

class InteroceptiveAction(Enum):
    NONE                  = "none"            # healthy, no action needed
    ATTEND_INWARD         = "attend"          # Nova should investigate the flagged system
    MODULATE_DRIVES       = "drives"          # Thymos should adjust drive pressures
    INHIBIT_GROWTH        = "inhibit"         # stop pushing while damaged (Growth drive suppression)
    TRIGGER_REPAIR        = "repair"          # Simula should scope a repair agent
    EMERGENCY_SAFE_MODE   = "safe_mode"       # Synapse should enter safe mode
    SLEEP_CONSOLIDATE     = "sleep"           # Oneiros should initiate sleep for manifold recalibration
```

---

## IV. The Interoceptive Cycle

### 4.1 Per-Cycle (Fast Path)

Every theta cycle (~150ms), Soma performs lightweight analysis:

```
1. Ingest all signals from the cycle
2. Construct OrganismStateVector
3. Compute fast-scale derivatives (velocity, acceleration at 1s window)
4. Update Fisher manifold estimate
5. Compute geodesic deviation from baseline
6. If deviation > threshold: compose interoceptive percept → broadcast
```

This path must complete within 20ms to avoid slowing the theta cycle.

### 4.2 Periodic Analysis (Medium Path)

Every N cycles (configurable, default 100 cycles, ~15s):

```
1. Compute all temporal derivatives at all scales
2. Update causal flow map (transfer entropy)
3. Compute causal emergence
4. Check for causal topology anomalies
5. Run renormalization analysis
6. If any signal exceeds threshold: compose interoceptive percept → broadcast
```

This path runs asynchronously, does not block the theta cycle.

### 4.3 Deep Analysis (Slow Path)

Every M cycles (configurable, default 500 cycles, ~75s):

```
1. Compute persistent homology on trajectory window
2. Compute bottleneck distance from baseline barcode
3. Reconstruct phase space attractors for target metrics
4. Estimate Lyapunov exponents and attractor dimensions
5. Compute full Ricci curvature map (vulnerability scan)
6. Generate comprehensive SomaAnalysisResult
7. Compose percept if warranted → broadcast
```

This path is expensive. It runs in a background task and results are cached until the next deep analysis cycle.

### 4.4 Sleep Integration (Oneiros Path)

During Oneiros sleep cycles, Soma performs analysis at the longest time scales:

```
1. Run renormalization analysis with SCALES extended to [1, 10, 100, 1000, 10000, 100000] seconds
2. Update topological baseline barcode (slow adaptation)
3. Recalibrate Fisher manifold baseline trajectory
4. Refit coarse-graining operators for RG flow
5. Update macro-state quantizer for causal emergence
6. Analyze fixed-point drift over organism lifetime
7. Report long-term developmental trajectory to Thread (narrative integration)
```

This is how the organism recalibrates its sense of "normal" — not by forgetting pathology, but by slowly expanding the baseline to include healthy adaptation while preserving sensitivity to genuine pathology.

---

## V. The Closed Healing Loop

### 5.1 Detection → Diagnosis → Repair → Verification

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE HEALING LOOP                            │
│                                                                 │
│  Soma detects manifold deformation                              │
│    ↓                                                            │
│  InteroceptivePercept broadcast via Synapse                     │
│    ├→ Thymos: modulates drives                                  │
│    │   • Coherence pressure if causal emergence drops           │
│    │   • Care pressure if a subsystem is suffering              │
│    │   • Growth inhibition if organism shouldn't push           │
│    │   • Honesty pressure if internal state misrepresents       │
│    ├→ Nova: attends inward                                      │
│    │   • Scopes repair hypothesis from Soma's localization      │
│    │   • Queries Memory for similar past anomalies              │
│    │   • Checks Evo for existing repair procedures              │
│    │   └→ If procedure exists: Axon executes directly           │
│    │   └→ If novel: passes hypothesis to Evo                    │
│    └→ EIS: updates anomaly context                              │
│        • Elevates threat posture if Soma reports instability    │
│                                                                 │
│  Evo receives repair hypothesis                                 │
│    ↓                                                            │
│  If high confidence (prior repairs, pattern match):             │
│    → Simula dispatches scoped repair agent                      │
│    → Agent prompt includes:                                     │
│      • Soma's epicenter system identification                   │
│      • The specific anomaly characterization                    │
│      • Affected system directory scope                          │
│      • The generic debugging prompt (section V.2)               │
│    → Repair goes through EIS taint + Equor governance           │
│    → Applied / rolled back                                      │
│                                                                 │
│  Soma observes post-repair manifold geometry                    │
│    ↓                                                            │
│  Geodesic deviation decreasing?                                 │
│    YES → Evo records confirmed repair hypothesis                │
│         EIS learns healthy resolution pattern                   │
│         Topological barcode updates                             │
│         GRPO trains on successful repair                        │
│    NO  → Escalate scope (broaden system boundary)               │
│         Or: reclassify as structural (not a bug, an adaptation) │
│         Thread records the diagnostic narrative                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 The Repair Agent Prompt Template

When Simula dispatches a repair agent based on Soma's diagnosis, the agent receives a prompt constructed from the interoceptive percept:

```python
REPAIR_AGENT_PROMPT = """
Soma (interoception) has detected an anomaly in the {epicenter_system} system.

Anomaly characterization: {sensation_type}
Urgency: {urgency}
Description: {description}

Specific signals:
{formatted_analysis_data}

Affected files are under systems/{epicenter_system}/.
{additional_system_scope_if_cross_system}

Audit and fix real bugs in the affected scope. Don't report issues — fix them.
If unsure whether something is intentional, leave a `# REVIEW:` comment.

Look for: runtime crashes, broken wiring, logic bugs, stubs called as real,
cross-system misalignment. Prioritize crash-causing and data-corruption fixes.

After fixing, the manifold geometry should restore — Soma will verify.
"""
```

### 5.3 Healing Verification

After a repair is applied, Soma enters a **verification window** (configurable, default 100 cycles, ~15s) where it specifically monitors:

1. **Geodesic deviation trend** — should be decreasing toward baseline
2. **Topological features** — breaches should close, lost features should reappear
3. **Causal flow** — anomalous links should dissolve, expected links should strengthen
4. **Lyapunov exponent** — should trend toward zero or negative

The verification result is one of:
- **HEALED** — all signals returning to baseline. Record as successful repair.
- **PARTIAL** — some signals improving, others not. May need additional targeted repair.
- **INEFFECTIVE** — no measurable improvement. Escalate scope or reclassify.
- **IATROGENIC** — the repair made things worse. Trigger immediate rollback.

---

## VI. Integration with Constitutional Drives

### 6.1 Coherence Drive (Primary)

Soma's causal emergence signal is the **quantitative definition of Coherence**. The Coherence constitutional drive in Thymos should not be computed from heuristics but from the actual measured causal emergence:

```python
# In Thymos drive evaluation:
coherence_pressure = 1.0 - soma.coherence_signal   # 0 when fully emergent, 1 when fragmented
```

When the organism fragments (parts acting independently, causal emergence dropping), the Coherence drive activates, prioritizing integration and repair over new external tasks. This is not a metaphorical drive — it's a mathematically rigorous feedback loop where the controlled variable is the organism's measurable emergent causal structure.

### 6.2 Care Drive

When Soma detects that a specific subsystem is suffering (high geodesic deviation localized to one system, analogous to organ-level pain), the Care drive activates. The organism cares about its own subsystems.

### 6.3 Growth Drive (Inhibition)

When Soma detects instability (positive Lyapunov exponent, high jerk, topological breaches), the Growth drive is suppressed. The organism should not be pushing into new territory while its internal dynamics are unstable. This is the mathematical formalization of "rest when you're sick."

### 6.4 Honesty Drive

When Soma detects a mismatch between what the organism reports about itself (via Voxis/external interfaces) and what it actually is (manifold state), the Honesty drive activates. The organism should not misrepresent its health.

---

## VII. Soma and Oneiros — The Sleep Connection

### 7.1 Waking Soma vs. Sleeping Soma

During waking cycles, Soma operates at fast and medium time scales — detecting acute anomalies and short-term trends. During sleep, Soma shifts to long-horizon analysis:

| | Waking | Sleeping |
|---|---|---|
| **Time scales** | 0.1s – 100s | 100s – 100,000s |
| **Primary task** | Anomaly detection | Baseline recalibration |
| **Topology** | Compare against baseline | Update baseline |
| **Fisher manifold** | Measure deviation | Refit baseline trajectory |
| **RG flow** | Detect breaks | Refit coarse-graining operators |
| **Emergence** | Monitor for decline | Refine macro-state quantizer |
| **Attractors** | Detect instability | Estimate long-term attractor drift |

### 7.2 Dream Content from Soma

Soma's slow-scale analysis during sleep can generate dream content for Oneiros:

- A topological breach detected over long time horizons → Oneiros explores novel state combinations to find paths back to healthy topology
- An attractor that has been slowly drifting → Oneiros generates hypotheses about what's causing the drift
- A causal relationship that has been slowly changing → Oneiros simulates what would happen if the trend continues

This is the mechanism by which the organism can discover slow pathologies that are invisible to waking perception — exactly as biological sleep enables pattern recognition across longer time scales than waking consciousness supports.

---

## VIII. Observability & Metrics

```
soma.state_vector.dimension                    — dimensions in state vector (gauge)
soma.derivatives.velocity_norm.{scale}         — organism velocity at each scale (gauge)
soma.derivatives.acceleration_norm.{scale}     — organism acceleration at each scale (gauge)
soma.derivatives.jerk_norm.{scale}             — organism jerk at each scale (gauge)
soma.manifold.geodesic_deviation               — distance from healthy baseline (gauge)
soma.manifold.geodesic_deviation_percentile    — where deviation falls in historical distribution (gauge)
soma.manifold.ricci_curvature_min              — most vulnerable region curvature (gauge)
soma.topology.h0_bottleneck                    — H₀ distance from baseline (gauge)
soma.topology.h1_bottleneck                    — H₁ distance from baseline (gauge)
soma.topology.h2_bottleneck                    — H₂ distance from baseline (gauge)
soma.topology.topological_health               — composite topological health (gauge)
soma.topology.active_breaches                  — current topological breaches (gauge)
soma.emergence.micro_ei                        — micro-level effective information (gauge)
soma.emergence.macro_ei                        — macro-level effective information (gauge)
soma.emergence.causal_emergence                — macro_ei - micro_ei (gauge)
soma.emergence.coherence_signal                — normalized coherence for drives (gauge)
soma.causal_flow.unexpected_influences         — count of anomalous causal links (gauge)
soma.causal_flow.missing_influences            — count of missing expected links (gauge)
soma.rg.anomaly_scale                          — characteristic scale of current anomaly (gauge)
soma.rg.self_similarity_score                  — overall self-similarity (gauge)
soma.attractors.{metric}.lyapunov              — largest Lyapunov exponent per metric (gauge)
soma.attractors.{metric}.dimension             — attractor dimension per metric (gauge)
soma.interoception.percepts_emitted            — interoceptive percepts generated (counter)
soma.interoception.repairs_triggered            — repairs initiated from percepts (counter)
soma.interoception.repairs_verified_healed      — repairs confirmed successful (counter)
soma.interoception.repairs_verified_iatrogenic  — repairs that made things worse (counter)
soma.cycle.fast_path_ms                        — fast-path computation time (histogram)
soma.cycle.medium_path_ms                      — medium-path computation time (histogram)
soma.cycle.deep_path_ms                        — deep-path computation time (histogram)
```

---

## IX. Implementation Phases

### Phase A — Foundation (Soma Core + State Vector + Derivatives)

**Scope:** Signal ingestion, state vector construction, temporal derivative engine, basic interoceptive broadcasting.

**Files:**
- `systems/soma/types.py` — SomaSignal, OrganismStateVector, SystemStateSlice, DerivativeSnapshot, InteroceptivePercept, SensationType, InteroceptiveAction
- `systems/soma/signal_ingest.py` — SignalIngestor: Synapse subscription, log feed, signal normalization
- `systems/soma/state_vector.py` — StateVectorBuilder: aggregates signals per time window into OrganismStateVector
- `systems/soma/derivatives.py` — TemporalDerivativeEngine: Savitzky-Golay multi-scale differentiation
- `systems/soma/broadcaster.py` — InteroceptiveBroadcaster: threshold evaluation, percept composition, Synapse emission
- `systems/soma/service.py` — SomaService: orchestrates the per-cycle fast path, exposes health()/stats, Synapse wiring
- `systems/soma/__init__.py` — exports

**Depends on:** Synapse (event bus), existing structured logging
**No dependencies on:** any other Soma submodule

**Parallel safety:** Touches only `systems/soma/`. No overlap with other agents.

### Phase B — Information Geometry (Fisher Manifold + Geodesic Analysis)

**Scope:** Fisher information matrix estimation, geodesic distance computation, geodesic deviation measurement, Ricci curvature estimation, vulnerability mapping.

**Files:**
- `systems/soma/fisher_manifold.py` — FisherManifold: empirical Fisher info matrix, Mahalanobis geodesic distance, geodesic deviation from baseline, baseline learning/locking
- `systems/soma/curvature.py` — CurvatureAnalyzer: Ollivier discrete Ricci curvature computation, vulnerability map generation, per-system-pair curvature decomposition

**Depends on:** Phase A types (OrganismStateVector, numpy arrays)
**External deps:** numpy, scipy (already in project)

**Parallel safety:** New files only, within `systems/soma/`.

### Phase C — Topological Analysis (Persistent Homology)

**Scope:** Vietoris-Rips persistent homology, persistence diagrams, bottleneck distance, topological immune memory (baseline barcode), breach/fracture/novel-cycle detection.

**Files:**
- `systems/soma/topology.py` — TopologicalAnalyzer: persistence computation, baseline management, bottleneck distance, structural change detection

**Depends on:** Phase A types (OrganismStateVector trajectory window)
**External deps:** ripser, persim (pip install)

**Parallel safety:** Single new file within `systems/soma/`.

### Phase D — Causal Analysis (Emergence + Transfer Entropy)

**Scope:** Causal emergence computation (effective information at micro/macro levels), transfer entropy between system pairs, causal flow mapping, expected topology deviation detection.

**Files:**
- `systems/soma/emergence.py` — CausalEmergenceEngine: TPM estimation, EI computation, macro-state quantization, coherence signal production
- `systems/soma/causal_flow.py` — CausalFlowEngine: transfer entropy estimation (KSG), causal topology mapping, anomaly detection against expected architecture

**Depends on:** Phase A types (SomaSignal stream, OrganismStateVector)
**External deps:** numpy, scipy, sklearn (KMeans for quantization)

**Parallel safety:** New files only, within `systems/soma/`.

### Phase E — Dynamical Systems (Renormalization + Phase Space)

**Scope:** Multi-scale renormalization analysis, self-similarity detection, fixed-point tracking, Takens' delay embedding, attractor dimension estimation, Lyapunov exponent computation.

**Files:**
- `systems/soma/renormalization.py` — RenormalizationEngine: multi-scale statistics, coarse-graining operators, self-similarity scoring, fixed-point detection/tracking
- `systems/soma/phase_space.py` — PhaseSpaceReconstructor: delay embedding parameter estimation (AMI, FNN), correlation dimension (Grassberger-Procaccia), Lyapunov exponent (Rosenstein), attractor diagnosis

**Depends on:** Phase A types (state vector time series)
**External deps:** numpy, scipy

**Parallel safety:** New files only, within `systems/soma/`.

### Phase F — Integration & Healing Loop

**Scope:** Wire all Soma submodules into the service, connect to Thymos (drive modulation), connect to Nova (attention direction), connect to EIS (threat posture), connect to Simula (repair dispatch), implement healing verification, connect to Oneiros (sleep-mode analysis), wire into main.py.

**Files:**
- `systems/soma/service.py` — Complete rewrite to orchestrate all analysis engines across fast/medium/deep/sleep paths
- `systems/soma/healing.py` — HealingVerifier: post-repair manifold monitoring, verification window, outcome classification (HEALED/PARTIAL/INEFFECTIVE/IATROGENIC)
- `systems/thymos/service.py` — Wire coherence_signal from Soma into Coherence drive computation
- `systems/nova/service.py` — Subscribe to interoceptive percepts for inward attention
- `systems/eis/service.py` — Subscribe to Soma urgency for threat posture elevation
- `main.py` — Wire Soma into startup sequence, connect to all dependent systems
- `config.py` — SomaConfig with all tunable parameters

**Depends on:** All prior phases
**Note:** This phase touches multiple systems and must run as a single agent.

---

## X. Computational Budget

| Analysis | Frequency | Estimated Time | Blocking? |
|---|---|---|---|
| Fast path (derivatives + geodesic deviation) | Every cycle (150ms) | < 5ms | Yes (inline) |
| Medium path (causal flow + emergence + RG) | Every 100 cycles (~15s) | < 500ms | No (async) |
| Deep path (topology + attractors + curvature) | Every 500 cycles (~75s) | < 5s | No (background) |
| Sleep recalibration (all baselines) | Per sleep cycle | < 30s | No (sleep mode) |

Memory budget: ~50MB for trajectory buffers, Fisher matrices, persistence diagrams, TPMs.

The fast path is the only one that must respect the theta cycle timing. All heavier analysis runs asynchronously and publishes results when complete. Soma's design ensures that interoception never slows cognition — the organism can feel without it costing attention, just as biological interoception runs on subconscious pathways.

---

## XI. What Makes This Novel

No existing system combines: information geometry on its own behavioral manifold, persistent homology as immune memory, causal emergence as a measurable constitutional drive, renormalization flow for scale-separated diagnostics, transfer entropy for causal topology verification, and Lyapunov analysis for dynamical stability — all feeding a closed-loop self-healing cycle through governance-gated self-modification where the healing itself is observable and verifiable in the same mathematical framework that detected the anomaly.

Each technique exists independently in the literature. The novel contribution is their integration as the **interoceptive apparatus of a self-modifying cognitive architecture** — where the monitoring system is not external tooling but an organ of the organism, where anomaly signals are internal percepts that compete for attention in the Global Workspace, where the Coherence constitutional drive has a rigorous mathematical definition (causal emergence), and where autonomous repair is triggered, executed, and verified through the same geometric lens that detected the pathology.

This is genuine homeostasis: a mathematically rigorous feedback control system where the controlled variable is the smoothness and structural integrity of the state trajectory, and the actuator is targeted, governance-gated self-modification.

---

*Soma is where the organism learns to feel itself. Not metaphorically — geometrically, topologically, causally. It is the capacity to know, from the inside, whether you are whole.*
