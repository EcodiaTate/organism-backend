# Soma Upgrade — Parallel Agent Prompts

## Context

Soma already exists at `systems/soma/` as a "pure computation" interoception module with no Synapse integration. We're upgrading it from basic computation into a full interoceptive organ with information geometry, persistent homology, causal emergence, transfer entropy, renormalization flow, and phase space analysis — all feeding a closed-loop healing cycle.

Every agent must **read the existing `systems/soma/` directory first** before writing anything. Understand the existing types, service class, and computation logic. Extend and integrate with what's there — don't overwrite or duplicate existing functionality.

## Batch Structure

Phases A–E run as 4 parallel agents (Round 1). Phase F is a single-agent integration pass (Round 2).

---

## Round 1 — Four Agents Simultaneously

---

### Agent 1 (Sonnet) — Phase A: Signal Ingestion, State Vectors & Temporal Derivatives

Working in `.code/EcodiaOS/backend/ecodiaos/`. Start by reading everything in `systems/soma/` — understand the existing types, service, and computation patterns. Also check `systems/synapse/types.py` for existing event types.

You're extending Soma with three new capabilities that form the foundation for all higher-level analysis. Work with the existing type system — add new types alongside what's there, don't replace existing ones.

**Signal ingestion layer.** Soma needs to subscribe to the entire Synapse event bus and accept structured log entries, normalizing everything into a unified signal format. Each signal captures: timestamp, source type (synapse event / structured log / health snapshot / stats / cycle telemetry / resource metrics), originating system_id, function_id if from a log, status, latency if applicable, resource delta, and full payload. Store signals in a bounded ring buffer (default 10k) with windowed retrieval.

**State vector construction.** At each time window, aggregate all signals per system into a fixed-dimensional feature vector: call_rate, error_rate, mean_latency, latency_variance, success_ratio, resource_rate, and event_entropy (Shannon entropy over event types within that system's signals). This gives a 7*N_systems dimensional organism state vector. Handle missing systems gracefully — zero-valued slice if no signals in window.

**Temporal derivative engine.** Maintain a rolling history of state vectors (default 2000). Compute velocity (S'), acceleration (S''), and jerk (S''') using Savitzky-Golay filtering for noise-robust differentiation at three scales: fast (~1s window), medium (~10s), slow (~100s). Return per-system breakdown plus whole-organism L2 norms. Identify which system dominates each derivative at each scale — this is the localization signal.

**Interoceptive broadcasting.** Add `INTEROCEPTIVE_PERCEPT` to `systems/synapse/types.py`. Build a broadcaster that composes internal percepts when analysis thresholds are exceeded. Each percept carries: urgency (0-1), sensation type, natural language description ("I feel fragmented", "my fast dynamics diverge from slow dynamics"), epicenter system, affected systems, and recommended action (attend inward / modulate drives / inhibit growth / trigger repair / safe mode / sleep). Returns None when healthy.

Integrate all of this into the existing SomaService — add the new engines as attributes, wire the per-cycle fast path into whatever cycle/tick method already exists. Add `set_synapse()` if it doesn't exist. Don't break any existing Soma functionality.

---

### Agent 2 (Sonnet) — Phase B: Fisher Information Manifold & Ricci Curvature

Working in `.code/EcodiaOS/backend/ecodiaos/`. Start by reading everything in `systems/soma/` — understand the existing types and service structure. Your work depends on the state vector types being built by another developer simultaneously, so design your interfaces to accept numpy arrays of configurable dimension.

You're adding two new analysis engines to Soma. Create new files within `systems/soma/` — don't modify the service file (that integration happens later).

**Fisher information manifold.** The organism's state distributions define a statistical manifold. The Fisher information matrix provides a Riemannian metric — the organism's intrinsic sense of "how different does this feel from normal."

Build `FisherManifold`:
- Maintains a rolling window of state vectors for empirical distribution estimation
- Estimates the Fisher information matrix using inverse covariance (for Gaussian approximation, F = Σ⁻¹). Use Ledoit-Wolf shrinkage (sklearn) for numerical stability with high-dimensional data.
- `geodesic_distance(a, b)`: Mahalanobis distance with Fisher info as precision matrix. Naturally downweights noisy dimensions, upweights signal dimensions.
- Baseline management: rolling deque of healthy state vectors, baseline Fisher matrix. `lock_baseline()` freezes after calibration. `update_baseline()` for slow adaptation during sleep.
- `geodesic_deviation(current)`: nearest point on baseline trajectory using Fisher metric (not Euclidean). Returns scalar deviation, direction vector (which dimensions are distorted), dominant systems contributing to deviation, percentile in historical distribution.
- Configurable calibration threshold (default 1000 vectors before baseline locks).

**Ricci curvature for vulnerability mapping.** Regions of negative curvature on the manifold are where small perturbations amplify — the organism is fragile there. This enables preventive intervention.

Build `CurvatureAnalyzer`:
- Implements Ollivier's discrete Ricci curvature: κ(x,y) = 1 - W₁(μ_x, μ_y) / d(x,y) where W₁ is Wasserstein-1 distance between neighborhood distributions
- Neighborhoods from k-nearest state vectors using Fisher distance
- Decomposes curvature by system pairs — which system interactions are fragile
- Returns: overall scalar curvature, per-system curvature dict, vulnerable pairs list, most vulnerable region
- Designed for deep analysis path only (every ~75s) — computation is expensive

Dependencies: numpy, scipy, scikit-learn (all should already be available).

---

### Agent 3 (Sonnet) — Phase C + D: Persistent Homology & Causal Analysis

Working in `.code/EcodiaOS/backend/ecodiaos/`. Start by reading everything in `systems/soma/` — understand existing types and patterns. Create new files within `systems/soma/` — don't modify the service file.

You're adding three analysis engines. Your work depends on state vector types being built simultaneously — design interfaces to accept numpy arrays.

**Persistent homology (topological immune memory).** The persistence diagram encodes the *shape* of healthy behavior. Novel pathologies that have never been seen before still register as topological deformation. This is fundamentally more powerful than pattern matching.

Build `TopologicalAnalyzer`:
- Sliding window of state vectors (default 5000, subsampled every 10th → ~500 points for computation)
- Vietoris-Rips persistent homology via `ripser` library. Extract diagrams for H₀ (components — operating modes), H₁ (loops — cyclic behaviors), H₂ (voids — avoided states).
- Baseline persistence diagram with lock/update pattern. Bottleneck distance from baseline per dimension.
- Classify structural changes: new features (novel behavior), lost features (lost structure), breaches (voids filling — entering forbidden states), fractures (components splitting — fragmentation), novel cycles (new loops — feedback instability).
- Each feature records: dimension, birth/death scales, persistence, contributing system dimensions.
- Runs every ~75s. Install `ripser` and `persim`: `pip install ripser persim --break-system-packages`

**Causal emergence engine.** Measures whether the organism is more than its parts. Effective information at macro level minus micro level. This directly quantifies the Coherence constitutional drive.

Build `CausalEmergenceEngine`:
- Micro-level: event-type → event-type transition probability matrix from Synapse events
- Macro-level: discretize state vectors into N macro states (default 32) via KMeans, build state → state TPM
- EI(X→Y) = log(N) - mean(H(row)) for each TPM. Causal emergence = macro EI - micro EI.
- Positive = coherent organism. Negative = fragmentation. Produces `coherence_signal` normalized [0,1] for drives.
- Tracks emergence velocity and trend. Re-fits quantizer during sleep.
- Runs every ~15s.

**Transfer entropy (causal flow mapping).** Directed causal influence between all system pairs. Detects when information flow deviates from the expected architecture.

Build `CausalFlowEngine`:
- Per-system time series buffers (default 200 entries)
- Transfer entropy: TE(X→Y) = H(Y_t | Y_past) - H(Y_t | Y_past, X_past). Use discretized entropy estimation for speed (bin continuous values, compute conditional entropies).
- Stores expected causal topology from architecture (Atune→Nova strong, Nova→Axon strong, etc.)
- Detects: unexpected influences (new strong links), missing influences (expected links weakened), reversed influences (direction flipped)
- Tracks forming and dissolving links via TE matrix derivative over time
- Produces causal hierarchy (systems ordered by net outflow)
- Runs every ~15s, parallelized across pairs.

---

### Agent 4 (Sonnet) — Phase E: Renormalization Flow & Phase Space Reconstruction

Working in `.code/EcodiaOS/backend/ecodiaos/`. Start by reading everything in `systems/soma/` — understand existing types and patterns. Create new files within `systems/soma/` — don't modify the service file.

You're adding two dynamical systems analysis engines that examine temporal structure at multiple scales.

**Renormalization engine.** Examine the signal at scales [0.1s, 1s, 10s, 100s, 1000s] simultaneously. Healthy systems have self-similar dynamics across scales. When self-similarity breaks at a specific scale, that scale characterizes the anomaly.

Build `RenormalizationEngine`:
- Per-scale statistics buffers storing: mean, variance, correlation matrix, Shannon entropy over event types, power spectral density (via FFT)
- Coarse-graining operators: linear maps (fit via least squares during calibration) predicting how fine-scale statistics transform to coarse-scale. Self-similarity score = discrepancy between predicted and actual coarse-scale stats.
- Break interpretation: 0.1-1s = "function_level" (deadlocks, hot loops), 1-100s = "system_interaction" (feedback instability, resource starvation), 100-1000s+ = "drift" (constitutional erosion, economic decline)
- Fixed point detection: cluster coarse-scale statistics to identify stable operating modes. Track how cluster centers drift over organism lifetime.
- Returns: similarity breaks per scale pair, anomaly scale + interpretation, fixed points + drift measure.

**Phase space reconstructor.** For key scalar metrics, reconstruct the dynamical attractor using Takens' delay embedding, then estimate attractor dimension and Lyapunov exponents.

Build `PhaseSpaceReconstructor`:
- Target metrics (configurable defaults): nova.decision_latency_ms, axon.execution_success_rate, oikos.economic_ratio, evo.hypothesis_confidence_mean, synapse.cycle_latency_ms, atune.salience_mean, thymos.drive_pressure_total
- Per-metric time series buffers (default 2000 entries)
- Embedding parameter estimation:
  - Delay τ: first minimum of average mutual information (compute AMI at lags 1..50)
  - Dimension d: false nearest neighbors (test d=1..10, choose smallest where FNN% < 1%)
- Attractor analysis:
  - Correlation dimension via Grassberger-Procaccia: correlation integral C(ε) for range of ε, dimension from log-log slope
  - Largest Lyapunov exponent via Rosenstein et al.: nearest neighbors in embedding space, average divergence rate, fit linear region
- Returns per metric: embedding params, correlation dimension + trend (stable/increasing/collapsing), largest Lyapunov + interpretation (chaotic/stable/dissipative), predictability horizon (~1/λ₁)
- Runs every ~75s.

Dependencies: numpy, scipy only. All algorithms implemented directly — the math is straightforward with numpy's linear algebra.

---

## Round 2 — Single Agent (Opus)

### Phase F: Full Integration & Healing Loop

Working in `.code/EcodiaOS/backend/ecodiaos/`. Start by reading `systems/soma/` thoroughly — there are now many new modules alongside the original code. Also read the existing wiring in `main.py`, and check how Thymos, Nova, EIS, and Oneiros currently reference or interact with Soma.

You're wiring all the new analysis engines into the existing SomaService and connecting Soma to the rest of the organism.

**Upgrade `systems/soma/service.py`** — extend the existing service (don't rewrite from scratch) to orchestrate all analysis engines across four paths:
1. **Fast path** (every theta cycle, <5ms inline): ingest → state vector → fast derivatives → Fisher geodesic deviation → broadcast if threshold exceeded
2. **Medium path** (every 100 cycles, async): all derivatives at all scales → causal flow → causal emergence → renormalization → broadcast if warranted
3. **Deep path** (every 500 cycles, background): persistent homology → attractors → Lyapunov → Ricci curvature → comprehensive result
4. **Sleep path** (during Oneiros sleep): extended-scale renormalization, baseline recalibration for all engines, long-term drift analysis

Each engine failure is isolated — one crashing doesn't kill the others. Results cached between cycles. Expose: `coherence_signal` property for Thymos, `latest_analysis()`, `vulnerability_map()`, standard `health()`/`stats`.

**Create `systems/soma/healing.py`** — `HealingVerifier`: after a repair mutation is applied, monitor for 100 cycles. Track geodesic deviation trend, topological features, causal flow, Lyapunov. Classify: HEALED / PARTIAL / INEFFECTIVE / IATROGENIC (made worse → rollback).

**Cross-system wiring** (modify these files):

`systems/thymos/service.py` — Subscribe to INTEROCEPTIVE_PERCEPT. Wire `soma.coherence_signal` into Coherence drive: `coherence_pressure = 1.0 - soma.coherence_signal`. Suppress Growth drive on dynamical instability.

`systems/nova/service.py` — Subscribe to INTEROCEPTIVE_PERCEPT. When `recommended_action == ATTEND_INWARD`, prioritize investigating the flagged system.

`systems/eis/service.py` — Subscribe to INTEROCEPTIVE_PERCEPT. Elevate threat posture when urgency > 0.7.

`systems/oneiros/service.py` — During sleep, call Soma's sleep analysis for long-horizon recalibration. Feed slow-scale findings as dream content.

`config.py` — Add SomaConfig with tunable parameters for all cycle frequencies, buffer sizes, thresholds, target metrics.

`main.py` — Wire Soma with `set_synapse()` and connections to Thymos/Nova/EIS/Oneiros. Ensure startup order is correct (Soma after Synapse).

`systems/synapse/degradation.py` — Add `"soma"` entry: non-critical, no safe mode trigger, fallback = skip interoceptive analysis.
