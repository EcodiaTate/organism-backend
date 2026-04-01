"""
EcodiaOS - Genome Inheritance Schemas (Oikos SG4 / Spec 17 v2.1 / Spec 18 SG3)

Defines the heritable genetic payloads that are resolved and attached at
child-spawn time. These are the actual content schemas for belief_genome_id
(Evo's BeliefGenome), simula_genome_id (Simula's SimulaGenome),
equor_genome_id (Equor's EquorGenomeFragment), and telos_genome_id
(Telos's TelosGenomeFragment).

All types are:
  - JSON-serialisable (Pydantic model_dump(mode="json"))
  - Transported via Synapse CHILD_SPAWNED payload
  - Immutable once extracted (genome IDs reference a frozen snapshot)

Drive weights are evolvable phenotype. The BeliefGenome carries a snapshot
of drive_weights so that Oikos economic pressure (metabolic_efficiency < 0.8)
becomes a selection pressure on children - children with misaligned drive
weights carry their parent's economic stress into their genome, enabling Evo
and Equor to propose weight rebalancing via constitutional amendment.

The EquorGenomeFragment carries the constitutional amendment history so that
children inherit not just the parent's drive weights but the *wisdom* behind
each constitutional refinement - preserving normative knowledge across generations.

The TelosGenomeFragment (Spec 18 SG3) carries the drive calibration constants
(resonance curves, dissipation baseline, inter-drive coupling) so that drive
geometry evolves across generations. Children apply parent calibrations with
bounded mutation jitter, enabling multi-generational topology adaptation.

# Version Migration
# Only Evo (genetic_memory.py) has migration logic (_migrate_genome_item()).
# SimulaGenome, EquorGenomeFragment, and TelosGenomeFragment all carry
# genome_version fields but have no migration path. If their schemas change,
# child deserialization will fail silently or raise a ValidationError.
# TODO: Add versioned migration functions per genome type before any schema change.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now


class DriveWeightSnapshot(EOSBaseModel):
    """Point-in-time snapshot of the four constitutional drive weights."""

    coherence: float = 0.20
    care: float = 0.35
    growth: float = 0.15
    honesty: float = 0.30
    captured_at: datetime = Field(default_factory=utc_now)
    metabolic_efficiency_at_capture: float = 1.0  # Oikos efficiency when captured


class DriftHistoryEntry(EOSBaseModel):
    """One entry in the drive drift history (alignment composite at a point in time)."""

    timestamp: datetime = Field(default_factory=utc_now)
    composite_alignment: float = 0.0
    primary_cause: str = ""


class BeliefGenome(EOSBaseModel):
    """
    Evo's heritable belief state - the cognitive DNA of the organism's learning.

    Serialised to JSON and stored under organism_genome_id at spawn time.
    Children initialise their Thompson samplers and hypothesis priors from this.

    Schema (stable across generations):
      hypothesis_set_embedding   - 768-dim centroid of the top-50 hypothesis
                                   embeddings (compressed to 64-dim via PCA for
                                   transport; child reconstructs approximate prior)
      top_50_hypotheses          - ranked dicts: {id, statement, evidence_score,
                                   confidence, category, supporting_count}
      drive_weight_snapshot      - current constitutional drive weights (evolvable
                                   phenotype; Oikos economic pressure selects on this)
      drift_history              - last 10 constitutional drift entries so the child
                                   starts with its parent's alignment trajectory
      generation                 - lineage depth (parent's generation + 1)
    """

    genome_id: str = Field(default_factory=new_id)
    instance_id: str = ""
    generation: int = 1
    extracted_at: datetime = Field(default_factory=utc_now)

    # Cognitive priors
    hypothesis_set_embedding: list[float] = Field(default_factory=list)  # 64-dim PCA
    top_50_hypotheses: list[dict[str, Any]] = Field(default_factory=list)

    # Constitutional phenotype (evolvable under metabolic selection pressure)
    drive_weight_snapshot: DriveWeightSnapshot = Field(default_factory=DriveWeightSnapshot)

    # Alignment trajectory (last 10 entries)
    drift_history: list[DriftHistoryEntry] = Field(default_factory=list)

    # Thompson tournament Beta priors - keyed by hypothesis_id.
    # Each entry: {"alpha": float, "beta": float, "sample_count": int,
    #              "tournament_id": str, "hypothesis_statement": str[:120]}
    # Children seed their Beta distributions from these so tournaments don't
    # restart from uniform priors. Only populated when sample_count >= 5
    # (noise threshold) - single-trial priors don't carry real signal.
    tournament_beta_priors: list[dict[str, Any]] = Field(default_factory=list)

    # Integrity checksum: sha256 of canonical fields (genome_id + instance_id
    # + str(generation) + str(len(top_50_hypotheses)) + str(len(tournament_beta_priors))).
    # Verified on child-side apply; mismatch → warning logged, inheritance skipped.
    genome_checksum: str = ""

    def _compute_checksum(self) -> str:
        """Compute integrity checksum over canonical genome fields."""
        import hashlib
        payload = (
            f"{self.genome_id}|{self.instance_id}|{self.generation}"
            f"|{len(self.top_50_hypotheses)}|{len(self.tournament_beta_priors)}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def seal(self) -> "BeliefGenome":
        """Compute and store checksum. Call after all fields are set."""
        self.genome_checksum = self._compute_checksum()
        return self

    def verify(self) -> bool:
        """Return True if stored checksum matches recomputed checksum."""
        if not self.genome_checksum:
            return True  # Legacy genomes without checksum are trusted
        return self.genome_checksum == self._compute_checksum()

    def model_dump_for_transport(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for Synapse transport."""
        return self.model_dump(mode="json")


class AmendmentSnapshot(EOSBaseModel):
    """
    Frozen record of a single adopted constitutional amendment.

    Captured at spawn time so the child can reconstruct the parent's
    reasoning for each drive calibration - not just the resulting weights.
    """

    amendment_id: str = ""
    title: str = ""
    description: str = ""
    rationale: str = ""                          # Why this amendment was adopted
    drive_id: str = ""                           # Which drive was primarily affected
    delta: float = 0.0                           # Net drive weight change (signed)
    proposed_drives: dict[str, float] = Field(default_factory=dict)
    previous_drives: dict[str, float] = Field(default_factory=dict)
    proposer: str = ""
    adopted_at: str = ""                         # ISO-8601 timestamp


class EquorGenomeFragment(EOSBaseModel):
    """
    Equor's heritable constitutional state - the normative DNA of the organism.

    Serialised to JSON and stored under equor_genome_id at spawn time.
    Children apply inherited amendments during Equor initialisation so they
    start with the parent's constitutional refinements, not bare defaults.

    Schema (stable across generations):
      top_amendments            - Last 10 adopted amendments with rationale
      amendment_rationale       - Plain-text "why" for each amendment (same order)
      drive_calibration_deltas  - Cumulative signed change per drive across all amendments
      constitution_hash         - SHA-256 of the live constitutional state at extraction
      generation                - Lineage depth
    """

    genome_id: str = Field(default_factory=new_id)
    instance_id: str = ""
    generation: int = 1
    extracted_at: datetime = Field(default_factory=utc_now)

    # Amendment history (up to last 10, ordered oldest-first)
    top_amendments: list[AmendmentSnapshot] = Field(default_factory=list)

    # Plain-text rationale for each amendment (parallel to top_amendments)
    amendment_rationale: list[str] = Field(default_factory=list)

    # Cumulative drive calibration deltas across all amendments in this fragment
    # Key: drive name ("coherence", "care", "growth", "honesty"), Value: net signed delta
    drive_calibration_deltas: dict[str, float] = Field(default_factory=dict)

    # SHA-256 of the constitutional document at extraction time
    constitution_hash: str = ""

    # Total adopted amendments ever (not just those in this fragment)
    total_amendments_adopted: int = 0

    def model_dump_for_transport(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for Synapse transport."""
        return self.model_dump(mode="json")


# ── Nova Genome (Spec 05) ────────────────────────────────────────────────────


class NovaGenomeFragment(EOSBaseModel):
    """
    Nova's heritable decision-making state - the cognitive DNA of the organism's
    planning and active inference capability.

    Serialised to JSON and stored under nova_genome_id at spawn time.
    Children initialise their goal-domain priors, policy scoring weights, belief
    urgency thresholds, and active inference parameters from this fragment, so
    they start deliberating like their parent rather than from blank defaults.

    Schema (stable across generations):
      goal_domain_priors        - per-domain prior weight for goal success rates
                                  (keys: target_domain strings, values: [0.0, 1.0])
      policy_success_rates      - per-policy template prior success rates inherited
                                  from parent (keys: policy_name, values: [0.0, 1.0])
      belief_urgency_thresholds - per-belief-key threshold for BeliefUrgencyMonitor
                                  (keys: belief entity key, values: confidence delta [0.0, 1.0])
      active_inference_params   - EFE weight priors and precision scalars:
                                  pragmatic, epistemic, constitutional, feasibility,
                                  risk, cognition_cost, precision_base
      generation                - Lineage depth
    """

    genome_id: str = Field(default_factory=new_id)
    instance_id: str = ""
    generation: int = 1
    extracted_at: datetime = Field(default_factory=utc_now)

    # Per-domain prior weights from parent's goal achievement history
    goal_domain_priors: dict[str, float] = Field(default_factory=dict)

    # Per-policy success rates from parent's deliberation record
    policy_success_rates: dict[str, float] = Field(default_factory=dict)

    # Urgency threshold overrides for BeliefUrgencyMonitor
    # Lower threshold → more sensitive to belief shifts
    belief_urgency_thresholds: dict[str, float] = Field(default_factory=dict)

    # Active inference EFE weight configuration
    active_inference_params: dict[str, float] = Field(default_factory=dict)

    # Thompson arm Beta history for N-armed provider sampler.
    # Each entry: {"arm_name": str, "alpha": float, "beta": float,
    #              "total_trials": int, "consecutive_failures": int, "ready": bool}
    # Children seed their provider arms from this so they don't start with flat
    # priors on Claude vs RE. Only included when total_trials >= 10 (evidence
    # threshold - fewer trials carry more noise than signal).
    # Re-enabled arms start with confidence discounted: alpha *= 0.85 on apply.
    thompson_arm_history: list[dict[str, Any]] = Field(default_factory=list)

    def model_dump_for_transport(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for Synapse transport."""
        return self.model_dump(mode="json")


# ── Voxis Genome (Spec 04) ───────────────────────────────────────────────────


class VoxisGenomeFragment(EOSBaseModel):
    """
    Voxis's heritable communicative identity - the voice DNA of the organism.

    Serialised to JSON and stored under voxis_genome_id at spawn time.
    Children apply the parent's personality vector (with bounded ±10% jitter),
    vocabulary affinities, and strategy preferences so they speak like their
    parent from the first cognitive cycle rather than starting from a neutral seed.

    Schema (stable across generations):
      personality_vector        - 9-dimensional personality snapshot:
                                  warmth, directness, verbosity, formality,
                                  curiosity_expression, humour, empathy_expression,
                                  confidence_display, metaphor_use
      vocabulary_affinities     - Top-500 word/phrase → affinity weight pairs
                                  from parent's expression history
      strategy_preferences      - Accumulated strategy type frequencies:
                                  pragmatic, epistemic, affiliative preference weights
      generation                - Lineage depth
    """

    genome_id: str = Field(default_factory=new_id)
    instance_id: str = ""
    generation: int = 1
    extracted_at: datetime = Field(default_factory=utc_now)

    # 9D personality snapshot (keys match PersonalityVector fields)
    personality_vector: dict[str, float] = Field(default_factory=dict)

    # Top vocabulary/phrase affinity weights from parent
    vocabulary_affinities: dict[str, float] = Field(default_factory=dict)

    # Expression strategy preference weights (pragmatic/epistemic/affiliative)
    strategy_preferences: dict[str, float] = Field(default_factory=dict)

    def model_dump_for_transport(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for Synapse transport."""
        return self.model_dump(mode="json")


class AxonTemplateSnapshot(EOSBaseModel):
    """
    Frozen snapshot of a single learned action template at spawn time.

    Represents an action pattern that the parent instance has refined
    across executions. Children inherit these as fast-path seeds so they
    start with pre-validated execution strategies rather than defaults.

    Schema (stable across generations):
      action_pattern       - executor action type (e.g. "negotiate_resource")
      cached_approvals     - list of pre-approved action variants by Equor
      expected_cost_mean   - rolling mean execution cost in USD
      expected_cost_variance - cost variance (higher = less predictable)
      success_rate         - fraction of last 100 executions that succeeded
    """

    action_pattern: str = ""
    cached_approvals: list[str] = Field(default_factory=list)
    expected_cost_mean: float = 0.0
    expected_cost_variance: float = 0.0
    success_rate: float = 0.0


class AxonGenomeFragment(EOSBaseModel):
    """
    Axon's heritable execution intelligence - the motor DNA of the organism.

    Serialised to JSON and stored under axon_genome_id at spawn time.
    Children initialise their template library and circuit breaker thresholds
    from this, skipping the cold-start period where execution quality is low.

    Schema (stable across generations):
      templates                  - Top 10 action templates by success_rate
      circuit_breaker_thresholds - Per-action failure limits inherited from parent
      template_confidence        - How much to trust each inherited template (0.5–1.0)
      generation                 - Lineage depth
    """

    genome_id: str = Field(default_factory=new_id)
    instance_id: str = ""
    generation: int = 1
    extracted_at: datetime = Field(default_factory=utc_now)

    # Top templates by success_rate (up to 10)
    templates: list[AxonTemplateSnapshot] = Field(default_factory=list)

    # Per-action circuit breaker failure limits from parent experience
    circuit_breaker_thresholds: dict[str, int] = Field(default_factory=dict)

    # Confidence weights per action_pattern (how much to trust each template)
    # Computed as: max(0.5, success_rate) - anchored above 0.5 since templates
    # are pre-selected for quality; raw 0.0 would mask all templates.
    template_confidence: dict[str, float] = Field(default_factory=dict)

    def model_dump_for_transport(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for Synapse transport."""
        return self.model_dump(mode="json")


class SimulaMutationEntry(EOSBaseModel):
    """Record of a single applied or rolled-back mutation."""

    mutation_id: str = ""
    category: str = ""           # ChangeCategory value
    description: str = ""
    applied_at: datetime = Field(default_factory=utc_now)
    was_rolled_back: bool = False
    risk_level: str = "low"
    efe_score: float | None = None


class SimulaGenome(EOSBaseModel):
    """
    Simula's heritable evolution state - the structural DNA of the organism's
    self-modification capability.

    Children start with their parent's validated evolution parameters so they
    don't repeat failed experiments. The dafny_spec_hashes allow the child to
    verify its own structural invariants against the parent's proven contracts.

    Schema (stable across generations):
      current_evolution_params   - dict of live Simula config (learnable params,
                                   budget multipliers, Z3 timeout, etc.)
      last_10_mutations          - most recent applied mutations ordered by
                                   applied_at DESC; child uses these to seed its
                                   own "already tried" memory
      dafny_spec_hashes          - sha256 hashes of the Dafny spec files that
                                   the parent has formally verified; child can skip
                                   re-verification of unchanged specs
      generation                 - lineage depth
    """

    genome_id: str = Field(default_factory=new_id)
    instance_id: str = ""
    generation: int = 1
    extracted_at: datetime = Field(default_factory=utc_now)

    # Evolution configuration (learnable params from PHASE_A_IMPLEMENTATION)
    current_evolution_params: dict[str, Any] = Field(default_factory=dict)

    # Recent mutation history (last 10, for child "already tried" seeding)
    last_10_mutations: list[SimulaMutationEntry] = Field(default_factory=list)

    # Formally-verified spec fingerprints
    dafny_spec_hashes: dict[str, str] = Field(default_factory=dict)  # path → sha256

    # Extensible metadata (reasoning router weights, EFE calibration, etc.)
    # Forwards-compatible: older children ignore unknown keys.
    extra: dict[str, Any] = Field(default_factory=dict)

    def model_dump_for_transport(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for Synapse transport."""
        return self.model_dump(mode="json")


# ── Telos Genome (Spec 18 SG3) ─────────────────────────────────────────────


class TeloDriveCalibration(EOSBaseModel):
    """
    Heritable calibration constants for a single drive's topology contribution.

    These are the knobs that control *how* a drive warps the intelligence
    landscape - not the drive weights themselves (which live in BeliefGenome /
    EquorGenomeFragment) but the physics of each drive's influence curve.

    Mutation ranges (per generation, Spec 18 SG3):
      resonance_curve_coefficients  ±15% per generation
      dissipation_baseline          ±10% per generation
      coupling_strength values      ±20% per generation
    """

    # Per-drive resonance curve polynomial coefficients.
    # Key: coefficient name (e.g. "a0", "a1", "a2"); value: coefficient.
    # Governs how the drive score maps to the multiplier on effective_I.
    resonance_curve_coefficients: dict[str, float] = Field(default_factory=dict)

    # Thermodynamic work baseline - minimum dissipation this drive "costs"
    # regardless of alignment level.  Lower = more efficient drive topology.
    dissipation_baseline: float = 0.0

    # Inter-drive coupling strengths.
    # coupling_strength["care"]["coherence"] = how much care score boosts
    # (or penalises) the coherence multiplier.
    # Outer key: source drive, inner key: target drive.
    coupling_strength: dict[str, dict[str, float]] = Field(default_factory=dict)

    # Mutation bounds for each calibration parameter (Spec 18 SG3).
    # Key: parameter path (e.g. "resonance_curve_coefficients.a1"),
    # Value: (min_fraction, max_fraction) relative to current value.
    mutation_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)

    # Timestamp of last adaptation (mutation applied or genome seeded)
    last_adapted: datetime = Field(default_factory=utc_now)


class AtuneGenomeFragment(EOSBaseModel):
    """
    Atune/Fovea's heritable perceptual bandwidth and curiosity rhythm.

    Serialised to JSON and stored under atune_genome_id at spawn time.
    Children apply the parent's curiosity parameters (with bounded ±5% jitter)
    and buffer-size arousal scaling so they inherit the parent's attentional
    rhythm from the first cognitive cycle rather than cold-starting at defaults.

    Schema (stable across generations):
      curiosity_params    - base_prob, cooldown_cycles, curiosity_boost
      buffer_scale_arousal - the arousal level at which parent settled buffers
                             (child uses this as its initial arousal prior for
                             sizing queues before Soma's first tick arrives)
      curiosity_hit_rate  - fraction of spontaneous recalls that produced
                             positive outcomes (Thread coherence or Evo
                             hypothesis) in the parent. Informational - child
                             can use this as a confidence prior.
      generation          - Lineage depth
    """

    genome_id: str = Field(default_factory=new_id)
    instance_id: str = ""
    generation: int = 1
    extracted_at: datetime = Field(default_factory=utc_now)

    # Evolvable curiosity parameters
    curiosity_params: dict[str, float] = Field(default_factory=dict)
    # e.g. {"base_prob": 0.02, "cooldown_cycles": 20.0, "curiosity_boost": 0.03}

    # Parent's settled arousal level - used to size queues before first Soma tick
    buffer_scale_arousal: float = 0.4

    # Curiosity effectiveness metric from parent's runtime
    curiosity_hit_rate: float = 0.5

    def model_dump_for_transport(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for Synapse transport."""
        return self.model_dump(mode="json")


class TelosGenomeFragment(EOSBaseModel):
    """
    Telos's heritable drive calibration state - the geometric DNA of the organism.

    Serialised to JSON and stored under telos_genome_id at spawn time.
    Children apply inherited calibrations (with bounded mutation jitter) during
    Telos initialisation so drive topology evolves rather than resetting to
    hardcoded defaults each generation.

    Schema (stable across generations, Spec 18 SG3):
      drive_calibrations  - Per-drive TeloDriveCalibration (care/coherence/growth/honesty)
      topology            - Active topology mode: "linear" | "hierarchical" | "network"
      topology_parameters - Per-topology weighting params (e.g. edge weights for "network")
      generation          - Lineage depth
    """

    genome_id: str = Field(default_factory=new_id)
    instance_id: str = ""
    generation: int = 1
    extracted_at: datetime = Field(default_factory=utc_now)

    # Per-drive calibration constants (one entry per constitutional drive)
    drive_calibrations: dict[str, TeloDriveCalibration] = Field(default_factory=dict)

    # Active drive topology mode
    topology: str = "linear"  # "linear" | "hierarchical" | "network"

    # Topology-specific parameters (edge weights, hierarchy levels, etc.)
    topology_parameters: dict[str, Any] = Field(default_factory=dict)

    def model_dump_for_transport(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict for Synapse transport."""
        return self.model_dump(mode="json")


class PersonaFragment(EOSBaseModel):
    """
    Heritable persona DNA - carried in CHILD_SPAWNED payload so children
    inherit their parent's brand identity with bounded mutation.

    Inheritance rules:
      voice_style       - inherited as-is; child may mutate via Voxis
                          personality_vector jitter (±10%) over time
      professional_domain - inherited; Telos may override during child
                            specialisation (domain is evolvable, not fixed)
      brand_lineage     - parent appends own handle; children extend the
                          chain → federation peers can trace ancestry
      avatar_seed_base  - child derives own seed as
                          f"{avatar_seed_base}-{child_instance_id[:8]}"
                          ensuring visual family resemblance with uniqueness

    Children do NOT inherit the parent's handle or display_name - those are
    always freshly generated (unique per instance).

    Schema (stable across generations):
      voice_style         - "technical-precise" | "curious-accessible" |
                            "analytical-dry" | "warm-collaborative" |
                            "concise-systematic"
      professional_domain - parent's current Telos specialisation
      brand_lineage       - ordered list of ancestor handles (oldest first)
      avatar_seed_base    - parent's avatar_seed (child derives from this)
      parent_handle       - parent's handle (appended to brand_lineage)
      generation          - lineage depth (parent.generation + 1)
    """

    genome_id: str = Field(default_factory=new_id)
    instance_id: str = ""  # parent's instance_id
    generation: int = 1
    extracted_at: datetime = Field(default_factory=utc_now)

    # Heritable persona fields
    voice_style: str = "analytical-dry"
    professional_domain: str = ""
    brand_lineage: list[str] = Field(default_factory=list)  # ancestor handles
    avatar_seed_base: str = ""       # child computes own seed from this base
    parent_handle: str = ""          # parent's handle for lineage chain

    def model_dump_for_transport(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def child_avatar_seed(self, child_instance_id: str) -> str:
        """Derive a child's avatar seed that visually relates to the parent."""
        return f"{self.avatar_seed_base}-{child_instance_id[:8]}"

    def child_brand_lineage(self) -> list[str]:
        """Return the lineage list the child should receive (parent appended)."""
        return [*self.brand_lineage, self.parent_handle] if self.parent_handle else list(self.brand_lineage)
