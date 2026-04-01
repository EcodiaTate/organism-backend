"""
EcodiaOS - Genome Distance Calculator (Spec 26, Speciation Bible §8.4)

Computes Hamming-style distance between two organism genomes for:
  1. Speciation detection - distance > threshold = distinct species
  2. Mating compatibility - reproductively isolated instances cannot exchange
     genetic material
  3. Phylogenetic tree construction - distance matrix for lineage visualization

Distance is deterministic: same inputs always produce the same output.

Per-segment distance methods:
  evo     - Jaccard distance of hypothesis IDs + cosine of drive weight vectors
  simula  - normalized L2 of learnable config parameter vectors
  telos   - cosine distance of drive calibration vectors
  equor   - Jaccard of adopted amendment IDs + L1 of drive calibration deltas

Total = 0.30 × evo + 0.25 × simula + 0.25 × telos + 0.20 × equor
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


# -- Segment weights in the total distance formula ---------------------------

_SEGMENT_WEIGHTS: dict[str, float] = {
    "evo": 0.30,
    "simula": 0.25,
    "telos": 0.25,
    "equor": 0.20,
}


@dataclass(frozen=True)
class GenomeDistance:
    """Genome distance between two instances, computed per-segment."""

    evo_distance: float = 0.0       # Normalized [0, 1]
    simula_distance: float = 0.0
    telos_distance: float = 0.0
    equor_distance: float = 0.0
    total_distance: float = 0.0     # Weighted combination
    is_reproductively_isolated: bool = False  # total > threshold


class GenomeDistanceCalculator:
    """
    Compute Hamming-style distance between two organism genomes.

    Inputs are raw genome dicts as transported via CHILD_SPAWNED payload
    (i.e. the serialised forms of BeliefGenome, SimulaGenome, TelosGenomeFragment,
    EquorGenomeFragment). The calculator handles missing segments gracefully -
    absent segments contribute 0.0 to the total (maximum similarity).

    Parameters
    ----------
    speciation_threshold : float
        Distance above which two instances are considered reproductively
        isolated. Sourced from config.mitosis_speciation_distance_threshold.
    """

    def __init__(self, speciation_threshold: float = 0.3) -> None:
        self._threshold = speciation_threshold
        self._log = logger.bind(subsystem="mitosis.genome_distance")

    # -- Public API ----------------------------------------------------------

    def compute(self, genome_a: dict, genome_b: dict) -> GenomeDistance:
        """
        Compute normalized genome distance between two serialised genomes.

        Each genome dict is expected to have top-level keys matching segment
        names: "evo", "simula", "telos", "equor". Each value is the dict
        representation of the corresponding genome fragment (BeliefGenome,
        SimulaGenome, etc.).

        Returns GenomeDistance with per-segment and weighted total distances.
        """
        evo_d = self._evo_distance(
            genome_a.get("evo") or {},
            genome_b.get("evo") or {},
        )
        simula_d = self._simula_distance(
            genome_a.get("simula") or {},
            genome_b.get("simula") or {},
        )
        telos_d = self._telos_distance(
            genome_a.get("telos") or {},
            genome_b.get("telos") or {},
        )
        equor_d = self._equor_distance(
            genome_a.get("equor") or {},
            genome_b.get("equor") or {},
        )

        total = (
            _SEGMENT_WEIGHTS["evo"]    * evo_d
            + _SEGMENT_WEIGHTS["simula"] * simula_d
            + _SEGMENT_WEIGHTS["telos"]  * telos_d
            + _SEGMENT_WEIGHTS["equor"]  * equor_d
        )
        total = min(1.0, max(0.0, total))

        isolated = total > self._threshold

        result = GenomeDistance(
            evo_distance=round(evo_d, 6),
            simula_distance=round(simula_d, 6),
            telos_distance=round(telos_d, 6),
            equor_distance=round(equor_d, 6),
            total_distance=round(total, 6),
            is_reproductively_isolated=isolated,
        )

        self._log.debug(
            "genome_distance_computed",
            evo=result.evo_distance,
            simula=result.simula_distance,
            telos=result.telos_distance,
            equor=result.equor_distance,
            total=result.total_distance,
            isolated=isolated,
            threshold=self._threshold,
        )

        return result

    def is_reproductively_isolated(self, genome_a: dict, genome_b: dict) -> bool:
        """Return True if these two instances cannot exchange genetic material."""
        return self.compute(genome_a, genome_b).is_reproductively_isolated

    # -- Per-segment distance methods ----------------------------------------

    def _evo_distance(self, seg_a: dict, seg_b: dict) -> float:
        """
        Evo segment distance:
          0.5 × Jaccard distance of hypothesis IDs
          + 0.5 × cosine distance of drive weight vectors

        If both segments are empty, returns 0.0 (identical / no data).
        """
        if not seg_a and not seg_b:
            return 0.0

        # -- Hypothesis Jaccard distance ------------------------------------
        hyps_a: set[str] = {
            str(h.get("id", h.get("hypothesis_id", i)))
            for i, h in enumerate(seg_a.get("top_50_hypotheses", []))
        }
        hyps_b: set[str] = {
            str(h.get("id", h.get("hypothesis_id", i)))
            for i, h in enumerate(seg_b.get("top_50_hypotheses", []))
        }
        jaccard_hyp = _jaccard_distance(hyps_a, hyps_b)

        # -- Drive weight cosine distance -----------------------------------
        drives = ["coherence", "care", "growth", "honesty"]
        dws_a = seg_a.get("drive_weight_snapshot") or {}
        dws_b = seg_b.get("drive_weight_snapshot") or {}
        vec_a = [float(dws_a.get(d, 0.25)) for d in drives]
        vec_b = [float(dws_b.get(d, 0.25)) for d in drives]
        cosine_d = _cosine_distance(vec_a, vec_b)

        return 0.5 * jaccard_hyp + 0.5 * cosine_d

    def _simula_distance(self, seg_a: dict, seg_b: dict) -> float:
        """
        Simula segment distance: normalized L2 of learnable config parameter
        vectors.

        Parameters are compared by name. Missing parameters contribute max
        distance (1.0) to that dimension. Distance is normalized to [0, 1]
        by the number of parameters.
        """
        if not seg_a and not seg_b:
            return 0.0

        params_a: dict[str, float] = {
            str(p.get("name", i)): float(p.get("value", 0.0))
            for i, p in enumerate(seg_a.get("parameters", []))
            if isinstance(p, dict)
        }
        params_b: dict[str, float] = {
            str(p.get("name", i)): float(p.get("value", 0.0))
            for i, p in enumerate(seg_b.get("parameters", []))
            if isinstance(p, dict)
        }

        all_keys = set(params_a) | set(params_b)
        if not all_keys:
            return 0.0

        # Normalized L2: sum of squared per-param deltas, divided by param count
        # Each param assumed bounded [0, 1]; delta in [0, 1] so squared in [0, 1]
        sq_sum = 0.0
        for k in all_keys:
            va = params_a.get(k, 0.0)
            vb = params_b.get(k, 0.0)
            # Normalize by expected range; if values > 1, clamp contribution
            delta = abs(va - vb)
            # Clamp to [0, 1] per dimension
            sq_sum += min(1.0, delta) ** 2

        return min(1.0, math.sqrt(sq_sum / len(all_keys)))

    def _telos_distance(self, seg_a: dict, seg_b: dict) -> float:
        """
        Telos segment distance: cosine distance of drive calibration vectors.

        The telos genome fragment stores per-drive calibration dicts under
        "drive_calibrations". We flatten the numeric fields of each drive
        into a single vector for comparison.
        """
        if not seg_a and not seg_b:
            return 0.0

        drives = ["coherence", "care", "growth", "honesty"]
        _CALIB_FIELDS = ["resonance_curve_coefficients", "dissipation_baseline", "coupling_strength"]

        def _flatten_calib(seg: dict) -> list[float]:
            calibs: dict[str, dict] = seg.get("drive_calibrations") or {}
            vec: list[float] = []
            for drive in drives:
                calib = calibs.get(drive) or {}
                for field_name in _CALIB_FIELDS:
                    val = calib.get(field_name)
                    if isinstance(val, list):
                        vec.extend(float(v) for v in val)
                    elif isinstance(val, (int, float)):
                        vec.append(float(val))
                    else:
                        vec.append(0.0)
            return vec

        vec_a = _flatten_calib(seg_a)
        vec_b = _flatten_calib(seg_b)

        if not vec_a and not vec_b:
            return 0.0

        # Pad to equal length
        max_len = max(len(vec_a), len(vec_b))
        if len(vec_a) < max_len:
            vec_a += [0.0] * (max_len - len(vec_a))
        if len(vec_b) < max_len:
            vec_b += [0.0] * (max_len - len(vec_b))

        return _cosine_distance(vec_a, vec_b)

    def _equor_distance(self, seg_a: dict, seg_b: dict) -> float:
        """
        Equor segment distance:
          0.5 × Jaccard distance of adopted amendment IDs
          + 0.5 × L1 distance of drive calibration deltas (normalised)

        Amendment IDs are strings. Drive calibration deltas are per-drive
        float dicts stored under "drive_calibration_deltas".
        """
        if not seg_a and not seg_b:
            return 0.0

        # -- Amendment Jaccard distance -------------------------------------
        amend_a: set[str] = {
            str(a.get("amendment_id", a.get("id", i)))
            for i, a in enumerate(seg_a.get("amendments", []))
            if isinstance(a, dict)
        }
        amend_b: set[str] = {
            str(a.get("amendment_id", a.get("id", i)))
            for i, a in enumerate(seg_b.get("amendments", []))
            if isinstance(a, dict)
        }
        jaccard_amend = _jaccard_distance(amend_a, amend_b)

        # -- Drive calibration delta L1 distance ----------------------------
        drives = ["coherence", "care", "growth", "honesty"]
        deltas_a: dict[str, float] = seg_a.get("drive_calibration_deltas") or {}
        deltas_b: dict[str, float] = seg_b.get("drive_calibration_deltas") or {}

        if deltas_a or deltas_b:
            l1_total = sum(
                abs(float(deltas_a.get(d, 0.0)) - float(deltas_b.get(d, 0.0)))
                for d in drives
            )
            # Normalize: each delta bounded ~[-2, 2] → max L1 across 4 drives ≈ 8
            l1_norm = min(1.0, l1_total / 4.0)
        else:
            l1_norm = 0.0

        return 0.5 * jaccard_amend + 0.5 * l1_norm


# -- Math utilities ----------------------------------------------------------

def _jaccard_distance(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard distance = 1 - |A ∩ B| / |A ∪ B|. Empty sets → 0.0."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return 1.0 - intersection / union


def _cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine distance = 1 - cosine_similarity. Zero vectors → 1.0."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        # One vector is all-zeros: maximum distance
        return 1.0
    similarity = dot / (norm_a * norm_b)
    # Clamp to [-1, 1] before subtracting to avoid floating-point overflow
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity
