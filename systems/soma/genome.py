"""
EcodiaOS - Soma Genome Extraction & Seeding

Heritable state: setpoint parameters, phase-space configuration,
developmental stage thresholds, allostatic baselines.

Child starts at developmental stage 0 but with parent's calibrated setpoints.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment
from systems.soma.types import ALL_DIMENSIONS, InteroceptiveDimension

if TYPE_CHECKING:
    from systems.soma.service import SomaService

logger = structlog.get_logger()

# Noise parameters for child genome inheritance (GAP 6)
_SETPOINT_NOISE_FRACTION: float = 0.05   # ±5% on setpoints
_DYNAMICS_NOISE_FRACTION: float = 0.02   # ±2% on dynamics weights


class SomaGenomeExtractor:
    """Extracts Soma interoceptive calibration for genome transmission."""

    def __init__(self, service: SomaService) -> None:
        self._service = service
        self._log = logger.bind(subsystem="soma.genome")

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        try:
            setpoints = self._extract_setpoints()
            phase_space_config = self._extract_phase_space_config()
            stage_thresholds = self._extract_stage_thresholds()
            allostatic_baselines = self._extract_allostatic_baselines()

            if not setpoints:
                return build_segment(SystemID.SOMA, {}, version=0)

            payload = {
                "setpoints": setpoints,
                "phase_space_config": phase_space_config,
                "stage_thresholds": stage_thresholds,
                "allostatic_baselines": allostatic_baselines,
            }

            self._log.info(
                "soma_genome_extracted",
                setpoint_count=len(setpoints),
            )
            return build_segment(SystemID.SOMA, payload, version=1)

        except Exception as exc:
            self._log.error("soma_genome_extract_failed", error=str(exc))
            return build_segment(SystemID.SOMA, {}, version=0)

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        if segment.version == 0 or not segment.payload:
            return True

        if not check_schema_version(segment):
            return False
        if not verify_segment(segment):
            return False

        try:
            payload = segment.payload
            self._apply_setpoints(payload.get("setpoints", {}))
            self._apply_phase_space_config(payload.get("phase_space_config", {}))
            self._apply_allostatic_baselines(payload.get("allostatic_baselines", {}))
            # Do NOT apply stage thresholds as stage - child starts at stage 0

            self._log.info("soma_genome_seeded")
            return True

        except Exception as exc:
            self._log.error("soma_genome_seed_failed", error=str(exc))
            return False

    async def export_somatic_genome(self) -> OrganGenomeSegment:
        """
        Export Soma's full calibrated state as a heritable genome segment
        for Mitosis child spawning (GAP 6).

        Includes setpoints, dynamics matrix, phase-space config, and
        allostatic baselines. Identical to extract_genome_segment() but
        also embeds the dynamics matrix so children inherit coupling
        weights (with noise applied at seeding time).
        """
        try:
            setpoints = self._extract_setpoints()
            phase_space_config = self._extract_phase_space_config()
            stage_thresholds = self._extract_stage_thresholds()
            allostatic_baselines = self._extract_allostatic_baselines()
            dynamics_matrix = self._extract_dynamics_matrix()

            if not setpoints:
                return build_segment(SystemID.SOMA, {}, version=0)

            payload: dict = {
                "setpoints": setpoints,
                "phase_space_config": phase_space_config,
                "stage_thresholds": stage_thresholds,
                "allostatic_baselines": allostatic_baselines,
            }
            if dynamics_matrix:
                payload["dynamics_matrix"] = dynamics_matrix

            self._log.info(
                "soma_somatic_genome_exported",
                setpoint_count=len(setpoints),
                has_dynamics=bool(dynamics_matrix),
            )
            return build_segment(SystemID.SOMA, payload, version=2)

        except Exception as exc:
            self._log.error("soma_somatic_genome_export_failed", error=str(exc))
            return build_segment(SystemID.SOMA, {}, version=0)

    async def seed_child_from_genome(self, segment: OrganGenomeSegment) -> bool:
        """
        Apply parent genome to a child instance with heritable noise.

        Implements GAP 6 inheritance contract:
        - Setpoints:        ±5% uniform noise per dimension
        - Dynamics weights: ±2% uniform noise per weight
        - Developmental stage: always starts at REFLEXIVE (ignored from genome)
        - Phase-space config and allostatic baselines: inherited exactly

        This ensures children are similar to - but not identical to - their parent,
        enabling phenotypic variation under selection pressure.
        """
        if segment.version == 0 or not segment.payload:
            return True

        if not check_schema_version(segment):
            return False
        if not verify_segment(segment):
            return False

        try:
            payload = segment.payload

            # Apply setpoints with ±5% noise
            noisy_setpoints = _apply_setpoint_noise(
                payload.get("setpoints", {}), _SETPOINT_NOISE_FRACTION,
            )
            self._apply_setpoints(noisy_setpoints)

            # Apply dynamics with ±2% noise
            noisy_dynamics = _apply_dynamics_noise(
                payload.get("dynamics_matrix", []), _DYNAMICS_NOISE_FRACTION,
            )
            if noisy_dynamics:
                self._apply_dynamics_matrix(noisy_dynamics)

            # Inherit config exactly (not phenotypic, just structural)
            self._apply_phase_space_config(payload.get("phase_space_config", {}))
            self._apply_allostatic_baselines(payload.get("allostatic_baselines", {}))

            self._log.info(
                "soma_child_genome_seeded",
                setpoint_noise_pct=_SETPOINT_NOISE_FRACTION * 100,
                dynamics_noise_pct=_DYNAMICS_NOISE_FRACTION * 100,
            )
            return True

        except Exception as exc:
            self._log.error("soma_child_genome_seed_failed", error=str(exc))
            return False

    # ── Extraction helpers ─────────────────────────────────────────

    def _extract_setpoints(self) -> dict:
        """Extract interoceptive setpoints from the live allostatic controller.

        Reads directly from the controller's setpoints property so the extracted
        genome reflects the organism's current learned/adapted calibration, not
        static config defaults. Falls back to config if the controller is absent.
        """
        try:
            controller = getattr(self._service, "_controller", None)
            if controller is not None and hasattr(controller, "setpoints"):
                raw = controller.setpoints
                # raw is dict[InteroceptiveDimension, float]
                return {
                    dim.value: float(val)
                    for dim, val in raw.items()
                    if isinstance(dim, InteroceptiveDimension)
                }

            # Fallback: read from config.setpoints dict
            config = self._service._config
            if hasattr(config, "setpoints") and isinstance(config.setpoints, dict):
                return {k: float(v) for k, v in config.setpoints.items()}

            return {}
        except Exception:
            return {}

    def _extract_phase_space_config(self) -> dict:
        """Extract phase-space model configuration."""
        try:
            config = self._service._config
            ps: dict[str, object] = {}
            if hasattr(config, "max_attractors"):
                ps["max_attractors"] = int(config.max_attractors)
            if hasattr(config, "trajectory_buffer_size"):
                ps["trajectory_buffer_size"] = int(config.trajectory_buffer_size)
            if hasattr(config, "prediction_ewm_span"):
                ps["prediction_ewm_span"] = float(config.prediction_ewm_span)
            return ps
        except Exception:
            return {}

    def _extract_stage_thresholds(self) -> list[dict]:
        """Extract developmental stage transition thresholds."""
        try:
            config = self._service._config
            if hasattr(config, "stage_thresholds"):
                thresholds = config.stage_thresholds
                if isinstance(thresholds, list):
                    return [
                        {"stage": t.get("stage", ""), "threshold": t.get("threshold", 0.0)}
                        if isinstance(t, dict)
                        else {"stage": str(t), "threshold": 0.0}
                        for t in thresholds
                    ]
            return []
        except Exception:
            return []

    def _extract_allostatic_baselines(self) -> dict:
        """Extract learned allostatic baselines."""
        try:
            baselines: dict[str, float] = {}
            config = self._service._config
            if hasattr(config, "urgency_threshold"):
                baselines["urgency_threshold"] = float(config.urgency_threshold)
            if hasattr(config, "setpoint_adaptation_alpha"):
                baselines["setpoint_adaptation_alpha"] = float(
                    config.setpoint_adaptation_alpha
                )
            return baselines
        except Exception:
            return {}

    # ── Seeding helpers ────────────────────────────────────────────

    def _apply_setpoints(self, setpoints: dict) -> None:
        """Apply inherited setpoints directly to the live allostatic controller.

        Writes to controller.setpoints[InteroceptiveDimension] so the organism
        immediately starts operating with the parent's calibrated allostatic targets.
        Falls back to config if the controller is absent.
        """
        if not setpoints:
            return
        try:
            controller = getattr(self._service, "_controller", None)
            if controller is not None and hasattr(controller, "setpoints"):
                for dim_str, value in setpoints.items():
                    try:
                        dim = InteroceptiveDimension(dim_str)
                        controller.setpoints[dim] = float(value)
                    except (ValueError, KeyError):
                        pass  # Skip unknown dimension strings
                return
        except Exception:
            pass

        # Fallback: write to config.setpoints dict
        config = self._service._config
        if hasattr(config, "setpoints") and isinstance(config.setpoints, dict):
            config.setpoints.update({k: float(v) for k, v in setpoints.items()})

    def _apply_phase_space_config(self, ps: dict) -> None:
        """Apply inherited phase-space config."""
        if not ps:
            return
        config = self._service._config
        for key, value in ps.items():
            if hasattr(config, key):
                setattr(config, key, type(getattr(config, key))(value))

    def _apply_allostatic_baselines(self, baselines: dict) -> None:
        """Apply inherited allostatic baselines."""
        if not baselines:
            return
        config = self._service._config
        for key, value in baselines.items():
            if hasattr(config, key):
                setattr(config, key, float(value))

    def _extract_dynamics_matrix(self) -> list[list[float]]:
        """Extract the 9×9 cross-dimension coupling matrix from the live predictor."""
        try:
            predictor = getattr(self._service, "_predictor", None)
            if predictor is None:
                return []
            dm = getattr(predictor, "dynamics_matrix", None)
            if dm is None:
                return []
            # dm may be np.ndarray or list[list[float]]
            if hasattr(dm, "tolist"):
                return dm.tolist()
            return [list(row) for row in dm]
        except Exception:
            return []

    def _apply_dynamics_matrix(self, matrix: list[list[float]]) -> None:
        """Apply an inherited dynamics matrix to the live predictor."""
        if not matrix:
            return
        try:
            predictor = getattr(self._service, "_predictor", None)
            if predictor is not None and hasattr(predictor, "update_dynamics"):
                predictor.update_dynamics(matrix)
                # Sync to counterfactual engine
                counterfactual = getattr(self._service, "_counterfactual", None)
                if counterfactual is not None and hasattr(counterfactual, "set_dynamics"):
                    counterfactual.set_dynamics(predictor.dynamics_matrix)
        except Exception:
            pass


# ─── Noise helpers for child inheritance (GAP 6) ──────────────────


def _apply_setpoint_noise(
    setpoints: dict,
    noise_fraction: float,
) -> dict:
    """Return a copy of setpoints with ±noise_fraction uniform noise per dim.

    Each value is perturbed by `value * U(-noise_fraction, +noise_fraction)`.
    Result is clamped to [0, 1] for all dimensions except VALENCE which is
    clamped to [-1, 1].
    """
    if not setpoints:
        return setpoints

    result: dict = {}
    for dim_key, value in setpoints.items():
        noise = random.uniform(-noise_fraction, noise_fraction)
        noisy = float(value) * (1.0 + noise)
        # VALENCE range is [-1, 1]; all others [0, 1]
        if str(dim_key) == "valence":
            noisy = max(-1.0, min(1.0, noisy))
        else:
            noisy = max(0.0, min(1.0, noisy))
        result[dim_key] = noisy
    return result


def _apply_dynamics_noise(
    matrix: list[list[float]],
    noise_fraction: float,
) -> list[list[float]]:
    """Return a copy of the dynamics matrix with ±noise_fraction noise per weight.

    Zero weights stay zero (no spurious coupling created from nothing).
    """
    if not matrix:
        return matrix

    result: list[list[float]] = []
    for row in matrix:
        noisy_row: list[float] = []
        for w in row:
            if w == 0.0:
                noisy_row.append(0.0)
            else:
                noise = random.uniform(-noise_fraction, noise_fraction)
                noisy_row.append(w * (1.0 + noise))
        result.append(noisy_row)
    return result
