"""
EcodiaOS - Oikos Genome Extraction & Seeding

Heritable state: cost model parameters, mitosis strategy config,
bounty evaluation heuristics, yield strategy preferences.
NOT balances or positions.

Child starts with parent's economic wisdom, not parent's money
(seed capital is separate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment

if TYPE_CHECKING:
    from systems.oikos.service import OikosService

logger = structlog.get_logger()


class OikosGenomeExtractor:
    """Extracts Oikos economic wisdom for genome transmission."""

    def __init__(self, service: OikosService) -> None:
        self._service = service
        self._log = logger.bind(subsystem="oikos.genome")

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        try:
            cost_model_params = self._extract_cost_model_params()
            mitosis_config = self._extract_mitosis_config()
            bounty_heuristics = self._extract_bounty_heuristics()
            yield_preferences = self._extract_yield_preferences()

            if not cost_model_params and not mitosis_config:
                return build_segment(SystemID.OIKOS, {}, version=0)

            payload = {
                "cost_model_params": cost_model_params,
                "mitosis_config": mitosis_config,
                "bounty_heuristics": bounty_heuristics,
                "yield_preferences": yield_preferences,
            }

            self._log.info("oikos_genome_extracted")
            return build_segment(SystemID.OIKOS, payload, version=1)

        except Exception as exc:
            self._log.error("oikos_genome_extract_failed", error=str(exc))
            return build_segment(SystemID.OIKOS, {}, version=0)

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        if segment.version == 0 or not segment.payload:
            return True

        if not check_schema_version(segment):
            return False
        if not verify_segment(segment):
            return False

        try:
            payload = segment.payload
            self._apply_cost_model_params(payload.get("cost_model_params", {}))
            self._apply_mitosis_config(payload.get("mitosis_config", {}))
            self._apply_bounty_heuristics(payload.get("bounty_heuristics", {}))
            self._apply_yield_preferences(payload.get("yield_preferences", {}))

            self._log.info("oikos_genome_seeded")
            return True

        except Exception as exc:
            self._log.error("oikos_genome_seed_failed", error=str(exc))
            return False

    # ── Extraction helpers ─────────────────────────────────────────

    def _extract_cost_model_params(self) -> dict:
        """Extract basal metabolic rate model parameters."""
        try:
            config = self._service._config
            params: dict[str, object] = {}

            if hasattr(config, "basal_cost_usd_per_hour"):
                params["basal_cost_usd_per_hour"] = str(config.basal_cost_usd_per_hour)
            if hasattr(config, "llm_cost_multiplier"):
                params["llm_cost_multiplier"] = float(config.llm_cost_multiplier)
            if hasattr(config, "compute_cost_multiplier"):
                params["compute_cost_multiplier"] = float(
                    config.compute_cost_multiplier
                )
            if hasattr(config, "survival_reserve_target_days"):
                params["survival_reserve_target_days"] = int(
                    config.survival_reserve_target_days
                )
            if hasattr(config, "starvation_thresholds"):
                params["starvation_thresholds"] = config.starvation_thresholds

            return params
        except Exception:
            return {}

    def _extract_mitosis_config(self) -> dict:
        """Extract mitosis strategy parameters."""
        try:
            config = self._service._config
            mc: dict[str, object] = {}

            if hasattr(config, "min_parent_runway_days"):
                mc["min_parent_runway_days"] = int(config.min_parent_runway_days)
            if hasattr(config, "min_seed_capital_usd"):
                mc["min_seed_capital_usd"] = str(config.min_seed_capital_usd)
            if hasattr(config, "max_seed_pct_of_net_worth"):
                mc["max_seed_pct_of_net_worth"] = float(
                    config.max_seed_pct_of_net_worth
                )
            if hasattr(config, "min_parent_efficiency"):
                mc["min_parent_efficiency"] = float(config.min_parent_efficiency)
            if hasattr(config, "max_active_children"):
                mc["max_active_children"] = int(config.max_active_children)
            if hasattr(config, "dividend_rate"):
                mc["dividend_rate"] = str(config.dividend_rate)

            return mc
        except Exception:
            return {}

    def _extract_bounty_heuristics(self) -> dict:
        """Extract bounty evaluation heuristics."""
        try:
            config = self._service._config
            bh: dict[str, object] = {}

            if hasattr(config, "min_bounty_value_usd"):
                bh["min_bounty_value_usd"] = str(config.min_bounty_value_usd)
            if hasattr(config, "max_bounty_time_hours"):
                bh["max_bounty_time_hours"] = int(config.max_bounty_time_hours)
            if hasattr(config, "bounty_confidence_threshold"):
                bh["bounty_confidence_threshold"] = float(
                    config.bounty_confidence_threshold
                )

            return bh
        except Exception:
            return {}

    def _extract_yield_preferences(self) -> dict:
        """Extract yield strategy preferences."""
        try:
            config = self._service._config
            yp: dict[str, object] = {}

            if hasattr(config, "max_deployment_pct"):
                yp["max_deployment_pct"] = float(config.max_deployment_pct)
            if hasattr(config, "min_apy_threshold"):
                yp["min_apy_threshold"] = float(config.min_apy_threshold)
            if hasattr(config, "risk_tolerance"):
                yp["risk_tolerance"] = float(config.risk_tolerance)
            if hasattr(config, "rebalance_interval_hours"):
                yp["rebalance_interval_hours"] = int(
                    config.rebalance_interval_hours
                )

            return yp
        except Exception:
            return {}

    # ── Seeding helpers ────────────────────────────────────────────

    def _apply_cost_model_params(self, params: dict) -> None:
        if not params:
            return
        config = self._service._config
        for key, value in params.items():
            if hasattr(config, key):
                current = getattr(config, key)
                try:
                    setattr(config, key, type(current)(value))
                except (ValueError, TypeError):
                    pass

    def _apply_mitosis_config(self, mc: dict) -> None:
        if not mc:
            return
        config = self._service._config
        for key, value in mc.items():
            if hasattr(config, key):
                current = getattr(config, key)
                try:
                    setattr(config, key, type(current)(value))
                except (ValueError, TypeError):
                    pass

    def _apply_bounty_heuristics(self, bh: dict) -> None:
        if not bh:
            return
        config = self._service._config
        for key, value in bh.items():
            if hasattr(config, key):
                current = getattr(config, key)
                try:
                    setattr(config, key, type(current)(value))
                except (ValueError, TypeError):
                    pass

    def _apply_yield_preferences(self, yp: dict) -> None:
        if not yp:
            return
        config = self._service._config
        for key, value in yp.items():
            if hasattr(config, key):
                current = getattr(config, key)
                try:
                    setattr(config, key, type(current)(value))
                except (ValueError, TypeError):
                    pass
