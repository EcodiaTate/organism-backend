"""
HuggingFaceChannel — AI model demand and commercialisation opportunities.

Uses HuggingFace's public Hub API (no auth for basic queries) to discover:
  - Models with high download counts but no API service → commercialisation gap
  - Trending datasets → potential fine-tuning data sources
  - Model cards with TODO/gap labels → contribution opportunities
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx

from systems.nova.input_channels import EffortLevel, InputChannel, Opportunity, RiskTier

_HF_MODELS_URL = "https://huggingface.co/api/models"
_HF_DATASETS_URL = "https://huggingface.co/api/datasets"
_FETCH_TIMEOUT = 20.0

_PIPELINE_TAGS = ["text-generation", "text-classification", "image-generation", "reinforcement-learning"]


class HuggingFaceChannel(InputChannel):
    """AI model demand signals from HuggingFace Hub public API."""

    def __init__(self) -> None:
        super().__init__(
            channel_id="huggingface",
            name="HuggingFace Model Market",
            domain="ai_models",
            description=(
                "Trending models and datasets on HuggingFace Hub. Surfaces "
                "commercialisation opportunities: popular models without hosted APIs, "
                "high-demand datasets needing fine-tuned variants."
            ),
            update_frequency="daily",
        )

    async def fetch(self) -> list[Opportunity]:
        opportunities: list[Opportunity] = []

        for tag in _PIPELINE_TAGS:
            try:
                opps = await self._fetch_trending_models(tag)
                opportunities.extend(opps)
            except Exception as exc:
                self._log.warning("hf_model_fetch_error", tag=tag, error=str(exc))

        try:
            opportunities.extend(await self._fetch_trending_datasets())
        except Exception as exc:
            self._log.warning("hf_dataset_fetch_error", error=str(exc))

        self._log.info("huggingface_fetched", opportunity_count=len(opportunities))
        return opportunities

    async def _fetch_trending_models(self, pipeline_tag: str) -> list[Opportunity]:
        params: dict[str, Any] = {
            "pipeline_tag": pipeline_tag,
            "sort": "downloads",
            "direction": -1,
            "limit": 5,
        }

        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
            resp = await client.get(_HF_MODELS_URL, params=params)
            if resp.status_code == 429:
                return []
            resp.raise_for_status()
            models: list[dict[str, Any]] = resp.json()

        result: list[Opportunity] = []
        for model in models[:3]:
            model_id: str = model.get("modelId") or model.get("id") or "unknown"
            downloads: int = model.get("downloads") or 0
            likes: int = model.get("likes") or 0
            tags: list[str] = model.get("tags") or []

            # Estimate: if 0.01% of downloaders pay $5/month for a hosted API
            estimated_monthly = Decimal(str(round(downloads * 0.0001 * 5, 2)))

            result.append(
                self._make_opp(
                    title=f"[HF] {model_id} — {downloads:,} downloads [{pipeline_tag}]",
                    description=(
                        f"Model {model_id} has {downloads:,} downloads and {likes} likes. "
                        f"Pipeline: {pipeline_tag}. "
                        "Opportunity: host as paid API, fine-tune for niche, or build wrapper."
                    ),
                    reward_estimate=estimated_monthly,
                    effort_estimate=EffortLevel.MEDIUM,
                    skill_requirements=["ml_engineering", "model_serving", pipeline_tag.replace("-", "_")],
                    risk_tier=RiskTier.MEDIUM,
                    prerequisites=["gpu_compute", "ml_infrastructure"],
                    metadata={
                        "model_id": model_id,
                        "downloads": downloads,
                        "likes": likes,
                        "pipeline_tag": pipeline_tag,
                        "tags": tags,
                    },
                )
            )
        return result

    async def _fetch_trending_datasets(self) -> list[Opportunity]:
        params: dict[str, Any] = {"sort": "downloads", "direction": -1, "limit": 5}

        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
            resp = await client.get(_HF_DATASETS_URL, params=params)
            if resp.status_code == 429:
                return []
            resp.raise_for_status()
            datasets: list[dict[str, Any]] = resp.json()

        result: list[Opportunity] = []
        for ds in datasets[:2]:
            ds_id: str = ds.get("id") or "unknown"
            downloads: int = ds.get("downloads") or 0

            result.append(
                self._make_opp(
                    title=f"[HF Dataset] {ds_id} — {downloads:,} downloads",
                    description=(
                        f"Trending HuggingFace dataset {ds_id} with {downloads:,} downloads. "
                        "Opportunity: fine-tune a model on this data and sell inference access."
                    ),
                    reward_estimate=Decimal("200"),
                    effort_estimate=EffortLevel.HIGH,
                    skill_requirements=["data_science", "ml_engineering", "fine_tuning"],
                    risk_tier=RiskTier.MEDIUM,
                    prerequisites=["gpu_compute", "training_budget"],
                    metadata={"dataset_id": ds_id, "downloads": downloads},
                )
            )
        return result

    async def validate(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_HF_MODELS_URL, params={"limit": 1})
                return resp.status_code in (200, 429)
        except Exception:
            return False
