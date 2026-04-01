"""
EcodiaOS - Logos: World Model

The compressed generative core of EOS cognition.

This is not a database. This is not a list of facts.
This is a generative model - a compact set of rules that can PRODUCE
the observations EOS has encountered, and PREDICT observations it hasn't.

The World Model is the culmination of the entire Compression Cascade.
Everything else in memory is either:
  (a) raw material being processed toward the World Model, or
  (b) residue that hasn't been compressed into it yet

The intelligence of EOS is primarily the intelligence of this model.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from primitives.common import new_id, utc_now
from systems.logos.types import (
    CausalLink,
    EmpiricalInvariant,
    ExperienceDelta,
    GenerativeSchema,
    Prediction,
    PriorDistribution,
    WorldModelUpdate,
    WorldModelUpdateType,
)

logger = structlog.get_logger("logos.world_model")


class CausalGraph:
    """
    Compressed causal structure of EOS's experienced reality.
    Directed graph of causal links between entity/concept IDs.
    """

    def __init__(self) -> None:
        self._links: dict[str, CausalLink] = {}  # keyed by "{cause_id}->{effect_id}"

    @property
    def link_count(self) -> int:
        return len(self._links)

    @property
    def links(self) -> dict[str, CausalLink]:
        """Read-only access to the link dictionary (keyed by 'cause->effect')."""
        return self._links

    def add_link(self, link: CausalLink) -> None:
        key = f"{link.cause_id}->{link.effect_id}"
        existing = self._links.get(key)
        if existing is not None:
            # Bayesian update: strengthen observed connection
            existing.strength = min(1.0, existing.strength + 0.05)
            existing.observations += 1
            existing.last_observed = utc_now()
        else:
            self._links[key] = link

    def revise_link(self, cause_id: str, effect_id: str, new_strength: float) -> bool:
        key = f"{cause_id}->{effect_id}"
        link = self._links.get(key)
        if link is None:
            return False
        link.strength = max(0.0, min(1.0, new_strength))
        return True

    def remove_weak_links(self, threshold: float = 0.05) -> int:
        """Remove links below strength threshold. Returns count removed."""
        weak = [k for k, v in self._links.items() if v.strength < threshold]
        for k in weak:
            del self._links[k]
        return len(weak)

    def project(self, context: dict[str, Any]) -> list[CausalLink]:
        """Retrieve causal expectations relevant to a context."""
        relevant: list[CausalLink] = []
        context_ids: set[str] = set()
        for v in context.values():
            if isinstance(v, str):
                context_ids.add(v)
            elif isinstance(v, list):
                context_ids.update(str(x) for x in v)

        for link in self._links.values():
            if link.cause_id in context_ids or link.domain in context.get("domains", []):
                relevant.append(link)
        return relevant

    def estimate_complexity(self) -> float:
        """Estimate description length of the causal graph in bits."""
        # Each link: ~50 bits (two IDs + strength + domain)
        return len(self._links) * 50.0


class WorldModel:
    """
    The compressed generative core of EOS cognition.

    Maintains generative schemas, causal structure, predictive priors,
    and empirical invariants. Measures the intelligence ratio I = K(reality) / K(model).
    """

    def __init__(self) -> None:
        self.generative_schemas: dict[str, GenerativeSchema] = {}
        self.causal_structure: CausalGraph = CausalGraph()
        self.predictive_priors: dict[str, PriorDistribution] = {}
        self.empirical_invariants: list[EmpiricalInvariant] = []

        # Model complexity (description length in bits, estimated)
        self.current_complexity: float = 100.0  # Baseline

        # Explanatory coverage (fraction of episodic memory predictable from model)
        self.coverage: float = 0.0

        # Tracking for intelligence ratio.
        # _total_explained_bits accumulates information_content of integrated deltas.
        # Perfectly-predicted episodes (predicted_count) cost 0 bits to explain.
        self._total_explained_bits: float = 0.0
        self._total_episodes_explained: int = 0
        self._total_episodes_predicted: int = 0  # Discarded as redundant
        self._total_episodes_received: int = 0

        # Prediction accuracy tracking (rolling window)
        self._prediction_attempts: int = 0
        self._prediction_correct: int = 0

        # Schema growth: per-hour rate using monotonic time
        self._last_hour_wall: float = time.monotonic()
        self._schemas_at_last_hour: int = 0
        self._schema_growth_rate: float = 0.0

    async def predict(self, context: dict[str, Any]) -> Prediction:
        """Generate a prediction for what will happen in this context."""
        relevant_schemas = self._retrieve_relevant_schemas(context)
        relevant_priors = self._retrieve_relevant_priors(context)
        causal_expectations = self.causal_structure.project(context)

        expected_content = self._synthesize_prediction(
            relevant_schemas, relevant_priors, causal_expectations
        )
        confidence = self._compute_prediction_confidence(
            context, relevant_schemas, relevant_priors
        )

        return Prediction(
            expected_content=expected_content,
            confidence=confidence,
            generating_schemas=[s.id for s in relevant_schemas],
        )

    async def integrate(self, delta: ExperienceDelta) -> WorldModelUpdate:
        """
        Integrate a new experience delta into the World Model.
        This is how the model grows.

        A successful integration either:
        (a) Updates a prior distribution (Bayesian update)
        (b) Extends a generative schema (schema covers a new case)
        (c) Creates a new schema (a new pattern discovered)
        (d) Revises a causal connection (causality was wrong)
        (e) Violates an empirical invariant (requires constitution-level review)

        Redundant deltas (discard_after_encoding=True) still contribute to coverage
        tracking - the world model perfectly predicted them, so they count as explained.
        """
        self._total_episodes_received += 1

        if delta.discard_after_encoding or delta.delta_content is None:
            # The model perfectly predicted this experience - already explained.
            self._total_episodes_predicted += 1
            self._update_coverage()
            return WorldModelUpdate(
                update_type=WorldModelUpdateType.PRIOR_UPDATED,
                coverage_delta=0.0,
            )

        content = delta.delta_content
        schemas_added = 0
        schemas_extended = 0
        priors_updated = 0
        causal_added = 0
        causal_revised = 0
        invariants_violated = 0

        # Check for invariant violations first (highest priority update type)
        for inv in self.empirical_invariants:
            if inv.statement in content.violated_priors:
                invariants_violated += 1
                inv.confidence *= 0.8  # Weaken violated invariant
                inv.last_tested = utc_now()
                logger.warning(
                    "invariant_violation_detected",
                    invariant_id=inv.id,
                    statement=inv.statement[:80],
                    new_confidence=inv.confidence,
                )

        if invariants_violated > 0:
            update_type = WorldModelUpdateType.INVARIANT_VIOLATED
        elif content.novel_entities:
            # Novel entities suggest a new schema or extension
            matched_schema = self._find_matching_schema(content)
            if matched_schema is not None:
                matched_schema.instance_count += 1
                matched_schema.last_instantiated = utc_now()
                schemas_extended = 1
                update_type = WorldModelUpdateType.SCHEMA_EXTENDED
            else:
                # Create a new schema from the novel content
                new_schema = GenerativeSchema(
                    id=new_id(),
                    name=f"schema_{len(self.generative_schemas)}",
                    domain=content.content.get("domain", "general"),
                    description="; ".join(content.novel_entities[:3]),
                    pattern=dict(content.content),
                    instance_count=1,
                )
                self.generative_schemas[new_schema.id] = new_schema
                schemas_added = 1
                self._update_schema_growth_rate()
                update_type = WorldModelUpdateType.SCHEMA_CREATED
        elif content.novel_relations:
            # Novel relations → extend the causal graph
            for rel in content.novel_relations:
                if "->" in rel:
                    parts = rel.split("->", 1)
                    cause = parts[0].strip()
                    effect = parts[1].strip()
                else:
                    cause = rel
                    effect = content.content.get("domain", "unknown")
                self.causal_structure.add_link(CausalLink(
                    cause_id=cause,
                    effect_id=effect,
                    domain=content.content.get("domain", "general"),
                ))
                causal_added += 1
            update_type = WorldModelUpdateType.CAUSAL_REVISED
        else:
            # Standard Bayesian prior update
            context_key = self._stable_context_key(content.content)
            prior = self.predictive_priors.get(context_key)
            if prior is not None:
                prior.sample_count += 1
                prior.last_updated = utc_now()
            else:
                self.predictive_priors[context_key] = PriorDistribution(
                    context_key=context_key,
                    sample_count=1,
                )
            priors_updated = 1
            update_type = WorldModelUpdateType.PRIOR_UPDATED

        # Update model complexity estimate
        old_complexity = self.current_complexity
        self._recompute_complexity()
        complexity_delta = self.current_complexity - old_complexity

        # Track what fraction of reality is now explained.
        # information_content is the fraction of the delta that was novel (0-1),
        # so we accumulate it as a proxy for bits of reality absorbed.
        self._total_explained_bits += delta.information_content
        self._total_episodes_explained += 1
        old_coverage = self.coverage
        self._update_coverage()
        coverage_delta = self.coverage - old_coverage

        update = WorldModelUpdate(
            update_type=update_type,
            schemas_added=schemas_added,
            schemas_extended=schemas_extended,
            priors_updated=priors_updated,
            causal_links_added=causal_added,
            causal_links_revised=causal_revised,
            invariants_tested=len(self.empirical_invariants),
            invariants_violated=invariants_violated,
            complexity_delta=complexity_delta,
            coverage_delta=coverage_delta,
        )

        logger.info(
            "world_model_integrated",
            update_type=update_type.value,
            schemas_added=schemas_added,
            priors_updated=priors_updated,
            causal_added=causal_added,
            complexity=self.current_complexity,
            coverage=self.coverage,
        )

        return update

    def measure_intelligence_ratio(self) -> float:
        """
        I = K(reality_modeled) / K(model)

        K(reality_modeled) = total information content (bits) of all deltas absorbed
                             plus 1 bit each for perfectly-predicted episodes.
        K(model) = current_complexity (description length of the world model in bits)

        A higher ratio means the model explains more per bit of its own description.
        This is the primary AGI progress metric.
        """
        predicted_contribution = float(self._total_episodes_predicted)
        total_explained = self._total_explained_bits + predicted_contribution
        return total_explained / max(self.current_complexity, 1.0)

    def record_prediction_outcome(self, correct: bool) -> None:
        """Track prediction accuracy for the intelligence dashboard."""
        self._prediction_attempts += 1
        if correct:
            self._prediction_correct += 1

    @property
    def prediction_accuracy(self) -> float:
        if self._prediction_attempts == 0:
            return 0.0
        return self._prediction_correct / self._prediction_attempts

    @property
    def schema_growth_rate(self) -> float:
        """Schemas created per hour, computed over the trailing hour window."""
        self._maybe_tick_schema_hour()
        return self._schema_growth_rate

    def register_schema(self, schema: GenerativeSchema) -> bool:
        """Register a schema in the world model, checking for domain duplicates.

        Returns True if the schema was registered, False if a duplicate
        covering the same domain already exists.
        """
        # Check for duplicate: same domain + high description overlap
        for existing in self.generative_schemas.values():
            if existing.domain == schema.domain and existing.description == schema.description:
                # Strengthen existing instead of adding duplicate
                existing.instance_count += schema.instance_count
                existing.last_instantiated = utc_now()
                logger.debug(
                    "schema_merged_with_existing",
                    existing_id=existing.id,
                    new_id=schema.id,
                )
                return False

        self.generative_schemas[schema.id] = schema
        self._update_schema_growth_rate()
        self._recompute_complexity()
        logger.info(
            "schema_registered",
            schema_id=schema.id,
            domain=schema.domain,
            description=schema.description[:80],
        )
        return True

    def ingest_invariant(self, invariant: EmpiricalInvariant) -> None:
        """Ingest a causal invariant (e.g., from Kairos)."""
        self.empirical_invariants.append(invariant)
        self._recompute_complexity()
        logger.info(
            "invariant_ingested",
            invariant_id=invariant.id,
            statement=invariant.statement[:80],
            source=invariant.source,
        )

    def count_observations_explained_by(self, invariant_id: str) -> int:
        """
        Return the number of episodes / observations the invariant explains.

        Uses the invariant's own observation_count field, which is incremented
        each time the invariant is validated against a new episode.
        Returns 0 if the invariant is unknown to the world model.
        """
        for inv in self.empirical_invariants:
            if inv.id == invariant_id:
                return inv.observation_count
        return 0

    def estimate_description_length_without(self, invariant_id: str) -> float:
        """
        Estimate world model total description length (bits) without this invariant.

        Removes the invariant's compressed representation (~80 bits) but adds back
        the raw observations it was compressing (observation_count bits each,
        estimated at 50 bits/observation without the rule).
        Returns current complexity unchanged when the invariant is unknown.
        """
        for inv in self.empirical_invariants:
            if inv.id == invariant_id:
                # Cost of the invariant rule itself in the model
                invariant_rule_bits = 80.0
                # Cost of raw observations the invariant was encoding
                raw_observation_bits = inv.observation_count * 50.0
                # Description length without the invariant = current - rule + raw obs
                return max(
                    self.current_complexity - invariant_rule_bits + raw_observation_bits,
                    100.0,
                )
        return self.current_complexity

    def snapshot(self) -> dict:
        """M5: Serialize entire world model state for Mitosis genome cloning."""
        return {
            "generative_schemas": {
                sid: {
                    "id": s.id,
                    "name": s.name,
                    "domain": s.domain,
                    "description": s.description,
                    "pattern": s.pattern,
                    "instance_count": s.instance_count,
                    "compression_ratio": s.compression_ratio,
                    "created_at": s.created_at.isoformat(),
                    "last_instantiated": s.last_instantiated.isoformat(),
                }
                for sid, s in self.generative_schemas.items()
            },
            "causal_links": {
                key: {
                    "cause_id": link.cause_id,
                    "effect_id": link.effect_id,
                    "strength": link.strength,
                    "domain": link.domain,
                    "observations": link.observations,
                    "last_observed": link.last_observed.isoformat(),
                }
                for key, link in self.causal_structure.links.items()
            },
            "predictive_priors": {
                key: {
                    "context_key": prior.context_key,
                    "variance": prior.variance,
                    "sample_count": prior.sample_count,
                    "last_updated": prior.last_updated.isoformat(),
                }
                for key, prior in self.predictive_priors.items()
            },
            "empirical_invariants": [
                {
                    "id": inv.id,
                    "statement": inv.statement,
                    "domain": inv.domain,
                    "observation_count": inv.observation_count,
                    "confidence": inv.confidence,
                    "source": inv.source,
                }
                for inv in self.empirical_invariants
            ],
            "metrics": {
                "current_complexity": self.current_complexity,
                "coverage": self.coverage,
                "intelligence_ratio": self.measure_intelligence_ratio(),
                "total_explained_bits": self._total_explained_bits,
                "total_episodes_explained": self._total_episodes_explained,
                "total_episodes_predicted": self._total_episodes_predicted,
                "total_episodes_received": self._total_episodes_received,
            },
        }

    def restore_from_snapshot(self, data: dict) -> None:
        """M5: Load world model state from a serialized snapshot."""
        # Restore schemas
        for sid, sdata in data.get("generative_schemas", {}).items():
            schema = GenerativeSchema(
                id=sdata["id"],
                name=sdata.get("name", ""),
                domain=sdata.get("domain", ""),
                description=sdata.get("description", ""),
                pattern=sdata.get("pattern", {}),
                instance_count=sdata.get("instance_count", 0),
                compression_ratio=sdata.get("compression_ratio", 0.0),
            )
            self.generative_schemas[sid] = schema

        # Restore causal links
        for _key, ldata in data.get("causal_links", {}).items():
            link = CausalLink(
                cause_id=ldata["cause_id"],
                effect_id=ldata["effect_id"],
                strength=ldata.get("strength", 0.5),
                domain=ldata.get("domain", ""),
                observations=ldata.get("observations", 0),
            )
            self.causal_structure.add_link(link)

        # Restore priors
        for key, pdata in data.get("predictive_priors", {}).items():
            prior = PriorDistribution(
                context_key=pdata["context_key"],
                variance=pdata.get("variance", 1.0),
                sample_count=pdata.get("sample_count", 0),
            )
            self.predictive_priors[key] = prior

        # Restore invariants
        for idata in data.get("empirical_invariants", []):
            invariant = EmpiricalInvariant(
                id=idata["id"],
                statement=idata.get("statement", ""),
                domain=idata.get("domain", ""),
                observation_count=idata.get("observation_count", 0),
                confidence=idata.get("confidence", 1.0),
                source=idata.get("source", ""),
            )
            self.empirical_invariants.append(invariant)

        # Restore metrics
        metrics = data.get("metrics", {})
        if metrics:
            self.current_complexity = metrics.get("current_complexity", 100.0)
            self.coverage = metrics.get("coverage", 0.0)
            self._total_explained_bits = metrics.get("total_explained_bits", 0.0)
            self._total_episodes_explained = metrics.get("total_episodes_explained", 0)
            self._total_episodes_predicted = metrics.get("total_episodes_predicted", 0)
            self._total_episodes_received = metrics.get("total_episodes_received", 0)

        self._recompute_complexity()
        logger.info(
            "world_model_restored_from_snapshot",
            schemas=len(self.generative_schemas),
            causal_links=self.causal_structure.link_count,
            priors=len(self.predictive_priors),
            invariants=len(self.empirical_invariants),
        )

    def get_context_stability_age(self, context_key: str) -> float:
        """
        How long (seconds) a context's prior has been stable.
        Used by Fovea for prediction confidence.
        """
        prior = self.predictive_priors.get(context_key)
        if prior is None:
            return 0.0
        age = (utc_now() - prior.last_updated).total_seconds()
        return age

    def get_historical_accuracy(self, domain: str | None = None) -> float:
        """
        Historical prediction accuracy, optionally filtered by domain.
        Used by Fovea for calibrating predictions.

        When domain is specified, returns accuracy only for schemas in that domain.
        Falls back to overall accuracy when no domain-specific data is available.
        """
        if domain is None:
            return self.prediction_accuracy

        # Domain-filtered accuracy: fraction of schemas in the domain that
        # have been confirmed (instance_count > 1 means the schema predicted
        # successfully at least once).
        domain_schemas = [
            s for s in self.generative_schemas.values()
            if s.domain == domain
        ]
        if not domain_schemas:
            return self.prediction_accuracy  # fallback to overall

        confirmed = sum(1 for s in domain_schemas if s.instance_count > 1)
        return confirmed / len(domain_schemas)

    # ─── Internal helpers ────────────────────────────────────────

    def _update_coverage(self) -> None:
        """
        Recompute coverage as the fraction of received episodes the model explains.
        An episode is explained if it was perfectly predicted (discarded) or integrated.
        """
        total = self._total_episodes_received
        if total == 0:
            self.coverage = 0.0
            return
        explained = self._total_episodes_predicted + self._total_episodes_explained
        self.coverage = min(explained / total, 1.0)

    def _update_schema_growth_rate(self) -> None:
        """Call whenever a new schema is created. Updates the hourly growth rate."""
        self._maybe_tick_schema_hour()
        current_count = len(self.generative_schemas)
        schemas_this_hour = current_count - self._schemas_at_last_hour
        self._schema_growth_rate = float(max(schemas_this_hour, 0))

    def _maybe_tick_schema_hour(self) -> None:
        """Roll the hourly schema growth window if an hour has passed."""
        now = time.monotonic()
        elapsed = now - self._last_hour_wall
        if elapsed >= 3600.0:
            current_count = len(self.generative_schemas)
            schemas_this_hour = current_count - self._schemas_at_last_hour
            hours_elapsed = elapsed / 3600.0
            self._schema_growth_rate = schemas_this_hour / hours_elapsed
            self._schemas_at_last_hour = current_count
            self._last_hour_wall = now

    def _stable_context_key(self, content: dict[str, Any]) -> str:
        """
        Produce a stable string key for a content dict, safe even when values
        are not directly sortable (e.g., nested dicts or lists).
        """
        try:
            sorted_pairs = sorted(
                (str(k), str(v)) for k, v in content.items()
            )
            return str(hash(tuple(sorted_pairs)))[:16]
        except Exception:
            return str(hash(tuple(sorted(str(k) for k in content))))[:16]

    def _retrieve_relevant_schemas(
        self, context: dict[str, Any]
    ) -> list[GenerativeSchema]:
        """Find schemas relevant to the given context."""
        domain = context.get("domain", "")
        relevant: list[GenerativeSchema] = []
        for schema in self.generative_schemas.values():
            if schema.domain == domain or not domain or any(
                str(v) in schema.description
                for v in context.values()
                if isinstance(v, str)
            ):
                relevant.append(schema)
        return relevant[:10]  # Limit to top 10

    def _retrieve_relevant_priors(
        self, context: dict[str, Any]
    ) -> list[PriorDistribution]:
        """Find priors relevant to the given context."""
        context_key = self._stable_context_key(context)
        prior = self.predictive_priors.get(context_key)
        if prior is not None:
            return [prior]
        return []

    def _synthesize_prediction(
        self,
        schemas: list[GenerativeSchema],
        priors: list[PriorDistribution],
        causal_expectations: list[CausalLink],
    ) -> dict[str, Any]:
        """Synthesize a prediction from schemas, priors, and causal expectations."""
        prediction: dict[str, Any] = {}

        # Incorporate schema patterns
        for schema in schemas:
            prediction.update(schema.pattern)

        # Incorporate causal expectations
        if causal_expectations:
            prediction["expected_effects"] = [
                {"effect": link.effect_id, "strength": link.strength}
                for link in causal_expectations[:5]
            ]

        # Incorporate prior distributions
        if priors:
            prediction["prior_sample_count"] = sum(p.sample_count for p in priors)

        return prediction

    def _compute_prediction_confidence(
        self,
        context: dict[str, Any],
        schemas: list[GenerativeSchema],
        priors: list[PriorDistribution],
    ) -> float:
        """Compute confidence in a prediction (0-1)."""
        if not schemas and not priors:
            return 0.0

        # Confidence from schema coverage
        schema_confidence = min(len(schemas) / 3.0, 1.0) * 0.5

        # Confidence from prior sample counts
        total_samples = sum(p.sample_count for p in priors) if priors else 0
        prior_confidence = min(total_samples / 10.0, 1.0) * 0.5

        return schema_confidence + prior_confidence

    def _find_matching_schema(self, delta_content: Any) -> GenerativeSchema | None:
        """Find an existing schema that can absorb the novel content."""
        if not delta_content.novel_entities:
            return None

        novel_set = set(delta_content.novel_entities)
        best_match: GenerativeSchema | None = None
        best_overlap = 0.0

        for schema in self.generative_schemas.values():
            desc_words = set(schema.description.split())
            overlap = len(novel_set & desc_words) / max(len(novel_set), 1)
            if overlap > best_overlap and overlap > 0.3:
                best_overlap = overlap
                best_match = schema

        return best_match

    def _recompute_complexity(self) -> None:
        """Recompute the total description length of the world model in bits."""
        # Schema complexity: each schema ~100 bits base + 10 bits per instance ref
        schema_bits = sum(
            100.0 + (s.instance_count * 10.0)
            for s in self.generative_schemas.values()
        )

        # Causal graph complexity
        causal_bits = self.causal_structure.estimate_complexity()

        # Prior complexity: ~30 bits per prior
        prior_bits = len(self.predictive_priors) * 30.0

        # Invariant complexity: ~80 bits per invariant
        invariant_bits = len(self.empirical_invariants) * 80.0

        self.current_complexity = max(
            schema_bits + causal_bits + prior_bits + invariant_bits,
            100.0,  # Minimum baseline complexity
        )
