"""
EcodiaOS - Voxis Genome Extraction & Seeding

Implements GenomeExtractionProtocol for the Voxis expression system.
Extracts the organism's accumulated voice identity - the personality vector,
vocabulary affinities, thematic references, and expression strategy
preferences - so a child instance inherits its parent's communicative
character rather than starting from a neutral seed.

Heritable state (what crosses the generational boundary):
    - PersonalityVector (9 dimensions: warmth, directness, verbosity,
      formality, curiosity_expression, humour, empathy_expression,
      confidence_display, metaphor_use)
    - Vocabulary affinities (word/phrase → preference weight)
    - Thematic references (preferred analogy domains)
    - Expression strategy preferences (accumulated strategy type frequencies)

NOT heritable:
    - Conversation state (too ephemeral)
    - Diversity tracker history (child starts fresh)
    - Silence decision cache (instance-specific)
    - Affect colouring state (driven by live Thymos input)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# 9 personality dimensions in canonical order - must match PersonalityVector fields
_PERSONALITY_KEYS: list[str] = [
    "warmth",
    "directness",
    "verbosity",
    "formality",
    "curiosity_expression",
    "humour",
    "empathy_expression",
    "confidence_display",
    "metaphor_use",
]

# Limit how many vocabulary affinities and thematic references we export
_MAX_VOCABULARY_AFFINITIES: int = 500
_MAX_THEMATIC_REFERENCES: int = 100
_MAX_STRATEGY_PREFERENCES: int = 50


class VoxisGenomeExtractor:
    """
    Extracts and seeds the Voxis system's heritable state via the
    organism-wide GenomeExtractionProtocol.

    The Voxis genome captures the organism's evolved voice: how warm,
    direct, verbose, formal it has become; which words it gravitates
    toward; which domains it draws analogies from; and which expression
    strategies it prefers. A child instance seeded from this genome
    starts speaking like its parent rather than a blank personality.
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._log = logger.bind(subsystem="voxis.genome")

    # ─── GenomeExtractionProtocol ────────────────────────────────────────────

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """
        Serialise the Voxis system's heritable state into an OrganGenomeSegment.

        Reads the personality_json field from the Self node in Neo4j (the
        canonical personality store), extracts vocabulary affinities and
        thematic references embedded within it, and queries expression
        strategy statistics from persisted episodes.

        Returns an empty segment (version=0) if no personality exists yet.
        """
        personality_vector = await self._extract_personality_vector()
        vocabulary_affinities = await self._extract_vocabulary_affinities()
        thematic_references = await self._extract_thematic_references()
        strategy_preferences = await self._extract_strategy_preferences()

        # If there is no personality at all, the instance has not been
        # born or initialised - return an empty segment so Mitosis
        # knows there is nothing to inherit.
        if not personality_vector:
            self._log.info("voxis_genome_empty", reason="no_personality_vector")
            return build_segment(
                system_id=SystemID.VOXIS,
                payload={
                    "personality_vector": {},
                    "vocabulary_affinities": {},
                    "thematic_references": [],
                    "strategy_preferences": [],
                },
                version=0,
            )

        payload: dict[str, Any] = {
            "personality_vector": personality_vector,
            "vocabulary_affinities": vocabulary_affinities,
            "thematic_references": thematic_references,
            "strategy_preferences": strategy_preferences,
        }

        segment = build_segment(
            system_id=SystemID.VOXIS,
            payload=payload,
            version=1,
        )

        self._log.info(
            "voxis_genome_extracted",
            personality_dims=len(personality_vector),
            vocabulary_count=len(vocabulary_affinities),
            thematic_count=len(thematic_references),
            strategy_count=len(strategy_preferences),
            size_bytes=segment.size_bytes,
        )

        return segment

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """
        Restore Voxis heritable state from a parent's genome segment.

        Verifies payload_hash integrity and schema_version compatibility
        before writing. Updates the Self node's personality_json with the
        inherited personality vector, vocabulary affinities, and thematic
        references.

        Returns True on success, False on any failure.
        """
        # ── Integrity checks ──────────────────────────────────────────
        if not check_schema_version(segment):
            self._log.error(
                "voxis_genome_seed_rejected",
                reason="incompatible_schema_version",
                schema_version=segment.schema_version,
            )
            return False

        if not verify_segment(segment):
            self._log.error(
                "voxis_genome_seed_rejected",
                reason="payload_hash_mismatch",
            )
            return False

        payload = segment.payload
        personality_vector: dict[str, float] = payload.get("personality_vector", {})
        vocabulary_affinities: dict[str, float] = payload.get("vocabulary_affinities", {})
        thematic_references: list[str] = payload.get("thematic_references", [])
        strategy_preferences: list[dict[str, Any]] = payload.get("strategy_preferences", [])

        if not personality_vector:
            self._log.warning(
                "voxis_genome_seed_skipped",
                reason="empty_personality_vector",
            )
            return False

        # ── Seed personality on Self node ─────────────────────────────
        personality_seeded = await self._seed_personality(
            personality_vector, vocabulary_affinities, thematic_references,
        )

        # ── Seed strategy preferences ────────────────────────────────
        strategies_seeded = await self._seed_strategy_preferences(strategy_preferences)

        success = personality_seeded

        self._log.info(
            "voxis_genome_seeded",
            personality_seeded=personality_seeded,
            strategies_seeded=strategies_seeded,
            success=success,
        )

        return success

    # ─── Extraction Helpers ──────────────────────────────────────────────────

    async def _extract_personality_vector(self) -> dict[str, float]:
        """
        Extract the personality vector from the Self node's personality_json field.

        Returns a dict of dimension name -> float value, or empty dict if
        no personality is stored.
        """
        try:
            rows = await self._neo4j.execute_read(
                "MATCH (s:Self) RETURN s.personality_json AS pj LIMIT 1"
            )
        except Exception as exc:
            self._log.warning("genome_extract_personality_failed", error=str(exc))
            return {}

        if not rows:
            return {}

        raw = rows[0].get("pj")
        if raw is None:
            return {}

        # personality_json may be stored as a JSON string or a Neo4j map
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                self._log.warning("genome_personality_json_parse_failed")
                return {}

        if not isinstance(raw, dict):
            return {}

        # Extract only the 9 canonical personality dimensions
        vector: dict[str, float] = {}
        for key in _PERSONALITY_KEYS:
            val = raw.get(key)
            if val is not None:
                try:
                    vector[key] = float(val)
                except (ValueError, TypeError):
                    pass

        return vector

    async def _extract_vocabulary_affinities(self) -> dict[str, float]:
        """
        Extract vocabulary affinities from the Self node's personality_json.

        Vocabulary affinities are nested inside the personality_json dict
        under the key 'vocabulary_affinities'.
        """
        try:
            rows = await self._neo4j.execute_read(
                "MATCH (s:Self) RETURN s.personality_json AS pj LIMIT 1"
            )
        except Exception as exc:
            self._log.warning("genome_extract_vocab_failed", error=str(exc))
            return {}

        if not rows:
            return {}

        raw = rows[0].get("pj")
        if raw is None:
            return {}

        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}

        if not isinstance(raw, dict):
            return {}

        vocab = raw.get("vocabulary_affinities")
        if not isinstance(vocab, dict):
            return {}

        # Limit export size and ensure values are floats
        affinities: dict[str, float] = {}
        # Sort by weight descending and take top N
        sorted_items = sorted(vocab.items(), key=lambda kv: kv[1], reverse=True)
        for word, weight in sorted_items[:_MAX_VOCABULARY_AFFINITIES]:
            try:
                affinities[str(word)] = float(weight)
            except (ValueError, TypeError):
                pass

        return affinities

    async def _extract_thematic_references(self) -> list[str]:
        """
        Extract thematic references from the Self node's personality_json.

        Thematic references are nested inside the personality_json dict
        under the key 'thematic_references'.
        """
        try:
            rows = await self._neo4j.execute_read(
                "MATCH (s:Self) RETURN s.personality_json AS pj LIMIT 1"
            )
        except Exception as exc:
            self._log.warning("genome_extract_thematic_failed", error=str(exc))
            return []

        if not rows:
            return []

        raw = rows[0].get("pj")
        if raw is None:
            return []

        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return []

        if not isinstance(raw, dict):
            return []

        refs = raw.get("thematic_references")
        if not isinstance(refs, list):
            return []

        return [str(r) for r in refs[:_MAX_THEMATIC_REFERENCES] if r]

    async def _extract_strategy_preferences(self) -> list[dict[str, Any]]:
        """
        Extract expression strategy usage statistics from persisted episodes.

        Aggregates strategy types (intent_type, speech_register, hedge_level)
        by frequency to capture the organism's evolved expression preferences.
        Returns a list of {strategy_type, value, count} dicts sorted by count.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (e:Episode)
                WHERE e.source = 'voxis'
                  AND e.strategy_intent_type IS NOT NULL
                WITH e.strategy_intent_type AS intent_type,
                     e.strategy_speech_register AS speech_register,
                     e.strategy_hedge_level AS hedge_level,
                     count(e) AS usage_count
                RETURN intent_type, speech_register, hedge_level, usage_count
                ORDER BY usage_count DESC
                LIMIT $limit
                """,
                {"limit": _MAX_STRATEGY_PREFERENCES},
            )
        except Exception as exc:
            self._log.warning("genome_extract_strategies_failed", error=str(exc))
            return []

        preferences: list[dict[str, Any]] = []
        for row in rows:
            pref: dict[str, Any] = {
                "intent_type": str(row.get("intent_type", "response")),
                "speech_register": str(row.get("speech_register", "neutral")),
                "hedge_level": str(row.get("hedge_level", "minimal")),
                "usage_count": int(row.get("usage_count", 0)),
            }
            preferences.append(pref)

        return preferences

    # ─── Seeding Helpers ─────────────────────────────────────────────────────

    async def _seed_personality(
        self,
        personality_vector: dict[str, float],
        vocabulary_affinities: dict[str, float],
        thematic_references: list[str],
    ) -> bool:
        """
        Write the inherited personality to the child's Self node.

        Combines the 9 personality dimensions, vocabulary affinities, and
        thematic references into a single personality_json dict and persists
        it on the Self node - matching the format VoxisService._load_state()
        reads at boot.
        """
        # Build the full personality_json dict
        personality_json: dict[str, Any] = {}
        for key in _PERSONALITY_KEYS:
            if key in personality_vector:
                personality_json[key] = personality_vector[key]

        if vocabulary_affinities:
            personality_json["vocabulary_affinities"] = vocabulary_affinities

        if thematic_references:
            personality_json["thematic_references"] = thematic_references

        serialised = json.dumps(personality_json, sort_keys=True, separators=(",", ":"))

        try:
            await self._neo4j.execute_write(
                "MATCH (s:Self) SET s.personality_json = $pj",
                {"pj": serialised},
            )
            self._log.info(
                "voxis_personality_seeded",
                dimensions=len(personality_vector),
                vocabulary_count=len(vocabulary_affinities),
                thematic_count=len(thematic_references),
            )
            return True
        except Exception as exc:
            self._log.error("voxis_seed_personality_failed", error=str(exc))
            return False

    async def _seed_strategy_preferences(
        self,
        preferences: list[dict[str, Any]],
    ) -> int:
        """
        Persist inherited strategy preferences as VoxisStrategyPref nodes.

        These give the child's expression policy selector historical context
        about which strategy combinations the parent favoured, enabling
        faster convergence to the parent's expression style.
        """
        seeded = 0
        for pref in preferences:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (:VoxisStrategyPref {
                        intent_type: $intent_type,
                        speech_register: $speech_register,
                        hedge_level: $hedge_level,
                        usage_count: $usage_count,
                        source: "parent_genome",
                        created_at: datetime()
                    })
                    """,
                    {
                        "intent_type": str(pref.get("intent_type", "response")),
                        "speech_register": str(pref.get("speech_register", "neutral")),
                        "hedge_level": str(pref.get("hedge_level", "minimal")),
                        "usage_count": int(pref.get("usage_count", 0)),
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug(
                    "seed_strategy_pref_failed",
                    intent_type=pref.get("intent_type"),
                    error=str(exc),
                )
        return seeded
