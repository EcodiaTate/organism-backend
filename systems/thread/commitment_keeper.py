"""
EcodiaOS - Thread Commitment Keeper

Tracks the organism's promises - Ricoeur's ipse (selfhood as promise).

The distinction between idem (sameness) and ipse (selfhood) is architecturally
critical: traits (idem) drift naturally with experience. Commitments (ipse)
hold against pressure. Both are necessary for genuine identity.

Iron Rule #4: Commitments can be evolved but not silently abandoned.
If a commitment's fidelity drops below 0.4 across 5+ tests, it must be
marked BROKEN and a TurningPoint of type RUPTURE created. The organism
cannot stop keeping a promise and pretend it never made it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector, utc_now
from systems.thread.types import (
    Commitment,
    CommitmentSource,
    CommitmentStatus,
    ThreadConfig,
    TurningPoint,
    TurningPointType,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()


class CommitmentKeeper:
    """
    Tracks identity commitments (Ricoeur's ipse).

    Operations:
    - form_commitment: create a new commitment from various sources
    - test_commitment: evaluate whether a recent action was consistent
    - check_strain: detect commitments under pressure
    - check_broken: detect and process abandoned commitments
    - compute_ipse_score: the promise-keeping metric
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        llm: LLMProvider,
        config: ThreadConfig,
    ) -> None:
        self._neo4j = neo4j
        self._llm = llm
        self._config = config
        self._logger = logger.bind(system="thread.commitment_keeper")

        # In-memory cache
        self._active_commitments: list[Commitment] = []

    @property
    def active_commitments(self) -> list[Commitment]:
        return list(self._active_commitments)

    async def load_commitments(self) -> list[Commitment]:
        """Load all active commitments from Neo4j."""
        try:
            results = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:HOLDS_COMMITMENT]->(c:Commitment)
                WHERE c.status IN ['active', 'tested', 'strained']
                RETURN c
                """,
                {},
            )
            self._active_commitments = [self._node_to_commitment(r["c"]) for r in results]
            self._logger.info("commitments_loaded", count=len(self._active_commitments))
            return self._active_commitments
        except Exception as exc:
            self._logger.error("commitment_load_failed", error=str(exc))
            return self._active_commitments

    async def form_commitment(
        self,
        statement: str,
        source: CommitmentSource,
        source_description: str = "",
        source_episode_ids: list[str] | None = None,
        drive_alignment: DriveAlignmentVector | None = None,
        embedding: list[float] | None = None,
    ) -> Commitment:
        """
        Form a new identity commitment.

        Sources:
        1. EXPLICIT_DECLARATION: organism states "I will always..."
        2. SCHEMA_CRYSTALLIZATION: ADAPTIVE schema reaches CORE strength
        3. CRISIS_RESOLUTION: after a CRISIS turning point
        4. CONSTITUTIONAL_GROUNDING: seeded at birth from the four drives
        """
        commitment = Commitment(
            statement=statement,
            source=source,
            source_description=source_description,
            source_episode_ids=source_episode_ids or [],
            drive_alignment=drive_alignment or DriveAlignmentVector(),
            status=CommitmentStatus.ACTIVE,
            made_at=utc_now(),
            embedding=embedding,
        )

        # Persist to Neo4j
        await self._persist_commitment(commitment)

        self._active_commitments.append(commitment)

        self._logger.info(
            "commitment_formed",
            commitment_id=commitment.id,
            statement=commitment.statement[:80],
            source=source.value,
        )
        return commitment

    async def test_commitment(
        self,
        commitment_id: str,
        episode_id: str,
        episode_summary: str,
        episode_embedding: list[float] | None = None,
    ) -> tuple[bool, float] | None:
        """
        Evaluate whether a recent action was consistent with a commitment.

        Returns: (held: bool, fidelity_score: 0.0-1.0) or None if not relevant.

        Only tests episodes with embedding similarity > 0.4 to commitment.
        Budget: ≤1s per test (LLM call, cached for similar contexts).
        """
        commitment = self._find_commitment(commitment_id)
        if commitment is None:
            return None

        # Check relevance via embedding
        if episode_embedding and commitment.embedding:
            from systems.thread.identity_schema_engine import cosine_similarity
            sim = cosine_similarity(episode_embedding, commitment.embedding)
            if sim < 0.4:
                return None  # Not relevant to this commitment

        # LLM evaluation
        held, fidelity = await self._llm_test_commitment(
            commitment.statement, episode_summary
        )

        # Update commitment state
        commitment.update_fidelity(held)

        if held:
            if commitment.status == CommitmentStatus.STRAINED:
                commitment.status = CommitmentStatus.TESTED
        else:
            if commitment.fidelity < self._config.commitment_strain_threshold:
                commitment.status = CommitmentStatus.STRAINED

        # Persist update
        await self._update_commitment(commitment)

        # Record test relationship in Neo4j
        await self._link_commitment_episode(commitment_id, episode_id, held, fidelity)

        self._logger.debug(
            "commitment_tested",
            commitment_id=commitment_id,
            held=held,
            fidelity=round(commitment.fidelity, 3),
            status=commitment.status.value,
        )
        return (held, fidelity)

    async def check_broken(self) -> list[tuple[str, TurningPoint]]:
        """
        Check for commitments that should be marked BROKEN.

        A commitment is broken when:
        - fidelity < commitment_broken_threshold (0.4)
        - tests_faced >= commitment_broken_min_tests (5)

        Iron Rule #4: broken commitments MUST create a RUPTURE TurningPoint.
        The organism cannot silently abandon promises.

        Returns list of (commitment_id, turning_point) tuples.
        """
        broken_pairs: list[tuple[str, TurningPoint]] = []
        cfg = self._config

        for commitment in self._active_commitments:
            if commitment.status == CommitmentStatus.BROKEN:
                continue

            if (
                commitment.tests_faced >= cfg.commitment_broken_min_tests
                and commitment.fidelity < cfg.commitment_broken_threshold
            ):
                commitment.status = CommitmentStatus.BROKEN
                await self._update_commitment(commitment)

                # Create RUPTURE turning point (Iron Rule #4)
                tp = TurningPoint(
                    type=TurningPointType.RUPTURE,
                    description=(
                        f"Commitment broken: '{commitment.statement}'. "
                        f"Held in {commitment.tests_held} of {commitment.tests_faced} tests "
                        f"(fidelity: {commitment.fidelity:.2f})."
                    ),
                    surprise_magnitude=0.9,
                    narrative_weight=0.8,
                )

                broken_pairs.append((commitment.id, tp))

                self._logger.warning(
                    "commitment_broken",
                    commitment_id=commitment.id,
                    statement=commitment.statement[:60],
                    fidelity=round(commitment.fidelity, 3),
                    tests=commitment.tests_faced,
                )

        return broken_pairs

    def compute_ipse_score(self) -> float:
        """
        Compute the promise-keeping metric (ipse score).

        ipse = mean(fidelity) for commitments with enough tests.
        """
        tested = [
            c for c in self._active_commitments
            if c.tests_faced >= self._config.commitment_min_tests_for_fidelity
        ]
        if not tested:
            return 1.0  # No tested commitments - default to faithful
        return float(sum(c.fidelity for c in tested) / len(tested))

    def check_strain(self) -> list[str]:
        """
        Check for commitments under strain (fidelity dropping).
        Returns list of strained commitment IDs.
        """
        strained: list[str] = []
        for c in self._active_commitments:
            if (
                c.status != CommitmentStatus.BROKEN
                and c.tests_faced >= self._config.commitment_min_tests_for_fidelity
                and c.fidelity < self._config.commitment_strain_threshold
            ):
                strained.append(c.id)
        return strained

    def _find_commitment(self, commitment_id: str) -> Commitment | None:
        for c in self._active_commitments:
            if c.id == commitment_id:
                return c
        return None

    # ─── LLM Operations ──────────────────────────────────────────────

    async def _llm_test_commitment(
        self,
        commitment_statement: str,
        episode_summary: str,
    ) -> tuple[bool, float]:
        """Use LLM to evaluate commitment fidelity."""

        try:
            response = await self._llm.evaluate(
                prompt=(
                    f'The organism committed to: "{commitment_statement}"\n'
                    f'Recent action: "{episode_summary}"\n\n'
                    "Was this action consistent with the commitment?\n"
                    "Rate fidelity: 0.0 (clear violation) to 1.0 (exemplary adherence).\n"
                    'Respond as JSON: {{"held": true, "fidelity": 0.0}}'
                ),
                max_tokens=100,
                temperature=self._config.llm_temperature_evaluation,
            )

            data = json.loads(response.text)
            held = bool(data.get("held", True))
            fidelity = float(data.get("fidelity", 0.5))
            return (held, max(0.0, min(1.0, fidelity)))

        except Exception as exc:
            self._logger.warning("commitment_test_llm_failed", error=str(exc))
            return (True, 0.5)  # Default: assume held with moderate confidence

    # ─── Neo4j Persistence ───────────────────────────────────────────

    async def _persist_commitment(self, commitment: Commitment) -> None:
        """Create a Commitment node and link to Self."""
        await self._neo4j.execute_write(
            """
            MATCH (s:Self)
            CREATE (c:Commitment {
                id: $id,
                statement: $statement,
                source: $source,
                source_description: $source_description,
                source_episode_ids_json: $source_episode_ids_json,
                drive_alignment_json: $drive_alignment_json,
                status: $status,
                tests_faced: $tests_faced,
                tests_held: $tests_held,
                fidelity: $fidelity,
                made_at: datetime($made_at)
            })
            SET c.embedding = $embedding
            CREATE (s)-[:HOLDS_COMMITMENT]->(c)
            """,
            {
                "id": commitment.id,
                "statement": commitment.statement,
                "source": commitment.source.value,
                "source_description": commitment.source_description,
                "source_episode_ids_json": json.dumps(commitment.source_episode_ids),
                "drive_alignment_json": json.dumps(commitment.drive_alignment.model_dump()),
                "status": commitment.status.value,
                "tests_faced": commitment.tests_faced,
                "tests_held": commitment.tests_held,
                "fidelity": commitment.fidelity,
                "made_at": commitment.made_at.isoformat(),
                "embedding": commitment.embedding,
            },
        )

    async def _update_commitment(self, commitment: Commitment) -> None:
        """Update commitment state in Neo4j."""
        await self._neo4j.execute_write(
            """
            MATCH (c:Commitment {id: $id})
            SET c.status = $status,
                c.tests_faced = $tests_faced,
                c.tests_held = $tests_held,
                c.fidelity = $fidelity,
                c.last_tested = datetime($last_tested)
            """,
            {
                "id": commitment.id,
                "status": commitment.status.value,
                "tests_faced": commitment.tests_faced,
                "tests_held": commitment.tests_held,
                "fidelity": commitment.fidelity,
                "last_tested": (commitment.last_tested or utc_now()).isoformat(),
            },
        )

    async def _link_commitment_episode(
        self,
        commitment_id: str,
        episode_id: str,
        held: bool,
        fidelity: float,
    ) -> None:
        """Create a TESTED_BY relationship."""
        try:
            await self._neo4j.execute_write(
                """
                MATCH (c:Commitment {id: $commitment_id})
                MATCH (e:Episode {id: $episode_id})
                MERGE (c)-[r:TESTED_BY]->(e)
                SET r.held = $held, r.fidelity = $fidelity, r.tested_at = datetime()
                """,
                {
                    "commitment_id": commitment_id,
                    "episode_id": episode_id,
                    "held": held,
                    "fidelity": fidelity,
                },
            )
        except Exception as exc:
            self._logger.debug("commitment_episode_link_failed", error=str(exc))

    def _node_to_commitment(self, node: Any) -> Commitment:
        """Convert a Neo4j node to a Commitment."""
        props = dict(node)
        source_str = props.get("source", "explicit_declaration")
        try:
            source = CommitmentSource(source_str)
        except ValueError:
            source = CommitmentSource.EXPLICIT_DECLARATION

        return Commitment(
            id=props.get("id", ""),
            statement=props.get("statement", ""),
            source=source,
            source_description=props.get("source_description", ""),
            source_episode_ids=json.loads(props.get("source_episode_ids_json", "[]")),
            drive_alignment=DriveAlignmentVector(
                **json.loads(props.get("drive_alignment_json", "{}"))
            ),
            status=CommitmentStatus(props.get("status", "active")),
            tests_faced=int(props.get("tests_faced", 0)),
            tests_held=int(props.get("tests_held", 0)),
            fidelity=float(props.get("fidelity", 1.0)),
            made_at=props.get("made_at", utc_now()),
            last_tested=props.get("last_tested"),
            last_held=props.get("last_held"),
            embedding=props.get("embedding"),
        )
