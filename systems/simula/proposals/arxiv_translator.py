"""
EcodiaOS - Simula ArXiv Proposal Translator

Translates raw arXiv technique dictionaries (produced by an upstream
scraper/worker) into Simula's native EvolutionProposal schema, then
dispatches them to the SimulaCodeAgent pipeline for autonomous
implementation - gated behind EXPERIMENTAL governance review.

Pipeline:
  1. Validate & normalise the raw arXiv dict
  2. Map the abstract technique to a concrete target directory
  3. Infer ChangeCategory + build ChangeSpec
  4. Construct EvolutionProposal with EXPERIMENTAL flag
  5. Dispatch to SimulaService.process_proposal() (governance-gated)

Every arXiv-sourced proposal enters the pipeline with
``source="arxiv"`` and ``category=ADD_SYSTEM_CAPABILITY``, which is a
GOVERNANCE_REQUIRED category.  This forces Equor governance review
before any code is merged - no arXiv proposal can land autonomously.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel
from systems.simula.evolution_types import (
    GOVERNANCE_REQUIRED,
    ChangeCategory,
    ChangeSpec,
    EvolutionProposal,
    ProposalResult,
    ProposalStatus,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger().bind(module="simula.arxiv_translator")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class ExperimentalStatus(enum.StrEnum):
    """Lifecycle states for experimental arXiv-sourced proposals."""

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


# Keyword → target directory mapping.  Each entry maps a set of
# technique keywords (lowercased) to the sub-directory under
# ``systems/simula/`` where the implementation should land.
_TECHNIQUE_DIRECTORY_MAP: list[tuple[list[str], str]] = [
    (["e-graph", "egraph", "equality saturation", "eqsat"], "egraph"),
    (["synthesis", "neurosymbolic", "hysynth", "cegis", "sketch"], "synthesis"),
    (["verification", "dafny", "lean", "proof", "formal"], "verification"),
    (["retrieval", "swe-grep", "code search", "rag"], "retrieval"),
    (["learning", "grpo", "fine-tuning", "lilo", "rlhf"], "learning"),
    (["repair", "diffusion", "denoise", "patch", "apr"], "agents"),
    (["orchestration", "multi-agent", "dag", "pipeline"], "orchestration"),
    (["debugging", "causal", "root cause", "fault"], "debugging"),
    (["coevolution", "adversarial", "hard negative", "robustness"], "coevolution"),
    (["inspector", "vulnerability", "zero-day", "exploit", "fuzzing"], "inspector"),
    (["audit", "hash chain", "c2pa", "credential", "provenance"], "audit"),
    (["symbolic", "smt", "z3", "constraint", "solver"], "verification"),
    (["filter", "xdp", "ebpf", "shield", "firewall"], "inspector"),
    (["correlation", "anomaly", "drift", "monitoring"], "correlation"),
    (["protocol", "state machine", "fsm"], "protocol"),
]

# Fallback when no keyword matches.
_DEFAULT_TARGET_DIR = "proposals/experimental"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ArxivTechnique(EOSBaseModel):
    """
    Normalised representation of a technique scraped from arXiv.

    The upstream worker emits raw dicts; this model validates and
    structures them before translation.
    """

    paper_id: str  # e.g. "2401.12345"
    title: str
    abstract: str
    technique_name: str  # e.g. "E-Graph Optimization"
    technique_keywords: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    arxiv_url: str = ""
    relevance_score: float = 0.0  # 0.0–1.0, set by upstream ranker


class ArxivTranslationResult(EOSBaseModel):
    """Outcome of translating + dispatching one arXiv technique."""

    proposal: EvolutionProposal
    target_directory: str
    experimental_status: ExperimentalStatus = ExperimentalStatus.PENDING_REVIEW
    dispatch_result: ProposalResult | None = None


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


class ArxivProposalTranslator:
    """
    Translates raw arXiv technique dictionaries into
    ``EvolutionProposal`` objects and dispatches them through the
    Simula pipeline with mandatory governance gating.

    Usage::

        from systems.simula.proposals.arxiv_translator import (
            ArxivProposalTranslator,
        )

        translator = ArxivProposalTranslator()
        result = await translator.translate_and_dispatch(
            raw_technique={"paper_id": "2401.12345", ...},
            simula_service=service,
        )
    """

    def __init__(
        self,
        target_dir_map: list[tuple[list[str], str]] | None = None,
        min_relevance_score: float = 0.0,
        neo4j: Neo4jClient | None = None,
    ) -> None:
        self._dir_map = target_dir_map or _TECHNIQUE_DIRECTORY_MAP
        self._min_relevance = min_relevance_score
        self._neo4j = neo4j
        self._log = logger

    # -- public API ---------------------------------------------------------

    def translate(self, raw: dict[str, Any]) -> ArxivTranslationResult:
        """
        Translate a raw arXiv technique dict into an ``EvolutionProposal``.

        Does NOT dispatch - call ``dispatch`` or ``translate_and_dispatch``
        to push into the Simula pipeline.

        Raises ``ValueError`` if required fields are missing.
        """
        technique = self._validate(raw)
        target_dir = self._resolve_target_directory(technique)
        proposal = self._build_proposal(technique, target_dir)

        self._log.info(
            "arxiv_translated",
            paper_id=technique.paper_id,
            technique=technique.technique_name,
            target_dir=target_dir,
            proposal_id=proposal.id,
        )

        return ArxivTranslationResult(
            proposal=proposal,
            target_directory=target_dir,
            experimental_status=ExperimentalStatus.PENDING_REVIEW,
        )

    async def dispatch(
        self,
        result: ArxivTranslationResult,
        simula_service: Any,
    ) -> ArxivTranslationResult:
        """
        Dispatch a translated proposal into ``SimulaService.process_proposal``.

        The proposal's category (``ADD_SYSTEM_CAPABILITY``) is in
        ``GOVERNANCE_REQUIRED``, so the pipeline will park it in
        ``AWAITING_GOVERNANCE`` until Equor approves.

        Before dispatching, the paper and extracted technique are written to
        the Neo4j memory graph so the SimulaCodeAgent can retrieve the source
        theory when it later implements the proposal.  Graph write failures
        are logged and silently swallowed - they never block dispatch.
        """
        assert result.proposal.category in GOVERNANCE_REQUIRED, (
            f"ArXiv proposals must use a governance-gated category, "
            f"got {result.proposal.category.value}"
        )

        # -- Memory injection (graceful degradation) ---------------------------
        await self._write_to_memory(result)

        self._log.info(
            "arxiv_dispatching",
            proposal_id=result.proposal.id,
            category=result.proposal.category.value,
        )

        pipeline_result: ProposalResult = await simula_service.process_proposal(
            result.proposal,
        )
        result.dispatch_result = pipeline_result

        self._log.info(
            "arxiv_dispatched",
            proposal_id=result.proposal.id,
            pipeline_status=pipeline_result.status.value,
        )
        return result

    async def _write_to_memory(self, result: ArxivTranslationResult) -> None:
        """
        Write PaperNode + ConceptNode to the Neo4j graph.

        Silently returns on any error - callers must not depend on this
        completing successfully (graceful degradation contract).
        """
        if self._neo4j is None:
            return

        from systems.simula.proposals.paper_memory import (
            upsert_paper_and_concept,
        )

        # Recover technique metadata from the proposal's change_spec
        # (the translator always stores paper_id in code_hint and the
        # technique name in the description prefix).
        technique_name = self._extract_technique_name(result.proposal.description)
        paper_id, arxiv_url, abstract = self._extract_paper_fields(
            result.proposal.change_spec
        )
        title = result.proposal.description.removeprefix(
            f"[EXPERIMENTAL/arXiv] Implement '{technique_name}' from paper {paper_id}: "
        )

        await upsert_paper_and_concept(
            self._neo4j,
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            arxiv_url=arxiv_url,
            technique_name=technique_name,
        )

    @staticmethod
    def _extract_technique_name(description: str) -> str:
        """
        Pull technique name from the canonical description format:
        ``[EXPERIMENTAL/arXiv] Implement '<technique>' from paper ...``
        """
        prefix = "[EXPERIMENTAL/arXiv] Implement '"
        if description.startswith(prefix):
            rest = description[len(prefix):]
            return rest.split("'")[0]
        return description[:80]

    @staticmethod
    def _extract_paper_fields(
        change_spec: ChangeSpec,
    ) -> tuple[str, str, str]:
        """Return (paper_id, arxiv_url, abstract) from a ChangeSpec."""
        # code_hint format: "Target directory: ...\nTechnique: ...\nPaper: <id>"
        paper_id = ""
        for line in (change_spec.code_hint or "").splitlines():
            if line.startswith("Paper: "):
                paper_id = line.removeprefix("Paper: ").strip()
                break

        # additional_context contains "Source: arXiv (<url>)" on first line
        arxiv_url = ""
        for line in (change_spec.additional_context or "").splitlines():
            if line.startswith("Source: arXiv ("):
                arxiv_url = line.removeprefix("Source: arXiv (").rstrip(")")
                break

        # abstract is the first 300 chars embedded in capability_description
        abstract = ""
        cap = change_spec.capability_description or ""
        abstract_marker = "Abstract: "
        idx = cap.find(abstract_marker)
        if idx != -1:
            abstract = cap[idx + len(abstract_marker):]

        return paper_id, arxiv_url, abstract

    async def translate_and_dispatch(
        self,
        raw_technique: dict[str, Any],
        simula_service: Any,
    ) -> ArxivTranslationResult:
        """Convenience: translate + dispatch in one call."""
        result = self.translate(raw_technique)

        if result.proposal.category not in GOVERNANCE_REQUIRED:
            self._log.error(
                "arxiv_safety_block",
                proposal_id=result.proposal.id,
                reason="ArXiv proposals must be governance-gated",
            )
            result.dispatch_result = ProposalResult(
                status=ProposalStatus.REJECTED,
                reason="ArXiv proposals must use a GOVERNANCE_REQUIRED category",
            )
            return result

        return await self.dispatch(result, simula_service)

    # -- internal -----------------------------------------------------------

    def _validate(self, raw: dict[str, Any]) -> ArxivTechnique:
        """Validate and normalise a raw arXiv dict into ``ArxivTechnique``."""
        required = ("paper_id", "title", "technique_name")
        missing = [k for k in required if not raw.get(k)]
        if missing:
            raise ValueError(
                f"ArXiv technique dict missing required fields: {missing}"
            )

        # Normalise keywords from abstract if not supplied
        keywords = raw.get("technique_keywords") or []
        if not keywords and raw.get("abstract"):
            keywords = self._extract_keywords(raw["abstract"])

        return ArxivTechnique(
            paper_id=str(raw["paper_id"]),
            title=str(raw["title"]),
            abstract=str(raw.get("abstract", "")),
            technique_name=str(raw["technique_name"]),
            technique_keywords=keywords,
            authors=raw.get("authors", []),
            arxiv_url=str(raw.get("arxiv_url", "")),
            relevance_score=float(raw.get("relevance_score", 0.0)),
        )

    def _resolve_target_directory(self, technique: ArxivTechnique) -> str:
        """
        Map a technique to a target directory under ``systems/simula/``.

        Matches against technique_name, technique_keywords, and abstract
        (in that priority order).  Returns the first match.
        """
        # Build a single searchable string
        searchable = " ".join([
            technique.technique_name.lower(),
            " ".join(k.lower() for k in technique.technique_keywords),
            technique.abstract.lower()[:500],
        ])

        for keywords, directory in self._dir_map:
            for keyword in keywords:
                if keyword in searchable:
                    return f"systems/simula/{directory}"

        return f"systems/simula/{_DEFAULT_TARGET_DIR}"

    def _build_proposal(
        self,
        technique: ArxivTechnique,
        target_dir: str,
    ) -> EvolutionProposal:
        """
        Build a governance-gated ``EvolutionProposal`` from an arXiv technique.

        Key design decisions:
          - ``source="arxiv"`` distinguishes from evo/governance/bounty origins
          - ``category=ADD_SYSTEM_CAPABILITY`` forces Equor governance review
          - ``risk_assessment`` flags EXPERIMENTAL status prominently
          - ``change_spec.code_hint`` encodes the target directory
        """
        description = (
            f"[EXPERIMENTAL/arXiv] Implement '{technique.technique_name}' "
            f"from paper {technique.paper_id}: {technique.title}"
        )

        change_spec = ChangeSpec(
            capability_description=(
                f"Implement {technique.technique_name} based on "
                f"arXiv paper {technique.paper_id}. "
                f"Abstract: {technique.abstract[:300]}"
            ),
            affected_systems=["simula"],
            additional_context=(
                f"Source: arXiv ({technique.arxiv_url or technique.paper_id})\n"
                f"Authors: {', '.join(technique.authors[:5])}\n"
                f"Keywords: {', '.join(technique.technique_keywords[:10])}\n"
                f"Relevance: {technique.relevance_score:.2f}\n"
                f"EXPERIMENTAL: Requires human/Equor governance approval before merge."
            ),
            code_hint=(
                f"Target directory: {target_dir}\n"
                f"Technique: {technique.technique_name}\n"
                f"Paper: {technique.paper_id}"
            ),
        )

        return EvolutionProposal(
            source="arxiv",
            category=ChangeCategory.ADD_SYSTEM_CAPABILITY,
            description=description,
            change_spec=change_spec,
            evidence=[technique.paper_id],
            expected_benefit=(
                f"Incorporate state-of-the-art technique "
                f"'{technique.technique_name}' from recent research. "
                f"Relevance score: {technique.relevance_score:.2f}"
            ),
            risk_assessment=(
                f"EXPERIMENTAL - arXiv-sourced proposal. "
                f"Risk level: HIGH (unvalidated research technique). "
                f"Requires Equor governance approval before any code is merged. "
                f"Paper: {technique.paper_id}"
            ),
            status=ProposalStatus.PROPOSED,
        )

    @staticmethod
    def _extract_keywords(abstract: str) -> list[str]:
        """
        Extract candidate keywords from an abstract by matching against
        the known technique vocabulary.  Zero LLM tokens.
        """
        text = abstract.lower()
        found: list[str] = []
        for keywords, _dir in _TECHNIQUE_DIRECTORY_MAP:
            for kw in keywords:
                if kw in text and kw not in found:
                    found.append(kw)
        return found
