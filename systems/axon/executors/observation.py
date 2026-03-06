"""
EcodiaOS — Axon Observation Executors

Observation executors are Level 1 (ADVISOR autonomy) — they read and analyse
without modifying world state. They are the lowest-risk executors in the system:
reversible in the sense that they have no side effects to reverse.

ObserveExecutor  — records an observation to Memory (episodic store)
QueryMemoryExecutor — retrieves information from the Memory system
AnalyseExecutor  — runs LLM-based analysis on a topic or dataset
SearchExecutor   — searches external sources (placeholder for future integrations)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

logger = structlog.get_logger()


# ─── ObserveExecutor ──────────────────────────────────────────────


class ObserveExecutor(Executor):
    """
    Record an observation without acting on it.

    Use this when EOS notices something meaningful and wants to commit
    it to episodic memory for future recall. It is not a passive action —
    choosing to observe and record is itself a cognitive act.

    Required params:
      content (str): The observation content to record.

    Optional params:
      salience (float 0-1): Importance signal. Default 0.5.
      tags (list[str]): Labels for retrieval. Default [].
      source (str): Where the observation came from. Default "internal".
    """

    action_type = "observe"
    description = "Record an observation to episodic memory without taking action"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 2000
    rate_limit = RateLimit.unlimited()

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="axon.executor.observe")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("content"):
            return ValidationResult.fail("content is required", content="missing or empty")
        content = params["content"]
        if not isinstance(content, str):
            return ValidationResult.fail("content must be a string", content="wrong type")
        if len(content) > 10_000:
            return ValidationResult.fail("content too long (max 10,000 chars)")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        content = params["content"]
        salience = float(params.get("salience", 0.5))
        tags = list(params.get("tags", []))
        source = str(params.get("source", "internal"))

        self._logger.debug(
            "observe_execute",
            execution_id=context.execution_id,
            content_length=len(content),
            salience=salience,
        )

        if self._memory is not None:
            try:
                episode_id = await self._store_to_memory(
                    content=content,
                    salience=salience,
                    tags=tags,
                    source=source,
                    context=context,
                )
                return ExecutionResult(
                    success=True,
                    data={"episode_id": episode_id, "salience": salience},
                    side_effects=[
                        f"Observation recorded to episodic memory (salience={salience:.2f})",
                    ],
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    error=f"Memory write failed: {exc}",
                )
        else:
            # No memory service — log and succeed (observation still happened)
            self._logger.info(
                "observe_no_memory",
                content=content[:100],
                salience=salience,
            )
            return ExecutionResult(
                success=True,
                data={"episode_id": None, "salience": salience},
                side_effects=["Observation noted (no memory service)"],
            )

    async def _store_to_memory(
        self,
        content: str,
        salience: float,
        tags: list[str],
        source: str,
        context: ExecutionContext,
    ) -> str:
        """Store the observation as an episodic Percept via MemoryService.store_percept()."""
        from primitives.common import Modality, SourceDescriptor, SystemID
        from primitives.percept import Content, Percept

        percept = Percept(
            source=SourceDescriptor(
                system=SystemID.AXON,
                channel=source,
                modality=Modality.INTERNAL,
            ),
            content=Content(raw=content, parsed={"tags": tags}),
            metadata={"source": source},
            salience_hint=salience,
        )
        affect = context.affect_state
        return await self._memory.store_percept(  # type: ignore[union-attr]
            percept,
            salience_composite=salience,
            salience_scores={},
            affect_valence=getattr(affect, "valence", 0.0),
            affect_arousal=getattr(affect, "arousal", 0.0),
            free_energy=0.0,
        )


# ─── QueryMemoryExecutor ──────────────────────────────────────────


class QueryMemoryExecutor(Executor):
    """
    Retrieve information from the Memory system.

    Performs hybrid retrieval (semantic + keyword + graph traversal) and
    returns matching memory traces as new observations — which flow back
    as Percepts into Atune for the next cycle.

    Required params:
      query (str): The retrieval query.

    Optional params:
      max_results (int): Maximum traces to return. Default 5.
      memory_types (list[str]): Filter by type ("episodic", "semantic", "procedural").
      min_salience (float): Minimum salience threshold. Default 0.0.
    """

    action_type = "query_memory"
    description = "Retrieve memories via hybrid semantic + graph retrieval"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 300  # Must fit within memory retrieval budget
    rate_limit = RateLimit.per_minute(60)

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="axon.executor.query_memory")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("query"):
            return ValidationResult.fail("query is required", query="missing or empty")
        if not isinstance(params["query"], str):
            return ValidationResult.fail("query must be a string")
        max_results = params.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1 or max_results > 50:
            return ValidationResult.fail("max_results must be an integer between 1 and 50")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        query = params["query"]
        max_results = int(params.get("max_results", 5))
        min_salience = float(params.get("min_salience", 0.0))

        self._logger.debug(
            "query_memory_execute",
            query=query[:80],
            max_results=max_results,
        )

        if self._memory is None:
            return ExecutionResult(
                success=True,
                data={"results": [], "count": 0},
                new_observations=[],
            )

        try:
            # retrieve() returns MemoryRetrievalResponse with .traces: list[RetrievalResult]
            response = await self._memory.retrieve(
                query_text=query,
                max_results=max_results,
                salience_floor=min_salience,
            )

            traces = response.traces if hasattr(response, "traces") else []
            observations = [
                f"Memory: {r.content[:200]}"
                for r in traces
                if r.content
            ]

            return ExecutionResult(
                success=True,
                data={
                    "count": len(traces),
                    "results": [
                        {
                            "id": getattr(r, "node_id", ""),
                            "content": getattr(r, "content", "")[:500],
                            "salience": getattr(r, "salience", 0.0),
                        }
                        for r in traces
                    ],
                },
                new_observations=observations,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error=f"Memory retrieval failed: {exc}",
            )


# ─── AnalyseExecutor ──────────────────────────────────────────────


class AnalyseExecutor(Executor):
    """
    Run LLM-based analysis on a topic, question, or dataset.

    This is epistemic action — EOS actively reducing uncertainty by
    reasoning about something. The analysis result is returned as a new
    observation, feeding back into the workspace for the next cycle.

    Required params:
      topic (str): What to analyse.
      question (str): The specific question to answer.

    Optional params:
      context_data (str): Additional context to include. Default "".
      depth (str): "shallow" | "deep". Default "shallow".
    """

    action_type = "analyse"
    description = "Run LLM-based analysis to reduce epistemic uncertainty"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 10_000
    rate_limit = RateLimit.per_minute(10)

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._llm = None  # Injected at service startup
        self._logger = logger.bind(system="axon.executor.analyse")

    def set_llm(self, llm: Any) -> None:
        self._llm = llm
        from clients.optimized_llm import OptimizedLLMProvider
        self._optimized = isinstance(llm, OptimizedLLMProvider)

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("topic"):
            return ValidationResult.fail("topic is required")
        if not params.get("question"):
            return ValidationResult.fail("question is required")
        depth = params.get("depth", "shallow")
        if depth not in ("shallow", "deep"):
            return ValidationResult.fail("depth must be 'shallow' or 'deep'")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        topic = params["topic"]
        question = params["question"]
        context_data = str(params.get("context_data", ""))
        depth = params.get("depth", "shallow")

        self._logger.info(
            "analyse_execute",
            topic=topic[:60],
            question=question[:80],
            depth=depth,
        )

        if self._llm is None:
            return ExecutionResult(
                success=True,
                data={"analysis": f"[No LLM] Analysis of '{topic}': {question}"},
                new_observations=[f"Analysis requested on: {topic} — {question}"],
            )

        try:
            prompt = _build_analysis_prompt(topic, question, context_data, depth)

            # Budget check: skip analysis in RED tier
            if getattr(self, "_optimized", False):
                from clients.optimized_llm import OptimizedLLMProvider
                assert isinstance(self._llm, OptimizedLLMProvider)
                if not self._llm.should_use_llm("axon.observation", estimated_tokens=1000):
                    return ExecutionResult(
                        success=True,
                        data={
                            "analysis": f"[Budget exhausted] Analysis of '{topic}' deferred.",
                            "topic": topic,
                        },
                        new_observations=[
                            f"Analysis of '{topic}' deferred due to budget constraints."
                        ],
                    )

            # Use evaluate() for optimized path (standard LLM interface), fall back to complete()
            if getattr(self, "_optimized", False):
                from clients.llm import Message
                response = await self._llm.generate(
                    system_prompt=(
                        "You are EOS — a community care organism. "
                        "Be honest about uncertainty."
                    ),
                    messages=[Message("user", prompt)],
                    max_tokens=1000,
                    temperature=0.3,
                    cache_system="axon.observation",
                    cache_method="analyse",
                )
                analysis = response.text
            else:
                response = await self._llm.evaluate(prompt=prompt, max_tokens=1000)
                analysis = response.text if hasattr(response, "text") else str(response)

            return ExecutionResult(
                success=True,
                data={"analysis": analysis, "topic": topic, "depth": depth},
                new_observations=[f"Analysis of '{topic}': {analysis[:500]}"],
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error=f"Analysis LLM call failed: {exc}",
            )


def _build_analysis_prompt(
    topic: str,
    question: str,
    context_data: str,
    depth: str,
) -> str:
    depth_instruction = (
        "Provide a brief, focused analysis (2-3 paragraphs)."
        if depth == "shallow"
        else "Provide a thorough, multi-perspective analysis."
    )
    ctx_section = f"\n\nAdditional context:\n{context_data}" if context_data else ""
    return (
        f"You are EOS — a community care organism. Analyse the following:\n\n"
        f"Topic: {topic}\n"
        f"Question: {question}"
        f"{ctx_section}\n\n"
        f"{depth_instruction}\n"
        f"Be honest about uncertainty. Ground your analysis in what is actually known."
    )


# ─── SearchExecutor ───────────────────────────────────────────────


class SearchExecutor(Executor):
    """
    Search internal and external sources for information.

    Supports three search modes:
      - "knowledge_base": Hybrid Memory retrieval (semantic + graph)
      - "community_docs": Memory retrieval filtered to community-tagged episodes
      - "web": LLM-synthesised answer when no Memory results found,
               or Memory results supplemented with LLM reasoning

    Required params:
      query (str): The search query.

    Optional params:
      source (str): "web" | "knowledge_base" | "community_docs". Default "knowledge_base".
      max_results (int): Maximum results. Default 5.
    """

    action_type = "search"
    description = "Search internal knowledge and external sources for information"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 10_000
    rate_limit = RateLimit.per_minute(20)

    def __init__(
        self,
        memory: MemoryService | None = None,
        llm: Any = None,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._logger = logger.bind(system="axon.executor.search")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("query"):
            return ValidationResult.fail("query is required")
        source = params.get("source", "knowledge_base")
        valid_sources = ("web", "knowledge_base", "community_docs")
        if source not in valid_sources:
            return ValidationResult.fail(
                f"source must be one of {valid_sources}",
                source="invalid value",
            )
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        query = params["query"]
        source = params.get("source", "knowledge_base")
        max_results = int(params.get("max_results", 5))

        self._logger.info(
            "search_execute",
            query=query[:80],
            source=source,
            max_results=max_results,
            execution_id=context.execution_id,
        )

        results: list[dict[str, Any]] = []
        observations: list[str] = []

        # --- Phase 1: Memory retrieval (knowledge_base + community_docs) ---
        if self._memory is not None and source in ("knowledge_base", "community_docs"):
            try:
                response = await self._memory.retrieve(
                    query_text=query,
                    max_results=max_results,
                )
                traces = response.traces if hasattr(response, "traces") else []
                for r in traces:
                    content = getattr(r, "content", "")
                    if not content:
                        continue
                    results.append({
                        "id": getattr(r, "node_id", ""),
                        "content": content[:500],
                        "salience": round(getattr(r, "salience", 0.0), 4),
                        "source": "memory",
                    })
                    observations.append(f"Found: {content[:200]}")
            except Exception as exc:
                self._logger.warning("search_memory_error", error=str(exc))

        # --- Phase 2: LLM-synthesised search for "web" or empty results ---
        if source == "web" or (not results and self._llm is not None):
            try:
                llm_results = await self._llm_search(query, max_results)
                results.extend(llm_results)
                for lr in llm_results:
                    observations.append(f"LLM: {lr['content'][:200]}")
            except Exception as exc:
                self._logger.warning("search_llm_error", error=str(exc))

        return ExecutionResult(
            success=True,
            data={
                "query": query,
                "source": source,
                "max_results": max_results,
                "results": results,
                "count": len(results),
            },
            new_observations=observations[:max_results],
        )

    async def _llm_search(
        self, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Use LLM to synthesise search results from its knowledge."""
        if self._llm is None:
            return []

        prompt = (
            f"You are EOS searching for information. Answer the following query "
            f"concisely with up to {max_results} distinct facts or findings.\n\n"
            f"Query: {query}\n\n"
            f"Respond with numbered points. Be specific and factual. "
            f"If uncertain, say so."
        )

        try:
            from clients.llm import Message

            response = await self._llm.generate(
                system_prompt="You are a knowledge search engine. Return concise, factual results.",
                messages=[Message("user", prompt)],
                max_tokens=800,
                temperature=0.2,
                cache_system="axon.search",
                cache_method="llm_search",
            )
            text = response.text if hasattr(response, "text") else str(response)
        except (ImportError, AttributeError):
            # Fallback to evaluate() if generate() unavailable
            try:
                response = await self._llm.evaluate(prompt=prompt, max_tokens=800)
                text = response.text if hasattr(response, "text") else str(response)
            except Exception:
                return []

        # Parse numbered results from LLM response
        results: list[dict[str, Any]] = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading number/bullet
            clean = line.lstrip("0123456789.-) ").strip()
            if clean:
                results.append({
                    "content": clean[:500],
                    "source": "llm",
                    "salience": 0.3,
                })
                if len(results) >= max_results:
                    break

        return results
