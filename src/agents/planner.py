"""Planner agent (PydanticAI) on DeepSeek V4 Pro via OpenRouter.

Typed input: IssueContext (issue text + repo metadata + RAG hits + episodic memory hits).
Typed output: ChangePlan (target_files: list[str], rationale: str, approach: str).
Free-form text outside the typed schema is rejected by the output parser.

Uses pgvector RAG retrieval and episodic store query as tools.
Locked to deepseek/deepseek-v4-pro model ID (VAL-PLANNER-002).
Validates output schema on every call (VAL-PLANNER-005).

Architecture reference: §2.3 Specialized Agents.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID, uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.agents.models import ChangePlan, IssueContext
from src.llm.client import get_temperature
from src.llm.cost import estimate_cost_tiktoken
from src.memory.episodic.models import CreateDecisionParams

if TYPE_CHECKING:
    from src.agents.models import EpisodicFact, RAGHit  # noqa: F401

load_dotenv()

logger = logging.getLogger(__name__)

# ── Model ID (VAL-PLANNER-002) ────────────────────────────────────
# Locked to deepseek/deepseek-v4-pro via OpenRouter.
_PLANNER_MODEL: str = "deepseek/deepseek-v4-pro"

# PydanticAI uses "openrouter:<model_id>" format for OpenRouter models.
_PYDANTIC_AI_MODEL: str = f"openrouter:{_PLANNER_MODEL}"


# ── RAG Retriever Protocol ─────────────────────────────────────────


class RAGRetriever(Protocol):
    """Protocol for RAG retrieval backends.

    Implementations must return results scoped to the given repo_url
    (VAL-PLANNER-003).
    """

    async def retrieve(
        self,
        *,
        query: str,
        repo_url: str,
        top_k: int = 8,
    ) -> list[dict[str, Any]]: ...


    async def close(self) -> None: ...


# ── Episodic Store Protocol ────────────────────────────────────────


class EpisodicStoreProtocol(Protocol):
    """Minimal protocol for episodic store operations needed by the Planner.

    Only read operations are exposed to agents per architecture §2.6.
    """

    async def get_planner_context(
        self,
        repo_url: str,
        *,
        recent_limit: int = 5,
    ) -> dict[str, Any]: ...

    async def create_decision(
        self,
        params: CreateDecisionParams,
    ) -> Any: ...

    async def close(self) -> None: ...


# ── Planner Dependencies ───────────────────────────────────────────


class PlannerDeps(BaseModel):
    """Dependencies injected into the Planner agent at runtime.

    These are NOT passed to the LLM — they are available to tool
    implementations via RunContext.deps.
    """

    model_config = {"arbitrary_types_allowed": True}

    episodic_store: Any | None = Field(
        default=None,
        description="Episodic store for reading repo_facts and persisting decisions",
    )
    rag_retriever: Any | None = Field(
        default=None,
        description="RAG retriever for pgvector semantic search",
    )
    task_id: UUID = Field(
        default_factory=uuid4,
        description="Current task ID for decision persistence",
    )
    trace_id: str = Field(
        default="",
        description="Langfuse trace ID for span hierarchy",
    )
    repo_url: str = Field(
        default="",
        description="Current repo URL for RAG/episodic scoping",
    )
    issue_context: IssueContext | None = Field(
        default=None,
        description="Full IssueContext with RAG hits and episodic facts",
    )


@dataclass(frozen=True)
class PlannerRunResult:
    """Planner output plus usage metadata for orchestrator accounting."""

    plan: ChangePlan
    tokens_in: int
    tokens_out: int
    cached_tokens: int
    cost_usd: Decimal


# ── System Prompt ───────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the Planner agent in an autonomous multi-agent SDLC system.
Your job is to analyze a GitHub issue and produce a structured change plan.

Given:
- The issue text (bug report, feature request, etc.)
- Repository metadata (language, file tree, etc.)
- RAG retrieval results (relevant code chunks from the repo)
- Episodic memory (past decisions, outcomes, and repo-specific facts)

You MUST produce a ChangePlan with:
1. target_files: List of file paths that need to be modified (at least 1)
2. rationale: A detailed explanation (at least 20 characters) of WHY these changes are needed
3. approach: A high-level strategy for implementing the changes
4. estimated_complexity: "low", "medium", or "high"

RULES:
- Your output MUST be a valid ChangePlan. Do NOT include free-form text.
- If you are unsure about a file path, include it with a note in the approach.
- Use the RAG retrieval tool to search for relevant code before planning.
- Use the episodic query tool to check past decisions for this repo.
- The rationale must be substantive — explain the root cause, not just restate the issue.
- The target_files list must only include files that actually need modification.
"""


# ── Create the Agent ───────────────────────────────────────────────

planner = Agent(
    model=_PYDANTIC_AI_MODEL,
    output_type=ChangePlan,
    name="planner",
    deps_type=PlannerDeps,
    system_prompt=_SYSTEM_PROMPT,
    retries=3,  # Allow PydanticAI to retry on validation failure
)


# ── Tool: RAG Retrieval ────────────────────────────────────────────


@planner.tool
async def _rag_retrieval(ctx: RunContext[PlannerDeps], query: str) -> str:
    """Search the repository code using semantic RAG retrieval.

    Returns relevant code chunks scoped to the current repo_url.
    Use this to find the exact code that needs to be modified.

    Args:
        query: Natural language search query (e.g., "subtract function implementation")
    """
    if ctx.deps.rag_retriever is None:
        return "RAG retrieval not available. Proceed without code search results."

    try:
        # Support both dict-based and RetrievalResult-based retrievers.
        # SemanticStore has retrieve_dicts that returns list[dict],
        # while mock retrievers use retrieve directly.
        from src.memory.semantic.store import SemanticStore

        if isinstance(ctx.deps.rag_retriever, SemanticStore):
            results: list[dict[str, Any]] = await ctx.deps.rag_retriever.retrieve_dicts(
                query=query,
                repo_url=ctx.deps.repo_url,
                top_k=8,
            )
        else:
            results = await ctx.deps.rag_retriever.retrieve(
                query=query,
                repo_url=ctx.deps.repo_url,
                top_k=8,
            )
        if not results:
            return "No RAG results found for this query."

        # Format results for the LLM
        formatted: list[str] = []
        for i, hit in enumerate(results[:8], 1):
            file_path = hit.get("file_path", "unknown")
            chunk_text = hit.get("chunk_text", "")
            score = hit.get("score", 0.0)
            formatted.append(
                f"[{i}] {file_path} (score: {score:.3f}):\n{chunk_text}"
            )
        return "\n\n".join(formatted)
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        return f"RAG retrieval error: {exc}"


# ── Tool: Episodic Memory Query ────────────────────────────────────


@planner.tool
async def _episodic_query(ctx: RunContext[PlannerDeps], repo_url: str) -> str:
    """Query episodic memory for facts and history about this repository.

    Returns repo_facts, recent decisions, and outcomes for the given repo.
    Use this to understand past changes and known issues.

    Args:
        repo_url: The GitHub repository URL to query facts for
    """
    if ctx.deps.episodic_store is None:
        return "Episodic store not available. Proceed without historical context."

    try:
        context: dict[str, Any] = await ctx.deps.episodic_store.get_planner_context(
            repo_url=repo_url,
            recent_limit=5,
        )
        if not context:
            return "No episodic context found for this repository."

        return json.dumps(context, default=str, indent=2)
    except Exception as exc:
        logger.warning("Episodic query failed: %s", exc)
        return f"Episodic query error: {exc}"


# ── Persistence Helper ─────────────────────────────────────────────


async def persist_change_plan(
    *,
    store: Any,
    task_id: UUID,
    plan: ChangePlan,
    step_index: int = 0,
) -> None:
    """Persist a successfully parsed ChangePlan as a decision row.

    Writes to the decisions table with:
      - agent = 'planner'
      - decision_type = 'change_plan'
      - decision_data = full Pydantic JSON of the ChangePlan

    This is called by the orchestrator after a successful Planner run
    (VAL-PLANNER-006).
    """
    params = CreateDecisionParams(
        task_id=task_id,
        agent="planner",
        step_index=step_index,
        decision_type="change_plan",
        decision_data=plan.model_dump(mode="json"),
    )
    await store.create_decision(params)
    logger.info(
        "Persisted ChangePlan for task %s: target_files=%s",
        task_id,
        plan.target_files,
    )


# ── Convenience: Run Planner with Full Context ────────────────────


def _extract_usage_tokens(result: Any) -> tuple[int, int]:
    """Extract request/response token counts from a PydanticAI result."""
    usage_attr = getattr(result, "usage", None)
    usage = usage_attr() if callable(usage_attr) else usage_attr
    if usage is None:
        return 0, 0

    tokens_in = (
        getattr(usage, "request_tokens", 0)
        or getattr(usage, "input_tokens", 0)
        or getattr(usage, "prompt_tokens", 0)
        or 0
    )
    tokens_out = (
        getattr(usage, "response_tokens", 0)
        or getattr(usage, "output_tokens", 0)
        or getattr(usage, "completion_tokens", 0)
        or 0
    )
    return int(tokens_in), int(tokens_out)


async def run_planner(
    *,
    issue_context: IssueContext,
    episodic_store: Any | None = None,
    rag_retriever: Any | None = None,
    task_id: UUID | None = None,
    trace_id: str = "",
    return_metadata: bool = False,
) -> ChangePlan | PlannerRunResult:
    """Run the Planner agent with full IssueContext.

    This is the primary entry point for the orchestrator. It:
    1. Builds PlannerDeps with the provided store/retriever
    2. Constructs the user prompt from the IssueContext
    3. Runs the Planner agent
    4. Persists the resulting ChangePlan as a decision row

    Args:
        issue_context: The full typed context for the issue.
        episodic_store: Optional episodic store for repo_facts + persistence.
        rag_retriever: Optional RAG retriever for code search.
        task_id: The current task's UUID (for decision persistence).
        trace_id: Langfuse trace ID for span hierarchy.

        return_metadata: Return PlannerRunResult with usage/cost metadata.

    Returns:
        A Pydantic-valid ChangePlan, or PlannerRunResult when requested.
    """
    if task_id is None:
        task_id = uuid4()

    deps = PlannerDeps(
        episodic_store=episodic_store,
        rag_retriever=rag_retriever,
        task_id=task_id,
        trace_id=trace_id,
        repo_url=issue_context.repo_url,
        issue_context=issue_context,
    )

    # Build the user prompt from IssueContext
    user_prompt = _build_planner_prompt(issue_context)

    result = await planner.run(
        user_prompt,
        deps=deps,
        model_settings={"temperature": get_temperature()},
    )
    plan: ChangePlan = result.output
    tokens_in, tokens_out = _extract_usage_tokens(result)
    cost_usd = estimate_cost_tiktoken(
        model=_PLANNER_MODEL,
        prompt_tokens=tokens_in,
        completion_tokens=tokens_out,
    )

    # Persist the ChangePlan if a store is available
    if episodic_store is not None:
        await persist_change_plan(
            store=episodic_store,
            task_id=task_id,
            plan=plan,
            step_index=0,
        )

    if return_metadata:
        return PlannerRunResult(
            plan=plan,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached_tokens=0,
            cost_usd=cost_usd,
        )

    return plan


def _build_planner_prompt(ctx: IssueContext) -> str:
    """Build the user prompt for the Planner from an IssueContext.

    Includes issue text, repo metadata, pre-fetched RAG hits,
    and episodic facts so the Planner has immediate context
    even before using tools.
    """
    parts: list[str] = []

    # Issue information
    parts.append(f"## Issue (#{ctx.issue_number})")
    parts.append(ctx.issue_text)
    parts.append("")

    # Repo metadata
    parts.append(f"## Repository: {ctx.repo_url}")
    if ctx.repo_files:
        parts.append("### Key Files:")
        for path, content in list(ctx.repo_files.items())[:10]:
            truncated = content[:2000] + ("…" if len(content) > 2000 else "")
            parts.append(f"**{path}:**\n```\n{truncated}\n```")
    parts.append("")

    # Pre-fetched RAG hits (from IssueContext)
    if ctx.rag_hits:
        parts.append("### RAG Hits (pre-fetched):")
        for i, hit in enumerate(ctx.rag_hits[:8], 1):
            parts.append(f"[{i}] {hit.file_path} (score: {hit.score:.3f}):\n{hit.chunk_text}")
        parts.append("")

    # Episodic facts
    if ctx.repo_facts:
        parts.append("### Repo Facts:")
        for repo_fact in ctx.repo_facts:
            parts.append(f"- {repo_fact}")
        parts.append("")

    if ctx.episodic_facts:
        parts.append("### Episodic Facts:")
        for epi_fact in ctx.episodic_facts:
            parts.append(f"- {epi_fact.fact_kind}: {epi_fact.fact_value}")
        parts.append("")

    if ctx.recent_decisions:
        parts.append("### Recent Decisions:")
        for d in ctx.recent_decisions[:5]:
            parts.append(f"- {d}")
        parts.append("")

    parts.append(
        "Analyze the issue and produce a ChangePlan. "
        "Use the RAG retrieval tool if you need more code context. "
        "Use the episodic query tool if you need more repo history."
    )

    return "\n".join(parts)
