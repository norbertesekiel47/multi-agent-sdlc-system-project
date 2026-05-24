"""Coder agent (PydanticAI) on DeepSeek V4 Flash via OpenRouter.

Typed input: ChangePlan (+ optional ReviewResult from prior turn).
Typed output: CodeEdit (diff, touched_files, diff_hash).
Free-form text outside the typed schema is rejected by the output parser.

Tools:
  - sandbox file read
  - sandbox file write (apply_diff)
  - pgvector RAG retrieval

Prompt cache markers on static repo-context block (§2.10).
Locked to deepseek/deepseek-chat-v3-0324 model ID (VAL-CODER-003).

The ``run_coder`` function uses ``LLMClient.chat_with_cache()``
with ``_build_coder_prompt()`` so that the StructuredPrompt
with cache markers reaches OpenRouter.  Token counts (including
cached_tokens) are extracted from the LLM response and returned
via ``CoderRunResult`` for accumulation by the orchestrator.

Architecture reference: §2.3 Specialized Agents.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID, uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.agents.models import ChangePlan, CodeEdit, ReviewResult
from src.llm.caching import StructuredPrompt, build_structured_prompt
from src.llm.client import get_llm_client
from src.llm.cost import estimate_cost_tiktoken
from src.memory.episodic.models import CreateDecisionParams

if TYPE_CHECKING:
    from src.agents.models import RAGHit  # noqa: F401

load_dotenv()

logger = logging.getLogger(__name__)


# ── Coder Run Result ───────────────────────────────────────────────


@dataclass
class CoderRunResult:
    """Result of a Coder agent run, including token and cost metadata.

    The orchestrator extracts ``tokens_in``, ``tokens_out``, and
    ``cached_tokens`` from this result and accumulates them into
    the task's ``total_tokens_in/out/cached`` state fields.
    """

    edit: CodeEdit
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    cost_usd: Decimal = Decimal("0")


# ── Model ID (VAL-CODER-003) ────────────────────────────────────
# Locked to deepseek/deepseek-chat-v3-0324 via OpenRouter.
# NOTE: deepseek-v4-flash is aspirational but unstable on OpenRouter;
# deepseek-chat-v3-0324 is the stable fallback per AGENTS.md.
_CODER_MODEL: str = "deepseek/deepseek-chat-v3-0324"

# PydanticAI uses "openrouter:<model_id>" format for OpenRouter models.
_PYDANTIC_AI_MODEL: str = f"openrouter:{_CODER_MODEL}"


# ── RAG Retriever Protocol ─────────────────────────────────────────


class RAGRetriever(Protocol):
    """Protocol for RAG retrieval backends.

    Implementations must return results scoped to the given repo_url
    (VAL-CODER-004).
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
    """Minimal protocol for episodic store operations needed by the Coder.

    Only create_decision is exposed (reads go through the orchestrator).
    """

    async def create_decision(
        self,
        params: CreateDecisionParams,
    ) -> Any: ...

    async def close(self) -> None: ...


# ── Coder Dependencies ─────────────────────────────────────────────


class CoderDeps(BaseModel):
    """Dependencies injected into the Coder agent at runtime.

    These are NOT passed to the LLM — they are available to tool
    implementations via RunContext.deps.
    """

    model_config = {"arbitrary_types_allowed": True}

    sandbox_manager: Any | None = Field(
        default=None,
        description="Sandbox manager for file read/write/apply_diff operations",
    )
    rag_retriever: Any | None = Field(
        default=None,
        description="RAG retriever for pgvector semantic search",
    )
    episodic_store: Any | None = Field(
        default=None,
        description="Episodic store for persisting decisions",
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
        description="Current repo URL for RAG scoping",
    )


# ── System Prompt (static block for caching) ──────────────────────

_SYSTEM_PROMPT = """\
You are the Coder agent in an autonomous multi-agent SDLC system.
Your job is to produce a code edit that fixes a GitHub issue,
following a ChangePlan produced by the Planner.

Given:
- A ChangePlan specifying target files and approach
- Optional ReviewResult from a prior review (if the previous edit was rejected)
- Repository context from RAG retrieval

You MUST produce a CodeEdit with:
1. diff: A unified diff string that can be applied with `patch -p1`
2. touched_files: List of file paths modified by this diff
3. diff_hash: SHA-256 hash of the diff (for loop detection / reviewer rejection tracking)

RULES:
- Your output MUST be a valid CodeEdit. Do NOT include free-form text.
- Always read the files you need to edit before producing a diff.
- Use the RAG retrieval tool to search for relevant code context.
- Use the sandbox_apply_diff tool to apply your diff in the sandbox.
- When given a ReviewResult with verdict=reject_with_changes, address the issues
  in your new diff. Do NOT produce the same diff as before.
- The diff must be a valid unified diff format that `patch -p1` can apply.
- The touched_files list must match the files actually changed in the diff.
"""


# ── Create the Agent ───────────────────────────────────────────────

coder = Agent(
    model=_PYDANTIC_AI_MODEL,
    output_type=CodeEdit,
    name="coder",
    deps_type=CoderDeps,
    system_prompt=_SYSTEM_PROMPT,
    retries=3,  # Allow PydanticAI to retry on validation failure
)


# ── Tool: Sandbox File Read ────────────────────────────────────────


@coder.tool
async def _sandbox_read_file(ctx: RunContext[CoderDeps], path: str) -> str:
    """Read a file from the sandbox workspace.

    Use this to understand the current code before editing.

    Args:
        path: Relative path to the file (e.g. 'src/calculator.py')
    """
    if ctx.deps.sandbox_manager is None:
        return "Sandbox not available. Proceed without file reading."

    try:
        content: str = await ctx.deps.sandbox_manager.read_file(path)
        return content
    except Exception as exc:
        logger.warning("Sandbox read_file failed for %s: %s", path, exc)
        return f"Error reading {path}: {exc}"


# ── Tool: Sandbox Apply Diff ──────────────────────────────────────


@coder.tool
async def _sandbox_apply_diff(ctx: RunContext[CoderDeps], diff: str) -> str:
    """Apply a unified diff to files in the sandbox workspace.

    The diff is applied through the sandbox's apply_diff method,
    never via a host-side filesystem call (VAL-CODER-005).

    Args:
        diff: The unified diff to apply
    """
    if ctx.deps.sandbox_manager is None:
        return "Sandbox not available. Cannot apply diff."

    try:
        await ctx.deps.sandbox_manager.apply_diff(diff)
        return "Successfully applied diff"
    except Exception as exc:
        logger.warning("Sandbox apply_diff failed: %s", exc)
        return f"Error applying diff: {exc}"


# ── Tool: RAG Retrieval ────────────────────────────────────────────


@coder.tool
async def _rag_retrieval(ctx: RunContext[CoderDeps], query: str) -> str:
    """Search the repository code using semantic RAG retrieval.

    Returns relevant code chunks scoped to the current repo_url
    (VAL-CODER-004).  Use this to find the exact code that needs
    to be modified.

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


# ── JSON Parsing Helper ─────────────────────────────────────────────


def _extract_json_from_response(content: str) -> dict[str, Any]:
    """Extract JSON object from LLM response content.

    The LLM may return JSON wrapped in markdown code fences.
    This function strips fences and parses the JSON.
    """
    # Strip markdown code fences if present
    stripped = content.strip()
    if stripped.startswith("```"):
        # Remove opening fence (e.g. ```json or ```)
        first_newline = stripped.index("\n") if "\n" in stripped else len(stripped)
        stripped = stripped[first_newline + 1 :]
        # Remove closing fence
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()

    result: dict[str, Any] = json.loads(stripped)
    return result


# ── Persistence Helper ─────────────────────────────────────────────


def _compute_diff_hash(diff: str) -> str:
    """Compute SHA-256 hash of a diff for loop detection (VAL-CODER-006)."""
    return hashlib.sha256(diff.encode()).hexdigest()


async def persist_code_edit(
    *,
    store: Any,
    task_id: UUID,
    edit: CodeEdit,
    step_index: int = 1,
) -> None:
    """Persist a successfully parsed CodeEdit as a decision row.

    Writes to the decisions table with:
      - agent = 'coder'
      - decision_type = 'code_edit'
      - decision_data = full Pydantic JSON of the CodeEdit
        (includes diff_hash for loop detection / same-fix tracking)

    This is called by the orchestrator after a successful Coder run
    (VAL-CODER-006).
    """
    params = CreateDecisionParams(
        task_id=task_id,
        agent="coder",
        step_index=step_index,
        decision_type="code_edit",
        decision_data=edit.model_dump(mode="json"),
    )
    await store.create_decision(params)
    logger.info(
        "Persisted CodeEdit for task %s: touched_files=%s, diff_hash=%s",
        task_id,
        edit.touched_files,
        edit.diff_hash[:12] if edit.diff_hash else "(none)",
    )


# ── Prompt Builder with Cache Markers ──────────────────────────────


def _build_coder_prompt(
    change_plan: ChangePlan,
    review_result: ReviewResult | None = None,
    repo_context: str = "",
) -> StructuredPrompt:
    """Build a structured prompt with static and dynamic blocks.

    Static block (cacheable): system instructions + repo context.
    Dynamic block: current ChangePlan + optional prior ReviewResult.

    The static block is tagged with cache markers per OpenRouter
    protocol (§2.10).
    """
    # Static part: system instructions + repo context
    system_instructions = _SYSTEM_PROMPT

    # Dynamic part: change plan + review result
    dynamic_parts: list[str] = []
    dynamic_parts.append(
        f"## Change Plan\n{json.dumps(change_plan.model_dump(mode='json'), indent=2)}"
    )

    if review_result is not None:
        dynamic_parts.append(
            f"## Prior Review Result\n"
            f"Verdict: {review_result.verdict}\n"
            f"Issues: {json.dumps(review_result.issues, indent=2)}\n\n"
            "Address these issues in your new CodeEdit. "
            "Do NOT produce the same diff as before."
        )

    return build_structured_prompt(
        system_instructions=system_instructions,
        repo_context=repo_context,
        current_edit=dynamic_parts[0] if dynamic_parts else "",
        prior_review=dynamic_parts[1] if len(dynamic_parts) > 1 else "",
    )


# ── Convenience: Run Coder with Full Context ───────────────────────


async def run_coder(
    *,
    change_plan: ChangePlan,
    review_result: ReviewResult | None = None,
    sandbox_manager: Any | None = None,
    rag_retriever: Any | None = None,
    episodic_store: Any | None = None,
    task_id: UUID | None = None,
    trace_id: str = "",
    repo_url: str = "",
    repo_context: str = "",
) -> CoderRunResult:
    """Run the Coder agent with a ChangePlan and optional ReviewResult.

    This is the primary entry point for the orchestrator. It:
    1. Pre-reads relevant files from the sandbox (tool execution)
    2. Pre-fetches RAG context if a retriever is available
    3. Builds a structured prompt using ``_build_coder_prompt()``
       with cache markers on the static (repo context) block
    4. Calls ``LLMClient.chat_with_cache()`` so cache markers
       reach OpenRouter (§2.10)
    5. Parses the JSON response as ``CodeEdit``
    6. Applies the diff to the sandbox
    7. Computes diff_hash if not already set
    8. Persists the resulting CodeEdit as a decision row
    9. Returns ``CoderRunResult`` with token/cost metadata for
       accumulation by the orchestrator

    Args:
        change_plan: The ChangePlan from the Planner.
        review_result: Optional prior ReviewResult (for reject_with_changes).
        sandbox_manager: Sandbox manager for file operations.
        rag_retriever: RAG retriever for code search.
        episodic_store: Episodic store for decision persistence.
        task_id: The current task's UUID (for decision persistence).
        trace_id: Langfuse trace ID for span hierarchy.
        repo_url: Current repo URL for RAG scoping.
        repo_context: Pre-fetched repo context string.

    Returns:
        A ``CoderRunResult`` containing the CodeEdit and token/cost metadata.
    """
    if task_id is None:
        task_id = uuid4()

    # ── Pre-read relevant files from sandbox ────────────────────
    repo_context_parts: list[str] = []
    if repo_context:
        repo_context_parts.append(repo_context)

    if sandbox_manager is not None:
        for target_file in change_plan.target_files[:5]:
            try:
                content = await sandbox_manager.read_file(target_file)
                repo_context_parts.append(
                    f"### File: {target_file}\n```\n{content[:3000]}\n```"
                )
            except Exception:
                pass

    # ── Pre-fetch RAG context ──────────────────────────────────
    if rag_retriever is not None:
        try:
            from src.memory.semantic.store import SemanticStore

            if isinstance(rag_retriever, SemanticStore):
                rag_results: list[dict[str, Any]] = await rag_retriever.retrieve_dicts(
                    query=change_plan.rationale[:200],
                    repo_url=repo_url,
                    top_k=8,
                )
            else:
                rag_results = await rag_retriever.retrieve(
                    query=change_plan.rationale[:200],
                    repo_url=repo_url,
                    top_k=8,
                )
            for i, hit in enumerate(rag_results[:5], 1):
                file_path = hit.get("file_path", "unknown")
                chunk_text = hit.get("chunk_text", "")
                repo_context_parts.append(
                    f"[RAG {i}] {file_path}:\n{chunk_text[:1000]}"
                )
        except Exception as exc:
            logger.warning("RAG retrieval failed in run_coder: %s", exc)

    full_repo_context = "\n\n".join(repo_context_parts) if repo_context_parts else ""

    # ── Build structured prompt with cache markers ──────────────
    structured_prompt = _build_coder_prompt(
        change_plan=change_plan,
        review_result=review_result,
        repo_context=full_repo_context,
    )

    # ── Call LLM with cache markers via chat_with_cache ─────────
    llm_client = get_llm_client()
    llm_result = await llm_client.chat_with_cache(
        structured_prompt=structured_prompt,
        model=_CODER_MODEL,
        task_id=str(task_id),
        trace_id=trace_id,
        agent_name="coder",
        temperature=0.2,
    )

    # ── Parse JSON response as CodeEdit ────────────────────────
    try:
        parsed = _extract_json_from_response(llm_result.content)
        edit = CodeEdit.model_validate(parsed)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Coder LLM response could not be parsed as CodeEdit: %s. "
            "Falling back to PydanticAI agent.run()",
            exc,
        )
        # Fallback: use PydanticAI agent.run() with the dynamic block
        deps = CoderDeps(
            sandbox_manager=sandbox_manager,
            rag_retriever=rag_retriever,
            episodic_store=episodic_store,
            task_id=task_id,
            trace_id=trace_id,
            repo_url=repo_url,
        )
        user_prompt = _build_user_prompt(change_plan, review_result)
        result = await coder.run(user_prompt, deps=deps)
        edit = result.output
        # Extract usage from PydanticAI result if available
        if hasattr(result, "usage") and result.usage:
            llm_result.usage_input = getattr(result.usage, "request_tokens", 0) or 0
            llm_result.usage_output = getattr(result.usage, "response_tokens", 0) or 0

    # Compute diff_hash if not already set (VAL-CODER-006)
    if not edit.diff_hash:
        edit = CodeEdit(
            diff=edit.diff,
            touched_files=edit.touched_files,
            diff_hash=_compute_diff_hash(edit.diff),
        )

    # ── Apply diff to sandbox ──────────────────────────────────
    if sandbox_manager is not None:
        try:
            await sandbox_manager.apply_diff(edit.diff)
            logger.info("Applied Coder diff to sandbox for task %s", task_id)
        except Exception as exc:
            logger.warning("Sandbox apply_diff failed: %s", exc)

    # ── Persist the CodeEdit if a store is available ────────────
    if episodic_store is not None:
        await persist_code_edit(
            store=episodic_store,
            task_id=task_id,
            edit=edit,
            step_index=1,
        )

    # ── Compute cost from token usage ──────────────────────────
    cost_usd = (
        llm_result.cost_usd
        if llm_result.cost_usd > Decimal("0")
        else estimate_cost_tiktoken(
            model=_CODER_MODEL,
            prompt_tokens=llm_result.usage_input,
            completion_tokens=llm_result.usage_output,
        )
    )

    return CoderRunResult(
        edit=edit,
        tokens_in=llm_result.usage_input,
        tokens_out=llm_result.usage_output,
        cached_tokens=llm_result.cached_tokens,
        cost_usd=cost_usd,
    )


def _build_user_prompt(
    change_plan: ChangePlan,
    review_result: ReviewResult | None = None,
) -> str:
    """Build the user prompt for the Coder from ChangePlan and ReviewResult."""
    parts: list[str] = []

    # ChangePlan
    parts.append("## Change Plan")
    parts.append(f"Target files: {change_plan.target_files}")
    parts.append(f"Rationale: {change_plan.rationale}")
    parts.append(f"Approach: {change_plan.approach}")
    if change_plan.estimated_complexity:
        parts.append(f"Complexity: {change_plan.estimated_complexity}")
    parts.append("")

    # Prior ReviewResult (if present)
    if review_result is not None:
        parts.append("## Prior Review Result")
        parts.append(f"Verdict: {review_result.verdict}")
        if review_result.issues:
            parts.append("Issues to address:")
            for issue in review_result.issues:
                parts.append(f"  - {issue}")
        parts.append(
            "\nIMPORTANT: Address the above issues in your new CodeEdit. "
            "Do NOT produce the same diff as the previously rejected edit."
        )
        parts.append("")

    parts.append(
        "Produce a CodeEdit with a unified diff that implements the changes "
        "described in the Change Plan. Use the sandbox_read_file tool to "
        "read files before editing. Use the rag_retrieval tool if you need "
        "more code context. Use the sandbox_apply_diff tool to apply your "
        "diff in the sandbox."
    )

    return "\n".join(parts)
