"""Reviewer agent (PydanticAI) on DeepSeek V4 Flash via OpenRouter.

Typed input: CodeEdit + the diff being reviewed.
Typed output: ReviewResult (verdict: accept|reject_with_changes|reject, issues list).
Free-form text outside the typed schema is rejected by the output parser.

Tools:
  - sandbox file read
  - static-analysis runner (ruff/mypy via sandbox run_command)
  - security-pattern scanner

Same prompt cache markers as Coder (§2.10).
Locked to deepseek/deepseek-chat-v3-0324 model ID (VAL-REVIEWER-008).

The ``run_reviewer`` function uses ``LLMClient.chat_with_cache()``
with ``_build_reviewer_prompt()`` so that the StructuredPrompt
with cache markers reaches OpenRouter.  Token counts (including
cached_tokens) are extracted from the LLM response and returned
via ``ReviewerRunResult`` for accumulation by the orchestrator.

Architecture reference: §2.3 Specialized Agents.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID, uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.agents.models import CodeEdit, ReviewResult
from src.llm.caching import StructuredPrompt, build_structured_prompt
from src.llm.client import get_llm_client
from src.llm.cost import estimate_cost_tiktoken
from src.memory.episodic.models import CreateDecisionParams

if TYPE_CHECKING:
    pass

load_dotenv()

logger = logging.getLogger(__name__)


# ── Reviewer Run Result ─────────────────────────────────────────────


@dataclass
class ReviewerRunResult:
    """Result of a Reviewer agent run, including token and cost metadata.

    The orchestrator extracts ``tokens_in``, ``tokens_out``, and
    ``cached_tokens`` from this result and accumulates them into
    the task's ``total_tokens_in/out/cached`` state fields.
    """

    review: ReviewResult
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    cost_usd: Decimal = Decimal("0")


# ── Model ID (VAL-REVIEWER-008) ──────────────────────────────────
# Locked to deepseek/deepseek-chat-v3-0324 via OpenRouter.
# NOTE: deepseek-v4-flash is aspirational but unstable on OpenRouter;
# deepseek-chat-v3-0324 is the stable fallback per AGENTS.md.
_REVIEWER_MODEL: str = "deepseek/deepseek-chat-v3-0324"

# PydanticAI uses "openrouter:<model_id>" format for OpenRouter models.
_PYDANTIC_AI_MODEL: str = f"openrouter:{_REVIEWER_MODEL}"


# ── Episodic Store Protocol ────────────────────────────────────────


class EpisodicStoreProtocol(Protocol):
    """Minimal protocol for episodic store operations needed by the Reviewer.

    Only create_decision is exposed (reads go through the orchestrator).
    """

    async def create_decision(
        self,
        params: CreateDecisionParams,
    ) -> Any: ...

    async def close(self) -> None: ...


# ── Reviewer Dependencies ──────────────────────────────────────────


class ReviewerDeps(BaseModel):
    """Dependencies injected into the Reviewer agent at runtime.

    These are NOT passed to the LLM — they are available to tool
    implementations via RunContext.deps.
    """

    model_config = {"arbitrary_types_allowed": True}

    sandbox_manager: Any | None = Field(
        default=None,
        description="Sandbox manager for file read and command execution",
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
        description="Current repo URL for context scoping",
    )


# ── System Prompt (static block for caching) ──────────────────────

_SYSTEM_PROMPT = """\
You are the Reviewer agent in an autonomous multi-agent SDLC system.
Your job is to review a code edit (unified diff) produced by the Coder
and produce a structured review result.

Given:
- A CodeEdit with a unified diff and list of touched files
- The diff content to review

You MUST produce a ReviewResult with:
1. verdict: One of exactly three values:
   - "accept": The diff is correct, complete, and safe to merge
   - "reject_with_changes": The diff has issues that should be fixed
     (the Coder will retry with this feedback)
   - "reject": The diff is fundamentally flawed and should not be retried
2. issues: A list of specific problems found (empty for accept)

RULES:
- Your output MUST be a valid ReviewResult. Do NOT include free-form text.
- The verdict MUST be exactly one of: accept, reject_with_changes, reject.
  No other values are allowed (e.g., NOT "lgtm", "approved", "yes").
- Use the sandbox_read_file tool to read the modified files before reviewing.
- Use the run_static_analysis tool to run ruff or mypy on the changed files.
- Use the security_pattern_scan tool to check for vulnerability patterns.
- For "accept", issues should be empty or contain only minor suggestions.
- For "reject_with_changes", issues must list specific problems to fix.
- For "reject", explain why the approach is fundamentally flawed.
- Always run static analysis before making your verdict.
- Consider: correctness, style, security, performance, edge cases.
"""


# ── Security patterns to scan for ──────────────────────────────────

_SECURITY_PATTERNS: list[tuple[str, str]] = [
    (
        r"sk-[a-zA-Z0-9]{20,}",
        "Hardcoded API key pattern detected (sk-* prefix)",
    ),
    (
        r"github_pat_[a-zA-Z0-9]{20,}",
        "Hardcoded GitHub PAT pattern detected",
    ),
    (
        r"gho_[a-zA-Z0-9]{20,}",
        "Hardcoded GitHub OAuth token pattern detected",
    ),
    (
        r"or-[a-zA-Z0-9]{20,}",
        "Hardcoded OpenRouter key pattern detected",
    ),
    (
        r"rm\s+-rf\s+/(?!workspace)",
        "Destructive rm -rf outside workspace detected",
    ),
    (
        r"eval\s*\(",
        "eval() usage detected (potential code injection)",
    ),
    (
        r"subprocess\.call\s*\([^)]*shell\s*=\s*True",
        "subprocess with shell=True detected (command injection risk)",
    ),
    (
        r"SELECT\s+\*\s+FROM\s+\w+\s*;",
        "Unparameterized SQL query detected (SQL injection risk)",
    ),
]


# ── Create the Agent ───────────────────────────────────────────────

reviewer = Agent(
    model=_PYDANTIC_AI_MODEL,
    output_type=ReviewResult,
    name="reviewer",
    deps_type=ReviewerDeps,
    system_prompt=_SYSTEM_PROMPT,
    retries=3,  # Allow PydanticAI to retry on validation failure
)


# ── Tool: Sandbox File Read ────────────────────────────────────────


@reviewer.tool
async def _sandbox_read_file(ctx: RunContext[ReviewerDeps], path: str) -> str:
    """Read a file from the sandbox workspace.

    Use this to inspect the modified files after the Coder's diff is applied.

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


# ── Tool: Static Analysis Runner ───────────────────────────────────


@reviewer.tool
async def _run_static_analysis(
    ctx: RunContext[ReviewerDeps], command: str,
) -> str:
    """Run a static analysis tool in the sandbox (VAL-REVIEWER-003).

    Use this to run ruff check, mypy, or other linters on the
    changed files before making your review verdict.

    Args:
        command: The static analysis command to run
                 (e.g., 'ruff check src/', 'mypy src/calculator.py')
    """
    if ctx.deps.sandbox_manager is None:
        return "Sandbox not available. Cannot run static analysis."

    try:
        output: str = await ctx.deps.sandbox_manager.run_command(command)
        if not output.strip():
            return "Static analysis passed: no issues found."
        return output
    except Exception as exc:
        # Static analysis tool not found or error — not a blocker
        logger.warning("Static analysis command failed: %s", exc)
        return f"Static analysis error: {exc}. Proceed without analysis."


# ── Tool: Security Pattern Scanner ─────────────────────────────────


@reviewer.tool
async def _security_pattern_scan(
    ctx: RunContext[ReviewerDeps], file_path: str,
) -> str:
    """Scan a file for common security vulnerability patterns.

    Checks for hardcoded secrets, command injection, SQL injection,
    and other patterns that should not be in the diff.

    Args:
        file_path: Relative path to the file to scan
                 (e.g., 'src/config.py')
    """
    if ctx.deps.sandbox_manager is None:
        return "Sandbox not available. Cannot scan for security patterns."

    try:
        content: str = await ctx.deps.sandbox_manager.read_file(file_path)
    except Exception as exc:
        return f"Error reading {file_path} for security scan: {exc}"

    findings: list[str] = []
    for pattern, description in _SECURITY_PATTERNS:
        matches = re.findall(pattern, content)
        if matches:
            findings.append(f"⚠ {description} (found {len(matches)} occurrence(s))")

    if not findings:
        return f"Security scan of {file_path}: no known vulnerability patterns found."

    return f"Security scan of {file_path}:\n" + "\n".join(findings)


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


async def persist_review_result(
    *,
    store: Any,
    task_id: UUID,
    review: ReviewResult,
    step_index: int = 2,
) -> None:
    """Persist a successfully parsed ReviewResult as a decision row.

    Writes to the decisions table with:
      - agent = 'reviewer'
      - decision_type = 'review_verdict'
      - decision_data = full Pydantic JSON of the ReviewResult

    This is called by the orchestrator after a successful Reviewer run
    (VAL-REVIEWER-007).
    """
    params = CreateDecisionParams(
        task_id=task_id,
        agent="reviewer",
        step_index=step_index,
        decision_type="review_verdict",
        decision_data=review.model_dump(mode="json"),
    )
    await store.create_decision(params)
    logger.info(
        "Persisted ReviewResult for task %s: verdict=%s, issues=%d",
        task_id,
        review.verdict,
        len(review.issues),
    )


# ── Prompt Builder with Cache Markers ──────────────────────────────


def _build_reviewer_prompt(
    code_edit: CodeEdit,
    repo_context: str = "",
) -> StructuredPrompt:
    """Build a structured prompt with static and dynamic blocks.

    Static block (cacheable): system instructions + repo context.
    Dynamic block: current CodeEdit diff.

    Same cache markers as Coder (§2.10).
    """
    system_instructions = _SYSTEM_PROMPT

    dynamic_parts: list[str] = []
    dynamic_parts.append(
        f"## Code Edit to Review\n"
        f"Touched files: {code_edit.touched_files}\n"
        f"Diff hash: {code_edit.diff_hash}\n\n"
        f"```diff\n{code_edit.diff}\n```"
    )

    return build_structured_prompt(
        system_instructions=system_instructions,
        repo_context=repo_context,
        current_edit=dynamic_parts[0] if dynamic_parts else "",
    )


# ── Convenience: Run Reviewer with Full Context ────────────────────


async def run_reviewer(
    *,
    code_edit: CodeEdit,
    sandbox_manager: Any | None = None,
    episodic_store: Any | None = None,
    task_id: UUID | None = None,
    trace_id: str = "",
    repo_url: str = "",
    repo_context: str = "",
) -> ReviewerRunResult:
    """Run the Reviewer agent with a CodeEdit.

    This is the primary entry point for the orchestrator. It:
    1. Pre-reads modified files from the sandbox (tool execution)
    2. Runs static analysis (ruff/mypy) in the sandbox
    3. Runs security pattern scanning on touched files
    4. Builds a structured prompt using ``_build_reviewer_prompt()``
       with cache markers on the static (repo context) block
    5. Calls ``LLMClient.chat_with_cache()`` so cache markers
       reach OpenRouter (§2.10)
    6. Parses the JSON response as ``ReviewResult``
    7. Sets diff_hash to match the CodeEdit being reviewed
    8. Persists the resulting ReviewResult as a decision row
    9. Returns ``ReviewerRunResult`` with token/cost metadata for
       accumulation by the orchestrator

    Args:
        code_edit: The CodeEdit to review.
        sandbox_manager: Sandbox manager for file read and static analysis.
        episodic_store: Episodic store for decision persistence.
        task_id: The current task's UUID (for decision persistence).
        trace_id: Langfuse trace ID for span hierarchy.
        repo_url: Current repo URL for context scoping.
        repo_context: Pre-fetched repo context string.

    Returns:
        A ``ReviewerRunResult`` containing the ReviewResult and token/cost metadata.
    """
    if task_id is None:
        task_id = uuid4()

    # ── Pre-read modified files from sandbox ────────────────────
    repo_context_parts: list[str] = []
    if repo_context:
        repo_context_parts.append(repo_context)

    if sandbox_manager is not None:
        for touched_file in code_edit.touched_files[:5]:
            try:
                content = await sandbox_manager.read_file(touched_file)
                repo_context_parts.append(
                    f"### File: {touched_file}\n```\n{content[:3000]}\n```"
                )
            except Exception:
                pass

    # ── Run static analysis (ruff/mypy) ─────────────────────────
    static_analysis_results: list[str] = []
    if sandbox_manager is not None:
        for touched_file in code_edit.touched_files[:3]:
            try:
                output = await sandbox_manager.run_command(
                    f"ruff check {touched_file} 2>&1 || true"
                )
                if output.strip():
                    static_analysis_results.append(
                        f"Ruff check of {touched_file}:\n{output[:1000]}"
                    )
            except Exception:
                pass
            try:
                output = await sandbox_manager.run_command(
                    f"mypy {touched_file} 2>&1 || true"
                )
                if output.strip():
                    static_analysis_results.append(
                        f"Mypy check of {touched_file}:\n{output[:1000]}"
                    )
            except Exception:
                pass

    # ── Run security pattern scanning ───────────────────────────
    security_results: list[str] = []
    if sandbox_manager is not None:
        for touched_file in code_edit.touched_files[:5]:
            try:
                file_content = await sandbox_manager.read_file(touched_file)
                for pattern, description in _SECURITY_PATTERNS:
                    matches = re.findall(pattern, file_content)
                    if matches:
                        security_results.append(
                            f"⚠ {description} in {touched_file} "
                            f"({len(matches)} occurrence(s))"
                        )
            except Exception:
                pass

    # Combine repo context with analysis results
    if static_analysis_results:
        repo_context_parts.append(
            "## Static Analysis Results\n" + "\n".join(static_analysis_results)
        )
    if security_results:
        repo_context_parts.append(
            "## Security Scan Results\n" + "\n".join(security_results)
        )

    full_repo_context = "\n\n".join(repo_context_parts) if repo_context_parts else ""

    # ── Build structured prompt with cache markers ──────────────
    structured_prompt = _build_reviewer_prompt(
        code_edit=code_edit,
        repo_context=full_repo_context,
    )

    # ── Call LLM with cache markers via chat_with_cache ─────────
    llm_client = get_llm_client()
    llm_result = await llm_client.chat_with_cache(
        structured_prompt=structured_prompt,
        model=_REVIEWER_MODEL,
        task_id=str(task_id),
        trace_id=trace_id,
        agent_name="reviewer",
        temperature=0.2,
    )

    # ── Parse JSON response as ReviewResult ─────────────────────
    try:
        parsed = _extract_json_from_response(llm_result.content)
        review = ReviewResult.model_validate(parsed)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Reviewer LLM response could not be parsed as ReviewResult: %s. "
            "Falling back to PydanticAI agent.run()",
            exc,
        )
        # Fallback: use PydanticAI agent.run() with the dynamic block
        deps = ReviewerDeps(
            sandbox_manager=sandbox_manager,
            episodic_store=episodic_store,
            task_id=task_id,
            trace_id=trace_id,
            repo_url=repo_url,
        )
        user_prompt = _build_user_prompt(code_edit)
        result = await reviewer.run(user_prompt, deps=deps)
        review = result.output
        # Extract usage from PydanticAI result if available
        if hasattr(result, "usage") and result.usage:
            llm_result.usage_input = getattr(result.usage, "request_tokens", 0) or 0
            llm_result.usage_output = getattr(result.usage, "response_tokens", 0) or 0

    # Ensure diff_hash matches the CodeEdit being reviewed
    if not review.diff_hash and code_edit.diff_hash:
        review = ReviewResult(
            verdict=review.verdict,
            issues=review.issues,
            diff_hash=code_edit.diff_hash,
        )

    # Persist the ReviewResult if a store is available
    if episodic_store is not None:
        await persist_review_result(
            store=episodic_store,
            task_id=task_id,
            review=review,
            step_index=2,
        )

    # ── Compute cost from token usage ──────────────────────────
    cost_usd = (
        llm_result.cost_usd
        if llm_result.cost_usd > Decimal("0")
        else estimate_cost_tiktoken(
            model=_REVIEWER_MODEL,
            prompt_tokens=llm_result.usage_input,
            completion_tokens=llm_result.usage_output,
        )
    )

    return ReviewerRunResult(
        review=review,
        tokens_in=llm_result.usage_input,
        tokens_out=llm_result.usage_output,
        cached_tokens=llm_result.cached_tokens,
        cost_usd=cost_usd,
    )


def _build_user_prompt(code_edit: CodeEdit) -> str:
    """Build the user prompt for the Reviewer from a CodeEdit."""
    parts: list[str] = []

    # Code edit to review
    parts.append("## Code Edit to Review")
    parts.append(f"Touched files: {code_edit.touched_files}")
    if code_edit.diff_hash:
        parts.append(f"Diff hash: {code_edit.diff_hash}")
    parts.append("")
    parts.append("### Unified Diff")
    parts.append(f"```diff\n{code_edit.diff}\n```")
    parts.append("")

    parts.append(
        "Review this code edit and produce a ReviewResult. "
        "Use the sandbox_read_file tool to inspect the modified files. "
        "Use the run_static_analysis tool to run ruff or mypy. "
        "Use the security_pattern_scan tool to check for vulnerabilities. "
        "Your verdict must be one of: accept, reject_with_changes, reject."
    )

    return "\n".join(parts)
