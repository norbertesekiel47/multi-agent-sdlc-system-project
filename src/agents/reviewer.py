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

Architecture reference: §2.3 Specialized Agents.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID, uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.agents.models import CodeEdit, ReviewResult
from src.llm.caching import StructuredPrompt, build_structured_prompt
from src.memory.episodic.models import CreateDecisionParams

if TYPE_CHECKING:
    pass

load_dotenv()

logger = logging.getLogger(__name__)

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
) -> ReviewResult:
    """Run the Reviewer agent with a CodeEdit.

    This is the primary entry point for the orchestrator. It:
    1. Builds ReviewerDeps with the provided sandbox/store
    2. Constructs the user prompt from the CodeEdit
    3. Runs the Reviewer agent
    4. Sets diff_hash on the ReviewResult to match the CodeEdit
    5. Persists the resulting ReviewResult as a decision row

    Args:
        code_edit: The CodeEdit to review.
        sandbox_manager: Sandbox manager for file read and static analysis.
        episodic_store: Episodic store for decision persistence.
        task_id: The current task's UUID (for decision persistence).
        trace_id: Langfuse trace ID for span hierarchy.
        repo_url: Current repo URL for context scoping.
        repo_context: Pre-fetched repo context string.

    Returns:
        A Pydantic-valid ReviewResult with diff_hash matching the CodeEdit.
    """
    if task_id is None:
        task_id = uuid4()

    deps = ReviewerDeps(
        sandbox_manager=sandbox_manager,
        episodic_store=episodic_store,
        task_id=task_id,
        trace_id=trace_id,
        repo_url=repo_url,
    )

    # Build the user prompt
    user_prompt = _build_user_prompt(code_edit)

    result = await reviewer.run(user_prompt, deps=deps)
    review: ReviewResult = result.output

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

    return review


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
