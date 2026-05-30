"""QA agent (PydanticAI) on DeepSeek V4 Flash via OpenRouter.

Typed input: CodeEdit (post-review).
Typed output: TestReport (passed, failed, failed_test_names, generated_test_files).
Free-form text outside the typed schema is rejected by the output parser.

Tools:
  - sandbox file write (for generated tests)
  - sandbox test runner (run_tests)

Locked to deepseek/deepseek-chat-v3-0324 model ID (VAL-QA-001, VAL-QA-006).

The ``run_qa`` function uses ``LLMClient.chat_with_cache()``
with ``_build_qa_prompt()`` so that the StructuredPrompt
with cache markers reaches OpenRouter.  Token counts (including
cached_tokens) are extracted from the LLM response and returned
via ``QARunResult`` for accumulation by the orchestrator.

Architecture reference: §2.3 Specialized Agents — QA Agent.
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
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, RunContext

from src.agents.models import CodeEdit, TestReport
from src.llm.caching import StructuredPrompt, build_structured_prompt
from src.llm.client import get_llm_client, get_temperature
from src.llm.cost import estimate_cost_tiktoken
from src.memory.episodic.models import CreateDecisionParams

if TYPE_CHECKING:
    pass

load_dotenv()

logger = logging.getLogger(__name__)


# ── QA Run Result ──────────────────────────────────────────────────


@dataclass
class QARunResult:
    """Result of a QA agent run, including token and cost metadata.

    The orchestrator extracts ``tokens_in``, ``tokens_out``, and
    ``cached_tokens`` from this result and accumulates them into
    the task's ``total_tokens_in/out/cached`` state fields.
    """

    report: TestReport
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    cost_usd: Decimal = Decimal("0")


# ── Model ID (VAL-QA-001, VAL-QA-006) ─────────────────────────────
# Locked to deepseek/deepseek-chat-v3-0324 via OpenRouter.
# NOTE: deepseek-v4-flash is aspirational but unstable on OpenRouter;
# deepseek-chat-v3-0324 is the stable fallback per AGENTS.md.
_QA_MODEL: str = "deepseek/deepseek-chat-v3-0324"

# PydanticAI uses "openrouter:<model_id>" format for OpenRouter models.
_PYDANTIC_AI_MODEL: str = f"openrouter:{_QA_MODEL}"


# ── Episodic Store Protocol ────────────────────────────────────────


class EpisodicStoreProtocol(Protocol):
    """Minimal protocol for episodic store operations needed by the QA agent.

    Only create_decision is exposed (reads go through the orchestrator).
    """

    async def create_decision(
        self,
        params: CreateDecisionParams,
    ) -> Any: ...

    async def close(self) -> None: ...


# ── QA Dependencies ───────────────────────────────────────────────


class QADeps(BaseModel):
    """Dependencies injected into the QA agent at runtime.

    These are NOT passed to the LLM — they are available to tool
    implementations via RunContext.deps.
    """

    model_config = {"arbitrary_types_allowed": True}

    sandbox_manager: Any | None = Field(
        default=None,
        description="Sandbox manager for file write and test execution",
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
You are the QA agent in an autonomous multi-agent SDLC system.
Your job is to generate and run tests for a code edit (unified diff)
that has been reviewed and accepted by the Reviewer.

Given:
- A CodeEdit with a unified diff and list of touched files
- Access to the sandbox where the code has been applied

You MUST produce a TestReport with:
1. passed: Number of passing tests (integer >= 0)
2. failed: Number of failing tests (integer >= 0)
3. failed_test_names: List of failing test names (MUST have exactly `failed` entries)
4. generated_test_files: List of test files you created

RULES:
- Your output MUST be a valid TestReport. Do NOT include free-form text.
- Include the FULL content of each generated test file in the
  ``generated_test_contents`` field (map of file_path → source code).
  This is CRITICAL: if you cannot use sandbox_write_file (e.g. when
  called via chat_with_cache without tool access), the orchestrator
  will write files from generated_test_contents to the sandbox.
- Also list each file path in ``generated_test_files`` (for traceability).
- If you DO have tool access, use sandbox_write_file to write test files
  into the sandbox AND populate generated_test_contents as a backup.
- Use the sandbox_run_tests tool to execute the test suite.
- The number of entries in failed_test_names MUST exactly equal the `failed` count.
  For example, if failed=2, failed_test_names must have exactly 2 entries.
  If failed=0, failed_test_names must be empty [].
- Always write test files BEFORE running tests.
- Generate tests that cover the modified functionality.
- Focus on edge cases, error handling, and the specific bug fix.
- If tests fail, report the exact failure names — do not omit them.
- Prefer pytest-style tests with descriptive names.
"""


# ── Create the Agent ───────────────────────────────────────────────

qa = Agent(
    model=_PYDANTIC_AI_MODEL,
    output_type=TestReport,
    name="qa",
    deps_type=QADeps,
    system_prompt=_SYSTEM_PROMPT,
    retries=3,  # Allow PydanticAI to retry on validation failure
)


# ── Tool: Sandbox File Write ───────────────────────────────────────


@qa.tool
async def _sandbox_write_file(
    ctx: RunContext[QADeps], path: str, content: str,
) -> str:
    """Write a test file to the sandbox workspace (VAL-QA-002).

    Use this to create test files that will be executed by
    sandbox_run_tests.

    Args:
        path: Relative path for the test file
              (e.g., 'tests/test_calculator.py')
        content: The test file content
    """
    if ctx.deps.sandbox_manager is None:
        return "Sandbox not available. Cannot write test file."

    try:
        await ctx.deps.sandbox_manager.write_file(path, content)
        return f"Successfully written test file: {path}"
    except Exception as exc:
        logger.warning("Sandbox write_file failed for %s: %s", path, exc)
        return f"Error writing {path}: {exc}"


# ── Tool: Sandbox Test Runner ─────────────────────────────────────


@qa.tool
async def _sandbox_run_tests(
    ctx: RunContext[QADeps], command: str = "pytest",
) -> str:
    """Run the test suite in the sandbox (VAL-QA-003).

    Use this to execute the test suite after writing test files.
    Returns the test output with pass/fail summary.

    Args:
        command: The test command to run (default: 'pytest')
    """
    if ctx.deps.sandbox_manager is None:
        return "Sandbox not available. Cannot run tests."

    try:
        output: str = await ctx.deps.sandbox_manager.run_tests(command)
        if not output.strip():
            return "Test run produced no output."
        return output
    except Exception as exc:
        logger.warning("Sandbox run_tests failed: %s", exc)
        return f"Error running tests: {exc}"


# ── Tool: Sandbox File Read ────────────────────────────────────────


@qa.tool
async def _sandbox_read_file(ctx: RunContext[QADeps], path: str) -> str:
    """Read a file from the sandbox workspace.

    Use this to inspect the source code before writing tests.

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


# ── Test Output Parser ──────────────────────────────────────────────


def _parse_test_output(output: str) -> tuple[int, int, list[str]]:
    """Parse pytest output to extract pass/fail counts and failed test names.

    Returns (passed, failed, failed_test_names).

    Handles common pytest output formats:
    - "3 passed, 1 failed in 0.5s"
    - "5 passed in 0.3s"
    - "FAILED test_name - AssertionError"
    - "1 failed, 3 passed"
    """
    passed = 0
    failed = 0
    failed_test_names: list[str] = []

    # Extract pass count
    pass_match = re.search(r"(\d+)\s+passed", output)
    if pass_match:
        passed = int(pass_match.group(1))

    # Extract fail count
    fail_match = re.search(r"(\d+)\s+failed", output)
    if fail_match:
        failed = int(fail_match.group(1))

    # Extract failed test names from FAILED lines
    for line in output.split("\n"):
        line = line.strip()
        # Match lines like "FAILED tests/test_foo.py::test_bar - AssertionError"
        # or "FAILED test_foo"
        if line.startswith("FAILED "):
            # Extract the test name (after "FAILED " and before " -" or end of line)
            test_name_part = line[7:]  # Remove "FAILED "
            # Remove the trailing error message if present
            if " - " in test_name_part:
                test_name_part = test_name_part[: test_name_part.index(" - ")]
            # Remove file path prefix if present (keep ::test_name)
            test_name = (
                test_name_part.split("::")[-1]
                if "::" in test_name_part
                else test_name_part
            )
            failed_test_names.append(test_name)

    return passed, failed, failed_test_names


# ── Persistence Helper ─────────────────────────────────────────────


async def persist_test_report(
    *,
    store: Any,
    task_id: UUID,
    report: TestReport,
    step_index: int = 3,
) -> None:
    """Persist a successfully parsed TestReport as a decision row.

    Writes to the decisions table with:
      - agent = 'qa'
      - decision_type = 'test_report'
      - decision_data = full Pydantic JSON of the TestReport

    This is called by the orchestrator after a successful QA run
    (VAL-QA-004).
    """
    params = CreateDecisionParams(
        task_id=task_id,
        agent="qa",
        step_index=step_index,
        decision_type="test_report",
        decision_data=report.model_dump(mode="json"),
    )
    await store.create_decision(params)
    logger.info(
        "Persisted TestReport for task %s: passed=%d, failed=%d",
        task_id,
        report.passed,
        report.failed,
    )


# ── Prompt Builder with Cache Markers ──────────────────────────────


def _build_qa_prompt(
    code_edit: CodeEdit,
    repo_context: str = "",
) -> StructuredPrompt:
    """Build a structured prompt with static and dynamic blocks.

    Static block (cacheable): system instructions + repo context.
    Dynamic block: current CodeEdit diff + touched files.

    Same cache marker pattern as Coder/Reviewer (§2.10).
    """
    system_instructions = _SYSTEM_PROMPT

    dynamic_parts: list[str] = []
    dynamic_parts.append(
        f"## Code Edit to Test\n"
        f"Touched files: {code_edit.touched_files}\n"
        f"Diff hash: {code_edit.diff_hash}\n\n"
        f"```diff\n{code_edit.diff}\n```"
    )

    return build_structured_prompt(
        system_instructions=system_instructions,
        repo_context=repo_context,
        current_edit=dynamic_parts[0] if dynamic_parts else "",
    )


# ── Convenience: Run QA with Full Context ─────────────────────────


async def run_qa(
    *,
    code_edit: CodeEdit,
    sandbox_manager: Any | None = None,
    episodic_store: Any | None = None,
    task_id: UUID | None = None,
    trace_id: str = "",
    repo_url: str = "",
    repo_context: str = "",
) -> QARunResult:
    """Run the QA agent with a CodeEdit.

    This is the primary entry point for the orchestrator. It:
    1. Pre-reads modified files from the sandbox (tool execution)
    2. Builds a structured prompt using ``_build_qa_prompt()``
       with cache markers on the static (repo context) block
    3. Calls ``LLMClient.chat_with_cache()`` so cache markers
       reach OpenRouter (§2.10)
    4. Parses the JSON response as ``TestReport``
    5. Writes generated test files to the sandbox
    6. Runs tests in the sandbox and parses the output
    7. Reconciles the TestReport with actual test output
    8. Persists the resulting TestReport as a decision row
    9. Returns ``QARunResult`` with token/cost metadata for
       accumulation by the orchestrator

    Args:
        code_edit: The CodeEdit to test.
        sandbox_manager: Sandbox manager for file write and test execution.
        episodic_store: Episodic store for decision persistence.
        task_id: The current task's UUID (for decision persistence).
        trace_id: Langfuse trace ID for span hierarchy.
        repo_url: Current repo URL for context scoping.
        repo_context: Pre-fetched repo context string.

    Returns:
        A ``QARunResult`` containing the TestReport and token/cost metadata.
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
                logger.debug(
                    "repo-context read failed for %s; skipping", touched_file, exc_info=True
                )

    full_repo_context = "\n\n".join(repo_context_parts) if repo_context_parts else ""

    # ── Build structured prompt with cache markers ──────────────
    structured_prompt = _build_qa_prompt(
        code_edit=code_edit,
        repo_context=full_repo_context,
    )

    # ── Call LLM with cache markers via chat_with_cache ─────────
    llm_client = get_llm_client()
    llm_result = await llm_client.chat_with_cache(
        structured_prompt=structured_prompt,
        model=_QA_MODEL,
        task_id=str(task_id),
        trace_id=trace_id,
        agent_name="qa",
        temperature=get_temperature(),
    )

    # ── Parse JSON response as TestReport ──────────────────────
    try:
        parsed = _extract_json_from_response(llm_result.content)
        report = TestReport.model_validate(parsed)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "QA LLM response could not be parsed as TestReport: %s. "
            "Falling back to PydanticAI agent.run()",
            exc,
        )
        # Fallback: use PydanticAI agent.run() with the dynamic block
        deps = QADeps(
            sandbox_manager=sandbox_manager,
            episodic_store=episodic_store,
            task_id=task_id,
            trace_id=trace_id,
            repo_url=repo_url,
        )
        user_prompt = _build_user_prompt(code_edit)
        result = await qa.run(user_prompt, deps=deps)
        report = result.output
        # Extract usage from PydanticAI result if available
        if hasattr(result, "usage") and result.usage:
            llm_result.usage_input = getattr(result.usage, "request_tokens", 0) or 0
            llm_result.usage_output = getattr(result.usage, "response_tokens", 0) or 0

    # ── Write generated test files to sandbox ───────────────────
    # When chat_with_cache is used, the LLM cannot invoke tools, so
    # test file content is carried in generated_test_contents.
    # The orchestrator writes each entry via sandbox_write_file.
    if sandbox_manager is not None and report.generated_test_contents:
        for test_file_path, test_file_content in report.generated_test_contents.items():
            try:
                await sandbox_manager.write_file(test_file_path, test_file_content)
                logger.info(
                    "Wrote generated test file from generated_test_contents: %s",
                    test_file_path,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to write generated test file %s: %s", test_file_path, exc,
                )
        # Ensure generated_test_files list is consistent with generated_test_contents
        content_paths = set(report.generated_test_contents.keys())
        listed_paths = set(report.generated_test_files)
        missing_from_list = content_paths - listed_paths
        if missing_from_list:
            # Add any paths that are in contents but missing from the list
            try:
                report = TestReport(
                    passed=report.passed,
                    failed=report.failed,
                    failed_test_names=report.failed_test_names,
                    generated_test_files=report.generated_test_files + list(missing_from_list),
                    generated_test_contents=report.generated_test_contents,
                )
            except ValidationError:
                logger.warning(
                    "Could not reconcile generated_test_files with generated_test_contents"
                )

    # ── Run tests in sandbox and reconcile ──────────────────────
    if sandbox_manager is not None:
        try:
            test_output = await sandbox_manager.run_tests()
            actual_passed, actual_failed, actual_failed_names = _parse_test_output(
                test_output,
            )
            # Override the LLM-reported counts with actual test output
            # This ensures the TestReport reflects reality (VAL-QA-003)
            try:
                report = TestReport(
                    passed=actual_passed,
                    failed=actual_failed,
                    failed_test_names=actual_failed_names,
                    generated_test_files=report.generated_test_files,
                    generated_test_contents=report.generated_test_contents,
                )
            except ValidationError:
                # If actual counts don't validate, keep the LLM report
                logger.warning(
                    "Actual test output could not be reconciled into TestReport. "
                    "Keeping LLM-reported counts."
                )
        except Exception as exc:
            logger.warning("Sandbox run_tests failed: %s. Using LLM-reported counts.", exc)

    # Persist the TestReport if a store is available
    if episodic_store is not None:
        await persist_test_report(
            store=episodic_store,
            task_id=task_id,
            report=report,
            step_index=3,
        )

    # ── Compute cost from token usage ──────────────────────────
    cost_usd = (
        llm_result.cost_usd
        if llm_result.cost_usd > Decimal("0")
        else estimate_cost_tiktoken(
            model=_QA_MODEL,
            prompt_tokens=llm_result.usage_input,
            completion_tokens=llm_result.usage_output,
        )
    )

    return QARunResult(
        report=report,
        tokens_in=llm_result.usage_input,
        tokens_out=llm_result.usage_output,
        cached_tokens=llm_result.cached_tokens,
        cost_usd=cost_usd,
    )


def _build_user_prompt(code_edit: CodeEdit) -> str:
    """Build the user prompt for the QA agent from a CodeEdit."""
    parts: list[str] = []

    # Code edit to test
    parts.append("## Code Edit to Test")
    parts.append(f"Touched files: {code_edit.touched_files}")
    if code_edit.diff_hash:
        parts.append(f"Diff hash: {code_edit.diff_hash}")
    parts.append("")
    parts.append("### Unified Diff")
    parts.append(f"```diff\n{code_edit.diff}\n```")
    parts.append("")

    parts.append(
        "Generate and run tests for this code edit and produce a TestReport. "
        "Use the sandbox_write_file tool to create test files. "
        "Use the sandbox_run_tests tool to execute the test suite. "
        "Use the sandbox_read_file tool to inspect the source code before writing tests. "
        "Your TestReport must have passed >= 0, failed >= 0, and "
        "failed_test_names length must equal failed count. "
        "If failed > 0, list every failing test name."
    )

    return "\n".join(parts)
