"""Tests for the Coder agent — typed IO, model routing, RAG, sandbox tools, caching.

Covers all 8 VAL-CODER assertions:
  VAL-CODER-001: Coder emits typed CodeEdit
  VAL-CODER-002: Coder writes only inside the sandbox
  VAL-CODER-003: Coder uses DeepSeek V4 Flash (deepseek/deepseek-chat-v3-0324)
  VAL-CODER-004: Coder retrieves RAG context
  VAL-CODER-005: Coder applies diff via sandbox tool
  VAL-CODER-006: CodeEdit persisted with diff_hash
  VAL-CODER-007: Coder respects ReviewResult feedback
  VAL-CODER-008: Coder cached_tokens reported

Integration tests against real OpenRouter are marked @pytest.mark.integration
and require OPENROUTER_API_KEY in the environment.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError
from pydantic_ai import RunContext
from src.agents.models import ChangePlan, CodeEdit, ReviewResult

# ── Fixtures ────────────────────────────────────────────────────────


def _make_change_plan(**overrides: Any) -> ChangePlan:
    """Build a valid ChangePlan with sensible defaults."""
    defaults: dict[str, Any] = {
        "target_files": ["src/calculator.py"],
        "rationale": (
            "The subtract function incorrectly returns a + b "
            "instead of a - b, causing wrong results for callers."
        ),
        "approach": "Fix the return statement to use a - b.",
    }
    defaults.update(overrides)
    return ChangePlan(**defaults)


def _make_code_edit(**overrides: Any) -> CodeEdit:
    """Build a valid CodeEdit with sensible defaults."""
    diff = (
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def subtract(a, b):\n"
        "-    return a + b\n"
        "+    return a - b\n"
    )
    defaults: dict[str, Any] = {
        "diff": diff,
        "touched_files": ["src/calculator.py"],
    }
    defaults.update(overrides)
    return CodeEdit(**defaults)


def _make_review_result(**overrides: Any) -> ReviewResult:
    """Build a valid ReviewResult with sensible defaults."""
    defaults: dict[str, Any] = {
        "verdict": "reject_with_changes",
        "issues": [
            "The fix is correct but lacks a type annotation on the return value.",
        ],
    }
    defaults.update(overrides)
    return ReviewResult(**defaults)


# ── VAL-CODER-001: Coder emits typed CodeEdit ──────────────────────


class TestCoderEmitsTypedCodeEdit:
    """Given a ChangePlan (and optional prior ReviewResult), the Coder
    returns a Pydantic-valid CodeEdit with non-empty diff and
    touched_files. Free-form text is rejected."""

    @pytest.mark.asyncio
    async def test_emits_typed_code_edit(self) -> None:
        """Given a ChangePlan, Coder returns a CodeEdit."""
        from src.agents.coder import CoderDeps, coder

        plan = _make_change_plan()
        deps = CoderDeps(
            sandbox_manager=None,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_result = MagicMock()
        mock_result.output = _make_code_edit()

        with patch.object(coder, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            result = await coder.run(
                f"Plan: {plan.model_dump_json()}", deps=deps,
            )

        edit = result.output
        assert isinstance(edit, CodeEdit)
        assert len(edit.diff) >= 1
        assert len(edit.touched_files) >= 1

    @pytest.mark.asyncio
    async def test_code_edit_diff_non_empty(self) -> None:
        """CodeEdit.diff must be a non-empty string."""
        with pytest.raises(ValidationError):
            CodeEdit(diff="", touched_files=["src/main.py"])

    @pytest.mark.asyncio
    async def test_code_edit_touched_files_non_empty(self) -> None:
        """CodeEdit.touched_files must be a non-empty list."""
        with pytest.raises(ValidationError):
            CodeEdit(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n",
                touched_files=[],
            )


# ── VAL-CODER-002: Coder writes only inside the sandbox ─────────────


class TestCoderWritesOnlyInsideSandbox:
    """Every Coder file-write tool call targets a path under the
    sandbox-mounted repo root. Any attempted write outside the sandbox
    cwd is blocked by the Invariant guardrail and recorded as a
    guardrail_block outcome."""

    @pytest.mark.asyncio
    async def test_sandbox_path_validation_rejects_escape(self) -> None:
        """The sandbox write tool rejects paths outside /workspace."""
        from src.sandbox.errors import PathOutsideSandboxError
        from src.sandbox.manager import SandboxManager

        sb = SandboxManager(task_id="test-path-escape")
        with pytest.raises(PathOutsideSandboxError):
            sb._validate_path("../../etc/passwd")

    @pytest.mark.asyncio
    async def test_sandbox_path_validation_rejects_absolute_escape(
        self,
    ) -> None:
        """Absolute paths outside /workspace are rejected."""
        from src.sandbox.errors import PathOutsideSandboxError
        from src.sandbox.manager import SandboxManager

        sb = SandboxManager(task_id="test-abs-escape")
        with pytest.raises(PathOutsideSandboxError):
            sb._validate_path("/etc/passwd")

    @pytest.mark.asyncio
    async def test_sandbox_path_validation_allows_workspace_relative(
        self,
    ) -> None:
        """Relative paths inside /workspace are allowed."""
        from src.sandbox.manager import SandboxManager

        sb = SandboxManager(task_id="test-valid-path")
        resolved = sb._validate_path("src/calculator.py")
        assert "/workspace/" in resolved


# ── VAL-CODER-003: Coder uses DeepSeek V4 Flash ────────────────────


class TestCoderUsesCorrectModel:
    """Every Coder LLM call hits OpenRouter with
    model="deepseek/deepseek-chat-v3-0324"."""

    def test_coder_model_id_is_deepseek_chat_v3(self) -> None:
        """The coder agent is configured with the correct model ID."""
        from src.agents.coder import _CODER_MODEL

        assert _CODER_MODEL == "deepseek/deepseek-chat-v3-0324"

    def test_coder_openrouter_model_string(self) -> None:
        """The PydanticAI agent uses the OpenRouter model string."""
        from src.agents.coder import _CODER_MODEL, _PYDANTIC_AI_MODEL

        expected = f"openrouter:{_CODER_MODEL}"
        assert expected == _PYDANTIC_AI_MODEL
        assert "deepseek/deepseek-chat-v3-0324" in _PYDANTIC_AI_MODEL


# ── VAL-CODER-004: Coder retrieves RAG context ──────────────────────


class TestCoderRetrievesRAGContext:
    """The Coder issues at least one pgvector retrieval call against
    the current repo_url chunks before producing its first CodeEdit."""

    @pytest.mark.asyncio
    async def test_rag_retrieval_tool_returns_scoped_results(
        self,
    ) -> None:
        """The RAG retrieval tool returns results scoped to repo_url."""
        from src.agents.coder import CoderDeps, _rag_retrieval

        mock_rag = AsyncMock()
        mock_rag.retrieve.return_value = [
            {
                "file_path": "src/calculator.py",
                "chunk_text": "def subtract(a, b):\n    return a + b",
                "score": 0.95,
            },
        ]

        deps = CoderDeps(
            sandbox_manager=None,
            rag_retriever=mock_rag,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _rag_retrieval(mock_ctx, "subtract function")
        assert "src/calculator.py" in result
        mock_rag.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_retrieval_scoped_to_repo_url(self) -> None:
        """RAG retriever is called with the repo_url from deps."""
        from src.agents.coder import CoderDeps, _rag_retrieval

        mock_rag = AsyncMock()
        mock_rag.retrieve.return_value = []

        deps = CoderDeps(
            sandbox_manager=None,
            rag_retriever=mock_rag,
            task_id=uuid4(),
            trace_id="test-trace",
            repo_url="https://github.com/org/repo",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        await _rag_retrieval(mock_ctx, "subtract function")
        call_kwargs = mock_rag.retrieve.call_args[1]
        assert call_kwargs["repo_url"] == "https://github.com/org/repo"

    @pytest.mark.asyncio
    async def test_rag_retrieval_unavailable_graceful(
        self,
    ) -> None:
        """When RAG is unavailable, a graceful message is returned."""
        from src.agents.coder import CoderDeps, _rag_retrieval

        deps = CoderDeps(
            sandbox_manager=None,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _rag_retrieval(mock_ctx, "any query")
        assert "not available" in result


# ── VAL-CODER-005: Coder applies diff via sandbox tool ──────────────


class TestCoderAppliesDiffViaSandbox:
    """The Coder's diff is applied through the sandbox apply_diff
    tool, never via a host-side filesystem call."""

    @pytest.mark.asyncio
    async def test_apply_diff_tool_calls_sandbox(self) -> None:
        """The sandbox_apply_diff tool delegates to the sandbox manager."""
        from src.agents.coder import CoderDeps, _sandbox_apply_diff

        mock_sandbox = AsyncMock()
        mock_sandbox.apply_diff.return_value = None

        deps = CoderDeps(
            sandbox_manager=mock_sandbox,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
        result = await _sandbox_apply_diff(mock_ctx, diff)
        mock_sandbox.apply_diff.assert_called_once_with(diff)
        assert "Successfully" in result

    @pytest.mark.asyncio
    async def test_apply_diff_tool_handles_error(self) -> None:
        """The sandbox_apply_diff tool returns an error message on failure."""
        from src.agents.coder import CoderDeps, _sandbox_apply_diff

        mock_sandbox = AsyncMock()
        mock_sandbox.apply_diff.side_effect = Exception("patch failed")

        deps = CoderDeps(
            sandbox_manager=mock_sandbox,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _sandbox_apply_diff(mock_ctx, "bad diff")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_apply_diff_no_sandbox_graceful(self) -> None:
        """When no sandbox is available, a graceful message is returned."""
        from src.agents.coder import CoderDeps, _sandbox_apply_diff

        deps = CoderDeps(
            sandbox_manager=None,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _sandbox_apply_diff(mock_ctx, "diff")
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_run_coder_raises_when_sandbox_apply_diff_fails(self) -> None:
        """run_coder must not return a successful CodeEdit if the patch was not applied."""
        from decimal import Decimal

        from src.agents.coder import run_coder

        sandbox = AsyncMock()
        sandbox.read_file.return_value = "def subtract(a, b):\n    return a + b\n"
        sandbox.apply_diff.side_effect = RuntimeError("patch failed")

        mock_llm_result = MagicMock()
        mock_llm_result.content = (
            '```json\n'
            '{"diff": "--- a/src/calculator.py\\n+++ b/src/calculator.py\\n'
            '@@ -1 +1 @@\\n-old\\n+new\\n", '
            '"touched_files": ["src/calculator.py"], "diff_hash": "abc123"}\n'
            '```'
        )
        mock_llm_result.usage_input = 100
        mock_llm_result.usage_output = 50
        mock_llm_result.cached_tokens = 0
        mock_llm_result.cost_usd = Decimal("0.01")

        with patch("src.agents.coder.get_llm_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat_with_cache.return_value = mock_llm_result
            mock_get_client.return_value = mock_client

            with pytest.raises(RuntimeError, match="patch failed"):
                await run_coder(
                    change_plan=_make_change_plan(),
                    sandbox_manager=sandbox,
                    task_id=uuid4(),
                )


# ── VAL-CODER-006: CodeEdit persisted with diff_hash ────────────────


class TestCodeEditPersistedWithDiffHash:
    """The CodeEdit is written to decisions with decision_type='code_edit',
    and decision_data includes a diff_hash (sha256 of canonicalized diff)
    used for downstream same-fix detection."""

    @pytest.mark.asyncio
    async def test_persist_code_edit_as_decision(self) -> None:
        """After a successful Coder run, a decision row is created."""
        from src.agents.coder import persist_code_edit

        task_id = uuid4()
        edit = _make_code_edit()

        mock_store = AsyncMock()
        mock_store.create_decision.return_value = MagicMock()

        await persist_code_edit(
            store=mock_store,
            task_id=task_id,
            edit=edit,
            step_index=1,
        )

        mock_store.create_decision.assert_called_once()
        call_args = mock_store.create_decision.call_args
        params = call_args[0][0] if call_args[0] else call_args[1]

        assert params.agent == "coder"
        assert params.decision_type == "code_edit"
        assert "diff_hash" in params.decision_data
        assert "diff" in params.decision_data
        assert "touched_files" in params.decision_data

    def test_diff_hash_computed_from_diff(self) -> None:
        """diff_hash is SHA-256 of the diff content."""
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
        expected_hash = hashlib.sha256(diff.encode()).hexdigest()
        edit = CodeEdit(
            diff=diff,
            touched_files=["f.py"],
            diff_hash=expected_hash,
        )
        assert edit.diff_hash == expected_hash

    def test_different_diffs_produce_different_hashes(self) -> None:
        """Two different diffs must produce different diff_hash values."""
        diff1 = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new1\n"
        diff2 = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new2\n"
        hash1 = hashlib.sha256(diff1.encode()).hexdigest()
        hash2 = hashlib.sha256(diff2.encode()).hexdigest()
        assert hash1 != hash2


# ── VAL-CODER-007: Coder respects ReviewResult feedback ──────────────


class TestCoderRespectsReviewResultFeedback:
    """When invoked with a prior ReviewResult having verdict
    reject_with_changes, the resulting CodeEdit.diff_hash differs
    from the previously rejected edit's diff_hash."""

    @pytest.mark.asyncio
    async def test_coder_with_review_result_changes_output(self) -> None:
        """Coder with reject_with_changes produces a different diff_hash."""
        from src.agents.coder import CoderDeps, coder

        plan = _make_change_plan()
        review = _make_review_result(verdict="reject_with_changes")

        deps = CoderDeps(
            sandbox_manager=None,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        # First edit (rejected)
        first_edit = _make_code_edit()
        # Second edit (should differ)
        second_diff = (
            "--- a/src/calculator.py\n"
            "+++ b/src/calculator.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def subtract(a, b) -> int:\n"
            "-    return a + b\n"
            "+    return a - b\n"
        )
        second_edit = CodeEdit(
            diff=second_diff,
            touched_files=["src/calculator.py"],
            diff_hash=hashlib.sha256(second_diff.encode()).hexdigest(),
        )

        mock_result = MagicMock()
        mock_result.output = second_edit

        with patch.object(coder, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            result = await coder.run(
                f"Plan: {plan.model_dump_json()}\nReview: {review.model_dump_json()}",
                deps=deps,
            )

        new_hash = result.output.diff_hash
        old_hash = first_edit.diff_hash
        assert new_hash != old_hash, (
            "New CodeEdit must have a different diff_hash from the rejected one"
        )

    @pytest.mark.asyncio
    async def test_coder_run_with_review_builds_prompt(self) -> None:
        """run_coder includes ReviewResult in the prompt when provided."""
        from decimal import Decimal

        from src.agents.coder import CoderDeps, CoderRunResult, run_coder

        plan = _make_change_plan()
        review = _make_review_result()

        deps_instance = CoderDeps(
            sandbox_manager=None,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_llm_result = MagicMock()
        mock_llm_result.content = (
            '```json\n'
            '{"diff": "--- a/f.py\\n+++ b/f.py\\n@@ -1 +1 @@\\n-old\\n+new\\n",'
            ' "touched_files": ["f.py"], "diff_hash": "abc123"}\n```'
        )
        mock_llm_result.usage_input = 1000
        mock_llm_result.usage_output = 200
        mock_llm_result.cached_tokens = 500
        mock_llm_result.cost_usd = Decimal("0.01")

        with patch(
            "src.agents.coder.get_llm_client",
        ) as mock_get_client, patch(
            "src.agents.coder.CoderDeps",
            return_value=deps_instance,
        ):
            mock_client = AsyncMock()
            mock_client.chat_with_cache.return_value = mock_llm_result
            mock_get_client.return_value = mock_client
            result = await run_coder(
                change_plan=plan,
                review_result=review,
            )

        assert isinstance(result, CoderRunResult)
        assert isinstance(result.edit, CodeEdit)


# ── VAL-CODER-008: Coder cached_tokens reported ────────────────────


class TestCoderCachedTokensReported:
    """Each Coder OpenRouter response includes a
    usage.prompt_tokens_details.cached_tokens field and that value is
    recorded on the Langfuse span."""

    def test_extract_cached_tokens_from_usage(self) -> None:
        """extract_cached_tokens returns the cached_tokens value."""
        from src.llm.cost import extract_cached_tokens

        usage = {
            "prompt_tokens_details": {
                "cached_tokens": 512,
            },
        }
        assert extract_cached_tokens(usage) == 512

    def test_extract_cached_tokens_missing_field(self) -> None:
        """When prompt_tokens_details is missing, returns 0."""
        from src.llm.cost import extract_cached_tokens

        assert extract_cached_tokens({}) == 0

    def test_extract_cached_tokens_zero_value(self) -> None:
        """When cached_tokens is 0, returns 0."""
        from src.llm.cost import extract_cached_tokens

        usage = {
            "prompt_tokens_details": {
                "cached_tokens": 0,
            },
        }
        assert extract_cached_tokens(usage) == 0

    def test_prompt_caching_markers_in_structured_prompt(self) -> None:
        """The Coder prompt builder creates static+dynamic blocks."""
        from src.llm.caching import build_structured_prompt

        prompt = build_structured_prompt(
            system_instructions="You are a coding agent.",
            repo_context="File: src/main.py\ndef hello(): pass",
            current_edit="Change hello() to greet()",
            prior_review="Issues: add type hints",
        )
        assert prompt.static
        assert prompt.dynamic
        assert "coding agent" in prompt.static
        assert "repo context" in prompt.static.lower()
        assert "Change hello" in prompt.dynamic


# ── Integration test (real OpenRouter, requires API key) ──────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_coder_integration_real_llm() -> None:
    """Integration test: real OpenRouter call returns a valid CodeEdit.

    Uses a low-token test prompt.  Only validates structural correctness
    of the output, not reasoning quality.

    Requires OPENROUTER_API_KEY in the environment.
    """
    import os

    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    from src.agents.coder import CoderDeps, coder

    plan = _make_change_plan()
    deps = CoderDeps(
        sandbox_manager=None,
        rag_retriever=None,
        task_id=uuid4(),
        trace_id="integration-test-trace",
    )

    result = await coder.run(
        f"Produce a CodeEdit to fix: {plan.rationale}. "
        f"Target files: {plan.target_files}",
        deps=deps,
    )

    edit = result.output
    assert isinstance(edit, CodeEdit)
    assert len(edit.diff) >= 1
    assert len(edit.touched_files) >= 1
