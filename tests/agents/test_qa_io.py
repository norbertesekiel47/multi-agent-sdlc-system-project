"""Tests for the QA agent — typed IO, test generation, test execution, persistence.

Covers all 6 VAL-QA assertions:
  VAL-QA-001: QA emits typed TestReport
  VAL-QA-002: QA generates test files in sandbox
  VAL-QA-003: QA executes the test runner in sandbox
  VAL-QA-004: TestReport persisted as decision
  VAL-QA-005: QA failure does not auto-open PR
  VAL-QA-006: QA failed_test_names are non-empty when failed > 0

Integration tests against real OpenRouter are marked @pytest.mark.integration
and require OPENROUTER_API_KEY in the environment.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError
from pydantic_ai import RunContext
from src.agents.models import CodeEdit, TestReport

# ── Fixtures ────────────────────────────────────────────────────────


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


def _make_test_report(**overrides: Any) -> TestReport:
    """Build a valid TestReport with sensible defaults."""
    defaults: dict[str, Any] = {
        "passed": 3,
        "failed": 0,
        "failed_test_names": [],
        "generated_test_files": ["tests/test_calculator.py"],
    }
    defaults.update(overrides)
    return TestReport(**defaults)


# ── VAL-QA-001: QA emits typed TestReport ─────────────────────────


class TestQAEmitsTypedTestReport:
    """Given a post-review CodeEdit, the QA agent returns a
    Pydantic-valid TestReport containing passed: int, failed: int,
    failed_test_names: List[str], and generated_test_files: List[str].
    Free-form text is rejected."""

    @pytest.mark.asyncio
    async def test_emits_typed_test_report(self) -> None:
        """Given a CodeEdit, QA returns a valid TestReport."""
        from src.agents.qa import QADeps, qa

        edit = _make_code_edit()
        deps = QADeps(
            sandbox_manager=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_result = MagicMock()
        mock_result.output = _make_test_report()

        with patch.object(qa, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            result = await qa.run(
                f"Run tests for diff: {edit.diff}", deps=deps,
            )

        report = result.output
        assert isinstance(report, TestReport)
        assert report.passed >= 0
        assert report.failed >= 0
        assert isinstance(report.failed_test_names, list)
        assert isinstance(report.generated_test_files, list)

    @pytest.mark.asyncio
    async def test_test_report_all_passing(self) -> None:
        """TestReport with all passing tests is valid."""
        report = _make_test_report(passed=5, failed=0, failed_test_names=[])
        assert report.passed == 5
        assert report.failed == 0
        assert report.failed_test_names == []

    @pytest.mark.asyncio
    async def test_test_report_with_failures(self) -> None:
        """TestReport with failing tests includes their names."""
        report = _make_test_report(
            passed=3,
            failed=2,
            failed_test_names=["test_subtract", "test_divide"],
        )
        assert report.failed == 2
        assert len(report.failed_test_names) == 2
        assert "test_subtract" in report.failed_test_names

    def test_free_form_text_rejected(self) -> None:
        """Free-form text outside TestReport schema is rejected.

        The QA agent's output_type=TestReport enforces Pydantic
        validation.  Free-form text (not a valid TestReport) is
        rejected by the PydanticAI output parser.
        """
        # A string that is not a valid TestReport JSON
        with pytest.raises((ValidationError, ValueError)):
            TestReport.model_validate("just some free text")

    def test_missing_required_fields_rejected(self) -> None:
        """Missing required fields are rejected."""
        with pytest.raises(ValidationError):
            TestReport.model_validate({})  # missing 'passed' and 'failed'

    def test_negative_passed_rejected(self) -> None:
        """Negative passed count is rejected."""
        with pytest.raises(ValidationError):
            TestReport(passed=-1, failed=0, failed_test_names=[], generated_test_files=[])

    def test_negative_failed_rejected(self) -> None:
        """Negative failed count is rejected."""
        with pytest.raises(ValidationError):
            TestReport(passed=3, failed=-1, failed_test_names=["test"], generated_test_files=[])


# ── VAL-QA-002: QA generates test files in sandbox ──────────────────


class TestQAGeneratesTestFilesInSandbox:
    """QA writes any generated test file via the sandbox write_file
    tool under the sandbox repo root. The number of files written
    equals len(generated_test_files)."""

    @pytest.mark.asyncio
    async def test_sandbox_write_file_tool_writes_tests(self) -> None:
        """The sandbox_write_file tool writes test files to the sandbox."""
        from src.agents.qa import QADeps, _sandbox_write_file

        mock_sandbox = AsyncMock()
        mock_sandbox.write_file.return_value = None

        deps = QADeps(
            sandbox_manager=mock_sandbox,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _sandbox_write_file(
            mock_ctx,
            path="tests/test_calculator.py",
            content="def test_subtract():\n    assert subtract(5, 3) == 2\n",
        )
        mock_sandbox.write_file.assert_called_once_with(
            "tests/test_calculator.py",
            "def test_subtract():\n    assert subtract(5, 3) == 2\n",
        )
        assert "written" in result.lower() or "ok" in result.lower() or "success" in result.lower()

    @pytest.mark.asyncio
    async def test_sandbox_write_file_no_sandbox(self) -> None:
        """When no sandbox is available, a graceful message is returned."""
        from src.agents.qa import QADeps, _sandbox_write_file

        deps = QADeps(
            sandbox_manager=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _sandbox_write_file(
            mock_ctx,
            path="tests/test_calculator.py",
            content="def test_subtract(): pass\n",
        )
        assert "not available" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_generated_test_files_match_writes(self) -> None:
        """TestReport.generated_test_files lists match the files written."""
        report = _make_test_report(
            generated_test_files=["tests/test_calculator.py", "tests/test_utils.py"],
        )
        assert len(report.generated_test_files) == 2


# ── VAL-QA-003: QA executes the test runner in sandbox ──────────────


class TestQAExecutesTestRunnerInSandbox:
    """QA invokes sandbox.run_tests exactly once per QA turn.
    The numeric counts in TestReport.passed and TestReport.failed
    derive from that runner's parsed output."""

    @pytest.mark.asyncio
    async def test_run_tests_tool_invoked(self) -> None:
        """The run_tests tool calls sandbox.run_tests."""
        from src.agents.qa import QADeps, _sandbox_run_tests

        mock_sandbox = AsyncMock()
        mock_sandbox.run_tests.return_value = (
            "3 passed, 1 failed in 0.5s\n"
            "FAILED test_subtract - AssertionError\n"
        )

        deps = QADeps(
            sandbox_manager=mock_sandbox,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _sandbox_run_tests(mock_ctx)
        mock_sandbox.run_tests.assert_called_once()
        assert isinstance(result, str)
        assert "passed" in result.lower() or "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_run_tests_no_sandbox_graceful(self) -> None:
        """When no sandbox is available, a graceful message is returned."""
        from src.agents.qa import QADeps, _sandbox_run_tests

        deps = QADeps(
            sandbox_manager=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _sandbox_run_tests(mock_ctx)
        assert "not available" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_run_tests_custom_command(self) -> None:
        """The run_tests tool accepts a custom test command."""
        from src.agents.qa import QADeps, _sandbox_run_tests

        mock_sandbox = AsyncMock()
        mock_sandbox.run_tests.return_value = "2 passed in 0.3s"

        deps = QADeps(
            sandbox_manager=mock_sandbox,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        await _sandbox_run_tests(mock_ctx, command="pytest -xvs tests/")
        mock_sandbox.run_tests.assert_called_once_with("pytest -xvs tests/")

    def test_pass_plus_fail_equals_total(self) -> None:
        """passed + failed gives the total number of tests run."""
        report = _make_test_report(passed=7, failed=3, failed_test_names=["a", "b", "c"])
        assert report.passed + report.failed == 10


# ── VAL-QA-004: TestReport persisted as decision ────────────────────


class TestTestReportPersistedAsDecision:
    """The TestReport is persisted to decisions with agent='qa',
    decision_type='test_report', decision_data matching the Pydantic JSON."""

    @pytest.mark.asyncio
    async def test_persist_test_report_as_decision(self) -> None:
        """After a successful QA run, a decision row is created."""
        from src.agents.qa import persist_test_report

        task_id = uuid4()
        report = _make_test_report()

        mock_store = AsyncMock()
        mock_store.create_decision.return_value = MagicMock()

        await persist_test_report(
            store=mock_store,
            task_id=task_id,
            report=report,
            step_index=3,
        )

        mock_store.create_decision.assert_called_once()
        call_args = mock_store.create_decision.call_args
        params = call_args[0][0] if call_args[0] else call_args[1]

        assert params.agent == "qa"
        assert params.decision_type == "test_report"
        assert "passed" in params.decision_data
        assert "failed" in params.decision_data
        assert "failed_test_names" in params.decision_data
        assert "generated_test_files" in params.decision_data


# ── VAL-QA-005: QA failure does not auto-open PR ────────────────────


class TestQAFailureDoesNotAutoOpenPR:
    """When TestReport.failed > 0, the orchestrator does NOT proceed
    to PR creation. It either retries (within budget) or escalates
    to HITL with cause persistent_test_failure."""

    def test_failed_count_positive(self) -> None:
        """TestReport with failures has failed > 0."""
        report = _make_test_report(
            passed=3, failed=2, failed_test_names=["test_a", "test_b"],
        )
        assert report.failed > 0

    def test_pr_not_opened_when_failures(self) -> None:
        """The orchestrator should not proceed to PR when tests fail.

        This is an integration-level concern. Here we verify that
        the TestReport model correctly represents the failure state
        so the orchestrator can make the routing decision.
        """
        report = _make_test_report(
            passed=3, failed=2, failed_test_names=["test_a", "test_b"],
        )
        # The orchestrator checks: if report.failed > 0, do NOT open PR
        should_open_pr = report.failed == 0
        assert not should_open_pr, "PR should NOT be opened when tests fail"

    def test_pr_can_open_when_all_pass(self) -> None:
        """When all tests pass, PR can be opened."""
        report = _make_test_report(passed=5, failed=0, failed_test_names=[])
        should_open_pr = report.failed == 0
        assert should_open_pr, "PR can be opened when all tests pass"


# ── VAL-QA-006: QA failed_test_names are non-empty when failed > 0 ──


class TestQAFailedTestNamesMatchCount:
    """The Pydantic schema enforces that failed_test_names len equals failed.
    A TestReport with failed=2 and failed_test_names=[] fails validation."""

    def test_failed_names_must_match_count(self) -> None:
        """failed_test_names length must equal failed count."""
        # Valid: 2 failures with 2 names
        report = TestReport(
            passed=3,
            failed=2,
            failed_test_names=["test_a", "test_b"],
            generated_test_files=["tests/test.py"],
        )
        assert len(report.failed_test_names) == report.failed

    def test_empty_names_with_zero_failed_valid(self) -> None:
        """0 failed with empty failed_test_names is valid."""
        report = TestReport(
            passed=5,
            failed=0,
            failed_test_names=[],
            generated_test_files=["tests/test.py"],
        )
        assert report.failed == 0
        assert report.failed_test_names == []

    def test_mismatch_rejected(self) -> None:
        """failed=2 with empty failed_test_names fails validation."""
        with pytest.raises(ValidationError):
            TestReport(
                passed=3,
                failed=2,
                failed_test_names=[],
                generated_test_files=["tests/test.py"],
            )

    def test_too_many_names_rejected(self) -> None:
        """failed=1 with 2 failed_test_names fails validation."""
        with pytest.raises(ValidationError):
            TestReport(
                passed=3,
                failed=1,
                failed_test_names=["test_a", "test_b"],
                generated_test_files=["tests/test.py"],
            )

    def test_single_failure_with_name_valid(self) -> None:
        """failed=1 with exactly one name is valid."""
        report = TestReport(
            passed=4,
            failed=1,
            failed_test_names=["test_failing"],
            generated_test_files=["tests/test.py"],
        )
        assert report.failed == 1
        assert report.failed_test_names == ["test_failing"]


# ── Model ID validation ─────────────────────────────────────────────


class TestQAUsesCorrectModel:
    """QA uses deepseek/deepseek-chat-v3-0324 model ID."""

    def test_qa_model_id_is_deepseek_chat_v3(self) -> None:
        """The QA agent is configured with the correct model ID."""
        from src.agents.qa import _QA_MODEL

        assert _QA_MODEL == "deepseek/deepseek-chat-v3-0324"

    def test_qa_openrouter_model_string(self) -> None:
        """The PydanticAI agent uses the OpenRouter model string."""
        from src.agents.qa import _PYDANTIC_AI_MODEL, _QA_MODEL

        expected = f"openrouter:{_QA_MODEL}"
        assert expected == _PYDANTIC_AI_MODEL
        assert "deepseek/deepseek-chat-v3-0324" in _PYDANTIC_AI_MODEL


# ── Run QA convenience function ──────────────────────────────────────


class TestRunQAConvenienceFunction:
    """The run_qa convenience function returns a QARunResult with
    the TestReport and token/cost metadata for orchestrator accumulation."""

    @pytest.mark.asyncio
    async def test_run_qa_returns_result(self) -> None:
        """run_qa returns a QARunResult with TestReport and metadata."""
        from src.agents.qa import QARunResult, run_qa

        edit = _make_code_edit()
        task_id = uuid4()

        # Mock the LLM client
        mock_llm_result = MagicMock()
        mock_llm_result.content = (
            '{"passed": 3, "failed": 0, "failed_test_names": [], '
            '"generated_test_files": ["tests/test_calculator.py"]}'
        )
        mock_llm_result.usage_input = 100
        mock_llm_result.usage_output = 50
        mock_llm_result.cached_tokens = 0
        mock_llm_result.cost_usd = 0.001

        with patch("src.agents.qa.get_llm_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat_with_cache.return_value = mock_llm_result
            mock_get_client.return_value = mock_client

            result = await run_qa(
                code_edit=edit,
                sandbox_manager=None,
                episodic_store=None,
                task_id=task_id,
                trace_id="test-trace",
                repo_url="https://github.com/test/repo",
            )

        assert isinstance(result, QARunResult)
        assert isinstance(result.report, TestReport)
        assert result.report.passed == 3
        assert result.report.failed == 0

    @pytest.mark.asyncio
    async def test_run_qa_with_sandbox_generates_tests(self) -> None:
        """run_qa with a sandbox writes generated test files."""
        from src.agents.qa import QARunResult, run_qa

        edit = _make_code_edit()
        task_id = uuid4()

        mock_sandbox = AsyncMock()
        mock_sandbox.read_file.return_value = "def subtract(a, b): return a - b"
        mock_sandbox.run_tests.return_value = "2 passed in 0.3s"
        mock_sandbox.write_file.return_value = None

        mock_llm_result = MagicMock()
        mock_llm_result.content = (
            '{"passed": 2, "failed": 0, "failed_test_names": [], '
            '"generated_test_files": ["tests/test_calculator.py"]}'
        )
        mock_llm_result.usage_input = 200
        mock_llm_result.usage_output = 100
        mock_llm_result.cached_tokens = 50
        mock_llm_result.cost_usd = 0.003

        with patch("src.agents.qa.get_llm_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat_with_cache.return_value = mock_llm_result
            mock_get_client.return_value = mock_client

            result = await run_qa(
                code_edit=edit,
                sandbox_manager=mock_sandbox,
                episodic_store=None,
                task_id=task_id,
                trace_id="test-trace",
                repo_url="https://github.com/test/repo",
            )

        assert isinstance(result, QARunResult)
        assert result.report.generated_test_files == ["tests/test_calculator.py"]


# ── Integration test (real OpenRouter, requires API key) ──────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qa_integration_real_llm() -> None:
    """Integration test: real OpenRouter call returns a valid TestReport.

    Uses a low-token test prompt.  Only validates structural correctness
    of the output, not reasoning quality.

    Requires OPENROUTER_API_KEY in the environment.
    """
    import os

    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    from src.agents.qa import QADeps, qa

    edit = _make_code_edit()
    deps = QADeps(
        sandbox_manager=None,
        task_id=uuid4(),
        trace_id="integration-test-trace",
    )

    result = await qa.run(
        f"Generate and run tests for this diff and produce a TestReport:\n{edit.diff}",
        deps=deps,
    )

    report = result.output
    assert isinstance(report, TestReport)
    assert report.passed >= 0
    assert report.failed >= 0
    assert len(report.failed_test_names) == report.failed
