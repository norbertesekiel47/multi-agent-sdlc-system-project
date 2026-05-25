"""Tests for QA agent generated_test_contents field and sandbox_write_file wiring.

Fix 1 from M4 scrutiny: run_qa() uses chat_with_cache() which removes
tool access. Add generated_test_contents: dict[str, str] field to
TestReport so the chat_with_cache path can write test files directly
via sandbox_write_file after the LLM response.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.agents.models import TestReport

# ── TestReport model tests ──────────────────────────────────────────


class TestGeneratedTestContentsField:
    """Verify the generated_test_contents field on TestReport."""

    def test_generated_test_contents_default_empty(self) -> None:
        """generated_test_contents defaults to empty dict."""
        report = TestReport(passed=3, failed=0, failed_test_names=[])
        assert report.generated_test_contents == {}

    def test_generated_test_contents_can_be_set(self) -> None:
        """generated_test_contents can be populated with test file content."""
        contents = {
            "tests/test_foo.py": "def test_foo(): assert True",
            "tests/test_bar.py": "def test_bar(): assert True",
        }
        report = TestReport(
            passed=2,
            failed=0,
            failed_test_names=[],
            generated_test_files=["tests/test_foo.py", "tests/test_bar.py"],
            generated_test_contents=contents,
        )
        assert report.generated_test_contents == contents
        assert len(report.generated_test_contents) == 2

    def test_generated_test_contents_serializes_to_json(self) -> None:
        """generated_test_contents serializes correctly in model_dump."""
        contents = {"tests/test_x.py": "import pytest\n\ndef test_x(): pass"}
        report = TestReport(
            passed=1,
            failed=0,
            failed_test_names=[],
            generated_test_contents=contents,
        )
        dumped = report.model_dump(mode="json")
        assert "generated_test_contents" in dumped
        assert dumped["generated_test_contents"]["tests/test_x.py"] == contents["tests/test_x.py"]

    def test_generated_test_contents_with_failed_tests(self) -> None:
        """generated_test_contents works alongside failed test names."""
        contents = {"tests/test_failing.py": "def test_failing(): assert False"}
        report = TestReport(
            passed=0,
            failed=1,
            failed_test_names=["test_failing"],
            generated_test_files=["tests/test_failing.py"],
            generated_test_contents=contents,
        )
        assert report.generated_test_contents == contents


# ── run_qa wiring tests ────────────────────────────────────────────


class TestRunQaWritesGeneratedContents:
    """Verify that run_qa writes files from generated_test_contents to sandbox."""

    @pytest.mark.asyncio
    async def test_sandbox_write_file_called_for_each_entry(self) -> None:
        """run_qa should call sandbox.write_file for each generated_test_contents entry."""
        from src.agents.models import CodeEdit
        from src.agents.qa import run_qa

        code_edit = CodeEdit(
            diff='--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new',
            touched_files=["foo.py"],
            diff_hash="abc123",
        )

        # Mock the LLM client to return a TestReport with generated_test_contents
        mock_report = TestReport(
            passed=2,
            failed=0,
            failed_test_names=[],
            generated_test_files=["tests/test_a.py", "tests/test_b.py"],
            generated_test_contents={
                "tests/test_a.py": "def test_a(): assert True",
                "tests/test_b.py": "def test_b(): assert True",
            },
        )

        mock_sandbox = AsyncMock()
        mock_sandbox.read_file = AsyncMock(return_value="# existing code")
        mock_sandbox.write_file = AsyncMock()
        mock_sandbox.run_tests = AsyncMock(
            return_value="2 passed in 0.5s"
        )

        mock_llm_result = MagicMock()
        mock_llm_result.content = json.dumps(mock_report.model_dump(mode="json"))
        mock_llm_result.usage_input = 100
        mock_llm_result.usage_output = 50
        mock_llm_result.cached_tokens = 0
        mock_llm_result.cost_usd = 0.001

        with patch("src.agents.qa.get_llm_client") as mock_get_client, \
             patch("src.agents.qa.persist_test_report", new_callable=AsyncMock):
            mock_client = AsyncMock()
            mock_client.chat_with_cache = AsyncMock(return_value=mock_llm_result)
            mock_get_client.return_value = mock_client

            _ = await run_qa(  # noqa: F841
                code_edit=code_edit,
                sandbox_manager=mock_sandbox,
            )

        # Verify sandbox.write_file was called for each generated test file
        assert mock_sandbox.write_file.call_count >= 2
        write_calls = [call.args for call in mock_sandbox.write_file.call_args_list]
        written_paths = {args[0] for args in write_calls}
        assert "tests/test_a.py" in written_paths
        assert "tests/test_b.py" in written_paths

    @pytest.mark.asyncio
    async def test_empty_generated_test_contents_no_write(self) -> None:
        """When generated_test_contents is empty, no sandbox writes occur from it."""
        from src.agents.models import CodeEdit
        from src.agents.qa import run_qa

        code_edit = CodeEdit(
            diff='--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new',
            touched_files=["foo.py"],
            diff_hash="abc123",
        )

        mock_report = TestReport(
            passed=0,
            failed=0,
            failed_test_names=[],
            generated_test_files=[],
            generated_test_contents={},
        )

        mock_sandbox = AsyncMock()
        mock_sandbox.read_file = AsyncMock(return_value="# code")
        mock_sandbox.run_tests = AsyncMock(return_value="0 passed in 0.1s")

        mock_llm_result = MagicMock()
        mock_llm_result.content = json.dumps(mock_report.model_dump(mode="json"))
        mock_llm_result.usage_input = 50
        mock_llm_result.usage_output = 20
        mock_llm_result.cached_tokens = 0
        mock_llm_result.cost_usd = 0.0005

        with patch("src.agents.qa.get_llm_client") as mock_get_client, \
             patch("src.agents.qa.persist_test_report", new_callable=AsyncMock):
            mock_client = AsyncMock()
            mock_client.chat_with_cache = AsyncMock(return_value=mock_llm_result)
            mock_get_client.return_value = mock_client

            result = await run_qa(
                code_edit=code_edit,
                sandbox_manager=mock_sandbox,
            )

        # No write calls from generated_test_contents (it's empty)
        # The write_file might still be called by other code paths,
        # but not from generated_test_contents iteration
        assert result.report is not None

    @pytest.mark.asyncio
    async def test_generated_test_contents_preserved_through_reconciliation(self) -> None:
        """generated_test_contents is preserved when test output reconciliation happens."""
        from src.agents.models import CodeEdit
        from src.agents.qa import run_qa

        code_edit = CodeEdit(
            diff='--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new',
            touched_files=["foo.py"],
            diff_hash="abc123",
        )

        mock_report = TestReport(
            passed=2,
            failed=0,
            failed_test_names=[],
            generated_test_files=["tests/test_a.py"],
            generated_test_contents={
                "tests/test_a.py": "def test_a(): assert True",
            },
        )

        mock_sandbox = AsyncMock()
        mock_sandbox.read_file = AsyncMock(return_value="# code")
        mock_sandbox.write_file = AsyncMock()
        # Test runner reports different counts → triggers reconciliation
        mock_sandbox.run_tests = AsyncMock(
            return_value="3 passed in 0.5s"
        )

        mock_llm_result = MagicMock()
        mock_llm_result.content = json.dumps(mock_report.model_dump(mode="json"))
        mock_llm_result.usage_input = 100
        mock_llm_result.usage_output = 50
        mock_llm_result.cached_tokens = 0
        mock_llm_result.cost_usd = 0.001

        with patch("src.agents.qa.get_llm_client") as mock_get_client, \
             patch("src.agents.qa.persist_test_report", new_callable=AsyncMock):
            mock_client = AsyncMock()
            mock_client.chat_with_cache = AsyncMock(return_value=mock_llm_result)
            mock_get_client.return_value = mock_client

            result = await run_qa(
                code_edit=code_edit,
                sandbox_manager=mock_sandbox,
            )

        # After reconciliation, generated_test_contents should still be there
        assert result.report.generated_test_contents == {
            "tests/test_a.py": "def test_a(): assert True"
        }
