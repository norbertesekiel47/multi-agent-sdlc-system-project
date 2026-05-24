"""Tests for agent models — typed IO validation.

Verifies that Pydantic models enforce the correct constraints:
- IssueContext requires non-empty fields and supports RAG/episodic fields
- RAGHit validates file_path and chunk_text are non-empty
- EpisodicFact validates fact_kind
- ChangePlan requires non-empty target_files and rationale ≥ 20 chars
- ChangePlan rejects whitespace-only rationale
- CodeEdit requires non-empty diff and touched_files
- TestReport requires non-negative passed/failed
- ReviewResult verdict must be in the allowed set
- SingleAgentOutput combines all sub-models
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.agents.models import (
    ChangePlan,
    CodeEdit,
    EpisodicFact,
    IssueContext,
    RAGHit,
    ReviewResult,
    SingleAgentOutput,
    TestReport,
)


class TestRAGHit:
    """Tests for RAGHit model validation."""

    def test_valid_rag_hit(self) -> None:
        hit = RAGHit(file_path="src/calculator.py", chunk_text="def add(a, b):", score=0.95)
        assert hit.file_path == "src/calculator.py"
        assert hit.score == 0.95

    def test_empty_file_path_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RAGHit(file_path="", chunk_text="some code", score=0.5)

    def test_empty_chunk_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RAGHit(file_path="src/main.py", chunk_text="", score=0.5)

    def test_default_score(self) -> None:
        hit = RAGHit(file_path="src/main.py", chunk_text="code")
        assert hit.score == 0.0


class TestEpisodicFact:
    """Tests for EpisodicFact model validation."""

    def test_valid_episodic_fact(self) -> None:
        fact = EpisodicFact(fact_kind="language", fact_value={"value": "python"})
        assert fact.fact_kind == "language"

    def test_default_fact_value(self) -> None:
        fact = EpisodicFact(fact_kind="test_command")
        assert fact.fact_value == {}


class TestIssueContext:
    """Tests for IssueContext model validation."""

    def test_valid_issue_context(self) -> None:
        ctx = IssueContext(
            repo_url="https://github.com/org/repo",
            issue_number=1,
            issue_text="Bug: something is broken",
        )
        assert ctx.repo_url == "https://github.com/org/repo"
        assert ctx.issue_number == 1

    def test_empty_repo_url_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IssueContext(
                repo_url="",
                issue_number=1,
                issue_text="Bug",
            )

    def test_empty_issue_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IssueContext(
                repo_url="https://github.com/org/repo",
                issue_number=1,
                issue_text="",
            )

    def test_zero_issue_number_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IssueContext(
                repo_url="https://github.com/org/repo",
                issue_number=0,
                issue_text="Bug",
            )

    def test_negative_issue_number_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IssueContext(
                repo_url="https://github.com/org/repo",
                issue_number=-1,
                issue_text="Bug",
            )

    def test_optional_fields_default(self) -> None:
        ctx = IssueContext(
            repo_url="https://github.com/org/repo",
            issue_number=1,
            issue_text="Bug",
        )
        assert ctx.repo_files == {}
        assert ctx.repo_facts == []
        assert ctx.rag_hits == []
        assert ctx.episodic_facts == []
        assert ctx.recent_decisions == []

    def test_with_rag_hits(self) -> None:
        ctx = IssueContext(
            repo_url="https://github.com/org/repo",
            issue_number=1,
            issue_text="Bug",
            rag_hits=[
                RAGHit(file_path="src/main.py", chunk_text="def foo():", score=0.9),
            ],
        )
        assert len(ctx.rag_hits) == 1
        assert ctx.rag_hits[0].file_path == "src/main.py"

    def test_with_episodic_facts(self) -> None:
        ctx = IssueContext(
            repo_url="https://github.com/org/repo",
            issue_number=1,
            issue_text="Bug",
            episodic_facts=[
                EpisodicFact(fact_kind="language", fact_value={"value": "python"}),
            ],
        )
        assert len(ctx.episodic_facts) == 1


class TestChangePlan:
    """Tests for ChangePlan model validation."""

    def test_valid_change_plan(self) -> None:
        plan = ChangePlan(
            target_files=["src/calculator.py"],
            rationale="The subtract function returns a + b instead of a - b",
        )
        assert plan.target_files == ["src/calculator.py"]
        assert len(plan.rationale) >= 20

    def test_empty_target_files_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChangePlan(
                target_files=[],
                rationale="This is a valid rationale with enough length",
            )

    def test_short_rationale_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChangePlan(
                target_files=["src/main.py"],
                rationale="Too short",
            )

    def test_rationale_min_20_chars(self) -> None:
        plan = ChangePlan(
            target_files=["src/main.py"],
            rationale="12345678901234567890",  # exactly 20
        )
        assert len(plan.rationale) == 20

    def test_whitespace_rationale_rejected(self) -> None:
        """VAL-PLANNER-007: whitespace-only rationale is rejected."""
        with pytest.raises(ValidationError):
            ChangePlan(
                target_files=["src/main.py"],
                rationale="                    ",  # 20 spaces
            )

    def test_estimated_complexity_default(self) -> None:
        plan = ChangePlan(
            target_files=["src/main.py"],
            rationale="This is a change plan with enough rationale length",
        )
        assert plan.estimated_complexity == "medium"


class TestCodeEdit:
    """Tests for CodeEdit model validation."""

    def test_valid_code_edit(self) -> None:
        edit = CodeEdit(
            diff="--- a/src/main.py\n+++ b/src/main.py\n@@ -1,1 +1,1 @@\n-old\n+new",
            touched_files=["src/main.py"],
        )
        assert edit.diff.startswith("---")

    def test_empty_diff_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CodeEdit(diff="", touched_files=["src/main.py"])

    def test_empty_touched_files_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CodeEdit(
                diff="some diff content",
                touched_files=[],
            )


class TestTestReport:
    """Tests for TestReport model validation."""

    def test_valid_test_report(self) -> None:
        report = TestReport(passed=5, failed=2, failed_test_names=["test_a", "test_b"])
        assert report.passed == 5
        assert report.failed == 2

    def test_negative_passed_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TestReport(passed=-1, failed=0)

    def test_negative_failed_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TestReport(passed=0, failed=-1)

    def test_all_passing(self) -> None:
        report = TestReport(passed=10, failed=0, failed_test_names=[])
        assert report.failed == 0
        assert report.failed_test_names == []

    def test_not_collected_by_pytest(self) -> None:
        """TestReport has __test__ = False to prevent pytest collection."""
        assert TestReport.__test__ is False


class TestReviewResult:
    """Tests for ReviewResult model validation."""

    def test_accept_verdict(self) -> None:
        result = ReviewResult(verdict="accept")
        assert result.verdict == "accept"

    def test_reject_with_changes_verdict(self) -> None:
        result = ReviewResult(verdict="reject_with_changes", issues=["missing test"])
        assert result.verdict == "reject_with_changes"

    def test_reject_verdict(self) -> None:
        result = ReviewResult(verdict="reject", issues=["wrong approach"])
        assert result.verdict == "reject"


class TestSingleAgentOutput:
    """Tests for SingleAgentOutput model validation."""

    def test_valid_output(self) -> None:
        plan = ChangePlan(
            target_files=["src/calculator.py"],
            rationale="The subtract function returns a + b instead of a - b",
        )
        edit = CodeEdit(
            diff=(
                "--- a/src/calculator.py\n"
                "+++ b/src/calculator.py\n"
                "@@ -1,1 +1,1 @@\n"
                "-return a + b\n"
                "+return a - b"
            ),
            touched_files=["src/calculator.py"],
        )
        report = TestReport(passed=5, failed=0)
        output = SingleAgentOutput(
            plan=plan,
            code_edit=edit,
            test_report=report,
            ready_for_pr=True,
            summary="Fixed subtract bug",
        )
        assert output.ready_for_pr is True
        assert output.plan.target_files == ["src/calculator.py"]

    def test_default_ready_for_pr_false(self) -> None:
        plan = ChangePlan(
            target_files=["src/main.py"],
            rationale="This is a change plan with enough rationale length",
        )
        edit = CodeEdit(
            diff="some diff",
            touched_files=["src/main.py"],
        )
        report = TestReport(passed=3, failed=2, failed_test_names=["test_a", "test_b"])
        output = SingleAgentOutput(
            plan=plan,
            code_edit=edit,
            test_report=report,
        )
        assert output.ready_for_pr is False
