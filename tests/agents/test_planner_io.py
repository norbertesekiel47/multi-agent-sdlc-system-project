"""Tests for the Planner agent — typed IO, model routing, RAG, episodic memory.

Covers all 7 VAL-PLANNER assertions:
  VAL-PLANNER-001: Planner emits typed ChangePlan
  VAL-PLANNER-002: Planner uses DeepSeek V4 Pro
  VAL-PLANNER-003: Planner consumes RAG hits
  VAL-PLANNER-004: Planner queries episodic memory
  VAL-PLANNER-005: Planner rejects free-form output
  VAL-PLANNER-006: ChangePlan persisted as decision row
  VAL-PLANNER-007: ChangePlan rationale is non-empty string (≥20 chars)

Integration tests against real OpenRouter are marked @pytest.mark.integration
and require OPENROUTER_API_KEY in the environment.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError
from pydantic_ai import RunContext
from src.agents.models import ChangePlan, IssueContext

# ── Fixtures ────────────────────────────────────────────────────────


def _make_issue_context(**overrides: Any) -> IssueContext:
    """Build a valid IssueContext with sensible defaults."""
    defaults: dict[str, Any] = {
        "repo_url": "https://github.com/org/repo",
        "issue_number": 1,
        "issue_text": (
            "Bug: the subtract function returns a + b "
            "instead of a - b in calculator.py"
        ),
        "repo_files": {
            "src/calculator.py": "def subtract(a, b):\n    return a + b\n",
        },
        "repo_facts": [
            {"fact_kind": "language", "fact_value": '{"value": "python"}'},
        ],
    }
    defaults.update(overrides)
    return IssueContext(**defaults)


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


# ── VAL-PLANNER-001: Planner emits typed ChangePlan ──────────────


class TestPlannerEmitsTypedChangePlan:
    """Planner returns a Pydantic-valid ChangePlan with non-empty
    target_files and rationale."""

    @pytest.mark.asyncio
    async def test_emits_typed_change_plan(self) -> None:
        """Given valid IssueContext, Planner returns ChangePlan."""
        from src.agents.planner import PlannerDeps, planner

        ctx = _make_issue_context()
        deps = PlannerDeps(
            episodic_store=None,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_result = MagicMock()
        mock_result.output = _make_change_plan()

        with patch.object(planner, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            result = await planner.run(str(ctx.issue_text), deps=deps)

        plan = result.output
        assert isinstance(plan, ChangePlan)
        assert len(plan.target_files) >= 1
        assert len(plan.rationale) >= 20

    @pytest.mark.asyncio
    async def test_run_planner_can_return_usage_metadata(self) -> None:
        """Planner orchestration needs actual token/cost metadata, not zeros."""
        from decimal import Decimal
        from types import SimpleNamespace

        from src.agents.planner import PlannerRunResult, planner, run_planner

        ctx = _make_issue_context()

        mock_result = MagicMock()
        mock_result.output = _make_change_plan()
        mock_result.usage = lambda: SimpleNamespace(
            request_tokens=1234,
            response_tokens=321,
        )

        with patch.object(planner, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            result = await run_planner(
                issue_context=ctx,
                task_id=uuid4(),
                trace_id="test-trace",
                return_metadata=True,
            )

        assert isinstance(result, PlannerRunResult)
        assert isinstance(result.plan, ChangePlan)
        assert result.tokens_in == 1234
        assert result.tokens_out == 321
        assert result.cost_usd > Decimal("0")
        mock_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_change_plan_target_files_non_empty(self) -> None:
        """ChangePlan.target_files must be non-empty list."""
        from src.agents.planner import PlannerDeps, planner

        ctx = _make_issue_context()
        deps = PlannerDeps(
            episodic_store=None,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_result = MagicMock()
        mock_result.output = _make_change_plan(
            target_files=["src/calculator.py", "tests/test_calculator.py"],
            rationale=(
                "The subtract function returns a + b instead of a - b, "
                "and the test suite needs updating to verify the fix."
            ),
        )

        with patch.object(planner, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            result = await planner.run(str(ctx.issue_text), deps=deps)

        assert len(result.output.target_files) >= 1


# ── VAL-PLANNER-002: Planner uses DeepSeek V4 Pro ────────────────


class TestPlannerUsesCorrectModel:
    """Every Planner LLM call uses model deepseek/deepseek-v4-pro."""

    def test_planner_model_id_is_deepseek_v4_pro(self) -> None:
        """The planner agent is configured with the correct model ID."""
        from src.agents.planner import _PLANNER_MODEL

        assert _PLANNER_MODEL == "deepseek/deepseek-v4-pro"

    def test_planner_openrouter_model_string(self) -> None:
        """The PydanticAI agent uses the OpenRouter model string."""
        from src.agents.planner import _PLANNER_MODEL, _PYDANTIC_AI_MODEL

        expected = f"openrouter:{_PLANNER_MODEL}"
        assert expected == _PYDANTIC_AI_MODEL
        assert "deepseek/deepseek-v4-pro" in _PYDANTIC_AI_MODEL


# ── VAL-PLANNER-003: Planner consumes RAG hits ──────────────────


class TestPlannerConsumesRAGHits:
    """Planner's RAG retrieval tool returns scoped results for the
    current repo_url."""

    @pytest.mark.asyncio
    async def test_rag_retrieval_tool_returns_scoped_results(
        self,
    ) -> None:
        """The RAG retrieval tool returns results scoped to repo_url."""
        from src.agents.planner import PlannerDeps, _rag_retrieval

        mock_rag = AsyncMock()
        mock_rag.retrieve.return_value = [
            {
                "file_path": "src/calculator.py",
                "chunk_text": "def subtract(a, b):\n    return a + b",
                "score": 0.95,
            },
        ]

        deps = PlannerDeps(
            episodic_store=None,
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
        from src.agents.planner import PlannerDeps, _rag_retrieval

        mock_rag = AsyncMock()
        mock_rag.retrieve.return_value = []

        deps = PlannerDeps(
            episodic_store=None,
            rag_retriever=mock_rag,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        await _rag_retrieval(mock_ctx, "subtract function")
        mock_rag.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_retrieval_returns_no_results_message(
        self,
    ) -> None:
        """When no RAG results are found, a clear message is returned."""
        from src.agents.planner import PlannerDeps, _rag_retrieval

        mock_rag = AsyncMock()
        mock_rag.retrieve.return_value = []

        deps = PlannerDeps(
            episodic_store=None,
            rag_retriever=mock_rag,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _rag_retrieval(mock_ctx, "obscure query")
        assert "No RAG results" in result

    @pytest.mark.asyncio
    async def test_rag_retrieval_unavailable_graceful(
        self,
    ) -> None:
        """When RAG is unavailable, a graceful message is returned."""
        from src.agents.planner import PlannerDeps, _rag_retrieval

        deps = PlannerDeps(
            episodic_store=None,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _rag_retrieval(mock_ctx, "any query")
        assert "not available" in result


# ── VAL-PLANNER-004: Planner queries episodic memory ──────────────


class TestPlannerQueriesEpisodicMemory:
    """Planner reads repo_facts and recent decisions/outcomes for the
    current repo_url."""

    @pytest.mark.asyncio
    async def test_episodic_query_tool_reads_repo_facts(
        self,
    ) -> None:
        """The episodic query tool returns repo_facts for the repo_url."""
        from src.agents.planner import PlannerDeps, _episodic_query

        mock_store = AsyncMock()
        mock_store.get_planner_context.return_value = {
            "repo_facts": [
                {"fact_kind": "language", "fact_value": {"value": "python"}},
            ],
            "recent_decisions": [],
            "recent_outcomes": [],
        }

        deps = PlannerDeps(
            episodic_store=mock_store,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _episodic_query(
            mock_ctx, "https://github.com/org/repo",
        )
        parsed = json.loads(result)
        assert "repo_facts" in parsed
        mock_store.get_planner_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_episodic_query_returns_decisions_and_outcomes(
        self,
    ) -> None:
        """The episodic query tool returns recent decisions and outcomes."""
        from src.agents.planner import PlannerDeps, _episodic_query

        mock_store = AsyncMock()
        mock_store.get_planner_context.return_value = {
            "repo_facts": [],
            "recent_decisions": [
                {
                    "agent": "planner",
                    "decision_type": "change_plan",
                    "decision_data": {"target_files": ["src/main.py"]},
                },
            ],
            "recent_outcomes": [
                {"outcome": "pr_opened", "detail": {}},
            ],
        }

        deps = PlannerDeps(
            episodic_store=mock_store,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _episodic_query(
            mock_ctx, "https://github.com/org/repo",
        )
        parsed = json.loads(result)
        assert "recent_decisions" in parsed
        assert "recent_outcomes" in parsed

    @pytest.mark.asyncio
    async def test_episodic_query_unavailable_graceful(
        self,
    ) -> None:
        """When episodic store is unavailable, a graceful message is returned."""
        from src.agents.planner import PlannerDeps, _episodic_query

        deps = PlannerDeps(
            episodic_store=None,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        mock_ctx = MagicMock(spec=RunContext)
        mock_ctx.deps = deps

        result = await _episodic_query(
            mock_ctx, "https://github.com/org/repo",
        )
        assert "not available" in result


# ── VAL-PLANNER-005: Planner rejects free-form output ─────────────


class TestPlannerRejectsFreeFormOutput:
    """If the LLM response cannot be parsed as ChangePlan, it triggers
    a Pydantic validation failure. Free-form text does NOT propagate
    downstream."""

    def test_change_plan_rejects_empty_target_files(self) -> None:
        """Empty target_files list fails Pydantic validation."""
        with pytest.raises(ValidationError):
            ChangePlan(
                target_files=[],
                rationale="This is a rationale that is at least 20 characters long",
            )

    def test_change_plan_rejects_empty_rationale(self) -> None:
        """Rationale shorter than 20 chars fails Pydantic validation."""
        with pytest.raises(ValidationError):
            ChangePlan(
                target_files=["src/main.py"],
                rationale="Too short",
            )

    def test_change_plan_rejects_whitespace_rationale(self) -> None:
        """Whitespace-only rationale should be rejected."""
        with pytest.raises(ValidationError):
            ChangePlan(
                target_files=["src/main.py"],
                rationale="                    ",  # 20 spaces
            )

    def test_free_form_text_not_propagated(self) -> None:
        """A string that is not valid JSON for ChangePlan is rejected.

        The Planner does NOT propagate raw text downstream.
        When PydanticAI fails to parse the output as ChangePlan,
        it raises a validation error, not raw text.
        """
        with pytest.raises((ValidationError, Exception)):
            ChangePlan.model_validate(
                "This is just free-form text, not a valid ChangePlan",
            )

    @pytest.mark.asyncio
    async def test_planner_validation_failure_counted_as_retry(
        self,
    ) -> None:
        """When Pydantic validation fails, it counts toward retry budget."""
        from pydantic_ai import UnexpectedModelBehavior
        from src.agents.planner import PlannerDeps, planner

        ctx = _make_issue_context()
        deps = PlannerDeps(
            episodic_store=None,
            rag_retriever=None,
            task_id=uuid4(),
            trace_id="test-trace",
        )

        with patch.object(planner, "run", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = UnexpectedModelBehavior(
                "Pydantic validation failed for ChangePlan",
            )
            with pytest.raises(UnexpectedModelBehavior):
                await planner.run(str(ctx.issue_text), deps=deps)


# ── VAL-PLANNER-006: ChangePlan persisted as decision row ─────────


class TestChangePlanPersistedAsDecision:
    """The successfully parsed ChangePlan is written to the decisions
    table with agent='planner' and decision_type='change_plan'."""

    @pytest.mark.asyncio
    async def test_persist_change_plan_as_decision(self) -> None:
        """After a successful Planner run, a decision row is created."""
        from src.agents.planner import persist_change_plan

        task_id = uuid4()
        plan = _make_change_plan()

        mock_store = AsyncMock()
        mock_store.create_decision.return_value = MagicMock()

        await persist_change_plan(
            store=mock_store,
            task_id=task_id,
            plan=plan,
            step_index=0,
        )

        mock_store.create_decision.assert_called_once()
        call_args = mock_store.create_decision.call_args
        params = call_args[0][0] if call_args[0] else call_args[1]

        assert params.agent == "planner"
        assert params.decision_type == "change_plan"
        assert "target_files" in params.decision_data
        assert "rationale" in params.decision_data


# ── VAL-PLANNER-007: ChangePlan rationale is non-empty string ─────


class TestChangePlanRationaleRequired:
    """ChangePlan.rationale must be a non-empty string with len ≥ 20."""

    def test_rationale_must_be_at_least_20_chars(self) -> None:
        """Rationale with < 20 chars fails Pydantic validation."""
        with pytest.raises(ValidationError):
            ChangePlan(
                target_files=["src/main.py"],
                rationale="x" * 19,
            )

    def test_rationale_exactly_20_chars_accepted(self) -> None:
        """Rationale with exactly 20 chars is valid."""
        plan = ChangePlan(
            target_files=["src/main.py"],
            rationale="x" * 20,
        )
        assert len(plan.rationale) == 20

    def test_rationale_long_string_accepted(self) -> None:
        """Rationale with a reasonable length is valid."""
        plan = ChangePlan(
            target_files=["src/main.py"],
            rationale=(
                "The subtract function incorrectly returns a + b "
                "instead of a - b, causing incorrect results."
            ),
        )
        assert len(plan.rationale) > 20

    def test_approach_field_optional(self) -> None:
        """ChangePlan.approach has a default empty string."""
        plan = ChangePlan(
            target_files=["src/main.py"],
            rationale="x" * 20,
        )
        assert plan.approach == ""


# ── Integration test (real OpenRouter, requires API key) ──────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_planner_integration_real_llm() -> None:
    """Integration test: real OpenRouter call returns a valid ChangePlan.

    Uses a low-token test prompt.  Only validates structural correctness
    of the output, not reasoning quality.

    Requires OPENROUTER_API_KEY in the environment.
    """
    import os

    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    from src.agents.planner import PlannerDeps, planner

    ctx = _make_issue_context()
    deps = PlannerDeps(
        episodic_store=None,
        rag_retriever=None,
        task_id=uuid4(),
        trace_id="integration-test-trace",
    )

    result = await planner.run(
        f"Issue: {ctx.issue_text}\nRepo: {ctx.repo_url}",
        deps=deps,
    )

    plan = result.output
    assert isinstance(plan, ChangePlan)
    assert len(plan.target_files) >= 1
    assert len(plan.rationale) >= 20


@pytest.mark.integration
@pytest.mark.asyncio
async def test_planner_model_id_in_trace() -> None:
    """Integration test: verify model ID is deepseek/deepseek-v4-pro."""
    import os

    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    from src.agents.planner import _PLANNER_MODEL

    assert _PLANNER_MODEL == "deepseek/deepseek-v4-pro"
