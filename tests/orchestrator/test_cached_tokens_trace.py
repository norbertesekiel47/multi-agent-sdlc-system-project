"""Tests for m5-fix-cached-tokens-trace — blocking issue fix.

Covers:
  - _emit_trace_event accepts and passes cached_tokens parameter
  - All callers in supervisor_only.py pass actual cached_tokens
    from LLM results
  - All callers in hybrid.py pass actual cached_tokens
    from LLM results
  - WebSocket events for running tasks include non-zero
    cached_tokens when caching applies
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from src.agents.coder import CoderRunResult
from src.agents.models import ChangePlan, CodeEdit, ReviewResult, TestReport
from src.agents.qa import QARunResult
from src.agents.reviewer import ReviewerRunResult
from src.orchestrator import OrchestratorState

_SAMPLE_DIFF = (
    "--- a/src/calculator.py\n"
    "+++ b/src/calculator.py\n"
    "@@ -1,1 +1,1 @@\n"
    "-return a - b\n"
    "+return a + (-b)\n"
)


def _make_state(**overrides: object) -> OrchestratorState:
    """Create an OrchestratorState with sensible defaults for tests."""
    defaults = {
        "task_id": str(uuid4()),
        "repo_url": "https://github.com/example/test-repo",
        "issue_number": 1,
        "issue_text": "Bug in subtraction",
        "topology": "supervisor_only",
        "step_index": 0,
        "total_cost_usd": Decimal("0"),
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "total_tokens_cached": 0,
    }
    defaults.update(overrides)
    return OrchestratorState(**defaults)  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════
# Test 1: _emit_trace_event accepts and passes cached_tokens
# ══════════════════════════════════════════════════════════════════════


class TestEmitTraceEventCachedTokens:
    """_emit_trace_event must accept cached_tokens and pass it to
    TraceEvent."""

    @pytest.mark.asyncio
    async def test_emit_trace_event_accepts_cached_tokens_param(
        self,
    ) -> None:
        """_emit_trace_event signature includes cached_tokens parameter."""
        import inspect

        from src.orchestrator.supervisor_only import _emit_trace_event

        sig = inspect.signature(_emit_trace_event)
        assert "cached_tokens" in sig.parameters, (
            "_emit_trace_event must accept cached_tokens parameter"
        )
        # Default should be 0
        assert sig.parameters["cached_tokens"].default == 0

    @pytest.mark.asyncio
    async def test_emit_trace_event_passes_cached_tokens_to_trace_event(
        self,
    ) -> None:
        """When cached_tokens=500 is passed, TraceEvent.cached_tokens
        is 500."""
        from src.orchestrator.supervisor_only import _emit_trace_event
        from src.tracing.ws_broadcaster import get_trace_broadcaster

        broadcaster = get_trace_broadcaster()
        queue = await broadcaster.subscribe("test-task-ct")

        await _emit_trace_event(
            task_id="test-task-ct",
            trace_id="tr1",
            span_id="s1",
            parent_span_id=None,
            name="coder",
            event_type="node_end",
            tokens_in=1000,
            tokens_out=200,
            cached_tokens=500,
            cost_usd=Decimal("0.03"),
        )

        msg = queue.get_nowait()
        data = json.loads(msg)
        assert data["cached_tokens"] == 500, (
            "TraceEvent must carry the cached_tokens value"
        )
        assert data["tokens_in"] == 1000
        assert data["tokens_out"] == 200

        await broadcaster.unsubscribe("test-task-ct", queue)

    @pytest.mark.asyncio
    async def test_emit_trace_event_default_cached_tokens_zero(
        self,
    ) -> None:
        """When cached_tokens is not passed, TraceEvent.cached_tokens
        defaults to 0."""
        from src.orchestrator.supervisor_only import _emit_trace_event
        from src.tracing.ws_broadcaster import get_trace_broadcaster

        broadcaster = get_trace_broadcaster()
        queue = await broadcaster.subscribe("test-task-default-ct")

        await _emit_trace_event(
            task_id="test-task-default-ct",
            trace_id="tr1",
            span_id="s1",
            parent_span_id=None,
            name="supervisor",
            event_type="node_start",
        )

        msg = queue.get_nowait()
        data = json.loads(msg)
        assert data["cached_tokens"] == 0, (
            "TraceEvent.cached_tokens must default to 0 when not passed"
        )

        await broadcaster.unsubscribe("test-task-default-ct", queue)


# ══════════════════════════════════════════════════════════════════════
# Test 2: supervisor_only.py callers pass actual cached_tokens
# ══════════════════════════════════════════════════════════════════════


class TestSupervisorOnlyCachedTokensInTraceEvents:
    """Node functions in supervisor_only.py must pass cached_tokens
    from LLM results to _emit_trace_event for node_end events."""

    @pytest.mark.asyncio
    async def test_coder_node_end_passes_cached_tokens(self) -> None:
        """run_coder passes cached_tokens from CoderRunResult."""
        from contextlib import ExitStack

        from src.orchestrator.supervisor_only import run_coder

        state = _make_state(
            change_plan=ChangePlan(
                target_files=["src/calculator.py"],
                rationale="Fix the subtraction logic",
                approach="Fix the subtraction logic",
            ),
            trace_id="test-trace",
            supervisor_span_id="sup-span",
        )

        mock_edit = CodeEdit(
            diff=_SAMPLE_DIFF,
            touched_files=["src/calculator.py"],
            diff_hash="abc123",
        )
        mock_coder_result = CoderRunResult(
            edit=mock_edit,
            tokens_in=1500,
            tokens_out=500,
            cached_tokens=800,
            cost_usd=Decimal("0.05"),
        )

        mock_tracing = MagicMock()
        mock_tracing.create_span.return_value = "coder-span"
        mock_emit = AsyncMock()

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_tracing_client",
                    return_value=mock_tracing,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_sandbox",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_store",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_sandbox_proxy",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_recording_proxy",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only._emit_trace_event",
                    new_callable=AsyncMock,
                )
            )
            mock_emit = stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only._emit_trace_event",
                    new_callable=AsyncMock,
                )
            )
            stack.enter_context(
                patch(
                    "src.agents.coder.run_coder",
                    return_value=mock_coder_result,
                )
            )
            await run_coder(state)

        node_end_calls = [
            c
            for c in mock_emit.call_args_list
            if c.kwargs.get("event_type") == "node_end"
            and c.kwargs.get("name") == "coder"
        ]
        assert len(node_end_calls) >= 1
        assert node_end_calls[0].kwargs.get("cached_tokens") == 800, (
            "run_coder node_end must pass cached_tokens=800"
        )

    @pytest.mark.asyncio
    async def test_reviewer_node_end_passes_cached_tokens(self) -> None:
        """run_reviewer passes cached_tokens from ReviewerRunResult."""
        from contextlib import ExitStack

        from src.orchestrator.supervisor_only import run_reviewer

        state = _make_state(
            code_edit=CodeEdit(
                diff=_SAMPLE_DIFF,
                touched_files=["src/calculator.py"],
                diff_hash="abc123",
            ),
            trace_id="test-trace",
            supervisor_span_id="sup-span",
        )

        mock_review = ReviewResult(
            verdict="accept",
            issues=[],
            diff_hash="abc123",
        )
        mock_reviewer_result = ReviewerRunResult(
            review=mock_review,
            tokens_in=2000,
            tokens_out=300,
            cached_tokens=1200,
            cost_usd=Decimal("0.03"),
        )

        mock_tracing = MagicMock()
        mock_tracing.create_span.return_value = "reviewer-span"

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_tracing_client",
                    return_value=mock_tracing,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_sandbox",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_store",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_sandbox_proxy",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_recording_proxy",
                    return_value=None,
                )
            )
            mock_emit = stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only._emit_trace_event",
                    new_callable=AsyncMock,
                )
            )
            stack.enter_context(
                patch(
                    "src.agents.reviewer.run_reviewer",
                    return_value=mock_reviewer_result,
                )
            )
            await run_reviewer(state)

        node_end_calls = [
            c
            for c in mock_emit.call_args_list
            if c.kwargs.get("event_type") == "node_end"
            and c.kwargs.get("name") == "reviewer"
        ]
        assert len(node_end_calls) >= 1
        assert node_end_calls[0].kwargs.get("cached_tokens") == 1200, (
            "run_reviewer node_end must pass cached_tokens=1200"
        )

    @pytest.mark.asyncio
    async def test_qa_node_end_passes_cached_tokens(self) -> None:
        """run_qa passes cached_tokens from QARunResult."""
        from contextlib import ExitStack

        from src.orchestrator.supervisor_only import run_qa

        state = _make_state(
            code_edit=CodeEdit(
                diff=_SAMPLE_DIFF,
                touched_files=["src/calculator.py"],
                diff_hash="abc123",
            ),
            trace_id="test-trace",
            supervisor_span_id="sup-span",
        )

        mock_report = TestReport(
            passed=5,
            failed=0,
            failed_test_names=[],
            generated_test_files=["tests/test_calc.py"],
            generated_test_contents={"tests/test_calc.py": "def test_add(): pass"},
        )
        mock_qa_result = QARunResult(
            report=mock_report,
            tokens_in=1800,
            tokens_out=400,
            cached_tokens=600,
            cost_usd=Decimal("0.04"),
        )

        mock_tracing = MagicMock()
        mock_tracing.create_span.return_value = "qa-span"

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_tracing_client",
                    return_value=mock_tracing,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_sandbox",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_store",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_sandbox_proxy",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only.get_recording_proxy",
                    return_value=None,
                )
            )
            mock_emit = stack.enter_context(
                patch(
                    "src.orchestrator.supervisor_only._emit_trace_event",
                    new_callable=AsyncMock,
                )
            )
            stack.enter_context(
                patch(
                    "src.agents.qa.run_qa",
                    return_value=mock_qa_result,
                )
            )
            await run_qa(state)

        node_end_calls = [
            c
            for c in mock_emit.call_args_list
            if c.kwargs.get("event_type") == "node_end"
            and c.kwargs.get("name") == "qa"
        ]
        assert len(node_end_calls) >= 1
        assert node_end_calls[0].kwargs.get("cached_tokens") == 600, (
            "run_qa node_end must pass cached_tokens=600"
        )


# ══════════════════════════════════════════════════════════════════════
# Test 3: hybrid.py callers pass actual cached_tokens
# ══════════════════════════════════════════════════════════════════════


class TestHybridCachedTokensInTraceEvents:
    """Node functions in hybrid.py must pass cached_tokens from
    LLM results to _emit_trace_event for node_end events."""

    @pytest.mark.asyncio
    async def test_hybrid_reviewer_node_end_passes_cached_tokens(
        self,
    ) -> None:
        """run_reviewer_hybrid passes cached_tokens from
        ReviewerRunResult."""
        from contextlib import ExitStack

        from src.orchestrator.hybrid import run_reviewer_hybrid

        state = _make_state(
            code_edit=CodeEdit(
                diff=_SAMPLE_DIFF,
                touched_files=["src/calculator.py"],
                diff_hash="abc123",
            ),
            trace_id="test-trace",
            supervisor_span_id="sup-span",
            peer_handoff_count=0,
        )

        mock_review = ReviewResult(
            verdict="accept",
            issues=[],
            diff_hash="abc123",
        )
        mock_reviewer_result = ReviewerRunResult(
            review=mock_review,
            tokens_in=2000,
            tokens_out=300,
            cached_tokens=1200,
            cost_usd=Decimal("0.03"),
        )

        mock_tracing = MagicMock()
        mock_tracing.create_span.return_value = "reviewer-span"

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "src.orchestrator.hybrid.get_tracing_client",
                    return_value=mock_tracing,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.hybrid.get_sandbox",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.hybrid.get_store",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.hybrid.get_sandbox_proxy",
                    return_value=None,
                )
            )
            mock_emit = stack.enter_context(
                patch(
                    "src.orchestrator.hybrid._emit_trace_event",
                    new_callable=AsyncMock,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.hybrid._run_reviewer_agent",
                    return_value=mock_reviewer_result,
                )
            )
            await run_reviewer_hybrid(state)

        node_end_calls = [
            c
            for c in mock_emit.call_args_list
            if c.kwargs.get("event_type") == "node_end"
            and c.kwargs.get("name") == "reviewer"
        ]
        assert len(node_end_calls) >= 1
        assert node_end_calls[0].kwargs.get("cached_tokens") == 1200, (
            "run_reviewer_hybrid node_end must pass cached_tokens=1200"
        )

    @pytest.mark.asyncio
    async def test_hybrid_peer_coder_node_end_passes_cached_tokens(
        self,
    ) -> None:
        """run_peer_coder passes cached_tokens from CoderRunResult."""
        from contextlib import ExitStack

        from src.orchestrator.hybrid import run_peer_coder

        state = _make_state(
            change_plan=ChangePlan(
                target_files=["src/calculator.py"],
                rationale="Fix the subtraction logic",
                approach="Fix the subtraction logic",
            ),
            review_result=ReviewResult(
                verdict="reject_with_changes",
                issues=["Missing error handling"],
                diff_hash="abc123",
            ),
            trace_id="test-trace",
            supervisor_span_id="sup-span",
            step_index=1,
            peer_handoff_count=0,
            last_reviewer_span_id="reviewer-span",
        )

        mock_edit = CodeEdit(
            diff=_SAMPLE_DIFF,
            touched_files=["src/calculator.py"],
            diff_hash="def456",
        )
        mock_coder_result = CoderRunResult(
            edit=mock_edit,
            tokens_in=1500,
            tokens_out=500,
            cached_tokens=800,
            cost_usd=Decimal("0.05"),
        )

        mock_tracing = MagicMock()
        mock_tracing.create_span.return_value = "coder-span"

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "src.orchestrator.hybrid.get_tracing_client",
                    return_value=mock_tracing,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.hybrid.get_sandbox",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.hybrid.get_store",
                    return_value=None,
                )
            )
            stack.enter_context(
                patch(
                    "src.orchestrator.hybrid.get_sandbox_proxy",
                    return_value=None,
                )
            )
            mock_emit = stack.enter_context(
                patch(
                    "src.orchestrator.hybrid._emit_trace_event",
                    new_callable=AsyncMock,
                )
            )
            stack.enter_context(
                patch(
                    "src.agents.coder.run_coder",
                    return_value=mock_coder_result,
                )
            )
            await run_peer_coder(state)

        node_end_calls = [
            c
            for c in mock_emit.call_args_list
            if c.kwargs.get("event_type") == "node_end"
            and c.kwargs.get("name") == "coder"
        ]
        assert len(node_end_calls) >= 1
        assert node_end_calls[0].kwargs.get("cached_tokens") == 800, (
            "run_peer_coder node_end must pass cached_tokens=800"
        )


# ══════════════════════════════════════════════════════════════════════
# Test 4: End-to-end WS event includes non-zero cached_tokens
# ══════════════════════════════════════════════════════════════════════


class TestWebSocketEventCachedTokens:
    """WebSocket events for running tasks include non-zero
    cached_tokens when caching applies."""

    @pytest.mark.asyncio
    async def test_coder_trace_event_broadcasts_nonzero_cached_tokens(
        self,
    ) -> None:
        """When _emit_trace_event is called with cached_tokens>0,
        the broadcast JSON includes that value."""
        from src.orchestrator.supervisor_only import _emit_trace_event
        from src.tracing.ws_broadcaster import get_trace_broadcaster

        broadcaster = get_trace_broadcaster()
        task_id = "test-ws-cached-tokens"
        queue = await broadcaster.subscribe(task_id)

        await _emit_trace_event(
            task_id=task_id,
            trace_id="tr1",
            span_id="s1",
            parent_span_id=None,
            name="coder",
            event_type="node_end",
            tokens_in=1500,
            tokens_out=500,
            cached_tokens=800,
            cost_usd=Decimal("0.05"),
            metadata={"agent_name": "coder"},
        )

        msg = queue.get_nowait()
        data = json.loads(msg)

        # Verify the contract: cached_tokens is top-level, non-zero
        assert "cached_tokens" in data
        assert data["cached_tokens"] == 800
        assert data["tokens_in"] == 1500
        assert data["tokens_out"] == 500
        assert data["cost_usd"] == 0.05  # float, not Decimal string

        await broadcaster.unsubscribe(task_id, queue)
