"""Tests for m5-fix-backend-frontend-contract — three blocking issues.

Covers:
  1. WS trace events include cost_usd as float (not Decimal string),
     top-level agent field, and cached_tokens.
  2. GET /tasks/{id} returns trace_history array populated from stored spans.
  3. Repo substring filter applied in SQL query (not client-side),
     pagination works correctly with repo filter.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.memory.episodic.models import (
    CreateDecisionParams,
    CreateOutcomeParams,
    CreateTaskParams,
    TaskRow,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_mock_store() -> AsyncMock:
    """Create a mock EpisodicStore that doesn't need a real DB connection."""
    store = AsyncMock()
    store._pool = MagicMock()  # truthy to pass the pool check
    store.create_task = AsyncMock(return_value=TaskRow(
        id=uuid4(),
        repo_url="https://github.com/test/repo",
        issue_number=1,
        issue_text="Fix the bug",
        topology="hybrid",
        status="running",
        started_at=datetime.now(UTC),
    ))
    store.get_task = AsyncMock(return_value=None)  # no task found by default
    store.list_tasks = AsyncMock(return_value=[])
    store.get_latest_outcomes = AsyncMock(return_value={})
    store.get_trace_history = AsyncMock(return_value=[])
    store.get_decisions_for_task = AsyncMock(return_value=[])
    store.get_outcomes_for_task = AsyncMock(return_value=[])
    store.update_task_totals = AsyncMock()
    store.close = AsyncMock()
    return store


@pytest.fixture
def mock_store() -> AsyncMock:
    """Mock EpisodicStore for sync TestClient tests."""
    return _make_mock_store()


@pytest.fixture
def client(mock_store: AsyncMock) -> TestClient:
    """FastAPI test client with mocked store dependency."""
    from src.api.main import get_store
    app.dependency_overrides[get_store] = lambda: mock_store
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()


# ══════════════════════════════════════════════════════════════════════
# Fix 1: WS trace events — cost_usd as float, top-level agent, cached_tokens
# ══════════════════════════════════════════════════════════════════════


class TestTraceEventCostUsdIsFloat:
    """cost_usd must be a float in the WS JSON payload, not a Decimal string."""

    def test_cost_usd_serialized_as_float(self) -> None:
        """cost_usd=Decimal('0.0100') serializes as float 0.01, not '0.0100'."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="llm_completion",
            task_id="t1",
            trace_id="tr1",
            span_id="s1",
            name="coder.llm_call",
            cost_usd=Decimal("0.0100"),
        )
        data = json.loads(event.to_json())
        assert isinstance(data["cost_usd"], float), (
            f"cost_usd should be float, got {type(data['cost_usd']).__name__}: {data['cost_usd']!r}"
        )
        assert data["cost_usd"] == 0.01

    def test_cost_usd_zero_serialized_as_float(self) -> None:
        """cost_usd=Decimal('0') serializes as float 0.0."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="node_start",
            task_id="t1",
            trace_id="tr1",
            span_id="s1",
            name="planner",
            cost_usd=Decimal("0"),
        )
        data = json.loads(event.to_json())
        assert isinstance(data["cost_usd"], float)
        assert data["cost_usd"] == 0.0

    def test_cost_usd_large_value_serialized_as_float(self) -> None:
        """Large Decimal values serialize as float without string wrapping."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="llm_completion",
            task_id="t1",
            trace_id="tr1",
            span_id="s1",
            name="coder.llm_call",
            cost_usd=Decimal("1.2345"),
        )
        data = json.loads(event.to_json())
        assert isinstance(data["cost_usd"], float)
        assert data["cost_usd"] == 1.2345


class TestTraceEventTopLevelAgent:
    """agent must be a top-level field, not nested in metadata.agent_name."""

    def test_agent_extracted_from_metadata_agent_name(self) -> None:
        """metadata.agent_name → top-level agent field."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="node_start",
            task_id="t1",
            trace_id="tr1",
            span_id="s1",
            name="planner",
            metadata={"agent_name": "planner"},
        )
        data = json.loads(event.to_json())
        assert data["agent"] == "planner"
        # metadata.agent_name still preserved for backward compat
        assert data["metadata"]["agent_name"] == "planner"

    def test_agent_extracted_from_metadata_agent(self) -> None:
        """metadata.agent → top-level agent field (LLM completion pattern)."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="llm_completion",
            task_id="t1",
            trace_id="tr1",
            span_id="s1",
            name="coder.llm_call",
            metadata={"agent": "coder", "model": "deepseek/deepseek-chat-v3-0324"},
        )
        data = json.loads(event.to_json())
        assert data["agent"] == "coder"

    def test_agent_none_when_no_metadata(self) -> None:
        """No metadata → agent is None."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="ping",
            task_id="t1",
            trace_id="tr1",
            span_id="s1",
            name="ping",
        )
        data = json.loads(event.to_json())
        assert data["agent"] is None

    def test_agent_none_when_metadata_has_no_agent_name(self) -> None:
        """Metadata without agent_name or agent → agent is None."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="tool_call",
            task_id="t1",
            trace_id="tr1",
            span_id="s1",
            name="sandbox.read_file",
            metadata={"tool": "read_file"},
        )
        data = json.loads(event.to_json())
        assert data["agent"] is None

    def test_agent_name_preferred_over_agent(self) -> None:
        """When both metadata.agent_name and metadata.agent exist, agent_name wins."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="node_start",
            task_id="t1",
            trace_id="tr1",
            span_id="s1",
            name="planner",
            metadata={"agent_name": "planner", "agent": "other"},
        )
        data = json.loads(event.to_json())
        assert data["agent"] == "planner"


class TestTraceEventCachedTokens:
    """cached_tokens must be present as a top-level field in WS events."""

    def test_cached_tokens_present_in_event(self) -> None:
        """cached_tokens field is present in serialized output."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="llm_completion",
            task_id="t1",
            trace_id="tr1",
            span_id="s1",
            name="coder.llm_call",
            cached_tokens=200,
        )
        data = json.loads(event.to_json())
        assert "cached_tokens" in data
        assert data["cached_tokens"] == 200

    def test_cached_tokens_zero_when_not_cached(self) -> None:
        """cached_tokens defaults to 0."""
        from src.tracing.ws_broadcaster import TraceEvent

        event = TraceEvent(
            type="node_start",
            task_id="t1",
            trace_id="tr1",
            span_id="s1",
            name="planner",
        )
        data = json.loads(event.to_json())
        assert data["cached_tokens"] == 0


# ══════════════════════════════════════════════════════════════════════
# Fix 2: trace_history in GET /tasks/{id}
# ══════════════════════════════════════════════════════════════════════


class TestTraceHistoryInTaskDetail:
    """GET /tasks/{id} returns trace_history array populated from stored spans."""

    def test_trace_history_field_present_in_response(
        self, client: TestClient, mock_store: AsyncMock,
    ) -> None:
        """TaskDetailResponse includes trace_history field."""
        task_id = uuid4()
        now = datetime.now(UTC)
        mock_store.get_task.return_value = TaskRow(
            id=task_id,
            repo_url="https://github.com/test/repo",
            issue_number=1,
            issue_text="Fix the bug",
            topology="hybrid",
            status="running",
            started_at=now,
        )
        mock_store.get_trace_history.return_value = []
        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert "trace_history" in body

    def test_trace_history_populated_from_decisions(
        self, client: TestClient, mock_store: AsyncMock,
    ) -> None:
        """trace_history contains events reconstructed from decisions."""
        task_id = uuid4()
        now = datetime.now(UTC)
        mock_store.get_task.return_value = TaskRow(
            id=task_id,
            repo_url="https://github.com/test/repo",
            issue_number=1,
            issue_text="Fix the bug",
            topology="hybrid",
            status="running",
            started_at=now,
        )
        mock_store.get_trace_history.return_value = [
            {
                "type": "node_end",
                "task_id": str(task_id),
                "trace_id": str(task_id),
                "span_id": str(uuid4()),
                "parent_span_id": None,
                "name": "planner",
                "span_type": "span",
                "started_at": now.isoformat(),
                "ended_at": now.isoformat(),
                "tokens_in": 100,
                "tokens_out": 50,
                "cached_tokens": 0,
                "cost_usd": 0.01,
                "status": "ok",
                "agent": "planner",
                "metadata": {"decision_type": "change_plan"},
            },
        ]
        resp = client.get(f"/tasks/{task_id}")
        body = resp.json()
        assert len(body["trace_history"]) == 1
        assert body["trace_history"][0]["agent"] == "planner"
        assert isinstance(body["trace_history"][0]["cost_usd"], float)

    def test_trace_history_empty_for_fresh_task(
        self, client: TestClient, mock_store: AsyncMock,
    ) -> None:
        """Fresh task with no decisions returns empty trace_history."""
        task_id = uuid4()
        now = datetime.now(UTC)
        mock_store.get_task.return_value = TaskRow(
            id=task_id,
            repo_url="https://github.com/test/repo",
            issue_number=1,
            issue_text="Fix the bug",
            topology="hybrid",
            status="running",
            started_at=now,
        )
        mock_store.get_trace_history.return_value = []
        resp = client.get(f"/tasks/{task_id}")
        body = resp.json()
        assert body["trace_history"] == []


class TestGetTraceHistoryMethod:
    """EpisodicStore.get_trace_history reconstructs trace events from DB."""

    @pytest.mark.asyncio
    async def test_trace_history_from_decisions_and_outcomes(self) -> None:
        """get_trace_history returns events from decisions + outcomes."""
        from src.memory.episodic.store import EpisodicStore

        dsn = (
            f"postgresql://{__import__('os').getenv('POSTGRES_USER', 'sdlc_swarm')}"
            f":{__import__('os').getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
            f"@{__import__('os').getenv('POSTGRES_HOST', 'localhost')}"
            f":{__import__('os').getenv('POSTGRES_PORT', '5433')}"
            f"/{__import__('os').getenv('POSTGRES_DB', 'sdlc_swarm')}"
        )

        async with EpisodicStore(dsn=dsn) as store:
            # Create a task
            params = CreateTaskParams(
                repo_url=f"https://github.com/test/trace-hist-{uuid4().hex[:8]}",
                issue_number=1,
                issue_text="Test trace history",
                topology="hybrid",
            )
            task = await store.create_task(params)

            # Add a decision
            await store.create_decision(
                CreateDecisionParams(
                    task_id=task.id,
                    agent="planner",
                    step_index=0,
                    decision_type="change_plan",
                    decision_data={
                        "target_files": ["src/main.py"],
                        "rationale": "Fix the bug",
                    },
                )
            )

            # Add an outcome
            await store.create_outcome(
                CreateOutcomeParams(
                    task_id=task.id,
                    outcome="pr_opened",
                    detail={"pr_url": "https://github.com/test/repo/pull/1"},
                )
            )

            # Get trace history
            history = await store.get_trace_history(task.id)
            assert len(history) >= 2  # at least decision + outcome events

            # Check the decision event
            decision_events = [e for e in history if e["type"] == "node_end"]
            assert len(decision_events) >= 1
            de = decision_events[0]
            assert de["agent"] == "planner"
            assert de["cost_usd"] == 0.0  # no cost in decision_data
            assert isinstance(de["cost_usd"], float)

            # Check the outcome event
            outcome_events = [e for e in history if e["type"] == "task_complete"]
            assert len(outcome_events) >= 1

    @pytest.mark.asyncio
    async def test_trace_history_hitl_interrupt_events(self) -> None:
        """Loop/uncertainty outcomes produce hitl_interrupt events in trace_history."""
        from src.memory.episodic.store import EpisodicStore

        dsn = (
            f"postgresql://{__import__('os').getenv('POSTGRES_USER', 'sdlc_swarm')}"
            f":{__import__('os').getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
            f"@{__import__('os').getenv('POSTGRES_HOST', 'localhost')}"
            f":{__import__('os').getenv('POSTGRES_PORT', '5433')}"
            f"/{__import__('os').getenv('POSTGRES_DB', 'sdlc_swarm')}"
        )

        async with EpisodicStore(dsn=dsn) as store:
            params = CreateTaskParams(
                repo_url=f"https://github.com/test/trace-hitl-{uuid4().hex[:8]}",
                issue_number=1,
                issue_text="Test hitl trace",
                topology="hybrid",
            )
            task = await store.create_task(params)

            # Add a loop_detected outcome
            await store.create_outcome(
                CreateOutcomeParams(
                    task_id=task.id,
                    outcome="loop_detected",
                    detail={"tool": "sandbox.run_command", "count": 3},
                )
            )

            history = await store.get_trace_history(task.id)
            hitl_events = [e for e in history if e["type"] == "hitl_interrupt"]
            assert len(hitl_events) >= 1
            assert hitl_events[0]["name"] == "loop_detected"
            assert hitl_events[0]["metadata"]["cause"] == "loop_detected"


# ══════════════════════════════════════════════════════════════════════
# Fix 3: Repo ILIKE filter in SQL query (not client-side)
# ══════════════════════════════════════════════════════════════════════


class TestRepoFilterInSQL:
    """Repo substring filter must be applied in SQL, not client-side."""

    @pytest.mark.asyncio
    async def test_list_tasks_with_repo_filter(self) -> None:
        """list_tasks(repo='foo') applies ILIKE filter in SQL."""
        from src.memory.episodic.store import EpisodicStore

        dsn = (
            f"postgresql://{__import__('os').getenv('POSTGRES_USER', 'sdlc_swarm')}"
            f":{__import__('os').getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
            f"@{__import__('os').getenv('POSTGRES_HOST', 'localhost')}"
            f":{__import__('os').getenv('POSTGRES_PORT', '5433')}"
            f"/{__import__('os').getenv('POSTGRES_DB', 'sdlc_swarm')}"
        )

        async with EpisodicStore(dsn=dsn) as store:
            ns = uuid4().hex[:8]
            # Create two tasks with different repos
            t1 = await store.create_task(
                CreateTaskParams(
                    repo_url=f"https://github.com/test/{ns}-alpha-repo",
                    issue_number=1,
                    issue_text="Task A",
                    topology="hybrid",
                )
            )
            t2 = await store.create_task(
                CreateTaskParams(
                    repo_url=f"https://github.com/test/{ns}-beta-repo",
                    issue_number=2,
                    issue_text="Task B",
                    topology="hybrid",
                )
            )

            # Filter by substring "alpha" — should only match t1
            results = await store.list_tasks(repo=f"{ns}-alpha", limit=1000)
            ids = {r.id for r in results}
            assert t1.id in ids
            assert t2.id not in ids

            # Filter by substring ns — should match both
            results_all = await store.list_tasks(repo=ns, limit=1000)
            ids_all = {r.id for r in results_all}
            assert t1.id in ids_all
            assert t2.id in ids_all

    @pytest.mark.asyncio
    async def test_list_tasks_repo_filter_case_insensitive(self) -> None:
        """Repo filter is case-insensitive (ILIKE)."""
        from src.memory.episodic.store import EpisodicStore

        dsn = (
            f"postgresql://{__import__('os').getenv('POSTGRES_USER', 'sdlc_swarm')}"
            f":{__import__('os').getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
            f"@{__import__('os').getenv('POSTGRES_HOST', 'localhost')}"
            f":{__import__('os').getenv('POSTGRES_PORT', '5433')}"
            f"/{__import__('os').getenv('POSTGRES_DB', 'sdlc_swarm')}"
        )

        async with EpisodicStore(dsn=dsn) as store:
            ns = uuid4().hex[:8]
            t1 = await store.create_task(
                CreateTaskParams(
                    repo_url=f"https://github.com/TEST/{ns}-MyRepo",
                    issue_number=1,
                    issue_text="Case test",
                    topology="hybrid",
                )
            )

            # Search with lowercase substring
            results = await store.list_tasks(repo=f"{ns}-myrepo", limit=1000)
            ids = {r.id for r in results}
            assert t1.id in ids

    @pytest.mark.asyncio
    async def test_list_tasks_repo_filter_pagination(self) -> None:
        """Repo filter with LIMIT/OFFSET works correctly (filter applied before pagination)."""
        from src.memory.episodic.store import EpisodicStore

        dsn = (
            f"postgresql://{__import__('os').getenv('POSTGRES_USER', 'sdlc_swarm')}"
            f":{__import__('os').getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
            f"@{__import__('os').getenv('POSTGRES_HOST', 'localhost')}"
            f":{__import__('os').getenv('POSTGRES_PORT', '5433')}"
            f"/{__import__('os').getenv('POSTGRES_DB', 'sdlc_swarm')}"
        )

        async with EpisodicStore(dsn=dsn) as store:
            ns = uuid4().hex[:8]
            # Create 3 matching tasks and 2 non-matching
            match_ids = []
            for i in range(3):
                t = await store.create_task(
                    CreateTaskParams(
                        repo_url=f"https://github.com/test/{ns}-match-repo-{i}",
                        issue_number=i,
                        issue_text=f"Match {i}",
                        topology="hybrid",
                    )
                )
                match_ids.append(t.id)

            for i in range(2):
                await store.create_task(
                    CreateTaskParams(
                        repo_url=f"https://github.com/test/{ns}-nomatch-zz-{i}",
                        issue_number=100 + i,
                        issue_text=f"No match {i}",
                        topology="hybrid",
                    )
                )

            # With limit=2 and repo filter, we should get only matching tasks
            page1 = await store.list_tasks(repo=f"{ns}-match", limit=2, offset=0)
            assert len(page1) == 2
            assert all(r.id in match_ids for r in page1)

            page2 = await store.list_tasks(repo=f"{ns}-match", limit=2, offset=2)
            assert len(page2) == 1
            assert page2[0].id in match_ids

    @pytest.mark.asyncio
    async def test_list_tasks_empty_repo_filter_returns_all(self) -> None:
        """Empty/whitespace-only repo filter does not add an ILIKE condition."""
        from src.memory.episodic.store import EpisodicStore

        dsn = (
            f"postgresql://{__import__('os').getenv('POSTGRES_USER', 'sdlc_swarm')}"
            f":{__import__('os').getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
            f"@{__import__('os').getenv('POSTGRES_HOST', 'localhost')}"
            f":{__import__('os').getenv('POSTGRES_PORT', '5433')}"
            f"/{__import__('os').getenv('POSTGRES_DB', 'sdlc_swarm')}"
        )

        async with EpisodicStore(dsn=dsn) as store:
            ns = uuid4().hex[:8]
            t1 = await store.create_task(
                CreateTaskParams(
                    repo_url=f"https://github.com/test/{ns}-repo",
                    issue_number=1,
                    issue_text="Task",
                    topology="hybrid",
                )
            )

            # Empty string repo filter should not filter — results should
            # include our task when scoped to our namespace
            results_empty = await store.list_tasks(repo="", limit=1000)
            results_ws = await store.list_tasks(repo="   ", limit=1000)
            results_none = await store.list_tasks(limit=1000)

            # All three should produce the same count (no repo filter applied)
            # Verify by checking that our task appears when we use repo_url exact match
            results_scoped = await store.list_tasks(
                repo_url=f"https://github.com/test/{ns}-repo", limit=1000,
            )
            ids_scoped = {r.id for r in results_scoped}
            assert t1.id in ids_scoped

            # Verify empty/whitespace repo does not narrow results vs no-repo
            # (same tasks for our namespace)
            ids_empty = {r.id for r in results_empty if ns in r.repo_url}
            ids_ws = {r.id for r in results_ws if ns in r.repo_url}
            ids_none = {r.id for r in results_none if ns in r.repo_url}
            assert ids_empty == ids_none
            assert ids_ws == ids_none


class TestListTasksAPIRepoFilter:
    """GET /tasks?repo=... passes repo to SQL filter."""

    def test_repo_filter_passed_to_store(
        self, client: TestClient, mock_store: AsyncMock,
    ) -> None:
        """The repo query parameter is passed to store.list_tasks."""
        mock_store.list_tasks.return_value = []
        resp = client.get("/tasks?repo=test-org")
        assert resp.status_code == 200
        # Verify store.list_tasks was called with repo parameter
        call_args = mock_store.list_tasks.call_args
        assert call_args is not None
        kwargs = call_args.kwargs if call_args.kwargs else {}
        assert kwargs.get("repo") == "test-org"
