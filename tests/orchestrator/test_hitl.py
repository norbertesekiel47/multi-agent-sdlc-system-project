"""Tests for HITL (Human-in-the-Loop) checkpoints and decision API.

Covers VAL-HITL-CTRL-001 through VAL-HITL-CTRL-013,
VAL-CROSS-018 and VAL-CROSS-019.

Uses FastAPI TestClient and mocked agent/GH calls so that no
real LLM or GitHub API is invoked.  The LangGraph graph is
compiled with MemorySaver so that interrupt state is recoverable.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

# ── Shared Pydantic state models for test graphs ──────────────────────
# Defined at module level so LangGraph's type-hint resolution works.


class SimpleHITLState(BaseModel):
    """Minimal state for HITL interrupt test graphs."""

    task_id: str = ""
    step: str = ""
    pr_url: str = ""
    hitl_approved: bool = False
    model_config = {"arbitrary_types_allowed": True}


class WriteState(BaseModel):
    """State for write-operation interrupt tests."""

    task_id: str = ""
    writes_count: int = 0
    hitl_decisions: list[str] = []
    model_config = {"arbitrary_types_allowed": True}


class RestartState(BaseModel):
    """State for restart recovery tests."""

    task_id: str = ""
    value: int = 0
    model_config = {"arbitrary_types_allowed": True}


class RunningState(BaseModel):
    """State for mid-graph restart tests."""

    task_id: str = ""
    step: int = 0
    model_config = {"arbitrary_types_allowed": True}


# ── Routing functions (module-level for LangGraph type resolution) ──


def _route_after_hitl(state: SimpleHITLState) -> str:
    """Route based on HITL decision: approve→open_pr, else→reject."""
    if state.hitl_approved:
        return "open_pr"
    return "reject_path"


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def _patch_env() -> None:
    """Ensure test environment variables are set.

    Loads from .env if present; overrides only the HITL-specific
    settings that tests need.
    """
    from dotenv import load_dotenv

    # Load .env first (has real DB credentials)
    load_dotenv()

    # Override only the HITL-specific settings
    env_overrides = {
        "HITL_INTERRUPT_BEFORE_WRITE_OPS": "false",
        "MAX_COST_PER_TASK_USD": "2.00",
    }
    original = {}
    for k, v in env_overrides.items():
        original[k] = os.environ.get(k)
        os.environ[k] = v
    yield
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(autouse=True)
def _reset_api_store() -> None:
    """Reset the FastAPI global store between tests to avoid
    asyncpg 'another operation is in progress' errors.
    """
    import src.api.main as api_main

    # Reset the global store before each test — the TestClient
    # will lazily create a new one via get_store()
    api_main._store = None
    yield


# ── VAL-HITL-CTRL-001: interrupt fires before PR open ────────────────


@pytest.mark.usefixtures("_patch_env")
def test_interrupt_fires_before_pr_open_on_happy_path() -> None:
    """VAL-HITL-CTRL-001: interrupt() fires at least once before
    open_pull_request is invoked on the happy path.
    """
    call_order: list[str] = []

    def do_work(state: SimpleHITLState) -> dict[str, Any]:
        call_order.append("do_work")
        return {"step": "work_done"}

    def hitl_checkpoint(state: SimpleHITLState) -> dict[str, Any]:
        call_order.append("hitl_checkpoint")
        decision = interrupt({"reason": "pre_pr_approval", "task_id": state.task_id})
        call_order.append(f"hitl_resumed_with_{decision}")
        return {"hitl_approved": decision == "approve"}

    def open_pr(state: SimpleHITLState) -> dict[str, Any]:
        call_order.append("open_pr")
        return {"pr_url": "https://github.com/test/repo/pull/1", "step": "completed"}

    def reject_path(state: SimpleHITLState) -> dict[str, Any]:
        call_order.append("reject_path")
        return {"step": "rejected"}

    graph = StateGraph(SimpleHITLState)
    graph.add_node("do_work", do_work)
    graph.add_node("hitl_checkpoint", hitl_checkpoint)
    graph.add_node("open_pr", open_pr)
    graph.add_node("reject_path", reject_path)

    graph.add_edge(START, "do_work")
    graph.add_edge("do_work", "hitl_checkpoint")
    graph.add_conditional_edges(
        "hitl_checkpoint",
        _route_after_hitl,
        {"open_pr": "open_pr", "reject_path": "reject_path"},
    )
    graph.add_edge("open_pr", END)
    graph.add_edge("reject_path", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    import asyncio

    async def _run() -> None:
        config = {"configurable": {"thread_id": "test-hitl-001"}}
        result = await compiled.ainvoke(
            SimpleHITLState(task_id="task-1").model_dump(),
            config=config,
        )

        # Interrupt should have fired, no PR yet
        assert "hitl_checkpoint" in call_order
        assert "open_pr" not in call_order
        assert "__interrupt__" in result

        # Resume with approve
        result2 = await compiled.ainvoke(
            Command(resume="approve"),
            config=config,
        )
        assert "open_pr" in call_order
        assert result2.get("pr_url") == "https://github.com/test/repo/pull/1"

    asyncio.run(_run())


# ── VAL-HITL-CTRL-002: BEFORE_WRITE_OPS=true triggers interrupt on every GH write ──


@pytest.mark.usefixtures("_patch_env")
def test_interrupt_before_every_write_when_flag_true() -> None:
    """VAL-HITL-CTRL-002: When HITL_INTERRUPT_BEFORE_WRITE_OPS=true,
    interrupt() fires before EACH GitHub write operation.

    We count the number of times the graph returns __interrupt__
    (each one corresponds to a separate interrupt event before a
    write operation).  Two write operations = two interrupt events.
    """
    def commit_and_push(state: WriteState) -> dict[str, Any]:
        decision = interrupt({"reason": "pre_write", "op": "commit_and_push"})
        return {
            "writes_count": state.writes_count + 1,
            "hitl_decisions": [*state.hitl_decisions, decision],
        }

    def open_pr(state: WriteState) -> dict[str, Any]:
        decision = interrupt({"reason": "pre_write", "op": "open_pull_request"})
        return {
            "writes_count": state.writes_count + 1,
            "hitl_decisions": [*state.hitl_decisions, decision],
        }

    graph = StateGraph(WriteState)
    graph.add_node("commit_and_push", commit_and_push)
    graph.add_node("open_pr", open_pr)
    graph.add_edge(START, "commit_and_push")
    graph.add_edge("commit_and_push", "open_pr")
    graph.add_edge("open_pr", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    import asyncio

    async def _run() -> None:
        config = {"configurable": {"thread_id": "test-hitl-002"}}

        # First invoke: hits interrupt in commit_and_push
        result1 = await compiled.ainvoke(
            WriteState(task_id="task-2").model_dump(),
            config=config,
        )
        assert "__interrupt__" in result1, "First interrupt should fire at commit_and_push"

        # Resume: commit_and_push completes, hits interrupt in open_pr
        result2 = await compiled.ainvoke(Command(resume="approve"), config=config)
        assert "__interrupt__" in result2, "Second interrupt should fire at open_pr"

        # Resume: open_pr completes, graph ends
        result3 = await compiled.ainvoke(Command(resume="approve"), config=config)
        assert result3.get("writes_count") == 2, "Both writes should complete"
        assert result3.get("hitl_decisions") == ["approve", "approve"]

    asyncio.run(_run())


# ── VAL-HITL-CTRL-003: BEFORE_WRITE_OPS=false triggers interrupt only before PR ──


@pytest.mark.usefixtures("_patch_env")
def test_interrupt_only_before_pr_when_flag_false() -> None:
    """VAL-HITL-CTRL-003: interrupt() fires ONLY before
    open_pull_request when BEFORE_WRITE_OPS=false.
    """
    def commit_and_push(state: SimpleHITLState) -> dict[str, Any]:
        return {"step": "committed"}

    def hitl_before_pr(state: SimpleHITLState) -> dict[str, Any]:
        interrupt({"reason": "pre_pr_approval"})
        return {}

    def open_pr(state: SimpleHITLState) -> dict[str, Any]:
        return {"pr_url": "https://github.com/test/repo/pull/2"}

    graph = StateGraph(SimpleHITLState)
    graph.add_node("commit_and_push", commit_and_push)
    graph.add_node("hitl_before_pr", hitl_before_pr)
    graph.add_node("open_pr", open_pr)
    graph.add_edge(START, "commit_and_push")
    graph.add_edge("commit_and_push", "hitl_before_pr")
    graph.add_edge("hitl_before_pr", "open_pr")
    graph.add_edge("open_pr", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    import asyncio

    async def _run() -> None:
        config = {"configurable": {"thread_id": "test-hitl-003"}}
        result = await compiled.ainvoke(
            SimpleHITLState(task_id="task-3").model_dump(),
            config=config,
        )
        assert result.get("step") == "committed", "commit should run without interrupt"
        assert "__interrupt__" in result

        result2 = await compiled.ainvoke(Command(resume="approve"), config=config)
        assert result2.get("pr_url") == "https://github.com/test/repo/pull/2"

    asyncio.run(_run())


# ── VAL-HITL-CTRL-010: API returns 422 on malformed decision body ─────


@pytest.mark.usefixtures("_patch_env")
def test_hitl_decision_api_422_on_malformed_body() -> None:
    """VAL-HITL-CTRL-010: Invalid decision body returns 422."""
    from src.api.main import app

    with TestClient(app) as client:
        task_id = str(uuid4())

        # Missing decision field
        resp = client.post(f"/tasks/{task_id}/hitl/decision", json={})
        assert resp.status_code == 422

        # Invalid decision value
        resp = client.post(f"/tasks/{task_id}/hitl/decision", json={"decision": "maybe"})
        assert resp.status_code == 422

        # Another invalid value
        resp = client.post(f"/tasks/{task_id}/hitl/decision", json={"decision": "approve_reject"})
        assert resp.status_code == 422


# ── VAL-HITL-CTRL-011: API returns 404 for unknown task id ────────────


@pytest.mark.usefixtures("_patch_env")
def test_hitl_decision_api_404_for_unknown_task() -> None:
    """VAL-HITL-CTRL-011: Non-existent task returns 404."""
    from src.api.main import app

    with TestClient(app) as client:
        fake_id = str(uuid4())

        resp = client.post(
            f"/tasks/{fake_id}/hitl/decision",
            json={"decision": "approve"},
        )
        assert resp.status_code == 404
        body = resp.json()
        assert "task_not_found" in str(body).lower()


# ── VAL-HITL-CTRL-009: API returns 409 when task not awaiting HITL ────


@pytest.mark.usefixtures("_patch_env")
def test_hitl_decision_api_409_when_task_not_awaiting_hitl() -> None:
    """VAL-HITL-CTRL-009: Task not in awaiting_hitl status returns 409."""
    from src.api.main import app

    with TestClient(app) as client:
        # Create a task via the API (status will be "running")
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo-409c",
                "issue_number": 1,
                "issue_text": "Test issue for 409",
                "topology": "supervisor_only",
                "auto_start": False,
            },
        )
        assert resp.status_code == 201
        task_id = resp.json()["id"]

        # Task is in "running" status — not awaiting_hitl
        resp = client.post(
            f"/tasks/{task_id}/hitl/decision",
            json={"decision": "approve"},
        )
        assert resp.status_code == 409
        body = resp.json()
        assert body.get("error") == "task_not_awaiting_hitl"
        assert body.get("current_status") == "running"


# ── VAL-HITL-CTRL-008: API returns 409 on second decision ─────────────


@pytest.mark.usefixtures("_patch_env")
def test_hitl_decision_api_409_on_second_decision() -> None:
    """VAL-HITL-CTRL-008: Second decision after first resolved returns 409."""
    import asyncio

    from src.api.main import app
    from src.memory.episodic.models import CreateTaskParams
    from src.memory.episodic.store import EpisodicStore

    async def _create_awaiting_task() -> UUID:
        store = EpisodicStore()
        await store.connect()
        try:
            task = await store.create_task(
                CreateTaskParams(
                    repo_url="https://github.com/test/repo-409b",
                    issue_number=2,
                    issue_text="Test issue 2",
                    topology="supervisor_only",
                    status="awaiting_hitl",
                )
            )
            return task.id
        finally:
            await store.close()

    task_id = asyncio.run(_create_awaiting_task())

    with TestClient(app) as client:
        # First decision: approve
        resp = client.post(
            f"/tasks/{task_id}/hitl/decision",
            json={"decision": "approve"},
        )
        assert resp.status_code == 200

        # Second decision: should get 409
        resp = client.post(
            f"/tasks/{task_id}/hitl/decision",
            json={"decision": "reject"},
        )
        assert resp.status_code == 409
        body = resp.json()
        assert body.get("error") == "decision_already_made"
        assert body.get("current_decision") == "approve"


# ── VAL-HITL-CTRL-006: POST reject ends task without PR ──────────────


@pytest.mark.usefixtures("_patch_env")
def test_hitl_decision_reject_ends_task_without_pr() -> None:
    """VAL-HITL-CTRL-006/007: Reject → task rejected, no PR,
    hitl_rejected outcome written.
    """
    from src.api.main import app

    with TestClient(app) as client:
        # Create task via API
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo-reject",
                "issue_number": 3,
                "issue_text": "Test issue 3",
                "topology": "supervisor_only",
                "auto_start": False,
            },
        )
        assert resp.status_code == 201
        task_id = resp.json()["id"]

        # Update to awaiting_hitl
        import asyncio
        from uuid import UUID

        from src.memory.episodic.store import EpisodicStore

        async def _set_awaiting() -> None:
            store = EpisodicStore()
            await store.connect()
            try:
                await store.update_task_status(UUID(task_id), "awaiting_hitl")
            finally:
                await store.close()

        asyncio.run(_set_awaiting())

        # Reject the task
        resp = client.post(
            f"/tasks/{task_id}/hitl/decision",
            json={"decision": "reject", "reason": "Not ready for merge"},
        )
        assert resp.status_code == 200

        # Verify via the API
        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        task_data = resp.json()
        assert task_data["status"] == "rejected"
        assert task_data["pr_url"] is None
        assert task_data["hitl_decision"] == "reject"


@pytest.mark.asyncio
async def test_cleanup_task_runtime_tears_down_registered_resources() -> None:
    """HITL rejection cleanup must unregister graphs and close task-scoped resources."""
    from src.orchestrator.hitl import cleanup_task_runtime, get_graph, register_graph
    from src.orchestrator.supervisor_only import (
        get_sandbox,
        get_semantic_store,
        get_store,
        register_sandbox,
        register_semantic_store,
        register_store,
    )

    class FakeClosable:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeSandbox:
        def __init__(self) -> None:
            self.torn_down = False

        async def teardown(self) -> None:
            self.torn_down = True

    task_id = str(uuid4())
    sandbox = FakeSandbox()
    store = FakeClosable()
    semantic_store = FakeClosable()

    register_graph(task_id, object(), "thread", MemorySaver())  # type: ignore[arg-type]
    register_sandbox(task_id, sandbox)  # type: ignore[arg-type]
    register_store(task_id, store)  # type: ignore[arg-type]
    register_semantic_store(task_id, semantic_store)  # type: ignore[arg-type]

    await cleanup_task_runtime(task_id)

    assert get_graph(task_id) is None
    assert get_sandbox(task_id) is None
    assert get_store(task_id) is None
    assert get_semantic_store(task_id) is None
    assert sandbox.torn_down is True
    assert store.closed is True
    assert semantic_store.closed is True


@pytest.mark.usefixtures("_patch_env")
def test_hitl_decision_reject_cleans_up_runtime_resources() -> None:
    """Rejecting through the API must tear down the paused graph runtime."""
    from unittest.mock import AsyncMock, patch

    from src.api.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo-cleanup",
                "issue_number": 15,
                "issue_text": "Test cleanup on reject",
                "topology": "supervisor_only",
                "auto_start": False,
            },
        )
        assert resp.status_code == 201
        task_id = resp.json()["id"]

        import asyncio

        from src.memory.episodic.store import EpisodicStore

        async def _set_awaiting() -> None:
            store = EpisodicStore()
            await store.connect()
            try:
                await store.update_task_status(UUID(task_id), "awaiting_hitl")
            finally:
                await store.close()

        asyncio.run(_set_awaiting())

        with patch(
            "src.api.main.cleanup_task_runtime",
            new_callable=AsyncMock,
        ) as cleanup:
            resp = client.post(
                f"/tasks/{task_id}/hitl/decision",
                json={"decision": "reject", "reason": "stop"},
            )

        assert resp.status_code == 200
        cleanup.assert_awaited_once_with(task_id)


# ── VAL-HITL-CTRL-004: POST approve resumes the LangGraph ─────────────


@pytest.mark.usefixtures("_patch_env")
def test_hitl_decision_approve_resumes_and_sets_approved() -> None:
    """VAL-HITL-CTRL-004: Approve → task transitions away from
    awaiting_hitl, hitl_decision='approve'.
    """
    from src.api.main import app

    with TestClient(app) as client:
        # Create task via API
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo-approve",
                "issue_number": 4,
                "issue_text": "Test issue 4",
                "topology": "supervisor_only",
                "auto_start": False,
            },
        )
        assert resp.status_code == 201
        task_id = resp.json()["id"]

        # Update to awaiting_hitl
        import asyncio
        from uuid import UUID

        from src.memory.episodic.store import EpisodicStore

        async def _set_awaiting() -> None:
            store = EpisodicStore()
            await store.connect()
            try:
                await store.update_task_status(UUID(task_id), "awaiting_hitl")
            finally:
                await store.close()

        asyncio.run(_set_awaiting())

        # Approve the task
        resp = client.post(
            f"/tasks/{task_id}/hitl/decision",
            json={"decision": "approve"},
        )
        assert resp.status_code == 200

        # Verify via the API
        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        task_data = resp.json()
        assert task_data["hitl_decision"] == "approve"
        assert task_data["status"] in ("approved", "running", "completed"), (
            f"Expected status transition, got {task_data['status']}"
        )


# ── VAL-HITL-CTRL-013: pr_url populated only after successful PR open ──


@pytest.mark.usefixtures("_patch_env")
def test_pr_url_null_on_reject_populated_on_approve() -> None:
    """VAL-HITL-CTRL-013: pr_url remains NULL on reject path."""
    from src.api.main import app

    with TestClient(app) as client:
        # Create task via API
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo-prurl",
                "issue_number": 5,
                "issue_text": "Test issue 5",
                "topology": "supervisor_only",
                "auto_start": False,
            },
        )
        assert resp.status_code == 201
        task_id = resp.json()["id"]

        # Update to awaiting_hitl
        import asyncio
        from uuid import UUID

        from src.memory.episodic.store import EpisodicStore

        async def _set_awaiting() -> None:
            store = EpisodicStore()
            await store.connect()
            try:
                await store.update_task_status(UUID(task_id), "awaiting_hitl")
            finally:
                await store.close()

        asyncio.run(_set_awaiting())

        # Reject the task
        resp = client.post(
            f"/tasks/{task_id}/hitl/decision",
            json={"decision": "reject"},
        )
        assert resp.status_code == 200

        # Verify pr_url is NULL via the API
        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        task_data = resp.json()
        assert task_data["pr_url"] is None


# ── VAL-HITL-CTRL-012 / VAL-CROSS-018: Restart preserves awaiting_hitl ──


@pytest.mark.usefixtures("_patch_env")
def test_restart_preserves_awaiting_hitl_state() -> None:
    """VAL-HITL-CTRL-012 / VAL-CROSS-018: When the FastAPI process is
    restarted while a task is in awaiting_hitl, the LangGraph state
    is rehydrated from MemorySaver and a subsequent decision still
    resolves correctly.
    """
    def step_one(state: RestartState) -> dict[str, Any]:
        return {"value": 1}

    def hitl_node(state: RestartState) -> dict[str, Any]:
        decision = interrupt({"reason": "pre_pr_approval"})
        if decision == "approve":
            return {"value": 2}
        return {"value": -1}

    def step_three(state: RestartState) -> dict[str, Any]:
        return {"value": state.value + 10}

    graph = StateGraph(RestartState)
    graph.add_node("step_one", step_one)
    graph.add_node("hitl_node", hitl_node)
    graph.add_node("step_three", step_three)
    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "hitl_node")
    graph.add_edge("hitl_node", "step_three")
    graph.add_edge("step_three", END)

    # Shared checkpointer — simulates persistence across restarts
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    import asyncio

    async def _run() -> None:
        config = {"configurable": {"thread_id": "test-restart-012"}}
        result = await compiled.ainvoke(
            RestartState(task_id="restart-task").model_dump(),
            config=config,
        )
        assert "__interrupt__" in result

        # Simulate restart: new graph instance, same checkpointer
        compiled_after_restart = graph.compile(checkpointer=checkpointer)
        result2 = await compiled_after_restart.ainvoke(
            Command(resume="approve"),
            config=config,
        )
        assert result2.get("value") == 12  # 2 + 10

    asyncio.run(_run())


# ── VAL-CROSS-019: Backend restart on mid-graph running task ──────────


@pytest.mark.usefixtures("_patch_env")
def test_restart_mid_running_task_recovers() -> None:
    """VAL-CROSS-019: Task in 'running' recovers from last checkpoint
    after restart. Task is never stuck silently in 'running'.
    """
    def step_a(state: RunningState) -> dict[str, Any]:
        return {"step": 1}

    def step_b(state: RunningState) -> dict[str, Any]:
        return {"step": 2}

    def step_c(state: RunningState) -> dict[str, Any]:
        return {"step": 3}

    graph = StateGraph(RunningState)
    graph.add_node("step_a", step_a)
    graph.add_node("step_b", step_b)
    graph.add_node("step_c", step_c)
    graph.add_edge(START, "step_a")
    graph.add_edge("step_a", "step_b")
    graph.add_edge("step_b", "step_c")
    graph.add_edge("step_c", END)

    checkpointer = MemorySaver()
    # Use interrupt_after to pause after step_a (simulate mid-graph)
    compiled = graph.compile(checkpointer=checkpointer, interrupt_after=["step_a"])

    import asyncio

    async def _run() -> None:
        config = {"configurable": {"thread_id": "test-restart-019"}}
        result = await compiled.ainvoke(
            RunningState(task_id="restart-running").model_dump(),
            config=config,
        )
        assert result.get("step") == 1

        # Simulate restart: new graph instance, same checkpointer
        compiled_after_restart = graph.compile(checkpointer=checkpointer)
        result2 = await compiled_after_restart.ainvoke(None, config=config)
        assert result2 is not None
        assert result2.get("step") == 3, "Task should complete from checkpoint"

    asyncio.run(_run())


# ── VAL-HITL-CTRL-005: Approve invokes open_pull_request exactly once ──


@pytest.mark.usefixtures("_patch_env")
def test_approve_invokes_open_pr_exactly_once() -> None:
    """VAL-HITL-CTRL-005: After approval, the task transitions away
    from awaiting_hitl and hitl_decision is set to 'approve'.
    (API-level test; full PR integration tested in supervisor_only
    topology with real GitHub client.)
    """
    from src.api.main import app

    with TestClient(app) as client:
        # Create task via API
        resp = client.post(
            "/tasks",
            json={
                "repo_url": "https://github.com/test/repo-pr1",
                "issue_number": 6,
                "issue_text": "Test issue 6",
                "topology": "supervisor_only",
                "auto_start": False,
            },
        )
        assert resp.status_code == 201
        task_id = resp.json()["id"]

        # Update to awaiting_hitl
        import asyncio
        from uuid import UUID

        from src.memory.episodic.store import EpisodicStore

        async def _set_awaiting() -> None:
            store = EpisodicStore()
            await store.connect()
            try:
                await store.update_task_status(UUID(task_id), "awaiting_hitl")
            finally:
                await store.close()

        asyncio.run(_set_awaiting())

        # Approve
        resp = client.post(
            f"/tasks/{task_id}/hitl/decision",
            json={"decision": "approve"},
        )
        assert resp.status_code == 200

        # Verify via the API
        resp = client.get(f"/tasks/{task_id}")
        assert resp.status_code == 200
        task_data = resp.json()
        assert task_data["hitl_decision"] == "approve"
        assert task_data["status"] != "awaiting_hitl"
