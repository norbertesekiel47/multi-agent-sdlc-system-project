"""SDLC-Swarm FastAPI backend — API gateway.

Endpoints:
  - GET /health — structured health check
  - POST /tasks — create a task row, return 201 with {id}
  - GET /tasks — list tasks with optional filters
  - GET /tasks/{id} — return task detail with all fields
  - WS /events/stream — live trace event stream scoped by task_id

Error responses are always JSON {error, ...} with no Python tracebacks
or internal module names (VAL-BACKEND-API-002).

src/api/ never imports src/llm/ or openrouter/openai directly
(VAL-BACKEND-API-003).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from uuid import UUID

import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.errors import (
    generic_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from src.api.models import (
    CreateTaskRequest,
    CreateTaskResponse,
    HITLDecisionRequest,
    HITLDecisionResponse,
    ListTasksResponse,
    TaskDetailResponse,
    TaskListItemResponse,
)
from src.logging.secret_filter import install_secret_filter
from src.memory.episodic.models import CreateTaskParams
from src.memory.episodic.store import EpisodicStore
from src.tracing.ws_broadcaster import get_trace_broadcaster

# Load environment variables from .env at import time (before any config reads).
load_dotenv()

# Install secret redaction filter on root logger so that PAT/API-key values
# never appear in any log output or Langfuse span.
install_secret_filter()

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# ── FastAPI app ──────────────────────────────────────────────────────

app = FastAPI(
    title="SDLC-Swarm",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware — allow the Next.js dashboard (port 3101) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3101",
        "http://127.0.0.1:3101",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom exception handlers (VAL-BACKEND-API-002)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


# ── Health response model ────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Structured health payload per VAL-BACKEND-API-001."""

    status: str
    version: str
    db: str
    langfuse: str


# ── Dependency injection ─────────────────────────────────────────────

# Shared EpisodicStore instance (lazily initialized)
_store: EpisodicStore | None = None


async def get_store() -> EpisodicStore:
    """FastAPI dependency that provides an EpisodicStore.

    The store is shared across requests and lazily connected.
    """
    global _store  # noqa: PLW0603
    if _store is None or _store._pool is None:
        _store = EpisodicStore()
        await _store.connect()
    return _store


@app.on_event("shutdown")
async def _shutdown_store() -> None:
    """Close the shared store pool on shutdown."""
    global _store  # noqa: PLW0603
    if _store is not None:
        await _store.close()
        _store = None


# ── Background orchestrator ─────────────────────────────────────────


def _start_orchestrator_background(task_id: str, store: EpisodicStore) -> None:
    """Start the orchestrator for a task in a background asyncio task.

    This is fire-and-forget — errors are logged but don't affect the
    POST /tasks response.
    """

    async def _run() -> None:
        from src.orchestrator import Orchestrator

        try:
            orchestrator = Orchestrator(store=store)
            await orchestrator.run_task(task_id=task_id)
        except Exception:
            logger.exception("Background orchestrator failed for task %s", task_id)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run())
    except RuntimeError:
        logger.warning("No running event loop; cannot start orchestrator for task %s", task_id)


# ── GET /health ──────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return structured health status.

    Checks Postgres connectivity and Langfuse reachability.
    Responds within 200 ms (VAL-BACKEND-API-001).
    """
    db_status = await _check_postgres()
    langfuse_status = await _check_langfuse()

    overall = "ok" if db_status == "ok" and langfuse_status != "unreachable" else "degraded"

    return HealthResponse(
        status=overall,
        version=__version__,
        db=db_status,
        langfuse=langfuse_status,
    )


async def _check_postgres() -> str:
    """Probe Postgres+pgvector on port 5433."""
    dsn = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'sdlc_swarm')}"
        f":{os.getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
        f":{os.getenv('POSTGRES_PORT', '5433')}"
        f"/{os.getenv('POSTGRES_DB', 'sdlc_swarm')}"
    )
    try:
        conn = await asyncpg.connect(dsn)
        await conn.execute("SELECT 1")
        await conn.close()
        return "ok"
    except Exception:
        return "unreachable"


async def _check_langfuse() -> str:
    """Probe Langfuse on port 3110."""
    langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3110")
    for path in ("/api/health", "/"):
        url = f"{langfuse_host}{path}"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(url)
                if 200 <= resp.status_code < 300:
                    return "ok"
                if resp.status_code < 500:
                    continue
        except Exception:
            continue
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{langfuse_host}/")
            if 200 <= resp.status_code < 300:
                return "ok"
            return "degraded"
    except Exception:
        return "unreachable"


# ── POST /tasks ──────────────────────────────────────────────────────


@app.post("/tasks", status_code=201, response_model=CreateTaskResponse)
async def create_task(
    body: CreateTaskRequest,
    store: EpisodicStore = Depends(get_store),  # noqa: B008
) -> CreateTaskResponse:
    """Create a new task row and return 201 with {id}.

    Validates topology, repo URL, and issue number.
    Creates the row in Postgres via the EpisodicStore.
    If ``auto_start`` is True (default), also kicks off the
    orchestrator in the background.
    """
    params = CreateTaskParams(
        repo_url=body.repo_url,
        issue_number=body.issue_number,
        issue_text=body.issue_text,
        topology=body.topology,
        status="running",
    )
    task = await store.create_task(params)
    logger.info(
        "Created task %s (repo=%s, issue=%d, topology=%s)",
        task.id, task.repo_url, task.issue_number, task.topology,
    )

    # Auto-start the orchestrator in the background
    if body.auto_start:
        _start_orchestrator_background(str(task.id), store)

    return CreateTaskResponse(id=task.id)


# ── GET /tasks ───────────────────────────────────────────────────────


@app.get("/tasks", response_model=ListTasksResponse)
async def list_tasks(
    repo_url: str | None = Query(default=None),
    status: str | None = Query(default=None),
    outcome: str | None = Query(default=None),
    topology: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    store: EpisodicStore = Depends(get_store),  # noqa: B008
) -> ListTasksResponse:
    """List tasks with optional filters."""
    rows = await store.list_tasks(
        repo_url=repo_url,
        status=status,
        outcome=outcome,
        topology=topology,
        limit=limit,
        offset=offset,
    )
    items = [
        TaskListItemResponse(
            id=r.id,
            repo_url=r.repo_url,
            issue_number=r.issue_number,
            topology=r.topology,
            status=r.status,
            total_cost_usd=r.total_cost_usd,
            started_at=r.started_at,
            ended_at=r.ended_at,
        )
        for r in rows
    ]
    return ListTasksResponse(
        tasks=items,
        total=len(items),
        limit=limit,
        offset=offset,
    )


# ── GET /tasks/{id} ─────────────────────────────────────────────────


@app.get("/tasks/{task_id}", response_model=TaskDetailResponse)
async def get_task(
    task_id: UUID,
    store: EpisodicStore = Depends(get_store),  # noqa: B008
) -> TaskDetailResponse:
    """Return task detail with all fields, including HITL enrichment."""
    row = await store.get_task(task_id)
    if row is None:
        raise HTTPException(status_code=404, detail="task_not_found")

    # Enrich with HITL-related data from decisions and outcomes
    pending_diff: str | None = None
    hitl_cause: str | None = None
    hitl_cause_detail: dict[str, object] | None = None
    review_summary: str | None = None
    test_summary: str | None = None
    reject_reason: str | None = None

    # For tasks that are or were in HITL state, fetch decisions and outcomes
    if row.status in ("awaiting_hitl", "approved", "rejected", "completed", "failed"):
        decisions = await store.get_decisions_for_task(task_id)
        outcomes = await store.get_outcomes_for_task(task_id)

        # Extract latest code_edit diff
        for d in reversed(decisions):
            if d.decision_type == "code_edit" and "diff" in d.decision_data:
                pending_diff = d.decision_data["diff"]
                break

        # Extract latest review verdict summary
        for d in reversed(decisions):
            if d.decision_type == "review_verdict":
                verdict = d.decision_data.get("verdict", "")
                issues = d.decision_data.get("issues", [])
                parts = [f"Verdict: {verdict}"]
                if issues:
                    parts.append(f"Issues ({len(issues)}): " + "; ".join(str(i) for i in issues[:5]))
                review_summary = "\n".join(parts)
                break

        # Extract latest test report summary
        for d in reversed(decisions):
            if d.decision_type == "test_report":
                passed = d.decision_data.get("passed", 0)
                failed = d.decision_data.get("failed", 0)
                failed_names = d.decision_data.get("failed_test_names", [])
                parts = [f"Tests: {passed} passed, {failed} failed"]
                if failed_names:
                    parts.append("Failed: " + ", ".join(str(n) for n in failed_names[:5]))
                test_summary = "\n".join(parts)
                break

        # Extract HITL cause from outcomes (loop_detected, uncertainty_escalation, etc.)
        for o in reversed(outcomes):
            if o.outcome in ("loop_detected", "uncertainty_escalation",
                             "retry_budget_exhausted", "guardrail_block",
                             "cost_budget_exhausted"):
                hitl_cause = o.outcome
                hitl_cause_detail = o.detail
                break

        # Extract reject reason if task was rejected via HITL
        if row.hitl_decision == "reject":
            for o in reversed(outcomes):
                if o.outcome == "hitl_rejected" and o.detail and "reason" in o.detail:
                    reject_reason = o.detail["reason"]
                    break

    return TaskDetailResponse(
        id=row.id,
        repo_url=row.repo_url,
        issue_number=row.issue_number,
        issue_text=row.issue_text,
        topology=row.topology,
        status=row.status,
        total_cost_usd=row.total_cost_usd,
        total_tokens_in=row.total_tokens_in,
        total_tokens_out=row.total_tokens_out,
        total_tokens_cached=row.total_tokens_cached,
        agent_costs=row.agent_costs,
        hitl_decision=row.hitl_decision,
        pr_url=row.pr_url,
        started_at=row.started_at,
        ended_at=row.ended_at,
        pending_diff=pending_diff,
        hitl_cause=hitl_cause,
        hitl_cause_detail=hitl_cause_detail,
        review_summary=review_summary,
        test_summary=test_summary,
        reject_reason=reject_reason,
    )


# ── POST /tasks/{id}/hitl/decision ────────────────────────────────────


@app.post("/tasks/{task_id}/hitl/decision", response_model=HITLDecisionResponse)
async def hitl_decision(
    task_id: UUID,
    body: HITLDecisionRequest,
    store: EpisodicStore = Depends(get_store),  # noqa: B008
) -> HITLDecisionResponse | JSONResponse:
    """Resolve an HITL interrupt for a task.

    VAL-HITL-CTRL-004: POST approve resumes the LangGraph.
    VAL-HITL-CTRL-006: POST reject ends task without PR.
    VAL-HITL-CTRL-008: Second decision returns 409.
    VAL-HITL-CTRL-009: Task not in awaiting_hitl returns 409.
    VAL-HITL-CTRL-010: Invalid body returns 422 (handled by Pydantic).
    VAL-HITL-CTRL-011: Unknown task returns 404.
    """
    # Check task exists
    task = await store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task_not_found")

    # VAL-HITL-CTRL-008: Decision already made (check BEFORE status
    # because after a decision, the status may have changed)
    if task.hitl_decision is not None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "decision_already_made",
                "current_decision": task.hitl_decision,
            },
        )

    # VAL-HITL-CTRL-009: Task must be in awaiting_hitl status
    if task.status != "awaiting_hitl":
        return JSONResponse(
            status_code=409,
            content={
                "error": "task_not_awaiting_hitl",
                "current_status": task.status,
            },
        )

    decision = body.decision
    reason = body.reason

    if decision == "approve":
        # VAL-HITL-CTRL-004: Approve → resume LangGraph
        # VAL-HITL-CTRL-005: Approve → open_pull_request invoked (by resumed graph)
        # Record the HITL decision on the task row (without finishing the task)
        if store._pool is None:
            raise HTTPException(status_code=500, detail="store_not_connected")
        await store._pool.execute(
            "UPDATE tasks SET hitl_decision = $2 WHERE id = $1",
            task_id,
            "approve",
        )
        await store.update_task_status(task_id, "approved")

        # Try to resume the orchestrator graph if it's checkpointed
        await _resume_orchestrator_if_active(task_id, "approve")

        # Check current status (might have been updated by resumed graph)
        updated_task = await store.get_task(task_id)
        final_status = updated_task.status if updated_task else "approved"

        # If status is still 'approved' (graph didn't complete yet),
        # mark as running so the graph can continue
        if final_status == "approved":
            await store.update_task_status(task_id, "running")

        logger.info(
            "HITL approve for task %s, reason=%s",
            task_id,
            reason,
        )

        # Re-fetch to get final status
        updated_task = await store.get_task(task_id)
        final_status = updated_task.status if updated_task else "running"

        return HITLDecisionResponse(
            task_id=task_id,
            decision="approve",
            status=final_status,
        )

    else:
        # decision == "reject"
        # VAL-HITL-CTRL-006: Reject → no PR, task rejected
        # VAL-HITL-CTRL-007: Write hitl_rejected outcome
        from src.memory.episodic.models import CreateOutcomeParams

        await store.finish_task(
            task_id,
            "rejected",
            hitl_decision="reject",
        )
        await store.create_outcome(
            CreateOutcomeParams(
                task_id=task_id,
                outcome="hitl_rejected",
                detail={"reason": reason} if reason else {},
            )
        )

        logger.info(
            "HITL reject for task %s, reason=%s",
            task_id,
            reason,
        )

        return HITLDecisionResponse(
            task_id=task_id,
            decision="reject",
            status="rejected",
        )


async def _resume_orchestrator_if_active(task_id: UUID, decision: str) -> None:
    """Resume a paused LangGraph for the given task if it's checkpointed.

    Looks up the task's compiled graph and checkpointer, then resumes
    execution with the HITL decision.  This is the core of the
    interrupt/resume flow.

    If no active graph is found (e.g., the task hasn't started yet
    or the graph already completed), this is a no-op.
    """
    from src.orchestrator.hitl import resume_graph

    try:
        resumed = await resume_graph(str(task_id), decision)
        if resumed:
            logger.info("Resumed graph for task %s with decision=%s", task_id, decision)
    except Exception:
        logger.warning(
            "Failed to resume graph for task %s (decision=%s)",
            task_id,
            decision,
            exc_info=True,
        )


# ── WS /events/stream ────────────────────────────────────────────────


@app.websocket("/events/stream")
async def events_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for live trace event streaming.

    Accepts a ``task_id`` query parameter to scope events.
    Sends JSON trace events as they are published by the
    tracing subsystem.
    """
    await websocket.accept()
    task_id = websocket.query_params.get("task_id", "")
    broadcaster = get_trace_broadcaster()

    if not task_id:
        await websocket.send_json({"type": "error", "error": "task_id query parameter required"})
        await websocket.close()
        return

    queue = await broadcaster.subscribe(task_id)
    try:
        while True:
            try:
                # Wait for trace events with a timeout for liveness check
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_text(message)
            except TimeoutError:
                # Send a keepalive ping
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
            except WebSocketDisconnect:
                break
            except Exception:
                logger.warning("WebSocket send error for task %s", task_id)
                break
    finally:
        await broadcaster.unsubscribe(task_id, queue)
        with contextlib.suppress(Exception):
            await websocket.close()
