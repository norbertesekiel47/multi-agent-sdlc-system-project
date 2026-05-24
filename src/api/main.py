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
    """Return task detail with all fields."""
    row = await store.get_task(task_id)
    if row is None:
        raise HTTPException(status_code=404, detail="task_not_found")
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
        hitl_decision=row.hitl_decision,
        pr_url=row.pr_url,
        started_at=row.started_at,
        ended_at=row.ended_at,
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
