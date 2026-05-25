"""HITL (Human-in-the-Loop) checkpoint management for LangGraph.

Manages the lifecycle of LangGraph interrupt/resume:

1. **Graph Registry** — Maps task_id → (compiled_graph, checkpointer)
   so the HITL decision endpoint can resume the right graph.

2. **Resume Logic** — When a decision comes in (approve/reject),
   resumes the paused graph with ``Command(resume=decision)``.

3. **Restart Recovery** — Uses a Postgres-backed checkpointer
   (``AsyncPostgresSaver``) when available so that state survives
   backend restarts.  Falls back to ``MemorySaver`` for testing.

Architecture: §2.2 LangGraph Orchestrator, §4.3 Security (HITL
mandatory before any GitHub write).

Validation: VAL-HITL-CTRL-001 through VAL-HITL-CTRL-013,
VAL-CROSS-018, VAL-CROSS-019.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

logger = logging.getLogger(__name__)


# ── Graph Registry ───────────────────────────────────────────────────
# Maps task_id → (compiled_graph, thread_id, checkpointer)
# The checkpointer is shared across the application so that
# Postgres-backed state survives restarts.

_GraphEntry = tuple[CompiledStateGraph, str, MemorySaver | Any]  # type: ignore[type-arg]

_active_graphs: dict[str, _GraphEntry] = {}

# Shared MemorySaver instance for all tasks (used when
# PostgresSaver is not available).
_shared_memory_saver: MemorySaver | None = None


def get_shared_checkpointer() -> MemorySaver:
    """Return the shared MemorySaver checkpointer.

    Creates one lazily if it doesn't exist yet.
    For production, use get_postgres_checkpointer() instead.
    """
    global _shared_memory_saver  # noqa: PLW0603
    if _shared_memory_saver is None:
        _shared_memory_saver = MemorySaver()
    return _shared_memory_saver


def get_postgres_checkpointer() -> Any:
    """Return an AsyncPostgresSaver checkpointer if available.

    Falls back to MemorySaver if the Postgres checkpointer
    cannot be initialized (e.g., missing dependency).
    """
    try:
        import importlib.util

        if importlib.util.find_spec("langgraph.checkpoint.postgres.aio") is None:
            raise ImportError("langgraph-checkpoint-postgres not installed")  # noqa: TRY301

        dsn = (
            f"postgresql://{os.getenv('POSTGRES_USER', 'sdlc_swarm')}"
            f":{os.getenv('POSTGRES_PASSWORD', 'sdlc_swarm_dev')}"
            f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
            f":{os.getenv('POSTGRES_PORT', '5433')}"
            f"/{os.getenv('POSTGRES_DB', 'sdlc_swarm')}"
        )

        # AsyncPostgresSaver needs setup() called before use
        # We return a factory that will create and setup the checkpointer
        return _PostgresCheckpointerFactory(dsn)
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not available, "
            "falling back to MemorySaver"
        )
        return get_shared_checkpointer()


class _PostgresCheckpointerFactory:
    """Factory that creates AsyncPostgresSaver instances on demand.

    The checkpointer must be set up asynchronously before first use.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._checkpointer: Any = None

    async def get(self) -> Any:
        """Return an initialized AsyncPostgresSaver."""
        if self._checkpointer is None:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            self._checkpointer = AsyncPostgresSaver.from_conn_string(self._dsn)
            await self._checkpointer.setup()
        return self._checkpointer


def register_graph(
    task_id: str,
    compiled_graph: CompiledStateGraph,  # type: ignore[type-arg]
    thread_id: str,
    checkpointer: MemorySaver | Any,
) -> None:
    """Register a compiled graph for a task so the HITL endpoint
    can resume it later.

    Called by the Orchestrator before starting the graph.
    """
    _active_graphs[task_id] = (compiled_graph, thread_id, checkpointer)
    logger.debug("Registered graph for task %s (thread=%s)", task_id, thread_id)


def unregister_graph(task_id: str) -> None:
    """Remove a graph from the registry after it completes."""
    _active_graphs.pop(task_id, None)
    logger.debug("Unregistered graph for task %s", task_id)


def get_graph(task_id: str) -> _GraphEntry | None:
    """Look up a registered graph by task_id."""
    return _active_graphs.get(task_id)


async def _call_optional_async(obj: Any, method_name: str) -> None:
    """Call a cleanup method that may be sync or async."""
    method = getattr(obj, method_name, None)
    if method is None:
        return
    result = method()
    if inspect.isawaitable(result):
        await result


async def cleanup_task_runtime(task_id: str) -> None:
    """Unregister and close task-scoped graph, sandbox, store, and guardrail resources."""
    unregister_graph(task_id)

    from src.orchestrator.supervisor_only import (
        get_sandbox,
        get_store,
        unregister_guardrail,
        unregister_sandbox,
        unregister_semantic_store,
        unregister_store,
    )

    sandbox = get_sandbox(task_id)
    store = get_store(task_id)
    semantic_store = unregister_semantic_store(task_id)

    unregister_guardrail(task_id)
    unregister_sandbox(task_id)
    unregister_store(task_id)

    if sandbox is not None:
        await _call_optional_async(sandbox, "teardown")
    if store is not None:
        await _call_optional_async(store, "close")
    if semantic_store is not None:
        await _call_optional_async(semantic_store, "close")

    logger.debug("Cleaned up runtime resources for task %s", task_id)


async def resume_graph(task_id: str, decision: str) -> bool:
    """Resume a paused LangGraph with the given HITL decision.

    Returns True if a graph was found and resumed, False otherwise.

    This is the core of the HITL interrupt/resume flow:
    - VAL-HITL-CTRL-004: Approve resumes the graph
    - VAL-HITL-CTRL-006: Reject → graph resumes onto rejection path

    The graph must have been compiled with a checkpointer so
    the interrupted state can be rehydrated.
    """
    entry = get_graph(task_id)
    if entry is None:
        logger.warning("No registered graph for task %s; cannot resume", task_id)
        return False

    compiled_graph, thread_id, checkpointer = entry

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    # If the checkpointer is a factory, get the actual checkpointer
    if isinstance(checkpointer, _PostgresCheckpointerFactory):
        checkpointer = await checkpointer.get()
        # Re-compile the graph with the actual checkpointer
        # Actually, the graph was already compiled with the checkpointer
        # We just need to update our reference
        _active_graphs[task_id] = (compiled_graph, thread_id, checkpointer)

    try:
        result = await compiled_graph.ainvoke(  # type: ignore[call-overload]
            Command(resume=decision),
            config=config,
        )

        # Check if the graph completed or hit another interrupt
        if "__interrupt__" in result:
            logger.info(
                "Graph for task %s hit another interrupt after resume",
                task_id,
            )
            # The task stays in awaiting_hitl — the next interrupt
            # needs another decision
        else:
            # Graph completed — unregister and let the orchestrator
            # handle post-processing
            logger.info("Graph for task %s completed after HITL resume", task_id)
            try:
                # Trigger post-graph processing (update outcomes, etc.)
                await _post_graph_completion(task_id, result)
            finally:
                await cleanup_task_runtime(task_id)

        return True

    except Exception:
        logger.error(
            "Failed to resume graph for task %s",
            task_id,
            exc_info=True,
        )
        return False


async def _post_graph_completion(task_id: str, final_state: dict[str, Any]) -> None:
    """After the graph completes post-HITL, update the episodic store
    with the final state (outcome, pr_url, costs, etc.).
    """
    from uuid import UUID

    from src.memory.episodic.models import CreateOutcomeParams
    from src.memory.episodic.store import EpisodicStore

    try:
        store = EpisodicStore()
        await store.connect()
        try:
            task_uuid = UUID(task_id)

            # Get current pr_url and outcome from the state
            pr_url = final_state.get("pr_url", "")
            outcome = final_state.get("outcome", "success")
            status = final_state.get("status", "completed")

            # Update task totals
            total_cost = final_state.get("total_cost_usd")
            total_tokens_in = final_state.get("total_tokens_in")
            total_tokens_out = final_state.get("total_tokens_out")
            total_tokens_cached = final_state.get("total_tokens_cached")
            agent_costs = final_state.get("agent_costs")

            if total_cost is not None:
                from decimal import Decimal
                if not isinstance(total_cost, Decimal):
                    total_cost = Decimal(str(total_cost))
                await store.update_task_totals(
                    task_uuid,
                    total_cost_usd=total_cost,
                    total_tokens_in=total_tokens_in,
                    total_tokens_out=total_tokens_out,
                    total_tokens_cached=total_tokens_cached,
                    agent_costs=agent_costs,
                )

            # Write outcome if present
            if outcome:
                await store.create_outcome(
                    CreateOutcomeParams(
                        task_id=task_uuid,
                        outcome=outcome,
                        detail={
                            "pr_url": pr_url,
                            "post_hitl": True,
                        },
                    )
                )

            # Finish the task
            await store.finish_task(
                task_uuid,
                status,
                pr_url=pr_url or None,
            )

            logger.info(
                "Post-graph completion for task %s: outcome=%s, pr_url=%s",
                task_id,
                outcome,
                pr_url,
            )

        finally:
            await store.close()
    except Exception:
        logger.error(
            "Failed to update episodic store after graph completion for task %s",
            task_id,
            exc_info=True,
        )


def is_before_write_ops() -> bool:
    """Check the HITL_INTERRUPT_BEFORE_WRITE_OPS env var.

    VAL-HITL-CTRL-002: When true, interrupt fires before EVERY
    GitHub write operation (commit_and_push, open_pull_request).

    VAL-HITL-CTRL-003: When false (default), interrupt fires only
    before open_pull_request.
    """
    return os.getenv("HITL_INTERRUPT_BEFORE_WRITE_OPS", "false").lower() in (
        "true",
        "1",
        "yes",
    )
