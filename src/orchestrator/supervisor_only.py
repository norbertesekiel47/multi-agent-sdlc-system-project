"""LangGraph supervisor_only topology — sequential routing through Supervisor.

Supervisor routes between Planner → Coder → Reviewer → QA sequentially.
No peer handoff between Coder and Reviewer — all routing goes through
the Supervisor node (VAL-TOPOLOGY-002, VAL-TOPOLOGY-003).

If Reviewer verdict is ``reject_with_changes``, the Supervisor
re-routes to Coder (sequential loop, not peer swarm).  The
Supervisor owns the retry budget and enforces it.

Architecture reference: §2.2 LangGraph Orchestrator, §5 Topology Configurations.

Span hierarchy for Langfuse traces::

    Supervisor (routing span)
      ├── Planner (LLM span)
      ├── Coder (LLM span)
      ├── Reviewer (LLM span)
      └── (QA in M4)

All agent spans have the Supervisor span as their parent —
never another agent span.  This distinguishes supervisor_only
from the hybrid topology where Coder⇄Reviewer peer handoff
creates direct agent-to-agent parent edges.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph

from src.agents.models import ChangePlan, CodeEdit, IssueContext, ReviewResult
from src.github_client.client import GitHubClient, canonicalize_repo_url
from src.llm.cost import estimate_cost_tiktoken, get_max_cost_per_task
from src.orchestrator import OrchestratorState
from src.sandbox.manager import SandboxManager
from src.tracing.langfuse import SpanType, get_tracing_client
from src.tracing.ws_broadcaster import TraceEvent, get_trace_broadcaster

logger = logging.getLogger(__name__)

# Maximum retries per (agent, step) before escalation.
_MAX_RETRIES_PER_STEP: int = 3

# Branch prefix for PRs
_BRANCH_PREFIX = "sdlc-swarm/"


# ── Helper functions ────────────────────────────────────────────────


def _trunc_json(obj: Any, max_len: int = 500) -> str:
    """Serialize a Pydantic model to JSON and truncate for span I/O."""
    import json

    raw = json.dumps(obj.model_dump(mode="json"), sort_keys=True, default=str)
    if len(raw) > max_len:
        return raw[:max_len] + "…"
    return raw


# ── Sandbox Registry ───────────────────────────────────────────────
# LangGraph serializes state between nodes, so SandboxManager
# objects cannot live in the state.  Instead, we use a task-scoped
# registry that nodes look up by task_id.  The Orchestrator
# provisions the sandbox before the graph runs and tears it down
# after.

_active_sandboxes: dict[str, SandboxManager] = {}


def register_sandbox(task_id: str, sandbox: SandboxManager) -> None:
    """Register a sandbox for a task (called by Orchestrator before graph)."""
    _active_sandboxes[task_id] = sandbox


def get_sandbox(task_id: str) -> SandboxManager | None:
    """Look up the sandbox for a task (called by node functions)."""
    return _active_sandboxes.get(task_id)


def unregister_sandbox(task_id: str) -> None:
    """Remove a sandbox from the registry (called by Orchestrator after graph)."""
    _active_sandboxes.pop(task_id, None)


# ── Episodic Store Registry ─────────────────────────────────────────
# Same pattern as sandbox — store reference can't be serialized.

from src.memory.episodic.store import EpisodicStore  # noqa: E402
from src.memory.semantic.store import SemanticStore  # noqa: E402

_active_stores: dict[str, EpisodicStore] = {}
_active_semantic_stores: dict[str, SemanticStore] = {}


def register_store(task_id: str, store: EpisodicStore) -> None:
    """Register an episodic store for a task."""
    _active_stores[task_id] = store


def get_store(task_id: str) -> EpisodicStore | None:
    """Look up the episodic store for a task."""
    return _active_stores.get(task_id)


def unregister_store(task_id: str) -> None:
    """Remove an episodic store from the registry."""
    _active_stores.pop(task_id, None)


def register_semantic_store(task_id: str, store: SemanticStore) -> None:
    """Register a semantic store for a task."""
    _active_semantic_stores[task_id] = store


def get_semantic_store(task_id: str) -> SemanticStore | None:
    """Look up the semantic store for a task."""
    return _active_semantic_stores.get(task_id)


def unregister_semantic_store(task_id: str) -> SemanticStore | None:
    """Remove a semantic store from the registry and return it."""
    return _active_semantic_stores.pop(task_id, None)


# ── Routing functions ───────────────────────────────────────────────


def route_after_planner(state: OrchestratorState) -> Literal["run_coder"]:
    """After Planner completes, always route to Coder.

    The Planner always succeeds (PydanticAI retries internally)
    or the orchestrator handles the failure at a higher level.
    """
    return "run_coder"


def route_after_coder(state: OrchestratorState) -> Literal["run_reviewer"]:
    """After Coder completes, always route to Reviewer.

    In supervisor_only, there is no direct Coder→QA path;
    the Reviewer must always inspect the code first.
    """
    return "run_reviewer"


def route_after_review(
    state: OrchestratorState,
) -> Literal["run_coder", "run_supervisor_finalize", "halt_retry_exhausted"]:
    """After Reviewer, route based on verdict.

    - ``accept``: advance to Supervisor finalize (→ PR or end)
    - ``reject_with_changes``: re-route to Coder (if retry budget
      allows), otherwise halt with retry_budget_exhausted
    - ``reject``: advance to Supervisor finalize (terminal rejection)

    VAL-TOPOLOGY-003: In supervisor_only, the re-route goes through
    the Supervisor (this routing function IS the Supervisor's
    routing logic).  There is NO direct Reviewer→Coder peer edge.
    """
    review = state.review_result
    if review is None:
        # No review result — should not happen, advance to finalize
        logger.warning("route_after_review called with no review_result")
        return "run_supervisor_finalize"

    if review.verdict == "accept":
        # Accept: advance to finalize
        return "run_supervisor_finalize"

    if review.verdict == "reject":
        # Terminal rejection: advance to finalize (will set outcome)
        return "run_supervisor_finalize"

    if review.verdict == "reject_with_changes":
        # Check retry budget for the current coder step
        step_key = f"coder_{state.step_index}"
        current_retries = state.retry_counters.get(step_key, 0)

        if current_retries >= _MAX_RETRIES_PER_STEP:
            # Retry budget exhausted — halt
            logger.warning(
                "Retry budget exhausted for %s (task %s): %d attempts",
                step_key,
                state.task_id,
                current_retries,
            )
            return "halt_retry_exhausted"

        # Re-route to Coder (sequential loop through Supervisor)
        logger.info(
            "Re-routing to Coder after reject_with_changes (attempt %d/%d)",
            current_retries + 1,
            _MAX_RETRIES_PER_STEP,
        )
        return "run_coder"

    # Unknown verdict — advance to finalize
    logger.warning("Unknown review verdict: %s", review.verdict)
    return "run_supervisor_finalize"


# ── Helper: emit Langfuse trace events ─────────────────────────────


async def _emit_trace_event(
    *,
    task_id: str,
    trace_id: str,
    span_id: str | None,
    parent_span_id: str | None,
    name: str,
    event_type: str,
    metadata: dict[str, Any] | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: Decimal = Decimal("0"),
) -> None:
    """Broadcast a trace event via the WebSocket broadcaster."""
    broadcaster = get_trace_broadcaster()
    await broadcaster.publish(
        TraceEvent(
            type=event_type,
            task_id=task_id,
            trace_id=trace_id,
            span_id=span_id or "",
            parent_span_id=parent_span_id,
            name=name,
            span_type=SpanType.SPAN,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )
    )


# ── Node functions ──────────────────────────────────────────────────


async def run_planner(state: OrchestratorState) -> dict[str, Any]:
    """Node: Run the Planner agent.

    Creates a Langfuse span for the Planner under the Supervisor
    parent span.  Calls run_planner() with the IssueContext built
    from the task state.  Persists the ChangePlan as a decision row.

    VAL-TOPOLOGY-002: Planner is the first agent in the sequence.
    VAL-TOPOLOGY-003: Planner span's parent is the Supervisor span.
    """
    task_id = state.task_id
    trace_id = state.trace_id if hasattr(state, "trace_id") and state.trace_id else uuid4().hex
    supervisor_span_id = (
        state.supervisor_span_id
        if hasattr(state, "supervisor_span_id") and state.supervisor_span_id
        else ""
    )

    logger.info("Planner node for task %s", task_id)

    # Create Planner span under Supervisor parent (VAL-TOPOLOGY-003)
    tracing = get_tracing_client()
    planner_span_id = tracing.create_span(
        trace_id=trace_id,
        parent_span_id=supervisor_span_id or None,
        name="planner",
        span_type=SpanType.SPAN,
        input_data={"issue_text": state.issue_text[:500]},
        metadata={
            "task_id": task_id,
            "agent_name": "planner",
            "topology": "supervisor_only",
        },
        start_time=datetime.now(UTC),
    )

    # Broadcast node_start event
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=planner_span_id or "",
        parent_span_id=supervisor_span_id,
        name="planner",
        event_type="node_start",
        metadata={"agent_name": "planner"},
    )

    # Build IssueContext from state
    sandbox = get_sandbox(task_id)
    repo_files: dict[str, str] = {}
    if sandbox is not None:
        try:
            file_list = await sandbox.run_command(
                "find . -type f -name '*.py' -not -path '*/.git/*' | head -20"
            )
            for filepath in file_list.strip().split("\n"):
                filepath = filepath.strip().lstrip("./")
                if filepath and filepath.endswith(".py"):
                    try:
                        content = await sandbox.read_file(filepath)
                        repo_files[filepath] = content[:2000]
                    except Exception:
                        pass
        except Exception:
            pass

    # ── Episodic memory pre-fetch (VAL-CROSS-013, VAL-CROSS-014) ───
    # Query the episodic store for repo_facts, recent decisions,
    # and recent outcomes scoped to the current repo_url BEFORE
    # building the IssueContext.  This ensures the Planner has
    # immediate context from prior tasks on the same repo.
    repo_facts_data: list[dict[str, Any]] = []
    recent_decisions_data: list[dict[str, Any]] = []
    recent_outcomes_data: list[dict[str, Any]] = []

    store = get_store(task_id)
    if store is not None:
        try:
            episodic_ctx = await store.get_planner_context(
                repo_url=state.repo_url,
                recent_limit=5,
            )
            repo_facts_data = episodic_ctx.get("repo_facts", [])
            recent_decisions_data = episodic_ctx.get("recent_decisions", [])
            recent_outcomes_data = episodic_ctx.get("recent_outcomes", [])
            logger.info(
                "Planner pre-fetched episodic context for %s: "
                "%d facts, %d decisions, %d outcomes",
                state.repo_url,
                len(repo_facts_data),
                len(recent_decisions_data),
                len(recent_outcomes_data),
            )
        except Exception as exc:
            logger.warning(
                "Episodic context pre-fetch failed for task %s: %s",
                task_id, exc,
            )

    issue_context = IssueContext(
        repo_url=state.repo_url,
        issue_number=state.issue_number,
        issue_text=state.issue_text,
        repo_files=repo_files,
        repo_facts=repo_facts_data,
        recent_decisions=recent_decisions_data,
    )

    # Run the Planner agent
    plan: ChangePlan | None = None
    cost_usd = Decimal("0")
    tokens_in = 0
    tokens_out = 0

    try:
        from src.agents.planner import run_planner as _run_planner_agent

        semantic_store = get_semantic_store(task_id)
        plan = await _run_planner_agent(
            issue_context=issue_context,
            episodic_store=store,
            rag_retriever=semantic_store,
            task_id=UUID(task_id) if task_id else None,
            trace_id=trace_id,
        )

        # Estimate cost from the Planner model
        cost_usd = estimate_cost_tiktoken(
            model="deepseek/deepseek-v4-pro",
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
        )

    except Exception as exc:
        logger.error("Planner agent failed for task %s: %s", task_id, exc)
        tracing.update_span(
            trace_id=trace_id,
            span_id=planner_span_id or "",
            output_data={"error": str(exc)[:500]},
            end_time=datetime.now(UTC),
            level="ERROR",
        )
        await _emit_trace_event(
            task_id=task_id,
            trace_id=trace_id,
            span_id=planner_span_id or "",
            parent_span_id=supervisor_span_id,
            name="planner",
            event_type="node_end",
            metadata={"agent_name": "planner", "error": str(exc)[:200]},
        )
        return {
            "errors": [f"Planner failed: {exc}"],
            "outcome": "sandbox_failure",
            "status": "failed",
        }

    # Update Planner span end
    tracing.update_span(
        trace_id=trace_id,
        span_id=planner_span_id or "",
        output_data=plan.model_dump(mode="json") if plan else None,
        end_time=datetime.now(UTC),
        metadata={
            "agent_name": "planner",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": str(cost_usd),
        },
    )

    # Broadcast node_end event
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=planner_span_id or "",
        parent_span_id=supervisor_span_id,
        name="planner",
        event_type="node_end",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        metadata={"agent_name": "planner"},
    )

    new_total_cost = state.total_cost_usd + cost_usd

    # Check cost budget
    max_cost = get_max_cost_per_task()
    if new_total_cost > max_cost:
        logger.warning(
            "Cost budget exceeded for task %s: $%s > $%s",
            task_id, new_total_cost, max_cost,
        )
        return {
            "outcome": "cost_budget_exhausted",
            "total_cost_usd": new_total_cost,
            "status": "failed",
        }

    return {
        "change_plan": plan,
        "issue_context": issue_context,
        "total_cost_usd": new_total_cost,
        "total_tokens_in": state.total_tokens_in + tokens_in,
        "total_tokens_out": state.total_tokens_out + tokens_out,
        "step_index": state.step_index,
    }


async def run_coder(state: OrchestratorState) -> dict[str, Any]:
    """Node: Run the Coder agent.

    Creates a Langfuse span for the Coder under the Supervisor
    parent span.  Calls run_coder() with the ChangePlan and
    optional ReviewResult from prior review.  Persists the
    CodeEdit as a decision row.

    On reject_with_changes loops, the Coder receives the prior
    ReviewResult as feedback to produce a different diff
    (VAL-CODER-007).

    VAL-TOPOLOGY-003: Coder span's parent is the Supervisor span.
    """
    task_id = state.task_id
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""

    step_key = f"coder_{state.step_index}"
    current_retries = state.retry_counters.get(step_key, 0)

    logger.info(
        "Coder node for task %s (step %d, retry %d)",
        task_id, state.step_index, current_retries,
    )

    # Create Coder span under Supervisor parent (VAL-TOPOLOGY-003)
    tracing = get_tracing_client()
    coder_span_id = tracing.create_span(
        trace_id=trace_id,
        parent_span_id=supervisor_span_id or None,
        name="coder",
        span_type=SpanType.SPAN,
        input_data={
            "change_plan": _trunc_json(state.change_plan) if state.change_plan else None,
            "review_result": _trunc_json(state.review_result) if state.review_result else None,
        },
        metadata={
            "task_id": task_id,
            "agent_name": "coder",
            "retry": current_retries,
        },
        start_time=datetime.now(UTC),
    )

    # Broadcast node_start event
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=coder_span_id or "",
        parent_span_id=supervisor_span_id,
        name="coder",
        event_type="node_start",
        metadata={"agent_name": "coder", "retry": current_retries},
    )

    # Run the Coder agent
    edit: CodeEdit | None = None
    cost_usd = Decimal("0")
    tokens_in = 0
    tokens_out = 0

    if state.change_plan is not None:
        try:
            from src.agents.coder import run_coder as _run_coder_agent

            sandbox = get_sandbox(task_id)
            store = get_store(task_id)
            edit = await _run_coder_agent(
                change_plan=state.change_plan,
                review_result=state.review_result,
                sandbox_manager=sandbox,
                episodic_store=store,
                task_id=UUID(task_id) if task_id else None,
                trace_id=trace_id,
                repo_url=state.repo_url,
            )

            cost_usd = estimate_cost_tiktoken(
                model="deepseek/deepseek-chat-v3-0324",
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
            )

        except Exception as exc:
            logger.error("Coder agent failed for task %s: %s", task_id, exc)
            tracing.update_span(
                trace_id=trace_id,
                span_id=coder_span_id or "",
                output_data={"error": str(exc)[:500]},
                end_time=datetime.now(UTC),
                level="ERROR",
            )
            await _emit_trace_event(
                task_id=task_id,
                trace_id=trace_id,
                span_id=coder_span_id or "",
                parent_span_id=supervisor_span_id,
                name="coder",
                event_type="node_end",
                metadata={"agent_name": "coder", "error": str(exc)[:200]},
            )
            return {
                "errors": [f"Coder failed: {exc}"],
                "outcome": "sandbox_failure",
                "status": "failed",
            }

    # Update Coder span end
    tracing.update_span(
        trace_id=trace_id,
        span_id=coder_span_id or "",
        output_data=edit.model_dump(mode="json") if edit else None,
        end_time=datetime.now(UTC),
        metadata={
            "agent_name": "coder",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": str(cost_usd),
        },
    )

    # Broadcast node_end event
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=coder_span_id or "",
        parent_span_id=supervisor_span_id,
        name="coder",
        event_type="node_end",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        metadata={"agent_name": "coder"},
    )

    # Increment retry counter
    new_retry_counters = dict(state.retry_counters)
    new_retry_counters[step_key] = current_retries + 1

    new_total_cost = state.total_cost_usd + cost_usd

    # Check cost budget
    max_cost = get_max_cost_per_task()
    if new_total_cost > max_cost:
        return {
            "outcome": "cost_budget_exhausted",
            "total_cost_usd": new_total_cost,
            "status": "failed",
        }

    return {
        "code_edit": edit,
        "retry_counters": new_retry_counters,
        "total_cost_usd": new_total_cost,
        "total_tokens_in": state.total_tokens_in + tokens_in,
        "total_tokens_out": state.total_tokens_out + tokens_out,
    }


async def run_reviewer(state: OrchestratorState) -> dict[str, Any]:
    """Node: Run the Reviewer agent.

    Creates a Langfuse span for the Reviewer under the Supervisor
    parent span.  Calls run_reviewer() with the CodeEdit.
    Persists the ReviewResult as a decision row.

    VAL-TOPOLOGY-003: Reviewer span's parent is the Supervisor span,
    NOT the Coder span.  This is the key structural invariant.
    """
    task_id = state.task_id
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""

    logger.info("Reviewer node for task %s (step %d)", task_id, state.step_index)

    # Create Reviewer span under Supervisor parent (VAL-TOPOLOGY-003)
    tracing = get_tracing_client()
    reviewer_span_id = tracing.create_span(
        trace_id=trace_id,
        parent_span_id=supervisor_span_id or None,
        name="reviewer",
        span_type=SpanType.SPAN,
        input_data={
            "code_edit": _trunc_json(state.code_edit) if state.code_edit else None,
        },
        metadata={
            "task_id": task_id,
            "agent_name": "reviewer",
        },
        start_time=datetime.now(UTC),
    )

    # Broadcast node_start event
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=reviewer_span_id or "",
        parent_span_id=supervisor_span_id,
        name="reviewer",
        event_type="node_start",
        metadata={"agent_name": "reviewer"},
    )

    # Run the Reviewer agent
    review: ReviewResult | None = None
    cost_usd = Decimal("0")
    tokens_in = 0
    tokens_out = 0

    if state.code_edit is not None:
        try:
            from src.agents.reviewer import run_reviewer as _run_reviewer_agent

            sandbox = get_sandbox(task_id)
            store = get_store(task_id)
            review = await _run_reviewer_agent(
                code_edit=state.code_edit,
                sandbox_manager=sandbox,
                episodic_store=store,
                task_id=UUID(task_id) if task_id else None,
                trace_id=trace_id,
                repo_url=state.repo_url,
            )

            cost_usd = estimate_cost_tiktoken(
                model="deepseek/deepseek-chat-v3-0324",
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
            )

        except Exception as exc:
            logger.error("Reviewer agent failed for task %s: %s", task_id, exc)
            tracing.update_span(
                trace_id=trace_id,
                span_id=reviewer_span_id or "",
                output_data={"error": str(exc)[:500]},
                end_time=datetime.now(UTC),
                level="ERROR",
            )
            await _emit_trace_event(
                task_id=task_id,
                trace_id=trace_id,
                span_id=reviewer_span_id or "",
                parent_span_id=supervisor_span_id,
                name="reviewer",
                event_type="node_end",
                metadata={"agent_name": "reviewer", "error": str(exc)[:200]},
            )
            return {
                "errors": [f"Reviewer failed: {exc}"],
                "outcome": "sandbox_failure",
                "status": "failed",
            }

    # Update Reviewer span end
    tracing.update_span(
        trace_id=trace_id,
        span_id=reviewer_span_id or "",
        output_data=review.model_dump(mode="json") if review else None,
        end_time=datetime.now(UTC),
        metadata={
            "agent_name": "reviewer",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": str(cost_usd),
            "verdict": review.verdict if review else None,
        },
    )

    # Broadcast node_end event
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=reviewer_span_id or "",
        parent_span_id=supervisor_span_id,
        name="reviewer",
        event_type="node_end",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        metadata={"agent_name": "reviewer", "verdict": review.verdict if review else None},
    )

    new_total_cost = state.total_cost_usd + cost_usd

    # Check cost budget
    max_cost = get_max_cost_per_task()
    if new_total_cost > max_cost:
        return {
            "outcome": "cost_budget_exhausted",
            "total_cost_usd": new_total_cost,
            "status": "failed",
        }

    return {
        "review_result": review,
        "total_cost_usd": new_total_cost,
        "total_tokens_in": state.total_tokens_in + tokens_in,
        "total_tokens_out": state.total_tokens_out + tokens_out,
    }


async def run_supervisor(state: OrchestratorState) -> dict[str, Any]:
    """Node: Supervisor routing node.

    The Supervisor is deterministic routing logic — not an LLM
    agent in supervisor_only topology (§2.3).  It:
    - Creates the top-level Langfuse span for the task
    - Routes between agents based on the current state
    - Owns the retry budget
    - Triggers HITL interrupts (in M4)
    - Initiates the final PR open call

    In Langfuse traces, this node creates a "supervisor" span
    that is the parent of all agent spans (planner, coder,
    reviewer).  This ensures no peer-handoff parent edges
    (VAL-TOPOLOGY-003).
    """
    task_id = state.task_id
    logger.info("Supervisor node for task %s", task_id)

    # Create the supervisor span (top-level under the trace)
    trace_id = uuid4().hex
    tracing = get_tracing_client()

    trace_span_id = tracing.create_trace(
        trace_id=trace_id,
        name=f"task.{task_id}.supervisor_only",
        metadata={
            "task_id": task_id,
            "topology": "supervisor_only",
        },
    )

    supervisor_span_id = tracing.create_span(
        trace_id=trace_id,
        parent_span_id=trace_span_id,
        name="supervisor",
        span_type=SpanType.SPAN,
        input_data={
            "repo_url": state.repo_url,
            "issue_number": state.issue_number,
        },
        metadata={
            "task_id": task_id,
            "agent_name": "supervisor",
            "topology": "supervisor_only",
        },
        start_time=datetime.now(UTC),
    )

    # Broadcast supervisor node_start event
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=supervisor_span_id or "",
        parent_span_id=trace_span_id,
        name="supervisor",
        event_type="node_start",
        metadata={"agent_name": "supervisor"},
    )

    return {
        "trace_id": trace_id,
        "supervisor_span_id": supervisor_span_id or "",
    }


async def run_supervisor_finalize(state: OrchestratorState) -> dict[str, Any]:
    """Node: Supervisor finalization after the review cycle.

    Handles:
    - Accept verdict: commit + push + open PR via GitHub client
    - Reject verdict: mark task as failed with appropriate outcome
    - Cost budget check

    In M4, this node will also handle HITL interrupts before PR.

    VAL-TOPOLOGY-003: The finalize node runs under the Supervisor
    parent span, maintaining the no-peer-handoff invariant.
    """
    task_id = state.task_id
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""

    review = state.review_result

    if review is not None and review.verdict == "reject":
        # Terminal rejection
        logger.info(
            "Supervisor finalize: terminal rejection for task %s",
            task_id,
        )
        tracing = get_tracing_client()
        tracing.update_span(
            trace_id=trace_id,
            span_id=supervisor_span_id,
            output_data={"outcome": "review_rejected", "verdict": "reject"},
            end_time=datetime.now(UTC),
        )
        await _emit_trace_event(
            task_id=task_id,
            trace_id=trace_id,
            span_id=supervisor_span_id,
            parent_span_id=None,
            name="supervisor",
            event_type="node_end",
            metadata={"agent_name": "supervisor", "outcome": "review_rejected"},
        )
        return {
            "outcome": "review_rejected",
            "status": "failed",
        }

    # Accept — attempt PR creation
    pr_url = ""
    sandbox = get_sandbox(task_id)
    if sandbox is not None and state.code_edit is not None:
        try:
            pat = os.getenv("GITHUB_PAT", "")
            username = os.getenv("GITHUB_USERNAME", "")
            gh_client = GitHubClient(pat=pat, username=username)

            repo_slug = canonicalize_repo_url(state.repo_url).replace(
                "https://github.com/", ""
            )
            branch_name = f"{_BRANCH_PREFIX}fix-issue-{state.issue_number}"

            gh_client.create_branch(str(sandbox.workspace_dir), branch_name)

            commit_msg = (
                f"fix: resolve issue #{state.issue_number}\n\n"
                f"Automated fix by SDLC-Swarm (supervisor_only topology)"
            )
            gh_client.commit_and_push(
                str(sandbox.workspace_dir), branch_name, commit_msg
            )

            pr_ref = gh_client.open_pull_request(
                repo=repo_slug,
                head_branch=branch_name,
                title=f"fix: resolve issue #{state.issue_number}",
                body=f"Automated fix for issue #{state.issue_number}",
            )
            pr_url = pr_ref.html_url
            logger.info("PR opened: %s", pr_url)

        except Exception as exc:
            logger.error("PR failed for task %s: %s", task_id, exc)

    # Update supervisor span end
    tracing = get_tracing_client()
    tracing.update_span(
        trace_id=trace_id,
        span_id=supervisor_span_id,
        output_data={"outcome": "pr_opened" if pr_url else "success", "pr_url": pr_url},
        end_time=datetime.now(UTC),
    )
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=supervisor_span_id,
        parent_span_id=None,
        name="supervisor",
        event_type="node_end",
        metadata={"agent_name": "supervisor", "outcome": "pr_opened" if pr_url else "success"},
    )

    if pr_url:
        return {
            "pr_url": pr_url,
            "outcome": "pr_opened",
            "status": "completed",
        }

    return {
        "outcome": "success",
        "status": "completed",
    }


async def halt_retry_exhausted(state: OrchestratorState) -> dict[str, Any]:
    """Node: Halt the task because the retry budget is exhausted.

    Writes an outcome row and marks the task as failed.
    In M4, this will also trigger an HITL interrupt.
    """
    task_id = state.task_id
    logger.warning(
        "Retry budget exhausted for task %s at step %d",
        task_id, state.step_index,
    )

    # Update supervisor span end with failure
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""
    tracing = get_tracing_client()
    tracing.update_span(
        trace_id=trace_id,
        span_id=supervisor_span_id,
        output_data={"outcome": "retry_budget_exhausted"},
        end_time=datetime.now(UTC),
        level="ERROR",
    )
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=supervisor_span_id,
        parent_span_id=None,
        name="supervisor",
        event_type="node_end",
        metadata={"agent_name": "supervisor", "outcome": "retry_budget_exhausted"},
    )

    return {
        "outcome": "retry_budget_exhausted",
        "status": "failed",
    }


# ── Index Repo Node ─────────────────────────────────────────────────
# Indexes the cloned repo into pgvector before the Planner runs.
# VAL-RAG-008: Indexer runs on task intake before Planner.


async def index_repo(state: OrchestratorState) -> dict[str, Any]:
    """Node: Index the cloned repo into the pgvector semantic store.

    Walks the cloned repo, filters by extension, chunks via
    token-aware splitter, embeds via OpenAI text-embedding-3-small,
    and writes rows to repo_chunks.

    VAL-RAG-008: The indexer span must end before the first Planner
    LLM completion span starts.

    This node runs between run_supervisor and run_planner.
    """
    task_id = state.task_id
    repo_url = state.repo_url

    logger.info("Indexing repo for task %s: %s", task_id, repo_url)

    # Get the sandbox (already provisioned by the Orchestrator)
    sandbox = get_sandbox(task_id)
    if sandbox is None:
        logger.error("No sandbox found for task %s during indexing", task_id)
        return {"errors": ["No sandbox available for indexing"]}

    # Create Langfuse span for indexing
    tracing = get_tracing_client()
    trace_id = state.trace_id
    supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""

    indexer_span_id = tracing.create_span(
        trace_id=trace_id,
        parent_span_id=supervisor_span_id or None,
        name="indexer.embed",
        span_type=SpanType.SPAN,
        input_data={"repo_url": repo_url},
        metadata={"task_id": task_id, "agent_name": "indexer"},
        start_time=datetime.now(UTC),
    )

    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=indexer_span_id or "",
        parent_span_id=supervisor_span_id,
        name="indexer.embed",
        event_type="node_start",
        metadata={"agent_name": "indexer"},
    )

    chunk_count = 0
    try:
        from src.memory.semantic.store import SemanticStore

        # Use the registered semantic store if available, otherwise create one
        semantic_store = get_semantic_store(task_id)
        if semantic_store is None:
            semantic_store = SemanticStore()
            await semantic_store.connect()

        chunk_count = await semantic_store.index_repo(
            repo_url=repo_url,
            repo_path=str(sandbox.workspace_dir),
        )
    except Exception as exc:
        logger.warning("Indexing failed for task %s: %s", task_id, exc)
        # Indexing failure is non-fatal — the Planner can still work
        # without RAG hits, just with degraded context.

    # Update indexer span end
    tracing.update_span(
        trace_id=trace_id,
        span_id=indexer_span_id or "",
        output_data={"chunk_count": chunk_count},
        end_time=datetime.now(UTC),
        metadata={"task_id": task_id, "chunk_count": chunk_count},
    )

    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=indexer_span_id or "",
        parent_span_id=supervisor_span_id,
        name="indexer.embed",
        event_type="node_end",
        metadata={"agent_name": "indexer", "chunk_count": chunk_count},
    )

    logger.info(
        "Indexing complete for task %s: %d chunks", task_id, chunk_count
    )

    return {"step_index": state.step_index}


# ── Build the supervisor_only graph ────────────────────────────────


def build_supervisor_only_graph() -> StateGraph:  # type: ignore[type-arg]
    """Build the LangGraph state machine for the supervisor_only topology.

    The graph topology is::

        START → run_supervisor → index_repo → run_planner →
        route_after_planner → run_coder → route_after_coder →
        run_reviewer → route_after_review
        → (accept: run_supervisor_finalize → END) |
          (reject_with_changes: run_coder → ... loop) |
          (retry exhausted: halt_retry_exhausted → END)

    VAL-TOPOLOGY-002: The agent span first-occurrence order is
    exactly [planner, coder, reviewer] under the Supervisor parent.

    VAL-TOPOLOGY-003: No Coder→Reviewer peer handoff edges.  All
    routing decisions are made by the Supervisor (route_after_review).

    VAL-RAG-008: index_repo runs before run_planner.
    """
    graph = StateGraph(OrchestratorState)

    # ── Add agent nodes ────────────────────────────────────────
    graph.add_node("run_supervisor", run_supervisor)
    graph.add_node("index_repo", index_repo)
    graph.add_node("run_planner", run_planner)
    graph.add_node("run_coder", run_coder)
    graph.add_node("run_reviewer", run_reviewer)
    graph.add_node("run_supervisor_finalize", run_supervisor_finalize)
    graph.add_node("halt_retry_exhausted", halt_retry_exhausted)

    # ── Add edges ───────────────────────────────────────────────

    # START → Supervisor
    graph.add_edge(START, "run_supervisor")

    # Supervisor → Index Repo (VAL-RAG-008: index before planner)
    graph.add_edge("run_supervisor", "index_repo")

    # Index Repo → Planner (sequential)
    graph.add_edge("index_repo", "run_planner")

    # Planner → Coder (sequential)
    graph.add_edge("run_planner", "run_coder")

    # Coder → Reviewer (sequential)
    graph.add_edge("run_coder", "run_reviewer")

    # Reviewer → conditional routing (this IS the Supervisor's
    # routing decision, not a peer handoff)
    graph.add_conditional_edges(
        "run_reviewer",
        route_after_review,
        {
            "run_coder": "run_coder",  # reject_with_changes loop
            "run_supervisor_finalize": "run_supervisor_finalize",  # accept/reject
            "halt_retry_exhausted": "halt_retry_exhausted",  # budget exhausted
        },
    )

    # Supervisor finalize → END
    graph.add_edge("run_supervisor_finalize", END)

    # Halt retry exhausted → END
    graph.add_edge("halt_retry_exhausted", END)

    return graph
