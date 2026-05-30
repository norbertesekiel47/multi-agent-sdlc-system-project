"""LangGraph supervisor_only topology — sequential routing through Supervisor.

Supervisor routes between Planner → Coder → Reviewer → QA sequentially.
No peer handoff between Coder and Reviewer — all routing goes through
the Supervisor node (VAL-TOPOLOGY-002, VAL-TOPOLOGY-003).

If Reviewer verdict is ``accept``, the Supervisor routes to QA.
If Reviewer verdict is ``reject_with_changes``, the Supervisor
re-routes to Coder (sequential loop, not peer swarm).  The
Supervisor owns the retry budget and enforces it.

If QA produces a TestReport with failures (failed > 0), the
Supervisor does NOT proceed to PR creation (VAL-QA-005).

HITL checkpoints (M4):
  - VAL-HITL-CTRL-001: interrupt() fires before PR open (mandatory).
  - VAL-HITL-CTRL-002: When HITL_INTERRUPT_BEFORE_WRITE_OPS=true,
    interrupt() also fires before commit_and_push.
  - VAL-HITL-CTRL-003: When HITL_INTERRUPT_BEFORE_WRITE_OPS=false
    (default), interrupt fires ONLY before PR open.

Architecture reference: §2.2 LangGraph Orchestrator, §5 Topology Configurations.

Span hierarchy for Langfuse traces::

    Supervisor (routing span)
      ├── Planner (LLM span)
      ├── Coder (LLM span)
      ├── Reviewer (LLM span)
      └── QA (LLM span)

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
from typing import Any
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from src.agents.models import ChangePlan, CodeEdit, IssueContext, ReviewResult, TestReport
from src.github_client.client import GitHubClient, canonicalize_repo_url
from src.llm.cost import estimate_cost_tiktoken, get_max_cost_per_task
from src.orchestrator import OrchestratorState
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


# ── Per-task registries ────────────────────────────────────────────
# Sandbox, episodic-store, semantic-store, and guardrail handles live
# in src.orchestrator.registries (non-serializable, looked up by
# task_id).  They are re-exported here because hybrid.py imports them
# from this module and tests patch them at this path.

from src.guardrails.errors import GuardrailViolation  # noqa: E402

# Failure-mode + cost bookkeeping helpers live in
# src.orchestrator.bookkeeping.  Re-exported here because hybrid.py
# imports some of them from this module and tests patch/import them at
# this path.
from src.orchestrator.bookkeeping import (  # noqa: E402
    _accumulate_agent_cost,
    _check_loop_detection,
    _get_tool_call_history_update,
    _reconstruct_uncertainty,
    _record_diff_rejection_check,
    _record_pydantic_failure,
    _record_pydantic_success,
    _record_tool_call_and_check,
    _sync_uncertainty_to_state,
)

# HITL escalation nodes live in src.orchestrator.hitl_nodes.  Re-exported
# here because hybrid.py imports them from this module and tests
# patch/import them at this path.
from src.orchestrator.hitl_nodes import (  # noqa: E402
    _handle_guardrail_violation,
    hitl_guardrail_escalation,
    hitl_loop_detected,
    hitl_uncertainty_escalation,
)
from src.orchestrator.recording import (  # noqa: E402
    RecordingSandboxProxy,
    _clear_recordings,
    _get_recordings,
    _process_tool_call_recordings,
    _tool_call_recordings,
    get_recording_proxy,
)
from src.orchestrator.registries import (  # noqa: E402
    get_guardrail,
    get_sandbox,
    get_sandbox_proxy,
    get_semantic_store,
    get_store,
    register_guardrail,
    register_sandbox,
    register_semantic_store,
    register_store,
    unregister_guardrail,
    unregister_sandbox,
    unregister_semantic_store,
    unregister_store,
)

# Conditional-edge routing functions live in src.orchestrator.routing.
# Re-exported here because hybrid.py imports some of them from this
# module and tests patch/import them at this path.
from src.orchestrator.routing import (  # noqa: E402
    route_after_coder,
    route_after_commit_and_push,
    route_after_planner,
    route_after_qa,
    route_after_review,
)

__all__ = [
    "RecordingSandboxProxy",
    "_accumulate_agent_cost",
    "_check_loop_detection",
    "_clear_recordings",
    "_get_recordings",
    "_get_tool_call_history_update",
    "_handle_guardrail_violation",
    "_process_tool_call_recordings",
    "_record_diff_rejection_check",
    "_record_pydantic_failure",
    "_record_pydantic_success",
    "_record_tool_call_and_check",
    "_reconstruct_uncertainty",
    "_sync_uncertainty_to_state",
    "_tool_call_recordings",
    "get_guardrail",
    "get_recording_proxy",
    "get_sandbox",
    "get_sandbox_proxy",
    "get_semantic_store",
    "get_store",
    "hitl_guardrail_escalation",
    "hitl_loop_detected",
    "hitl_uncertainty_escalation",
    "register_guardrail",
    "register_sandbox",
    "register_semantic_store",
    "register_store",
    "route_after_coder",
    "route_after_commit_and_push",
    "route_after_planner",
    "route_after_qa",
    "route_after_review",
    "unregister_guardrail",
    "unregister_sandbox",
    "unregister_semantic_store",
    "unregister_store",
]


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
    cached_tokens: int = 0,
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
            cached_tokens=cached_tokens,
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
                        logger.debug(
                            "repo-file index read failed for %s; skipping", filepath, exc_info=True
                        )
        except Exception:
            logger.debug(
                "repo-file indexing failed; continuing with partial context", exc_info=True
            )

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
    cached_tokens = 0

    try:
        from src.agents.planner import PlannerRunResult
        from src.agents.planner import run_planner as _run_planner_agent

        semantic_store = get_semantic_store(task_id)
        planner_result = await _run_planner_agent(
            issue_context=issue_context,
            episodic_store=store,
            rag_retriever=semantic_store,
            task_id=UUID(task_id) if task_id else None,
            trace_id=trace_id,
            return_metadata=True,
        )
        if isinstance(planner_result, PlannerRunResult):
            plan = planner_result.plan
            tokens_in = planner_result.tokens_in
            tokens_out = planner_result.tokens_out
            cached_tokens = planner_result.cached_tokens
            cost_usd = planner_result.cost_usd
        else:
            plan = planner_result
            # Backward-compatible fallback for tests or custom agent shims.
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
        # Record Pydantic validation failure if applicable
        if isinstance(exc, ValidationError):
            escalation_update = _record_pydantic_failure(
                agent_name="planner",
                step_index=state.step_index,
                state=state,
            )
            if escalation_update is not None and escalation_update.get(
                "outcome"
            ) == "uncertainty_escalation":
                return {**escalation_update}
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
        cached_tokens=cached_tokens,
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

    # Record successful Pydantic validation (resets failure counter)
    success_update = _record_pydantic_success(
        agent_name="planner",
        step_index=state.step_index,
        state=state,
    )

    # Accumulate per-agent cost
    agent_cost_update = _accumulate_agent_cost(
        agent_name="planner",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cached_tokens=0,
        cost_usd=cost_usd,
        state=state,
    )

    return {
        "change_plan": plan,
        "issue_context": issue_context,
        "total_cost_usd": new_total_cost,
        "total_tokens_in": state.total_tokens_in + tokens_in,
        "total_tokens_out": state.total_tokens_out + tokens_out,
        "step_index": state.step_index,
        **success_update,
        **agent_cost_update,
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
    cached_tokens = 0

    if state.change_plan is not None:
        try:
            from src.agents.coder import CoderRunResult
            from src.agents.coder import run_coder as _run_coder_agent

            sandbox = get_sandbox(task_id)
            store = get_store(task_id)
            # Use recording sandbox proxy for agent deps
            # (VAL-GUARDRAIL-011: middleware intercepts tool calls before executor)
            # Also records tool calls for loop detection + uncertainty escalation
            sandbox_proxy = get_recording_proxy(task_id) or get_sandbox_proxy(task_id)
            coder_result: CoderRunResult = await _run_coder_agent(
                change_plan=state.change_plan,
                review_result=state.review_result,
                sandbox_manager=sandbox_proxy or sandbox,
                episodic_store=store,
                task_id=UUID(task_id) if task_id else None,
                trace_id=trace_id,
                repo_url=state.repo_url,
            )
            edit = coder_result.edit
            tokens_in = coder_result.tokens_in
            tokens_out = coder_result.tokens_out
            cached_tokens = coder_result.cached_tokens
            cost_usd = coder_result.cost_usd

        except GuardrailViolation as gv:
            # VAL-GUARDRAIL-007..009: report to Langfuse + outcomes + HITL
            logger.warning(
                "Guardrail violation in Coder for task %s: %s",
                task_id, gv.rule_name,
            )
            return await _handle_guardrail_violation(
                violation=gv,
                task_id=task_id,
                trace_id=trace_id,
                span_id=coder_span_id,
                parent_span_id=supervisor_span_id,
            )
        except ValidationError as ve:
            # Record Pydantic validation failure for uncertainty escalation
            logger.warning(
                "Coder output failed Pydantic validation for task %s: %s",
                task_id, ve,
            )
            tracing.update_span(
                trace_id=trace_id,
                span_id=coder_span_id or "",
                output_data={"error": f"ValidationError: {ve}"[:500]},
                end_time=datetime.now(UTC),
                level="ERROR",
            )
            escalation_update = _record_pydantic_failure(
                agent_name="coder",
                step_index=state.step_index,
                state=state,
            )
            if escalation_update is not None:
                return {**escalation_update}
            return {
                "errors": [f"Coder validation failed: {ve}"],
                "outcome": "sandbox_failure",
                "status": "failed",
            }
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
            "cached_tokens": cached_tokens,
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
        cached_tokens=cached_tokens,
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

    # Record successful Pydantic validation (resets failure counter)
    success_update = _record_pydantic_success(
        agent_name="coder",
        step_index=state.step_index,
        state=state,
    )

    # Process recorded tool calls for loop detection + uncertainty
    tool_recordings_update = await _process_tool_call_recordings(
        agent_name="coder",
        step_index=state.step_index,
        state=state,
        trace_id=trace_id,
        parent_span_id=supervisor_span_id,
    )
    if tool_recordings_update is not None and tool_recordings_update.get("outcome") in (
        "loop_detected",
        "uncertainty_escalation",
    ):
        return {**tool_recordings_update, **success_update}

    # Accumulate per-agent cost
    coder_agent_cost_update = _accumulate_agent_cost(
        agent_name="coder",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        state=state,
    )

    return {
        "code_edit": edit,
        "retry_counters": new_retry_counters,
        "total_cost_usd": new_total_cost,
        "total_tokens_in": state.total_tokens_in + tokens_in,
        "total_tokens_out": state.total_tokens_out + tokens_out,
        "total_tokens_cached": state.total_tokens_cached + cached_tokens,
        **success_update,
        **(tool_recordings_update or {}),
        **coder_agent_cost_update,
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
    cached_tokens = 0

    if state.code_edit is not None:
        try:
            from src.agents.reviewer import ReviewerRunResult
            from src.agents.reviewer import run_reviewer as _run_reviewer_agent

            sandbox = get_sandbox(task_id)
            store = get_store(task_id)
            # Use recording sandbox proxy (VAL-GUARDRAIL-011)
            sandbox_proxy = get_recording_proxy(task_id) or get_sandbox_proxy(task_id)
            reviewer_result: ReviewerRunResult = await _run_reviewer_agent(
                code_edit=state.code_edit,
                sandbox_manager=sandbox_proxy or sandbox,
                episodic_store=store,
                task_id=UUID(task_id) if task_id else None,
                trace_id=trace_id,
                repo_url=state.repo_url,
            )
            review = reviewer_result.review
            tokens_in = reviewer_result.tokens_in
            tokens_out = reviewer_result.tokens_out
            cached_tokens = reviewer_result.cached_tokens
            cost_usd = reviewer_result.cost_usd

        except GuardrailViolation as gv:
            # VAL-GUARDRAIL-007..009: report to Langfuse + outcomes + HITL
            logger.warning(
                "Guardrail violation in Reviewer for task %s: %s",
                task_id, gv.rule_name,
            )
            return await _handle_guardrail_violation(
                violation=gv,
                task_id=task_id,
                trace_id=trace_id,
                span_id=reviewer_span_id,
                parent_span_id=supervisor_span_id,
            )
        except ValidationError as ve:
            # Record Pydantic validation failure for uncertainty escalation
            logger.warning(
                "Reviewer output failed Pydantic validation for task %s: %s",
                task_id, ve,
            )
            tracing.update_span(
                trace_id=trace_id,
                span_id=reviewer_span_id or "",
                output_data={"error": f"ValidationError: {ve}"[:500]},
                end_time=datetime.now(UTC),
                level="ERROR",
            )
            escalation_update = _record_pydantic_failure(
                agent_name="reviewer",
                step_index=state.step_index,
                state=state,
            )
            if escalation_update is not None:
                return {**escalation_update}
            return {
                "errors": [f"Reviewer validation failed: {ve}"],
                "outcome": "sandbox_failure",
                "status": "failed",
            }
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
            "cached_tokens": cached_tokens,
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
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        metadata={
            "agent_name": "reviewer",
            "verdict": review.verdict if review else None,
        },
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

    # Record successful Pydantic validation (resets failure counter)
    success_update = _record_pydantic_success(
        agent_name="reviewer",
        step_index=state.step_index,
        state=state,
    )

    # Record diff rejection for uncertainty escalation tracking
    # (VAL-UNCERTAINTY-003: same diff_hash rejected twice → escalation)
    diff_rejection_update: dict[str, Any] = {}
    if (
        review is not None
        and review.verdict == "reject_with_changes"
        and state.code_edit is not None
        and state.code_edit.diff_hash
    ):
        rejection_result = _record_diff_rejection_check(
            diff_hash=state.code_edit.diff_hash,
            agent_name="reviewer",
            step_index=state.step_index,
            state=state,
        )
        if (
            rejection_result is not None
            and rejection_result.get("outcome") == "uncertainty_escalation"
        ):
            # Same fix rejected twice → uncertainty escalation
            return {**rejection_result}
        # Merge non-escalation updates
        diff_rejection_update = {
            k: v for k, v in (rejection_result or {}).items()
            if k != "outcome" and k != "status" and k != "errors"
        }

    # Process recorded tool calls for loop detection + uncertainty
    tool_recordings_update = await _process_tool_call_recordings(
        agent_name="reviewer",
        step_index=state.step_index,
        state=state,
        trace_id=trace_id,
        parent_span_id=supervisor_span_id,
    )
    if tool_recordings_update is not None and tool_recordings_update.get("outcome") in (
        "loop_detected",
        "uncertainty_escalation",
    ):
        return {**tool_recordings_update, **success_update, **diff_rejection_update}

    # Accumulate per-agent cost
    reviewer_agent_cost_update = _accumulate_agent_cost(
        agent_name="reviewer",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        state=state,
    )

    return {
        "review_result": review,
        "total_cost_usd": new_total_cost,
        "total_tokens_in": state.total_tokens_in + tokens_in,
        "total_tokens_out": state.total_tokens_out + tokens_out,
        "total_tokens_cached": state.total_tokens_cached + cached_tokens,
        **success_update,
        **diff_rejection_update,
        **(tool_recordings_update or {}),
        **reviewer_agent_cost_update,
    }


# ── Route after QA ──────────────────────────────────────────────────


# ── Node: Run QA Agent ──────────────────────────────────────────────


async def run_qa(state: OrchestratorState) -> dict[str, Any]:
    """Node: Run the QA agent.

    Creates a Langfuse span for the QA under the Supervisor
    parent span.  Calls run_qa() with the CodeEdit.
    Persists the TestReport as a decision row.

    VAL-QA-001: QA emits typed TestReport.
    VAL-QA-002: QA generates test files in sandbox.
    VAL-QA-003: QA executes the test runner in sandbox.
    VAL-QA-004: TestReport persisted as decision.
    VAL-QA-005: QA failure does not auto-open PR.
    VAL-QA-006: failed_test_names length must equal failed.

    VAL-TOPOLOGY-003: QA span's parent is the Supervisor span.
    """
    task_id = state.task_id
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""

    logger.info("QA node for task %s (step %d)", task_id, state.step_index)

    # Create QA span under Supervisor parent (VAL-TOPOLOGY-003)
    tracing = get_tracing_client()
    qa_span_id = tracing.create_span(
        trace_id=trace_id,
        parent_span_id=supervisor_span_id or None,
        name="qa",
        span_type=SpanType.SPAN,
        input_data={
            "code_edit": _trunc_json(state.code_edit) if state.code_edit else None,
        },
        metadata={
            "task_id": task_id,
            "agent_name": "qa",
        },
        start_time=datetime.now(UTC),
    )

    # Broadcast node_start event
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=qa_span_id or "",
        parent_span_id=supervisor_span_id,
        name="qa",
        event_type="node_start",
        metadata={"agent_name": "qa"},
    )

    # Run the QA agent
    report: TestReport | None = None
    cost_usd = Decimal("0")
    tokens_in = 0
    tokens_out = 0
    cached_tokens = 0

    if state.code_edit is not None:
        try:
            from src.agents.qa import QARunResult
            from src.agents.qa import run_qa as _run_qa_agent

            sandbox = get_sandbox(task_id)
            store = get_store(task_id)
            # Use recording sandbox proxy (VAL-GUARDRAIL-011)
            sandbox_proxy = get_recording_proxy(task_id) or get_sandbox_proxy(task_id)
            qa_result: QARunResult = await _run_qa_agent(
                code_edit=state.code_edit,
                sandbox_manager=sandbox_proxy or sandbox,
                episodic_store=store,
                task_id=UUID(task_id) if task_id else None,
                trace_id=trace_id,
                repo_url=state.repo_url,
            )
            report = qa_result.report
            tokens_in = qa_result.tokens_in
            tokens_out = qa_result.tokens_out
            cached_tokens = qa_result.cached_tokens
            cost_usd = qa_result.cost_usd

        except GuardrailViolation as gv:
            # VAL-GUARDRAIL-007..009: report to Langfuse + outcomes + HITL
            logger.warning(
                "Guardrail violation in QA for task %s: %s",
                task_id, gv.rule_name,
            )
            return await _handle_guardrail_violation(
                violation=gv,
                task_id=task_id,
                trace_id=trace_id,
                span_id=qa_span_id,
                parent_span_id=supervisor_span_id,
            )
        except ValidationError as ve:
            # Record Pydantic validation failure for uncertainty escalation
            logger.warning(
                "QA output failed Pydantic validation for task %s: %s",
                task_id, ve,
            )
            tracing.update_span(
                trace_id=trace_id,
                span_id=qa_span_id or "",
                output_data={"error": f"ValidationError: {ve}"[:500]},
                end_time=datetime.now(UTC),
                level="ERROR",
            )
            escalation_update = _record_pydantic_failure(
                agent_name="qa",
                step_index=state.step_index,
                state=state,
            )
            if escalation_update is not None:
                return {**escalation_update}
            return {
                "errors": [f"QA validation failed: {ve}"],
                "outcome": "sandbox_failure",
                "status": "failed",
            }
        except Exception as exc:
            logger.error("QA agent failed for task %s: %s", task_id, exc)
            tracing.update_span(
                trace_id=trace_id,
                span_id=qa_span_id or "",
                output_data={"error": str(exc)[:500]},
                end_time=datetime.now(UTC),
                level="ERROR",
            )
            await _emit_trace_event(
                task_id=task_id,
                trace_id=trace_id,
                span_id=qa_span_id or "",
                parent_span_id=supervisor_span_id,
                name="qa",
                event_type="node_end",
                metadata={"agent_name": "qa", "error": str(exc)[:200]},
            )
            return {
                "errors": [f"QA failed: {exc}"],
                "outcome": "sandbox_failure",
                "status": "failed",
            }

    # Update QA span end
    tracing.update_span(
        trace_id=trace_id,
        span_id=qa_span_id or "",
        output_data=report.model_dump(mode="json") if report else None,
        end_time=datetime.now(UTC),
        metadata={
            "agent_name": "qa",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cached_tokens": cached_tokens,
            "cost_usd": str(cost_usd),
            "passed": report.passed if report else None,
            "failed": report.failed if report else None,
        },
    )

    # Broadcast node_end event
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=qa_span_id or "",
        parent_span_id=supervisor_span_id,
        name="qa",
        event_type="node_end",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        metadata={
            "agent_name": "qa",
            "passed": report.passed if report else None,
            "failed": report.failed if report else None,
        },
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

    # Record successful Pydantic validation (resets failure counter)
    success_update = _record_pydantic_success(
        agent_name="qa",
        step_index=state.step_index,
        state=state,
    )

    # Process recorded tool calls for loop detection + uncertainty
    tool_recordings_update = await _process_tool_call_recordings(
        agent_name="qa",
        step_index=state.step_index,
        state=state,
        trace_id=trace_id,
        parent_span_id=supervisor_span_id,
    )
    if tool_recordings_update is not None and tool_recordings_update.get("outcome") in (
        "loop_detected",
        "uncertainty_escalation",
    ):
        return {**tool_recordings_update, **success_update}

    # Accumulate per-agent cost
    qa_agent_cost_update = _accumulate_agent_cost(
        agent_name="qa",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        state=state,
    )

    return {
        "test_report": report,
        "total_cost_usd": new_total_cost,
        "total_tokens_in": state.total_tokens_in + tokens_in,
        "total_tokens_out": state.total_tokens_out + tokens_out,
        "total_tokens_cached": state.total_tokens_cached + cached_tokens,
        **success_update,
        **(tool_recordings_update or {}),
        **qa_agent_cost_update,
    }


async def halt_test_failure(state: OrchestratorState) -> dict[str, Any]:
    """Node: Halt the task because tests failed (VAL-QA-005).

    When TestReport.failed > 0, the orchestrator does NOT open a PR.
    Writes an outcome row and marks the task as failed.

    VAL-UNCERTAINTY-002: When persistent_test_failure trigger fires
    (3 consecutive failures), outcome becomes uncertainty_escalation
    instead of persistent_test_failure.
    """
    task_id = state.task_id
    report = state.test_report

    logger.warning(
        "Test failure halt for task %s: %d passed, %d failed",
        task_id,
        report.passed if report else 0,
        report.failed if report else 0,
    )

    # Check for uncertainty escalation: persistent test failure
    # Track test failure in state
    new_test_failure_count = state.test_failure_count + 1

    # Check if this triggers persistent_test_failure (3 consecutive)
    from src.failure_modes.uncertainty import _PERSISTENT_TEST_FAILURE_THRESHOLD

    if new_test_failure_count >= _PERSISTENT_TEST_FAILURE_THRESHOLD and not state.uncertainty_fired:
        # Persistent test failure → uncertainty escalation
        logger.warning(
            "Uncertainty escalation: persistent_test_failure for task %s (count=%d)",
            task_id, new_test_failure_count,
        )

        # Update supervisor span end with escalation
        trace_id = getattr(state, "trace_id", "") or uuid4().hex
        supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""
        tracing = get_tracing_client()
        tracing.update_span(
            trace_id=trace_id,
            span_id=supervisor_span_id,
            output_data={
                "outcome": "uncertainty_escalation",
                "trigger": "persistent_test_failure",
                "failed_test_names": report.failed_test_names if report else [],
            },
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
            metadata={
                "agent_name": "supervisor",
                "outcome": "uncertainty_escalation",
                "trigger": "persistent_test_failure",
            },
        )

        # Write outcome row
        store = get_store(task_id)
        if store is not None:
            from src.memory.episodic.models import CreateOutcomeParams
            await store.create_outcome(
                CreateOutcomeParams(
                    task_id=UUID(task_id),
                    outcome="uncertainty_escalation",
                    detail={
                        "trigger": "persistent_test_failure",
                        "consecutive_failures": new_test_failure_count,
                        "failed_test_names": report.failed_test_names if report else [],
                    },
                )
            )
            await store.update_task_status(UUID(task_id), "awaiting_hitl")

        return {
            "outcome": "uncertainty_escalation",
            "status": "awaiting_hitl",
            "test_failure_count": new_test_failure_count,
            "uncertainty_fired": True,
        }

    # Update supervisor span end with failure
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""
    tracing = get_tracing_client()
    tracing.update_span(
        trace_id=trace_id,
        span_id=supervisor_span_id,
        output_data={
            "outcome": "persistent_test_failure",
            "failed_test_names": report.failed_test_names if report else [],
        },
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
        metadata={
            "agent_name": "supervisor",
            "outcome": "persistent_test_failure",
            "failed_test_names": report.failed_test_names if report else [],
        },
    )

    return {
        "outcome": "persistent_test_failure",
        "status": "failed",
        "test_failure_count": new_test_failure_count,
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
    - Reject verdict: mark task as failed with appropriate outcome
    - Accept verdict: route to HITL pre-PR checkpoint

    In M4, the HITL interrupt fires before any PR open.
    After the HITL decision is resolved, the graph continues
    to commit+push+PR open.

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

    # Accept — route to HITL checkpoint
    return {"status": "pre_hitl"}


# ── HITL interrupt nodes (M4) ───────────────────────────────────────


async def hitl_pre_commit_push(state: OrchestratorState) -> dict[str, Any]:
    """Node: HITL interrupt before commit_and_push.

    VAL-HITL-CTRL-002: When HITL_INTERRUPT_BEFORE_WRITE_OPS=true,
    this interrupt fires before commit_and_push.
    The task status is set to 'awaiting_hitl' while paused.
    """
    task_id = state.task_id
    logger.info("HITL pre-commit-push checkpoint for task %s", task_id)

    # Update task status to awaiting_hitl
    store = get_store(task_id)
    if store is not None:
        await store.update_task_status(UUID(task_id), "awaiting_hitl")

    # Fire the interrupt — graph pauses here
    decision = interrupt({
        "reason": "pre_commit_push",
        "task_id": task_id,
        "op": "commit_and_push",
    })

    if decision == "reject":
        # Rejected — no commit, no PR
        if store is not None:
            from src.memory.episodic.models import CreateOutcomeParams
            await store.finish_task(UUID(task_id), "rejected", hitl_decision="reject")
            await store.create_outcome(
                CreateOutcomeParams(
                    task_id=UUID(task_id),
                    outcome="hitl_rejected",
                    detail={"checkpoint": "pre_commit_push"},
                )
            )
        return {"outcome": "hitl_rejected", "status": "rejected"}

    # Approved — continue
    if store is not None:
        await store.update_task_status(UUID(task_id), "running")

    return {"hitl_decision": "approve"}


async def hitl_pre_pr(state: OrchestratorState) -> dict[str, Any]:
    """Node: HITL interrupt before PR open.

    VAL-HITL-CTRL-001: This interrupt fires on EVERY task before
    the PR is opened.  Mandatory checkpoint.

    VAL-HITL-CTRL-003: When HITL_INTERRUPT_BEFORE_WRITE_OPS=false
    (default), this is the ONLY interrupt point.

    The task status is set to 'awaiting_hitl' while paused.
    After approval, the graph proceeds to commit+push+PR open.
    After rejection, the task ends with outcome='hitl_rejected'.
    """
    task_id = state.task_id
    logger.info("HITL pre-PR checkpoint for task %s", task_id)

    # Update task status to awaiting_hitl
    store = get_store(task_id)
    if store is not None:
        await store.update_task_status(UUID(task_id), "awaiting_hitl")

    # Broadcast HITL interrupt event
    broadcaster = get_trace_broadcaster()
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    await broadcaster.publish(
        TraceEvent(
            type="hitl_interrupt",
            task_id=task_id,
            trace_id=trace_id,
            span_id="",
            parent_span_id="",
            name="hitl_pre_pr",
            span_type=SpanType.SPAN,
            metadata={
                "reason": "pre_pr_approval",
                "agent_name": "supervisor",
            },
        )
    )

    # Fire the interrupt — graph pauses here
    decision = interrupt({
        "reason": "pre_pr_approval",
        "task_id": task_id,
        "op": "open_pull_request",
    })

    if decision == "reject":
        # Rejected — no PR opened
        if store is not None:
            from src.memory.episodic.models import CreateOutcomeParams
            await store.finish_task(UUID(task_id), "rejected", hitl_decision="reject")
            await store.create_outcome(
                CreateOutcomeParams(
                    task_id=UUID(task_id),
                    outcome="hitl_rejected",
                    detail={"checkpoint": "pre_pr_approval"},
                )
            )
        return {"outcome": "hitl_rejected", "status": "rejected"}

    # Approved — mark decision and proceed to commit+push+PR
    if store is not None:
        # Update hitl_decision field but keep status as running
        # (the finish_task at the end of the graph will set final status)
        # We need to set hitl_decision without finishing the task
        # Use a direct DB update
        pool = store.pool
        if pool is not None:
            await pool.execute(
                "UPDATE tasks SET hitl_decision = $2 WHERE id = $1",
                UUID(task_id),
                "approve",
            )
        await store.update_task_status(UUID(task_id), "approved")
        await store.update_task_status(UUID(task_id), "running")

    return {"hitl_decision": "approve"}


async def run_commit_and_push(state: OrchestratorState) -> dict[str, Any]:
    """Node: Commit and push changes to the GitHub branch.

    Separated from finalize so the HITL interrupt can fire
    between the commit and the PR open.
    """
    task_id = state.task_id
    sandbox = get_sandbox(task_id)

    if sandbox is None:
        logger.error("No sandbox for commit_and_push on task %s", task_id)
        return {
            "errors": ["No sandbox available for commit"],
            "outcome": "github_delivery_failed",
            "status": "failed",
        }

    try:
        pat = os.getenv("GITHUB_PAT", "")
        username = os.getenv("GITHUB_USERNAME", "")
        gh_client = GitHubClient(pat=pat, username=username)

        branch_name = f"{_BRANCH_PREFIX}fix-issue-{state.issue_number}"

        gh_client.create_branch(str(sandbox.workspace_dir), branch_name)

        commit_msg = (
            f"fix: resolve issue #{state.issue_number}\n\n"
            f"Automated fix by SDLC-Swarm (supervisor_only topology)"
        )
        gh_client.commit_and_push(
            str(sandbox.workspace_dir), branch_name, commit_msg
        )
        logger.info("Committed and pushed for task %s on branch %s", task_id, branch_name)
    except Exception as exc:
        logger.error("Commit/push failed for task %s: %s", task_id, exc)
        return {
            "errors": [f"Commit/push failed: {exc}"],
            "outcome": "github_delivery_failed",
            "status": "failed",
        }

    return {"step": "committed"}


async def run_open_pr(state: OrchestratorState) -> dict[str, Any]:
    """Node: Open a pull request on GitHub.

    VAL-HITL-CTRL-005: open_pull_request is invoked exactly once
    after HITL approval.
    VAL-HITL-CTRL-013: pr_url is populated only after successful PR open.
    """
    task_id = state.task_id
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""

    sandbox = get_sandbox(task_id)
    pr_url = ""
    error_message = ""

    if sandbox is not None and state.code_edit is not None:
        try:
            pat = os.getenv("GITHUB_PAT", "")
            username = os.getenv("GITHUB_USERNAME", "")
            gh_client = GitHubClient(pat=pat, username=username)

            repo_slug = canonicalize_repo_url(state.repo_url).replace(
                "https://github.com/", ""
            )
            branch_name = f"{_BRANCH_PREFIX}fix-issue-{state.issue_number}"

            pr_ref = gh_client.open_pull_request(
                repo=repo_slug,
                head_branch=branch_name,
                title=f"fix: resolve issue #{state.issue_number}",
                body=f"Automated fix for issue #{state.issue_number}",
            )
            pr_url = pr_ref.html_url
            logger.info("PR opened: %s", pr_url)

        except Exception as exc:
            error_message = f"PR open failed: {exc}"
            logger.error("PR open failed for task %s: %s", task_id, exc)
    else:
        error_message = "PR open failed: missing sandbox or code edit"
        logger.error("%s for task %s", error_message, task_id)

    # Update supervisor span end
    tracing = get_tracing_client()
    outcome = "pr_opened" if pr_url else "github_delivery_failed"
    tracing.update_span(
        trace_id=trace_id,
        span_id=supervisor_span_id,
        output_data={"outcome": outcome, "pr_url": pr_url},
        end_time=datetime.now(UTC),
    )
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=supervisor_span_id,
        parent_span_id=None,
        name="supervisor",
        event_type="node_end",
        metadata={"agent_name": "supervisor", "outcome": outcome},
    )

    if pr_url:
        return {
            "pr_url": pr_url,
            "outcome": "pr_opened",
            "status": "completed",
        }

    return {
        "errors": [error_message],
        "outcome": "github_delivery_failed",
        "status": "failed",
    }


async def halt_cost_budget_exhausted(state: OrchestratorState) -> dict[str, Any]:
    """Terminal node for exhausted per-task cost budget."""
    return {
        "outcome": "cost_budget_exhausted",
        "status": "failed",
    }


async def halt_github_delivery_failed(state: OrchestratorState) -> dict[str, Any]:
    """Terminal node for failed commit/push/PR delivery."""
    return {
        "outcome": "github_delivery_failed",
        "status": "failed",
        "errors": state.errors,
    }


async def halt_task_failed(state: OrchestratorState) -> dict[str, Any]:
    """Terminal node for failed agent/sandbox stages."""
    return {
        "outcome": state.outcome or "failed",
        "status": "failed",
        "errors": state.errors,
    }


async def halt_retry_exhausted(state: OrchestratorState) -> dict[str, Any]:
    """Node: Halt the task because the retry budget is exhausted.

    VAL-RETRY-002: Third failure triggers escalation, not a fourth attempt.
    VAL-RETRY-004: After retry_budget_exhausted is emitted, no further
    agent spans appear in the trace until HITL resolution.
    Writes an outcome row and triggers HITL interrupt.
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

    # Write outcome row
    store = get_store(task_id)
    if store is not None:
        from src.memory.episodic.models import CreateOutcomeParams
        await store.create_outcome(
            CreateOutcomeParams(
                task_id=UUID(task_id),
                outcome="retry_budget_exhausted",
                detail={
                    "step_index": state.step_index,
                    "retry_counters": state.retry_counters,
                },
            )
        )
        # Update task status to awaiting_hitl
        await store.update_task_status(UUID(task_id), "awaiting_hitl")

    # Broadcast HITL interrupt event
    broadcaster = get_trace_broadcaster()
    await broadcaster.publish(
        TraceEvent(
            type="hitl_interrupt",
            task_id=task_id,
            trace_id=trace_id,
            span_id="",
            parent_span_id="",
            name="hitl_retry_budget_exhausted",
            span_type=SpanType.SPAN,
            metadata={
                "cause": "retry_budget_exhausted",
                "detail": f"Retry budget exhausted at step {state.step_index}",
                "outcome": "retry_budget_exhausted",
            },
        )
    )

    # Fire the interrupt — graph pauses here for HITL
    decision = interrupt({
        "reason": "retry_budget_exhausted",
        "task_id": task_id,
        "cause": "retry_budget_exhausted",
        "explanation": f"Retry budget exhausted at step {state.step_index}",
    })

    if decision == "reject" or decision is None:
        # Rejected — end the task
        if store is not None:
            await store.finish_task(UUID(task_id), "rejected", hitl_decision="reject")
        return {"outcome": "retry_budget_exhausted", "status": "failed"}

    # Approved — resume with reset retry budget
    if store is not None:
        await store.update_task_status(UUID(task_id), "running")

    return {
        "hitl_decision": "approve",
        "outcome": "",
        "status": "running",
        "retry_counters": {},  # Reset all retry counters
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

    The graph topology includes failure-mode mitigation nodes (§2.9):

        START → run_supervisor → index_repo → run_planner →
        route_after_planner → run_coder → route_after_coder →
        run_reviewer → route_after_review
        → (accept: run_qa → route_after_qa →
            (all pass: run_supervisor_finalize →
              hitl_pre_commit_push (if BEFORE_WRITE_OPS) →
              hitl_pre_pr →
              run_commit_and_push → run_open_pr → END) |
            (failures: halt_test_failure → END or uncertainty_escalation)) |
          (reject_with_changes: run_coder → ... loop) |
          (reject: run_supervisor_finalize → END) |
          (retry exhausted: halt_retry_exhausted → HITL) |
          (loop_detected: hitl_loop_detected → HITL) |
          (uncertainty_escalation: hitl_uncertainty_escalation → HITL)

    VAL-TOPOLOGY-002: The agent span first-occurrence order is
    exactly [planner, coder, reviewer, qa] under the Supervisor parent.

    VAL-TOPOLOGY-003: No Coder→Reviewer peer handoff edges.  All
    routing decisions are made by the Supervisor (route_after_review).

    VAL-RAG-008: index_repo runs before run_planner.

    VAL-QA-001: QA emits typed TestReport.
    VAL-QA-005: QA failure does not auto-open PR.

    VAL-HITL-CTRL-001: interrupt() fires before PR open on every task.
    VAL-HITL-CTRL-002: BEFORE_WRITE_OPS=true → interrupt before every write.
    VAL-HITL-CTRL-003: BEFORE_WRITE_OPS=false → interrupt only before PR.

    VAL-RETRY-002: Retry exhaustion triggers HITL interrupt.
    VAL-LOOP-DETECT-004: Loop detection triggers HITL interrupt.
    VAL-UNCERTAINTY-007: Uncertainty escalation triggers HITL interrupt.
    """
    from src.orchestrator.hitl import is_before_write_ops

    graph = StateGraph(OrchestratorState)

    # ── Add agent nodes ────────────────────────────────────────
    graph.add_node("run_supervisor", run_supervisor)
    graph.add_node("index_repo", index_repo)
    graph.add_node("run_planner", run_planner)
    graph.add_node("run_coder", run_coder)
    graph.add_node("run_reviewer", run_reviewer)
    graph.add_node("run_qa", run_qa)
    graph.add_node("run_supervisor_finalize", run_supervisor_finalize)
    graph.add_node("halt_cost_budget_exhausted", halt_cost_budget_exhausted)
    graph.add_node("halt_github_delivery_failed", halt_github_delivery_failed)
    graph.add_node("halt_task_failed", halt_task_failed)
    graph.add_node("halt_retry_exhausted", halt_retry_exhausted)
    graph.add_node("halt_test_failure", halt_test_failure)

    # ── HITL checkpoint nodes (M4) ────────────────────────────
    graph.add_node("hitl_pre_pr", hitl_pre_pr)
    graph.add_node("hitl_pre_commit_push", hitl_pre_commit_push)
    graph.add_node("run_commit_and_push", run_commit_and_push)
    graph.add_node("run_open_pr", run_open_pr)

    # ── Guardrail escalation HITL node (VAL-GUARDRAIL-009) ────
    graph.add_node("hitl_guardrail_escalation", hitl_guardrail_escalation)

    # ── Failure-mode HITL escalation nodes (§2.9) ──────────────
    graph.add_node("hitl_loop_detected", hitl_loop_detected)
    graph.add_node("hitl_uncertainty_escalation", hitl_uncertainty_escalation)

    # ── Add edges ───────────────────────────────────────────────

    # START → Supervisor
    graph.add_edge(START, "run_supervisor")

    # Supervisor → Index Repo (VAL-RAG-008: index before planner)
    graph.add_edge("run_supervisor", "index_repo")

    # Index Repo → Planner (sequential)
    graph.add_edge("index_repo", "run_planner")

    # Planner → conditional routing (guardrail + loop + uncertainty + normal flow)
    graph.add_conditional_edges(
        "run_planner",
        route_after_planner,
        {
            "halt_cost_budget_exhausted": "halt_cost_budget_exhausted",
            "halt_task_failed": "halt_task_failed",
            "hitl_guardrail_escalation": "hitl_guardrail_escalation",
            "hitl_loop_detected": "hitl_loop_detected",
            "hitl_uncertainty_escalation": "hitl_uncertainty_escalation",
            "run_coder": "run_coder",
        },
    )

    # Coder → conditional routing (guardrail + loop + uncertainty + normal flow)
    graph.add_conditional_edges(
        "run_coder",
        route_after_coder,
        {
            "halt_cost_budget_exhausted": "halt_cost_budget_exhausted",
            "halt_task_failed": "halt_task_failed",
            "hitl_guardrail_escalation": "hitl_guardrail_escalation",
            "hitl_loop_detected": "hitl_loop_detected",
            "hitl_uncertainty_escalation": "hitl_uncertainty_escalation",
            "run_reviewer": "run_reviewer",
        },
    )

    # Reviewer → conditional routing (guardrail + loop + uncertainty + verdict routing)
    graph.add_conditional_edges(
        "run_reviewer",
        route_after_review,
        {
            "halt_cost_budget_exhausted": "halt_cost_budget_exhausted",
            "halt_task_failed": "halt_task_failed",
            "hitl_guardrail_escalation": "hitl_guardrail_escalation",
            "hitl_loop_detected": "hitl_loop_detected",
            "hitl_uncertainty_escalation": "hitl_uncertainty_escalation",
            "run_qa": "run_qa",  # accept → QA (VAL-REVIEWER-004)
            "run_coder": "run_coder",  # reject_with_changes loop
            "run_supervisor_finalize": "run_supervisor_finalize",  # reject
            "halt_retry_exhausted": "halt_retry_exhausted",  # budget exhausted
        },
    )

    # QA → conditional routing (guardrail + loop + uncertainty + test results)
    graph.add_conditional_edges(
        "run_qa",
        route_after_qa,
        {
            "halt_cost_budget_exhausted": "halt_cost_budget_exhausted",
            "halt_task_failed": "halt_task_failed",
            "hitl_guardrail_escalation": "hitl_guardrail_escalation",
            "hitl_loop_detected": "hitl_loop_detected",
            "hitl_uncertainty_escalation": "hitl_uncertainty_escalation",
            "run_supervisor_finalize": "run_supervisor_finalize",  # all pass → HITL + PR
            "halt_test_failure": "halt_test_failure",  # test failures → halt/escalation
        },
    )

    # ── HITL checkpoint path ──────────────────────────────────
    # After finalize (accept path), go through HITL checkpoints
    # before committing and opening the PR.

    if is_before_write_ops():
        # VAL-HITL-CTRL-002: Interrupt before EVERY write op
        # finalize → hitl_pre_commit_push → hitl_pre_pr → commit → open_pr
        graph.add_edge("run_supervisor_finalize", "hitl_pre_commit_push")
        graph.add_edge("hitl_pre_commit_push", "hitl_pre_pr")
    else:
        # VAL-HITL-CTRL-003: Interrupt ONLY before PR open
        # finalize → hitl_pre_pr → commit → open_pr
        graph.add_edge("run_supervisor_finalize", "hitl_pre_pr")

    graph.add_edge("hitl_pre_pr", "run_commit_and_push")
    graph.add_conditional_edges(
        "run_commit_and_push",
        route_after_commit_and_push,
        {
            "run_open_pr": "run_open_pr",
            "halt_github_delivery_failed": "halt_github_delivery_failed",
        },
    )
    graph.add_edge("run_open_pr", END)
    graph.add_edge("halt_cost_budget_exhausted", END)
    graph.add_edge("halt_github_delivery_failed", END)
    graph.add_edge("halt_task_failed", END)

    # Halt retry exhausted → END (VAL-RETRY-002: now triggers HITL)
    graph.add_edge("halt_retry_exhausted", END)

    # Halt test failure → END (VAL-QA-005; may produce uncertainty_escalation)
    graph.add_edge("halt_test_failure", END)

    # Guardrail escalation → END (after HITL decision resolved)
    graph.add_edge("hitl_guardrail_escalation", END)

    # Loop detection HITL → END (VAL-LOOP-DETECT-004)
    graph.add_edge("hitl_loop_detected", END)

    # Uncertainty escalation HITL → END (VAL-UNCERTAINTY-007)
    graph.add_edge("hitl_uncertainty_escalation", END)

    return graph
