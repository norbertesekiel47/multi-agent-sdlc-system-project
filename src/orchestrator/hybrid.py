"""LangGraph hybrid topology — supervisor routing + Coder ⇄ Reviewer peer handoff.

Same as supervisor_only plus a typed peer handoff between Coder ⇄ Reviewer.
When Reviewer returns ``reject_with_changes``, the handoff goes directly
Reviewer → Coder (peer edge) rather than through the Supervisor.

The Langfuse trace shows at least one Reviewer → Coder peer-handoff
parent edge when reject_with_changes occurs.  Concurrent tasks are
isolated: distinct sandbox containers, distinct Docker networks, no
cross-task decisions contamination, distinct Langfuse trace IDs.

Topology comparison (§5)::

    single_agent     1 agent, linear
    supervisor_only  4 agents, Supervisor routes all, no peer edges
    hybrid           4 agents, Supervisor routes most,
                    but Coder⇄Reviewer uses peer handoff

Span hierarchy for hybrid (accept path — same as supervisor_only)::

    Supervisor (routing span)
      ├── Planner (LLM span)
      ├── Coder (LLM span)
      ├── Reviewer (LLM span)
      └── QA (LLM span)

Span hierarchy for hybrid (reject_with_changes path — peer handoff)::

    Supervisor (routing span)
      ├── Planner (LLM span)
      ├── Coder (LLM span)        ← parent: Supervisor
      ├── Reviewer (LLM span)     ← parent: Supervisor
      │   └── Coder (peer span)   ← parent: Reviewer (PEER HANDOFF)
      │       └── Reviewer (peer) ← parent: peer Coder
      └── QA (LLM span)

VAL-TOPOLOGY-004: In hybrid, on a task that requires at least one
fix-review cycle, the trace contains at least one Reviewer→Coder
transition where the Coder span's parent is the Reviewer span
(peer handoff), not the Supervisor.

VAL-TOPOLOGY-005: When Reviewer issues reject_with_changes twice
in a row, the trace contains the sequence
coder→reviewer→coder→reviewer→coder with peer-handoff parent edges.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph

from src.agents.models import CodeEdit, ReviewResult
from src.guardrails.errors import GuardrailViolation
from src.llm.cost import get_max_cost_per_task
from src.orchestrator import OrchestratorState
from src.orchestrator.supervisor_only import (
    _MAX_RETRIES_PER_STEP,
    _emit_trace_event,
    _trunc_json,
    get_sandbox,
    get_store,
    halt_cost_budget_exhausted,
    halt_github_delivery_failed,
    halt_retry_exhausted,
    halt_task_failed,
    halt_test_failure,
    hitl_pre_commit_push,
    hitl_pre_pr,
    index_repo,
    route_after_commit_and_push,
    run_commit_and_push,
    run_open_pr,
    run_planner,
    run_qa,
    run_supervisor,
    run_supervisor_finalize,
)
from src.orchestrator.supervisor_only import (
    _accumulate_agent_cost as _accumulate_agent_cost,
)
from src.orchestrator.supervisor_only import (
    _handle_guardrail_violation as _handle_guardrail_violation,
)
from src.orchestrator.supervisor_only import (
    get_sandbox_proxy as get_sandbox_proxy,
)
from src.orchestrator.supervisor_only import (
    hitl_guardrail_escalation as hitl_guardrail_escalation,
)
from src.orchestrator.supervisor_only import (
    hitl_loop_detected as hitl_loop_detected,
)
from src.orchestrator.supervisor_only import (
    hitl_uncertainty_escalation as hitl_uncertainty_escalation,
)
from src.orchestrator.supervisor_only import (
    register_guardrail as register_guardrail,
)
from src.orchestrator.supervisor_only import (
    register_sandbox as register_sandbox,
)
from src.orchestrator.supervisor_only import (
    register_semantic_store as register_semantic_store,
)
from src.orchestrator.supervisor_only import (
    register_store as register_store,
)
from src.orchestrator.supervisor_only import (
    route_after_coder as route_after_coder,
)
from src.orchestrator.supervisor_only import (
    route_after_planner as route_after_planner,
)
from src.orchestrator.supervisor_only import (
    unregister_guardrail as unregister_guardrail,
)
from src.orchestrator.supervisor_only import (
    unregister_sandbox as unregister_sandbox,
)
from src.orchestrator.supervisor_only import (
    unregister_semantic_store as unregister_semantic_store,
)
from src.orchestrator.supervisor_only import (
    unregister_store as unregister_store,
)
from src.sandbox.manager import SandboxManager
from src.tracing.langfuse import SpanType, get_tracing_client

logger = logging.getLogger(__name__)


# ── Routing function for hybrid ────────────────────────────────────


def route_after_review_hybrid(
    state: OrchestratorState,
) -> Literal[
    "halt_cost_budget_exhausted",
    "halt_task_failed",
    "hitl_guardrail_escalation",
    "hitl_loop_detected",
    "hitl_uncertainty_escalation",
    "run_qa",
    "run_peer_coder",
    "run_supervisor_finalize",
    "halt_retry_exhausted",
]:
    """After Reviewer, route based on verdict (hybrid topology).

    Same as supervisor_only's route_after_review EXCEPT:
    - ``reject_with_changes``: route to **run_peer_coder** (peer
      handoff via peer edge, NOT through Supervisor).
    All other routing is the same.
    """
    if state.outcome == "cost_budget_exhausted":
        return "halt_cost_budget_exhausted"
    # Check for guardrail violations first
    if state.outcome == "guardrail_block":
        return "hitl_guardrail_escalation"
    # Check for loop detection
    if state.outcome == "loop_detected":
        return "hitl_loop_detected"
    # Check for uncertainty escalation
    if state.outcome == "uncertainty_escalation":
        return "hitl_uncertainty_escalation"
    if state.status == "failed":
        return "halt_task_failed"

    review = state.review_result
    if review is None:
        logger.warning("route_after_review_hybrid called with no review_result")
        return "run_supervisor_finalize"

    if review.verdict == "accept":
        # Accept: advance to QA (VAL-REVIEWER-004)
        return "run_qa"

    if review.verdict == "reject":
        # Terminal rejection
        return "run_supervisor_finalize"

    if review.verdict == "reject_with_changes":
        # Check retry budget
        step_key = f"coder_{state.step_index}"
        current_retries = state.retry_counters.get(step_key, 0)

        if current_retries >= _MAX_RETRIES_PER_STEP:
            logger.warning(
                "Retry budget exhausted for %s (task %s): %d attempts",
                step_key,
                state.task_id,
                current_retries,
            )
            return "halt_retry_exhausted"

        # VAL-TOPOLOGY-004: Route to peer Coder.
        logger.info(
            "Peer-handoff to Coder after reject_with_changes "
            "(attempt %d/%d, handoff #%d)",
            current_retries + 1,
            _MAX_RETRIES_PER_STEP,
            state.peer_handoff_count + 1,
        )
        return "run_peer_coder"

    logger.warning("Unknown review verdict: %s", review.verdict)
    return "run_supervisor_finalize"


# ── Route after QA ──────────────────────────────────────────────────
# Same logic as supervisor_only


def route_after_qa_hybrid(
    state: OrchestratorState,
) -> Literal[
    "halt_cost_budget_exhausted",
    "halt_task_failed",
    "hitl_guardrail_escalation",
    "hitl_loop_detected",
    "hitl_uncertainty_escalation",
    "run_supervisor_finalize",
    "halt_test_failure",
]:
    """After QA, route based on test results (same as supervisor_only)."""
    if state.outcome == "cost_budget_exhausted":
        return "halt_cost_budget_exhausted"
    # Check for guardrail violations first
    if state.outcome == "guardrail_block":
        return "hitl_guardrail_escalation"
    # Check for loop detection
    if state.outcome == "loop_detected":
        return "hitl_loop_detected"
    # Check for uncertainty escalation
    if state.outcome == "uncertainty_escalation":
        return "hitl_uncertainty_escalation"
    if state.status == "failed":
        return "halt_task_failed"

    report = state.test_report
    if report is None:
        logger.warning("route_after_qa_hybrid called with no test_report")
        return "run_supervisor_finalize"

    if report.failed > 0:
        return "halt_test_failure"

    return "run_supervisor_finalize"


# ── Route after peer Coder ──────────────────────────────────────────


def route_after_peer_coder(
    state: OrchestratorState,
) -> Literal[
    "halt_cost_budget_exhausted",
    "halt_task_failed",
    "hitl_guardrail_escalation",
    "hitl_loop_detected",
    "hitl_uncertainty_escalation",
    "run_reviewer",
]:
    """After peer Coder, route to guardrail/loop/uncertainty escalation or Reviewer."""
    if state.outcome == "cost_budget_exhausted":
        return "halt_cost_budget_exhausted"
    if state.outcome == "guardrail_block":
        return "hitl_guardrail_escalation"
    if state.outcome == "loop_detected":
        return "hitl_loop_detected"
    if state.outcome == "uncertainty_escalation":
        return "hitl_uncertainty_escalation"
    if state.status == "failed":
        return "halt_task_failed"
    return "run_reviewer"


# ── Node: Reviewer (hybrid) ─────────────────────────────────────────


async def run_reviewer_hybrid(state: OrchestratorState) -> dict[str, Any]:
    """Node: Run the Reviewer agent in hybrid topology.

    Same as supervisor_only's run_reviewer, but additionally:
    - Saves the Reviewer span ID to ``last_reviewer_span_id``
      so the peer Coder can use it as its parent span
    - Increments ``peer_handoff_count`` for tracking

    This is the key enabler for the peer handoff:
    when the next Coder span uses the Reviewer span as parent,
    the Langfuse trace shows a direct Reviewer → Coder parent edge.
    """
    task_id = state.task_id
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""

    # Determine parent span for this Reviewer invocation:
    # - First Reviewer: parent is Supervisor (same as supervisor_only)
    # - Subsequent Reviewer (after peer Coder): parent is the
    #   peer Coder span (continuing the peer loop)
    # We track whether this is a peer-loop Reviewer via peer_handoff_count
    is_peer_loop = state.peer_handoff_count > 0 and state.last_reviewer_span_id != ""

    logger.info(
        "Reviewer node (hybrid) for task %s (step %d, peer_loop=%s)",
        task_id,
        state.step_index,
        is_peer_loop,
    )

    # Create Reviewer span
    tracing = get_tracing_client()

    # In peer loop, the parent is the last Coder span (not Supervisor)
    # But we don't track last_coder_span_id — instead, in the first
    # Reviewer after a peer Coder, the parent is still Supervisor.
    # The PEER HANDOFF is captured by the Coder having Reviewer as parent,
    # not the other way around.
    # So the Reviewer's parent is always the Supervisor span.
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
            "topology": "hybrid",
            "peer_loop": is_peer_loop,
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
        metadata={"agent_name": "reviewer", "topology": "hybrid", "peer_loop": is_peer_loop},
    )

    # Run the Reviewer agent
    review: ReviewResult | None = None
    cost_usd = Decimal("0")
    tokens_in = 0
    tokens_out = 0
    cached_tokens = 0

    if state.code_edit is not None:
        try:
            reviewer_result = await _run_reviewer_agent(
                code_edit=state.code_edit,
                sandbox=get_sandbox(task_id),
                store=get_store(task_id),
                task_id=UUID(task_id) if task_id else None,
                trace_id=trace_id,
                repo_url=state.repo_url,
                sandbox_proxy=get_sandbox_proxy(task_id),
            )
            review = reviewer_result.review
            tokens_in = reviewer_result.tokens_in
            tokens_out = reviewer_result.tokens_out
            cached_tokens = reviewer_result.cached_tokens
            cost_usd = reviewer_result.cost_usd

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

    reviewer_agent_cost_update = _accumulate_agent_cost(
        agent_name="reviewer",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        state=state,
    )

    result: dict[str, Any] = {
        "review_result": review,
        "total_cost_usd": new_total_cost,
        "total_tokens_in": state.total_tokens_in + tokens_in,
        "total_tokens_out": state.total_tokens_out + tokens_out,
        "total_tokens_cached": state.total_tokens_cached + cached_tokens,
        # Save the Reviewer span ID for peer-handoff Coder parent
        "last_reviewer_span_id": reviewer_span_id or "",
        **reviewer_agent_cost_update,
    }

    # If this is a peer-loop Reviewer that results in reject_with_changes,
    # increment the peer_handoff_count
    if is_peer_loop and review and review.verdict == "reject_with_changes":
        result["peer_handoff_count"] = state.peer_handoff_count + 1

    return result


# ── Node: Peer Coder handoff ───────────────────────────────────────


async def run_peer_coder(state: OrchestratorState) -> dict[str, Any]:
    """Node: Run the Coder agent as a peer-handoff from Reviewer.

    This is the KEY differentiator from supervisor_only.  The Coder
    span's parent is the Reviewer span (not the Supervisor span),
    creating a direct Reviewer → Coder parent edge in the Langfuse
    trace. This implements the typed peer-handoff behavior used by
    the hybrid topology.

    VAL-TOPOLOGY-004: On a task with at least one fix-review cycle,
    the trace contains at least one Coder span whose parent span
    is a Reviewer span.

    VAL-TOPOLOGY-005: When multiple reject_with_changes cycles occur,
    each subsequent peer Coder uses the latest Reviewer span as parent,
    creating a chain of peer-handoff edges.
    """
    task_id = state.task_id
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""

    # The parent for the peer Coder span is the REVIEWER span
    # (not the Supervisor span).  This is the structural invariant
    # that distinguishes hybrid from supervisor_only.
    last_reviewer_span_id = state.last_reviewer_span_id

    step_key = f"coder_{state.step_index}"
    current_retries = state.retry_counters.get(step_key, 0)

    logger.info(
        "Peer Coder node (hybrid) for task %s (step %d, retry %d, handoff #%d)",
        task_id,
        state.step_index,
        current_retries,
        state.peer_handoff_count + 1,
    )

    # Create Coder span with Reviewer as parent (VAL-TOPOLOGY-004)
    tracing = get_tracing_client()
    coder_span_id = tracing.create_span(
        trace_id=trace_id,
        parent_span_id=last_reviewer_span_id or supervisor_span_id or None,
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
            "handoff_type": "peer",
            "parent_agent_name": "reviewer",
            "peer_handoff_count": state.peer_handoff_count + 1,
            "topology": "hybrid",
        },
        start_time=datetime.now(UTC),
    )

    # Broadcast node_start event
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=coder_span_id or "",
        parent_span_id=last_reviewer_span_id or supervisor_span_id,
        name="coder",
        event_type="node_start",
        metadata={
            "agent_name": "coder",
            "retry": current_retries,
            "handoff_type": "peer",
            "parent_agent_name": "reviewer",
        },
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
            # Use guardrail-wrapped sandbox proxy (VAL-GUARDRAIL-011)
            sandbox_proxy = get_sandbox_proxy(task_id)
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
            # VAL-GUARDRAIL-007..009
            logger.warning(
                "Guardrail violation in peer Coder for task %s: %s",
                task_id, gv.rule_name,
            )
            return await _handle_guardrail_violation(
                violation=gv,
                task_id=task_id,
                trace_id=trace_id,
                span_id=coder_span_id,
                parent_span_id=last_reviewer_span_id,
            )
        except Exception as exc:
            logger.error("Peer Coder agent failed for task %s: %s", task_id, exc)
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
                parent_span_id=last_reviewer_span_id or supervisor_span_id,
                name="coder",
                event_type="node_end",
                metadata={"agent_name": "coder", "error": str(exc)[:200]},
            )
            return {
                "errors": [f"Peer Coder failed: {exc}"],
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
            "handoff_type": "peer",
        },
    )

    # Broadcast node_end event
    await _emit_trace_event(
        task_id=task_id,
        trace_id=trace_id,
        span_id=coder_span_id or "",
        parent_span_id=last_reviewer_span_id or supervisor_span_id,
        name="coder",
        event_type="node_end",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        metadata={
            "agent_name": "coder",
            "handoff_type": "peer",
        },
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
        "total_tokens_cached": state.total_tokens_cached + cached_tokens,
        # Increment peer handoff count
        "peer_handoff_count": state.peer_handoff_count + 1,
        **_accumulate_agent_cost(
            agent_name="coder",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            state=state,
        ),
    }


# ── Helper: Run Reviewer agent ──────────────────────────────────────


async def _run_reviewer_agent(
    *,
    code_edit: CodeEdit,
    sandbox: SandboxManager | None,
    store: Any,
    task_id: UUID | None,
    trace_id: str,
    repo_url: str,
    sandbox_proxy: Any | None = None,
) -> Any:
    """Run the Reviewer PydanticAI agent and return the result.

    Extracted from the node function for testability.
    """
    from src.agents.reviewer import run_reviewer as _run_reviewer_agent

    return await _run_reviewer_agent(
        code_edit=code_edit,
        sandbox_manager=sandbox_proxy or sandbox,
        episodic_store=store,
        task_id=task_id,
        trace_id=trace_id,
        repo_url=repo_url,
    )


# ── Build the hybrid graph ──────────────────────────────────────────


def build_hybrid_graph() -> StateGraph:  # type: ignore[type-arg]
    """Build the LangGraph state machine for the hybrid topology.

    Same as supervisor_only PLUS typed peer handoff between
    Coder ⇄ Reviewer:

    - When Reviewer returns ``accept``: routes to QA (same as
      supervisor_only)
    - When Reviewer returns ``reject_with_changes``: routes to
      ``run_peer_coder`` (peer handoff, Coder span parent = Reviewer)
    - ``run_peer_coder`` always routes back to ``run_reviewer_hybrid``
      (completing the Coder ⇄ Reviewer loop)

    Graph topology::

        START → run_supervisor → index_repo → run_planner →
        run_coder → run_reviewer → route_after_review_hybrid
        → (accept: run_qa → route_after_qa_hybrid →
            (all pass: run_supervisor_finalize →
              hitl_pre_commit_push (optional) →
              hitl_pre_pr → run_commit_and_push → run_open_pr → END) |
            (failures: halt_test_failure → END)) |
          (reject_with_changes: run_peer_coder →
            run_reviewer_hybrid → route_after_review_hybrid → ... loop) |
          (reject: run_supervisor_finalize → END) |
          (retry exhausted: halt_retry_exhausted → END)

    VAL-TOPOLOGY-004: On reject_with_changes, the Coder span's parent
    is the Reviewer span, not the Supervisor span.

    VAL-TOPOLOGY-005: Multiple reject_with_changes cycles create
    a chain of peer-handoff edges in the trace.

    VAL-TOPOLOGY-006: topology flag validated at API level.
    VAL-TOPOLOGY-007: tasks.topology persisted matches request.
    """
    from src.orchestrator.hitl import is_before_write_ops

    graph = StateGraph(OrchestratorState)

    # ── Add agent nodes (same as supervisor_only) ─────────────
    graph.add_node("run_supervisor", run_supervisor)
    graph.add_node("index_repo", index_repo)
    graph.add_node("run_planner", run_planner)
    graph.add_node("run_coder", _run_coder_supervisor)  # First Coder (under Supervisor)
    graph.add_node("run_reviewer", run_reviewer_hybrid)  # Hybrid Reviewer
    graph.add_node("run_qa", run_qa)
    graph.add_node("run_supervisor_finalize", run_supervisor_finalize)
    graph.add_node("halt_cost_budget_exhausted", halt_cost_budget_exhausted)
    graph.add_node("halt_github_delivery_failed", halt_github_delivery_failed)
    graph.add_node("halt_task_failed", halt_task_failed)
    graph.add_node("halt_retry_exhausted", halt_retry_exhausted)
    graph.add_node("halt_test_failure", halt_test_failure)

    # ── Add hybrid-specific peer nodes ────────────────────────
    graph.add_node("run_peer_coder", run_peer_coder)  # VAL-TOPOLOGY-004

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

    # Supervisor → Index Repo → Planner
    graph.add_edge("run_supervisor", "index_repo")
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
        route_after_review_hybrid,
        {
            "halt_cost_budget_exhausted": "halt_cost_budget_exhausted",
            "halt_task_failed": "halt_task_failed",
            "hitl_guardrail_escalation": "hitl_guardrail_escalation",
            "hitl_loop_detected": "hitl_loop_detected",
            "hitl_uncertainty_escalation": "hitl_uncertainty_escalation",
            "run_qa": "run_qa",  # accept → QA
            "run_peer_coder": "run_peer_coder",
            # reject_with_changes → peer Coder (VAL-TOPOLOGY-004)
            "run_supervisor_finalize": "run_supervisor_finalize",  # reject
            "halt_retry_exhausted": "halt_retry_exhausted",  # budget exhausted
        },
    )

    # ── Peer Coder → conditional routing ────────────────────────
    # VAL-TOPOLOGY-005: Coder⇄Reviewer loop via peer handoff
    graph.add_conditional_edges(
        "run_peer_coder",
        route_after_peer_coder,
        {
            "halt_cost_budget_exhausted": "halt_cost_budget_exhausted",
            "halt_task_failed": "halt_task_failed",
            "hitl_guardrail_escalation": "hitl_guardrail_escalation",
            "hitl_loop_detected": "hitl_loop_detected",
            "hitl_uncertainty_escalation": "hitl_uncertainty_escalation",
            "run_reviewer": "run_reviewer",
        },
    )

    # QA → conditional routing (guardrail + loop + uncertainty + test results)
    graph.add_conditional_edges(
        "run_qa",
        route_after_qa_hybrid,
        {
            "halt_cost_budget_exhausted": "halt_cost_budget_exhausted",
            "halt_task_failed": "halt_task_failed",
            "hitl_guardrail_escalation": "hitl_guardrail_escalation",
            "hitl_loop_detected": "hitl_loop_detected",
            "hitl_uncertainty_escalation": "hitl_uncertainty_escalation",
            "run_supervisor_finalize": "run_supervisor_finalize",
            "halt_test_failure": "halt_test_failure",
        },
    )

    # ── HITL checkpoint path ──────────────────────────────────
    if is_before_write_ops():
        graph.add_edge("run_supervisor_finalize", "hitl_pre_commit_push")
        graph.add_edge("hitl_pre_commit_push", "hitl_pre_pr")
    else:
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

    # Halt nodes → END
    graph.add_edge("halt_cost_budget_exhausted", END)
    graph.add_edge("halt_github_delivery_failed", END)
    graph.add_edge("halt_task_failed", END)
    graph.add_edge("halt_retry_exhausted", END)
    graph.add_edge("halt_test_failure", END)

    # Guardrail escalation → END (after HITL decision resolved)
    graph.add_edge("hitl_guardrail_escalation", END)

    # Loop detection HITL → END (VAL-LOOP-DETECT-004)
    graph.add_edge("hitl_loop_detected", END)

    # Uncertainty escalation HITL → END (VAL-UNCERTAINTY-007)
    graph.add_edge("hitl_uncertainty_escalation", END)

    return graph


# ── First Coder node (under Supervisor) ──────────────────────────────


async def _run_coder_supervisor(state: OrchestratorState) -> dict[str, Any]:
    """Node: Run the first Coder agent under the Supervisor parent.

    This is the initial Coder invocation (before any peer handoff).
    Its span parent is the Supervisor span, same as in supervisor_only.

    Subsequent Coder invocations (after reject_with_changes) go
    through run_peer_coder, whose parent is the Reviewer span.
    """
    task_id = state.task_id
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    supervisor_span_id = getattr(state, "supervisor_span_id", "") or ""

    step_key = f"coder_{state.step_index}"
    current_retries = state.retry_counters.get(step_key, 0)

    logger.info(
        "Coder node (hybrid, under Supervisor) for task %s (step %d, retry %d)",
        task_id,
        state.step_index,
        current_retries,
    )

    # Create Coder span under Supervisor parent
    tracing = get_tracing_client()
    coder_span_id = tracing.create_span(
        trace_id=trace_id,
        parent_span_id=supervisor_span_id or None,
        name="coder",
        span_type=SpanType.SPAN,
        input_data={
            "change_plan": _trunc_json(state.change_plan) if state.change_plan else None,
        },
        metadata={
            "task_id": task_id,
            "agent_name": "coder",
            "retry": current_retries,
            "topology": "hybrid",
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
        metadata={"agent_name": "coder", "topology": "hybrid"},
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
            # Use guardrail-wrapped sandbox proxy (VAL-GUARDRAIL-011)
            sandbox_proxy = get_sandbox_proxy(task_id)
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

    return {
        "code_edit": edit,
        "retry_counters": new_retry_counters,
        "total_cost_usd": new_total_cost,
        "total_tokens_in": state.total_tokens_in + tokens_in,
        "total_tokens_out": state.total_tokens_out + tokens_out,
        "total_tokens_cached": state.total_tokens_cached + cached_tokens,
        **_accumulate_agent_cost(
            agent_name="coder",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            state=state,
        ),
    }
