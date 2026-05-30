"""HITL escalation graph nodes for the supervisor_only / hybrid topologies.

Holds the human-in-the-loop escalation nodes — guardrail-violation
handling, loop-detected escalation, and uncertainty escalation — that
raise LangGraph ``interrupt()`` to pause the graph for human review.
Each node updates the task status, broadcasts a ``hitl_interrupt``
dashboard event, fires ``interrupt()``, and resolves the task based on
the human's approve/reject decision.  These symbols are re-exported from
``src.orchestrator.supervisor_only`` so existing imports and test patch
paths keep working.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

from langgraph.types import interrupt

from src.guardrails.errors import GuardrailViolation
from src.orchestrator import OrchestratorState
from src.orchestrator.registries import get_store
from src.tracing.langfuse import SpanType
from src.tracing.ws_broadcaster import TraceEvent, get_trace_broadcaster

logger = logging.getLogger(__name__)


async def _handle_guardrail_violation(
    *,
    violation: GuardrailViolation,
    task_id: str,
    trace_id: str,
    span_id: str | None,
    parent_span_id: str | None,
) -> dict[str, Any]:
    """Handle a GuardrailViolation from an agent tool call.

    VAL-GUARDRAIL-007: Emits a Langfuse span tagged guardrail.violation.
    VAL-GUARDRAIL-008: Writes an outcomes row with outcome=guardrail_block.
    VAL-GUARDRAIL-009: Halts the agent and escalates to HITL.

    Returns the state update dict that the calling node should return.
    The caller must then trigger interrupt() for HITL escalation.
    """
    from src.guardrails.middleware import report_guardrail_violation

    await report_guardrail_violation(
        violation=violation,
        task_id=task_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
    )

    logger.warning(
        "Guardrail violation in task %s: rule=%s, tool=%s",
        task_id,
        violation.rule_name,
        violation.tool_name,
    )

    # Return state update for the node — the graph's HITL interrupt
    # node will pick this up and trigger interrupt()
    return {
        "errors": [f"Guardrail violation: {violation.rule_name} — {violation.detail}"],
        "outcome": "guardrail_block",
        "status": "awaiting_hitl",
    }


# ── Guardrail escalation HITL node (VAL-GUARDRAIL-009) ──────────────


async def hitl_guardrail_escalation(state: OrchestratorState) -> dict[str, Any]:
    """Node: HITL interrupt for guardrail violation escalation.

    VAL-GUARDRAIL-009: When a guardrail rule fires, the offending
    agent is halted and the orchestrator raises a LangGraph interrupt()
    that surfaces the violation to HITL with the rule name and a
    human-readable explanation.

    The violating agent node catches the GuardrailViolation, writes
    the Langfuse span and outcomes row, then routes here.  This node
    fires interrupt() with the violation details, pausing the graph
    until a human reviews and resolves the escalation.
    """
    task_id = state.task_id
    logger.warning(
        "Guardrail escalation HITL interrupt for task %s, outcome=%s",
        task_id,
        state.outcome,
    )

    # Build violation details from the error messages stored in state
    error_detail = "; ".join(state.errors) if state.errors else "Unknown guardrail violation"

    # Update task status to awaiting_hitl
    store = get_store(task_id)
    if store is not None:
        await store.update_task_status(UUID(task_id), "awaiting_hitl")

    # Broadcast HITL interrupt event for the dashboard
    broadcaster = get_trace_broadcaster()
    trace_id = getattr(state, "trace_id", "") or uuid4().hex
    await broadcaster.publish(
        TraceEvent(
            type="hitl_interrupt",
            task_id=task_id,
            trace_id=trace_id,
            span_id="",
            parent_span_id="",
            name="hitl_guardrail_escalation",
            span_type=SpanType.SPAN,
            metadata={
                "cause": "guardrail_block",
                "detail": error_detail,
                "outcome": state.outcome,
            },
        )
    )

    # Fire the interrupt — graph pauses here (VAL-GUARDRAIL-009)
    decision = interrupt({
        "reason": "guardrail_block",
        "task_id": task_id,
        "cause": "guardrail_block",
        "explanation": error_detail,
    })

    if decision == "reject" or decision is None:
        # Rejected or dismissed — end the task
        if store is not None:
            from src.memory.episodic.models import CreateOutcomeParams
            await store.finish_task(UUID(task_id), "rejected", hitl_decision="reject")
            await store.create_outcome(
                CreateOutcomeParams(
                    task_id=UUID(task_id),
                    outcome="hitl_rejected",
                    detail={"cause": "guardrail_block", "original_errors": state.errors},
                )
            )
        return {"outcome": "hitl_rejected", "status": "rejected"}

    # Approved — the human has acknowledged the guardrail violation
    # and chosen to proceed (rare, but possible for e.g. rm -rf
    # inside a temp build directory that happens to be outside cwd)
    if store is not None:
        await store.update_task_status(UUID(task_id), "running")

    return {"hitl_decision": "approve", "outcome": "", "status": "running"}


# ── Loop detection HITL escalation node (VAL-LOOP-DETECT-004) ────


async def hitl_loop_detected(state: OrchestratorState) -> dict[str, Any]:
    """Node: HITL interrupt for loop detection escalation.

    VAL-LOOP-DETECT-004: A ``loop_detected`` outcome triggers a
    LangGraph ``interrupt()`` and surfaces a HITL escalation event
    on the dashboard with cause ``loop_detected``.

    VAL-CROSS-015: User can abort (reject), producing terminal outcome.
    VAL-CROSS-016: User can provide guidance (approve with guidance).
    """
    task_id = state.task_id
    logger.warning(
        "Loop detection HITL interrupt for task %s", task_id,
    )

    # Update task status to awaiting_hitl
    store = get_store(task_id)
    if store is not None:
        await store.update_task_status(UUID(task_id), "awaiting_hitl")

    # Broadcast HITL interrupt event for the dashboard
    broadcaster = get_trace_broadcaster()
    trace_id = getattr(state, "trace_id", "") or uuid4().hex

    # Get loop detection context from errors
    error_detail = "; ".join(state.errors) if state.errors else "Loop detected"

    await broadcaster.publish(
        TraceEvent(
            type="hitl_interrupt",
            task_id=task_id,
            trace_id=trace_id,
            span_id="",
            parent_span_id="",
            name="hitl_loop_detected",
            span_type=SpanType.SPAN,
            metadata={
                "cause": "loop_detected",
                "detail": error_detail,
                "outcome": state.outcome,
            },
        )
    )

    # Fire the interrupt — graph pauses here
    decision = interrupt({
        "reason": "loop_detected",
        "task_id": task_id,
        "cause": "loop_detected",
        "explanation": error_detail,
    })

    if decision == "reject" or decision is None:
        # VAL-CROSS-015: Abort path — terminal outcome
        if store is not None:
            from src.memory.episodic.models import CreateOutcomeParams
            await store.finish_task(UUID(task_id), "rejected", hitl_decision="reject")
            await store.create_outcome(
                CreateOutcomeParams(
                    task_id=UUID(task_id),
                    outcome="loop_detected",
                    detail={"cause": "loop_detected", "aborted": True},
                )
            )
        return {"outcome": "loop_detected", "status": "failed"}

    # VAL-CROSS-016: Guidance path — resume with injected guidance
    if store is not None:
        await store.update_task_status(UUID(task_id), "running")

    return {"hitl_decision": "approve", "outcome": "", "status": "running"}


# ── Uncertainty escalation HITL node (VAL-UNCERTAINTY-007) ────────


async def hitl_uncertainty_escalation(state: OrchestratorState) -> dict[str, Any]:
    """Node: HITL interrupt for uncertainty escalation.

    VAL-UNCERTAINTY-007: Every ``uncertainty_escalation`` outcome
    corresponds to a LangGraph ``interrupt()`` span, surfacing on
    the dashboard HITL view.

    The user can abort (reject) or provide guidance (approve).
    """
    task_id = state.task_id
    logger.warning(
        "Uncertainty escalation HITL interrupt for task %s", task_id,
    )

    # Update task status to awaiting_hitl
    store = get_store(task_id)
    if store is not None:
        await store.update_task_status(UUID(task_id), "awaiting_hitl")

    # Broadcast HITL interrupt event for the dashboard
    broadcaster = get_trace_broadcaster()
    trace_id = getattr(state, "trace_id", "") or uuid4().hex

    error_detail = "; ".join(state.errors) if state.errors else "Uncertainty escalation"

    await broadcaster.publish(
        TraceEvent(
            type="hitl_interrupt",
            task_id=task_id,
            trace_id=trace_id,
            span_id="",
            parent_span_id="",
            name="hitl_uncertainty_escalation",
            span_type=SpanType.SPAN,
            metadata={
                "cause": "uncertainty_escalation",
                "detail": error_detail,
                "outcome": state.outcome,
            },
        )
    )

    # Fire the interrupt — graph pauses here
    decision = interrupt({
        "reason": "uncertainty_escalation",
        "task_id": task_id,
        "cause": "uncertainty_escalation",
        "explanation": error_detail,
    })

    if decision == "reject" or decision is None:
        # Abort — terminal outcome
        if store is not None:
            from src.memory.episodic.models import CreateOutcomeParams
            await store.finish_task(UUID(task_id), "rejected", hitl_decision="reject")
            await store.create_outcome(
                CreateOutcomeParams(
                    task_id=UUID(task_id),
                    outcome="uncertainty_escalation",
                    detail={"cause": "uncertainty_escalation", "aborted": True},
                )
            )
        return {"outcome": "uncertainty_escalation", "status": "failed"}

    # Guidance path — resume
    if store is not None:
        await store.update_task_status(UUID(task_id), "running")

    return {"hitl_decision": "approve", "outcome": "", "status": "running"}
