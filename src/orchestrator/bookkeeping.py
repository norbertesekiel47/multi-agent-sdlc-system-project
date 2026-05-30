"""Failure-mode and cost bookkeeping for the supervisor orchestrator.

Holds the helpers that track failure-mode and cost signals on the
``OrchestratorState`` between LangGraph nodes: loop detection over the
sliding window of tool calls, uncertainty-escalation reconstruction and
sync, Pydantic validation failure/success tracking, per-agent cost
accumulation, and diff-rejection checks.  These were extracted from
``supervisor_only.py`` and are re-exported there so they remain
importable as ``supervisor_only.<name>``.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from src.orchestrator import OrchestratorState
from src.orchestrator.registries import get_store
from src.tracing.langfuse import SpanType, get_tracing_client
from src.tracing.ws_broadcaster import TraceEvent, get_trace_broadcaster

if TYPE_CHECKING:
    from src.failure_modes.uncertainty import UncertaintyEscalation

logger = logging.getLogger(__name__)


async def _check_loop_detection(
    *,
    agent_name: str,
    tool_name: str,
    args: dict[str, Any],
    state: OrchestratorState,
    trace_id: str,
    parent_span_id: str | None,
) -> dict[str, Any] | None:
    """Check if a tool call triggers loop detection.

    Records the tool call in the sliding window and checks for
    3 identical (tool_name, args_hash) in the last 5 calls.

    Returns a state update dict with outcome='loop_detected' if
    detected, or None if no loop.
    """
    from src.failure_modes.loop_detection import LoopDetector

    # Reconstruct the detector from state
    detector = LoopDetector(window_size=5, threshold=3)

    # Replay existing tool call history for this agent
    history = state.tool_call_history.get(agent_name, [])
    for rec in history:
        from src.failure_modes.loop_detection import ToolCallRecord
        detector._windows.setdefault(agent_name, deque(maxlen=5))
        detector._windows[agent_name].append(
            ToolCallRecord(
                tool_name=rec["tool_name"],
                args_hash=rec["args_hash"],
            )
        )

    # Record the new tool call
    record = detector.record(agent_name, tool_name, args)

    # Check for loop
    result = detector.check(agent_name)

    if result is not None:
        # Loop detected — report to Langfuse
        tracing = get_tracing_client()
        span_id = tracing.create_span(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name="failure_mode.loop_detected",
            span_type=SpanType.SPAN,
            input_data={"tool_name": tool_name, "agent_name": agent_name},
            output_data=result.to_outcome_data(),
            metadata={
                "failure_mode": "loop_detected",
                "agent_name": agent_name,
                "tool_name": tool_name,
                "count": result.count,
            },
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            level="ERROR",
        )

        # Broadcast
        broadcaster = get_trace_broadcaster()
        await broadcaster.publish(
            TraceEvent(
                type="failure_mode",
                task_id=state.task_id,
                trace_id=trace_id,
                span_id=span_id or "",
                parent_span_id=parent_span_id or "",
                name="failure_mode.loop_detected",
                span_type=SpanType.SPAN,
                metadata=result.to_outcome_data()["detail"],
            )
        )

        # Write outcome
        store = get_store(state.task_id)
        if store is not None:
            from src.memory.episodic.models import CreateOutcomeParams
            await store.create_outcome(
                CreateOutcomeParams(
                    task_id=UUID(state.task_id),
                    outcome="loop_detected",
                    detail=result.to_outcome_data()["detail"],
                )
            )

        # Update tool call history in state
        new_history = dict(state.tool_call_history)
        agent_history = list(history)
        agent_history.append({"tool_name": tool_name, "args_hash": record.args_hash})
        # Trim to last 5
        if len(agent_history) > 5:
            agent_history = agent_history[-5:]
        new_history[agent_name] = agent_history

        return {
            "errors": [f"Loop detected: {agent_name} called {tool_name} {result.count} times"],
            "outcome": "loop_detected",
            "status": "awaiting_hitl",
            "tool_call_history": new_history,
        }

    # No loop — update tool call history and return None
    new_history = dict(state.tool_call_history)
    agent_history = list(history)
    agent_history.append({"tool_name": tool_name, "args_hash": record.args_hash})
    # Trim to last 5
    if len(agent_history) > 5:
        agent_history = agent_history[-5:]
    new_history[agent_name] = agent_history

    # Return just the history update (no outcome change)
    # The caller should merge this into state
    return None


def _get_tool_call_history_update(
    agent_name: str,
    tool_name: str,
    args: dict[str, Any],
    state: OrchestratorState,
) -> dict[str, list[dict[str, str]]]:
    """Get the updated tool_call_history after recording a tool call.

    This is a lightweight version that doesn't check for loops —
    used when the loop detection check is done separately.
    """
    from src.failure_modes.loop_detection import compute_args_hash

    args_hash = compute_args_hash(args)
    new_history = dict(state.tool_call_history)
    agent_history = list(new_history.get(agent_name, []))
    agent_history.append({"tool_name": tool_name, "args_hash": args_hash})
    # Keep only last 5 entries
    if len(agent_history) > 5:
        agent_history = agent_history[-5:]
    new_history[agent_name] = agent_history
    return new_history


# ── Uncertainty Escalation reconstruction from state ──────────────


def _reconstruct_uncertainty(state: OrchestratorState) -> UncertaintyEscalation:
    """Reconstruct an UncertaintyEscalation instance from state fields.

    LangGraph serializes state between nodes, so we can't store
    live objects.  Instead, we reconstruct from the state fields
    each time we need to check/record.
    """
    from src.failure_modes.uncertainty import UncertaintyEscalation

    ue = UncertaintyEscalation()
    # Restore internal state from OrchestratorState fields
    ue._pydantic_fail_counters = dict(state.pydantic_fail_counters)
    ue._test_failure_count = state.test_failure_count
    ue._rejected_diff_hashes = dict(state.rejected_diff_hashes)
    ue._tool_error_windows = {
        k: deque(v, maxlen=10)
        for k, v in state.tool_error_windows.items()
    }
    ue._trigger_fired = state.uncertainty_fired
    return ue


def _sync_uncertainty_to_state(
    ue: UncertaintyEscalation,
) -> dict[str, Any]:
    """Sync the UncertaintyEscalation state back to a state update dict.

    Returns only the fields that need updating.
    """
    return {
        "pydantic_fail_counters": dict(ue._pydantic_fail_counters),
        "test_failure_count": ue._test_failure_count,
        "rejected_diff_hashes": dict(ue._rejected_diff_hashes),
        "tool_error_windows": {
            k: list(v) for k, v in ue._tool_error_windows.items()
        },
        "uncertainty_fired": ue._trigger_fired,
    }


# ── Record a tool call outcome and check for escalation ──────────


async def _record_tool_call_and_check(
    *,
    agent_name: str,
    tool_name: str,
    args: dict[str, Any],
    success: bool,
    step_index: int,
    state: OrchestratorState,
    trace_id: str,
    parent_span_id: str | None,
) -> dict[str, Any] | None:
    """Record a tool call outcome and check for loop detection + uncertainty.

    This is called after every tool call dispatched from an agent node.
    It:
    1. Records the tool call in the loop detection sliding window
    2. Checks if loop detection triggers
    3. Records the tool call success/failure for uncertainty escalation
    4. Checks if uncertainty escalation triggers

    Returns a state update dict if escalation is detected, or None.
    """
    # ── Loop detection ────────────────────────────────────────
    loop_result = await _check_loop_detection(
        agent_name=agent_name,
        tool_name=tool_name,
        args=args,
        state=state,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
    )
    if loop_result is not None:
        # Loop detected — merge uncertainty state update into the loop result
        ue = _reconstruct_uncertainty(state)
        trigger = ue.record_tool_call(agent_name, success, step_index)
        ue_update = _sync_uncertainty_to_state(ue)
        loop_result.update(ue_update)
        return loop_result

    # ── Uncertainty escalation (tool error rate) ──────────────
    ue = _reconstruct_uncertainty(state)
    trigger = ue.record_tool_call(agent_name, success, step_index)
    ue_update = _sync_uncertainty_to_state(ue)

    if trigger is not None:
        # Tool error rate exceeded → uncertainty escalation
        logger.warning(
            "Uncertainty escalation: %s for agent=%s after tool call %s",
            trigger.trigger, agent_name, tool_name,
        )
        # Report to Langfuse
        tracing = get_tracing_client()
        tracing.create_span(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name="failure_mode.uncertainty_escalation",
            span_type=SpanType.SPAN,
            input_data={"tool_name": tool_name, "agent_name": agent_name},
            output_data=trigger.to_outcome_data(),
            metadata={
                "failure_mode": "uncertainty_escalation",
                "trigger": trigger.trigger,
                "agent_name": agent_name,
            },
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            level="ERROR",
        )

        # Write outcome
        store = get_store(state.task_id)
        if store is not None:
            from src.memory.episodic.models import CreateOutcomeParams
            await store.create_outcome(
                CreateOutcomeParams(
                    task_id=UUID(state.task_id),
                    outcome="uncertainty_escalation",
                    detail=trigger.to_outcome_data()["detail"],
                )
            )

        return {
            "errors": [
                f"Uncertainty escalation: {trigger.trigger} for agent {agent_name}"
            ],
            "outcome": "uncertainty_escalation",
            "status": "awaiting_hitl",
            **ue_update,
        }

    # No escalation — return the updated tool call history and
    # uncertainty state (the caller should merge these)
    new_history = _get_tool_call_history_update(agent_name, tool_name, args, state)
    return {
        "tool_call_history": new_history,
        **ue_update,
    }


# ── Record Pydantic validation failure and check escalation ──────


def _record_pydantic_failure(
    *,
    agent_name: str,
    step_index: int,
    state: OrchestratorState,
) -> dict[str, Any] | None:
    """Record a Pydantic validation failure and check for escalation.

    Returns a state update dict if escalation is detected, or None.
    Called when an agent's output fails Pydantic validation.
    """
    ue = _reconstruct_uncertainty(state)
    trigger = ue.record_pydantic_failure(agent_name, step_index)
    ue_update = _sync_uncertainty_to_state(ue)

    if trigger is not None:
        logger.warning(
            "Uncertainty escalation: pydantic_validation_3x for %s:%d",
            agent_name, step_index,
        )
        return {
            "errors": [
                f"Uncertainty escalation: pydantic_validation_3x for "
                f"{agent_name} at step {step_index}"
            ],
            "outcome": "uncertainty_escalation",
            "status": "awaiting_hitl",
            **ue_update,
        }

    # No escalation yet — just sync the counter back
    if ue_update.get("pydantic_fail_counters") == state.pydantic_fail_counters:
        return None
    return ue_update


def _record_pydantic_success(
    *,
    agent_name: str,
    step_index: int,
    state: OrchestratorState,
) -> dict[str, Any]:
    """Record a successful Pydantic validation (resets counter).

    Returns the state update dict with reset counter.
    """
    ue = _reconstruct_uncertainty(state)
    ue.record_pydantic_success(agent_name, step_index)
    return _sync_uncertainty_to_state(ue)


def _accumulate_agent_cost(
    *,
    agent_name: str,
    tokens_in: int,
    tokens_out: int,
    cached_tokens: int,
    cost_usd: Decimal,
    state: OrchestratorState,
) -> dict[str, dict[str, dict[str, int | str]]]:
    """Return an ``agent_costs`` state update that accumulates per-agent tokens and cost.

    The ``agent_costs`` dict maps agent_name → {
        "tokens_in": int, "tokens_out": int, "cached_tokens": int,
        "cost_usd": str  (Decimal serialized as string for JSONB)
    }.
    """
    current = state.agent_costs.copy() if state.agent_costs else {}
    existing = current.get(agent_name, {})
    new_tokens_in = int(existing.get("tokens_in", 0)) + tokens_in
    new_tokens_out = int(existing.get("tokens_out", 0)) + tokens_out
    new_cached_tokens = int(existing.get("cached_tokens", 0)) + cached_tokens
    prev_cost = Decimal(str(existing.get("cost_usd", "0")))
    new_cost = prev_cost + cost_usd
    current[agent_name] = {
        "tokens_in": new_tokens_in,
        "tokens_out": new_tokens_out,
        "cached_tokens": new_cached_tokens,
        "cost_usd": str(new_cost),
    }
    return {"agent_costs": current}


# ── Record diff rejection and check escalation ────────────────────


def _record_diff_rejection_check(
    *,
    diff_hash: str,
    agent_name: str,
    step_index: int,
    state: OrchestratorState,
) -> dict[str, Any] | None:
    """Record a diff rejection and check for same_fix_rejected_twice.

    Called in route_after_review when verdict is reject_with_changes.
    Returns a state update dict if escalation is detected, or None.
    """
    if not diff_hash:
        return None

    ue = _reconstruct_uncertainty(state)
    trigger = ue.record_diff_rejection(diff_hash, agent_name, step_index)
    ue_update = _sync_uncertainty_to_state(ue)

    if trigger is not None:
        logger.warning(
            "Uncertainty escalation: same_fix_rejected_twice for hash=%s…",
            diff_hash[:12],
        )
        return {
            "errors": [
                f"Uncertainty escalation: same_fix_rejected_twice "
                f"for diff_hash {diff_hash[:12]}…"
            ],
            "outcome": "uncertainty_escalation",
            "status": "awaiting_hitl",
            **ue_update,
        }

    # No escalation — just sync the rejected hashes back
    if ue_update.get("rejected_diff_hashes") == state.rejected_diff_hashes:
        return None
    return ue_update
