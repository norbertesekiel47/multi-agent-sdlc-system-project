"""Tool-call recording sandbox proxy for the LangGraph orchestrator.

Holds the per-task tool-call recording machinery used to capture every agent
tool call (run_command, write_file, read_file, apply_diff, run_tests) so the
orchestrator can replay them after each agent turn through loop detection and
uncertainty escalation.  The ``RecordingSandboxProxy`` wraps the
``GuardrailSandboxProxy`` and records each call into a per-task registry; the
orchestrator reads and clears those recordings via
``_process_tool_call_recordings`` once the agent node completes.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from src.guardrails.middleware import GuardrailSandboxProxy
from src.orchestrator import OrchestratorState
from src.orchestrator.registries import get_sandbox_proxy, get_store
from src.tracing.langfuse import SpanType, get_tracing_client

logger = logging.getLogger(__name__)


# ── Recording Sandbox Proxy ───────────────────────────────────────
# Wraps the GuardrailSandboxProxy to record tool calls for
# loop detection and uncertainty escalation tracking.
# After each agent turn, the orchestrator reads the recorded
# tool calls and processes them through _check_loop_detection
# and _record_tool_call_and_check.


class _ToolCallRecord:
    """A single recorded tool call with outcome."""

    __slots__ = ("tool_name", "args", "success", "error_msg")

    def __init__(
        self,
        tool_name: str,
        args: dict[str, Any],
        success: bool,
        error_msg: str = "",
    ) -> None:
        self.tool_name = tool_name
        self.args = args
        self.success = success
        self.error_msg = error_msg


# Per-task list of tool call recordings (cleared after each agent turn)
_tool_call_recordings: dict[str, list[_ToolCallRecord]] = {}


def _get_recordings(task_id: str) -> list[_ToolCallRecord]:
    """Get the tool call recordings for a task."""
    return _tool_call_recordings.get(task_id, [])


def _clear_recordings(task_id: str) -> None:
    """Clear the tool call recordings for a task."""
    _tool_call_recordings.pop(task_id, None)


class RecordingSandboxProxy:
    """Wraps a GuardrailSandboxProxy to record tool call outcomes.

    Agents receive this proxy as their ``sandbox_manager`` dep.
    Every tool call is recorded for loop detection and uncertainty
    escalation tracking.  The orchestrator reads the recordings
    after each agent turn and processes them.
    """

    def __init__(self, inner: GuardrailSandboxProxy, task_id: str) -> None:
        self._inner = inner
        self._task_id = task_id
        if task_id not in _tool_call_recordings:
            _tool_call_recordings[task_id] = []

    def _record(
        self,
        tool_name: str,
        args: dict[str, Any],
        success: bool,
        error_msg: str = "",
    ) -> None:
        _tool_call_recordings.setdefault(self._task_id, []).append(
            _ToolCallRecord(tool_name, args, success, error_msg)
        )

    # ── Tool surface (record + delegate) ─────────────────────────

    async def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """Execute a command and record the outcome."""
        try:
            result = await self._inner.run_command(command, cwd=cwd, timeout=timeout)
            self._record("run_command", {"command": command}, success=True)
            return result
        except Exception as exc:
            self._record(
                "run_command",
                {"command": command},
                success=False,
                error_msg=str(exc)[:200],
            )
            raise

    async def write_file(self, path: str, content: str) -> None:
        """Write a file and record the outcome."""
        try:
            await self._inner.write_file(path, content)
            self._record("write_file", {"path": path}, success=True)
        except Exception as exc:
            self._record("write_file", {"path": path}, success=False, error_msg=str(exc)[:200])
            raise

    async def read_file(self, path: str) -> str:
        """Read a file and record the outcome."""
        try:
            result = await self._inner.read_file(path)
            self._record("read_file", {"path": path}, success=True)
            return result
        except Exception as exc:
            self._record("read_file", {"path": path}, success=False, error_msg=str(exc)[:200])
            raise

    async def apply_diff(self, diff: str) -> None:
        """Apply a diff and record the outcome."""
        try:
            await self._inner.apply_diff(diff)
            self._record("apply_diff", {"diff_length": len(diff)}, success=True)
        except Exception as exc:
            self._record(
                "apply_diff",
                {"diff_length": len(diff)},
                success=False,
                error_msg=str(exc)[:200],
            )
            raise

    async def run_tests(self, test_command: str = "pytest") -> str:
        """Run tests and record the outcome."""
        try:
            result = await self._inner.run_tests(test_command)
            self._record("run_tests", {"test_command": test_command}, success=True)
            return result
        except Exception as exc:
            self._record(
                "run_tests",
                {"test_command": test_command},
                success=False,
                error_msg=str(exc)[:200],
            )
            raise

    # ── Direct delegation (no recording) ─────────────────────────

    @property
    def is_running(self) -> bool:
        return self._inner.is_running

    @property
    def container_id(self) -> str | None:
        return self._inner.container_id

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def workspace_dir(self) -> Any:
        return self._inner.workspace_dir

    async def setup(self) -> None:
        await self._inner.setup()

    async def teardown(self) -> None:
        await self._inner.teardown()

    @property
    def unwrapped(self) -> GuardrailSandboxProxy:
        """Return the underlying GuardrailSandboxProxy."""
        return self._inner


def get_recording_proxy(task_id: str) -> RecordingSandboxProxy | None:
    """Create a RecordingSandboxProxy wrapping the guardrail proxy.

    Returns None if no guardrail proxy exists for the task.
    """
    base_proxy = get_sandbox_proxy(task_id)
    if base_proxy is None:
        return None
    return RecordingSandboxProxy(base_proxy, task_id)


async def _process_tool_call_recordings(
    *,
    agent_name: str,
    step_index: int,
    state: OrchestratorState,
    trace_id: str,
    parent_span_id: str | None,
) -> dict[str, Any] | None:
    """Process tool call recordings from the last agent turn.

    Reads the recordings from the per-task registry, processes them
    through loop detection and uncertainty escalation, and returns
    a state update dict if escalation is detected.

    This is called after each agent node completes. It:
    1. Processes each recorded tool call through _check_loop_detection
    2. Records tool call outcomes for uncertainty escalation
    3. Returns an escalation state update if triggered, or None
    """
    from src.orchestrator.supervisor_only import (
        _check_loop_detection,
        _get_tool_call_history_update,
        _reconstruct_uncertainty,
        _sync_uncertainty_to_state,
    )

    recordings = _get_recordings(state.task_id)
    if not recordings:
        return None

    # Process each recording
    accumulated_update: dict[str, Any] = {}
    for rec in recordings:
        # Record tool call for uncertainty escalation (tool error rate)
        ue = _reconstruct_uncertainty(state)
        # Merge any accumulated updates into the state we use
        if accumulated_update.get("tool_error_windows"):
            state_data = state.model_dump()
            state_data.update(accumulated_update)
            # Reconstruct state with updated fields
            ue._tool_error_windows = {
                k: deque(v, maxlen=10)
                for k, v in state_data.get("tool_error_windows", {}).items()
            }

        trigger = ue.record_tool_call(agent_name, rec.success, step_index)
        ue_update = _sync_uncertainty_to_state(ue)
        accumulated_update.update(ue_update)

        # Check loop detection for this tool call
        loop_result = await _check_loop_detection(
            agent_name=agent_name,
            tool_name=rec.tool_name,
            args=rec.args,
            state=state,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        if loop_result is not None:
            # Loop detected — return immediately
            loop_result.update(ue_update)
            _clear_recordings(state.task_id)
            return loop_result

        # Update tool call history
        history_update = _get_tool_call_history_update(
            agent_name, rec.tool_name, rec.args, state,
        )
        accumulated_update["tool_call_history"] = history_update

        if trigger is not None:
            # Tool error rate exceeded → uncertainty escalation
            logger.warning(
                "Uncertainty escalation: %s after tool call %s by %s",
                trigger.trigger, rec.tool_name, agent_name,
            )
            # Report to Langfuse
            tracing = get_tracing_client()
            tracing.create_span(
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                name="failure_mode.uncertainty_escalation",
                span_type=SpanType.SPAN,
                input_data={"tool_name": rec.tool_name, "agent_name": agent_name},
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
                    ),
                )

            _clear_recordings(state.task_id)
            return {
                "errors": [
                    f"Uncertainty escalation: {trigger.trigger} for agent {agent_name}"
                ],
                "outcome": "uncertainty_escalation",
                "status": "awaiting_hitl",
                **accumulated_update,
            }

    # Clear recordings for the next agent turn
    _clear_recordings(state.task_id)

    # Return accumulated updates if any changed
    has_changes = any(
        accumulated_update.get(k) != getattr(state, k, None)
        for k in ("tool_call_history", "tool_error_windows", "uncertainty_fired")
    )
    return accumulated_update if has_changes else None
